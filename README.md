# R E V I V E

Update: 1st place award at HackPrinceton 2026 for the Hardware x AI track!

**Distributed LLM inference across consumer devices. Phones, tablets, Raspberry Pis, and laptops form a swarm that thinks together.**

REVIVE turns everyday devices into a collective AI system. No cloud. No GPUs. Just the devices in the room.

The repo implements **two complementary strategies** for edge LLM inference, plus a shared build pipeline that feeds them. The full system design, contracts between modules, and deployment flow are in [ARCHITECTURE.md](ARCHITECTURE.md).

**Strategy 1 — Mixture-of-Agents (MoA)** &nbsp;(`revive/`, `android/`, `rpi/`, `macos/`)
Each device runs a *complete* small model with a specialized role — Reasoner, Writer, Critic, Factchecker, etc. A coordinator fans queries out in parallel; an aggregator synthesizes the best answer. Only text (kilobytes) crosses the network, so consumer WiFi jitter is tolerated and any device can drop out without breaking the swarm.

**Strategy 2 — True Pipeline-Parallel Distribution** &nbsp;(`true-distribution/`)
One Qwen3 model with its transformer layers *physically split* across multiple devices. Hidden states travel over HTTP; each worker owns only a slice of the weights. A **$10 Arduino Uno** runs the cluster's management plane (BMC) — it owns the authoritative partition table and refuses to let the coordinator route through a node it has declared dead. Verified token-for-token correct against the reference HuggingFace model. Details in [`true-distribution/README.md`](true-distribution/README.md).

**Build pipeline** &nbsp;(`LLM/`)
Offline pipeline that takes Qwen3 bases and produces role-specialized, device-tiered GGUFs: expanded data generation (Haiku + local Qwen3-4B distillation), QLoRA fine-tuning, ShortGPT layer pruning, imatrix-calibrated K-quant export at Q2_K / Q3_K_S / Q4_K_M / Q5_K_M. Emits a full role × device-tier matrix with a SHA-256 `manifest.json`. Currently feeds Strategy 1; Strategy 2 consumes fp16 weights directly from HuggingFace.

Built at HackPrinceton Spring 2026.

---

## Table of Contents

- [Project Status](#project-status)
- [How It Works](#how-it-works)
- [Architecture](#architecture)
- [Supported Platforms](#supported-platforms)
- [Models](#models)
- [Quick Start](#quick-start)
  - [iPhone / iPad](#iphone--ipad)
  - [Android](#android)
  - [Raspberry Pi](#raspberry-pi)
  - [MacBook](#macbook)
- [The Agent System](#the-agent-system)
- [Query Routing](#query-routing)
- [MoA Aggregation](#moa-aggregation)
- [Web Dashboard](#web-dashboard)
- [API Reference](#api-reference)
- [Training & Fine-tuning](#training--fine-tuning)
  - [LLM Pipeline (recommended)](#llm-pipeline-recommended)
  - [Training on AWS](#training-on-aws)
  - [Legacy Pipeline](#legacy-pipeline)
- [Protocol Specification](#protocol-specification)
- [Project Structure](#project-structure)
- [Performance](#performance)
- [FAQ](#faq)

---

## Project Status

**MoA Runtime — production:**
- iOS/iPadOS app with Worker and Coordinator modes, in-process llama.cpp via Metal (`revive/`)
- Android app with native JNI llama.cpp, foreground service, Android NSD discovery (`android/`)
- Raspberry Pi mesh aggregator with systemd auto-start (`rpi/`)
- MacBook CLI (Metal on Apple Silicon, CPU on Intel) with Worker + interactive Coordinator modes (`macos/`)
- Two web dashboards: PWA bundled with the iOS coordinator (`revive/Coordinator/WebDashboard/`) and a standalone mDNS aggregator UI (`dashboard/`)
- mDNS auto-discovery and OpenAI-compatible `POST /v1/chat/completions` on every platform

**True Pipeline-Parallel Distribution — working end-to-end** (`true-distribution/`):
- Layer-sliced Qwen3 loading with per-slice KV cache
- N-stage HTTP ring with a custom binary wire protocol (fp16 hidden states, ~2 KB/token)
- Heterogeneous layer partitioner (PipeEdge-DP + greedy fallback) that minimizes the slowest pipeline stage
- Arduino Uno BMC firmware (`arduino/revive_bmc.ino`) plus a Python simulator (`bmc_sim.py`) that speaks the identical line protocol over TCP
- Multi-BMC HA controller with leader election and replica failover
- Live chaos-engineering dashboard: kill/heal buttons, BMC event log, per-stage latency bars, token streaming
- Benchmark CLI reporting tok/s, prefill vs decode, per-stage latency, wire bandwidth
- 11 self-contained tests (correctness verified token-for-token against single-model HuggingFace reference under greedy sampling)

**LLM Build Pipeline — functional end-to-end** (`LLM/`):
- Expanded dataset generation: Claude Haiku + local Qwen3-4B teacher distillation → ~1500 examples/role (`LLM/data/`)
- Per-role QLoRA fine-tuning for all 8 roles on Qwen3-0.6B and Qwen3-1.7B bases (`LLM/train/`)
- Role-aware ShortGPT block-importance layer pruning (`LLM/prune/`)
- imatrix-calibrated K-quant export at Q2_K / Q3_K_S / Q4_K_M / Q5_K_M (`LLM/quantize/`)
- Role × device-tier matrix export producing up to 32 GGUFs with a SHA-256 `manifest.json` (`LLM/export/`)
- Per-role evaluation harness against a Qwen3-4B teacher (`LLM/eval/`)
- AWS EC2 spot-instance launcher that trains end-to-end and uploads to S3 (`LLM/scripts/aws_*.sh`)

**Research & probes:**
- `llamacpp-stages/tests/probe_embd.py` — gate test determining whether `llama-cpp-python`'s `batch.embd` input path bypasses only the embedding lookup (needed for a future llama.cpp-backed pipeline-parallel path).

**Documentation:**
- [ARCHITECTURE.md](ARCHITECTURE.md) — system-level design of MoA + LLM build pipeline, module contracts, deployment flow, glossary
- [PROTOCOL.md](PROTOCOL.md) — mDNS service type, TXT record fields, HTTP API contract, default ports
- [LLM/README.md](LLM/README.md) — build-pipeline internals and quick start
- [true-distribution/README.md](true-distribution/README.md) — pipeline-parallel architecture, BMC protocol, chaos-demo walkthrough, benchmark numbers

**On the roadmap:**
- MoA ([ARCHITECTURE.md §6](ARCHITECTURE.md#6-future-work)): on-device tier auto-selection from `manifest.json`; heterogeneous prompt formats; swarm-level speculative decoding; modular LoRA adapter distribution.
- True Distribution ([true-distribution/README.md §Status](true-distribution/README.md#status)): pipeline overlap for long-prompt prefill (Jupiter intra-sequence PP); iOS port (Swift worker wrapping a forked llama.cpp with a `llama_decode_layers(start, end)` API); GGUF Q4_K_M quantization for 4× memory + bandwidth savings.

---

## How It Works

*This section describes Strategy 1 (MoA). The pipeline-parallel flow is documented separately in [`true-distribution/README.md`](true-distribution/README.md).*

```
You type a question
       │
       ▼
┌─────────────────────┐
│  Coordinator (iPad   │ ◄── Classifies query type
│  or Raspberry Pi)    │     (simple fact? code? creative?)
└────────┬────────────┘
         │ Fan-out to matching roles
    ┌────┼────┬────┬────┐
    ▼    ▼    ▼    ▼    ▼
  📱    📱    📱   📱   💻
 iPhone  iPhone Pixel  Pi  MacBook
 Reasoner Writer Critic Fact Drafter
    │    │    │    │    │
    └────┴────┴────┴────┘
         │ Collect responses
         ▼
┌─────────────────────┐
│  Aggregator Model    │ ◄── Synthesizes best answer
│  (runs on coordinator)│     from all agents
└─────────────────────┘
         │
         ▼
   Final answer
```

1. **Discovery**: Devices find each other automatically via Bonjour/mDNS on the local WiFi network. Zero configuration.
2. **Classification**: A fast "Spotter" agent classifies the query (complex reasoning, code, math, creative, etc.) and determines which agent roles are needed.
3. **Fan-out**: The coordinator sends the query to all relevant workers in parallel, each running with its specialized system prompt.
4. **Inference**: Each device runs a complete GGUF model locally via llama.cpp. iPhones use Metal GPU acceleration, Android uses ARM NEON, MacBooks use Metal, Raspberry Pis use ARM NEON.
5. **Aggregation**: The coordinator's aggregator model synthesizes the best answer from all agent responses — preferring the Factchecker for accuracy, the Writer for clarity, and the Reasoner for logical structure.

The key insight: **many small models with different perspectives, running in parallel on separate devices, produce better answers than any single small model alone.** This is the Mixture-of-Agents pattern applied to edge devices.

---

## Architecture

The sections below describe **Strategy 1 (MoA)**. For the pipeline-parallel internals — layer partitioning, hidden-state wire format, Arduino BMC control plane, chaos-engineering demo — see [`true-distribution/README.md`](true-distribution/README.md).

### MoA vs. Pipeline-Parallel: when to use each

The two strategies answer different questions. Both are implemented in this repo.

| | **MoA** (`revive/`, `android/`, `rpi/`, `macos/`) | **Pipeline-parallel** (`true-distribution/`) |
|--|----|----|
| What's distributed | Different *models* on different devices | Different *layers* of one model on different devices |
| Wire payload | Text (kilobytes per query) | fp16 hidden states (~2 KB per token, per hop) |
| Failure mode | Graceful degrade — missing role omitted from synthesis | Hard stop — BMC refuses to route through a dead node |
| WiFi sensitivity | Insensitive to jitter (text is small, work is parallel) | Sensitive to per-hop latency (serial dependency between stages) |
| Best for | Many moderate devices answering in parallel with different perspectives | Running *one bigger model* than any single device could host |
| Ensemble method | Text-level MoA synthesis | None — a single coherent forward pass |

The MoA choice below is optimized for consumer WiFi's 5–50ms latency with unpredictable jitter: each device runs a **complete** small model independently, only the text outputs (kilobytes) travel over the network, and if a phone overheats or drops off the swarm keeps working. `true-distribution/` makes the opposite trade-off and proves it works — the measured bandwidth on a 3-worker Qwen3-0.6B ring is 58 KB/s, 214× under 802.11n WiFi capacity. Pipeline parallelism is compute-bound, not network-bound, once hidden states are fp16.

### Components

| Component | Role | Runs On |
|-----------|------|---------|
| **Worker** | Loads a GGUF model, serves inference via HTTP, advertises via mDNS | Any device |
| **Coordinator** | Discovers workers, classifies queries, routes to agents, aggregates responses | iPad, Pi, or MacBook |
| **Aggregator Model** | Fine-tuned to synthesize multi-agent responses into a single best answer | Coordinator device |
| **Spotter** | Fast classification model that determines query type and needed roles | Any worker |
| **Web Dashboard** | Browser-based real-time visualization of the swarm | Served by coordinator |

### Network Stack

- **Discovery**: Bonjour/mDNS (`_revive._tcp` service type)
- **Transport**: HTTP/1.1 REST
- **API Format**: OpenAI-compatible `/v1/chat/completions`
- **Serialization**: JSON
- **Scope**: Local WiFi network (same subnet)

---

## Supported Platforms

| Platform | Build Method | Hardware Acceleration | Status |
|----------|-------------|----------------------|--------|
| **iPhone / iPad** | Xcode project (xcframework) | Metal GPU + Accelerate + NEON | Production |
| **Android** | Gradle + NDK (Kotlin/Compose) | ARM NEON (CPU), optionally Vulkan | Production |
| **Raspberry Pi** | Python + cmake (llama-server) | ARM NEON | Production |
| **MacBook** | Python + cmake (llama-server) | Metal GPU (Apple Silicon) | Production |

All four platforms speak the same protocol, discover each other via mDNS, and expose the same HTTP API. Any combination works together on the same WiFi network.

---

## Models

REVIVE uses the **Qwen3** model family, fine-tuned per role and quantized to GGUF. Qwen3 was chosen because:

- The 1.7B model matches previous-generation 3B quality
- Dual thinking/non-thinking mode maps naturally to role specialization
- GQA (grouped-query attention) halves KV cache memory — critical on phones
- Native ChatML format matches our prompt structure

### Device Tiers

The `LLM/` pipeline produces role-specialized models at four compression levels matched to device RAM:

| Tier | RAM | Quant | Pruning | Example devices |
|------|-----|-------|---------|----------------|
| `ewaste` | 1–3 GB | Q2_K | Aggressive (−40% layers) | iPhone 6s/7, Pi 3, 3GB Android |
| `budget` | 3–4 GB | Q3_K_S | Moderate (−25% layers) | iPhone 8/X, Pi 4 4GB, 4GB Android |
| `standard` | 4–6 GB | Q4_K_M | None | iPhone 12/13/14, Pi 5 8GB, 6GB Android |
| `modern` | 6 GB+ | Q5_K_M | None | iPhone 15 Pro, Pixel 8, MacBook M-series |

Pruning only applies to simple roles (Spotter: −40%, Drafter/Concise: −25%). Reasoning-heavy roles (Reasoner, Critic, Factchecker, Aggregator) are never pruned.

### Model Filename Convention

```
revive-{role}-qwen3-{size}-{tier}-{quant}.gguf

Examples:
  revive-reasoner-qwen3-1.7b-modern-Q5_K_M.gguf    ← iPhone 15 Pro
  revive-writer-qwen3-1.7b-standard-Q4_K_M.gguf    ← iPhone 13/14
  revive-spotter-qwen3-0.6b-budget-Q3_K_S.gguf     ← Pi 4 4GB
  revive-drafter-qwen3-0.6b-ewaste-Q2_K.gguf       ← iPhone 7
```

A `manifest.json` is emitted alongside all GGUFs listing role, tier, quant, size in bytes, and SHA-256 for every file.

Models are loaded from local device storage. There is no cloud dependency at inference time.

---

## Quick Start

### iPhone / iPad

**Prerequisites**: Xcode 15+, macOS, a clone of [llama.cpp](https://github.com/ggerganov/llama.cpp)

```bash
# 1. Clone the repo
git clone https://github.com/your-org/revive.git
cd revive

# 2. Build llama.xcframework (one-time, 15-20 min)
cd /path/to/llama.cpp
bash build-xcframework.sh

# 3. Generate Xcode project
brew install xcodegen  # if needed
xcodegen generate

# 4. Open in Xcode
open revive.xcodeproj
# Set your signing team, build and run on device
```

**On the device**:

1. Transfer a `.gguf` model file to the app's Documents folder (via Finder, Files app, or AirDrop)
2. Launch REVIVE
3. Choose **Worker** or **Coordinator** mode:
   - **Worker**: Select a role (Reasoner, Writer, etc.), pick the model file, tap Start
   - **Coordinator**: Auto-discovers workers on the same WiFi, loads aggregator model from Documents

**Worker mode** starts an HTTP server on port 50001, advertises via Bonjour, and shows live metrics (tok/s, thermal state, battery, request count). The device stays awake via a silent audio session.

**Coordinator mode** shows a SwiftUI dashboard with worker cards, a chat interface, and mode toggles (Swarm vs Speed). It also serves a web dashboard on port 8080.

### Android

**Prerequisites**: Android Studio, NDK, a clone of llama.cpp next to the project

```bash
# 1. Clone llama.cpp alongside the project
git clone --depth 1 https://github.com/ggerganov/llama.cpp

# 2. Open android/ in Android Studio
# 3. Build and run on device (arm64)
```

**On the device**:

1. Place a `.gguf` model in the app's external files directory (`Android/data/com.revive.worker/files/models/`)
2. Launch REVIVE
3. Select a model, pick a role, set the port, tap **START WORKER**
4. The app starts a foreground service (persists when backgrounded), advertises via Android NSD, and serves HTTP inference

The Android app uses JNI to call llama.cpp natively — no Termux or Python required.

**Termux fallback** (for devices where you can't build the app):

```bash
# In Termux
cd ~/revive/android
ROLE=drafter PORT=8080 bash setup.sh
bash ~/start_worker.sh
```

### Raspberry Pi

**Prerequisites**: Raspberry Pi 4 or 5, Raspberry Pi OS (64-bit), WiFi or Ethernet on the same LAN

```bash
# One-command setup
git clone https://github.com/your-org/revive.git
cd revive
bash rpi/setup.sh
```

This installs all dependencies, builds llama.cpp, downloads the appropriate model (auto-selected by available RAM), and creates a systemd service.

```bash
# Start the mesh aggregator
sudo systemctl start revive-aggregator

# View logs
journalctl -u revive-aggregator -f

# Access the web dashboard
# Open http://<pi-ip>:9090 in any browser
```

The Raspberry Pi runs as the **mesh aggregator** — the central hub of the swarm. It discovers all workers on the LAN via mDNS, routes queries to the right agents, and synthesizes responses using a local aggregator model. The systemd service auto-starts on boot and restarts on failure.

### MacBook

**Prerequisites**: macOS, Homebrew, Python 3

```bash
# One-command setup
cd revive
bash macos/setup.sh
```

This builds llama.cpp with Metal (auto-detected on Apple Silicon), downloads the appropriate model, and creates launch scripts.

```bash
# Worker mode (joins the swarm as a specific role)
bash macos/start-worker.sh reasoner

# Coordinator mode (interactive REPL)
bash macos/start-coordinator.sh
```

**Coordinator interactive mode**:

```
REVIVE Swarm CLI — type your query (Ctrl+C to exit)

> What causes the seasons on Earth?
[Querying swarm...]
[Reasoner] The seasons are caused by Earth's 23.5° axial tilt...
[Critic] A common misconception is that seasons are caused by distance from the Sun...
[Factchecker] Earth's orbital eccentricity is only 0.017, meaning distance varies by ~3.4%...

[SWARM] Earth's seasons are caused by its 23.5° axial tilt relative to its orbital plane...
(3 agents, 4200ms, COMPLEX_REASONING)

> status
  Workers: 4
    reasoner     | ios      | 192.168.1.20:50001 | idle
    writer       | ios      | 192.168.1.21:50001 | idle
    critic       | android  | 192.168.1.30:8080  | idle
    drafter      | macos    | 192.168.1.40:8080  | idle
```

The MacBook CLI auto-detects Apple Silicon and uses Metal GPU acceleration (99 GPU layers offloaded). On Intel Macs, it falls back to CPU.

---

## The Agent System

REVIVE defines 8 agent roles, each with a distinct personality and system prompt:

| Role | Purpose | System Prompt Summary | Recommended Model |
|------|---------|----------------------|-------------------|
| **Reasoner** | Step-by-step logical analysis | "Think step by step. Show your reasoning chain explicitly." | Qwen3-1.7B |
| **Writer** | Clear, engaging prose | "Write clear, well-structured, engaging responses." | Qwen3-1.7B |
| **Concise** | Maximum brevity | "Answer in as few words as possible while being complete." | Qwen3-0.6B |
| **Critic** | Devil's advocate | "Identify flaws, edge cases, counterarguments." | Qwen3-1.7B |
| **Factchecker** | Verification focus | "Focus on verifiable, accurate information. Flag uncertainty." | Qwen3-1.7B |
| **Drafter** | Fast first-pass | "Produce a fast first-pass answer. Speed over polish." | Qwen3-0.6B |
| **Spotter** | Query classification | "Classify the query into one category." | Qwen3-0.6B |
| **Aggregator** | MoA synthesis | "Synthesize the single best answer from multiple agents." | Fine-tuned Qwen3-1.7B |

Each worker runs exactly one role. The coordinator assigns queries to the appropriate subset of roles based on the query type.

### Worker Weight

Each worker has a dynamic **weight** score used for aggregation priority:

```
weight = min(tokens_per_second / 40, 1.0) × thermal_penalty
```

Thermal penalties:
- Nominal/Fair: 1.0x
- Serious: 0.5x
- Critical: 0.1x

Higher-weight workers' responses are given more consideration during aggregation.

---

## Query Routing

When a query arrives, the coordinator follows this flow:

### Step 1: Classification via Spotter

The Spotter agent (a fast 0.6B model) classifies the query into one of six types:

| Query Type | Example | Target Roles |
|-----------|---------|-------------|
| `SIMPLE_FACT` | "What is the capital of Australia?" | Reasoner, Concise |
| `COMPLEX_REASONING` | "Should AI be granted legal personhood?" | Reasoner, Writer, Critic, Factchecker, Drafter |
| `CREATIVE` | "Write a poem about distributed computing" | Writer, Reasoner, Critic |
| `CODE` | "Explain the CAP theorem" | Reasoner, Factchecker, Critic |
| `MATH` | "Explain the Monty Hall problem" | Reasoner, Factchecker, Concise |
| `OPINION` | "Is nuclear power necessary for climate change?" | Writer, Critic, Reasoner |

### Step 2: Parallel Fan-out

The coordinator queries all matching workers simultaneously using Swift's `TaskGroup` (iOS) or `asyncio.gather` (Pi/Mac). Timeouts are adjusted by complexity:

- Simple facts: 6 second timeout
- Complex queries: 12 second timeout
- Speed mode: 8 second timeout

### Step 3: Aggregation

If the query type needs aggregation and multiple agents responded, the aggregator model synthesizes a final answer. Otherwise, the best single response is returned directly.

### Speed Mode

Speed mode bypasses full MoA aggregation. It pairs a **Drafter** (fast speculation) with the highest-weight **Verifier** (usually a Reasoner). This trades answer quality for lower latency — useful for simple questions or when the swarm is small.

---

## MoA Aggregation

The aggregator uses a ChatML prompt to synthesize responses:

```
<|im_start|>system
You are the Aggregator of a distributed AI swarm. You receive multiple
responses from specialized agents and synthesize the single best answer
by taking the strongest elements from each, resolving contradictions
(prefer Factchecker and Critic over Drafter), using the Writer's clarity,
and the Reasoner's logic. Output only the final synthesized answer.<|im_end|>
<|im_start|>user
Original question: {query}

Agent responses:

[The Reasoner — qwen3-1.7b-q4_k_m]:
{reasoner_response}

---

[The Critic — qwen3-1.7b-q4_k_m]:
{critic_response}

---

[The Writer — qwen3-0.6b-q4_k_m]:
{writer_response}

Synthesize the single best answer:<|im_end|>
<|im_start|>assistant
```

The aggregator is fine-tuned specifically for this task using QLoRA on 2000+ synthetic examples (see [Training](#training--fine-tuning)).

---

## Web Dashboard

The coordinator serves a browser-based dashboard accessible from any device on the LAN.

**iOS coordinator**: `http://<ipad-ip>:8080`
**Raspberry Pi**: `http://<pi-ip>:9090`

### Features

- **Live swarm topology canvas** — coordinator at center, workers in orbit, animated particles during inference
- **Agent cards** — role, model, tok/s, status, device metrics, weight bar
- **Chat interface** — submit queries, see individual agent responses + synthesized answer
- **Mode toggle** — switch between Swarm and Speed mode
- **Impact calculator** — estimates power savings if N% of the world's 5.3B phones joined the swarm vs. traditional GPU clusters
- **PWA-enabled** — add to home screen on mobile devices
- **Auto-refresh** — polls swarm state every 1.5 seconds

### Visual Design

Dark cyberpunk theme (#0a0a0a background, #00ff88 primary green). Canvas renders topology with glowing nodes, pulsing rings during generation, and particle effects showing data flow from workers to coordinator. Each agent role has a distinct color that carries through the entire UI.

---

## API Reference

All REVIVE nodes expose the same base API. The aggregator adds swarm-specific endpoints.

### Worker Endpoints

#### `GET /health`

```json
{"status": "ok", "role": "reasoner", "platform": "ios", "uptime": 3600}
```

#### `POST /v1/chat/completions`

OpenAI-compatible inference endpoint.

**Request:**
```json
{
  "model": "qwen3-1.7b-q4_k_m",
  "messages": [
    {"role": "system", "content": "You are a rigorous analytical thinker."},
    {"role": "user", "content": "What causes the seasons?"}
  ],
  "max_tokens": 150,
  "temperature": 0.7,
  "stream": false
}
```

**Response:**
```json
{
  "choices": [{"message": {"role": "assistant", "content": "..."}}],
  "metrics": {
    "tokens_generated": 42,
    "tokens_per_second": 18.5,
    "time_to_first_token_ms": 120,
    "total_time_ms": 2300,
    "thermal_state": "nominal",
    "battery_percent": 85,
    "memory_used_mb": 512
  }
}
```

### Aggregator Endpoints (Coordinator Only)

#### `POST /v1/swarm/query`

Submit a query to the full swarm.

```json
{
  "query": "What are the implications of quantum computing on encryption?",
  "mode": "swarm",
  "timeout_seconds": 12
}
```

**Response:**
```json
{
  "query": "...",
  "query_type": "COMPLEX_REASONING",
  "agent_responses": [
    {"role": "reasoner", "model": "qwen3-1.7b", "content": "..."},
    {"role": "critic", "model": "qwen3-0.6b", "content": "..."}
  ],
  "final_answer": "Synthesized answer...",
  "mode": "swarm",
  "total_time_ms": 4500,
  "agents_responded": 3
}
```

#### `GET /v1/swarm/status`

Full swarm topology.

```json
{
  "aggregator": {"host": "raspberrypi", "platform": "rpi", "uptime": 3600},
  "workers": [
    {"name": "REVIVE-reasoner-iPhone", "role": "reasoner", "host": "192.168.1.20", "port": 50001, "status": "idle", "platform": "ios", "metrics": {...}}
  ],
  "total_workers": 4,
  "total_tps": 72.3
}
```

#### `POST /v1/swarm/register`

Manual worker registration for devices that can't do mDNS.

```json
{
  "host": "192.168.1.42",
  "port": 8080,
  "role": "reasoner",
  "model": "qwen3-1.7b-q4_k_m",
  "platform": "android",
  "ram_mb": 6144
}
```

### iOS Web Dashboard API (Port 8080)

#### `GET /api/state`

Returns full coordinator state including workers, chat messages, mode, and query status.

#### `POST /api/query`

Triggers a query from the web dashboard:
```json
{"query": "...", "mode": "swarm"}
```

#### `POST /api/mode`

Switch between swarm and speed mode:
```json
{"mode": "speed"}
```

---

## Training & Fine-tuning

REVIVE includes a complete fine-tuning pipeline that produces role-specialized, device-tiered GGUF models from Qwen3 base weights.

### LLM Pipeline (recommended)

The `LLM/` directory is the current pipeline. It produces a full tier matrix — up to 32 GGUFs covering every role × device tier combination.

```
Data generation
  ├── Claude Haiku  → 750 examples/role  (diverse queries, ~$2 total)
  └── Qwen3-4B teacher (local) → 750 examples/role (higher-signal distillation)
       ↓ merge_datasets.py → ~1500 examples/role
       ↓
QLoRA fine-tune (Unsloth, r=32, 3 epochs, A10G ~30min/role)
       ↓ merged 16-bit HF checkpoint
       ↓
Role-aware pruning (ShortGPT block importance)
  Spotter:         −40% layers
  Drafter/Concise: −25% layers
  Reasoner/Writer/Critic/Factchecker/Aggregator: no pruning
       ↓
imatrix-calibrated quantization per device tier
  ewaste  → Q2_K   │  budget   → Q3_K_S
  standard→ Q4_K_M │  modern   → Q5_K_M
       ↓
LLM/output/gguf/revive-{role}-qwen3-{size}-{tier}-{quant}.gguf
+ manifest.json
```

**Quick start:**

```bash
cd LLM
pip install -r requirements.txt

# Minimum viable ship: standard tier only (~6 hrs on g5.xlarge)
ANTHROPIC_API_KEY=sk-... bash scripts/bootstrap.sh

# Full tier matrix: all 32 GGUFs
bash scripts/compress_all.sh

# Validate all roles hit their quality bar
python3 -m LLM.eval.eval_role --all
```

**Training configuration:**

| | Aggregator | Large roles | Small roles |
|--|-----------|-------------|-------------|
| Base model | Qwen3-1.7B | Qwen3-1.7B | Qwen3-0.6B |
| LoRA rank | 32 | 32 | 32 |
| Sequence length | 4096 | 2048 | 1024 |
| Epochs | 3 | 3 | 3 |
| Learning rate | 2e-4 | 2e-4 | 2e-4 |

---

### Training on AWS

The recommended way to run training is on an EC2 g5.xlarge (A10G 24GB, ~$0.30/hr spot). Three scripts handle the full lifecycle:

**Prerequisites:** AWS CLI configured (`aws configure`), an AWS account with EC2 + S3 + IAM permissions.

```bash
export ANTHROPIC_API_KEY=sk-...
export S3_BUCKET=my-revive-models    # will be created if it doesn't exist
export AWS_REGION=us-east-1          # optional, defaults to us-east-1

# Preview config without launching
bash LLM/scripts/aws_launch.sh --dry-run

# Launch spot instance (~$1.50–6 total, ~6 hrs)
bash LLM/scripts/aws_launch.sh
```

The instance:
1. Installs all dependencies (Unsloth, llama.cpp with CUDA, Python packages)
2. Downloads the Qwen3-4B teacher GGUF from HuggingFace
3. Runs `bootstrap.sh` (data generation → training → standard tier export)
4. Uploads all GGUFs + `manifest.json` to `s3://{bucket}/revive-models/`
5. Self-terminates when done

```bash
# When training completes (~6 hrs), pull models locally
S3_BUCKET=my-revive-models bash LLM/scripts/aws_pull.sh
# → syncs to LLM/output/gguf/
```

Monitor progress:
```bash
aws ec2 get-console-output --instance-id <id> --region us-east-1
```

**Estimated cost:** $1.50–6 on spot (g5.xlarge ~$0.30/hr). On-demand fallback: ~$6 if spot capacity is unavailable.

---

### Legacy Pipeline

`training/` is the original single-tier pipeline (300 examples/role, Q4_K_M only). It stays as a reference and fallback. The `LLM/` pipeline imports its prompt banks without modifying it.

```bash
# Generate data + train all roles (single Q4_K_M tier)
ANTHROPIC_API_KEY=sk-... bash training/train_all.sh
```

---

## Protocol Specification

See [PROTOCOL.md](PROTOCOL.md) for the full cross-platform protocol specification, including:

- mDNS service type and TXT record fields
- HTTP API contract for all endpoints
- Port conventions per platform
- Model recommendations per device class

### mDNS Service Discovery

All REVIVE nodes advertise as `_revive._tcp` with these TXT record fields:

| Key | Description |
|-----|-------------|
| `role` | Agent role (reasoner, writer, etc.) |
| `model` | Model filename |
| `ram` | Device RAM in MB |
| `port` | HTTP server port |
| `platform` | ios, android, rpi, macos |
| `caps` | Hardware capabilities (metal, neon, vulkan, cpu) |

### Default Ports

| Platform | Port |
|----------|------|
| iOS worker | 50001 |
| iOS web dashboard | 8080 |
| Android worker | 8080 |
| Raspberry Pi aggregator | 9090 |
| MacBook worker/coordinator | 8080 |

---

## Project Structure

```
revive/
├── LLM/                             # Model build pipeline (offline, GPU workstation)
│   ├── train/
│   │   ├── train_qwen_role.py       # QLoRA fine-tune per role (Unsloth)
│   │   └── train_all.sh             # Train all 8 roles sequentially
│   ├── data/
│   │   ├── generate_expanded_dataset.py  # 750 examples/role via Claude Haiku
│   │   ├── distill_from_qwen4b.py        # 750 examples/role via local teacher
│   │   ├── merge_datasets.py             # Dedup + merge → ~1500/role
│   │   ├── build_calibration.py          # Seed imatrix calibration prompts
│   │   └── calibration_prompts/          # Per-role calibration text files
│   ├── prune/
│   │   ├── layer_prune.py           # ShortGPT block importance pruning
│   │   ├── heal_lora.py             # Optional LoRA heal after pruning
│   │   └── prune_profiles.yaml      # Per-role drop fractions
│   ├── quantize/
│   │   ├── imatrix_gen.py           # Role-specific importance matrix
│   │   └── quantize_tiers.py        # Q2_K/Q3_K_S/Q4_K_M/Q5_K_M export
│   ├── export/
│   │   ├── export_tier_matrix.py    # Orchestrates full role × tier matrix
│   │   ├── export_single.sh         # Single role/tier export
│   │   └── manifest.py              # manifest.json writer
│   ├── eval/
│   │   ├── eval_role.py             # Per-role quality metrics vs teacher
│   │   └── eval_tier.py             # Sweep quant/prune variants for a role
│   ├── common/
│   │   ├── role_registry.py         # Role → base model + seq_len mapping
│   │   ├── device_tiers.yaml        # Tier → quant + prune + RAM envelope
│   │   └── gguf_io.py               # llama.cpp subprocess helpers
│   ├── scripts/
│   │   ├── bootstrap.sh             # End-to-end: data → train → standard tier
│   │   ├── compress_all.sh          # Full tier matrix export
│   │   ├── quick_test.sh            # Smoke test binaries + spotter export
│   │   ├── aws_launch.sh            # Launch EC2 spot instance for training
│   │   ├── aws_user_data.sh         # EC2 boot script (installs, trains, uploads)
│   │   └── aws_pull.sh              # Sync S3 artifacts → local gguf/
│   └── requirements.txt
│
├── training/                        # Legacy single-tier pipeline (reference)
│   ├── generate_dataset.py          # Aggregator training data (Claude Haiku)
│   ├── generate_role_dataset.py     # Per-role training data
│   ├── train.py                     # QLoRA aggregator training
│   ├── train_role.py                # QLoRA per-role training
│   └── train_all.sh                 # Full legacy pipeline
│
├── revive/                          # iOS app (Swift/SwiftUI)
│   ├── reviveApp.swift              # Entry point, mode selection
│   ├── Info.plist                   # Bonjour permissions, background modes
│   ├── revive.entitlements          # Network access (sandbox disabled)
│   ├── Coordinator/
│   │   ├── CoordinatorView.swift    # Coordinator UI (swarm overview, chat, metrics)
│   │   ├── CoordinatorWebServer.swift  # HTTP server for web dashboard (port 8080)
│   │   ├── SwarmManager.swift       # Bonjour discovery, parallel fan-out
│   │   ├── QueryRouter.swift        # Query classification + routing
│   │   ├── Aggregator.swift         # MoA synthesis on local model
│   │   └── WebDashboard/            # Browser-accessible dashboard
│   │       ├── index.html           # PWA-enabled dashboard
│   │       ├── style.css            # Cyberpunk dark theme
│   │       └── dashboard.js         # Canvas topology, realtime polling, chat
│   ├── Worker/
│   │   ├── WorkerView.swift         # Worker setup + telemetry UI
│   │   └── InferenceHandler.swift   # HTTP → llama inference bridge
│   └── Shared/
│       ├── Models.swift             # AgentRole, QueryType, WorkerInfo, etc.
│       ├── BonjourService.swift     # mDNS advertisement + discovery
│       ├── HTTPServer.swift         # Minimal HTTP/1.1 server (Network.framework)
│       ├── LibLlama.swift           # Swift binding to llama.cpp
│       ├── Metrics.swift            # Device telemetry (thermal, battery, memory)
│       └── Extensions.swift         # Hex color parser
│
├── android/                         # Android app (Kotlin/Jetpack Compose)
│   ├── app/
│   │   ├── build.gradle.kts         # Gradle config with NDK + Compose
│   │   └── src/main/
│   │       ├── AndroidManifest.xml  # Permissions, foreground service
│   │       ├── cpp/
│   │       │   ├── CMakeLists.txt   # Builds llama.cpp via NDK
│   │       │   └── native-lib.cpp   # JNI bridge (tokenize, inference)
│   │       ├── java/com/revive/worker/
│   │       │   ├── MainActivity.kt  # Compose UI (model picker, role selector)
│   │       │   ├── LlamaJNI.kt      # JNI bindings + LlamaEngine wrapper
│   │       │   ├── HttpServer.kt    # Ktor server (/health, /v1/chat/completions)
│   │       │   ├── NsdAdvertiser.kt # Android NSD (native mDNS)
│   │       │   ├── WorkerService.kt # Foreground service for background inference
│   │       │   └── DeviceMetrics.kt # Battery, thermal, memory
│   │       └── res/values/themes.xml
│   ├── build.gradle.kts
│   ├── settings.gradle.kts
│   ├── setup.sh                     # Termux fallback setup
│   └── advertise.py                 # Termux mDNS advertiser
│
├── rpi/                             # Raspberry Pi mesh aggregator
│   ├── aggregator.py                # Main coordinator: discovery, routing, MoA, web UI
│   ├── discovery.py                 # SwarmDiscovery + ServiceAdvertiser (shared)
│   ├── worker.py                    # llama-server subprocess manager
│   ├── metrics.py                   # CPU temp, memory, thermal state
│   ├── setup.sh                     # One-command install + systemd service
│   └── requirements.txt             # Python dependencies
│
├── macos/                           # MacBook CLI
│   ├── revive-cli.py                # Worker or coordinator mode, Metal GPU
│   └── setup.sh                     # Build llama.cpp, download model
│
├── true-distribution/               # Pipeline-parallel runtime (one model, layers split)
│   ├── pipeline/
│   │   ├── worker.py                # Pipeline stage: layer-sliced forward + per-stage KV cache
│   │   ├── coordinator.py           # Ring driver: streams tokens, respects BMC's view
│   │   ├── partitioner.py           # Greedy + PipeEdge-DP layer assignment
│   │   ├── protocol.py              # Tensor wire format (coordinator ↔ worker)
│   │   ├── bmc_protocol.py          # Host ↔ Arduino line protocol (v2)
│   │   ├── bmc_sim.py               # Python BMC firmware simulator (TCP)
│   │   ├── controller.py            # Host-side BMC client (serial or TCP)
│   │   ├── multi_bmc.py             # Multi-BMC HA (leader election, replica failover)
│   │   └── net_utils.py
│   ├── arduino/
│   │   ├── revive_bmc.ino           # Production BMC firmware for Arduino Uno
│   │   └── README.md                # How to flash and verify
│   ├── dashboard/
│   │   ├── server.py                # Embeds BMC sim + controller + coordinator
│   │   └── index.html               # Topology, chaos buttons, per-stage latency, token stream
│   ├── scripts/
│   │   ├── launch_cluster.sh        # Full stack: workers + dashboard (SERIAL= for Arduino)
│   │   ├── launch_local.sh          # Spawn N workers for ad-hoc dev
│   │   ├── stop_local.sh            # Kill workers
│   │   ├── benchmark.py             # Throughput + per-stage latency + bandwidth report
│   │   └── demo_full.py             # CLI chaos-engineering demo
│   ├── tests/                       # 11 self-contained test runners (no pytest needed)
│   └── README.md                    # Pipeline-parallel architecture + BMC protocol
│
├── dashboard/                       # Standalone mDNS swarm dashboard (runs independently)
│   ├── server.py                    # aiohttp server with Bonjour discovery, SSE streaming
│   └── index.html                   # Live worker list + chat
│
├── llamacpp-stages/                 # Research: llama.cpp slicing probes
│   └── tests/probe_embd.py          # Gate test for batch.embd bypass path
│
├── ARCHITECTURE.md                  # System design (MoA + LLM build pipeline)
├── PROTOCOL.md                      # Cross-platform mDNS/HTTP protocol spec
├── project.yml                      # XcodeGen project config
└── setup.sh                         # iOS one-time setup (build xcframework)
```

---

## Performance

Measured with Qwen3-1.7B Q4_K_M, 150-token generation:

| Device | tok/s | 150-token time | Hardware |
|--------|-------|---------------|----------|
| MacBook M2 Pro | 60-80 | 2-3s | Metal GPU |
| iPhone 15 Pro | 25-35 | 4-6s | Metal GPU (A17 Pro) |
| iPhone 14 | 15-20 | 7-10s | Metal GPU (A15) |
| Pixel 8 | 8-15 | 10-18s | ARM NEON (CPU) |
| Raspberry Pi 5 (8GB) | 3-6 | 25-50s | ARM NEON |

With the MoA pattern and 4 workers, total swarm throughput is the **sum** of individual device speeds — the queries run in parallel. A swarm of 4 iPhones produces ~100 tok/s aggregate throughput.

### End-to-End Query Latency

| Query Type | Agents Used | Typical Latency |
|-----------|-------------|----------------|
| Simple fact | 2 (reasoner + concise) | 3-6s |
| Complex reasoning | 5 (all roles) | 6-12s |
| Code | 3 (reasoner + factchecker + critic) | 5-10s |
| Creative | 3 (writer + reasoner + critic) | 5-10s |

Latency is bounded by the **slowest responding agent within the timeout window**, not the sum. Faster devices finish first; the coordinator starts aggregation as soon as all responses arrive (or the timeout is reached).

### Power Efficiency

Each phone draws ~2-3W during inference. A swarm of 10 phones uses 20-30W total — compared to 300W+ for a single A100 GPU, while providing comparable throughput for small model inference.

---

## FAQ

**Q: Do all devices need to be on the same WiFi network?**
Yes. REVIVE uses Bonjour/mDNS for discovery, which works within a single broadcast domain (same WiFi network or LAN). Devices on different networks won't find each other. You can use manual registration (`POST /v1/swarm/register`) to add devices across networks if you handle routing.

**Q: What happens if a device drops out mid-query?**
The coordinator uses timeouts. If a worker doesn't respond within the timeout (6-12 seconds depending on query type), its response is simply omitted. The aggregator synthesizes from whatever responses arrived.

**Q: Can I use different models on different devices?**
Yes. Each worker loads its own GGUF model independently. The coordinator doesn't care which model a worker runs — it only cares about the role. A Reasoner running Qwen3-1.7B and a Reasoner running Llama-3.2-1B will both be queried.

**Q: Does the coordinator need a model?**
The coordinator loads an aggregator model for MoA synthesis. Without it, it falls back to returning the highest-weight worker's response directly. For best results, use the fine-tuned aggregator GGUF.

**Q: Can I use this without fine-tuning?**
Yes. The base Qwen3 models work well with the role-based system prompts. Fine-tuning improves aggregation quality by ~15-20% (based on our evaluations) but is not required to get started.

**Q: How many devices can join the swarm?**
There's no hard limit. Bonjour can discover hundreds of services. The coordinator queries workers in parallel using async task groups. Practically, 3-8 devices is the sweet spot — enough for diverse perspectives without excessive aggregation latency.

**Q: Can the Raspberry Pi be a worker instead of the aggregator?**
Yes. Set `REVIVE_PORT=8080` and change the role in the systemd service. But the Pi's low inference speed (3-6 tok/s) makes it more useful as a coordinator that delegates to faster phones.

**Q: Does this work offline?**
Yes, completely. All inference runs locally on each device. The only network dependency is device-to-device communication over local WiFi. No internet connection is needed after the initial model download.

---

## License

MIT
