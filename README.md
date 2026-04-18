# R E V I V E

**Distributed LLM inference across consumer devices. Phones, tablets, Raspberry Pis, and laptops form a swarm that thinks together.**

REVIVE turns everyday devices into a collective AI system. Each device loads a small language model and takes on a specialized role — Reasoner, Writer, Critic, Factchecker — while a coordinator fans out queries across the swarm and synthesizes responses using Mixture-of-Agents (MoA). No cloud. No GPUs. Just the devices in the room.

Built at HackPrinceton 2025.

---

## Table of Contents

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
- [Protocol Specification](#protocol-specification)
- [Project Structure](#project-structure)
- [Performance](#performance)
- [FAQ](#faq)

---

## How It Works

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

### Why MoA Instead of Model Sharding

Traditional distributed inference splits one model across devices (pipeline parallelism). This requires tight synchronization and sends megabytes of activation tensors between devices at every layer boundary. Consumer WiFi has 5-50ms latency with unpredictable jitter — pipeline parallelism falls apart.

REVIVE takes the opposite approach. Each device runs a **complete** small model independently. Only the text outputs (kilobytes) travel over the network. If a device drops out, the others still work. If a phone overheats, it gracefully degrades. The coordinator collects whatever responses arrive within the timeout and synthesizes from what it has.

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

REVIVE uses the **Qwen3** model family in GGUF format, quantized to Q4_K_M. Qwen3 was chosen because:

- The 1.7B model matches previous-generation 3B quality
- Dual thinking/non-thinking mode maps naturally to role specialization (fast classification vs. deep reasoning)
- GQA (grouped-query attention) halves KV cache memory — critical on phones
- Native ChatML format matches our prompt structure

### Recommended Models by Device

| Device | RAM | Model | Quant | Size | Role |
|--------|-----|-------|-------|------|------|
| iPhone 15 Pro | 8GB | Qwen3-1.7B | Q4_K_M | ~1.1GB | Any role |
| iPhone 13/14 | 4-6GB | Qwen3-0.6B | Q4_K_M | ~0.4GB | Spotter, Drafter |
| Android (6GB+) | 6GB+ | Qwen3-1.7B | Q4_K_M | ~1.1GB | Any role |
| Android (4GB) | 4GB | Qwen3-0.6B | Q4_K_M | ~0.4GB | Spotter, Drafter |
| Raspberry Pi 5 (8GB) | 8GB | Qwen3-1.7B | Q4_K_M | ~1.1GB | Aggregator |
| Raspberry Pi 4 (4GB) | 4GB | Qwen3-0.6B | Q4_K_M | ~0.4GB | Aggregator |
| MacBook (M-series) | 8GB+ | Qwen3-4B | Q4_K_M | ~2.5GB | Any role |

Models are loaded from the device's local storage. There is no cloud dependency.

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

REVIVE includes a complete fine-tuning pipeline to produce specialized models for each agent role and the aggregator.

### Pipeline Overview

```
1. Generate synthetic data    →  Claude API (Haiku, ~$8 for 2000 examples)
2. QLoRA fine-tune on Qwen3  →  EC2 g5.xlarge (A10G) or local GPU
3. Export to GGUF (Q4_K_M)   →  Ready for on-device deployment
```

### Aggregator Training

The aggregator model learns to synthesize multi-agent responses. Training data includes device metrics context so the model understands that responses from faster devices may be more thorough.

```bash
# Generate 2000 aggregation training examples
ANTHROPIC_API_KEY=sk-... python3 training/generate_dataset.py --n 2000

# Train aggregator (QLoRA on Qwen3-1.7B, LoRA rank 64)
python3 training/train.py --data data.jsonl --output ./output --lora-r 64 --seq-len 4096

# Output: output/revive-aggregator-1.7b-Q4_K_M.gguf
```

Training data format includes device context:
```
[Reasoner — iPhone 15 Pro @ 30 tok/s]: Step-by-step analysis...
[Writer — Pixel 7 @ 12 tok/s]: Clear explanation...
[Critic — Raspberry Pi @ 5 tok/s]: However, this overlooks...
```

The model learns to weight responses by quality (not just combine them) and to catch and correct factually wrong answers from individual agents.

### Role-Specific Training

Each agent role can be fine-tuned to better embody its persona:

```bash
# Generate role-specific training data
ANTHROPIC_API_KEY=sk-... python3 training/generate_role_dataset.py --role all --n 300

# Train individual roles
python3 training/train_role.py --role reasoner --data data-reasoner.jsonl
python3 training/train_role.py --role spotter  --data data-spotter.jsonl --base-model Qwen/Qwen3-0.6B

# Or train everything at once
bash training/train_all.sh
```

### Training Configuration

| Parameter | Aggregator | Large Roles | Small Roles |
|-----------|-----------|-------------|-------------|
| Base model | Qwen3-1.7B | Qwen3-1.7B | Qwen3-0.6B |
| LoRA rank | 64 | 32 | 32 |
| LoRA alpha | 128 | 64 | 64 |
| Sequence length | 4096 | 2048 | 1024 |
| Epochs | 3 | 3 | 3 |
| Learning rate | 2e-4 | 2e-4 | 2e-4 |
| Quantization | 4-bit (QLoRA) | 4-bit (QLoRA) | 4-bit (QLoRA) |
| Export | Q4_K_M GGUF | Q4_K_M GGUF | Q4_K_M GGUF |

Training uses [Unsloth](https://github.com/unslothai/unsloth) for 2x faster QLoRA and [TRL](https://github.com/huggingface/trl) SFTTrainer. Estimated time: ~30 minutes per role on an A10G GPU.

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
├── training/                        # Fine-tuning pipeline
│   ├── generate_dataset.py          # Synthetic aggregation data via Claude API
│   ├── generate_role_dataset.py     # Per-role fine-tuning data
│   ├── train.py                     # QLoRA fine-tune aggregator (Unsloth)
│   ├── train_role.py                # QLoRA fine-tune individual roles
│   └── train_all.sh                 # Master script: generate + train all
│
├── PROTOCOL.md                      # Cross-platform protocol specification
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
