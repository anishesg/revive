# REVIVE Architecture

This document describes the system-level design of REVIVE. It complements:

- **`README.md`** — user-facing quick start and conceptual overview
- **`PROTOCOL.md`** — wire-level mDNS + HTTP contract between nodes
- **`LLM/README.md`** — build-time pipeline internals

If you're new to the codebase, read this first, then dive into the two other docs as needed.

---

## 1. Two-Module Design

REVIVE is intentionally split into two independent modules that meet at a narrow, well-defined interface:

```
                        ┌──────────────────────────────┐
                        │     LLM Module               │
                        │     (LLM/)                   │
                        │                              │
                        │  Builds role-specialized,    │
                        │  device-tiered GGUF models   │
                        │  from Qwen3 base weights     │
                        └──────────────┬───────────────┘
                                       │
                                       │  Contract: GGUF files + manifest.json
                                       │  (see §4)
                                       ▼
                        ┌──────────────────────────────┐
                        │   Distributed System Module  │
                        │   (revive/, android/, rpi/,  │
                        │    macos/)                   │
                        │                              │
                        │  Discovers peers, fans out   │
                        │  queries, runs inference,    │
                        │  synthesizes MoA responses   │
                        └──────────────────────────────┘
```

**Why two modules:** They have fundamentally different lifecycles, dependencies, and target environments.

| Aspect | LLM Module | Distributed System Module |
|--------|-----------|---------------------------|
| Runs on | ML workstation (A10G GPU) + MacBook | Any REVIVE-participating device |
| Lifecycle | Offline, batch, periodic | Online, continuous, real-time |
| Output | Static artifacts (`.gguf`, `manifest.json`) | Ephemeral inference responses |
| Language | Python (HuggingFace, llama.cpp tooling) | Swift (iOS), Kotlin + C++ (Android), Python (Pi/Mac) |
| Fails like | Bad eval scores, OOM during training | Network partitions, thermal throttling |

The contract between them (§4) is intentionally minimal: the LLM module produces GGUF files with a specific filename convention and a manifest. The distributed system module knows nothing about training, pruning, or quantization — it only knows how to load a GGUF by path and serve inference.

---

## 2. The Distributed System Module

### 2.1 Components

```
                     ┌─────────────────┐
                     │   Coordinator   │  ◄── iPad, Pi, or MacBook
                     │   (SwarmManager,│     Discovers workers,
                     │   QueryRouter,  │     classifies queries,
                     │   Aggregator)   │     synthesizes answers
                     └────────┬────────┘
                              │ HTTP fan-out
          ┌──────────┬────────┼────────┬──────────┐
          ▼          ▼        ▼        ▼          ▼
       ┌──────┐  ┌──────┐ ┌──────┐ ┌──────┐   ┌──────┐
       │Worker│  │Worker│ │Worker│ │Worker│   │Worker│
       │ iOS  │  │ iOS  │ │Android│ │ Pi  │   │MacBook│
       │Reasoner│ │Writer│ │Critic│ │Fact-│   │Drafter│
       │      │  │      │ │      │ │check│   │       │
       └──────┘  └──────┘ └──────┘ └──────┘   └──────┘
```

Each component has one job:

**Worker** — Loads exactly one GGUF into `llama.cpp`, exposes `/health` and `/v1/chat/completions`, advertises itself via mDNS with its role and capabilities. Implementations:
- `revive/Worker/` (Swift, iOS/iPadOS)
- `android/app/src/main/java/com/revive/worker/` (Kotlin + JNI C++)
- `rpi/worker.py` + `macos/revive-cli.py` (Python, spawns `llama-server` subprocess)

**Coordinator** — Discovers workers via mDNS, classifies incoming queries, fans them out to matching roles, collects responses within a timeout, hands them to the aggregator. Implementations:
- `revive/Coordinator/SwarmManager.swift` + `QueryRouter.swift` (iOS)
- `rpi/aggregator.py` (Pi as mesh aggregator)
- `macos/revive-cli.py --coordinator` (MacBook CLI)

**Aggregator** — A model-backed synthesizer. Receives agent responses and produces one final answer via MoA synthesis. Implementations:
- `revive/Coordinator/Aggregator.swift` (iOS, in-process `llama.cpp`)
- `rpi/aggregator.py::_aggregate` (Pi, subprocess `llama-server`)

**Web Dashboard** — PWA served by the coordinator, polls `/api/state` every 1.5 s, renders the swarm topology and chat. Files: `revive/Coordinator/WebDashboard/`.

### 2.2 Query lifecycle

```
 1. User submits query
       │
       ▼
 2. Coordinator.QueryRouter.classify(query)
       │    uses Spotter worker (or local fast model)
       ▼
 3. For each (role needed by query_type):
       coordinator → GET worker /v1/chat/completions
       in parallel (asyncio.gather / Swift TaskGroup)
       │
       │  Each worker builds ChatML:
       │    <|im_start|>system
       │    {role.systemPrompt}<|im_end|>
       │    <|im_start|>user
       │    {query}<|im_end|>
       │    <|im_start|>assistant
       │
       │  then calls llama.cpp and streams back
       │
       ▼
 4. Coordinator collects responses within timeout
       (6s simple, 12s complex)
       │
       ▼
 5. If MoA needed and ≥2 responses:
       Aggregator.synthesize(query, responses) → final answer
    Else:
       return highest-weight single response
```

The weight on step 5 comes from runtime metrics reported by each worker:
`weight = min(tokens_per_second / 40, 1.0) × thermal_penalty`

### 2.3 Discovery & transport

- **Service type:** `_revive._tcp` via Bonjour/mDNS
- **TXT record:** `role`, `model`, `ram`, `port`, `platform`, `caps` (see `PROTOCOL.md`)
- **Transport:** HTTP/1.1, OpenAI-compatible JSON on `/v1/chat/completions`
- **Scope:** Single broadcast domain (same WiFi/LAN). Manual `POST /v1/swarm/register` for cross-subnet.

### 2.4 Per-platform inference layer

All four platforms eventually call into `llama.cpp` but wire it up differently:

| Platform | Wrapper | GPU offload | Model hosting |
|----------|---------|-------------|---------------|
| iOS | `revive/Shared/LibLlama.swift` (in-process) | Metal (A-series) | Documents folder |
| Android | `android/app/src/main/cpp/native-lib.cpp` + `LlamaJNI.kt` (in-process) | ARM NEON, optional Vulkan | external files dir |
| Raspberry Pi | `rpi/worker.py` spawns `llama-server` subprocess | ARM NEON (CPU) | `REVIVE_MODEL_PATH` env var |
| MacBook | `macos/revive-cli.py` spawns `llama-server` subprocess | Metal (Apple Silicon, 99 layers offloaded) | `REVIVE_MODEL_PATH` env var |

All four accept **any GGUF by filesystem path** — they don't parse filenames, don't hardcode model names, and don't know about tiers or roles at the model level. The role is a worker-side config (enum/string), not a property baked into the model. This is why the LLM module can ship new tier variants without touching any platform code.

---

## 3. The LLM Module (`LLM/`)

### 3.1 Pipeline overview

```
  HuggingFace      training/          LLM/data/
  Qwen3-0.6B       prompt banks  ──►  generate_expanded_dataset.py
  Qwen3-1.7B                          distill_from_qwen4b.py
  Qwen3-4B                            merge_datasets.py
       │                                     │
       │                                     ▼
       │                              per-role JSONL (~1500 ex)
       │                                     │
       │                                     ▼
       │                         LLM/train/train_qwen_role.py
       │                         (QLoRA r=32, α=64, 3 epochs)
       │                                     │
       │                                     ▼
       │                         merged HF checkpoint (16-bit)
       │                                     │
       │                   ┌─────────────────┼──────────────────┐
       │                   ▼                 ▼                  ▼
       │             prune/layer_   (skip prune      (skip prune
       │             prune.py        for standard     for modern
       │             (aggressive/    + budget tiers)  tier)
       │             moderate)
       │                   │                 │                  │
       └──►convert_hf_to_gguf.py ──► fp16 GGUF per (role, prune-profile)
                                            │
                                            ▼
                             LLM/quantize/imatrix_gen.py
                             (calibration over role-specific prompts)
                                            │
                                            ▼
                             LLM/quantize/quantize_tiers.py
                             (Q2_K/Q3_K_S/Q4_K_M/Q5_K_M per tier)
                                            │
                                            ▼
                     LLM/output/gguf/
                     revive-{role}-qwen3-{size}-{tier}-{quant}.gguf
                     + manifest.json (SHA-256, size_bytes, role, tier, quant)
```

### 3.2 Track breakdown

The pipeline has three independent tracks that cross at the merged HF checkpoint:

**Track 1: Data quality.** Expanded Haiku dataset (750 ex/role) + local Qwen3-4B teacher distillation (750 ex/role) → deduped ~1500 ex/role. Roughly 5× the original `training/` default. Files: `data/generate_expanded_dataset.py`, `data/distill_from_qwen4b.py`, `data/merge_datasets.py`.

**Track 2: Role-aware pruning.** ShortGPT block-importance heuristic (arXiv 2403.17887): score each transformer block by the angular change between its input and output hidden states across calibration prompts; drop the N lowest-scoring blocks. Profiles per role (`prune/prune_profiles.yaml`) — aggressive for Spotter, moderate for Drafter/Concise, none for reasoning-heavy roles. Files: `prune/layer_prune.py`, optional `prune/heal_lora.py` if quality regresses.

**Track 3: Heterogeneous quantization.** imatrix-calibrated K-quants per device tier:
- `ewaste` → Q2_K (1–3 GB RAM devices)
- `budget` → Q3_K_S (3–4 GB)
- `standard` → Q4_K_M (4–6 GB, matches current ship state)
- `modern` → Q5_K_M (6+ GB)

Files: `quantize/imatrix_gen.py`, `quantize/quantize_tiers.py`, `common/device_tiers.yaml`.

### 3.3 Relationship to `training/`

`training/` is the legacy Qwen3 QLoRA pipeline from the initial REVIVE ship. **`LLM/` does not modify `training/`.** It imports prompt banks and system prompts from `training/generate_role_dataset.py` as a Python module and mirrors hyperparameters from `training/train_role.py`, but owns its own scripts and output directory. This lets the legacy pipeline stay as a reference/fallback while the new tier-aware pipeline evolves independently.

### 3.4 Evaluation

`LLM/eval/eval_role.py` runs role-appropriate metrics against a Qwen3-4B teacher:

| Role | Metric | Passing |
|------|--------|---------|
| Spotter | Classification accuracy (exact match on 6 categories) | ≥95% |
| Drafter, Concise | Agreement rate (Haiku judge, semantic equivalence) | ≥70% |
| Reasoner, Factchecker | Agreement rate + factuality regression | ≥60%, ≤10% regression |
| Writer | BLEU vs teacher | within 10% of teacher |
| Critic | Flaw-detection precision on adversarial set | TBD |
| Aggregator | Human spot-check on 50 syntheses | manual |

`LLM/eval/eval_tier.py` sweeps quant/prune variants for a role and reports tok/s + exact-agreement-vs-modern, so you can decide whether a compressed tier is actually usable or needs to fall back one tier up.

---

## 4. The Contract Between Modules

This is the thin interface both modules must honor. Changes here require coordinated updates on both sides.

### 4.1 File artifacts

Every deployable model is a single GGUF file at:

```
LLM/output/gguf/revive-{role}-qwen3-{size}-{tier}-{quant}.gguf
```

Where:
- `role` ∈ {spotter, drafter, concise, reasoner, writer, critic, factchecker, aggregator}
- `size` ∈ {0.6b, 1.7b} (per `LLM/common/role_registry.py`)
- `tier` ∈ {ewaste, budget, standard, modern} (per `LLM/common/device_tiers.yaml`)
- `quant` ∈ {Q2_K, Q3_K_S, Q4_K_M, Q5_K_M}

A sibling `manifest.json` lists every emitted file with role, tier, quant, size in bytes, and SHA-256. Schema:

```json
{
  "version": 1,
  "entries": [
    {
      "file": "revive-spotter-qwen3-0.6b-ewaste-Q2_K.gguf",
      "role": "spotter",
      "base": "qwen3",
      "size": "0.6b",
      "tier": "ewaste",
      "quant": "Q2_K",
      "size_bytes": 145678901,
      "sha256": "abc123..."
    }
  ]
}
```

### 4.2 Prompt format

All workers speak **ChatML** because Qwen3 was pretrained on it:

```
<|im_start|>system
{role_system_prompt}<|im_end|>
<|im_start|>user
{query}<|im_end|>
<|im_start|>assistant
```

The canonical per-role system prompts live in `revive/Shared/Models.swift` (`AgentRole.systemPrompt`) and are mirrored in `rpi/aggregator.py::SYSTEM_PROMPTS`, `macos/revive-cli.py`, and `training/generate_role_dataset.py::SYSTEM_PROMPTS`. If you change them, update all five.

### 4.3 Tier selection

The distributed system module currently does **not** auto-select the right tier GGUF per device — the operator picks which GGUF to deploy to each device (copy to iOS Documents folder, set `REVIVE_MODEL_PATH` on Pi/Mac). The `manifest.json` is there to enable automatic selection in a future version, where `SwarmManager.swift` or `rpi/aggregator.py` reads the manifest, compares to the device's advertised `ram` TXT field, and picks the highest tier that fits. See §6 (Future Work).

### 4.4 What the distributed system knows about the LLM

Almost nothing, by design:
- It accepts any valid GGUF by path.
- It assumes ChatML at inference time (could be generalized, see §6).
- It reads `role` from worker config, never from the model file.
- It reports `model` as a string in mDNS TXT records for observability only — the coordinator doesn't branch on it.

This looseness is load-bearing. It means the LLM module can ship a new base (Llama-3.2, Gemma, TinyLLM) or a new quant tier without any device-side change, as long as the new GGUF speaks ChatML.

---

## 5. Deployment Flow

End-to-end, from Qwen3 base weights to a query answered on a phone:

```
 1. Build-time (offline, ML workstation)
    ─────────────────────────────────────
    a. bash LLM/scripts/bootstrap.sh
         → generates data, trains 8 roles, emits standard tier GGUFs
    b. bash LLM/scripts/compress_all.sh
         → emits full tier matrix (up to 32 GGUFs) + manifest.json
    c. python LLM/eval/eval_role.py --all
         → validates each role meets its passing metric

 2. Distribution (manual, operator)
    ─────────────────────────────────
    Pick the tier that matches each device's RAM.
      iPhone 15 Pro  → revive-reasoner-qwen3-1.7b-modern-Q5_K_M.gguf
      Pixel 8        → revive-writer-qwen3-1.7b-standard-Q4_K_M.gguf
      Pi 4 (4GB)     → revive-spotter-qwen3-0.6b-budget-Q3_K_S.gguf
      iPhone 7       → revive-drafter-qwen3-0.6b-ewaste-Q2_K.gguf

    Transfer:
      iOS      → Files app or Finder → app's Documents folder
      Android  → adb push to Android/data/com.revive.worker/files/models/
      Pi/Mac   → scp + export REVIVE_MODEL_PATH=/path/to/model.gguf

 3. Runtime (online, local WiFi)
    ─────────────────────────────
    a. Each device starts in Worker mode with its assigned role + GGUF.
    b. Devices auto-advertise via mDNS.
    c. One device runs Coordinator mode and discovers the others.
    d. User submits a query via the Coordinator's chat UI or web dashboard.
    e. QueryRouter classifies, fans out to matching workers.
    f. Workers run llama.cpp locally, return responses + metrics.
    g. Aggregator synthesizes a final answer.
```

No step involves the internet after artifact distribution. Everything runs on local WiFi with local weights.

---

## 6. Future Work

These are acknowledged gaps in the current architecture, in rough priority order.

### 6.1 Tier auto-selection from manifest

Today an operator has to pick the right GGUF for each device. With `manifest.json` already emitted, the missing piece is a device-side selector:

- On worker startup, read the `ram` from the device and the `manifest.json` (fetched from coordinator or bundled).
- Pick the highest-tier GGUF whose RAM envelope matches.
- If multiple roles available for the device, pick one based on swarm coverage (coordinator tells the worker what role is needed).

Changes needed: `rpi/aggregator.py` serves `/manifest.json`, `revive/Worker/WorkerView.swift` and Android equivalent add a "pick best for this device" flow.

### 6.2 Heterogeneous prompt formats

Currently all workers speak ChatML (Qwen3 native). If we later ship a TinyLLM-based or Llama-3-based worker, the prompt format differs. The clean extension is:

- `mDNS TXT: prompt_format ∈ {chatml, alpaca, llama3, tinyllm_role_query}`
- Coordinator caches the format per worker and builds the request accordingly.
- Default: `chatml` (backwards compatible with every current Qwen3 deployment).

### 6.3 Swarm-level speculative decoding

Today "Speed Mode" pairs a Drafter with a Verifier at the **agent** level — each runs full generation independently. True spec decoding would have the Drafter emit N candidate tokens, send them to the Verifier, and have the Verifier validate in one batched forward pass. This requires:

- A new `/v1/completions/verify` endpoint on workers
- Drafter-Verifier pairing logic in `QueryRouter.swift` / `rpi/aggregator.py`
- A spec-compatible decoder in `LibLlama.swift`

Payoff is large for simple queries on well-provisioned networks. Not started.

### 6.4 Modular LoRA adapter swarm

Instead of shipping N role-specific GGUFs, ship one base model + N tiny LoRA adapters (~few MB each). Workers merge the adapter for their assigned role at load time. Total swarm storage drops from 8 × 1.1 GB to 1.1 GB + 8 × ~10 MB.

Blockers: on-device LoRA merge in `llama.cpp` is workable on iOS/MacBook but awkward on Android/Pi. Revisit when llama.cpp improves this.

### 6.5 Observability and evals in production

Per-query telemetry beyond single-worker metrics: end-to-end latency distribution, aggregator cache hit rate, per-role quality scored by a reference judge, thermal correlation with quality. Needed before we can tune defaults confidently.

---

## 7. Glossary

| Term | Meaning |
|------|---------|
| **MoA** | Mixture-of-Agents. Multiple specialized models run in parallel; a synthesizer picks the best elements from each. The alternative REVIVE rejects is pipeline parallelism (one model split across devices). |
| **Role** | One of 8 worker personalities: Spotter, Drafter, Concise, Reasoner, Writer, Critic, Factchecker, Aggregator. Determined by config at worker start, not by the model file. |
| **Tier** | Device-class bucket. ewaste/budget/standard/modern, bucketed by RAM. Determines which quant + prune profile the LLM module emits for a given role. |
| **GGUF** | llama.cpp's self-contained model file format. Quantized weights + tokenizer + chat template in one file. |
| **imatrix** | Importance matrix computed by llama.cpp's `./llama-imatrix` over calibration prompts; required input for high-quality Q2/Q3 quantization. |
| **Block importance (ShortGPT)** | Pruning heuristic from arXiv 2403.17887: score each transformer block by the angular change between its input and output hidden states; drop the least-changing blocks first. |
| **QLoRA** | Quantized Low-Rank Adaptation. Fine-tunes a 4-bit-quantized base model via small LoRA adapters, enabling training on consumer GPUs. |
| **ChatML** | Qwen3's native prompt format: `<|im_start|>{role}\n{content}<|im_end|>` turn markers. |

---

## 8. Navigation

| Looking for... | Read... |
|----------------|---------|
| User quick-start (how do I join the swarm?) | `README.md` |
| mDNS TXT fields, HTTP endpoint contracts | `PROTOCOL.md` |
| Current ship state & performance numbers | `README.md` §Performance |
| How to train a new role | `LLM/README.md` |
| How to add a new device tier | `LLM/common/device_tiers.yaml` |
| How to change a role's system prompt | `revive/Shared/Models.swift::AgentRole.systemPrompt` + mirror files (§4.2) |
| On-device inference wrapper code | `revive/Shared/LibLlama.swift`, `android/.../native-lib.cpp`, `rpi/worker.py`, `macos/revive-cli.py` |
| Coordinator routing logic | `revive/Coordinator/QueryRouter.swift`, `rpi/aggregator.py` |
| Aggregator (MoA synthesis) | `revive/Coordinator/Aggregator.swift`, `rpi/aggregator.py::_aggregate` |
