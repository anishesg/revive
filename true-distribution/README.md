# REVIVE — True Distributed LLM Inference

One Qwen3 model. Its layers are physically split across multiple
devices — phones, a Raspberry Pi, a laptop — connected over HTTP. Each
device holds only a slice of the weights. No single node can answer
alone. Together they do a full forward pass per token.

A **$10 Arduino Uno** is the cluster's management plane. It owns the
authoritative partition table, detects worker failures via heartbeat
timeouts, and refuses to let the coordinator route through a node it's
declared dead. Unplug the Arduino, the cluster stops scheduling. This
mirrors how real datacenters use a BMC (Baseboard Management Controller)
separate from the compute path.

For local development, `pipeline/bmc_sim.py` speaks the exact same line
protocol as the Arduino firmware (`arduino/revive_bmc.ino`) over TCP, so
the whole system is testable without any hardware plugged in.

## Architecture

```
                  ┌──────────────────────────┐
                  │  Arduino Uno (BMC)       │   control plane
                  │  • partition table       │   (serial @ 115200, USB)
                  │  • heartbeat watchdog    │   or TCP to bmc_sim for dev
                  │  • fault detection       │
                  └─────────────┬────────────┘
                                │
                 ┌──────────────▼──────────────┐
                 │  ClusterController (Python) │   relays between BMC
                 └──────────────┬──────────────┘   and workers/dashboard
                                │
        ┌──────────────┬────────┼────────┬──────────────┐
        ▼              ▼        │        ▼              ▼
    ┌────────┐    ┌────────┐    │    ┌────────┐    ┌────────┐
    │iPhone  │    │iPad    │    │    │Android │    │Raspberry│
    │layers  │    │layers  │    │    │layers  │    │Pi layers│
    │ [0..A) │───▶│ [A..B) │───▶│───▶│ [B..C) │───▶│[C..L)  │
    └────────┘    └────────┘    │    └────────┘    └────────┘
        token ids        hidden │  hidden        hidden    next token
                                │                              │
                          ┌─────▼──────┐                       │
                          │ Coordinator│◀──────────────────────┘
                          │  + Dash    │   tokens back to dashboard UI
                          └────────────┘
```

## Quick start (2 virtual nodes on your Mac)

```bash
# One-time setup
python3 -m venv .venv
source .venv/bin/activate
pip install torch transformers aiohttp

# Launch the whole stack (BMC sim embedded in the dashboard process)
scripts/launch_cluster.sh Qwen/Qwen3-0.6B 2

# Open http://127.0.0.1:4100 — type a prompt, watch the ring,
# click KILL on a worker to chaos-engineer a failure, click HEAL to recover.
```

## Testing

```bash
PYTHONPATH=. python tests/test_partitioner.py       # scheduler correctness
PYTHONPATH=. python tests/test_stage_smoke.py       # layer-sliced loading
PYTHONPATH=. python tests/test_correctness.py       # distributed == single-model
PYTHONPATH=. python tests/test_generation_local.py  # greedy decode token-match
PYTHONPATH=. python tests/test_bmc_integration.py   # BMC protocol + failover
```

Every test is a self-contained `__main__` runner; no pytest required.

## Protocol reference

The Arduino line protocol lives in `pipeline/bmc_protocol.py`. It is the
exact same wire format spoken by `arduino/revive_bmc.ino` (real hardware)
and `pipeline/bmc_sim.py` (dev simulator). This is the API surface between
the host and the cluster's single source of truth.

## Files

```
pipeline/
  protocol.py          tensor wire format (coordinator ↔ worker)
  worker.py            a pipeline stage: layer-sliced forward, own KV cache
  coordinator.py       runs the ring, streams tokens, respects BMC's view
  partitioner.py       greedy + PipeEdge-DP layer assignment (same algo as
                       the Arduino firmware)
  bmc_protocol.py      host ↔ Arduino line protocol spec
  bmc_sim.py           in-process Python implementation of the Arduino
                       firmware, speaks the identical protocol over TCP
  controller.py        host-side client of the BMC (TCP or serial)

arduino/
  revive_bmc.ino       production BMC firmware for Arduino Uno, flashable

dashboard/
  server.py            web dashboard (embeds BMC sim + controller + coord)
  index.html           BMC state, topology, chaos buttons, token stream

tests/
  test_partitioner.py         scheduler tests
  test_stage_smoke.py         PipelineStage can load + forward
  test_correctness.py         distributed hidden states == single-model
  test_generation_local.py    distributed greedy == single-model greedy
  test_decode_debug.py        diagnostic tool for KV cache issues
  test_trace.py               layer-by-layer magnitude comparison
  test_one_stage.py           generation with one stage covering all layers
  test_bmc_integration.py     BMC protocol + chaos + heal

scripts/
  launch_local.sh     spawn N workers for ad-hoc dev
  stop_local.sh       kill workers
  launch_cluster.sh   workers + dashboard (full stack)
  demo_full.py        CLI chaos-engineering demo
```

## What works, what doesn't yet

Works:
- Layer-sliced loading with per-slice KV cache
- End-to-end ring inference (prefill + decode) correct to reference
- BMC-authoritative partitioning (greedy + DP reference)
- Heartbeat-based liveness
- Failover detection (cluster refuses to route through dead nodes)
- Recovery on re-announcement (heartbeat after dead → ALIVE)
- Web dashboard with live topology visualization and chaos buttons
- All state machines testable without hardware via TCP simulator

Not yet:
- Dynamic layer-range reload on failover (workers launch with fixed ranges;
  recovery requires the dead node to come back, not layers migrating)
- Jupiter-style intra-sequence pipeline parallelism (M5 stretch)
- iOS port — translate `pipeline/worker.py` to Swift wrapping a forked
  llama.cpp that exposes `llama_decode_layers(start, end)`
- Quantization — prototype runs fp16; production would use GGUF Q4_K_M
- Real Arduino over USB serial (SerialLink class is stubbed; pyserial-asyncio
  integration is ~20 lines away when hardware shows up)
