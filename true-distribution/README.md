# REVIVE — True Distributed LLM Inference

One Qwen3 model. Its layers are physically split across multiple devices —
phones, an iPad, Raspberry Pis, an old Android — connected over HTTP. Each
device holds only a slice of the weights. No single node can answer alone.
Together they do a full forward pass per token.

A **$10 Arduino Uno** is the cluster's management plane. It owns the
authoritative partition table, detects worker failures via heartbeat
timeouts, and refuses to let the coordinator route through a node it has
declared dead. Unplug the Arduino and the cluster stops scheduling. This
mirrors how real datacenters use a BMC (Baseboard Management Controller)
separate from the compute path.

For local development, `pipeline/bmc_sim.py` speaks the exact same line
protocol as the Arduino firmware (`arduino/revive_bmc.ino`) over TCP, so
the whole system is testable without any hardware plugged in.

## What's here

- **Pipeline-parallel inference** — Qwen3 model layers split across N
  workers, hidden states travel over HTTP, KV cache lives per-stage on
  each worker. Verified token-for-token correct against the reference
  HuggingFace model under greedy sampling.
- **BMC control plane** — single source of truth for cluster membership,
  partition assignment, and failure detection. Same line protocol on real
  Arduino and Python simulator.
- **PipeEdge-DP layer partitioner** — minimizes the slowest pipeline
  stage given heterogeneous worker capabilities. Greedy fallback that
  fits in 100 lines of C for the Arduino's 2 KB of RAM.
- **Web dashboard** — live cluster topology, BMC event log, per-token
  streaming, per-stage latency bars showing the pipeline bottleneck,
  one-click chaos engineering (kill / heal a worker).
- **Benchmark CLI** — measures tok/s, prefill vs decode, per-stage
  latency, wire bandwidth used. Output is screenshot-ready for the pitch.

## Architecture

```
              ┌───────────────────────────────────┐
              │   Arduino Uno  (BMC firmware)      │   control plane
              │   • partition table                │   USB serial @ 115200
              │   • heartbeat watchdog             │   (or TCP to bmc_sim)
              │   • failure detection              │
              └─────────────┬─────────────────────┘
                            │ line protocol (bmc_protocol.py)
              ┌─────────────▼─────────────────────┐
              │    ClusterController (Python)      │   bridges BMC ↔
              │    + Dashboard server              │   workers + UI
              └──┬──────┬──────┬─────────┬─────────┘
                 │      │      │         │  hidden states (fp16, ~2KB/token)
                 ▼      ▼      ▼         ▼
            ┌──────┐┌──────┐┌──────┐┌──────┐
            │iPhone││ iPad ││Andrd ││ Pi   │   compute plane
            │L0..A ││LA..B ││LB..C ││LC..L │
            └──────┘└──────┘└──────┘└──────┘
```

## Quick start (2 virtual nodes on your Mac)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch transformers aiohttp pyserial-asyncio numpy

scripts/launch_cluster.sh Qwen/Qwen3-0.6B 2
# open http://127.0.0.1:4100 — type a prompt, watch the ring,
# click KILL on a worker to chaos-engineer a failure, click HEAL to recover.
```

For a real Arduino in the loop:
```bash
SERIAL=auto scripts/launch_cluster.sh   # auto-detect Arduino on USB
# or:
SERIAL=/dev/tty.usbmodem14101 scripts/launch_cluster.sh
```

## Measured performance

3-worker ring on a single Mac, Qwen3-0.6B (28 layers split 10/9/9), fp16,
30 tokens per query:

```
mean tok/s             14.59
median tok/s           15.87
mean prefill (ms)     455.3
mean per-token (ms)    56.3

per-stage latency breakdown
─────────────────────────────────────────────
0  w0  [0..10)    16.2 ms   26.5%   ████████
1  w1  [10..19)   15.4 ms   25.2%   ████████
2  w2  [19..28)   29.5 ms   48.3%   ████████████████

→ bottleneck: worker w2 at 29.5 ms/step (48% of pipeline time)

wire bandwidth used
─────────────────────────────────────────────
hidden state size            2048 bytes (1024 × fp16)
ring hops per token          2
bandwidth at 14.6 tok/s      58 KB/s
headroom on 100 Mbps WiFi    214× over
```

The bandwidth math is the unlock — pipeline parallelism over even
802.11n WiFi is two orders of magnitude under the link capacity. The
constraint is per-stage compute, not the network.

## Demo script (chaos engineering)

```bash
# Terminal 1
scripts/launch_cluster.sh Qwen/Qwen3-0.6B 3

# Browser → http://127.0.0.1:4100
# 1. Type:  "What is the capital of France?"
#    → answer streams in, per-stage latency bar appears
# 2. Click KILL on worker w1
#    → BMC emits DEAD w1 in event log, partition repartitions
#    → cluster pill flips from HEALTHY (green) to DEGRADED (amber)
# 3. Type any prompt
#    → "GENERATION ABORTED: BMC declared worker w1 dead" in red
# 4. Click HEAL on w1
#    → BMC emits ALIVE w1, partition restored, pill back to HEALTHY
# 5. Type any prompt → works again
```

## Tests

```bash
PYTHONPATH=. python tests/test_partitioner.py       # scheduler correctness
PYTHONPATH=. python tests/test_stage_smoke.py       # layer-sliced loading
PYTHONPATH=. python tests/test_correctness.py       # distributed == single-model
PYTHONPATH=. python tests/test_one_stage.py         # decode works (single stage)
PYTHONPATH=. python tests/test_generation_local.py  # 2-stage greedy == reference
PYTHONPATH=. python tests/test_three_node.py        # 3-stage greedy == reference
PYTHONPATH=. python tests/test_bmc_integration.py   # BMC + chaos + heal
```

All tests are self-contained `__main__` runners; no pytest required.

## Benchmarking

```bash
# Run 5 queries, 50 tokens each, against a running ring
PYTHONPATH=. python -m scripts.benchmark \
    --model Qwen/Qwen3-0.6B \
    --workers 127.0.0.1:50100 127.0.0.1:50101 127.0.0.1:50102 \
    --runs 5 --tokens 50
```

## Files

```
pipeline/
  protocol.py          tensor wire format (coordinator ↔ worker)
  worker.py            a pipeline stage: layer-sliced forward, own KV cache
  coordinator.py       runs the ring, streams tokens, respects BMC's view
  partitioner.py       greedy + PipeEdge-DP layer assignment
  bmc_protocol.py      host ↔ Arduino line protocol spec (v2)
  bmc_sim.py           Python implementation of the Arduino firmware,
                       speaks the identical protocol over TCP
  controller.py        host-side client of the BMC (TCP or serial)
  multi_bmc.py         HA controller for clusters with multiple BMCs
                       (leader election, replica failover)

arduino/
  revive_bmc.ino       production BMC firmware for Arduino Uno
  README.md            how to flash and verify

dashboard/
  server.py            web dashboard (embeds BMC sim + controller + coord)
  index.html           BMC state, topology, chaos buttons, token stream

scripts/
  launch_local.sh      spawn N workers for ad-hoc dev
  launch_cluster.sh    workers + dashboard (full stack); SERIAL= for Arduino
  stop_local.sh        kill workers
  benchmark.py         throughput + per-stage latency + bandwidth report
  demo_full.py         CLI chaos-engineering demo

tests/
  test_partitioner.py        scheduler tests
  test_stage_smoke.py        PipelineStage can load + forward
  test_correctness.py        distributed hidden states == single-model
  test_one_stage.py          decode works with one stage covering all layers
  test_generation_local.py   2-stage distributed greedy == single-model greedy
  test_three_node.py         3-stage distributed greedy == single-model greedy
  test_bmc_integration.py    BMC protocol + chaos + heal
  test_decode_debug.py       diagnostic tool for KV cache issues
  test_decode_args.py        confirms transformers cache args are well-formed
  test_trace.py              layer-by-layer activation magnitude comparison
```

## Status

What's done:
- Layer-sliced loading with per-slice KV cache (M1)
- N-stage HTTP ring with custom binary wire protocol (M2)
- Heterogeneous BMC-driven layer partitioner (M3)
- Live dashboard with chaos controls and per-stage latency viz (M4)
- Real Arduino USB serial integration with auto-detection
- Multi-BMC HA (leader election, replica failover)
- Benchmark CLI

Deferred:
- **Pipeline overlap (M5).** Single-user greedy decode is serially
  dependent — token N+1's input is the sampled output of token N, so
  there's no parallelism to exploit. Only relevant for prefill of long
  prompts (Jupiter intra-sequence PP) or with speculative decoding.
- **iOS port.** The Python worker is the reference implementation. The
  iOS port translates `pipeline/worker.py` to Swift wrapping a forked
  llama.cpp that exposes a `llama_decode_layers(start, end)` API. ~1
  week of C++ + Swift work.
- **Quantization.** Prototype runs fp16; production would use GGUF
  Q4_K_M (4× memory + bandwidth savings).
