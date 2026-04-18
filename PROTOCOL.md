# REVIVE Cross-Platform Protocol Specification

## Overview

All REVIVE nodes (iPhone, Android, Raspberry Pi, MacBook) communicate using this protocol.
The Raspberry Pi acts as the **mesh aggregator** — the central coordination hub.
Any device can be a **worker** (runs inference) or the **aggregator** (routes + synthesizes).

## Service Discovery (mDNS/Bonjour)

**Service type:** `_revive._tcp`

**TXT record fields:**
| Key       | Type   | Description                                         |
|-----------|--------|-----------------------------------------------------|
| `role`    | string | Agent role: reasoner, writer, concise, critic, factchecker, drafter, spotter, aggregator |
| `model`   | string | Model filename (e.g., `qwen3-1.7b-q4_k_m`)         |
| `ram`     | string | Device RAM in MB                                    |
| `port`    | string | HTTP server port                                    |
| `platform`| string | `ios`, `android`, `rpi`, `macos`                    |
| `caps`    | string | Comma-separated: `metal`, `neon`, `vulkan`, `cpu`   |

## HTTP API

All nodes expose:

### `GET /health`
Returns: `{"status": "ok", "role": "<role>", "platform": "<platform>", "uptime": <seconds>}`

### `POST /v1/chat/completions`
OpenAI-compatible endpoint.

**Request:**
```json
{
  "model": "qwen3-1.7b-q4_k_m",
  "messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}],
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

### `GET /metrics`
Returns current device telemetry without running inference.

```json
{
  "platform": "rpi",
  "role": "aggregator",
  "thermal_state": "nominal",
  "battery_percent": -1,
  "memory_used_mb": 1024,
  "memory_total_mb": 4096,
  "cpu_temp_c": 52.3,
  "tokens_per_second_last": 12.4,
  "uptime_seconds": 3600,
  "active_workers": 5
}
```

### `POST /v1/swarm/query` (Aggregator only)
Submit a query to the full swarm. The aggregator handles routing + synthesis.

**Request:**
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
    {"role": "reasoner", "model": "qwen3-1.7b", "content": "...", "metrics": {...}},
    {"role": "critic", "model": "qwen3-0.6b", "content": "...", "metrics": {...}}
  ],
  "final_answer": "Synthesized answer from aggregator...",
  "mode": "swarm",
  "total_time_ms": 4500,
  "agents_responded": 3
}
```

### `POST /v1/swarm/register` (Aggregator only)
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

### `GET /v1/swarm/status` (Aggregator only)
Returns full swarm topology.

```json
{
  "aggregator": {"host": "192.168.1.10", "platform": "rpi", "uptime": 3600},
  "workers": [
    {"name": "REVIVE-reasoner-iPhone", "role": "reasoner", "host": "192.168.1.20", "port": 50001, "status": "idle", "platform": "ios", "metrics": {...}},
    {"name": "REVIVE-writer-Pixel", "role": "writer", "host": "192.168.1.30", "port": 8080, "status": "idle", "platform": "android", "metrics": {...}}
  ],
  "total_workers": 2,
  "total_tps": 35.2
}
```

## Port Conventions

| Platform | Default Port |
|----------|-------------|
| iOS      | 50001       |
| Android  | 8080        |
| Raspberry Pi (aggregator) | 9090 |
| Raspberry Pi (worker)     | 8080 |
| MacBook  | 8080        |

## Model Recommendations by Device

| Device           | RAM   | Recommended Model      | Quant  | Size  |
|-----------------|-------|------------------------|--------|-------|
| iPhone 15 Pro   | 8GB   | Qwen3-1.7B            | Q4_K_M | ~1.1GB |
| iPhone 13/14    | 4-6GB | Qwen3-0.6B            | Q4_K_M | ~0.4GB |
| Android (6GB+)  | 6GB+  | Qwen3-1.7B            | Q4_K_M | ~1.1GB |
| Android (4GB)   | 4GB   | Qwen3-0.6B            | Q4_K_M | ~0.4GB |
| Raspberry Pi 5  | 8GB   | Qwen3-1.7B            | Q4_K_M | ~1.1GB |
| Raspberry Pi 4  | 4GB   | Qwen3-0.6B            | Q4_K_M | ~0.4GB |
| MacBook (M-series)| 8GB+ | Qwen3-4B             | Q4_K_M | ~2.5GB |
