#!/usr/bin/env python3
"""
REVIVE Mesh Aggregator — Raspberry Pi
Central coordinator for the phone swarm. Discovers workers via mDNS,
routes queries to specialized agents, aggregates responses via MoA synthesis.

Usage:
    python3 aggregator.py
    # or via systemd: sudo systemctl start revive-aggregator

Environment:
    REVIVE_MODEL_PATH     Path to aggregator GGUF model
    REVIVE_LLAMA_SERVER   Path to llama-server binary
    REVIVE_PORT           HTTP port (default 9090)
"""
import asyncio
import json
import logging
import os
import signal
import socket
import sys
import time
from pathlib import Path

import aiohttp
from aiohttp import web

# Allow imports when run as script or module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from discovery import ServiceAdvertiser, SwarmDiscovery, WorkerNode
import metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("revive.aggregator")

# ── Agent roles and query classification ─────────────────────────────────────

AGENT_SYSTEM_PROMPTS = {
    "reasoner": "You are a rigorous analytical thinker. Think step by step. Show your reasoning chain explicitly. Prioritize logical correctness over brevity.",
    "writer": "You are an eloquent communicator. Write clear, well-structured, engaging responses. Prioritize readability and flow.",
    "concise": "You are a master of brevity. Answer in as few words as possible while being complete and accurate.",
    "critic": "You are a devil's advocate. Identify flaws, edge cases, counterarguments, and unstated assumptions.",
    "factchecker": "You are a fact-checker. Focus only on verifiable, accurate information. Flag anything uncertain with [uncertain].",
    "drafter": "You are a quick-response generator. Produce a fast first-pass answer. Speed matters more than polish.",
    "spotter": "Classify the query into EXACTLY one category (reply with only the word): SIMPLE_FACT, COMPLEX_REASONING, CREATIVE, CODE, MATH, OPINION",
    "aggregator": "You are the Aggregator of a distributed AI swarm. You receive multiple responses from specialized agents and synthesize the single best answer.",
}

QUERY_TYPE_ROLES = {
    "SIMPLE_FACT": ["reasoner", "concise"],
    "COMPLEX_REASONING": ["reasoner", "writer", "critic", "factchecker", "drafter"],
    "CREATIVE": ["writer", "reasoner", "critic"],
    "CODE": ["reasoner", "factchecker", "critic"],
    "MATH": ["reasoner", "factchecker", "concise"],
    "OPINION": ["writer", "critic", "reasoner"],
}

NEEDS_AGGREGATION = {"COMPLEX_REASONING", "CREATIVE", "CODE", "MATH", "OPINION"}


class MeshAggregator:
    """Coordinates the swarm from the Raspberry Pi."""

    def __init__(self, model_path: str, llama_bin: str, port: int = 9090):
        self.port = port
        self.model_path = model_path
        self.llama_bin = llama_bin
        self.discovery = SwarmDiscovery(
            on_worker_added=self._on_worker_added,
            on_worker_removed=self._on_worker_removed,
        )
        self.advertiser = ServiceAdvertiser(
            role="aggregator", model=Path(model_path).stem,
            port=port, platform="rpi",
            ram_mb=metrics.memory_total_mb(), caps="neon",
        )
        self._local_worker = None
        self._http_session: aiohttp.ClientSession | None = None
        self._app = web.Application()
        self._setup_routes()
        self._start_time = time.time()

    def _setup_routes(self):
        self._app.router.add_get("/health", self._handle_health)
        self._app.router.add_get("/metrics", self._handle_metrics)
        self._app.router.add_get("/v1/swarm/status", self._handle_swarm_status)
        self._app.router.add_post("/v1/swarm/query", self._handle_swarm_query)
        self._app.router.add_post("/v1/swarm/register", self._handle_register)
        self._app.router.add_post("/v1/chat/completions", self._handle_completions)
        self._app.router.add_get("/", self._handle_dashboard)
        self._app.router.add_static("/static", os.path.join(os.path.dirname(__file__), "static"), show_index=False)

    # ── Lifecycle ────────────────────────────────────────────────────────────

    async def start(self):
        self._http_session = aiohttp.ClientSession()

        log.info("Starting local llama-server for aggregation...")
        from worker import LlamaWorker
        self._local_worker = LlamaWorker(
            model_path=self.model_path,
            llama_server_bin=self.llama_bin,
            port=self.port + 1,
        )
        await self._local_worker.start()

        self.discovery.start()
        self.advertiser.start()

        asyncio.create_task(self._health_check_loop())

        log.info("Mesh aggregator ready on port %d", self.port)

    async def stop(self):
        self.advertiser.stop()
        self.discovery.stop()
        if self._local_worker:
            await self._local_worker.stop()
        if self._http_session:
            await self._http_session.close()

    # ── Discovery callbacks ──────────────────────────────────────────────────

    def _on_worker_added(self, node: WorkerNode):
        log.info("⚡ Worker joined swarm: %s (%s) on %s", node.name, node.role, node.platform)

    def _on_worker_removed(self, name: str):
        log.info("Worker left swarm: %s", name)

    # ── Health check loop ────────────────────────────────────────────────────

    async def _health_check_loop(self):
        while True:
            await asyncio.sleep(10)
            for name, worker in list(self.discovery.workers.items()):
                try:
                    async with self._http_session.get(
                        f"{worker.url}/health",
                        timeout=aiohttp.ClientTimeout(total=3),
                    ) as resp:
                        if resp.status == 200:
                            worker.last_seen = time.time()
                            worker.status = "idle"
                        else:
                            worker.status = "offline"
                except Exception:
                    if worker.age_seconds > 30:
                        worker.status = "offline"

    # ── Query routing ────────────────────────────────────────────────────────

    async def _classify_query(self, query: str) -> str:
        spotters = self.discovery.get_workers_by_role("spotter")
        if spotters:
            content = await self._query_remote_worker(spotters[0], f"Classify this query: {query}", max_tokens=10)
            category = content.strip().split()[0].upper() if content else ""
            if category in QUERY_TYPE_ROLES:
                return category
        return "COMPLEX_REASONING"

    async def _query_remote_worker(self, worker: WorkerNode, prompt: str, max_tokens: int = 150, system: str | None = None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {"messages": messages, "max_tokens": max_tokens, "temperature": 0.7, "stream": False}

        try:
            async with self._http_session.post(
                f"{worker.url}/v1/chat/completions",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=15),
            ) as resp:
                data = await resp.json()

            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            resp_metrics = data.get("metrics", {})

            self.discovery.update_metrics(
                worker.name,
                tps=resp_metrics.get("tokens_per_second", 0),
                thermal=resp_metrics.get("thermal_state", "unknown"),
                battery=resp_metrics.get("battery_percent", -1),
                memory_mb=resp_metrics.get("memory_used_mb", 0),
            )
            return content

        except Exception as e:
            log.error("Worker %s failed: %s", worker.name, e)
            self.discovery.mark_offline(worker.name)
            return ""

    async def _fan_out_query(self, query: str, target_roles: list[str], timeout: float = 12.0) -> list[dict]:
        """Query all matching workers in parallel."""
        tasks = []
        for role in target_roles:
            workers = self.discovery.get_workers_by_role(role)
            for w in workers:
                system = AGENT_SYSTEM_PROMPTS.get(role, "")
                tasks.append(self._timed_query(w, query, system, timeout))

        if not tasks:
            return []

        results = await asyncio.gather(*tasks, return_exceptions=True)
        responses = []
        for r in results:
            if isinstance(r, dict) and r.get("content"):
                responses.append(r)
        return responses

    async def _timed_query(self, worker: WorkerNode, query: str, system: str, timeout: float) -> dict:
        try:
            content = await asyncio.wait_for(
                self._query_remote_worker(worker, query, system=system),
                timeout=timeout,
            )
            return {"role": worker.role, "model": worker.model, "content": content, "worker": worker.name}
        except asyncio.TimeoutError:
            log.warning("Worker %s timed out", worker.name)
            return {"role": worker.role, "model": worker.model, "content": "", "worker": worker.name}

    async def _aggregate(self, query: str, responses: list[dict]) -> str:
        """Synthesize multiple agent responses using the local aggregator model."""
        if not self._local_worker or not self._local_worker.is_ready:
            best = max(responses, key=lambda r: len(r.get("content", "")))
            return best.get("content", "")

        agent_section = "\n\n---\n\n".join(
            f"[{r['role'].title()}]: {r['content']}" for r in responses if r.get("content")
        )

        system = AGENT_SYSTEM_PROMPTS["aggregator"]
        user_prompt = f"Original question: {query}\n\nAgent responses:\n\n{agent_section}\n\nSynthesize the single best answer:"

        content, _ = await self._local_worker.complete(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=300,
        )
        return content or responses[0].get("content", "")

    async def process_query(self, query: str, mode: str = "swarm", timeout: float = 12.0) -> dict:
        t0 = time.monotonic()

        query_type = await self._classify_query(query)
        log.info("Query classified as %s", query_type)

        target_roles = QUERY_TYPE_ROLES.get(query_type, QUERY_TYPE_ROLES["COMPLEX_REASONING"])
        responses = await self._fan_out_query(query, target_roles, timeout)

        if not responses:
            return {
                "query": query, "query_type": query_type,
                "agent_responses": [], "final_answer": "No agents responded. Check swarm connectivity.",
                "mode": mode, "total_time_ms": int((time.monotonic() - t0) * 1000),
                "agents_responded": 0,
            }

        if query_type in NEEDS_AGGREGATION and len(responses) > 1:
            final = await self._aggregate(query, responses)
        else:
            final = responses[0].get("content", "")

        total_ms = int((time.monotonic() - t0) * 1000)

        return {
            "query": query, "query_type": query_type,
            "agent_responses": [{"role": r["role"], "model": r["model"], "content": r["content"]} for r in responses],
            "final_answer": final, "mode": mode,
            "total_time_ms": total_ms, "agents_responded": len(responses),
        }

    # ── HTTP handlers ────────────────────────────────────────────────────────

    async def _handle_health(self, request: web.Request) -> web.Response:
        return web.json_response({
            "status": "ok", "role": "aggregator", "platform": "rpi",
            "uptime": metrics.uptime_seconds(),
        })

    async def _handle_metrics(self, request: web.Request) -> web.Response:
        workers = self.discovery.get_all_workers()
        return web.json_response(metrics.full_status(active_workers=len(workers)))

    async def _handle_swarm_status(self, request: web.Request) -> web.Response:
        workers = self.discovery.get_all_workers()
        total_tps = sum(w.last_tps for w in workers)

        return web.json_response({
            "aggregator": {
                "host": socket.gethostname(),
                "platform": "rpi",
                "uptime": metrics.uptime_seconds(),
                "model": Path(self.model_path).stem,
            },
            "workers": [
                {
                    "name": w.name, "role": w.role, "host": w.host, "port": w.port,
                    "status": w.status, "platform": w.platform, "model": w.model,
                    "metrics": {
                        "tokens_per_second": w.last_tps,
                        "thermal_state": w.last_thermal,
                        "battery_percent": w.last_battery,
                        "memory_used_mb": w.last_memory_mb,
                    },
                }
                for w in workers
            ],
            "total_workers": len(workers),
            "total_tps": round(total_tps, 1),
        })

    async def _handle_swarm_query(self, request: web.Request) -> web.Response:
        body = await request.json()
        query = body.get("query", "")
        mode = body.get("mode", "swarm")
        timeout = body.get("timeout_seconds", 12)

        if not query:
            return web.json_response({"error": "missing 'query' field"}, status=400)

        result = await self.process_query(query, mode, timeout)
        return web.json_response(result)

    async def _handle_register(self, request: web.Request) -> web.Response:
        body = await request.json()
        node = self.discovery.add_manual_worker(
            host=body["host"], port=body["port"], role=body.get("role", "drafter"),
            model=body.get("model", "unknown"), platform=body.get("platform", "unknown"),
            ram_mb=body.get("ram_mb", 2048),
        )
        return web.json_response({"status": "registered", "name": node.name})

    async def _handle_completions(self, request: web.Request) -> web.Response:
        """Direct inference on the local aggregator model."""
        body = await request.json()
        if not self._local_worker or not self._local_worker.is_ready:
            return web.json_response({"error": "local model not ready"}, status=503)

        messages = body.get("messages", [])
        content, resp_metrics = await self._local_worker.complete(
            messages=messages,
            max_tokens=body.get("max_tokens", 150),
            temperature=body.get("temperature", 0.7),
        )
        return web.json_response({
            "choices": [{"message": {"role": "assistant", "content": content}}],
            "metrics": resp_metrics,
        })

    async def _handle_dashboard(self, request: web.Request) -> web.Response:
        static_dir = os.path.join(os.path.dirname(__file__), "static")
        index_path = os.path.join(static_dir, "index.html")
        if os.path.exists(index_path):
            return web.FileResponse(index_path)
        return web.Response(text=self._inline_dashboard(), content_type="text/html")

    def _inline_dashboard(self) -> str:
        return """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>REVIVE Mesh</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0a0a0a;color:#ccc;font-family:'SF Mono',monospace;padding:20px}
h1{color:#00ff88;font-size:24px;margin-bottom:20px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:16px;margin-bottom:24px}
.card{background:#111;border:1px solid #222;border-radius:8px;padding:16px}
.card h3{color:#00ff88;font-size:14px;margin-bottom:8px}
.card .role{color:#4a90d9;font-weight:bold}
.card .metric{color:#888;font-size:12px;margin-top:4px}
.status{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px}
.status.idle{background:#00ff88}.status.generating{background:#f5a623}.status.offline{background:#d0021b}
#chat{margin-top:24px}
#messages{height:300px;overflow-y:auto;background:#111;border:1px solid #222;border-radius:8px;padding:12px;margin-bottom:12px}
.msg{margin-bottom:8px;font-size:13px}
.msg.user{color:#4a90d9}.msg.assistant{color:#00ff88}
#input-row{display:flex;gap:8px}
#query{flex:1;background:#111;border:1px solid #333;color:#fff;padding:10px;border-radius:6px;font-family:inherit}
button{background:#00ff88;color:#000;border:none;padding:10px 20px;border-radius:6px;cursor:pointer;font-weight:bold}
</style></head><body>
<h1>R E V I V E — Mesh Aggregator</h1>
<div id="status-bar" style="color:#888;font-size:12px;margin-bottom:16px">Loading...</div>
<div class="grid" id="workers"></div>
<div id="chat">
<div id="messages"></div>
<div id="input-row">
<input id="query" placeholder="Ask the swarm..." onkeydown="if(event.key==='Enter')ask()">
<button onclick="ask()">Send</button>
</div></div>
<script>
const API = '';
async function refresh() {
  try {
    const r = await fetch(API + '/v1/swarm/status');
    const d = await r.json();
    document.getElementById('status-bar').textContent =
      `Workers: ${d.total_workers} | Total TPS: ${d.total_tps} | Aggregator uptime: ${d.aggregator.uptime}s`;
    const grid = document.getElementById('workers');
    grid.innerHTML = d.workers.map(w => `
      <div class="card">
        <h3><span class="status ${w.status}"></span>${w.name}</h3>
        <div class="role">${w.role} — ${w.platform}</div>
        <div class="metric">Model: ${w.model}</div>
        <div class="metric">TPS: ${w.metrics.tokens_per_second} | Thermal: ${w.metrics.thermal_state}</div>
        <div class="metric">Battery: ${w.metrics.battery_percent >= 0 ? w.metrics.battery_percent + '%' : 'N/A'} | RAM: ${w.metrics.memory_used_mb}MB</div>
      </div>`).join('');
  } catch(e) { console.error(e); }
}
async function ask() {
  const input = document.getElementById('query');
  const q = input.value.trim(); if(!q) return;
  input.value = '';
  const msgs = document.getElementById('messages');
  msgs.innerHTML += `<div class="msg user">You: ${q}</div>`;
  msgs.scrollTop = msgs.scrollHeight;
  try {
    const r = await fetch(API + '/v1/swarm/query', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({query: q, mode: 'swarm'})
    });
    const d = await r.json();
    d.agent_responses.forEach(a => {
      msgs.innerHTML += `<div class="msg" style="color:#888">[${a.role}]: ${a.content.substring(0, 200)}${a.content.length > 200 ? '...' : ''}</div>`;
    });
    msgs.innerHTML += `<div class="msg assistant">Swarm: ${d.final_answer}</div>`;
    msgs.innerHTML += `<div class="msg" style="color:#555;font-size:11px">${d.agents_responded} agents | ${d.total_time_ms}ms | ${d.query_type}</div>`;
    msgs.scrollTop = msgs.scrollHeight;
  } catch(e) { msgs.innerHTML += `<div class="msg" style="color:red">Error: ${e}</div>`; }
}
refresh(); setInterval(refresh, 5000);
</script></body></html>"""


# ── Main entry point ─────────────────────────────────────────────────────────

async def main():
    model_path = os.environ.get("REVIVE_MODEL_PATH", os.path.expanduser("~/revive/models/qwen3-1.7b-q4_k_m.gguf"))
    llama_bin = os.environ.get("REVIVE_LLAMA_SERVER", os.path.expanduser("~/revive/llama.cpp/build/bin/llama-server"))
    port = int(os.environ.get("REVIVE_PORT", "9090"))

    if not os.path.exists(model_path):
        log.error("Model not found: %s", model_path)
        log.error("Run setup.sh first, or set REVIVE_MODEL_PATH")
        sys.exit(1)

    if not os.path.exists(llama_bin):
        log.error("llama-server not found: %s", llama_bin)
        log.error("Run setup.sh first, or set REVIVE_LLAMA_SERVER")
        sys.exit(1)

    agg = MeshAggregator(model_path, llama_bin, port)

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(shutdown(agg)))

    await agg.start()

    runner = web.AppRunner(agg._app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()

    log.info("═══════════════════════════════════════")
    log.info("  REVIVE Mesh Aggregator running")
    log.info("  Port: %d", port)
    log.info("  Model: %s", os.path.basename(model_path))
    log.info("  Dashboard: http://0.0.0.0:%d", port)
    log.info("═══════════════════════════════════════")

    await asyncio.Event().wait()


async def shutdown(agg: MeshAggregator):
    log.info("Shutting down...")
    await agg.stop()
    raise SystemExit(0)


if __name__ == "__main__":
    asyncio.run(main())
