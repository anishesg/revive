#!/usr/bin/env python3
"""
REVIVE MacBook CLI — Worker or Coordinator mode.
Uses llama.cpp with Metal GPU acceleration on Apple Silicon.

Usage:
    # Worker mode (default):
    python3 revive-cli.py --mode worker --role reasoner

    # Coordinator mode (routes + aggregates):
    python3 revive-cli.py --mode coordinator

    # Interactive chat with the swarm (coordinator talks to all workers):
    python3 revive-cli.py --mode coordinator --interactive

Environment:
    REVIVE_MODEL_PATH   Path to GGUF model
    REVIVE_LLAMA_SERVER Path to llama-server binary
"""
import argparse
import asyncio
import json
import logging
import os
import platform
import signal
import socket
import subprocess
import sys
import time

import aiohttp
from aiohttp import web

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rpi"))
from discovery import ServiceAdvertiser, SwarmDiscovery, WorkerNode

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("revive.macos")

AGENT_SYSTEM_PROMPTS = {
    "reasoner": "You are a rigorous analytical thinker. Think step by step. Show your reasoning chain explicitly.",
    "writer": "You are an eloquent communicator. Write clear, well-structured, engaging responses.",
    "concise": "You are a master of brevity. Answer in as few words as possible while being complete and accurate.",
    "critic": "You are a devil's advocate. Identify flaws, edge cases, counterarguments.",
    "factchecker": "You are a fact-checker. Focus only on verifiable, accurate information.",
    "drafter": "You are a quick-response generator. Produce a fast first-pass answer.",
    "spotter": "Classify the query into EXACTLY one category: SIMPLE_FACT, COMPLEX_REASONING, CREATIVE, CODE, MATH, OPINION",
    "aggregator": "You are the Aggregator of a distributed AI swarm. Synthesize the single best answer from multiple agent responses.",
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


def get_metal_gpu_layers() -> int:
    """Determine GPU layers based on Apple Silicon availability."""
    try:
        result = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True)
        if "Apple" in result.stdout:
            return 99
    except Exception:
        pass
    return 0


def get_system_ram_mb() -> int:
    try:
        result = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True)
        return int(result.stdout.strip()) // (1024 * 1024)
    except Exception:
        return 8192


def get_cpu_temp() -> float:
    """macOS doesn't expose CPU temp easily. Return -1 if unavailable."""
    return -1.0


def get_thermal_state() -> str:
    try:
        result = subprocess.run(["pmset", "-g", "therm"], capture_output=True, text=True)
        if "CPU_Scheduler_Limit" in result.stdout:
            for line in result.stdout.split("\n"):
                if "CPU_Scheduler_Limit" in line:
                    val = int(line.split("=")[-1].strip())
                    if val < 50:
                        return "critical"
                    if val < 80:
                        return "serious"
                    if val < 100:
                        return "fair"
        return "nominal"
    except Exception:
        return "unknown"


def get_battery_percent() -> int:
    try:
        result = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True)
        for line in result.stdout.split("\n"):
            if "%" in line:
                pct = line.split("\t")[1].split("%")[0] if "\t" in line else ""
                if pct.isdigit():
                    return int(pct)
        return -1
    except Exception:
        return -1


def get_memory_used_mb() -> int:
    try:
        import psutil
        return int(psutil.virtual_memory().used / (1024 * 1024))
    except ImportError:
        return 0


class MacWorker:
    """Manages a local llama-server with Metal acceleration."""

    def __init__(self, model_path: str, llama_bin: str, port: int = 8081, ctx_size: int = 4096):
        self.model_path = model_path
        self.llama_bin = llama_bin
        self.port = port
        self.ctx_size = ctx_size
        self._process: subprocess.Popen | None = None
        self._session: aiohttp.ClientSession | None = None
        self._ready = False

    async def start(self):
        gpu_layers = get_metal_gpu_layers()
        threads = max(1, os.cpu_count() - 2)

        cmd = [
            self.llama_bin,
            "-m", self.model_path,
            "--host", "0.0.0.0" if gpu_layers == 0 else "127.0.0.1",
            "--port", str(self.port),
            "-c", str(self.ctx_size),
            "-ngl", str(gpu_layers),
            "-t", str(threads),
            "--log-disable",
        ]
        log.info("Starting llama-server: port=%d gpu_layers=%d threads=%d", self.port, gpu_layers, threads)
        self._process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        self._session = aiohttp.ClientSession()

        for _ in range(30):
            await asyncio.sleep(1)
            try:
                async with self._session.get(f"http://127.0.0.1:{self.port}/health") as resp:
                    if resp.status == 200:
                        self._ready = True
                        log.info("llama-server ready (Metal: %s)", "yes" if gpu_layers > 0 else "no")
                        return
            except (aiohttp.ClientError, ConnectionRefusedError):
                pass
        log.error("llama-server failed to start")

    async def stop(self):
        if self._session:
            await self._session.close()
        if self._process:
            self._process.send_signal(signal.SIGTERM)
            self._process.wait(timeout=5)

    @property
    def is_ready(self) -> bool:
        return self._ready and self._process is not None and self._process.poll() is None

    async def complete(self, messages: list[dict], max_tokens: int = 150, temperature: float = 0.7) -> tuple[str, dict]:
        if not self.is_ready or not self._session:
            return "", {}

        t0 = time.monotonic()
        try:
            async with self._session.post(
                f"http://127.0.0.1:{self.port}/v1/chat/completions",
                json={"messages": messages, "max_tokens": max_tokens, "temperature": temperature, "stream": False},
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                data = await resp.json()

            t1 = time.monotonic()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            usage = data.get("usage", {})
            tokens = usage.get("completion_tokens", max(1, len(content.split()) // 2))
            total_ms = int((t1 - t0) * 1000)
            tps = tokens / max(t1 - t0, 0.01)

            metrics = {
                "tokens_generated": tokens,
                "tokens_per_second": round(tps, 1),
                "time_to_first_token_ms": int((t1 - t0) * 100),
                "total_time_ms": total_ms,
                "thermal_state": get_thermal_state(),
                "battery_percent": get_battery_percent(),
                "memory_used_mb": get_memory_used_mb(),
            }
            return content, metrics
        except Exception as e:
            log.error("Inference error: %s", e)
            return "", {}


class MacCLI:
    """Main CLI application. Runs as worker or coordinator."""

    def __init__(self, args):
        self.args = args
        self.model_path = args.model or os.environ.get("REVIVE_MODEL_PATH", "")
        self.llama_bin = args.llama_server or os.environ.get("REVIVE_LLAMA_SERVER", "")
        self.port = args.port
        self.role = args.role
        self.mode = args.mode

        self.worker: MacWorker | None = None
        self.discovery = SwarmDiscovery()
        self.advertiser: ServiceAdvertiser | None = None
        self._http_session: aiohttp.ClientSession | None = None
        self._start_time = time.time()

    async def run(self):
        if not self.model_path or not os.path.exists(self.model_path):
            log.error("Model not found: %s", self.model_path)
            log.error("Set --model or REVIVE_MODEL_PATH")
            sys.exit(1)
        if not self.llama_bin or not os.path.exists(self.llama_bin):
            log.error("llama-server not found: %s", self.llama_bin)
            log.error("Set --llama-server or REVIVE_LLAMA_SERVER")
            sys.exit(1)

        self._http_session = aiohttp.ClientSession()

        self.worker = MacWorker(self.model_path, self.llama_bin, port=self.port + 1, ctx_size=args.ctx_size)
        await self.worker.start()

        ram_mb = get_system_ram_mb()
        caps = "metal" if get_metal_gpu_layers() > 0 else "cpu"

        self.advertiser = ServiceAdvertiser(
            role=self.role if self.mode == "worker" else "aggregator",
            model=os.path.basename(self.model_path).replace(".gguf", ""),
            port=self.port, platform="macos", ram_mb=ram_mb, caps=caps,
        )
        self.advertiser.start()
        self.discovery.start()

        app = web.Application()
        app.router.add_get("/health", self._handle_health)
        app.router.add_post("/v1/chat/completions", self._handle_completions)
        app.router.add_get("/v1/swarm/status", self._handle_status)
        app.router.add_post("/v1/swarm/query", self._handle_swarm_query)
        app.router.add_post("/v1/swarm/register", self._handle_register)

        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "0.0.0.0", self.port)
        await site.start()

        log.info("═══════════════════════════════════════")
        log.info("  REVIVE MacBook %s", self.mode.upper())
        log.info("  Role: %s | Port: %d", self.role, self.port)
        log.info("  Model: %s", os.path.basename(self.model_path))
        log.info("  Metal: %s | RAM: %dMB", caps, ram_mb)
        log.info("═══════════════════════════════════════")

        if self.args.interactive and self.mode == "coordinator":
            await asyncio.sleep(3)
            await self._interactive_loop()
        else:
            await asyncio.Event().wait()

    async def _interactive_loop(self):
        print("\n\033[92mREVIVE Swarm CLI — type your query (Ctrl+C to exit)\033[0m\n")
        while True:
            try:
                query = await asyncio.get_event_loop().run_in_executor(None, lambda: input("\033[94m> \033[0m"))
                if not query.strip():
                    continue
                if query.strip().lower() in ("quit", "exit"):
                    break
                if query.strip().lower() == "status":
                    workers = self.discovery.get_all_workers()
                    print(f"\n  Workers: {len(workers)}")
                    for w in workers:
                        print(f"    {w.role:12} | {w.platform:8} | {w.host}:{w.port} | {w.status}")
                    print()
                    continue

                print("\033[90mQuerying swarm...\033[0m")
                result = await self._process_swarm_query(query)

                for resp in result.get("agent_responses", []):
                    print(f"\033[93m[{resp['role']}]\033[0m {resp['content'][:300]}")
                    print()

                print(f"\033[92m[SWARM]\033[0m {result['final_answer']}")
                print(f"\033[90m({result['agents_responded']} agents, {result['total_time_ms']}ms, {result['query_type']})\033[0m\n")

            except (EOFError, KeyboardInterrupt):
                break

    async def _process_swarm_query(self, query: str, timeout: float = 15.0) -> dict:
        t0 = time.monotonic()
        query_type = "COMPLEX_REASONING"
        target_roles = QUERY_TYPE_ROLES.get(query_type, QUERY_TYPE_ROLES["COMPLEX_REASONING"])

        tasks = []
        for role in target_roles:
            for w in self.discovery.get_workers_by_role(role):
                tasks.append(self._query_worker(w, query, role, timeout))

        responses = []
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            responses = [r for r in results if isinstance(r, dict) and r.get("content")]

        if len(responses) > 1 and self.worker and self.worker.is_ready:
            agent_section = "\n\n---\n\n".join(f"[{r['role'].title()}]: {r['content']}" for r in responses)
            final, _ = await self.worker.complete(
                messages=[
                    {"role": "system", "content": AGENT_SYSTEM_PROMPTS["aggregator"]},
                    {"role": "user", "content": f"Original question: {query}\n\nAgent responses:\n\n{agent_section}\n\nSynthesize the single best answer:"},
                ],
                max_tokens=300,
            )
        elif responses:
            final = responses[0]["content"]
        else:
            if self.worker and self.worker.is_ready:
                final, _ = await self.worker.complete(
                    messages=[{"role": "user", "content": query}], max_tokens=300,
                )
            else:
                final = "No workers available and local model not ready."

        return {
            "query": query, "query_type": query_type,
            "agent_responses": responses, "final_answer": final,
            "mode": "swarm", "total_time_ms": int((time.monotonic() - t0) * 1000),
            "agents_responded": len(responses),
        }

    async def _query_worker(self, worker: WorkerNode, query: str, role: str, timeout: float) -> dict:
        system = AGENT_SYSTEM_PROMPTS.get(role, "")
        messages = [{"role": "system", "content": system}, {"role": "user", "content": query}]

        try:
            async with self._http_session.post(
                f"{worker.url}/v1/chat/completions",
                json={"messages": messages, "max_tokens": 150, "temperature": 0.7, "stream": False},
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as resp:
                data = await resp.json()

            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            return {"role": role, "model": worker.model, "content": content}
        except Exception as e:
            log.debug("Worker %s failed: %s", worker.name, e)
            return {"role": role, "model": worker.model, "content": ""}

    # ── HTTP handlers ────────────────────────────────────────────────────

    async def _handle_health(self, request: web.Request) -> web.Response:
        return web.json_response({
            "status": "ok", "role": self.role, "platform": "macos",
            "uptime": int(time.time() - self._start_time),
        })

    async def _handle_completions(self, request: web.Request) -> web.Response:
        body = await request.json()
        if not self.worker or not self.worker.is_ready:
            return web.json_response({"error": "model not ready"}, status=503)
        content, met = await self.worker.complete(
            messages=body.get("messages", []),
            max_tokens=body.get("max_tokens", 150),
            temperature=body.get("temperature", 0.7),
        )
        return web.json_response({"choices": [{"message": {"role": "assistant", "content": content}}], "metrics": met})

    async def _handle_status(self, request: web.Request) -> web.Response:
        workers = self.discovery.get_all_workers()
        return web.json_response({
            "aggregator": {"host": socket.gethostname(), "platform": "macos", "uptime": int(time.time() - self._start_time)},
            "workers": [{"name": w.name, "role": w.role, "host": w.host, "port": w.port, "status": w.status, "platform": w.platform} for w in workers],
            "total_workers": len(workers),
        })

    async def _handle_swarm_query(self, request: web.Request) -> web.Response:
        body = await request.json()
        result = await self._process_swarm_query(body.get("query", ""), body.get("timeout_seconds", 15))
        return web.json_response(result)

    async def _handle_register(self, request: web.Request) -> web.Response:
        body = await request.json()
        self.discovery.add_manual_worker(
            host=body["host"], port=body["port"], role=body.get("role", "drafter"),
            model=body.get("model", "unknown"), platform=body.get("platform", "unknown"),
        )
        return web.json_response({"status": "registered"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="REVIVE MacBook CLI")
    parser.add_argument("--mode", choices=["worker", "coordinator"], default="worker")
    parser.add_argument("--role", default="reasoner", help="Agent role (worker mode)")
    parser.add_argument("--port", type=int, default=8080, help="HTTP port")
    parser.add_argument("--model", default=None, help="Path to GGUF model")
    parser.add_argument("--llama-server", default=None, help="Path to llama-server binary")
    parser.add_argument("--ctx-size", type=int, default=4096, help="Context size")
    parser.add_argument("--interactive", action="store_true", help="Interactive chat mode (coordinator)")
    args = parser.parse_args()

    cli = MacCLI(args)
    try:
        asyncio.run(cli.run())
    except (KeyboardInterrupt, SystemExit):
        log.info("Shutting down")
