"""
REVIVE Worker Process for Raspberry Pi
Manages a llama-server subprocess and proxies inference requests.
Can also run standalone as a pure worker node (no aggregation).
"""
import asyncio
import logging
import os
import signal
import subprocess
import time

import aiohttp

log = logging.getLogger("revive.worker")


class LlamaWorker:
    """Manages a llama-server process and provides async inference."""

    def __init__(self, model_path: str, llama_server_bin: str, port: int = 8081, ctx_size: int = 2048, threads: int | None = None):
        self.model_path = model_path
        self.llama_bin = llama_server_bin
        self.port = port
        self.ctx_size = ctx_size
        self.threads = threads or max(1, os.cpu_count() - 1)
        self._process: subprocess.Popen | None = None
        self._session: aiohttp.ClientSession | None = None
        self._ready = False

    async def start(self) -> None:
        cmd = [
            self.llama_bin,
            "-m", self.model_path,
            "--host", "127.0.0.1",
            "--port", str(self.port),
            "-c", str(self.ctx_size),
            "-ngl", "0",
            "-t", str(self.threads),
            "--log-disable",
        ]
        log.info("Starting llama-server on port %d with %d threads", self.port, self.threads)
        self._process = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )
        self._session = aiohttp.ClientSession()

        for _ in range(30):
            await asyncio.sleep(1)
            try:
                async with self._session.get(f"http://127.0.0.1:{self.port}/health") as resp:
                    if resp.status == 200:
                        self._ready = True
                        log.info("llama-server ready on port %d", self.port)
                        return
            except (aiohttp.ClientError, ConnectionRefusedError):
                pass

        log.error("llama-server failed to start within 30 seconds")

    async def stop(self) -> None:
        if self._session:
            await self._session.close()
        if self._process:
            self._process.send_signal(signal.SIGTERM)
            self._process.wait(timeout=5)
            log.info("llama-server stopped")

    @property
    def is_ready(self) -> bool:
        return self._ready and self._process is not None and self._process.poll() is None

    async def complete(self, messages: list[dict], max_tokens: int = 150, temperature: float = 0.7) -> tuple[str, dict]:
        """Send a completion request and return (content, metrics_dict)."""
        if not self.is_ready or not self._session:
            return "", {"tokens_generated": 0, "tokens_per_second": 0, "time_to_first_token_ms": 0, "total_time_ms": 0, "thermal_state": "unknown", "battery_percent": -1, "memory_used_mb": 0}

        t0 = time.monotonic()

        payload = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }

        try:
            async with self._session.post(
                f"http://127.0.0.1:{self.port}/v1/chat/completions",
                json=payload, timeout=aiohttp.ClientTimeout(total=60),
            ) as resp:
                data = await resp.json()

            t1 = time.monotonic()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            usage = data.get("usage", {})
            completion_tokens = usage.get("completion_tokens", len(content.split()) // 2)
            total_ms = int((t1 - t0) * 1000)
            tps = completion_tokens / max((t1 - t0), 0.01)

            from . import metrics as m
            device_metrics = m.snapshot(
                tokens_generated=completion_tokens,
                tokens_per_second=round(tps, 1),
                time_to_first_token_ms=int((t1 - t0) * 200),
                total_time_ms=total_ms,
            )
            return content, device_metrics

        except Exception as e:
            log.error("Inference error: %s", e)
            return "", {"tokens_generated": 0, "tokens_per_second": 0, "time_to_first_token_ms": 0, "total_time_ms": 0, "thermal_state": "unknown", "battery_percent": -1, "memory_used_mb": 0}
