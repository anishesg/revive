"""In-process Python simulator of the Arduino BMC firmware.

Speaks the EXACT same line protocol as arduino/revive_bmc.ino. When the
real Arduino is plugged in we swap this for a serial link — nothing else
in the system changes.

Run standalone:

    python -m pipeline.bmc_sim --port 45555

Then the controller connects to localhost:45555 and speaks the protocol
over TCP instead of serial.
"""
from __future__ import annotations
import argparse
import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from .partitioner import WorkerProfile, greedy_partition, validate_partition
from .bmc_protocol import (
    HEARTBEAT_TIMEOUT_MS,
    PROTOCOL_VERSION,
    parse_event,
)

log = logging.getLogger("bmc-sim")


@dataclass
class SimWorker:
    id: str
    score_x100: int
    ram_mb: int
    last_hb_ms: float
    last_tps_x10: int = 0
    last_temp_c: int = 0
    alive: bool = True
    ever_seen: bool = True


class BMCCore:
    """The actual firmware logic, running in Python. Mirrors revive_bmc.ino
    function-for-function so we can cross-validate later."""

    MAX_WORKERS = 6

    def __init__(self, writer: asyncio.StreamWriter, loop: asyncio.AbstractEventLoop):
        self.writer = writer
        self.loop = loop
        self.workers: list[SimWorker] = []
        self.num_layers = 0
        self.cluster_state = "down"
        self.infer_active = False
        self._boot()

    def _now_ms(self) -> float:
        return self.loop.time() * 1000

    def _emit(self, line: str):
        self.writer.write((line + "\n").encode("utf-8"))

    def _boot(self):
        self._emit(f"READY {PROTOCOL_VERSION}")

    def _set_state(self, new_state: str):
        if new_state == self.cluster_state:
            return
        self.cluster_state = new_state
        self._emit(f"STATE {new_state}")

    def _find(self, wid: str) -> Optional[SimWorker]:
        for w in self.workers:
            if w.id == wid:
                return w
        return None

    def _repartition(self):
        alive = [w for w in self.workers if w.alive]
        if not alive or self.num_layers == 0:
            self._set_state("down")
            return
        self._set_state("degraded" if len(alive) < len(self.workers) else "healthy")

        # Mirror the Arduino's integer-only greedy, just for fidelity.
        total = sum(w.score_x100 for w in alive)
        cur = 0
        remaining = len(alive)
        assignments: list[tuple[str, int, int]] = []
        for w in alive:
            if remaining == 1:
                end = self.num_layers
            else:
                share = (self.num_layers * w.score_x100) // total
                if share < 1:
                    share = 1
                end = cur + share
                max_end = self.num_layers - (remaining - 1)
                if end > max_end:
                    end = max_end
            assignments.append((w.id, cur, end))
            cur = end
            remaining -= 1

        try:
            validate_partition(assignments, self.num_layers)
        except ValueError as e:
            self._emit(f"INFO partition invariant broken: {e}")
            return
        line = "PARTITION " + " ".join(f"{i}:{s}:{e}" for (i, s, e) in assignments)
        self._emit(line)

    # ─── command handlers ──────────────────────────────────────────────
    def on_hello(self, rest: str):
        self._emit(f"READY {PROTOCOL_VERSION}")

    def on_reg(self, rest: str):
        parts = rest.split()
        if len(parts) != 3:
            self._emit("INFO bad REG")
            return
        wid, score_s, ram_s = parts
        score = int(score_s)
        ram = int(ram_s)
        existing = self._find(wid)
        is_new = existing is None
        if is_new:
            if len(self.workers) >= self.MAX_WORKERS:
                self._emit("INFO worker table full")
                return
            self.workers.append(SimWorker(id=wid, score_x100=score, ram_mb=ram,
                                           last_hb_ms=self._now_ms()))
        else:
            existing.score_x100 = score
            existing.ram_mb = ram
            existing.last_hb_ms = self._now_ms()
            existing.alive = True
            existing.ever_seen = True
        if is_new or self.num_layers > 0:
            self._repartition()
        self._emit(f"ACK REG {wid}")

    def on_unreg(self, rest: str):
        wid = rest.strip()
        w = self._find(wid)
        if not w:
            return
        w.alive = False
        self._repartition()
        self._emit(f"ACK UNREG {wid}")

    def on_hb(self, rest: str):
        parts = rest.split()
        if not parts:
            return
        wid = parts[0]
        w = self._find(wid)
        if not w:
            return
        was_dead = not w.alive
        w.last_hb_ms = self._now_ms()
        w.alive = True
        if len(parts) > 1: w.last_tps_x10 = int(parts[1])
        if len(parts) > 2: w.last_temp_c = int(parts[2])
        if was_dead:
            self._emit(f"ALIVE {wid}")
            self._repartition()

    def on_model(self, rest: str):
        try:
            self.num_layers = int(rest.strip())
        except ValueError:
            self._emit("INFO bad MODEL")
            return
        if self.workers:
            self._repartition()
        self._emit(f"ACK MODEL {self.num_layers}")

    def on_infer(self, rest: str):
        self.infer_active = rest.strip().startswith("START")
        self._emit(f"ACK INFER {'START' if self.infer_active else 'END'}")

    def on_fail(self, rest: str):
        wid = rest.strip()
        w = self._find(wid)
        if not w:
            return
        w.alive = False
        self._emit(f"DEAD {wid}")
        self._repartition()

    def on_query(self, rest: str):
        self._emit(f"STATE {self.cluster_state}")
        if self.workers and self.num_layers:
            self._repartition()

    def on_reset(self, rest: str):
        self.workers.clear()
        self.num_layers = 0
        self.infer_active = False
        self._set_state("down")
        self._emit("INFO cluster state cleared")

    def handle_line(self, line: str):
        cmd, rest = parse_event(line)
        if not cmd:
            return
        handler = getattr(self, f"on_{cmd.lower()}", None)
        if handler:
            handler(rest)
        else:
            self._emit(f"INFO unknown cmd {cmd}")

    def health_check(self):
        now = self._now_ms()
        any_change = False
        for w in self.workers:
            if w.alive and (now - w.last_hb_ms > HEARTBEAT_TIMEOUT_MS):
                w.alive = False
                self._emit(f"DEAD {w.id}")
                any_change = True
        if any_change:
            self._repartition()


# ─── TCP transport (one connection at a time — emulating the single USB serial link) ──

async def _handle_conn(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
    loop = asyncio.get_running_loop()
    core = BMCCore(writer, loop)
    stop = asyncio.Event()

    async def health_loop():
        while not stop.is_set():
            core.health_check()
            try:
                await asyncio.wait_for(stop.wait(), timeout=1.0)
            except asyncio.TimeoutError:
                pass

    hc_task = asyncio.create_task(health_loop())
    try:
        while True:
            line = await reader.readline()
            if not line:
                break
            core.handle_line(line.decode("utf-8", errors="replace"))
            await writer.drain()
    finally:
        stop.set()
        await hc_task
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass
        log.info("connection closed")


async def serve(host: str, port: int):
    server = await asyncio.start_server(_handle_conn, host, port)
    addrs = ", ".join(str(s.getsockname()) for s in server.sockets)
    log.info(f"BMC simulator listening on {addrs}")
    async with server:
        await server.serve_forever()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=45555)
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s [%(name)s] %(message)s")
    asyncio.run(serve(args.host, args.port))


if __name__ == "__main__":
    main()
