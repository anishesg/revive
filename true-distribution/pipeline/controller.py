"""Cluster controller.

Speaks the BMC line protocol to the Arduino (or simulator). The BMC owns
cluster membership, heartbeats, and partition decisions. This module:

  - Connects to the BMC (TCP for sim, serial for real Arduino)
  - Forwards worker REG / HB / UNREG events upstream
  - Receives PARTITION / DEAD / ALIVE / STATE events from the BMC
  - Exposes the current partition to the coordinator

The coordinator NEVER decides partitioning locally. It always asks the
controller, which reflects what the BMC most recently announced. If the
BMC goes silent, the coordinator can't schedule new inferences — by design.
"""
from __future__ import annotations
import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Optional

from .bmc_protocol import (
    HEARTBEAT_INTERVAL_MS,
    encode_hb,
    encode_reg,
    parse_event,
)
from .partitioner import WorkerProfile

log = logging.getLogger("controller")


@dataclass
class ClusterView:
    """A snapshot of what the BMC most recently told us."""
    state: str = "down"
    partition: list[tuple[str, int, int]] = field(default_factory=list)
    dead_workers: set[str] = field(default_factory=set)
    version: int = 0  # bumps every BMC update so coordinator can detect changes


class BMCLink:
    """A bidirectional line-oriented stream to the BMC. Concrete transports
    (TCPSimLink, SerialLink) subclass this."""

    async def connect(self) -> None: raise NotImplementedError
    async def close(self) -> None: raise NotImplementedError
    async def send(self, line: str) -> None: raise NotImplementedError
    def lines(self) -> "asyncio.Queue[str]":
        """Queue of lines received from the BMC."""
        raise NotImplementedError


class TCPSimLink(BMCLink):
    """Connects to the BMC simulator via TCP. Used for local dev. The real
    hardware path uses SerialLink with the exact same API."""
    def __init__(self, host: str = "127.0.0.1", port: int = 45555):
        self.host = host
        self.port = port
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._read_task: Optional[asyncio.Task] = None

    async def connect(self):
        self._reader, self._writer = await asyncio.open_connection(self.host, self.port)
        self._read_task = asyncio.create_task(self._reader_loop())
        log.info(f"connected to BMC sim at {self.host}:{self.port}")

    async def _reader_loop(self):
        assert self._reader is not None
        try:
            while True:
                line = await self._reader.readline()
                if not line:
                    break
                await self._queue.put(line.decode("utf-8", errors="replace").rstrip())
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.warning(f"bmc reader stopped: {e}")

    async def send(self, line: str):
        if not self._writer:
            raise RuntimeError("not connected")
        self._writer.write((line + "\n").encode("utf-8"))
        await self._writer.drain()

    async def close(self):
        if self._read_task:
            self._read_task.cancel()
            try:
                await self._read_task
            except asyncio.CancelledError:
                pass
        if self._writer:
            self._writer.close()
            try:
                await self._writer.wait_closed()
            except Exception:
                pass

    def lines(self) -> asyncio.Queue:
        return self._queue


class SerialLink(BMCLink):
    """Real Arduino over USB serial. Pulls pyserial-asyncio only when this
    class is instantiated, so the simulator path works without the extra
    dependency installed.

    On macOS, plugging in an Arduino Uno gives /dev/tty.usbmodem* and/or
    /dev/cu.usbmodem*. Prefer cu.* (non-blocking open). Linux: /dev/ttyACM0.
    Use find_arduino_device() to autodetect.
    """
    def __init__(self, device: str, baud: int = 115200, boot_delay_s: float = 2.0):
        self.device = device
        self.baud = baud
        self.boot_delay_s = boot_delay_s
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._reader = None
        self._writer = None
        self._task = None

    async def connect(self):
        import serial_asyncio  # lazy import
        self._reader, self._writer = await serial_asyncio.open_serial_connection(
            url=self.device, baudrate=self.baud
        )
        # Arduino Uno resets when the host opens the serial port (DTR toggle).
        # Give the firmware its bootloader + setup() time before we talk to it.
        log.info(f"opened {self.device} @ {self.baud} baud; waiting {self.boot_delay_s}s for Arduino reset")
        await asyncio.sleep(self.boot_delay_s)
        self._task = asyncio.create_task(self._reader_loop())
        log.info(f"connected to Arduino at {self.device}")

    async def _reader_loop(self):
        assert self._reader is not None
        try:
            while True:
                line = await self._reader.readline()
                if not line:
                    break
                await self._queue.put(line.decode("utf-8", errors="replace").rstrip())
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.warning(f"serial reader stopped: {e}")

    async def send(self, line: str):
        if not self._writer:
            raise RuntimeError("not connected")
        self._writer.write((line + "\n").encode("utf-8"))
        await self._writer.drain()

    async def close(self):
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        if self._writer:
            self._writer.close()

    def lines(self) -> asyncio.Queue:
        return self._queue


def find_arduino_device() -> Optional[str]:
    """Scan attached serial ports for something that looks like an Arduino Uno.
    Returns a /dev/... path or None. Requires pyserial (not pyserial-asyncio)."""
    try:
        from serial.tools import list_ports
    except ImportError:
        return None

    # Prefer matches with VID/PID of the classic Uno or common clones
    arduino_vids = {
        0x2341,  # Arduino LLC / Arduino SA
        0x2A03,  # Arduino SRL
        0x1A86,  # QinHeng CH340 (clone Unos)
        0x0403,  # FTDI (older clones)
        0x10C4,  # SiLabs CP210x (some clones)
    }
    candidates = []
    for port in list_ports.comports():
        if port.vid in arduino_vids:
            candidates.append(port.device)
        elif "usbmodem" in (port.device or "") or "ttyACM" in (port.device or ""):
            candidates.append(port.device)

    if not candidates:
        return None
    # On macOS there are two entries per device (tty.* and cu.*); cu.* is
    # preferred because it won't block on modem control lines.
    cu = [c for c in candidates if "/cu." in c]
    return (cu or candidates)[0]


class ClusterController:
    """Host-side controller. Keeps a ClusterView synced with the BMC.

    Public API (called by the coordinator):
      - await register_worker(profile)
      - await unregister_worker(id)
      - await heartbeat(id, tps, temp_c)
      - await set_num_layers(n)
      - await mark_inference(active: bool)
      - await inject_failure(id)         # for chaos demos
      - view: ClusterView                # latest snapshot
      - on_partition_change: optional callback(ClusterView)
    """
    def __init__(self, link: BMCLink):
        self.link = link
        self.view = ClusterView()
        self.on_partition_change: Optional[Callable[[ClusterView], Awaitable[None]]] = None
        self._event_log: list[tuple[float, str, str]] = []  # (ts, direction, line)
        self._event_log_max = 500
        self._reader_task: Optional[asyncio.Task] = None
        self._hb_task: Optional[asyncio.Task] = None
        self._ready = asyncio.Event()

    # ─── lifecycle ──────────────────────────────────────────────────────

    async def start(self):
        await self.link.connect()
        self._reader_task = asyncio.create_task(self._reader_loop())
        # Wait up to 2s for the BMC READY line
        try:
            await asyncio.wait_for(self._ready.wait(), timeout=2.0)
        except asyncio.TimeoutError:
            log.warning("BMC did not say READY within 2s — continuing anyway")
        # Kick it with a HELLO in case we missed the boot banner
        await self._send("HELLO")

    async def stop(self):
        if self._hb_task:
            self._hb_task.cancel()
        if self._reader_task:
            self._reader_task.cancel()
        await self.link.close()

    # ─── command API ───────────────────────────────────────────────────

    async def set_num_layers(self, n: int):
        await self._send(f"MODEL {n}")

    async def register_worker(self, profile: WorkerProfile):
        await self._send(encode_reg(profile.id, profile.capability_score, profile.ram_mb))

    async def unregister_worker(self, worker_id: str):
        await self._send(f"UNREG {worker_id}")

    async def heartbeat(self, worker_id: str, tps: float = 0, temp_c: int = 0):
        await self._send(encode_hb(worker_id, tps, temp_c))

    async def mark_inference(self, active: bool):
        await self._send(f"INFER {'START' if active else 'END'}")

    async def inject_failure(self, worker_id: str):
        """Chaos engineering: tell BMC to consider this worker dead."""
        await self._send(f"FAIL {worker_id}")

    async def query(self):
        await self._send("QUERY")

    async def reset(self):
        await self._send("RESET")

    # ─── internals ─────────────────────────────────────────────────────

    async def _send(self, line: str):
        self._log("→ ", line)
        await self.link.send(line)

    async def _reader_loop(self):
        queue = self.link.lines()
        while True:
            line = await queue.get()
            self._log("← ", line)
            await self._handle_line(line)

    async def _handle_line(self, line: str):
        cmd, rest = parse_event(line)
        if cmd == "READY":
            self._ready.set()
        elif cmd == "STATE":
            self.view.state = rest.strip()
            self.view.version += 1
            await self._notify_change()
        elif cmd == "PARTITION":
            assignments = []
            for tok in rest.split():
                try:
                    wid, s, e = tok.split(":")
                    assignments.append((wid, int(s), int(e)))
                except ValueError:
                    log.warning(f"bad PARTITION token: {tok}")
            self.view.partition = assignments
            self.view.dead_workers -= {a[0] for a in assignments}
            self.view.version += 1
            await self._notify_change()
        elif cmd == "DEAD":
            self.view.dead_workers.add(rest.strip())
            self.view.version += 1
            await self._notify_change()
        elif cmd == "ALIVE":
            self.view.dead_workers.discard(rest.strip())
            self.view.version += 1
            await self._notify_change()
        elif cmd == "INFO":
            log.info(f"BMC: {rest}")
        elif cmd == "ACK":
            pass  # quiet
        elif cmd:
            log.debug(f"unhandled BMC event: {line}")

    async def _notify_change(self):
        cb = self.on_partition_change
        if cb:
            try:
                await cb(self.view)
            except Exception as e:
                log.exception(f"partition-change callback failed: {e}")

    def _log(self, direction: str, line: str):
        ts = time.time()
        self._event_log.append((ts, direction, line))
        if len(self._event_log) > self._event_log_max:
            self._event_log = self._event_log[-self._event_log_max :]

    def recent_events(self, n: int = 50) -> list[tuple[float, str, str]]:
        return self._event_log[-n:]
