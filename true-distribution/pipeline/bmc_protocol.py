"""Wire protocol between the host (Mac) and the Arduino BMC.

Line-oriented text protocol, one message per line. Simple so it fits in
2 KB of Arduino RAM and is trivial to debug by watching the serial port.

─── Host → BMC (commands) ─────────────────────────────────────────────────

  HELLO                              handshake; BMC should respond READY
  MODEL <num_layers>                 set total layer count for this cluster
  REG <id> <score_x100> <ram_mb>     register a worker. score_x100 is
                                     capability relative to baseline × 100
                                     (so 150 = 1.5×), integer to avoid floats
  UNREG <id>                         voluntary unregister
  HB <id> <tps_x10> <temp_c>         heartbeat from a worker
  INFER <seq_id>                     mark inference event (for LED state)
  FAIL <id>                          inject failure (testing/chaos)
  QUERY                              ask BMC for current partition
  RESET                              drop all cluster state

─── BMC → Host (events) ───────────────────────────────────────────────────

  READY <version>                    boot handshake
  INFO <text>                        log line (for dashboard)
  STATE <healthy|degraded|down>      cluster health state transition
  PARTITION <id>:<start>:<end> ...   new partition assignment
  DEAD <id>                          BMC declared this worker dead
  ALIVE <id>                         worker came back (HB after being dead)
  ACK <cmd>                          generic acknowledgement

Worker IDs are short ASCII strings (max 8 chars): phone1, pi, ipad, etc.
All numbers are integers. Scores are ×100 fixed point, tps is ×10.

Arduino state held (fits easily in 2 KB):
  workers[MAX_WORKERS=6]:
    id[9] (8 + null),
    score_x100 (uint16),
    ram_mb (uint16),
    last_hb_ms (uint32),
    last_tps_x10 (uint16),
    last_temp_c (int8),
    flags (uint8: alive, ever_seen, is_first, is_last)
  num_layers (uint16)
  num_active (uint8)
  heartbeat_timeout_ms = 6000 (2 missed @ 3s cadence)
"""
from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Optional


PROTOCOL_VERSION = 1
HEARTBEAT_INTERVAL_MS = 3000
HEARTBEAT_TIMEOUT_MS = 6000


@dataclass
class Partition:
    assignments: list[tuple[str, int, int]]  # [(id, start, end), ...]

    def to_line(self) -> str:
        parts = " ".join(f"{w}:{s}:{e}" for (w, s, e) in self.assignments)
        return f"PARTITION {parts}"

    @classmethod
    def parse(cls, line: str) -> "Partition":
        body = line[len("PARTITION "):]
        out = []
        for tok in body.split():
            wid, s, e = tok.split(":")
            out.append((wid, int(s), int(e)))
        return cls(assignments=out)


def encode_reg(worker_id: str, score: float, ram_mb: int) -> str:
    return f"REG {worker_id} {int(score * 100)} {ram_mb}"


def encode_hb(worker_id: str, tps: float, temp_c: int) -> str:
    return f"HB {worker_id} {int(tps * 10)} {int(temp_c)}"


_PARTITION_RE = re.compile(r"PARTITION\s+(.*)$")


def parse_event(line: str) -> tuple[str, str]:
    """Split a BMC event line into (type, rest)."""
    line = line.strip()
    if not line:
        return ("", "")
    parts = line.split(" ", 1)
    return (parts[0], parts[1] if len(parts) > 1 else "")
