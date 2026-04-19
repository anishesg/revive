"""Write a JSON manifest describing every emitted GGUF.

Consumed downstream (v2) by on-device auto-selection: SwarmManager.swift
or rpi/aggregator.py can read this to pick the right GGUF for each
discovered worker's RAM tier.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


FILENAME_RE = re.compile(
    r"revive-(?P<role>[^-]+)-qwen3-(?P<size>[0-9.]+b)-(?P<tier>[^-]+)-(?P<quant>[A-Z0-9_]+)\.gguf"
)


def _sha256(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def write_manifest(paths: list[Path], out: Path) -> None:
    entries = []
    for p in sorted(paths):
        m = FILENAME_RE.match(p.name)
        if not m:
            continue
        entries.append({
            "file": p.name,
            "role": m.group("role"),
            "base": "qwen3",
            "size": m.group("size"),
            "tier": m.group("tier"),
            "quant": m.group("quant"),
            "size_bytes": p.stat().st_size,
            "sha256": _sha256(p),
        })
    payload = {"version": 1, "entries": entries}
    out.write_text(json.dumps(payload, indent=2))
