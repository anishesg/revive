#!/usr/bin/env python3
"""Emit per-tier quantized GGUFs from one fp16 GGUF + imatrix."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from LLM.common import gguf_io  # noqa: E402

TIER_QUANTS = {
    "ewaste": "Q2_K",
    "budget": "Q3_K_S",
    "standard": "Q4_K_M",
    "modern": "Q5_K_M",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", required=True)
    parser.add_argument("--fp16", required=True)
    parser.add_argument("--imatrix", default=None)
    parser.add_argument("--tiers", nargs="+", default=list(TIER_QUANTS.keys()))
    parser.add_argument("--size-label", required=True, help="e.g. 0.6b, 1.7b")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else (
        REPO_ROOT / "LLM" / "output" / "gguf"
    )
    imatrix = Path(args.imatrix) if args.imatrix else None

    for tier in args.tiers:
        quant = TIER_QUANTS[tier]
        name = gguf_io.gguf_name(args.role, args.size_label, tier, quant)
        out_path = out_dir / name
        print(f"[quantize] {tier} -> {out_path}")
        gguf_io.quantize(Path(args.fp16), out_path, quant, imatrix=imatrix)
        print(f"[quantize] {name}: {gguf_io.human_size(out_path)}")


if __name__ == "__main__":
    main()
