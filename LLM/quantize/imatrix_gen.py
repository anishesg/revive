#!/usr/bin/env python3
"""Compute a role-specific importance matrix for quantization calibration.

imatrix is required for Q2_K/Q3_K to retain quality at ultra-low bits;
without it, K-quants degrade badly on small models. Runs llama.cpp's
./imatrix binary on an fp16 GGUF over the role's calibration prompts.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from LLM.common import gguf_io  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", required=True)
    parser.add_argument("--fp16", required=True, help="Path to fp16 GGUF")
    parser.add_argument("--out", default=None)
    parser.add_argument("--calibration", default=None)
    parser.add_argument("--chunks", type=int, default=100)
    args = parser.parse_args()

    calib = Path(args.calibration) if args.calibration else (
        REPO_ROOT / "LLM" / "data" / "calibration_prompts" / f"{args.role}.txt"
    )
    if not calib.exists():
        raise SystemExit(
            f"Calibration prompts missing: {calib}. Run data/build_calibration.py."
        )

    out = Path(args.out) if args.out else (
        REPO_ROOT / "LLM" / "output" / "imatrix" / f"{args.role}.dat"
    )
    gguf_io.run_imatrix(Path(args.fp16), calib, out, chunks=args.chunks)
    print(f"[imatrix] wrote {out}")


if __name__ == "__main__":
    main()
