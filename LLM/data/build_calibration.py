#!/usr/bin/env python3
"""Seed per-role calibration prompts for imatrix from training/ prompt banks.

Writes one .txt file per role under data/calibration_prompts/. These are
UNLABELED queries consumed by imatrix during quantization — not training data.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from training.generate_role_dataset import DIVERSE_QUERIES  # noqa: E402

from LLM.common.role_registry import ALL_ROLE_NAMES  # noqa: E402

OUT_DIR = REPO_ROOT / "LLM" / "data" / "calibration_prompts"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for role in ALL_ROLE_NAMES:
        path = OUT_DIR / f"{role}.txt"
        if path.exists() and path.stat().st_size > 0:
            print(f"[keep] {path} already populated")
            continue
        with path.open("w") as f:
            for q in DIVERSE_QUERIES:
                f.write(q + "\n")
        print(f"[seed] {path}: {len(DIVERSE_QUERIES)} prompts")


if __name__ == "__main__":
    main()
