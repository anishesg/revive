#!/usr/bin/env python3
"""Generate expanded per-role training data via Claude Haiku.

Imports prompt banks and the per-example generator from
training/generate_role_dataset.py without modifying it. Default is 750
examples per role (friend's plan scaling), up from training/'s default 300.

Usage:
  ANTHROPIC_API_KEY=sk-... python3 -m LLM.data.generate_expanded_dataset \\
      --role all --n 750
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from training.generate_role_dataset import (  # noqa: E402
    DIVERSE_QUERIES,
    ROLE_TRAINING_PROMPTS,
    generate_example,
)

from LLM.common.role_registry import ALL_ROLE_NAMES  # noqa: E402

# training/generate_role_dataset.py lacks an aggregator prompt; it has its own
# pipeline in training/generate_dataset.py. We only cover the per-role teachers
# here. Aggregator data still comes from training/generate_dataset.py.
ROLES_WITH_PROMPTS = tuple(r for r in ALL_ROLE_NAMES if r in ROLE_TRAINING_PROMPTS)

DEFAULT_OUT = REPO_ROOT / "LLM" / "output" / "data"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", default="all")
    parser.add_argument("--n", type=int, default=750)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--sleep", type=float, default=0.3)
    args = parser.parse_args()

    import anthropic  # imported late so --help works without the lib

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit("Set ANTHROPIC_API_KEY")
    client = anthropic.Anthropic(api_key=api_key)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    roles = ROLES_WITH_PROMPTS if args.role == "all" else [args.role]
    for role in roles:
        if role not in ROLE_TRAINING_PROMPTS:
            print(f"[skip] {role}: no per-role prompt in training/generate_role_dataset.py")
            continue
        out_path = out_dir / f"{role}.haiku.jsonl"
        existing = []
        if out_path.exists():
            with out_path.open() as f:
                existing = [json.loads(line) for line in f]
            print(f"[resume] {role}: {len(existing)} existing")

        with out_path.open("a") as f:
            while len(existing) < args.n:
                query = random.choice(DIVERSE_QUERIES)
                ex = generate_example(client, role, query)
                if ex and ex.get("output"):
                    f.write(json.dumps(ex) + "\n")
                    f.flush()
                    existing.append(ex)
                    print(f"[{role}] {len(existing)}/{args.n}")
                time.sleep(args.sleep)

        print(f"[done] {role}: {len(existing)} -> {out_path}")

    print(f"\nEstimated cost: ${len(roles) * args.n * 0.003:.2f}")


if __name__ == "__main__":
    main()
