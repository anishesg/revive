#!/usr/bin/env python3
"""Merge Haiku and Qwen3-4B teacher datasets per role, dedupe by (input, output)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / "LLM" / "output" / "data"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    for role_path in sorted(data_dir.glob("*.haiku.jsonl")):
        role = role_path.name.removesuffix(".haiku.jsonl")
        qwen_path = data_dir / f"{role}.qwen4b.jsonl"
        merged_path = data_dir / f"{role}.jsonl"

        seen: set[tuple[str, str]] = set()
        merged = []
        for src in (role_path, qwen_path):
            if not src.exists():
                continue
            with src.open() as f:
                for line in f:
                    ex = json.loads(line)
                    key = (ex.get("input", ""), ex.get("output", ""))
                    if key in seen:
                        continue
                    seen.add(key)
                    merged.append(ex)

        with merged_path.open("w") as f:
            for ex in merged:
                f.write(json.dumps(ex) + "\n")
        print(f"[merge] {role}: {len(merged)} unique -> {merged_path}")


if __name__ == "__main__":
    main()
