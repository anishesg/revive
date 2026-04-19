#!/usr/bin/env python3
"""Distill role training data from a local Qwen3-4B teacher.

Reuses the prompt bank from training/generate_role_dataset.py. For each
(role, query), asks Qwen3-4B to produce the ideal response under the role's
system prompt, then writes it in the same JSONL schema as the Haiku data so
both streams merge cleanly.

Usage:
  python3 -m LLM.data.distill_from_qwen4b \\
      --role all --n 750 \\
      --teacher ~/revive/models/qwen3-4b-q4_k_m.gguf
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from training.generate_role_dataset import (  # noqa: E402
    DIVERSE_QUERIES,
    SYSTEM_PROMPTS,
)

from LLM.common.role_registry import ALL_ROLE_NAMES  # noqa: E402

DEFAULT_OUT = REPO_ROOT / "LLM" / "output" / "data"
DEFAULT_TEACHER = REPO_ROOT / "models" / "qwen3-4b-q4_k_m.gguf"

SPOTTER_CATEGORIES = (
    "SIMPLE_FACT", "COMPLEX_REASONING", "CREATIVE", "CODE", "MATH", "OPINION",
)


def build_prompt(role: str, query: str) -> str:
    sys_prompt = SYSTEM_PROMPTS.get(role, f"You are a specialized {role} agent.")
    return (
        f"<|im_start|>system\n{sys_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{query}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def clean_spotter_output(text: str) -> str:
    upper = text.upper()
    for cat in SPOTTER_CATEGORIES:
        if cat in upper:
            return cat
    return "COMPLEX_REASONING"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", default="all")
    parser.add_argument("--n", type=int, default=750)
    parser.add_argument("--teacher", default=str(DEFAULT_TEACHER))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--n-ctx", type=int, default=2048)
    parser.add_argument("--max-tokens", type=int, default=400)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--n-gpu-layers", type=int, default=-1,
                        help="-1 = all layers on GPU/Metal (default)")
    args = parser.parse_args()

    from llama_cpp import Llama  # late import

    teacher_path = Path(args.teacher)
    if not teacher_path.exists():
        raise SystemExit(
            f"Teacher model not found at {teacher_path}. "
            "Run macos/setup.sh to download Qwen3-4B, or pass --teacher."
        )

    print(f"[teacher] loading {teacher_path}...")
    llm = Llama(
        model_path=str(teacher_path),
        n_ctx=args.n_ctx,
        n_gpu_layers=args.n_gpu_layers,
        verbose=False,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    roles = ALL_ROLE_NAMES if args.role == "all" else (args.role,)

    for role in roles:
        if role == "aggregator":
            print(f"[skip] {role}: handled by training/generate_dataset.py")
            continue
        if role not in SYSTEM_PROMPTS:
            print(f"[skip] {role}: no system prompt")
            continue

        out_path = out_dir / f"{role}.qwen4b.jsonl"
        existing = []
        if out_path.exists():
            with out_path.open() as f:
                existing = [json.loads(line) for line in f]
            print(f"[resume] {role}: {len(existing)} existing")

        with out_path.open("a") as f:
            while len(existing) < args.n:
                query = random.choice(DIVERSE_QUERIES)
                prompt = build_prompt(role, query)
                out = llm(
                    prompt,
                    max_tokens=args.max_tokens if role != "spotter" else 8,
                    temperature=args.temperature if role != "spotter" else 0.0,
                    stop=["<|im_end|>", "<|im_start|>"],
                )
                raw = out["choices"][0]["text"].strip()
                if not raw:
                    continue
                response = clean_spotter_output(raw) if role == "spotter" else raw
                ex = {
                    "instruction": SYSTEM_PROMPTS[role],
                    "input": query,
                    "output": response,
                }
                f.write(json.dumps(ex) + "\n")
                f.flush()
                existing.append(ex)
                print(f"[{role}] {len(existing)}/{args.n}")

        print(f"[done] {role}: {len(existing)} -> {out_path}")


if __name__ == "__main__":
    main()
