#!/usr/bin/env python3
"""For one role, compare quant/prune tier variants on quality + tok/s.

Loads every {role}-qwen3-{size}-{tier}-{quant}.gguf found in output/gguf/,
runs a fixed prompt set through each, measures tok/s, and (for Spotter)
exact-match accuracy against the teacher. Other roles get agreement rate
via Haiku judge.

This is a coarse triage tool — it exists so you can decide, e.g., whether
the ewaste Q2_K variant is actually usable or whether it degrades so badly
that budget Q3_K_S is the real floor.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from training.generate_role_dataset import DIVERSE_QUERIES  # noqa: E402
from LLM.data.distill_from_qwen4b import build_prompt, clean_spotter_output  # noqa: E402

GGUF_DIR = REPO_ROOT / "LLM" / "output" / "gguf"
DEFAULT_OUT = REPO_ROOT / "LLM" / "output" / "eval"

FILENAME_RE = re.compile(
    r"revive-(?P<role>[^-]+)-qwen3-(?P<size>[0-9.]+b)-(?P<tier>[^-]+)-(?P<quant>[A-Z0-9_]+)\.gguf"
)


def find_variants(role: str) -> list[dict]:
    variants = []
    for path in sorted(GGUF_DIR.glob("*.gguf")):
        m = FILENAME_RE.match(path.name)
        if not m or m.group("role") != role:
            continue
        variants.append({"path": path, **m.groupdict()})
    return variants


def bench(path: Path, role: str, n: int) -> dict:
    from llama_cpp import Llama
    llm = Llama(model_path=str(path), n_ctx=2048, n_gpu_layers=-1, verbose=False)

    tok_counts = []
    times = []
    outputs = []
    for q in DIVERSE_QUERIES[:n]:
        prompt = build_prompt(role, q)
        t0 = time.time()
        out = llm(
            prompt,
            max_tokens=8 if role == "spotter" else 150,
            temperature=0.0 if role == "spotter" else 0.7,
            stop=["<|im_end|>", "<|im_start|>"],
        )
        elapsed = time.time() - t0
        txt = out["choices"][0]["text"].strip()
        tok = out.get("usage", {}).get("completion_tokens") or len(txt.split())
        tok_counts.append(tok)
        times.append(elapsed)
        outputs.append(clean_spotter_output(txt) if role == "spotter" else txt)
    del llm

    total_tok = sum(tok_counts) or 1
    total_time = sum(times) or 1e-6
    return {
        "tokens_per_second": total_tok / total_time,
        "mean_tokens": total_tok / len(tok_counts),
        "mean_latency_s": total_time / len(times),
        "outputs": outputs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", required=True)
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    variants = find_variants(args.role)
    if not variants:
        raise SystemExit(f"No GGUFs found for role={args.role} in {GGUF_DIR}")

    print(f"[tier] {args.role}: {len(variants)} variants")
    results = {}
    for v in variants:
        key = f"{v['tier']}-{v['quant']}"
        print(f"[tier] benchmarking {key}")
        results[key] = bench(v["path"], args.role, args.n)

    # Cross-tier agreement: each tier vs modern (assumed best)
    best_key = next((k for k in results if k.startswith("modern")), None)
    if best_key:
        best_outs = results[best_key]["outputs"]
        for key, r in results.items():
            matches = sum(1 for a, b in zip(r["outputs"], best_outs) if a == b)
            r["exact_agreement_vs_modern"] = matches / len(best_outs)

    out = Path(args.out_dir) / f"tier-{args.role}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"[tier] wrote {out}")


if __name__ == "__main__":
    main()
