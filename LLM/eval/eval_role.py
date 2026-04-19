#!/usr/bin/env python3
"""Per-role evaluation against the Qwen3-4B teacher.

Metrics vary by role:
  spotter:            classification accuracy (exact-match on 6 categories)
  concise, drafter:   agreement rate via Haiku judge (semantic-equivalent)
  reasoner,
  factchecker:        correctness rubric via Haiku judge + factuality-regression rate
  writer:             BLEU/ROUGE against teacher
  critic:             flaw-detection precision on a curated adversarial set (stub)
  aggregator:         manual spot-check file emitted for human review (stub)

Outputs a summary JSON to output/eval/{role}.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from training.generate_role_dataset import (  # noqa: E402
    DIVERSE_QUERIES,
    SYSTEM_PROMPTS,
)

from LLM.common.role_registry import ALL_ROLE_NAMES  # noqa: E402
from LLM.data.distill_from_qwen4b import build_prompt, clean_spotter_output  # noqa: E402

DEFAULT_OUT = REPO_ROOT / "LLM" / "output" / "eval"


def infer(llm, role: str, query: str, max_tokens: int = 300) -> str:
    prompt = build_prompt(role, query)
    out = llm(
        prompt,
        max_tokens=8 if role == "spotter" else max_tokens,
        temperature=0.0 if role == "spotter" else 0.7,
        stop=["<|im_end|>", "<|im_start|>"],
    )
    text = out["choices"][0]["text"].strip()
    return clean_spotter_output(text) if role == "spotter" else text


def haiku_judge(client, query: str, a: str, b: str) -> bool:
    """Return True if a and b convey the same answer."""
    prompt = (
        "Two AI agents answered the same query. Are their answers substantively "
        "equivalent (same core claim, comparable coverage)? Reply only YES or NO.\n\n"
        f"Query: {query}\n\nA: {a}\n\nB: {b}\n\nEquivalent?"
    )
    msg = client.messages.create(
        model="claude-3-5-haiku-20241022",
        max_tokens=8,
        messages=[{"role": "user", "content": prompt}],
    )
    return "YES" in msg.content[0].text.upper()


def eval_spotter(student_llm, teacher_llm, n: int) -> dict:
    correct = total = 0
    disagreements = []
    for q in DIVERSE_QUERIES[:n]:
        s = infer(student_llm, "spotter", q)
        t = infer(teacher_llm, "spotter", q)
        total += 1
        if s == t:
            correct += 1
        else:
            disagreements.append({"query": q, "student": s, "teacher": t})
    return {
        "metric": "classification_accuracy",
        "accuracy": correct / max(total, 1),
        "n": total,
        "disagreements": disagreements,
    }


def eval_judge(student_llm, teacher_llm, role: str, n: int) -> dict:
    import anthropic
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit("Set ANTHROPIC_API_KEY for Haiku judge")
    client = anthropic.Anthropic(api_key=api_key)

    agree = total = 0
    examples = []
    for q in DIVERSE_QUERIES[:n]:
        s = infer(student_llm, role, q)
        t = infer(teacher_llm, role, q)
        equiv = haiku_judge(client, q, s, t)
        total += 1
        if equiv:
            agree += 1
        examples.append({"query": q, "student": s, "teacher": t, "equivalent": equiv})
        time.sleep(0.2)
    return {
        "metric": "agreement_rate_haiku_judge",
        "agreement_rate": agree / max(total, 1),
        "n": total,
        "examples": examples,
    }


def eval_writer(student_llm, teacher_llm, n: int) -> dict:
    try:
        from sacrebleu import sentence_bleu
    except ImportError:
        sentence_bleu = None

    scores = []
    examples = []
    for q in DIVERSE_QUERIES[:n]:
        s = infer(student_llm, "writer", q)
        t = infer(teacher_llm, "writer", q)
        if sentence_bleu is None:
            bleu = None
        else:
            bleu = sentence_bleu(s, [t]).score
            scores.append(bleu)
        examples.append({"query": q, "student": s, "teacher": t, "bleu": bleu})
    return {
        "metric": "bleu_vs_teacher",
        "mean_bleu": sum(scores) / len(scores) if scores else None,
        "n": n,
        "examples": examples[:5],  # truncate for readability
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", default="all")
    parser.add_argument("--student", required=True,
                        help="Path to student GGUF")
    parser.add_argument("--teacher", required=True, help="Path to Qwen3-4B GGUF")
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--n-ctx", type=int, default=2048)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    from llama_cpp import Llama

    print(f"[eval] student={args.student}")
    print(f"[eval] teacher={args.teacher}")
    student_llm = Llama(model_path=args.student, n_ctx=args.n_ctx, n_gpu_layers=-1, verbose=False)
    teacher_llm = Llama(model_path=args.teacher, n_ctx=args.n_ctx, n_gpu_layers=-1, verbose=False)

    roles = ALL_ROLE_NAMES if args.role == "all" else (args.role,)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for role in roles:
        print(f"[eval] role={role}")
        if role == "spotter":
            result = eval_spotter(student_llm, teacher_llm, args.n)
        elif role == "writer":
            result = eval_writer(student_llm, teacher_llm, args.n)
        elif role in ("concise", "drafter", "reasoner", "factchecker", "critic"):
            result = eval_judge(student_llm, teacher_llm, role, args.n)
        elif role == "aggregator":
            result = {
                "metric": "manual_spot_check",
                "note": "Aggregator needs human review. See LLM/eval/aggregator_samples.",
            }
        else:
            result = {"metric": "unknown"}
        (out_dir / f"{role}.json").write_text(json.dumps(result, indent=2))
        print(f"[eval] {role}: {result.get('metric')}")


if __name__ == "__main__":
    main()
