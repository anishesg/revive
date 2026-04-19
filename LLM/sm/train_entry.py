#!/usr/bin/env python3
"""SageMaker entrypoint for REVIVE role training.

Runs inside a SageMaker training container. One job trains one or more of the
8 REVIVE roles (spotter, drafter, concise, reasoner, writer, critic,
factchecker, aggregator) using QLoRA via Unsloth, and writes merged 16-bit HF
checkpoints to SM_MODEL_DIR so SageMaker tars them up into model.tar.gz.

Data source (in priority order):
  1. SM_CHANNEL_TRAINING — JSONL files named {role}.jsonl mounted at
     /opt/ml/input/data/training/. Produced by LLM.data.merge_datasets.
  2. Inline synthetic bootstrap — a small per-role set built from the
     calibration prompt banks shipped with this repo. Enough to prove the
     plumbing end-to-end when no external dataset is staged.

Hyperparameters are read from /opt/ml/input/config/hyperparameters.json
(SageMaker serializes all values as strings, so we parse on read).

Required sidecars in source_dir: LLM/common/role_registry.py, LLM/data/
calibration_prompts/*.txt.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

# SageMaker copies the *contents* of source_dir into /opt/ml/code, so what was
# LLM/common/ locally becomes /opt/ml/code/common/ at runtime. Expose both
# _LLM_ROOT (for local `python -m`) and its parent (for `import common` and
# `import LLM.common`) so the imports below resolve in either context.
_HERE = Path(__file__).resolve().parent
_LLM_ROOT = _HERE.parent
for p in (_LLM_ROOT, _LLM_ROOT.parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

SM_MODEL_DIR = Path(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
SM_OUTPUT_DIR = Path(os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data"))
SM_CHANNEL_TRAINING = os.environ.get("SM_CHANNEL_TRAINING")


def _load_roles():
    try:
        from common.role_registry import ROLES, get, ALL_ROLE_NAMES
    except ModuleNotFoundError:
        from LLM.common.role_registry import ROLES, get, ALL_ROLE_NAMES
    return ROLES, get, ALL_ROLE_NAMES


def _read_calibration(role: str) -> list[str]:
    path = _LLM_ROOT / "data" / "calibration_prompts" / f"{role}.txt"
    if not path.exists():
        return []
    return [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]


# Per-role system instruction. Matches the live REVIVE MoA prompt style so the
# synthetic bootstrap is directionally correct even if small.
_ROLE_INSTRUCTIONS = {
    "spotter": "You are Spotter. Read the input and list the 3-5 most important entities, numbers, or claims. One per line, terse.",
    "drafter": "You are Drafter. Write a fast, rough first-pass answer. 2-4 sentences. No preamble.",
    "concise": "You are Concise. Rewrite the input as one short sentence that keeps the key fact.",
    "reasoner": "You are Reasoner. Think step by step. Show your reasoning as 3-5 numbered steps, then a final answer.",
    "writer": "You are Writer. Produce a polished, well-structured answer. Use clear paragraphs.",
    "critic": "You are Critic. Identify 2-3 concrete weaknesses in the draft and state how to fix each.",
    "factchecker": "You are Factchecker. For each claim in the input, mark [OK], [SUSPECT], or [WRONG] and briefly say why.",
    "aggregator": "You are Aggregator. Combine the role outputs into a single coherent answer. Resolve conflicts using Factchecker and Critic signals.",
}


def _synthetic_example(role: str, prompt: str) -> dict:
    """Deterministic placeholder output so the loss signal is stable.

    Real training should mount a proper dataset via SM_CHANNEL_TRAINING. This
    fallback exists so a first-pass job can smoke-test infra end-to-end.
    """
    instruction = _ROLE_INSTRUCTIONS[role]
    if role == "spotter":
        output = "- key entity\n- key number\n- core claim"
    elif role == "drafter":
        output = f"Short draft addressing: {prompt} Main point, then one supporting detail."
    elif role == "concise":
        output = prompt.split(".")[0][:120] + "."
    elif role == "reasoner":
        output = "1. Identify what is being asked.\n2. List relevant facts.\n3. Apply reasoning to reach an answer.\nFinal: <concise answer>."
    elif role == "writer":
        output = f"{prompt}\n\nOpening framing. Core exposition with two supporting points. Closing synthesis."
    elif role == "critic":
        output = "1. Claim X lacks evidence — cite a source.\n2. Logic gap between A and B — add a bridging step."
    elif role == "factchecker":
        output = "[OK] main claim aligns with consensus.\n[SUSPECT] secondary claim — verify primary source."
    else:  # aggregator
        output = f"Synthesized answer to: {prompt} Combines draft + reasoning, corrects per critic/factchecker."
    return {"instruction": instruction, "input": prompt, "output": output}


def _build_synthetic_dataset(role: str, n: int) -> list[dict]:
    prompts = _read_calibration(role) or _read_calibration("reasoner")
    if not prompts:
        prompts = ["Explain the concept in simple terms.", "What are the tradeoffs involved?"]
    rng = random.Random(42)
    out = []
    for i in range(n):
        p = prompts[i % len(prompts)] if i < len(prompts) else rng.choice(prompts)
        out.append(_synthetic_example(role, p))
    return out


def _load_training_jsonl(role: str) -> list[dict] | None:
    if not SM_CHANNEL_TRAINING:
        return None
    path = Path(SM_CHANNEL_TRAINING) / f"{role}.jsonl"
    if not path.exists():
        return None
    rows = [json.loads(line) for line in path.open() if line.strip()]
    print(f"[data] {role}: loaded {len(rows)} rows from {path}")
    return rows


def _format_chatml(ex: dict) -> dict:
    return {
        "text": (
            f"<|im_start|>system\n{ex['instruction']}<|im_end|>\n"
            f"<|im_start|>user\n{ex['input']}<|im_end|>\n"
            f"<|im_start|>assistant\n{ex['output']}<|im_end|>"
        )
    }


def train_role(role_name: str, *, epochs: int, batch: int, lora_r: int,
               lr: float, synthetic_n: int, max_seq_len_cap: int | None) -> Path:
    ROLES, get, _ = _load_roles()
    role = get(role_name)

    seq_len = role.seq_len if not max_seq_len_cap else min(role.seq_len, max_seq_len_cap)
    print(f"[train] role={role.name} base={role.base_model} seq_len={seq_len}")

    rows = _load_training_jsonl(role.name)
    if rows is None:
        rows = _build_synthetic_dataset(role.name, synthetic_n)
        print(f"[data] {role.name}: no input channel -> synthetic bootstrap ({len(rows)} rows)")

    # Lazy imports so failures in one role don't block import-time discovery.
    from unsloth import FastLanguageModel, is_bfloat16_supported
    from trl import SFTTrainer
    from transformers import TrainingArguments
    from datasets import Dataset

    # Unsloth's phone-home stats call (`_get_statistics`) hangs for 120s and
    # then raises on SageMaker because the container's outbound route to
    # their telemetry endpoint is slow/blocked. Replace it with a no-op; it's
    # pure telemetry and unrelated to model loading.
    import unsloth.models.llama as _ul
    import unsloth.models._utils as _uu
    _noop = lambda *a, **kw: None
    _ul.get_statistics = _noop
    _uu.get_statistics = _noop
    _uu._get_statistics = _noop

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=role.base_model,
        max_seq_length=seq_len,
        dtype=None,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=lora_r * 2,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    dataset = Dataset.from_list([_format_chatml(ex) for ex in rows])

    role_out = SM_MODEL_DIR / role.name
    ckpt_dir = role_out / "checkpoints"
    role_out.mkdir(parents=True, exist_ok=True)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=seq_len,
        dataset_num_proc=2,
        packing=True,
        args=TrainingArguments(
            per_device_train_batch_size=batch,
            gradient_accumulation_steps=4,
            warmup_ratio=0.05,
            num_train_epochs=epochs,
            learning_rate=lr,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=10,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=42,
            output_dir=str(ckpt_dir),
            save_strategy="epoch",
            report_to="none",
        ),
    )
    trainer.train()

    merged_dir = role_out / "hf"
    model.save_pretrained_merged(str(merged_dir), tokenizer, save_method="merged_16bit")
    print(f"[train] {role.name}: merged 16-bit checkpoint -> {merged_dir}")
    return merged_dir


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--roles", default="reasoner",
                   help="Comma-separated role names, or 'all'.")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--synthetic-n", type=int, default=64,
                   help="Rows to synthesize per role when no input channel is mounted.")
    p.add_argument("--max-seq-len", type=int, default=1024,
                   help="Upper bound on seq_len; lets small instances run 1.7B roles.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    _, _, ALL_ROLE_NAMES = _load_roles()
    roles = ALL_ROLE_NAMES if args.roles.strip() == "all" else [
        r.strip() for r in args.roles.split(",") if r.strip()
    ]

    SM_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    SM_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = {"roles": [], "epochs": args.epochs, "synthetic_n": args.synthetic_n}
    for r in roles:
        out = train_role(
            r,
            epochs=args.epochs,
            batch=args.batch,
            lora_r=args.lora_r,
            lr=args.lr,
            synthetic_n=args.synthetic_n,
            max_seq_len_cap=args.max_seq_len,
        )
        manifest["roles"].append({"role": r, "path": str(out.relative_to(SM_MODEL_DIR))})

    (SM_MODEL_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[done] trained {len(manifest['roles'])} role(s); manifest at {SM_MODEL_DIR / 'manifest.json'}")


if __name__ == "__main__":
    main()
