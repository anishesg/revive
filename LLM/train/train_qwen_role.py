#!/usr/bin/env python3
"""QLoRA fine-tune Qwen3 for a role on the LLM/ merged dataset.

Mirrors training/train_role.py hyperparameters. Writes merged 16-bit HF
checkpoint to LLM/output/merged/{role}/. GGUF export is done
separately by export/export_tier_matrix.py so we can emit multiple tiers
from one training run.

Usage:
  python3 -m LLM.train.train_qwen_role --role reasoner
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from LLM.common.role_registry import get as get_role  # noqa: E402

DEFAULT_DATA_DIR = REPO_ROOT / "LLM" / "output" / "data"
DEFAULT_MERGED_DIR = REPO_ROOT / "LLM" / "output" / "merged"


def format_chatml(ex: dict) -> dict:
    return {
        "text": (
            f"<|im_start|>system\n{ex['instruction']}<|im_end|>\n"
            f"<|im_start|>user\n{ex['input']}<|im_end|>\n"
            f"<|im_start|>assistant\n{ex['output']}<|im_end|>"
        )
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", required=True)
    parser.add_argument("--data", default=None, help="Default: output/data/{role}.jsonl")
    parser.add_argument("--output-dir", default=str(DEFAULT_MERGED_DIR))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--batch", type=int, default=4)
    args = parser.parse_args()

    role = get_role(args.role)
    data_path = Path(args.data) if args.data else DEFAULT_DATA_DIR / f"{role.name}.jsonl"
    if not data_path.exists():
        raise SystemExit(
            f"Missing {data_path}. Run data/generate_expanded_dataset.py and "
            "data/merge_datasets.py first."
        )

    from unsloth import FastLanguageModel, is_bfloat16_supported  # noqa
    from trl import SFTTrainer
    from transformers import TrainingArguments
    from datasets import Dataset

    print(f"[train] role={role.name} base={role.base_model} seq_len={role.seq_len}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=role.base_model,
        max_seq_length=role.seq_len,
        dtype=None,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_r * 2,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    examples = [json.loads(line) for line in data_path.open()]
    print(f"[train] {len(examples)} examples from {data_path}")
    dataset = Dataset.from_list([format_chatml(ex) for ex in examples])

    output_dir = Path(args.output_dir) / role.name
    output_dir.mkdir(parents=True, exist_ok=True)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=role.seq_len,
        dataset_num_proc=2,
        packing=True,
        args=TrainingArguments(
            per_device_train_batch_size=args.batch,
            gradient_accumulation_steps=4,
            warmup_ratio=0.05,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=10,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=42,
            output_dir=str(output_dir / "checkpoints"),
            save_strategy="epoch",
            report_to="none",
        ),
    )

    trainer.train()

    merged_dir = output_dir / "hf"
    model.save_pretrained_merged(str(merged_dir), tokenizer, save_method="merged_16bit")
    print(f"[train] merged 16-bit checkpoint -> {merged_dir}")


if __name__ == "__main__":
    main()
