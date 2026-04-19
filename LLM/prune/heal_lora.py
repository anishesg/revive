#!/usr/bin/env python3
"""Optional: short QLoRA healing pass after pruning.

Only run if post-prune eval shows >15% perplexity drop. Reuses the
training loop from LLM.train.train_qwen_role but with fewer
epochs and a smaller learning rate to recover quality without overfitting
the small post-prune model to the distilled dataset.

Usage:
  python3 -m LLM.prune.heal_lora --role spotter \\
      --base LLM/output/merged/spotter/hf-pruned \\
      --data LLM/output/data/spotter.jsonl \\
      --out  LLM/output/merged/spotter/hf-pruned-healed
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", required=True)
    parser.add_argument("--base", required=True, help="Pruned HF checkpoint dir")
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--lora-r", type=int, default=16)
    args = parser.parse_args()

    from unsloth import FastLanguageModel, is_bfloat16_supported
    from trl import SFTTrainer
    from transformers import TrainingArguments
    from datasets import Dataset

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base,
        max_seq_length=1024,
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

    def fmt(ex):
        return {
            "text": (
                f"<|im_start|>system\n{ex['instruction']}<|im_end|>\n"
                f"<|im_start|>user\n{ex['input']}<|im_end|>\n"
                f"<|im_start|>assistant\n{ex['output']}<|im_end|>"
            )
        }

    examples = [json.loads(l) for l in Path(args.data).open()]
    dataset = Dataset.from_list([fmt(ex) for ex in examples])

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=1024,
        packing=True,
        args=TrainingArguments(
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=10,
            optim="adamw_8bit",
            lr_scheduler_type="cosine",
            seed=42,
            output_dir=str(out_dir / "checkpoints"),
            save_strategy="no",
            report_to="none",
        ),
    )
    trainer.train()
    model.save_pretrained_merged(str(out_dir), tokenizer, save_method="merged_16bit")
    print(f"[heal] healed checkpoint -> {out_dir}")


if __name__ == "__main__":
    main()
