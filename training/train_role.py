#!/usr/bin/env python3
"""
QLoRA fine-tune Qwen3 for a specific agent role.
Produces role-specialized GGUF models for each worker type.

Usage:
  python3 train_role.py --role reasoner --data data-reasoner.jsonl --base-model Qwen/Qwen3-1.7B
  python3 train_role.py --role spotter  --data data-spotter.jsonl  --base-model Qwen/Qwen3-0.6B

Recommended base models by role:
  - spotter, drafter, concise: Qwen3-0.6B (fast classification/short answers)
  - reasoner, writer, critic, factchecker: Qwen3-1.7B (quality generation)
"""
import os
import json
import argparse
from pathlib import Path


ROLE_BASE_MODELS = {
    "spotter": "Qwen/Qwen3-0.6B",
    "drafter": "Qwen/Qwen3-0.6B",
    "concise": "Qwen/Qwen3-0.6B",
    "reasoner": "Qwen/Qwen3-1.7B",
    "writer": "Qwen/Qwen3-1.7B",
    "critic": "Qwen/Qwen3-1.7B",
    "factchecker": "Qwen/Qwen3-1.7B",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", required=True, help="Agent role to fine-tune")
    parser.add_argument("--data", required=True, help="JSONL training data")
    parser.add_argument("--base-model", default=None, help="Override base model")
    parser.add_argument("--output", default="./output")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--batch", type=int, default=4)
    args = parser.parse_args()

    base_model = args.base_model or ROLE_BASE_MODELS.get(args.role, "Qwen/Qwen3-1.7B")
    is_small = "0.6B" in base_model
    seq_len = 1024 if is_small else 2048

    from unsloth import FastLanguageModel, is_bfloat16_supported
    import torch
    from trl import SFTTrainer
    from transformers import TrainingArguments
    from datasets import Dataset

    print(f"[train_role] Role: {args.role}")
    print(f"[train_role] Base model: {base_model}")
    print(f"[train_role] Seq length: {seq_len}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=seq_len,
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

    print(f"[train_role] Loading {args.data}...")
    examples = []
    with open(args.data) as f:
        for line in f:
            examples.append(json.loads(line))
    print(f"[train_role] {len(examples)} examples")

    def format_example(ex):
        return {
            "text": (
                f"<|im_start|>system\n{ex['instruction']}<|im_end|>\n"
                f"<|im_start|>user\n{ex['input']}<|im_end|>\n"
                f"<|im_start|>assistant\n{ex['output']}<|im_end|>"
            )
        }

    dataset = Dataset.from_list([format_example(ex) for ex in examples])

    output_dir = os.path.join(args.output, args.role)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=seq_len,
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
            output_dir=output_dir,
            save_strategy="epoch",
            report_to="none",
        ),
    )

    trainer.train()
    print(f"[train_role] Training complete for {args.role}")

    merged_dir = os.path.join(output_dir, "merged")
    model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")

    model_size = "0.6b" if is_small else "1.7b"
    gguf_name = f"revive-{args.role}-{model_size}"
    model.save_pretrained_gguf(
        os.path.join(output_dir, gguf_name),
        tokenizer,
        quantization_method="q4_k_m",
    )

    print(f"\n[train_role] Done. GGUF: {output_dir}/{gguf_name}-Q4_K_M.gguf")
    print(f"[train_role] Deploy to worker device running as {args.role}")


if __name__ == "__main__":
    main()
