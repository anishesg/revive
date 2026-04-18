#!/usr/bin/env python3
"""
QLoRA fine-tune Qwen3-1.7B for MoA aggregation on the REVIVE dataset.
Run on EC2 g5.xlarge (A10G 24GB VRAM).

Usage:
  python3 train.py --data data.jsonl --output ./output
"""
import os
import json
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",   default="data.jsonl")
    parser.add_argument("--output", default="./output")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr",     type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--batch",  type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=4096)
    args = parser.parse_args()

    # ── Imports (after pip install) ───────────────────────────────────────────
    from unsloth import FastLanguageModel
    from unsloth import is_bfloat16_supported
    import torch
    from trl import SFTTrainer
    from transformers import TrainingArguments
    from datasets import Dataset

    # ── Load base model with Unsloth (4-bit QLoRA) ───────────────────────────
    print("[train] Loading Qwen/Qwen3-1.7B with 4-bit quantization...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="Qwen/Qwen3-1.7B",
        max_seq_length=args.seq_len,
        dtype=None,
        load_in_4bit=True,
    )

    # ── Add LoRA adapters ─────────────────────────────────────────────────────
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_r * 2,  # 128 with default r=64
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    # ── Load dataset ──────────────────────────────────────────────────────────
    print(f"[train] Loading dataset from {args.data}...")
    examples = []
    with open(args.data) as f:
        for line in f:
            ex = json.loads(line)
            examples.append(ex)

    print(f"[train] {len(examples)} training examples")

    # Format as ChatML (matches what the iOS app sends at inference time)
    def format_example(ex):
        return {
            "text": (
                f"<|im_start|>system\n{ex['instruction']}<|im_end|>\n"
                f"<|im_start|>user\n{ex['input']}<|im_end|>\n"
                f"<|im_start|>assistant\n{ex['output']}<|im_end|>"
            )
        }

    formatted = [format_example(ex) for ex in examples]
    split_idx = int(len(formatted) * 0.95)
    train_data = formatted[:split_idx]
    eval_data = formatted[split_idx:]
    dataset = Dataset.from_list(train_data)
    eval_dataset = Dataset.from_list(eval_data)
    print(f"[train] {len(train_data)} train, {len(eval_data)} eval examples")

    # ── Training ──────────────────────────────────────────────────────────────
    print("[train] Starting training...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=args.seq_len,
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
            eval_strategy="epoch",
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=42,
            output_dir=args.output,
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            report_to="none",
        ),
    )

    trainer.train()
    print("[train] Training complete.")

    # ── Save merged model ─────────────────────────────────────────────────────
    merged_dir = os.path.join(args.output, "merged")
    print(f"[train] Saving merged model to {merged_dir}...")
    model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")

    # ── Export to GGUF (Q4_K_M) ───────────────────────────────────────────────
    gguf_path = os.path.join(args.output, "revive-aggregator-1.7b-Q4_K_M.gguf")
    print(f"[train] Exporting to GGUF: {gguf_path}")
    model.save_pretrained_gguf(
        os.path.join(args.output, "revive-aggregator-1.7b"),
        tokenizer,
        quantization_method="q4_k_m",
    )

    print(f"\n[train] Done. GGUF model at: {gguf_path}")
    print("[train] Transfer to iPad: Documents/revive-aggregator-1.7b-Q4_K_M.gguf")


if __name__ == "__main__":
    main()
