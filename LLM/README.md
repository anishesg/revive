# LLM

Composite pipeline that sits above `training/`. Takes Qwen3 base models and produces a per-device-tier matrix of GGUFs so every class of hardware in the REVIVE swarm — from an iPhone 6s to an M2 MacBook — gets a build sized to its RAM and CPU.

Three tracks:

1. **Expanded data + distillation** (`data/`) — Claude Haiku for diversity, local Qwen3-4B as a teacher for bulk high-signal examples. ~1500 ex/role, 5× the current training set.
2. **Role-aware structural pruning** (`prune/`) — drop transformer blocks that contribute little to a specific role. Aggressive for Spotter, moderate for Drafter/Concise, none for Reasoner/Aggregator.
3. **Heterogeneous quantization** (`quantize/`) — imatrix-calibrated Q2_K / Q3_K_S / Q4_K_M / Q5_K_M exports per role, targeting four device tiers.

## Quick start

```bash
cd LLM
pip install -r requirements.txt

# One-off: verify llama.cpp binaries are reachable
bash scripts/quick_test.sh --check-only

# Minimum viable ship: generate data → train 8 roles → export standard tier
ANTHROPIC_API_KEY=sk-... bash scripts/bootstrap.sh

# Full tier matrix: up to 32 GGUFs + manifest.json
bash scripts/compress_all.sh
```

## Outputs

```
output/
├── data/{role}.jsonl              # merged Haiku + Qwen3-4B distilled examples
├── merged/{role}/                 # HF checkpoints after QLoRA merge
├── imatrix/{role}.dat             # per-role importance matrix
└── gguf/revive-{role}-qwen3-{size}-{tier}-{quant}.gguf
```

## Relationship to `training/`

`training/` is the original Qwen3 fine-tuning pipeline and stays untouched. `LLM/` imports its prompt banks as a Python module (no edits) and mirrors its QLoRA hyperparameters in `train/train_qwen_role.py`. The one-file-per-role output pattern from `training/` is replaced by a (role × tier) matrix export here.
