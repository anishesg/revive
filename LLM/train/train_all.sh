#!/usr/bin/env bash
set -euo pipefail

# Fine-tune all 8 roles sequentially. Assumes data/ has already been generated
# and merged. Run from repo root.

ROLES=(spotter drafter concise reasoner writer critic factchecker aggregator)

for role in "${ROLES[@]}"; do
    echo "=== Training ${role} ==="
    python3 -m LLM.train.train_qwen_role --role "${role}"
done

echo "All roles trained. Merged HF checkpoints in LLM/output/merged/."
