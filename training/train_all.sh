#!/bin/bash
# REVIVE — Full Training Pipeline
# Generates datasets and fine-tunes all agent role models + aggregator.
#
# Prerequisites:
#   pip install unsloth trl transformers datasets anthropic huggingface_hub
#   export ANTHROPIC_API_KEY=sk-...
#
# Usage:
#   bash train_all.sh           # Full pipeline
#   bash train_all.sh --skip-gen  # Skip data generation, use existing data

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

SKIP_GEN=false
if [ "$1" = "--skip-gen" ]; then
    SKIP_GEN=true
fi

echo "╔══════════════════════════════════════════════════╗"
echo "║     REVIVE — Full Training Pipeline               ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""

# ── 1. Generate aggregator dataset ───────────────────────────────────────────
if [ "$SKIP_GEN" = false ]; then
    echo "═══ Step 1: Generating aggregator dataset (2000 examples) ═══"
    python3 generate_dataset.py --n 2000 --out data.jsonl
    echo ""

    echo "═══ Step 2: Generating role-specific datasets ═══"
    python3 generate_role_dataset.py --role all --n 300 --out-dir .
    echo ""
fi

# ── 2. Train aggregator ─────────────────────────────────────────────────────
echo "═══ Step 3: Training aggregator model (Qwen3-1.7B) ═══"
python3 train.py --data data.jsonl --output ./output/aggregator --epochs 3 --lora-r 64 --seq-len 4096
echo ""

# ── 3. Train individual roles ────────────────────────────────────────────────
echo "═══ Step 4: Training role-specific models ═══"

for ROLE in reasoner writer critic factchecker; do
    if [ -f "data-${ROLE}.jsonl" ]; then
        echo "  Training: $ROLE (Qwen3-1.7B)"
        python3 train_role.py --role "$ROLE" --data "data-${ROLE}.jsonl" --output ./output
        echo ""
    fi
done

for ROLE in spotter drafter concise; do
    if [ -f "data-${ROLE}.jsonl" ]; then
        echo "  Training: $ROLE (Qwen3-0.6B)"
        python3 train_role.py --role "$ROLE" --data "data-${ROLE}.jsonl" --output ./output
        echo ""
    fi
done

# ── 4. Summary ───────────────────────────────────────────────────────────────
echo "╔══════════════════════════════════════════════════╗"
echo "║              Training Complete!                   ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""
echo "Output models:"
find ./output -name "*.gguf" -exec echo "  {}" \;
echo ""
echo "Deploy:"
echo "  Aggregator GGUF → iPad/Pi coordinator (Documents/)"
echo "  Role GGUFs → worker phones (Documents/)"
echo ""
echo "Model sizes (approximate):"
echo "  Qwen3-1.7B roles (reasoner, writer, etc.): ~1.1GB each"
echo "  Qwen3-0.6B roles (spotter, drafter):       ~0.4GB each"
echo "  Aggregator (1.7B):                          ~1.1GB"
