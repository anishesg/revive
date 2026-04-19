#!/usr/bin/env bash
# Sync trained GGUF artifacts from S3 to LLM/output/gguf/ locally.
#
# Usage:
#   export S3_BUCKET=my-revive-models
#   bash LLM/scripts/aws_pull.sh
set -euo pipefail

: "${S3_BUCKET:?Set S3_BUCKET}"
AWS_REGION="${AWS_REGION:-us-east-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${SCRIPT_DIR}/../output/gguf"
mkdir -p "${OUT_DIR}"

echo "Syncing s3://${S3_BUCKET}/revive-models/ -> ${OUT_DIR}/"
aws s3 sync "s3://${S3_BUCKET}/revive-models/" "${OUT_DIR}/" \
    --region "${AWS_REGION}"

echo ""
echo "=== Downloaded models ==="
find "${OUT_DIR}" -name "*.gguf" | sort | while read -r f; do
    SIZE="$(du -sh "$f" | cut -f1)"
    echo "  ${SIZE}  $(basename "$f")"
done

if [ -f "${OUT_DIR}/manifest.json" ]; then
    echo ""
    echo "manifest.json present."
fi
