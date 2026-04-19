#!/usr/bin/env bash
# Runs on the EC2 instance at boot (passed as --user-data to aws ec2 run-instances).
# Installs dependencies, builds llama.cpp, runs bootstrap.sh, uploads to S3, terminates.
#
# Environment variables injected by aws_launch.sh via sed substitution:
#   ANTHROPIC_API_KEY, S3_BUCKET, GITHUB_REPO, AWS_REGION
set -euo pipefail

ANTHROPIC_API_KEY="__ANTHROPIC_API_KEY__"
S3_BUCKET="__S3_BUCKET__"
GITHUB_REPO="__GITHUB_REPO__"
AWS_REGION="__AWS_REGION__"

LOG="/var/log/revive-train.log"
exec > >(tee -a "$LOG") 2>&1

echo "[$(date)] === REVIVE training bootstrap starting ==="

# ── System deps ──────────────────────────────────────────────────────────────
apt-get update -qq
apt-get install -y -qq git cmake build-essential python3-pip python3-venv awscli

# ── Clone repo ────────────────────────────────────────────────────────────────
cd /home/ubuntu
git clone "https://github.com/${GITHUB_REPO}.git" revive
cd revive

# ── Python env ────────────────────────────────────────────────────────────────
python3 -m venv /home/ubuntu/venv
source /home/ubuntu/venv/bin/activate

pip install --upgrade pip --quiet

# Unsloth needs special install for CUDA (not plain PyPI)
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" --quiet
pip install --no-deps trl peft accelerate --quiet

# Rest of requirements (unsloth covers torch/transformers; skip to avoid conflicts)
pip install anthropic datasets huggingface_hub pyyaml sentencepiece llama-cpp-python --quiet

# ── Build llama.cpp (CUDA) ────────────────────────────────────────────────────
echo "[$(date)] Building llama.cpp with CUDA..."
git clone --depth 1 https://github.com/ggerganov/llama.cpp /home/ubuntu/llama.cpp
cd /home/ubuntu/llama.cpp
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_TESTS=OFF 2>&1 | tail -5
cmake --build build --target llama-quantize llama-imatrix llama-cli -j "$(nproc)"

# Symlink so gguf_io.py can find them
ln -sf /home/ubuntu/llama.cpp/build/bin/llama-quantize /usr/local/bin/llama-quantize
ln -sf /home/ubuntu/llama.cpp/build/bin/llama-imatrix  /usr/local/bin/llama-imatrix
ln -sf /home/ubuntu/llama.cpp/build/bin/llama-cli      /usr/local/bin/llama-cli
# convert script (Python, lives in source tree)
ln -sf /home/ubuntu/llama.cpp/convert_hf_to_gguf.py    /usr/local/bin/convert_hf_to_gguf.py

cd /home/ubuntu/revive

# ── Download Qwen3-4B teacher GGUF ───────────────────────────────────────────
echo "[$(date)] Downloading Qwen3-4B teacher..."
mkdir -p /home/ubuntu/revive/models
huggingface-cli download bartowski/Qwen3-4B-GGUF \
    --include "Qwen3-4B-Q4_K_M.gguf" \
    --local-dir /home/ubuntu/revive/models

# ── Run bootstrap ─────────────────────────────────────────────────────────────
echo "[$(date)] Running bootstrap.sh..."
export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY}"
export REVIVE_TEACHER="/home/ubuntu/revive/models/Qwen3-4B-Q4_K_M.gguf"

bash LLM/scripts/bootstrap.sh

# ── Upload artifacts to S3 ────────────────────────────────────────────────────
echo "[$(date)] Uploading GGUFs to s3://${S3_BUCKET}/revive-models/..."
aws s3 sync LLM/output/gguf/ "s3://${S3_BUCKET}/revive-models/" \
    --region "${AWS_REGION}" \
    --exclude "_fp16*"

echo "[$(date)] Upload complete."
echo "[$(date)] Output at: s3://${S3_BUCKET}/revive-models/"

# ── Self-terminate ────────────────────────────────────────────────────────────
INSTANCE_ID="$(curl -s http://169.254.169.254/latest/meta-data/instance-id)"
echo "[$(date)] Terminating instance ${INSTANCE_ID}..."
aws ec2 terminate-instances --instance-ids "${INSTANCE_ID}" --region "${AWS_REGION}"
