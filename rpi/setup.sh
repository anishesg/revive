#!/bin/bash
# REVIVE Raspberry Pi Mesh Aggregator Setup
# Tested on: Raspberry Pi 4 (4GB) and Pi 5 (8GB) with Raspberry Pi OS (64-bit)
#
# Usage:
#   curl -sSL <raw-url>/rpi/setup.sh | bash
#   OR
#   bash setup.sh
#
# This installs llama.cpp, downloads models, and sets up the mesh aggregator service.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REVIVE_DIR="$HOME/revive"
MODEL_DIR="$REVIVE_DIR/models"
VENV_DIR="$REVIVE_DIR/venv"

echo "╔══════════════════════════════════════════════════╗"
echo "║        REVIVE — Raspberry Pi Setup               ║"
echo "║        Mesh Aggregator + Worker Node              ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""

# ── 1. System dependencies ───────────────────────────────────────────────────
echo "[1/6] Installing system dependencies..."
sudo apt-get update -qq
sudo apt-get install -y -qq \
    build-essential cmake git python3 python3-pip python3-venv \
    avahi-daemon avahi-utils libnss-mdns \
    libcurl4-openssl-dev 2>/dev/null

sudo systemctl enable avahi-daemon
sudo systemctl start avahi-daemon
echo "  ✓ System packages installed"

# ── 2. Build llama.cpp ───────────────────────────────────────────────────────
echo "[2/6] Building llama.cpp..."
if [ ! -d "$REVIVE_DIR/llama.cpp" ]; then
    git clone --depth 1 https://github.com/ggerganov/llama.cpp "$REVIVE_DIR/llama.cpp"
fi

cd "$REVIVE_DIR/llama.cpp"
git pull --ff-only 2>/dev/null || true

cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_OPENMP=ON \
    -DBUILD_SHARED_LIBS=OFF \
    -DLLAMA_CURL=OFF
cmake --build build --config Release -j$(nproc)
echo "  ✓ llama.cpp built"

# ── 3. Python virtual environment ────────────────────────────────────────────
echo "[3/6] Setting up Python environment..."
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
pip install -q --upgrade pip
pip install -q zeroconf aiohttp huggingface_hub psutil
echo "  ✓ Python environment ready"

# ── 4. Download models ───────────────────────────────────────────────────────
echo "[4/6] Downloading models..."
mkdir -p "$MODEL_DIR"

TOTAL_RAM_MB=$(free -m | awk '/^Mem:/{print $2}')
echo "  Detected RAM: ${TOTAL_RAM_MB}MB"

if [ "$TOTAL_RAM_MB" -ge 6000 ]; then
    echo "  Downloading Qwen3-1.7B (aggregator model)..."
    python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='Qwen/Qwen3-1.7B-GGUF',
    filename='qwen3-1.7b-q4_k_m.gguf',
    local_dir='$MODEL_DIR'
)
print('  ✓ Qwen3-1.7B downloaded')
"
    AGGREGATOR_MODEL="qwen3-1.7b-q4_k_m.gguf"
else
    echo "  Downloading Qwen3-0.6B (aggregator model — limited RAM)..."
    python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='Qwen/Qwen3-0.6B-GGUF',
    filename='qwen3-0.6b-q4_k_m.gguf',
    local_dir='$MODEL_DIR'
)
print('  ✓ Qwen3-0.6B downloaded')
"
    AGGREGATOR_MODEL="qwen3-0.6b-q4_k_m.gguf"
fi

# ── 5. Copy aggregator code ─────────────────────────────────────────────────
echo "[5/6] Installing aggregator..."
mkdir -p "$REVIVE_DIR/aggregator"

if [ -f "$SCRIPT_DIR/aggregator.py" ]; then
    cp "$SCRIPT_DIR/aggregator.py" "$REVIVE_DIR/aggregator/"
    cp "$SCRIPT_DIR/discovery.py" "$REVIVE_DIR/aggregator/"
    cp "$SCRIPT_DIR/worker.py" "$REVIVE_DIR/aggregator/"
    cp "$SCRIPT_DIR/metrics.py" "$REVIVE_DIR/aggregator/"
else
    echo "  (aggregator scripts should be in $REVIVE_DIR/aggregator/)"
fi
echo "  ✓ Aggregator installed"

# ── 6. Create systemd service ────────────────────────────────────────────────
echo "[6/6] Creating systemd service..."
sudo tee /etc/systemd/system/revive-aggregator.service > /dev/null << EOF
[Unit]
Description=REVIVE Mesh Aggregator
After=network-online.target avahi-daemon.service
Wants=network-online.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$REVIVE_DIR/aggregator
Environment=REVIVE_MODEL_PATH=$MODEL_DIR/$AGGREGATOR_MODEL
Environment=REVIVE_LLAMA_SERVER=$REVIVE_DIR/llama.cpp/build/bin/llama-server
Environment=REVIVE_PORT=9090
ExecStart=$VENV_DIR/bin/python3 $REVIVE_DIR/aggregator/aggregator.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable revive-aggregator
echo "  ✓ Systemd service created"

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║                 Setup Complete!                   ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""
echo "  Model: $AGGREGATOR_MODEL"
echo "  Port:  9090"
echo ""
echo "  Start:   sudo systemctl start revive-aggregator"
echo "  Status:  sudo systemctl status revive-aggregator"
echo "  Logs:    journalctl -u revive-aggregator -f"
echo ""
echo "  Manual:  source $VENV_DIR/bin/activate && python3 $REVIVE_DIR/aggregator/aggregator.py"
echo ""
echo "  The aggregator will auto-discover all REVIVE workers on the LAN."
echo "  Web dashboard: http://$(hostname -I | awk '{print $1}'):9090"
