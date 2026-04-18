#!/bin/bash
# REVIVE MacBook Setup
# Works on both Intel and Apple Silicon Macs.
# Apple Silicon gets Metal GPU acceleration automatically.

set -e

REVIVE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_DIR="$REVIVE_DIR/llama.cpp"
MODEL_DIR="$REVIVE_DIR/models"
VENV_DIR="$REVIVE_DIR/.venv"

echo "╔══════════════════════════════════════════════════╗"
echo "║         REVIVE — MacBook Setup                    ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""

# ── 1. Prerequisites ─────────────────────────────────────────────────────────
echo "[1/5] Checking prerequisites..."
for tool in cmake git python3; do
    if ! command -v "$tool" &>/dev/null; then
        echo "  Missing: $tool — install via: brew install $tool"
        exit 1
    fi
done
echo "  ✓ Prerequisites satisfied"

# ── 2. Build llama.cpp ───────────────────────────────────────────────────────
echo "[2/5] Building llama.cpp..."
if [ ! -d "$LLAMA_DIR" ]; then
    git clone --depth 1 https://github.com/ggerganov/llama.cpp "$LLAMA_DIR"
fi

cd "$LLAMA_DIR"
git pull --ff-only 2>/dev/null || true

ARCH=$(uname -m)
if [ "$ARCH" = "arm64" ]; then
    echo "  Apple Silicon detected — enabling Metal acceleration"
    cmake -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_METAL=ON \
        -DBUILD_SHARED_LIBS=OFF
else
    echo "  Intel Mac — CPU-only build"
    cmake -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_METAL=OFF \
        -DBUILD_SHARED_LIBS=OFF
fi

cmake --build build --config Release -j$(sysctl -n hw.ncpu)
echo "  ✓ llama.cpp built"

# ── 3. Python environment ────────────────────────────────────────────────────
echo "[3/5] Setting up Python environment..."
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
pip install -q --upgrade pip
pip install -q aiohttp zeroconf psutil huggingface_hub
echo "  ✓ Python environment ready"

# ── 4. Download model ────────────────────────────────────────────────────────
echo "[4/5] Downloading model..."
mkdir -p "$MODEL_DIR"

TOTAL_RAM_MB=$(sysctl -n hw.memsize | awk '{print int($1/1024/1024)}')
echo "  System RAM: ${TOTAL_RAM_MB}MB"

if [ "$TOTAL_RAM_MB" -ge 12000 ]; then
    MODEL_NAME="qwen3-4b-q4_k_m.gguf"
    echo "  Downloading Qwen3-4B (your Mac can handle it)..."
    python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id='Qwen/Qwen3-4B-GGUF', filename='$MODEL_NAME', local_dir='$MODEL_DIR')
print('  ✓ Qwen3-4B downloaded')
"
elif [ "$TOTAL_RAM_MB" -ge 6000 ]; then
    MODEL_NAME="qwen3-1.7b-q4_k_m.gguf"
    echo "  Downloading Qwen3-1.7B..."
    python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id='Qwen/Qwen3-1.7B-GGUF', filename='$MODEL_NAME', local_dir='$MODEL_DIR')
print('  ✓ Qwen3-1.7B downloaded')
"
else
    MODEL_NAME="qwen3-0.6b-q4_k_m.gguf"
    echo "  Downloading Qwen3-0.6B (low RAM fallback)..."
    python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id='Qwen/Qwen3-0.6B-GGUF', filename='$MODEL_NAME', local_dir='$MODEL_DIR')
print('  ✓ Qwen3-0.6B downloaded')
"
fi

# ── 5. Create launch scripts ─────────────────────────────────────────────────
echo "[5/5] Creating launch scripts..."

cat > "$REVIVE_DIR/macos/start-worker.sh" << SCRIPT
#!/bin/bash
source "$VENV_DIR/bin/activate"
ROLE="\${1:-reasoner}"
echo "Starting REVIVE worker (role: \$ROLE)..."
python3 "$REVIVE_DIR/macos/revive-cli.py" \\
    --mode worker \\
    --role "\$ROLE" \\
    --port 8080 \\
    --model "$MODEL_DIR/$MODEL_NAME" \\
    --llama-server "$LLAMA_DIR/build/bin/llama-server"
SCRIPT
chmod +x "$REVIVE_DIR/macos/start-worker.sh"

cat > "$REVIVE_DIR/macos/start-coordinator.sh" << SCRIPT
#!/bin/bash
source "$VENV_DIR/bin/activate"
echo "Starting REVIVE coordinator..."
python3 "$REVIVE_DIR/macos/revive-cli.py" \\
    --mode coordinator \\
    --port 8080 \\
    --model "$MODEL_DIR/$MODEL_NAME" \\
    --llama-server "$LLAMA_DIR/build/bin/llama-server" \\
    --interactive
SCRIPT
chmod +x "$REVIVE_DIR/macos/start-coordinator.sh"

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║                 Setup Complete!                   ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""
echo "  Model: $MODEL_NAME"
echo "  Metal: $([ "$ARCH" = "arm64" ] && echo "Enabled" || echo "Disabled")"
echo ""
echo "  Worker mode:      bash macos/start-worker.sh [role]"
echo "  Coordinator mode: bash macos/start-coordinator.sh"
echo ""
echo "  Available roles: reasoner, writer, concise, critic, factchecker, drafter"
