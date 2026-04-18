#!/data/data/com.termux/files/usr/bin/bash
# REVIVE Android Worker Setup
# Run inside Termux on the Android phone.
# This installs llama.cpp and downloads the worker model.

set -e

ROLE="${ROLE:-drafter}"
MODEL_NAME="${MODEL_NAME:-qwen3-0.6b-q4_k_m.gguf}"
PORT="${PORT:-8080}"

echo "=== REVIVE Android Worker Setup ==="
echo "Role: $ROLE | Model: $MODEL_NAME | Port: $PORT"

# ── 1. System packages ────────────────────────────────────────────────────────
pkg update -y
pkg install -y cmake clang git python3 pip

# ── 2. Build llama.cpp ────────────────────────────────────────────────────────
if [ ! -d "$HOME/llama.cpp" ]; then
  echo "[+] Cloning llama.cpp..."
  git clone --depth 1 https://github.com/ggerganov/llama.cpp "$HOME/llama.cpp"
fi

cd "$HOME/llama.cpp"
git pull --ff-only || true

echo "[+] Building llama.cpp (this takes 5-10 minutes on first run)..."
cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_CURL=OFF \
  -DBUILD_SHARED_LIBS=OFF
cmake --build build --config Release -j$(nproc)

echo "[+] Build complete."

# ── 3. Download model ─────────────────────────────────────────────────────────
mkdir -p "$HOME/models"
MODEL_PATH="$HOME/models/$MODEL_NAME"

if [ ! -f "$MODEL_PATH" ]; then
  echo "[+] Downloading $MODEL_NAME ..."
  pip install -q huggingface_hub

  case "$ROLE" in
    drafter|spotter)
      python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='Qwen/Qwen3-0.6B-GGUF',
    filename='qwen3-0.6b-q4_k_m.gguf',
    local_dir='$HOME/models'
)
print('Downloaded Qwen3-0.6B')
"
      ;;
    reasoner|writer|critic|factchecker)
      python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='Qwen/Qwen3-1.7B-GGUF',
    filename='qwen3-1.7b-q4_k_m.gguf',
    local_dir='$HOME/models'
)
print('Downloaded Qwen3-1.7B')
"
      MODEL_NAME="qwen3-1.7b-q4_k_m.gguf"
      MODEL_PATH="$HOME/models/$MODEL_NAME"
      ;;
    *)
      echo "Unknown role: $ROLE. Downloading TinyLlama as fallback."
      python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF',
    filename='tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf',
    local_dir='$HOME/models'
)
print('Downloaded TinyLlama')
"
      MODEL_NAME="tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
      MODEL_PATH="$HOME/models/$MODEL_NAME"
      ;;
  esac
else
  echo "[+] Model already downloaded: $MODEL_PATH"
fi

# ── 4. Install mDNS advertiser ────────────────────────────────────────────────
pip install -q zeroconf

# ── 5. Write start script ─────────────────────────────────────────────────────
cat > "$HOME/start_worker.sh" << SCRIPT
#!/data/data/com.termux/files/usr/bin/bash
# Starts the llama-server + mDNS advertiser in the background

MODEL_PATH="$MODEL_PATH"
ROLE="$ROLE"
PORT="$PORT"

echo "Starting REVIVE worker: \$ROLE on port \$PORT"

# Start llama-server
"$HOME/llama.cpp/build/bin/llama-server" \\
  -m "\$MODEL_PATH" \\
  --host 0.0.0.0 \\
  --port "\$PORT" \\
  -c 1024 \\
  -ngl 0 \\
  --log-disable &
SERVER_PID=\$!
echo "llama-server PID: \$SERVER_PID"

sleep 3

# Start mDNS advertiser
python3 "$HOME/revive_advertise.py" --role "\$ROLE" --port "\$PORT" &
MDNS_PID=\$!
echo "mDNS advertiser PID: \$MDNS_PID"

# Keep alive; kill both on Ctrl-C
trap "kill \$SERVER_PID \$MDNS_PID 2>/dev/null" EXIT
wait \$SERVER_PID
SCRIPT
chmod +x "$HOME/start_worker.sh"

# ── 6. Write mDNS advertiser ──────────────────────────────────────────────────
cp "$(dirname "$0")/advertise.py" "$HOME/revive_advertise.py"

echo ""
echo "=== Setup complete! ==="
echo ""
echo "To start the worker:"
echo "  bash ~/start_worker.sh"
echo ""
echo "The phone will advertise itself as REVIVE-${ROLE} on the local network."
echo "The coordinator iPad will discover it automatically."
