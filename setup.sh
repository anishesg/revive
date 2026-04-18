#!/bin/bash
# REVIVE — One-time project setup
# Run from: /Users/anish/Desktop/revive/
#
# Prerequisites: Xcode 15+, Homebrew, xcodegen (brew install xcodegen)
# This script builds llama.xcframework and regenerates the Xcode project.

set -e

REVIVE_DIR="$(cd "$(dirname "$0")" && pwd)"
LLAMA_DIR="$REVIVE_DIR/../llama.cpp"

echo "=== REVIVE iOS Setup ==="
echo "project:   $REVIVE_DIR"
echo "llama.cpp: $LLAMA_DIR"
echo ""

# ── 1. Check prerequisites ────────────────────────────────────────────────────
for tool in cmake xcodegen xcodebuild; do
  if ! command -v "$tool" &>/dev/null; then
    echo "Missing: $tool"
    case "$tool" in
      cmake|xcodegen) echo "  Install with: brew install $tool" ;;
      xcodebuild) echo "  Install Xcode from the App Store" ;;
    esac
    exit 1
  fi
done
echo "✓ Prerequisites satisfied"

# ── 2. Clone llama.cpp if needed ──────────────────────────────────────────────
if [ ! -d "$LLAMA_DIR" ]; then
  echo "Cloning llama.cpp..."
  git clone --depth 1 https://github.com/ggerganov/llama.cpp "$LLAMA_DIR"
fi

# ── 3. Build llama.xcframework ────────────────────────────────────────────────
XCFW="$LLAMA_DIR/build-apple/llama.xcframework"
if [ -d "$XCFW" ]; then
  echo "✓ llama.xcframework already built"
else
  echo "Building llama.xcframework (15-20 min)..."
  cd "$LLAMA_DIR"
  bash build-xcframework.sh
  echo "✓ llama.xcframework built"
fi

# ── 4. Generate Xcode project ─────────────────────────────────────────────────
cd "$REVIVE_DIR"
xcodegen generate
echo "✓ revive.xcodeproj generated"

echo ""
echo "=== Setup complete! ==="
echo ""
echo "Next steps:"
echo "  1. open revive.xcodeproj"
echo "  2. Select your team in Signing & Capabilities"
echo "  3. Build and run on each device"
echo ""
echo "Worker phones:     run in Worker mode, place GGUF model in Documents/"
echo "Coordinator iPad:  run in Coordinator mode, same WiFi network"
echo "Web dashboard:     open http://<iPad-IP>:8080 in any browser"
