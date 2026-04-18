#!/bin/bash
# REVIVE — One-time project setup
# Run from: /Users/anish/Desktop/revive/
#
# Prerequisites: Xcode 15+, Homebrew, xcodegen (brew install xcodegen)
# This script builds llama.xcframework and regenerates the Xcode project.

set -e

LLAMA_DIR="/Users/anish/Desktop/llama.cpp"
REVIVE_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== REVIVE Setup ==="
echo "llama.cpp: $LLAMA_DIR"
echo "project:   $REVIVE_DIR"
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

# ── 2. Build llama.xcframework ────────────────────────────────────────────────
XCFW="$LLAMA_DIR/build-apple/llama.xcframework"
if [ -d "$XCFW" ]; then
  echo "✓ llama.xcframework already built"
else
  echo "⚙  Building llama.xcframework (15-20 min)..."
  cd "$LLAMA_DIR"
  bash build-xcframework.sh
  echo "✓ llama.xcframework built"
fi

# ── 3. Generate Xcode project ─────────────────────────────────────────────────
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
