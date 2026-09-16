#!/usr/bin/env bash
# Downloads the quantized model the app expects. Run once before the first build.
set -euo pipefail
cd "$(dirname "$0")/.."

REPO="tiiuae/Falcon-H1-Tiny-90M-Instruct-GGUF"
FILE="Falcon-H1-Tiny-90M-Instruct-Q4_K_M.gguf"
SHA256="e7d655345a876ecb073ea86f5ce68e412b99090c34a6bde8a2982bc48afb2612"

mkdir -p models
if [ -f "models/$FILE" ] && shasum -a 256 "models/$FILE" | grep -q "$SHA256"; then
  echo "models/$FILE already present and verified."
  exit 0
fi

echo "Downloading $FILE (~59 MB) from $REPO ..."
curl -fL --progress-bar -o "models/$FILE" \
  "https://huggingface.co/$REPO/resolve/main/$FILE"

echo -n "Verifying checksum... "
if shasum -a 256 "models/$FILE" | grep -q "$SHA256"; then
  echo "ok"
else
  echo "MISMATCH"
  echo "Expected $SHA256"
  shasum -a 256 "models/$FILE"
  exit 1
fi
