#!/usr/bin/env bash
# Downloads the GGUF models the app bundles. Run once before the first build —
# `make project` needs them present, because they are bundle resources.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p models

# repo|file|sha256
MODELS=(
"tiiuae/Falcon-H1-Tiny-90M-Instruct-GGUF|Falcon-H1-Tiny-90M-Instruct-Q4_K_M.gguf|e7d655345a876ecb073ea86f5ce68e412b99090c34a6bde8a2982bc48afb2612"
"unsloth/SmolLM2-135M-Instruct-GGUF|SmolLM2-135M-Instruct-Q4_K_M.gguf|ed5fa30c487b282ec156c29062f1222e5c20875a944ac98289dbd242e947f747"
)

for ENTRY in "${MODELS[@]}"; do
  IFS='|' read -r REPO FILE SHA <<< "$ENTRY"
  if [ -f "models/$FILE" ] && shasum -a 256 "models/$FILE" | grep -q "$SHA"; then
    echo "$FILE already present and verified."
    continue
  fi
  echo "Downloading $FILE from $REPO ..."
  curl -fL --progress-bar -o "models/$FILE" \
    "https://huggingface.co/$REPO/resolve/main/$FILE"
  printf "Verifying %s... " "$FILE"
  if shasum -a 256 "models/$FILE" | grep -q "$SHA"; then
    echo "ok"
  else
    echo "CHECKSUM MISMATCH"
    echo "  expected $SHA"
    shasum -a 256 "models/$FILE"
    exit 1
  fi
done
echo
echo "Models ready. Next: make project"
