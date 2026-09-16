#!/usr/bin/env bash
# Builds upstream llama.cpp for watchOS and stages it into vendor/llamacpp.
#
# One one-line CMake patch, plus two build flags.
#
# THE PATCH (ggml/cmake/common.cmake): ggml decides whether it is on ARM with
#   if (CMAKE_OSX_ARCHITECTURES STREQUAL "arm64" ...)
# STREQUAL is exact, so "arm64_32" does not match and ggml silently falls back
# to GENERIC SCALAR kernels on Apple Watch — it prints only
#   "Unknown CPU architecture. Falling back to generic implementations."
# Changing STREQUAL to MATCHES "^arm64" gets the real NEON kernels.
#
# The two flags:
#
#   -D_DARWIN_C_SOURCE     raises __DARWIN_C_LEVEL to __DARWIN_C_FULL, which is
#                          what exposes the BSD types (u_char/u_int/u_short) that
#                          Apple's own proc.h, sysctl.h and ucred.h need, AND
#                          _SC_PHYS_PAGES, which sits behind the same gate.
#                          One flag, both problems.
#
#   -DLLAMA_SUBPROCESS=OFF watchOS forbids posix_spawn. llama.cpp already has
#                          this flag; its CMake check lists iOS, Android and
#                          Emscripten but not watchOS.
#
# Device builds need BOTH arm64_32 and arm64 slices, so the static libraries
# are lipo'd universal or the arm64 slice fails to link.
set -euo pipefail

SRC="${LLAMA_CPP_DIR:-/tmp/llama.cpp}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/vendor/llamacpp"

FLAGS=(
  -G Xcode
  -DCMAKE_SYSTEM_NAME=watchOS
  -DCMAKE_OSX_DEPLOYMENT_TARGET=10.0
  -DCMAKE_C_FLAGS=-D_DARWIN_C_SOURCE
  -DCMAKE_CXX_FLAGS=-D_DARWIN_C_SOURCE
  -DLLAMA_SUBPROCESS=OFF
  -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_APP=OFF
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_CURL=OFF
  -DGGML_METAL=OFF -DGGML_BLAS=OFF -DGGML_ACCELERATE=OFF -DGGML_OPENMP=OFF
  -DBUILD_SHARED_LIBS=OFF
)
LIBS=(libllama libggml libggml-base libggml-cpu)

[ -d "$SRC" ] || git clone --depth 1 https://github.com/ggml-org/llama.cpp "$SRC"
cd "$SRC"
echo "llama.cpp @ $(git log -1 --format=%h)"

# Idempotent: teach ggml that arm64_32 is ARM.
if grep -q 'CMAKE_OSX_ARCHITECTURES      STREQUAL "arm64"' ggml/cmake/common.cmake; then
  sed -i.bak 's|if (CMAKE_OSX_ARCHITECTURES      STREQUAL "arm64" OR|if (CMAKE_OSX_ARCHITECTURES      MATCHES "^arm64" OR|' \
      ggml/cmake/common.cmake
  echo "  patched ggml/cmake/common.cmake (arm64_32 now detected as ARM)"
else
  echo "  arch-detection patch already applied"
fi

build() { # sysroot arch tag
  rm -rf "build-$3"
  local arch_line
  arch_line=$(cmake -B "build-$3" "${FLAGS[@]}" -DCMAKE_OSX_SYSROOT="$1" \
                -DCMAKE_OSX_ARCHITECTURES="$2" 2>&1 | grep -E "GGML_SYSTEM_ARCH|Unknown CPU" | head -1)
  echo "    $3: ${arch_line# *}"
  cmake --build "build-$3" --config Release -- -quiet >/dev/null
  echo "  built $3 ($2)"
}
build watchos          arm64_32 dev32
build watchos          arm64    dev64
build watchsimulator   arm64    sim

mkdir -p "$OUT/include" "$OUT/lib/device" "$OUT/lib/simulator"
cp include/llama.h ggml/include/*.h "$OUT/include/"

for L in "${LIBS[@]}"; do
  lipo -create "$(find build-dev32 -name "$L.a" | head -1)" \
               "$(find build-dev64 -name "$L.a" | head -1)" \
       -output "$OUT/lib/device/$L.a"
  cp "$(find build-sim -name "$L.a" | head -1)" "$OUT/lib/simulator/$L.a"
done

echo "staged into vendor/llamacpp:"
for L in "${LIBS[@]}"; do
  printf "  %-16s %s\n" "$L.a" "$(lipo -info "$OUT/lib/device/$L.a" | sed 's/.*are: //;s/.*is architecture: //')"
done
