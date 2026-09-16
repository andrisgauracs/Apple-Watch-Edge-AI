# Running llama.cpp on watchOS

**It works.** `libllama.a`, `libggml-base.a` and `libggml-cpu.a` all build for
`arm64_32` / watchOS. Reproduce everything with:

```bash
./tools/build_llamacpp.sh
```

## What it takes

**Two CMake flags, and one one-line patch.**

### `-D_DARWIN_C_SOURCE`

Apple's own headers — `proc.h`, `sysctl.h`, `ucred.h` — reference the BSD types
`u_char` / `u_int` / `u_short`, which watchOS hides by default. The same gate
(`__DARWIN_C_LEVEL >= __DARWIN_C_FULL`) also hides `sysconf(_SC_PHYS_PAGES)`,
which `ggml-cpu.cpp` calls.

One flag fixes both. An earlier version of this document claimed a source patch
was needed to guard `_SC_PHYS_PAGES`; that was wrong — it was already fixed by
this flag. Verified by building pristine upstream with nothing but these flags.

### `-DLLAMA_SUBPROCESS=OFF`

watchOS forbids `posix_spawn`. llama.cpp already has this flag for exactly this
reason; its CMake check lists iOS, Android and Emscripten but **not** watchOS.

### The patch that actually matters

`ggml/cmake/common.cmake` decides whether it is building for ARM with:

```cmake
if (CMAKE_OSX_ARCHITECTURES STREQUAL "arm64" OR ...)
```

`STREQUAL` is exact string equality, so **`arm64_32` does not match**. ggml
falls through to `UNKNOWN` and compiles its **generic scalar kernels**. The only
sign is one easily-missed line:

```
CMake Warning: Unknown CPU architecture. Falling back to generic implementations.
```

The build succeeds. The app runs. It is just needlessly slow, because the entire
NEON path has been compiled out.

```diff
-    if (CMAKE_OSX_ARCHITECTURES      STREQUAL "arm64" OR
+    if (CMAKE_OSX_ARCHITECTURES      MATCHES "^arm64" OR
```

With the fix, configure reports `GGML_SYSTEM_ARCH: ARM` / `ARM detected`, the
arch-specific `quants` and `repack` sources compile, and the resulting
`arm64_32` objects contain real vector code:

| object | vector instructions |
|---|---|
| `sgemm.o` | 896 |
| `repack.o` | 498 |
| `quants.o` | 224 |

```
fmla.4s  v2, v3, v1     ← 4-wide fused multiply-add, inside an arm64_32 object
```

This is an upstream bug and worth reporting.

## Integration traps

**ggml's CPU backend vanishes when statically linked.** It registers itself from
a static initialiser, which the linker discards from a static library. Without
`-Wl,-force_load` on `libggml-cpu.a` the app links cleanly and then has no
backend at runtime.

**watchOS device builds need two slices.** Xcode emits both `arm64_32` and
`arm64`, so the static libraries must be `lipo`'d universal or the arm64 slice
fails with "symbol(s) not found for architecture arm64".

## About arm64_32

Full A64 instruction set, 32-bit pointers:

```bash
xcrun --sdk watchos clang -target arm64_32-apple-watchos11.0 \
      -dM -E -x c /dev/null | grep -E '__ARM_ARCH |__ARM_NEON |__SIZEOF_POINTER__'
```
```
#define __ARM_ARCH 8            ← 64-bit instruction set
#define __ARM_NEON 1            ← full 128-bit vector unit
#define __SIZEOF_POINTER__ 4    ← 32-bit pointers
```

There is no conversion step from arm64 to arm64_32 — it is a compiler target.
The one real consequence of 32-bit pointers: a watch cannot address more than
about 2 GB, so it can never run a large model regardless of quantization.
