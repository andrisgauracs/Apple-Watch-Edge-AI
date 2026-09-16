# Running a language model on an Apple Watch Series 6

A build log, written so you can follow it start to finish. Every number was
measured on the machine that produced it; where something is an estimate rather
than a measurement, it says so.

Target: Apple Watch Series 6 (S6 SiP, dual-core ~1.8 GHz, 1 GB RAM), watchOS 26.
Host: a MacBook with Xcode 26.

---

## Step 0 — Decide what runs the model

Three options: convert to **Core ML**, cross-compile **llama.cpp**, or write your
own engine. I tried all three, in the worst possible order. Here is the right one.

### Core ML is out, and the reason is the model

Falcon-H1 is a *hybrid*. Each of its 24 blocks runs a Mamba-2 state-space mixer
and grouped-query attention **in parallel** on the same normalized input, then
sums them. The Mamba-2 half is a selective scan: a fixed-size state updated one
token at a time, where each step depends on the previous one.

Core ML has no state-space primitive. You would either unroll the scan at a fixed
sequence length — locking you to one context size — or carry the state yourself
through a conversion pipeline whose failure mode is a graph that converts
perfectly and then produces subtly wrong numbers. Core ML on watchOS also has no
GPU compute path, and you write the token loop yourself either way.

That architecture is worth understanding, because it is *why* a watch is a good
fit. The attention half needs a KV cache growing at **24 KB per token**. The
Mamba-2 half carries a **constant-size state no matter how long the conversation
runs**. Half the model's mixing is free, memory-wise, forever.

### Just use llama.cpp

llama.cpp's own `build-xcframework.sh` builds macOS, iOS, tvOS and visionOS —
**watchOS is not on that list**, and nothing in the project claims support. So it
looks like unexplored territory. It isn't blocked, though: see Step 3.

I wrote my own 1,800-line C engine before checking this. It worked, and it was
**4.4× slower** than llama.cpp. Writing your own is an excellent way to
understand the problem and a poor way to ship. That engine is gone from this repo.

---

## Step 1 — Environment setup

```bash
xcodebuild -version                          # Xcode 26
xcrun --sdk watchos --show-sdk-version       # 26.5+
```

**This breaks first.** The watchOS SDK being present does *not* mean Xcode can
build for watchOS. The platform components are a separate ~4 GB download:

```
xcodebuild: error: Unable to find a destination matching { generic:1, platform:watchOS }
    error: watchOS is not installed. Please download and install the platform
           from Xcode > Settings > Components.
```

```bash
xcodebuild -downloadPlatform watchOS
```

---

## Step 2 — Get the models

Both models ship as published GGUF files, so **there is no conversion step
anywhere in this project**. The app reads the `.gguf` unchanged.

```bash
./tools/fetch_model.sh
```

| model | quant | size |
|---|---|---|
| Falcon-H1-Tiny-90M-Instruct | Q4_K_M | 57 MB |
| SmolLM2-135M-Instruct | Q4_K_M | 104 MB |

Inspect one before trusting it:

```bash
python3 tools/gguf_dump.py models/Falcon-H1-Tiny-90M-Instruct-Q4_K_M.gguf
```

Falcon-H1's hybrid design is visible in a single block — attention and
state-space tensors side by side:

```
blk.0.attn_q.weight     [512, 512]    Q4_K
blk.0.attn_k.weight     [512, 128]    Q4_K
blk.0.ssm_in.weight     [512, 1688]   Q4_K
blk.0.ssm_conv1d.weight [4, 896]      F32
```

---

## Step 3 — Build llama.cpp for watchOS

```bash
./tools/build_llamacpp.sh
```

That script encodes everything below. The libraries are already committed, so you
only need it to regenerate them or bump versions.

### What arm64_32 actually is

watchOS does not use standard `arm64`. It uses `arm64_32`, and the name makes it
sound like a cut-down half-speed ARM. It isn't:

```bash
xcrun --sdk watchos clang -target arm64_32-apple-watchos11.0 \
      -dM -E -x c /dev/null | grep -E '__ARM_ARCH |__ARM_NEON |__SIZEOF_POINTER__'
```
```
#define __ARM_ARCH 8            ← full 64-bit instruction set
#define __ARM_NEON 1            ← full 128-bit vector unit
#define __SIZEOF_POINTER__ 4    ← 32-bit pointers
```

The instruction set is intact, NEON included. Only the pointers shrank. There is
**no conversion step** from arm64 to arm64_32 — it is a compiler target, so the
same source compiles twice.

The one real consequence: 32-bit addresses span 4 GB, and a single app gets
rather less. **A watch can never run a large model regardless of quantization.**
57 MB is nowhere near that ceiling.

### Two flags make it build

```bash
-DCMAKE_C_FLAGS=-D_DARWIN_C_SOURCE -DCMAKE_CXX_FLAGS=-D_DARWIN_C_SOURCE
-DLLAMA_SUBPROCESS=OFF
```

`_DARWIN_C_SOURCE` raises `__DARWIN_C_LEVEL` to `__DARWIN_C_FULL`, which exposes
the BSD types (`u_char`/`u_int`/`u_short`) that **Apple's own** `proc.h`,
`sysctl.h` and `ucred.h` need — and `_SC_PHYS_PAGES`, which sits behind the same
gate and which `ggml-cpu.cpp` calls. One flag, both problems.

`LLAMA_SUBPROCESS=OFF` because watchOS forbids `posix_spawn`. llama.cpp already
has this flag; its CMake check lists iOS, Android and Emscripten but not watchOS.

### The one-line fix that matters most

With just those flags it **builds, links and runs** — and is needlessly slow. The
only clue is one line in the build log:

```
CMake Warning: Unknown CPU architecture. Falling back to generic implementations.
```

`ggml/cmake/common.cmake` decides whether it is on ARM with:

```cmake
if (CMAKE_OSX_ARCHITECTURES STREQUAL "arm64" OR ...)
```

`STREQUAL` is exact string equality, so **`arm64_32` does not match**. ggml falls
through to `UNKNOWN` and compiles its **generic scalar kernels**, throwing away
the entire NEON path on a device whose whole value proposition is that it has one.

```diff
-    if (CMAKE_OSX_ARCHITECTURES      STREQUAL "arm64" OR
+    if (CMAKE_OSX_ARCHITECTURES      MATCHES "^arm64" OR
```

You can see the difference in the object files. Before, one `quants.o` (generic).
After, two (generic *and* ARM), and real vector code in the `arm64_32` library:

| object | vector instructions |
|---|---|
| `sgemm.o` | 896 |
| `repack.o` | 498 |
| `quants.o` | 224 |

```
fmla.4s  v2, v3, v1     ← a 4-wide FMA, inside a watchOS object file
```

This is an upstream bug worth reporting.

---

## Step 4 — Link it into the app

Two traps, both of which produce a *successful build*.

**ggml's CPU backend disappears.** It registers itself from a static initializer,
which the linker discards from a static library. The app links cleanly and has no
backend at runtime. You need:

```
-Wl,-force_load,$(SRCROOT)/vendor/llamacpp/lib/device/libggml-cpu.a
```

**Device builds need two slices.** Xcode emits both `arm64_32` and `arm64` for
watchOS, so the static libraries must be `lipo`'d universal, or the arm64 slice
fails with `symbol(s) not found for architecture arm64`.

---

## Step 5 — Wire it into SwiftUI

Three files: `LlamaCppEngine.swift` (owns the `llama_context`),
`LLMRunner.swift` (drives it, publishes state), `ContentView.swift`.

**Concurrency.** The engine is queue-confined and `@unchecked Sendable`; the
runner is `@MainActor` and its decode loop is `nonisolated`. The first version
mutated C pointers from a background queue under a `@MainActor` class — it
compiled, with a pile of warnings that were all describing real data races.

**SwiftUI on watchOS is a subset.** `Menu` and `.textSelection` do not exist.
Expect this class of surprise.

**Tokens can end mid-UTF-8.** A token is a byte sequence, not a character, and a
multi-byte scalar can split across two tokens. Appending raw bytes to a `String`
produces replacement characters on screen. `UTF8Stream` buffers the partial tail.

**Throttle the UI.** SwiftUI re-lays-out the whole transcript on every update, so
redraws are backed off as the output grows — otherwise the UI competes with
inference for the same two cores.

---

## Step 6 — Get it onto the Watch

**Developer Mode is a chicken-and-egg.** On watchOS the toggle only appears in
Settings → Privacy & Security *after Xcode has connected once*. You cannot enable
it first.

**The Watch sleeps its Wi-Fi.** It is discovered over the network, not over its
charger — the charging puck is inductive and carries no data. Seconds after the
screen dims the radio powers down; the Watch holds a DHCP lease and answers
nothing. Keep it **on power and awake**, unlocked, on the same network, with
Xcode's Devices window open.

Confirm you actually built for the watch:

```bash
lipo -info WatchLLM.app/WatchLLM     # arm64_32 arm64
```

If installs start failing with *"The developer disk image could not be mounted"*,
that usually means Xcode updated and the old image on the Watch is stale.
Restart the Watch.

---

## Step 7 — Web tools

Four keyless tools: weather, Bitcoin, USD→EUR, Wikipedia. A tool runs **before**
generation and its result is folded into the prompt, so the model itself never
touches the network.

Definitions live in `WatchLLM/Tools/Tools.json`, which is git-ignored — with no
such file the app has zero tools and is fully offline. Test without rebuilding:

```bash
./tools/test_tool.py "what is bitcoin trading at"
```

**Fail closed.** If a tool matches but its fetch fails, the model is never asked.
The first version generated anyway and the model filled the gap with confident
invented statistics, which is the worst possible outcome for a question that
exists only because it needs live data.

---

## Step 8 — What the models actually do with the facts

This is the part no amount of reasoning gets you. All of it was measured.

**A tiny model will ignore a fact in its own context if it has a strong enough
memorized answer.** Handed "It is 20.6 degrees Celsius outside", Falcon-H1
answered with a different number every time. Fact in the system turn, fact in the
user turn, an explicit "answer using only this fact", a maximally direct question
— all four failed.

**What fixed it was relabeling the data, not moving it:**

```
"It is 20.6 degrees Celsius outside."          → Falcon 0/6 phrasings correct
"Live sensor reading: the temperature          → Falcon 6/6
 outside is 20.6 degrees Celsius."
```

A bare statement reads like a claim the model can dispute with its prior. "Live
sensor reading:" reads like data.

**Phrasing of the question matters as much as the fact.** With the plain framing,
SmolLM2 answered correctly for *"What is the temperature outside?"* and refused
for *"What is the temperature outside right now?"* — two extra words trigger the
instruction-tuned "I don't have real-time access" reflex. Asking *"how cold is
it"* made it invent −20 °C.

**Negative instructions backfire.** Adding "Do not say you lack real-time access"
produced: *"No, I'm not lacking real-time access."*

**Don't hand it arithmetic.** Precomputing a ratio to help the model is what
*broke* it — a ratio in context invites more arithmetic, and it produced
`834 * 127,610 = 1056,707,51`. Give it the numbers, not the maths.

**Stop it at one sentence.** Across every failed test the first sentence was
correct and only the continuation invented things. Tool-backed answers now stop
at the first sentence terminator, with a digit guard so `25.7` isn't mistaken for
a sentence end. You don't fix a small model's rambling with better prompting; you
fix it by not letting it ramble.

**Use greedy decoding for facts.** Even at temperature 0.15 the models drifted
onto different digits between runs. Temperature 0 also makes runs reproducible.

**Neither model is better.** They fail on opposite tools — Falcon on weather,
SmolLM2 on currency — with identical inputs.

---

## What I'd tell someone attempting this

- **Check whether the obvious tool already works before building your own.** I
  wrote an entire inference engine, and llama.cpp turned out to run here, faster,
  with two build flags.
- **arm64_32 is not the hard part.** It is the thing everyone expects to be hard,
  and portable NEON C compiles for it unchanged. The hard parts were a hidden
  4 GB download, an app that built successfully with no model inside it, and a
  one-word CMake bug that silently removed every vector instruction.
- **A successful build proves almost nothing.** Three separate problems here
  compiled cleanly and failed later: the missing model resource, the discarded
  ggml backend, and the scalar-kernel fallback.
- **The wall is the model, not the chip.** A Series 6 is not the bottleneck people
  assume. The ceiling is the ~4 GB address space, and a 57 MB model is nowhere
  near it.
