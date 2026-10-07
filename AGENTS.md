# Notes for coding agents

This file is for automated contributors (and humans who want the short version). The README
documents the library for users; [ROADMAP.md](ROADMAP.md) says what comes next; the
[CHANGELOG](CHANGELOG.md) records what changed.

## What the project is

Spingalett is a neural-network library in C23: training (dense, convolutional, pooling, batch
normalization layers), deployment models in FP32 down to INT2, a standalone inference engine for
microcontrollers (one C file, no heap), Python bindings over ctypes, and the DigitPad demo app.
Its promise is speed: **a change must not make anything slower**, and kernels are measured, not
assumed (see [Performance work](#performance-work)).

## Layout

| Path | Contents |
|---|---|
| `Include/Spingalett/Spingalett.h` | Public training and deployment API (`Spingalett.Inference.h`: the engine's API, included by it) |
| `Src/Spingalett.Private.h`, `Src/Spingalett.Network.h` | Internal declarations; the network is opaque to users |
| `Src/Spingalett.Engine.h` | Internals shared by the engine and the desktop library (layer table, kernels, scratch rules) |
| `Src/Spingalett.Inference.c` | The standalone engine: `.slett` parsing, per-sample kernels for every precision. Must stay free of heap, stdio, files and OpenMP (`model.engine_symbols` checks) and compile as C99 |
| `Src/Spingalett.Model.c` | Batched deployment models (`spingalett_model_predict`) |
| `Src/Spingalett.Int8Tiles.c` | Batched INT8 tile kernels (AVX-512 VNNI, AVX-VNNI, Arm dot product) |
| `Src/Spingalett.GEMM.c` | Float matrix kernels, including implicit im2col and epilogues |
| `Src/Kernels/` | Files that recompile a kernel source with other instruction sets for run-time dispatch |
| `Src/Spingalett.Training.c`, `Batch.c`, `Conv.c`, `Norm.c` | Training loop, batched forward/backward, convolution and normalization layers |
| `Src/Spingalett.Serialize.c`, `docs/ModelFormat.md` | `.slett` files; `DatasetFile.c` and `docs/DatasetFormat.md` for `.slettd` |
| `Tests/` | `Spingalett.Tests.c` (groups, see below), `EngineTests.c`, `GemmTests.c`, Python tests, `Layout.c` |
| `Examples/`, `Apps/DigitPad/`, `Bindings/Python/` | Examples and tools, the demo app, the bindings |

## Build and test

```sh
cmake -S . -B Build -DCMAKE_BUILD_TYPE=Release -DBUILD_WITH_OPENMP=ON
cmake --build Build --parallel
ctest --test-dir Build --output-on-failure          # all groups, about 10 s
Bin/SpingalettTests model                           # one group: grad conv norm equiv cont optim
                                                    # sched dropout gen predict valid step data io model xor
```

Executables go to `Bin/` and static libraries to `Lib/` in the source tree. Point extra build
trees elsewhere (`-DSPINGALETT_BIN_DIR=... -DSPINGALETT_LIB_DIR=...`), or they overwrite the main
build's binaries. The default build uses `-march=native`; `-DSPINGALETT_NATIVE_ARCH=OFF` with
`-DCMAKE_C_FLAGS=-march=x86-64-v3` gives the release configuration with run-time kernel dispatch.

Before a pull request, run what CI runs (`.github/workflows/ci.yml`) that the change can affect:

- GCC and Clang, minimal (no OpenMP) and full (`-DBUILD_WITH_OPENMP=ON -DBUILD_WITH_OPENBLAS=ON`);
- a baseline build (`-DSPINGALETT_NATIVE_ARCH=OFF`), which runs the portable and dispatched kernels;
- AddressSanitizer and UndefinedBehaviorSanitizer (Debug, `-fsanitize=address,undefined`);
- the engine alone: `cc -std=c99 -Wall -Wextra -Wpedantic -Werror -DSPINGALETT_INFERENCE_ONLY -IInclude -ISrc -c Src/Spingalett.Inference.c`;
- for SIMD changes, every instruction set the code has a path for. AArch64 cross builds run under
  `qemu-aarch64` (`QEMU_CPU=cortex-a53` for no dot product, `max` for all features); MinGW builds
  run under Wine; `Examples/Embedded/run-qemu.sh` runs the engine on a Cortex-M4.

## Invariants the tests hold you to

- **Determinism.** Training gives the same bits on any number of threads and single-threaded with
  the built-in kernels: reductions run over chunks fixed by the shape, never by the thread count,
  and partial sums are added in a fixed order. Random draws (shuffling, dropout, augmentation) come
  from a seed and the sample's position, not from a shared generator.
- **Integer inference is exact.** Batched prediction of integer models equals single runs
  (`spingalett_model_run`) bit for bit, on every backend and platform. A new integer kernel must
  produce the same 32-bit sums; tests compare with `== 0`, not with a tolerance.
- **IEEE order in inference.** `Inference.c` and `Model.c` are compiled without reassociation or
  contraction; epilogues use `spingalett_int_output()` so every path rounds alike.
- **Formats.** `.slett` versions 3 to 5 stay loadable; the writer picks the lowest version that
  can hold the network. Any change to the format updates `docs/ModelFormat.md` and adds a version.
- **ABI with Python.** `Tests/Spingalett.Layout.c` and `test_python_layout.py` check that the
  ctypes structures match the C ones; a new field in a public struct needs both sides.
- **Engine scratch.** The engine's workspace size comes from `slett_conv_scratch()`; kernels may
  only use what it reserves.

## Performance work

- Measure on the machine at hand with interleaved runs (the cloud VMs are shared and vary by up to
  20%): `Bin/Benchmark [threads]` against `python Examples/benchmark_pytorch.py [threads]`, and a
  build of the previous commit (a `git worktree`) for before/after numbers. Report medians.
- Read the assembly of hot loops. GCC tends to keep arrays of vector accumulators in memory and its
  partial redundancy elimination copies `dpbusd` accumulators on every step: kernels name their
  accumulators and carry `optimize("no-tree-pre")` (see `Int8Tiles.c`). Check for stack traffic in
  the loop before trusting a benchmark.
- New instruction-set paths need a dispatch story for builds without `-march=native`: compile the
  kernel again through `Src/Kernels/` with the right flags and choose at run time, as the GEMM and
  INT8 tile kernels do.
- Numbers quoted in README and CHANGELOG are measured, with the machine named.

## Style

- C23 for the library, C99 for `Inference.c`. Four-space indentation, lines up to about 120
  columns, `snake_case` with the `spingalett_` prefix for public names, `slett_` for format
  helpers, file-local names without a prefix.
- Every file starts with the SPDX header (`MIT`, copyright pka_human).
- Comments say what a block computes and why, in full sentences; match the density of the code
  around them. Documentation is formal English and states facts, not intentions.
- Public API changes go to `Spingalett.h` with a comment, the README, the Python bindings and their
  README, and the CHANGELOG (breaking changes under **Changed**). Before 1.0 a minor release may
  break the API; the soname carries the minor version.
- Commits are small and topical, with a subject line in the imperative and a body that says what
  changed and why (with measurements for performance work).

## Releases

The version lives in `CMakeLists.txt` (`project(... VERSION ...)`), `Bindings/Python/pyproject.toml`
and `Bindings/Python/spingalett.py`. `.github/workflows/release.yml` builds packages for Linux
(x86-64, x86-64-v3, AArch64), Windows and macOS (universal, with OpenMP), trains the DigitPad model
(2 epochs on pull requests, 30 for releases) and attaches the AppImage and the Windows zip. The
maintainer merges pull requests and pushes tags.
