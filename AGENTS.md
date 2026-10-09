# Notes for coding agents

This file is for automated contributors (and humans who want the short version). The README
documents the library for users; [ROADMAP.md](ROADMAP.md) says what comes next; the
[CHANGELOG](CHANGELOG.md) records what changed.

## What the project is

Spingalett is a neural-network library in C23: training (dense, convolutional, pooling, batch
normalization, adding and concatenating layers, as chains or graphs), deployment models in FP32 down
to INT2, the runtime (a library of the deployment models alone), a standalone inference engine for
microcontrollers (one C file, no heap), ONNX and PyTorch import, GPU backends through CUDA (its own
kernels) and Vulkan compute, Python bindings over ctypes (wheels with the library inside), and the
DigitPad demo app.
Its promise is speed: **a change must not make anything slower**, and kernels are measured, not
assumed (see [Performance work](#performance-work)).

## Layout

| Path | Contents |
|---|---|
| `Include/Spingalett/Spingalett.h` | Public training and deployment API; it includes `Spingalett.Runtime.h` (the runtime's API: versions, errors, settings, deployment models), which includes `Spingalett.Inference.h` (the engine's) |
| `Src/Spingalett.Private.h`, `Src/Spingalett.Network.h` | Internal declarations; the network is opaque to users |
| `Src/Spingalett.Engine.h` | Internals shared by the engine and the desktop library (layer table, kernels, scratch rules) |
| `Src/Spingalett.Inference.c` | The standalone engine: `.slett` parsing, per-sample kernels for every precision. Must stay free of heap, stdio, files and OpenMP (`model.engine_symbols` checks) and compile as C99 |
| `Src/Spingalett.Model.c` | Batched deployment models (`spingalett_model_predict`): weights prepared once per owned model, a workspace kept between calls |
| `cmake/Runtime.cmake`, `Src/Spingalett.Runtime.map` | The runtime library: the sources that load and run models, compiled with the library's flags (`spingalett_build`) and `SPINGALETT_RUNTIME`, its exports and their symbol versions |
| `Src/Spingalett.Int8Tiles.c` | Batched INT8 tile kernels (AVX-512 VNNI, AVX-VNNI, Arm dot product) |
| `Src/Spingalett.GEMM.c` | Float matrix kernels, including implicit im2col and epilogues |
| `Src/Spingalett.ConvGEMM.c` | Convolutions as indirect products: windows read through per-tap pointers (forward, data and weight gradients) |
| `Src/Spingalett.Graph.c` | Adding, concatenating and global pooling layers, upsampling; the plan by which inference outputs share memory |
| `Src/Spingalett.Onnx.c`, `Src/Spingalett.Torch.c` | ONNX import (own protocol buffer reader, external data files); PyTorch state dicts (zip, a pickle interpreter that runs nothing) and safetensors |
| `Src/Spingalett.Import.c` | What the importers share: files mapped into memory, tensors decoded and reordered to channels-last in one pass |
| `Src/Kernels/` | Files that recompile a kernel source with other instruction sets for run-time dispatch |
| `Src/Spingalett.Training.c`, `Batch.c`, `Conv.c`, `Norm.c` | Training loop, batched forward/backward over the graph, convolution (and transposed convolution: the convolution passes swapped) and normalization (batch, layer) layers |
| `Src/Spingalett.Serialize.c`, `docs/ModelFormat.md` | `.slett` files; `DatasetFile.c` and `docs/DatasetFormat.md` for `.slettd` (coders, readers) |
| `Src/Spingalett.Thread.c` | A portable thread, lock and condition (POSIX threads or Win32), used by the data set reader |
| `Src/Gpu/` | The GPU backends under one executor: `Spingalett.Gpu.c` (a network on the GPU: the batch path recorded as commands), `Spingalett.GpuKernels.c` (the matrix product's tiles, timed on first use, a table a backend; convolution geometries), `Spingalett.Device.h`/`.c` (the device interface: `SpgGpuOps` a backend, the calling thread's backend), `Spingalett.Vulkan.c` (the loader opened at run time; `Shaders/*.comp`, GLSL listed in `Spingalett.Kernels.def`, `SPG_KERNEL_H` ones built a second time with `SPG_HALF` for buffers kept as bfloat16, `half.glsl`), `Spingalett.Cuda.c` (the driver opened at run time; commands captured as CUDA graphs; `Cuda/*.cu` compiled to PTX by Clang, units listed in `Cuda/Kernels.def`, gzip-compressed into the library and expanded by `Spingalett.Gunzip.c`); `Src/Spingalett.Gpu.h` is what the rest of the library calls |
| `Tests/` | `Spingalett.Tests.c` (groups, see below), `EngineTests.c`, `GemmTests.c`, `GpuTests.c` (the GPU's matrix kernel; `bench` times every tile), `RuntimeTests.c` (the runtime alone, C99), Python tests, `Layout.c`; `make_test_models.py` writes the PyTorch models of `Tests/Data` |
| `Examples/`, `Apps/DigitPad/`, `Bindings/Python/spingalett/` | Examples and tools (`Examples/Runtime/`: programs of the runtime), the demo app, the bindings (a package; `setup.py` builds wheels) |

## Build and test

```sh
cmake -S . -B Build -DCMAKE_BUILD_TYPE=Release -DBUILD_WITH_OPENMP=ON
cmake --build Build --parallel
ctest --test-dir Build --output-on-failure          # all groups, about 10 s
Bin/SpingalettTests model                           # one group: grad conv norm graph onnx equiv cont optim
                                                    # sched dropout gen predict valid step data io model gpu cuda xor
```

Executables go to `Bin/` and static libraries to `Lib/` in the source tree. Point extra build
trees elsewhere (`-DSPINGALETT_BIN_DIR=... -DSPINGALETT_LIB_DIR=...`), or they overwrite the main
build's binaries. The default build uses `-march=native`; `-DSPINGALETT_NATIVE_ARCH=OFF` with
`-DCMAKE_C_FLAGS=-march=x86-64-v3` gives the release configuration with run-time kernel dispatch.

Before a pull request, run what CI runs (`.github/workflows/ci.yml`) that the change can affect:

- GCC and Clang, minimal (no OpenMP) and full (`-DBUILD_WITH_OPENMP=ON -DBUILD_WITH_OPENBLAS=ON`);
- a baseline build (`-DSPINGALETT_NATIVE_ARCH=OFF`), which runs the portable and dispatched kernels;
- AddressSanitizer and UndefinedBehaviorSanitizer (Debug, `-fsanitize=address,undefined`), with GCC
  and with Clang, whose UBSan also catches offsets applied to null pointers
  (`LSAN_OPTIONS=suppressions=Tests/lsan.supp`: LLVM's OpenMP runtime keeps memory until exit);
- the engine alone: `cc -std=c99 -Wall -Wextra -Wpedantic -Werror -DSPINGALETT_INFERENCE_ONLY -IInclude -ISrc -c Src/Spingalett.Inference.c`;
- the runtime alone (`-DSPINGALETT_RUNTIME_ONLY=ON`) for changes to the files it is built from
  (`cmake/Runtime.cmake`) or to its header;
- for SIMD changes, every instruction set the code has a path for. AArch64 cross builds run under
  `qemu-aarch64` (`QEMU_CPU=cortex-a53` for no dot product, `max` for all features); MinGW builds
  run under Wine; `Examples/Embedded/run-qemu.sh` runs the engine on a Cortex-M4. AVX-512 paths on
  machines without it: Intel's SDE (`sde64 -spr -- Bin/SpingalettTests conv`) on a build with
  run-time dispatch (`-DSPINGALETT_NATIVE_ARCH=OFF`);
- for GPU changes, `Bin/SpingalettGpuTests all` and `Bin/SpingalettTests gpu` on a GPU (both skip
  without a Vulkan device), and a build with `-DSPINGALETT_VULKAN=OFF`. CI runs the GPU tests on
  Mesa's lavapipe, a CPU device with 32 KB of shared memory a workgroup and another subgroup size:
  kernels must not assume either. For the CUDA backend, `SPINGALETT_GPU_BACKEND=cuda
  Bin/SpingalettGpuTests all` and `Bin/SpingalettTests cuda` (the `gpu` group on CUDA) on an NVIDIA GPU,
  and a build with `-DSPINGALETT_CUDA=OFF`; CI has no NVIDIA GPU, so these run on the developer's
  machine only, while CI compiles every unit (Clang with the NVPTX target).

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
- **Formats.** `.slett` versions 3 to 7 stay loadable; the writer picks the lowest version that
  can hold the network (chains 3 to 5, graphs 6, transposed convolutions, upsampling and layer
  normalization 7). Any change to the format updates `docs/ModelFormat.md` and adds a version.
- **Graphs.** Layers run in index order, every layer after its inputs. A layer read by several gets
  their gradients in a fixed order (first written, the rest added), so graphs keep determinism.
  Chains must compute what they computed before graphs existed; `graph` group tests both.
- **ABI.** From 1.0 a release keeps the ABI of the one before (semantic versioning): `api.abi`
  compares the public structs' layouts, the enumerators, the integer constants and the functions'
  declarations with `Tests/Data/abi.txt` (`Tests/check_abi.py`). A new field takes reserved space
  (the `reserved` array shrinks by what it uses), a new function goes into a new node of
  `Src/Spingalett.map` (the symbol versions; the test checks that it lists the headers' functions and
  that the library exports them), and the baseline is recorded again at each release
  (`python Tests/check_abi.py update Bin/SpingalettAbi Tests/Data/abi.txt`).
- **The runtime gives the library's bits.** `libspingalett-runtime` is the library's model code
  compiled again with the same flags (`spingalett_build`), so a model computes the same in both:
  `runtime.library` compares the runtime with the full library's outputs on every kind of layer and
  precision (`SpingalettTests export-models`). Code that the runtime's sources need from elsewhere
  moves into them (as `spingalett_read_file()` and `spingalett_sample_correct()` did), and what they
  hold for networks, training or the GPU is left out with `#if !defined(SPINGALETT_RUNTIME)`, so that
  the runtime links without the rest even in Debug builds; its exports are exactly
  `Src/Spingalett.Runtime.map` (`runtime.abi`). A function added to `Spingalett.Runtime.h` or
  `Spingalett.Inference.h` goes into both maps, in the same node. `Tests/Data/runtime` holds models
  of past releases, which every 1.x runtime must run.
- **ABI with Python.** `Tests/Spingalett.Layout.c` and `test_python_layout.py` check that the
  ctypes structures match the C ones; a new field in a public struct needs both sides.
- **Engine scratch.** The engine's workspace size comes from `slett_conv_scratch()`; kernels may
  only use what it reserves.
- **Data set readers.** A reader's sample order is a function of its seed (drawn when it opens)
  and the pass or chunk number, never of how far ahead chunks are decoded or on how many threads;
  the `data` group compares a background thread, chunks decoded on the OpenMP threads and one chunk
  at a time. Decoding on a worker thread sets errors without logging and hands them to the caller.
- **The GPU repeats itself.** A GPU run gives the same bits every time on one device: sums run in
  fixed orders (fixed slices, fixed trees), never through atomics or subgroup operations whose order
  the driver chooses. Every tile of the matrix kernel adds an output's products in the order of k,
  (on the matrix units 16 values of k at a time, in that order), which is what lets tiles be chosen
  by timing; a kernel change must keep that (`SpingalettGpuTests` compares the tiles bit for bit, on
  either backend). GPU results match the CPU's up to rounding only (the `gpu` group compares trained
  networks by their outputs), and the two backends match each other the same way.
- **GPU dispatches state their memory.** The executor records a barrier only where a dispatch reads
  or writes what an earlier one wrote or read (`Recorder` in `Spingalett.Gpu.c`): every dispatch
  lists every range it reads and writes, or it will race with its neighbours.
- **Parameters on the GPU.** A trainer made with `COMPUTE_VULKAN`, and the copy `train()` leaves in
  the network on the GPU (`gpu_kept`; `predict()` and the next `train()` run on it), keep the
  network's parameters on the device: every public function that reads a network's parameters calls
  `spingalett_network_sync()` first, and every one that writes them on the host calls
  `spingalett_network_written()` after. A new reader or writer needs the call, or it sees stale
  values (the `gpu` group's `gpu_trainer`, `gpu_callback` and `gpu_kept` read and rewrite weights
  between steps and calls). Code that uses the kept copy holds it (`spingalett_network_hold_gpu()`),
  as sync and its release do; a change of the network's layers releases it first. `predict()`'s own
  copy takes the parameters again only when the network's `host_version` moved (every write, every
  copy back): code that changes parameters or running statistics on the host without
  `spingalett_network_written()` leaves it stale (`gpu_predict_cached`).
- **Data sets on the GPU.** Training on `spingalett_device_data_new()` sets gives the bits of training
  on the host's arrays: `rows.comp` gathers, augments and smooths as `gpu_fill()`, `augment_image()`
  and `smooth_targets()` do, so a change to one is a change to the other (`gpu_device_data` compares
  them, with inputs, targets or both on the GPU). Rows read where they are (inference, and training
  chunks that repeat every epoch) are of the precision the network keeps its inputs in: the set's
  floats, or its bfloat16 copy (`data_half()`), never floats in place of bfloat16.
- **Activations as bfloat16.** With `PRECISION_BFLOAT16` the executor keeps layers' outputs and
  gradients as bfloat16 (`kept_half()`): products find which operands are such by address
  (`product()`), and kernels with variants (`kernel()`, `half_words[]`) get a mask of their push
  constants' bfloat16 buffers. A kernel that reads or writes activations through anything but
  `ld()`/`st()` (`half.glsl`) or the products' loads reads garbage in bfloat16: every new kernel
  that touches activations needs a variant and an entry in `half_words[]`.
- **Normalizations applied by their readers.** On the GPU a batch normalization read only by an
  addition (`fold[]`) or by a depthwise convolution (`pro[]`, dwconv.comp's PRO) has no outputs: its
  reader applies it to the normalization's input as it reads. Code that reads a layer's outputs goes
  through `outputs_of()` and, for a `pro[]` layer, applies the normalization (the forward pass, the
  weight gradient, the derivative in the data gradient), or it reads a buffer that was never written.
- **Lazy training state.** Gradients and optimizer moments exist on the host once a network has
  trained on the CPU, made a trainer, or had its GPU copy's parameters brought back
  (`spingalett_training_state()`, which `spingalett_gpu_download()` calls); code that reads them
  before (saving optimizer state, `spingalett_get_parameters()` of a gradient) treats them as zero.
- **Shared models.** A `SpingalettModel` may be used from several threads at once: what
  `spingalett_model_predict()` caches in a model it owns is built under the owner's lock and only
  read afterwards (`model_shared` in the `model` group runs four threads; run it under
  ThreadSanitizer when changing that code).

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
- A parallel region whose `if` clause is false still costs about 0.2 us: loops that are often too
  small for threads use `SPINGALETT_PARALLEL_FOR(cond, for ...)` (`Spingalett.Private.h`), which runs
  them as plain code then. Integer kernels need far more work than float ones before threads pay.
- A change to a kernel that must keep its results (most of them) is checked by hashing the weights
  after an epoch, or the saved files, against a build of the previous commit, and with a profile
  (`gprofng collect app` works with AVX-512, Valgrind does not; Release builds are stripped, so
  profile a `RelWithDebInfo` build).
- Numbers quoted in README and CHANGELOG are measured, with the machine named. Laptops throttle:
  watch the temperature and frequency (`/proc/cpuinfo`) and let a long run finish before measuring.
- `-DSPINGALETT_NO_DIRECT_CONV` builds the library without the indirect convolution kernels, for
  comparisons with the products of gathered windows.
- CUDA kernels take their modes as template parameters wherever a loop's arrays or loads depend on
  them (the products' tiles, operand modes and ways of reading each operand; the depthwise
  convolutions' windows, modes, padding and maps kept as bfloat16): a mode read from the spec at run
  time left arrays in local memory and branches between loads, 2 to 15 times slower. Each instance
  is a unit of PTX (Kernels.def, expanded by cmake/Cuda.cmake); check a new kernel's registers, spills
  and stack frame with `ptxas -v` (from NVIDIA's `nvidia-cuda-nvcc` wheel) before timing it. Calls
  (`noinline`) pass structs through the stack and cost the caller registers: keep them off every
  output's path (the products hold the sloped epilogues only; `Spingalett.Cuda.c` runs the others as
  a pass of `epi.cu`). Kernels whose threads loop with the grid's stride get a wave of blocks
  (`grid_stride()`).
- GPU: `SPINGALETT_GPU_PROFILE=1` prints the time of every kernel and mode at exit (timestamps
  around each dispatch; trial runs of the tile choice excluded; with CUDA events around every
  launch, which inflate the times of small kernels: compare backends by wall time);
  `Bin/SpingalettGpuTests bench` times every tile on the products of ResNet-20 and the benchmark's
  MLP (`SPINGALETT_BENCH_TILES=1` lists each tile's time, `bench bf16` the matrix units' tiles,
  `SPINGALETT_GPU_BACKEND=cuda` the CUDA backend's, `SPINGALETT_BENCH_CONV="n h w c out kh kw sh sw
  ph pw groups"` any convolution's). Compare against
  `python Examples/benchmark_pytorch.py --cuda` (and `--cuda-fp32`, `--cuda-bf16`) on the same GPU. Laptop GPUs
  throttle hard when warm (`nvidia-smi --query-gpu=temperature.gpu,clocks.sm --format=csv`): with
  the CPU busy, an RTX 4050 fell to 57% of its clock.

## Style

- C23 for the library, C99 for `Inference.c`. Four-space indentation, lines up to about 120
  columns, `snake_case` with the `spingalett_` prefix for public functions and builders,
  `Spingalett` for public types, `SPINGALETT_` for enumerators and macros, `slett_` for format
  helpers, file-local names without a prefix. A new public name gets its short alias of 0.x only if
  it renames one (`Spingalett.Short.h`: other aliases would take names from the programs that
  include it). The library's sources (the internal headers define `SPINGALETT_SHORT_NAMES`), the
  examples, the apps and the README's code use the short names, which read more easily; the
  descriptions, the headers' comments and `docs/Reference.md` use the prefixed ones, which are also
  the exported symbols.
- Every file starts with the SPDX header (`MIT`, copyright pka_human).
- Comments say what a block computes and why, in full sentences; match the density of the code
  around them. Documentation is formal English and states facts, not intentions.
- Public API changes go to `Spingalett.h` with a comment, the README, the Python bindings and their
  README, and the CHANGELOG. From 1.0 the API and ABI follow semantic versioning (README,
  **Compatibility**): a 1.x release only adds, and the soname carries the major version.
- Commits are small and topical, with a subject line in the imperative and a body that says what
  changed and why (with measurements for performance work).

## Releases

The version lives in `CMakeLists.txt` (`project(... VERSION ...)`), `Bindings/Python/pyproject.toml`,
`Bindings/Python/spingalett/__init__.py` and `packaging/vcpkg/spingalett/vcpkg.json`. A release records
its ABI (`python Tests/check_abi.py update Bin/SpingalettAbi Tests/Data/abi.txt`), and one that adds
functions gives them a new node in `Src/Spingalett.map` (and `Src/Spingalett.Runtime.map`, for the
runtime's). `.github/workflows/release.yml` builds packages for Linux
(x86-64, x86-64-v3, AArch64), Windows and macOS (universal, with OpenMP), each also as a runtime
archive (`SPINGALETT_RUNTIME_ONLY`), trains the DigitPad model
(2 epochs on pull requests, 30 for releases) and attaches the AppImage and the Windows zip. The
maintainer merges pull requests and pushes tags.
