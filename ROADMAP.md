# Roadmap

Where Spingalett is going after 0.13, roughly in order. Plans change as the work shows what is
worth doing; the [CHANGELOG](CHANGELOG.md) records what was actually done. Every release keeps the
project's rule: nothing gets slower, and new kernels are measured against the previous release and
against PyTorch on the same machine.

## Done so far

| Version | Highlights |
|---|---|
| 0.1 - 0.4 | Dense networks, optimizers, schedules, dropout, data generators, validation and early stopping, the step API, Python bindings, `.slettd` data sets, DigitPad |
| 0.5 | Deployment models (FP32 to INT2), the standalone inference engine, C header export, Cortex-M example |
| 0.6 | Faster matrix kernels, run-time choice of AVX2 / AVX-512 kernels |
| 0.7 | Convolutions and pooling, opaque network handle, model format 4 |
| 0.8 | Batch normalization (folded at inference), grouped and depthwise convolutions, augmentation, CIFAR-10, INT8 tile kernels (AVX-512 VNNI, AVX-VNNI, Arm dot product), macOS packages with OpenMP |
| 0.9 | "Bottlenecks": `.slettd` format 2 (rANS coder, input shape, class names, several sets of targets), streaming readers that decode ahead or on the OpenMP threads, 8-bit data sets in memory, `DatasetTool cifar` and image folders; models that prepare their weights once, dot-product kernels for products of a few rows, faster pooling and depthwise convolutions, model files three to ten times as fast, lighter Python calls |
| 0.10 | "Graphs": networks as directed acyclic graphs (addition, concatenation, global average pooling; residual networks, ResNet-20 to 56 for CIFAR-10), `.slett` format 6 with outputs sharing memory in the engine, ONNX import, PyTorch weights (torch.save, safetensors), Python wheels for PyPI, convolutions as indirect matrix products, label smoothing, reduce on plateau |
| 0.11 | "GPU": training, `predict()` and `evaluate()` on a GPU through Vulkan compute (NVIDIA, AMD, Intel, Apple through MoltenVK), every kind of layer, deterministic; one matrix kernel for dense layers, convolutions and their gradients with tiles chosen by timing; the backend in every package and wheel, tested on lavapipe in CI |
| 0.12 | "Tensor cores": matrix products in bfloat16 on the GPU's matrix units (opt-in, faster than single precision), the custom-loop API on the GPU, fixes from a review of the GPU backend; ONNX import and PyTorch weights ten to a hundred times as fast (mapped files, one pass per tensor, external data files); gradients and optimizer state allocated when a network first trains; the first wheels on PyPI |
| 0.13 | "Layers": transposed convolutions, upsampling and layer normalization on the CPU and the GPU, in `.slett` format 7, the engine and deployment models in every precision; their import from ONNX (ConvTranspose, Resize, LayerNormalization) and PyTorch; a U-Net segmentation example; the GPU's loss kernel for wide outputs |

## 0.14: the release candidate

The last minor version of 0.x: whatever would break programs after 1.0 happens here, and nothing
else.

- **Names:** every public name under the library's prefix (`spingalett_`, `Spingalett`,
  `SPINGALETT_`): today the network type, the argument structs, the enumerations (`LAYER_DENSE`,
  `ACT_RELU`, ...) and the builder macros (`layer()`, `train()`, `predict()`, ...) are not, and
  collide with other code. The short names stay available from a header of their own, so that
  programs written for 0.x keep compiling.
- **Structs that can grow:** reserved fields (or a size field) in the public argument and
  description structs, so that 1.x can add options without breaking the ABI; one way of reporting
  errors everywhere; anything deprecated removed.
- **Zero means the default:** a field left out of a builder's arguments does what its name
  suggests. Today `act_func` 0 is `ACT_SIGMOID`, so an addition or concatenation written without
  `.act_func = ACT_NONE` applies a sigmoid; in 1.0 a zeroed activation is none.
- **Formats:** `.slett` version 7 and `.slettd` version 2 frozen as the 1.0 formats (later versions
  only add kinds of layers or coders, and every 1.x engine reads every 1.x file it can run).
- **C++ wrapper:** a header-only `spingalett.hpp` with RAII types (`Network`, `Model`, `Dataset`),
  `std::span` inputs, `std::expected` for errors, and the builder as a fluent API.
- **Documentation:** a reference generated from the headers and a tutorial path (MNIST, CIFAR-10,
  a U-Net, deployment to a microcontroller).
- **Room for CUDA:** `COMPUTE_CUDA` reserved in `ComputeMode` (it falls back to the CPU until 1.1),
  and a device layer under the GPU executor, so that the CUDA backend of 1.1 changes no API.

## 1.0: stability

0.14 with what its users find fixed: the C API and ABI follow semantic versioning from then on, the
soname becomes `libspingalett.so.1`, and a CMake package, pkg-config file, vcpkg and Conan recipes
ship with it.

## Any time: work that changes no API

Performance and backends that programs only notice by their speed land in whichever release is
next, before or after 1.0.

### The GPU against PyTorch

Measured with 0.13 on an RTX 4050 Laptop GPU against PyTorch 2.14 with cuDNN (`Bin/Benchmark gpu`,
`benchmark_pytorch.py --cuda`, `--cuda-fp32`, `--cuda-bf16`). In single precision Spingalett trains
ResNet-20 and the MNIST CNN 1.1 to 1.3 times as fast as PyTorch and the U-Net as fast: the batch
path is recorded once and replayed, with no launch per operation, and biases and activations run in
the products' epilogues. It is behind in two places:

- **bfloat16.** PyTorch's autocast trains ResNet-20 1.2 times and the U-Net 1.3 times as fast, and
  infers the U-Net 1.7 times as fast. The matrix units speed Spingalett's products up by 15% only
  (the U-Net's: 2,094 ms of the GPU's 3,080 in single precision, 1,776 ms in bfloat16): their
  operands are read in single precision and converted on load, so the memory traffic stays, and
  maps of 16 to 64 channels are bound by it (by an estimate of the bytes each pass moves, 100 to
  160 of the GPU's 192 GB/s). A third of
  the GPU's time goes to passes outside the products, which bfloat16 does not touch: batch
  normalization's three passes each way, activation derivatives, concatenations, pooling.
- **Large dense products.** The 784-512-1000-10 MLP trains full batches at 58% of PyTorch's speed
  and infers at 39%. The best tile of the matrix kernel reaches 6.2 to 6.6 TFLOPS, about half the
  GPU's single-precision peak; cuBLAS hides the latency of memory behind asynchronous copies into
  shared memory, which Vulkan does not have. The tile chosen by timing is up to 35% slower than the
  best on some products (the 784 -> 512 layer's data gradient: 356 against 266 us), and once was 2.7
  times slower, timed while the GPU's clocks were still low. And the samples come from host memory
  every epoch, while the PyTorch benchmark keeps them on the GPU (in a program, PyTorch would copy
  them too).

Steps, in order, each measured against the release before and against PyTorch:

1. **The tile choice:** the GPU brought to its clocks before the candidates are timed, every tile
   timed rather than a shortlist, the median of several runs kept. Up to 35% on single products.
2. **Activations in bfloat16** between layers whose products run in bfloat16, and the passes
   between them (normalizations, activations, concatenations, pooling, upsampling) reading and
   writing bfloat16: half the memory traffic. Target: PyTorch's autocast speed on the U-Net and
   ResNet-20.
3. **Fewer passes:** batch normalization's sums gathered in the epilogue of the convolution before
   it (per tile, added in a fixed order), its normalization and activation applied as the next
   layer reads its input; concatenated layers writing straight into their channels of the
   concatenation; a kernel of its own for depthwise convolutions; the first layer's three channels
   padded to four for vector loads.
4. **The matrix kernel:** larger tiles on the matrix units (more accumulators a subgroup, steps of k
   double-buffered through shared memory), and products in FP16 with single-precision sums as a
   precision of their own (Vulkan has no TF32, which PyTorch's convolutions use by default).
5. **Winograd convolutions** (F(2x2, 3x3)) for 3 x 3 convolutions of stride 1, on the GPU and the
   CPU: 2.25 times fewer multiplications, fixed transforms (deterministic), their rounding measured
   against the direct products, used for the shapes where they are measured to pay.

### Elsewhere

- **Deployment models on the GPU**: `spingalett_model_predict()` with FP16 weights, and with INT8
  weights where the result can stay exact against the CPU's (integer sums are; activations other
  than piecewise-linear ones go through `expf` and `tanhf` of the C library, which a GPU does not
  reproduce bit for bit).
- **Integer transposed convolutions in batches**: the phases of the stride as convolutions on the
  integer tile kernels (deployment models run them sample by sample through the engine's pass).
- **CPU kernels**: convolution windows as runs of `kernel_w x channels` floats (fast first layers),
  weights packed once per deployment model; per-sample INT8 kernels on AVX-512 VNNI for dense layers
  in the engine and tile kernels for depthwise and grouped integer convolutions.
- **INT8 calibration** from sample data, and a data set coder whose streams decode in independent
  lanes (the rANS coder decodes 30 to 40 MB/s per thread, each byte's context depending on the
  last).

**Measured and set aside:** keeping training sets in GPU memory and gathering (and augmenting)
batches there. With an i7-12650H feeding an RTX 4050 it won 5 to 11% for full-batch training over
many epochs and lost 2 to 3% for mini-batches, where the host's copies already overlap the GPU's
work; it may return for slower hosts.

## 1.1: a CUDA backend

The first large work of 1.x, started once steps 1 to 4 above have shown how far Vulkan goes. What
it cannot reach lies in what CUDA exposes and Vulkan does not: asynchronous copies into shared
memory (`cp.async`), TF32 on the matrix units, `ldmatrix`, control of registers and shared memory.
`COMPUTE_CUDA` (reserved in 0.14) runs on NVIDIA GPUs; Vulkan stays the backend for AMD, Intel and
Apple GPUs, and for NVIDIA ones where CUDA is not built.

- **Own kernels, no cuDNN or cuBLAS.** The CUDA driver (libcuda, which NVIDIA's driver installs)
  is opened at run time, as the Vulkan loader is, so the library still loads and runs without it.
  The kernels are CUDA C, compiled when the library is built to PTX (and to cubins for current
  architectures), which the driver compiles for GPUs that come later: running needs no CUDA
  toolkit, and packages stay a few megabytes.
- **Why not cuDNN and cuBLAS:** they are half a gigabyte to a gigabyte of libraries for each CUDA
  version; their fastest backward algorithms add through atomics, so runs would no longer repeat
  their bits (PyTorch's deterministic mode gives those algorithms up, and their speed with them);
  and the library measures its own kernels rather than wrapping others'. They remain what the
  backend is measured against (`benchmark_pytorch.py --cuda`).
- **The executor of the Vulkan backend, on the device layer of 0.14:** the batch path recorded
  once per kind of chunk as a CUDA graph and replayed, as command buffers are now; the same fixed
  orders of summation (no atomics, fixed slices), so CUDA runs repeat their bits too, and agree
  with the CPU up to rounding.
- **Kernels:** matrix products on `mma.sync` (TF32, bfloat16, FP16) fed by multi-stage `cp.async`
  pipelines and `ldmatrix`, tiles chosen by timing; convolutions as implicit products over the tap
  tables the Vulkan kernels use; activations in bfloat16 (step 2 above); deployment models on the
  GPU.
- **Targets on the RTX 4050:** PyTorch's speed with cuDNN in each precision on ResNet-20, the MNIST
  CNN and the U-Net, and within 10% of it on the MLP's full batches.
- **Tests** as for Vulkan: the GPU against the CPU up to rounding, two runs bit for bit, the matrix
  kernel's tiles bit for bit; CI compiles the kernels (nvcc in a container), and the GPU tests run
  on an NVIDIA GPU before each release.

## After 1.0

- **Quantization-aware training:** fake-quantized forward passes with straight-through gradients,
  so INT4 and INT2 models keep their accuracy (INT2 loses most of it on CIFAR-10 today).
- **NEON kernels for training**, so that Apple silicon and AArch64 servers train as fast as x86
  does; SVE where available.
- **Attention and sequence models:** embeddings, multi-head attention (on the layer normalization
  of 0.13), and a small transformer example; recurrent layers if there is demand.
- **More language bindings:** Rust, C#, and a JavaScript / WebAssembly build of the engine.
- **More of the engine on microcontrollers:** CMSIS-style kernels for Cortex-M55 (Helium/MVE),
  RISC-V vector, and a way to run models larger than RAM layer by layer from flash.
