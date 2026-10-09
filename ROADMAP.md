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
| 0.13.1 | "GPU": activations and their gradients in bfloat16 on the GPU, the network's copy kept there between calls, inputs and parameters written into device memory, tiles chosen by device timestamps, inference in chunks that stay in cache, faster pooling, weight gradients split only where it pays: ahead of PyTorch with cuDNN in every workload but the MLP's inference with PyTorch's data already in GPU memory |
| 0.14 (in progress) | Data sets in the GPU's memory (`spingalett_device_data_new()`): ahead of PyTorch with cuDNN in every workload, its data in GPU memory or not |

## 0.14: the release candidate

The last minor version of 0.x: whatever would break programs after 1.0 happens here, and nothing
else.

- **Data sets on the GPU** (done): `SpingalettDeviceData` and the `device_*` fields of the
  argument structs, which change their layout.

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

On an RTX 4050 Laptop GPU against PyTorch 2.14 with cuDNN (`Bin/Benchmark gpu` against
`benchmark_pytorch.py --cuda`, `--cuda-fp32`, `--cuda-bf16` and `--host-data`, README), Spingalett
0.14 is ahead in every workload, with the data in GPU memory (its data sets) or in host memory:
ResNet-20, the MNIST CNN (with and without batch normalization) and the U-Net train 1.2 to 1.8 times
as fast as PyTorch and run 1.8 to 2.4 times as fast in single precision, 1.4 to 1.7 and 1.8 to 2.1
times as fast as its autocast in bfloat16; the 784-512-1000-10 MLP trains mini-batches 3.2 to 5.8
times as fast and infers 1.19 and 1.3 times as fast. The margins left thin are the MLP's full
batches: 1.1 times in single precision (both at the device's 6.3 to 6.5 TFLOPS), 1.04 times in
bfloat16, where the products on the matrix units reach 15 to 17 TFLOPS on its shapes and cuBLAS 17.4
to 19.5.

Done in 0.13.1: the tile choice, activations and gradients in bfloat16, the pooling backward pass
(part of 2), larger tiles on the matrix units (part of 1). Done in 0.14: data sets on the GPU. The
next steps, each measured against the release before and against PyTorch:

1. **The matrix units' kernel:** stores to shared memory without bank conflicts (a swizzled
   layout), the epilogue writing four results a thread at once, split sums of weight gradients
   chosen by the tile the timing picked rather than by a single-precision estimate; target: cuBLAS's
   17 to 19.5 TFLOPS on the MLP's products, the weight gradients' among them.
2. **Fewer passes:** batch normalization's sums gathered in the epilogue of the convolution before
   it, its normalization and activation applied as the next layer reads its input; concatenated
   layers writing straight into their channels of the concatenation; a kernel of its own for
   depthwise convolutions; the first layer's three channels padded to four for vector loads.
3. **Gathered rows in the products:** the first layer's products reading a shuffled mini-batch's
   rows of a data set through their indices, rather than after a gather (mini-batches of the MLP:
   two small dispatches a step).
4. **Products in FP16** with single-precision sums as a precision of their own (Vulkan has no
   TF32, which PyTorch's convolutions use by default).
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

**Measured before:** keeping training sets in GPU memory and gathering (and augmenting) batches
there, done for every training run. With an i7-12650H feeding an RTX 4050 it won 5 to 11% for
full-batch training over many epochs and lost 2 to 3% for mini-batches, where the host's copies
already overlap the GPU's work: step 3 above makes it the caller's choice, for data used again and
again.

## 1.1: a CUDA backend

The first large work of 1.x, started once the steps above have shown how far Vulkan goes. What
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
