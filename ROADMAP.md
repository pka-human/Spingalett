# Roadmap

Where Spingalett is going after 0.12, roughly in order. Plans change as the work shows what is
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

## 0.13: layers

0.12 took the GPU further (bfloat16 on its matrix units, the custom-loop API there) and made model
import fast. Next, what U-Net-style networks and more of ONNX need, on the CPU, the GPU and in the
engine:

- **Transposed convolutions** (strides, padding, output padding, groups): the data gradient of a
  convolution run forward, which the kernels of both backends already compute by phases.
- **Upsampling**: nearest and bilinear by integer factors (ONNX Resize and Upsample, PyTorch's
  `nn.Upsample`).
- **Layer normalization** over the channels of each cell (and over a vector), with its
  parameters; the first step towards attention.
- `.slett` format version 7 for them, the engine and deployment models in every precision, ONNX
  (`ConvTranspose`, `Resize`, `LayerNormalization`) and PyTorch weights, and an example that
  segments images.

**Measured and set aside:** keeping training sets in GPU memory and gathering (and augmenting)
batches there. With an i7-12650H feeding an RTX 4050 it won 5 to 11% for full-batch training over
many epochs and lost 2 to 3% for mini-batches, where the host's copies already overlap the GPU's
work; it may return for slower hosts.

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
- **Formats:** `.slett` version 7 and `.slettd` version 2 frozen as the 1.0 formats (later versions
  only add kinds of layers or coders, and every 1.x engine reads every 1.x file it can run).
- **C++ wrapper:** a header-only `spingalett.hpp` with RAII types (`Network`, `Model`, `Dataset`),
  `std::span` inputs, `std::expected` for errors, and the builder as a fluent API.
- **Documentation:** a reference generated from the headers and a tutorial path (MNIST, CIFAR-10,
  a U-Net, deployment to a microcontroller).

## 1.0: stability

0.14 with what its users find fixed: the C API and ABI follow semantic versioning from then on, the
soname becomes `libspingalett.so.1`, and a CMake package, pkg-config file, vcpkg and Conan recipes
ship with it.

## Any time: work that changes no API

Performance and backends that programs only notice by their speed land in whichever release is
next, before or after 1.0:

- **Deployment models on the GPU**: `spingalett_model_predict()` with FP16 weights, and with INT8
  weights where the result can stay exact against the CPU's (integer sums are; activations other
  than piecewise-linear ones go through `expf` and `tanhf` of the C library, which a GPU does not
  reproduce bit for bit).
- **Fewer GPU passes**: batch normalization's sums with the convolution before it, its
  normalization applied as the next layer reads its input; a kernel of its own for depthwise
  convolutions; the first layer's three channels padded to four for vector loads.
- **CPU kernels**: convolution windows as runs of `kernel_w x channels` floats (fast first layers),
  weights packed once per deployment model; per-sample INT8 kernels on AVX-512 VNNI for dense layers
  in the engine and tile kernels for depthwise and grouped integer convolutions.
- **INT8 calibration** from sample data, and a data set coder whose streams decode in independent
  lanes (the rANS coder decodes 30 to 40 MB/s per thread, each byte's context depending on the
  last).

## After 1.0

- **CUDA backend** for NVIDIA GPUs, next to the portable one, if Vulkan cannot reach their speed.
- **Quantization-aware training:** fake-quantized forward passes with straight-through gradients,
  so INT4 and INT2 models keep their accuracy (INT2 loses most of it on CIFAR-10 today).
- **NEON kernels for training**, so that Apple silicon and AArch64 servers train as fast as x86
  does; SVE where available.
- **Attention and sequence models:** embeddings, layer normalization, multi-head attention, and
  a small transformer example; recurrent layers if there is demand.
- **More language bindings:** Rust, C#, and a JavaScript / WebAssembly build of the engine.
- **More of the engine on microcontrollers:** CMSIS-style kernels for Cortex-M55 (Helium/MVE),
  RISC-V vector, and a way to run models larger than RAM layer by layer from flash.
