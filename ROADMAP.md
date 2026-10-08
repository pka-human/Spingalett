# Roadmap

Where Spingalett is going after 0.10, roughly in order. Plans change as the work shows what is
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

## 0.11: a first GPU backend

**GPU training and inference.** A compute-mode backend behind the existing `ComputeMode` switch,
starting with the matrix products, convolutions (as indirect products, like the CPU kernels of
0.10) and the elementwise passes (activations, batch normalization, additions and concatenations of
graphs), keeping parameters and activations resident on the device during training and copying
only batches in and losses out. The candidates are Vulkan compute (portable: AMD, Intel, NVIDIA and
Apple through MoltenVK; SPIR-V compiled at build time and embedded) and CUDA; the first one decides
the internal interface that later backends implement. Deterministic reductions stay the default:
fixed-order trees rather than atomics, so a GPU run gives the same bits every time.

**Smaller items considered for 0.11:**

- the convolution kernels of 0.10, further: kernel rows as one run of `kernel_w x channels` floats
  (fewer pointers, and fast first layers of one or three channels), phases for the data gradient
  of strided convolutions (three quarters of its products are zeros today), weights packed once
  per deployment model;
- per-sample INT8 kernels on AVX-512 VNNI for dense layers in the engine (512-bit rows), and tile
  kernels for depthwise and grouped integer convolutions in batched prediction;
- AVX2 tiles for processors without VNNI, if they beat the current kernels;
- INT8 calibration from sample data (per-layer activation ranges instead of per-sample scaling),
  as an option for models where it helps accuracy;
- layer normalization (needed later for attention), upsampling and transposed convolutions (U-Net
  style networks, more of ONNX);
- threads idling in the barriers of mini-batch steps on dense networks (a quarter of the time on
  four threads in 0.9's profiles), and the weight gradient of depthwise convolutions;
- gradients and optimizer state allocated on the first training step, so that networks loaded
  only for inference take a quarter of the memory;
- a faster data set decoder (the rANS coder decodes 30 to 40 MB/s per thread; streaming CIFAR-10
  costs about a fifth of an epoch of a small CNN).

## 1.0: stability

- **API freeze:** the C API and ABI follow semantic versioning from 1.0 on; the soname becomes
  `libspingalett.so.1`. Before the freeze: a review of every public name and struct, reserved
  fields where growth is expected, and removal of anything deprecated.
- **C++ wrapper:** a header-only `spingalett.hpp` with RAII types (`Network`, `Model`, `Dataset`),
  `std::span` inputs, exceptions or `std::expected` for errors, and the builder as a fluent API.
- **Documentation:** a reference generated from the headers, a tutorial path (MNIST, CIFAR-10,
  deployment to a microcontroller), and the file formats frozen at their 1.0 versions.
- **Packaging:** a CMake package and pkg-config file in every release archive (as now), plus
  vcpkg and Conan recipes.

## Later

- **CUDA / cuDNN backend** for NVIDIA GPUs at full speed, next to the portable GPU backend.
- **Quantization-aware training:** fake-quantized forward passes with straight-through gradients,
  so INT4 and INT2 models keep their accuracy (INT2 loses most of it on CIFAR-10 today).
- **NEON kernels for training**, so that Apple silicon and AArch64 servers train as fast as x86
  does; SVE where available.
- **Attention and sequence models:** embeddings, layer normalization, multi-head attention, and
  a small transformer example; recurrent layers if there is demand.
- **More language bindings:** Rust, C#, and a JavaScript / WebAssembly build of the engine.
- **More of the engine on microcontrollers:** CMSIS-style kernels for Cortex-M55 (Helium/MVE),
  RISC-V vector, and a way to run models larger than RAM layer by layer from flash.
