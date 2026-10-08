# Roadmap

Where Spingalett is going after 0.9, roughly in order. Plans change as the work shows what is
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

## 0.10: networks as graphs, interoperability, GPU groundwork

**Residual connections and graphs.** Today a network is a chain of layers. 0.10 turns it into a
directed acyclic graph of layers, so that a layer can take the outputs of several earlier ones:

- `add` (residual connections, ResNet blocks) and `concat` along channels (Inception-style and
  U-Net-style networks) as layer kinds;
- a builder API next to the chain one, e.g. a layer argument naming its inputs, with chains still
  built exactly as today;
- training, the step API, `predict()`, models and the engine walking the graph in a fixed
  topological order, keeping bit-identical results across thread counts;
- buffer reuse: activations freed as soon as their last consumer has run, so a deep residual
  network needs memory for its widest cut rather than for all of its layers;
- `.slett` format version 6 for graphs (versions 3 to 5 still written for chains);
- global average pooling as its own layer (today `avg_pool2d` with the full kernel), and the
  examples: a small ResNet for CIFAR-10, aiming at about 92% test accuracy.

**ONNX import.** `spingalett_import_onnx()` and `ModelTool import`: reading the layers Spingalett
has (Gemm/MatMul, Conv with groups, pooling, BatchNormalization, Add, Concat, Relu and the other
activations, Softmax, Flatten/Reshape where it is a no-op in channels-last order) into a network
or straight into a deployment model, with the NCHW to NHWC weight reordering done on import. A
protobuf reader of our own (no dependency), errors that name the unsupported operator, and tests
against models exported from PyTorch.

**Python wheels on PyPI.** `pip install spingalett` with the shared library inside: manylinux
x86-64 and AArch64, Windows, macOS universal, built by the release workflow from the same packages
it already makes; NumPy arrays in and out as today. Typed stubs (`.pyi`) for editors.

**A first GPU backend.** A compute-mode backend behind the existing `ComputeMode` switch, starting
with the matrix products, convolutions (implicit GEMM) and elementwise epilogues, keeping
parameters resident on the device during training. The candidates are Vulkan compute (portable,
works on AMD, Intel, NVIDIA and Apple through MoltenVK) and CUDA; the first one decides the
internal interface that later backends implement. Deterministic reductions stay the default.

**Smaller items considered for 0.10:**

- per-sample INT8 kernels on AVX-512 VNNI for dense layers in the engine (512-bit rows), and tile
  kernels for depthwise and grouped integer convolutions in batched prediction;
- AVX2 tiles for processors without VNNI, if they beat the current kernels;
- INT8 calibration from sample data (per-layer activation ranges instead of per-sample scaling),
  as an option for models where it helps accuracy;
- layer normalization (needed later for attention), label smoothing, and a
  reduce-on-plateau learning-rate schedule;
- what the 0.9 profiles left: packing for convolution weight gradients (a tenth of a small CNN's
  training step), the weight gradient of depthwise convolutions, and threads idling in the
  barriers of mini-batch steps on dense networks (a quarter of the time on four threads);
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
