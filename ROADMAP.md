# Roadmap

Where Spingalett is going after 0.11, roughly in order. Plans change as the work shows what is
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

## 0.12: the GPU, further

0.11 brought training and inference on the GPU through Vulkan compute: every kind of layer,
deterministic, on par with PyTorch and cuDNN on convolutional networks. Next:

- **Tensor cores and other matrix units** through `VK_KHR_cooperative_matrix` (NVIDIA, AMD RDNA 3,
  Intel Arc), as an opt-in mixed precision: products of FP16 or BF16 operands accumulated in FP32,
  master weights in FP32, loss scaling where FP16 needs it. Single precision stays the default.
- **Deployment models on the GPU**: `spingalett_model_predict()` with FP16 and INT8 weights (the
  dot product extension, `VK_KHR_shader_integer_dot_product`), exact against the CPU's integer
  results like every other backend.
- **The step API on the GPU** (`spingalett_trainer_*`), so that custom losses and loops train there
  too, and per-sample training through chunks of one.
- **Fewer passes**: batch normalization's sums in the epilogue of the convolution before it, its
  normalization applied as the next layer reads its input, pooling fused into the convolution it
  follows; a kernel of its own for depthwise convolutions; the first layer's three channels padded
  to four for vector loads.
- **More of the queue**: copies on a transfer queue, and the inputs of a whole epoch in device
  memory when they fit (gathered and augmented on the GPU).

**CPU items carried over:**

- the convolution kernels of 0.10, further: kernel rows as one run of `kernel_w x channels` floats
  (fewer pointers, and fast first layers of one or three channels), weights packed once per
  deployment model;
- per-sample INT8 kernels on AVX-512 VNNI for dense layers in the engine (512-bit rows), and tile
  kernels for depthwise and grouped integer convolutions in batched prediction;
- INT8 calibration from sample data (per-layer activation ranges instead of per-sample scaling),
  as an option for models where it helps accuracy;
- layer normalization (needed later for attention), upsampling and transposed convolutions (U-Net
  style networks, more of ONNX);
- gradients and optimizer state allocated on the first training step, so that networks loaded
  only for inference take a quarter of the memory;
- a faster data set decoder (the rANS coder decodes 30 to 40 MB/s per thread).

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
