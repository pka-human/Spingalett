# Contributing

Bug reports and pull requests are welcome. Please make sure the test suite passes
(`ctest --test-dir Build --output-on-failure`) for both a minimal build and a build with
`-DBUILD_WITH_OPENMP=ON -DBUILD_WITH_OPENBLAS=ON`, and add tests for new behaviour. CI runs the
suite with GCC and Clang, without `-march=native` (portable kernels) and under AddressSanitizer
and UndefinedBehaviorSanitizer. [AGENTS.md](AGENTS.md) summarizes the layout, the checks and the
invariants the tests enforce, for coding agents and new contributors alike.

## Project layout

```
Include/Spingalett/   Public headers (Spingalett.h, the runtime's Spingalett.Runtime.h, the inference
                      engine's Spingalett.Inference.h) and the CMake-generated configuration header
                      template
Src/                  Library sources (network, training, kernels, serialization, inference engine, ...)
Src/Gpu/              The GPU backends: device layer, network executor, Vulkan's compute shaders
                      (Shaders/) and the CUDA kernels (Cuda/)
Examples/             XOR, MNIST (dense and convolutional), CIFAR-10, U-Net segmentation, CharGPT (a
                      character-level GPT), Pretrained (a Hugging Face or GGUF language model continuing a
                      prompt), throughput benchmark (C and PyTorch counterpart), DatasetTool, ModelTool
Examples/Runtime/     Programs of the runtime alone: RunModel, RunGraph (graphs of operators), Chat, LlmBench (times sessions as llama-bench does)
Examples/Embedded/    MNIST on a Cortex-M4 (QEMU) with the standalone inference engine
docs/                 The guides, file format specifications (models, data sets), graphs of operators, performance,
                      the API reference (docs/README.md lists them)
Images/               The logo and the README's chart (make_speed_chart.py writes it)
Apps/DigitPad/        Digit-drawing demo app, its trainer and AppImage and Windows packaging
Tests/                Test suite (CTest) and fixtures
Bindings/             Bindings for Python, Rust, .NET (C#), Go, Java and PHP; the engine in WebAssembly
                      for JavaScript
packaging/            Conan, vcpkg, AUR, conda recipes and the container image (Formula/: Homebrew,
                      bucket/: Scoop, flake.nix: Nix)
cmake/                CMake package, pkg-config, runtime and inference-only build helpers
```
