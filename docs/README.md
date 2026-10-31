# Spingalett's documentation

## What Spingalett is

Spingalett is a neural-network library written in C23 for training and running fully connected,
convolutional and transformer networks (language models among them: GPT-2 and LLaMA in form, trained
on token files and generating text), chains of layers or graphs of them (residual connections,
concatenated branches), on the CPU and on GPUs: NVIDIA's through CUDA, with kernels of its own (no CUDA toolkit,
cuBLAS or cuDNN), and others through Vulkan compute. It depends only on the C standard library (the
CUDA driver and the Vulkan loader are opened at run time where there are some): on the CPU, batch
training and inference run as matrix-matrix products on built-in AVX-512, AVX2, NEON or portable kernels (on
x86-64, chosen for the processor at run time), with OpenMP and OpenBLAS as optional build-time
accelerators; results are the same bits on one thread and many. Networks are declared with C23
designated initializers and all parameters live in flat contiguous arrays. For deployment, a trained network becomes a read-only
model in FP32, FP16, BF16, INT8, INT4 or INT2 that runs with integer kernels where the weights are
integers, in place from memory, a compiled-in array or flash, on desktops and on microcontrollers
alike; programs that only run models link the runtime, a library of that part alone. Language models'
GGUF files (LLaMA 1 to 3.x, Mistral, Qwen2 to Qwen3, Gemma 3, GPT-2) run as they are stored, their
quantized types computed in their blocks, faster than llama.cpp on the same processor. Any model built of ONNX's operators runs as
a graph of operators: ONNX files and PyTorch's exports (classifiers, transformers' encoders and decoders, CLIP,
Whisper, Stable Diffusion's VAE and UNet), with ONNX's semantics, fused into the library's kernels and the same bits
on any number of threads. Models also come in
from ONNX files and PyTorch weights as networks to train (GPT-2 and LLaMA-like checkpoints in Python), and bindings for
Python (wheels with the library inside), C++, Rust, .NET, Go, Java and PHP are included, and for
JavaScript the inference engine in WebAssembly, which runs models in Node and in browsers.

## Start here

| Page | What it holds |
|---|---|
| [Installing and building](Installing.md) | Packages, archives, building from source, every CMake option, using the library from a project |
| [Getting started](GettingStarted.md) | The first program, the short and prefixed names, the examples, errors and logging |
| [Tutorial](Tutorial.md) | Six programs, each adding to the one before, up to a model on a microcontroller |

## Guides

| Page | What it holds |
|---|---|
| [Networks](Networks.md) | Layers, convolutions, normalization, graphs of layers, reading a network back |
| [Training](Training.md) | `spingalett_train()`, validation, schedules, generators, a program's own loop, checkpoints, reproducibility |
| [Data sets](Datasets.md) | Data in memory, `.slettd` files, readers that stream into training |
| [Language models](LanguageModels.md) | Transformers, sessions and generation, tokenizers, Hugging Face checkpoints, GGUF files |
| [Models from elsewhere](Importing.md) | ONNX and PyTorch as networks that train, ONNX models run as they are |
| [Graphs of operators](Graphs.md) | Running any ONNX model, the operators, fusions, graphs on the GPU |
| [Inference and deployment](Deployment.md) | Running and saving networks, deployment models, the runtime, microcontrollers, WebAssembly, ModelTool |
| [Backends and the GPU](GPU.md) | Threads and kernels on the CPU, CUDA and Vulkan, sessions on the GPU, models split with the CPU |
| [Bindings](Bindings.md) | Python, C++, Rust, .NET, Go, Java, PHP |

## Reference

| Page | What it holds |
|---|---|
| [API reference](Reference.md) | Every declaration of the headers, with its comment |
| [Features](Features.md) | Everything the library has, by area |
| [Performance](Performance.md) | What was measured, how, on which machine, against what |
| [Compatibility](Compatibility.md) | What every 1.x release keeps |
| [Model format](ModelFormat.md) | `.slett` files, all versions |
| [Data set format](DatasetFormat.md) | `.slettd` files |

## The project

[Roadmap](../ROADMAP.md) · [Changelog](../CHANGELOG.md) · [Contributing and the project's layout](../CONTRIBUTING.md) ·
[Notes for coding agents](../AGENTS.md)
