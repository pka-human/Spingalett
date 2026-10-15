# Roadmap

Where Spingalett is going after 1.0, roughly in order. Plans change as the work shows what is
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
| 0.14 | Data sets in the GPU's memory (`spingalett_device_data_new()`), depthwise convolutions on a kernel of their own that applies the batch normalization before it, training chunks of 4,096 samples, networks made on the GPU without new memory: ahead of PyTorch with cuDNN by 1.14 times at least in every workload, its data in GPU memory or not |
| 1.0 | "Stability": the API, ABI and formats of 0.14 kept by semantic versioning (symbol versions, the `api.abi` test against the last release), the names of 0.x only on request, pkg-config, Conan and vcpkg recipes |
| 1.1 | "CUDA": a backend for NVIDIA GPUs on the library's own kernels (PTX compiled by Clang, no CUDA toolkit, cuBLAS or cuDNN), ahead of PyTorch with cuDNN by 1.2 to 7.7 times in every workload and of the Vulkan backend; batch normalizations in their product's epilogue at inference on both backends; the runtime library for programs that only run models. Why not cuDNN and cuBLAS: half a gigabyte to a gigabyte of libraries for each CUDA version, backward algorithms that add through atomics (runs would no longer repeat their bits), and the library measures its own kernels rather than wrapping others'; they remain what the backend is measured against |
| 1.2 | "Transformers": embeddings, multi-head attention (grouped-query, causal, rotary), RMS normalization, products of layers, GELU and SiLU, the sparse cross-entropy, `.slett` format 8, in training on the CPU and both GPU backends, the engine, deployment models and the runtime; token files, text generation, GPT-2 and LLaMA checkpoints in Python, a character-level GPT example; on CUDA flash attention on the tensor cores, register-tiled attention in single precision, GELU and SiLU in the products' epilogues, the weights' bfloat16 copy for sequences and images, block order kept in L2: ahead of PyTorch at its fastest (`torch.compile` with `max-autotune`) in every workload but the GPT in bfloat16 (0.91 times in training); bindings for Rust, .NET, Go and Java; Debian and RPM packages, AUR, Homebrew, Scoop, Nix, a container image, conda |
| 1.3 | "Language models in production": sessions with a key-value cache on the CPU and both GPU backends (batched generation, sliding windows), deployment models predicting on the GPU (integer ones to the CPU's bits) and weight-only INT8, INT4 and INT2 decoding, tokenizers and Hugging Face and GGUF checkpoints in C and every binding, tied embeddings, activation recomputation, gradient accumulation, schedules by step, padding, sliding windows and cross-attention in training, ONNX import of transformers, the engine in WebAssembly (npm), PHP bindings; ahead of PyTorch where 1.2 was not: the GPT in bfloat16 on CUDA ahead of `torch.compile` (1.02 times in training, 1.05 in inference), transformers' training on the CPU (linear layers' weight gradients on the dense products in slots of tokens, GELU and SiLU in vectors, element-wise passes in pieces: the GPT 1.05 times as fast as PyTorch on one thread and 1.64 times on eight), and on Vulkan flash attention on cooperative matrices and register-tiled attention in single precision (the GPT trained 3.1 times as fast in bfloat16 and 1.9 times in single precision, there within 5% of CUDA) |

## Language models: any GGUF, faster than llama.cpp

From 1.4 the releases aim at what llama.cpp and PyTorch do for language models: run the checkpoints
people download (any GGUF file of a common architecture, and Hugging Face's directories) on the CPU and
both GPU backends, faster than llama.cpp on the same machine and file. A model's file holds its weights
and hyper-parameters, not its computation: every architecture is code of the library's, built from
blocks (norms, attention and its positions, gated MLPs, experts, recurrent layers). 1.3 has the blocks of
GPT-2, LLaMA 1 and 2, Mistral and Qwen2/2.5 (and loads those GGUF files, their weights expanded to
float); 1.4 to 1.6 add the rest, in order of the models they open, and from 1.4 a command-line program
in a repository of its own, on the runtime library, serves them (**SaLaM**, below).

Each release measures against llama.cpp on the same GGUF files (`llama-bench`: prompt processing of 512
tokens, generation of 128, at 1 and 8 threads, on CUDA and on Vulkan) as well as against PyTorch, and
checks each family on real checkpoints: the logits of a prompt against transformers', the tokens of
greedy generation against llama.cpp's.

## 1.4: the language models of today

Dense models of the current families, from GGUF files computed as they are stored.

- **GGUF weights computed in their blocks**, not expanded to float: Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, the
  k-quants Q2_K to Q6_K, IQ4_NL and IQ4_XS as deployment-model precisions, their files mapped and used in
  place (a 7B model in Q4_K_M takes about 4 GB, not the 28 GB of its floats). Kernels: products of one
  token (generation, bound by memory: the blocks streamed at the memory's bandwidth, the activations
  quantized to 8 bits for integer dot products) and of many (a prompt: blocks decoded into tiles of the
  matrix kernels, or into integer tiles) on AVX2, AVX-512 VNNI and NEON with the dot product, on CUDA
  (dp4a and the tensor cores' integer products) and on Vulkan (integer dot products, cooperative
  matrices). Sums in fixed orders, as everywhere: the same bits on any number of threads.
- **The blocks of today's dense families**: scaled rotary embeddings (LLaMA 3's, YaRN, linear and NTK
  scaling: LLaMA 3.1 and 3.2 are refused today), rotary embeddings on part of a head (Phi, GLM,
  StableLM), RMS normalization of each head's queries and keys inside attention (QK-norm: Qwen3, Gemma 3,
  OLMo 2), Gemma's soft-capping of logits (`tanh(x / c) x c`, of attention's and of the output's), its
  normalizations scaled by `1 + w`, embeddings scaled by the square root of the width and norms after
  the attention and the MLP (the last three at loading, or as graphs of existing layers), on the CPU, both
  GPU backends, the engine, deployment models and sessions, in training as well.
- **Architectures as descriptions**: a table of each family (its hyper-parameters, its tensors' names in
  GGUF and in Hugging Face's directories, the blocks in their order) from which the loaders build the
  network, so that a family that recombines known blocks is a description, not code. Verified on LLaMA
  3.2 (1B, 3B), Qwen3 (0.6B to 8B), Gemma 2 and 3, Phi-3 and Phi-4 mini, OLMo 2, GLM-4, SmolLM3 and
  Mistral's current small models.
- **The runtime reads language models**: GGUF files and Hugging Face's directories loaded by the runtime
  library into deployment models, their tokenizers with them (the runtime holds `Tokenizer.c` and the
  GGUF reader), so that a program serves a model with the runtime alone.
- **Generation as llama.cpp offers it**: min-p sampling, repetition, frequency and presence penalties,
  stop strings; chat templates (the subset of Jinja that GGUF's `tokenizer.chat_template` uses, rendered
  in C); the key-value cache in FP16 or 8 bits; a prompt's cache kept between requests and the context
  shifted when it fills.
- **Checkpoints** that save and resume a training run exactly (done: the run's seeds, Adam's corrections,
  its schedule, plateau and best epoch, the readers' positions; autosaves are checkpoints).

## 1.5: mixture of experts

- **A layer of experts** with a learned router (top-k gating, softmax or sigmoid scores, shared
  experts), the experts' products batched by expert, in inference and in training (the load-balancing
  loss): Mixtral, Qwen3's and Qwen3.5's MoE models, gpt-oss, Llama 4, GLM-4.5 Air, OLMoE.
- **Attention sinks** (gpt-oss: a learned logit in each head's softmax) and sliding windows alternating
  with full attention by layer; MXFP4 weights (gpt-oss's GGUF files).
- **Experts kept on the host** while attention and the dense layers run on the GPU (the offload that
  makes 30B models of 3B active parameters fit a laptop's GPU), and layers split between the CPU and the
  GPU for any model larger than the GPU's memory.
- **Speculative decoding** with a draft model (several tokens checked in one pass, the bits of plain
  sampling kept).

## 1.6: hybrid models: linear attention and state spaces

- **Gated DeltaNet** (Qwen3.5, Qwen3-Next) with its short causal convolution and gated output, and
  attention with a gated output beside it; **Mamba-2** (Falcon-H1, Granite 4, Nemotron-H). Layers whose
  state is a matrix carried from token to token instead of a key-value cache: sessions keep it per
  sequence (rewound by checkpoints of the state), training runs the chunked parallel form, the GPU
  backends scan in fixed orders. With 1.5's experts this opens every Qwen3.5 model, Qwen3.5-2B among them.
- **Recurrent layers** (GRU, LSTM) on the same scans, as layers of the same graphs as attention.
- **Multimodal inputs**: the vision encoders of these models (Qwen3.5's, Gemma 3's: patch embeddings,
  their own attention) and audio frames, inputs of several kinds in one network.
- **ALiBi positions** and block-sparse masks in the flash kernels, for the models that use them.

## 1.7: Arm, Apple and the edge

The CPU and GPU paths that 1.0 to 1.6 tuned on x86 and NVIDIA, brought to the other hardware the
library runs on.

- **NEON kernels for training** (and SVE where available), so that Apple silicon and AArch64
  servers train as fast as x86 does; AArch64 matrix kernels chosen at run time as x86's are.
- **A Metal backend** for Apple GPUs (MoltenVK runs the Vulkan backend there today), on the same
  executor, with bfloat16 products where the GPU has them and 1.4's GGUF kernels.
- **Quantization-aware training**: fake-quantized forward passes with straight-through gradients,
  so that INT4 and INT2 models (and language models' INT4 weights) keep their accuracy (INT2 loses
  most of it on CIFAR-10 today).
- **More of the engine on microcontrollers**: CMSIS-style kernels for Cortex-M55 (Helium/MVE) and
  RISC-V vector, transformers' layers within a microcontroller's memory, and models larger than RAM
  run layer by layer from flash.

## 1.8: scale

Models that do not fit one GPU's memory or time (planned for 1.4 until the language models came first).

- **Data-parallel training on several GPUs** of one machine: each GPU a share of the batch, the
  gradients summed in a fixed order (a deterministic all-reduce over peer copies, no NCCL), the same
  bits as one GPU taking the whole batch; inference of a model split across GPUs by layers and by rows.
- **Sharded optimizer state** (each GPU keeps the moments of its share of the parameters) and the
  optimizer's state, or parameters, offloaded to the host's memory; checkpoints sharded across files.
- **Data sets of billions of tokens**: token files mapped into memory and read without copies,
  shuffled windows across files, the GPU's data sets streamed in chunks.
- **Multi-head latent attention** (DeepSeek V3, Kimi K2: the compressed key-value cache), for the models
  that need several GPUs or the host's memory anyway.
- **FP16 products with loss scaling** (planned for 1.3, moved): products on the matrix units with FP16
  operands for GPUs whose matrix units take FP16 but not bfloat16 (NVIDIA's Turing generation, Vulkan
  devices with FP16 cooperative matrices only), the loss scaled and the scale lowered on steps whose
  gradients overflow, deterministically. Bfloat16 runs on every GPU the CUDA backend takes (Ampere and
  later) and needs no scaling, and the kernels' variants would double the matrix units' units.

## SaLaM: Spingalett's language models on the command line

A program in a repository of its own (working name SaLaM, "Spingalett Language Models"), built on the
runtime library alone, that does for Spingalett what llama.cpp's programs do for llama.cpp: one binary
without dependencies, packaged where the library is.

- **0.1, with 1.4**: `salam run model.gguf -p "..."` (a prompt's continuation), `salam chat` (the model's
  chat template, the conversation's cache kept between turns), `salam bench` (prompt processing and
  generation in tokens per second, as `llama-bench` reports them), GGUF files and Hugging Face's
  directories, the CPU, CUDA or Vulkan chosen by flag, sampling options as llama.cpp's, models fetched
  from Hugging Face by name.
- **0.2, with 1.5**: a server with an OpenAI-compatible HTTP API (completions, chat, streaming) that
  batches its requests in one session, the experts' offload and layer splits by flag, speculative
  decoding with a draft model.
- **0.3, with 1.6**: the hybrid models, images in chats (the vision encoders), grammars that constrain
  generation (GBNF, JSON schemas).

## 2.0: what 1.x cannot add

A major release only for what would break programs; candidates so far:

- tensors of more than three axes and of variable length in the API (sequences of any length per
  call rather than a fixed input layer), which the transformer layers of 1.2 approximate with a
  fixed context;
- layers of the program's own (forward and backward callbacks, with GPU kernels) in networks the
  library trains;
- training across machines.

## Distribution

Where Spingalett can be installed from, and what each channel needs: a version where it works from
this repository and its releases alone, and what publishing it elsewhere takes.

| Channel | State | What it takes |
|---|---|---|
| PyPI (`pip install spingalett`) | published since 0.12 | trusted publishing from `release.yml` |
| GitHub releases (library, runtime, DigitPad, wheels) | published | `release.yml` |
| CMake package, pkg-config | in every archive | — |
| vcpkg and Conan recipes (`packaging/`) | in the repository | submitting them to vcpkg's registry and Conan Center: pull requests from the maintainer's accounts |
| Homebrew (`brew tap pka-human/spingalett https://github.com/pka-human/Spingalett`) | 1.2 | `Formula/spingalett.rb` in this repository; homebrew-core asks for a project with more users first |
| Scoop (`scoop bucket add spingalett https://github.com/pka-human/Spingalett`) | 1.2 | `bucket/spingalett.json` in this repository |
| Nix (`nix profile install github:pka-human/Spingalett`) | 1.2 | `flake.nix` in this repository; nixpkgs: a pull request |
| Debian, Ubuntu, Fedora (`.deb`, `.rpm` on the release page) | 1.2 | built by `release.yml`; a PPA or COPR repository needs the maintainer's Launchpad or Fedora account |
| Container image (`ghcr.io/pka-human/spingalett`) | 1.2 | pushed by `release.yml` with its own token |
| Go modules (`go get github.com/pka-human/Spingalett/Bindings/Go`) | 1.2 | nothing: Go fetches from the repository |
| Rust (`cargo add spingalett --git https://github.com/pka-human/Spingalett`) | 1.2 | crates.io: an API token, then `cargo publish` of the two crates |
| .NET (`Spingalett.1.2.0.nupkg` on the release page) | 1.2 | nuget.org: an API key, then `dotnet nuget push` |
| Java (`spingalett-1.2.0.jar` on the release page) | 1.2 | Maven Central: a Sonatype account and a signing key |
| AUR (`spingalett`, `spingalett-git`) | `packaging/aur` in 1.2 | an AUR account with an SSH key: the maintainer pushes the PKGBUILDs |
| conda-forge | `packaging/conda` in 1.2 | a pull request to conda-forge/staged-recipes from the maintainer's account |
| winget, Chocolatey | after 1.2 | manifests submitted from the maintainer's accounts |
| npm (the engine in WebAssembly) | 1.3 | an npm account |
| PHP (Composer, Packagist) | 1.3 | Packagist reads `composer.json` at a repository's root: a mirror repository of `Bindings/PHP` that `release.yml` pushes to (a token for it), and a Packagist account |

## Any time: work that changes no API

Performance and backends that programs only notice by their speed land in whichever release of 1.x
is next.

### The GPU against PyTorch

On an RTX 4050 Laptop GPU against PyTorch 2.14 with cuDNN (`Bin/Benchmark gpu` against
`benchmark_pytorch.py --cuda`, `--cuda-fp32`, `--cuda-bf16` and `--host-data`, README), Spingalett
0.14 is ahead in every workload, with the data in GPU memory (its data sets) or in host memory:
the MNIST CNN (with and without batch normalization) trains 1.6 to 1.9 times as fast as PyTorch and
runs 2.1 to 2.4 times as fast in single precision, 1.5 to 1.6 and 1.8 to 2.1 times as fast as its
autocast in bfloat16; ResNet-20 and the U-Net run 1.75 to 1.9 times as fast in either precision and
train 1.3 to 1.45 times as fast in bfloat16; the MobileNet-style network 1.45 and 4.5 times, 1.26 and
3 times; the 784-512-1000-10 MLP trains mini-batches 3.7 to 6.2 times as fast and infers 1.23 and 1.3
times as fast. The margins left thin, 1.14 to 1.25 times, are the MLP's full batches (single
precision at the device's 6.3 to 6.5 TFLOPS, as cuBLAS; bfloat16 at 15 to 18 TFLOPS against
cuBLAS's 17 to 19.5) and the single-precision training of ResNet-20 and the U-Net against PyTorch's
TF32 convolutions.

Done in 0.13.1: the tile choice, activations and gradients in bfloat16, the pooling backward pass
(part of 3), larger tiles on the matrix units (part of 1). Done in 0.14: data sets on the GPU; a
kernel of its own for depthwise convolutions, four pixels a thread with the filters in registers,
which applies the batch normalization it reads and sums that normalization's backward pass in its
data gradient (part of 3: the MobileNet-style network trains 3.4 and 12 times as fast as in 0.13.1);
training chunks of 4,096 samples where no batch normalization groups them; networks made on the GPU
in freed memory, their parameters' upload not waited for. Done in 1.1: the CUDA backend (above),
whose margins over PyTorch are wider in every workload (1.2 to 7.7 times; README); on both backends
a batch normalization applied in the epilogue of the product it reads at inference (part of 3), its
coefficients computed once until the parameters change, and three chunks in flight; on CUDA a
kernel of its own for a first layer's few channels (part of 3). Done in 1.2 (CUDA): the weights'
bfloat16 copy wherever products take 256 rows of a layer (sequences and images, not only batches of
256 samples: ResNet-20 and the U-Net infer 1.1 to 1.14 times as fast in bfloat16), blocks of a
product in bands that keep their rows of A in L2, GELU's and SiLU's epilogues in the products. Done in
1.3: on CUDA the matrix units' products in tiles of 128 x 128 over three stages of asynchronous copies
(`gemm_pipe.cu`) and flash attention's forward pass of 32 queries a warp; on Vulkan flash attention on
cooperative matrices, attention in register tiles in single precision and GELU and SiLU in the products'
epilogues. The next steps, each measured against the release before and against PyTorch:

1. **The matrix units' kernels:** on Vulkan, the GPT in bfloat16 trains at 0.92 times and infers at 0.85 times
   the CUDA backend's speed (1.3), its products on cooperative matrices behind CUDA's three-stage 128 x 128
   tiles. Measured in 1.3 and left out: two steps of the sum in shared memory and larger tiles (training 0.5%
   slower), blocks in bands with results stored eight at a time (the GPT's inference 4% faster, the MLP's and the
   CNN's bfloat16 runs from host arrays 2 to 5% slower; branch `vk-tiles-clean`). On both backends, split sums of
   weight gradients chosen by the tile the timing picked rather than by a single-precision estimate, and attention
   of layers with padding, windows or cross-attention on the matrix units (they run on the register tiles of
   single precision).
2. **The single-precision kernel for convolutions of few channels.** ResNet-20's and the U-Net's
   convolutions (16 to 64 filters, k = 144 to 576 values of the window) reach 3.3 to 4.5 TFLOPS, the
   MLP's products 5.6 to 6.4, and they take 70% of those networks' training time in single
   precision. Measured in 0.14 by taking parts of gemm.comp out: the epilogue cost 9 to 11% of the
   16-filter products (now four results a store), the windows' address arithmetic about 5%, and a
   variant reading each thread's rows of A straight into registers, B alone through shared memory,
   ran slower than the tiles it would replace: with 16 filters each value of A loaded serves 16
   multiply-adds, and on Ada the integer work around it shares the units of half the FP32 lanes.
   What is left to try: A in shared memory as rows of k (one vector store per load, where it is four
   scalar ones), loops that take tile after tile (the next tile's first loads under the current
   tile's sums), and Winograd convolutions (step 6), which cut the multiplications themselves.
   Target: ResNet-20 and the U-Net trained 1.4 times as fast as PyTorch with TF32.
3. **Fewer passes:** batch normalization's sums gathered in the epilogue of the convolution before
   it, its normalization and activation applied by the products that read its outputs (done for
   depthwise readers); concatenated layers writing straight into their channels of the
   concatenation; the first layer's three channels padded to four for vector loads.
4. **Gathered rows in the products:** the first layer's products reading a shuffled mini-batch's
   rows of a data set through their indices, rather than after a gather (mini-batches of the MLP:
   two small dispatches a step).
5. **Products in FP16** with single-precision sums as a precision of their own (Vulkan has no
   TF32, which PyTorch's convolutions use by default).
6. **Winograd convolutions** (F(2x2, 3x3)) for 3 x 3 convolutions of stride 1, on the GPU and the
   CPU: 2.25 times fewer multiplications, fixed transforms (deterministic), their rounding measured
   against the direct products, used for the shapes where they are measured to pay.

### Elsewhere

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
already overlap the GPU's work: data sets on the GPU (0.14) make it the caller's choice, for data used
again and again. In 0.14, on the RTX 4050: column sums (batch normalization's statistics and backward
sums) read four channels at a time and over more workgroups gained nothing, bound by memory;
inference chunks larger than 32 MB of activations slowed even the MLP, whose layers' outputs stay in
cache between products; training chunks of 8,192 samples were no faster than of 4,096; the next
epoch's first chunk submitted before the last one's losses come back would win 2 to 4% of full-batch
epochs of 6 ms, and is left for now (it runs before early stopping and callbacks decide).
