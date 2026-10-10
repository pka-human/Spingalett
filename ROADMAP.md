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

## 1.2: transformers, more languages, more packages (in progress)

Spingalett learns the networks of today's language models, in every part of the library at once
(training on the CPU and both GPU backends, the model format, the engine, deployment models, the
runtime, the importers and the bindings), and reaches more languages and package managers. Nothing
of 1.1 changes: a 1.1 program builds and runs against 1.2 unchanged.

### The layers of transformers

A sequence of `n` tokens is a layer of `1 x n` cells (the input layer holds the tokens' indices as
floats), so that what the library has per cell (layer normalization, dropout, additions, products
of layers) serves sequences as it serves images.

- **Embeddings** (`spingalett_embedding()`): a table of `vocabulary` rows, a learned position vector
  per cell when asked (GPT-2's absolute positions). The table's gradient is summed per token in the
  order of the batch (on the GPU through a stable sort of the batch's tokens, so that it stays
  deterministic without atomics).
- **Attention** (`spingalett_attention()`): multi-head scaled dot-product attention over a layer of
  queries, keys and values side by side, with grouped-query and multi-query attention (`kv_heads`),
  a causal mask, and rotary position embeddings (`rope_theta`, LLaMA's). On the CPU as blocked
  matrix products a thread per sample and group of heads; on CUDA as flash attention on the tensor
  cores (the scores never stored, the backward pass recomputing them, its two halves in two kernels
  so that no sum depends on the order of atomics); on Vulkan as tiles of the same scheme in single
  precision.
- **RMS normalization** (`spingalett_rms_norm()`), **products of layers**
  (`spingalett_multiply_layers()`, SwiGLU's gate), **GELU** (exact and tanh) and **SiLU**, whose
  derivatives need the values before the activation (training keeps them for those layers), and
  **linear layers over tokens** (`spingalett_linear()`: a 1 x 1 convolution).
- **The sparse cross-entropy loss** (`SPINGALETT_LOSS_SPARSE_CROSS_ENTROPY`): a softmax over each
  cell's channels against one class index a cell (a negative one for none, as padding), with label
  smoothing; targets are `spingalett_target_size()` values a sample, not one-hot rows of the
  vocabulary.
- **Format 8** of `.slett` files for them; the engine and deployment models run them in every
  precision (an embedding table in INT8 or INT4 is a table of rows with their scales).
- **Language models end to end:** a reader of token files (`.bin` files of 16- or 32-bit tokens, as
  nanoGPT writes them) that serves windows of a context and their next tokens; generation from a
  prompt (temperature, top-k, top-p); PyTorch weights of GPT-2- and LLaMA-style models (an embedding
  takes its token and position tables, attention's projections concatenated); an example that trains
  a character-level GPT and samples from it; the GPT in `Examples/Benchmark.c` against PyTorch with
  `scaled_dot_product_attention` and `torch.compile`.

### The GPU against PyTorch at its fastest

1.1 was measured against PyTorch's defaults. `benchmark_pytorch.py --max` runs PyTorch at its
fastest: `torch.compile(mode="max-autotune")` (Triton kernels, cuDNN and cuBLAS chosen by timing,
the passes replayed as CUDA graphs), `cudnn.benchmark`, channels-last tensors and fused optimizers.
On the RTX 4050, 1.1 stays ahead of it in single precision and TF32 (by 1.05 times in the U-Net's
training to 2.7 times) and in most of bfloat16, but not everywhere: ResNet-20 trains 1.03 times as
fast and infers 1.06 times as fast, and the U-Net trains at 0.94 times PyTorch's speed. 1.2's target
is a clear margin in every workload against that configuration: the matrix units' kernel for
convolutions (shared memory without bank conflicts, tiles that start the next tile's loads under
the current sums), batch normalization's sums in the epilogue of the convolution before it, and the
passes between products fused where PyTorch's compiler fuses them.

### Bindings

The C API stays the source of truth; each binding wraps it whole (networks and layers, training,
the step API, prediction and evaluation, deployment models, data sets, the GPU) and loads the same
shared library, so that it gains every backend and kernel of a release on the day it ships.

- **Rust** (`Bindings/Rust`): `spingalett-sys` (the declarations, checked against the headers in CI)
  and `spingalett`, a safe crate (owners that free on drop, slices, `Result` for errors).
- **C#** (`Bindings/CSharp`): a .NET 8 library over P/Invoke (`LibraryImport`, spans, `IDisposable`
  owners), packaged for NuGet with the native libraries of every platform inside.
- **Go** (`Bindings/Go`): a cgo package (`go get github.com/pka-human/Spingalett/Bindings/Go`).
- **Java** (`Bindings/Java`): the Foreign Function and Memory API of Java 22 (no JNI code to build).

### Packages

Where Spingalett can be installed from, with what each channel needs from the maintainer (see
**Distribution** below): a Homebrew tap and a Scoop bucket served from this repository, a Nix
flake, an AUR package description, `.deb` and `.rpm` packages and a container image built by the
release workflow, and recipes for conda-forge.

## 1.3: language models in production

What a model needs after training: running it fast for many tokens, from the formats people have
weights in.

- **Inference with a key-value cache:** generation that computes each new token once (the keys and
  values of the tokens before kept per layer), on the CPU (deployment models: `spingalett_model_*`
  gains a session for a sequence) and on the GPU; batched generation of several sequences.
- **Deployment models on the GPU**: `spingalett_model_predict()` with FP16, BF16 and INT8 weights
  (exact against the CPU where the sums are integers), and weight-only INT4 and INT8 matrix kernels
  for language models (the weights read once per token, dequantized in registers).
- **Tokenizers**: byte-level BPE (GPT-2, tiktoken's encodings) and SentencePiece-style models from
  `tokenizer.json`, encoding and decoding in C, with the special tokens of chat formats.
- **Weights from the formats they come in**: Hugging Face checkpoints of LLaMA, Mistral, Qwen and
  GPT-2 by their names (safetensors, sharded), and GGUF files.
- **Training features of language models**: tied input and output embeddings, activation
  recomputation (checkpointing) to fit longer contexts, gradient accumulation and learning-rate
  schedules by step in `spingalett_train()`, sequences of different lengths in one batch, sliding
  window attention, cross-attention (encoder-decoder models), and FP16 products with loss scaling.
- **ONNX import of transformers**: products of two activations, softmax over an axis, Gather,
  layer normalization over sequences, attention patterns folded into the attention layer.
- **A WebAssembly build of the engine** and a JavaScript package around it (npm), for models that
  run in browsers.

## 1.4: scale

Models that do not fit one GPU's memory or time.

- **Data-parallel training on several GPUs** of one machine: each GPU a share of the batch, the
  gradients summed in a fixed order (a deterministic all-reduce over peer copies, no NCCL), the same
  bits as one GPU taking the whole batch.
- **Sharded optimizer state** (each GPU keeps the moments of its share of the parameters) and the
  optimizer's state, or parameters, offloaded to the host's memory.
- **Data sets of billions of tokens**: token files mapped into memory and read without copies,
  shuffled windows across files, the GPU's data sets streamed in chunks.
- **Checkpoints** that save and resume a run exactly (the readers' positions, the generators'
  states), sharded across files for large models.
- **More GPU backends' reach**: Metal on Apple GPUs (MoltenVK runs the Vulkan backend there today).

## 2.0: what 1.x cannot add

A major release only for what would break programs; candidates so far:

- tensors of more than three axes and of variable length in the API (sequences of any length per
  call rather than a fixed input layer), which the transformer layers of 1.2 approximate with a
  fixed context;
- layers of the program's own (forward and backward callbacks, with GPU kernels) in networks the
  library trains;
- training across machines.

## Distribution

Where Spingalett can be installed from, and what each channel needs. "Today" means it works from
this repository and its releases alone, without accounts elsewhere.

| Channel | State | What it takes |
|---|---|---|
| PyPI (`pip install spingalett`) | published since 0.12 | trusted publishing from `release.yml` |
| GitHub releases (library, runtime, DigitPad, wheels) | published | `release.yml` |
| CMake package, pkg-config | in every archive | — |
| vcpkg and Conan recipes (`packaging/`) | in the repository | submitting them to vcpkg's registry and Conan Center: pull requests from the maintainer's accounts |
| Homebrew (`brew tap pka-human/spingalett https://github.com/pka-human/Spingalett`) | today, 1.2 | `Formula/spingalett.rb` in this repository; homebrew-core asks for a project with more users first |
| Scoop (`scoop bucket add spingalett https://github.com/pka-human/Spingalett`) | today, 1.2 | `bucket/spingalett.json` in this repository |
| Nix (`nix profile install github:pka-human/Spingalett`) | today, 1.2 | `flake.nix` in this repository; nixpkgs: a pull request |
| Debian, Ubuntu, Fedora (`.deb`, `.rpm` on the release page) | today, 1.2 | built by `release.yml`; a PPA or COPR repository needs the maintainer's Launchpad or Fedora account |
| Container image (`ghcr.io/pka-human/spingalett`) | today, 1.2 | pushed by `release.yml` with its own token |
| Go modules (`go get github.com/pka-human/Spingalett/Bindings/Go`) | today, 1.2 | nothing: Go fetches from the repository |
| Rust (`cargo add spingalett --git https://github.com/pka-human/Spingalett`) | today, 1.2 | crates.io: an API token as the secret `CARGO_REGISTRY_TOKEN`, then `release.yml` publishes |
| .NET (`.nupkg` on the release page) | today, 1.2 | nuget.org: an API key as the secret `NUGET_API_KEY`, then `release.yml` publishes |
| Java (`.jar` on the release page; JitPack builds from tags) | today, 1.2 | Maven Central: a Sonatype account and a signing key |
| AUR (`spingalett`, `spingalett-bin`) | `packaging/aur` in 1.2 | an AUR account with an SSH key: the maintainer pushes the PKGBUILDs |
| conda-forge | `packaging/conda` in 1.2 | a pull request to conda-forge/staged-recipes from the maintainer's account |
| winget, Chocolatey | after 1.2 | manifests submitted from the maintainer's accounts |
| npm (the engine in WebAssembly) | 1.3 | an npm account |

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
kernel of its own for a first layer's few channels (part of 3). The next steps, each measured
against the release before and against PyTorch:

1. **The matrix units' kernel:** stores to shared memory without bank conflicts (a swizzled
   layout), the epilogue writing four results a thread at once, split sums of weight gradients
   chosen by the tile the timing picked rather than by a single-precision estimate; target: cuBLAS's
   17 to 19.5 TFLOPS on the MLP's products, the weight gradients' among them.
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
already overlap the GPU's work: data sets on the GPU (0.14) make it the caller's choice, for data used
again and again. In 0.14, on the RTX 4050: column sums (batch normalization's statistics and backward
sums) read four channels at a time and over more workgroups gained nothing, bound by memory;
inference chunks larger than 32 MB of activations slowed even the MLP, whose layers' outputs stay in
cache between products; training chunks of 8,192 samples were no faster than of 4,096; the next
epoch's first chunk submitted before the last one's losses come back would win 2 to 4% of full-batch
epochs of 6 ms, and is left for now (it runs before early stopping and callbacks decide).

## Later, in no particular release

- **Quantization-aware training:** fake-quantized forward passes with straight-through gradients,
  so INT4 and INT2 models keep their accuracy (INT2 loses most of it on CIFAR-10 today).
- **NEON kernels for training**, so that Apple silicon and AArch64 servers train as fast as x86
  does; SVE where available.
- **Recurrent layers** (GRU, LSTM) and state-space layers (Mamba-style scans), if there is demand.
- **More of the engine on microcontrollers:** CMSIS-style kernels for Cortex-M55 (Helium/MVE),
  RISC-V vector, and a way to run models larger than RAM layer by layer from flash.
