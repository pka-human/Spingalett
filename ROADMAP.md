# Roadmap

Where Spingalett is going after 1.0, roughly in order. Plans change as the work shows what is
worth doing; the [CHANGELOG](CHANGELOG.md) records what was actually done. Every release keeps the
project's rule: nothing gets slower, and new kernels are measured against the previous release and
against PyTorch on the same machine.

## What the work keeps in view

- **Training as much as inference.** Spingalett was made to train efficiently, not only to run what others trained:
  every kind of model the library runs it should also train, on the CPU and on the GPU, as fast as the heavyweights
  or faster (the GPT on the CPU and in bfloat16 on CUDA already is), and in a program's own loop, a step at a time
  in fixed memory. Graphs train from 1.7, as soon as they run on the GPU, not at 2.0.
- **The GPU as much as the CPU.** Most work runs on GPUs. The library's own kernels (no cuDNN or cuBLAS) beat PyTorch
  on dense and convolutional networks and `torch.compile` on the GPT by sound mathematics and algorithms and by
  carrying nothing they do not need; every version that moves the CPU forward plans the GPU's side too, measured
  against the strongest rival on the same card (`torch.compile`, llama.cpp's CUDA and Vulkan builds).
- **Ahead, not level.** Matching the heavyweights is not the aim: every workload the project tracks (training and
  inference, CPU and GPU, layer networks, graphs and language models) runs clearly faster than the strongest rival on
  the same machine (PyTorch eager and `torch.compile` with `max-autotune`, ONNX Runtime, llama.cpp), by 1.3 times or
  more, measured end to end as a program sees it (inputs from and outputs to the host where the rival's are too).
  A margin under 1.1 times counts as a gap: it stays listed below with its cause and the next step until it closes,
  and a release names its gaps in the CHANGELOG.
- **Embedded in other programs.** A C library without dependencies, an engine without a heap and results that
  repeat their bits are what a game engine (creatures' minds, opponents, the same decisions on every client of a
  lockstep game, replays that replay), a robot (trained in simulation, run on the real machine, a failure replayed
  in the simulation bit for bit) or a server can link. Trees (1.11) and learning from rewards (1.12) serve exactly
  there.

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
| 1.4 | "GGUF on the CPU": language models' GGUF files run as they are stored (`spingalett_model_open()`: LLaMA 1 to 3.x, Mistral, Qwen2 and 2.5, Qwen3, Gemma 3, GPT-2), GGUF's quantization types as block precisions of `.slett` format 10 (Q8_0 to IQ4_XS, quantized byte for byte as llama.cpp's quantizers do), computed in their blocks by AVX2 and VNNI kernels with the engine's bits, the attention of today's families (normalized queries and keys, rotary frequencies of their own, embeddings' scales), sessions' attention in vectors, prompts split between the threads by their work; on an i7-12650H faster than llama.cpp on the same files (prompts 1.6 to 2.9 times as fast, generation 1.00 to 1.10 times); checkpoints of training runs that resume to the bits of an uninterrupted run, positions of data set readers |
| 1.5 | "Graphs of operators": any ONNX model run as it is (`spingalett_graph_open()`, `spingalett_graph_run()`, in the runtime, Python and C++): 155 operators with ONNX's semantics over tensors of up to eight axes with symbolic sizes, `.slett` format 11, constants folded, common subexpressions, fusions into the library's kernels (products by constant matrices with their epilogues, attention in blocks, layer and RMS normalization, GELU and SiLU written out, convolutions with their batch normalization and activation), the same bits on any number of threads; ONNX's 1262 conformance cases, whole models against PyTorch and ONNX files of other tools against ONNX Runtime; language models in programs (tokenizers in the runtime, chat templates, sampling as llama.cpp offers it, SmolLM3, OLMo 2, Phi-3 and GLM-4 from GGUF, a chat on the runtime alone) |
| 1.6 | "On the GPU": graphs of operators on CUDA and Vulkan (values in bfloat16 between products, a fused attention reading where products leave their rows, format 12); GGUF's language models on the GPU from their images, weights in their blocks, with the CPU's bits in every block product; models larger than the GPU's memory split with the CPU |

## Every model: graphs of operators

Up to 1.4 a model is a chain or graph of the library's fifteen kinds of layers over one sample of at most three
axes, each kind a record of fixed fields, and every family of models beyond them is code of the library's: a new
field in the record, a branch in the kernels of the CPU, CUDA and Vulkan, a branch in the converters. That grows
with every family rather than with every new primitive, and it leaves out whatever is not made of those layers:
models of several inputs (diffusion's latents, timesteps and texts), tensors of four axes and more, broadcasting,
group normalization, the text encoders of image models.

From 1.5 a model can be a graph of operators over tensors of any rank, with its weights and its configuration. The
operators are primitives at the level of ONNX's kernels and a few composite ones with a reference decomposition into
them; the library's layers and kernels (attention, normalizations, gated MLPs, the block precisions, sessions) are
what its passes fuse a graph into, not what a model must be made of. A graph of known operators runs on any later
runtime without code of its family, slowly where no pass applies; the code of an architecture lives only in the
converters (ONNX, GGUF, Hugging Face's directories, PyTorch) and in the passes. The models of formats 3 to 10, the
engine for microcontrollers and the training API go on as they are: the graph is a new path beside them, which the
layer models join once it runs them no slower (1.6).

The target that keeps this honest is diffusion: SDXL and FLUX.1 run with no line of code specific to diffusion in the
executor (1.8). Language models keep their bar: on the same GGUF files faster than llama.cpp (`llama-bench`: prompt
processing of 512 tokens, generation of 128, at 1 and 8 threads, on CUDA and on Vulkan) and against PyTorch, each
family checked on real checkpoints (the logits of a prompt against transformers', the tokens of greedy generation
against llama.cpp's). **SaLaM** (below), a program of its own on the runtime, serves them.

## 1.5: graphs of operators (released)

The foundation, on the CPU, with the language-model work of 1.5 that needed no new format.

- **The IR**: tensors of up to eight axes, their sizes numbers or named symbols, the symbols checked at every run;
  float32, float16, bfloat16, int64, int32, int8, uint8 and bool; several named inputs and outputs. 155 operators with
  ONNX's names and semantics (opsets 7 to 28, earlier forms where they differ): element-wise arithmetic, comparisons,
  logic and bitwise operators and functions with broadcasting, products and convolutions of one to three spatial
  axes, pooling, normalizations, reductions, shapes and data, Resize in every mode, quantization, Einsum, opset 23's
  Attention and RotaryEmbedding, opset 27's LinearAttention and CausalConvWithState, SwiGLU; three of the library's
  own that fusions make (Linear, FusedAttention, Conv with an activation).
- **`.slett` format 11**: values (type, shape with symbols), nodes, typed attributes, the opset, constants at offsets
  of 64 used where a mapped file holds them; checksums of the tables, not the constants.
- **The executor** (`spingalett_graph_open()`, `spingalett_graph_run()`; in the runtime and Python): kernels
  for every operator that sum in fixed orders (the same bits on any number of threads), the library's row kernel
  under products by constants (packed once) and its channels-last kernels under convolutions, a pool of buffers kept
  between runs, values freed after their last reader, shape operators as views.
- **Passes**: constant folding, dead nodes, common subexpressions; fusions: Linear (a product by a constant matrix,
  its bias and activation), FusedAttention (Q K^T, scale, mask, softmax and V in blocks of 64 queries, the heads'
  transposes taken in), GELU and SiLU from their written-out forms, a convolution's activation; each compared with
  the graph without it.
- **Converters**: ONNX one operator to one; PyTorch through `torch.onnx` in Python (`Graph.from_torch()`).
- **Checked**: ONNX's 1262 conformance cases of single operators that hold the library's types (the `operators`
  group, on one and four threads, under the sanitizers of GCC and Clang); whole models exported from PyTorch against
  PyTorch (`Tests/check_graph_models.py`): ResNet-50, ViT-B/16, BERT-base, GPT-2, CLIP's text encoder, Whisper's
  encoder, Stable Diffusion's VAE decoder, a UNet of SDXL's architecture, within 1e-5 and at 0.63 to 0.94 times
  PyTorch's speed; transformers.js's ONNX files against ONNX Runtime (`Tests/check_graph_files.py`). The layer
  models and their benchmarks are untouched and no slower (Examples/Benchmark.c's CPU workloads and GGUF sessions
  measured against 1.4).
- **Language models in programs**: the tokenizers in the runtime; chat templates (the Jinja of
  `tokenizer.chat_template` rendered as transformers renders it); sampling as llama.cpp offers it (min-p, repeat,
  frequency and presence penalties, stop strings, a stream of tokens) and draws that sort only what they reach;
  SmolLM3, OLMo 2, Phi-3 and GLM-4 from GGUF files on 1.4's layers; a chat in the terminal on the runtime alone.
- **Decided**: an opset of the library's own, ONNX's semantics where ONNX has them (free references) and its own
  operators where it has not; no control flow in graphs (generation loops and samplers stay code of the host);
  training through graphs waits for 2.0 (the training API stays the layers'; brought forward to 1.7, beside it);
  the engine for microcontrollers runs layer models until a static subset of graphs reaches it (1.13).
- **Moved to 1.6**, where language models and the GPU need them: GGUF's block types as types of constants, the
  metadata of a model in format 11 (a tokenizer, a chat template, generation's defaults), shapes inferred when a
  graph opens (kernels compute them as they run in 1.5) and composite operators' decompositions into primitives
  for backends without their kernels.

## 1.6: graphs and language models on the GPU (released)

What 1.6 holds: graphs of operators on CUDA and Vulkan, GGUF's language models on the GPU from their images in their
blocks, models larger than the GPU's memory split with the CPU. What was planned for it besides went to 1.7 (below): the GPU
was ready to ship, and SaLaM waits for it.

- **Graphs on the GPU ahead of PyTorch by a margin** (the aim above). Where dev/1.6 stands on an RTX 4050 Laptop GPU
  against PyTorch 2.14.1 (ms, medians; Spingalett's times take the inputs from and the outputs to the host, PyTorch's
  with its tensors already on the GPU, then with them on the host too):

  | Model | bfloat16 | PyTorch fp16 autocast (on GPU / from host) | single precision | PyTorch fp32 |
  |---|---:|---:|---:|---:|
  | ResNet-50 | 1.86 | 1.90 | 3.50 | 3.71 |
  | BERT-base, 2 x 128 tokens | 3.40 | 3.79 / 3.94 | 10.1 | 9.0 |
  | GPT-2, 64 tokens | 3.76 | 3.03 / 4.19 | 7.2 | 4.7 |
  | CLIP's text encoder | 1.86 | 1.94 | 4.74 | 5.13 |
  | Whisper's encoder | 3.51 | 3.04 | 8.59 | 9.74 |
  | ViT-B/16 | 3.47 | 3.30 | 10.2 | 8.6 |
  | A UNet of SDXL's architecture | 4.01 | 13.6 | 8.05 | 11.45 |
  | Stable Diffusion's VAE decoder | 54 | 58 | 138 | 80 |

  None is yet 1.3 times ahead but the UNet. What is left, by the profiles: attention on the matrix units at 10 TFLOPS
  (Whisper, ViT; the softmax between the products, blocks that do not fill the device), copies around attention where
  exporters split a joint QKV product (GPT-2), normalizations written twice for their readers in two precisions, the
  residual additions and their activations as passes of their own (ResNet), convolutions of single precision in VAE's
  sizes, products of few rows bound by their weights' memory (GPT-2, CLIP); then `torch.compile` as the rival.
- **Language models on the GPU, against llama.cpp's Vulkan build on the same files.** On CUDA prompts run 1.21 to 1.37 times
  as fast; generation is level (0.99 to 1.12), since a token's products are bound by the memory's bandwidth, which both
  reach to within a few percent: 1.3 times is not there to be had, and it counts as a gap only where it falls under level.
  Under 1.3 for prompts: Qwen3-1.7B, Qwen3-4B and Llama-3.2-1B (the copies of few bytes from many rows into shared memory,
  a third of bmm's time in Q4_K).
- **Vulkan behind CUDA**: the kernels exist for every type with the CPU's bits, prompts on matrix units of 8-bit integers
  where the device has them, but sessions are still a gap on NVIDIA's Vulkan driver (the table in the CHANGELOG): no
  normalizations with the sums and quantizations around them (bnorm), no gated units' pairs or joined products in bmm.comp's
  pipelined loads. AMD's and Intel's GPUs are not measured, for want of one.
- **A model split between the GPU and the CPU**: Qwen3-8B Q4_K_M on a 6 GB card keeps all but its output layer on the GPU,
  which costs 0.83 times llama.cpp's generation (10 ms a token on the CPU) where llama.cpp fits it whole; copies of weights
  could overlap the products that do not read them (one stream on CUDA today).

## 1.7: training through graphs

Planned for 2.0 until 1.6: any model the library runs, it trains. Added beside the layer networks' API (new
functions, nothing of 1.x removed), on the executors of 1.5 and 1.6, so that a model is trained where it runs.

- **From 1.6, language models as graphs**: GGUF's block types as types of constants, a model's metadata in format 11 (a
  tokenizer, a chat template, the family as a hint to the passes), composite operators' decompositions; `Gguf.c` and
  `Pretrained.c` as generators of graphs, families added as converters alone (Gemma 2, Phi-4 mini, YaRN, Granite, Mistral's
  and Qwen's newest dense models), Hugging Face's directories quantized as they are read; sessions of graphs (a session's
  named state, a prompt's cache kept between requests, the context shifted when it fills), no slower than 1.4's on
  `llama-bench`; the layer models read as graphs where that is no slower.
- **From 1.6, the rest**: the block kernels on NEON with the dot product and AVX-512's 512-bit vectors; SentencePiece's
  Unigram and WordPiece tokenizers; operators of the program's own, registered with a reference kernel.
- **Gradients of operators**: for every operator its derivative written as operators of the IR (MatMul's two
  products, Conv's data and filter gradients as the transposed convolutions the kernels already have, the reductions'
  broadcasts back, Gather's scatter as a fixed-order sum), a backward graph built when a graph is opened for training
  and fused as forward graphs are; the fused operators' backward passes are the layer networks' kernels (attention's
  flash backward, layer and RMS normalization, convolutions channels last, GELU and SiLU from the values before them).
- **The same bits on any number of threads and every run on one GPU**: a value read by several nodes gets their
  gradients in a fixed order (first written, the rest added), sums over fixed chunks; the layer networks' rule kept.
- **The training loop of the library**: the graph's constants marked trainable (an ONNX initializer, a safetensors
  tensor), losses as operators (cross-entropy over classes and tokens, mean squared error), the optimizers,
  schedules, gradient accumulation, clipping, mixed precision (bfloat16 on the matrix units, weights in single
  precision) and checkpoints that resume to the bits of an uninterrupted run, a step at a time in a program's loop
  in planned memory. Python: `Graph.from_torch()` modules fine-tuned without PyTorch at run time.
- **Fine-tuning language models on one consumer GPU**: LoRA adapters on a frozen base (its products in bfloat16, or
  in 1.4's block types: QLoRA on Q4 and Q8 weights the GPU keeps in their blocks), only the adapters' gradients and
  moments kept, the base's activations recomputed where memory asks; full fine-tuning where it fits, with the
  optimizer's moments in bfloat16 or on the host. Target: Qwen3 1.7B and 4B fine-tuned on a 6 GB laptop GPU (an RTX
  4050), Qwen3 0.6B fully, against PyTorch with PEFT (and bitsandbytes for QLoRA).
- **Measured** against PyTorch eager and `torch.compile` training the same models on the same machine (ResNet-50,
  ViT-B/16, BERT-base fine-tuned, the SDXL-style UNet, Qwen3 0.6B to 4B), by the margin of the aim above; the layer
  networks' training then moves onto graphs where it is no slower, as their inference does in 1.6.

## 1.8: diffusion

- **SDXL first**: its UNet, VAE and text encoders (CLIP-L, OpenCLIP bigG) from diffusers' directories and GGUF files,
  in float16 and block dtypes; then **FLUX.1** (schnell and dev) with T5-XXL (Unigram) and its double and single
  stream blocks, the only code of either being its converter.
- **The GPU's kernels and fusions** for what diffusion spends its time in: group normalization with SiLU, adaLN's
  scale and shift, attention over the tokens of an image, convolutions in bfloat16 and block dtypes, memory planned
  for 1024 x 1024 images on a laptop's GPU.
- **Sampling outside the graph**: Euler, DPM++ 2M and flow matching (with its shifted schedule), classifier-free
  guidance as a batch of two, seeds and noise the same on every backend; a library of the runtime that SaLaM and the
  examples call.
- Measured against diffusers with `torch.compile` and against stable-diffusion.cpp on the same files: seconds an
  image at 1024 x 1024 on the CPU, CUDA and Vulkan.

## 1.9: mixture of experts

- **A layer of experts** with a learned router (top-k gating, softmax or sigmoid scores, shared
  experts), the experts' products batched by expert, in inference and in training (the load-balancing
  loss): Mixtral, Qwen3's and Qwen3.5's MoE models, gpt-oss, Llama 4, GLM-4.5 Air, OLMoE. In graphs a composite
  operator (`TopK`, `Softmax`, gathers and batched `MatMul` its reference decomposition), with kernels of its own.
- **Attention sinks** (gpt-oss: a learned logit in each head's softmax) and sliding windows alternating
  with full attention by layer; MXFP4 weights (gpt-oss's GGUF files), as a dtype.
- **Experts kept on the host** while attention and the dense layers run on the GPU (the offload that
  makes 30B models of 3B active parameters fit a laptop's GPU), and layers split between the CPU and the
  GPU for any model larger than the GPU's memory: placement as a pass over the graph.
- **Speculative decoding** with a draft model (several tokens checked in one pass, the bits of plain
  sampling kept).

## 1.10: hybrid models: linear attention and state spaces

- **Gated DeltaNet** (Qwen3.5, Qwen3-Next) with its short causal convolution and gated output, and
  attention with a gated output beside it; **Mamba-2** (Falcon-H1, Granite 4, Nemotron-H). Layers whose
  state is a matrix carried from token to token instead of a key-value cache: sessions keep it per
  sequence (rewound by checkpoints of the state), training runs the chunked parallel form, the GPU
  backends scan in fixed orders. In graphs the scans are composite operators whose reference decomposition is
  their chunked form, the state the session's named state of 1.6. With 1.9's experts this opens every Qwen3.5
  model, Qwen3.5-2B among them.
- **Recurrent layers** (GRU, LSTM) on the same scans, as layers of the same graphs as attention.
- **Multimodal inputs**: the vision encoders of these models (Qwen3.5's, Gemma 3's: patch embeddings,
  their own attention) and audio frames, inputs of several kinds in one graph.
- **ALiBi positions** and block-sparse masks in the flash kernels, for the models that use them.

## 1.11: beyond neural networks: trees and the classical toolbox

Most of the machine learning in production still runs on tables, and there gradient-boosted trees
(XGBoost, LightGBM, CatBoost) beat neural networks; they are also the models most often run on
microcontrollers. Spingalett's promises carry over: the same bits on any number of threads and on the
GPU, kernels measured against the best libraries, models run by the engine without a heap.

- **Gradient-boosted decision trees**: features binned once into at most 256 bins (quantiles,
  categorical values, a bin for missing values), histograms of gradients and hessians summed over fixed
  blocks of rows in a fixed order (the same trees on 1 thread or 16, and on the GPU, whose histograms
  sum in shared memory without atomics), trees grown by depth or by leaf, L1 and L2 regularization,
  minimum hessians and gains, rows and columns sampled from a seed, missing values sent the learned way,
  categorical splits, monotone constraints; the library's losses (squared error, logistic, softmax over
  classes) and Poisson, quantile and ranking (LambdaMART) objectives; validation, early stopping,
  callbacks and checkpoints as `spingalett_train()` has them. **Random forests** and extremely
  randomized trees on the same machinery.
- **Trees in deployment**: ensembles in a `.slett` section of their own, predicted in batches by
  branch-free traversal (QuickScorer's bit vectors, SIMD over samples), on the GPU, by the engine
  (thresholds on bins as integers: no floats needed on a microcontroller) and the runtime, exported as C
  headers; models imported from XGBoost (JSON), LightGBM (text) and scikit-learn through ONNX-ML's tree
  ensembles.
- **The classical toolbox** on the existing matrix kernels: linear and logistic regression with L1 and
  L2 penalties (coordinate descent, L-BFGS), k-means (k-means++ seeding, mini-batches, on the GPU), PCA
  (randomized SVD), exact k-nearest neighbours; standardization and one-hot encoding saved with a model
  as its first steps; metrics (ROC AUC, F1, confusion matrices) and cross-validation splits.
- **In Python**, estimators with scikit-learn's interface (`fit`, `predict`, `predict_proba`), so that
  they drop into existing pipelines.

Measured against XGBoost, LightGBM and CatBoost (time to the same validation loss and prediction
throughput on Higgs, Covertype and Epsilon, on the CPU and the GPU) and against Treelite's compiled
trees.

## 1.12: learning from rewards: reinforcement learning and evolution

Networks trained by the consequences of what they do rather than by labelled examples, wherever a
program can score an outcome: characters and opponents in games (a character that learns to walk,
rewarded for distance covered and penalized for falling), the controllers of robots and drones trained
in simulation, the control of machines, buildings' heating and data centres' cooling, traffic signals,
bidding and trading, recommendations and prices chosen as bandits choose, decisions inside compilers and
databases (as LLVM's inlining policy), scheduling and routing, and language models tuned by rewards and
preferences. Evolution takes what has no gradient at all: a design's dimensions, a strategy's rules, a
game's balance, a network's own shape and hyper-parameters. The library is a C library with a runtime and
an engine without dependencies, which is what a game, a robot, a compiler or a server can link, and its
training repeats its bits, which reinforcement learning, known for runs that cannot be reproduced, rarely
does.

- **Environments as the program has them**: the program steps its world (a physics engine, a game's
  frame loop, a simulator of a plant or a market, logs replayed) and the library acts and learns, in
  either direction of control: the library calls the program's `reset` and `step` (observations, a
  reward, whether the episode ended or was cut short), or the program's loop calls the agent (`act` for
  each agent's observation, `reward` and `end` as they come), for one agent or thousands stepped side by
  side on the program's threads. Observations and actions of any shape: vectors, images, discrete
  choices, continuous ones, several at once. Rewards may be given as named terms (forward speed, a fall,
  energy spent, a constraint broken), logged one by one, since shaping them is most of the work.
- **Algorithms**: PPO (discrete actions and continuous ones from Gaussian policies, generalized
  advantage estimation, clipping, an entropy bonus), SAC for continuous control from a replay buffer,
  DQN with its double and dueling variants for discrete actions; observations and rewards normalized by
  running statistics saved with the policy; recurrent policies on 1.10's layers. Several agents in one
  environment, each with a policy of its own or one shared, and self-play against a league of their past
  versions; tree search guided by a network's policy and value (AlphaZero's) for board games and for
  planning where the program can simulate a move. Every draw from a seed, the environments' too: a run
  repeats its bits on any number of threads, and resumes from a checkpoint as supervised training does.
- **Bandits** for decisions whose reward comes at once (which item to recommend, which price or variant
  to show): epsilon-greedy, LinUCB and Thompson sampling over 1.11's linear models or over networks, and
  a new policy evaluated on logged decisions before it serves (inverse propensity and doubly robust
  estimates).
- **Learning from logs** where exploring is costly or unsafe: behaviour cloning from recorded play or
  operation (a person's demonstrations, a running system's decisions) to start from before rewards take
  over, and offline reinforcement learning (implicit and conservative Q-learning) on recorded
  transitions, read as `.slettd` data sets.
- **Language models from rewards and preferences**: the transformers of 1.4 to 1.10 tuned by GRPO (groups
  of completions sampled by the sessions' batched generation, scored by the program's function: tests
  passed, an answer checked, a format kept) and by DPO from pairs of preferred and rejected answers;
  low-rank adapters (LoRA), which train a small share of the parameters, serve supervised fine-tuning as
  well, and are merged into the model or kept beside it for the runtime to apply.
- **The policy deployed** as any network: a `.slett` file the game's runtime evaluates each frame in
  microseconds, a server per request, the engine on a robot's or a drone's microcontroller.
- **Evolution**, for problems without gradients and for policies that rewards alone train badly: genetic
  algorithms over a network's parameters or any genome (real numbers, integers, bits, or permutations
  such as orders of jobs and routes, with the operators each needs), with populations, tournament
  selection and elitism, crossover by gene, by layer, at points or by order, and Gaussian, reset and
  swap mutations at rates that may decay; several objectives at once with their Pareto front (NSGA-II);
  fitness from the program's callback (the distance a character walked, a design's simulated strength, a
  schedule's length) evaluated in parallel. Evolution strategies (OpenAI's, CMA-ES for small genomes)
  whose perturbations are rebuilt from seeds rather than stored; topologies evolved as graphs of layers
  (neurons and layers added or removed by mutation, as NEAT does); population-based training, which
  trains networks side by side and evolves their hyper-parameters (learning rates, dropout, reward
  weights) by copying the best and perturbing them. Seeded throughout: a population evolves the same way
  on any number of threads.
- **In Python**, Gymnasium's environments wrap directly (`sg.rl.PPO(env)`), PettingZoo's for several
  agents; examples train a walker in a physics simulation, a character in a Godot scene through the C
  API, a bandit on logged clicks, a schedule by evolution and a small language model on arithmetic by
  GRPO.

Measured against Stable-Baselines3 and CleanRL (wall time to the same returns on Hopper, Walker2d,
HalfCheetah and Humanoid in MuJoCo and on Atari games, on the CPU and the GPU), Vowpal Wabbit for bandits,
TRL for GRPO and DPO, and evosax, EvoTorch, DEAP and pymoo for evolution.

## 1.13: Arm, Apple and the edge

The CPU and GPU paths that 1.0 to 1.12 tuned on x86 and NVIDIA, brought to the other hardware the
library runs on.

- **NEON kernels for training** (and SVE where available), so that Apple silicon and AArch64
  servers train as fast as x86 does; AArch64 matrix kernels chosen at run time as x86's are.
- **A Metal backend** for Apple GPUs (MoltenVK runs the Vulkan backend there today), an executor of
  graphs like 1.6's, with bfloat16 products where the GPU has them and GGUF's block types.
- **Quantization-aware training**: fake-quantized forward passes with straight-through gradients,
  so that INT4 and INT2 models (and language models' INT4 weights) keep their accuracy (INT2 loses
  most of it on CIFAR-10 today).
- **More of the engine on microcontrollers**: CMSIS-style kernels for Cortex-M55 (Helium/MVE) and
  RISC-V vector, transformers' layers within a microcontroller's memory, and models larger than RAM
  run layer by layer from flash. **Graphs in the engine**: the subset of 1.5's graphs of static shapes
  and operators whose workspace is known when they are exported, run by the engine without a heap (C99,
  as now) and in WebAssembly; an export outside the subset names the operator that keeps it out.

## 1.14: scale

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
without dependencies, packaged where the library is. It waits for its repository (the maintainer's account: 0.1's
part of the library shipped with 1.5); `Examples/Runtime/Chat.c` is its seed.

- **0.1, with 1.5** (the runtime's tokenizers, chat templates and sampling are there): `salam run model.gguf -p
  "..."` (a prompt's continuation), `salam chat` (the model's chat template, the conversation's cache kept between
  turns), `salam bench` (prompt processing and generation in tokens per second, as `llama-bench` reports them), GGUF
  files, sampling options as llama.cpp's, models fetched from Hugging Face by name.
- **0.2, with 1.6**: the GPU (CUDA or Vulkan by flag), a model split between the GPU and host memory by flag or by
  the GPU's free memory (`SpingalettSessionOptions.gpu_memory`), a server with an
  OpenAI-compatible HTTP API (completions, chat, streaming) that batches its requests in one session.
- **0.3, with 1.8**: `salam image` (SDXL and FLUX.1: a prompt, a seed, a sampler, steps).
- **0.4, with 1.9 and 1.10**: the experts' offload, speculative decoding with a draft model,
  the hybrid models, images in chats (the vision encoders), grammars that constrain generation (GBNF, JSON schemas).

## 2.0: what 1.x cannot add

A major release only for what would break programs; candidates so far:

- the network of fixed layers replaced by graphs (trained through graphs from 1.7, beside it): the layer API
  removed once every program it serves has a graph to move to;
- layers of the program's own in networks the library trains (their backward passes; operators of the
  program's own run in graphs from 1.6);
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
`benchmark_pytorch.py --cuda`, `--cuda-fp32`, `--cuda-bf16` and `--host-data`, [Performance](docs/Performance.md)), Spingalett
0.14 is ahead in every workload, with the data in GPU memory (its data sets) or in host memory:
the MNIST CNN (with and without batch normalization) trains 1.6 to 1.9 times as fast as PyTorch and
runs 2.1 to 2.4 times as fast in single precision, 1.5 to 1.6 and 1.8 to 2.1 times as fast as its
autocast in bfloat16; ResNet-20 and the U-Net run 1.75 to 1.10 times as fast in either precision and
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
whose margins over PyTorch are wider in every workload (1.2 to 7.7 times; [Performance](docs/Performance.md)); on both backends
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
