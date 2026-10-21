# Changelog

All notable changes to this project are documented in this file. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/): from 1.0 only a major release breaks the API or ABI
(before 1.0 a minor release could, and listed such changes under **Changed**).

## [Unreleased]

### Added
- **Chat templates**: `spingalett_chat_render()` (`SpingalettChatArgs`, `SpingalettChatMessage`) renders a conversation
  with a language model's Jinja chat template as Hugging Face's transformers renders it (`apply_chat_template()`:
  jinja2's sandbox with trim_blocks and lstrip_blocks, loop controls, its `tojson`, `raise_exception()` and
  `strftime_now()`), and `spingalett_tokenizer_chat_template()` gives a tokenizer's: a GGUF file's
  `tokenizer.chat_template`, a directory's `chat_template.jinja` or `tokenizer_config.json`'s `chat_template`. The
  renderer (`Spingalett.Chat.c`, in the runtime too) has the part of Jinja templates use: statements (`if`, `for` with
  its conditions, unpacking, `loop` and loop controls, `set` of names, namespaces' attributes and blocks, macros,
  `generation`), Python's values and operators, slices, the methods of strings and dicts, 30 filters and 25 tests;
  messages come as roles and texts, or with tool calls and parts in a JSON object of variables that also carries tools,
  documents and the flags templates read (`enable_thinking`). The new `chat` group renders 12 conversations with the
  templates of Qwen2.5, Qwen3, Llama 3.2, Gemma 2 and 3, SmolLM2 and 3, Phi-3.5 and Phi-4 mini, Mistral 7B v0.3,
  DeepSeek-R1's distillations, Granite 3.3, OLMo 2, GLM-4 (two) and Falcon 3 to transformers' text byte for byte, or
  to their errors (192 cases, `Tests/make_chat_tests.py`). Python: `Tokenizer.chat_template`,
  `Tokenizer.apply_chat_template(messages, add_generation_prompt, tools, **variables)` and `render_chat_template()`.
- **`Examples/Runtime/Chat.c`**: a chat with a GGUF model in the terminal on the runtime alone: the model's chat
  template renders the conversation every turn, the session keeps the tokens it shares with the turn before and
  is fed the rest, the reply streams as it is drawn (min-p, top-p, temperature, a repeat penalty), and ends at EOS or
  a token that ends turns in the common templates.
- **Streams of tokens**: `SpingalettSampling.on_token` (`SpingalettTokenCallback`) is called with each token
  `spingalett_session_generate()` and `spingalett_generate()` draw, as they draw it; false ends the sequence.
  Python's `generate()` methods take `on_token`.
- **SmolLM3, OLMo 2, Phi-3 and GLM-4 from GGUF files**: `spingalett_model_open()` opens the architectures `smollm3`
  (every fourth layer without rotary embeddings), `olmo2` (the attention's and the MLP's outputs normalized and not
  their inputs, queries and keys normalized over all heads as layers of their own; OLMo 3's windows), `phi3` (Phi-3
  and Phi-3.5: queries, keys and values in one matrix, the gate and up in one, LongRoPE's short or long factors by
  the window and its magnitude of the rotations, which scales the scores) and `glm4` (half of each head turned, in
  llama.cpp's adjacent pairs, the outputs normalized); rotary embeddings on part of a head where the file's
  `rope.dimension_count` says so. The `pretrained` group compares each with transformers' model (within 3.2e-6).
- **Sampling as llama.cpp offers it**: `SpingalettSampling`, given to `spingalett_session_generate()` and
  `spingalett_generate()` through their new `sampling` field (which takes a word of their reserved space): min-p
  (after the temperature, top-k and top-p, as transformers applies it), llama.cpp's repeat, frequency and presence
  penalties over a window of the last tokens of the prompt and those generated (64 by default, or all), and stop
  strings, which end a sequence at the token whose text completes one (decoded by the tokenizer given). The `gen`
  group checks every option against a reference that sorts every token, on the logits of the same session; Python's
  `generate()` methods take them as keyword arguments, and the Rust, .NET, Java and PHP bindings know the field.
- **Tokenizers in the runtime**: `libspingalett-runtime` holds the tokenizers (`spingalett_tokenizer_load()` and the
  rest, declared now in `Spingalett.Runtime.h`), GGUF vocabularies included, so that a program that serves a
  language model needs the runtime alone; its tests check them on the expectations of JSON and GGUF tokenizers.

### Changed
- **Draws sort only what they reach**: a draw sorted every token of the vocabulary by probability (152,000 for
  Qwen2.5) for every token it drew; it now sorts them a part at a time, the most likely first, as far as top-k, top-p,
  min-p and the draw itself reach, and sums the probabilities of a draw without top-k in the order of the tokens. A
  session of Qwen2.5-0.5B-Instruct (Q4_K_M) draws 128 tokens at temperature 0.8 and top-p 0.95 at 104.9 tokens a
  second instead of 42.0 (the most likely tokens: 113.9), on six performance cores of an i7-12650H. The tokens of a
  draw without top-k may differ from 1.4's where a draw falls within rounding of two tokens' boundary.

## [1.4.0] - 2026-10-21

"GGUF on the CPU": language models' GGUF files run as llama.cpp's users have them, faster than llama.cpp on the
same processor. `spingalett_model_open()` makes a deployment model of a GGUF file of LLaMA 1 to 3.x, Mistral,
Qwen2 and 2.5, Qwen3, Gemma 3 or GPT-2 with its tensors as the file stores them: GGUF's quantization types are
block precisions of `.slett` format 10 (Q8_0 to IQ4_XS, quantized byte for byte as llama.cpp's quantizers do),
computed in their blocks by AVX2 and VNNI kernels with the engine's bits (chosen at run time in the packages),
with the attention of today's families (normalized queries and keys, rotary frequencies of their own, scaled
embeddings). On the six performance cores of an i7-12650H prompts of 512 tokens run 1.6 to 2.9 times as fast as
llama.cpp's on seven files of Qwen2.5, Qwen3, Llama 3.2 and Gemma 3, and generation 1.00 to 1.10 times as fast
(README, **Performance**). Training runs resume from checkpoints to the bits of a run that never stopped. Programs
of 1.3 build and run unchanged; files of 1.3 load unchanged.

### Added
- **GGUF files run as they are stored**: `spingalett_model_open()` (`SpingalettModelOpenArgs`: the path and the
  window), and `spingalett_model_load()` on a GGUF file, make the deployment model of a language model's GGUF
  file directly, its tensors copied into the model's `.slett` image as the file stores them (the quantized types
  as block precisions, below; F16, BF16 and F32 as FP16, BFLOAT16 and FLOAT32), without the float network of
  `spingalett_load_pretrained()`: a 1B model of 800 MB takes 800 MB and opens in under a second. The
  architectures `llama` (LLaMA 1 to 3.x, with LLaMA 3's rotary frequencies of `rope_freqs.weight` and linear
  scaling; Mistral), `qwen2`, `qwen3`, `gemma3` and `gpt2`; queries', keys' and values' tensors of different types
  (Q4_K_M keeps `attn_v` in Q6_K in half the layers) become layers of their own, concatenated. On
  Qwen2.5-0.5B-Instruct (Q4_0, Q4_K_M, Q8_0), Llama-3.2-1B-Instruct (Q4_K_M), Qwen3-0.6B (Q8_0) and Gemma-3-1B-it
  (Q4_K_M) the most likely token of every position of a prompt is llama.cpp's; the `pretrained` group checks
  tiny LLaMA, LLaMA 3, Qwen2, Qwen3, Gemma 3 and GPT-2 checkpoints, converted as llama.cpp's converter writes them,
  against PyTorch and transformers (within 3e-6). The runtime opens GGUF files too (`Spingalett.Gguf.c`,
  `Spingalett.Image.c`: images written layer by layer), Python's `Model.load(path, context=0)` and PHP's
  `Model::load($path, $context)` take the window, and `ModelTool convert model.gguf model.slett --context N`
  writes the model as a `.slett` file (`ModelTool info` describes either, with every kind of layer named).
- **The attention of today's language models** in `.slett` format 10, in the engine, batched prediction and
  sessions: each head's queries and keys RMS-normalized before their rotation (Qwen3's and Gemma 3's `q_norm` and
  `k_norm`), rotary frequencies of the layer's own (LLaMA 3's and linear scaling), part of a head rotated, a scale
  of the scores; embeddings that scale their vectors (Gemma's square root of the width). Networks
  (`spingalett_load()`, training, the GPU) refuse them by name for now.
- **Block precisions**, GGUF's quantization types as deployment precisions (`.slett` format 10):
  `SPINGALETT_PRECISION_Q8_0`, `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, the k-quants `Q2_K` to `Q6_K`, `IQ4_NL` and
  `IQ4_XS`, for dense layers, linear layers and embeddings, each row stored as llama.cpp stores a row of the
  type (docs/ModelFormat.md, **Block precisions**). A network saved or made a model in one quantizes its weights
  as llama.cpp's quantizers do without an importance matrix, byte for byte (checked against ggml's), layers whose
  rows are no whole blocks taking llama.cpp's fallback types. Such a layer quantizes its input to 8 bits in
  blocks of the type's size and sums each block's sub-blocks in integers, scaled by their integer scales, the
  blocks' terms then added in float block after block: the engine, batched prediction, sessions and the runtime
  give the same bits on any number of threads (the `model` group: dense layers of every type within the bound
  of the inputs' quantization, and language models in every type whose engine, batched and session logits are
  equal). `spingalett_load()` reads such files with the weights decoded as llama.cpp's `dequantize_row_*()`
  decodes them. In every binding, and `ModelTool` and `RunModel` name them. With AVX2 their kernels run a
  token's rows eight at a time against its codes (dpbusd with VNNI; the k-quants' sub-blocks scaled in 16 bits,
  dpwssd), the codes taken into registers as they are read and the next rows prefetched, and a prompt's tokens in
  groups of eight with each row's codes broadcast against a group's, the sums exact in the lanes: the bits of
  the engine's reference, which the `model` group compares. x86-64 builds for a baseline (the release packages
  and wheels) carry these kernels for AVX2, AVX-VNNI and AVX-512 VNNI, and sessions' attention for AVX2, and
  choose them at run time, with the same bits.
- **Checkpoints of training runs**: `spingalett_save_checkpoint()` and `spingalett_load_checkpoint()` (in
  Python `Network.save_checkpoint()` and `Network.load_checkpoint()`) keep a network, its optimizer state
  and the state of the `spingalett_train()` run that trained it, as its last epoch left it: the seeds of
  its dropout masks, augmentations and epochs' orders, Adam's bias corrections as its steps multiplied
  them (which `powf()` of the step count does not give), its epochs and steps (which schedules by epoch
  and by step count), reduce-on-plateau's scale, the best epoch and early stopping's patience, with
  `restore_best_weights` the best epoch's parameters, and the place of the reader it read through
  `spingalett_dataset_generator()`. The `spingalett_train()` after `spingalett_load_checkpoint()`, given
  the run's arguments again with `.epochs` all the run's epochs, runs the epochs left, its reader moved to
  the run's place, and a run stopped at an epoch's end and carried on so trains the same bits as one that
  never stopped, on any number of threads (the new `resume` group: 23 runs of two parts, dense,
  convolutional and language models, arrays, streamed and in-memory `.slettd` readers and token files,
  schedules by epoch and by step, gradient accumulation, plateaus, early stopping, best weights restored
  from before the checkpoint, callbacks that draw from the generator, checkpoints written by autosaves).
  Autosaves in FLOAT32 with the optimizer state, the defaults, are checkpoints. A checkpoint is a FLOAT32
  `.slett` image with a record of the run after it, so `spingalett_load()`, the engine and the runtime read
  it as a model (docs/ModelFormat.md, **Checkpoints**). From pull request 11 by Nazar Ponomarenko, whose
  record held the library's generator and replayed the shuffles of the epochs already run, kept a resumed
  network resuming in every later call, and restored best weights it did not hold.
- **Positions of data set readers**: `spingalett_dataset_get_position()` and
  `spingalett_dataset_set_position()` with `SpingalettDatasetPosition`: the reader's seed, the pass it
  stands in and the samples of that pass already read, which name the samples to come exactly, since a
  pass's order is a function of the seed and the pass number. A reader moved to a position reads what
  it would have read from there, bit for bit, on any number of threads, with or without its decoding
  thread (chunks decoded ahead are dropped; a chunk the position lies inside is decoded at once), for
  `.slettd` files streamed or kept in memory and for token files. Contributed by Nazar Ponomarenko
  (pull request 11).

### Changed
- **Sessions' attention in vectors**: a session keeps each head's keys transposed in blocks of eight positions,
  sums eight scores side by side (each over the head's values in their order, as the engine sums one) and
  computes the online softmax's exponentials eight at a time, the logits still batched prediction's bits.
  Attention's exponentials are the engine's own `expf` (glibc's algorithm in double without fused multiply-adds,
  which the WebAssembly engine already computes), so that its results are the same on every platform (glibc on
  processors with FMA differs on 2 of the 2^32 floats; other C libraries on more).
- **Prompts of sessions on many threads**: a prompt's attention is split between the threads in pieces of equal
  work (a causal prompt's last rows attend to the most keys), its normalizations, sums and products of layers and
  activations are split between them by rows (they ran on the calling thread alone: SiLU through libm's `expf` on
  2.5 million values a layer of Qwen2.5-0.5B), the block precisions' tiles take two groups of eight tokens a pass
  (each row's codes broadcast once for both), and attention with AVX2 sums the scores of four blocks of keys side
  by side and keeps a slice of the head's output in registers across them. The logits are the same bits. On the six
  performance cores of an i7-12650H a prompt of 512 tokens of Qwen2.5-0.5B-Instruct in Q4_0 takes 1,462 tokens a
  second instead of 808, in Q4_K_M 1,384, against llama.cpp's 866 and 476; generation is llama.cpp's speed or
  faster on every file measured (README, **Performance**).
- **The order of an epoch** in `spingalett_train()` on arrays is drawn from a seed the run draws once
  and the epoch's number, as data set readers draw their passes, instead of shuffling the previous
  epoch's order with the library's generator: any epoch's order is known without the epochs before it,
  and a callback that draws from the generator no longer changes the orders that follow. A run
  therefore orders its samples otherwise than 1.3 did from the same `spingalett_seed()`, and draws one
  value of the generator for its orders instead of one for each sample of each epoch. The speed is the
  same (the MLP of `Examples/Benchmark.c`, mini-batches of 64: 21,060 and 21,174 samples/s on one
  thread, 51,844 and 51,540 on eight, medians of six runs each on an i7-12650H).
- **`Examples/Pretrained.c`** opens GGUF files with `spingalett_model_open()`, makes a directory's network a
  model, and continues the prompt in a session (the prompt fed at once, then a token at a time).

### Fixed
- `ModelTool info` read the names of the kinds of layers after global pooling (transposed convolutions and the
  layers of 1.2) out of its table (1.2 on), and with `RunModel` the names of GELU, its tanh form and SiLU (SiLU
  printed as "fp16"); `RunModel` the names of the layers of 1.2 too.

## [1.3.0] - 2026-10-15

"Language models in production": language models served where 1.2 trained them, on the CPU and both GPU
backends: sessions with a key-value cache (batched generation, sliding windows), deployment models
predicting on the GPU (integer ones to the CPU's bits) and decoding with weights in INT8, INT4 and INT2,
tokenizers and Hugging Face and GGUF checkpoints in C and every binding; in training tied embeddings,
activation recomputation, gradient accumulation, padding, sliding windows and cross-attention; ONNX
import of transformers, the engine in WebAssembly for JavaScript, bindings for PHP. And the GPT of
`Examples/Benchmark.c` is ahead of PyTorch where 1.2 was behind it: in bfloat16 on CUDA ahead of
`torch.compile` with `max-autotune`, on the CPU 1.05 times as fast as PyTorch in training on one thread
and 1.64 times on eight (1.2: 0.78 and 0.49 times), and on Vulkan 3.1 times as fast as 1.2 in bfloat16
and 1.9 times in single precision (README, **Performance**). Programs of 1.2 build and run unchanged;
files of 1.2 load unchanged.

### Added
- **Activation recomputation** (`SpingalettTrainArgs.recompute`, checkpointing) on the CPU: the training
  workspace keeps the outputs of the layers several layers read (a transformer's residual stream, one a
  block), of batch normalizations and of the output layer; the others' outputs, gradients, values before
  their activation and dropout masks share memory between the segments between kept layers, and the backward
  pass computes a segment's outputs again before it (the last segment's still there), propagates its
  gradients and takes its layers' parameter gradients while they are there (layers tied to an embedding
  after every segment, in the table's order). The weights are those of training without it, bit for bit
  (transformer group: GPT-2-like with dropout, LLaMA-like, a tied head, an encoder-decoder, a Mistral-like
  model and a convolutional graph with batch normalization, on one thread and many, updated in the backward
  pass, clipped and accumulated). A GPT-like network of six blocks of 256 channels over 256 tokens trains in
  chunks of 16 samples with a peak of 141 MB instead of 216 MB, in 18.2 s instead of 14.2 s on one thread
  and 7.8 s instead of 5.6 s on eight (i7-12650H, medians of interleaved runs). The GPU keeps every layer.
  Python `train(..., recompute=True)` and the other bindings' options.
- **Cross-attention and encoder-decoders**: an attention layer of two inputs (`.inputs = {q, kv}`) takes the
  queries of layer `q` over the keys and values of the cells of layer `kv` (`2 kv_heads x head size`
  channels), with grouped queries and ignoring the padding of `kv`'s tokens, on the CPU, both GPU backends
  (the passes of `SPG_ATTN_EXT`, the keys and values a second input of the push constants; the units of
  other layers keep their PTX), the engine and deployment models. An embedding reads part of its input
  (`SpingalettLayerArgs.token_offset`, `token_count`), so that one input holds the source's tokens and the
  target's. `.slett` format 9 stores both (an attention entry of two inputs, an embedding's token offset at
  offset 104); `SpingalettNetworkLayer.token_offset`, `SpingalettLayerInfo.token_offset`; Python
  `add_embedding(..., token_offset=, token_count=)` and `add_attention(..., inputs=[q, kv])`, the C++
  builder's `cross_attention()` and the other bindings' `tokens` options. Sessions refuse such models (they
  run causal language models). The transformer group checks an encoder-decoder's gradients numerically and
  which keys the decoder's queries read; on the GPU heads of 32 to 256 match the CPU (predictions 6e-7, steps
  1.7e-5 at most) on Vulkan and CUDA.
- **Sliding window attention** (`SpingalettLayerArgs.window` on attention, Mistral's): each cell attends to
  the keys less than `window` cells away, with `.causal` the cell and the `window - 1` before it, without
  the `window - 1` on either side, on the CPU, both GPU backends, the engine and deployment models. The CPU
  and the GPU's kernels of single precision skip the blocks of keys outside every query's window (in
  bfloat16 on CUDA such layers run on those kernels, not `flash.cu`), and sessions on the GPU attend over
  the last `window` keys of their cache (`attn_cache`); `.slett` format 9 stores the window in
  the attention's momentum field. `SpingalettNetworkLayer.window`, `SpingalettLayerInfo.window`; Python
  `add_attention(..., window=w)`, the C++ builder's and the other bindings' options.
- **Sequences of different lengths in one batch** (`SpingalettLayerArgs.ignore_padding` on attention): the
  keys of padding, tokens of -1 in the network's input (traced through the layers that keep cells one for
  one to the embedding the keys' cells come from), left out of the softmax, a query without a key giving
  zeros, on the CPU, both GPU backends (the passes of such layers in `attn.comp` behind a specialization
  constant and in CUDA units of their own, `rattn_d*_o*x`, `attn_d256x`, so that the other layers' kernels
  stay as they were; in bfloat16 on CUDA such layers run on those units, not `flash.cu`), the engine and
  deployment models; `.slett` format 9 records it in the attention's flags. With the sparse cross-entropy's
  mean over a sample's real tokens, a padded batch's gradient is the mean of the sequences' own, each in a
  network of its length (transformer group: 8e-8 of 0.2 over right padding with non-causal attention and
  left padding with causal rotary attention; 0.19 apart without the option; the gpu group compares heads of
  32 to 256 with the CPU).
- **Tied input and output embeddings** (`SpingalettLayerArgs.tied`, `.tied = e` on a linear or dense layer):
  a language model's head computing with embedding `e`'s table, as GPT-2's and LLaMA's tied heads do, on
  the CPU, both GPU backends, the engine, deployment models and the runtime. The table is stored once and
  stepped by the sum of both layers' gradients, added in a fixed order (on the CPU the embedding's first,
  on the GPU the head's); the head keeps its biases. `.slett` format 9 records the tie (offset 104 of the
  head's entry, its weights and scales those of the embedding's section), which the writer picks only
  for networks with a tie; `SpingalettNetworkLayer.tied` and `SpingalettLayerInfo.tied` name it. In Python
  `add_linear(..., tied=e)`, and `gpt2_from_state_dict()` and `llama_from_state_dict()` tie the head when
  the checkpoint does (no `lm_head.weight`, or the embeddings' values) instead of copying the table; the
  C++ builder's `tied_linear()` and the other bindings' `tied` options.
- **Gradient accumulation** in `spingalett_train()` (`SpingalettTrainArgs.accumulation_steps`, on the CPU
  and both GPU backends): an optimizer step after every `accumulation_steps` mini-batches of
  `batch_size` samples (with a generator, calls of it), with the mean gradient of the step's samples
  and the workspace of one mini-batch. The step's mini-batches are its chunks, so its gradient is
  that of one batch of their samples computed in chunks of a mini-batch, bit for bit (a full batch
  of 5,000 samples in chunks of 2,048 against 3 x 2,048 accumulated, from arrays and from a
  generator), and differs from a batch computed in one chunk in the order of its sums only; dropout
  masks and augmentation are drawn by the sample's position in its step.
- **Learning-rate schedules by step** (`SpingalettTrainArgs.lr_per_step`): the schedule evaluated
  before every optimizer step, with the steps taken and the steps of the call in place of the epochs
  (the built-in schedules' `warmup_epochs` and `step_size` counting steps), reduce on plateau
  scaling each step's rate; two more built-in schedules, `spingalett_lr_linear_decay()` and
  `spingalett_lr_warmup_linear()` (a warm-up, then a straight line down to `min_lr`, as BERT and
  GPT-2 were trained). Python: `TrainConfig.accumulation_steps`, `lr_per_step`, `LinearDecay`,
  `WarmupLinear`; the other bindings' options take `accumulation_steps`.
- **Bindings for PHP** 8.0 and later (`Bindings/PHP`: the Composer package `pka-human/spingalett`,
  PHP's FFI extension): `spingalett.h`, the library's C declarations as FFI parses them (the `*_args`
  functions, no macros), loaded with the library that `SPINGALETT_LIBRARY`, `SPINGALETT_LIB_DIR` or
  the system's loader gives, or once for all requests (`Library::preload()` under `opcache.preload`,
  which PHP's default `ffi.enable=preload` needs outside the command line); classes for networks of
  every kind of layer, training on arrays or token files, prediction, evaluation, generation,
  parameters, files and deployment models, PHP arrays copied into the library's floats once a call.
  Their tests (`tests/run.php`, no PHPUnit needed) train XOR and a small LLaMA-like model whose
  generation they check, compare every declaration of `spingalett.h` with the C headers and every
  structure's layout with the C compiler's (`SpingalettLayout`); CI runs them on PHP 8.0 and 8.3
  against the installed library. (PHP 7.4's FFI passes no structure holding arrays or enumerations by
  value, which every `*_args` function takes.)
- **The inference engine in WebAssembly, and a JavaScript package around it** (`Bindings/JavaScript`,
  the npm package `spingalett`): `Src/Spingalett.Inference.c` built by Clang for wasm32 and `wasm-ld`
  without a C library (`build.sh`), its portable kernels vectorized with SIMD128, the mathematical
  functions it calls in `wasm/Spingalett.Libm.c` (`expf` glibc's algorithm, `tanhf` and `erff`
  correctly rounded: glibc 2.43's results over all 2^32 floats but two of its FMA variant's), a module
  of 62 KB that imports nothing. An ES module with TypeScript declarations for Node 18 and later and
  browsers with WebAssembly SIMD loads the module from a URL, a path, bytes, a `fetch()` response or a
  compiled module, and models from their bytes; describes their layers; runs samples and batches; and
  for language models gives the next token's logits and generates (greedy, temperature, top-k, top-p,
  a seed, stop tokens) with the tokens `spingalett_generate()` chooses. Its tests (`npm test`) hold it
  to the native engine on the models of earlier releases and on a GPT and a LLaMA in FP32 and INT8
  (`test/references`: integer models bit for bit, float ones within 1e-5 of each sample's largest
  output), and to the same code built natively (every model bit for bit). CI builds and tests it;
  releases attach the package (`npm pack`) and the module.
- **ONNX import of transformers** (`spingalett_import_onnx()`, `ModelTool import`,
  `Network.from_onnx()` and `Network.from_torch()` in Python), as both of PyTorch's exporters write
  them at opsets 14 to 23 (from an example of one sample, or with a dynamic batch), into the layers
  the library builds such models of:
  - inputs of token ids `[N, T]` (an input layer of `T` indices) and sequences `[N, T, C]` (a layer of
    `1 x T` cells); `Gather` from a constant table as an embedding, a learned position table added
    to it as its positions (constant tables of token types added too);
  - `MatMul` and `Gemm` by constant weights on the tokens (also on tokens as rows `[N T, C]`, as
    Hugging Face's `Conv1D` writes them) as linear layers, a constant added as their bias;
    `LayerNormalization`, `RMSNormalization` and their written-out forms over each token;
  - element-wise expressions recognised as the activations they compute, by their values at points
    where activations differ: GELU (`Gelu`, its erf and tanh forms written out), SiLU
    (`x * Sigmoid(x)`), ReLU, sigmoid, tanh; `Mul` of two layers (SwiGLU's and GeGLU's gates) as a
    multiplication layer; the same of vectors and maps (an MLP's GELU and layer normalization written
    out, `Gelu` and `RMSNormalization` of vectors);
  - attention written out folded into one attention layer: queries, keys and values from one product
    or three, split into heads (`Reshape`, `Transpose`, `Split`, `Slice`, `Gather`, `Squeeze`), the
    scores scaled (`Mul` or `Div`, of the scores or of the queries and keys), masked causally (`Where`
    or an added triangle of `-inf` or of a large negative value), `Softmax` over the keys, the
    product with the values; keys and values repeated for grouped-query and multi-query attention
    (`Expand`, `Tile`); rotary embeddings (`rotate_half` with tables of cosines and sines, their base
    read from the tables); the queries', keys' and values' products merged into one linear layer,
    their biases and any scale other than `1/sqrt(head size)` folded into its weights;
  - opset 23's `Attention` and `RMSNormalization`, and the fused operators of ONNX Runtime's
    transformer optimizer: `com.microsoft`'s `Attention`, `MultiHeadAttention`,
    `SkipLayerNormalization`, `SkipSimplifiedLayerNormalization`, `SimplifiedLayerNormalization`,
    `Gelu`, `FastGelu` and `BiasGelu`;
  - nodes whose inputs are all constants computed on import (`Shape`, `Gather`, `Slice`, `Concat`,
    `Range`, `Trilu`, `Where`, arithmetic and the others of shape computations); `ReduceMean` over a
    sequence's tokens as global average pooling;
  - language models (token ids in, logits a token out) take the sparse cross-entropy loss;
  - what no layer computes is named in the error: a softmax over each token's channels, a product
    of two computed tensors outside an attention, masks and caches given as inputs, rotary
    embeddings over part of a head or of interleaved pairs.
  GPT-2-, LLaMA-, Gemma- and BERT-like models and `nn.TransformerEncoderLayer` agree with PyTorch to
  the rounding of single precision (within 3e-6 on the tests' models: `onnx_gpt*`, `onnx_llama*`,
  `onnx_encoder*`, `onnx_ort_fused` in `Tests/Data`); a model of GPT-2 small's size (124M parameters,
  650 MB) imports in 0.57 s on an i7-12650H.
- **Inference with a key-value cache on the CPU:** sessions of deployment models run a causal language
  model (its first layer an embedding of the tokens, its other layers treating each position alone but
  for causal attention) a token at a time. Each attention layer keeps the keys (rotated, with rotary
  embeddings) and values of every token a sequence has been fed, so that a new token costs one row
  through every layer and its attention over those instead of a pass over the window.
  `spingalett_session_new()` (`SpingalettSessionOptions`: `max_length`, `sequences`) opens one, or
  refuses a model with the reason; `spingalett_session_feed()` (`SpingalettSessionFeed`) appends tokens
  to several sequences in one pass, a prompt at once, the last layer computed on the rows whose logits
  are asked for only; `spingalett_session_rewind()`, `spingalett_session_copy()`,
  `spingalett_session_length()`, `spingalett_session_vocabulary()`, `spingalett_session_free()`. The
  logits are those `spingalett_model_predict()` gives each position of the whole window, bit for bit,
  in every precision and however the tokens come (alone, as a prompt, with other sequences): the float
  linear layers run on their filters packed once per model by new kernels for few rows that sum in the
  order of the batched forward kernel, the integer ones sum in integers, attention attends as the
  engine does. Several sessions may run on one model from several threads. Part of the runtime. Tests:
  `model` (GPT-2-like and LLaMA-like models, narrow and wide, every precision, one thread and many,
  against batched prediction; refused models and arguments; four threads), `runtime.library` (the
  runtime's sessions give the library's bits).
- **Batched generation:** `spingalett_session_generate()` continues several sequences of a session
  together, the steps' rows computed at once, each sequence drawing as `spingalett_generate()` draws
  (temperature, top-k, top-p, stop tokens; sequence b with seed + b): the same tokens as alone. Tests:
  `gen`.
- New functions `spingalett_session_new()`, `spingalett_session_free()`, `spingalett_session_feed()`,
  `spingalett_session_rewind()`, `spingalett_session_copy()`, `spingalett_session_length()`,
  `spingalett_session_vocabulary()` and `spingalett_session_generate_args()`
  (`spingalett_session_generate()`), in the symbol version `SPINGALETT_1.3` of the library and the
  runtime; new structs `SpingalettSessionOptions`, `SpingalettSessionFeed` and
  `SpingalettSessionGenerateArgs`. In Python, `Model.session()` (`Session`: `feed()`, `feed_many()`,
  `rewind()`, `copy()`, `length()`, `generate()`).
- **Integer deployment models on the GPU, the CPU's bits:** in the full library's GPU modes
  `spingalett_model_predict()` of an INT8, INT4 or INT2 model that owns its image runs on the GPU
  (CUDA and Vulkan) and gives the bits of `spingalett_model_run()`, for every model whose layers the GPU
  computes as the CPU does: each integer layer's inputs quantized per sample as
  `spingalett_quantize_activations()` quantizes them (`iquant.comp`: the largest magnitude from the
  values' bits, the divisions rounded as IEEE's, on Vulkan by long division of the significands; a sample
  in slices of 4,096 values over several workgroups), the codes (INT4 and INT2 unpacked) multiplying them
  on the matrix kernels as floats or bfloat16, which hold them exactly, in sums of at most 1,040 products
  of INT8 codes (16,512 of INT4, 66,048 of INT2) that stay below 2^24, longer sums in slices added as
  integers; depthwise and transposed convolutions window by window (`ipass.comp`); the outputs
  `spingalett_int_output()`'s, pooling, global pooling, additions, products and concatenations of layers,
  batch normalization and upsampling in the CPU's order of operations, the multiplies and additions
  rounded alone (`precise`, CUDA's `mul.rn` and `add.rn`); the activations none, ReLU and leaky ReLU inside
  the network, the last layer's others (softmax, sigmoid, ...) on the CPU. Models with other layers or
  activations inside (GELU, SiLU, sigmoid, tanh, normalizations of layers, attention, embeddings) stay on
  the CPU, decided once per model and backend. Chunks of up to 4,096 samples whose activations keep to
  32 MB, two of them in flight (the next one's inputs copied in while one runs). On an RTX 4050 Laptop GPU
  against the 16 threads of an Intel Core i7-12650H (samples/s, INT8, medians of three interleaved
  rounds): the benchmark's MLP over 20,000 samples 658,000 on the CPU, 2,490,000 on CUDA, 2,010,000 on
  Vulkan; the CNN of MNIST 86,400, 567,000, 367,000; ResNet-20 10,300, 69,900, 56,900; the MobileNet-style
  network 15,400, 89,400, 76,400; the U-Net 3,130, 15,200, 14,100 (INT4 alike: the MLP 647,000, 2,500,000,
  2,010,000). Tests (gpu and cuda groups): an MLP, a chain of convolutions and a graph of every such layer
  in INT8, INT4 and INT2 the bits of `spingalett_model_run()` and of the CPU's batched prediction (9,000
  samples in chunks, products in slices, inputs quantized in slices), a model with GELU inside on the CPU.
- **Sessions of integer language models on the GPU multiply their codes** (weight-only): the decoder of a
  session of an INT8, INT4 or INT2 model keeps its linear layers' codes on the device as the image stores
  them (each row padded to 16 bytes), with their scales, and multiplies them by the activations in the
  GPU's precision, widening the codes to floats in registers as it reads them (`wqmv.comp`: 32 rows of
  weights a workgroup, four (two) a group of 32 invocations, each reading a value of the activations once
  for them; passes of up to eight rows of activations, each reading the codes once; a row's sums in the
  order of k, then the invocations' by a tree in shared memory, or a warp's shuffles: the same bits on
  every run, and a row's whatever rows run with it), where they multiplied the weights widened to floats
  or bfloat16 on the matrix kernels. The logits are those of the model's weights as floats to rounding
  (4e-7 of the CPU's session of the network the image makes, 5e-3 in bfloat16), near the CPU's integer
  session's (which quantizes each token's activations: 9e-3 in the tests), not its bits. Greedy decoding
  after a prompt of 16, tokens/s on an RTX 4050 Laptop GPU, medians of three interleaved rounds, before
  and after: the GPT of `Examples/Benchmark.c` (6 blocks of 384) in INT8 on CUDA 1,025 -> 3,386 in single
  precision, 2,223 -> 3,406 in bfloat16, on Vulkan 804 -> 2,248 and 1,270 -> 2,304 (INT4 3,214, 3,236, 2,194,
  2,180); a model of GPT-2 small's size (12 blocks of 768, a vocabulary of 50,257, a window of 512) in INT8
  on CUDA 186 -> 680 and 368 -> 684, on Vulkan 142 -> 443 and 256 -> 431 (INT4 838, 854, 553, 553; the same
  model in BF16 186, 368, 144, 253), four sequences together in bfloat16 1,250 -> 1,953 on CUDA (INT4
  2,053) and 932 -> 1,365 on Vulkan (INT4 1,326). Tests (gpu and cuda groups): GPT- and LLaMA-like models
  and one of 66 channels in INT8, INT4 and INT2 against their weights as floats and the CPU's integer
  session, a prompt the bits of its tokens one at a time and of two sequences fed together.
- **Tokenizers** of language models in C: `spingalett_tokenizer_load()` reads a Hugging Face
  `tokenizer.json` (or a model directory, whose `tokenizer_config.json` names BOS and EOS),
  `spingalett_tokenize()` encodes UTF-8 text and `spingalett_detokenize()` decodes ids, with
  `spingalett_tokenizer_info()`, `spingalett_tokenizer_token()` and `spingalett_tokenizer_id()`
  (`SpingalettTokenizer`, `SpingalettTokenizerInfo`, `SPINGALETT_NO_TOKEN`). Byte-level BPE (GPT-2,
  SmolLM), tiktoken's encodings (LLaMA 3's cl100k, Qwen2's with single digits and NFC, o200k) and
  SentencePiece's BPE with byte fallback (LLaMA 2, Mistral, Phi-3), as the `tokenizers` library runs
  them: added tokens with their strip rules, special ones found only when asked (`.special`), the
  normalizers, pre-tokenizers, decoders and templates those files use, merges by rank through a heap
  (n log n in a piece's length; SentencePiece's texts merged word by word where the vocabulary allows
  it). The pre-tokenizers' regular expressions are matched by hand with Oniguruma's semantics, over
  Unicode tables written by `Src/make_unicode.py`. On the tokenizers of GPT-2, LLaMA 3.2, Qwen2.5,
  SmolLM2, TinyLlama, Mistral 7B and Phi-3 the ids and texts equal those of `tokenizers` 0.23, and
  encoding runs 3.6 to 8.3 times as fast (0.9 MB of prose and code on an i7-12650H, one thread). The
  `tokenizer` test group checks six small tokenizers of these families
  (`Tests/make_tokenizer_tests.py`) against the library's ids.
- **Hugging Face checkpoints in C**: `spingalett_load_pretrained()` (`SpingalettPretrainedArgs`: `path`,
  `context`) makes a network of a model directory, `config.json` and safetensors weights (one file, or
  the shards of `model.safetensors.index.json`; F32, F16, BF16), of the model types `llama`, `mistral`,
  `qwen2` and `gpt2`: the networks of the Python bindings' `llama_from_state_dict()` and
  `gpt2_from_state_dict()`, which give the same logits bit for bit, with Qwen2's biases, explicit
  `head_dim`, tied embeddings, GPT-2's Conv1D weights. The arrays are reserved at their final sizes and
  each tensor is decoded once from the mapped file into its layer (SmolLM2-135M loads in 0.17 s on an
  i7-12650H). Mistral's sliding window is the attention layers' `window`, a head missing from the
  checkpoint is tied to the embedding; scaled rotary embeddings are refused. On SmolLM2-135M and Qwen2.5-0.5B-Instruct the logits agree with
  `transformers` within 2.4e-4 and 4e-4 and the greedy continuations are the same. The `pretrained` test
  group compares tiny LLaMA,
  Qwen2, Mistral and GPT-2 checkpoints (`Tests/make_pretrained_tests.py`) with PyTorch.
- **GGUF files**, llama.cpp's (versions 2 and 3), in `spingalett_load_pretrained()` and
  `spingalett_tokenizer_load()` / `_from_memory()`: the network of `general.architecture` llama
  (LLaMA, Mistral), qwen2 or gpt2 with the hyper-parameters of the metadata, its weights dequantized as
  llama.cpp's `dequantize_row_*()` computes them (F32, F16, BF16, Q8_0, Q4_0, Q4_1, Q5_0, Q5_1, Q2_K,
  Q3_K, Q4_K, Q5_K, Q6_K, IQ4_NL, IQ4_XS; in IEEE order, the same weights on every platform), LLaMA's
  queries and keys taken back from llama.cpp's order of their rows; the vocabulary of the `gpt2` model
  as a byte-level BPE by the pre-tokenizer the file names, that of the `llama` model as SentencePiece's
  BPE of its scores. On quantized SmolLM2-135M, TinyStories and Qwen2.5-0.5B-Instruct files the weights
  equal the `gguf` package's dequantization bit for bit, and their vocabularies give the ids of the
  `tokenizer.json` files they came from; the `pretrained` group checks GGUF files of the weights of
  `hf_llama`, `hf_qwen2` and `hf_gpt2` (their directories' logits bit for bit), a model of fifteen
  types (each matrix against a reference written
  with llama.cpp's formulas and checked against `gguf`'s), a SentencePiece vocabulary against the
  `sentencepiece` library's ids, and damaged files.
- Tokenizers and pretrained models in every binding: Python's `sg.Tokenizer` (`encode()`, `decode()`,
  `token()`, `id()`, `bos`, `eos`, ...) and `Network.from_pretrained(path, context)`, which give the
  logits of the 1.2 loader `llama_from_state_dict()` bit for bit without PyTorch; C++'s
  `spingalett::Tokenizer` and `Network::from_pretrained()`; Rust's, C#'s, Go's and Java's `Tokenizer`
  and pretrained loading, each tested on the `tokenizer` group's expectations and on `hf_llama` in both of
  its forms, their structures compared with the C layout. The functions of 1.3 are bound optionally in
  Python and Java, so that the bindings still load a library of 1.2.
- `Examples/Pretrained.c`: a Hugging Face or GGUF language model continuing a prompt, printing each
  token as it comes.
- An internal JSON reader (`Src/Spingalett.Json.c`) for the files of language models.

### Changed
- **The GPT in bfloat16 on CUDA ahead of `torch.compile`**: products of dense layers whose operands are kept
  as bfloat16 on tiles of 128 x 128 outputs over four warps of 64 x 64 (`gemm_pipe.cu`: three steps of k in
  shared memory copied two steps ahead, chunks of 16 bytes XOR-swizzled within their rows, fragments a step
  ahead, the epilogue from the registers, eight consecutive outputs a thread stored in 16 bytes; each
  epilogue a loop of its own), adding in the order of k over the same steps as the other tiles, so that every
  tile still gives the same bits; flash attention's forward pass with two tiles of queries a warp, its rows
  loaded and stored 16 bytes at a time; GELU's erf through the approximate reciprocal; training and inference
  chunks whose largest layer keeps to the GPU's cache (2^24 values: the GPT in chunks of 16 sequences in
  training, 32 in inference). On an RTX 4050 Laptop GPU the GPT of `Examples/Benchmark.c` in bfloat16, its
  data in GPU memory, trains 197,600 tokens a second and infers 684,800, against `torch.compile` with
  `max-autotune`'s 193,000 and 655,000 (medians of three runs of the GPT alone, each from a GPU at 60 C;
  1.2.0, measured the same way: 180,900 and 583,800).
- **Transformers' training on the CPU** (the roadmap's step 2 of 1.3): element-wise passes over a chunk in
  pieces of 8,192 values rather than a sample at a time, GELU, its tanh form and SiLU in vectors while training
  (erf as the GPU computes it; inference keeps libm's), the indirect products' weights packed on the threads,
  network chunks of sequences grown to products of 2,048 rows, attention's softmax exponentials in vectors,
  linear layers' weight gradients on the dense products in slots of 2,048 tokens summed in their order
  (`SpingalettGemmHooks.tall`), and large tiles of the indirect products handed out one at a time; every
  change keeps the bits the same on any number of threads. On an Intel Core i7-12650H the GPT of
  `Examples/Benchmark.c` trains 1,616 tokens a second on one thread (1.2.0: 1,150) and 7,065 on eight (1.2.0:
  2,190), against PyTorch 2.14.1's 1,534 and 4,314, and infers 3,893 and 19,043 (1.2.0: 3,655 and 9,656),
  medians of three interleaved runs.
- **Attention in single precision on Vulkan in register tiles** (`rattn.comp`, as CUDA's `rattn.cu` computes it):
  a workgroup of 64 queries (keys in the keys' gradient pass), each invocation four of them and a share of a tile
  of keys, the scores and sums in registers, a row's maximum and sum combined over its eight lanes by subgroup
  shuffles in a fixed pattern (the same bits on every run), the next tile loaded while one is used; for heads of
  up to 128 values where the device has subgroups of 8 lanes or more and up to 48 KB of shared memory (the
  others, and layers with padding, windows or cross-attention, run on `attn.comp` as before). On an RTX 4050
  Laptop GPU the GPT of `Examples/Benchmark.c` in single precision on Vulkan trains 70,967 tokens a second
  instead of 40,085 and infers 226,062 instead of 143,567 (CUDA: 74,821 and 235,811), medians of three runs
  from a GPU at 60 C.
- **Flash attention on Vulkan**: networks in bfloat16 run attention's forward pass and its gradients on the
  matrix units (`flash.comp`: cooperative matrices of bfloat16, `VK_KHR_cooperative_matrix`, as CUDA's
  `flash.cu` does with `mma.sync`) for heads of 32, 64 and 128, causal or not, rotary, grouped: a subgroup
  takes 16 queries (keys for the keys' and values' gradients), the scores pass through shared memory for the
  softmax, whose row maxima and sums each row's invocations combine in a fixed pattern (the same bits on
  every run). Devices without bfloat16 cooperative matrices or the room in shared memory, and builds whose
  glslc lacks `GL_EXT_bfloat16`, run `attn.comp` as before. On an RTX 4050 Laptop GPU the passes of
  `Examples/Benchmark.c`'s GPT (chunks of 16 sequences of 256 tokens, 6 heads of 64) take 0.11, 0.13 and
  0.18 ms instead of 2.2, 3.9 and 3.9 ms (CUDA's `flash.cu`: 0.11, 0.12, 0.18), and the GPT in bfloat16 on
  Vulkan trains 179,000 tokens a second instead of 53,800 (3.3 times) and infers 572,000 instead of 225,000
  (2.5 times). `SpingalettGpuTests` compares the two ways on the same inputs (`bench attn` times them).
- Training on Vulkan applies GELU and SiLU in the products' epilogues, which store the values before the
  activation for its derivative too (`SPG_GEMM_PRE`, as CUDA's products do, in pipelines of their own so
  that the other products' code stays as it was), rather than the product writing those and an
  element-wise pass the activation: the GPT of `Examples/Benchmark.c` trains 1.01 times as fast on an RTX
  4050 Laptop GPU, in bfloat16 and in single precision.
- `spingalett_generate()` on the CPU runs a causal language model as its deployment model of single
  precision in a session (kept in the network until its parameters or layers change): the prompt fed
  at once, then each token once; once the tokens fill the window, the window of the last tokens is fed
  again from its first position, as the prediction over the window does, so the tokens are those of
  1.2 (the logits equal the network's prediction to rounding, within 1e-6 relative in the tests). On an
  Intel Core i7-12650H the CharGPT model (256 wide, 4 blocks, a window of 128) generates 3230 tokens a
  second on one thread, 44 times 1.2's 72.7, and 3568 on six threads (121 in 1.2); a call generating
  one token takes 1.8 ms instead of 11.7. Other networks and the GPU modes predict over the window as
  before.

### Fixed
- `spingalett_device_data_new()` in the CPU's compute modes made sets only in Vulkan's memory, so that a
  library built without Vulkan, or a machine without a Vulkan device, made none; it now takes CUDA's
  there. The `cuda` group's tests asked the compute mode's GPU, not CUDA, whether it multiplies in
  bfloat16, and when told no left that precision set: in a build without Vulkan the later tests ran in
  bfloat16 and failed (20 checks).
- Integer deployment models of language models (INT8, INT4, INT2; in the engine, batched and in the
  runtime) quantized the input of each linear layer with one scale for the whole sequence, so that a
  later token changed the logits of the earlier ones of a causal model. In a model whose first layer is
  an embedding, integer convolutions of 1 x 1 windows, stride 1, no padding and one group now quantize
  each token's channels with a scale of their own (`docs/ModelFormat.md`); their results differ from
  1.2's within the quantization's error. Tests: `transformer` (later tokens leave earlier logits
  unchanged, bit for bit, batched and in single runs).

## [1.2.0] - 2026-10-11

"Transformers": the networks of today's language models in every part of the library at once,
training on the CPU and both GPU backends, the model format, the engine, deployment models and the
runtime, with what language models need around them (files of tokens, generation, GPT-2 and LLaMA
checkpoints). On CUDA, attention runs on the tensor cores in bfloat16 and in register tiles in
single precision, GELU and SiLU in the products' epilogues; the GPT of `Examples/Benchmark.c` trains
1.97 times as fast as in the first transformer build of this release in single precision and 1.28
times as fast in bfloat16: as fast as PyTorch with `torch.compile` in single precision and at 0.91
times its speed in bfloat16, while every other workload runs ahead of PyTorch at its fastest
(README, **Performance**). And more languages and package managers: bindings for Rust, .NET, Go and
Java, and Debian, RPM, AUR, Homebrew, Scoop, Nix, container and conda packages. Programs of 1.1
build and run unchanged; files of 1.1 load unchanged.

### Added
- **The layers of transformers**, on the CPU, CUDA and Vulkan, in `.slett` format 8, the engine,
  deployment models of every precision and the runtime; a sequence is a layer of `1 x n` cells, the
  input layer holding token ids:
  - `spingalett_embedding()` (`SPINGALETT_LAYER_EMBEDDING`): a table of `vocabulary` rows of
    `neurons_amount` values, with `positions` a learned vector per position (its biases); its
    gradient summed per token in the batch's order (on the GPU through a stable radix sort of the
    tokens, without atomics);
  - `spingalett_attention()` (`SPINGALETT_LAYER_ATTENTION`): multi-head attention over the queries,
    keys and values a linear layer packs before it, `heads` query heads and `kv_heads` key and value
    heads (grouped-query and multi-query attention), a causal mask (`causal`), rotary position
    embeddings (`rope_theta`, the `rotate_half` form of GPT-NeoX and Hugging Face's LLaMA); the
    scores never stored (the backward pass recomputes them from each query's log-sum-exp); on the CPU
    as blocked matrix products a thread per sample and group of heads, on the GPU in tiles;
  - `spingalett_rms_norm()` (`SPINGALETT_LAYER_RMS_NORM`), `spingalett_multiply_layers()`
    (`SPINGALETT_LAYER_MULTIPLY`, the element-wise product of layers: SwiGLU's gate) and
    `spingalett_linear()` (a 1 x 1 convolution: a linear map of each position);
  - the activations `SPINGALETT_ACT_GELU`, `SPINGALETT_ACT_GELU_TANH` and `SPINGALETT_ACT_SILU`, whose
    derivatives take the values before the activation (training keeps those of such layers);
  - the loss `SPINGALETT_LOSS_SPARSE_CROSS_ENTROPY`: a softmax over each cell's channels against one
    class index a cell (an index outside the classes leaves the cell out), label smoothing applied
    within the loss; `spingalett_target_size()` gives the targets a sample.
  The layers' new arguments (`vocabulary`, `heads`, `kv_heads`, `rope_theta`, `causal`, `positions`)
  take reserved words of `SpingalettLayerArgs`, `SpingalettNetworkLayer` and `SpingalettLayerInfo`.
  Tests: the group `transformer` (gradients against finite differences, determinism, a model that
  learns, every precision of deployment models, validation), and the GPU groups against the CPU.
- **Language models:** `spingalett_dataset_open_tokens()` reads the windows of a file of token ids
  (nanoGPT's `.bin` of uint16, llm.c's with its header, uint16 or uint32) as a data set reader:
  each sample a context of tokens and the tokens after them, a pass in an order of its seed, the file
  mapped; `spingalett_generate()` continues a prompt a token at a time (greedy, temperature, top-k,
  top-p, stop tokens, a seed). `Examples/CharGPT.c` trains a character-level GPT on any text file and
  samples it. In Python, `Network.train_from_tokens()`, `Network.generate()`,
  `gpt2_from_state_dict()` (GPT-2 from Hugging Face's or nanoGPT's state dict) and
  `llama_from_state_dict()` (LLaMA, Mistral, Qwen2 in Hugging Face's names), compared with references
  written in PyTorch.
- New functions `spingalett_target_size()`, `spingalett_dataset_open_tokens()` and
  `spingalett_generate_args()` (`spingalett_generate()`), in the symbol version `SPINGALETT_1.2`; new
  types `SpingalettTokenReaderOptions` and `SpingalettGenerateArgs`.
- **CUDA:** flash attention on the tensor cores for networks in bfloat16 (`mma.sync` in bfloat16 with
  single-precision sums, heads of 32, 64 and 128, a unit per head size and pass, tiles of activations
  loaded sixteen bytes a thread and the next tile into registers during the current one's products);
  in single precision attention in register tiles (`rattn.cu`: a thread four rows and its share of the
  other side's tile, its scores and sums in registers): 7.3 ms to 0.72 ms for the GPT's forward pass,
  8.5 to 0.90 ms and 8.9 to 1.19 ms for its two backward passes on the RTX 4050.
- **Bindings** for Rust (`Bindings/Rust`: the crates `spingalett-sys` and `spingalett`), .NET 8 and
  later (`Bindings/CSharp`: the package `Spingalett`, `LibraryImport`), Go (`Bindings/Go`: cgo,
  pkg-config) and Java 22 and later (`Bindings/Java`: the foreign function and memory API, Maven):
  networks of every kind of layer, training on arrays or token files, prediction, evaluation,
  generation, parameters, files and deployment models; their tests compare every structure they lay
  out with the C compiler's layout (`SpingalettLayout`), and CI runs them against the installed
  library. Releases attach the `.nupkg` and the `.jar`.
- **Packages:** Debian and RPM packages of the library, its runtime and tools (CPack, built and
  installed by the release workflow), PKGBUILDs for Arch Linux (`packaging/aur`: `spingalett`,
  `spingalett-git`), a Homebrew formula and a Scoop manifest served from this repository
  (`Formula/`, `bucket/`), a Nix flake, a container image (`ghcr.io/pka-human/spingalett`, both GPU
  backends, the tools, headers and Python bindings) and a recipe for conda-forge.
- C++ (`Spingalett.hpp`): `Builder::linear()`, `embedding()`, `attention()`, `rms_norm()`,
  `multiply_layers()`; `Network::target_size()`, `train_tokens()`, `generate()` with `Sampling`; the
  new activations, the sparse cross-entropy and `Compute::Cuda`.
- `Examples/Benchmark.c`: a GPT (nanoGPT's character-level configuration, 10.9M parameters) in tokens
  per second; workloads and the GPU's precisions chosen by name (`gpt`, `fp32`, `bf16`, ...).
  `benchmark_pytorch.py --max` runs PyTorch at its fastest (`torch.compile` with `max-autotune` and
  CUDA graphs, `cudnn.benchmark`, channels-last tensors, fused optimizers), `--gpt` the GPT alone.

### Changed
- CUDA, bfloat16: the weights' bfloat16 copy that the products read is made where products take 256
  rows of a layer (samples times its cells), not only for chunks of 256 samples, so that sequences
  and images use it: ResNet-20 trains 1.04 times as fast and infers 1.14 times as fast, the U-Net 1.03
  and 1.10 times, the MNIST CNN infers 1.2 times as fast, the MobileNet-style network 1.03 and 1.07 to
  1.11 times, the GPT 1.11 and 1.14 times.
- CUDA: the products' units built with `GENERAL` (dense layers' modes, 74 units) hold every
  activation and derivative in their epilogue, and store the values before the activation where
  training keeps them (`SPG_GEMM_PRE`), in place of a pass over the product's outputs; the other
  units are unchanged.
- CUDA: a product's blocks take its tiles in bands of eight tiles of rows over every tile of columns,
  so that a band's rows of A stay in L2 (the same bits): on the matrix units always (the MLP infers
  1.1 times as fast in bfloat16), in single precision where A and C do not fit in L2 together, the
  backend launching the blocks so (the GPT infers 1.1 times as fast; with the bands computed in the
  kernel, its largest tiles spilled registers).
- GPU: 1 x 1 convolutions of stride 1 (transformers' linear layers, pointwise convolutions) read their
  maps as matrices in all three products: the MobileNet-style network infers 6% faster on CUDA.

### Fixed
- Training on CUDA logged its compute mode as `VULKAN (the GPU's name)`; it logs `CUDA (...)`.
- Built by Clang, the host's label smoothing made a fused multiply-add of `keep * t + share`, while
  the GPU's smoothing of data sets in its memory rounds the product first: training on device data
  sets differed in the last bits from training on host arrays. The product is rounded first on both.

## [1.1.0] - 2026-10-10

"CUDA": a GPU backend for NVIDIA GPUs on the library's own kernels (`SPINGALETT_COMPUTE_CUDA`, which
1.0 reserved), compiled to PTX by Clang and carried in the library: neither the CUDA toolkit nor
cuBLAS or cuDNN is needed to build or to run. On an RTX 4050 Laptop GPU it trains every workload of
`Examples/Benchmark.c` 1.2 to 2.1 times as fast as PyTorch 2.14 with cuDNN in single precision
(against PyTorch's TF32 convolutions) and 1.25 to 2.1 times in bfloat16 (mini-batches of the MLP 4.8
and 7.7 times), infers 1.3 to 5.8 and 1.5 to 4.2 times as fast, and is ahead of the Vulkan backend
everywhere but in single-precision U-Net inference, where they are even (README, **Performance**).
And "Runtime": programs that only run trained models link `libspingalett-runtime`, the library's
deployment models without its networks, training, data sets, importers or GPU backends: 510 KB
where the library takes 2.0 MB without the CUDA backend's kernels (x86-64 release builds), and the
same bits. Programs of
1.0 run unchanged; the CPU's kernels did not change.

### Added
- **The CUDA backend:** `SPINGALETT_COMPUTE_CUDA` (`ComputeMode.CUDA` in Python) runs
  `spingalett_train()` (full batch and mini-batches), the trainer's passes, `spingalett_predict()`
  and `spingalett_evaluate()` on NVIDIA GPUs of compute capability 8.0 or later (Ampere and newer)
  with a driver of CUDA 11.0 or later, which the library opens at run time. It runs the Vulkan
  backend's executor (every kind of layer, activation, loss and optimizer; data sets in GPU memory;
  bfloat16 on the tensor cores with `spingalett_set_gpu_precision()`; the network's copy kept
  between calls), with the same determinism: every sum in a fixed order, the same bits on every run.
  `spingalett_cuda_device()` (a new function, in the symbol version `SPINGALETT_1.1`; `cuda_device()`
  in Python) names the GPU, or returns `NULL` without one; `SPINGALETT_CUDA_DEVICE=n` picks one.
  Its kernels (`Src/Gpu/Cuda`): matrix products in single precision as tiles chosen by timing (two
  buffers of shared memory filled by asynchronous copies, operand modes and vector loads as
  instances, the convolutions' windows read through their tap tables, epilogues inline); products on
  the tensor cores (`mma.sync` in bfloat16 with `ldmatrix`, results through shared memory to
  coalesced stores); a direct kernel for products of few k over a convolution's windows, which a
  network's first convolution has; depthwise convolutions whose windows, strides, modes and
  bfloat16 maps are instances, applying the normalization before them as they read; normalization,
  pooling, additions and optimizer steps four values a thread. Each is a unit of PTX, compressed in
  the library and inflated when first used, when the driver compiles it and keeps it in its cache;
  commands run as CUDA graphs, a chunk's inputs copied on a stream of their own. CMake builds it
  where Clang compiles CUDA (`SPINGALETT_CUDA`), or from another build's PTX (`SPINGALETT_PTX_DIR`).
  The Linux and Windows packages and the wheels have it; macOS builds do not. Its PTX takes 10.6 MB
  of the library's 12.6 MB on x86-64. Tests: `SpingalettGpuTests` and the `cuda` group, which
  compare the kernels with references and the tiles bit for bit, and the GPU with the CPU.
- **The runtime:** `Spingalett.Runtime.h` declares what it has: `spingalett_model_load()`,
  `spingalett_model_from_memory()`, `spingalett_model_predict()`, `spingalett_model_evaluate()`,
  `spingalett_model_free()`, the engine of `Spingalett.Inference.h`, the version, errors, logging,
  the compute mode and threads. These are the library's functions, which `Spingalett.h` now declares
  through that header, built from the library's sources with its flags and kernels (chosen at run
  time in release builds), so that a program written for the runtime runs with either library and
  gets the same results; `runtime.library` compares them bit for bit on 121 models (every kind of
  layer, every precision, one thread and four, single runs, evaluation). It reads `.slett` files of
  versions 3 to 7 (what 0.5 and later write); files of versions 1 and 2 hold networks, and it refuses
  them with `SPINGALETT_ERR_FORMAT_VERSION`, naming `ModelTool convert`. CMake builds it next to the
  library (`SPINGALETT_RUNTIME`, on by default) or alone (`SPINGALETT_RUNTIME_ONLY`, which needs no
  `glslc` or Vulkan headers); it installs as `Spingalett::runtime` and pkg-config's
  `spingalett-runtime`, with the soname `libspingalett-runtime.so.1` and the library's symbol
  versions. Every release archive carries it, and `spingalett-runtime-*` archives hold it alone. The
  Conan recipe has a `runtime_only` option, and both packages give `Spingalett::runtime`.
- `RunModel` (`Examples/Runtime/RunModel.c`, installed with the runtime): a model's layers and the
  time of one sample and of batches, or the outputs of a file of raw float samples.

### Changed
- GPU inference (both backends): a batch normalization that a dense or convolution layer without
  activation feeds alone runs in that layer's product, in its epilogue (`SPG_EPI_SCALE_ACT`; for a
  depthwise convolution in its kernel's), with the layer's biases folded into its coefficients, and
  the layer's own outputs are never written; the normalizations' coefficients are computed once
  until the parameters change rather than in every chunk. The convolutional network with batch
  normalization infers 1.3 times as fast in bfloat16 on Vulkan, the MobileNet-style network 1.2 to 1.3
  times, ResNet-20 and the U-Net 1.1 times.
- GPU: three chunks in flight instead of two, so that the host fills one while the device works on
  the two before it: the MLP of `Examples/Benchmark.c` infers 1.2 times as fast from host arrays.
- GPU: inputs kept as bfloat16 are rounded from the caller's floats straight into the memory they
  are copied from, on the OpenMP threads, where they were rounded on one thread at submission
  (backends that cannot write into device memory, such as CUDA).
- The examples, DigitPad and the code in the README use the short names of `Spingalett.Short.h`
  (`NeuralNetwork`, `layer()`, `train()`, `ACT_RELU`), which read more easily; the descriptions and
  `docs/Reference.md` keep the prefixed names that the library declares and exports. The README says
  why the library needs its prefix.
- `Spingalett.Short.h` takes the short names of the log levels, the compute modes and
  `EvalMetrics` from `Spingalett.Runtime.h`, which gives them when `SPINGALETT_SHORT_NAMES` is
  defined, as `Spingalett.Inference.h` gives the engine's.

## [1.0.0] - 2026-10-09

"Stability": the API, the ABI and the formats of 0.14, which every 1.x release keeps (semantic
versioning, stated in the README's **Compatibility**). The soname is `libspingalett.so.1`, its
functions carry the symbol version `SPINGALETT_1.0`, and the `api.abi` test compares each release
with the one before. The names of 0.x without the prefix come only on request; pkg-config, Conan and
vcpkg find or build the library. Programs of 0.14 must be built again for the soname, and those that
use the short names include `Spingalett.Short.h`. No kernel changed since 0.14.

### Added
- **Packages:** a pkg-config file, `spingalett.pc`, which finds the installation from its own
  directory (an extracted release archive too); a Conan recipe (`conan create packaging/conan`) and a
  vcpkg port (`vcpkg install spingalett --overlay-ports=packaging/vcpkg`) that build the library of
  the checkout with OpenMP and, as an option, the GPU backend; CI builds both and a program against
  them.
- **The ABI held by tests:** the exported functions carry symbol versions on ELF platforms
  (`SPINGALETT_1.0`, `Src/Spingalett.map`), and the `api.abi` test compares the public structs'
  sizes and fields, the enumerators, the constants and the functions' declarations with those of the
  last release (`Tests/Data/abi.txt`): additions pass, changes fail. The README states what 1.x keeps
  (**Compatibility**).

### Changed
- **The names of 0.x only on request:** `Spingalett.h` no longer includes `Spingalett.Short.h`. A
  program written for 0.x includes `<Spingalett/Spingalett.Short.h>` in place of `Spingalett.h`, or
  defines `SPINGALETT_SHORT_NAMES` before including it (`Spingalett.Inference.h` gives the engine's
  names then). They are macros in the global namespace: `LOG_DEBUG`, `LOG_INFO` and `LOG_WARNING`
  are also `<syslog.h>`'s, and a program's own `train()`, `layer()` or `predict()` was replaced by
  the library's. `SPINGALETT_NO_SHORT_NAMES` is no longer needed; `Spingalett.hpp` refuses
  `SPINGALETT_SHORT_NAMES`, whose macros would replace its methods.
- The CMake package accepts any 1.x of at least the version asked for (`SameMajorVersion`); the
  library exports nothing but its functions (an OpenMP lock was exported).
- Python: the bindings load the library by its soname of 1.x (`libspingalett.so.1`,
  `libspingalett.1.dylib`) and accept a library of their major version and at least their minor one,
  where they needed the same major.minor version.

### Fixed
- Offsets applied to null pointers, undefined behaviour that Clang's UndefinedBehaviorSanitizer
  reports (GCC's does not): the data gradient of grouped convolutions without an activation, and
  training on inputs in the GPU's memory. CI runs the sanitizers with Clang too.

## [0.14.0] - 2026-10-09

"Release candidate": the last minor version of 0.x, with the API and the formats 1.0 is to keep.
Every public name is under the library's prefix, structs end with reserved space, a layer without an
activation has none, errors are reported one way, and `Spingalett.hpp` gives a C++23 interface;
`docs/Reference.md` and `docs/Tutorial.md` document them. Programs written for 0.x compile unchanged
through the short names (`Spingalett.Short.h`) but must be built again: the soname is 0.14, the
exported symbols are the prefixed ones, and a layer that left out `.act_func` now has no activation
where it had a sigmoid.

Data sets can live in the GPU's memory, the setting PyTorch's GPU benchmark measures. On an RTX 4050
Laptop GPU, Spingalett is ahead of PyTorch 2.14 with cuDNN in every workload of
`Examples/Benchmark.c`, by 1.14 times at least, with the data in the GPU's memory or in the host's
(README); against 0.13.1 it trains the MobileNet-style network 3.4 and 12 times as fast and infers it
4 and 13 times as fast (single precision and bfloat16).

### Added
- **Documentation:** `docs/Reference.md`, every declaration of the public headers with its comment
  (written by `docs/make_reference.py`; the `docs.reference` test fails when it falls behind the
  headers), and `docs/Tutorial.md`, a path through the examples from XOR to a model on a
  microcontroller. `docs/ModelFormat.md` and `docs/DatasetFormat.md` state that `.slett` version 7 and
  `.slettd` version 2 are the formats of 1.0, which later versions only extend.
- **A C++ interface**, header-only: `#include <Spingalett/Spingalett.hpp>` (C++23). The library's objects as
  move-only owners (`spingalett::Network`, `Model`, `Dataset`, `DeviceData`), data as `std::span`,
  errors as `std::expected<T, spingalett::Error>` (the library's code and message; nothing throws),
  scoped enums, `TrainOptions` (and every other field of `SpingalettTrainArgs` through its `raw`), and
  a fluent `Builder` that makes the network in `build()`, reporting the first layer the library
  refuses. `raw()` gives the C object for what the wrapper does not cover.
- Data sets in the GPU's memory: `spingalett_device_data_new(values, count, size)` copies rows of
  floats to the device once (`spingalett_device_data_free()`, `_count()`, `_size()`, `_read()`), and the
  new fields `device_inputs` and `device_targets` of `TrainArgs` (with `device_val_inputs` and
  `device_val_targets`), `PredictArgs` and `EvaluateArgs` take them in place of arrays, either or
  both. With `COMPUTE_VULKAN`, `train()` writes only the rows' indices of each chunk: the GPU gathers
  the rows, and augments and smooths them as the host does (a kernel of its own, `rows.comp`), so
  that training on a set gives the bits of training on the host's arrays. Rows that come in order
  (`predict()`, `evaluate()`, full batches) are read where they are; with `PRECISION_BFLOAT16` a set
  makes a copy of its rows as bfloat16 on first use (half its size again), which networks read as
  the inputs they keep as such. On the CPU those calls copy the rows back first. Python:
  `sg.DeviceData(array)`, which `train()`, `validation_data`, `forward()` and `evaluate()` take in
  place of arrays. On an RTX 4050 Laptop GPU the fully connected network of `Examples/Benchmark.c`
  infers 3.52 M samples per second from a set against 2.61 M from host arrays (bfloat16: 8.01 M
  against 5.00 M).

### Changed
- **Public structs can grow:** every one ends with `uint64_t reserved[SPINGALETT_RESERVED]` (8 words,
  zero), so that 1.x releases add fields without changing the layout: programs built before leave them
  zero through the builders, and zero means a new field's default. Structs the library fills have them
  zeroed. The Python bindings mirror them.
- **One way of reporting errors:** a function that fails returns false, NULL, NaN, `SPINGALETT_NO_LAYER`
  or `SPINGALETT_TRAIN_FAILED` and sets the thread's error code and message (stated in `Spingalett.h`).
  `spingalett_save()` and `spingalett_set_compute_mode()` returned nothing and now return whether they
  succeeded (the latter is false for a value of no mode, which it ignored).
- **Every public name has the library's prefix.** Types are `Spingalett*` (`SpingalettNetwork`,
  `SpingalettTrainArgs`, `SpingalettActivationFunction`, ...), enumerators `SPINGALETT_*`
  (`SPINGALETT_ACT_RELU`, `SPINGALETT_LAYER_CONV2D`, ...; the weight initializations, shortened,
  `SPINGALETT_INIT_HE` and the like), and the builders and functions `spingalett_*`:
  `spingalett_network_new()`, `spingalett_layer()`, `spingalett_conv2d()` and the other layers,
  `spingalett_train()`, `spingalett_predict()`, `spingalett_evaluate()`, `spingalett_forward()`,
  `spingalett_save()`, `spingalett_load()`, `spingalett_load_from_memory()`, `spingalett_activate()`,
  `spingalett_derivative()`, `spingalett_print_network()`, `spingalett_network_free()`; the functions the
  builders call are `spingalett_network_new_args()`, `spingalett_append_layer()` and
  `spingalett_*_args()`. The names of 0.x stay available from `Spingalett/Spingalett.Short.h` (and
  `Spingalett.Inference.h` for the engine's), which `Spingalett.h` includes unless
  `SPINGALETT_NO_SHORT_NAMES` is defined: programs written for 0.x compile unchanged, and programs
  that define that macro may use `layer`, `train`, `LOG_INFO` or `NeuralNetwork` for their own. The
  library's exported symbols are the prefixed ones (`train_struct_arguments` and the others are gone
  from the binary).
- **A layer without `.act_func` has no activation.** `ACT_NONE` is 0 in `ActivationFunction` (which now
  reads none, sigmoid, ReLU, tanh, leaky ReLU, FOO52, softmax); before, an activation left out was
  `ACT_SIGMOID`, so that an addition or concatenation written without `.act_func = ACT_NONE` applied a
  sigmoid. Programs that relied on the implicit sigmoid give `.act_func = ACT_SIGMOID`. In Python,
  `Layer` and `Network.add_layer()` default to `Activation.NONE` too, and `Activation`'s values follow
  the C enum. `.slett` files keep their activation codes (0 is sigmoid there, 6 none): files of every
  version load as before.
- `TrainArgs`, `PredictArgs` and `EvaluateArgs` have new fields (above): programs built against 0.13
  need to be built again (the soname carries the minor version).
- Depthwise convolutions (a group a channel) run on the GPU on a kernel of their own, which the matrix
  kernel ran as products of k = taps that mostly multiplied padding: a MobileNet-style network (the
  new fifth workload of `Examples/Benchmark.c`) trains 2.9 times as fast in single precision and 10
  times as fast in bfloat16 on an RTX 4050 Laptop GPU, ahead of PyTorch with cuDNN (1.2 and 1.08 times
  as fast as it), and infers 2.8 and 8.6 times as fast.
- On the GPU a batch normalization read only by a depthwise convolution (of channels in fours) is
  applied by the convolution to the values it reads, in its forward pass, its weight gradient and the
  derivative of its data gradient: the normalization's outputs are never written or read back.
  MobileNet-style training runs 10.6% faster in single precision (12,052 to 13,335 samples/s) and 8%
  in bfloat16 (26,395 to 28,516), inference 34% and 20% (58,693 to 78,612; 94,507 to 113,703), on an
  RTX 4050 Laptop GPU. The convolution's data gradient also sums what the normalization's backward
  pass needs, which then reads its gradient once less: training another 4% faster in single
  precision (13,391 to 13,927) and 1.5% in bfloat16.
- The GPU's depthwise kernel computes four adjacent pixels of a row a thread, with the filters in
  registers and the loads of overlapping windows shared: MobileNet-style training in bfloat16 runs 8.7%
  faster (24,424 to 26,545 samples/s) and its inference 17% (81,941 to 95,869) on an RTX 4050 Laptop
  GPU; in single precision, where the kernel is bound by memory, inference gains 8.4%.
- The GPU's single-precision matrix kernel writes its results four at a time where C (and what the
  epilogue reads with it) is aligned for it: training in single precision 2 to 10% faster (the
  convolutional network 103,116 to 108,934 samples/s, the U-Net 3,875 to 4,096, ResNet-20 10,232 to
  10,475), inference 4 to 9% (ResNet-20 32,170 to 34,896).
- The GPU trains in chunks of up to 4,096 samples (2,048 in networks with batch normalization, whose
  statistics are taken over each chunk as on the CPU): the fully connected network trains full
  batches 6 to 11% faster (bfloat16: 2,841,939 to 3,040,430 samples/s).
- `train()` on the GPU starts faster: the memory of networks freed before is kept for the next (up to
  an eighth of the device's memory, given back when an allocation fails), and the parameters' upload is
  not waited for. The fully connected network trains full batches 7.3% faster in single precision
  (1,095,058 to 1,174,516 samples/s) and 5.9% in bfloat16, where five epochs took some 2.4 ms more
  than their GPU time to start.
- The first training of a process on a machine whose GPU driver has not compiled the kernels yet
  makes the pipelines the tile choice times on several threads: 1.1 s instead of 2.75 s for the
  fully connected network, 2.7 s instead of 6.2 s for ResNet-20 in bfloat16.
- `predict()` and `evaluate()` on the GPU keep the parameters their copy of the network has while the
  network's do not change (they copied them on every call: 0.75 ms of a call of 3.2 for the fully
  connected network), and command buffers freed are used again (making one took 0.3 ms on NVIDIA's
  driver).
- Inference on the GPU runs in chunks sized by the bytes their activations take as they are kept
  (bfloat16 counted as two), no longer capped at 2,048 samples (inputs from host arrays still go in
  chunks of at most 2,048): in bfloat16 the convolutional networks infer 1.12 and 1.14 times as fast.
- `Examples/Benchmark.c` times inference on a second call, as PyTorch's is timed after its warm-up
  (the first call makes the network's copy on the GPU, the counterpart of moving a PyTorch model
  there), runs every GPU workload on the host's arrays and on data sets in the GPU's memory, and has
  a fifth workload, a MobileNet-style network (`benchmark_pytorch.py` too).

### Fixed
- The GPU's chunks keep a layer's values within the signed 32-bit indices its kernels compute: chunks
  of 2,048 samples of layers of 2^20 values or more read and wrote outside their buffers. Networks of
  up to 2^25 values a sample run in smaller chunks.
- A chunk filled ahead on the GPU for a step that did not come (early stopping, a callback) no longer
  leaves its rows' flags to the next `train()` of the network's copy.
- `Examples/XOR.c` seeds the generator: started from the clock, about one run in five hundred ended
  in XOR's local minimum and failed the release workflow's check of the installed package.

## [0.13.1] - 2026-10-09

"GPU": the GPU backend faster in every workload, ahead of PyTorch with cuDNN in all but the fully
connected network's inference with PyTorch's data already in GPU memory (which skips the copy over
the bus that Spingalett's host arrays take). On an RTX 4050 Laptop GPU against 0.13.0
(`Bin/Benchmark gpu`, medians of three interleaved runs): ResNet-20 trains 1.2 times as fast in
single precision and 1.84 times in bfloat16, the U-Net 1.2 and 1.8 times and infers 1.55 and 2.5
times as fast, the MNIST CNN trains 1.4 and 1.9 times as fast, and the MLP trains full batches 1.9
and 3.3 times as fast. No API changes.

### Changed
- The network keeps its copy on the GPU: `train()` keeps the parameters on the device between
  epochs (they were copied back after each, with the gradients and moments) and leaves the copy in
  the network when it returns. The parameters come back where they are read (a callback, an
  autosave, the NaN check, the best epoch's copy, and every function that reads them); the next
  `train()` with the same batch size and optimizer trains on the copy again, and `predict()` and
  `evaluate()` run on it, or take its parameters on the device into their own copy when its chunks
  are much smaller than theirs. The network frees it when it is freed, gains a layer, trains on the
  CPU or makes a trainer. Concurrent `predict()` calls give the bits of one.
- With `PRECISION_BFLOAT16`, the GPU keeps the layers' outputs (all but the output layer's, and the
  network's inputs) and their gradients as bfloat16, the values the products round them to anyway,
  as PyTorch's autocast does: half the memory and half the bytes every pass moves. The matrix units'
  kernel reads them eight to a load and moves their bits to shared memory as they are;
  normalizations, pooling, additions, concatenations, upsampling and dropout read and write bfloat16
  and compute in single precision (variants of their kernels built with 16-bit storage, which only
  devices with bfloat16 matrix units are asked for); the host rounds the inputs as it sends them,
  halving the bytes; networks of large chunks keep a bfloat16 copy of the weights, which the
  optimizer writes with them. Parameters, their gradients and the optimizer stay in single
  precision. CIFAR-10 with `Examples/CIFAR10.c resnet20` reaches the same test accuracy after 10
  epochs (85.40%, single precision 85.18%) at 15,800 samples per second against 8,680.
  `SPINGALETT_GPU_NO_BF16_STORAGE=1` keeps them in single precision.
- A network on the GPU takes its buffers from a few allocations instead of one each, moves its
  arrays in one submission instead of one per array, has the device zero a fresh network's moments,
  and where the host can write into all of the device's memory (resizable BAR, unified memory), the
  host writes each chunk's inputs and the parameters straight into it instead of the device copying
  them from host memory (`SPINGALETT_GPU_NO_HOST_WRITES=1` keeps the copies). The host's gradients
  and moments are made when they first come back from the GPU.
- The matrix products choose their tiles by the device's timestamps, in rounds that run every
  candidate once in one submission, after keeping an idle device busy for 25 ms to raise its clocks:
  the tile chosen is within noise of the fastest (it was up to 25% slower), and timing takes half as
  long (the 255 products of `Bin/Benchmark gpu`: 10.7 to 5.4 s, once per process). Larger tiles on
  the matrix units (2 x 2 to 4 x 2 accumulators a subgroup, their loops unrolled so that the
  accumulators stay in registers) and threads of `gemm.comp` grouped over 4 x 8 blocks.
- Weight gradients split their sums over the batch into slices only where the slices' partial sums
  cost little against the operands: the MLP's run unsplit (13% faster full batches in bfloat16),
  convolutions' split as before.
- Inference on the GPU runs in chunks of at most 32 MB of activations (at least 64 samples), whose
  layers' outputs stay in the GPU's cache from one layer to the next: the U-Net infers 1.57 times as
  fast in single precision.
- The pooling backward pass takes windows that tile the input (as 2 x 2 max pooling of stride 2) a
  thread per window instead of every cell searching its window again: 5.3 times as fast.
- Full-batch training fills the next epoch's first chunk while the device trains on the last of
  this one, and the host rounds inputs kept as bfloat16 straight into the device's memory as it
  gathers them.
- `SPINGALETT_GPU_PROFILE` times copies too, and reports how long the device spent on each
  submission.

### Added
- `Examples/benchmark_pytorch.py --host-data`: PyTorch with its data in host memory, as
  Spingalett's functions take it. `SpingalettGpuTests bench dense` (and `bench bf16 dense half`,
  operands kept as bfloat16) times the MLP's products.

### Fixed
- Parameters a training callback wrote on the GPU (`spingalett_set_parameters()`) were ignored by
  the epochs after it: they now go to the device before the next epoch.

## [0.13.0] - 2026-10-09

"Layers": transposed convolutions, upsampling and layer normalization, for U-Nets and more of ONNX,
on the CPU, the GPU, in files, the engine and deployment models, imported from ONNX and PyTorch.

### Added
- Transposed 2D convolutions: `conv_transpose2d()` (`LAYER_CONV_TRANSPOSE2D`; Python
  `ConvTranspose2D`, `add_conv_transpose2d()`) with strides, padding, output padding
  (`LayerArgs.output_padding`, `output_padding_h`, `output_padding_w`) and groups; the output has
  `(in - 1) stride - 2 padding + kernel + output_padding` cells an axis. A filter per output
  channel, as a convolution's. The passes are those of the convolution transposed, swapped: the
  forward pass is its data gradient, the data gradient its forward pass, the weight gradient its
  weight gradient with input and output gradient exchanged, on the CPU's kernels and the GPU's.
- Upsampling: `upsample2d()` (`LAYER_UPSAMPLE`, `UpsampleMode`; Python `Upsample2D`,
  `add_upsample()`) by integer factors (`stride_h`, `stride_w`, default 2), `UPSAMPLE_NEAREST` or
  `UPSAMPLE_BILINEAR` (the cells' centres aligned and the edges repeated, PyTorch's
  `align_corners=False`; weights from integers, the same on every platform).
- Layer normalization: `layer_norm()` (`LAYER_LAYER_NORM`; Python `LayerNorm`, `add_layer_norm()`)
  over each cell's channels (a dense layer's outputs are one cell), gamma and beta per channel, no
  running statistics, no weight decay on gamma.
- All three train on the CPU and the GPU (deterministic, the GPU's results within rounding of the
  CPU's), save in `.slett` format version 7 (`docs/ModelFormat.md`; networks without them keep
  their versions), and run in the engine and in deployment models in every precision, batch
  normalization folded into a transposed convolution before it. Float deployment models run
  transposed convolutions through the convolution kernels in batches, integer ones through the
  engine's pass, which they reproduce exactly.
- ONNX import: `ConvTranspose` (groups, strides, padding, output padding), `Resize` and `Upsample`
  by constant integer factors (nearest where it reads cell floor(o / factor): asymmetric
  coordinates rounded down or half-pixel ones rounded; linear with half-pixel coordinates), and
  `LayerNormalization` over a vector or over a map's channels between a `Transpose` to channels last
  and one back (LayerNorm2d). A U-Net exported by both PyTorch exporters and a network with both
  kinds of layer normalization import within 6e-8 of PyTorch's outputs.
- PyTorch weights: `ConvTranspose2d` (weight `[in, out / groups, kh, kw]`) and `LayerNorm` modules.
- `Examples/Segmentation.c`: a U-Net segments synthetic 64 x 64 images of circles, squares and
  triangles (a sigmoid a pixel and class, new images every epoch from a generator; `bilinear`,
  `ln`, `gpu`, `bf16`). 12 epochs on an RTX 4050 Laptop GPU: 15 s, mean IoU 0.882, pixel accuracy
  98.5%; the INT8 model keeps 0.878.
- `Bin/Benchmark` and `benchmark_pytorch.py` time that U-Net.

### Changed
- `LayerType` has three new values before `LAYER_TYPE_COUNT`; `LayerArgs`, `SpingalettNetworkLayer`
  and `SpingalettLayerInfo` have new fields (output padding, the upsampling mode), mirrored in the
  Python structures. `SPINGALETT_FORMAT_VERSION` is 7.
- GPU: the output layer's kernel takes rows of more than 64 outputs a workgroup each (sums in a fixed
  tree): the U-Net's loss kernel takes 45 us instead of 2,548 us a batch; narrower rows keep their
  thread a row and their results.
- Float deployment models run transposed convolutions in batches through the convolution kernels
  (they ran sample by sample through the engine): the U-Net's FP32 model predicts 3,150 images a
  second instead of 2,440, its FP16 model 3,190 instead of 2,140 (i7-12650H, 16 threads).
- Nothing slower: interleaved runs of `Bin/Benchmark` on the i7-12650H and its RTX 4050 Laptop GPU,
  0.12.0 against 0.13.0 (three each, one and eight threads), agree within 4% on every CPU workload
  and 5% on every GPU one, either way.
- The U-Net, images per second, training and inference (medians of three interleaved runs): one
  thread 146 and 511 (PyTorch 2.14: 134 and 292), eight threads 598 and 2,467 (365 and 754); on the
  GPU 3,279 and 6,552 (PyTorch with TF32: 3,283 and 6,329; in single precision 3,059 and 6,564),
  with bfloat16 products 3,707 and 6,880 (PyTorch's autocast, which keeps activations in bfloat16:
  4,950 and 11,918).

### Fixed
- The cross-entropy of sigmoid outputs saturated at 1 was NaN (a target of 1) or infinite (a target
  of 0), on the CPU and the GPU: the reported losses, validation and anything monitoring them went
  NaN once per-pixel sigmoids saturated. The bound on 1 - o is now taken through `fmaxf()`, which
  reassociation cannot fold back into a comparison of o.
- The header's comment on ONNX import said external data files were not read (0.12 reads them).

## [0.12.0] - 2026-10-09

"Tensor cores": the GPU's matrix units in bfloat16, the custom-loop API on the GPU, fixes from a
review of the GPU backend; model import ten to a hundred times as fast; networks that predict hold
their parameters once.

### Added
- `spingalett_set_gpu_precision(PRECISION_BFLOAT16)` (Python `set_gpu_precision()`): the GPU's matrix
  products in bfloat16 on its matrix units (tensor cores; `VK_KHR_cooperative_matrix` with
  `VK_KHR_shader_bfloat16`), the products added in single precision; the parameters, the optimizer
  and batch normalization stay in single precision, and so do products smaller than a block of the
  matrix units. Runs stay deterministic. It returns whether the GPU multiplies in the precision set.
  The tiles of the matrix units are many subgroups with one or two accumulators each, all of them
  timed on first use. On an RTX 4050 Laptop GPU (`Bin/Benchmark gpu`, medians of three interleaved
  runs, samples per second, single precision then bfloat16): ResNet-20 trains at 8,750 and 9,960 and
  infers at 22,630 and 25,300; the MNIST CNN trains at 73,680 and 85,950 and infers at 219,600 and
  257,300; the 784-512-1000-10 MLP trains mini-batches of 64 at 225,500 and 264,100. ResNet-20
  reaches 91.61% test accuracy on CIFAR-10 in bfloat16 (`Examples/CIFAR10.c bf16`), 91.55% in single
  precision. PyTorch 2.14 with autocast to bfloat16 on the same GPU (`benchmark_pytorch.py
  --cuda-bf16`, new) trains ResNet-20 at 12,160 and the MNIST CNN at 99,120.
- The custom-loop API on the GPU: a trainer made with `COMPUTE_VULKAN` (`spingalett_trainer_new()`,
  Python `Trainer`) runs its forward, backward and optimizer passes there, the outputs coming back
  for the caller's loss and gradients adding up on the device until the step; the parameters stay
  on the GPU between passes, and functions that read the network copy them back first. ResNet-20 in
  batches of 128 through `spingalett_train_on_batch()`: 7,372 samples/s, as fast as `train()`, against
  972 on the CPU.
- ONNX models whose weights are kept in external files (`torch.onnx.export` of models over 2 GB,
  `onnx.save_model(..., save_as_external_data=True)`) import from their path; the files must lie in
  the model's folder.
- PyTorch weights stored as views (a transposed or permuted tensor in a state dict) and 32-bit
  integer tensors load.
- `Bin/Benchmark gpu` runs the GPU's rows only, and the GPU's rows run again in bfloat16 where the
  GPU has matrix units; `SpingalettGpuTests bench` lists every tile's time with
  `SPINGALETT_BENCH_TILES` set.

### Changed
- Networks allocate their gradients and optimizer state when they first train (`train()`,
  `spingalett_trainer_new()`, a file with optimizer state, or setting a gradient), so that networks
  loaded or imported to predict hold their parameters once instead of four times. Parameter arrays
  grow by half again when a layer does not fit, instead of to the exact size every time: building a
  network of 1000 layers, which copied its parameters once per layer, takes 0.02 s instead of 5.6 s.
- ONNX import and PyTorch weights: files are mapped instead of read, and each tensor is converted
  straight into its layer in one pass (no float copy of every initializer, no table of indices as
  large as the tensor, no copy of the network's parameters to restore on error: everything is
  checked before anything is written); names resolve through hash tables. On an i7-12650H:
  ResNet-50 (100 MB) imports in 0.085 s instead of 1.02 s, peaking at 360 MB instead of 1139 MB; a
  126M-parameter VGG head (504 MB) in 0.17 s instead of 1.94 s, peaking at 1271 MB instead of
  4747 MB, and its state dict loads in 0.08 s instead of 1.0 s.
- GPU: the executor waits for the older of the two chunks in flight before filling the next one
  (it waited for both), and a chunk's last barrier no longer holds back the next one's copies; with
  the other changes the MLP trains 7 to 11% faster on the RTX 4050 Laptop GPU and infers 19% faster,
  and the MNIST CNN trains 5% faster (`Bin/Benchmark`, against 0.11).
- GPU: matrix products of more row tiles than a dispatch may have go in parts, tiles whose columns
  would exceed the device's limit are not chosen, and the gradient norm's sums loop over at most
  65535 workgroups (Intel and lavapipe allow 65535 workgroups in x; every device in y).
- GPU: tanh takes an odd polynomial below |x| = 0.3, as the CPU's vector kernel does (the formula
  through exp lost its relative precision there); the matrix kernel's accumulators are `precise`.
- Building: a glslc too old for the bfloat16 kernel (Ubuntu 24.04's) builds the library without it,
  which then multiplies in single precision; release packages and wheels take shaders compiled once
  with a current shaderc, checked by spirv-val.
- Nothing on the CPU: interleaved runs of `Bin/Benchmark` on the i7-12650H, 0.11.0 against 0.12.0,
  agree within 3% on every workload.

### Fixed
- GPU: validation during training ran batch normalization with the validation chunk's own
  statistics and moved the running statistics towards the validation data; it uses the running
  statistics now, as the CPU does.
- GPU: a concatenation (or a one-input addition, which ONNX import makes for activations it cannot
  fuse) that directly followed another could read its input without a barrier.
- GPU: a chunk whose command recording failed stayed cached and was submitted half-recorded by the
  next call of its size; a lost device made predictions and losses read garbage as success.
- GPU: the bfloat16 kernel used the Vulkan memory model and 16-bit floats without the device
  features enabled (NVIDIA tolerates it, a stricter driver may refuse the pipeline); it takes its
  subgroup from `gl_SubgroupID` and asks for complete subgroups where the device allows.
- GPU: the 32-bit index check skipped the input layer; devices were asked for Vulkan 1.2 features
  before their version was checked; the pipeline cache passed NULL to `memcpy`.
- README: the DeepWiki badge (its image server refuses GitHub's image proxy) and the introduction,
  which still said the library ran on the CPU only.

## [0.11.0] - 2026-10-08

"GPU": training and inference on a GPU through Vulkan compute, deterministic, with every kind of
layer the CPU runs.

### Added
- `COMPUTE_VULKAN`: `train()` (full-batch and mini-batch strategies), `predict()` and `evaluate()`
  on a GPU through Vulkan compute (NVIDIA, AMD and Intel GPUs; Apple GPUs through MoltenVK);
  `spingalett_gpu_device()` names the device (the first discrete GPU, or `SPINGALETT_GPU_DEVICE`).
  The Vulkan loader is opened at run time: the library loads and runs on the CPU without Vulkan, and
  without a usable device the mode falls back to the CPU with a warning. Every kind of layer,
  activation, loss and optimizer runs on the GPU, with dropout (the CPU's masks), gradient clipping,
  label smoothing, augmentation (on the host, overlapped), validation and early stopping; the
  parameters stay on the device for a `train()` call and come back at the end of every epoch.
  Sums run in fixed orders, so GPU runs repeat bit for bit. Python: `ComputeMode.VULKAN`,
  `gpu_device()`. Measured on an RTX 4050 Laptop GPU (`Bin/Benchmark`, medians of three runs,
  samples per second; in parentheses the eight threads of the i7-12650H, then PyTorch 2.14 with CUDA
  and cuDNN on the same GPU, with its default TF32 and in single precision): ResNet-20 trains at
  8,900 (1,573; 8,508 and 7,139) and infers at 22,892 (6,163; 19,742 and 19,409); the MNIST CNN
  trains at 62,926 (14,298; 56,906 and 59,993) and infers at 227,667 (51,250; 132,016 and 143,565);
  with batch normalization 48,387 (10,791; 48,481 and 48,477) and 149,371 (53,411; 107,254 and
  114,126); the 784-512-1000-10 MLP trains mini-batches of 64 at 188,642 (52,688; 80,581 and
  80,413), full batches at 497,191 (112,991; about 1,050,000) and infers at 967,009 (284,002; about
  2,960,000, its data already on the GPU).
- The GPU's matrix kernel serves dense layers, convolutions (windows read through a tap table,
  groups), data gradients (one stride-1 product per phase of a stride) and weight gradients (sums
  split into slices added in a fixed order), with vector loads along each operand's contiguous axis;
  the tile of each product shape is chosen by timing up to eight candidates on first use (about a
  second for ResNet-20; `SPINGALETT_GPU_TUNE=0`: estimated), which cannot change a result since every
  tile adds in the same order.
  `SPINGALETT_GPU_PROFILE=1` prints the GPU time per kernel at exit.
- `Examples/CIFAR10.c gpu` trains on the GPU: ResNet-20 reaches the same 91.55% test accuracy as on
  the CPU, in 9 minutes for 100 epochs (93 on twelve threads of the i7-12650H).
- Building: `SPINGALETT_VULKAN` (`AUTO`, `ON`, `OFF`) needs `glslc` and the Vulkan headers;
  `SPINGALETT_SPIRV_DIR` takes another build's compiled shaders. Release packages and wheels carry
  the backend.
- Tests: `SpingalettGpuTests` checks the matrix kernel in every mode and tile against double
  precision and that tiles agree bit for bit; the `gpu` group trains networks with every kind of
  layer on the GPU and the CPU and compares them. CI runs both on Mesa's lavapipe.
- `Examples/benchmark_pytorch.py --cuda` (PyTorch's defaults) and `--cuda-fp32` (no TF32).
- `Examples/CIFAR10.c resnet32` reaches 92.39% test accuracy in 100 epochs (INT8 model 92.43%,
  480 KB).

### Changed
- `ComputeMode` has a new value, `COMPUTE_VULKAN`, before `COMPUTE_COUNT`.
- Nothing on the CPU: interleaved runs of `Bin/Benchmark` on the i7-12650H, 0.10.0 against 0.11.0,
  agree within 2% on every workload and thread count (the runs of either vary by more).

### Fixed
- README: a repeated phrase in the performance section.

## [0.10.0] - 2026-10-08

"Graphs": networks become directed acyclic graphs of layers (residual connections, concatenated
branches), models come in from ONNX and PyTorch, the Python package goes to PyPI with the library
inside, and convolutions read their windows straight from the image.

### Added
- Networks as graphs: `LayerArgs.inputs` and `.input_count` name the earlier layers a layer reads
  (by default the one before it, so chains are built as before), and `layer()` and the other
  builders return the new layer's index (`SPINGALETT_NO_LAYER` on error). New kinds of layers:
  `LAYER_ADD` (`add_layers()`, the sum of layers of one shape: residual connections),
  `LAYER_CONCAT` (`concat_layers()`, layers side by side along the channels) and
  `LAYER_GLOBAL_AVG_POOL` (`global_avg_pool2d()`). A layer reads at most `SPINGALETT_MAX_INPUTS`
  (16) layers; the last layer is the output, and every other layer must feed a later one before a
  network trains or predicts. Training, the step API, `predict()`, `forward()`, `evaluate()`,
  deployment models in every precision and the inference engine run graphs; training stays
  bit-identical on any number of threads (a layer read by several gets their gradients in a fixed
  order). `SpingalettNetworkLayer` and `SpingalettLayerInfo` list a layer's inputs.
- `.slett` format version 6 (see `docs/ModelFormat.md`): each layer's inputs and where its output
  lives among the engine's activations, planned by the writer so that outputs whose lives do not
  overlap share memory. Chains are still written as versions 3 to 5. Batch normalizations fold into
  products that feed nothing else.
- ONNX import: `spingalett_import_onnx()` and `_from_memory()`, `ModelTool import`, Python
  `Network.from_onnx()` and `Network.from_torch()` (through an ONNX export in memory). A protocol
  buffer reader of the library's own; Conv (groups, strides, symmetric and same padding), Gemm,
  MatMul with a constant operand (a following Add becomes the bias), MaxPool, AveragePool,
  GlobalAveragePool and ReduceMean over the height and width, BatchNormalization, Add, Concat along
  the channels, Relu, Sigmoid, Tanh, LeakyRelu, Softmax, Flatten, flattening Reshape, Identity,
  Dropout and Constant; weights reordered for channels-last data; errors name the operator and node.
  Models exported from PyTorch 2.14 with either exporter agree with PyTorch within 1e-7.
- PyTorch weights: `spingalett_load_pytorch()` and `_from_memory()` copy a state dict saved with
  `torch.save` (or a checkpoint holding one) or a safetensors file into a network of the same
  layers, module by module in order (or in an explicit order), checking shapes and leaving the
  network unchanged on error. The pickle inside `.pt` files is interpreted without running any of
  it. Python: `Network.load_pytorch()` also takes a state dict directly.
- Label smoothing: `TrainArgs.label_smoothing`, `spingalett_trainer_set_label_smoothing()`, Python
  `TrainConfig.label_smoothing`.
- Reduce on plateau: `TrainArgs.lr_plateau_factor`, `.lr_plateau_patience` and `.lr_plateau_min_lr`
  scale the learning rate (the schedule's, when there is one) after epochs without improvement of
  the monitored value; Python `TrainConfig` has the same fields.
- Python wheels with the library inside (manylinux 2.28 x86-64 and AArch64, Windows x86-64, macOS
  universal2, for any Python 3), built and tested by the release workflow, attached to releases and
  uploaded to PyPI by trusted publishing. The bindings are a typed package (`py.typed`).
- `Examples/CIFAR10.c` builds ResNet-20 to ResNet-56 (`resnet20`, `resnet32`, ..., `wide`), trained
  with SGD, momentum, a cosine schedule and label smoothing: ResNet-20 reaches 91.55% test accuracy
  in 100 epochs (INT8 model 91.58%, 281 KB); `Bin/Benchmark` and `Examples/benchmark_pytorch.py`
  measure ResNet-20 (on the i7-12650H, 1.1 to 1.8 times PyTorch's training speed and 1.5 to 2.8
  times its inference speed).
- Tests: the `graph` group (gradient checks of residual, projected, branching and multiply-read
  layers, determinism, inference in every precision, files, folding, memory, validation) and the
  `onnx` group (models and state dicts exported from PyTorch, written by
  `Tests/make_test_models.py`, unsupported operators, truncated files); the importers ran 24,000
  mutated files under AddressSanitizer and UBSan.

### Changed
- Convolutions run as indirect matrix products: the matrix kernels read each pixel's window from the
  image through one pointer per kernel tap (Dukhan's indirect convolution) instead of gathering
  windows and transposing them into panels, for the forward pass, the data gradient (of strided
  convolutions a phase of input cells at a time, over the taps that cover them only: a quarter of
  the products of a 3 x 3 kernel with stride 2) and the weight gradient (tiles of filters by window
  elements summed over the pixels in registers, in slots fixed by the shape). Threads take tiles
  dynamically, which keeps the performance cores of hybrid processors busy. Measured on an
  i7-12650H (medians of three interleaved runs of `Bin/Benchmark`, samples per second, 0.9 ->
  0.10): the MNIST CNN trains at 2,615 -> 3,090 on one thread and 12,230 -> 14,591 on eight and
  infers at 9,174 -> 11,183 and 44,227 -> 50,862; with batch normalization 2,259 -> 2,639,
  9,745 -> 11,027, 9,205 -> 11,378 and 43,980 -> 55,646. ResNet-20 trains at 234 -> 377 and
  986 -> 1,584 and infers at 958 -> 1,387 and 4,798 -> 6,402 (against 0.10 built without the new
  kernels, `-DSPINGALETT_NO_DIRECT_CONV`). Dense networks and deployment models are unchanged.
  Float results of convolutions change in the last bits (another summation order); training
  remains the same bits on any number of threads. Depthwise convolutions, kernels of more than 64
  taps and OpenBLAS keep the previous products.
- `layer_struct_arguments()` returns the layer's index (it returned nothing).
- `LayerArgs`, `SpingalettNetworkLayer`, `SpingalettLayerInfo`, `SpingalettModel` and `TrainArgs`
  have new fields; zero-initialized arguments keep the previous behavior.
- Inference outputs share memory where their lives do not overlap: `predict()` needs a few of the
  widest layers per sample instead of all of them, which also makes it faster.
- The Python bindings are a package (`Bindings/Python/spingalett/`); `import spingalett` is
  unchanged.

### Fixed
- Release builds with GCC 16 crashed in `evaluate()` (and four test groups failed): with LTO, GCC
  16.1 dropped the stack realignment of a function that inlined the thread-local error state and
  still read its arguments through it. `set_error()` now stays out of line.

## [0.9.0] - 2026-10-08

"Bottlenecks": every part of the library profiled and its slow paths removed, from data set files
through training and inference to model files and the Python bindings. Measurements on a 4-vCPU
Xeon @ 2.1 GHz with AVX-512, 4 threads unless noted.

### Added
- `ROADMAP.md` (plans for the next releases, 1.0 and later) and `AGENTS.md` (layout, checks and
  invariants for coding agents and contributors).
- `.slettd` format version 2 (see `docs/DatasetFormat.md`): metadata records for the input shape,
  names of sets of targets, class names, and further sets of targets for the same samples (such as
  CIFAR-100's fine and coarse labels); and an rANS coder of half-bytes that decodes dense data such
  as photographs two to three times as fast as the binary range coder at about the same size.
  The writer chooses the coder per stream, and still writes version 1 for data sets that need
  neither. CIFAR-10's training set takes 112.1 MB (`xz -9` of the original batches: 116.3 MB).
- `SpingalettDataset.height`, `.width`, `.channels` and `.class_names`, filled by
  `spingalett_load_idx()`, `spingalett_load_cifar()` (class names from `batches.meta.txt`,
  `fine_label_names.txt` or `coarse_label_names.txt` next to the batches) and `.slettd` files, kept
  by `spingalett_dataset_split()`; `spingalett_dataset_set_class_names()`.
- `DatasetSaveOptions.target_name`, `.extra_targets` and `.extra_target_count`
  (`SpingalettTargetSet`), `spingalett_load_dataset_targets()` and
  `spingalett_load_dataset_from_memory_targets()`; `SpingalettDatasetInfo.format_version`,
  `.height`, `.width`, `.channels`, `.target_set_count` and `.target_set`;
  `spingalett_dataset_target_set_name()`, `spingalett_dataset_class_name()` and
  `spingalett_dataset_target_set_size()`.
- `spingalett_dataset_open_ex()` with `DatasetReaderOptions`: `in_memory` decodes the file once and
  keeps its values in their compact form (a byte per 8-bit value, a quarter of float32) with
  every pass shuffling all samples; `target_set` serves another set of targets; `no_prefetch`
  keeps decoding off a background thread. `spingalett_dataset_open_u8()` trains from 8-bit data in
  memory without a float copy.
- `DatasetTool cifar` (CIFAR-10, and CIFAR-100 with both sets of labels), `DatasetTool images`
  (one subfolder per class; PNG, JPEG, BMP and other formats through stb_image, optional resizing)
  and `DatasetTool verify`; `DatasetTool info` prints the shape, sets of targets and class names.
- Python: `save_dataset(..., shape, class_names, target_name, extra_targets)`,
  `load_dataset(path, target_set)`, the shape and sets of targets in `dataset_info()`, and
  `train_from_file(..., in_memory, prefetch, target_set)`.

### Changed
- Deployment models made by the library (`spingalett_model_from_network()`, `_load()`,
  `_from_memory()`) prepare what batched prediction runs on once, on the first call that needs it:
  FP16 and BF16 weights expanded to float, INT4 and INT2 rows unpacked, integer rows interleaved
  for the tile kernels, transposed filters and row sums. Before, `spingalett_model_predict()` built
  them, and all of its buffers, on every call; it now also keeps the last call's workspace for the
  next one. Models stay safe to share between threads (the preparation takes a lock, the workspace
  an atomic exchange); models filled in by `spingalett_model_init()` own nothing and prepare on
  every call as before. The prepared forms take memory next to the image: up to the float size of
  FP16 layers and four times the packed size of INT2 layers. `predict()` of one sample of a
  784-256-128-10 MLP: FP32 63 -> 12 us, FP16 403 -> 8 us, INT8 10.5 -> 4.4 us, INT4 27 -> 4.2 us;
  batches of 1024: FP16 2.04 -> 0.92 us, INT8 0.54 -> 0.31 us per sample.
- Matrix products of at most 8 rows (4 with AVX2, 2 without AVX) from row-major operands, such as
  a dense layer on one sample or a few, take dot products instead of packing all of the weights:
  the FP32 MLP above predicts 8 samples in 2.2 us each instead of 7.9, and training it with batches
  of 2, 4 and 8 takes 0.33, 0.175 and 0.076 s per 4096 samples instead of 0.57, 0.28 and 0.093.
  Float outputs of one sample or a few may now differ in the last bits from those of the same
  samples in a larger batch; integer models still compute exactly what single runs compute, and
  results still do not depend on the thread count.
- Loops that would run on one thread no longer enter an OpenMP parallel region (whose false `if`
  clause still cost about 0.2 us), and integer layers start threads from 2^19 multiply-adds instead
  of 2^15: `predict()` of one sample of a 4-8-8-2 INT8 model takes 0.21 us instead of 2.19.
- Pooling (forward and backward) runs a whole block of 16 channels without a test per channel, so
  that it vectorizes; depthwise convolutions with one filter per channel and 16, 32 or 64 channels
  sum their taps in registers, forward and for the data gradient; the tile kernels' epilogue, window
  gathering and the activation maximum of integer layers lost their overheads. Results are the same
  bits as before. Batched INT8 prediction of a small CIFAR-sized CNN: 36.6 -> 33.3 us per sample;
  training a depthwise-separable block on 32 x 32 x 3 images: 3017 -> 3514 samples/s, a small CNN
  with batch normalization 7188 -> 7529.
- A mini-batch step gathers and augments its samples on the OpenMP threads.
- Model files load and save three to ten times as fast: CRC-32 eight bytes at a time in the library
  (the engine alone keeps its byte-wise loop and 1 KB table), FP16 conversion eight values at a
  time with F16C (the same halves as the portable conversion for every value but NaN, which keeps
  the portable path), and networks built without reallocating every parameter array for each layer
  (adding a layer copied and zeroed all of them; loaders now reserve the final sizes). Files are
  byte for byte the same. A 924,930-parameter MLP: FP32 save 10.8 -> 2.5 ms, load 26 -> 8.3 ms, as
  a model 9.5 -> 2.4 ms; FP16 save 15.2 -> 1.4 ms; INT8 load 13 -> 1.3 ms.
- Python: `Network.input_size` / `output_size` read the sizes instead of describing every layer,
  and per-batch calls pass arrays by address: `forward()` of a 4-8-2 network 42.6 -> 7.8 us, of the
  MLP above on one sample 69 -> 22 us, `Model.predict()` of one sample 9.1 -> 5.4 us,
  `Network.from_bytes()` of that MLP 6.2 -> 0.75 ms.
- Streaming readers decode the next chunks on a background thread when a processor is free for it
  (fewer OpenMP threads than processors), and otherwise several chunks at a time on the OpenMP
  threads when they are needed: a thread competing with the OpenMP threads for the processors
  stalls their barriers. Chunks stay in their compact form until a batch is read, and batches
  convert to float on the OpenMP threads. One epoch of a small CNN on CIFAR-10 streamed from a
  `.slettd` file: 14.7 s with 0.8, 5.5 s now (4.5 s from float arrays, 4.6 s from the file in
  memory).
- A reader's shuffled order is drawn from a seed taken when it opens and from the number of the
  pass or chunk, so it is the same with or without a background thread and on any number of
  threads; it differs from 0.8's order for the same `spingalett_seed()`.
- `spingalett_save_dataset()` compresses chunks in parallel (CIFAR-10's training set: 12.4 s
  before, 2.2 s on 4 threads; MNIST: 1.11 s before, 0.47 s).
- `SpingalettDataset` has new fields: a data set built by hand must be zero-initialized
  (`SpingalettDataset d = {0};`), because `spingalett_save_dataset()` reads them and
  `spingalett_dataset_free()` frees `class_names`. The loaders initialize every field.
- `SpingalettDatasetInfo` and `DatasetSaveOptions` have new fields; zero-initialized options keep
  the previous behavior.
- Python: `uint8` inputs are 8-bit images, value q read as q / 255, in `train()`, `forward()`,
  `predict()`, `evaluate()` and `save_dataset()` (before, they were converted to floats 0 to 255).
  `train()` keeps them as bytes and converts a batch at a time. Targets keep their values.
- `DigitPadTrain` prints each epoch's time and the time since training began.

### Fixed
- `.slettd` readers reject index entries whose offset plus size wraps around (they read before the
  file's buffer), and metadata records or set numbers of 2^32 - 1, which plus one named the inputs
  (a name record shorter than 4 bytes scanned past the metadata). `spingalett_save_dataset()`
  rejects `extra_target_count` of 255 or more instead of overflowing the set count. The Python
  bindings reject a negative `target_set`.
- Streaming and loading `.slettd` files of 2 GB or more on Windows, where `long` file offsets have
  32 bits.
- Threads opening data sets at the same time no longer race to build the table of 8-bit values.
- Builds with `SPINGALETT_PORTABLE_KERNELS` and without `-march=native` link again (broken in 0.8).

## [0.8.0] - 2026-10-08

Deeper convolutional networks: batch normalization, grouped and depthwise convolutions, image
augmentation and a CIFAR-10 example.

### Added
- `batch_norm()` (`LAYER_BATCH_NORM`): per-channel normalization of the previous layer with the
  batch's statistics while training and running averages of them (`.momentum`, default 0.1;
  `.epsilon`, default 1e-5) in `predict()`, `forward()`, `evaluate()` and models; an activation
  and dropout of its own. Gamma and beta are the layer's weights and biases for every optimizer,
  clipping, the step API and `spingalett_get_parameters()`; weight decay leaves them alone.
  `PARAM_RUNNING_MEAN` and `PARAM_RUNNING_VARIANCE` read and write the statistics, which
  `restore_best_weights` restores with the weights. Training stays bit-identical on any number of
  threads; gradients are checked numerically after convolutions, dense layers, the input and
  pooling.
- Inference with a normalization after a dense or convolution layer costs no more than the layer
  alone: `predict()` and `forward()` apply it in the layer's matrix-product epilogue, and deployment
  models (`spingalett_model_from_network()`, and files saved in other precisions than FP32 without
  optimizer state) fold it into the layer's weights and biases. Other normalizations remain layers
  that the inference engine runs in every precision.
- Grouped convolutions: `conv2d(.groups = g)` splits input channels and filters into `g` groups;
  as many groups as input channels make a depthwise convolution, with any number of filters per
  channel. Depthwise convolutions run directly (all of a pixel's channels at once), other groups as
  one matrix product per group, in training and in the engine (integer results bit-exact against a
  reference, batched identical to single runs).
- `.slett` format version 5 (written only for networks that need it): layer table entries of 80
  bytes adding convolution groups and the normalization's epsilon and momentum; normalizations
  store gamma, beta and the running statistics in FLOAT32 (docs/ModelFormat.md).
- Image augmentation in `train()`: `.augment_shift` (random shifts by up to that many cells, zeros
  shifted in) and `.augment_flip` (mirror images for half of the samples), drawn per sample and
  step from a seed, so results do not depend on the thread count.
- `spingalett_load_cifar()` reads CIFAR-10 and CIFAR-100 binary batches as 32 x 32 x 3
  channels-last images.
- `Examples/CIFAR10.c` and `Examples/download_cifar10.sh`: a batch-normalized network with
  augmentation, optionally depthwise-separable, evaluated as FP16 and INT8 models.
- Python: `BatchNorm`, `add_batch_norm()`, `get_running_statistics()` / `set_running_statistics()`,
  `Conv2D(groups=...)`, `load_cifar()`, `TrainConfig.augment_shift` / `augment_flip`.
- Batched integer inference on processors with byte dot-product instructions:
  `spingalett_model_predict()` and `spingalett_model_evaluate()` run integer convolutions (with one
  group) and dense layers (from 12 samples) in tiles of 12 pixels or samples against weight rows
  interleaved four bytes at a time, with kernels for AVX-512 VNNI, AVX-VNNI and the Arm dot
  product extension (Src/Spingalett.Int8Tiles.c). Builds without `-march=native` carry them and
  choose at run time: the x86-64 packages, and on AArch64 Linux. The INT8 CIFAR-10 network
  predicts 3.1 times as fast as in 0.7 on one thread (2,520 images/s) and 2.7 times on four, 2.5
  and 2.1 times as fast as its FP32 model. Results are unchanged (integer sums are exact), and
  still identical to single runs.
- The macOS package is built with OpenMP and carries LLVM's OpenMP runtime (`libomp.dylib`, with
  its license) next to the library, which finds it through `@loader_path`.
- `Bin/Benchmark` and `Examples/benchmark_pytorch.py` also measure the convolutional network with
  batch normalization: Spingalett trains it 1.5 to 2 times as fast as PyTorch and infers it 4.5 to
  5.6 times as fast.

### Changed
- `SPINGALETT_FORMAT_VERSION` is 5. `LayerArgs`, `TrainArgs`, `SpingalettNetworkLayer` and
  `SpingalettLayerInfo` gained fields, and `ParameterKind` gained two values: rebuild programs
  against the new headers.
- DigitPad's model is a batch-normalized convolutional network: 99.58% MNIST test accuracy and
  99.45% on randomly distorted digits (the MLP: 99.27% and 98.44%).
- The inference engine's INT8 convolutions take four pixels against four filters at a time, with
  the filters' sums computed once per layer on AVX-512 VNNI: one CIFAR-10 image through
  `spingalett_model_run()` takes 1.7 to 1.8 times less time. GCC's partial redundancy elimination
  is off for the integer dot-product kernels, where it copied the sums through memory on every
  step. The engine's scratch for an INT8 convolution grows by 4 bytes per filter.
- The shared library's soname is `libspingalett.so.0.8`.

## [0.7.0] - 2026-10-07

Convolutional networks: training, saving, and deployment down to microcontrollers, with the
network becoming an opaque handle.

### Added
- Layer types (`LayerType`): `conv2d()` (2D convolution with any kernel, stride and padding, also
  rectangular), `max_pool2d()` and `avg_pool2d()` next to dense layers. Tensors are channels-last
  (`height x width x channels` per sample); the input layer takes a shape (`.height`, `.width`,
  `.channels`), and a dense layer reads what precedes it as a flat vector. Every training strategy,
  optimizer, schedule, dropout, the step API, `predict()` and `evaluate()` work with them. Gradients
  are checked numerically for every layer kind, activation and compute mode.
- Convolutions run as matrix products whose input windows the matrix kernels gather themselves
  (implicit im2col), with the bias and activation applied while each tile of the result is in
  cache; weight gradients are split over threads in slots fixed by the shape. On one core the
  convolutional network of `Examples/MNIST_CNN.c` trains 1.6 times and infers 4 times as fast as
  PyTorch (1.8 and 4 times on four cores).
- Accessors for the opaque network: `spingalett_layer_count()`, `spingalett_network_layer()`
  (`SpingalettNetworkLayer`: type, shape, outputs, activation, dropout, window, parameter counts),
  `spingalett_input_size()`, `spingalett_output_size()`, `spingalett_parameter_count()`,
  `spingalett_network_loss()`, `spingalett_optimizer_steps()`, and `spingalett_get_parameters()` /
  `spingalett_set_parameters()` for the weights, biases and gradients of a layer (`ParameterKind`).
- `.slett` format version 4 for networks with convolution or pooling layers: 64-byte layer table
  entries with the layer's kind, shape, window, stride and padding, the input shape in the header
  (docs/ModelFormat.md). Networks of dense layers are still written as version 3.
- The inference engine and deployment models run convolutions and pooling in every precision:
  integer convolutions quantize each sample's input once and take integer dot products of the
  filters with each window; windows shorter than 32 values (a first layer over one or three
  channels) accumulate all filters at once. Batched prediction of integer models remains
  identical to single runs. The MNIST CNN keeps 98.9% test accuracy in INT8 (424 KB) and 98.75% in
  INT4, and runs on the Cortex-M4 example with 208 KB of RAM. `SpingalettLayerInfo` describes
  kinds, input and output shapes and windows; `ModelTool info` prints them.
- Python: `Input`, `Conv2D`, `MaxPool2D`, `AvgPool2D` layer specs, `add_input()`, `add_conv2d()`,
  `add_max_pool2d()`, `add_avg_pool2d()`, `Network.layers` / `layer(i)` (`LayerDescription`),
  convolution weights as `(filters, kernel_h, kernel_w, channels)` arrays, `LayerType`, and model
  layer shapes in `LayerInfo`.
- `Examples/MNIST_CNN.c`: a two-convolution MNIST network (about 99% test accuracy after two
  epochs), evaluated as FP16, INT8 and INT4 models. `Bin/Benchmark` and
  `Examples/benchmark_pytorch.py` measure it too.
- A test that trains a convolutional and a dense network on 1, 3 and 4 threads and single-threaded
  and requires identical weights.

### Changed
- **Breaking:** `NeuralNetwork` is an opaque type; read networks through the accessors above
  (`net->topology`, `net->weights` and the other fields are gone from the public header).
  `LayerArgs` gained the fields of convolution and pooling layers, `SpingalettLayerInfo` and
  `SpingalettModel` gained fields, and `LayerType` moved to `Spingalett.Inference.h` (included by
  `Spingalett.h`): rebuild programs against the new headers. `SPINGALETT_FORMAT_VERSION` is 4.
- Products with at most 16 columns from row-major operands (output layers of a few units) run as
  blocks of dot products instead of mostly idle matrix panels: a 64 x 10 x 1000 product takes 23
  instead of 49 us on one thread and 9 instead of 58 us on four.
- The matrix kernels decide whether to use threads from the size and kind of each product
  (products below about 10 us of work run on one thread). A 32-64-64-1 regression network trains
  65% faster with OpenMP (7% faster on one thread) and infers 40% faster (10%).
- A dense layer's bias and activation run in its matrix product's epilogue. Epilogues apply
  activations row by row, so results do not depend on how threads split a product: with the
  built-in kernels, training gives the same bits on any number of threads and single-threaded.
- K blocks of the matrix kernels have equal sizes (288 runs as 2 x 144 rather than 256 + 32), and
  when the whole right operand fits the pack buffer, each tile of the result runs through all K
  blocks while it is in cache. The 784-512-1000-10 benchmark network runs as fast as in 0.6.
- Batch training and `predict()` take fewer samples per chunk when a sample's activations are
  large, so that convolutional networks need tens rather than hundreds of megabytes.
- The shared library's soname is `libspingalett.so.0.7`.

## [0.6.0] - 2026-10-07

A performance release: the same API and file formats, faster kernels.

### Added
- Run-time choice of matrix kernels on x86-64. Libraries built without `-march=native` with GCC
  or Clang (the release packages among them) also contain the GEMM kernels for AVX2+FMA and for
  AVX-512 and run the best set the processor supports. On a Cascade Lake core the baseline x86-64
  package now trains 3.6 to 4.9 times as fast and infers 5 times as fast. The CMake option
  `SPINGALETT_CPU_DISPATCH` (on by default) controls it.
- `spingalett_cpu_kernels()` names the matrix kernels in use (`"AVX-512"`, `"AVX2"`, `"AVX"`,
  `"SSE2"`, `"NEON"` or `"C"`); Python: `cpu_kernels()`. `Bin/Benchmark` prints it.

### Changed
- GEMM: the last row panel of a block runs a kernel of 8 or 4 rows (AVX-512) or 4 or 2 rows (AVX)
  instead of computing zero rows; transposed operands are packed with SIMD transposes; a call is
  one OpenMP parallel region, and products with few rows (mini-batches) pack their shared operand
  once for all threads, which then split the columns. A 64-row product runs 35% faster on one
  thread and 55% faster on four; mini-batch training (784-512-1000-10, batches of 64, Adam) is 36%
  faster on four threads and 5% faster on one. Threads still split the result only by rows and
  columns, so results do not depend on the thread count.
- Batch training applies the optimizer to each layer as soon as its gradient is complete, while
  the gradient is still in cache (steps with gradient clipping keep the old order).
- Inference engine: INT4 and INT2 codes are decoded in registers with AVX2 and NEON, against
  activations permuted once per layer into the order the codes are decoded in; four weight rows
  share every activation load in all precisions. One sample through 784-512-1000-10 takes 18 us in
  INT4 (61 before) and 17 us in INT2 (177 before); through 784-256-128-10, 5 us in INT4 and INT2
  (17 and 44 before) and 10.5 us in FP16 and BF16 (16 and 18 before). Results are unchanged: the
  integer kernels remain bit-exact and batched prediction still equals single runs.
- The shared library's soname is `libspingalett.so.0.6`.

## [0.5.0] - 2026-10-06

### Added
- Deployment models (`SpingalettModel`): read-only networks that keep their weights in the
  precision they are stored in and compute with them. INT8, INT4 and INT2 layers quantize their
  input to 8 bits per sample and accumulate int8 x int8 products in 32-bit integers, with kernels
  for AVX2 (with VNNI where the compiler targets it), SSE2, NEON (with the dot-product extension
  where available), the Arm DSP extension and portable C; FP32, FP16 and BF16 layers compute in
  float. One sample through the 784-512-1000-10 benchmark network takes 20 us in INT8 against
  131 us with `forward()`; INT8 and INT4 keep the MNIST accuracy of the example networks.
  `spingalett_model_from_network()`, `spingalett_model_load()`, `spingalett_model_from_memory()`,
  `spingalett_model_predict()` (batched, OpenMP; identical to single runs),
  `spingalett_model_evaluate()` and `spingalett_model_free()`.
- The inference engine, `Spingalett.Inference.h` with `Src/Spingalett.Inference.c`: checks a
  `.slett` image (`spingalett_model_init()`) and runs it in place (`spingalett_model_run()`) with a
  workspace from the caller, without allocation, I/O or global state, so images can live in flash.
  Compiled with `-DSPINGALETT_INFERENCE_ONLY` it needs nothing beyond `memcpy`, `memset`, `expf`,
  `tanhf` and `lrintf` (about 6 KB of code on a Cortex-M4); the CMake option
  `SPINGALETT_INFERENCE_ONLY` builds the library as the engine alone.
- `spingalett_save_to_memory()`, `load_spingalett_from_memory()` and `spingalett_free()`.
- `spingalett_export_c_header()`: a model as a C header (aligned byte array plus size, input,
  output and workspace macros) for compiling it into firmware.
- `ModelTool` (`Examples/ModelTool.c`, installed with the library): `info`, `convert`, `header`,
  `eval` (accuracy in every precision) and `bench`.
- `Examples/Embedded`: MNIST on a Cortex-M4F under QEMU, the INT8 model in flash and 2.8 KB of RAM.
- `docs/ModelFormat.md`, the specification of the model format.
- Python: `Model` (`load`, `from_bytes`, `predict`, `evaluate`, `layers`, `to_bytes`),
  `Network.to_model()`, `Network.to_bytes()`, `Network.from_bytes()` and
  `Network.export_c_header()`.
- `Examples/MNIST.c` reports the accuracy of the trained network in FP16, INT8 and INT4;
  `Examples/Benchmark.c` measures deployment models in every precision.

### Changed
- Model files are written in format version 3: a magic number, a layer table and 16-byte aligned
  sections, little-endian, with CRC-32 checksums of the header and the contents. Files of versions
  1 and 2 still load; 0.4 cannot read files written by 0.5.
- Integer precisions store one scale per weight row instead of one per tensor; INT2 uses ternary
  thresholds (0.7 of the mean magnitude per row) with the mean magnitude of the kept weights as the
  scale. Biases are always stored in float, and so is the optimizer state.
- `load_spingalett()` reads the file into memory and parses it there; damaged files fail with
  `SPINGALETT_ERR_INVALID` (checksum or layout) or `SPINGALETT_ERR_FILE_IO` (truncated).
- `ActivationFunction`, `LossFunction`, `PrecisionMode`, the error codes and `SPINGALETT_API` are
  defined in `Spingalett.Inference.h`, which `Spingalett.h` includes.
- The shared library's soname is `libspingalett.so.0.5`.

### Fixed
- Data set files opened from several threads at once raced on the CRC-32 table, which was built on
  first use; it is now a constant shared with the model format.

## [0.4.1] - 2026-10-03

### Added
- DigitPad for Windows. `Apps/DigitPad/Package/build-windows-zip.sh` builds
  `DigitPad-<version>-windows-x86_64.zip`: `DigitPad.exe`, the trained model, `libspingalett.dll`,
  `SDL2.dll` and the runtime DLLs they load (collected from the import tables), a `README.txt` and
  the licences. It runs in an MSYS2 UCRT64 shell or cross-compiles on Linux with MinGW-w64.
  Releases attach the zip next to the AppImage.
- `DigitPad.exe` is a GUI program with an icon, version information and a manifest that selects
  UTF-8 as the process code page, so the model loads from folders with non-ASCII names. It prints
  `--help`, `--classify` and `--verbose` output to the console it was started from.
- `Apps/DigitPad/Package/test-windows-zip.ps1`: unpacks the zip into a folder with a non-ASCII
  name, classifies `seven.pgm`, then opens the window, draws a 7 with the mouse and checks the
  prediction.

### Changed
- The release workflow trains the DigitPad model once and packages the same model in the AppImage
  and the Windows zip. Both are tested by classifying `seven.pgm`.
- A release published for an older tag through `workflow_dispatch` is no longer marked as the
  latest release.
- DigitPad decides whether to double its window from the usable display area instead of the
  display mode.

### Fixed
- `DigitPadTrain` exits with an error when it cannot write the model, instead of reporting
  success.
- DigitPad `--classify` reports errors on stderr only, without opening a message box. The footer
  shows the model's file name after a `\` as well as a `/`.
- DigitPad links `SDL2main` before SDL2, which MinGW requires.

## [0.4.0] - 2026-10-03

### Added
- Validation during training: `val_inputs`, `val_targets` and `val_count` in `TrainArgs` are
  evaluated after every epoch. The best epoch of a monitored quantity (`monitor`: validation
  loss or accuracy, or training loss) is tracked; `early_stopping_patience` and
  `early_stopping_min_delta` stop training when it no longer improves, and
  `restore_best_weights` ends training with the best epoch's parameters, kept in memory.
- `evaluate()`: mean loss and accuracy over a data set.
- A low-level training API for custom loops and losses: `spingalett_trainer_new()`,
  `spingalett_trainer_forward()`, `spingalett_trainer_backward()` (built-in loss),
  `spingalett_trainer_backward_output_grads()` (custom loss from dL/d(output)),
  `spingalett_trainer_step()` with `OptimizerArgs`, `spingalett_trainer_zero_grad()` and
  `spingalett_train_on_batch()`. Backward passes accumulate, so one step can span several.
- Data sets: `spingalett_load_idx()` (MNIST format), `spingalett_load_csv()`,
  `spingalett_dataset_shuffle()`, `spingalett_dataset_split()` and `spingalett_dataset_free()`.
- Python: `validation_data`, `monitor`, `early_stopping_patience`, `early_stopping_min_delta` and
  `restore_best_weights` for `Network.train()`, `Network.evaluate()`, `Trainer`,
  `Network.get_weight_gradients()` / `get_bias_gradients()`, `load_idx()` and `load_csv()`.
- `.slettd` data set files: `spingalett_save_dataset()`, `spingalett_load_dataset()`,
  `spingalett_load_dataset_from_memory()` and a streaming reader (`spingalett_dataset_open()`,
  `spingalett_dataset_read()`, and `spingalett_dataset_generator()` for `train()`). Values are kept
  in the smallest lossless encoding (8-bit, half, float32, class indices for one-hot targets) or a
  requested lossy one, and compressed with an adaptive context-model range coder in independently
  decodable chunks with CRC-32 checksums: the MNIST training set takes 7.8 MB (IDX: 47.1 MB).
  Specified in `docs/DatasetFormat.md`; `Examples/DatasetTool.c` converts IDX and CSV files. Python:
  `save_dataset()`, `load_dataset()`, `dataset_info()` and `Network.train_from_file()`.
- Prebuilt release archives for Linux x86-64 (baseline and AVX2/FMA builds) and ARM64, Windows
  x86-64 (MinGW-built DLL with MinGW and MSVC import libraries) and macOS (universal), built,
  tested and published by `.github/workflows/release.yml` for every tag. `DatasetTool` is
  installed with the library.
- `SPINGALETT_STATIC`: define it when compiling the sources into a program or a static library.
- DigitPad (`Apps/DigitPad`, CMake option `BUILD_APPS`): a desktop app that classifies digits
  drawn with the mouse, its trainer (99.27% MNIST test accuracy with on-the-fly augmentation) and
  a script that packages app and model as a Linux AppImage.

### Changed
- The epoch callback is now `bool (*)(NeuralNetwork *, const TrainProgress *, void *user_data)`:
  `TrainProgress` carries the epoch, training loss, learning rate, validation metrics and the
  best epoch so far, and `TrainArgs.callback_data` is passed as `user_data`. In Python the
  callback is `callback(network, progress)`.
- `train()` returns a `TrainReport` (status, epochs run, last losses and metrics, best epoch);
  in Python a `TrainResult`.
- The training loss is computed in every epoch, not only in reported ones.
- Model files use the extension `.slett` (`SPINGALETT_MODEL_EXTENSION`), appended when a file name
  has none; files saved as `.nn` load as before.
- A generator that answers 0 to the first request of an epoch is asked once more before training
  stops, so generators that mark the end of each pass with a 0 also work with `sample_count`.
- The shared library's soname carries the minor version while the major version is 0
  (`libspingalett.so.0.4`), because 0.x minor releases are not ABI compatible.
- `Examples/MNIST.c` holds out 5,000 training images for validation, keeps the best epoch and
  evaluates the test set once.

### Fixed
- Windows builds with MinGW: aligned buffers came from `aligned_alloc`, which the Windows C
  runtime does not provide, and were released with `free()`; they now use `_aligned_malloc` and
  `_aligned_free`, as MSVC builds already did.
- The examples measure time with a fallback where C11 `timespec_get` is missing (some Windows C
  runtimes).

## [0.3.0] - 2026-10-03

### Added
- Native single-precision matrix multiplication with AVX-512, AVX/FMA and portable C kernels and
  OpenMP parallelism. Batch training and inference no longer need OpenBLAS to run as
  matrix-matrix products.
- `predict()`: batched inference over many samples (Python: `Network.forward` on a 2-D array).
- `Examples/MNIST.c` (with `Examples/download_mnist.sh`): an MLP reaching about 98% test
  accuracy in a few seconds.
- `Examples/benchmark_pytorch.py`, the PyTorch counterpart of `Examples/Benchmark.c`.

### Changed
- Full-batch and mini-batch training use matrix-matrix products in every compute mode; OpenBLAS
  is one of two GEMM providers. Without OpenBLAS, full-batch training of the benchmark network is
  about 7x faster than in 0.2 and on par with OpenBLAS when using OpenMP.
- Large batches are processed in chunks of 2048 samples whose gradients are accumulated, which
  bounds the memory used by full-batch training on large datasets.
- `Examples/Benchmark.c` reports full-batch, mini-batch and inference throughput per backend.

## [0.2.0] - 2026-10-03

### Added
- Dropout: `LayerArgs.dropout_rate` (inverted dropout on hidden layers, all backends and strategies).
- Learning-rate schedules: `TrainArgs.lr_scheduler` / `lr_scheduler_data` with the built-ins
  `spingalett_lr_cosine_decay`, `_linear_warmup`, `_step_decay` and `_warmup_cosine`.
- Data generators: `MODE_GENERATOR_FUNCTION` with `TrainArgs.generator` / `generator_data`.
- Gradient clipping (`max_grad_norm`) for per-sample training; it previously applied to batch
  training only.
- `TrainArgs.blas_num_threads` to control OpenBLAS threads during training (automatic by default).
- `TrainArgs.do_not_shuffle`, `WEIGHT_INITIALIZATION_LECUN`.
- `spingalett_seed()`, `spingalett_version()` and the `SPINGALETT_VERSION_*` macros.
- Error codes `SPINGALETT_ERR_FILE_IO` and `SPINGALETT_ERR_FORMAT_VERSION`.
- Python bindings in `Bindings/Python`.
- Test suite (`ctest`) and GitHub Actions CI, including sanitizer and portable (non-AVX) builds.
- CMake package configuration (`find_package(Spingalett)`, target `Spingalett::spingalett`) and
  the options `BUILD_TESTS`, `SPINGALETT_NATIVE_ARCH`, `SPINGALETT_BIN_DIR`, `SPINGALETT_LIB_DIR`.

### Changed
- `WEIGHT_INITIALIZATION_XAVIER` is Glorot normal (variance 2 / (fan_in + fan_out)). The previous
  behavior (variance 1 / fan_in) is `WEIGHT_INITIALIZATION_LECUN`.
- Adam, AdamW and RMSProp add epsilon to the square root of the second moment (`sqrt(v) + eps`),
  as PyTorch and TensorFlow do, instead of `sqrt(v + eps)`.
- Per-sample and mini-batch training reshuffle the samples every epoch (`do_not_shuffle` restores
  the previous order).
- Weight decay is an L2 term for SGD, Momentum, RMSProp and Adam in every strategy. Batch RMSProp
  previously used a decoupled term. AdamW keeps decoupled decay.
- Cross-entropy loss requires a softmax or sigmoid output layer; other outputs are rejected
  instead of training with an incorrect gradient.
- Model files use format version 2 (per-layer dropout rates); version 1 files remain loadable.
- `load_spingalett()` returns `NULL` for truncated files.
- `TrainArgs.inputs` and `targets` are `const float *`. `NeuralNetwork`, `LayerArgs` and
  `TrainArgs` gained fields; code using designated initializers or positional arguments keeps
  compiling, but binaries must be rebuilt.
- Training is substantially faster: denormals are flushed to zero while training, the optimizers
  use shared AVX kernels in every strategy, tanh/sigmoid/softmax are vectorized, and OpenMP
  only parallelizes layers that are large enough (per-sample training up to 11x, OpenBLAS
  full-batch training +43%).

### Fixed
- Adam/AdamW bias correction restarted on every `train()` call and after loading a checkpoint.
- Truncated files were loaded as partially initialized networks; an unrelated earlier error made
  `load_spingalett()` fail.
- 32-bit overflow of weight counts in the serializer.
- Softmax output layers wider than 65,535 neurons looped forever.
- FP16 and BF16 conversion truncated instead of rounding to nearest even.
- Stale gradients after switching compute backends between `train()` calls.
- Invalid training mode, strategy or optimizer values were used as array indices.
- The compute-mode fallback warning was logged on every `forward()` call.
- The `Benchmark` example oversubscribed CPUs with a hard-coded thread count.

## [0.1.0]

Initial release.

[Unreleased]: https://github.com/pka-human/Spingalett/compare/v1.4.0...HEAD
[1.4.0]: https://github.com/pka-human/Spingalett/compare/v1.3.0...v1.4.0
[1.3.0]: https://github.com/pka-human/Spingalett/compare/v1.2.0...v1.3.0
[1.2.0]: https://github.com/pka-human/Spingalett/compare/v1.1.0...v1.2.0
[1.1.0]: https://github.com/pka-human/Spingalett/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/pka-human/Spingalett/compare/v0.14.0...v1.0.0
[0.14.0]: https://github.com/pka-human/Spingalett/compare/v0.13.1...v0.14.0
[0.13.1]: https://github.com/pka-human/Spingalett/compare/v0.13.0...v0.13.1
[0.13.0]: https://github.com/pka-human/Spingalett/compare/v0.12.0...v0.13.0
[0.12.0]: https://github.com/pka-human/Spingalett/compare/v0.11.0...v0.12.0
[0.11.0]: https://github.com/pka-human/Spingalett/compare/v0.10.0...v0.11.0
[0.10.0]: https://github.com/pka-human/Spingalett/compare/v0.9.0...v0.10.0
[0.9.0]: https://github.com/pka-human/Spingalett/compare/v0.8.0...v0.9.0
[0.8.0]: https://github.com/pka-human/Spingalett/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/pka-human/Spingalett/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/pka-human/Spingalett/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/pka-human/Spingalett/compare/v0.4.1...v0.5.0
[0.4.1]: https://github.com/pka-human/Spingalett/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/pka-human/Spingalett/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/pka-human/Spingalett/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/pka-human/Spingalett/compare/0a1dd16...v0.2.0
