# Language models

Transformers trained and run by the library, sessions and generation, tokenizers, Hugging Face checkpoints and GGUF files.

## Transformers and language models

A sequence is a row of cells: the input layer holds a window of token ids (as floats), an embedding
turns each into a vector, and the layers after it work on each position's channels, attention
mixing the positions. `spingalett_linear()` is the projection of each position (a 1 x 1
convolution), `spingalett_attention()` reads the queries, keys and values a linear layer packs
before it (the query heads' vectors, then the key heads', then the value heads'), and the output
layer gives each position the logits of every token of the vocabulary, which the sparse
cross-entropy compares with the next token. A GPT-2 small, and the changes that make a LLaMA:

```c
NeuralNetwork *net = new_spingalett(.loss_func = LOSS_SPARSE_CROSS_ENTROPY);
layer(.net = net, .neurons_amount = 1024);                          /* a window of 1024 tokens */
uint32_t e = embedding(.net = net, .vocabulary = 50257, .neurons_amount = 768,
                       .positions = true), x = e;                   /* learned positions, as GPT-2's */
for (int b = 0; b < 12; b++) {
    layer_norm(.net = net);
    linear(.net = net, .neurons_amount = 3 * 768);                  /* queries, keys and values */
    attention(.net = net, .heads = 12, .causal = true);
    uint32_t a = linear(.net = net, .neurons_amount = 768);
    x = add_layers(.net = net, .inputs = {x, a});
    layer_norm(.net = net);
    linear(.net = net, .neurons_amount = 4 * 768, .act_func = ACT_GELU_TANH);
    a = linear(.net = net, .neurons_amount = 768);
    x = add_layers(.net = net, .inputs = {x, a});
}
layer_norm(.net = net);
linear(.net = net, .neurons_amount = 50257, .tied = e);             /* each position's logits, with the
                                                                       embedding's table (tied, as GPT-2's) */

/* LLaMA: RMS normalization, grouped-query attention with rotary embeddings, a SwiGLU MLP */
uint32_t n = rms_norm(.net = net, .epsilon = 1e-5f);
linear(.net = net, .neurons_amount = (32 + 2 * 8) * 128);           /* 32 query heads, 8 key/value heads */
attention(.net = net, .heads = 32, .kv_heads = 8, .rope_theta = 500000.0f, .causal = true);
/* ... */
uint32_t g = linear(.net = net, .inputs = {n}, .neurons_amount = 14336, .act_func = ACT_SILU);
uint32_t u = linear(.net = net, .inputs = {n}, .neurons_amount = 14336);
multiply_layers(.net = net, .inputs = {g, u});
```

Language models train on files of token ids, as nanoGPT's and llm.c's preparation scripts write them,
read in windows by `spingalett_dataset_open_tokens()` (the file is mapped, not read), and
`spingalett_generate()` continues a prompt a token at a time:

```c
SpingalettTokenReaderOptions o = {.context = 1024, .stride = 512, .shuffle = true};
SpingalettDatasetReader *r = spingalett_dataset_open_tokens("train.bin", &o);
train(.net = net, .training_mode = MODE_GENERATOR_FUNCTION, .generator = spingalett_dataset_generator,
      .generator_data = r, .sample_count = spingalett_dataset_info(r).count, .epochs = 1,
      .training_strategy = STRATEGY_SMALL_BATCH, .batch_size = 16, .optimizer_type = OPTIMIZER_ADAMW,
      .learning_rate = 6e-4f, .weight_decay = 0.1f, .beta2 = 0.95f, .max_grad_norm = 1.0f);

uint32_t prompt[] = {15496, 11, 314}, tokens[200];
uint32_t made = spingalett_generate(.net = net, .prompt = prompt, .prompt_length = 3, .tokens = tokens,
                                    .count = 200, .temperature = 0.8f, .top_k = 40);
```

| Builder or call | Arguments | What it does |
|---|---|---|
| `spingalett_embedding()` | `vocabulary`, `neurons_amount` (the width), `positions` | Each token id's vector (ids outside the vocabulary give zeros); with `positions`, a learned vector per position added (the layer's biases) |
| `spingalett_linear()` | `neurons_amount`, `act_func`, `inputs`, `tied` | Each position's channels times a matrix, plus biases; with `tied`, an embedding's index, that embedding's table is the matrix (tied input and output embeddings: one table, its gradient the sum of both layers') |
| `spingalett_attention()` | `heads`, `kv_heads` (0: as many; fewer: grouped queries, 1: multi-query), `causal`, `rope_theta` (0: none), `ignore_padding`, `window` (0: none), `act_func` | Softmax attention of each query head over the keys of its group, `1 / sqrt(head size)` scaled, the keys of padding left out with `ignore_padding`, the keys `window` or more cells away with `window` (Mistral's sliding window: with `causal` a cell and the `window - 1` before it); outputs the heads' vectors side by side |
| `spingalett_rms_norm()` | `epsilon` (1e-5) | Each position's channels over their root mean square, times a gain (no biases) |
| `spingalett_multiply_layers()` | `inputs` | The element-wise product of layers of one shape |
| `spingalett_dataset_open_tokens()` | `context`, `token_bytes` (2 or 4), `stride`, `offset`, `shuffle` | Windows of a token file: inputs and, as targets, the tokens after them |
| `spingalett_generate()` | `prompt`, `count`, `temperature` (0: greedy), `top_k`, `top_p`, `seed`, `stop_tokens` | Tokens continuing the prompt |

A deployment model of a causal language model (its first layer an embedding of the tokens, its other
layers treating each position alone but for causal attention) runs a token at a time in a session:
each attention layer keeps the keys and values of the tokens a sequence has been fed, so that a new
token costs one row through every layer and its attention over those, not a pass over the window. A
session holds several sequences; the tokens of several fed in one call (a prompt, or a token of each
sequence) run through the layers together, so that sequences generated together share each product.
The logits are those `spingalett_model_predict()` gives each position of the whole window, bit for
bit, however the tokens come, in every precision (integer models too: the linear layers of a language
model quantize each token alone); sessions are part of the runtime. On the CPU `spingalett_generate()`
runs one, on the network's deployment model of single precision (kept in the network until its
parameters change), with the tokens it gave before: the CharGPT model (256 wide, 4 blocks, a window of
128) generates 3230 tokens a second on one thread of an Intel Core i7-12650H where predicting the
window for every token gave 72.7, and 3568 on six threads where it gave 121.

In the GPU modes a session runs on the GPU, its model a network there and the keys and values in the GPU's
memory. An integer model's linear layers then multiply its codes, kept on the GPU with their scales and
widened to floats as they are read (weight-only: 8, 4 or 2 bits read a weight a token; the activations in
the GPU's precision): the logits are those of the model's weights as floats to rounding, near the CPU's
integer session's but not its bits. A model of GPT-2 small's size (12 blocks of 768, a vocabulary of
50,257) generates 684 tokens a second in INT8 and 854 in INT4 on CUDA on an RTX 4050 Laptop GPU, where
the same model in BF16 gives 368 (four sequences together: 1,953 and 2,053).

A language model of a GGUF file runs on the GPU straight from its image, on CUDA and on Vulkan, in the runtime too: its
weights stay in their blocks (a 4B model in Q4_K_M takes 2.5 GB of the GPU, not 16), a block layer's products give the
CPU's bits, prompts run on the matrix units and tokens at what the memory's bandwidth allows. On CUDA on an RTX 4050 Laptop
GPU, against llama.cpp's Vulkan build on the same files, the GPU at 60 C or less before each run (`Examples/Runtime/LlmBench.c`
and `llama-bench -p 512 -n 128 -ngl 99`, tokens a second):

| Model | Prompt of 512 | llama.cpp | Generation | llama.cpp |
|---|---:|---:|---:|---:|
| Qwen3-0.6B Q8_0 | 16,447 | 12,120 | 222 | 207 |
| Qwen3-1.7B Q8_0 | 8,117 | 6,543 | 90.7 | 90.2 |
| Qwen3-4B Q4_K_M | 3,601 | 2,974 | 64.9 | 65.2 |
| Llama-3.2-1B Q4_K_M | 12,682 | 10,473 | 196 | 198 |
| Qwen2.5-0.5B Q4_K_M | 22,215 | 16,265 | 326 | 290 |

A model the GPU's memory does not hold splits (`SpingalettSessionOptions.gpu_memory` bounds what a session takes; 0 takes
what the device has free, less 128 MB). Its first layers keep their weights on the GPU, as many as fit up to the end of a
block; a token runs them there and the rest on all the CPU's threads, while a prompt (32 tokens or more) runs every layer but the
last on the GPU, the later layers' weights copied over the bus as their turn comes, so that it keeps most of its speed. Qwen3-4B Q4_K_M with
2 GB of the GPU processes a prompt at 2,465 tokens a second and generates 34.0, with 0.7 GB 1,502 and 19.2 (llama.cpp with 20 of its
37 layers on the GPU 1,490 and 30.9, with none 1,062 and 19.3). A split session's logits are the CPU's to rounding, and a prompt fed at once
equals its tokens fed one at a time to rounding as well, not to the bit: the two take different kernels for the later layers.

```c
SpingalettModel *model = spingalett_model_from_network(net, PRECISION_INT8);   /* or spingalett_model_load() */
SpingalettSession *s = spingalett_session_new(model, &(SpingalettSessionOptions){.sequences = 4});
const uint32_t *prompts[4] = {p0, p1, p2, p3};
uint32_t lengths[4] = {3, 5, 1, 2}, tokens[4 * 100], made[4];
spingalett_session_generate(.session = s, .sequences = 4, .prompts = prompts, .prompt_lengths = lengths,
                            .tokens = tokens, .made = made, .count = 100, .temperature = 0.8f, .top_k = 40,
                            .seed = 1);                             /* sequence b's tokens from tokens + 100 b */

float logits[50257];                                                /* or a token at a time */
spingalett_session_rewind(s, 0, 0);
spingalett_session_feed(s, &(SpingalettSessionFeed){.sequence = 0, .tokens = p0, .count = 3, .logits = logits}, 1);
spingalett_session_free(s);
```

| Call | Arguments | What it does |
|---|---|---|
| `spingalett_session_new()` | a model; `max_length` (the model's window at most), `sequences` | A session, or NULL with the reason for a model that is no causal language model |
| `spingalett_session_feed()` | per sequence: `tokens`, `count`, `logits`, `all` | Appends tokens to several sequences in one pass; the logits of the token after the last (or after each) |
| `spingalett_session_rewind()`, `spingalett_session_copy()`, `spingalett_session_length()` | a sequence, a length; two sequences | Truncates a sequence; copies one (a prompt fed once for several continuations) |
| `spingalett_session_generate()` | `sequences`, `prompts`, `count`, `temperature`, `top_k`, `top_p`, `seed` (sequence b: seed + b), `stop_tokens` | Sequences continued together, each drawing as `spingalett_generate()` draws |

- The rotary embeddings rotate the halves of each head's queries and keys (the `rotate_half` form
  of GPT-NeoX and of Hugging Face's LLaMA weights). GELU comes in its erf form (`ACT_GELU`) and
  GPT-2's tanh form (`ACT_GELU_TANH`), SiLU as `ACT_SILU`.
- The sparse cross-entropy takes one class index a position (`spingalett_target_size()` per sample);
  an index outside the classes leaves the position out of the loss. Label smoothing applies.
- Sequences of different lengths train in one batch padded with the token -1 (inputs and targets): an
  embedding reads it as zeros, the loss leaves its positions out (a sample's loss is the mean over its real
  tokens), and attention with `.ignore_padding = true` leaves its keys out (their tokens in the network's
  input, which the keys' cells come from through an embedding). The gradient of a padded batch is then the
  mean of the sequences' own, each trained alone in a network of its length, up to the order of the sums,
  on the CPU and the GPU. Causal attention over right padding never reaches it anyway; non-causal attention
  and left padding need the option. Batch normalization takes its statistics over padded cells too.
- Attention never stores the scores: the CPU computes a block of queries at a time, the backward
  pass recomputes the probabilities from each query's log-sum-exp. On the GPU it runs in tiles
  (register-tiled kernels in single precision, flash attention on the matrix units in bfloat16, on
  CUDA and Vulkan alike), deterministic like everything else.
- A sliding window costs what it reads: the CPU and the GPU's kernels of single precision skip the blocks of
  keys outside every query's window (in bfloat16 on CUDA such layers run on those kernels, not on flash
  attention).
- Encoder-decoders take the source's tokens and then the target's in one input: an embedding reads each part
  (`.token_offset`, `.token_count`; the decoder's names the input layer, `.inputs = {0}, .input_count = 1`),
  and cross-attention, an attention of two inputs, takes the decoder's queries over the encoder's keys and
  values (a linear layer of `2 kv_heads x head size` outputs reading the encoder), ignoring the source's
  padding with `.ignore_padding`:

  ```c
  uint32_t src = embedding(.net = net, .vocabulary = V, .neurons_amount = 512, .positions = true,
                           .token_count = 64);                 /* the first 64 tokens: the source */
  /* ... the encoder's blocks, its output e ... */
  uint32_t kv = linear(.net = net, .inputs = {e}, .neurons_amount = 2 * 512);
  uint32_t tgt = embedding(.net = net, .inputs = {0}, .input_count = 1, .vocabulary = V,
                           .neurons_amount = 512, .positions = true, .token_offset = 64);
  /* ... the decoder's causal self-attention, its queries q = linear(512) ... */
  attention(.net = net, .inputs = {q, kv}, .heads = 8, .ignore_padding = true);
  ```
- Networks of these layers save to `.slett` files (format 8; 9 with a tied head, its table stored once,
  padding ignored, a sliding window, cross-attention or embeddings of part of their input),
  become deployment models in every precision (the linear maps in INT8 down to INT2) and run on the
  runtime and the engine.
- `Examples/CharGPT.c` trains a character-level GPT on any text file and samples it after every
  epoch. In Python, `gpt2_from_state_dict()` and `llama_from_state_dict()` make networks of GPT-2
  checkpoints (Hugging Face's or nanoGPT's) and LLaMA-like ones (LLaMA, Mistral, Qwen2), the head tied
  to the embedding when the checkpoint ties them.

## Tokenizers

`spingalett_tokenizer_load()` reads the tokenizer of a language model: a Hugging Face
`tokenizer.json` file, a model directory holding one (its `tokenizer_config.json` then names the
BOS and EOS tokens) or a GGUF file (see below). `spingalett_tokenize()` turns UTF-8 text into token ids and
`spingalett_detokenize()` ids back into text, id for id and byte for byte as Hugging Face's
`tokenizers` library does:

```c
SpingalettTokenizer *tok = spingalett_tokenizer_load("models/SmolLM2-135M");     /* NULL on error */
uint32_t ids[256];
int64_t n = spingalett_tokenize(.tokenizer = tok, .text = "<|im_start|>user\nHello!<|im_end|>",
                                .tokens = ids, .capacity = 256, .special = true);
char text[1024];
spingalett_detokenize(.tokenizer = tok, .tokens = ids, .count = (size_t)n, .text = text, .capacity = sizeof text);
uint32_t end = spingalett_tokenizer_id(tok, "<|im_end|>");                /* a chat format's stop token */
SpingalettTokenizerInfo info = spingalett_tokenizer_info(tok);            /* vocabulary, BOS, EOS, ... */
spingalett_tokenizer_free(tok);
```

| Family | Models | What the file holds |
|---|---|---|
| Byte-level BPE | GPT-2, SmolLM | ByteLevel pre-tokenizer (GPT-2's expression), digits apart |
| tiktoken's encodings | LLaMA 3 (cl100k), Qwen2 (cl100k with single digits, NFC), GPT-4o-like (o200k) | a Split by the expression, then ByteLevel; ignore_merges |
| SentencePiece's BPE | LLaMA 2, TinyLlama, Mistral, Phi-3 | Prepend and Replace of `▁` (or Metaspace), byte fallback to `<0xXX>` tokens |

- The steps are those of the `tokenizers` library: added tokens found first (leftmost, longest,
  with their `lstrip`, `rstrip` and `single_word` rules), the normalizers (NFC, Prepend, Replace,
  Strip), the pre-tokenizers (ByteLevel, Split, Metaspace, Digits), the BPE merges by rank, the
  template of the post-processor (`.add_special`: LLaMA's BOS), and in decoding ByteLevel,
  ByteFallback, Replace, Fuse, Strip and Metaspace. Merges go through a heap of candidate pairs, so a
  piece of n characters takes n log n steps; a SentencePiece text, which is one piece, is merged word
  by word where the vocabulary shows that no merge joins a word to the `▁` after it (the same
  tokens, without a heap of the whole text). Other models (WordPiece, Unigram), normalizers and
  expressions are refused with an error that names them.
- The pre-tokenizers' regular expressions are matched by hand, not by a regular expression library:
  GPT-2's, cl100k's (with up to three or single digits) and o200k's (with tekken's variant) are
  recognized by their text and matched with the semantics of the Oniguruma expressions the
  `tokenizers` library runs, letters, numbers and marks from the Unicode tables of
  `Src/Spingalett.Unicode.h` (written by `Src/make_unicode.py`).
- Special tokens written in a text become their ids only with `.special = true`; by default they are
  text like any other, so that text from users cannot forge a chat format's control tokens. Added
  tokens that are not special are always found. `.skip_special` leaves special tokens out of
  decoded text; bytes that are no UTF-8 (a character split between tokens) decode to U+FFFD.
- Both calls return the full count (tokens, or bytes of text) also when the array is too small,
  writing what fits: call again with room for all. Input that is not UTF-8 is an error.
- On the tokenizers of GPT-2, LLaMA 3.2, Qwen2.5, SmolLM2, TinyLlama, Mistral 7B (v0.1 and v0.3) and
  Phi-3, the ids and decoded texts equal those of `tokenizers` 0.23 on texts of every script, emoji,
  combining marks, whitespace runs, digits and special tokens. On an i7-12650H 0.9 MB of prose and
  code encode in 0.055 s with GPT-2's tokenizer (0.30 s in `tokenizers` 0.23), 0.030 s with LLaMA
  3's (0.25 s), 0.075 s with Qwen2.5's (0.27 s) and 0.067 s with TinyLlama's (0.31 s), medians of
  seven runs on one thread.

## Pretrained language models

`spingalett_load_pretrained()` makes a network of a Hugging Face model directory: `config.json` and
the weights in `model.safetensors`, or in the shards `model.safetensors.index.json` lists. The model
types are `llama`, `mistral`, `qwen2` and `gpt2`; the network takes a window of `.context` token ids
and gives every position's logits, so it generates, trains further (fine-tuning), saves and deploys
like any other:

```c
NeuralNetwork *net = spingalett_load_pretrained(.path = "models/SmolLM2-135M", .context = 512);
SpingalettTokenizer *tok = spingalett_tokenizer_load("models/SmolLM2-135M");
uint32_t prompt[512], tokens[64];
int64_t n = spingalett_tokenize(.tokenizer = tok, .text = "The history of computing", .tokens = prompt, .capacity = 512);
uint32_t made = spingalett_generate(.net = net, .prompt = prompt, .prompt_length = (uint32_t)n, .tokens = tokens,
                                    .count = 64, .temperature = 0.7f, .top_p = 0.9f,
                                    .stop_tokens = &(uint32_t){spingalett_tokenizer_info(tok).eos}, .stop_count = 1);
```

- LLaMA, Mistral and Qwen2 become RMS normalizations, grouped-query attention with rotary
  embeddings (the queries', keys' and values' projections in one linear layer, with Qwen2's biases;
  a `head_dim` of its own), and SwiGLU as a SiLU layer times a linear one; GPT-2 layer
  normalizations, learned positions, its GELU (`gelu_new`: the tanh form) and its Conv1D weights,
  transposed. Without `lm_head.weight` (tied embeddings) the head is tied to the embedding (`.tied`:
  one table, trained by both layers). These are the networks of the Python bindings' `llama_from_state_dict()` and `gpt2_from_state_dict()`,
  which give the same logits bit for bit.
- Weights in F32, F16 or BF16 are decoded straight from the mapped files into the network, whose
  arrays are reserved at their final sizes first, so a model is never copied: SmolLM2-135M (BF16)
  loads in 0.17 s on an i7-12650H.
- `Examples/Pretrained.c` continues a prompt with any of these models, printing each token as it
  comes: `Bin/Pretrained models/SmolLM2-135M "The three primary colors are" 64`.
- `.context` 0 takes the model's positions (GPT-2's all, LLaMA's up to 2048). Mistral's sliding
  window is the attention layers' `window`. Scaled rotary embeddings (LLaMA 3.1's
  `llama3`, YaRN) are refused, as is anything else the layers do not compute, with an error that
  names it.
- On SmolLM2-135M the logits are those of `transformers` within 2.4e-4 (of logits up to 42), on
  Qwen2.5-0.5B-Instruct within 4e-4 (of 24), and the greedy continuations are the same; on Hugging
  Face's tiny random LLaMA and GPT-2 within 2e-7. The
  `pretrained` test group checks tiny LLaMA (BF16, sharded), Qwen2 (F16, biases, tied), Mistral
  and GPT-2 checkpoints (`Tests/make_pretrained_tests.py`) against PyTorch, within 2e-6.

GGUF files, llama.cpp's, load the same way: `.path` names the file, `spingalett_tokenizer_load()` reads
its vocabulary, and the network is that of its `general.architecture` (`llama`, which covers Mistral,
`qwen2` or `gpt2`) with the hyper-parameters of its metadata:

```c
NeuralNetwork *net = spingalett_load_pretrained(.path = "SmolLM2-135M.Q4_K_M.gguf", .context = 512);
SpingalettTokenizer *tok = spingalett_tokenizer_load("SmolLM2-135M.Q4_K_M.gguf");
```

- Quantized weights become floats as llama.cpp's `dequantize_row_*()` computes them: F32, F16, BF16,
  Q8_0, Q4_0, Q4_1, Q5_0, Q5_1, the k-quants Q2_K to Q6_K, and IQ4_NL and IQ4_XS, which quantized
  files of widths that are no multiple of 256 hold where k-quants cannot be. The network keeps
  floats (quantize it again with `spingalett_model_from_network()` to deploy it). The other i-quants
  are refused, naming the tensor and its type. On quantized files of SmolLM2-135M, TinyStories and
  Qwen2.5-0.5B (tensors of Q8_0, Q4_0, Q4_1, Q5_0, Q5_1, Q3_K, Q4_K, Q5_K, Q6_K and IQ4_NL), and on
  random blocks of every type above, the weights are those of the `gguf` Python package's
  dequantization, bit for bit (the dequantization is compiled in IEEE order, so they are the same on
  every platform); a Q4_K_M file of SmolLM2-135M loads in 0.18 s on an i7-12650H.
- llama.cpp's converter stores LLaMA's queries and keys with the rows of each head's two halves
  interleaved, for its rotation of adjacent pairs; they are put back in Hugging Face's order, which
  the library's rotation of halves expects (Qwen2's are stored in that order already).
- A vocabulary of the `gpt2` model is a byte-level BPE of its merges, cut by the pre-tokenizer
  `tokenizer.ggml.pre` names (`llama-bpe`, `qwen2`, `smollm`, `gpt-2`, ...); one of the `llama` model
  is SentencePiece's BPE of its scores, with byte fallback and the space before each text and after
  each special token, as llama.cpp and the `sentencepiece` library encode it. Control tokens are
  special, user-defined tokens are found in any text, and `tokenizer.ggml.add_bos_token` gives the
  BOS of `.add_special`. The GGUF vocabularies of SmolLM2, Qwen2.5 and TinyStories give the ids of
  the `tokenizer.json` files they were converted from (TinyStories' finds its special tokens in the
  normalized text, which llama.cpp does not: with `.special` those differ).

## GGUF models as they are stored

`spingalett_model_open()` (and `spingalett_model_load()`, which takes the defaults) makes a deployment model
of a GGUF file directly: its tensors go into the model's `.slett` image as the file stores them, so a 1B model
of 800 MB takes 800 MB and loads in well under a second, and the quantized types are precisions the kernels
compute with (format 10's block precisions, below). Sessions then run it a token at a time:

```c
SpingalettModel *model = spingalett_model_open(.path = "Llama-3.2-1B-Instruct-Q4_K_M.gguf", .context = 4096);
SpingalettTokenizer *tok = spingalett_tokenizer_load("Llama-3.2-1B-Instruct-Q4_K_M.gguf");
SpingalettSession *s = spingalett_session_new(model, NULL);
float *logits = malloc(spingalett_session_vocabulary(s) * sizeof(float));
spingalett_session_feed(s, &(SpingalettSessionFeed){.tokens = prompt, .count = n, .logits = logits}, 1);
/* pick the next token of logits, feed it, and so on (Examples/Pretrained.c) */
```

- The architectures are `llama` (LLaMA 1 to 3.x, with LLaMA 3's frequencies of `rope_freqs.weight`, and linear
  scaling; Mistral), `qwen2`, `qwen3` (queries and keys normalized in each head), `gemma3` (embeddings scaled
  by the square root of the width, normalized outputs of the attention and the MLP, five layers of six attending
  through a window with a rotary base of their own) and `gpt2`. Their attention is format 10's: each head's
  queries and keys normalized, rotary frequencies of the layer's own, part of a head rotated, a scale of the
  scores. A queries', keys' or values' tensor of another type than the others (Q4_K_M keeps `attn_v` in Q6_K in
  half the layers) becomes a layer of its own, concatenated.
- `.context`, the model's window, is the most tokens a session's sequence holds (0: the model's positions, 4096
  at most); sessions allocate their keys and values for it. `ModelTool convert model.gguf model.slett --context
  N` writes the model's image as a `.slett` file, which the engine, the runtime and every binding open.
- Checked against llama.cpp on Qwen2.5-0.5B-Instruct (Q4_0, Q4_K_M, Q8_0), Llama-3.2-1B-Instruct (Q4_K_M),
  Qwen3-0.6B (Q8_0) and Gemma-3-1B-it (Q4_K_M): the same most likely token at every position of a prompt,
  logits a few percent apart, as both are from the logits of float weights (each quantizes the activations
  its own way); and in the `pretrained` group against PyTorch and transformers on tiny LLaMA, LLaMA 3, Qwen2,
  Qwen3, Gemma 3 and GPT-2 checkpoints converted as llama.cpp's converter writes them.
- The runtime opens GGUF files too; Python's `Model.load(path, context=...)` and PHP's `Model::load()` take the
  window.

**Block precisions.** `SPINGALETT_PRECISION_Q8_0`, `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q2_K` to `Q6_K`, `IQ4_NL` and
`IQ4_XS` are GGUF's types as precisions of dense layers, linear layers and embeddings, each row stored as
llama.cpp stores it (docs/ModelFormat.md, **Block precisions**). A network saved in one (or made a model in one)
is quantized as llama.cpp's quantizers do without an importance matrix, byte for byte. Such a layer quantizes its
input to 8 bits in blocks of the type's size and sums each block's sub-blocks in integers, scaled by their integer
scales, the blocks' terms then added in float in their order: the engine, batched prediction, sessions and the
runtime give the same bits on any number of threads. On x86-64 a token's rows go eight at a time against its
codes, the next rows prefetched; a prompt's tokens go in groups of eight, the codes of each row broadcast against
a group's (dpbusd with VNNI); packages built for a baseline choose these kernels for AVX2, AVX-VNNI or AVX-512
VNNI at run time. The engine computes them in C on any processor, microcontrollers included.
