/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/*
 * Spingalett runtime.
 *
 * What a program needs to run trained models: .slett files loaded as deployment models
 * (SpingalettModel, Spingalett.Inference.h), batched prediction on all cores with the library's
 * kernels (those for AVX2, AVX-512, VNNI and the Arm dot product chosen at run time), evaluation,
 * errors, logging and the thread settings. The runtime library (libspingalett-runtime; the CMake
 * target Spingalett::runtime, the pkg-config package spingalett-runtime) has these functions and no
 * others: no training, data sets, importers or GPU backend. They are the full library's functions,
 * which Spingalett.h declares through this header, so a program written for it runs with either
 * library and gets the same results from both.
 */
#ifndef SPINGALETT_RUNTIME_H
#define SPINGALETT_RUNTIME_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include "Spingalett.Config.h"
#include "Spingalett.Inference.h"   /* SPINGALETT_API, shared enums, error codes, the inference engine */

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    SPINGALETT_LOG_DEBUG,
    SPINGALETT_LOG_INFO,
    SPINGALETT_LOG_WARNING,
    SPINGALETT_LOG_ERROR,
    SPINGALETT_LOG_NONE
} SpingalettLogLevel;

typedef void (*SpingalettLogCallback)(SpingalettLogLevel level, const char *message);

/* Where the library computes. Deployment models (spingalett_model_predict()) run on the CPU: on all its
   threads with SPINGALETT_COMPUTE_OPENMP, on one otherwise; in the full library's GPU modes those of
   float weights (FP32, FP16, BF16) and sessions run on the GPU, and so do integer ones whose every layer the
   GPU computes as the CPU does (their outputs the CPU's to the bit), the others on all the CPU's threads. The
   runtime has no OpenBLAS: there SPINGALETT_COMPUTE_OPENBLAS runs single-threaded, with a warning, as in a full
   library built without it; nor a GPU. */
typedef enum {
    SPINGALETT_COMPUTE_SINGLE_THREADED,
    SPINGALETT_COMPUTE_OPENMP,
    SPINGALETT_COMPUTE_OPENBLAS,
    SPINGALETT_COMPUTE_CUDA,                   /* an NVIDIA GPU through CUDA (spingalett_cuda_device()); the CPU parts
                                       of training run as with SPINGALETT_COMPUTE_OPENMP */
    SPINGALETT_COMPUTE_VULKAN,                 /* the GPU through Vulkan compute (spingalett_gpu_device()); the CPU
                                       parts of training run as with SPINGALETT_COMPUTE_OPENMP */
    SPINGALETT_COMPUTE_COUNT
} SpingalettComputeMode;

/* Result of spingalett_model_evaluate(), spingalett_evaluate() and the per-epoch validation pass. */
typedef struct {
    float loss;                     /* mean over samples of the network's loss (see spingalett_evaluate()) */
    float accuracy;                 /* fraction of samples whose output argmax matches the target's
                                       argmax; with a single output, both on the same side of 0.5; with
                                       SPINGALETT_LOSS_SPARSE_CROSS_ENTROPY the mean over samples of the
                                       fraction of their cells with targets whose argmax is the target */
    uint64_t reserved[SPINGALETT_RESERVED];
} SpingalettEvalMetrics;

/* Library version (the header's SPINGALETT_VERSION_* macros describe the headers in use). */
SPINGALETT_API const char *spingalett_version(void);

/* Errors. A function that fails returns false, NULL, NaN (losses and metrics) or, in the full library,
   SPINGALETT_NO_LAYER or a report whose status is SPINGALETT_TRAIN_FAILED, and sets the calling thread's
   error: its code
   (SPINGALETT_ERR_*) and a message saying what failed. A call that succeeds leaves the error as it was,
   so a program checks the result, then the error; spingalett_clear_error() resets it. The engine's
   functions (Spingalett.Inference.h), which keep no state, return the code instead. */
SPINGALETT_API int spingalett_last_error_code(void);
SPINGALETT_API const char *spingalett_last_error_message(void);
SPINGALETT_API void spingalett_clear_error(void);

/* Instruction set of the matrix-multiplication kernels that training and batched inference use:
   "AVX-512", "AVX2", "AVX", "SSE2", "NEON" or "C". x86-64 libraries built without
   SPINGALETT_NATIVE_ARCH (such as the release binaries) choose AVX-512 or AVX2 kernels at run time
   when the processor has them. */
SPINGALETT_API const char *spingalett_cpu_kernels(void);

SPINGALETT_API SpingalettComputeMode spingalett_get_compute_mode(void);
/* Returns false (the error set) for a value of no mode. */
SPINGALETT_API bool spingalett_set_compute_mode(SpingalettComputeMode mode);
/* Threads of SPINGALETT_COMPUTE_OPENMP; 0, the default, leaves OpenMP's (one per core). */
SPINGALETT_API unsigned spingalett_get_num_threads(void);
SPINGALETT_API void spingalett_set_num_threads(unsigned n);

/* Messages go to the callback, or to stdout (stderr from SPINGALETT_LOG_WARNING on) without one;
   those below the level are dropped. */
SPINGALETT_API void spingalett_set_log_callback(SpingalettLogCallback cb);
SPINGALETT_API void spingalett_set_log_level(SpingalettLogLevel level);

/* false drops the messages below SPINGALETT_LOG_WARNING as well (on by default). */
SPINGALETT_API void spingalett_set_verbose(bool enabled);
SPINGALETT_API bool spingalett_get_verbose(void);

/*
 * Deployment. A SpingalettModel (Spingalett.Inference.h) is a read-only network that computes in the
 * precision its weights are stored in: INT8, INT4 and INT2 layers use integer kernels. The functions
 * below create models that own their image; release them with spingalett_model_free. Models made by
 * spingalett_model_init over a caller's image need no release.
 */

/* Reads a .slett file. Images (format versions 3 to SPINGALETT_FORMAT_VERSION) are used as stored;
   the full library converts files of versions 1 and 2 in the precision they were saved in, which
   the runtime reports as SPINGALETT_ERR_FORMAT_VERSION. A GGUF file (llama.cpp's) becomes the model of
   its language model, as spingalett_model_open() with the defaults makes it. */
SPINGALETT_API SpingalettModel *spingalett_model_load(const char *path);

typedef struct {
    const char *path;               /* a .slett file, or a language model's GGUF file */
    uint32_t context;               /* GGUF: the model's window, the tokens a sequence holds at most (0: the
                                       model's positions, 4096 at most) */
    uint64_t reserved[SPINGALETT_RESERVED];
} SpingalettModelOpenArgs;

/* Opens a model file. A GGUF file of a language model (general.architecture llama, qwen2, qwen3 or gpt2:
   LLaMA 1 to 3.x, Mistral, Qwen2 and 2.5, Qwen3, GPT-2) becomes its network as a deployment model: its tensors
   used as the file stores them, the quantized types as the block precisions of format 10 (Q8_0, Q4_0, Q4_1,
   Q5_0, Q5_1, Q2_K to Q6_K, IQ4_NL, IQ4_XS; other types are refused), F16, BF16 and F32 as FP16, BFLOAT16 and
   FLOAT32; LLaMA's queries and keys taken back from llama.cpp's order of their rows, LLaMA 3's scaled rotary
   embeddings as their frequencies, Qwen3's normalized queries and keys in its attention. The model takes a
   window of `context` token ids and gives every position's logits; sessions (spingalett_session_new()) run
   it a token at a time. NULL with the error set. */
#define spingalett_model_open(...) spingalett_model_open_args((SpingalettModelOpenArgs){__VA_ARGS__})
SPINGALETT_API SpingalettModel *spingalett_model_open_args(SpingalettModelOpenArgs args);
/* Same, from an image in memory, which is copied. */
SPINGALETT_API SpingalettModel *spingalett_model_from_memory(const void *data, size_t size);
SPINGALETT_API void spingalett_model_free(SpingalettModel *model);
/* Batched inference: inputs [count x input_size] give outputs [count x output_size]. Uses all
   threads in SPINGALETT_COMPUTE_OPENMP mode; in the full library's GPU modes, a model that owns its image
   (spingalett_model_load(), _from_memory(), _from_network()) runs on the GPU, made there by the first call and
   kept, its calls one at a time: one of float weights as a network, the outputs the CPU's to rounding; one of
   integer weights (INT8, INT4, INT2) as the CPU computes it, the outputs spingalett_model_run()'s bit for bit,
   where the GPU computes every layer so (integer products, convolutions of every kind, pooling, additions,
   products and concatenations of layers, batch normalization, upsampling, the activations none, ReLU and leaky
   ReLU inside the network and any after its last layer, which the CPU applies); a model with others (GELU,
   SiLU, sigmoid or tanh inside the network, normalizations of layers, attention) runs on the CPU, the GPU's
   memory too small or the device lost as well. Returns false on error. */
SPINGALETT_API bool spingalett_model_predict(const SpingalettModel *model, const float *inputs, uint32_t count,
                                             float *outputs);
/* Mean loss (the model's loss function, as spingalett_evaluate() computes it) and accuracy over a data set. */
SPINGALETT_API SpingalettEvalMetrics spingalett_model_evaluate(const SpingalettModel *model, const float *inputs,
                                                     const float *targets, uint32_t count);

/*
 * Sessions: a causal language model run a token at a time, with a key-value cache. The model's input is a
 * window of token ids read by an embedding (its first layer), its output each position's logits (the
 * vocabulary: the output's channels), and its layers treat each position alone (linear layers: 1 x 1
 * convolutions of one group; normalizations; additions, products and concatenations of layers) but for
 * causal attention, whose keys (rotated, with rotary embeddings) and values the session keeps for every
 * token it has been fed. A token fed then costs one row through every layer and its queries' attention
 * over the tokens before it, instead of a pass over the whole window: a prompt fed at once runs its
 * tokens through each layer together, and so do the tokens of several sequences fed in one call (batched
 * generation). A session holds `sequences` sequences of at most max_length tokens (the model's window at
 * most, positions 0 to max_length - 1), each with a cache of its own.
 *
 * The logits of a position are those spingalett_model_predict() gives it for the window holding the
 * sequence's tokens: bit for bit for integer models, and for float ones too where batched prediction
 * multiplies with the library's kernels (every mode but SPINGALETT_COMPUTE_OPENBLAS, which takes
 * OpenBLAS's order: equal to rounding); however the tokens were fed, alone or with others. A session runs
 * on the CPU, on the threads of the compute mode, or in the full library's GPU modes on the GPU (its model
 * a network there, its products in the GPU's precision, the keys and values in its memory: the logits the
 * CPU's to rounding, still the same bits however the tokens were fed), and is used from one thread at a
 * time; any number of sessions may run on one model at once, from any threads. The model must outlive its
 * sessions. On the GPU the linear layers of an integer model (INT8, INT4, INT2) multiply its codes, kept in
 * the GPU's memory with their scales and widened as they are read (each weight's 8, 4 or 2 bits read once a
 * step), the activations in the GPU's precision and quantized nowhere: the logits those of the model's weights
 * as floats to rounding, near the CPU's integer session's but not its bits (which quantizes each token's
 * activations, and attention's exponentials are libm's).
 */
typedef struct SpingalettSession SpingalettSession;
typedef struct SpingalettTokenizer SpingalettTokenizer;

typedef struct {
    uint32_t max_length;            /* tokens a sequence holds at most: the model's window at most; 0 for it */
    uint32_t sequences;             /* sequences the session holds (0: 1) */
    uint64_t reserved[SPINGALETT_RESERVED];
} SpingalettSessionOptions;

/* Opens a session on a causal language model (options NULL: the defaults). NULL with the error set for
   other models (the message says what the session cannot run), or when memory runs out. */
SPINGALETT_API SpingalettSession *spingalett_session_new(const SpingalettModel *model,
                                                         const SpingalettSessionOptions *options);
SPINGALETT_API void spingalett_session_free(SpingalettSession *session);
/* Logits a position has: the size of the vocabulary. */
SPINGALETT_API uint32_t spingalett_session_vocabulary(const SpingalettSession *session);
/* Tokens a sequence holds (0 for a sequence out of range). */
SPINGALETT_API uint32_t spingalett_session_length(const SpingalettSession *session, uint32_t sequence);

/* Tokens for one sequence of a session (spingalett_session_feed()). */
typedef struct {
    uint32_t sequence;              /* which of the session's sequences */
    const uint32_t *tokens;         /* [count] the tokens after those the sequence holds */
    uint32_t count;
    float *logits;                  /* [vocabulary]: the logits of the token after the last of them; with all,
                                       [count x vocabulary]: of the token after each of them; NULL: none */
    bool all;
    uint64_t reserved[SPINGALETT_RESERVED];
} SpingalettSessionFeed;

/* Appends tokens to `count` sequences of the session (each named once), the rows of all of them computed
   together, and writes the logits asked for. A token outside the vocabulary embeds as zeros, as in
   spingalett_model_predict(). Returns false (the sequences unchanged) when a sequence would hold more than
   max_length tokens, a feed asks for logits without tokens, or memory runs out. */
SPINGALETT_API bool spingalett_session_feed(SpingalettSession *session, const SpingalettSessionFeed *feeds,
                                            uint32_t count);
/* Truncates a sequence to its first `length` tokens (no more than it holds): the next token fed goes to
   position `length`. 0 empties it. */
SPINGALETT_API bool spingalett_session_rewind(SpingalettSession *session, uint32_t sequence, uint32_t length);
/* Makes sequence `to` a copy of sequence `from` (its tokens and their keys and values): a prompt fed once
   serves several continuations. */
SPINGALETT_API bool spingalett_session_copy(SpingalettSession *session, uint32_t from, uint32_t to);

/* A token drawn for sequence `sequence` (spingalett_session_generate(), spingalett_generate()). */
typedef bool (*SpingalettTokenCallback)(void *user, uint32_t sequence, uint32_t token);

/* More of llama.cpp's ways of drawing tokens, for spingalett_session_generate() and spingalett_generate(): the
   penalties apply to the logits before the draw (and before the most likely token is taken at temperature 0), in
   llama.cpp's way; min_p after the temperature, top_k and top_p, as transformers applies it. */
typedef struct {
    float min_p;                    /* draws only among tokens at least min_p times as likely as the most likely (0:
                                       all) */
    float repetition_penalty;       /* llama.cpp's repeat_penalty: the logits of the tokens of the window divided by it
                                       (multiplied when they are not positive); 0 or 1: none */
    float frequency_penalty;        /* subtracted from their logits for each time they occur in the window */
    float presence_penalty;         /* subtracted once from the logits of those that occur in it */
    uint32_t penalty_window;        /* the last tokens the penalties look at, of the prompt and those generated (0:
                                       64, llama.cpp's; UINT32_MAX: all) */
    const SpingalettTokenizer *tokenizer;   /* decodes the tokens for stop_strings */
    const char *const *stop_strings;        /* [stop_string_count] texts that end a sequence's generation once the
                                               text of the tokens it generated holds one (the token that completes it
                                               is its last) */
    uint32_t stop_string_count;
    SpingalettTokenCallback on_token;       /* called with each token as it is drawn, in the order of the draws (a
                                               stream of text); false ends that sequence's generation (the token is
                                               its last) */
    void *user;                     /* on_token's first argument */
    uint64_t reserved[SPINGALETT_RESERVED - 2];
} SpingalettSampling;

typedef struct {
    SpingalettSession *session;
    uint32_t sequences;             /* sequences 0 .. sequences - 1 of the session generate (0: 1) */
    const uint32_t *const *prompts; /* [sequences] tokens fed to each first, at least one each, after what
                                       it holds */
    const uint32_t *prompt_lengths; /* [sequences] */
    uint32_t *tokens;               /* [sequences x count] the tokens generated: sequence b's from b count on */
    uint32_t *made;                 /* [sequences] how many each generated (NULL: not reported) */
    uint32_t count;                 /* tokens each generates at most */
    float temperature;              /* the logits divided by it before the softmax; 0: the most likely token
                                       each step (the first of equals), no draws */
    uint32_t top_k;                 /* draws among the top_k most likely tokens only (0: all) */
    float top_p;                    /* and among the most likely whose probabilities add up to top_p at least
                                       (0 or 1: all) */
    uint64_t seed;                  /* sequence b draws from a generator seeded with seed + b */
    const uint32_t *stop_tokens;    /* [stop_count] tokens that end a sequence's generation once generated */
    uint32_t stop_count;
    const SpingalettSampling *sampling;     /* min-p, penalties of repeated tokens, stop strings (NULL: none) */
    uint64_t reserved[SPINGALETT_RESERVED - 1];
} SpingalettSessionGenerateArgs;

/* Continues several sequences together, a token each per step, the steps' rows computed at once, each
   sequence drawing as spingalett_generate() draws (the same tokens for the same logits and seed, whatever
   other sequences run with it). A sequence stops after count tokens, after a stop token (written as its
   last), or after the token that follows a full sequence (max_length tokens). Afterwards each holds its
   prompt and the tokens generated, but the token after a full sequence. Returns how many tokens the
   sequences wrote together, 0 on error. */
#define spingalett_session_generate(...) spingalett_session_generate_args((SpingalettSessionGenerateArgs){__VA_ARGS__})
SPINGALETT_API uint32_t spingalett_session_generate_args(SpingalettSessionGenerateArgs args);

/*
 * Tokenizers of language models: the byte-pair encodings of Hugging Face's tokenizer.json files (GPT-2's
 * byte-level BPE and tiktoken's encodings as LLaMA 3, Qwen2 and SmolLM use them; SentencePiece's BPE with
 * byte fallback as LLaMA 2, Mistral and Phi-3 use it) and of GGUF files' vocabularies (the BPE of merges
 * of their "gpt2" model, by the pre-tokenizer they name; SentencePiece's BPE of scores of their "llama"
 * model, as llama.cpp and sentencepiece encode). A text is encoded as Hugging Face's tokenizers
 * library encodes it: the added tokens found in it, the rest normalized (NFC, prepended and replaced
 * strings), cut into pieces by the pre-tokenizers (byte-level, the regular expressions of GPT-2, cl100k
 * and o200k, digits, Metaspace, splits on strings) and each piece merged pair by pair in the order of the
 * merges, in time n log n of the piece's length. Decoding reverses it (byte-level, byte fallback,
 * Metaspace, replacements, stripping). Other models (WordPiece, Unigram) and patterns are refused, naming
 * what is missing. A tokenizer is read-only once loaded: threads may share it.
 */

/* An id no token has. */
#define SPINGALETT_NO_TOKEN UINT32_MAX

typedef struct {
    uint32_t vocabulary;            /* token ids: the largest one + 1 (the model's tokens and the added ones) */
    uint32_t added;                 /* added tokens (special ones, such as "<|endoftext|>", and others) */
    uint32_t bos, eos;              /* the tokens beginning and ending a text (SPINGALETT_NO_TOKEN when unknown):
                                       tokenizer_config.json's bos_token and eos_token, a GGUF file's ids, else
                                       the template's tokens or special tokens named so ("<s>", "</s>",
                                       "<|endoftext|>") */
    uint32_t pad, unk;              /* padding, and the token of what the vocabulary lacks */
    uint64_t reserved[SPINGALETT_RESERVED];
} SpingalettTokenizerInfo;

/* Reads a tokenizer.json file, a model directory holding one (its tokenizer_config.json, when there is
   one, names the BOS and EOS tokens) or a GGUF file's vocabulary. NULL on error. */
SPINGALETT_API SpingalettTokenizer *spingalett_tokenizer_load(const char *path);
/* The bytes of a tokenizer.json file, or of a GGUF file, in memory; they are not kept. */
SPINGALETT_API SpingalettTokenizer *spingalett_tokenizer_from_memory(const void *data, size_t size);
SPINGALETT_API void spingalett_tokenizer_free(SpingalettTokenizer *tokenizer);
SPINGALETT_API SpingalettTokenizerInfo spingalett_tokenizer_info(const SpingalettTokenizer *tokenizer);
/* The token of an id as the vocabulary writes it (in UTF-8: byte-level vocabularies write a space as U+0120,
   SentencePiece's as U+2581; "<|im_start|>"), valid while the tokenizer lives; NULL for an id of no token. */
SPINGALETT_API const char *spingalett_tokenizer_token(const SpingalettTokenizer *tokenizer, uint32_t id);
/* The id of a token as the vocabulary writes it (added tokens first), SPINGALETT_NO_TOKEN when none. */
SPINGALETT_API uint32_t spingalett_tokenizer_id(const SpingalettTokenizer *tokenizer, const char *token);

typedef struct {
    const SpingalettTokenizer *tokenizer;
    const char *text;               /* UTF-8 (other bytes are an error) */
    size_t length;                  /* its bytes; 0: up to its NUL */
    uint32_t *tokens;               /* [capacity] receives the ids, or NULL with capacity 0 to count them */
    size_t capacity;
    bool special;                   /* special tokens written in the text ("<|im_start|>") become their ids;
                                       otherwise they are text like any other (the default, so that text
                                       from users cannot forge them). Added tokens that are not special
                                       are always found. */
    bool add_special;               /* the tokens the tokenizer adds around a text (LLaMA's BOS), as Hugging
                                       Face's encode(add_special_tokens=True) */
    uint64_t reserved[SPINGALETT_RESERVED];
} SpingalettTokenizeArgs;

/* Encodes a text. Returns its number of tokens, also when that exceeds capacity (the first capacity are
   written: call again with room for all), -1 on error. */
#define spingalett_tokenize(...) spingalett_tokenize_args((SpingalettTokenizeArgs){__VA_ARGS__})
SPINGALETT_API int64_t spingalett_tokenize_args(SpingalettTokenizeArgs args);

typedef struct {
    const SpingalettTokenizer *tokenizer;
    const uint32_t *tokens;         /* [count] */
    size_t count;
    char *text;                     /* [capacity] receives the UTF-8 text and a NUL (NULL with capacity 0 to
                                       measure it); bytes that are no UTF-8, as a lone byte of a character
                                       split between tokens, become U+FFFD */
    size_t capacity;
    bool skip_special;              /* leaves special tokens out */
    uint64_t reserved[SPINGALETT_RESERVED];
} SpingalettDetokenizeArgs;

/* Decodes tokens into text. Returns the text's bytes, also when they exceed capacity - 1 (the first
   capacity - 1 are written, with a NUL), -1 on error (ids of no token among them). */
#define spingalett_detokenize(...) spingalett_detokenize_args((SpingalettDetokenizeArgs){__VA_ARGS__})
SPINGALETT_API int64_t spingalett_detokenize_args(SpingalettDetokenizeArgs args);

/*
 * Chat templates: the Jinja templates with which language models turn a conversation into the text they were
 * trained on, rendered as Hugging Face's transformers renders them (apply_chat_template(): trim_blocks and
 * lstrip_blocks, loop controls, tojson, raise_exception() and strftime_now()), the templates of today's families
 * among them (Qwen2.5 and Qwen3, Llama 3.x, Gemma 2 and 3, SmolLM, Phi, Mistral, DeepSeek, Granite, OLMo, GLM,
 * Falcon) with their tools and flags. A tokenizer has the template of its GGUF file (tokenizer.chat_template) or of
 * its directory (chat_template.jinja, or tokenizer_config.json's chat_template).
 */

/* A message of a conversation. */
typedef struct {
    const char *role;               /* "system", "user", "assistant", "tool", ... */
    const char *content;            /* its text (NULL: none) */
} SpingalettChatMessage;

/* The tokenizer's chat template (UTF-8, valid while the tokenizer lives), NULL when it has none. */
SPINGALETT_API const char *spingalett_tokenizer_chat_template(const SpingalettTokenizer *tokenizer);

typedef struct {
    const SpingalettTokenizer *tokenizer;   /* its template, and its BOS and EOS tokens as bos_token and eos_token
                                               (NULL: chat_template alone) */
    const char *chat_template;      /* a template of one's own (NULL: the tokenizer's) */
    const SpingalettChatMessage *messages;  /* [count]: the template's messages, each a role and a content */
    uint32_t count;
    bool add_generation_prompt;     /* the text that opens the assistant's reply after the messages */
    const char *variables;          /* a JSON object of more variables (NULL: none): tools, documents, flags the
                                       template reads (enable_thinking), messages with tool calls or parts (which
                                       replace messages[]), special tokens of one's own */
    uint64_t time;                  /* the time strftime_now() formats (local time), seconds since 1970 (0: now) */
    char *text;                     /* [capacity] receives the UTF-8 text and a NUL (NULL with capacity 0 to
                                       measure it) */
    size_t capacity;
    uint64_t reserved[SPINGALETT_RESERVED];
} SpingalettChatArgs;

/* Renders a conversation with a chat template. Returns the text's bytes, also when they exceed capacity - 1 (the
   first capacity - 1 are written, with a NUL), -1 on error: the template's raise_exception() (its message is the
   error's: "Conversation roles must alternate ..."), what Python would refuse (a string added to a list), or a
   part of Jinja the renderer does not have, named. */
#define spingalett_chat_render(...) spingalett_chat_render_args((SpingalettChatArgs){__VA_ARGS__})
SPINGALETT_API int64_t spingalett_chat_render_args(SpingalettChatArgs args);

/*
 * Graphs of operators (docs/Graphs.md): models as graphs of 155 operators over tensors of up to eight axes, their
 * sizes numbers or named symbols (a batch, a sequence's tokens), several named inputs and outputs: ONNX files, read
 * one operator to one, and `.slett` files of format 11, which spingalett_graph_save() writes. The operators are
 * ONNX's, with ONNX's semantics (opsets 7 to 28: broadcasting, attributes, opset 23's attention and rotary
 * embeddings), computed on the CPU in single precision for floats (float16 and bfloat16 constants and inputs widened,
 * outputs narrowed back) and in their own types for integers and booleans, every operator summing in fixed orders:
 * the same bits on any number of threads, in SPINGALETT_COMPUTE_OPENMP's threads. When a graph opens, constants are
 * folded, what no output needs and repeated nodes are left out, and chains of operators are fused into the library's
 * kernels (products by constant matrices with their bias and activation, attention, GELU, a convolution's
 * activation). A graph runs one call at a time (calls from several threads wait for each other); its outputs live in
 * its memory until its next run. With SPINGALETT_GRAPH_PROFILE=1 in the environment, a graph prints the time of each
 * operator when it is freed.
 */
typedef struct SpingalettGraph SpingalettGraph;

/* Element types of graphs' tensors (ONNX's TensorProto codes). */
typedef enum {
    SPINGALETT_DTYPE_UNDEFINED = 0,
    SPINGALETT_DTYPE_FLOAT32 = 1,
    SPINGALETT_DTYPE_UINT8 = 2,
    SPINGALETT_DTYPE_INT8 = 3,
    SPINGALETT_DTYPE_INT32 = 6,
    SPINGALETT_DTYPE_INT64 = 7,
    SPINGALETT_DTYPE_BOOL = 9,                  /* one byte, 0 or 1 */
    SPINGALETT_DTYPE_FLOAT16 = 10,
    SPINGALETT_DTYPE_BFLOAT16 = 16
} SpingalettDType;

/* Most axes of a graph's tensors. */
#define SPINGALETT_MAX_RANK 8

typedef struct {
    const char *path;               /* an ONNX file (its external data files beside it) or a .slett file of format 11 */
    const void *data;               /* or its bytes (path NULL), copied: they need not outlive the call */
    size_t size;
    uint64_t reserved[SPINGALETT_RESERVED];
} SpingalettGraphOpenArgs;

/* Opens a graph; NULL with the error set (an operator the library does not have is named, with the file's opset). */
#define spingalett_graph_open(...) spingalett_graph_open_args((SpingalettGraphOpenArgs){__VA_ARGS__})
SPINGALETT_API SpingalettGraph *spingalett_graph_open_args(SpingalettGraphOpenArgs args);
SPINGALETT_API void spingalett_graph_free(SpingalettGraph *graph);

/* A graph's input or output as it declares it. */
typedef struct {
    const char *name;               /* valid while the graph lives */
    SpingalettDType dtype;
    uint32_t rank;
    int64_t shape[SPINGALETT_MAX_RANK];         /* sizes; -1 where a symbol, or nothing, gives it */
    const char *symbols[SPINGALETT_MAX_RANK];   /* the symbols' names (NULL for numbers and unknown sizes) */
    uint64_t reserved[4];
} SpingalettTensorInfo;

SPINGALETT_API uint32_t spingalett_graph_input_count(const SpingalettGraph *graph);
SPINGALETT_API uint32_t spingalett_graph_output_count(const SpingalettGraph *graph);
/* Input or output i (all zeros past the last). */
SPINGALETT_API SpingalettTensorInfo spingalett_graph_input(const SpingalettGraph *graph, uint32_t i);
SPINGALETT_API SpingalettTensorInfo spingalett_graph_output(const SpingalettGraph *graph, uint32_t i);

/* A tensor given to or taken from a run, row-major. */
typedef struct {
    const char *name;               /* the graph's input or output (NULL: the graph's inputs, or outputs, in order) */
    SpingalettDType dtype;          /* inputs: its type (FLOAT32 for a graph's float16 or bfloat16 input is widened) */
    uint32_t rank;
    int64_t shape[SPINGALETT_MAX_RANK];
    void *data;                     /* inputs: the values (read only, aligned for their type); outputs: set by the
                                       run, in the graph's memory until its next run or its end */
    uint64_t reserved[4];
} SpingalettTensor;

typedef struct {
    SpingalettGraph *graph;
    const SpingalettTensor *inputs; /* [input_count]: every input of the graph */
    uint32_t input_count;
    SpingalettTensor *outputs;      /* [output_count]: the outputs wanted; their type, rank, shape and data set */
    uint32_t output_count;
    uint64_t reserved[SPINGALETT_RESERVED];
} SpingalettGraphRunArgs;

/* Runs a graph: false with the error set (a shape the graph refuses, a name it lacks, a node that fails: named). */
#define spingalett_graph_run(...) spingalett_graph_run_args((SpingalettGraphRunArgs){__VA_ARGS__})
SPINGALETT_API bool spingalett_graph_run_args(SpingalettGraphRunArgs args);
/* Writes a graph as a .slett file of format 11 (its constants as the graph holds them). */
SPINGALETT_API bool spingalett_graph_save(const SpingalettGraph *graph, const char *path);
/* The graph's nodes, and node i's operator (its ONNX name; NULL past the last). */
SPINGALETT_API uint32_t spingalett_graph_node_count(const SpingalettGraph *graph);
SPINGALETT_API const char *spingalett_graph_node_op(const SpingalettGraph *graph, uint32_t i);

#ifdef __cplusplus
}
#endif

#endif /* SPINGALETT_RUNTIME_H */

/* The runtime's names of 0.x, without the prefix, when SPINGALETT_SHORT_NAMES is defined (as the
   engine's: outside the include guard, so that including the header again after defining it gives
   them). */
#if defined(SPINGALETT_SHORT_NAMES) && !defined(SPINGALETT_RUNTIME_SHORT_NAMES)
#define SPINGALETT_RUNTIME_SHORT_NAMES
typedef SpingalettLogLevel LogLevel;
typedef SpingalettLogCallback LogCallback;
typedef SpingalettComputeMode ComputeMode;
typedef SpingalettEvalMetrics EvalMetrics;
#define LOG_DEBUG SPINGALETT_LOG_DEBUG
#define LOG_INFO SPINGALETT_LOG_INFO
#define LOG_WARNING SPINGALETT_LOG_WARNING
#define LOG_ERROR SPINGALETT_LOG_ERROR
#define LOG_NONE SPINGALETT_LOG_NONE
#define COMPUTE_SINGLE_THREADED SPINGALETT_COMPUTE_SINGLE_THREADED
#define COMPUTE_OPENMP SPINGALETT_COMPUTE_OPENMP
#define COMPUTE_OPENBLAS SPINGALETT_COMPUTE_OPENBLAS
#define COMPUTE_CUDA SPINGALETT_COMPUTE_CUDA
#define COMPUTE_VULKAN SPINGALETT_COMPUTE_VULKAN
#define COMPUTE_COUNT SPINGALETT_COMPUTE_COUNT
#endif
