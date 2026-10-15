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
   the runtime reports as SPINGALETT_ERR_FORMAT_VERSION. */
SPINGALETT_API SpingalettModel *spingalett_model_load(const char *path);
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
    uint64_t reserved[SPINGALETT_RESERVED];
} SpingalettSessionGenerateArgs;

/* Continues several sequences together, a token each per step, the steps' rows computed at once, each
   sequence drawing as spingalett_generate() draws (the same tokens for the same logits and seed, whatever
   other sequences run with it). A sequence stops after count tokens, after a stop token (written as its
   last), or after the token that follows a full sequence (max_length tokens). Afterwards each holds its
   prompt and the tokens generated, but the token after a full sequence. Returns how many tokens the
   sequences wrote together, 0 on error. */
#define spingalett_session_generate(...) spingalett_session_generate_args((SpingalettSessionGenerateArgs){__VA_ARGS__})
SPINGALETT_API uint32_t spingalett_session_generate_args(SpingalettSessionGenerateArgs args);

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
