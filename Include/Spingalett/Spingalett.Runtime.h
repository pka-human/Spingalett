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

/* Where the library computes. Deployment models (spingalett_model_predict()) run on the CPU in every
   mode: on all its threads with SPINGALETT_COMPUTE_OPENMP and SPINGALETT_COMPUTE_VULKAN, on one
   otherwise. The runtime has no OpenBLAS: there SPINGALETT_COMPUTE_OPENBLAS runs single-threaded, with
   a warning, as in a full library built without it. */
typedef enum {
    SPINGALETT_COMPUTE_SINGLE_THREADED,
    SPINGALETT_COMPUTE_OPENMP,
    SPINGALETT_COMPUTE_OPENBLAS,
    SPINGALETT_COMPUTE_CUDA,                   /* reserved: falls back to the CPU */
    SPINGALETT_COMPUTE_VULKAN,                 /* the GPU through Vulkan compute (spingalett_gpu_device()); the CPU
                                       parts of training run as with SPINGALETT_COMPUTE_OPENMP */
    SPINGALETT_COMPUTE_COUNT
} SpingalettComputeMode;

/* Result of spingalett_model_evaluate(), spingalett_evaluate() and the per-epoch validation pass. */
typedef struct {
    float loss;                     /* mean over samples of the network's loss (see spingalett_evaluate()) */
    float accuracy;                 /* fraction of samples whose output argmax matches the target's
                                       argmax; with a single output, both on the same side of 0.5 */
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
   threads in SPINGALETT_COMPUTE_OPENMP mode. Returns false on error. */
SPINGALETT_API bool spingalett_model_predict(const SpingalettModel *model, const float *inputs, uint32_t count,
                                             float *outputs);
/* Mean loss (the model's loss function, as spingalett_evaluate() computes it) and accuracy over a data set. */
SPINGALETT_API SpingalettEvalMetrics spingalett_model_evaluate(const SpingalettModel *model, const float *inputs,
                                                     const float *targets, uint32_t count);

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
