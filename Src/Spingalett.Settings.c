/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

#include "Spingalett.Private.h"
#include "Spingalett.Gpu.h"
#include <stdarg.h>
#include <stdio.h>
#include <stdatomic.h>

#if defined(_OPENMP)
#include <omp.h>
#endif

#if defined(__SSE__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 1)
#include <xmmintrin.h>
#define SPINGALETT_FP_MXCSR 1
#endif

static _Atomic ComputeMode s_compute_mode = COMPUTE_SINGLE_THREADED;
#if !defined(SPINGALETT_RUNTIME)
static _Atomic PrecisionMode s_gpu_precision = PRECISION_FLOAT32;
#endif
static _Atomic unsigned s_num_threads = 0;

static _Atomic LogLevel s_log_level = LOG_INFO;
static _Atomic(LogCallback) s_log_callback;     /* zero-initialized: no callback (AppleClang rejects = NULL) */
static _Atomic bool s_verbose = true;

ComputeMode spingalett_get_compute_mode(void) {
    return atomic_load(&s_compute_mode);
}

bool spingalett_set_compute_mode(ComputeMode mode) {
    if ((unsigned)mode >= COMPUTE_COUNT) {
        set_error(SPINGALETT_ERR_INVALID, "spingalett_set_compute_mode: unknown mode");
        return false;
    }
    atomic_store(&s_compute_mode, mode);
    return true;
}

unsigned spingalett_get_num_threads(void) {
    return atomic_load(&s_num_threads);
}

void spingalett_set_num_threads(unsigned n) {
    atomic_store(&s_num_threads, n);
#if defined(_OPENMP)
    if (n > 0)
        omp_set_num_threads((int)n);
#endif
}

void spingalett_set_log_callback(LogCallback cb) {
    atomic_store(&s_log_callback, cb);
}

void spingalett_set_log_level(LogLevel level) {
    atomic_store(&s_log_level, level);
}

void spingalett_set_verbose(bool enabled) {
    atomic_store(&s_verbose, enabled);
}

bool spingalett_get_verbose(void) {
    return atomic_load(&s_verbose);
}

void spingalett_log(LogLevel level, const char *fmt, ...) {
    if (level < atomic_load(&s_log_level))
        return;

    if (!atomic_load(&s_verbose) && level < LOG_WARNING)
        return;

    char buf[1024];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);

    LogCallback cb = atomic_load(&s_log_callback);
    if (cb) {
        cb(level, buf);
    } else {
        FILE *out = (level >= LOG_WARNING) ? stderr : stdout;
        static const char *level_names[] = {"DEBUG", "INFO", "WARN", "ERROR", "NONE"};
        fprintf(out, "[%s] %s\n", level_names[(unsigned)level < LOG_NONE ? level : LOG_NONE], buf);
    }
}

static _Atomic bool s_fallback_warned[COMPUTE_COUNT];

static ComputeMode fallback_to_single_threaded(ComputeMode requested, const char *name) {
    if (!atomic_exchange(&s_fallback_warned[requested], true))
        spingalett_log(LOG_WARNING, "%s requested but not available. Falling back to single-threaded.", name);
    return COMPUTE_SINGLE_THREADED;
}

/*
 * Subnormal floats are ~100x slower on x86. Optimizer moments of weights that stop receiving
 * gradient (dead ReLUs, dropped units) decay geometrically into that range after a few hundred
 * steps and made per-sample training collapse from ~1500 to ~200 samples/s. Training therefore
 * flushes denormals to zero (FTZ + DAZ) and restores the caller's mode afterwards. The control
 * register is per thread, so the training code applies this on every OpenMP worker as well.
 */
static _Thread_local unsigned long long s_saved_fp_mode;

void spingalett_fp_flush_denormals_begin(void) {
#if defined(SPINGALETT_FP_MXCSR)
    unsigned csr = _mm_getcsr();
    s_saved_fp_mode = csr;
    _mm_setcsr(csr | 0x8040u);                      /* FTZ (bit 15) | DAZ (bit 6) */
#elif defined(__aarch64__) && defined(__GNUC__)
    unsigned long long fpcr;
    __asm__ volatile("mrs %0, fpcr" : "=r"(fpcr));
    s_saved_fp_mode = fpcr;
    fpcr |= 1ull << 24;                             /* FZ */
    __asm__ volatile("msr fpcr, %0" : : "r"(fpcr));
#endif
}

void spingalett_fp_flush_denormals_end(void) {
#if defined(SPINGALETT_FP_MXCSR)
    _mm_setcsr((unsigned)s_saved_fp_mode);
#elif defined(__aarch64__) && defined(__GNUC__)
    unsigned long long fpcr = s_saved_fp_mode;
    __asm__ volatile("msr fpcr, %0" : : "r"(fpcr));
#endif
}

ComputeMode resolve_compute_mode(void) {
    ComputeMode mode = spingalett_get_compute_mode();

    switch (mode) {
        case COMPUTE_OPENMP:
#if !defined(_OPENMP)
            return fallback_to_single_threaded(mode, "OpenMP");
#else
            return mode;
#endif
        case COMPUTE_OPENBLAS:
#if !defined(SPINGALETT_HAS_OPENBLAS)
            return fallback_to_single_threaded(mode, "OpenBLAS");
#else
            return mode;
#endif
        case COMPUTE_CUDA:
        case COMPUTE_VULKAN:
            /* what runs on the CPU uses its threads */
#if defined(_OPENMP)
            return COMPUTE_OPENMP;
#else
            return COMPUTE_SINGLE_THREADED;
#endif
        default:
            return mode;
    }
}

/* The GPU: the full library's (the runtime computes on the CPU, and COMPUTE_VULKAN gives its models the
   CPU's threads, as in the full library). */
#if !defined(SPINGALETT_RUNTIME)
bool spingalett_use_gpu(void) {
    const ComputeMode mode = spingalett_get_compute_mode();
    if (mode != COMPUTE_VULKAN && mode != COMPUTE_CUDA) return false;
    if (spingalett_gpu_select(mode)) return true;
    const bool cuda = mode == COMPUTE_CUDA;
    if (!atomic_exchange(&s_fallback_warned[mode], true))
        spingalett_log(LOG_WARNING, "%s requested but %s. Falling back to the CPU.", cuda ? "CUDA" : "Vulkan",
#if defined(SPINGALETT_HAS_VULKAN) && defined(SPINGALETT_HAS_CUDA)
                       cuda ? "no usable device was found (compute capability 8.0 or later, a CUDA 11 driver)"
                            : "no usable device was found (Vulkan 1.2 with buffer device addresses)"
#elif defined(SPINGALETT_HAS_VULKAN)
                       cuda ? "the library was built without it"
                            : "no usable device was found (Vulkan 1.2 with buffer device addresses)"
#elif defined(SPINGALETT_HAS_CUDA)
                       cuda ? "no usable device was found (compute capability 8.0 or later, a CUDA 11 driver)"
                            : "the library was built without it"
#else
                       "the library was built without it"
#endif
        );
    return false;
}

/* The name of a backend's device, the calling thread's backend left as it was. */
static const char *device_of(ComputeMode mode) {
    return spingalett_gpu_name_of(mode);
}

const char *spingalett_gpu_device(void) {
    return device_of(COMPUTE_VULKAN);
}

const char *spingalett_cuda_device(void) {
    return device_of(COMPUTE_CUDA);
}

bool spingalett_set_gpu_precision(PrecisionMode precision) {
    if (precision != PRECISION_FLOAT32 && precision != PRECISION_BFLOAT16) return false;
    atomic_store(&s_gpu_precision, precision);
    /* (what the backend of the compute mode does: CUDA's for COMPUTE_CUDA, else Vulkan's) */
    const ComputeMode mode = spingalett_get_compute_mode() == COMPUTE_CUDA ? COMPUTE_CUDA : COMPUTE_VULKAN;
    return spingalett_gpu_name_of(mode) && (precision == PRECISION_FLOAT32 || spingalett_gpu_bf16_of(mode));
}

PrecisionMode spingalett_get_gpu_precision(void) {
    return atomic_load(&s_gpu_precision);
}
#endif
