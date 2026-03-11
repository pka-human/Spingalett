/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

#include "Spingalett.Private.h"
#include <stdarg.h>
#include <stdio.h>
#include <stdatomic.h>

#if defined(_OPENMP)
#include <omp.h>
#endif

static _Atomic ComputeMode s_compute_mode = COMPUTE_SINGLE_THREADED;
static _Atomic unsigned s_num_threads = 0;

static _Atomic LogLevel s_log_level = LOG_INFO;
static _Atomic(LogCallback) s_log_callback = NULL;
static _Atomic bool s_verbose = true;

ComputeMode spingalett_get_compute_mode(void) {
    return atomic_load(&s_compute_mode);
}

void spingalett_set_compute_mode(ComputeMode mode) {
    if (mode < COMPUTE_COUNT)
        atomic_store(&s_compute_mode, mode);
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
        fprintf(out, "[%s] %s\n", level_names[level < LOG_NONE ? level : LOG_NONE], buf);
    }
}

ComputeMode resolve_compute_mode(void) {
    ComputeMode mode = spingalett_get_compute_mode();

    switch (mode) {
        case COMPUTE_OPENMP:
#if !defined(_OPENMP)
            spingalett_log(LOG_WARNING, "OpenMP requested but not available. Falling back to single-threaded.");
            return COMPUTE_SINGLE_THREADED;
#else
            return mode;
#endif
        case COMPUTE_OPENBLAS:
#if !defined(SPINGALETT_HAS_OPENBLAS)
            spingalett_log(LOG_WARNING, "OpenBLAS requested but not available. Falling back to single-threaded.");
            return COMPUTE_SINGLE_THREADED;
#else
            return mode;
#endif
        case COMPUTE_CUDA:
#if !defined(SPINGALETT_HAS_CUDA)
            spingalett_log(LOG_WARNING, "CUDA requested but not available. Falling back to single-threaded.");
            return COMPUTE_SINGLE_THREADED;
#else
            return mode;
#endif
        default:
            return mode;
    }
}
