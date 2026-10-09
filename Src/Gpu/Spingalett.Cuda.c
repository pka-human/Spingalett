/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/*
 * The GPU through CUDA (spg_cuda_ops, Spingalett.Device.h). The driver (libcuda.so.1, nvcuda.dll) is
 * opened at run time and nothing links against CUDA; the first device of compute capability 8.0 or
 * later is used (SPINGALETT_CUDA_DEVICE picks one by its index). The kernels are PTX compiled from
 * Src/Gpu/Cuda and embedded (Spingalett.CudaKernels.h), a module a unit, which the driver compiles for
 * the device when a unit is first dispatched (and caches on disk).
 *
 * Commands are lists of launches, copies and fills, recorded into a CUDA graph when they end, so that a
 * submission is one launch of the graph; commands with timestamps, or every command with
 * SPINGALETT_GPU_PROFILE, run their list on the stream instead, with events around what they time.
 * All work goes to one stream, in submission order: a barrier orders nothing more.
 */

#include "Spingalett/Spingalett.Config.h"
#include "Spingalett.Device.h"
#include "Spingalett.Thread.h"
#include "Spingalett.GpuPush.h"
#if defined(__GNUC__)
#pragma GCC diagnostic ignored "-Woverlength-strings"   /* the PTX, a string a unit */
#endif
#include "Spingalett.CudaKernels.h"        /* spg_cuda_units[] */
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#include <windows.h>
static void *open_driver(void) { return (void *)LoadLibraryA("nvcuda.dll"); }
static void *driver_symbol(void *lib, const char *name) {
    return (void *)GetProcAddress((HMODULE)lib, name);
}
#else
#include <dlfcn.h>
static void *open_driver(void) {
    void *lib = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
    return lib ? lib : dlopen("libcuda.so", RTLD_NOW | RTLD_LOCAL);
}
static void *driver_symbol(void *lib, const char *name) { return dlsym(lib, name); }
#endif

/* ------------------------------------------------------------------------- the driver's functions */

typedef int CUresult;
typedef int CUdevice;
typedef struct CUctx_st *CUcontext;
typedef struct CUmod_st *CUmodule;
typedef struct CUfunc_st *CUfunction;
typedef struct CUstream_st *CUstream;
typedef struct CUevent_st *CUevent;
typedef struct CUgraph_st *CUgraph;
typedef struct CUgraphExec_st *CUgraphExec;
typedef unsigned long long CUdeviceptr;

#define CU_ATTRIBUTE_MULTIPROCESSOR_COUNT     16
#define CU_ATTRIBUTE_COMPUTE_MAJOR            75
#define CU_ATTRIBUTE_COMPUTE_MINOR            76
#define CU_ATTRIBUTE_UNIFIED_ADDRESSING       41
#define CU_ATTRIBUTE_SHARED_PER_BLOCK_OPTIN   97
#define CU_FUNC_MAX_DYNAMIC_SHARED            8
#define CU_FUNC_NUM_REGS                      4
#define CU_FUNC_SHARED_BYTES                  1
#define CU_FUNC_LOCAL_BYTES                   3
#define CU_STREAM_NON_BLOCKING                1
#define CU_CAPTURE_MODE_RELAXED               2
#define CU_MEMHOSTALLOC_PORTABLE              1
#define CU_MEMHOSTALLOC_DEVICEMAP             2
#define CU_EVENT_DISABLE_TIMING               2
#define CU_JIT_INFO_LOG_BUFFER                3
#define CU_JIT_INFO_LOG_BUFFER_SIZE           4
#define CU_JIT_ERROR_LOG_BUFFER               5
#define CU_JIT_ERROR_LOG_BUFFER_SIZE          6

/* name, symbol in the driver, parameters */
#define CU_FUNCTIONS(X) \
    X(cuInit, "cuInit", (unsigned)) \
    X(cuDeviceGetCount, "cuDeviceGetCount", (int *)) \
    X(cuDeviceGet, "cuDeviceGet", (CUdevice *, int)) \
    X(cuDeviceGetName, "cuDeviceGetName", (char *, int, CUdevice)) \
    X(cuDeviceGetAttribute, "cuDeviceGetAttribute", (int *, int, CUdevice)) \
    X(cuDeviceTotalMem, "cuDeviceTotalMem_v2", (size_t *, CUdevice)) \
    X(cuDevicePrimaryCtxRetain, "cuDevicePrimaryCtxRetain", (CUcontext *, CUdevice)) \
    X(cuCtxSetCurrent, "cuCtxSetCurrent", (CUcontext)) \
    X(cuStreamCreate, "cuStreamCreate", (CUstream *, unsigned)) \
    X(cuStreamSynchronize, "cuStreamSynchronize", (CUstream)) \
    X(cuModuleLoadDataEx, "cuModuleLoadDataEx", (CUmodule *, const void *, unsigned, int *, void **)) \
    X(cuModuleGetFunction, "cuModuleGetFunction", (CUfunction *, CUmodule, const char *)) \
    X(cuFuncGetAttribute, "cuFuncGetAttribute", (int *, int, CUfunction)) \
    X(cuFuncSetAttribute, "cuFuncSetAttribute", (CUfunction, int, int)) \
    X(cuMemAlloc, "cuMemAlloc_v2", (CUdeviceptr *, size_t)) \
    X(cuMemFree, "cuMemFree_v2", (CUdeviceptr)) \
    X(cuMemHostAlloc, "cuMemHostAlloc", (void **, size_t, unsigned)) \
    X(cuMemFreeHost, "cuMemFreeHost", (void *)) \
    X(cuMemHostGetDevicePointer, "cuMemHostGetDevicePointer_v2", (CUdeviceptr *, void *, unsigned)) \
    X(cuMemcpyAsync, "cuMemcpyAsync", (CUdeviceptr, CUdeviceptr, size_t, CUstream)) \
    X(cuMemsetD32Async, "cuMemsetD32Async", (CUdeviceptr, unsigned, size_t, CUstream)) \
    X(cuLaunchKernel, "cuLaunchKernel", (CUfunction, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, \
                                         unsigned, CUstream, void **, void **)) \
    X(cuEventCreate, "cuEventCreate", (CUevent *, unsigned)) \
    X(cuEventDestroy, "cuEventDestroy_v2", (CUevent)) \
    X(cuEventRecord, "cuEventRecord", (CUevent, CUstream)) \
    X(cuEventSynchronize, "cuEventSynchronize", (CUevent)) \
    X(cuEventElapsedTime, "cuEventElapsedTime", (float *, CUevent, CUevent)) \
    X(cuStreamBeginCapture, "cuStreamBeginCapture_v2", (CUstream, int)) \
    X(cuStreamEndCapture, "cuStreamEndCapture", (CUstream, CUgraph *)) \
    X(cuGraphInstantiate, "cuGraphInstantiateWithFlags", (CUgraphExec *, CUgraph, unsigned long long)) \
    X(cuGraphLaunch, "cuGraphLaunch", (CUgraphExec, CUstream)) \
    X(cuGraphExecDestroy, "cuGraphExecDestroy", (CUgraphExec)) \
    X(cuGraphDestroy, "cuGraphDestroy", (CUgraph))

#define DECLARE(name, symbol, params) static CUresult (*name) params;
CU_FUNCTIONS(DECLARE)
#undef DECLARE

/* ------------------------------------------------------------------------- the device */

#define UNIT_COUNT (sizeof spg_cuda_units / sizeof spg_cuda_units[0])
#define PROFILE_OPS 8192u

typedef struct {
    atomic_int state;               /* 0: not loaded, 1: loaded, 2: failed */
    CUmodule module;
    CUfunction function;
} Unit;

static struct {
    atomic_int state;               /* 0: not tried, 1: opening, 2: open, 3: unavailable */
    void *driver;
    CUdevice device;
    CUcontext context;
    CUstream stream, capture;       /* the work's stream; the one commands are captured on */
    char name[256];
    uint64_t memory;
    uint32_t sms, shared_optin;
    Unit units[UNIT_COUNT];
    SpgSignal *lock;                /* loading units, submitting, capturing, the profile */
    bool profile;
    struct { char label[48]; double ms; uint64_t calls; } stats[256];
    uint32_t stat_count;
    double busy_ms;
    uint64_t submissions;
} cu;

/* Every thread that calls the driver makes the device's context current first. */
static _Thread_local CUcontext bound;
static void bind(void) {
    if (bound != cu.context) {
        cuCtxSetCurrent(cu.context);
        bound = cu.context;
    }
}

static void profile_report(void) {
    if (cu.stat_count == 0) return;
    double total = 0.0;
    for (uint32_t k = 0; k < cu.stat_count; k++) total += cu.stats[k].ms;
    for (uint32_t a = 0; a < cu.stat_count; a++)
        for (uint32_t b = a + 1; b < cu.stat_count; b++)
            if (cu.stats[b].ms > cu.stats[a].ms) {
                __typeof__(cu.stats[0]) t = cu.stats[a];
                cu.stats[a] = cu.stats[b];
                cu.stats[b] = t;
            }
    fprintf(stderr, "CUDA time by kernel (%.1f ms in all, %.1f ms from first to last event of %llu submissions):\n",
            total, cu.busy_ms, (unsigned long long)cu.submissions);
    for (uint32_t k = 0; k < cu.stat_count; k++)
        fprintf(stderr, "  %-46s %10.2f ms %5.1f%% %9llu calls %8.1f us each\n", cu.stats[k].label, cu.stats[k].ms,
                100.0 * cu.stats[k].ms / total, (unsigned long long)cu.stats[k].calls,
                1000.0 * cu.stats[k].ms / (double)cu.stats[k].calls);
}

static void profile_add(const char *label, double ms) {
    for (uint32_t k = 0; k < cu.stat_count; k++)
        if (!strcmp(cu.stats[k].label, label)) {
            cu.stats[k].ms += ms;
            cu.stats[k].calls++;
            return;
        }
    if (cu.stat_count < sizeof cu.stats / sizeof cu.stats[0]) {
        snprintf(cu.stats[cu.stat_count].label, sizeof cu.stats[0].label, "%s", label);
        cu.stats[cu.stat_count].ms = ms;
        cu.stats[cu.stat_count++].calls = 1;
    }
}

static bool open_device(void) {
    cu.driver = open_driver();
    if (!cu.driver) return false;
#define FETCH(name, symbol, params) \
    if (!(*(void **)&name = driver_symbol(cu.driver, symbol))) return false;
    CU_FUNCTIONS(FETCH)
#undef FETCH
    int count = 0;
    if (cuInit(0) != 0 || cuDeviceGetCount(&count) != 0 || count == 0) return false;
    const char *env = getenv("SPINGALETT_CUDA_DEVICE");
    int first = 0, last = count - 1;
    if (env && *env) first = last = atoi(env);
    for (int index = first; index <= last && index < count; index++) {
        CUdevice d;
        int major = 0, minor = 0, unified = 0;
        if (cuDeviceGet(&d, index) != 0 || cuDeviceGetAttribute(&major, CU_ATTRIBUTE_COMPUTE_MAJOR, d) != 0 ||
            cuDeviceGetAttribute(&minor, CU_ATTRIBUTE_COMPUTE_MINOR, d) != 0 ||
            cuDeviceGetAttribute(&unified, CU_ATTRIBUTE_UNIFIED_ADDRESSING, d) != 0)
            continue;
        /* the kernels are PTX of sm_80: Ampere and later */
        if (major < 8 || !unified) continue;
        cu.device = d;
        size_t total = 0;
        int sms = 0, shared = 0;
        if (cuDeviceGetName(cu.name, sizeof cu.name, d) != 0 || cuDeviceTotalMem(&total, d) != 0 ||
            cuDeviceGetAttribute(&sms, CU_ATTRIBUTE_MULTIPROCESSOR_COUNT, d) != 0 ||
            cuDeviceGetAttribute(&shared, CU_ATTRIBUTE_SHARED_PER_BLOCK_OPTIN, d) != 0 ||
            cuDevicePrimaryCtxRetain(&cu.context, d) != 0)
            return false;
        cu.memory = total;
        cu.sms = (uint32_t)sms;
        cu.shared_optin = (uint32_t)shared;
        bind();
        if (cuStreamCreate(&cu.stream, CU_STREAM_NON_BLOCKING) != 0 ||
            cuStreamCreate(&cu.capture, CU_STREAM_NON_BLOCKING) != 0)
            return false;
        cu.lock = spg_signal_create();
        const char *prof = getenv("SPINGALETT_GPU_PROFILE");
        cu.profile = prof && *prof && *prof != '0';
        if (cu.profile) atexit(profile_report);
        return cu.lock != NULL;
    }
    return false;
}

static bool cub_open(void) {
    int expected = 0;
    if (atomic_compare_exchange_strong(&cu.state, &expected, 1)) atomic_store(&cu.state, open_device() ? 2 : 3);
    while (atomic_load(&cu.state) == 1) {}
    if (atomic_load(&cu.state) != 2) return false;
    bind();
    return true;
}

static const char *cub_device_name(void) { return cub_open() ? cu.name : NULL; }
static uint64_t cub_memory(void) { return cub_open() ? cu.memory : 0; }
/* what a block may ask for (the product's tiles: dynamic shared memory, beyond the 48 KB of static) */
static uint32_t cub_shared_memory(void) { return cub_open() ? cu.shared_optin : 48u * 1024u; }
static uint32_t cub_subgroup_size(void) { return 32u; }
static uint32_t cub_max_workgroups(uint32_t axis) { return axis == 0 ? 0x7FFFFFFFu : 65535u; }
static bool cub_mma_bf16(void) { return false; }
static bool cub_bf16_storage(void) { return false; }
static bool cub_host_writes(void) { return false; }

/* ------------------------------------------------------------------------- kernels */

/* Bytes of dynamic shared memory a kernel takes (the product's stages). */
static uint32_t dynamic_shared(SpgKernel kernel, const uint32_t *spec, uint32_t count) {
    return kernel == SPG_KERNEL_gemm && count >= 3 ? SPG_CUDA_GEMM_SHARED(spec[0], spec[1], spec[2]) : 0u;
}

/* Threads a block of each kernel (gemm: its spec's THREADS). */
static uint32_t block_threads(SpgKernel kernel, const uint32_t *spec, uint32_t count) {
    switch (kernel) {
        case SPG_KERNEL_bn: case SPG_KERNEL_bn_h: case SPG_KERNEL_output: return 64u;
        case SPG_KERNEL_gemm: case SPG_KERNEL_gemm_mma: return count > 9 ? spec[9] : 0u;
        default: return 256u;
    }
}

/* The unit that runs a kernel with these constants: the kernel's own, its tile's or window's. */
static int unit_of(SpgKernel kernel, const uint32_t *spec, uint32_t count) {
    char name[64];
    const char *base = spg_kernel_names[kernel];
    size_t len = strlen(base);
    if (len > 2 && !strcmp(base + len - 2, "_h")) len -= 2;             /* the bfloat16 variants: HALF in the spec */
    snprintf(name, sizeof name, "%.*s", (int)len, base);
    if (kernel == SPG_KERNEL_gemm) {
        if (count < 12) return -1;
        /* the convolution modes read four channels at once in instances of their own (A 3, B 3) */
        const uint32_t a = spec[5] == SPG_A_CONV && (spec[11] & 1u) ? 3u : spec[5];
        const uint32_t b = spec[6] == SPG_B_CONV && (spec[11] & 2u) ? 3u : spec[6];
        snprintf(name, sizeof name, "gemm_%ux%ux%u_%ux%u_a%ub%u", spec[0], spec[1], spec[2], spec[3], spec[4], a, b);
    } else if (kernel == SPG_KERNEL_dwconv || kernel == SPG_KERNEL_dwconv_h) {
        if (count >= 7) {
            char window[64];
            snprintf(window, sizeof window, "dwconv_k%u%us%u%u", spec[3], spec[4], spec[5], spec[6]);
            for (uint32_t u = 0; u < UNIT_COUNT; u++)
                if (!strcmp(spg_cuda_units[u].name, window)) return (int)u;
        }
    }
    for (uint32_t u = 0; u < UNIT_COUNT; u++)
        if (!strcmp(spg_cuda_units[u].name, name)) return (int)u;
    return -1;
}

/* The unit's function, its module given to the driver on first use. */
static CUfunction unit_function(int u) {
    if (u < 0) return NULL;
    Unit *unit = &cu.units[u];
    int state = atomic_load(&unit->state);
    if (state == 1) return unit->function;
    if (state == 2) return NULL;
    spg_lock(cu.lock);
    if (atomic_load(&unit->state) == 0) {
        char log[4096] = "", entry[96];
        int options[] = {CU_JIT_ERROR_LOG_BUFFER, CU_JIT_ERROR_LOG_BUFFER_SIZE};
        void *values[] = {log, (void *)(uintptr_t)sizeof log};
        snprintf(entry, sizeof entry, "spg_%s", spg_cuda_units[u].name);
        bool ok = cuModuleLoadDataEx(&unit->module, spg_cuda_units[u].ptx, 2, options, values) == 0 &&
                  cuModuleGetFunction(&unit->function, unit->module, entry) == 0;
        int fixed = 0;              /* (dynamic shared memory up to what the device allows a block) */
        if (ok && cuFuncGetAttribute(&fixed, CU_FUNC_SHARED_BYTES, unit->function) == 0)
            cuFuncSetAttribute(unit->function, CU_FUNC_MAX_DYNAMIC_SHARED, (int)cu.shared_optin - fixed);
        if (!ok) fprintf(stderr, "Spingalett: CUDA kernel %s: %s\n", spg_cuda_units[u].name, log);
        atomic_store(&unit->state, ok ? 1 : 2);
    }
    spg_unlock(cu.lock);
    return atomic_load(&unit->state) == 1 ? unit->function : NULL;
}

/* Gives the driver the units these sets of constants use, on several threads. */
typedef struct { SpgKernel kernel; const uint32_t *specs; uint32_t count, n, first, step; } Prepare;

static void prepare_some(void *arg) {
    const Prepare *p = (const Prepare *)arg;
    bind();
    for (uint32_t k = p->first; k < p->n; k += p->step) unit_function(unit_of(p->kernel, p->specs + k * p->count, p->count));
}

static void cub_prepare(SpgKernel kernel, const uint32_t *specs, uint32_t count, uint32_t n) {
    if (!cub_open() || n == 0) return;
    enum { THREADS = 4 };
    Prepare parts[THREADS];
    SpgThread *threads[THREADS] = {NULL};
    for (uint32_t t = 0; t < THREADS; t++) {
        parts[t] = (Prepare){kernel, specs, count, n, t, THREADS};
        if (t > 0) threads[t] = spg_thread_start(prepare_some, &parts[t]);
        if (t > 0 && !threads[t]) prepare_some(&parts[t]);
    }
    prepare_some(&parts[0]);
    for (uint32_t t = 1; t < THREADS; t++)
        if (threads[t]) spg_thread_join(threads[t]);
}

/* ------------------------------------------------------------------------- memory */

/* A buffer of its own: device memory, or pinned host memory the device reads and writes in place. Its
   `buffer` says which (1: device, 2: host); its `memory` is the allocation. */
static bool cub_buffer_create(SpgGpuBuffer *b, size_t bytes, bool host_visible) {
    memset(b, 0, sizeof *b);
    if (!cub_open()) return false;
    if (bytes == 0) bytes = 16;
    bytes = (bytes + 255u) & ~(size_t)255u;
    if (host_visible) {
        void *host = NULL;
        CUdeviceptr address = 0;
        if (cuMemHostAlloc(&host, bytes, CU_MEMHOSTALLOC_PORTABLE | CU_MEMHOSTALLOC_DEVICEMAP) != 0) return false;
        if (cuMemHostGetDevicePointer(&address, host, 0) != 0) {
            cuMemFreeHost(host);
            return false;
        }
        *b = (SpgGpuBuffer){(void *)2, host, address, host, bytes, 0, SPG_BACKEND_CUDA};
        return true;
    }
    CUdeviceptr address = 0;
    if (cuMemAlloc(&address, bytes) != 0) return false;
    *b = (SpgGpuBuffer){(void *)1, (void *)(uintptr_t)address, address, NULL, bytes, 0, SPG_BACKEND_CUDA};
    return true;
}

static void cub_buffer_free(SpgGpuBuffer *b) {
    if (!b || !b->buffer) return;
    if (b->memory) {                                    /* an arena's buffers leave their memory to it */
        bind();
        if (b->buffer == (void *)2) cuMemFreeHost(b->memory);
        else cuMemFree((CUdeviceptr)(uintptr_t)b->memory);
    }
    memset(b, 0, sizeof *b);
}

/* An arena: a block of device memory and one of host memory, the buffers 256-byte ranges of them. */
typedef struct { SpgGpuBuffer *buffer; size_t bytes; SpgMemory kind; } Ask;
typedef struct {
    Ask *asks;
    uint32_t count, cap;
    SpgGpuBuffer blocks[2];
    bool failed;
} CuArena;

static void *cub_arena_create(void) {
    return calloc(1, sizeof(CuArena));
}

static void cub_arena_add(void *arena, SpgGpuBuffer *buffer, size_t bytes, SpgMemory kind) {
    CuArena *a = (CuArena *)arena;
    memset(buffer, 0, sizeof *buffer);
    if (bytes == 0) return;
    if (a->count == a->cap) {
        const uint32_t cap = a->cap ? 2u * a->cap : 64u;
        Ask *grown = (Ask *)realloc(a->asks, cap * sizeof *grown);
        if (!grown) { a->failed = true; return; }
        a->asks = grown;
        a->cap = cap;
    }
    a->asks[a->count++] = (Ask){buffer, (bytes + 255u) & ~(size_t)255u, kind};
}

static bool cub_arena_commit(void *arena) {
    CuArena *a = (CuArena *)arena;
    if (a->failed || !cub_open()) return false;
    size_t need[2] = {0, 0};
    for (uint32_t k = 0; k < a->count; k++) need[a->asks[k].kind != SPG_MEMORY_DEVICE] += a->asks[k].bytes;
    for (int host = 0; host < 2; host++)
        if (need[host] && !cub_buffer_create(&a->blocks[host], need[host], host != 0)) return false;
    size_t at[2] = {0, 0};
    for (uint32_t k = 0; k < a->count; k++) {
        const Ask *ask = &a->asks[k];
        const int host = ask->kind != SPG_MEMORY_DEVICE;
        const SpgGpuBuffer *block = &a->blocks[host];
        *ask->buffer = (SpgGpuBuffer){block->buffer, NULL, block->address + at[host],
                                      block->mapped ? (char *)block->mapped + at[host] : NULL, ask->bytes, at[host],
                                      SPG_BACKEND_CUDA};
        at[host] += ask->bytes;
    }
    return true;
}

static void cub_arena_free(void *arena) {
    CuArena *a = (CuArena *)arena;
    if (!a) return;
    cub_buffer_free(&a->blocks[0]);
    cub_buffer_free(&a->blocks[1]);
    free(a->asks);
    free(a);
}

/* ------------------------------------------------------------------------- commands */

typedef enum { OP_LAUNCH, OP_COPY, OP_FILL, OP_STAMP } OpKind;

typedef struct {
    OpKind kind;
    CUfunction function;
    uint32_t grid[3], block, shared;
    uint64_t push[SPG_PUSH_BYTES / 8];
    uint32_t spec[SPG_SPEC_MAX];
    uint64_t src, dst, bytes;
    uint32_t value;
    char label[48];                 /* profiling */
} Op;

typedef struct {
    Op *ops;
    uint32_t count, cap;
    bool ok, pending, untimed;
    CUevent done;
    CUevent *stamps;                /* spg_gpu_commands_stamps() */
    uint32_t stamp_count;
    CUgraphExec exec;               /* the list as a graph (NULL: run on the stream) */
    CUevent *timed;                 /* profiling: events around each op of a run */
    uint32_t timed_count;
    char (*labels)[48];
} CuCommands;

static void *cub_commands_create(void) {
    if (!cub_open()) return NULL;
    CuCommands *c = (CuCommands *)calloc(1, sizeof *c);
    if (c && cuEventCreate(&c->done, CU_EVENT_DISABLE_TIMING) != 0) {
        free(c);
        return NULL;
    }
    return c;
}

static bool cub_wait(void *commands);

static void drop_graph(CuCommands *c) {
    if (c->exec) cuGraphExecDestroy(c->exec);
    c->exec = NULL;
}

static void cub_commands_free(void *commands) {
    CuCommands *c = (CuCommands *)commands;
    if (!c) return;
    bind();
    if (c->pending) cub_wait(c);
    drop_graph(c);
    for (uint32_t k = 0; k < c->stamp_count; k++) cuEventDestroy(c->stamps[k]);
    for (uint32_t k = 0; k < 2u * c->timed_count; k++) cuEventDestroy(c->timed[k]);
    cuEventDestroy(c->done);
    free(c->stamps);
    free(c->timed);
    free(c->labels);
    free(c->ops);
    free(c);
}

static void cub_commands_untimed(void *commands) {
    ((CuCommands *)commands)->untimed = true;
}

static bool cub_commands_stamps(void *commands, uint32_t count) {
    CuCommands *c = (CuCommands *)commands;
    bind();
    c->stamps = (CUevent *)calloc(count, sizeof(CUevent));
    if (!c->stamps) return false;
    for (uint32_t k = 0; k < count; k++)
        if (cuEventCreate(&c->stamps[k], 0) != 0) {
            c->stamp_count = k;
            return false;
        }
    c->stamp_count = count;
    return true;
}

static Op *add_op(CuCommands *c, OpKind kind) {
    if (c->count == c->cap) {
        const uint32_t cap = c->cap ? 2u * c->cap : 64u;
        Op *grown = (Op *)realloc(c->ops, cap * sizeof *grown);
        if (!grown) { c->ok = false; return NULL; }
        c->ops = grown;
        c->cap = cap;
    }
    Op *op = &c->ops[c->count++];
    memset(op, 0, sizeof *op);
    op->kind = kind;
    return op;
}

static void cub_timestamp(void *commands, uint32_t index) {
    CuCommands *c = (CuCommands *)commands;
    if (index >= c->stamp_count) return;
    Op *op = add_op(c, OP_STAMP);
    if (op) op->value = index;
}

static bool cub_timestamps(void *commands, double *ns, uint32_t count) {
    CuCommands *c = (CuCommands *)commands;
    if (count > c->stamp_count || count == 0) return false;
    bind();
    ns[0] = 0.0;
    for (uint32_t k = 1; k < count; k++) {
        float ms = 0.0f;
        if (cuEventElapsedTime(&ms, c->stamps[0], c->stamps[k]) != 0) return false;
        ns[k] = (double)ms * 1e6;
    }
    return true;
}

static bool cub_record_begin(void *commands) {
    CuCommands *c = (CuCommands *)commands;
    if (c->pending && !cub_wait(c)) return false;
    drop_graph(c);
    c->count = 0;
    c->ok = true;
    return true;
}

static void cub_dispatch(void *commands, SpgKernel kernel, const uint32_t *spec, uint32_t spec_count, const void *push,
                         uint32_t push_size, uint32_t gx, uint32_t gy, uint32_t gz) {
    CuCommands *c = (CuCommands *)commands;
    if (gx == 0 || gy == 0 || gz == 0) return;
    const CUfunction f = spec_count <= SPG_SPEC_MAX && push_size <= SPG_PUSH_BYTES
                       ? unit_function(unit_of(kernel, spec, spec_count)) : NULL;
    const uint32_t threads = block_threads(kernel, spec, spec_count);
    Op *op = f && threads ? add_op(c, OP_LAUNCH) : NULL;
    if (!op) { c->ok = false; return; }
    op->function = f;
    op->grid[0] = gx; op->grid[1] = gy; op->grid[2] = gz;
    op->block = threads;
    op->shared = dynamic_shared(kernel, spec, spec_count);
    memcpy(op->push, push, push_size);
    memcpy(op->spec, spec, spec_count * sizeof(uint32_t));
    if (cu.profile) {
        int len = snprintf(op->label, sizeof op->label, "%s", spg_kernel_names[kernel]);
        if (kernel == SPG_KERNEL_gemm && spec_count >= 8)
            snprintf(op->label + len, sizeof op->label - (size_t)len, " A%u B%u E%u %ux%u", spec[5], spec[6], spec[7],
                     spec[0], spec[1]);
        else
            for (uint32_t k = 0; k < spec_count && k < 2 && len < 40; k++)
                len += snprintf(op->label + len, sizeof op->label - (size_t)len, " %u", spec[k]);
    }
}

static void cub_barrier(void *commands) { (void)commands; }        /* one stream: everything is in order */
static void cub_barrier_host(void *commands) { (void)commands; }

static void cub_copy(void *commands, const SpgGpuBuffer *src, size_t src_offset, const SpgGpuBuffer *dst,
                     size_t dst_offset, size_t bytes) {
    if (bytes == 0) return;
    Op *op = add_op((CuCommands *)commands, OP_COPY);
    if (!op) return;
    op->src = src->address + src_offset;
    op->dst = dst->address + dst_offset;
    op->bytes = bytes;
    snprintf(op->label, sizeof op->label, "copy%s", src->mapped ? " from host" : dst->mapped ? " to host" : "");
}

static void cub_fill(void *commands, const SpgGpuBuffer *dst, size_t offset, size_t bytes, uint32_t value) {
    if (bytes == 0) return;
    Op *op = add_op((CuCommands *)commands, OP_FILL);
    if (!op) return;
    op->dst = dst->address + offset;
    op->bytes = bytes;
    op->value = value;
    snprintf(op->label, sizeof op->label, "fill");
}

/* Puts op k of the list on stream s. */
static bool run_op(CuCommands *c, uint32_t k, CUstream s) {
    Op *op = &c->ops[k];
    switch (op->kind) {
        case OP_LAUNCH: {
            void *args[2] = {op->push, op->spec};
            return cuLaunchKernel(op->function, op->grid[0], op->grid[1], op->grid[2], op->block, 1, 1, op->shared, s,
                                  args, NULL) == 0;
        }
        case OP_COPY: return cuMemcpyAsync(op->dst, op->src, op->bytes, s) == 0;
        case OP_FILL: return cuMemsetD32Async(op->dst, op->value, op->bytes / 4u, s) == 0;
        case OP_STAMP: return cuEventRecord(c->stamps[op->value], s) == 0;
    }
    return false;
}

static bool cub_record_end(void *commands) {
    CuCommands *c = (CuCommands *)commands;
    if (!c->ok) return false;
    bind();
    /* a graph of the list, unless it has timestamps or is profiled */
    bool stamped = false;
    for (uint32_t k = 0; k < c->count && !stamped; k++) stamped = c->ops[k].kind == OP_STAMP;
    if (stamped || (cu.profile && !c->untimed) || c->count == 0) return true;
    spg_lock(cu.lock);
    CUgraph graph = NULL;
    bool ok = cuStreamBeginCapture(cu.capture, CU_CAPTURE_MODE_RELAXED) == 0;
    for (uint32_t k = 0; ok && k < c->count; k++) ok = run_op(c, k, cu.capture);
    ok = cuStreamEndCapture(cu.capture, &graph) == 0 && ok;
    ok = ok && cuGraphInstantiate(&c->exec, graph, 0) == 0;
    if (graph) cuGraphDestroy(graph);
    spg_unlock(cu.lock);
    if (!ok) c->exec = NULL;        /* run on the stream instead */
    return true;
}

static bool cub_submit(void *commands) {
    CuCommands *c = (CuCommands *)commands;
    if (c->pending && !cub_wait(c)) return false;
    bind();
    spg_lock(cu.lock);
    bool ok = true;
    if (c->exec) {
        ok = cuGraphLaunch(c->exec, cu.stream) == 0;
    } else if (cu.profile && !c->untimed) {
        /* events around every launch, copy and fill */
        if (c->timed_count < c->count) {
            CUevent *grown = (CUevent *)realloc(c->timed, 2u * c->count * sizeof(CUevent));
            char (*labels)[48] = (char (*)[48])realloc(c->labels, c->count * sizeof *labels);
            if (grown) c->timed = grown;
            if (labels) c->labels = labels;
            for (uint32_t k = 2u * c->timed_count; grown && labels && k < 2u * c->count; k++) cuEventCreate(&c->timed[k], 0);
            if (grown && labels) c->timed_count = c->count;
        }
        for (uint32_t k = 0; ok && k < c->count; k++) {
            const bool timed = k < c->timed_count && k < PROFILE_OPS && c->ops[k].kind != OP_STAMP;
            if (timed) cuEventRecord(c->timed[2u * k], cu.stream);
            ok = run_op(c, k, cu.stream);
            if (timed) cuEventRecord(c->timed[2u * k + 1u], cu.stream);
        }
    } else {
        for (uint32_t k = 0; ok && k < c->count; k++) ok = run_op(c, k, cu.stream);
    }
    ok = ok && cuEventRecord(c->done, cu.stream) == 0;
    spg_unlock(cu.lock);
    c->pending = ok;
    return ok;
}

static bool cub_wait(void *commands) {
    CuCommands *c = (CuCommands *)commands;
    if (!c->pending) return true;
    c->pending = false;
    bind();
    const bool ok = cuEventSynchronize(c->done) == 0;
    if (ok && cu.profile && !c->untimed && !c->exec && c->timed_count) {
        spg_lock(cu.lock);
        float first = 0.0f, last = 0.0f;
        for (uint32_t k = 0; k < c->count && k < c->timed_count && k < PROFILE_OPS; k++) {
            if (c->ops[k].kind == OP_STAMP) continue;
            float ms = 0.0f;
            if (cuEventElapsedTime(&ms, c->timed[2u * k], c->timed[2u * k + 1u]) == 0) profile_add(c->ops[k].label, ms);
            if (cuEventElapsedTime(&ms, c->timed[0], c->timed[2u * k + 1u]) == 0 && ms > last) last = ms;
        }
        cu.busy_ms += (double)(last - first);
        cu.submissions++;
        spg_unlock(cu.lock);
    }
    return ok;
}

const SpgGpuOps spg_cuda_ops = {
    cub_open, cub_device_name, cub_memory, cub_shared_memory, cub_subgroup_size, cub_max_workgroups, cub_mma_bf16,
    cub_bf16_storage, cub_host_writes, cub_buffer_create, cub_buffer_free, cub_arena_create, cub_arena_add,
    cub_arena_commit, cub_arena_free, cub_commands_create, cub_commands_free, cub_commands_untimed,
    cub_commands_stamps, cub_timestamp, cub_timestamps, cub_record_begin, cub_dispatch, cub_prepare, cub_barrier,
    cub_barrier_host, cub_copy, cub_fill, cub_record_end, cub_submit, cub_wait,
};
