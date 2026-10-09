/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/* The matrix product of the GPU backend: the choice of tiles and loads for gemm.comp. */

#include "Spingalett.GpuKernels.h"
#include "Spingalett.Thread.h"
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#if !defined(TIME_UTC) && defined(_WIN32)
#include <windows.h>
#endif

/* Tiles of gemm.comp: rows x columns per workgroup, the k-step, rows x columns per thread (multiples
   of four). Columns of 144 fit the windows of 3 x 3 convolutions over multiples of 16 channels
   (weight gradients). */
typedef struct { uint32_t bm, bn, bk, tm, tn; } Tile;
static const Tile fp32_tiles[] = {
    {128, 128, 16, 8, 8}, {128, 64, 16, 8, 4}, {64, 128, 16, 4, 8}, {64, 64, 16, 4, 4}, {128, 32, 16, 4, 4},
    {32, 128, 16, 4, 4}, {256, 32, 16, 8, 4}, {512, 16, 16, 8, 4}, {256, 16, 16, 8, 4}, {256, 16, 16, 4, 4},
    {16, 256, 16, 4, 4}, {64, 32, 16, 4, 4}, {32, 64, 16, 4, 4}, {128, 16, 16, 4, 4}, {16, 128, 16, 4, 4},
    {32, 32, 16, 4, 4}, {64, 16, 16, 4, 4}, {16, 64, 16, 4, 4}, {16, 144, 16, 4, 4}, {32, 144, 16, 8, 4},
    {64, 144, 16, 8, 8}, {256, 16, 8, 4, 16}, {512, 16, 8, 4, 16}, {256, 32, 8, 4, 8}, {128, 16, 16, 4, 16},
    {256, 64, 8, 8, 8}, {128, 32, 16, 8, 4}, {128, 64, 8, 8, 4},
};

/* Tiles of gemm_mma.comp: rows x columns per workgroup, the k-step, accumulators of 16 x 16 per
   subgroup in rows x columns (the workgroup's subgroups tile it). */
static const Tile mma_tiles[] = {
    {128, 64, 32, 2, 1}, {64, 64, 32, 1, 1}, {128, 32, 32, 2, 1}, {64, 32, 32, 1, 1}, {32, 64, 32, 1, 1},
    {64, 128, 32, 1, 2}, {128, 64, 32, 1, 1}, {32, 64, 32, 1, 2}, {32, 32, 32, 1, 1},
    {64, 32, 32, 2, 1}, {128, 16, 32, 2, 1}, {64, 16, 32, 2, 1}, {32, 16, 32, 1, 1}, {16, 32, 32, 1, 1},
    {64, 64, 32, 2, 2}, {64, 64, 64, 2, 2}, {128, 64, 32, 2, 2}, {128, 64, 32, 4, 2}, {128, 128, 32, 4, 2},
    {128, 128, 32, 2, 4},
};

/* The tiles of the CUDA backend's product in single precision (Src/Gpu/Cuda/gemm.cu): those of
   Src/Gpu/Cuda/Kernels.def, which compiles each for every operand mode the executor uses. */
static const Tile cuda_fp32_tiles[] = {
#define SPG_CUDA_UNIT(name, source, defines)
#define SPG_CUDA_GEMM_MODES(a, b, vec)
#define SPG_CUDA_DW_WINDOW(kh, kw, sh, sw)
#define SPG_CUDA_GEMM_TILE(bm, bn, bk, tm, tn) {bm, bn, bk, tm, tn},
#include "Cuda/Kernels.def"
#undef SPG_CUDA_UNIT
#undef SPG_CUDA_GEMM_MODES
#undef SPG_CUDA_DW_WINDOW
#undef SPG_CUDA_GEMM_TILE
};

typedef struct { const Tile *list; uint32_t count; } Table;

/* The tiles of the calling thread's backend. */
static Table table(bool mma) {
    if (spg_gpu_using() == SPG_BACKEND_CUDA)
        return (Table){cuda_fp32_tiles, mma ? 0u : sizeof cuda_fp32_tiles / sizeof cuda_fp32_tiles[0]};
    return mma ? (Table){mma_tiles, sizeof mma_tiles / sizeof mma_tiles[0]}
               : (Table){fp32_tiles, sizeof fp32_tiles / sizeof fp32_tiles[0]};
}

static uint32_t tile_threads(const Tile *t, bool mma) {
    return mma ? (t->bm / (16u * t->tm)) * (t->bn / (16u * t->tn)) * spg_gpu_subgroup_size()
               : (t->bm / t->tm) * (t->bn / t->tn);
}

/* Bytes of shared memory a tile takes: gemm.comp's BK rows of BM / 4 + 1 and BN / 4 + 1 vec4;
   gemm_mma.comp's BK rows of BM + 8 and BN + 8 bfloat16, and a 16 x 16 block of floats a subgroup. */
static uint32_t tile_shared(const Tile *t, bool mma) {
    if (spg_gpu_using() == SPG_BACKEND_CUDA) return SPG_CUDA_GEMM_SHARED(t->bm, t->bn, t->bk);
    if (mma) return ((t->bk + 8u) * (t->bm + t->bn) + 8u * 2u * t->bk + t->bk * 16u) * 2u +
                    tile_threads(t, true) / spg_gpu_subgroup_size() * 1024u;       /* the larger of the layouts */
    return t->bk * (t->bm / 4u + 1u + t->bn / 4u + 1u) * 16u;
}

/* Whether the device can run the tile on N columns: its shared memory, its threads, and the tiles of
   columns in y (those of rows go in parts when there are more than a dispatch may have). */
static bool tile_fits(const Tile *t, bool mma, uint32_t N) {
    return tile_shared(t, mma) <= spg_gpu_shared_memory() && tile_threads(t, mma) <= 1024u &&
           (N + t->bn - 1) / t->bn <= spg_gpu_max_workgroups(1);
}

uint32_t spg_gemm_tiles(bool mma) {
    return table(mma).count;
}

void spg_gemm_tile(bool mma, uint32_t index, uint32_t *bm, uint32_t *bn, uint32_t *bk, uint32_t *tm, uint32_t *tn) {
    const Tile *t = &table(mma).list[index];
    *bm = t->bm; *bn = t->bn; *bk = t->bk; *tm = t->tm; *tn = t->tn;
}

/* The tile that wastes the least work on padding per unit of register reuse, among those that give
   enough workgroups to fill the device when any does; smaller workgroups count as less reuse. */
static Tile choose_tile(uint32_t M, uint32_t N, uint32_t z, bool mma) {
    const Table tab = table(mma);
    double best = 0.0;
    Tile chosen = tab.list[0];
    for (int pass = 0; pass < 2 && best == 0.0; pass++) {
        for (uint32_t k = 0; k < tab.count; k++) {
            const Tile *t = &tab.list[k];
            uint64_t wg = (uint64_t)((M + t->bm - 1) / t->bm) * ((N + t->bn - 1) / t->bn) * z;
            if ((pass == 0 && wg < 64) || !tile_fits(t, mma, N)) continue;
            double padded = (double)((M + t->bm - 1) / t->bm * t->bm) * ((N + t->bn - 1) / t->bn * t->bn);
            uint32_t threads = tile_threads(t, mma);
            double reuse = (double)(t->tm * t->tn) / (t->tm + t->tn) * (threads >= 128 ? 1.0 : 0.75);
            double cost = padded / reuse;
            if (best == 0.0 || cost < best) { best = cost; chosen = *t; }
        }
    }
    return chosen;
}

uint64_t spg_gemm_workgroups(uint32_t M, uint32_t N, uint32_t z) {
    Tile t = choose_tile(M, N, z, false);
    return (uint64_t)((M + t.bm - 1) / t.bm) * ((N + t.bn - 1) / t.bn) * z;
}

uint32_t spg_gemm_split(uint32_t M, uint32_t N, uint32_t K, uint32_t G, uint32_t *slice_k) {
    const uint64_t wg = spg_gemm_workgroups(M, N, G), outputs = (uint64_t)G * M * N;
    /* (the slices' partial sums, written and read, at most an eighth of the operands' floats: the
       MLP's weight gradients, 401,408 and 512,000 outputs of 2,048 terms, trained 13% faster in
       bfloat16 unsplit than in four slices, while convolutions' of a few thousand outputs over a
       hundred thousand pixels split as before) */
    uint32_t slices = 1;
    while (slices < 1024 && wg * slices < 512 && K / (2u * slices) >= 128u &&
           2u * slices * outputs <= SPG_SPLIT_FLOATS && 16u * slices * outputs <= (uint64_t)(M + N) * K * G)
        slices *= 2;
    *slice_k = ((K + slices - 1) / slices + 7u) & ~7u;
    return (K + *slice_k - 1) / *slice_k;
}

static bool aligned(uint64_t address, uint32_t group, uint32_t width) {
    return address % 16u == 0 && group % width == 0;
}

/* Whether operand A (or B) may be read w values of its contiguous axis at a time: along k, every slice
   must start at a multiple of w and k end at one; along m (A_COL) or n (B_ROW, B_CONV), the dimension
   and the stride must be multiples of w (convolutions: the caller checks their channels). */
/* Whether gemm.comp's epilogue may write four results of a row at once: C, and what the epilogue reads
   with it, at 16 bytes, rows (and groups) of whole vectors. */
static bool vector_c(const SpgGemmPush *p, const SpgGemmMode *m) {
    if (m->epi == SPG_EPI_PARTIAL) return p->c % 16u == 0 && p->N % 4u == 0;
    if (!aligned(p->c, p->c_group, 4u) || p->ldc % 4u != 0) return false;
    if (m->epi == SPG_EPI_BIAS_ACT && (p->flags & SPG_GEMM_BIAS)) return p->e0 % 16u == 0;
    if (m->epi == SPG_EPI_SCALE_ACT) return p->e0 % 16u == 0 && p->e1 % 16u == 0;
    if (m->epi == SPG_EPI_DERIV) return p->e0 % 16u == 0;
    return true;
}

static bool vectors(const SpgGemmPush *p, const SpgGemmMode *m, bool a, uint32_t w) {
    const bool k_ok = p->K % w == 0 && (p->slices == 1 || p->slice_k % w == 0);
    if (a) {
        if (!aligned(p->a, p->a_group, w)) return false;
        if (m->amode == SPG_A_COL) return p->M % w == 0 && p->lda % w == 0;
        return k_ok && (m->amode != SPG_A_ROW || p->lda % w == 0);
    }
    if (!aligned(p->b, p->b_group, w)) return false;
    if (m->bmode == SPG_B_COL) return k_ok && p->ldb % w == 0;
    return p->N % w == 0 && (m->bmode != SPG_B_ROW || p->ldb % w == 0);
}

/* ------------------------------------------------------------------------- tile choice by timing */

/*
 * Every tile computes each output as the same chain of fused multiply-adds in the order of k (on the
 * matrix units, of 16 x 16 x 16 products in the order of k), so the choice of tile changes the
 * speed, never the bits: products are timed with each candidate tile on their first use (outputs to
 * scratch memory) and the fastest is kept for the life of the process. SPINGALETT_GPU_TUNE=0 keeps
 * the estimate of choose_tile() instead.
 *
 * Times are the device's (timestamps), in rounds that run every candidate once, all in one
 * submission: a round of single runs to weed out the slow, then rounds of the rest, of which each
 * candidate's fastest counts, so that clocks still rising or other work on the device weigh on no
 * tile in particular. A device idle for a while is first kept busy for 25 ms, to raise its clocks.
 */
typedef struct {
    uint32_t key[16];
    uint32_t tile;                  /* index of the fastest in its table */
} Choice;

/* (a tuner a backend: the choices and scratch memory of one device) */
static struct {
    SpgSignal *lock;
    atomic_int state;               /* 0: not set up, 1: setting up, 2: ready */
    bool off;
    Choice *choices;
    size_t count, cap;
    SpgGpuBuffer scratch;
    double last;                    /* when the last product was timed (spg_seconds()) */
} tuners[SPG_BACKEND_COUNT];
#define tuner tuners[spg_gpu_using()]

static bool tuner_ready(void) {
    int expected = 0;
    if (atomic_compare_exchange_strong(&tuner.state, &expected, 1)) {
        const char *env = getenv("SPINGALETT_GPU_TUNE");
        tuner.off = env && *env == '0';
        tuner.lock = spg_signal_create();
        atomic_store(&tuner.state, 2);
    }
    while (atomic_load(&tuner.state) == 1) {}
    return tuner.lock && !tuner.off;
}

void spg_gemm_release(void) {
    if (atomic_load(&tuner.state) != 2 || !tuner.lock) return;
    spg_lock(tuner.lock);
    spg_gpu_buffer_free(&tuner.scratch);
    spg_unlock(tuner.lock);
}

static void make_key(uint32_t *key, const SpgGemmPush *p, const SpgGemmMode *m, uint32_t vec, bool mma) {
    const uint32_t values[16] = {m->amode, m->bmode, m->epi, vec, m->phased, m->groups, p->M, p->N, p->K, p->slices,
                                 p->slice_k, p->lda, p->ldb, p->ldc, mma, mma ? m->half : 0u};
    memcpy(key, values, sizeof values);
}

/* The specialization constants of the product with tile t (14 for gemm_mma.comp, the first 12 for
   gemm.comp). */
static void tile_spec(uint32_t spec[14], const SpgGemmMode *m, uint32_t vec, const Tile *t, bool mma) {
    const uint32_t values[14] = {t->bm, t->bn, t->bk, t->tm, t->tn, m->amode, m->bmode, m->epi, m->act,
                                 tile_threads(t, mma), m->phased ? 1u : 0u, vec, spg_gpu_subgroup_size(), m->half};
    memcpy(spec, values, sizeof values);
}

/* Records the product with tile t (gemm_mma.comp with mma). */
static void dispatch_tile(SpgGpuCommands *c, const SpgGemmPush *p, const SpgGemmMode *m, uint32_t vec, const Tile *t,
                          bool mma) {
    uint32_t spec[14];
    tile_spec(spec, m, vec, t, mma);
    /* tiles of rows in parts of at most what a dispatch may have in x */
    const uint32_t tiles = (p->M + t->bm - 1) / t->bm, most = spg_gpu_max_workgroups(0);
    SpgGemmPush q = *p;
    for (q.m_tile0 = 0; q.m_tile0 < tiles; q.m_tile0 += most)
        spg_gpu_dispatch(c, mma ? SPG_KERNEL_gemm_mma : SPG_KERNEL_gemm, spec, mma ? 14u : 12u, &q, sizeof q,
                         tiles - q.m_tile0 < most ? tiles - q.m_tile0 : most, (p->N + t->bn - 1) / t->bn,
                         m->groups * p->slices);
}

double spg_seconds(void) {
#if defined(TIME_UTC)                    /* C11 timespec_get; some Windows C runtimes lack it */
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
#elif defined(_WIN32)
    LARGE_INTEGER frequency, counter;
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart / (double)frequency.QuadPart;
#else
    return (double)clock() / CLOCKS_PER_SEC;
#endif
}

/* Device seconds per run of each of `count` candidates (0: not timed): `rounds` rounds of `reps` runs
   of each, the fastest round kept; false when the commands could not run. */
static bool time_tiles(const SpgGemmPush *p, const SpgGemmMode *m, uint32_t vec, bool mma, const int *order,
                       uint32_t count, uint32_t rounds, uint32_t reps, double *seconds) {
    const Table tab = table(mma);
    SpgGpuCommands *c = spg_gpu_commands_create();
    if (!c) return false;
    spg_gpu_commands_untimed(c);
    const uint32_t stamps = 2u * rounds * count;
    bool ok = spg_gpu_commands_stamps(c, stamps) && spg_gpu_record_begin(c);
    for (uint32_t r = 0; ok && r < rounds; r++)
        for (uint32_t k = 0; k < count; k++) {
            spg_gpu_timestamp(c, 2u * (r * count + k));
            for (uint32_t rep = 0; rep < reps; rep++) {
                dispatch_tile(c, p, m, vec, &tab.list[order[k]], mma);
                spg_gpu_barrier(c);
            }
            spg_gpu_timestamp(c, 2u * (r * count + k) + 1u);
        }
    double *ns = (double *)malloc(stamps * sizeof(double));
    ok = ok && ns && spg_gpu_record_end(c) && spg_gpu_submit(c) && spg_gpu_wait(c) && spg_gpu_timestamps(c, ns, stamps);
    for (uint32_t k = 0; ok && k < count; k++) {
        seconds[k] = 0.0;
        for (uint32_t r = 0; r < rounds; r++) {
            const double t = (ns[2u * (r * count + k) + 1u] - ns[2u * (r * count + k)]) * 1e-9 / reps;
            if (t > 0.0 && (seconds[k] == 0.0 || t < seconds[k])) seconds[k] = t;
        }
    }
    free(ns);
    spg_gpu_commands_free(c);
    return ok;
}

/* The fastest tile for this product, timed now if it is new; -1 when it cannot be timed. */
static int tuned_tile(const SpgGemmPush *p, const SpgGemmMode *m, uint32_t vec, bool mma) {
    if (m->c_floats == 0 || !tuner_ready()) return -1;
    const Table tab = table(mma);
    uint32_t key[16];
    make_key(key, p, m, vec, mma);
    int found = -1;
    spg_lock(tuner.lock);
    for (size_t k = 0; k < tuner.count && found < 0; k++)
        if (!memcmp(tuner.choices[k].key, key, sizeof key)) found = (int)tuner.choices[k].tile;
    if (found >= 0 || (tuner.count == tuner.cap && !(tuner.cap = tuner.cap ? 2 * tuner.cap : 64,
                                                      tuner.choices = realloc(tuner.choices, tuner.cap * sizeof(Choice))))) {
        spg_unlock(tuner.lock);
        return found;
    }
    /* the outputs go to scratch memory, so that timing changes nothing (as floats: room for bfloat16) */
    const uint64_t bytes = (m->epi == SPG_EPI_PARTIAL ? (uint64_t)m->groups * p->slices * p->M * p->N : m->c_floats) * 4u;
    if (tuner.scratch.size < bytes) {
        spg_gpu_buffer_free(&tuner.scratch);
        if (!spg_gpu_buffer_create(&tuner.scratch, bytes, false)) {
            spg_unlock(tuner.lock);
            return -1;
        }
    }
    SpgGemmPush q = *p;
    q.c = tuner.scratch.address;
    q.beta = 0.0f;
    /* candidates: the tiles that waste at most half again the least padding, at most sixteen of them,
       those with the most register reuse among those that give enough workgroups (when some do) */
    const uint32_t z = m->groups * p->slices;
    double least = 0.0;
    for (uint32_t k = 0; k < tab.count; k++) {
        const Tile *t = &tab.list[k];
        double padded = (double)((p->M + t->bm - 1) / t->bm * t->bm) * ((p->N + t->bn - 1) / t->bn * t->bn);
        if ((least == 0.0 || padded < least) && tile_fits(t, mma, p->N)) least = padded;
    }
    int order[64];
    double score[64];
    uint32_t count = 0;
    for (uint32_t k = 0; k < tab.count && count < 64; k++) {
        const Tile *t = &tab.list[k];
        double padded = (double)((p->M + t->bm - 1) / t->bm * t->bm) * ((p->N + t->bn - 1) / t->bn * t->bn);
        if (padded > 1.5 * least || !tile_fits(t, mma, p->N)) continue;
        double wg = (double)((p->M + t->bm - 1) / t->bm) * ((p->N + t->bn - 1) / t->bn) * z;
        double reuse = (double)(t->tm * t->tn) / (t->tm + t->tn);
        score[count] = reuse * (wg < 128.0 ? wg / 128.0 : 1.0) * least / padded;
        order[count++] = (int)k;
    }
    for (uint32_t a = 0; a < count; a++)            /* best score first */
        for (uint32_t b = a + 1; b < count; b++)
            if (score[b] > score[a]) {
                double ts = score[a]; score[a] = score[b]; score[b] = ts;
                int to = order[a]; order[a] = order[b]; order[b] = to;
            }
    if (count > 16u) count = 16u;
    /* the candidates' pipelines made at once, on several threads (the driver's first compilations of
       them take most of a first training's time on a machine) */
    const uint32_t constants = mma ? 14u : 12u;
    uint32_t specs[16 * 14];
    for (uint32_t k = 0; k < count; k++) {
        uint32_t spec[14];
        tile_spec(spec, m, vec, &tab.list[order[k]], mma);
        memcpy(specs + k * constants, spec, constants * sizeof(uint32_t));
    }
    spg_gpu_prepare(mma ? SPG_KERNEL_gemm_mma : SPG_KERNEL_gemm, specs, constants, count);
    double t[64];
    int chosen = -1;
    /* a device idle for a second or more (or never tuned) is kept busy for 25 ms first */
    if (count > 0 && (tuner.last == 0.0 || spg_seconds() - tuner.last > 1.0) &&
        time_tiles(&q, m, vec, mma, order, 1, 1, 1, t) && t[0] > 0.0) {
        uint32_t reps = (uint32_t)(0.025 / t[0]);
        time_tiles(&q, m, vec, mma, order, 1, 1, reps < 1u ? 1u : reps > 4096u ? 4096u : reps, t);
    }
    /* single runs of every candidate; then rounds of those within half again of the fastest (eight at
       most), each run long enough (20 us) for the timestamps' grain */
    if (count > 0 && time_tiles(&q, m, vec, mma, order, count, 1, 1, t)) {
        double fastest = 0.0;
        for (uint32_t k = 0; k < count; k++)
            if (t[k] > 0.0 && (fastest == 0.0 || t[k] < fastest)) fastest = t[k];
        uint32_t kept = 0;
        for (uint32_t k = 0; k < count; k++)            /* in order of score, which ties keep */
            if (t[k] > 0.0 && t[k] <= 1.5 * fastest && kept < 8u) order[kept++] = order[k];
        const uint32_t reps = fastest > 0.0 && fastest < 20e-6 ? (uint32_t)(20e-6 / fastest) + 1u : 1u;
        if (kept > 0 && time_tiles(&q, m, vec, mma, order, kept, 4, reps, t)) {
            double best = 0.0;
            for (uint32_t k = 0; k < kept; k++)
                if (t[k] > 0.0 && (chosen < 0 || t[k] < best)) { best = t[k]; chosen = order[k]; }
        }
    }
    tuner.last = spg_seconds();
    if (chosen >= 0) {
        memcpy(tuner.choices[tuner.count].key, key, sizeof key);
        tuner.choices[tuner.count++].tile = (uint32_t)chosen;
    }
    spg_unlock(tuner.lock);
    return chosen;
}

void spg_gemm(SpgGpuCommands *c, SpgGemmPush *p, const SpgGemmMode *mode) {
    if (p->slices == 0) { p->slices = 1; p->slice_k = p->K; }
    if (p->M == 0 || p->N == 0) return;
    const bool mma = mode->bf16 && spg_gpu_mma_bf16();
    SpgGemmMode m = *mode;
    if (!mma) m.half = 0;
    /* vectors of four floats, four bfloat16 or (wide) eight bfloat16 of the operands kept as such */
    const bool wide_a = m.vec_a && (m.half & 1u) && m.wide_a && vectors(p, &m, true, 8u),
               wide_b = m.vec_b && (m.half & 2u) && m.wide_b && vectors(p, &m, false, 8u);
    const bool va = m.vec_a && (wide_a || vectors(p, &m, true, 4u)), vb = m.vec_b && (wide_b || vectors(p, &m, false, 4u));
    mode = &m;
    const uint32_t vec = (va ? 1u : 0u) | (vb ? 2u : 0u) | (wide_a ? 4u : 0u) | (wide_b ? 8u : 0u) |
                         (!mma && vector_c(p, &m) ? 16u : 0u);
    const Table tab = table(mma);
    int tuned = mode->tile ? -1 : tuned_tile(p, mode, vec, mma);
    Tile t = mode->tile && mode->tile <= tab.count && tile_fits(&tab.list[mode->tile - 1], mma, p->N) ? tab.list[mode->tile - 1]
           : tuned >= 0 ? tab.list[tuned] : choose_tile(p->M, p->N, mode->groups * p->slices, mma);
    dispatch_tile(c, p, mode, vec, &t, mma);
}

/* ------------------------------------------------------------------------- convolution geometry */

static void geometry(uint32_t *geo, uint32_t rh, uint32_t rw, uint32_t gh, uint32_t gw, uint32_t gc, uint32_t sh,
                     uint32_t sw, uint32_t ph, uint32_t pw) {
    memset(geo, 0, SPG_GEO_HEADER * sizeof(uint32_t));
    geo[0] = rh; geo[1] = rw; geo[2] = gh; geo[3] = gw; geo[4] = gc;
    geo[5] = sh; geo[6] = sw; geo[7] = ph; geo[8] = pw;
}

bool spg_conv_geometry(uint32_t *geo, SpgConvGeometry *info, uint32_t h, uint32_t w, uint32_t in, uint32_t oh,
                       uint32_t ow, uint32_t out, uint32_t groups, uint32_t kh, uint32_t kw, uint32_t sh, uint32_t sw,
                       uint32_t ph, uint32_t pw) {
    const uint32_t CG = in / groups, OG = out / groups, taps = kh * kw;
    if (sh * sw > SPG_MAX_PHASES || kh > 255u || kw > 255u) return false;
    memset(info, 0, sizeof *info);
    info->phases = sh * sw;
    size_t at = (SPG_GEO_HEADER + (size_t)taps * CG + 3u) & ~(size_t)3u;
    if (geo) {
        /* forward and weight gradients: output pixels read input windows, k = (tap, channel) */
        geometry(geo, oh, ow, h, w, in, sh, sw, ph, pw);
        for (uint32_t t = 0, k = 0; t < taps; t++)
            for (uint32_t c = 0; c < CG; c++, k++) geo[SPG_GEO_HEADER + k] = (t / kw) | (t % kw) << 8 | c << 16;
    }
    /* data gradients: input pixel (y, x) = (py + sh ry, px + sw rx) of phase (py, px) gets dY at output
       pixel (ry + (py + ph - ty) / sh, rx + ...) through every tap (ty, tx) with py + ph - ty a multiple
       of sh: a convolution of stride 1 over the phase's pixels, its offsets shifted by (bh, bw) >= 0 */
    uint32_t first = 0;
    for (uint32_t py = 0; py < sh; py++)
        for (uint32_t px = 0; px < sw; px++) {
            const uint32_t ph_index = py * sw + px;
            int bh = 0, bw = 0;
            uint32_t count = 0;
            for (uint32_t ty = 0; ty < kh; ty++)
                for (uint32_t tx = 0; tx < kw; tx++) {
                    int dy = (int)(py + ph) - (int)ty, dx = (int)(px + pw) - (int)tx;
                    if (((dy % (int)sh) + (int)sh) % (int)sh || ((dx % (int)sw) + (int)sw) % (int)sw) continue;
                    if (-dy / (int)sh > bh) bh = -dy / (int)sh;
                    if (-dx / (int)sw > bw) bw = -dx / (int)sw;
                    count++;
                }
            info->phase[ph_index].at = at;
            info->phase[ph_index].rh = h > py ? (h - py + sh - 1) / sh : 0;
            info->phase[ph_index].rw = w > px ? (w - px + sw - 1) / sw : 0;
            info->phase[ph_index].taps = count;
            info->phase[ph_index].first = first;
            if (geo) {
                uint32_t *d = geo + at;
                geometry(d, info->phase[ph_index].rh, info->phase[ph_index].rw, oh, ow, out, 1, 1, (uint32_t)bh,
                         (uint32_t)bw);
                d[11] = h; d[12] = w; d[13] = py; d[14] = px; d[15] = sh | sw << 16;
                uint32_t k = 0;
                for (uint32_t ty = 0; ty < kh; ty++)
                    for (uint32_t tx = 0; tx < kw; tx++) {
                        int dy = (int)(py + ph) - (int)ty, dx = (int)(px + pw) - (int)tx;
                        if (((dy % (int)sh) + (int)sh) % (int)sh || ((dx % (int)sw) + (int)sw) % (int)sw) continue;
                        uint32_t oy = (uint32_t)(dy / (int)sh + bh), ox = (uint32_t)(dx / (int)sw + bw);
                        if (oy > 255u || ox > 255u) return false;
                        for (uint32_t f = 0; f < OG; f++, k++) d[SPG_GEO_HEADER + k] = oy | ox << 8 | f << 16;
                    }
            }
            first += count;
            at += (SPG_GEO_HEADER + (size_t)count * OG + 3u) & ~(size_t)3u;
        }
    /* the tap order: the taps of each phase, phase after phase, in row order within one */
    info->order = at;
    info->size = at + taps;
    if (geo) {
        uint32_t rank = 0;
        for (uint32_t py = 0; py < sh; py++)
            for (uint32_t px = 0; px < sw; px++)
                for (uint32_t ty = 0; ty < kh; ty++)
                    for (uint32_t tx = 0; tx < kw; tx++) {
                        int dy = (int)(py + ph) - (int)ty, dx = (int)(px + pw) - (int)tx;
                        if (((dy % (int)sh) + (int)sh) % (int)sh || ((dx % (int)sw) + (int)sw) % (int)sw) continue;
                        geo[at + ty * kw + tx] = rank++;
                    }
    }
    return true;
}
