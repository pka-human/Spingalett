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

/* Tiles: rows x columns per workgroup, the k-step, rows x columns per thread (multiples of four).
   Columns of 144 fit the windows of 3 x 3 convolutions over multiples of 16 channels (weight
   gradients). */
typedef struct { uint32_t bm, bn, bk, tm, tn; } Tile;
static const Tile tiles[] = {
    {128, 128, 16, 8, 8}, {128, 64, 16, 8, 4}, {64, 128, 16, 4, 8}, {64, 64, 16, 4, 4}, {128, 32, 16, 4, 4},
    {32, 128, 16, 4, 4}, {256, 32, 16, 8, 4}, {512, 16, 16, 8, 4}, {256, 16, 16, 8, 4}, {256, 16, 16, 4, 4},
    {16, 256, 16, 4, 4}, {64, 32, 16, 4, 4}, {32, 64, 16, 4, 4}, {128, 16, 16, 4, 4}, {16, 128, 16, 4, 4},
    {32, 32, 16, 4, 4}, {64, 16, 16, 4, 4}, {16, 64, 16, 4, 4}, {16, 144, 16, 4, 4}, {32, 144, 16, 8, 4},
    {64, 144, 16, 8, 8}, {256, 16, 8, 4, 16}, {512, 16, 8, 4, 16}, {256, 32, 8, 4, 8}, {128, 16, 16, 4, 16},
    {256, 64, 8, 8, 8}, {128, 32, 16, 8, 4}, {128, 64, 8, 8, 4},
};

/* Bytes of shared memory a tile takes (gemm.comp: BK rows of BM / 4 + 1 and BN / 4 + 1 vec4). */
static uint32_t tile_shared(const Tile *t) {
    return t->bk * (t->bm / 4u + 1u + t->bn / 4u + 1u) * 16u;
}

static bool tile_fits(const Tile *t) {
    return tile_shared(t) <= spg_gpu_shared_memory();
}

uint32_t spg_gemm_tiles(void) {
    return (uint32_t)(sizeof tiles / sizeof tiles[0]);
}

void spg_gemm_tile(uint32_t index, uint32_t *bm, uint32_t *bn, uint32_t *bk, uint32_t *tm, uint32_t *tn) {
    const Tile *t = &tiles[index];
    *bm = t->bm; *bn = t->bn; *bk = t->bk; *tm = t->tm; *tn = t->tn;
}

/* The tile that wastes the least work on padding per unit of register reuse, among those that give
   enough workgroups to fill the device when any does; smaller workgroups count as less reuse. */
static Tile choose_tile(uint32_t M, uint32_t N, uint32_t z) {
    double best = 0.0;
    Tile chosen = tiles[0];
    for (int pass = 0; pass < 2 && best == 0.0; pass++) {
        for (size_t k = 0; k < sizeof tiles / sizeof tiles[0]; k++) {
            const Tile *t = &tiles[k];
            uint64_t wg = (uint64_t)((M + t->bm - 1) / t->bm) * ((N + t->bn - 1) / t->bn) * z;
            if ((pass == 0 && wg < 64) || !tile_fits(t)) continue;
            double padded = (double)((M + t->bm - 1) / t->bm * t->bm) * ((N + t->bn - 1) / t->bn * t->bn);
            uint32_t threads = (t->bm / t->tm) * (t->bn / t->tn);
            double reuse = (double)(t->tm * t->tn) / (t->tm + t->tn) * (threads >= 128 ? 1.0 : 0.75);
            double cost = padded / reuse;
            if (best == 0.0 || cost < best) { best = cost; chosen = *t; }
        }
    }
    return chosen;
}

uint64_t spg_gemm_workgroups(uint32_t M, uint32_t N, uint32_t z) {
    Tile t = choose_tile(M, N, z);
    return (uint64_t)((M + t.bm - 1) / t.bm) * ((N + t.bn - 1) / t.bn) * z;
}

uint32_t spg_gemm_split(uint32_t M, uint32_t N, uint32_t K, uint32_t G, uint32_t *slice_k) {
    const uint64_t wg = spg_gemm_workgroups(M, N, G), outputs = (uint64_t)G * M * N;
    uint32_t slices = 1;
    while (slices < 1024 && wg * slices < 512 && K / (2u * slices) >= 128u &&
           2u * slices * outputs <= SPG_SPLIT_FLOATS)
        slices *= 2;
    *slice_k = ((K + slices - 1) / slices + 7u) & ~7u;
    return (K + *slice_k - 1) / *slice_k;
}

static bool aligned(uint64_t address, uint32_t group) {
    return address % 16u == 0 && group % 4u == 0;
}

/* ------------------------------------------------------------------------- tile choice by timing */

/*
 * Every tile computes each output as the same chain of fused multiply-adds in the order of k, so the
 * choice of tile changes the speed, never the bits: products are timed with each candidate tile on
 * their first use (outputs to scratch memory) and the fastest is kept for the life of the process.
 * SPINGALETT_GPU_TUNE=0 keeps the estimate of choose_tile() instead.
 */
typedef struct {
    uint32_t key[14];
    uint32_t tile;                  /* index of the fastest */
} Choice;

static struct {
    SpgSignal *lock;
    atomic_int state;               /* 0: not set up, 1: setting up, 2: ready */
    bool off;
    Choice *choices;
    size_t count, cap;
    SpgGpuBuffer scratch;
} tuner;

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

static void make_key(uint32_t *key, const SpgGemmPush *p, const SpgGemmMode *m, uint32_t vec) {
    const uint32_t values[14] = {m->amode, m->bmode, m->epi, vec, m->phased, m->groups, p->M, p->N, p->K, p->slices,
                                 p->slice_k, p->lda, p->ldb, p->ldc};
    memcpy(key, values, sizeof values);
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

/* Seconds per run of the product with tile t (0 when it cannot run). */
static double time_tile(const SpgGemmPush *p, const SpgGemmMode *m, uint32_t vec, const Tile *t, int reps) {
    SpgGpuCommands *c = spg_gpu_commands_create();
    if (!c) return 0.0;
    spg_gpu_commands_untimed(c);
    const uint32_t threads = (t->bm / t->tm) * (t->bn / t->tn);
    uint32_t spec[12] = {t->bm, t->bn, t->bk, t->tm, t->tn, m->amode, m->bmode, m->epi, m->act, threads,
                         m->phased ? 1u : 0u, vec};
    bool ok = spg_gpu_record_begin(c);
    for (int r = 0; ok && r < reps; r++) {
        spg_gpu_dispatch(c, SPG_KERNEL_gemm, spec, 12, p, sizeof *p, (p->M + t->bm - 1) / t->bm,
                         (p->N + t->bn - 1) / t->bn, m->groups * p->slices);
        spg_gpu_barrier(c);
    }
    ok = ok && spg_gpu_record_end(c);
    double start = spg_seconds();
    ok = ok && spg_gpu_submit(c) && spg_gpu_wait(c);
    double s = spg_seconds() - start;
    spg_gpu_commands_free(c);
    return ok ? s / reps : 0.0;
}

/* The fastest tile for this product, timed now if it is new; -1 when it cannot be timed. */
static int tuned_tile(const SpgGemmPush *p, const SpgGemmMode *m, uint32_t vec) {
    if (m->c_floats == 0 || !tuner_ready()) return -1;
    uint32_t key[14];
    make_key(key, p, m, vec);
    int found = -1;
    spg_lock(tuner.lock);
    for (size_t k = 0; k < tuner.count && found < 0; k++)
        if (!memcmp(tuner.choices[k].key, key, sizeof key)) found = (int)tuner.choices[k].tile;
    if (found >= 0 || (tuner.count == tuner.cap && !(tuner.cap = tuner.cap ? 2 * tuner.cap : 64,
                                                      tuner.choices = realloc(tuner.choices, tuner.cap * sizeof(Choice))))) {
        spg_unlock(tuner.lock);
        return found;
    }
    /* the outputs go to scratch memory, so that timing changes nothing */
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
    /* candidates: the tiles that waste at most half again the least padding, at most eight of them,
       those with the most register reuse among those that give enough workgroups (when some do) */
    const uint32_t z = m->groups * p->slices;
    double least = 0.0;
    for (size_t k = 0; k < sizeof tiles / sizeof tiles[0]; k++) {
        double padded = (double)((p->M + tiles[k].bm - 1) / tiles[k].bm * tiles[k].bm) *
                        ((p->N + tiles[k].bn - 1) / tiles[k].bn * tiles[k].bn);
        if ((least == 0.0 || padded < least) && tile_fits(&tiles[k])) least = padded;
    }
    int order[sizeof tiles / sizeof tiles[0]];
    double score[sizeof tiles / sizeof tiles[0]];
    uint32_t count = 0;
    for (size_t k = 0; k < sizeof tiles / sizeof tiles[0]; k++) {
        const Tile *t = &tiles[k];
        double padded = (double)((p->M + t->bm - 1) / t->bm * t->bm) * ((p->N + t->bn - 1) / t->bn * t->bn);
        if (padded > 1.5 * least || !tile_fits(t)) continue;
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
    if (count > 8) count = 8;
    double best = 0.0;
    int chosen = -1;
    for (uint32_t c = 0; c < count; c++) {
        const Tile *t = &tiles[order[c]];
        time_tile(&q, m, vec, t, 1);            /* the pipeline made, the caches warm */
        double s = time_tile(&q, m, vec, t, 3);
        if (s > 0.0 && s < 0.0005) s = time_tile(&q, m, vec, t, (int)(0.0015 / s) + 1);  /* at least ~1.5 ms */
        if (s > 0.0 && (chosen < 0 || s < best)) { best = s; chosen = order[c]; }
    }
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
    /* vectors along k need every slice to start at a multiple of four and k to end at one; along m
       (A_COL) or n (B_ROW, B_CONV), the dimension a multiple of four */
    const bool k_ok = p->K % 4u == 0 && (p->slices == 1 || p->slice_k % 4u == 0);
    bool va = mode->vec_a && aligned(p->a, p->a_group);
    if (mode->amode == SPG_A_COL) va = va && p->M % 4u == 0 && p->lda % 4u == 0;
    else va = va && k_ok && (mode->amode != SPG_A_ROW || p->lda % 4u == 0);
    bool vb = mode->vec_b && aligned(p->b, p->b_group);
    if (mode->bmode == SPG_B_COL) vb = vb && k_ok && p->ldb % 4u == 0;
    else vb = vb && p->N % 4u == 0 && (mode->bmode != SPG_B_ROW || p->ldb % 4u == 0);
    const uint32_t vec = (va ? 1u : 0u) | (vb ? 2u : 0u);
    int tuned = mode->tile ? -1 : tuned_tile(p, mode, vec);
    Tile t = mode->tile && tile_fits(&tiles[mode->tile - 1]) ? tiles[mode->tile - 1]
           : tuned >= 0 ? tiles[tuned] : choose_tile(p->M, p->N, mode->groups * p->slices);
    const uint32_t threads = (t.bm / t.tm) * (t.bn / t.tn);
    uint32_t spec[12] = {t.bm, t.bn, t.bk, t.tm, t.tn, mode->amode, mode->bmode, mode->epi, mode->act, threads,
                         mode->phased ? 1u : 0u, vec};
    spg_gpu_dispatch(c, SPG_KERNEL_gemm, spec, 12, p, sizeof *p, (p->M + t.bm - 1) / t.bm, (p->N + t.bn - 1) / t.bn,
                     mode->groups * p->slices);
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
