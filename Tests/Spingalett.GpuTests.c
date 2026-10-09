/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/*
 * The GPU's matrix product (Src/Gpu/Shaders/gemm.comp on Vulkan, Src/Gpu/Cuda/gemm.cu on CUDA) against a
 * reference in double precision: dense layers, convolutions (groups, strides, padding, rectangular
 * windows) and their data and weight gradients, with every tile and with and without vector loads.
 * Without a usable device the tests are skipped (exit code 77). SPINGALETT_GPU_BACKEND=cuda tests the
 * CUDA backend.
 *
 *   SpingalettGpuTests            the tests, with the tile chosen and a third of the others, in single
 *                                 precision and (with matrix units) in bfloat16 on them
 *   SpingalettGpuTests all        with every tile
 *   SpingalettGpuTests bench      times every tile on the products of ResNet-20 at 128 samples
 *   SpingalettGpuTests bench bf16 the same on the matrix units ("dense" after either: the MLP's
 *                                 products only; "half" with bf16: their operands and results kept
 *                                 as bfloat16, as training in bfloat16 keeps them; "fw" with half: the
 *                                 MLP's weights as floats, as training keeps them; SPINGALETT_BENCH_CONV
 *                                 = "n h w c out kh kw sh sw ph pw groups" times that convolution instead
 *                                 of ResNet-20's)
 */

#include "Spingalett.GpuKernels.h"
#include <Spingalett/Spingalett.Inference.h>     /* the activations (SPINGALETT_ACT_NONE) */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures;
static uint32_t tile_step = 3;      /* every tile_step-th tile is tried */
static bool mma;                    /* the products in bfloat16 on the matrix units (gemm_mma.comp) */
static bool half;                   /* bench: operands and results kept as bfloat16 */
static bool float_weights;          /* and with "fw" the MLP's weights as floats, as in training */

/* x rounded to bfloat16, to the nearest (ties to even), as the matrix kernel converts its operands */
static float bf16(float x) {
    uint32_t u;
    memcpy(&u, &x, 4);
    if ((u & 0x7F800000u) != 0x7F800000u) u = (u + 0x7FFFu + ((u >> 16) & 1u)) & 0xFFFF0000u;
    memcpy(&x, &u, 4);
    return x;
}

/* The operands as the kernel sees them: rounded to bfloat16 on the matrix units. */
static float *operands(const float *x, size_t n) {
    float *r = (float *)malloc(n * sizeof(float));
    for (size_t i = 0; i < n; i++) r[i] = mma ? bf16(x[i]) : x[i];
    return r;
}

static double seconds(void) {
    return spg_seconds();
}

static uint64_t rng = 0x9E3779B97F4A7C15ull;
static float uniform(void) {
    rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
    return (float)((rng >> 40) * (1.0 / 16777216.0)) * 2.0f - 1.0f;
}

/* A device buffer holding n floats (or uints) copied from host memory through a host-visible one. */
static bool upload(SpgGpuBuffer *dst, const void *src, size_t bytes) {
    SpgGpuBuffer staging;
    if (!spg_gpu_buffer_create(dst, bytes, false) || !spg_gpu_buffer_create(&staging, bytes, true)) return false;
    memcpy(staging.mapped, src, bytes);
    SpgGpuCommands *c = spg_gpu_commands_create();
    bool ok = c && spg_gpu_record_begin(c);
    if (ok) {
        spg_gpu_copy(c, &staging, 0, dst, 0, bytes);
        spg_gpu_barrier(c);
        ok = spg_gpu_record_end(c) && spg_gpu_submit(c) && spg_gpu_wait(c);
    }
    spg_gpu_commands_free(c);
    spg_gpu_buffer_free(&staging);
    return ok;
}

static bool download(void *dst, const SpgGpuBuffer *src, size_t bytes) {
    SpgGpuBuffer staging;
    if (!spg_gpu_buffer_create(&staging, bytes, true)) return false;
    SpgGpuCommands *c = spg_gpu_commands_create();
    bool ok = c && spg_gpu_record_begin(c);
    if (ok) {
        spg_gpu_barrier(c);
        spg_gpu_copy(c, src, 0, &staging, 0, bytes);
        spg_gpu_barrier(c);
        ok = spg_gpu_record_end(c) && spg_gpu_submit(c) && spg_gpu_wait(c);
    }
    if (ok) memcpy(dst, staging.mapped, bytes);
    spg_gpu_commands_free(c);
    spg_gpu_buffer_free(&staging);
    return ok;
}

/* Runs `count` products `reps` times; returns the seconds per run (0 on failure). */
static double run_many(const SpgGemmPush *p, const SpgGemmMode *m, uint32_t count, int reps) {
    SpgGpuCommands *c = spg_gpu_commands_create();
    bool ok = c && spg_gpu_record_begin(c);
    for (int r = 0; ok && r < reps; r++)
        for (uint32_t k = 0; k < count; k++) {
            SpgGemmPush q = p[k];
            spg_gemm(c, &q, &m[k]);
            spg_gpu_barrier(c);
        }
    ok = ok && spg_gpu_record_end(c);
    double start = seconds();
    ok = ok && spg_gpu_submit(c) && spg_gpu_wait(c);
    double t = seconds() - start;
    spg_gpu_commands_free(c);
    return ok ? t / reps : 0.0;
}

static double run(const SpgGemmPush *p, const SpgGemmMode *m, int reps) {
    return run_many(p, m, 1, reps);
}

/* ------------------------------------------------------------------------- convolutions */

typedef struct {
    uint32_t n, h, w, c, out, kh, kw, sh, sw, ph, pw, groups;
} Conv;

static uint32_t out_h(const Conv *v) { return (v->h + 2 * v->ph - v->kh) / v->sh + 1; }
static uint32_t out_w(const Conv *v) { return (v->w + 2 * v->pw - v->kw) / v->sw + 1; }

/* y = conv(x, w) (forward), dx = conv^T(dy, w) (data gradient), dw = dy^T im2col(x) (weights) */
static void reference(const Conv *v, const float *x, const float *w, const float *dy, double *y, double *dx,
                      double *dw) {
    const uint32_t OH = out_h(v), OW = out_w(v), G = v->groups, CG = v->c / G, OG = v->out / G;
    const uint32_t K = v->kh * v->kw * CG;
    memset(dx, 0, (size_t)v->n * v->h * v->w * v->c * sizeof(double));
    memset(dw, 0, (size_t)v->out * K * sizeof(double));
    for (uint32_t s = 0; s < v->n; s++)
        for (uint32_t oy = 0; oy < OH; oy++)
            for (uint32_t ox = 0; ox < OW; ox++)
                for (uint32_t f = 0; f < v->out; f++) {
                    const uint32_t g = f / OG;
                    const size_t o = (((size_t)s * OH + oy) * OW + ox) * v->out + f;
                    double acc = 0.0;
                    for (uint32_t ky = 0; ky < v->kh; ky++)
                        for (uint32_t kx = 0; kx < v->kw; kx++) {
                            int64_t iy = (int64_t)oy * v->sh - v->ph + ky, ix = (int64_t)ox * v->sw - v->pw + kx;
                            if (iy < 0 || ix < 0 || iy >= v->h || ix >= v->w) continue;
                            for (uint32_t c = 0; c < CG; c++) {
                                const size_t xi = (((size_t)s * v->h + iy) * v->w + ix) * v->c + g * CG + c;
                                const size_t wi = (size_t)f * K + (ky * v->kw + kx) * CG + c;
                                acc += (double)x[xi] * w[wi];
                                dx[xi] += (double)dy[o] * w[wi];
                                dw[wi] += (double)dy[o] * x[xi];
                            }
                        }
                    y[o] = acc;
                }
}

static double compare(const float *got, const double *want, size_t n) {
    double worst = 0.0, scale = 1e-6;
    for (size_t i = 0; i < n; i++) scale = fmax(scale, fabs(want[i]));
    for (size_t i = 0; i < n; i++) worst = fmax(worst, fabs(got[i] - want[i]) / scale);
    return worst;
}

typedef struct {
    SpgGpuBuffer x, w, wt, dy, y, dx, dw, part, geo;
    SpgConvGeometry info;
} ConvBuffers;

static void conv_free(ConvBuffers *b) {
    SpgGpuBuffer *all[] = {&b->x, &b->w, &b->wt, &b->dy, &b->y, &b->dx, &b->dw, &b->part, &b->geo};
    for (size_t k = 0; k < sizeof all / sizeof all[0]; k++) spg_gpu_buffer_free(all[k]);
}

/* The products of a convolution as Spingalett.Gpu.c records them: pass 0 the forward one, 1 the data
   gradient (one per phase of the stride), 2 the weight gradient in `slices` slices. Returns their
   number; vector loads when `vec`, the tile chosen (by timing when `timed`). */
static uint32_t conv_push(const Conv *v, const ConvBuffers *b, int pass, uint32_t slices, SpgGemmPush *p,
                          SpgGemmMode *m, bool vec, bool timed) {
    const uint32_t OH = out_h(v), OW = out_w(v), G = v->groups, CG = v->c / G, OG = v->out / G;
    const uint32_t taps = v->kh * v->kw, K = taps * CG;
    const uint64_t nx = (uint64_t)v->n * v->h * v->w * v->c, ny = (uint64_t)v->n * OH * OW * v->out;
    if (pass == 0) {
        p[0] = (SpgGemmPush){.a = b->x.address, .b = b->w.address, .c = b->y.address, .geo = b->geo.address,
                             .M = v->n * OH * OW, .N = OG, .K = K, .ldb = K, .ldc = v->out, .a_group = CG,
                             .b_group = OG * K, .c_group = OG, .alpha = 1.0f};
        m[0] = (SpgGemmMode){SPG_A_CONV, SPG_B_COL, SPG_EPI_STORE, SPINGALETT_ACT_NONE, G, false, vec && CG % 4 == 0 && v->c % 4 == 0,
                             vec, 0, timed ? ny : 0, mma};
        return 1;
    }
    if (pass == 1) {
        for (uint32_t ph = 0; ph < b->info.phases; ph++) {
            p[ph] = (SpgGemmPush){.a = b->dy.address, .b = b->wt.address + 4u * b->info.phase[ph].first * OG * CG,
                                  .c = b->dx.address, .geo = b->geo.address + 4u * b->info.phase[ph].at,
                                  .M = v->n * b->info.phase[ph].rh * b->info.phase[ph].rw, .N = CG,
                                  .K = b->info.phase[ph].taps * OG, .ldb = CG, .ldc = v->c, .a_group = OG,
                                  .b_group = taps * OG * CG, .c_group = CG, .alpha = 1.0f};
            m[ph] = (SpgGemmMode){SPG_A_CONV, SPG_B_ROW, SPG_EPI_STORE, SPINGALETT_ACT_NONE, G, b->info.phases > 1,
                                  vec && OG % 4 == 0 && v->out % 4 == 0, vec, 0, timed ? nx : 0, mma};
        }
        return b->info.phases;
    }
    const uint32_t pixels = v->n * OH * OW, slice_k = ((pixels + slices - 1) / slices + 7u) & ~7u;
    p[0] = (SpgGemmPush){.a = b->dy.address, .b = b->x.address, .c = slices > 1 ? b->part.address : b->dw.address,
                         .geo = b->geo.address, .M = OG, .N = K, .K = pixels, .lda = v->out, .ldc = K,
                         .a_group = OG, .b_group = CG, .c_group = OG * K, .alpha = 1.0f,
                         .slices = (pixels + slice_k - 1) / slice_k, .slice_k = slice_k};
    m[0] = (SpgGemmMode){SPG_A_COL, SPG_B_CONV, p[0].slices > 1 ? SPG_EPI_PARTIAL : SPG_EPI_STORE, SPINGALETT_ACT_NONE, G, false, vec,
                         vec && CG % 4 == 0 && v->c % 4 == 0, 0, timed ? (uint64_t)v->out * K : 0, mma};
    return 1;
}

static bool conv_setup(const Conv *v, ConvBuffers *b, float *x, float *w, float *dy) {
    const uint32_t OH = out_h(v), OW = out_w(v), G = v->groups, CG = v->c / G, OG = v->out / G;
    const uint32_t taps = v->kh * v->kw, K = taps * CG;
    const size_t nx = (size_t)v->n * v->h * v->w * v->c, ny = (size_t)v->n * OH * OW * v->out, nw = (size_t)v->out * K;
    if (!spg_conv_geometry(NULL, &b->info, v->h, v->w, v->c, OH, OW, v->out, G, v->kh, v->kw, v->sh, v->sw, v->ph,
                           v->pw))
        return false;
    uint32_t *geo = (uint32_t *)malloc(b->info.size * sizeof(uint32_t));
    spg_conv_geometry(geo, &b->info, v->h, v->w, v->c, OH, OW, v->out, G, v->kh, v->kw, v->sh, v->sw, v->ph, v->pw);
    /* the weights regrouped as wtrans.comp does: wt[g][order[tap]][f][c] */
    const uint32_t *order = geo + b->info.order;
    float *wt = (float *)malloc(nw * sizeof(float));
    for (uint32_t f = 0; f < v->out; f++)
        for (uint32_t t = 0; t < taps; t++)
            for (uint32_t c = 0; c < CG; c++)
                wt[(((size_t)(f / OG) * taps + order[t]) * OG + f % OG) * CG + c] = w[(size_t)f * K + t * CG + c];
    bool ok = upload(&b->x, x, nx * 4) && upload(&b->w, w, nw * 4) && upload(&b->wt, wt, nw * 4) &&
              upload(&b->dy, dy, ny * 4) && upload(&b->geo, geo, b->info.size * 4) &&
              spg_gpu_buffer_create(&b->y, ny * 4, false) && spg_gpu_buffer_create(&b->dx, nx * 4, false) &&
              spg_gpu_buffer_create(&b->dw, nw * 4, false) && spg_gpu_buffer_create(&b->part, (size_t)SPG_SPLIT_FLOATS * 4, false);
    free(wt);
    free(geo);
    return ok;
}

/* The partial sums of split weight gradients, added on the host. */
static void add_slices(float *dw, const float *part, uint32_t slices, size_t per_group, uint32_t G) {
    for (uint32_t g = 0; g < G; g++)
        for (size_t i = 0; i < per_group; i++) {
            double sum = 0.0;
            for (uint32_t s = 0; s < slices; s++) sum += part[((size_t)g * slices + s) * per_group + i];
            dw[(size_t)g * per_group + i] = (float)sum;
        }
}

static void test_conv(const Conv *v) {
    const uint32_t OH = out_h(v), OW = out_w(v), G = v->groups, CG = v->c / G, OG = v->out / G;
    const uint32_t K = v->kh * v->kw * CG;
    const size_t nx = (size_t)v->n * v->h * v->w * v->c, ny = (size_t)v->n * OH * OW * v->out, nw = (size_t)v->out * K;
    float *x = malloc(nx * 4), *w = malloc(nw * 4), *dy = malloc(ny * 4), *got = malloc((nx + ny + (size_t)SPG_SPLIT_FLOATS) * 4);
    double *y = malloc(ny * 8), *dx = malloc(nx * 8), *dw = malloc(nw * 8);
    for (size_t i = 0; i < nx; i++) x[i] = uniform();
    for (size_t i = 0; i < nw; i++) w[i] = uniform();
    for (size_t i = 0; i < ny; i++) dy[i] = uniform();
    float *rx = operands(x, nx), *rw = operands(w, nw), *rdy = operands(dy, ny);
    reference(v, rx, rw, rdy, y, dx, dw);
    free(rx); free(rw); free(rdy);
    ConvBuffers b = {0};
    if (!conv_setup(v, &b, x, w, dy)) {
        printf("  conv setup failed\n");
        failures++;
    }
    /* every tile and load width gives the same bits: each output is one chain of fused multiply-adds
       in the order of k (weight gradients: for the same slices) */
    float *first[4] = {malloc(ny * 4), malloc(nx * 4), malloc(nw * 4), malloc(nw * 4)};
    bool seen[4] = {false, false, false, false}, same = true;
    double worst[3] = {0, 0, 0};
    for (uint32_t tile = 0; tile <= spg_gemm_tiles(mma); tile += tile ? tile_step : 1)
        for (int vec = 0; vec < 2; vec++)
            for (int pass = 0; pass < 3; pass++) {
                uint32_t slices = pass == 2 ? (tile % 3 == 0 ? 1u : 3u) : 1u;
                SpgGemmPush p[SPG_MAX_PHASES];
                SpgGemmMode m[SPG_MAX_PHASES];
                uint32_t count = conv_push(v, &b, pass, slices, p, m, vec, tile == 0);
                for (uint32_t k = 0; k < count; k++) m[k].tile = tile;
                if (run_many(p, m, count, 1) == 0.0) { failures++; continue; }
                if (pass == 0) {
                    download(got, &b.y, ny * 4);
                    worst[0] = fmax(worst[0], compare(got, y, ny));
                } else if (pass == 1) {
                    download(got, &b.dx, nx * 4);
                    worst[1] = fmax(worst[1], compare(got, dx, nx));
                } else {
                    if (p[0].slices > 1) {
                        download(got + nw, &b.part, (size_t)p[0].slices * nw * 4);
                        add_slices(got, got + nw, p[0].slices, (size_t)OG * K, G);
                    } else {
                        download(got, &b.dw, nw * 4);
                    }
                    worst[2] = fmax(worst[2], compare(got, dw, nw));
                }
                const int kind = pass < 2 ? pass : (p[0].slices > 1 ? 3 : 2);
                const size_t floats = pass == 0 ? ny : pass == 1 ? nx : nw;
                if (!seen[kind]) memcpy(first[kind], got, floats * 4);
                else same = same && !memcmp(first[kind], got, floats * 4);
                seen[kind] = true;
            }
    for (int k = 0; k < 4; k++) free(first[k]);
    bool ok = worst[0] < 1e-5 && worst[1] < 1e-5 && worst[2] < 1e-5 && same;
    if (!same) printf("  tiles differ in their results\n");
    failures += !ok;
    printf("  %sconv %ux%ux%u -> %u, %ux%u window, stride %ux%u, padding %ux%u, %u groups: forward %.1e, data %.1e, "
           "weights %.1e  %s\n", mma ? "bf16 " : "", v->h, v->w, v->c, v->out, v->kh, v->kw, v->sh, v->sw, v->ph, v->pw, v->groups, worst[0],
           worst[1], worst[2], ok ? "ok" : "FAILED");
    conv_free(&b);
    free(x); free(w); free(dy); free(got); free(y); free(dx); free(dw);
}

/* ------------------------------------------------------------------------- dense layers */

static void test_dense(uint32_t n, uint32_t in, uint32_t out) {
    float *x = malloc((size_t)n * in * 4), *w = malloc((size_t)out * in * 4), *d = malloc((size_t)n * out * 4);
    float *got = malloc(((size_t)n * in + (size_t)n * out + (size_t)out * in) * 4);
    double *y = malloc((size_t)n * out * 8), *dx = malloc((size_t)n * in * 8), *dw = malloc((size_t)out * in * 8);
    for (size_t i = 0; i < (size_t)n * in; i++) x[i] = uniform();
    for (size_t i = 0; i < (size_t)out * in; i++) w[i] = uniform();
    for (size_t i = 0; i < (size_t)n * out; i++) d[i] = uniform();
    float *xr = operands(x, (size_t)n * in), *wr = operands(w, (size_t)out * in), *dr = operands(d, (size_t)n * out);
    for (uint32_t s = 0; s < n; s++)
        for (uint32_t j = 0; j < out; j++) {
            double acc = 0;
            for (uint32_t i = 0; i < in; i++) acc += (double)xr[(size_t)s * in + i] * wr[(size_t)j * in + i];
            y[(size_t)s * out + j] = acc;
        }
    for (uint32_t s = 0; s < n; s++)
        for (uint32_t i = 0; i < in; i++) {
            double acc = 0;
            for (uint32_t j = 0; j < out; j++) acc += (double)dr[(size_t)s * out + j] * wr[(size_t)j * in + i];
            dx[(size_t)s * in + i] = acc;
        }
    for (uint32_t j = 0; j < out; j++)
        for (uint32_t i = 0; i < in; i++) {
            double acc = 0;
            for (uint32_t s = 0; s < n; s++) acc += (double)dr[(size_t)s * out + j] * xr[(size_t)s * in + i];
            dw[(size_t)j * in + i] = acc;
        }
    free(xr); free(wr); free(dr);
    SpgGpuBuffer bx, bw, bd, by, bdx, bdw;
    bool ok = upload(&bx, x, (size_t)n * in * 4) && upload(&bw, w, (size_t)out * in * 4) &&
              upload(&bd, d, (size_t)n * out * 4) && spg_gpu_buffer_create(&by, (size_t)n * out * 4, false) &&
              spg_gpu_buffer_create(&bdx, (size_t)n * in * 4, false) && spg_gpu_buffer_create(&bdw, (size_t)out * in * 4, false);
    double worst[3] = {0, 0, 0};
    for (uint32_t tile = 0; ok && tile <= spg_gemm_tiles(mma); tile += tile ? tile_step : 1)
        for (int vec = 0; vec < 2; vec++) {
            SpgGemmPush p = {.a = bx.address, .b = bw.address, .c = by.address, .M = n, .N = out, .K = in, .lda = in,
                             .ldb = in, .ldc = out, .alpha = 1.0f};
            SpgGemmMode m = {SPG_A_ROW, SPG_B_COL, SPG_EPI_STORE, SPINGALETT_ACT_NONE, 1, false, vec, vec, tile, 0, mma};
            run(&p, &m, 1);
            download(got, &by, (size_t)n * out * 4);
            worst[0] = fmax(worst[0], compare(got, y, (size_t)n * out));
            p = (SpgGemmPush){.a = bd.address, .b = bw.address, .c = bdx.address, .M = n, .N = in, .K = out,
                              .lda = out, .ldb = in, .ldc = in, .alpha = 1.0f};
            m = (SpgGemmMode){SPG_A_ROW, SPG_B_ROW, SPG_EPI_STORE, SPINGALETT_ACT_NONE, 1, false, vec, vec, tile, 0, mma};
            run(&p, &m, 1);
            download(got, &bdx, (size_t)n * in * 4);
            worst[1] = fmax(worst[1], compare(got, dx, (size_t)n * in));
            p = (SpgGemmPush){.a = bd.address, .b = bx.address, .c = bdw.address, .M = out, .N = in, .K = n,
                              .lda = out, .ldb = in, .ldc = in, .alpha = 1.0f};
            m = (SpgGemmMode){SPG_A_COL, SPG_B_ROW, SPG_EPI_STORE, SPINGALETT_ACT_NONE, 1, false, vec, vec, tile, 0, mma};
            run(&p, &m, 1);
            download(got, &bdw, (size_t)out * in * 4);
            worst[2] = fmax(worst[2], compare(got, dw, (size_t)out * in));
        }
    bool good = ok && worst[0] < 1e-5 && worst[1] < 1e-5 && worst[2] < 1e-5;
    failures += !good;
    printf("  %sdense %u x %u -> %u: forward %.1e, data %.1e, weights %.1e  %s\n", mma ? "bf16 " : "", n, in, out, worst[0], worst[1],
           worst[2], good ? "ok" : "FAILED");
    SpgGpuBuffer *all[] = {&bx, &bw, &bd, &by, &bdx, &bdw};
    for (size_t k = 0; k < 6; k++) spg_gpu_buffer_free(all[k]);
    free(x); free(w); free(d); free(got); free(y); free(dx); free(dw);
}

/* ------------------------------------------------------------------------- benchmark */


/* One line of the benchmark: the time of the tile chosen by timing and of the fastest; with
   SPINGALETT_BENCH_TILES set, every tile's, fastest first. */
static void bench_report(const char *name, bool mma, double flops, double chosen, const double *times, uint32_t count) {
    uint32_t order[64], n = 0;
    for (uint32_t t = 0; t < count && n < 64; t++)
        if (times[t] > 0) order[n++] = t;
    for (uint32_t a = 0; a < n; a++)
        for (uint32_t b = a + 1; b < n; b++)
            if (times[order[b]] < times[order[a]]) { uint32_t x = order[a]; order[a] = order[b]; order[b] = x; }
    if (n == 0) return;
    uint32_t bm, bn, bk, tm, tn;
    spg_gemm_tile(mma, order[0], &bm, &bn, &bk, &tm, &tn);
    printf("  %-8s chosen %7.1f us %5.2f TFLOPS   best %7.1f us %5.2f TFLOPS (%ux%u k%u, %ux%u a thread)\n", name,
           chosen * 1e6, flops / chosen * 1e-12, times[order[0]] * 1e6, flops / times[order[0]] * 1e-12, bm, bn, bk, tm,
           tn);
    if (!getenv("SPINGALETT_BENCH_TILES")) return;
    for (uint32_t k = 0; k < n; k++) {
        spg_gemm_tile(mma, order[k], &bm, &bn, &bk, &tm, &tn);
        printf("           %7.1f us  %3ux%-3u k%-2u %ux%u\n", times[order[k]] * 1e6, bm, bn, bk, tm, tn);
    }
}
static void bench(bool dense_only) {
    static const Conv shapes[] = {
        {128, 32, 32, 16, 16, 3, 3, 1, 1, 1, 1, 1}, {128, 32, 32, 16, 32, 3, 3, 2, 2, 1, 1, 1},
        {128, 16, 16, 32, 32, 3, 3, 1, 1, 1, 1, 1}, {128, 16, 16, 32, 64, 3, 3, 2, 2, 1, 1, 1},
        {128, 8, 8, 64, 64, 3, 3, 1, 1, 1, 1, 1}, {128, 32, 32, 3, 16, 3, 3, 1, 1, 1, 1, 1},
    };
    static const char *names[] = {"forward", "data", "weights"};
    /* SPINGALETT_BENCH_CONV="n h w c out kh kw sh sw ph pw groups": that convolution only */
    Conv only;
    const char *one = getenv("SPINGALETT_BENCH_CONV");
    const bool single = one && sscanf(one, "%u %u %u %u %u %u %u %u %u %u %u %u", &only.n, &only.h, &only.w, &only.c, &only.out,
                                      &only.kh, &only.kw, &only.sh, &only.sw, &only.ph, &only.pw, &only.groups) == 12;
    for (size_t k = 0; !dense_only && k < (single ? 1u : sizeof shapes / sizeof shapes[0]); k++) {
        const Conv *v = single ? &only : &shapes[k];
        const uint32_t OH = out_h(v), OW = out_w(v), K = v->kh * v->kw * v->c / v->groups;
        const size_t nx = (size_t)v->n * v->h * v->w * v->c, ny = (size_t)v->n * OH * OW * v->out,
                     nw = (size_t)v->out * K;
        float *x = malloc(nx * 4), *w = malloc(nw * 4), *dy = malloc(ny * 4);
        for (size_t i = 0; i < nx; i++) x[i] = uniform();
        for (size_t i = 0; i < nw; i++) w[i] = uniform();
        for (size_t i = 0; i < ny; i++) dy[i] = uniform();
        ConvBuffers b = {0};
        conv_setup(v, &b, x, w, dy);
        /* with "half": the maps kept as bfloat16 (the floats' high halves), the filters as floats, as in training */
        for (int map = 0; half && map < 2; map++) {
            const float *src = map ? dy : x;
            const size_t count = map ? ny : nx;
            uint16_t *h16 = malloc(count * 2);
            for (size_t i = 0; i < count; i++) {
                uint32_t u;
                memcpy(&u, &src[i], 4);
                h16[i] = (uint16_t)(u >> 16);
            }
            SpgGpuBuffer *dst = map ? &b.dy : &b.x;
            spg_gpu_buffer_free(dst);
            upload(dst, h16, count * 2);
            free(h16);
        }
        const double flops = 2.0 * (double)ny * K;
        printf("conv %ux%ux%u -> %ux%ux%u, %ux%u window, stride %u (%.0f MFLOP a product)\n", v->h, v->w, v->c, OH, OW,
               v->out, v->kh, v->kw, v->sh, flops * 1e-6);
        for (int pass = 0; pass < 3; pass++) {
            double chosen = 0, times[64] = {0};
            for (uint32_t tile = 0; tile <= spg_gemm_tiles(mma); tile++) {
                SpgGemmPush p[SPG_MAX_PHASES];
                SpgGemmMode m[SPG_MAX_PHASES];
                uint32_t slices = 1, slice_k;
                if (pass == 2)          /* the split Spingalett.Gpu.c makes */
                    slices = spg_gemm_split(v->out / v->groups, K, v->n * OH * OW, v->groups, &slice_k);
                /* tile 0: chosen by timing (as in training); the others forced */
                uint32_t count = conv_push(v, &b, pass, slices, p, m, true, tile == 0);
                const uint32_t CG = v->c / v->groups, OG = v->out / v->groups;
                for (uint32_t j = 0; j < count; j++) {
                    m[j].tile = tile;
                    if (!half) continue;
                    /* A, and C of the forward pass and data gradient, as bfloat16; with the weight gradient's B */
                    m[j].half = pass == 2 ? 3u : 5u;
                    m[j].wide_a = pass == 0 ? CG % 8u == 0 && v->c % 8u == 0 : pass == 1 ? OG % 8u == 0 && v->out % 8u == 0 : true;
                    m[j].wide_b = pass == 2 ? CG % 8u == 0 && v->c % 8u == 0 : true;
                }
                run_many(p, m, count, 2);
                double t = run_many(p, m, count, 20);
                if (tile == 0) chosen = t;
                else if (tile <= 64) times[tile - 1] = t;
            }
            bench_report(names[pass], mma, flops, chosen, times, spg_gemm_tiles(mma));
        }
        conv_free(&b);
        free(x); free(w); free(dy);
    }
    /* the MLP of Examples/Benchmark.c: products of 2048 rows, forward, data and weight gradients */
    static const uint32_t dense[][2] = {{784, 512}, {512, 1000}};
    for (size_t k = 0; k < 2; k++) {
        const uint32_t n = 2048, in = dense[k][0], out = dense[k][1];
        const double flops = 2.0 * n * in * out;
        SpgGpuBuffer bx, bw, bd, by;
        float *host = malloc((size_t)n * (in > out ? in : out) * 4);
        for (size_t i = 0; i < (size_t)n * (in > out ? in : out); i++) host[i] = uniform();
        /* (kept as bfloat16: the floats' high halves, the bytes the products read) */
        const size_t size = half ? 2u : 4u;
        uint16_t *h16 = malloc((size_t)n * (in > out ? in : out) * 2);
        for (size_t i = 0; half && i < (size_t)n * (in > out ? in : out); i++) {
            uint32_t u;
            memcpy(&u, &host[i], 4);
            h16[i] = (uint16_t)(u >> 16);
        }
        const void *src = half ? (const void *)h16 : (const void *)host;
        upload(&bx, src, (size_t)n * in * size);
        if (float_weights) upload(&bw, host, (size_t)out * in * 4u);
        else upload(&bw, src, (size_t)out * in * size);
        upload(&bd, src, (size_t)n * out * size);
        spg_gpu_buffer_create(&by, (size_t)(n > out ? n : out) * (in > out ? in : out) * 4, false);
        printf("dense %u x %u -> %u (%.0f MFLOP a product)\n", n, in, out, flops * 1e-6);
        static const char *names[] = {"forward", "data", "weights"};
        for (int pass = 0; pass < 3; pass++) {
            double chosen = 0, times[64] = {0};
            for (uint32_t tile = 0; tile <= spg_gemm_tiles(mma); tile++) {
                SpgGemmPush p;
                SpgGemmMode m;
                if (pass == 0) {
                    p = (SpgGemmPush){.a = bx.address, .b = bw.address, .c = by.address, .M = n, .N = out, .K = in,
                                      .lda = in, .ldb = in, .ldc = out, .alpha = 1.0f};
                    m = (SpgGemmMode){SPG_A_ROW, SPG_B_COL, SPG_EPI_STORE, SPINGALETT_ACT_NONE, 1, false, true, true, tile,
                                      (uint64_t)n * out, mma};
                } else if (pass == 1) {
                    p = (SpgGemmPush){.a = bd.address, .b = bw.address, .c = by.address, .M = n, .N = in, .K = out,
                                      .lda = out, .ldb = in, .ldc = in, .alpha = 1.0f};
                    m = (SpgGemmMode){SPG_A_ROW, SPG_B_ROW, SPG_EPI_STORE, SPINGALETT_ACT_NONE, 1, false, true, true, tile,
                                      (uint64_t)n * in, mma};
                } else {
                    p = (SpgGemmPush){.a = bd.address, .b = bx.address, .c = by.address, .M = out, .N = in, .K = n,
                                      .lda = out, .ldb = in, .ldc = in, .alpha = 1.0f};
                    m = (SpgGemmMode){SPG_A_COL, SPG_B_ROW, SPG_EPI_STORE, SPINGALETT_ACT_NONE, 1, false, true, true, tile,
                                      (uint64_t)out * in, mma};
                }
                /* kept as bfloat16: A and B, and C but the weight gradients' */
                m.half = half ? (pass == 2 ? 3u : float_weights ? 5u : 7u) : 0u;
                m.wide_a = m.wide_b = true;
                run(&p, &m, 2);
                double t = run(&p, &m, 20);
                if (tile == 0) chosen = t;
                else if (tile <= 64) times[tile - 1] = t;
            }
            bench_report(names[pass], mma, flops, chosen, times, spg_gemm_tiles(mma));
        }
        SpgGpuBuffer *all[] = {&bx, &bw, &bd, &by};
        for (size_t j = 0; j < 4; j++) spg_gpu_buffer_free(all[j]);
        free(host);
        free(h16);
    }
}

int main(int argc, char **argv) {
    const char *backend = getenv("SPINGALETT_GPU_BACKEND");
    const bool cuda = backend && !strcmp(backend, "cuda");
    spg_gpu_use(cuda ? SPG_BACKEND_CUDA : SPG_BACKEND_VULKAN);
    if (!spg_gpu_open()) {
        printf("no usable %s device: GPU tests skipped\n", cuda ? "CUDA" : "Vulkan");
        return 77;
    }
    printf("device: %s (%s)\n", spg_gpu_device_name(), cuda ? "CUDA" : "Vulkan");
    printf("matrix units: %s\n", spg_gpu_mma_bf16() ? "bfloat16 cooperative matrices" : "none");
    if (argc > 1 && !strcmp(argv[1], "bench")) {
        bool dense = false;
        for (int k = 2; k < argc; k++) {
            if (!strcmp(argv[k], "bf16")) mma = spg_gpu_mma_bf16();
            if (!strcmp(argv[k], "dense")) dense = true;
            if (!strcmp(argv[k], "half")) half = spg_gpu_mma_bf16() && spg_gpu_bf16_storage();
            if (!strcmp(argv[k], "fw")) float_weights = true;
        }
        bench(dense);
        return 0;
    }
    if (argc > 1 && !strcmp(argv[1], "all")) tile_step = 1;
    static const Conv convs[] = {
        {2, 6, 7, 4, 8, 3, 3, 1, 1, 1, 1, 1}, {2, 9, 8, 16, 16, 3, 3, 2, 2, 1, 1, 1},
        {3, 5, 6, 3, 5, 2, 3, 1, 2, 0, 1, 1}, {2, 8, 8, 8, 12, 3, 3, 1, 1, 1, 1, 2},
        {2, 7, 7, 6, 6, 3, 3, 1, 1, 1, 1, 6}, {1, 10, 9, 32, 16, 1, 1, 2, 2, 0, 0, 1},
        {4, 6, 6, 16, 24, 5, 5, 1, 1, 2, 2, 4}, {2, 12, 12, 12, 8, 3, 3, 3, 2, 1, 0, 1},
    };
    for (int pass = 0; pass < (spg_gpu_mma_bf16() ? 2 : 1); pass++) {
        mma = pass == 1;
        for (size_t k = 0; k < sizeof convs / sizeof convs[0]; k++) test_conv(&convs[k]);
        test_dense(37, 50, 21);
        test_dense(64, 784, 128);
        test_dense(5, 3, 7);
    }
    printf(failures ? "%d FAILED\n" : "ALL PASSED\n", failures);
    return failures != 0;
}
