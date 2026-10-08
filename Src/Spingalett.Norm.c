/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/*
 * Batch normalization over batches of channels-last tensors: the n samples of a layer are
 * n x height x width rows of `channels` floats, and every channel is normalized over all of its
 * rows. Weight layer l (layer l + 1 normalizing layer l's output) holds gamma as its weights and
 * beta as its biases.
 *
 * Per-channel sums run over fixed blocks of rows and channels whose partial sums are added in a
 * fixed order (blocks in float, totals in double), so they come out the same on any number of
 * threads. Element-wise passes (normalization, the data gradient) apply activations and their
 * derivatives sample by sample, so a sample's result does not depend on the rest of its batch.
 */

#include "Spingalett.Private.h"
#include <math.h>
#include <string.h>

#if defined(_OPENMP)
#include <omp.h>
#endif

#define SLAB        64u         /* channels per work item of a reduction */
#define BLOCK       64u         /* rows summed in float before they are added in double */
#define MAX_CHUNKS  64u         /* row ranges a reduction is split into, at most */
#define MIN_CHUNK   256u        /* rows per range, at least */

/* Rows per range of a reduction over m rows: depends on m only, never on the thread count. */
static uint64_t chunk_rows(uint64_t m) {
    uint64_t rows = (m + MAX_CHUNKS - 1) / MAX_CHUNKS;
    return rows < MIN_CHUNK ? MIN_CHUNK : rows;
}

size_t spingalett_bn_scratch_doubles(const NeuralNetwork *net, uint32_t capacity) {
    size_t need = 0;
    for (uint32_t l = 1; l < net->layers; l++) {
        const LayerShape *s = &net->shapes[l];
        if (s->type != LAYER_BATCH_NORM) continue;
        uint64_t m = (uint64_t)capacity * s->height * s->width;
        uint64_t chunks = (m + chunk_rows(m) - 1) / chunk_rows(m);
        size_t d = (size_t)chunks * 2u * s->channels;
        if (d > need) need = d;
    }
    return need;
}

/* Sums over rows [r0, r1) of channels [c0, c0 + w) (rows C floats apart) into t1 and t2: (a - k)
   and (a - k)^2, or b and b (a - k) when b is given; floats over BLOCK rows, added to the doubles.
   Slabs of 16, 32 and 64 channels get loops of constant width, which keep the sums in registers. */
#define BLOCK_SUMS(W)                                                                               \
    for (uint64_t r = r0; r < r1; r += BLOCK) {                                                     \
        uint64_t re = r + BLOCK < r1 ? r + BLOCK : r1;                                              \
        float f1[SLAB] = {0}, f2[SLAB] = {0};                                                       \
        if (b) {                                                                                    \
            for (uint64_t i = r; i < re; i++) {                                                     \
                const float *x = a + i * C + c0, *g = b + i * C + c0;                               \
                for (uint32_t c = 0; c < (W); c++) { f1[c] += g[c]; f2[c] += g[c] * (x[c] - k[c]); } \
            }                                                                                       \
        } else {                                                                                    \
            for (uint64_t i = r; i < re; i++) {                                                     \
                const float *x = a + i * C + c0;                                                    \
                for (uint32_t c = 0; c < (W); c++) { float d = x[c] - k[c]; f1[c] += d; f2[c] += d * d; } \
            }                                                                                       \
        }                                                                                           \
        for (uint32_t c = 0; c < (W); c++) { t1[c] += f1[c]; t2[c] += f2[c]; }                       \
    }

static void block_sums(const float *restrict a, const float *restrict b, const float *restrict k, uint64_t r0,
                       uint64_t r1, uint32_t C, uint32_t c0, uint32_t w, double *restrict t1, double *restrict t2) {
    switch (w) {
        case 16: BLOCK_SUMS(16) break;
        case 32: BLOCK_SUMS(32) break;
        case 64: BLOCK_SUMS(64) break;
        default: BLOCK_SUMS(w) break;
    }
}

/* Per channel, over m rows of C floats: s1 = sum of (a - shift), s2 = sum of (a - shift)^2 when b is
   NULL; otherwise s1 = sum of b, s2 = sum of b * (a - shift). Totals in double, out[0..C) and
   out[C..2C). */
static void channel_sums(const float *a, const float *b, const float *shift, uint64_t m, uint32_t C,
                         double *partial, double *out, ComputeMode mode) {
    const uint64_t rows = chunk_rows(m), chunks = (m + rows - 1) / rows;
    const uint32_t slabs = (C + SLAB - 1) / SLAB;
    const int64_t items = (int64_t)(chunks * slabs);
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if(spingalett_use_omp(mode, m * C) && items > 1)
#endif
    for (int64_t it = 0; it < items; it++) {
        uint64_t chunk = (uint64_t)it / slabs;
        uint32_t s0 = (uint32_t)((uint64_t)it % slabs) * SLAB, sw = C - s0 < SLAB ? C - s0 : SLAB;
        uint64_t r0 = chunk * rows, r1 = r0 + rows < m ? r0 + rows : m;
        double *p = partial + chunk * 2u * C, t1[SLAB] = {0}, t2[SLAB] = {0};
        block_sums(a, b, shift + s0, r0, r1, C, s0, sw, t1, t2);
        memcpy(p + s0, t1, sw * sizeof(double));
        memcpy(p + C + s0, t2, sw * sizeof(double));
    }
    (void)mode;
    for (uint32_t c = 0; c < 2u * C; c++) out[c] = 0.0;
    for (uint64_t chunk = 0; chunk < chunks; chunk++)
        for (uint32_t c = 0; c < 2u * C; c++) out[c] += partial[chunk * 2u * C + c];
}

/* Element-wise passes over rows of C channels with per-channel coefficients, of constant width for
   16, 32 and 64 channels. */
#define ROWS_BY_WIDTH(C, BODY)                                                                      \
    switch (C) {                                                                                    \
        case 16: { const uint32_t W = 16; BODY } break;                                             \
        case 32: { const uint32_t W = 32; BODY } break;                                             \
        case 64: { const uint32_t W = 64; BODY } break;                                             \
        default: { const uint32_t W = (C); BODY } break;                                            \
    }

/* y = act(x * a + b) per channel, sample by sample (rows of `per` floats, C channels each); ReLU
   is applied in the same pass. */
static void affine_activate(const float *x, float *y, uint32_t n, uint64_t per, uint32_t C, const float *a,
                            const float *b, ActivationFunction act, ComputeMode mode) {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if(spingalett_use_omp(mode, (uint64_t)n * per) && n > 1)
#endif
    for (int64_t s = 0; s < (int64_t)n; s++) {
        const float *restrict xs = x + (uint64_t)s * per;
        float *restrict ys = y + (uint64_t)s * per;
        if (act == ACT_RELU) {
            ROWS_BY_WIDTH(C, for (uint64_t i = 0; i < per; i += W) for (uint32_t c = 0; c < W; c++) {
                float v = xs[i + c] * a[c] + b[c];
                ys[i + c] = v > 0.0f ? v : 0.0f;
            })
            continue;
        }
        ROWS_BY_WIDTH(C, for (uint64_t i = 0; i < per; i += W) for (uint32_t c = 0; c < W; c++)
                             ys[i + c] = xs[i + c] * a[c] + b[c];)
        apply_activation_batch(ys, (uint32_t)per, act);
    }
    (void)mode;
}

void spingalett_bn_forward_train(NeuralNetwork *net, uint32_t l, const float *x, float *y, uint32_t n,
                                 ActivationFunction act, float *stats, double *scratch, ComputeMode mode) {
    const LayerShape *s = &net->shapes[l + 1];
    const uint32_t C = s->channels;
    const uint64_t per = (uint64_t)s->height * s->width * C, m = (uint64_t)n * s->height * s->width;
    const float *gamma = net->weights + net->weight_offsets[l], *beta = net->biases + net->bias_offsets[l];
    float *rm = net->running_mean + net->bias_offsets[l], *rv = net->running_var + net->bias_offsets[l];
    float *mean = stats, *inv = stats + C;
    /* the first row shifts the sums, which keeps the variance accurate when the mean is large */
    double *sums = scratch, *partial = scratch + 2u * C;
    channel_sums(x, NULL, x, m, C, partial, sums, mode);
    for (uint32_t c = 0; c < C; c++) {
        double d = sums[c] / (double)m, var = sums[C + c] / (double)m - d * d;
        if (var < 0.0) var = 0.0;
        double mu = (double)x[c] + d;
        mean[c] = (float)mu;
        inv[c] = (float)(1.0 / sqrt(var + (double)s->eps));
        rm[c] += s->momentum * ((float)mu - rm[c]);
        if (m > 1)                          /* the running variance is unbiased */
            rv[c] += s->momentum * ((float)(var * (double)m / (double)(m - 1)) - rv[c]);
    }
    /* the normalization's coefficients after the statistics: a = gamma / std, b = beta - mean * a */
    float *ca = stats + 2u * C, *cb = stats + 3u * C;
    for (uint32_t c = 0; c < C; c++) {
        ca[c] = gamma[c] * inv[c];
        cb[c] = beta[c] - mean[c] * ca[c];
    }
    affine_activate(x, y, n, per, C, ca, cb, act, mode);
}

void spingalett_bn_forward(const NeuralNetwork *net, uint32_t l, const float *x, float *y, uint32_t n,
                           ActivationFunction act, float *coef, ComputeMode mode) {
    const LayerShape *s = &net->shapes[l + 1];
    const uint32_t C = s->channels;
    const uint64_t per = (uint64_t)s->height * s->width * C;
    const uint64_t b0 = net->bias_offsets[l];
    spingalett_bn_coefficients(net->weights + net->weight_offsets[l], net->biases + b0, net->running_mean + b0,
                               net->running_var + b0, s->eps, C, coef, coef + C);
    affine_activate(x, y, n, per, C, coef, coef + C, act, mode);
}

void spingalett_bn_backward_sums(const NeuralNetwork *net, uint32_t l, const float *x, const float *dy, uint32_t n,
                                 const float *stats, double *sums, double *scratch, ComputeMode mode) {
    const LayerShape *s = &net->shapes[l + 1];
    const uint32_t C = s->channels;
    const uint64_t m = (uint64_t)n * s->height * s->width;
    channel_sums(x, dy, stats, m, C, scratch, sums, mode);
    /* sum of dy * xhat = sum of dy * (x - mean) / std */
    for (uint32_t c = 0; c < C; c++) sums[C + c] *= (double)stats[C + c];
}

void spingalett_bn_backward_data(const NeuralNetwork *net, uint32_t l, const float *x, const float *dy, float *dx,
                                 uint32_t n, const float *stats, const double *sums, ActivationFunction act,
                                 float *coef, ComputeMode mode) {
    const LayerShape *s = &net->shapes[l + 1];
    const uint32_t C = s->channels;
    const uint64_t per = (uint64_t)s->height * s->width * C, m = (uint64_t)n * s->height * s->width;
    const float *gamma = net->weights + net->weight_offsets[l];
    /* dx = gamma / std * (dy - mean(dy) - xhat * mean(dy * xhat)) = k dy + p x + q */
    float *k = coef, *p = coef + C, *q = coef + 2u * C;
    for (uint32_t c = 0; c < C; c++) {
        double inv = stats[C + c], kk = (double)gamma[c] * inv;
        double m1 = sums[c] / (double)m, m2 = sums[C + c] / (double)m;
        k[c] = (float)kk;
        p[c] = (float)(-kk * inv * m2);
        q[c] = (float)(-kk * m1 + kk * inv * m2 * (double)stats[c]);
    }
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if(spingalett_use_omp(mode, (uint64_t)n * per) && n > 1)
#endif
    for (int64_t smp = 0; smp < (int64_t)n; smp++) {
        const float *restrict xs = x + (uint64_t)smp * per, *restrict gs = dy + (uint64_t)smp * per;
        float *restrict ds = dx + (uint64_t)smp * per;
        ROWS_BY_WIDTH(C, for (uint64_t i = 0; i < per; i += W) for (uint32_t c = 0; c < W; c++)
                             ds[i + c] = k[c] * gs[i + c] + (p[c] * xs[i + c] + q[c]);)
        if (act != ACT_NONE) apply_derivative_batch(ds, xs, per, act);
    }
    (void)mode;
}

/* ---- layer normalization: each cell over its channels ---- */

size_t spingalett_ln_scratch_doubles(const NeuralNetwork *net, uint32_t capacity) {
    size_t need = 0;
    for (uint32_t l = 1; l < net->layers; l++) {
        const LayerShape *s = &net->shapes[l];
        if (s->type != LAYER_LAYER_NORM) continue;
        uint64_t m = (uint64_t)capacity * s->height * s->width;
        uint64_t chunks = (m + chunk_rows(m) - 1) / chunk_rows(m);
        size_t d = (size_t)chunks * 2u * s->channels;
        if (d > need) need = d;
    }
    return need;
}

void spingalett_ln_forward(const NeuralNetwork *net, uint32_t l, const float *x, float *y, uint32_t n,
                           ActivationFunction act, float *stats, ComputeMode mode) {
    const LayerShape *s = &net->shapes[l + 1];
    const uint32_t C = s->channels, cells = s->height * s->width;
    const uint64_t per = (uint64_t)cells * C;
    const float *gamma = net->weights + net->weight_offsets[l], *beta = net->biases + net->bias_offsets[l];
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if(spingalett_use_omp(mode, (uint64_t)n * per) && n > 1)
#endif
    for (int64_t smp = 0; smp < (int64_t)n; smp++) {
        float *ys = y + (uint64_t)smp * per;
        spingalett_engine_layer_norm(x + (uint64_t)smp * per, cells, C, gamma, beta, s->eps, ys,
                                     stats ? stats + (uint64_t)smp * 2u * cells : NULL);
        if (act != ACT_NONE && act != ACT_SOFTMAX) apply_activation_batch(ys, (uint32_t)per, act);
    }
    (void)mode;
}

/* dx = rstd (g - (sum g + xhat sum g xhat) / C) per cell, g = gamma dy, xhat = (x - mean) rstd; then
   times the derivative of the input's activation */
void spingalett_ln_backward_data(const NeuralNetwork *net, uint32_t l, const float *x, const float *dy, float *dx,
                                 uint32_t n, const float *stats, ActivationFunction act, ComputeMode mode) {
    const LayerShape *s = &net->shapes[l + 1];
    const uint32_t C = s->channels, cells = s->height * s->width;
    const uint64_t per = (uint64_t)cells * C;
    const float *gamma = net->weights + net->weight_offsets[l], inv = 1.0f / (float)C;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if(spingalett_use_omp(mode, (uint64_t)n * per) && n > 1)
#endif
    for (int64_t smp = 0; smp < (int64_t)n; smp++) {
        const float *xs = x + (uint64_t)smp * per, *gs = dy + (uint64_t)smp * per, *st = stats + (uint64_t)smp * 2u * cells;
        float *ds = dx + (uint64_t)smp * per;
        for (uint32_t p = 0; p < cells; p++) {
            const float mean = st[2u * p], rstd = st[2u * p + 1u];
            const float *v = xs + (uint64_t)p * C, *g = gs + (uint64_t)p * C;
            float *d = ds + (uint64_t)p * C, a = 0.0f, b = 0.0f;
            for (uint32_t c = 0; c < C; c++) {
                float gc = gamma[c] * g[c];
                a += gc;
                b += gc * ((v[c] - mean) * rstd);
            }
            for (uint32_t c = 0; c < C; c++) {
                float xhat = (v[c] - mean) * rstd;
                d[c] = rstd * (gamma[c] * g[c] - (a + xhat * b) * inv);
            }
        }
        if (act != ACT_NONE) apply_derivative_batch(ds, xs, per, act);
    }
    (void)mode;
}

/* gamma's gradient, sum of dy xhat, and beta's, sum of dy, over every cell of the batch: ranges of
   cells fixed by their number, summed in float over blocks and in double over the ranges, so that
   the result does not depend on the thread count; g = scale * sum + beta_g * g */
void spingalett_ln_backward_params(NeuralNetwork *net, uint32_t l, const float *x, const float *dy, uint32_t n,
                                   const float *stats, float scale, float beta_g, double *partial, ComputeMode mode) {
    const LayerShape *s = &net->shapes[l + 1];
    const uint32_t C = s->channels;
    const uint64_t m = (uint64_t)n * s->height * s->width, rows = chunk_rows(m), chunks = (m + rows - 1) / rows;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if(spingalett_use_omp(mode, m * C) && chunks > 1)
#endif
    for (int64_t chunk = 0; chunk < (int64_t)chunks; chunk++) {
        const uint64_t r0 = (uint64_t)chunk * rows, r1 = r0 + rows < m ? r0 + rows : m;
        double *t1 = partial + (uint64_t)chunk * 2u * C, *t2 = t1 + C;
        for (uint32_t c = 0; c < C; c++) t1[c] = t2[c] = 0.0;
        for (uint64_t r = r0; r < r1; r += BLOCK) {
            const uint64_t re = r + BLOCK < r1 ? r + BLOCK : r1;
            for (uint32_t c0 = 0; c0 < C; c0 += SLAB) {
                const uint32_t w = C - c0 < SLAB ? C - c0 : SLAB;
                float f1[SLAB] = {0}, f2[SLAB] = {0};
                for (uint64_t i = r; i < re; i++) {
                    const float mean = stats[2u * i], rstd = stats[2u * i + 1u];
                    const float *v = x + i * C + c0, *g = dy + i * C + c0;
                    for (uint32_t c = 0; c < w; c++) { f1[c] += g[c]; f2[c] += g[c] * ((v[c] - mean) * rstd); }
                }
                for (uint32_t c = 0; c < w; c++) { t1[c0 + c] += f1[c]; t2[c0 + c] += f2[c]; }
            }
        }
    }
    (void)mode;
    float *gW = net->grad_weights + net->weight_offsets[l], *gB = net->grad_biases + net->bias_offsets[l];
    for (uint32_t c = 0; c < C; c++) {
        double sb = 0.0, sg = 0.0;
        for (uint64_t chunk = 0; chunk < chunks; chunk++) {
            sb += partial[chunk * 2u * C + c];
            sg += partial[chunk * 2u * C + C + c];
        }
        float dg = (float)sg * scale, db = (float)sb * scale;
        gW[c] = beta_g == 0.0f ? dg : dg + beta_g * gW[c];
        gB[c] = beta_g == 0.0f ? db : db + beta_g * gB[c];
    }
}
