/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/*
 * The layers of transformers over batches (training and predict() on the CPU): embeddings and
 * attention. RMS normalization lives with the other normalizations (Spingalett.Norm.c), products of
 * layers with the layers that combine others (Spingalett.Graph.c), the engine's per-sample passes in
 * Spingalett.Inference.c.
 *
 * Attention runs a task per sample and group of heads sharing keys and values: the queries in blocks
 * of ATTN_BLOCK rows, their scores against the keys as a matrix product (the keys up to the block's
 * last row when causal), a softmax per row, and the values weighted by it as another product. The
 * backward pass recomputes the scores from each query's log of the softmax's sum (kept by the forward
 * pass) instead of storing them: per block dP = dO V^T, dS = P (dP - rowsum(P dP)), dQ = dS K / sqrt(d),
 * and dK, dV summed over the blocks and the group's heads in their order. A task is one thread's
 * work from start to end, on a matrix product scratch of its own, so that every sum is taken in an
 * order fixed by the shape, whatever the thread count.
 */

#include "Spingalett.Private.h"
#include <math.h>
#include <string.h>
#include <float.h>

#define ATTN_BLOCK 64u              /* query rows a block */
#define EMBED_SLAB 16u              /* columns of an embedding's gradient a work item */

/* ------------------------------------------------------------------------- embeddings */

void spingalett_embedding_forward(const NeuralNetwork *net, uint32_t l, const float *x, float *y, uint32_t n,
                                  ActivationFunction act, ComputeMode mode) {
    const LayerShape *s = &net->shapes[l + 1];
    const uint32_t tokens = net->topology[spingalett_source(net, l + 1)], d = s->channels, V = s->vocabulary;
    const uint64_t per = (uint64_t)tokens * d;
    const float *W = net->weights + net->weight_offsets[l];
    const float *pos = s->positions ? net->biases + net->bias_offsets[l] : NULL;
    SPINGALETT_PARALLEL_FOR(spingalett_use_omp(mode, (uint64_t)n * per) && n > 1,
        for (int64_t smp = 0; smp < (int64_t)n; smp++) {
            const float *xs = x + (uint64_t)smp * tokens;
            float *ys = y + (uint64_t)smp * per;
            for (uint32_t p = 0; p < tokens; p++) {
                float *o = ys + (uint64_t)p * d;
                const float t = xs[p];
                if (t >= 0.0f && t < (float)V) memcpy(o, W + (uint64_t)(uint32_t)t * d, (size_t)d * sizeof(float));
                else memset(o, 0, (size_t)d * sizeof(float));
                if (pos) spingalett_vec_axpy(o, pos + (uint64_t)p * d, d, 1.0f);
            }
            if (act != ACT_NONE && act != ACT_SOFTMAX) apply_activation_batch(ys, (uint32_t)per, act);
        }
    );
    (void)mode;
}

/* The table's gradient by slabs of EMBED_SLAB columns, each a thread's: every token of the batch in its
   order adds its row of dy; the positions' gradient by position, the samples in their order. */
void spingalett_embedding_backward(NeuralNetwork *net, uint32_t l, const float *x, const float *dy, uint32_t n,
                                   float scale, float beta, ComputeMode mode) {
    const LayerShape *s = &net->shapes[l + 1];
    const uint32_t tokens = net->topology[spingalett_source(net, l + 1)], d = s->channels, V = s->vocabulary;
    const uint64_t per = (uint64_t)tokens * d, table = (uint64_t)V * d;
    float *gW = net->grad_weights + net->weight_offsets[l];
    if (beta == 0.0f) memset(gW, 0, (size_t)table * sizeof(float));
    else if (beta != 1.0f) spingalett_vec_scale(gW, table, beta);
    const uint32_t slabs = (d + EMBED_SLAB - 1u) / EMBED_SLAB;
    SPINGALETT_PARALLEL_FOR(spingalett_use_omp(mode, (uint64_t)n * per) && slabs > 1,
        for (int64_t slab = 0; slab < (int64_t)slabs; slab++) {
            const uint32_t c0 = (uint32_t)slab * EMBED_SLAB, w = d - c0 < EMBED_SLAB ? d - c0 : EMBED_SLAB;
            for (uint32_t smp = 0; smp < n; smp++)
                for (uint32_t p = 0; p < tokens; p++) {
                    const float t = x[(uint64_t)smp * tokens + p];
                    if (!(t >= 0.0f && t < (float)V)) continue;
                    float *g = gW + (uint64_t)(uint32_t)t * d + c0;
                    const float *v = dy + (uint64_t)smp * per + (uint64_t)p * d + c0;
                    for (uint32_t c = 0; c < w; c++) g[c] += scale * v[c];
                }
        }
    );
    if (s->positions) {
        float *gP = net->grad_biases + net->bias_offsets[l];
        SPINGALETT_PARALLEL_FOR(spingalett_use_omp(mode, (uint64_t)n * per) && tokens > 1,
            for (int64_t p = 0; p < (int64_t)tokens; p++) {
                float *g = gP + (uint64_t)p * d;
                for (uint32_t c = 0; c < d; c++) {
                    float sum = 0.0f;
                    for (uint32_t smp = 0; smp < n; smp++) sum += dy[(uint64_t)smp * per + (uint64_t)p * d + c];
                    g[c] = beta == 0.0f ? scale * sum : scale * sum + beta * g[c];
                }
            }
        );
    }
    (void)mode;
}

/* ------------------------------------------------------------------------- attention */

typedef struct {
    uint32_t cells, heads, kv, head, half, C, out_c, group;
    float scale;
    bool causal;
} AttnShape;

static AttnShape attn_shape(const NeuralNetwork *net, uint32_t l) {
    const LayerShape *s = &net->shapes[l + 1];
    AttnShape a;
    a.cells = s->height * s->width;
    a.heads = s->heads;
    a.kv = s->kv_heads;
    a.head = s->channels / s->heads;
    a.half = a.head / 2u;
    a.C = (a.heads + 2u * a.kv) * a.head;
    a.out_c = s->channels;
    a.group = a.heads / a.kv;
    a.scale = 1.0f / sqrtf((float)a.head);
    a.causal = s->causal;
    return a;
}

size_t spingalett_attention_scratch(const NeuralNetwork *net, uint32_t l, bool training) {
    const AttnShape a = attn_shape(net, l);
    const size_t B = ATTN_BLOCK, T = a.cells, d = a.head;
    /* scores (and their gradient), a block of rotated queries (and its gradient), rotated keys (and the
       sums of the keys' and values' gradients) */
    return training ? 2u * B * T + 2u * B * d + 3u * T * d + B + 16u : B * T + B * d + T * d + 16u;
}

/* Rows of head vectors rotated by their positions' angles (inverse: by minus them, the transpose),
   row r at position p0 + r. */
static void rope_rows(const float *src, size_t lds, float *dst, size_t ldd, uint32_t rows, uint32_t p0, uint32_t half,
                      const float *table, uint32_t cells, bool inverse) {
    for (uint32_t r = 0; r < rows; r++) {
        const float *a = src + r * lds, *cs = table + (size_t)(p0 + r) * half;
        const float *sn = table + (size_t)cells * half + (size_t)(p0 + r) * half;
        float *o = dst + r * ldd;
        for (uint32_t i = 0; i < half; i++) {
            const float u = a[i], v = a[i + half], c = cs[i], sv = inverse ? -sn[i] : sn[i];
            o[i] = u * c - v * sv;
            o[i + half] = v * c + u * sv;
        }
    }
}

/* Softmax of a block's score rows in place, query i0 + r reading keys up to it when causal (the rest
   zero); lse gets each row's log of the sum. */
static void block_softmax(float *S, size_t ld, uint32_t rows, uint32_t keys, uint32_t i0, bool causal, float *lse) {
    for (uint32_t r = 0; r < rows; r++) {
        float *row = S + r * ld;
        const uint32_t end = causal ? i0 + r + 1u : keys;
        float m = -FLT_MAX, sum = 0.0f;
        for (uint32_t j = 0; j < end; j++) m = row[j] > m ? row[j] : m;
        for (uint32_t j = 0; j < end; j++) {
            row[j] = expf(row[j] - m);
            sum += row[j];
        }
        const float inv = 1.0f / sum;
        for (uint32_t j = 0; j < end; j++) row[j] *= inv;
        for (uint32_t j = end; j < keys; j++) row[j] = 0.0f;
        if (lse) lse[r] = m + logf(sum);
    }
}

static int attn_threads(const BatchWorkspace *ws, ComputeMode mode, uint64_t tasks) {
    return mode == COMPUTE_OPENMP && tasks > 1 ? ws->attn_threads : 1;
}

void spingalett_attention_forward(const NeuralNetwork *net, uint32_t l, const float *x, float *y, uint32_t n,
                                  float *lse, const float *rope, BatchWorkspace *ws, ComputeMode mode) {
    const AttnShape a = attn_shape(net, l);
    const size_t per_thread = spingalett_attention_scratch(net, l, ws->training);
    const uint64_t tasks = (uint64_t)n * a.kv;
    const int threads = attn_threads(ws, mode, tasks);
    SPINGALETT_PARALLEL_FOR_THREADS(threads, threads > 1,
        for (int64_t task = 0; task < (int64_t)tasks; task++) {
            const int tid = threads > 1 ? spingalett_thread_num() : 0;
            float *S = ws->attn + (size_t)tid * per_thread, *Qr = S + (size_t)ATTN_BLOCK * a.cells;
            float *Kr = Qr + (size_t)ATTN_BLOCK * a.head;
            SpingalettGemmScratch *gemm = ws->attn_gemm[tid];
            const uint32_t smp = (uint32_t)(task / a.kv), g = (uint32_t)(task % a.kv);
            const float *xs = x + (size_t)smp * a.cells * a.C;
            const float *K = xs + (size_t)(a.heads + g) * a.head, *V = xs + (size_t)(a.heads + a.kv + g) * a.head;
            size_t ldk = a.C;
            if (rope) {
                rope_rows(K, a.C, Kr, a.head, a.cells, 0, a.half, rope, a.cells, false);
                K = Kr;
                ldk = a.head;
            }
            for (uint32_t h = g * a.group; h < (g + 1u) * a.group; h++)
                for (uint32_t i0 = 0; i0 < a.cells; i0 += ATTN_BLOCK) {
                    const uint32_t rows = a.cells - i0 < ATTN_BLOCK ? a.cells - i0 : ATTN_BLOCK;
                    const uint32_t keys = a.causal ? i0 + rows : a.cells;
                    const float *Q = xs + (size_t)i0 * a.C + (size_t)h * a.head;
                    size_t ldq = a.C;
                    if (rope) {
                        rope_rows(Q, a.C, Qr, a.head, rows, i0, a.half, rope, a.cells, false);
                        Q = Qr;
                        ldq = a.head;
                    }
                    spingalett_gemm_native(gemm, false, true, rows, keys, a.head, a.scale, Q, ldq, K, ldk, 0.0f, S,
                                           a.cells, false);
                    block_softmax(S, a.cells, rows, keys, i0, a.causal,
                                  lse ? lse + ((size_t)smp * a.heads + h) * a.cells + i0 : NULL);
                    spingalett_gemm_native(gemm, false, false, rows, a.head, keys, 1.0f, S, a.cells, V, a.C, 0.0f,
                                           y + ((size_t)smp * a.cells + i0) * a.out_c + (size_t)h * a.head, a.out_c,
                                           false);
                }
        }
    );
}

void spingalett_attention_backward(const NeuralNetwork *net, uint32_t l, const float *x, const float *dy, float *dx,
                                   uint32_t n, const float *lse, const float *rope, BatchWorkspace *ws, ComputeMode mode) {
    const AttnShape a = attn_shape(net, l);
    const size_t per_thread = spingalett_attention_scratch(net, l, true);
    const uint64_t tasks = (uint64_t)n * a.kv;
    const int threads = attn_threads(ws, mode, tasks);
    SPINGALETT_PARALLEL_FOR_THREADS(threads, threads > 1,
        for (int64_t task = 0; task < (int64_t)tasks; task++) {
            const int tid = threads > 1 ? spingalett_thread_num() : 0;
            const size_t B = ATTN_BLOCK, T = a.cells, d = a.head;
            float *P = ws->attn + (size_t)tid * per_thread, *dP = P + B * T, *Qr = dP + B * T, *dQ = Qr + B * d;
            float *Kr = dQ + B * d, *dK = Kr + T * d, *dV = dK + T * d, *D = dV + T * d;
            SpingalettGemmScratch *gemm = ws->attn_gemm[tid];
            const uint32_t smp = (uint32_t)(task / a.kv), g = (uint32_t)(task % a.kv);
            const float *xs = x + (size_t)smp * T * a.C;
            float *dxs = dx + (size_t)smp * T * a.C;
            const float *K = xs + (size_t)(a.heads + g) * d, *V = xs + (size_t)(a.heads + a.kv + g) * d;
            size_t ldk = a.C;
            if (rope) {
                rope_rows(K, a.C, Kr, d, a.cells, 0, a.half, rope, a.cells, false);
                K = Kr;
                ldk = d;
            }
            memset(dK, 0, T * d * sizeof(float));
            memset(dV, 0, T * d * sizeof(float));
            for (uint32_t h = g * a.group; h < (g + 1u) * a.group; h++)
                for (uint32_t i0 = 0; i0 < a.cells; i0 += ATTN_BLOCK) {
                    const uint32_t rows = a.cells - i0 < ATTN_BLOCK ? a.cells - i0 : ATTN_BLOCK;
                    const uint32_t keys = a.causal ? i0 + rows : a.cells;
                    const float *Q = xs + (size_t)i0 * a.C + (size_t)h * d;
                    size_t ldq = a.C;
                    if (rope) {
                        rope_rows(Q, a.C, Qr, d, rows, i0, a.half, rope, a.cells, false);
                        Q = Qr;
                        ldq = d;
                    }
                    const float *dO = dy + ((size_t)smp * T + i0) * a.out_c + (size_t)h * d;
                    const float *L = lse + ((size_t)smp * a.heads + h) * T + i0;
                    /* P from the scores and the forward pass's log sums */
                    spingalett_gemm_native(gemm, false, true, rows, keys, a.head, a.scale, Q, ldq, K, ldk, 0.0f, P, T,
                                           false);
                    for (uint32_t r = 0; r < rows; r++) {
                        float *row = P + r * T;
                        const uint32_t end = a.causal ? i0 + r + 1u : keys;
                        for (uint32_t j = 0; j < end; j++) row[j] = expf(row[j] - L[r]);
                        for (uint32_t j = end; j < keys; j++) row[j] = 0.0f;
                    }
                    /* dS = P (dP - D) with dP = dO V^T and D = rowsum(dO O) = rowsum(P dP) (the outputs before
                       the layer's activation need not be kept) */
                    spingalett_gemm_native(gemm, false, true, rows, keys, a.head, 1.0f, dO, a.out_c, V, a.C, 0.0f, dP, T,
                                           false);
                    for (uint32_t r = 0; r < rows; r++) {
                        float dot = 0.0f;
                        for (uint32_t j = 0; j < keys; j++) dot += P[r * T + j] * dP[r * T + j];
                        D[r] = dot;
                        for (uint32_t j = 0; j < keys; j++) dP[r * T + j] = P[r * T + j] * (dP[r * T + j] - D[r]);
                    }
                    /* dQ = dS K / sqrt(d): into its place, rotated back when the queries were rotated */
                    if (rope) {
                        spingalett_gemm_native(gemm, false, false, rows, a.head, keys, a.scale, dP, T, K, ldk, 0.0f, dQ,
                                               d, false);
                        rope_rows(dQ, d, dxs + (size_t)i0 * a.C + (size_t)h * d, a.C, rows, i0, a.half, rope, a.cells,
                                  true);
                    } else {
                        spingalett_gemm_native(gemm, false, false, rows, a.head, keys, a.scale, dP, T, K, ldk, 0.0f,
                                               dxs + (size_t)i0 * a.C + (size_t)h * d, a.C, false);
                    }
                    /* dK += dS^T Q / sqrt(d), dV += P^T dO */
                    spingalett_gemm_native(gemm, true, false, keys, a.head, rows, a.scale, dP, T, Q, ldq, 1.0f, dK, d,
                                           false);
                    spingalett_gemm_native(gemm, true, false, keys, a.head, rows, 1.0f, P, T, dO, a.out_c, 1.0f, dV, d,
                                           false);
                }
            float *dKx = dxs + (size_t)(a.heads + g) * d, *dVx = dxs + (size_t)(a.heads + a.kv + g) * d;
            if (rope) {
                rope_rows(dK, d, dKx, a.C, a.cells, 0, a.half, rope, a.cells, true);
            } else {
                for (uint32_t j = 0; j < T; j++) memcpy(dKx + (size_t)j * a.C, dK + (size_t)j * d, d * sizeof(float));
            }
            for (uint32_t j = 0; j < T; j++) memcpy(dVx + (size_t)j * a.C, dV + (size_t)j * d, d * sizeof(float));
        }
    );
}
