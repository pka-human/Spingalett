/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/* attn.comp: attention in tiles of BQ rows that never store the scores (FORWARD, PRE, DQ, DKV), in single
   precision, reading and writing activations kept as bfloat16 through ld()/st(). A unit per head size at
   most (DMAX 32, 64, 128, 256 with tiles of 32, 16, 8, 4 rows, THREADS 128, 64, 64, 16): the rows'
   threads, keys and columns as constants, so that the arrays of a thread's sums stay in registers.
   Spec: OP, BQ, DMAX, THREADS, HALF (words: x 0, y 1, dy 2, dx 3). */

#include "common.cuh"

#if !defined(SPG_ENTRY)
#define SPG_ENTRY spg_attn_d64
#define DMAX 64u
#endif
#if DMAX <= 32
#define BQ 32u
#define THREADS 128u
#elif DMAX <= 64
#define BQ 16u
#define THREADS 64u
#elif DMAX <= 128
#define BQ 8u
#define THREADS 64u
#else
#define BQ 4u
#define THREADS 16u
#endif
#define TPR (THREADS / BQ)
#define KPT (BQ / TPR)
#define CPT (DMAX / TPR)

#define FORWARD 0u
#define PRE     1u
#define DQ      2u
#define DKV     3u
#define FLAG_CAUSAL 1u
#define FLAG_ROPE   2u
#define FLAG_STATS  4u

struct Shared {
    float a[BQ * DMAX], b[BQ * DMAX], k[BQ * DMAX], v[BQ * DMAX];
    float p[BQ * BQ], s[BQ * BQ], l[BQ], d[BQ], red[THREADS];
};

/* value c of a query or key vector at float `at` of x, at position pos: rotated with ROPE */
DEVICE float rotated(const SpgAttnPush &p, uint32_t half, uint32_t at, uint32_t pos, uint32_t c) {
    const float v = ld(p.x, at + c, 0u, half);
    if (!(p.flags & FLAG_ROPE)) return v;
    const uint32_t hd = p.d / 2u, i = c < hd ? c : c - hd;
    const float cs = F(p.table)[pos * hd + i], sn = F(p.table)[p.cells * hd + pos * hd + i];
    return c < hd ? v * cs - ld(p.x, at + c + hd, 0u, half) * sn : v * cs + ld(p.x, at + c - hd, 0u, half) * sn;
}

/* the maximum (or the sum) of v over a row's threads, in their order */
DEVICE float row_combine(float *red, uint32_t tid, uint32_t r, float v, bool maximum) {
    red[tid] = v;
    barrier();
    float acc = red[r * TPR];
#pragma unroll
    for (uint32_t k = 1; k < TPR; k++) acc = maximum ? max_(acc, red[r * TPR + k]) : acc + red[r * TPR + k];
    barrier();
    return acc;
}

/* g rotated back: the gradient of the vector before the rotation, from a row of the tile t */
DEVICE float unrotated(const SpgAttnPush &p, const float *t, uint32_t pos, uint32_t c) {
    float v = t[c];
    if (!(p.flags & FLAG_ROPE)) return v;
    const uint32_t hd = p.d / 2u, i = c < hd ? c : c - hd;
    const float cs = F(p.table)[pos * hd + i], sn = F(p.table)[p.cells * hd + pos * hd + i];
    return c < hd ? v * cs + t[c + hd] * sn : v * cs - t[c - hd] * sn;
}

extern "C" __global__ void __launch_bounds__(THREADS) SPG_ENTRY(const SpgAttnPush p, const Spec spec) {
    const uint32_t OP = spec.v[0], HALF = spec.v[4];
    __shared__ Shared sh;
    const uint32_t tid = thread_x(), r = tid / TPR, q = tid % TPR, d = p.d;
    const uint32_t C = (p.heads + 2u * p.kv) * d, out_c = p.heads * d, group = p.heads / p.kv;
    const bool causal = (p.flags & FLAG_CAUSAL) != 0u;
    const float NEG = -3.402823466e38f;
    const uint32_t rows = p.n * p.heads * p.cells;
    if (OP == PRE) {
        for (uint32_t e = global_x(); e < rows; e += blocks_x() * THREADS) {
            const uint32_t i = e % p.cells, h = (e / p.cells) % p.heads, s = e / (p.cells * p.heads);
            const uint32_t at = (s * p.cells + i) * out_c + h * d;
            float dot = 0.0f;
            for (uint32_t c = 0; c < d; c++) dot = fma_(ld(p.dy, at + c, 2u, HALF), ld(p.x, at + c, 0u, HALF), dot);
            F(p.stats)[rows + e] = dot;
        }
        return;
    }
    const uint32_t s = block_z();
    if (OP == FORWARD || OP == DQ) {
        const uint32_t h = block_y(), g = h / group, i0 = block_x() * BQ, i = i0 + r;
        for (uint32_t e = tid; e < BQ * d; e += THREADS) {
            const uint32_t rr = e / d, c = e % d, ii = i0 + rr, at = (s * p.cells + ii) * C + h * d;
            sh.a[rr * DMAX + c] = ii < p.cells ? rotated(p, HALF, at, ii, c) * p.scale : 0.0f;
            if (OP == DQ) sh.b[rr * DMAX + c] = ii < p.cells ? ld(p.dy, (s * p.cells + ii) * out_c + h * d + c, 2u, HALF) : 0.0f;
        }
        const uint32_t stat = (s * p.heads + h) * p.cells + i;
        const float lse = OP == DQ && i < p.cells ? F(p.stats)[stat] : 0.0f;
        const float delta = OP == DQ && i < p.cells ? F(p.stats)[rows + stat] : 0.0f;
        float m = NEG, l = 0.0f, acc[CPT];
#pragma unroll
        for (uint32_t k = 0; k < CPT; k++) acc[k] = 0.0f;
        const uint32_t keys = causal ? umin(p.cells, i0 + BQ) : p.cells;
        for (uint32_t j0 = 0; j0 < keys; j0 += BQ) {
            barrier();
            for (uint32_t e = tid; e < BQ * d; e += THREADS) {
                const uint32_t jj = e / d, c = e % d, j = j0 + jj, at = (s * p.cells + j) * C;
                sh.k[jj * DMAX + c] = j < p.cells ? rotated(p, HALF, at + (p.heads + g) * d, j, c) : 0.0f;
                sh.v[jj * DMAX + c] = j < p.cells ? ld(p.x, at + (p.heads + p.kv + g) * d + c, 0u, HALF) : 0.0f;
            }
            barrier();
            float sc[KPT], dp[KPT], top = NEG;
#pragma unroll
            for (uint32_t k = 0; k < KPT; k++) {
                const uint32_t jj = q + TPR * k, j = j0 + jj;
                float dot = 0.0f, dot2 = 0.0f;
                for (uint32_t c = 0; c < d; c++) {
                    dot = fma_(sh.a[r * DMAX + c], sh.k[jj * DMAX + c], dot);
                    if (OP == DQ) dot2 = fma_(sh.b[r * DMAX + c], sh.v[jj * DMAX + c], dot2);
                }
                const bool valid = i < p.cells && j < p.cells && (!causal || j <= i);
                sc[k] = valid ? dot : NEG;
                dp[k] = dot2;
                top = max_(top, sc[k]);
            }
            if (OP == FORWARD) {
                const float mt = row_combine(sh.red, tid, r, top, true), mnew = max_(m, mt);
                const float alpha = mnew == NEG ? 1.0f : exp_(m - mnew);
                float sum = 0.0f;
#pragma unroll
                for (uint32_t k = 0; k < KPT; k++) {
                    const float pk = sc[k] == NEG ? 0.0f : exp_(sc[k] - mnew);
                    sh.p[r * BQ + q + TPR * k] = pk;
                    sum += pk;
                }
                l = l * alpha + row_combine(sh.red, tid, r, sum, false);
                m = mnew;
#pragma unroll
                for (uint32_t k = 0; k < CPT; k++) acc[k] *= alpha;
                for (uint32_t jj = 0; jj < BQ; jj++) {
                    const float pj = sh.p[r * BQ + jj];
#pragma unroll
                    for (uint32_t k = 0; k < CPT; k++) acc[k] = fma_(pj, sh.v[jj * DMAX + q + TPR * k], acc[k]);
                }
            } else {
#pragma unroll
                for (uint32_t k = 0; k < KPT; k++) {
                    const float pk = sc[k] == NEG ? 0.0f : exp_(sc[k] - lse);
                    sh.p[r * BQ + q + TPR * k] = pk * (dp[k] - delta);
                }
                barrier();
                for (uint32_t jj = 0; jj < BQ; jj++) {
                    const float ds = sh.p[r * BQ + jj];
#pragma unroll
                    for (uint32_t k = 0; k < CPT; k++) acc[k] = fma_(ds, sh.k[jj * DMAX + q + TPR * k], acc[k]);
                }
            }
        }
        if (OP == FORWARD) {
            const float inv = 1.0f / l;
            if (i < p.cells) {
#pragma unroll
                for (uint32_t k = 0; k < CPT; k++) {
                    const uint32_t c = q + TPR * k;
                    if (c < d) st(p.y, (s * p.cells + i) * out_c + h * d + c, 1u, HALF, acc[k] * inv);
                }
                if ((p.flags & FLAG_STATS) && q == 0u) F(p.stats)[stat] = m + log_(l);
            }
            return;
        }
        barrier();
#pragma unroll
        for (uint32_t k = 0; k < CPT; k++) sh.a[r * DMAX + q + TPR * k] = acc[k] * p.scale;
        barrier();
        if (i < p.cells)
#pragma unroll
            for (uint32_t k = 0; k < CPT; k++) {
                const uint32_t c = q + TPR * k;
                if (c < d) st(p.dx, (s * p.cells + i) * C + h * d + c, 3u, HALF, unrotated(p, sh.a + r * DMAX, i, c));
            }
        return;
    }
    /* DKV */
    const uint32_t g = block_y(), j0 = block_x() * BQ, j = j0 + r;
    for (uint32_t e = tid; e < BQ * d; e += THREADS) {
        const uint32_t jj = e / d, c = e % d, jk = j0 + jj, at = (s * p.cells + jk) * C;
        sh.k[jj * DMAX + c] = jk < p.cells ? rotated(p, HALF, at + (p.heads + g) * d, jk, c) : 0.0f;
        sh.v[jj * DMAX + c] = jk < p.cells ? ld(p.x, at + (p.heads + p.kv + g) * d + c, 0u, HALF) : 0.0f;
    }
    float dk[CPT], dv[CPT];
#pragma unroll
    for (uint32_t k = 0; k < CPT; k++) dk[k] = dv[k] = 0.0f;
    for (uint32_t h = g * group; h < (g + 1u) * group; h++)
        for (uint32_t i0 = causal ? j0 : 0u; i0 < p.cells; i0 += BQ) {
            barrier();
            for (uint32_t e = tid; e < BQ * d; e += THREADS) {
                const uint32_t rr = e / d, c = e % d, ii = i0 + rr, at = (s * p.cells + ii) * C + h * d;
                sh.a[rr * DMAX + c] = ii < p.cells ? rotated(p, HALF, at, ii, c) * p.scale : 0.0f;
                sh.b[rr * DMAX + c] = ii < p.cells ? ld(p.dy, (s * p.cells + ii) * out_c + h * d + c, 2u, HALF) : 0.0f;
            }
            if (tid < BQ) {
                const uint32_t ii = i0 + tid, stat = (s * p.heads + h) * p.cells + ii;
                sh.l[tid] = ii < p.cells ? F(p.stats)[stat] : 0.0f;
                sh.d[tid] = ii < p.cells ? F(p.stats)[rows + stat] : 0.0f;
            }
            barrier();
#pragma unroll
            for (uint32_t k = 0; k < KPT; k++) {
                const uint32_t rr = q + TPR * k, ii = i0 + rr;
                float dot = 0.0f, dot2 = 0.0f;
                for (uint32_t c = 0; c < d; c++) {
                    dot = fma_(sh.a[rr * DMAX + c], sh.k[r * DMAX + c], dot);
                    dot2 = fma_(sh.b[rr * DMAX + c], sh.v[r * DMAX + c], dot2);
                }
                const bool valid = ii < p.cells && j < p.cells && (!causal || j <= ii);
                const float pk = valid ? exp_(dot - sh.l[rr]) : 0.0f;
                sh.p[rr * BQ + r] = pk;
                sh.s[rr * BQ + r] = pk * (dot2 - sh.d[rr]);
            }
            barrier();
            for (uint32_t rr = 0; rr < BQ; rr++) {
                const float pk = sh.p[rr * BQ + r], ds = sh.s[rr * BQ + r];
#pragma unroll
                for (uint32_t k = 0; k < CPT; k++) {
                    const uint32_t c = q + TPR * k;
                    dv[k] = fma_(pk, sh.b[rr * DMAX + c], dv[k]);
                    dk[k] = fma_(ds, sh.a[rr * DMAX + c], dk[k]);
                }
            }
        }
    barrier();
#pragma unroll
    for (uint32_t k = 0; k < CPT; k++) sh.k[r * DMAX + q + TPR * k] = dk[k];
    barrier();
    if (j >= p.cells) return;
    const uint32_t at = (s * p.cells + j) * C;
#pragma unroll
    for (uint32_t k = 0; k < CPT; k++) {
        const uint32_t c = q + TPR * k;
        if (c >= d) continue;
        st(p.dx, at + (p.heads + g) * d + c, 3u, HALF, unrotated(p, sh.k + r * DMAX, j, c));
        st(p.dx, at + (p.heads + p.kv + g) * d + c, 3u, HALF, dv[k]);
    }
}
