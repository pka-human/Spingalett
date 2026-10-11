/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/*
 * rattn: attn.comp's passes in single precision for heads of up to 128 values, with the scores and sums of
 * every thread in registers. A block takes R rows (queries; keys in DKV), every thread RM = 4 of them and a
 * share of the other side's tile of B rows: the TX threads of a row's group (consecutive lanes of a warp)
 * hold its scores at columns tx, tx + TX, ... and its sums at chunks of four values tx, tx + TX, ... of the
 * head. Tiles sit in shared memory as rows of D + 4 floats, which the threads of a quarter-warp read as
 * float4 from distinct banks; a row group passes its probabilities through shared memory to itself (a warp
 * sync, no barrier). Heads of exactly D values are loaded four values at a time, the next tile into
 * registers while the block computes on the current one.
 *
 *   FORWARD  S = Q K^T over tiles of keys, the softmax online (a row's maximum and sum combined over its
 *            group's lanes, xor 1, 2, 4 ...), O += P V; y = O / sum, the log sum with STATS.
 *   PRE      rowsum(dO o), as attn.cu computes it.
 *   DQ       S, P = exp(S - lse), dP = dO V^T, dS = P (dP - rowsum(dO o)), dQ += dS K.
 *   DKV      per block of keys, over the group's query heads and tiles of queries in order: S^T = K Q^T, P^T,
 *            dP^T = V dO^T, dV += P^T dO, dS^T, dK += dS^T Q.
 *
 * Queries come scaled (and queries and keys rotated, ROPE) as they are loaded; a product adds its terms in
 * the order of the head's values, a sum over keys in their order: the same bits on every run. A unit per
 * head size D and pass (OPC); the head d may be smaller than D (zeros past it). Spec: OP, BQ, DMAX, THREADS,
 * HALF (words: x 0, y 1, dy 2, dx 3), the tiles those of SPG_RATTN_ROWS and SPG_RATTN_THREADS.
 */

#include "common.cuh"

#if !defined(SPG_ENTRY)
#define SPG_ENTRY spg_rattn_d64_o0
#define D 64u
#define OPC 0u
#endif
#define R SPG_RATTN_ROWS(D, OPC)
#define T SPG_RATTN_THREADS(D, OPC)
#if OPC == 0
#define B (D <= 32u ? 64u : D <= 64u ? 32u : 16u)   /* rows of the other side's tiles */
#define KVS (D > 64u)                               /* keys and values in turn in one tile */
#else
#define B (D <= 32u ? 32u : D <= 64u ? 16u : 8u)
#define KVS 0
#endif
#define RM 4u                   /* rows a thread */
#define TY (R / RM)             /* row groups */
#define TX (T / TY)             /* threads a row group */
#define CN (B / TX)             /* scores of a row a thread */
#define CH (D / TX / 4u)        /* chunks of four of a row of the head a thread */
#define SQ (D + 4u)             /* floats a row of the tiles of D */
#define SP (B + 4u)             /* floats a row of probabilities */
#define LC (B * D / 4u / T)     /* chunks of four of a tile of B rows a thread loads */

#define FORWARD 0u
#define PRE     1u
#define DQ      2u
#define DKV     3u
#define FLAG_CAUSAL 1u
#define FLAG_ROPE   2u
#define FLAG_STATS  4u

DEVICE float shfl_xor(float v, uint32_t mask) { return __nvvm_shfl_sync_bfly_f32(0xFFFFFFFFu, v, mask, 0x1F); }
DEVICE void warp_sync() { __nvvm_bar_warp_sync(0xFFFFFFFFu); }
/* the maximum (sum) of a row's values over its group's lanes */
DEVICE float group_max(float v) {
#pragma unroll
    for (uint32_t m = 1; m < TX; m <<= 1) v = max_(v, shfl_xor(v, m));
    return v;
}
DEVICE float group_sum(float v) {
#pragma unroll
    for (uint32_t m = 1; m < TX; m <<= 1) v += shfl_xor(v, m);
    return v;
}
DEVICE float4_ fma4s(float a, float4_ b, float4_ c) {
    return float4_{fma_(a, b.x, c.x), fma_(a, b.y, c.y), fma_(a, b.z, c.z), fma_(a, b.w, c.w)};
}

/* Where a tile's rows come from: float `head` on of each cell's `stride` floats of base (push-constant word
   `word`), rotated with rotate (ROPE), times mul. */
struct Rows {
    uint64_t base;
    uint32_t head, stride, word;
    bool rotate;
    float mul;
};

/* Value c of row pos (zero past the cells or the head's d values) */
DEVICE float value(const SpgAttnPush &p, uint32_t HALF, uint32_t s, uint32_t pos, uint32_t c, const Rows &r) {
    const uint32_t d = p.d, hd = d / 2u;
    if (pos >= p.cells || c >= d) return 0.0f;
    const uint32_t at = (s * p.cells + pos) * r.stride + r.head;
    float v = ld(r.base, at + c, r.word, HALF);
    if (r.rotate && (p.flags & FLAG_ROPE)) {
        const uint32_t i = c < hd ? c : c - hd;
        const float cs = F(p.table)[pos * hd + i], sn = F(p.table)[p.cells * hd + pos * hd + i];
        v = c < hd ? v * cs - ld(r.base, at + c + hd, r.word, HALF) * sn : v * cs + ld(r.base, at + c - hd, r.word, HALF) * sn;
    }
    return v * r.mul;
}

/* Values c .. c + 3 of row pos, a head of exactly D values: as value() computes them, four at a time */
DEVICE float4_ chunk(const SpgAttnPush &p, uint32_t HALF, uint32_t s, uint32_t pos, uint32_t c, const Rows &r) {
    if (pos >= p.cells) return float4_{0.0f, 0.0f, 0.0f, 0.0f};
    const uint32_t at = (s * p.cells + pos) * r.stride + r.head;
    float4_ v = ld4(r.base, at + c, r.word, HALF);
    if (r.rotate && (p.flags & FLAG_ROPE)) {
        const uint32_t hd = D / 2u, i = c < hd ? c : c - hd;
        const float4_ w = ld4(r.base, at + (c < hd ? c + hd : c - hd), r.word, HALF);
        const float4_ cs = *(const float4_ *)(F(p.table) + pos * hd + i);
        const float4_ sn = *(const float4_ *)(F(p.table) + p.cells * hd + pos * hd + i);
        v = c < hd ? float4_{v.x * cs.x - w.x * sn.x, v.y * cs.y - w.y * sn.y, v.z * cs.z - w.z * sn.z, v.w * cs.w - w.w * sn.w}
                   : float4_{v.x * cs.x + w.x * sn.x, v.y * cs.y + w.y * sn.y, v.z * cs.z + w.z * sn.z, v.w * cs.w + w.w * sn.w};
    }
    return scale4(v, r.mul);
}

/* Rows r0 .. r0 + count - 1 into a tile [count][SQ] */
DEVICE void load_rows(const SpgAttnPush &p, uint32_t HALF, float *tile, uint32_t s, uint32_t r0, uint32_t count,
                      const Rows &r) {
    if (p.d == D) {
        for (uint32_t e = thread_x(); e < count * (D / 4u); e += T) {
            const uint32_t row = e / (D / 4u), c = 4u * (e % (D / 4u));
            *(float4_ *)(tile + row * SQ + c) = chunk(p, HALF, s, r0 + row, c, r);
        }
        return;
    }
    for (uint32_t e = thread_x(); e < count * D; e += T) tile[(e / D) * SQ + e % D] = value(p, HALF, s, r0 + e / D, e % D, r);
}

/* A tile of B rows from r0 into this thread's registers (heads of exactly D values), then into shared memory */
DEVICE void fetch(const SpgAttnPush &p, uint32_t HALF, uint32_t s, uint32_t r0, const Rows &r, float4_ (&v)[LC]) {
#pragma unroll
    for (uint32_t k = 0; k < LC; k++) {
        const uint32_t e = thread_x() + T * k;
        v[k] = chunk(p, HALF, s, r0 + e / (D / 4u), 4u * (e % (D / 4u)), r);
    }
}
DEVICE void put(float *tile, const float4_ (&v)[LC]) {
#pragma unroll
    for (uint32_t k = 0; k < LC; k++) {
        const uint32_t e = thread_x() + T * k;
        *(float4_ *)(tile + (e / (D / 4u)) * SQ + 4u * (e % (D / 4u))) = v[k];
    }
}
/* The tile of B rows from r0 into shared memory: from the registers fetch() filled, or loaded now */
DEVICE void place(const SpgAttnPush &p, uint32_t HALF, float *tile, uint32_t s, uint32_t r0, const Rows &r,
                  const float4_ (&v)[LC]) {
    if (p.d == D) put(tile, v);
    else load_rows(p, HALF, tile, s, r0, B, r);
}

/* s[m][n] += a's row (ry RM + m) . b's row (tx + TX n), the head's values four at a time, in order */
DEVICE void scores(const float *a, const float *b, uint32_t ry, uint32_t tx, float (&s)[RM][CN]) {
#pragma unroll 2
    for (uint32_t c = 0; c < D; c += 4u) {
        float4_ x[RM], y[CN];
#pragma unroll
        for (uint32_t m = 0; m < RM; m++) x[m] = *(const float4_ *)(a + (ry * RM + m) * SQ + c);
#pragma unroll
        for (uint32_t n = 0; n < CN; n++) y[n] = *(const float4_ *)(b + (tx + TX * n) * SQ + c);
#pragma unroll
        for (uint32_t m = 0; m < RM; m++)
#pragma unroll
            for (uint32_t n = 0; n < CN; n++) {
                s[m][n] = fma_(x[m].x, y[n].x, s[m][n]);
                s[m][n] = fma_(x[m].y, y[n].y, s[m][n]);
                s[m][n] = fma_(x[m].z, y[n].z, s[m][n]);
                s[m][n] = fma_(x[m].w, y[n].w, s[m][n]);
            }
    }
}

/* acc[m][k] += the row group's probabilities pt[ry RM + m][j] times v's row j at chunk tx + TX k, over the
   tile's B rows j in order */
DEVICE void accumulate(const float *pt, const float *v, uint32_t ry, uint32_t tx, float4_ (&acc)[RM][CH]) {
#pragma unroll 1
    for (uint32_t j = 0; j < B; j += 4u) {
        float4_ w[RM];
#pragma unroll
        for (uint32_t m = 0; m < RM; m++) w[m] = *(const float4_ *)(pt + (ry * RM + m) * SP + j);
#pragma unroll
        for (uint32_t jj = 0; jj < 4u; jj++)
#pragma unroll
            for (uint32_t k = 0; k < CH; k++) {
                const float4_ y = *(const float4_ *)(v + (j + jj) * SQ + 4u * (tx + TX * k));
#pragma unroll
                for (uint32_t m = 0; m < RM; m++) acc[m][k] = fma4s(get4(w[m], jj), y, acc[m][k]);
            }
    }
}

/* The gradient of the vector before the rotation, value c of a tile's row at position pos */
DEVICE float unrotated(const SpgAttnPush &p, const float *row, uint32_t pos, uint32_t c) {
    const float v = row[c];
    if (!(p.flags & FLAG_ROPE)) return v;
    const uint32_t hd = p.d / 2u, i = c < hd ? c : c - hd;
    const float cs = F(p.table)[pos * hd + i], sn = F(p.table)[p.cells * hd + pos * hd + i];
    return c < hd ? v * cs + row[c + hd] * sn : v * cs - row[c - hd] * sn;
}

/* A thread's sums (rows ry RM + m, chunks tx + TX k) into a tile, times mul */
DEVICE void spill(float *tile, uint32_t ry, uint32_t tx, const float4_ (&acc)[RM][CH], float mul) {
#pragma unroll
    for (uint32_t m = 0; m < RM; m++)
#pragma unroll
        for (uint32_t k = 0; k < CH; k++) *(float4_ *)(tile + (ry * RM + m) * SQ + 4u * (tx + TX * k)) = scale4(acc[m][k], mul);
}

struct Shared {
#if OPC == FORWARD
    float q[R * SQ];
#if KVS
    float kv[B * SQ];
#else
    float k[B * SQ], v[B * SQ];
#endif
    float p[R * SP];
#elif OPC == DQ
    float q[R * SQ], g[R * SQ], k[B * SQ], v[B * SQ], p[R * SP];
#elif OPC == DKV
    float k[R * SQ], v[R * SQ], q[B * SQ], g[B * SQ], p[R * SP], l[B], dl[B];
#else
    float none;
#endif
};

extern "C" __global__ void __launch_bounds__(T) SPG_ENTRY(const SpgAttnPush p, const Spec spec) {
    const uint32_t HALF = spec.v[4];
    const uint32_t d = p.d, out_c = p.heads * d, rows = p.n * p.heads * p.cells;
#if OPC == PRE
    for (uint32_t e = global_x(); e < rows; e += blocks_x() * T) {
        const uint32_t i = e % p.cells, h = (e / p.cells) % p.heads, s = e / (p.cells * p.heads);
        const uint32_t at = (s * p.cells + i) * out_c + h * d;
        float dot = 0.0f;
        for (uint32_t c = 0; c < d; c++) dot = fma_(ld(p.dy, at + c, 2u, HALF), ld(p.x, at + c, 0u, HALF), dot);
        F(p.stats)[rows + e] = dot;
    }
#else
    __shared__ __attribute__((aligned(16))) Shared sh;
    const uint32_t tid = thread_x(), tx = tid % TX, ry = tid / TX, s = block_z();
    const uint32_t C = (p.heads + 2u * p.kv) * d, group = p.heads / p.kv;
    const bool causal = (p.flags & FLAG_CAUSAL) != 0u, vec = d == D;
    const float NEG = -3.402823466e38f;
    float4_ ra[LC], rb[LC];         /* the next tile's rows, as fetch() loads them */
#if OPC == FORWARD || OPC == DQ
    const uint32_t h = block_y(), kh = h / group, i0 = block_x() * R, sa = (s * p.heads + h) * p.cells;
    const Rows keys_of = {p.x, (p.heads + kh) * d, C, 0u, true, 1.0f};
    const Rows values_of = {p.x, (p.heads + p.kv + kh) * d, C, 0u, false, 1.0f};
    load_rows(p, HALF, sh.q, s, i0, R, Rows{p.x, h * d, C, 0u, true, p.scale});
#if OPC == DQ
    load_rows(p, HALF, sh.g, s, i0, R, Rows{p.dy, h * d, out_c, 2u, false, 1.0f});
    float lse[RM], del[RM];
#pragma unroll
    for (uint32_t m = 0; m < RM; m++) {
        const uint32_t i = i0 + ry * RM + m;
        lse[m] = i < p.cells ? F(p.stats)[sa + i] : 0.0f;
        del[m] = i < p.cells ? F(p.stats)[rows + sa + i] : 0.0f;
    }
#else
    float mx[RM], sum[RM];
#pragma unroll
    for (uint32_t m = 0; m < RM; m++) mx[m] = NEG, sum[m] = 0.0f;
#endif
    float4_ acc[RM][CH];
#pragma unroll
    for (uint32_t m = 0; m < RM; m++)
#pragma unroll
        for (uint32_t k = 0; k < CH; k++) acc[m][k] = float4_{0.0f, 0.0f, 0.0f, 0.0f};
    const uint32_t keys = causal ? umin(p.cells, i0 + R) : p.cells;
    if (vec) {
        fetch(p, HALF, s, 0u, keys_of, ra);
        fetch(p, HALF, s, 0u, values_of, rb);
    }
    for (uint32_t j0 = 0; j0 < keys; j0 += B) {
        const bool next = vec && j0 + B < keys;
        float sc[RM][CN];
#pragma unroll
        for (uint32_t m = 0; m < RM; m++)
#pragma unroll
            for (uint32_t n = 0; n < CN; n++) sc[m][n] = 0.0f;
        barrier();
#if KVS
        place(p, HALF, sh.kv, s, j0, keys_of, ra);
        barrier();
        if (next) fetch(p, HALF, s, j0 + B, keys_of, ra);
        scores(sh.q, sh.kv, ry, tx, sc);
        barrier();
        place(p, HALF, sh.kv, s, j0, values_of, rb);
        barrier();
        if (next) fetch(p, HALF, s, j0 + B, values_of, rb);
        const float *vt = sh.kv;
#else
        place(p, HALF, sh.k, s, j0, keys_of, ra);
        place(p, HALF, sh.v, s, j0, values_of, rb);
        barrier();
        if (next) {
            fetch(p, HALF, s, j0 + B, keys_of, ra);
            fetch(p, HALF, s, j0 + B, values_of, rb);
        }
        scores(sh.q, sh.k, ry, tx, sc);
        const float *vt = sh.v;
#endif
#if OPC == FORWARD
#pragma unroll
        for (uint32_t m = 0; m < RM; m++) {
            const uint32_t i = i0 + ry * RM + m;
            float top = NEG;
#pragma unroll
            for (uint32_t n = 0; n < CN; n++) {
                const uint32_t j = j0 + tx + TX * n;
                if (!(i < p.cells && j < p.cells && (!causal || j <= i))) sc[m][n] = NEG;
                top = max_(top, sc[m][n]);
            }
            const float mn = max_(mx[m], group_max(top)), alpha = mn == NEG ? 1.0f : exp_(mx[m] - mn);
            float part = 0.0f;
#pragma unroll
            for (uint32_t n = 0; n < CN; n++) {
                const float pv = sc[m][n] == NEG ? 0.0f : exp_(sc[m][n] - mn);
                sh.p[(ry * RM + m) * SP + tx + TX * n] = pv;
                part += pv;
            }
            sum[m] = sum[m] * alpha + group_sum(part);
            mx[m] = mn;
#pragma unroll
            for (uint32_t k = 0; k < CH; k++) acc[m][k] = scale4(acc[m][k], alpha);
        }
        warp_sync();
        accumulate(sh.p, vt, ry, tx, acc);
#else
        float dp[RM][CN];
#pragma unroll
        for (uint32_t m = 0; m < RM; m++)
#pragma unroll
            for (uint32_t n = 0; n < CN; n++) dp[m][n] = 0.0f;
        scores(sh.g, vt, ry, tx, dp);
#pragma unroll
        for (uint32_t m = 0; m < RM; m++) {
            const uint32_t i = i0 + ry * RM + m;
#pragma unroll
            for (uint32_t n = 0; n < CN; n++) {
                const uint32_t j = j0 + tx + TX * n;
                const bool valid = i < p.cells && j < p.cells && (!causal || j <= i);
                sh.p[(ry * RM + m) * SP + tx + TX * n] = valid ? exp_(sc[m][n] - lse[m]) * (dp[m][n] - del[m]) : 0.0f;
            }
        }
        warp_sync();
        accumulate(sh.p, sh.k, ry, tx, acc);
#endif
    }
#if OPC == FORWARD
#pragma unroll
    for (uint32_t m = 0; m < RM; m++) {
        const uint32_t i = i0 + ry * RM + m;
        if (i >= p.cells) continue;
        const float inv = 1.0f / sum[m];
        const uint32_t at = (s * p.cells + i) * out_c + h * d;
#pragma unroll
        for (uint32_t k = 0; k < CH; k++) {
            const uint32_t c = 4u * (tx + TX * k);
            const float4_ o = scale4(acc[m][k], inv);
            if (vec) {
                st4(p.y, at + c, 1u, HALF, o);
            } else {
#pragma unroll
                for (uint32_t e = 0; e < 4u; e++)
                    if (c + e < d) st(p.y, at + c + e, 1u, HALF, get4(o, e));
            }
        }
        if ((p.flags & FLAG_STATS) && tx == 0u) F(p.stats)[sa + i] = mx[m] + log_(sum[m]);
    }
#else
    /* dQ times the scale, rotated back through shared memory */
    barrier();
    spill(sh.q, ry, tx, acc, p.scale);
    barrier();
    for (uint32_t e = tid; e < R * D; e += T) {
        const uint32_t row = e / D, c = e % D, i = i0 + row;
        if (i < p.cells && c < d) st(p.dx, (s * p.cells + i) * C + h * d + c, 3u, HALF, unrotated(p, sh.q + row * SQ, i, c));
    }
#endif
#elif OPC == DKV
    /* the block's keys and values of head kh; the queries of the group's heads */
    const uint32_t kh = block_y(), j0 = block_x() * R;
    load_rows(p, HALF, sh.k, s, j0, R, Rows{p.x, (p.heads + kh) * d, C, 0u, true, 1.0f});
    load_rows(p, HALF, sh.v, s, j0, R, Rows{p.x, (p.heads + p.kv + kh) * d, C, 0u, false, 1.0f});
    float4_ dk[RM][CH], dv[RM][CH];
#pragma unroll
    for (uint32_t m = 0; m < RM; m++)
#pragma unroll
        for (uint32_t k = 0; k < CH; k++) dk[m][k] = dv[m][k] = float4_{0.0f, 0.0f, 0.0f, 0.0f};
    const uint32_t first = causal ? (j0 / B) * B : 0u;
    for (uint32_t h = kh * group; h < (kh + 1u) * group; h++) {
        const uint32_t sa = (s * p.heads + h) * p.cells;
        const Rows queries_of = {p.x, h * d, C, 0u, true, p.scale}, grads_of = {p.dy, h * d, out_c, 2u, false, 1.0f};
        if (vec) {
            fetch(p, HALF, s, first, queries_of, ra);
            fetch(p, HALF, s, first, grads_of, rb);
        }
        for (uint32_t q0 = first; q0 < p.cells; q0 += B) {
            barrier();
            place(p, HALF, sh.q, s, q0, queries_of, ra);
            place(p, HALF, sh.g, s, q0, grads_of, rb);
            for (uint32_t e = tid; e < B; e += T) {
                const bool live = q0 + e < p.cells;
                sh.l[e] = live ? F(p.stats)[sa + q0 + e] : 0.0f;
                sh.dl[e] = live ? F(p.stats)[rows + sa + q0 + e] : 0.0f;
            }
            barrier();
            if (vec && q0 + B < p.cells) {
                fetch(p, HALF, s, q0 + B, queries_of, ra);
                fetch(p, HALF, s, q0 + B, grads_of, rb);
            }
            float st_[RM][CN], dpt[RM][CN];
#pragma unroll
            for (uint32_t m = 0; m < RM; m++)
#pragma unroll
                for (uint32_t n = 0; n < CN; n++) st_[m][n] = dpt[m][n] = 0.0f;
            scores(sh.k, sh.q, ry, tx, st_);
            scores(sh.v, sh.g, ry, tx, dpt);
            /* P^T = exp(S^T - lse[query]) into the row group's probabilities; dS^T = P^T (dP^T - delta[query]) */
#pragma unroll
            for (uint32_t m = 0; m < RM; m++) {
                const uint32_t key = j0 + ry * RM + m;
#pragma unroll
                for (uint32_t n = 0; n < CN; n++) {
                    const uint32_t qc = tx + TX * n, query = q0 + qc;
                    const bool valid = query < p.cells && key < p.cells && (!causal || key <= query);
                    const float pv = valid ? exp_(st_[m][n] - sh.l[qc]) : 0.0f;
                    sh.p[(ry * RM + m) * SP + qc] = pv;
                    st_[m][n] = pv * (dpt[m][n] - sh.dl[qc]);
                }
            }
            warp_sync();
            accumulate(sh.p, sh.g, ry, tx, dv);
            warp_sync();
#pragma unroll
            for (uint32_t m = 0; m < RM; m++)
#pragma unroll
                for (uint32_t n = 0; n < CN; n++) sh.p[(ry * RM + m) * SP + tx + TX * n] = st_[m][n];
            warp_sync();
            accumulate(sh.p, sh.q, ry, tx, dk);
        }
    }
    /* dV as it is; dK (of scaled queries) rotated back through shared memory */
    barrier();
    spill(sh.k, ry, tx, dk, 1.0f);
    spill(sh.v, ry, tx, dv, 1.0f);
    barrier();
    for (uint32_t e = tid; e < R * D; e += T) {
        const uint32_t row = e / D, c = e % D, j = j0 + row;
        if (j >= p.cells || c >= d) continue;
        const uint32_t at = (s * p.cells + j) * C;
        st(p.dx, at + (p.heads + kh) * d + c, 3u, HALF, unrotated(p, sh.k + row * SQ, j, c));
        st(p.dx, at + (p.heads + p.kv + kh) * d + c, 3u, HALF, sh.v[row * SQ + c]);
    }
#endif
#endif
}
