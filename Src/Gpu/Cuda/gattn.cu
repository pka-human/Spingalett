/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/*
 * gattn: FusedAttention of graphs in single precision for heads of up to D values (a unit per D: 32, 64, 128), as
 * rattn.cu computes attention's forward pass: a block takes R queries of a batch and head, every thread RM = 4 of them
 * and a share of each tile of B keys; the TX threads of a row's group hold its scores at columns tx, tx + TX, ... and
 * its sums at chunks of four values tx, tx + TX, ... of the head. Tiles sit in shared memory as rows of D + 4 floats.
 * Queries come scaled; the keep mask replaces scores by the fill, the mask adds to them (or, of booleans, leaves keys
 * out); a product adds its terms in the order of the head's values, a sum over keys in their order: the same bits on
 * every run. Rows, heads and batches lie where the strides say (SpgGattnPush).
 */

#include "common.cuh"

#if !defined(SPG_ENTRY)
#define SPG_ENTRY spg_gattn_d64
#define D 64u
#endif
#define R SPG_GATTN_ROWS
#define T SPG_GATTN_THREADS
#define B (D <= 32u ? 64u : D <= 64u ? 32u : 16u)     /* keys a tile */
#define KVS (D > 64u)                                   /* keys and values in turn in one tile */
#define RM 4u
#define TY (R / RM)
#define TX (T / TY)
#define CN (B / TX)
#define CH (D / TX / 4u)
#define SQ (D + 4u)
#define SP (B + 4u)

DEVICE float shfl_xor(float v, uint32_t mask) { return __nvvm_shfl_sync_bfly_f32(0xFFFFFFFFu, v, mask, 0x1F); }
DEVICE void warp_sync() { __nvvm_bar_warp_sync(0xFFFFFFFFu); }
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

/* rows r0 .. r0 + count - 1 of `rows` (base: the first row's address, stride: elements between rows, cs between a
   row's values) into a tile, times mul (zeros past the rows and the head's d values) */
DEVICE void load_rows(const SpgGattnPush &p, float *tile, uint64_t base, uint32_t stride, uint32_t cs, uint32_t r0,
                      uint32_t count, uint32_t rows, float mul) {
    if (p.flags & SPG_GATTN_VEC) {
        for (uint32_t e = thread_x(); e < count * (D / 4u); e += T) {
            const uint32_t row = e / (D / 4u), c = 4u * (e % (D / 4u)), pos = r0 + row;
            const float4_ v = pos < rows ? *(const float4_ *)(F(base) + (uint64_t)pos * stride + c) : float4_{0.0f, 0.0f, 0.0f, 0.0f};
            *(float4_ *)(tile + row * SQ + c) = scale4(v, mul);
        }
        return;
    }
    for (uint32_t e = thread_x(); e < count * D; e += T) {
        const uint32_t row = e / D, c = e % D, pos = r0 + row;
        tile[row * SQ + c] = pos < rows && c < p.d ? F(base)[(uint64_t)pos * stride + (uint64_t)c * cs] * mul : 0.0f;
    }
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

/* acc[m][k] += the row group's probabilities times v's rows, over the tile's B keys in order */
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

struct Shared {
    float q[R * SQ];
#if KVS
    float kv[B * SQ];
#else
    float k[B * SQ], v[B * SQ];
#endif
    float p[R * SP];
};

extern "C" __global__ void __launch_bounds__(T) SPG_ENTRY(const SpgGattnPush p, const Spec spec) {
    (void)spec;
    __shared__ __attribute__((aligned(16))) Shared sh;
    const uint32_t *st = U(p.strides);
    const uint32_t tid = thread_x(), tx = tid % TX, ry = tid / TX;
    const uint32_t bh = block_y(), b = bh / p.heads, h = bh % p.heads, i0 = block_x() * R;
    const uint64_t qb = p.q + 4ull * ((uint64_t)b * st[0] + (uint64_t)h * st[1]);
    const uint64_t kb = p.k + 4ull * ((uint64_t)b * st[3] + (uint64_t)h * st[4]);
    const uint64_t vb = p.v + 4ull * ((uint64_t)b * st[6] + (uint64_t)h * st[7]);
    const uint64_t mbase = (uint64_t)b * st[12] + (uint64_t)h * st[13], wbase = (uint64_t)b * st[16] + (uint64_t)h * st[17];
    const float NEG = -__builtin_inff();
    load_rows(p, sh.q, qb, st[2], 1u, i0, R, p.queries, p.scale);
    float mx[RM], sum[RM];
    float4_ acc[RM][CH];
#pragma unroll
    for (uint32_t m = 0; m < RM; m++) {
        mx[m] = NEG;
        sum[m] = 0.0f;
#pragma unroll
        for (uint32_t k = 0; k < CH; k++) acc[m][k] = float4_{0.0f, 0.0f, 0.0f, 0.0f};
    }
    for (uint32_t j0 = 0; j0 < p.keys; j0 += B) {
        float sc[RM][CN];
#pragma unroll
        for (uint32_t m = 0; m < RM; m++)
#pragma unroll
            for (uint32_t n = 0; n < CN; n++) sc[m][n] = 0.0f;
        barrier();
#if KVS
        load_rows(p, sh.kv, kb, st[5], st[20], j0, B, p.keys, 1.0f);
        barrier();
        scores(sh.q, sh.kv, ry, tx, sc);
        barrier();
        load_rows(p, sh.kv, vb, st[8], 1u, j0, B, p.keys, 1.0f);
        barrier();
        const float *vt = sh.kv;
#else
        load_rows(p, sh.k, kb, st[5], st[20], j0, B, p.keys, 1.0f);
        load_rows(p, sh.v, vb, st[8], 1u, j0, B, p.keys, 1.0f);
        barrier();
        scores(sh.q, sh.k, ry, tx, sc);
        const float *vt = sh.v;
#endif
#pragma unroll
        for (uint32_t m = 0; m < RM; m++) {
            const uint32_t i = i0 + ry * RM + m;
            float top = NEG;
#pragma unroll
            for (uint32_t n = 0; n < CN; n++) {
                const uint32_t j = j0 + tx + TX * n;
                float v = sc[m][n];
                if (i >= p.queries || j >= p.keys) {
                    v = NEG;
                } else {
                    if ((p.flags & SPG_GATTN_KEEP) &&
                        !((const uint8_t *)p.keep)[wbase + (uint64_t)i * st[18] + (uint64_t)j * st[19]])
                        v = p.fill;
                    if (p.flags & SPG_GATTN_MASK) v += F(p.mask)[mbase + (uint64_t)i * st[14] + (uint64_t)j * st[15]];
                    else if ((p.flags & SPG_GATTN_BOOL) &&
                             !((const uint8_t *)p.mask)[mbase + (uint64_t)i * st[14] + (uint64_t)j * st[15]])
                        v = NEG;
                }
                sc[m][n] = v;
                top = max_(top, v);
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
    }
    const uint64_t yb = p.y + 4ull * ((uint64_t)b * st[9] + (uint64_t)h * st[10]);
#pragma unroll
    for (uint32_t m = 0; m < RM; m++) {
        const uint32_t i = i0 + ry * RM + m;
        if (i >= p.queries) continue;
        const float inv = sum[m] > 0.0f ? 1.0f / sum[m] : (p.flags & SPG_GATTN_SAFE) ? 0.0f : __builtin_nanf("");
#pragma unroll
        for (uint32_t k = 0; k < CH; k++) {
            const uint32_t c = 4u * (tx + TX * k);
            const float4_ v = sum[m] > 0.0f || !(p.flags & SPG_GATTN_SAFE) ? scale4(acc[m][k], inv)
                                                                           : float4_{0.0f, 0.0f, 0.0f, 0.0f};
            float *y = F(yb) + (uint64_t)i * st[11] + c;
            if ((p.flags & SPG_GATTN_VEC) && c + 3u < p.d) {
                *(float4_ *)y = v;
            } else {
                if (c < p.d) y[0] = v.x;
                if (c + 1u < p.d) y[1] = v.y;
                if (c + 2u < p.d) y[2] = v.z;
                if (c + 3u < p.d) y[3] = v.w;
            }
        }
    }
}
