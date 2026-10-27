/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/*
 * bmm.comp: bdot's product of many rows (a prompt's tokens) on the matrix units, the CPU's bits: a workgroup takes BM
 * rows of the quantized activations and BN rows of weights of a block precision (TYPE), four warps of 32 x 32
 * outputs. For each block of k, chunk after chunk of 32 values: the rows' codes into shared memory, the weights' codes
 * decoded into it as signed bytes with each sub-block of 16's integer scale and minimum (as spingalett_block_dot()
 * reads them), the products of each sub-block by mma.sync m16n8k16 of 8-bit integers (exact), times the sub-block's
 * scale, added in integers (I1), the minima times the rows' sums of 16 codes (I2); then each output's block term (d dx)
 * I1 (+ (m dx) I2), every product and sum rounded alone, added to its sum block after block. The bias then the sum, the
 * activation of spec 2, the product by e0's value: bdot's epilogue. Push: SpgBdotPush (r0 the first row of the
 * dispatch, rows its rows). A unit a TYPE (bmm_<type>).
 */

#include "common.cuh"

#if !defined(SPG_ENTRY)
#define SPG_ENTRY spg_bmm_q8_0
#define TYPE 0u
#endif

#define KQ (TYPE == SPG_B_Q2_K || TYPE == SPG_B_Q3_K || TYPE == SPG_B_Q4_K || TYPE == SPG_B_Q5_K || \
            TYPE == SPG_B_Q6_K || TYPE == SPG_B_IQ4_XS)
#define BV (KQ ? 256u : 32u)
#define CPB (BV / 32u)
#define BYTES (TYPE == SPG_B_Q8_0 ? 34u : TYPE == SPG_B_Q4_0 ? 18u : TYPE == SPG_B_Q4_1 ? 20u : TYPE == SPG_B_Q5_0 ? 22u : \
               TYPE == SPG_B_Q5_1 ? 24u : TYPE == SPG_B_Q2_K ? 84u : TYPE == SPG_B_Q3_K ? 110u : TYPE == SPG_B_Q4_K ? 144u : \
               TYPE == SPG_B_Q5_K ? 176u : TYPE == SPG_B_Q6_K ? 210u : TYPE == SPG_B_IQ4_NL ? 18u : 136u)
#define MINS (TYPE == SPG_B_Q4_1 || TYPE == SPG_B_Q5_1 || TYPE == SPG_B_Q2_K || TYPE == SPG_B_Q4_K || TYPE == SPG_B_Q5_K)

#define BM 64u
#define BN 64u
#define SROW 48u                            /* bytes a row of a chunk in shared memory: 32 codes, 16 apart */

typedef signed char int8_t;

DEVICE float half_at(const uint8_t *b) { return from_half((uint32_t)b[0] | (uint32_t)b[1] << 8); }
DEVICE uint32_t get16(const uint8_t *b) { return (uint32_t)b[0] | (uint32_t)b[1] << 8; }
DEVICE uint32_t get32(const uint8_t *b) {
    return (uint32_t)b[0] | (uint32_t)b[1] << 8 | (uint32_t)b[2] << 16 | (uint32_t)b[3] << 24;
}
DEVICE int32_t iq4_value(uint32_t code) {
    const int8_t v[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
    return v[code];
}
DEVICE void scale_min_k4(uint32_t j, const uint8_t *q, int32_t &d, int32_t &m) {
    if (j < 4u) {
        d = q[j] & 63;
        m = q[j + 4u] & 63;
    } else {
        d = (q[j + 4u] & 0xF) | ((q[j - 4u] >> 6) << 4);
        m = (q[j + 4u] >> 4) | ((q[j] >> 6) << 4);
    }
}
DEVICE int32_t q3_scale(const uint8_t *packed, uint32_t is) {
    const uint32_t kmask1 = 0x03030303u, kmask2 = 0x0f0f0f0fu;
    const uint32_t a0 = get32(packed), a1 = get32(packed + 4), tmp = get32(packed + 8);
    uint32_t aux;
    switch (is / 4u) {
        case 0: aux = (a0 & kmask2) | ((tmp & kmask1) << 4); break;
        case 1: aux = (a1 & kmask2) | (((tmp >> 2) & kmask1) << 4); break;
        case 2: aux = ((a0 >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4); break;
        default: aux = ((a1 >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4); break;
    }
    return (int32_t)((aux >> (8u * (is % 4u))) & 0xFFu) - 32;
}
DEVICE int32_t iq4xs_scale(const uint8_t *block, uint32_t ib) {
    const uint32_t high = get16(block + 2), low = block[4u + ib / 2u];
    return (int32_t)(((low >> (4u * (ib % 2u))) & 0xFu) | (((high >> (2u * ib)) & 3u) << 4)) - 32;
}

/* Sub-block j of 16 values of block w: their codes (signed bytes whose products with the inputs' codes, times sc, add
   up to the CPU's I1; the minimum's mn times the inputs' sum of 16 to its I2) */
DEVICE void decode16(const uint8_t *w, uint32_t j, int8_t (&q)[16], int32_t &sc, int32_t &mn) {
    sc = 1;
    mn = 0;
    switch (TYPE) {
        case SPG_B_Q8_0:
            for (uint32_t l = 0; l < 16u; l++) q[l] = (int8_t)w[2u + 16u * j + l];
            break;
        case SPG_B_Q4_0: case SPG_B_IQ4_NL: case SPG_B_Q4_1: {
            const uint8_t *qs = w + (TYPE == SPG_B_Q4_1 ? 4 : 2);
            for (uint32_t l = 0; l < 16u; l++) {
                const uint32_t code = j ? qs[l] >> 4 : qs[l] & 0x0F;
                q[l] = (int8_t)(TYPE == SPG_B_Q4_0 ? (int32_t)code - 8 : TYPE == SPG_B_IQ4_NL ? iq4_value(code) : (int32_t)code);
            }
            if (TYPE == SPG_B_Q4_1) mn = 1;
            break;
        }
        case SPG_B_Q5_0: case SPG_B_Q5_1: {
            const bool one = TYPE == SPG_B_Q5_1;
            const uint32_t qh = get32(w + (one ? 4 : 2));
            const uint8_t *qs = w + (one ? 8 : 6);
            for (uint32_t l = 0; l < 16u; l++) {
                const uint32_t v = 16u * j + l;
                const int32_t code = (j ? qs[l] >> 4 : qs[l] & 0x0F) | (int32_t)(((qh >> v) & 1u) << 4);
                q[l] = (int8_t)(one ? code : code - 16);
            }
            if (one) mn = 1;
            break;
        }
        case SPG_B_Q2_K: case SPG_B_Q3_K: {
            const bool three = TYPE == SPG_B_Q3_K;
            const uint32_t h = j / 8u, jj = (j % 8u) / 2u, part = j % 2u, shift = 2u * jj;
            const uint8_t *qb = w + (three ? 32 : 16) + 32u * h + 16u * part, *hm = w + 16u * part;
            for (uint32_t l = 0; l < 16u; l++) {
                int32_t code = (qb[l] >> shift) & 3;
                if (three) code -= (hm[l] >> (4u * h + jj)) & 1u ? 0 : 4;
                q[l] = (int8_t)code;
            }
            if (three) {
                sc = q3_scale(w + 96, j);
            } else {
                sc = w[j] & 0xF;
                mn = w[j] >> 4;
            }
            break;
        }
        case SPG_B_Q4_K: case SPG_B_Q5_K: {
            const bool five = TYPE == SPG_B_Q5_K;
            const uint32_t c = j / 2u, lo = 16u * (j % 2u);
            const uint8_t *qh = w + 16, *qb = w + (five ? 48 : 16) + 32u * (c / 2u);
            for (uint32_t l = 0; l < 16u; l++) {
                int32_t code = c % 2u ? qb[lo + l] >> 4 : qb[lo + l] & 0xF;
                if (five) code |= (int32_t)(((qh[lo + l] >> c) & 1u) << 4);
                q[l] = (int8_t)code;
            }
            scale_min_k4(c, w + 4, sc, mn);
            break;
        }
        case SPG_B_Q6_K: {
            const uint32_t h = j / 8u, g = (j % 8u) / 2u, l0 = 16u * (j % 2u);
            const uint8_t *lb = w + 64u * h + 32u * (g % 2u), *hb = w + 128 + 32u * h;
            for (uint32_t l = 0; l < 16u; l++) {
                const uint32_t low = g < 2u ? lb[l0 + l] & 0xF : lb[l0 + l] >> 4;
                q[l] = (int8_t)((int32_t)(low | (((hb[l0 + l] >> (2u * g)) & 3u) << 4)) - 32);
            }
            sc = ((const int8_t *)(w + 192))[j];
            break;
        }
        default: {                              /* IQ4_XS */
            const uint32_t ib = j / 2u, upper = j % 2u;
            const uint8_t *qs = w + 8u + 16u * ib;
            for (uint32_t l = 0; l < 16u; l++) q[l] = (int8_t)iq4_value(upper ? qs[l] >> 4 : qs[l] & 0xF);
            sc = iq4xs_scale(w, ib);
            break;
        }
    }
}

/* A block's scale d and minimum's m */
DEVICE void block_scales(const uint8_t *w, float &d, float &m) {
    m = 0.0f;
    switch (TYPE) {
        case SPG_B_Q4_1: case SPG_B_Q5_1: d = half_at(w); m = half_at(w + 2); break;
        case SPG_B_Q2_K: d = half_at(w + 80); m = -half_at(w + 82); break;
        case SPG_B_Q3_K: d = half_at(w + 108); break;
        case SPG_B_Q4_K: case SPG_B_Q5_K: d = half_at(w); m = -half_at(w + 2); break;
        case SPG_B_Q6_K: d = half_at(w + 208); break;
        default: d = half_at(w); break;
    }
}

DEVICE uint32_t shared_address(const void *p) {
    uint64_t r;
    asm("cvta.to.shared.u64 %0, %1;" : "=l"(r) : "l"(p));
    return (uint32_t)r;
}
DEVICE void ldsm4(uint32_t at, uint32_t (&r)[4]) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(at) : "memory");
}
/* d = a b of 8-bit integers (16 x 16 by 16 x 8), exact */
DEVICE void mma_s8(int32_t (&d)[4], uint32_t a0, uint32_t a1, uint32_t b0) {
    asm("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
        : "r"(a0), "r"(a1), "r"(b0), "r"(0), "r"(0), "r"(0), "r"(0));
}

struct Shared {
    uint8_t a[BM * SROW], b[BN * SROW];         /* a chunk's codes: the rows', the weights' */
    int32_t sc[BN][2], mn[BN][2], xsum[BM][2];  /* its sub-blocks' scales and minima, the rows' sums of 16 */
    float dx[BM], dw[BN], mw[BN];               /* the block's scales */
};

extern "C" __global__ void __launch_bounds__(128) SPG_ENTRY(const SpgBdotPush p, const Spec spec) {
    __shared__ __attribute__((aligned(16))) Shared sh;
    const uint32_t tid = thread_x(), lane = tid & 31u, warp = tid >> 5, wm = warp >> 1, wn = warp & 1u;
    const uint32_t g = lane >> 2, t = lane & 3u;
    const uint32_t n0 = block_x() * BN, m0 = block_y() * BM, blocks = p.k / BV;
    const uint32_t rows = p.rows;
    float acc[2][4][4];
    int32_t i1[2][4][4], i2[2][4][4];
#pragma unroll
    for (uint32_t a = 0; a < 2u; a++)
#pragma unroll
        for (uint32_t b = 0; b < 4u; b++)
#pragma unroll
            for (uint32_t e = 0; e < 4u; e++) acc[a][b][e] = 0.0f;
    for (uint32_t blk = 0; blk < blocks; blk++) {
#pragma unroll
        for (uint32_t a = 0; a < 2u; a++)
#pragma unroll
            for (uint32_t b = 0; b < 4u; b++)
#pragma unroll
                for (uint32_t e = 0; e < 4u; e++) i1[a][b][e] = 0, i2[a][b][e] = 0;
        for (uint32_t c = 0; c < CPB; c++) {
            barrier();
            {
                /* this thread: the rows' and the weights' halves of 16 (row tid / 2, half tid % 2) */
                const uint32_t r = tid >> 1, half = tid & 1u, k0 = blk * BV + 32u * c + 16u * half;
                const uint32_t m = m0 + r;
                uint32_t words[4] = {0u, 0u, 0u, 0u};
                int32_t sum = 0;
                if (m < rows) {
                    const uint32_t *src = (const uint32_t *)((const int8_t *)p.q + (uint64_t)(p.r0 + m) * p.k + k0);
                    for (uint32_t w = 0; w < 4u; w++) words[w] = src[w];
                    sum = ((const int32_t *)p.xs)[(uint64_t)(p.r0 + m) * (p.k / 16u) + k0 / 16u];
                }
                uint32_t *dst = (uint32_t *)(sh.a + r * SROW + 16u * half);
                for (uint32_t w = 0; w < 4u; w++) dst[w] = words[w];
                sh.xsum[r][half] = sum;
                const uint32_t n = n0 + r;
                int8_t q[16];
                int32_t sc = 0, mn = 0;
                if (n < p.n) {
                    decode16((const uint8_t *)p.w + (uint64_t)n * p.stride + (uint64_t)blk * BYTES, 2u * c + half, q, sc, mn);
                } else {
                    for (uint32_t l = 0; l < 16u; l++) q[l] = 0;
                }
                uint32_t *bd = (uint32_t *)(sh.b + r * SROW + 16u * half);
                for (uint32_t w = 0; w < 4u; w++)
                    bd[w] = (uint32_t)(uint8_t)q[4u * w] | (uint32_t)(uint8_t)q[4u * w + 1u] << 8 |
                            (uint32_t)(uint8_t)q[4u * w + 2u] << 16 | (uint32_t)(uint8_t)q[4u * w + 3u] << 24;
                sh.sc[r][half] = sc;
                sh.mn[r][half] = mn;
                if (c == 0u && half == 0u) {
                    float d = 0.0f, mm = 0.0f;
                    if (n < p.n) block_scales((const uint8_t *)p.w + (uint64_t)n * p.stride + (uint64_t)blk * BYTES, d, mm);
                    sh.dw[r] = d;
                    sh.mw[r] = mm;
                    sh.dx[r] = m < rows ? F(p.dx)[(uint64_t)(p.r0 + m) * blocks + blk] : 0.0f;
                }
            }
            barrier();
            /* A fragments of the warp's two tiles of 16 rows, B of its four of 8 columns: both halves of 16 */
            uint32_t af[2][4], bf[2][4];
#pragma unroll
            for (uint32_t a = 0; a < 2u; a++)
                ldsm4(shared_address(sh.a + (32u * wm + 16u * a + (lane & 15u)) * SROW + 16u * (lane >> 4)), af[a]);
#pragma unroll
            for (uint32_t b = 0; b < 2u; b++)
                ldsm4(shared_address(sh.b + (32u * wn + 16u * b + (lane & 7u) + 8u * (lane >> 4)) * SROW +
                                     16u * ((lane >> 3) & 1u)), bf[b]);
#pragma unroll
            for (uint32_t h = 0; h < 2u; h++)
#pragma unroll
                for (uint32_t a = 0; a < 2u; a++)
#pragma unroll
                    for (uint32_t b = 0; b < 4u; b++) {
                        int32_t s[4];
                        /* (ldmatrix x4 of B: matrices n 0-7 k 0-15, n 0-7 k 16-31, n 8-15 k 0-15, n 8-15 k 16-31) */
                        mma_s8(s, af[a][2u * h], af[a][2u * h + 1u], bf[b >> 1][2u * (b & 1u) + h]);
                        const uint32_t col = 32u * wn + 8u * b + 2u * t, row = 32u * wm + 16u * a + g;
#pragma unroll
                        for (uint32_t e = 0; e < 4u; e++) {
                            const uint32_t cn = col + (e & 1u), rm = row + 8u * (e >> 1);
                            i1[a][b][e] += sh.sc[cn][h] * s[e];
                            if (MINS) i2[a][b][e] += sh.mn[cn][h] * sh.xsum[rm][h];
                        }
                    }
        }
        /* the block's terms, in order */
#pragma unroll
        for (uint32_t a = 0; a < 2u; a++)
#pragma unroll
            for (uint32_t b = 0; b < 4u; b++)
#pragma unroll
                for (uint32_t e = 0; e < 4u; e++) {
                    const uint32_t cn = 32u * wn + 8u * b + 2u * t + (e & 1u), rm = 32u * wm + 16u * a + g + 8u * (e >> 1);
                    const float dx = sh.dx[rm];
                    float term = (sh.dw[cn] * dx) * (float)i1[a][b][e];
                    if (MINS) term = term + (sh.mw[cn] * dx) * (float)i2[a][b][e];
                    acc[a][b][e] = acc[a][b][e] + term;
                }
    }
    const uint32_t act = spec.v[2];
#pragma unroll
    for (uint32_t a = 0; a < 2u; a++)
#pragma unroll
        for (uint32_t b = 0; b < 4u; b++)
#pragma unroll
            for (uint32_t e = 0; e < 4u; e++) {
                const uint32_t n = n0 + 32u * wn + 8u * b + 2u * t + (e & 1u), m = m0 + 32u * wm + 16u * a + g + 8u * (e >> 1);
                if (n >= p.n || m >= rows) continue;
                const uint64_t at = (uint64_t)(p.r0 + m) * p.ldy + n;
                float v = F(p.bias)[n] + acc[a][b][e];
                if (act) v = activate(v, act);
                if (p.e0) v = v * F(p.e0)[at];
                F(p.y)[at] = v;
            }
}
