/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/*
 * bmm.comp: bdot's product of many rows (a prompt's tokens) on the matrix units, the CPU's bits: a workgroup takes BM
 * rows of the quantized activations and BN rows of weights of a block precision (TYPE), four warps of 32 x 32
 * outputs, over k in stages of CH chunks of 32 values. A stage's memory comes while the one before is computed: the
 * rows' codes, sums of 16 and blocks' scales by asynchronous copies into the other half of their memory, the
 * weights' blocks by loads of whole words into registers (every row's memory in consecutive words: loads of whole
 * sectors), stored when the stage begins. Then the weights' codes are decoded from their blocks in shared memory as
 * signed bytes with each sub-block's integer scale and minimum (as spingalett_block_dot() reads them), the products
 * of each sub-block by mma.sync of 8-bit integers (exact; m16n8k32 where a scale covers 32 values, m16n8k16 where it
 * covers 16), times the sub-block's scale, added in integers (I1), the minima times the rows' sums (I2); then each
 * output's block term (d dx) I1 (+ (m dx) I2), every product and sum rounded alone, added to its sum block after
 * block. The bias then the sum, the activation of spec 2, the product by e0's value: bdot's epilogue. Push:
 * SpgBdotPush (r0 the first row of the dispatch, rows its rows); groups: the tiles of rows, then of weights. A unit
 * a TYPE (bmm_<type>). The weights' buffer holds 16 bytes more than its rows (their last word read whole).
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

#define KS (TYPE == SPG_B_Q2_K || TYPE == SPG_B_Q3_K || TYPE == SPG_B_Q6_K ? 16u : 32u)   /* values a scale's */
#define ONE (BV == 32u)                     /* blocks of 32: sub-blocks of scale 1 (and minimum 1, with MINS) */

#define BM 64u
#define BN 64u
#define CH (ONE ? 1u : 4u)                  /* chunks of 32 values of k a stage */
/* words of a row of weights a stage keeps (its blocks, unaligned; odd, rows in different banks) */
#define RW (((ONE ? CH * BYTES : BYTES) / 4u + 2u) | 1u)
#define RG 10u                               /* words of weights a thread loads together */

typedef signed char int8_t;
typedef short int16_t;
struct uint4_ {
    uint32_t x, y, z, w;
} __attribute__((aligned(16)));

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
        case SPG_B_Q8_0: {
            /* (two bytes' alignment: sixteen-bit loads) */
            const uint16_t *h = (const uint16_t *)(w + 2u + 16u * j);
            for (uint32_t l = 0; l < 8u; l++) {
                const uint32_t v = h[l];
                q[2u * l] = (int8_t)(v & 0xFFu);
                q[2u * l + 1u] = (int8_t)(v >> 8);
            }
            break;
        }
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

/* asynchronous copies to shared memory: `bytes` of 4 (16) from src, the rest of them zeros */
DEVICE void copy4(uint32_t dst, const void *src, uint32_t bytes) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;" ::"r"(dst), "l"(src), "r"(bytes) : "memory");
}
DEVICE void copy16(uint32_t dst, const void *src, uint32_t bytes) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(dst), "l"(src), "r"(bytes) : "memory");
}
DEVICE void copies_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
DEVICE void copies_wait() { asm volatile("cp.async.wait_group 0;" ::: "memory"); }
DEVICE void sts128(void *at, uint32_t x, uint32_t y, uint32_t z, uint32_t w) {
    asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};" :: "r"(shared_address(at)), "r"(x), "r"(y), "r"(z), "r"(w)
                 : "memory");
}
/* d = a b of 8-bit integers (16 x 32 by 32 x 8), exact */
DEVICE void mma_s8_k32(int32_t (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
        "{%10, %11, %12, %13};"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(0), "r"(0), "r"(0), "r"(0));
}

/* (float)i, exactly: blocks of 32 sum less than 2^22 in magnitude, (float)(1.5 2^23 + i) by its bits, less 1.5 2^23
   (conversions run at a quarter of the rate of additions) */
DEVICE float to_float(int32_t i) {
    if (!ONE) return (float)i;
    const float big = 12582912.0f;
    return __builtin_bit_cast(float, i + 0x4B400000) - big;
}

/* A chunk's row of 32 codes in shared memory: its halves of 16 swapped in rows 4-7 of every 8 (ldmatrix's eight rows
   of a half in different banks without padding) */
DEVICE uint32_t at32(uint32_t r, uint32_t half) { return 32u * r + 16u * (half ^ ((r >> 2) & 1u)); }

struct Shared {
    uint8_t a[2][CH][BM * 32u];                         /* the rows' codes of a stage's chunks (two stages) */
    uint8_t b[CH][BN * 32u];                            /* the weights' codes decoded */
    uint32_t raw[BN][RW];                               /* the weights' blocks of the stage as stored */
    int8_t sc[CH][BN][2], mn[CH][BN][2];                /* the chunks' sub-blocks' scales and minima */
    int32_t xsum[2][CH][MINS ? BM : 1][2];              /* the rows' sums of 16 codes (two stages) */
    float dx[2][CH][BM], dw[CH][BN], mw[CH][BN];        /* the scales of the stage's blocks (a slot each) */
};

extern "C" __global__ void __launch_bounds__(128) SPG_ENTRY(const SpgBdotPush p, const Spec spec) {
    __shared__ __attribute__((aligned(16))) Shared sh;
    const uint32_t tid = thread_x(), lane = tid & 31u, warp = tid >> 5, wm = warp >> 1, wn = warp & 1u;
    const uint32_t g = lane >> 2, t = lane & 3u;
    /* (the tiles of rows of one tile of weights side by side: the weights read from memory once) */
    const uint32_t m0 = block_x() * BM, n0 = block_y() * BN, blocks = p.k / BV, chunks = p.k / 32u;
    const uint32_t rows = p.rows, stages = (chunks + CH - 1u) / CH;
    float acc[2][4][4];
    int32_t i1[2][4][4], i2[2][4][4];
#pragma unroll
    for (uint32_t a = 0; a < 2u; a++)
#pragma unroll
        for (uint32_t b = 0; b < 4u; b++)
#pragma unroll
            for (uint32_t e = 0; e < 4u; e++) acc[a][b][e] = 0.0f, i1[a][b][e] = 0, i2[a][b][e] = 0;
    /* The stages in a pipeline: while a stage is decoded and multiplied, the next one's rows (codes, sums of 16, the
       blocks' scales) come by asynchronous copies into the other half of their memory and its weights' blocks by
       loads into registers (stored when the stage begins). A stage's weights' blocks: its blocks of 32, or the block
       of 256 that a stage begins every other stage (half a block a stage). */
    uint32_t raw[(BN * RW + 127u) / 128u];
    for (uint32_t s = 0; s <= stages; s++) {
        /* stage s's memory requested (s < stages), then stage s - 1 decoded and multiplied (s > 0) */
        if (s > 0u) {
            const uint32_t c0 = (s - 1u) * CH;
            if (c0 % CPB == 0u) {
#pragma unroll
                for (uint32_t i = 0; i < (BN * RW + 127u) / 128u; i++) {
                    const uint32_t u = tid + 128u * i, r = u / RW, w = u % RW;
                    if (r < BN) sh.raw[r][w] = raw[i];
                }
            }
            copies_wait();
        }
        barrier();
        if (s < stages) {
            const uint32_t c0 = s * CH, cs = chunks - c0 < CH ? chunks - c0 : CH, blk0 = c0 / CPB, nb = ONE ? cs : 1u;
            const uint32_t buf = s & 1u, dbuf = ONE ? buf : blk0 & 1u;
            /* the rows' memory by asynchronous copies, every row's in consecutive words (whole sectors; zeros past the
               rows) */
#pragma unroll
            for (uint32_t i = 0; i < BM * 8u / 128u; i++) {
                const uint32_t u = tid + 128u * i, r = u >> 3, gi = u & 7u, m = m0 + r, live = m < rows;
                if (gi >> 1 >= cs) continue;
                const uint64_t row = (uint64_t)(p.r0 + (live ? m : 0u));
                copy16(shared_address(sh.a[buf][gi >> 1] + at32(r, gi & 1u)),
                       (const int8_t *)p.q + row * p.k + 32u * c0 + 16u * gi, live ? 16u : 0u);
                if (MINS)
                    copy4(shared_address(&sh.xsum[buf][gi >> 1][r][gi & 1u]),
                          (const int32_t *)p.xs + row * (p.k / 16u) + 2u * c0 + gi, live ? 4u : 0u);
            }
            if (c0 % CPB == 0u) {
#pragma unroll
                for (uint32_t i = 0; i < (BM * CH + 127u) / 128u; i++) {
                    const uint32_t u = tid + 128u * i, r = u / CH, bi = u % CH, m = m0 + r, live = m < rows;
                    if (r < BM && bi < nb)
                        copy4(shared_address(&sh.dx[dbuf][bi][r]),
                              F(p.dx) + (uint64_t)(p.r0 + (live ? m : 0u)) * blocks + blk0 + bi, live ? 4u : 0u);
                }
                /* the weights' blocks from the word their first byte is in (loads: four-byte copies cost more
                   instructions than the loads and the stores) */
                const uint32_t words = (nb * BYTES + 3u) / 4u + 1u;
#pragma unroll
                for (uint32_t i = 0; i < (BN * RW + 127u) / 128u; i++) {
                    const uint32_t u = tid + 128u * i, r = u / RW, w = u % RW, n = n0 + r;
                    raw[i] = 0u;
                    if (r < BN && w < words && n < p.n) {
                        const uint64_t first = (uint64_t)n * p.stride + (uint64_t)blk0 * BYTES;
                        raw[i] = ((const uint32_t *)(p.w + (first & ~(uint64_t)3)))[w];
                    }
                }
            }
            copies_commit();
        }
        if (s == 0u) continue;
        const uint32_t c0 = (s - 1u) * CH, cs = chunks - c0 < CH ? chunks - c0 : CH, blk0 = c0 / CPB, nb = ONE ? cs : 1u;
        const uint32_t buf = (s - 1u) & 1u, dbuf = ONE ? buf : blk0 & 1u;
        /* the weights' codes of the stage's chunks decoded (a thread a row and four of its halves of chunks), each
           block's scales */
        {
            const uint32_t r = tid & (BN - 1u), n = n0 + r;
            const uint8_t *row = (const uint8_t *)sh.raw[r] + (((uint64_t)n * p.stride + (uint64_t)blk0 * BYTES) & 3u);
            for (uint32_t v = tid >> 6; v < 2u * CH; v += 2u) {
                const uint32_t cc = v >> 1, half = v & 1u, c = c0 + cc;
                if (cc >= cs) break;
                int8_t q[16];
                int32_t sc = 0, mn = 0;
                if (n < p.n) {
                    decode16(row + (ONE ? cc * BYTES : 0u), 2u * (c % CPB) + half, q, sc, mn);
                } else {
                    for (uint32_t l = 0; l < 16u; l++) q[l] = 0;
                }
                uint32_t words[4];
                for (uint32_t w = 0; w < 4u; w++)
                    words[w] = (uint32_t)(uint8_t)q[4u * w] | (uint32_t)(uint8_t)q[4u * w + 1u] << 8 |
                               (uint32_t)(uint8_t)q[4u * w + 2u] << 16 | (uint32_t)(uint8_t)q[4u * w + 3u] << 24;
                sts128(sh.b[cc] + at32(r, half), words[0], words[1], words[2], words[3]);
                if (!ONE) {
                    sh.sc[cc][r][half] = (int8_t)sc;
                    if (MINS) sh.mn[cc][r][half] = (int8_t)mn;
                }
            }
            if (c0 % CPB == 0u)
                for (uint32_t bi = tid >> 6; bi < nb; bi += 2u) {
                    float d = 0.0f, mm = 0.0f;
                    if (n < p.n) block_scales(row + bi * BYTES, d, mm);
                    sh.dw[bi][r] = d;
                    sh.mw[bi][r] = mm;
                }
        }
        barrier();
#pragma unroll 1
        for (uint32_t cc = 0; cc < cs; cc++) {
            const uint32_t c = c0 + cc;
            /* A fragments of the warp's two tiles of 16 rows, B of its four of 8 columns: both halves of 16 (ldmatrix
               x4 of A: rows 0-7 k 0-15, rows 8-15 k 0-15, rows 0-7 k 16-31, rows 8-15 k 16-31; of B: n 0-7 k 0-15,
               n 0-7 k 16-31, n 8-15 k 0-15, n 8-15 k 16-31) */
            uint32_t af[2][4], bf[2][4];
#pragma unroll
            for (uint32_t a = 0; a < 2u; a++)
                ldsm4(shared_address(sh.a[buf][cc] + at32(32u * wm + 16u * a + (lane & 15u), lane >> 4)), af[a]);
#pragma unroll
            for (uint32_t b = 0; b < 2u; b++)
                ldsm4(shared_address(sh.b[cc] + at32(32u * wn + 16u * b + (lane & 7u) + 8u * (lane >> 4), (lane >> 3) & 1u)),
                      bf[b]);
            if (KS == 16u) {
                /* sub-blocks of 16: each half's products times its scale */
#pragma unroll
                for (uint32_t h = 0; h < 2u; h++) {
                    int32_t sc[4][2], mn[4][2], xr[2][2];
#pragma unroll
                    for (uint32_t b = 0; b < 4u; b++)
#pragma unroll
                        for (uint32_t e = 0; e < 2u; e++) {
                            const uint32_t cn = 32u * wn + 8u * b + 2u * t + e;
                            sc[b][e] = sh.sc[cc][cn][h];
                            mn[b][e] = MINS ? sh.mn[cc][cn][h] : 0;
                        }
#pragma unroll
                    for (uint32_t a = 0; a < 2u; a++)
#pragma unroll
                        for (uint32_t e = 0; e < 2u; e++)
                            xr[a][e] = MINS ? sh.xsum[buf][cc][MINS ? 32u * wm + 16u * a + g + 8u * e : 0u][h] : 0;
#pragma unroll
                    for (uint32_t a = 0; a < 2u; a++)
#pragma unroll
                        for (uint32_t b = 0; b < 4u; b++) {
                            int32_t s4[4];
                            mma_s8(s4, af[a][2u * h], af[a][2u * h + 1u], bf[b >> 1][2u * (b & 1u) + h]);
#pragma unroll
                            for (uint32_t e = 0; e < 4u; e++) {
                                i1[a][b][e] += sc[b][e & 1u] * s4[e];
                                if (MINS) i2[a][b][e] += mn[b][e & 1u] * xr[a][e >> 1];
                            }
                        }
                }
            } else {
                /* sub-blocks of 32 (blocks of 32: a scale of 1, the minimum's sums the rows' alone): products of 32 */
                int32_t sc[4][2], mn[4][2], xr[2][2];
#pragma unroll
                for (uint32_t b = 0; b < 4u; b++)
#pragma unroll
                    for (uint32_t e = 0; e < 2u; e++) {
                        const uint32_t cn = 32u * wn + 8u * b + 2u * t + e;
                        sc[b][e] = ONE ? 1 : sh.sc[cc][cn][0];
                        mn[b][e] = MINS && !ONE ? sh.mn[cc][cn][0] : 1;
                    }
#pragma unroll
                for (uint32_t a = 0; a < 2u; a++)
#pragma unroll
                    for (uint32_t e = 0; e < 2u; e++) {
                        const uint32_t rm = MINS ? 32u * wm + 16u * a + g + 8u * e : 0u;
                        xr[a][e] = MINS ? sh.xsum[buf][cc][rm][0] + sh.xsum[buf][cc][rm][1] : 0;
                    }
#pragma unroll
                for (uint32_t a = 0; a < 2u; a++)
#pragma unroll
                    for (uint32_t b = 0; b < 4u; b++) {
                        int32_t s4[4];
                        mma_s8_k32(s4, af[a], bf[b >> 1][2u * (b & 1u)], bf[b >> 1][2u * (b & 1u) + 1u]);
#pragma unroll
                        for (uint32_t e = 0; e < 4u; e++) {
                            i1[a][b][e] = ONE ? s4[e] : i1[a][b][e] + sc[b][e & 1u] * s4[e];
                            if (MINS) i2[a][b][e] = ONE ? xr[a][e >> 1] : i2[a][b][e] + mn[b][e & 1u] * xr[a][e >> 1];
                        }
                    }
            }
            if (c % CPB == CPB - 1u) {
                /* the block's terms, in order */
                const uint32_t slot = ONE ? cc : 0u;
#pragma unroll
                for (uint32_t a = 0; a < 2u; a++)
#pragma unroll
                    for (uint32_t b = 0; b < 4u; b++)
#pragma unroll
                        for (uint32_t e = 0; e < 4u; e++) {
                            const uint32_t cn = 32u * wn + 8u * b + 2u * t + (e & 1u);
                            const uint32_t rm = 32u * wm + 16u * a + g + 8u * (e >> 1);
                            const float dx = sh.dx[dbuf][slot][rm];
                            float term = (sh.dw[slot][cn] * dx) * to_float(i1[a][b][e]);
                            if (MINS) term = term + (sh.mw[slot][cn] * dx) * to_float(i2[a][b][e]);
                            acc[a][b][e] = acc[a][b][e] + term;
                            i1[a][b][e] = 0;
                            i2[a][b][e] = 0;
                        }
            }
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
