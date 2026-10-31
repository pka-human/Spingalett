/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/*
 * bdot.comp: rows of activations quantized by bquant times a linear layer's weights of a block precision (GGUF's
 * types as stored), the bits of the CPU's spingalett_block_dot() (Spingalett.Inference.c): in each block the integer
 * sums of its sub-blocks' products times their integer scales (I1) and, of the types with minima, the minima times
 * the sub-blocks' sums of codes (I2), exact in any order; then the block's term (d dx) I1 (+ (m dx) I2), each
 * product and sum rounded alone (the unit is compiled without contraction), added to the row's sum block after block;
 * the bias then the sum (then the activation of spec 2, and the product by e0's value). A warp a row of weights: lane l takes the l-th chunk of 32 values of each round of 32, the
 * integers of a block's chunks (eight of a super-block of 256) summed across their lanes, the blocks' terms added in
 * their order through shuffles. A unit a TYPE (SPG_B_*) and ROWS, the rows of activations a pass (bdot_<type>_r<ROWS>).
 */

#include "common.cuh"

#if !defined(SPG_ENTRY)
#define SPG_ENTRY spg_bdot_q8_0_r1
#define TYPE 0u
#define ROWS 1u
#endif

#define KQ (TYPE == SPG_B_Q2_K || TYPE == SPG_B_Q3_K || TYPE == SPG_B_Q4_K || TYPE == SPG_B_Q5_K || \
            TYPE == SPG_B_Q6_K || TYPE == SPG_B_IQ4_XS)
#define BV (KQ ? 256u : 32u)                    /* values a block */
#define CPB (BV / 32u)                          /* chunks of 32 a block */
#define BYTES (TYPE == SPG_B_Q8_0 ? 34u : TYPE == SPG_B_Q4_0 ? 18u : TYPE == SPG_B_Q4_1 ? 20u : TYPE == SPG_B_Q5_0 ? 22u : \
               TYPE == SPG_B_Q5_1 ? 24u : TYPE == SPG_B_Q2_K ? 84u : TYPE == SPG_B_Q3_K ? 110u : TYPE == SPG_B_Q4_K ? 144u : \
               TYPE == SPG_B_Q5_K ? 176u : TYPE == SPG_B_Q6_K ? 210u : TYPE == SPG_B_IQ4_NL ? 18u : 136u)

typedef signed char int8_t;

DEVICE float shfl_f(float v, uint32_t lane) { return __nvvm_shfl_sync_idx_f32(0xFFFFFFFFu, v, lane, 0x1F); }
DEVICE int32_t shfl_add(int32_t v, uint32_t mask) { return v + (int32_t)__nvvm_shfl_sync_bfly_i32(0xFFFFFFFFu, v, mask, 0x1F); }

DEVICE float half_at(const uint8_t *b) { return from_half((uint32_t)b[0] | (uint32_t)b[1] << 8); }
DEVICE uint32_t get16(const uint8_t *b) { return (uint32_t)b[0] | (uint32_t)b[1] << 8; }
DEVICE uint32_t get32(const uint8_t *b) {
    return (uint32_t)b[0] | (uint32_t)b[1] << 8 | (uint32_t)b[2] << 16 | (uint32_t)b[3] << 24;
}

/* IQ4_NL's and IQ4_XS's sixteen values (llama.cpp's kvalues_iq4nl) */
DEVICE int32_t iq4_value(uint32_t code) {
    const int8_t v[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
    return v[code];
}

/* The products of a block of 32 4-bit codes (value j the low nibble of byte j, value j + 16 its high one; with fifth
   bits, bit j of qh above value j's) with x: the codes less offset, or their values in IQ4_NL's table */
DEVICE int32_t nibbles_dot(const uint8_t *qs, uint32_t qh, int32_t offset, bool table, bool fifth, const int8_t *x) {
    int32_t s = 0;
    for (uint32_t j = 0; j < 16u; j++) {
        int32_t lo = qs[j] & 0x0F, hi = qs[j] >> 4;
        if (fifth) {
            lo |= (int32_t)(((qh >> j) & 1u) << 4);
            hi |= (int32_t)(((qh >> (j + 16u)) & 1u) << 4);
        }
        if (table) lo = iq4_value((uint32_t)lo), hi = iq4_value((uint32_t)hi);
        s += (lo - offset) * x[j] + (hi - offset) * x[j + 16];
    }
    return s;
}

/* The 6-bit scale and minimum j of a Q4_K or Q5_K super-block (llama.cpp's get_scale_min_k4()) */
DEVICE void scale_min_k4(uint32_t j, const uint8_t *q, int32_t &d, int32_t &m) {
    if (j < 4u) {
        d = q[j] & 63;
        m = q[j + 4u] & 63;
    } else {
        d = (q[j + 4u] & 0xF) | ((q[j - 4u] >> 6) << 4);
        m = (q[j + 4u] >> 4) | ((q[j] >> 6) << 4);
    }
}

/* Q3_K's 6-bit scale is (of sixteen) from its twelve bytes, less 32 */
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

/* IQ4_XS's 6-bit scale of sub-block ib, less 32 */
DEVICE int32_t iq4xs_scale(const uint8_t *block, uint32_t ib) {
    const uint32_t high = get16(block + 2), low = block[4u + ib / 2u];
    return (int32_t)(((low >> (4u * (ib % 2u))) & 0xFu) | (((high >> (2u * ib)) & 3u) << 4)) - 32;
}

/* four signed bytes' products with four, added to c */
DEVICE int32_t dp4a(uint32_t a, uint32_t b, int32_t c) {
    int32_t d;
    asm("dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d;
}
/* a word from two bytes' alignment (two loads of 16 bits), and from four's */
DEVICE uint32_t word2(const uint8_t *b) {
    const uint16_t *h = (const uint16_t *)b;
    return (uint32_t)h[0] | (uint32_t)h[1] << 16;
}
DEVICE uint32_t word4(const uint8_t *b) { return *(const uint32_t *)b; }

/* A block's scale d and minimum's m (0: none) */
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
#define MINS (TYPE == SPG_B_Q4_1 || TYPE == SPG_B_Q5_1 || TYPE == SPG_B_Q2_K || TYPE == SPG_B_Q4_K || TYPE == SPG_B_Q5_K)

/* Chunk c (of the block's CPB) of block w with its 32 codes x: its part of I1 and I2 (xs: the block's sums of 16) */
DEVICE void chunk_dot(const uint8_t *w, uint32_t c, const int8_t *x, const int32_t *xs, int32_t &i1, int32_t &i2) {
    i1 = 0;
    i2 = 0;
    /* (x: the chunk's 32 codes, a multiple of 32 bytes in: eight words) */
    const uint32_t *xw = (const uint32_t *)x;
    switch (TYPE) {
        case SPG_B_Q8_0:
            for (uint32_t k = 0; k < 8u; k++) i1 = dp4a(word2(w + 2 + 4u * k), xw[k], i1);
            break;
        case SPG_B_Q4_0: {
            /* (q - 8) x: the codes' products less 8 times the sums of x, sixteen values each */
            for (uint32_t k = 0; k < 4u; k++) {
                const uint32_t q = word2(w + 2 + 4u * k);
                i1 = dp4a(q & 0x0F0F0F0Fu, xw[k], i1);
                i1 = dp4a((q >> 4) & 0x0F0F0F0Fu, xw[k + 4u], i1);
            }
            i1 -= 8 * (xs[0] + xs[1]);
            break;
        }
        case SPG_B_IQ4_NL: i1 = nibbles_dot(w + 2, 0u, 0, true, false, x); break;
        case SPG_B_Q4_1: i1 = nibbles_dot(w + 4, 0u, 0, false, false, x); i2 = xs[0] + xs[1]; break;
        case SPG_B_Q5_0: i1 = nibbles_dot(w + 6, get32(w + 2), 16, false, true, x); break;
        case SPG_B_Q5_1: i1 = nibbles_dot(w + 8, get32(w + 4), 0, false, true, x); i2 = xs[0] + xs[1]; break;
        case SPG_B_Q2_K: case SPG_B_Q3_K: {
            /* sub-blocks 2c and 2c + 1 of 16: value v = 128 h + 32 j + 16 part + l in bits 2 j of byte 32 h + 16 part + l */
            const bool three = TYPE == SPG_B_Q3_K;
            const uint8_t *q = w + (three ? 32 : 16), *hm = w;
            for (uint32_t is = 2u * c; is < 2u * c + 2u; is++) {
                const uint32_t h = is / 8u, j = (is % 8u) / 2u, part = is % 2u, shift = 2u * j;
                const uint8_t *qb = q + 32u * h + 16u * part;
                const int8_t *xb = x + 16u * (is - 2u * c);
                int32_t s = 0;
                for (uint32_t l = 0; l < 16u; l++) {
                    int32_t code = (qb[l] >> shift) & 3;
                    if (three) code -= (hm[16u * part + l] >> (4u * h + j)) & 1u ? 0 : 4;
                    s += code * xb[l];
                }
                if (three) {
                    i1 += q3_scale(w + 96, is) * s;
                } else {
                    i1 += (w[is] & 0xF) * s;
                    i2 += (w[is] >> 4) * xs[is];
                }
            }
            break;
        }
        case SPG_B_Q4_K: case SPG_B_Q5_K: {
            /* sub-block c of 32: value 64 (c / 2) + 32 (c % 2) + l in the low (c even) or high nibble of byte 32 (c / 2) + l
               (with a fifth bit, bit c of qh[l]); the blocks four bytes aligned (144 and 176 bytes) */
            const bool five = TYPE == SPG_B_Q5_K;
            const uint8_t *qh = w + 16, *q = w + (five ? 48 : 16);
            int32_t sc, mn, s = 0;
            scale_min_k4(c, w + 4, sc, mn);
            const uint8_t *qb = q + 32u * (c / 2u);
            const uint32_t shift = 4u * (c % 2u);
            for (uint32_t k = 0; k < 8u; k++) {
                uint32_t code = (word4(qb + 4u * k) >> shift) & 0x0F0F0F0Fu;
                if (five) code |= ((word4(qh + 4u * k) >> c) & 0x01010101u) << 4;
                s = dp4a(code, xw[k], s);
            }
            i1 = sc * s;
            i2 = mn * (xs[2u * c] + xs[2u * c + 1u]);
            break;
        }
        case SPG_B_Q6_K: {
            /* sub-blocks 2c and 2c + 1 of 16, a signed scale each: value 128 h + 32 g + l from byte 64 h + l (g even)
               or 64 h + 32 + l (g odd) of ql, its low nibble (g < 2) or high one, and bits 2 g of byte 32 h + l of qh;
               the codes less 32 as their products less 32 times the sums of x (blocks of 210 bytes: two aligned) */
            const uint32_t h = c / 4u, g = c % 4u;
            const uint8_t *lb = w + 64u * h + 32u * (g % 2u), *hb = w + 128 + 32u * h;
            const int8_t *sc = (const int8_t *)(w + 192);
            const uint32_t lshift = g < 2u ? 0u : 4u, hshift = 2u * g;
            for (uint32_t half = 0; half < 2u; half++) {
                int32_t s = 0;
                for (uint32_t k = 4u * half; k < 4u * half + 4u; k++) {
                    const uint32_t code = ((word2(lb + 4u * k) >> lshift) & 0x0F0F0F0Fu) |
                                          (((word2(hb + 4u * k) >> hshift) & 0x03030303u) << 4);
                    s = dp4a(code, xw[k], s);
                }
                const uint32_t is = 2u * c + half;
                i1 += sc[is] * (s - 32 * xs[is]);
            }
            break;
        }
        default:                                /* IQ4_XS: sub-block c of 32, IQ4_NL's codes */
            i1 = iq4xs_scale(w, c) * nibbles_dot(w + 8u + 16u * c, 0u, 0, true, false, x);
            break;
    }
}

extern "C" __global__ void __launch_bounds__(256) SPG_ENTRY(const SpgBdotPush p, const Spec spec) {
    const uint32_t lane = thread_x() & 31u, j = block_x() * 8u + (thread_x() >> 5);
    if (j >= p.n) return;                       /* (a whole warp) */
    const uint32_t rows = p.rows < ROWS ? p.rows : ROWS, blocks = p.k / BV, chunks = p.k / 32u;
    const uint8_t *row = (const uint8_t *)p.w + (uint64_t)j * p.stride;
    float acc[ROWS];
    for (uint32_t r = 0; r < ROWS; r++) acc[r] = 0.0f;
    for (uint32_t c0 = 0; c0 < chunks; c0 += 32u) {
        const uint32_t c = c0 + lane, b = c / CPB;
        const bool live = c < chunks;
        const uint8_t *w = row + (uint64_t)(live ? b : 0u) * BYTES;
        float d, m;
        block_scales(w, d, m);
        float t[ROWS];
#pragma unroll
        for (uint32_t r = 0; r < ROWS; r++) {
            int32_t i1 = 0, i2 = 0;
            const uint64_t xr = (uint64_t)(p.r0 + (r < rows ? r : 0u));
            if (live && r < rows)
                chunk_dot(w, c % CPB, (const int8_t *)p.q + xr * p.k + 32u * c, (const int32_t *)p.xs + xr * (p.k / 16u) + b * (BV / 16u),
                          i1, i2);
            if (CPB > 1u) {
                /* (a super-block's chunks: their integers summed across their eight lanes, exact) */
                for (uint32_t h = 1u; h < CPB; h <<= 1) {
                    i1 = shfl_add(i1, h);
                    if (MINS) i2 = shfl_add(i2, h);
                }
            }
            const float dx = live && r < rows ? F(p.dx)[xr * blocks + b] : 0.0f;
            float term = (d * dx) * (float)i1;
            if (MINS) term = term + (m * dx) * (float)i2;
            t[r] = term;
        }
        /* the round's blocks' terms in their order */
        const uint32_t first = c0 / CPB, count = blocks - first < 32u / CPB ? blocks - first : 32u / CPB;
        for (uint32_t g = 0; g < count; g++)
#pragma unroll
            for (uint32_t r = 0; r < ROWS; r++) acc[r] = acc[r] + shfl_f(t[r], g * CPB);
    }
    if (lane == 0u) {
        const uint32_t act = spec.v[2];
        for (uint32_t r = 0; r < rows; r++) {
            const uint64_t at = (uint64_t)(p.r0 + r) * p.ldy + j;
            float v = F(p.bias)[j] + acc[r];
            if (act) v = activate(v, act);
            if (p.e0) v = v * F(p.e0)[at];
            F(p.y)[at] = v;
        }
    }
}
