/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/*
 * Layers of block precisions (GGUF's types) over many cells. Each cell is quantized in blocks of the type's
 * size (32 or 256 values: a scale a block, the sums of every 16 codes), then every output is the engine's
 * spingalett_block_dot() of its row with the cell: the integer sums of a block's sub-blocks, scaled by their
 * integer scales and added in integers, then the block's float term added to the row's sum, block after
 * block. The outputs are split between threads by rows, each computed whole by one of them, so that a cell's
 * outputs do not depend on the cells computed with it or on the threads.
 *
 * With AVX2 a row's block is first unpacked: its codes as unsigned bytes (the code plus an offset of the type:
 * 8 for Q4_0, 128 for Q8_0 and the IQ4 types' table, ...) and its sub-blocks' integer scales and minima, so that
 * one kernel serves every type. Rows are taken four at a time: each row's products of a block are summed in
 * eight lanes of 32 bits (dpbusd with VNNI, maddubs and madd without), the lanes of the four rows reduced into
 * one vector, and the float terms of the four rows computed side by side, each lane in the reference's order.
 */

#include "Spingalett.Blocks.h"
#include <float.h>
#include <math.h>
#include <string.h>

#if !defined(SPINGALETT_PORTABLE_KERNELS) && defined(__AVX2__) && defined(__F16C__)
#  include <immintrin.h>
#  define BLOCK_AVX2 1
#  if defined(__AVXVNNI__)
#    define BLOCK_VNNI(acc, u, s) _mm256_dpbusd_avx_epi32((acc), (u), (s))
#  elif defined(__AVX512VNNI__) && defined(__AVX512VL__)
#    define BLOCK_VNNI(acc, u, s) _mm256_dpbusd_epi32((acc), (u), (s))
#  endif
#endif

/* Rows of the weights a thread takes at a time. */
#define BLOCK_ROWS 64u

uint64_t spingalett_block_scratch(const SlettLayer *L, uint64_t items) {
    return slett_block_input_bytes(items * L->row_len);
}

/* An input quantized in blocks: item i's codes at xq + i C, its blocks' scales at dx + i C / S, its sums of 16
   codes at sums + i C / 16. */
typedef struct {
    const int8_t *xq;
    const float *dx;
    const int32_t *sums;
    uint32_t C, S;
} Quantized;

#if defined(BLOCK_AVX2)

/* ------------------------------------------------------------------------- the types */

/* What the kernels need of a type: its block, its sub-blocks, the offset its unpacked codes carry, whether it has
   minima, and whether its codes reach above 63 (whose products of two do not fit 16 bits). */
typedef struct {
    uint32_t values, sub;
    int32_t offset;
    bool mins, wide;
} BlockType;

static BlockType block_type(PrecisionMode p) {
    switch (p) {
        case PRECISION_Q8_0:   return (BlockType){32, 32, 128, false, true};
        case PRECISION_Q4_0:   return (BlockType){32, 32, 8, false, false};
        case PRECISION_Q4_1:   return (BlockType){32, 32, 0, true, false};
        case PRECISION_Q5_0:   return (BlockType){32, 32, 16, false, false};
        case PRECISION_Q5_1:   return (BlockType){32, 32, 0, true, false};
        case PRECISION_Q2_K:   return (BlockType){256, 16, 0, true, false};
        case PRECISION_Q3_K:   return (BlockType){256, 16, 4, false, false};
        case PRECISION_Q4_K:   return (BlockType){256, 32, 0, true, false};
        case PRECISION_Q5_K:   return (BlockType){256, 32, 0, true, false};
        case PRECISION_Q6_K:   return (BlockType){256, 16, 32, false, false};
        case PRECISION_IQ4_NL: return (BlockType){32, 32, 128, false, true};
        default:               return (BlockType){256, 32, 128, false, true};      /* IQ4_XS */
    }
}

/* A row's block unpacked: the weight of code c (stored c + offset) in sub-block j is d sc[j] c, plus m mn[j] for
   the types with minima. */
typedef struct {
    float d, m;
    int32_t sc[16], mn[16];
} Unpacked;

static inline float half_of(const uint8_t *p) { return _cvtsh_ss(slett_get16(p)); }

/* The values of IQ4_NL's and IQ4_XS's codes plus 128, as bytes. */
static const uint8_t iq4_unsigned[16] = {1, 24, 45, 63, 79, 93, 106, 118, 129, 141, 153, 166, 181, 197, 217, 241};

/* 32 codes of 4 bits: value j the low nibble of byte j, value j + 16 its high one. */
static inline __m256i nibbles(const uint8_t *qs) {
    const __m128i v = _mm_loadu_si128((const __m128i *)(const void *)qs), low = _mm_set1_epi8(0x0F);
    return _mm256_set_m128i(_mm_and_si128(_mm_srli_epi16(v, 4), low), _mm_and_si128(v, low));
}

/* 32 bits as 32 bytes: byte k 0x10 when bit k is set. */
static inline __m256i fifth_bits(uint32_t qh) {
    const __m256i bytes = _mm256_shuffle_epi8(_mm256_set1_epi32((int)qh),
                                              _mm256_set_epi64x(0x0303030303030303, 0x0202020202020202,
                                                                0x0101010101010101, 0x0000000000000000));
    const __m256i bit = _mm256_set1_epi64x((long long)0x8040201008040201ull);
    return _mm256_and_si256(_mm256_cmpeq_epi8(_mm256_and_si256(bytes, bit), bit), _mm256_set1_epi8(0x10));
}

/* The 6-bit scale and minimum j of a Q4_K or Q5_K super-block (llama.cpp's get_scale_min_k4()). */
static inline void scale_min_k4(uint32_t j, const uint8_t *q, int32_t *d, int32_t *m) {
    if (j < 4u) {
        *d = q[j] & 63;
        *m = q[j + 4u] & 63;
    } else {
        *d = (q[j + 4u] & 0xF) | ((q[j - 4u] >> 6) << 4);
        *m = (q[j + 4u] >> 4) | ((q[j] >> 6) << 4);
    }
}

static inline __attribute__((always_inline)) void unpack(PrecisionMode p, const uint8_t *w, uint8_t *u, Unpacked *s) {
    const __m256i low = _mm256_set1_epi8(0x0F), three = _mm256_set1_epi8(3);
    switch (p) {
        case PRECISION_Q8_0:
            s->d = half_of(w);
            s->sc[0] = 1;
            _mm256_storeu_si256((__m256i *)(void *)u, _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)(const void *)(w + 2)),
                                                                        _mm256_set1_epi8((char)0x80)));
            return;
        case PRECISION_Q4_0:
        case PRECISION_IQ4_NL: {
            s->d = half_of(w);
            s->sc[0] = 1;
            __m256i c = nibbles(w + 2);
            if (p == PRECISION_IQ4_NL)
                c = _mm256_shuffle_epi8(_mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)(const void *)iq4_unsigned)), c);
            _mm256_storeu_si256((__m256i *)(void *)u, c);
            return;
        }
        case PRECISION_Q4_1:
            s->d = half_of(w);
            s->m = half_of(w + 2);
            s->sc[0] = s->mn[0] = 1;
            _mm256_storeu_si256((__m256i *)(void *)u, nibbles(w + 4));
            return;
        case PRECISION_Q5_0:
        case PRECISION_Q5_1: {
            const bool one = p == PRECISION_Q5_1;
            s->d = half_of(w);
            s->m = one ? half_of(w + 2) : 0.0f;
            s->sc[0] = s->mn[0] = 1;
            _mm256_storeu_si256((__m256i *)(void *)u, _mm256_or_si256(nibbles(w + (one ? 8 : 6)),
                                                                       fifth_bits(slett_get32(w + (one ? 4 : 2)))));
            return;
        }
        case PRECISION_Q2_K:
        case PRECISION_Q3_K: {
            /* value 128 h + 32 j + l: bits 2 j of byte 32 h + l (Q3_K: plus 4 for bit 4 h + j of hmask[l]) */
            const bool three_bits = p == PRECISION_Q3_K;
            const uint8_t *q = w + (three_bits ? 32 : 16);
            const __m256i hm = _mm256_loadu_si256((const __m256i *)(const void *)w);
            for (uint32_t h = 0; h < 2u; h++) {
                const __m256i v = _mm256_loadu_si256((const __m256i *)(const void *)(q + 32u * h));
                for (uint32_t j = 0; j < 4u; j++) {
                    __m256i c = _mm256_and_si256(_mm256_srli_epi16(v, (int)(2u * j)), three);
                    if (three_bits) {
                        const __m256i bit = _mm256_and_si256(_mm256_srli_epi16(hm, (int)(4u * h + j)), _mm256_set1_epi8(1));
                        c = _mm256_or_si256(c, _mm256_slli_epi16(bit, 2));
                    }
                    _mm256_storeu_si256((__m256i *)(void *)(u + 128u * h + 32u * j), c);
                }
            }
            if (three_bits) {
                const uint32_t kmask1 = 0x03030303u, kmask2 = 0x0f0f0f0fu;
                uint32_t aux[4];
                aux[0] = slett_get32(w + 96);
                aux[1] = slett_get32(w + 100);
                const uint32_t tmp = slett_get32(w + 104);
                aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
                aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
                aux[0] = (aux[0] & kmask2) | ((tmp & kmask1) << 4);
                aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
                for (uint32_t k = 0; k < 16u; k++) s->sc[k] = (int32_t)((aux[k / 4u] >> (8u * (k % 4u))) & 0xFFu) - 32;
                s->d = half_of(w + 108);
            } else {
                for (uint32_t k = 0; k < 16u; k++) {
                    s->sc[k] = w[k] & 0xF;
                    s->mn[k] = w[k] >> 4;
                }
                s->d = half_of(w + 80);
                s->m = -half_of(w + 82);
            }
            return;
        }
        case PRECISION_Q4_K:
        case PRECISION_Q5_K: {
            /* value 64 c + 32 half + l: the low (half 0) or high nibble of byte 32 c + l (Q5_K: plus 16 for bit 2 c +
               half of qh[l]) */
            const bool five = p == PRECISION_Q5_K;
            const uint8_t *q = w + (five ? 48 : 16);
            const __m256i qh = _mm256_loadu_si256((const __m256i *)(const void *)(w + 16)), one = _mm256_set1_epi8(1);
            for (uint32_t c = 0; c < 4u; c++) {
                const __m256i v = _mm256_loadu_si256((const __m256i *)(const void *)(q + 32u * c));
                __m256i lo = _mm256_and_si256(v, low), hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), low);
                if (five) {
                    lo = _mm256_or_si256(lo, _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qh, (int)(2u * c)), one), 4));
                    hi = _mm256_or_si256(hi, _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qh, (int)(2u * c + 1u)), one), 4));
                }
                _mm256_storeu_si256((__m256i *)(void *)(u + 64u * c), lo);
                _mm256_storeu_si256((__m256i *)(void *)(u + 64u * c + 32u), hi);
            }
            for (uint32_t j = 0; j < 8u; j++) scale_min_k4(j, w + 4, &s->sc[j], &s->mn[j]);
            s->d = half_of(w);
            s->m = -half_of(w + 2);
            return;
        }
        case PRECISION_Q6_K: {
            /* value 128 h + 32 g + l: the low (g < 2) or high nibble of ql[64 h + 32 (g mod 2) + l] plus 16 times bits
               2 g of qh[32 h + l] */
            for (uint32_t h = 0; h < 2u; h++) {
                const __m256i l0 = _mm256_loadu_si256((const __m256i *)(const void *)(w + 64u * h));
                const __m256i l1 = _mm256_loadu_si256((const __m256i *)(const void *)(w + 64u * h + 32u));
                const __m256i hb = _mm256_loadu_si256((const __m256i *)(const void *)(w + 128u + 32u * h));
                const __m256i lows[4] = {_mm256_and_si256(l0, low), _mm256_and_si256(l1, low),
                                         _mm256_and_si256(_mm256_srli_epi16(l0, 4), low),
                                         _mm256_and_si256(_mm256_srli_epi16(l1, 4), low)};
                for (uint32_t g = 0; g < 4u; g++) {
                    const __m256i high = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(hb, (int)(2u * g)), three), 4);
                    _mm256_storeu_si256((__m256i *)(void *)(u + 128u * h + 32u * g), _mm256_or_si256(lows[g], high));
                }
            }
            for (uint32_t k = 0; k < 16u; k++) s->sc[k] = (int8_t)w[192 + k];
            s->d = half_of(w + 208);
            return;
        }
        default: {                                  /* IQ4_XS */
            const __m256i table = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)(const void *)iq4_unsigned));
            const uint32_t high = slett_get16(w + 2);
            for (uint32_t ib = 0; ib < 8u; ib++) {
                _mm256_storeu_si256((__m256i *)(void *)(u + 32u * ib), _mm256_shuffle_epi8(table, nibbles(w + 8u + 16u * ib)));
                const uint32_t lowb = w[4u + ib / 2u];
                s->sc[ib] = (int32_t)(((lowb >> (4u * (ib % 2u))) & 0xFu) | (((high >> (2u * ib)) & 3u) << 4)) - 32;
            }
            s->d = half_of(w);
            return;
        }
    }
}

/* ------------------------------------------------------------------------- products */

/* The products of 32 unsigned codes with 32 signed codes, in eight lanes of four. Without VNNI, codes of 63 at most
   go through maddubs (16-bit pairs cannot saturate); wider ones as signed codes (u - 128) by the sign trick, the
   128 x taken back. */
static inline __m256i products(__m256i u, __m256i x, bool wide) {
#if defined(BLOCK_VNNI)
    (void)wide;
    return BLOCK_VNNI(_mm256_setzero_si256(), u, x);
#else
    const __m256i ones = _mm256_set1_epi16(1);
    if (!wide) return _mm256_madd_epi16(_mm256_maddubs_epi16(u, x), ones);
    const __m256i q = _mm256_xor_si256(u, _mm256_set1_epi8((char)0x80));
    const __m256i part = _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_sign_epi8(q, q), _mm256_sign_epi8(x, q)), ones);
    const __m256i x128 = _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_set1_epi8((char)0x80), x), ones);
    return _mm256_add_epi32(part, x128);
#endif
}

/* The sums of four accumulators: lane r of the result is the total of a_r. */
static inline __m128i reduce4(__m256i a0, __m256i a1, __m256i a2, __m256i a3) {
    const __m256i t = _mm256_hadd_epi32(_mm256_hadd_epi32(a0, a1), _mm256_hadd_epi32(a2, a3));
    return _mm_add_epi32(_mm256_castsi256_si128(t), _mm256_extracti128_si256(t, 1));
}

/* The integer sum of a row's unpacked block with an item's codes, scaled by the sub-blocks' scales, in eight lanes
   (the offset's part not taken off). */
static inline __attribute__((always_inline)) __m256i block_lanes(const BlockType *t, const uint8_t *u, const Unpacked *s, const int8_t *x) {
    __m256i acc = _mm256_setzero_si256();
    for (uint32_t k = 0; k < t->values; k += 32u) {
        const __m256i p = products(_mm256_loadu_si256((const __m256i *)(const void *)(u + k)),
                                   _mm256_loadu_si256((const __m256i *)(const void *)(x + k)), t->wide);
        if (t->values == 32u) return p;                         /* (one sub-block, of scale 1) */
        const __m256i sc = t->sub == 32u ? _mm256_set1_epi32(s->sc[k / 32u])
                         : _mm256_set_m128i(_mm_set1_epi32(s->sc[k / 16u + 1u]), _mm_set1_epi32(s->sc[k / 16u]));
        acc = _mm256_add_epi32(acc, _mm256_mullo_epi32(p, sc));
    }
    return acc;
}

/* The offset's part of I1 (the offset times the scaled sums of the item's codes) and I2 of a row's block. */
static inline __attribute__((always_inline)) void block_terms(const BlockType *t, const Unpacked *s, const int32_t *X, int32_t *offset_part, int32_t *i2) {
    const uint32_t subs = t->values / t->sub, per = t->sub / 16u;
    int32_t o = 0, m = 0;
    for (uint32_t j = 0; j < subs; j++) {
        const int32_t Xj = per == 2u ? X[2u * j] + X[2u * j + 1u] : X[j];
        o += s->sc[j] * Xj;
        if (t->mins) m += s->mn[j] * Xj;
    }
    *offset_part = t->offset * o;
    *i2 = m;
}

/* The codes of a block of 32 (unsigned, with the type's offset) in a vector, and its scale and minimum. */
static inline __attribute__((always_inline)) __m256i codes32(PrecisionMode p, const uint8_t *w, float *d, float *m) {
    *d = half_of(w);
    switch (p) {
        case PRECISION_Q8_0:
            return _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)(const void *)(w + 2)), _mm256_set1_epi8((char)0x80));
        case PRECISION_Q4_0:
            return nibbles(w + 2);
        case PRECISION_IQ4_NL:
            return _mm256_shuffle_epi8(_mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)(const void *)iq4_unsigned)),
                                       nibbles(w + 2));
        case PRECISION_Q4_1:
            *m = half_of(w + 2);
            return nibbles(w + 4);
        case PRECISION_Q5_0:
            return _mm256_or_si256(nibbles(w + 6), fifth_bits(slett_get32(w + 2)));
        default:                                    /* Q5_1 */
            *m = half_of(w + 2);
            return _mm256_or_si256(nibbles(w + 8), fifth_bits(slett_get32(w + 4)));
    }
}

/* Rows j0 .. j0 + 4 n - 1 for `items` items: y[i OC + j] = bias[j] + the row's dot product with item i, the four rows
   of a group side by side, each lane's float sum in the reference's order; up to 8 items at a time share each
   row's unpacked block. Compiled once for each type (the type a constant). */
static inline __attribute__((always_inline)) void rows_by_four_of(const PrecisionMode p, const uint8_t *weights, size_t row,
                                                                  const float *bias, uint32_t j0, uint32_t n,
                                                                  const Quantized *q, uint32_t items, float *y, uint32_t OC) {
    const BlockType t = block_type(p);
    const uint32_t S = t.values, blocks = q->C / S, B = slett_block_bytes(p), per = q->C / 16u;
    uint8_t u[4][256] __attribute__((aligned(32)));
    Unpacked s[4];
    memset(s, 0, sizeof s);
    for (uint32_t g = 0; g < n; g++) {
        const uint32_t j = j0 + 4u * g;
        const uint8_t *w0 = weights + (size_t)j * row;
        for (uint32_t i0 = 0; i0 < items; i0 += 8u) {
            const uint32_t count = items - i0 < 8u ? items - i0 : 8u;
            __m128 acc[8];
            for (uint32_t i = 0; i < count; i++) acc[i] = _mm_setzero_ps();
            for (uint32_t b = 0; b < blocks; b++) {
                const uint8_t *wb = w0 + (size_t)b * B;
                if (S == 32u) {
                    /* the four rows' codes in registers; one sub-block of scale 1, the offset's and the minimum's parts
                       those of the item's sum, alike in every row */
                    float d[4], m[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                    const __m256i c0 = codes32(p, wb, &d[0], &m[0]), c1 = codes32(p, wb + row, &d[1], &m[1]);
                    const __m256i c2 = codes32(p, wb + 2u * row, &d[2], &m[2]), c3 = codes32(p, wb + 3u * row, &d[3], &m[3]);
                    const __m128 dv = _mm_loadu_ps(d), mv = _mm_loadu_ps(m);
                    for (uint32_t i = 0; i < count; i++) {
                        const uint32_t item = i0 + i;
                        const __m256i x = _mm256_loadu_si256((const __m256i *)(const void *)(q->xq + (size_t)item * q->C +
                                                                                              (size_t)b * 32u));
                        const int32_t X = q->sums[(size_t)item * per + 2u * b] + q->sums[(size_t)item * per + 2u * b + 1u];
                        __m128i i1 = reduce4(products(c0, x, t.wide), products(c1, x, t.wide), products(c2, x, t.wide),
                                             products(c3, x, t.wide));
                        if (t.offset) i1 = _mm_sub_epi32(i1, _mm_set1_epi32(t.offset * X));
                        const __m128 dx = _mm_set1_ps(q->dx[(size_t)item * blocks + b]);
                        __m128 term = _mm_mul_ps(_mm_mul_ps(dv, dx), _mm_cvtepi32_ps(i1));
                        if (t.mins) term = _mm_add_ps(term, _mm_mul_ps(_mm_mul_ps(mv, dx), _mm_set1_ps((float)X)));
                        acc[i] = _mm_add_ps(acc[i], term);
                    }
                    continue;
                }
                for (uint32_t r = 0; r < 4u; r++) unpack(p, wb + (size_t)r * row, u[r], &s[r]);
                const __m128 dv = _mm_set_ps(s[3].d, s[2].d, s[1].d, s[0].d), mv = _mm_set_ps(s[3].m, s[2].m, s[1].m, s[0].m);
                for (uint32_t i = 0; i < count; i++) {
                    const uint32_t item = i0 + i;
                    const int8_t *x = q->xq + (size_t)item * q->C + (size_t)b * S;
                    const int32_t *X = q->sums + (size_t)item * per + (size_t)b * (S / 16u);
                    __m128i i1 = reduce4(block_lanes(&t, u[0], &s[0], x), block_lanes(&t, u[1], &s[1], x),
                                         block_lanes(&t, u[2], &s[2], x), block_lanes(&t, u[3], &s[3], x));
                    int32_t off0, off1, off2, off3, m0, m1, m2, m3;
                    block_terms(&t, &s[0], X, &off0, &m0);
                    block_terms(&t, &s[1], X, &off1, &m1);
                    block_terms(&t, &s[2], X, &off2, &m2);
                    block_terms(&t, &s[3], X, &off3, &m3);
                    if (t.offset) i1 = _mm_sub_epi32(i1, _mm_set_epi32(off3, off2, off1, off0));
                    const __m128 dx = _mm_set1_ps(q->dx[(size_t)item * blocks + b]);
                    __m128 term = _mm_mul_ps(_mm_mul_ps(dv, dx), _mm_cvtepi32_ps(i1));
                    if (t.mins)
                        term = _mm_add_ps(term, _mm_mul_ps(_mm_mul_ps(mv, dx), _mm_cvtepi32_ps(_mm_set_epi32(m3, m2, m1, m0))));
                    acc[i] = _mm_add_ps(acc[i], term);
                }
            }
            for (uint32_t i = 0; i < count; i++) {
                float out[4];
                _mm_storeu_ps(out, acc[i]);
                for (uint32_t r = 0; r < 4u; r++) y[(size_t)(i0 + i) * OC + j + r] = bias[j + r] + out[r];
            }
        }
    }
}

static void rows_by_four(PrecisionMode p, const uint8_t *weights, size_t row, const float *bias, uint32_t j0, uint32_t n,
                         const Quantized *q, uint32_t items, float *y, uint32_t OC) {
#define ROWS_OF(type) case type: rows_by_four_of(type, weights, row, bias, j0, n, q, items, y, OC); return;
    switch (p) {
        ROWS_OF(PRECISION_Q8_0) ROWS_OF(PRECISION_Q4_0) ROWS_OF(PRECISION_Q4_1) ROWS_OF(PRECISION_Q5_0)
        ROWS_OF(PRECISION_Q5_1) ROWS_OF(PRECISION_Q2_K) ROWS_OF(PRECISION_Q3_K) ROWS_OF(PRECISION_Q4_K)
        ROWS_OF(PRECISION_Q5_K) ROWS_OF(PRECISION_Q6_K) ROWS_OF(PRECISION_IQ4_NL)
        default: rows_by_four_of(PRECISION_IQ4_XS, weights, row, bias, j0, n, q, items, y, OC); return;
    }
#undef ROWS_OF
}

/* ------------------------------------------------------------------------- quantizing the inputs */

/* n values (a multiple of block) in blocks, as spingalett_quantize_blocks() quantizes them, bit for bit: per block the
   largest magnitude (NaN for a block with a value that is not finite, whose codes are 0), its scale amax / 127 and
   the codes round(x 127 / amax) to nearest even; then the sums of every 16 codes. */
static void quantize_row(const float *x, uint32_t n, uint32_t block, int8_t *xq, float *dx, int32_t *sums) {
    const __m256 sign = _mm256_set1_ps(-0.0f), big = _mm256_set1_ps(FLT_MAX);
    for (uint32_t b = 0; b < n / block; b++) {
        const float *xb = x + (size_t)b * block;
        int8_t *qb = xq + (size_t)b * block;
        __m256 vmax = _mm256_setzero_ps(), vbad = _mm256_setzero_ps();
        for (uint32_t k = 0; k < block; k += 8u) {
            const __m256 a = _mm256_andnot_ps(sign, _mm256_loadu_ps(xb + k));
            vbad = _mm256_or_ps(vbad, _mm256_cmp_ps(a, big, _CMP_NLE_UQ));
            vmax = _mm256_max_ps(vmax, a);
        }
        __m128 mx = _mm_max_ps(_mm256_castps256_ps128(vmax), _mm256_extractf128_ps(vmax, 1));
        mx = _mm_max_ps(mx, _mm_movehl_ps(mx, mx));
        mx = _mm_max_ss(mx, _mm_shuffle_ps(mx, mx, 0x55));
        const float amax = _mm_cvtss_f32(mx);
        if (_mm256_movemask_ps(vbad) || amax == 0.0f) {
            memset(qb, 0, block);
            dx[b] = _mm256_movemask_ps(vbad) ? NAN : 0.0f;
            continue;
        }
        const __m256 inv = _mm256_set1_ps(127.0f / amax);
        for (uint32_t k = 0; k < block; k += 32u) {
            const __m256i i0 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(xb + k), inv));
            const __m256i i1 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(xb + k + 8), inv));
            const __m256i i2 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(xb + k + 16), inv));
            const __m256i i3 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(xb + k + 24), inv));
            /* (the packs interleave the lanes of 128 bits; the permutation puts them back in order) */
            const __m256i packed = _mm256_packs_epi16(_mm256_packs_epi32(i0, i1), _mm256_packs_epi32(i2, i3));
            _mm256_storeu_si256((__m256i *)(void *)(qb + k),
                                _mm256_permutevar8x32_epi32(packed, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7)));
        }
        dx[b] = amax / 127.0f;
    }
    /* the sums of 16 codes: of pairs of bytes, of fours, then of the two quarters of each half */
    const __m256i ones8 = _mm256_set1_epi8(1), ones16 = _mm256_set1_epi16(1);
    for (uint32_t k = 0; k < n; k += 32u) {
        const __m256i v = _mm256_loadu_si256((const __m256i *)(const void *)(xq + k));
        const __m256i fours = _mm256_madd_epi16(_mm256_maddubs_epi16(ones8, v), ones16);
        const __m256i eights = _mm256_hadd_epi32(fours, fours);
        const __m256i sixteens = _mm256_hadd_epi32(eights, eights);
        sums[k / 16u] = _mm256_cvtsi256_si32(sixteens);
        sums[k / 16u + 1u] = _mm_cvtsi128_si32(_mm256_extracti128_si256(sixteens, 1));
    }
}

#endif /* BLOCK_AVX2 */

/* Item i of n values quantized in blocks of S. */
static void quantize_item(const float *x, uint32_t C, uint32_t S, int8_t *xq, float *dx, int32_t *sums) {
#if defined(BLOCK_AVX2)
    quantize_row(x, C, S, xq, dx, sums);
#else
    spingalett_quantize_blocks(x, C, S, xq, dx, sums);
#endif
}

/* Rows j0 .. end - 1 of the layer for every item. */
static void rows_task(PrecisionMode p, const uint8_t *weights, size_t row, const float *bias, uint32_t j0, uint32_t end,
                      const Quantized *q, uint64_t items, float *y, uint32_t OC) {
    uint32_t j = j0;
#if defined(BLOCK_AVX2)
    const uint32_t fours = (end - j0) / 4u;
    rows_by_four(p, weights, row, bias, j0, fours, q, (uint32_t)items, y, OC);
    j += 4u * fours;
#endif
    const uint32_t C = q->C, S = q->S;
    for (uint64_t i = 0; i < items; i++)
        for (uint32_t r = j; r < end; r++)
            y[i * OC + r] = bias[r] + spingalett_block_dot(p, weights + r * row, q->xq + i * C, q->dx + i * (C / S),
                                                           q->sums + i * (C / 16u), C);
}

void spingalett_block_layer(const uint8_t *image, const SlettLayer *L, const float *x, float *y, uint64_t items,
                            uint8_t *scratch, int threads) {
    const PrecisionMode p = L->precision;
    const uint32_t C = L->row_len, OC = L->rows, S = slett_block_values(p);
    const uint64_t n = items * C;
    int8_t *xq = (int8_t *)scratch;
    float *dx = (float *)(void *)(scratch + slett_align(n));
    int32_t *sums = (int32_t *)(void *)(scratch + slett_align(n) + slett_align(n / 8u));
    const float *bias = (const float *)(const void *)(image + L->biases);
    const uint8_t *weights = image + L->weights;
    const size_t row = (size_t)spingalett_slett_row_bytes(p, C);
    const Quantized q = {xq, dx, sums, C, S};
    (void)threads;
    SPINGALETT_PARALLEL_FOR_THREADS(threads, threads > 1 && items > 1,
        for (int64_t i = 0; i < (int64_t)items; i++)
            quantize_item(x + (size_t)i * C, C, S, xq + (size_t)i * C, dx + (size_t)i * (C / S), sums + (size_t)i * (C / 16u));
    );
    const int64_t blocks = (int64_t)((OC + BLOCK_ROWS - 1u) / BLOCK_ROWS);
    SPINGALETT_PARALLEL_FOR_THREADS(threads, threads > 1,
        for (int64_t b = 0; b < blocks; b++) {
            const uint32_t j0 = (uint32_t)b * BLOCK_ROWS, end = OC - j0 < BLOCK_ROWS ? OC : j0 + BLOCK_ROWS;
            rows_task(p, weights, row, bias, j0, end, &q, items, y, OC);
        }
    );
}
