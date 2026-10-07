/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/*
 * .slettd data set files (layout: docs/DatasetFormat.md).
 *
 * Each chunk holds one stream of inputs and one stream per set of targets. A stream is the encoded
 * values of the chunk split into byte planes (byte k of every value together), either stored or
 * coded with an adaptive context model. Two coders exist: a binary range coder (a "zero?" decision,
 * then the eight bits of a non-zero byte down a bit tree), strongest on sparse data such as MNIST,
 * and an rANS coder of half-bytes (two 16-symbol decisions per byte, two interleaved states),
 * which decodes dense data such as photographs two to three times as fast at the same size. Both
 * model every byte from the previous byte and from the bytes `stride` and `stride + 1` positions
 * back, where the writer picks the stride that predicts best (for images, the row length, so the
 * pixel above is part of the context), and the writer picks the coder per stream from the first
 * chunk.
 *
 * Readers either load whole files (chunks decode in parallel), stream them (a background thread
 * reads and decodes the next chunks while the caller trains), or keep their values in memory in
 * the compact encoded form and convert a batch at a time.
 */

#include "Spingalett.Private.h"
#include "Spingalett.Thread.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#if defined(_OPENMP)
#include <omp.h>
#endif

#if !defined(SPINGALETT_PORTABLE_KERNELS)
#  if defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
#    include <emmintrin.h>
#    define DS_SSE2 1
#  elif defined(__aarch64__) && defined(__ARM_NEON)
#    include <arm_neon.h>
#    define DS_NEON 1
#  endif
#endif

#define SLETTD_VERSION_MAX    2
#define SLETTD_HEADER_SIZE    64
#define SLETTD_CHUNK_TARGET   (1u << 20)    /* encoded bytes per chunk the writer aims for */
#define SLETTD_MAX_SETS       255u          /* sets of targets */
#define SLETTD_MAX_EXPANSION  8192u         /* decoded bytes per coded byte, beyond what the coders reach */

enum { STREAM_STORED = 0, STREAM_CODED = 1, STREAM_RANS = 2 };
enum { META_SHAPE = 1, META_TARGETS = 2, META_NAME = 3, META_CLASSES = 4 };

/* ---------------------------------------------------------------- little-endian helpers */

static void put16(uint8_t *p, uint16_t v) { p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8); }
static void put32(uint8_t *p, uint32_t v) { for (int i = 0; i < 4; i++) p[i] = (uint8_t)(v >> (8 * i)); }
static void put64(uint8_t *p, uint64_t v) { for (int i = 0; i < 8; i++) p[i] = (uint8_t)(v >> (8 * i)); }
static uint16_t get16(const uint8_t *p) { return (uint16_t)(p[0] | p[1] << 8); }
static uint32_t get32(const uint8_t *p) { return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24; }
static uint64_t get64(const uint8_t *p) { return (uint64_t)get32(p) | (uint64_t)get32(p + 4) << 32; }
static uint32_t float_bits(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
static float bits_float(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }

/* CRC-32: spingalett_crc32 (Spingalett.Inference.c), shared with model files. */
#define crc32_update spingalett_crc32

/* ---------------------------------------------------------------- binary range coder */

#define RC_TOP      (1u << 24)
#define PROB_BITS   11
#define PROB_ONE    (1u << PROB_BITS)
#define MOVE_BITS   4

typedef struct {
    uint8_t *out;
    size_t pos, cap;
    uint64_t low;
    uint32_t range;
    uint8_t cache;
    uint64_t cache_size;
    bool overflow;
} RcEncoder;

static void rc_put(RcEncoder *e, uint8_t b) {
    if (e->pos < e->cap) e->out[e->pos++] = b;
    else e->overflow = true;
}

static void rc_shift_low(RcEncoder *e) {
    if ((uint32_t)e->low < 0xFF000000u || (e->low >> 32) != 0) {
        uint8_t carry = (uint8_t)(e->low >> 32), temp = e->cache;
        do {
            rc_put(e, (uint8_t)(temp + carry));
            temp = 0xFF;
        } while (--e->cache_size != 0);
        e->cache = (uint8_t)(e->low >> 24);
    }
    e->cache_size++;
    e->low = (e->low & 0x00FFFFFFu) << 8;
}

static inline void rc_encode_bit(RcEncoder *e, uint16_t *p, unsigned bit) {
    uint32_t bound = (e->range >> PROB_BITS) * *p;
    if (!bit) {
        e->range = bound;
        *p = (uint16_t)(*p + ((PROB_ONE - *p) >> MOVE_BITS));
    } else {
        e->low += bound;
        e->range -= bound;
        *p = (uint16_t)(*p - (*p >> MOVE_BITS));
    }
    while (e->range < RC_TOP) {
        e->range <<= 8;
        rc_shift_low(e);
    }
}

/* ---------------------------------------------------------------- byte model of the binary coder */

/* "Is the byte zero?" in a context of the previous byte and the two bytes at the stride, then the
   value of a non-zero byte as 8 binary decisions down a bit tree, in a smaller context. */
#define CTX_ZERO 2048
#define CTX_TREE 256

typedef struct {
    uint16_t zero[CTX_ZERO];
    uint16_t tree[CTX_TREE][256];
} ByteModel;

static void model_reset(ByteModel *m) {
    for (size_t i = 0; i < CTX_ZERO; i++) m->zero[i] = PROB_ONE / 2;
    for (size_t c = 0; c < CTX_TREE; c++)
        for (size_t i = 0; i < 256; i++) m->tree[c][i] = PROB_ONE / 2;
}

static inline void contexts(const uint8_t *b, size_t i, size_t stride, uint32_t *cz, uint32_t *ct) {
    uint32_t p1 = i >= 1 ? b[i - 1] : 0;
    uint32_t ps = i >= stride ? b[i - stride] : 0;
    uint32_t ps1 = i >= stride + 1 ? b[i - stride - 1] : 0;
    *cz = (p1 >> 4) << 7 | (ps >> 4) << 3 | ps1 >> 5;
    *ct = (p1 >> 4) << 4 | ps >> 4;
}

static void encode_plane(RcEncoder *e, ByteModel *m, const uint8_t *b, size_t n, size_t stride) {
    model_reset(m);
    for (size_t i = 0; i < n; i++) {
        uint32_t cz, ct;
        contexts(b, i, stride, &cz, &ct);
        unsigned v = b[i];
        rc_encode_bit(e, &m->zero[cz], v != 0);
        if (v) {
            uint16_t *t = m->tree[ct];
            unsigned node = 1;
            for (int k = 7; k >= 0; k--) {
                unsigned bit = (v >> k) & 1u;
                rc_encode_bit(e, &t[node], bit);
                node = node << 1 | bit;
            }
        }
    }
}

/* Codes `planes` planes of n bytes each into out (at most cap bytes); returns the size, or 0 when
   the result would not be smaller than cap. */
static size_t encode_stream(const uint8_t *data, size_t n, unsigned planes, size_t stride,
                            ByteModel *m, uint8_t *out, size_t cap) {
    RcEncoder e = {.out = out, .cap = cap, .range = 0xFFFFFFFFu, .cache_size = 1};
    for (unsigned p = 0; p < planes && !e.overflow; p++)
        encode_plane(&e, m, data + p * n, n, stride);
    for (int i = 0; i < 5; i++) rc_shift_low(&e);
    return e.overflow ? 0 : e.pos;
}

/* One binary decision with the decoder's state in locals (stores to the output bytes would make
   the compiler reload a state kept in memory after every byte). */
#define RC_BIT(prob, bit) do { \
        uint32_t p_ = *(prob), bound_ = (range >> PROB_BITS) * p_; \
        if (code < bound_) { \
            range = bound_; \
            *(prob) = (uint16_t)(p_ + ((PROB_ONE - p_) >> MOVE_BITS)); \
            (bit) = 0; \
        } else { \
            code -= bound_; \
            range -= bound_; \
            *(prob) = (uint16_t)(p_ - (p_ >> MOVE_BITS)); \
            (bit) = 1; \
        } \
        if (range < RC_TOP) { \
            range <<= 8; \
            code = (code << 8) | (pos < size ? in[pos] : 0u); \
            pos++; \
        } \
    } while (0)

static bool decode_stream(const uint8_t *in, size_t size, uint8_t *data, size_t n, unsigned planes,
                          size_t stride, ByteModel *m) {
    size_t pos = 0;
    uint32_t range = 0xFFFFFFFFu, code = 0;
    for (int i = 0; i < 5; i++, pos++) code = (code << 8) | (pos < size ? in[pos] : 0u);
    for (unsigned p = 0; p < planes; p++) {
        uint8_t *b = data + p * n;
        model_reset(m);
        for (size_t i = 0; i < n; i++) {
            uint32_t cz, ct;
            contexts(b, i, stride, &cz, &ct);
            unsigned bit, v = 0;
            RC_BIT(&m->zero[cz], bit);
            if (bit) {
                uint16_t *t = m->tree[ct];
                unsigned node = 1;
                for (int k = 0; k < 8; k++) {
                    RC_BIT(&t[node], bit);
                    node = node << 1 | bit;
                }
                v = node & 0xFFu;
            }
            b[i] = (uint8_t)v;
        }
    }
    return pos <= size;            /* reading past the end means the stream is corrupt */
}

#undef RC_BIT

/* ---------------------------------------------------------------- rANS coder of half-bytes */

/* A byte is two symbols of 16 values, its high and its low half, each coded with an adaptive
   frequency table in a context: the high half in one of the previous byte's and the bytes at the
   stride (1024 contexts), the low half in one of the previous byte's, the byte at the stride and
   the high half (4096). The two halves of a byte alternate between two rANS states (32 bits,
   renormalized 16 bits at a time), which the decoder advances side by side. */

#define RANS_SCALE  15
#define RANS_M      (1u << RANS_SCALE)          /* table total */
#define RANS_L      (1u << 16)                  /* lower bound of a state */
#define NIB_RATE_MIN 3                          /* adaptation shift of a new table ... */
#define NIB_RATE_MAX 6                          /* ... growing to this after a few updates */
#define CTX_HI 1024
#define CTX_LO 4096

/* A table: c[k] (k < 15) is the cumulative frequency of symbols 0..k, those of symbols 0 and 16
   being 0 and RANS_M; c[15] counts updates (it sets the adaptation rate). Every symbol keeps a
   frequency of at least 1. */
typedef struct { int16_t c[16]; } Nibbles;

typedef struct {
    Nibbles hi[CTX_HI];
    Nibbles lo[CTX_LO];
} NibbleModel;

static void nibbles_reset(Nibbles *t) {
    for (int k = 0; k < 15; k++) t->c[k] = (int16_t)((k + 1) * (RANS_M / 16));
    t->c[15] = 0;
}

static void nibble_model_reset(NibbleModel *m) {
    for (size_t i = 0; i < CTX_HI; i++) nibbles_reset(&m->hi[i]);
    for (size_t i = 0; i < CTX_LO; i++) nibbles_reset(&m->lo[i]);
}

static inline uint32_t nib_start(const Nibbles *t, unsigned s) { return s ? (uint32_t)(uint16_t)t->c[s - 1] : 0u; }
static inline uint32_t nib_end(const Nibbles *t, unsigned s) { return s < 15 ? (uint32_t)(uint16_t)t->c[s] : RANS_M; }
static inline int nib_rate(const Nibbles *t) { return NIB_RATE_MIN + t->c[15]; }

#if !defined(DS_SSE2) && !defined(DS_NEON)
/* floor(d / 2^r), as an arithmetic shift computes it */
static inline int floor_shift(int d, int r) { return d >= 0 ? d >> r : -((-d + (1 << r) - 1) >> r); }
#endif

/* After symbol s, each cumulative frequency moves 2^-rate of the way towards k + 1 (symbols up to
   s) or RANS_M - 15 + k (beyond s): frequencies stay at least 1, and the symbol gains. The vector
   versions compute the same values. */
static inline void nib_update(Nibbles *t, unsigned s) {
    int r = nib_rate(t);
#if defined(DS_SSE2)
    const __m128i idx0 = _mm_setr_epi16(1, 2, 3, 4, 5, 6, 7, 8), idx1 = _mm_setr_epi16(9, 10, 11, 12, 13, 14, 15, 16);
    const __m128i sv = _mm_set1_epi16((short)s), add = _mm_set1_epi16((short)(RANS_M - 16));
    const __m128i cnt = _mm_cvtsi32_si128(r);
    __m128i a = _mm_loadu_si128((const __m128i *)(const void *)t->c);
    __m128i b = _mm_loadu_si128((const __m128i *)(const void *)(t->c + 8));
    __m128i ta = _mm_add_epi16(idx0, _mm_and_si128(_mm_cmpgt_epi16(idx0, sv), add));
    __m128i tb = _mm_add_epi16(idx1, _mm_and_si128(_mm_cmpgt_epi16(idx1, sv), add));
    a = _mm_add_epi16(a, _mm_sra_epi16(_mm_sub_epi16(ta, a), cnt));
    b = _mm_add_epi16(b, _mm_sra_epi16(_mm_sub_epi16(tb, b), cnt));
    int count = t->c[15];
    b = _mm_insert_epi16(b, count < NIB_RATE_MAX - NIB_RATE_MIN ? count + 1 : count, 7);
    _mm_storeu_si128((__m128i *)(void *)t->c, a);
    _mm_storeu_si128((__m128i *)(void *)(t->c + 8), b);
#elif defined(DS_NEON)
    const int16_t idx[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    int16x8_t i0 = vld1q_s16(idx), i1 = vld1q_s16(idx + 8);
    int16x8_t sv = vdupq_n_s16((int16_t)s), add = vdupq_n_s16((int16_t)(RANS_M - 16)), sh = vdupq_n_s16((int16_t)-r);
    int16x8_t a = vld1q_s16(t->c), b = vld1q_s16(t->c + 8);
    int16x8_t ta = vaddq_s16(i0, vandq_s16(vreinterpretq_s16_u16(vcgtq_s16(i0, sv)), add));
    int16x8_t tb = vaddq_s16(i1, vandq_s16(vreinterpretq_s16_u16(vcgtq_s16(i1, sv)), add));
    a = vaddq_s16(a, vshlq_s16(vsubq_s16(ta, a), sh));
    b = vaddq_s16(b, vshlq_s16(vsubq_s16(tb, b), sh));
    int count = t->c[15];
    vst1q_s16(t->c, a);
    vst1q_s16(t->c + 8, b);
    t->c[15] = (int16_t)(count < NIB_RATE_MAX - NIB_RATE_MIN ? count + 1 : count);
#else
    for (int k = 0; k < 15; k++) {
        int target = k + 1 <= (int)s ? k + 1 : (int)RANS_M - 15 + k;
        t->c[k] = (int16_t)(t->c[k] + floor_shift(target - t->c[k], r));
    }
    if (t->c[15] < NIB_RATE_MAX - NIB_RATE_MIN) t->c[15]++;
#endif
}

#if defined(DS_SSE2)
#  if defined(_MSC_VER) && !defined(__clang__)
#    include <intrin.h>
static inline unsigned lowest_bit(uint32_t v) { unsigned long i; _BitScanForward(&i, v); return (unsigned)i; }
#  else
static inline unsigned lowest_bit(uint32_t v) { return (unsigned)__builtin_ctz(v); }
#  endif
#endif

/* The symbol whose range holds slot: the number of k < 15 with c[k] <= slot. */
static inline unsigned nib_find(const Nibbles *t, uint32_t slot) {
#if defined(DS_SSE2)
    const __m128i v = _mm_set1_epi16((short)slot);
    __m128i a = _mm_loadu_si128((const __m128i *)(const void *)t->c);
    __m128i b = _mm_loadu_si128((const __m128i *)(const void *)(t->c + 8));
    uint32_t above = (uint32_t)_mm_movemask_epi8(_mm_cmpgt_epi16(a, v)) |
                     (uint32_t)_mm_movemask_epi8(_mm_cmpgt_epi16(b, v)) << 16;
    above &= 0x3FFFFFFFu;               /* not the update count */
    /* the tables increase: the entries above slot are a suffix, two mask bits each */
    return above ? lowest_bit(above) / 2 : 15u;
#elif defined(DS_NEON)
    int16x8_t v = vdupq_n_s16((int16_t)slot);
    uint16x8_t a = vcleq_s16(vld1q_s16(t->c), v), b = vcleq_s16(vld1q_s16(t->c + 8), v);
    b = vsetq_lane_u16(0, b, 7);
    return (unsigned)(vaddvq_u16(vshrq_n_u16(a, 15)) + vaddvq_u16(vshrq_n_u16(b, 15)));
#else
    unsigned s = 0;
    while (s < 15 && (uint32_t)(uint16_t)t->c[s] <= slot) s++;
    return s;
#endif
}

static inline void nib_contexts(const uint8_t *b, size_t i, size_t stride, uint32_t *ch, uint32_t *cl) {
    uint32_t p1 = i >= 1 ? b[i - 1] : 0;
    uint32_t ps = i >= stride ? b[i - stride] : 0;
    uint32_t ps1 = i >= stride + 1 ? b[i - stride - 1] : 0;
    *ch = (p1 >> 4) << 6 | (ps >> 4) << 2 | ps1 >> 6;
    *cl = ((p1 >> 4) << 4 | ps >> 4) << 4;      /* + the high half */
}

typedef struct { uint16_t start, freq; } RansSymbol;

/* Codes `planes` planes of n bytes into out (at most cap bytes, even); returns the size, or 0 when
   it would not fit. The model runs forwards to find each symbol's frequency; rANS then codes the
   symbols backwards, so that the decoder reads them forwards. sym holds 2 n planes entries. */
static size_t rans_encode(const uint8_t *data, size_t n, unsigned planes, size_t stride, NibbleModel *m,
                          RansSymbol *sym, uint8_t *out, size_t cap) {
    size_t count = 0;
    for (unsigned p = 0; p < planes; p++) {
        const uint8_t *b = data + p * n;
        nibble_model_reset(m);
        for (size_t i = 0; i < n; i++) {
            uint32_t ch, cl;
            nib_contexts(b, i, stride, &ch, &cl);
            unsigned hi = b[i] >> 4, lo = b[i] & 15u;
            Nibbles *th = &m->hi[ch], *tl = &m->lo[cl | hi];
            sym[count++] = (RansSymbol){(uint16_t)nib_start(th, hi), (uint16_t)(nib_end(th, hi) - nib_start(th, hi))};
            nib_update(th, hi);
            sym[count++] = (RansSymbol){(uint16_t)nib_start(tl, lo), (uint16_t)(nib_end(tl, lo) - nib_start(tl, lo))};
            nib_update(tl, lo);
        }
    }
    cap &= ~(size_t)1;
    size_t at = cap;                            /* 16-bit words, written from the end */
    uint32_t x[2] = {RANS_L, RANS_L};
    for (size_t k = count; k-- > 0;) {
        uint32_t *xs = &x[k & 1u], f = sym[k].freq;
        if (*xs >= ((RANS_L >> RANS_SCALE) << 16) * f) {
            if (at < 2) return 0;
            at -= 2;
            put16(out + at, (uint16_t)*xs);
            *xs >>= 16;
        }
        *xs = ((*xs / f) << RANS_SCALE) + (*xs % f) + sym[k].start;
    }
    if (at < 8) return 0;
    at -= 8;
    put32(out + at, x[0]);
    put32(out + at + 4, x[1]);
    size_t len = cap - at;
    memmove(out, out + at, len);
    return len;
}

static bool rans_decode(const uint8_t *in, size_t size, uint8_t *data, size_t n, unsigned planes, size_t stride,
                        NibbleModel *m) {
    if (size < 8 || size % 2 != 0) return false;
    uint32_t x0 = get32(in), x1 = get32(in + 4);
    size_t pos = 8;
    for (unsigned p = 0; p < planes; p++) {
        uint8_t *b = data + p * n;
        nibble_model_reset(m);
        for (size_t i = 0; i < n; i++) {
            uint32_t ch, cl;
            nib_contexts(b, i, stride, &ch, &cl);
            Nibbles *th = &m->hi[ch];
            uint32_t slot = x0 & (RANS_M - 1);
            unsigned hi = nib_find(th, slot);
            uint32_t start = nib_start(th, hi);
            x0 = (nib_end(th, hi) - start) * (x0 >> RANS_SCALE) + slot - start;
            if (x0 < RANS_L) {
                x0 = x0 << 16 | (pos + 2 <= size ? get16(in + pos) : 0u);
                pos += 2;
            }
            nib_update(th, hi);
            Nibbles *tl = &m->lo[cl | hi];
            slot = x1 & (RANS_M - 1);
            unsigned lo = nib_find(tl, slot);
            start = nib_start(tl, lo);
            x1 = (nib_end(tl, lo) - start) * (x1 >> RANS_SCALE) + slot - start;
            if (x1 < RANS_L) {
                x1 = x1 << 16 | (pos + 2 <= size ? get16(in + pos) : 0u);
                pos += 2;
            }
            nib_update(tl, lo);
            b[i] = (uint8_t)(hi << 4 | lo);
        }
    }
    /* the encoder started from RANS_L and wrote exactly what the decoder reads */
    return pos == size && x0 == RANS_L && x1 == RANS_L;
}

/* Picks the stride whose context predicts a sample of the stream best (static order-1 entropy
   estimate over the first plane bytes of the first chunk). */
static size_t choose_stride(const uint8_t *data, size_t n, unsigned planes, uint32_t width) {
    size_t cand[64], nc = 0;
    for (size_t s = 1; s <= 4; s++) cand[nc++] = s;
    for (uint32_t d = 5; d <= width && d <= 4096 && nc + 3 <= 60; d++)
        if (width % d == 0) {
            cand[nc++] = d - 1;
            cand[nc++] = d;
            cand[nc++] = d + 1;
        }
    if (width > 4096 && width <= (1u << 20)) cand[nc++] = width;

    size_t sample = n < (1u << 18) ? n : (1u << 18);
    uint32_t *counts = (uint32_t *)malloc(CTX_TREE * 256 * sizeof(uint32_t));
    uint32_t *totals = (uint32_t *)malloc(CTX_TREE * sizeof(uint32_t));
    size_t best = 1;
    double best_cost = INFINITY;
    for (size_t c = 0; counts && totals && c < nc; c++) {
        size_t s = cand[c];
        if (s >= sample) continue;
        double cost = 0;
        for (unsigned p = 0; p < planes; p++) {
            const uint8_t *b = data + p * n;
            memset(counts, 0, CTX_TREE * 256 * sizeof(uint32_t));
            memset(totals, 0, CTX_TREE * sizeof(uint32_t));
            for (size_t i = 0; i < sample; i++) {
                uint32_t cz, ct;
                contexts(b, i, s, &cz, &ct);
                counts[ct * 256 + b[i]]++;
                totals[ct]++;
            }
            for (size_t k = 0; k < (size_t)CTX_TREE * 256; k++)
                if (counts[k]) cost -= counts[k] * log2((double)counts[k] / totals[k / 256]);
        }
        if (cost < best_cost) { best_cost = cost; best = s; }
    }
    free(counts);
    free(totals);
    return best;
}

/* ---------------------------------------------------------------- value encodings */

/* bytes per stored value; `size` is the stream's size as loaded (classes, for CLASS) */
static unsigned value_width(DatasetEncoding e, uint32_t size) {
    switch (e) {
        case DATASET_ENCODING_FLOAT32: return 4;
        case DATASET_ENCODING_FP16:
        case DATASET_ENCODING_BFLOAT16: return 2;
        case DATASET_ENCODING_CLASS: return size <= 256 ? 1 : 2;
        default: return 1;
    }
}

/* values per sample in a stream */
static uint32_t stream_width(DatasetEncoding e, uint32_t size) {
    return e == DATASET_ENCODING_CLASS ? 1 : size;
}

static float unit_lut[256];
static bool unit_ready;

static void unit_init(void) {
    if (unit_ready) return;
    for (int q = 0; q < 256; q++) unit_lut[q] = (float)((double)q / 255.0);   /* exactly q / 255.0f */
    unit_ready = true;
}

static int unit_quantize(float v) {
    if (!(v >= 0.0f && v <= 1.0f)) return v > 1.0f ? 255 : 0;
    int q = (int)floor((double)v * 255.0 + 0.5);
    return q > 255 ? 255 : q;
}

static bool all_unit(const float *v, size_t n) {
    for (size_t i = 0; i < n; i++) {
        if (!(v[i] >= 0.0f && v[i] <= 1.0f)) return false;
        if (unit_lut[unit_quantize(v[i])] != v[i]) return false;
    }
    return true;
}

static bool all_half(const float *v, size_t n) {
    for (size_t i = 0; i < n; i++)
        if (spingalett_fp16_to_float(spingalett_float_to_fp16(v[i])) != v[i]) return false;
    return true;
}

static bool all_one_hot(const float *v, uint32_t rows, uint32_t width) {
    if (width < 2) return false;
    for (uint32_t r = 0; r < rows; r++) {
        uint32_t ones = 0;
        for (uint32_t k = 0; k < width; k++) {
            float x = v[(size_t)r * width + k];
            if (x == 1.0f) ones++;
            else if (x != 0.0f) return false;
        }
        if (ones != 1) return false;
    }
    return true;
}

static DatasetEncoding choose_encoding(const float *v, uint32_t rows, uint32_t width, bool targets) {
    size_t n = (size_t)rows * width;
    if (targets && all_one_hot(v, rows, width) && width <= 65536) return DATASET_ENCODING_CLASS;
    if (all_unit(v, n)) return DATASET_ENCODING_U8_UNIT;
    if (all_half(v, n)) return DATASET_ENCODING_FP16;
    return DATASET_ENCODING_FLOAT32;
}

/* U8_AFFINE parameters: per feature min and step. */
static void affine_params(const float *v, uint32_t rows, uint32_t width, float *params) {
    for (uint32_t f = 0; f < width; f++) {
        float lo = INFINITY, hi = -INFINITY;
        for (uint32_t r = 0; r < rows; r++) {
            float x = v[(size_t)r * width + f];
            if (x < lo) lo = x;
            if (x > hi) hi = x;
        }
        if (!(lo <= hi)) lo = hi = 0.0f;        /* all NaN */
        params[2 * f] = lo;
        params[2 * f + 1] = (hi - lo) / 255.0f;
    }
}

/* Encodes rows [0, rows) of src (row width `size`) into byte planes of n = rows * stream width values. */
static void encode_values(DatasetEncoding e, const float *src, uint32_t rows, uint32_t size,
                          const float *params, uint8_t *planes) {
    uint32_t w = stream_width(e, size);
    size_t n = (size_t)rows * w;
    if (e == DATASET_ENCODING_FP16) {
        uint16_t h[256];
        for (size_t i0 = 0; i0 < n; i0 += 256) {
            size_t len = n - i0 < 256 ? n - i0 : 256;
            spingalett_fp16_encode(src + i0, len, h);
            for (size_t i = 0; i < len; i++) {
                planes[i0 + i] = (uint8_t)h[i];
                planes[n + i0 + i] = (uint8_t)(h[i] >> 8);
            }
        }
        return;
    }
    for (size_t i = 0; i < n; i++) {
        switch (e) {
            case DATASET_ENCODING_FLOAT32: {
                uint32_t u = float_bits(src[i]);
                for (int k = 0; k < 4; k++) planes[k * n + i] = (uint8_t)(u >> (8 * k));
                break;
            }
            case DATASET_ENCODING_FP16:
            case DATASET_ENCODING_BFLOAT16: {
                uint16_t h = e == DATASET_ENCODING_FP16 ? spingalett_float_to_fp16(src[i]) : spingalett_float_to_bf16(src[i]);
                planes[i] = (uint8_t)h;
                planes[n + i] = (uint8_t)(h >> 8);
                break;
            }
            case DATASET_ENCODING_U8_UNIT:
                planes[i] = (uint8_t)unit_quantize(src[i]);
                break;
            case DATASET_ENCODING_U8_AFFINE: {
                uint32_t f = (uint32_t)(i % size);
                float lo = params[2 * f], step = params[2 * f + 1];
                double q = step > 0.0f ? floor(((double)src[i] - lo) / step + 0.5) : 0.0;
                planes[i] = (uint8_t)(q < 0 ? 0 : q > 255 ? 255 : q);
                break;
            }
            case DATASET_ENCODING_CLASS: {
                const float *row = src + i * size;
                uint32_t best = 0;
                for (uint32_t k = 1; k < size; k++) if (row[k] > row[best]) best = k;
                planes[i] = (uint8_t)best;
                if (size > 256) planes[n + i] = (uint8_t)(best >> 8);
                break;
            }
            default:
                break;
        }
    }
}

/* A stream's values in their stored form, sample by sample: for value v of a sample, its vw
   bytes, least significant first. */
typedef struct {
    DatasetEncoding enc;
    uint32_t size;                      /* floats per sample once decoded */
    uint32_t width;                     /* stored values per sample */
    unsigned vw;                        /* bytes per stored value */
    const float *params;                /* U8_AFFINE */
} ValueForm;

static ValueForm value_form(DatasetEncoding e, uint32_t size, const float *params) {
    return (ValueForm){e, size, stream_width(e, size), value_width(e, size), params};
}

/* The floats of one sample from its stored bytes (v: form.width values of form.vw bytes each,
   the byte k of value i at v[i * step + k * plane], i.e. planes or interleaved bytes). Returns
   false for a class index out of range. */
static bool decode_sample(const ValueForm *f, const uint8_t *v, size_t step, size_t plane, float *dst) {
    switch (f->enc) {
        case DATASET_ENCODING_FLOAT32:
            for (uint32_t i = 0; i < f->width; i++) {
                const uint8_t *b = v + i * step;
                dst[i] = bits_float((uint32_t)b[0] | (uint32_t)b[plane] << 8 | (uint32_t)b[2 * plane] << 16 |
                                    (uint32_t)b[3 * plane] << 24);
            }
            return true;
        case DATASET_ENCODING_FP16:
            for (uint32_t i = 0; i < f->width; i++)
                dst[i] = spingalett_fp16_to_float((uint16_t)(v[i * step] | v[i * step + plane] << 8));
            return true;
        case DATASET_ENCODING_BFLOAT16:
            for (uint32_t i = 0; i < f->width; i++)
                dst[i] = spingalett_bf16_to_float((uint16_t)(v[i * step] | v[i * step + plane] << 8));
            return true;
        case DATASET_ENCODING_U8_UNIT:
            for (uint32_t i = 0; i < f->width; i++) dst[i] = unit_lut[v[i * step]];
            return true;
        case DATASET_ENCODING_U8_AFFINE:
            for (uint32_t i = 0; i < f->width; i++)
                dst[i] = f->params[2 * i] + (float)v[i * step] * f->params[2 * i + 1];
            return true;
        case DATASET_ENCODING_CLASS: {
            uint32_t c = v[0] | (f->vw == 2 ? (uint32_t)v[plane] << 8 : 0u);
            if (c >= f->size) return false;
            memset(dst, 0, (size_t)f->size * sizeof(float));
            dst[c] = 1.0f;
            return true;
        }
        default:
            return false;
    }
}

/* Decodes the byte planes of `rows` samples into floats. */
static bool decode_values(const ValueForm *f, const uint8_t *planes, uint32_t rows, float *dst) {
    size_t n = (size_t)rows * f->width;
    for (uint32_t r = 0; r < rows; r++)
        if (!decode_sample(f, planes + (size_t)r * f->width, 1, n, dst + (size_t)r * f->size)) return false;
    return true;
}

/* ---------------------------------------------------------------- file layout */

typedef struct {
    DatasetEncoding enc;
    uint32_t size;                      /* floats per sample once loaded (classes, for CLASS) */
    uint32_t stride;                    /* coder context stride */
    float *params;                      /* U8_AFFINE: min, step per value, or NULL */
    char *name;                         /* sets of targets: a name, or NULL */
    char **class_names;                 /* sets of targets: size names, or NULL */
    uint8_t method;                     /* writer: the coder chosen for the stream */
} StreamInfo;

typedef struct {
    uint32_t version, count, chunk_samples, chunk_count;
    uint32_t streams;                   /* inputs, then the sets of targets */
    uint64_t file_size;
    uint32_t height, width, channels;
    StreamInfo *st;
    uint64_t *offset;                   /* per chunk */
    uint32_t *bytes;                    /* [chunk x stream] */
    uint32_t *crc;                      /* per chunk */
    size_t meta;                        /* bytes before the first chunk */
} Layout;

static void layout_free(Layout *L) {
    for (uint32_t s = 0; L->st && s < L->streams; s++) {
        free(L->st[s].params);
        free(L->st[s].name);
        free(L->st[s].class_names);
    }
    free(L->st);
    free(L->offset);
    free(L->bytes);
    free(L->crc);
    memset(L, 0, sizeof *L);
}

static ValueForm stream_form(const Layout *L, uint32_t s) {
    return value_form(L->st[s].enc, L->st[s].size, L->st[s].params);
}

static uint32_t chunk_rows(const Layout *L, uint32_t c) {
    uint64_t start = (uint64_t)c * L->chunk_samples;
    uint64_t left = L->count - start;
    return left < L->chunk_samples ? (uint32_t)left : L->chunk_samples;
}

/* raw (decoded) bytes of stream s for `rows` samples */
static uint64_t stream_raw(const Layout *L, uint32_t s, uint32_t rows) {
    const StreamInfo *st = &L->st[s];
    return (uint64_t)rows * stream_width(st->enc, st->size) * value_width(st->enc, st->size);
}

static size_t params_bytes(const Layout *L) {
    size_t b = 0;
    for (uint32_t s = 0; s < L->streams; s++)
        if (L->st[s].enc == DATASET_ENCODING_U8_AFFINE) b += (size_t)L->st[s].size * 8;
    return b;
}

static size_t index_entry(const Layout *L) { return 12 + 4 * (size_t)L->streams; }

static uint64_t chunk_bytes(const Layout *L, uint32_t c) {
    uint64_t b = 0;
    for (uint32_t s = 0; s < L->streams; s++) b += L->bytes[(size_t)c * L->streams + s];
    return b;
}

static bool fail(int code, const char *message) {
    set_error(code, message);
    spingalett_log(LOG_ERROR, "%s", message);
    return false;
}

static bool valid_encoding(DatasetEncoding e, uint32_t size, bool targets) {
    return e > DATASET_ENCODING_AUTO && e < DATASET_ENCODING_COUNT && size > 0 &&
           (e != DATASET_ENCODING_CLASS || (targets && size <= 65536));
}

static char *copy_string(const uint8_t *p, size_t len) {
    char *s = (char *)malloc(len + 1);
    if (!s) return NULL;
    memcpy(s, p, len);
    s[len] = '\0';
    return s;
}

/* Reads the metadata records (shape, sets of targets, names). */
static bool parse_metadata(const uint8_t *p, size_t size, Layout *L) {
    bool *described = (bool *)calloc(L->streams, sizeof(bool));
    if (!described) return fail(SPINGALETT_ERR_ALLOC, "out of memory reading a .slettd file");
    bool ok = true;
    const uint8_t *end = p + size;
    while (ok && p < end) {
        if (end - p < 6) { ok = false; break; }
        uint32_t tag = get16(p), len = get32(p + 2);
        p += 6;
        if ((size_t)(end - p) < len) { ok = false; break; }
        const uint8_t *q = p;
        p += len;
        if (tag == META_SHAPE) {
            ok = len >= 12;
            if (ok) { L->height = get32(q); L->width = get32(q + 4); L->channels = get32(q + 8); }
        } else if (tag == META_TARGETS) {
            uint32_t k = len >= 16 ? get32(q) : 0;
            ok = k >= 1 && k + 1 < L->streams && !described[k + 1];
            if (ok) {
                StreamInfo *st = &L->st[k + 1];
                st->size = get32(q + 4);
                st->enc = (DatasetEncoding)q[8];
                st->stride = get32(q + 12);
                described[k + 1] = true;
            }
        } else if (tag == META_NAME) {
            uint32_t k = len >= 4 ? get32(q) : UINT32_MAX;
            ok = k + 1 < L->streams && !L->st[k + 1].name && memchr(q + 4, 0, len - 4) == NULL;
            if (ok) ok = (L->st[k + 1].name = copy_string(q + 4, len - 4)) != NULL;
        } else if (tag == META_CLASSES) {
            uint32_t k = len >= 8 ? get32(q) : UINT32_MAX, n = len >= 8 ? get32(q + 4) : 0;
            ok = k + 1 < L->streams && !L->st[k + 1].class_names && n > 0 && n <= 65536;
            /* the strings, then one block of pointers and characters */
            const uint8_t *r = q + 8, *qend = q + len;
            size_t chars = 0;
            for (uint32_t i = 0; ok && i < n; i++) {
                ok = qend - r >= 2;
                uint32_t l = ok ? get16(r) : 0;
                ok = ok && (size_t)(qend - r - 2) >= l && memchr(r + 2, 0, l) == NULL;
                r += 2 + l;
                chars += l + 1;
            }
            char **names = ok ? (char **)malloc(((size_t)n + 1) * sizeof(char *) + chars) : NULL;
            if (ok && !names) ok = false;
            if (ok) {
                char *c = (char *)(names + n + 1);
                r = q + 8;
                for (uint32_t i = 0; i < n; i++) {
                    uint32_t l = get16(r);
                    memcpy(c, r + 2, l);
                    c[l] = '\0';
                    names[i] = c;
                    c += l + 1;
                    r += 2 + l;
                }
                names[n] = NULL;                    /* n is checked against the set's size below */
                L->st[k + 1].class_names = names;
            }
        }
        /* other tags: skipped */
    }
    for (uint32_t s = 2; ok && s < L->streams; s++) ok = described[s];
    for (uint32_t s = 1; ok && s < L->streams; s++) {
        StreamInfo *st = &L->st[s];
        if (st->class_names) {
            uint32_t n = 0;
            while (st->class_names[n]) n++;
            ok = n == st->size;
        }
    }
    free(described);
    return ok || fail(SPINGALETT_ERR_INVALID, "corrupt .slettd metadata");
}

/* Parses header, parameters, metadata and chunk index from the first `size` bytes of a file
   (`size` may be just the header, in which case *need tells how many bytes the metadata takes). */
static bool parse_header(const uint8_t *h, size_t size, Layout *L, size_t *need) {
    memset(L, 0, sizeof *L);
    if (size < SLETTD_HEADER_SIZE || memcmp(h, "SLETTD", 6) != 0)
        return fail(SPINGALETT_ERR_INVALID, "not a .slettd data set file");
    uint32_t version = get16(h + 6);
    if (version < 1 || version > SLETTD_VERSION_MAX)
        return fail(SPINGALETT_ERR_FORMAT_VERSION, "unsupported .slettd format version");
    if (crc32_update(0, h, 60) != get32(h + 60))
        return fail(SPINGALETT_ERR_INVALID, "corrupt .slettd header (checksum mismatch)");
    uint32_t sets = get32(h + 52), mbytes = get32(h + 56), pbytes = get32(h + 48);
    if (version == 1) {
        if (sets != 0 || mbytes != 0) return fail(SPINGALETT_ERR_INVALID, "corrupt .slettd header (inconsistent fields)");
        sets = 1;
    }
    if (sets < 1 || sets > SLETTD_MAX_SETS)
        return fail(SPINGALETT_ERR_INVALID, "corrupt .slettd header (inconsistent fields)");
    L->version = version;
    L->streams = 1 + sets;
    L->count = get32(h + 8);
    L->chunk_samples = get32(h + 24);
    L->chunk_count = get32(h + 28);
    L->file_size = get64(h + 40);
    L->st = (StreamInfo *)calloc(L->streams, sizeof(StreamInfo));
    if (!L->st) return fail(SPINGALETT_ERR_ALLOC, "out of memory reading a .slettd file");
    L->st[0] = (StreamInfo){.enc = (DatasetEncoding)h[20], .size = get32(h + 12), .stride = get32(h + 32)};
    L->st[1] = (StreamInfo){.enc = (DatasetEncoding)h[21], .size = get32(h + 16), .stride = get32(h + 36)};
    bool ok = L->chunk_samples > 0 &&
              L->chunk_count == (uint32_t)(((uint64_t)L->count + L->chunk_samples - 1) / L->chunk_samples);
    size_t meta = SLETTD_HEADER_SIZE + (size_t)pbytes + mbytes + (size_t)L->chunk_count * index_entry(L) + 4;
    *need = meta;
    L->meta = meta;
    if (!ok) {
        layout_free(L);
        return fail(SPINGALETT_ERR_INVALID, "corrupt .slettd header (inconsistent fields)");
    }
    if (size < meta) return true;       /* caller reads the rest and parses again */

    if (crc32_update(0, h + SLETTD_HEADER_SIZE, meta - SLETTD_HEADER_SIZE - 4) != get32(h + meta - 4)) {
        layout_free(L);
        return fail(SPINGALETT_ERR_INVALID, "corrupt .slettd index (checksum mismatch)");
    }
    if (!parse_metadata(h + SLETTD_HEADER_SIZE + pbytes, mbytes, L)) {
        layout_free(L);
        return false;
    }
    for (uint32_t s = 0; ok && s < L->streams; s++) {
        const StreamInfo *st = &L->st[s];
        ok = valid_encoding(st->enc, st->size, s > 0) && st->stride > 0 &&
             stream_raw(L, s, L->chunk_samples) < ((uint64_t)1 << 32);
    }
    ok = ok && pbytes == params_bytes(L);
    if (!ok) {
        layout_free(L);
        return fail(SPINGALETT_ERR_INVALID, "corrupt .slettd header (inconsistent fields)");
    }
    const uint8_t *p = h + SLETTD_HEADER_SIZE;
    for (uint32_t s = 0; s < L->streams; s++) {
        StreamInfo *st = &L->st[s];
        if (st->enc != DATASET_ENCODING_U8_AFFINE) continue;
        st->params = (float *)malloc((size_t)st->size * 2 * sizeof(float));
        if (!st->params) {
            layout_free(L);
            return fail(SPINGALETT_ERR_ALLOC, "out of memory reading a .slettd file");
        }
        for (uint32_t i = 0; i < 2 * st->size; i++, p += 4) st->params[i] = bits_float(get32(p));
    }
    uint32_t chunks = L->chunk_count;
    L->offset = (uint64_t *)malloc(((size_t)chunks + 1) * sizeof(uint64_t));
    L->bytes = (uint32_t *)malloc(((size_t)chunks * L->streams + 1) * sizeof(uint32_t));
    L->crc = (uint32_t *)malloc(((size_t)chunks + 1) * sizeof(uint32_t));
    if (!L->offset || !L->bytes || !L->crc) {
        layout_free(L);
        return fail(SPINGALETT_ERR_ALLOC, "out of memory reading a .slettd file");
    }
    p = h + SLETTD_HEADER_SIZE + pbytes + mbytes;
    for (uint32_t c = 0; c < chunks; c++, p += index_entry(L)) {
        L->offset[c] = get64(p);
        uint32_t rows = chunk_rows(L, c);
        bool plausible = true;
        for (uint32_t s = 0; s < L->streams; s++) {
            uint32_t b = get32(p + 8 + 4 * (size_t)s);
            L->bytes[(size_t)c * L->streams + s] = b;
            /* the coders cannot expand a byte beyond this, so a chunk that claims more is corrupt;
               this also stops tiny crafted files from requesting huge buffers */
            plausible = plausible && b > 0 && stream_raw(L, s, rows) <= (uint64_t)b * SLETTD_MAX_EXPANSION;
        }
        L->crc[c] = get32(p + 8 + 4 * (size_t)L->streams);
        if (L->offset[c] < meta || L->offset[c] + chunk_bytes(L, c) > L->file_size || !plausible) {
            layout_free(L);
            return fail(SPINGALETT_ERR_INVALID, "corrupt .slettd index (chunk outside the file)");
        }
    }
    return true;
}

/* ---------------------------------------------------------------- chunk decoding */

typedef struct {
    uint8_t *planes;                    /* decoded stream bytes */
    size_t planes_cap;
    ByteModel *model;
    NibbleModel *nibbles;
} Scratch;

static void scratch_free(Scratch *s) {
    free(s->planes);
    free(s->model);
    free(s->nibbles);
    memset(s, 0, sizeof *s);
}

static bool scratch_reserve(Scratch *s, size_t bytes) {
    if (bytes <= s->planes_cap) return true;
    uint8_t *p = (uint8_t *)realloc(s->planes, bytes);
    if (!p) return false;
    s->planes = p;
    s->planes_cap = bytes;
    return true;
}

/* Decodes stream s of chunk c (data: the stream's bytes) into the planes of the scratch. Sets the
   error without logging it (decoding also runs on background threads). */
static bool decode_planes(const Layout *L, uint32_t c, uint32_t s, const uint8_t *data, Scratch *sc) {
    uint32_t rows = chunk_rows(L, c), size = L->bytes[(size_t)c * L->streams + s];
    size_t raw = (size_t)stream_raw(L, s, rows);
    unsigned planes = value_width(L->st[s].enc, L->st[s].size);
    if (!scratch_reserve(sc, raw + 1)) {
        set_error(SPINGALETT_ERR_ALLOC, "out of memory decoding a .slettd chunk");
        return false;
    }
    bool ok;
    if (data[0] == STREAM_STORED) {
        ok = size == raw + 1;
        if (ok) memcpy(sc->planes, data + 1, raw);
    } else if (data[0] == STREAM_CODED) {
        if (!sc->model && !(sc->model = (ByteModel *)malloc(sizeof(ByteModel)))) {
            set_error(SPINGALETT_ERR_ALLOC, "out of memory decoding a .slettd chunk");
            return false;
        }
        ok = decode_stream(data + 1, size - 1, sc->planes, raw / planes, planes, L->st[s].stride, sc->model);
    } else if (data[0] == STREAM_RANS && L->version >= 2) {
        if (!sc->nibbles && !(sc->nibbles = (NibbleModel *)malloc(sizeof(NibbleModel)))) {
            set_error(SPINGALETT_ERR_ALLOC, "out of memory decoding a .slettd chunk");
            return false;
        }
        ok = rans_decode(data + 1, size - 1, sc->planes, raw / planes, planes, L->st[s].stride, sc->nibbles);
    } else {
        ok = false;
    }
    if (!ok) set_error(SPINGALETT_ERR_INVALID, "corrupt .slettd chunk (stream does not decode)");
    return ok;
}

static bool check_chunk(const Layout *L, uint32_t c, const uint8_t *data) {
    if (crc32_update(0, data, (size_t)chunk_bytes(L, c)) == L->crc[c]) return true;
    set_error(SPINGALETT_ERR_INVALID, "corrupt .slettd chunk (checksum mismatch)");
    return false;
}

static const uint8_t *stream_data(const Layout *L, uint32_t c, uint32_t s, const uint8_t *chunk) {
    for (uint32_t k = 0; k < s; k++) chunk += L->bytes[(size_t)c * L->streams + k];
    return chunk;
}

/* Decodes chunk c (its bytes at data) into rows of inputs and of the targets of stream `set`. */
static bool decode_chunk(const Layout *L, uint32_t c, uint32_t set, const uint8_t *data, float *inputs,
                         float *targets, Scratch *sc) {
    if (!check_chunk(L, c, data)) return false;
    uint32_t rows = chunk_rows(L, c), streams[2] = {0, set};
    float *out[2] = {inputs, targets};
    for (int k = 0; k < 2; k++) {
        ValueForm f = stream_form(L, streams[k]);
        if (!decode_planes(L, c, streams[k], stream_data(L, c, streams[k], data), sc)) return false;
        if (!decode_values(&f, sc->planes, rows, out[k])) {
            set_error(SPINGALETT_ERR_INVALID, "corrupt .slettd chunk (class index out of range)");
            return false;
        }
    }
    return true;
}

/* ---------------------------------------------------------------- writing */

static char *with_extension(const char *path) {
    const char *dot = strrchr(path, '.');
    const char *slash = strrchr(path, '/'), *backslash = strrchr(path, '\\');
    const char *sep = slash > backslash ? slash : backslash;
    bool has_ext = dot && (!sep || dot > sep);
    size_t len = strlen(path);
    char *out = (char *)malloc(len + sizeof SPINGALETT_DATASET_EXTENSION);
    if (!out) return NULL;
    memcpy(out, path, len + 1);
    if (!has_ext) strcat(out, SPINGALETT_DATASET_EXTENSION);
    return out;
}

/* The bytes of one encoded chunk, built before it is written. */
typedef struct {
    uint8_t *data;
    size_t size, cap;
    uint32_t *bytes;                    /* per stream */
    uint32_t crc;
    bool ok;
} ChunkOut;

typedef struct {
    uint8_t *planes, *coded;
    size_t cap;
    ByteModel *model;
    NibbleModel *nibbles;
    RansSymbol *symbols;
    size_t symbols_cap;
} Encoder;

static void encoder_free(Encoder *e) {
    free(e->planes);
    free(e->coded);
    free(e->model);
    free(e->nibbles);
    free(e->symbols);
    memset(e, 0, sizeof *e);
}

static bool encoder_reserve(Encoder *e, size_t raw) {
    if (!e->model && !(e->model = (ByteModel *)malloc(sizeof(ByteModel)))) return false;
    if (!e->nibbles && !(e->nibbles = (NibbleModel *)malloc(sizeof(NibbleModel)))) return false;
    if (raw + 16 > e->cap) {
        uint8_t *p = (uint8_t *)realloc(e->planes, raw + 16), *c = p ? (uint8_t *)realloc(e->coded, raw + 16) : NULL;
        if (p) e->planes = p;
        if (!c) return false;
        e->coded = c;
        e->cap = raw + 16;
    }
    if (2 * raw > e->symbols_cap) {
        RansSymbol *s = (RansSymbol *)realloc(e->symbols, (2 * raw + 1) * sizeof(RansSymbol));
        if (!s) return false;
        e->symbols = s;
        e->symbols_cap = 2 * raw;
    }
    return true;
}

/* Codes the planes of a stream with `method`; returns the coded size or 0 (not smaller than raw). */
static size_t code_planes(Encoder *e, uint8_t method, size_t raw, unsigned planes, size_t stride) {
    if (method == STREAM_RANS)
        return rans_encode(e->planes, raw / planes, planes, stride, e->nibbles, e->symbols, e->coded, raw);
    return encode_stream(e->planes, raw / planes, planes, stride, e->model, e->coded, raw);
}

static void encode_chunk(const Layout *L, const float *const *src, uint32_t c, bool compress, Encoder *e,
                         ChunkOut *out) {
    uint32_t rows = chunk_rows(L, c);
    uint64_t start = (uint64_t)c * L->chunk_samples;
    out->size = 0;
    out->crc = 0;
    out->ok = true;
    for (uint32_t s = 0; s < L->streams && out->ok; s++) {
        const StreamInfo *st = &L->st[s];
        size_t raw = (size_t)stream_raw(L, s, rows);
        out->ok = encoder_reserve(e, raw);
        if (out->ok && out->size + raw + 1 > out->cap) {
            size_t cap = (out->size + raw + 1) * 2;
            uint8_t *d = (uint8_t *)realloc(out->data, cap);
            out->ok = d != NULL;
            if (d) { out->data = d; out->cap = cap; }
        }
        if (!out->ok) break;
        encode_values(st->enc, src[s] + start * st->size, rows, st->size, st->params, e->planes);
        unsigned planes = value_width(st->enc, st->size);
        size_t len = compress ? code_planes(e, st->method, raw, planes, st->stride) : 0;
        uint8_t method = len > 0 && len < raw ? st->method : STREAM_STORED;
        const uint8_t *body = method == STREAM_STORED ? e->planes : e->coded;
        if (method == STREAM_STORED) len = raw;
        out->data[out->size] = method;
        memcpy(out->data + out->size + 1, body, len);
        out->crc = crc32_update(out->crc, out->data + out->size, len + 1);
        out->size += len + 1;
        out->bytes[s] = (uint32_t)(len + 1);
    }
}

static void put_record(uint8_t **p, uint32_t tag, uint32_t len) {
    put16(*p, (uint16_t)tag);
    put32(*p + 2, len);
    *p += 6;
}

/* The metadata block (shape, further sets of targets, names); with out NULL, its size. */
static size_t write_metadata(const Layout *L, uint8_t *out) {
    uint8_t buf[16], *p = out ? out : buf;
    size_t total = 0;
#define RECORD(tag, len) do { if (out) put_record(&p, (tag), (uint32_t)(len)); total += 6 + (size_t)(len); } while (0)
    if (L->height || L->width || L->channels) {
        RECORD(META_SHAPE, 12);
        if (out) { put32(p, L->height); put32(p + 4, L->width); put32(p + 8, L->channels); p += 12; }
    }
    for (uint32_t s = 1; s < L->streams; s++) {
        const StreamInfo *st = &L->st[s];
        uint32_t k = s - 1;
        if (k > 0) {
            RECORD(META_TARGETS, 16);
            if (out) {
                put32(p, k); put32(p + 4, st->size);
                p[8] = (uint8_t)st->enc; p[9] = p[10] = p[11] = 0;
                put32(p + 12, st->stride);
                p += 16;
            }
        }
        if (st->name) {
            size_t len = strlen(st->name);
            RECORD(META_NAME, 4 + len);
            if (out) { put32(p, k); memcpy(p + 4, st->name, len); p += 4 + len; }
        }
        if (st->class_names) {
            size_t len = 8;
            for (uint32_t i = 0; i < st->size; i++) len += 2 + strlen(st->class_names[i]);
            RECORD(META_CLASSES, len);
            if (out) {
                put32(p, k); put32(p + 4, st->size);
                p += 8;
                for (uint32_t i = 0; i < st->size; i++) {
                    size_t l = strlen(st->class_names[i]);
                    put16(p, (uint16_t)l);
                    memcpy(p + 2, st->class_names[i], l);
                    p += 2 + l;
                }
            }
        }
    }
#undef RECORD
    return total;
}

static bool names_valid(const char *const *names, uint32_t count) {
    for (uint32_t i = 0; names && i < count; i++)
        if (!names[i] || strlen(names[i]) > 65535) return false;
    return true;
}

bool spingalett_save_dataset(const SpingalettDataset *d, const char *path, const DatasetSaveOptions *options) {
    unit_init();
    DatasetSaveOptions opt = options ? *options : (DatasetSaveOptions){0};
    if (!d || !path || !d->inputs || !d->targets || d->count == 0 || d->input_size == 0 || d->target_size == 0)
        return fail(SPINGALETT_ERR_INVALID, "spingalett_save_dataset: empty or NULL data set or path");
    uint32_t sets = 1 + (opt.extra_targets ? opt.extra_target_count : 0);
    bool ok = sets <= SLETTD_MAX_SETS &&
              (unsigned)opt.input_encoding < DATASET_ENCODING_COUNT && opt.input_encoding != DATASET_ENCODING_CLASS &&
              (unsigned)opt.target_encoding < DATASET_ENCODING_COUNT &&
              !(opt.target_encoding == DATASET_ENCODING_CLASS && d->target_size > 65536) &&
              names_valid((const char *const *)d->class_names, d->target_size) &&
              (!opt.target_name || strlen(opt.target_name) < 65536);
    for (uint32_t k = 1; ok && k < sets; k++) {
        const SpingalettTargetSet *t = &opt.extra_targets[k - 1];
        ok = t->targets && t->size > 0 && (unsigned)t->encoding < DATASET_ENCODING_COUNT &&
             !(t->encoding == DATASET_ENCODING_CLASS && t->size > 65536) && names_valid(t->class_names, t->size) &&
             (!t->name || strlen(t->name) < 65536);
    }
    if (!ok) return fail(SPINGALETT_ERR_INVALID, "spingalett_save_dataset: invalid encoding, set of targets or name");

    Layout L = {.count = d->count, .streams = 1 + sets, .height = d->height, .width = d->width, .channels = d->channels};
    L.st = (StreamInfo *)calloc(L.streams, sizeof(StreamInfo));
    const float **src = (const float **)calloc(L.streams, sizeof(float *));
    if (!L.st || !src) {
        free(L.st); free(src);
        return fail(SPINGALETT_ERR_ALLOC, "spingalett_save_dataset: out of memory");
    }
    /* the layout borrows names and parameters it does not own until layout_free; names are copied */
    src[0] = d->inputs;
    src[1] = d->targets;
    L.st[0] = (StreamInfo){.size = d->input_size,
                           .enc = opt.input_encoding ? opt.input_encoding : choose_encoding(d->inputs, d->count, d->input_size, false)};
    L.st[1] = (StreamInfo){.size = d->target_size,
                           .enc = opt.target_encoding ? opt.target_encoding : choose_encoding(d->targets, d->count, d->target_size, true)};
    if (opt.target_name) L.st[1].name = copy_string((const uint8_t *)opt.target_name, strlen(opt.target_name));
    if (d->class_names) L.st[1].class_names = spingalett_copy_names((const char *const *)d->class_names, d->target_size);
    for (uint32_t k = 1; k < sets; k++) {
        const SpingalettTargetSet *t = &opt.extra_targets[k - 1];
        StreamInfo *st = &L.st[k + 1];
        src[k + 1] = t->targets;
        st->size = t->size;
        st->enc = t->encoding ? t->encoding : choose_encoding(t->targets, d->count, t->size, true);
        if (t->name) st->name = copy_string((const uint8_t *)t->name, strlen(t->name));
        if (t->class_names) st->class_names = spingalett_copy_names(t->class_names, t->size);
        ok = ok && (!t->name || st->name) && (!t->class_names || st->class_names);
    }
    ok = ok && (!opt.target_name || L.st[1].name) && (!d->class_names || L.st[1].class_names);

    uint64_t sample_bytes = 0;
    for (uint32_t s = 0; s < L.streams; s++) sample_bytes += stream_raw(&L, s, 1);
    uint64_t per_chunk = SLETTD_CHUNK_TARGET / sample_bytes;
    L.chunk_samples = per_chunk == 0 ? 1 : per_chunk > d->count ? d->count : (uint32_t)per_chunk;
    L.chunk_count = (uint32_t)(((uint64_t)d->count + L.chunk_samples - 1) / L.chunk_samples);
    for (uint32_t s = 0; ok && s < L.streams; s++)
        if (stream_raw(&L, s, L.chunk_samples) >= ((uint64_t)1 << 32)) {
            free(src);
            layout_free(&L);
            return fail(SPINGALETT_ERR_INVALID, "spingalett_save_dataset: a single sample exceeds 4 GB");
        }
    for (uint32_t s = 0; s < L.streams && ok; s++)
        if (L.st[s].enc == DATASET_ENCODING_U8_AFFINE) {
            L.st[s].params = (float *)malloc((size_t)L.st[s].size * 2 * sizeof(float));
            ok = L.st[s].params != NULL;
            if (ok) affine_params(src[s], d->count, L.st[s].size, L.st[s].params);
        }

    /* strides and coders from the first chunk: rANS where it is at most 1% larger (it decodes
       dense data two to three times as fast), the binary coder otherwise */
    Encoder enc0 = {0};
    bool any_rans = false;
    for (uint32_t s = 0; s < L.streams && ok; s++) {
        StreamInfo *st = &L.st[s];
        st->stride = 1;
        st->method = STREAM_CODED;
        if (opt.no_compression) continue;
        uint32_t rows = chunk_rows(&L, 0);
        size_t raw = (size_t)stream_raw(&L, s, rows);
        ok = encoder_reserve(&enc0, raw);
        if (!ok) break;
        encode_values(st->enc, src[s], rows, st->size, st->params, enc0.planes);
        unsigned w = value_width(st->enc, st->size);
        st->stride = (uint32_t)choose_stride(enc0.planes, raw / w, w, stream_width(st->enc, st->size));
        size_t binary = code_planes(&enc0, STREAM_CODED, raw, w, st->stride);
        size_t rans = code_planes(&enc0, STREAM_RANS, raw, w, st->stride);
        if (binary == 0) binary = raw;
        if (rans > 0 && rans <= binary + binary / 100) {
            st->method = STREAM_RANS;
            any_rans = true;
        }
    }
    encoder_free(&enc0);

    L.version = (L.streams > 2 || any_rans || L.height || L.width || L.channels || L.st[1].name || L.st[1].class_names)
                ? 2 : 1;
    size_t pbytes = params_bytes(&L), mbytes = L.version >= 2 ? write_metadata(&L, NULL) : 0;
    size_t meta = SLETTD_HEADER_SIZE + pbytes + mbytes + (size_t)L.chunk_count * index_entry(&L) + 4;
    uint8_t *head = ok ? (uint8_t *)calloc(meta, 1) : NULL;
    char *filename = ok ? with_extension(path) : NULL;
    if (!head || !filename) {
        free(head); free(filename); free(src); layout_free(&L);
        return fail(SPINGALETT_ERR_ALLOC, "spingalett_save_dataset: out of memory");
    }
    FILE *f = fopen(filename, "wb");
    if (!f) {
        spingalett_log(LOG_ERROR, "Cannot open %s for writing", filename);
        free(head); free(filename); free(src); layout_free(&L);
        return fail(SPINGALETT_ERR_FILE_IO, "spingalett_save_dataset: cannot open file for writing");
    }
    ok = fwrite(head, 1, meta, f) == meta;      /* placeholder, rewritten at the end */

    /* chunks in groups that the threads encode side by side, written in order */
    int threads = 1;
#if defined(_OPENMP)
    if (resolve_compute_mode() == COMPUTE_OPENMP) threads = omp_get_max_threads();
#endif
    uint32_t group = (uint32_t)(threads > 1 ? 2 * threads : 1);
    ChunkOut *outs = (ChunkOut *)calloc(group, sizeof(ChunkOut));
    uint32_t *outbytes = (uint32_t *)calloc((size_t)group * L.streams, sizeof(uint32_t));
    Encoder *encoders = (Encoder *)calloc((size_t)threads, sizeof(Encoder));
    ok = ok && outs && outbytes && encoders;
    for (uint32_t i = 0; outs && outbytes && i < group; i++) outs[i].bytes = outbytes + (size_t)i * L.streams;
    uint8_t *index = head + SLETTD_HEADER_SIZE + pbytes + mbytes;
    uint64_t pos = meta;
    bool encode_ok = true;
    for (uint32_t c0 = 0; ok && c0 < L.chunk_count; c0 += group) {
        int64_t n = (int64_t)(L.chunk_count - c0 < group ? L.chunk_count - c0 : group);
#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic) num_threads(threads) if(threads > 1 && n > 1)
#endif
        for (int64_t i = 0; i < n; i++) {
            int t = 0;
#if defined(_OPENMP)
            t = omp_get_thread_num();
#endif
            encode_chunk(&L, src, c0 + (uint32_t)i, !opt.no_compression, &encoders[t], &outs[i]);
        }
        for (int64_t i = 0; ok && i < n; i++) {
            uint32_t c = c0 + (uint32_t)i;
            encode_ok = encode_ok && outs[i].ok;
            ok = encode_ok && fwrite(outs[i].data, 1, outs[i].size, f) == outs[i].size;
            uint8_t *e = index + (size_t)c * index_entry(&L);
            put64(e, pos);
            for (uint32_t s = 0; s < L.streams; s++) put32(e + 8 + 4 * (size_t)s, outs[i].bytes[s]);
            put32(e + 8 + 4 * (size_t)L.streams, outs[i].crc);
            pos += outs[i].size;
        }
    }
    for (uint32_t i = 0; outs && i < group; i++) free(outs[i].data);
    for (int t = 0; encoders && t < threads; t++) encoder_free(&encoders[t]);
    free(outs);
    free(outbytes);
    free(encoders);

    /* header and metadata, now that sizes and offsets are known */
    memcpy(head, "SLETTD", 6);
    put16(head + 6, (uint16_t)L.version);
    put32(head + 8, L.count);
    put32(head + 12, L.st[0].size);
    put32(head + 16, L.st[1].size);
    head[20] = (uint8_t)L.st[0].enc;
    head[21] = (uint8_t)L.st[1].enc;
    head[22] = opt.no_compression ? 0 : 1;
    put32(head + 24, L.chunk_samples);
    put32(head + 28, L.chunk_count);
    put32(head + 32, L.st[0].stride);
    put32(head + 36, L.st[1].stride);
    put64(head + 40, pos);
    put32(head + 48, (uint32_t)pbytes);
    if (L.version >= 2) {
        put32(head + 52, L.streams - 1);
        put32(head + 56, (uint32_t)mbytes);
    }
    put32(head + 60, crc32_update(0, head, 60));
    uint8_t *p = head + SLETTD_HEADER_SIZE;
    for (uint32_t s = 0; s < L.streams; s++)
        if (L.st[s].params)
            for (uint32_t i = 0; i < 2 * L.st[s].size; i++, p += 4) put32(p, float_bits(L.st[s].params[i]));
    if (mbytes) write_metadata(&L, p);
    put32(head + meta - 4, crc32_update(0, head + SLETTD_HEADER_SIZE, meta - SLETTD_HEADER_SIZE - 4));
    ok = ok && fseek(f, 0, SEEK_SET) == 0 && fwrite(head, 1, meta, f) == meta;
    ok = (fclose(f) == 0) && ok;
    if (ok)
        spingalett_log(LOG_INFO, "Data set saved to %s: %u samples, %llu bytes", filename, L.count, (unsigned long long)pos);
    else if (!encode_ok)
        fail(SPINGALETT_ERR_ALLOC, "spingalett_save_dataset: out of memory");
    else
        fail(SPINGALETT_ERR_FILE_IO, "spingalett_save_dataset: write error (disk full?)");

    free(head); free(filename); free(src); layout_free(&L);
    return ok;
}

/* ---------------------------------------------------------------- loading */

/* The shape and the class names of stream `set` into the data set. */
static bool describe_dataset(const Layout *L, uint32_t set, SpingalettDataset *d) {
    d->height = L->height;
    d->width = L->width;
    d->channels = L->channels;
    if (!L->st[set].class_names) return true;
    d->class_names = spingalett_copy_names((const char *const *)L->st[set].class_names, L->st[set].size);
    return d->class_names != NULL || fail(SPINGALETT_ERR_ALLOC, "out of memory loading a data set");
}

static bool alloc_dataset(SpingalettDataset *d, const Layout *L, uint32_t set) {
    d->count = L->count;
    d->input_size = L->st[0].size;
    d->target_size = L->st[set].size;
    d->inputs = (float *)malloc(((size_t)L->count * d->input_size + 1) * sizeof(float));
    d->targets = (float *)malloc(((size_t)L->count * d->target_size + 1) * sizeof(float));
    if (d->inputs && d->targets && describe_dataset(L, set, d)) return true;
    spingalett_dataset_free(d);
    return fail(SPINGALETT_ERR_ALLOC, "out of memory loading a data set");
}

/* Runs fn on every chunk, in parallel when OpenMP threads are on; returns false with the first
   failing chunk's error. */
typedef bool (*ChunkFn)(const Layout *L, uint32_t c, Scratch *sc, void *ctx);

static bool for_each_chunk(const Layout *L, ChunkFn fn, void *ctx) {
    bool ok = true;
    int error_code = SPINGALETT_OK;               /* errors are thread-local: carry a worker's out */
    char error_message[SPINGALETT_ERRMSG_MAX] = "";
    int64_t chunks = (int64_t)L->chunk_count;
#if defined(_OPENMP)
#pragma omp parallel if(chunks > 1 && resolve_compute_mode() == COMPUTE_OPENMP)
#endif
    {
        Scratch sc = {0};
#if defined(_OPENMP)
#pragma omp for schedule(dynamic)
#endif
        for (int64_t c = 0; c < chunks; c++) {
            bool chunk_ok;
#if defined(_OPENMP)
#pragma omp atomic read
#endif
            chunk_ok = ok;
            if (!chunk_ok) continue;
            if (!fn(L, (uint32_t)c, &sc, ctx)) {
#if defined(_OPENMP)
#pragma omp critical(spingalett_dataset_error)
#endif
                if (error_code == SPINGALETT_OK) {
                    error_code = spingalett_last_error_code();
                    snprintf(error_message, sizeof error_message, "%s", spingalett_last_error_message());
                }
#if defined(_OPENMP)
#pragma omp atomic write
#endif
                ok = false;
            }
        }
        scratch_free(&sc);
    }
    if (!ok) fail(error_code, error_message);
    return ok;
}

typedef struct {
    const uint8_t *bytes;
    uint32_t set;
    SpingalettDataset *d;
} LoadContext;

static bool load_chunk(const Layout *L, uint32_t c, Scratch *sc, void *ctx) {
    const LoadContext *x = (const LoadContext *)ctx;
    uint64_t row = (uint64_t)c * L->chunk_samples;
    return decode_chunk(L, c, x->set, x->bytes + L->offset[c], x->d->inputs + row * x->d->input_size,
                        x->d->targets + row * x->d->target_size, sc);
}

/* Parses the layout of a whole file image in memory. */
static bool parse_image(const uint8_t *bytes, size_t size, Layout *L) {
    size_t need = 0;
    if (!parse_header(bytes, size, L, &need)) return false;
    if (size < need || size < L->file_size) {
        layout_free(L);
        return fail(SPINGALETT_ERR_FILE_IO, "truncated .slettd data set");
    }
    return L->offset != NULL || parse_header(bytes, size, L, &need);
}

static bool check_set(Layout *L, uint32_t target_set) {
    if (target_set + 1 < L->streams) return true;
    layout_free(L);
    return fail(SPINGALETT_ERR_INVALID, "the .slettd file has no such set of targets");
}

bool spingalett_load_dataset_from_memory_targets(const void *data, size_t size, uint32_t target_set,
                                                 SpingalettDataset *dataset) {
    unit_init();
    if (!dataset) return fail(SPINGALETT_ERR_INVALID, "spingalett_load_dataset: dataset is NULL");
    memset(dataset, 0, sizeof *dataset);
    if (!data) return fail(SPINGALETT_ERR_INVALID, "spingalett_load_dataset: data is NULL");
    Layout L;
    if (!parse_image((const uint8_t *)data, size, &L) || !check_set(&L, target_set)) return false;
    if (!alloc_dataset(dataset, &L, 1 + target_set)) { layout_free(&L); return false; }
    LoadContext ctx = {(const uint8_t *)data, 1 + target_set, dataset};
    bool ok = for_each_chunk(&L, load_chunk, &ctx);
    layout_free(&L);
    if (!ok) spingalett_dataset_free(dataset);
    return ok;
}

bool spingalett_load_dataset_from_memory(const void *data, size_t size, SpingalettDataset *dataset) {
    return spingalett_load_dataset_from_memory_targets(data, size, 0, dataset);
}

/* The whole file in memory. */
static uint8_t *read_whole(const char *path, size_t *size) {
    FILE *f = path ? fopen(path, "rb") : NULL;
    if (!f) {
        spingalett_log(LOG_ERROR, "Cannot open %s", path ? path : "(null)");
        fail(SPINGALETT_ERR_FILE_IO, "cannot open data set file");
        return NULL;
    }
    uint8_t *buf = NULL;
    long len = -1;
    if (fseek(f, 0, SEEK_END) == 0) len = ftell(f);
    bool ok = len >= 0 && fseek(f, 0, SEEK_SET) == 0 && (buf = (uint8_t *)malloc((size_t)len + 1)) != NULL &&
              fread(buf, 1, (size_t)len, f) == (size_t)len;
    fclose(f);
    if (!ok) {
        fail(buf || len < 0 ? SPINGALETT_ERR_FILE_IO : SPINGALETT_ERR_ALLOC, "cannot read data set file");
        free(buf);
        return NULL;
    }
    *size = (size_t)len;
    return buf;
}

bool spingalett_load_dataset_targets(const char *path, uint32_t target_set, SpingalettDataset *dataset) {
    if (!dataset) return fail(SPINGALETT_ERR_INVALID, "spingalett_load_dataset: dataset is NULL");
    memset(dataset, 0, sizeof *dataset);
    size_t size = 0;
    uint8_t *buf = read_whole(path, &size);
    if (!buf) return false;
    bool ok = spingalett_load_dataset_from_memory_targets(buf, size, target_set, dataset);
    free(buf);
    return ok;
}

bool spingalett_load_dataset(const char *path, SpingalettDataset *dataset) {
    return spingalett_load_dataset_targets(path, 0, dataset);
}

/* ---------------------------------------------------------------- readers */

/* Values kept in their stored form, sample after sample (form.width values of form.vw bytes each,
   least significant byte first). */
typedef struct {
    ValueForm form;
    uint8_t *data;
    float *params;                      /* owned copy of form.params */
} Compact;

static void compact_free(Compact *c) {
    free(c->data);
    free(c->params);
    memset(c, 0, sizeof *c);
}

static size_t compact_sample(const Compact *c) { return (size_t)c->form.width * c->form.vw; }

#define READER_SLOTS 3                  /* with a decoding thread: the chunk being read, two decoded ahead */
#define READER_BURST_MAX 64             /* chunks decoded side by side on the caller's threads */

typedef enum { SLOT_FREE, SLOT_REQUESTED, SLOT_DECODING, SLOT_READY, SLOT_FAILED } SlotState;

/* A chunk on its way to the caller: its compressed bytes, then the byte planes of the inputs and
   of the served targets, which reads turn into floats a sample at a time. */
typedef struct {
    SlotState state;
    uint32_t chunk;
    uint8_t *raw;
    size_t raw_cap;
    Scratch values[2];                  /* inputs, targets */
    int error_code;
    char error_message[SPINGALETT_ERRMSG_MAX];
} Slot;

struct SpingalettDatasetReader {
    Layout layout;
    uint32_t set;                       /* stream of the served targets */
    uint32_t count, input_size, target_size;
    bool shuffle, failed, pass_done;
    uint64_t seed;                      /* orders are drawn from it and a pass or chunk number */
    SpingalettDatasetInfo info;
    uint32_t *order;                    /* the current pass: samples (in memory) or chunks */
    uint64_t pass;

    /* in memory: every sample, a pass in the order of `order` */
    bool in_memory;
    Compact values[2];                  /* inputs, targets */
    uint32_t next;

    /* streaming: request q is chunk order[q % chunks] of pass q / chunks */
    FILE *file;
    ValueForm forms[2];                 /* inputs, targets */
    uint64_t requested, taken;          /* requests issued; chunks handed to the caller */
    Slot *slots, **jobs;
    uint32_t slot_count;
    Slot *current;                      /* the chunk being read */
    uint32_t *rows_order, rows, row;
    SpgThread *thread;                  /* NULL: requested chunks decode together on the caller's threads */
    SpgSignal *signal;
    bool stop;
};

/* Shuffles v with a generator seeded by the reader's seed and an index (2 x pass for the order of
   a pass, 2 x request + 1 for the rows of a chunk): the samples come in the same order however
   far ahead chunks are decoded, on any number of threads. */
static uint64_t mix64(uint64_t z) {
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

static void shuffle_u32(const SpingalettDatasetReader *r, uint64_t index, uint32_t *v, uint32_t n) {
    uint64_t state = mix64(r->seed ^ mix64(index + 0x9E3779B97F4A7C15ull));
    for (uint32_t i = n; i > 1; i--) {
        uint32_t j = (uint32_t)(mix64(state += 0x9E3779B97F4A7C15ull) % i), t = v[i - 1];
        v[i - 1] = v[j];
        v[j] = t;
    }
}

/* Fills `order` with pass p's order of n items. */
static void pass_order(SpingalettDatasetReader *r, uint64_t p, uint32_t n) {
    for (uint32_t i = 0; i < n; i++) r->order[i] = i;
    if (r->shuffle) shuffle_u32(r, 2 * p, r->order, n);
}

/* Checks the class indices of n values in byte planes (one plane, or two for 256 classes and more). */
static bool check_classes(const ValueForm *f, const uint8_t *planes, size_t n) {
    if (f->enc != DATASET_ENCODING_CLASS) return true;
    for (size_t i = 0; i < n; i++) {
        uint32_t cls = planes[i] | (f->vw == 2 ? (uint32_t)planes[n + i] << 8 : 0u);
        if (cls >= f->size) {
            set_error(SPINGALETT_ERR_INVALID, "corrupt .slettd chunk (class index out of range)");
            return false;
        }
    }
    return true;
}

static void stop_thread(SpingalettDatasetReader *r) {
    if (!r->thread) return;
    spg_lock(r->signal);
    r->stop = true;
    spg_wake(r->signal);
    spg_unlock(r->signal);
    spg_thread_join(r->thread);
    r->thread = NULL;
}

void spingalett_dataset_close(SpingalettDatasetReader *r) {
    if (!r) return;
    stop_thread(r);
    spg_signal_free(r->signal);
    if (r->file) fclose(r->file);
    layout_free(&r->layout);
    for (int k = 0; k < 2; k++) compact_free(&r->values[k]);
    free(r->order);
    for (uint32_t k = 0; r->slots && k < r->slot_count; k++) {
        free(r->slots[k].raw);
        scratch_free(&r->slots[k].values[0]);
        scratch_free(&r->slots[k].values[1]);
    }
    free(r->slots);
    free(r->jobs);
    free(r->rows_order);
    free(r);
}

/* Reads the compressed bytes of a slot's chunk (on the decoding thread, or on the caller's). */
static bool slot_read(SpingalettDatasetReader *r, Slot *s) {
    const Layout *L = &r->layout;
    size_t size = (size_t)chunk_bytes(L, s->chunk);
    if (size > s->raw_cap) {
        uint8_t *p = (uint8_t *)realloc(s->raw, size);
        if (!p) {
            set_error(SPINGALETT_ERR_ALLOC, "out of memory reading a .slettd chunk");
            return false;
        }
        s->raw = p;
        s->raw_cap = size;
    }
    if (fseek(r->file, (long)L->offset[s->chunk], SEEK_SET) != 0 || fread(s->raw, 1, size, r->file) != size) {
        set_error(SPINGALETT_ERR_FILE_IO, "truncated .slettd data set");
        return false;
    }
    return true;
}

/* Decodes a slot's chunk into the byte planes of its inputs and targets. Touches nothing but the
   slot, so that several slots decode at once. */
static bool slot_decode(const SpingalettDatasetReader *r, Slot *s) {
    const Layout *L = &r->layout;
    uint32_t c = s->chunk, streams[2] = {0, r->set};
    if (!check_chunk(L, c, s->raw)) return false;
    for (int k = 0; k < 2; k++) {
        Scratch *v = &s->values[k];
        if (!decode_planes(L, c, streams[k], stream_data(L, c, streams[k], s->raw), v) ||
            !check_classes(&r->forms[k], v->planes, (size_t)chunk_rows(L, c) * r->forms[k].width))
            return false;
    }
    return true;
}

/* Marks a slot ready, or failed with the error of this thread (set without logging). */
static void slot_done(Slot *s, bool ok) {
    s->state = ok ? SLOT_READY : SLOT_FAILED;
    if (!ok) {
        s->error_code = spingalett_last_error_code();
        snprintf(s->error_message, sizeof s->error_message, "%s", spingalett_last_error_message());
    }
}

static void reader_thread(void *arg) {
    SpingalettDatasetReader *r = (SpingalettDatasetReader *)arg;
    spg_lock(r->signal);
    for (;;) {
        Slot *job = NULL;
        /* the oldest request first */
        for (uint64_t q = r->taken; q < r->requested && !job; q++)
            if (r->slots[q % r->slot_count].state == SLOT_REQUESTED) job = &r->slots[q % r->slot_count];
        if (r->stop) break;
        if (!job) {
            spg_wait(r->signal);
            continue;
        }
        job->state = SLOT_DECODING;
        spg_unlock(r->signal);
        bool ok = slot_read(r, job) && slot_decode(r, job);
        spg_lock(r->signal);
        slot_done(job, ok);
        spg_wake(r->signal);
    }
    spg_unlock(r->signal);
}

/* Decodes every requested chunk on the caller's thread: read one after the other, then decoded
   side by side by the OpenMP threads, which have nothing else to do while the caller waits. */
static void reader_burst(SpingalettDatasetReader *r) {
    int64_t n = 0;
    for (uint64_t q = r->taken; q < r->requested; q++) {
        Slot *s = &r->slots[q % r->slot_count];
        if (s->state != SLOT_REQUESTED) continue;
        if (slot_read(r, s)) r->jobs[n++] = s;
        else slot_done(s, false);
    }
#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic) if(n > 1 && resolve_compute_mode() == COMPUTE_OPENMP)
#endif
    for (int64_t i = 0; i < n; i++) slot_done(r->jobs[i], slot_decode(r, r->jobs[i]));
}

/* Issues requests into the free slots (one stays with the chunk being read); a pass's order is
   drawn when its first chunk is requested. Called with the lock held when there is a thread. */
static void reader_request(SpingalettDatasetReader *r) {
    const uint32_t chunks = r->layout.chunk_count;
    while (r->requested - r->taken < r->slot_count - (r->current ? 1u : 0u)) {
        uint64_t q = r->requested;
        if (q % chunks == 0) pass_order(r, q / chunks, chunks);
        Slot *s = &r->slots[q % r->slot_count];
        s->chunk = r->order[q % chunks];
        s->state = SLOT_REQUESTED;
        r->requested++;
    }
    if (r->thread) spg_wake(r->signal);
}

/* Hands the slot of the chunk that has been read back to the queue. */
static void reader_release(SpingalettDatasetReader *r) {
    if (!r->current) return;
    if (r->thread) spg_lock(r->signal);
    r->current->state = SLOT_FREE;
    r->current = NULL;
    if (r->thread) spg_unlock(r->signal);
}

/* Takes the next chunk (waiting for the decoding thread, or decoding the requested chunks here). */
static bool reader_take(SpingalettDatasetReader *r) {
    reader_release(r);
    Slot *s = &r->slots[r->taken % r->slot_count];
    if (r->thread) spg_lock(r->signal);
    reader_request(r);
    if (r->thread) {
        while (s->state == SLOT_REQUESTED || s->state == SLOT_DECODING) spg_wait(r->signal);
    } else if (s->state == SLOT_REQUESTED) {
        reader_burst(r);
    }
    bool ok = s->state == SLOT_READY;
    uint64_t q = r->taken++;
    r->current = s;
    reader_request(r);
    if (r->thread) spg_unlock(r->signal);
    if (!ok) return fail(s->error_code, s->error_message);
    r->rows = chunk_rows(&r->layout, s->chunk);
    r->row = 0;
    for (uint32_t i = 0; i < r->rows; i++) r->rows_order[i] = i;
    if (r->shuffle) shuffle_u32(r, 2 * q + 1, r->rows_order, r->rows);
    return true;
}

static bool reader_alloc(SpingalettDatasetReader *r) {
    const Layout *L = &r->layout;
    return (r->order = (uint32_t *)malloc(((size_t)L->chunk_count + 1) * sizeof(uint32_t))) != NULL &&
           (r->rows_order = (uint32_t *)malloc(((size_t)L->chunk_samples + 1) * sizeof(uint32_t))) != NULL &&
           (r->slots = (Slot *)calloc(r->slot_count, sizeof(Slot))) != NULL &&
           (r->jobs = (Slot **)calloc(r->slot_count, sizeof(Slot *))) != NULL;
}

/* Decodes the planes of chunk c's input stream and served targets into the compact store. */
typedef struct {
    const uint8_t *bytes;
    SpingalettDatasetReader *r;
} CompactContext;

static bool compact_chunk(const Layout *L, uint32_t c, Scratch *sc, void *ctx) {
    const CompactContext *x = (const CompactContext *)ctx;
    const uint8_t *chunk = x->bytes + L->offset[c];
    if (!check_chunk(L, c, chunk)) return false;
    uint32_t rows = chunk_rows(L, c), streams[2] = {0, x->r->set};
    for (int k = 0; k < 2; k++) {
        const Compact *v = &x->r->values[k];
        size_t n = (size_t)rows * v->form.width, sample = compact_sample(v);
        if (!decode_planes(L, c, streams[k], stream_data(L, c, streams[k], chunk), sc) ||
            !check_classes(&v->form, sc->planes, n))            /* checked here, once */
            return false;
        uint8_t *dst = v->data + (size_t)c * L->chunk_samples * sample;
        if (v->form.vw == 1) {
            memcpy(dst, sc->planes, n);
        } else {
            for (size_t i = 0; i < n; i++)
                for (unsigned b = 0; b < v->form.vw; b++) dst[i * v->form.vw + b] = sc->planes[b * n + i];
        }
    }
    return true;
}

static bool compact_init(Compact *c, ValueForm form, uint32_t count) {
    c->form = form;
    if (form.params) {
        if (!(c->params = (float *)malloc((size_t)form.size * 2 * sizeof(float)))) return false;
        memcpy(c->params, form.params, (size_t)form.size * 2 * sizeof(float));
        c->form.params = c->params;
    }
    c->data = (uint8_t *)malloc((size_t)count * compact_sample(c) + 1);
    return c->data != NULL;
}

static void reader_info(SpingalettDatasetReader *r) {
    const Layout *L = &r->layout;
    SpingalettDatasetInfo *i = &r->info;
    i->count = r->count;
    i->input_size = r->input_size;
    i->target_size = r->target_size;
    i->input_encoding = L->st ? L->st[0].enc : r->values[0].form.enc;
    i->target_encoding = L->st ? L->st[r->set].enc : r->values[1].form.enc;
    i->chunk_count = L->chunk_count;
    i->file_size = L->file_size;
    i->format_version = L->version;
    i->height = L->height;
    i->width = L->width;
    i->channels = L->channels;
    i->target_set_count = L->st ? L->streams - 1 : 1;
    i->target_set = r->set - 1;
}

static SpingalettDatasetReader *reader_new(bool shuffle) {
    unit_init();
    SpingalettDatasetReader *r = (SpingalettDatasetReader *)calloc(1, sizeof *r);
    if (!r) {
        fail(SPINGALETT_ERR_ALLOC, "spingalett_dataset_open: out of memory");
        return NULL;
    }
    r->shuffle = shuffle;
    r->set = 1;
    r->seed = rng_next64();             /* one draw of the library's generator, on this thread */
    return r;
}

static void reader_start_memory_pass(SpingalettDatasetReader *r) {
    pass_order(r, r->pass++, r->count);
    r->next = 0;
}

SpingalettDatasetReader *spingalett_dataset_open_ex(const char *path, const DatasetReaderOptions *options) {
    DatasetReaderOptions o = options ? *options : (DatasetReaderOptions){0};
    SpingalettDatasetReader *r = reader_new(o.shuffle);
    if (!r) return NULL;
    Layout *L = &r->layout;
    bool ok;
    if (o.in_memory) {
        size_t size = 0;
        uint8_t *bytes = read_whole(path, &size);
        ok = bytes && parse_image(bytes, size, L) && check_set(L, o.target_set);
        if (ok) {
            r->set = 1 + o.target_set;
            r->count = L->count;
            r->input_size = L->st[0].size;
            r->target_size = L->st[r->set].size;
            ok = compact_init(&r->values[0], stream_form(L, 0), L->count) &&
                 compact_init(&r->values[1], stream_form(L, r->set), L->count) &&
                 (r->order = (uint32_t *)malloc(((size_t)L->count + 1) * sizeof(uint32_t))) != NULL;
            if (!ok) fail(SPINGALETT_ERR_ALLOC, "spingalett_dataset_open: out of memory");
            CompactContext ctx = {bytes, r};
            ok = ok && for_each_chunk(L, compact_chunk, &ctx);
        }
        free(bytes);
        if (ok) {
            r->in_memory = true;
            reader_start_memory_pass(r);
        }
    } else {
        r->file = path ? fopen(path, "rb") : NULL;
        ok = r->file != NULL;
        if (!ok) {
            spingalett_log(LOG_ERROR, "Cannot open %s", path ? path : "(null)");
            fail(SPINGALETT_ERR_FILE_IO, "cannot open data set file");
        }
        uint8_t head[SLETTD_HEADER_SIZE];
        size_t need = 0;
        if (ok && fread(head, 1, sizeof head, r->file) != sizeof head) ok = fail(SPINGALETT_ERR_FILE_IO, "truncated .slettd data set");
        ok = ok && parse_header(head, sizeof head, L, &need);
        uint8_t *meta = ok ? (uint8_t *)malloc(need) : NULL;
        if (ok && !meta) ok = fail(SPINGALETT_ERR_ALLOC, "spingalett_dataset_open: out of memory");
        if (ok) {
            memcpy(meta, head, sizeof head);
            if (fread(meta + sizeof head, 1, need - sizeof head, r->file) != need - sizeof head)
                ok = fail(SPINGALETT_ERR_FILE_IO, "truncated .slettd data set");
        }
        if (ok) layout_free(L);
        ok = ok && parse_header(meta, need, L, &need) && check_set(L, o.target_set);
        free(meta);
        if (ok) {
            r->set = 1 + o.target_set;
            r->count = L->count;
            r->input_size = L->st[0].size;
            r->target_size = L->st[r->set].size;
            r->forms[0] = stream_form(L, 0);
            r->forms[1] = stream_form(L, r->set);
            /* A thread decodes ahead when a processor is left for it. When the OpenMP threads use
               them all, a decoding thread would take turns with them, and every turn it takes
               stalls their barriers: the chunks then decode together on the caller's threads,
               one per thread, while training waits. */
            int threads = 1;
            bool spare = true;
#if defined(_OPENMP)
            if (resolve_compute_mode() == COMPUTE_OPENMP) threads = omp_get_max_threads();
            spare = threads < omp_get_num_procs();
#endif
            bool prefetch = !o.no_prefetch && spare && L->chunk_count > 1;
            r->slot_count = prefetch ? READER_SLOTS : threads < 1 ? 1u
                          : threads > READER_BURST_MAX ? READER_BURST_MAX : (uint32_t)threads;
            if (!reader_alloc(r)) ok = fail(SPINGALETT_ERR_ALLOC, "spingalett_dataset_open: out of memory");
            if (ok && prefetch) {
                r->signal = spg_signal_create();
                if (r->signal) r->thread = spg_thread_start(reader_thread, r);
                /* without a thread the caller decodes: the same samples in the same order */
            }
        }
    }
    if (!ok) {
        spingalett_dataset_close(r);
        return NULL;
    }
    reader_info(r);
    return r;
}

SpingalettDatasetReader *spingalett_dataset_open(const char *path, bool shuffle) {
    DatasetReaderOptions o = {.shuffle = shuffle};
    return spingalett_dataset_open_ex(path, &o);
}

SpingalettDatasetReader *spingalett_dataset_open_u8(const uint8_t *inputs, const float *targets, uint32_t count,
                                                    uint32_t input_size, uint32_t target_size, bool shuffle) {
    if (!inputs || !targets || count == 0 || input_size == 0 || target_size == 0) {
        fail(SPINGALETT_ERR_INVALID, "spingalett_dataset_open_u8: empty or NULL data");
        return NULL;
    }
    SpingalettDatasetReader *r = reader_new(shuffle);
    if (!r) return NULL;
    r->in_memory = true;
    r->count = count;
    r->input_size = input_size;
    r->target_size = target_size;
    bool ok = compact_init(&r->values[0], value_form(DATASET_ENCODING_U8_UNIT, input_size, NULL), count) &&
              compact_init(&r->values[1], value_form(DATASET_ENCODING_FLOAT32, target_size, NULL), count) &&
              (r->order = (uint32_t *)malloc(((size_t)count + 1) * sizeof(uint32_t))) != NULL;
    if (!ok) {
        fail(SPINGALETT_ERR_ALLOC, "spingalett_dataset_open_u8: out of memory");
        spingalett_dataset_close(r);
        return NULL;
    }
    memcpy(r->values[0].data, inputs, (size_t)count * input_size);
    uint8_t *t = r->values[1].data;
    for (size_t i = 0; i < (size_t)count * target_size; i++) put32(t + 4 * i, float_bits(targets[i]));
    reader_start_memory_pass(r);
    reader_info(r);
    return r;
}

SpingalettDatasetInfo spingalett_dataset_info(const SpingalettDatasetReader *r) {
    SpingalettDatasetInfo info = {0};
    return r ? r->info : info;
}

const char *spingalett_dataset_target_set_name(const SpingalettDatasetReader *r, uint32_t target_set) {
    if (!r || !r->layout.st || target_set + 1 >= r->layout.streams) return NULL;
    return r->layout.st[target_set + 1].name;
}

const char *spingalett_dataset_class_name(const SpingalettDatasetReader *r, uint32_t target_set, uint32_t index) {
    if (!r || !r->layout.st || target_set + 1 >= r->layout.streams) return NULL;
    const StreamInfo *st = &r->layout.st[target_set + 1];
    return st->class_names && index < st->size ? st->class_names[index] : NULL;
}

uint32_t spingalett_dataset_target_set_size(const SpingalettDatasetReader *r, uint32_t target_set) {
    if (!r) return 0;
    if (!r->layout.st) return target_set == 0 ? r->target_size : 0;
    return target_set + 1 < r->layout.streams ? r->layout.st[target_set + 1].size : 0;
}

/* Converts samples idx[0..n) of a store of values to floats (sample i's values at base + i x
   sample, value k's bytes at k x step + b x plane), side by side on the OpenMP threads for large
   batches. Class indices were checked when the values were decoded. */
static void convert_samples(const ValueForm *f, const uint8_t *base, size_t sample, size_t step, size_t plane,
                            const uint32_t *idx, uint32_t n, float *dst) {
    int64_t count = (int64_t)n;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if((uint64_t)n * f->width >= 65536 && resolve_compute_mode() == COMPUTE_OPENMP)
#endif
    for (int64_t k = 0; k < count; k++)
        (void)decode_sample(f, base + (size_t)idx[k] * sample, step, plane, dst + (size_t)k * f->size);
}

static uint32_t read_memory(SpingalettDatasetReader *r, float *inputs, float *targets, uint32_t max_samples) {
    uint32_t n = r->count - r->next < max_samples ? r->count - r->next : max_samples;
    for (int k = 0; k < 2; k++) {
        const Compact *v = &r->values[k];
        convert_samples(&v->form, v->data, compact_sample(v), v->form.vw, 1, r->order + r->next, n, k ? targets : inputs);
    }
    r->next += n;
    return n;
}

uint32_t spingalett_dataset_read(SpingalettDatasetReader *r, float *inputs, float *targets, uint32_t max_samples) {
    if (!r || !inputs || !targets) {
        set_error(SPINGALETT_ERR_INVALID, "spingalett_dataset_read: NULL argument");
        return 0;
    }
    if (r->failed) return 0;
    if (r->in_memory) {
        if (r->pass_done) {
            reader_start_memory_pass(r);
            r->pass_done = false;
        }
        uint32_t done = read_memory(r, inputs, targets, max_samples);
        if (done == 0) r->pass_done = true;
        return done;
    }
    const uint32_t chunks = r->layout.chunk_count;
    if (chunks == 0) return 0;
    if (r->pass_done) r->pass_done = false;     /* the next call starts the next pass */
    uint32_t done = 0;
    while (done < max_samples) {
        if (!r->current || r->row == r->rows) {
            /* the end of a pass: report it before taking the next pass's first chunk */
            if (r->current && r->taken % chunks == 0) {
                if (done > 0) break;
                r->pass_done = true;
                reader_release(r);
                return 0;
            }
            if (!reader_take(r)) {
                r->failed = true;
                return 0;
            }
        }
        uint32_t n = r->rows - r->row < max_samples - done ? r->rows - r->row : max_samples - done;
        for (int k = 0; k < 2; k++) {
            const ValueForm *f = &r->forms[k];
            float *dst = k ? targets + (size_t)done * r->target_size : inputs + (size_t)done * r->input_size;
            convert_samples(f, r->current->values[k].planes, f->width, 1, (size_t)r->rows * f->width,
                            r->rows_order + r->row, n, dst);
        }
        r->row += n;
        done += n;
    }
    return done;
}

uint32_t spingalett_dataset_generator(float *inputs, float *targets, uint32_t requested, void *reader) {
    return spingalett_dataset_read((SpingalettDatasetReader *)reader, inputs, targets, requested);
}
