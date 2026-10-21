/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/*
 * Attention of sessions over the keys they keep, in vectors, with the bits of the engine's spingalett_engine_attend().
 * The keys lie transposed in blocks of eight positions, so that eight scores are summed side by side, each over
 * the head's values in their order as the engine sums one; the online softmax then goes key after key as the
 * engine's does, its exponentials computed eight at a time by spingalett_expf()'s algorithm (in double, no fused
 * multiply-adds) and the weighted values added in vectors over the head (each value's sum in the order of the keys).
 * Builds that are not tuned for the build machine (SPINGALETT_BLOCK_DISPATCH: x86-64) compile it once more for AVX2
 * (Kernels/Spingalett.Attention.AVX2.c) and run that where the processor has it: the same bits.
 */

#include "Spingalett.Attention.h"
#include <float.h>
#include <string.h>

#if !defined(SPINGALETT_PORTABLE_KERNELS) && defined(__AVX2__)
#  include <immintrin.h>
#  define ATTEND_AVX2 1
#endif

#if defined(ATTEND_AVX2)
/* spingalett_expf() of eight floats; the arguments of its special cases (|x| >= 88, NaN) take it one by one. */
static inline __m256 expf8(__m256 x) {
    const __m256i bits = _mm256_castps_si256(x);
    const __m256i abstop = _mm256_and_si256(_mm256_srli_epi32(bits, 20), _mm256_set1_epi32(0x7FF));
    if (_mm256_movemask_epi8(_mm256_cmpgt_epi32(abstop, _mm256_set1_epi32(0x42A)))) {
        float v[8];
        _mm256_storeu_ps(v, x);
        for (int i = 0; i < 8; i++) v[i] = spingalett_expf(v[i]);
        return _mm256_loadu_ps(v);
    }
    const __m256d scale = _mm256_set1_pd(0x1.71547652b82fep+5), shift = _mm256_set1_pd(0x1.8p52);
    const __m256d c0 = _mm256_set1_pd(0x1.c6af84b912394p-5 / 32 / 32 / 32), c1 = _mm256_set1_pd(0x1.ebfce50fac4f3p-3 / 32 / 32);
    const __m256d c2 = _mm256_set1_pd(0x1.62e42ff0c52d6p-1 / 32), one = _mm256_set1_pd(1.0);
    const long long *table = (const long long *)(const void *)spingalett_exp2f_table();
    __m128 half[2];
    for (int h = 0; h < 2; h++) {
        const __m256d z = _mm256_mul_pd(scale, _mm256_cvtps_pd(h ? _mm256_extractf128_ps(x, 1) : _mm256_castps256_ps128(x)));
        __m256d kd = _mm256_add_pd(z, shift);
        const __m256i ki = _mm256_castpd_si256(kd);
        kd = _mm256_sub_pd(kd, shift);
        const __m256d r = _mm256_sub_pd(z, kd);
        const __m256i t = _mm256_add_epi64(_mm256_i64gather_epi64(table, _mm256_and_si256(ki, _mm256_set1_epi64x(31)), 8),
                                           _mm256_slli_epi64(ki, 47));
        const __m256d p = _mm256_add_pd(_mm256_mul_pd(c0, r), c1), r2 = _mm256_mul_pd(r, r);
        __m256d y = _mm256_add_pd(_mm256_mul_pd(c2, r), one);
        y = _mm256_add_pd(_mm256_mul_pd(p, r2), y);
        half[h] = _mm256_cvtpd_ps(_mm256_mul_pd(y, _mm256_castsi256_pd(t)));
    }
    return _mm256_set_m128(half[1], half[0]);
}
#endif

#if defined(SPINGALETT_ATTEND_VARIANT)
void SPINGALETT_ATTEND_VARIANT(const float *q, const float *blocks, const float *values, size_t value_stride,
                               uint32_t first, uint32_t count, uint32_t head, float scale, float *o);
#  define ATTEND_FUNCTION void SPINGALETT_ATTEND_VARIANT
#else
#  define ATTEND_FUNCTION static void attend_here
#endif

#if defined(ATTEND_AVX2)
/* The keys a span takes: four blocks, whose scores are summed side by side (four sums over the head's values at
   once, each in the order of the values). */
#define SPAN (4u * SPG_KEY_BLOCK)

/* The scores of keys lane0 .. lane0 + n - 1 of the span of blocks from kb into scores[], times scale. */
static inline __attribute__((always_inline)) void span_scores(const float *q, const float *kb, uint32_t head, uint32_t lane0,
                                                              uint32_t n, float scale, float *scores) {
    const uint32_t blocks = (lane0 + n + SPG_KEY_BLOCK - 1u) / SPG_KEY_BLOCK;
    const size_t step = (size_t)head * SPG_KEY_BLOCK;
    const __m256 sc = _mm256_set1_ps(scale);
    if (blocks == 4u) {
        __m256 s0 = _mm256_setzero_ps(), s1 = s0, s2 = s0, s3 = s0;
        for (uint32_t d = 0; d < head; d++) {
            const __m256 qd = _mm256_set1_ps(q[d]);
            const float *k = kb + (size_t)d * SPG_KEY_BLOCK;
            s0 = _mm256_add_ps(s0, _mm256_mul_ps(qd, _mm256_loadu_ps(k)));
            s1 = _mm256_add_ps(s1, _mm256_mul_ps(qd, _mm256_loadu_ps(k + step)));
            s2 = _mm256_add_ps(s2, _mm256_mul_ps(qd, _mm256_loadu_ps(k + 2u * step)));
            s3 = _mm256_add_ps(s3, _mm256_mul_ps(qd, _mm256_loadu_ps(k + 3u * step)));
        }
        _mm256_storeu_ps(scores, _mm256_mul_ps(s0, sc));
        _mm256_storeu_ps(scores + 8, _mm256_mul_ps(s1, sc));
        _mm256_storeu_ps(scores + 16, _mm256_mul_ps(s2, sc));
        _mm256_storeu_ps(scores + 24, _mm256_mul_ps(s3, sc));
        return;
    }
    for (uint32_t b = 0; b < blocks; b++) {
        __m256 sv = _mm256_setzero_ps();
        for (uint32_t d = 0; d < head; d++)
            sv = _mm256_add_ps(sv, _mm256_mul_ps(_mm256_set1_ps(q[d]), _mm256_loadu_ps(kb + b * step + (size_t)d * SPG_KEY_BLOCK)));
        _mm256_storeu_ps(scores + 8u * b, _mm256_mul_ps(sv, sc));
    }
}

/* Keys i0 .. i1 - 1 of a span added to o[0 .. 8 V - 1] (V vectors, in registers): per key o r where the running
   maximum grew (`grow`), then o + w v, as the engine adds them. */
#define VALUES_OF(V) \
static inline __attribute__((always_inline)) void values##V(float *o, const float *v, size_t stride, uint32_t i0, uint32_t i1, \
                                                            const float *r, const float *w, const bool *grows, bool grow) { \
    __m256 a[V]; \
    for (uint32_t k = 0; k < (V); k++) a[k] = _mm256_loadu_ps(o + 8u * k); \
    for (uint32_t i = i0; i < i1; i++, v += stride) { \
        if (grow && grows[i]) { \
            const __m256 ri = _mm256_set1_ps(r[i]); \
            for (uint32_t k = 0; k < (V); k++) a[k] = _mm256_mul_ps(a[k], ri); \
        } \
        const __m256 wi = _mm256_set1_ps(w[i]); \
        for (uint32_t k = 0; k < (V); k++) a[k] = _mm256_add_ps(a[k], _mm256_mul_ps(wi, _mm256_loadu_ps(v + 8u * k))); \
    } \
    for (uint32_t k = 0; k < (V); k++) _mm256_storeu_ps(o + 8u * k, a[k]); \
}
VALUES_OF(8)
VALUES_OF(1)
#undef VALUES_OF
#endif

ATTEND_FUNCTION(const float *q, const float *blocks, const float *values, size_t value_stride, uint32_t first,
                uint32_t count, uint32_t head, float scale, float *o) {
    float m = -FLT_MAX, sum = 0.0f;
    memset(o, 0, (size_t)head * sizeof(float));
#if defined(ATTEND_AVX2)
    for (uint32_t p = first, end = first + count; p < end;) {
        /* the keys of up to four blocks from p on: their scores side by side */
        const uint32_t lane0 = p % SPG_KEY_BLOCK, n = SPAN - lane0 < end - p ? SPAN - lane0 : end - p;
        float scores[SPAN], ra[SPAN] = {0}, wa[SPAN] = {0}, r[SPAN], w[SPAN];
        bool grows[SPAN], grow = false;
        span_scores(q, blocks + (size_t)(p / SPG_KEY_BLOCK) * head * SPG_KEY_BLOCK, head, lane0, n, scale, scores);
        /* the running maximum key by key, the arguments of the exponentials (0 for the lanes outside the keys) */
        for (uint32_t i = lane0; i < lane0 + n; i++) {
            grows[i] = scores[i] > m;
            if (grows[i]) {
                ra[i] = m - scores[i];
                m = scores[i];
                grow = true;
            }
            wa[i] = scores[i] - m;
        }
        for (uint32_t k = lane0 / 8u * 8u; k < lane0 + n; k += 8u) {
            _mm256_storeu_ps(w + k, expf8(_mm256_loadu_ps(wa + k)));
            if (grow) _mm256_storeu_ps(r + k, expf8(_mm256_loadu_ps(ra + k)));
        }
        for (uint32_t i = lane0; i < lane0 + n; i++) {
            if (grows[i]) sum *= r[i];
            sum += w[i];
        }
        /* the weighted values, a slice of the head at a time kept in registers across the keys */
        const float *v0 = values + (size_t)p * value_stride - (size_t)lane0 * value_stride;
        uint32_t d = 0;
        for (; d + 64u <= head; d += 64u) values8(o + d, v0 + d + (size_t)lane0 * value_stride, value_stride, lane0, lane0 + n,
                                                  r, w, grows, grow);
        for (; d + 8u <= head; d += 8u) values1(o + d, v0 + d + (size_t)lane0 * value_stride, value_stride, lane0, lane0 + n,
                                                r, w, grows, grow);
        for (; d < head; d++)
            for (uint32_t i = lane0; i < lane0 + n; i++) {
                if (grows[i]) o[d] *= r[i];
                o[d] += w[i] * v0[(size_t)i * value_stride + d];
            }
        p += n;
    }
#else
    for (uint32_t p = first, end = first + count; p < end;) {
        /* the keys of one block from p on, key by key */
        const uint32_t lane0 = p % SPG_KEY_BLOCK, n = SPG_KEY_BLOCK - lane0 < end - p ? SPG_KEY_BLOCK - lane0 : end - p;
        const float *kb = blocks + (size_t)(p / SPG_KEY_BLOCK) * head * SPG_KEY_BLOCK;
        for (uint32_t i = lane0; i < lane0 + n; i++) {
            float score = 0.0f;
            for (uint32_t d = 0; d < head; d++) score += q[d] * kb[(size_t)d * SPG_KEY_BLOCK + i];
            score *= scale;
            const float *v = values + (size_t)(p + i - lane0) * value_stride;
            if (score > m) {
                const float r = spingalett_expf(m - score);
                m = score;
                sum *= r;
                for (uint32_t d = 0; d < head; d++) o[d] *= r;
            }
            const float w = spingalett_expf(score - m);
            sum += w;
            for (uint32_t d = 0; d < head; d++) o[d] += w * v[d];
        }
        p += n;
    }
#endif
    const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
    for (uint32_t d = 0; d < head; d++) o[d] *= inv;
}

#if !defined(SPINGALETT_ATTEND_VARIANT)
#if defined(SPINGALETT_BLOCK_DISPATCH) && !defined(ATTEND_AVX2) && !defined(SPINGALETT_PORTABLE_KERNELS) && \
    (defined(__x86_64__) || defined(__i386__))
#  define ATTEND_DISPATCH_X86 1
void spingalett_attend_blocks_avx2(const float *q, const float *blocks, const float *values, size_t value_stride,
                                   uint32_t first, uint32_t count, uint32_t head, float scale, float *o);
#endif

void spingalett_attend_blocks(const float *q, const float *blocks, const float *values, size_t value_stride,
                              uint32_t first, uint32_t count, uint32_t head, float scale, float *o) {
#if defined(ATTEND_DISPATCH_X86)
    static int avx2 = -1;
    if (avx2 < 0) {
        __builtin_cpu_init();
        avx2 = __builtin_cpu_supports("avx2") ? 1 : 0;
    }
    if (avx2) {
        spingalett_attend_blocks_avx2(q, blocks, values, value_stride, first, count, head, scale, o);
        return;
    }
#endif
    attend_here(q, blocks, values, value_stride, first, count, head, scale, o);
}
#endif
