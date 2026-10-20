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

void spingalett_attend_blocks(const float *q, const float *blocks, const float *values, size_t value_stride,
                              uint32_t first, uint32_t count, uint32_t head, float scale, float *o) {
    float m = -FLT_MAX, sum = 0.0f;
    memset(o, 0, (size_t)head * sizeof(float));
    for (uint32_t p = first, end = first + count; p < end;) {
        /* the keys of one block from p on: their scores side by side */
        const uint32_t lane0 = p % SPG_KEY_BLOCK, n = SPG_KEY_BLOCK - lane0 < end - p ? SPG_KEY_BLOCK - lane0 : end - p;
        const float *kb = blocks + (size_t)(p / SPG_KEY_BLOCK) * head * SPG_KEY_BLOCK;
        float scores[SPG_KEY_BLOCK], r[SPG_KEY_BLOCK], w[SPG_KEY_BLOCK];
        bool grows[SPG_KEY_BLOCK];
#if defined(ATTEND_AVX2)
        __m256 sv = _mm256_setzero_ps();
        for (uint32_t d = 0; d < head; d++)
            sv = _mm256_add_ps(sv, _mm256_mul_ps(_mm256_set1_ps(q[d]), _mm256_loadu_ps(kb + (size_t)d * SPG_KEY_BLOCK)));
        _mm256_storeu_ps(scores, _mm256_mul_ps(sv, _mm256_set1_ps(scale)));
#else
        for (uint32_t i = lane0; i < lane0 + n; i++) {
            float score = 0.0f;
            for (uint32_t d = 0; d < head; d++) score += q[d] * kb[(size_t)d * SPG_KEY_BLOCK + i];
            scores[i] = score * scale;
        }
#endif
        /* the running maximum key by key, the arguments of the exponentials (0 for the lanes outside the keys) */
        float ra[SPG_KEY_BLOCK] = {0}, wa[SPG_KEY_BLOCK] = {0};
        for (uint32_t i = lane0; i < lane0 + n; i++) {
            grows[i] = scores[i] > m;
            if (grows[i]) {
                ra[i] = m - scores[i];
                m = scores[i];
            }
            wa[i] = scores[i] - m;
        }
#if defined(ATTEND_AVX2)
        _mm256_storeu_ps(r, expf8(_mm256_loadu_ps(ra)));
        _mm256_storeu_ps(w, expf8(_mm256_loadu_ps(wa)));
#else
        for (uint32_t i = lane0; i < lane0 + n; i++) {
            r[i] = grows[i] ? spingalett_expf(ra[i]) : 1.0f;
            w[i] = spingalett_expf(wa[i]);
        }
#endif
        for (uint32_t i = lane0; i < lane0 + n; i++) {
            const float *v = values + (size_t)(p + i - lane0) * value_stride;
            if (grows[i]) {
                sum *= r[i];
                for (uint32_t d = 0; d < head; d++) o[d] *= r[i];
            }
            sum += w[i];
            for (uint32_t d = 0; d < head; d++) o[d] += w[i] * v[d];
        }
        p += n;
    }
    const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
    for (uint32_t d = 0; d < head; d++) o[d] *= inv;
}
