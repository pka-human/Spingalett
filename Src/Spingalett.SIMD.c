/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

#include "Spingalett.Private.h"
#include <math.h>
#include <float.h>
#include <string.h>

#if defined(__AVX__)
#include <immintrin.h>

#if defined(__FMA__)
#define FMADD256(a, b, c)  _mm256_fmadd_ps((a), (b), (c))
#define FNMADD256(a, b, c) _mm256_fnmadd_ps((a), (b), (c))
#else
#define FMADD256(a, b, c)  _mm256_add_ps(_mm256_mul_ps((a), (b)), (c))
#define FNMADD256(a, b, c) _mm256_sub_ps((c), _mm256_mul_ps((a), (b)))
#endif

static inline float hsum256_ps(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    __m128 shuf = _mm_movehdup_ps(lo);
    __m128 sums = _mm_add_ps(lo, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}
#endif

#if defined(__AVX2__)
/* exp(x) via 2^n * P(r) with the Cephes polynomial (~1-2 ulp on the normal range).
   NaN is kept as the second operand of min/max so it propagates. */
static inline __m256 exp256_ps(__m256 x) {
    x = _mm256_min_ps(_mm256_set1_ps(88.3762626647949f), x);
    x = _mm256_max_ps(_mm256_set1_ps(-88.3762626647949f), x);

    __m256 fx = _mm256_floor_ps(FMADD256(x, _mm256_set1_ps(1.44269504088896341f), _mm256_set1_ps(0.5f)));
    x = FNMADD256(fx, _mm256_set1_ps(0.693359375f), x);
    x = FNMADD256(fx, _mm256_set1_ps(-2.12194440e-4f), x);

    __m256 y = _mm256_set1_ps(1.9875691500e-4f);
    y = FMADD256(y, x, _mm256_set1_ps(1.3981999507e-3f));
    y = FMADD256(y, x, _mm256_set1_ps(8.3334519073e-3f));
    y = FMADD256(y, x, _mm256_set1_ps(4.1665795894e-2f));
    y = FMADD256(y, x, _mm256_set1_ps(1.6666665459e-1f));
    y = FMADD256(y, x, _mm256_set1_ps(5.0000001201e-1f));
    y = FMADD256(y, _mm256_mul_ps(x, x), x);
    y = _mm256_add_ps(y, _mm256_set1_ps(1.0f));

    __m256i n = _mm256_add_epi32(_mm256_cvttps_epi32(fx), _mm256_set1_epi32(0x7F));
    return _mm256_mul_ps(y, _mm256_castsi256_ps(_mm256_slli_epi32(n, 23)));
}

static inline __m256 sigmoid256_ps(__m256 x) {
    __m256 one = _mm256_set1_ps(1.0f);
    return _mm256_div_ps(one, _mm256_add_ps(one, exp256_ps(_mm256_sub_ps(_mm256_setzero_ps(), x))));
}

/* tanh(|x|) = (1 - e) / (1 + e), e = exp(-2|x|); below |x| = 0.3 that form loses relative
   precision to cancellation, so an odd Taylor polynomial (error < 1e-9) is used there. */
static inline __m256 tanh256_ps(__m256 x) {
    __m256 sign_bit = _mm256_set1_ps(-0.0f);
    __m256 one = _mm256_set1_ps(1.0f);
    __m256 ax = _mm256_andnot_ps(sign_bit, x);

    __m256 e = exp256_ps(_mm256_mul_ps(ax, _mm256_set1_ps(-2.0f)));
    __m256 large = _mm256_div_ps(_mm256_sub_ps(one, e), _mm256_add_ps(one, e));

    __m256 x2 = _mm256_mul_ps(ax, ax);
    __m256 p = _mm256_set1_ps(-8.86323552990219656e-3f);
    p = FMADD256(p, x2, _mm256_set1_ps(2.18694885361552028e-2f));
    p = FMADD256(p, x2, _mm256_set1_ps(-5.39682539682539683e-2f));
    p = FMADD256(p, x2, _mm256_set1_ps(1.33333333333333333e-1f));
    p = FMADD256(p, x2, _mm256_set1_ps(-3.33333333333333333e-1f));
    __m256 small = FMADD256(_mm256_mul_ps(ax, x2), p, ax);

    __m256 r = _mm256_blendv_ps(large, small, _mm256_cmp_ps(ax, _mm256_set1_ps(0.3f), _CMP_LT_OQ));
    return _mm256_or_ps(r, _mm256_and_ps(sign_bit, x));
}
#endif

void apply_softmax(float *layer, uint32_t size) {
    float max_val = -FLT_MAX;
    for (uint32_t i = 0; i < size; i++)
        if (layer[i] > max_val) max_val = layer[i];

    float sum = 0.0f;
    uint32_t i = 0;
#if defined(__AVX2__)
    __m256 vmax = _mm256_set1_ps(max_val);
    __m256 vsum = _mm256_setzero_ps();
    for (; i + 8u <= size; i += 8u) {
        __m256 e = exp256_ps(_mm256_sub_ps(_mm256_loadu_ps(layer + i), vmax));
        _mm256_storeu_ps(layer + i, e);
        vsum = _mm256_add_ps(vsum, e);
    }
    sum = hsum256_ps(vsum);
#endif
    for (; i < size; i++) {
        layer[i] = expf(layer[i] - max_val);
        sum += layer[i];
    }

    if (sum > 0.0f)
        spingalett_vec_scale(layer, (uint64_t)size, 1.0f / sum);
}

void apply_activation_batch(float *data, uint32_t size, ActivationFunction act) {
    if (act == ACT_SOFTMAX) {
        apply_softmax(data, size);
        return;
    }
    apply_activation_bulk(data, (uint64_t)size, act);
}

void apply_activation_bulk(float *data, uint64_t total, ActivationFunction act) {
    if (act == ACT_NONE || (unsigned)act >= ACT_COUNT || act == ACT_SOFTMAX)
        return;

    uint64_t i = 0;

#if defined(__AVX__)
    if (act == ACT_RELU) {
        __m256 zero = _mm256_setzero_ps();
        for (; i + 32u <= total; i += 32u) {
            _mm256_storeu_ps(data + i,      _mm256_max_ps(_mm256_loadu_ps(data + i), zero));
            _mm256_storeu_ps(data + i + 8u,  _mm256_max_ps(_mm256_loadu_ps(data + i + 8u), zero));
            _mm256_storeu_ps(data + i + 16u, _mm256_max_ps(_mm256_loadu_ps(data + i + 16u), zero));
            _mm256_storeu_ps(data + i + 24u, _mm256_max_ps(_mm256_loadu_ps(data + i + 24u), zero));
        }
        for (; i + 8u <= total; i += 8u) {
            _mm256_storeu_ps(data + i, _mm256_max_ps(_mm256_loadu_ps(data + i), zero));
        }
    } else if (act == ACT_LEAKY_RELU) {
        __m256 zero = _mm256_setzero_ps();
        __m256 alpha = _mm256_set1_ps(0.01f);
        for (; i + 8u <= total; i += 8u) {
            __m256 v = _mm256_loadu_ps(data + i);
            __m256 mask = _mm256_cmp_ps(v, zero, _CMP_GT_OQ);
            _mm256_storeu_ps(data + i, _mm256_blendv_ps(_mm256_mul_ps(v, alpha), v, mask));
        }
    } else if (act == ACT_FOO52) {
        __m256 zero = _mm256_setzero_ps();
        __m256 one = _mm256_set1_ps(1.0f);
        __m256 alpha = _mm256_set1_ps(0.01f);
        for (; i + 8u <= total; i += 8u) {
            __m256 x = _mm256_loadu_ps(data + i);
            __m256 neg_mask = _mm256_cmp_ps(x, zero, _CMP_LT_OQ);
            __m256 over_mask = _mm256_cmp_ps(x, one, _CMP_GT_OQ);
            __m256 neg_result = _mm256_mul_ps(x, alpha);
            __m256 over_result = _mm256_add_ps(one, _mm256_mul_ps(_mm256_sub_ps(x, one), alpha));
            __m256 result = x;
            result = _mm256_blendv_ps(result, neg_result, neg_mask);
            result = _mm256_blendv_ps(result, over_result, over_mask);
            _mm256_storeu_ps(data + i, result);
        }
    }
#endif
#if defined(__AVX2__)
    if (act == ACT_SIGMOID) {
        for (; i + 8u <= total; i += 8u)
            _mm256_storeu_ps(data + i, sigmoid256_ps(_mm256_loadu_ps(data + i)));
    } else if (act == ACT_TANH) {
        for (; i + 8u <= total; i += 8u)
            _mm256_storeu_ps(data + i, tanh256_ps(_mm256_loadu_ps(data + i)));
    }
#endif

    for (; i < total; i++)
        data[i] = activate(data[i], act);
}

void apply_derivative_batch(float *deriv, const float *act_data, uint64_t total, ActivationFunction act) {
    uint64_t i = 0;

#if defined(__AVX__)
    if (act == ACT_RELU) {
        __m256 zero = _mm256_setzero_ps();
        __m256 one = _mm256_set1_ps(1.0f);
        for (; i + 16u <= total; i += 16u) {
            __m256 a0 = _mm256_loadu_ps(act_data + i);
            __m256 a1 = _mm256_loadu_ps(act_data + i + 8u);
            __m256 d0 = _mm256_loadu_ps(deriv + i);
            __m256 d1 = _mm256_loadu_ps(deriv + i + 8u);
            __m256 m0 = _mm256_and_ps(one, _mm256_cmp_ps(a0, zero, _CMP_GT_OQ));
            __m256 m1 = _mm256_and_ps(one, _mm256_cmp_ps(a1, zero, _CMP_GT_OQ));
            _mm256_storeu_ps(deriv + i, _mm256_mul_ps(d0, m0));
            _mm256_storeu_ps(deriv + i + 8u, _mm256_mul_ps(d1, m1));
        }
        for (; i + 8u <= total; i += 8u) {
            __m256 a = _mm256_loadu_ps(act_data + i);
            __m256 d = _mm256_loadu_ps(deriv + i);
            __m256 mask = _mm256_and_ps(one, _mm256_cmp_ps(a, zero, _CMP_GT_OQ));
            _mm256_storeu_ps(deriv + i, _mm256_mul_ps(d, mask));
        }
    } else if (act == ACT_LEAKY_RELU) {
        __m256 zero = _mm256_setzero_ps();
        __m256 one = _mm256_set1_ps(1.0f);
        __m256 alpha = _mm256_set1_ps(0.01f);
        for (; i + 8u <= total; i += 8u) {
            __m256 a = _mm256_loadu_ps(act_data + i);
            __m256 d = _mm256_loadu_ps(deriv + i);
            __m256 mask = _mm256_cmp_ps(a, zero, _CMP_GT_OQ);
            __m256 coeff = _mm256_blendv_ps(alpha, one, mask);
            _mm256_storeu_ps(deriv + i, _mm256_mul_ps(d, coeff));
        }
    } else if (act == ACT_FOO52) {
        __m256 zero = _mm256_setzero_ps();
        __m256 one = _mm256_set1_ps(1.0f);
        __m256 alpha = _mm256_set1_ps(0.01f);
        for (; i + 8u <= total; i += 8u) {
            __m256 a = _mm256_loadu_ps(act_data + i);
            __m256 d = _mm256_loadu_ps(deriv + i);
            __m256 outside = _mm256_or_ps(_mm256_cmp_ps(a, one, _CMP_GT_OQ), _mm256_cmp_ps(a, zero, _CMP_LT_OQ));
            _mm256_storeu_ps(deriv + i, _mm256_mul_ps(d, _mm256_blendv_ps(one, alpha, outside)));
        }
    } else if (act == ACT_SIGMOID || act == ACT_SOFTMAX) {
        __m256 one = _mm256_set1_ps(1.0f);
        for (; i + 8u <= total; i += 8u) {
            __m256 a = _mm256_loadu_ps(act_data + i);
            __m256 d = _mm256_loadu_ps(deriv + i);
            _mm256_storeu_ps(deriv + i, _mm256_mul_ps(d, _mm256_mul_ps(a, _mm256_sub_ps(one, a))));
        }
    } else if (act == ACT_TANH) {
        __m256 one = _mm256_set1_ps(1.0f);
        for (; i + 8u <= total; i += 8u) {
            __m256 a = _mm256_loadu_ps(act_data + i);
            __m256 d = _mm256_loadu_ps(deriv + i);
            _mm256_storeu_ps(deriv + i, _mm256_mul_ps(d, FNMADD256(a, a, one)));
        }
    }
#endif

    switch (act) {
        case ACT_RELU:
            for (; i < total; i++)
                deriv[i] *= (act_data[i] > 0.0f) ? 1.0f : 0.0f;
            break;
        case ACT_LEAKY_RELU:
            for (; i < total; i++)
                deriv[i] *= (act_data[i] > 0.0f) ? 1.0f : 0.01f;
            break;
        case ACT_SIGMOID:
        case ACT_SOFTMAX:
            for (; i < total; i++)
                deriv[i] *= act_data[i] * (1.0f - act_data[i]);
            break;
        case ACT_TANH:
            for (; i < total; i++)
                deriv[i] *= 1.0f - act_data[i] * act_data[i];
            break;
        case ACT_FOO52:
            for (; i < total; i++)
                deriv[i] *= (act_data[i] > 1.0f || act_data[i] < 0.0f) ? 0.01f : 1.0f;
            break;
        default:
            break;
    }
}

float spingalett_dot_product(const float *restrict a, const float *restrict b, uint64_t n) {
#if defined(__AVX__)
    __m256 sum0 = _mm256_setzero_ps();
    __m256 sum1 = _mm256_setzero_ps();
    __m256 sum2 = _mm256_setzero_ps();
    __m256 sum3 = _mm256_setzero_ps();
    uint64_t i = 0;
    for (; i + 32u <= n; i += 32u) {
        sum0 = FMADD256(_mm256_loadu_ps(a + i),       _mm256_loadu_ps(b + i),       sum0);
        sum1 = FMADD256(_mm256_loadu_ps(a + i + 8u),  _mm256_loadu_ps(b + i + 8u),  sum1);
        sum2 = FMADD256(_mm256_loadu_ps(a + i + 16u), _mm256_loadu_ps(b + i + 16u), sum2);
        sum3 = FMADD256(_mm256_loadu_ps(a + i + 24u), _mm256_loadu_ps(b + i + 24u), sum3);
    }
    for (; i + 8u <= n; i += 8u)
        sum0 = FMADD256(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), sum0);
    sum0 = _mm256_add_ps(sum0, sum1);
    sum2 = _mm256_add_ps(sum2, sum3);
    sum0 = _mm256_add_ps(sum0, sum2);
    float sum = hsum256_ps(sum0);
    for (; i < n; i++)
        sum += a[i] * b[i];
    return sum;
#else
    float sum = 0.0f;
    for (uint64_t i = 0; i < n; i++)
        sum += a[i] * b[i];
    return sum;
#endif
}

/* Sum of squares. Float lanes accumulate blocks of 4096 elements, block sums are added in
   double, so the result stays accurate for parameter vectors of any size. */
double spingalett_vec_sumsq(const float *x, uint64_t n) {
    const uint64_t block = 4096u;
    double total = 0.0;
    for (uint64_t b = 0; b < n; b += block) {
        uint64_t end = (n - b < block) ? n : b + block;
        uint64_t i = b;
        float sum = 0.0f;
#if defined(__AVX__)
        __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
        __m256 s2 = _mm256_setzero_ps(), s3 = _mm256_setzero_ps();
        for (; i + 32u <= end; i += 32u) {
            __m256 a0 = _mm256_loadu_ps(x + i),       a1 = _mm256_loadu_ps(x + i + 8u);
            __m256 a2 = _mm256_loadu_ps(x + i + 16u), a3 = _mm256_loadu_ps(x + i + 24u);
            s0 = FMADD256(a0, a0, s0); s1 = FMADD256(a1, a1, s1);
            s2 = FMADD256(a2, a2, s2); s3 = FMADD256(a3, a3, s3);
        }
        for (; i + 8u <= end; i += 8u) {
            __m256 a = _mm256_loadu_ps(x + i);
            s0 = FMADD256(a, a, s0);
        }
        sum = hsum256_ps(_mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3)));
#endif
        for (; i < end; i++)
            sum += x[i] * x[i];
        total += (double)sum;
    }
    return total;
}

float spingalett_vec_l2norm(const float *x, uint64_t n) {
    return (float)sqrt(spingalett_vec_sumsq(x, n));
}

void spingalett_vec_scale(float *data, uint64_t n, float scale) {
    uint64_t i = 0;
#if defined(__AVX__)
    __m256 vs = _mm256_set1_ps(scale);
    for (; i + 8u <= n; i += 8u)
        _mm256_storeu_ps(data + i, _mm256_mul_ps(_mm256_loadu_ps(data + i), vs));
#endif
    for (; i < n; i++)
        data[i] *= scale;
}

void spingalett_vec_mul(float *restrict y, const float *restrict x, uint64_t n) {
    uint64_t i = 0;
#if defined(__AVX__)
    for (; i + 8u <= n; i += 8u)
        _mm256_storeu_ps(y + i, _mm256_mul_ps(_mm256_loadu_ps(y + i), _mm256_loadu_ps(x + i)));
#endif
    for (; i < n; i++)
        y[i] *= x[i];
}

/* lowbias32 (Wellons): a fast 32-bit bijective hash with good avalanche. */
static inline uint32_t hash32(uint32_t x) {
    x ^= x >> 16; x *= 0x7FEB352Du;
    x ^= x >> 15; x *= 0x846CA68Bu;
    x ^= x >> 16;
    return x;
}

static inline uint64_t mix64(uint64_t z) {
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

void spingalett_dropout_apply(float *restrict y, float *restrict dmask, uint32_t n,
                              ActivationFunction act, float rate,
                              const DropoutContext *ctx, uint32_t layer, uint32_t position) {
    /* Unit j is kept when hash(base + j * golden) >= rate * 2^32. */
    uint32_t base = (uint32_t)(mix64(ctx->seed ^ mix64(ctx->step * 0x9E3779B97F4A7C15ull +
                                     ((uint64_t)position << 20) + layer)) >> 32);
    uint32_t threshold = (uint32_t)((double)rate * 4294967296.0);
    float scale = 1.0f / (1.0f - rate);

    float m[256];
    for (uint32_t k0 = 0; k0 < n; k0 += 256u) {
        uint32_t len = (n - k0 < 256u) ? n - k0 : 256u;
        for (uint32_t j = 0; j < len; j++)
            m[j] = hash32(base + (k0 + j) * 0x9E3779B9u) >= threshold ? scale : 0.0f;
        /* The derivative is taken from the unmasked activation, then both are masked. */
        memcpy(dmask + k0, m, len * sizeof(float));
        apply_derivative_batch(dmask + k0, y + k0, (uint64_t)len, act);
        spingalett_vec_mul(y + k0, m, (uint64_t)len);
    }
}

void spingalett_vec_scaled_copy(float *restrict dst, const float *restrict src, uint64_t n, float alpha) {
    uint64_t i = 0;
#if defined(__AVX__)
    __m256 va = _mm256_set1_ps(alpha);
    for (; i + 8u <= n; i += 8u)
        _mm256_storeu_ps(dst + i, _mm256_mul_ps(_mm256_loadu_ps(src + i), va));
#endif
    for (; i < n; i++)
        dst[i] = alpha * src[i];
}

void spingalett_vec_axpy(float *restrict y, const float *restrict x, uint64_t n, float alpha) {
    uint64_t i = 0;
#if defined(__AVX__)
    __m256 va = _mm256_set1_ps(alpha);
    for (; i + 8u <= n; i += 8u)
        _mm256_storeu_ps(y + i, FMADD256(va, _mm256_loadu_ps(x + i), _mm256_loadu_ps(y + i)));
#endif
    for (; i < n; i++)
        y[i] += alpha * x[i];
}

/*
 * Optimizer kernels. Weight decay is the classic coupled L2 term (g' = g + decay * w) for
 * every optimizer except AdamW, which instead passes decay = 0 and a decoupled
 * wd_factor = 1 - lr * weight_decay. Epsilon is added to the square root of the second
 * moment (sqrt(v) + eps), as in PyTorch and TensorFlow. The same kernels serve the
 * per-sample and batch paths, so both strategies follow identical update rules.
 */

void spingalett_sgd_update(float *restrict W, const float *restrict gW, uint64_t n, float lr, float decay) {
    uint64_t i = 0;
#if defined(__AVX__)
    __m256 v_lr = _mm256_set1_ps(lr);
    __m256 v_decay = _mm256_set1_ps(decay);
    for (; i + 8u <= n; i += 8u) {
        __m256 w = _mm256_loadu_ps(W + i);
        __m256 g = FMADD256(v_decay, w, _mm256_loadu_ps(gW + i));
        _mm256_storeu_ps(W + i, FNMADD256(v_lr, g, w));
    }
#endif
    for (; i < n; i++)
        W[i] -= lr * (gW[i] + decay * W[i]);
}

void spingalett_momentum_update(float *restrict W, float *restrict mW, const float *restrict gW, uint64_t n,
                                float lr, float momentum, float decay) {
    uint64_t i = 0;
#if defined(__AVX__)
    __m256 v_lr = _mm256_set1_ps(lr);
    __m256 v_mom = _mm256_set1_ps(momentum);
    __m256 v_decay = _mm256_set1_ps(decay);
    for (; i + 8u <= n; i += 8u) {
        __m256 w = _mm256_loadu_ps(W + i);
        __m256 g = FMADD256(v_decay, w, _mm256_loadu_ps(gW + i));
        __m256 m = FMADD256(v_mom, _mm256_loadu_ps(mW + i), g);
        _mm256_storeu_ps(mW + i, m);
        _mm256_storeu_ps(W + i, FNMADD256(v_lr, m, w));
    }
#endif
    for (; i < n; i++) {
        mW[i] = momentum * mW[i] + (gW[i] + decay * W[i]);
        W[i] -= lr * mW[i];
    }
}

void spingalett_rmsprop_update(float *restrict W, float *restrict vW, const float *restrict gW, uint64_t n,
                               float lr, float beta2, float epsilon, float decay) {
    float one_minus_b2 = 1.0f - beta2;
    uint64_t i = 0;
#if defined(__AVX__)
    __m256 v_lr = _mm256_set1_ps(lr);
    __m256 v_b2 = _mm256_set1_ps(beta2);
    __m256 v_1mb2 = _mm256_set1_ps(one_minus_b2);
    __m256 v_eps = _mm256_set1_ps(epsilon);
    __m256 v_decay = _mm256_set1_ps(decay);
    for (; i + 8u <= n; i += 8u) {
        __m256 w = _mm256_loadu_ps(W + i);
        __m256 g = FMADD256(v_decay, w, _mm256_loadu_ps(gW + i));
        __m256 v = FMADD256(v_b2, _mm256_loadu_ps(vW + i), _mm256_mul_ps(v_1mb2, _mm256_mul_ps(g, g)));
        _mm256_storeu_ps(vW + i, v);
        __m256 step = _mm256_div_ps(g, _mm256_add_ps(_mm256_sqrt_ps(v), v_eps));
        _mm256_storeu_ps(W + i, FNMADD256(v_lr, step, w));
    }
#endif
    for (; i < n; i++) {
        float g = gW[i] + decay * W[i];
        vW[i] = beta2 * vW[i] + one_minus_b2 * (g * g);
        W[i] -= lr * (g / (sqrtf(vW[i]) + epsilon));
    }
}

void spingalett_adam_update(float *restrict W, float *restrict mW, float *restrict vW, const float *restrict gW,
                            uint64_t n, float lr, float beta1, float beta2,
                            float m_factor, float v_factor, float epsilon,
                            float decay, float wd_factor) {
    float one_minus_b1 = 1.0f - beta1;
    float one_minus_b2 = 1.0f - beta2;
    uint64_t i = 0;
#if defined(__AVX__)
    __m256 v_lr = _mm256_set1_ps(lr);
    __m256 v_mf = _mm256_set1_ps(m_factor);
    __m256 v_vf = _mm256_set1_ps(v_factor);
    __m256 v_eps = _mm256_set1_ps(epsilon);
    __m256 v_decay = _mm256_set1_ps(decay);
    __m256 v_wd = _mm256_set1_ps(wd_factor);
    __m256 v_b1 = _mm256_set1_ps(beta1);
    __m256 v_b2 = _mm256_set1_ps(beta2);
    __m256 v_1mb1 = _mm256_set1_ps(one_minus_b1);
    __m256 v_1mb2 = _mm256_set1_ps(one_minus_b2);
    for (; i + 8u <= n; i += 8u) {
        __m256 w = _mm256_loadu_ps(W + i);
        __m256 g = FMADD256(v_decay, w, _mm256_loadu_ps(gW + i));
        __m256 m = FMADD256(v_b1, _mm256_loadu_ps(mW + i), _mm256_mul_ps(v_1mb1, g));
        __m256 v = FMADD256(v_b2, _mm256_loadu_ps(vW + i), _mm256_mul_ps(v_1mb2, _mm256_mul_ps(g, g)));
        _mm256_storeu_ps(mW + i, m);
        _mm256_storeu_ps(vW + i, v);
        __m256 denom = _mm256_add_ps(_mm256_sqrt_ps(_mm256_mul_ps(v, v_vf)), v_eps);
        __m256 step = _mm256_div_ps(_mm256_mul_ps(m, v_mf), denom);
        _mm256_storeu_ps(W + i, FNMADD256(v_lr, step, _mm256_mul_ps(w, v_wd)));
    }
#endif
    for (; i < n; i++) {
        float g = gW[i] + decay * W[i];
        mW[i] = beta1 * mW[i] + one_minus_b1 * g;
        vW[i] = beta2 * vW[i] + one_minus_b2 * (g * g);
        float m_hat = mW[i] * m_factor;
        float v_hat = vW[i] * v_factor;
        W[i] = W[i] * wd_factor - lr * (m_hat / (sqrtf(v_hat) + epsilon));
    }
}

float spingalett_clip_grad_norm(NeuralNetwork *net, float max_norm) {
    if (max_norm <= 0.0f) return 0.0f;

    /* Gradients of all layers are contiguous, so the global norm covers two flat arrays. */
    double total_sq = spingalett_vec_sumsq(net->grad_weights, net->total_weights)
                    + spingalett_vec_sumsq(net->grad_biases, net->total_biases);

    float grad_norm = (float)sqrt(total_sq);
    /* A non-finite norm is left alone so the NaN check can report the divergence. */
    if (isfinite(grad_norm) && grad_norm > max_norm) {
        float scale = max_norm / grad_norm;
        spingalett_vec_scale(net->grad_weights, net->total_weights, scale);
        spingalett_vec_scale(net->grad_biases, net->total_biases, scale);
    }
    return grad_norm;
}

float compute_sample_loss(const float *output, const float *target,
                          uint32_t output_size, LossFunction loss_func,
                          ActivationFunction output_act) {
    const float epsilon_log = 1e-9f;
    float loss = 0.0f;

    if (loss_func == LOSS_MSE) {
        for (uint32_t k = 0; k < output_size; k++) {
            float diff = output[k] - target[k];
            loss += diff * diff;
        }
    } else if (loss_func == LOSS_CROSS_ENTROPY) {
        if (output_act == ACT_SIGMOID) {
            for (uint32_t k = 0; k < output_size; k++) {
                float o = output[k];
                float t = target[k];
                if (o < epsilon_log) o = epsilon_log;
                if (o > 1.0f - epsilon_log) o = 1.0f - epsilon_log;
                loss -= (t * logf(o) + (1.0f - t) * logf(1.0f - o));
            }
        } else {
            for (uint32_t k = 0; k < output_size; k++) {
                float o = output[k];
                if (o < epsilon_log) o = epsilon_log;
                loss -= target[k] * logf(o);
            }
        }
    }

    return loss;
}
