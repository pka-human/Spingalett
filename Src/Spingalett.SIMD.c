#include "Spingalett.Private.h"
#include <math.h>
#include <float.h>
#include <string.h>

#if defined(__AVX__)
#include <immintrin.h>

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

void apply_activation_batch(float *data, uint32_t size, ActivationFunction act) {
    if (act == ACT_SOFTMAX) {
        apply_softmax(data, size);
        return;
    }
    apply_activation_bulk(data, (uint64_t)size, act);
}

void apply_activation_bulk(float *data, uint64_t total, ActivationFunction act) {
    if (act == ACT_NONE || act >= ACT_COUNT || act == ACT_SOFTMAX)
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
        for (; i < total; i++)
            if (data[i] < 0.0f) data[i] = 0.0f;
        return;
    }

    if (act == ACT_LEAKY_RELU) {
        __m256 zero = _mm256_setzero_ps();
        __m256 alpha = _mm256_set1_ps(0.01f);
        for (; i + 8u <= total; i += 8u) {
            __m256 v = _mm256_loadu_ps(data + i);
            __m256 mask = _mm256_cmp_ps(v, zero, _CMP_GT_OQ);
            _mm256_storeu_ps(data + i, _mm256_blendv_ps(_mm256_mul_ps(v, alpha), v, mask));
        }
        for (; i < total; i++)
            data[i] = data[i] > 0.0f ? data[i] : 0.01f * data[i];
        return;
    }

    if (act == ACT_FOO52) {
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
        for (; i < total; i++)
            data[i] = activate(data[i], ACT_FOO52);
        return;
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
        for (; i < total; i++)
            deriv[i] *= (act_data[i] > 0.0f) ? 1.0f : 0.0f;
        return;
    }

    if (act == ACT_LEAKY_RELU) {
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
        for (; i < total; i++)
            deriv[i] *= (act_data[i] > 0.0f) ? 1.0f : 0.01f;
        return;
    }

    if (act == ACT_SIGMOID || act == ACT_SOFTMAX) {
        __m256 one = _mm256_set1_ps(1.0f);
        for (; i + 8u <= total; i += 8u) {
            __m256 a = _mm256_loadu_ps(act_data + i);
            __m256 d = _mm256_loadu_ps(deriv + i);
            _mm256_storeu_ps(deriv + i, _mm256_mul_ps(d, _mm256_mul_ps(a, _mm256_sub_ps(one, a))));
        }
        for (; i < total; i++)
            deriv[i] *= act_data[i] * (1.0f - act_data[i]);
        return;
    }

    if (act == ACT_TANH) {
        __m256 one = _mm256_set1_ps(1.0f);
        for (; i + 8u <= total; i += 8u) {
            __m256 a = _mm256_loadu_ps(act_data + i);
            __m256 d = _mm256_loadu_ps(deriv + i);
#if defined(__FMA__)
            _mm256_storeu_ps(deriv + i, _mm256_mul_ps(d, _mm256_fnmadd_ps(a, a, one)));
#else
            _mm256_storeu_ps(deriv + i, _mm256_mul_ps(d, _mm256_sub_ps(one, _mm256_mul_ps(a, a))));
#endif
        }
        for (; i < total; i++)
            deriv[i] *= 1.0f - act_data[i] * act_data[i];
        return;
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
        __m256 a0 = _mm256_loadu_ps(a + i);
        __m256 b0 = _mm256_loadu_ps(b + i);
        __m256 a1 = _mm256_loadu_ps(a + i + 8u);
        __m256 b1 = _mm256_loadu_ps(b + i + 8u);
        __m256 a2 = _mm256_loadu_ps(a + i + 16u);
        __m256 b2 = _mm256_loadu_ps(b + i + 16u);
        __m256 a3 = _mm256_loadu_ps(a + i + 24u);
        __m256 b3 = _mm256_loadu_ps(b + i + 24u);
#if defined(__FMA__)
        sum0 = _mm256_fmadd_ps(a0, b0, sum0);
        sum1 = _mm256_fmadd_ps(a1, b1, sum1);
        sum2 = _mm256_fmadd_ps(a2, b2, sum2);
        sum3 = _mm256_fmadd_ps(a3, b3, sum3);
#else
        sum0 = _mm256_add_ps(sum0, _mm256_mul_ps(a0, b0));
        sum1 = _mm256_add_ps(sum1, _mm256_mul_ps(a1, b1));
        sum2 = _mm256_add_ps(sum2, _mm256_mul_ps(a2, b2));
        sum3 = _mm256_add_ps(sum3, _mm256_mul_ps(a3, b3));
#endif
    }
    for (; i + 8u <= n; i += 8u) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
#if defined(__FMA__)
        sum0 = _mm256_fmadd_ps(va, vb, sum0);
#else
        sum0 = _mm256_add_ps(sum0, _mm256_mul_ps(va, vb));
#endif
    }
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

void spingalett_vec_scale(float *data, uint64_t n, float scale) {
#if defined(__AVX__)
    __m256 vs = _mm256_set1_ps(scale);
    uint64_t i = 0;
    for (; i + 8u <= n; i += 8u) {
        __m256 v = _mm256_loadu_ps(data + i);
        _mm256_storeu_ps(data + i, _mm256_mul_ps(v, vs));
    }
    for (; i < n; i++)
        data[i] *= scale;
#else
    for (uint64_t i = 0; i < n; i++)
        data[i] *= scale;
#endif
}

void spingalett_vec_axpy(float *y, const float *x, uint64_t n, float alpha) {
#if defined(__AVX__)
    __m256 va = _mm256_set1_ps(alpha);
    uint64_t i = 0;
    for (; i + 8u <= n; i += 8u) {
        __m256 vy = _mm256_loadu_ps(y + i);
        __m256 vx = _mm256_loadu_ps(x + i);
#if defined(__FMA__)
        _mm256_storeu_ps(y + i, _mm256_fmadd_ps(va, vx, vy));
#else
        _mm256_storeu_ps(y + i, _mm256_add_ps(vy, _mm256_mul_ps(va, vx)));
#endif
    }
    for (; i < n; i++)
        y[i] += alpha * x[i];
#else
    for (uint64_t i = 0; i < n; i++)
        y[i] += alpha * x[i];
#endif
}

void spingalett_adam_update_avx(float *restrict W, float *restrict mW, float *restrict vW, const float *restrict gW,
                                uint64_t n, float lr, float beta1, float beta2,
                                float one_minus_b1, float one_minus_b2,
                                float m_factor, float v_factor, float epsilon) {
#if defined(__AVX__)
    __m256 v_lr = _mm256_set1_ps(lr);
    __m256 v_mf = _mm256_set1_ps(m_factor);
    __m256 v_vf = _mm256_set1_ps(v_factor);
    __m256 v_eps = _mm256_set1_ps(epsilon);
    __m256 v_b1 = _mm256_set1_ps(beta1);
    __m256 v_b2 = _mm256_set1_ps(beta2);
    __m256 v_1mb1 = _mm256_set1_ps(one_minus_b1);
    __m256 v_1mb2 = _mm256_set1_ps(one_minus_b2);
    uint64_t i = 0;
    for (; i + 8u <= n; i += 8u) {
        __m256 g = _mm256_loadu_ps(gW + i);
        __m256 m = _mm256_loadu_ps(mW + i);
        __m256 v = _mm256_loadu_ps(vW + i);
        __m256 w = _mm256_loadu_ps(W + i);
#if defined(__FMA__)
        m = _mm256_fmadd_ps(v_b1, m, _mm256_mul_ps(v_1mb1, g));
        v = _mm256_fmadd_ps(v_b2, v, _mm256_mul_ps(v_1mb2, _mm256_mul_ps(g, g)));
#else
        m = _mm256_add_ps(_mm256_mul_ps(v_b1, m), _mm256_mul_ps(v_1mb1, g));
        v = _mm256_add_ps(_mm256_mul_ps(v_b2, v), _mm256_mul_ps(v_1mb2, _mm256_mul_ps(g, g)));
#endif
        __m256 m_hat = _mm256_mul_ps(m, v_mf);
        __m256 v_hat = _mm256_add_ps(_mm256_mul_ps(v, v_vf), v_eps);
        __m256 step = _mm256_div_ps(m_hat, _mm256_sqrt_ps(v_hat));
        _mm256_storeu_ps(mW + i, m);
        _mm256_storeu_ps(vW + i, v);
        _mm256_storeu_ps(W + i, _mm256_sub_ps(w, _mm256_mul_ps(v_lr, step)));
    }
    for (; i < n; i++) {
        mW[i] = beta1 * mW[i] + one_minus_b1 * gW[i];
        vW[i] = beta2 * vW[i] + one_minus_b2 * (gW[i] * gW[i]);
        float m_hat = mW[i] * m_factor;
        float v_hat = vW[i] * v_factor + epsilon;
        W[i] -= lr * (m_hat / sqrtf(v_hat));
    }
#else
    for (uint64_t i = 0; i < n; i++) {
        mW[i] = beta1 * mW[i] + one_minus_b1 * gW[i];
        vW[i] = beta2 * vW[i] + one_minus_b2 * (gW[i] * gW[i]);
        float m_hat = mW[i] * m_factor;
        float v_hat = vW[i] * v_factor + epsilon;
        W[i] -= lr * (m_hat / sqrtf(v_hat));
    }
#endif
}

void spingalett_adamw_update_avx(float *restrict W, float *restrict mW, float *restrict vW, const float *restrict gW,
                                  uint64_t n, float lr, float beta1, float beta2,
                                  float one_minus_b1, float one_minus_b2,
                                  float m_factor, float v_factor, float epsilon,
                                  float wd_factor) {
#if defined(__AVX__)
    __m256 v_lr = _mm256_set1_ps(lr);
    __m256 v_mf = _mm256_set1_ps(m_factor);
    __m256 v_vf = _mm256_set1_ps(v_factor);
    __m256 v_eps = _mm256_set1_ps(epsilon);
    __m256 v_wd = _mm256_set1_ps(wd_factor);
    __m256 v_b1 = _mm256_set1_ps(beta1);
    __m256 v_b2 = _mm256_set1_ps(beta2);
    __m256 v_1mb1 = _mm256_set1_ps(one_minus_b1);
    __m256 v_1mb2 = _mm256_set1_ps(one_minus_b2);
    uint64_t i = 0;
    for (; i + 8u <= n; i += 8u) {
        __m256 g = _mm256_loadu_ps(gW + i);
        __m256 m = _mm256_loadu_ps(mW + i);
        __m256 v = _mm256_loadu_ps(vW + i);
        __m256 w = _mm256_loadu_ps(W + i);
#if defined(__FMA__)
        m = _mm256_fmadd_ps(v_b1, m, _mm256_mul_ps(v_1mb1, g));
        v = _mm256_fmadd_ps(v_b2, v, _mm256_mul_ps(v_1mb2, _mm256_mul_ps(g, g)));
#else
        m = _mm256_add_ps(_mm256_mul_ps(v_b1, m), _mm256_mul_ps(v_1mb1, g));
        v = _mm256_add_ps(_mm256_mul_ps(v_b2, v), _mm256_mul_ps(v_1mb2, _mm256_mul_ps(g, g)));
#endif
        __m256 m_hat = _mm256_mul_ps(m, v_mf);
        __m256 v_hat = _mm256_add_ps(_mm256_mul_ps(v, v_vf), v_eps);
        __m256 step = _mm256_div_ps(m_hat, _mm256_sqrt_ps(v_hat));
        _mm256_storeu_ps(mW + i, m);
        _mm256_storeu_ps(vW + i, v);
        w = _mm256_mul_ps(w, v_wd);
        _mm256_storeu_ps(W + i, _mm256_sub_ps(w, _mm256_mul_ps(v_lr, step)));
    }
    for (; i < n; i++) {
        mW[i] = beta1 * mW[i] + one_minus_b1 * gW[i];
        vW[i] = beta2 * vW[i] + one_minus_b2 * (gW[i] * gW[i]);
        float m_hat = mW[i] * m_factor;
        float v_hat = vW[i] * v_factor + epsilon;
        W[i] = W[i] * wd_factor - lr * (m_hat / sqrtf(v_hat));
    }
#else
    for (uint64_t i = 0; i < n; i++) {
        mW[i] = beta1 * mW[i] + one_minus_b1 * gW[i];
        vW[i] = beta2 * vW[i] + one_minus_b2 * (gW[i] * gW[i]);
        float m_hat = mW[i] * m_factor;
        float v_hat = vW[i] * v_factor + epsilon;
        W[i] = W[i] * wd_factor - lr * (m_hat / sqrtf(v_hat));
    }
#endif
}

void spingalett_sgd_update_avx(float *restrict W, const float *restrict gW, uint64_t n, float lr) {
#if defined(__AVX__)
    __m256 v_lr = _mm256_set1_ps(lr);
    uint64_t i = 0;
    for (; i + 8u <= n; i += 8u) {
        __m256 w = _mm256_loadu_ps(W + i);
        __m256 g = _mm256_loadu_ps(gW + i);
        _mm256_storeu_ps(W + i, _mm256_sub_ps(w, _mm256_mul_ps(v_lr, g)));
    }
    for (; i < n; i++)
        W[i] -= lr * gW[i];
#else
    for (uint64_t i = 0; i < n; i++)
        W[i] -= lr * gW[i];
#endif
}

void spingalett_sgd_decay_update_avx(float *restrict W, const float *restrict gW, uint64_t n, float lr, float decay) {
#if defined(__AVX__)
    __m256 v_lr = _mm256_set1_ps(lr);
    __m256 v_decay = _mm256_set1_ps(decay);
    uint64_t i = 0;
    for (; i + 8u <= n; i += 8u) {
        __m256 w = _mm256_loadu_ps(W + i);
        __m256 g = _mm256_loadu_ps(gW + i);
#if defined(__FMA__)
        __m256 g_decay = _mm256_fmadd_ps(v_decay, w, g);
#else
        __m256 g_decay = _mm256_add_ps(g, _mm256_mul_ps(v_decay, w));
#endif
        _mm256_storeu_ps(W + i, _mm256_sub_ps(w, _mm256_mul_ps(v_lr, g_decay)));
    }
    for (; i < n; i++)
        W[i] -= lr * (gW[i] + decay * W[i]);
#else
    for (uint64_t i = 0; i < n; i++)
        W[i] -= lr * (gW[i] + decay * W[i]);
#endif
}

void spingalett_momentum_update_avx(float *restrict W, float *restrict mW, const float *restrict gW, uint64_t n,
                                    float lr, float momentum, float decay) {
#if defined(__AVX__)
    __m256 v_lr = _mm256_set1_ps(lr);
    __m256 v_mom = _mm256_set1_ps(momentum);
    __m256 v_decay = _mm256_set1_ps(decay);
    uint64_t i = 0;
    for (; i + 8u <= n; i += 8u) {
        __m256 w = _mm256_loadu_ps(W + i);
        __m256 m = _mm256_loadu_ps(mW + i);
        __m256 g = _mm256_loadu_ps(gW + i);
#if defined(__FMA__)
        m = _mm256_fmadd_ps(v_mom, m, _mm256_fmadd_ps(v_decay, w, g));
#else
        m = _mm256_add_ps(_mm256_mul_ps(v_mom, m), _mm256_add_ps(g, _mm256_mul_ps(v_decay, w)));
#endif
        _mm256_storeu_ps(mW + i, m);
        _mm256_storeu_ps(W + i, _mm256_sub_ps(w, _mm256_mul_ps(v_lr, m)));
    }
    for (; i < n; i++) {
        mW[i] = momentum * mW[i] + decay * W[i] + gW[i];
        W[i] -= lr * mW[i];
    }
#else
    for (uint64_t i = 0; i < n; i++) {
        mW[i] = momentum * mW[i] + decay * W[i] + gW[i];
        W[i] -= lr * mW[i];
    }
#endif
}

void spingalett_rmsprop_update_avx(float *restrict W, float *restrict vW, const float *restrict gW, uint64_t n,
                                    float lr, float beta2, float one_minus_b2, float epsilon, float decay) {
#if defined(__AVX__)
    __m256 v_lr = _mm256_set1_ps(lr);
    __m256 v_b2 = _mm256_set1_ps(beta2);
    __m256 v_1mb2 = _mm256_set1_ps(one_minus_b2);
    __m256 v_eps = _mm256_set1_ps(epsilon);
    __m256 v_decay = _mm256_set1_ps(decay);
    uint64_t i = 0;
    for (; i + 8u <= n; i += 8u) {
        __m256 w = _mm256_loadu_ps(W + i);
        __m256 v = _mm256_loadu_ps(vW + i);
        __m256 g = _mm256_loadu_ps(gW + i);
#if defined(__FMA__)
        v = _mm256_fmadd_ps(v_b2, v, _mm256_mul_ps(v_1mb2, _mm256_mul_ps(g, g)));
#else
        v = _mm256_add_ps(_mm256_mul_ps(v_b2, v), _mm256_mul_ps(v_1mb2, _mm256_mul_ps(g, g)));
#endif
        __m256 v_hat = _mm256_add_ps(v, v_eps);
        __m256 step = _mm256_div_ps(g, _mm256_sqrt_ps(v_hat));
        _mm256_storeu_ps(vW + i, v);
#if defined(__FMA__)
        _mm256_storeu_ps(W + i, _mm256_sub_ps(w, _mm256_mul_ps(v_lr, _mm256_fmadd_ps(v_decay, w, step))));
#else
        _mm256_storeu_ps(W + i, _mm256_sub_ps(w, _mm256_mul_ps(v_lr, _mm256_add_ps(step, _mm256_mul_ps(v_decay, w)))));
#endif
    }
    for (; i < n; i++) {
        vW[i] = beta2 * vW[i] + one_minus_b2 * (gW[i] * gW[i]);
        float v_hat = vW[i] + epsilon;
        W[i] -= lr * (decay * W[i] + gW[i] / sqrtf(v_hat));
    }
#else
    for (uint64_t i = 0; i < n; i++) {
        vW[i] = beta2 * vW[i] + one_minus_b2 * (gW[i] * gW[i]);
        float v_hat = vW[i] + epsilon;
        W[i] -= lr * (decay * W[i] + gW[i] / sqrtf(v_hat));
    }
#endif
}

float spingalett_clip_grad_norm(NeuralNetwork *net, float max_norm) {
    if (max_norm <= 0.0f) return 0.0f;

    float total_sq = 0.0f;
    for (uint32_t l = 0; l + 1 < net->layers; l++) {
        uint64_t wcount = (uint64_t)net->topology[l] * (uint64_t)net->topology[l + 1];
        float *gW = SPINGALETT_GRAD_W_MTX_PTR(net, l);
        total_sq += spingalett_dot_product(gW, gW, wcount);

        uint32_t bcount = net->topology[l + 1];
        float *gB = net->grad_biases + net->bias_offsets[l];
        total_sq += spingalett_dot_product(gB, gB, (uint64_t)bcount);
    }

    float grad_norm = sqrtf(total_sq);
    if (grad_norm > max_norm) {
        float scale = max_norm / grad_norm;
        for (uint32_t l = 0; l + 1 < net->layers; l++) {
            uint64_t wcount = (uint64_t)net->topology[l] * (uint64_t)net->topology[l + 1];
            spingalett_vec_scale(SPINGALETT_GRAD_W_MTX_PTR(net, l), wcount, scale);
            spingalett_vec_scale(net->grad_biases + net->bias_offsets[l], (uint64_t)net->topology[l + 1], scale);
        }
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
