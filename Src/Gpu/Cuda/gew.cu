/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/* gew.comp: the element-wise operators of graphs (Spingalett.GpuGraph.c). Output value i is OP of the operands at its
   position, each read through its strides over the output's axes (0 where it broadcasts), or at i (FLAG_A_FLAT ...)
   where its layout is the output's. Spec: OP, TYPES (the element types of a, b, c and y, four bits each:
   SPG_T_*), FLAGS. 256 threads, a grid-stride loop. */

#include "common.cuh"

DEVICE float load(uint64_t x, uint32_t i, uint32_t type) {
    switch (type) {
        case SPG_T_U8: case SPG_T_BOOL: return (float)((const uint8_t *)x)[i];
        case SPG_T_I8: return (float)((const signed char *)x)[i];
        case SPG_T_I32: return (float)((const int32_t *)x)[i];
        case SPG_T_I64: return (float)((const int64_t *)x)[i];
        case SPG_T_BF16: return from_bf16(((const uint16_t *)x)[i]);
        default: return F(x)[i];
    }
}

/* to 32 bits truncated toward zero, as Cast does (NaN 0, beyond the range its ends) */
DEVICE int32_t to_int32(float v) {
    return isnan_(v) ? 0 : v >= 2147483648.0f ? 2147483647 : v <= -2147483648.0f ? -2147483647 - 1 : (int32_t)v;
}

/* floats to integers truncated toward zero, as Cast does (NaN 0, beyond the range its ends; to 8 bits, the low
   bits of that), to booleans whether not 0 */
DEVICE void store(uint64_t y, uint32_t i, uint32_t type, float v) {
    switch (type) {
        case SPG_T_BOOL: ((uint8_t *)y)[i] = v != 0.0f; break;
        case SPG_T_U8: ((uint8_t *)y)[i] = (uint8_t)to_int32(v); break;
        case SPG_T_I8: ((signed char *)y)[i] = (signed char)to_int32(v); break;
        case SPG_T_I32: ((int32_t *)y)[i] = to_int32(v); break;
        case SPG_T_I64:
            ((int64_t *)y)[i] = isnan_(v) ? 0 : v >= 9.2233720368547758e18f ? 9223372036854775807ll
                              : v <= -9.2233720368547758e18f ? -9223372036854775807ll - 1 : (int64_t)v;
            break;
        default: F(y)[i] = v;
    }
}

/* sin and cos with the argument first reduced by multiples of 2 pi in three parts (Cody and Waite), so that the
   approximate instructions see it within [-pi, pi]: time embeddings take them of hundreds of radians */
DEVICE float reduce_2pi(float x) {
    const float k = __builtin_rintf(x * 0.15915494309189535f);
    float r = fma_(-k, 6.28318548202514648f, x);
    r = fma_(-k, -1.74845560e-7f, r);
    return fma_(-k, -2.12823051e-15f, r);
}
DEVICE float sin_(float x) { return __nvvm_sin_approx_f(reduce_2pi(x)); }
DEVICE float cos_(float x) { return __nvvm_cos_approx_f(reduce_2pi(x)); }

/* x to the power y: exact for the usual exponents, a negative base to an integer power signed as it should be */
DEVICE float pow_(float x, float y) {
    if (y == 2.0f) return x * x;
    if (y == 1.0f) return x;
    if (y == 0.0f) return 1.0f;
    if (y == 0.5f) return sqrt_(x);
    if (y == 3.0f) return x * x * x;
    if (y == -1.0f) return 1.0f / x;
    const bool integer = __builtin_rintf(y) == y;
    if (x < 0.0f && !integer) return __builtin_nanf("");
    const float a = abs_(x);
    if (a == 0.0f) return y > 0.0f ? (integer && x < 0.0f && __builtin_fmodf(y, 2.0f) != 0.0f ? -0.0f : 0.0f)
                                   : __builtin_inff();
    const float m = __nvvm_ex2_approx_f(y * __nvvm_lg2_approx_f(a));
    return x < 0.0f && __builtin_fmodf(y, 2.0f) != 0.0f ? -m : m;
}

DEVICE float softplus_(float x) { return x > 20.0f ? x : log_(1.0f + exp_(x)); }

DEVICE float unary(uint32_t op, float a, float alpha, float beta) {
    switch (op) {
        case SPG_GEW_COPY: return a;
        case SPG_GEW_NEG: return -a;
        case SPG_GEW_ABS: return abs_(a);
        case SPG_GEW_EXP: return exp_(a);
        case SPG_GEW_LOG: return log_(a);
        case SPG_GEW_SQRT: return sqrt_(a);
        case SPG_GEW_RECIPROCAL: return 1.0f / a;
        case SPG_GEW_SIGMOID: return 1.0f / (1.0f + exp_(-a));
        case SPG_GEW_TANH: return tanh_(a);
        case SPG_GEW_RELU: return a > 0.0f ? a : 0.0f;
        case SPG_GEW_LEAKY: return a >= 0.0f ? a : alpha * a;
        case SPG_GEW_ELU: return a >= 0.0f ? a : alpha * (exp_(a) - 1.0f);
        case SPG_GEW_SELU: return beta * (a > 0.0f ? a : alpha * (exp_(a) - 1.0f));
        case SPG_GEW_GELU: return 0.5f * a * (1.0f + erf_(a * 0.70710678118654752f));
        case SPG_GEW_GELU_TANH: return 0.5f * a * (1.0f + tanh_(0.79788456080286536f * (a + 0.044715f * a * a * a)));
        case SPG_GEW_SILU: return a / (1.0f + exp_(-alpha * a));
        case SPG_GEW_HARD_SIGMOID: return max_(0.0f, min_(1.0f, alpha * a + beta));
        case SPG_GEW_HARD_SWISH: return a * max_(0.0f, min_(1.0f, a / 6.0f + 0.5f));
        case SPG_GEW_SOFTPLUS: return softplus_(a);
        case SPG_GEW_SOFTSIGN: return a / (1.0f + abs_(a));
        case SPG_GEW_SIN: return sin_(a);
        case SPG_GEW_COS: return cos_(a);
        case SPG_GEW_ERF: return erf_(a);
        case SPG_GEW_FLOOR: return __builtin_floorf(a);
        case SPG_GEW_CEIL: return __builtin_ceilf(a);
        case SPG_GEW_ROUND: return __builtin_rintf(a);
        case SPG_GEW_SIGN: return (float)((a > 0.0f) - (a < 0.0f));
        case SPG_GEW_CLIP: { const float v = a < alpha ? alpha : a; return v > beta ? beta : v; }
        case SPG_GEW_NOT: return a == 0.0f ? 1.0f : 0.0f;
        case SPG_GEW_ISNAN: return isnan_(a) ? 1.0f : 0.0f;
        case SPG_GEW_ISINF: return isinf_(a) && ((a > 0.0f && alpha != 0.0f) || (a < 0.0f && beta != 0.0f)) ? 1.0f : 0.0f;
        case SPG_GEW_MISH: return a * tanh_(softplus_(a));
        case SPG_GEW_CELU: return max_(0.0f, a) + min_(0.0f, alpha * (exp_(a / alpha) - 1.0f));
        case SPG_GEW_THRESHOLDED_RELU: return a > alpha ? a : 0.0f;
        case SPG_GEW_TO_HALF: return from_half(to_half(a));
        case SPG_GEW_TO_BF16: return from_bf16(to_bf16(a));
        case SPG_GEW_AFFINE: return fma_(a, alpha, beta);
        default: return a;
    }
}

DEVICE float binary(uint32_t op, float a, float b) {
    switch (op) {
        case SPG_GEW_ADD: return a + b;
        case SPG_GEW_SUB: return a - b;
        case SPG_GEW_MUL: return a * b;
        case SPG_GEW_DIV: return a / b;
        case SPG_GEW_POW: return pow_(a, b);
        case SPG_GEW_MAX: return isnan_(a) || isnan_(b) ? __builtin_nanf("") : a > b ? a : b;
        case SPG_GEW_MIN: return isnan_(a) || isnan_(b) ? __builtin_nanf("") : a < b ? a : b;
        case SPG_GEW_PRELU: return a >= 0.0f ? a : a * b;
        case SPG_GEW_EQUAL: return a == b ? 1.0f : 0.0f;
        case SPG_GEW_LESS: return a < b ? 1.0f : 0.0f;
        case SPG_GEW_LESS_EQUAL: return a <= b ? 1.0f : 0.0f;
        case SPG_GEW_GREATER: return a > b ? 1.0f : 0.0f;
        case SPG_GEW_GREATER_EQUAL: return a >= b ? 1.0f : 0.0f;
        case SPG_GEW_AND: return a != 0.0f && b != 0.0f ? 1.0f : 0.0f;
        case SPG_GEW_OR: return a != 0.0f || b != 0.0f ? 1.0f : 0.0f;
        case SPG_GEW_XOR: return (a != 0.0f) != (b != 0.0f) ? 1.0f : 0.0f;
        case SPG_GEW_FMOD: return __builtin_fmodf(a, b);
        case SPG_GEW_MOD: {     /* the sign of the divisor, as Python's % */
            const float m = __builtin_fmodf(a, b);
            return m != 0.0f && ((m < 0.0f) != (b < 0.0f)) ? m + b : m;
        }
        default: return a;
    }
}

/* the operand's index for output value i at coordinates c: i itself, 0, or through its strides */
DEVICE uint32_t index_of(uint32_t i, const uint32_t c[SPG_GEW_AXES], const uint32_t *s, uint32_t mode) {
    if (mode == 1u) return i;
    if (mode == 2u) return 0u;
    uint32_t at = 0;
#pragma unroll
    for (uint32_t d = 0; d < SPG_GEW_AXES; d++) at += c[d] * s[d];
    return at;
}

KERNEL(gew, SpgGewPush) {
    const uint32_t op = spec.v[0], types = spec.v[1], flags = spec.v[2];
    const uint32_t ta = types & 15u, tb = (types >> 4) & 15u, tc = (types >> 8) & 15u, ty = (types >> 12) & 15u;
    const uint32_t ma = flags & 3u, mb = (flags >> 2) & 3u, mc = (flags >> 4) & 3u;
    const bool strided = ma == 0u || (op >= SPG_GEW_ADD && mb == 0u) || (op >= SPG_GEW_WHERE && mc == 0u);
    const uint32_t stride = blocks_x() * threads_x();
    for (uint32_t i = global_x(); i < p.total; i += stride) {
        uint32_t c[SPG_GEW_AXES] = {0, 0, 0, 0};
        if (strided) {
            uint32_t r = i;
#pragma unroll
            for (uint32_t d = SPG_GEW_AXES; d-- > 1;) {
                c[d] = r % p.dims[d];
                r /= p.dims[d];
            }
            c[0] = r;
        }
        const float a = load(p.a, index_of(i, c, p.sa, ma), ta);
        float v;
        if (op < SPG_GEW_ADD) {
            v = unary(op, a, p.alpha, p.beta);
        } else if (op < SPG_GEW_WHERE) {
            v = binary(op, a, load(p.b, index_of(i, c, p.sb, mb), tb));
        } else {
            /* (Where reads only the side it takes) */
            v = a != 0.0f ? load(p.b, index_of(i, c, p.sb, mb), tb) : load(p.c, index_of(i, c, p.sc, mc), tc);
        }
        store(p.y, i, ty, v);
    }
}
