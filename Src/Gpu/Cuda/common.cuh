/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/*
 * Shared by the CUDA kernels of Src/Gpu/Cuda, which are compiled to PTX by clang without the CUDA
 * toolkit (cmake/Cuda.cmake: -nocudainc -nocudalib): the qualifiers and thread indices the toolkit's
 * headers would give, the math functions as single instructions, buffers as addresses, activations
 * and their derivatives as the CPU computes them (Spingalett.Activation.c) and as common.glsl does,
 * bfloat16, the step header and the hashes of dropout and augmentation.
 *
 * Every kernel takes the push constants of its Vulkan counterpart (Spingalett.GpuKernels.h), then the
 * specialization constants, Spec, which select modes at run time here (uniform branches).
 * Compiled with -ffp-contract=off: a multiply-add is fused only where a kernel says fma().
 */

#pragma once

#define __global__ __attribute__((global))
#define __device__ __attribute__((device))
#define __shared__ __attribute__((shared))
#define DEVICE static __device__ __attribute__((always_inline)) inline
#define MEMBER __device__ __attribute__((always_inline)) inline     /* (methods: kept in registers) */
#define __launch_bounds__(threads) __attribute__((launch_bounds(threads)))

typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef int int32_t;
typedef unsigned long long uint64_t;
typedef long long int64_t;

#include "../Spingalett.GpuPush.h"     /* the kernels' parameters */

/* The specialization constants of a dispatch (Spingalett.Device.h, SPG_SPEC_MAX). */
struct Spec {
    uint32_t v[16];
};

#define KERNEL(name, Push) extern "C" __global__ void spg_##name(const Push p, __attribute__((unused)) const Spec spec)

DEVICE uint32_t thread_x() { return __nvvm_read_ptx_sreg_tid_x(); }
DEVICE uint32_t block_x() { return __nvvm_read_ptx_sreg_ctaid_x(); }
DEVICE uint32_t block_y() { return __nvvm_read_ptx_sreg_ctaid_y(); }
DEVICE uint32_t block_z() { return __nvvm_read_ptx_sreg_ctaid_z(); }
DEVICE uint32_t blocks_x() { return __nvvm_read_ptx_sreg_nctaid_x(); }
DEVICE uint32_t threads_x() { return __nvvm_read_ptx_sreg_ntid_x(); }
DEVICE uint32_t global_x() { return block_x() * threads_x() + thread_x(); }
DEVICE void barrier() { __syncthreads(); }

/* math as the GPU computes it for GLSL: exp and log through exp2 and log2, sqrt rounded */
DEVICE float exp_(float x) { return __nvvm_ex2_approx_f(x * 1.442695040888963f); }
DEVICE float log_(float x) { return __nvvm_lg2_approx_f(x) * 0.6931471805599453f; }
DEVICE float sqrt_(float x) { return __nvvm_sqrt_rn_f(x); }
DEVICE float fma_(float a, float b, float c) { return __builtin_fmaf(a, b, c); }
DEVICE float abs_(float x) { return __builtin_fabsf(x); }
DEVICE float max_(float a, float b) { return a > b ? a : b; }
DEVICE float min_(float a, float b) { return a < b ? a : b; }
DEVICE uint32_t umin(uint32_t a, uint32_t b) { return a < b ? a : b; }
DEVICE uint32_t float_bits(float f) { return __builtin_bit_cast(uint32_t, f); }
DEVICE float bits_float(uint32_t u) { return __builtin_bit_cast(float, u); }
DEVICE bool isnan_(float f) { return f != f; }
DEVICE bool isinf_(float f) { return abs_(f) == __builtin_inff(); }

/* buffers: addresses of floats, uints, bfloat16 */
DEVICE float *F(uint64_t a) { return (float *)a; }
DEVICE uint32_t *U(uint64_t a) { return (uint32_t *)a; }
DEVICE uint16_t *H(uint64_t a) { return (uint16_t *)a; }

struct float4_ {
    float x, y, z, w;
} __attribute__((aligned(16)));
struct uint2_ {
    uint32_t x, y;
} __attribute__((aligned(8)));

/* bfloat16: from_bf16, and to the nearest, ties to even (NaN stays NaN), as the host rounds */
DEVICE float from_bf16(uint32_t h) { return bits_float(h << 16); }
DEVICE uint32_t to_bf16(float f) {
    uint32_t u = float_bits(f);
    if (isnan_(f)) return (u >> 16) | 0x40u;
    return (u + 0x7FFFu + ((u >> 16) & 1u)) >> 16;
}

/* Value i of buffer x, whose push-constant word is w: bfloat16 where bit w of the kernel's HALF says so
   (half.glsl), else a float. */
DEVICE float ld(uint64_t x, uint32_t i, uint32_t w, uint32_t half) {
    if ((half >> w) & 1u) return from_bf16(H(x)[i]);
    return F(x)[i];
}
DEVICE void st(uint64_t x, uint32_t i, uint32_t w, uint32_t half, float v) {
    if ((half >> w) & 1u) {
        H(x)[i] = (uint16_t)to_bf16(v);
        return;
    }
    F(x)[i] = v;
}
/* four values from i (a multiple of four) on */
DEVICE float4_ ld4(uint64_t x, uint32_t i, uint32_t w, uint32_t half) {
    if ((half >> w) & 1u) {
        const uint2_ u = ((const uint2_ *)x)[i >> 2];
        return float4_{bits_float(u.x << 16), bits_float(u.x & 0xFFFF0000u), bits_float(u.y << 16),
                       bits_float(u.y & 0xFFFF0000u)};
    }
    return ((const float4_ *)x)[i >> 2];
}
DEVICE void st4(uint64_t x, uint32_t i, uint32_t w, uint32_t half, float4_ v) {
    if ((half >> w) & 1u) {
        ((uint2_ *)x)[i >> 2] = uint2_{to_bf16(v.x) | to_bf16(v.y) << 16, to_bf16(v.z) | to_bf16(v.w) << 16};
        return;
    }
    ((float4_ *)x)[i >> 2] = v;
}

DEVICE float4_ add4(float4_ a, float4_ b) { return float4_{a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w}; }
DEVICE float4_ mul4(float4_ a, float4_ b) { return float4_{a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w}; }
DEVICE float4_ scale4(float4_ a, float s) { return float4_{a.x * s, a.y * s, a.z * s, a.w * s}; }
DEVICE float4_ fma4(float4_ a, float4_ b, float4_ c) {
    return float4_{fma_(a.x, b.x, c.x), fma_(a.y, b.y, c.y), fma_(a.z, b.z, c.z), fma_(a.w, b.w, c.w)};
}
DEVICE float get4(const float4_ &v, uint32_t k) { return k == 0 ? v.x : k == 1 ? v.y : k == 2 ? v.z : v.w; }
DEVICE void set4(float4_ &v, uint32_t k, float f) {
    if (k == 0) v.x = f;
    else if (k == 1) v.y = f;
    else if (k == 2) v.z = f;
    else v.w = f;
}

/* ActivationFunction */
#define ACT_NONE        0u
#define ACT_SIGMOID     1u
#define ACT_RELU        2u
#define ACT_TANH        3u
#define ACT_LEAKY_RELU  4u
#define ACT_FOO52       5u
#define ACT_SOFTMAX     6u

DEVICE float activate(float x, uint32_t act) {
    switch (act) {
        case ACT_RELU:          return x > 0.0f ? x : 0.0f;
        case ACT_LEAKY_RELU:    return x > 0.0f ? x : 0.01f * x;
        case ACT_SIGMOID:       return 1.0f / (1.0f + exp_(-x));
        case ACT_TANH: {        /* as common.glsl: an odd polynomial below |x| = 0.3, through exp above */
            float a = abs_(x), t;
            if (a < 0.3f) {
                float a2 = a * a, q = fma_(-8.86323552990219656e-3f, a2, 2.18694885361552028e-2f);
                q = fma_(q, a2, -5.39682539682539683e-2f);
                q = fma_(q, a2, 1.33333333333333333e-1f);
                q = fma_(q, a2, -3.33333333333333333e-1f);
                t = fma_(a * a2, q, a);
            } else {
                float e = exp_(-2.0f * a);
                t = (1.0f - e) / (1.0f + e);
            }
            return x < 0.0f ? -t : t;
        }
        case ACT_FOO52:         return x > 1.0f ? 1.0f + 0.01f * (x - 1.0f) : (x < 0.0f ? 0.01f * x : x);
        default:                return x;
    }
}

/* The derivative of the activation, from its output y. */
DEVICE float derivative(float y, uint32_t act) {
    switch (act) {
        case ACT_RELU:          return y > 0.0f ? 1.0f : 0.0f;
        case ACT_LEAKY_RELU:    return y > 0.0f ? 1.0f : 0.01f;
        case ACT_SIGMOID:
        case ACT_SOFTMAX:       return y * (1.0f - y);
        case ACT_TANH:          return 1.0f - y * y;
        case ACT_FOO52:         return (y > 1.0f || y < 0.0f) ? 0.01f : 1.0f;
        default:                return 1.0f;
    }
}

DEVICE float4_ activate4(float4_ v, uint32_t act) {
    return float4_{activate(v.x, act), activate(v.y, act), activate(v.z, act), activate(v.w, act)};
}

/* Step header (SpgStepHeader in Spingalett.Gpu.c), as common.glsl */
#define STEP_LR         0u
#define STEP_M_FACTOR   1u
#define STEP_V_FACTOR   2u
#define STEP_SEED_LO    3u
#define STEP_SEED_HI    4u
#define STEP_STEP_LO    5u
#define STEP_STEP_HI    6u
#define STEP_POSITION   7u
#define STEP_CLIP       8u
#define STEP_GRAD_SCALE 9u
#define STEP_INPUTS     10u
#define STEP_TARGETS    12u
#define STEP_AUGMENT_LO 14u
#define STEP_AUGMENT_HI 15u
#define STEP_AUGMENT    16u
#define STEP_KEEP       17u
#define STEP_SHARE      18u

DEVICE uint64_t header64(const uint32_t *h, uint32_t lo) { return (uint64_t)h[lo] | (uint64_t)h[lo + 1] << 32; }

/* splitmix64's finalizer (mix64 of common.glsl) */
DEVICE uint64_t mix64(uint64_t z) {
    z ^= z >> 30;
    z *= 0xBF58476D1CE4E5B9ull;
    z ^= z >> 27;
    z *= 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
