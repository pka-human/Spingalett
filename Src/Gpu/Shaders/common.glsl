/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/* Shared by every kernel: buffers as device addresses, activations and their derivatives as the
   CPU computes them (Spingalett.Activation.c), and the step header the host writes per step. */

#extension GL_EXT_buffer_reference : require
#extension GL_EXT_buffer_reference2 : require

layout(buffer_reference, std430, buffer_reference_align = 4) buffer F32 { float v[]; };
layout(buffer_reference, std430, buffer_reference_align = 16) buffer F32x4 { vec4 v[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) buffer U32 { uint v[]; };

#ifdef SPG_HALF
/* bfloat16 values in memory (kernels built with SPG_HALF, which enable 16-bit storage) */
layout(buffer_reference, std430, buffer_reference_align = 2) buffer BF16 { uint16_t v[]; };

float from_bf16(uint h) { return uintBitsToFloat(h << 16); }
/* to the nearest bfloat16, ties to even (NaN stays NaN), as the host rounds */
uint to_bf16(float f) {
    uint u = floatBitsToUint(f);
    if (isnan(f)) return (u >> 16) | 0x40u;
    return (u + 0x7FFFu + ((u >> 16) & 1u)) >> 16;
}
#endif

/* ActivationFunction */
#define ACT_NONE        0u
#define ACT_SIGMOID     1u
#define ACT_RELU        2u
#define ACT_TANH        3u
#define ACT_LEAKY_RELU  4u
#define ACT_FOO52       5u
#define ACT_SOFTMAX     6u
#define ACT_GELU        7u      /* these three: functions of their input, whose derivative needs it */
#define ACT_GELU_TANH   8u
#define ACT_SILU        9u

/* tanh() of some drivers overflows to NaN for large arguments: (1 - e) / (1 + e), e = exp(-2|x|),
   which loses relative precision below |x| = 0.3, where an odd Taylor polynomial takes over, as the
   CPU's vector kernel has (Spingalett.SIMD.c) */
float tanh_(float x) {
    float a = abs(x), t;
    if (a < 0.3) {
        float a2 = a * a, q = fma(-8.86323552990219656e-3, a2, 2.18694885361552028e-2);
        q = fma(q, a2, -5.39682539682539683e-2);
        q = fma(q, a2, 1.33333333333333333e-1);
        q = fma(q, a2, -3.33333333333333333e-1);
        t = fma(a * a2, q, a);
    } else {
        float e = exp(-2.0 * a);
        t = (1.0 - e) / (1.0 + e);
    }
    return x < 0.0 ? -t : t;
}

/* erf (GLSL has none): Abramowitz and Stegun 7.1.26, within 1.5e-7 of it */
float erf_(float x) {
    float a = abs(x), t = 1.0 / fma(0.3275911, a, 1.0);
    float q = fma(fma(fma(fma(1.061405429, t, -1.453152027), t, 1.421413741), t, -0.284496736), t, 0.254829592);
    float r = 1.0 - q * t * exp(-a * a);
    return x < 0.0 ? -r : r;
}

float activate(float x, uint act) {
    switch (act) {
        case ACT_RELU:          return x > 0.0 ? x : 0.0;
        case ACT_LEAKY_RELU:    return x > 0.0 ? x : 0.01 * x;
        case ACT_SIGMOID:       return 1.0 / (1.0 + exp(-x));
        case ACT_TANH:          return tanh_(x);
        case ACT_FOO52:         return x > 1.0 ? 1.0 + 0.01 * (x - 1.0) : (x < 0.0 ? 0.01 * x : x);
        case ACT_GELU:          return 0.5 * x * (1.0 + erf_(x * 0.70710678118654752));
        case ACT_GELU_TANH:     return 0.5 * x * (1.0 + tanh_(0.79788456080286536 * (x + 0.044715 * x * x * x)));
        case ACT_SILU:          return x / (1.0 + exp(-x));
        default:                return x;
    }
}

/* The derivative of the activation, from its output y, or for GELU and SiLU from its input. */
float derivative(float y, uint act) {
    switch (act) {
        case ACT_RELU:          return y > 0.0 ? 1.0 : 0.0;
        case ACT_LEAKY_RELU:    return y > 0.0 ? 1.0 : 0.01;
        case ACT_SIGMOID:
        case ACT_SOFTMAX:       return y * (1.0 - y);
        case ACT_TANH:          return 1.0 - y * y;
        case ACT_FOO52:         return (y > 1.0 || y < 0.0) ? 0.01 : 1.0;
        case ACT_GELU:          return 0.5 * (1.0 + erf_(y * 0.70710678118654752)) + y * 0.39894228040143268 * exp(-0.5 * y * y);
        case ACT_GELU_TANH: {
            float k = 0.79788456080286536, t = tanh_(k * (y + 0.044715 * y * y * y));
            return 0.5 * (1.0 + t) + 0.5 * y * (1.0 - t * t) * k * (1.0 + 3.0 * 0.044715 * y * y);
        }
        case ACT_SILU: {
            float s = 1.0 / (1.0 + exp(-y));
            return s * (1.0 + y * (1.0 - s));
        }
        default:                return 1.0;
    }
}

/* Step header (SpgStepHeader in Spingalett.Gpu.c): what changes from one optimizer step to the next. */
#define STEP_LR         0u      /* float: learning rate */
#define STEP_M_FACTOR   1u      /* float: Adam's bias corrections */
#define STEP_V_FACTOR   2u
#define STEP_SEED_LO    3u      /* uint: dropout seed */
#define STEP_SEED_HI    4u
#define STEP_STEP_LO    5u      /* uint: optimizer step of the samples (dropout) */
#define STEP_STEP_HI    6u
#define STEP_POSITION   7u      /* uint: position of the chunk's first sample in its step */
#define STEP_CLIP       8u      /* float: gradient scale of norm clipping (1: none) */
#define STEP_GRAD_SCALE 9u      /* float: the step API's gradient scale (one over its samples) */
#define STEP_INPUTS     10u     /* uvec2: the address of the data set on the GPU the inputs are gathered from */
#define STEP_TARGETS    12u     /* uvec2: and the targets (rows.comp) */
#define STEP_AUGMENT_LO 14u     /* uint: augmentation seed of the training run */
#define STEP_AUGMENT_HI 15u
#define STEP_AUGMENT    16u     /* uint: augment_shift, with bit 31 for augment_flip (0: none) */
#define STEP_KEEP       17u     /* float: label smoothing, t' = keep t + share (0: none) */
#define STEP_SHARE      18u

/* 64-bit arithmetic on (low, high) pairs, for the hashes of dropout and augmentation */
uvec2 mul64(uvec2 a, uvec2 b) {
    uint hi, lo;
    umulExtended(a.x, b.x, hi, lo);
    return uvec2(lo, hi + a.x * b.y + a.y * b.x);
}
uvec2 add64(uvec2 a, uvec2 b) {
    uint carry;
    uint lo = uaddCarry(a.x, b.x, carry);
    return uvec2(lo, a.y + b.y + carry);
}
uvec2 shr64(uvec2 a, uint k) {          /* 0 < k < 32 */
    return uvec2((a.x >> k) | (a.y << (32u - k)), a.y >> k);
}
uvec2 mix64(uvec2 z) {
    z ^= shr64(z, 30u);
    z = mul64(z, uvec2(0x1CE4E5B9u, 0xBF58476Du));
    z ^= shr64(z, 27u);
    z = mul64(z, uvec2(0x133111EBu, 0x94D049BBu));
    return z ^ shr64(z, 31u);
}
