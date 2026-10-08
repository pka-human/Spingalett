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

/* ActivationFunction */
#define ACT_SIGMOID     0u
#define ACT_RELU        1u
#define ACT_TANH        2u
#define ACT_LEAKY_RELU  3u
#define ACT_FOO52       4u
#define ACT_SOFTMAX     5u
#define ACT_NONE        6u

float activate(float x, uint act) {
    switch (act) {
        case ACT_RELU:          return x > 0.0 ? x : 0.0;
        case ACT_LEAKY_RELU:    return x > 0.0 ? x : 0.01 * x;
        case ACT_SIGMOID:       return 1.0 / (1.0 + exp(-x));
        case ACT_TANH: {        /* tanh() of some drivers overflows to NaN for large arguments */
            /* (1 - e) / (1 + e), e = exp(-2|x|), loses relative precision below |x| = 0.3: an odd
               Taylor polynomial there, as the CPU's vector kernel has (Spingalett.SIMD.c) */
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
        case ACT_FOO52:         return x > 1.0 ? 1.0 + 0.01 * (x - 1.0) : (x < 0.0 ? 0.01 * x : x);
        default:                return x;
    }
}

/* The derivative of the activation, from its output y. */
float derivative(float y, uint act) {
    switch (act) {
        case ACT_RELU:          return y > 0.0 ? 1.0 : 0.0;
        case ACT_LEAKY_RELU:    return y > 0.0 ? 1.0 : 0.01;
        case ACT_SIGMOID:
        case ACT_SOFTMAX:       return y * (1.0 - y);
        case ACT_TANH:          return 1.0 - y * y;
        case ACT_FOO52:         return (y > 1.0 || y < 0.0) ? 0.01 : 1.0;
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
