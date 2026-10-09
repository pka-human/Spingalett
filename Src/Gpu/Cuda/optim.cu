/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/* optim.comp: one optimizer step over n parameters, as Spingalett.SIMD.c updates them; with bit 8 of
   HALF the new weights are also written to wh as bfloat16. Spec: TYPE, HALF. 256 threads. */

#include "common.cuh"

#define SGD      0u
#define MOMENTUM 1u
#define RMSPROP  2u
#define ADAM     3u
#define ADAMW    4u

KERNEL(optim, SpgOptimPush) {
    const uint32_t TYPE = spec.v[0], HALF = spec.v[1];
    const uint32_t stride = blocks_x() * 256u;
    const uint32_t *header = U(p.header);
    const float lr = bits_float(header[STEP_LR]);
    const float m_factor = bits_float(header[STEP_M_FACTOR]), v_factor = bits_float(header[STEP_V_FACTOR]);
    float *W = F(p.w), *M = F(p.m), *V = F(p.v);
    const float *G = F(p.g);
    for (uint32_t i = global_x(); i < p.n; i += stride) {
        float w = W[i];
        if (TYPE == SGD) {
            w -= lr * (G[i] + p.decay * w);
        } else if (TYPE == MOMENTUM) {
            const float m = p.momentum * M[i] + (G[i] + p.decay * w);
            M[i] = m;
            w -= lr * m;
        } else if (TYPE == RMSPROP) {
            const float g = G[i] + p.decay * w;
            const float v = p.beta2 * V[i] + (1.0f - p.beta2) * (g * g);
            V[i] = v;
            w -= lr * (g / (sqrt_(v) + p.epsilon));
        } else {
            const float g = G[i] + (TYPE == ADAM ? p.decay * w : 0.0f);
            const float m = p.beta1 * M[i] + (1.0f - p.beta1) * g;
            const float v = p.beta2 * V[i] + (1.0f - p.beta2) * (g * g);
            M[i] = m;
            V[i] = v;
            const float wd = TYPE == ADAMW ? 1.0f - lr * p.decay : 1.0f;
            w = w * wd - lr * ((m * m_factor) / (sqrt_(v * v_factor) + p.epsilon));
        }
        W[i] = w;
        if ((HALF >> 8) & 1u) st(p.wh, i, 8u, HALF, w);
    }
}
