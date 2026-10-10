/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/* optim.comp: one optimizer step over n parameters, as Spingalett.SIMD.c updates them; with bit 8 of
   HALF the new weights are also written to wh as bfloat16. Spec: TYPE, HALF. 256 threads, a grid-stride loop,
   four parameters a thread where the buffers allow vectors. */

#include "common.cuh"

#define SGD      0u
#define MOMENTUM 1u
#define RMSPROP  2u
#define ADAM     3u
#define ADAMW    4u

/* The step of one parameter (w, its gradient g and moments m, v), as Spingalett.SIMD.c takes it. */
template <uint32_t TYPE>
DEVICE float step(const SpgOptimPush &p, float lr, float m_factor, float v_factor, float w, float g, float &m, float &v) {
    if (TYPE == SGD) return w - lr * (g + p.decay * w);
    if (TYPE == MOMENTUM) {
        m = p.momentum * m + (g + p.decay * w);
        return w - lr * m;
    }
    if (TYPE == RMSPROP) {
        const float gd = g + p.decay * w;
        v = p.beta2 * v + (1.0f - p.beta2) * (gd * gd);
        return w - lr * (gd / (sqrt_(v) + p.epsilon));
    }
    const float gd = g + (TYPE == ADAM ? p.decay * w : 0.0f);
    m = p.beta1 * m + (1.0f - p.beta1) * gd;
    v = p.beta2 * v + (1.0f - p.beta2) * (gd * gd);
    const float wd = TYPE == ADAMW ? 1.0f - lr * p.decay : 1.0f;
    return w * wd - lr * ((m * m_factor) / (sqrt_(v * v_factor) + p.epsilon));
}

/* The step over the n parameters: four a thread where the buffers allow vectors. */
template <uint32_t TYPE>
DEVICE void steps(const SpgOptimPush &p, uint32_t HALF) {
    const uint32_t stride = blocks_x() * 256u;
    const uint32_t *header = U(p.header);
    const float lr = bits_float(header[STEP_LR]);
    const float m_factor = bits_float(header[STEP_M_FACTOR]), v_factor = bits_float(header[STEP_V_FACTOR]);
    const bool moments = TYPE == MOMENTUM || TYPE == ADAM || TYPE == ADAMW, second = TYPE == RMSPROP || TYPE == ADAM || TYPE == ADAMW;
    const bool half = (HALF >> 8) & 1u;
    if (p.n % 4u == 0u && ((p.w | p.g | (moments ? p.m : 0u) | (second ? p.v : 0u)) & 15u) == 0u && (!half || (p.wh & 7u) == 0u)) {
        for (uint32_t q = global_x(); q < p.n / 4u; q += stride) {
            const float4_ w = ((const float4_ *)p.w)[q], g = ((const float4_ *)p.g)[q];
            float4_ m = moments ? ((const float4_ *)p.m)[q] : float4_{0.0f, 0.0f, 0.0f, 0.0f};
            float4_ v = second ? ((const float4_ *)p.v)[q] : float4_{0.0f, 0.0f, 0.0f, 0.0f};
            const float4_ out = {step<TYPE>(p, lr, m_factor, v_factor, w.x, g.x, m.x, v.x),
                                 step<TYPE>(p, lr, m_factor, v_factor, w.y, g.y, m.y, v.y),
                                 step<TYPE>(p, lr, m_factor, v_factor, w.z, g.z, m.z, v.z),
                                 step<TYPE>(p, lr, m_factor, v_factor, w.w, g.w, m.w, v.w)};
            ((float4_ *)p.w)[q] = out;
            if (moments) ((float4_ *)p.m)[q] = m;
            if (second) ((float4_ *)p.v)[q] = v;
            if (half) st4(p.wh, 4u * q, 8u, HALF, out);
        }
        return;
    }
    float *W = F(p.w), *M = F(p.m), *V = F(p.v);
    const float *G = F(p.g);
    for (uint32_t i = global_x(); i < p.n; i += stride) {
        float m = moments ? M[i] : 0.0f, v = second ? V[i] : 0.0f;
        const float w = step<TYPE>(p, lr, m_factor, v_factor, W[i], G[i], m, v);
        if (moments) M[i] = m;
        if (second) V[i] = v;
        W[i] = w;
        if (half) st(p.wh, i, 8u, HALF, w);
    }
}

KERNEL(optim, SpgOptimPush) {
    const uint32_t HALF = spec.v[1];
    switch (spec.v[0]) {
        case SGD: steps<SGD>(p, HALF); break;
        case MOMENTUM: steps<MOMENTUM>(p, HALF); break;
        case RMSPROP: steps<RMSPROP>(p, HALF); break;
        case ADAM: steps<ADAM>(p, HALF); break;
        default: steps<ADAMW>(p, HALF); break;
    }
}
