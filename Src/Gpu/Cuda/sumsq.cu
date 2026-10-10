/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/* sumsq.comp: gradient norm clipping in two passes of fixed order (PARTIAL: sums of squares over slices
   of 4096; CLIP: the scale). Spec: OP. 256 threads. */

#include "common.cuh"

#define PARTIAL 0u
#define CLIP    1u
#define SLICE   4096u

/* The sum of the block's 256 threads' t, in a fixed tree. */
DEVICE float total(float *s, float t) {
    const uint32_t tid = thread_x();
    s[tid] = t;
    barrier();
    for (uint32_t half_n = 128u; half_n > 0u; half_n /= 2u) {
        if (tid < half_n) s[tid] += s[tid + half_n];
        barrier();
    }
    const float sum = s[0];
    barrier();
    return sum;
}

KERNEL(sumsq, SpgSumsqPush) {
    const uint32_t OP = spec.v[0];
    __shared__ float s[256];
    const uint32_t tid = thread_x();
    if (OP == PARTIAL) {
        const uint32_t n = p.n_a + p.n_b;
        for (uint32_t w = block_x(); w < p.slices; w += blocks_x()) {
            float t = 0.0f;
            const uint32_t i0 = w * SLICE, end = umin(n, i0 + SLICE);
            for (uint32_t i = i0 + tid; i < end; i += 256u) {
                const float v = i < p.n_a ? F(p.a)[i] : F(p.b)[i - p.n_a];
                t = fma_(v, v, t);
            }
            t = total(s, t);
            if (tid == 0u) F(p.part)[w] = t;
        }
    } else {
        float t = 0.0f;
        for (uint32_t i = tid; i < p.slices; i += 256u) t += F(p.part)[i];
        const float norm = sqrt_(total(s, t));
        if (tid == 0u) F(p.scalars)[0] = (!isinf_(norm) && !isnan_(norm) && norm > p.max_norm) ? p.max_norm / norm : 1.0f;
    }
}
