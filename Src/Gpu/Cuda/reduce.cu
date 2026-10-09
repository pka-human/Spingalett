/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/* reduce.comp: out[o] = scale * (sum over s of part[g][s][r]) + beta out[o], o = g * width + r, with
   compensated sums in the order of s. 256 threads a block. */

#include "common.cuh"

KERNEL(reduce, SpgReducePush) {
    const uint32_t o = global_x();
    if (o >= p.total) return;
    const uint32_t g = o / p.width, r = o % p.width;
    const float *part = F(p.part);
    float sum = 0.0f, comp = 0.0f;
    for (uint32_t s = 0; s < p.slices; s++) {
        float y = part[(g * p.slices + s) * p.width + r] - comp;
        float t = sum + y;
        comp = (t - sum) - y;
        sum = t;
    }
    float v = p.scale * sum;
    if (p.beta != 0.0f) v += p.beta * F(p.out)[o];
    F(p.out)[o] = v;
}
