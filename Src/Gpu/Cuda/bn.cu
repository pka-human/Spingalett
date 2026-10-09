/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/* bn.comp: the per-channel coefficients of batch normalization, a thread a channel (TRAIN, INFER,
   BACKWARD). Spec: MODE, HALF (word: x 1). 64 threads. */

#include "common.cuh"

#define TRAIN    0u
#define INFER    1u
#define BACKWARD 2u

/* sums j (0 or 1) of channel c over the slices, compensated, in order */
DEVICE float total(const SpgBnPush &p, uint32_t c, uint32_t j) {
    float sum = 0.0f, comp = 0.0f;
    for (uint32_t s = 0; s < p.slices; s++) {
        float y = F(p.part)[s * 2u * p.C + j * p.C + c] - comp;
        float t = sum + y;
        comp = (t - sum) - y;
        sum = t;
    }
    return sum;
}

KERNEL(bn, SpgBnPush) {
    const uint32_t MODE = spec.v[0], HALF = spec.v[1];
    const uint32_t c = global_x(), C = p.C;
    if (c >= C) return;
    float *stats = F(p.stats);
    if (MODE == INFER) {
        const float a = F(p.gamma)[c] / sqrt_(F(p.rvar)[c] + p.eps);
        stats[2u * C + c] = a;
        stats[3u * C + c] = F(p.beta)[c] - F(p.rmean)[c] * a;
        return;
    }
    if (MODE == TRAIN) {
        const float d = total(p, c, 0u) / p.m, var = max_(total(p, c, 1u) / p.m - d * d, 0.0f);
        const float mean = ld(p.x, c, 1u, HALF) + d, inv = 1.0f / sqrt_(var + p.eps);
        stats[c] = mean;
        stats[C + c] = inv;
        F(p.rmean)[c] += p.momentum * (mean - F(p.rmean)[c]);
        if (p.m > 1.0f) F(p.rvar)[c] += p.momentum * (var * p.m / (p.m - 1.0f) - F(p.rvar)[c]);
        const float a = F(p.gamma)[c] * inv;
        stats[2u * C + c] = a;
        stats[3u * C + c] = F(p.beta)[c] - mean * a;
        return;
    }
    /* backward: s2 = sum of dy * xhat */
    const float inv = stats[C + c], s1 = total(p, c, 0u), s2 = total(p, c, 1u) * inv;
    const float dg = s2 * p.scale, db = s1 * p.scale;
    F(p.ggamma)[c] = p.beta_g == 0.0f ? dg : dg + p.beta_g * F(p.ggamma)[c];
    F(p.gbeta)[c] = p.beta_g == 0.0f ? db : db + p.beta_g * F(p.gbeta)[c];
    const float k = F(p.gamma)[c] * inv, m1 = s1 / p.m, m2 = s2 / p.m;
    F(p.coef)[c] = k;
    F(p.coef)[C + c] = -k * inv * m2;
    F(p.coef)[2u * C + c] = -k * m1 + k * inv * m2 * stats[c];
}
