/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/* ln.comp: layer normalization of channels-last cells, T threads (a power of two) a cell, their sums in
   a fixed tree. Spec: T, BACKWARD, ACT, HALF (words: x 0, y 1, dy 2). 256 threads. */

#include "common.cuh"

#define FLAG_STATS 1u

/* s[the cell's first thread] = the sum of its T threads' values */
DEVICE void tree(float *s1, float *s2, uint32_t T, uint32_t tid, uint32_t lane, bool both) {
    barrier();
    for (uint32_t h = T / 2u; h > 0u; h /= 2u) {
        if (lane < h) {
            s1[tid] += s1[tid + h];
            if (both) s2[tid] += s2[tid + h];
        }
        barrier();
    }
}

KERNEL(ln, SpgLnPush) {
    const uint32_t T = spec.v[0], BACKWARD = spec.v[1], ACT = spec.v[2], HALF = spec.v[3];
    __shared__ float s1[256], s2[256];
    const uint32_t tid = thread_x(), lane = tid % T, first = tid - lane, per = 256u / T, C = p.C;
    const float inv = 1.0f / (float)C;
    const float *gamma = F(p.gamma), *beta = F(p.beta);
    float *stats = F(p.stats);
    for (uint32_t base = block_x() * per; base < p.cells; base += blocks_x() * per) {
        const uint32_t cell = base + tid / T, at = cell * C;
        const bool live = cell < p.cells;
        if (BACKWARD == 0u) {
            float sum = 0.0f;
            if (live) for (uint32_t c = lane; c < C; c += T) sum += ld(p.x, at + c, 0u, HALF);
            s1[tid] = sum;
            tree(s1, s2, T, tid, lane, false);
            const float mean = s1[first] * inv;
            float var = 0.0f;
            if (live)
                for (uint32_t c = lane; c < C; c += T) {
                    const float d = ld(p.x, at + c, 0u, HALF) - mean;
                    var = fma_(d, d, var);
                }
            barrier();
            s1[tid] = var;
            tree(s1, s2, T, tid, lane, false);
            const float rstd = 1.0f / sqrt_(s1[first] * inv + p.eps);
            if (live) {
                for (uint32_t c = lane; c < C; c += T)
                    st(p.y, at + c, 1u, HALF, activate((ld(p.x, at + c, 0u, HALF) - mean) * rstd * gamma[c] + beta[c], ACT));
                if ((p.flags & FLAG_STATS) && lane == 0u) {
                    stats[2u * cell] = mean;
                    stats[2u * cell + 1u] = rstd;
                }
            }
        } else {
            const float mean = live ? stats[2u * cell] : 0.0f, rstd = live ? stats[2u * cell + 1u] : 0.0f;
            float a = 0.0f, b = 0.0f;
            if (live)
                for (uint32_t c = lane; c < C; c += T) {
                    const float gc = gamma[c] * ld(p.dy, at + c, 2u, HALF);
                    a += gc;
                    b = fma_(gc, (ld(p.x, at + c, 0u, HALF) - mean) * rstd, b);
                }
            s1[tid] = a;
            s2[tid] = b;
            tree(s1, s2, T, tid, lane, true);
            const float sa = s1[first], sb = s2[first];
            if (live)
                for (uint32_t c = lane; c < C; c += T) {
                    const float v = ld(p.x, at + c, 0u, HALF), xhat = (v - mean) * rstd;
                    const float d = rstd * (gamma[c] * ld(p.dy, at + c, 2u, HALF) - (sa + xhat * sb) * inv);
                    st(p.y, at + c, 1u, HALF, ACT == ACT_NONE ? d : d * derivative(v, ACT));
                }
        }
        barrier();
    }
}
