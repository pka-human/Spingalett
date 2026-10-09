/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/* eltwise.comp: element-wise passes over `total` floats, rows of n (BIAS_ACT, DERIV, MUL, ADD, DROPOUT,
   AFFINE, BDATA, SCALE, AFFINE_ADD, COPY). Spec: OP, ACT, HALF (words: y 0, x 1, a 2, b 3, z 8). 256
   threads, a grid-stride loop. */

#include "common.cuh"

#define BIAS_ACT   0u
#define DERIV      1u
#define MUL        2u
#define ADD        3u
#define DROPOUT    4u
#define AFFINE     5u
#define BDATA      6u
#define SCALE      7u
#define AFFINE_ADD 8u
#define COPY       9u
#define FLAG_A     1u

DEVICE uint32_t hash32(uint32_t x) {
    x ^= x >> 16; x *= 0x7FEB352Du;
    x ^= x >> 15; x *= 0x846CA68Bu;
    x ^= x >> 16;
    return x;
}

/* the mask's base for the sample at `position` of the step: the high word of
   mix64(seed ^ mix64(step * golden + (position << 20) + layer)) */
DEVICE uint32_t dropout_base(const SpgEltwisePush &p, uint32_t position) {
    const uint32_t *h = U(p.header);
    const uint64_t seed = header64(h, STEP_SEED_LO), step = header64(h, STEP_STEP_LO);
    const uint64_t v = step * 0x9E3779B97F4A7C15ull + ((uint64_t)position << 20) + p.layer;
    return (uint32_t)(mix64(seed ^ mix64(v)) >> 32);
}

KERNEL(eltwise, SpgEltwisePush) {
    const uint32_t OP = spec.v[0], ACT = spec.v[1], HALF = spec.v[2];
    const uint32_t stride = blocks_x() * 256u;
    for (uint32_t i = global_x(); i < p.total; i += stride) {
        if (OP == BIAS_ACT) {
            float v = ld(p.y, i, 0u, HALF);
            if (p.flags & FLAG_A) v += ld(p.a, i % p.n, 2u, HALF);
            st(p.y, i, 0u, HALF, activate(v, ACT));
        } else if (OP == DERIV) {
            st(p.y, i, 0u, HALF, ld(p.y, i, 0u, HALF) * derivative(ld(p.x, i, 1u, HALF), ACT));
        } else if (OP == MUL) {
            st(p.y, i, 0u, HALF, ld(p.y, i, 0u, HALF) * ld(p.x, i, 1u, HALF));
        } else if (OP == ADD) {
            st(p.y, i, 0u, HALF, ld(p.y, i, 0u, HALF) + ld(p.x, i, 1u, HALF));
        } else if (OP == DROPOUT) {
            const uint32_t s = i / p.n, j = i % p.n;
            const uint32_t base = dropout_base(p, U(p.header)[STEP_POSITION] + s);
            const float m = hash32(base + j * 0x9E3779B9u) >= p.threshold ? p.keep_scale : 0.0f;
            const float y = ld(p.y, i, 0u, HALF);
            st(p.x, i, 1u, HALF, m * derivative(y, ACT));
            st(p.y, i, 0u, HALF, y * m);
        } else if (OP == AFFINE) {
            const uint32_t c = i % p.n;
            st(p.y, i, 0u, HALF, activate(ld(p.x, i, 1u, HALF) * ld(p.a, c, 2u, HALF) + ld(p.b, c, 3u, HALF), ACT));
        } else if (OP == BDATA) {
            const uint32_t c = i % p.n;
            const float xin = ld(p.b, i, 3u, HALF);
            const float v = ld(p.a, c, 2u, HALF) * ld(p.x, i, 1u, HALF) +
                            (ld(p.a, p.n + c, 2u, HALF) * xin + ld(p.a, 2u * p.n + c, 2u, HALF));
            st(p.y, i, 0u, HALF, ACT == ACT_NONE ? v : v * derivative(xin, ACT));
        } else if (OP == SCALE) {
            st(p.y, i, 0u, HALF, ld(p.y, i, 0u, HALF) * ld(p.a, 0u, 2u, HALF));
        } else if (OP == AFFINE_ADD) {
            const uint32_t c = i % p.n;
            st(p.y, i, 0u, HALF,
               activate(ld(p.z, i, 8u, HALF) + (ld(p.x, i, 1u, HALF) * ld(p.a, c, 2u, HALF) + ld(p.b, c, 3u, HALF)), ACT));
        } else if (OP == COPY) {
            st(p.y, i, 0u, HALF, ld(p.x, i, 1u, HALF));
        }
    }
}
