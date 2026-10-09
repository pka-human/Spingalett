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

/* Element i through operation OP (a constant: a loop holds its code only). */
template <uint32_t OP>
DEVICE void element(const SpgEltwisePush &p, uint32_t ACT, uint32_t HALF, uint32_t i) {
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

/* Values i .. i + 3 (i a multiple of four): the operands indexed by i read and written four at a time,
   each value computed as element() computes it. */
template <uint32_t OP>
DEVICE void four(const SpgEltwisePush &p, uint32_t ACT, uint32_t HALF, uint32_t i) {
    float4_ y = {0.0f, 0.0f, 0.0f, 0.0f}, x = y, b = y, z = y;
    const bool reads_y = OP == BIAS_ACT || OP == DERIV || OP == MUL || OP == ADD || OP == DROPOUT || OP == SCALE;
    const bool reads_x = OP == DERIV || OP == MUL || OP == ADD || OP == AFFINE || OP == BDATA || OP == AFFINE_ADD || OP == COPY;
    if (reads_y) y = ld4(p.y, i, 0u, HALF);
    if (reads_x) x = ld4(p.x, i, 1u, HALF);
    if (OP == BDATA) b = ld4(p.b, i, 3u, HALF);
    if (OP == AFFINE_ADD) z = ld4(p.z, i, 8u, HALF);
    float4_ out, out_x = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
    for (uint32_t k = 0; k < 4u; k++) {
        const uint32_t e = i + k;
        float v = 0.0f;
        if (OP == BIAS_ACT) {
            v = get4(y, k);
            if (p.flags & FLAG_A) v += ld(p.a, e % p.n, 2u, HALF);
            v = activate(v, ACT);
        } else if (OP == DERIV) {
            v = get4(y, k) * derivative(get4(x, k), ACT);
        } else if (OP == MUL) {
            v = get4(y, k) * get4(x, k);
        } else if (OP == ADD) {
            v = get4(y, k) + get4(x, k);
        } else if (OP == DROPOUT) {
            const uint32_t s = e / p.n, j = e % p.n;
            const uint32_t base = dropout_base(p, U(p.header)[STEP_POSITION] + s);
            const float m = hash32(base + j * 0x9E3779B9u) >= p.threshold ? p.keep_scale : 0.0f;
            set4(out_x, k, m * derivative(get4(y, k), ACT));
            v = get4(y, k) * m;
        } else if (OP == AFFINE) {
            const uint32_t c = e % p.n;
            v = activate(get4(x, k) * ld(p.a, c, 2u, HALF) + ld(p.b, c, 3u, HALF), ACT);
        } else if (OP == BDATA) {
            const uint32_t c = e % p.n;
            const float xin = get4(b, k);
            v = ld(p.a, c, 2u, HALF) * get4(x, k) + (ld(p.a, p.n + c, 2u, HALF) * xin + ld(p.a, 2u * p.n + c, 2u, HALF));
            if (ACT != ACT_NONE) v *= derivative(xin, ACT);
        } else if (OP == SCALE) {
            v = get4(y, k) * ld(p.a, 0u, 2u, HALF);
        } else if (OP == AFFINE_ADD) {
            const uint32_t c = e % p.n;
            v = activate(get4(z, k) + (get4(x, k) * ld(p.a, c, 2u, HALF) + ld(p.b, c, 3u, HALF)), ACT);
        } else {
            v = get4(x, k);
        }
        set4(out, k, v);
    }
    if (OP == DROPOUT) st4(p.x, i, 1u, HALF, out_x);
    st4(p.y, i, 0u, HALF, out);
}

/* The pass of operation OP: four values a thread where the operands indexed by i allow vectors (16 bytes
   apart, `total` a multiple of four), else one. */
template <uint32_t OP>
DEVICE void pass(const SpgEltwisePush &p, uint32_t ACT, uint32_t HALF) {
    const uint32_t stride = blocks_x() * 256u;
    if (p.total % 4u == 0u && ((p.y | p.x | p.b | p.z) & 15u) == 0u) {
        for (uint32_t q = global_x(); q < p.total / 4u; q += stride) four<OP>(p, ACT, HALF, 4u * q);
        return;
    }
    for (uint32_t i = global_x(); i < p.total; i += stride) element<OP>(p, ACT, HALF, i);
}

KERNEL(eltwise, SpgEltwisePush) {
    const uint32_t ACT = spec.v[1], HALF = spec.v[2];
    switch (spec.v[0]) {
        case BIAS_ACT: pass<BIAS_ACT>(p, ACT, HALF); break;
        case DERIV: pass<DERIV>(p, ACT, HALF); break;
        case MUL: pass<MUL>(p, ACT, HALF); break;
        case ADD: pass<ADD>(p, ACT, HALF); break;
        case DROPOUT: pass<DROPOUT>(p, ACT, HALF); break;
        case AFFINE: pass<AFFINE>(p, ACT, HALF); break;
        case BDATA: pass<BDATA>(p, ACT, HALF); break;
        case SCALE: pass<SCALE>(p, ACT, HALF); break;
        case AFFINE_ADD: pass<AFFINE_ADD>(p, ACT, HALF); break;
        default: pass<COPY>(p, ACT, HALF); break;
    }
}
