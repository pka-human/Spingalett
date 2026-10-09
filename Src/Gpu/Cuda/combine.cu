/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/* combine.comp: additions, concatenations along the channels and global average pooling, forward and
   backward. Spec: OP, ACT, BACKWARD, HALF (words: x 0, y 1, b 2). 256 threads, a grid-stride loop. */

#include "common.cuh"

#define ADD   0u
#define SLICE 1u
#define GAP   2u
#define FLAG_ACCUMULATE 1u
#define FLAG_LAST       2u

KERNEL(combine, SpgCombinePush) {
    const uint32_t OP = spec.v[0], ACT = spec.v[1], BACKWARD = spec.v[2], HALF = spec.v[3];
    const uint32_t stride = blocks_x() * 256u;
    const bool accumulate = (p.flags & FLAG_ACCUMULATE) != 0u;
    if (OP == ADD) {
        const uint32_t total = p.n * p.cells * p.C;
        for (uint32_t i = global_x(); i < total; i += stride) {
            const float v = accumulate ? ld(p.y, i, 1u, HALF) + ld(p.x, i, 0u, HALF) : ld(p.x, i, 0u, HALF) + ld(p.b, i, 2u, HALF);
            st(p.y, i, 1u, HALF, (p.flags & FLAG_LAST) ? activate(v, ACT) : v);
        }
    } else if (OP == SLICE) {
        const uint32_t total = p.n * p.cells * p.ck;
        for (uint32_t i = global_x(); i < total; i += stride) {
            const uint32_t c = i % p.ck, q = i / p.ck, wide = q * p.C + p.c0 + c;
            if (BACKWARD == 0u) {
                st(p.y, wide, 1u, HALF, activate(ld(p.x, i, 0u, HALF), ACT));
            } else {
                const float v = accumulate ? ld(p.x, i, 0u, HALF) + ld(p.y, wide, 1u, HALF) : ld(p.y, wide, 1u, HALF);
                st(p.x, i, 0u, HALF, ACT == ACT_NONE ? v : v * derivative(ld(p.b, i, 2u, HALF), ACT));
            }
        }
    } else if (BACKWARD == 0u) {
        const uint32_t total = p.n * p.C;
        const float inv = 1.0f / (float)p.cells;
        for (uint32_t i = global_x(); i < total; i += stride) {
            const uint32_t c = i % p.C, s = i / p.C, base = s * p.cells * p.C + c;
            float v = ld(p.x, base, 0u, HALF);
            for (uint32_t q = 1u; q < p.cells; q++) v += ld(p.x, base + q * p.C, 0u, HALF);
            st(p.y, i, 1u, HALF, activate(v * inv, ACT));
        }
    } else {
        const uint32_t total = p.n * p.cells * p.C;
        const float inv = 1.0f / (float)p.cells;
        for (uint32_t i = global_x(); i < total; i += stride) {
            const uint32_t c = i % p.C, s = i / (p.cells * p.C);
            const float g = ld(p.y, s * p.C + c, 1u, HALF) * inv;
            const float v = accumulate ? ld(p.x, i, 0u, HALF) + g : g;
            st(p.x, i, 0u, HALF, ACT == ACT_NONE ? v : v * derivative(ld(p.b, i, 2u, HALF), ACT));
        }
    }
}
