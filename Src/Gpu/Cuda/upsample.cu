/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/* upsample.comp: nearest and bilinear upsampling by integer factors, forward (a thread an output) and
   backward (a thread an input cell, in the CPU's order). Spec: BILINEAR, BACKWARD, ACT, HALF (words: x 0,
   y 1, dy 2, dx 3). 256 threads, a grid-stride loop. */

#include "common.cuh"

#define FLAG_ACCUMULATE 1u

/* output row o of factor f over `size` rows reads rows r0 and r1, the second weighted w */
DEVICE void bilinear(uint32_t o, uint32_t f, uint32_t size, uint32_t &r0, uint32_t &r1, float &w) {
    const int num = 2 * (int)o + 1 - (int)f, den = 2 * (int)f;
    if (num <= 0) {
        r0 = 0u; r1 = 0u; w = 0.0f;
        return;
    }
    r0 = (uint32_t)(num / den);
    r1 = r0 + 1u < size ? r0 + 1u : r0;
    w = (float)(num % den) / (float)den;
}

/* the output rows (or columns) of factor f that read input row i */
DEVICE void readers(uint32_t i, uint32_t f, uint32_t out_size, uint32_t &o0, uint32_t &o1) {
    o0 = i == 0u ? 0u : (2u * f * (i - 1u) + f) / 2u;
    o1 = umin((2u * f * (i + 1u) + f) / 2u, out_size);
}

KERNEL(upsample, SpgUpsamplePush) {
    const uint32_t BILINEAR = spec.v[0], BACKWARD = spec.v[1], ACT = spec.v[2], HALF = spec.v[3];
    const uint32_t C = p.C, OH = p.H * p.SH, OW = p.W * p.SW, stride = blocks_x() * 256u;
    if (BACKWARD == 0u) {
        const uint32_t total = p.n * OH * OW * C;
        for (uint32_t i = global_x(); i < total; i += stride) {
            const uint32_t c = i % C, q = i / C, ox = q % OW, oy = (q / OW) % OH, s = q / (OW * OH);
            const uint32_t base = s * p.H * p.W * C + c;
            float v;
            if (BILINEAR == 0u) {
                v = ld(p.x, base + ((oy / p.SH) * p.W + ox / p.SW) * C, 0u, HALF);
            } else {
                uint32_t y0, y1, x0, x1;
                float ly, lx;
                bilinear(oy, p.SH, p.H, y0, y1, ly);
                bilinear(ox, p.SW, p.W, x0, x1, lx);
                const float hy = 1.0f - ly, hx = 1.0f - lx;
                const float top = hx * ld(p.x, base + (y0 * p.W + x0) * C, 0u, HALF) + lx * ld(p.x, base + (y0 * p.W + x1) * C, 0u, HALF);
                const float bottom = hx * ld(p.x, base + (y1 * p.W + x0) * C, 0u, HALF) + lx * ld(p.x, base + (y1 * p.W + x1) * C, 0u, HALF);
                v = hy * top + ly * bottom;
            }
            st(p.y, i, 1u, HALF, activate(v, ACT));
        }
        return;
    }
    const uint32_t total = p.n * p.H * p.W * C;
    const bool accumulate = (p.flags & FLAG_ACCUMULATE) != 0u;
    for (uint32_t i = global_x(); i < total; i += stride) {
        const uint32_t c = i % C, q = i / C, ix = q % p.W, iy = (q / p.W) % p.H, s = q / (p.W * p.H);
        const uint32_t obase = s * OH * OW * C + c;
        float g = accumulate ? ld(p.dx, i, 3u, HALF) : 0.0f;
        if (BILINEAR == 0u) {
            for (uint32_t oy = iy * p.SH; oy < (iy + 1u) * p.SH; oy++)
                for (uint32_t ox = ix * p.SW; ox < (ix + 1u) * p.SW; ox++) g += ld(p.dy, obase + (oy * OW + ox) * C, 2u, HALF);
        } else {
            uint32_t oy0, oy1, ox0, ox1;
            readers(iy, p.SH, OH, oy0, oy1);
            readers(ix, p.SW, OW, ox0, ox1);
            for (uint32_t oy = oy0; oy < oy1; oy++) {
                uint32_t y0, y1;
                float ly;
                bilinear(oy, p.SH, p.H, y0, y1, ly);
                if (y0 != iy && y1 != iy) continue;
                const float hy = 1.0f - ly;
                for (uint32_t ox = ox0; ox < ox1; ox++) {
                    uint32_t x0, x1;
                    float lx;
                    bilinear(ox, p.SW, p.W, x0, x1, lx);
                    if (x0 != ix && x1 != ix) continue;
                    const float hx = 1.0f - lx, d = ld(p.dy, obase + (oy * OW + ox) * C, 2u, HALF);
                    if (y0 == iy && x0 == ix) g += hy * hx * d;
                    if (y0 == iy && x1 == ix) g += hy * lx * d;
                    if (y1 == iy && x0 == ix) g += ly * hx * d;
                    if (y1 == iy && x1 == ix) g += ly * lx * d;
                }
            }
        }
        st(p.dx, i, 3u, HALF, ACT == ACT_NONE ? g : g * derivative(ld(p.x, i, 0u, HALF), ACT));
    }
}
