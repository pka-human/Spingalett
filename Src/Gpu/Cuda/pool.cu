/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/* pool.comp: max and average pooling of channels-last samples, forward (a thread an output),
   backward (a thread an input cell, the windows covering it in row order; BACKWARD 2: a thread a window
   of windows that tile the input). Spec: MAXPOOL, BACKWARD, ACT, HALF (words: x 0, y 1, dy 2, dx 3). 256
   threads, a grid-stride loop. */

#include "common.cuh"

DEVICE void window(const SpgPoolPush &p, uint32_t oh, uint32_t ow, uint32_t &h0, uint32_t &h1, uint32_t &w0,
                   uint32_t &w1) {
    const int hs = (int)(oh * p.SH) - (int)p.PH, ws = (int)(ow * p.SW) - (int)p.PW;
    h0 = (uint32_t)(hs > 0 ? hs : 0);
    h1 = (uint32_t)(hs + (int)p.KH < (int)p.H ? hs + (int)p.KH : (int)p.H);
    w0 = (uint32_t)(ws > 0 ? ws : 0);
    w1 = (uint32_t)(ws + (int)p.KW < (int)p.W ? ws + (int)p.KW : (int)p.W);
}

KERNEL(pool, SpgPoolPush) {
    const uint32_t MAXPOOL = spec.v[0], BACKWARD = spec.v[1], ACT = spec.v[2], HALF = spec.v[3];
    const uint32_t C = p.C, stride = blocks_x() * 256u;
    if (BACKWARD == 0u) {
        const uint32_t total = p.n * p.OH * p.OW * C;
        for (uint32_t i = global_x(); i < total; i += stride) {
            const uint32_t c = i % C, q = i / C, ow = q % p.OW, oh = (q / p.OW) % p.OH, s = q / (p.OW * p.OH);
            uint32_t h0, h1, w0, w1;
            window(p, oh, ow, h0, h1, w0, w1);
            const uint32_t base = s * p.H * p.W * C + c;
            float acc = ld(p.x, base + (h0 * p.W + w0) * C, 0u, HALF);
            for (uint32_t h = h0; h < h1; h++)
                for (uint32_t w = (h == h0 ? w0 + 1u : w0); w < w1; w++) {
                    const float v = ld(p.x, base + (h * p.W + w) * C, 0u, HALF);
                    if (MAXPOOL != 0u) { if (v > acc) acc = v; }
                    else acc += v;
                }
            st(p.y, i, 1u, HALF, MAXPOOL != 0u ? acc : acc * (1.0f / (float)((h1 - h0) * (w1 - w0))));
        }
        return;
    }
    if (BACKWARD == 2u) {
        const uint32_t total = p.n * p.OH * p.OW * C;
        for (uint32_t i = global_x(); i < total; i += stride) {
            const uint32_t c = i % C, q = i / C, ow = q % p.OW, oh = (q / p.OW) % p.OH, s = q / (p.OW * p.OH);
            const uint32_t base = s * p.H * p.W * C + c, h0 = oh * p.KH, w0 = ow * p.KW;
            const float d = ld(p.dy, i, 2u, HALF);
            uint32_t at_h = h0, at_w = w0;
            if (MAXPOOL != 0u) {
                float best = ld(p.x, base + (h0 * p.W + w0) * C, 0u, HALF);
                for (uint32_t h = h0; h < h0 + p.KH; h++)
                    for (uint32_t w = (h == h0 ? w0 + 1u : w0); w < w0 + p.KW; w++) {
                        const float v = ld(p.x, base + (h * p.W + w) * C, 0u, HALF);
                        if (v > best) { best = v; at_h = h; at_w = w; }
                    }
            }
            for (uint32_t h = h0; h < h0 + p.KH; h++)
                for (uint32_t w = w0; w < w0 + p.KW; w++) {
                    const uint32_t at = base + (h * p.W + w) * C;
                    const float g = MAXPOOL == 0u ? d * (1.0f / (float)(p.KH * p.KW)) : (h == at_h && w == at_w ? d : 0.0f);
                    st(p.dx, at, 3u, HALF, ACT == ACT_NONE ? g : g * derivative(ld(p.x, at, 0u, HALF), ACT));
                }
        }
        return;
    }
    const uint32_t total = p.n * p.H * p.W * C;
    for (uint32_t i = global_x(); i < total; i += stride) {
        const uint32_t c = i % C, q = i / C, iw = q % p.W, ih = (q / p.W) % p.H, s = q / (p.W * p.H);
        const uint32_t base = s * p.H * p.W * C + c, obase = s * p.OH * p.OW * C + c;
        const int top = (int)(ih + p.PH) - (int)p.KH + 1, left = (int)(iw + p.PW) - (int)p.KW + 1;
        const uint32_t oh0 = top <= 0 ? 0u : ((uint32_t)top + p.SH - 1u) / p.SH, oh1 = umin((ih + p.PH) / p.SH + 1u, p.OH);
        const uint32_t ow0 = left <= 0 ? 0u : ((uint32_t)left + p.SW - 1u) / p.SW, ow1 = umin((iw + p.PW) / p.SW + 1u, p.OW);
        float g = 0.0f;
        for (uint32_t oh = oh0; oh < oh1; oh++)
            for (uint32_t ow = ow0; ow < ow1; ow++) {
                uint32_t h0, h1, w0, w1;
                window(p, oh, ow, h0, h1, w0, w1);
                const float d = ld(p.dy, obase + (oh * p.OW + ow) * C, 2u, HALF);
                if (MAXPOOL == 0u) {
                    g += d * (1.0f / (float)((h1 - h0) * (w1 - w0)));
                    continue;
                }
                float best = ld(p.x, base + (h0 * p.W + w0) * C, 0u, HALF);
                uint32_t at_h = h0, at_w = w0;
                for (uint32_t h = h0; h < h1; h++)
                    for (uint32_t w = (h == h0 ? w0 + 1u : w0); w < w1; w++) {
                        const float v = ld(p.x, base + (h * p.W + w) * C, 0u, HALF);
                        if (v > best) { best = v; at_h = h; at_w = w; }
                    }
                if (at_h == ih && at_w == iw) g += d;
            }
        st(p.dx, i, 3u, HALF, ACT == ACT_NONE ? g : g * derivative(ld(p.x, i, 0u, HALF), ACT));
    }
}
