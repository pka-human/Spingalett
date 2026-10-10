/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/* pool.comp: max and average pooling of channels-last samples, forward (a thread an output),
   backward (a thread an input cell, the windows covering it in row order; BACKWARD 2: a thread a window
   of windows that tile the input). Spec: MAXPOOL, BACKWARD, ACT, HALF (words: x 0, y 1, dy 2, dx 3). 256
   threads, a grid-stride loop; four channels a thread where they come in fours (pool4()). */

#include "common.cuh"

DEVICE void window(const SpgPoolPush &p, uint32_t oh, uint32_t ow, uint32_t &h0, uint32_t &h1, uint32_t &w0,
                   uint32_t &w1) {
    const int hs = (int)(oh * p.SH) - (int)p.PH, ws = (int)(ow * p.SW) - (int)p.PW;
    h0 = (uint32_t)(hs > 0 ? hs : 0);
    h1 = (uint32_t)(hs + (int)p.KH < (int)p.H ? hs + (int)p.KH : (int)p.H);
    w0 = (uint32_t)(ws > 0 ? ws : 0);
    w1 = (uint32_t)(ws + (int)p.KW < (int)p.W ? ws + (int)p.KW : (int)p.W);
}

/* The forward pass and BACKWARD 2 four channels a thread (channels in fours, the maps at 16 bytes): each
   channel compared and summed as one alone would be. */
DEVICE void pool4(const SpgPoolPush &p, uint32_t MAXPOOL, uint32_t BACKWARD, uint32_t ACT, uint32_t HALF) {
    const uint32_t C = p.C, quads = C / 4u, stride = blocks_x() * 256u, total = p.n * p.OH * p.OW * quads;
    const float slope = act_slope(ACT);
    for (uint32_t t = global_x(); t < total; t += stride) {
        const uint32_t c = t % quads * 4u, q = t / quads, ow = q % p.OW, oh = (q / p.OW) % p.OH, s = q / (p.OW * p.OH);
        const uint32_t i = q * C + c;
        if (BACKWARD == 0u) {
            uint32_t h0, h1, w0, w1;
            window(p, oh, ow, h0, h1, w0, w1);
            const uint32_t base = s * p.H * p.W * C + c;
            float4_ acc = ld4(p.x, base + (h0 * p.W + w0) * C, 0u, HALF);
            for (uint32_t h = h0; h < h1; h++)
                for (uint32_t w = (h == h0 ? w0 + 1u : w0); w < w1; w++) {
                    const float4_ v = ld4(p.x, base + (h * p.W + w) * C, 0u, HALF);
                    if (MAXPOOL != 0u) {
                        if (v.x > acc.x) acc.x = v.x;
                        if (v.y > acc.y) acc.y = v.y;
                        if (v.z > acc.z) acc.z = v.z;
                        if (v.w > acc.w) acc.w = v.w;
                    } else {
                        acc = add4(acc, v);
                    }
                }
            st4(p.y, i, 1u, HALF, MAXPOOL != 0u ? acc : scale4(acc, 1.0f / (float)((h1 - h0) * (w1 - w0))));
            continue;
        }
        const uint32_t base = s * p.H * p.W * C + c, h0 = oh * p.KH, w0 = ow * p.KW;
        const float4_ d = ld4(p.dy, i, 2u, HALF);
        uint32_t at_h[4] = {h0, h0, h0, h0}, at_w[4] = {w0, w0, w0, w0};
        if (MAXPOOL != 0u) {
            float4_ best = ld4(p.x, base + (h0 * p.W + w0) * C, 0u, HALF);
            for (uint32_t h = h0; h < h0 + p.KH; h++)
                for (uint32_t w = (h == h0 ? w0 + 1u : w0); w < w0 + p.KW; w++) {
                    const float4_ v = ld4(p.x, base + (h * p.W + w) * C, 0u, HALF);
                    if (v.x > best.x) { best.x = v.x; at_h[0] = h; at_w[0] = w; }
                    if (v.y > best.y) { best.y = v.y; at_h[1] = h; at_w[1] = w; }
                    if (v.z > best.z) { best.z = v.z; at_h[2] = h; at_w[2] = w; }
                    if (v.w > best.w) { best.w = v.w; at_h[3] = h; at_w[3] = w; }
                }
        }
        for (uint32_t h = h0; h < h0 + p.KH; h++)
            for (uint32_t w = w0; w < w0 + p.KW; w++) {
                const uint32_t at = base + (h * p.W + w) * C;
                float4_ g;
                if (MAXPOOL == 0u) {
                    g = scale4(d, 1.0f / (float)(p.KH * p.KW));
                } else {
                    g = float4_{h == at_h[0] && w == at_w[0] ? d.x : 0.0f, h == at_h[1] && w == at_w[1] ? d.y : 0.0f,
                                h == at_h[2] && w == at_w[2] ? d.z : 0.0f, h == at_h[3] && w == at_w[3] ? d.w : 0.0f};
                }
                if (ACT != ACT_NONE) {
                    const float4_ x = ld4(p.x, at, 0u, HALF);
                    g = mul4(g, float4_{derivative_s(x.x, ACT, slope), derivative_s(x.y, ACT, slope),
                                        derivative_s(x.z, ACT, slope), derivative_s(x.w, ACT, slope)});
                }
                st4(p.dx, at, 3u, HALF, g);
            }
    }
}

KERNEL(pool, SpgPoolPush) {
    const uint32_t MAXPOOL = spec.v[0], BACKWARD = spec.v[1], ACT = spec.v[2], HALF = spec.v[3];
    const uint32_t C = p.C, stride = blocks_x() * 256u;
    if ((BACKWARD == 0u || BACKWARD == 2u) && C % 4u == 0u && ((p.x | p.y | p.dy | p.dx) & 15u) == 0u) {
        pool4(p, MAXPOOL, BACKWARD, ACT, HALF);
        return;
    }
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
