/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/* ipass.comp: the passes of integer models besides their products, each value computed in the CPU's order of
   operations, to the bit (the multiplies and additions round alone, mul.rn and add.rn: no multiply-add; divisions are
   IEEE's). Spec: OP, ACT, HALF (DIRECT: bit 0 x, bit 1 w kept as bfloat16). 256 threads, a grid-stride loop, four
   outputs at a time with VEC4. */

#include "common.cuh"

#define EPI    0u
#define DIRECT 1u
#define POOL   2u
#define GAP    3u
#define ADD    4u
#define MUL    5u
#define SLICE  6u
#define BN     7u
#define UP     8u

#define FLAG_FIRST 1u
#define FLAG_LAST  2u
#define FLAG_MAX   1u
#define FLAG_TRANSPOSED 1u
#define FLAG_BILINEAR   1u
#define FLAG_FLOATS 2u
#define FLAG_VEC4  4u
#define FLAG_DW4   8u
#define FLAG_OG4   16u

/* the activations of integer models' layers on the GPU, as the CPU's: none, ReLU (0 for x <= 0 and NaN), leaky ReLU */
DEVICE float int_act(float x, uint32_t act) {
    if (act == ACT_RELU) return x > 0.0f ? x : 0.0f;
    if (act == ACT_LEAKY_RELU) return x > 0.0f ? x : __nvvm_mul_rn_f(0.01f, x);
    return x;
}

DEVICE float4_ int_act4(float4_ v, uint32_t act) {
    return float4_{int_act(v.x, act), int_act(v.y, act), int_act(v.z, act), int_act(v.w, act)};
}

/* spingalett_int_output() */
DEVICE float int_output(float bias, float scale, float xs, int32_t acc) {
    return __nvvm_add_rn_f(bias, __nvvm_mul_rn_f(__nvvm_mul_rn_f(scale, xs), (float)acc));
}

/* the same of four outputs, their sums in floats (exact integers) */
DEVICE float4_ int_output4(float4_ bias, float4_ scale, float xs, float4_ sums) {
    return float4_{int_output(bias.x, scale.x, xs, (int32_t)sums.x), int_output(bias.y, scale.y, xs, (int32_t)sums.y),
                   int_output(bias.z, scale.z, xs, (int32_t)sums.z), int_output(bias.w, scale.w, xs, (int32_t)sums.w)};
}

/* x a + b, a + b and a b of four values, each rounded alone */
DEVICE float affine_rn(float x, float a, float b) { return __nvvm_add_rn_f(__nvvm_mul_rn_f(x, a), b); }
DEVICE float4_ affine4(float4_ x, float4_ a, float4_ b) {
    return float4_{affine_rn(x.x, a.x, b.x), affine_rn(x.y, a.y, b.y), affine_rn(x.z, a.z, b.z),
                   affine_rn(x.w, a.w, b.w)};
}
DEVICE float4_ add4_rn(float4_ a, float4_ b) {
    return float4_{__nvvm_add_rn_f(a.x, b.x), __nvvm_add_rn_f(a.y, b.y), __nvvm_add_rn_f(a.z, b.z),
                   __nvvm_add_rn_f(a.w, b.w)};
}
DEVICE float4_ mul4_rn(float4_ a, float4_ b) {
    return float4_{__nvvm_mul_rn_f(a.x, b.x), __nvvm_mul_rn_f(a.y, b.y), __nvvm_mul_rn_f(a.z, b.z),
                   __nvvm_mul_rn_f(a.w, b.w)};
}

/* a convolution's sums window by window at output (s, oh, ow) of filter j (DIRECT): the filters transposed, a tap's
   channel's codes of every filter side by side */
DEVICE float direct(const SpgIpassPush &p, uint32_t HALF, uint32_t s, uint32_t oh, uint32_t ow, uint32_t j) {
    const uint32_t C = p.out_c, CG = p.in_c / p.groups, g = j / (C / p.groups);
    const bool transposed = (p.flags & FLAG_TRANSPOSED) != 0u, floats = (p.flags & FLAG_FLOATS) != 0u;
    int32_t acc = 0;
    float sum = 0.0f;
    for (uint32_t th = 0; th < p.kh; th++) {
        int32_t ih;
        if (transposed) {
            const int32_t nh = (int32_t)(oh + p.ph) - (int32_t)th;
            if (nh < 0 || nh % (int32_t)p.sh != 0) continue;
            ih = nh / (int32_t)p.sh;
        } else {
            ih = (int32_t)(oh * p.sh) - (int32_t)p.ph + (int32_t)th;
        }
        if (ih < 0 || ih >= (int32_t)p.in_h) continue;
        for (uint32_t tw = 0; tw < p.kw; tw++) {
            int32_t iw;
            if (transposed) {
                const int32_t nw = (int32_t)(ow + p.pw) - (int32_t)tw;
                if (nw < 0 || nw % (int32_t)p.sw != 0) continue;
                iw = nw / (int32_t)p.sw;
            } else {
                iw = (int32_t)(ow * p.sw) - (int32_t)p.pw + (int32_t)tw;
            }
            if (iw < 0 || iw >= (int32_t)p.in_w) continue;
            const uint32_t cell = ((s * p.in_h + (uint32_t)ih) * p.in_w + (uint32_t)iw) * p.in_c + g * CG;
            const uint32_t tap = (th * p.kw + tw) * CG;
            if (floats)
                for (uint32_t c = 0; c < CG; c++)
                    sum = fma_(ld(p.x, cell + c, 0u, HALF), ld(p.w, (tap + c) * C + j, 1u, HALF), sum);
            else
                for (uint32_t c = 0; c < CG; c++)
                    acc += (int32_t)ld(p.x, cell + c, 0u, HALF) * (int32_t)ld(p.w, (tap + c) * C + j, 1u, HALF);
        }
    }
    return floats ? sum : (float)acc;
}

/* the sums of filters j .. j + 3 at output (s, oh, ow), in floats (DIRECT with DW4: a depthwise convolution's four
   channels; OG4: four filters of a group, a value of the input times their four codes) */
DEVICE float4_ direct4(const SpgIpassPush &p, uint32_t HALF, uint32_t s, uint32_t oh, uint32_t ow, uint32_t j) {
    const uint32_t C = p.out_c, CG = p.in_c / p.groups, g = j / (C / p.groups);
    const bool transposed = (p.flags & FLAG_TRANSPOSED) != 0u, dw = (p.flags & FLAG_DW4) != 0u;
    float4_ sum = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint32_t th = 0; th < p.kh; th++) {
        int32_t ih;
        if (transposed) {
            const int32_t nh = (int32_t)(oh + p.ph) - (int32_t)th;
            if (nh < 0 || nh % (int32_t)p.sh != 0) continue;
            ih = nh / (int32_t)p.sh;
        } else {
            ih = (int32_t)(oh * p.sh) - (int32_t)p.ph + (int32_t)th;
        }
        if (ih < 0 || ih >= (int32_t)p.in_h) continue;
        for (uint32_t tw = 0; tw < p.kw; tw++) {
            int32_t iw;
            if (transposed) {
                const int32_t nw = (int32_t)(ow + p.pw) - (int32_t)tw;
                if (nw < 0 || nw % (int32_t)p.sw != 0) continue;
                iw = nw / (int32_t)p.sw;
            } else {
                iw = (int32_t)(ow * p.sw) - (int32_t)p.pw + (int32_t)tw;
            }
            if (iw < 0 || iw >= (int32_t)p.in_w) continue;
            const uint32_t cell = ((s * p.in_h + (uint32_t)ih) * p.in_w + (uint32_t)iw) * p.in_c + g * CG;
            const uint32_t tap = (th * p.kw + tw) * CG;
            if (dw) {
                sum = fma4(ld4(p.x, cell, 0u, HALF), ld4(p.w, tap * C + j, 1u, HALF), sum);
            } else {
                for (uint32_t c = 0; c < CG; c++) {
                    const float x = ld(p.x, cell + c, 0u, HALF);
                    sum = fma4(float4_{x, x, x, x}, ld4(p.w, (tap + c) * C + j, 1u, HALF), sum);
                }
            }
        }
    }
    return sum;
}

KERNEL(ipass, SpgIpassPush) {
    const uint32_t OP = spec.v[0], ACT = spec.v[1], HALF = spec.v[2];
    const uint32_t stride = blocks_x() * 256u;
    if (OP == DIRECT && (p.flags & (FLAG_DW4 | FLAG_OG4))) {
        for (uint32_t t = global_x(); t < p.total / 4u; t += stride) {
            const uint32_t i = 4u * t, C = p.out_c, j = i % C, q = i / C, ow = q % p.out_w, oh = (q / p.out_w) % p.out_h;
            const uint32_t s = q / (p.out_w * p.out_h);
            const float4_ bias = ((const float4_ *)p.e0)[j >> 2], scale = ((const float4_ *)p.e1)[j >> 2];
            ((float4_ *)p.y)[t] = int_act4(int_output4(bias, scale, F(p.e2)[s], direct4(p, HALF, s, oh, ow, j)), ACT);
        }
        return;
    }
    if ((p.flags & FLAG_VEC4) && (OP == EPI || OP == ADD || OP == MUL || OP == BN || OP == SLICE)) {
        const float4_ *X = (const float4_ *)p.x, *E0 = (const float4_ *)p.e0, *E1 = (const float4_ *)p.e1;
        float4_ *Y = (float4_ *)p.y;
        for (uint32_t t = global_x(); t < p.total / 4u; t += stride) {
            const uint32_t i = 4u * t;
            float4_ v;
            if (OP == SLICE) {
                const uint32_t c = i % p.in_c, q = i / p.in_c;
                Y[(q * p.out_c + p.c0 + c) >> 2] = int_act4(X[t], ACT);
                continue;
            }
            if (OP == EPI) {
                const uint32_t C = p.out_c, m = i / C, j = i % C;
                v = int_act4(int_output4(E0[j >> 2], E1[j >> 2], F(p.w)[m / p.n], X[t]), ACT);
            } else if (OP == BN) {
                const uint32_t c = (i % p.out_c) >> 2;
                v = int_act4(affine4(X[t], E0[c], E1[c]), ACT);
            } else {
                const bool first = (p.flags & FLAG_FIRST) != 0u;
                const float4_ a = first ? X[t] : Y[t], b = first ? E0[t] : X[t];
                const float4_ r = OP == ADD ? add4_rn(a, b) : mul4_rn(a, b);
                v = (p.flags & FLAG_LAST) ? int_act4(r, ACT) : r;
            }
            Y[t] = v;
        }
        return;
    }
    for (uint32_t i = global_x(); i < p.total; i += stride) {
        float v;
        if (OP == EPI) {
            const uint32_t C = p.out_c, m = i / C, j = i % C;
            int32_t acc = 0;
            if (p.slices == 1u) {
                acc = (int32_t)F(p.x)[i];
            } else {
                const uint32_t OG = C / p.groups, g = j / OG, rows = p.total / C;
                for (uint32_t s = 0; s < p.slices; s++)
                    acc += (int32_t)F(p.x)[((g * p.slices + s) * rows + m) * OG + j % OG];
            }
            v = int_output(F(p.e0)[j], F(p.e1)[j], F(p.w)[m / p.n], acc);
        } else if (OP == DIRECT) {
            const uint32_t C = p.out_c, j = i % C, q = i / C, ow = q % p.out_w, oh = (q / p.out_w) % p.out_h;
            const uint32_t s = q / (p.out_w * p.out_h);
            v = int_output(F(p.e0)[j], F(p.e1)[j], F(p.e2)[s], (int32_t)direct(p, HALF, s, oh, ow, j));
        } else if (OP == POOL) {
            /* the engine's pool_forward(): the window clipped to the input, its first cell, then the others in order */
            const uint32_t C = p.out_c, c = i % C, q = i / C, ow = q % p.out_w, oh = (q / p.out_w) % p.out_h;
            const uint32_t s = q / (p.out_w * p.out_h);
            const int32_t hs = (int32_t)(oh * p.sh) - (int32_t)p.ph, ws = (int32_t)(ow * p.sw) - (int32_t)p.pw;
            const uint32_t h0 = (uint32_t)(hs > 0 ? hs : 0), w0 = (uint32_t)(ws > 0 ? ws : 0);
            const uint32_t h1 = (uint32_t)(hs + (int32_t)p.kh < (int32_t)p.in_h ? hs + (int32_t)p.kh : (int32_t)p.in_h);
            const uint32_t w1 = (uint32_t)(ws + (int32_t)p.kw < (int32_t)p.in_w ? ws + (int32_t)p.kw : (int32_t)p.in_w);
            const bool most = (p.flags & FLAG_MAX) != 0u;
            const uint32_t base = s * p.in_h * p.in_w;
            float acc = F(p.x)[(base + h0 * p.in_w + w0) * C + c];
            for (uint32_t h = h0; h < h1; h++)
                for (uint32_t w = h == h0 ? w0 + 1u : w0; w < w1; w++) {
                    const float x = F(p.x)[(base + h * p.in_w + w) * C + c];
                    if (most) acc = x > acc ? x : acc;
                    else acc = __nvvm_add_rn_f(acc, x);
                }
            const float inv = most ? 1.0f : __nvvm_div_rn_f(1.0f, (float)((h1 - h0) * (w1 - w0)));
            v = __nvvm_mul_rn_f(acc, inv);
        } else if (OP == GAP) {
            const uint32_t C = p.in_c, c = i % C, s = i / C, cells = p.in_h * p.in_w, base = s * cells * C + c;
            float acc = F(p.x)[base];
            for (uint32_t q = 1; q < cells; q++) acc = __nvvm_add_rn_f(acc, F(p.x)[base + q * C]);
            v = __nvvm_mul_rn_f(acc, p.inv);
        } else if (OP == ADD || OP == MUL) {
            const bool first = (p.flags & FLAG_FIRST) != 0u;
            const float a = first ? F(p.x)[i] : F(p.y)[i], b = first ? F(p.e0)[i] : F(p.x)[i];
            const float r = OP == ADD ? __nvvm_add_rn_f(a, b) : __nvvm_mul_rn_f(a, b);
            F(p.y)[i] = (p.flags & FLAG_LAST) ? int_act(r, ACT) : r;
            continue;
        } else if (OP == SLICE) {
            const uint32_t c = i % p.in_c, q = i / p.in_c;
            F(p.y)[q * p.out_c + p.c0 + c] = int_act(F(p.x)[i], ACT);
            continue;
        } else if (OP == BN) {
            const uint32_t c = i % p.out_c;
            v = affine_rn(F(p.x)[i], F(p.e0)[c], F(p.e1)[c]);
        } else {                            /* UP */
            const uint32_t C = p.out_c, c = i % C, q = i / C, ox = q % p.out_w, oy = (q / p.out_w) % p.out_h;
            const uint32_t s = q / (p.out_w * p.out_h), base = s * p.in_h * p.in_w;
            if (!(p.flags & FLAG_BILINEAR)) {
                v = F(p.x)[(base + (oy / p.sh) * p.in_w + ox / p.sw) * C + c];
            } else {
                /* spingalett_engine_upsample(): rows y0, y1 weighted hy, ly; columns x0, x1 weighted hx, lx */
                const uint32_t r = 4u * oy, k = 4u * (p.out_h + ox);
                const uint32_t y0 = U(p.e0)[r], y1 = U(p.e0)[r + 1u], x0 = U(p.e0)[k], x1 = U(p.e0)[k + 1u];
                const float ly = F(p.e0)[r + 2u], hy = F(p.e0)[r + 3u], lx = F(p.e0)[k + 2u], hx = F(p.e0)[k + 3u];
                const float a = F(p.x)[(base + y0 * p.in_w + x0) * C + c], b = F(p.x)[(base + y0 * p.in_w + x1) * C + c];
                const float d0 = F(p.x)[(base + y1 * p.in_w + x0) * C + c], d1 = F(p.x)[(base + y1 * p.in_w + x1) * C + c];
                const float top = __nvvm_add_rn_f(__nvvm_mul_rn_f(hx, a), __nvvm_mul_rn_f(lx, b));
                const float bottom = __nvvm_add_rn_f(__nvvm_mul_rn_f(hx, d0), __nvvm_mul_rn_f(lx, d1));
                v = __nvvm_add_rn_f(__nvvm_mul_rn_f(hy, top), __nvvm_mul_rn_f(ly, bottom));
            }
        }
        F(p.y)[i] = int_act(v, ACT);
    }
}
