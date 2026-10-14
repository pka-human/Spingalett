/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/* iquant.comp: the inputs of an integer layer quantized as spingalett_quantize_activations() quantizes them, a block
   of 256 threads a slice of a sample (WHOLE, MAX, QUANT). Spec: OP, HALF (words: q 1). The divisions are IEEE's
   (div.rn), the product rounds alone (mul.rn: no multiply-add), the rounding to integers goes to the nearest, ties to
   even (cvt.rni). */

#include "common.cuh"

#define WHOLE 0u
#define MAX   1u
#define QUANT 2u
#define FLAG_VEC4 1u

struct uint4_ {
    uint32_t x, y, z, w;
} __attribute__((aligned(16)));

DEVICE uint32_t umax(uint32_t a, uint32_t b) { return a > b ? a : b; }

DEVICE float quantized(float x, float inv, bool none) {
    if (none) return 0.0f;
    const float y = __builtin_rintf(__nvvm_mul_rn_f(x, inv));
    return y > 127.0f ? 127.0f : y < -127.0f ? -127.0f : y;
}

KERNEL(iquant, SpgIquantPush) {
    const uint32_t OP = spec.v[0], HALF = spec.v[1], t = thread_x();
    const bool vec4s = (p.flags & FLAG_VEC4) != 0u, half = (HALF >> 1) & 1u;
    __shared__ uint32_t top[256];
    for (uint32_t w = block_x(); w < p.n * p.slices; w += blocks_x()) {
        const uint32_t s = w / p.slices, slice = w % p.slices, base = s * p.stride, dst = s * p.qstride;
        const uint32_t first = slice * p.slice_len, end = umin(p.len, first + p.slice_len);
        uint32_t m = 0u;
        if (OP != QUANT) {
            if (vec4s) {
                for (uint32_t i = first + 4u * t; i < end; i += 1024u) {
                    const uint4_ b = ((const uint4_ *)p.x)[(base + i) >> 2];
                    m = umax(m, umax(umax(b.x & 0x7FFFFFFFu, b.y & 0x7FFFFFFFu), umax(b.z & 0x7FFFFFFFu, b.w & 0x7FFFFFFFu)));
                }
            } else {
                for (uint32_t i = first + t; i < end; i += 256u) m = umax(m, float_bits(F(p.x)[base + i]) & 0x7FFFFFFFu);
            }
            top[t] = m;
            barrier();
            for (uint32_t h = 128u; h > 0u; h >>= 1) {
                if (t < h) top[t] = umax(top[t], top[t + h]);
                barrier();
            }
            m = top[0];
            barrier();                      /* (top is written again by the next slice) */
            if (OP == MAX) {
                if (t == 0u) U(p.part)[w] = m;
                continue;
            }
        } else {
            for (uint32_t k = 0; k < p.slices; k++) m = umax(m, U(p.part)[s * p.slices + k]);
        }
        const bool bad = m >= 0x7F800000u, zero = m == 0u, none = bad || zero;
        const float amax = bits_float(m), inv = none ? 0.0f : __nvvm_div_rn_f(127.0f, amax);
        if (t == 0u && slice == 0u) F(p.xs)[s] = bad ? bits_float(0x7FC00000u) : zero ? 0.0f : __nvvm_div_rn_f(amax, 127.0f);
        if (vec4s) {
            for (uint32_t i = first + 4u * t; i < end; i += 1024u) {
                const float4_ x = ((const float4_ *)p.x)[(base + i) >> 2];
                const float4_ v = {quantized(x.x, inv, none), quantized(x.y, inv, none), quantized(x.z, inv, none),
                                   quantized(x.w, inv, none)};
                if (half)                   /* (integers of 8 bits: exact) */
                    ((uint2_ *)p.q)[(dst + i) >> 2] = uint2_{float_bits(v.x) >> 16 | (float_bits(v.y) & 0xFFFF0000u),
                                                             float_bits(v.z) >> 16 | (float_bits(v.w) & 0xFFFF0000u)};
                else
                    ((float4_ *)p.q)[(dst + i) >> 2] = v;
            }
        } else {
            for (uint32_t i = first + t; i < end; i += 256u) {
                const float v = quantized(F(p.x)[base + i], inv, none);
                if (half) H(p.q)[dst + i] = (uint16_t)(float_bits(v) >> 16);
                else F(p.q)[dst + i] = v;
            }
        }
    }
}
