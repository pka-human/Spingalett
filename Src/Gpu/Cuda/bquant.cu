/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/* bquant.comp: rows of activations quantized in blocks of BV values as spingalett_quantize_blocks() quantizes them,
   the bits of the CPU: a block's largest magnitude (or NaN's and infinities: codes 0, a scale NaN; all zeros: codes
   0, a scale 0), the codes rint(x * (127 / amax)) (div.rn, then mul.rn: no multiply-add, cvt.rni: to the nearest,
   ties to even) within -127 to 127, the scale amax / 127, the sums of every 16 codes. A warp a block, BV / 32 values
   a lane, consecutive; 256 threads a workgroup. */

#include "common.cuh"

DEVICE uint32_t shfl_max(uint32_t v, uint32_t mask) {
    const uint32_t o = __nvvm_shfl_sync_bfly_i32(0xFFFFFFFFu, v, mask, 0x1F);
    return v > o ? v : o;
}
DEVICE int32_t shfl_add(int32_t v, uint32_t mask) { return v + (int32_t)__nvvm_shfl_sync_bfly_i32(0xFFFFFFFFu, v, mask, 0x1F); }

KERNEL(bquant, SpgBquantPush) {
    const uint32_t lane = thread_x() & 31u, blocks = p.k / p.bv, per = p.bv / 32u;
    const uint32_t w = block_x() * 8u + (thread_x() >> 5);
    if (w >= p.rows * blocks) return;      /* (a whole warp) */
    const uint32_t r = w / blocks, b = w % blocks, at = b * p.bv + lane * per;
    const float *x = F(p.x) + (uint64_t)r * p.ldx + at;
    float v[8];
    uint32_t m = 0u;
    for (uint32_t i = 0; i < per; i++) {
        v[i] = x[i];
        const uint32_t a = float_bits(v[i]) & 0x7FFFFFFFu;
        m = a > m ? a : m;
    }
    /* (the largest magnitude, whatever the order; NaN's bits above infinity's) */
    for (uint32_t h = 16u; h > 0u; h >>= 1) m = shfl_max(m, h);
    const bool bad = m >= 0x7F800000u, zero = m == 0u;
    const float amax = bits_float(m), inv = bad || zero ? 0.0f : __nvvm_div_rn_f(127.0f, amax);
    signed char *q = (signed char *)p.q + (uint64_t)r * p.k + at;
    int32_t sum = 0;
    for (uint32_t i = 0; i < per; i++) {
        float y = bad || zero ? 0.0f : __builtin_rintf(__nvvm_mul_rn_f(v[i], inv));
        y = y > 127.0f ? 127.0f : y < -127.0f ? -127.0f : y;
        q[i] = (signed char)(int32_t)y;
        sum += (int32_t)y;
    }
    if (lane == 0u) F(p.dx)[(uint64_t)r * blocks + b] = bad ? bits_float(0x7FC00000u) : zero ? 0.0f : __nvvm_div_rn_f(amax, 127.0f);
    /* the sums of 16 codes: of 16 lanes (a value each), or of two (eight each) */
    if (per == 1u)
        for (uint32_t h = 8u; h > 0u; h >>= 1) sum = shfl_add(sum, h);
    else
        sum = shfl_add(sum, 1u);
    const uint32_t lanes16 = 16u / per;                 /* lanes a sum of 16 */
    if (lane % lanes16 == 0u) ((int32_t *)p.xs)[(uint64_t)r * (p.k / 16u) + (at >> 4)] = sum;
}
