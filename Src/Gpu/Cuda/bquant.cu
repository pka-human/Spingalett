/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/* bquant.comp: rows of activations quantized in blocks of BV values as spingalett_quantize_blocks() quantizes them,
   the bits of the CPU: a block's largest magnitude (or NaN's and infinities: codes 0, a scale NaN; all zeros: codes
   0, a scale 0), the codes rint(x * (127 / amax)) (div.rn, then mul.rn: no multiply-add, cvt.rni: to the nearest,
   ties to even) within -127 to 127, the scale amax / 127, the sums of every 16 codes. A lane four consecutive values
   at a time (a load of 16 bytes where the rows allow it), BV / 4 lanes a block up to a warp (blocks of 256: a lane's
   second four 128 values on), so that a warp reads 512 consecutive bytes; 256 threads a workgroup, SPG_BQUANT_BLOCKS
   blocks. */

#include "common.cuh"

DEVICE uint32_t shfl_max(uint32_t v, uint32_t mask) {
    const uint32_t o = __nvvm_shfl_sync_bfly_i32(0xFFFFFFFFu, v, mask, 0x1F);
    return v > o ? v : o;
}
DEVICE int32_t shfl_add(int32_t v, uint32_t mask) { return v + (int32_t)__nvvm_shfl_sync_bfly_i32(0xFFFFFFFFu, v, mask, 0x1F); }

KERNEL(bquant, SpgBquantPush) {
    const uint32_t lane = thread_x() & 31u, blocks = p.k / p.bv;
    const uint32_t lb = p.bv >= 128u ? 32u : p.bv / 4u, fours = p.bv / 4u / lb;      /* lanes a block, fours a lane */
    const uint32_t w = (block_x() * 8u + (thread_x() >> 5)) * (32u / lb) + lane / lb, l = lane % lb;
    /* (lanes past the last block load its first and store nothing: the shuffles take whole warps) */
    const bool live = w < p.rows * blocks;
    const uint32_t r = live ? w / blocks : 0u, b = live ? w % blocks : 0u;
    const float *x = F(p.x) + (uint64_t)r * p.ldx + b * p.bv;
    const bool aligned = ((p.x | (uint64_t)p.ldx * 4u | p.bv * 4u) & 15u) == 0u;
    float v[2][4];
    uint32_t m = 0u;
    for (uint32_t f = 0; f < fours; f++) {
        const uint32_t at = 4u * (l + lb * f);
        if (aligned) {
            const float4_ q = *(const float4_ *)(x + at);
            v[f][0] = q.x, v[f][1] = q.y, v[f][2] = q.z, v[f][3] = q.w;
        } else
            for (uint32_t i = 0; i < 4u; i++) v[f][i] = x[at + i];
        for (uint32_t i = 0; i < 4u; i++) {
            const uint32_t a = float_bits(v[f][i]) & 0x7FFFFFFFu;
            m = a > m ? a : m;
        }
    }
    /* (the largest magnitude, whatever the order; NaN's bits above infinity's) */
    for (uint32_t h = lb / 2u; h > 0u; h >>= 1) m = shfl_max(m, h);
    const bool bad = m >= 0x7F800000u, zero = m == 0u;
    const float amax = bits_float(m), inv = bad || zero ? 0.0f : __nvvm_div_rn_f(127.0f, amax);
    signed char *q = (signed char *)p.q;
    int32_t *xs = (int32_t *)p.xs + (uint64_t)r * (p.k / 16u) + b * (p.bv / 16u);
    for (uint32_t f = 0; f < fours; f++) {
        const uint32_t at = 4u * (l + lb * f);
        int32_t sum = 0;
        uint32_t codes = 0u;
        for (uint32_t i = 0; i < 4u; i++) {
            float y = bad || zero ? 0.0f : __builtin_rintf(__nvvm_mul_rn_f(v[f][i], inv));
            y = y > 127.0f ? 127.0f : y < -127.0f ? -127.0f : y;
            codes |= ((uint32_t)(int32_t)y & 0xFFu) << (8u * i);
            sum += (int32_t)y;
        }
        /* the sums of 16 codes: of four lanes' */
        sum = shfl_add(sum, 1u);
        sum = shfl_add(sum, 2u);
        if (live) {
            /* (four codes in one 64 of a tile's row) */
            const uint32_t j = b * p.bv + at;
            *(uint32_t *)(q + (p.tiled ? SPG_TILED_AT(r, j, p.k) : (uint64_t)r * p.k + j)) = codes;
            if ((l & 3u) == 0u) xs[at >> 4] = sum;
        }
    }
    if (live && l == 0u) F(p.dx)[(uint64_t)r * blocks + b] = bad ? bits_float(0x7FC00000u) : zero ? 0.0f : __nvvm_div_rn_f(amax, 127.0f);
}
