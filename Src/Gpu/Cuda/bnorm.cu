/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/* bnorm.comp: a session's RMS normalization of rows, with the sum of two layers before it and the quantization of its
   outputs after it, in one pass: s = a + b (with b; written to sum), y = s / sqrt(mean(s^2) + eps) times gamma as
   grow.cu's RMSNORM computes it (TPR threads a row, spec 0, each its values l = t, t + TPR, ... in order, the
   partial sums of squares combined in grow's tree: its bits), then with bv the outputs quantized in blocks of bv
   values as bquant.cu quantizes them (its bits: a block's largest magnitude, rint(y (127 / amax)) within -127 to 127,
   the scale amax / 127, the sums of 16 codes). 256 threads, 256 / TPR rows a workgroup, TPR from 32 to 256, VMAX
   values a thread at most (spec 1, a unit each: bnorm_v<VMAX>). */

#include "common.cuh"

#if !defined(SPG_ENTRY)
#define SPG_ENTRY spg_bnorm_v8
#define VMAX 8u
#endif
#define THREADS 256u

DEVICE uint32_t shfl_max(uint32_t v, uint32_t mask) {
    const uint32_t o = __nvvm_shfl_sync_bfly_i32(0xFFFFFFFFu, v, mask, 0x1F);
    return v > o ? v : o;
}
DEVICE int32_t shfl_add(int32_t v, uint32_t mask) { return v + (int32_t)__nvvm_shfl_sync_bfly_i32(0xFFFFFFFFu, v, mask, 0x1F); }

extern "C" __global__ void __launch_bounds__(256) SPG_ENTRY(const SpgBnormPush p, const Spec spec) {
    __shared__ float part[THREADS];
    __shared__ uint32_t peak[THREADS * VMAX / 32u];     /* the magnitudes' maxima of every 32 values of the rows */
    const uint32_t TPR = spec.v[0], per = THREADS / TPR, t = thread_x(), lane = t % TPR;
    const uint32_t r = block_x() * per + t / TPR, n = p.k / TPR;
    const bool live = r < p.rows;
    float v[VMAX];
    float sq = 0.0f;
#pragma unroll
    for (uint32_t i = 0; i < VMAX; i++) {
        if (i >= n) continue;
        const uint64_t at = (uint64_t)(live ? r : 0u) * p.k + lane + TPR * i;
        float s = live ? F(p.a)[at] : 0.0f;
        if (p.b) {
            s = s + (live ? F(p.b)[at] : 0.0f);
            if (live) F(p.sum)[at] = s;
        }
        v[i] = s;
        sq += s * s;
    }
    /* grow's tree */
    part[t] = sq;
    for (uint32_t h = TPR / 2u; h > 0u; h >>= 1) {
        barrier();
        if (lane < h) part[t] = part[t] + part[t + h];
    }
    barrier();
    const float var = part[t - lane] / (float)p.k, inv = 1.0f / sqrt_(var + p.eps);
#pragma unroll
    for (uint32_t i = 0; i < VMAX; i++) {
        if (i >= n) continue;
        const uint32_t l = lane + TPR * i;
        v[i] = (v[i] - 0.0f) * inv * F(p.gamma)[l] + 0.0f;
        if (live) F(p.y)[(uint64_t)r * p.k + l] = v[i];
    }
    if (!p.bv) return;
    /* the largest magnitude of every 32 values (a warp's for one i), then of every block (bv / 32 of them) */
    const uint32_t groups = p.k / 32u, row0 = (t / TPR) * groups;
#pragma unroll
    for (uint32_t i = 0; i < VMAX; i++) {
        if (i >= n) continue;
        uint32_t m = float_bits(v[i]) & 0x7FFFFFFFu;
        for (uint32_t h = 16u; h > 0u; h >>= 1) m = shfl_max(m, h);
        if (t % 32u == 0u) peak[row0 + (lane + TPR * i) / 32u] = m;
    }
    barrier();
    if (!live) return;
    const uint32_t blocks = p.k / p.bv;
#pragma unroll
    for (uint32_t i = 0; i < VMAX; i++) {
        if (i >= n) continue;
        const uint32_t l = lane + TPR * i, b = l / p.bv;
        uint32_t m = 0u;
        for (uint32_t g = b * (p.bv / 32u); g < (b + 1u) * (p.bv / 32u); g++) m = peak[row0 + g] > m ? peak[row0 + g] : m;
        const bool bad = m >= 0x7F800000u, zero = m == 0u;
        const float amax = bits_float(m), qi = bad || zero ? 0.0f : __nvvm_div_rn_f(127.0f, amax);
        float y = bad || zero ? 0.0f : __builtin_rintf(__nvvm_mul_rn_f(v[i], qi));
        y = y > 127.0f ? 127.0f : y < -127.0f ? -127.0f : y;
        ((signed char *)p.q)[(uint64_t)r * p.k + l] = (signed char)(int32_t)y;
        int32_t sum = (int32_t)y;
        for (uint32_t h = 8u; h > 0u; h >>= 1) sum = shfl_add(sum, h);
        if (l % 16u == 0u) ((int32_t *)p.xs)[(uint64_t)r * (p.k / 16u) + l / 16u] = sum;
        if (l % p.bv == 0u)
            F(p.dx)[(uint64_t)r * blocks + b] = bad ? bits_float(0x7FC00000u) : zero ? 0.0f : __nvvm_div_rn_f(amax, 127.0f);
    }
}
