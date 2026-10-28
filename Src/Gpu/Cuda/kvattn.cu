/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/*
 * kvattn.comp: a session's attention over the keys and values it keeps, in single precision (the CPU's to rounding).
 * PREP: a warp a row and head (queries' heads first, then the keys' and values'): the head's values through shared
 * memory, RMS-normalized by its gamma and rotated (value i with value i + rot / 2, by the table's cosine and sine of the
 * row's position) as spingalett_engine_head() does; queries to q (half_q: as bfloat16), keys and values into their
 * caches. ATTEND: a warp a
 * row and query head: the keys split between its eight warps, a lane values d = l, l + 32, ... of the query, a key and
 * a value (each key's score a sum over the warp's lanes), each warp's softmax online (the running maximum rescales
 * what came before, as spingalett_engine_attend() does), then the warps' sums combined by their maxima. 256 threads a
 * workgroup, heads of 256 values at most.
 */

#include "common.cuh"

#define WARPS 8u                    /* PREP: rows and heads a workgroup; ATTEND: warps a row and head */
#define HMAX 256u

DEVICE float shfl_xor(float v, uint32_t mask) { return __nvvm_shfl_sync_bfly_f32(0xFFFFFFFFu, v, mask, 0x1F); }
DEVICE void warp_sync() { __nvvm_bar_warp_sync(0xFFFFFFFFu); }

DEVICE float warp_sum(float v) {
    for (uint32_t h = 16u; h > 0u; h >>= 1) v += shfl_xor(v, h);
    return v;
}
DEVICE float warp_max(float v) {
    for (uint32_t h = 16u; h > 0u; h >>= 1) v = max_(v, shfl_xor(v, h));
    return v;
}

KERNEL(kvattn, SpgKvattnPush) {
    __shared__ float vec[WARPS][HMAX];
    __shared__ float weight[WARPS][32];
    const uint32_t lane = thread_x() & 31u, wi = thread_x() >> 5, w = block_x() * WARPS + wi;
    float *v = vec[wi];
    const uint32_t D = p.head, C = (p.heads + 2u * p.kv) * D, half = p.rot / 2u;
    if (p.op == SPG_KV_PREP) {
        const uint32_t per = p.heads + p.kv;
        if (w >= p.rows * per) return;
        const uint32_t r = w / per, h = w % per, seq = U(p.info)[2u * r], pos = U(p.info)[2u * r + 1u];
        const bool query = h < p.heads;
        const float *src = F(p.x) + (uint64_t)r * C + (uint64_t)h * D;
        for (uint32_t i = lane; i < D; i += 32u) v[i] = src[i];
        warp_sync();
        const uint64_t gamma = query ? p.gq : p.gk;
        if (gamma) {
            float s = 0.0f;
            for (uint32_t i = lane; i < D; i += 32u) s += v[i] * v[i];
            s = warp_sum(s);
            const float rr = 1.0f / sqrt_(s * (1.0f / (float)D) + p.eps);
            for (uint32_t i = lane; i < D; i += 32u) v[i] = v[i] * rr * F(gamma)[i];
            warp_sync();
        }
        if (half) {
            const float *cs = F(p.table) + (uint64_t)pos * half, *sn = F(p.table) + ((uint64_t)p.max_length + pos) * half;
            for (uint32_t i = lane; i < half; i += 32u) {
                const float a = v[i], b = v[i + half], c = cs[i], s = sn[i];
                v[i] = a * c - b * s;
                v[i + half] = b * c + a * s;
            }
            warp_sync();
        }
        if (query) {
            const uint64_t at = ((uint64_t)r * p.heads + h) * D;
            for (uint32_t i = lane; i < D; i += 32u) {
                if (p.half_q) H(p.q)[at + i] = (uint16_t)to_bf16(v[i]);
                else F(p.q)[at + i] = v[i];
            }
            return;
        }
        const uint32_t g = h - p.heads;
        const uint64_t at = (((uint64_t)seq * p.kv + g) * p.max_length + pos) * D;
        const float *vs = F(p.x) + (uint64_t)r * C + (uint64_t)(p.heads + p.kv + g) * D;
        for (uint32_t i = lane; i < D; i += 32u) {
            if (p.half) {
                H(p.keys)[at + i] = (uint16_t)to_bf16(v[i]);
                H(p.values)[at + i] = (uint16_t)to_bf16(vs[i]);
            } else {
                F(p.keys)[at + i] = v[i];
                F(p.values)[at + i] = vs[i];
            }
        }
        return;
    }
    /* ATTEND: a workgroup a row and query head, its warp wi the keys j = first + wi, + WARPS, ...: a lane the values
       d = lane, lane + 32, ... of the query, keys and values; each warp's softmax online, then the warps' combined */
    __shared__ float wm[WARPS], wl[WARPS];
    __shared__ float wo[WARPS][HMAX];
    const uint32_t rh = block_x();
    if (rh >= p.rows * p.heads) return;
    const uint32_t r = rh / p.heads, h = rh % p.heads, g = h / (p.heads / p.kv), per_lane = (D + 31u) / 32u;
    const uint32_t seq = U(p.info)[2u * r], pos = U(p.info)[2u * r + 1u];
    const uint32_t count = p.window && pos + 1u > p.window ? p.window : pos + 1u, first = pos + 1u - count;
    const float *q = F(p.q) + ((uint64_t)r * p.heads + h) * D;
    float qv[HMAX / 32u], o[HMAX / 32u];
    for (uint32_t k = 0; k < HMAX / 32u; k++) {
        const uint32_t d = lane + 32u * k;
        qv[k] = k < per_lane && d < D ? q[d] : 0.0f;
        o[k] = 0.0f;
    }
    const uint64_t base = ((uint64_t)seq * p.kv + g) * p.max_length * D;
    const float *keys = F(p.keys) + base, *values = F(p.values) + base;
    float m = -3.402823466e38f, sum = 0.0f;
    for (uint32_t j = first + wi; j <= pos; j += WARPS) {
        const float *kk = keys + (uint64_t)j * D, *vv = values + (uint64_t)j * D;
        float s = 0.0f;
        for (uint32_t k = 0; k < per_lane; k++) {
            const uint32_t d = lane + 32u * k;
            if (d < D) s += qv[k] * kk[d];
        }
        const float score = warp_sum(s) * p.scale;
        if (score > m) {
            const float rescale = exp_(m - score);
            sum *= rescale;
            for (uint32_t k = 0; k < per_lane; k++) o[k] *= rescale;
            m = score;
        }
        const float e = exp_(score - m);
        sum += e;
        for (uint32_t k = 0; k < per_lane; k++) {
            const uint32_t d = lane + 32u * k;
            if (d < D) o[k] += e * vv[d];
        }
    }
    if (lane == 0u) wm[wi] = m, wl[wi] = sum;
    for (uint32_t k = 0; k < per_lane; k++) {
        const uint32_t d = lane + 32u * k;
        if (d < D) wo[wi][d] = o[k];
    }
    barrier();
    if (wi != 0u) return;
    float top = wm[0];
    for (uint32_t w2 = 1; w2 < WARPS; w2++) top = max_(top, wm[w2]);
    float total = 0.0f, f[WARPS];
    for (uint32_t w2 = 0; w2 < WARPS; w2++) {
        f[w2] = wl[w2] > 0.0f ? exp_(wm[w2] - top) : 0.0f;
        total += wl[w2] * f[w2];
    }
    const float inv = total > 0.0f ? 1.0f / total : 0.0f;
    float *y = F(p.y) + (uint64_t)r * p.heads * D + (uint64_t)h * D;
    for (uint32_t d = lane; d < D; d += 32u) {
        float acc = 0.0f;
        for (uint32_t w2 = 0; w2 < WARPS; w2++) acc += wo[w2][d] * f[w2];
        y[d] = acc * inv;
    }
}
