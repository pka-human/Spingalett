/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/*
 * attn_cache.comp: attention a few rows at a time over the keys and values of the tokens before them, kept per
 * sequence (decoding with a key-value cache: sessions, spingalett_generate()). A row is one token of a sequence,
 * at a position; x holds each row's queries, keys and values as an attention layer's input does (heads + 2 kv
 * heads of d values), y its outputs (heads of d values), the cache the keys then the values of every
 * sequence's positions ([sequences][length][kv][d] each). Spec: OP, THREADS, HALF (words: x 0, y 1, cache 2).
 *
 *   APPEND  each row's keys (rotated at its position with ROPE) and values into the cache at its sequence and
 *           position; a thread a pair of values (c and c + d / 2: the rotation's pairs)
 *   ATTEND  each row's queries against its sequence's cached keys up to its own position (causal by
 *           construction), the softmax online, the values' sum; a block a row and head, its warps tiles of 32
 *           positions in turn (a lane a position's score), their parts added in the warps' order
 *
 * Every sum runs in a fixed order: the same bits on every run.
 */

#include "common.cuh"

#define APPEND 0u
#define ATTEND 1u
#define FLAG_ROPE 2u

DEVICE float shfl_xor(float v, uint32_t mask) { return __nvvm_shfl_sync_bfly_f32(0xFFFFFFFFu, v, mask, 0x1F); }

KERNEL(attn_cache, SpgCachePush) {
    const uint32_t OP = spec.v[0], HALF = spec.v[2];
    const uint32_t d = p.d, C = (p.heads + 2u * p.kv) * d, half_d = d / 2u;
    const uint64_t values = (uint64_t)p.sequences * p.length * p.kv * d;   /* where the values start */
    const bool rope = (p.flags & FLAG_ROPE) != 0u;
    if (OP == APPEND) {
        const uint32_t pairs = p.kv * half_d, total = p.n * pairs;
        for (uint32_t i = global_x(); i < total; i += blocks_x() * threads_x()) {
            const uint32_t r = i / pairs, kh = i % pairs / half_d, c = i % half_d;
            const uint32_t seq = U(p.rows)[2u * r], pos = U(p.rows)[2u * r + 1u];
            const uint32_t src = r * C + (p.heads + kh) * d;
            float k0 = ld(p.x, src + c, 0u, HALF), k1 = ld(p.x, src + c + half_d, 0u, HALF);
            if (rope) {
                const float cs = F(p.table)[pos * half_d + c], sn = F(p.table)[p.cells * half_d + pos * half_d + c];
                const float a = k0, b = k1;
                k0 = a * cs - b * sn;
                k1 = b * cs + a * sn;
            }
            const uint64_t at = (((uint64_t)seq * p.length + pos) * p.kv + kh) * d;
            st(p.cache, (uint32_t)(at + c), 2u, HALF, k0);
            st(p.cache, (uint32_t)(at + c + half_d), 2u, HALF, k1);
            st(p.cache, (uint32_t)(values + at + c), 2u, HALF, ld(p.x, src + p.kv * d + c, 0u, HALF));
            st(p.cache, (uint32_t)(values + at + c + half_d), 2u, HALF, ld(p.x, src + p.kv * d + c + half_d, 0u, HALF));
        }
        return;
    }
    /* ATTEND: a block a row and head (WARPS warps); the positions up to the row's in tiles of 32, tile t of the
       warp t % WARPS: lane j scores position j of the tile alone (the query from shared memory), the tile's
       maximum and sum over the warp, then each lane its channels' values (d / 32 of them: c = lane + 32 j), each
       position's weight passed from its lane; the warps' parts (maximum, sum, values) added in the warps' order */
    __shared__ float sq[256], sm[4], sl[4], so[4][256];
    const uint32_t lane = thread_x() % 32u, w = thread_x() / 32u, WARPS = threads_x() / 32u;
    const uint32_t r = block_x(), h = block_y();
    if (h >= p.heads || r >= p.n) return;
    const uint32_t seq = U(p.rows)[2u * r], pos = U(p.rows)[2u * r + 1u], kh = h / (p.heads / p.kv);
    for (uint32_t c = thread_x(); c < d; c += threads_x()) {
        float v = ld(p.x, r * C + h * d + c, 0u, HALF);
        if (rope) {
            const uint32_t pc = c < half_d ? c + half_d : c - half_d, ci = c < half_d ? c : c - half_d;
            const float x2 = ld(p.x, r * C + h * d + pc, 0u, HALF);
            const float cs = F(p.table)[pos * half_d + ci], sn = F(p.table)[p.cells * half_d + pos * half_d + ci];
            v = c < half_d ? v * cs - x2 * sn : v * cs + x2 * sn;
        }
        sq[c] = v;
    }
    barrier();
    const uint32_t per = (d + 31u) / 32u;   /* channels a lane: c = lane + 32 j < d (heads of up to 128) */
    float m = -3.402823466e38f, l = 0.0f, o[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const uint64_t base = (uint64_t)seq * p.length * p.kv * d;
    for (uint32_t t0 = 32u * w; t0 <= pos; t0 += 32u * WARPS) {
        const uint32_t t = t0 + lane;
        float s = -3.402823466e38f;
        if (t <= pos) {
            const uint64_t at = base + ((uint64_t)t * p.kv + kh) * d;
            float acc = 0.0f;
            for (uint32_t c = 0; c < d; c++) acc += sq[c] * ld(p.cache, (uint32_t)(at + c), 2u, HALF);
            s = acc * p.scale;
        }
        float tm = s;
        for (uint32_t mask = 16u; mask >= 1u; mask >>= 1) tm = max_(tm, shfl_xor(tm, mask));
        const float nm = max_(m, tm), a = exp_(m - nm), e = t <= pos ? exp_(s - nm) : 0.0f;
        float ts = e;
        for (uint32_t mask = 16u; mask >= 1u; mask >>= 1) ts += shfl_xor(ts, mask);
        l = l * a + ts;
        for (uint32_t j = 0; j < per && j < 4u; j++) o[j] *= a;
        const uint32_t count = pos + 1u - t0 < 32u ? pos + 1u - t0 : 32u;
        for (uint32_t k = 0; k < count; k++) {
            const float pk = __nvvm_shfl_sync_idx_f32(0xFFFFFFFFu, e, k, 0x1F);
            const uint64_t at = base + ((uint64_t)(t0 + k) * p.kv + kh) * d;
            for (uint32_t j = 0; j < per && j < 4u; j++)
                if (lane + 32u * j < d) o[j] += pk * ld(p.cache, (uint32_t)(values + at + lane + 32u * j), 2u, HALF);
        }
        m = nm;
    }
    if (lane == 0u) sm[w] = m, sl[w] = l;
    for (uint32_t j = 0; j < per && j < 4u; j++)
        if (lane + 32u * j < d) so[w][lane + 32u * j] = o[j];
    barrier();
    if (w != 0u) return;
    float M = sm[0];
    for (uint32_t k = 1; k < WARPS; k++) M = max_(M, sm[k]);
    float L = 0.0f, out[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint32_t k = 0; k < WARPS; k++) {
        const float f = sl[k] > 0.0f ? exp_(sm[k] - M) : 0.0f;
        L += sl[k] * f;
        for (uint32_t j = 0; j < per && j < 4u; j++)
            if (lane + 32u * j < d) out[j] += so[k][lane + 32u * j] * f;
    }
    const float inv = 1.0f / L;
    for (uint32_t j = 0; j < per && j < 4u; j++)
        if (lane + 32u * j < d) st(p.y, r * p.heads * d + h * d + lane + 32u * j, 1u, HALF, out[j] * inv);
}
