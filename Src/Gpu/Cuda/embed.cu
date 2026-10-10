/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/* embed.comp: token embeddings, forward, and the table's gradient through a stable sort of the batch's
   tokens (KEYS, then HIST, SCAN and SCATTER per eight bits of the token, then GRAD over each token's run
   of positions in their order). Spec: OP, HALF (words: y 2, dy 6). 256 threads. */

#include "common.cuh"

#define FORWARD 0u
#define KEYS    1u
#define HIST    2u
#define SCAN    3u
#define SCATTER 4u
#define GRAD    5u
#define BLOCK   256u

/* keys and values of the sort: four arrays of `total` uints (keys and positions, read from, written to) */
DEVICE uint32_t *keys_of(const SpgEmbedPush &p, uint32_t which) { return U(p.keys) + (uint64_t)which * p.total; }

KERNEL(embed, SpgEmbedPush) {
    const uint32_t OP = spec.v[0], HALF = spec.v[1];
    __shared__ uint32_t sk[BLOCK], counts[BLOCK];
    const uint32_t tid = thread_x();
    if (OP == FORWARD) {
        /* y[s][p] = row t of the table (zeros outside it) + the position's vector; four channels a thread
           where the rows come in fours */
        const float *w = F(p.w), *pos = F(p.pos), *x = F(p.x);
        const uint32_t d = p.d, quads = d % 4u == 0u ? d / 4u : 0u;
        if (quads) {
            const uint32_t total = p.n * p.tokens * quads;
            for (uint32_t i = global_x(); i < total; i += blocks_x() * 256u) {
                const uint32_t cell = i / quads, c = 4u * (i % quads), at = cell * d + c;
                const float t = x[cell];
                float4_ v = {0.0f, 0.0f, 0.0f, 0.0f};
                if (t >= 0.0f && t < (float)p.vocab) v = ((const float4_ *)w)[((uint64_t)(uint32_t)t * d + c) / 4u];
                if (p.pos) v = add4(v, ((const float4_ *)pos)[((cell % p.tokens) * d + c) / 4u]);
                st4(p.y, at, 2u, HALF, v);
            }
            return;
        }
        const uint32_t total = p.n * p.tokens * d;
        for (uint32_t i = global_x(); i < total; i += blocks_x() * 256u) {
            const uint32_t cell = i / d, c = i % d;
            const float t = x[cell];
            float v = t >= 0.0f && t < (float)p.vocab ? w[(uint64_t)(uint32_t)t * d + c] : 0.0f;
            if (p.pos) v += pos[(cell % p.tokens) * d + c];
            st(p.y, i, 2u, HALF, v);
        }
        return;
    }
    if (OP == KEYS) {
        /* every position's token (the vocabulary for none: sorted last, skipped) and its index */
        for (uint32_t i = global_x(); i < p.total; i += blocks_x() * 256u) {
            const float t = F(p.x)[i];
            keys_of(p, 0)[i] = t >= 0.0f && t < (float)p.vocab ? (uint32_t)t : p.vocab;
            keys_of(p, 1)[i] = i;
        }
        return;
    }
    if (OP == HIST || OP == SCATTER) {
        /* a block of BLOCK keys: how many have each value of the digit, or each key's place: where its
           digit's keys of this block start, plus the keys of that digit before it in the block (stable) */
        const uint32_t from = (p.flags & 1u) ? 2u : 0u, b = block_x(), i = b * BLOCK + tid;
        const bool live = i < p.total;
        const uint32_t key = live ? keys_of(p, from)[i] : 0u, digit = live ? (key >> p.shift) & 255u : 256u;
        sk[tid] = digit;
        barrier();
        if (OP == HIST) {
            uint32_t count = 0;
            for (uint32_t k = 0; k < BLOCK; k++) count += sk[k] == tid ? 1u : 0u;
            U(p.part)[tid * p.blocks + b] = count;
            return;
        }
        if (!live) return;
        uint32_t rank = 0;
        for (uint32_t k = 0; k < tid; k++) rank += sk[k] == digit ? 1u : 0u;
        const uint32_t at = U(p.part)[digit * p.blocks + b] + rank;
        keys_of(p, 2u - from)[at] = key;
        keys_of(p, 3u - from)[at] = keys_of(p, from + 1u)[i];
        return;
    }
    if (OP == SCAN) {
        /* the counts (digit-major, block-minor) into where each digit's keys of each block start: a block,
           each thread a run of them, the runs' sums scanned in shared memory */
        const uint32_t total = 256u * p.blocks, run = (total + BLOCK - 1u) / BLOCK, lo = tid * run;
        const uint32_t hi = lo + run < total ? lo + run : total;
        uint32_t *h = U(p.part), sum = 0;
        for (uint32_t k = lo; k < hi; k++) sum += h[k];
        counts[tid] = sum;
        barrier();
        if (tid == 0u) {
            uint32_t acc = 0;
            for (uint32_t k = 0; k < BLOCK; k++) {
                const uint32_t v = counts[k];
                counts[k] = acc;
                acc += v;
            }
        }
        barrier();
        uint32_t acc = counts[tid];
        for (uint32_t k = lo; k < hi; k++) {
            const uint32_t v = h[k];
            h[k] = acc;
            acc += v;
        }
        return;
    }
    /* GRAD: the first position of each token's run sums the run's rows of dy in their order, four channels
       a thread where the rows come in fours: gw[t] = beta gw[t] + scale sum */
    const uint32_t from = (p.flags & 1u) ? 2u : 0u, d = p.d, quads = d % 4u == 0u ? d / 4u : d;
    const uint32_t *key = keys_of(p, from), *at = keys_of(p, from + 1u);
    const uint32_t total = p.total * quads;
    for (uint32_t i = global_x(); i < total; i += blocks_x() * 256u) {
        const uint32_t j = i / quads, q = i % quads, t = key[j];
        if (t >= p.vocab || (j > 0u && key[j - 1u] == t)) continue;
        if (d % 4u == 0u) {
            float4_ sum = {0.0f, 0.0f, 0.0f, 0.0f};
            for (uint32_t k = j; k < p.total && key[k] == t; k++) sum = add4(sum, ld4(p.dy, at[k] * d + 4u * q, 6u, HALF));
            float4_ *g = (float4_ *)F(p.gw) + ((uint64_t)t * d + 4u * q) / 4u;
            *g = p.beta != 0.0f ? add4(scale4(*g, p.beta), scale4(sum, p.scale)) : scale4(sum, p.scale);
        } else {
            float sum = 0.0f;
            for (uint32_t k = j; k < p.total && key[k] == t; k++) sum += ld(p.dy, at[k] * d + q, 6u, HALF);
            float *g = F(p.gw) + (uint64_t)t * d + q;
            *g = p.beta != 0.0f ? p.beta * *g + p.scale * sum : p.scale * sum;
        }
    }
}
