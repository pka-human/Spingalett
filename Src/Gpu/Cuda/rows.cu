/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/* rows.comp: rows of a data set in the GPU's memory gathered into a chunk, inputs augmented and targets
   smoothed as the host does it. Spec: WHERE, VEC, HALF (word: dst 1). 256 threads, a grid-stride loop. */

#include "common.cuh"

/* h mod m, for m < 2^24, as rows.comp computes it */
DEVICE uint32_t mod64(uint64_t h, uint32_t m) {
    const uint32_t lo = (uint32_t)h;
    uint32_t r = (uint32_t)(h >> 32) % m;
    for (int b = 24; b >= 0; b -= 8) r = (r * 256u + ((lo >> (uint32_t)b) & 255u)) % m;
    return r;
}

KERNEL(rows, SpgRowsPush) {
    const uint32_t WHERE = spec.v[0], VEC = spec.v[1], HALF = spec.v[2];
    const uint32_t *header = U(p.header);
    const uint64_t base = header64(header, WHERE);
    const bool targets = WHERE == STEP_TARGETS;
    const float keep = bits_float(header[STEP_KEEP]), share = bits_float(header[STEP_SHARE]);
    const uint32_t augment = targets ? 0u : header[STEP_AUGMENT], k = augment & 0x7FFFFFFFu;
    const uint64_t seed = header64(header, STEP_AUGMENT_LO), step = header64(header, STEP_STEP_LO);
    const uint32_t units = p.size / VEC, total = p.n * units, stride = blocks_x() * 256u, WC = p.width * p.channels;
    const float *set = F(base);
    for (uint32_t u = global_x(); u < total; u += stride) {
        const uint32_t s = u / units, j = (u - s * units) * VEC, row = U(p.index)[s];
        const uint64_t at = (uint64_t)row * p.size;
        float4_ v = {0.0f, 0.0f, 0.0f, 0.0f};
        if (augment == 0u) {
            if (VEC == 4u) v = *(const float4_ *)(set + at + j);
            else v.x = set[at + j];
        } else {
            const uint64_t h = mix64(seed ^ mix64(step * 0x9E3779B97F4A7C15ull + (header[STEP_POSITION] + s)));
            const int dy = k != 0u ? (int)mod64(h, 2u * k + 1u) - (int)k : 0;
            const int dx = k != 0u ? (int)mod64(h >> 21, 2u * k + 1u) - (int)k : 0;
            const bool mirror = (augment >> 31) != 0u && ((uint32_t)(h >> 32) >> 10 & 1u) != 0u;
            for (uint32_t e = 0u; e < VEC; e++) {
                const uint32_t y = (j + e) / WC, x = (j + e - y * WC) / p.channels, c = j + e - y * WC - x * p.channels;
                const int sy = (int)y + dy, sx = (int)(mirror ? p.width - 1u - x : x) + dx;
                if (sy >= 0 && sy < (int)p.height && sx >= 0 && sx < (int)p.width)
                    set4(v, e, set[at + ((uint32_t)sy * p.width + (uint32_t)sx) * p.channels + c]);
            }
        }
        float4_ w = v;
        if (targets && keep != 0.0f) w = float4_{keep * v.x + share, keep * v.y + share, keep * v.z + share, keep * v.w + share};
        const uint32_t i = s * p.size + j;
        if (VEC == 1u) {
            st(p.dst, i, 1u, HALF, w.x);
            continue;
        }
        st4(p.dst, i, 1u, HALF, w);
    }
}
