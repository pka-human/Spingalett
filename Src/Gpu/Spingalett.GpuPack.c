/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/*
 * Weights of GGUF's block types packed for the device (sessions on the GPU: bmmp, bdotp, bembed): the values of the
 * image's blocks rearranged so that the matrix units' fragments and the loads take whole words and sectors, nothing
 * rounded. Q8_0: the codes, N rows of K bytes, then the blocks' scales as floats, N x K / 32. Q4_K: the codes of every
 * sub-block of 32 values in 16 bytes (byte i the code of value i, then of value i + 16 above it: a lane's word gives
 * both halves of a fragment of 32), N rows of K / 2 bytes; the sub-blocks' 6-bit scales and minima in pairs of bytes,
 * N x K / 32; the super-blocks' scale and minimum's negation (as the products take them) as pairs of floats, N x K / 256;
 * Q5_K as Q4_K, then the fifth bits, a word a sub-block (bit i value i's), N x K / 32.
 * Q6_K: the low four bits of the codes as Q4_K's, then their high two bits, eight bytes a sub-block of 32 (sixteen bits
 * a lane's word of low bits: the high bits of its four values of the first half of 16, then of the second's), N rows of
 * K / 4 bytes; the halves' signed scales in pairs of bytes, N x K / 32; the super-blocks' scales (and zeros) as pairs of
 * floats, N x K / 256. Blocks of 32 of 4-bit codes (Q4_0, Q4_1, Q5_0, Q5_1, IQ4_NL): their codes as GGUF keeps them (byte i
 * the code of value i, then of value i + 16), N rows of K / 2 bytes; the blocks' scales as floats (with Q4_1 and
 * Q5_1 pairs of the scale and the minimum), N x K / 32; Q5_0's and Q5_1's fifth bits, a word a block (bit i value
 * i's), N x K / 32.
 */

#include "Spingalett.GpuKernels.h"
#include "Spingalett.Engine.h"
#include <string.h>

/* The 6-bit scale and minimum j of a Q4_K or Q5_K super-block (llama.cpp's get_scale_min_k4()) */
static void scale_min_k4(uint32_t j, const uint8_t *q, uint32_t *d, uint32_t *m) {
    if (j < 4u) {
        *d = q[j] & 63u;
        *m = q[j + 4u] & 63u;
    } else {
        *d = (q[j + 4u] & 0xFu) | ((uint32_t)(q[j - 4u] >> 6) << 4);
        *m = (uint32_t)(q[j + 4u] >> 4) | ((uint32_t)(q[j] >> 6) << 4);
    }
}

static float half_at(const uint8_t *b) { return spingalett_fp16_to_float((uint16_t)(b[0] | b[1] << 8)); }

size_t spg_packed_bytes(uint32_t type, uint32_t N, uint32_t K) {
    switch (type) {
        case SPG_B_Q8_0: return (size_t)N * K + (size_t)N * (K / 32u) * sizeof(float);
        case SPG_B_Q4_K: return (size_t)N * (K / 2u) + (size_t)N * (K / 32u) * 2u + (size_t)N * (K / 256u) * 2u * sizeof(float);
        case SPG_B_Q5_K:
            return (size_t)N * (K / 2u) + (size_t)N * (K / 32u) * 2u + (size_t)N * (K / 256u) * 2u * sizeof(float) +
                   (size_t)N * (K / 32u) * 4u;
        case SPG_B_Q4_0: case SPG_B_IQ4_NL: return (size_t)N * (K / 2u) + (size_t)N * (K / 32u) * sizeof(float);
        case SPG_B_Q4_1: return (size_t)N * (K / 2u) + (size_t)N * (K / 32u) * 2u * sizeof(float);
        case SPG_B_Q5_0: return (size_t)N * (K / 2u) + (size_t)N * (K / 32u) * 4u + (size_t)N * (K / 32u) * sizeof(float);
        case SPG_B_Q5_1: return (size_t)N * (K / 2u) + (size_t)N * (K / 32u) * 4u + (size_t)N * (K / 32u) * 2u * sizeof(float);
        case SPG_B_Q6_K:
            return (size_t)N * (K / 2u) + (size_t)N * (K / 4u) + (size_t)N * (K / 32u) * 2u + (size_t)N * (K / 256u) * 2u * sizeof(float);
        default: return 0;
    }
}

uint32_t spg_packed_cols(uint32_t type) { return type == SPG_B_Q8_0 ? 128u : 64u; }

void spg_pack_blocks(uint32_t type, const uint8_t *src, uint32_t N, uint32_t K, uint8_t *dst) {
    if (type == SPG_B_Q8_0) {
        const uint32_t blocks = K / 32u;
        float *scales = (float *)(void *)(dst + (size_t)N * K);
        for (uint32_t j = 0; j < N; j++)
            for (uint32_t b = 0; b < blocks; b++) {
                const uint8_t *block = src + ((size_t)j * blocks + b) * 34u;
                memcpy(dst + (size_t)j * K + 32u * b, block + 2, 32);
                scales[(size_t)j * blocks + b] = half_at(block);
            }
        return;
    }
    if (type == SPG_B_Q4_0 || type == SPG_B_Q4_1 || type == SPG_B_Q5_0 || type == SPG_B_Q5_1 || type == SPG_B_IQ4_NL) {
        /* (GGUF's blocks: the scale, the minimum (Q4_1, Q5_1), the fifth bits (Q5_0, Q5_1), then the codes) */
        const bool one = type == SPG_B_Q4_1 || type == SPG_B_Q5_1, five = type == SPG_B_Q5_0 || type == SPG_B_Q5_1;
        const uint32_t blocks = K / 32u, bytes = 18u + (one ? 2u : 0u) + (five ? 4u : 0u), qs = bytes - 16u;
        float *scales = (float *)(void *)(dst + (size_t)N * (K / 2u));
        uint8_t *fifth = (uint8_t *)(scales + (size_t)N * blocks * (one ? 2u : 1u));
        for (uint32_t j = 0; j < N; j++)
            for (uint32_t b = 0; b < blocks; b++) {
                const uint8_t *block = src + ((size_t)j * blocks + b) * bytes;
                const size_t at = (size_t)j * blocks + b;
                memcpy(dst + (size_t)j * (K / 2u) + 16u * b, block + qs, 16);
                if (five) memcpy(fifth + 4u * at, block + (one ? 4 : 2), 4);
                if (one) {
                    scales[2u * at] = half_at(block);
                    scales[2u * at + 1u] = half_at(block + 2);
                } else {
                    scales[at] = half_at(block);
                }
            }
        return;
    }
    if (type == SPG_B_Q4_K || type == SPG_B_Q5_K) {
        const bool five = type == SPG_B_Q5_K;
        const uint32_t blocks = K / 256u, subs = K / 32u;
        uint8_t *codes = dst;
        uint16_t *pairs = (uint16_t *)(void *)(dst + (size_t)N * (K / 2u));
        float *dm = (float *)(void *)(dst + (size_t)N * (K / 2u) + (size_t)N * subs * 2u);
        uint32_t *fifth = (uint32_t *)(void *)(dm + (size_t)N * blocks * 2u);
        for (uint32_t j = 0; j < N; j++)
            for (uint32_t b = 0; b < blocks; b++) {
                /* (Q5_K: the high bits of value l of every sub-block in byte 16 + l, the codes from byte 48) */
                const uint8_t *block = src + ((size_t)j * blocks + b) * (five ? 176u : 144u), *qs = block + (five ? 48 : 16);
                dm[2u * ((size_t)j * blocks + b)] = half_at(block);
                dm[2u * ((size_t)j * blocks + b) + 1u] = -half_at(block + 2);
                for (uint32_t c = 0; c < 8u; c++) {
                    uint32_t sc, mn;
                    scale_min_k4(c, block + 4, &sc, &mn);
                    pairs[(size_t)j * subs + 8u * b + c] = (uint16_t)(sc | mn << 8);
                    /* value i of sub-block c: the low (c even) or high nibble of byte 32 (c / 2) + i */
                    const uint8_t *q = qs + 32u * (c / 2u);
                    const uint32_t shift = 4u * (c % 2u);
                    uint8_t *out = codes + (size_t)j * (K / 2u) + 16u * (8u * b + c);
                    for (uint32_t i = 0; i < 16u; i++)
                        out[i] = (uint8_t)(((q[i] >> shift) & 0xFu) | (((q[i + 16u] >> shift) & 0xFu) << 4));
                    if (five) {
                        uint32_t bits = 0;
                        for (uint32_t i = 0; i < 32u; i++) bits |= ((uint32_t)(block[16u + i] >> c) & 1u) << i;
                        fifth[(size_t)j * subs + 8u * b + c] = bits;
                    }
                }
            }
    }
    if (type == SPG_B_Q6_K) {
        const uint32_t blocks = K / 256u, subs = K / 32u;
        uint8_t *lows = dst, *highs = dst + (size_t)N * (K / 2u);
        uint16_t *pairs = (uint16_t *)(void *)(highs + (size_t)N * (K / 4u));
        float *dm = (float *)(void *)((uint8_t *)pairs + (size_t)N * subs * 2u);
        for (uint32_t j = 0; j < N; j++)
            for (uint32_t b = 0; b < blocks; b++) {
                const uint8_t *block = src + ((size_t)j * blocks + b) * 210u;
                const int8_t *sc = (const int8_t *)(block + 192);
                dm[2u * ((size_t)j * blocks + b)] = half_at(block + 208);
                dm[2u * ((size_t)j * blocks + b) + 1u] = 0.0f;
                for (uint32_t c = 0; c < 8u; c++) {
                    /* value i of sub-block c: its low four bits in the low (c % 4 < 2) or high nibble of byte
                       64 (c / 4) + 32 (c % 2) + i, its high two bits at 2 (c % 4) of byte 128 + 32 (c / 4) + i */
                    const uint32_t h = c / 4u, g = c % 4u;
                    const uint8_t *lb = block + 64u * h + 32u * (g % 2u), *hb = block + 128 + 32u * h;
                    uint32_t low[32], high[32];
                    for (uint32_t i = 0; i < 32u; i++) {
                        low[i] = g < 2u ? lb[i] & 0xFu : (uint32_t)(lb[i] >> 4);
                        high[i] = (hb[i] >> (2u * g)) & 3u;
                    }
                    const size_t sub = (size_t)j * subs + 8u * b + c;
                    uint8_t *out = lows + 16u * sub;
                    for (uint32_t i = 0; i < 16u; i++) out[i] = (uint8_t)(low[i] | low[i + 16u] << 4);
                    for (uint32_t t = 0; t < 4u; t++) {
                        uint32_t piece = 0;
                        for (uint32_t k = 0; k < 4u; k++) piece |= high[4u * t + k] << (2u * k) | high[16u + 4u * t + k] << (8u + 2u * k);
                        highs[8u * sub + 2u * t] = (uint8_t)piece;
                        highs[8u * sub + 2u * t + 1u] = (uint8_t)(piece >> 8);
                    }
                    pairs[sub] = (uint16_t)((uint8_t)sc[2u * c] | (uint32_t)(uint8_t)sc[2u * c + 1u] << 8);
                }
            }
    }
}
