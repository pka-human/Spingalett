/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/*
 * Weights of GGUF's block types packed for the device (sessions on the GPU: bmmp, bdotp, bembed): the values of the
 * image's blocks rearranged so that the matrix units' fragments and the loads take whole words and sectors, nothing
 * rounded, every row in one piece (a warp's loads of a row one stream of memory): its codes, then the rest of its
 * blocks (SPG_PACK_CODES(), SPG_PACK_AUX(), SPG_PACK_ROW()).
 *
 * Codes: Q8_0's as they are; 4-bit ones in sixteen bytes a sub-block of 32 values, byte i the code of value i, then of
 * value i + 16 above it (GGUF's order of Q4_0, IQ4_NL, IQ4_XS and of their fifth bits; Q4_K's, Q5_K's and Q6_K's low
 * bits rearranged so): a lane's word gives both halves of a fragment of 32 by a mask and a shift; Q6_K's high two bits
 * then, sixteen bytes a pair of sub-blocks, a word a lane t: its byte k the high bits of value 4t + k of the first
 * sub-block (bits 0 and 1), of its value 16 + 4t + k (2, 3), then of the second sub-block's (4, 5 and 6, 7), so that a
 * shift and a mask put each fragment's at bits 4 and 5 of its bytes.
 * The rest, in order: Q8_0, Q4_0, IQ4_NL: the blocks' scales (float16); Q4_1: the scale and the minimum; Q5_0: the
 * fifth bits, a word a block (bit i value i's), then the scales; Q5_1: the fifth bits, then the scales and minima;
 * Q4_K: the sub-blocks' 6-bit scale and minimum in a pair of bytes, then the super-blocks' scale and minimum (float16);
 * Q5_K: the fifth bits, a word a sub-block, then Q4_K's; IQ4_XS: the sub-blocks' signed scales in pairs of bytes (the
 * second 0), then the super-blocks' scale (and 0); Q6_K: the halves' signed scales in pairs of bytes, then the
 * super-blocks' scales.
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

size_t spg_packed_bytes(uint32_t type, uint32_t N, uint32_t K) { return (size_t)N * SPG_PACK_ROW(type, K); }

uint32_t spg_packed_cols(uint32_t type) { return type == SPG_B_Q8_0 ? 128u : 64u; }

/* sixteen bytes of 32 4-bit codes: byte i the code of value i, then of value i + 16 above it */
static void nibbles(uint8_t *out, const uint32_t *code) {
    for (uint32_t i = 0; i < 16u; i++) out[i] = (uint8_t)(code[i] | code[i + 16u] << 4);
}

void spg_pack_blocks(uint32_t type, const uint8_t *src, uint32_t N, uint32_t K, uint8_t *dst) {
    const size_t row = SPG_PACK_ROW(type, K), codes = SPG_PACK_CODES(type, K);
    if (!row) return;
    const uint32_t B = K / 32u, SBK = K / 256u;
    memset(dst, 0, (size_t)N * row);
    for (uint32_t j = 0; j < N; j++) {
        uint8_t *r = dst + (size_t)j * row, *aux = r + codes;
        switch (type) {
            case SPG_B_Q8_0:
                for (uint32_t b = 0; b < B; b++) {
                    const uint8_t *block = src + ((size_t)j * B + b) * 34u;
                    memcpy(r + 32u * b, block + 2, 32);
                    memcpy(aux + 2u * b, block, 2);
                }
                break;
            case SPG_B_Q4_0: case SPG_B_IQ4_NL: case SPG_B_Q4_1: case SPG_B_Q5_0: case SPG_B_Q5_1: {
                /* (GGUF's blocks: the scale, the minimum (Q4_1, Q5_1), the fifth bits (Q5_0, Q5_1), then the codes) */
                const bool one = type == SPG_B_Q4_1 || type == SPG_B_Q5_1, five = type == SPG_B_Q5_0 || type == SPG_B_Q5_1;
                const uint32_t bytes = 18u + (one ? 2u : 0u) + (five ? 4u : 0u);
                uint8_t *scales = aux + (five ? 4u * B : 0u);
                for (uint32_t b = 0; b < B; b++) {
                    const uint8_t *block = src + ((size_t)j * B + b) * bytes;
                    memcpy(r + 16u * b, block + bytes - 16u, 16);
                    if (five) memcpy(aux + 4u * b, block + (one ? 4 : 2), 4);
                    memcpy(scales + (one ? 4u : 2u) * b, block, one ? 4u : 2u);
                }
                break;
            }
            case SPG_B_Q4_K: case SPG_B_Q5_K: {
                const bool five = type == SPG_B_Q5_K;
                uint32_t *fifth = (uint32_t *)(void *)aux;
                uint16_t *pairs = (uint16_t *)(void *)(aux + (five ? 4u * B : 0u));
                uint8_t *dm = (uint8_t *)(pairs + B);
                for (uint32_t b = 0; b < SBK; b++) {
                    /* (Q5_K: the high bits of value l of every sub-block in byte 16 + l, the codes from byte 48) */
                    const uint8_t *block = src + ((size_t)j * SBK + b) * (five ? 176u : 144u), *qs = block + (five ? 48 : 16);
                    memcpy(dm + 4u * b, block, 4);
                    for (uint32_t c = 0; c < 8u; c++) {
                        uint32_t sc, mn, code[32];
                        scale_min_k4(c, block + 4, &sc, &mn);
                        pairs[8u * b + c] = (uint16_t)(sc | mn << 8);
                        /* value i of sub-block c: the low (c even) or high nibble of byte 32 (c / 2) + i */
                        for (uint32_t i = 0; i < 32u; i++) code[i] = (uint32_t)(qs[32u * (c / 2u) + i] >> (4u * (c % 2u))) & 0xFu;
                        nibbles(r + 16u * (8u * b + c), code);
                        if (five) {
                            uint32_t bits = 0;
                            for (uint32_t i = 0; i < 32u; i++) bits |= ((uint32_t)(block[16u + i] >> c) & 1u) << i;
                            fifth[8u * b + c] = bits;
                        }
                    }
                }
                break;
            }
            case SPG_B_IQ4_XS: {
                /* (GGUF's: the scale, the scales' high two bits, their low four bits, then the codes of every sub-block
                   in sixteen bytes as Q4_0's) */
                uint16_t *pairs = (uint16_t *)(void *)aux;
                uint8_t *dm = (uint8_t *)(pairs + B);
                for (uint32_t b = 0; b < SBK; b++) {
                    const uint8_t *block = src + ((size_t)j * SBK + b) * 136u;
                    const uint32_t high = (uint32_t)block[2] | (uint32_t)block[3] << 8;
                    memcpy(dm + 4u * b, block, 2);
                    memcpy(r + 128u * b, block + 8, 128);
                    for (uint32_t c = 0; c < 8u; c++) {
                        const int32_t sc = (int32_t)(((block[4u + c / 2u] >> (4u * (c % 2u))) & 0xFu) | (((high >> (2u * c)) & 3u) << 4)) - 32;
                        pairs[8u * b + c] = (uint16_t)(uint8_t)(int8_t)sc;
                    }
                }
                break;
            }
            case SPG_B_Q6_K: {
                uint8_t *highs = r + K / 2u;
                uint16_t *pairs = (uint16_t *)(void *)aux;
                uint8_t *d = (uint8_t *)(pairs + B);
                for (uint32_t b = 0; b < SBK; b++) {
                    const uint8_t *block = src + ((size_t)j * SBK + b) * 210u;
                    memcpy(d + 2u * b, block + 208, 2);
                    for (uint32_t c = 0; c < 8u; c++) {
                        /* value i of sub-block c: its low four bits in the low (c % 4 < 2) or high nibble of byte
                           64 (c / 4) + 32 (c % 2) + i, its high two bits at 2 (c % 4) of byte 128 + 32 (c / 4) + i */
                        const uint32_t h = c / 4u, g = c % 4u, sub = 8u * b + c;
                        const uint8_t *lb = block + 64u * h + 32u * (g % 2u), *hb = block + 128 + 32u * h;
                        uint32_t low[32], high[32];
                        for (uint32_t i = 0; i < 32u; i++) {
                            low[i] = g < 2u ? lb[i] & 0xFu : (uint32_t)(lb[i] >> 4);
                            high[i] = (hb[i] >> (2u * g)) & 3u;
                        }
                        nibbles(r + 16u * sub, low);
                        for (uint32_t t = 0; t < 4u; t++)
                            for (uint32_t k = 0; k < 4u; k++)
                                highs[16u * (sub / 2u) + 4u * t + k] |=
                                    (uint8_t)((high[4u * t + k] | high[16u + 4u * t + k] << 2) << (4u * (sub % 2u)));
                        pairs[sub] = (uint16_t)(block[192u + 2u * c] | (uint32_t)block[192u + 2u * c + 1u] << 8);
                    }
                }
                break;
            }
            default:
                break;
        }
    }
}
