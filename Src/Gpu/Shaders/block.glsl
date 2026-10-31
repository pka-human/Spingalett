/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/* Shared by the kernels of GGUF's block types in sessions (bdot.comp, bembed.comp): memory read in words, which needs
   neither 8-bit nor 16-bit storage of the device, and what those types' blocks hold. */

/* byte `at` of a buffer; a float16 and a word from any multiple of 2 and any place */
uint byte_at(U32 b, uint at) { return (b.v[at >> 2] >> ((at & 3u) * 8u)) & 0xFFu; }
uint half_at(U32 b, uint at) { return (b.v[at >> 2] >> ((at & 2u) * 8u)) & 0xFFFFu; }
uint word_at(U32 b, uint at) {
    const uint s = (at & 3u) * 8u, a = b.v[at >> 2];
    return s == 0u ? a : a >> s | b.v[(at >> 2) + 1u] << (32u - s);
}

/* a float16's value, subnormal ones too (unpackHalf2x16() may flush them) */
float from_half(uint h) {
    const uint e = (h >> 10) & 31u, m = h & 1023u, s = (h & 0x8000u) << 16;
    if (e == 31u) return uintBitsToFloat(s | 0x7F800000u | m << 13);
    if (e == 0u) {
        const float v = float(m) * 5.9604644775390625e-8;
        return s != 0u ? -v : v;
    }
    return uintBitsToFloat(s | (e + 112u) << 23 | m << 13);
}

/* IQ4_NL's and IQ4_XS's sixteen values (llama.cpp's kvalues_iq4nl) */
int iq4_value(uint code) {
    const int v[16] = int[16](-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113);
    return v[code];
}

/* The 6-bit scale and minimum j of a Q4_K or Q5_K super-block (llama.cpp's get_scale_min_k4()) from the 12 bytes at `at` */
void scale_min_k4(U32 b, uint at, uint j, out int d, out int m) {
    if (j < 4u) {
        d = int(byte_at(b, at + j) & 63u);
        m = int(byte_at(b, at + j + 4u) & 63u);
    } else {
        const uint q = byte_at(b, at + j + 4u);
        d = int((q & 0xFu) | (byte_at(b, at + j - 4u) >> 6) << 4);
        m = int((q >> 4) | (byte_at(b, at + j) >> 6) << 4);
    }
}

/* The signed scale of sub-block j of an IQ4_XS super-block from its first 8 bytes as GGUF keeps them (w0: the scale, then
   the scales' high two bits; w1: their low four bits) */
int iq4xs_scale(uint w0, uint w1, uint j) { return int(((w1 >> (4u * j)) & 0xFu) | ((w0 >> (16u + 2u * j)) & 3u) << 4) - 32; }

/* Q3_K's 6-bit scale is (of sixteen) from the twelve bytes at `at`, less 32 */
int q3_scale(U32 b, uint at, uint is) {
    const uint a0 = word_at(b, at), a1 = word_at(b, at + 4u), tmp = word_at(b, at + 8u);
    uint aux;
    switch (is / 4u) {
        case 0u: aux = (a0 & 0x0F0F0F0Fu) | (tmp & 0x03030303u) << 4; break;
        case 1u: aux = (a1 & 0x0F0F0F0Fu) | ((tmp >> 2) & 0x03030303u) << 4; break;
        case 2u: aux = ((a0 >> 4) & 0x0F0F0F0Fu) | ((tmp >> 4) & 0x03030303u) << 4; break;
        default: aux = ((a1 >> 4) & 0x0F0F0F0Fu) | ((tmp >> 6) & 0x03030303u) << 4; break;
    }
    return int((aux >> (8u * (is % 4u))) & 0xFFu) - 32;
}

/* The blocks' types (SPG_B_*) */
#define B_Q8_0 0u
#define B_Q4_0 1u
#define B_Q4_1 2u
#define B_Q5_0 3u
#define B_Q5_1 4u
#define B_Q2_K 5u
#define B_Q3_K 6u
#define B_Q4_K 7u
#define B_Q5_K 8u
#define B_Q6_K 9u
#define B_IQ4_NL 10u
#define B_IQ4_XS 11u
