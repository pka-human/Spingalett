/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/* bembed.comp: the embedding of a session's rows, a thread a value: the table's row of the row's token decoded as the
   CPU decodes it (spingalett_block_decode(): each value's products in the CPU's order, rounded alone), times the scale,
   plus the positions' row. */

#include "common.cuh"

typedef signed char int8_t;

DEVICE float half_at(const uint8_t *b) { return from_half((uint32_t)b[0] | (uint32_t)b[1] << 8); }
DEVICE uint32_t get16(const uint8_t *b) { return (uint32_t)b[0] | (uint32_t)b[1] << 8; }
DEVICE uint32_t get32(const uint8_t *b) {
    return (uint32_t)b[0] | (uint32_t)b[1] << 8 | (uint32_t)b[2] << 16 | (uint32_t)b[3] << 24;
}
DEVICE int32_t iq4_value(uint32_t code) {
    const int8_t v[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
    return v[code];
}
DEVICE void scale_min_k4(uint32_t j, const uint8_t *q, int32_t &d, int32_t &m) {
    if (j < 4u) {
        d = q[j] & 63;
        m = q[j + 4u] & 63;
    } else {
        d = (q[j + 4u] & 0xF) | ((q[j - 4u] >> 6) << 4);
        m = (q[j + 4u] >> 4) | ((q[j] >> 6) << 4);
    }
}
DEVICE int32_t q3_scale(const uint8_t *packed, uint32_t is) {
    const uint32_t kmask1 = 0x03030303u, kmask2 = 0x0f0f0f0fu;
    const uint32_t a0 = get32(packed), a1 = get32(packed + 4), tmp = get32(packed + 8);
    uint32_t aux;
    switch (is / 4u) {
        case 0: aux = (a0 & kmask2) | ((tmp & kmask1) << 4); break;
        case 1: aux = (a1 & kmask2) | (((tmp >> 2) & kmask1) << 4); break;
        case 2: aux = ((a0 >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4); break;
        default: aux = ((a1 >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4); break;
    }
    return (int32_t)((aux >> (8u * (is % 4u))) & 0xFFu) - 32;
}
DEVICE int32_t iq4xs_scale(const uint8_t *block, uint32_t ib) {
    const uint32_t high = get16(block + 2), low = block[4u + ib / 2u];
    return (int32_t)(((low >> (4u * (ib % 2u))) & 0xFu) | (((high >> (2u * ib)) & 3u) << 4)) - 32;
}

/* Value v of a block of type t (SPG_B_*), as spingalett_block_decode() computes it */
DEVICE float block_value(uint32_t t, const uint8_t *x, uint32_t v) {
    switch (t) {
        case SPG_B_Q8_0: return (float)(int8_t)x[2 + v] * half_at(x);
        case SPG_B_Q4_0: case SPG_B_IQ4_NL: {
            const int32_t code = v < 16u ? x[2 + v] & 0x0F : x[2 + v - 16u] >> 4;
            return (float)(t == SPG_B_Q4_0 ? code - 8 : iq4_value((uint32_t)code)) * half_at(x);
        }
        case SPG_B_Q4_1: {
            const int32_t code = v < 16u ? x[4 + v] & 0x0F : x[4 + v - 16u] >> 4;
            return (float)code * half_at(x) + half_at(x + 2);
        }
        case SPG_B_Q5_0: case SPG_B_Q5_1: {
            const bool one = t == SPG_B_Q5_1;
            const uint32_t qh = get32(x + (one ? 4 : 2));
            const uint8_t *qs = x + (one ? 8 : 6);
            const int32_t code = (v < 16u ? qs[v] & 0x0F : qs[v - 16u] >> 4) | (int32_t)(((qh >> v) & 1u) << 4);
            return one ? (float)code * half_at(x) + half_at(x + 2) : (float)(code - 16) * half_at(x);
        }
        case SPG_B_Q2_K: case SPG_B_Q3_K: {
            /* value v = 128 nb + 32 (shift / 2) + 16 part + l: bits `shift` of byte 32 nb + 16 part + l, scale is */
            const bool three = t == SPG_B_Q3_K;
            const uint32_t nb = v / 128u, shift = 2u * ((v % 128u) / 32u), part = (v % 32u) / 16u, l = v % 16u;
            const uint32_t is = 8u * nb + shift + part;
            const uint8_t *q = x + (three ? 32 : 16) + 32u * nb;
            const int32_t code = (q[l + 16u * part] >> shift) & 3;
            if (three) {
                const float dl = half_at(x + 108) * (float)q3_scale(x + 96, is);
                const uint32_t m = 1u << (4u * nb + shift / 2u);
                return dl * (float)(code - ((x[l + 16u * part] & m) ? 0 : 4));
            }
            const uint8_t s = x[is];
            const float dl = half_at(x + 80) * (float)(s & 0xF), ml = half_at(x + 82) * (float)(s >> 4);
            return dl * (float)code - ml;
        }
        case SPG_B_Q4_K: case SPG_B_Q5_K: {
            const bool five = t == SPG_B_Q5_K;
            const uint32_t j = v / 64u, half = (v % 64u) / 32u, l = v % 32u;
            int32_t s, mn;
            scale_min_k4(2u * j + half, x + 4, s, mn);
            const float d1 = half_at(x) * (float)s, m1 = half_at(x + 2) * (float)mn;
            const uint8_t *q = x + (five ? 48 : 16) + 32u * j;
            int32_t code = half ? q[l] >> 4 : q[l] & 0xF;
            if (five && (x[16 + l] >> (2u * j + half) & 1u)) code += 16;
            return d1 * (float)code - m1;
        }
        case SPG_B_Q6_K: {
            const uint32_t nb = v / 128u, r = v % 128u, quarter = r / 32u, l = r % 32u, is = l / 16u;
            const uint8_t *ql = x + 64u * nb, *qh = x + 128 + 32u * nb;
            const int8_t *sc = (const int8_t *)(x + 192) + 8u * nb;
            const uint32_t low = quarter & 1u ? ql[l + 32u] : ql[l];
            const int32_t code = (int32_t)((quarter < 2u ? low & 0xF : low >> 4) | (((qh[l] >> (2u * quarter)) & 3u) << 4)) - 32;
            return half_at(x + 208) * (float)sc[is + 2u * quarter] * (float)code;
        }
        default: {                              /* IQ4_XS */
            const uint32_t ib = v / 32u, j = v % 32u;
            const uint8_t *qs = x + 8 + 16u * ib;
            const uint32_t code = j < 16u ? qs[j] & 0xF : qs[j - 16u] >> 4;
            return half_at(x) * (float)iq4xs_scale(x, ib) * (float)iq4_value(code);
        }
    }
}

KERNEL(bembed, SpgBembedPush) {
    const uint32_t i = global_x();
    if (i >= p.rows * p.d) return;
    const uint32_t r = i / p.d, c = i % p.d, token = U(p.token)[r];
    float v = 0.0f;
    if (token < p.vocabulary) {
        const uint8_t *row = (const uint8_t *)p.table + (uint64_t)token * p.stride;
        if (p.type == SPG_EMB_F32) {
            v = F((uint64_t)row)[c];
        } else if (p.type == SPG_EMB_F16) {
            v = from_half(get16(row + 2u * c));
        } else if (p.type == SPG_EMB_BF16) {
            v = from_bf16(get16(row + 2u * c));
        } else if (p.type == SPG_EMB_PACKED) {
            /* (Q8_0 packed: the code times its block's scale, as spingalett_block_decode()) */
            const float d = F(p.table + (uint64_t)p.vocabulary * p.d)[(uint64_t)token * (p.d / 32u) + c / 32u];
            v = (float)(int8_t)row[c] * d;
        } else {
            const uint32_t t = p.type - SPG_EMB_BLOCK, bv = t >= SPG_B_Q2_K && t != SPG_B_IQ4_NL ? 256u : 32u;
            const uint32_t bytes = t == SPG_B_Q8_0 ? 34u : t == SPG_B_Q4_0 ? 18u : t == SPG_B_Q4_1 ? 20u : t == SPG_B_Q5_0 ? 22u
                                 : t == SPG_B_Q5_1 ? 24u : t == SPG_B_Q2_K ? 84u : t == SPG_B_Q3_K ? 110u : t == SPG_B_Q4_K ? 144u
                                 : t == SPG_B_Q5_K ? 176u : t == SPG_B_Q6_K ? 210u : t == SPG_B_IQ4_NL ? 18u : 136u;
            v = block_value(t, row + (uint64_t)(c / bv) * bytes, c % bv);
        }
    }
    if (p.scale != 0.0f) v *= p.scale;
    if (p.positions) v += F(p.positions)[(uint64_t)U(p.info)[2u * r + 1u] * p.d + c];
    F(p.y)[(uint64_t)r * p.d + c] = v;
}
