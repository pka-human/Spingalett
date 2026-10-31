/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/* Shared by bdot.comp and bdots.comp: rows of activations quantized by bquant.comp times weights of a block precision packed
   as sessions keep them (Spingalett.GpuPack.c), each block's integers and term as spingalett_block_dot() computes them. */

layout(buffer_reference, std430, buffer_reference_align = 16) buffer U32x4 { uvec4 v[]; };

/* four signed bytes' products with four */
int dot4(uint a, uint b) {
    return (int(a << 24) >> 24) * (int(b << 24) >> 24) + (int(a << 16) >> 24) * (int(b << 16) >> 24) +
           (int(a << 8) >> 24) * (int(b << 8) >> 24) + (int(a) >> 24) * (int(b) >> 24);
}

uint nl_values(uint q) {
    return (uint(iq4_value(q & 15u)) & 0xFFu) | (uint(iq4_value((q >> 8) & 15u)) & 0xFFu) << 8 |
           (uint(iq4_value((q >> 16) & 15u)) & 0xFFu) << 16 | (uint(iq4_value((q >> 24) & 15u)) & 0xFFu) << 24;
}

/* the low four bits of x at bit 4 of a word's four bytes */
uint spread1(uint x) { return ((x * 0x00204081u) & 0x01010101u) << 4; }

/* The products of sixteen bytes of 32 4-bit codes (byte i the code of value i, then of value i + 16 above it) with x0 and x1,
   with fifth bits f above them, or their values in IQ4's table */
int nibbles_dot(uvec4 w, uvec4 x0, uvec4 x1, uint f) {
    const bool nl = TYPE == B_IQ4_NL || TYPE == B_IQ4_XS, five = TYPE == B_Q5_0 || TYPE == B_Q5_1 || TYPE == B_Q5_K;
    int s = 0;
    for (uint k = 0u; k < 4u; k++) {
        uint lo = w[k] & 0x0F0F0F0Fu, hi = (w[k] >> 4) & 0x0F0F0F0Fu;
        if (five) {
            lo |= spread1((f >> (4u * k)) & 0xFu);
            hi |= spread1((f >> (16u + 4u * k)) & 0xFu);
        }
        if (nl) {
            lo = nl_values(lo);
            hi = nl_values(hi);
        }
        s += dot4(lo, x0[k]) + dot4(hi, x1[k]);
    }
    return s;
}

/* The bytes of a row at `at`: its codes, then the rest of its blocks, of the row's activations r */
struct Row {
    uint at, aux, scales, heads, highs;     /* weights: the row; what else its blocks hold, their scales, heads, Q6_K's highs */
    uint x, xs, dx;                         /* activations: the row's codes, their sums, the blocks' scales */
};

Row row_of(uint row, uint r) {
    Row a;
    const uint subs = p.k / 32u;
    const bool five = TYPE == B_Q5_0 || TYPE == B_Q5_1 || TYPE == B_Q5_K;
    a.at = row * p.stride;
    a.highs = a.at + p.k / 2u;
    a.aux = a.at + (TYPE == B_Q8_0 ? p.k : TYPE == B_Q6_K ? p.k / 2u + p.k / 4u : p.k / 2u);
    a.heads = a.aux + (five ? 4u * subs : 0u);
    a.scales = TYPE == B_Q6_K ? a.aux + 2u * subs : a.heads;
    a.x = r * p.k;
    a.xs = r * (p.k / 16u);
    a.dx = r * (p.k / (KQ ? 256u : 32u));
    return a;
}

/* The term of block b of one sub-block (Q8_0, Q4_0, Q4_1, Q5_0, Q5_1, IQ4_NL) */
float block_term(Row a, uint b) {
    const bool one = TYPE == B_Q4_1 || TYPE == B_Q5_1, five = TYPE == B_Q5_0 || TYPE == B_Q5_1;
    const int off = TYPE == B_Q4_0 ? 8 : TYPE == B_Q5_0 ? 16 : 0;
    const uvec4 x0 = U32x4(p.q).v[(a.x + 32u * b) >> 4], x1 = U32x4(p.q).v[((a.x + 32u * b) >> 4) + 1u];
    int i1, sum = 0;
    if (TYPE == B_Q8_0) {
        const uvec4 w0 = U32x4(p.w).v[(a.at + 32u * b) >> 4], w1 = U32x4(p.w).v[((a.at + 32u * b) >> 4) + 1u];
        i1 = dot4(w0.x, x0.x) + dot4(w0.y, x0.y) + dot4(w0.z, x0.z) + dot4(w0.w, x0.w) +
             dot4(w1.x, x1.x) + dot4(w1.y, x1.y) + dot4(w1.z, x1.z) + dot4(w1.w, x1.w);
    } else {
        const uvec4 w = U32x4(p.w).v[(a.at + 16u * b) >> 4];
        sum = int(p.xs.v[a.xs + 2u * b]) + int(p.xs.v[a.xs + 2u * b + 1u]);
        i1 = nibbles_dot(w, x0, x1, five ? p.w.v[(a.aux + 4u * b) >> 2] : 0u) - off * sum;
    }
    const float dxv = p.dx.v[a.dx + b], d = from_half(half_at(p.w, a.scales + (one ? 4u : 2u) * b));
    precise float term = (d * dxv) * float(i1);
    if (one) term = term + (from_half(half_at(p.w, a.scales + 4u * b + 2u)) * dxv) * float(sum);
    return term;
}

/* The integers of sub-block c of a super-block of eight (Q4_K, Q5_K, Q6_K, IQ4_XS), added to i1 and i2 */
void sub_block(Row a, uint c, inout int i1, inout int i2) {
    const uvec4 w = U32x4(p.w).v[(a.at + 16u * c) >> 4];
    const uvec4 x0 = U32x4(p.q).v[(a.x + 32u * c) >> 4], x1 = U32x4(p.q).v[((a.x + 32u * c) >> 4) + 1u];
    if (TYPE == B_Q6_K) {
        /* (each half of 16 alone: four words of low bits, and the high two bits of the pair of sub-blocks' words:
           bits 0 to 3 of their bytes the first's, 4 to 7 the second's) */
        const uvec4 hb = U32x4(p.w).v[(a.highs + 16u * (c / 2u)) >> 4];
        const uint shift = 4u * (c % 2u);
        int lo = 0, hi = 0;
        for (uint k = 0u; k < 4u; k++) {
            const uint h = hb[k] >> shift;
            lo += dot4((w[k] & 0x0F0F0F0Fu) | (h & 0x03030303u) << 4, x0[k]);
            hi += dot4(((w[k] >> 4) & 0x0F0F0F0Fu) | (h & 0x0C0C0C0Cu) << 2, x1[k]);
        }
        const uint pr = half_at(p.w, a.aux + 2u * c);
        i1 += (int(pr << 24) >> 24) * (lo - 32 * int(p.xs.v[a.xs + 2u * c]));
        i1 += (int(pr << 16) >> 24) * (hi - 32 * int(p.xs.v[a.xs + 2u * c + 1u]));
        return;
    }
    const int s = nibbles_dot(w, x0, x1, TYPE == B_Q5_K ? p.w.v[(a.aux + 4u * c) >> 2] : 0u);
    const uint b = c / 8u, e = c % 8u;
    if (TYPE == B_IQ4_XS) {
        i1 += iq4xs_scale(word_at(p.w, a.heads + 8u * b), word_at(p.w, a.heads + 8u * b + 4u), e) * s;
    } else {
        int sc, mn;
        scale_min_k4(p.w, a.heads + 16u * b + 4u, e, sc, mn);
        i1 += sc * s;
        i2 += mn * (int(p.xs.v[a.xs + 2u * c]) + int(p.xs.v[a.xs + 2u * c + 1u]));
    }
}

/* The term of super-block b from its sub-blocks' integers */
float super_term(Row a, uint b, int i1, int i2) {
    const bool six = TYPE == B_Q6_K, xs4 = TYPE == B_IQ4_XS;
    const float dxv = p.dx.v[a.dx + b];
    const float d = from_half(half_at(p.w, six ? a.scales + 2u * b : a.heads + (xs4 ? 8u : 16u) * b));
    precise float term = (d * dxv) * float(i1);
    if (!six && !xs4) {
        const float m = -from_half(half_at(p.w, a.heads + 16u * b + 2u));
        term = term + (m * dxv) * float(i2);
    }
    return term;
}

/* The output: the bias then the sum, the activation, a gated unit's other half */
void block_out(uint j, uint r, float a0, float a1) {
    const uint at = r * p.ldy + j;
    float v;
    if (PAIR != 0u) {
        v = p.bias.v[2u * j] + a0;
        if (ACT != 0u) v = activate(v, ACT);
        v = v * (p.bias.v[2u * j + 1u] + a1);
    } else {
        v = p.bias.v[j] + a0;
        if (ACT != 0u) v = activate(v, ACT);
        if (GATED != 0u) v = v * p.e0.v[at];
    }
    p.y.v[at] = v;
}
