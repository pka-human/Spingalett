/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/*
 * What the matrix kernels (gemm.comp in single precision, gemm_mma.comp on matrix units) share: the
 * operand modes and their loads into registers, the geometry of convolutions and the epilogues.
 * The kernel declares the specialization constants BM, BN, BK, AMODE, BMODE, EPI, ACT, THREADS,
 * PHASED, VEC and HALF before including this file. See gemm.comp for the modes.
 *
 * HALF: what is kept in memory as bfloat16 (bit 0: A, 1: B, 2: C, 3: e0 of EPI_DERIV), for kernels
 * that define SPG_HALF (and enable 16-bit storage). Values are rounded to bfloat16 to the nearest,
 * ties to even, by to_bf16(), as the host rounds its inputs.
 */

#define A_ROW   0u
#define A_COL   1u
#define A_CONV  2u
#define B_ROW   0u
#define B_COL   1u
#define B_CONV  2u

#define EPI_STORE       0u      /* C = alpha AB + beta C */
#define EPI_BIAS_ACT    1u      /* C = act(AB + bias[column]) */
#define EPI_SCALE_ACT   2u      /* C = act(AB * scale[column] + shift[column]) */
#define EPI_PARTIAL     3u      /* partial[z] = AB */
#define EPI_DERIV       4u      /* C = (alpha AB + beta C) * act'(e0), e0 laid out like C */

#define FLAG_BIAS       1u

layout(push_constant) uniform Push {
    F32 a, b, c, e0, e1;
    U32 geo;
    uint M, N, K;
    uint lda, ldb, ldc;
    uint a_group, b_group, c_group;     /* added to a, b and c per group */
    uint slices, slice_k;
    float alpha, beta;
    uint flags;
    uint m_tile0;                       /* the first tile of rows: a product of more tiles than a
                                           dispatch may have is dispatched in parts */
} p;

/* geometry: rows over an RH x RW grid of pixels, gathered from a GH x GW x GC tensor */
#define GEO_RH 0u
#define GEO_RW 1u
#define GEO_GH 2u
#define GEO_GW 3u
#define GEO_GC 4u
#define GEO_SH 5u
#define GEO_SW 6u
#define GEO_PH 7u
#define GEO_PW 8u
#define GEO_CH 11u              /* PHASED: the grid C is laid out on, the phase's origin and steps */
#define GEO_CW 12u
#define GEO_CY 13u
#define GEO_CX 14u
#define GEO_CS 15u              /* step_h | step_w << 16 */
#define GEO_TAPS 16u

/* VEC: A and B read as vectors (bits 0 and 1), of eight bfloat16 rather than four values (bits 2, 3); C
   written four results at a time (bit 4, gemm.comp) */
const bool VA = (VEC & 1u) != 0u, VB = (VEC & 2u) != 0u;
const bool HA = (HALF & 1u) != 0u, HB = (HALF & 2u) != 0u, HC = (HALF & 4u) != 0u, HE = (HALF & 8u) != 0u;

#ifdef SPG_HALF
layout(buffer_reference, std430, buffer_reference_align = 8) buffer BF16x4 { u16vec4 v[]; };
struct Raw8 { u16vec4 lo, hi; };
layout(buffer_reference, std430, buffer_reference_align = 16) buffer BF16x8 { Raw8 v[]; };
#endif
/* the axis a thread's loads run along: k (true) or the other one */
const bool A_KFAST = AMODE != A_COL, B_KFAST = BMODE == B_COL;
/* values per load: four floats, or eight values kept as bfloat16 (16 bytes either way), or one */
const uint WA = VA ? ((VEC & 4u) != 0u ? 8u : 4u) : 1u, WB = VB ? ((VEC & 8u) != 0u ? 8u : 4u) : 1u;
const bool RA = WA == 8u, RB = WB == 8u;                     /* loads of raw bfloat16 (ra, rb) */
const uint LA = (BM * BK / WA + THREADS - 1u) / THREADS;    /* loads per thread and step */
const uint LB = (BN * BK / WB + THREADS - 1u) / THREADS;
uint tid, m0, n0, kend, aoff, boff;
uint RH, RW, GH, GW, GC, SH, SW, PH, PW;
int abase[LA], ah[LA], aw[LA];              /* A_CONV: each loaded row's sample and window */
int btap_h[LB], btap_w[LB], btap_c[LB];     /* B_CONV: each loaded column's tap, */
uint bsample[LB], by[LB], bx[LB];           /* and the output pixel (sample offset, y, x) of its k */
vec4 fa[RA ? 1u : LA], fb[RB ? 1u : LB];
#ifdef SPG_HALF
u16vec4 ra[RA ? 2u * LA : 1u], rb[RB ? 2u * LB : 1u];       /* eight bfloat16 a load, as they are in memory */
#endif

/* the row (column) and k of the i-th load of A (B): the first of WA (WB) consecutive ones along the
   contiguous axis */
uint a_row(uint i) {
    uint e = tid + i * THREADS;
    return A_KFAST ? e / (BK / WA) : (e % (BM / WA)) * WA;
}
uint a_k(uint i) {
    uint e = tid + i * THREADS;
    return A_KFAST ? (e % (BK / WA)) * WA : e / (BM / WA);
}
uint b_col(uint i) {
    uint e = tid + i * THREADS;
    return B_KFAST ? e / (BK / WB) : (e % (BN / WB)) * WB;
}
uint b_k(uint i) {
    uint e = tid + i * THREADS;
    return B_KFAST ? (e % (BK / WB)) * WB : e / (BN / WB);
}
bool a_live(uint i) { return tid + i * THREADS < BM * BK / WA; }
bool b_live(uint i) { return tid + i * THREADS < BN * BK / WB; }

void prepare(uint kbeg) {
    if (AMODE == A_CONV || BMODE == B_CONV) {
        RH = p.geo.v[GEO_RH]; RW = p.geo.v[GEO_RW];
        GH = p.geo.v[GEO_GH]; GW = p.geo.v[GEO_GW]; GC = p.geo.v[GEO_GC];
        SH = p.geo.v[GEO_SH]; SW = p.geo.v[GEO_SW]; PH = p.geo.v[GEO_PH]; PW = p.geo.v[GEO_PW];
    }
    if (AMODE == A_CONV) {
        [[unroll]] for (uint i = 0; i < LA; i++) {
            uint m = m0 + a_row(i);
            uint n = m / (RH * RW), r = m % (RH * RW), y = r / RW, x = r % RW;
            abase[i] = int(n * GH * GW * GC);
            ah[i] = int(y * SH) - int(PH);
            aw[i] = int(x * SW) - int(PW);
            if (m >= p.M || !a_live(i)) ah[i] = -(1 << 28);    /* never inside */
        }
    }
    if (BMODE == B_CONV) {
        [[unroll]] for (uint i = 0; i < LB; i++) {
            uint n = n0 + b_col(i);
            uint t = n < p.N ? p.geo.v[GEO_TAPS + n] : 0u;
            btap_h[i] = int(t & 255u);
            btap_w[i] = n < p.N ? int((t >> 8) & 255u) : (1 << 28);
            btap_c[i] = int(t >> 16);
            /* k advances by BK a step: the pixel is followed rather than divided out every time */
            uint k = kbeg + b_k(i), r = k % (RH * RW);
            bsample[i] = (k / (RH * RW)) * GH * GW * GC;
            by[i] = r / RW;
            bx[i] = r % RW;
        }
    }
}

/* WA (or WB) consecutive values from x at index at (a multiple of four with vectors), floats or (h16)
   bfloat16 */
vec4 load(F32 x, uint at, bool vec, bool h16) {
#ifdef SPG_HALF
    if (h16 && vec) {
        uvec4 h = uvec4(BF16x4(x).v[at >> 2]);
        return vec4(from_bf16(h.x), from_bf16(h.y), from_bf16(h.z), from_bf16(h.w));
    }
    if (h16) return vec4(from_bf16(uint(BF16(x).v[at])), 0.0, 0.0, 0.0);
#endif
    if (vec) return F32x4(x).v[at >> 2];
    return vec4(x.v[at], 0.0, 0.0, 0.0);
}

/* Element at of x, a float or (h16) a bfloat16, and its store. */
float get(F32 x, uint at, bool h16) {
#ifdef SPG_HALF
    if (h16) return from_bf16(uint(BF16(x).v[at]));
#endif
    return x.v[at];
}
void put(F32 x, uint at, bool h16, float v) {
#ifdef SPG_HALF
    if (h16) {
        BF16(x).v[at] = uint16_t(to_bf16(v));
        return;
    }
#endif
    x.v[at] = v;
}

void fetch(uint k0) {
    [[unroll]] for (uint i = 0; i < LA; i++) {
        uint m = m0 + a_row(i), k = k0 + a_k(i), at = 0u;
        /* vectors need all their values inside: m or k a multiple of the width below such a bound */
        bool inside = a_live(i) && m < p.M && k < kend;
        if (inside) {
            if (AMODE == A_ROW) {
                at = aoff + m * p.lda + k;
            } else if (AMODE == A_COL) {
                at = aoff + k * p.lda + m;
            } else {
                uint t = p.geo.v[GEO_TAPS + k];
                int y = ah[i] + int(t & 255u), x = aw[i] + int((t >> 8) & 255u);
                inside = uint(y) < GH && uint(x) < GW;
                at = uint(abase[i]) + (uint(y) * GW + uint(x)) * GC + (t >> 16) + aoff;
            }
        }
#ifdef SPG_HALF
        if (RA) {
            Raw8 v = Raw8(u16vec4(0us), u16vec4(0us));
            if (inside) v = BF16x8(p.a).v[at >> 3];
            ra[2u * i] = v.lo;
            ra[2u * i + 1u] = v.hi;
            continue;
        }
#endif
        fa[i] = inside ? load(p.a, at, VA, HA) : vec4(0.0);
    }
    [[unroll]] for (uint i = 0; i < LB; i++) {
        uint n = n0 + b_col(i), k = k0 + b_k(i), at = 0u;
        bool inside = b_live(i) && n < p.N && k < kend;
        if (inside) {
            if (BMODE == B_ROW) {
                at = boff + k * p.ldb + n;
            } else if (BMODE == B_COL) {
                at = boff + n * p.ldb + k;
            } else {
                /* k: an output pixel (sample, y, x) of the RH x RW grid; n: a tap of its window */
                int y = int(by[i] * SH) - int(PH) + btap_h[i], x = int(bx[i] * SW) - int(PW) + btap_w[i];
                inside = uint(y) < GH && uint(x) < GW;
                at = bsample[i] + (uint(y) * GW + uint(x)) * GC + uint(btap_c[i]) + boff;
            }
        }
#ifdef SPG_HALF
        if (RB) {
            Raw8 v = Raw8(u16vec4(0us), u16vec4(0us));
            if (inside) v = BF16x8(p.b).v[at >> 3];
            rb[2u * i] = v.lo;
            rb[2u * i + 1u] = v.hi;
        } else
#endif
        fb[i] = inside ? load(p.b, at, VB, HB) : vec4(0.0);
        if (BMODE == B_CONV) {
            bx[i] += BK;
            while (bx[i] >= RW) {
                bx[i] -= RW;
                if (++by[i] == RH) { by[i] = 0u; bsample[i] += GH * GW * GC; }
            }
        }
    }
}

/* The row of C that row m of the product is stored to: itself, or with PHASED its pixel in the grid
   of the phase's convolution input. */
uint c_row(uint m) {
    if (PHASED == 0u) return m;
    const uint CH = p.geo.v[GEO_CH], CW = p.geo.v[GEO_CW], CY = p.geo.v[GEO_CY], CX = p.geo.v[GEO_CX];
    const uint CSH = p.geo.v[GEO_CS] & 0xFFFFu, CSW = p.geo.v[GEO_CS] >> 16;
    uint n = m / (RH * RW), r = m % (RH * RW);
    return (n * CH + CY + CSH * (r / RW)) * CW + CX + CSW * (r % RW);
}

/* Output (m, n) of workgroup z's product, v, through the epilogue (row: c_row(m)). */
/* (precise: no multiply-add contracted, so that gemm.comp's four results at a time, store_c4(), give the
   same bits) */
void store_c(uint z, uint coff, uint m, uint row, uint n, precise float v) {
    if (EPI == EPI_PARTIAL) {
        p.c.v[(z * p.M + m) * p.N + n] = v;
        return;
    }
    uint at = coff + row * p.ldc + n;
    if (EPI == EPI_STORE) {
        v *= p.alpha;
        if (p.beta != 0.0) v += p.beta * get(p.c, at, HC);
    } else if (EPI == EPI_BIAS_ACT) {
        if ((p.flags & FLAG_BIAS) != 0u) v += p.e0.v[coff + n];
        v = activate(v, ACT);
    } else if (EPI == EPI_SCALE_ACT) {
        v = activate(v * p.e0.v[coff + n] + p.e1.v[coff + n], ACT);
    } else if (EPI == EPI_DERIV) {
        v *= p.alpha;
        if (p.beta != 0.0) v += p.beta * get(p.c, at, HC);
        v *= derivative(get(p.e0, at, HE), ACT);
    }
    put(p.c, at, HC, v);
}
