/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/*
 * gflash: FusedAttention of graphs on the matrix units (mma.sync m16n8k16, bfloat16 operands, single-precision sums)
 * when their products are in bfloat16, as flash.cu's FORWARD computes attention: a warp takes MT tiles of 16 queries
 * of a batch and head, a block WARPS of them; tiles of BT keys and their values pass through shared memory (rows of
 * D + 8 bfloat16). Queries, keys and values are read as floats where strides put them (SpgGattnPush, as gattn.cu
 * reads them) and rounded to bfloat16 as they are loaded; the scores are scaled in single precision, then the keep
 * mask and the mask apply as in gattn.cu, the softmax online in registers (a row's maximum and sum combined over the
 * four threads that hold it, in a fixed pattern), O += P V with P rounded to bfloat16. Every product adds its 16
 * values of k at a time in the order of k: the same bits on every run. A unit per head size D (32, 64, 128), and for
 * heads of 32 and 64 one of blocks of half the queries (_m1: MTS 1), for sequences too short to fill the device.
 */

#include "common.cuh"

#if !defined(SPG_ENTRY)
#define SPG_ENTRY spg_gflash_d64
#define D 64u
#endif
#define WARPS (D <= 64u ? 4u : 2u)         /* warps a block */
#if defined(MTS)
#define MT MTS                             /* tiles of 16 queries a warp: units _m1 for short sequences (more blocks) */
#else
#define MT (D <= 64u ? 2u : 1u)
#endif
#define ROWS (16u * MT * WARPS)            /* queries of a block */
#define BT (D <= 64u ? 64u : 32u)          /* keys a tile */
#define SD (D + 8u)                        /* bfloat16 a row in shared memory */
#define NT (BT / 8u)                       /* n-tiles of 8 of a tile of scores */
#define DT (D / 8u)                        /* n-tiles of 8 of a row of D */
#define KS (D / 16u)                       /* k-steps of 16 over D */

DEVICE uint32_t lane_id() { return thread_x() & 31u; }
DEVICE uint32_t warp_id() { return thread_x() >> 5; }
DEVICE uint32_t shared_address(const void *p) {
    uint64_t r;
    asm("cvta.to.shared.u64 %0, %1;" : "=l"(r) : "l"(p));
    return (uint32_t)r;
}
DEVICE void ldsm4(uint32_t at, uint32_t (&r)[4]) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(at) : "memory");
}
DEVICE void ldsm4t(uint32_t at, uint32_t (&r)[4]) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(at) : "memory");
}
DEVICE void mma(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
        "{%0, %1, %2, %3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
/* two values rounded to bfloat16 (to the nearest, ties to even, as to_bf16() rounds all but NaN) in a word, the
   first in the low half: one instruction */
DEVICE uint32_t pack2(float lo, float hi) {
    uint32_t r;
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
    return r;
}
DEVICE float shfl_xor(float v, uint32_t mask) { return __nvvm_shfl_sync_bfly_f32(0xFFFFFFFFu, v, mask, 0x1F); }

/* x[b] (b < 4): this thread's pair of values of fragment b of a group of four, of a row the four threads of a quad
   hold (thread t its columns 2t and 2t + 1 of each fragment); after the exchange (a 4 x 4 transpose in two steps of
   shuffles), x[b] holds columns 2b and 2b + 1 of fragment t: the eight columns of one fragment */
DEVICE void quad_transpose(float (&x)[4][2], uint32_t t) {
#pragma unroll
    for (uint32_t b = 0; b < 4u; b += 2u)       /* with the thread t ^ 1 */
#pragma unroll
        for (uint32_t k = 0; k < 2u; k++) {
            const float r = shfl_xor((t & 1u) ? x[b][k] : x[b + 1u][k], 1u);
            if (t & 1u) x[b][k] = r;
            else x[b + 1u][k] = r;
        }
#pragma unroll
    for (uint32_t d = 0; d < 2u; d++)           /* with the thread t ^ 2 */
#pragma unroll
        for (uint32_t k = 0; k < 2u; k++) {
            const float r = shfl_xor((t & 2u) ? x[d][k] : x[2u + d][k], 2u);
            if (t & 2u) x[d][k] = r;
            else x[2u + d][k] = r;
        }
}

/* the maximum (sum) of a row's values over the four threads of a quad that hold it: xor 1, then xor 2 */
DEVICE float quad_max(float v) { v = max_(v, shfl_xor(v, 1u)); return max_(v, shfl_xor(v, 2u)); }
DEVICE float quad_sum(float v) { v += shfl_xor(v, 1u); return v + shfl_xor(v, 2u); }

struct uint4_ {
    uint32_t x, y, z, w;
} __attribute__((aligned(16)));

/* Rows r0 .. r0 + count - 1 (of `rows`) of floats into a shared tile [row][SD] of bfloat16: row r at base + r stride,
   its value c at c cs; zeros past the rows and the head's d values. Eight values a thread at a time, two loads of four
   where the rows are of exactly D contiguous values at 16 bytes (vec). */
DEVICE void load_tile(const SpgGattnPush &p, uint16_t *tile, uint64_t base, uint32_t stride, uint32_t cs, uint32_t r0,
                      uint32_t count, uint32_t rows, bool vec) {
    for (uint32_t e = thread_x(); e < count * (D / 8u); e += 32u * WARPS) {
        const uint32_t row = e / (D / 8u), c = 8u * (e % (D / 8u)), pos = r0 + row;
        float v[8];
        if (vec) {
            const float4_ zero = {0.0f, 0.0f, 0.0f, 0.0f};
            const float *at = F(base) + (uint64_t)pos * stride + c;
            const float4_ a = pos < rows ? ((const float4_ *)at)[0] : zero, b = pos < rows ? ((const float4_ *)at)[1] : zero;
            v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w; v[4] = b.x; v[5] = b.y; v[6] = b.z; v[7] = b.w;
        } else {
#pragma unroll
            for (uint32_t k = 0; k < 8u; k++)
                v[k] = pos < rows && c + k < p.d ? F(base)[(uint64_t)pos * stride + (uint64_t)(c + k) * cs] : 0.0f;
        }
        *(uint4_ *)(tile + row * SD + c) = uint4_{pack2(v[0], v[1]), pack2(v[2], v[3]), pack2(v[4], v[5]), pack2(v[6], v[7])};
    }
}

/* A operand fragments (16 rows x 16 k) of a warp's rows r0.. of a shared tile, k-step ks */
DEVICE void frag_a(const uint16_t *tile, uint32_t r0, uint32_t ks, uint32_t (&a)[4]) {
    const uint32_t l = lane_id();
    ldsm4(shared_address(tile + (r0 + (l & 15u)) * SD + 16u * ks + 8u * (l >> 4)), a);
}
/* B fragments of two n-tiles (n0, n0 + 8) of k-step ks, from a tile stored [n][k] */
DEVICE void frag_b(const uint16_t *tile, uint32_t n0, uint32_t ks, uint32_t (&b)[4]) {
    const uint32_t l = lane_id();
    ldsm4(shared_address(tile + (n0 + (l & 7u) + 8u * (l >> 4)) * SD + 16u * ks + 8u * ((l >> 3) & 1u)), b);
}
/* B fragments of two n-tiles (n0, n0 + 8) of k-step ks (rows k0 = 16 ks), from a tile stored [k][n] */
DEVICE void frag_bt(const uint16_t *tile, uint32_t n0, uint32_t ks, uint32_t (&b)[4]) {
    const uint32_t l = lane_id();
    ldsm4t(shared_address(tile + (16u * ks + (l & 7u) + 8u * ((l >> 3) & 1u)) * SD + n0 + 8u * (l >> 4)), b);
}

/* The rows of the scores [16][BT] (c-fragments of NT n-tiles) as A fragments of k-step j (keys 16 j..) */
DEVICE void scores_a(const float (&s)[NT][4], uint32_t j, uint32_t (&a)[4]) {
    a[0] = pack2(s[2u * j][0], s[2u * j][1]);
    a[1] = pack2(s[2u * j][2], s[2u * j][3]);
    a[2] = pack2(s[2u * j + 1u][0], s[2u * j + 1u][1]);
    a[3] = pack2(s[2u * j + 1u][2], s[2u * j + 1u][3]);
}

struct Shared {
    uint16_t a[ROWS * SD];          /* the block's queries */
    uint16_t t[BT * SD], u[BT * SD];    /* a tile of keys, of values */
};

extern "C" __global__ void __launch_bounds__(32u * WARPS) SPG_ENTRY(const SpgGattnPush p, const Spec spec) {
    (void)spec;
    __shared__ __attribute__((aligned(16))) Shared sh;
    const uint32_t *st = U(p.strides);
    const uint32_t l = lane_id(), w = warp_id(), g = l >> 2, t = l & 3u;
    const uint32_t bh = block_y(), b = bh / p.heads, h = bh % p.heads, i0 = block_x() * ROWS, r0 = 16u * MT * w;
    const uint64_t qb = p.q + 4ull * ((uint64_t)b * st[0] + (uint64_t)h * st[1]);
    const uint64_t kb = p.k + 4ull * ((uint64_t)b * st[3] + (uint64_t)h * st[4]);
    const uint64_t vb = p.v + 4ull * ((uint64_t)b * st[6] + (uint64_t)h * st[7]);
    const uint64_t mbase = (uint64_t)b * st[12] + (uint64_t)h * st[13], wbase = (uint64_t)b * st[16] + (uint64_t)h * st[17];
    const bool vec = (p.flags & SPG_GATTN_VEC) != 0u, masked = (p.flags & (SPG_GATTN_MASK | SPG_GATTN_BOOL | SPG_GATTN_KEEP)) != 0u;
    const float NEG = -__builtin_inff(), scale = p.scale;
    load_tile(p, sh.a, qb, st[2], 1u, i0, ROWS, p.queries, vec);
    barrier();
    uint32_t qa[MT][KS][4];
#pragma unroll
    for (uint32_t mt = 0; mt < MT; mt++)
#pragma unroll
        for (uint32_t k = 0; k < KS; k++) frag_a(sh.a, r0 + 16u * mt, k, qa[mt][k]);
    float o[MT][DT][4], m_[MT][2], l_[MT][2];
#pragma unroll
    for (uint32_t mt = 0; mt < MT; mt++) {
#pragma unroll
        for (uint32_t j = 0; j < DT; j++) o[mt][j][0] = o[mt][j][1] = o[mt][j][2] = o[mt][j][3] = 0.0f;
        m_[mt][0] = m_[mt][1] = NEG;
        l_[mt][0] = l_[mt][1] = 0.0f;
    }
    for (uint32_t j0 = 0; j0 < p.keys; j0 += BT) {
        barrier();
        load_tile(p, sh.t, kb, st[5], st[20], j0, BT, p.keys, vec);
        load_tile(p, sh.u, vb, st[8], 1u, j0, BT, p.keys, vec);
        barrier();
        float sc[MT][NT][4];
#pragma unroll
        for (uint32_t mt = 0; mt < MT; mt++)
#pragma unroll
            for (uint32_t j = 0; j < NT; j++) sc[mt][j][0] = sc[mt][j][1] = sc[mt][j][2] = sc[mt][j][3] = 0.0f;
#pragma unroll
        for (uint32_t k = 0; k < KS; k++)
#pragma unroll
            for (uint32_t j = 0; j < NT; j += 2u) {
                uint32_t bf[4];
                frag_b(sh.t, 8u * j, k, bf);
#pragma unroll
                for (uint32_t mt = 0; mt < MT; mt++) {
                    mma(sc[mt][j], qa[mt][k], bf[0], bf[1]);
                    mma(sc[mt][j + 1u], qa[mt][k], bf[2], bf[3]);
                }
            }
#pragma unroll
        for (uint32_t mt = 0; mt < MT; mt++) {
            /* the scale, then the masks (and keys past the end, queries past the end) on the tiles that need them:
               this thread's rows g and g + 8 of the tile */
            const uint32_t q0 = i0 + r0 + 16u * mt, ra = q0 + g, rb = ra + 8u;
            const bool edge = masked || j0 + BT > p.keys || q0 + 16u > p.queries;
#pragma unroll
            for (uint32_t j = 0; j < NT; j++)
#pragma unroll
                for (uint32_t e = 0; e < 4u; e++) {
                    float v = sc[mt][j][e] * scale;
                    if (edge) {
                        const uint32_t key = j0 + 8u * j + 2u * t + (e & 1u), row = e < 2u ? ra : rb;
                        if (key >= p.keys || row >= p.queries) {
                            v = NEG;
                        } else {
                            if ((p.flags & SPG_GATTN_KEEP) &&
                                !((const uint8_t *)p.keep)[wbase + (uint64_t)row * st[18] + (uint64_t)key * st[19]])
                                v = p.fill;
                            if (p.flags & SPG_GATTN_MASK) v += F(p.mask)[mbase + (uint64_t)row * st[14] + (uint64_t)key * st[15]];
                            else if ((p.flags & SPG_GATTN_BOOL) &&
                                     !((const uint8_t *)p.mask)[mbase + (uint64_t)row * st[14] + (uint64_t)key * st[15]])
                                v = NEG;
                        }
                    }
                    sc[mt][j][e] = v;
                }
            float ta = NEG, tb = NEG;
#pragma unroll
            for (uint32_t j = 0; j < NT; j++) {
                ta = max_(ta, max_(sc[mt][j][0], sc[mt][j][1]));
                tb = max_(tb, max_(sc[mt][j][2], sc[mt][j][3]));
            }
            ta = quad_max(ta);
            tb = quad_max(tb);
            const float na = max_(m_[mt][0], ta), nb = max_(m_[mt][1], tb);
            const float aa = na == NEG ? 1.0f : exp_(m_[mt][0] - na), ab = nb == NEG ? 1.0f : exp_(m_[mt][1] - nb);
            float suma = 0.0f, sumb = 0.0f;
#pragma unroll
            for (uint32_t j = 0; j < NT; j++) {
                sc[mt][j][0] = sc[mt][j][0] == NEG ? 0.0f : exp_(sc[mt][j][0] - na);
                sc[mt][j][1] = sc[mt][j][1] == NEG ? 0.0f : exp_(sc[mt][j][1] - na);
                sc[mt][j][2] = sc[mt][j][2] == NEG ? 0.0f : exp_(sc[mt][j][2] - nb);
                sc[mt][j][3] = sc[mt][j][3] == NEG ? 0.0f : exp_(sc[mt][j][3] - nb);
                suma += sc[mt][j][0] + sc[mt][j][1];
                sumb += sc[mt][j][2] + sc[mt][j][3];
            }
            l_[mt][0] = l_[mt][0] * aa + suma;
            l_[mt][1] = l_[mt][1] * ab + sumb;
            m_[mt][0] = na;
            m_[mt][1] = nb;
#pragma unroll
            for (uint32_t j = 0; j < DT; j++) {
                o[mt][j][0] *= aa; o[mt][j][1] *= aa;
                o[mt][j][2] *= ab; o[mt][j][3] *= ab;
            }
        }
        /* O += P V: P's rows as A fragments, V's [key][d] tile transposed into B fragments */
#pragma unroll
        for (uint32_t k = 0; k < BT / 16u; k++) {
            uint32_t pa[MT][4];
#pragma unroll
            for (uint32_t mt = 0; mt < MT; mt++) scores_a(sc[mt], k, pa[mt]);
#pragma unroll
            for (uint32_t j = 0; j < DT; j += 2u) {
                uint32_t bf[4];
                frag_bt(sh.u, 8u * j, k, bf);
#pragma unroll
                for (uint32_t mt = 0; mt < MT; mt++) {
                    mma(o[mt][j], pa[mt], bf[0], bf[1]);
                    mma(o[mt][j + 1u], pa[mt], bf[2], bf[3]);
                }
            }
        }
    }
    /* y = O / sum (a row of no key: zeros with SAFE, else NaN), this thread's columns 8 j + 2 t, + 1 of rows g, g + 8 */
    const uint64_t yb = p.y + 4ull * ((uint64_t)b * st[9] + (uint64_t)h * st[10]);
    const bool safe = (p.flags & SPG_GATTN_SAFE) != 0u;
#pragma unroll
    for (uint32_t mt = 0; mt < MT; mt++) {
        const uint32_t ra = i0 + r0 + 16u * mt + g, rb = ra + 8u;
        l_[mt][0] = quad_sum(l_[mt][0]);
        l_[mt][1] = quad_sum(l_[mt][1]);
#pragma unroll
        for (uint32_t hh = 0; hh < 2u; hh++) {
            const uint32_t row = hh ? rb : ra;
            const float sum = l_[mt][hh];
            const float inv = sum > 0.0f ? 1.0f / sum : safe ? 0.0f : __builtin_nanf("");
            if (row >= p.queries) continue;
            float *y = F(yb) + (uint64_t)row * st[11];
#pragma unroll
            for (uint32_t j = 0; j < DT; j++) {
                const uint32_t c = 8u * j + 2u * t;
                const float v0 = sum > 0.0f || !safe ? o[mt][j][2u * hh] * inv : 0.0f;
                const float v1 = sum > 0.0f || !safe ? o[mt][j][2u * hh + 1u] * inv : 0.0f;
                if (c < p.d) y[c] = v0;
                if (c + 1u < p.d) y[c + 1u] = v1;
            }
        }
    }
}
