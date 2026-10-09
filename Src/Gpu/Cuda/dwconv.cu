/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/*
 * dwconv.comp: depthwise convolutions (APPLY, SPREAD, WEIGHTS) with the epilogues of the products,
 * PRO (a batch normalization applied to x as it is read) and SUMS (the normalization's backward sums
 * of a SPREAD). Every sum runs in the order dwconv.comp's does.
 *
 * Spec: MODE, EPI, ACT, KH, KW, SH, SW, VEC, PW, PRO, SUMS, HALF (words: x 0, y 2, e0 3). The window and
 * the stride are template parameters of the common kernels (spg_dwconv_k<KH><KW>s<SH><SW>), whose
 * filters and windows then stay in registers; spg_dwconv takes any from the spec. 256 threads.
 */

#include "common.cuh"

#define APPLY   0u
#define SPREAD  1u
#define WEIGHTS 2u
#define EPI_STORE    0u
#define EPI_BIAS_ACT 1u
#define EPI_DERIV    4u
#define PX 4u

/* What a dispatch's spec gives, with the window and stride (constants when the template has them). */
template <uint32_t KH, uint32_t KW, uint32_t SH, uint32_t SW>
struct Dw {
    const SpgDwconvPush &p;
    uint32_t MODE, EPI, ACT, VEC, PW, PRO, SUMS, HALF;
    float4_ dy_sum, dyx_sum;

    __device__ Dw(const SpgDwconvPush &push, const Spec &s) : p(push) {
        MODE = s.v[0]; EPI = s.v[1]; ACT = s.v[2]; VEC = s.v[7]; PW = s.v[8]; PRO = s.v[9]; SUMS = s.v[10];
        HALF = s.v[11];
        dy_sum = dyx_sum = float4_{0.0f, 0.0f, 0.0f, 0.0f};
    }

    DEVICE float4_ norm_a(const SpgDwconvPush &p, uint32_t pro, uint32_t c) {
        return pro != 0u ? ((const float4_ *)p.bn)[(2u * p.in_c + c) >> 2] : float4_{1.0f, 1.0f, 1.0f, 1.0f};
    }
    DEVICE float4_ norm_b(const SpgDwconvPush &p, uint32_t pro, uint32_t c) {
        return pro != 0u ? ((const float4_ *)p.bn)[(3u * p.in_c + c) >> 2] : float4_{0.0f, 0.0f, 0.0f, 0.0f};
    }
    __device__ float4_ normalized(float4_ v, float4_ a, float4_ b) const {
        if (PRO == 0u) return v;
        return activate4(add4(mul4(v, a), b), PRO - 1u);
    }

    /* the epilogue of value i of the result */
    __device__ void finish(uint32_t i, uint32_t channel, float v) const {
        if (EPI == EPI_BIAS_ACT) {
            v = activate(v + F(p.e0)[channel], ACT);
        } else {
            if (p.beta != 0.0f) v += p.beta * ld(p.y, i, 2u, HALF);
            if (EPI == EPI_DERIV) v *= derivative(ld(p.e0, i, 3u, HALF), ACT);
        }
        st(p.y, i, 2u, HALF, v);
    }

    /* and of values i .. i + 3, channels channel .. + 3 */
    __device__ void finish4(uint32_t i, uint32_t channel, float4_ v) {
        if (EPI == EPI_BIAS_ACT) {
            for (uint32_t k = 0u; k < 4u; k++) set4(v, k, activate(get4(v, k) + F(p.e0)[channel + k], ACT));
        } else {
            if (p.beta != 0.0f) v = add4(v, scale4(ld4(p.y, i, 2u, HALF), p.beta));
            if (EPI == EPI_DERIV) {
                const float4_ x = ld4(p.e0, i, 3u, HALF), e = normalized(x, norm_a(p, PRO, channel), norm_b(p, PRO, channel));
                for (uint32_t k = 0u; k < 4u; k++) set4(v, k, get4(v, k) * derivative(get4(e, k), ACT));
                if (SUMS != 0u) {
                    float4_ g = v;                          /* the gradient as stored */
                    if ((HALF >> 2u) & 1u)
                        g = float4_{from_bf16(to_bf16(v.x)), from_bf16(to_bf16(v.y)), from_bf16(to_bf16(v.z)),
                                    from_bf16(to_bf16(v.w))};
                    const float4_ mean = ((const float4_ *)p.bn)[channel >> 2];
                    dy_sum = add4(dy_sum, g);
                    dyx_sum = fma4(g, float4_{x.x - mean.x, x.y - mean.y, x.z - mean.z, x.w - mean.w}, dyx_sum);
                }
            }
        }
        st4(p.y, i, 2u, HALF, v);
    }

    /* the filters of channels o .. o + 3 at tap t */
    __device__ float4_ w4(uint32_t o, uint32_t t, uint32_t taps) const {
        const float *w = F(p.w);
        return float4_{w[o * taps + t], w[(o + 1u) * taps + t], w[(o + 2u) * taps + t], w[(o + 3u) * taps + t]};
    }
};

/* The kernel body for window KH x KW and stride SH x SW (constants), or any (KH = 0: from the spec, the
   filters kept in arrays of the largest window, which spill). */
template <uint32_t KH_, uint32_t KW_, uint32_t SH_, uint32_t SW_>
DEVICE void dwconv(const SpgDwconvPush &p, const Spec &spec) {
    const uint32_t KH = KH_ ? KH_ : spec.v[3], KW = KW_ ? KW_ : spec.v[4];
    const uint32_t SH = SH_ ? SH_ : spec.v[5], SW = SW_ ? SW_ : spec.v[6];
    constexpr uint32_t TAPS = KH_ ? KH_ * KW_ : SPG_DW_TAPS;
    constexpr uint32_t NCMAX = KH_ ? (PX - 1u) * SW_ + KW_ : (PX - 1u) * SPG_DW_TAPS + SPG_DW_TAPS;
    Dw<KH_, KW_, SH_, SW_> d(p, spec);
    const uint32_t taps = KH * KW;
    __shared__ float4_ red[256], red2[256];
    const uint32_t tid = thread_x();

    if (d.MODE == WEIGHTS) {
        /* runs of WRUN output pixels of a row with VEC 4 and windows of up to 9 taps */
        const uint32_t WRUN = 1u + (PX - 1u) * (d.VEC / 4u) * ((9u / taps + 8u) / 9u);
        const uint32_t quads = p.out_c / d.VEC, lanes = p.lanes, per = 256u / lanes;
        const uint32_t q = block_y() * per + tid % per, lane = tid / per;
        const bool live = lane < lanes && q < quads;
        const uint32_t o = q * d.VEC, c = o / p.og, NC = (WRUN - 1u) * SW + KW, per_row = (p.out_w + WRUN - 1u) / WRUN;
        const uint32_t first = block_x() * p.rows, end = umin(first + p.rows, p.pixels);
        const float4_ na = live ? d.norm_a(p, d.PRO, c) : float4_{1.0f, 1.0f, 1.0f, 1.0f};
        const float4_ nb = live ? d.norm_b(p, d.PRO, c) : float4_{0.0f, 0.0f, 0.0f, 0.0f};
        float4_ acc[TAPS];
        for (uint32_t t = 0u; t < taps; t++) acc[t] = float4_{0.0f, 0.0f, 0.0f, 0.0f};
        for (uint32_t unit = first + lane; live && unit < end; unit += lanes) {
            const uint32_t row = unit / per_row, ox0 = unit % per_row * WRUN, oy = row % p.out_h, n = row / p.out_h;
            float4_ dv[PX];
            bool valid[PX];
            for (uint32_t j = 0u; j < WRUN; j++) {
                const uint32_t at = (row * p.out_w + ox0 + j) * p.out_c + o;
                valid[j] = ox0 + j < p.out_w;
                if (!valid[j]) dv[j] = float4_{0.0f, 0.0f, 0.0f, 0.0f};
                else if (d.VEC == 4u) dv[j] = ld4(p.y, at, 2u, d.HALF);
                else { const float s = ld(p.y, at, 2u, d.HALF); dv[j] = float4_{s, s, s, s}; }
            }
            const int y0 = (int)(oy * SH) - (int)p.ph, x0 = (int)(ox0 * SW) - (int)d.PW;
            for (uint32_t ky = 0u; ky < KH; ky++) {
                const int iy = y0 + (int)ky;
                if (iy < 0 || iy >= (int)p.in_h) continue;
                const uint32_t base = (n * p.in_h + (uint32_t)iy) * p.in_w;
                float4_ v[NCMAX];
                bool inside[NCMAX];
                for (uint32_t k = 0u; k < NC; k++) {
                    const int ix = x0 + (int)k;
                    const uint32_t at = (base + (uint32_t)ix) * p.in_c + c;
                    inside[k] = ix >= 0 && ix < (int)p.in_w;
                    if (!inside[k]) v[k] = float4_{0.0f, 0.0f, 0.0f, 0.0f};
                    else if (d.VEC == 4u) v[k] = d.normalized(ld4(p.x, at, 0u, d.HALF), na, nb);
                    else { const float s = ld(p.x, at, 0u, d.HALF); v[k] = float4_{s, s, s, s}; }
                }
                for (uint32_t kx = 0u; kx < KW; kx++)
                    for (uint32_t j = 0u; j < WRUN; j++)
                        if (valid[j] && inside[j * SW + kx]) acc[ky * KW + kx] = fma4(dv[j], v[j * SW + kx], acc[ky * KW + kx]);
            }
        }
        /* each tap's lanes added pairwise, 1 apart, then 2, 4, ... */
        for (uint32_t t = 0u; t < taps; t++) {
            red[tid] = acc[t];
            barrier();
            for (uint32_t step = 1u; step < lanes; step *= 2u) {
                if (lane % (2u * step) == 0u && lane + step < lanes) red[tid] = add4(red[tid], red[tid + step * per]);
                barrier();
            }
            if (lane == 0u && live)
                for (uint32_t k = 0u; k < d.VEC; k++) F(p.w)[(block_x() * p.out_c + o + k) * taps + t] = get4(red[tid], k);
            barrier();
        }
        return;
    }

    const uint32_t stride = blocks_x() * 256u;
    if (d.VEC == 4u) {
        const bool runs = d.MODE == APPLY || PX % SW == 0u;
        const uint32_t quads = (d.MODE == APPLY ? p.out_c : p.in_c) / 4u, width = d.MODE == APPLY ? p.out_w : p.in_w;
        const uint32_t per_row = runs ? (width + PX - 1u) / PX : width;
        for (uint32_t q = global_x(); q < p.total; q += stride) {
            const uint32_t c = q % quads * 4u, pixel = q / quads, row = pixel / per_row, run = pixel % per_row;
            if (d.MODE == APPLY) {
                /* output pixels ox0 .. + PX - 1 of row `row`, channels c .. c + 3 */
                const uint32_t ox0 = run * PX, NC = (PX - 1u) * SW + KW, oy = row % p.out_h, n = row / p.out_h;
                const int y0 = (int)(oy * SH) - (int)p.ph, x0 = (int)(ox0 * SW) - (int)d.PW;
                const float4_ na = d.norm_a(p, d.PRO, c), nb = d.norm_b(p, d.PRO, c);
                float4_ w[TAPS], sum[PX];
                for (uint32_t t = 0u; t < taps; t++) w[t] = d.w4(c, t, taps);
                for (uint32_t j = 0u; j < PX; j++) sum[j] = float4_{0.0f, 0.0f, 0.0f, 0.0f};
                for (uint32_t ky = 0u; ky < KH; ky++) {
                    const int iy = y0 + (int)ky;
                    if (iy < 0 || iy >= (int)p.in_h) continue;
                    const uint32_t base = (n * p.in_h + (uint32_t)iy) * p.in_w;
                    float4_ v[NCMAX];
                    bool inside[NCMAX];
                    for (uint32_t k = 0u; k < NC; k++) {
                        const int ix = x0 + (int)k;
                        inside[k] = ix >= 0 && ix < (int)p.in_w;
                        v[k] = inside[k] ? d.normalized(ld4(p.x, (base + (uint32_t)ix) * p.in_c + c, 0u, d.HALF), na, nb)
                                         : float4_{0.0f, 0.0f, 0.0f, 0.0f};
                    }
                    for (uint32_t j = 0u; j < PX; j++)
                        for (uint32_t kx = 0u; kx < KW; kx++)
                            if (inside[j * SW + kx]) sum[j] = fma4(v[j * SW + kx], w[ky * KW + kx], sum[j]);
                }
                for (uint32_t j = 0u; j < PX; j++)
                    if (ox0 + j < p.out_w) d.finish4((row * p.out_w + ox0 + j) * p.out_c + c, c, sum[j]);
            } else if (runs) {
                /* input pixels ix0 .. + PX - 1 of row `row`, channels c .. c + 3 */
                const uint32_t ix0 = run * PX, BIAS = 8u * SW;
                const int LO = -(int)((KW - 1u + SW - 1u) / SW), NC = (int)((PX - 1u + d.PW) / SW) - LO + 1;
                const uint32_t iy = row % p.in_h, n = row / p.in_h;
                const int ox0 = (int)(ix0 / SW) + LO;
                float4_ w[TAPS], sum[PX];
                for (uint32_t t = 0u; t < taps; t++) w[t] = d.w4(c, t, taps);
                for (uint32_t j = 0u; j < PX; j++) sum[j] = float4_{0.0f, 0.0f, 0.0f, 0.0f};
                for (uint32_t ky = 0u; ky < KH; ky++) {
                    const int ty = (int)(iy + p.ph) - (int)ky;
                    if (ty < 0 || (uint32_t)ty % SH != 0u || (uint32_t)ty / SH >= p.out_h) continue;
                    const uint32_t base = (n * p.out_h + (uint32_t)ty / SH) * p.out_w;
                    float4_ v[NCMAX + 8u];
                    bool inside[NCMAX + 8u];
                    for (int k = 0; k < NC; k++) {
                        const int ox = ox0 + k;
                        inside[k] = ox >= 0 && ox < (int)p.out_w;
                        v[k] = inside[k] ? ld4(p.x, (base + (uint32_t)ox) * p.out_c + c, 0u, d.HALF) : float4_{0.0f, 0.0f, 0.0f, 0.0f};
                    }
                    for (uint32_t j = 0u; j < PX; j++)
                        for (uint32_t kx = 0u; kx < KW; kx++) {
                            const uint32_t t = j + d.PW + BIAS - kx;
                            const int k = (int)(t / SW) - (int)(BIAS / SW) - LO;
                            if (t % SW == 0u && inside[k]) sum[j] = fma4(v[k], w[ky * KW + kx], sum[j]);
                        }
                }
                for (uint32_t j = 0u; j < PX; j++)
                    if (ix0 + j < p.in_w) d.finish4((row * p.in_w + ix0 + j) * p.in_c + c, c, sum[j]);
            } else {
                const uint32_t i = (row * p.in_w + run) * p.in_c + c, ix = run, iy = row % p.in_h, n = row / p.in_h;
                float4_ sum = {0.0f, 0.0f, 0.0f, 0.0f};
                for (uint32_t ky = 0u; ky < KH; ky++) {
                    const int ty = (int)(iy + p.ph) - (int)ky;
                    if (ty < 0 || (uint32_t)ty % SH != 0u || (uint32_t)ty / SH >= p.out_h) continue;
                    const uint32_t oy = (uint32_t)ty / SH;
                    for (uint32_t kx = 0u; kx < KW; kx++) {
                        const int tx = (int)(ix + d.PW) - (int)kx;
                        if (tx < 0 || (uint32_t)tx % SW != 0u || (uint32_t)tx / SW >= p.out_w) continue;
                        sum = add4(sum, mul4(ld4(p.x, ((n * p.out_h + oy) * p.out_w + (uint32_t)tx / SW) * p.out_c + c, 0u, d.HALF),
                                             d.w4(c, ky * KW + kx, taps)));
                    }
                }
                d.finish4(i, c, sum);
            }
        }
        if (d.MODE == SPREAD && d.SUMS != 0u) {
            /* the block's sums of each channel, its threads' (lanes of in_c / 4) added pairwise */
            const uint32_t quads4 = p.in_c / 4u, lanes = 256u / quads4, lane = tid / quads4;
            red[tid] = d.dy_sum;
            red2[tid] = d.dyx_sum;
            barrier();
            for (uint32_t step = 1u; step < lanes; step *= 2u) {
                if (lane % (2u * step) == 0u && lane + step < lanes) {
                    red[tid] = add4(red[tid], red[tid + step * quads4]);
                    red2[tid] = add4(red2[tid], red2[tid + step * quads4]);
                }
                barrier();
            }
            if (lane == 0u) {
                const uint32_t base = block_x() * 2u * p.in_c + tid * 4u;
                for (uint32_t k = 0u; k < 4u; k++) {
                    F(p.part)[base + k] = get4(red[tid], k);
                    F(p.part)[base + p.in_c + k] = get4(red2[tid], k);
                }
            }
        }
        return;
    }

    const float *W = F(p.w);
    for (uint32_t i = global_x(); i < p.total; i += stride) {
        if (d.MODE == APPLY) {
            const uint32_t o = i % p.out_c, pixel = i / p.out_c, ox = pixel % p.out_w, oy = (pixel / p.out_w) % p.out_h;
            const uint32_t n = pixel / (p.out_w * p.out_h), c = o / p.og;
            const int y0 = (int)(oy * SH) - (int)p.ph, x0 = (int)(ox * SW) - (int)d.PW;
            float sum = 0.0f;
            for (uint32_t ky = 0u; ky < KH; ky++) {
                const int iy = y0 + (int)ky;
                if (iy < 0 || iy >= (int)p.in_h) continue;
                for (uint32_t kx = 0u; kx < KW; kx++) {
                    const int ix = x0 + (int)kx;
                    if (ix < 0 || ix >= (int)p.in_w) continue;
                    sum += ld(p.x, ((n * p.in_h + (uint32_t)iy) * p.in_w + (uint32_t)ix) * p.in_c + c, 0u, d.HALF) *
                           W[o * taps + ky * KW + kx];
                }
            }
            d.finish(i, o, sum);
        } else if (d.MODE == SPREAD) {
            const uint32_t c = i % p.in_c, pixel = i / p.in_c, ix = pixel % p.in_w, iy = (pixel / p.in_w) % p.in_h;
            const uint32_t n = pixel / (p.in_w * p.in_h);
            float sum = 0.0f;
            for (uint32_t m = 0u; m < p.og; m++) {
                const uint32_t o = c * p.og + m;
                for (uint32_t ky = 0u; ky < KH; ky++) {
                    const int ty = (int)(iy + p.ph) - (int)ky;
                    if (ty < 0 || (uint32_t)ty % SH != 0u || (uint32_t)ty / SH >= p.out_h) continue;
                    const uint32_t oy = (uint32_t)ty / SH;
                    for (uint32_t kx = 0u; kx < KW; kx++) {
                        const int tx = (int)(ix + d.PW) - (int)kx;
                        if (tx < 0 || (uint32_t)tx % SW != 0u || (uint32_t)tx / SW >= p.out_w) continue;
                        sum += ld(p.x, ((n * p.out_h + oy) * p.out_w + (uint32_t)tx / SW) * p.out_c + o, 0u, d.HALF) *
                               W[o * taps + ky * KW + kx];
                    }
                }
            }
            d.finish(i, c, sum);
        }
    }
}

/* the unit's instance: SPG_ENTRY and WINDOW (KH, KW, SH, SW; 0, 0, 0, 0 for any) from cmake/Cuda.cmake */
#if !defined(SPG_ENTRY)
#define SPG_ENTRY spg_dwconv
#define WINDOW 0, 0, 0, 0
#endif

extern "C" __global__ void SPG_ENTRY(const SpgDwconvPush p, const Spec spec) { dwconv<WINDOW>(p, spec); }
