/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/*
 * dwconv.comp: depthwise convolutions (APPLY, SPREAD, WEIGHTS) with the epilogues of the products,
 * PRO (a batch normalization applied to x as it is read) and SUMS (the normalization's backward sums
 * of a SPREAD). Every sum runs in the order dwconv.comp's does.
 *
 * Spec: MODE, EPI, ACT, KH, KW, SH, SW, VEC, PW, PRO, SUMS, HALF (words: x 0, y 2, e0 3). The window, the
 * stride, the mode, VEC and the padding are template parameters of the common kernels
 * (spg_dwconv_k<KH><KW>s<SH><SW>_m<MODE>v<VEC>, padded (KW - 1) / 2), whose filters, windows and runs then
 * stay in registers; spg_dwconv takes any from the spec. 256 threads.
 */

#include "common.cuh"

/* (the loops unrolled for the constant windows run to bounds from the spec in spg_dwconv) */
#pragma clang diagnostic ignored "-Wpass-failed"

#define APPLY   0u
#define SPREAD  1u
#define WEIGHTS 2u
#define EPI_STORE    0u
#define EPI_BIAS_ACT 1u
#define EPI_DERIV    4u
#define PX 4u
#define ANY 255u                /* a template parameter taken from the spec */

/* What a dispatch's spec gives, with the window and stride (constants when the template has them). */
template <uint32_t HALF_>
struct Dw {
    const SpgDwconvPush &p;
    uint32_t MODE, EPI, ACT, VEC, PW, PRO, SUMS, HALF;
    float slope, pro_slope;                 /* act_slope() of ACT and of PRO's activation */
    float4_ dy_sum, dyx_sum;

    MEMBER Dw(const SpgDwconvPush &push, const Spec &s) : p(push) {
        MODE = s.v[0]; EPI = s.v[1]; ACT = s.v[2]; VEC = s.v[7]; PW = s.v[8]; PRO = s.v[9]; SUMS = s.v[10];
        HALF = HALF_ != ANY ? HALF_ : s.v[11];
        slope = act_slope(ACT);
        pro_slope = PRO ? act_slope(PRO - 1u) : 1.0f;
        dy_sum = dyx_sum = float4_{0.0f, 0.0f, 0.0f, 0.0f};
    }

    DEVICE float4_ norm_a(const SpgDwconvPush &p, uint32_t pro, uint32_t c) {
        return pro != 0u ? ((const float4_ *)p.bn)[(2u * p.in_c + c) >> 2] : float4_{1.0f, 1.0f, 1.0f, 1.0f};
    }
    DEVICE float4_ norm_b(const SpgDwconvPush &p, uint32_t pro, uint32_t c) {
        return pro != 0u ? ((const float4_ *)p.bn)[(3u * p.in_c + c) >> 2] : float4_{0.0f, 0.0f, 0.0f, 0.0f};
    }
    MEMBER float4_ normalized(float4_ v, float4_ a, float4_ b) const {
        if (PRO == 0u) return v;
        return activate4s(add4(mul4(v, a), b), PRO - 1u, pro_slope);
    }

    /* the epilogue of value i of the result */
    MEMBER void finish(uint32_t i, uint32_t channel, float v) const {
        if (EPI == EPI_BIAS_ACT) {
            v = activate(v + F(p.e0)[channel], ACT);
        } else {
            if (p.beta != 0.0f) v += p.beta * ld(p.y, i, 2u, HALF);
            if (EPI == EPI_DERIV) v *= derivative(ld(p.e0, i, 3u, HALF), ACT);
        }
        st(p.y, i, 2u, HALF, v);
    }

    /* and of values i .. i + 3, channels channel .. + 3 (the normalization of e0 and SUMS: SPREAD only) */
    template <bool SPREADS>
    MEMBER void finish4(uint32_t i, uint32_t channel, float4_ v) {
        if (EPI == EPI_BIAS_ACT) {
            v = activate4s(add4(v, float4_{F(p.e0)[channel], F(p.e0)[channel + 1u], F(p.e0)[channel + 2u], F(p.e0)[channel + 3u]}),
                           ACT, slope);
        } else {
            if (p.beta != 0.0f) v = add4(v, scale4(ld4(p.y, i, 2u, HALF), p.beta));
            if (EPI == EPI_DERIV) {
                const float4_ x = ld4(p.e0, i, 3u, HALF);
                const float4_ e = SPREADS ? normalized(x, norm_a(p, PRO, channel), norm_b(p, PRO, channel)) : x;
                v = mul4(v, float4_{derivative_s(e.x, ACT, slope), derivative_s(e.y, ACT, slope), derivative_s(e.z, ACT, slope),
                                    derivative_s(e.w, ACT, slope)});
                if (SPREADS && SUMS != 0u) {
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
    MEMBER float4_ w4(uint32_t o, uint32_t t, uint32_t taps) const {
        const float *w = F(p.w);
        return float4_{w[o * taps + t], w[(o + 1u) * taps + t], w[(o + 2u) * taps + t], w[(o + 3u) * taps + t]};
    }
};

/* The kernel body for window KH x KW and stride SH x SW (constants), or any (KH = 0: from the spec, the
   filters kept in arrays of the largest window, which spill), and MODE, VEC and PW (or ANY). */
template <uint32_t KH_, uint32_t KW_, uint32_t SH_, uint32_t SW_, uint32_t MODE_, uint32_t VEC_, uint32_t PW_, uint32_t HALF_>
DEVICE void dwconv(const SpgDwconvPush &p, const Spec &spec) {
    const uint32_t KH = KH_ ? KH_ : spec.v[3], KW = KW_ ? KW_ : spec.v[4];
    const uint32_t SH = SH_ ? SH_ : spec.v[5], SW = SW_ ? SW_ : spec.v[6];
    const uint32_t MODE = MODE_ != ANY ? MODE_ : spec.v[0], VEC = VEC_ != ANY ? VEC_ : spec.v[7];
    const uint32_t PW = PW_ != ANY ? PW_ : spec.v[8];
    constexpr uint32_t TAPS = KH_ ? KH_ * KW_ : SPG_DW_TAPS;
    constexpr uint32_t ROWMAX = KW_ ? KW_ : SPG_DW_TAPS;      /* a window's row */
    Dw<HALF_> d(p, spec);
    const uint32_t taps = KH * KW;
    __shared__ float4_ red[256], red2[256];
    const uint32_t tid = thread_x();

    if (MODE == WEIGHTS) {
        /* runs of WRUN output pixels of a row with VEC 4 and windows of up to 9 taps */
        const uint32_t WRUN = 1u + (PX - 1u) * (VEC / 4u) * ((9u / taps + 8u) / 9u);
        const uint32_t quads = p.out_c / VEC, lanes = p.lanes, per = 256u / lanes;
        const uint32_t q = block_y() * per + tid % per, lane = tid / per;
        const bool live = lane < lanes && q < quads;
        const uint32_t o = q * VEC, c = o / p.og, NC = (WRUN - 1u) * SW + KW, per_row = (p.out_w + WRUN - 1u) / WRUN;
        const uint32_t first = block_x() * p.rows, end = umin(first + p.rows, p.pixels);
        const float4_ na = live ? d.norm_a(p, d.PRO, c) : float4_{1.0f, 1.0f, 1.0f, 1.0f};
        const float4_ nb = live ? d.norm_b(p, d.PRO, c) : float4_{0.0f, 0.0f, 0.0f, 0.0f};
        float4_ acc[TAPS];
        #pragma unroll
        for (uint32_t t = 0u; t < taps; t++) acc[t] = float4_{0.0f, 0.0f, 0.0f, 0.0f};
        for (uint32_t unit = first + lane; live && unit < end; unit += lanes) {
            const uint32_t row = unit / per_row, ox0 = unit % per_row * WRUN, oy = row % p.out_h, n = row / p.out_h;
            float4_ dv[PX];
            bool valid[PX];
            #pragma unroll
            for (uint32_t j = 0u; j < WRUN; j++) {
                const uint32_t at = (row * p.out_w + ox0 + j) * p.out_c + o;
                valid[j] = ox0 + j < p.out_w;
                if (!valid[j]) dv[j] = float4_{0.0f, 0.0f, 0.0f, 0.0f};
                else if (VEC == 4u) dv[j] = ld4(p.y, at, 2u, d.HALF);
                else { const float s = ld(p.y, at, 2u, d.HALF); dv[j] = float4_{s, s, s, s}; }
            }
            const int y0 = (int)(oy * SH) - (int)p.ph, x0 = (int)(ox0 * SW) - (int)PW;
            #pragma unroll
            for (uint32_t ky = 0u; ky < KH; ky++) {
                const int iy = y0 + (int)ky;
                if (iy < 0 || iy >= (int)p.in_h) continue;
                const uint32_t base = (n * p.in_h + (uint32_t)iy) * p.in_w;
                /* input column k into the taps it is of (kx = k - j SW): for each tap the pixels in order */
                #pragma unroll
                for (uint32_t k = 0u; k < NC; k++) {
                    const int ix = x0 + (int)k;
                    if (ix < 0 || ix >= (int)p.in_w) continue;
                    const uint32_t at = (base + (uint32_t)ix) * p.in_c + c;
                    float4_ v;
                    if (VEC == 4u) v = d.normalized(ld4(p.x, at, 0u, d.HALF), na, nb);
                    else { const float s = ld(p.x, at, 0u, d.HALF); v = float4_{s, s, s, s}; }
                    #pragma unroll
                    for (uint32_t j = 0u; j < WRUN; j++) {
                        const int kx = (int)k - (int)(j * SW);
                        if (valid[j] && kx >= 0 && kx < (int)KW) acc[ky * KW + kx] = fma4(dv[j], v, acc[ky * KW + kx]);
                    }
                }
            }
        }
        /* each tap's lanes added pairwise, 1 apart, then 2, 4, ... */
        #pragma unroll
        for (uint32_t t = 0u; t < taps; t++) {
            red[tid] = acc[t];
            barrier();
            for (uint32_t step = 1u; step < lanes; step *= 2u) {
                if (lane % (2u * step) == 0u && lane + step < lanes) red[tid] = add4(red[tid], red[tid + step * per]);
                barrier();
            }
            if (lane == 0u && live)
                #pragma unroll
                for (uint32_t k = 0u; k < VEC; k++) F(p.w)[(block_x() * p.out_c + o + k) * taps + t] = get4(red[tid], k);
            barrier();
        }
        return;
    }

    const uint32_t stride = blocks_x() * 256u;
    if (VEC == 4u) {
        const bool runs = MODE == APPLY || PX % SW == 0u;
        const uint32_t quads = (MODE == APPLY ? p.out_c : p.in_c) / 4u, width = MODE == APPLY ? p.out_w : p.in_w;
        const uint32_t per_row = runs ? (width + PX - 1u) / PX : width;
        for (uint32_t q = global_x(); q < p.total; q += stride) {
            const uint32_t c = q % quads * 4u, pixel = q / quads, row = pixel / per_row, run = pixel % per_row;
            if (MODE == APPLY) {
                /* output pixels ox0 .. + PX - 1 of row `row`, channels c .. c + 3 */
                const uint32_t ox0 = run * PX, NC = (PX - 1u) * SW + KW, oy = row % p.out_h, n = row / p.out_h;
                const int y0 = (int)(oy * SH) - (int)p.ph, x0 = (int)(ox0 * SW) - (int)PW;
                const float4_ na = d.norm_a(p, d.PRO, c), nb = d.norm_b(p, d.PRO, c);
                float4_ sum[PX];
                #pragma unroll
                for (uint32_t j = 0u; j < PX; j++) sum[j] = float4_{0.0f, 0.0f, 0.0f, 0.0f};
                #pragma unroll
                for (uint32_t ky = 0u; ky < KH; ky++) {
                    const int iy = y0 + (int)ky;
                    if (iy < 0 || iy >= (int)p.in_h) continue;
                    const uint32_t base = (n * p.in_h + (uint32_t)iy) * p.in_w;
                    float4_ w[ROWMAX];
                    #pragma unroll
                    for (uint32_t kx = 0u; kx < KW; kx++) w[kx] = d.w4(c, ky * KW + kx, taps);
                    /* input column k, as it comes, into the outputs whose windows hold it (tap kx = k - j SW):
                       for each output the taps in order still */
                    #pragma unroll
                    for (uint32_t k = 0u; k < NC; k++) {
                        const int ix = x0 + (int)k;
                        if (ix < 0 || ix >= (int)p.in_w) continue;
                        const float4_ v = d.normalized(ld4(p.x, (base + (uint32_t)ix) * p.in_c + c, 0u, d.HALF), na, nb);
                        #pragma unroll
                        for (uint32_t j = 0u; j < PX; j++) {
                            const int kx = (int)k - (int)(j * SW);
                            if (kx >= 0 && kx < (int)KW) sum[j] = fma4(v, w[kx], sum[j]);
                        }
                    }
                }
                #pragma unroll
                for (uint32_t j = 0u; j < PX; j++)
                    if (ox0 + j < p.out_w) d.template finish4<false>((row * p.out_w + ox0 + j) * p.out_c + c, c, sum[j]);
            } else if (runs) {
                /* input pixels ix0 .. + PX - 1 of row `row`, channels c .. c + 3 */
                const uint32_t ix0 = run * PX, BIAS = 8u * SW;
                const int LO = -(int)((KW - 1u + SW - 1u) / SW), NC = (int)((PX - 1u + PW) / SW) - LO + 1;
                const uint32_t iy = row % p.in_h, n = row / p.in_h;
                const int ox0 = (int)(ix0 / SW) + LO;
                float4_ sum[PX];
                #pragma unroll
                for (uint32_t j = 0u; j < PX; j++) sum[j] = float4_{0.0f, 0.0f, 0.0f, 0.0f};
                (void)BIAS;
                #pragma unroll
                for (uint32_t ky = 0u; ky < KH; ky++) {
                    const int ty = (int)(iy + p.ph) - (int)ky;
                    if (ty < 0 || (uint32_t)ty % SH != 0u || (uint32_t)ty / SH >= p.out_h) continue;
                    const uint32_t base = (n * p.out_h + (uint32_t)ty / SH) * p.out_w;
                    float4_ w[ROWMAX];
                    #pragma unroll
                    for (uint32_t kx = 0u; kx < KW; kx++) w[kx] = d.w4(c, ky * KW + kx, taps);
                    /* output column ox0 + k, from the last, into the input pixels it reaches (tap
                       kx = j + PW - SW (k + LO)): for each input pixel the taps in order still */
                    #pragma unroll
                    for (int k = NC - 1; k >= 0; k--) {
                        const int ox = ox0 + k;
                        if (ox < 0 || ox >= (int)p.out_w) continue;
                        const float4_ v = ld4(p.x, (base + (uint32_t)ox) * p.out_c + c, 0u, d.HALF);
                        #pragma unroll
                        for (uint32_t j = 0u; j < PX; j++) {
                            const int kx = (int)(j + PW) - (int)SW * (k + LO);
                            if (kx >= 0 && kx < (int)KW) sum[j] = fma4(v, w[kx], sum[j]);
                        }
                    }
                }
                #pragma unroll
                for (uint32_t j = 0u; j < PX; j++)
                    if (ix0 + j < p.in_w) d.template finish4<true>((row * p.in_w + ix0 + j) * p.in_c + c, c, sum[j]);
            } else {
                const uint32_t i = (row * p.in_w + run) * p.in_c + c, ix = run, iy = row % p.in_h, n = row / p.in_h;
                float4_ sum = {0.0f, 0.0f, 0.0f, 0.0f};
                #pragma unroll
                for (uint32_t ky = 0u; ky < KH; ky++) {
                    const int ty = (int)(iy + p.ph) - (int)ky;
                    if (ty < 0 || (uint32_t)ty % SH != 0u || (uint32_t)ty / SH >= p.out_h) continue;
                    const uint32_t oy = (uint32_t)ty / SH;
                    #pragma unroll
                    for (uint32_t kx = 0u; kx < KW; kx++) {
                        const int tx = (int)(ix + PW) - (int)kx;
                        if (tx < 0 || (uint32_t)tx % SW != 0u || (uint32_t)tx / SW >= p.out_w) continue;
                        sum = add4(sum, mul4(ld4(p.x, ((n * p.out_h + oy) * p.out_w + (uint32_t)tx / SW) * p.out_c + c, 0u, d.HALF),
                                             d.w4(c, ky * KW + kx, taps)));
                    }
                }
                d.template finish4<true>(i, c, sum);
            }
        }
        if (MODE == SPREAD && d.SUMS != 0u) {
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
                #pragma unroll
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
        if (MODE == APPLY) {
            const uint32_t o = i % p.out_c, pixel = i / p.out_c, ox = pixel % p.out_w, oy = (pixel / p.out_w) % p.out_h;
            const uint32_t n = pixel / (p.out_w * p.out_h), c = o / p.og;
            const int y0 = (int)(oy * SH) - (int)p.ph, x0 = (int)(ox * SW) - (int)PW;
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
        } else if (MODE == SPREAD) {
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
                        const int tx = (int)(ix + PW) - (int)kx;
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

/* the unit's instance: SPG_ENTRY and WINDOW (KH, KW, SH, SW, MODE, VEC, PW; 0, 0, 0, 0, ANY, ANY, ANY for
   any) from cmake/Cuda.cmake */
#if !defined(SPG_ENTRY)
#define SPG_ENTRY spg_dwconv
#define WINDOW 0, 0, 0, 0, ANY, ANY, ANY
#endif

/* (at most 64 registers a thread in APPLY and SPREAD, four blocks an SM: the loads of more warps in flight
   hide each other's latency better than the loads a thread would keep ahead with more) */
template <uint32_t KH, uint32_t KW, uint32_t SH, uint32_t SW, uint32_t MODE, uint32_t VEC, uint32_t PW>
struct Occupancy {
    static constexpr uint32_t BLOCKS = MODE == APPLY || MODE == SPREAD ? 4u : 2u;
};

extern "C" __global__ void __attribute__((launch_bounds(256, Occupancy<WINDOW>::BLOCKS)))
SPG_ENTRY(const SpgDwconvPush p, const Spec spec) {
    if (spec.v[11] == 0u) dwconv<WINDOW, 0u>(p, spec);      /* single precision: loads without a branch */
    else dwconv<WINDOW, ANY>(p, spec);
}
