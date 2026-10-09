/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/* The parameters of the GPU kernels, the same for both backends: push constants laid out as the
   GLSL blocks of Src/Gpu/Shaders (std430: addresses 8 bytes) and taken by value by the CUDA kernels of
   Src/Gpu/Cuda, and the kernels' modes. Plain C, without includes: uint32_t, uint64_t and bool come from
   whoever includes it (Spingalett.GpuKernels.h, Src/Gpu/Cuda/common.cuh). */

#pragma once

/* gemm.comp */
typedef struct {
    uint64_t a, b, c, e0, e1, geo;
    uint32_t M, N, K, lda, ldb, ldc;
    uint32_t a_group, b_group, c_group;
    uint32_t slices, slice_k;
    float alpha, beta;
    uint32_t flags;
    uint32_t m_tile0;               /* set by spg_gemm(): the first tile of rows of a dispatch */
} SpgGemmPush;

enum { SPG_A_ROW, SPG_A_COL, SPG_A_CONV };
enum { SPG_B_ROW, SPG_B_COL, SPG_B_CONV };
enum { SPG_EPI_STORE, SPG_EPI_BIAS_ACT, SPG_EPI_SCALE_ACT, SPG_EPI_PARTIAL, SPG_EPI_DERIV };
#define SPG_GEMM_BIAS 1u            /* flags: EPI_BIAS_ACT adds e0 */

/* colsum.comp */
typedef struct {
    uint64_t x, dy, k, part;
    uint32_t R, C, slice_rows, pad;
} SpgColsumPush;
enum { SPG_COLSUM_SUM, SPG_COLSUM_SHIFTED, SPG_COLSUM_DY, SPG_COLSUM_LN };

/* reduce.comp */
typedef struct {
    uint64_t part, out;
    uint32_t total, width, slices;
    float scale, beta;
    uint32_t pad;
} SpgReducePush;

/* bn.comp */
typedef struct {
    uint64_t part, x, gamma, beta, rmean, rvar, stats, coef, ggamma, gbeta;
    uint32_t C, slices;
    float m, eps, momentum, scale, beta_g;
    uint32_t pad;
} SpgBnPush;
enum { SPG_BN_TRAIN, SPG_BN_INFER, SPG_BN_BACKWARD };

/* eltwise.comp */
typedef struct {
    uint64_t y, x, a, b, header;
    uint32_t total, n, layer, threshold;
    float keep_scale;
    uint32_t flags;
    uint64_t z;
} SpgEltwisePush;
enum { SPG_ELT_BIAS_ACT, SPG_ELT_DERIV, SPG_ELT_MUL, SPG_ELT_ADD, SPG_ELT_DROPOUT, SPG_ELT_AFFINE, SPG_ELT_BDATA,
       SPG_ELT_SCALE, SPG_ELT_AFFINE_ADD, SPG_ELT_COPY };

/* output.comp */
typedef struct {
    uint64_t y, t, delta, loss;
    uint32_t rows, n;
} SpgOutputPush;
enum { SPG_OUT_SOFTMAX, SPG_OUT_LOSS, SPG_OUT_GRADS };

/* pool.comp */
typedef struct {
    uint64_t x, y, dy, dx;
    uint32_t n, H, W, C, OH, OW, KH, KW, SH, SW, PH, PW;
} SpgPoolPush;

/* upsample.comp */
typedef struct {
    uint64_t x, y, dy, dx;
    uint32_t n, H, W, C, SH, SW, flags, pad;
} SpgUpsamplePush;

/* ln.comp */
typedef struct {
    uint64_t x, y, dy, gamma, beta, stats;
    uint32_t cells, C, flags;
    float eps;
} SpgLnPush;

/* combine.comp */
typedef struct {
    uint64_t x, y, b;
    uint32_t n, cells, C, c0, ck, flags;
} SpgCombinePush;
enum { SPG_COMBINE_ADD, SPG_COMBINE_SLICE, SPG_COMBINE_GAP };

/* dwconv.comp */
typedef struct {
    uint64_t x, w, y, e0, bn, part;
    uint32_t total, in_h, in_w, in_c, out_h, out_w, out_c, og;
    uint32_t ph, pixels, rows, lanes;       /* (the padding in x is a constant of the kernel) */
    float beta;
} SpgDwconvPush;
enum { SPG_DW_APPLY, SPG_DW_SPREAD, SPG_DW_WEIGHTS };
#define SPG_DW_TAPS 49u             /* taps at most */

/* rows.comp */
typedef struct {
    uint64_t index, dst, header;
    uint32_t n, size;
    uint32_t height, width, channels;
} SpgRowsPush;

/* wtrans.comp */
typedef struct {
    uint64_t w, wt, order;
    uint32_t total, OG, CG, taps;
} SpgWtransPush;

/* optim.comp */
typedef struct {
    uint64_t w, m, v, g, header;
    uint32_t n;
    float decay, momentum, beta1, beta2, epsilon;
    uint64_t wh;                    /* the weights' bfloat16 copy, written with them (0: none) */
} SpgOptimPush;

/* sumsq.comp */
typedef struct {
    uint64_t a, b, part, scalars;
    uint32_t n_a, n_b, slices;
    float max_norm;
} SpgSumsqPush;
enum { SPG_SUMSQ_PARTIAL, SPG_SUMSQ_CLIP };

#define SPG_GEO_HEADER  16u         /* uints before the taps of a convolution's geometry (gemm.comp) */
#define SPG_STEP_HEADER 20u         /* uints of the step header (common.glsl) */

/* The CUDA backend's product in single precision (Src/Gpu/Cuda/gemm.cu): the steps of the sum in shared
   memory at once, and the bytes of shared memory a tile of BM x BN, BK a step, takes. */
#define SPG_CUDA_GEMM_STAGES 2u     /* (two buffers) */
#define SPG_CUDA_GEMM_SHARED(bm, bn, bk) (SPG_CUDA_GEMM_STAGES * (bk) * ((bm) + 4u + (bn) + 4u) * 4u)
/* gemm_mma.cu: stages of its pipeline, and bytes of shared memory a tile takes (the larger of an operand's
   layouts: x rows of bk + 8 bfloat16, or bk rows of x + 8) */
#define SPG_CUDA_MMA_STAGES 2u
#define SPG_CUDA_MMA_ROWS(x, bk) ((x) * ((bk) + 8u) > (bk) * ((x) + 8u) ? (x) * ((bk) + 8u) : (bk) * ((x) + 8u))
#define SPG_CUDA_MMA_SHARED(bm, bn, bk) (SPG_CUDA_MMA_STAGES * (SPG_CUDA_MMA_ROWS(bm, bk) + SPG_CUDA_MMA_ROWS(bn, bk)) * 2u)
