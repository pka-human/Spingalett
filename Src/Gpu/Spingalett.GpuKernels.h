/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/* The parameters of the GPU kernels (Src/Gpu/Shaders): push constants laid out as their GLSL blocks
   (std430: addresses 8 bytes), their modes, and the matrix product with its choice of tiles. */

#pragma once

#include "Spingalett.Vulkan.h"

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

typedef struct {
    uint32_t amode, bmode, epi, act;
    uint32_t groups;                /* in z, times the slices */
    bool phased;                    /* rows are a phase of a convolution's data gradient */
    bool vec_a, vec_b;              /* four values of the contiguous axis at once may be read (the
                                       layout allows it; addresses and sizes are checked here) */
    uint32_t tile;                  /* 1 + an index of the tile table to use it (tests), 0: chosen */
    uint64_t c_floats;              /* floats from c that the product may write: with them, the tile
                                       is chosen by timing the candidates on a scratch copy of C */
    bool bf16;                      /* operands rounded to bfloat16 on the matrix units (when the device
                                       has them; single precision otherwise) */
    uint32_t half;                  /* with bf16 on the matrix units: A, B, C, e0 (bits 0 to 3) kept in
                                       memory as bfloat16 (spg_gpu_bf16_storage()) */
    bool wide_a, wide_b;            /* operands kept as bfloat16: eight values of the contiguous axis at
                                       once may be read (convolutions: channels a group multiples of
                                       eight; checked here as vec_a and vec_b are) */
} SpgGemmMode;

/* Records C = A B (gemm.comp) with the tile that suits M, N and the workgroups in z (no barrier). */
void spg_gemm(SpgGpuCommands *c, SpgGemmPush *p, const SpgGemmMode *mode);
/* The workgroups spg_gemm() runs for an M x N product with z in z. */
uint64_t spg_gemm_workgroups(uint32_t M, uint32_t N, uint32_t z);
/* Slices (of *slice_k, a multiple of 8) to split a sum over K into, so that a product with few
   outputs still fills the device; their partial products take slices x G x M x N floats, at most
   SPG_SPLIT_FLOATS. */
#define SPG_SPLIT_FLOATS (1u << 22)
uint32_t spg_gemm_split(uint32_t M, uint32_t N, uint32_t K, uint32_t G, uint32_t *slice_k);
/* The tiles of gemm.comp, or of gemm_mma.comp with mma: their number, and rows x columns x k-step
   and rows x columns per thread (gemm_mma.comp: accumulators per subgroup) of one. */
uint32_t spg_gemm_tiles(bool mma);
void spg_gemm_tile(bool mma, uint32_t index, uint32_t *bm, uint32_t *bn, uint32_t *bk, uint32_t *tm, uint32_t *tn);

/* A monotonic clock, in seconds. */
double spg_seconds(void);

/* Frees what the tile choice keeps (scratch memory); the choices themselves are kept. */
void spg_gemm_release(void);

/* The geometries of a convolution for gemm.comp: the forward one (also for weight gradients), then
   one per phase of the data gradient (input pixels (py + sh y, px + sw x)), whose products read the
   weights regrouped by wtrans.comp in the tap order `order`. */
#define SPG_MAX_PHASES 64u
typedef struct {
    uint32_t phases;                /* sh x sw */
    struct {
        size_t at;                  /* uints from the start */
        uint32_t rh, rw;            /* its pixels per sample: rh x rw */
        uint32_t taps;              /* taps that reach them */
        uint32_t first;             /* its first tap in the regrouped weights */
    } phase[SPG_MAX_PHASES];
    size_t order;                   /* uints from the start: the tap order (one per tap) */
    size_t size;                    /* uints in all */
} SpgConvGeometry;

/* Fills `info` and, unless geo is NULL, the geometries, for a layer with `in` channels in an h x w
   input, `out` channels in an oh x ow output, `groups` groups. False when the stride has over
   SPG_MAX_PHASES phases or a tap does not fit its byte. */
bool spg_conv_geometry(uint32_t *geo, SpgConvGeometry *info, uint32_t h, uint32_t w, uint32_t in, uint32_t oh,
                       uint32_t ow, uint32_t out, uint32_t groups, uint32_t kh, uint32_t kw, uint32_t sh, uint32_t sw,
                       uint32_t ph, uint32_t pw);

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
       SPG_ELT_SCALE, SPG_ELT_AFFINE_ADD };

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
} SpgOptimPush;

/* sumsq.comp */
typedef struct {
    uint64_t a, b, part, scalars;
    uint32_t n_a, n_b, slices;
    float max_norm;
} SpgSumsqPush;
enum { SPG_SUMSQ_PARTIAL, SPG_SUMSQ_CLIP };

#define SPG_GEO_HEADER  16u         /* uints before the taps of a convolution's geometry (gemm.comp) */
#define SPG_STEP_HEADER 16u         /* uints of the step header (common.glsl) */
