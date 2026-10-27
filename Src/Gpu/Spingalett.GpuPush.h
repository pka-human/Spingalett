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
enum { SPG_EPI_STORE, SPG_EPI_BIAS_ACT, SPG_EPI_SCALE_ACT, SPG_EPI_PARTIAL, SPG_EPI_DERIV,
       SPG_EPI_BIAS_PRE };          /* (Vulkan's EPI_BIAS_ACT with SPG_GEMM_PRE: a pipeline of its own, set by
                                       spg_gemm(), so that the others stay as they were) */
#define SPG_GEMM_BIAS 1u            /* flags: EPI_BIAS_ACT adds e0 */
#define SPG_GEMM_PRE  2u            /* EPI_BIAS_ACT also stores its values before the activation at e1, laid out
                                       and kept like C (gemm.comp, gemm_mma.comp; CUDA's units with GENERAL and
                                       epi.cu) */
#define SPG_GEMM_BAND_SHIFT 8u      /* CUDA's gemm.cu: bits 8 to 11, the log of the tiles of rows a band of blocks
                                       takes (gemm_common.cuh's band_tile()), which the backend sets */

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
enum { SPG_BN_TRAIN, SPG_BN_INFER, SPG_BN_BACKWARD, SPG_BN_FOLD };

/* eltwise.comp */
typedef struct {
    uint64_t y, x, a, b, header;
    uint32_t total, n, layer, threshold;
    float keep_scale;
    uint32_t flags;
    uint64_t z;
} SpgEltwisePush;
enum { SPG_ELT_BIAS_ACT, SPG_ELT_DERIV, SPG_ELT_MUL, SPG_ELT_ADD, SPG_ELT_DROPOUT, SPG_ELT_AFFINE, SPG_ELT_BDATA,
       SPG_ELT_SCALE, SPG_ELT_AFFINE_ADD, SPG_ELT_COPY, SPG_ELT_ACT, SPG_ELT_PRODUCT };
#define SPG_ELT_FLAG_A   1u         /* BIAS_ACT adds a */
#define SPG_ELT_FLAG_ADD 2u         /* PRODUCT adds to y */

/* output.comp */
typedef struct {
    uint64_t y, t, delta, loss, part;
    uint32_t rows, n, cells;
    float smoothing;
    uint32_t flags, pad;
} SpgOutputPush;
enum { SPG_OUT_SOFTMAX, SPG_OUT_LOSS, SPG_OUT_GRADS, SPG_OUT_COUNT, SPG_OUT_SUM };
#define SPG_OUT_ACCURACY 1u         /* SUM: the samples' accuracies too, after their losses */

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
#define SPG_LN_STATS 1u             /* flags: keep each cell's statistics (training) */
#define SPG_LN_RMS   2u             /* RMS normalization */

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

/* embed.comp */
typedef struct {
    uint64_t x, w, y, pos, keys, part, dy, gw;
    uint32_t total, tokens, d, vocab, n, shift, blocks, flags;
    float scale, beta;
} SpgEmbedPush;
enum { SPG_EMBED_FORWARD, SPG_EMBED_KEYS, SPG_EMBED_HIST, SPG_EMBED_SCAN, SPG_EMBED_SCATTER, SPG_EMBED_GRAD };
#define SPG_EMBED_FROM 1u           /* the sort's keys in its second pair of arrays */
#define SPG_EMBED_POS  2u           /* FORWARD adds the positions' vectors */
#define SPG_EMBED_ROWS 4u           /* FORWARD: n rows of one token each, at the positions keys[] gives (decoding) */
#define SPG_EMBED_BLOCK 256u        /* keys a block of HIST and SCATTER */

/* attn.comp */
typedef struct {
    uint64_t x, y, dy, dx, stats, table;
    uint32_t n, cells, heads, kv, d, flags;
    float scale;
    uint32_t window;                /* SPG_ATTN_EXT: keys less than window cells away (0: all; less than cells) */
    /* SPG_ATTN_EXT: padding (SPG_ATTN_PAD), the keys' tokens: tokens[s tokens_stride + tokens_at + j] for key j of
       sample s, a negative one a padded key (without, an address and stride that keep loads in x) */
    uint64_t tokens;
    uint32_t tokens_stride, tokens_at;
    /* SPG_ATTN_EXT with SPG_ATTN_CROSS: the keys and values (word 12), 2 kv d floats a cell of kcells cells a sample
       (x then holds the queries, heads d floats a cell; DQ's dx their gradient, DKV's that of kvx); other passes of
       EXT x (loads that stay in x), the others 0 */
    uint64_t kvx;
    uint32_t kcells, pad_;
} SpgAttnPush;
enum { SPG_ATTN_FORWARD, SPG_ATTN_PRE, SPG_ATTN_DQ, SPG_ATTN_DKV };
#define SPG_ATTN_EXT    4u          /* OP: the pass of a layer with padding, a window or cross-attention, the units
                                       that read them (CUDA: _x) */
#define SPG_ATTN_CAUSAL 1u
#define SPG_ATTN_ROPE   2u
#define SPG_ATTN_STATS  4u          /* FORWARD keeps each query's log of the softmax's sum */
#define SPG_ATTN_PAD    8u          /* (EXT) keys of padding left out */
#define SPG_ATTN_CROSS  16u         /* (EXT) cross-attention: keys and values in kvx */
/* rattn.cu (CUDA, single precision, heads up to 128): rows of a block (queries; keys in DKV) and its threads,
   for a unit's head size d (32, 64, 128) and pass */
#define SPG_RATTN_ROWS(d, op) ((d) <= 64u || (op) == SPG_ATTN_FORWARD ? 64u : 32u)
#define SPG_RATTN_THREADS(d, op) \
    ((op) == SPG_ATTN_PRE ? 256u : (d) > 64u && (op) != SPG_ATTN_FORWARD ? 64u : 128u)

/* attn_cache.comp: attention of rows (tokens of sequences at positions: rows[2r], rows[2r + 1]) over a cache of
   keys then values ([sequences][length][kv][d] each); with a sliding window, over the last `window` positions up to
   a row's own (0: all of them) */
typedef struct {
    uint64_t x, y, cache, rows, table;
    uint32_t n, heads, kv, d, length, sequences, flags, cells;
    float scale;
    uint32_t window;
} SpgCachePush;
enum { SPG_CACHE_APPEND, SPG_CACHE_ATTEND };
#define SPG_CACHE_THREADS 64u       /* Vulkan's workgroups of ATTEND (a row and head); CUDA's: a warp a row and head */

/* iquant.comp: the activations of integer models quantized as spingalett_quantize_activations() quantizes them, the
   len values of each of n samples (stride floats apart) to q (qstride values apart: floats, or bfloat16 with HALF)
   and the sample's scale to xs, a sample in slices of slice_len values (the slices' largest magnitudes in part) */
typedef struct {
    uint64_t x, q, xs, part;
    uint32_t n, len, stride, qstride, slices, slice_len, flags, pad;
} SpgIquantPush;
enum { SPG_IQ_WHOLE, SPG_IQ_MAX, SPG_IQ_QUANT };
#define SPG_IQ_VEC4  1u             /* four values at a time (sizes and addresses in fours) */
#define SPG_IQ_SLICE 4096u          /* values a slice */

/* ipass.comp: the passes of integer models other than their products, every value computed as the CPU computes it
   (Spingalett.Model.c, Spingalett.Inference.c): spec OP, ACT (none, ReLU or leaky ReLU), HALF */
typedef struct {
    uint64_t x, y, w, e0, e1, e2;
    uint32_t total, n;
    uint32_t in_h, in_w, in_c, out_h, out_w, out_c;
    uint32_t kh, kw, sh, sw, ph, pw;
    uint32_t groups, slices, c0, flags;
    float inv;
    uint32_t pad;
} SpgIpassPush;
enum { SPG_IP_EPI, SPG_IP_DIRECT, SPG_IP_POOL, SPG_IP_GAP, SPG_IP_ADD, SPG_IP_MUL, SPG_IP_SLICE, SPG_IP_BN, SPG_IP_UP };
#define SPG_IP_FIRST      1u        /* ADD, MUL: y = x op e0, else y = y op x */
#define SPG_IP_LAST       2u        /* ADD, MUL: the activation after it */
#define SPG_IP_MAX        1u        /* POOL: max pooling, else average pooling */
#define SPG_IP_TRANSPOSED 1u        /* DIRECT: a transposed convolution's windows */
#define SPG_IP_BILINEAR   1u        /* UP: bilinear (its table at e0), else the nearest */
#define SPG_IP_FLOATS     2u        /* DIRECT: the sums in floats (exact: few enough of them) */
#define SPG_IP_VEC4       4u        /* EPI, ADD, MUL, BN, SLICE: four values at a time (outputs, channels, addresses in
                                       fours) */
#define SPG_IP_DW4        8u        /* DIRECT with FLOATS: a depthwise convolution's four channels at a time */
#define SPG_IP_OG4        16u       /* DIRECT with FLOATS: four filters of a group at a time */

/* bquant.comp: rows of activations quantized in blocks of bv values (32 or 256) as spingalett_quantize_blocks()
   quantizes them, the bits of the CPU: x [rows][ldx floats] to codes q [rows][k] (signed bytes), a scale a block dx
   [rows][k / bv] and the sum of every 16 codes xs [rows][k / 16] (int32). A warp a block. */
typedef struct {
    uint64_t x, q, dx, xs;
    uint32_t rows, k, bv, ldx;
} SpgBquantPush;

/* bdot.comp: rows of activations quantized by bquant times a linear layer's weights of a block precision (TYPE: the
   precision less SPINGALETT_PRECISION_Q8_0), y[r][j] = bias[j] + spingalett_block_dot() of row j, the bits of the CPU:
   a row of weights `stride` bytes of whole blocks as GGUF stores them, k values; rows r0 to r0 + rows - 1 of the
   bquant arrays (SPG_BDOT_ROWS at most a dispatch), y's rows ldy floats apart. A warp a row of weights. */
typedef struct {
    uint64_t w, q, dx, xs, bias, y;
    uint32_t rows, n, k, stride, r0, ldy;
} SpgBdotPush;
#define SPG_BDOT_ROWS 8u
enum { SPG_B_Q8_0, SPG_B_Q4_0, SPG_B_Q4_1, SPG_B_Q5_0, SPG_B_Q5_1, SPG_B_Q2_K, SPG_B_Q3_K, SPG_B_Q4_K, SPG_B_Q5_K,
       SPG_B_Q6_K, SPG_B_IQ4_NL, SPG_B_IQ4_XS, SPG_B_TYPES };

/* wqmv.comp: products of rows of activations with a linear layer's weights kept as integer codes (INT8, INT4, INT2:
   BITS) and a scale a row of weights, the codes widened in registers: y[r][n] = act(bias[n] + scale[n] sum_k
   x[r][k] code[n][k]) for the `rows` rows from r0 on (SPG_WQ_ROWS at most). A row of codes takes `stride` bytes (a
   multiple of 16), its codes in the image's order. */
typedef struct {
    uint64_t x, y, codes, scale, bias;
    uint32_t rows, N, K, stride, r0, pad;
} SpgWqPush;
#define SPG_WQ_ROWS 8u
/* rows of weights 32 invocations take in a pass of `rows` rows of x (a workgroup of 256: eight times that) */
#define SPG_WQ_NR(rows) ((rows) >= 8u ? 2u : 4u)

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

/* gew.comp: the element-wise operators of graphs (Spingalett.GpuGraph.c): output value i = OP(a, b, c) at its position,
   each operand read through its strides over the output's axes (0 where it broadcasts), SPG_GEW_AXES of them at most
   after merging (innermost last, sizes 1 before fewer). Spec: OP, TYPES (the types of a, b, c and y, four bits each,
   from the lowest), FLAGS (two bits an operand, a's lowest: 0 through its strides, 1 at the output's index, 2 its first
   value only). Unary operators below SPG_GEW_ADD, binary below SPG_GEW_WHERE. */
#define SPG_GEW_AXES 4u
typedef struct {
    uint64_t y, a, b, c;
    uint32_t total, pad;
    uint32_t dims[SPG_GEW_AXES];
    uint32_t sa[SPG_GEW_AXES], sb[SPG_GEW_AXES], sc[SPG_GEW_AXES];
    float alpha, beta;
} SpgGewPush;
enum { SPG_T_F32, SPG_T_U8, SPG_T_I32, SPG_T_I64, SPG_T_BOOL, SPG_T_I8, SPG_T_BF16 };     /* (bfloat16: read only) */
enum {
    SPG_GEW_COPY, SPG_GEW_NEG, SPG_GEW_ABS, SPG_GEW_EXP, SPG_GEW_LOG, SPG_GEW_SQRT, SPG_GEW_RECIPROCAL, SPG_GEW_SIGMOID,
    SPG_GEW_TANH, SPG_GEW_RELU, SPG_GEW_LEAKY, SPG_GEW_ELU, SPG_GEW_SELU, SPG_GEW_GELU, SPG_GEW_GELU_TANH, SPG_GEW_SILU,
    SPG_GEW_HARD_SIGMOID, SPG_GEW_HARD_SWISH, SPG_GEW_SOFTPLUS, SPG_GEW_SOFTSIGN, SPG_GEW_SIN, SPG_GEW_COS, SPG_GEW_ERF,
    SPG_GEW_FLOOR, SPG_GEW_CEIL, SPG_GEW_ROUND, SPG_GEW_SIGN, SPG_GEW_CLIP, SPG_GEW_NOT, SPG_GEW_ISNAN, SPG_GEW_ISINF,
    SPG_GEW_MISH, SPG_GEW_CELU, SPG_GEW_THRESHOLDED_RELU, SPG_GEW_TO_HALF, SPG_GEW_TO_BF16, SPG_GEW_AFFINE,
    SPG_GEW_ADD = 48, SPG_GEW_SUB, SPG_GEW_MUL, SPG_GEW_DIV, SPG_GEW_POW, SPG_GEW_MAX, SPG_GEW_MIN, SPG_GEW_PRELU,
    SPG_GEW_EQUAL, SPG_GEW_LESS, SPG_GEW_LESS_EQUAL, SPG_GEW_GREATER, SPG_GEW_GREATER_EQUAL, SPG_GEW_AND, SPG_GEW_OR,
    SPG_GEW_XOR, SPG_GEW_FMOD, SPG_GEW_MOD,
    SPG_GEW_WHERE = 96
};

/* gcopy.comp: values copied through strides (Transpose, Slice, Concat, Split, Expand, Tile and Pad of graphs): value i
   of the dispatch, at its coordinates over SPG_COPY_AXES axes (innermost last, sizes 1 before fewer), is
   x[base + sum c[d] sx[d]] written to y[sum c[d] sy[d]] (strides in elements; x's may be negative, base keeping its
   offsets from x's address, its lowest value read, at 0 and above). Spec: SIZE (bytes an element: 1, 2 (CUDA's), 4, 8 or
   16). */
#define SPG_COPY_AXES 6u
typedef struct {
    uint64_t y, x;
    uint32_t total, base;
    uint32_t dims[SPG_COPY_AXES];
    int32_t sx[SPG_COPY_AXES], sy[SPG_COPY_AXES];
} SpgCopyPush;

/* gcvt.comp: floats copied as gcopy copies them (SpgCopyPush, its strides in elements), each rounded to bfloat16 (to
   the nearest, ties to even) as it is written: the operands of products on the matrix units. */

/* ggather.comp: Gather of graphs: y[o][j][e] = x[o][idx[j]][e] over `outer` x count x inner values (an index below 0
   counts from the axis's end; one outside it gives zeros). Spec: SIZE (bytes an element: 1, 4 or 8), the indices'
   type (SPG_T_I32 or SPG_T_I64). */
typedef struct {
    uint64_t y, x, idx;
    uint32_t total, inner, count, axis;
} SpgGatherPush;

/* grow.comp: operators of graphs over rows: row q of outer x inner (o = q / inner, i = q % inner) holds the len values
   x[(o len + l) inner + i]; TPR threads take a row (1 where inner is above 1: neighbouring threads then read
   neighbouring rows), each its values l = lane, lane + TPR, ... in order, their partial results combined in a fixed
   tree. Reductions write a value a row to y (ArgMax and ArgMin an int64), the others a row of len values (LayerNorm
   with its means and inverse deviations at aux, rows each). Spec: OP (SPG_ROW_*), TPR. */
typedef struct {
    uint64_t x, y, a, b, aux;
    uint32_t outer, len, inner, rows;
    uint32_t groups, cpg, pix, flags;       /* GroupNorm: groups, channels a group, values a channel */
    float eps;
    uint32_t act;                   /* GroupNorm: the activation of its outputs (ActivationFunction) */
} SpgRowPush;
enum {
    SPG_ROW_SUM, SPG_ROW_MEAN, SPG_ROW_MAX, SPG_ROW_MIN, SPG_ROW_PROD, SPG_ROW_L1, SPG_ROW_L2, SPG_ROW_SUMSQ,
    SPG_ROW_LOGSUMEXP, SPG_ROW_LOGSUM, SPG_ROW_ARGMAX, SPG_ROW_ARGMIN,
    SPG_ROW_SOFTMAX = 16, SPG_ROW_LOGSOFTMAX, SPG_ROW_HARDMAX, SPG_ROW_LAYERNORM, SPG_ROW_RMSNORM, SPG_ROW_GROUPNORM,
    SPG_ROW_GROUPNORM_LAST          /* channels last: row (batch, group) of pix pixels' cpg channels, every groups cpg
                                       values a pixel */
};
#define SPG_ROW_LAST     1u         /* ArgMax, ArgMin: the last of equal values */
#define SPG_ROW_A_ONE    2u         /* LayerNorm, RMSNorm: one scale for every column (else a column each) */
#define SPG_ROW_B        4u         /* LayerNorm: a bias (b), like the scale */
#define SPG_ROW_B_ONE    8u
#define SPG_ROW_STATS    16u        /* LayerNorm: the means and inverse deviations at aux */
#define SPG_ROW_BY_GROUP 32u        /* GroupNorm: the scale and bias a group (opset 18), else a channel */
#define SPG_ROW_HALF_Y   64u        /* LayerNorm, RMSNorm, GroupNorm: y kept as bfloat16 (CUDA's) */

/* gattn.comp: FusedAttention of graphs, softmax(Where(keep, scale Q K^T, fill) + mask) V in blocks of SPG_GATTN_ROWS
   queries of a batch and head (block_y: batch heads + head), over tiles of keys, the softmax online. Q, K, V and Y are
   rows of d values (a unit's head size D at most), their batches, heads and rows strides[] apart (elements: queries
   0..2, keys 3..5, values 6..8, outputs 9..11); the mask (floats added, or booleans that leave keys out) and keep
   (booleans: Where's condition) strides 12..15 and 16..19 over batch, head, query and key (0 where they broadcast),
   a key's values strides[20] apart (K^T's columns, else 1). A row of no key gives zeros with SAFE, else NaN. Spec: none (a unit per D: gattn_d32, _d64, _d128). */
typedef struct {
    uint64_t q, k, v, y, mask, keep, strides;
    uint32_t queries, keys, d, heads;
    float scale, fill;
    uint32_t flags, pad;
} SpgGattnPush;
#define SPG_GATTN_ROWS 64u
#define SPG_GATTN_THREADS 128u
#define SPG_GATTN_MASK  1u          /* floats added to the scores */
#define SPG_GATTN_BOOL  2u          /* booleans: false leaves a key out */
#define SPG_GATTN_KEEP  4u
#define SPG_GATTN_SAFE  8u
#define SPG_GATTN_VEC   16u         /* heads of exactly D values, rows and addresses in eights: loads of eight */
#define SPG_GATTN_HALF_Q 32u        /* gflash: queries, keys, values, the output kept as bfloat16 */
#define SPG_GATTN_HALF_K 64u
#define SPG_GATTN_HALF_V 128u
#define SPG_GATTN_HALF_Y 256u

/* gpool.comp: MaxPool, AveragePool and LpPool of graphs over channels-first maps of up to three spatial axes (as three,
   the first of size 1 for fewer): output value (row, o0, o1, o2) of rows (batch x channels) maps from its window. */
typedef struct {
    uint64_t x, y;
    uint32_t rows, total;
    uint32_t in[3], out[3], k[3], s[3], pb[3], pe[3], dil[3];
    uint32_t flags;
    float p;
} SpgPoolGPush;
#define SPG_GPOOL_MAX     1u
#define SPG_GPOOL_LP      2u
#define SPG_GPOOL_INCLUDE 4u        /* AveragePool: padding counted (count_include_pad) */
#define SPG_GPOOL_LAST    8u        /* channels last ([N, H, W, C]): rows the channels, the batch the rest */

/* gresize.comp: Resize of graphs through the taps of ir_resize_taps(): output value i at (o0, o1, o2, o3) (four axes,
   sizes 1 before fewer) is the sum over every axis d's T[d] taps of the product of their weights times the input at
   their indices (an index below 0: the extrapolation value); the taps a pair (index, weight bits) each, axis d's from
   pair at[d] on, T[d] an output index. */
typedef struct {
    uint64_t x, y, taps;
    uint32_t total, pad;
    uint32_t dims[4], xs[4], T[4], at[4];
    float extrapolation, pad2;
} SpgResizePush;

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
/* gemm_pipe.cu: bytes of shared memory a tile takes (its stages of BM + BN rows of BK bfloat16, unpadded) */
#define SPG_CUDA_PIPE_SHARED(bm, bn, bk, stages) ((stages) * ((bm) + (bn)) * (bk) * 2u)
