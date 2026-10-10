/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/* Internals of the inference engine (Spingalett.Inference.c) shared with the rest of the library:
   the .slett format version 3 to 6 layout (docs/ModelFormat.md), CRC-32, half and bfloat16
   conversion, and the kernels batched inference reuses so that it computes exactly what
   spingalett_model_run computes. */

#pragma once

/* The library's sources use the names of 0.x (Spingalett.Short.h). */
#if !defined(SPINGALETT_SHORT_NAMES)
#define SPINGALETT_SHORT_NAMES
#endif
#include <Spingalett/Spingalett.Inference.h>
#include <math.h>
#include <string.h>

#define SLETT_MAGIC              "SLETTM"
#define SLETT_HEADER_SIZE        64u
#define SLETT_LAYER_ENTRY_SIZE   48u          /* version 3 */
#define SLETT_LAYER_ENTRY_SIZE_4 64u          /* version 4: shapes and windows added */
#define SLETT_LAYER_ENTRY_SIZE_5 80u          /* version 5: groups and batch normalization added */
#define SLETT_LAYER_ENTRY_SIZE_6 112u         /* version 6: inputs and the output's place added */
#define SLETT_MAX_EXTENT         65535u       /* largest height, width, kernel, stride or padding */
#define SLETT_SECTION_ALIGN      16u          /* every section starts at a multiple of this */
#define SLETT_FLAG_OPTIMIZER     0x01u
#define SLETT_MAX_LAYERS         65536u
#define SLETT_MAX_INT_INPUTS     131072u      /* 127 * 127 * inputs must fit in an int32 */

static inline bool spingalett_precision_is_int(PrecisionMode p) {
    return p == PRECISION_INT8 || p == PRECISION_INT4 || p == PRECISION_INT2;
}

/* One entry of the layer table: entry i describes layer i + 1 and what feeds it, the output of layer
   i before version 6, of its inputs (the first gives the input shape) from version 6 on. */
typedef struct {
    uint32_t inputs, outputs;
    ActivationFunction activation;
    PrecisionMode precision;
    float dropout;
    uint64_t weights, scales, biases, optimizer;    /* section offsets; scales and optimizer may be 0 */
    LayerType type;                                 /* LAYER_DENSE in version 3 */
    uint32_t in_h, in_w, in_c, out_h, out_w, out_c; /* 1 x 1 x units around dense layers */
    uint32_t kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w;
    uint32_t groups;                                /* conv: channel groups (1 before version 5) */
    float eps, momentum;                            /* batch normalization */
    uint32_t rows, row_len;                         /* the weight matrix: 0 x 0 for pooling, channels
                                                       x 1 (gamma) for batch normalization */
    uint32_t input_count, input0;                   /* the layers it reads (layer i before version 6) */
    uint64_t input_list;                            /* version 6: offset of all of them (input_count > 1) */
    uint64_t act_offset;                            /* version 6: byte offset of the output in the
                                                       workspace's activations (0 for the last layer) */
    uint32_t mode;                                  /* version 7: upsampling: UpsampleMode */
    uint32_t vocabulary;                            /* version 8: embedding: rows (= rows) */
    uint32_t heads, kv_heads;                       /* version 8: attention */
    float theta;                                    /* version 8: attention: rotary embeddings' base */
    bool causal, positions;                         /* version 8: attention, embedding */
} SlettLayer;

typedef struct {
    uint32_t version;           /* 3 to 8 */
    uint32_t layers;            /* including the input layer */
    LossFunction loss;
    uint8_t flags;
    uint64_t time_step;
    uint64_t size;              /* file size recorded in the header */
    uint32_t max_width;         /* widest hidden layer (0 without hidden layers) */
    uint32_t max_int_inputs;    /* widest input of an integer layer (0 without integer layers) */
    uint64_t conv_scratch;      /* bytes of scratch the largest convolution needs (slett_conv_scratch) */
    uint64_t activations;       /* bytes of the layers' outputs in a workspace: two of the widest
                                   before version 6, as the header records from version 6 on */
} SlettInfo;

/* Little-endian field access (format 3 is little-endian; the engine refuses big-endian hosts). */
static inline uint16_t slett_get16(const uint8_t *p) { uint16_t v; memcpy(&v, p, 2); return v; }
static inline uint32_t slett_get32(const uint8_t *p) { uint32_t v; memcpy(&v, p, 4); return v; }
static inline uint64_t slett_get64(const uint8_t *p) { uint64_t v; memcpy(&v, p, 8); return v; }
static inline void slett_put16(uint8_t *p, uint16_t v) { memcpy(p, &v, 2); }
static inline void slett_put32(uint8_t *p, uint32_t v) { memcpy(p, &v, 4); }
static inline void slett_put64(uint8_t *p, uint64_t v) { memcpy(p, &v, 8); }
/* Activations as .slett files code them, fixed since format version 1: 0 sigmoid, 1 ReLU, 2 tanh,
   3 leaky ReLU, 4 FOO52, 5 softmax, 6 none; from version 8 also 7 GELU, 8 GELU's tanh approximation,
   9 SiLU. ActivationFunction has none first, then the same order (GELU is 7 in both). slett_act()
   gives ACT_COUNT for a code that names none. */
#define SLETT_ACT_NONE 6u
static inline uint8_t slett_act_code(ActivationFunction act) {
    return act == ACT_NONE ? (uint8_t)SLETT_ACT_NONE : act >= ACT_GELU ? (uint8_t)act : (uint8_t)(act - 1);
}
static inline ActivationFunction slett_act(uint32_t code) {
    return code == SLETT_ACT_NONE ? ACT_NONE : code < SLETT_ACT_NONE ? (ActivationFunction)(code + 1u)
         : code < (uint32_t)ACT_COUNT ? (ActivationFunction)code : ACT_COUNT;
}
static inline uint64_t slett_align(uint64_t x) { return (x + SLETT_SECTION_ALIGN - 1) & ~(uint64_t)(SLETT_SECTION_ALIGN - 1); }

bool spingalett_host_is_little_endian(void);

/* Bytes of one stored weight row of `inputs` values. */
uint64_t spingalett_slett_row_bytes(PrecisionMode precision, uint32_t inputs);

/* Validates a format 3 to 6 image (no alignment requirement: fields are read with memcpy). */
int spingalett_slett_validate(const uint8_t *image, size_t size, SlettInfo *info);
/* Entry `index` (0-based) of the layer table of a validated image, with its input shape. */
void spingalett_slett_layer(const uint8_t *image, uint32_t index, SlettLayer *layer);
/* Input k of layer L (a layer index: 0 is the network's input). */
static inline uint32_t spingalett_slett_input(const uint8_t *image, const SlettLayer *L, uint32_t k) {
    return k == 0 ? L->input0 : slett_get32(image + L->input_list + 4u * k);
}
/* Byte offset of the output of layer j >= 1 among a version 6 image's activations. */
uint64_t spingalett_slett_act_offset(const uint8_t *image, uint32_t j);
/* Height, width and units of the output of layer j of a validated image (0: the input layer). */
void spingalett_slett_output_shape(const uint8_t *image, uint32_t j, uint32_t *height, uint32_t *width,
                                   uint32_t *units);

/* A convolution or pooling layer's output extent along one axis: windows of `kernel` cells,
   `stride` apart, over `size` cells padded by `pad` on both sides (0 when none fits). */
static inline uint32_t slett_window_count(uint32_t size, uint32_t kernel, uint32_t stride, uint32_t pad) {
    uint64_t padded = (uint64_t)size + 2u * pad;
    return kernel == 0 || stride == 0 || padded < kernel ? 0u : (uint32_t)((padded - kernel) / stride + 1u);
}

/* CRC-32 (IEEE, as zlib): crc = spingalett_crc32(0, data, n), or chained over several blocks. */
uint32_t spingalett_crc32(uint32_t crc, const void *data, size_t n);

/* IEEE half and bfloat16 conversions, round to nearest even. */
uint16_t spingalett_float_to_fp16(float x);
float    spingalett_fp16_to_float(uint16_t h);
uint16_t spingalett_float_to_bf16(float x);
float    spingalett_bf16_to_float(uint16_t h);

/* Unpacks n stored INT4 or INT2 codes (row layout of docs/ModelFormat.md) to int8 values. */
void spingalett_unpack_int4(const uint8_t *src, int8_t *dst, uint32_t n);
void spingalett_unpack_int2(const uint8_t *src, int8_t *dst, uint32_t n);

/* Quantizes n activations to q in -127..127 with x ~ q * scale and returns the scale: the largest
   magnitude / 127, 0 when all are 0, NaN (with q all 0) when any value is NaN or infinite. */
float spingalett_quantize_activations(const float *x, uint32_t n, int8_t *q);
int32_t spingalett_dot_i8(const int8_t *a, const int8_t *b, uint32_t n);
/* acc[r] = dot of row r (rows `stride` bytes apart) with x, for r = 0..3. */
void spingalett_dot_i8_rows4(const int8_t *w, size_t stride, const int8_t *x, uint32_t n, int32_t acc[4]);
/* acc[4 r + p] = dot of row r of w with vector p of x (rows and vectors their strides apart), for r
   and p in 0..3: each row is read once for four vectors (twice, for two each, with AVX2 alone,
   whose 16 registers hold fewer sums). wsum holds the rows' sums
   (spingalett_sum_i8) when spingalett_dot_i8_4x4_sums is true; the kernel does not read it
   otherwise, and callers may skip the sums. */
void spingalett_dot_i8_4x4(const int8_t *w, size_t w_stride, const int8_t *x, size_t x_stride, uint32_t n,
                           const int32_t wsum[4], int32_t acc[16]);
extern const bool spingalett_dot_i8_4x4_sums;
int32_t spingalett_sum_i8(const int8_t *w, uint32_t n);

/* Convolutions with windows shorter than this run filter-major: their filters are transposed into
   the workspace (columns), and each output pixel accumulates all filters at once, reading its
   window straight from the input; longer windows are gathered and dotted with each filter row. */
#define SLETT_COLUMN_WINDOW 32u

static inline bool slett_conv_columns(const SlettLayer *L) {
    return L->type == LAYER_CONV2D && L->groups == 1 && L->row_len < SLETT_COLUMN_WINDOW;
}

/* Depthwise convolutions (one input channel per group) run channel-major: their filters are
   transposed tap by tap, wt[t * rows + j], and each output pixel accumulates all filters at once. */
static inline bool slett_conv_depthwise(const SlettLayer *L) {
    return L->type == LAYER_CONV2D && L->groups > 1 && L->in_c == L->groups;
}

/* Forward passes of the layers that combine others, on one sample, in the order the batched ones
   use: an addition's inputs in their order, a concatenation's side by side, the cells of a global
   pooling one after the other (y[c] = (sum of x[p][c]) * (1 / cells)). */
void spingalett_engine_add(const float *const *x, uint32_t count, float *y, uint32_t n);
void spingalett_engine_concat(const float *const *x, const uint32_t *channels, uint32_t count, float *y,
                              uint32_t cells);
void spingalett_engine_global_pool(const float *x, float *y, uint32_t cells, uint32_t channels);
/* x (in_h x in_w x channels) upsampled by sh x sw into y: copies (UPSAMPLE_NEAREST) or bilinear with
   the cells' centres aligned and the edges repeated; the same everywhere it runs (training, batched
   models, the engine). */
void spingalett_engine_upsample(const float *x, uint32_t in_h, uint32_t in_w, uint32_t channels, uint32_t sh,
                                uint32_t sw, uint32_t mode, float *y);
/* The rows and weight of the input row (or column) that output row o reads from below, bilinearly:
   o reads rows r0 and r0 + 1 (or r0 alone at the edges), the second weighted w. */
void spingalett_engine_bilinear(uint32_t o, uint32_t factor, uint32_t in, uint32_t *r0, uint32_t *r1, float *w);
/* Layer normalization of `cells` cells of `channels` values: y = gamma (x - mean) / sqrt(var + eps)
   + beta per cell, mean and variance over its channels (float sums in channel order); stats, when
   not NULL, gets each cell's mean and 1 / sqrt(var + eps). */
void spingalett_engine_layer_norm(const float *x, uint32_t cells, uint32_t channels, const float *gamma,
                                  const float *beta, float eps, float *y, float *stats);

/* RMS normalization of `cells` cells of `channels` values: y = gamma x / sqrt(mean of x^2 + eps) per cell
   (float sums in channel order); stats, when not NULL, gets each cell's 1 / sqrt(mean of x^2 + eps). */
void spingalett_engine_rms_norm(const float *x, uint32_t cells, uint32_t channels, const float *gamma, float eps,
                                float *y, float *stats);
/* y = x[0] x[1] ... over n values, the inputs multiplied in their order. */
void spingalett_engine_multiply(const float *const *x, uint32_t count, float *y, uint32_t n);
/* The rotary position embeddings of `cells` positions and heads of `head` values: table[p * half + i] the
   cosine and table[cells * half + p * half + i] the sine of p theta^(-2 i / head), for i below half = head
   / 2, computed in double and rounded to float (alike on every platform that rounds double alike: the
   same table for every backend). A vector's halves are paired: x'[i] = x[i] cos - x[i + half] sin,
   x'[i + half] = x[i + half] cos + x[i] sin. */
void spingalett_engine_rope_table(uint32_t cells, uint32_t head, float theta, float *table);
/* The attention of one sample, as the engine computes it: x of `cells` cells of (heads + 2 kv) x head
   values (queries, keys, values), y of cells x heads x head values; each query's scores in the order of
   the keys, its softmax online (the running maximum and sum, the values' weighted sum rescaled when the
   maximum grows), causal: the keys up to its cell. With a table (the rotary embeddings) the queries and
   keys are rotated first, in scratch of slett_attention_floats() floats. */
void spingalett_engine_attention(const float *x, uint32_t cells, uint32_t heads, uint32_t kv, uint32_t head,
                                 bool causal, const float *table, float *y, float *scratch);
/* An embedding of one sample: a row of L's table (in its precision, as floats) for every input value, zeros
   for values outside it, plus the position's vector with positions. */
void spingalett_engine_embedding(const uint8_t *image, const SlettLayer *L, const float *x, float *y);
static inline uint64_t slett_attention_floats(uint32_t cells, uint32_t kv, uint32_t head, bool rope) {
    return rope ? (uint64_t)cells * kv * head + (uint64_t)cells * head + head : 0u;
}

/* GELU, its tanh approximation and SiLU of one value. */
static inline float spingalett_engine_activate_one(float x, ActivationFunction act) {
    if (act == ACT_GELU) return 0.5f * x * (1.0f + erff(x * 0.70710678118654752f));
    if (act == ACT_GELU_TANH) {
        const float u = 0.79788456080286536f * (x + 0.044715f * x * x * x);
        return 0.5f * x * (1.0f + tanhf(u));
    }
    if (act == ACT_SILU) return x / (1.0f + expf(-x));
    return x;
}

/* Bytes of engine scratch layer L needs: a convolution its transposed filters and one pixel's sums,
   or a gathered window (of its group's channels; an INT8 convolution with one group four windows
   and the sums of its filters); batch normalization its coefficients; attention with rotary
   embeddings its table and rotated keys (0 for other layers). */
static inline uint64_t slett_conv_scratch(const SlettLayer *L) {
    if (L->type == LAYER_BATCH_NORM) return (uint64_t)L->out_c * 8u;
    if (L->type == LAYER_ATTENTION)
        return 4u * slett_attention_floats(L->out_h * L->out_w, L->kv_heads, L->out_c / L->heads, L->theta > 0.0f);
    if (L->type == LAYER_CONV_TRANSPOSE2D) return (uint64_t)L->row_len * 4u;     /* a window of its group */
    if (L->type != LAYER_CONV2D) return 0;
    if (!slett_conv_columns(L) && !slett_conv_depthwise(L)) {
        uint64_t window = (uint64_t)L->row_len * 4u;
        if (L->precision == PRECISION_INT8 && L->groups <= 1) return slett_align(window) + (uint64_t)L->rows * 4u;
        return window;
    }
    uint64_t elem = spingalett_precision_is_int(L->precision) ? 1u : 4u;
    return slett_align((uint64_t)L->row_len * L->rows * elem) + (uint64_t)L->rows * 4u;
}

/* The filters of convolution L transposed, wt[k * rows + j] = weight k of filter j: the stored
   codes as bytes for integer precisions, floats otherwise. */
void spingalett_conv_transpose_filters(const uint8_t *image, const SlettLayer *L, void *wt);
/* The integer sums of output pixel (oh, ow) of depthwise convolution L with transposed filters wt,
   from its quantized input xq (one sum per filter). */
void spingalett_conv_depthwise_i8(const int8_t *xq, const SlettLayer *L, uint32_t oh, uint32_t ow, const int8_t *wt,
                                  int32_t *acc);
/* The window of output pixel (oh, ow) over the channels of group g of convolution L (as
   spingalett_gather_window, kernel_h x kernel_w x in_c / groups values). */
void spingalett_gather_group_window(const void *x, const SlettLayer *L, uint32_t oh, uint32_t ow, uint32_t g,
                                    void *window, size_t elem);
/* The integer sums of output pixel (oh, ow) of convolution L with transposed filters wt, from its
   quantized input xq (one sum per filter). */
void spingalett_conv_columns_i8(const int8_t *xq, const SlettLayer *L, uint32_t oh, uint32_t ow, const int8_t *wt,
                                int32_t *acc);

/* Copies the window of output pixel (oh, ow) of convolution L from its input x, elem bytes per value
   (floats or quantized bytes): kernel_h runs of kernel_w x channels values, zeros where the window
   leaves the input. */
void spingalett_gather_window(const void *x, const SlettLayer *L, uint32_t oh, uint32_t ow, void *window, size_t elem);
/* A transposed convolution on one sample, as the engine runs it (a window of each output pixel's
   inputs, zero where no input cell reaches it, dotted with the filters of its group); activation
   included. window: slett_conv_scratch() bytes, xq: the quantized input of integer precisions. */
void spingalett_engine_conv_transpose(const uint8_t *image, const SlettLayer *L, const float *x, float *y, int8_t *xq,
                                      void *window);

/* y[j] = bias[j] + scale[j] * x_scale * acc[j], the output of an integer layer before activation. */
static inline float spingalett_int_output(float bias, float row_scale, float x_scale, int32_t acc) {
    return bias + (row_scale * x_scale) * (float)acc;
}

/* Batch normalization at inference as y = a x + b per channel: a = gamma / sqrt(var + eps),
   b = beta - mean a (every implementation computes these alike, so their results agree). */
static inline void spingalett_bn_coefficients(const float *gamma, const float *beta, const float *mean,
                                              const float *var, float eps, uint32_t channels, float *a, float *b) {
    for (uint32_t c = 0; c < channels; c++) {
        a[c] = gamma[c] / sqrtf(var[c] + eps);
        b[c] = beta[c] - mean[c] * a[c];
    }
}

/* The layer's activation on one sample's outputs, in place. */
void spingalett_engine_activate(float *y, uint32_t n, ActivationFunction act);
