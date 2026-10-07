/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/* Internals of the inference engine (Spingalett.Inference.c) shared with the rest of the library:
   the .slett format version 3, 4 and 5 layout (docs/ModelFormat.md), CRC-32, half and bfloat16
   conversion, and the kernels batched inference reuses so that it computes exactly what
   spingalett_model_run computes. */

#pragma once

#include <Spingalett/Spingalett.Inference.h>
#include <math.h>
#include <string.h>

#define SLETT_MAGIC              "SLETTM"
#define SLETT_HEADER_SIZE        64u
#define SLETT_LAYER_ENTRY_SIZE   48u          /* version 3 */
#define SLETT_LAYER_ENTRY_SIZE_4 64u          /* version 4: shapes and windows added */
#define SLETT_LAYER_ENTRY_SIZE_5 80u          /* version 5: groups and batch normalization added */
#define SLETT_MAX_EXTENT         65535u       /* largest height, width, kernel, stride or padding */
#define SLETT_SECTION_ALIGN      16u          /* every section starts at a multiple of this */
#define SLETT_FLAG_OPTIMIZER     0x01u
#define SLETT_MAX_LAYERS         65536u
#define SLETT_MAX_INT_INPUTS     131072u      /* 127 * 127 * inputs must fit in an int32 */

static inline bool spingalett_precision_is_int(PrecisionMode p) {
    return p == PRECISION_INT8 || p == PRECISION_INT4 || p == PRECISION_INT2;
}

/* One entry of the layer table: weight layer i connects layer i-1 (inputs) to layer i (outputs). */
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
} SlettLayer;

typedef struct {
    uint32_t version;           /* 3, 4 or 5 */
    uint32_t layers;            /* including the input layer */
    LossFunction loss;
    uint8_t flags;
    uint64_t time_step;
    uint64_t size;              /* file size recorded in the header */
    uint32_t max_width;         /* widest hidden layer (0 without hidden layers) */
    uint32_t max_int_inputs;    /* widest input of an integer layer (0 without integer layers) */
    uint64_t conv_scratch;      /* bytes of scratch the largest convolution needs (slett_conv_scratch) */
} SlettInfo;

/* Little-endian field access (format 3 is little-endian; the engine refuses big-endian hosts). */
static inline uint16_t slett_get16(const uint8_t *p) { uint16_t v; memcpy(&v, p, 2); return v; }
static inline uint32_t slett_get32(const uint8_t *p) { uint32_t v; memcpy(&v, p, 4); return v; }
static inline uint64_t slett_get64(const uint8_t *p) { uint64_t v; memcpy(&v, p, 8); return v; }
static inline void slett_put16(uint8_t *p, uint16_t v) { memcpy(p, &v, 2); }
static inline void slett_put32(uint8_t *p, uint32_t v) { memcpy(p, &v, 4); }
static inline void slett_put64(uint8_t *p, uint64_t v) { memcpy(p, &v, 8); }
static inline uint64_t slett_align(uint64_t x) { return (x + SLETT_SECTION_ALIGN - 1) & ~(uint64_t)(SLETT_SECTION_ALIGN - 1); }

bool spingalett_host_is_little_endian(void);

/* Bytes of one stored weight row of `inputs` values. */
uint64_t spingalett_slett_row_bytes(PrecisionMode precision, uint32_t inputs);

/* Validates a format 3, 4 or 5 image (no alignment requirement: fields are read with memcpy). */
int spingalett_slett_validate(const uint8_t *image, size_t size, SlettInfo *info);
/* Entry `index` (0-based) of the layer table of a validated image, with its input shape. */
void spingalett_slett_layer(const uint8_t *image, uint32_t index, SlettLayer *layer);

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

/* Convolutions with windows shorter than this run filter-major: their filters are transposed into
   the workspace (columns), and each output pixel accumulates all filters at once, reading its
   window straight from the input; longer windows are gathered and dotted with each filter row. */
#define SLETT_COLUMN_WINDOW 32u

static inline bool slett_conv_columns(const SlettLayer *L) {
    return L->type == LAYER_CONV2D && L->row_len < SLETT_COLUMN_WINDOW;
}

/* Bytes of engine scratch layer L needs: a convolution its transposed filters and one pixel's sums,
   or a gathered window; batch normalization its coefficients (0 for other layers). */
static inline uint64_t slett_conv_scratch(const SlettLayer *L) {
    if (L->type == LAYER_BATCH_NORM) return (uint64_t)L->out_c * 8u;
    if (L->type != LAYER_CONV2D) return 0;
    if (!slett_conv_columns(L)) return (uint64_t)L->row_len * 4u;
    uint64_t elem = spingalett_precision_is_int(L->precision) ? 1u : 4u;
    return slett_align((uint64_t)L->row_len * L->rows * elem) + (uint64_t)L->rows * 4u;
}

/* The filters of convolution L transposed, wt[k * rows + j] = weight k of filter j: the stored
   codes as bytes for integer precisions, floats otherwise. */
void spingalett_conv_transpose_filters(const uint8_t *image, const SlettLayer *L, void *wt);
/* The integer sums of output pixel (oh, ow) of convolution L with transposed filters wt, from its
   quantized input xq (one sum per filter). */
void spingalett_conv_columns_i8(const int8_t *xq, const SlettLayer *L, uint32_t oh, uint32_t ow, const int8_t *wt,
                                int32_t *acc);

/* Copies the window of output pixel (oh, ow) of convolution L from its input x, elem bytes per value
   (floats or quantized bytes): kernel_h runs of kernel_w x channels values, zeros where the window
   leaves the input. */
void spingalett_gather_window(const void *x, const SlettLayer *L, uint32_t oh, uint32_t ow, void *window, size_t elem);

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
