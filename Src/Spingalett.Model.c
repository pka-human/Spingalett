/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/*
 * Deployment models on top of the inference engine: models that own their image (from a network,
 * a file or a copied buffer), batched and multi-threaded inference that computes exactly what
 * spingalett_model_run computes for the integer layers, evaluation, and C header export.
 */

#include "Spingalett.Private.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <math.h>

#if defined(_OPENMP)
#include <omp.h>
#endif

/* ------------------------------------------------------------------------- owning models */

/* Wraps an image the model takes ownership of (released on failure as well). */
static SpingalettModel *adopt_image(void *image, size_t size) {
    SpingalettModel *model = (SpingalettModel *)malloc(sizeof *model);
    if (!model) {
        spingalett_aligned_free(image);
        set_error(SPINGALETT_ERR_ALLOC, "model: allocation failed");
        return NULL;
    }
    if (spingalett_model_init(model, image, size) != SPINGALETT_OK) {
        spingalett_aligned_free(image);
        free(model);
        return NULL;
    }
    model->owner_ = image;
    return model;
}

SpingalettModel *spingalett_model_from_network(const NeuralNetwork *net, PrecisionMode precision) {
    size_t size = 0;
    void *image = spingalett_save_deployment(net, precision, &size);
    return image ? adopt_image(image, size) : NULL;
}

/* Takes ownership of data (an aligned buffer): format 3 is used as it is, older formats are
   converted in their own precision. */
static SpingalettModel *model_from_buffer(void *data, size_t size) {
    if (size >= 6 && memcmp(data, SLETT_MAGIC, 6) == 0)
        return adopt_image(data, size);
    PrecisionMode precision = PRECISION_FLOAT32;
    NeuralNetwork *net = spingalett_load_from_memory_ex(data, size, &precision);
    spingalett_aligned_free(data);
    if (!net) return NULL;
    SpingalettModel *model = spingalett_model_from_network(net, precision);
    free_network(net);
    return model;
}

SpingalettModel *spingalett_model_from_memory(const void *data, size_t size) {
    if (!data) {
        set_error(SPINGALETT_ERR_INVALID, "model: data is NULL");
        return NULL;
    }
    void *copy = spingalett_aligned_alloc(size);
    if (!copy) {
        set_error(SPINGALETT_ERR_ALLOC, "model: allocation failed");
        return NULL;
    }
    memcpy(copy, data, size);
    return model_from_buffer(copy, size);
}

SpingalettModel *spingalett_model_load(const char *path) {
    if (!path) {
        set_error(SPINGALETT_ERR_INVALID, "model: path is NULL");
        return NULL;
    }
    size_t size = 0;
    void *data = spingalett_read_file(path, &size);
    return data ? model_from_buffer(data, size) : NULL;
}

void spingalett_model_free(SpingalettModel *model) {
    if (!model) return;
    spingalett_aligned_free(model->owner_);
    free(model);
}

/* ------------------------------------------------------------------------- batched inference */

#define MODEL_CHUNK 1024u                        /* samples per pass through the layers, at most */

typedef struct {
    float *act[2];                              /* [chunk x max_width] each */
    int8_t *xq;                                 /* [chunk x max_int_inputs] */
    float *xs;                                  /* per-sample activation scales */
    int8_t *rows;                               /* four unpacked INT4 / INT2 rows, per thread */
    int8_t *window;                             /* a quantized convolution window or one pixel's sums, per thread */
    size_t window_stride;
    float **dequant;                            /* FP16 / BF16 layers expanded to float, per layer */
    int8_t **unpacked;                          /* integer convolution filters as bytes, per layer:
                                                   transposed for short windows, rows for INT4 / INT2 */
    int8_t **interleaved;                       /* integer weight rows interleaved for tiles of samples
                                                   or pixels (spingalett_i8_interleaved_tile), per layer */
    int32_t **sums;                             /* the sums of those rows, or of integer convolution
                                                   filters run four pixels at a time, per layer */
    int8_t *tile;                               /* a tile of activations and its sums, per thread */
    size_t tile_stride, tile_sums;              /* bytes per thread, offset of the sums */
    float *conv;                                /* gathered windows of float convolutions (OpenBLAS) */
    float *norm;                                /* batch normalization coefficients */
    SpingalettGemmScratch *gemm;
    int threads;
} PredictWorkspace;

static void predict_workspace_free(PredictWorkspace *w, uint32_t layers) {
    spingalett_aligned_free(w->act[0]);
    spingalett_aligned_free(w->act[1]);
    spingalett_aligned_free(w->xq);
    spingalett_aligned_free(w->xs);
    spingalett_aligned_free(w->rows);
    spingalett_aligned_free(w->window);
    spingalett_aligned_free(w->conv);
    spingalett_aligned_free(w->norm);
    spingalett_aligned_free(w->tile);
    for (uint32_t i = 0; i < layers; i++) {
        if (w->dequant) spingalett_aligned_free(w->dequant[i]);
        if (w->unpacked) spingalett_aligned_free(w->unpacked[i]);
        if (w->interleaved) spingalett_aligned_free(w->interleaved[i]);
        if (w->sums) spingalett_aligned_free(w->sums[i]);
    }
    free(w->dequant);
    free(w->unpacked);
    free(w->interleaved);
    free(w->sums);
    spingalett_gemm_scratch_free(w->gemm);
}

static LayerShape input_shape(const SlettLayer *L) {
    return (LayerShape){.type = LAYER_DENSE, .height = L->in_h, .width = L->in_w, .channels = L->in_c};
}

static LayerShape output_shape(const SlettLayer *L) {
    return (LayerShape){L->type, L->out_h, L->out_w, L->out_c, L->kernel_h, L->kernel_w,
                        L->stride_h, L->stride_w, L->pad_h, L->pad_w, L->groups, L->eps, L->momentum};
}

/* The INT4 or INT2 rows of layer L (stored at src) as bytes, row_len apart. */
static void unpack_rows(const uint8_t *src, const SlettLayer *L, int8_t *u) {
    size_t row = (size_t)spingalett_slett_row_bytes(L->precision, L->row_len);
    for (uint32_t j = 0; j < L->rows; j++) {
        if (L->precision == PRECISION_INT4) spingalett_unpack_int4(src + j * row, u + (size_t)j * L->row_len, L->row_len);
        else spingalett_unpack_int2(src + j * row, u + (size_t)j * L->row_len, L->row_len);
    }
}

static bool predict_workspace_create(const SpingalettModel *model, uint32_t chunk, ComputeMode mode, PredictWorkspace *w) {
    memset(w, 0, sizeof *w);
    const uint8_t *image = (const uint8_t *)model->image;
    w->threads = 1;
#if defined(_OPENMP)
    if (mode == COMPUTE_OPENMP) w->threads = omp_get_max_threads();
#endif
    uint32_t int_inputs = 0, int_window = 0, dense_int_inputs = 0;
    size_t conv_floats = 0, tile_x = 0, tile_acc = 0;
    bool packed = false, has_float = false;
    w->dequant = (float **)calloc(model->layer_count, sizeof(float *));
    w->unpacked = (int8_t **)calloc(model->layer_count, sizeof(int8_t *));
    w->interleaved = (int8_t **)calloc(model->layer_count, sizeof(int8_t *));
    w->sums = (int32_t **)calloc(model->layer_count, sizeof(int32_t *));
    if (!w->dequant || !w->unpacked || !w->interleaved || !w->sums) return false;
    uint32_t norm_channels = 0;
    const bool interleave = spingalett_i8_interleaved();
    for (uint32_t i = 0; i < model->layer_count; i++) {
        SlettLayer L;
        spingalett_slett_layer(image, i, &L);
        if (L.type == LAYER_BATCH_NORM && L.out_c > norm_channels) norm_channels = L.out_c;
        if (L.rows == 0 || L.type == LAYER_BATCH_NORM) continue;      /* pooling, normalization */
        bool conv = L.type == LAYER_CONV2D, is_int = spingalett_precision_is_int(L.precision);
        /* where the target has the kernel, integer convolutions with one group run in tiles of
           pixels, and dense layers in tiles of samples (given that many), against interleaved rows */
        bool tiled = is_int && interleave && (conv ? L.groups <= 1 : chunk >= SPINGALETT_I8_TILE);
        bool columns = !tiled && (slett_conv_columns(&L) || slett_conv_depthwise(&L));     /* transposed filters */
        if (is_int) {
            if (L.inputs > int_inputs) int_inputs = L.inputs;
            if (tiled) {
                size_t x = (size_t)SPINGALETT_I8_TILE * spingalett_i8_interleaved_len(L.row_len);
                size_t acc = (size_t)SPINGALETT_I8_TILE * spingalett_i8_interleaved_rows(L.rows) * sizeof(int32_t);
                if (x > tile_x) tile_x = x;
                if (acc > tile_acc) tile_acc = acc;
            } else if (conv) {
                /* one pixel's sums, a group's window, or the windows of four pixels */
                uint32_t per_thread = columns ? L.rows * 4u : L.groups > 1 ? L.row_len : 4u * L.row_len;
                if (per_thread > int_window) int_window = per_thread;
            }
            /* dense layers run as before on fewer samples than a tile (the last chunk) */
            if (!conv && L.inputs > dense_int_inputs) dense_int_inputs = L.inputs;
            packed |= !conv && L.precision != PRECISION_INT8;
        } else {
            has_float = true;
            if (conv) {
                LayerShape in = input_shape(&L), out = output_shape(&L);
                size_t need = spingalett_conv_forward_scratch(&in, &out, chunk, mode);
                if (need > conv_floats) conv_floats = need;
            }
        }
        size_t n = (size_t)L.rows * L.row_len;
        const uint8_t *src = image + L.weights;
        if (L.precision == PRECISION_FP16 || L.precision == PRECISION_BFLOAT16) {
            float *d = (float *)spingalett_aligned_alloc(n * sizeof(float));
            if (!d) return false;
            for (size_t k = 0; k < n; k++) {
                uint16_t h = slett_get16(src + 2u * k);
                d[k] = L.precision == PRECISION_FP16 ? spingalett_fp16_to_float(h) : spingalett_bf16_to_float(h);
            }
            w->dequant[i] = d;
        }
        if (tiled) {
            const int8_t *rows = (const int8_t *)src;
            size_t stride = (size_t)spingalett_slett_row_bytes(L.precision, L.row_len);
            int8_t *u = NULL;
            if (L.precision != PRECISION_INT8) {
                u = (int8_t *)spingalett_aligned_alloc(n);
                if (!u) return false;
                unpack_rows(src, &L, u);
                rows = u;
                stride = L.row_len;
            }
            const uint32_t R = spingalett_i8_interleaved_rows(L.rows);
            w->interleaved[i] = (int8_t *)spingalett_aligned_alloc((size_t)R * spingalett_i8_interleaved_len(L.row_len));
            w->sums[i] = (int32_t *)spingalett_aligned_alloc((size_t)R * sizeof(int32_t));
            if (w->interleaved[i] && w->sums[i])
                spingalett_i8_interleave(rows, stride, L.rows, L.row_len, w->interleaved[i], w->sums[i]);
            spingalett_aligned_free(u);
            if (!w->interleaved[i] || !w->sums[i]) return false;
            continue;
        }
        if (conv && is_int && (columns || L.precision != PRECISION_INT8)) {
            int8_t *u = (int8_t *)spingalett_aligned_alloc(n);
            if (!u) return false;
            if (columns)                        /* filter-major, as the engine runs short windows and
                                                   depthwise convolutions */
                spingalett_conv_transpose_filters(image, &L, u);
            else
                unpack_rows(src, &L, u);
            w->unpacked[i] = u;
        }
        if (conv && is_int && !columns && L.groups <= 1 && spingalett_dot_i8_4x4_sums) {
            const int8_t *rows = w->unpacked[i] ? w->unpacked[i] : (const int8_t *)src;
            int32_t *sums = (int32_t *)spingalett_aligned_alloc((size_t)L.rows * sizeof(int32_t));
            if (!sums) return false;
            for (uint32_t j = 0; j < L.rows; j++) sums[j] = spingalett_sum_i8(rows + (size_t)j * L.row_len, L.row_len);
            w->sums[i] = sums;
        }
    }
    size_t width = (size_t)chunk * (model->max_width_ ? model->max_width_ : 1u);
    w->act[0] = (float *)spingalett_aligned_alloc(width * sizeof(float));
    w->act[1] = (float *)spingalett_aligned_alloc(width * sizeof(float));
    if (!w->act[0] || !w->act[1]) return false;
    if (int_inputs) {
        w->xq = (int8_t *)spingalett_aligned_alloc((size_t)chunk * int_inputs);
        w->xs = (float *)spingalett_aligned_alloc((size_t)chunk * sizeof(float));
        if (!w->xq || !w->xs) return false;
        if (packed) {
            w->rows = (int8_t *)spingalett_aligned_alloc((size_t)w->threads * 4u * dense_int_inputs);
            if (!w->rows) return false;
        }
        if (int_window) {
            w->window_stride = ((size_t)int_window + 63u) & ~(size_t)63u;    /* threads on their own cache lines */
            w->window = (int8_t *)spingalett_aligned_alloc((size_t)w->threads * w->window_stride);
            if (!w->window) return false;
        }
        if (tile_x) {
            w->tile_sums = (tile_x + 63u) & ~(size_t)63u;
            w->tile_stride = (w->tile_sums + tile_acc + 63u) & ~(size_t)63u;
            w->tile = (int8_t *)spingalett_aligned_alloc((size_t)w->threads * w->tile_stride);
            if (!w->tile) return false;
            /* the vectors of a tile past the last pixel or sample keep the values they have: they
               must have some */
            memset(w->tile, 0, (size_t)w->threads * w->tile_stride);
        }
    }
    if (conv_floats) {
        w->conv = (float *)spingalett_aligned_alloc(conv_floats * sizeof(float));
        if (!w->conv) return false;
    }
    if (norm_channels) {
        w->norm = (float *)spingalett_aligned_alloc(2u * (size_t)norm_channels * sizeof(float));
        if (!w->norm) return false;
    }
    if (has_float && mode != COMPUTE_OPENBLAS) {
        w->gemm = spingalett_gemm_scratch_create(w->threads);
        if (!w->gemm) return false;
    }
    return true;
}

static void activate_samples(float *y, uint32_t n, uint32_t size, ActivationFunction act, ComputeMode mode) {
    if (act == ACT_NONE) return;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if(spingalett_use_omp(mode, (uint64_t)n * size))
#endif
    for (int64_t s = 0; s < (int64_t)n; s++)
        spingalett_engine_activate(y + (size_t)s * size, size, act);
    (void)mode;
}

/* An integer convolution over n samples, as spingalett_model_run computes it: each sample's input
   quantized as a whole, each output pixel the integer dot products of the filters with its
   quantized window (filters: rows of bytes, or transposed for short windows). */
/* The outputs of n integer rows for SPINGALETT_I8_TILE vectors (count of them real) against their
   interleaved rows: vector i of the tile (in w->tile of thread t, its bytes in place) goes to ys[i]
   with activation scale xs[i]. */
static void tile_outputs(const int8_t *interleaved, const int32_t *sums, const float *bias, const float *scale,
                         uint32_t rows, uint32_t n, PredictWorkspace *w, int t, uint32_t count, float *const ys[],
                         const float xs[]) {
    int8_t *tile = w->tile + (size_t)t * w->tile_stride;
    int32_t *acc = (int32_t *)(void *)(tile + w->tile_sums);
    const uint32_t R = spingalett_i8_interleaved_rows(rows);
    spingalett_i8_interleaved_tile(interleaved, sums, rows, n, tile, spingalett_i8_interleaved_len(n), acc, R);
    for (uint32_t i = 0; i < count; i++) {
        const int32_t *a = acc + (size_t)i * R;
        float *y = ys[i], x_scale = xs[i];
        for (uint32_t j = 0; j < rows; j++) y[j] = spingalett_int_output(bias[j], scale[j], x_scale, a[j]);
    }
}

static void predict_int_conv(const uint8_t *image, const SlettLayer *L, const int8_t *filters,
                             const int8_t *interleaved, const int32_t *sums, const float *x, float *y, uint32_t n,
                             PredictWorkspace *w, ComputeMode mode) {
    const uint32_t in = L->inputs, out = L->outputs, OC = L->rows, K = L->row_len, C = L->in_c;
    const uint64_t pixels = (uint64_t)L->out_h * L->out_w, items = (uint64_t)n * pixels;
    const float *bias = (const float *)(const void *)(image + L->biases);
    const float *scale = (const float *)(const void *)(image + L->scales);
    const bool pointwise = L->kernel_h == 1 && L->kernel_w == 1 && L->stride_h == 1 && L->stride_w == 1 &&
                           L->pad_h == 0 && L->pad_w == 0;
    bool parallel = spingalett_use_omp(mode, items * K * OC);

#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if(parallel)
#endif
    for (int64_t s = 0; s < (int64_t)n; s++)
        w->xs[s] = spingalett_quantize_activations(x + (size_t)s * in, in, w->xq + (size_t)s * in);

    if (interleaved) {
        /* a tile of pixels at a time against all filters */
        const uint32_t len = spingalett_i8_interleaved_len(K);
        const int64_t tiles = (int64_t)((items + SPINGALETT_I8_TILE - 1u) / SPINGALETT_I8_TILE);
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if(parallel)
#endif
        for (int64_t tile = 0; tile < tiles; tile++) {
            int t = 0;
#if defined(_OPENMP)
            t = omp_get_thread_num();
#endif
            const uint64_t g0 = (uint64_t)tile * SPINGALETT_I8_TILE;
            const uint32_t count = items - g0 < SPINGALETT_I8_TILE ? (uint32_t)(items - g0) : SPINGALETT_I8_TILE;
            int8_t *windows = w->tile + (size_t)t * w->tile_stride;
            float *ys[SPINGALETT_I8_TILE], xs[SPINGALETT_I8_TILE];
            for (uint32_t i = 0; i < count; i++) {
                uint64_t g = g0 + i, s = g / pixels, p = g % pixels;
                const int8_t *xq = w->xq + (size_t)s * in;
                if (pointwise) memcpy(windows + (size_t)i * len, xq + p * C, K);
                else spingalett_gather_window(xq, L, (uint32_t)(p / L->out_w), (uint32_t)(p % L->out_w), windows + (size_t)i * len, 1);
                ys[i] = y + (size_t)s * out + p * OC;
                xs[i] = w->xs[s];
            }
            tile_outputs(interleaved, sums, bias, scale, OC, K, w, t, count, ys, xs);
        }
        activate_samples(y, n, out, L->activation, mode);
        return;
    }

    /* short windows, depthwise and grouped convolutions pixel by pixel; the others four pixels at
       a time, so that each filter is read once for four windows */
    const bool quads = !slett_conv_columns(L) && !slett_conv_depthwise(L) && L->groups <= 1;
    const int64_t units = quads ? (int64_t)((items + 3u) / 4u) : (int64_t)items;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if(parallel)
#endif
    for (int64_t unit = 0; unit < units; unit++) {
        int t = 0;
#if defined(_OPENMP)
        t = omp_get_thread_num();
#endif
        if (quads) {
            uint64_t g0 = (uint64_t)unit * 4u, count = items - g0 < 4u ? items - g0 : 4u;
            int8_t *windows = w->window + (size_t)t * w->window_stride;
            uint64_t smp[4];
            float *ys[4];
            for (uint64_t i = 0; i < count; i++) {
                uint64_t g = g0 + i, s = g / pixels, p = g % pixels;
                const int8_t *xq = w->xq + (size_t)s * in;
                if (pointwise) memcpy(windows + i * K, xq + p * C, K);
                else spingalett_gather_window(xq, L, (uint32_t)(p / L->out_w), (uint32_t)(p % L->out_w), windows + i * K, 1);
                smp[i] = s;
                ys[i] = y + (size_t)s * out + p * OC;
            }
            for (uint32_t j = 0; j < OC; j += 4) {
                uint32_t rows = OC - j < 4u ? OC - j : 4u;
                const int8_t *block = filters + (size_t)j * K;
                if (rows == 4 && count == 4) {
                    int32_t acc[16];
                    spingalett_dot_i8_4x4(block, K, windows, K, K, sums ? sums + j : NULL, acc);
                    for (uint32_t r = 0; r < 4; r++)
                        for (uint32_t i = 0; i < 4; i++)
                            ys[i][j + r] = spingalett_int_output(bias[j + r], scale[j + r], w->xs[smp[i]], acc[4 * r + i]);
                    continue;
                }
                for (uint64_t i = 0; i < count; i++) {
                    int32_t acc[4];
                    if (rows == 4) spingalett_dot_i8_rows4(block, K, windows + i * K, K, acc);
                    else for (uint32_t r = 0; r < rows; r++) acc[r] = spingalett_dot_i8(block + (size_t)r * K, windows + i * K, K);
                    for (uint32_t r = 0; r < rows; r++)
                        ys[i][j + r] = spingalett_int_output(bias[j + r], scale[j + r], w->xs[smp[i]], acc[r]);
                }
            }
            continue;
        }
        uint64_t item = (uint64_t)unit;
        uint64_t s = item / pixels, p = item % pixels;
        const int8_t *xq = w->xq + (size_t)s * in;
        if (slett_conv_columns(L) || slett_conv_depthwise(L)) {
            int32_t *acc = (int32_t *)(void *)(w->window + (size_t)t * w->window_stride);
            float *yo = y + (size_t)s * out + p * OC;
            if (slett_conv_columns(L))
                spingalett_conv_columns_i8(xq, L, (uint32_t)(p / L->out_w), (uint32_t)(p % L->out_w), filters, acc);
            else
                spingalett_conv_depthwise_i8(xq, L, (uint32_t)(p / L->out_w), (uint32_t)(p % L->out_w), filters, acc);
            for (uint32_t j = 0; j < OC; j++) yo[j] = spingalett_int_output(bias[j], scale[j], w->xs[s], acc[j]);
            continue;
        }
        /* grouped: each group's filters with the window of its channels */
        const uint32_t OG = OC / L->groups;
        int8_t *window = w->window + (size_t)t * w->window_stride;
        float *yo = y + (size_t)s * out + p * OC;
        for (uint32_t g = 0; g < L->groups; g++) {
            spingalett_gather_group_window(xq, L, (uint32_t)(p / L->out_w), (uint32_t)(p % L->out_w), g, window, 1);
            for (uint32_t j = g * OG, e = j + OG; j < e; j += 4) {
                uint32_t rows = e - j < 4u ? e - j : 4u;
                const int8_t *block = filters + (size_t)j * K;
                int32_t acc[4];
                if (rows == 4) spingalett_dot_i8_rows4(block, K, window, K, acc);
                else for (uint32_t r = 0; r < rows; r++) acc[r] = spingalett_dot_i8(block + (size_t)r * K, window, K);
                for (uint32_t r = 0; r < rows; r++) yo[j + r] = spingalett_int_output(bias[j + r], scale[j + r], w->xs[s], acc[r]);
            }
        }
    }
    (void)parallel;
    activate_samples(y, n, out, L->activation, mode);
}

/* One layer over n samples: x [n x inputs] -> y [n x outputs], activation included. */
static void predict_layer(const uint8_t *image, const SlettLayer *L, const float *dequant, const int8_t *unpacked,
                          const int8_t *interleaved, const int32_t *sums, const float *x, float *y, uint32_t n,
                          PredictWorkspace *w, ComputeMode mode) {
    uint32_t in = L->inputs, out = L->outputs;
    const float *bias = (const float *)(const void *)(image + L->biases);

    if (L->type == LAYER_BATCH_NORM) {
        /* as the engine computes it: y = x a + b per channel, then the activation */
        const uint32_t C = L->out_c;
        const float *stats = (const float *)(const void *)(image + L->scales);
        float *a = w->norm, *b = w->norm + C;
        spingalett_bn_coefficients((const float *)(const void *)(image + L->weights), bias, stats, stats + C, L->eps, C,
                                   a, b);
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if(spingalett_use_omp(mode, (uint64_t)n * out))
#endif
        for (int64_t s = 0; s < (int64_t)n; s++) {
            const float *xs = x + (size_t)s * out;
            float *ys = y + (size_t)s * out;
            for (uint32_t i = 0; i < out; i += C)
                for (uint32_t c = 0; c < C; c++) ys[i + c] = xs[i + c] * a[c] + b[c];
            spingalett_engine_activate(ys, out, L->activation);
        }
        return;
    }
    if (L->type != LAYER_DENSE) {
        LayerShape is = input_shape(L), os = output_shape(L);
        if (L->type != LAYER_CONV2D) {
            spingalett_pool_forward_shapes(&is, &os, x, y, n, mode);
            activate_samples(y, n, out, L->activation, mode);
        } else if (spingalett_precision_is_int(L->precision)) {
            predict_int_conv(image, L, unpacked ? unpacked : (const int8_t *)(image + L->weights), interleaved, sums, x, y,
                             n, w, mode);
        } else {
            const float *W = dequant ? dequant : (const float *)(const void *)(image + L->weights);
            spingalett_conv_forward_shapes(&is, &os, W, bias, x, y, n, L->activation, w->conv, w->gemm, mode);
            if (L->activation == ACT_SOFTMAX) activate_samples(y, n, out, ACT_SOFTMAX, mode);
        }
        return;
    }

    if (!spingalett_precision_is_int(L->precision)) {
        const float *W = dequant ? dequant : (const float *)(const void *)(image + L->weights);
        spingalett_gemm(w->gemm, mode, false, true, n, out, in, 1.0f, x, in, W, in, 0.0f, y, out);
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if(spingalett_use_omp(mode, (uint64_t)n * out))
#endif
        for (int64_t s = 0; s < (int64_t)n; s++) {
            float *row = y + (size_t)s * out;
            spingalett_vec_axpy(row, bias, out, 1.0f);
            spingalett_engine_activate(row, out, L->activation);
        }
        return;
    }

    const float *scale = (const float *)(const void *)(image + L->scales);
    const uint8_t *weights = image + L->weights;
    size_t row_bytes = (size_t)spingalett_slett_row_bytes(L->precision, in);
    bool parallel = spingalett_use_omp(mode, (uint64_t)n * in * out);

#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if(parallel)
#endif
    for (int64_t s = 0; s < (int64_t)n; s++)
        w->xs[s] = spingalett_quantize_activations(x + (size_t)s * in, in, w->xq + (size_t)s * in);

    if (interleaved && n >= SPINGALETT_I8_TILE) {
        /* a tile of samples at a time against all rows */
        const uint32_t len = spingalett_i8_interleaved_len(in);
        const int64_t tiles = ((int64_t)n + SPINGALETT_I8_TILE - 1) / SPINGALETT_I8_TILE;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if(parallel)
#endif
        for (int64_t tile = 0; tile < tiles; tile++) {
            int t = 0;
#if defined(_OPENMP)
            t = omp_get_thread_num();
#endif
            const uint32_t s0 = (uint32_t)tile * SPINGALETT_I8_TILE;
            const uint32_t count = n - s0 < SPINGALETT_I8_TILE ? n - s0 : SPINGALETT_I8_TILE;
            int8_t *xt = w->tile + (size_t)t * w->tile_stride;
            float *ys[SPINGALETT_I8_TILE];
            for (uint32_t i = 0; i < count; i++) {
                memcpy(xt + (size_t)i * len, w->xq + (size_t)(s0 + i) * in, in);
                ys[i] = y + (size_t)(s0 + i) * out;
            }
            tile_outputs(interleaved, sums, bias, scale, out, in, w, t, count, ys, w->xs + s0);
        }
    } else {
        /* blocks of four weight rows against every sample; INT4 and INT2 rows are unpacked once per
           block */
        int64_t blocks = ((int64_t)out + 3) / 4;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if(parallel)
#endif
        for (int64_t b = 0; b < blocks; b++) {
            uint32_t j = (uint32_t)b * 4u, rows = out - j < 4u ? out - j : 4u;
            const int8_t *block = (const int8_t *)(weights + (size_t)j * row_bytes);
            size_t stride = row_bytes;
            if (L->precision != PRECISION_INT8) {
                int t = 0;
#if defined(_OPENMP)
                t = omp_get_thread_num();
#endif
                int8_t *unpacked = w->rows + (size_t)t * 4u * in;
                for (uint32_t r = 0; r < rows; r++) {
                    const uint8_t *src = weights + (size_t)(j + r) * row_bytes;
                    if (L->precision == PRECISION_INT4) spingalett_unpack_int4(src, unpacked + (size_t)r * in, in);
                    else spingalett_unpack_int2(src, unpacked + (size_t)r * in, in);
                }
                block = unpacked;
                stride = in;
            }
            uint32_t s = 0;
            int32_t wsum[4] = {0, 0, 0, 0};
            if (rows == 4 && n >= 4u && spingalett_dot_i8_4x4_sums)
                for (uint32_t r = 0; r < 4; r++) wsum[r] = spingalett_sum_i8(block + (size_t)r * stride, in);
            if (rows == 4)                      /* four samples at a time: each row read once for four */
                for (; s + 4u <= n; s += 4u) {
                    int32_t acc[16];
                    spingalett_dot_i8_4x4(block, stride, w->xq + (size_t)s * in, in, in, wsum, acc);
                    for (uint32_t i = 0; i < 4; i++) {
                        float *ys = y + (size_t)(s + i) * out + j;
                        for (uint32_t r = 0; r < 4; r++)
                            ys[r] = spingalett_int_output(bias[j + r], scale[j + r], w->xs[s + i], acc[4 * r + i]);
                    }
                }
            for (; s < n; s++) {
                const int8_t *xq = w->xq + (size_t)s * in;
                float *ys = y + (size_t)s * out + j;
                int32_t acc[4];
                if (rows == 4) spingalett_dot_i8_rows4(block, stride, xq, in, acc);
                else for (uint32_t r = 0; r < rows; r++) acc[r] = spingalett_dot_i8(block + (size_t)r * stride, xq, in);
                for (uint32_t r = 0; r < rows; r++)
                    ys[r] = spingalett_int_output(bias[j + r], scale[j + r], w->xs[s], acc[r]);
            }
        }
    }
    (void)parallel;

#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if(spingalett_use_omp(mode, (uint64_t)n * out))
#endif
    for (int64_t s = 0; s < (int64_t)n; s++)
        spingalett_engine_activate(y + (size_t)s * out, out, L->activation);
}

bool spingalett_model_predict(const SpingalettModel *model, const float *inputs, uint32_t count, float *outputs) {
    if (!model || !model->image || !inputs || !outputs) {
        set_error(SPINGALETT_ERR_INVALID, "model predict: model, inputs or outputs is NULL");
        return false;
    }
    if (count == 0) return true;

    ComputeMode mode = resolve_compute_mode();
    /* fewer samples per pass when the layers are wide (convolutions) */
    uint32_t width = model->max_width_ > model->input_size ? model->max_width_ : model->input_size;
    uint32_t cap = SPINGALETT_BATCH_FLOATS / 4u / (width ? width : 1u);
    if (cap > MODEL_CHUNK) cap = MODEL_CHUNK;
    if (cap < 1) cap = 1;
    uint32_t chunk = count < cap ? count : cap;
    PredictWorkspace w;
    if (!predict_workspace_create(model, chunk, mode, &w)) {
        predict_workspace_free(&w, model->layer_count);
        set_error(SPINGALETT_ERR_ALLOC, "model predict: workspace allocation failed");
        return false;
    }

    const uint8_t *image = (const uint8_t *)model->image;
    for (uint32_t start = 0; start < count; start += chunk) {
        uint32_t n = count - start < chunk ? count - start : chunk;
        const float *x = inputs + (size_t)start * model->input_size;
        for (uint32_t i = 0; i < model->layer_count; i++) {
            SlettLayer L;
            spingalett_slett_layer(image, i, &L);
            float *y = i + 1 == model->layer_count ? outputs + (size_t)start * model->output_size : w.act[i & 1u];
            predict_layer(image, &L, w.dequant[i], w.unpacked[i], w.interleaved[i], w.sums[i], x, y, n, &w, mode);
            x = y;
        }
    }
    predict_workspace_free(&w, model->layer_count);
    return true;
}

EvalMetrics spingalett_model_evaluate(const SpingalettModel *model, const float *inputs, const float *targets, uint32_t count) {
    EvalMetrics m = {NAN, NAN};
    if (!model || !model->image || !inputs || !targets || count == 0) {
        set_error(SPINGALETT_ERR_INVALID, "model evaluate: model, inputs or targets is NULL, or count is 0");
        return m;
    }
    uint32_t chunk = count < SPINGALETT_BATCH_CHUNK ? count : SPINGALETT_BATCH_CHUNK;
    uint32_t out = model->output_size;
    float *buf = (float *)spingalett_aligned_alloc((size_t)chunk * out * sizeof(float));
    if (!buf) {
        set_error(SPINGALETT_ERR_ALLOC, "model evaluate: allocation failed");
        return m;
    }
    SlettLayer last;
    spingalett_slett_layer((const uint8_t *)model->image, model->layer_count - 1, &last);
    double loss = 0.0;
    uint32_t correct = 0;
    for (uint32_t start = 0; start < count; start += chunk) {
        uint32_t n = count - start < chunk ? count - start : chunk;
        if (!spingalett_model_predict(model, inputs + (size_t)start * model->input_size, n, buf)) {
            spingalett_aligned_free(buf);
            return m;
        }
        for (uint32_t s = 0; s < n; s++) {
            const float *o = buf + (size_t)s * out, *t = targets + ((size_t)start + s) * out;
            loss += compute_sample_loss(o, t, out, model->loss, last.activation);
            correct += spingalett_sample_correct(o, t, out);
        }
    }
    spingalett_aligned_free(buf);
    m.loss = (float)(loss / count);
    m.accuracy = (float)correct / (float)count;
    return m;
}

/* ------------------------------------------------------------------------- C header export */

static bool is_identifier(const char *s) {
    if (!s || !(isalpha((unsigned char)*s) || *s == '_')) return false;
    for (; *s; s++)
        if (!(isalnum((unsigned char)*s) || *s == '_')) return false;
    return true;
}

bool spingalett_export_c_header(const NeuralNetwork *net, const char *path, const char *name, PrecisionMode precision) {
    if (!path || !is_identifier(name) || strlen(name) > 200) {
        set_error(SPINGALETT_ERR_INVALID, "export: path is NULL or name is not a C identifier");
        return false;
    }
    SpingalettModel *model = spingalett_model_from_network(net, precision);
    if (!model) return false;

    char upper[208];
    size_t len = strlen(name);
    for (size_t i = 0; i <= len; i++) upper[i] = (char)toupper((unsigned char)name[i]);

    FILE *fp = fopen(path, "w");
    if (!fp) {
        spingalett_model_free(model);
        set_error(SPINGALETT_ERR_FILE_IO, "export: cannot open file for writing");
        return false;
    }
    char topology[512];
    size_t used = (size_t)snprintf(topology, sizeof topology, "%u", net->topology[0]);
    for (uint32_t l = 1; l < net->layers && used < sizeof topology - 16; l++)
        used += (size_t)snprintf(topology + used, sizeof topology - used, "-%u", net->topology[l]);
    if (net->layers > 1 && used >= sizeof topology - 16) snprintf(topology + used, sizeof topology - used, "-...");

    const uint8_t *img = (const uint8_t *)model->image;
    fprintf(fp,
        "/*\n"
        " * Generated by Spingalett %s: a %s network, %s weights, %zu bytes.\n"
        " * The array is a .slett model image (format version %u). Run it with the inference engine:\n"
        " *\n"
        " *   static SpingalettModel model;\n"
        " *   static float workspace[%s_WORKSPACE / sizeof(float)];\n"
        " *   spingalett_model_init(&model, %s, %s_SIZE);\n"
        " *   spingalett_model_run(&model, input, output, workspace);\n"
        " *\n"
        " * Include this header in one source file: the array has internal linkage.\n"
        " */\n"
        "#ifndef %s_H_INCLUDED\n"
        "#define %s_H_INCLUDED\n\n"
        "#include <stdint.h>\n\n"
        "#define %s_SIZE %zuu\n"
        "#define %s_INPUTS %uu\n"
        "#define %s_OUTPUTS %uu\n"
        "#define %s_WORKSPACE %zuu\n\n"
        "#ifndef SPINGALETT_ALIGN16\n"
        "#  if defined(__cplusplus) && __cplusplus >= 201103L\n"
        "#    define SPINGALETT_ALIGN16 alignas(16)\n"
        "#  elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L\n"
        "#    define SPINGALETT_ALIGN16 _Alignas(16)\n"
        "#  elif defined(__GNUC__) || defined(__clang__)\n"
        "#    define SPINGALETT_ALIGN16 __attribute__((aligned(16)))\n"
        "#  elif defined(_MSC_VER)\n"
        "#    define SPINGALETT_ALIGN16 __declspec(align(16))\n"
        "#  else\n"
        "#    define SPINGALETT_ALIGN16\n"
        "#  endif\n"
        "#endif\n\n"
        "SPINGALETT_ALIGN16 static const uint8_t %s[%s_SIZE] = {\n",
        spingalett_version(), topology, precision_names[precision], model->image_size, (unsigned)slett_get16(img + 6),
        upper, name, upper, upper, upper,
        upper, model->image_size, upper, model->input_size, upper, model->output_size, upper, model->workspace_size,
        name, upper);
    for (size_t i = 0; i < model->image_size; i += 16) {
        fputs("   ", fp);
        size_t end = model->image_size - i < 16 ? model->image_size : i + 16;
        for (size_t k = i; k < end; k++) fprintf(fp, " 0x%02x,", img[k]);
        fputc('\n', fp);
    }
    fprintf(fp, "};\n\n#endif\n");
    bool ok = !ferror(fp);
    if (fclose(fp) != 0) ok = false;
    spingalett_model_free(model);
    if (!ok) set_error(SPINGALETT_ERR_FILE_IO, "export: write error");
    return ok;
}
