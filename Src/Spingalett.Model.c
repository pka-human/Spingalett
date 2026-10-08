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
#include "Spingalett.Thread.h"
#include <stdatomic.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <math.h>

#if defined(_OPENMP)
#include <omp.h>
#endif

/* ------------------------------------------------------------------------- owning models */

/* A layer's weights in the forms batched prediction runs on, built from the image. */
typedef struct {
    float *dequant;                             /* FP16 / BF16 weights as float */
    int8_t *unpacked;                           /* integer convolution filters transposed (short windows,
                                                   depthwise convolutions), or INT4 / INT2 rows as bytes */
    int32_t *sums;                              /* sums of the integer rows, for the 4 x 4 kernels */
    int8_t *interleaved;                        /* rows interleaved for tiles of samples or pixels
                                                   (spingalett_i8_interleaved_tile) */
    int32_t *tile_sums;                         /* the sums of the interleaved rows */
    bool ready;                                 /* all but a dense layer's interleaved rows are built */
} PreparedLayer;

typedef struct PredictWorkspace PredictWorkspace;

/* What a model made by the library owns: its image, its weights prepared for batched prediction
   (built by the first call that needs them, under the lock, and only read afterwards), and the
   workspace of an earlier call, which the next call takes if it fits. Models filled in by
   spingalett_model_init own nothing, and prepare their weights on every call. */
typedef struct {
    void *image;
    SpgSignal *lock;
    PreparedLayer *layers;
    _Atomic(PredictWorkspace *) spare;
} ModelOwner;

static void owner_free(ModelOwner *owner, uint32_t layers);

/* Wraps an image the model takes ownership of (released on failure as well). */
static SpingalettModel *adopt_image(void *image, size_t size) {
    SpingalettModel *model = (SpingalettModel *)malloc(sizeof *model);
    ModelOwner *owner = (ModelOwner *)calloc(1, sizeof *owner);
    SpgSignal *lock = spg_signal_create();
    if (!model || !owner || !lock) {
        spingalett_aligned_free(image);
        free(model);
        free(owner);
        spg_signal_free(lock);
        set_error(SPINGALETT_ERR_ALLOC, "model: allocation failed");
        return NULL;
    }
    if (spingalett_model_init(model, image, size) != SPINGALETT_OK) {
        spingalett_aligned_free(image);
        free(model);
        free(owner);
        spg_signal_free(lock);
        return NULL;
    }
    owner->image = image;
    owner->lock = lock;
    atomic_init(&owner->spare, NULL);
    model->owner_ = owner;
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
    owner_free((ModelOwner *)model->owner_, model->layer_count);
    free(model);
}

/* ------------------------------------------------------------------------- batched inference */

#define MODEL_CHUNK 1024u                        /* samples per pass through the layers, at most */

struct PredictWorkspace {
    float *act[2];                              /* [chunk x max_width] each; for a graph (format 6) act[0]
                                                   holds every output at chunk times its offset */
    int8_t *xq;                                 /* [chunk x max_int_inputs] */
    float *xs;                                  /* per-sample activation scales */
    int8_t *window;                             /* a quantized convolution window or one pixel's sums, per thread */
    size_t window_stride;
    int8_t *tile;                               /* a tile of activations and its sums, per thread */
    size_t tile_stride, tile_sums;              /* bytes per thread, offset of the sums */
    float *conv;                                /* gathered windows of float convolutions (OpenBLAS) */
    float *norm;                                /* batch normalization coefficients */
    SpingalettGemmScratch *gemm;
    /* what the workspace was made for, and its size */
    int threads;
    uint32_t chunk;
    ComputeMode mode;
    bool dense_tiles;
    size_t bytes;
};

/* A workspace that is no larger than this stays with its model for the next call. */
#define SPARE_WORKSPACE_MAX ((size_t)32 << 20)

static void predict_workspace_free(PredictWorkspace *w) {
    if (!w) return;
    spingalett_aligned_free(w->act[0]);
    spingalett_aligned_free(w->act[1]);
    spingalett_aligned_free(w->xq);
    spingalett_aligned_free(w->xs);
    spingalett_aligned_free(w->window);
    spingalett_aligned_free(w->conv);
    spingalett_aligned_free(w->norm);
    spingalett_aligned_free(w->tile);
    spingalett_gemm_scratch_free(w->gemm);
    free(w);
}

static void prepared_free(PreparedLayer *layers, uint32_t count) {
    if (!layers) return;
    for (uint32_t i = 0; i < count; i++) {
        PreparedLayer *p = &layers[i];
        spingalett_aligned_free(p->dequant);
        spingalett_aligned_free(p->unpacked);
        spingalett_aligned_free(p->sums);
        spingalett_aligned_free(p->interleaved);
        spingalett_aligned_free(p->tile_sums);
    }
    free(layers);
}

static void owner_free(ModelOwner *owner, uint32_t layers) {
    if (!owner) return;
    prepared_free(owner->layers, layers);
    predict_workspace_free(atomic_load(&owner->spare));
    spg_signal_free(owner->lock);
    spingalett_aligned_free(owner->image);
    free(owner);
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

/* Whether integer layer L runs in tiles against interleaved rows: where the target has the
   kernel, convolutions with one group always, and dense layers in calls with a tile of samples. */
static bool layer_tiled(const SlettLayer *L, bool dense_tiles) {
    if (!spingalett_precision_is_int(L->precision) || L->rows == 0 || !spingalett_i8_interleaved()) return false;
    return L->type == LAYER_CONV2D ? L->groups <= 1 : L->type == LAYER_DENSE && dense_tiles;
}

/* Integer convolutions that run filter-major, as the engine runs short windows and depthwise
   convolutions. */
static bool layer_columns(const SlettLayer *L) {
    return L->type == LAYER_CONV2D && spingalett_precision_is_int(L->precision) && !layer_tiled(L, false) &&
           (slett_conv_columns(L) || slett_conv_depthwise(L));
}

/* Interleaves the rows of integer layer L (bytes: its INT4 / INT2 rows as bytes, or NULL). */
static bool prepare_tiles(const uint8_t *image, const SlettLayer *L, const int8_t *bytes, PreparedLayer *p) {
    const int8_t *rows = (const int8_t *)(image + L->weights);
    size_t stride = (size_t)spingalett_slett_row_bytes(L->precision, L->row_len);
    int8_t *u = NULL;
    if (L->precision != PRECISION_INT8) {
        if (!bytes) {
            u = (int8_t *)spingalett_aligned_alloc((size_t)L->rows * L->row_len);
            if (!u) return false;
            unpack_rows(image + L->weights, L, u);
            bytes = u;
        }
        rows = bytes;
        stride = L->row_len;
    }
    const uint32_t R = spingalett_i8_interleaved_rows(L->rows);
    int8_t *interleaved = (int8_t *)spingalett_aligned_alloc((size_t)R * spingalett_i8_interleaved_len(L->row_len));
    int32_t *sums = (int32_t *)spingalett_aligned_alloc((size_t)R * sizeof(int32_t));
    bool ok = interleaved && sums;
    if (ok) {
        spingalett_i8_interleave(rows, stride, L->rows, L->row_len, interleaved, sums);
        p->interleaved = interleaved;
        p->tile_sums = sums;
    } else {
        spingalett_aligned_free(interleaved);
        spingalett_aligned_free(sums);
    }
    spingalett_aligned_free(u);
    return ok;
}

/* Builds what layer L runs on in every call (pieces already there are kept, so a failed attempt
   can be resumed). */
static bool prepare_layer(const uint8_t *image, const SlettLayer *L, PreparedLayer *p) {
    if (L->rows == 0 || L->type == LAYER_BATCH_NORM) return true;      /* pooling, normalization */
    const size_t n = (size_t)L->rows * L->row_len;
    const uint8_t *src = image + L->weights;
    if ((L->precision == PRECISION_FP16 || L->precision == PRECISION_BFLOAT16) && !p->dequant) {
        float *d = (float *)spingalett_aligned_alloc(n * sizeof(float));
        if (!d) return false;
        for (size_t k = 0; k < n; k++) {
            uint16_t h = slett_get16(src + 2u * k);
            d[k] = L->precision == PRECISION_FP16 ? spingalett_fp16_to_float(h) : spingalett_bf16_to_float(h);
        }
        p->dequant = d;
    }
    if (!spingalett_precision_is_int(L->precision)) return true;
    const bool conv = L->type == LAYER_CONV2D, columns = layer_columns(L);
    if (conv && layer_tiled(L, false)) return p->interleaved || prepare_tiles(image, L, NULL, p);
    if ((columns || L->precision != PRECISION_INT8) && !p->unpacked) {
        int8_t *u = (int8_t *)spingalett_aligned_alloc(n);
        if (!u) return false;
        if (columns) spingalett_conv_transpose_filters(image, L, u);
        else unpack_rows(src, L, u);
        p->unpacked = u;
    }
    /* the kernels of four rows by four vectors take the rows' sums */
    if (!columns && (!conv || L->groups <= 1) && spingalett_dot_i8_4x4_sums && !p->sums) {
        const int8_t *rows = p->unpacked ? p->unpacked : (const int8_t *)src;
        int32_t *sums = (int32_t *)spingalett_aligned_alloc((size_t)L->rows * sizeof(int32_t));
        if (!sums) return false;
        for (uint32_t j = 0; j < L->rows; j++) sums[j] = spingalett_sum_i8(rows + (size_t)j * L->row_len, L->row_len);
        p->sums = sums;
    }
    return true;
}

/* Prepares every layer for a call (dense_tiles: with a tile of samples or more). */
static bool prepare_model(const SpingalettModel *model, PreparedLayer *layers, bool dense_tiles) {
    const uint8_t *image = (const uint8_t *)model->image;
    for (uint32_t i = 0; i < model->layer_count; i++) {
        SlettLayer L;
        spingalett_slett_layer(image, i, &L);
        PreparedLayer *p = &layers[i];
        if (!p->ready) {
            if (!prepare_layer(image, &L, p)) return false;
            p->ready = true;
        }
        if (L.type == LAYER_DENSE && layer_tiled(&L, dense_tiles) && !p->interleaved &&
            !prepare_tiles(image, &L, p->unpacked, p))
            return false;
    }
    return true;
}

static void *workspace_alloc(PredictWorkspace *w, size_t bytes) {
    w->bytes += bytes;
    return spingalett_aligned_alloc(bytes);
}

static PredictWorkspace *predict_workspace_create(const SpingalettModel *model, uint32_t chunk, ComputeMode mode,
                                                  bool dense_tiles) {
    PredictWorkspace *w = (PredictWorkspace *)calloc(1, sizeof *w);
    if (!w) return NULL;
    const uint8_t *image = (const uint8_t *)model->image;
    w->threads = 1;
#if defined(_OPENMP)
    if (mode == COMPUTE_OPENMP) w->threads = omp_get_max_threads();
#endif
    w->chunk = chunk;
    w->mode = mode;
    w->dense_tiles = dense_tiles;
    uint32_t int_inputs = 0, int_window = 0, norm_channels = 0;
    size_t conv_floats = 0, tile_x = 0, tile_acc = 0;
    bool has_float = false, ok = true;
    for (uint32_t i = 0; i < model->layer_count; i++) {
        SlettLayer L;
        spingalett_slett_layer(image, i, &L);
        if (L.type == LAYER_BATCH_NORM && L.out_c > norm_channels) norm_channels = L.out_c;
        if (L.rows == 0 || L.type == LAYER_BATCH_NORM) continue;      /* pooling, normalization */
        bool conv = L.type == LAYER_CONV2D;
        if (spingalett_precision_is_int(L.precision)) {
            if (L.inputs > int_inputs) int_inputs = L.inputs;
            if (layer_tiled(&L, dense_tiles)) {
                size_t x = (size_t)SPINGALETT_I8_TILE * spingalett_i8_interleaved_len(L.row_len);
                size_t acc = (size_t)SPINGALETT_I8_TILE * spingalett_i8_interleaved_rows(L.rows) * sizeof(int32_t);
                if (x > tile_x) tile_x = x;
                if (acc > tile_acc) tile_acc = acc;
            } else if (conv) {
                /* one pixel's sums, a group's window, or the windows of four pixels */
                uint32_t per_thread = layer_columns(&L) ? L.rows * 4u : L.groups > 1 ? L.row_len : 4u * L.row_len;
                if (per_thread > int_window) int_window = per_thread;
            }
        } else {
            has_float = true;
            if (conv) {
                LayerShape in = input_shape(&L), out = output_shape(&L);
                size_t need = spingalett_conv_forward_scratch(&in, &out, chunk, mode);
                if (need > conv_floats) conv_floats = need;
            }
        }
    }
    if (slett_get16(image + 6) >= 6u) {
        w->act[0] = (float *)workspace_alloc(w, (model->activations_ ? model->activations_ : 4u) * (size_t)chunk);
        ok = w->act[0] != NULL;
    } else {
        size_t width = (size_t)chunk * (model->max_width_ ? model->max_width_ : 1u);
        w->act[0] = (float *)workspace_alloc(w, width * sizeof(float));
        w->act[1] = (float *)workspace_alloc(w, width * sizeof(float));
        ok = w->act[0] && w->act[1];
    }
    if (ok && int_inputs) {
        w->xq = (int8_t *)workspace_alloc(w, (size_t)chunk * int_inputs);
        w->xs = (float *)workspace_alloc(w, (size_t)chunk * sizeof(float));
        ok = w->xq && w->xs;
        if (ok && int_window) {
            w->window_stride = ((size_t)int_window + 63u) & ~(size_t)63u;    /* threads on their own cache lines */
            w->window = (int8_t *)workspace_alloc(w, (size_t)w->threads * w->window_stride);
            ok = w->window != NULL;
        }
        if (ok && tile_x) {
            w->tile_sums = (tile_x + 63u) & ~(size_t)63u;
            w->tile_stride = (w->tile_sums + tile_acc + 63u) & ~(size_t)63u;
            w->tile = (int8_t *)workspace_alloc(w, (size_t)w->threads * w->tile_stride);
            ok = w->tile != NULL;
            /* the vectors of a tile past the last pixel or sample keep the values they have: they
               must have some */
            if (ok) memset(w->tile, 0, (size_t)w->threads * w->tile_stride);
        }
    }
    if (ok && conv_floats) {
        w->conv = (float *)workspace_alloc(w, conv_floats * sizeof(float));
        ok = w->conv != NULL;
    }
    if (ok && norm_channels) {
        w->norm = (float *)workspace_alloc(w, 2u * (size_t)norm_channels * sizeof(float));
        ok = w->norm != NULL;
    }
    if (ok && has_float && mode != COMPUTE_OPENBLAS) {
        w->gemm = spingalett_gemm_scratch_create(w->threads);
        w->bytes += spingalett_gemm_scratch_bytes(w->threads);
        ok = w->gemm != NULL;
    }
    if (!ok) {
        predict_workspace_free(w);
        return NULL;
    }
    return w;
}

/* An integer multiply-add costs a fraction of a float one, so integer layers start threads only
   for more work than SPINGALETT_OMP_MIN_WORK (a dense layer of 784 x 256 on one sample does not:
   its threads would take longer to start than the layer to run). */
#define INT_OMP_MIN_WORK ((uint64_t)1 << 19)

static bool use_omp_int(ComputeMode mode, uint64_t work) {
    return mode == COMPUTE_OPENMP && work >= INT_OMP_MIN_WORK;
}

static void activate_samples(float *y, uint32_t n, uint32_t size, ActivationFunction act, ComputeMode mode) {
    if (act == ACT_NONE) return;
    SPINGALETT_PARALLEL_FOR(spingalett_use_omp(mode, (uint64_t)n * size),
        for (int64_t s = 0; s < (int64_t)n; s++)
            spingalett_engine_activate(y + (size_t)s * size, size, act);
    );
    (void)mode;
}

/* An integer convolution over n samples, as spingalett_model_run computes it: each sample's input
   quantized as a whole, each output pixel the integer dot products of the filters with its
   quantized window (filters: rows of bytes, or transposed for short windows). */
/* The outputs of n integer rows for SPINGALETT_I8_TILE vectors (count of them real) against their
   interleaved rows: vector i of the tile (in w->tile of thread t, its bytes in place) goes to ys[i]
   with activation scale xs[i]. */
/* The outputs of `rows` integer sums (restrict: no overlap to check before every vector). */
static inline void int_outputs(float *restrict y, const float *restrict bias, const float *restrict scale,
                               float x_scale, const int32_t *restrict a, uint32_t rows) {
    for (uint32_t j = 0; j < rows; j++) y[j] = spingalett_int_output(bias[j], scale[j], x_scale, a[j]);
}

static void tile_outputs(const int8_t *interleaved, const int32_t *sums, const float *bias, const float *scale,
                         uint32_t rows, uint32_t n, PredictWorkspace *w, int t, uint32_t count, float *const ys[],
                         const float xs[]) {
    int8_t *tile = w->tile + (size_t)t * w->tile_stride;
    int32_t *acc = (int32_t *)(void *)(tile + w->tile_sums);
    const uint32_t R = spingalett_i8_interleaved_rows(rows);
    spingalett_i8_interleaved_tile(interleaved, sums, rows, n, tile, spingalett_i8_interleaved_len(n), acc, R);
    for (uint32_t i = 0; i < count; i++) int_outputs(ys[i], bias, scale, xs[i], acc + (size_t)i * R, rows);
}

static void predict_int_conv(const uint8_t *image, const SlettLayer *L, const PreparedLayer *prep, const float *x,
                             float *y, uint32_t n, PredictWorkspace *w, ComputeMode mode) {
    const int8_t *filters = prep->unpacked ? prep->unpacked : (const int8_t *)(image + L->weights);
    const uint32_t in = L->inputs, out = L->outputs, OC = L->rows, K = L->row_len, C = L->in_c;
    const uint64_t pixels = (uint64_t)L->out_h * L->out_w, items = (uint64_t)n * pixels;
    const float *bias = (const float *)(const void *)(image + L->biases);
    const float *scale = (const float *)(const void *)(image + L->scales);
    const bool pointwise = L->kernel_h == 1 && L->kernel_w == 1 && L->stride_h == 1 && L->stride_w == 1 &&
                           L->pad_h == 0 && L->pad_w == 0;
    bool parallel = use_omp_int(mode, items * K * OC);

    SPINGALETT_PARALLEL_FOR(parallel,
        for (int64_t s = 0; s < (int64_t)n; s++)
            w->xs[s] = spingalett_quantize_activations(x + (size_t)s * in, in, w->xq + (size_t)s * in);
    );

    if (prep->interleaved) {
        /* a tile of pixels at a time against all filters */
        const uint32_t len = spingalett_i8_interleaved_len(K);
        const int64_t tiles = (int64_t)((items + SPINGALETT_I8_TILE - 1u) / SPINGALETT_I8_TILE);
        SPINGALETT_PARALLEL_FOR(parallel,
            for (int64_t tile = 0; tile < tiles; tile++) {
                int t = spingalett_thread_num();
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
                tile_outputs(prep->interleaved, prep->tile_sums, bias, scale, OC, K, w, t, count, ys, xs);
            }
        );
        activate_samples(y, n, out, L->activation, mode);
        return;
    }

    /* short windows, depthwise and grouped convolutions pixel by pixel; the others four pixels at
       a time, so that each filter is read once for four windows */
    const bool quads = !slett_conv_columns(L) && !slett_conv_depthwise(L) && L->groups <= 1;
    const int64_t units = quads ? (int64_t)((items + 3u) / 4u) : (int64_t)items;
    SPINGALETT_PARALLEL_FOR(parallel,
        for (int64_t unit = 0; unit < units; unit++) {
            int t = spingalett_thread_num();
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
                        spingalett_dot_i8_4x4(block, K, windows, K, K, prep->sums ? prep->sums + j : NULL, acc);
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
    );
    (void)parallel;
    activate_samples(y, n, out, L->activation, mode);
}

/* An addition, concatenation or global pooling over n samples, sample by sample with the engine's
   functions, so that the results are the engine's (x[k]: input k, units[k] per sample). */
static void predict_combine(const SlettLayer *L, const float *const *x, const uint32_t *units, const uint32_t *channels,
                            float *y, uint32_t n, ComputeMode mode) {
    SPINGALETT_PARALLEL_FOR(spingalett_use_omp(mode, (uint64_t)n * L->outputs * L->input_count),
        for (int64_t s = 0; s < (int64_t)n; s++) {
            const float *xs[SPINGALETT_MAX_INPUTS];
            for (uint32_t k = 0; k < L->input_count; k++) xs[k] = x[k] + (size_t)s * units[k];
            float *ys = y + (size_t)s * L->outputs;
            if (L->type == LAYER_ADD) spingalett_engine_add(xs, L->input_count, ys, L->outputs);
            else if (L->type == LAYER_CONCAT) spingalett_engine_concat(xs, channels, L->input_count, ys, L->out_h * L->out_w);
            else spingalett_engine_global_pool(xs[0], ys, L->in_h * L->in_w, L->in_c);
            spingalett_engine_activate(ys, L->outputs, L->activation);
        }
    );
    (void)mode;
}

/* One layer over n samples: x [n x inputs] -> y [n x outputs], activation included. A dense
   layer's interleaved rows are read only in calls with dense_tiles (others may be building them). */
static void predict_layer(const uint8_t *image, const SlettLayer *L, const PreparedLayer *prep, bool dense_tiles,
                          const float *x, float *y, uint32_t n, PredictWorkspace *w, ComputeMode mode) {
    uint32_t in = L->inputs, out = L->outputs;
    const float *dequant = prep->dequant;
    const float *bias = (const float *)(const void *)(image + L->biases);

    if (L->type == LAYER_BATCH_NORM) {
        /* as the engine computes it: y = x a + b per channel, then the activation */
        const uint32_t C = L->out_c;
        const float *stats = (const float *)(const void *)(image + L->scales);
        float *a = w->norm, *b = w->norm + C;
        spingalett_bn_coefficients((const float *)(const void *)(image + L->weights), bias, stats, stats + C, L->eps, C,
                                   a, b);
        SPINGALETT_PARALLEL_FOR(spingalett_use_omp(mode, (uint64_t)n * out),
            for (int64_t s = 0; s < (int64_t)n; s++) {
                const float *xs = x + (size_t)s * out;
                float *ys = y + (size_t)s * out;
                for (uint32_t i = 0; i < out; i += C)
                    for (uint32_t c = 0; c < C; c++) ys[i + c] = xs[i + c] * a[c] + b[c];
                spingalett_engine_activate(ys, out, L->activation);
            }
        );
        return;
    }
    if (L->type != LAYER_DENSE) {
        LayerShape is = input_shape(L), os = output_shape(L);
        if (L->type != LAYER_CONV2D) {
            spingalett_pool_forward_shapes(&is, &os, x, y, n, mode);
            activate_samples(y, n, out, L->activation, mode);
        } else if (spingalett_precision_is_int(L->precision)) {
            predict_int_conv(image, L, prep, x, y, n, w, mode);
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
        SPINGALETT_PARALLEL_FOR(spingalett_use_omp(mode, (uint64_t)n * out),
            for (int64_t s = 0; s < (int64_t)n; s++) {
                float *row = y + (size_t)s * out;
                spingalett_vec_axpy(row, bias, out, 1.0f);
                spingalett_engine_activate(row, out, L->activation);
            }
        );
        return;
    }

    const float *scale = (const float *)(const void *)(image + L->scales);
    const uint8_t *weights = image + L->weights;
    size_t row_bytes = (size_t)spingalett_slett_row_bytes(L->precision, in);
    bool parallel = use_omp_int(mode, (uint64_t)n * in * out);

    SPINGALETT_PARALLEL_FOR(parallel,
        for (int64_t s = 0; s < (int64_t)n; s++)
            w->xs[s] = spingalett_quantize_activations(x + (size_t)s * in, in, w->xq + (size_t)s * in);
    );

    const int8_t *interleaved = dense_tiles ? prep->interleaved : NULL;
    if (interleaved && n >= SPINGALETT_I8_TILE) {
        /* a tile of samples at a time against all rows */
        const uint32_t len = spingalett_i8_interleaved_len(in);
        const int64_t tiles = ((int64_t)n + SPINGALETT_I8_TILE - 1) / SPINGALETT_I8_TILE;
        SPINGALETT_PARALLEL_FOR(parallel,
            for (int64_t tile = 0; tile < tiles; tile++) {
                int t = spingalett_thread_num();
                const uint32_t s0 = (uint32_t)tile * SPINGALETT_I8_TILE;
                const uint32_t count = n - s0 < SPINGALETT_I8_TILE ? n - s0 : SPINGALETT_I8_TILE;
                int8_t *xt = w->tile + (size_t)t * w->tile_stride;
                float *ys[SPINGALETT_I8_TILE];
                for (uint32_t i = 0; i < count; i++) {
                    memcpy(xt + (size_t)i * len, w->xq + (size_t)(s0 + i) * in, in);
                    ys[i] = y + (size_t)(s0 + i) * out;
                }
                tile_outputs(interleaved, prep->tile_sums, bias, scale, out, in, w, t, count, ys, w->xs + s0);
            }
        );
    } else {
        /* blocks of four weight rows (INT4 and INT2 rows unpacked to bytes) against every sample */
        const int8_t *rows_base = prep->unpacked ? prep->unpacked : (const int8_t *)weights;
        const size_t stride = prep->unpacked ? in : row_bytes;
        int64_t blocks = ((int64_t)out + 3) / 4;
        SPINGALETT_PARALLEL_FOR(parallel,
            for (int64_t b = 0; b < blocks; b++) {
                uint32_t j = (uint32_t)b * 4u, rows = out - j < 4u ? out - j : 4u;
                const int8_t *block = rows_base + (size_t)j * stride;
                uint32_t s = 0;
                static const int32_t no_sums[4] = {0, 0, 0, 0};
                const int32_t *wsum = prep->sums && rows == 4 ? prep->sums + j : no_sums;
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
        );
    }
    (void)parallel;

    SPINGALETT_PARALLEL_FOR(spingalett_use_omp(mode, (uint64_t)n * out),
        for (int64_t s = 0; s < (int64_t)n; s++)
            spingalett_engine_activate(y + (size_t)s * out, out, L->activation);
    );
}

bool spingalett_model_predict(const SpingalettModel *model, const float *inputs, uint32_t count, float *outputs) {
    if (!model || !model->image || !inputs || !outputs) {
        set_error(SPINGALETT_ERR_INVALID, "model predict: model, inputs or outputs is NULL");
        return false;
    }
    if (count == 0) return true;

    ComputeMode mode = resolve_compute_mode();
    const uint8_t *image = (const uint8_t *)model->image;
    const bool graph = slett_get16(image + 6) >= 6u;
    /* fewer samples per pass when the layers are wide (convolutions): the outputs of a graph and
       the two alternating buffers of a chain hold at most SPINGALETT_BATCH_FLOATS / 2 floats */
    uint64_t width = graph ? model->activations_ / 8u : model->max_width_;
    if (width < model->input_size) width = model->input_size;
    uint64_t cap64 = SPINGALETT_BATCH_FLOATS / 4u / (width ? width : 1u);
    uint32_t cap = cap64 > MODEL_CHUNK ? MODEL_CHUNK : (uint32_t)cap64;
    if (cap > MODEL_CHUNK) cap = MODEL_CHUNK;
    if (cap < 1) cap = 1;
    uint32_t chunk = count < cap ? count : cap;
    const bool dense_tiles = chunk >= SPINGALETT_I8_TILE;

    /* the weights: prepared once for a model that owns them, for this call otherwise */
    ModelOwner *owner = (ModelOwner *)model->owner_;
    PreparedLayer *layers = NULL;
    bool ok;
    if (owner) {
        spg_lock(owner->lock);
        if (!owner->layers) owner->layers = (PreparedLayer *)calloc(model->layer_count, sizeof(PreparedLayer));
        layers = owner->layers;
        ok = layers && prepare_model(model, layers, dense_tiles);
        spg_unlock(owner->lock);
    } else {
        layers = (PreparedLayer *)calloc(model->layer_count, sizeof(PreparedLayer));
        ok = layers && prepare_model(model, layers, dense_tiles);
    }

    /* the workspace of an earlier call when it fits this one */
    PredictWorkspace *w = owner ? atomic_exchange(&owner->spare, NULL) : NULL;
    int threads = 1;
#if defined(_OPENMP)
    if (mode == COMPUTE_OPENMP) threads = omp_get_max_threads();
#endif
    if (w && (w->chunk < chunk || w->threads != threads || w->mode != mode || (dense_tiles && !w->dense_tiles))) {
        predict_workspace_free(w);
        w = NULL;
    }
    if (ok && !w) w = predict_workspace_create(model, chunk, mode, dense_tiles);
    if (!ok || !w) {
        predict_workspace_free(w);
        if (!owner) prepared_free(layers, model->layer_count);
        set_error(SPINGALETT_ERR_ALLOC, "model predict: workspace allocation failed");
        return false;
    }

    for (uint32_t start = 0; start < count; start += chunk) {
        uint32_t n = count - start < chunk ? count - start : chunk;
        const float *x = inputs + (size_t)start * model->input_size;
        float *out = outputs + (size_t)start * model->output_size;
        for (uint32_t i = 0; i < model->layer_count; i++) {
            SlettLayer L;
            spingalett_slett_layer(image, i, &L);
            if (!graph) {
                float *y = i + 1 == model->layer_count ? out : w->act[i & 1u];
                predict_layer(image, &L, &layers[i], dense_tiles, x, y, n, w, mode);
                x = y;
                continue;
            }
            /* a graph: each output at chunk times its place in the engine's activations */
            float *y = i + 1 == model->layer_count ? out : w->act[0] + L.act_offset / 4u * chunk;
            const float *xs[SPINGALETT_MAX_INPUTS];
            uint32_t units[SPINGALETT_MAX_INPUTS], channels[SPINGALETT_MAX_INPUTS];
            for (uint32_t k = 0; k < L.input_count; k++) {
                uint32_t j = spingalett_slett_input(image, &L, k), h, wd;
                spingalett_slett_output_shape(image, j, &h, &wd, &units[k]);
                channels[k] = units[k] / (h * wd);
                xs[k] = j == 0 ? x : w->act[0] + spingalett_slett_act_offset(image, j) / 4u * chunk;
            }
            if (L.type == LAYER_ADD || L.type == LAYER_CONCAT || L.type == LAYER_GLOBAL_AVG_POOL)
                predict_combine(&L, xs, units, channels, y, n, mode);
            else
                predict_layer(image, &L, &layers[i], dense_tiles, xs[0], y, n, w, mode);
        }
    }
    if (owner && w->bytes <= SPARE_WORKSPACE_MAX) w = atomic_exchange(&owner->spare, w);
    predict_workspace_free(w);
    if (!owner) prepared_free(layers, model->layer_count);
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
