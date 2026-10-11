/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/*
 * Sessions of deployment models (spingalett_session_*, Spingalett.Runtime.h): a causal language model run
 * a token at a time. Every layer of such a model but attention treats each position alone, so a new token
 * costs one row through each of them; an attention layer keeps the keys (rotated, with rotary embeddings)
 * and values of each sequence's tokens, and a new token's queries attend over those. A feed takes the rows
 * of every sequence it is given in passes of up to `capacity` rows, the rows of a pass through each layer
 * together, and the last layer (the logits) on the rows whose logits are asked for only.
 *
 * A row gets the bits batched prediction (spingalett_model_predict()) gives its position in the whole
 * window: the float linear layers multiply in the order of the forward kernel (spingalett_conv_rows() on
 * panels packed once per model), the integer ones quantize each token alone (SlettLayer.cells) and sum in
 * integers, attention attends as the engine does (spingalett_engine_attend()), the other layers are
 * batched prediction's own run on the rows, and an activation takes the vector or scalar path it takes at
 * the row's place in a whole sample (activate_rows()). So a row does not depend on the rows computed with
 * it: a prompt fed at once, a token at a time, or with other sequences gives the same bits.
 */

#include "Spingalett.Model.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#if defined(_OPENMP)
#include <omp.h>
#endif

/* Multiply-adds from which few rows of a linear layer split its columns between the threads (float, and
   integer, whose multiply-adds cost less). */
#define ROWS_PARALLEL_WORK ((uint64_t)1 << 17)
#define INT_ROWS_PARALLEL_WORK ((uint64_t)1 << 19)

/* A pass takes a thread for this many multiply-adds of its linear layers, all of them at most, and its
   parallel loops run on that team alone: a token's pass is short, and more threads than its work keeps
   busy, or teams of changing sizes (OpenMP wakes the threads a smaller team left asleep), cost more than
   they save (a token of the CharGPT model took three times as long on 16 threads as on one). */
#define PASS_THREAD_WORK ((uint64_t)1 << 19)

/* Integer linear layers run on few rows (a token of each sequence) with their columns split between
   threads, up to this many rows; more go through batched prediction's pass, a tile of rows a thread. */
#define INT_ROWS SPINGALETT_I8_TILE

/* The keys and values one attention layer keeps for every sequence. */
typedef struct {
    uint32_t heads, kv, head;
    float *keys, *values;           /* [sequences x max_length x kv x head] each, the keys rotated */
    float *table;                   /* the rotary embeddings of max_length positions (cosines, then sines), or NULL */
} Cache;

struct SpingalettSession {
    const SpingalettModel *model;
    uint32_t cells;                 /* the model's window: the positions of every layer */
    uint32_t max_length, sequences, vocabulary;
    uint32_t *length;               /* [sequences] tokens each holds */
    uint32_t layers;                /* the model's weight layers */
    uint64_t token_work;            /* multiply-adds of a token's row through the linear layers */
    int team;                       /* threads of the current pass */
    Cache *caches;                  /* [layers]: those of its attention layers */
    PreparedLayer *prepared;        /* the model's (prepared under its lock), or the session's own */
    /* a pass: up to capacity rows, m times the cells (the workspace's chunk) */
    uint32_t capacity, m;
    PredictWorkspace *w;
    uint32_t *row_sequence, *row_position;
    float *row_token, **row_logits;     /* the logits a row writes, or NULL */
    int32_t *row_sums;                  /* the sums of the rows' permuted INT4 and INT2 activations */
    uint32_t *picked, *picked_position; /* the rows whose logits a pass writes, and their positions */
    float *queries;                 /* a rotated query a thread */
    uint32_t query_floats;
    float *gathered, *out;          /* the last layer's inputs on the picked rows, and its outputs */
    size_t gathered_floats, out_floats;
};

/* ------------------------------------------------------------------------- the model */

static const char *layer_names[LAYER_TYPE_COUNT] = {"dense", "convolution", "max pooling", "average pooling",
    "batch normalization", "add", "concat", "global pooling", "transposed convolution", "upsampling",
    "layer normalization", "embedding", "attention", "RMS normalization", "multiply"};

/* Whether a session runs the model: why not in *why. cells: the window, vocabulary: logits a position. */
static bool session_model(const SpingalettModel *model, uint32_t *cells, uint32_t *vocabulary, char *why, size_t size) {
    const uint8_t *image = (const uint8_t *)model->image;
    if (!spingalett_slett_language_model(image)) {
        snprintf(why, size, "its first layer is no embedding of the input tokens");
        return false;
    }
    SlettLayer L;
    spingalett_slett_layer(image, 0, &L);
    *cells = L.out_h * L.out_w;
    for (uint32_t i = 1; i < model->layer_count; i++) {
        spingalett_slett_layer(image, i, &L);
        const char *name = (unsigned)L.type < LAYER_TYPE_COUNT ? layer_names[L.type] : "unknown";
        bool fits = L.out_h * L.out_w == *cells;
        for (uint32_t k = 0; k < L.input_count && fits; k++) {
            uint32_t j = spingalett_slett_input(image, &L, k), h, w, units;
            spingalett_slett_output_shape(image, j, &h, &w, &units);
            fits = j > 0 && h * w == *cells;
        }
        if (!fits) {
            snprintf(why, size, "layer %u (%s) reads the tokens or changes the positions", i + 1, name);
            return false;
        }
        if (L.activation == ACT_SOFTMAX) {
            snprintf(why, size, "layer %u's softmax spans every position", i + 1);
            return false;
        }
        switch (L.type) {
            case LAYER_CONV2D:
                if (slett_conv_pointwise(&L) && L.groups == 1) continue;
                snprintf(why, size, "layer %u is a convolution of more than one position, or of groups", i + 1);
                return false;
            case LAYER_ATTENTION:
                if (L.causal) continue;
                snprintf(why, size, "layer %u's attention is not causal", i + 1);
                return false;
            case LAYER_LAYER_NORM:
            case LAYER_RMS_NORM:
            case LAYER_BATCH_NORM:
            case LAYER_ADD:
            case LAYER_MULTIPLY:
            case LAYER_CONCAT:
                continue;
            default:
                snprintf(why, size, "layer %u (%s) mixes positions", i + 1, name);
                return false;
        }
    }
    *vocabulary = L.out_c;
    return true;
}

bool spingalett_session_runs(const SpingalettModel *model) {
    uint32_t cells, vocabulary;
    char why[160];
    return model && model->image && session_model(model, &cells, &vocabulary, why, sizeof why);
}

static void caches_free(Cache *caches, uint32_t layers) {
    if (!caches) return;
    for (uint32_t i = 0; i < layers; i++) {
        spingalett_aligned_free(caches[i].keys);
        spingalett_aligned_free(caches[i].values);
        spingalett_aligned_free(caches[i].table);
    }
    free(caches);
}

void spingalett_session_free(SpingalettSession *s) {
    if (!s) return;
    caches_free(s->caches, s->layers);
    spingalett_model_release_layers(s->model, s->prepared);
    spingalett_predict_workspace_free(s->w);
    free(s->length);
    free(s->row_sequence);
    free(s->row_position);
    free(s->row_token);
    free(s->row_logits);
    free(s->row_sums);
    free(s->picked);
    free(s->picked_position);
    spingalett_aligned_free(s->queries);
    spingalett_aligned_free(s->gathered);
    spingalett_aligned_free(s->out);
    free(s);
}

/* The workspace for the compute mode and its threads (made again when they changed). */
static bool session_workspace(SpingalettSession *s, ComputeMode mode) {
    int threads = 1;
#if defined(_OPENMP)
    if (mode == COMPUTE_OPENMP) threads = omp_get_max_threads();
#endif
    if (s->w && s->w->mode == mode && s->w->threads == threads) return true;
    spingalett_predict_workspace_free(s->w);
    spingalett_aligned_free(s->queries);
    s->w = spingalett_predict_workspace_create(s->model, s->m, mode, false, true);
    s->queries = (float *)spingalett_aligned_alloc((size_t)threads * s->query_floats * sizeof(float));
    return s->w && s->queries;
}

SpingalettSession *spingalett_session_new(const SpingalettModel *model, const SpingalettSessionOptions *options) {
    if (!model || !model->image) {
        set_error(SPINGALETT_ERR_INVALID, "session: model is NULL");
        return NULL;
    }
    char why[160], message[256];
    uint32_t cells = 0, vocabulary = 0;
    if (!session_model(model, &cells, &vocabulary, why, sizeof why)) {
        snprintf(message, sizeof message, "session: the model is no causal language model: %s", why);
        set_error(SPINGALETT_ERR_INVALID, message);
        return NULL;
    }
    const uint32_t max_length = options && options->max_length ? options->max_length : cells;
    const uint32_t sequences = options && options->sequences ? options->sequences : 1u;
    if (max_length > cells) {
        snprintf(message, sizeof message, "session: max_length %u exceeds the model's window of %u tokens", max_length,
                 cells);
        set_error(SPINGALETT_ERR_INVALID, message);
        return NULL;
    }
    SpingalettSession *s = (SpingalettSession *)calloc(1, sizeof *s);
    if (!s) {
        set_error(SPINGALETT_ERR_ALLOC, "session: allocation failed");
        return NULL;
    }
    s->model = model;
    s->cells = cells;
    s->max_length = max_length;
    s->sequences = sequences;
    s->vocabulary = vocabulary;
    s->layers = model->layer_count;
    /* a pass holds the window's rows, and a token of every sequence */
    s->m = (sequences + cells - 1) / cells;
    s->capacity = s->m * cells;
    s->length = (uint32_t *)calloc(sequences, sizeof(uint32_t));
    s->caches = (Cache *)calloc(s->layers, sizeof(Cache));
    s->row_sequence = (uint32_t *)malloc((size_t)s->capacity * sizeof(uint32_t));
    s->row_position = (uint32_t *)malloc((size_t)s->capacity * sizeof(uint32_t));
    s->row_token = (float *)malloc((size_t)s->capacity * sizeof(float));
    s->row_logits = (float **)malloc((size_t)s->capacity * sizeof(float *));
    s->row_sums = (int32_t *)malloc((size_t)s->capacity * sizeof(int32_t));
    s->picked = (uint32_t *)malloc((size_t)s->capacity * sizeof(uint32_t));
    s->picked_position = (uint32_t *)malloc((size_t)s->capacity * sizeof(uint32_t));
    bool ok = s->length && s->caches && s->row_sequence && s->row_position && s->row_token && s->row_logits &&
              s->row_sums && s->picked && s->picked_position;
    const uint8_t *image = (const uint8_t *)model->image;
    for (uint32_t i = 0; ok && i < s->layers; i++) {
        SlettLayer L;
        spingalett_slett_layer(image, i, &L);
        if (L.type == LAYER_CONV2D) s->token_work += (uint64_t)L.rows * L.row_len;
        if (L.type != LAYER_ATTENTION) continue;
        Cache *c = &s->caches[i];
        c->heads = L.heads, c->kv = L.kv_heads, c->head = L.out_c / L.heads;
        const size_t floats = (size_t)sequences * max_length * c->kv * c->head;
        c->keys = (float *)spingalett_aligned_alloc(floats * sizeof(float));
        c->values = (float *)spingalett_aligned_alloc(floats * sizeof(float));
        ok = c->keys && c->values;
        if (ok && L.theta > 0.0f) {
            c->table = (float *)spingalett_aligned_alloc((size_t)max_length * c->head * sizeof(float));
            ok = c->table != NULL;
            if (ok) spingalett_engine_rope_table(max_length, c->head, L.theta, c->table);
        }
        if (c->head > s->query_floats) s->query_floats = c->head;
    }
    if (ok) {
        s->prepared = spingalett_model_layers(model, false, true);
        ok = s->prepared && session_workspace(s, resolve_compute_mode());
    }
    if (!ok) {
        spingalett_session_free(s);
        set_error(SPINGALETT_ERR_ALLOC, "session: allocation failed");
        return NULL;
    }
    return s;
}

uint32_t spingalett_session_vocabulary(const SpingalettSession *s) {
    return s ? s->vocabulary : 0u;
}

uint32_t spingalett_session_length(const SpingalettSession *s, uint32_t sequence) {
    return s && sequence < s->sequences ? s->length[sequence] : 0u;
}

bool spingalett_session_rewind(SpingalettSession *s, uint32_t sequence, uint32_t length) {
    if (!s || sequence >= s->sequences || length > s->length[sequence]) {
        set_error(SPINGALETT_ERR_INVALID, "session rewind: no such sequence, or a length beyond its tokens");
        return false;
    }
    s->length[sequence] = length;
    return true;
}

bool spingalett_session_copy(SpingalettSession *s, uint32_t from, uint32_t to) {
    if (!s || from >= s->sequences || to >= s->sequences) {
        set_error(SPINGALETT_ERR_INVALID, "session copy: no such sequence");
        return false;
    }
    if (from == to) return true;
    for (uint32_t i = 0; i < s->layers; i++) {
        const Cache *c = &s->caches[i];
        if (!c->keys) continue;
        const size_t per = (size_t)c->kv * c->head, sequence = (size_t)s->max_length * per;
        memcpy(c->keys + to * sequence, c->keys + from * sequence, s->length[from] * per * sizeof(float));
        memcpy(c->values + to * sequence, c->values + from * sequence, s->length[from] * per * sizeof(float));
    }
    s->length[to] = s->length[from];
    return true;
}

/* ------------------------------------------------------------------------- passes */

/* The activation of `rows` rows of C values at positions pos[], as batched prediction applies it to a
   whole sample of `cells` cells. Element-wise activations give a value the same wherever it lies; sigmoid
   and tanh take a vector path for the groups of 8 values the sample holds whole from its first on, and
   the scalar one for the rest (apply_activation_bulk()), so each value takes the path of its place. */
static void activate_rows(float *y, uint32_t rows, uint32_t C, const uint32_t *pos, uint32_t cells,
                          ActivationFunction act) {
    if (act == ACT_NONE) return;
    if (act != ACT_SIGMOID && act != ACT_TANH) {
        apply_activation_bulk(y, (uint64_t)rows * C, act);
        return;
    }
    const uint64_t whole = (uint64_t)cells * C / 8u * 8u;
    for (uint32_t r = 0; r < rows; r++) {
        float *v = y + (size_t)r * C;
        const uint64_t first = (uint64_t)pos[r] * C;
        for (uint32_t c = 0; c < C;) {
            if (first + c >= whole) {
                v[c] = spingalett_activate_value(v[c], act);
                c++;
                continue;
            }
            uint32_t n = C - c < 8u ? C - c : 8u;
            if (whole - (first + c) < n) n = (uint32_t)(whole - (first + c));
            float t[8] = {0};
            memcpy(t, v + c, n * sizeof(float));
            apply_activation_bulk(t, 8, act);
            memcpy(v + c, t, n * sizeof(float));
            c += n;
        }
    }
}


/* An integer linear layer on few rows, its filters split between threads in blocks: each row quantized
   alone, as batched prediction quantizes each cell, then a block's integer sums against the interleaved
   filters for a tile of the rows at once (four rows and more), or against the stored rows by the engine's
   kernels for one vector (spingalett_engine_int_rows()). The sums are integers and the outputs
   spingalett_int_output()'s: batched prediction's bits. */
static void session_int_linear(SpingalettSession *s, const SlettLayer *L, const PreparedLayer *prep, const float *x,
                               float *y, uint32_t rows, ComputeMode mode) {
    const uint8_t *image = (const uint8_t *)s->model->image;
    PredictWorkspace *w = s->w;
    const uint32_t C = L->in_c, OC = L->rows, block = 64, len = spingalett_i8_interleaved_len(C);
    const uint32_t R = spingalett_i8_interleaved_rows(OC);
    const bool tiles = prep->interleaved && rows >= 4;
    const float *bias = (const float *)(const void *)(image + L->biases);
    const float *scale = (const float *)(const void *)(image + L->scales);
    for (uint32_t r = 0; r < rows; r++) {
        w->xs[r] = spingalett_quantize_activations(x + (size_t)r * C, C, w->xq + (size_t)r * C);
        s->row_sums[r] = tiles ? 0 : spingalett_engine_permute(w->xq + (size_t)r * C, C, L->precision);
    }
    (void)mode;
    SPINGALETT_PARALLEL_FOR_THREADS(s->team, s->team > 1 && (uint64_t)rows * OC * C >= INT_ROWS_PARALLEL_WORK,
        for (int64_t it = 0; it < (int64_t)((OC + block - 1) / block); it++) {
            const uint32_t j0 = (uint32_t)it * block, n = OC - j0 < block ? OC - j0 : block;
            if (!tiles) {
                for (uint32_t r = 0; r < rows; r++)
                    spingalett_engine_int_rows(image, L, j0, n, w->xq + (size_t)r * C, w->xs[r], s->row_sums[r],
                                               y + (size_t)r * OC + j0);
                continue;
            }
            /* (the filters interleaved in blocks of 16: block j0 / 16 at j0 len bytes) */
            int8_t *tile = w->tile + (size_t)spingalett_thread_num() * w->tile_stride;
            int32_t *acc = (int32_t *)(void *)(tile + w->tile_sums);
            for (uint32_t r = 0; r < rows; r++) memcpy(tile + (size_t)r * len, w->xq + (size_t)r * C, C);
            spingalett_i8_interleaved_tile(prep->interleaved + (size_t)j0 * len, prep->tile_sums + j0, n, C, tile, len,
                                           acc, R);
            for (uint32_t r = 0; r < rows; r++)
                for (uint32_t j = 0; j < n; j++)
                    y[(size_t)r * OC + j0 + j] = spingalett_int_output(bias[j0 + j], scale[j0 + j], w->xs[r],
                                                                       acc[(size_t)r * R + j]);
        }
    );
}

/* Attention on `rows` rows (row r of the pass, or the picked row r): their keys and values kept first,
   then each query over the keys of its sequence up to its own position. */
static void session_attention(SpingalettSession *s, const Cache *c, const float *x, float *y, uint32_t rows,
                              ComputeMode mode) {
    const uint32_t heads = c->heads, kv = c->kv, head = c->head, half = head / 2u, group = heads / kv;
    const uint32_t C = (heads + 2u * kv) * head, out_c = heads * head;
    const size_t per = (size_t)kv * head, sequence = (size_t)s->max_length * per;
    const float *sines = c->table ? c->table + (size_t)s->max_length * half : NULL;
    for (uint32_t r = 0; r < rows; r++) {
        const uint32_t p = s->row_position[r];
        float *keys = c->keys + s->row_sequence[r] * sequence + p * per;
        float *values = c->values + s->row_sequence[r] * sequence + p * per;
        for (uint32_t g = 0; g < kv; g++) {
            const float *k = x + (size_t)r * C + (size_t)(heads + g) * head;
            if (c->table) spingalett_engine_rope_rotate(k, keys + (size_t)g * head, half, c->table + (size_t)p * half,
                                                        sines + (size_t)p * half);
            else memcpy(keys + (size_t)g * head, k, head * sizeof(float));
            memcpy(values + (size_t)g * head, x + (size_t)r * C + (size_t)(heads + kv + g) * head, head * sizeof(float));
        }
    }
    const float scale = 1.0f / sqrtf((float)head);
    uint64_t work = 0;
    for (uint32_t r = 0; r < rows; r++) work += (uint64_t)(s->row_position[r] + 1u) * head * heads * 2u;
    SPINGALETT_PARALLEL_FOR_THREADS(s->team, s->team > 1 && spingalett_use_omp(mode, work),
        for (int64_t it = 0; it < (int64_t)rows * heads; it++) {
            const uint32_t r = (uint32_t)(it / heads), h = (uint32_t)(it % heads), g = h / group;
            const uint32_t p = s->row_position[r];
            const float *q = x + (size_t)r * C + (size_t)h * head;
            if (c->table) {
                float *rotated = s->queries + (size_t)spingalett_thread_num() * s->query_floats;
                spingalett_engine_rope_rotate(q, rotated, half, c->table + (size_t)p * half, sines + (size_t)p * half);
                q = rotated;
            }
            const size_t at = s->row_sequence[r] * sequence + (size_t)g * head;
            spingalett_engine_attend(q, c->keys + at, per, c->values + at, per, p + 1u, head, scale,
                                     y + (size_t)r * out_c + (size_t)h * head);
        }
    );
}

/* Layer L on `rows` rows at positions pos[] (inputs x[k] of channels[k] values a row, y of L's channels). */
static void session_layer(SpingalettSession *s, uint32_t index, const SlettLayer *L, const float *const *x,
                          const uint32_t *channels, float *y, uint32_t rows, const uint32_t *pos, ComputeMode mode) {
    const uint8_t *image = (const uint8_t *)s->model->image;
    const PreparedLayer *prep = &s->prepared[index];
    PredictWorkspace *w = s->w;
    if (L->type == LAYER_EMBEDDING) {
        for (uint32_t r = 0; r < rows; r++) spingalett_engine_embed(image, L, s->row_token[r], pos[r], y + (size_t)r * L->out_c);
    } else if (L->type == LAYER_ATTENTION) {
        session_attention(s, &s->caches[index], x[0], y, rows, mode);
    } else if (prep->panels) {
        /* a float linear layer: its bias and activation in the epilogue, row by row as batched prediction's */
        SpingalettBiasActivation epilogue = {(const float *)(const void *)(image + L->biases), L->activation, NULL};
        SpingalettGemmHooks hooks = {NULL, NULL, spingalett_epilogue_bias_activation, &epilogue};
        spingalett_conv_rows(x[0], L->in_c, rows, prep->panels, L->row_len, L->rows, y, L->out_c, &hooks,
                             s->team > 1 && (uint64_t)rows * L->rows * L->row_len >= ROWS_PARALLEL_WORK, s->team);
        return;
    } else if (L->type == LAYER_CONV2D && rows <= INT_ROWS) {
        session_int_linear(s, L, prep, x[0], y, rows, mode);
    } else {
        /* batched prediction's pass on a sample of `rows` cells, the activation after it */
        SlettLayer R = *L;
        R.in_h = R.out_h = 1;
        R.in_w = R.out_w = rows;
        R.inputs = rows * L->in_c;
        R.outputs = rows * L->out_c;
        R.activation = ACT_NONE;
        if (L->type == LAYER_ADD || L->type == LAYER_MULTIPLY || L->type == LAYER_CONCAT) {
            uint32_t units[SPINGALETT_MAX_INPUTS];
            for (uint32_t k = 0; k < L->input_count; k++) units[k] = rows * channels[k];
            spingalett_predict_combine(&R, x, units, channels, y, 1, mode);
        } else {
            spingalett_predict_layer(image, &R, prep, false, x[0], y, 1, w, mode);
        }
    }
    activate_rows(y, rows, L->out_c, pos, s->cells, L->activation);
}

/* Grows *buffer to hold floats. */
static bool reserve(float **buffer, size_t *have, size_t floats) {
    if (*have >= floats) return true;
    spingalett_aligned_free(*buffer);
    *buffer = (float *)spingalett_aligned_alloc(floats * sizeof(float));
    *have = *buffer ? floats : 0;
    return *buffer != NULL;
}

/* One pass over the first `rows` rows: every layer, the last on the rows that write logits. */
static bool session_pass(SpingalettSession *s, uint32_t rows, ComputeMode mode) {
    const SpingalettModel *model = s->model;
    const uint8_t *image = (const uint8_t *)model->image;
    float *arena = s->w->act[0];
    const uint64_t team = mode == COMPUTE_OPENMP ? rows * s->token_work / PASS_THREAD_WORK : 1u;
    s->team = team < 1 ? 1 : team > (uint64_t)s->w->threads ? s->w->threads : (int)team;
    uint32_t picked = 0;
    for (uint32_t r = 0; r < rows; r++)
        if (s->row_logits[r]) s->picked_position[picked] = s->row_position[r], s->picked[picked++] = r;
    for (uint32_t i = 0; i < model->layer_count; i++) {
        SlettLayer L;
        spingalett_slett_layer(image, i, &L);
        const float *x[SPINGALETT_MAX_INPUTS];
        uint32_t channels[SPINGALETT_MAX_INPUTS];
        for (uint32_t k = 0; k < L.input_count; k++) {
            uint32_t j = spingalett_slett_input(image, &L, k), h, wd, units;
            spingalett_slett_output_shape(image, j, &h, &wd, &units);
            channels[k] = units / (h * wd);
            x[k] = j == 0 ? NULL : arena + spingalett_slett_act_offset(image, j) / 4u * s->m;
        }
        if (i + 1 < model->layer_count) {
            session_layer(s, i, &L, x, channels, arena + L.act_offset / 4u * s->m, rows, s->row_position, mode);
            continue;
        }
        /* the logits: of the picked rows only (attention keeps every row's keys and values, an embedding
           reads every row's token) */
        if (!picked) break;
        const bool all = picked == rows || L.type == LAYER_ATTENTION || L.type == LAYER_EMBEDDING;
        const uint32_t n = all ? rows : picked;
        if (!reserve(&s->out, &s->out_floats, (size_t)n * L.out_c)) return false;
        if (!all) {
            size_t floats = 0;
            for (uint32_t k = 0; k < L.input_count; k++) floats += (size_t)picked * channels[k];
            if (!reserve(&s->gathered, &s->gathered_floats, floats)) return false;
            float *at = s->gathered;
            for (uint32_t k = 0; k < L.input_count; k++) {
                for (uint32_t r = 0; r < picked; r++)
                    memcpy(at + (size_t)r * channels[k], x[k] + (size_t)s->picked[r] * channels[k],
                           channels[k] * sizeof(float));
                x[k] = at;
                at += (size_t)picked * channels[k];
            }
        }
        session_layer(s, i, &L, x, channels, s->out, n, all ? s->row_position : s->picked_position, mode);
        for (uint32_t r = 0; r < picked; r++)
            memcpy(s->row_logits[s->picked[r]], s->out + (size_t)(all ? s->picked[r] : r) * L.out_c,
                   L.out_c * sizeof(float));
    }
    return true;
}

bool spingalett_session_feed(SpingalettSession *s, const SpingalettSessionFeed *feeds, uint32_t count) {
    if (!s || (count && !feeds)) {
        set_error(SPINGALETT_ERR_INVALID, "session feed: session or feeds is NULL");
        return false;
    }
    for (uint32_t f = 0; f < count; f++) {
        const SpingalettSessionFeed *e = &feeds[f];
        bool ok = e->sequence < s->sequences && (e->tokens || !e->count) && (e->count || !e->logits) &&
                  e->count <= s->max_length - s->length[e->sequence < s->sequences ? e->sequence : 0];
        for (uint32_t g = 0; g < f && ok; g++) ok = feeds[g].sequence != e->sequence;
        if (!ok) {
            set_error(SPINGALETT_ERR_INVALID, "session feed: a sequence out of range or named twice, more tokens than "
                      "max_length holds, or logits without tokens");
            return false;
        }
    }
    const ComputeMode mode = resolve_compute_mode();
    if (!session_workspace(s, mode)) {
        set_error(SPINGALETT_ERR_ALLOC, "session feed: workspace allocation failed");
        return false;
    }
    /* the rows in passes of up to capacity: the tokens of each feed in turn */
    for (uint32_t f = 0, t = 0;;) {
        uint32_t rows = 0;
        while (rows < s->capacity && f < count) {
            const SpingalettSessionFeed *e = &feeds[f];
            if (t >= e->count) {
                f++, t = 0;
                continue;
            }
            s->row_sequence[rows] = e->sequence;
            s->row_position[rows] = s->length[e->sequence] + t;
            s->row_token[rows] = (float)e->tokens[t];
            s->row_logits[rows] = !e->logits ? NULL : e->all ? e->logits + (size_t)t * s->vocabulary
                                : t + 1 == e->count ? e->logits : NULL;
            rows++, t++;
        }
        if (!rows) break;
        if (!session_pass(s, rows, mode)) {
            set_error(SPINGALETT_ERR_ALLOC, "session feed: allocation failed");
            return false;
        }
    }
    for (uint32_t f = 0; f < count; f++) s->length[feeds[f].sequence] += feeds[f].count;
    return true;
}

/* ------------------------------------------------------------------------- generation */

/* by probability, the larger first; equals by token */
static int by_probability(const void *a, const void *b) {
    const SpingalettCandidate *x = (const SpingalettCandidate *)a, *y = (const SpingalettCandidate *)b;
    if (x->p != y->p) return x->p > y->p ? -1 : 1;
    return x->token < y->token ? -1 : x->token > y->token;
}

static uint64_t splitmix64(uint64_t *state) {
    uint64_t z = (*state += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

uint32_t spingalett_next_token(const float *logits, uint32_t V, float temperature, uint32_t top_k, float top_p,
                               SpingalettCandidate *c, uint64_t *state) {
    uint32_t best = 0;
    for (uint32_t v = 1; v < V; v++)
        if (logits[v] > logits[best]) best = v;
    if (!(temperature > 0.0f)) return best;
    const double inv = 1.0 / temperature, top = logits[best];
    for (uint32_t v = 0; v < V; v++) c[v] = (SpingalettCandidate){(float)exp(((double)logits[v] - top) * inv), v};
    qsort(c, V, sizeof *c, by_probability);
    uint32_t kept = top_k && top_k < V ? top_k : V;
    double total = 0.0;
    for (uint32_t k = 0; k < kept; k++) total += c[k].p;
    if (top_p > 0.0f && top_p < 1.0f) {
        double sum = 0.0;
        for (uint32_t k = 0; k < kept; k++) {
            sum += c[k].p;
            if (sum >= top_p * total) {
                kept = k + 1u;
                total = sum;
                break;
            }
        }
    }
    const double u = (double)(splitmix64(state) >> 11) * (1.0 / 9007199254740992.0) * total;
    double sum = 0.0;
    for (uint32_t k = 0; k < kept; k++) {
        sum += c[k].p;
        if (u < sum) return c[k].token;
    }
    return c[kept - 1u].token;
}

uint32_t spingalett_session_generate_args(SpingalettSessionGenerateArgs args) {
    SpingalettSession *s = args.session;
    const uint32_t B = args.sequences ? args.sequences : 1u;
    bool ok = s && B <= s->sequences && args.prompts && args.prompt_lengths && (args.tokens || !args.count) &&
              (args.stop_tokens || !args.stop_count);
    for (uint32_t b = 0; ok && b < B; b++)
        ok = args.prompts[b] && args.prompt_lengths[b] && args.prompt_lengths[b] <= s->max_length - s->length[b];
    if (!ok) {
        set_error(SPINGALETT_ERR_INVALID, "session generate: no session, more sequences than it holds, a prompt "
                  "missing or longer than max_length leaves room for, or no room for the tokens");
        return 0;
    }
    const uint32_t V = s->vocabulary;
    float *logits = (float *)malloc((size_t)B * V * sizeof(float));
    SpingalettCandidate *c = (SpingalettCandidate *)malloc((size_t)V * sizeof *c);
    SpingalettSessionFeed *feeds = (SpingalettSessionFeed *)calloc(B, sizeof *feeds);
    uint64_t *state = (uint64_t *)malloc((size_t)B * sizeof(uint64_t));
    uint32_t *made = (uint32_t *)calloc(B, sizeof(uint32_t)), *next = (uint32_t *)malloc((size_t)B * sizeof(uint32_t));
    bool *going = (bool *)malloc((size_t)B * sizeof(bool));
    ok = logits && c && feeds && state && made && next && going;
    if (ok) {
        for (uint32_t b = 0; b < B; b++) {
            feeds[b] = (SpingalettSessionFeed){.sequence = b, .tokens = args.prompts[b], .count = args.prompt_lengths[b],
                                               .logits = logits + (size_t)b * V};
            state[b] = args.seed + b;
            going[b] = args.count > 0;
        }
        ok = spingalett_session_feed(s, feeds, B);
    } else {
        set_error(SPINGALETT_ERR_ALLOC, "session generate: out of memory");
    }
    uint32_t total = 0;
    /* a step: a token for each sequence still going, then those that go on fed together */
    for (bool any = true; ok && any;) {
        uint32_t n = 0;
        any = false;
        for (uint32_t b = 0; b < B; b++) {
            if (!going[b]) continue;
            const float *row = logits + (size_t)b * V;
            const uint32_t token = spingalett_next_token(row, V, args.temperature, args.top_k, args.top_p, c, &state[b]);
            args.tokens[(size_t)b * args.count + made[b]++] = token;
            total++;
            bool stop = made[b] == args.count || s->length[b] == s->max_length;
            for (uint32_t k = 0; k < args.stop_count && !stop; k++) stop = args.stop_tokens[k] == token;
            going[b] = !stop;
            any = any || going[b];
            if (s->length[b] == s->max_length) continue;
            next[b] = token;
            feeds[n++] = (SpingalettSessionFeed){.sequence = b, .tokens = &next[b], .count = 1,
                                                 .logits = stop ? NULL : logits + (size_t)b * V};
        }
        if (n) ok = spingalett_session_feed(s, feeds, n);
    }
    if (ok && args.made) memcpy(args.made, made, (size_t)B * sizeof(uint32_t));
    free(logits); free(c); free(feeds); free(state); free(made); free(next); free(going);
    return ok ? total : 0;
}
