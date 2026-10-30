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
#include "Spingalett.Blocks.h"
#include "Spingalett.Attention.h"
#include "Spingalett.Gpu.h"
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
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
    uint32_t window;                /* a sliding window: the keys of the last `window` positions (0: all) */
    float *keys;                    /* [sequences x kv] blocks of SPG_KEY_BLOCK keys (spingalett_key_store()), prepared */
    float *values;                  /* [sequences x max_length x kv x head] */
    size_t key_floats;              /* the keys of one sequence and head: max_length rounded up to blocks, x head */
    float *table;                   /* the rotary embeddings of max_length positions (cosines, then sines), or NULL */
    uint32_t rot;                   /* the values of a head they rotate */
    const float *gq, *gk;           /* the gammas of the queries' and keys' normalization, or NULL */
    float eps, scale;               /* its epsilon; the scores' scale */
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
    /* on a GPU from the model's image (Spingalett.GpuSession.c: models of block precisions), steps of gs_rows rows: the
       rows' token ids, the rows whose logits a step writes, and those logits */
    SpgGpuSession *gs;
    uint32_t gs_rows, *gs_tokens, *gs_pick;
    float *gs_logits;
    /* a model split between them (Spingalett.GpuSession.c): the first layer the CPU runs on small steps (0: not split);
       the positions of each sequence the GPU's caches of the later layers hold */
    uint32_t first;
    uint32_t *device_length;
#if !defined(SPINGALETT_RUNTIME)
    /* on a GPU (its compute modes, the full library): the model as a network there, run a step of rows at a time
       by a decoder that keeps the caches; the logits of a step's rows */
    NeuralNetwork *gpu_net;
    SpgGpuNet *gpu;
    SpgGpuDecoder *decoder;
    float *gpu_logits;
    uint32_t *row_token_id;
#endif
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
    if (L.token_offset != 0 || *cells != L.inputs) {
        snprintf(why, size, "its embedding reads part of the input");
        return false;
    }
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
    spingalett_gpu_session_free(s->gs);
    free(s->gs_tokens);
    free(s->gs_pick);
    free(s->gs_logits);
    free(s->device_length);
#if !defined(SPINGALETT_RUNTIME)
    spingalett_gpu_decoder_free(s->decoder);
    spingalett_gpu_net_free(s->gpu);
    if (s->gpu_net) spingalett_network_free(s->gpu_net);
    free(s->gpu_logits);
    free(s->row_token_id);
#endif
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

/* Rows a step of the GPU's session from the image (the logits of SPG_SESSION_LOGITS of them at most). */
#define GS_ROWS 512u

/* The session on the GPU of the compute mode straight from the model's image (Spingalett.GpuSession.c: weights of
   floats, float16, bfloat16 and GGUF's block types), its first layers alone (s->first, the CPU running the rest) where
   the budget, or the memory the device has free, does not hold them all; false (and the session elsewhere) when the
   GPU cannot run it, with *memory when memory is what it lacks. */
static bool session_image_gpu(SpingalettSession *s, uint64_t budget, bool *memory) {
    if (!spingalett_use_gpu()) return false;
    const char *why = NULL;
    s->gs_rows = s->capacity < GS_ROWS ? s->capacity : GS_ROWS;
    uint64_t room = spingalett_gpu_session_room();
    if (budget && budget < room) room = budget;
    /* (the driver may tell of more than it then gives: less each time) */
    for (int tries = 0; tries < 3 && !s->gs; tries++, room -= room / 4u) {
        s->gs = spingalett_gpu_session_new(s->model, s->sequences, s->max_length, s->gs_rows, room, &why);
        *memory = !s->gs && why && strstr(why, "memory");
        if (!*memory) break;
    }
    if (s->gs && spingalett_gpu_session_layers(s->gs) < s->layers) {
        s->first = spingalett_gpu_session_layers(s->gs);
        s->device_length = (uint32_t *)calloc(s->sequences, sizeof(uint32_t));
        if (!s->device_length) {
            spingalett_gpu_session_free(s->gs);
            s->gs = NULL;
            s->first = 0;
        }
    }
    s->gs_tokens = (uint32_t *)malloc((size_t)s->gs_rows * sizeof(uint32_t));
    s->gs_pick = (uint32_t *)malloc((size_t)s->gs_rows * sizeof(uint32_t));
    s->gs_logits = (float *)malloc((size_t)SPG_SESSION_LOGITS * s->vocabulary * sizeof(float));
    if (s->gs && s->gs_tokens && s->gs_pick && s->gs_logits) {
        spingalett_log(LOG_DEBUG, "session: on the GPU, from the model's image");
        if (s->first) spingalett_log(LOG_DEBUG, "session: the first %u of the model's %u layers on the GPU, the rest on the CPU",
                                     s->first, s->layers);
        return true;
    }
    spingalett_log(LOG_DEBUG, "session: not on the GPU from the model's image%s%s", why ? ": " : "", why ? why : "");
    spingalett_gpu_session_free(s->gs);
    free(s->gs_tokens);
    free(s->gs_pick);
    free(s->gs_logits);
    s->gs = NULL;
    s->gs_tokens = s->gs_pick = NULL;
    s->gs_logits = NULL;
    s->first = 0;
    return false;
}

static bool session_pass(SpingalettSession *s, uint32_t rows, ComputeMode mode, uint32_t first);
static void session_keep(SpingalettSession *s, const Cache *c, const float *x, uint32_t rows);

/* Rows from which a split session's step runs every layer but the last on the GPU: the rest of its layers' weights come
   over the bus, a cost the CPU's time for fewer rows does not repay (Qwen3-4B on an RTX 4050: 64 MB a block over the bus,
   a CPU's block about a tenth of a second for a prompt's 512 rows, a millisecond and a half a token). */
#define LARGE_STEP_ROWS 32u

/* What a large step's attention will read of its sequences and the GPU's caches of the later layers do not hold (tokens
   the CPU ran), from the host's: false when memory runs out. */
static bool session_fill_gpu(SpingalettSession *s, uint32_t rows) {
    for (uint32_t r = 0; r < rows; r++) {
        const uint32_t q = s->row_sequence[r], p = s->row_position[r], from = s->device_length[q];
        if (r && q == s->row_sequence[r - 1u] && p == s->row_position[r - 1u] + 1u) continue;
        if (from >= p) continue;
        const uint32_t count = p - from;
        for (uint32_t i = s->first; i < s->layers; i++) {
            const Cache *c = &s->caches[i];
            if (!c->keys) continue;
            const size_t floats = (size_t)count * c->kv * c->head;
            float *keys = (float *)malloc(2u * floats * sizeof(float));
            if (!keys) return false;
            float *values = keys + floats;
            for (uint32_t g = 0; g < c->kv; g++) {
                const float *blocks = c->keys + ((size_t)q * c->kv + g) * c->key_floats;
                for (uint32_t t = 0; t < count; t++) {
                    const uint32_t at = from + t;
                    const float *key = blocks + (size_t)(at / SPG_KEY_BLOCK) * c->head * SPG_KEY_BLOCK + at % SPG_KEY_BLOCK;
                    float *to = keys + ((size_t)g * count + t) * c->head;
                    for (uint32_t d = 0; d < c->head; d++) to[d] = key[(size_t)d * SPG_KEY_BLOCK];
                    memcpy(values + ((size_t)g * count + t) * c->head,
                           c->values + (size_t)q * s->max_length * c->kv * c->head + ((size_t)at * c->kv + g) * c->head,
                           c->head * sizeof(float));
                }
            }
            const bool ok = spingalett_gpu_session_fill(s->gs, i, q, from, count, keys, values);
            free(keys);
            if (!ok) return false;
        }
        s->device_length[q] = p;
    }
    return true;
}

/* A step of a split session: its first layers on the GPU, the rest on the CPU from their last one's rows; or, of a large
   step, every layer but the last on the GPU, the keys and values of the later attention layers kept here too. */
static bool session_step_split(SpingalettSession *s, uint32_t rows) {
    const uint8_t *image = (const uint8_t *)s->model->image;
    const bool large = rows >= LARGE_STEP_ROWS && spingalett_gpu_session_large(s->gs);
    const uint32_t from = large ? s->layers - 1u : s->first;
    float *values = s->w->act[0] + spingalett_slett_act_offset(image, from) / 4u * s->m;
    if (large && !session_fill_gpu(s, rows)) return false;
    if (!spingalett_gpu_session_step_split(s->gs, rows, s->gs_tokens, s->row_sequence, s->row_position, large, values))
        return false;
    if (large) {
        s->team = s->w->threads;
        for (uint32_t i = s->first; i < s->layers; i++)
            if (s->caches[i].keys) session_keep(s, &s->caches[i], spingalett_gpu_session_inputs(s->gs, i), rows);
        for (uint32_t r = 0; r < rows; r++)
            if (s->device_length[s->row_sequence[r]] <= s->row_position[r])
                s->device_length[s->row_sequence[r]] = s->row_position[r] + 1u;
    }
    return session_pass(s, rows, COMPUTE_OPENMP, from);
}

/* A feed on the GPU's session from the image: the feeds' tokens in steps of its rows, the logits copied where asked; with
   the model's first layers alone there, every row's values go on into the CPU's pass. */
static bool session_feed_gs(SpingalettSession *s, const SpingalettSessionFeed *feeds, uint32_t count) {
    if (s->first && !session_workspace(s, COMPUTE_OPENMP)) {
        set_error(SPINGALETT_ERR_ALLOC, "session feed: workspace allocation failed");
        return false;
    }
    for (uint32_t f = 0, t = 0;;) {
        uint32_t rows = 0, picked = 0;
        while (rows < s->gs_rows && (s->first || picked < SPG_SESSION_LOGITS) && f < count) {
            const SpingalettSessionFeed *e = &feeds[f];
            if (t >= e->count) {
                f++, t = 0;
                continue;
            }
            s->row_sequence[rows] = e->sequence;
            s->row_position[rows] = s->length[e->sequence] + t;
            s->gs_tokens[rows] = e->tokens[t];
            s->row_logits[rows] = !e->logits ? NULL : e->all ? e->logits + (size_t)t * s->vocabulary
                                : t + 1 == e->count ? e->logits : NULL;
            if (s->row_logits[rows]) s->gs_pick[picked++] = rows;
            rows++, t++;
        }
        if (!rows) break;
        if (s->first) {
            if (!session_step_split(s, rows)) {
                set_error(SPINGALETT_ERR_ALLOC, "session feed: the GPU failed");
                return false;
            }
            continue;
        }
        if (!spingalett_gpu_session_step(s->gs, rows, s->gs_tokens, s->row_sequence, s->row_position, picked, s->gs_pick,
                                         s->gs_logits)) {
            set_error(SPINGALETT_ERR_ALLOC, "session feed: the GPU failed");
            return false;
        }
        for (uint32_t k = 0; k < picked; k++)
            memcpy(s->row_logits[s->gs_pick[k]], s->gs_logits + (size_t)k * s->vocabulary, s->vocabulary * sizeof(float));
    }
    for (uint32_t f = 0; f < count; f++) s->length[feeds[f].sequence] += feeds[f].count;
    return true;
}

#if !defined(SPINGALETT_RUNTIME)
/* The session on the GPU of the compute mode: the model's image made a network there (its weights as floats,
   the products in the GPU's precision), a decoder with the caches of every sequence, which multiplies an integer
   model's codes in its linear layers (weight-only: spingalett_gpu_decoder_quantize()); false (and the session on the
   CPU) when the GPU cannot run it. */
static bool session_gpu(SpingalettSession *s) {
    if (!spingalett_use_gpu()) return false;
    const char *why = NULL;
    spingalett_log_quiet(1);           /* (the network's layers are the model's: not logged again) */
    s->gpu_net = spingalett_load_from_memory(s->model->image, s->model->image_size);
    spingalett_log_quiet(-1);
    if (!s->gpu_net || !spingalett_gpu_supports(s->gpu_net, &why)) goto fail;
    s->gpu = spingalett_gpu_net_create(s->gpu_net, s->m, NULL);
    if (!s->gpu || !spingalett_gpu_upload(s->gpu)) goto fail;
    s->decoder = spingalett_gpu_decoder_new(s->gpu, s->sequences, s->max_length, &why);
    if (!s->decoder) goto fail;
    const int coded = spingalett_gpu_decoder_quantize(s->decoder, (const uint8_t *)s->model->image);
    if (coded > 0) spingalett_log(LOG_DEBUG, "session: on the GPU, %d linear layers multiply the model's codes", coded);
    else if (coded < 0) spingalett_log(LOG_DEBUG, "session: on the GPU, the model's weights as floats (no memory for its codes)");
    const uint32_t rows = spingalett_gpu_decoder_rows(s->decoder);
    s->gpu_logits = (float *)malloc((size_t)rows * s->vocabulary * sizeof(float));
    s->row_token_id = (uint32_t *)malloc((size_t)rows * sizeof(uint32_t));
    if (!s->gpu_logits || !s->row_token_id) goto fail;
    s->capacity = rows < s->capacity ? rows : s->capacity;
    return true;
fail:
    spingalett_log(LOG_DEBUG, "session: on the CPU (the GPU cannot run the model%s%s)", why ? ": " : "", why ? why : "");
    spingalett_gpu_decoder_free(s->decoder);
    spingalett_gpu_net_free(s->gpu);
    if (s->gpu_net) spingalett_network_free(s->gpu_net);
    free(s->gpu_logits);
    free(s->row_token_id);
    s->decoder = NULL, s->gpu = NULL, s->gpu_net = NULL, s->gpu_logits = NULL, s->row_token_id = NULL;
    return false;
}
#endif

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
    if ((uint64_t)(sequences / cells + 1u) * cells > UINT32_MAX) {     /* (a pass's rows count in 32 bits) */
        set_error(SPINGALETT_ERR_INVALID, "session: too many sequences");
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
    /* (a model the device's memory does not hold but in floats holds even less: on the CPU) */
    bool memory = false;
    if (ok && session_image_gpu(s, options ? options->gpu_memory : 0u, &memory) && !s->first) return s;
#if !defined(SPINGALETT_RUNTIME)
    if (ok && !s->gs && !memory && session_gpu(s)) return s;
#endif
    const uint8_t *image = (const uint8_t *)model->image;
    for (uint32_t i = s->first; ok && i < s->layers; i++) {
        SlettLayer L;
        spingalett_slett_layer(image, i, &L);
        if (L.type == LAYER_CONV2D) s->token_work += (uint64_t)L.rows * L.row_len;
        if (L.type != LAYER_ATTENTION) continue;
        Cache *c = &s->caches[i];
        SlettAttention a;
        spingalett_engine_attention_of(image, &L, NULL, &a);
        c->heads = L.heads, c->kv = L.kv_heads, c->head = L.out_c / L.heads, c->window = L.window;
        c->rot = a.rot, c->gq = a.gq, c->gk = a.gk, c->eps = a.eps, c->scale = a.scale;
        const size_t floats = (size_t)sequences * max_length * c->kv * c->head;
        c->key_floats = (size_t)(max_length + SPG_KEY_BLOCK - 1u) / SPG_KEY_BLOCK * SPG_KEY_BLOCK * c->head;
        c->keys = (float *)spingalett_aligned_calloc((size_t)sequences * c->kv * c->key_floats, sizeof(float));
        c->values = (float *)spingalett_aligned_alloc(floats * sizeof(float));
        ok = c->keys && c->values;
        if (ok && L.theta > 0.0f) {
            c->table = (float *)spingalett_aligned_alloc((size_t)max_length * c->head * sizeof(float));
            ok = c->table != NULL;
            if (ok)
                spingalett_engine_rope_table(max_length, c->rot, L.theta,
                                             L.freqs ? (const float *)(const void *)(image + L.scales) : NULL, c->table);
        }
        if (c->head > s->query_floats) s->query_floats = c->head;
    }
    if (ok) {
        s->prepared = spingalett_model_layers(model, false, true);
        ok = s->prepared && session_workspace(s, s->first ? COMPUTE_OPENMP : resolve_compute_mode());
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
    if (s->device_length && s->device_length[sequence] > length) s->device_length[sequence] = length;
    return true;
}

bool spingalett_session_copy(SpingalettSession *s, uint32_t from, uint32_t to) {
    if (!s || from >= s->sequences || to >= s->sequences) {
        set_error(SPINGALETT_ERR_INVALID, "session copy: no such sequence");
        return false;
    }
    if (from == to) return true;
    if (s->gs) {
        if (!spingalett_gpu_session_copy(s->gs, from, to, s->length[from])) {
            set_error(SPINGALETT_ERR_ALLOC, "session copy: the GPU failed");
            return false;
        }
        if (!s->first) {
            s->length[to] = s->length[from];
            return true;
        }
        s->device_length[to] = s->device_length[from];
    }
#if !defined(SPINGALETT_RUNTIME)
    if (s->decoder) {
        if (!spingalett_gpu_decoder_copy(s->decoder, from, to, s->length[from])) {
            set_error(SPINGALETT_ERR_ALLOC, "session copy: the GPU failed");
            return false;
        }
        s->length[to] = s->length[from];
        return true;
    }
#endif
    for (uint32_t i = 0; i < s->layers; i++) {
        const Cache *c = &s->caches[i];
        if (!c->keys) continue;
        const size_t per = (size_t)c->kv * c->head, sequence = (size_t)s->max_length * per;
        const size_t blocks = (s->length[from] + SPG_KEY_BLOCK - 1u) / SPG_KEY_BLOCK * SPG_KEY_BLOCK * c->head;
        for (uint32_t g = 0; g < c->kv; g++)
            memcpy(c->keys + ((size_t)to * c->kv + g) * c->key_floats, c->keys + ((size_t)from * c->kv + g) * c->key_floats,
                   blocks * sizeof(float));
        memcpy(c->values + to * sequence, c->values + from * sequence, s->length[from] * per * sizeof(float));
    }
    s->length[to] = s->length[from];
    return true;
}

/* ------------------------------------------------------------------------- passes */

/* Values of row-wise passes (normalizations, sums and products of layers, activations) from which a pass's rows
   are split between its team: a prompt's are millions (SiLU's exponentials among them), a token's few. */
#define ROWS_PARALLEL_VALUES ((uint64_t)1 << 15)

/* Whether the team splits a row-wise pass of `rows` rows of C values, and the first row of piece t of `pieces`. */
static bool rows_split(const SpingalettSession *s, uint32_t rows, uint32_t C) {
    return s->team > 1 && rows > 1 && (uint64_t)rows * C >= ROWS_PARALLEL_VALUES;
}
static uint32_t piece_row(uint32_t rows, int64_t pieces, int64_t t) {
    return (uint32_t)((uint64_t)rows * (uint64_t)t / (uint64_t)pieces);
}

/* The activation of `rows` rows of C values at positions pos[], as batched prediction applies it to a
   whole sample of `cells` cells. Element-wise activations give a value the same wherever it lies; sigmoid
   and tanh take a vector path for the groups of 8 values the sample holds whole from its first on, and
   the scalar one for the rest (apply_activation_bulk()), so each value takes the path of its place. */
static void activate_some(float *y, uint32_t rows, uint32_t C, const uint32_t *pos, uint32_t cells,
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

/* activate_some() on the rows, split between the team when they are many. */
static void activate_rows(const SpingalettSession *s, float *y, uint32_t rows, uint32_t C, const uint32_t *pos,
                          ActivationFunction act) {
    if (act == ACT_NONE) return;
    const bool split = rows_split(s, rows, C);
    const int64_t pieces = split ? s->team : 1;
    SPINGALETT_PARALLEL_FOR_THREADS(s->team, split,
        for (int64_t t = 0; t < pieces; t++) {
            const uint32_t r0 = piece_row(rows, pieces, t), r1 = piece_row(rows, pieces, t + 1);
            activate_some(y + (size_t)r0 * C, r1 - r0, C, pos + r0, s->cells, act);
        }
    );
}

/* A normalization, sum or product of layers on rows split between the team: each row computed as the engine
   computes a cell (spingalett_predict_layer() and spingalett_predict_combine() on a sample of the rows). */
static void split_rows(const SpingalettSession *s, const uint8_t *image, const SlettLayer *L, const float *const *x,
                       const uint32_t *channels, float *y, uint32_t rows) {
    const int64_t pieces = s->team;
    const float *gamma = (const float *)(const void *)(image + L->weights);
    const float *beta = (const float *)(const void *)(image + L->biases);
    SPINGALETT_PARALLEL_FOR_THREADS(s->team, true,
        for (int64_t t = 0; t < pieces; t++) {
            const uint32_t r0 = piece_row(rows, pieces, t), n = piece_row(rows, pieces, t + 1) - r0;
            float *ys = y + (size_t)r0 * L->out_c;
            if (L->type == LAYER_RMS_NORM) {
                spingalett_engine_rms_norm(x[0] + (size_t)r0 * L->in_c, n, L->out_c, gamma, L->eps, ys, NULL);
            } else if (L->type == LAYER_LAYER_NORM) {
                spingalett_engine_layer_norm(x[0] + (size_t)r0 * L->in_c, n, L->out_c, gamma, beta, L->eps, ys, NULL);
            } else {
                const float *xs[SPINGALETT_MAX_INPUTS];
                for (uint32_t k = 0; k < L->input_count; k++) xs[k] = x[k] + (size_t)r0 * channels[k];
                if (L->type == LAYER_ADD) spingalett_engine_add(xs, L->input_count, ys, n * L->out_c);
                else spingalett_engine_multiply(xs, L->input_count, ys, n * L->out_c);
            }
        }
    );
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

/* The keys a query at position p attends to, and the work of one of its heads in units of a key (a query's own
   costs as many as 16 keys). */
static uint32_t attended(const Cache *c, uint32_t p) {
    return c->window && p + 1u > c->window ? c->window : p + 1u;
}
#define QUERY_WORK 16u

/* The first query head (row r head h: item r heads + h) of piece t of `pieces` pieces of about equal work: the rows of
   a prompt attend to more keys the later they stand, so that pieces of as many heads would leave the threads of the
   first rows waiting for those of the last. */
static int64_t attention_piece(const SpingalettSession *s, const Cache *c, uint32_t rows, uint64_t total, int64_t pieces,
                               int64_t t) {
    if (t >= pieces) return (int64_t)rows * c->heads;
    const uint64_t target = total * (uint64_t)t / (uint64_t)pieces;
    uint64_t sum = 0;
    for (uint32_t r = 0; r < rows; r++) {
        const uint64_t each = attended(c, s->row_position[r]) + QUERY_WORK, row = each * c->heads;
        if (sum + row > target) return (int64_t)r * c->heads + (int64_t)((target - sum + each - 1u) / each);
        sum += row;
    }
    return (int64_t)rows * c->heads;
}

/* The keys and values of `rows` rows of an attention layer's inputs kept (each row alone: split between the team when
   the rows are many). */
static void session_keep(SpingalettSession *s, const Cache *c, const float *x, uint32_t rows) {
    const uint32_t heads = c->heads, kv = c->kv, head = c->head, half = c->rot / 2u;
    const uint32_t C = (heads + 2u * kv) * head, rot = c->table ? c->rot : 0u;
    const size_t per = (size_t)kv * head, sequence = (size_t)s->max_length * per;
    const float *sines = c->table ? c->table + (size_t)s->max_length * half : NULL;
    const bool prepared = c->table || c->gk;
    const bool split = rows_split(s, rows, C);
    const int64_t pieces = split ? s->team : 1;
    SPINGALETT_PARALLEL_FOR_THREADS(s->team, split,
        for (int64_t t = 0; t < pieces; t++) {
            /* (a prepared key, before it goes into its block) */
            float *key = s->queries + (size_t)spingalett_thread_num() * s->query_floats;
            const uint32_t r0 = piece_row(rows, pieces, t), r1 = piece_row(rows, pieces, t + 1);
            for (uint32_t r = r0; r < r1; r++) {
                const uint32_t p = s->row_position[r];
                float *values = c->values + s->row_sequence[r] * sequence + p * per;
                for (uint32_t g = 0; g < kv; g++) {
                    const float *k = x + (size_t)r * C + (size_t)(heads + g) * head;
                    if (prepared)
                        spingalett_engine_head(k, key, head, c->gk, c->eps, rot, c->table ? c->table + (size_t)p * half : NULL,
                                               sines ? sines + (size_t)p * half : NULL);
                    spingalett_key_store(c->keys + ((size_t)s->row_sequence[r] * kv + g) * c->key_floats, head, p,
                                         prepared ? key : k);
                    memcpy(values + (size_t)g * head, x + (size_t)r * C + (size_t)(heads + kv + g) * head, head * sizeof(float));
                }
            }
        }
    );
}

/* Attention on `rows` rows (row r of the pass, or the picked row r): their keys and values kept first,
   then each query over the keys of its sequence up to its own position (with a window, the last `window` of
   them), the query heads split between the threads in pieces of equal work. Every key is live: tokens are 0 or
   more, so that a layer ignoring padding finds none. */
static void session_attention(SpingalettSession *s, const Cache *c, const float *x, float *y, uint32_t rows,
                              ComputeMode mode) {
    const uint32_t heads = c->heads, kv = c->kv, head = c->head, half = c->rot / 2u, group = heads / kv;
    const uint32_t C = (heads + 2u * kv) * head, out_c = heads * head, rot = c->table ? c->rot : 0u;
    const size_t per = (size_t)kv * head, sequence = (size_t)s->max_length * per;
    const float *sines = c->table ? c->table + (size_t)s->max_length * half : NULL;
    const bool prepared = c->table || c->gk;
    session_keep(s, c, x, rows);
    const float scale = c->scale;
    uint64_t total = 0;
    for (uint32_t r = 0; r < rows; r++) total += (uint64_t)(attended(c, s->row_position[r]) + QUERY_WORK) * heads;
    const bool threads = s->team > 1 && spingalett_use_omp(mode, total * head * 2u);
    const int64_t pieces = threads ? s->team : 1;
    SPINGALETT_PARALLEL_FOR_THREADS(s->team, threads,
        for (int64_t t = 0; t < pieces; t++) {
            const int64_t end = attention_piece(s, c, rows, total, pieces, t + 1);
            for (int64_t it = attention_piece(s, c, rows, total, pieces, t); it < end; it++) {
                const uint32_t r = (uint32_t)(it / heads), h = (uint32_t)(it % heads), g = h / group;
                const uint32_t p = s->row_position[r];
                const float *q = x + (size_t)r * C + (size_t)h * head;
                if (prepared) {
                    float *ready = s->queries + (size_t)spingalett_thread_num() * s->query_floats;
                    spingalett_engine_head(q, ready, head, c->gq, c->eps, rot,
                                           c->table ? c->table + (size_t)p * half : NULL,
                                           sines ? sines + (size_t)p * half : NULL);
                    q = ready;
                }
                const uint32_t keys = attended(c, p);
                spingalett_attend_blocks(q, c->keys + ((size_t)s->row_sequence[r] * kv + g) * c->key_floats,
                                         c->values + s->row_sequence[r] * sequence + (size_t)g * head, per, p + 1u - keys,
                                         keys, head, scale, y + (size_t)r * out_c + (size_t)h * head);
            }
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
    } else if (slett_block_layer(L)) {
        /* a linear layer of a block precision: its bias in, each row quantized alone */
        spingalett_block_layer(image, L, x[0], y, rows, w->blockq, s->team);
    } else if (L->type == LAYER_CONV2D && rows <= INT_ROWS) {
        session_int_linear(s, L, prep, x[0], y, rows, mode);
    } else if ((L->type == LAYER_RMS_NORM || L->type == LAYER_LAYER_NORM || L->type == LAYER_ADD ||
                L->type == LAYER_MULTIPLY) && rows_split(s, rows, L->out_c)) {
        split_rows(s, image, L, x, channels, y, rows);
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
            spingalett_predict_layer(image, &R, prep, false, NULL, 0, x[0], NULL, y, 1, w, mode);
        }
    }
    activate_rows(s, y, rows, L->out_c, pos, L->activation);
}

/* Grows *buffer to hold floats. */
static bool reserve(float **buffer, size_t *have, size_t floats) {
    if (*have >= floats) return true;
    spingalett_aligned_free(*buffer);
    *buffer = (float *)spingalett_aligned_alloc(floats * sizeof(float));
    *have = *buffer ? floats : 0;
    return *buffer != NULL;
}

/* One pass over the first `rows` rows: every layer from `first` on, the last on the rows that write logits. */
static bool session_pass(SpingalettSession *s, uint32_t rows, ComputeMode mode, uint32_t first) {
    const SpingalettModel *model = s->model;
    const uint8_t *image = (const uint8_t *)model->image;
    float *arena = s->w->act[0];
    const uint64_t team = mode == COMPUTE_OPENMP ? rows * s->token_work / PASS_THREAD_WORK : 1u;
    s->team = team < 1 ? 1 : team > (uint64_t)s->w->threads ? s->w->threads : (int)team;
    uint32_t picked = 0;
    for (uint32_t r = 0; r < rows; r++)
        if (s->row_logits[r]) s->picked_position[picked] = s->row_position[r], s->picked[picked++] = r;
    for (uint32_t i = first; i < model->layer_count; i++) {
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

#if !defined(SPINGALETT_RUNTIME)
/* A feed on the GPU: the feeds' tokens in steps of the decoder's rows, every row's logits copied where asked. */
static bool session_feed_gpu(SpingalettSession *s, const SpingalettSessionFeed *feeds, uint32_t count) {
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
            s->row_token_id[rows] = e->tokens[t];
            s->row_logits[rows] = !e->logits ? NULL : e->all ? e->logits + (size_t)t * s->vocabulary
                                : t + 1 == e->count ? e->logits : NULL;
            rows++, t++;
        }
        if (!rows) break;
        bool any = false;
        for (uint32_t r = 0; r < rows; r++) any = any || s->row_logits[r];
        if (!spingalett_gpu_decoder_step(s->decoder, rows, s->row_token_id, s->row_sequence, s->row_position,
                                         any ? s->gpu_logits : NULL)) {
            set_error(SPINGALETT_ERR_ALLOC, "session feed: the GPU failed");
            return false;
        }
        for (uint32_t r = 0; r < rows; r++)
            if (s->row_logits[r])
                memcpy(s->row_logits[r], s->gpu_logits + (size_t)r * s->vocabulary, s->vocabulary * sizeof(float));
    }
    for (uint32_t f = 0; f < count; f++) s->length[feeds[f].sequence] += feeds[f].count;
    return true;
}
#endif

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
    if (s->gs) return session_feed_gs(s, feeds, count);
#if !defined(SPINGALETT_RUNTIME)
    if (s->decoder) return session_feed_gpu(s, feeds, count);
#endif
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
        if (!session_pass(s, rows, mode, 0)) {
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

/* Sorts the next part of the candidates c[sorted .. V): those of probability at least *threshold (lowered 256 times
   until some are), or all that are left once it is past the smallest normal float. Every candidate of a part is more
   likely than every one after it, so the parts in their order are the order of all. The new end of the sorted part. */
static uint32_t sort_more(SpingalettCandidate *c, uint32_t sorted, uint32_t V, float *threshold) {
    while (sorted < V) {
        uint32_t m = sorted;
        if (*threshold >= FLT_MIN) {
            for (uint32_t v = sorted; v < V; v++)
                if (c[v].p >= *threshold) {
                    const SpingalettCandidate t = c[m];
                    c[m++] = c[v];
                    c[v] = t;
                }
            *threshold *= 1.0f / 256.0f;
        } else {
            m = V;
        }
        if (m > sorted) {
            qsort(c + sorted, m - sorted, sizeof *c, by_probability);
            return m;
        }
    }
    return V;
}

uint32_t spingalett_next_token(const float *logits, uint32_t V, const SpgDraw *d, SpingalettCandidate *c,
                               uint64_t *state) {
    uint32_t best = 0;
    for (uint32_t v = 1; v < V; v++)
        if (logits[v] > logits[best]) best = v;
    if (!(d->temperature > 0.0f)) return best;
    const double inv = 1.0 / d->temperature, top = logits[best];
    double total = 0.0;
    for (uint32_t v = 0; v < V; v++) {
        c[v] = (SpingalettCandidate){(float)exp(((double)logits[v] - top) * inv), v};
        total += c[v].p;
    }
    /* the candidates in order of probability as far as the draw needs them, sorted a part at a time from the most
       likely (of probability 1) down */
    uint32_t sorted = 0, kept = V;
    float threshold = 1.0f / 256.0f;
    if (d->top_k && d->top_k < V) {
        kept = d->top_k;
        while (sorted < kept) sorted = sort_more(c, sorted, V, &threshold);
        total = 0.0;
        for (uint32_t k = 0; k < kept; k++) total += c[k].p;
    }
    if (d->top_p > 0.0f && d->top_p < 1.0f) {
        double sum = 0.0;
        for (uint32_t k = 0; k < kept; k++) {
            while (k >= sorted) sorted = sort_more(c, sorted, V, &threshold);
            sum += c[k].p;
            if (sum >= d->top_p * total) {
                kept = k + 1u;
                total = sum;
                break;
            }
        }
    }
    if (d->min_p > 0.0f) {
        /* the most likely has probability 1: the others need min_p */
        uint32_t k = 1;
        for (; k < kept; k++) {
            while (k >= sorted) sorted = sort_more(c, sorted, V, &threshold);
            if (c[k].p < d->min_p) break;
        }
        if (k < kept) {
            kept = k;
            total = 0.0;
            for (uint32_t j = 0; j < kept; j++) total += c[j].p;
        }
    }
    const double u = (double)(splitmix64(state) >> 11) * (1.0 / 9007199254740992.0) * total;
    double sum = 0.0;
    for (uint32_t k = 0; k < kept; k++) {
        while (k >= sorted) sorted = sort_more(c, sorted, V, &threshold);
        sum += c[k].p;
        if (u < sum) return c[k].token;
    }
    while (sorted < kept) sorted = sort_more(c, sorted, V, &threshold);
    return c[kept - 1u].token;
}

static int by_token(const void *a, const void *b) {
    const uint32_t x = *(const uint32_t *)a, y = *(const uint32_t *)b;
    return x < y ? -1 : x > y;
}

uint32_t spingalett_penalty_window(const SpingalettSampling *s) {
    if (!s || ((s->repetition_penalty == 0.0f || s->repetition_penalty == 1.0f) && s->frequency_penalty == 0.0f &&
               s->presence_penalty == 0.0f))
        return 0;
    return s->penalty_window ? s->penalty_window : 64u;
}

void spingalett_penalize(float *logits, uint32_t V, const SpingalettSampling *s, const uint32_t *a, uint32_t na,
                         const uint32_t *b, uint32_t nb, uint32_t *scratch) {
    const uint32_t window = spingalett_penalty_window(s);
    if (!window) return;
    /* the window's last tokens, sorted so that each token's occurrences lie together */
    const uint64_t all = (uint64_t)na + nb;
    const uint32_t n = all < window ? (uint32_t)all : window;
    for (uint32_t k = 0; k < n; k++) {
        const uint64_t at = all - n + k;
        scratch[k] = at < na ? a[at] : b[at - na];
    }
    qsort(scratch, n, sizeof *scratch, by_token);
    const float rep = s->repetition_penalty == 0.0f ? 1.0f : s->repetition_penalty;
    for (uint32_t k = 0; k < n;) {
        uint32_t e = k + 1u;
        while (e < n && scratch[e] == scratch[k]) e++;
        const uint32_t t = scratch[k];
        if (t < V) {
            float l = logits[t];
            if (rep != 1.0f) l = l <= 0.0f ? l * rep : l / rep;
            logits[t] = l - (float)(e - k) * s->frequency_penalty - s->presence_penalty;
        }
        k = e;
    }
}

bool spingalett_stop_string(const SpingalettSampling *s, const uint32_t *generated, uint32_t made) {
    if (!s || !s->stop_string_count || !s->stop_strings || !s->tokenizer || !made) return false;
    size_t longest = 0;
    for (uint32_t k = 0; k < s->stop_string_count; k++)
        if (s->stop_strings[k] && strlen(s->stop_strings[k]) > longest) longest = strlen(s->stop_strings[k]);
    if (!longest) return false;
    /* the text of the last tokens (each gives a byte at least, and two more cover a character they split), with and
       without the newest one: a stop string the newest completed is in the first and not in the second */
    const uint32_t tail = (uint64_t)made < (uint64_t)longest + 3u ? made : (uint32_t)longest + 3u;
    const uint32_t *from = generated + made - tail;
    bool stop = false;
    char *with = NULL, *without = NULL;
    for (int pass = 0; pass < 2; pass++) {
        const size_t count = pass ? tail - 1u : tail;
        int64_t n = spingalett_detokenize(.tokenizer = s->tokenizer, .tokens = from, .count = count);
        char *text = n >= 0 ? (char *)malloc((size_t)n + 1u) : NULL;
        if (text && spingalett_detokenize(.tokenizer = s->tokenizer, .tokens = from, .count = count, .text = text,
                                          .capacity = (size_t)n + 1u) != n) {
            free(text);
            text = NULL;
        }
        if (pass) without = text;
        else with = text;
    }
    for (uint32_t k = 0; with && k < s->stop_string_count && !stop; k++) {
        const char *x = s->stop_strings[k];
        stop = x && *x && strstr(with, x) && !(without && strstr(without, x));
    }
    free(with);
    free(without);
    return stop;
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
    /* the penalties' window of tokens, which may reach every token of a prompt and those generated */
    uint32_t span = spingalett_penalty_window(args.sampling), longest = 0;
    for (uint32_t b = 0; b < B; b++) longest = args.prompt_lengths[b] > longest ? args.prompt_lengths[b] : longest;
    if ((uint64_t)span > (uint64_t)longest + args.count) span = longest + args.count;
    uint32_t *window = (uint32_t *)malloc(((size_t)span + 1u) * sizeof(uint32_t));
    const SpgDraw draw = {args.temperature, args.top_k, args.top_p, args.sampling ? args.sampling->min_p : 0.0f};
    ok = logits && c && feeds && state && made && next && going && window;
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
            float *row = logits + (size_t)b * V;
            uint32_t *generated = args.tokens + (size_t)b * args.count;
            spingalett_penalize(row, V, args.sampling, args.prompts[b], args.prompt_lengths[b], generated, made[b], window);
            const uint32_t token = spingalett_next_token(row, V, &draw, c, &state[b]);
            generated[made[b]++] = token;
            total++;
            bool stop = made[b] == args.count || s->length[b] == s->max_length;
            for (uint32_t k = 0; k < args.stop_count && !stop; k++) stop = args.stop_tokens[k] == token;
            stop = stop || spingalett_stop_string(args.sampling, generated, made[b]);
            if (args.sampling && args.sampling->on_token && !args.sampling->on_token(args.sampling->user, b, token))
                stop = true;
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
    free(logits); free(c); free(feeds); free(state); free(made); free(next); free(going); free(window);
    return ok ? total : 0;
}
