/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/*
 * .slett images written layer by layer (Spingalett.Image.h), and the plan by which the outputs of an inference
 * pass share memory, which the writers of networks (Spingalett.Serialize.c) and of these images share. The
 * images are what spingalett_save_deployment() would write of the same network, laid out in the same order:
 * the header, the layer table, then each layer's input list, weights, row scales and biases.
 */

#include "Spingalett.Image.h"
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------------- memory plan */

typedef struct {
    uint64_t offset, size;
    uint32_t last;              /* the last step that reads it */
} LiveBuffer;

uint64_t spingalett_plan_buffers(uint32_t outputs, const uint64_t *sizes, uint32_t count, const uint32_t *steps,
                                 const uint32_t *const *reads, const uint32_t *read_count, uint64_t align,
                                 uint64_t *offsets) {
    uint32_t *last = (uint32_t *)malloc(((size_t)outputs + 1) * sizeof(uint32_t));
    LiveBuffer *live = (LiveBuffer *)malloc(((size_t)outputs + 1) * sizeof(LiveBuffer));
    if (!last || !live) {
        free(last);
        free(live);
        return UINT64_MAX;
    }
    for (uint32_t t = 0; t < outputs; t++) {
        last[t] = 0;
        offsets[t] = UINT64_MAX;
    }
    for (uint32_t s = 0; s < count; s++) {     /* readers come after their producer */
        last[steps[s]] = s;                    /* an output nothing reads lives for its own step */
        for (uint32_t k = 0; k < read_count[s]; k++) last[reads[s][k]] = s;
    }

    uint64_t total = 0;
    uint32_t n = 0;                            /* live buffers, by increasing offset */
    for (uint32_t s = 0; s < count; s++) {
        uint32_t keep = 0;
        for (uint32_t i = 0; i < n; i++)
            if (live[i].last >= s) live[keep++] = live[i];
        n = keep;
        uint32_t t = steps[s];
        if (sizes[t] == 0) continue;
        uint64_t size = (sizes[t] + align - 1) / align * align, at = 0;
        uint32_t pos = 0;
        /* the lowest gap that holds it */
        for (; pos < n; pos++) {
            if (at + size <= live[pos].offset) break;
            uint64_t end = live[pos].offset + live[pos].size;
            if (end > at) at = end;
        }
        memmove(live + pos + 1, live + pos, (size_t)(n - pos) * sizeof(LiveBuffer));
        live[pos] = (LiveBuffer){at, size, last[t]};
        n++;
        offsets[t] = at;
        if (at + size > total) total = at + size;
    }
    free(last);
    free(live);
    return total;
}

/* ------------------------------------------------------------------------- element types */

size_t spingalett_dtype_size(int dtype) {
    switch (dtype) {
        case SPG_DTYPE_F32: case SPG_DTYPE_I32: return 4u;
        case SPG_DTYPE_F16: case SPG_DTYPE_BF16: return 2u;
        case SPG_DTYPE_F64: case SPG_DTYPE_I64: return 8u;
        default: return 0u;
    }
}

void spingalett_decode(float *dst, const uint8_t *src, int dtype, size_t n) {
    switch (dtype) {
        case SPG_DTYPE_F32:
            memcpy(dst, src, n * sizeof(float));
            break;
        case SPG_DTYPE_F16:
            for (size_t i = 0; i < n; i++) dst[i] = spingalett_fp16_to_float(slett_get16(src + 2u * i));
            break;
        case SPG_DTYPE_BF16:
            for (size_t i = 0; i < n; i++) dst[i] = spingalett_bf16_to_float(slett_get16(src + 2u * i));
            break;
        case SPG_DTYPE_F64:
            for (size_t i = 0; i < n; i++) { double v; memcpy(&v, src + 8u * i, 8); dst[i] = (float)v; }
            break;
        case SPG_DTYPE_I32:
            for (size_t i = 0; i < n; i++) { int32_t v; memcpy(&v, src + 4u * i, 4); dst[i] = (float)v; }
            break;
        default:
            for (size_t i = 0; i < n; i++) { int64_t v; memcpy(&v, src + 8u * i, 8); dst[i] = (float)v; }
            break;
    }
}

/* ------------------------------------------------------------------------- the builder */

typedef struct {
    SpgLayerSpec spec;
    uint32_t in_c, out_c, units;    /* the first input's channels, the output's, its units */
    uint32_t rows, row_len;         /* the weights' */
    uint64_t biases;                /* floats of biases (0: none) */
    uint64_t gammas, freqs;         /* an attention's floats of its normalization's gammas and its frequencies */
    uint64_t list, weights, scales, bias_at, act;   /* offsets in the image */
} BuiltLayer;

struct SpgImageBuilder {
    uint32_t cells;
    LossFunction loss;
    BuiltLayer *layers;             /* [count + 1]: layers[0] the input */
    uint32_t count, capacity;
};

SpgImageBuilder *spingalett_image_new(uint32_t cells, LossFunction loss) {
    if (cells == 0 || cells > SLETT_MAX_EXTENT) {
        set_error(SPINGALETT_ERR_INVALID, "image: a window of 1 to 65535 tokens");
        return NULL;
    }
    SpgImageBuilder *b = (SpgImageBuilder *)calloc(1, sizeof *b);
    if (!b) {
        set_error(SPINGALETT_ERR_ALLOC, "image: allocation failed");
        return NULL;
    }
    b->cells = cells;
    b->loss = loss;
    b->capacity = 64;
    b->layers = (BuiltLayer *)calloc(b->capacity, sizeof(BuiltLayer));
    if (!b->layers) {
        free(b);
        set_error(SPINGALETT_ERR_ALLOC, "image: allocation failed");
        return NULL;
    }
    b->layers[0].out_c = 1;
    b->layers[0].units = cells;
    return b;
}

void spingalett_image_free(SpgImageBuilder *b) {
    if (!b) return;
    free(b->layers);
    free(b);
}

static uint32_t refuse(const char *msg) {
    set_error(SPINGALETT_ERR_INVALID, msg);
    return 0;
}

uint32_t spingalett_image_layer(SpgImageBuilder *b, const SpgLayerSpec *spec) {
    if (b->count + 2 > SLETT_MAX_LAYERS) return refuse("image: too many layers");
    if (b->count + 2 > b->capacity) {
        BuiltLayer *grown = (BuiltLayer *)realloc(b->layers, (size_t)b->capacity * 2u * sizeof(BuiltLayer));
        if (!grown) {
            set_error(SPINGALETT_ERR_ALLOC, "image: allocation failed");
            return 0;
        }
        b->layers = grown;
        b->capacity *= 2u;
    }
    const uint32_t index = b->count + 1;
    BuiltLayer L;
    memset(&L, 0, sizeof L);
    L.spec = *spec;
    if (spec->input_count == 0 || spec->input_count > SPINGALETT_MAX_INPUTS) return refuse("image: a layer reads no input");
    for (uint32_t k = 0; k < spec->input_count; k++)
        if (spec->inputs[k] >= index || (spec->inputs[k] == 0 && spec->type != LAYER_EMBEDDING))
            return refuse("image: a layer reads a later layer, or the input without being an embedding");
    const BuiltLayer *first = &b->layers[spec->inputs[0]];
    L.in_c = first->out_c;
    switch (spec->type) {
        case LAYER_EMBEDDING:
            if (spec->inputs[0] != 0 || !spec->channels || !spec->vocabulary) return refuse("image: invalid embedding");
            L.out_c = spec->channels;
            L.rows = spec->vocabulary;
            L.row_len = spec->channels;
            L.biases = spec->positions ? (uint64_t)b->cells * spec->channels : 0u;
            break;
        case LAYER_CONV2D:
            if (!spec->channels) return refuse("image: a linear layer without outputs");
            L.out_c = spec->channels;
            L.rows = spec->channels;
            L.row_len = L.in_c;
            L.biases = L.rows;
            if (spec->tied) {
                const BuiltLayer *e = spec->tied < index ? &b->layers[spec->tied] : NULL;
                if (!e || e->spec.type != LAYER_EMBEDDING || e->rows != L.rows || e->row_len != L.row_len ||
                    e->spec.precision != spec->precision)
                    return refuse("image: a layer tied to no embedding of its shape and precision");
            }
            break;
        case LAYER_RMS_NORM:
        case LAYER_LAYER_NORM:
            if (!(spec->eps > 0.0f && spec->eps < 1.0f)) return refuse("image: a normalization's epsilon out of range");
            L.out_c = L.in_c;
            L.rows = L.in_c;
            L.row_len = 1;
            L.biases = spec->type == LAYER_LAYER_NORM ? L.rows : 0u;
            L.spec.precision = PRECISION_FLOAT32;
            break;
        case LAYER_ATTENTION: {
            const uint32_t heads = spec->heads, kv = spec->kv_heads;
            const uint32_t head = heads ? spec->channels / heads : 0u;
            const uint32_t rot = spec->rotary ? spec->rotary : head;
            if (!heads || !kv || heads % kv || !head || spec->channels % heads ||
                (uint64_t)L.in_c != (uint64_t)(heads + 2u * kv) * head || (spec->theta > 0.0f && (rot % 2u || rot > head)))
                return refuse("image: an attention whose input is not its heads' queries, keys and values");
            if ((spec->qk_norm && !(spec->eps > 0.0f && spec->eps < 1.0f)) || (spec->freqs && !(spec->theta > 0.0f)) ||
                (spec->rotary && !(spec->theta > 0.0f)))
                return refuse("image: an attention's normalization or rotary embeddings are invalid");
            L.out_c = spec->channels;
            L.gammas = spec->qk_norm ? 2u * (uint64_t)head : 0u;
            L.freqs = spec->freqs ? rot / 2u : 0u;
            break;
        }
        case LAYER_ADD:
        case LAYER_MULTIPLY:
        case LAYER_CONCAT:
            for (uint32_t k = 0; k < spec->input_count; k++) {
                const uint32_t c = b->layers[spec->inputs[k]].out_c;
                if (spec->type != LAYER_CONCAT && c != L.in_c) return refuse("image: the inputs of an addition differ");
                L.out_c += c;
            }
            if (spec->type != LAYER_CONCAT) L.out_c = L.in_c;
            break;
        default:
            return refuse("image: a kind of layer images of language models do not hold");
    }
    if ((uint64_t)L.out_c * b->cells > UINT32_MAX) return refuse("image: a layer's output exceeds 2^32 values");
    L.units = L.out_c * b->cells;
    b->layers[index] = L;
    b->count++;
    return index;
}

/* The oldest version that holds the layers: 10 with a block precision, 9 with a tied layer or a window, else 8. */
static uint16_t image_version(const SpgImageBuilder *b) {
    uint16_t version = 8u;
    for (uint32_t i = 1; i <= b->count; i++) {
        const SpgLayerSpec *s = &b->layers[i].spec;
        if ((b->layers[i].rows && spingalett_precision_is_block(s->precision)) || b->layers[i].gammas ||
            b->layers[i].freqs || s->rotary || s->score_scale != 0.0f)
            return 10u;
        if (s->tied || s->window) version = 9u;
    }
    return version;
}

uint8_t *spingalett_image_layout(SpgImageBuilder *b, size_t *size) {
    const uint32_t L = b->count, layers = L + 1;
    if (L == 0) {
        set_error(SPINGALETT_ERR_INVALID, "image: no layers");
        return NULL;
    }
    /* where every output but the last lives among the engine's activations */
    uint64_t *sizes = (uint64_t *)malloc((size_t)layers * sizeof(uint64_t));
    uint64_t *act = (uint64_t *)malloc((size_t)layers * sizeof(uint64_t));
    uint32_t *steps = (uint32_t *)malloc((size_t)layers * sizeof(uint32_t));
    const uint32_t **reads = (const uint32_t **)malloc((size_t)layers * sizeof(uint32_t *));
    uint32_t *read_count = (uint32_t *)malloc((size_t)layers * sizeof(uint32_t));
    uint64_t arena = UINT64_MAX;
    if (sizes && act && steps && reads && read_count) {
        for (uint32_t t = 0; t < layers; t++) sizes[t] = t == 0 || t == L ? 0u : 4u * (uint64_t)b->layers[t].units;
        for (uint32_t l = 1; l <= L; l++) {
            steps[l - 1] = l;
            reads[l - 1] = b->layers[l].spec.inputs;
            read_count[l - 1] = b->layers[l].spec.input_count;
        }
        arena = spingalett_plan_buffers(layers, sizes, L, steps, reads, read_count, SLETT_SECTION_ALIGN, act);
    }
    for (uint32_t l = 1; arena != UINT64_MAX && l <= L; l++) b->layers[l].act = l == L ? 0u : act[l];
    free(sizes);
    free(act);
    free(steps);
    free(reads);
    free(read_count);
    if (arena == UINT64_MAX) {
        set_error(SPINGALETT_ERR_ALLOC, "image: allocation failed");
        return NULL;
    }
    /* the sections, in table order */
    uint64_t pos = slett_align(SLETT_HEADER_SIZE + (uint64_t)L * SLETT_LAYER_ENTRY_SIZE_6);
    for (uint32_t l = 1; l <= L; l++) {
        BuiltLayer *e = &b->layers[l];
        const PrecisionMode p = e->spec.precision;
        if (e->spec.input_count > 1) {
            e->list = pos;
            pos = slett_align(pos + 4u * (uint64_t)e->spec.input_count);
        }
        if (e->gammas) {                        /* an attention's gammas and frequencies */
            e->weights = pos;
            pos = slett_align(pos + 4u * e->gammas);
        }
        if (e->freqs) {
            e->scales = pos;
            pos = slett_align(pos + 4u * e->freqs);
        }
        if (!e->rows) continue;
        const uint64_t row = spingalett_slett_row_bytes(p, e->row_len);
        if (!row) {
            set_error(SPINGALETT_ERR_INVALID, "image: a layer's rows are no whole blocks of its precision");
            return NULL;
        }
        if (e->spec.tied) {
            e->weights = b->layers[e->spec.tied].weights;
            e->scales = b->layers[e->spec.tied].scales;
        } else {
            e->weights = pos;
            pos = slett_align(pos + row * e->rows);
            if (spingalett_precision_is_int(p)) {
                e->scales = pos;
                pos = slett_align(pos + 4u * (uint64_t)e->rows);
            }
        }
        if (e->biases) {
            e->bias_at = pos;
            pos = slett_align(pos + 4u * e->biases);
        }
    }
    if (pos > (uint64_t)SIZE_MAX - SPINGALETT_ALIGNMENT) {
        set_error(SPINGALETT_ERR_ALLOC, "image: the model is too large for memory");
        return NULL;
    }
    uint8_t *img = (uint8_t *)spingalett_aligned_calloc((size_t)pos, 1);
    if (!img) {
        set_error(SPINGALETT_ERR_ALLOC, "image: allocation failed");
        return NULL;
    }
    memcpy(img, SLETT_MAGIC, 6);
    slett_put16(img + 6, image_version(b));
    slett_put32(img + 8, layers);
    img[12] = (uint8_t)b->loss;
    slett_put64(img + 24, pos);
    slett_put32(img + 32, 1u);                  /* the input: one row of cells */
    slett_put32(img + 36, b->cells);
    slett_put64(img + 40, arena);
    for (uint32_t l = 1; l <= L; l++) {
        const BuiltLayer *e = &b->layers[l];
        const SpgLayerSpec *s = &e->spec;
        uint8_t *t = img + SLETT_HEADER_SIZE + (size_t)(l - 1) * SLETT_LAYER_ENTRY_SIZE_6;
        uint32_t bits;
        slett_put32(t, b->layers[s->inputs[0]].units);
        slett_put32(t + 4, e->units);
        t[8] = slett_act_code(s->activation);
        t[9] = (uint8_t)s->precision;
        t[10] = (uint8_t)s->type;
        slett_put64(t + 16, e->weights);
        slett_put64(t + 24, e->scales);
        slett_put64(t + 32, e->bias_at);
        slett_put16(t + 48, 1u);
        slett_put16(t + 50, (uint16_t)b->cells);
        if (s->type == LAYER_CONV2D)            /* a 1 x 1 window, stride 1, one group */
            for (int k = 0; k < 4; k++) slett_put16(t + 52 + 2 * k, 1u);
        slett_put32(t + 64, s->type == LAYER_CONV2D ? 1u : s->type == LAYER_ATTENTION ? s->heads
                            : s->type == LAYER_EMBEDDING ? s->vocabulary : 0u);
        memcpy(&bits, &s->eps, 4);
        if (s->type == LAYER_RMS_NORM || s->type == LAYER_LAYER_NORM) slett_put32(t + 68, bits);
        if (s->type == LAYER_ATTENTION) {
            slett_put32(t + 72, s->window);
            slett_put32(t + 76, (s->causal ? 1u : 0u) | (s->qk_norm ? 4u : 0u) | (s->freqs ? 8u : 0u));
            slett_put32(t + 104, s->kv_heads);
            memcpy(&bits, &s->theta, 4);
            slett_put32(t + 108, bits);
            if (s->qk_norm) {
                memcpy(&bits, &s->eps, 4);
                slett_put32(t + 68, bits);
            }
            slett_put16(t + 52, (uint16_t)s->rotary);       /* (format 10) */
            memcpy(&bits, &s->score_scale, 4);
            slett_put32(t + 56, bits);
        }
        if (s->type == LAYER_EMBEDDING) slett_put32(t + 76, s->positions ? 1u : 0u);
        if (s->type == LAYER_CONV2D && s->tied) slett_put32(t + 104, s->tied);
        slett_put32(t + 80, s->input_count);
        slett_put32(t + 84, s->inputs[0]);
        slett_put64(t + 88, e->list);
        slett_put64(t + 96, e->act);
        for (uint32_t k = 0; e->list && k < s->input_count; k++) slett_put32(img + e->list + 4u * k, s->inputs[k]);
    }
    *size = (size_t)pos;
    return img;
}

uint8_t *spingalett_image_weights(const SpgImageBuilder *b, uint8_t *image, uint32_t index, uint32_t *rows,
                                  uint32_t *row_len) {
    const BuiltLayer *e = &b->layers[index];
    if (rows) *rows = e->rows;
    if (row_len) *row_len = e->row_len;
    return e->weights ? image + e->weights : NULL;
}

float *spingalett_image_scales(const SpgImageBuilder *b, uint8_t *image, uint32_t index) {
    const BuiltLayer *e = &b->layers[index];
    return e->scales ? (float *)(void *)(image + e->scales) : NULL;
}

float *spingalett_image_biases(const SpgImageBuilder *b, uint8_t *image, uint32_t index) {
    const BuiltLayer *e = &b->layers[index];
    return e->bias_at ? (float *)(void *)(image + e->bias_at) : NULL;
}

void spingalett_image_seal(uint8_t *image, size_t size) {
    slett_put32(image + 56, spingalett_crc32(0, image + SLETT_HEADER_SIZE, size - SLETT_HEADER_SIZE));
    slett_put32(image + 60, spingalett_crc32(0, image, 60));
}
