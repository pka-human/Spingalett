/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/*
 * .slett model files. Writing produces format version 3, version 4 for networks with convolution or
 * pooling layers, version 5 for networks with batch normalization or grouped convolutions, or
 * version 6 for graphs (docs/ModelFormat.md): a header, a layer table and 16-byte aligned sections,
 * little-endian, with per-row scales for the integer precisions and CRC-32 checksums, so that a
 * file image doubles as an in-place inference model. Reading also accepts versions 1 and 2 (native
 * byte order, one scale per tensor).
 */

#include "Spingalett.Private.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>

/* ------------------------------------------------------------------------- writing (formats 3 to 6) */

static float row_abs_max(const float *w, uint32_t n, bool *finite) {
    float amax = 0.0f;
    *finite = true;
    for (uint32_t i = 0; i < n; i++) {
        float a = fabsf(w[i]);
        if (!(a <= FLT_MAX)) *finite = false;
        else if (a > amax) amax = a;
    }
    return amax;
}

static int8_t quantize_value(float v, float inv, int limit) {
    long q = lrintf(v * inv);
    return (int8_t)(q > limit ? limit : (q < -limit ? -limit : q));
}

/* Stores one weight row in `precision` at dst; returns its scale (0 for the float formats). A row
   with a NaN or infinite weight gets scale NaN, so it reads back as NaN rather than silently. */
static float quantize_row(const float *w, uint32_t n, PrecisionMode precision, uint8_t *dst) {
    bool finite;
    float amax;
    switch (precision) {
        case PRECISION_FLOAT32:
            memcpy(dst, w, (size_t)n * 4u);
            return 0.0f;
        case PRECISION_FP16: {
            uint16_t h[256];
            for (uint32_t i0 = 0; i0 < n; i0 += 256) {
                uint32_t len = n - i0 < 256 ? n - i0 : 256;
                spingalett_fp16_encode(w + i0, len, h);
                for (uint32_t i = 0; i < len; i++) slett_put16(dst + 2u * (i0 + i), h[i]);
            }
            return 0.0f;
        }
        case PRECISION_BFLOAT16:
            for (uint32_t i = 0; i < n; i++) slett_put16(dst + 2u * i, spingalett_float_to_bf16(w[i]));
            return 0.0f;
        case PRECISION_INT8:
            amax = row_abs_max(w, n, &finite);
            if (!finite || amax == 0.0f) return finite ? 0.0f : NAN;       /* dst is zeroed */
            for (uint32_t i = 0; i < n; i++) dst[i] = (uint8_t)quantize_value(w[i], 127.0f / amax, 127);
            return amax / 127.0f;
        case PRECISION_INT4:
            amax = row_abs_max(w, n, &finite);
            if (!finite || amax == 0.0f) return finite ? 0.0f : NAN;
            for (uint32_t i = 0; i < n; i++) {
                uint8_t code = (uint8_t)quantize_value(w[i], 7.0f / amax, 7) & 0x0Fu;
                dst[i / 2u] |= (uint8_t)(i & 1u ? code << 4 : code);
            }
            return amax / 7.0f;
        case PRECISION_INT2: {
            /* Ternary weights (Li and Liu, 2016): threshold 0.7 * mean |w|; the scale is the mean
               magnitude of the weights above it. */
            double total = 0.0;
            finite = true;
            for (uint32_t i = 0; i < n; i++) {
                float a = fabsf(w[i]);
                if (!(a <= FLT_MAX)) finite = false;
                total += a;
            }
            if (!finite) return NAN;
            float threshold = (float)(0.7 * total / n);
            double kept = 0.0;
            uint32_t count = 0;
            for (uint32_t i = 0; i < n; i++)
                if (fabsf(w[i]) > threshold) { kept += fabsf(w[i]); count++; }
            if (count == 0) return 0.0f;
            for (uint32_t i = 0; i < n; i++) {
                uint8_t code = w[i] > threshold ? 1u : (w[i] < -threshold ? 3u : 0u);
                dst[i / 4u] |= (uint8_t)(code << ((i & 3u) * 2u));
            }
            return (float)(kept / count);
        }
        default:                            /* the block precisions: whole blocks, their scales inside */
            spingalett_block_quantize(precision, w, dst, n);
            return 0.0f;
    }
}

/* The oldest format version that holds the network: 3 for dense layers only, 4 with convolution or
   pooling layers, 5 with batch normalization or grouped convolutions, 6 for graphs, 7 with
   transposed convolutions, upsampling or layer normalization, 8 with the layers, activations or the
   loss of transformers, 9 with the training features of language models (layers tied to an embedding,
   attention ignoring padding or of a sliding window, cross-attention, embeddings of part of their input), 10 with
   layers of the block precisions. */
static PrecisionMode layer_precision(const NeuralNetwork *net, uint32_t l, PrecisionMode precision);

static uint16_t format_version(const NeuralNetwork *net, PrecisionMode precision) {
    for (uint32_t l = 0; l + 1 < net->layers; l++)
        if (spingalett_precision_is_block(layer_precision(net, l, precision))) return 10u;
    for (uint32_t l = 1; l < net->layers; l++) {
        const LayerShape *s = &net->shapes[l];
        if (s->tied || s->mask || s->window || (s->type == LAYER_ATTENTION && spingalett_input_count(net, l) == 2) ||
            (s->type == LAYER_EMBEDDING &&
             (s->token_offset || (uint64_t)s->height * s->width != net->topology[spingalett_source(net, l)])))
            return 9u;
    }
    if (net->loss_func == LOSS_SPARSE_CROSS_ENTROPY) return 8u;
    for (uint32_t l = 1; l < net->layers; l++)
        if (net->shapes[l].type > LAYER_LAYER_NORM || net->act_func[l - 1] >= ACT_GELU) return 8u;
    for (uint32_t l = 1; l < net->layers; l++)
        if (net->shapes[l].type > LAYER_GLOBAL_AVG_POOL) return 7u;
    if (net->graph) return 6u;
    uint16_t version = 3u;
    for (uint32_t l = 1; l < net->layers; l++) {
        const LayerShape *s = &net->shapes[l];
        if (s->type == LAYER_BATCH_NORM || (s->type == LAYER_CONV2D && s->groups > 1)) return 5u;
        if (s->type != LAYER_DENSE) version = 4u;
    }
    return version;
}

/* Normalizations keep their parameters (and batch normalization its statistics) in FLOAT32 whatever
   the file's precision. A block precision applies to dense layers, linear layers (1 x 1 convolutions of one
   group) and embeddings whose rows are whole blocks: of a k-quant or IQ4_XS, rows of whole blocks of 32 take
   the type of 32 llama.cpp falls back to (Q2_K, Q3_K and IQ4_XS IQ4_NL, Q4_K Q5_0, Q5_K Q5_1, Q6_K Q8_0), and
   other layers FP16. */
static PrecisionMode layer_precision(const NeuralNetwork *net, uint32_t l, PrecisionMode precision) {
    const LayerShape *s = &net->shapes[l + 1];
    if (spingalett_normalization(s->type)) return PRECISION_FLOAT32;
    if (!spingalett_precision_is_block(precision) || spingalett_weight_rows(net, l) == 0) return precision;
    const bool linear = s->type == LAYER_DENSE || s->type == LAYER_EMBEDDING ||
                        (s->type == LAYER_CONV2D && s->kernel_h == 1 && s->kernel_w == 1 && s->stride_h == 1 &&
                         s->stride_w == 1 && s->pad_h == 0 && s->pad_w == 0 && s->groups <= 1);
    const uint32_t n = spingalett_weight_row_len(net, l);
    if (!linear || n % 32u) return PRECISION_FP16;
    if (n % slett_block_values(precision) == 0) return precision;
    switch (precision) {
        case PRECISION_Q4_K: return PRECISION_Q5_0;
        case PRECISION_Q5_K: return PRECISION_Q5_1;
        case PRECISION_Q6_K: return PRECISION_Q8_0;
        default:             return PRECISION_IQ4_NL;
    }
}

/* Byte offsets of the layers' outputs among the engine's activations (version 6), every layer
   computed in turn and its output kept until its last reader has run; returns their bytes, or
   UINT64_MAX when out of memory. */
static uint64_t engine_plan(const NeuralNetwork *net, uint64_t *offsets) {
    const uint32_t layers = net->layers;
    uint64_t *sizes = (uint64_t *)malloc((size_t)layers * sizeof(uint64_t));
    uint32_t *steps = (uint32_t *)malloc((size_t)layers * sizeof(uint32_t));
    const uint32_t **reads = (const uint32_t **)malloc((size_t)layers * sizeof(uint32_t *));
    uint32_t *read_count = (uint32_t *)malloc((size_t)layers * sizeof(uint32_t));
    uint64_t total = UINT64_MAX;
    if (sizes && steps && reads && read_count) {
        for (uint32_t t = 0; t < layers; t++) sizes[t] = t == 0 || t + 1 == layers ? 0u : 4u * (uint64_t)net->topology[t];
        for (uint32_t l = 1; l < layers; l++) {
            steps[l - 1] = l;
            reads[l - 1] = spingalett_inputs(net, l);
            read_count[l - 1] = spingalett_input_count(net, l);
        }
        total = spingalett_plan_buffers(layers, sizes, layers - 1, steps, reads, read_count, SLETT_SECTION_ALIGN, offsets);
    }
    free(sizes);
    free(steps);
    free(reads);
    free(read_count);
    return total;
}

static void *save_image(const NeuralNetwork *net, PrecisionMode precision, bool save_optimizer, size_t *size) {
    if (net->layers < 2 || net->layers > SLETT_MAX_LAYERS) {
        set_error(SPINGALETT_ERR_INVALID, "save: network must have 2 to 65536 layers");
        return NULL;
    }
    if ((unsigned)precision >= PRECISION_COUNT) {
        set_error(SPINGALETT_ERR_INVALID, "save: invalid precision mode");
        return NULL;
    }
    if (!spingalett_host_is_little_endian()) {
        set_error(SPINGALETT_ERR_INVALID, "save: the .slett format needs a little-endian host");
        return NULL;
    }
    const uint16_t version = format_version(net, precision);
    const size_t entry_size = version == 3u ? SLETT_LAYER_ENTRY_SIZE : version == 4u ? SLETT_LAYER_ENTRY_SIZE_4
                            : version == 5u ? SLETT_LAYER_ENTRY_SIZE_5 : SLETT_LAYER_ENTRY_SIZE_6;
    uint32_t L = net->layers - 1;              /* weight layers */
    for (uint32_t l = 0; l < L; l++)
        if (spingalett_precision_is_int(layer_precision(net, l, precision)) &&
            net->shapes[l + 1].type != LAYER_EMBEDDING && spingalett_matrix_row_len(net, l) > SLETT_MAX_INT_INPUTS) {
            set_error(SPINGALETT_ERR_INVALID, "save: integer precisions allow at most 131072 weights per output");
            return NULL;
        }

    /* sections: weights, scales, biases, optimizer, and in version 6 the list of inputs */
    uint64_t *off = (uint64_t *)calloc((size_t)L * 5u, sizeof(uint64_t));
    /* version 6: where every layer's output lives among the engine's activations */
    uint64_t *act = version >= 6u ? (uint64_t *)malloc((size_t)net->layers * sizeof(uint64_t)) : NULL;
    uint64_t arena = 0;
    if (!off || (version >= 6u && !act)) {
        free(off);
        free(act);
        set_error(SPINGALETT_ERR_ALLOC, "save: allocation failed");
        return NULL;
    }
    if (act) {
        arena = engine_plan(net, act);
        if (arena == UINT64_MAX) {
            free(off);
            free(act);
            set_error(SPINGALETT_ERR_ALLOC, "save: allocation failed");
            return NULL;
        }
    }
    uint64_t pos = slett_align(SLETT_HEADER_SIZE + (uint64_t)L * entry_size);
    for (uint32_t l = 0; l < L; l++) {
        uint32_t rows = spingalett_weight_rows(net, l), row_len = spingalett_weight_row_len(net, l);
        PrecisionMode p = layer_precision(net, l, precision);
        if (version >= 6u && spingalett_input_count(net, l + 1) > 1) {
            off[4u * L + l] = pos;
            pos = slett_align(pos + 4u * (uint64_t)spingalett_input_count(net, l + 1));
        }
        if (rows == 0) continue;                /* pooling, adding, concatenating: no parameters */
        const uint64_t biases = spingalett_bias_count(net, l);
        const uint32_t tied = net->shapes[l + 1].tied;
        if (tied) {                             /* the weights and scales of the embedding's table */
            off[4 * l] = off[4u * (tied - 1u)];
            off[4 * l + 1] = off[4u * (tied - 1u) + 1u];
        } else {
            off[4 * l] = pos;
            pos = slett_align(pos + spingalett_slett_row_bytes(p, row_len) * rows);
        }
        if (!tied && (spingalett_precision_is_int(p) || net->shapes[l + 1].type == LAYER_BATCH_NORM)) {
            /* row scales; batch normalization: its running mean and variance */
            off[4 * l + 1] = pos;
            pos = slett_align(pos + (uint64_t)rows * (spingalett_precision_is_int(p) ? 4u : 8u));
        }
        if (biases) {                           /* (none for RMS normalization, embeddings without positions) */
            off[4 * l + 2] = pos;
            pos = slett_align(pos + biases * 4u);
        }
        if (save_optimizer) {
            off[4 * l + 3] = pos;
            pos = slett_align(pos + ((uint64_t)rows * row_len * 2u + biases * 2u) * 4u);
        }
    }
    if (pos > (uint64_t)SIZE_MAX - SPINGALETT_ALIGNMENT) {
        free(off);
        free(act);
        set_error(SPINGALETT_ERR_ALLOC, "save: network too large for memory");
        return NULL;
    }
    uint8_t *img = (uint8_t *)spingalett_aligned_calloc((size_t)pos, 1);
    if (!img) {
        free(off);
        free(act);
        set_error(SPINGALETT_ERR_ALLOC, "save: image allocation failed");
        return NULL;
    }

    memcpy(img, SLETT_MAGIC, 6);
    slett_put16(img + 6, version);
    slett_put32(img + 8, net->layers);
    img[12] = (uint8_t)net->loss_func;
    img[13] = save_optimizer ? SLETT_FLAG_OPTIMIZER : 0u;
    slett_put64(img + 16, save_optimizer ? net->time_step : 0u);
    slett_put64(img + 24, pos);
    if (version >= 4u) {
        slett_put32(img + 32, net->shapes[0].height);
        slett_put32(img + 36, net->shapes[0].width);
    }
    if (version >= 6u) slett_put64(img + 40, arena);

    for (uint32_t l = 0; l < L; l++) {
        uint32_t rows = spingalett_weight_rows(net, l), row_len = spingalett_weight_row_len(net, l);
        const LayerShape *shape = &net->shapes[l + 1];
        PrecisionMode p = layer_precision(net, l, precision);
        uint8_t *e = img + SLETT_HEADER_SIZE + (size_t)l * entry_size;
        slett_put32(e, net->topology[spingalett_source(net, l + 1)]);
        slett_put32(e + 4, net->topology[l + 1]);
        e[8] = slett_act_code(net->act_func[l]);
        e[9] = (uint8_t)p;
        uint32_t bits;
        memcpy(&bits, &net->dropout_rates[l + 1], 4);
        slett_put32(e + 12, bits);
        for (int k = 0; k < 4; k++) slett_put64(e + 16 + 8 * k, off[4 * l + k]);
        if (version >= 4u) {
            e[10] = (uint8_t)shape->type;
            const uint32_t fields[8] = {shape->height, shape->width, shape->kernel_h, shape->kernel_w,
                                        shape->stride_h, shape->stride_w, shape->pad_h, shape->pad_w};
            for (int k = 0; k < 8; k++) slett_put16(e + 48 + 2 * k, (uint16_t)fields[k]);
        }
        if (version >= 5u) {
            /* groups; an attention's heads, an embedding's vocabulary (version 8) */
            slett_put32(e + 64, spingalett_filters(shape->type) ? shape->groups : shape->type == LAYER_ATTENTION
                                ? shape->heads : shape->type == LAYER_EMBEDDING ? shape->vocabulary : 0u);
            memcpy(&bits, &shape->eps, 4);
            slett_put32(e + 68, bits);
            memcpy(&bits, &shape->momentum, 4);
            /* (version 9: an attention's sliding window in the momentum's field) */
            slett_put32(e + 72, shape->type == LAYER_ATTENTION && version >= 9u ? shape->window : bits);
        }
        /* the upsampling mode; attention's flags (causal; version 9: ignoring padding, the embedding of the tokens in
           bits 16 to 31), embeddings' positions */
        if (version >= 7u)
            slett_put32(e + 76, shape->type == LAYER_UPSAMPLE ? shape->mode
                                : shape->type == LAYER_ATTENTION ? (shape->causal ? 1u : 0u) |
                                                                   (shape->mask ? 2u | shape->mask << 16 : 0u)
                                : shape->type == LAYER_EMBEDDING ? (shape->positions ? 1u : 0u) : 0u);
        if (version >= 8u && shape->type == LAYER_ATTENTION) {
            slett_put32(e + 104, shape->kv_heads);
            memcpy(&bits, &shape->theta, 4);
            slett_put32(e + 108, bits);
        }
        if (version >= 9u && shape->tied) slett_put32(e + 104, shape->tied);     /* the embedding whose table */
        if (version >= 9u && shape->type == LAYER_EMBEDDING) slett_put32(e + 104, shape->token_offset);
        if (version >= 6u) {
            const uint32_t count = spingalett_input_count(net, l + 1), *in = spingalett_inputs(net, l + 1);
            slett_put32(e + 80, count);
            slett_put32(e + 84, in[0]);
            slett_put64(e + 88, count > 1 ? off[4u * L + l] : 0u);
            slett_put64(e + 96, l + 1 < L ? act[l + 1] : 0u);
            for (uint32_t k = 0; count > 1 && k < count; k++) slett_put32(img + off[4u * L + l] + 4u * k, in[k]);
        }
        if (rows == 0) continue;

        const float *W = net->weights + net->weight_offsets[l];
        size_t row = (size_t)spingalett_slett_row_bytes(p, row_len);
        for (uint32_t j = 0; !shape->tied && j < rows; j++) {     /* (a tied layer's rows: the table's) */
            float scale = quantize_row(W + (size_t)j * row_len, row_len, p, img + off[4 * l] + (size_t)j * row);
            if (spingalett_precision_is_int(p)) memcpy(img + off[4 * l + 1] + 4u * (size_t)j, &scale, 4);
        }
        if (shape->type == LAYER_BATCH_NORM) {
            memcpy(img + off[4 * l + 1], net->running_mean + net->bias_offsets[l], (size_t)rows * 4u);
            memcpy(img + off[4 * l + 1] + (size_t)rows * 4u, net->running_var + net->bias_offsets[l], (size_t)rows * 4u);
        }
        const size_t bbytes = (size_t)spingalett_bias_count(net, l) * 4u;
        if (bbytes) memcpy(img + off[4 * l + 2], net->biases + net->bias_offsets[l], bbytes);
        if (save_optimizer && net->opt_m_weights) {     /* without, before any training: the zeros the image has */
            uint8_t *o = img + off[4 * l + 3];
            size_t wbytes = (size_t)rows * row_len * 4u;
            memcpy(o, net->opt_m_weights + net->weight_offsets[l], wbytes);
            memcpy(o + wbytes, net->opt_v_weights + net->weight_offsets[l], wbytes);
            memcpy(o + 2u * wbytes, net->opt_m_biases + net->bias_offsets[l], bbytes);
            memcpy(o + 2u * wbytes + bbytes, net->opt_v_biases + net->bias_offsets[l], bbytes);
        }
    }
    free(off);
    free(act);

    slett_put32(img + 56, spingalett_crc32(0, img + SLETT_HEADER_SIZE, (size_t)pos - SLETT_HEADER_SIZE));
    slett_put32(img + 60, spingalett_crc32(0, img, 60));
    *size = (size_t)pos;
    return img;
}

/* Whether layer l is a batch normalization that folds into its input: a dense or convolution layer
   without activation that feeds nothing else (uses: the inputs of later layers naming each layer). */
static bool foldable(const NeuralNetwork *net, const uint32_t *uses, uint32_t l) {
    if (l < 2 || net->shapes[l].type != LAYER_BATCH_NORM) return false;
    uint32_t s = spingalett_source(net, l);
    if (s == 0 || net->shapes[s].tied) return false;     /* (a tied layer's rows are the table's) */
    LayerType type = net->shapes[s].type;
    return (type == LAYER_DENSE || spingalett_filters(type)) && net->act_func[s - 1] == ACT_NONE && uses[s] == 1;
}

/* A copy of net with its foldable batch normalizations folded into the layers before them: those
   layers' weight rows scaled by gamma / sqrt(var + eps) and their biases shifted alike, taking the
   normalization's activation and dropout rate, and the layers that read a normalization reading
   them instead. NULL with *failed false when there is nothing to fold; the optimizer state is not
   copied. */
static NeuralNetwork *fold_batch_norm(const NeuralNetwork *net, bool *failed) {
    *failed = false;
    const uint32_t L = net->layers;
    uint32_t *uses = (uint32_t *)calloc(3u * (size_t)L, sizeof(uint32_t));
    if (!uses) {
        set_error(SPINGALETT_ERR_ALLOC, "save: allocation failed");
        *failed = true;
        return NULL;
    }
    uint32_t *map = uses + L, *into = uses + 2u * L;    /* new index of each layer; the normalization folded into it */
    for (uint32_t l = 1; l < L; l++)
        for (uint32_t k = 0; k < spingalett_input_count(net, l); k++) uses[spingalett_inputs(net, l)[k]]++;
    bool any = false;
    for (uint32_t l = 2; l < L; l++)
        if (foldable(net, uses, l)) {
            into[spingalett_source(net, l)] = l;
            any = true;
        }
    if (!any) {
        free(uses);
        return NULL;
    }

    NeuralNetwork *src = (NeuralNetwork *)net;      /* only read */
    NeuralNetwork *f = new_spingalett_struct_arguments((NeuralNetworkArgs){ .loss_func = net->loss_func });
    bool ok = f != NULL;
    for (uint32_t l = 0; ok && l < L; l++) {
        if (foldable(net, uses, l)) {
            map[l] = map[spingalett_source(net, l)];
            continue;
        }
        LayerArgs a = spingalett_describe_layer(src, l);
        a.net = f;
        for (uint32_t k = 0; k < a.input_count; k++) a.inputs[k] = map[a.inputs[k]];
        if (a.tied) a.tied = map[a.tied];
        if (into[l]) {
            a.act_func = net->act_func[into[l] - 1];
            a.dropout_rate = net->dropout_rates[into[l]];
        }
        ok = spingalett_add_layer(a);
        map[l] = ok ? f->layers - 1 : 0u;
    }
    float *coef = ok ? (float *)malloc(2u * (size_t)net->total_biases * sizeof(float) + sizeof(float)) : NULL;
    if (!ok || !coef) {
        free(coef);
        free(uses);
        free_network(f);
        if (ok) set_error(SPINGALETT_ERR_ALLOC, "save: allocation failed");
        *failed = true;
        return NULL;
    }

    for (uint32_t l = 1; l < L; l++) {
        if (foldable(net, uses, l)) continue;   /* folded into its input below */
        const uint32_t k = map[l] - 1;          /* weight layer l - 1 becomes weight layer k */
        uint64_t rows = spingalett_weight_rows(net, l - 1), len = spingalett_weight_row_len(net, l - 1);
        uint64_t sw = net->weight_offsets[l - 1], sb = net->bias_offsets[l - 1], nb = spingalett_bias_count(net, l - 1);
        uint64_t dw = f->weight_offsets[k], db = f->bias_offsets[k];
        memcpy(f->weights + dw, net->weights + sw, (size_t)(rows * len) * sizeof(float));
        memcpy(f->biases + db, net->biases + sb, (size_t)nb * sizeof(float));
        memcpy(f->running_mean + db, net->running_mean + sb, (size_t)nb * sizeof(float));
        memcpy(f->running_var + db, net->running_var + sb, (size_t)nb * sizeof(float));
        if (into[l]) {
            /* weight layer into[l] - 1 is the normalization of this layer's rows (one channel each) */
            const uint32_t b = into[l];
            uint64_t nw = net->weight_offsets[b - 1], nb = net->bias_offsets[b - 1];
            float *a = coef, *c = coef + rows;
            spingalett_bn_coefficients(net->weights + nw, net->biases + nb, net->running_mean + nb, net->running_var + nb,
                                       net->shapes[b].eps, (uint32_t)rows, a, c);
            for (uint64_t j = 0; j < rows; j++) {
                float *row = f->weights + dw + j * len;
                for (uint64_t i = 0; i < len; i++) row[i] *= a[j];
                f->biases[db + j] = f->biases[db + j] * a[j] + c[j];
            }
        }
    }
    free(coef);
    free(uses);
    f->time_step = net->time_step;
    return f;
}

void *spingalett_save_deployment(const NeuralNetwork *net, PrecisionMode precision, size_t *size) {
    if (size) *size = 0;
    if (!net || !size) {
        set_error(SPINGALETT_ERR_INVALID, "save: net or size is NULL");
        return NULL;
    }
    spingalett_network_sync(net);
    bool failed;
    NeuralNetwork *folded = fold_batch_norm(net, &failed);
    if (failed) return NULL;
    void *img = save_image(folded ? folded : net, precision, false, size);
    free_network(folded);
    return img;
}

void *spingalett_save_to_memory(const NeuralNetwork *net, PrecisionMode precision, bool save_optimizer, size_t *size) {
    if (size) *size = 0;
    if (!net || !size) {
        set_error(SPINGALETT_ERR_INVALID, "save: net or size is NULL");
        return NULL;
    }
    spingalett_network_sync(net);
    /* quantized files without optimizer state are for deployment: normalizations are folded */
    if (precision != PRECISION_FLOAT32 && !save_optimizer)
        return spingalett_save_deployment(net, precision, size);
    return save_image(net, precision, save_optimizer, size);
}

void spingalett_free(void *ptr) {
    spingalett_aligned_free(ptr);
}

static const char *find_last_separator(const char *path) {
    const char *slash = strrchr(path, '/');
    const char *backslash = strrchr(path, '\\');
    if (slash && backslash)
        return (slash > backslash) ? slash : backslash;
    return slash ? slash : backslash;
}

/* The name a model is saved under: SPINGALETT_MODEL_EXTENSION appended to one without an extension
   (in *allocated, which the caller frees). NULL when out of memory. */
static const char *model_filename(const char *name, char **allocated) {
    *allocated = NULL;
    const char *dot = strrchr(name, '.');
    const char *last_sep = find_last_separator(name);
    if (dot && (!last_sep || dot > last_sep)) return name;
    size_t len = strlen(name);
    *allocated = (char *)malloc(len + sizeof SPINGALETT_MODEL_EXTENSION);
    if (!*allocated) {
        set_error(SPINGALETT_ERR_ALLOC, "save: filename allocation failed");
        return NULL;
    }
    memcpy(*allocated, name, len);
    memcpy(*allocated + len, SPINGALETT_MODEL_EXTENSION, sizeof SPINGALETT_MODEL_EXTENSION);
    return *allocated;
}

bool spingalett_save_args(SaveArgs args) {
    if (!args.net || !args.filename) {
        set_error(SPINGALETT_ERR_INVALID, "save: net or filename is NULL");
        return false;
    }

    char *allocated_filename = NULL;
    const char *target_filename = model_filename(args.filename, &allocated_filename);
    if (!target_filename) return false;

    size_t size = 0;
    void *img = spingalett_save_to_memory(args.net, args.precision, !args.do_not_save_optimizer, &size);
    if (!img) {
        free(allocated_filename);
        return false;
    }

    bool ok = false;
    FILE *fp = fopen(target_filename, "wb");
    if (!fp) {
        set_error(SPINGALETT_ERR_FILE_IO, "save: cannot open file for writing");
    } else {
        ok = fwrite(img, 1, size, fp) == size;
        if (fclose(fp) != 0) ok = false;
        if (!ok) {
            set_error(SPINGALETT_ERR_FILE_IO, "save: write error (disk full?)");
            spingalett_log(LOG_ERROR, "Write error saving to %s", target_filename);
        } else {
            spingalett_log(LOG_INFO, "Network saved to %s", target_filename);
            spingalett_log(LOG_INFO, "Save info: format=v%u, layers=%u, weights=%llu, biases=%llu, precision=%s, optimizer=%s, bytes=%zu",
                (unsigned)slett_get16((const uint8_t *)img + 6), slett_get32((const uint8_t *)img + 8),
                (unsigned long long)args.net->total_weights, (unsigned long long)args.net->total_biases,
                precision_names[args.precision], args.do_not_save_optimizer ? "OFF" : "ON", size);
        }
    }
    spingalett_free(img);
    free(allocated_filename);
    return ok;
}

/* ------------------------------------------------------------------------- reading */

/* Builds an empty network of the given shape (weights zero). */
static NeuralNetwork *make_network(LossFunction loss, uint32_t layers, const uint32_t *topology,
                                   const ActivationFunction *act, const float *dropout) {
    NeuralNetwork *net = new_spingalett_struct_arguments((NeuralNetworkArgs){ .loss_func = loss });
    if (!net) return NULL;
    for (uint32_t l = 0; l < layers; l++) {
        LayerArgs largs = {0};
        largs.net = net;
        largs.neurons_amount = topology[l];
        if (l > 0) largs.act_func = act[l - 1];
        largs.weight_initialization = WEIGHT_INITIALIZATION_NONE;
        largs.dropout_rate = dropout[l];
        if (!spingalett_add_layer(largs)) {
            free_network(net);
            return NULL;
        }
    }
    return net;
}

/* The network a format 3 to 6 image describes, its parameters 0. */
static NeuralNetwork *network_of_image(const uint8_t *p, const SlettInfo *info) {
    NeuralNetwork *net = new_spingalett_struct_arguments((NeuralNetworkArgs){ .loss_func = info->loss });
    if (!net) return NULL;
    SlettLayer first;
    spingalett_slett_layer(p, 0, &first);
    LayerArgs input = {0};
    input.net = net;
    input.neurons_amount = first.inputs;
    if (info->version >= 4) {
        input.height = first.in_h;
        input.width = first.in_w;
        input.channels = first.in_c;
    }
    /* room for every layer's neurons and parameters at once (adding layers then moves nothing) */
    uint64_t neurons = first.inputs, weights = 0, biases = 0;
    for (uint32_t l = 0; l + 1 < info->layers; l++) {
        SlettLayer e;
        spingalett_slett_layer(p, l, &e);
        neurons += e.outputs;
        weights += e.tied ? 0u : (uint64_t)e.rows * e.row_len;
        biases += e.type == LAYER_EMBEDDING ? (e.positions ? e.outputs : 0u) : e.type == LAYER_RMS_NORM ? 0u : e.rows;
    }
    (void)spingalett_network_reserve(net, neurons, weights, biases);
    bool ok = spingalett_add_layer(input);
    for (uint32_t l = 0; ok && l + 1 < info->layers; l++) {
        SlettLayer e;
        spingalett_slett_layer(p, l, &e);
        LayerArgs a = {0};
        a.net = net;
        a.type = e.type;
        a.act_func = e.activation;
        a.dropout_rate = e.dropout;
        a.weight_initialization = WEIGHT_INITIALIZATION_NONE;
        a.input_count = e.input_count;
        for (uint32_t k = 0; k < e.input_count; k++) a.inputs[k] = spingalett_slett_input(p, &e, k);
        a.tied = e.tied;
        if (e.type == LAYER_DENSE) {
            a.neurons_amount = e.outputs;
        } else if (e.type == LAYER_BATCH_NORM || e.type == LAYER_LAYER_NORM || e.type == LAYER_RMS_NORM) {
            a.epsilon = e.eps;
            a.momentum = e.momentum;
        } else if (e.type == LAYER_EMBEDDING) {
            a.neurons_amount = e.out_c;
            a.vocabulary = e.vocabulary;
            a.positions = e.positions;
            if (e.token_offset || e.out_h * e.out_w != e.inputs) {      /* (version 9: a range of tokens) */
                a.token_offset = e.token_offset;
                a.token_count = e.out_h * e.out_w;
            }
        } else if (e.type == LAYER_ATTENTION) {
            if (e.qk_norm || e.freqs || e.score_scale != 0.0f || (e.theta > 0.0f && e.rotary != e.out_c / e.heads)) {
                free_network(net);
                set_error(SPINGALETT_ERR_INVALID, "load: attention that normalizes its queries and keys, has rotary "
                          "frequencies or a scale of its own, or rotates part of its heads (format 10) runs as a model, "
                          "not yet as a network");
                return NULL;
            }
            a.heads = e.heads;
            a.kv_heads = e.kv_heads;
            a.rope_theta = e.theta;
            a.causal = e.causal;
            a.window = e.window;
        } else if (e.type == LAYER_ADD || e.type == LAYER_CONCAT || e.type == LAYER_GLOBAL_AVG_POOL ||
                   e.type == LAYER_MULTIPLY) {
            /* the inputs give the shape */
        } else if (e.type == LAYER_UPSAMPLE) {
            a.stride_h = e.stride_h;
            a.stride_w = e.stride_w;
            a.upsample = (UpsampleMode)e.mode;
        } else {
            const bool filters = e.type == LAYER_CONV2D || e.type == LAYER_CONV_TRANSPOSE2D;
            a.filters = filters ? e.out_c : 0;
            a.groups = filters ? e.groups : 0;
            if (e.type == LAYER_CONV_TRANSPOSE2D) {
                a.output_padding_h = e.out_h - ((e.in_h - 1) * e.stride_h - 2 * e.pad_h + e.kernel_h);
                a.output_padding_w = e.out_w - ((e.in_w - 1) * e.stride_w - 2 * e.pad_w + e.kernel_w);
            }
            a.kernel_h = e.kernel_h; a.kernel_w = e.kernel_w;
            a.stride_h = e.stride_h; a.stride_w = e.stride_w;
            a.padding_h = e.pad_h; a.padding_w = e.pad_w;
        }
        a.ignore_padding = e.mask != 0;
        /* (the network's own trace of the padding's tokens: the file's) */
        ok = spingalett_add_layer(a) && net->topology[l + 1] == e.outputs && net->shapes[l + 1].mask == e.mask;
    }
    if (!ok) {
        free_network(net);
        set_error(SPINGALETT_ERR_INVALID, "load: the layer table describes no valid network");
        return NULL;
    }
    return net;
}

static NeuralNetwork *load_image(const uint8_t *p, size_t size, PrecisionMode *precision) {
    SlettInfo info;
    if (spingalett_slett_validate(p, size, &info) != SPINGALETT_OK) return NULL;
    NeuralNetwork *net = network_of_image(p, &info);
    if (!net) return NULL;
    if ((info.flags & SLETT_FLAG_OPTIMIZER) && !spingalett_training_state(net)) {
        free_network(net);
        return NULL;
    }

    uint32_t L = info.layers - 1;
    int8_t *codes = NULL;
    bool first = true;
    /* the file's precision: that of the first dense or convolution layer (normalizations are FLOAT32) */
    *precision = PRECISION_FLOAT32;
    for (uint32_t l = 0; l < L; l++) {
        SlettLayer e;
        spingalett_slett_layer(p, l, &e);
        if (first && (e.type == LAYER_DENSE || e.type == LAYER_CONV2D || e.type == LAYER_CONV_TRANSPOSE2D ||
                      e.type == LAYER_EMBEDDING)) {
            *precision = e.precision;
            first = false;
        }
        if (e.rows == 0) continue;
        if (e.rows != spingalett_weight_rows(net, l) || e.row_len != spingalett_matrix_row_len(net, l)) {
            free(codes);
            free_network(net);
            set_error(SPINGALETT_ERR_INVALID, "load: weight shapes differ from the network's");
            return NULL;
        }
        float *W = net->weights + net->weight_offsets[l];
        const uint8_t *src = p + e.weights;
        size_t row = (size_t)spingalett_slett_row_bytes(e.precision, e.row_len);
        if (e.precision == PRECISION_INT4 || e.precision == PRECISION_INT2) {
            int8_t *grown = (int8_t *)realloc(codes, e.row_len);
            if (!grown) {
                free(codes);
                free_network(net);
                set_error(SPINGALETT_ERR_ALLOC, "load: allocation failed");
                return NULL;
            }
            codes = grown;
        }
        for (uint32_t j = 0; !e.tied && j < e.rows; j++) {      /* (a tied layer's rows: the table's) */
            float *dst = W + (size_t)j * e.row_len;
            const uint8_t *r = src + (size_t)j * row;
            float scale = 0.0f;
            if (spingalett_precision_is_int(e.precision)) memcpy(&scale, p + e.scales + 4u * (size_t)j, 4);
            switch (e.precision) {
                case PRECISION_FLOAT32:
                    memcpy(dst, r, (size_t)e.row_len * 4u);
                    break;
                case PRECISION_FP16:
                    for (uint32_t k = 0; k < e.row_len; k++) dst[k] = spingalett_fp16_to_float(slett_get16(r + 2u * k));
                    break;
                case PRECISION_BFLOAT16:
                    for (uint32_t k = 0; k < e.row_len; k++) dst[k] = spingalett_bf16_to_float(slett_get16(r + 2u * k));
                    break;
                case PRECISION_INT8:
                    for (uint32_t k = 0; k < e.row_len; k++) dst[k] = (float)(int8_t)r[k] * scale;
                    break;
                case PRECISION_INT4:
                case PRECISION_INT2:
                    if (e.precision == PRECISION_INT4) spingalett_unpack_int4(r, codes, e.row_len);
                    else spingalett_unpack_int2(r, codes, e.row_len);
                    for (uint32_t k = 0; k < e.row_len; k++) dst[k] = (float)codes[k] * scale;
                    break;
                default:                        /* the block precisions */
                    spingalett_block_decode(e.precision, r, dst, e.row_len);
                    break;
            }
        }
        const size_t bbytes = (size_t)spingalett_bias_count(net, l) * 4u;
        if (bbytes) memcpy(net->biases + net->bias_offsets[l], p + e.biases, bbytes);
        if (e.type == LAYER_BATCH_NORM) {
            memcpy(net->running_mean + net->bias_offsets[l], p + e.scales, (size_t)e.rows * 4u);
            memcpy(net->running_var + net->bias_offsets[l], p + e.scales + (size_t)e.rows * 4u, (size_t)e.rows * 4u);
        }
        if (info.flags & SLETT_FLAG_OPTIMIZER) {
            const uint8_t *o = p + e.optimizer;
            size_t wbytes = e.tied ? 0u : (size_t)e.rows * e.row_len * 4u;
            memcpy(net->opt_m_weights + net->weight_offsets[l], o, wbytes);
            memcpy(net->opt_v_weights + net->weight_offsets[l], o + wbytes, wbytes);
            memcpy(net->opt_m_biases + net->bias_offsets[l], o + 2u * wbytes, bbytes);
            memcpy(net->opt_v_biases + net->bias_offsets[l], o + 2u * wbytes + bbytes, bbytes);
        }
    }
    free(codes);
    net->time_step = info.time_step;
    return net;
}

/* Versions 1 and 2: native byte order, every array in the file's precision with one scale per
   array for the integer precisions. */
typedef struct {
    const uint8_t *p;
    size_t left;
} Cursor;

static bool take(Cursor *c, void *dst, size_t n) {
    if (n > c->left) return false;
    memcpy(dst, c->p, n);
    c->p += n;
    c->left -= n;
    return true;
}

static bool read_legacy_array(Cursor *c, float *data, uint64_t size, PrecisionMode precision) {
    if (precision == PRECISION_FLOAT32)
        return size <= c->left / 4u && take(c, data, (size_t)size * 4u);
    if (precision == PRECISION_FP16 || precision == PRECISION_BFLOAT16) {
        if (size > c->left / 2u) return false;
        for (uint64_t i = 0; i < size; i++) {
            uint16_t h;
            take(c, &h, 2);
            data[i] = precision == PRECISION_FP16 ? spingalett_fp16_to_float(h) : spingalett_bf16_to_float(h);
        }
        return true;
    }
    float max_val;
    if (!take(c, &max_val, 4)) return false;
    uint64_t bytes = precision == PRECISION_INT8 ? size : (precision == PRECISION_INT4 ? (size + 1) / 2 : (size + 3) / 4);
    if (bytes > c->left) return false;
    const uint8_t *b = c->p;
    for (uint64_t i = 0; i < size; i++) {
        if (!(max_val > 0.0f)) { data[i] = 0.0f; continue; }
        if (precision == PRECISION_INT8) {
            data[i] = ((float)(int8_t)b[i] / 127.0f) * max_val;
        } else if (precision == PRECISION_INT4) {
            unsigned uq = (i & 1u) ? (unsigned)(b[i / 2] >> 4) : (unsigned)(b[i / 2] & 0x0Fu);
            data[i] = ((float)((int)(uq ^ 8u) - 8) / 7.0f) * max_val;
        } else {
            unsigned uq = (b[i / 4] >> ((i % 4) * 2u)) & 3u;
            data[i] = (float)((int)(uq ^ 2u) - 2) * max_val;
        }
    }
    c->p += bytes;
    c->left -= (size_t)bytes;
    return true;
}

static NeuralNetwork *load_legacy(const uint8_t *data, size_t size, PrecisionMode *precision) {
    Cursor c = {data, size};
    uint16_t version = 0;
    uint32_t layers = 0;
    uint8_t loss = 0, has_optimizer = 0, prec = 0;
    uint64_t ts = 0;
    take(&c, &version, 2);
    if (!take(&c, &layers, 4) || !take(&c, &loss, 1) || !take(&c, &has_optimizer, 1) ||
        (has_optimizer && !take(&c, &ts, 8)) || !take(&c, &prec, 1)) {
        set_error(SPINGALETT_ERR_FILE_IO, "load: file is truncated");
        return NULL;
    }
    if (layers < 2 || layers > SLETT_MAX_LAYERS || loss >= LOSS_COUNT || prec >= PRECISION_COUNT) {
        set_error(SPINGALETT_ERR_INVALID, "load: invalid header");
        return NULL;
    }
    *precision = (PrecisionMode)prec;

    uint32_t *topology = (uint32_t *)malloc((size_t)layers * sizeof(uint32_t));
    ActivationFunction *act = (ActivationFunction *)malloc((size_t)(layers - 1) * sizeof(ActivationFunction));
    float *dropout = (float *)calloc(layers, sizeof(float));
    NeuralNetwork *net = NULL;
    bool ok = topology && act && dropout;
    if (!ok) set_error(SPINGALETT_ERR_ALLOC, "load: allocation failed");
    if (ok && !take(&c, topology, (size_t)layers * 4u)) {
        set_error(SPINGALETT_ERR_FILE_IO, "load: file is truncated");
        ok = false;
    }
    for (uint32_t i = 0; ok && i < layers; i++)
        if (topology[i] == 0) { set_error(SPINGALETT_ERR_INVALID, "load: invalid topology"); ok = false; }
    for (uint32_t i = 0; ok && i + 1 < layers; i++) {
        uint8_t a;
        if (!take(&c, &a, 1) || slett_act(a) == ACT_COUNT) {
            set_error(SPINGALETT_ERR_INVALID, "load: invalid activation function");
            ok = false;
        } else {
            act[i] = slett_act(a);
        }
    }
    if (ok && version >= 2) {                         /* v2: dropout rate of every non-input layer */
        ok = take(&c, dropout + 1, (size_t)(layers - 1) * 4u);
        for (uint32_t l = 1; ok && l < layers; l++) ok = dropout[l] >= 0.0f && dropout[l] < 1.0f;
        if (!ok) set_error(SPINGALETT_ERR_INVALID, "load: invalid dropout rates");
    }
    if (ok) net = make_network((LossFunction)loss, layers, topology, act, dropout);
    free(topology);
    free(act);
    free(dropout);
    if (!net) return NULL;
    net->time_step = ts;
    if (has_optimizer && !spingalett_training_state(net)) {
        free_network(net);
        return NULL;
    }

    bool read_ok = true;
    for (uint32_t l = 0; l + 1 < layers && read_ok; l++) {
        uint64_t wcount = (uint64_t)net->topology[l] * net->topology[l + 1], bcount = net->topology[l + 1];
        uint64_t woff = net->weight_offsets[l], boff = net->bias_offsets[l];
        read_ok = read_legacy_array(&c, net->weights + woff, wcount, *precision);
        if (read_ok && has_optimizer)
            read_ok = read_legacy_array(&c, net->opt_m_weights + woff, wcount, *precision) &&
                      read_legacy_array(&c, net->opt_v_weights + woff, wcount, *precision);
        if (read_ok) read_ok = read_legacy_array(&c, net->biases + boff, bcount, *precision);
        if (read_ok && has_optimizer)
            read_ok = read_legacy_array(&c, net->opt_m_biases + boff, bcount, *precision) &&
                      read_legacy_array(&c, net->opt_v_biases + boff, bcount, *precision);
    }
    if (!read_ok) {
        set_error(SPINGALETT_ERR_FILE_IO, "load: file is truncated");
        free_network(net);
        return NULL;
    }
    return net;
}

NeuralNetwork *spingalett_load_from_memory_ex(const void *data, size_t size, PrecisionMode *precision) {
    PrecisionMode dummy;
    if (!precision) precision = &dummy;
    if (!data) {
        set_error(SPINGALETT_ERR_INVALID, "load: data is NULL");
        return NULL;
    }
    const uint8_t *p = (const uint8_t *)data;
    if (size >= 6 && memcmp(p, SLETT_MAGIC, 6) == 0)
        return load_image(p, size, precision);
    uint16_t version = 0;
    if (size < 2) {
        set_error(SPINGALETT_ERR_FILE_IO, "load: failed to read format version");
        return NULL;
    }
    memcpy(&version, p, 2);
    if (version != 1 && version != 2) {
        set_error(SPINGALETT_ERR_FORMAT_VERSION, "load: unsupported format version");
        spingalett_log(LOG_ERROR, "Unsupported file format version %u (supported: 1..%u)", (unsigned)version,
                       (unsigned)SPINGALETT_FORMAT_VERSION);
        return NULL;
    }
    return load_legacy(p, size, precision);
}

NeuralNetwork *spingalett_load_from_memory(const void *data, size_t size) {
    return spingalett_load_from_memory_ex(data, size, NULL);
}

NeuralNetwork *spingalett_load(const char *filename) {
    if (!filename) {
        set_error(SPINGALETT_ERR_INVALID, "load: filename is NULL");
        return NULL;
    }
    size_t size = 0;
    void *data = spingalett_read_file(filename, &size);
    if (!data) return NULL;
    PrecisionMode precision = PRECISION_FLOAT32;
    NeuralNetwork *net = spingalett_load_from_memory_ex(data, size, &precision);
    spingalett_aligned_free(data);
    if (!net) {
        spingalett_log(LOG_ERROR, "Could not load %s: %s", filename, spingalett_last_error_message());
        return NULL;
    }
    spingalett_log(LOG_INFO, "Network loaded from %s", filename);
    spingalett_log(LOG_INFO, "Load info: layers=%u, weights=%llu, biases=%llu, precision=%s",
        net->layers, (unsigned long long)net->total_weights, (unsigned long long)net->total_biases,
        precision_names[precision]);
    return net;
}

/* ------------------------------------------------------------------------- checkpoints */

/*
 * A checkpoint is the network's FLOAT32 image with its optimizer state and, after the image's last byte
 * (the size its header holds), a record of the run that trained it (docs/ModelFormat.md, Checkpoints):
 *
 *   offset size
 *      0    6   magic "SLETTR"
 *      6    2   record version, 1
 *      8    4   flags: bit 0 the run's state (offsets 16 to 143), bit 1 a reader's place (120),
 *               bit 2 the best epoch's parameters (after offset 152)
 *     12    4   reserved, 0
 *     16    8   the network's optimizer steps at the end of the run's last epoch
 *     24    8   the run's epochs completed          32  8  its optimizer steps
 *     40    8   the steps of all its epochs (0: not yet known)
 *     48    8   the dropout seed                    56  8  the augmentation seed
 *     64    8   the seed of the epochs' orders
 *     72    4   beta1 to the steps, as multiplied   76  4  beta2 likewise
 *     80    4   the schedule's last learning rate   84  4  the plateau's scale
 *     88    4   the best monitored value            92  4  reserved, 0
 *     96    8   the best epoch (0: none)           104  8  epochs without improvement
 *    112    8   epochs since the plateau's last reduction
 *    120   24   the reader's seed, pass and position (SpingalettDatasetPosition)
 *    144    8   n, floats of the best epoch's parameters (weights, biases, running means and variances)
 *    152   4n   those parameters
 *  152+4n   4   CRC-32 of the record's bytes before it
 */
#define RUN_MAGIC     "SLETTR"
#define RUN_VERSION   1u
#define RUN_HEADER    152u
#define RUN_STATE     1u
#define RUN_READER    2u
#define RUN_BEST      4u

static void put_float(uint8_t *p, float v) { uint32_t u; memcpy(&u, &v, 4); slett_put32(p, u); }
static float get_float(const uint8_t *p) { uint32_t u = slett_get32(p); float v; memcpy(&v, &u, 4); return v; }

bool spingalett_save_checkpoint(NeuralNetwork *net, const char *path) {
    if (!net || !path) {
        set_error(SPINGALETT_ERR_INVALID, "checkpoint: net or path is NULL");
        return false;
    }
    /* a run's state while the network stands where the run's last epoch left it */
    const bool state = net->run.valid && net->run.time_step == net->time_step;
    if (net->run.valid && !state)
        spingalett_log(LOG_WARNING, "Checkpoint: the network has taken steps since its run's last epoch, whose "
                       "state is left out");
    const float *best = !state || !net->run.best_epoch || net->run.best_epoch == net->run.epochs ? NULL
                      : net->run.best_params ? net->run.best_params : net->run.saved_best;
    const uint64_t n = best ? net->total_weights + 3u * net->total_biases : 0u;
    size_t image_size = 0;
    uint8_t *image = (uint8_t *)spingalett_save_to_memory(net, PRECISION_FLOAT32, true, &image_size);
    if (!image) return false;
    const size_t record = RUN_HEADER + (size_t)n * sizeof(float) + 4u;
    uint8_t *r = (uint8_t *)calloc(1, record);
    if (!r) {
        spingalett_free(image);
        set_error(SPINGALETT_ERR_ALLOC, "checkpoint: out of memory");
        return false;
    }
    memcpy(r, RUN_MAGIC, 6);
    slett_put16(r + 6, RUN_VERSION);
    if (state) {
        const bool reader = net->run.has_reader;
        slett_put32(r + 8, RUN_STATE | (reader ? RUN_READER : 0u) | (best ? RUN_BEST : 0u));
        slett_put64(r + 16, net->run.time_step);
        slett_put64(r + 24, net->run.epochs);
        slett_put64(r + 32, net->run.steps);
        slett_put64(r + 40, net->run.total_steps);
        slett_put64(r + 48, net->run.dropout_seed);
        slett_put64(r + 56, net->run.augment_seed);
        slett_put64(r + 64, net->run.order_seed);
        put_float(r + 72, net->run.beta1_pow);
        put_float(r + 76, net->run.beta2_pow);
        put_float(r + 80, net->run.base_lr);
        put_float(r + 84, net->run.lr_scale);
        put_float(r + 88, net->run.best);
        slett_put64(r + 96, net->run.best_epoch);
        slett_put64(r + 104, net->run.no_improvement);
        slett_put64(r + 112, net->run.stale);
        if (reader) {
            slett_put64(r + 120, net->run.reader.seed);
            slett_put64(r + 128, net->run.reader.pass);
            slett_put64(r + 136, net->run.reader.position);
        }
        slett_put64(r + 144, n);
        if (best) memcpy(r + RUN_HEADER, best, (size_t)n * sizeof(float));
    }
    slett_put32(r + record - 4u, spingalett_crc32(0, r, record - 4u));

    bool ok = false;
    FILE *fp = fopen(path, "wb");
    if (!fp) {
        set_error(SPINGALETT_ERR_FILE_IO, "checkpoint: cannot open the file for writing");
    } else {
        ok = fwrite(image, 1, image_size, fp) == image_size && fwrite(r, 1, record, fp) == record;
        if (fclose(fp) != 0) ok = false;
        if (!ok) set_error(SPINGALETT_ERR_FILE_IO, "checkpoint: write error (disk full?)");
    }
    free(r);
    spingalett_free(image);
    if (ok)
        spingalett_log(LOG_INFO, "Checkpoint saved to %s: %zu bytes of model, %s", path, image_size,
                       state ? "the run's state after its epoch" : "no run's state");
    else
        spingalett_log(LOG_ERROR, "Write error saving a checkpoint to %s", path);
    return ok;
}

NeuralNetwork *spingalett_load_checkpoint(const char *path) {
    if (!path) {
        set_error(SPINGALETT_ERR_INVALID, "checkpoint: path is NULL");
        return NULL;
    }
    size_t size = 0;
    uint8_t *data = (uint8_t *)spingalett_read_file(path, &size);
    if (!data) return NULL;
    /* the record, after the image's last byte; a file that ends there is a model */
    uint64_t image = size >= SLETT_HEADER_SIZE && !memcmp(data, SLETT_MAGIC, 6) && slett_get16(data + 6) >= 3u
                   ? slett_get64(data + 24) : size;
    if (image > size) image = size;                 /* a file cut short: the loader says so */
    const uint8_t *r = image < size ? data + image : NULL;
    const size_t record = r ? size - (size_t)image : 0u;
    uint32_t flags = 0;
    uint64_t n = 0;
    if (r) {
        bool sound = record >= RUN_HEADER + 4u && !memcmp(r, RUN_MAGIC, 6);
        if (sound) {
            flags = slett_get32(r + 8);
            n = slett_get64(r + 144);
            sound = slett_get16(r + 6) == RUN_VERSION && n <= (record - RUN_HEADER - 4u) / sizeof(float) &&
                    record == RUN_HEADER + n * sizeof(float) + 4u && ((flags & RUN_BEST) != 0) == (n > 0) &&
                    slett_get32(r + record - 4u) == spingalett_crc32(0, r, record - 4u);
        }
        if (!sound) {
            spingalett_aligned_free(data);
            set_error(SPINGALETT_ERR_INVALID, "checkpoint: the bytes after the model are no sound record of a run");
            spingalett_log(LOG_ERROR, "Could not load %s: %s", path, spingalett_last_error_message());
            return NULL;
        }
    }
    NeuralNetwork *net = spingalett_load_from_memory_ex(data, (size_t)image, NULL);
    if (net && (flags & RUN_STATE)) {
        if (n && n != net->total_weights + 3u * net->total_biases) {
            set_error(SPINGALETT_ERR_INVALID, "checkpoint: the best epoch's parameters are not the network's");
            free_network(net);
            net = NULL;
        } else if (n && !(net->run.saved_best = (float *)malloc((size_t)n * sizeof(float)))) {
            set_error(SPINGALETT_ERR_ALLOC, "checkpoint: out of memory");
            free_network(net);
            net = NULL;
        }
    }
    if (net && (flags & RUN_STATE)) {
        if (n) memcpy(net->run.saved_best, r + RUN_HEADER, (size_t)n * sizeof(float));
        net->run.valid = net->run.resume = true;
        net->run.has_reader = (flags & RUN_READER) != 0;
        net->run.time_step = slett_get64(r + 16);
        net->run.epochs = slett_get64(r + 24);
        net->run.steps = slett_get64(r + 32);
        net->run.total_steps = slett_get64(r + 40);
        net->run.dropout_seed = slett_get64(r + 48);
        net->run.augment_seed = slett_get64(r + 56);
        net->run.order_seed = slett_get64(r + 64);
        net->run.beta1_pow = get_float(r + 72);
        net->run.beta2_pow = get_float(r + 76);
        net->run.base_lr = get_float(r + 80);
        net->run.lr_scale = get_float(r + 84);
        net->run.best = get_float(r + 88);
        net->run.best_epoch = slett_get64(r + 96);
        net->run.no_improvement = slett_get64(r + 104);
        net->run.stale = slett_get64(r + 112);
        net->run.reader = (SpingalettDatasetPosition){
            .seed = slett_get64(r + 120), .pass = slett_get64(r + 128), .position = slett_get64(r + 136)};
    }
    spingalett_aligned_free(data);
    if (!net) {
        spingalett_log(LOG_ERROR, "Could not load %s: %s", path, spingalett_last_error_message());
        return NULL;
    }
    if (net->run.resume)
        spingalett_log(LOG_INFO, "Checkpoint loaded from %s: a run after its epoch %llu, %llu steps", path,
                       (unsigned long long)net->run.epochs, (unsigned long long)net->time_step);
    else
        spingalett_log(LOG_INFO, "Checkpoint loaded from %s without a run's state", path);
    return net;
}

bool spingalett_autosave(const SaveArgs *args) {
    if (args->precision != PRECISION_FLOAT32 || args->do_not_save_optimizer)
        return spingalett_save_args(*args);
    char *allocated = NULL;
    const char *name = model_filename(args->filename, &allocated);
    bool ok = name && spingalett_save_checkpoint(args->net, name);
    free(allocated);
    return ok;
}
