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
        default:
            return 0.0f;
    }
}

/* The oldest format version that holds the network: 3 for dense layers only, 4 with convolution or
   pooling layers, 5 with batch normalization or grouped convolutions, 6 for graphs. */
static uint16_t format_version(const NeuralNetwork *net) {
    if (net->graph) return 6u;
    uint16_t version = 3u;
    for (uint32_t l = 1; l < net->layers; l++) {
        const LayerShape *s = &net->shapes[l];
        if (s->type == LAYER_BATCH_NORM || (s->type == LAYER_CONV2D && s->groups > 1)) return 5u;
        if (s->type != LAYER_DENSE) version = 4u;
    }
    return version;
}

/* Batch normalization keeps its parameters and statistics in FLOAT32 whatever the file's precision. */
static inline PrecisionMode layer_precision(const NeuralNetwork *net, uint32_t l, PrecisionMode precision) {
    return net->shapes[l + 1].type == LAYER_BATCH_NORM ? PRECISION_FLOAT32 : precision;
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
    const uint16_t version = format_version(net);
    const size_t entry_size = version == 3u ? SLETT_LAYER_ENTRY_SIZE : version == 4u ? SLETT_LAYER_ENTRY_SIZE_4
                            : version == 5u ? SLETT_LAYER_ENTRY_SIZE_5 : SLETT_LAYER_ENTRY_SIZE_6;
    uint32_t L = net->layers - 1;              /* weight layers */
    for (uint32_t l = 1; l < net->layers; l++)
        if (net->shapes[l].type > LAYER_GLOBAL_AVG_POOL) {
            set_error(SPINGALETT_ERR_INVALID, "save: transposed convolutions, upsampling and layer normalization "
                                              "cannot be saved yet");
            return NULL;
        }
    for (uint32_t l = 0; l < L; l++)
        if (spingalett_precision_is_int(layer_precision(net, l, precision)) &&
            spingalett_weight_row_len(net, l) > SLETT_MAX_INT_INPUTS) {
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
        off[4 * l] = pos;
        pos = slett_align(pos + spingalett_slett_row_bytes(p, row_len) * rows);
        if (spingalett_precision_is_int(p) || net->shapes[l + 1].type == LAYER_BATCH_NORM) {
            /* row scales; batch normalization: its running mean and variance */
            off[4 * l + 1] = pos;
            pos = slett_align(pos + (uint64_t)rows * (spingalett_precision_is_int(p) ? 4u : 8u));
        }
        off[4 * l + 2] = pos;
        pos = slett_align(pos + (uint64_t)rows * 4u);
        if (save_optimizer) {
            off[4 * l + 3] = pos;
            pos = slett_align(pos + ((uint64_t)rows * row_len * 2u + (uint64_t)rows * 2u) * 4u);
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
        e[8] = (uint8_t)net->act_func[l];
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
            slett_put32(e + 64, shape->type == LAYER_CONV2D ? shape->groups : 0u);
            memcpy(&bits, &shape->eps, 4);
            slett_put32(e + 68, bits);
            memcpy(&bits, &shape->momentum, 4);
            slett_put32(e + 72, bits);
        }
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
        for (uint32_t j = 0; j < rows; j++) {
            float scale = quantize_row(W + (size_t)j * row_len, row_len, p, img + off[4 * l] + (size_t)j * row);
            if (spingalett_precision_is_int(p)) memcpy(img + off[4 * l + 1] + 4u * (size_t)j, &scale, 4);
        }
        if (shape->type == LAYER_BATCH_NORM) {
            memcpy(img + off[4 * l + 1], net->running_mean + net->bias_offsets[l], (size_t)rows * 4u);
            memcpy(img + off[4 * l + 1] + (size_t)rows * 4u, net->running_var + net->bias_offsets[l], (size_t)rows * 4u);
        }
        memcpy(img + off[4 * l + 2], net->biases + net->bias_offsets[l], (size_t)rows * 4u);
        if (save_optimizer && net->opt_m_weights) {     /* without, before any training: the zeros the image has */
            uint8_t *o = img + off[4 * l + 3];
            size_t wbytes = (size_t)rows * row_len * 4u, bbytes = (size_t)rows * 4u;
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
    if (s == 0) return false;
    LayerType type = net->shapes[s].type;
    return (type == LAYER_DENSE || type == LAYER_CONV2D) && net->act_func[s - 1] == ACT_NONE && uses[s] == 1;
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
        LayerArgs a = spingalett_layer_args(src, l);
        a.net = f;
        for (uint32_t k = 0; k < a.input_count; k++) a.inputs[k] = map[a.inputs[k]];
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
        uint64_t sw = net->weight_offsets[l - 1], sb = net->bias_offsets[l - 1];
        uint64_t dw = f->weight_offsets[k], db = f->bias_offsets[k];
        memcpy(f->weights + dw, net->weights + sw, (size_t)(rows * len) * sizeof(float));
        memcpy(f->biases + db, net->biases + sb, (size_t)rows * sizeof(float));
        memcpy(f->running_mean + db, net->running_mean + sb, (size_t)rows * sizeof(float));
        memcpy(f->running_var + db, net->running_var + sb, (size_t)rows * sizeof(float));
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

void save_spingalett_struct_arguments(SaveArgs args) {
    if (!args.net || !args.filename) {
        set_error(SPINGALETT_ERR_INVALID, "save: net or filename is NULL");
        return;
    }

    char *allocated_filename = NULL;
    const char *target_filename = args.filename;
    const char *dot = strrchr(target_filename, '.');
    const char *last_sep = find_last_separator(target_filename);
    if (!(dot && (!last_sep || dot > last_sep))) {
        size_t len = strlen(target_filename);
        allocated_filename = (char *)malloc(len + sizeof SPINGALETT_MODEL_EXTENSION);
        if (!allocated_filename) {
            set_error(SPINGALETT_ERR_ALLOC, "save: filename allocation failed");
            return;
        }
        memcpy(allocated_filename, target_filename, len);
        memcpy(allocated_filename + len, SPINGALETT_MODEL_EXTENSION, sizeof SPINGALETT_MODEL_EXTENSION);
        target_filename = allocated_filename;
    }

    size_t size = 0;
    void *img = spingalett_save_to_memory(args.net, args.precision, !args.do_not_save_optimizer, &size);
    if (!img) {
        free(allocated_filename);
        return;
    }

    FILE *fp = fopen(target_filename, "wb");
    if (!fp) {
        set_error(SPINGALETT_ERR_FILE_IO, "save: cannot open file for writing");
    } else {
        bool ok = fwrite(img, 1, size, fp) == size;
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
}

/* ------------------------------------------------------------------------- reading */

void *spingalett_read_file(const char *path, size_t *size) {
    *size = 0;
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        set_error(SPINGALETT_ERR_FILE_IO, "load: cannot open file for reading");
        return NULL;
    }
    long length = -1;
    if (fseek(fp, 0, SEEK_END) == 0) length = ftell(fp);
    if (length < 0 || fseek(fp, 0, SEEK_SET) != 0) {
        fclose(fp);
        set_error(SPINGALETT_ERR_FILE_IO, "load: cannot determine the file size");
        return NULL;
    }
    void *data = spingalett_aligned_alloc((size_t)length);
    if (!data) {
        fclose(fp);
        set_error(SPINGALETT_ERR_ALLOC, "load: file buffer allocation failed");
        return NULL;
    }
    bool ok = fread(data, 1, (size_t)length, fp) == (size_t)length;
    fclose(fp);
    if (!ok) {
        spingalett_aligned_free(data);
        set_error(SPINGALETT_ERR_FILE_IO, "load: read error");
        return NULL;
    }
    *size = (size_t)length;
    return data;
}

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
        weights += (uint64_t)e.rows * e.row_len;
        biases += e.rows;
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
        if (e.type == LAYER_DENSE) {
            a.neurons_amount = e.outputs;
        } else if (e.type == LAYER_BATCH_NORM) {
            a.epsilon = e.eps;
            a.momentum = e.momentum;
        } else if (e.type == LAYER_ADD || e.type == LAYER_CONCAT || e.type == LAYER_GLOBAL_AVG_POOL) {
            /* the inputs give the shape */
        } else {
            a.filters = e.type == LAYER_CONV2D ? e.out_c : 0;
            a.groups = e.type == LAYER_CONV2D ? e.groups : 0;
            a.kernel_h = e.kernel_h; a.kernel_w = e.kernel_w;
            a.stride_h = e.stride_h; a.stride_w = e.stride_w;
            a.padding_h = e.pad_h; a.padding_w = e.pad_w;
        }
        ok = spingalett_add_layer(a) && net->topology[l + 1] == e.outputs;
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
        if (first && (e.type == LAYER_DENSE || e.type == LAYER_CONV2D)) { *precision = e.precision; first = false; }
        if (e.rows == 0) continue;
        if (e.rows != spingalett_weight_rows(net, l) || e.row_len != spingalett_weight_row_len(net, l)) {
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
        for (uint32_t j = 0; j < e.rows; j++) {
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
                default:
                    if (e.precision == PRECISION_INT4) spingalett_unpack_int4(r, codes, e.row_len);
                    else spingalett_unpack_int2(r, codes, e.row_len);
                    for (uint32_t k = 0; k < e.row_len; k++) dst[k] = (float)codes[k] * scale;
                    break;
            }
        }
        memcpy(net->biases + net->bias_offsets[l], p + e.biases, (size_t)e.rows * 4u);
        if (e.type == LAYER_BATCH_NORM) {
            memcpy(net->running_mean + net->bias_offsets[l], p + e.scales, (size_t)e.rows * 4u);
            memcpy(net->running_var + net->bias_offsets[l], p + e.scales + (size_t)e.rows * 4u, (size_t)e.rows * 4u);
        }
        if (info.flags & SLETT_FLAG_OPTIMIZER) {
            const uint8_t *o = p + e.optimizer;
            size_t wbytes = (size_t)e.rows * e.row_len * 4u, bbytes = (size_t)e.rows * 4u;
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
        if (!take(&c, &a, 1) || a >= ACT_COUNT) { set_error(SPINGALETT_ERR_INVALID, "load: invalid activation function"); ok = false; }
        else act[i] = (ActivationFunction)a;
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

NeuralNetwork *load_spingalett_from_memory(const void *data, size_t size) {
    return spingalett_load_from_memory_ex(data, size, NULL);
}

NeuralNetwork *load_spingalett(const char *filename) {
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
