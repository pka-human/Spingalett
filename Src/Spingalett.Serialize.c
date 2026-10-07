/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/*
 * .slett model files. Writing produces format version 3 (docs/ModelFormat.md): a header, a layer
 * table and 16-byte aligned sections, little-endian, with per-row scales for the integer precisions
 * and CRC-32 checksums, so that a file image doubles as an in-place inference model. Reading also
 * accepts versions 1 and 2 (native byte order, one scale per tensor).
 */

#include "Spingalett.Private.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>

/* ------------------------------------------------------------------------- writing (format 3) */

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
        case PRECISION_FP16:
            for (uint32_t i = 0; i < n; i++) slett_put16(dst + 2u * i, spingalett_float_to_fp16(w[i]));
            return 0.0f;
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

void *spingalett_save_to_memory(const NeuralNetwork *net, PrecisionMode precision, bool save_optimizer, size_t *size) {
    if (size) *size = 0;
    if (!net || !size) {
        set_error(SPINGALETT_ERR_INVALID, "save: net or size is NULL");
        return NULL;
    }
    if (net->layers < 2 || net->layers > SLETT_MAX_LAYERS) {
        set_error(SPINGALETT_ERR_INVALID, "save: network must have 2 to 65536 layers");
        return NULL;
    }
    if (!spingalett_all_dense(net)) {     /* format version 4: next commit */
        set_error(SPINGALETT_ERR_INVALID, "save: convolution and pooling layers cannot be saved yet");
        return NULL;
    }
    if ((unsigned)precision >= PRECISION_COUNT) {
        set_error(SPINGALETT_ERR_INVALID, "save: invalid precision mode");
        return NULL;
    }
    if (!spingalett_host_is_little_endian()) {
        set_error(SPINGALETT_ERR_INVALID, "save: format 3 needs a little-endian host");
        return NULL;
    }
    uint32_t L = net->layers - 1;              /* weight layers */
    if (spingalett_precision_is_int(precision))
        for (uint32_t l = 0; l < L; l++)
            if (net->topology[l] > SLETT_MAX_INT_INPUTS) {
                set_error(SPINGALETT_ERR_INVALID, "save: integer precisions allow at most 131072 inputs per layer");
                return NULL;
            }

    uint64_t *off = (uint64_t *)malloc((size_t)L * 4u * sizeof(uint64_t));   /* weights, scales, biases, optimizer */
    if (!off) {
        set_error(SPINGALETT_ERR_ALLOC, "save: allocation failed");
        return NULL;
    }
    uint64_t pos = slett_align(SLETT_HEADER_SIZE + (uint64_t)L * SLETT_LAYER_ENTRY_SIZE);
    for (uint32_t l = 0; l < L; l++) {
        uint32_t in = net->topology[l], out = net->topology[l + 1];
        off[4 * l] = pos;
        pos = slett_align(pos + spingalett_slett_row_bytes(precision, in) * out);
        off[4 * l + 1] = 0;
        if (spingalett_precision_is_int(precision)) {
            off[4 * l + 1] = pos;
            pos = slett_align(pos + (uint64_t)out * 4u);
        }
        off[4 * l + 2] = pos;
        pos = slett_align(pos + (uint64_t)out * 4u);
        off[4 * l + 3] = 0;
        if (save_optimizer) {
            off[4 * l + 3] = pos;
            pos = slett_align(pos + ((uint64_t)in * out * 2u + (uint64_t)out * 2u) * 4u);
        }
    }
    if (pos > (uint64_t)SIZE_MAX - SPINGALETT_ALIGNMENT) {
        free(off);
        set_error(SPINGALETT_ERR_ALLOC, "save: network too large for memory");
        return NULL;
    }
    uint8_t *img = (uint8_t *)spingalett_aligned_calloc((size_t)pos, 1);
    if (!img) {
        free(off);
        set_error(SPINGALETT_ERR_ALLOC, "save: image allocation failed");
        return NULL;
    }

    memcpy(img, SLETT_MAGIC, 6);
    slett_put16(img + 6, 3);
    slett_put32(img + 8, net->layers);
    img[12] = (uint8_t)net->loss_func;
    img[13] = save_optimizer ? SLETT_FLAG_OPTIMIZER : 0u;
    slett_put64(img + 16, save_optimizer ? net->time_step : 0u);
    slett_put64(img + 24, pos);

    for (uint32_t l = 0; l < L; l++) {
        uint32_t in = net->topology[l], out = net->topology[l + 1];
        uint8_t *e = img + SLETT_HEADER_SIZE + (size_t)l * SLETT_LAYER_ENTRY_SIZE;
        slett_put32(e, in);
        slett_put32(e + 4, out);
        e[8] = (uint8_t)net->act_func[l];
        e[9] = (uint8_t)precision;
        uint32_t dropout;
        memcpy(&dropout, &net->dropout_rates[l + 1], 4);
        slett_put32(e + 12, dropout);
        for (int k = 0; k < 4; k++) slett_put64(e + 16 + 8 * k, off[4 * l + k]);

        const float *W = net->weights + net->weight_offsets[l];
        size_t row = (size_t)spingalett_slett_row_bytes(precision, in);
        for (uint32_t j = 0; j < out; j++) {
            float scale = quantize_row(W + (size_t)j * in, in, precision, img + off[4 * l] + (size_t)j * row);
            if (off[4 * l + 1]) memcpy(img + off[4 * l + 1] + 4u * (size_t)j, &scale, 4);
        }
        memcpy(img + off[4 * l + 2], net->biases + net->bias_offsets[l], (size_t)out * 4u);
        if (save_optimizer) {
            uint8_t *o = img + off[4 * l + 3];
            size_t wbytes = (size_t)in * out * 4u, bbytes = (size_t)out * 4u;
            memcpy(o, net->opt_m_weights + net->weight_offsets[l], wbytes);
            memcpy(o + wbytes, net->opt_v_weights + net->weight_offsets[l], wbytes);
            memcpy(o + 2u * wbytes, net->opt_m_biases + net->bias_offsets[l], bbytes);
            memcpy(o + 2u * wbytes + bbytes, net->opt_v_biases + net->bias_offsets[l], bbytes);
        }
    }
    free(off);

    slett_put32(img + 56, spingalett_crc32(0, img + SLETT_HEADER_SIZE, (size_t)pos - SLETT_HEADER_SIZE));
    slett_put32(img + 60, spingalett_crc32(0, img, 60));
    *size = (size_t)pos;
    return img;
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
                (unsigned)SPINGALETT_FORMAT_VERSION, args.net->layers,
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

static NeuralNetwork *load_v3(const uint8_t *p, size_t size, PrecisionMode *precision) {
    SlettInfo info;
    if (spingalett_slett_validate(p, size, &info) != SPINGALETT_OK) return NULL;

    uint32_t L = info.layers - 1;
    uint32_t *topology = (uint32_t *)malloc((size_t)info.layers * sizeof(uint32_t));
    ActivationFunction *act = (ActivationFunction *)malloc((size_t)L * sizeof(ActivationFunction));
    float *dropout = (float *)calloc(info.layers, sizeof(float));
    NeuralNetwork *net = NULL;
    if (topology && act && dropout) {
        for (uint32_t l = 0; l < L; l++) {
            SlettLayer e;
            spingalett_slett_layer(p, l, &e);
            topology[l] = e.inputs;
            topology[l + 1] = e.outputs;
            act[l] = e.activation;
            dropout[l + 1] = e.dropout;
            if (l == 0) *precision = e.precision;
        }
        net = make_network(info.loss, info.layers, topology, act, dropout);
    } else {
        set_error(SPINGALETT_ERR_ALLOC, "load: allocation failed");
    }
    free(topology);
    free(act);
    free(dropout);
    if (!net) return NULL;

    int8_t *codes = NULL;
    for (uint32_t l = 0; l < L; l++) {
        SlettLayer e;
        spingalett_slett_layer(p, l, &e);
        float *W = net->weights + net->weight_offsets[l];
        const uint8_t *src = p + e.weights;
        size_t row = (size_t)spingalett_slett_row_bytes(e.precision, e.inputs);
        if (e.precision == PRECISION_INT4 || e.precision == PRECISION_INT2) {
            int8_t *grown = (int8_t *)realloc(codes, e.inputs);
            if (!grown) {
                free(codes);
                free_network(net);
                set_error(SPINGALETT_ERR_ALLOC, "load: allocation failed");
                return NULL;
            }
            codes = grown;
        }
        for (uint32_t j = 0; j < e.outputs; j++) {
            float *dst = W + (size_t)j * e.inputs;
            const uint8_t *r = src + (size_t)j * row;
            float scale = 0.0f;
            if (spingalett_precision_is_int(e.precision)) memcpy(&scale, p + e.scales + 4u * (size_t)j, 4);
            switch (e.precision) {
                case PRECISION_FLOAT32:
                    memcpy(dst, r, (size_t)e.inputs * 4u);
                    break;
                case PRECISION_FP16:
                    for (uint32_t k = 0; k < e.inputs; k++) dst[k] = spingalett_fp16_to_float(slett_get16(r + 2u * k));
                    break;
                case PRECISION_BFLOAT16:
                    for (uint32_t k = 0; k < e.inputs; k++) dst[k] = spingalett_bf16_to_float(slett_get16(r + 2u * k));
                    break;
                case PRECISION_INT8:
                    for (uint32_t k = 0; k < e.inputs; k++) dst[k] = (float)(int8_t)r[k] * scale;
                    break;
                default:
                    if (e.precision == PRECISION_INT4) spingalett_unpack_int4(r, codes, e.inputs);
                    else spingalett_unpack_int2(r, codes, e.inputs);
                    for (uint32_t k = 0; k < e.inputs; k++) dst[k] = (float)codes[k] * scale;
                    break;
            }
        }
        memcpy(net->biases + net->bias_offsets[l], p + e.biases, (size_t)e.outputs * 4u);
        if (info.flags & SLETT_FLAG_OPTIMIZER) {
            const uint8_t *o = p + e.optimizer;
            size_t wbytes = (size_t)e.inputs * e.outputs * 4u, bbytes = (size_t)e.outputs * 4u;
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
        return load_v3(p, size, precision);
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
