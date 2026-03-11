/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

#include "Spingalett.Private.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

static uint16_t float_to_fp16(float x) {
    uint32_t f;
    memcpy(&f, &x, sizeof(float));

    uint32_t sign  = (f >> 16) & 0x8000u;
    uint32_t f_exp = (f >> 23) & 0xFFu;
    uint32_t f_man = f & 0x007FFFFFu;

    if (f_exp == 0)
        return (uint16_t)sign;

    if (f_exp == 255) {
        if (f_man == 0) return (uint16_t)(sign | 0x7C00u);
        return (uint16_t)(sign | 0x7E00u);
    }

    int32_t exp = (int32_t)f_exp - 127;

    if (exp > 15)
        return (uint16_t)(sign | 0x7C00u);

    if (exp < -24)
        return (uint16_t)sign;

    if (exp < -14) {
        uint32_t full = 0x00800000u | f_man;
        int32_t shift = -1 - exp;
        return (uint16_t)(sign | (uint16_t)(full >> shift));
    }

    return (uint16_t)(sign | ((uint32_t)(exp + 15) << 10) | (f_man >> 13));
}

static float fp16_to_float(uint16_t h) {
    uint32_t sign  = ((uint32_t)(h & 0x8000u)) << 16;
    uint32_t h_exp = (h >> 10) & 0x1Fu;
    uint32_t h_man = h & 0x03FFu;
    uint32_t f;

    if (h_exp == 0) {
        if (h_man == 0) {
            f = sign;
        } else {
            h_exp = 1;
            while (!(h_man & 0x0400u)) {
                h_man <<= 1;
                h_exp++;
            }
            h_man &= 0x03FFu;
            f = sign | ((uint32_t)(114 - h_exp) << 23) | (h_man << 13);
        }
    } else if (h_exp == 31) {
        f = (h_man == 0) ? (sign | 0x7F800000u) : (sign | 0x7FC00000u);
    } else {
        f = sign | ((h_exp + 112) << 23) | (h_man << 13);
    }

    float result;
    memcpy(&result, &f, sizeof(float));
    return result;
}

static uint16_t float_to_bf16(float x) {
    uint32_t f;
    memcpy(&f, &x, sizeof(float));
    return (uint16_t)(f >> 16);
}

static float bf16_to_float(uint16_t h) {
    uint32_t f = ((uint32_t)h) << 16;
    float x;
    memcpy(&x, &f, sizeof(float));
    return x;
}

static bool write_array_compressed(float *data, uint32_t size, PrecisionMode precision, FILE *fp) {
    if (precision == PRECISION_FLOAT32) {
        return fwrite(data, sizeof(float), size, fp) == size;
    }

    if (precision == PRECISION_FP16) {
        uint16_t *buf = (uint16_t *)malloc(size * sizeof(uint16_t));
        if (!buf) { set_error(SPINGALETT_ERR_ALLOC, "FP16 write buffer allocation failed"); return false; }
        for (uint32_t i = 0; i < size; i++) buf[i] = float_to_fp16(data[i]);
        bool ok = fwrite(buf, sizeof(uint16_t), size, fp) == size;
        free(buf);
        return ok;
    }

    if (precision == PRECISION_BFLOAT16) {
        uint16_t *buf = (uint16_t *)malloc(size * sizeof(uint16_t));
        if (!buf) { set_error(SPINGALETT_ERR_ALLOC, "BF16 write buffer allocation failed"); return false; }
        for (uint32_t i = 0; i < size; i++) buf[i] = float_to_bf16(data[i]);
        bool ok = fwrite(buf, sizeof(uint16_t), size, fp) == size;
        free(buf);
        return ok;
    }

    if (precision == PRECISION_INT8) {
        float max_val = 0.0f;
        for (uint32_t i = 0; i < size; i++) {
            float a = fabsf(data[i]);
            if (a > max_val) max_val = a;
        }
        if (fwrite(&max_val, sizeof(float), 1, fp) != 1) return false;

        int8_t *buf = (int8_t *)calloc(size, sizeof(int8_t));
        if (!buf) { set_error(SPINGALETT_ERR_ALLOC, "INT8 write buffer allocation failed"); return false; }
        if (max_val > 0.0f) {
            for (uint32_t i = 0; i < size; i++)
                buf[i] = (int8_t)roundf((data[i] / max_val) * 127.0f);
        }
        bool ok = fwrite(buf, sizeof(int8_t), size, fp) == size;
        free(buf);
        return ok;
    }

    if (precision == PRECISION_INT4) {
        float max_val = 0.0f;
        for (uint32_t i = 0; i < size; i++) {
            float a = fabsf(data[i]);
            if (a > max_val) max_val = a;
        }
        if (fwrite(&max_val, sizeof(float), 1, fp) != 1) return false;

        uint32_t byte_count = (size + 1) / 2;
        uint8_t *buf = (uint8_t *)calloc(byte_count, sizeof(uint8_t));
        if (!buf) { set_error(SPINGALETT_ERR_ALLOC, "INT4 write buffer allocation failed"); return false; }

        if (max_val > 0.0f) {
            for (uint32_t i = 0; i < size; i++) {
                float scaled = (data[i] / max_val) * 7.0f;
                int8_t q = (int8_t)roundf(scaled);
                if (q > 7) q = 7;
                if (q < -8) q = -8;
                uint8_t uq = (uint8_t)(q & 0x0Fu);
                uint32_t bi = i / 2;
                if ((i & 1u) == 0u)
                    buf[bi] |= uq;
                else
                    buf[bi] |= (uint8_t)(uq << 4);
            }
        }

        bool ok = fwrite(buf, sizeof(uint8_t), byte_count, fp) == byte_count;
        free(buf);
        return ok;
    }

    if (precision == PRECISION_INT2) {
        float max_val = 0.0f;
        for (uint32_t i = 0; i < size; i++) {
            float a = fabsf(data[i]);
            if (a > max_val) max_val = a;
        }
        if (fwrite(&max_val, sizeof(float), 1, fp) != 1) return false;

        uint32_t byte_count = (size + 3) / 4;
        uint8_t *buf = (uint8_t *)calloc(byte_count, sizeof(uint8_t));
        if (!buf) { set_error(SPINGALETT_ERR_ALLOC, "INT2 write buffer allocation failed"); return false; }

        if (max_val > 0.0f) {
            for (uint32_t i = 0; i < size; i++) {
                float scaled = data[i] / max_val;
                int8_t q;
                if (scaled > 0.5f) q = 1;
                else if (scaled < -0.5f) q = -1;
                else q = 0;
                uint8_t uq = (uint8_t)((uint8_t)q & 0x03u);
                uint32_t bi = i / 4;
                uint8_t shift = (uint8_t)((i % 4) * 2u);
                buf[bi] |= (uint8_t)(uq << shift);
            }
        }

        bool ok = fwrite(buf, sizeof(uint8_t), byte_count, fp) == byte_count;
        free(buf);
        return ok;
    }

    return false;
}

static bool read_array_compressed(float *data, uint32_t size, PrecisionMode precision, FILE *fp) {
    if (precision == PRECISION_FLOAT32) {
        return fread(data, sizeof(float), size, fp) == size;
    }

    if (precision == PRECISION_FP16) {
        uint16_t *buf = (uint16_t *)malloc(size * sizeof(uint16_t));
        if (!buf) { set_error(SPINGALETT_ERR_ALLOC, "FP16 read buffer allocation failed"); return false; }
        bool ok = fread(buf, sizeof(uint16_t), size, fp) == size;
        if (ok) {
            for (uint32_t i = 0; i < size; i++) data[i] = fp16_to_float(buf[i]);
        }
        free(buf);
        return ok;
    }

    if (precision == PRECISION_BFLOAT16) {
        uint16_t *buf = (uint16_t *)malloc(size * sizeof(uint16_t));
        if (!buf) { set_error(SPINGALETT_ERR_ALLOC, "BF16 read buffer allocation failed"); return false; }
        bool ok = fread(buf, sizeof(uint16_t), size, fp) == size;
        if (ok) {
            for (uint32_t i = 0; i < size; i++) data[i] = bf16_to_float(buf[i]);
        }
        free(buf);
        return ok;
    }

    if (precision == PRECISION_INT8) {
        float max_val = 0.0f;
        if (fread(&max_val, sizeof(float), 1, fp) != 1) return false;

        int8_t *buf = (int8_t *)malloc(size * sizeof(int8_t));
        if (!buf) { set_error(SPINGALETT_ERR_ALLOC, "INT8 read buffer allocation failed"); return false; }
        bool ok = fread(buf, sizeof(int8_t), size, fp) == size;

        if (ok) {
            if (max_val > 0.0f) {
                for (uint32_t i = 0; i < size; i++)
                    data[i] = ((float)buf[i] / 127.0f) * max_val;
            } else {
                memset(data, 0, size * sizeof(float));
            }
        }
        free(buf);
        return ok;
    }

    if (precision == PRECISION_INT4) {
        float max_val = 0.0f;
        if (fread(&max_val, sizeof(float), 1, fp) != 1) return false;

        uint32_t byte_count = (size + 1) / 2;
        uint8_t *buf = (uint8_t *)malloc(byte_count * sizeof(uint8_t));
        if (!buf) { set_error(SPINGALETT_ERR_ALLOC, "INT4 read buffer allocation failed"); return false; }
        bool ok = fread(buf, sizeof(uint8_t), byte_count, fp) == byte_count;

        if (ok) {
            if (max_val > 0.0f) {
                for (uint32_t i = 0; i < size; i++) {
                    uint32_t bi = i / 2;
                    uint8_t packed = buf[bi];
                    uint8_t uq = ((i & 1u) == 0u) ? (packed & 0x0Fu) : (packed >> 4);
                    int8_t q = (uq & 0x08u) ? (int8_t)(uq | 0xF0u) : (int8_t)uq;
                    data[i] = ((float)q / 7.0f) * max_val;
                }
            } else {
                memset(data, 0, size * sizeof(float));
            }
        }

        free(buf);
        return ok;
    }

    if (precision == PRECISION_INT2) {
        float max_val = 0.0f;
        if (fread(&max_val, sizeof(float), 1, fp) != 1) return false;

        uint32_t byte_count = (size + 3) / 4;
        uint8_t *buf = (uint8_t *)malloc(byte_count * sizeof(uint8_t));
        if (!buf) { set_error(SPINGALETT_ERR_ALLOC, "INT2 read buffer allocation failed"); return false; }
        bool ok = fread(buf, sizeof(uint8_t), byte_count, fp) == byte_count;

        if (ok) {
            if (max_val > 0.0f) {
                for (uint32_t i = 0; i < size; i++) {
                    uint32_t bi = i / 4;
                    uint8_t shift = (uint8_t)((i % 4) * 2u);
                    uint8_t uq = (buf[bi] >> shift) & 0x03u;
                    int8_t q = (uq & 0x02u) ? (int8_t)(uq | 0xFCu) : (int8_t)uq;
                    data[i] = (float)q * max_val;
                }
            } else {
                memset(data, 0, size * sizeof(float));
            }
        }

        free(buf);
        return ok;
    }

    return false;
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

    if ((unsigned)args.precision >= PRECISION_COUNT) {
        set_error(SPINGALETT_ERR_INVALID, "save: invalid precision mode");
        return;
    }

    char *allocated_filename = NULL;
    const char *target_filename = args.filename;

    const char *dot = strrchr(target_filename, '.');
    const char *last_sep = find_last_separator(target_filename);

    bool has_ext = (dot && (!last_sep || dot > last_sep));

    if (!has_ext) {
        size_t len = strlen(target_filename);
        allocated_filename = (char *)malloc(len + 4);
        if (!allocated_filename) {
            set_error(SPINGALETT_ERR_ALLOC, "save: filename allocation failed");
            return;
        }
        strcpy(allocated_filename, target_filename);
        strcat(allocated_filename, ".nn");
        target_filename = allocated_filename;
    }

    FILE *fp = fopen(target_filename, "wb");
    if (!fp) {
        set_error(SPINGALETT_ERR_INVALID, "save: cannot open file for writing");
        free(allocated_filename);
        return;
    }

    NeuralNetwork *net = args.net;
    bool write_ok = true;

    uint16_t format_version = SPINGALETT_FORMAT_VERSION;
    if (fwrite(&format_version, sizeof(uint16_t), 1, fp) != 1) write_ok = false;

    if (write_ok && fwrite(&net->layers, sizeof(uint32_t), 1, fp) != 1) write_ok = false;

    uint8_t loss_type = (uint8_t)net->loss_func;
    if (write_ok && fwrite(&loss_type, sizeof(uint8_t), 1, fp) != 1) write_ok = false;

    uint8_t has_optimizer = args.do_not_save_optimizer ? 0 : 1;
    if (write_ok && fwrite(&has_optimizer, sizeof(uint8_t), 1, fp) != 1) write_ok = false;

    if (write_ok && has_optimizer) {
        uint64_t ts = net->time_step;
        if (fwrite(&ts, sizeof(uint64_t), 1, fp) != 1) write_ok = false;
    }

    uint8_t precision_type = (uint8_t)args.precision;
    if (write_ok && fwrite(&precision_type, sizeof(uint8_t), 1, fp) != 1) write_ok = false;

    if (write_ok && fwrite(net->topology, sizeof(uint32_t), net->layers, fp) != net->layers) write_ok = false;

    for (uint32_t i = 0; i < net->layers - 1 && write_ok; i++) {
        uint8_t act = (uint8_t)net->act_func[i];
        if (fwrite(&act, sizeof(uint8_t), 1, fp) != 1) write_ok = false;
    }

    if (!write_ok) {
        set_error(SPINGALETT_ERR_INVALID, "save: failed to write header");
        spingalett_log(LOG_ERROR, "Failed to write file header to %s", target_filename);
        fclose(fp);
        free(allocated_filename);
        return;
    }

    uint64_t total_weights = 0;
    uint64_t total_biases = 0;

    for (uint32_t l = 0; l < net->layers - 1 && write_ok; l++) {
        uint32_t in_dim  = net->topology[l];
        uint32_t out_dim = net->topology[l + 1];
        uint64_t woff    = net->weight_offsets[l];
        uint64_t boff    = net->bias_offsets[l];
        uint32_t wcount  = in_dim * out_dim;

        total_weights += wcount;
        total_biases  += out_dim;

        if (!write_array_compressed(net->weights + woff, wcount, args.precision, fp)) write_ok = false;

        if (write_ok && has_optimizer) {
            if (!write_array_compressed(net->opt_m_weights + woff, wcount, args.precision, fp)) write_ok = false;
            if (write_ok && !write_array_compressed(net->opt_v_weights + woff, wcount, args.precision, fp)) write_ok = false;
        }

        if (write_ok && !write_array_compressed(net->biases + boff, out_dim, args.precision, fp)) write_ok = false;
        if (write_ok && has_optimizer) {
            if (!write_array_compressed(net->opt_m_biases + boff, out_dim, args.precision, fp)) write_ok = false;
            if (write_ok && !write_array_compressed(net->opt_v_biases + boff, out_dim, args.precision, fp)) write_ok = false;
        }
    }

    fclose(fp);

    if (!write_ok) {
        set_error(SPINGALETT_ERR_INVALID, "save: write error (disk full?)");
        spingalett_log(LOG_ERROR, "Write error saving to %s", target_filename);
    } else {
        spingalett_log(LOG_INFO, "Network saved to %s", target_filename);
        spingalett_log(LOG_INFO, "Save info: format=v%u, layers=%u, weights=%llu, biases=%llu, precision=%s, optimizer=%s",
            (unsigned)format_version, net->layers,
            (unsigned long long)total_weights, (unsigned long long)total_biases,
            precision_names[precision_type],
            has_optimizer ? "ON" : "OFF");
    }

    free(allocated_filename);
}

NeuralNetwork *load_spingalett(const char *filename) {
    if (!filename) {
        set_error(SPINGALETT_ERR_INVALID, "load: filename is NULL");
        return NULL;
    }

    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        set_error(SPINGALETT_ERR_INVALID, "load: cannot open file for reading");
        return NULL;
    }

    uint16_t format_version = 0;
    if (fread(&format_version, sizeof(uint16_t), 1, fp) != 1) {
        set_error(SPINGALETT_ERR_INVALID, "load: failed to read format version");
        fclose(fp);
        return NULL;
    }

    if (format_version != SPINGALETT_FORMAT_VERSION) {
        set_error(SPINGALETT_ERR_INVALID, "load: unsupported format version");
        spingalett_log(LOG_ERROR, "Unsupported file format version %u (expected %u)", (unsigned)format_version, (unsigned)SPINGALETT_FORMAT_VERSION);
        fclose(fp);
        return NULL;
    }

    uint32_t layers = 0;
    if (fread(&layers, sizeof(uint32_t), 1, fp) != 1) {
        set_error(SPINGALETT_ERR_INVALID, "load: failed to read layer count");
        fclose(fp);
        return NULL;
    }

    if (layers < 2) {
        set_error(SPINGALETT_ERR_INVALID, "load: network must have at least 2 layers");
        fclose(fp);
        return NULL;
    }

    uint8_t loss_type = 0;
    if (fread(&loss_type, sizeof(uint8_t), 1, fp) != 1 || loss_type >= LOSS_COUNT) {
        set_error(SPINGALETT_ERR_INVALID, "load: invalid loss type");
        fclose(fp);
        return NULL;
    }

    uint8_t has_optimizer = 0;
    if (fread(&has_optimizer, sizeof(uint8_t), 1, fp) != 1) {
        set_error(SPINGALETT_ERR_INVALID, "load: failed to read optimizer flag");
        fclose(fp);
        return NULL;
    }

    uint64_t ts = 0;
    if (has_optimizer) {
        if (fread(&ts, sizeof(uint64_t), 1, fp) != 1) {
            set_error(SPINGALETT_ERR_INVALID, "load: failed to read time step");
            fclose(fp);
            return NULL;
        }
    }

    uint8_t precision_type = 0;
    if (fread(&precision_type, sizeof(uint8_t), 1, fp) != 1 || precision_type >= PRECISION_COUNT) {
        set_error(SPINGALETT_ERR_INVALID, "load: invalid precision type");
        fclose(fp);
        return NULL;
    }
    PrecisionMode precision = (PrecisionMode)precision_type;

    uint32_t *topology = (uint32_t *)malloc(sizeof(uint32_t) * layers);
    if (!topology) {
        set_error(SPINGALETT_ERR_ALLOC, "load: topology allocation failed");
        fclose(fp);
        return NULL;
    }

    if (fread(topology, sizeof(uint32_t), layers, fp) != layers) {
        set_error(SPINGALETT_ERR_INVALID, "load: failed to read topology");
        free(topology);
        fclose(fp);
        return NULL;
    }

    ActivationFunction *act_func_array = (ActivationFunction *)malloc(sizeof(ActivationFunction) * (layers - 1));
    if (!act_func_array) {
        set_error(SPINGALETT_ERR_ALLOC, "load: activation function array allocation failed");
        free(topology);
        fclose(fp);
        return NULL;
    }

    for (uint32_t i = 0; i < layers - 1; i++) {
        uint8_t act = 0;
        if (fread(&act, sizeof(uint8_t), 1, fp) != 1 || act >= ACT_COUNT) {
            set_error(SPINGALETT_ERR_INVALID, "load: invalid activation function");
            free(topology);
            free(act_func_array);
            fclose(fp);
            return NULL;
        }
        act_func_array[i] = (ActivationFunction)act;
    }

    uint64_t total_weights = 0;
    uint64_t total_biases = 0;
    for (uint32_t l = 0; l < layers - 1; l++) {
        total_weights += (uint64_t)topology[l] * (uint64_t)topology[l + 1];
        total_biases  += (uint64_t)topology[l + 1];
    }

    NeuralNetwork *net = new_spingalett_struct_arguments((NeuralNetworkArgs){ .loss_func = (LossFunction)loss_type });
    if (!net) {
        set_error(SPINGALETT_ERR_ALLOC, "load: network creation failed");
        free(topology);
        free(act_func_array);
        fclose(fp);
        return NULL;
    }
    net->time_step = ts;

    for (uint32_t l = 0; l < layers; l++) {
        LayerArgs largs = {0};
        largs.net = net;
        largs.neurons_amount = topology[l];
        if (l > 0)
            largs.act_func = act_func_array[l - 1];
        largs.weight_initialization = WEIGHT_INITIALIZATION_NONE;
        layer_struct_arguments(largs);

        if (spingalett_last_error_code() != SPINGALETT_OK) {
            free(topology);
            free(act_func_array);
            free_network(net);
            fclose(fp);
            return NULL;
        }
    }

    free(topology);

    bool read_ok = true;
    for (uint32_t l = 0; l < net->layers - 1 && read_ok; l++) {
        uint32_t in_dim  = net->topology[l];
        uint32_t out_dim = net->topology[l + 1];
        uint64_t woff    = net->weight_offsets[l];
        uint64_t boff    = net->bias_offsets[l];
        uint32_t wcount  = in_dim * out_dim;

        read_ok = read_array_compressed(net->weights + woff, wcount, precision, fp);

        if (has_optimizer && read_ok) {
            read_ok = read_array_compressed(net->opt_m_weights + woff, wcount, precision, fp);
            if (read_ok)
                read_ok = read_array_compressed(net->opt_v_weights + woff, wcount, precision, fp);
        }

        if (read_ok)
            read_ok = read_array_compressed(net->biases + boff, out_dim, precision, fp);

        if (has_optimizer && read_ok) {
            read_ok = read_array_compressed(net->opt_m_biases + boff, out_dim, precision, fp);
            if (read_ok)
                read_ok = read_array_compressed(net->opt_v_biases + boff, out_dim, precision, fp);
        }
    }

    free(act_func_array);
    fclose(fp);

    if (!read_ok) {
        set_error(SPINGALETT_ERR_INVALID, "load: file appears truncated or corrupt");
        spingalett_log(LOG_WARNING, "Network file may be truncated: %s", filename);
    }

    spingalett_log(LOG_INFO, "Network loaded from %s", filename);
    spingalett_log(LOG_INFO, "Load info: format=v%u, layers=%u, weights=%llu, biases=%llu, precision=%s, optimizer=%s",
        (unsigned)format_version, layers,
        (unsigned long long)total_weights, (unsigned long long)total_biases,
        precision_type < PRECISION_COUNT ? precision_names[precision_type] : "UNKNOWN",
        has_optimizer ? "ON" : "OFF");

    return net;
}
