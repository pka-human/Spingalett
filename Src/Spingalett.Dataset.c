/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/* In-memory data sets: IDX, CIFAR and CSV readers, shuffling and hold-out splits. */

#include "Spingalett.Private.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

void spingalett_dataset_free(SpingalettDataset *d) {
    if (!d) return;
    free(d->inputs);
    free(d->targets);
    memset(d, 0, sizeof *d);
}

static bool dataset_alloc(SpingalettDataset *d, uint32_t count, uint32_t input_size, uint32_t target_size) {
    d->count = count;
    d->input_size = input_size;
    d->target_size = target_size;
    d->inputs = (float *)malloc(((size_t)count * input_size + 1) * sizeof(float));
    d->targets = (float *)calloc((size_t)count * target_size + 1, sizeof(float));
    if (d->inputs && d->targets) return true;
    spingalett_dataset_free(d);
    set_error(SPINGALETT_ERR_ALLOC, "dataset allocation failed");
    return false;
}

/* ---------------------------------------------------------------- CIFAR */

#define CIFAR_SIDE   32u
#define CIFAR_PIXELS (CIFAR_SIDE * CIFAR_SIDE)

bool spingalett_load_cifar(const char *const *paths, uint32_t path_count, uint32_t num_classes,
                           SpingalettDataset *dataset) {
    if (!dataset) { set_error(SPINGALETT_ERR_INVALID, "spingalett_load_cifar: dataset is NULL"); return false; }
    memset(dataset, 0, sizeof *dataset);
    if (!paths || path_count == 0 || (num_classes != 10 && num_classes != 20 && num_classes != 100)) {
        set_error(SPINGALETT_ERR_INVALID, "spingalett_load_cifar: no paths, or num_classes is not 10, 20 or 100");
        return false;
    }
    /* CIFAR-10: one label; CIFAR-100: the coarse label (20 classes), then the fine one (100) */
    const size_t labels = num_classes == 10 ? 1u : 2u, label_at = num_classes == 100 ? 1u : 0u;
    const size_t record = labels + 3u * CIFAR_PIXELS;
    uint8_t **data = (uint8_t **)calloc(path_count, sizeof(uint8_t *));
    size_t *sizes = (size_t *)calloc(path_count, sizeof(size_t));
    uint64_t count = 0;
    bool ok = data && sizes;
    if (!ok) set_error(SPINGALETT_ERR_ALLOC, "spingalett_load_cifar: allocation failed");
    for (uint32_t i = 0; ok && i < path_count; i++) {
        data[i] = (uint8_t *)spingalett_read_file(paths[i], &sizes[i]);
        if (!data[i]) {
            spingalett_log(LOG_ERROR, "Cannot read %s", paths[i] ? paths[i] : "(null)");
            ok = false;
        } else if (sizes[i] == 0 || sizes[i] % record != 0) {
            spingalett_log(LOG_ERROR, "%s is not a CIFAR batch of %zu-byte records", paths[i], record);
            set_error(SPINGALETT_ERR_INVALID, "not a CIFAR batch file (size is not a multiple of the record size)");
            ok = false;
        } else {
            count += sizes[i] / record;
        }
    }
    if (ok && count > UINT32_MAX) {
        set_error(SPINGALETT_ERR_INVALID, "spingalett_load_cifar: more than 2^32 - 1 samples");
        ok = false;
    }
    ok = ok && dataset_alloc(dataset, (uint32_t)count, 3u * CIFAR_PIXELS, num_classes);
    for (uint32_t i = 0, s = 0; ok && i < path_count; i++)
        for (size_t r = 0; ok && r < sizes[i] / record; r++, s++) {
            const uint8_t *rec = data[i] + r * record, *planes = rec + labels;
            if (rec[label_at] >= num_classes) {
                set_error(SPINGALETT_ERR_INVALID, "CIFAR label out of range");
                ok = false;
                break;
            }
            dataset->targets[(size_t)s * num_classes + rec[label_at]] = 1.0f;
            float *x = dataset->inputs + (size_t)s * 3u * CIFAR_PIXELS;
            for (uint32_t p = 0; p < CIFAR_PIXELS; p++)
                for (uint32_t c = 0; c < 3; c++)       /* as spingalett_load_idx: v / 255 exactly */
                    x[3u * p + c] = (float)((double)planes[c * CIFAR_PIXELS + p] / 255.0);
        }
    for (uint32_t i = 0; data && i < path_count; i++) spingalett_aligned_free(data[i]);
    free(data);
    free(sizes);
    if (!ok) spingalett_dataset_free(dataset);
    return ok;
}

/* ---------------------------------------------------------------- IDX */

typedef struct {
    uint8_t type;           /* 0x08 ubyte, 0x09 byte, 0x0B short, 0x0C int, 0x0D float, 0x0E double */
    uint32_t count;         /* first dimension */
    uint64_t sample_size;   /* product of the other dimensions */
} IdxHeader;

static bool read_be(FILE *f, unsigned char *b, size_t n) {
    return fread(b, 1, n, f) == n;
}

static bool idx_header(FILE *f, const char *path, IdxHeader *h) {
    unsigned char b[4];
    if (!read_be(f, b, 4) || b[0] != 0 || b[1] != 0 || b[3] == 0) {
        spingalett_log(LOG_ERROR, "%s is not an IDX file", path);
        set_error(SPINGALETT_ERR_INVALID, "not an IDX file");
        return false;
    }
    h->type = b[2];
    h->sample_size = 1;
    for (unsigned d = 0; d < b[3]; d++) {
        unsigned char v[4];
        if (!read_be(f, v, 4)) {
            set_error(SPINGALETT_ERR_FILE_IO, "IDX header is truncated");
            return false;
        }
        uint32_t dim = (uint32_t)v[0] << 24 | (uint32_t)v[1] << 16 | (uint32_t)v[2] << 8 | v[3];
        if (d == 0) h->count = dim;
        else h->sample_size *= dim;
        if (h->sample_size > UINT32_MAX) {
            set_error(SPINGALETT_ERR_INVALID, "IDX sample size exceeds 2^32 values");
            return false;
        }
    }
    return true;
}

static FILE *open_read(const char *path) {
    FILE *f = path ? fopen(path, "rb") : NULL;
    if (!f) {
        spingalett_log(LOG_ERROR, "Cannot open %s", path ? path : "(null)");
        set_error(SPINGALETT_ERR_FILE_IO, "cannot open file");
    }
    return f;
}

bool spingalett_load_idx(const char *images_path, const char *labels_path, uint32_t num_classes,
                         SpingalettDataset *dataset) {
    if (!dataset) { set_error(SPINGALETT_ERR_INVALID, "spingalett_load_idx: dataset is NULL"); return false; }
    memset(dataset, 0, sizeof *dataset);
    FILE *fi = open_read(images_path);
    FILE *fl = fi ? open_read(labels_path) : NULL;
    IdxHeader hi, hl;
    bool ok = fl && idx_header(fi, images_path, &hi) && idx_header(fl, labels_path, &hl);
    if (ok && (hi.type != 0x08 && hi.type != 0x0D && hi.type != 0x0E)) {
        set_error(SPINGALETT_ERR_INVALID, "IDX images must be unsigned bytes, float or double");
        ok = false;
    }
    if (ok && (hl.type != 0x08 || hl.sample_size != 1 || hl.count != hi.count)) {
        set_error(SPINGALETT_ERR_INVALID, "IDX labels must be one unsigned byte per image");
        ok = false;
    }

    unsigned char *labels = NULL, *raw = NULL;
    size_t values = ok ? (size_t)hi.count * hi.sample_size : 0;
    size_t width = ok ? (hi.type == 0x08 ? 1 : hi.type == 0x0D ? 4 : 8) : 0;
    if (ok) {
        labels = (unsigned char *)malloc((size_t)hi.count + 1);
        raw = (unsigned char *)malloc(values * width + 1);
        ok = labels && raw;
        if (!ok) set_error(SPINGALETT_ERR_ALLOC, "IDX read buffer allocation failed");
    }
    if (ok && (fread(raw, width, values, fi) != values || fread(labels, 1, hi.count, fl) != hi.count)) {
        set_error(SPINGALETT_ERR_FILE_IO, "IDX file is truncated");
        ok = false;
    }

    if (ok) {
        uint32_t classes = num_classes;
        if (classes == 0)
            for (uint32_t i = 0; i < hi.count; i++)
                if (labels[i] + 1u > classes) classes = labels[i] + 1u;
        for (uint32_t i = 0; ok && i < hi.count; i++)
            if (labels[i] >= classes) {
                set_error(SPINGALETT_ERR_INVALID, "IDX label out of range for num_classes");
                ok = false;
            }
        ok = ok && dataset_alloc(dataset, hi.count, (uint32_t)hi.sample_size, classes);
    }
    if (ok) {
        for (size_t i = 0; i < values; i++) {
            const unsigned char *p = raw + i * width;
            if (hi.type == 0x08) {
                /* in double: the library is built with -freciprocal-math, and v * (1 / 255.0f) is
                   off by an ulp for half of all bytes; this matches v / 255.0f exactly */
                dataset->inputs[i] = (float)((double)p[0] / 255.0);
            } else if (hi.type == 0x0D) {
                uint32_t u = (uint32_t)p[0] << 24 | (uint32_t)p[1] << 16 | (uint32_t)p[2] << 8 | p[3];
                float v;
                memcpy(&v, &u, sizeof v);
                dataset->inputs[i] = v;
            } else {
                uint64_t u = 0;
                for (int k = 0; k < 8; k++) u = u << 8 | p[k];
                double v;
                memcpy(&v, &u, sizeof v);
                dataset->inputs[i] = (float)v;
            }
        }
        for (uint32_t i = 0; i < hi.count; i++)
            dataset->targets[(size_t)i * dataset->target_size + labels[i]] = 1.0f;
    }

    free(raw);
    free(labels);
    if (fi) fclose(fi);
    if (fl) fclose(fl);
    if (!ok) spingalett_dataset_free(dataset);
    return ok;
}

/* ---------------------------------------------------------------- CSV */

static char *read_text_file(const char *path, size_t *size) {
    FILE *f = open_read(path);
    if (!f) return NULL;
    size_t cap = 1 << 16, len = 0;
    char *buf = (char *)malloc(cap);
    while (buf) {
        size_t got = fread(buf + len, 1, cap - len - 1, f);
        len += got;
        if (len < cap - 1) break;
        char *grown = (char *)realloc(buf, cap * 2);
        if (!grown) { free(buf); buf = NULL; break; }
        buf = grown;
        cap *= 2;
    }
    bool failed = ferror(f) != 0;
    fclose(f);
    if (!buf || failed) {
        free(buf);
        set_error(buf ? SPINGALETT_ERR_FILE_IO : SPINGALETT_ERR_ALLOC, "cannot read CSV file");
        return NULL;
    }
    buf[len] = '\0';
    *size = len;
    return buf;
}

/* Parses the comma-separated fields of one line into out (growing it); returns the field count,
   or -1 when a field is not a number. */
static long parse_csv_line(char *line, float **out, size_t *len, size_t *cap) {
    long fields = 0;
    char *p = line;
    for (;;) {
        while (*p == ' ' || *p == '\t') p++;
        char *end;
        float v = strtof(p, &end);
        if (end == p) return -1;
        while (*end == ' ' || *end == '\t' || *end == '\r') end++;
        if (*end != ',' && *end != '\0') return -1;
        if (*len == *cap) {
            size_t ncap = *cap ? *cap * 2 : 1024;
            float *grown = (float *)realloc(*out, ncap * sizeof(float));
            if (!grown) return -2;
            *out = grown;
            *cap = ncap;
        }
        (*out)[(*len)++] = v;
        fields++;
        if (*end == '\0') return fields;
        p = end + 1;
    }
}

bool spingalett_load_csv(const char *path, uint32_t target_columns, uint32_t num_classes,
                         SpingalettDataset *dataset) {
    if (!dataset) { set_error(SPINGALETT_ERR_INVALID, "spingalett_load_csv: dataset is NULL"); return false; }
    memset(dataset, 0, sizeof *dataset);
    if (target_columns == 0 || (num_classes > 0 && target_columns != 1)) {
        set_error(SPINGALETT_ERR_INVALID, "spingalett_load_csv: need target_columns >= 1 (exactly 1 with num_classes)");
        return false;
    }
    size_t size;
    char *text = read_text_file(path, &size);
    if (!text) return false;

    float *values = NULL;
    size_t len = 0, cap = 0, rows = 0;
    long columns = 0;
    bool ok = true, first = true;
    size_t line_no = 0;
    for (char *line = text; ok && line && *line; ) {
        char *next = strchr(line, '\n');
        if (next) *next++ = '\0';
        line_no++;
        char *q = line;
        while (*q == ' ' || *q == '\t' || *q == '\r') q++;
        if (*q) {
            size_t before = len;
            long n = parse_csv_line(line, &values, &len, &cap);
            if (n == -1 && first) {
                len = before;               /* a non-numeric first line is a header */
            } else if (n < 0 || (columns && n != columns)) {
                spingalett_log(LOG_ERROR, "%s:%zu: %s", path, line_no,
                               n == -2 ? "out of memory" : n < 0 ? "non-numeric field" : "wrong number of columns");
                set_error(n == -2 ? SPINGALETT_ERR_ALLOC : SPINGALETT_ERR_INVALID, "malformed CSV file");
                ok = false;
            } else {
                columns = n;
                rows++;
            }
            first = false;
        }
        line = next;
    }
    free(text);

    if (ok && (rows == 0 || rows > UINT32_MAX || columns <= (long)target_columns)) {
        set_error(SPINGALETT_ERR_INVALID, "CSV file has no data rows or not more columns than target_columns");
        ok = false;
    }
    uint32_t in_cols = ok ? (uint32_t)(columns - (long)target_columns) : 0;
    uint32_t target_size = num_classes > 0 ? num_classes : target_columns;
    if (ok && num_classes > 0)
        for (size_t r = 0; ok && r < rows; r++) {
            float label = values[r * (size_t)columns + in_cols];
            if (!(label >= 0.0f && label < (float)num_classes && label == floorf(label))) {
                spingalett_log(LOG_ERROR, "%s: row %zu has label %g, expected an integer in [0, %u)",
                               path, r + 1, (double)label, num_classes);
                set_error(SPINGALETT_ERR_INVALID, "CSV class label out of range");
                ok = false;
            }
        }
    ok = ok && dataset_alloc(dataset, (uint32_t)rows, in_cols, target_size);
    for (size_t r = 0; ok && r < rows; r++) {
        const float *row = values + r * (size_t)columns;
        memcpy(dataset->inputs + r * in_cols, row, in_cols * sizeof(float));
        float *t = dataset->targets + r * target_size;
        if (num_classes > 0) t[(uint32_t)row[in_cols]] = 1.0f;
        else memcpy(t, row + in_cols, target_columns * sizeof(float));
    }
    free(values);
    return ok;
}

/* ---------------------------------------------------------------- shuffling and splits */

void spingalett_dataset_shuffle(SpingalettDataset *d) {
    if (!d || d->count < 2) return;
    float *tmp = (float *)malloc(((size_t)d->input_size + d->target_size) * sizeof(float));
    if (!tmp) { set_error(SPINGALETT_ERR_ALLOC, "spingalett_dataset_shuffle: allocation failed"); return; }
    size_t in = d->input_size, out = d->target_size;
    for (uint32_t i = d->count - 1; i > 0; i--) {
        uint32_t j = (uint32_t)(rng_next64() % ((uint64_t)i + 1));
        if (j == i) continue;
        float *xi = d->inputs + i * in, *xj = d->inputs + j * in;
        float *yi = d->targets + i * out, *yj = d->targets + j * out;
        memcpy(tmp, xi, in * sizeof(float)); memcpy(xi, xj, in * sizeof(float)); memcpy(xj, tmp, in * sizeof(float));
        memcpy(tmp, yi, out * sizeof(float)); memcpy(yi, yj, out * sizeof(float)); memcpy(yj, tmp, out * sizeof(float));
    }
    free(tmp);
}

bool spingalett_dataset_split(SpingalettDataset *d, uint32_t count, SpingalettDataset *tail) {
    if (!d || !tail || tail == d || count == 0 || count >= d->count) {
        set_error(SPINGALETT_ERR_INVALID, "spingalett_dataset_split: need 0 < count < dataset size");
        return false;
    }
    if (!dataset_alloc(tail, count, d->input_size, d->target_size)) return false;
    uint32_t keep = d->count - count;
    memcpy(tail->inputs, d->inputs + (size_t)keep * d->input_size, (size_t)count * d->input_size * sizeof(float));
    memcpy(tail->targets, d->targets + (size_t)keep * d->target_size, (size_t)count * d->target_size * sizeof(float));
    d->count = keep;
    return true;
}
