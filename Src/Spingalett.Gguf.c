/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/*
 * GGUF files (llama.cpp's): their metadata, tensors and vocabulary as networks
 * (spingalett_load_pretrained()) and tokenizers (spingalett_tokenizer_load()).
 *
 * A file of version 2 or 3 holds a header, key/value pairs of metadata, the tensors' names, shapes
 * (innermost dimension first), types and offsets, then their data from a multiple of general.alignment
 * (32 unless it says). Every count, length and offset is checked against the file before it is used.
 *
 * The weights are dequantized to floats as llama.cpp's dequantize_row_*() does, block by block: F32,
 * F16, BF16, Q8_0, Q4_0, Q4_1, Q5_0, Q5_1 (blocks of 32 values, a half-precision scale and, for the _1
 * types, a minimum), the k-quants Q2_K, Q3_K, Q4_K, Q5_K and Q6_K (super-blocks of 256 values with
 * quantized scales), and IQ4_NL and IQ4_XS (4-bit codes of a table of values, which quantized files of
 * widths that are no multiple of 256 use where k-quants cannot be). The other i-quants, of codebooks, are
 * refused. The network of general.architecture llama (LLaMA, Mistral), qwen2 or gpt2 takes
 * its hyper-parameters from the metadata; llama.cpp's converter stores LLaMA's queries and keys with the
 * two halves of each head's rows interleaved (for its rotation of adjacent pairs), which are put back in
 * Hugging Face's order, the rotation of halves the library computes. The vocabulary (tokenizer.ggml.*)
 * becomes a byte-level BPE of merges ("gpt2") or SentencePiece's BPE of scores ("llama").
 */

#include "Spingalett.Pretrained.h"
#include "Spingalett.Tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------------- the file */

enum {
    GGUF_UINT8, GGUF_INT8, GGUF_UINT16, GGUF_INT16, GGUF_UINT32, GGUF_INT32, GGUF_FLOAT32, GGUF_BOOL, GGUF_STRING,
    GGUF_ARRAY, GGUF_UINT64, GGUF_INT64, GGUF_FLOAT64, GGUF_TYPES
};

/* ggml's tensor types that are read */
enum {
    GGML_F32 = 0, GGML_F16 = 1, GGML_Q4_0 = 2, GGML_Q4_1 = 3, GGML_Q5_0 = 6, GGML_Q5_1 = 7, GGML_Q8_0 = 8,
    GGML_Q2_K = 10, GGML_Q3_K = 11, GGML_Q4_K = 12, GGML_Q5_K = 13, GGML_Q6_K = 14, GGML_IQ4_NL = 20,
    GGML_IQ4_XS = 23, GGML_BF16 = 30
};

/* The sixteen values of IQ4_NL's and IQ4_XS's 4-bit codes (kvalues_iq4nl). */
static const int8_t iq4_values[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

typedef struct {
    const char *key;
    uint32_t key_len;
    uint32_t type;                  /* GGUF_* */
    uint32_t item_type;             /* an array's */
    uint64_t count;                 /* an array's items */
    const uint8_t *value;           /* the value (an array's first item) in the file */
} Kv;

typedef struct {
    const char *name;
    uint32_t name_len;
    uint32_t ndims;
    uint64_t dims[4];               /* innermost first */
    uint32_t type;                  /* GGML_* */
    const uint8_t *data;
    uint64_t bytes;
} GgufTensor;

typedef struct {
    const uint8_t *data;
    size_t size;
    uint32_t version;
    Kv *kvs;
    uint64_t kv_count;
    GgufTensor *tensors;
    uint64_t tensor_count;
    char error[200];
} Gguf;

typedef struct { const uint8_t *p, *end; bool ok; } Cursor;

static const uint8_t *take(Cursor *c, uint64_t n) {
    if (!c->ok || n > (uint64_t)(c->end - c->p)) { c->ok = false; return NULL; }
    const uint8_t *at = c->p;
    c->p += n;
    return at;
}

static uint32_t take32(Cursor *c) { const uint8_t *p = take(c, 4); return p ? slett_get32(p) : 0; }
static uint64_t take64(Cursor *c) { const uint8_t *p = take(c, 8); return p ? slett_get64(p) : 0; }

/* A string: its length (below 2^31) and bytes. */
static const char *take_string(Cursor *c, uint32_t *len) {
    uint64_t n = take64(c);
    if (n >= (1u << 31)) { c->ok = false; return NULL; }
    *len = (uint32_t)n;
    return (const char *)take(c, n);
}

static size_t scalar_size(uint32_t type) {
    switch (type) {
        case GGUF_UINT8: case GGUF_INT8: case GGUF_BOOL: return 1;
        case GGUF_UINT16: case GGUF_INT16: return 2;
        case GGUF_UINT32: case GGUF_INT32: case GGUF_FLOAT32: return 4;
        case GGUF_UINT64: case GGUF_INT64: case GGUF_FLOAT64: return 8;
        default: return 0;
    }
}

/* Skips a value of a type (arrays of arrays to a few levels). */
static void skip_value(Cursor *c, uint32_t type, int depth) {
    if (type == GGUF_STRING) {
        uint32_t len;
        (void)take_string(c, &len);
    } else if (type == GGUF_ARRAY) {
        uint32_t item = take32(c);
        uint64_t count = take64(c);
        if (depth > 3 || item >= GGUF_TYPES) { c->ok = false; return; }
        size_t size = scalar_size(item);
        if (size) {
            if (count > (uint64_t)(c->end - c->p) / size) c->ok = false;
            else c->p += count * size;
            return;
        }
        for (uint64_t k = 0; k < count && c->ok; k++) skip_value(c, item, depth + 1);
    } else {
        size_t size = scalar_size(type);
        if (!size) c->ok = false;
        else (void)take(c, size);
    }
}

/* Bytes of n values of a ggml type (n a multiple of its block), 0 for a type that is not read. */
static uint64_t type_bytes(uint32_t type, uint64_t n, bool *whole_blocks) {
    uint32_t block, size;
    switch (type) {
        case GGML_F32: block = 1; size = 4; break;
        case GGML_F16: case GGML_BF16: block = 1; size = 2; break;
        case GGML_Q4_0: block = 32; size = 18; break;
        case GGML_Q4_1: block = 32; size = 20; break;
        case GGML_Q5_0: block = 32; size = 22; break;
        case GGML_Q5_1: block = 32; size = 24; break;
        case GGML_Q8_0: block = 32; size = 34; break;
        case GGML_Q2_K: block = 256; size = 84; break;
        case GGML_Q3_K: block = 256; size = 110; break;
        case GGML_Q4_K: block = 256; size = 144; break;
        case GGML_Q5_K: block = 256; size = 176; break;
        case GGML_Q6_K: block = 256; size = 210; break;
        case GGML_IQ4_NL: block = 32; size = 18; break;
        case GGML_IQ4_XS: block = 256; size = 136; break;
        default: return 0;
    }
    *whole_blocks = n % block == 0;
    return n / block * size;
}

static bool gfail(Gguf *g, const char *msg) {
    if (!g->error[0]) snprintf(g->error, sizeof g->error, "GGUF: %s", msg);
    return false;
}

static void gguf_free(Gguf *g) {
    free(g->kvs);
    free(g->tensors);
    g->kvs = NULL;
    g->tensors = NULL;
}

/* The header, metadata and tensor infos of a file in memory. */
static bool gguf_parse(Gguf *g, const uint8_t *data, size_t size) {
    memset(g, 0, sizeof *g);
    g->data = data;
    g->size = size;
    Cursor c = {data, data + size, true};
    const uint8_t *magic = take(&c, 4);
    if (!magic || memcmp(magic, "GGUF", 4)) return gfail(g, "not a GGUF file");
    g->version = take32(&c);
    if (c.ok && (g->version < 2 || g->version > 3)) {
        char msg[64];
        snprintf(msg, sizeof msg, "version %u is not read (2 and 3 are)", g->version);
        return gfail(g, msg);
    }
    g->tensor_count = take64(&c);
    g->kv_count = take64(&c);
    /* a pair takes 13 bytes at least, a tensor's info 24: counts beyond the file are refused before allocating */
    if (!c.ok || g->kv_count > size / 13 || g->tensor_count > size / 24) return gfail(g, "the header is damaged");
    g->kvs = (Kv *)calloc(g->kv_count + 1, sizeof(Kv));
    g->tensors = (GgufTensor *)calloc(g->tensor_count + 1, sizeof(GgufTensor));
    if (!g->kvs || !g->tensors) return gfail(g, "out of memory");
    uint64_t alignment = 32;
    for (uint64_t k = 0; k < g->kv_count && c.ok; k++) {
        Kv *kv = &g->kvs[k];
        kv->key = take_string(&c, &kv->key_len);
        kv->type = take32(&c);
        if (!c.ok || kv->type >= GGUF_TYPES) { c.ok = false; break; }
        if (kv->type == GGUF_ARRAY) {
            const uint8_t *at = c.p;
            kv->item_type = take32(&c);
            kv->count = take64(&c);
            kv->value = c.p;
            c.p = at;
        } else {
            kv->value = c.p;
        }
        skip_value(&c, kv->type, 0);
        if (c.ok && kv->type == GGUF_UINT32 && kv->key_len == 17 && !memcmp(kv->key, "general.alignment", 17))
            alignment = slett_get32(kv->value);
    }
    if (!c.ok) return gfail(g, "the metadata is damaged");
    if (!alignment || (alignment & (alignment - 1)) || alignment > (1u << 20)) return gfail(g, "the alignment is invalid");
    uint64_t *offsets = (uint64_t *)malloc((g->tensor_count + 1) * sizeof(uint64_t));
    if (!offsets) return gfail(g, "out of memory");
    for (uint64_t k = 0; k < g->tensor_count && c.ok; k++) {
        GgufTensor *t = &g->tensors[k];
        t->name = take_string(&c, &t->name_len);
        t->ndims = take32(&c);
        if (!c.ok || t->ndims == 0 || t->ndims > 4) { c.ok = false; break; }
        for (uint32_t d = 0; d < t->ndims; d++) t->dims[d] = take64(&c);
        t->type = take32(&c);
        offsets[k] = take64(&c);
    }
    if (!c.ok) { free(offsets); return gfail(g, "the tensors' descriptions are damaged"); }
    /* the data: from the next multiple of the alignment */
    const uint64_t at = (uint64_t)(c.p - data), start = (at + alignment - 1) / alignment * alignment;
    bool ok = true;
    for (uint64_t k = 0; k < g->tensor_count && ok; k++) {
        GgufTensor *t = &g->tensors[k];
        uint64_t n = 1;
        for (uint32_t d = 0; d < t->ndims && ok; d++) {
            if (t->dims[d] == 0 || n > (1ull << 40) / t->dims[d]) ok = false;
            else n *= t->dims[d];
        }
        bool whole = true;
        t->bytes = ok ? type_bytes(t->type, t->dims[0], &whole) * (n / t->dims[0]) : 0;
        if (!ok) break;
        if (!t->bytes || !whole) continue;          /* a type that is not read: refused if a layer needs it */
        if (start > size || offsets[k] > size - start || t->bytes > size - start - offsets[k]) ok = false;
        else t->data = data + start + offsets[k];
    }
    free(offsets);
    return ok || gfail(g, "a tensor lies outside the file");
}

static const Kv *find_kv(const Gguf *g, const char *key) {
    const size_t n = strlen(key);
    for (uint64_t k = 0; k < g->kv_count; k++)
        if (g->kvs[k].key_len == n && !memcmp(g->kvs[k].key, key, n)) return &g->kvs[k];
    return NULL;
}

/* An integer of any integer type (and a bool), false when the key is missing or of another type. */
static bool kv_int(const Gguf *g, const char *key, uint64_t *out) {
    const Kv *kv = find_kv(g, key);
    if (!kv) return false;
    const uint8_t *p = kv->value;
    switch (kv->type) {
        case GGUF_UINT8: case GGUF_BOOL: *out = p[0]; return true;
        case GGUF_INT8: *out = (uint64_t)(int64_t)(int8_t)p[0]; return (int8_t)p[0] >= 0;
        case GGUF_UINT16: *out = slett_get16(p); return true;
        case GGUF_INT16: *out = (uint64_t)(int64_t)(int16_t)slett_get16(p); return (int16_t)slett_get16(p) >= 0;
        case GGUF_UINT32: *out = slett_get32(p); return true;
        case GGUF_INT32: *out = (uint64_t)(int64_t)(int32_t)slett_get32(p); return (int32_t)slett_get32(p) >= 0;
        case GGUF_UINT64: case GGUF_INT64: *out = slett_get64(p); return (int64_t)*out >= 0;
        default: return false;
    }
}

static uint32_t kv_u32(const Gguf *g, const char *key, uint32_t fallback) {
    uint64_t v;
    return kv_int(g, key, &v) && v <= UINT32_MAX ? (uint32_t)v : fallback;
}

static float kv_float(const Gguf *g, const char *key, float fallback) {
    const Kv *kv = find_kv(g, key);
    if (!kv) return fallback;
    if (kv->type == GGUF_FLOAT32) { float f; uint32_t b = slett_get32(kv->value); memcpy(&f, &b, 4); return f; }
    if (kv->type == GGUF_FLOAT64) { double d; uint64_t b = slett_get64(kv->value); memcpy(&d, &b, 8); return (float)d; }
    uint64_t v;
    return kv_int(g, key, &v) ? (float)v : fallback;
}

/* A string value copied into out (cap bytes, NUL-terminated); false when missing, not a string or too long. */
static bool kv_string(const Gguf *g, const char *key, char *out, size_t cap) {
    const Kv *kv = find_kv(g, key);
    if (!kv || kv->type != GGUF_STRING) return false;
    uint64_t n = slett_get64(kv->value);
    if (n >= cap) return false;
    memcpy(out, kv->value + 8, (size_t)n);
    out[n] = '\0';
    return true;
}

/* ------------------------------------------------------------------------- dequantization */

static float half(const uint8_t *p) { return spingalett_fp16_to_float(slett_get16(p)); }

/* The 6-bit scale and minimum j of a Q4_K or Q5_K super-block (get_scale_min_k4()). */
static void scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (uint8_t)((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4));
        *m = (uint8_t)((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
    }
}

/* n values (a multiple of the type's block) of a ggml type at x into y. */
static void dequantize(const uint8_t *x, uint32_t type, float *y, uint64_t n) {
    switch (type) {
        case GGML_F32: spingalett_decode(y, x, SPG_DTYPE_F32, (size_t)n); return;
        case GGML_F16: spingalett_decode(y, x, SPG_DTYPE_F16, (size_t)n); return;
        case GGML_BF16: spingalett_decode(y, x, SPG_DTYPE_BF16, (size_t)n); return;
        default: break;
    }
    for (uint64_t i = 0; i < n;) {
        switch (type) {
            case GGML_Q4_0: {                   /* d, qs[16]: (q - 8) d */
                const float d = half(x);
                for (int j = 0; j < 16; j++) {
                    y[j] = (float)((x[2 + j] & 0x0F) - 8) * d;
                    y[j + 16] = (float)((x[2 + j] >> 4) - 8) * d;
                }
                x += 18; y += 32; i += 32;
                break;
            }
            case GGML_Q4_1: {                   /* d, m, qs[16]: q d + m */
                const float d = half(x), m = half(x + 2);
                for (int j = 0; j < 16; j++) {
                    y[j] = (float)(x[4 + j] & 0x0F) * d + m;
                    y[j + 16] = (float)(x[4 + j] >> 4) * d + m;
                }
                x += 20; y += 32; i += 32;
                break;
            }
            case GGML_Q5_0: {                   /* d, qh (32 fifth bits), qs[16]: (q - 16) d */
                const float d = half(x);
                const uint32_t qh = slett_get32(x + 2);
                for (int j = 0; j < 16; j++) {
                    const uint8_t h0 = (uint8_t)(((qh >> j) << 4) & 0x10), h1 = (uint8_t)((qh >> (j + 12)) & 0x10);
                    y[j] = (float)(((x[6 + j] & 0x0F) | h0) - 16) * d;
                    y[j + 16] = (float)(((x[6 + j] >> 4) | h1) - 16) * d;
                }
                x += 22; y += 32; i += 32;
                break;
            }
            case GGML_Q5_1: {                   /* d, m, qh, qs[16]: q d + m */
                const float d = half(x), m = half(x + 2);
                const uint32_t qh = slett_get32(x + 4);
                for (int j = 0; j < 16; j++) {
                    const uint8_t h0 = (uint8_t)(((qh >> j) << 4) & 0x10), h1 = (uint8_t)((qh >> (j + 12)) & 0x10);
                    y[j] = (float)((x[8 + j] & 0x0F) | h0) * d + m;
                    y[j + 16] = (float)((x[8 + j] >> 4) | h1) * d + m;
                }
                x += 24; y += 32; i += 32;
                break;
            }
            case GGML_Q8_0: {                   /* d, qs[32] signed: q d */
                const float d = half(x);
                for (int j = 0; j < 32; j++) y[j] = (float)(int8_t)x[2 + j] * d;
                x += 34; y += 32; i += 32;
                break;
            }
            case GGML_Q2_K: {                   /* scales[16] (4-bit scale and minimum), qs[64], d, dmin */
                const uint8_t *sc = x, *q = x + 16;
                const float d = half(x + 80), dmin = half(x + 82);
                int is = 0;
                for (int nb = 0; nb < 256; nb += 128) {
                    int shift = 0;
                    for (int j = 0; j < 4; j++) {
                        uint8_t s = sc[is++];
                        float dl = d * (float)(s & 0xF), ml = dmin * (float)(s >> 4);
                        for (int l = 0; l < 16; l++) *y++ = dl * (float)((int8_t)((q[l] >> shift) & 3)) - ml;
                        s = sc[is++];
                        dl = d * (float)(s & 0xF);
                        ml = dmin * (float)(s >> 4);
                        for (int l = 0; l < 16; l++) *y++ = dl * (float)((int8_t)((q[l + 16] >> shift) & 3)) - ml;
                        shift += 2;
                    }
                    q += 32;
                }
                x += 84; i += 256;
                break;
            }
            case GGML_Q3_K: {                   /* hmask[32], qs[64], scales[12] (6-bit), d */
                const uint8_t *hm = x, *q = x + 32;
                const float d_all = half(x + 108);
                /* the sixteen 6-bit scales, unpacked as llama.cpp's kmask1, kmask2 shuffle does */
                uint32_t aux[4];
                const uint32_t kmask1 = 0x03030303u, kmask2 = 0x0f0f0f0fu;
                aux[0] = slett_get32(x + 96); aux[1] = slett_get32(x + 100); aux[2] = slett_get32(x + 104);
                const uint32_t tmp = aux[2];
                aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
                aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
                aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
                aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
                int8_t scales[16];
                for (int k = 0; k < 16; k++) scales[k] = (int8_t)(uint8_t)(aux[k / 4] >> (8 * (k % 4)));
                int is = 0;
                uint8_t m = 1;
                for (int nb = 0; nb < 256; nb += 128) {
                    int shift = 0;
                    for (int j = 0; j < 4; j++) {
                        float dl = d_all * (float)(scales[is++] - 32);
                        for (int l = 0; l < 16; l++)
                            *y++ = dl * (float)((int8_t)((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4));
                        dl = d_all * (float)(scales[is++] - 32);
                        for (int l = 0; l < 16; l++)
                            *y++ = dl * (float)((int8_t)((q[l + 16] >> shift) & 3) - ((hm[l + 16] & m) ? 0 : 4));
                        shift += 2;
                        m = (uint8_t)(m << 1);
                    }
                    q += 32;
                }
                x += 110; i += 256;
                break;
            }
            case GGML_Q4_K: {                   /* d, dmin, scales[12], qs[128] */
                const float d = half(x), dmin = half(x + 2);
                const uint8_t *sc = x + 4, *q = x + 16;
                int is = 0;
                for (int j = 0; j < 256; j += 64) {
                    uint8_t s, mn;
                    scale_min_k4(is, sc, &s, &mn);
                    const float d1 = d * s, m1 = dmin * mn;
                    scale_min_k4(is + 1, sc, &s, &mn);
                    const float d2 = d * s, m2 = dmin * mn;
                    for (int l = 0; l < 32; l++) *y++ = d1 * (float)(q[l] & 0xF) - m1;
                    for (int l = 0; l < 32; l++) *y++ = d2 * (float)(q[l] >> 4) - m2;
                    q += 32;
                    is += 2;
                }
                x += 144; i += 256;
                break;
            }
            case GGML_Q5_K: {                   /* d, dmin, scales[12], qh[32], qs[128] */
                const float d = half(x), dmin = half(x + 2);
                const uint8_t *sc = x + 4, *qh = x + 16, *ql = x + 48;
                int is = 0;
                uint8_t u1 = 1, u2 = 2;
                for (int j = 0; j < 256; j += 64) {
                    uint8_t s, mn;
                    scale_min_k4(is, sc, &s, &mn);
                    const float d1 = d * s, m1 = dmin * mn;
                    scale_min_k4(is + 1, sc, &s, &mn);
                    const float d2 = d * s, m2 = dmin * mn;
                    for (int l = 0; l < 32; l++) *y++ = d1 * (float)((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1;
                    for (int l = 0; l < 32; l++) *y++ = d2 * (float)((ql[l] >> 4) + (qh[l] & u2 ? 16 : 0)) - m2;
                    ql += 32;
                    is += 2;
                    u1 = (uint8_t)(u1 << 2);
                    u2 = (uint8_t)(u2 << 2);
                }
                x += 176; i += 256;
                break;
            }
            case GGML_Q6_K: {                   /* ql[128], qh[64], scales[16] signed, d */
                const float d = half(x + 208);
                const uint8_t *ql = x, *qh = x + 128;
                const int8_t *sc = (const int8_t *)(x + 192);
                for (int nb = 0; nb < 256; nb += 128) {
                    for (int l = 0; l < 32; l++) {
                        const int is = l / 16;
                        const int8_t q1 = (int8_t)((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                        const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                        const int8_t q3 = (int8_t)((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                        const int8_t q4 = (int8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                        y[l] = d * (float)sc[is] * (float)q1;
                        y[l + 32] = d * (float)sc[is + 2] * (float)q2;
                        y[l + 64] = d * (float)sc[is + 4] * (float)q3;
                        y[l + 96] = d * (float)sc[is + 6] * (float)q4;
                    }
                    y += 128;
                    ql += 64;
                    qh += 32;
                    sc += 8;
                }
                x += 210; i += 256;
                break;
            }
            case GGML_IQ4_NL: {                 /* d, qs[16]: d times the code's value */
                const float d = half(x);
                for (int j = 0; j < 16; j++) {
                    y[j] = d * (float)iq4_values[x[2 + j] & 0xF];
                    y[j + 16] = d * (float)iq4_values[x[2 + j] >> 4];
                }
                x += 18; y += 32; i += 32;
                break;
            }
            default: {                          /* GGML_IQ4_XS: d, scales_h, scales_l[4] (6-bit scales of 32 values), qs[128] */
                const float d = half(x);
                const uint16_t scales_h = slett_get16(x + 2);
                const uint8_t *scales_l = x + 4, *qs = x + 8;
                for (int ib = 0; ib < 8; ib++) {
                    const int ls = ((scales_l[ib / 2] >> (4 * (ib % 2))) & 0xF) | (((scales_h >> (2 * ib)) & 3) << 4);
                    const float dl = d * (float)(ls - 32);
                    for (int j = 0; j < 16; j++) {
                        y[j] = dl * (float)iq4_values[qs[j] & 0xF];
                        y[j + 16] = dl * (float)iq4_values[qs[j] >> 4];
                    }
                    y += 32;
                    qs += 16;
                }
                x += 136; i += 256;
                break;
            }
        }
    }
}

/* ------------------------------------------------------------------------- the tensors of a network */

typedef struct {
    SpgTensorSource base;
    const Gguf *g;
    SpgArch arch;
    bool permuted;                  /* LLaMA's queries and keys in llama.cpp's order */
    uint32_t heads, kv_heads;
} GgufSource;

static void gguf_part_name(SpgPart part, uint32_t block, char *name, size_t cap) {
    static const char *const names[SPG_PART_COUNT] = {
        [SPG_PART_EMBED] = "token_embd.weight", [SPG_PART_POSITIONS] = "position_embd.weight",
        [SPG_PART_NORM] = "output_norm.weight", [SPG_PART_NORM_BIAS] = "output_norm.bias", [SPG_PART_HEAD] = "output.weight",
        [SPG_PART_ATTN_NORM] = "attn_norm.weight", [SPG_PART_ATTN_NORM_BIAS] = "attn_norm.bias",
        [SPG_PART_Q] = "attn_q.weight", [SPG_PART_K] = "attn_k.weight", [SPG_PART_V] = "attn_v.weight",
        [SPG_PART_Q_BIAS] = "attn_q.bias", [SPG_PART_K_BIAS] = "attn_k.bias", [SPG_PART_V_BIAS] = "attn_v.bias",
        [SPG_PART_QKV] = "attn_qkv.weight", [SPG_PART_QKV_BIAS] = "attn_qkv.bias",
        [SPG_PART_O] = "attn_output.weight", [SPG_PART_O_BIAS] = "attn_output.bias",
        [SPG_PART_FFN_NORM] = "ffn_norm.weight", [SPG_PART_FFN_NORM_BIAS] = "ffn_norm.bias",
        [SPG_PART_GATE] = "ffn_gate.weight", [SPG_PART_GATE_BIAS] = "ffn_gate.bias",
        [SPG_PART_UP] = "ffn_up.weight", [SPG_PART_UP_BIAS] = "ffn_up.bias",
        [SPG_PART_DOWN] = "ffn_down.weight", [SPG_PART_DOWN_BIAS] = "ffn_down.bias",
    };
    if (part <= SPG_PART_HEAD) snprintf(name, cap, "%s", names[part]);
    else snprintf(name, cap, "blk.%u.%s", block, names[part]);
}

static const GgufTensor *find_gguf_tensor(const Gguf *g, const char *name) {
    const size_t n = strlen(name);
    for (uint64_t k = 0; k < g->tensor_count; k++)
        if (g->tensors[k].name_len == n && !memcmp(g->tensors[k].name, name, n)) return &g->tensors[k];
    return NULL;
}

static bool gguf_has(SpgTensorSource *s, SpgPart part, uint32_t block) {
    char name[64];
    gguf_part_name(part, block, name, sizeof name);
    return find_gguf_tensor(((GgufSource *)s)->g, name) != NULL;
}

/* The rows of each head back in Hugging Face's order: llama.cpp's row h d + 2 i + j holds Hugging Face's
   h d + j d / 2 + i (cols values a row; tmp holds a head's rows). */
static void unpermute(float *w, uint32_t heads, uint32_t rows, uint32_t cols, float *tmp) {
    const uint32_t d = rows / heads, half_d = d / 2;
    for (uint32_t h = 0; h < heads; h++) {
        float *head = w + (size_t)h * d * cols;
        memcpy(tmp, head, (size_t)d * cols * sizeof(float));
        for (uint32_t i = 0; i < half_d; i++)
            for (uint32_t j = 0; j < 2; j++)
                memcpy(head + ((size_t)j * half_d + i) * cols, tmp + ((size_t)2 * i + j) * cols, (size_t)cols * sizeof(float));
    }
}

static bool gguf_read(SpgTensorSource *s, SpgPart part, uint32_t block, float *dst, uint32_t rows, uint32_t cols) {
    GgufSource *src = (GgufSource *)s;
    char name[64];
    gguf_part_name(part, block, name, sizeof name);
    const GgufTensor *t = find_gguf_tensor(src->g, name);
    if (!t) { snprintf(s->error, sizeof s->error, "GGUF: the file has no %s", name); return false; }
    const bool vector = rows == 1 && part != SPG_PART_EMBED && part != SPG_PART_HEAD;
    /* innermost first: a matrix [rows][cols] is cols x rows; learned positions may be more than the context */
    const bool fits = vector ? t->ndims == 1 && t->dims[0] == cols
                    : t->ndims == 2 && t->dims[0] == cols &&
                      (t->dims[1] == rows || (part == SPG_PART_POSITIONS && t->dims[1] > rows));
    if (!fits) {
        snprintf(s->error, sizeof s->error, "GGUF: %s has shape [%llu, %llu], not [%u, %u]", name,
                 (unsigned long long)t->dims[0], (unsigned long long)(t->ndims > 1 ? t->dims[1] : 1), cols, vector ? 1u : rows);
        return false;
    }
    if (!t->data) {
        snprintf(s->error, sizeof s->error, "GGUF: %s is of the type %u, which is not read (F32, F16, BF16, Q4_0, Q4_1, "
                 "Q5_0, Q5_1, Q8_0, Q2_K to Q6_K, IQ4_NL and IQ4_XS are)", name, t->type);
        return false;
    }
    const bool q = part == SPG_PART_Q || part == SPG_PART_Q_BIAS, k = part == SPG_PART_K || part == SPG_PART_K_BIAS;
    const uint32_t heads = q ? src->heads : src->kv_heads, r = vector ? cols : rows, c = vector ? 1 : cols;
    if (src->permuted && (q || k) && (r % heads || (r / heads) % 2)) {
        snprintf(s->error, sizeof s->error, "GGUF: %s does not split into heads of an even size", name);
        return false;
    }
    if (s->dry) return true;
    /* rows in parallel, as many as a type's block allows a chunk */
    bool whole;
    const uint64_t row_bytes = type_bytes(t->type, cols, &whole);
    const int64_t n = (int64_t)rows;
    SPINGALETT_PARALLEL_FOR((uint64_t)rows * cols >= (1u << 20),
        for (int64_t row = 0; row < n; row++)
            dequantize(t->data + (uint64_t)row * row_bytes, t->type, dst + (uint64_t)row * cols, cols);
    );
    /* LLaMA's queries and keys, and their biases (vectors of one value a row) */
    if (src->permuted && (q || k)) {
        float *tmp = (float *)malloc((size_t)(r / heads) * c * sizeof(float));
        if (!tmp) { snprintf(s->error, sizeof s->error, "GGUF: out of memory"); return false; }
        unpermute(dst, heads, r, c, tmp);
        free(tmp);
    }
    return true;
}

NeuralNetwork *spingalett_gguf_network(const uint8_t *data, size_t size, uint32_t context) {
    Gguf g;
    char msg[240];
    if (!gguf_parse(&g, data, size)) {
        set_error(SPINGALETT_ERR_INVALID, g.error);
        gguf_free(&g);
        return NULL;
    }
    char arch[32] = "", key[96];
    (void)kv_string(&g, "general.architecture", arch, sizeof arch);
    SpgTransformer t;
    memset(&t, 0, sizeof t);
    const bool gpt2 = !strcmp(arch, "gpt2"), qwen2 = !strcmp(arch, "qwen2");
    if (!gpt2 && !qwen2 && strcmp(arch, "llama")) {
        snprintf(msg, sizeof msg, "GGUF: the architecture '%s' is not supported (llama, qwen2 and gpt2 are)",
                 arch[0] ? arch : "(none)");
        set_error(SPINGALETT_ERR_INVALID, msg);
        gguf_free(&g);
        return NULL;
    }
#define ARCH_KEY(name) (snprintf(key, sizeof key, "%s." name, arch), key)
    t.arch = gpt2 ? SPG_ARCH_GPT2 : SPG_ARCH_LLAMA;
    t.width = kv_u32(&g, ARCH_KEY("embedding_length"), 0);
    t.layers = kv_u32(&g, ARCH_KEY("block_count"), 0);
    t.hidden = kv_u32(&g, ARCH_KEY("feed_forward_length"), 0);
    t.heads = kv_u32(&g, ARCH_KEY("attention.head_count"), 0);
    t.kv_heads = kv_u32(&g, ARCH_KEY("attention.head_count_kv"), t.heads);
    t.head_size = kv_u32(&g, ARCH_KEY("attention.key_length"), t.heads ? t.width / t.heads : 0);
    t.eps = gpt2 ? kv_float(&g, ARCH_KEY("attention.layer_norm_epsilon"), 1e-5f)
                 : kv_float(&g, ARCH_KEY("attention.layer_norm_rms_epsilon"), 1e-6f);
    t.theta = kv_float(&g, ARCH_KEY("rope.freq_base"), 10000.0f);
    t.act = ACT_GELU_TANH;
    const uint32_t positions = kv_u32(&g, ARCH_KEY("context_length"), 0);
    t.context = context ? context : gpt2 ? positions : positions && positions < 2048 ? positions : 2048;
    const uint32_t rotary = kv_u32(&g, ARCH_KEY("rope.dimension_count"), t.head_size);
    char scaling[32] = "none";
    (void)kv_string(&g, ARCH_KEY("rope.scaling.type"), scaling, sizeof scaling);
    const float factor = kv_float(&g, ARCH_KEY("rope.scaling.factor"), 1.0f);
#undef ARCH_KEY
    const GgufTensor *embed = find_gguf_tensor(&g, "token_embd.weight");
    t.vocabulary = embed && embed->ndims == 2 && embed->dims[1] < (1u << 24) ? (uint32_t)embed->dims[1] : 0;
    const char *problem = !embed ? "the file has no token_embd.weight"
                        : gpt2 && context > positions ? "the context exceeds GPT-2's positions"
                        : !gpt2 && rotary != t.head_size ? "rotary embeddings on part of a head are not supported"
                        : !gpt2 && ((strcmp(scaling, "none") && factor != 1.0f) || find_gguf_tensor(&g, "rope_freqs.weight"))
                          ? "scaled rotary embeddings are not supported" : NULL;
    NeuralNetwork *net = NULL;
    if (problem) {
        snprintf(msg, sizeof msg, "GGUF: %s", problem);
        set_error(SPINGALETT_ERR_INVALID, msg);
    } else {
        GgufSource src = {.base = {.has = gguf_has, .read = gguf_read}, .g = &g, .arch = t.arch,
                          .permuted = !gpt2 && !qwen2, .heads = t.heads, .kv_heads = t.kv_heads};
        net = spingalett_build_transformer(&t, &src.base);
    }
    gguf_free(&g);
    return net;
}

/* ------------------------------------------------------------------------- the vocabulary */

/* The strings of an array of strings, NULL when the key is missing or holds anything else. */
static SpgString *kv_strings(const Gguf *g, const char *key, uint32_t *count) {
    const Kv *kv = find_kv(g, key);
    if (!kv || kv->type != GGUF_ARRAY || kv->item_type != GGUF_STRING || kv->count >= (1u << 24)) return NULL;
    SpgString *s = (SpgString *)malloc((size_t)(kv->count + 1) * sizeof(SpgString));
    if (!s) return NULL;
    Cursor c = {kv->value, g->data + g->size, true};
    for (uint64_t k = 0; k < kv->count && c.ok; k++) s[k].s = take_string(&c, &s[k].n);
    if (!c.ok) { free(s); return NULL; }
    *count = (uint32_t)kv->count;
    return s;
}

/* An array of count numbers as floats (scores) or integers (token types). */
static void *kv_numbers(const Gguf *g, const char *key, uint32_t count, bool floats) {
    const Kv *kv = find_kv(g, key);
    if (!kv || kv->type != GGUF_ARRAY || kv->count != count) return NULL;
    const size_t size = scalar_size(kv->item_type);
    if (!size) return NULL;
    void *out = malloc((size_t)count * 4 + 4);
    if (!out) return NULL;
    for (uint32_t k = 0; k < count; k++) {
        const uint8_t *p = kv->value + (size_t)k * size;
        if (floats) {
            float f;
            if (kv->item_type == GGUF_FLOAT32) { uint32_t b = slett_get32(p); memcpy(&f, &b, 4); }
            else if (kv->item_type == GGUF_FLOAT64) { double d; uint64_t b = slett_get64(p); memcpy(&d, &b, 8); f = (float)d; }
            else { free(out); return NULL; }
            ((float *)out)[k] = f;
        } else {
            ((int32_t *)out)[k] = size == 1 ? (int32_t)(int8_t)p[0] : size == 2 ? (int32_t)(int16_t)slett_get16(p)
                                                                       : (int32_t)slett_get32(p);
        }
    }
    return out;
}

SpingalettTokenizer *spingalett_gguf_tokenizer(const uint8_t *data, size_t size) {
    Gguf g;
    if (!gguf_parse(&g, data, size)) {
        set_error(SPINGALETT_ERR_INVALID, g.error);
        gguf_free(&g);
        return NULL;
    }
    char model[32] = "", pre[64] = "";
    SpgTokenizerSpec spec;
    memset(&spec, 0, sizeof spec);
    spec.model = kv_string(&g, "tokenizer.ggml.model", model, sizeof model) ? model : NULL;
    spec.pre = kv_string(&g, "tokenizer.ggml.pre", pre, sizeof pre) ? pre : NULL;
    SpgString *tokens = kv_strings(&g, "tokenizer.ggml.tokens", &spec.count);
    SpgString *merges = kv_strings(&g, "tokenizer.ggml.merges", &spec.merge_count);
    float *scores = tokens ? (float *)kv_numbers(&g, "tokenizer.ggml.scores", spec.count, true) : NULL;
    int32_t *types = tokens ? (int32_t *)kv_numbers(&g, "tokenizer.ggml.token_type", spec.count, false) : NULL;
    spec.tokens = tokens;
    spec.merges = merges;
    spec.scores = scores;
    spec.types = types;
    spec.bos = kv_u32(&g, "tokenizer.ggml.bos_token_id", SPINGALETT_NO_TOKEN);
    spec.eos = kv_u32(&g, "tokenizer.ggml.eos_token_id", SPINGALETT_NO_TOKEN);
    spec.unk = kv_u32(&g, "tokenizer.ggml.unknown_token_id", SPINGALETT_NO_TOKEN);
    spec.pad = kv_u32(&g, "tokenizer.ggml.padding_token_id", SPINGALETT_NO_TOKEN);
    const bool spm = spec.model && !strcmp(spec.model, "llama");
    spec.add_space_prefix = kv_u32(&g, "tokenizer.ggml.add_space_prefix", spm) != 0;
    spec.add_bos = kv_u32(&g, "tokenizer.ggml.add_bos_token", spm) != 0;
    spec.add_eos = kv_u32(&g, "tokenizer.ggml.add_eos_token", 0) != 0;
    SpingalettTokenizer *tok = NULL;
    if (!tokens) set_error(SPINGALETT_ERR_INVALID, "GGUF: the file has no vocabulary (tokenizer.ggml.tokens)");
    else tok = spingalett_tokenizer_build(&spec);
    free(tokens);
    free(merges);
    free(scores);
    free(types);
    gguf_free(&g);
    return tok;
}
