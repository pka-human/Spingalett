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
#include "Spingalett.Image.h"
#include "Spingalett.Tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

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

/* The block precision of a ggml type, PRECISION_COUNT for the others. */
static PrecisionMode block_precision(uint32_t type) {
    switch (type) {
        case GGML_Q8_0: return PRECISION_Q8_0;
        case GGML_Q4_0: return PRECISION_Q4_0;
        case GGML_Q4_1: return PRECISION_Q4_1;
        case GGML_Q5_0: return PRECISION_Q5_0;
        case GGML_Q5_1: return PRECISION_Q5_1;
        case GGML_Q2_K: return PRECISION_Q2_K;
        case GGML_Q3_K: return PRECISION_Q3_K;
        case GGML_Q4_K: return PRECISION_Q4_K;
        case GGML_Q5_K: return PRECISION_Q5_K;
        case GGML_Q6_K: return PRECISION_Q6_K;
        case GGML_IQ4_NL: return PRECISION_IQ4_NL;
        case GGML_IQ4_XS: return PRECISION_IQ4_XS;
        default: return PRECISION_COUNT;
    }
}

/* n values (a multiple of the type's block) of a ggml type at x into y, as llama.cpp's dequantize_row_*(). */
static void dequantize(const uint8_t *x, uint32_t type, float *y, uint64_t n) {
    switch (type) {
        case GGML_F32: spingalett_decode(y, x, SPG_DTYPE_F32, (size_t)n); return;
        case GGML_F16: spingalett_decode(y, x, SPG_DTYPE_F16, (size_t)n); return;
        case GGML_BF16: spingalett_decode(y, x, SPG_DTYPE_BF16, (size_t)n); return;
        default: spingalett_block_decode(block_precision(type), x, y, (uint32_t)n); return;
    }
}

/* ------------------------------------------------------------------------- the tensors of a model */

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
        [SPG_PART_Q_NORM] = "attn_q_norm.weight", [SPG_PART_K_NORM] = "attn_k_norm.weight",
        [SPG_PART_ATTN_POST_NORM] = "post_attention_norm.weight", [SPG_PART_FFN_POST_NORM] = "post_ffw_norm.weight",
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

#if !defined(SPINGALETT_RUNTIME)
typedef struct {
    SpgTensorSource base;
    const Gguf *g;
    SpgArch arch;
    bool permuted;                  /* LLaMA's queries and keys in llama.cpp's order */
    uint32_t heads, kv_heads;
} GgufSource;

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
#endif

/* The transformer a parsed file describes (its hyper-parameters, from its metadata), and whether its queries and
   keys are in llama.cpp's order (LLaMA's); false with msg set when its architecture or a feature is not read.
   *freqs receives LLaMA 3's rotary frequencies (rope_freqs.weight's factors) or a linear scaling's, head_size / 2
   floats allocated, or NULL; `networks`: for spingalett_load_pretrained(), whose networks have neither those nor
   the normalized queries and keys of Qwen3. */
static bool gguf_describe(const Gguf *g, uint32_t context, bool networks, SpgTransformer *t, bool *permuted,
                          float **freqs, char *msg, size_t cap) {
    char arch[32] = "", key[96];
    (void)kv_string(g, "general.architecture", arch, sizeof arch);
    memset(t, 0, sizeof *t);
    *freqs = NULL;
    const bool gpt2 = !strcmp(arch, "gpt2"), qwen2 = !strcmp(arch, "qwen2"), qwen3 = !strcmp(arch, "qwen3");
    const bool gemma3 = !strcmp(arch, "gemma3"), llama = !strcmp(arch, "llama"), smollm3 = !strcmp(arch, "smollm3");
    const bool olmo2 = !strcmp(arch, "olmo2"), phi3 = !strcmp(arch, "phi3"), glm4 = !strcmp(arch, "glm4");
    if (!gpt2 && !qwen2 && !qwen3 && !gemma3 && !llama && !smollm3 && !olmo2 && !phi3 && !glm4) {
        snprintf(msg, cap, "GGUF: the architecture '%s' is not supported (llama, qwen2, qwen3, gemma3, smollm3, olmo2, "
                 "phi3, glm4 and gpt2 are)", arch[0] ? arch : "(none)");
        return false;
    }
#define ARCH_KEY(name) (snprintf(key, sizeof key, "%s." name, arch), key)
    t->arch = gpt2 ? SPG_ARCH_GPT2 : SPG_ARCH_LLAMA;
    t->width = kv_u32(g, ARCH_KEY("embedding_length"), 0);
    t->layers = kv_u32(g, ARCH_KEY("block_count"), 0);
    t->hidden = kv_u32(g, ARCH_KEY("feed_forward_length"), 0);
    t->heads = kv_u32(g, ARCH_KEY("attention.head_count"), 0);
    t->kv_heads = kv_u32(g, ARCH_KEY("attention.head_count_kv"), t->heads);
    t->head_size = kv_u32(g, ARCH_KEY("attention.key_length"), t->heads ? t->width / t->heads : 0);
    t->eps = gpt2 ? kv_float(g, ARCH_KEY("attention.layer_norm_epsilon"), 1e-5f)
                  : kv_float(g, ARCH_KEY("attention.layer_norm_rms_epsilon"), 1e-6f);
    t->theta = kv_float(g, ARCH_KEY("rope.freq_base"), 10000.0f);
    t->window = kv_u32(g, ARCH_KEY("attention.sliding_window"), 0);
    t->act = ACT_GELU_TANH;
    t->qk_norm = qwen3 || gemma3;
    t->gate = gemma3 ? ACT_GELU_TANH : ACT_SILU;
    if (gemma3) {
        /* Gemma 3: embeddings times the square root of the width, normalized outputs, of every six layers five
           attending through a window with a rotary base of their own (llama.cpp's defaults where the file has none) */
        t->embed_scale = sqrtf((float)t->width);
        t->post_norms = true;
        t->window_pattern = kv_u32(g, ARCH_KEY("attention.sliding_window_pattern"), 6);
        t->theta_local = kv_float(g, ARCH_KEY("rope.freq_base_swa"), 10000.0f);
    }
    if (smollm3) t->nope_interval = 4;
    if (olmo2) {
        /* OLMo 2: the attention's and the MLP's outputs normalized, their inputs not; queries and keys normalized
           over all heads; OLMo 3's windows in a pattern of four */
        t->no_pre_norms = t->wide_qk_norm = t->post_norms = true;
        if (t->window) {
            t->window_pattern = 4;
            t->theta_local = kv_float(g, ARCH_KEY("rope.freq_base_swa"), t->theta);
        }
    }
    t->fused_up = phi3 || glm4;
    t->post_norms = t->post_norms || glm4;
    const float magnitude = kv_float(g, ARCH_KEY("rope.scaling.attn_factor"), 1.0f);
    const uint32_t original = kv_u32(g, ARCH_KEY("rope.scaling.original_context_length"), 0);
    const float softcap = kv_float(g, ARCH_KEY("final_logit_softcapping"), 0.0f);
    const uint32_t positions = kv_u32(g, ARCH_KEY("context_length"), 0);
    const uint32_t most = networks ? 2048u : 4096u;
    t->context = context ? context : gpt2 ? positions : positions && positions < most ? positions : most;
    const uint32_t rotary = kv_u32(g, ARCH_KEY("rope.dimension_count"), t->head_size);
    char scaling[32] = "none";
    (void)kv_string(g, ARCH_KEY("rope.scaling.type"), scaling, sizeof scaling);
    const float factor = kv_float(g, ARCH_KEY("rope.scaling.factor"), 1.0f);
#undef ARCH_KEY
    const GgufTensor *embed = find_gguf_tensor(g, "token_embd.weight");
    /* rotary factors: LLaMA 3's, or Phi-3's LongRoPE (the long ones for windows beyond the original context) */
    const GgufTensor *factors = find_gguf_tensor(g, "rope_freqs.weight");
    if (!factors && find_gguf_tensor(g, "rope_factors_short.weight"))
        factors = find_gguf_tensor(g, t->context > original && original ? "rope_factors_long.weight"
                                                                          : "rope_factors_short.weight");
    t->vocabulary = embed && embed->ndims == 2 && embed->dims[1] < (1u << 24) ? (uint32_t)embed->dims[1] : 0;
    const bool linear = !strcmp(scaling, "linear") && factor != 1.0f, scaled = (strcmp(scaling, "none") && factor != 1.0f) ||
                        factors;
    const bool partial = !gpt2 && rotary != t->head_size;
    const char *problem = !embed ? "the file has no token_embd.weight"
                        : gpt2 && context > positions ? "the context exceeds GPT-2's positions"
                        : partial && (networks || !rotary || rotary > t->head_size || rotary % 2u || (llama && rotary))
                          ? "rotary embeddings on part of a head run as a model (spingalett_model_load()), of a head's "
                            "first values"
                        : networks && (t->qk_norm || t->post_norms || smollm3 || phi3 || glm4)
                          ? "the layers of Qwen3, Gemma 3, SmolLM3, OLMo 2, Phi-3 and GLM-4 run as a model "
                            "(spingalett_model_load()), not yet as a network"
                        : magnitude != 1.0f && partial
                          ? "a magnitude of the rotations (rope.scaling.attn_factor) on part of a head is not supported"
                        : softcap != 0.0f ? "soft-capped logits are not supported"
                        : networks && scaled ? "scaled rotary embeddings run as a model (spingalett_model_load()), not "
                                               "yet as a network"
                        : scaled && !factors && !linear ? "this scaling of the rotary embeddings is not supported "
                                                          "(LLaMA 3's frequencies and linear scaling are)"
                        : factors && (factors->ndims != 1 || factors->dims[0] != rotary / 2u || !factors->data ||
                                      (factors->type != GGML_F32 && factors->type != GGML_F16))
                          ? "the rotary factors are not those of half the rotated values" : NULL;
    if (problem) {
        snprintf(msg, cap, "GGUF: %s", problem);
        return false;
    }
    /* llama.cpp rotates adjacent pairs of LLaMA's and SmolLM3's (which its converter reorders) and of GLM-4's (as
       they are): their rows go back to the order of the halves, which the engine rotates */
    *permuted = llama || smollm3 || glm4;
    if (partial) t->rotary = rotary;
    /* Phi-3's LongRoPE turns the queries and keys with a magnitude: their scores are its square times larger */
    if (magnitude != 1.0f) t->score_scale = magnitude * magnitude;
    if (!gpt2 && scaled) {
        /* each pair's frequency theta^(-2 i / rotated) divided by its factor (llama.cpp's ggml_rope_ext()), or by the
           linear scaling's */
        const uint32_t half = rotary / 2u;
        float *f = (float *)malloc((size_t)half * sizeof(float)), *ff = factors ? (float *)malloc((size_t)half * 4u) : NULL;
        if (!f || (factors && !ff)) {
            free(f);
            free(ff);
            snprintf(msg, cap, "GGUF: out of memory");
            return false;
        }
        if (ff) dequantize(factors->data, factors->type, ff, half);
        for (uint32_t i = 0; i < half; i++)
            f[i] = (float)(pow((double)t->theta, -2.0 * (double)i / (double)rotary) / (ff ? (double)ff[i] : (double)factor));
        free(ff);
        *freqs = f;
        t->freqs = f;
    }
    return true;
}

#if !defined(SPINGALETT_RUNTIME)
NeuralNetwork *spingalett_gguf_network(const uint8_t *data, size_t size, uint32_t context) {
    Gguf g;
    char msg[240];
    if (!gguf_parse(&g, data, size)) {
        set_error(SPINGALETT_ERR_INVALID, g.error);
        gguf_free(&g);
        return NULL;
    }
    SpgTransformer t;
    bool permuted = false;
    float *freqs = NULL;
    NeuralNetwork *net = NULL;
    if (!gguf_describe(&g, context, true, &t, &permuted, &freqs, msg, sizeof msg)) {
        set_error(SPINGALETT_ERR_INVALID, msg);
    } else {
        GgufSource src = {.base = {.has = gguf_has, .read = gguf_read}, .g = &g, .arch = t.arch,
                          .permuted = permuted, .heads = t.heads, .kv_heads = t.kv_heads};
        net = spingalett_build_transformer(&t, &src.base);
    }
    free(freqs);
    gguf_free(&g);
    return net;
}
#endif

/* ------------------------------------------------------------------------- deployment models */

/* The precision a tensor's type is stored in, PRECISION_COUNT for the types no layer computes with. */
static PrecisionMode tensor_precision(uint32_t type) {
    switch (type) {
        case GGML_F32: return PRECISION_FLOAT32;
        case GGML_F16: return PRECISION_FP16;
        case GGML_BF16: return PRECISION_BFLOAT16;
        default: return block_precision(type);
    }
}

typedef struct {
    const Gguf *g;
    SpgImageBuilder *b;
    uint8_t *image;
    bool permuted;                  /* LLaMA's queries and keys in llama.cpp's order */
    uint32_t heads, kv_heads, head;
    uint32_t rot;                   /* the rotated values of a head */
    char error[256];
} ImageWriter;

/* A part's tensor, of `cols` values a row and `rows` rows (0: a vector of cols values), of a type that is read; NULL
   with the error set otherwise (optional parts: NULL without an error when the file lacks them). */
static const GgufTensor *part_tensor(ImageWriter *w, SpgPart part, uint32_t block, uint64_t cols, uint64_t rows,
                                     bool optional) {
    char name[64];
    gguf_part_name(part, block, name, sizeof name);
    const GgufTensor *t = find_gguf_tensor(w->g, name);
    if (!t) {
        if (!optional && !w->error[0]) snprintf(w->error, sizeof w->error, "GGUF: the file has no %s", name);
        return NULL;
    }
    const bool fits = rows ? t->ndims == 2 && t->dims[0] == cols && (t->dims[1] == rows ||
                                                                      (part == SPG_PART_POSITIONS && t->dims[1] > rows))
                           : t->ndims == 1 && t->dims[0] == cols;
    if (!fits) {
        if (!w->error[0])
            snprintf(w->error, sizeof w->error, "GGUF: %s has shape [%llu, %llu], not [%llu, %llu]", name,
                     (unsigned long long)t->dims[0], (unsigned long long)(t->ndims > 1 ? t->dims[1] : 1),
                     (unsigned long long)cols, (unsigned long long)(rows ? rows : 1));
        return NULL;
    }
    if (!t->data || tensor_precision(t->type) == PRECISION_COUNT || (!rows && t->type != GGML_F32 && t->type != GGML_F16 &&
                                                                     t->type != GGML_BF16)) {
        if (!w->error[0])
            snprintf(w->error, sizeof w->error, "GGUF: %s is of the type %u, which is not read (F32, F16, BF16, Q4_0, "
                     "Q4_1, Q5_0, Q5_1, Q8_0, Q2_K to Q6_K, IQ4_NL and IQ4_XS are; vectors F32, F16 and BF16)", name, t->type);
        return NULL;
    }
    return t;
}

/* Row r of a tensor of queries or keys (heads of d rows, the first rot of each rotated) in the order of the halves
   the engine rotates: llama.cpp's row h d + 2 i + j of a head's rotated rows (adjacent pairs) holds h d + j rot / 2 +
   i, the rows past rot their own. */
static uint64_t hf_row(uint64_t r, uint32_t d, uint32_t rot) {
    const uint64_t h = r / d, within = r % d;
    if (within >= rot) return r;
    const uint64_t j = within / (rot / 2u), i = within % (rot / 2u);
    return h * d + 2u * i + j;
}

/* rows rows of tensor t from row `first` on into dst (in the order of the halves with `permute`: heads of `head` rows,
   `rot` of them rotated). */
static void copy_rows_from(uint8_t *dst, const GgufTensor *t, uint64_t first, uint64_t rows, bool permute, uint32_t head,
                           uint32_t rot) {
    bool whole;
    const uint64_t bytes = type_bytes(t->type, t->dims[0], &whole);
    const int64_t n = (int64_t)rows;
    const uint8_t *src = t->data + first * bytes;
    SPINGALETT_PARALLEL_FOR(rows * bytes >= (1u << 22),
        for (int64_t r = 0; r < n; r++)
            memcpy(dst + (uint64_t)r * bytes, src + (permute ? hf_row((uint64_t)r, head, rot) : (uint64_t)r) * bytes,
                   (size_t)bytes);
    );
}

static void copy_rows(uint8_t *dst, const GgufTensor *t, uint64_t rows, bool permute, uint32_t head) {
    copy_rows_from(dst, t, 0, rows, permute, head, head);
}

/* n values of a vector tensor (F32, F16, BF16) from value `first` on into dst, in the order of the halves with
   `permute`. */
static void copy_floats_from(float *dst, const GgufTensor *t, uint64_t first, uint64_t n, bool permute, uint32_t head,
                             uint32_t rot) {
    for (uint64_t i = 0; i < n; i++) {
        const uint64_t at = first + (permute ? hf_row(i, head, rot) : i);
        dequantize(t->data + at * (t->type == GGML_F32 ? 4u : 2u), t->type, dst + i, 1);
    }
}

/* The layers of a LLaMA-like or GPT-2 block's linear maps: the queries', keys' and values' in one layer when their
   tensors share a type (else three, concatenated), the attention, its output's and the MLP's. */
typedef struct {
    uint32_t attn_norm, q, k, v, q_norm, k_norm, qkv, attention, o, attn_post, ffn_norm, gate, up, down, ffn_post;
    const GgufTensor *tq, *tk, *tv, *tqkv, *to, *tgate, *tup, *tdown;
} ImageBlock;

static uint32_t add(ImageWriter *w, SpgLayerSpec spec) {
    const uint32_t index = w->error[0] ? 0u : spingalett_image_layer(w->b, &spec);
    if (!index && !w->error[0]) snprintf(w->error, sizeof w->error, "%s", spingalett_last_error_message());
    return index;
}

#define SPEC(...) ((SpgLayerSpec){__VA_ARGS__})
#define INPUTS(...) .input_count = sizeof((uint32_t[]){__VA_ARGS__}) / sizeof(uint32_t), .inputs = {__VA_ARGS__}

uint8_t *spingalett_gguf_image(const uint8_t *data, size_t size, uint32_t context, size_t *image_size) {
    Gguf g;
    if (!gguf_parse(&g, data, size)) {
        set_error(SPINGALETT_ERR_INVALID, g.error);
        gguf_free(&g);
        return NULL;
    }
    SpgTransformer t;
    bool permuted = false;
    float *freqs = NULL;
    ImageWriter w = {.g = &g};
    ImageBlock *blocks = NULL;
    uint8_t *image = NULL;
    if (!gguf_describe(&g, context, false, &t, &permuted, &freqs, w.error, sizeof w.error)) goto done;
    const bool gpt2 = t.arch == SPG_ARCH_GPT2;
    const uint64_t W = t.width, V = t.vocabulary, F = t.hidden, hd = t.head_size;
    const uint64_t Q = (uint64_t)t.heads * hd, KV = (uint64_t)t.kv_heads * hd;
    if (!W || !V || !F || !t.layers || !t.heads || !t.kv_heads || t.heads % t.kv_heads || !hd || (!gpt2 && hd % 2u) ||
        W > (1u << 20) || F > (1u << 20) || Q + 2u * KV > (1u << 20)) {
        snprintf(w.error, sizeof w.error, "GGUF: the model's hyper-parameters are invalid");
        goto done;
    }
    w.permuted = permuted, w.heads = t.heads, w.kv_heads = t.kv_heads, w.head = (uint32_t)hd;
    w.rot = t.rotary ? t.rotary : (uint32_t)hd;
    w.b = spingalett_image_new(t.context, LOSS_SPARSE_CROSS_ENTROPY);
    blocks = (ImageBlock *)calloc(t.layers, sizeof(ImageBlock));
    if (!w.b || !blocks) {
        snprintf(w.error, sizeof w.error, "%s", w.b ? "GGUF: out of memory" : spingalett_last_error_message());
        goto done;
    }
    /* the tensors where the hyper-parameters say, then the layers */
    const GgufTensor *embed = part_tensor(&w, SPG_PART_EMBED, 0, W, V, false);
    const GgufTensor *positions = gpt2 ? part_tensor(&w, SPG_PART_POSITIONS, 0, W, t.context, false) : NULL;
    const GgufTensor *head = part_tensor(&w, SPG_PART_HEAD, 0, W, V, true);
    for (uint32_t b = 0; b < t.layers && !w.error[0]; b++) {
        ImageBlock *k = &blocks[b];
        /* the queries', keys' and values' rows in one tensor (GPT-2's, Phi-3's) or three */
        k->tqkv = part_tensor(&w, SPG_PART_QKV, b, W, Q + 2u * KV, !gpt2);
        if (!k->tqkv) {
            k->tq = part_tensor(&w, SPG_PART_Q, b, W, Q, false);
            k->tk = part_tensor(&w, SPG_PART_K, b, W, KV, false);
            k->tv = part_tensor(&w, SPG_PART_V, b, W, KV, false);
        }
        if (!gpt2 && !t.fused_up) k->tgate = part_tensor(&w, SPG_PART_GATE, b, W, F, false);
        k->to = part_tensor(&w, SPG_PART_O, b, Q, W, false);
        k->tup = part_tensor(&w, SPG_PART_UP, b, W, t.fused_up ? 2u * F : F, false);
        k->tdown = part_tensor(&w, SPG_PART_DOWN, b, F, W, false);
        if (!t.no_pre_norms) {
            (void)part_tensor(&w, SPG_PART_ATTN_NORM, b, W, 0, false);
            (void)part_tensor(&w, SPG_PART_FFN_NORM, b, W, 0, false);
        }
        if (t.qk_norm || t.wide_qk_norm) {
            (void)part_tensor(&w, SPG_PART_Q_NORM, b, t.wide_qk_norm ? Q : hd, 0, false);
            (void)part_tensor(&w, SPG_PART_K_NORM, b, t.wide_qk_norm ? KV : hd, 0, false);
        }
        if (t.post_norms) {
            (void)part_tensor(&w, SPG_PART_ATTN_POST_NORM, b, W, 0, false);
            (void)part_tensor(&w, SPG_PART_FFN_POST_NORM, b, W, 0, false);
        }
    }
    (void)part_tensor(&w, SPG_PART_NORM, 0, W, 0, false);
    if (w.error[0]) goto done;

    const LayerType norm = gpt2 ? LAYER_LAYER_NORM : LAYER_RMS_NORM;
    const uint32_t e = add(&w, SPEC(.type = LAYER_EMBEDDING, .precision = tensor_precision(embed->type), INPUTS(0),
                                    .channels = (uint32_t)W, .vocabulary = (uint32_t)V, .positions = gpt2,
                                    .embed_scale = t.embed_scale));
    uint32_t x = e;
    for (uint32_t b = 0; b < t.layers; b++) {
        ImageBlock *k = &blocks[b];
        const uint32_t in = t.no_pre_norms ? x : (k->attn_norm = add(&w, SPEC(.type = norm, INPUTS(x), .eps = t.eps)));
        if (k->tqkv) {
            k->qkv = add(&w, SPEC(.type = LAYER_CONV2D, .precision = tensor_precision(k->tqkv->type), INPUTS(in),
                                  .channels = (uint32_t)(Q + 2u * KV)));
        } else if (k->tq->type == k->tk->type && k->tq->type == k->tv->type && !t.wide_qk_norm) {
            k->qkv = add(&w, SPEC(.type = LAYER_CONV2D, .precision = tensor_precision(k->tq->type), INPUTS(in),
                                  .channels = (uint32_t)(Q + 2u * KV)));
        } else {
            k->q = add(&w, SPEC(.type = LAYER_CONV2D, .precision = tensor_precision(k->tq->type), INPUTS(in),
                                .channels = (uint32_t)Q));
            k->k = add(&w, SPEC(.type = LAYER_CONV2D, .precision = tensor_precision(k->tk->type), INPUTS(in),
                                .channels = (uint32_t)KV));
            k->v = add(&w, SPEC(.type = LAYER_CONV2D, .precision = tensor_precision(k->tv->type), INPUTS(in),
                                .channels = (uint32_t)KV));
            uint32_t q = k->q, kk = k->k;
            if (t.wide_qk_norm) {
                /* OLMo 2's queries and keys normalized over all their heads, as layers of their own */
                q = k->q_norm = add(&w, SPEC(.type = LAYER_RMS_NORM, INPUTS(k->q), .eps = t.eps));
                kk = k->k_norm = add(&w, SPEC(.type = LAYER_RMS_NORM, INPUTS(k->k), .eps = t.eps));
            }
            k->qkv = add(&w, SPEC(.type = LAYER_CONCAT, INPUTS(q, kk, k->v)));
        }
        /* with a pattern of windows, the last layer of each pattern attends to every key, with the frequencies of the
           model, the others through the window with their own base; SmolLM3's every fourth layer turns nothing */
        const bool windowed = t.window && (!t.window_pattern || b % t.window_pattern < t.window_pattern - 1u);
        const bool turns = !gpt2 && !(t.nope_interval && (b + 1u) % t.nope_interval == 0);
        const float theta = !turns ? 0.0f : windowed && t.window_pattern && t.theta_local > 0.0f ? t.theta_local : t.theta;
        k->attention = add(&w, SPEC(.type = LAYER_ATTENTION, INPUTS(k->qkv), .channels = (uint32_t)Q, .heads = t.heads,
                                    .kv_heads = t.kv_heads, .theta = theta, .causal = true,
                                    .window = windowed ? t.window : 0u, .qk_norm = t.qk_norm, .eps = t.qk_norm ? t.eps : 0.0f,
                                    .rotary = theta > 0.0f ? t.rotary : 0u,
                                    .score_scale = t.score_scale > 0.0f ? t.score_scale / sqrtf((float)hd) : 0.0f,
                                    .freqs = t.freqs != NULL && theta > 0.0f && !(windowed && t.window_pattern)));
        k->o = add(&w, SPEC(.type = LAYER_CONV2D, .precision = tensor_precision(k->to->type), INPUTS(k->attention),
                            .channels = (uint32_t)W));
        uint32_t attended = k->o;
        if (t.post_norms) attended = k->attn_post = add(&w, SPEC(.type = norm, INPUTS(k->o), .eps = t.eps));
        x = add(&w, SPEC(.type = LAYER_ADD, INPUTS(x, attended)));
        const uint32_t ffn_in = t.no_pre_norms ? x : (k->ffn_norm = add(&w, SPEC(.type = norm, INPUTS(x), .eps = t.eps)));
        uint32_t mlp;
        if (gpt2) {
            mlp = k->up = add(&w, SPEC(.type = LAYER_CONV2D, .activation = t.act, .precision = tensor_precision(k->tup->type),
                                       INPUTS(ffn_in), .channels = (uint32_t)F));
        } else {
            const GgufTensor *tgate = t.fused_up ? k->tup : k->tgate;
            k->gate = add(&w, SPEC(.type = LAYER_CONV2D, .activation = t.gate, .precision = tensor_precision(tgate->type),
                                   INPUTS(ffn_in), .channels = (uint32_t)F));
            k->up = add(&w, SPEC(.type = LAYER_CONV2D, .precision = tensor_precision(k->tup->type), INPUTS(ffn_in),
                                 .channels = (uint32_t)F));
            mlp = add(&w, SPEC(.type = LAYER_MULTIPLY, INPUTS(k->gate, k->up)));
        }
        k->down = add(&w, SPEC(.type = LAYER_CONV2D, .precision = tensor_precision(k->tdown->type), INPUTS(mlp),
                               .channels = (uint32_t)W));
        uint32_t fed = k->down;
        if (t.post_norms) fed = k->ffn_post = add(&w, SPEC(.type = norm, INPUTS(k->down), .eps = t.eps));
        x = add(&w, SPEC(.type = LAYER_ADD, INPUTS(x, fed)));
    }
    const uint32_t final_norm = add(&w, SPEC(.type = norm, INPUTS(x), .eps = t.eps));
    const uint32_t logits = add(&w, SPEC(.type = LAYER_CONV2D, .precision = tensor_precision((head ? head : embed)->type),
                                         INPUTS(final_norm), .channels = (uint32_t)V, .tied = head ? 0u : e));
    if (w.error[0]) goto done;
    image = spingalett_image_layout(w.b, image_size);
    if (!image) {
        snprintf(w.error, sizeof w.error, "%s", spingalett_last_error_message());
        goto done;
    }

    /* the parameters, tensor by tensor where the image holds them */
#define WEIGHTS(layer) spingalett_image_weights(w.b, image, (layer), NULL, NULL)
#define BIASES(layer) spingalett_image_biases(w.b, image, (layer))
#define VECTOR(part, b, n, dst, permute) do { \
        const GgufTensor *v_ = part_tensor(&w, (part), (b), (n), 0, true); \
        if (v_) copy_floats_from((dst), v_, 0, (n), (permute), (uint32_t)hd, w.rot); \
    } while (0)
    copy_rows(WEIGHTS(e), embed, V, false, 0);
    if (gpt2) {
        bool whole;
        float *at = BIASES(e);
        for (uint32_t c = 0; c < t.context; c++)
            dequantize(positions->data + (uint64_t)c * type_bytes(positions->type, W, &whole), positions->type,
                       at + (uint64_t)c * W, W);
    }
    for (uint32_t b = 0; b < t.layers; b++) {
        const ImageBlock *k = &blocks[b];
        if (!t.no_pre_norms) VECTOR(SPG_PART_ATTN_NORM, b, W, (float *)(void *)WEIGHTS(k->attn_norm), false);
        if (gpt2) VECTOR(SPG_PART_ATTN_NORM_BIAS, b, W, BIASES(k->attn_norm), false);
        if (k->tqkv) {
            /* the queries', keys' and values' rows of one tensor, the first two in the order of the halves where
               llama.cpp turns adjacent pairs */
            bool whole;
            const uint64_t row = type_bytes(k->tqkv->type, W, &whole);
            uint8_t *rows = WEIGHTS(k->qkv);
            float *bias = BIASES(k->qkv);
            copy_rows_from(rows, k->tqkv, 0, Q, w.permuted, (uint32_t)hd, w.rot);
            copy_rows_from(rows + Q * row, k->tqkv, Q, KV, w.permuted, (uint32_t)hd, w.rot);
            copy_rows_from(rows + (Q + KV) * row, k->tqkv, Q + KV, KV, false, 0, 0);
            const GgufTensor *vb = part_tensor(&w, SPG_PART_QKV_BIAS, b, Q + 2u * KV, 0, true);
            if (vb) {
                copy_floats_from(bias, vb, 0, Q, w.permuted, (uint32_t)hd, w.rot);
                copy_floats_from(bias + Q, vb, Q, KV, w.permuted, (uint32_t)hd, w.rot);
                copy_floats_from(bias + Q + KV, vb, Q + KV, KV, false, 0, 0);
            }
        } else {
            /* the queries', keys' and values' rows (and biases), in one layer or three */
            const bool fused = k->q == 0;
            uint8_t *rows[3] = {WEIGHTS(fused ? k->qkv : k->q), NULL, NULL};
            float *biases[3] = {BIASES(fused ? k->qkv : k->q), NULL, NULL};
            bool whole;
            const uint64_t qrow = type_bytes(k->tq->type, W, &whole), krow = type_bytes(k->tk->type, W, &whole);
            rows[1] = fused ? rows[0] + Q * qrow : WEIGHTS(k->k);
            rows[2] = fused ? rows[1] + KV * krow : WEIGHTS(k->v);
            biases[1] = fused ? biases[0] + Q : BIASES(k->k);
            biases[2] = fused ? biases[1] + KV : BIASES(k->v);
            copy_rows_from(rows[0], k->tq, 0, Q, w.permuted, (uint32_t)hd, w.rot);
            copy_rows_from(rows[1], k->tk, 0, KV, w.permuted, (uint32_t)hd, w.rot);
            copy_rows(rows[2], k->tv, KV, false, 0);
            VECTOR(SPG_PART_Q_BIAS, b, Q, biases[0], w.permuted);
            VECTOR(SPG_PART_K_BIAS, b, KV, biases[1], w.permuted);
            VECTOR(SPG_PART_V_BIAS, b, KV, biases[2], false);
        }
        if (t.qk_norm) {
            float *gammas = (float *)(void *)WEIGHTS(k->attention);
            VECTOR(SPG_PART_Q_NORM, b, hd, gammas, false);
            VECTOR(SPG_PART_K_NORM, b, hd, gammas + hd, false);
        }
        if (t.wide_qk_norm) {
            VECTOR(SPG_PART_Q_NORM, b, Q, (float *)(void *)WEIGHTS(k->q_norm), w.permuted);
            VECTOR(SPG_PART_K_NORM, b, KV, (float *)(void *)WEIGHTS(k->k_norm), w.permuted);
        }
        float *freqs_at = spingalett_image_scales(w.b, image, k->attention);
        if (t.freqs && freqs_at) memcpy(freqs_at, t.freqs, (size_t)(w.rot / 2u) * sizeof(float));
        if (t.post_norms) {
            VECTOR(SPG_PART_ATTN_POST_NORM, b, W, (float *)(void *)WEIGHTS(k->attn_post), false);
            VECTOR(SPG_PART_FFN_POST_NORM, b, W, (float *)(void *)WEIGHTS(k->ffn_post), false);
        }
        copy_rows(WEIGHTS(k->o), k->to, W, false, 0);
        VECTOR(SPG_PART_O_BIAS, b, W, BIASES(k->o), false);
        if (!t.no_pre_norms) VECTOR(SPG_PART_FFN_NORM, b, W, (float *)(void *)WEIGHTS(k->ffn_norm), false);
        if (gpt2) VECTOR(SPG_PART_FFN_NORM_BIAS, b, W, BIASES(k->ffn_norm), false);
        if (t.fused_up) {
            /* the gate's rows, then the up's, of one tensor */
            copy_rows_from(WEIGHTS(k->gate), k->tup, 0, F, false, 0, 0);
            copy_rows_from(WEIGHTS(k->up), k->tup, F, F, false, 0, 0);
        } else {
            if (!gpt2) {
                copy_rows(WEIGHTS(k->gate), k->tgate, F, false, 0);
                VECTOR(SPG_PART_GATE_BIAS, b, F, BIASES(k->gate), false);
            }
            copy_rows(WEIGHTS(k->up), k->tup, F, false, 0);
            VECTOR(SPG_PART_UP_BIAS, b, F, BIASES(k->up), false);
        }
        copy_rows(WEIGHTS(k->down), k->tdown, W, false, 0);
        VECTOR(SPG_PART_DOWN_BIAS, b, W, BIASES(k->down), false);
    }
    VECTOR(SPG_PART_NORM, 0, W, (float *)(void *)WEIGHTS(final_norm), false);
    if (gpt2) VECTOR(SPG_PART_NORM_BIAS, 0, W, BIASES(final_norm), false);
    if (head) copy_rows(WEIGHTS(logits), head, V, false, 0);
#undef WEIGHTS
#undef BIASES
#undef VECTOR
    spingalett_image_seal(image, *image_size);
done:
    if (w.error[0]) {
        spingalett_aligned_free(image);
        image = NULL;
        set_error(SPINGALETT_ERR_INVALID, w.error);
    }
    spingalett_image_free(w.b);
    free(blocks);
    free(freqs);
    gguf_free(&g);
    return image;
}

#undef SPEC
#undef INPUTS

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
    const Kv *chat = find_kv(&g, "tokenizer.chat_template");
    if (chat && chat->type == GGUF_STRING && slett_get64(chat->value) < (1u << 24)) {
        spec.chat_template = (const char *)chat->value + 8;
        spec.chat_template_length = (uint32_t)slett_get64(chat->value);
    }
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
