/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/*
 * .slettd data set files (layout: docs/DatasetFormat.md).
 *
 * Each chunk holds two streams, inputs and targets. A stream is the encoded values of the chunk
 * split into byte planes (byte k of every value together), either stored or coded with an
 * adaptive binary range coder. The coder models every byte from the previous byte and from the
 * bytes `stride` and `stride + 1` positions back, where the writer picks the stride that predicts
 * best (for 28x28 images it finds the row length, so the pixel above is part of the context).
 */

#include "Spingalett.Private.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#if defined(_OPENMP)
#include <omp.h>
#endif

#define SLETTD_VERSION        1
#define SLETTD_HEADER_SIZE    64
#define SLETTD_INDEX_ENTRY    20
#define SLETTD_CHUNK_TARGET   (1u << 20)    /* encoded bytes per chunk the writer aims for */

enum { STREAM_STORED = 0, STREAM_CODED = 1 };

/* ---------------------------------------------------------------- little-endian helpers */

static void put16(uint8_t *p, uint16_t v) { p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8); }
static void put32(uint8_t *p, uint32_t v) { for (int i = 0; i < 4; i++) p[i] = (uint8_t)(v >> (8 * i)); }
static void put64(uint8_t *p, uint64_t v) { for (int i = 0; i < 8; i++) p[i] = (uint8_t)(v >> (8 * i)); }
static uint16_t get16(const uint8_t *p) { return (uint16_t)(p[0] | p[1] << 8); }
static uint32_t get32(const uint8_t *p) { return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24; }
static uint64_t get64(const uint8_t *p) { return (uint64_t)get32(p) | (uint64_t)get32(p + 4) << 32; }
static uint32_t float_bits(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
static float bits_float(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }

/* ---------------------------------------------------------------- CRC-32 (IEEE, reflected) */

static uint32_t crc_table[256];
static bool crc_ready;

static void crc_init(void) {
    if (crc_ready) return;
    for (uint32_t i = 0; i < 256; i++) {
        uint32_t c = i;
        for (int k = 0; k < 8; k++) c = (c >> 1) ^ (0xEDB88320u & (0u - (c & 1u)));
        crc_table[i] = c;
    }
    crc_ready = true;
}

static uint32_t crc32_update(uint32_t crc, const uint8_t *p, size_t n) {
    crc = ~crc;
    for (size_t i = 0; i < n; i++) crc = crc_table[(crc ^ p[i]) & 0xFFu] ^ (crc >> 8);
    return ~crc;
}

/* ---------------------------------------------------------------- range coder */

#define RC_TOP      (1u << 24)
#define PROB_BITS   11
#define PROB_ONE    (1u << PROB_BITS)
#define MOVE_BITS   4

typedef struct {
    uint8_t *out;
    size_t pos, cap;
    uint64_t low;
    uint32_t range;
    uint8_t cache;
    uint64_t cache_size;
    bool overflow;
} RcEncoder;

static void rc_put(RcEncoder *e, uint8_t b) {
    if (e->pos < e->cap) e->out[e->pos++] = b;
    else e->overflow = true;
}

static void rc_shift_low(RcEncoder *e) {
    if ((uint32_t)e->low < 0xFF000000u || (e->low >> 32) != 0) {
        uint8_t carry = (uint8_t)(e->low >> 32), temp = e->cache;
        do {
            rc_put(e, (uint8_t)(temp + carry));
            temp = 0xFF;
        } while (--e->cache_size != 0);
        e->cache = (uint8_t)(e->low >> 24);
    }
    e->cache_size++;
    e->low = (e->low & 0x00FFFFFFu) << 8;
}

static inline void rc_encode_bit(RcEncoder *e, uint16_t *p, unsigned bit) {
    uint32_t bound = (e->range >> PROB_BITS) * *p;
    if (!bit) {
        e->range = bound;
        *p = (uint16_t)(*p + ((PROB_ONE - *p) >> MOVE_BITS));
    } else {
        e->low += bound;
        e->range -= bound;
        *p = (uint16_t)(*p - (*p >> MOVE_BITS));
    }
    while (e->range < RC_TOP) {
        e->range <<= 8;
        rc_shift_low(e);
    }
}

typedef struct {
    const uint8_t *in;
    size_t pos, size;
    uint32_t range, code;
    bool overrun;
} RcDecoder;

static inline uint8_t rc_get(RcDecoder *d) {
    if (d->pos < d->size) return d->in[d->pos++];
    d->overrun = true;
    return 0;
}

static inline unsigned rc_decode_bit(RcDecoder *d, uint16_t *p) {
    uint32_t bound = (d->range >> PROB_BITS) * *p;
    unsigned bit;
    if (d->code < bound) {
        d->range = bound;
        *p = (uint16_t)(*p + ((PROB_ONE - *p) >> MOVE_BITS));
        bit = 0;
    } else {
        d->code -= bound;
        d->range -= bound;
        *p = (uint16_t)(*p - (*p >> MOVE_BITS));
        bit = 1;
    }
    if (d->range < RC_TOP) {
        d->range <<= 8;
        d->code = (d->code << 8) | rc_get(d);
    }
    return bit;
}

/* ---------------------------------------------------------------- byte model */

/* "Is the byte zero?" in a context of the previous byte and the two bytes at the stride, then the
   value of a non-zero byte as 8 binary decisions down a bit tree, in a smaller context. */
#define CTX_ZERO 2048
#define CTX_TREE 256

typedef struct {
    uint16_t zero[CTX_ZERO];
    uint16_t tree[CTX_TREE][256];
} ByteModel;

static void model_reset(ByteModel *m) {
    for (size_t i = 0; i < CTX_ZERO; i++) m->zero[i] = PROB_ONE / 2;
    for (size_t c = 0; c < CTX_TREE; c++)
        for (size_t i = 0; i < 256; i++) m->tree[c][i] = PROB_ONE / 2;
}

static inline void contexts(const uint8_t *b, size_t i, size_t stride, uint32_t *cz, uint32_t *ct) {
    uint32_t p1 = i >= 1 ? b[i - 1] : 0;
    uint32_t ps = i >= stride ? b[i - stride] : 0;
    uint32_t ps1 = i >= stride + 1 ? b[i - stride - 1] : 0;
    *cz = (p1 >> 4) << 7 | (ps >> 4) << 3 | ps1 >> 5;
    *ct = (p1 >> 4) << 4 | ps >> 4;
}

static void encode_plane(RcEncoder *e, ByteModel *m, const uint8_t *b, size_t n, size_t stride) {
    model_reset(m);
    for (size_t i = 0; i < n; i++) {
        uint32_t cz, ct;
        contexts(b, i, stride, &cz, &ct);
        unsigned v = b[i];
        rc_encode_bit(e, &m->zero[cz], v != 0);
        if (v) {
            uint16_t *t = m->tree[ct];
            unsigned node = 1;
            for (int k = 7; k >= 0; k--) {
                unsigned bit = (v >> k) & 1u;
                rc_encode_bit(e, &t[node], bit);
                node = node << 1 | bit;
            }
        }
    }
}

static void decode_plane(RcDecoder *d, ByteModel *m, uint8_t *b, size_t n, size_t stride) {
    model_reset(m);
    for (size_t i = 0; i < n; i++) {
        uint32_t cz, ct;
        contexts(b, i, stride, &cz, &ct);
        unsigned v = 0;
        if (rc_decode_bit(d, &m->zero[cz])) {
            uint16_t *t = m->tree[ct];
            unsigned node = 1;
            for (int k = 0; k < 8; k++) node = node << 1 | rc_decode_bit(d, &t[node]);
            v = node & 0xFFu;
        }
        b[i] = (uint8_t)v;
    }
}

/* Codes `planes` planes of n bytes each into out (at most cap bytes); returns the size, or 0 when
   the result would not be smaller than cap. */
static size_t encode_stream(const uint8_t *data, size_t n, unsigned planes, size_t stride,
                            ByteModel *m, uint8_t *out, size_t cap) {
    RcEncoder e = {.out = out, .cap = cap, .range = 0xFFFFFFFFu, .cache_size = 1};
    for (unsigned p = 0; p < planes && !e.overflow; p++)
        encode_plane(&e, m, data + p * n, n, stride);
    for (int i = 0; i < 5; i++) rc_shift_low(&e);
    return e.overflow ? 0 : e.pos;
}

static bool decode_stream(const uint8_t *in, size_t size, uint8_t *data, size_t n, unsigned planes,
                          size_t stride, ByteModel *m) {
    RcDecoder d = {.in = in, .size = size, .range = 0xFFFFFFFFu};
    for (int i = 0; i < 5; i++) d.code = (d.code << 8) | rc_get(&d);
    for (unsigned p = 0; p < planes && !d.overrun; p++)
        decode_plane(&d, m, data + p * n, n, stride);
    return !d.overrun;
}

/* Picks the stride whose context predicts a sample of the stream best (static order-1 entropy
   estimate over the first plane bytes of the first chunk). */
static size_t choose_stride(const uint8_t *data, size_t n, unsigned planes, uint32_t width) {
    size_t cand[64], nc = 0;
    for (size_t s = 1; s <= 4; s++) cand[nc++] = s;
    for (uint32_t d = 5; d <= width && d <= 4096 && nc + 3 <= 60; d++)
        if (width % d == 0) {
            cand[nc++] = d - 1;
            cand[nc++] = d;
            cand[nc++] = d + 1;
        }
    if (width > 4096 && width <= (1u << 20)) cand[nc++] = width;

    size_t sample = n < (1u << 18) ? n : (1u << 18);
    uint32_t *counts = (uint32_t *)malloc(CTX_TREE * 256 * sizeof(uint32_t));
    uint32_t *totals = (uint32_t *)malloc(CTX_TREE * sizeof(uint32_t));
    size_t best = 1;
    double best_cost = INFINITY;
    for (size_t c = 0; counts && totals && c < nc; c++) {
        size_t s = cand[c];
        if (s >= sample) continue;
        double cost = 0;
        for (unsigned p = 0; p < planes; p++) {
            const uint8_t *b = data + p * n;
            memset(counts, 0, CTX_TREE * 256 * sizeof(uint32_t));
            memset(totals, 0, CTX_TREE * sizeof(uint32_t));
            for (size_t i = 0; i < sample; i++) {
                uint32_t cz, ct;
                contexts(b, i, s, &cz, &ct);
                counts[ct * 256 + b[i]]++;
                totals[ct]++;
            }
            for (size_t k = 0; k < (size_t)CTX_TREE * 256; k++)
                if (counts[k]) cost -= counts[k] * log2((double)counts[k] / totals[k / 256]);
        }
        if (cost < best_cost) { best_cost = cost; best = s; }
    }
    free(counts);
    free(totals);
    return best;
}

/* ---------------------------------------------------------------- value encodings */

static unsigned value_width(DatasetEncoding e, uint32_t target_size) {
    switch (e) {
        case DATASET_ENCODING_FLOAT32: return 4;
        case DATASET_ENCODING_FP16:
        case DATASET_ENCODING_BFLOAT16: return 2;
        case DATASET_ENCODING_CLASS: return target_size <= 256 ? 1 : 2;
        default: return 1;
    }
}

/* values per sample in a stream */
static uint32_t stream_width(DatasetEncoding e, uint32_t size) {
    return e == DATASET_ENCODING_CLASS ? 1 : size;
}

static float unit_lut[256];
static bool unit_ready;

static void unit_init(void) {
    if (unit_ready) return;
    for (int q = 0; q < 256; q++) unit_lut[q] = (float)((double)q / 255.0);   /* exactly q / 255.0f */
    unit_ready = true;
}

static int unit_quantize(float v) {
    if (!(v >= 0.0f && v <= 1.0f)) return v > 1.0f ? 255 : 0;
    int q = (int)floor((double)v * 255.0 + 0.5);
    return q > 255 ? 255 : q;
}

static bool all_unit(const float *v, size_t n) {
    for (size_t i = 0; i < n; i++) {
        if (!(v[i] >= 0.0f && v[i] <= 1.0f)) return false;
        if (unit_lut[unit_quantize(v[i])] != v[i]) return false;
    }
    return true;
}

static bool all_half(const float *v, size_t n) {
    for (size_t i = 0; i < n; i++)
        if (spingalett_fp16_to_float(spingalett_float_to_fp16(v[i])) != v[i]) return false;
    return true;
}

static bool all_one_hot(const float *v, uint32_t rows, uint32_t width) {
    if (width < 2) return false;
    for (uint32_t r = 0; r < rows; r++) {
        uint32_t ones = 0;
        for (uint32_t k = 0; k < width; k++) {
            float x = v[(size_t)r * width + k];
            if (x == 1.0f) ones++;
            else if (x != 0.0f) return false;
        }
        if (ones != 1) return false;
    }
    return true;
}

static DatasetEncoding choose_encoding(const float *v, uint32_t rows, uint32_t width, bool targets) {
    size_t n = (size_t)rows * width;
    if (targets && all_one_hot(v, rows, width) && width <= 65536) return DATASET_ENCODING_CLASS;
    if (all_unit(v, n)) return DATASET_ENCODING_U8_UNIT;
    if (all_half(v, n)) return DATASET_ENCODING_FP16;
    return DATASET_ENCODING_FLOAT32;
}

/* U8_AFFINE parameters: per feature min and step. */
static void affine_params(const float *v, uint32_t rows, uint32_t width, float *params) {
    for (uint32_t f = 0; f < width; f++) {
        float lo = INFINITY, hi = -INFINITY;
        for (uint32_t r = 0; r < rows; r++) {
            float x = v[(size_t)r * width + f];
            if (x < lo) lo = x;
            if (x > hi) hi = x;
        }
        if (!(lo <= hi)) lo = hi = 0.0f;        /* all NaN */
        params[2 * f] = lo;
        params[2 * f + 1] = (hi - lo) / 255.0f;
    }
}

/* Encodes rows [0, rows) of src (row width `size`) into byte planes of n = rows * stream width values. */
static void encode_values(DatasetEncoding e, const float *src, uint32_t rows, uint32_t size,
                          const float *params, uint8_t *planes) {
    uint32_t w = stream_width(e, size);
    size_t n = (size_t)rows * w;
    for (size_t i = 0; i < n; i++) {
        switch (e) {
            case DATASET_ENCODING_FLOAT32: {
                uint32_t u = float_bits(src[i]);
                for (int k = 0; k < 4; k++) planes[k * n + i] = (uint8_t)(u >> (8 * k));
                break;
            }
            case DATASET_ENCODING_FP16:
            case DATASET_ENCODING_BFLOAT16: {
                uint16_t h = e == DATASET_ENCODING_FP16 ? spingalett_float_to_fp16(src[i]) : spingalett_float_to_bf16(src[i]);
                planes[i] = (uint8_t)h;
                planes[n + i] = (uint8_t)(h >> 8);
                break;
            }
            case DATASET_ENCODING_U8_UNIT:
                planes[i] = (uint8_t)unit_quantize(src[i]);
                break;
            case DATASET_ENCODING_U8_AFFINE: {
                uint32_t f = (uint32_t)(i % size);
                float lo = params[2 * f], step = params[2 * f + 1];
                double q = step > 0.0f ? floor(((double)src[i] - lo) / step + 0.5) : 0.0;
                planes[i] = (uint8_t)(q < 0 ? 0 : q > 255 ? 255 : q);
                break;
            }
            case DATASET_ENCODING_CLASS: {
                const float *row = src + i * size;
                uint32_t best = 0;
                for (uint32_t k = 1; k < size; k++) if (row[k] > row[best]) best = k;
                planes[i] = (uint8_t)best;
                if (size > 256) planes[n + i] = (uint8_t)(best >> 8);
                break;
            }
            default:
                break;
        }
    }
}

static bool decode_values(DatasetEncoding e, const uint8_t *planes, uint32_t rows, uint32_t size,
                          const float *params, float *dst) {
    uint32_t w = stream_width(e, size);
    size_t n = (size_t)rows * w;
    if (e == DATASET_ENCODING_CLASS)
        memset(dst, 0, (size_t)rows * size * sizeof(float));
    for (size_t i = 0; i < n; i++) {
        switch (e) {
            case DATASET_ENCODING_FLOAT32: {
                uint32_t u = 0;
                for (int k = 0; k < 4; k++) u |= (uint32_t)planes[k * n + i] << (8 * k);
                dst[i] = bits_float(u);
                break;
            }
            case DATASET_ENCODING_FP16:
                dst[i] = spingalett_fp16_to_float((uint16_t)(planes[i] | planes[n + i] << 8));
                break;
            case DATASET_ENCODING_BFLOAT16:
                dst[i] = spingalett_bf16_to_float((uint16_t)(planes[i] | planes[n + i] << 8));
                break;
            case DATASET_ENCODING_U8_UNIT:
                dst[i] = unit_lut[planes[i]];
                break;
            case DATASET_ENCODING_U8_AFFINE: {
                uint32_t f = (uint32_t)(i % size);
                dst[i] = params[2 * f] + (float)planes[i] * params[2 * f + 1];
                break;
            }
            case DATASET_ENCODING_CLASS: {
                uint32_t c = planes[i] | (size > 256 ? (uint32_t)planes[n + i] << 8 : 0u);
                if (c >= size) return false;
                dst[i * size + c] = 1.0f;
                break;
            }
            default:
                return false;
        }
    }
    return true;
}

/* ---------------------------------------------------------------- file header */

typedef struct {
    uint32_t count, input_size, target_size;
    DatasetEncoding enc[2];             /* inputs, targets */
    uint32_t stride[2];
    uint32_t chunk_samples, chunk_count;
    uint64_t file_size;
    float *params[2];                   /* U8_AFFINE: min, step per feature, or NULL */
    uint64_t *offset;                   /* per chunk */
    uint32_t *bytes[2];                 /* stream sizes per chunk */
    uint32_t *crc;
} Layout;

static void layout_free(Layout *L) {
    free(L->params[0]); free(L->params[1]);
    free(L->offset); free(L->bytes[0]); free(L->bytes[1]); free(L->crc);
    memset(L, 0, sizeof *L);
}

static uint32_t stream_size(const Layout *L, int s) { return s ? L->target_size : L->input_size; }

static size_t params_bytes(const Layout *L) {
    size_t b = 0;
    for (int s = 0; s < 2; s++)
        if (L->enc[s] == DATASET_ENCODING_U8_AFFINE) b += (size_t)stream_size(L, s) * 8;
    return b;
}

static uint32_t chunk_rows(const Layout *L, uint32_t c) {
    uint64_t start = (uint64_t)c * L->chunk_samples;
    uint64_t left = L->count - start;
    return left < L->chunk_samples ? (uint32_t)left : L->chunk_samples;
}

/* raw (decoded) bytes of stream s for `rows` samples */
static uint64_t stream_raw(const Layout *L, int s, uint32_t rows) {
    return (uint64_t)rows * stream_width(L->enc[s], stream_size(L, s)) * value_width(L->enc[s], L->target_size);
}

static bool fail(int code, const char *message) {
    set_error(code, message);
    spingalett_log(LOG_ERROR, "%s", message);
    return false;
}

/* Parses header, parameters and chunk index from the first `size` bytes of a file (`size` may be
   just the header, in which case *need tells how many bytes the metadata takes). */
static bool parse_header(const uint8_t *h, size_t size, Layout *L, size_t *need) {
    memset(L, 0, sizeof *L);
    if (size < SLETTD_HEADER_SIZE || memcmp(h, "SLETTD", 6) != 0)
        return fail(SPINGALETT_ERR_INVALID, "not a .slettd data set file");
    if (get16(h + 6) != SLETTD_VERSION)
        return fail(SPINGALETT_ERR_FORMAT_VERSION, "unsupported .slettd format version");
    if (crc32_update(0, h, 60) != get32(h + 60))
        return fail(SPINGALETT_ERR_INVALID, "corrupt .slettd header (checksum mismatch)");
    L->count = get32(h + 8);
    L->input_size = get32(h + 12);
    L->target_size = get32(h + 16);
    L->enc[0] = (DatasetEncoding)h[20];
    L->enc[1] = (DatasetEncoding)h[21];
    L->chunk_samples = get32(h + 24);
    L->chunk_count = get32(h + 28);
    L->stride[0] = get32(h + 32);
    L->stride[1] = get32(h + 36);
    L->file_size = get64(h + 40);
    uint32_t pbytes = get32(h + 48);
    bool ok = L->input_size > 0 && L->target_size > 0 && L->chunk_samples > 0 &&
              L->enc[0] > DATASET_ENCODING_AUTO && L->enc[0] < DATASET_ENCODING_CLASS &&
              L->enc[1] > DATASET_ENCODING_AUTO && L->enc[1] < DATASET_ENCODING_COUNT &&
              (L->enc[1] != DATASET_ENCODING_CLASS || L->target_size <= 65536) &&
              L->chunk_count == (uint32_t)(((uint64_t)L->count + L->chunk_samples - 1) / L->chunk_samples) &&
              L->stride[0] > 0 && L->stride[1] > 0 && pbytes == params_bytes(L) &&
              stream_raw(L, 0, L->chunk_samples) < ((uint64_t)1 << 32) &&
              stream_raw(L, 1, L->chunk_samples) < ((uint64_t)1 << 32);
    if (!ok)
        return fail(SPINGALETT_ERR_INVALID, "corrupt .slettd header (inconsistent fields)");
    size_t meta = SLETTD_HEADER_SIZE + pbytes + (size_t)L->chunk_count * SLETTD_INDEX_ENTRY + 4;
    *need = meta;
    if (size < meta) return true;       /* caller reads the rest and parses again */

    if (crc32_update(0, h + SLETTD_HEADER_SIZE, meta - SLETTD_HEADER_SIZE - 4) != get32(h + meta - 4))
        return fail(SPINGALETT_ERR_INVALID, "corrupt .slettd index (checksum mismatch)");
    const uint8_t *p = h + SLETTD_HEADER_SIZE;
    for (int s = 0; s < 2; s++) {
        if (L->enc[s] != DATASET_ENCODING_U8_AFFINE) continue;
        uint32_t n = stream_size(L, s);
        L->params[s] = (float *)malloc((size_t)n * 2 * sizeof(float));
        if (!L->params[s]) {
            layout_free(L);
            return fail(SPINGALETT_ERR_ALLOC, "out of memory reading a .slettd file");
        }
        for (uint32_t i = 0; i < 2 * n; i++, p += 4) L->params[s][i] = bits_float(get32(p));
    }
    uint32_t chunks = L->chunk_count;
    L->offset = (uint64_t *)malloc((chunks + 1) * sizeof(uint64_t));
    L->bytes[0] = (uint32_t *)malloc((chunks + 1) * sizeof(uint32_t));
    L->bytes[1] = (uint32_t *)malloc((chunks + 1) * sizeof(uint32_t));
    L->crc = (uint32_t *)malloc((chunks + 1) * sizeof(uint32_t));
    if (!L->offset || !L->bytes[0] || !L->bytes[1] || !L->crc) {
        layout_free(L);
        return fail(SPINGALETT_ERR_ALLOC, "out of memory reading a .slettd file");
    }
    for (uint32_t c = 0; c < chunks; c++, p += SLETTD_INDEX_ENTRY) {
        L->offset[c] = get64(p);
        L->bytes[0][c] = get32(p + 8);
        L->bytes[1][c] = get32(p + 12);
        L->crc[c] = get32(p + 16);
        uint64_t end = L->offset[c] + L->bytes[0][c] + L->bytes[1][c];
        /* the coder cannot compress more than about 750:1 (saturated probabilities), so a chunk that
           claims more is corrupt; this also stops tiny crafted files from requesting huge buffers */
        uint32_t rows = chunk_rows(L, c);
        bool plausible = stream_raw(L, 0, rows) <= (uint64_t)L->bytes[0][c] * 4096 &&
                         stream_raw(L, 1, rows) <= (uint64_t)L->bytes[1][c] * 4096;
        if (L->offset[c] < meta || end > L->file_size || L->bytes[0][c] == 0 || L->bytes[1][c] == 0 || !plausible) {
            layout_free(L);
            return fail(SPINGALETT_ERR_INVALID, "corrupt .slettd index (chunk outside the file)");
        }
    }
    return true;
}

/* ---------------------------------------------------------------- chunk decoding */

typedef struct {
    uint8_t *planes;                    /* decoded stream bytes */
    size_t planes_cap;
    ByteModel *model;
} Scratch;

static void scratch_free(Scratch *s) {
    free(s->planes);
    free(s->model);
}

static bool scratch_reserve(Scratch *s, size_t bytes) {
    if (!s->model && !(s->model = (ByteModel *)malloc(sizeof(ByteModel)))) return false;
    if (bytes <= s->planes_cap) return true;
    uint8_t *p = (uint8_t *)realloc(s->planes, bytes);
    if (!p) return false;
    s->planes = p;
    s->planes_cap = bytes;
    return true;
}

/* Decodes chunk c, whose bytes are at data, into rows of inputs/targets. */
static bool decode_chunk(const Layout *L, uint32_t c, const uint8_t *data, float *inputs, float *targets,
                         Scratch *sc) {
    uint32_t rows = chunk_rows(L, c);
    if (crc32_update(0, data, (size_t)L->bytes[0][c] + L->bytes[1][c]) != L->crc[c])
        return fail(SPINGALETT_ERR_INVALID, "corrupt .slettd chunk (checksum mismatch)");
    const uint8_t *p = data;
    for (int s = 0; s < 2; s++) {
        size_t raw = (size_t)stream_raw(L, s, rows), size = L->bytes[s][c];
        unsigned planes = value_width(L->enc[s], L->target_size);
        if (!scratch_reserve(sc, raw + 1))
            return fail(SPINGALETT_ERR_ALLOC, "out of memory decoding a .slettd chunk");
        bool ok;
        if (p[0] == STREAM_STORED) {
            ok = size == raw + 1;
            if (ok) memcpy(sc->planes, p + 1, raw);
        } else if (p[0] == STREAM_CODED) {
            ok = decode_stream(p + 1, size - 1, sc->planes, raw / planes, planes, L->stride[s], sc->model);
        } else {
            ok = false;
        }
        ok = ok && decode_values(L->enc[s], sc->planes, rows, stream_size(L, s), L->params[s], s ? targets : inputs);
        if (!ok) return fail(SPINGALETT_ERR_INVALID, "corrupt .slettd chunk (stream does not decode)");
        p += size;
    }
    return true;
}

/* ---------------------------------------------------------------- writing */

static char *with_extension(const char *path) {
    const char *dot = strrchr(path, '.');
    const char *slash = strrchr(path, '/'), *backslash = strrchr(path, '\\');
    const char *sep = slash > backslash ? slash : backslash;
    bool has_ext = dot && (!sep || dot > sep);
    size_t len = strlen(path);
    char *out = (char *)malloc(len + sizeof SPINGALETT_DATASET_EXTENSION);
    if (!out) return NULL;
    memcpy(out, path, len + 1);
    if (!has_ext) strcat(out, SPINGALETT_DATASET_EXTENSION);
    return out;
}

bool spingalett_save_dataset(const SpingalettDataset *d, const char *path, const DatasetSaveOptions *options) {
    crc_init();
    unit_init();
    DatasetSaveOptions opt = options ? *options : (DatasetSaveOptions){0};
    if (!d || !path || !d->inputs || !d->targets || d->count == 0 || d->input_size == 0 || d->target_size == 0)
        return fail(SPINGALETT_ERR_INVALID, "spingalett_save_dataset: empty or NULL data set or path");
    if ((unsigned)opt.input_encoding >= DATASET_ENCODING_COUNT || opt.input_encoding == DATASET_ENCODING_CLASS ||
        (unsigned)opt.target_encoding >= DATASET_ENCODING_COUNT ||
        (opt.target_encoding == DATASET_ENCODING_CLASS && d->target_size > 65536))
        return fail(SPINGALETT_ERR_INVALID, "spingalett_save_dataset: invalid encoding");

    Layout L = {.count = d->count, .input_size = d->input_size, .target_size = d->target_size};
    const float *src[2] = {d->inputs, d->targets};
    L.enc[0] = opt.input_encoding ? opt.input_encoding : choose_encoding(d->inputs, d->count, d->input_size, false);
    L.enc[1] = opt.target_encoding ? opt.target_encoding : choose_encoding(d->targets, d->count, d->target_size, true);

    uint64_t sample_bytes = stream_raw(&L, 0, 1) + stream_raw(&L, 1, 1);
    uint64_t per_chunk = SLETTD_CHUNK_TARGET / sample_bytes;
    L.chunk_samples = per_chunk == 0 ? 1 : per_chunk > d->count ? d->count : (uint32_t)per_chunk;
    L.chunk_count = (uint32_t)(((uint64_t)d->count + L.chunk_samples - 1) / L.chunk_samples);
    if (stream_raw(&L, 0, L.chunk_samples) >= ((uint64_t)1 << 32) || stream_raw(&L, 1, L.chunk_samples) >= ((uint64_t)1 << 32))
        return fail(SPINGALETT_ERR_INVALID, "spingalett_save_dataset: a single sample exceeds 4 GB");

    bool ok = true;
    for (int s = 0; s < 2 && ok; s++)
        if (L.enc[s] == DATASET_ENCODING_U8_AFFINE) {
            L.params[s] = (float *)malloc((size_t)stream_size(&L, s) * 2 * sizeof(float));
            ok = L.params[s] != NULL;
            if (ok) affine_params(src[s], d->count, stream_size(&L, s), L.params[s]);
        }
    size_t raw_max = (size_t)(stream_raw(&L, 0, L.chunk_samples) > stream_raw(&L, 1, L.chunk_samples)
                              ? stream_raw(&L, 0, L.chunk_samples) : stream_raw(&L, 1, L.chunk_samples));
    uint8_t *planes = (uint8_t *)malloc(raw_max + 1), *coded = (uint8_t *)malloc(raw_max + 1);
    ByteModel *model = (ByteModel *)malloc(sizeof(ByteModel));
    size_t pbytes = params_bytes(&L), meta = SLETTD_HEADER_SIZE + pbytes + (size_t)L.chunk_count * SLETTD_INDEX_ENTRY + 4;
    uint8_t *head = (uint8_t *)calloc(meta, 1);
    char *filename = with_extension(path);
    ok = ok && planes && coded && model && head && filename;
    if (!ok) {
        free(planes); free(coded); free(model); free(head); free(filename); layout_free(&L);
        return fail(SPINGALETT_ERR_ALLOC, "spingalett_save_dataset: out of memory");
    }

    /* strides from the first chunk */
    for (int s = 0; s < 2; s++) {
        L.stride[s] = 1;
        if (opt.no_compression) continue;
        uint32_t rows = chunk_rows(&L, 0);
        encode_values(L.enc[s], src[s], rows, stream_size(&L, s), L.params[s], planes);
        unsigned w = value_width(L.enc[s], L.target_size);
        L.stride[s] = (uint32_t)choose_stride(planes, (size_t)stream_raw(&L, s, rows) / w, w,
                                              stream_width(L.enc[s], stream_size(&L, s)));
    }

    FILE *f = fopen(filename, "wb");
    if (!f) {
        spingalett_log(LOG_ERROR, "Cannot open %s for writing", filename);
        free(planes); free(coded); free(model); free(head); free(filename); layout_free(&L);
        return fail(SPINGALETT_ERR_FILE_IO, "spingalett_save_dataset: cannot open file for writing");
    }
    ok = fwrite(head, 1, meta, f) == meta;      /* placeholder, rewritten at the end */
    uint8_t *index = head + SLETTD_HEADER_SIZE + pbytes;
    uint64_t pos = meta;
    for (uint32_t c = 0; ok && c < L.chunk_count; c++) {
        uint32_t rows = chunk_rows(&L, c), crc = 0;
        uint64_t start = (uint64_t)c * L.chunk_samples;
        put64(index + (size_t)c * SLETTD_INDEX_ENTRY, pos);
        for (int s = 0; ok && s < 2; s++) {
            uint32_t size = stream_size(&L, s);
            encode_values(L.enc[s], src[s] + start * size, rows, size, L.params[s], planes);
            size_t raw = (size_t)stream_raw(&L, s, rows);
            unsigned w = value_width(L.enc[s], L.target_size);
            size_t len = opt.no_compression ? 0 : encode_stream(planes, raw / w, w, L.stride[s], model, coded, raw);
            uint8_t method = len > 0 && len < raw ? STREAM_CODED : STREAM_STORED;
            const uint8_t *body = method == STREAM_CODED ? coded : planes;
            if (method == STREAM_STORED) len = raw;
            crc = crc32_update(crc, &method, 1);
            crc = crc32_update(crc, body, len);
            ok = fputc(method, f) != EOF && fwrite(body, 1, len, f) == len;
            put32(index + (size_t)c * SLETTD_INDEX_ENTRY + 8 + 4 * s, (uint32_t)(len + 1));
            pos += len + 1;
        }
        put32(index + (size_t)c * SLETTD_INDEX_ENTRY + 16, crc);
    }

    /* header and metadata, now that sizes and offsets are known */
    memcpy(head, "SLETTD", 6);
    put16(head + 6, SLETTD_VERSION);
    put32(head + 8, L.count);
    put32(head + 12, L.input_size);
    put32(head + 16, L.target_size);
    head[20] = (uint8_t)L.enc[0];
    head[21] = (uint8_t)L.enc[1];
    head[22] = opt.no_compression ? 0 : 1;
    put32(head + 24, L.chunk_samples);
    put32(head + 28, L.chunk_count);
    put32(head + 32, L.stride[0]);
    put32(head + 36, L.stride[1]);
    put64(head + 40, pos);
    put32(head + 48, (uint32_t)pbytes);
    put32(head + 60, crc32_update(0, head, 60));
    uint8_t *p = head + SLETTD_HEADER_SIZE;
    for (int s = 0; s < 2; s++)
        if (L.params[s])
            for (uint32_t i = 0; i < 2 * stream_size(&L, s); i++, p += 4) put32(p, float_bits(L.params[s][i]));
    put32(head + meta - 4, crc32_update(0, head + SLETTD_HEADER_SIZE, meta - SLETTD_HEADER_SIZE - 4));
    ok = ok && fseek(f, 0, SEEK_SET) == 0 && fwrite(head, 1, meta, f) == meta;
    ok = (fclose(f) == 0) && ok;
    if (ok)
        spingalett_log(LOG_INFO, "Data set saved to %s: %u samples, %llu bytes", filename, L.count, (unsigned long long)pos);
    else
        fail(SPINGALETT_ERR_FILE_IO, "spingalett_save_dataset: write error (disk full?)");

    free(planes); free(coded); free(model); free(head); free(filename); layout_free(&L);
    return ok;
}

/* ---------------------------------------------------------------- loading */

static bool alloc_dataset(SpingalettDataset *d, const Layout *L) {
    d->count = L->count;
    d->input_size = L->input_size;
    d->target_size = L->target_size;
    d->inputs = (float *)malloc(((size_t)L->count * L->input_size + 1) * sizeof(float));
    d->targets = (float *)malloc(((size_t)L->count * L->target_size + 1) * sizeof(float));
    if (d->inputs && d->targets) return true;
    spingalett_dataset_free(d);
    return fail(SPINGALETT_ERR_ALLOC, "out of memory loading a data set");
}

bool spingalett_load_dataset_from_memory(const void *data, size_t size, SpingalettDataset *dataset) {
    crc_init();
    unit_init();
    if (!dataset) return fail(SPINGALETT_ERR_INVALID, "spingalett_load_dataset: dataset is NULL");
    memset(dataset, 0, sizeof *dataset);
    if (!data) return fail(SPINGALETT_ERR_INVALID, "spingalett_load_dataset: data is NULL");
    const uint8_t *bytes = (const uint8_t *)data;
    Layout L;
    size_t need = 0;
    if (!parse_header(bytes, size, &L, &need)) return false;
    if (size < need || size < L.file_size) {
        layout_free(&L);
        return fail(SPINGALETT_ERR_FILE_IO, "truncated .slettd data set");
    }
    if (!L.offset && !parse_header(bytes, size, &L, &need)) return false;
    if (!alloc_dataset(dataset, &L)) { layout_free(&L); return false; }

    bool ok = true;
    int error_code = SPINGALETT_OK;               /* errors are thread-local: carry a worker's out */
    char error_message[SPINGALETT_ERRMSG_MAX] = "";
    int64_t chunks = (int64_t)L.chunk_count;
#if defined(_OPENMP)
#pragma omp parallel if(chunks > 1 && resolve_compute_mode() == COMPUTE_OPENMP)
#endif
    {
        Scratch sc = {0};
#if defined(_OPENMP)
#pragma omp for schedule(dynamic)
#endif
        for (int64_t c = 0; c < chunks; c++) {
            bool chunk_ok;
#if defined(_OPENMP)
#pragma omp atomic read
#endif
            chunk_ok = ok;
            if (!chunk_ok) continue;
            uint64_t row = (uint64_t)c * L.chunk_samples;
            if (!decode_chunk(&L, (uint32_t)c, bytes + L.offset[c], dataset->inputs + row * L.input_size,
                              dataset->targets + row * L.target_size, &sc)) {
#if defined(_OPENMP)
#pragma omp critical(spingalett_dataset_error)
#endif
                if (error_code == SPINGALETT_OK) {
                    error_code = spingalett_last_error_code();
                    snprintf(error_message, sizeof error_message, "%s", spingalett_last_error_message());
                }
#if defined(_OPENMP)
#pragma omp atomic write
#endif
                ok = false;
            }
        }
        scratch_free(&sc);
    }
    layout_free(&L);
    if (!ok) {
        set_error(error_code, error_message);
        spingalett_dataset_free(dataset);
    }
    return ok;
}

bool spingalett_load_dataset(const char *path, SpingalettDataset *dataset) {
    if (!dataset) return fail(SPINGALETT_ERR_INVALID, "spingalett_load_dataset: dataset is NULL");
    memset(dataset, 0, sizeof *dataset);
    FILE *f = path ? fopen(path, "rb") : NULL;
    if (!f) {
        spingalett_log(LOG_ERROR, "Cannot open %s", path ? path : "(null)");
        return fail(SPINGALETT_ERR_FILE_IO, "cannot open data set file");
    }
    uint8_t *buf = NULL;
    long size = -1;
    if (fseek(f, 0, SEEK_END) == 0) size = ftell(f);
    bool ok = size >= 0 && fseek(f, 0, SEEK_SET) == 0 && (buf = (uint8_t *)malloc((size_t)size + 1)) != NULL &&
              fread(buf, 1, (size_t)size, f) == (size_t)size;
    fclose(f);
    if (!ok) {
        free(buf);
        return fail(buf ? SPINGALETT_ERR_FILE_IO : SPINGALETT_ERR_ALLOC, "cannot read data set file");
    }
    ok = spingalett_load_dataset_from_memory(buf, (size_t)size, dataset);
    free(buf);
    return ok;
}

/* ---------------------------------------------------------------- streaming reader */

struct SpingalettDatasetReader {
    FILE *file;
    Layout layout;
    bool shuffle, failed, pass_done;
    uint32_t *chunk_order, next_chunk;
    uint8_t *raw;                       /* compressed bytes of the current chunk */
    size_t raw_cap;
    float *inputs, *targets;            /* decoded current chunk */
    uint32_t *rows_order, rows, row;
    Scratch scratch;
};

void spingalett_dataset_close(SpingalettDatasetReader *r) {
    if (!r) return;
    if (r->file) fclose(r->file);
    layout_free(&r->layout);
    free(r->chunk_order);
    free(r->raw);
    free(r->inputs);
    free(r->targets);
    free(r->rows_order);
    scratch_free(&r->scratch);
    free(r);
}

static void shuffle_u32(uint32_t *v, uint32_t n) {
    for (uint32_t i = n; i > 1; i--) {
        uint32_t j = (uint32_t)(rng_next64() % i), t = v[i - 1];
        v[i - 1] = v[j];
        v[j] = t;
    }
}

static void reader_start_pass(SpingalettDatasetReader *r) {
    for (uint32_t c = 0; c < r->layout.chunk_count; c++) r->chunk_order[c] = c;
    if (r->shuffle) shuffle_u32(r->chunk_order, r->layout.chunk_count);
    r->next_chunk = 0;
    r->rows = r->row = 0;
    r->pass_done = false;
}

SpingalettDatasetReader *spingalett_dataset_open(const char *path, bool shuffle) {
    crc_init();
    unit_init();
    SpingalettDatasetReader *r = (SpingalettDatasetReader *)calloc(1, sizeof *r);
    if (!r) { fail(SPINGALETT_ERR_ALLOC, "spingalett_dataset_open: out of memory"); return NULL; }
    r->shuffle = shuffle;
    r->file = path ? fopen(path, "rb") : NULL;
    if (!r->file) {
        spingalett_log(LOG_ERROR, "Cannot open %s", path ? path : "(null)");
        fail(SPINGALETT_ERR_FILE_IO, "cannot open data set file");
        spingalett_dataset_close(r);
        return NULL;
    }
    uint8_t head[SLETTD_HEADER_SIZE];
    size_t need = 0;
    bool ok = fread(head, 1, sizeof head, r->file) == sizeof head;
    if (!ok) fail(SPINGALETT_ERR_FILE_IO, "truncated .slettd data set");
    ok = ok && parse_header(head, sizeof head, &r->layout, &need);
    uint8_t *meta = ok ? (uint8_t *)malloc(need) : NULL;
    if (ok && !meta) ok = fail(SPINGALETT_ERR_ALLOC, "spingalett_dataset_open: out of memory");
    if (ok) {
        memcpy(meta, head, sizeof head);
        if (fread(meta + sizeof head, 1, need - sizeof head, r->file) != need - sizeof head)
            ok = fail(SPINGALETT_ERR_FILE_IO, "truncated .slettd data set");
    }
    ok = ok && parse_header(meta, need, &r->layout, &need);
    free(meta);
    const Layout *L = &r->layout;
    if (ok) {
        uint32_t cs = L->chunk_samples;
        r->chunk_order = (uint32_t *)malloc(((size_t)L->chunk_count + 1) * sizeof(uint32_t));
        r->inputs = (float *)malloc(((size_t)cs * L->input_size + 1) * sizeof(float));
        r->targets = (float *)malloc(((size_t)cs * L->target_size + 1) * sizeof(float));
        r->rows_order = (uint32_t *)malloc(((size_t)cs + 1) * sizeof(uint32_t));
        if (!r->chunk_order || !r->inputs || !r->targets || !r->rows_order)
            ok = fail(SPINGALETT_ERR_ALLOC, "spingalett_dataset_open: out of memory");
    }
    if (!ok) {
        spingalett_dataset_close(r);
        return NULL;
    }
    reader_start_pass(r);
    return r;
}

SpingalettDatasetInfo spingalett_dataset_info(const SpingalettDatasetReader *r) {
    SpingalettDatasetInfo info = {0};
    if (!r) return info;
    const Layout *L = &r->layout;
    info.count = L->count;
    info.input_size = L->input_size;
    info.target_size = L->target_size;
    info.input_encoding = L->enc[0];
    info.target_encoding = L->enc[1];
    info.chunk_count = L->chunk_count;
    info.file_size = L->file_size;
    return info;
}

static bool reader_load_chunk(SpingalettDatasetReader *r, uint32_t c) {
    const Layout *L = &r->layout;
    size_t size = (size_t)L->bytes[0][c] + L->bytes[1][c];
    if (size > r->raw_cap) {
        uint8_t *p = (uint8_t *)realloc(r->raw, size);
        if (!p) return fail(SPINGALETT_ERR_ALLOC, "out of memory reading a .slettd chunk");
        r->raw = p;
        r->raw_cap = size;
    }
    if (fseek(r->file, (long)L->offset[c], SEEK_SET) != 0 || fread(r->raw, 1, size, r->file) != size)
        return fail(SPINGALETT_ERR_FILE_IO, "truncated .slettd data set");
    if (!decode_chunk(L, c, r->raw, r->inputs, r->targets, &r->scratch)) return false;
    r->rows = chunk_rows(L, c);
    r->row = 0;
    for (uint32_t i = 0; i < r->rows; i++) r->rows_order[i] = i;
    if (r->shuffle) shuffle_u32(r->rows_order, r->rows);
    return true;
}

uint32_t spingalett_dataset_read(SpingalettDatasetReader *r, float *inputs, float *targets, uint32_t max_samples) {
    if (!r || !inputs || !targets) {
        set_error(SPINGALETT_ERR_INVALID, "spingalett_dataset_read: NULL argument");
        return 0;
    }
    if (r->failed) return 0;
    if (r->pass_done) reader_start_pass(r);
    const Layout *L = &r->layout;
    uint32_t done = 0;
    while (done < max_samples) {
        if (r->row == r->rows) {
            if (r->next_chunk == L->chunk_count) break;
            if (!reader_load_chunk(r, r->chunk_order[r->next_chunk++])) {
                r->failed = true;
                return 0;
            }
        }
        uint32_t i = r->rows_order[r->row++];
        memcpy(inputs + (size_t)done * L->input_size, r->inputs + (size_t)i * L->input_size, L->input_size * sizeof(float));
        memcpy(targets + (size_t)done * L->target_size, r->targets + (size_t)i * L->target_size, L->target_size * sizeof(float));
        done++;
    }
    if (done == 0) r->pass_done = true;     /* this call reported the end of the pass */
    return done;
}

uint32_t spingalett_dataset_generator(float *inputs, float *targets, uint32_t requested, void *reader) {
    return spingalett_dataset_read((SpingalettDatasetReader *)reader, inputs, targets, requested);
}
