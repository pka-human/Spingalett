/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/*
 * gzip decompression (RFC 1952 around RFC 1951's deflate), for the PTX units the CUDA backend embeds
 * compressed (cmake/PtxToC.cmake). Codes are decoded a bit at a time, canonically, as zlib's puff.c does:
 * slower than table lookups, ample for the few hundred kilobytes a unit expands to, and short.
 */

#include "Spingalett.Gunzip.h"
#include <stdlib.h>
#include <string.h>

typedef struct {
    const uint8_t *in;
    size_t in_size, in_pos;
    uint32_t bits, count;           /* bits not yet used, the lowest first */
    uint8_t *out;
    size_t out_size, out_pos;
    bool failed;
} Stream;

/* The next `need` bits (at most 16), the first of them the lowest. */
static uint32_t take(Stream *s, uint32_t need) {
    uint32_t v = s->bits;
    while (s->count < need) {
        if (s->in_pos == s->in_size) {
            s->failed = true;
            return 0;
        }
        v |= (uint32_t)s->in[s->in_pos++] << s->count;
        s->count += 8;
    }
    s->bits = v >> need;
    s->count -= need;
    return v & ((1u << need) - 1u);
}

/* A canonical code: how many codes of each length, and the symbols in the order of their codes. */
typedef struct {
    uint16_t count[16];
    uint16_t symbol[288];
} Code;

/* The code of n symbols with these lengths (0: unused); false if they overflow the lengths. Incomplete
   codes are allowed (a code of one distance), their missing codes failing when met. */
static bool build(Code *c, const uint8_t *lengths, uint32_t n) {
    uint16_t offset[16];
    memset(c->count, 0, sizeof c->count);
    for (uint32_t i = 0; i < n; i++) c->count[lengths[i]]++;
    int left = 1;
    for (uint32_t len = 1; len < 16; len++) {
        left = 2 * left - c->count[len];
        if (left < 0) return false;
    }
    offset[1] = 0;
    for (uint32_t len = 1; len < 15; len++) offset[len + 1] = (uint16_t)(offset[len] + c->count[len]);
    for (uint32_t i = 0; i < n; i++)
        if (lengths[i]) c->symbol[offset[lengths[i]]++] = (uint16_t)i;
    return true;
}

/* The next symbol of code c, or -1. */
static int decode(Stream *s, const Code *c) {
    int code = 0, first = 0, index = 0;
    for (uint32_t len = 1; len < 16; len++) {
        code |= (int)take(s, 1);
        const int n = c->count[len];
        if (code - first < n) return c->symbol[index + code - first];
        index += n;
        first = (first + n) << 1;
        code <<= 1;
    }
    return -1;
}

/* A block's literals and copies until its end. */
static bool inflate_codes(Stream *s, const Code *lit, const Code *dist) {
    static const uint16_t len_base[29] = {3,  4,  5,  6,  7,  8,  9,  10, 11,  13,  15,  17,  19,  23, 27,
                                          31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258};
    static const uint8_t len_extra[29] = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0};
    static const uint16_t dist_base[30] = {1,   2,   3,   4,   5,   7,    9,    13,   17,   25,   33,   49,   65,    97,    129,
                                           193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577};
    static const uint8_t dist_extra[30] = {0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13};
    for (;;) {
        const int symbol = decode(s, lit);
        if (symbol < 0 || s->failed) return false;
        if (symbol < 256) {
            if (s->out_pos == s->out_size) return false;
            s->out[s->out_pos++] = (uint8_t)symbol;
        } else if (symbol == 256) {
            return true;
        } else {
            const int l = symbol - 257;
            if (l >= 29) return false;
            const size_t len = len_base[l] + take(s, len_extra[l]);
            const int d = decode(s, dist);
            if (d < 0 || d >= 30) return false;
            const size_t back = dist_base[d] + take(s, dist_extra[d]);
            if (s->failed || back > s->out_pos || len > s->out_size - s->out_pos) return false;
            for (size_t i = 0; i < len; i++, s->out_pos++) s->out[s->out_pos] = s->out[s->out_pos - back];
        }
    }
}

static bool inflate_stored(Stream *s) {
    s->bits = s->count = 0;                     /* to the next byte */
    if (s->in_size - s->in_pos < 4) return false;
    const uint8_t *h = s->in + s->in_pos;
    const size_t len = h[0] | (size_t)h[1] << 8;
    if ((size_t)(h[2] | h[3] << 8) != (~len & 0xFFFFu)) return false;
    s->in_pos += 4;
    if (len > s->in_size - s->in_pos || len > s->out_size - s->out_pos) return false;
    memcpy(s->out + s->out_pos, s->in + s->in_pos, len);
    s->in_pos += len;
    s->out_pos += len;
    return true;
}

static bool inflate_fixed(Stream *s) {
    uint8_t lengths[288];
    Code lit, dist;
    memset(lengths, 8, 144);
    memset(lengths + 144, 9, 112);
    memset(lengths + 256, 7, 24);
    memset(lengths + 280, 8, 8);
    build(&lit, lengths, 288);
    memset(lengths, 5, 30);
    build(&dist, lengths, 30);
    return inflate_codes(s, &lit, &dist);
}

static bool inflate_dynamic(Stream *s) {
    static const uint8_t order[19] = {16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15};
    uint8_t lengths[320] = {0};
    Code lit, dist;
    const uint32_t nlit = take(s, 5) + 257, ndist = take(s, 5) + 1, ncode = take(s, 4) + 4;
    if (s->failed || nlit > 286 || ndist > 30) return false;
    for (uint32_t i = 0; i < ncode; i++) lengths[order[i]] = (uint8_t)take(s, 3);
    if (!build(&lit, lengths, 19)) return false;
    /* the lengths of both codes, run-length coded with the code of lengths */
    for (uint32_t i = 0; i < nlit + ndist;) {
        const int symbol = decode(s, &lit);
        if (symbol < 0 || s->failed) return false;
        if (symbol < 16) {
            lengths[i++] = (uint8_t)symbol;
            continue;
        }
        uint8_t value = 0;
        uint32_t repeat;
        if (symbol == 16) {
            if (i == 0) return false;
            value = lengths[i - 1];
            repeat = 3 + take(s, 2);
        } else if (symbol == 17) {
            repeat = 3 + take(s, 3);
        } else {
            repeat = 11 + take(s, 7);
        }
        if (i + repeat > nlit + ndist) return false;
        while (repeat--) lengths[i++] = value;
    }
    if (lengths[256] == 0) return false;        /* no end of block */
    if (!build(&lit, lengths, nlit) || !build(&dist, lengths + nlit, ndist)) return false;
    return inflate_codes(s, &lit, &dist);
}

static uint32_t crc32(const uint8_t *p, size_t n) {
    uint32_t table[256], c = 0xFFFFFFFFu;
    for (uint32_t i = 0; i < 256; i++) {
        uint32_t t = i;
        for (int k = 0; k < 8; k++) t = t >> 1 ^ (0xEDB88320u & (0u - (t & 1u)));
        table[i] = t;
    }
    for (size_t i = 0; i < n; i++) c = c >> 8 ^ table[(c ^ p[i]) & 255u];
    return ~c;
}

static uint32_t le32(const uint8_t *p) { return p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24; }

char *spg_gunzip(const uint8_t *gz, size_t size, size_t *length) {
    /* the header: magic, deflate, flags, time, extra flags, system; then what the flags announce */
    if (size < 18 || gz[0] != 0x1F || gz[1] != 0x8B || gz[2] != 8 || (gz[3] & 0xE0)) return NULL;
    const uint8_t flags = gz[3];
    size_t pos = 10;
    if (flags & 4) {                            /* FEXTRA */
        if (size - pos < 2) return NULL;
        pos += 2 + (gz[pos] | (size_t)gz[pos + 1] << 8);
    }
    for (int f = 8; f <= 16; f <<= 1)           /* FNAME, FCOMMENT: zero-terminated */
        if (flags & f) {
            while (pos < size && gz[pos]) pos++;
            pos++;
        }
    if (flags & 2) pos += 2;                    /* FHCRC */
    if (pos > size - 8) return NULL;
    const size_t expanded = le32(gz + size - 4);
    Stream s = {gz, size - 8, pos, 0, 0, malloc(expanded + 1), expanded, 0, false};
    if (!s.out) return NULL;
    bool last = false, ok = true;
    while (ok && !last) {
        last = take(&s, 1) != 0;
        const uint32_t type = take(&s, 2);
        ok = !s.failed && (type == 0 ? inflate_stored(&s) : type == 1 ? inflate_fixed(&s) : type == 2 && inflate_dynamic(&s));
    }
    if (!ok || s.out_pos != expanded || crc32(s.out, expanded) != le32(gz + size - 8)) {
        free(s.out);
        return NULL;
    }
    s.out[expanded] = 0;
    if (length) *length = expanded;
    return (char *)s.out;
}
