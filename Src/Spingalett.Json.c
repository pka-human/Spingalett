/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/*
 * JSON documents (Spingalett.Json.h): one pass over the text that checks it and lays its values out
 * in document order, containers kept open on an explicit stack (no recursion, so that the depth of a
 * hostile file costs no stack), and helpers that read the values. Numbers are converted without the C
 * library's strtod, whose decimal point follows the locale.
 */

#include "Spingalett.Json.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    const char *p, *end, *text;
    SpgJson *json;
    uint32_t cap;
    bool failed;
} Parser;

static bool fail(Parser *ps, const char *what) {
    if (!ps->failed)
        snprintf(ps->json->error, sizeof ps->json->error, "JSON: %s at byte %zu", what, (size_t)(ps->p - ps->text));
    ps->failed = true;
    return false;
}

static uint32_t new_value(Parser *ps, SpgJsonType type) {
    SpgJson *j = ps->json;
    if (j->count == ps->cap) {
        if (ps->cap >= UINT32_MAX / 2u) { fail(ps, "too many values"); return 0; }
        uint32_t cap = ps->cap ? 2u * ps->cap : 256u;
        SpgJsonValue *v = (SpgJsonValue *)realloc(j->values, (size_t)cap * sizeof(SpgJsonValue));
        if (!v) { fail(ps, "out of memory"); return 0; }
        j->values = v;
        ps->cap = cap;
    }
    SpgJsonValue *v = &j->values[j->count];
    memset(v, 0, sizeof *v);
    v->type = (uint8_t)type;
    v->start = (uint32_t)(ps->p - ps->text);
    return j->count++;
}

static void space(Parser *ps) {
    while (ps->p < ps->end && (*ps->p == ' ' || *ps->p == '\t' || *ps->p == '\n' || *ps->p == '\r')) ps->p++;
}

static int hex_digit(char c) {
    return c >= '0' && c <= '9' ? c - '0' : c >= 'a' && c <= 'f' ? c - 'a' + 10 : c >= 'A' && c <= 'F' ? c - 'A' + 10 : -1;
}

/* Four hex digits at p (the caller checked there are four bytes); -1 when they are not. */
static int32_t hex4(const char *p) {
    int32_t v = 0;
    for (int k = 0; k < 4; k++) {
        int d = hex_digit(p[k]);
        if (d < 0) return -1;
        v = v * 16 + d;
    }
    return v;
}

/* The bytes of a well-formed UTF-8 sequence at p (at most n available), 0 when it is not one: no
   overlong forms, no surrogates, nothing above U+10FFFF. */
static int utf8_length(const unsigned char *p, size_t n) {
    unsigned c = p[0];
    if (c < 0x80) return 1;
    if (c < 0xC2) return 0;
    if (c < 0xE0) return n >= 2 && (p[1] & 0xC0) == 0x80 ? 2 : 0;
    if (c < 0xF0) {
        if (n < 3 || (p[1] & 0xC0) != 0x80 || (p[2] & 0xC0) != 0x80) return 0;
        if (c == 0xE0 && p[1] < 0xA0) return 0;             /* overlong */
        if (c == 0xED && p[1] >= 0xA0) return 0;            /* surrogates */
        return 3;
    }
    if (c < 0xF5) {
        if (n < 4 || (p[1] & 0xC0) != 0x80 || (p[2] & 0xC0) != 0x80 || (p[3] & 0xC0) != 0x80) return 0;
        if (c == 0xF0 && p[1] < 0x90) return 0;             /* overlong */
        if (c == 0xF4 && p[1] >= 0x90) return 0;            /* above U+10FFFF */
        return 4;
    }
    return 0;
}

static unsigned utf8_put(char *out, uint32_t cp) {
    if (cp < 0x80) { out[0] = (char)cp; return 1; }
    if (cp < 0x800) { out[0] = (char)(0xC0 | (cp >> 6)); out[1] = (char)(0x80 | (cp & 0x3F)); return 2; }
    if (cp < 0x10000) {
        out[0] = (char)(0xE0 | (cp >> 12)); out[1] = (char)(0x80 | ((cp >> 6) & 0x3F)); out[2] = (char)(0x80 | (cp & 0x3F));
        return 3;
    }
    out[0] = (char)(0xF0 | (cp >> 18)); out[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
    out[2] = (char)(0x80 | ((cp >> 6) & 0x3F)); out[3] = (char)(0x80 | (cp & 0x3F));
    return 4;
}

/* Reads the escape at p (after the backslash): the code point it stands for and the characters it
   takes, or 0 when it is malformed. A surrogate pair is one escape of 11 characters. */
static unsigned escape_at(const char *p, const char *end, uint32_t *cp) {
    if (p >= end) return 0;
    switch (*p) {
        case '"': *cp = '"'; return 1;
        case '\\': *cp = '\\'; return 1;
        case '/': *cp = '/'; return 1;
        case 'b': *cp = '\b'; return 1;
        case 'f': *cp = '\f'; return 1;
        case 'n': *cp = '\n'; return 1;
        case 'r': *cp = '\r'; return 1;
        case 't': *cp = '\t'; return 1;
        case 'u': break;
        default: return 0;
    }
    if (end - p < 5) return 0;
    int32_t u = hex4(p + 1);
    if (u < 0 || (u >= 0xDC00 && u < 0xE000)) return 0;     /* not hex, or a lone trailing surrogate */
    if (u < 0xD800 || u >= 0xDC00) { *cp = (uint32_t)u; return 5; }
    /* a leading surrogate: its trailing one must follow */
    if (end - p < 11 || p[5] != '\\' || p[6] != 'u') return 0;
    int32_t lo = hex4(p + 7);
    if (lo < 0xDC00 || lo >= 0xE000) return 0;
    *cp = 0x10000u + (((uint32_t)u - 0xD800u) << 10) + ((uint32_t)lo - 0xDC00u);
    return 11;
}

static bool parse_string(Parser *ps) {
    uint32_t v = new_value(ps, SPG_JSON_STRING);
    if (ps->failed) return false;
    ps->p++;                                /* the quote */
    const char *s = ps->p;
    uint32_t decoded = 0;
    bool escaped = false;
    while (ps->p < ps->end && *ps->p != '"') {
        unsigned char c = (unsigned char)*ps->p;
        if (c < 0x20) return fail(ps, "a control character in a string");
        if (c == '\\') {
            uint32_t cp;
            unsigned n = escape_at(ps->p + 1, ps->end, &cp);
            if (!n) return fail(ps, "a malformed escape");
            char tmp[4];
            decoded += utf8_put(tmp, cp);
            ps->p += 1 + n;
            escaped = true;
            continue;
        }
        int n = utf8_length((const unsigned char *)ps->p, (size_t)(ps->end - ps->p));
        if (!n) return fail(ps, "malformed UTF-8");
        ps->p += n;
        decoded += (uint32_t)n;
    }
    if (ps->p >= ps->end) return fail(ps, "an unterminated string");
    SpgJsonValue *sv = &ps->json->values[v];
    sv->start = (uint32_t)(s - ps->text);
    sv->length = (uint32_t)(ps->p - s);
    sv->count = decoded;
    sv->escaped = escaped;
    sv->next = v + 1;
    ps->p++;
    return true;
}

static bool is_digit(char c) { return c >= '0' && c <= '9'; }

static bool parse_number(Parser *ps) {
    uint32_t v = new_value(ps, SPG_JSON_NUMBER);
    if (ps->failed) return false;
    const char *s = ps->p;
    if (ps->p < ps->end && *ps->p == '-') ps->p++;
    if (ps->p >= ps->end || !is_digit(*ps->p)) return fail(ps, "a malformed number");
    if (*ps->p == '0') ps->p++;
    else while (ps->p < ps->end && is_digit(*ps->p)) ps->p++;
    if (ps->p < ps->end && *ps->p == '.') {
        ps->p++;
        if (ps->p >= ps->end || !is_digit(*ps->p)) return fail(ps, "a malformed number");
        while (ps->p < ps->end && is_digit(*ps->p)) ps->p++;
    }
    if (ps->p < ps->end && (*ps->p == 'e' || *ps->p == 'E')) {
        ps->p++;
        if (ps->p < ps->end && (*ps->p == '+' || *ps->p == '-')) ps->p++;
        if (ps->p >= ps->end || !is_digit(*ps->p)) return fail(ps, "a malformed number");
        while (ps->p < ps->end && is_digit(*ps->p)) ps->p++;
    }
    ps->json->values[v].length = (uint32_t)(ps->p - s);
    ps->json->values[v].next = v + 1;
    return true;
}

static bool parse_literal(Parser *ps) {
    static const char *const words[] = {"null", "false", "true"};
    for (int k = 0; k < 3; k++) {
        size_t n = strlen(words[k]);
        if ((size_t)(ps->end - ps->p) >= n && !memcmp(ps->p, words[k], n)) {
            uint32_t v = new_value(ps, k ? SPG_JSON_BOOL : SPG_JSON_NULL);
            if (ps->failed) return false;
            ps->json->values[v].count = k == 2;
            ps->json->values[v].length = (uint32_t)n;
            ps->json->values[v].next = v + 1;
            ps->p += n;
            return true;
        }
    }
    return fail(ps, "an unexpected character");
}

bool spingalett_json_parse(SpgJson *json, const char *text, size_t size) {
    memset(json, 0, sizeof *json);
    json->text = text;
    Parser ps = {text, text + size, text, json, 0, false};
    if (!text || size >= UINT32_MAX) {
        snprintf(json->error, sizeof json->error, "JSON: %s", text ? "the document exceeds 4 GB" : "no text");
        return false;
    }
    /* value 0: no value, a null whose subtree ends where the document's begins */
    new_value(&ps, SPG_JSON_NULL);
    if (ps.failed) return false;
    json->values[0].next = SPG_JSON_ROOT;
    /* open containers: their indices, and whether the next token must be a key (objects) */
    uint32_t stack[SPG_JSON_MAX_DEPTH];
    uint32_t depth = 0;
    /* states: a value expected (after '[', ',' in an array, ':' or at the start), or a key */
    bool want_value = true;
    for (;;) {
        space(&ps);
        if (ps.p >= ps.end) { fail(&ps, depth ? "an unterminated container" : "no value"); break; }
        if (want_value) {
            char c = *ps.p;
            if (c == '{' || c == '[') {
                if (depth == SPG_JSON_MAX_DEPTH) { fail(&ps, "nesting too deep"); break; }
                uint32_t v = new_value(&ps, c == '{' ? SPG_JSON_OBJECT : SPG_JSON_ARRAY);
                if (ps.failed) break;
                stack[depth++] = v;
                ps.p++;
                space(&ps);
                if (ps.p < ps.end && *ps.p == (c == '{' ? '}' : ']')) {
                    /* empty: closed below like any container */
                    want_value = false;
                    goto close;
                }
                if (c == '{') {
                    if (ps.p >= ps.end || *ps.p != '"') { fail(&ps, "a key expected"); break; }
                    goto key;
                }
                continue;
            }
            bool ok = c == '"' ? parse_string(&ps) : (c == '-' || is_digit(c)) ? parse_number(&ps) : parse_literal(&ps);
            if (!ok) break;
            want_value = false;
        }
        /* after a value: a comma, a closing bracket, or the end of the document */
        space(&ps);
        if (!depth) {
            if (ps.p != ps.end) fail(&ps, "text after the value");
            break;
        }
        if (ps.p >= ps.end) { fail(&ps, "an unterminated container"); break; }
    close:;
        SpgJsonValue *top = &json->values[stack[depth - 1]];
        const bool object = top->type == SPG_JSON_OBJECT;
        if (*ps.p == ',') {
            ps.p++;
            top->count++;                   /* the element before it */
            if (object) {
                space(&ps);
                if (ps.p >= ps.end || *ps.p != '"') { fail(&ps, "a key expected"); break; }
                goto key;
            }
            want_value = true;
            continue;
        }
        if (*ps.p != (object ? '}' : ']')) { fail(&ps, object ? "',' or '}' expected" : "',' or ']' expected"); break; }
        /* a container ends: an element was counted unless it was empty */
        if (json->count > stack[depth - 1] + 1u) top->count++;
        ps.p++;
        top->length = (uint32_t)(ps.p - text) - top->start;
        top->next = json->count;
        depth--;
        want_value = false;
        if (!depth) {
            space(&ps);
            if (ps.p != ps.end) fail(&ps, "text after the value");
            break;
        }
        space(&ps);
        if (ps.p >= ps.end) { fail(&ps, "an unterminated container"); break; }
        goto close;
    key:
        if (!parse_string(&ps)) break;
        space(&ps);
        if (ps.p >= ps.end || *ps.p != ':') { fail(&ps, "':' expected"); break; }
        ps.p++;
        want_value = true;
    }
    if (ps.failed) {
        free(json->values);
        json->values = NULL;
        json->count = 0;
        return false;
    }
    return true;
}

void spingalett_json_free(SpgJson *json) {
    free(json->values);
    json->values = NULL;
    json->count = 0;
}

uint32_t spingalett_json_first(const SpgJson *json, uint32_t v) {
    if (v >= json->count) return 0;
    const SpgJsonValue *c = &json->values[v];
    if ((c->type != SPG_JSON_ARRAY && c->type != SPG_JSON_OBJECT) || c->count == 0) return 0;
    return v + 1;
}

uint32_t spingalett_json_element(const SpgJson *json, uint32_t v, uint32_t i) {
    if (v >= json->count || json->values[v].type != SPG_JSON_ARRAY || i >= json->values[v].count) return 0;
    uint32_t e = v + 1;
    while (i--) e = json->values[e].next;
    return e;
}

/* Decodes string v into out (values[v].count bytes, no NUL). */
static void decode(const SpgJson *json, const SpgJsonValue *s, char *out) {
    const char *p = json->text + s->start, *end = p + s->length;
    if (!s->escaped) { memcpy(out, p, s->length); return; }
    while (p < end) {
        if (*p != '\\') { *out++ = *p++; continue; }
        uint32_t cp = 0;
        unsigned n = escape_at(p + 1, end, &cp);     /* checked when parsed */
        out += utf8_put(out, cp);
        p += 1 + n;
    }
}

bool spingalett_json_is(const SpgJson *json, uint32_t v, const char *s) {
    if (v >= json->count || json->values[v].type != SPG_JSON_STRING) return false;
    const SpgJsonValue *sv = &json->values[v];
    size_t n = strlen(s);
    if (sv->count != n) return false;
    if (!sv->escaped) return n == 0 || memcmp(json->text + sv->start, s, n) == 0;
    char small[256];
    char *buf = n < sizeof small ? small : (char *)malloc(n + 1);
    if (!buf) return false;
    decode(json, sv, buf);
    bool same = memcmp(buf, s, n) == 0;
    if (buf != small) free(buf);
    return same;
}

uint32_t spingalett_json_member(const SpgJson *json, uint32_t v, const char *key) {
    if (v >= json->count || json->values[v].type != SPG_JSON_OBJECT) return 0;
    const uint32_t members = json->values[v].count;
    uint32_t k = v + 1;
    for (uint32_t m = 0; m < members; m++) {
        if (spingalett_json_is(json, k, key)) return k + 1;
        k = json->values[k + 1].next;
    }
    return 0;
}

bool spingalett_json_copy(const SpgJson *json, uint32_t v, char *out) {
    if (v >= json->count || json->values[v].type != SPG_JSON_STRING) return false;
    decode(json, &json->values[v], out);
    out[json->values[v].count] = '\0';
    return true;
}

char *spingalett_json_strdup(const SpgJson *json, uint32_t v, size_t *length) {
    if (v >= json->count || json->values[v].type != SPG_JSON_STRING) return NULL;
    char *out = (char *)malloc((size_t)json->values[v].count + 1u);
    if (!out) return NULL;
    spingalett_json_copy(json, v, out);
    if (length) *length = json->values[v].count;
    return out;
}

bool spingalett_json_number(const SpgJson *json, uint32_t v, double *out) {
    if (v >= json->count || json->values[v].type != SPG_JSON_NUMBER) return false;
    const char *p = json->text + json->values[v].start, *end = p + json->values[v].length;
    /* up to 19 significant digits in m, the rest scaling the decimal exponent; the grammar was checked */
    bool negative = *p == '-';
    if (negative) p++;
    uint64_t m = 0;
    int digits = 0;
    int64_t exponent = 0;
    for (; p < end && is_digit(*p); p++) {
        if (digits < 19) { m = 10u * m + (uint64_t)(*p - '0'); if (m) digits++; }
        else exponent++;
    }
    if (p < end && *p == '.')
        for (p++; p < end && is_digit(*p); p++)
            if (digits < 19) { m = 10u * m + (uint64_t)(*p - '0'); if (m) digits++; exponent--; }
    if (p < end && (*p == 'e' || *p == 'E')) {
        p++;
        bool neg = *p == '-';
        if (*p == '+' || *p == '-') p++;
        int64_t e = 0;
        for (; p < end && is_digit(*p); p++) if (e < 100000) e = 10 * e + (*p - '0');
        exponent += neg ? -e : e;
    }
    double r;
    static const double tens[] = {1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15,
                                  1e16, 1e17, 1e18, 1e19, 1e20, 1e21, 1e22};
    if (m == 0) {
        r = 0.0;
    } else if (m < (1ull << 53) && exponent >= -22 && exponent <= 22) {
        /* both operands exact: one rounding, the correctly rounded value */
        r = exponent >= 0 ? (double)m * tens[exponent] : (double)m / tens[-exponent];
    } else if (exponent > 400) {
        r = HUGE_VAL;
    } else if (exponent < -400) {
        r = 0.0;
    } else if (exponent < -300) {
        r = (double)m * pow(10.0, (double)(exponent + 300)) * 1e-300;
    } else {
        r = (double)m * pow(10.0, (double)exponent);
    }
    *out = negative ? -r : r;
    return true;
}

bool spingalett_json_integer(const SpgJson *json, uint32_t v, int64_t min, int64_t max, int64_t *out) {
    double d;
    if (!spingalett_json_number(json, v, &d) || !(d >= (double)min && d <= (double)max) || d != floor(d)) return false;
    /* integers beyond 2^53 come from their digits, exactly */
    const char *p = json->text + json->values[v].start, *end = p + json->values[v].length;
    bool plain = true;
    for (const char *q = p; q < end; q++) plain &= *q == '-' || is_digit(*q);
    if (plain && json->values[v].length < 19) {
        int64_t x = 0;
        bool neg = *p == '-';
        for (p += neg; p < end; p++) x = 10 * x + (*p - '0');
        x = neg ? -x : x;
        if (x < min || x > max) return false;
        *out = x;
        return true;
    }
    *out = (int64_t)d;
    return true;
}

bool spingalett_json_bool(const SpgJson *json, uint32_t v, bool *out) {
    if (v >= json->count || json->values[v].type != SPG_JSON_BOOL) return false;
    *out = json->values[v].count != 0;
    return true;
}
