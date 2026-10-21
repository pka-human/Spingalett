/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/*
 * Chat templates: the Jinja templates with which language models' tokenizers turn a conversation into text,
 * rendered as Hugging Face's transformers renders them (apply_chat_template(): jinja2's sandbox with
 * trim_blocks and lstrip_blocks and loop controls, its own tojson, raise_exception() and strftime_now()).
 *
 * The part of Jinja that templates use: text, {{ expressions }}, {# comments #}, whitespace control ({%- -%},
 * {{- -}}), {% if %} / {% elif %} / {% else %}, {% for %} (with an else, a condition, unpacked items, loop.index
 * and the rest, break and continue), {% set %} (names, a namespace's attributes, blocks), {% macro %} and calls,
 * {% generation %} (transformers' mark of what the assistant wrote: its text as it is). Expressions have Python's
 * values and semantics: none, booleans, integers, floats, strings, lists, dicts and namespaces; arithmetic,
 * comparisons, in, and, or, not, the conditional, ~, attributes and items (methods first for attributes, items
 * first for subscripts), slices with steps, the filters and tests below, the methods of strings and dicts, and the
 * globals range(), namespace(), dict(), raise_exception() and strftime_now(). What a template asks for beyond
 * that is an error naming it, as is what Python would refuse (a string added to a list); undefined values print
 * as nothing and are false, as jinja2's default undefined is.
 *
 * A render parses the template into a tree in an arena, evaluates it with variables of its own (the messages,
 * tools, add_generation_prompt, the tokenizer's special tokens and those a JSON object adds), and frees the arena:
 * nothing is kept between renders, and threads may render at once.
 */

#include "Spingalett.Chat.h"
#include "Spingalett.Json.h"
#include "Spingalett.Private.h"
#include <float.h>
#include <math.h>
#include <setjmp.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ------------------------------------------------------------------------- the arena and errors */

typedef struct Block {
    struct Block *next;
    size_t used, size;
    _Alignas(16) unsigned char data[];
} Block;

typedef struct Render Render;
static _Noreturn void fail(Render *r, const char *format, ...);
typedef struct Node Node;
static void exec(Render *r, const Node *n);

typedef enum { UNDEF, NONE, BOOL, INT, FLOAT, STR, LIST, DICT, FUNC, MACRO } Kind;

typedef struct Value Value;
typedef struct List List;
typedef struct Dict Dict;

struct Value {
    uint8_t kind;
    union {
        bool b;
        int64_t i;
        double f;
        struct {
            const char *p;
            size_t n;
        } s;
        List *l;
        Dict *d;
        struct {
            int id;                 /* a global or a method */
            const Value *self;      /* a method's object */
        } fn;
        const Node *macro;
    };
};

struct List {
    Value *items;
    uint32_t count, cap;
};

struct Dict {
    Value *keys, *values;
    uint32_t count, cap;
    bool ns;                        /* a namespace(): its attributes may be set */
};

/* ------------------------------------------------------------------------- the tree */

typedef enum {
    /* statements */
    N_BODY, N_TEXT, N_PRINT, N_IF, N_FOR, N_SET, N_SET_BLOCK, N_MACRO, N_BREAK, N_CONTINUE,
    /* expressions */
    E_LITERAL, E_NAME, E_LIST, E_DICT, E_ATTR, E_ITEM, E_SLICE, E_CALL, E_FILTER, E_TEST, E_NEG, E_POS, E_NOT,
    E_BINARY, E_COMPARE, E_AND, E_OR, E_COND, E_CONCAT
} NodeType;

typedef struct {
    const char *p;
    size_t n;
} Str;

struct Node {
    uint8_t type;
    int op;                         /* E_BINARY, E_COMPARE: the operator; E_TEST: negated */
    Str name;                       /* names, attributes, filters, tests, a set's target */
    Value value;                    /* E_LITERAL */
    Node *a, *b, *c, *d;            /* operands; N_IF: condition, body, else; N_FOR: iterable, body, else, condition */
    Node **list;                    /* bodies, items, arguments, a compare chain's operands */
    uint32_t count;
    Str *names;                     /* keyword arguments' names, a for's targets, a macro's parameters */
    uint32_t name_count;
};

/* ------------------------------------------------------------------------- the render */

typedef struct Frame {
    struct Frame *parent;
    Dict vars;
} Frame;

struct Render {
    Block *blocks;
    jmp_buf on_error;
    char error[256];
    const char *source;
    /* the tokens */
    struct Token *tokens;
    uint32_t token_count, at;
    /* the output */
    char *out;
    size_t out_n, out_cap;
    Frame *root, *frame;
    int jump;                       /* 1: break, 2: continue */
    int depth;
    time_t now;
};

static void *alloc(Render *r, size_t size) {
    size = (size + 15u) & ~(size_t)15u;
    if (!r->blocks || r->blocks->size - r->blocks->used < size) {
        size_t want = size > 65536u ? size : 65536u;
        Block *b = (Block *)malloc(sizeof(Block) + want);
        if (!b) fail(r, "out of memory");
        b->next = r->blocks;
        b->used = 0;
        b->size = want;
        r->blocks = b;
    }
    void *p = r->blocks->data + r->blocks->used;
    r->blocks->used += size;
    return p;
}

static _Noreturn void fail(Render *r, const char *format, ...) {
    va_list args;
    va_start(args, format);
    vsnprintf(r->error, sizeof r->error, format, args);
    va_end(args);
    longjmp(r->on_error, 1);
}

static Str str_of(const char *p, size_t n) {
    Str s = {p, n};
    return s;
}

static bool str_is(Str s, const char *word) {
    size_t n = strlen(word);
    return s.n == n && !memcmp(s.p, word, n);
}

/* ------------------------------------------------------------------------- values */

static Value undef(void) {
    Value v;
    memset(&v, 0, sizeof v);
    v.kind = UNDEF;
    return v;
}

static Value none(void) {
    Value v = undef();
    v.kind = NONE;
    return v;
}

static Value boolean(bool b) {
    Value v = undef();
    v.kind = BOOL;
    v.b = b;
    return v;
}

static Value integer(int64_t i) {
    Value v = undef();
    v.kind = INT;
    v.i = i;
    return v;
}

static Value real(double f) {
    Value v = undef();
    v.kind = FLOAT;
    v.f = f;
    return v;
}

static Value string(const char *p, size_t n) {
    Value v = undef();
    v.kind = STR;
    v.s.p = p;
    v.s.n = n;
    return v;
}

static Value cstring(const char *p) {
    return string(p, strlen(p));
}

static Value copy_string(Render *r, const char *p, size_t n) {
    char *c = (char *)alloc(r, n + 1);
    memcpy(c, p, n);
    c[n] = '\0';
    return string(c, n);
}

static Value new_list(Render *r, uint32_t cap) {
    Value v = undef();
    v.kind = LIST;
    v.l = (List *)alloc(r, sizeof(List));
    v.l->count = 0;
    v.l->cap = cap ? cap : 4;
    v.l->items = (Value *)alloc(r, v.l->cap * sizeof(Value));
    return v;
}

static void list_push(Render *r, List *l, Value item) {
    if (l->count == l->cap) {
        Value *grown = (Value *)alloc(r, 2u * l->cap * sizeof(Value));
        memcpy(grown, l->items, l->count * sizeof(Value));
        l->items = grown;
        l->cap *= 2u;
    }
    l->items[l->count++] = item;
}

static Value new_dict(Render *r) {
    Value v = undef();
    v.kind = DICT;
    v.d = (Dict *)alloc(r, sizeof(Dict));
    memset(v.d, 0, sizeof(Dict));
    return v;
}

static bool equal(const Value *a, const Value *b);

static Value *dict_find(Dict *d, const char *p, size_t n) {
    for (uint32_t k = 0; k < d->count; k++)
        if (d->keys[k].kind == STR && d->keys[k].s.n == n && !memcmp(d->keys[k].s.p, p, n)) return &d->values[k];
    return NULL;
}

static Value *dict_find_value(Dict *d, const Value *key) {
    if (key->kind == STR) return dict_find(d, key->s.p, key->s.n);
    for (uint32_t k = 0; k < d->count; k++)
        if (equal(&d->keys[k], key)) return &d->values[k];
    return NULL;
}

static void dict_set(Render *r, Dict *d, Value key, Value value) {
    Value *at = dict_find_value(d, &key);
    if (at) {
        *at = value;
        return;
    }
    if (d->count == d->cap) {
        uint32_t cap = d->cap ? 2u * d->cap : 8u;
        Value *keys = (Value *)alloc(r, cap * sizeof(Value)), *values = (Value *)alloc(r, cap * sizeof(Value));
        if (d->count) {
            memcpy(keys, d->keys, d->count * sizeof(Value));
            memcpy(values, d->values, d->count * sizeof(Value));
        }
        d->keys = keys;
        d->values = values;
        d->cap = cap;
    }
    d->keys[d->count] = key;
    d->values[d->count++] = value;
}

static bool truthy(const Value *v) {
    switch (v->kind) {
        case BOOL: return v->b;
        case INT: return v->i != 0;
        case FLOAT: return v->f != 0.0;
        case STR: return v->s.n != 0;
        case LIST: return v->l->count != 0;
        case DICT: return v->d->count != 0;
        case FUNC: case MACRO: return true;
        default: return false;
    }
}

static const char *kind_name(const Value *v) {
    static const char *names[] = {"undefined", "NoneType", "bool", "int", "float", "str", "list", "dict", "function",
                                  "macro"};
    if (v->kind == DICT && v->d->ns) return "Namespace";
    return names[v->kind];
}

static bool numeric(const Value *v) {
    return v->kind == INT || v->kind == FLOAT || v->kind == BOOL;
}

static double as_double(const Value *v) {
    return v->kind == FLOAT ? v->f : v->kind == BOOL ? (double)v->b : (double)v->i;
}

static int64_t as_int(const Value *v) {
    return v->kind == BOOL ? (int64_t)v->b : v->i;
}

static bool equal(const Value *a, const Value *b) {
    if (numeric(a) && numeric(b)) {
        if (a->kind != FLOAT && b->kind != FLOAT) return as_int(a) == as_int(b);
        return as_double(a) == as_double(b);
    }
    if (a->kind != b->kind) return false;
    switch (a->kind) {
        case UNDEF: case NONE: return true;
        case STR: return a->s.n == b->s.n && !memcmp(a->s.p, b->s.p, a->s.n);
        case LIST:
            if (a->l->count != b->l->count) return false;
            for (uint32_t k = 0; k < a->l->count; k++)
                if (!equal(&a->l->items[k], &b->l->items[k])) return false;
            return true;
        case DICT:
            if (a->d->count != b->d->count) return false;
            for (uint32_t k = 0; k < a->d->count; k++) {
                Value *other = dict_find_value(b->d, &a->d->keys[k]);
                if (!other || !equal(&a->d->values[k], other)) return false;
            }
            return true;
        case FUNC: return a->fn.id == b->fn.id && a->fn.self == b->fn.self;
        case MACRO: return a->macro == b->macro;
        default: return false;
    }
}

/* Python's ordering: -1, 0 or 1 (numbers, strings, lists); an error for other types. */
static int order(Render *r, const Value *a, const Value *b) {
    if (numeric(a) && numeric(b)) {
        if (a->kind != FLOAT && b->kind != FLOAT) return as_int(a) < as_int(b) ? -1 : as_int(a) > as_int(b);
        double x = as_double(a), y = as_double(b);
        return x < y ? -1 : x > y;
    }
    if (a->kind == STR && b->kind == STR) {
        size_t n = a->s.n < b->s.n ? a->s.n : b->s.n;
        int c = memcmp(a->s.p, b->s.p, n);
        if (c) return c < 0 ? -1 : 1;
        return a->s.n < b->s.n ? -1 : a->s.n > b->s.n;
    }
    if (a->kind == LIST && b->kind == LIST) {
        for (uint32_t k = 0; k < a->l->count && k < b->l->count; k++) {
            if (equal(&a->l->items[k], &b->l->items[k])) continue;
            return order(r, &a->l->items[k], &b->l->items[k]);
        }
        return a->l->count < b->l->count ? -1 : a->l->count > b->l->count;
    }
    fail(r, "'<' not supported between instances of '%s' and '%s'", kind_name(a), kind_name(b));
}

/* ------------------------------------------------------------------------- text */

typedef struct {
    Render *r;
    char *p;
    size_t n, cap;
} Text;

static void text_add(Text *t, const char *p, size_t n) {
    if (t->n + n + 1 > t->cap) {
        size_t cap = t->cap ? t->cap : 64u;
        while (cap < t->n + n + 1) cap *= 2u;
        char *grown = (char *)alloc(t->r, cap);
        if (t->n) memcpy(grown, t->p, t->n);
        t->p = grown;
        t->cap = cap;
    }
    if (n) memcpy(t->p + t->n, p, n);
    t->n += n;
    t->p[t->n] = '\0';
}

static void text_cstr(Text *t, const char *s) {
    text_add(t, s, strlen(s));
}

static Value text_value(Text *t) {
    return t->n ? string(t->p, t->n) : string("", 0);
}

/* A float as Python's repr() writes it: the shortest digits that read back as it, positional from 1e-4 to 1e16. */
static void float_repr(double f, char *out, size_t size) {
    if (isnan(f)) { snprintf(out, size, "nan"); return; }
    if (isinf(f)) { snprintf(out, size, f < 0 ? "-inf" : "inf"); return; }
    if (f == 0.0) { snprintf(out, size, signbit(f) ? "-0.0" : "0.0"); return; }
    char e[40];
    int digits = 1;
    for (; digits <= 17; digits++) {
        snprintf(e, sizeof e, "%.*e", digits - 1, f);
        if (strtod(e, NULL) == f) break;
    }
    /* e: [-]d[.ddd]e[+-]xx */
    const char *p = e;
    char mantissa[24];
    size_t m = 0;
    bool negative = *p == '-';
    if (negative) p++;
    for (; *p && *p != 'e'; p++)
        if (*p != '.') mantissa[m++] = *p;
    mantissa[m] = '\0';
    int exponent = atoi(p + 1);
    while (m > 1 && mantissa[m - 1] == '0') mantissa[--m] = '\0';
    char *o = out;
    size_t left = size;
#define PUT(...) do { int w_ = snprintf(o, left, __VA_ARGS__); if (w_ < 0 || (size_t)w_ >= left) return; o += w_; left -= (size_t)w_; } while (0)
    if (negative) PUT("-");
    if (exponent >= -4 && exponent < 16) {
        if (exponent < 0) {
            PUT("0.");
            for (int k = -1; k > exponent; k--) PUT("0");
            PUT("%s", mantissa);
        } else if ((int)m <= exponent + 1) {
            PUT("%s", mantissa);
            for (int k = (int)m; k <= exponent; k++) PUT("0");
            PUT(".0");
        } else {
            PUT("%.*s.%s", exponent + 1, mantissa, mantissa + exponent + 1);
        }
    } else {
        PUT("%c", mantissa[0]);
        if (m > 1) PUT(".%s", mantissa + 1);
        PUT("e%c%02d", exponent < 0 ? '-' : '+', exponent < 0 ? -exponent : exponent);
    }
#undef PUT
}

/* Whether the bytes at p start a printable character as Python's repr() keeps it (ASCII 32..126, or a character
   beyond ASCII that is not a control or a separator: those of U+0080..U+00A0 and the line separators escape). */
static size_t utf8_length(const unsigned char *p, size_t n) {
    if (p[0] < 0x80) return 1;
    size_t len = p[0] >= 0xF0 ? 4 : p[0] >= 0xE0 ? 3 : 2;
    return len <= n ? len : n;
}

static void repr_into(Render *r, Text *t, const Value *v);

static void repr_string(Text *t, const char *p, size_t n) {
    bool single = false, dbl = false;
    for (size_t k = 0; k < n; k++) single |= p[k] == '\'', dbl |= p[k] == '"';
    const char quote = single && !dbl ? '"' : '\'';
    text_add(t, &quote, 1);
    for (size_t k = 0; k < n;) {
        const unsigned char c = (unsigned char)p[k];
        char esc[12];
        if (c == '\\') text_cstr(t, "\\\\");
        else if (c == (unsigned char)quote) { esc[0] = '\\'; esc[1] = (char)c; text_add(t, esc, 2); }
        else if (c == '\n') text_cstr(t, "\\n");
        else if (c == '\r') text_cstr(t, "\\r");
        else if (c == '\t') text_cstr(t, "\\t");
        else if (c < 0x20 || c == 0x7F) { snprintf(esc, sizeof esc, "\\x%02x", c); text_cstr(t, esc); }
        else if (c < 0x80) text_add(t, (const char *)&p[k], 1);
        else {
            size_t len = utf8_length((const unsigned char *)p + k, n - k);
            uint32_t cp = len == 2 ? ((c & 0x1Fu) << 6) | (p[k + 1] & 0x3Fu) : 0;
            if (len == 2 && cp <= 0xA0) { snprintf(esc, sizeof esc, "\\x%02x", cp); text_cstr(t, esc); }
            else text_add(t, p + k, len);
            k += len;
            continue;
        }
        k++;
    }
    text_add(t, &quote, 1);
}

static void repr_into(Render *r, Text *t, const Value *v) {
    char number[48];
    switch (v->kind) {
        case UNDEF: return;
        case NONE: text_cstr(t, "None"); return;
        case BOOL: text_cstr(t, v->b ? "True" : "False"); return;
        case INT: snprintf(number, sizeof number, "%lld", (long long)v->i); text_cstr(t, number); return;
        case FLOAT: float_repr(v->f, number, sizeof number); text_cstr(t, number); return;
        case STR: repr_string(t, v->s.p, v->s.n); return;
        case LIST:
            text_cstr(t, "[");
            for (uint32_t k = 0; k < v->l->count; k++) {
                if (k) text_cstr(t, ", ");
                repr_into(r, t, &v->l->items[k]);
            }
            text_cstr(t, "]");
            return;
        case DICT:
            if (v->d->ns) {
                text_cstr(t, "<Namespace ");
                for (uint32_t k = 0; k < v->d->count; k++) {
                    if (k) text_cstr(t, ", ");
                    text_add(t, v->d->keys[k].s.p, v->d->keys[k].s.n);
                    text_cstr(t, "=");
                    repr_into(r, t, &v->d->values[k]);
                }
                text_cstr(t, ">");
                return;
            }
            text_cstr(t, "{");
            for (uint32_t k = 0; k < v->d->count; k++) {
                if (k) text_cstr(t, ", ");
                repr_into(r, t, &v->d->keys[k]);
                text_cstr(t, ": ");
                repr_into(r, t, &v->d->values[k]);
            }
            text_cstr(t, "}");
            return;
        default: text_cstr(t, "<function>"); return;
    }
}

/* str() of a value into t: strings as they are, the rest as repr() writes them, undefined as nothing. */
static void str_into(Render *r, Text *t, const Value *v) {
    if (v->kind == STR) text_add(t, v->s.p, v->s.n);
    else repr_into(r, t, v);
}

static Value to_str(Render *r, const Value *v) {
    if (v->kind == STR) return *v;
    Text t = {r, NULL, 0, 0};
    str_into(r, &t, v);
    return text_value(&t);
}

/* json.dumps(v, ensure_ascii=False, indent, separators, sort_keys) as transformers' tojson calls it. */
static void json_into(Render *r, Text *t, const Value *v, int64_t indent, int level, const char *item_sep,
                      const char *key_sep, bool sort_keys) {
    char number[48];
    switch (v->kind) {
        case NONE: text_cstr(t, "null"); return;
        case BOOL: text_cstr(t, v->b ? "true" : "false"); return;
        case INT: snprintf(number, sizeof number, "%lld", (long long)v->i); text_cstr(t, number); return;
        case FLOAT:
            if (isnan(v->f)) text_cstr(t, "NaN");
            else if (isinf(v->f)) text_cstr(t, v->f < 0 ? "-Infinity" : "Infinity");
            else { float_repr(v->f, number, sizeof number); text_cstr(t, number); }
            return;
        case STR:
            text_cstr(t, "\"");
            for (size_t k = 0; k < v->s.n; k++) {
                const unsigned char c = (unsigned char)v->s.p[k];
                if (c == '"') text_cstr(t, "\\\"");
                else if (c == '\\') text_cstr(t, "\\\\");
                else if (c == '\n') text_cstr(t, "\\n");
                else if (c == '\r') text_cstr(t, "\\r");
                else if (c == '\t') text_cstr(t, "\\t");
                else if (c == '\b') text_cstr(t, "\\b");
                else if (c == '\f') text_cstr(t, "\\f");
                else if (c < 0x20) { snprintf(number, sizeof number, "\\u%04x", c); text_cstr(t, number); }
                else text_add(t, (const char *)&c, 1);
            }
            text_cstr(t, "\"");
            return;
        case LIST: case DICT: {
            const bool list = v->kind == LIST;
            const uint32_t n = list ? v->l->count : v->d->count;
            text_cstr(t, list ? "[" : "{");
            if (!n) { text_cstr(t, list ? "]" : "}"); return; }
            uint32_t *at = (uint32_t *)alloc(r, n * sizeof(uint32_t));
            for (uint32_t k = 0; k < n; k++) at[k] = k;
            if (!list && sort_keys)
                for (uint32_t k = 1; k < n; k++)
                    for (uint32_t j = k; j > 0 && order(r, &v->d->keys[at[j - 1]], &v->d->keys[at[j]]) > 0; j--) {
                        uint32_t s = at[j];
                        at[j] = at[j - 1];
                        at[j - 1] = s;
                    }
            for (uint32_t k = 0; k < n; k++) {
                if (k) text_cstr(t, item_sep);
                if (indent >= 0) {
                    text_cstr(t, "\n");
                    for (int64_t s = 0; s < indent * (level + 1); s++) text_cstr(t, " ");
                }
                if (!list) {
                    const Value *key = &v->d->keys[at[k]];
                    if (key->kind == STR) json_into(r, t, key, indent, level + 1, item_sep, key_sep, sort_keys);
                    else {
                        Value s = to_str(r, key);
                        if (key->kind == BOOL) s = cstring(key->b ? "true" : "false");
                        else if (key->kind == NONE) s = cstring("null");
                        json_into(r, t, &s, indent, level + 1, item_sep, key_sep, sort_keys);
                    }
                    text_cstr(t, key_sep);
                }
                json_into(r, t, list ? &v->l->items[at[k]] : &v->d->values[at[k]], indent, level + 1, item_sep,
                          key_sep, sort_keys);
            }
            if (indent >= 0) {
                text_cstr(t, "\n");
                for (int64_t s = 0; s < indent * level; s++) text_cstr(t, " ");
            }
            text_cstr(t, list ? "]" : "}");
            return;
        }
        default:
            fail(r, "Object of type %s is not JSON serializable", v->kind == UNDEF ? "Undefined" : kind_name(v));
    }
}

/* ------------------------------------------------------------------------- the lexer */

typedef enum { T_TEXT, T_PRINT_OPEN, T_PRINT_CLOSE, T_BLOCK_OPEN, T_BLOCK_CLOSE, T_NAME, T_STRING, T_INT, T_FLOAT,
               T_OP, T_EOF } TokenType;

typedef struct Token {
    uint8_t type;
    Str text;                       /* names, operators, raw text and numbers as written; strings decoded */
    uint32_t line;
} Token;

static void push_token(Render *r, uint32_t *cap, Token token) {
    if (r->token_count == *cap) {
        uint32_t grown_cap = *cap ? 2u * *cap : 256u;
        Token *grown = (Token *)alloc(r, grown_cap * sizeof(Token));
        if (r->token_count) memcpy(grown, r->tokens, r->token_count * sizeof(Token));
        r->tokens = grown;
        *cap = grown_cap;
    }
    r->tokens[r->token_count++] = token;
}

static bool is_space(char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v';
}

static bool is_name_start(char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_';
}

static bool is_digit(char c) {
    return c >= '0' && c <= '9';
}

/* A string literal's escapes decoded (Python's: \n \t \r \\ \' \" \xHH \uXXXX, others kept). */
static Str unescape(Render *r, const char *p, size_t n) {
    char *out = (char *)alloc(r, n + 1);
    size_t m = 0;
    for (size_t k = 0; k < n; k++) {
        if (p[k] != '\\' || k + 1 == n) { out[m++] = p[k]; continue; }
        char c = p[++k];
        uint32_t cp = 0;
        int hex = c == 'x' ? 2 : c == 'u' ? 4 : c == 'U' ? 8 : 0;
        if (hex && k + (size_t)hex < n) {
            for (int h = 0; h < hex; h++) {
                char d = p[k + 1 + (size_t)h];
                cp = cp * 16u + (uint32_t)(is_digit(d) ? d - '0' : (d | 32) - 'a' + 10);
            }
            k += (size_t)hex;
            if (cp < 0x80) out[m++] = (char)cp;
            else if (cp < 0x800) { out[m++] = (char)(0xC0 | cp >> 6); out[m++] = (char)(0x80 | (cp & 0x3F)); }
            else if (cp < 0x10000) {
                out[m++] = (char)(0xE0 | cp >> 12);
                out[m++] = (char)(0x80 | ((cp >> 6) & 0x3F));
                out[m++] = (char)(0x80 | (cp & 0x3F));
            } else {
                out[m++] = (char)(0xF0 | cp >> 18);
                out[m++] = (char)(0x80 | ((cp >> 12) & 0x3F));
                out[m++] = (char)(0x80 | ((cp >> 6) & 0x3F));
                out[m++] = (char)(0x80 | (cp & 0x3F));
            }
            continue;
        }
        switch (c) {
            case 'n': out[m++] = '\n'; break;
            case 't': out[m++] = '\t'; break;
            case 'r': out[m++] = '\r'; break;
            case '\\': out[m++] = '\\'; break;
            case '\'': out[m++] = '\''; break;
            case '"': out[m++] = '"'; break;
            case '\n': break;
            default: out[m++] = '\\'; out[m++] = c; break;
        }
    }
    out[m] = '\0';
    return str_of(out, m);
}

/* The template into tokens, its whitespace controlled as jinja2's lexer with trim_blocks and lstrip_blocks
   controls it: "{%-" and "{{-" strip the whitespace before them, "-%}" and "-}}" the whitespace after them; a block
   or a comment takes the newline after it, and the spaces and tabs before it on its line ("+" keeps them). */
static void lex(Render *r, const char *src, size_t n) {
    uint32_t cap = 0, line = 1;
    size_t k = 0;
    bool strip_next = false, trim_newline = false;
    while (k < n) {
        /* text up to the next tag, after what the tag before it took (the whitespace "-" strips, the newline of
           trim_blocks); line_starting: whether that left the text at a line's start, as jinja2's lexer has it */
        size_t start = k;
        while (k < n && !(src[k] == '{' && k + 1 < n && (src[k + 1] == '{' || src[k + 1] == '%' || src[k + 1] == '#')))
            k++;
        const char *tp = src + start;
        size_t tn = k - start;
        bool line_starting = start == 0;
        if (strip_next) {
            size_t s = 0;
            while (s < tn && is_space(tp[s])) s++;
            if (s) line_starting = tp[s - 1] == '\n';
            tp += s;
            tn -= s;
        } else if (trim_newline && tn && *tp == '\n') {
            tp++;
            tn--;
            line_starting = true;
        }
        strip_next = trim_newline = false;
        if (k < n) {
            const char kind = src[k + 1];
            const bool minus = k + 2 < n && src[k + 2] == '-', plus = k + 2 < n && src[k + 2] == '+';
            if (minus) {
                while (tn && is_space(tp[tn - 1])) tn--;
            } else if (kind != '{' && !plus) {
                /* lstrip_blocks: the whitespace between the line's start and a block or comment */
                size_t l = tn;
                while (l && tp[l - 1] != '\n') l--;
                if (l > 0 || line_starting) {
                    bool blank = true;
                    for (size_t c = l; c < tn && blank; c++) blank = is_space(tp[c]);
                    if (blank) tn = l;
                }
            }
        }
        if (tn) {
            Token t = {T_TEXT, str_of(tp, tn), line};
            push_token(r, &cap, t);
        }
        for (size_t c = start; c < k; c++) line += src[c] == '\n';
        if (k >= n) break;
        const char kind = src[k + 1];
        k += 2;
        if (k < n && (src[k] == '-' || src[k] == '+')) k++;
        if (kind == '#') {
            /* a comment */
            size_t end = k;
            while (end + 1 < n && !(src[end] == '#' && src[end + 1] == '}')) end++;
            if (end + 1 >= n) fail(r, "line %u: a comment is not closed", line);
            for (size_t c = k; c < end; c++) line += src[c] == '\n';
            strip_next = end > k && src[end - 1] == '-';
            trim_newline = !strip_next && !(end > k && src[end - 1] == '+');
            k = end + 2;
            continue;
        }
        Token open = {kind == '{' ? T_PRINT_OPEN : T_BLOCK_OPEN, str_of(src + k - 2, 2), line};
        push_token(r, &cap, open);
        /* the tag's tokens up to its end */
        int depth = 0;
        for (;;) {
            while (k < n && is_space(src[k])) line += src[k] == '\n', k++;
            if (k >= n) fail(r, "line %u: a tag is not closed", line);
            const char close = kind == '{' ? '}' : '%';
            if (depth == 0) {
                size_t at = k;
                bool tag_minus = false, tag_plus = false;
                if ((src[at] == '-' || src[at] == '+') && at + 2 < n && src[at + 1] == close && src[at + 2] == '}')
                    tag_minus = src[at] == '-', tag_plus = src[at] == '+', at++;
                if (at + 1 < n && src[at] == close && src[at + 1] == '}') {
                    Token t = {kind == '{' ? T_PRINT_CLOSE : T_BLOCK_CLOSE, str_of(src + at, 2), line};
                    push_token(r, &cap, t);
                    k = at + 2;
                    strip_next = tag_minus;
                    trim_newline = !tag_minus && !tag_plus && kind == '%';
                    break;
                }
            }
            const char c = src[k];
            if (is_name_start(c)) {
                size_t s = k;
                while (k < n && (is_name_start(src[k]) || is_digit(src[k]))) k++;
                Token t = {T_NAME, str_of(src + s, k - s), line};
                push_token(r, &cap, t);
            } else if (is_digit(c)) {
                size_t s = k;
                bool fraction = false;
                while (k < n && (is_digit(src[k]) || src[k] == '_')) k++;
                if (k + 1 < n && src[k] == '.' && is_digit(src[k + 1])) {
                    fraction = true;
                    k++;
                    while (k < n && (is_digit(src[k]) || src[k] == '_')) k++;
                }
                if (k < n && (src[k] == 'e' || src[k] == 'E')) {
                    size_t e = k + 1;
                    if (e < n && (src[e] == '+' || src[e] == '-')) e++;
                    if (e < n && is_digit(src[e])) {
                        fraction = true;
                        k = e;
                        while (k < n && is_digit(src[k])) k++;
                    }
                }
                Token t = {fraction ? T_FLOAT : T_INT, str_of(src + s, k - s), line};
                push_token(r, &cap, t);
            } else if (c == '\'' || c == '"') {
                size_t s = ++k;
                while (k < n && src[k] != c) {
                    if (src[k] == '\\' && k + 1 < n) k++;
                    line += src[k] == '\n';
                    k++;
                }
                if (k >= n) fail(r, "line %u: a string is not closed", line);
                Token t = {T_STRING, unescape(r, src + s, k - s), line};
                push_token(r, &cap, t);
                k++;
            } else {
                static const char *const two[] = {"//", "**", "==", "!=", "<=", ">="};
                size_t len = 1;
                for (size_t o = 0; o < sizeof two / sizeof *two; o++)
                    if (k + 1 < n && src[k] == two[o][0] && src[k + 1] == two[o][1]) len = 2;
                if (len == 1 && !strchr("+-*/%~<>=.,:|()[]{}!", c))
                    fail(r, "line %u: unexpected character '%c'", line, c);
                if (c == '(' || c == '[' || c == '{') depth++;
                if ((c == ')' || c == ']' || c == '}') && depth > 0) depth--;
                Token t = {T_OP, str_of(src + k, len), line};
                push_token(r, &cap, t);
                k += len;
            }
        }
    }
    Token end = {T_EOF, str_of("", 0), line};
    push_token(r, &cap, end);
}

/* ------------------------------------------------------------------------- the parser */

static Token *peek(Render *r) {
    return &r->tokens[r->at];
}

static Token *next_token(Render *r) {
    Token *t = &r->tokens[r->at];
    if (t->type != T_EOF) r->at++;
    return t;
}

static bool at_op(Render *r, const char *op) {
    Token *t = peek(r);
    return t->type == T_OP && str_is(t->text, op);
}

static bool at_name(Render *r, const char *name) {
    Token *t = peek(r);
    return t->type == T_NAME && str_is(t->text, name);
}

static void expect_op(Render *r, const char *op) {
    Token *t = next_token(r);
    if (t->type != T_OP || !str_is(t->text, op))
        fail(r, "line %u: expected '%s', found '%.*s'", t->line, op, (int)t->text.n, t->text.p);
}

static Str expect_name(Render *r) {
    Token *t = next_token(r);
    if (t->type != T_NAME) fail(r, "line %u: expected a name, found '%.*s'", t->line, (int)t->text.n, t->text.p);
    return t->text;
}

static Node *node(Render *r, NodeType type) {
    Node *n = (Node *)alloc(r, sizeof(Node));
    memset(n, 0, sizeof(Node));
    n->type = (uint8_t)type;
    return n;
}

typedef struct {
    Node **items;
    uint32_t count, cap;
} Nodes;

static void nodes_push(Render *r, Nodes *list, Node *n) {
    if (list->count == list->cap) {
        uint32_t cap = list->cap ? 2u * list->cap : 8u;
        Node **grown = (Node **)alloc(r, cap * sizeof(Node *));
        if (list->count) memcpy(grown, list->items, list->count * sizeof(Node *));
        list->items = grown;
        list->cap = cap;
    }
    list->items[list->count++] = n;
}

static Node *expression(Render *r);
static Node *or_expr(Render *r);
static Node *unary(Render *r, bool filters);

/* Call arguments up to ")": positional ones into n->list, keyword ones (name=value) into n->names and after them. */
static void arguments(Render *r, Node *n) {
    Nodes args = {0};
    Nodes kw = {0};
    Str names[32];
    uint32_t named = 0;
    while (!at_op(r, ")")) {
        Token *t = peek(r);
        if (t->type == T_NAME && r->tokens[r->at + 1].type == T_OP && str_is(r->tokens[r->at + 1].text, "=")) {
            if (named == 32) fail(r, "line %u: too many keyword arguments", t->line);
            names[named++] = t->text;
            r->at += 2;
            nodes_push(r, &kw, expression(r));
        } else {
            nodes_push(r, &args, expression(r));
        }
        if (!at_op(r, ",")) break;
        next_token(r);
    }
    expect_op(r, ")");
    n->count = args.count;
    n->list = (Node **)alloc(r, (args.count + kw.count + 1u) * sizeof(Node *));
    for (uint32_t k = 0; k < args.count; k++) n->list[k] = args.items[k];
    for (uint32_t k = 0; k < kw.count; k++) n->list[args.count + k] = kw.items[k];
    n->name_count = named;
    n->names = (Str *)alloc(r, (named + 1u) * sizeof(Str));
    memcpy(n->names, names, named * sizeof(Str));
}

static Node *primary(Render *r) {
    Token *t = next_token(r);
    Node *n;
    switch (t->type) {
        case T_STRING: {
            n = node(r, E_LITERAL);
            n->value = string(t->text.p, t->text.n);
            /* adjacent strings are one */
            while (peek(r)->type == T_STRING) {
                Token *more = next_token(r);
                Text joined = {r, NULL, 0, 0};
                text_add(&joined, n->value.s.p, n->value.s.n);
                text_add(&joined, more->text.p, more->text.n);
                n->value = text_value(&joined);
            }
            return n;
        }
        case T_INT: case T_FLOAT: {
            char digits[64];
            size_t m = 0;
            for (size_t k = 0; k < t->text.n && m + 1 < sizeof digits; k++)
                if (t->text.p[k] != '_') digits[m++] = t->text.p[k];
            digits[m] = '\0';
            n = node(r, E_LITERAL);
            n->value = t->type == T_INT ? integer(strtoll(digits, NULL, 10)) : real(strtod(digits, NULL));
            return n;
        }
        case T_NAME:
            n = node(r, E_LITERAL);
            if (str_is(t->text, "true") || str_is(t->text, "True")) n->value = boolean(true);
            else if (str_is(t->text, "false") || str_is(t->text, "False")) n->value = boolean(false);
            else if (str_is(t->text, "none") || str_is(t->text, "None")) n->value = none();
            else {
                n->type = E_NAME;
                n->name = t->text;
            }
            return n;
        case T_OP:
            if (str_is(t->text, "(")) {
                if (at_op(r, ")")) {            /* an empty tuple */
                    next_token(r);
                    n = node(r, E_LIST);
                    return n;
                }
                Node *e = expression(r);
                if (at_op(r, ",")) {            /* a tuple, as a list */
                    Nodes items = {0};
                    nodes_push(r, &items, e);
                    while (at_op(r, ",")) {
                        next_token(r);
                        if (at_op(r, ")")) break;
                        nodes_push(r, &items, expression(r));
                    }
                    e = node(r, E_LIST);
                    e->list = items.items;
                    e->count = items.count;
                }
                expect_op(r, ")");
                return e;
            }
            if (str_is(t->text, "[")) {
                Nodes items = {0};
                while (!at_op(r, "]")) {
                    nodes_push(r, &items, expression(r));
                    if (!at_op(r, ",")) break;
                    next_token(r);
                }
                expect_op(r, "]");
                n = node(r, E_LIST);
                n->list = items.items;
                n->count = items.count;
                return n;
            }
            if (str_is(t->text, "{")) {
                Nodes items = {0};
                while (!at_op(r, "}")) {
                    nodes_push(r, &items, expression(r));
                    expect_op(r, ":");
                    nodes_push(r, &items, expression(r));
                    if (!at_op(r, ",")) break;
                    next_token(r);
                }
                expect_op(r, "}");
                n = node(r, E_DICT);
                n->list = items.items;
                n->count = items.count;
                return n;
            }
            break;
        default:
            break;
    }
    fail(r, "line %u: unexpected '%.*s'", t->line, (int)t->text.n, t->text.p);
}

/* .name, [index or slice], (arguments) after a primary */
static Node *postfix(Render *r, Node *n) {
    for (;;) {
        if (at_op(r, ".")) {
            next_token(r);
            Token *t = next_token(r);
            Node *a = node(r, E_ATTR);
            a->a = n;
            if (t->type == T_NAME) a->name = t->text;
            else if (t->type == T_INT) {        /* x.0: an item */
                a->type = E_ITEM;
                a->b = node(r, E_LITERAL);
                a->b->value = integer(strtoll(t->text.p, NULL, 10));
            } else fail(r, "line %u: expected an attribute", t->line);
            n = a;
        } else if (at_op(r, "[")) {
            next_token(r);
            Node *parts[3] = {NULL, NULL, NULL};
            int part = 0;
            bool slice = false;
            while (!at_op(r, "]")) {
                if (at_op(r, ":")) {
                    next_token(r);
                    slice = true;
                    if (++part > 2) fail(r, "line %u: a slice of more than three parts", peek(r)->line);
                    continue;
                }
                parts[part] = expression(r);
            }
            expect_op(r, "]");
            Node *s = node(r, slice ? E_SLICE : E_ITEM);
            s->a = n;
            if (slice) {
                s->b = parts[0];
                s->c = parts[1];
                s->d = parts[2];
            } else {
                if (!parts[0]) fail(r, "line %u: an empty subscript", peek(r)->line);
                s->b = parts[0];
            }
            n = s;
        } else if (at_op(r, "(")) {
            next_token(r);
            Node *c = node(r, E_CALL);
            c->a = n;
            arguments(r, c);
            n = c;
        } else {
            return n;
        }
    }
}

/* name, then (arguments) or, for tests, one argument written without parentheses */
static Node *filter_or_test(Render *r, Node *operand, NodeType type) {
    Node *f = node(r, type);
    f->a = operand;
    f->name = expect_name(r);
    while (at_op(r, ".")) {                 /* a dotted name, as "is divisibleby" never needs but filters may */
        next_token(r);
        Str more = expect_name(r);
        f->name.n = (size_t)(more.p + more.n - f->name.p);
    }
    if (at_op(r, "(")) {
        next_token(r);
        arguments(r, f);
    } else if (type == E_TEST) {
        Token *t = peek(r);
        const bool starts = t->type == T_STRING || t->type == T_INT || t->type == T_FLOAT ||
                            (t->type == T_NAME && !str_is(t->text, "and") && !str_is(t->text, "or") &&
                             !str_is(t->text, "else") && !str_is(t->text, "is") && !str_is(t->text, "in") &&
                             !str_is(t->text, "if") && !str_is(t->text, "not")) ||
                            (t->type == T_OP && (str_is(t->text, "[") || str_is(t->text, "{") || str_is(t->text, "-")));
        if (starts) {
            f->list = (Node **)alloc(r, sizeof(Node *));
            f->list[0] = unary(r, false);
            f->count = 1;
        }
    }
    return f;
}

static Node *unary(Render *r, bool filters) {
    Node *n;
    if (at_op(r, "-") || at_op(r, "+")) {
        const bool neg = at_op(r, "-");
        next_token(r);
        n = node(r, neg ? E_NEG : E_POS);
        n->a = unary(r, false);
    } else {
        n = postfix(r, primary(r));
    }
    if (!filters) return n;
    for (;;) {
        if (at_op(r, "|")) {
            next_token(r);
            n = filter_or_test(r, n, E_FILTER);
        } else if (at_name(r, "is")) {
            next_token(r);
            bool negated = false;
            if (at_name(r, "not")) {
                next_token(r);
                negated = true;
            }
            n = filter_or_test(r, n, E_TEST);
            n->op = negated;
        } else if (at_op(r, "(")) {
            next_token(r);
            Node *c = node(r, E_CALL);
            c->a = n;
            arguments(r, c);
            n = c;
        } else {
            return n;
        }
    }
}

enum { OP_ADD = 1, OP_SUB, OP_MUL, OP_DIV, OP_FLOORDIV, OP_MOD, OP_POW, OP_EQ, OP_NE, OP_LT, OP_LE, OP_GT, OP_GE,
       OP_IN, OP_NOT_IN };

static Node *binary(Render *r, int op, Node *a, Node *b) {
    Node *n = node(r, E_BINARY);
    n->op = op;
    n->a = a;
    n->b = b;
    return n;
}

static Node *power(Render *r) {
    Node *n = unary(r, true);
    while (at_op(r, "**")) {
        next_token(r);
        n = binary(r, OP_POW, n, unary(r, true));
    }
    return n;
}

static Node *product(Render *r) {
    Node *n = power(r);
    for (;;) {
        int op = at_op(r, "*") ? OP_MUL : at_op(r, "/") ? OP_DIV : at_op(r, "//") ? OP_FLOORDIV : at_op(r, "%") ? OP_MOD : 0;
        if (!op) return n;
        next_token(r);
        n = binary(r, op, n, power(r));
    }
}

static Node *concat(Render *r) {
    Node *n = product(r);
    if (!at_op(r, "~")) return n;
    Nodes parts = {0};
    nodes_push(r, &parts, n);
    while (at_op(r, "~")) {
        next_token(r);
        nodes_push(r, &parts, product(r));
    }
    Node *c = node(r, E_CONCAT);
    c->list = parts.items;
    c->count = parts.count;
    return c;
}

static Node *sum(Render *r) {
    Node *n = concat(r);
    for (;;) {
        int op = at_op(r, "+") ? OP_ADD : at_op(r, "-") ? OP_SUB : 0;
        if (!op) return n;
        next_token(r);
        n = binary(r, op, n, concat(r));
    }
}

static Node *compare(Render *r) {
    Node *n = sum(r);
    Nodes operands = {0};
    int ops[16];
    uint32_t count = 0;
    for (;;) {
        int op = 0;
        if (at_op(r, "==")) op = OP_EQ;
        else if (at_op(r, "!=")) op = OP_NE;
        else if (at_op(r, "<")) op = OP_LT;
        else if (at_op(r, "<=")) op = OP_LE;
        else if (at_op(r, ">")) op = OP_GT;
        else if (at_op(r, ">=")) op = OP_GE;
        else if (at_name(r, "in")) op = OP_IN;
        else if (at_name(r, "not") && r->tokens[r->at + 1].type == T_NAME && str_is(r->tokens[r->at + 1].text, "in")) {
            op = OP_NOT_IN;
            next_token(r);
        }
        if (!op) break;
        next_token(r);
        if (count == 16) fail(r, "line %u: too long a chain of comparisons", peek(r)->line);
        if (!count) nodes_push(r, &operands, n);
        ops[count++] = op;
        nodes_push(r, &operands, sum(r));
    }
    if (!count) return n;
    Node *c = node(r, E_COMPARE);
    c->list = operands.items;
    c->count = operands.count;
    c->names = (Str *)alloc(r, count * sizeof(Str));
    c->name_count = count;
    for (uint32_t k = 0; k < count; k++) c->names[k] = str_of(NULL, (size_t)ops[k]);
    return c;
}

static Node *not_expr(Render *r) {
    if (at_name(r, "not")) {
        next_token(r);
        Node *n = node(r, E_NOT);
        n->a = not_expr(r);
        return n;
    }
    return compare(r);
}

static Node *and_expr(Render *r) {
    Node *n = not_expr(r);
    while (at_name(r, "and")) {
        next_token(r);
        Node *a = node(r, E_AND);
        a->a = n;
        a->b = not_expr(r);
        n = a;
    }
    return n;
}

static Node *or_expr(Render *r) {
    Node *n = and_expr(r);
    while (at_name(r, "or")) {
        next_token(r);
        Node *o = node(r, E_OR);
        o->a = n;
        o->b = and_expr(r);
        n = o;
    }
    return n;
}

static Node *expression(Render *r) {
    Node *n = or_expr(r);
    while (at_name(r, "if")) {
        next_token(r);
        Node *c = node(r, E_COND);
        c->a = or_expr(r);
        c->b = n;
        if (at_name(r, "else")) {
            next_token(r);
            c->c = or_expr(r);
        }
        n = c;
    }
    return n;
}

static void expect_close(Render *r) {
    Token *t = next_token(r);
    if (t->type != T_BLOCK_CLOSE) fail(r, "line %u: expected the end of the tag, found '%.*s'", t->line, (int)t->text.n,
                                       t->text.p);
}

/* Statements up to a block tag whose first name is one of `ends` (NULL-terminated), which is left to the caller
   (positioned on that name); at the top level up to the template's end. */
static Node *body(Render *r, const char *const *ends, Str *found) {
    Nodes items = {0};
    for (;;) {
        Token *t = next_token(r);
        if (t->type == T_EOF) {
            if (ends) {
                const char *const *last = ends;
                while (last[1]) last++;
                fail(r, "line %u: the template ends inside a block (expected {%% %s %%})", t->line, *last);
            }
            break;
        }
        if (t->type == T_TEXT) {
            Node *n = node(r, N_TEXT);
            n->name = t->text;
            nodes_push(r, &items, n);
            continue;
        }
        if (t->type == T_PRINT_OPEN) {
            Node *n = node(r, N_PRINT);
            n->a = expression(r);
            Token *c = next_token(r);
            if (c->type != T_PRINT_CLOSE) fail(r, "line %u: expected '}}', found '%.*s'", c->line, (int)c->text.n, c->text.p);
            nodes_push(r, &items, n);
            continue;
        }
        /* a block tag */
        Str word = expect_name(r);
        if (ends) {
            bool end = false;
            for (const char *const *e = ends; *e; e++) end |= str_is(word, *e);
            if (end) {
                *found = word;
                break;
            }
        }
        const uint32_t line = t->line;
        if (str_is(word, "if")) {
            Node *n = node(r, N_IF);
            n->a = expression(r);
            expect_close(r);
            static const char *const if_ends[] = {"elif", "else", "endif", NULL};
            Str end;
            n->b = body(r, if_ends, &end);
            Node *tail = n;
            while (str_is(end, "elif")) {
                Node *e = node(r, N_IF);
                e->a = expression(r);
                expect_close(r);
                e->b = body(r, if_ends, &end);
                Node *wrap = node(r, N_BODY);
                wrap->list = (Node **)alloc(r, sizeof(Node *));
                wrap->list[0] = e;
                wrap->count = 1;
                tail->c = wrap;
                tail = e;
            }
            if (str_is(end, "else")) {
                expect_close(r);
                static const char *const else_ends[] = {"endif", NULL};
                tail->c = body(r, else_ends, &end);
            }
            expect_close(r);
            nodes_push(r, &items, n);
        } else if (str_is(word, "for")) {
            Node *n = node(r, N_FOR);
            Str targets[8];
            uint32_t count = 0;
            const bool paren = at_op(r, "(");
            if (paren) next_token(r);
            for (;;) {
                if (count == 8) fail(r, "line %u: too many loop variables", line);
                targets[count++] = expect_name(r);
                if (!at_op(r, ",")) break;
                next_token(r);
            }
            if (paren) expect_op(r, ")");
            n->names = (Str *)alloc(r, count * sizeof(Str));
            memcpy(n->names, targets, count * sizeof(Str));
            n->name_count = count;
            if (!at_name(r, "in")) fail(r, "line %u: expected 'in' in a for", line);
            next_token(r);
            n->a = or_expr(r);
            if (at_name(r, "if")) {
                next_token(r);
                n->d = or_expr(r);
            }
            if (at_name(r, "recursive")) fail(r, "line %u: recursive loops are not supported", line);
            expect_close(r);
            static const char *const for_ends[] = {"else", "endfor", NULL};
            Str end;
            n->b = body(r, for_ends, &end);
            if (str_is(end, "else")) {
                expect_close(r);
                static const char *const else_ends[] = {"endfor", NULL};
                n->c = body(r, else_ends, &end);
            }
            expect_close(r);
            nodes_push(r, &items, n);
        } else if (str_is(word, "set")) {
            Node *n = node(r, N_SET);
            n->name = expect_name(r);
            if (at_op(r, ".")) {                /* ns.attribute */
                next_token(r);
                n->a = node(r, E_NAME);
                n->a->name = n->name;
                n->name = expect_name(r);
            } else if (at_op(r, ",")) {         /* a, b = ... */
                Str targets[8];
                uint32_t count = 0;
                targets[count++] = n->name;
                while (at_op(r, ",")) {
                    next_token(r);
                    if (count == 8) fail(r, "line %u: too many names to set", line);
                    targets[count++] = expect_name(r);
                }
                n->names = (Str *)alloc(r, count * sizeof(Str));
                memcpy(n->names, targets, count * sizeof(Str));
                n->name_count = count;
            }
            if (at_op(r, "=")) {
                next_token(r);
                n->b = expression(r);
                if (at_op(r, ",")) {            /* a tuple */
                    Nodes parts = {0};
                    nodes_push(r, &parts, n->b);
                    while (at_op(r, ",")) {
                        next_token(r);
                        nodes_push(r, &parts, expression(r));
                    }
                    n->b = node(r, E_LIST);
                    n->b->list = parts.items;
                    n->b->count = parts.count;
                }
                expect_close(r);
            } else {
                /* {% set name %}...{% endset %} */
                if (n->a || n->names) fail(r, "line %u: a block set takes one name", line);
                n->type = N_SET_BLOCK;
                expect_close(r);
                static const char *const set_ends[] = {"endset", NULL};
                Str end;
                n->b = body(r, set_ends, &end);
                expect_close(r);
            }
            nodes_push(r, &items, n);
        } else if (str_is(word, "macro")) {
            Node *n = node(r, N_MACRO);
            n->name = expect_name(r);
            expect_op(r, "(");
            Str params[32];
            Nodes defaults = {0};
            uint32_t count = 0;
            while (!at_op(r, ")")) {
                if (count == 32) fail(r, "line %u: too many parameters", line);
                params[count++] = expect_name(r);
                Node *def = NULL;
                if (at_op(r, "=")) {
                    next_token(r);
                    def = expression(r);
                }
                nodes_push(r, &defaults, def);
                if (!at_op(r, ",")) break;
                next_token(r);
            }
            expect_op(r, ")");
            expect_close(r);
            n->names = (Str *)alloc(r, (count + 1u) * sizeof(Str));
            memcpy(n->names, params, count * sizeof(Str));
            n->name_count = count;
            n->list = defaults.items;
            n->count = defaults.count;
            static const char *const macro_ends[] = {"endmacro", NULL};
            Str end;
            n->b = body(r, macro_ends, &end);
            expect_close(r);
            nodes_push(r, &items, n);
        } else if (str_is(word, "break") || str_is(word, "continue")) {
            expect_close(r);
            nodes_push(r, &items, node(r, str_is(word, "break") ? N_BREAK : N_CONTINUE));
        } else if (str_is(word, "generation")) {
            expect_close(r);
            static const char *const generation_ends[] = {"endgeneration", NULL};
            Str end;
            nodes_push(r, &items, body(r, generation_ends, &end));
            expect_close(r);
        } else if (str_is(word, "do")) {
            Node *n = node(r, N_PRINT);
            n->a = expression(r);
            n->op = 1;                          /* evaluated, not printed */
            expect_close(r);
            nodes_push(r, &items, n);
        } else {
            fail(r, "line %u: the tag '%.*s' is not supported", line, (int)word.n, word.p);
        }
    }
    Node *b = node(r, N_BODY);
    b->list = items.items;
    b->count = items.count;
    return b;
}

/* ------------------------------------------------------------------------- evaluation */

enum {
    G_RANGE = 1, G_NAMESPACE, G_DICT, G_RAISE, G_STRFTIME, G_CYCLER_NONE,
    /* methods */
    M_STRIP = 100, M_LSTRIP, M_RSTRIP, M_SPLIT, M_RSPLIT, M_STARTSWITH, M_ENDSWITH, M_REPLACE, M_UPPER, M_LOWER,
    M_TITLE, M_CAPITALIZE, M_FIND, M_RFIND, M_COUNT, M_JOIN, M_ITEMS, M_KEYS, M_VALUES, M_GET, M_ISDIGIT,
    M_ISALPHA, M_ISSPACE, M_SPLITLINES, M_INDEX
};

static Value function(int id, const Value *self) {
    Value v = undef();
    v.kind = FUNC;
    v.fn.id = id;
    v.fn.self = self;
    return v;
}

static Value lookup(Render *r, Str name) {
    for (Frame *f = r->frame; f; f = f->parent) {
        Value *v = dict_find(&f->vars, name.p, name.n);
        if (v) return *v;
    }
    static const struct { const char *name; int id; } globals[] = {
        {"range", G_RANGE}, {"namespace", G_NAMESPACE}, {"dict", G_DICT}, {"raise_exception", G_RAISE},
        {"strftime_now", G_STRFTIME},
    };
    for (size_t k = 0; k < sizeof globals / sizeof *globals; k++)
        if (str_is(name, globals[k].name)) return function(globals[k].id, NULL);
    return undef();
}

static void set_var(Render *r, Str name, Value v) {
    dict_set(r, &r->frame->vars, copy_string(r, name.p, name.n), v);
}

static Value eval(Render *r, const Node *n);

static Value method(Render *r, const Value *self, Str name) {
    static const struct { const char *name; int id; uint8_t kind; } methods[] = {
        {"strip", M_STRIP, STR}, {"lstrip", M_LSTRIP, STR}, {"rstrip", M_RSTRIP, STR}, {"split", M_SPLIT, STR},
        {"rsplit", M_RSPLIT, STR}, {"startswith", M_STARTSWITH, STR}, {"endswith", M_ENDSWITH, STR},
        {"replace", M_REPLACE, STR}, {"upper", M_UPPER, STR}, {"lower", M_LOWER, STR}, {"title", M_TITLE, STR},
        {"capitalize", M_CAPITALIZE, STR}, {"find", M_FIND, STR}, {"rfind", M_RFIND, STR}, {"count", M_COUNT, STR},
        {"join", M_JOIN, STR}, {"isdigit", M_ISDIGIT, STR}, {"isalpha", M_ISALPHA, STR}, {"isspace", M_ISSPACE, STR},
        {"splitlines", M_SPLITLINES, STR}, {"index", M_INDEX, STR}, {"items", M_ITEMS, DICT}, {"keys", M_KEYS, DICT},
        {"values", M_VALUES, DICT}, {"get", M_GET, DICT}, {"count", M_COUNT, LIST}, {"index", M_INDEX, LIST},
    };
    if (self->kind == DICT && self->d->ns) return undef();
    for (size_t k = 0; k < sizeof methods / sizeof *methods; k++)
        if (methods[k].kind == self->kind && str_is(name, methods[k].name)) {
            Value *bound = (Value *)alloc(r, sizeof(Value));
            *bound = *self;
            return function(methods[k].id, bound);
        }
    return undef();
}

/* obj.name: a method or attribute first, then an item (jinja2's getattr) */
static Value attribute(Render *r, const Value *obj, Str name) {
    if (obj->kind == UNDEF) return undef();
    Value m = method(r, obj, name);
    if (m.kind != UNDEF) return m;
    if (obj->kind == DICT) {
        Value *v = dict_find(obj->d, name.p, name.n);
        return v ? *v : undef();
    }
    return undef();
}

static int64_t index_of(Render *r, const Value *index, uint32_t count) {
    if (index->kind != INT && index->kind != BOOL) fail(r, "indices must be integers, not %s", kind_name(index));
    int64_t i = as_int(index);
    if (i < 0) i += count;
    return i;
}

/* obj[index]: an item first, then an attribute (jinja2's getitem) */
static Value item(Render *r, const Value *obj, const Value *index) {
    switch (obj->kind) {
        case DICT: {
            Value *v = obj->d->ns ? NULL : dict_find_value(obj->d, index);
            if (obj->d->ns && index->kind == STR) v = dict_find(obj->d, index->s.p, index->s.n);
            if (v) return *v;
            return index->kind == STR ? attribute(r, obj, str_of(index->s.p, index->s.n)) : undef();
        }
        case LIST: {
            if (index->kind == STR) return attribute(r, obj, str_of(index->s.p, index->s.n));
            int64_t i = index_of(r, index, obj->l->count);
            return i >= 0 && i < obj->l->count ? obj->l->items[i] : undef();
        }
        case STR: {
            if (index->kind == STR) return attribute(r, obj, str_of(index->s.p, index->s.n));
            /* characters, not bytes */
            uint32_t chars = 0;
            for (size_t k = 0; k < obj->s.n; k += utf8_length((const unsigned char *)obj->s.p + k, obj->s.n - k)) chars++;
            int64_t i = index_of(r, index, chars);
            if (i < 0 || i >= chars) return undef();
            size_t k = 0;
            for (int64_t c = 0; c < i; c++) k += utf8_length((const unsigned char *)obj->s.p + k, obj->s.n - k);
            return string(obj->s.p + k, utf8_length((const unsigned char *)obj->s.p + k, obj->s.n - k));
        }
        default:
            return undef();
    }
}

/* Python's slice indices: start, stop, step over count elements; the positions it takes into out. */
static uint32_t slice_positions(Render *r, const Node *n, uint32_t count, uint32_t *out) {
    int64_t step = 1;
    if (n->d) {
        Value s = eval(r, n->d);
        if (s.kind != NONE) step = as_int(&s);
        if (step == 0) fail(r, "slice step cannot be zero");
    }
    int64_t start = step > 0 ? 0 : (int64_t)count - 1, stop = step > 0 ? (int64_t)count : -1;
    if (n->b) {
        Value s = eval(r, n->b);
        if (s.kind != NONE) {
            start = as_int(&s);
            if (start < 0) start += count;
            if (step > 0) start = start < 0 ? 0 : start > count ? count : start;
            else start = start < -1 ? -1 : start >= count ? (int64_t)count - 1 : start;
        }
    }
    if (n->c) {
        Value s = eval(r, n->c);
        if (s.kind != NONE) {
            stop = as_int(&s);
            if (stop < 0) stop += count;
            if (step > 0) stop = stop < 0 ? 0 : stop > count ? count : stop;
            else stop = stop < -1 ? -1 : stop >= count ? (int64_t)count - 1 : stop;
        }
    }
    uint32_t m = 0;
    for (int64_t i = start; step > 0 ? i < stop : i > stop; i += step) out[m++] = (uint32_t)i;
    return m;
}

static Value slice(Render *r, const Node *n) {
    Value obj = eval(r, n->a);
    if (obj.kind == LIST) {
        uint32_t *at = (uint32_t *)alloc(r, (obj.l->count + 1u) * sizeof(uint32_t));
        uint32_t m = slice_positions(r, n, obj.l->count, at);
        Value out = new_list(r, m);
        for (uint32_t k = 0; k < m; k++) list_push(r, out.l, obj.l->items[at[k]]);
        return out;
    }
    if (obj.kind == STR) {
        uint32_t chars = 0;
        for (size_t k = 0; k < obj.s.n; k += utf8_length((const unsigned char *)obj.s.p + k, obj.s.n - k)) chars++;
        size_t *starts = (size_t *)alloc(r, (chars + 1u) * sizeof(size_t));
        uint32_t c = 0;
        for (size_t k = 0; k < obj.s.n; k += utf8_length((const unsigned char *)obj.s.p + k, obj.s.n - k)) starts[c++] = k;
        starts[chars] = obj.s.n;
        uint32_t *at = (uint32_t *)alloc(r, (chars + 1u) * sizeof(uint32_t));
        uint32_t m = slice_positions(r, n, chars, at);
        Text t = {r, NULL, 0, 0};
        for (uint32_t k = 0; k < m; k++) text_add(&t, obj.s.p + starts[at[k]], starts[at[k] + 1] - starts[at[k]]);
        return text_value(&t);
    }
    if (obj.kind == UNDEF) return undef();
    fail(r, "'%s' object is not subscriptable", kind_name(&obj));
}

/* Iterable values as a list: lists, a dict's keys, a string's characters. */
static Value iterable(Render *r, const Value *v) {
    if (v->kind == LIST) return *v;
    if (v->kind == DICT) {
        Value out = new_list(r, v->d->count);
        for (uint32_t k = 0; k < v->d->count; k++) list_push(r, out.l, v->d->keys[k]);
        return out;
    }
    if (v->kind == STR) {
        Value out = new_list(r, (uint32_t)v->s.n);
        for (size_t k = 0; k < v->s.n;) {
            size_t len = utf8_length((const unsigned char *)v->s.p + k, v->s.n - k);
            list_push(r, out.l, string(v->s.p + k, len));
            k += len;
        }
        return out;
    }
    if (v->kind == UNDEF) return new_list(r, 1);
    fail(r, "'%s' object is not iterable", kind_name(v));
}

static bool is_strip_char(const Value *chars, const char *p, size_t len) {
    if (!chars || chars->kind != STR) return len == 1 && is_space(*p);
    for (size_t k = 0; k + len <= chars->s.n; k++)
        if (!memcmp(chars->s.p + k, p, len)) return true;
    return false;
}

static Value strip(Render *r, const Value *s, const Value *chars, bool left, bool right) {
    (void)r;
    size_t a = 0, b = s->s.n;
    while (left && a < b) {
        size_t len = utf8_length((const unsigned char *)s->s.p + a, b - a);
        if (!is_strip_char(chars, s->s.p + a, len)) break;
        a += len;
    }
    while (right && b > a) {
        size_t c = b - 1;
        while (c > a && ((unsigned char)s->s.p[c] & 0xC0) == 0x80) c--;
        if (!is_strip_char(chars, s->s.p + c, b - c)) break;
        b = c;
    }
    return string(s->s.p + a, b - a);
}

static Value split(Render *r, const Value *s, const Value *sep, int64_t max, bool from_right) {
    Value out = new_list(r, 4);
    const char *p = s->s.p;
    size_t n = s->s.n;
    if (!sep || sep->kind == NONE || sep->kind == UNDEF) {
        /* runs of whitespace; none at the ends */
        size_t k = 0;
        int64_t made = 0;
        if (from_right && max >= 0) fail(r, "rsplit() of whitespace with a limit is not supported");
        while (k < n) {
            while (k < n && is_space(p[k])) k++;
            if (k >= n) break;
            size_t start = k;
            if (max >= 0 && made == max) {
                size_t e = n;
                while (e > start && is_space(p[e - 1])) e--;
                list_push(r, out.l, string(p + start, e - start));
                break;
            }
            while (k < n && !is_space(p[k])) k++;
            list_push(r, out.l, string(p + start, k - start));
            made++;
        }
        return out;
    }
    if (sep->kind != STR || !sep->s.n) fail(r, "empty separator");
    const size_t m = sep->s.n;
    if (!from_right) {
        size_t start = 0;
        int64_t made = 0;
        for (size_t k = 0; k + m <= n;) {
            if ((max < 0 || made < max) && !memcmp(p + k, sep->s.p, m)) {
                list_push(r, out.l, string(p + start, k - start));
                k += m;
                start = k;
                made++;
            } else {
                k++;
            }
        }
        list_push(r, out.l, string(p + start, n - start));
        return out;
    }
    /* from the right: the pieces found backwards, then reversed */
    Value pieces = new_list(r, 4);
    size_t end = n;
    int64_t made = 0;
    for (size_t k = n; k >= m;) {
        if ((max < 0 || made < max) && !memcmp(p + k - m, sep->s.p, m)) {
            list_push(r, pieces.l, string(p + k, end - k));
            k -= m;
            end = k;
            made++;
        } else {
            k--;
        }
    }
    list_push(r, pieces.l, string(p, end));
    for (uint32_t k = pieces.l->count; k-- > 0;) list_push(r, out.l, pieces.l->items[k]);
    return out;
}

static Value replace(Render *r, const Value *s, const Value *old, const Value *with, int64_t count) {
    if (old->kind != STR || with->kind != STR) fail(r, "replace() takes strings");
    Text t = {r, NULL, 0, 0};
    const size_t m = old->s.n;
    int64_t made = 0;
    size_t k = 0;
    if (!m) {
        /* between every character */
        for (; k <= s->s.n; ) {
            if (count < 0 || made < count) { text_add(&t, with->s.p, with->s.n); made++; }
            if (k == s->s.n) break;
            size_t len = utf8_length((const unsigned char *)s->s.p + k, s->s.n - k);
            text_add(&t, s->s.p + k, len);
            k += len;
        }
        return text_value(&t);
    }
    while (k < s->s.n) {
        if (k + m <= s->s.n && (count < 0 || made < count) && !memcmp(s->s.p + k, old->s.p, m)) {
            text_add(&t, with->s.p, with->s.n);
            k += m;
            made++;
        } else {
            text_add(&t, s->s.p + k, 1);
            k++;
        }
    }
    return text_value(&t);
}

static Value change_case(Render *r, const Value *s, int mode) {
    /* ASCII letters only: 1 upper, 2 lower, 3 title, 4 capitalize */
    char *out = (char *)alloc(r, s->s.n + 1);
    bool start = true;
    for (size_t k = 0; k < s->s.n; k++) {
        char c = s->s.p[k];
        const bool letter = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
        bool up = mode == 1 || ((mode == 3 || mode == 4) && start);
        if (mode == 4 && k > 0) up = false;
        if (letter) c = up ? (char)(c & ~32) : (char)(c | 32);
        out[k] = c;
        start = mode == 3 ? !letter && !((unsigned char)c >= 0x80) : false;
    }
    out[s->s.n] = '\0';
    return string(out, s->s.n);
}

static bool starts_with(const Value *s, const Value *prefix, bool end) {
    if (prefix->kind == LIST) {
        for (uint32_t k = 0; k < prefix->l->count; k++)
            if (starts_with(s, &prefix->l->items[k], end)) return true;
        return false;
    }
    if (prefix->kind != STR || prefix->s.n > s->s.n) return false;
    return !memcmp(s->s.p + (end ? s->s.n - prefix->s.n : 0), prefix->s.p, prefix->s.n);
}

static int64_t find(const Value *s, const Value *sub, bool from_right) {
    if (sub->kind != STR) return -1;
    if (sub->s.n > s->s.n) return -1;
    if (!from_right) {
        for (size_t k = 0; k + sub->s.n <= s->s.n; k++)
            if (!memcmp(s->s.p + k, sub->s.p, sub->s.n)) return (int64_t)k;
    } else {
        for (size_t k = s->s.n - sub->s.n + 1; k-- > 0;)
            if (!memcmp(s->s.p + k, sub->s.p, sub->s.n)) return (int64_t)k;
    }
    return -1;
}

/* the characters of s before byte offset k */
static int64_t chars_before(const Value *s, int64_t k) {
    if (k < 0) return k;
    int64_t c = 0;
    for (int64_t b = 0; b < k; b += (int64_t)utf8_length((const unsigned char *)s->s.p + b, s->s.n - (size_t)b)) c++;
    return c;
}

static const Value *keyword(const Node *n, const Value *values, const char *name) {
    for (uint32_t k = 0; k < n->name_count; k++)
        if (str_is(n->names[k], name)) return &values[n->count + k];
    return NULL;
}

static Value call_function(Render *r, const Value *f, const Node *n, const Value *args, uint32_t count);

static void need(Render *r, uint32_t count, uint32_t least, uint32_t most, const char *name) {
    if (count < least || count > most) fail(r, "%s() takes %u to %u arguments (%u given)", name, least, most, count);
}

static Value call_method(Render *r, int id, const Value *self, const Node *n, const Value *a, uint32_t count) {
    (void)n;
    switch (id) {
        case M_STRIP: case M_LSTRIP: case M_RSTRIP:
            need(r, count, 0, 1, "strip");
            return strip(r, self, count ? &a[0] : NULL, id != M_RSTRIP, id != M_LSTRIP);
        case M_SPLIT: case M_RSPLIT:
            need(r, count, 0, 2, "split");
            return split(r, self, count ? &a[0] : NULL, count > 1 ? as_int(&a[1]) : -1, id == M_RSPLIT);
        case M_STARTSWITH: case M_ENDSWITH:
            need(r, count, 1, 1, "startswith");
            return boolean(starts_with(self, &a[0], id == M_ENDSWITH));
        case M_REPLACE:
            need(r, count, 2, 3, "replace");
            return replace(r, self, &a[0], &a[1], count > 2 ? as_int(&a[2]) : -1);
        case M_UPPER: return change_case(r, self, 1);
        case M_LOWER: return change_case(r, self, 2);
        case M_TITLE: return change_case(r, self, 3);
        case M_CAPITALIZE: {
            Value lower = change_case(r, self, 2);
            return change_case(r, &lower, 4);
        }
        case M_FIND: case M_RFIND:
            need(r, count, 1, 1, "find");
            return integer(chars_before(self, find(self, &a[0], id == M_RFIND)));
        case M_INDEX:
            need(r, count, 1, 1, "index");
            if (self->kind == LIST) {
                for (uint32_t k = 0; k < self->l->count; k++)
                    if (equal(&self->l->items[k], &a[0])) return integer(k);
                fail(r, "list.index(x): x not in list");
            } else {
                int64_t at = find(self, &a[0], false);
                if (at < 0) fail(r, "substring not found");
                return integer(chars_before(self, at));
            }
        case M_COUNT:
            need(r, count, 1, 1, "count");
            if (self->kind == LIST) {
                int64_t c = 0;
                for (uint32_t k = 0; k < self->l->count; k++) c += equal(&self->l->items[k], &a[0]);
                return integer(c);
            } else {
                if (a[0].kind != STR || !a[0].s.n) fail(r, "count() takes a string that is not empty");
                int64_t c = 0;
                for (size_t k = 0; k + a[0].s.n <= self->s.n;)
                    if (!memcmp(self->s.p + k, a[0].s.p, a[0].s.n)) { c++; k += a[0].s.n; } else k++;
                return integer(c);
            }
        case M_JOIN: {
            need(r, count, 1, 1, "join");
            Value items = iterable(r, &a[0]);
            Text t = {r, NULL, 0, 0};
            for (uint32_t k = 0; k < items.l->count; k++) {
                if (k) text_add(&t, self->s.p, self->s.n);
                if (items.l->items[k].kind != STR) fail(r, "sequence item %u: expected str instance, %s found", k,
                                                         kind_name(&items.l->items[k]));
                text_add(&t, items.l->items[k].s.p, items.l->items[k].s.n);
            }
            return text_value(&t);
        }
        case M_ISDIGIT: case M_ISALPHA: case M_ISSPACE: {
            if (!self->s.n) return boolean(false);
            for (size_t k = 0; k < self->s.n; k++) {
                char c = self->s.p[k];
                bool ok = id == M_ISDIGIT ? is_digit(c) : id == M_ISSPACE ? is_space(c)
                                                        : ((c | 32) >= 'a' && (c | 32) <= 'z');
                if (!ok) return boolean(false);
            }
            return boolean(true);
        }
        case M_SPLITLINES: {
            Value out = new_list(r, 4);
            size_t start = 0;
            for (size_t k = 0; k < self->s.n; k++)
                if (self->s.p[k] == '\n' || self->s.p[k] == '\r') {
                    list_push(r, out.l, string(self->s.p + start, k - start));
                    if (self->s.p[k] == '\r' && k + 1 < self->s.n && self->s.p[k + 1] == '\n') k++;
                    start = k + 1;
                }
            if (start < self->s.n) list_push(r, out.l, string(self->s.p + start, self->s.n - start));
            return out;
        }
        case M_ITEMS: case M_KEYS: case M_VALUES: {
            Value out = new_list(r, self->d->count);
            for (uint32_t k = 0; k < self->d->count; k++) {
                if (id == M_KEYS) list_push(r, out.l, self->d->keys[k]);
                else if (id == M_VALUES) list_push(r, out.l, self->d->values[k]);
                else {
                    Value pair = new_list(r, 2);
                    list_push(r, pair.l, self->d->keys[k]);
                    list_push(r, pair.l, self->d->values[k]);
                    list_push(r, out.l, pair);
                }
            }
            return out;
        }
        case M_GET: {
            need(r, count, 1, 2, "get");
            Value *v = dict_find_value(self->d, &a[0]);
            return v ? *v : count > 1 ? a[1] : none();
        }
        default:
            fail(r, "an unknown method");
    }
}

static Value call_macro(Render *r, const Node *macro, const Node *n, const Value *args, uint32_t count) {
    if (count > macro->name_count) fail(r, "the macro '%.*s' takes %u arguments", (int)macro->name.n, macro->name.p,
                                        macro->name_count);
    if (++r->depth > 64) fail(r, "macros call each other too deeply");
    Frame frame = {r->root, {0}};
    Frame *saved = r->frame;
    r->frame = &frame;
    for (uint32_t k = 0; k < macro->name_count; k++) {
        Value v = undef();
        if (k < count) v = args[k];
        else {
            bool given = false;
            for (uint32_t w = 0; n && w < n->name_count; w++)
                if (n->names[w].n == macro->names[k].n && !memcmp(n->names[w].p, macro->names[k].p, macro->names[k].n)) {
                    v = args[n->count + w];
                    given = true;
                }
            if (!given && macro->list[k]) v = eval(r, macro->list[k]);
        }
        set_var(r, macro->names[k], v);
    }
    /* the macro's body into a text of its own */
    char *out = r->out;
    size_t out_n = r->out_n, out_cap = r->out_cap;
    r->out = NULL;
    r->out_n = r->out_cap = 0;
    exec(r, macro->b);
    Value text = r->out ? copy_string(r, r->out, r->out_n) : string("", 0);
    free(r->out);
    r->out = out;
    r->out_n = out_n;
    r->out_cap = out_cap;
    r->frame = saved;
    r->depth--;
    return text;
}

static Value call_function(Render *r, const Value *f, const Node *n, const Value *args, uint32_t count) {
    if (f->kind == MACRO) return call_macro(r, f->macro, n, args, count);
    if (f->kind != FUNC) fail(r, "'%s' object is not callable", kind_name(f));
    if (f->fn.self) return call_method(r, f->fn.id, f->fn.self, n, args, count);
    switch (f->fn.id) {
        case G_RANGE: {
            need(r, count, 1, 3, "range");
            int64_t start = 0, stop, step = 1;
            if (count == 1) stop = as_int(&args[0]);
            else {
                start = as_int(&args[0]);
                stop = as_int(&args[1]);
                if (count == 3) step = as_int(&args[2]);
            }
            if (!step) fail(r, "range() arg 3 must not be zero");
            Value out = new_list(r, 8);
            for (int64_t i = start; step > 0 ? i < stop : i > stop; i += step) {
                if (out.l->count >= (1u << 20)) fail(r, "range() too long");
                list_push(r, out.l, integer(i));
            }
            return out;
        }
        case G_NAMESPACE: case G_DICT: {
            Value d = new_dict(r);
            d.d->ns = f->fn.id == G_NAMESPACE;
            for (uint32_t k = 0; k < count; k++) {
                if (args[k].kind != DICT) fail(r, "namespace() and dict() take a dict or keywords");
                for (uint32_t j = 0; j < args[k].d->count; j++) dict_set(r, d.d, args[k].d->keys[j], args[k].d->values[j]);
            }
            for (uint32_t k = 0; n && k < n->name_count; k++)
                dict_set(r, d.d, copy_string(r, n->names[k].p, n->names[k].n), args[n->count + k]);
            return d;
        }
        case G_RAISE: {
            Value message = count ? to_str(r, &args[0]) : cstring("");
            fail(r, "%.*s", (int)message.s.n, message.s.p);
        }
        case G_STRFTIME: {
            need(r, count, 1, 1, "strftime_now");
            Value format = to_str(r, &args[0]);
            char fmt[128], out[256];
            if (format.s.n >= sizeof fmt) fail(r, "strftime_now(): too long a format");
            memcpy(fmt, format.s.p, format.s.n);
            fmt[format.s.n] = '\0';
            struct tm local;
            const time_t now = r->now;
#if defined(_WIN32)
            localtime_s(&local, &now);
#else
            localtime_r(&now, &local);
#endif
            size_t len = strftime(out, sizeof out, fmt, &local);
            return copy_string(r, out, len);
        }
        default:
            fail(r, "an unknown function");
    }
}

/* ------------------------------------------------------------------------- filters and tests */

static Value attribute_path(Render *r, const Value *v, const Value *path) {
    /* "a.b" or an integer, as jinja2's make_attrgetter */
    if (path->kind != STR) return item(r, v, path);
    Value cur = *v;
    size_t start = 0;
    for (size_t k = 0; k <= path->s.n; k++)
        if (k == path->s.n || path->s.p[k] == '.') {
            Value key = string(path->s.p + start, k - start);
            bool digits = key.s.n > 0;
            for (size_t c = 0; c < key.s.n; c++) digits &= is_digit(key.s.p[c]);
            if (digits) key = integer(strtoll(key.s.p, NULL, 10));
            cur = item(r, &cur, &key);
            start = k + 1;
        }
    return cur;
}

static bool test(Render *r, Str name, const Value *v, const Value *args, uint32_t count);

static Value filter(Render *r, const Node *n, Value v, const Value *a, uint32_t count) {
    const Str name = n->name;
    if (str_is(name, "trim")) {
        Value s = to_str(r, &v);
        return strip(r, &s, count ? &a[0] : NULL, true, true);
    }
    if (str_is(name, "length") || str_is(name, "count")) {
        if (v.kind == STR) return integer(chars_before(&v, (int64_t)v.s.n));
        if (v.kind == LIST) return integer(v.l->count);
        if (v.kind == DICT) return integer(v.d->count);
        if (v.kind == UNDEF) return integer(0);
        fail(r, "object of type '%s' has no len()", kind_name(&v));
    }
    if (str_is(name, "tojson")) {
        const Value *indent = keyword(n, a - 0, "indent");
        int64_t spaces = -1;
        if (count) spaces = a[0].kind == NONE ? -1 : as_int(&a[0]);
        if (indent) spaces = indent->kind == NONE ? -1 : as_int(indent);
        const Value *sort = keyword(n, a, "sort_keys");
        const Value *separators = keyword(n, a, "separators");
        const char *item_sep = spaces >= 0 ? "," : ", ", *key_sep = ": ";
        if (separators && separators->kind == LIST && separators->l->count == 2 &&
            separators->l->items[0].kind == STR && separators->l->items[1].kind == STR) {
            item_sep = copy_string(r, separators->l->items[0].s.p, separators->l->items[0].s.n).s.p;
            key_sep = copy_string(r, separators->l->items[1].s.p, separators->l->items[1].s.n).s.p;
        }
        Text t = {r, NULL, 0, 0};
        json_into(r, &t, &v, spaces, 0, item_sep, key_sep, sort && truthy(sort));
        return text_value(&t);
    }
    if (str_is(name, "string")) return to_str(r, &v);
    if (str_is(name, "upper") || str_is(name, "lower") || str_is(name, "title") || str_is(name, "capitalize")) {
        Value s = to_str(r, &v);
        if (str_is(name, "capitalize")) {
            Value lower = change_case(r, &s, 2);
            return change_case(r, &lower, 4);
        }
        return change_case(r, &s, str_is(name, "upper") ? 1 : str_is(name, "lower") ? 2 : 3);
    }
    if (str_is(name, "join")) {
        Value items = iterable(r, &v);
        const Value *attr = keyword(n, a, "attribute");
        Value sep = count ? to_str(r, &a[0]) : cstring("");
        Text t = {r, NULL, 0, 0};
        for (uint32_t k = 0; k < items.l->count; k++) {
            if (k) text_add(&t, sep.s.p, sep.s.n);
            Value e = attr ? attribute_path(r, &items.l->items[k], attr) : items.l->items[k];
            str_into(r, &t, &e);
        }
        return text_value(&t);
    }
    if (str_is(name, "first") || str_is(name, "last")) {
        Value items = iterable(r, &v);
        if (!items.l->count) return undef();
        return items.l->items[str_is(name, "first") ? 0 : items.l->count - 1];
    }
    if (str_is(name, "list")) return iterable(r, &v);
    if (str_is(name, "reverse")) {
        if (v.kind == STR) {
            Value chars = iterable(r, &v);
            Text t = {r, NULL, 0, 0};
            for (uint32_t k = chars.l->count; k-- > 0;) text_add(&t, chars.l->items[k].s.p, chars.l->items[k].s.n);
            return text_value(&t);
        }
        Value items = iterable(r, &v), out = new_list(r, items.l->count);
        for (uint32_t k = items.l->count; k-- > 0;) list_push(r, out.l, items.l->items[k]);
        return out;
    }
    if (str_is(name, "items")) {
        if (v.kind == UNDEF) return new_list(r, 1);
        if (v.kind != DICT) fail(r, "items: a %s is no mapping", kind_name(&v));
        return call_method(r, M_ITEMS, &v, NULL, NULL, 0);
    }
    if (str_is(name, "default") || str_is(name, "d")) {
        const bool boolean_mode = count > 1 && truthy(&a[1]);
        if (v.kind == UNDEF || (boolean_mode && !truthy(&v))) return count ? a[0] : cstring("");
        return v;
    }
    if (str_is(name, "int")) {
        if (v.kind == INT) return v;
        if (v.kind == FLOAT) return integer((int64_t)v.f);
        if (v.kind == BOOL) return integer(v.b);
        if (v.kind == STR) {
            char buf[64];
            size_t m = v.s.n < sizeof buf - 1 ? v.s.n : sizeof buf - 1;
            memcpy(buf, v.s.p, m);
            buf[m] = '\0';
            char *end;
            long long i = strtoll(buf, &end, 10);
            if (end != buf) return integer(i);
            double f = strtod(buf, &end);
            if (end != buf) return integer((int64_t)f);
        }
        return count ? a[0] : integer(0);
    }
    if (str_is(name, "float")) {
        if (numeric(&v)) return real(as_double(&v));
        if (v.kind == STR) {
            char buf[64];
            size_t m = v.s.n < sizeof buf - 1 ? v.s.n : sizeof buf - 1;
            memcpy(buf, v.s.p, m);
            buf[m] = '\0';
            char *end;
            double f = strtod(buf, &end);
            if (end != buf) return real(f);
        }
        return count ? a[0] : real(0.0);
    }
    if (str_is(name, "abs")) {
        if (v.kind == FLOAT) return real(fabs(v.f));
        if (numeric(&v)) return integer(as_int(&v) < 0 ? -as_int(&v) : as_int(&v));
        fail(r, "bad operand type for abs(): '%s'", kind_name(&v));
    }
    if (str_is(name, "replace")) {
        Value s = to_str(r, &v);
        need(r, count, 2, 3, "replace");
        Value old = to_str(r, &a[0]), with = to_str(r, &a[1]);
        return replace(r, &s, &old, &with, count > 2 ? as_int(&a[2]) : -1);
    }
    if (str_is(name, "safe") || str_is(name, "e") || str_is(name, "escape")) {
        if (str_is(name, "safe")) return v;
        Value s = to_str(r, &v);
        Text t = {r, NULL, 0, 0};
        for (size_t k = 0; k < s.s.n; k++) {
            const char c = s.s.p[k];
            if (c == '&') text_cstr(&t, "&amp;");
            else if (c == '<') text_cstr(&t, "&lt;");
            else if (c == '>') text_cstr(&t, "&gt;");
            else if (c == '"') text_cstr(&t, "&#34;");
            else if (c == '\'') text_cstr(&t, "&#39;");
            else text_add(&t, &c, 1);
        }
        return text_value(&t);
    }
    if (str_is(name, "indent")) {
        Value s = to_str(r, &v);
        int64_t width = count ? as_int(&a[0]) : 4;
        const Value *first_kw = keyword(n, a, "first");
        bool first = count > 1 ? truthy(&a[1]) : first_kw ? truthy(first_kw) : false;
        Text t = {r, NULL, 0, 0};
        bool line_start = first;
        for (size_t k = 0; k < s.s.n; k++) {
            if (line_start && s.s.p[k] != '\n')
                for (int64_t w = 0; w < width; w++) text_cstr(&t, " ");
            line_start = false;
            text_add(&t, s.s.p + k, 1);
            if (s.s.p[k] == '\n') line_start = true;
        }
        return text_value(&t);
    }
    if (str_is(name, "unique")) {
        Value items = iterable(r, &v), out = new_list(r, items.l->count);
        for (uint32_t k = 0; k < items.l->count; k++) {
            bool seen = false;
            for (uint32_t j = 0; j < out.l->count && !seen; j++) seen = equal(&out.l->items[j], &items.l->items[k]);
            if (!seen) list_push(r, out.l, items.l->items[k]);
        }
        return out;
    }
    if (str_is(name, "sum")) {
        Value items = iterable(r, &v), total = integer(0);
        const Value *attr = keyword(n, a, "attribute");
        for (uint32_t k = 0; k < items.l->count; k++) {
            Value e = attr ? attribute_path(r, &items.l->items[k], attr) : items.l->items[k];
            if (!numeric(&e)) fail(r, "unsupported operand type(s) for +: '%s' and '%s'", kind_name(&total), kind_name(&e));
            total = total.kind == FLOAT || e.kind == FLOAT ? real(as_double(&total) + as_double(&e))
                                                           : integer(as_int(&total) + as_int(&e));
        }
        return total;
    }
    if (str_is(name, "map")) {
        Value items = iterable(r, &v), out = new_list(r, items.l->count);
        const Value *attr = keyword(n, a, "attribute"), *def = keyword(n, a, "default");
        for (uint32_t k = 0; k < items.l->count; k++) {
            Value e;
            if (attr) {
                e = attribute_path(r, &items.l->items[k], attr);
                if (e.kind == UNDEF && def) e = *def;
            } else {
                if (!count || a[0].kind != STR) fail(r, "map() takes a filter's name or attribute=");
                Node fake = *n;
                fake.name = str_of(a[0].s.p, a[0].s.n);
                fake.count = count - 1;
                fake.name_count = 0;
                e = filter(r, &fake, items.l->items[k], a + 1, count - 1);
            }
            list_push(r, out.l, e);
        }
        return out;
    }
    if (str_is(name, "select") || str_is(name, "reject") || str_is(name, "selectattr") || str_is(name, "rejectattr")) {
        const bool attr = name.n > 6, keep = name.p[0] == 's';
        Value items = iterable(r, &v), out = new_list(r, items.l->count);
        if (attr && !count) fail(r, "%.*s() takes an attribute", (int)name.n, name.p);
        for (uint32_t k = 0; k < items.l->count; k++) {
            Value e = attr ? attribute_path(r, &items.l->items[k], &a[0]) : items.l->items[k];
            const uint32_t at = attr ? 1u : 0u;
            bool pass;
            if (count > at) {
                if (a[at].kind != STR) fail(r, "a test's name is a string");
                pass = test(r, str_of(a[at].s.p, a[at].s.n), &e, a + at + 1, count - at - 1);
            } else {
                pass = truthy(&e);
            }
            if (pass == keep) list_push(r, out.l, items.l->items[k]);
        }
        return out;
    }
    if (str_is(name, "dictsort")) {
        if (v.kind != DICT) fail(r, "dictsort takes a dict");
        Value pairs = call_method(r, M_ITEMS, &v, NULL, NULL, 0);
        for (uint32_t k = 1; k < pairs.l->count; k++)
            for (uint32_t j = k; j > 0 && order(r, &pairs.l->items[j - 1].l->items[0], &pairs.l->items[j].l->items[0]) > 0; j--) {
                Value s = pairs.l->items[j];
                pairs.l->items[j] = pairs.l->items[j - 1];
                pairs.l->items[j - 1] = s;
            }
        return pairs;
    }
    if (str_is(name, "sort")) {
        Value items = iterable(r, &v), out = new_list(r, items.l->count);
        const Value *attr = keyword(n, a, "attribute"), *rev = keyword(n, a, "reverse");
        for (uint32_t k = 0; k < items.l->count; k++) list_push(r, out.l, items.l->items[k]);
        const bool reverse = (count && truthy(&a[0])) || (rev && truthy(rev));
        for (uint32_t k = 1; k < out.l->count; k++)
            for (uint32_t j = k; j > 0; j--) {
                Value x = attr ? attribute_path(r, &out.l->items[j - 1], attr) : out.l->items[j - 1];
                Value y = attr ? attribute_path(r, &out.l->items[j], attr) : out.l->items[j];
                int c = order(r, &x, &y);
                if (reverse ? c >= 0 : c <= 0) break;
                Value s = out.l->items[j];
                out.l->items[j] = out.l->items[j - 1];
                out.l->items[j - 1] = s;
            }
        return out;
    }
    if (str_is(name, "batch") || str_is(name, "min") || str_is(name, "max") || str_is(name, "round")) {
        if (str_is(name, "round")) {
            double f = as_double(&v), digits = count ? (double)as_int(&a[0]) : 0.0, scale = pow(10.0, digits);
            return real(round(f * scale) / scale);
        }
        Value items = iterable(r, &v);
        if (str_is(name, "batch")) fail(r, "the filter batch is not supported");
        if (!items.l->count) return undef();
        Value best = items.l->items[0];
        for (uint32_t k = 1; k < items.l->count; k++) {
            int c = order(r, &items.l->items[k], &best);
            if (str_is(name, "min") ? c < 0 : c > 0) best = items.l->items[k];
        }
        return best;
    }
    if (str_is(name, "wordcount")) {
        Value s = to_str(r, &v), words = split(r, &s, NULL, -1, false);
        return integer(words.l->count);
    }
    fail(r, "no filter named '%.*s'", (int)name.n, name.p);
}

static bool test(Render *r, Str name, const Value *v, const Value *a, uint32_t count) {
    if (str_is(name, "defined")) return v->kind != UNDEF;
    if (str_is(name, "undefined")) return v->kind == UNDEF;
    if (str_is(name, "none")) return v->kind == NONE;
    if (str_is(name, "boolean")) return v->kind == BOOL;
    if (str_is(name, "true")) return v->kind == BOOL && v->b;
    if (str_is(name, "false")) return v->kind == BOOL && !v->b;
    if (str_is(name, "integer")) return v->kind == INT;
    if (str_is(name, "float")) return v->kind == FLOAT;
    if (str_is(name, "number")) return numeric(v);
    if (str_is(name, "string")) return v->kind == STR;
    if (str_is(name, "mapping")) return v->kind == DICT;
    if (str_is(name, "iterable")) return v->kind == LIST || v->kind == DICT || v->kind == STR;
    if (str_is(name, "sequence")) return v->kind == LIST || v->kind == DICT || v->kind == STR;
    if (str_is(name, "callable")) return v->kind == FUNC || v->kind == MACRO;
    if (str_is(name, "lower") || str_is(name, "upper")) {
        if (v->kind != STR) return false;
        Value c = change_case(r, v, str_is(name, "lower") ? 2 : 1);
        return equal(&c, v);
    }
    if (str_is(name, "even") || str_is(name, "odd")) {
        if (!numeric(v)) fail(r, "the test %.*s takes a number", (int)name.n, name.p);
        int64_t i = as_int(v);
        return ((i % 2 + 2) % 2 == 0) == str_is(name, "even");
    }
    if (str_is(name, "divisibleby")) {
        need(r, count, 1, 1, "divisibleby");
        int64_t d = as_int(&a[0]);
        if (!d) fail(r, "division by zero");
        return as_int(v) % d == 0;
    }
    if (count == 1) {
        if (str_is(name, "eq") || str_is(name, "equalto") || str_is(name, "==")) return equal(v, &a[0]);
        if (str_is(name, "ne") || str_is(name, "!=")) return !equal(v, &a[0]);
        if (str_is(name, "sameas")) return equal(v, &a[0]) && v->kind == a[0].kind;
        if (str_is(name, "lt") || str_is(name, "lessthan")) return order(r, v, &a[0]) < 0;
        if (str_is(name, "le")) return order(r, v, &a[0]) <= 0;
        if (str_is(name, "gt") || str_is(name, "greaterthan")) return order(r, v, &a[0]) > 0;
        if (str_is(name, "ge")) return order(r, v, &a[0]) >= 0;
        if (str_is(name, "in")) {
            Value n = a[0];
            Node fake;
            memset(&fake, 0, sizeof fake);
            (void)fake;
            if (n.kind == STR && v->kind == STR) return find(&n, v, false) >= 0;
            Value items = iterable(r, &n);
            for (uint32_t k = 0; k < items.l->count; k++)
                if (equal(&items.l->items[k], v)) return true;
            return false;
        }
    }
    fail(r, "no test named '%.*s'", (int)name.n, name.p);
}

/* ------------------------------------------------------------------------- expressions */

static Value arithmetic(Render *r, int op, const Value *a, const Value *b) {
    if (op == OP_ADD) {
        if (a->kind == STR && b->kind == STR) {
            Text t = {r, NULL, 0, 0};
            text_add(&t, a->s.p, a->s.n);
            text_add(&t, b->s.p, b->s.n);
            return text_value(&t);
        }
        if (a->kind == LIST && b->kind == LIST) {
            Value out = new_list(r, a->l->count + b->l->count);
            for (uint32_t k = 0; k < a->l->count; k++) list_push(r, out.l, a->l->items[k]);
            for (uint32_t k = 0; k < b->l->count; k++) list_push(r, out.l, b->l->items[k]);
            return out;
        }
    }
    if (op == OP_MUL && ((a->kind == STR && (b->kind == INT || b->kind == BOOL)) ||
                         (b->kind == STR && (a->kind == INT || a->kind == BOOL)))) {
        const Value *s = a->kind == STR ? a : b;
        int64_t times = as_int(a->kind == STR ? b : a);
        Text t = {r, NULL, 0, 0};
        for (int64_t k = 0; k < times && k < (1 << 20); k++) text_add(&t, s->s.p, s->s.n);
        return text_value(&t);
    }
    if (!numeric(a) || !numeric(b)) {
        static const char *const names[] = {"", "+", "-", "*", "/", "//", "%", "**"};
        if (op == OP_ADD && (a->kind == STR || b->kind == STR))
            fail(r, "can only concatenate %s (not \"%s\") to %s", kind_name(a->kind == STR ? a : b),
                 kind_name(a->kind == STR ? b : a), kind_name(a->kind == STR ? a : b));
        fail(r, "unsupported operand type(s) for %s: '%s' and '%s'", names[op], kind_name(a), kind_name(b));
    }
    const bool floats = a->kind == FLOAT || b->kind == FLOAT;
    if (op == OP_DIV) {
        if (as_double(b) == 0.0) fail(r, "division by zero");
        return real(as_double(a) / as_double(b));
    }
    if (floats) {
        double x = as_double(a), y = as_double(b);
        switch (op) {
            case OP_ADD: return real(x + y);
            case OP_SUB: return real(x - y);
            case OP_MUL: return real(x * y);
            case OP_FLOORDIV: if (y == 0.0) fail(r, "float floor division by zero"); return real(floor(x / y));
            case OP_MOD: {
                if (y == 0.0) fail(r, "float modulo");
                double m = fmod(x, y);
                if (m != 0.0 && (m < 0) != (y < 0)) m += y;
                return real(m);
            }
            default: return real(pow(x, y));
        }
    }
    int64_t x = as_int(a), y = as_int(b);
    switch (op) {
        case OP_ADD: return integer(x + y);
        case OP_SUB: return integer(x - y);
        case OP_MUL: return integer(x * y);
        case OP_FLOORDIV: case OP_MOD: {
            if (!y) fail(r, "integer division or modulo by zero");
            int64_t q = x / y, m = x % y;
            if (m != 0 && ((m < 0) != (y < 0))) { q--; m += y; }
            return integer(op == OP_MOD ? m : q);
        }
        default: {
            if (y < 0) return real(pow((double)x, (double)y));
            int64_t p = 1;
            for (int64_t k = 0; k < y; k++) p *= x;
            return integer(p);
        }
    }
}

static bool contains(Render *r, const Value *container, const Value *v) {
    if (container->kind == STR) {
        if (v->kind != STR) fail(r, "'in <string>' requires string as left operand, not %s", kind_name(v));
        return find(container, v, false) >= 0;
    }
    if (container->kind == DICT) return dict_find_value(container->d, v) != NULL;
    if (container->kind == LIST) {
        for (uint32_t k = 0; k < container->l->count; k++)
            if (equal(&container->l->items[k], v)) return true;
        return false;
    }
    if (container->kind == UNDEF) return false;
    fail(r, "argument of type '%s' is not iterable", kind_name(container));
}

static Value eval(Render *r, const Node *n) {
    switch (n->type) {
        case E_LITERAL: return n->value;
        case E_NAME: return lookup(r, n->name);
        case E_LIST: {
            Value out = new_list(r, n->count);
            for (uint32_t k = 0; k < n->count; k++) list_push(r, out.l, eval(r, n->list[k]));
            return out;
        }
        case E_DICT: {
            Value out = new_dict(r);
            for (uint32_t k = 0; k + 1 < n->count; k += 2) dict_set(r, out.d, eval(r, n->list[k]), eval(r, n->list[k + 1]));
            return out;
        }
        case E_ATTR: {
            Value obj = eval(r, n->a);
            return attribute(r, &obj, n->name);
        }
        case E_ITEM: {
            Value obj = eval(r, n->a), index = eval(r, n->b);
            return item(r, &obj, &index);
        }
        case E_SLICE: return slice(r, n);
        case E_CALL: case E_FILTER: {
            const uint32_t all = n->count + n->name_count;
            Value *args = (Value *)alloc(r, (all + 1u) * sizeof(Value));
            for (uint32_t k = 0; k < all; k++) args[k] = eval(r, n->list[k]);
            if (n->type == E_FILTER) return filter(r, n, eval(r, n->a), args, n->count);
            Value f = eval(r, n->a);
            if (f.kind == UNDEF) {
                if (n->a->type == E_NAME) fail(r, "'%.*s' is undefined", (int)n->a->name.n, n->a->name.p);
                fail(r, "a call of an undefined value");
            }
            return call_function(r, &f, n, args, n->count);
        }
        case E_TEST: {
            Value v = eval(r, n->a);
            Value *args = (Value *)alloc(r, (n->count + 1u) * sizeof(Value));
            for (uint32_t k = 0; k < n->count; k++) args[k] = eval(r, n->list[k]);
            return boolean(test(r, n->name, &v, args, n->count) != (bool)n->op);
        }
        case E_NEG: case E_POS: {
            Value v = eval(r, n->a);
            if (!numeric(&v)) fail(r, "bad operand type for unary %c: '%s'", n->type == E_NEG ? '-' : '+', kind_name(&v));
            if (n->type == E_POS) return v.kind == BOOL ? integer(v.b) : v;
            return v.kind == FLOAT ? real(-v.f) : integer(-as_int(&v));
        }
        case E_NOT: {
            Value v = eval(r, n->a);
            return boolean(!truthy(&v));
        }
        case E_AND: {
            Value a = eval(r, n->a);
            return truthy(&a) ? eval(r, n->b) : a;
        }
        case E_OR: {
            Value a = eval(r, n->a);
            return truthy(&a) ? a : eval(r, n->b);
        }
        case E_COND: {
            Value c = eval(r, n->a);
            if (truthy(&c)) return eval(r, n->b);
            return n->c ? eval(r, n->c) : undef();
        }
        case E_CONCAT: {
            Text t = {r, NULL, 0, 0};
            for (uint32_t k = 0; k < n->count; k++) {
                Value v = eval(r, n->list[k]);
                str_into(r, &t, &v);
            }
            return text_value(&t);
        }
        case E_BINARY: {
            Value a = eval(r, n->a), b = eval(r, n->b);
            return arithmetic(r, n->op, &a, &b);
        }
        case E_COMPARE: {
            Value left = eval(r, n->list[0]);
            for (uint32_t k = 0; k < n->name_count; k++) {
                Value right = eval(r, n->list[k + 1]);
                const int op = (int)n->names[k].n;
                bool ok;
                switch (op) {
                    case OP_EQ: ok = equal(&left, &right); break;
                    case OP_NE: ok = !equal(&left, &right); break;
                    case OP_LT: ok = order(r, &left, &right) < 0; break;
                    case OP_LE: ok = order(r, &left, &right) <= 0; break;
                    case OP_GT: ok = order(r, &left, &right) > 0; break;
                    case OP_GE: ok = order(r, &left, &right) >= 0; break;
                    case OP_IN: ok = contains(r, &right, &left); break;
                    default: ok = !contains(r, &right, &left); break;
                }
                if (!ok) return boolean(false);
                left = right;
            }
            return boolean(true);
        }
        default:
            fail(r, "a statement where an expression belongs");
    }
}

/* ------------------------------------------------------------------------- statements */

static void emit(Render *r, const char *p, size_t n) {
    if (r->out_n + n + 1 > r->out_cap) {
        size_t cap = r->out_cap ? r->out_cap : 1024u;
        while (cap < r->out_n + n + 1) cap *= 2u;
        char *grown = (char *)realloc(r->out, cap);
        if (!grown) fail(r, "out of memory");
        r->out = grown;
        r->out_cap = cap;
    }
    memcpy(r->out + r->out_n, p, n);
    r->out_n += n;
    r->out[r->out_n] = '\0';
}

static void assign(Render *r, const Str *names, uint32_t count, Value v) {
    if (count == 1) {
        set_var(r, names[0], v);
        return;
    }
    Value items = iterable(r, &v);
    if (items.l->count != count) fail(r, "cannot unpack %u values into %u names", items.l->count, count);
    for (uint32_t k = 0; k < count; k++) set_var(r, names[k], items.l->items[k]);
}

static void exec(Render *r, const Node *n) {
    switch (n->type) {
        case N_BODY:
            for (uint32_t k = 0; k < n->count && !r->jump; k++) exec(r, n->list[k]);
            return;
        case N_TEXT:
            emit(r, n->name.p, n->name.n);
            return;
        case N_PRINT: {
            Value v = eval(r, n->a);
            if (n->op) return;
            if (v.kind == STR) emit(r, v.s.p, v.s.n);
            else {
                Text t = {r, NULL, 0, 0};
                str_into(r, &t, &v);
                emit(r, t.p ? t.p : "", t.n);
            }
            return;
        }
        case N_IF: {
            Value c = eval(r, n->a);
            if (truthy(&c)) exec(r, n->b);
            else if (n->c) exec(r, n->c);
            return;
        }
        case N_FOR: {
            Value source = eval(r, n->a), all = iterable(r, &source);
            /* the items that pass the loop's condition, which sees each item as the loop variables */
            Frame frame = {r->frame, {0}};
            r->frame = &frame;
            Value items = all;
            if (n->d) {
                items = new_list(r, all.l->count);
                for (uint32_t k = 0; k < all.l->count; k++) {
                    assign(r, n->names, n->name_count, all.l->items[k]);
                    Value c = eval(r, n->d);
                    if (truthy(&c)) list_push(r, items.l, all.l->items[k]);
                }
            }
            const uint32_t count = items.l->count;
            Value loop = new_dict(r);
            for (uint32_t k = 0; k < count; k++) {
                loop.d->count = 0;
                dict_set(r, loop.d, cstring("index"), integer(k + 1));
                dict_set(r, loop.d, cstring("index0"), integer(k));
                dict_set(r, loop.d, cstring("revindex"), integer(count - k));
                dict_set(r, loop.d, cstring("revindex0"), integer(count - k - 1));
                dict_set(r, loop.d, cstring("first"), boolean(k == 0));
                dict_set(r, loop.d, cstring("last"), boolean(k + 1 == count));
                dict_set(r, loop.d, cstring("length"), integer(count));
                dict_set(r, loop.d, cstring("previtem"), k ? items.l->items[k - 1] : undef());
                dict_set(r, loop.d, cstring("nextitem"), k + 1 < count ? items.l->items[k + 1] : undef());
                set_var(r, str_of("loop", 4), loop);
                assign(r, n->names, n->name_count, items.l->items[k]);
                exec(r, n->b);
                if (r->jump == 1) {
                    r->jump = 0;
                    break;
                }
                r->jump = 0;
            }
            r->frame = frame.parent;
            if (!count && n->c) exec(r, n->c);
            return;
        }
        case N_SET: {
            Value v = eval(r, n->b);
            if (n->a) {
                Value target = eval(r, n->a);
                if (target.kind != DICT || !target.d->ns)
                    fail(r, "cannot assign attribute on a %s (only on a namespace)", kind_name(&target));
                dict_set(r, target.d, copy_string(r, n->name.p, n->name.n), v);
            } else if (n->names) {
                assign(r, n->names, n->name_count, v);
            } else {
                set_var(r, n->name, v);
            }
            return;
        }
        case N_SET_BLOCK: {
            char *out = r->out;
            size_t out_n = r->out_n, out_cap = r->out_cap;
            r->out = NULL;
            r->out_n = r->out_cap = 0;
            exec(r, n->b);
            Value text = r->out ? copy_string(r, r->out, r->out_n) : string("", 0);
            free(r->out);
            r->out = out;
            r->out_n = out_n;
            r->out_cap = out_cap;
            set_var(r, n->name, text);
            return;
        }
        case N_MACRO: {
            Value m = undef();
            m.kind = MACRO;
            m.macro = n;
            set_var(r, n->name, m);
            return;
        }
        case N_BREAK: r->jump = 1; return;
        case N_CONTINUE: r->jump = 2; return;
        default: {
            Value v = eval(r, n);
            (void)v;
            return;
        }
    }
}

/* ------------------------------------------------------------------------- values of JSON and of calls */

static Value from_json(Render *r, const SpgJson *j, uint32_t v) {
    const SpgJsonValue *x = &j->values[v];
    switch (x->type) {
        case SPG_JSON_NULL: return none();
        case SPG_JSON_BOOL: return boolean(x->count != 0);
        case SPG_JSON_NUMBER: {
            bool fraction = false;
            for (uint32_t k = 0; k < x->length; k++) {
                const char c = j->text[x->start + k];
                fraction |= c == '.' || c == 'e' || c == 'E';
            }
            int64_t i;
            if (!fraction && spingalett_json_integer(j, v, INT64_MIN, INT64_MAX, &i)) return integer(i);
            double f = 0.0;
            spingalett_json_number(j, v, &f);
            return real(f);
        }
        case SPG_JSON_STRING: {
            char *s = (char *)alloc(r, x->count + 1u);
            spingalett_json_copy(j, v, s);
            return string(s, x->count);
        }
        case SPG_JSON_ARRAY: {
            Value out = new_list(r, x->count);
            for (uint32_t e = spingalett_json_first(j, v); e && e < x->next; e = j->values[e].next)
                list_push(r, out.l, from_json(r, j, e));
            return out;
        }
        default: {
            Value out = new_dict(r);
            for (uint32_t e = spingalett_json_first(j, v); e && e < x->next; e = j->values[e + 1].next)
                dict_set(r, out.d, from_json(r, j, e), from_json(r, j, e + 1));
            return out;
        }
    }
}

int64_t spingalett_chat_text(const SpgChatRender *c, char **text) {
    Render *r = (Render *)calloc(1, sizeof(Render));
    *text = NULL;
    if (!r) {
        set_error(SPINGALETT_ERR_ALLOC, "chat template: out of memory");
        return -1;
    }
    SpgJson json;
    memset(&json, 0, sizeof json);
    bool parsed = false;
    int64_t result = -1;
    if (setjmp(r->on_error)) {
        char message[300];
        snprintf(message, sizeof message, "chat template: %s", r->error);
        set_error(SPINGALETT_ERR_INVALID, message);
        free(r->out);
        r->out = NULL;
        goto done;
    }
    r->now = c->time ? (time_t)c->time : time(NULL);
    lex(r, c->source, strlen(c->source));
    Node *tree = body(r, NULL, NULL);
    Frame root = {NULL, {0}};
    r->root = r->frame = &root;
    /* the variables: transformers' defaults, the special tokens, the messages, then the JSON object's */
    set_var(r, str_of("tools", 5), none());
    set_var(r, str_of("documents", 9), none());
    set_var(r, str_of("add_generation_prompt", 21), boolean(c->add_generation_prompt));
    if (c->bos_token) set_var(r, str_of("bos_token", 9), copy_string(r, c->bos_token, strlen(c->bos_token)));
    if (c->eos_token) set_var(r, str_of("eos_token", 9), copy_string(r, c->eos_token, strlen(c->eos_token)));
    Value messages = new_list(r, c->message_count);
    for (uint32_t k = 0; k < c->message_count; k++) {
        Value m = new_dict(r);
        dict_set(r, m.d, cstring("role"), c->roles[k] ? copy_string(r, c->roles[k], strlen(c->roles[k])) : none());
        dict_set(r, m.d, cstring("content"), c->contents[k] ? copy_string(r, c->contents[k], strlen(c->contents[k])) : none());
        list_push(r, messages.l, m);
    }
    set_var(r, str_of("messages", 8), messages);
    if (c->variables && c->variables[0]) {
        if (!spingalett_json_parse(&json, c->variables, strlen(c->variables)))
            fail(r, "variables: %s", json.error);
        parsed = true;
        if (json.values[SPG_JSON_ROOT].type != SPG_JSON_OBJECT) fail(r, "variables: not a JSON object");
        Value vars = from_json(r, &json, SPG_JSON_ROOT);
        for (uint32_t k = 0; k < vars.d->count; k++) set_var(r, str_of(vars.d->keys[k].s.p, vars.d->keys[k].s.n), vars.d->values[k]);
    }
    exec(r, tree);
    result = (int64_t)r->out_n;
    *text = r->out ? r->out : (char *)calloc(1, 1);
    if (!*text) {
        set_error(SPINGALETT_ERR_ALLOC, "chat template: out of memory");
        result = -1;
    }
    r->out = NULL;
done:
    if (parsed) spingalett_json_free(&json);
    while (r->blocks) {
        Block *next = r->blocks->next;
        free(r->blocks);
        r->blocks = next;
    }
    free(r);
    return result;
}
