/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/*
 * PyTorch weights: state dicts written by torch.save (a zip archive whose data.pkl pickles a
 * dictionary of tensors, their bytes in data/<key>) and safetensors files (a JSON header and the
 * bytes), copied into a network of the same architecture.
 *
 * The pickle is read by an interpreter of the opcodes torch.save writes that builds data only: it
 * calls nothing, imports nothing and knows the few callables of a state dict (OrderedDict,
 * torch._utils._rebuild_tensor_v2 and _rebuild_parameter), treating any other as an opaque value,
 * so a file can describe tensors but cannot run code. The zip archive must store its entries
 * uncompressed, as torch.save does.
 *
 * Tensors are grouped by module (the name up to its last dot) and the modules assigned, in order,
 * to the layers with parameters: dense layers take weight [out, in] and bias, convolutions weight
 * [out, in / groups, kh, kw] and bias, batch normalizations weight, bias, running_mean and
 * running_var; num_batches_tracked is ignored. Filters are reordered to channels-last and the
 * columns of a dense layer that reads a map from (c, h, w) to (h, w, c), as ONNX import does.
 */

#include "Spingalett.Private.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

typedef struct { const char *s; size_t n; } Name;

static bool name_is(Name a, const char *b) { return a.n == strlen(b) && (a.n == 0 || memcmp(a.s, b, a.n) == 0); }
static bool name_eq(Name a, Name b) { return a.n == b.n && (a.n == 0 || memcmp(a.s, b.s, a.n) == 0); }

/* A tensor of the file: its name, type, shape and strides (in elements), and its bytes. */
typedef struct {
    Name name;
    int dtype;
    int ndims;
    int64_t dims[8], strides[8];
    const uint8_t *data;        /* the storage */
    uint64_t offset;            /* first element within it */
    uint64_t bytes;             /* bytes of the storage */
} Weight;

typedef struct {
    Weight *items;
    size_t count, cap;
    char error[256];
} Weights;

static bool werr(Weights *w, const char *msg) {
    if (!w->error[0]) snprintf(w->error, sizeof w->error, "PyTorch weights: %s", msg);
    return false;
}

static Weight *new_weight(Weights *w) {
    if (w->count == w->cap) {
        size_t cap = w->cap ? 2 * w->cap : 32;
        Weight *p = (Weight *)realloc(w->items, cap * sizeof(Weight));
        if (!p) return NULL;
        w->items = p;
        w->cap = cap;
    }
    Weight *t = &w->items[w->count++];
    memset(t, 0, sizeof *t);
    return t;
}

/* Elements of the shape; UINT64_MAX when the product overflows. */
static uint64_t elements(const Weight *t) {
    uint64_t n = 1;
    for (int d = 0; d < t->ndims && d < 8; d++) {
        uint64_t e = t->dims[d] < 0 ? 0u : (uint64_t)t->dims[d];
        if (e && n > UINT64_MAX / e) return UINT64_MAX;
        n *= e;
    }
    return n;
}

/* Element i (in row-major order of the shape) as a float; false when it lies outside the storage. */
static bool element(const Weight *t, uint64_t i, float *v) {
    uint64_t at = t->offset;
    for (int d = t->ndims - 1; d >= 0; d--) {
        uint64_t extent = t->dims[d] > 0 ? (uint64_t)t->dims[d] : 1u;
        at += (i % extent) * (uint64_t)t->strides[d];
        i /= extent;
    }
    size_t size = spingalett_dtype_size(t->dtype);
    if (size == 0 || at >= t->bytes / size) return false;
    spingalett_decode(v, t->data + at * size, t->dtype, 1);
    return true;
}

static void contiguous_strides(Weight *t) {
    int64_t s = 1;
    for (int d = t->ndims - 1; d >= 0; d--) {
        t->strides[d] = s;
        s *= t->dims[d] > 0 ? t->dims[d] : 1;
    }
}

/* ------------------------------------------------------------------------- safetensors */

typedef struct {
    const char *p, *end;
    bool ok;
} Json;

static void json_space(Json *j) {
    while (j->p < j->end && (*j->p == ' ' || *j->p == '\t' || *j->p == '\n' || *j->p == '\r')) j->p++;
}

static bool json_char(Json *j, char c) {
    json_space(j);
    if (j->p < j->end && *j->p == c) { j->p++; return true; }
    return false;
}

/* A string without escapes needing translation (names of tensors and types); escapes are skipped. */
static Name json_string(Json *j) {
    Name n = {NULL, 0};
    if (!json_char(j, '"')) { j->ok = false; return n; }
    n.s = j->p;
    while (j->p < j->end && *j->p != '"') {
        if (*j->p == '\\') j->p++;
        j->p++;
    }
    if (j->p >= j->end) { j->ok = false; return n; }
    n.n = (size_t)(j->p - n.s);
    j->p++;
    return n;
}

static int64_t json_int(Json *j) {
    json_space(j);
    bool neg = j->p < j->end && *j->p == '-';
    if (neg) j->p++;
    if (j->p >= j->end || *j->p < '0' || *j->p > '9') { j->ok = false; return 0; }
    int64_t v = 0;
    while (j->p < j->end && *j->p >= '0' && *j->p <= '9') {
        if (v > (INT64_MAX - 9) / 10) { j->ok = false; return 0; }
        v = v * 10 + (*j->p++ - '0');
    }
    return neg ? -v : v;
}

/* Skips any value (the metadata). */
static void json_skip(Json *j, int depth) {
    json_space(j);
    if (!j->ok || j->p >= j->end || depth > 32) { j->ok = false; return; }
    char c = *j->p;
    if (c == '"') { (void)json_string(j); return; }
    if (c == '{' || c == '[') {
        char close = c == '{' ? '}' : ']';
        j->p++;
        if (json_char(j, close)) return;
        do {
            if (c == '{') { (void)json_string(j); if (!json_char(j, ':')) { j->ok = false; return; } }
            json_skip(j, depth + 1);
        } while (j->ok && json_char(j, ','));
        if (!json_char(j, close)) j->ok = false;
        return;
    }
    while (j->p < j->end && *j->p != ',' && *j->p != '}' && *j->p != ']') j->p++;
}

static bool read_safetensors(const uint8_t *data, size_t size, Weights *w) {
    if (size < 8) return werr(w, "the file is truncated");
    uint64_t header = slett_get64(data);
    if (header > size - 8) return werr(w, "the safetensors header exceeds the file");
    const uint8_t *body = data + 8 + header;
    const uint64_t body_size = size - 8 - header;
    Json j = {(const char *)data + 8, (const char *)data + 8 + header, true};
    if (!json_char(&j, '{')) return werr(w, "not a safetensors file");
    if (json_char(&j, '}')) return true;
    do {
        Name name = json_string(&j);
        if (!j.ok || !json_char(&j, ':')) break;
        if (name_is(name, "__metadata__")) { json_skip(&j, 0); continue; }
        Weight *t = new_weight(w);
        if (!t) return werr(w, "out of memory");
        t->name = name;
        t->dtype = SPG_DTYPE_OTHER;
        uint64_t begin = 0, end = 0;
        bool have_offsets = false;
        if (!json_char(&j, '{')) { j.ok = false; break; }
        do {
            Name key = json_string(&j);
            if (!j.ok || !json_char(&j, ':')) { j.ok = false; break; }
            if (name_is(key, "dtype")) {
                Name v = json_string(&j);
                t->dtype = name_is(v, "F32") ? SPG_DTYPE_F32 : name_is(v, "F16") ? SPG_DTYPE_F16
                         : name_is(v, "BF16") ? SPG_DTYPE_BF16 : name_is(v, "F64") ? SPG_DTYPE_F64
                         : name_is(v, "I32") ? SPG_DTYPE_I32 : name_is(v, "I64") ? SPG_DTYPE_I64 : SPG_DTYPE_OTHER;
            } else if (name_is(key, "shape")) {
                if (!json_char(&j, '[')) { j.ok = false; break; }
                if (!json_char(&j, ']')) {
                    do {
                        int64_t v = json_int(&j);
                        if (t->ndims < 8) t->dims[t->ndims] = v;
                        t->ndims++;
                    } while (j.ok && json_char(&j, ','));
                    if (!json_char(&j, ']')) j.ok = false;
                }
            } else if (name_is(key, "data_offsets")) {
                if (!json_char(&j, '[')) { j.ok = false; break; }
                begin = (uint64_t)json_int(&j);
                if (!json_char(&j, ',')) { j.ok = false; break; }
                end = (uint64_t)json_int(&j);
                if (!json_char(&j, ']')) j.ok = false;
                have_offsets = true;
            } else {
                json_skip(&j, 0);
            }
        } while (j.ok && json_char(&j, ','));
        if (!j.ok || !json_char(&j, '}')) { j.ok = false; break; }
        if (t->ndims > 8 || !have_offsets || begin > end || end > body_size)
            return werr(w, "a tensor's shape or offsets are invalid");
        t->data = body + begin;
        t->bytes = end - begin;
        contiguous_strides(t);
        if (spingalett_dtype_size(t->dtype) && elements(t) * spingalett_dtype_size(t->dtype) != t->bytes)
            return werr(w, "a tensor's size does not match its shape");
    } while (j.ok && json_char(&j, ','));
    if (!j.ok || !json_char(&j, '}')) return werr(w, "the safetensors header cannot be read");
    return true;
}

/* ------------------------------------------------------------------------- zip archives */

typedef struct {
    Name name;
    const uint8_t *data;
    uint64_t size;
} ZipEntry;

/* The entries of a zip archive whose members are stored (uncompressed). */
static bool read_zip(const uint8_t *p, size_t size, ZipEntry **entries, size_t *count, Weights *w) {
    *entries = NULL;
    *count = 0;
    if (size < 22) return werr(w, "the archive is truncated");
    /* the end of central directory record, within the last 64 KB + 22 bytes */
    size_t eocd = SIZE_MAX;
    for (size_t at = size - 22 + 1; at-- > 0 && size - at <= 65557u;)
        if (slett_get32(p + at) == 0x06054b50u) { eocd = at; break; }
    if (eocd == SIZE_MAX) return werr(w, "not a zip archive (no central directory)");
    uint64_t n = slett_get16(p + eocd + 10), dir_size = slett_get32(p + eocd + 12), dir = slett_get32(p + eocd + 16);
    if (n == 0xFFFFu || dir == 0xFFFFFFFFu) {
        /* zip64: the locator before the record gives the zip64 record */
        if (eocd < 20 || slett_get32(p + eocd - 20) != 0x07064b50u) return werr(w, "the zip64 directory is missing");
        uint64_t rec = slett_get64(p + eocd - 20 + 8);
        if (rec > size - 56 || slett_get32(p + rec) != 0x06064b50u) return werr(w, "the zip64 directory is invalid");
        n = slett_get64(p + rec + 32);
        dir_size = slett_get64(p + rec + 40);
        dir = slett_get64(p + rec + 48);
    }
    if (dir > size || dir_size > size - dir || n > dir_size / 46u) return werr(w, "the zip directory is invalid");
    ZipEntry *e = (ZipEntry *)calloc(n ? (size_t)n : 1u, sizeof(ZipEntry));
    if (!e) return werr(w, "out of memory");
    uint64_t at = dir;
    for (uint64_t k = 0; k < n; k++) {
        if (at > size - 46 || slett_get32(p + at) != 0x02014b50u) { free(e); return werr(w, "a zip directory entry is invalid"); }
        uint16_t method = slett_get16(p + at + 10), name_len = slett_get16(p + at + 28), extra = slett_get16(p + at + 30),
                 comment = slett_get16(p + at + 32);
        uint64_t csize = slett_get32(p + at + 20), usize = slett_get32(p + at + 24), local = slett_get32(p + at + 42);
        if (at + 46 + name_len + extra + comment > size) { free(e); return werr(w, "a zip directory entry is truncated"); }
        /* zip64 sizes and offset in the extra field */
        const uint8_t *x = p + at + 46 + name_len, *xend = x + extra;
        while (xend - x >= 4) {
            uint16_t id = slett_get16(x), len = slett_get16(x + 2);
            if (id == 0x0001u && x + 4 + len <= xend) {
                const uint8_t *v = x + 4, *vend = x + 4 + len;
                if (usize == 0xFFFFFFFFu && vend - v >= 8) { usize = slett_get64(v); v += 8; }
                if (csize == 0xFFFFFFFFu && vend - v >= 8) { csize = slett_get64(v); v += 8; }
                if (local == 0xFFFFFFFFu && vend - v >= 8) { local = slett_get64(v); }
            }
            x += 4u + len;
        }
        e[k].name = (Name){(const char *)p + at + 46, name_len};
        if (local > size - 30 || slett_get32(p + local) != 0x04034b50u) { free(e); return werr(w, "a zip member is invalid"); }
        uint64_t start = local + 30u + slett_get16(p + local + 26) + slett_get16(p + local + 28);
        if (start > size || csize > size - start) { free(e); return werr(w, "a zip member exceeds the archive"); }
        if (method != 0 || csize != usize) {
            /* members that are not weights may be compressed; a compressed storage fails when used */
            e[k].data = NULL;
        } else {
            e[k].data = p + start;
        }
        e[k].size = csize;
        at += 46u + name_len + extra + comment;
    }
    *entries = e;
    *count = (size_t)n;
    return true;
}

/* ------------------------------------------------------------------------- pickle */

enum { OB_NONE, OB_BOOL, OB_INT, OB_FLOAT, OB_STR, OB_TUPLE, OB_LIST, OB_DICT, OB_GLOBAL, OB_MARK, OB_STORAGE,
       OB_TENSOR, OB_OPAQUE };

typedef struct {
    int kind;
    int64_t i;
    double f;
    Name s, s2;                 /* strings; globals: module and name; storages: their key */
    int dtype;                  /* storages and tensors */
    uint32_t *items, count, cap;        /* tuples, lists, dicts (keys and values alternate) */
    uint32_t storage;           /* tensors: the storage object, offset, shape and strides */
    int64_t offset, dims[8], strides[8];
    int ndims;
} Ob;

typedef struct {
    Ob *obs;
    uint32_t count, cap;
    uint32_t *stack, depth, stack_cap;
    uint32_t *memo, memo_cap, memo_size;        /* memo_size: the entries set (MEMOIZE adds one) */
    bool ok;
} Unpickler;

static uint32_t new_ob(Unpickler *u, int kind) {
    if (u->count == u->cap) {
        uint32_t cap = u->cap ? 2 * u->cap : 256;
        Ob *o = (Ob *)realloc(u->obs, cap * sizeof(Ob));
        if (!o) { u->ok = false; return 0; }
        u->obs = o;
        u->cap = cap;
    }
    Ob *o = &u->obs[u->count];
    memset(o, 0, sizeof *o);
    o->kind = kind;
    return u->count++;
}

static void push(Unpickler *u, uint32_t ob) {
    if (u->depth == u->stack_cap) {
        uint32_t cap = u->stack_cap ? 2 * u->stack_cap : 256;
        uint32_t *s = (uint32_t *)realloc(u->stack, cap * sizeof(uint32_t));
        if (!s) { u->ok = false; return; }
        u->stack = s;
        u->stack_cap = cap;
    }
    u->stack[u->depth++] = ob;
}

static uint32_t pop(Unpickler *u) {
    if (u->depth == 0) { u->ok = false; return 0; }
    return u->stack[--u->depth];
}

static void add_item(Unpickler *u, uint32_t container, uint32_t item) {
    Ob *c = &u->obs[container];
    if (c->count == c->cap) {
        uint32_t cap = c->cap ? 2 * c->cap : 8;
        uint32_t *it = (uint32_t *)realloc(c->items, cap * sizeof(uint32_t));
        if (!it) { u->ok = false; return; }
        c->items = it;
        c->cap = cap;
    }
    c->items[c->count++] = item;
}

/* The objects above the topmost mark, which is removed: their start in the stack. */
static uint32_t mark_start(Unpickler *u) {
    for (uint32_t k = u->depth; k > 0; k--)
        if (u->obs[u->stack[k - 1]].kind == OB_MARK) return k;
    u->ok = false;
    return u->depth + 1;
}

static uint32_t make_tuple(Unpickler *u, uint32_t from) {
    uint32_t t = new_ob(u, OB_TUPLE);
    for (uint32_t k = from; u->ok && k < u->depth; k++) add_item(u, t, u->stack[k]);
    return t;
}

static void memo_put(Unpickler *u, uint64_t index, uint32_t ob) {
    if (index > (1u << 26)) { u->ok = false; return; }
    if (index >= u->memo_cap) {
        uint32_t cap = u->memo_cap ? u->memo_cap : 64;
        while (cap <= index) cap *= 2;
        uint32_t *m = (uint32_t *)realloc(u->memo, cap * sizeof(uint32_t));
        if (!m) { u->ok = false; return; }
        for (uint32_t k = u->memo_cap; k < cap; k++) m[k] = UINT32_MAX;
        u->memo = m;
        u->memo_cap = cap;
    }
    if (u->memo[index] == UINT32_MAX) u->memo_size++;
    u->memo[index] = ob;
}

static uint32_t memo_get(Unpickler *u, uint64_t index) {
    if (index >= u->memo_cap || u->memo[index] == UINT32_MAX) { u->ok = false; return 0; }
    return u->memo[index];
}

static bool global_is(const Ob *o, const char *module, const char *name) {
    return o->kind == OB_GLOBAL && name_is(o->s, module) && name_is(o->s2, name);
}

static int storage_dtype(const Ob *type) {
    static const struct { const char *name; int dtype; } types[] = {
        {"FloatStorage", SPG_DTYPE_F32}, {"HalfStorage", SPG_DTYPE_F16}, {"BFloat16Storage", SPG_DTYPE_BF16},
        {"DoubleStorage", SPG_DTYPE_F64}, {"IntStorage", SPG_DTYPE_I32}, {"LongStorage", SPG_DTYPE_I64},
    };
    for (size_t k = 0; k < sizeof types / sizeof types[0]; k++)
        if (global_is(type, "torch", types[k].name)) return types[k].dtype;
    return SPG_DTYPE_OTHER;
}

/* A tuple of integers as a shape or strides. */
static bool int_tuple(const Unpickler *u, uint32_t t, int64_t *out, int *n) {
    const Ob *o = &u->obs[t];
    if (o->kind != OB_TUPLE || o->count > 8) return false;
    for (uint32_t k = 0; k < o->count; k++) {
        const Ob *v = &u->obs[o->items[k]];
        if (v->kind != OB_INT) return false;
        out[k] = v->i;
    }
    *n = (int)o->count;
    return true;
}

static void reduce(Unpickler *u) {
    uint32_t args = pop(u), callable = pop(u);
    if (!u->ok) return;
    const Ob *f = &u->obs[callable], *a = &u->obs[args];
    if (global_is(f, "collections", "OrderedDict")) {
        push(u, new_ob(u, OB_DICT));
        return;
    }
    if ((global_is(f, "torch._utils", "_rebuild_tensor_v2") || global_is(f, "torch._utils", "_rebuild_tensor")) &&
        a->kind == OB_TUPLE && a->count >= 4) {
        uint32_t storage = a->items[0], offset = a->items[1];
        uint32_t t = new_ob(u, OB_TENSOR);
        if (!u->ok) return;
        Ob *o = &u->obs[t];
        a = &u->obs[args];
        int nd = 0, ns = 0;
        if (u->obs[storage].kind != OB_STORAGE || u->obs[offset].kind != OB_INT ||
            !int_tuple(u, a->items[2], o->dims, &nd) || !int_tuple(u, a->items[3], o->strides, &ns) || nd != ns) {
            o->kind = OB_OPAQUE;
        } else {
            o->storage = storage;
            o->offset = u->obs[offset].i;
            o->ndims = nd;
            o->dtype = u->obs[storage].dtype;
        }
        push(u, t);
        return;
    }
    if ((global_is(f, "torch._utils", "_rebuild_parameter") ||
         global_is(f, "torch._utils", "_rebuild_parameter_with_state")) && a->kind == OB_TUPLE && a->count >= 1) {
        push(u, a->items[0]);
        return;
    }
    push(u, new_ob(u, OB_OPAQUE));         /* anything else: a value the weights do not need */
}

/* Reads a pickle; the result is the object STOP leaves. Protocols 2 to 5 as torch.save writes them. */
static uint32_t unpickle(Unpickler *u, const uint8_t *p, size_t size) {
    const uint8_t *end = p + size;
#define NEED(n) do { if ((size_t)(end - p) < (size_t)(n)) { u->ok = false; return 0; } } while (0)
    while (u->ok) {
        NEED(1);
        uint8_t op = *p++;
        switch (op) {
            case 0x80: NEED(1); p++; break;                                     /* PROTO */
            case 0x95: NEED(8); p += 8; break;                                  /* FRAME */
            case '.': return u->depth ? pop(u) : (u->ok = false, 0u);           /* STOP */
            case '(': push(u, new_ob(u, OB_MARK)); break;                       /* MARK */
            case ')': push(u, new_ob(u, OB_TUPLE)); break;                      /* EMPTY_TUPLE */
            case '}': push(u, new_ob(u, OB_DICT)); break;                       /* EMPTY_DICT */
            case ']': push(u, new_ob(u, OB_LIST)); break;                       /* EMPTY_LIST */
            case 'N': push(u, new_ob(u, OB_NONE)); break;                       /* NONE */
            case 0x88: case 0x89: {                                             /* NEWTRUE, NEWFALSE */
                uint32_t b = new_ob(u, OB_BOOL);
                if (u->ok) u->obs[b].i = op == 0x88;
                push(u, b);
                break;
            }
            case 'K': case 'M': case 'J': {                                     /* BININT1, BININT2, BININT */
                size_t n = op == 'K' ? 1u : op == 'M' ? 2u : 4u;
                NEED(n);
                int64_t v = op == 'K' ? p[0] : op == 'M' ? (int64_t)slett_get16(p) : (int64_t)(int32_t)slett_get32(p);
                p += n;
                uint32_t o = new_ob(u, OB_INT);
                if (u->ok) u->obs[o].i = v;
                push(u, o);
                break;
            }
            case 0x8a: {                                                        /* LONG1 */
                NEED(1);
                uint8_t n = *p++;
                NEED(n);
                int64_t v = 0;
                for (uint8_t k = 0; k < n && k < 8; k++) v |= (int64_t)p[k] << (8 * k);
                if (n > 0 && n < 8 && (p[n - 1] & 0x80u)) v -= (int64_t)1 << (8 * n);
                p += n;
                uint32_t o = new_ob(u, OB_INT);
                if (u->ok) u->obs[o].i = v;
                push(u, o);
                break;
            }
            case 'G': {                                                         /* BINFLOAT, big-endian */
                NEED(8);
                uint64_t bits = 0;
                for (int k = 0; k < 8; k++) bits = bits << 8 | p[k];
                p += 8;
                uint32_t o = new_ob(u, OB_FLOAT);
                if (u->ok) memcpy(&u->obs[o].f, &bits, 8);
                push(u, o);
                break;
            }
            case 'X': case 0x8c: case 0x8d: case 'B': case 'C': case 0x8e: case 'U': case 'T': {
                /* BINUNICODE, SHORT_BINUNICODE, BINUNICODE8, BINBYTES, SHORT_BINBYTES, BINBYTES8,
                   SHORT_BINSTRING, BINSTRING */
                size_t lw = op == 0x8c || op == 'C' || op == 'U' ? 1u : op == 0x8d || op == 0x8e ? 8u : 4u;
                NEED(lw);
                uint64_t n = lw == 1 ? p[0] : lw == 4 ? slett_get32(p) : slett_get64(p);
                p += lw;
                NEED(n);
                uint32_t o = new_ob(u, OB_STR);
                if (u->ok) u->obs[o].s = (Name){(const char *)p, (size_t)n};
                p += n;
                push(u, o);
                break;
            }
            case 'c': {                                                         /* GLOBAL module\nname\n */
                const uint8_t *nl1 = memchr(p, '\n', (size_t)(end - p));
                const uint8_t *nl2 = nl1 ? memchr(nl1 + 1, '\n', (size_t)(end - nl1 - 1)) : NULL;
                if (!nl2) { u->ok = false; return 0; }
                Name module = {(const char *)p, (size_t)(nl1 - p)}, name = {(const char *)nl1 + 1, (size_t)(nl2 - nl1 - 1)};
                p = nl2 + 1;
                uint32_t o = new_ob(u, OB_GLOBAL);
                if (u->ok) { u->obs[o].s = module; u->obs[o].s2 = name; }
                push(u, o);
                break;
            }
            case 0x93: {                                                        /* STACK_GLOBAL */
                uint32_t name = pop(u), module = pop(u);
                if (!u->ok || u->obs[name].kind != OB_STR || u->obs[module].kind != OB_STR) { u->ok = false; return 0; }
                Name m = u->obs[module].s, nm = u->obs[name].s;
                uint32_t o = new_ob(u, OB_GLOBAL);
                if (u->ok) { u->obs[o].s = m; u->obs[o].s2 = nm; }
                push(u, o);
                break;
            }
            case 'q': NEED(1); memo_put(u, p[0], u->depth ? u->stack[u->depth - 1] : 0); p += 1; break;     /* BINPUT */
            case 'r': NEED(4); memo_put(u, slett_get32(p), u->depth ? u->stack[u->depth - 1] : 0); p += 4; break; /* LONG_BINPUT */
            case 0x94: memo_put(u, u->memo_size, u->depth ? u->stack[u->depth - 1] : 0); break;              /* MEMOIZE */
            case 'h': NEED(1); push(u, memo_get(u, p[0])); p += 1; break;      /* BINGET */
            case 'j': NEED(4); push(u, memo_get(u, slett_get32(p))); p += 4; break; /* LONG_BINGET */
            case 't': {                                                         /* TUPLE */
                uint32_t from = mark_start(u);
                if (!u->ok) return 0;
                uint32_t t = make_tuple(u, from);
                u->depth = from - 1;
                push(u, t);
                break;
            }
            case 0x85: case 0x86: case 0x87: {                                  /* TUPLE1, TUPLE2, TUPLE3 */
                uint32_t n = (uint32_t)(op - 0x84u);
                if (u->depth < n) { u->ok = false; return 0; }
                uint32_t t = make_tuple(u, u->depth - n);
                u->depth -= n;
                push(u, t);
                break;
            }
            case 'Q': {                                                         /* BINPERSID */
                uint32_t pid = pop(u);
                if (!u->ok) return 0;
                const Ob *t = &u->obs[pid];
                /* ('storage', storage type, key, location, size) */
                uint32_t o = new_ob(u, OB_OPAQUE);
                if (!u->ok) return 0;
                t = &u->obs[pid];
                if (t->kind == OB_TUPLE && t->count >= 3 && u->obs[t->items[0]].kind == OB_STR &&
                    name_is(u->obs[t->items[0]].s, "storage") && u->obs[t->items[1]].kind == OB_GLOBAL &&
                    u->obs[t->items[2]].kind == OB_STR) {
                    u->obs[o].kind = OB_STORAGE;
                    u->obs[o].dtype = storage_dtype(&u->obs[t->items[1]]);
                    u->obs[o].s = u->obs[t->items[2]].s;
                }
                push(u, o);
                break;
            }
            case 'R': reduce(u); break;                                         /* REDUCE */
            case 'b': (void)pop(u); break;                                      /* BUILD: the state is not needed */
            case 's': {                                                         /* SETITEM */
                uint32_t value = pop(u), key = pop(u);
                if (!u->ok || !u->depth) { u->ok = false; return 0; }
                uint32_t d = u->stack[u->depth - 1];
                if (u->obs[d].kind == OB_DICT) { add_item(u, d, key); add_item(u, d, value); }
                break;
            }
            case 'u': case 'e': {                                               /* SETITEMS, APPENDS */
                uint32_t from = mark_start(u);
                if (!u->ok || from < 2) { u->ok = false; return 0; }
                uint32_t c = u->stack[from - 2];
                if ((op == 'u' && u->obs[c].kind == OB_DICT && (u->depth - from) % 2 == 0) ||
                    (op == 'e' && u->obs[c].kind == OB_LIST))
                    for (uint32_t k = from; u->ok && k < u->depth; k++) add_item(u, c, u->stack[k]);
                u->depth = from - 1;
                break;
            }
            case 'a': {                                                         /* APPEND */
                uint32_t v = pop(u);
                if (!u->ok || !u->depth) { u->ok = false; return 0; }
                uint32_t c = u->stack[u->depth - 1];
                if (u->obs[c].kind == OB_LIST) add_item(u, c, v);
                break;
            }
            default:
                u->ok = false;
                return 0;
        }
    }
#undef NEED
    return 0;
}

/* The dictionary of tensors in a pickled object: itself, or a value of a checkpoint dictionary
   (state_dict, model_state_dict or model first). */
static uint32_t find_state_dict(const Unpickler *u, uint32_t ob, int depth) {
    const Ob *o = &u->obs[ob];
    if (o->kind != OB_DICT || depth > 4) return UINT32_MAX;
    uint32_t tensors = 0;
    for (uint32_t k = 0; k + 1 < o->count; k += 2) tensors += u->obs[o->items[k + 1]].kind == OB_TENSOR;
    if (tensors > 0) return ob;
    static const char *const preferred[] = {"state_dict", "model_state_dict", "model", NULL};
    for (int pass = 0; pass < 2; pass++)
        for (uint32_t k = 0; k + 1 < o->count; k += 2) {
            const Ob *key = &u->obs[o->items[k]];
            bool named = false;
            for (int p = 0; preferred[p]; p++) named |= key->kind == OB_STR && name_is(key->s, preferred[p]);
            if (pass == 0 && !named) continue;
            uint32_t found = find_state_dict(u, o->items[k + 1], depth + 1);
            if (found != UINT32_MAX) return found;
        }
    return UINT32_MAX;
}

static bool read_torch_zip(const uint8_t *data, size_t size, Weights *w, Unpickler *u) {
    ZipEntry *entries;
    size_t count;
    if (!read_zip(data, size, &entries, &count, w)) return false;
    const ZipEntry *pkl = NULL;
    for (size_t k = 0; k < count; k++) {
        Name n = entries[k].name;
        if (n.n >= 8 && memcmp(n.s + n.n - 8, "data.pkl", 8) == 0 && (n.n == 8 || n.s[n.n - 9] == '/')) { pkl = &entries[k]; break; }
    }
    if (!pkl || !pkl->data) {
        free(entries);
        return werr(w, "the archive holds no stored data.pkl (is it a torch.save file?)");
    }
    const size_t prefix = pkl->name.n - 8;          /* "archive/" */
    u->ok = true;
    uint32_t root = unpickle(u, pkl->data, (size_t)pkl->size);
    if (!u->ok) {
        free(entries);
        return werr(w, "data.pkl cannot be read (it is not a state dict of tensors)");
    }
    uint32_t dict = find_state_dict(u, root, 0);
    if (dict == UINT32_MAX) {
        free(entries);
        return werr(w, "the file holds no dictionary of tensors");
    }
    const Ob *d = &u->obs[dict];
    for (uint32_t k = 0; k + 1 < d->count; k += 2) {
        const Ob *key = &u->obs[d->items[k]], *t = &u->obs[d->items[k + 1]];
        if (key->kind != OB_STR || t->kind != OB_TENSOR) continue;
        const Ob *storage = &u->obs[t->storage];
        /* the storage's bytes: archive/data/<key> */
        const ZipEntry *member = NULL;
        for (size_t e = 0; e < count && !member; e++) {
            Name n = entries[e].name;
            if (n.n == prefix + 5 + storage->s.n && memcmp(n.s, pkl->name.s, prefix) == 0 &&
                memcmp(n.s + prefix, "data/", 5) == 0 && memcmp(n.s + prefix + 5, storage->s.s, storage->s.n) == 0)
                member = &entries[e];
        }
        if (!member || !member->data) {
            free(entries);
            return werr(w, "a tensor's storage is missing or compressed");
        }
        Weight *x = new_weight(w);
        if (!x) { free(entries); return werr(w, "out of memory"); }
        x->name = key->s;
        x->dtype = t->dtype;
        x->ndims = t->ndims;
        memcpy(x->dims, t->dims, sizeof x->dims);
        memcpy(x->strides, t->strides, sizeof x->strides);
        x->data = member->data;
        x->bytes = member->size;
        x->offset = t->offset < 0 ? UINT64_MAX : (uint64_t)t->offset;
    }
    free(entries);
    return true;
}

/* ------------------------------------------------------------------------- assignment */

/* The module of a tensor: its name up to the last dot (empty without one), and the part after it. */
static Name module_of(Name n, Name *suffix) {
    size_t dot = n.n;
    while (dot > 0 && n.s[dot - 1] != '.') dot--;
    *suffix = (Name){n.s + dot, n.n - dot};
    return (Name){n.s, dot ? dot - 1 : 0};
}

/* Natural order of module names: runs of digits compare as numbers ("2" before "10"). */
static int natural_compare(Name a, Name b) {
    size_t i = 0, j = 0;
    while (i < a.n && j < b.n) {
        if (a.s[i] >= '0' && a.s[i] <= '9' && b.s[j] >= '0' && b.s[j] <= '9') {
            size_t i0 = i, j0 = j;
            while (i < a.n && a.s[i] == '0') i++;
            while (j < b.n && b.s[j] == '0') j++;
            size_t si = i, sj = j;
            while (i < a.n && a.s[i] >= '0' && a.s[i] <= '9') i++;
            while (j < b.n && b.s[j] >= '0' && b.s[j] <= '9') j++;
            if (i - si != j - sj) return i - si < j - sj ? -1 : 1;
            int c = memcmp(a.s + si, b.s + sj, i - si);
            if (c) return c;
            (void)i0; (void)j0;
            continue;
        }
        if (a.s[i] != b.s[j]) return (unsigned char)a.s[i] < (unsigned char)b.s[j] ? -1 : 1;
        i++, j++;
    }
    return a.n - i == b.n - j ? 0 : a.n - i < b.n - j ? -1 : 1;
}

static const Weight *tensor_of(const Weights *w, Name module, const char *suffix) {
    for (size_t k = 0; k < w->count; k++) {
        Name s, m = module_of(w->items[k].name, &s);
        if (name_eq(m, module) && name_is(s, suffix)) return &w->items[k];
    }
    return NULL;
}

/* Whether tensor t has the shape given, a type that converts to float and every element within its
   storage; sets the error naming module.what otherwise. *at: its bytes from the first element when it
   is stored in row-major order of the shape, NULL when its strides differ. */
static bool check_tensor(Weights *w, const Weight *t, const int64_t *shape, int ndims, Name module, const char *what,
                         const uint8_t **at) {
    bool fits = t->ndims == ndims;
    for (int d = 0; fits && d < ndims; d++) fits = t->dims[d] == shape[d];
    if (!fits || spingalett_dtype_size(t->dtype) == 0 || t->offset == UINT64_MAX) {
        char msg[200], dims[64] = "";
        for (int d = 0, used = 0; d < t->ndims && used < 56; d++)
            used += snprintf(dims + used, sizeof dims - used, "%s%lld", d ? ", " : "", (long long)t->dims[d]);
        snprintf(msg, sizeof msg, "%.*s.%s has shape [%s], not the %s the network has", (int)module.n, module.s, what, dims,
                 ndims == 4 ? "convolution filters" : ndims == 2 ? "dense weights" : "per-channel size");
        return werr(w, msg);
    }
    /* the last element: within the storage */
    const uint64_t size = spingalett_dtype_size(t->dtype);
    uint64_t last = t->offset, expect = 1;
    bool rows = true;
    for (int d = t->ndims - 1; d >= 0; d--) {
        uint64_t extent = (uint64_t)t->dims[d];
        if (extent == 0) { *at = NULL; return true; }
        if (t->strides[d] < 0 || (extent > 1 && (uint64_t)t->strides[d] > (UINT64_MAX - last) / (extent - 1)))
            return werr(w, "a tensor's data exceeds its storage");
        last += (extent - 1) * (uint64_t)t->strides[d];
        rows &= extent == 1 || (uint64_t)t->strides[d] == expect;
        expect *= extent;
    }
    if (last >= t->bytes / size) return werr(w, "a tensor's data exceeds its storage");
    *at = rows ? t->data + t->offset * size : NULL;
    return true;
}

/* The elements of t in row-major order into dst, through its strides. */
static void gather(const Weight *t, float *dst) {
    uint64_t n = elements(t);
    for (uint64_t i = 0; i < n; i++) (void)element(t, i, &dst[i]);
}

/* What one module's tensors become: in the first pass every tensor is checked and the scratch counted,
   in the second (write) they are copied, which cannot fail then. */
typedef struct {
    bool write;
    float *scratch;
    size_t scratch_need;
} Pass;

static void need(Pass *p, size_t floats) { if (floats > p->scratch_need) p->scratch_need = floats; }

static bool put_vector(Weights *w, Pass *p, const Weight *t, uint32_t n, float *dst, Name module, const char *what) {
    const int64_t shape[1] = {n};
    const uint8_t *at;
    if (!check_tensor(w, t, shape, 1, module, what, &at)) return false;
    if (!p->write) return true;
    if (at) spingalett_decode(dst, at, t->dtype, n);
    else gather(t, dst);
    return true;
}

static bool put_filters(Weights *w, Pass *p, const Weight *t, uint32_t OC, uint32_t CG, uint32_t KH, uint32_t KW,
                        float *dst, Name module) {
    const int64_t shape[4] = {OC, CG, KH, KW};
    const uint8_t *at;
    if (!check_tensor(w, t, shape, 4, module, "weight", &at)) return false;
    const size_t count = (size_t)OC * CG * KH * KW, filters = spingalett_filters_scratch(CG, KH, KW);
    if (!p->write) { need(p, at ? filters : count + filters); return true; }
    int dtype = t->dtype;
    if (!at) {                              /* strided: gathered first */
        gather(t, p->scratch + filters);
        at = (const uint8_t *)(p->scratch + filters);
        dtype = SPG_DTYPE_F32;
    }
    spingalett_import_filters(dst, at, dtype, OC, CG, KH, KW, p->scratch);
    return true;
}

static bool put_dense(Weights *w, Pass *p, const Weight *t, uint32_t rows, uint32_t cols, uint32_t C, uint32_t HW,
                      float *dst, Name module) {
    const int64_t shape[2] = {rows, cols};
    const uint8_t *at;
    if (!check_tensor(w, t, shape, 2, module, "weight", &at)) return false;
    /* a transposed view ([cols][rows] in memory, as x @ W saves it) reads as a transposed product */
    const bool transposed = !at && t->strides[0] == 1 && t->strides[1] == (int64_t)rows;
    const size_t count = (size_t)rows * cols, scratch = spingalett_dense_scratch(rows, cols, transposed);
    if (!p->write) { need(p, at || transposed ? scratch : count + scratch); return true; }
    int dtype = t->dtype;
    if (transposed) {
        at = t->data + t->offset * spingalett_dtype_size(dtype);
    } else if (!at) {
        gather(t, p->scratch + scratch);
        at = (const uint8_t *)(p->scratch + scratch);
        dtype = SPG_DTYPE_F32;
    }
    spingalett_import_dense(dst, at, dtype, rows, cols, C, HW, transposed, 1.0f, p->scratch);
    return true;
}

/* Copies module `mod` into layer l (a dense, convolution or batch normalization layer). */
static bool put_module(NeuralNetwork *net, Weights *w, Pass *p, uint32_t l, Name mod) {
    char msg[200];
    const LayerShape *s = &net->shapes[l];
    const uint32_t src = spingalett_source(net, l), rows = spingalett_weight_rows(net, l - 1);
    float *W = net->weights + net->weight_offsets[l - 1], *b = net->biases + net->bias_offsets[l - 1];
    const Weight *tw = tensor_of(w, mod, "weight"), *tb = tensor_of(w, mod, "bias");
    if (s->type == LAYER_BATCH_NORM) {
        const Weight *mean = tensor_of(w, mod, "running_mean"), *var = tensor_of(w, mod, "running_var");
        if (!mean || !var) {
            snprintf(msg, sizeof msg, "module '%.*s' is no batch normalization, which layer %u is", (int)mod.n, mod.s, l);
            return werr(w, msg);
        }
        uint64_t o = net->bias_offsets[l - 1];
        if (p->write)
            for (uint32_t c = 0; c < rows; c++) { W[c] = 1.0f; b[c] = 0.0f; }     /* without affine parameters */
        return (!tw || put_vector(w, p, tw, rows, W, mod, "weight")) && (!tb || put_vector(w, p, tb, rows, b, mod, "bias")) &&
               put_vector(w, p, mean, rows, net->running_mean + o, mod, "running_mean") &&
               put_vector(w, p, var, rows, net->running_var + o, mod, "running_var");
    }
    if (!tw) {
        snprintf(msg, sizeof msg, "module '%.*s' has no weight for layer %u", (int)mod.n, mod.s, l);
        return werr(w, msg);
    }
    if (p->write)
        for (uint32_t r = 0; r < rows; r++) b[r] = 0.0f;
    const LayerShape *in = &net->shapes[src];
    bool ok = s->type == LAYER_CONV2D
            ? put_filters(w, p, tw, rows, in->channels / s->groups, s->kernel_h, s->kernel_w, W, mod)
            : put_dense(w, p, tw, rows, net->topology[src], in->channels, in->height * in->width, W, mod);
    return ok && (!tb || put_vector(w, p, tb, rows, b, mod, "bias"));
}

static bool assign(NeuralNetwork *net, Weights *w, const char *const *names, uint32_t name_count, bool sort) {
    /* the modules, in their order in the file (sorted naturally for safetensors) or as given */
    Name *modules = (Name *)malloc((w->count + name_count + 1) * sizeof(Name));
    if (!modules) return werr(w, "out of memory");
    size_t m = 0;
    if (names) {
        for (uint32_t k = 0; k < name_count; k++) modules[m++] = (Name){names[k], strlen(names[k])};
    } else {
        for (size_t k = 0; k < w->count; k++) {
            Name s, mod = module_of(w->items[k].name, &s);
            if (name_is(s, "num_batches_tracked")) continue;
            bool seen = false;
            for (size_t j = 0; j < m && !seen; j++) seen = name_eq(modules[j], mod);
            if (!seen) modules[m++] = mod;
        }
        if (sort)
            for (size_t i = 1; i < m; i++)
                for (size_t j = i; j > 0 && natural_compare(modules[j - 1], modules[j]) > 0; j--) {
                    Name t = modules[j]; modules[j] = modules[j - 1]; modules[j - 1] = t;
                }
    }
    /* every module checked before any parameter changes, so that an error leaves the network as it was */
    Pass pass = {0};
    bool ok = true;
    for (int write = 0; ok && write < 2; write++) {
        pass.write = write;
        if (write && pass.scratch_need) {
            pass.scratch = (float *)malloc(pass.scratch_need * sizeof(float));
            if (!pass.scratch) { ok = werr(w, "out of memory"); break; }
        }
        size_t next = 0;
        for (uint32_t l = 1; ok && l < net->layers; l++) {
            const LayerType type = net->shapes[l].type;
            if (type != LAYER_DENSE && type != LAYER_CONV2D && type != LAYER_BATCH_NORM) continue;
            if (next == m) {
                char msg[200];
                snprintf(msg, sizeof msg, "the file has weights for %zu layers; the network has more (layer %u is the "
                         "first without)", m, l);
                ok = werr(w, msg);
                break;
            }
            ok = put_module(net, w, &pass, l, modules[next++]);
        }
        if (ok && next < m) {
            char msg[200];
            snprintf(msg, sizeof msg, "the file has weights for %zu layers, the network %zu ('%.*s' is left)", m, next,
                     (int)modules[next].n, modules[next].s);
            ok = werr(w, msg);
        }
    }
    free(pass.scratch);
    free(modules);
    return ok;
}

bool spingalett_load_pytorch_from_memory(NeuralNetwork *net, const void *data, size_t size, const char *const *modules,
                                         uint32_t module_count) {
    if (!net || !data || (module_count && !modules)) {
        set_error(SPINGALETT_ERR_INVALID, "PyTorch weights: net, data or modules is NULL");
        return false;
    }
    Weights w = {0};
    Unpickler u = {0};
    const uint8_t *p = (const uint8_t *)data;
    bool zip = size >= 4 && slett_get32(p) == 0x04034b50u;
    bool ok = zip ? read_torch_zip(p, size, &w, &u) : read_safetensors(p, size, &w);
    ok = ok && assign(net, &w, modules, module_count, !zip);
    for (uint32_t k = 0; k < u.count; k++) free(u.obs[k].items);
    free(u.obs);
    free(u.stack);
    free(u.memo);
    free(w.items);
    if (!ok) {
        set_error(SPINGALETT_ERR_INVALID, w.error[0] ? w.error : "PyTorch weights: the file cannot be read");
        spingalett_log(LOG_ERROR, "%s", spingalett_last_error_message());
        return false;
    }
    return true;
}

bool spingalett_load_pytorch(NeuralNetwork *net, const char *path, const char *const *modules, uint32_t module_count) {
    if (!path) {
        set_error(SPINGALETT_ERR_INVALID, "PyTorch weights: path is NULL");
        return false;
    }
    SpgFileView file;
    if (!spingalett_file_open(&file, path)) return false;
    bool ok = spingalett_load_pytorch_from_memory(net, file.data, file.size, modules, module_count);
    spingalett_file_close(&file);
    return ok;
}
