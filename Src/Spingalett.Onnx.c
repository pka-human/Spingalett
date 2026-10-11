/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/*
 * ONNX import: a reader of the protocol buffer wire format, just what ONNX models need (no
 * dependency), and a translation of the graph's nodes into layers.
 *
 * ONNX tensors are NCHW, Spingalett's channels-last: convolution filters [OC][IC/g][KH][KW] become
 * [OC][KH][KW][IC/g], and a dense layer that reads a flattened map (Flatten, or Reshape to two
 * dimensions) has its weight columns reordered from (c, h, w) to (h, w, c), so the network computes
 * what the model does on the same image in channels-last order. Activations merge into the layer
 * that produces their input when nothing else reads it; an addition of a constant after a product
 * (MatMul then Add) becomes its bias; Identity, Dropout and the flattening operators produce no
 * layer.
 *
 * Nodes whose inputs are all constants are computed on import: the shape arithmetic exporters leave in
 * graphs. Shapes are those of one sample (models exported with one, or with a dynamic batch).
 */

#include "Spingalett.Private.h"
#include <stdarg.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

/* ------------------------------------------------------------------------- protocol buffers */

typedef struct {
    const uint8_t *p, *end;
    bool ok;
} Pb;

enum { WIRE_VARINT = 0, WIRE_FIXED64 = 1, WIRE_BYTES = 2, WIRE_FIXED32 = 5 };

static uint64_t pb_varint(Pb *r) {
    uint64_t v = 0;
    for (int shift = 0; shift < 64; shift += 7) {
        if (r->p >= r->end) { r->ok = false; return 0; }
        uint8_t b = *r->p++;
        v |= (uint64_t)(b & 0x7Fu) << shift;
        if (!(b & 0x80u)) return v;
    }
    r->ok = false;
    return 0;
}

/* The next field's number and wire type; false at the end or on an error. */
static bool pb_next(Pb *r, uint32_t *field, uint32_t *wire) {
    if (!r->ok || r->p >= r->end) return false;
    uint64_t key = pb_varint(r);
    *field = (uint32_t)(key >> 3);
    *wire = (uint32_t)(key & 7u);
    return r->ok && *field != 0;
}

/* A length-delimited field's bytes, as a reader of their own. */
static Pb pb_bytes(Pb *r) {
    uint64_t n = pb_varint(r);
    Pb sub = {r->p, r->p, false};
    if (!r->ok || n > (uint64_t)(r->end - r->p)) { r->ok = false; return sub; }
    sub.end = r->p + n;
    sub.ok = true;
    r->p += n;
    return sub;
}

static void pb_skip(Pb *r, uint32_t wire) {
    switch (wire) {
        case WIRE_VARINT: (void)pb_varint(r); break;
        case WIRE_FIXED64: if (r->end - r->p < 8) r->ok = false; else r->p += 8; break;
        case WIRE_BYTES: (void)pb_bytes(r); break;
        case WIRE_FIXED32: if (r->end - r->p < 4) r->ok = false; else r->p += 4; break;
        default: r->ok = false; break;
    }
}

static float pb_float(Pb *r) {
    float f = 0.0f;
    if (r->end - r->p < 4) { r->ok = false; return f; }
    memcpy(&f, r->p, 4);
    r->p += 4;
    return f;
}

/* A string field compared with a C string. */
typedef struct { const char *s; size_t n; } Str;

static Str pb_str(Pb *r) {
    Pb b = pb_bytes(r);
    return (Str){(const char *)b.p, (size_t)(b.end - b.p)};
}

static bool str_is(Str a, const char *b) { return a.n == strlen(b) && (a.n == 0 || memcmp(a.s, b, a.n) == 0); }
static bool str_eq(Str a, Str b) { return a.n == b.n && (a.n == 0 || memcmp(a.s, b.s, a.n) == 0); }

/* ------------------------------------------------------------------------- tensors and attributes */

#define MAX_DIMS 8
#define MAX_NODE_INPUTS 16
#define MAX_NODE_OUTPUTS 16
#define MAX_ATTRS 16
#define MAX_INTS 16                 /* integers an attribute's list may hold */

enum {
    ONNX_FLOAT = 1, ONNX_UINT8 = 2, ONNX_INT8 = 3, ONNX_INT32 = 6, ONNX_INT64 = 7, ONNX_BOOL = 9, ONNX_FLOAT16 = 10,
    ONNX_DOUBLE = 11, ONNX_BFLOAT16 = 16
};

/* Element types of one byte, which spingalett_decode() does not read. */
static bool byte_type(int32_t type) { return type == ONNX_UINT8 || type == ONNX_INT8 || type == ONNX_BOOL; }

typedef struct {
    Str name;
    int64_t dims[MAX_DIMS];
    int ndims;
    int32_t type;
    Pb raw;                     /* raw_data, or the bytes of the external file */
    bool has_raw, external;
    Pb typed;                   /* float_data / int32_data / int64_data / double_data */
    uint32_t typed_wire;
    Str location;               /* external data: the file (relative to the model's), offset and length */
    uint64_t offset, length;
    bool has_length;
} TensorRef;

/* A decimal string as a number; false when it is not one. */
static bool str_u64(Str s, uint64_t *v) {
    *v = 0;
    if (s.n == 0 || s.n > 19) return false;
    for (size_t k = 0; k < s.n; k++) {
        if (s.s[k] < '0' || s.s[k] > '9') return false;
        *v = *v * 10u + (uint64_t)(s.s[k] - '0');
    }
    return true;
}

static void read_tensor(Pb r, TensorRef *t) {
    memset(t, 0, sizeof *t);
    uint32_t f, w;
    while (pb_next(&r, &f, &w)) {
        if (f == 1 && w == WIRE_VARINT) {
            if (t->ndims < MAX_DIMS) t->dims[t->ndims] = (int64_t)pb_varint(&r);
            else (void)pb_varint(&r);
            t->ndims++;
        } else if (f == 1 && w == WIRE_BYTES) {
            Pb d = pb_bytes(&r);
            while (d.ok && d.p < d.end) {
                int64_t v = (int64_t)pb_varint(&d);
                if (t->ndims < MAX_DIMS) t->dims[t->ndims] = v;
                t->ndims++;
            }
        } else if (f == 2 && w == WIRE_VARINT) {
            t->type = (int32_t)pb_varint(&r);
        } else if (f == 8 && w == WIRE_BYTES) {
            t->name = pb_str(&r);
        } else if (f == 9 && w == WIRE_BYTES) {
            t->raw = pb_bytes(&r);
            t->has_raw = true;
        } else if ((f == 4 || f == 5 || f == 7 || f == 10) && (w == WIRE_BYTES || w == WIRE_FIXED32 || w == WIRE_VARINT ||
                                                               w == WIRE_FIXED64)) {
            /* typed data: packed (one bytes field) or one field per value; the first field is kept and
               the values are read from it on (fields repeat back to back in practice) */
            if (!t->typed.ok) {
                t->typed = (Pb){r.p, r.end, true};
                t->typed_wire = w;
            }
            pb_skip(&r, w);
        } else if (f == 14 && w == WIRE_VARINT) {
            t->external = pb_varint(&r) == 1;
        } else if (f == 13 && w == WIRE_BYTES) {
            /* external_data: key-value pairs */
            Pb e = pb_bytes(&r);
            Str key = {0}, value = {0};
            uint32_t ef, ew;
            while (pb_next(&e, &ef, &ew)) {
                if (ef == 1 && ew == WIRE_BYTES) key = pb_str(&e);
                else if (ef == 2 && ew == WIRE_BYTES) value = pb_str(&e);
                else pb_skip(&e, ew);
            }
            if (str_is(key, "location")) t->location = value;
            else if (str_is(key, "offset") && !str_u64(value, &t->offset)) t->offset = UINT64_MAX;
            else if (str_is(key, "length")) t->has_length = str_u64(value, &t->length) || (t->length = UINT64_MAX, true);
        } else {
            pb_skip(&r, w);
        }
    }
}

/* Elements of the shape; UINT64_MAX when the product overflows (no file holds that many). */
static uint64_t tensor_count(const TensorRef *t) {
    uint64_t n = 1;
    for (int d = 0; d < t->ndims && d < MAX_DIMS; d++) {
        uint64_t e = t->dims[d] < 0 ? 0u : (uint64_t)t->dims[d];
        if (e && n > UINT64_MAX / e) return UINT64_MAX;
        n *= e;
    }
    return n;
}

/* The element type of an ONNX tensor type (SPG_DTYPE_OTHER: not numeric). */
static int onnx_dtype(int32_t type) {
    switch (type) {
        case ONNX_FLOAT: return SPG_DTYPE_F32;
        case ONNX_INT32: return SPG_DTYPE_I32;
        case ONNX_INT64: return SPG_DTYPE_I64;
        case ONNX_FLOAT16: return SPG_DTYPE_F16;
        case ONNX_DOUBLE: return SPG_DTYPE_F64;
        case ONNX_BFLOAT16: return SPG_DTYPE_BF16;
        default: return SPG_DTYPE_OTHER;
    }
}

/* Elements the tensor's data can hold at most (raw: its bytes over the element size; typed fields:
   one byte each at least), so that a shape larger than the file is rejected before allocating. */
static uint64_t tensor_capacity(const TensorRef *t) {
    if (t->has_raw) {
        uint64_t size = byte_type(t->type) ? 1u : spingalett_dtype_size(onnx_dtype(t->type));
        return size ? (uint64_t)(t->raw.end - t->raw.p) / size : 0u;
    }
    return t->typed.ok ? (uint64_t)(t->typed.end - t->typed.p) : 0u;
}

/* The tensor's values as floats (count of them); false when the type is not numeric or the data is
   short. */
static bool tensor_floats(const TensorRef *t, float *out, uint64_t count) {
    if (t->external) return false;
    if (t->has_raw && byte_type(t->type)) {
        if ((uint64_t)(t->raw.end - t->raw.p) < count) return false;
        for (uint64_t i = 0; i < count; i++)
            out[i] = t->type == ONNX_INT8 ? (float)(int8_t)t->raw.p[i] : t->type == ONNX_BOOL ? (float)(t->raw.p[i] != 0)
                                                                            : (float)t->raw.p[i];
        return true;
    }
    if (t->has_raw) {
        uint64_t size = spingalett_dtype_size(onnx_dtype(t->type));
        if (size == 0 || (uint64_t)(t->raw.end - t->raw.p) < count * size) return false;
        spingalett_decode(out, t->raw.p, onnx_dtype(t->type), (size_t)count);
        return true;
    }
    if (count == 0) return true;
    if (!t->typed.ok) return false;
    /* typed fields: the field number tells the type, packed or repeated */
    Pb r = t->typed;
    uint64_t i = 0;
    uint32_t f, w;
    while (i < count && pb_next(&r, &f, &w)) {
        if (f != 4 && f != 5 && f != 7 && f != 10) { pb_skip(&r, w); continue; }
        if (w == WIRE_BYTES) {
            Pb d = pb_bytes(&r);
            while (d.ok && d.p < d.end && i < count) {
                if (f == 4) out[i++] = pb_float(&d);
                else if (f == 10) { double v = 0; if (d.end - d.p >= 8) memcpy(&v, d.p, 8); d.p += 8; out[i++] = (float)v; }
                else out[i++] = (float)(int64_t)pb_varint(&d);
            }
        } else if (w == WIRE_FIXED32 && f == 4) {
            out[i++] = pb_float(&r);
        } else if (w == WIRE_FIXED64 && f == 10) {
            double v = 0;
            if (r.end - r.p >= 8) memcpy(&v, r.p, 8);
            r.p += 8;
            out[i++] = (float)v;
        } else if (w == WIRE_VARINT) {
            out[i++] = (float)(int64_t)pb_varint(&r);
        } else {
            pb_skip(&r, w);
        }
    }
    return i == count;
}

/* An attribute: its name and value (an integer, a float, a list of integers or floats, a tensor). */
typedef struct {
    Str name;
    int64_t i;
    float f;
    int64_t ints[MAX_INTS];
    int nints;
    float floats[MAX_DIMS];
    int nfloats;
    Str s;
    Pb t;
    bool has_t;
} Attr;

static void read_attr(Pb r, Attr *a) {
    memset(a, 0, sizeof *a);
    uint32_t f, w;
    while (pb_next(&r, &f, &w)) {
        if (f == 1 && w == WIRE_BYTES) a->name = pb_str(&r);
        else if (f == 2 && w == WIRE_FIXED32) a->f = pb_float(&r);
        else if (f == 3 && w == WIRE_VARINT) a->i = (int64_t)pb_varint(&r);
        else if (f == 4 && w == WIRE_BYTES) a->s = pb_str(&r);
        else if (f == 5 && w == WIRE_BYTES) { a->t = pb_bytes(&r); a->has_t = true; }
        else if (f == 7 && w == WIRE_FIXED32) { float v = pb_float(&r); if (a->nfloats < MAX_DIMS) a->floats[a->nfloats] = v; a->nfloats++; }
        else if (f == 7 && w == WIRE_BYTES) {
            Pb d = pb_bytes(&r);
            while (d.ok && d.p < d.end) { float v = pb_float(&d); if (a->nfloats < MAX_DIMS) a->floats[a->nfloats] = v; a->nfloats++; }
        } else if (f == 8 && w == WIRE_VARINT) { int64_t v = (int64_t)pb_varint(&r); if (a->nints < MAX_INTS) a->ints[a->nints] = v; a->nints++; }
        else if (f == 8 && w == WIRE_BYTES) {
            Pb d = pb_bytes(&r);
            while (d.ok && d.p < d.end) { int64_t v = (int64_t)pb_varint(&d); if (a->nints < MAX_INTS) a->ints[a->nints] = v; a->nints++; }
        } else pb_skip(&r, w);
    }
}

typedef struct {
    Str op, name, domain;
    Str inputs[MAX_NODE_INPUTS], outputs[MAX_NODE_OUTPUTS];
    int ninputs, noutputs;
    Attr attrs[MAX_ATTRS];
    int nattrs;
} Node;

static bool read_node(Pb r, Node *n) {
    memset(n, 0, sizeof *n);
    uint32_t f, w;
    while (pb_next(&r, &f, &w)) {
        if (f == 1 && w == WIRE_BYTES) {
            Str s = pb_str(&r);
            if (n->ninputs < MAX_NODE_INPUTS) n->inputs[n->ninputs] = s;
            n->ninputs++;
        } else if (f == 2 && w == WIRE_BYTES) {
            Str s = pb_str(&r);
            if (n->noutputs < MAX_NODE_OUTPUTS) n->outputs[n->noutputs] = s;
            n->noutputs++;
        } else if (f == 3 && w == WIRE_BYTES) {
            n->name = pb_str(&r);
        } else if (f == 4 && w == WIRE_BYTES) {
            n->op = pb_str(&r);
        } else if (f == 7 && w == WIRE_BYTES) {
            n->domain = pb_str(&r);
        } else if (f == 5 && w == WIRE_BYTES) {
            Pb a = pb_bytes(&r);
            if (n->nattrs < MAX_ATTRS) read_attr(a, &n->attrs[n->nattrs++]);
        } else {
            pb_skip(&r, w);
        }
    }
    return r.ok && n->ninputs <= MAX_NODE_INPUTS && n->noutputs <= MAX_NODE_OUTPUTS;
}

static const Attr *attr(const Node *n, const char *name) {
    for (int k = 0; k < n->nattrs; k++)
        if (str_is(n->attrs[k].name, name)) return &n->attrs[k];
    return NULL;
}

static int64_t attr_int(const Node *n, const char *name, int64_t fallback) {
    const Attr *a = attr(n, name);
    return a ? a->i : fallback;
}

static float attr_float(const Node *n, const char *name, float fallback) {
    const Attr *a = attr(n, name);
    return a ? a->f : fallback;
}

/* ------------------------------------------------------------------------- the translation */

/* A named value of the graph: a layer's output, viewed as NCHW (c x h x w; flat: as the [N, c h w]
   rows of a flattened map), or a constant. */
typedef struct {
    Str name;
    bool constant;
    uint32_t layer;
    uint32_t c, h, w;
    bool flat;
    bool vector;                /* [N, F] in the file (not flat: a vector of its own, such as a dense output) */
    bool nhwc;                  /* a map transposed to [N, H, W, C]: what layer normalization over its
                                   channels reads, and elementwise operators keep */
    bool alias;                 /* another name of a layer's output (made by a node that read one) */
    TensorRef tensor;           /* constants: where their data is (in the file) */
    float *data;                /* constants: their values as floats, once something needed them */
    bool owned;                 /* data is this value's to free */
    int64_t dims[MAX_DIMS];
    int ndims;
    uint64_t count;
    int32_t ctype;              /* constants: the ONNX element type */
} Value;

/* Names to numbers: an open-addressing hash table. */
typedef struct { Str name; uint32_t value; bool used; } Slot;
typedef struct { Slot *slots; size_t cap, count; } Table;

static uint64_t str_hash(Str s) {
    uint64_t h = 1469598103934665603ull;                    /* FNV-1a */
    for (size_t k = 0; k < s.n; k++) h = (h ^ (uint8_t)s.s[k]) * 1099511628211ull;
    return h;
}

/* The slot of name: its entry, or the free slot where it goes (the table has room). */
static Slot *table_slot(const Table *t, Str name) {
    size_t k = (size_t)str_hash(name) & (t->cap - 1);
    while (t->slots[k].used && !str_eq(t->slots[k].name, name)) k = (k + 1) & (t->cap - 1);
    return &t->slots[k];
}

/* Room for one more entry: the table at most half full. */
static bool table_room(Table *t) {
    if (2 * (t->count + 1) <= t->cap) return true;
    size_t cap = t->cap ? 2 * t->cap : 64;
    Table grown = {(Slot *)calloc(cap, sizeof(Slot)), cap, t->count};
    if (!grown.slots) return false;
    for (size_t k = 0; k < t->cap; k++)
        if (t->slots[k].used) *table_slot(&grown, t->slots[k].name) = t->slots[k];
    free(t->slots);
    *t = grown;
    return true;
}

static const Slot *table_find(const Table *t, Str name) {
    if (!t->cap) return NULL;
    const Slot *slot = table_slot(t, name);
    return slot->used ? slot : NULL;
}

/* An external data file, mapped once for every tensor in it. */
typedef struct { Str location; SpgFileView view; } External;

typedef struct {
    NeuralNetwork *net;
    Value *values;
    size_t nvalues, cap;
    Table names;                /* value names: the index of the latest value of each */
    Table uses;                 /* every node input and graph output: how often each name is read */
    const char *dir;            /* the model's folder (with its separator), for external data; NULL: none */
    External *externals;
    size_t nexternals;
    char error[256];
    bool bias_open;             /* the last layer is a product whose bias a following Add may set */
    int64_t opset;              /* the default domain's operator set (0: not given) */
} Importer;

static bool fail(Importer *im, const char *fmt, Str a, Str b) {
    if (!im->error[0])
        snprintf(im->error, sizeof im->error, fmt, (int)a.n, a.s, (int)b.n, b.s);
    return false;
}

static Value *find(Importer *im, Str name) {
    const Slot *slot = table_find(&im->names, name);
    return slot ? &im->values[slot->value] : NULL;
}

static Value *add_value(Importer *im, Str name) {
    if (im->nvalues == im->cap) {
        size_t cap = im->cap ? 2 * im->cap : 64;
        Value *v = (Value *)realloc(im->values, cap * sizeof(Value));
        if (!v) return NULL;
        im->values = v;
        im->cap = cap;
    }
    if (!table_room(&im->names)) return NULL;
    Slot *slot = table_slot(&im->names, name);
    if (!slot->used) *slot = (Slot){name, 0, true}, im->names.count++;
    slot->value = (uint32_t)im->nvalues;
    Value *v = &im->values[im->nvalues++];
    memset(v, 0, sizeof *v);
    v->name = name;
    return v;
}

/* Readers of a value (node inputs and graph outputs). */
static uint32_t readers(const Importer *im, Str name) {
    const Slot *slot = table_find(&im->uses, name);
    return slot ? slot->value : 0u;
}

/* A constant's values as floats: decoded when first needed. NULL when they cannot be read. */
static const float *constant_floats(Importer *im, Value *v) {
    static const float none = 0.0f;
    if (v->data || v->count == 0) return v->data ? v->data : &none;
    v->data = (float *)malloc((size_t)v->count * sizeof(float));
    v->owned = v->data != NULL;
    if (v->data && !tensor_floats(&v->tensor, v->data, v->count)) {
        free(v->data);
        v->data = NULL;
        v->owned = false;
    }
    if (!v->data) fail(im, "ONNX import: the constant '%.*s' cannot be read%.*s", v->name, (Str){"", 0});
    return v->data;
}

/* A constant's data for a weight conversion: its bytes in the file when they are raw values of a
   numeric type, else its values decoded. */
static const uint8_t *constant_bytes(Importer *im, Value *v, int *dtype) {
    const TensorRef *t = &v->tensor;
    const uint64_t size = spingalett_dtype_size(onnx_dtype(t->type));
    if (!v->data && t->has_raw && !t->external && size && (uint64_t)(t->raw.end - t->raw.p) / size >= v->count) {
        *dtype = onnx_dtype(t->type);
        return t->raw.p;
    }
    *dtype = SPG_DTYPE_F32;
    return (const uint8_t *)constant_floats(im, v);
}

/* Readers of a layer's output under all of its names, the nodes that only renamed it aside. */
static uint32_t layer_readers(const Importer *im, uint32_t layer) {
    uint32_t total = 0, aliases = 0;
    for (size_t k = 0; k < im->nvalues; k++) {
        const Value *v = &im->values[k];
        if (v->constant || v->layer != layer) continue;
        total += readers(im, v->name);
        aliases += v->alias;
    }
    return total - aliases;
}

/* Operators that may read a map transposed to channels last: elementwise ones, layer normalization
   (over its channels) and the transposition back. */
static bool reads_nhwc(const Node *n) {
    static const char *const ops[] = {"LayerNormalization", "Transpose", "Relu", "Sigmoid", "Tanh", "LeakyRelu"};
    for (size_t k = 0; k < sizeof ops / sizeof ops[0]; k++)
        if (str_is(n->op, ops[k])) return true;
    return false;
}

static const Value *layer_input(Importer *im, const Node *n, int k) {
    if (k >= n->ninputs) { fail(im, "ONNX import: %.*s node '%.*s' lacks an input", n->op, n->name); return NULL; }
    const Value *v = find(im, n->inputs[k]);
    if (!v) { fail(im, "ONNX import: input '%.*s' of %.*s is not produced by an earlier node", n->inputs[k], n->op); return NULL; }
    if (v->constant) { fail(im, "ONNX import: %.*s reads the constant '%.*s' where a layer output is expected", n->op, n->inputs[k]); return NULL; }
    if (v->nhwc && !reads_nhwc(n)) {
        fail(im, "ONNX import: %.*s reads '%.*s', a map transposed to channels last (which only layer normalization "
             "and elementwise operators may)", n->op, n->inputs[k]);
        return NULL;
    }
    return v;
}

/* Input k as a constant of count values (any count: 0), its values decoded. */
static Value *constant_input(Importer *im, const Node *n, int k, uint64_t count) {
    if (k >= n->ninputs || n->inputs[k].n == 0) return NULL;
    Value *v = find(im, n->inputs[k]);
    if (!v || !v->constant || (count && v->count != count)) {
        fail(im, "ONNX import: %.*s needs the constant '%.*s' of the expected size", n->op, n->inputs[k]);
        return NULL;
    }
    return constant_floats(im, v) ? v : NULL;
}

/* A new layer's output, viewed with the shape Spingalett gives it. */
static Value *layer_output(Importer *im, const Node *n, uint32_t layer) {
    Value *v = add_value(im, n->outputs[0]);
    if (!v) return NULL;
    SpingalettNetworkLayer d;
    spingalett_network_layer(im->net, layer, &d);
    v->layer = layer;
    v->c = d.channels;
    v->h = d.height;
    v->w = d.width;
    return v;
}

static bool added(Importer *im, uint32_t layer, const Node *n) {
    if (layer != SPINGALETT_NO_LAYER) return true;
    char msg[160];
    snprintf(msg, sizeof msg, "%s", spingalett_last_error_message());
    if (!im->error[0])
        snprintf(im->error, sizeof im->error, "ONNX import: %.*s node '%.*s': %s", (int)n->op.n, n->op.s, (int)n->name.n,
                 n->name.s, msg);
    return false;
}

static float *layer_weights(NeuralNetwork *net, uint32_t layer) { return net->weights + net->weight_offsets[layer - 1]; }
static float *layer_biases(NeuralNetwork *net, uint32_t layer) { return net->biases + net->bias_offsets[layer - 1]; }

/* Symmetric padding from ONNX pads [top, left, bottom, right] (or auto_pad). */
static bool window_attrs(Importer *im, const Node *n, uint32_t *kh, uint32_t *kw, uint32_t *sh, uint32_t *sw,
                         uint32_t *ph, uint32_t *pw, uint32_t in_h, uint32_t in_w) {
    const Attr *ks = attr(n, "kernel_shape"), *st = attr(n, "strides"), *pd = attr(n, "pads"), *dl = attr(n, "dilations");
    if (!ks || ks->nints != 2) return fail(im, "ONNX import: %.*s node '%.*s' needs a 2-D kernel_shape", n->op, n->name);
    *kh = (uint32_t)ks->ints[0];
    *kw = (uint32_t)ks->ints[1];
    *sh = st && st->nints == 2 ? (uint32_t)st->ints[0] : 1u;
    *sw = st && st->nints == 2 ? (uint32_t)st->ints[1] : 1u;
    *ph = *pw = 0;
    if (dl && (dl->nints != 2 || dl->ints[0] != 1 || dl->ints[1] != 1))
        return fail(im, "ONNX import: %.*s node '%.*s' has dilations, which are not supported", n->op, n->name);
    if (attr_int(n, "ceil_mode", 0) != 0)
        return fail(im, "ONNX import: %.*s node '%.*s' rounds its output size up (ceil_mode), which is not supported",
                    n->op, n->name);
    const Attr *ap = attr(n, "auto_pad");
    if (ap && ap->s.n && !str_is(ap->s, "NOTSET") && !str_is(ap->s, "VALID")) {
        /* SAME_UPPER / SAME_LOWER with stride 1 and odd kernels pad kernel / 2 on each side */
        if (*sh != 1 || *sw != 1 || *kh % 2 == 0 || *kw % 2 == 0)
            return fail(im, "ONNX import: %.*s node '%.*s' pads asymmetrically (auto_pad), which is not supported",
                        n->op, n->name);
        *ph = *kh / 2;
        *pw = *kw / 2;
    } else if (pd && pd->nints == 4) {
        if (pd->ints[0] != pd->ints[2] || pd->ints[1] != pd->ints[3])
            return fail(im, "ONNX import: %.*s node '%.*s' pads asymmetrically, which is not supported", n->op, n->name);
        *ph = (uint32_t)pd->ints[0];
        *pw = (uint32_t)pd->ints[1];
    }
    (void)in_h; (void)in_w;
    return true;
}

/* The activation an ONNX operator applies, ACT_COUNT when it is none of them. */
static ActivationFunction activation_of(const Node *n, Importer *im) {
    if (str_is(n->op, "Relu")) return ACT_RELU;
    if (str_is(n->op, "Sigmoid")) return ACT_SIGMOID;
    if (str_is(n->op, "Tanh")) return ACT_TANH;
    if (str_is(n->op, "Softmax")) return ACT_SOFTMAX;
    if (str_is(n->op, "LeakyRelu")) {
        float alpha = attr_float(n, "alpha", 0.01f);
        if (fabsf(alpha - 0.01f) > 1e-6f) {
            fail(im, "ONNX import: LeakyRelu node '%.*s' has a slope other than 0.01%.*s", n->name, (Str){"", 0});
            return ACT_NONE;
        }
        return ACT_LEAKY_RELU;
    }
    return ACT_COUNT;
}

static bool convert_activation(Importer *im, const Node *n, ActivationFunction act) {
    const Value *x = layer_input(im, n, 0);
    if (!x) return false;
    NeuralNetwork *net = im->net;
    if (act == ACT_SOFTMAX) {
        int64_t axis = attr_int(n, "axis", -1);
        if ((!(x->h == 1 && x->w == 1) && !x->flat) || x->nhwc)
            return fail(im, "ONNX import: Softmax node '%.*s' over a map%.*s is not supported (only over vectors)",
                        n->name, (Str){"", 0});
        (void)axis;
    }
    LayerType type = x->layer ? net->shapes[x->layer].type : LAYER_DENSE;
    bool fusable = x->layer > 0 && net->act_func[x->layer - 1] == ACT_NONE && layer_readers(im, x->layer) == 1 &&
                   type != LAYER_MAX_POOL2D && type != LAYER_AVG_POOL2D && type != LAYER_UPSAMPLE;
    Value copy = *x;
    copy.alias = fusable;
    if (fusable) {
        net->act_func[x->layer - 1] = act;
    } else {
        /* a layer of its own: an addition of one input */
        uint32_t l = add_layers(.net = net, .inputs = {x->layer}, .input_count = 1, .act_func = act);
        if (!added(im, l, n)) return false;
        copy.layer = l;
    }
    Value *y = add_value(im, n->outputs[0]);
    if (!y) return false;
    Str name = y->name;
    *y = copy;
    y->name = name;
    y->constant = false;
    im->bias_open = false;
    return true;
}

static bool convert_conv(Importer *im, const Node *n) {
    const Value *x = layer_input(im, n, 0);
    if (!x) return false;
    if (x->flat) return fail(im, "ONNX import: Conv node '%.*s' reads a flattened tensor%.*s", n->name, (Str){"", 0});
    Value *W = find(im, n->inputs[1]);
    if (!W || !W->constant || W->ndims != 4)
        return fail(im, "ONNX import: Conv node '%.*s' needs constant 4-D weights%.*s", n->name, (Str){"", 0});
    uint32_t kh, kw, sh, sw, ph, pw;
    if (!window_attrs(im, n, &kh, &kw, &sh, &sw, &ph, &pw, x->h, x->w)) return false;
    const uint32_t OC = (uint32_t)W->dims[0], CG = (uint32_t)W->dims[1], groups = (uint32_t)attr_int(n, "group", 1);
    if (W->dims[2] != kh || W->dims[3] != kw || groups == 0 || CG * groups != x->c)
        return fail(im, "ONNX import: Conv node '%.*s' has weights that do not fit its input%.*s", n->name, (Str){"", 0});
    const Value *B = n->ninputs > 2 && n->inputs[2].n ? constant_input(im, n, 2, OC) : NULL;
    if (n->ninputs > 2 && n->inputs[2].n && !B) return false;
    int dtype;
    const uint8_t *src = constant_bytes(im, W, &dtype);
    float *scratch = (float *)malloc(spingalett_filters_scratch(CG, kh, kw) * sizeof(float));
    if (!src || !scratch) {
        free(scratch);
        return fail(im, "ONNX import: the weights of Conv node '%.*s' cannot be read%.*s", n->name, (Str){"", 0});
    }
    uint32_t l = conv2d(.net = im->net, .inputs = {x->layer}, .input_count = 1, .filters = OC, .kernel_h = kh,
                        .kernel_w = kw, .stride_h = sh, .stride_w = sw, .padding_h = ph, .padding_w = pw,
                        .groups = groups, .act_func = ACT_NONE, .weight_initialization = WEIGHT_INITIALIZATION_NONE);
    if (!added(im, l, n)) { free(scratch); return false; }
    /* [OC][CG][KH][KW] -> [OC][KH][KW][CG] */
    spingalett_import_filters(layer_weights(im->net, l), src, dtype, OC, CG, kh, kw, scratch);
    free(scratch);
    if (B) memcpy(layer_biases(im->net, l), B->data, OC * sizeof(float));
    im->bias_open = B == NULL;
    return layer_output(im, n, l) != NULL;
}

/* A dense layer y = x W^T + b from rows of W (out x in, row-major) as the reading of x requires:
   the columns of a flattened map reordered from (c, h, w) to (h, w, c). */
static bool dense_layer(Importer *im, const Node *n, const Value *x, Value *W, bool transposed, uint32_t out,
                        uint32_t in, const float *bias, float alpha, float beta) {
    if (in != x->c * x->h * x->w)
        return fail(im, "ONNX import: %.*s node '%.*s' has weights that do not fit its input", n->op, n->name);
    if (x->h * x->w > 1 && !x->flat)
        return fail(im, "ONNX import: %.*s node '%.*s' multiplies a map that is not flattened", n->op, n->name);
    const uint32_t C = x->c, HW = x->h * x->w;
    int dtype;
    const uint8_t *src = constant_bytes(im, W, &dtype);
    float *scratch = (float *)malloc(spingalett_dense_scratch(out, in, transposed) * sizeof(float));
    if (!src || !scratch) {
        free(scratch);
        return fail(im, "ONNX import: the weights of %.*s node '%.*s' cannot be read", n->op, n->name);
    }
    uint32_t l = layer(.net = im->net, .inputs = {x->layer}, .input_count = 1, .neurons_amount = out,
                       .act_func = ACT_NONE, .weight_initialization = WEIGHT_INITIALIZATION_NONE);
    if (!added(im, l, n)) { free(scratch); return false; }
    /* ONNX column k = (c, p) of a flattened map in c-major order; ours (p, c) */
    spingalett_import_dense(layer_weights(im->net, l), src, dtype, out, in, C, HW, transposed, alpha, scratch);
    free(scratch);
    float *b = layer_biases(im->net, l);
    for (uint32_t o = 0; o < out; o++) b[o] = bias ? beta * bias[o] : 0.0f;
    im->bias_open = bias == NULL;
    Value *y = layer_output(im, n, l);
    if (y) y->vector = true;
    return y != NULL;
}

static bool convert_gemm(Importer *im, const Node *n) {
    const Value *x = layer_input(im, n, 0);
    if (!x) return false;
    if (attr_int(n, "transA", 0) != 0)
        return fail(im, "ONNX import: Gemm node '%.*s' transposes its input (transA)%.*s", n->name, (Str){"", 0});
    Value *W = find(im, n->inputs[1]);
    if (!W || !W->constant || W->ndims != 2)
        return fail(im, "ONNX import: Gemm node '%.*s' needs constant 2-D weights%.*s", n->name, (Str){"", 0});
    bool transB = attr_int(n, "transB", 0) != 0;
    uint32_t out = (uint32_t)(transB ? W->dims[0] : W->dims[1]), in = (uint32_t)(transB ? W->dims[1] : W->dims[0]);
    const Value *C = n->ninputs > 2 && n->inputs[2].n ? constant_input(im, n, 2, 0) : NULL;
    if (n->ninputs > 2 && n->inputs[2].n && (!C || (C->count != out && C->count != 1)))
        return fail(im, "ONNX import: Gemm node '%.*s' has a bias of another size%.*s", n->name, (Str){"", 0});
    float *bias = NULL;
    if (C) {
        bias = (float *)malloc(out * sizeof(float));
        if (!bias) return fail(im, "ONNX import: out of memory%.*s%.*s", (Str){"", 0}, (Str){"", 0});
        for (uint32_t o = 0; o < out; o++) bias[o] = C->count == 1 ? C->data[0] : C->data[o];
    }
    bool ok = dense_layer(im, n, x, W, !transB, out, in, bias, attr_float(n, "alpha", 1.0f), attr_float(n, "beta", 1.0f));
    free(bias);
    return ok;
}

static bool convert_matmul(Importer *im, const Node *n) {
    const Value *x = layer_input(im, n, 0);
    if (!x) return false;
    Value *W = find(im, n->inputs[1]);
    if (!W || !W->constant || W->ndims != 2)
        return fail(im, "ONNX import: MatMul node '%.*s' needs a constant 2-D right operand%.*s", n->name, (Str){"", 0});
    return dense_layer(im, n, x, W, true, (uint32_t)W->dims[1], (uint32_t)W->dims[0], NULL, 1.0f, 1.0f);
}

static bool convert_add(Importer *im, const Node *n) {
    if (n->ninputs != 2) return fail(im, "ONNX import: Add node '%.*s' needs two inputs%.*s", n->name, (Str){"", 0});
    Value *a = find(im, n->inputs[0]), *b = find(im, n->inputs[1]);
    if (!a || !b) return fail(im, "ONNX import: an input of Add node '%.*s' is unknown%.*s", n->name, (Str){"", 0});
    if (a->constant && b->constant) return fail(im, "ONNX import: Add node '%.*s' of two constants%.*s", n->name, (Str){"", 0});
    if (a->constant || b->constant) {
        /* a bias: per output of the product just added (MatMul, Gemm or Conv without one) */
        const Value *x = a->constant ? b : a;
        Value *c = a->constant ? a : b;
        NeuralNetwork *net = im->net;
        LayerType type = x->layer ? net->shapes[x->layer].type : LAYER_DENSE;
        uint32_t rows = x->layer ? spingalett_weight_rows(net, x->layer - 1) : 0;
        if (!im->bias_open || x->layer + 1 != net->layers || (type != LAYER_DENSE && !spingalett_filters(type)) ||
            net->act_func[x->layer - 1] != ACT_NONE || layer_readers(im, x->layer) != 1 ||
            (c->count != rows && c->count != 1))
            return fail(im, "ONNX import: Add node '%.*s' adds a constant that is not the bias of a product%.*s", n->name,
                        (Str){"", 0});
        const float *add = constant_floats(im, c);
        if (!add) return false;
        float *bias = layer_biases(net, x->layer);
        for (uint32_t o = 0; o < rows; o++) bias[o] += c->count == 1 ? add[0] : add[o];
        im->bias_open = false;
        Value copy = *x;                /* before add_value, which may move the values */
        copy.alias = true;
        Value *y = add_value(im, n->outputs[0]);
        if (!y) return false;
        Str name = y->name;
        *y = copy;
        y->name = name;
        return true;
    }
    if (a->flat != b->flat || a->nhwc != b->nhwc || a->c != b->c || a->h != b->h || a->w != b->w)
        return fail(im, "ONNX import: Add node '%.*s' adds tensors of different shapes (broadcasting is not supported)%.*s",
                    n->name, (Str){"", 0});
    const bool flat = a->flat, vector = a->vector && b->vector, nhwc = a->nhwc;
    uint32_t l = add_layers(.net = im->net, .inputs = {a->layer, b->layer}, .input_count = 2, .act_func = ACT_NONE);
    if (!added(im, l, n)) return false;
    im->bias_open = false;
    Value *y = layer_output(im, n, l);
    if (y) y->flat = flat, y->vector = vector, y->nhwc = nhwc;
    return y != NULL;
}

static bool convert_concat(Importer *im, const Node *n) {
    if (n->ninputs < 1 || n->ninputs > SPINGALETT_MAX_INPUTS)
        return fail(im, "ONNX import: Concat node '%.*s' joins 1 to 16 inputs%.*s", n->name, (Str){"", 0});
    LayerArgs args = {.net = im->net, .type = LAYER_CONCAT, .input_count = (uint32_t)n->ninputs, .act_func = ACT_NONE};
    bool flat = false, maps = false, vector = true;
    for (int k = 0; k < n->ninputs; k++) {
        const Value *v = layer_input(im, n, k);
        if (!v) return false;
        args.inputs[k] = v->layer;
        flat |= v->flat;
        maps |= v->h * v->w > 1;
        vector &= v->vector;
    }
    int64_t axis = attr_int(n, "axis", 1);
    if (flat || (maps ? axis != 1 && axis != -3 : axis != 1 && axis != -1))
        return fail(im, "ONNX import: Concat node '%.*s' joins along another axis than the channels%.*s", n->name,
                    (Str){"", 0});
    uint32_t l = layer_struct_arguments(args);
    if (!added(im, l, n)) return false;
    im->bias_open = false;
    Value *y = layer_output(im, n, l);
    if (y) y->vector = vector;
    return y != NULL;
}

static bool convert_pool(Importer *im, const Node *n, bool max) {
    const Value *x = layer_input(im, n, 0);
    if (!x) return false;
    if (x->flat) return fail(im, "ONNX import: %.*s node '%.*s' pools a flattened tensor", n->op, n->name);
    uint32_t kh, kw, sh, sw, ph, pw;
    if (!window_attrs(im, n, &kh, &kw, &sh, &sw, &ph, &pw, x->h, x->w)) return false;
    if (!max && (ph || pw) && attr_int(n, "count_include_pad", 0) != 0)
        return fail(im, "ONNX import: AveragePool node '%.*s' counts padding cells (count_include_pad)%.*s", n->name,
                    (Str){"", 0});
    if (max && attr_int(n, "storage_order", 0) != 0)
        return fail(im, "ONNX import: MaxPool node '%.*s' stores column-major%.*s", n->name, (Str){"", 0});
    uint32_t l = layer_struct_arguments((LayerArgs){.net = im->net, .type = max ? LAYER_MAX_POOL2D : LAYER_AVG_POOL2D,
                                                    .inputs = {x->layer}, .input_count = 1, .kernel_h = kh,
                                                    .kernel_w = kw, .stride_h = sh, .stride_w = sw, .padding_h = ph,
                                                    .padding_w = pw});
    if (!added(im, l, n)) return false;
    im->bias_open = false;
    return layer_output(im, n, l) != NULL;
}

static bool convert_batch_norm(Importer *im, const Node *n) {
    const Value *x = layer_input(im, n, 0);
    if (!x) return false;
    if (x->flat && x->h * x->w > 1)
        return fail(im, "ONNX import: BatchNormalization node '%.*s' normalizes a flattened map%.*s", n->name, (Str){"", 0});
    if (attr_int(n, "training_mode", 0) != 0)
        return fail(im, "ONNX import: BatchNormalization node '%.*s' is in training mode%.*s", n->name, (Str){"", 0});
    const Value *p[4];
    for (int k = 0; k < 4; k++)
        if (!(p[k] = constant_input(im, n, k + 1, x->c))) return false;
    float eps = attr_float(n, "epsilon", 1e-5f), momentum = attr_float(n, "momentum", 0.9f);
    uint32_t l = batch_norm(.net = im->net, .inputs = {x->layer}, .input_count = 1, .act_func = ACT_NONE,
                            .epsilon = eps > 0.0f && eps < 1.0f ? eps : 1e-5f,
                            .momentum = momentum > 0.0f && momentum < 1.0f ? 1.0f - momentum : 0.1f);
    if (!added(im, l, n)) return false;
    NeuralNetwork *net = im->net;
    const uint64_t b = net->bias_offsets[l - 1];
    const uint32_t C = x->c;
    const bool flat = x->flat, vector = x->vector;
    memcpy(layer_weights(net, l), p[0]->data, C * sizeof(float));
    memcpy(layer_biases(net, l), p[1]->data, C * sizeof(float));
    memcpy(net->running_mean + b, p[2]->data, C * sizeof(float));
    memcpy(net->running_var + b, p[3]->data, C * sizeof(float));
    im->bias_open = false;
    Value *y = layer_output(im, n, l);
    if (y) y->flat = flat, y->vector = vector;
    return y != NULL;
}

/* The same layer under another name (Identity, Dropout at inference) or flattened (Flatten,
   Reshape to two dimensions). */
static bool alias(Importer *im, const Node *n, bool flatten) {
    const Value *x = find(im, n->inputs[0]);
    if (!x) return fail(im, "ONNX import: input '%.*s' of %.*s is unknown", n->inputs[0], n->op);
    if (flatten && x->nhwc)
        return fail(im, "ONNX import: %.*s flattens '%.*s', a map transposed to channels last", n->op, n->inputs[0]);
    Value copy = *x;
    if (flatten && !x->constant) copy.flat = true, copy.vector = true;
    copy.alias = !x->constant;
    copy.owned = false;             /* a renamed constant shares the data (freed with the values) */
    Value *y = add_value(im, n->outputs[0]);
    if (!y) return false;
    Str name = y->name;
    *y = copy;
    y->name = name;
    return true;
}

static bool convert_reshape(Importer *im, const Node *n) {
    const Value *x = find(im, n->inputs[0]);
    Value *shape = n->ninputs > 1 ? find(im, n->inputs[1]) : NULL;
    if (!x || x->constant || !shape || !shape->constant || x->nhwc)
        return fail(im, "ONNX import: Reshape node '%.*s' needs a layer output and a constant shape%.*s", n->name, (Str){"", 0});
    const float *dims = constant_floats(im, shape);
    if (!dims) return false;
    /* to [N, everything]: a flattening; to the shape it has: nothing */
    uint64_t units = (uint64_t)x->c * x->h * x->w;
    if (shape->count == 2 && (dims[1] == -1.0f || (uint64_t)dims[1] == units))
        return alias(im, n, true);
    if (shape->count == 4 && (uint64_t)dims[1] == x->c && (uint64_t)dims[2] == x->h && (uint64_t)dims[3] == x->w)
        return alias(im, n, false);
    return fail(im, "ONNX import: Reshape node '%.*s' changes the shape other than by flattening it%.*s", n->name,
                (Str){"", 0});
}

static bool convert_constant(Importer *im, const Node *n) {
    const Attr *a = attr(n, "value");
    if (!a || !a->has_t) {
        const Attr *f = attr(n, "value_float"), *i = attr(n, "value_ints"), *fs = attr(n, "value_floats");
        const Attr *one = attr(n, "value_int");
        if ((i && i->nints > MAX_INTS) || (fs && fs->nfloats > MAX_DIMS))
            return fail(im, "ONNX import: Constant node '%.*s' holds a longer list than the importer reads%.*s", n->name,
                        (Str){"", 0});
        Value *v = add_value(im, n->outputs[0]);
        if (!v) return false;
        v->constant = true;
        if (f || one) {
            v->count = 1;
            v->ndims = 0;
            v->ctype = f ? ONNX_FLOAT : ONNX_INT64;
            v->data = (float *)malloc(sizeof(float));
            v->owned = true;
            if (v->data) v->data[0] = f ? f->f : (float)one->i;
        } else if (i || fs) {
            const int count = i ? i->nints : fs->nfloats;
            v->count = (uint64_t)count;
            v->ndims = 1;
            v->dims[0] = count;
            v->ctype = i ? ONNX_INT64 : ONNX_FLOAT;
            v->data = (float *)malloc((size_t)(count ? count : 1) * sizeof(float));
            v->owned = true;
            for (int k = 0; v->data && k < count; k++) v->data[k] = i ? (float)i->ints[k] : fs->floats[k];
        } else {
            return fail(im, "ONNX import: Constant node '%.*s' holds no tensor%.*s", n->name, (Str){"", 0});
        }
        return v->data != NULL;
    }
    TensorRef t;
    read_tensor(a->t, &t);
    if (t.ndims > MAX_DIMS || tensor_count(&t) > tensor_capacity(&t))
        return fail(im, "ONNX import: Constant node '%.*s' holds less data than its shape needs%.*s", n->name, (Str){"", 0});
    Value *v = add_value(im, n->outputs[0]);
    if (!v) return false;
    v->constant = true;
    v->ndims = t.ndims > MAX_DIMS ? MAX_DIMS : t.ndims;
    memcpy(v->dims, t.dims, sizeof v->dims);
    v->count = tensor_count(&t);
    v->ctype = t.type;
    v->tensor = t;                  /* decoded when something reads it */
    return true;
}

/* The same layer's output under a new name: another view of it (a transposition). */
static bool view(Importer *im, const Node *n, const Value *x, bool nhwc) {
    Value copy = *x;
    copy.alias = true;
    copy.nhwc = nhwc;
    Value *y = add_value(im, n->outputs[0]);
    if (!y) return false;
    Str name = y->name;
    *y = copy;
    y->name = name;
    return true;
}

/* A map to channels last and back: around layer normalization over its channels (LayerNorm2d). */
static bool convert_transpose(Importer *im, const Node *n) {
    const Value *x = layer_input(im, n, 0);
    if (!x) return false;
    const Attr *p = attr(n, "perm");
    const bool to_last = p && p->nints == 4 && p->ints[0] == 0 && p->ints[1] == 2 && p->ints[2] == 3 && p->ints[3] == 1;
    const bool to_first = p && p->nints == 4 && p->ints[0] == 0 && p->ints[1] == 3 && p->ints[2] == 1 && p->ints[3] == 2;
    if (x->flat || x->vector || !(x->nhwc ? to_first : to_last))
        return fail(im, "ONNX import: Transpose node '%.*s' permutes other axes than a map's channels to the last "
                    "place and back%.*s", n->name, (Str){"", 0});
    return view(im, n, x, to_last);
}

/* ConvTranspose: weights [C_in][C_out / group][KH][KW]. */
static bool convert_conv_transpose(Importer *im, const Node *n) {
    const Value *x = layer_input(im, n, 0);
    if (!x) return false;
    if (x->flat || x->vector)
        return fail(im, "ONNX import: ConvTranspose node '%.*s' reads no map%.*s", n->name, (Str){"", 0});
    Value *W = find(im, n->inputs[1]);
    if (!W || !W->constant || W->ndims != 4)
        return fail(im, "ONNX import: ConvTranspose node '%.*s' needs constant 4-D weights%.*s", n->name, (Str){"", 0});
    const uint32_t IC = (uint32_t)W->dims[0], OG = (uint32_t)W->dims[1], kh = (uint32_t)W->dims[2];
    const uint32_t kw = (uint32_t)W->dims[3], groups = (uint32_t)attr_int(n, "group", 1);
    const Attr *ks = attr(n, "kernel_shape"), *st = attr(n, "strides"), *pd = attr(n, "pads");
    const Attr *dl = attr(n, "dilations"), *op = attr(n, "output_padding"), *os = attr(n, "output_shape");
    const Attr *ap = attr(n, "auto_pad");
    if (groups == 0 || IC != x->c || IC % groups != 0 || OG == 0 ||
        (ks && (ks->nints != 2 || ks->ints[0] != kh || ks->ints[1] != kw)))
        return fail(im, "ONNX import: ConvTranspose node '%.*s' has weights that do not fit its input%.*s", n->name,
                    (Str){"", 0});
    if (dl && (dl->nints != 2 || dl->ints[0] != 1 || dl->ints[1] != 1))
        return fail(im, "ONNX import: ConvTranspose node '%.*s' has dilations, which are not supported%.*s", n->name,
                    (Str){"", 0});
    if (ap && ap->s.n && !str_is(ap->s, "NOTSET") && !str_is(ap->s, "VALID"))
        return fail(im, "ONNX import: ConvTranspose node '%.*s' pads by auto_pad, which is not supported%.*s", n->name,
                    (Str){"", 0});
    const bool strided = st && st->nints == 2, padded_out = op && op->nints == 2;
    const uint32_t sh = strided ? (uint32_t)st->ints[0] : 1u, sw = strided ? (uint32_t)st->ints[1] : 1u;
    uint32_t ph = 0, pw = 0;
    if (pd && pd->nints == 4) {
        if (pd->ints[0] != pd->ints[2] || pd->ints[1] != pd->ints[3])
            return fail(im, "ONNX import: ConvTranspose node '%.*s' pads asymmetrically, which is not supported%.*s",
                        n->name, (Str){"", 0});
        ph = (uint32_t)pd->ints[0];
        pw = (uint32_t)pd->ints[1];
    }
    const uint32_t oph = padded_out ? (uint32_t)op->ints[0] : 0u, opw = padded_out ? (uint32_t)op->ints[1] : 0u;
    /* an output shape: only the one the strides, padding and output padding give */
    const int64_t oh = ((int64_t)x->h - 1) * sh - 2 * (int64_t)ph + kh + oph;
    const int64_t ow = ((int64_t)x->w - 1) * sw - 2 * (int64_t)pw + kw + opw;
    if (os && !(os->nints >= 2 && os->ints[os->nints - 2] == oh && os->ints[os->nints - 1] == ow))
        return fail(im, "ONNX import: ConvTranspose node '%.*s' asks for an output shape (output_shape) its padding "
                    "does not give%.*s", n->name, (Str){"", 0});
    const uint32_t OC = OG * groups;
    const Value *B = n->ninputs > 2 && n->inputs[2].n ? constant_input(im, n, 2, OC) : NULL;
    if (n->ninputs > 2 && n->inputs[2].n && !B) return false;
    int dtype;
    const uint8_t *src = constant_bytes(im, W, &dtype);
    float *scratch = (float *)malloc(spingalett_transposed_filters_scratch(OG, kh, kw) * sizeof(float));
    if (!src || !scratch) {
        free(scratch);
        return fail(im, "ONNX import: the weights of ConvTranspose node '%.*s' cannot be read%.*s", n->name,
                    (Str){"", 0});
    }
    uint32_t l = conv_transpose2d(.net = im->net, .inputs = {x->layer}, .input_count = 1, .filters = OC, .kernel_h = kh,
                                  .kernel_w = kw, .stride_h = sh, .stride_w = sw, .padding_h = ph, .padding_w = pw,
                                  .output_padding_h = oph, .output_padding_w = opw, .groups = groups,
                                  .act_func = ACT_NONE, .weight_initialization = WEIGHT_INITIALIZATION_NONE);
    if (!added(im, l, n)) { free(scratch); return false; }
    spingalett_import_transposed_filters(layer_weights(im->net, l), src, dtype, IC, OG, groups, kh, kw, scratch);
    free(scratch);
    if (B) memcpy(layer_biases(im->net, l), B->data, OC * sizeof(float));
    im->bias_open = B == NULL;
    return layer_output(im, n, l) != NULL;
}

/* Upsampling by integer factors: Resize (scales or sizes; nearest, or linear between the cells'
   centres), and Upsample (opsets 7 to 9, and Resize of opset 10, two inputs: nearest). Nearest
   reads input cell floor(o / factor): asymmetric coordinates rounded down, or the cells' centres
   rounded to the nearest. */
static bool convert_resize(Importer *im, const Node *n) {
    const Value *x = layer_input(im, n, 0);
    if (!x) return false;
    if (x->flat || x->vector)
        return fail(im, "ONNX import: %.*s node '%.*s' resizes no map", n->op, n->name);
    const bool old = str_is(n->op, "Upsample") || n->ninputs == 2;     /* the operators before opset 11 */
    const Attr *ma = attr(n, "mode");
    const Str mode = ma && ma->s.n ? ma->s : (Str){"nearest", 7};
    const bool linear = str_is(mode, "linear") || str_is(mode, "bilinear");
    if (!linear && !str_is(mode, "nearest"))
        return fail(im, "ONNX import: %.*s node '%.*s' interpolates other than by nearest or linear", n->op, n->name);
    /* the factors of the four axes: scales (an attribute of Upsample-7, an input), or sizes over the input's */
    float scale[4] = {1.0f, 1.0f, 0.0f, 0.0f};
    const float in[4] = {1.0f, (float)x->c, (float)x->h, (float)x->w};
    int64_t axes[4] = {0, 1, 2, 3};
    int naxes = 4;
    const Attr *aa = attr(n, "axes");
    if (aa && aa->nints >= 1 && aa->nints <= 4) {
        naxes = aa->nints;
        for (int k = 0; k < naxes; k++) axes[k] = aa->ints[k] < 0 ? aa->ints[k] + 4 : aa->ints[k];
    }
    bool have = false;
    const Attr *sa = attr(n, "scales");
    if (sa && sa->nfloats == 4) {
        for (int k = 0; k < 4; k++) scale[k] = sa->floats[k];
        have = true;
    }
    const int si = old ? 1 : 2;
    Value *sv = !have && si < n->ninputs && n->inputs[si].n ? find(im, n->inputs[si]) : NULL;
    Value *zv = !old && n->ninputs > 3 && n->inputs[3].n ? find(im, n->inputs[3]) : NULL;
    if (sv && sv->constant && sv->count == (uint64_t)naxes) {
        const float *f = constant_floats(im, sv);
        if (!f) return false;
        for (int k = 0; k < naxes; k++)
            if (axes[k] >= 0 && axes[k] < 4) scale[axes[k]] = f[k];
        have = true;
    } else if (zv && zv->constant && zv->count == (uint64_t)naxes) {
        const float *f = constant_floats(im, zv);
        if (!f) return false;
        for (int k = 0; k < naxes; k++)
            if (axes[k] >= 1 && axes[k] < 4) scale[axes[k]] = f[k] / in[axes[k]];
        have = true;
    }
    const float sh = scale[2], sw = scale[3];
    if (!have || scale[0] != 1.0f || scale[1] != 1.0f || sh < 1.0f || sw < 1.0f || sh > 65535.0f || sw > 65535.0f ||
        sh != floorf(sh) || sw != floorf(sw))
        return fail(im, "ONNX import: %.*s node '%.*s' resizes other than by constant integer factors of a map's height "
                    "and width", n->op, n->name);
    const uint32_t fh = (uint32_t)sh, fw = (uint32_t)sw, most = fh > fw ? fh : fw;
    /* the coordinates: what gives floor(o / factor) for nearest, the cells' centres for linear */
    const Attr *ca = attr(n, "coordinate_transformation_mode"), *na = attr(n, "nearest_mode");
    const Str ctm = old ? (Str){"asymmetric", 10} : ca && ca->s.n ? ca->s : (Str){"half_pixel", 10};
    const Str nm = old ? (Str){"floor", 5} : na && na->s.n ? na->s : (Str){"round_prefer_floor", 18};
    const bool centres = str_is(ctm, "half_pixel") || str_is(ctm, "pytorch_half_pixel") ||
                         str_is(ctm, "half_pixel_symmetric");
    bool fits;
    if (linear) fits = centres;
    else if (centres) fits = str_is(nm, "round_prefer_floor") || str_is(nm, "round_prefer_ceil") || most == 1;
    else if (str_is(ctm, "asymmetric"))
        fits = str_is(nm, "floor") || (str_is(nm, "round_prefer_floor") && most <= 2) || most == 1;
    else fits = most == 1;
    if (!fits)
        return fail(im, "ONNX import: %.*s node '%.*s' maps coordinates other than as the upsampling layer does "
                    "(nearest: floor of asymmetric or rounded half-pixel coordinates; linear: half-pixel)", n->op,
                    n->name);
    if (fh == 1 && fw == 1) return view(im, n, x, false);
    uint32_t l = upsample2d(.net = im->net, .inputs = {x->layer}, .input_count = 1, .stride_h = fh, .stride_w = fw,
                            .upsample = linear ? UPSAMPLE_BILINEAR : UPSAMPLE_NEAREST);
    if (!added(im, l, n)) return false;
    im->bias_open = false;
    return layer_output(im, n, l) != NULL;
}

/* LayerNormalization over a vector, or over the channels of a map transposed to channels last. */
static bool convert_layer_norm(Importer *im, const Node *n) {
    const Value *x = layer_input(im, n, 0);
    if (!x) return false;
    const int64_t axis = attr_int(n, "axis", -1);
    bool channels;
    if (x->nhwc) channels = axis == -1 || axis == 3;
    else if (x->vector || x->flat) channels = x->h * x->w == 1 && (axis == -1 || axis == 1);
    else channels = x->h * x->w == 1 && (axis == 1 || axis == -3);    /* [N, C, 1, 1] */
    if (!channels)
        return fail(im, "ONNX import: LayerNormalization node '%.*s' normalizes other axes than a vector or a map's "
                    "channels (transposed to the last place)%.*s", n->name, (Str){"", 0});
    const Value *gamma = constant_input(im, n, 1, x->c);
    if (!gamma) return false;
    const Value *beta = n->ninputs > 2 && n->inputs[2].n ? constant_input(im, n, 2, x->c) : NULL;
    if (n->ninputs > 2 && n->inputs[2].n && !beta) return false;
    const float eps = attr_float(n, "epsilon", 1e-5f);
    const bool flat = x->flat, vector = x->vector, nhwc = x->nhwc;
    const uint32_t C = x->c;
    uint32_t l = layer_norm(.net = im->net, .inputs = {x->layer}, .input_count = 1, .act_func = ACT_NONE,
                            .epsilon = eps > 0.0f && eps < 1.0f ? eps : 1e-5f);
    if (!added(im, l, n)) return false;
    memcpy(layer_weights(im->net, l), gamma->data, C * sizeof(float));
    if (beta) memcpy(layer_biases(im->net, l), beta->data, C * sizeof(float));
    else memset(layer_biases(im->net, l), 0, C * sizeof(float));
    im->bias_open = false;
    Value *y = layer_output(im, n, l);
    if (y) y->flat = flat, y->vector = vector, y->nhwc = nhwc;
    return y != NULL;
}

/* ------------------------------------------------------------------------- errors */

static bool vfail(Importer *im, Str op, Str name, const char *fmt, va_list ap) {
    if (im->error[0]) return false;
    const int k = snprintf(im->error, sizeof im->error, "ONNX import: %.*s node '%.*s' ", (int)op.n, op.s, (int)name.n,
                           name.s);
    if (k > 0 && (size_t)k < sizeof im->error) vsnprintf(im->error + k, sizeof im->error - (size_t)k, fmt, ap);
    return false;
}

/* An error naming the node: "ONNX import: <op> node '<name>' " and the message. */
static bool node_fail(Importer *im, const Node *n, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vfail(im, n->op, n->name, fmt, ap);
    va_end(ap);
    return false;
}

static bool out_of_memory(Importer *im) {
    return fail(im, "ONNX import: out of memory%.*s%.*s", (Str){"", 0}, (Str){"", 0});
}

/* A value's shape in the file with a batch of one, for Shape; its rank. */
static int value_shape(const Value *v, int64_t *dims) {
    if (v->constant) {
        memcpy(dims, v->dims, sizeof v->dims);
        return v->ndims;
    }
    dims[0] = 1;
    if (v->vector || v->flat) {
        dims[1] = (int64_t)v->c * v->h * v->w;
        return 2;
    }
    if (v->nhwc) dims[1] = v->h, dims[2] = v->w, dims[3] = v->c;
    else dims[1] = v->c, dims[2] = v->h, dims[3] = v->w;
    return 4;
}

/* ------------------------------------------------------------------------- constants evaluated */

/* Values a computed constant may hold: 2^26, or as many as its node's inputs hold (a transposed weight
   matrix), so that a small file cannot make a large constant. */
#define FOLD_LIMIT ((uint64_t)1 << 26)

typedef struct {
    const float *d;
    int64_t dims[MAX_DIMS];
    int nd;
    uint64_t n;
    int32_t type;
} Const;

static bool int_type(int32_t t) {
    return t == 2 || t == 3 || t == 4 || t == 5 || t == 6 || t == 7 || t == 12 || t == 13;
}

/* A constant's value as an integer, clamped to int64's range (NaN: 0): the files hold any float. */
static int64_t to_i64(float f) {
    if (!(f == f)) return 0;
    if (f >= 9.2e18f) return INT64_MAX;
    if (f <= -9.2e18f) return INT64_MIN;
    return (int64_t)f;
}

/* Input k as a constant, decoded; false (error set) when it is none or holds no data. */
static bool const_arg(Importer *im, const Node *n, int k, Const *c) {
    Value *v = k < n->ninputs && n->inputs[k].n ? find(im, n->inputs[k]) : NULL;
    if (!v || !v->constant) return node_fail(im, n, "needs a constant as its input %d", k);
    const float *d = constant_floats(im, v);
    if (!d) return false;
    c->d = d;
    c->nd = v->ndims;
    memcpy(c->dims, v->dims, sizeof c->dims);
    c->n = v->count;
    c->type = v->ctype ? v->ctype : ONNX_FLOAT;
    for (int i = 0; i < c->nd; i++)
        if (c->dims[i] < 0) return node_fail(im, n, "reads a constant of a negative size");
    return true;
}

/* An optional constant input: false only on an error (absent: c->n = 0, c->d = NULL). */
static bool const_opt(Importer *im, const Node *n, int k, Const *c) {
    memset(c, 0, sizeof *c);
    if (k >= n->ninputs || n->inputs[k].n == 0) return true;
    return const_arg(im, n, k, c);
}

static uint64_t dims_count(const int64_t *dims, int nd) {
    uint64_t count = 1;
    for (int i = 0; i < nd; i++) {
        if (dims[i] < 0) return UINT64_MAX;
        if (dims[i] && count > UINT64_MAX / (uint64_t)dims[i]) return UINT64_MAX;
        count *= (uint64_t)dims[i];
    }
    return count;
}

/* Room for a computed constant of count values; NULL (error set) when it would be too large. */
static float *fold_alloc(Importer *im, const Node *n, uint64_t count) {
    uint64_t inputs = 0;
    for (int k = 0; k < n->ninputs && k < MAX_NODE_INPUTS; k++) {
        const Value *v = n->inputs[k].n ? find(im, n->inputs[k]) : NULL;
        if (v && v->constant) inputs += v->count;
    }
    if (count > FOLD_LIMIT && count > inputs) {
        node_fail(im, n, "computes a constant of more values than 2^26 and its inputs");
        return NULL;
    }
    float *d = (float *)malloc((size_t)(count ? count : 1) * sizeof(float));
    if (!d) out_of_memory(im);
    return d;
}

/* Output k as a computed constant: data (owned from here on) of the shape dims. */
static bool const_out(Importer *im, const Node *n, int k, float *data, const int64_t *dims, int nd, int32_t type) {
    Value *v = k < n->noutputs && k < MAX_NODE_OUTPUTS ? add_value(im, n->outputs[k]) : NULL;
    if (!v) {
        free(data);
        return k < n->noutputs ? out_of_memory(im) : node_fail(im, n, "lacks an output");
    }
    v->constant = true;
    v->data = data;
    v->owned = true;
    v->ndims = nd;
    memcpy(v->dims, dims, (size_t)nd * sizeof *dims);
    v->count = dims_count(dims, nd);
    v->ctype = type;
    return true;
}

/* numpy's broadcasting of two shapes; false when they do not broadcast. */
static bool broadcast(const int64_t *a, int na, const int64_t *b, int nb, int64_t *out, int *no) {
    const int n = na > nb ? na : nb;
    if (n > MAX_DIMS) return false;
    for (int i = 0; i < n; i++) {
        const int64_t x = i - (n - na) >= 0 ? a[i - (n - na)] : 1, y = i - (n - nb) >= 0 ? b[i - (n - nb)] : 1;
        if (x != y && x != 1 && y != 1) return false;
        out[i] = x == 1 ? y : x;
    }
    *no = n;
    return true;
}

/* The index into a tensor of shape dims (broadcast to out) of out's element i. */
static uint64_t bcast_at(const int64_t *dims, int nd, const int64_t *out, int no, uint64_t i) {
    uint64_t at = 0, stride = 1;
    for (int d = no - 1; d >= 0; d--) {
        const uint64_t size = (uint64_t)out[d], idx = i % size;
        i /= size;
        const int sd = d - (no - nd);
        if (sd >= 0) {
            if (dims[sd] != 1) at += idx * stride;
            stride *= (uint64_t)dims[sd];
        }
    }
    return at;
}

enum { F_ADD, F_SUB, F_MUL, F_DIV, F_POW, F_MOD, F_MAX, F_MIN, F_EQ, F_LT, F_LE, F_GT, F_GE, F_AND, F_OR, F_XOR };

static double fold2(int op, double x, double y, bool integer, bool cmod, bool *ok) {
    switch (op) {
        case F_ADD: return x + y;
        case F_SUB: return x - y;
        case F_MUL: return x * y;
        case F_DIV:
            if (integer) {
                if (y == 0.0) { *ok = false; return 0.0; }
                return trunc(x / y);
            }
            return x / y;
        case F_POW: return pow(x, y);
        case F_MOD:
            if (y == 0.0) { *ok = !integer; return integer ? 0.0 : NAN; }
            return cmod ? fmod(x, y) : x - floor(x / y) * y;
        case F_MAX: return x > y ? x : y;
        case F_MIN: return x < y ? x : y;
        case F_EQ: return x == y;
        case F_LT: return x < y;
        case F_LE: return x <= y;
        case F_GT: return x > y;
        case F_GE: return x >= y;
        case F_AND: return x != 0.0 && y != 0.0;
        case F_OR: return x != 0.0 || y != 0.0;
        default: return (x != 0.0) != (y != 0.0);
    }
}

static bool fold_binary(Importer *im, const Node *n, int op) {
    Const a, b;
    if (n->ninputs != 2) return node_fail(im, n, "needs two inputs");
    if (!const_arg(im, n, 0, &a) || !const_arg(im, n, 1, &b)) return false;
    int64_t dims[MAX_DIMS];
    int nd;
    if (!broadcast(a.dims, a.nd, b.dims, b.nd, dims, &nd)) return node_fail(im, n, "reads shapes that do not broadcast");
    const uint64_t count = dims_count(dims, nd);
    float *out = fold_alloc(im, n, count);
    if (!out) return false;
    const bool integer = int_type(a.type), cmod = attr_int(n, "fmod", 0) != 0;
    bool ok = true;
    for (uint64_t i = 0; i < count; i++) {
        double r = fold2(op, a.d[bcast_at(a.dims, a.nd, dims, nd, i)], b.d[bcast_at(b.dims, b.nd, dims, nd, i)],
                         integer, cmod, &ok);
        out[i] = (float)(integer && op <= F_MIN ? trunc(r) : r);
    }
    if (!ok) { free(out); return node_fail(im, n, "divides integers by zero"); }
    return const_out(im, n, 0, out, dims, nd, op >= F_EQ ? ONNX_BOOL : a.type);
}

static const char *const unary_names[] = {"Neg", "Sqrt", "Reciprocal", "Exp", "Log", "Cos", "Sin", "Abs", "Floor",
                                          "Ceil", "Round", "Not", "Sign", "Relu", "Sigmoid", "Tanh", "Erf"};

/* The unary operator evaluated on constants (its index in unary_names), or -1. */
static int unary_fold_op(Str op) {
    for (int k = 0; k < (int)(sizeof unary_names / sizeof unary_names[0]); k++)
        if (str_is(op, unary_names[k])) return k;
    return -1;
}

static double fold1(int op, double x) {
    switch (op) {
        case 0: return -x;
        case 1: return sqrt(x);
        case 2: return 1.0 / x;
        case 3: return exp(x);
        case 4: return log(x);
        case 5: return cos(x);
        case 6: return sin(x);
        case 7: return fabs(x);
        case 8: return floor(x);
        case 9: return ceil(x);
        case 10: return nearbyint(x);
        case 11: return x == 0.0;
        case 12: return (x > 0.0) - (x < 0.0);
        case 13: return x > 0.0 ? x : 0.0;
        case 14: return 1.0 / (1.0 + exp(-x));
        case 15: return tanh(x);
        default: return erf(x);
    }
}

static bool fold_unary(Importer *im, const Node *n) {
    Const a;
    if (!const_arg(im, n, 0, &a)) return false;
    float *out = fold_alloc(im, n, a.n);
    if (!out) return false;
    const int op = unary_fold_op(n->op);
    for (uint64_t i = 0; i < a.n; i++) out[i] = (float)fold1(op, a.d[i]);
    return const_out(im, n, 0, out, a.dims, a.nd, a.type);
}

static bool fold_cast(Importer *im, const Node *n) {
    Const a;
    if (!const_arg(im, n, 0, &a)) return false;
    const int32_t to = (int32_t)attr_int(n, "to", ONNX_FLOAT);
    float *out = fold_alloc(im, n, a.n);
    if (!out) return false;
    for (uint64_t i = 0; i < a.n; i++)
        out[i] = to == ONNX_BOOL ? (float)(a.d[i] != 0.0f) : int_type(to) ? truncf(a.d[i]) : a.d[i];
    return const_out(im, n, 0, out, a.dims, a.nd, to);
}

static bool fold_where(Importer *im, const Node *n) {
    Const c, x, y;
    if (!const_arg(im, n, 0, &c) || !const_arg(im, n, 1, &x) || !const_arg(im, n, 2, &y)) return false;
    int64_t t[MAX_DIMS], dims[MAX_DIMS];
    int nt, nd;
    if (!broadcast(c.dims, c.nd, x.dims, x.nd, t, &nt) || !broadcast(t, nt, y.dims, y.nd, dims, &nd))
        return node_fail(im, n, "reads shapes that do not broadcast");
    const uint64_t count = dims_count(dims, nd);
    float *out = fold_alloc(im, n, count);
    if (!out) return false;
    for (uint64_t i = 0; i < count; i++)
        out[i] = c.d[bcast_at(c.dims, c.nd, dims, nd, i)] != 0.0f ? x.d[bcast_at(x.dims, x.nd, dims, nd, i)]
                                                                  : y.d[bcast_at(y.dims, y.nd, dims, nd, i)];
    return const_out(im, n, 0, out, dims, nd, x.type);
}

/* An axis attribute or input normalized against rank r; false when outside it. */
static bool norm_axis(int64_t axis, int r, int *out) {
    if (axis < -r || axis >= r) return false;
    *out = (int)(axis < 0 ? axis + r : axis);
    return true;
}

static bool fold_shape(Importer *im, const Node *n) {
    Value *x = n->ninputs > 0 ? find(im, n->inputs[0]) : NULL;
    if (!x) return node_fail(im, n, "reads an unknown value");
    int64_t dims[MAX_DIMS];
    const int nd = value_shape(x, dims);
    int64_t start = attr_int(n, "start", 0), end = attr_int(n, "end", nd);
    if (start < 0) start += nd;
    if (end < 0) end += nd;
    start = start < 0 ? 0 : start > nd ? nd : start;
    end = end < start ? start : end > nd ? nd : end;
    float *out = fold_alloc(im, n, (uint64_t)(end - start));
    if (!out) return false;
    for (int64_t i = start; i < end; i++) out[i - start] = (float)dims[i];
    const int64_t len = end - start;
    return const_out(im, n, 0, out, &len, 1, ONNX_INT64);
}

static bool fold_size(Importer *im, const Node *n) {
    Value *x = n->ninputs > 0 ? find(im, n->inputs[0]) : NULL;
    if (!x) return node_fail(im, n, "reads an unknown value");
    int64_t dims[MAX_DIMS];
    const int nd = value_shape(x, dims);
    float *out = fold_alloc(im, n, 1);
    if (!out) return false;
    out[0] = (float)dims_count(dims, nd);
    return const_out(im, n, 0, out, dims, 0, ONNX_INT64);
}

static bool fold_gather(Importer *im, const Node *n) {
    Const d, ix;
    if (!const_arg(im, n, 0, &d) || !const_arg(im, n, 1, &ix)) return false;
    int axis;
    if (!norm_axis(attr_int(n, "axis", 0), d.nd, &axis)) return node_fail(im, n, "gathers along an axis it lacks");
    if (d.nd - 1 + ix.nd > MAX_DIMS) return node_fail(im, n, "gathers more than 8 axes");
    int64_t dims[MAX_DIMS];
    int nd = 0;
    for (int i = 0; i < axis; i++) dims[nd++] = d.dims[i];
    for (int i = 0; i < ix.nd; i++) dims[nd++] = ix.dims[i];
    for (int i = axis + 1; i < d.nd; i++) dims[nd++] = d.dims[i];
    const uint64_t outer = dims_count(d.dims, axis), inner = dims_count(d.dims + axis + 1, d.nd - axis - 1);
    const int64_t size = d.dims[axis];
    const uint64_t count = dims_count(dims, nd);
    float *out = fold_alloc(im, n, count);
    if (!out) return false;
    for (uint64_t o = 0, at = 0; o < outer; o++)
        for (uint64_t j = 0; j < ix.n; j++) {
            int64_t k = to_i64(ix.d[j]);
            if (k < 0) k += size;
            if (k < 0 || k >= size) { free(out); return node_fail(im, n, "gathers an index outside its data"); }
            memcpy(out + at, d.d + (o * (uint64_t)size + (uint64_t)k) * inner, (size_t)inner * sizeof(float));
            at += inner;
        }
    return const_out(im, n, 0, out, dims, nd, d.type);
}

/* The axes of Unsqueeze, Squeeze or ReduceMean: an attribute (older operator sets) or input 1. */
static bool axes_arg(Importer *im, const Node *n, int64_t *axes, int *count) {
    const Attr *a = attr(n, "axes");
    *count = 0;
    if (a) {
        if (a->nints > MAX_DIMS) return node_fail(im, n, "lists more than 8 axes");
        for (int k = 0; k < a->nints; k++) axes[k] = a->ints[k];
        *count = a->nints;
        return true;
    }
    Const c;
    if (!const_opt(im, n, 1, &c)) return false;
    if (c.n > MAX_DIMS) return node_fail(im, n, "lists more than 8 axes");
    for (uint64_t k = 0; k < c.n; k++) axes[k] = to_i64(c.d[k]);
    *count = (int)c.n;
    return true;
}

/* The shape after Unsqueeze (insert) or Squeeze of the given axes; false when they do not fit. */
static bool squeezed(const int64_t *dims, int nd, const int64_t *axes, int na, bool insert, int64_t *out, int *no) {
    bool mark[MAX_DIMS] = {false};
    const int r = insert ? nd + na : nd;
    if (r > MAX_DIMS) return false;
    for (int k = 0; k < na; k++) {
        int a;
        if (!norm_axis(axes[k], r, &a) || mark[a] || (!insert && dims[a] != 1)) return false;
        mark[a] = true;
    }
    int m = 0, j = 0;
    for (int i = 0; i < r; i++) {
        if (insert) out[m++] = mark[i] ? 1 : dims[j++];
        else if (!(na ? mark[i] : dims[i] == 1)) out[m++] = dims[i];
    }
    *no = m;
    return true;
}

static bool fold_squeeze(Importer *im, const Node *n, bool insert) {
    Const d;
    int64_t axes[MAX_DIMS], dims[MAX_DIMS];
    int na, nd;
    if (!const_arg(im, n, 0, &d) || !axes_arg(im, n, axes, &na)) return false;
    if (!squeezed(d.dims, d.nd, axes, na, insert, dims, &nd)) return node_fail(im, n, "has axes that do not fit");
    float *out = fold_alloc(im, n, d.n);
    if (!out) return false;
    memcpy(out, d.d, (size_t)d.n * sizeof(float));
    return const_out(im, n, 0, out, dims, nd, d.type);
}

static bool fold_concat(Importer *im, const Node *n) {
    Const c[MAX_NODE_INPUTS];
    if (n->ninputs < 1) return node_fail(im, n, "joins nothing");
    for (int k = 0; k < n->ninputs; k++)
        if (!const_arg(im, n, k, &c[k])) return false;
    int axis;
    if (!norm_axis(attr_int(n, "axis", 0), c[0].nd, &axis)) return node_fail(im, n, "joins along an axis it lacks");
    int64_t dims[MAX_DIMS];
    memcpy(dims, c[0].dims, sizeof dims);
    dims[axis] = 0;
    for (int k = 0; k < n->ninputs; k++) {
        if (c[k].nd != c[0].nd) return node_fail(im, n, "joins tensors of different ranks");
        for (int i = 0; i < c[0].nd; i++)
            if (i != axis && c[k].dims[i] != c[0].dims[i]) return node_fail(im, n, "joins tensors of other shapes");
        dims[axis] += c[k].dims[axis];
    }
    const uint64_t count = dims_count(dims, c[0].nd), outer = dims_count(dims, axis);
    float *out = fold_alloc(im, n, count);
    if (!out) return false;
    uint64_t at = 0;
    for (uint64_t o = 0; o < outer; o++)
        for (int k = 0; k < n->ninputs; k++) {
            const uint64_t chunk = dims_count(c[k].dims + axis, c[k].nd - axis);
            memcpy(out + at, c[k].d + o * chunk, (size_t)chunk * sizeof(float));
            at += chunk;
        }
    return const_out(im, n, 0, out, dims, c[0].nd, c[0].type);
}

/* Slice's starts, ends, axes and steps (inputs, or attributes before operator set 10), resolved against
   the shape dims into a first index, a step and a count per axis. */
static bool slice_args(Importer *im, const Node *n, const int64_t *dims, int nd, int64_t *first, int64_t *step,
                       int64_t *count) {
    int64_t s[MAX_DIMS], e[MAX_DIMS], ax[MAX_DIMS], st[MAX_DIMS];
    int ns;
    const Attr *as = attr(n, "starts");
    if (as) {
        const Attr *ae = attr(n, "ends"), *aa = attr(n, "axes");
        ns = as->nints;
        if (!ae || ae->nints != ns || ns > MAX_DIMS || (aa && aa->nints != ns))
            return node_fail(im, n, "has starts, ends and axes of different lengths");
        for (int k = 0; k < ns; k++) s[k] = as->ints[k], e[k] = ae->ints[k], ax[k] = aa ? aa->ints[k] : k, st[k] = 1;
    } else {
        Const cs, ce, ca, cst;
        if (!const_arg(im, n, 1, &cs) || !const_arg(im, n, 2, &ce) || !const_opt(im, n, 3, &ca) ||
            !const_opt(im, n, 4, &cst))
            return false;
        ns = (int)cs.n;
        if (ce.n != cs.n || cs.n > MAX_DIMS || (ca.d && ca.n != cs.n) || (cst.d && cst.n != cs.n))
            return node_fail(im, n, "has starts, ends, axes and steps of different lengths");
        for (int k = 0; k < ns; k++) {
            /* floats hold the int64 extremes exporters use for "to the end" as values beyond any axis */
            s[k] = to_i64(cs.d[k]);
            e[k] = to_i64(ce.d[k]);
            ax[k] = ca.d ? to_i64(ca.d[k]) : k;
            st[k] = cst.d ? to_i64(cst.d[k]) : 1;
        }
    }
    for (int i = 0; i < nd; i++) first[i] = 0, step[i] = 1, count[i] = dims[i];
    bool seen[MAX_DIMS] = {false};
    for (int k = 0; k < ns; k++) {
        int a;
        if (!norm_axis(ax[k], nd, &a) || seen[a] || st[k] == 0 || st[k] == INT64_MIN)
            return node_fail(im, n, "slices axes that do not fit");
        seen[a] = true;
        const int64_t size = dims[a];
        int64_t lo = s[k], hi = e[k];
        if (lo < 0) lo = lo == INT64_MIN ? -1 - size : lo + size;
        if (hi < 0) hi = hi == INT64_MIN ? -1 - size : hi + size;
        if (st[k] > 0) {
            lo = lo < 0 ? 0 : lo > size ? size : lo;
            hi = hi < 0 ? 0 : hi > size ? size : hi;
            count[a] = hi > lo ? (hi - lo + st[k] - 1) / st[k] : 0;
        } else {
            lo = lo < 0 ? -1 : lo > size - 1 ? size - 1 : lo;
            hi = hi < -1 ? -1 : hi > size - 1 ? size - 1 : hi;
            count[a] = lo > hi ? (lo - hi - st[k] - 1) / -st[k] : 0;
        }
        first[a] = lo;
        step[a] = st[k];
    }
    return true;
}

static bool fold_slice(Importer *im, const Node *n) {
    Const d;
    if (!const_arg(im, n, 0, &d)) return false;
    int64_t first[MAX_DIMS], step[MAX_DIMS], dims[MAX_DIMS];
    if (!slice_args(im, n, d.dims, d.nd, first, step, dims)) return false;
    const uint64_t count = dims_count(dims, d.nd);
    float *out = fold_alloc(im, n, count);
    if (!out) return false;
    for (uint64_t i = 0; i < count; i++) {
        uint64_t rest = i, at = 0, stride = 1;
        for (int a = d.nd - 1; a >= 0; a--) {
            const uint64_t idx = rest % (uint64_t)dims[a];
            rest /= (uint64_t)dims[a];
            at += (uint64_t)(first[a] + (int64_t)idx * step[a]) * stride;
            stride *= (uint64_t)d.dims[a];
        }
        out[i] = d.d[at];
    }
    return const_out(im, n, 0, out, dims, d.nd, d.type);
}

/* A Reshape's target for count elements of the shape dims: 0 copies the input's axis (unless
   allowzero), -1 takes what is left. */
static bool reshape_target(const float *shape, uint64_t ns, const int64_t *dims, int nd, uint64_t count,
                           bool allowzero, int64_t *out) {
    if (ns > MAX_DIMS) return false;
    int infer = -1;
    uint64_t known = 1;
    for (int i = 0; i < (int)ns; i++) {
        const float s = shape[i];
        if (s == 0.0f && !allowzero) {
            if (i >= nd) return false;
            out[i] = dims[i];
        } else if (s == -1.0f) {
            if (infer >= 0) return false;
            infer = i;
            out[i] = 1;
        } else if (s < 0.0f || s != floorf(s) || s > 1e12f) {
            return false;
        } else {
            out[i] = (int64_t)s;
        }
        if (out[i] && known > UINT64_MAX / 2 / (uint64_t)out[i]) return false;
        known *= (uint64_t)out[i];
    }
    if (infer >= 0) {
        if (known == 0 || count % known) return false;
        out[infer] = (int64_t)(count / known);
        return true;
    }
    return known == count;
}

static bool fold_reshape(Importer *im, const Node *n) {
    Const d, s;
    if (!const_arg(im, n, 0, &d)) return false;
    int64_t dims[MAX_DIMS];
    int nd;
    if (str_is(n->op, "Flatten")) {
        int axis;
        if (!norm_axis(attr_int(n, "axis", 1), d.nd + 1, &axis)) return node_fail(im, n, "flattens at an axis it lacks");
        dims[0] = (int64_t)dims_count(d.dims, axis);
        dims[1] = (int64_t)dims_count(d.dims + axis, d.nd - axis);
        nd = 2;
    } else {
        if (!const_arg(im, n, 1, &s)) return false;
        if (!reshape_target(s.d, s.n, d.dims, d.nd, d.n, attr_int(n, "allowzero", 0) != 0, dims))
            return node_fail(im, n, "reshapes to a shape of another size");
        nd = (int)s.n;
    }
    float *out = fold_alloc(im, n, d.n);
    if (!out) return false;
    memcpy(out, d.d, (size_t)d.n * sizeof(float));
    return const_out(im, n, 0, out, dims, nd, d.type);
}

/* A Transpose's permutation of rank r (by default reversed); false when it is none. */
static bool transpose_perm(const Node *n, int r, int64_t *perm) {
    const Attr *p = attr(n, "perm");
    bool seen[MAX_DIMS] = {false};
    if (p && p->nints != r) return false;
    for (int i = 0; i < r; i++) {
        perm[i] = p ? p->ints[i] : r - 1 - i;
        if (perm[i] < 0 || perm[i] >= r || seen[perm[i]]) return false;
        seen[perm[i]] = true;
    }
    return true;
}

static bool fold_transpose(Importer *im, const Node *n) {
    Const d;
    if (!const_arg(im, n, 0, &d)) return false;
    int64_t perm[MAX_DIMS], dims[MAX_DIMS], stride[MAX_DIMS];
    if (!transpose_perm(n, d.nd, perm)) return node_fail(im, n, "has no permutation of its input's axes");
    int64_t s = 1;
    for (int i = d.nd - 1; i >= 0; i--) stride[i] = s, s *= d.dims[i];
    for (int i = 0; i < d.nd; i++) dims[i] = d.dims[perm[i]];
    float *out = fold_alloc(im, n, d.n);
    if (!out) return false;
    for (uint64_t i = 0; i < d.n; i++) {
        uint64_t rest = i, at = 0;
        for (int a = d.nd - 1; a >= 0; a--) {
            at += (rest % (uint64_t)dims[a]) * (uint64_t)stride[perm[a]];
            rest /= (uint64_t)dims[a];
        }
        out[i] = d.d[at];
    }
    return const_out(im, n, 0, out, dims, d.nd, d.type);
}

static bool fold_expand(Importer *im, const Node *n) {
    Const d, s;
    if (!const_arg(im, n, 0, &d) || !const_arg(im, n, 1, &s)) return false;
    int64_t shape[MAX_DIMS], dims[MAX_DIMS];
    int nd;
    if (s.n > MAX_DIMS) return node_fail(im, n, "expands to more than 8 axes");
    for (uint64_t k = 0; k < s.n; k++) {
        if (!(s.d[k] >= 0.0f && s.d[k] <= 1e12f)) return node_fail(im, n, "expands to a size that does not fit");
        shape[k] = to_i64(s.d[k]);
    }
    if (!broadcast(d.dims, d.nd, shape, (int)s.n, dims, &nd))
        return node_fail(im, n, "expands to a shape that does not fit");
    const uint64_t count = dims_count(dims, nd);
    float *out = fold_alloc(im, n, count);
    if (!out) return false;
    for (uint64_t i = 0; i < count; i++) out[i] = d.d[bcast_at(d.dims, d.nd, dims, nd, i)];
    return const_out(im, n, 0, out, dims, nd, d.type);
}

static bool fold_constant_of_shape(Importer *im, const Node *n) {
    Const s;
    if (!const_arg(im, n, 0, &s)) return false;
    int64_t dims[MAX_DIMS];
    if (s.n > MAX_DIMS) return node_fail(im, n, "makes more than 8 axes");
    for (uint64_t k = 0; k < s.n; k++) {
        if (!(s.d[k] >= 0.0f && s.d[k] <= 1e12f)) return node_fail(im, n, "makes a size that does not fit");
        dims[k] = to_i64(s.d[k]);
    }
    float value = 0.0f;
    int32_t type = ONNX_FLOAT;
    const Attr *a = attr(n, "value");
    if (a && a->has_t) {
        TensorRef t;
        read_tensor(a->t, &t);
        if (tensor_count(&t) != 1 || !tensor_floats(&t, &value, 1)) return node_fail(im, n, "has a value it cannot read");
        type = t.type;
    }
    const uint64_t count = dims_count(dims, (int)s.n);
    float *out = fold_alloc(im, n, count);
    if (!out) return false;
    for (uint64_t i = 0; i < count; i++) out[i] = value;
    return const_out(im, n, 0, out, dims, (int)s.n, type);
}

static bool fold_range(Importer *im, const Node *n) {
    Const a, b, c;
    if (!const_arg(im, n, 0, &a) || !const_arg(im, n, 1, &b) || !const_arg(im, n, 2, &c)) return false;
    if (a.n != 1 || b.n != 1 || c.n != 1 || c.d[0] == 0.0f) return node_fail(im, n, "needs scalars and a nonzero step");
    const double steps = ceil(((double)b.d[0] - a.d[0]) / c.d[0]);
    const int64_t count = steps > 0.0 ? (steps > 1e12 ? (int64_t)1e12 : (int64_t)steps) : 0;
    float *out = fold_alloc(im, n, (uint64_t)count);
    if (!out) return false;
    for (int64_t i = 0; i < count; i++) out[i] = (float)(a.d[0] + (double)i * c.d[0]);
    return const_out(im, n, 0, out, &count, 1, a.type);
}

static bool fold_trilu(Importer *im, const Node *n) {
    Const d, k;
    if (!const_arg(im, n, 0, &d) || !const_opt(im, n, 1, &k)) return false;
    if (d.nd < 2) return node_fail(im, n, "needs at least two axes");
    const int64_t shift = k.d ? to_i64(k.d[0]) : 0, rows = d.dims[d.nd - 2], cols = d.dims[d.nd - 1];
    const bool upper = attr_int(n, "upper", 1) != 0;
    float *out = fold_alloc(im, n, d.n);
    if (!out) return false;
    for (uint64_t i = 0; i < d.n; i++) {
        const int64_t j = (int64_t)(i % (uint64_t)cols), r = (int64_t)(i / (uint64_t)cols % (uint64_t)rows);
        out[i] = (upper ? j - r >= shift : j - r <= shift) ? d.d[i] : 0.0f;
    }
    return const_out(im, n, 0, out, d.dims, d.nd, d.type);
}

static bool fold_matmul(Importer *im, const Node *n) {
    Const a, b;
    if (!const_arg(im, n, 0, &a) || !const_arg(im, n, 1, &b)) return false;
    if (a.nd < 1 || b.nd < 1) return node_fail(im, n, "multiplies scalars");
    /* vectors as a row and a column; batches broadcast */
    int64_t ad[MAX_DIMS], bd[MAX_DIMS], lead[MAX_DIMS], dims[MAX_DIMS];
    int na = a.nd, nb = b.nd, nl;
    memcpy(ad, a.dims, sizeof ad);
    memcpy(bd, b.dims, sizeof bd);
    if (na == 1) ad[1] = ad[0], ad[0] = 1, na = 2;
    if (nb == 1) bd[1] = 1, nb = 2;
    const int64_t M = ad[na - 2], K = ad[na - 1], N = bd[nb - 1];
    if (bd[nb - 2] != K || !broadcast(ad, na - 2, bd, nb - 2, lead, &nl) || nl + 2 > MAX_DIMS)
        return node_fail(im, n, "multiplies shapes that do not fit");
    int nd = 0;
    for (int i = 0; i < nl; i++) dims[nd++] = lead[i];
    if (a.nd > 1) dims[nd++] = M;
    if (b.nd > 1) dims[nd++] = N;
    const uint64_t batches = dims_count(lead, nl), count = batches * (uint64_t)M * (uint64_t)N;
    if (count > FOLD_LIMIT || (K && count > FOLD_LIMIT * 16 / (uint64_t)K))
        return node_fail(im, n, "multiplies constants too large to compute on import");
    float *out = fold_alloc(im, n, count);
    if (!out) return false;
    for (uint64_t t = 0; t < batches; t++) {
        const float *pa = a.d + bcast_at(ad, na - 2, lead, nl, t) * (uint64_t)(M * K);
        const float *pb = b.d + bcast_at(bd, nb - 2, lead, nl, t) * (uint64_t)(K * N);
        for (int64_t i = 0; i < M; i++)
            for (int64_t j = 0; j < N; j++) {
                double sum = 0.0;
                for (int64_t k = 0; k < K; k++) sum += (double)pa[i * K + k] * pb[k * N + j];
                out[(t * (uint64_t)M + (uint64_t)i) * (uint64_t)N + (uint64_t)j] = (float)sum;
            }
    }
    return const_out(im, n, 0, out, dims, nd, a.type);
}

/* Split's sizes along its axis of a tensor of size total: input 1, the attribute (before operator set
   13), or equal parts (num_outputs, or the outputs). */
static bool split_sizes(Importer *im, const Node *n, int64_t total, int64_t *sizes) {
    const int parts = n->noutputs;
    Const s;
    const Attr *a = attr(n, "split");
    if (parts < 1 || parts > MAX_NODE_OUTPUTS) return node_fail(im, n, "splits into 1 to 16 outputs");
    if (!const_opt(im, n, 1, &s)) return false;
    int64_t sum = 0;
    if (s.d || a) {
        if ((s.d ? (int)s.n : a->nints) != parts || (a && a->nints > MAX_INTS))
            return node_fail(im, n, "lists as many sizes as it has outputs");
        for (int k = 0; k < parts; k++) {
            sizes[k] = s.d ? to_i64(s.d[k]) : a->ints[k];
            if (sizes[k] < 0 || sizes[k] > total) return node_fail(im, n, "splits into sizes that do not fit");
            sum += sizes[k];
        }
        return sum == total ? true : node_fail(im, n, "splits into sizes that do not add up to its input's");
    }
    const int64_t each = (total + parts - 1) / parts;
    for (int k = 0; k < parts; k++) {
        sizes[k] = total - sum < each ? total - sum : each;
        sum += sizes[k];
    }
    return true;
}

static bool fold_split(Importer *im, const Node *n) {
    Const d;
    if (!const_arg(im, n, 0, &d)) return false;
    int axis;
    if (!norm_axis(attr_int(n, "axis", 0), d.nd, &axis)) return node_fail(im, n, "splits along an axis it lacks");
    int64_t sizes[MAX_NODE_OUTPUTS];
    if (!split_sizes(im, n, d.dims[axis], sizes)) return false;
    const uint64_t outer = dims_count(d.dims, axis), inner = dims_count(d.dims + axis + 1, d.nd - axis - 1);
    for (int k = 0, at = 0; k < n->noutputs; at += (int)sizes[k], k++) {
        int64_t dims[MAX_DIMS];
        memcpy(dims, d.dims, sizeof dims);
        dims[axis] = sizes[k];
        const uint64_t chunk = (uint64_t)sizes[k] * inner;
        float *out = fold_alloc(im, n, outer * chunk);
        if (!out) return false;
        for (uint64_t o = 0; o < outer; o++)
            memcpy(out + o * chunk, d.d + (o * (uint64_t)d.dims[axis] + (uint64_t)at) * inner,
                   (size_t)chunk * sizeof(float));
        if (n->outputs[k].n && !const_out(im, n, k, out, dims, d.nd, d.type)) return false;
        if (!n->outputs[k].n) free(out);
    }
    return true;
}

static bool fold_tile(Importer *im, const Node *n) {
    Const d, r;
    if (!const_arg(im, n, 0, &d) || !const_arg(im, n, 1, &r)) return false;
    if (r.n != (uint64_t)d.nd) return node_fail(im, n, "repeats other axes than its input has");
    int64_t dims[MAX_DIMS];
    for (int i = 0; i < d.nd; i++) {
        if (!(r.d[i] >= 0.0f && r.d[i] <= 1e9f)) return node_fail(im, n, "repeats a number of times that does not fit");
        dims[i] = d.dims[i] * to_i64(r.d[i]);
    }
    const uint64_t count = dims_count(dims, d.nd);
    float *out = fold_alloc(im, n, count);
    if (!out) return false;
    for (uint64_t i = 0; i < count; i++) {
        uint64_t rest = i, at = 0, stride = 1;
        for (int a = d.nd - 1; a >= 0; a--) {
            at += (rest % (uint64_t)dims[a]) % (uint64_t)d.dims[a] * stride;
            rest /= (uint64_t)dims[a];
            stride *= (uint64_t)d.dims[a];
        }
        out[i] = d.d[at];
    }
    return const_out(im, n, 0, out, dims, d.nd, d.type);
}

/* The binary operators evaluated on constants, by name: their F_ code, or -1. */
static int binary_op(Str op) {
    static const char *const names[] = {"Add", "Sub", "Mul", "Div", "Pow", "Mod", "Max", "Min", "Equal", "Less",
                                        "LessOrEqual", "Greater", "GreaterOrEqual", "And", "Or", "Xor"};
    for (int k = 0; k < (int)(sizeof names / sizeof names[0]); k++)
        if (str_is(op, names[k])) return k;
    return -1;
}

/* Whether a node computes constants on import: every input it reads a constant (Shape and Size read
   only their input's shape). */
static bool folds(Importer *im, const Node *n) {
    const Str op = n->op;
    if (n->domain.n && !str_is(n->domain, "ai.onnx")) return false;
    if (str_is(op, "Shape") || str_is(op, "Size")) {
        const Value *x = n->ninputs > 0 ? find(im, n->inputs[0]) : NULL;
        return x != NULL;
    }
    if (!(binary_op(op) >= 0 || unary_fold_op(op) >= 0 || str_is(op, "Cast") || str_is(op, "Where") ||
          str_is(op, "Gather") || str_is(op, "Unsqueeze") || str_is(op, "Squeeze") || str_is(op, "Concat") ||
          str_is(op, "Slice") || str_is(op, "Reshape") || str_is(op, "Flatten") || str_is(op, "Transpose") ||
          str_is(op, "Expand") || str_is(op, "ConstantOfShape") || str_is(op, "Range") || str_is(op, "Trilu") ||
          str_is(op, "MatMul") || str_is(op, "Split") || str_is(op, "Tile")))
        return false;
    int present = 0;
    for (int k = 0; k < n->ninputs && k < MAX_NODE_INPUTS; k++) {
        if (!n->inputs[k].n) continue;
        const Value *v = find(im, n->inputs[k]);
        if (!v || !v->constant) return false;
        present++;
    }
    return present > 0;
}

static bool fold(Importer *im, const Node *n) {
    const Str op = n->op;
    const int b = binary_op(op);
    if (b >= 0) return fold_binary(im, n, b);
    if (unary_fold_op(op) >= 0) return fold_unary(im, n);
    if (str_is(op, "Cast")) return fold_cast(im, n);
    if (str_is(op, "Where")) return fold_where(im, n);
    if (str_is(op, "Shape")) return fold_shape(im, n);
    if (str_is(op, "Size")) return fold_size(im, n);
    if (str_is(op, "Gather")) return fold_gather(im, n);
    if (str_is(op, "Unsqueeze")) return fold_squeeze(im, n, true);
    if (str_is(op, "Squeeze")) return fold_squeeze(im, n, false);
    if (str_is(op, "Concat")) return fold_concat(im, n);
    if (str_is(op, "Slice")) return fold_slice(im, n);
    if (str_is(op, "Reshape") || str_is(op, "Flatten")) return fold_reshape(im, n);
    if (str_is(op, "Transpose")) return fold_transpose(im, n);
    if (str_is(op, "Expand")) return fold_expand(im, n);
    if (str_is(op, "ConstantOfShape")) return fold_constant_of_shape(im, n);
    if (str_is(op, "Range")) return fold_range(im, n);
    if (str_is(op, "Trilu")) return fold_trilu(im, n);
    if (str_is(op, "MatMul")) return fold_matmul(im, n);
    if (str_is(op, "Split")) return fold_split(im, n);
    return fold_tile(im, n);
}

static bool convert_node(Importer *im, const Node *n) {
    if (n->domain.n && !str_is(n->domain, "ai.onnx"))
        return fail(im, "ONNX import: operator %.*s of the domain %.*s is not supported", n->op, n->domain);
    if (folds(im, n)) return fold(im, n);
    ActivationFunction act = activation_of(n, im);
    if (im->error[0]) return false;
    if (act != ACT_COUNT) return convert_activation(im, n, act);
    if (str_is(n->op, "Conv")) return convert_conv(im, n);
    if (str_is(n->op, "Gemm")) return convert_gemm(im, n);
    if (str_is(n->op, "MatMul")) return convert_matmul(im, n);
    if (str_is(n->op, "Add")) return convert_add(im, n);
    if (str_is(n->op, "Concat")) return convert_concat(im, n);
    if (str_is(n->op, "MaxPool")) return convert_pool(im, n, true);
    if (str_is(n->op, "AveragePool")) return convert_pool(im, n, false);
    if (str_is(n->op, "BatchNormalization")) return convert_batch_norm(im, n);
    if (str_is(n->op, "ConvTranspose")) return convert_conv_transpose(im, n);
    if (str_is(n->op, "Resize") || str_is(n->op, "Upsample")) return convert_resize(im, n);
    if (str_is(n->op, "LayerNormalization")) return convert_layer_norm(im, n);
    if (str_is(n->op, "Transpose")) return convert_transpose(im, n);
    if (str_is(n->op, "Identity") || str_is(n->op, "Dropout")) return alias(im, n, false);
    if (str_is(n->op, "Flatten")) {
        if (attr_int(n, "axis", 1) != 1)
            return fail(im, "ONNX import: Flatten node '%.*s' keeps more than the batch axis%.*s", n->name, (Str){"", 0});
        return alias(im, n, true);
    }
    if (str_is(n->op, "Reshape")) return convert_reshape(im, n);
    if (str_is(n->op, "Constant")) return convert_constant(im, n);
    if (str_is(n->op, "ReduceMean")) {
        /* over the height and width of a map: global average pooling */
        const Value *x = layer_input(im, n, 0);
        if (!x) return false;
        const Attr *a = attr(n, "axes");
        Value *ax = n->ninputs > 1 && n->inputs[1].n ? find(im, n->inputs[1]) : NULL;
        const float *axv = !a && ax && ax->constant ? constant_floats(im, ax) : NULL;
        int64_t axes[2] = {0, 0};
        int count = a ? a->nints : axv ? (int)ax->count : 0;
        for (int k = 0; k < count && k < 2; k++) axes[k] = a ? a->ints[k] : (int64_t)axv[k];
        for (int k = 0; k < 2; k++) if (axes[k] < 0) axes[k] += 4;
        if (x->flat || count != 2 || !((axes[0] == 2 && axes[1] == 3) || (axes[0] == 3 && axes[1] == 2)))
            return fail(im, "ONNX import: ReduceMean node '%.*s' averages other axes than a map's height and width%.*s",
                        n->name, (Str){"", 0});
        uint32_t l = global_avg_pool2d(.net = im->net, .inputs = {x->layer}, .input_count = 1);
        if (!added(im, l, n)) return false;
        im->bias_open = false;
        Value *y = layer_output(im, n, l);
        if (y) y->vector = attr_int(n, "keepdims", 1) == 0;       /* [N, C] */
        return y != NULL;
    }
    if (str_is(n->op, "GlobalAveragePool")) {
        const Value *x = layer_input(im, n, 0);
        if (!x) return false;
        uint32_t l = global_avg_pool2d(.net = im->net, .inputs = {x->layer}, .input_count = 1);
        if (!added(im, l, n)) return false;
        im->bias_open = false;
        return layer_output(im, n, l) != NULL;
    }
    return fail(im, "ONNX import: operator %.*s (node '%.*s') is not supported", n->op, n->name);
}

/* The shape of a graph input: [N, C, H, W] or [N, F] (N may be symbolic). */
static bool input_shape(Pb vi, Str *name, int64_t *dims, int *ndims) {
    *ndims = 0;
    uint32_t f, w;
    bool typed = false;
    while (pb_next(&vi, &f, &w)) {
        if (f == 1 && w == WIRE_BYTES) { *name = pb_str(&vi); continue; }
        if (f != 2 || w != WIRE_BYTES) { pb_skip(&vi, w); continue; }
        Pb type = pb_bytes(&vi);                        /* TypeProto */
        while (pb_next(&type, &f, &w)) {
            if (f != 1 || w != WIRE_BYTES) { pb_skip(&type, w); continue; }
            Pb tensor = pb_bytes(&type);                /* TypeProto.Tensor */
            while (pb_next(&tensor, &f, &w)) {
                if (f != 2 || w != WIRE_BYTES) { pb_skip(&tensor, w); continue; }
                Pb shape = pb_bytes(&tensor);           /* TensorShapeProto */
                typed = true;
                while (pb_next(&shape, &f, &w)) {
                    if (f != 1 || w != WIRE_BYTES) { pb_skip(&shape, w); continue; }
                    Pb dim = pb_bytes(&shape);
                    int64_t v = -1;
                    while (pb_next(&dim, &f, &w)) {
                        if (f == 1 && w == WIRE_VARINT) v = (int64_t)pb_varint(&dim);
                        else pb_skip(&dim, w);
                    }
                    if (*ndims < MAX_DIMS) dims[*ndims] = v;
                    (*ndims)++;
                }
            }
        }
    }
    return typed && vi.ok;
}

/* A name read once more. */
static bool count_use(Importer *im, Str name) {
    if (!table_room(&im->uses)) return false;
    Slot *slot = table_slot(&im->uses, name);
    if (!slot->used) *slot = (Slot){name, 0, true}, im->uses.count++;
    slot->value++;
    return true;
}

/* Points a tensor kept in an external file at its bytes there: the file named relative to the
   model's folder, and inside it (mapped once for all of its tensors). */
static bool resolve_external(Importer *im, TensorRef *t) {
    const Str none = {"", 0}, loc = t->location;
    if (!im->dir)
        return fail(im, "ONNX import: initializer '%.*s' keeps its data in an external file, which only an import "
                    "from the model's path can read%.*s", t->name, none);
    bool inside = loc.n > 0 && loc.s[0] != '/' && loc.s[0] != '\\' && !(loc.n > 1 && loc.s[1] == ':') &&
                  !memchr(loc.s, 0, loc.n);
    for (size_t k = 0, start = 0; inside && k <= loc.n; k++)
        if (k == loc.n || loc.s[k] == '/' || loc.s[k] == '\\') {
            inside = !(k - start == 2 && loc.s[start] == '.' && loc.s[start + 1] == '.');
            start = k + 1;
        }
    if (!inside)
        return fail(im, "ONNX import: initializer '%.*s' names an external file outside the model's folder ('%.*s')",
                    t->name, loc);
    External *e = NULL;
    for (size_t k = 0; k < im->nexternals && !e; k++)
        if (str_eq(im->externals[k].location, loc)) e = &im->externals[k];
    if (!e) {
        External *grown = (External *)realloc(im->externals, (im->nexternals + 1) * sizeof(External));
        size_t dir = strlen(im->dir);
        char *path = (char *)malloc(dir + loc.n + 1);
        if (grown) im->externals = grown;
        if (!grown || !path) { free(path); return fail(im, "ONNX import: out of memory%.*s%.*s", none, none); }
        memcpy(path, im->dir, dir);
        memcpy(path + dir, loc.s, loc.n);
        path[dir + loc.n] = 0;
        e = &im->externals[im->nexternals];
        bool opened = spingalett_file_open(&e->view, path);
        free(path);
        if (!opened)
            return fail(im, "ONNX import: the external data file '%.*s' of initializer '%.*s' cannot be read", loc, t->name);
        e->location = loc;
        im->nexternals++;
    }
    const uint64_t size = e->view.size;
    const uint64_t length = t->has_length ? t->length : t->offset <= size ? size - t->offset : 0;
    if (t->offset > size || length > size - t->offset)
        return fail(im, "ONNX import: the data of initializer '%.*s' lies outside its external file '%.*s'", t->name, loc);
    t->raw = (Pb){e->view.data + t->offset, e->view.data + t->offset + length, true};
    t->has_raw = true;
    t->external = false;
    return true;
}

/* The import of a model in memory; dir: its folder (ending in a separator, or empty), for external
   data, or NULL. */
static NeuralNetwork *import_onnx(const void *data, size_t size, const char *dir) {
    if (!data) {
        set_error(SPINGALETT_ERR_INVALID, "ONNX import: data is NULL");
        return NULL;
    }
    Pb model = {(const uint8_t *)data, (const uint8_t *)data + size, true}, graph = {NULL, NULL, false};
    uint32_t f, w;
    int64_t opset = 0;
    while (pb_next(&model, &f, &w)) {
        if (f == 7 && w == WIRE_BYTES) {
            graph = pb_bytes(&model);
        } else if (f == 8 && w == WIRE_BYTES) {
            /* an operator set: its domain and version */
            Pb o = pb_bytes(&model);
            Str domain = {"", 0};
            int64_t version = 0;
            uint32_t of, ow;
            while (pb_next(&o, &of, &ow)) {
                if (of == 1 && ow == WIRE_BYTES) domain = pb_str(&o);
                else if (of == 2 && ow == WIRE_VARINT) version = (int64_t)pb_varint(&o);
                else pb_skip(&o, ow);
            }
            if (domain.n == 0 || str_is(domain, "ai.onnx")) opset = version;
        } else {
            pb_skip(&model, w);
        }
    }
    if (!model.ok || !graph.ok) {
        set_error(SPINGALETT_ERR_INVALID, "ONNX import: not an ONNX model (no graph)");
        return NULL;
    }

    Importer im = {.dir = dir, .opset = opset};
    bool ok = true;
    /* readers of every value: node inputs (but Shape's and Size's, which read no values) and graph
       outputs */
    Pb g = graph;
    while (ok && pb_next(&g, &f, &w)) {
        if (f == 1 && w == WIRE_BYTES) {
            Node n;
            ok = read_node(pb_bytes(&g), &n);
            const bool shape = str_is(n.op, "Shape") || str_is(n.op, "Size");
            for (int k = 0; ok && !shape && k < n.ninputs && k < MAX_NODE_INPUTS; k++) ok = count_use(&im, n.inputs[k]);
        } else if (f == 12 && w == WIRE_BYTES) {
            Str name = {0};
            int64_t dims[MAX_DIMS];
            int nd;
            (void)input_shape(pb_bytes(&g), &name, dims, &nd);
            ok = count_use(&im, name);
        } else {
            pb_skip(&g, w);
        }
    }
    ok = ok && g.ok;

    /* constants: the initializers, read where they are when a node needs them */
    g = graph;
    while (ok && pb_next(&g, &f, &w)) {
        if (f != 5 || w != WIRE_BYTES) { pb_skip(&g, w); continue; }
        TensorRef t;
        read_tensor(pb_bytes(&g), &t);
        if (t.external && !resolve_external(&im, &t)) { ok = false; break; }
        if (t.ndims > MAX_DIMS || tensor_count(&t) > tensor_capacity(&t)) {
            ok = fail(&im, "ONNX import: initializer '%.*s' holds less data than its shape needs%.*s", t.name, (Str){"", 0});
            break;
        }
        Value *v = add_value(&im, t.name);
        ok = v != NULL;
        if (!ok) break;
        v->constant = true;
        v->ndims = t.ndims > MAX_DIMS ? MAX_DIMS : t.ndims;
        memcpy(v->dims, t.dims, sizeof v->dims);
        v->count = tensor_count(&t);
        v->ctype = t.type;
        v->tensor = t;
    }

    /* the input: the graph input that is no initializer */
    NeuralNetwork *net = ok ? new_spingalett_struct_arguments((NeuralNetworkArgs){.loss_func = LOSS_MSE}) : NULL;
    im.net = net;
    ok = ok && net;
    g = graph;
    bool have_input = false;
    while (ok && pb_next(&g, &f, &w)) {
        if (f != 11 || w != WIRE_BYTES) { pb_skip(&g, w); continue; }
        Str name = {0};
        int64_t dims[MAX_DIMS];
        int nd;
        bool typed = input_shape(pb_bytes(&g), &name, dims, &nd);
        if (find(&im, name)) continue;
        if (have_input) { ok = fail(&im, "ONNX import: the model has several inputs ('%.*s' is another)%.*s", name, (Str){"", 0}); break; }
        have_input = true;
        if (!typed || (nd != 4 && nd != 2) || dims[1] <= 0 || (nd == 4 && (dims[2] <= 0 || dims[3] <= 0))) {
            ok = fail(&im, "ONNX import: input '%.*s' must have the shape [N, C, H, W] or [N, F]%.*s", name, (Str){"", 0});
            break;
        }
        uint32_t l = nd == 4 ? layer(.net = net, .height = (uint32_t)dims[2], .width = (uint32_t)dims[3],
                                     .channels = (uint32_t)dims[1])
                             : layer(.net = net, .neurons_amount = (uint32_t)dims[1]);
        Value *v = l == 0 ? add_value(&im, name) : NULL;
        ok = v != NULL;
        if (!ok) { fail(&im, "ONNX import: input '%.*s' cannot be an input layer%.*s", name, (Str){"", 0}); break; }
        v->layer = 0;
        v->c = (uint32_t)dims[1];
        v->h = nd == 4 ? (uint32_t)dims[2] : 1u;
        v->w = nd == 4 ? (uint32_t)dims[3] : 1u;
        v->vector = nd == 2;
    }
    if (ok && !have_input) ok = fail(&im, "ONNX import: the model has no input%.*s%.*s", (Str){"", 0}, (Str){"", 0});

    /* the nodes, in their (topological) order */
    g = graph;
    while (ok && pb_next(&g, &f, &w)) {
        if (f != 1 || w != WIRE_BYTES) { pb_skip(&g, w); continue; }
        Node n;
        ok = read_node(pb_bytes(&g), &n);
        if (!ok) { fail(&im, "ONNX import: a node cannot be read%.*s%.*s", (Str){"", 0}, (Str){"", 0}); break; }
        if (n.noutputs < 1) continue;
        ok = convert_node(&im, &n);
    }

    /* the output: the last layer */
    g = graph;
    int outputs = 0;
    while (ok && pb_next(&g, &f, &w)) {
        if (f != 12 || w != WIRE_BYTES) { pb_skip(&g, w); continue; }
        Str name = {0};
        int64_t dims[MAX_DIMS];
        int nd;
        (void)input_shape(pb_bytes(&g), &name, dims, &nd);
        const Value *v = find(&im, name);
        if (++outputs > 1) ok = fail(&im, "ONNX import: the model has several outputs ('%.*s' is another)%.*s", name, (Str){"", 0});
        else if (!v || v->constant || v->layer + 1 != net->layers || net->layers < 2)
            ok = fail(&im, "ONNX import: output '%.*s' is not the last layer the model computes%.*s", name, (Str){"", 0});
    }
    if (ok) {
        ActivationFunction last = net->act_func[net->layers - 2];
        net->loss_func = last == ACT_SOFTMAX || last == ACT_SIGMOID ? LOSS_CROSS_ENTROPY : LOSS_MSE;
        ok = spingalett_check_graph(net, "ONNX import");
    }

    for (size_t k = 0; k < im.nvalues; k++)
        if (im.values[k].owned) free(im.values[k].data);
    free(im.values);
    free(im.names.slots);
    free(im.uses.slots);
    for (size_t k = 0; k < im.nexternals; k++) spingalett_file_close(&im.externals[k].view);
    free(im.externals);
    if (!ok) {
        if (im.error[0]) {
            set_error(SPINGALETT_ERR_INVALID, im.error);
            spingalett_log(LOG_ERROR, "%s", im.error);
        } else if (spingalett_last_error_code() == SPINGALETT_OK) {
            set_error(SPINGALETT_ERR_INVALID, "ONNX import: the model cannot be read");
        }
        free_network(net);
        return NULL;
    }
    return net;
}

NeuralNetwork *spingalett_import_onnx(const char *path) {
    if (!path) {
        set_error(SPINGALETT_ERR_INVALID, "ONNX import: path is NULL");
        return NULL;
    }
    SpgFileView file;
    if (!spingalett_file_open(&file, path)) return NULL;
    /* the folder, for external data: the path up to its last separator */
    size_t dir = strlen(path);
    while (dir > 0 && path[dir - 1] != '/' && path[dir - 1] != '\\') dir--;
    char *folder = (char *)malloc(dir + 1);
    NeuralNetwork *net = NULL;
    if (folder) {
        memcpy(folder, path, dir);
        folder[dir] = 0;
        net = import_onnx(file.data, file.size, folder);
    } else {
        set_error(SPINGALETT_ERR_ALLOC, "ONNX import: out of memory");
    }
    free(folder);
    spingalett_file_close(&file);
    return net;
}

NeuralNetwork *spingalett_import_onnx_from_memory(const void *data, size_t size) { return import_onnx(data, size, NULL); }
