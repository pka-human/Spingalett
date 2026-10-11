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
 *
 * Transformers: a sequence of T tokens is a layer of 1 x T cells (an input [N, T] of token indices, or
 * [N, T, C] of values), its ONNX tensors [N, T, C] (or [N T, C]: rows). Nodes reading sequences are
 * translated lazily: a product of tokens by constant weights, an embedding, an element-wise expression
 * or a rearrangement of the tokens' channels (Reshape, Transpose, Split, Slice, Expand ...) becomes a
 * description that is made a layer only when something reads it as one. That lets patterns spread over
 * many nodes become one layer: an element-wise expression is an activation (recognised by evaluating
 * it) or a normalization (recognised by its form); the queries' and keys' product, scaled, masked and
 * through a softmax, times the values, is an attention layer whose input is the queries', keys' and
 * values' products merged into one linear layer (the scale folded into the queries' weights, rotary
 * embeddings recognised from their tables); a learned position table added to an embedding is its
 * positions.
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
    int64_t dims[MAX_DIMS];     /* constants, sequences and lazy values: the shape in the file */
    int ndims;
    uint64_t count;
    int32_t ctype;              /* constants: the ONNX element type */
    bool seq;                   /* tokens x channels, [N, T, C] or as rows [N T, C] (a layer of 1 x T cells, or
                                   a lazy value that will be one) */
    uint32_t lazy;              /* 1 + the index of the computation it stands for in im->lazies (0: none) */
} Value;

#define NO_VALUE UINT32_MAX
#define MAX_FACTORS 4

/* An axis of a view is a run of factors, major to minor: the tokens, or channels stride apart (stride 0:
   one channel repeated, as Expand repeats the keys and values of grouped-query attention). */
typedef struct { uint32_t size, stride; bool token; } Factor;

/* Per-token values seen through Reshape, Transpose, Split, Slice, Gather, Unsqueeze, Squeeze and Expand:
   the tensor's axes as factors over the tokens and the channels of base (a sequence) from offset on,
   times scale; rotated first by rotary embeddings of base theta over heads of rope_head channels
   (theta 0: none). */
typedef struct {
    uint32_t base, offset;
    int nd;
    uint8_t nf[MAX_DIMS];
    Factor f[MAX_DIMS][MAX_FACTORS];
    float scale, theta;
    uint32_t rope_head;
} View;

/* What a lazy value stands for. */
enum {
    LZ_LINEAR = 1,              /* scale (x W) + bias for each token of the layer src, then act */
    LZ_EMBED,                   /* the rows of a table for the input's tokens (times tscale), plus positions */
    LZ_EXPR,                    /* an element-wise expression of one value (an activation, a normalization) */
    LZ_VIEW,                    /* a view of a sequence */
    LZ_ROT,                     /* rotate_half of a view: its heads' halves swapped, the first negated */
    LZ_COS,                     /* a view times a table (the rotary embeddings' cosines) */
    LZ_SIN,                     /* a rotated view times a table (their sines) */
    LZ_SCORES,                  /* queries times keys, scaled, masked (attention) */
    LZ_PROBS                    /* the scores' softmax over the keys */
};

typedef struct {
    uint8_t kind;
    Str op, node;               /* the node that made it, named in errors */
    uint32_t made;              /* the layer computing it, SPINGALETT_NO_LAYER until something needed one */
    union {
        struct {
            uint32_t src, w, in, rows, row0, out;   /* rows of the weights w ([rows][in], or [in][rows]
                                                       transposed) from row0 on, out of them */
            bool transposed;
            float *scale, *bias;                    /* out each (NULL: ones, zeros), owned */
            ActivationFunction act;
        } lin;
        struct { uint32_t table, pos; float tscale, pscale; ActivationFunction act; } emb;
        struct { uint32_t base, root; } ex;
        struct { View v; uint32_t table, rows; } view;      /* rows: the sequence its channels make */
        struct { View q, k; float scale; bool causal; } sc;
    };
} Lazy;

/* A node of an element-wise expression of one value (E_X): constants (E_K, or one a channel E_VEC: a
   constant's value index), the operators (binary E_ADD to E_POW, unary E_NEG to E_GELU_TANH), and the
   mean over the channels (E_MEAN, of normalizations). size counts the nodes below it with repeats; form
   tells a scalar from a value a token (a mean) and one a channel; mean and vecs mark nodes with a mean
   or a constant a channel below them. */
enum {
    E_X, E_K, E_VEC, E_ADD, E_SUB, E_MUL, E_DIV, E_POW, E_NEG, E_SQRT, E_RECIP, E_EXP, E_ERF, E_TANH, E_SIGMOID,
    E_RELU, E_GELU, E_GELU_TANH, E_MEAN
};
enum { FORM_SCALAR, FORM_REDUCED, FORM_FULL };
#define EXPR_SIZE 255               /* nodes an expression may hold, counted with repeats */

typedef struct {
    uint8_t op, form;
    bool mean, vecs;
    uint16_t size;
    uint32_t a, b, vec;
    float k;
} Enode;

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
    Lazy *lazies;               /* what lazy values stand for */
    size_t nlazies, lcap;
    Enode *enodes;              /* the nodes of element-wise expressions */
    size_t nenodes, ecap;
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

/* A new value, under its name or (named false) under none, as the sequences a node makes on the way to
   its output. Values stay where they are in the array (their indices) but may move in memory. */
static Value *new_value(Importer *im, Str name, bool named) {
    if (im->nvalues == im->cap) {
        size_t cap = im->cap ? 2 * im->cap : 64;
        Value *v = cap < NO_VALUE ? (Value *)realloc(im->values, cap * sizeof(Value)) : NULL;
        if (!v) return NULL;
        im->values = v;
        im->cap = cap;
    }
    if (named) {
        if (!table_room(&im->names)) return NULL;
        Slot *slot = table_slot(&im->names, name);
        if (!slot->used) *slot = (Slot){name, 0, true}, im->names.count++;
        slot->value = (uint32_t)im->nvalues;
    }
    Value *v = &im->values[im->nvalues++];
    memset(v, 0, sizeof *v);
    v->name = name;
    return v;
}

static Value *add_value(Importer *im, Str name) { return new_value(im, name, true); }

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

/* Whether v is a name of the layer's output: a layer value, or a lazy one made that layer. */
static bool names_layer(const Importer *im, const Value *v, uint32_t layer) {
    if (v->constant) return false;
    return v->lazy ? im->lazies[v->lazy - 1].made == layer : v->layer == layer;
}

/* Readers of a layer's output under all of its names, the nodes that only renamed it aside. */
static uint32_t layer_readers(const Importer *im, uint32_t layer) {
    uint32_t total = 0, aliases = 0;
    for (size_t k = 0; k < im->nvalues; k++) {
        const Value *v = &im->values[k];
        if (!names_layer(im, v, layer)) continue;
        total += readers(im, v->name);
        aliases += v->alias;
    }
    return total > aliases ? total - aliases : 0u;
}

/* Operators that may read a map transposed to channels last: elementwise ones, layer normalization
   (over its channels) and the transposition back. */
static bool reads_nhwc(const Node *n) {
    static const char *const ops[] = {"LayerNormalization", "Transpose", "Relu", "Sigmoid", "Tanh", "LeakyRelu"};
    for (size_t k = 0; k < sizeof ops / sizeof ops[0]; k++)
        if (str_is(n->op, ops[k])) return true;
    return false;
}

static bool materialize(Importer *im, Value *v);

/* Input k as a layer's output (a lazy value made a layer first). */
static const Value *layer_input(Importer *im, const Node *n, int k) {
    if (k >= n->ninputs) { fail(im, "ONNX import: %.*s node '%.*s' lacks an input", n->op, n->name); return NULL; }
    Value *v = find(im, n->inputs[k]);
    if (!v) { fail(im, "ONNX import: input '%.*s' of %.*s is not produced by an earlier node", n->inputs[k], n->op); return NULL; }
    if (v->constant) { fail(im, "ONNX import: %.*s reads the constant '%.*s' where a layer output is expected", n->op, n->inputs[k]); return NULL; }
    if (v->lazy && !materialize(im, v)) return NULL;
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
    if (str_is(n->op, "Gelu")) {
        const Attr *a = attr(n, "approximate");
        return a && str_is(a->s, "tanh") ? ACT_GELU_TANH : ACT_GELU;
    }
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

/* LayerNormalization (rms: RMSNormalization) over a vector, or over the channels of a map transposed to
   channels last. */
static bool convert_layer_norm(Importer *im, const Node *n, bool rms) {
    const Value *x = layer_input(im, n, 0);
    if (!x) return false;
    const int64_t axis = attr_int(n, "axis", -1);
    bool channels;
    if (x->nhwc) channels = axis == -1 || axis == 3;
    else if (x->vector || x->flat) channels = x->h * x->w == 1 && (axis == -1 || axis == 1);
    else channels = x->h * x->w == 1 && (axis == 1 || axis == -3);    /* [N, C, 1, 1] */
    if (!channels)
        return fail(im, "ONNX import: %.*s node '%.*s' normalizes other axes than a vector or a map's channels "
                    "(transposed to the last place)", n->op, n->name);
    const Value *gamma = constant_input(im, n, 1, x->c);
    if (!gamma) return false;
    const bool shifted = !rms && n->ninputs > 2 && n->inputs[2].n;
    const Value *beta = shifted ? constant_input(im, n, 2, x->c) : NULL;
    if (shifted && !beta) return false;
    const float eps = attr_float(n, "epsilon", 1e-5f), e = eps > 0.0f && eps < 1.0f ? eps : 1e-5f;
    const bool flat = x->flat, vector = x->vector, nhwc = x->nhwc;
    const uint32_t C = x->c;
    uint32_t l = rms ? rms_norm(.net = im->net, .inputs = {x->layer}, .input_count = 1, .act_func = ACT_NONE, .epsilon = e)
                     : layer_norm(.net = im->net, .inputs = {x->layer}, .input_count = 1, .act_func = ACT_NONE, .epsilon = e);
    if (!added(im, l, n)) return false;
    memcpy(layer_weights(im->net, l), gamma->data, C * sizeof(float));
    if (!rms && beta) memcpy(layer_biases(im->net, l), beta->data, C * sizeof(float));
    else if (!rms) memset(layer_biases(im->net, l), 0, C * sizeof(float));
    im->bias_open = false;
    Value *y = layer_output(im, n, l);
    if (y) y->flat = flat, y->vector = vector, y->nhwc = nhwc;
    return y != NULL;
}

/* ------------------------------------------------------------------------- lazy values */

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

/* An error naming the node that made lazy value li. */
static bool lazy_fail(Importer *im, uint32_t li, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vfail(im, im->lazies[li].op, im->lazies[li].node, fmt, ap);
    va_end(ap);
    return false;
}

static bool out_of_memory(Importer *im) {
    return fail(im, "ONNX import: out of memory%.*s%.*s", (Str){"", 0}, (Str){"", 0});
}

/* A new lazy value's description, made by node n: its index, NO_VALUE when out of memory. The array may
   move, so its entries are not held across calls that add some. */
static uint32_t new_lazy(Importer *im, uint8_t kind, const Node *n) {
    if (im->nlazies == im->lcap) {
        const size_t cap = im->lcap ? 2 * im->lcap : 64;
        Lazy *grown = cap < NO_VALUE ? (Lazy *)realloc(im->lazies, cap * sizeof(Lazy)) : NULL;
        if (!grown) { out_of_memory(im); return NO_VALUE; }
        im->lazies = grown;
        im->lcap = cap;
    }
    Lazy *z = &im->lazies[im->nlazies];
    memset(z, 0, sizeof *z);
    z->kind = kind;
    z->op = n->op;
    z->node = n->name;
    z->made = SPINGALETT_NO_LAYER;
    return (uint32_t)im->nlazies++;
}

/* A value standing for lazy value li, of the shape dims (named false: without a name). */
static Value *lazy_value(Importer *im, Str name, bool named, uint32_t li, const int64_t *dims, int nd, bool seq) {
    Value *v = new_value(im, name, named);
    if (!v) { out_of_memory(im); return NULL; }
    v->lazy = li + 1;
    v->seq = seq;
    v->ndims = nd;
    memcpy(v->dims, dims, (size_t)nd * sizeof *dims);
    return v;
}

static uint32_t value_index(const Importer *im, const Value *v) { return (uint32_t)(v - im->values); }

/* What a value is: K_CONST, K_LAYER (a layer's output, or a lazy value made one) or the kind of its lazy
   value. */
enum { K_LAYER = 0, K_CONST = 100 };
static int kind_of(const Importer *im, const Value *v) {
    if (v->constant) return K_CONST;
    if (!v->lazy) return K_LAYER;
    const Lazy *z = &im->lazies[v->lazy - 1];
    return z->made != SPINGALETT_NO_LAYER ? K_LAYER : z->kind;
}

/* What computes a value, telling two names of one computation the same: its lazy value, or its layer. */
static uint64_t source_of(const Value *v) { return v->lazy ? ((uint64_t)1 << 32) | v->lazy : v->layer; }

/* A sequence's tokens and channels ([1, T, C] or [T, C]). */
static uint32_t seq_tokens(const Value *v) { return (uint32_t)(v->ndims == 2 ? v->dims[0] : v->dims[1]); }
static uint32_t seq_channels(const Value *v) { return (uint32_t)v->dims[v->ndims - 1]; }

/* A value's shape in the file with a batch of one, for Shape; its rank. */
static int value_shape(const Value *v, int64_t *dims) {
    if (v->constant || v->seq || v->lazy) {
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

static void set_layer(Importer *im, Value *v, uint32_t layer) {
    SpingalettNetworkLayer d;
    spingalett_network_layer(im->net, layer, &d);
    v->layer = layer;
    v->c = d.channels;
    v->h = d.height;
    v->w = d.width;
}

/* A sequence layer's output as a new value (named false: without a name) of the shape dims. */
static Value *seq_value(Importer *im, Str name, bool named, uint32_t layer, const int64_t *dims, int nd) {
    Value *v = new_value(im, name, named);
    if (!v) { out_of_memory(im); return NULL; }
    set_layer(im, v, layer);
    v->seq = true;
    v->ndims = nd;
    memcpy(v->dims, dims, (size_t)nd * sizeof *dims);
    return v;
}

/* A layer made for lazy value li: false (error set, naming its node) when the builder refused it. */
static bool lazy_added(Importer *im, uint32_t li, uint32_t layer) {
    if (layer != SPINGALETT_NO_LAYER) return true;
    char msg[160];
    snprintf(msg, sizeof msg, "%s", spingalett_last_error_message());
    return lazy_fail(im, li, "%s", msg);
}

static uint32_t make(Importer *im, uint32_t li);

/* A lazy value made a layer (once: later names of it get the same layer). Adds no values, so pointers
   to them stay valid. */
static bool materialize(Importer *im, Value *v) {
    if (!v->lazy) return true;
    const uint32_t li = v->lazy - 1;
    uint32_t l = im->lazies[li].made;
    if (l == SPINGALETT_NO_LAYER) {
        l = make(im, li);
        if (l == SPINGALETT_NO_LAYER) return false;
        im->lazies[li].made = l;
        im->bias_open = false;
    }
    set_layer(im, v, l);
    if (im->lazies[li].kind == LZ_VIEW) v->seq = true;      /* a slice of a product's channels */
    return true;
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

/* ------------------------------------------------------------------------- products and embeddings */

/* A lazy product's whole weight matrix as [rows][in] floats (malloc'd); NULL with the error set. */
static float *product_matrix(Importer *im, uint32_t li) {
    const Lazy *z = &im->lazies[li];
    const uint32_t rows = z->lin.rows, in = z->lin.in;
    const bool transposed = z->lin.transposed;
    int dtype;
    const uint8_t *src = constant_bytes(im, &im->values[z->lin.w], &dtype);
    if (!src) return NULL;
    float *m = (float *)malloc((size_t)rows * in * sizeof(float));
    float *scratch = (float *)malloc(spingalett_dense_scratch(rows, in, transposed) * sizeof(float));
    if (!m || !scratch) {
        free(m);
        free(scratch);
        out_of_memory(im);
        return NULL;
    }
    spingalett_import_dense(m, src, dtype, rows, in, in, 1, transposed, 1.0f, scratch);
    free(scratch);
    return m;
}

/* Output o of a lazy product into w (its in weights, from m, the whole matrix) and *b, times factor. */
static void product_row(const Lazy *z, const float *m, uint32_t o, float factor, float *w, float *b) {
    const float s = (z->lin.scale ? z->lin.scale[o] : 1.0f) * factor;
    const float *row = m + (size_t)(z->lin.row0 + o) * z->lin.in;
    for (uint32_t k = 0; k < z->lin.in; k++) w[k] = s * row[k];
    *b = (z->lin.bias ? z->lin.bias[o] : 0.0f) * factor;
}

/* A lazy product made a linear layer (a 1 x 1 convolution over the tokens) with the activation act. */
static uint32_t make_linear(Importer *im, uint32_t li, ActivationFunction act) {
    const Lazy z = im->lazies[li];
    const uint32_t in = z.lin.in, out = z.lin.out;
    uint32_t l = layer_struct_arguments((LayerArgs){.net = im->net, .type = LAYER_CONV2D, .kernel = 1, .filters = out,
                                                    .inputs = {z.lin.src}, .input_count = 1, .act_func = act,
                                                    .weight_initialization = WEIGHT_INITIALIZATION_NONE});
    if (!lazy_added(im, li, l)) return SPINGALETT_NO_LAYER;
    float *w = layer_weights(im->net, l), *b = layer_biases(im->net, l);
    if (z.lin.row0 == 0 && out == z.lin.rows) {
        /* every row: decoded into the layer's weights in place */
        int dtype;
        const uint8_t *src = constant_bytes(im, &im->values[z.lin.w], &dtype);
        const size_t floats = spingalett_dense_scratch(out, in, z.lin.transposed);
        float *scratch = src ? (float *)malloc(floats * sizeof(float)) : NULL;
        if (!scratch) {
            if (src) out_of_memory(im);
            return SPINGALETT_NO_LAYER;
        }
        spingalett_import_dense(w, src, dtype, out, in, in, 1, z.lin.transposed, 1.0f, scratch);
        free(scratch);
        for (uint32_t o = 0; o < out; o++) {
            if (z.lin.scale && z.lin.scale[o] != 1.0f)
                for (uint32_t k = 0; k < in; k++) w[(size_t)o * in + k] *= z.lin.scale[o];
            b[o] = z.lin.bias ? z.lin.bias[o] : 0.0f;
        }
        return l;
    }
    float *m = product_matrix(im, li);
    if (!m) return SPINGALETT_NO_LAYER;
    for (uint32_t o = 0; o < out; o++) product_row(&z, m, o, 1.0f, w + (size_t)o * in, b + o);
    free(m);
    return l;
}

/* A lazy embedding made a layer reading the input's tokens, with the activation act. */
static uint32_t make_embedding(Importer *im, uint32_t li, ActivationFunction act) {
    const Lazy z = im->lazies[li];
    Value *table = &im->values[z.emb.table];
    const uint32_t V = (uint32_t)table->dims[0], d = (uint32_t)table->dims[1];
    uint32_t l = embedding(.net = im->net, .inputs = {0}, .input_count = 1, .vocabulary = V, .neurons_amount = d,
                           .positions = z.emb.pos != NO_VALUE, .act_func = act,
                           .weight_initialization = WEIGHT_INITIALIZATION_NONE);
    if (!lazy_added(im, li, l)) return SPINGALETT_NO_LAYER;
    int dtype;
    const uint8_t *src = constant_bytes(im, table, &dtype);
    if (!src) return SPINGALETT_NO_LAYER;
    float *w = layer_weights(im->net, l);
    spingalett_decode(w, src, dtype, (size_t)V * d);
    if (z.emb.tscale != 1.0f)
        for (size_t i = 0; i < (size_t)V * d; i++) w[i] *= z.emb.tscale;
    if (z.emb.pos != NO_VALUE) {
        Value *pos = &im->values[z.emb.pos];
        const float *p = constant_floats(im, pos);
        if (!p) return SPINGALETT_NO_LAYER;
        float *b = layer_biases(im->net, l);
        for (uint64_t i = 0; i < pos->count; i++) b[i] = z.emb.pscale * p[i];
    }
    return l;
}

/* ------------------------------------------------------------------------- element-wise expressions */

/* A new expression node: its index, NO_VALUE (error set) when the expression would grow too large or
   take the mean of a mean. */
static uint32_t enode(Importer *im, const Node *n, uint8_t op, uint32_t a, uint32_t b, float k, uint32_t vec) {
    unsigned size = 1;
    uint8_t form = op == E_K ? FORM_SCALAR : FORM_FULL;
    bool mean = false, vecs = op == E_VEC;
    if (op >= E_ADD && op <= E_POW) {
        const Enode *x = &im->enodes[a], *y = &im->enodes[b];
        size += (unsigned)x->size + y->size;
        form = x->form > y->form ? x->form : y->form;
        mean = x->mean || y->mean;
        vecs = x->vecs || y->vecs;
    } else if (op >= E_NEG) {
        const Enode *x = &im->enodes[a];
        if (op == E_MEAN && x->form != FORM_FULL) {
            node_fail(im, n, "takes the mean of a value that is not one a channel");
            return NO_VALUE;
        }
        size += x->size;
        form = op == E_MEAN ? FORM_REDUCED : x->form;
        mean = op == E_MEAN || x->mean;
        vecs = x->vecs;
    }
    if (size > EXPR_SIZE) {
        node_fail(im, n, "ends an element-wise expression of more than 255 operations");
        return NO_VALUE;
    }
    if (im->nenodes == im->ecap) {
        const size_t cap = im->ecap ? 2 * im->ecap : 256;
        Enode *grown = cap < NO_VALUE ? (Enode *)realloc(im->enodes, cap * sizeof(Enode)) : NULL;
        if (!grown) { out_of_memory(im); return NO_VALUE; }
        im->enodes = grown;
        im->ecap = cap;
    }
    im->enodes[im->nenodes] = (Enode){.op = op, .form = form, .mean = mean, .vecs = vecs, .size = (uint16_t)size,
                                      .a = a, .b = b, .vec = vec, .k = k};
    return (uint32_t)im->nenodes++;
}

/* Whether v can be what an expression is of: a layer's output, or a product or embedding not made yet. */
static bool expr_base(const Importer *im, const Value *v) {
    const int kind = kind_of(im, v);
    return kind == K_LAYER || kind == LZ_LINEAR || kind == LZ_EMBED;
}

/* The channels a constant may have one value each of when combined with x: a sequence's or a vector's
   (0: none). */
static uint32_t per_channel(const Value *x, int *rank) {
    if (x->seq) { *rank = x->ndims; return seq_channels(x); }
    if (x->vector && !x->flat) { *rank = 2; return x->c; }
    *rank = 0;
    return 0;
}

/* A constant as an operand of an expression of x: a scalar (E_K) or one value a channel (E_VEC). */
static uint32_t expr_const(Importer *im, const Node *n, Value *c, const Value *x) {
    const float *d = constant_floats(im, c);
    if (!d) return NO_VALUE;
    if (c->count == 1) return enode(im, n, E_K, 0, 0, d[0], NO_VALUE);
    int rank;
    const uint32_t C = per_channel(x, &rank);
    bool fits = C > 1 && c->count == C && c->ndims >= 1 && c->ndims <= rank && c->dims[c->ndims - 1] == (int64_t)C;
    if (!fits) {
        node_fail(im, n, "combines a computed tensor with a constant that is neither a scalar nor one value a channel");
        return NO_VALUE;
    }
    return enode(im, n, E_VEC, 0, 0, 0.0f, value_index(im, c));
}

typedef struct {
    bool rms;
    float eps;
    uint32_t gamma, beta;       /* E_VEC constants (value indices), or NO_VALUE: the scalars */
    float gk, bk;
} Norm;

static bool match_norm(const Importer *im, uint32_t root, Norm *nm);

/* Whether an expression is a whole normalization (with its scale and shift). */
static bool finished_norm(const Importer *im, uint32_t root) {
    Norm nm;
    return im->enodes[root].mean && im->enodes[root].form == FORM_FULL && match_norm(im, root, &nm);
}

/* Whether an operator reading expression v extends it rather than reading it made a layer: anything does
   until it is a whole normalization, which only a constant's sum or product (affine: its scale and shift)
   extends, and only as its one reader (a normalization read twice is read as it is). */
static bool extends(const Importer *im, const Value *v, bool affine) {
    if (!finished_norm(im, im->lazies[v->lazy - 1].ex.root)) return true;
    return affine && (v->name.n == 0 || readers(im, v->name) <= 1);
}

/* An operand of an element-wise operator as an expression node: an expression's root, or a new leaf of
   the value; *base gets the value the expression is of. affine: the operator adds or multiplies a
   constant (see extends()); a normalization it does not extend is made a layer first, and read. */
static uint32_t expr_operand(Importer *im, const Node *n, Value *v, bool affine, uint32_t *base) {
    if (kind_of(im, v) == LZ_EXPR) {
        const uint32_t li = v->lazy - 1, root = im->lazies[li].ex.root;
        if (extends(im, v, affine)) {
            *base = im->lazies[li].ex.base;
            return root;
        }
        if (!materialize(im, v)) return NO_VALUE;
    } else if (!expr_base(im, v)) {
        node_fail(im, n, "applies an element-wise operator to '%.*s', which no layer computes", (int)v->name.n,
                  v->name.s);
        return NO_VALUE;
    }
    *base = value_index(im, v);
    return enode(im, n, E_X, 0, 0, 0.0f, NO_VALUE);
}

/* What an operand's expression would be of (expr_operand's base), as source_of() tells it. */
static uint64_t operand_source(const Importer *im, const Value *v, bool affine) {
    if (kind_of(im, v) == LZ_EXPR && extends(im, v, affine))
        return source_of(&im->values[im->lazies[v->lazy - 1].ex.base]);
    return source_of(v);
}

/* A new value (named: output 0 of the node) standing for the expression root of the value base, shaped
   as base (one value a token when the expression is a mean's): its index, NO_VALUE on an error. */
static uint32_t expr_value(Importer *im, const Node *n, uint32_t base, uint32_t root, bool named) {
    if (root == NO_VALUE) return NO_VALUE;
    const uint32_t li = new_lazy(im, LZ_EXPR, n);
    if (li == NO_VALUE) return NO_VALUE;
    im->lazies[li].ex.base = base;
    im->lazies[li].ex.root = root;
    Value copy = im->values[base];
    int64_t dims[MAX_DIMS];
    const int nd = value_shape(&copy, dims);
    if (im->enodes[root].form != FORM_FULL && nd > 0) dims[nd - 1] = 1;
    Value *y = new_value(im, named ? n->outputs[0] : (Str){"", 0}, named);
    if (!y) return out_of_memory(im), NO_VALUE;
    const Str name = y->name;
    *y = copy;
    y->name = name;
    y->constant = false;
    y->alias = false;
    y->owned = false;
    y->data = NULL;
    y->layer = 0;
    y->lazy = li + 1;
    y->ndims = nd;
    memcpy(y->dims, dims, sizeof dims);
    return value_index(im, y);
}

static bool expr_output(Importer *im, const Node *n, uint32_t base, uint32_t root) {
    return expr_value(im, n, base, root, true) != NO_VALUE;
}

/* The expression operator of a unary ONNX operator (E_X: none of them). */
static uint8_t unary_expr(const Node *n) {
    const Str op = n->op;
    if (str_is(op, "Neg")) return E_NEG;
    if (str_is(op, "Sqrt")) return E_SQRT;
    if (str_is(op, "Reciprocal")) return E_RECIP;
    if (str_is(op, "Exp")) return E_EXP;
    if (str_is(op, "Erf")) return E_ERF;
    if (str_is(op, "Tanh")) return E_TANH;
    if (str_is(op, "Sigmoid")) return E_SIGMOID;
    if (str_is(op, "Relu")) return E_RELU;
    if (str_is(op, "Gelu")) {
        const Attr *a = attr(n, "approximate");
        return a && str_is(a->s, "tanh") ? E_GELU_TANH : E_GELU;
    }
    return E_X;
}

static bool expr_unary(Importer *im, const Node *n, uint8_t op, Value *x) {
    uint32_t base;
    const uint32_t a = expr_operand(im, n, x, false, &base);
    return a != NO_VALUE && expr_output(im, n, base, enode(im, n, op, a, 0, 0.0f, NO_VALUE));
}

/* a op b of one value as an expression. */
static bool expr_binary(Importer *im, const Node *n, uint8_t op, Value *a, Value *b) {
    uint32_t ba, bb;
    const uint32_t ea = expr_operand(im, n, a, false, &ba);
    if (ea == NO_VALUE) return false;
    const uint32_t eb = expr_operand(im, n, b, false, &bb);
    if (eb == NO_VALUE) return false;
    return expr_output(im, n, ba, enode(im, n, op, ea, eb, 0.0f, NO_VALUE));
}

/* An expression's value at x (a scalar function of its value; E_VEC and E_MEAN give NaN). */
static double expr_at(const Importer *im, uint32_t e, double x) {
    const Enode *n = &im->enodes[e];
    double a = n->op >= E_ADD && n->op != E_MEAN ? expr_at(im, n->a, x) : 0.0;
    double b = n->op >= E_ADD && n->op <= E_POW ? expr_at(im, n->b, x) : 0.0;
    switch (n->op) {
        case E_X: return x;
        case E_K: return n->k;
        case E_ADD: return a + b;
        case E_SUB: return a - b;
        case E_MUL: return a * b;
        case E_DIV: return a / b;
        case E_POW: return pow(a, b);
        case E_NEG: return -a;
        case E_SQRT: return sqrt(a);
        case E_RECIP: return 1.0 / a;
        case E_EXP: return exp(a);
        case E_ERF: return erf(a);
        case E_TANH: return tanh(a);
        case E_SIGMOID: return 1.0 / (1.0 + exp(-a));
        case E_RELU: return a > 0.0 ? a : 0.0;
        case E_GELU: return 0.5 * a * (1.0 + erf(a / sqrt(2.0)));
        case E_GELU_TANH: return 0.5 * a * (1.0 + tanh(0.7978845608028654 * (a + 0.044715 * a * a * a)));
        default: return NAN;
    }
}

static double activation_at(ActivationFunction act, double x) {
    switch (act) {
        case ACT_RELU: return x > 0.0 ? x : 0.0;
        case ACT_SIGMOID: return 1.0 / (1.0 + exp(-x));
        case ACT_TANH: return tanh(x);
        case ACT_LEAKY_RELU: return x > 0.0 ? x : 0.01 * x;
        case ACT_GELU: return 0.5 * x * (1.0 + erf(x / sqrt(2.0)));
        case ACT_GELU_TANH: return 0.5 * x * (1.0 + tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)));
        case ACT_SILU: return x / (1.0 + exp(-x));
        default: return x;
    }
}

/* The activation an expression of one value computes (ACT_NONE: the value itself; ACT_COUNT: none of
   them), told by its values at points over the range where activations differ: GELU's forms differ by
   up to 2e-4 there, far above the rounding of the constants exporters write. */
static ActivationFunction expr_activation(const Importer *im, uint32_t root) {
    static const double xs[] = {-7.0, -4.5, -3.0, -2.2, -1.6, -1.1, -0.75, -0.4, -0.15, 0.0, 0.1, 0.3, 0.6, 0.95, 1.4,
                                2.1, 2.9, 4.0, 6.5};
    static const ActivationFunction acts[] = {ACT_NONE, ACT_RELU, ACT_SIGMOID, ACT_TANH, ACT_LEAKY_RELU, ACT_GELU,
                                              ACT_GELU_TANH, ACT_SILU};
    double f[sizeof xs / sizeof xs[0]];
    for (size_t i = 0; i < sizeof xs / sizeof xs[0]; i++) f[i] = expr_at(im, root, xs[i]);
    for (size_t a = 0; a < sizeof acts / sizeof acts[0]; a++) {
        bool all = true;
        for (size_t i = 0; all && i < sizeof xs / sizeof xs[0]; i++) {
            const double g = activation_at(acts[a], xs[i]);
            all = fabs(f[i] - g) <= 1e-5 * (1.0 + fabs(g));
        }
        if (all) return acts[a];
    }
    return ACT_COUNT;
}

/* The distinct leaves of an expression: the reads of its value it stands for. */
static uint32_t expr_leaves(const Importer *im, uint32_t root) {
    uint32_t stack[2 * EXPR_SIZE + 2], seen[EXPR_SIZE + 1], leaves = 0;
    int top = 0, nseen = 0;
    stack[top++] = root;
    while (top > 0) {
        const uint32_t e = stack[--top];
        bool dup = false;
        for (int i = 0; i < nseen && !dup; i++) dup = seen[i] == e;
        if (dup) continue;
        seen[nseen++] = e;
        const Enode *x = &im->enodes[e];
        if (x->op == E_X) leaves++;
        if (x->op >= E_ADD) stack[top++] = x->a;
        if (x->op >= E_ADD && x->op <= E_POW) stack[top++] = x->b;
    }
    return leaves;
}

/* ---- normalizations by their form */

static const Enode *en(const Importer *im, uint32_t e) { return &im->enodes[e]; }
static bool is_k(const Importer *im, uint32_t e, float k) { return en(im, e)->op == E_K && en(im, e)->k == k; }
static bool is_const_e(const Importer *im, uint32_t e) { return en(im, e)->op == E_K || en(im, e)->op == E_VEC; }

static bool same_e(const Importer *im, uint32_t a, uint32_t b) {
    if (a == b) return true;
    const Enode *x = en(im, a), *y = en(im, b);
    if (x->op != y->op || x->size != y->size) return false;
    if (x->op == E_X) return true;
    if (x->op == E_K) return x->k == y->k;
    if (x->op == E_VEC) return x->vec == y->vec;
    if (!same_e(im, x->a, y->a)) return false;
    return x->op > E_POW || same_e(im, x->b, y->b);
}

/* t = inner^2: Pow(inner, 2) or inner * inner. */
static bool square_of(const Importer *im, uint32_t t, uint32_t *inner) {
    const Enode *x = en(im, t);
    if (x->op == E_POW && is_k(im, x->b, 2.0f)) { *inner = x->a; return true; }
    if (x->op == E_MUL && same_e(im, x->a, x->b)) { *inner = x->a; return true; }
    return false;
}

/* t = m + eps (eps 0: t = m). */
static void plus_eps(const Importer *im, uint32_t t, uint32_t *m, float *eps) {
    const Enode *x = en(im, t);
    *m = t;
    *eps = 0.0f;
    if (x->op == E_ADD && en(im, x->b)->op == E_K) *m = x->a, *eps = en(im, x->b)->k;
    else if (x->op == E_ADD && en(im, x->a)->op == E_K) *m = x->b, *eps = en(im, x->a)->k;
}

/* t = sqrt(m + eps), or (m + eps)^0.5. */
static bool std_of(const Importer *im, uint32_t t, uint32_t *m, float *eps) {
    const Enode *x = en(im, t);
    if (x->op == E_SQRT) { plus_eps(im, x->a, m, eps); return true; }
    if (x->op == E_POW && is_k(im, x->b, 0.5f)) { plus_eps(im, x->a, m, eps); return true; }
    return false;
}

/* t = 1 / sqrt(m + eps): Reciprocal, 1 / sqrt, (m + eps)^-0.5, sqrt(1 / (m + eps)). */
static bool rstd_of(const Importer *im, uint32_t t, uint32_t *m, float *eps) {
    const Enode *x = en(im, t);
    if (x->op == E_RECIP) return std_of(im, x->a, m, eps);
    if (x->op == E_DIV && is_k(im, x->a, 1.0f)) return std_of(im, x->b, m, eps);
    if (x->op == E_POW && is_k(im, x->b, -0.5f)) { plus_eps(im, x->a, m, eps); return true; }
    if (x->op == E_SQRT && en(im, x->a)->op == E_RECIP) { plus_eps(im, en(im, x->a)->a, m, eps); return true; }
    return false;
}

/* A normalization of the expression's value over its channels with its scale and shift:
   gamma x / sqrt(mean(x^2) + eps) (RMS) or gamma (x - mean(x)) / sqrt(mean((x - mean(x))^2) + eps) + beta,
   as exporters write them out. */
static bool match_norm(const Importer *im, uint32_t root, Norm *nm) {
    uint32_t t = root;
    nm->gamma = nm->beta = NO_VALUE;
    nm->gk = 1.0f;
    nm->bk = 0.0f;
    const Enode *x = en(im, t);
    /* the shift, then the scale (a product, or a quotient by a scalar) */
    if (x->op == E_ADD && (is_const_e(im, x->a) || is_const_e(im, x->b))) {
        const uint32_t c = is_const_e(im, x->b) ? x->b : x->a;
        t = c == x->b ? x->a : x->b;
        if (en(im, c)->op == E_VEC) nm->beta = en(im, c)->vec;
        else nm->bk = en(im, c)->k;
        x = en(im, t);
    }
    if (x->op == E_MUL && (is_const_e(im, x->a) || is_const_e(im, x->b))) {
        const uint32_t c = is_const_e(im, x->b) ? x->b : x->a;
        t = c == x->b ? x->a : x->b;
        if (en(im, c)->op == E_VEC) nm->gamma = en(im, c)->vec;
        else nm->gk = en(im, c)->k;
        x = en(im, t);
    } else if (x->op == E_DIV && en(im, x->b)->op == E_K && en(im, x->b)->k != 0.0f) {
        nm->gk = 1.0f / en(im, x->b)->k;
        t = x->a;
        x = en(im, t);
    }
    /* the value normalized times the reciprocal of the root, or divided by the root */
    uint32_t v = 0, m = 0, sq = 0, inner = 0;
    float eps = 0.0f;
    bool found = false;
    if (x->op == E_MUL) {
        if (rstd_of(im, x->b, &m, &eps)) v = x->a, found = true;
        else if (rstd_of(im, x->a, &m, &eps)) v = x->b, found = true;
    } else if (x->op == E_DIV) {
        if (std_of(im, x->b, &m, &eps)) v = x->a, found = true;
    }
    if (!found || !(eps > 0.0f && eps < 1.0f) || en(im, m)->op != E_MEAN) return false;
    if (!square_of(im, en(im, m)->a, &sq)) return false;
    inner = sq;
    const Enode *ve = en(im, v);
    if (ve->op == E_X && en(im, inner)->op == E_X) {
        nm->rms = true;
    } else if (ve->op == E_SUB && en(im, ve->a)->op == E_X && en(im, ve->b)->op == E_MEAN &&
               en(im, en(im, ve->b)->a)->op == E_X && same_e(im, inner, v)) {
        nm->rms = false;
    } else {
        return false;
    }
    if (nm->rms && (nm->beta != NO_VALUE || nm->bk != 0.0f)) return false;
    nm->eps = eps;
    return true;
}

/* A normalization layer reading the value base. */
static uint32_t make_norm(Importer *im, uint32_t li, uint32_t bi, const Norm *nm) {
    Value *b = &im->values[bi];
    if (!materialize(im, b)) return SPINGALETT_NO_LAYER;
    int rank;
    const uint32_t C = per_channel(b, &rank);
    if (C == 0) {
        lazy_fail(im, li, "normalizes other axes than the channels of a sequence or a vector");
        return SPINGALETT_NO_LAYER;
    }
    const float *gamma = nm->gamma != NO_VALUE ? constant_floats(im, &im->values[nm->gamma]) : NULL;
    const float *beta = nm->beta != NO_VALUE ? constant_floats(im, &im->values[nm->beta]) : NULL;
    if ((nm->gamma != NO_VALUE && !gamma) || (nm->beta != NO_VALUE && !beta)) return SPINGALETT_NO_LAYER;
    const uint32_t l = nm->rms ? rms_norm(.net = im->net, .inputs = {b->layer}, .input_count = 1, .epsilon = nm->eps)
                               : layer_norm(.net = im->net, .inputs = {b->layer}, .input_count = 1, .epsilon = nm->eps);
    if (!lazy_added(im, li, l)) return SPINGALETT_NO_LAYER;
    float *w = layer_weights(im->net, l);
    for (uint32_t c = 0; c < C; c++) w[c] = nm->gk * (gamma ? gamma[c] : 1.0f);
    if (!nm->rms) {
        float *s = layer_biases(im->net, l);
        for (uint32_t c = 0; c < C; c++) s[c] = beta ? beta[c] : nm->bk;
    }
    return l;
}

/* The activation act applied to the value base: a product or embedding made with it, a layer of its
   own with it fused when the expression stands for every read of the layer, or a layer of one input. */
static uint32_t apply_activation(Importer *im, uint32_t li, uint32_t bi, ActivationFunction act, uint32_t root) {
    Value *b = &im->values[bi];
    const int kind = kind_of(im, b);
    if (act != ACT_NONE && kind == LZ_LINEAR && im->lazies[b->lazy - 1].lin.act == ACT_NONE)
        return make_linear(im, b->lazy - 1, act);
    if (act != ACT_NONE && kind == LZ_EMBED && im->lazies[b->lazy - 1].emb.act == ACT_NONE)
        return make_embedding(im, b->lazy - 1, act);
    if (!materialize(im, b)) return SPINGALETT_NO_LAYER;
    const uint32_t L = b->layer;
    if (act == ACT_NONE) return L;
    NeuralNetwork *net = im->net;
    const LayerType type = net->shapes[L].type;
    if (L > 0 && net->act_func[L - 1] == ACT_NONE && type != LAYER_MAX_POOL2D && type != LAYER_AVG_POOL2D &&
        type != LAYER_UPSAMPLE && type != LAYER_GLOBAL_AVG_POOL && layer_readers(im, L) == expr_leaves(im, root)) {
        net->act_func[L - 1] = act;
        return L;
    }
    const uint32_t l = add_layers(.net = net, .inputs = {L}, .input_count = 1, .act_func = act);
    return lazy_added(im, li, l) ? l : SPINGALETT_NO_LAYER;
}

/* An expression made a layer: an activation of its value, or a normalization of it. */
static uint32_t make_expr(Importer *im, uint32_t li) {
    const uint32_t root = im->lazies[li].ex.root, bi = im->lazies[li].ex.base;
    const Enode *e = &im->enodes[root];
    if (e->form != FORM_FULL) {
        lazy_fail(im, li, "computes one value a token (a mean), which no layer outputs on its own");
        return SPINGALETT_NO_LAYER;
    }
    if (e->mean) {
        Norm nm;
        if (!match_norm(im, root, &nm)) {
            lazy_fail(im, li, "computes means over the channels in a form that is neither layer nor RMS normalization");
            return SPINGALETT_NO_LAYER;
        }
        return make_norm(im, li, bi, &nm);
    }
    if (e->vecs) {
        lazy_fail(im, li, "adds or multiplies a constant a channel that is no layer's bias or normalization's scale");
        return SPINGALETT_NO_LAYER;
    }
    const ActivationFunction act = expr_activation(im, root);
    if (act == ACT_COUNT) {
        lazy_fail(im, li, "computes an element-wise function that is none of the activations Spingalett has");
        return SPINGALETT_NO_LAYER;
    }
    return apply_activation(im, li, bi, act, root);
}

/* ------------------------------------------------------------------------- views of sequences */

static uint64_t dim_size(const View *v, int d) {
    uint64_t s = 1;
    for (int k = 0; k < v->nf[d]; k++) s *= v->f[d][k].size;
    return s;
}

static int view_shape(const View *v, int64_t *dims) {
    for (int d = 0; d < v->nd; d++) dims[d] = (int64_t)dim_size(v, d);
    return v->nd;
}

/* A sequence as it is, [1, T, C] or [T, C]. */
static void plain_view(const Value *s, uint32_t si, View *v) {
    memset(v, 0, sizeof *v);
    v->base = si;
    v->scale = 1.0f;
    v->nd = s->ndims;
    const uint32_t T = seq_tokens(s), C = seq_channels(s);
    if (T > 1) v->f[v->nd - 2][v->nf[v->nd - 2]++] = (Factor){T, 1, true};
    if (C > 1) v->f[v->nd - 1][v->nf[v->nd - 1]++] = (Factor){C, 1, false};
}

/* Whether v is a sequence a view may be of: a sequence layer, product, embedding or expression (not a
   mean's). */
static bool seq_base(const Importer *im, const Value *v) {
    const int kind = kind_of(im, v);
    if (!v->seq || v->ndims < 2 || v->ndims > 3 || (v->ndims == 3 && v->dims[0] != 1)) return false;
    if (kind == LZ_EXPR) return im->enodes[im->lazies[v->lazy - 1].ex.root].form == FORM_FULL;
    return kind == K_LAYER || kind == LZ_LINEAR || kind == LZ_EMBED;
}

/* The view of value vi: a view's own, or a sequence's plain one. */
static bool view_of(Importer *im, const Node *n, uint32_t vi, View *v) {
    const Value *x = &im->values[vi];
    if (kind_of(im, x) == LZ_VIEW) {
        *v = im->lazies[x->lazy - 1].view.v;
        return true;
    }
    if (seq_base(im, x)) {
        plain_view(x, vi, v);
        return true;
    }
    return node_fail(im, n, "rearranges '%.*s', which is not a sequence of tokens", (int)x->name.n, x->name.s);
}

/* Whether an axis's factors run over consecutive channels from the first (row-major); *count: how many. */
static bool chan_run(const Factor *f, int nf, uint32_t *count) {
    uint64_t stride = 1;
    for (int k = nf - 1; k >= 0; k--) {
        if (f[k].token || f[k].stride != stride) return false;
        stride *= f[k].size;
    }
    *count = (uint32_t)stride;
    return true;
}

/* Whether a view shows its sequence as it is: [1, T, C] or [T, C], every channel in order, unscaled. */
static bool view_plain(const Importer *im, const View *v) {
    const Value *b = &im->values[v->base];
    if (v->offset || v->scale != 1.0f || v->theta != 0.0f || (v->nd != 2 && v->nd != 3)) return false;
    if (v->nd == 3 && v->nf[0] != 0) return false;
    const int td = v->nd - 2, cd = v->nd - 1;
    const uint32_t T = seq_tokens(b), C = seq_channels(b);
    const bool tokens = T == 1 ? v->nf[td] == 0 : v->nf[td] == 1 && v->f[td][0].token && v->f[td][0].size == T;
    uint32_t count;
    return tokens && chan_run(v->f[cd], v->nf[cd], &count) && count == C;
}

/* Whether a view holds whole tokens: every axis but the last at most the tokens (all of them), the last
   a run of channels (from *off, *count of them), unscaled: what per-token operators read. */
static bool view_per_token(const Importer *im, const View *v, uint32_t *off, uint32_t *count) {
    if (v->nd < 1 || v->scale != 1.0f || v->theta != 0.0f) return false;
    const uint32_t T = seq_tokens(&im->values[v->base]);
    for (int d = 0; d < v->nd - 1; d++) {
        if (v->nf[d] > 1 || (v->nf[d] == 1 && (!v->f[d][0].token || v->f[d][0].size != T))) return false;
    }
    *off = v->offset;
    return chan_run(v->f[v->nd - 1], v->nf[v->nd - 1], count);
}

/* The factors of a view in order, those of size one left out. */
static int view_factors(const View *v, Factor *q) {
    int n = 0;
    for (int d = 0; d < v->nd; d++)
        for (int k = 0; k < v->nf[d]; k++)
            if (v->f[d][k].size > 1) q[n++] = v->f[d][k];
    return n;
}

/* Reshape: the factors regrouped into axes of the target's sizes, a factor of channels split where an
   axis ends inside it. The tokens are never split. */
static bool view_reshape(View *v, const int64_t *target, int nt) {
    Factor q[MAX_DIMS * MAX_FACTORS];
    const int nq = view_factors(v, q);
    int qi = 0;
    View r = *v;
    r.nd = nt;
    for (int d = 0; d < nt; d++) {
        r.nf[d] = 0;
        uint64_t need = (uint64_t)target[d];
        if (need == 0) return false;
        while (need > 1) {
            if (qi >= nq || r.nf[d] == MAX_FACTORS) return false;
            const Factor f = q[qi];
            if (need % f.size == 0) {
                r.f[d][r.nf[d]++] = f;
                need /= f.size;
                qi++;
            } else if (f.size % need == 0 && !f.token) {
                const uint64_t minor = f.size / need, stride = (uint64_t)f.stride * minor;
                if (stride > UINT32_MAX) return false;
                r.f[d][r.nf[d]++] = (Factor){(uint32_t)need, (uint32_t)stride, false};
                q[qi] = (Factor){(uint32_t)minor, f.stride, false};
                need = 1;
            } else {
                return false;
            }
        }
    }
    if (qi != nq) return false;
    *v = r;
    return true;
}

static bool view_transpose(View *v, const int64_t *perm) {
    View r = *v;
    for (int d = 0; d < v->nd; d++) {
        r.nf[d] = v->nf[perm[d]];
        memcpy(r.f[d], v->f[perm[d]], sizeof r.f[d]);
    }
    *v = r;
    return true;
}

/* Unsqueeze (insert) or Squeeze of the given axes (Squeeze of none: every axis of size one). */
static bool view_squeeze(View *v, const int64_t *axes, int na, bool insert) {
    const int r = insert ? v->nd + na : v->nd;
    bool mark[MAX_DIMS] = {false};
    if (r > MAX_DIMS) return false;
    for (int k = 0; k < na; k++) {
        int a;
        if (!norm_axis(axes[k], r, &a) || mark[a] || (!insert && dim_size(v, a) != 1)) return false;
        mark[a] = true;
    }
    View x = *v;
    int m = 0;
    for (int i = 0, j = 0; i < r; i++) {
        int from;
        if (insert) {
            if (mark[i]) { x.nf[m++] = 0; continue; }
            from = j++;
        } else {
            if (na ? mark[i] : dim_size(v, i) == 1) continue;
            from = i;
        }
        x.nf[m] = v->nf[from];
        memcpy(x.f[m], v->f[from], sizeof x.f[m]);
        m++;
    }
    x.nd = m;
    *v = x;
    return true;
}

/* Slice of [start, end) along an axis (step 1): a range of a run of channels moves the offset. */
static bool view_slice(View *v, int axis, int64_t start, int64_t count) {
    const uint64_t size = dim_size(v, axis);
    if (start == 0 && (uint64_t)count == size) return true;
    if (count <= 0 || v->nf[axis] != 1 || v->f[axis][0].token) return false;
    Factor *f = &v->f[axis][0];
    const uint64_t offset = v->offset + (uint64_t)start * f->stride;
    if (offset > UINT32_MAX) return false;
    v->offset = (uint32_t)offset;
    f->size = (uint32_t)count;
    if (f->size == 1) v->nf[axis] = 0;
    return true;
}

/* Expand: axes of size one repeated to the target's sizes (numpy's broadcasting). */
static bool view_expand(View *v, const int64_t *shape, int ns) {
    const int r = v->nd > ns ? v->nd : ns;
    if (r > MAX_DIMS) return false;
    View x = *v;
    x.nd = r;
    for (int d = r - 1; d >= 0; d--) {
        const int sd = d - (r - v->nd);
        x.nf[d] = sd >= 0 ? v->nf[sd] : 0;
        if (sd >= 0) memcpy(x.f[d], v->f[sd], sizeof x.f[d]);
    }
    for (int d = 0; d < r; d++) {
        const int64_t t = d - (r - ns) >= 0 ? shape[d - (r - ns)] : 1;
        const uint64_t s = dim_size(&x, d);
        if (t == (int64_t)s || t == 1) continue;
        if (s != 1 || t < 1 || t > UINT32_MAX || x.nf[d] == MAX_FACTORS) return false;
        x.f[d][x.nf[d]++] = (Factor){(uint32_t)t, 0, false};
    }
    *v = x;
    return true;
}

static bool same_view(const Importer *im, const View *a, const View *b) {
    if (source_of(&im->values[a->base]) != source_of(&im->values[b->base]) || a->offset != b->offset ||
        a->nd != b->nd || a->scale != b->scale || a->theta != b->theta || a->rope_head != b->rope_head)
        return false;
    for (int d = 0; d < a->nd; d++) {
        if (a->nf[d] != b->nf[d]) return false;
        for (int k = 0; k < a->nf[d]; k++)
            if (a->f[d][k].size != b->f[d][k].size || a->f[d][k].stride != b->f[d][k].stride ||
                a->f[d][k].token != b->f[d][k].token)
                return false;
    }
    return true;
}

/* Output k of a node: a view, as the sequence itself when it shows it plainly. */
static bool view_output(Importer *im, const Node *n, int k, const View *v) {
    int64_t dims[MAX_DIMS];
    const int nd = view_shape(v, dims);
    if (view_plain(im, v)) {
        Value copy = im->values[v->base];
        Value *y = add_value(im, n->outputs[k]);
        if (!y) return out_of_memory(im);
        const Str name = y->name;
        *y = copy;
        y->name = name;
        y->alias = false;           /* a read of the sequence through other views may come to it too */
        y->ndims = nd;
        memcpy(y->dims, dims, sizeof dims);
        return true;
    }
    const uint32_t li = new_lazy(im, LZ_VIEW, n);
    if (li == NO_VALUE) return false;
    im->lazies[li].view.v = *v;
    im->lazies[li].view.rows = NO_VALUE;
    im->lazies[li].view.table = NO_VALUE;
    return lazy_value(im, n->outputs[k], true, li, dims, nd, false) != NULL;
}

/* The sequence holding a per-token view's channels: its base when the view shows all of them, a product
   of the base's rows when it shows some (made once a view); *arr gets the view. NO_VALUE (error set)
   otherwise. */
static uint32_t view_rows(Importer *im, const Node *n, uint32_t vi, View *arr) {
    const Value *x = &im->values[vi];
    if (kind_of(im, x) != LZ_VIEW) {
        if (!seq_base(im, x)) {
            node_fail(im, n, "reads '%.*s', which is not a sequence of tokens", (int)x->name.n, x->name.s);
            return NO_VALUE;
        }
        plain_view(x, vi, arr);
        return vi;
    }
    const uint32_t li = x->lazy - 1;
    *arr = im->lazies[li].view.v;
    uint32_t off, count;
    if (!view_per_token(im, arr, &off, &count)) {
        node_fail(im, n, "reads tokens' channels rearranged (in heads, or transposed) where a layer reads tokens");
        return NO_VALUE;
    }
    const uint32_t bi = arr->base;
    const Value *b = &im->values[bi];
    if (off == 0 && count == seq_channels(b)) return bi;
    if (im->lazies[li].view.rows != NO_VALUE) return im->lazies[li].view.rows;
    if (kind_of(im, b) != LZ_LINEAR) {
        node_fail(im, n, "reads part of the channels of '%.*s', which no layer selects", (int)b->name.n, b->name.s);
        return NO_VALUE;
    }
    /* the product of those rows of the base's */
    const uint32_t src = b->lazy - 1;
    const int nd = b->ndims;
    int64_t dims[3];
    memcpy(dims, b->dims, (size_t)nd * sizeof *dims);
    dims[nd - 1] = count;
    const uint32_t sl = new_lazy(im, LZ_LINEAR, n);
    if (sl == NO_VALUE) return NO_VALUE;
    Lazy *z = &im->lazies[sl];
    const Lazy *p = &im->lazies[src];
    z->lin = p->lin;
    z->lin.row0 = p->lin.row0 + off;
    z->lin.out = count;
    z->lin.scale = z->lin.bias = NULL;
    if (p->lin.scale && !(z->lin.scale = (float *)malloc(count * sizeof(float)))) return out_of_memory(im), NO_VALUE;
    if (p->lin.bias && !(z->lin.bias = (float *)malloc(count * sizeof(float)))) return out_of_memory(im), NO_VALUE;
    if (p->lin.scale) memcpy(z->lin.scale, p->lin.scale + off, count * sizeof(float));
    if (p->lin.bias) memcpy(z->lin.bias, p->lin.bias + off, count * sizeof(float));
    Value *y = lazy_value(im, (Str){"", 0}, false, sl, dims, nd, true);
    if (!y) return NO_VALUE;
    const uint32_t yi = value_index(im, y);
    im->lazies[li].view.rows = yi;
    return yi;
}

/* Output k of a node: the sequence si arranged as the per-token view arr (its channels si's). */
static bool rows_output(Importer *im, const Node *n, int k, uint32_t si, const View *arr) {
    View v = *arr;
    v.base = si;
    v.offset = 0;
    v.scale = 1.0f;
    v.theta = 0.0f;
    v.rope_head = 0;
    const uint32_t C = seq_channels(&im->values[si]);
    const int cd = v.nd - 1;
    v.nf[cd] = 0;
    if (C > 1) v.f[cd][v.nf[cd]++] = (Factor){C, 1, false};
    return view_output(im, n, k, &v);
}

/* Whether two per-token views arrange their tokens alike (their channels aside). */
static bool same_arrangement(const View *a, const View *b) {
    if (a->nd != b->nd) return false;
    for (int d = 0; d < a->nd - 1; d++)
        if (a->nf[d] != b->nf[d]) return false;
    return true;
}

/* ------------------------------------------------------------------------- rotary embeddings */

/* The table a view is multiplied by as [T][head] at table + t ts + i hs: broadcast against the view, it
   varies along its tokens and its last axis (a head of channels in order) only. */
static bool rope_table(Importer *im, const View *x, Value *tv, const float **data, uint64_t *ts, uint64_t *hs,
                       uint32_t *T, uint32_t *head) {
    const int cd = x->nd - 1;
    if (x->nd < 1 || x->nf[cd] != 1 || x->f[cd][0].token || x->f[cd][0].stride != 1) return false;
    *head = x->f[cd][0].size;
    int td = -1;
    for (int d = 0; d < cd; d++)
        if (x->nf[d] == 1 && x->f[d][0].token) td = d;
    *T = td >= 0 ? x->f[td][0].size : 1;
    if (tv->ndims > x->nd + 2) return false;
    const float *d = constant_floats(im, tv);
    if (!d) return false;
    *data = d;
    *ts = *hs = 0;
    uint64_t stride = 1;
    for (int j = tv->ndims - 1; j >= 0; j--) {
        const int i = j - (tv->ndims - x->nd);
        const int64_t size = tv->dims[j];
        if (size != 1) {
            if (i == td && size == (int64_t)*T) *ts = stride;
            else if (i == cd && size == (int64_t)*head) *hs = stride;
            else return false;
        }
        stride *= (uint64_t)size;
    }
    return *hs != 0 || *head == 1;
}

/* The base theta of rotary embeddings whose tables are cs and sn (as rope_table() gives them):
   cos and sin of t theta^(-2i / head) at position t, frequency i of each half of a head. */
static bool rope_theta(const float *cs, const float *sn, uint64_t ts, uint64_t hs, uint32_t T, uint32_t head,
                       float *theta) {
    if (head < 2 || head % 2) return false;
    const uint32_t half = head / 2;
    double th = 10000.0;
    if (T > 1 && half > 1) {
        const double a = atan2(sn[ts + hs], cs[ts + hs]);         /* position 1, frequency 1: theta^(-2 / head) */
        if (!(a > 0.0 && a < 1.0)) return false;
        th = exp(-(double)half * log(a));
    }
    if (!(th > 1.0 && th < 1e30)) return false;
    for (uint32_t t = 0; t < T; t++)
        for (uint32_t i = 0; i < half; i++) {
            const double angle = t * pow(th, -2.0 * i / head), tol = 1e-5 + 5e-7 * t;
            for (uint32_t h = 0; h < head; h += half) {
                const uint64_t at = t * ts + (i + h) * hs;
                if (!(fabs(cs[at] - cos(angle)) <= tol && fabs(sn[at] - sin(angle)) <= tol)) return false;
            }
        }
    *theta = (float)th;
    return true;
}

/* rotate_half: Concat(-x2, x1) of a view's heads' halves along its last axis, from a (-x2) and b (x1). */
static bool rot_half(const Importer *im, const View *a, const View *b, View *x) {
    const int cd = a->nd - 1;
    if (a->nd != b->nd || a->nd < 1 || source_of(&im->values[a->base]) != source_of(&im->values[b->base]) ||
        a->theta != 0.0f || b->theta != 0.0f || a->scale != -b->scale)
        return false;
    for (int d = 0; d < cd; d++) {
        if (a->nf[d] != b->nf[d]) return false;
        for (int k = 0; k < a->nf[d]; k++)
            if (a->f[d][k].size != b->f[d][k].size || a->f[d][k].stride != b->f[d][k].stride ||
                a->f[d][k].token != b->f[d][k].token)
                return false;
    }
    if (a->nf[cd] != 1 || b->nf[cd] != 1) return false;
    const Factor fa = a->f[cd][0], fb = b->f[cd][0];
    if (fa.token || fb.token || fa.stride != 1 || fb.stride != 1 || fa.size != fb.size ||
        a->offset != b->offset + fb.size)
        return false;
    *x = *b;
    x->f[cd][0].size = 2 * fb.size;
    return true;
}

/* ------------------------------------------------------------------------- attention */

/* The heads of a view whose last two axes are tokens and a head (transposed: a head and tokens): the
   factors of its leading axes broadcast against lead (an axis of one where lead's is longer is
   repeated: stride 0), adjacent ones merged where one runs on into the other. */
typedef struct {
    Factor f[MAX_DIMS * MAX_FACTORS];
    int n;
    uint32_t T, head, hstride;
} Heads;

static bool heads_of(const View *v, bool transposed, const int64_t *lead, int nlead, Heads *h) {
    if (v->nd < 2 || v->nd - 2 > nlead) return false;
    const int td = transposed ? v->nd - 1 : v->nd - 2, hd = transposed ? v->nd - 2 : v->nd - 1;
    if (v->nf[td] > 1 || (v->nf[td] == 1 && !v->f[td][0].token)) return false;
    if (v->nf[hd] > 1 || (v->nf[hd] == 1 && v->f[hd][0].token)) return false;
    h->T = v->nf[td] ? v->f[td][0].size : 1;
    h->head = v->nf[hd] ? v->f[hd][0].size : 1;
    h->hstride = v->nf[hd] ? v->f[hd][0].stride : 1;
    h->n = 0;
    for (int i = 0; i < nlead; i++) {
        const int d = i - (nlead - (v->nd - 2));
        const uint64_t s = d >= 0 ? dim_size(v, d) : 1;
        if (s == (uint64_t)lead[i]) {
            for (int k = 0; d >= 0 && k < v->nf[d]; k++) {
                if (v->f[d][k].token) return false;
                h->f[h->n++] = v->f[d][k];
            }
        } else if (s == 1 && lead[i] <= UINT32_MAX) {
            h->f[h->n++] = (Factor){(uint32_t)lead[i], 0, false};
        } else {
            return false;
        }
    }
    int m = 0;
    for (int k = 0; k < h->n; k++) {
        if (h->f[k].size == 1) continue;
        if (m > 0 && (uint64_t)h->f[m - 1].stride == (uint64_t)h->f[k].stride * h->f[k].size &&
            (uint64_t)h->f[m - 1].size * h->f[k].size <= UINT32_MAX) {
            h->f[m - 1].size *= h->f[k].size;
            h->f[m - 1].stride = h->f[k].stride;
        } else {
            h->f[m++] = h->f[k];
        }
    }
    h->n = m;
    return true;
}

static void free_distinct(float **m, int count) {
    for (int i = 0; i < count; i++) {
        bool dup = false;
        for (int j = 0; j < i; j++) dup = dup || m[j] == m[i];
        if (!dup) free(m[i]);
    }
}

/* The key and value heads of Heads h for H query heads: kv of them (stride: channels from one to the
   next), each read by H / kv query heads in a row, as the attention layer groups them. */
static bool kv_heads(const Heads *h, uint32_t H, uint32_t *kv, uint32_t *stride) {
    *stride = 0;
    if (h->n == 0) { *kv = 1; return H == 1; }
    if (h->n == 1 && h->f[0].stride == 0) { *kv = 1; return h->f[0].size == H; }
    if (h->n == 1) { *kv = h->f[0].size; *stride = h->f[0].stride; return *kv == H; }
    if (h->n == 2 && h->f[0].stride != 0 && h->f[1].stride == 0) {
        *kv = h->f[0].size;
        *stride = h->f[0].stride;
        return (uint64_t)*kv * h->f[1].size == H;
    }
    return false;
}

/* Attention of the queries q ([.., T, head]) over the keys k (transposed: [.., head, T]) and values v
   ([.., T, head]), their product times scale, causal or not, lead the scores' leading axes: one attention
   layer, its input the queries', keys' and values' products merged into one linear layer (the scale
   beyond 1 / sqrt(head) and the values' scale folded into their weights), or the layer holding all three
   in its order, or their layers side by side. Output k of the node is its result arranged as the
   queries were ([.., T, head]), or, plain, the layer itself. */
static bool make_attention(Importer *im, const Node *n, const View *q, const View *k, const View *v, float scale,
                           bool causal, const int64_t *lead, int nlead, int out, bool plain) {
    Heads qh, kh, vh;
    if (!heads_of(q, false, lead, nlead, &qh) || !heads_of(k, true, lead, nlead, &kh) ||
        !heads_of(v, false, lead, nlead, &vh))
        return node_fail(im, n, "multiplies queries, keys and values that are not split into heads as the attention "
                         "layer reads them");
    if (qh.n > 1 || (qh.n == 1 && qh.f[0].stride == 0))
        return node_fail(im, n, "repeats or splits the queries' heads in a way the attention layer does not");
    const uint32_t H = qh.n ? qh.f[0].size : 1, qst = qh.n ? qh.f[0].stride : 0;
    uint32_t KV, kst, kv2, vst;
    if (!kv_heads(&kh, H, &KV, &kst) || !kv_heads(&vh, H, &kv2, &vst) || kv2 != KV)
        return node_fail(im, n, "groups the keys' and values' heads other than consecutive query heads per key head");
    const uint32_t T = qh.T, hd = qh.head;
    if (kh.T != T || vh.T != T || kh.head != hd || vh.head != hd)
        return node_fail(im, n, "attends with queries, keys and values of different sizes");
    if ((q->theta == 0.0f) != (k->theta == 0.0f) || v->theta != 0.0f ||
        (q->theta != 0.0f &&
         (fabsf(q->theta - k->theta) > 1e-4f * q->theta || q->rope_head != hd || k->rope_head != hd)))
        return node_fail(im, n, "rotates the queries and the keys differently (rotary embeddings)");
    const float f = scale * q->scale * k->scale * sqrtf((float)hd), vf = v->scale;
    const uint32_t C3 = (H + 2 * KV) * hd;
    Value *qb = &im->values[q->base], *kb = &im->values[k->base], *vb = &im->values[v->base];
    NeuralNetwork *net = im->net;
    uint32_t input;
    const bool pending = kind_of(im, qb) == LZ_LINEAR && kind_of(im, kb) == LZ_LINEAR && kind_of(im, vb) == LZ_LINEAR;
    const Lazy *lq = pending ? &im->lazies[qb->lazy - 1] : NULL, *lk = pending ? &im->lazies[kb->lazy - 1] : NULL;
    const Lazy *lv = pending ? &im->lazies[vb->lazy - 1] : NULL;
    if (pending && lq->lin.act == ACT_NONE && lk->lin.act == ACT_NONE && lv->lin.act == ACT_NONE &&
        lq->lin.src == lk->lin.src && lq->lin.src == lv->lin.src && lq->lin.in == lk->lin.in &&
        lq->lin.in == lv->lin.in) {
        /* the three products merged: rows of queries (times f), keys and values (times vf) */
        const uint32_t in = lq->lin.in, src = lq->lin.src;
        const uint32_t lis[3] = {qb->lazy - 1, kb->lazy - 1, vb->lazy - 1};
        float *m[3] = {NULL, NULL, NULL};
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < i && !m[i]; j++)
                if (lis[j] == lis[i]) m[i] = m[j];
            if (!m[i] && !(m[i] = product_matrix(im, lis[i]))) {
                free_distinct(m, i);
                return false;
            }
        }
        input = layer_struct_arguments((LayerArgs){.net = net, .type = LAYER_CONV2D, .kernel = 1, .filters = C3,
                                                   .inputs = {src}, .input_count = 1, .act_func = ACT_NONE,
                                                   .weight_initialization = WEIGHT_INITIALIZATION_NONE});
        if (input != SPINGALETT_NO_LAYER) {
            float *w = layer_weights(net, input), *b = layer_biases(net, input);
            uint32_t r = 0;
            for (uint32_t h = 0; h < H; h++)
                for (uint32_t d = 0; d < hd; d++, r++)
                    product_row(&im->lazies[lis[0]], m[0], q->offset + h * qst + d * qh.hstride, f,
                                w + (size_t)r * in, b + r);
            for (uint32_t g = 0; g < KV; g++)
                for (uint32_t d = 0; d < hd; d++, r++)
                    product_row(&im->lazies[lis[1]], m[1], k->offset + g * kst + d * kh.hstride, 1.0f,
                                w + (size_t)r * in, b + r);
            for (uint32_t g = 0; g < KV; g++)
                for (uint32_t d = 0; d < hd; d++, r++)
                    product_row(&im->lazies[lis[2]], m[2], v->offset + g * vst + d * vh.hstride, vf,
                                w + (size_t)r * in, b + r);
        }
        free_distinct(m, 3);
        if (!added(im, input, n)) return false;
    } else {
        if (fabsf(f - 1.0f) > 1e-5f || vf != 1.0f)
            return node_fail(im, n, "scales the scores by %g instead of 1/sqrt(%u), and its queries come from no "
                             "product the scale could be folded into", (double)(scale * q->scale * k->scale), hd);
        if (!materialize(im, qb) || !materialize(im, kb) || !materialize(im, vb)) return false;
        const bool one = source_of(qb) == source_of(kb) && source_of(qb) == source_of(vb);
        const bool qok = (H == 1 || qst == hd) && qh.hstride == 1, kok = (KV == 1 || kst == hd) && kh.hstride == 1;
        const bool vok = (KV == 1 || vst == hd) && vh.hstride == 1;
        if (one && qok && kok && vok && qb->seq && seq_channels(qb) == C3 && q->offset == 0 && k->offset == H * hd &&
            v->offset == (H + KV) * hd) {
            input = qb->layer;
        } else if (qok && kok && vok && qb->seq && kb->seq && vb->seq && q->offset == 0 && k->offset == 0 &&
                   v->offset == 0 && seq_channels(qb) == H * hd && seq_channels(kb) == KV * hd &&
                   seq_channels(vb) == KV * hd) {
            input = concat_layers(.net = net, .inputs = {qb->layer, kb->layer, vb->layer}, .input_count = 3,
                                  .act_func = ACT_NONE);
            if (!added(im, input, n)) return false;
        } else {
            return node_fail(im, n, "takes its queries, keys and values from parts of layers that cannot be read as "
                             "one layer's queries, keys and values");
        }
    }
    const uint32_t l = attention(.net = net, .inputs = {input}, .input_count = 1, .heads = H, .kv_heads = KV,
                                 .causal = causal, .rope_theta = q->theta);
    if (!added(im, l, n)) return false;
    im->bias_open = false;
    const int64_t rows[3] = {1, T, (int64_t)H * hd};
    Value *a = seq_value(im, (Str){"", 0}, false, l, rows, 3);
    if (!a) return false;
    const uint32_t ai = value_index(im, a);
    View o;
    if (plain) {
        plain_view(&im->values[ai], ai, &o);
        return view_output(im, n, out, &o);
    }
    /* arranged as the queries: their heads' factors over the result's heads */
    memset(&o, 0, sizeof o);
    o.base = ai;
    o.scale = 1.0f;
    o.nd = nlead + 2;
    for (int i = 0; i < nlead; i++) {
        const int d = i - (nlead - (q->nd - 2));
        if (d < 0 || dim_size(q, d) != (uint64_t)lead[i]) continue;
        for (int j = 0; j < q->nf[d]; j++) {
            const Factor fq = q->f[d][j];
            if (fq.size == 1) continue;
            o.f[i][o.nf[i]++] = (Factor){fq.size, qst ? fq.stride / qst * hd : 0, false};
        }
    }
    if (T > 1) o.f[nlead][o.nf[nlead]++] = (Factor){T, 1, true};
    if (hd > 1) o.f[nlead + 1][o.nf[nlead + 1]++] = (Factor){hd, 1, false};
    return view_output(im, n, out, &o);
}

/* MatMul of queries [.., T, head] by keys [.., head, T]: the scores, lazy until the softmax and the values
   complete the attention. */
static bool convert_scores(Importer *im, const Node *n, uint32_t qi, uint32_t ki) {
    View q, k;
    if (!view_of(im, n, qi, &q) || !view_of(im, n, ki, &k)) return false;
    int64_t qd[MAX_DIMS], kd[MAX_DIMS], dims[MAX_DIMS];
    int nl;
    view_shape(&q, qd);
    view_shape(&k, kd);
    if (q.nd < 2 || k.nd < 2 || qd[q.nd - 1] != kd[k.nd - 2] || !broadcast(qd, q.nd - 2, kd, k.nd - 2, dims, &nl) ||
        nl + 2 > MAX_DIMS)
        return node_fail(im, n, "multiplies two computed tensors that are no attention's queries and keys");
    dims[nl] = qd[q.nd - 2];
    dims[nl + 1] = kd[k.nd - 1];
    const uint32_t li = new_lazy(im, LZ_SCORES, n);
    if (li == NO_VALUE) return false;
    Lazy *z = &im->lazies[li];
    z->sc.q = q;
    z->sc.k = k;
    z->sc.scale = 1.0f;
    z->sc.causal = false;
    return lazy_value(im, n->outputs[0], true, li, dims, nl + 2, false) != NULL;
}

/* Scores or their softmax again, changed: a new lazy value of the same shape as x's. */
static Lazy *scores_again(Importer *im, const Node *n, const Value *x, uint8_t kind) {
    const uint32_t from = x->lazy - 1;
    int64_t dims[MAX_DIMS];
    memcpy(dims, x->dims, sizeof dims);
    const int nd = x->ndims;
    const uint32_t li = new_lazy(im, kind, n);
    if (li == NO_VALUE) return NULL;
    im->lazies[li].sc = im->lazies[from].sc;
    return lazy_value(im, n->outputs[0], true, li, dims, nd, false) ? &im->lazies[li] : NULL;
}

enum { MASK_ADD, MASK_KEEP, MASK_DROP };

/* What a constant mask over scores of T x T does (broadcast: one for every head): 0 nothing, 1 causal,
   -1 anything else, -2 an error. how: added (0 keeps, -1e4 and below mask: exp() makes 0 of them), or a
   Where's condition, true where the scores are kept or where they are masked. */
static int mask_kind(Importer *im, Value *m, uint32_t T, int how) {
    const float *d = constant_floats(im, m);
    if (!d) return -2;
    if (m->ndims < 1) return how == MASK_ADD && d[0] == 0.0f ? 0 : -1;
    const int64_t cols = m->dims[m->ndims - 1], rows = m->ndims > 1 ? m->dims[m->ndims - 2] : 1;
    for (int i = 0; i + 2 < m->ndims; i++)
        if (m->dims[i] != 1) return -1;
    if ((cols != T && cols != 1) || (rows != T && rows != 1)) return -1;
    bool causal = true, none = true;
    for (uint32_t i = 0; i < T; i++)
        for (uint32_t j = 0; j < T; j++) {
            const float x = d[(rows == 1 ? 0 : (uint64_t)i * (uint64_t)cols) + (cols == 1 ? 0 : j)];
            const bool keep = how == MASK_ADD ? x == 0.0f : how == MASK_KEEP ? x != 0.0f : x == 0.0f;
            if (how == MASK_ADD && !keep && !(x <= -1e4f)) return -1;
            if (keep != (j <= i)) causal = false;
            if (!keep) none = false;
        }
    return none ? 0 : causal ? 1 : -1;
}

static bool scores_masked(Importer *im, const Node *n, Value *s, Value *m, int how) {
    const int kind = mask_kind(im, m, (uint32_t)s->dims[s->ndims - 1], how);
    if (kind == -2) return false;
    if (kind < 0) return node_fail(im, n, "masks the attention's scores other than causally");
    Lazy *z = scores_again(im, n, s, LZ_SCORES);
    if (!z) return false;
    z->sc.causal |= kind == 1;
    return true;
}

/* A constant's single value, NAN when it holds more (or cannot be read). */
static float scalar_of(Importer *im, Value *c) {
    if (!c || !c->constant || c->count != 1) return NAN;
    const float *d = constant_floats(im, c);
    return d ? d[0] : NAN;
}

/* opset 23's Attention: queries, keys and values [N, heads, T, head] (or [N, T, heads x head] with
   q_num_heads and kv_num_heads), the keys not transposed, without mask or cache. */
static bool convert_attention_op(Importer *im, const Node *n) {
    if (n->ninputs < 3) return node_fail(im, n, "lacks queries, keys or values");
    for (int k = 3; k < n->ninputs; k++)
        if (n->inputs[k].n) return node_fail(im, n, "reads a mask or a cache, which the attention layer does not have");
    for (int k = 1; k < n->noutputs; k++)
        if (n->outputs[k].n && readers(im, n->outputs[k]))
            return node_fail(im, n, "outputs its keys, values or scores to other nodes");
    if (attr_float(n, "softcap", 0.0f) != 0.0f || attr_int(n, "left_window_size", -1) >= 0 ||
        attr_int(n, "right_window_size", -1) >= 0)
        return node_fail(im, n, "caps its scores or attends over a window, which the attention layer does not");
    View q, k, v;
    uint32_t idx[3];
    for (int i = 0; i < 3; i++) {
        const Value *x = find(im, n->inputs[i]);
        if (!x) return node_fail(im, n, "reads an unknown value");
        idx[i] = value_index(im, x);
    }
    if (!view_of(im, n, idx[0], &q) || !view_of(im, n, idx[1], &k) || !view_of(im, n, idx[2], &v)) return false;
    const bool flat = q.nd == 3;
    if (flat) {
        /* [N, T, heads x head] into [N, heads, T, head] */
        const int64_t H = attr_int(n, "q_num_heads", 0), KV = attr_int(n, "kv_num_heads", 0);
        int64_t qd[MAX_DIMS], kd[MAX_DIMS];
        view_shape(&q, qd);
        view_shape(&k, kd);
        static const int64_t perm[4] = {0, 2, 1, 3};
        if (H <= 0 || KV <= 0 || k.nd != 3 || v.nd != 3 || qd[2] % H || kd[2] % KV)
            return node_fail(im, n, "needs its numbers of heads (q_num_heads, kv_num_heads), dividing its channels");
        const int64_t tq[4] = {qd[0], qd[1], H, qd[2] / H}, tk[4] = {kd[0], kd[1], KV, kd[2] / KV};
        if (!view_reshape(&q, tq, 4) || !view_reshape(&k, tk, 4) || !view_reshape(&v, tk, 4))
            return node_fail(im, n, "reads queries, keys or values it cannot split into heads");
        view_transpose(&q, perm);
        view_transpose(&k, perm);
        view_transpose(&v, perm);
    }
    if (q.nd != 4 || k.nd != 4 || v.nd != 4)
        return node_fail(im, n, "reads queries, keys or values of other ranks than 3 or 4");
    /* the keys' and values' heads repeated in place for grouped queries, the keys transposed */
    const uint64_t H = dim_size(&q, 1), KV = dim_size(&k, 1);
    if (KV != dim_size(&v, 1) || KV == 0 || H % KV)
        return node_fail(im, n, "has key and value heads that do not divide its query heads");
    if (H != KV) {
        if (k.nf[1] == MAX_FACTORS || v.nf[1] == MAX_FACTORS) return node_fail(im, n, "has heads it cannot repeat");
        k.f[1][k.nf[1]++] = (Factor){(uint32_t)(H / KV), 0, false};
        v.f[1][v.nf[1]++] = (Factor){(uint32_t)(H / KV), 0, false};
    }
    static const int64_t swap[4] = {0, 1, 3, 2};
    view_transpose(&k, swap);
    int64_t lead[2] = {(int64_t)dim_size(&q, 0), (int64_t)H};
    const uint64_t hd = dim_size(&q, 3);
    const Attr *sa = attr(n, "scale");
    const float scale = sa ? sa->f : 1.0f / sqrtf((float)hd);
    return make_attention(im, n, &q, &k, &v, scale, attr_int(n, "is_causal", 0) != 0, lead, 2, 0, flat);
}

/* ------------------------------------------------------------------------- nodes of sequences */

/* A product of per-token values (value xi) by constant weights W ([in][out], or [out][in] when not
   transposed) times alpha, plus beta times bias (value ci, or NO_VALUE): a lazy product, a sequence
   without a name (its index; NO_VALUE on an error), *arr the arrangement of the input. */
static uint32_t token_linear(Importer *im, const Node *n, uint32_t xi, uint32_t wi, bool transposed, float alpha,
                             uint32_t ci, float beta, View *arr) {
    const uint32_t si = view_rows(im, n, xi, arr);
    if (si == NO_VALUE || !materialize(im, &im->values[si])) return NO_VALUE;
    const Value *s = &im->values[si], *W = &im->values[wi];
    if (W->ndims < 2) return node_fail(im, n, "multiplies by weights of fewer than two axes"), NO_VALUE;
    for (int i = 0; i + 2 < W->ndims; i++)
        if (W->dims[i] != 1) return node_fail(im, n, "multiplies by weights of more than two axes"), NO_VALUE;
    const int64_t r0 = W->dims[W->ndims - 2], r1 = W->dims[W->ndims - 1];
    const int64_t in = transposed ? r0 : r1, out = transposed ? r1 : r0;
    if (in != (int64_t)seq_channels(s) || out <= 0 || out > UINT32_MAX)
        return node_fail(im, n, "has weights that do not fit its input"), NO_VALUE;
    float *bias = NULL, *scale = NULL;
    if (ci != NO_VALUE) {
        Value *c = &im->values[ci];
        const float *d = constant_floats(im, c);
        if (!d) return NO_VALUE;
        if (c->count != 1 && c->count != (uint64_t)out) return node_fail(im, n, "has a bias of another size"), NO_VALUE;
        if (!(bias = (float *)malloc((size_t)out * sizeof(float)))) return out_of_memory(im), NO_VALUE;
        for (int64_t o = 0; o < out; o++) bias[o] = beta * d[c->count == 1 ? 0 : o];
    }
    if (alpha != 1.0f) {
        if (!(scale = (float *)malloc((size_t)out * sizeof(float)))) { free(bias); return out_of_memory(im), NO_VALUE; }
        for (int64_t o = 0; o < out; o++) scale[o] = alpha;
    }
    int64_t dims[3];
    const int nd = s->ndims;
    memcpy(dims, s->dims, (size_t)nd * sizeof *dims);
    dims[nd - 1] = out;
    const uint32_t src = s->layer;
    const uint32_t li = new_lazy(im, LZ_LINEAR, n);
    if (li == NO_VALUE) { free(bias); free(scale); return NO_VALUE; }
    Lazy *z = &im->lazies[li];
    z->lin.src = src;
    z->lin.w = wi;
    z->lin.in = (uint32_t)in;
    z->lin.rows = z->lin.out = (uint32_t)out;
    z->lin.row0 = 0;
    z->lin.transposed = transposed;
    z->lin.scale = scale;
    z->lin.bias = bias;
    z->lin.act = ACT_NONE;
    const Value *y = lazy_value(im, (Str){"", 0}, false, li, dims, nd, true);
    return y ? value_index(im, y) : NO_VALUE;
}

/* The product of token_linear() as output 0 of the node, arranged as its input. */
static bool token_product(Importer *im, const Node *n, uint32_t xi, uint32_t wi, bool transposed, float alpha,
                          uint32_t ci, float beta) {
    View arr;
    const uint32_t r = token_linear(im, n, xi, wi, transposed, alpha, ci, beta, &arr);
    return r != NO_VALUE && rows_output(im, n, 0, r, &arr);
}

static bool tok_matmul(Importer *im, const Node *n) {
    const bool gemm = str_is(n->op, "Gemm");
    Value *a = n->ninputs > 1 ? find(im, n->inputs[0]) : NULL, *b = n->ninputs > 1 ? find(im, n->inputs[1]) : NULL;
    if (!a || !b) return node_fail(im, n, "reads an unknown value");
    if (b->constant && !a->constant) {
        if (gemm && attr_int(n, "transA", 0) != 0) return node_fail(im, n, "transposes its input (transA)");
        Value *c = gemm && n->ninputs > 2 && n->inputs[2].n ? find(im, n->inputs[2]) : NULL;
        if (gemm && n->ninputs > 2 && n->inputs[2].n && (!c || !c->constant))
            return node_fail(im, n, "adds a bias that is not constant");
        return token_product(im, n, value_index(im, a), value_index(im, b), gemm ? attr_int(n, "transB", 0) == 0 : true,
                             gemm ? attr_float(n, "alpha", 1.0f) : 1.0f, c ? value_index(im, c) : NO_VALUE,
                             gemm ? attr_float(n, "beta", 1.0f) : 1.0f);
    }
    if (gemm || a->constant || b->constant)
        return node_fail(im, n, "multiplies a constant by computed values from the left, which no layer does");
    if (kind_of(im, a) == LZ_SCORES)
        return node_fail(im, n, "multiplies the attention's scores by values without a softmax between them");
    if (kind_of(im, a) == LZ_PROBS) {
        const Lazy z = im->lazies[a->lazy - 1];
        View v;
        if (!view_of(im, n, value_index(im, b), &v)) return false;
        int64_t lead[MAX_DIMS];
        const int nlead = a->ndims - 2;
        memcpy(lead, a->dims, sizeof lead);
        return make_attention(im, n, &z.sc.q, &z.sc.k, &v, z.sc.scale, z.sc.causal, lead, nlead, 0, false);
    }
    return convert_scores(im, n, value_index(im, a), value_index(im, b));
}

/* Reshape, Flatten, Transpose, Unsqueeze, Squeeze, Expand, Tile, Slice, Gather (of a constant index) and
   Split of a sequence or a view of one: a view. */
static bool tok_view(Importer *im, const Node *n) {
    Value *x = n->ninputs > 0 ? find(im, n->inputs[0]) : NULL;
    if (!x) return node_fail(im, n, "reads an unknown value");
    if (x->constant) return node_fail(im, n, "reads a constant by computed indices");
    View v;
    if (!view_of(im, n, value_index(im, x), &v)) return false;
    int64_t dims[MAX_DIMS], t[MAX_DIMS];
    const int nd = view_shape(&v, dims);
    const uint64_t count = dims_count(dims, nd);
    const Str op = n->op;
    bool ok = true;
    if (str_is(op, "Reshape")) {
        Const s;
        if (!const_arg(im, n, 1, &s)) return false;
        ok = reshape_target(s.d, s.n, dims, nd, count, attr_int(n, "allowzero", 0) != 0, t) &&
             view_reshape(&v, t, (int)s.n);
    } else if (str_is(op, "Flatten")) {
        int axis;
        ok = norm_axis(attr_int(n, "axis", 1), nd + 1, &axis);
        if (ok) {
            t[0] = (int64_t)dims_count(dims, axis);
            t[1] = (int64_t)dims_count(dims + axis, nd - axis);
            ok = view_reshape(&v, t, 2);
        }
    } else if (str_is(op, "Transpose")) {
        ok = transpose_perm(n, nd, t) && view_transpose(&v, t);
    } else if (str_is(op, "Unsqueeze") || str_is(op, "Squeeze")) {
        int na;
        if (!axes_arg(im, n, t, &na)) return false;
        ok = view_squeeze(&v, t, na, str_is(op, "Unsqueeze"));
    } else if (str_is(op, "Expand")) {
        Const s;
        if (!const_arg(im, n, 1, &s)) return false;
        ok = s.n <= MAX_DIMS;
        for (uint64_t k = 0; ok && k < s.n; k++) t[k] = to_i64(s.d[k]);
        ok = ok && view_expand(&v, t, (int)s.n);
    } else if (str_is(op, "Tile")) {
        /* each axis repeated whole: a repeat (stride 0) before its factors */
        Const r;
        if (!const_arg(im, n, 1, &r)) return false;
        ok = r.n == (uint64_t)nd;
        for (int a = 0; ok && a < nd; a++) {
            if (r.d[a] == 1.0f) continue;
            ok = r.d[a] >= 1.0f && r.d[a] <= 65536.0f && r.d[a] == floorf(r.d[a]) && v.nf[a] < MAX_FACTORS;
            if (!ok) break;
            memmove(&v.f[a][1], &v.f[a][0], v.nf[a] * sizeof(Factor));
            v.f[a][0] = (Factor){(uint32_t)r.d[a], 0, false};
            v.nf[a]++;
        }
    } else if (str_is(op, "Slice")) {
        int64_t first[MAX_DIMS], step[MAX_DIMS], cnt[MAX_DIMS];
        if (!slice_args(im, n, dims, nd, first, step, cnt)) return false;
        for (int a = 0; ok && a < nd; a++) {
            if (first[a] == 0 && cnt[a] == dims[a] && (step[a] == 1 || cnt[a] <= 1)) continue;
            ok = (step[a] == 1 || cnt[a] <= 1) && view_slice(&v, a, first[a], cnt[a]);
        }
    } else if (str_is(op, "Gather")) {
        Const ix;
        int axis;
        if (!const_arg(im, n, 1, &ix)) return false;
        ok = ix.n == 1 && ix.nd <= 1 && norm_axis(attr_int(n, "axis", 0), nd, &axis);
        if (ok) {
            int64_t k = to_i64(ix.d[0]);
            if (k < 0) k += dims[axis];
            ok = k >= 0 && k < dims[axis] && view_slice(&v, axis, k, 1);
            const int64_t ax = axis;
            if (ok && ix.nd == 0) ok = view_squeeze(&v, &ax, 1, false);
        }
    } else {                                                    /* Split */
        int axis;
        int64_t sizes[MAX_NODE_OUTPUTS];
        if (!norm_axis(attr_int(n, "axis", 0), nd, &axis)) return node_fail(im, n, "splits along an axis it lacks");
        if (!split_sizes(im, n, dims[axis], sizes)) return false;
        int64_t at = 0;
        for (int k = 0; k < n->noutputs; at += sizes[k], k++) {
            View p = v;
            if (!view_slice(&p, axis, at, sizes[k]))
                return node_fail(im, n, "splits the tokens, or channels in heads, which no layer does");
            if (n->outputs[k].n && !view_output(im, n, k, &p)) return false;
        }
        return true;
    }
    if (!ok)
        return node_fail(im, n, "rearranges tokens in a way that splits or mixes them, or by axes that do not fit");
    return view_output(im, n, 0, &v);
}

/* x op c (c op x when const_first) for a sequence, layer or expression x: a product's bias or scale, an
   embedding's positions or scale, or an expression; a new value (named: output 0's), its index
   (NO_VALUE on an error). */
static uint32_t rows_const(Importer *im, const Node *n, uint8_t op, uint32_t xi, uint32_t ci, bool const_first,
                           bool named) {
    Value *x = &im->values[xi], *c = &im->values[ci];
    const int kind = kind_of(im, x);
    const float *d = constant_floats(im, c);
    if (!d) return NO_VALUE;
    const Str name = named ? n->outputs[0] : (Str){"", 0};
    /* a scale folds into a product read by nothing else (a value read twice may be part of an expression,
       x times erf(x / sqrt(2)) say); a bias always does */
    const bool alone = x->name.n == 0 || readers(im, x->name) <= 1;
    const bool linear_op = op == E_ADD || (!const_first && op == E_SUB) ||
                           (alone && (op == E_MUL || (!const_first && op == E_DIV)));
    if (kind == LZ_LINEAR && im->lazies[x->lazy - 1].lin.act == ACT_NONE && linear_op) {
        const uint32_t from = x->lazy - 1, out = im->lazies[from].lin.out;
        int rank;
        const uint32_t C = per_channel(x, &rank);
        if (c->count == 1 || (c->count == out && C == out && c->ndims >= 1 && c->ndims <= rank &&
                              c->dims[c->ndims - 1] == (int64_t)out)) {
            /* the product's bias or scale */
            float *scale = (float *)malloc(out * sizeof(float)), *bias = (float *)malloc(out * sizeof(float));
            if (!scale || !bias) { free(scale); free(bias); return out_of_memory(im), NO_VALUE; }
            const Lazy *p = &im->lazies[from];
            for (uint32_t o = 0; o < out; o++) {
                const float k = d[c->count == 1 ? 0 : o];
                scale[o] = p->lin.scale ? p->lin.scale[o] : 1.0f;
                bias[o] = p->lin.bias ? p->lin.bias[o] : 0.0f;
                if (op == E_ADD) bias[o] += k;
                else if (op == E_SUB) bias[o] -= k;
                else if (op == E_MUL) scale[o] *= k, bias[o] *= k;
                else scale[o] /= k, bias[o] /= k;
            }
            const uint32_t li = new_lazy(im, LZ_LINEAR, n);
            if (li == NO_VALUE) { free(scale); free(bias); return NO_VALUE; }
            im->lazies[li].lin = im->lazies[from].lin;
            im->lazies[li].lin.scale = scale;
            im->lazies[li].lin.bias = bias;
            int64_t dims[MAX_DIMS];
            memcpy(dims, im->values[xi].dims, sizeof dims);
            const Value *y = lazy_value(im, name, named, li, dims, im->values[xi].ndims, true);
            return y ? value_index(im, y) : NO_VALUE;
        }
    }
    if (kind == LZ_EMBED && im->lazies[x->lazy - 1].emb.act == ACT_NONE) {
        const uint32_t from = x->lazy - 1, T = seq_tokens(x), dd = seq_channels(x);
        const Lazy *p = &im->lazies[from];
        bool positions = op == E_ADD && c->count == (uint64_t)T * dd && c->ndims >= 2 &&
                         c->dims[c->ndims - 1] == (int64_t)dd && c->dims[c->ndims - 2] == (int64_t)T;
        for (int i = 0; positions && i + 2 < c->ndims; i++) positions = c->dims[i] == 1;
        const bool scaled = alone && c->count == 1 && (op == E_MUL || (op == E_DIV && !const_first));
        uint32_t pos = ci;
        if (positions && p->emb.pos != NO_VALUE) {
            /* a second table a position (BERT's token types, when they are constants): the sum of both */
            float *sum = (float *)malloc((size_t)T * dd * sizeof(float));
            const float *was = constant_floats(im, &im->values[p->emb.pos]);
            if (!sum || !was) { free(sum); return was ? (out_of_memory(im), NO_VALUE) : NO_VALUE; }
            for (size_t i = 0; i < (size_t)T * dd; i++) sum[i] = p->emb.pscale * was[i] + d[i];
            Value *v = new_value(im, (Str){"", 0}, false);
            if (!v) { free(sum); return out_of_memory(im), NO_VALUE; }
            v->constant = v->owned = true;
            v->data = sum;
            v->ndims = 2;
            v->dims[0] = T;
            v->dims[1] = dd;
            v->count = (uint64_t)T * dd;
            v->ctype = ONNX_FLOAT;
            pos = value_index(im, v);
        }
        if (positions || scaled) {
            /* learned positions added (absolute position embeddings), or the table scaled */
            const uint32_t li = new_lazy(im, LZ_EMBED, n);
            if (li == NO_VALUE) return NO_VALUE;
            im->lazies[li].emb = im->lazies[from].emb;
            if (positions) {
                im->lazies[li].emb.pos = pos;
                im->lazies[li].emb.pscale = 1.0f;
            } else {
                const float k = op == E_MUL ? d[0] : 1.0f / d[0];
                im->lazies[li].emb.tscale *= k;
                im->lazies[li].emb.pscale *= k;
            }
            int64_t dims[MAX_DIMS];
            memcpy(dims, im->values[xi].dims, sizeof dims);
            const Value *y = lazy_value(im, name, named, li, dims, im->values[xi].ndims, true);
            return y ? value_index(im, y) : NO_VALUE;
        }
    }
    uint32_t base;
    const uint32_t xe = expr_operand(im, n, x, op != E_POW, &base);
    if (xe == NO_VALUE) return NO_VALUE;
    const uint32_t ce = expr_const(im, n, c, x);
    if (ce == NO_VALUE) return NO_VALUE;
    return expr_value(im, n, base, const_first ? enode(im, n, op, ce, xe, 0.0f, NO_VALUE)
                                               : enode(im, n, op, xe, ce, 0.0f, NO_VALUE), named);
}

/* Output 0 of a node: a lazy value of the kind with the view v (and a table), shaped as x. */
static bool view_lazy(Importer *im, const Node *n, uint8_t kind, const View *v, uint32_t table, const Value *x) {
    int64_t dims[MAX_DIMS];
    memcpy(dims, x->dims, sizeof dims);
    const int nd = x->ndims;
    const uint32_t li = new_lazy(im, kind, n);
    if (li == NO_VALUE) return false;
    im->lazies[li].view.v = *v;
    im->lazies[li].view.table = table;
    im->lazies[li].view.rows = NO_VALUE;
    return lazy_value(im, n->outputs[0], true, li, dims, nd, false) != NULL;
}

/* x op c (c op x when const_first) where x is no constant. */
static bool tok_binary_const(Importer *im, const Node *n, uint8_t op, uint32_t xi, uint32_t ci, bool const_first) {
    Value *x = &im->values[xi], *c = &im->values[ci];
    const int kind = kind_of(im, x);
    const float k = scalar_of(im, c);
    const bool scaling = (op == E_MUL || (op == E_DIV && !const_first)) && k == k && k != 0.0f;
    const float factor = op == E_MUL ? k : 1.0f / k;
    if (kind == LZ_SCORES) {
        if (scaling) {
            Lazy *z = scores_again(im, n, x, LZ_SCORES);
            if (z) z->sc.scale *= factor;
            return z != NULL;
        }
        if (op == E_ADD) return scores_masked(im, n, x, c, MASK_ADD);
        return node_fail(im, n, "combines the attention's scores with a constant other than a scale or a mask");
    }
    if (kind == LZ_VIEW) {
        View v = im->lazies[x->lazy - 1].view.v;
        uint32_t off, cnt;
        if (view_per_token(im, &v, &off, &cnt) && (c->count == 1 || c->count == cnt)) {
            View arr;
            const uint32_t si = view_rows(im, n, xi, &arr);
            const uint32_t r = si == NO_VALUE ? NO_VALUE : rows_const(im, n, op, si, ci, const_first, false);
            return r != NO_VALUE && rows_output(im, n, 0, r, &arr);
        }
        if (scaling) {
            v.scale *= factor;
            return view_output(im, n, 0, &v);
        }
        if (op == E_MUL && v.theta == 0.0f) return view_lazy(im, n, LZ_COS, &v, ci, x);
        return node_fail(im, n, "combines tokens' heads with a constant other than a scale or rotary embeddings' table");
    }
    if (kind == LZ_ROT) {
        if (op == E_MUL) return view_lazy(im, n, LZ_SIN, &im->lazies[x->lazy - 1].view.v, ci, x);
        return node_fail(im, n, "combines rotated heads with a constant other than rotary embeddings' table");
    }
    if (kind == LZ_COS || kind == LZ_SIN || kind == LZ_PROBS)
        return node_fail(im, n, "combines part of an attention with a constant in a way the importer does not know");
    return rows_const(im, n, op, xi, ci, const_first, true) != NO_VALUE;
}

/* The sum of a view times the cosines and the view rotated times the sines: the view with rotary
   embeddings, their base told by the tables. */
static bool rope_add(Importer *im, const Node *n, const Value *cv, const Value *sv) {
    const Lazy zc = im->lazies[cv->lazy - 1], zs = im->lazies[sv->lazy - 1];
    if (!same_view(im, &zc.view.v, &zs.view.v))
        return node_fail(im, n, "adds rotary embeddings' terms of different queries or keys");
    View x = zc.view.v;
    const float *cs, *sn;
    uint64_t cts, chs, sts, shs;
    uint32_t T, head, T2, head2;
    float theta;
    if (!rope_table(im, &x, &im->values[zc.view.table], &cs, &cts, &chs, &T, &head) ||
        !rope_table(im, &x, &im->values[zs.view.table], &sn, &sts, &shs, &T2, &head2) || T != T2 || head != head2)
        return im->error[0] ? false : node_fail(im, n, "adds rotary embeddings whose tables do not fit its heads");
    /* the tables read at the same places: copied when their layouts differ */
    float *sn2 = NULL;
    if (sts != cts || shs != chs) {
        if (!(sn2 = (float *)malloc((size_t)T * head * sizeof(float)))) return out_of_memory(im);
        float *cs2 = (float *)malloc((size_t)T * head * sizeof(float));
        if (!cs2) { free(sn2); return out_of_memory(im); }
        for (uint32_t t = 0; t < T; t++)
            for (uint32_t i = 0; i < head; i++) {
                cs2[(size_t)t * head + i] = cs[t * cts + i * chs];
                sn2[(size_t)t * head + i] = sn[t * sts + i * shs];
            }
        const bool ok = rope_theta(cs2, sn2, head, 1, T, head, &theta);
        free(cs2);
        free(sn2);
        if (!ok) return node_fail(im, n, "adds rotary embeddings of other angles than t theta^(-2i / head)");
    } else if (!rope_theta(cs, sn, cts, chs, T, head, &theta)) {
        return node_fail(im, n, "adds rotary embeddings of other angles than t theta^(-2i / head)");
    }
    x.theta = theta;
    x.rope_head = head;
    return view_output(im, n, 0, &x);
}

/* a op b of two values that are layers' outputs, sequences or expressions: an expression when both are of
   one value, else a layer adding or multiplying them. */
static bool combine(Importer *im, const Node *n, uint8_t op, Value *a, Value *b) {
    const bool ea = kind_of(im, a) == LZ_EXPR || expr_base(im, a), eb = kind_of(im, b) == LZ_EXPR || expr_base(im, b);
    if (ea && eb && operand_source(im, a, false) == operand_source(im, b, false)) return expr_binary(im, n, op, a, b);
    if (op != E_ADD && op != E_MUL)
        return node_fail(im, n, "subtracts, divides or raises two computed tensors, which no layer does");
    if (!materialize(im, a) || !materialize(im, b)) return false;
    const bool same = a->seq == b->seq &&
                      (a->seq ? seq_tokens(a) == seq_tokens(b) && seq_channels(a) == seq_channels(b)
                              : a->flat == b->flat && a->nhwc == b->nhwc && a->c == b->c && a->h == b->h &&
                                a->w == b->w);
    if (!same) return node_fail(im, n, "combines tensors of different shapes (broadcasting is not supported)");
    const uint32_t l = op == E_ADD ? add_layers(.net = im->net, .inputs = {a->layer, b->layer}, .input_count = 2)
                                   : multiply_layers(.net = im->net, .inputs = {a->layer, b->layer}, .input_count = 2);
    if (!added(im, l, n)) return false;
    im->bias_open = false;
    Value copy = *a;
    copy.vector = a->vector && b->vector;
    Value *y = add_value(im, n->outputs[0]);
    if (!y) return out_of_memory(im);
    const Str name = y->name;
    *y = copy;
    y->name = name;
    y->lazy = 0;
    y->alias = false;
    y->owned = false;
    set_layer(im, y, l);
    return true;
}

static bool tok_binary(Importer *im, const Node *n, uint8_t op) {
    if (n->ninputs != 2) return node_fail(im, n, "needs two inputs");
    Value *a = find(im, n->inputs[0]), *b = find(im, n->inputs[1]);
    if (!a || !b) return node_fail(im, n, "reads an unknown value");
    const int ka = kind_of(im, a), kb = kind_of(im, b);
    if (ka == K_CONST || kb == K_CONST) {
        const bool cf = ka == K_CONST;
        return tok_binary_const(im, n, op, value_index(im, cf ? b : a), value_index(im, cf ? a : b), cf);
    }
    if (op == E_ADD && ((ka == LZ_COS && kb == LZ_SIN) || (ka == LZ_SIN && kb == LZ_COS)))
        return rope_add(im, n, ka == LZ_COS ? a : b, ka == LZ_COS ? b : a);
    if (ka == LZ_VIEW || kb == LZ_VIEW)
        return node_fail(im, n, "combines tokens' heads or rearranged tokens, which no layer does");
    if (ka >= LZ_ROT || kb >= LZ_ROT)
        return node_fail(im, n, "combines parts of an attention in a way the importer does not know");
    return combine(im, n, op, a, b);
}

static bool tok_unary(Importer *im, const Node *n, uint8_t op) {
    Value *x = n->ninputs > 0 ? find(im, n->inputs[0]) : NULL;
    if (!x) return node_fail(im, n, "reads an unknown value");
    const int kind = kind_of(im, x);
    if (kind == LZ_VIEW) {
        View v = im->lazies[x->lazy - 1].view.v;
        if (op == E_NEG) {
            v.scale = -v.scale;
            return view_output(im, n, 0, &v);
        }
        uint32_t off, cnt;
        if (!view_per_token(im, &v, &off, &cnt))
            return node_fail(im, n, "applies a function to tokens' heads, which no layer does");
        View arr;
        const uint32_t si = view_rows(im, n, value_index(im, x), &arr);
        if (si == NO_VALUE) return false;
        uint32_t base;
        const uint32_t e = expr_operand(im, n, &im->values[si], false, &base);
        const uint32_t r = e == NO_VALUE ? NO_VALUE
                                         : expr_value(im, n, base, enode(im, n, op, e, 0, 0.0f, NO_VALUE), false);
        return r != NO_VALUE && rows_output(im, n, 0, r, &arr);
    }
    if (kind >= LZ_ROT) return node_fail(im, n, "applies a function to part of an attention, which no layer does");
    return expr_unary(im, n, op, x);
}

static bool tok_where(Importer *im, const Node *n) {
    Value *c = n->ninputs == 3 ? find(im, n->inputs[0]) : NULL;
    Value *x = n->ninputs == 3 ? find(im, n->inputs[1]) : NULL, *y = n->ninputs == 3 ? find(im, n->inputs[2]) : NULL;
    if (!c || !x || !y) return node_fail(im, n, "reads an unknown value");
    if (c->constant && kind_of(im, y) == LZ_SCORES && scalar_of(im, x) <= -1e4f)
        return scores_masked(im, n, y, c, MASK_DROP);
    if (c->constant && kind_of(im, x) == LZ_SCORES && scalar_of(im, y) <= -1e4f)
        return scores_masked(im, n, x, c, MASK_KEEP);
    return node_fail(im, n, "selects between computed values other than attention scores and their mask");
}

static bool tok_softmax(Importer *im, const Node *n) {
    Value *x = n->ninputs > 0 ? find(im, n->inputs[0]) : NULL;
    if (!x) return node_fail(im, n, "reads an unknown value");
    if (kind_of(im, x) != LZ_SCORES)
        return node_fail(im, n, "takes a softmax over each token's channels, which no layer computes (only an "
                         "attention's over its keys, or a whole vector's)");
    int axis;
    if (!norm_axis(attr_int(n, "axis", im->opset && im->opset < 13 ? 1 : -1), x->ndims, &axis) || axis != x->ndims - 1)
        return node_fail(im, n, "takes the scores' softmax over other axes than the keys");
    return scores_again(im, n, x, LZ_PROBS) != NULL;
}

/* ReduceMean over the channels (an expression: normalizations' means) or the tokens (global average
   pooling) of a sequence. */
static bool tok_reduce_mean(Importer *im, const Node *n) {
    Value *x = n->ninputs > 0 ? find(im, n->inputs[0]) : NULL;
    if (!x) return node_fail(im, n, "reads an unknown value");
    int64_t axes[MAX_DIMS], dims[MAX_DIMS];
    int na, a0;
    if (!axes_arg(im, n, axes, &na)) return false;
    const int nd = value_shape(x, dims);
    const bool keep = attr_int(n, "keepdims", 1) != 0;
    if (na != 1 || !norm_axis(axes[0], nd, &a0))
        return node_fail(im, n, "averages other axes than the channels or the tokens");
    if (a0 == nd - 1 && keep && (kind_of(im, x) == LZ_EXPR || expr_base(im, x))) {
        uint32_t base;
        const uint32_t e = expr_operand(im, n, x, false, &base);
        return e != NO_VALUE && expr_output(im, n, base, enode(im, n, E_MEAN, e, 0, 0.0f, NO_VALUE));
    }
    if (x->seq && nd == 3 && a0 == 1) {
        View arr;
        const uint32_t si = view_rows(im, n, value_index(im, x), &arr);
        if (si == NO_VALUE || !materialize(im, &im->values[si])) return false;
        const uint32_t C = seq_channels(&im->values[si]);
        const uint32_t l = global_avg_pool2d(.net = im->net, .inputs = {im->values[si].layer}, .input_count = 1);
        if (!added(im, l, n)) return false;
        im->bias_open = false;
        if (keep) {
            const int64_t one[3] = {1, 1, C};
            return seq_value(im, n->outputs[0], true, l, one, 3) != NULL;
        }
        Value *y = layer_output(im, n, l);
        if (y) y->vector = true;
        return y != NULL;
    }
    return node_fail(im, n, "averages other axes than the channels or the tokens");
}

/* LayerNormalization and RMSNormalization (SimplifiedLayerNormalization) over each token's channels. */
static bool tok_norm(Importer *im, const Node *n, bool rms) {
    Value *x = n->ninputs > 1 ? find(im, n->inputs[0]) : NULL;
    if (!x) return node_fail(im, n, "reads an unknown value");
    View arr;
    const uint32_t si = view_rows(im, n, value_index(im, x), &arr);
    if (si == NO_VALUE || !materialize(im, &im->values[si])) return false;
    int axis;
    if (!norm_axis(attr_int(n, "axis", -1), arr.nd, &axis) || axis != arr.nd - 1)
        return node_fail(im, n, "normalizes other axes than each token's channels");
    const Value *s = &im->values[si];
    const uint32_t C = seq_channels(s), src = s->layer;
    int64_t dims[3];
    const int nd = s->ndims;
    memcpy(dims, s->dims, (size_t)nd * sizeof *dims);
    const Value *gamma = constant_input(im, n, 1, C);
    if (!gamma) return false;
    const Value *beta = !rms && n->ninputs > 2 && n->inputs[2].n ? constant_input(im, n, 2, C) : NULL;
    if (!rms && n->ninputs > 2 && n->inputs[2].n && !beta) return false;
    const float eps = attr_float(n, "epsilon", 1e-5f), e = eps > 0.0f && eps < 1.0f ? eps : 1e-5f;
    const uint32_t l = rms ? rms_norm(.net = im->net, .inputs = {src}, .input_count = 1, .epsilon = e)
                           : layer_norm(.net = im->net, .inputs = {src}, .input_count = 1, .epsilon = e);
    if (!added(im, l, n)) return false;
    im->bias_open = false;
    memcpy(layer_weights(im->net, l), gamma->data, C * sizeof(float));
    if (!rms) {
        if (beta) memcpy(layer_biases(im->net, l), beta->data, C * sizeof(float));
        else memset(layer_biases(im->net, l), 0, C * sizeof(float));
    }
    const Value *y = seq_value(im, (Str){"", 0}, false, l, dims, nd);
    return y && rows_output(im, n, 0, value_index(im, y), &arr);
}

/* Concat along the last axis: rotate_half's halves of a view, or sequences side by side. */
static bool tok_concat(Importer *im, const Node *n) {
    if (n->ninputs < 1 || n->ninputs > SPINGALETT_MAX_INPUTS) return node_fail(im, n, "joins 1 to 16 inputs");
    Value *a = find(im, n->inputs[0]), *b = n->ninputs == 2 ? find(im, n->inputs[1]) : NULL;
    if (!a) return node_fail(im, n, "reads an unknown value");
    if (b && kind_of(im, a) == LZ_VIEW && kind_of(im, b) == LZ_VIEW) {
        const View va = im->lazies[a->lazy - 1].view.v, vb = im->lazies[b->lazy - 1].view.v;
        View x;
        int axis;
        if (norm_axis(attr_int(n, "axis", 0), va.nd, &axis) && axis == va.nd - 1 && rot_half(im, &va, &vb, &x))
            return view_lazy(im, n, LZ_ROT, &x, NO_VALUE, a);
    }
    /* sequences side by side along the channels */
    LayerArgs args = {.net = im->net, .type = LAYER_CONCAT, .input_count = (uint32_t)n->ninputs, .act_func = ACT_NONE};
    View first;
    int64_t dims[3];
    int nd = 0;
    for (int k = 0; k < n->ninputs; k++) {
        const Value *x = find(im, n->inputs[k]);
        if (!x) return node_fail(im, n, "reads an unknown value");
        View arr;
        const uint32_t si = view_rows(im, n, value_index(im, x), &arr);
        if (si == NO_VALUE || !materialize(im, &im->values[si])) return false;
        int axis;
        if (!norm_axis(attr_int(n, "axis", 0), arr.nd, &axis) || axis != arr.nd - 1 ||
            (k && !same_arrangement(&first, &arr)))
            return node_fail(im, n, "joins tokens along another axis than their channels");
        const Value *s = &im->values[si];
        if (k == 0) {
            first = arr;
            nd = s->ndims;
            memcpy(dims, s->dims, (size_t)nd * sizeof *dims);
            dims[nd - 1] = 0;
        }
        dims[nd - 1] += seq_channels(s);
        args.inputs[k] = s->layer;
    }
    const uint32_t l = layer_struct_arguments(args);
    if (!added(im, l, n)) return false;
    im->bias_open = false;
    const Value *y = seq_value(im, (Str){"", 0}, false, l, dims, nd);
    return y && rows_output(im, n, 0, value_index(im, y), &first);
}

/* Cast of values to floats, or of the input's token indices to another integer type: another name. */
static bool tok_cast(Importer *im, const Node *n) {
    const int64_t to = attr_int(n, "to", ONNX_FLOAT);
    const Value *x = n->ninputs > 0 ? find(im, n->inputs[0]) : NULL;
    const bool tokens = x && !x->lazy && !x->seq && x->layer == 0 && x->vector && int_type(x->ctype);
    if (to != ONNX_FLOAT && to != ONNX_FLOAT16 && to != ONNX_DOUBLE && to != ONNX_BFLOAT16 &&
        !(tokens && int_type((int32_t)to)))
        return node_fail(im, n, "casts computed values to integers");
    return alias(im, n, false);
}

/* Gather of a constant table's rows by the input's tokens: an embedding (lazy, so that a position table
   added to it becomes its positions). */
static bool convert_gather(Importer *im, const Node *n) {
    Value *t = n->ninputs == 2 ? find(im, n->inputs[0]) : NULL, *ix = n->ninputs == 2 ? find(im, n->inputs[1]) : NULL;
    if (!t || !ix) return node_fail(im, n, "reads an unknown value");
    if (!t->constant) return node_fail(im, n, "gathers from computed values by computed indices");
    if (attr_int(n, "axis", 0) != 0 || t->ndims != 2 || t->dims[0] <= 0 || t->dims[1] <= 0 || t->dims[0] > UINT32_MAX ||
        t->dims[1] > UINT32_MAX)
        return node_fail(im, n, "gathers other than the rows of a table");
    if (ix->lazy || ix->seq || ix->layer != 0 || !ix->vector)
        return node_fail(im, n, "gathers rows by other indices than the model's input tokens");
    const int64_t dims[3] = {1, ix->c, t->dims[1]};
    const uint32_t ti = value_index(im, t);
    const uint32_t li = new_lazy(im, LZ_EMBED, n);
    if (li == NO_VALUE) return false;
    Lazy *z = &im->lazies[li];
    z->emb.table = ti;
    z->emb.pos = NO_VALUE;
    z->emb.tscale = z->emb.pscale = 1.0f;
    z->emb.act = ACT_NONE;
    return lazy_value(im, n->outputs[0], true, li, dims, 3, true) != NULL;
}

/* What a lazy value is made into when a layer reads it. */
static uint32_t make(Importer *im, uint32_t li) {
    switch (im->lazies[li].kind) {
        case LZ_LINEAR: return make_linear(im, li, im->lazies[li].lin.act);
        case LZ_EMBED: return make_embedding(im, li, im->lazies[li].emb.act);
        case LZ_EXPR: return make_expr(im, li);
        case LZ_VIEW: {
            const uint32_t rows = im->lazies[li].view.rows;
            if (rows != NO_VALUE && materialize(im, &im->values[rows])) return im->values[rows].layer;
            if (rows == NO_VALUE)
                lazy_fail(im, li, "outputs tokens rearranged (in heads, transposed, or part of their channels) where a "
                          "layer reads them");
            return SPINGALETT_NO_LAYER;
        }
        default:
            lazy_fail(im, li, "outputs part of an attention (rotated heads, scores or their softmax) where a layer "
                      "reads it");
            return SPINGALETT_NO_LAYER;
    }
}

/* Whether a node reads sequences or rearrangements of them (views, parts of attention), which the
   translation of transformers handles (1), or element-wise expressions of other layers not made layers
   yet (2); 0 otherwise. */
static int token_node(Importer *im, const Node *n) {
    int what = 0;
    for (int k = 0; k < n->ninputs && k < MAX_NODE_INPUTS; k++) {
        const Value *v = n->inputs[k].n ? find(im, n->inputs[k]) : NULL;
        if (!v || v->constant) continue;
        const int kind = kind_of(im, v);
        if (v->seq || (kind != K_LAYER && kind != LZ_EXPR)) return 1;
        if (kind == LZ_EXPR) what = 2;
    }
    return what;
}

/* A node reading expressions of vectors or maps: element-wise operators (activations among them) extend
   them, others read them made layers. */
static bool convert_expressions(Importer *im, const Node *n) {
    const Str op = n->op;
    if (str_is(op, "Identity") || str_is(op, "Dropout")) return alias(im, n, false);
    if (str_is(op, "Cast")) return tok_cast(im, n);
    if (str_is(op, "Add")) return tok_binary(im, n, E_ADD);
    if (str_is(op, "Sub")) return tok_binary(im, n, E_SUB);
    if (str_is(op, "Mul")) return tok_binary(im, n, E_MUL);
    if (str_is(op, "Div")) return tok_binary(im, n, E_DIV);
    if (str_is(op, "Pow")) return tok_binary(im, n, E_POW);
    const uint8_t u = unary_expr(n);
    if (u != E_X) return tok_unary(im, n, u);
    if (str_is(op, "ReduceMean")) return tok_reduce_mean(im, n);
    return false;
}

static bool convert_tokens(Importer *im, const Node *n) {
    const Str op = n->op;
    if (str_is(op, "Identity") || str_is(op, "Dropout")) return alias(im, n, false);
    if (str_is(op, "Cast")) return tok_cast(im, n);
    if (str_is(op, "Reshape") || str_is(op, "Flatten") || str_is(op, "Transpose") || str_is(op, "Unsqueeze") ||
        str_is(op, "Squeeze") || str_is(op, "Expand") || str_is(op, "Slice") || str_is(op, "Gather") ||
        str_is(op, "Split") || str_is(op, "Tile"))
        return tok_view(im, n);
    if (str_is(op, "MatMul") || str_is(op, "Gemm")) return tok_matmul(im, n);
    if (str_is(op, "Add")) return tok_binary(im, n, E_ADD);
    if (str_is(op, "Sub")) return tok_binary(im, n, E_SUB);
    if (str_is(op, "Mul")) return tok_binary(im, n, E_MUL);
    if (str_is(op, "Div")) return tok_binary(im, n, E_DIV);
    if (str_is(op, "Pow")) return tok_binary(im, n, E_POW);
    const uint8_t u = unary_expr(n);
    if (u != E_X) return tok_unary(im, n, u);
    if (str_is(op, "Where")) return tok_where(im, n);
    if (str_is(op, "Softmax")) return tok_softmax(im, n);
    if (str_is(op, "ReduceMean")) return tok_reduce_mean(im, n);
    if (str_is(op, "LayerNormalization")) return tok_norm(im, n, false);
    if (str_is(op, "RMSNormalization") || str_is(op, "SimplifiedLayerNormalization")) return tok_norm(im, n, true);
    if (str_is(op, "Concat")) return tok_concat(im, n);
    if (str_is(op, "Attention") && !n->domain.n) return convert_attention_op(im, n);
    return fail(im, "ONNX import: operator %.*s (node '%.*s') is not supported on sequences", n->op, n->name);
}

static bool convert_node(Importer *im, const Node *n) {
    if (n->domain.n && !str_is(n->domain, "ai.onnx"))
        return fail(im, "ONNX import: operator %.*s of the domain %.*s is not supported", n->op, n->domain);
    if (folds(im, n)) return fold(im, n);
    const int tokens = token_node(im, n);
    if (tokens == 1) return convert_tokens(im, n);
    if (tokens == 2) {
        if (convert_expressions(im, n) || im->error[0]) return !im->error[0];
        /* any other operator reads the expressions as layers */
        for (int k = 0; k < n->ninputs && k < MAX_NODE_INPUTS; k++) {
            Value *v = n->inputs[k].n ? find(im, n->inputs[k]) : NULL;
            if (v && !v->constant && v->lazy && !materialize(im, v)) return false;
        }
    }
    if (str_is(n->op, "Gather")) return convert_gather(im, n);
    if (str_is(n->op, "Cast")) return tok_cast(im, n);
    ActivationFunction act = activation_of(n, im);
    if (im->error[0]) return false;
    if (act != ACT_COUNT) return convert_activation(im, n, act);
    /* element-wise operators the importer had no layer for: products of layers, expressions */
    const int binary = binary_op(n->op);
    if (binary == F_MUL || binary == F_SUB || binary == F_DIV || binary == F_POW)
        return tok_binary(im, n, binary == F_MUL ? E_MUL : binary == F_SUB ? E_SUB : binary == F_DIV ? E_DIV : E_POW);
    const uint8_t unary = unary_expr(n);
    if (unary != E_X) return tok_unary(im, n, unary);
    if (str_is(n->op, "RMSNormalization") || str_is(n->op, "SimplifiedLayerNormalization"))
        return convert_layer_norm(im, n, true);
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
    if (str_is(n->op, "LayerNormalization")) return convert_layer_norm(im, n, false);
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
        /* over a vector: the mean of a normalization written out */
        const Value *v = n->ninputs > 0 ? find(im, n->inputs[0]) : NULL;
        if (v && !v->constant && v->vector && !v->flat) return tok_reduce_mean(im, n);
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

/* The shape of a graph input or output (N may be symbolic) and its element type. */
static bool input_shape(Pb vi, Str *name, int64_t *dims, int *ndims, int32_t *elem) {
    *ndims = 0;
    *elem = 0;
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
                if (f == 1 && w == WIRE_VARINT) { *elem = (int32_t)pb_varint(&tensor); continue; }
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
            int32_t elem;
            (void)input_shape(pb_bytes(&g), &name, dims, &nd, &elem);
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
    bool have_input = false, tokens = false;
    while (ok && pb_next(&g, &f, &w)) {
        if (f != 11 || w != WIRE_BYTES) { pb_skip(&g, w); continue; }
        Str name = {0};
        int64_t dims[MAX_DIMS];
        int nd;
        int32_t elem;
        bool typed = input_shape(pb_bytes(&g), &name, dims, &nd, &elem);
        if (find(&im, name)) continue;
        if (have_input) { ok = fail(&im, "ONNX import: the model has several inputs ('%.*s' is another)%.*s", name, (Str){"", 0}); break; }
        have_input = true;
        if (!typed || nd < 2 || nd > 4 || dims[1] <= 0 || dims[1] > UINT32_MAX ||
            (nd == 4 && (dims[2] <= 0 || dims[3] <= 0)) || (nd == 3 && (dims[2] <= 0 || dims[2] > UINT32_MAX))) {
            ok = fail(&im, "ONNX import: input '%.*s' must have the shape [N, C, H, W], [N, F] (or tokens [N, T]) or "
                      "[N, T, C]%.*s", name, (Str){"", 0});
            break;
        }
        /* a sequence [N, T, C] is a layer of 1 x T cells; tokens [N, T] (integers) a vector of T indices */
        uint32_t l = nd == 4 ? layer(.net = net, .height = (uint32_t)dims[2], .width = (uint32_t)dims[3],
                                     .channels = (uint32_t)dims[1])
                   : nd == 3 ? layer(.net = net, .height = 1, .width = (uint32_t)dims[1], .channels = (uint32_t)dims[2])
                             : layer(.net = net, .neurons_amount = (uint32_t)dims[1]);
        Value *v = l == 0 ? add_value(&im, name) : NULL;
        ok = v != NULL;
        if (!ok) { fail(&im, "ONNX import: input '%.*s' cannot be an input layer%.*s", name, (Str){"", 0}); break; }
        v->layer = 0;
        v->c = (uint32_t)(nd == 3 ? dims[2] : dims[1]);
        v->h = nd == 4 ? (uint32_t)dims[2] : 1u;
        v->w = nd == 4 ? (uint32_t)dims[3] : nd == 3 ? (uint32_t)dims[1] : 1u;
        v->vector = nd == 2;
        v->ctype = elem;
        tokens = nd == 2 && int_type(elem);
        if (nd == 3) {
            v->seq = true;
            v->ndims = 3;
            v->dims[0] = 1;
            v->dims[1] = dims[1];
            v->dims[2] = dims[2];
        }
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
    bool seq_out = false;
    while (ok && pb_next(&g, &f, &w)) {
        if (f != 12 || w != WIRE_BYTES) { pb_skip(&g, w); continue; }
        Str name = {0};
        int64_t dims[MAX_DIMS];
        int nd;
        int32_t elem;
        (void)input_shape(pb_bytes(&g), &name, dims, &nd, &elem);
        Value *v = find(&im, name);
        if (++outputs > 1) ok = fail(&im, "ONNX import: the model has several outputs ('%.*s' is another)%.*s", name, (Str){"", 0});
        else if (v && !v->constant && v->lazy && !materialize(&im, v)) ok = false;    /* made a layer now */
        else if (!v || v->constant || v->layer + 1 != net->layers || net->layers < 2)
            ok = fail(&im, "ONNX import: output '%.*s' is not the last layer the model computes%.*s", name, (Str){"", 0});
        seq_out = ok && v->seq;
    }
    if (ok) {
        ActivationFunction last = net->act_func[net->layers - 2];
        net->loss_func = last == ACT_SOFTMAX || last == ACT_SIGMOID ? LOSS_CROSS_ENTROPY : LOSS_MSE;
        /* a language model's logits: a class (the next token) a cell */
        if (tokens && seq_out && last == ACT_NONE) net->loss_func = LOSS_SPARSE_CROSS_ENTROPY;
        ok = spingalett_check_graph(net, "ONNX import");
    }

    for (size_t k = 0; k < im.nvalues; k++)
        if (im.values[k].owned) free(im.values[k].data);
    for (size_t k = 0; k < im.nlazies; k++)
        if (im.lazies[k].kind == LZ_LINEAR) free(im.lazies[k].lin.scale), free(im.lazies[k].lin.bias);
    free(im.lazies);
    free(im.enodes);
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
