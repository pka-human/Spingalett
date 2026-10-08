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
 */

#include "Spingalett.Private.h"
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

static bool str_is(Str a, const char *b) { return a.n == strlen(b) && memcmp(a.s, b, a.n) == 0; }
static bool str_eq(Str a, Str b) { return a.n == b.n && memcmp(a.s, b.s, a.n) == 0; }

/* ------------------------------------------------------------------------- tensors and attributes */

#define MAX_DIMS 8
#define MAX_NODE_INPUTS 16
#define MAX_ATTRS 16

enum { ONNX_FLOAT = 1, ONNX_INT32 = 6, ONNX_INT64 = 7, ONNX_FLOAT16 = 10, ONNX_DOUBLE = 11, ONNX_BFLOAT16 = 16 };

typedef struct {
    Str name;
    int64_t dims[MAX_DIMS];
    int ndims;
    int32_t type;
    Pb raw;                     /* raw_data, or the packed typed data */
    bool has_raw, packed_floats, packed_ints, external;
    Pb typed;                   /* float_data / int32_data / int64_data / double_data */
    uint32_t typed_wire;
} TensorRef;

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

static float half_to_float(uint16_t h) { return spingalett_fp16_to_float(h); }

/* Elements the tensor's data can hold at most (raw: its bytes over the element size; typed fields:
   one byte each at least), so that a shape larger than the file is rejected before allocating. */
static uint64_t tensor_capacity(const TensorRef *t) {
    if (t->has_raw) {
        uint64_t size = t->type == ONNX_FLOAT || t->type == ONNX_INT32 ? 4u
                      : t->type == ONNX_INT64 || t->type == ONNX_DOUBLE ? 8u
                      : t->type == ONNX_FLOAT16 || t->type == ONNX_BFLOAT16 ? 2u : 0u;
        return size ? (uint64_t)(t->raw.end - t->raw.p) / size : 0u;
    }
    return t->typed.ok ? (uint64_t)(t->typed.end - t->typed.p) : 0u;
}

/* The tensor's values as floats (count of them); false when the type is not numeric or the data is
   short. */
static bool tensor_floats(const TensorRef *t, float *out, uint64_t count) {
    if (t->external) return false;
    if (t->has_raw) {
        uint64_t size = t->type == ONNX_FLOAT || t->type == ONNX_INT32 ? 4u
                      : t->type == ONNX_INT64 || t->type == ONNX_DOUBLE ? 8u
                      : t->type == ONNX_FLOAT16 || t->type == ONNX_BFLOAT16 ? 2u : 0u;
        if (size == 0 || (uint64_t)(t->raw.end - t->raw.p) < count * size) return false;
        const uint8_t *p = t->raw.p;
        for (uint64_t i = 0; i < count; i++, p += size) {
            switch (t->type) {
                case ONNX_FLOAT: memcpy(&out[i], p, 4); break;
                case ONNX_INT32: { int32_t v; memcpy(&v, p, 4); out[i] = (float)v; break; }
                case ONNX_INT64: { int64_t v; memcpy(&v, p, 8); out[i] = (float)v; break; }
                case ONNX_DOUBLE: { double v; memcpy(&v, p, 8); out[i] = (float)v; break; }
                case ONNX_FLOAT16: out[i] = half_to_float(slett_get16(p)); break;
                default: out[i] = spingalett_bf16_to_float(slett_get16(p)); break;
            }
        }
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
    int64_t ints[MAX_DIMS];
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
        } else if (f == 8 && w == WIRE_VARINT) { int64_t v = (int64_t)pb_varint(&r); if (a->nints < MAX_DIMS) a->ints[a->nints] = v; a->nints++; }
        else if (f == 8 && w == WIRE_BYTES) {
            Pb d = pb_bytes(&r);
            while (d.ok && d.p < d.end) { int64_t v = (int64_t)pb_varint(&d); if (a->nints < MAX_DIMS) a->ints[a->nints] = v; a->nints++; }
        } else pb_skip(&r, w);
    }
}

typedef struct {
    Str op, name;
    Str inputs[MAX_NODE_INPUTS], outputs[4];
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
            if (n->noutputs < 4) n->outputs[n->noutputs] = s;
            n->noutputs++;
        } else if (f == 3 && w == WIRE_BYTES) {
            n->name = pb_str(&r);
        } else if (f == 4 && w == WIRE_BYTES) {
            n->op = pb_str(&r);
        } else if (f == 5 && w == WIRE_BYTES) {
            Pb a = pb_bytes(&r);
            if (n->nattrs < MAX_ATTRS) read_attr(a, &n->attrs[n->nattrs++]);
        } else {
            pb_skip(&r, w);
        }
    }
    return r.ok && n->ninputs <= MAX_NODE_INPUTS && n->noutputs <= 4;
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
    bool alias;                 /* another name of a layer's output (made by a node that read one) */
    float *data;                /* constants */
    int64_t dims[MAX_DIMS];
    int ndims;
    uint64_t count;
} Value;

typedef struct {
    NeuralNetwork *net;
    Value *values;
    size_t nvalues, cap;
    Str *uses;                  /* every node input and graph output, to count readers */
    size_t nuses;
    char error[256];
    bool bias_open;             /* the last layer is a product whose bias a following Add may set */
} Importer;

static bool fail(Importer *im, const char *fmt, Str a, Str b) {
    if (!im->error[0])
        snprintf(im->error, sizeof im->error, fmt, (int)a.n, a.s, (int)b.n, b.s);
    return false;
}

static Value *find(Importer *im, Str name) {
    for (size_t k = im->nvalues; k > 0; k--)
        if (str_eq(im->values[k - 1].name, name)) return &im->values[k - 1];
    return NULL;
}

static Value *add_value(Importer *im, Str name) {
    if (im->nvalues == im->cap) {
        size_t cap = im->cap ? 2 * im->cap : 64;
        Value *v = (Value *)realloc(im->values, cap * sizeof(Value));
        if (!v) return NULL;
        im->values = v;
        im->cap = cap;
    }
    Value *v = &im->values[im->nvalues++];
    memset(v, 0, sizeof *v);
    v->name = name;
    return v;
}

/* Readers of a value (node inputs and graph outputs). */
static uint32_t readers(const Importer *im, Str name) {
    uint32_t n = 0;
    for (size_t k = 0; k < im->nuses; k++) n += str_eq(im->uses[k], name);
    return n;
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

static const Value *layer_input(Importer *im, const Node *n, int k) {
    if (k >= n->ninputs) { fail(im, "ONNX import: %.*s node '%.*s' lacks an input", n->op, n->name); return NULL; }
    const Value *v = find(im, n->inputs[k]);
    if (!v) { fail(im, "ONNX import: input '%.*s' of %.*s is not produced by an earlier node", n->inputs[k], n->op); return NULL; }
    if (v->constant) { fail(im, "ONNX import: %.*s reads the constant '%.*s' where a layer output is expected", n->op, n->inputs[k]); return NULL; }
    return v;
}

static const Value *constant_input(Importer *im, const Node *n, int k, uint64_t count) {
    if (k >= n->ninputs || n->inputs[k].n == 0) return NULL;
    const Value *v = find(im, n->inputs[k]);
    if (!v || !v->constant || (count && v->count != count)) {
        fail(im, "ONNX import: %.*s needs the constant '%.*s' of the expected size", n->op, n->inputs[k]);
        return NULL;
    }
    return v;
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
        if (!(x->h == 1 && x->w == 1) && !x->flat)
            return fail(im, "ONNX import: Softmax node '%.*s' over a map%.*s is not supported (only over vectors)",
                        n->name, (Str){"", 0});
        (void)axis;
    }
    LayerType type = x->layer ? net->shapes[x->layer].type : LAYER_DENSE;
    bool fusable = x->layer > 0 && net->act_func[x->layer - 1] == ACT_NONE && layer_readers(im, x->layer) == 1 &&
                   type != LAYER_MAX_POOL2D && type != LAYER_AVG_POOL2D;
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
    const Value *W = find(im, n->inputs[1]);
    if (!W || !W->constant || W->ndims != 4)
        return fail(im, "ONNX import: Conv node '%.*s' needs constant 4-D weights%.*s", n->name, (Str){"", 0});
    uint32_t kh, kw, sh, sw, ph, pw;
    if (!window_attrs(im, n, &kh, &kw, &sh, &sw, &ph, &pw, x->h, x->w)) return false;
    const uint32_t OC = (uint32_t)W->dims[0], CG = (uint32_t)W->dims[1], groups = (uint32_t)attr_int(n, "group", 1);
    if (W->dims[2] != kh || W->dims[3] != kw || groups == 0 || CG * groups != x->c)
        return fail(im, "ONNX import: Conv node '%.*s' has weights that do not fit its input%.*s", n->name, (Str){"", 0});
    const Value *B = n->ninputs > 2 && n->inputs[2].n ? constant_input(im, n, 2, OC) : NULL;
    if (n->ninputs > 2 && n->inputs[2].n && !B) return false;
    uint32_t l = conv2d(.net = im->net, .inputs = {x->layer}, .input_count = 1, .filters = OC, .kernel_h = kh,
                        .kernel_w = kw, .stride_h = sh, .stride_w = sw, .padding_h = ph, .padding_w = pw,
                        .groups = groups, .act_func = ACT_NONE, .weight_initialization = WEIGHT_INITIALIZATION_NONE);
    if (!added(im, l, n)) return false;
    /* [OC][CG][KH][KW] -> [OC][KH][KW][CG] */
    float *dst = layer_weights(im->net, l);
    for (uint32_t o = 0; o < OC; o++)
        for (uint32_t c = 0; c < CG; c++)
            for (uint32_t i = 0; i < kh; i++)
                for (uint32_t j = 0; j < kw; j++)
                    dst[(((size_t)o * kh + i) * kw + j) * CG + c] = W->data[(((size_t)o * CG + c) * kh + i) * kw + j];
    if (B) memcpy(layer_biases(im->net, l), B->data, OC * sizeof(float));
    im->bias_open = B == NULL;
    return layer_output(im, n, l) != NULL;
}

/* A dense layer y = x W^T + b from rows of W (out x in, row-major) as the reading of x requires:
   the columns of a flattened map reordered from (c, h, w) to (h, w, c). */
static bool dense_layer(Importer *im, const Node *n, const Value *x, const float *W, bool transposed, uint32_t out,
                        uint32_t in, const float *bias, float alpha, float beta) {
    if (in != x->c * x->h * x->w)
        return fail(im, "ONNX import: %.*s node '%.*s' has weights that do not fit its input", n->op, n->name);
    if (x->h * x->w > 1 && !x->flat)
        return fail(im, "ONNX import: %.*s node '%.*s' multiplies a map that is not flattened", n->op, n->name);
    uint32_t l = layer(.net = im->net, .inputs = {x->layer}, .input_count = 1, .neurons_amount = out,
                       .act_func = ACT_NONE, .weight_initialization = WEIGHT_INITIALIZATION_NONE);
    if (!added(im, l, n)) return false;
    float *dst = layer_weights(im->net, l);
    const uint32_t C = x->c, HW = x->h * x->w;
    for (uint32_t o = 0; o < out; o++)
        for (uint32_t k = 0; k < in; k++) {
            /* ONNX column k = (c, p) in c-major order; ours (p, c) */
            uint32_t c = k / HW, p = k % HW;
            float v = transposed ? W[(size_t)k * out + o] : W[(size_t)o * in + k];
            dst[(size_t)o * in + (size_t)p * C + c] = alpha * v;
        }
    float *b = layer_biases(im->net, l);
    for (uint32_t o = 0; o < out; o++) b[o] = bias ? beta * bias[o] : 0.0f;
    im->bias_open = bias == NULL;
    Value *y = layer_output(im, n, l);
    return y != NULL;
}

static bool convert_gemm(Importer *im, const Node *n) {
    const Value *x = layer_input(im, n, 0);
    if (!x) return false;
    if (attr_int(n, "transA", 0) != 0)
        return fail(im, "ONNX import: Gemm node '%.*s' transposes its input (transA)%.*s", n->name, (Str){"", 0});
    const Value *W = find(im, n->inputs[1]);
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
    bool ok = dense_layer(im, n, x, W->data, !transB, out, in, bias, attr_float(n, "alpha", 1.0f), attr_float(n, "beta", 1.0f));
    free(bias);
    return ok;
}

static bool convert_matmul(Importer *im, const Node *n) {
    const Value *x = layer_input(im, n, 0);
    if (!x) return false;
    const Value *W = find(im, n->inputs[1]);
    if (!W || !W->constant || W->ndims != 2)
        return fail(im, "ONNX import: MatMul node '%.*s' needs a constant 2-D right operand%.*s", n->name, (Str){"", 0});
    return dense_layer(im, n, x, W->data, true, (uint32_t)W->dims[1], (uint32_t)W->dims[0], NULL, 1.0f, 1.0f);
}

static bool convert_add(Importer *im, const Node *n) {
    if (n->ninputs != 2) return fail(im, "ONNX import: Add node '%.*s' needs two inputs%.*s", n->name, (Str){"", 0});
    const Value *a = find(im, n->inputs[0]), *b = find(im, n->inputs[1]);
    if (!a || !b) return fail(im, "ONNX import: an input of Add node '%.*s' is unknown%.*s", n->name, (Str){"", 0});
    if (a->constant && b->constant) return fail(im, "ONNX import: Add node '%.*s' of two constants%.*s", n->name, (Str){"", 0});
    if (a->constant || b->constant) {
        /* a bias: per output of the product just added (MatMul, Gemm or Conv without one) */
        const Value *x = a->constant ? b : a, *c = a->constant ? a : b;
        NeuralNetwork *net = im->net;
        LayerType type = x->layer ? net->shapes[x->layer].type : LAYER_DENSE;
        uint32_t rows = x->layer ? spingalett_weight_rows(net, x->layer - 1) : 0;
        if (!im->bias_open || x->layer + 1 != net->layers || (type != LAYER_DENSE && type != LAYER_CONV2D) ||
            net->act_func[x->layer - 1] != ACT_NONE || layer_readers(im, x->layer) != 1 ||
            (c->count != rows && c->count != 1))
            return fail(im, "ONNX import: Add node '%.*s' adds a constant that is not the bias of a product%.*s", n->name,
                        (Str){"", 0});
        float *bias = layer_biases(net, x->layer);
        for (uint32_t o = 0; o < rows; o++) bias[o] += c->count == 1 ? c->data[0] : c->data[o];
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
    if (a->flat != b->flat || a->c != b->c || a->h != b->h || a->w != b->w)
        return fail(im, "ONNX import: Add node '%.*s' adds tensors of different shapes (broadcasting is not supported)%.*s",
                    n->name, (Str){"", 0});
    const bool flat = a->flat;
    uint32_t l = add_layers(.net = im->net, .inputs = {a->layer, b->layer}, .input_count = 2, .act_func = ACT_NONE);
    if (!added(im, l, n)) return false;
    im->bias_open = false;
    Value *y = layer_output(im, n, l);
    if (y) y->flat = flat;
    return y != NULL;
}

static bool convert_concat(Importer *im, const Node *n) {
    if (n->ninputs < 1 || n->ninputs > SPINGALETT_MAX_INPUTS)
        return fail(im, "ONNX import: Concat node '%.*s' joins 1 to 16 inputs%.*s", n->name, (Str){"", 0});
    LayerArgs args = {.net = im->net, .type = LAYER_CONCAT, .input_count = (uint32_t)n->ninputs, .act_func = ACT_NONE};
    bool flat = false, maps = false;
    for (int k = 0; k < n->ninputs; k++) {
        const Value *v = layer_input(im, n, k);
        if (!v) return false;
        args.inputs[k] = v->layer;
        flat |= v->flat;
        maps |= v->h * v->w > 1;
    }
    int64_t axis = attr_int(n, "axis", 1);
    if (flat || (maps ? axis != 1 && axis != -3 : axis != 1 && axis != -1))
        return fail(im, "ONNX import: Concat node '%.*s' joins along another axis than the channels%.*s", n->name,
                    (Str){"", 0});
    uint32_t l = layer_struct_arguments(args);
    if (!added(im, l, n)) return false;
    im->bias_open = false;
    return layer_output(im, n, l) != NULL;
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
    const bool flat = x->flat;
    memcpy(layer_weights(net, l), p[0]->data, C * sizeof(float));
    memcpy(layer_biases(net, l), p[1]->data, C * sizeof(float));
    memcpy(net->running_mean + b, p[2]->data, C * sizeof(float));
    memcpy(net->running_var + b, p[3]->data, C * sizeof(float));
    im->bias_open = false;
    Value *y = layer_output(im, n, l);
    if (y) y->flat = flat;
    return y != NULL;
}

/* The same layer under another name (Identity, Dropout at inference) or flattened (Flatten,
   Reshape to two dimensions). */
static bool alias(Importer *im, const Node *n, bool flatten) {
    const Value *x = find(im, n->inputs[0]);
    if (!x) return fail(im, "ONNX import: input '%.*s' of %.*s is unknown", n->inputs[0], n->op);
    Value copy = *x;
    if (flatten && !x->constant) copy.flat = true;
    copy.alias = !x->constant;
    if (x->constant && x->data) {   /* a renamed constant gets its own copy of the data */
        copy.data = (float *)malloc((size_t)(x->count ? x->count : 1) * sizeof(float));
        if (!copy.data) return false;
        memcpy(copy.data, x->data, (size_t)x->count * sizeof(float));
    }
    Value *y = add_value(im, n->outputs[0]);
    if (!y) return false;
    Str name = y->name;
    *y = copy;
    y->name = name;
    return true;
}

static bool convert_reshape(Importer *im, const Node *n) {
    const Value *x = find(im, n->inputs[0]);
    const Value *shape = n->ninputs > 1 ? find(im, n->inputs[1]) : NULL;
    if (!x || x->constant || !shape || !shape->constant)
        return fail(im, "ONNX import: Reshape node '%.*s' needs a layer output and a constant shape%.*s", n->name, (Str){"", 0});
    /* to [N, everything]: a flattening; to the shape it has: nothing */
    uint64_t units = (uint64_t)x->c * x->h * x->w;
    if (shape->count == 2 && (shape->data[1] == -1.0f || (uint64_t)shape->data[1] == units))
        return alias(im, n, true);
    if (shape->count == 4 && (uint64_t)shape->data[1] == x->c && (uint64_t)shape->data[2] == x->h &&
        (uint64_t)shape->data[3] == x->w)
        return alias(im, n, false);
    return fail(im, "ONNX import: Reshape node '%.*s' changes the shape other than by flattening it%.*s", n->name,
                (Str){"", 0});
}

static bool convert_constant(Importer *im, const Node *n) {
    const Attr *a = attr(n, "value");
    if (!a || !a->has_t) {
        const Attr *f = attr(n, "value_float"), *i = attr(n, "value_ints");
        Value *v = add_value(im, n->outputs[0]);
        if (!v) return false;
        v->constant = true;
        if (f) {
            v->count = 1;
            v->ndims = 0;
            v->data = (float *)malloc(sizeof(float));
            if (v->data) v->data[0] = f->f;
        } else if (i) {
            v->count = (uint64_t)i->nints;
            v->ndims = 1;
            v->dims[0] = i->nints;
            v->data = (float *)malloc((size_t)(i->nints ? i->nints : 1) * sizeof(float));
            for (int k = 0; v->data && k < i->nints && k < MAX_DIMS; k++) v->data[k] = (float)i->ints[k];
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
    v->data = (float *)malloc((size_t)(v->count ? v->count : 1) * sizeof(float));
    if (!v->data || !tensor_floats(&t, v->data, v->count))
        return fail(im, "ONNX import: Constant node '%.*s' holds data that cannot be read%.*s", n->name, (Str){"", 0});
    return true;
}

static bool convert_node(Importer *im, const Node *n) {
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
        const Value *ax = n->ninputs > 1 && n->inputs[1].n ? find(im, n->inputs[1]) : NULL;
        int64_t axes[2] = {0, 0};
        int count = a ? a->nints : ax && ax->constant ? (int)ax->count : 0;
        for (int k = 0; k < count && k < 2; k++) axes[k] = a ? a->ints[k] : (int64_t)ax->data[k];
        for (int k = 0; k < 2; k++) if (axes[k] < 0) axes[k] += 4;
        if (x->flat || count != 2 || !((axes[0] == 2 && axes[1] == 3) || (axes[0] == 3 && axes[1] == 2)))
            return fail(im, "ONNX import: ReduceMean node '%.*s' averages other axes than a map's height and width%.*s",
                        n->name, (Str){"", 0});
        uint32_t l = global_avg_pool2d(.net = im->net, .inputs = {x->layer}, .input_count = 1);
        if (!added(im, l, n)) return false;
        im->bias_open = false;
        return layer_output(im, n, l) != NULL;
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

NeuralNetwork *spingalett_import_onnx_from_memory(const void *data, size_t size) {
    if (!data) {
        set_error(SPINGALETT_ERR_INVALID, "ONNX import: data is NULL");
        return NULL;
    }
    Pb model = {(const uint8_t *)data, (const uint8_t *)data + size, true}, graph = {NULL, NULL, false};
    uint32_t f, w;
    while (pb_next(&model, &f, &w)) {
        if (f == 7 && w == WIRE_BYTES) graph = pb_bytes(&model);
        else pb_skip(&model, w);
    }
    if (!model.ok || !graph.ok) {
        set_error(SPINGALETT_ERR_INVALID, "ONNX import: not an ONNX model (no graph)");
        return NULL;
    }

    Importer im = {0};
    bool ok = true;
    /* readers of every value: node inputs and graph outputs */
    size_t uses = 0;
    for (int pass = 0; pass < 2 && ok; pass++) {
        Pb g = graph;
        while (pb_next(&g, &f, &w)) {
            if (f == 1 && w == WIRE_BYTES) {
                Node n;
                ok = read_node(pb_bytes(&g), &n) && ok;
                for (int k = 0; k < n.ninputs && k < MAX_NODE_INPUTS; k++, uses++)
                    if (pass) im.uses[uses] = n.inputs[k];
            } else if (f == 12 && w == WIRE_BYTES) {
                Str name = {0};
                int64_t dims[MAX_DIMS];
                int nd;
                (void)input_shape(pb_bytes(&g), &name, dims, &nd);
                if (pass) im.uses[uses] = name;
                uses++;
            } else {
                pb_skip(&g, w);
            }
        }
        ok = ok && g.ok;
        if (!pass && ok) {
            im.uses = (Str *)calloc(uses + 1, sizeof(Str));
            ok = im.uses != NULL;
            im.nuses = uses;
            uses = 0;
        }
    }

    /* constants: the initializers */
    Pb g = graph;
    while (ok && pb_next(&g, &f, &w)) {
        if (f != 5 || w != WIRE_BYTES) { pb_skip(&g, w); continue; }
        TensorRef t;
        read_tensor(pb_bytes(&g), &t);
        if (t.ndims > MAX_DIMS || (!t.external && tensor_count(&t) > tensor_capacity(&t))) {
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
        v->data = (float *)malloc((size_t)(v->count ? v->count : 1) * sizeof(float));
        if (!v->data || t.external || !tensor_floats(&t, v->data, v->count)) {
            ok = fail(&im, "ONNX import: initializer '%.*s' cannot be read%.*s", t.name,
                      t.external ? (Str){" (its data is in an external file)", 34} : (Str){"", 0});
        }
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

    for (size_t k = 0; k < im.nvalues; k++) free(im.values[k].data);
    free(im.values);
    free(im.uses);
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
    size_t size = 0;
    void *data = spingalett_read_file(path, &size);
    if (!data) return NULL;
    NeuralNetwork *net = spingalett_import_onnx_from_memory(data, size);
    spingalett_aligned_free(data);
    return net;
}
