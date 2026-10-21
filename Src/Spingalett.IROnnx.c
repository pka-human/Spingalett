/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/*
 * ONNX files as graphs (Spingalett.IR.h), one operator to one: the model's graph's initializers become constants
 * (raw or typed data, or external data files beside the model), its inputs and outputs the graph's (their shapes'
 * numbers and symbols kept), its nodes the graph's with their attributes; an operator the library does not have is
 * named with the file's opset. A reader of the protocol buffer format of its own, as small as ONNX needs.
 */

#include "Spingalett.IR.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    const uint8_t *p, *end;
    bool ok;
} Pb;

enum { WIRE_VARINT = 0, WIRE_FIXED64 = 1, WIRE_BYTES = 2, WIRE_FIXED32 = 5 };

static uint64_t varint(Pb *r) {
    uint64_t v = 0;
    for (int shift = 0; shift < 64; shift += 7) {
        if (r->p >= r->end) { r->ok = false; return 0; }
        const uint8_t b = *r->p++;
        v |= (uint64_t)(b & 0x7Fu) << shift;
        if (!(b & 0x80u)) return v;
    }
    r->ok = false;
    return 0;
}

static bool next(Pb *r, uint32_t *field, uint32_t *wire) {
    if (!r->ok || r->p >= r->end) return false;
    const uint64_t key = varint(r);
    *field = (uint32_t)(key >> 3);
    *wire = (uint32_t)(key & 7u);
    return r->ok;
}

static Pb sub(Pb *r) {
    Pb s = {NULL, NULL, false};
    const uint64_t n = varint(r);
    if (!r->ok || n > (uint64_t)(r->end - r->p)) { r->ok = false; return s; }
    s.p = r->p;
    s.end = r->p + n;
    s.ok = true;
    r->p += n;
    return s;
}

static void skip(Pb *r, uint32_t wire) {
    if (wire == WIRE_VARINT) (void)varint(r);
    else if (wire == WIRE_FIXED64) r->p = (size_t)(r->end - r->p) >= 8 ? r->p + 8 : (r->ok = false, r->end);
    else if (wire == WIRE_FIXED32) r->p = (size_t)(r->end - r->p) >= 4 ? r->p + 4 : (r->ok = false, r->end);
    else if (wire == WIRE_BYTES) (void)sub(r);
    else r->ok = false;
}

static float fixed32_float(Pb *r) {
    if (r->end - r->p < 4) { r->ok = false; return 0.0f; }
    float f;
    memcpy(&f, r->p, 4);
    r->p += 4;
    return f;
}

typedef struct {
    const char *p;
    size_t n;
} Str;

static Str string(Pb *r) {
    Pb s = sub(r);
    Str out = {(const char *)s.p, s.ok ? (size_t)(s.end - s.p) : 0u};
    return out;
}

static bool str_eq(Str s, const char *c) {
    return s.n == strlen(c) && !memcmp(s.p, c, s.n);
}

typedef struct {
    SpingalettGraph *g;
    const char *dir;
    int64_t opset;
    char error[300];
} Reader;

static bool fail(Reader *rd, const char *format, const char *a, const char *b) {
    if (!rd->error[0]) snprintf(rd->error, sizeof rd->error, format, a ? a : "", b ? b : "");
    return false;
}

/* ------------------------------------------------------------------------- tensors */

/* ONNX's TensorProto types the graphs hold, with what they read: FLOAT, UINT8, INT8, INT16 (as int32), INT32, INT64,
   BOOL, FLOAT16, DOUBLE (as float), UINT16 and UINT32 (as int64? int32 / int64), UINT64 (as int64), BFLOAT16. */
static uint8_t graph_dtype(int64_t onnx) {
    switch (onnx) {
        case 1: return SPINGALETT_DTYPE_FLOAT32;
        case 2: return SPINGALETT_DTYPE_UINT8;
        case 3: return SPINGALETT_DTYPE_INT8;
        case 4: case 5: return SPINGALETT_DTYPE_INT32;          /* UINT16, INT16 */
        case 6: return SPINGALETT_DTYPE_INT32;
        case 7: case 12: case 13: return SPINGALETT_DTYPE_INT64; /* INT64, UINT32, UINT64 */
        case 9: return SPINGALETT_DTYPE_BOOL;
        case 10: return SPINGALETT_DTYPE_FLOAT16;
        case 11: return SPINGALETT_DTYPE_FLOAT32;               /* DOUBLE */
        case 16: return SPINGALETT_DTYPE_BFLOAT16;
        default: return SPINGALETT_DTYPE_UNDEFINED;
    }
}

/* The raw bytes of a stored value of ONNX type `onnx`, element i, into the graph's dtype at out. */
static void convert_raw(int64_t onnx, const uint8_t *raw, uint64_t i, void *out) {
    switch (onnx) {
        case 1: memcpy((float *)out + i, raw + 4 * i, 4); break;
        case 2: case 3: case 9: ((uint8_t *)out)[i] = raw[i]; break;
        case 4: { uint16_t v; memcpy(&v, raw + 2 * i, 2); ((int32_t *)out)[i] = v; break; }
        case 5: { int16_t v; memcpy(&v, raw + 2 * i, 2); ((int32_t *)out)[i] = v; break; }
        case 6: memcpy((int32_t *)out + i, raw + 4 * i, 4); break;
        case 7: memcpy((int64_t *)out + i, raw + 8 * i, 8); break;
        case 12: { uint32_t v; memcpy(&v, raw + 4 * i, 4); ((int64_t *)out)[i] = v; break; }
        case 13: { uint64_t v; memcpy(&v, raw + 8 * i, 8); ((int64_t *)out)[i] = (int64_t)v; break; }
        case 10: case 16: memcpy((uint16_t *)out + i, raw + 2 * i, 2); break;
        case 11: { double d; memcpy(&d, raw + 8 * i, 8); ((float *)out)[i] = (float)d; break; }
        default: break;
    }
}

static size_t onnx_size(int64_t onnx) {
    switch (onnx) {
        case 1: case 6: case 12: return 4;
        case 2: case 3: case 9: return 1;
        case 4: case 5: case 10: case 16: return 2;
        case 7: case 11: case 13: return 8;
        default: return 0;
    }
}

/* The bytes of an external data file in the reader's directory, from offset on: the file mapped once, kept with the
   graph; NULL with the error set. */
static const uint8_t *external(Reader *rd, Str location, uint64_t offset, uint64_t length, uint64_t *available) {
    SpingalettGraph *g = rd->g;
    if (location.n == 0 || location.n >= 1024 || memchr(location.p, '\0', location.n)) {
        set_error(SPINGALETT_ERR_INVALID, "graph: an external data file of no name, or too long a name");
        return NULL;
    }
    char path[4096];
    snprintf(path, sizeof path, "%s/%.*s", rd->dir ? rd->dir : ".", (int)location.n, location.p);
    SpgFileView *f = NULL;
    for (uint32_t k = 0; k < g->external_count && !f; k++)
        if (!strcmp(g->external_paths[k], path)) f = &g->externals[k];
    if (!f) {
        SpgFileView *views = (SpgFileView *)realloc(g->externals, (g->external_count + 1u) * sizeof(SpgFileView));
        char **paths = views ? (char **)realloc(g->external_paths, (g->external_count + 1u) * sizeof(char *)) : NULL;
        if (views) g->externals = views;
        if (paths) g->external_paths = paths;
        if (!views || !paths) {
            set_error(SPINGALETT_ERR_ALLOC, "graph: out of memory");
            return NULL;
        }
        f = &g->externals[g->external_count];
        memset(f, 0, sizeof *f);
        g->external_paths[g->external_count] = ir_strdup(g, path, strlen(path));
        if (!g->external_paths[g->external_count] || !spingalett_file_open(f, path)) return NULL;
        g->external_count++;
    }
    if (offset > f->size || length > f->size - offset) {
        set_error(SPINGALETT_ERR_INVALID, "graph: an external data file shorter than its tensors");
        return NULL;
    }
    *available = f->size - offset;
    return f->data + offset;
}

/* A TensorProto as a constant value of its name (or of `name` when given); its index, IR_NONE on error. */
static uint32_t read_tensor(Reader *rd, Pb t, const char *name, size_t name_len) {
    int64_t dims[IR_MAX_RANK];
    uint32_t rank = 0;
    int64_t type = 0;
    Str tname = {NULL, 0}, location = {NULL, 0};
    Pb raw = {NULL, NULL, false}, floats = raw, int32s = raw, int64s = raw, doubles = raw, uint64s = raw;
    uint64_t ext_offset = 0, ext_length = 0;
    bool external_data = false, too_many = false;
    uint32_t field, wire;
    while (next(&t, &field, &wire)) {
        if (field == 1 && wire == WIRE_VARINT) {
            if (rank < IR_MAX_RANK) dims[rank++] = (int64_t)varint(&t);
            else { too_many = true; (void)varint(&t); }
        } else if (field == 1 && wire == WIRE_BYTES) {
            Pb d = sub(&t);
            while (d.ok && d.p < d.end) {
                if (rank < IR_MAX_RANK) dims[rank++] = (int64_t)varint(&d);
                else { too_many = true; (void)varint(&d); }
            }
        } else if (field == 2 && wire == WIRE_VARINT) type = (int64_t)varint(&t);
        else if (field == 4 && wire == WIRE_BYTES) floats = sub(&t);
        else if (field == 5 && wire == WIRE_BYTES) int32s = sub(&t);
        else if (field == 7 && wire == WIRE_BYTES) int64s = sub(&t);
        else if (field == 8 && wire == WIRE_BYTES) tname = string(&t);
        else if (field == 9 && wire == WIRE_BYTES) raw = sub(&t);
        else if (field == 10 && wire == WIRE_BYTES) doubles = sub(&t);
        else if (field == 11 && wire == WIRE_BYTES) uint64s = sub(&t);
        else if (field == 13 && wire == WIRE_BYTES) {
            Pb e = sub(&t);
            Str key = {NULL, 0}, value = {NULL, 0};
            uint32_t f2, w2;
            while (next(&e, &f2, &w2)) {
                if (f2 == 1 && w2 == WIRE_BYTES) key = string(&e);
                else if (f2 == 2 && w2 == WIRE_BYTES) value = string(&e);
                else skip(&e, w2);
            }
            char number[32];
            const size_t n = value.n < sizeof number - 1 ? value.n : sizeof number - 1;
            if (value.p) memcpy(number, value.p, n);
            number[n] = '\0';
            if (str_eq(key, "location")) location = value;
            else if (str_eq(key, "offset")) ext_offset = strtoull(number, NULL, 10);
            else if (str_eq(key, "length")) ext_length = strtoull(number, NULL, 10);
        } else if (field == 14 && wire == WIRE_VARINT) external_data = varint(&t) == 1;
        else if ((field == 4 || field == 5 || field == 7 || field == 10 || field == 11) && wire != WIRE_BYTES) {
            /* repeated values written unpacked: read in a second pass below */
            skip(&t, wire);
        } else skip(&t, wire);
    }
    if (!t.ok) return fail(rd, "graph: a damaged tensor%s%s", NULL, NULL), IR_NONE;
    const Str use = name ? (Str){name, name_len} : tname;
    char shown[200];
    snprintf(shown, sizeof shown, "%.*s", (int)(use.n < 150 ? use.n : 150), use.p ? use.p : "");
    if (too_many) return fail(rd, "graph: the tensor '%s' has more than eight axes%s", shown, NULL), IR_NONE;
    const uint8_t dtype = graph_dtype(type);
    if (!dtype) {
        char code[16];
        snprintf(code, sizeof code, "%lld", (long long)type);
        return fail(rd, "graph: the tensor '%s' is of ONNX's type %s, which graphs do not hold", shown, code), IR_NONE;
    }
    uint64_t count = 1;
    for (uint32_t k = 0; k < rank; k++) {
        if (dims[k] < 0) return fail(rd, "graph: the tensor '%s' has a negative size%s", shown, NULL), IR_NONE;
        count *= (uint64_t)dims[k];
    }
    SpingalettGraph *g = rd->g;
    const uint32_t v = ir_value(g, use.p, use.n);
    if (v == IR_NONE) return IR_NONE;
    IrValue *value = &g->values[v];
    value->kind = VALUE_CONST;
    value->declared = true;
    value->dtype = dtype;
    value->rank = (uint8_t)rank;
    for (uint32_t k = 0; k < rank; k++) value->dims[k] = dims[k];
    const size_t out_size = ir_dtype_size(dtype), in_size = onnx_size(type);
    void *out = ir_alloc(g, (size_t)count * out_size);
    if (!out) return IR_NONE;
    value->data = out;
    const uint8_t *bytes = NULL;
    uint64_t available = 0;
    if (external_data) {
        if (!location.p) return fail(rd, "graph: the tensor '%s' names no external file%s", shown, NULL), IR_NONE;
        bytes = external(rd, location, ext_offset, ext_length ? ext_length : count * in_size, &available);
        if (!bytes) return fail(rd, "graph: the external data of the tensor '%s' cannot be read (%s)", shown,
                                 spingalett_last_error_message()), IR_NONE;
    } else if (raw.ok) {
        bytes = raw.p;
        available = (uint64_t)(raw.end - raw.p);
    }
    if (bytes) {
        if (available < count * in_size)
            return fail(rd, "graph: the tensor '%s' holds fewer bytes than its shape needs%s", shown, NULL), IR_NONE;
        for (uint64_t i = 0; i < count; i++) convert_raw(type, bytes, i, out);
        return v;
    }
    /* typed fields (packed) */
    uint64_t i = 0;
    if (type == 1 && floats.ok) {
        for (; i < count && floats.p + 4 <= floats.end; i++) ((float *)out)[i] = fixed32_float(&floats);
    } else if (type == 11 && doubles.ok) {
        for (; i < count && doubles.p + 8 <= doubles.end; i++) {
            double d;
            memcpy(&d, doubles.p, 8);
            doubles.p += 8;
            ((float *)out)[i] = (float)d;
        }
    } else if (type == 7 && int64s.ok) {
        for (; i < count && int64s.p < int64s.end; i++) ((int64_t *)out)[i] = (int64_t)varint(&int64s);
    } else if ((type == 12 || type == 13) && uint64s.ok) {
        for (; i < count && uint64s.p < uint64s.end; i++) ((int64_t *)out)[i] = (int64_t)varint(&uint64s);
    } else if (int32s.ok) {
        for (; i < count && int32s.p < int32s.end; i++) {
            const int64_t x = (int64_t)varint(&int32s);
            if (out_size == 4) ((int32_t *)out)[i] = (int32_t)x;
            else if (out_size == 2) ((uint16_t *)out)[i] = (uint16_t)x;
            else ((uint8_t *)out)[i] = (uint8_t)x;
        }
    }
    if (i != count && count) return fail(rd, "graph: the tensor '%s' holds fewer values than its shape needs%s", shown,
                                         NULL), IR_NONE;
    return v;
}

/* ------------------------------------------------------------------------- inputs, outputs */

/* A ValueInfoProto's name, type and shape into value v (declared). */
static bool read_value_info(Reader *rd, Pb info, uint32_t *v_out, bool create) {
    Str name = {NULL, 0};
    int64_t elem = 0;
    int64_t dims[IR_MAX_RANK];
    Str params[IR_MAX_RANK];
    uint32_t rank = 0;
    bool shaped = false, too_many = false;
    uint32_t field, wire;
    while (next(&info, &field, &wire)) {
        if (field == 1 && wire == WIRE_BYTES) name = string(&info);
        else if (field == 2 && wire == WIRE_BYTES) {
            Pb type = sub(&info);
            uint32_t f2, w2;
            while (next(&type, &f2, &w2)) {
                if (f2 == 1 && w2 == WIRE_BYTES) {
                    Pb tensor = sub(&type);
                    uint32_t f3, w3;
                    while (next(&tensor, &f3, &w3)) {
                        if (f3 == 1 && w3 == WIRE_VARINT) elem = (int64_t)varint(&tensor);
                        else if (f3 == 2 && w3 == WIRE_BYTES) {
                            shaped = true;
                            Pb shape = sub(&tensor);
                            uint32_t f4, w4;
                            while (next(&shape, &f4, &w4)) {
                                if (f4 == 1 && w4 == WIRE_BYTES) {
                                    Pb dim = sub(&shape);
                                    int64_t value = -1;
                                    Str param = {NULL, 0};
                                    uint32_t f5, w5;
                                    while (next(&dim, &f5, &w5)) {
                                        if (f5 == 1 && w5 == WIRE_VARINT) value = (int64_t)varint(&dim);
                                        else if (f5 == 2 && w5 == WIRE_BYTES) param = string(&dim);
                                        else skip(&dim, w5);
                                    }
                                    if (rank < IR_MAX_RANK) {
                                        dims[rank] = value;
                                        params[rank++] = param;
                                    } else too_many = true;
                                } else skip(&shape, w4);
                            }
                        } else skip(&tensor, w3);
                    }
                } else skip(&type, w2);
            }
        } else skip(&info, wire);
    }
    if (!info.ok || !name.p) return fail(rd, "graph: a damaged input or output%s%s", NULL, NULL);
    char shown[200];
    snprintf(shown, sizeof shown, "%.*s", (int)(name.n < 150 ? name.n : 150), name.p);
    if (too_many) return fail(rd, "graph: '%s' has more than eight axes%s", shown, NULL);
    SpingalettGraph *g = rd->g;
    uint32_t v = create ? ir_value(g, name.p, name.n) : ir_find_value(g, name.p, name.n);
    if (v == IR_NONE) return create ? false : fail(rd, "graph: nothing computes the output '%s'%s", shown, NULL);
    IrValue *value = &g->values[v];
    if (create || value->kind == VALUE_NODE) {
        value->dtype = graph_dtype(elem);
        if (create && !value->dtype) {
            char code[16];
            snprintf(code, sizeof code, "%lld", (long long)elem);
            return fail(rd, "graph: the input '%s' is of ONNX's type %s, which graphs do not take", shown, code);
        }
        if (shaped) {
            value->declared = true;
            value->rank = (uint8_t)rank;
            for (uint32_t k = 0; k < rank; k++) {
                value->dims[k] = dims[k] >= 0 && !params[k].p ? dims[k] : -1;
                value->sym[k] = params[k].p ? ir_symbol(g, params[k].p, params[k].n) : 0;
            }
        } else if (create) {
            value->declared = false;
        }
    }
    *v_out = v;
    return true;
}

/* ------------------------------------------------------------------------- nodes */

static bool read_attribute(Reader *rd, Pb a, IrAttr *out) {
    SpingalettGraph *g = rd->g;
    Str name = {NULL, 0}, s = {NULL, 0};
    int64_t type = 0, i = 0;
    float f = 0.0f;
    Pb tensor = {NULL, NULL, false};
    int64_t ints[1024];
    float floats[1024];
    uint32_t nints = 0, nfloats = 0;
    bool has_i = false, has_f = false, has_s = false, graph = false;
    uint32_t field, wire;
    while (next(&a, &field, &wire)) {
        if (field == 1 && wire == WIRE_BYTES) name = string(&a);
        else if (field == 2 && wire == WIRE_FIXED32) { f = fixed32_float(&a); has_f = true; }
        else if (field == 3 && wire == WIRE_VARINT) { i = (int64_t)varint(&a); has_i = true; }
        else if (field == 4 && wire == WIRE_BYTES) { s = string(&a); has_s = true; }
        else if (field == 5 && wire == WIRE_BYTES) tensor = sub(&a);
        else if (field == 6 && wire == WIRE_BYTES) { graph = true; skip(&a, wire); }
        else if (field == 7 && wire == WIRE_FIXED32) { float x = fixed32_float(&a); if (nfloats < 1024) floats[nfloats++] = x; }
        else if (field == 7 && wire == WIRE_BYTES) {
            Pb p = sub(&a);
            while (p.ok && p.p + 4 <= p.end) { float x = fixed32_float(&p); if (nfloats < 1024) floats[nfloats++] = x; }
        } else if (field == 8 && wire == WIRE_VARINT) { int64_t x = (int64_t)varint(&a); if (nints < 1024) ints[nints++] = x; }
        else if (field == 8 && wire == WIRE_BYTES) {
            Pb p = sub(&a);
            while (p.ok && p.p < p.end) { int64_t x = (int64_t)varint(&p); if (nints < 1024) ints[nints++] = x; }
        } else if (field == 20 && wire == WIRE_VARINT) type = (int64_t)varint(&a);
        else skip(&a, wire);
    }
    if (!a.ok || !name.p) return fail(rd, "graph: a damaged attribute%s%s", NULL, NULL);
    memset(out, 0, sizeof *out);
    out->name = ir_strdup(g, name.p, name.n);
    if (!out->name) return false;
    char shown[100];
    snprintf(shown, sizeof shown, "%.*s", (int)(name.n < 80 ? name.n : 80), name.p);
    if (graph) return fail(rd, "graph: the attribute '%s' holds a subgraph (If, Loop and Scan are not supported)%s", shown,
                           NULL);
    /* the type: AttributeProto's, else what the fields hold */
    switch (type) {
        case 1: out->type = ATTR_FLOAT; out->f = f; break;
        case 2: out->type = ATTR_INT; out->i = i; break;
        case 3: out->type = ATTR_STRING; break;
        case 4: out->type = ATTR_TENSOR; break;
        case 6: out->type = ATTR_FLOATS; break;
        case 7: out->type = ATTR_INTS; break;
        default:
            out->type = tensor.ok ? ATTR_TENSOR : has_s ? ATTR_STRING : has_f ? ATTR_FLOAT : has_i ? ATTR_INT
                      : nfloats ? ATTR_FLOATS : ATTR_INTS;
            out->f = f;
            out->i = i;
            break;
    }
    if (out->type == ATTR_STRING) {
        out->s = ir_strdup(g, s.p ? s.p : "", s.n);
        if (!out->s) return false;
    } else if (out->type == ATTR_TENSOR) {
        if (!tensor.ok) return fail(rd, "graph: the attribute '%s' has no tensor%s", shown, NULL);
        out->tensor = read_tensor(rd, tensor, "", 0);
        if (out->tensor == IR_NONE) return false;
    } else if (out->type == ATTR_INTS) {
        if (nints >= 1024) return fail(rd, "graph: the attribute '%s' holds more than 1023 integers%s", shown, NULL);
        out->ints = (int64_t *)ir_alloc(g, (nints ? nints : 1u) * sizeof(int64_t));
        if (!out->ints) return false;
        memcpy(out->ints, ints, nints * sizeof(int64_t));
        out->count = nints;
    } else if (out->type == ATTR_FLOATS) {
        if (nfloats >= 1024) return fail(rd, "graph: the attribute '%s' holds more than 1023 floats%s", shown, NULL);
        out->floats = (float *)ir_alloc(g, (nfloats ? nfloats : 1u) * sizeof(float));
        if (!out->floats) return false;
        memcpy(out->floats, floats, nfloats * sizeof(float));
        out->count = nfloats;
    }
    return true;
}

/* A value read by name: a graph input, a constant, or the output of a node before or after (made now). */
static uint32_t use_value(Reader *rd, Str name) {
    if (!name.n) return IR_NONE;
    uint32_t v = ir_find_value(rd->g, name.p, name.n);
    if (v == IR_NONE) v = ir_value(rd->g, name.p, name.n);
    return v;
}

static bool read_node(Reader *rd, Pb n) {
    SpingalettGraph *g = rd->g;
    Str op = {NULL, 0}, domain = {NULL, 0}, nname = {NULL, 0};
    Str ins[64], outs[64];
    uint32_t nin = 0, nout = 0, nattr = 0;
    Pb attrs[64];
    uint32_t field, wire;
    while (next(&n, &field, &wire)) {
        if (field == 1 && wire == WIRE_BYTES) { Str s = string(&n); if (nin < 64) ins[nin] = s; nin++; }
        else if (field == 2 && wire == WIRE_BYTES) { Str s = string(&n); if (nout < 64) outs[nout] = s; nout++; }
        else if (field == 3 && wire == WIRE_BYTES) nname = string(&n);
        else if (field == 4 && wire == WIRE_BYTES) op = string(&n);
        else if (field == 5 && wire == WIRE_BYTES) { Pb a = sub(&n); if (nattr < 64) attrs[nattr] = a; nattr++; }
        else if (field == 7 && wire == WIRE_BYTES) domain = string(&n);
        else skip(&n, wire);
    }
    if (!n.ok || !op.p) return fail(rd, "graph: a damaged node%s%s", NULL, NULL);
    char shown_op[100], shown_domain[100];
    snprintf(shown_op, sizeof shown_op, "%.*s", (int)(op.n < 80 ? op.n : 80), op.p);
    snprintf(shown_domain, sizeof shown_domain, "%.*s", (int)(domain.n < 80 ? domain.n : 80), domain.p ? domain.p : "");
    if (nin > 64 || nout > 64 || nattr > 64)
        return fail(rd, "graph: a node of %s has more than 64 inputs, outputs or attributes%s", shown_op, NULL);
    const IrOp code = ir_op_of(op.p, op.n, domain.p ? domain.p : "", domain.n);
    if (code == IR_OP_COUNT) {
        char version[48];
        snprintf(version, sizeof version, "%s, opset %lld", domain.n ? shown_domain : "ONNX's domain", (long long)rd->opset);
        return fail(rd, "graph: the operator %s (%s) is not supported", shown_op, version);
    }
    const uint32_t k = ir_node(g, code);
    if (k == IR_NONE) return false;
    IrNode *node = &g->nodes[k];
    node->name = nname.n ? ir_strdup(g, nname.p, nname.n) : ir_strdup(g, op.p, op.n);
    node->inputs = (uint32_t *)ir_alloc(g, (nin ? nin : 1u) * sizeof(uint32_t));
    node->outputs = (uint32_t *)ir_alloc(g, (nout ? nout : 1u) * sizeof(uint32_t));
    node->attrs = (IrAttr *)ir_alloc(g, (nattr ? nattr : 1u) * sizeof(IrAttr));
    if (!node->name || !node->inputs || !node->outputs || !node->attrs) return false;
    node->input_count = nin;
    node->output_count = nout;
    node->attr_count = nattr;
    for (uint32_t a = 0; a < nattr; a++)
        if (!read_attribute(rd, attrs[a], &g->nodes[k].attrs[a])) return false;
    node = &g->nodes[k];            /* (values made above may have moved nothing, but nodes stay put: no new nodes) */
    for (uint32_t i = 0; i < nin; i++) {
        node->inputs[i] = use_value(rd, ins[i]);
        if (ins[i].n && node->inputs[i] == IR_NONE) return false;
    }
    for (uint32_t o = 0; o < nout; o++) {
        if (!outs[o].n) {
            node->outputs[o] = IR_NONE;
            continue;
        }
        uint32_t v = ir_find_value(g, outs[o].p, outs[o].n);
        if (v != IR_NONE && g->values[v].kind != VALUE_NODE) v = IR_NONE;    /* (a name an input already has) */
        if (v == IR_NONE) v = ir_value(g, outs[o].p, outs[o].n);
        if (v == IR_NONE) return false;
        node->outputs[o] = v;
    }
    return true;
}

/* ------------------------------------------------------------------------- the model */

SpingalettGraph *ir_read_onnx(const uint8_t *data, size_t size, const char *dir) {
    Pb m = {data, data + size, true};
    Pb graph = {NULL, NULL, false};
    int64_t opset = 0;
    Str producer = {NULL, 0};
    uint32_t field, wire;
    while (next(&m, &field, &wire)) {
        if (field == 7 && wire == WIRE_BYTES) graph = sub(&m);
        else if (field == 2 && wire == WIRE_BYTES) producer = string(&m);
        else if (field == 8 && wire == WIRE_BYTES) {
            Pb o = sub(&m);
            Str domain = {NULL, 0};
            int64_t version = 0;
            uint32_t f2, w2;
            while (next(&o, &f2, &w2)) {
                if (f2 == 1 && w2 == WIRE_BYTES) domain = string(&o);
                else if (f2 == 2 && w2 == WIRE_VARINT) version = (int64_t)varint(&o);
                else skip(&o, w2);
            }
            if (!domain.n || str_eq(domain, "ai.onnx")) opset = version;
        } else skip(&m, wire);
    }
    if (!m.ok || !graph.ok) {
        set_error(SPINGALETT_ERR_INVALID, "graph: not an ONNX model (no graph), nor a .slett graph");
        return NULL;
    }
    SpingalettGraph *g = ir_graph_new();
    if (!g) return NULL;
    g->opset = opset;
    if (producer.p) g->producer = ir_strdup(g, producer.p, producer.n);
    Reader rd;
    memset(&rd, 0, sizeof rd);
    rd.g = g;
    rd.dir = dir;
    rd.opset = opset;
    /* the initializers first (an input that an initializer also names is a constant), then the inputs and nodes */
    Pb pass = graph;
    bool ok = true;
    while (ok && next(&pass, &field, &wire)) {
        if (field == 5 && wire == WIRE_BYTES) ok = read_tensor(&rd, sub(&pass), NULL, 0) != IR_NONE;
        else if (field == 15 && wire == WIRE_BYTES) ok = fail(&rd, "graph: sparse initializers are not supported%s%s",
                                                              NULL, NULL);
        else skip(&pass, wire);
    }
    uint32_t inputs = 0, outputs = 0;
    pass = graph;
    while (ok && next(&pass, &field, &wire)) {
        if (field == 11 && wire == WIRE_BYTES) {
            Pb info = sub(&pass), probe = info;
            /* (the name first: an initializer's is no input) */
            Str name = {NULL, 0};
            uint32_t f2, w2;
            while (next(&probe, &f2, &w2)) {
                if (f2 == 1 && w2 == WIRE_BYTES) name = string(&probe);
                else skip(&probe, w2);
            }
            if (name.p && ir_find_value(g, name.p, name.n) != IR_NONE) continue;
            uint32_t v;
            ok = read_value_info(&rd, info, &v, true);
            if (ok) {
                g->values[v].kind = VALUE_INPUT;
                uint32_t *grown = (uint32_t *)realloc(g->inputs, (inputs + 1u) * sizeof(uint32_t));
                if (!grown) { ok = false; break; }
                g->inputs = grown;
                g->inputs[inputs++] = v;
            }
        } else skip(&pass, wire);
    }
    g->input_count = inputs;
    pass = graph;
    while (ok && next(&pass, &field, &wire)) {
        if (field == 1 && wire == WIRE_BYTES) ok = read_node(&rd, sub(&pass));
        else skip(&pass, wire);
    }
    pass = graph;
    while (ok && next(&pass, &field, &wire)) {
        if (field == 12 && wire == WIRE_BYTES) {
            uint32_t v;
            ok = read_value_info(&rd, sub(&pass), &v, false);
            if (ok) {
                uint32_t *grown = (uint32_t *)realloc(g->outputs, (outputs + 1u) * sizeof(uint32_t));
                if (!grown) { ok = false; break; }
                g->outputs = grown;
                g->outputs[outputs++] = v;
            }
        } else skip(&pass, wire);
    }
    g->output_count = outputs;
    if (ok && !pass.ok) ok = fail(&rd, "graph: a damaged ONNX graph%s%s", NULL, NULL);
    if (ok && !outputs) ok = fail(&rd, "graph: the model has no outputs%s%s", NULL, NULL);
    if (!ok) {
        if (rd.error[0]) set_error(SPINGALETT_ERR_INVALID, rd.error);
        else if (spingalett_last_error_code() == SPINGALETT_OK) set_error(SPINGALETT_ERR_ALLOC, "graph: out of memory");
        ir_graph_free(g);
        return NULL;
    }
    if (!ir_finish(g)) {
        ir_graph_free(g);
        return NULL;
    }
    return g;
}
