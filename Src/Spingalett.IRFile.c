/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/*
 * Graphs of operators in `.slett` files of format 11 (docs/ModelFormat.md, **Graphs**): a header, tables of
 * strings, values, nodes, edges and attributes, and the constants' bytes at offsets of 64, so that a mapped file's
 * constants are used where they lie. The checksums cover the header and the tables, not the constants, so that a
 * graph of gigabytes opens without reading them all.
 */

#include "Spingalett.IR.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define HEADER 128u
#define VALUE_ENTRY 112u
#define NODE_ENTRY 32u
#define ATTR_ENTRY 24u
#define CONST_ALIGN 64u

/* ------------------------------------------------------------------------- reading */

typedef struct {
    const uint8_t *data;
    size_t size;
    const char *strings;
    uint64_t strings_size;
} Image;

static SpingalettGraph *refuse(SpingalettGraph *g, const char *why) {
    char message[200];
    snprintf(message, sizeof message, "graph: a file of format 11 that is not valid (%s)", why);
    set_error(SPINGALETT_ERR_INVALID, message);
    ir_graph_free(g);
    return NULL;
}

/* The string at an offset of the strings section, NULL when it is out of the section. */
static const char *string_at(const Image *im, uint32_t offset) {
    return offset < im->strings_size ? im->strings + offset : NULL;
}

/* A table of count entries of size bytes at an offset: within the file, after the header. */
static bool table_fits(const Image *im, uint64_t offset, uint64_t count, uint64_t size) {
    return offset >= HEADER && offset % 8 == 0 && count <= (im->size - HEADER) / (size ? size : 1) &&
           offset <= im->size && count * size <= im->size - offset;
}

SpingalettGraph *ir_read_slett(const uint8_t *data, size_t size) {
    SpingalettGraph *g = ir_graph_new();
    if (!g) return NULL;
    if (size < HEADER || memcmp(data, "SLETTM", 6)) return refuse(g, "no header");
    if (slett_get16(data + 6) != 11) return refuse(g, "a version other than 11");
    const uint64_t file_size = slett_get64(data + 24);
    if (file_size > size || file_size < HEADER) return refuse(g, "a file shorter than its size");
    if (spingalett_crc32(0, data, 124) != slett_get32(data + 124)) return refuse(g, "the header's checksum");
    Image im = {data, (size_t)file_size, NULL, 0};
    const uint32_t nv = slett_get32(data + 8), nn = slett_get32(data + 12), ni = slett_get32(data + 16);
    const uint32_t no = slett_get32(data + 20), ns = slett_get32(data + 32), na = slett_get32(data + 36);
    const uint64_t strings = slett_get64(data + 48), values = slett_get64(data + 64), nodes = slett_get64(data + 72);
    const uint64_t edges = slett_get64(data + 80), attrs = slett_get64(data + 88), numbers = slett_get64(data + 96);
    const uint64_t numbers_size = slett_get64(data + 104), tables_end = slett_get64(data + 112);
    im.strings_size = slett_get64(data + 56);
    if (tables_end > file_size || tables_end < HEADER) return refuse(g, "the tables' end");
    if (spingalett_crc32(0, data + HEADER, (size_t)(tables_end - HEADER)) != slett_get32(data + 120))
        return refuse(g, "the tables' checksum");
    if (strings < HEADER || strings > tables_end || im.strings_size > tables_end - strings || !im.strings_size ||
        data[strings + im.strings_size - 1] != 0) return refuse(g, "the strings");
    im.strings = (const char *)data + strings;
    if (!table_fits(&im, values, nv, VALUE_ENTRY) || !table_fits(&im, nodes, nn, NODE_ENTRY) ||
        !table_fits(&im, attrs, na, ATTR_ENTRY) || !table_fits(&im, numbers, numbers_size, 1))
        return refuse(g, "a table out of the file");
    g->opset = slett_get32(data + 40);
    const char *producer = string_at(&im, slett_get32(data + 44));
    if (producer && *producer) g->producer = ir_strdup(g, producer, strlen(producer));
    /* symbols, then values */
    uint64_t edge_count = (uint64_t)ni + no + ns;
    for (uint32_t k = 0; k < nn; k++) {
        const uint8_t *e = data + nodes + (uint64_t)k * NODE_ENTRY;
        edge_count += (uint64_t)slett_get32(e + 12) + slett_get32(e + 16);
    }
    if (!table_fits(&im, edges, edge_count, 4)) return refuse(g, "the edges");
    const uint8_t *edge = data + edges;
    const uint64_t symbols_at = edge_count - ns;
    for (uint32_t s = 0; s < ns; s++) {
        const char *name = string_at(&im, slett_get32(edge + 4 * (symbols_at + s)));
        if (!name || ir_symbol(g, name, strlen(name)) != s + 1) return refuse(g, "a symbol");
    }
    for (uint32_t v = 0; v < nv; v++) {
        const uint8_t *e = data + values + (uint64_t)v * VALUE_ENTRY;
        const char *name = string_at(&im, slett_get32(e));
        if (!name) return refuse(g, "a value's name");
        if (ir_value(g, name, strlen(name)) != v) return refuse(g, "out of memory");
        IrValue *value = &g->values[v];
        value->dtype = e[4];
        value->rank = e[5];
        value->kind = e[6];
        value->declared = (e[7] & 1) != 0;
        if (value->rank > IR_MAX_RANK || value->kind > VALUE_NODE || (value->declared && !ir_dtype_size(value->dtype)))
            return refuse(g, "a value's type, rank or kind");
        uint64_t count = 1;
        for (uint32_t d = 0; d < IR_MAX_RANK; d++) {
            value->dims[d] = (int64_t)slett_get64(e + 8 + 8 * d);
            value->sym[d] = slett_get16(e + 72 + 2 * d);
            if (value->sym[d] > ns) return refuse(g, "a value's symbol");
            if (d < value->rank && value->dims[d] < -1) return refuse(g, "a value's size");
            if (d < value->rank) count = value->dims[d] < 0 || (value->dims[d] && count > UINT64_MAX / (uint64_t)value->dims[d])
                                       ? UINT64_MAX : count * (uint64_t)value->dims[d];
        }
        if (value->kind == VALUE_CONST) {
            const uint64_t at = slett_get64(e + 88), bytes = slett_get64(e + 96);
            if (!value->declared || count == UINT64_MAX || count > UINT64_MAX / 8 || bytes != count * ir_dtype_size(value->dtype) ||
                at % CONST_ALIGN || at < tables_end || at > file_size || bytes > file_size - at)
                return refuse(g, "a constant's bytes");
            /* (in place when the image is aligned for its type, as a mapped file is; copied otherwise) */
            value->data = (void *)(data + at);
            if ((uintptr_t)value->data % 8) {
                value->data = ir_alloc(g, (size_t)bytes);
                if (!value->data) return refuse(g, "out of memory");
                memcpy(value->data, data + at, (size_t)bytes);
            }
        }
    }
    /* nodes and their attributes */
    for (uint32_t k = 0; k < nn; k++) {
        const uint8_t *e = data + nodes + (uint64_t)k * NODE_ENTRY;
        const uint16_t op = slett_get16(e);
        const char *name = string_at(&im, slett_get32(e + 4));
        const uint64_t first = slett_get32(e + 8), nin = slett_get32(e + 12), nout = slett_get32(e + 16);
        const uint64_t first_attr = slett_get32(e + 20), nattr = slett_get32(e + 24);
        if (op >= IR_OP_COUNT || !name || first + nin + nout > symbols_at - ni - no || first_attr + nattr > na)
            return refuse(g, "a node");
        if (ir_node(g, (IrOp)op) != k) return refuse(g, "out of memory");
        IrNode *node = &g->nodes[k];
        node->name = ir_strdup(g, name, strlen(name));
        node->inputs = (uint32_t *)ir_alloc(g, (size_t)(nin ? nin : 1u) * sizeof(uint32_t));
        node->outputs = (uint32_t *)ir_alloc(g, (size_t)(nout ? nout : 1u) * sizeof(uint32_t));
        node->attrs = (IrAttr *)ir_alloc(g, (size_t)(nattr ? nattr : 1u) * sizeof(IrAttr));
        if (!node->name || !node->inputs || !node->outputs || !node->attrs) return refuse(g, "out of memory");
        node->input_count = (uint32_t)nin;
        node->output_count = (uint32_t)nout;
        node->attr_count = (uint32_t)nattr;
        for (uint64_t i = 0; i < nin + nout; i++) {
            const uint32_t v = slett_get32(edge + 4 * (first + i));
            if (v != IR_NONE && v >= nv) return refuse(g, "an edge");
            if (i < nin) node->inputs[i] = v;
            else node->outputs[i - nin] = v;
        }
        for (uint64_t a = 0; a < nattr; a++) {
            const uint8_t *ae = data + attrs + (first_attr + a) * ATTR_ENTRY;
            IrAttr *attr = &node->attrs[a];
            const char *aname = string_at(&im, slett_get32(ae));
            if (!aname) return refuse(g, "an attribute's name");
            attr->name = ir_strdup(g, aname, strlen(aname));
            attr->type = ae[4];
            const uint64_t word = slett_get64(ae + 8);
            attr->count = slett_get32(ae + 16);
            if (!attr->name) return refuse(g, "out of memory");
            switch (attr->type) {
                case ATTR_INT: attr->i = (int64_t)word; break;
                case ATTR_FLOAT: {
                    const uint32_t bits = (uint32_t)word;
                    memcpy(&attr->f, &bits, 4);
                    break;
                }
                case ATTR_INTS: case ATTR_FLOATS: {
                    const uint64_t width = attr->type == ATTR_INTS ? 8 : 4;
                    if (word % width || word > numbers_size || attr->count > (numbers_size - word) / width)
                        return refuse(g, "an attribute's numbers");
                    void *list = ir_alloc(g, (size_t)(attr->count ? attr->count : 1u) * width);
                    if (!list) return refuse(g, "out of memory");
                    memcpy(list, data + numbers + word, (size_t)(attr->count * width));
                    if (attr->type == ATTR_INTS) attr->ints = (int64_t *)list;
                    else attr->floats = (float *)list;
                    break;
                }
                case ATTR_STRING: {
                    const char *text = word <= UINT32_MAX ? string_at(&im, (uint32_t)word) : NULL;
                    if (!text) return refuse(g, "an attribute's string");
                    attr->s = ir_strdup(g, text, strlen(text));
                    if (!attr->s) return refuse(g, "out of memory");
                    break;
                }
                case ATTR_TENSOR:
                    if (word >= nv || g->values[word].kind != VALUE_CONST) return refuse(g, "an attribute's tensor");
                    attr->tensor = (uint32_t)word;
                    break;
                default: return refuse(g, "an attribute's type");
            }
        }
    }
    g->inputs = (uint32_t *)malloc((size_t)(ni ? ni : 1u) * sizeof(uint32_t));
    g->outputs = (uint32_t *)malloc((size_t)(no ? no : 1u) * sizeof(uint32_t));
    if (!g->inputs || !g->outputs) return refuse(g, "out of memory");
    g->input_count = ni;
    g->output_count = no;
    for (uint32_t k = 0; k < ni + no; k++) {
        const uint32_t v = slett_get32(edge + 4 * (symbols_at - ni - no + k));
        if (v >= nv || (k < ni && g->values[v].kind != VALUE_INPUT)) return refuse(g, "an input or output");
        if (k < ni) g->inputs[k] = v;
        else g->outputs[k - ni] = v;
    }
    if (!ir_finish(g)) {
        ir_graph_free(g);
        return NULL;
    }
    return g;
}

/* ------------------------------------------------------------------------- writing */

typedef struct {
    uint8_t *data;
    size_t size, cap;
} Buffer;

static bool grow(Buffer *b, size_t more) {
    if (b->size + more <= b->cap) return true;
    size_t cap = b->cap ? b->cap : 4096u;
    while (cap < b->size + more) cap *= 2;
    uint8_t *data = (uint8_t *)realloc(b->data, cap);
    if (!data) return false;
    memset(data + b->cap, 0, cap - b->cap);
    b->data = data;
    b->cap = cap;
    return true;
}

/* A string's offset in the strings section (appended; NULL as the empty string at 0). */
static bool put_string(Buffer *b, const char *s, uint32_t *offset) {
    if (!s || !*s) { *offset = 0; return true; }
    const size_t n = strlen(s) + 1;
    if (b->size + n > UINT32_MAX || !grow(b, n)) return false;
    *offset = (uint32_t)b->size;
    memcpy(b->data + b->size, s, n);
    b->size += n;
    return true;
}

/* A constant's bytes as files keep them: the source's when it was widened. */
static const void *const_bytes(const IrValue *v, uint8_t *dtype, uint64_t *bytes) {
    uint64_t count = 1;
    for (uint32_t d = 0; d < v->rank; d++) count *= (uint64_t)v->dims[d];
    *dtype = v->source ? v->source_dtype : v->dtype;
    *bytes = count * ir_dtype_size(*dtype);
    return v->source ? v->source : v->data;
}

bool ir_write_slett(const SpingalettGraph *g, const char *path) {
    Buffer strings = {0}, numbers = {0};
    bool ok = grow(&strings, 1);
    if (ok) strings.size = 1;                   /* the empty string */
    const uint32_t nv = g->value_count, nn = g->node_count;
    uint64_t edge_count = (uint64_t)g->input_count + g->output_count + g->symbol_count, na = 0;
    for (uint32_t k = 0; k < nn; k++) {
        edge_count += (uint64_t)g->nodes[k].input_count + g->nodes[k].output_count;
        na += g->nodes[k].attr_count;
    }
    /* the tables' layout: header, values, nodes, attributes, edges, strings and numbers last (their sizes known
       once the others are written) */
    const uint64_t values = HEADER, nodes = values + (uint64_t)nv * VALUE_ENTRY, attrs = nodes + (uint64_t)nn * NODE_ENTRY;
    const uint64_t edges = attrs + na * ATTR_ENTRY, edges_end = (edges + edge_count * 4 + 7) & ~(uint64_t)7;
    uint8_t *tables = (uint8_t *)calloc(1, (size_t)edges_end);
    if (!tables) ok = false;
    uint64_t constants = 0;     /* bytes of constants, from the end of the tables */
    for (uint32_t v = 0; ok && v < nv; v++) {
        const IrValue *value = &g->values[v];
        uint8_t *e = tables + values + (uint64_t)v * VALUE_ENTRY;
        uint32_t name;
        ok = put_string(&strings, value->name, &name);
        slett_put32(e, name);
        uint8_t dtype = value->dtype;
        uint64_t bytes = 0;
        if (value->kind == VALUE_CONST) const_bytes(value, &dtype, &bytes);
        e[4] = dtype;
        e[5] = value->rank;
        e[6] = value->kind;
        e[7] = value->declared ? 1 : 0;
        for (uint32_t d = 0; d < IR_MAX_RANK; d++) {
            slett_put64(e + 8 + 8 * d, (uint64_t)(d < value->rank ? value->dims[d] : -1));
            slett_put16(e + 72 + 2 * d, value->sym[d]);
        }
        if (value->kind == VALUE_CONST) {
            slett_put64(e + 88, constants);     /* (relative until the tables' end is known) */
            slett_put64(e + 96, bytes);
            constants += (bytes + CONST_ALIGN - 1) / CONST_ALIGN * CONST_ALIGN;
        }
    }
    uint64_t at_edge = 0, at_attr = 0;
    for (uint32_t k = 0; ok && k < nn; k++) {
        const IrNode *n = &g->nodes[k];
        uint8_t *e = tables + nodes + (uint64_t)k * NODE_ENTRY;
        uint32_t name;
        ok = put_string(&strings, n->name, &name);
        slett_put16(e, n->op);
        slett_put32(e + 4, name);
        slett_put32(e + 8, (uint32_t)at_edge);
        slett_put32(e + 12, n->input_count);
        slett_put32(e + 16, n->output_count);
        slett_put32(e + 20, (uint32_t)at_attr);
        slett_put32(e + 24, n->attr_count);
        for (uint32_t i = 0; i < n->input_count; i++) slett_put32(tables + edges + 4 * at_edge++, n->inputs[i]);
        for (uint32_t o = 0; o < n->output_count; o++) slett_put32(tables + edges + 4 * at_edge++, n->outputs[o]);
        for (uint32_t a = 0; ok && a < n->attr_count; a++) {
            const IrAttr *attr = &n->attrs[a];
            uint8_t *ae = tables + attrs + at_attr++ * ATTR_ENTRY;
            uint32_t aname;
            ok = put_string(&strings, attr->name, &aname);
            slett_put32(ae, aname);
            ae[4] = attr->type;
            slett_put32(ae + 16, attr->count);
            switch (attr->type) {
                case ATTR_INT: slett_put64(ae + 8, (uint64_t)attr->i); break;
                case ATTR_FLOAT: {
                    uint32_t bits;
                    memcpy(&bits, &attr->f, 4);
                    slett_put64(ae + 8, bits);
                    break;
                }
                case ATTR_INTS: case ATTR_FLOATS: {
                    const size_t width = attr->type == ATTR_INTS ? 8 : 4, len = (size_t)attr->count * width;
                    numbers.size = (numbers.size + 7) & ~(size_t)7;
                    ok = grow(&numbers, len + 8);
                    if (!ok) break;
                    slett_put64(ae + 8, numbers.size);
                    if (len) memcpy(numbers.data + numbers.size, attr->type == ATTR_INTS ? (const void *)attr->ints
                                                                                        : (const void *)attr->floats, len);
                    numbers.size += len;
                    break;
                }
                case ATTR_STRING: {
                    uint32_t text;
                    ok = put_string(&strings, attr->s, &text);
                    slett_put64(ae + 8, text);
                    break;
                }
                default: slett_put64(ae + 8, attr->tensor); break;
            }
        }
    }
    for (uint32_t k = 0; ok && k < g->input_count; k++) slett_put32(tables + edges + 4 * at_edge++, g->inputs[k]);
    for (uint32_t k = 0; ok && k < g->output_count; k++) slett_put32(tables + edges + 4 * at_edge++, g->outputs[k]);
    for (uint32_t s = 0; ok && s < g->symbol_count; s++) {
        uint32_t name;
        ok = put_string(&strings, g->symbols[s], &name);
        slett_put32(tables + edges + 4 * at_edge++, name);
    }
    uint32_t producer = 0;
    if (ok) ok = put_string(&strings, "Spingalett " SPINGALETT_VERSION_STRING, &producer);
    const uint64_t strings_at = edges_end, numbers_at = (strings_at + strings.size + 7) & ~(uint64_t)7;
    const uint64_t tables_end = numbers_at + numbers.size, data_at = (tables_end + CONST_ALIGN - 1) / CONST_ALIGN * CONST_ALIGN;
    FILE *fp = NULL;
    if (ok) {
        for (uint32_t v = 0; v < nv; v++)
            if (g->values[v].kind == VALUE_CONST) {
                uint8_t *e = tables + values + (uint64_t)v * VALUE_ENTRY;
                slett_put64(e + 88, slett_get64(e + 88) + data_at);
            }
        uint8_t header[HEADER];
        memset(header, 0, sizeof header);
        memcpy(header, "SLETTM", 6);
        slett_put16(header + 6, 11);
        slett_put32(header + 8, nv);
        slett_put32(header + 12, nn);
        slett_put32(header + 16, g->input_count);
        slett_put32(header + 20, g->output_count);
        slett_put64(header + 24, data_at + constants);
        slett_put32(header + 32, g->symbol_count);
        slett_put32(header + 36, (uint32_t)na);
        slett_put32(header + 40, (uint32_t)g->opset);
        slett_put32(header + 44, producer);
        slett_put64(header + 48, strings_at);
        slett_put64(header + 56, strings.size);
        slett_put64(header + 64, values);
        slett_put64(header + 72, nodes);
        slett_put64(header + 80, edges);
        slett_put64(header + 88, attrs);
        slett_put64(header + 96, numbers_at);
        slett_put64(header + 104, numbers.size);
        slett_put64(header + 112, tables_end);
        /* the tables' checksum over what is written from byte 128 to their end, padding included */
        static const uint8_t zeros[CONST_ALIGN] = {0};
        uint32_t crc = spingalett_crc32(0, tables + HEADER, (size_t)(edges_end - HEADER));
        crc = spingalett_crc32(crc, strings.data, strings.size);
        crc = spingalett_crc32(crc, zeros, (size_t)(numbers_at - strings_at - strings.size));
        if (numbers.size) crc = spingalett_crc32(crc, numbers.data, numbers.size);
        slett_put32(header + 120, crc);
        slett_put32(header + 124, spingalett_crc32(0, header, 124));
        fp = fopen(path, "wb");
        if (!fp) {
            set_error(SPINGALETT_ERR_FILE_IO, "graph: cannot open the file for writing");
            ok = false;
        } else {
            ok = fwrite(header, 1, HEADER, fp) == HEADER &&
                 fwrite(tables + HEADER, 1, (size_t)(edges_end - HEADER), fp) == edges_end - HEADER &&
                 fwrite(strings.data, 1, strings.size, fp) == strings.size &&
                 fwrite(zeros, 1, (size_t)(numbers_at - strings_at - strings.size), fp) == numbers_at - strings_at - strings.size &&
                 (!numbers.size || fwrite(numbers.data, 1, numbers.size, fp) == numbers.size) &&
                 fwrite(zeros, 1, (size_t)(data_at - tables_end), fp) == data_at - tables_end;
            for (uint32_t v = 0; ok && v < nv; v++) {
                if (g->values[v].kind != VALUE_CONST) continue;
                uint8_t dtype;
                uint64_t bytes;
                const void *src = const_bytes(&g->values[v], &dtype, &bytes);
                const uint64_t pad = (bytes + CONST_ALIGN - 1) / CONST_ALIGN * CONST_ALIGN - bytes;
                ok = (!bytes || fwrite(src, 1, (size_t)bytes, fp) == bytes) && fwrite(zeros, 1, (size_t)pad, fp) == pad;
            }
            if (fclose(fp) != 0) ok = false;
            if (!ok) set_error(SPINGALETT_ERR_FILE_IO, "graph: a write error (disk full?)");
        }
    } else {
        set_error(SPINGALETT_ERR_ALLOC, "graph: out of memory");
    }
    free(tables);
    free(strings.data);
    free(numbers.data);
    return ok;
}
