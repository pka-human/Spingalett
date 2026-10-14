/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/*
 * Deployment models of integer weights on the GPU (Spingalett.Gpu.h, spingalett_gpu_model_*): batched prediction
 * (spingalett_model_predict()) that gives the bits of spingalett_model_run() on the same machine, as the CPU's
 * batched prediction does, for every model whose layers the GPU computes as the CPU does:
 *
 *   - an integer layer's inputs are quantized as spingalett_quantize_activations() quantizes them (iquant.comp: the
 *     largest magnitude of the sample, its scale and reciprocal divided as IEEE divides, x times it rounded to the
 *     nearest integer, ties to even), then multiplied by the layer's codes (INT4 and INT2 unpacked to their values)
 *     on the matrix kernels, gemm.comp and gemm_mma.comp, as floats or bfloat16, both of which hold such integers
 *     exactly: a product of two is an integer of at most 127 x 127, and a sum of up to `exact_k()` of them stays
 *     below 2^24, where every float is an integer, so that the kernels' sums are the integer sums whatever the
 *     order they add in. Products over more k run in slices of that many (EPI_PARTIAL), whose sums are added as
 *     integers. Depthwise and transposed convolutions sum window by window (ipass.comp's DIRECT), in floats where
 *     the window is that short, else in integers.
 *   - the outputs are spingalett_int_output()'s: bias + (scale x_scale) acc, rounded as the CPU rounds (no
 *     multiply-add contracted), then the activation: none, ReLU or leaky ReLU, which need no libm;
 *   - pooling, global pooling, additions, products and concatenations of layers, batch normalization (with the
 *     coefficients the CPU computes) and upsampling compute each output in the order of operations the CPU's do
 *     (ipass.comp).
 *
 * The last layer's activation runs on the host where it needs libm (softmax, sigmoid, tanh, ...): on each sample's
 * outputs as batched prediction applies it. Layers that need libm inside the network (GELU, SiLU, sigmoid, tanh,
 * softmax, attention), normalizations of layers and cells (their square roots), embeddings and float layers keep
 * the model on the CPU, as `why` says.
 *
 * A chunk of up to `capacity` samples runs as one command buffer, recorded once per chunk size and staging slot and
 * kept: its inputs copied in, every layer's passes, its outputs copied out. Two slots of host-visible memory take
 * turns, so that the host copies a chunk's inputs in and the outputs of the one before out while the device runs a
 * chunk. The layers' outputs share memory as the engine's workspace does (the image's plan for graphs, two
 * alternating buffers for chains), times the chunk's samples.
 */

#include "Spingalett.Gpu.h"
#include "Spingalett.GpuKernels.h"
#include "Spingalett.Recorder.h"
#include <stdlib.h>
#include <string.h>

#define MODEL_KEPT      4u                  /* the commands of this many chunk sizes and slots are kept */
#define MODEL_CHUNK     4096u               /* samples a chunk, at most */
/* Bytes of a chunk's activations, at most, 64 samples at least: a chunk whose layers' values stay in the GPU's cache
   from one pass to the next runs faster than larger ones (as the networks' chunks, Spingalett.Gpu.c's
   INFERENCE_BYTES: on an RTX 4050 Laptop GPU the MNIST CNN of Examples/Benchmark.c in INT8 predicts 2 times as fast
   in chunks of 32 MB as of 256 MB, ResNet-20 1.7 times, MobileNet 1.7 times on CUDA, alike on Vulkan) */
#define MODEL_BYTES     (32ull << 20)
#define MAX_GROUPS      65535u

/* How a layer runs. */
enum { K_PRODUCT, K_DIRECT, K_POOL, K_BN, K_ADD, K_MUL, K_CONCAT, K_GAP, K_UP };

typedef struct {
    SlettLayer L;
    uint32_t kind;
    uint32_t act;                           /* the activation on the GPU (the host's: none) */
    bool pointwise;                         /* a product of the layer's maps as matrices (dense, 1 x 1 convolutions) */
    bool mma;                               /* its product on the matrix units */
    bool half;                              /* its quantized inputs and codes kept as bfloat16 (exact) */
    uint32_t slices, slice_k;               /* its product's slices of k */
    uint64_t w;                             /* bytes into `weights`: its codes */
    uint64_t prm;                           /* floats into `params`: biases then scales; the coefficients of
                                               batch normalization; bilinear upsampling's table */
    uint64_t geo;                           /* uints into `geo`: its convolution's geometry */
} GpuLayer;

struct SpgGpuModel {
    SpgBackend backend;
    const uint8_t *image;
    uint32_t count;                         /* layers (the image's entries) */
    GpuLayer *layers;
    bool graph;
    ActivationFunction host_act;            /* the last layer's activation where the host applies it, else none */
    uint32_t inputs, outputs, width;        /* values a sample: in, out, the widest hidden layer (chains) */
    uint32_t capacity;                      /* samples a chunk */
    uint64_t act_bytes;                     /* bytes of a sample's activations */
    SpgGpuArena *arena;
    SpgGpuBuffer weights, params, geo;
    SpgGpuBuffer act, q, xs, part, sums, output;
    SpgGpuBuffer input[2];                  /* a staging slot's chunk of inputs on the device: copied in while the
                                               chunk before runs (CUDA: the commands before the first barrier) */
    uint32_t at;                            /* the slot whose commands are being recorded */
    SpgGpuBuffer staging[2];                /* host-visible slots: a chunk's inputs, then its outputs */
    size_t out_at;
    uint32_t kept_n[MODEL_KEPT];            /* 2 n + the slot */
    SpgGpuCommands *kept[MODEL_KEPT];
    uint32_t next;
    bool lost;
};

static inline uint32_t groups_of(uint64_t items, uint32_t per) {
    uint64_t g = (items + per - 1u) / per;
    return g > MAX_GROUPS ? MAX_GROUPS : (uint32_t)g;
}

static inline uint64_t align16(uint64_t n) { return (n + 15u) & ~(uint64_t)15u; }

/* Whether every address is 16 bytes aligned (with values in fours: ipass.comp's VEC4). */
static inline bool fours(uint64_t a, uint64_t b, uint64_t c, uint64_t d) { return ((a | b | c | d) & 15u) == 0; }

/* The most k a slice of an integer product sums: every partial sum of products of activations (|q| <= 127) and
   codes (127, 8 and 2 at most) below 2^24, where floats are exact integers (a multiple of 8). */
static uint32_t exact_k(PrecisionMode precision) {
    const uint32_t code = precision == PRECISION_INT8 ? 127u : precision == PRECISION_INT4 ? 8u : 2u;
    return ((1u << 24) / (127u * code)) & ~7u;
}

/* Whether the GPU runs the activation as the CPU does (no libm). */
static bool exact_act(ActivationFunction act) {
    return act == ACT_NONE || act == ACT_RELU || act == ACT_LEAKY_RELU;
}

/* How layer L runs, or why the GPU cannot run it as the CPU does (NULL: it can). */
static const char *plan_layer(const SlettLayer *L, bool last, GpuLayer *g) {
    g->L = *L;
    g->pointwise = false;
    const bool ints = spingalett_precision_is_int(L->precision);
    if (L->tied) return "a layer tied to an embedding";
    switch (L->type) {
        case LAYER_DENSE:
            if (!ints) return "a layer of float weights";
            g->kind = K_PRODUCT;
            g->pointwise = true;
            break;
        case LAYER_CONV2D:
            if (!ints) return "a layer of float weights";
            if (L->kernel_h > 255u || L->kernel_w > 255u || L->stride_h * L->stride_w > SPG_MAX_PHASES)
                return "a convolution window over 255 cells or strides that multiply to over 64";
            g->kind = slett_conv_depthwise(L) ? K_DIRECT : K_PRODUCT;
            g->pointwise = slett_conv_pointwise(L) && L->groups <= 1;
            break;
        case LAYER_CONV_TRANSPOSE2D:
            if (!ints) return "a layer of float weights";
            g->kind = K_DIRECT;
            break;
        case LAYER_MAX_POOL2D: case LAYER_AVG_POOL2D: g->kind = K_POOL; break;
        case LAYER_BATCH_NORM: g->kind = K_BN; break;
        case LAYER_ADD: g->kind = K_ADD; break;
        case LAYER_MULTIPLY: g->kind = K_MUL; break;
        case LAYER_CONCAT: g->kind = K_CONCAT; break;
        case LAYER_GLOBAL_AVG_POOL: g->kind = K_GAP; break;
        case LAYER_UPSAMPLE: g->kind = K_UP; break;
        default:
            return "a layer the GPU does not compute as the CPU does (normalization of cells, embedding, attention)";
    }
    if (!last && !exact_act(L->activation)) return "an activation inside the network that needs libm";
    g->act = exact_act(L->activation) ? L->activation : ACT_NONE;
    return NULL;
}

/* ---- memory ---- */

/* The address of the outputs of layer j (0: the input) in a chunk. */
static uint64_t output_of(const SpgGpuModel *m, uint32_t j) {
    if (j == 0) return m->input[m->at].address;
    if (j == m->count) return m->output.address;
    if (m->graph) return m->act.address + spingalett_slett_act_offset(m->image, j) * m->capacity;
    return m->act.address + ((j - 1u) & 1u) * (uint64_t)m->capacity * m->width * 4u;
}

/* The address of input k of layer entry i. */
static uint64_t input_of(const SpgGpuModel *m, uint32_t i, uint32_t k) {
    return output_of(m, spingalett_slett_input(m->image, &m->layers[i].L, k));
}

/* Values a sample of the outputs of layer j. */
static uint32_t units_of(const SpgGpuModel *m, uint32_t j) {
    uint32_t h, w, units;
    spingalett_slett_output_shape(m->image, j, &h, &w, &units);
    return units;
}

/* ---- recording ---- */

/* An integer layer's inputs x quantized into q, their scales into xs (iquant.comp): a sample in slices of
   SPG_IQ_SLICE values, a workgroup each, their largest magnitudes found first where there are several. */
static void quantize(SpgGpuModel *m, Recorder *r, const GpuLayer *g, uint64_t x, uint32_t n) {
    const bool half = g->half;
    const uint32_t len = g->L.inputs, slices = (len + SPG_IQ_SLICE - 1u) / SPG_IQ_SLICE;
    SpgIquantPush p = {x, m->q.address, m->xs.address, m->part.address, n, len, len, len, slices, SPG_IQ_SLICE,
                       len % 4u == 0 && x % 16u == 0 ? SPG_IQ_VEC4 : 0u, 0};
    const SpgKernel k = half ? SPG_KERNEL_iquant_h : SPG_KERNEL_iquant;
    const Range part = {m->part.address, m->part.address + 4ull * n * slices};
    if (slices > 1u) {
        uint32_t spec[2] = {SPG_IQ_MAX, half ? 2u : 0u};
        Access a = {0};
        reads(&a, span(x, (uint64_t)n * len));
        writes(&a, part);
        need(r, &a);
        spg_gpu_dispatch(r->c, k, spec, 2, &p, sizeof p, groups_of((uint64_t)n * slices, 1u), 1, 1);
    }
    uint32_t spec[2] = {slices > 1u ? SPG_IQ_QUANT : SPG_IQ_WHOLE, half ? 2u : 0u};
    Access a = {0};
    reads(&a, span(x, (uint64_t)n * len));
    if (slices > 1u) reads(&a, part);
    writes(&a, (Range){m->q.address, m->q.address + (uint64_t)n * len * (half ? 2u : 4u)});
    writes(&a, span(m->xs.address, n));
    need(r, &a);
    spg_gpu_dispatch(r->c, k, spec, 2, &p, sizeof p, groups_of((uint64_t)n * slices, 1u), 1, 1);
}

static void pass(Recorder *r, const Access *a, uint32_t op, uint32_t act, uint32_t half, const SpgIpassPush *p) {
    uint32_t spec[3] = {op, act, half};
    need(r, a);
    const uint32_t items = p->flags & (SPG_IP_VEC4 | SPG_IP_DW4 | SPG_IP_OG4) ? p->total / 4u : p->total;
    spg_gpu_dispatch(r->c, half ? SPG_KERNEL_ipass_h : SPG_KERNEL_ipass, spec, 3, p, sizeof *p, groups_of(items, 256u), 1,
                     1);
}

/* An integer layer's product on the matrix kernels: its quantized inputs times its codes, the integer sums into y
   (slices 1) or `sums`, then its outputs from them (ipass.comp's EPI). */
static void product(SpgGpuModel *m, Recorder *r, const GpuLayer *g, uint64_t y, uint32_t n) {
    const SlettLayer *L = &g->L;
    const bool dense = L->type == LAYER_DENSE, half = g->half;
    const uint32_t G = dense ? 1u : L->groups, OC = L->rows, OG = OC / G, K = L->row_len;
    const uint32_t pixels = dense ? 1u : L->out_h * L->out_w, M = n * pixels, S = g->slices;
    const uint64_t weights = m->weights.address + g->w, outputs = (uint64_t)M * OC;
    const uint64_t c = S == 1u ? y : m->sums.address;
    SpgGemmPush p = {.a = m->q.address, .b = weights, .c = c, .M = M, .N = OG, .K = K, .ldb = K, .ldc = OC,
                     .slices = S, .slice_k = g->slice_k, .alpha = 1.0f, .beta = 0.0f};
    SpgGemmMode mode = {g->pointwise ? SPG_A_ROW : SPG_A_CONV, SPG_B_COL, S == 1u ? SPG_EPI_STORE : SPG_EPI_PARTIAL,
                        ACT_NONE, G, false, true, true, 0, outputs, g->mma, (half ? 1u : 0u) | (half ? 2u : 0u),
                        true, true};
    if (g->pointwise) {
        p.lda = dense ? K : L->in_c;
    } else {
        const uint32_t CG = L->in_c / G;
        p.geo = m->geo.address + 4u * g->geo;
        p.a_group = CG;
        p.b_group = OG * K;
        p.c_group = OG;
        mode.vec_a = CG % 4u == 0 && L->in_c % 4u == 0;
        mode.wide_a = CG % 8u == 0 && L->in_c % 8u == 0;
    }
    Access a = {0};
    reads(&a, (Range){m->q.address, m->q.address + (uint64_t)n * L->inputs * (half ? 2u : 4u)});
    reads(&a, (Range){weights, weights + (uint64_t)OC * K * (half ? 2u : 4u)});
    if (!g->pointwise) reads(&a, whole(&m->geo));
    writes(&a, span(c, S == 1u ? outputs : outputs * S));
    need(r, &a);
    spg_gemm(r->c, &p, &mode);
    SpgIpassPush e = {.x = c, .y = y, .w = m->xs.address, .e0 = m->params.address + 4u * g->prm,
                      .e1 = m->params.address + 4u * (g->prm + OC), .e2 = c, .total = (uint32_t)outputs, .n = pixels,
                      .out_c = OC, .groups = G, .slices = S};
    if (S == 1u && OC % 4u == 0 && fours(e.x, e.y, e.e0, e.e1)) e.flags = SPG_IP_VEC4;
    Access b = {0};
    reads(&b, span(c, S == 1u ? outputs : outputs * S));
    reads(&b, span(m->xs.address, n));
    reads(&b, span(e.e0, 2ull * OC));
    writes(&b, span(y, outputs));
    pass(r, &b, SPG_IP_EPI, g->act, 0, &e);
}

static void record_layer(SpgGpuModel *m, Recorder *r, uint32_t i, uint32_t n) {
    const GpuLayer *g = &m->layers[i];
    const SlettLayer *L = &g->L;
    const uint64_t y = output_of(m, i + 1u), x = input_of(m, i, 0);
    const uint64_t total = (uint64_t)n * L->outputs;
    SpgIpassPush p = {.x = x, .y = y, .w = x, .e0 = x, .e1 = x, .e2 = x, .total = (uint32_t)total, .n = n,
                      .in_h = L->in_h, .in_w = L->in_w, .in_c = L->in_c, .out_h = L->out_h, .out_w = L->out_w,
                      .out_c = L->out_c, .kh = L->kernel_h, .kw = L->kernel_w, .sh = L->stride_h, .sw = L->stride_w,
                      .ph = L->pad_h, .pw = L->pad_w, .groups = L->groups ? L->groups : 1u, .slices = 1};
    Access a = {0};
    switch (g->kind) {
        case K_PRODUCT:
            quantize(m, r, g, x, n);
            product(m, r, g, y, n);
            return;
        case K_DIRECT: {
            quantize(m, r, g, x, n);
            const bool half = g->half;
            p.x = m->q.address;
            p.w = m->weights.address + g->w;
            p.e0 = m->params.address + 4u * g->prm;
            p.e1 = m->params.address + 4u * (g->prm + L->rows);
            p.e2 = m->xs.address;
            /* (sums in floats where they stay exact: windows of exact_k() values at most) */
            p.flags = (L->type == LAYER_CONV_TRANSPOSE2D ? SPG_IP_TRANSPOSED : 0u) |
                      (L->row_len <= exact_k(L->precision) ? SPG_IP_FLOATS : 0u);
            /* four filters a thread: a depthwise convolution's (four channels) or a group's, in fours, aligned */
            const uint32_t G = L->groups ? L->groups : 1u, OG = L->rows / G, CG = L->in_c / G;
            if ((p.flags & SPG_IP_FLOATS) && L->rows % 4u == 0 && fours(y, p.e0, p.e1, 0)) {
                if (OG == 1u && CG == 1u) p.flags |= SPG_IP_DW4;
                else if (OG % 4u == 0) p.flags |= SPG_IP_OG4;
            }
            reads(&a, (Range){p.x, p.x + (uint64_t)n * L->inputs * (half ? 2u : 4u)});
            reads(&a, (Range){p.w, p.w + (uint64_t)L->rows * L->row_len * (half ? 2u : 4u)});
            reads(&a, span(p.e0, 2ull * L->rows));
            reads(&a, span(p.e2, n));
            writes(&a, span(y, total));
            pass(r, &a, SPG_IP_DIRECT, g->act, half ? 3u : 0u, &p);     /* (q and the codes) */
            return;
        }
        case K_POOL:
            p.flags = L->type == LAYER_MAX_POOL2D ? SPG_IP_MAX : 0u;
            reads(&a, span(x, (uint64_t)n * L->inputs));
            writes(&a, span(y, total));
            pass(r, &a, SPG_IP_POOL, g->act, 0, &p);
            return;
        case K_BN:
            p.e0 = m->params.address + 4u * g->prm;
            p.e1 = p.e0 + 4ull * L->out_c;
            if (L->out_c % 4u == 0 && fours(x, y, p.e0, p.e1)) p.flags = SPG_IP_VEC4;
            reads(&a, span(x, total));
            reads(&a, span(p.e0, 2ull * L->out_c));
            writes(&a, span(y, total));
            pass(r, &a, SPG_IP_BN, g->act, 0, &p);
            return;
        case K_GAP:
            p.total = n * L->in_c;
            p.inv = 1.0f / (float)(L->in_h * L->in_w);      /* (as the engine's spingalett_engine_global_pool()) */
            reads(&a, span(x, (uint64_t)n * L->inputs));
            writes(&a, span(y, (uint64_t)n * L->in_c));
            pass(r, &a, SPG_IP_GAP, g->act, 0, &p);
            return;
        case K_UP:
            p.flags = L->mode == UPSAMPLE_BILINEAR ? SPG_IP_BILINEAR : 0u;
            if (p.flags) {
                p.e0 = m->params.address + 4u * g->prm;
                reads(&a, span(p.e0, 4ull * (L->out_h + L->out_w)));
            }
            p.sh = L->stride_h;
            p.sw = L->stride_w;
            reads(&a, span(x, (uint64_t)n * L->inputs));
            writes(&a, span(y, total));
            pass(r, &a, SPG_IP_UP, g->act, 0, &p);
            return;
        case K_CONCAT: {
            /* each input's channels into their place: dispatches that write apart, without barriers between them */
            const uint32_t cells = L->out_h * L->out_w;
            for (uint32_t k = 0, c0 = 0; k < L->input_count; k++) {
                const uint32_t j = spingalett_slett_input(m->image, L, k), ck = units_of(m, j) / cells;
                SpgIpassPush s = p;
                s.x = output_of(m, j);
                s.total = n * cells * ck;
                s.in_c = ck;
                s.c0 = c0;
                if (ck % 4u == 0 && L->out_c % 4u == 0 && c0 % 4u == 0 && fours(s.x, y, 0, 0)) s.flags = SPG_IP_VEC4;
                Access b = {0};
                reads(&b, span(s.x, (uint64_t)s.total));
                writes(&b, span(y, total));
                b.group = 1u + i;
                pass(r, &b, SPG_IP_SLICE, g->act, 0, &s);
                c0 += ck;
            }
            return;
        }
        default: {                          /* K_ADD, K_MUL: the inputs in their order */
            if (L->input_count == 1u) {     /* (a copy, with the activation) */
                p.in_c = p.out_c = L->outputs;
                if (L->outputs % 4u == 0 && fours(x, y, 0, 0)) p.flags = SPG_IP_VEC4;
                reads(&a, span(x, total));
                writes(&a, span(y, total));
                pass(r, &a, SPG_IP_SLICE, g->act, 0, &p);
                return;
            }
            for (uint32_t k = 1; k < L->input_count; k++) {
                SpgIpassPush s = p;
                s.x = k == 1u ? x : input_of(m, i, k);
                s.e0 = input_of(m, i, 1);
                s.flags = (k == 1u ? SPG_IP_FIRST : 0u) | (k + 1u == L->input_count ? SPG_IP_LAST : 0u) |
                          (total % 4u == 0 && fours(s.x, s.e0, y, 0) ? SPG_IP_VEC4 : 0u);
                Access b = {0};
                reads(&b, span(s.x, total));
                reads(&b, span(k == 1u ? s.e0 : y, total));
                writes(&b, span(y, total));
                pass(r, &b, g->kind == K_ADD ? SPG_IP_ADD : SPG_IP_MUL, g->act, 0, &s);
            }
            return;
        }
    }
}

/* The commands of a chunk of n samples in staging slot `at`: kept ones, or recorded now in place of the oldest. Their
   inputs are copied to the slot's own buffer, which the chunk before does not read; the rest waits for that chunk
   (whose other buffers it reuses). */
static SpgGpuCommands *chunk_commands(SpgGpuModel *m, uint32_t n, uint32_t at) {
    for (uint32_t k = 0; k < MODEL_KEPT; k++)
        if (m->kept[k] && m->kept_n[k] == 2u * n + at) return m->kept[k];
    const uint32_t slot = m->next;
    m->next = (m->next + 1u) % MODEL_KEPT;
    spg_gpu_commands_free(m->kept[slot]);
    m->kept[slot] = NULL;
    SpgGpuCommands *c = spg_gpu_commands_create();
    Recorder *r = c && spg_gpu_record_begin(c) ? (Recorder *)calloc(1, sizeof *r) : NULL;
    if (!r) {
        spg_gpu_commands_free(c);
        return NULL;
    }
    r->c = c;
    m->at = at;
    spg_gpu_copy(c, &m->staging[at], 0, &m->input[at], 0, 4ull * n * m->inputs);
    barrier(r);
    for (uint32_t i = 0; i < m->count; i++) record_layer(m, r, i, n);
    barrier(r);
    spg_gpu_copy(c, &m->output, 0, &m->staging[at], m->out_at, 4ull * n * m->outputs);
    spg_gpu_barrier_host(c);
    free(r);
    if (!spg_gpu_record_end(c)) {
        spg_gpu_commands_free(c);
        return NULL;
    }
    m->kept[slot] = c;
    m->kept_n[slot] = 2u * n + at;
    return c;
}

/* ---- the model on the device ---- */

static void spingalett_gpu_model_free_here(SpgGpuModel *m) {
    if (!m) return;
    for (uint32_t k = 0; k < MODEL_KEPT; k++) spg_gpu_commands_free(m->kept[k]);
    spg_gpu_buffer_free(&m->staging[0]);
    spg_gpu_buffer_free(&m->staging[1]);
    spg_gpu_arena_free(m->arena);
    free(m->layers);
    free(m);
}

void spingalett_gpu_model_free(SpgGpuModel *m) {
    if (!m) return;
    const SpgBackend saved = spg_gpu_using();
    spg_gpu_use(m->backend);
    spingalett_gpu_model_free_here(m);
    spg_gpu_use(saved);
}

/* The codes of layer L as floats or bfloat16 (half) at dst: rows of row_len values, or transposed (k a row of every
   filter's codes, as DIRECT reads them). */
static void put_codes(const uint8_t *image, const SlettLayer *L, bool half, bool transposed, void *dst, int8_t *row) {
    const size_t stride = (size_t)spingalett_slett_row_bytes(L->precision, L->row_len);
    for (uint32_t j = 0; j < L->rows; j++) {
        const uint8_t *src = image + L->weights + j * stride;
        if (L->precision == PRECISION_INT8) memcpy(row, src, L->row_len);
        else if (L->precision == PRECISION_INT4) spingalett_unpack_int4(src, row, L->row_len);
        else spingalett_unpack_int2(src, row, L->row_len);
        for (uint32_t k = 0; k < L->row_len; k++) {
            const size_t at = transposed ? (size_t)k * L->rows + j : (size_t)j * L->row_len + k;
            if (half) ((uint16_t *)dst)[at] = spingalett_float_to_bf16((float)row[k]);
            else ((float *)dst)[at] = (float)row[k];
        }
    }
}

SpgGpuModel *spingalett_gpu_model_new(const SpingalettModel *model, const char **why) {
    const char *reason = NULL;
    if (!why) why = &reason;
    *why = "no GPU";
    if (!model || !model->image || !spg_gpu_open()) return NULL;
    const uint8_t *image = (const uint8_t *)model->image;
    SpgGpuModel *m = (SpgGpuModel *)calloc(1, sizeof *m);
    if (!m) return *why = "out of memory", NULL;
    m->backend = spg_gpu_using();
    m->image = image;
    m->count = model->layer_count;
    m->graph = slett_get16(image + 6) >= 6u;
    m->inputs = model->input_size;
    m->outputs = model->output_size;
    m->width = model->max_width_ ? model->max_width_ : 1u;
    m->layers = (GpuLayer *)calloc(m->count, sizeof *m->layers);
    if (!m->layers) {
        free(m);
        return *why = "out of memory", NULL;
    }
    /* the layers: how each runs, what of the device it takes */
    const bool mma = spg_gpu_mma_bf16();
    bool ints = false;
    uint64_t weight_bytes = 0, param_floats = 0, geo_uints = 0, q_values = 1, sum_floats = 0, widest = m->inputs;
    uint32_t longest = 1;                   /* the longest row of codes */
    for (uint32_t i = 0; i < m->count; i++) {
        GpuLayer *g = &m->layers[i];
        SlettLayer L;
        spingalett_slett_layer(image, i, &L);
        if ((*why = plan_layer(&L, i + 1u == m->count, g)) != NULL) {
            spingalett_gpu_model_free_here(m);
            return NULL;
        }
        if (L.outputs > widest) widest = L.outputs;
        if (i + 1u == m->count) m->host_act = exact_act(L.activation) ? ACT_NONE : L.activation;
        if (g->kind == K_PRODUCT || g->kind == K_DIRECT) {
            ints = true;
            if (L.row_len > longest) longest = L.row_len;
            const uint32_t pixels = L.type == LAYER_DENSE ? 1u : L.out_h * L.out_w;
            /* (the matrix units take products of 16 rows, columns and 32 k at least, as the networks' do) */
            g->mma = g->kind == K_PRODUCT && mma && L.row_len >= 32u && L.rows / (L.groups ? L.groups : 1u) >= 16u;
            g->half = (g->mma || g->kind == K_DIRECT) && spg_gpu_bf16_storage();
            g->slice_k = exact_k(L.precision);
            g->slices = (L.row_len + g->slice_k - 1u) / g->slice_k;
            if (g->slices == 1u) g->slice_k = L.row_len;
            g->w = weight_bytes;
            weight_bytes += align16((uint64_t)L.rows * L.row_len * (g->half ? 2u : 4u));
            g->prm = param_floats;
            param_floats += (2ull * L.rows + 3u) & ~3ull;
            if (L.inputs > q_values) q_values = L.inputs;
            if (g->slices > 1u && (uint64_t)g->slices * pixels * L.rows > sum_floats)
                sum_floats = (uint64_t)g->slices * pixels * L.rows;
            if (g->kind == K_PRODUCT && !g->pointwise) {
                SpgConvGeometry info;
                if (!spg_conv_geometry(NULL, &info, L.in_h, L.in_w, L.in_c, L.out_h, L.out_w, L.out_c, L.groups,
                                       L.kernel_h, L.kernel_w, L.stride_h, L.stride_w, L.pad_h, L.pad_w)) {
                    *why = "a convolution whose geometry the matrix kernels do not take";
                    spingalett_gpu_model_free_here(m);
                    return NULL;
                }
                g->geo = geo_uints;
                geo_uints += (info.size + 3u) & ~(size_t)3u;
            }
        } else if (g->kind == K_BN) {
            g->prm = param_floats;
            param_floats += (2ull * L.out_c + 3u) & ~3ull;
        } else if (g->kind == K_UP && L.mode == UPSAMPLE_BILINEAR) {
            g->prm = param_floats;
            param_floats += 4ull * (L.out_h + L.out_w);
        }
    }
    if (!ints) {
        spingalett_gpu_model_free_here(m);
        return *why = "a model of float weights", NULL;
    }
    /* a chunk: its samples' activations within MODEL_BYTES (64 samples at least, in a quarter of the device's memory
       at most), its layers' values indexed in 32 bits */
    m->act_bytes = m->graph ? (model->activations_ ? model->activations_ : 4u) : 8ull * m->width;
    const uint64_t per = m->act_bytes + 4ull * (q_values + sum_floats + 2u * m->inputs + m->outputs + 1u);
    uint64_t cap = MODEL_BYTES / per;
    if (cap < 64u) cap = 64u;
    if (cap > spg_gpu_memory() / 4u / per) cap = spg_gpu_memory() / 4u / per;
    if (cap > MODEL_CHUNK) cap = MODEL_CHUNK;
    if (cap > (uint64_t)INT32_MAX / widest) cap = (uint64_t)INT32_MAX / widest;
    if (sum_floats && cap > (uint64_t)INT32_MAX / sum_floats) cap = (uint64_t)INT32_MAX / sum_floats;
    m->capacity = cap ? (uint32_t)cap : 1u;
    if ((uint64_t)m->capacity * widest > INT32_MAX) {
        spingalett_gpu_model_free_here(m);
        return *why = "layers of over 2^31 values a sample", NULL;
    }
    m->arena = spg_gpu_arena_create();
    if (!m->arena) {
        spingalett_gpu_model_free_here(m);
        return *why = "out of memory", NULL;
    }
    const uint64_t C = m->capacity;
    spg_gpu_arena_add(m->arena, &m->weights, weight_bytes, SPG_MEMORY_DEVICE);
    spg_gpu_arena_add(m->arena, &m->params, 4u * (param_floats ? param_floats : 4u), SPG_MEMORY_DEVICE);
    spg_gpu_arena_add(m->arena, &m->geo, 4u * (geo_uints ? geo_uints : 4u), SPG_MEMORY_DEVICE);
    spg_gpu_arena_add(m->arena, &m->act, align16(m->act_bytes * C), SPG_MEMORY_DEVICE);
    spg_gpu_arena_add(m->arena, &m->q, align16(4u * q_values * C), SPG_MEMORY_DEVICE);
    spg_gpu_arena_add(m->arena, &m->xs, align16(4u * C), SPG_MEMORY_DEVICE);
    spg_gpu_arena_add(m->arena, &m->part, align16(4u * C * ((q_values + SPG_IQ_SLICE - 1u) / SPG_IQ_SLICE)),
                      SPG_MEMORY_DEVICE);
    spg_gpu_arena_add(m->arena, &m->sums, align16(4u * (sum_floats ? sum_floats * C : 4u)), SPG_MEMORY_DEVICE);
    spg_gpu_arena_add(m->arena, &m->input[0], align16(4ull * m->inputs * C), SPG_MEMORY_DEVICE);
    spg_gpu_arena_add(m->arena, &m->input[1], align16(4ull * m->inputs * C), SPG_MEMORY_DEVICE);
    spg_gpu_arena_add(m->arena, &m->output, align16(4ull * m->outputs * C), SPG_MEMORY_DEVICE);
    m->out_at = (size_t)align16(4ull * m->inputs * C);
    const size_t upload = (size_t)(weight_bytes + 4u * param_floats + 4u * geo_uints);
    const size_t slot = m->out_at + 4ull * m->outputs * C;
    if (!spg_gpu_arena_commit(m->arena) || !spg_gpu_buffer_create(&m->staging[0], slot > upload ? slot : upload, true) ||
        !spg_gpu_buffer_create(&m->staging[1], slot, true)) {
        spingalett_gpu_model_free_here(m);
        return *why = "out of device memory", NULL;
    }
    /* the weights, parameters and geometries: written to the staging memory, copied to the device once */
    uint8_t *host = (uint8_t *)m->staging[0].mapped;
    float *params = (float *)(void *)(host + weight_bytes);
    uint32_t *geo = (uint32_t *)(void *)(host + weight_bytes + 4u * param_floats);
    int8_t *row = (int8_t *)malloc(longest);
    bool ok = row != NULL;
    for (uint32_t i = 0; ok && i < m->count; i++) {
        const GpuLayer *g = &m->layers[i];
        const SlettLayer *L = &g->L;
        if (g->kind == K_PRODUCT || g->kind == K_DIRECT) {
            put_codes(image, L, g->half, g->kind == K_DIRECT, host + g->w, row);
            memcpy(params + g->prm, image + L->biases, 4u * (size_t)L->rows);
            memcpy(params + g->prm + L->rows, image + L->scales, 4u * (size_t)L->rows);
            if (g->kind == K_PRODUCT && !g->pointwise) {
                SpgConvGeometry info;
                spg_conv_geometry(geo + g->geo, &info, L->in_h, L->in_w, L->in_c, L->out_h, L->out_w, L->out_c,
                                  L->groups, L->kernel_h, L->kernel_w, L->stride_h, L->stride_w, L->pad_h, L->pad_w);
            }
        } else if (g->kind == K_BN) {
            /* the coefficients as batched prediction and the engine compute them */
            const uint32_t Cn = L->out_c;
            const float *stats = (const float *)(const void *)(image + L->scales);
            spingalett_bn_coefficients((const float *)(const void *)(image + L->weights),
                                       (const float *)(const void *)(image + L->biases), stats, stats + Cn, L->eps, Cn,
                                       params + g->prm, params + g->prm + Cn);
        } else if (g->kind == K_UP && L->mode == UPSAMPLE_BILINEAR) {
            /* the rows' and columns' sources and weights, as spingalett_engine_upsample() takes them */
            float *t = params + g->prm;
            for (uint32_t o = 0; o < L->out_h + L->out_w; o++) {
                const bool rows = o < L->out_h;
                uint32_t r0, r1;
                float l;
                spingalett_engine_bilinear(rows ? o : o - L->out_h, rows ? L->stride_h : L->stride_w,
                                           rows ? L->in_h : L->in_w, &r0, &r1, &l);
                const float h = 1.0f - l;
                memcpy(t + 4u * o, &r0, 4);
                memcpy(t + 4u * o + 1u, &r1, 4);
                t[4u * o + 2u] = l;
                t[4u * o + 3u] = h;
            }
        }
    }
    free(row);
    SpgGpuCommands *c = ok ? spg_gpu_commands_create() : NULL;
    ok = c && spg_gpu_record_begin(c);
    if (ok) {
        spg_gpu_copy(c, &m->staging[0], 0, &m->weights, 0, weight_bytes);
        if (param_floats) spg_gpu_copy(c, &m->staging[0], weight_bytes, &m->params, 0, 4u * param_floats);
        if (geo_uints) spg_gpu_copy(c, &m->staging[0], weight_bytes + 4u * param_floats, &m->geo, 0, 4u * geo_uints);
        spg_gpu_barrier(c);
        ok = spg_gpu_record_end(c) && spg_gpu_submit(c) && spg_gpu_wait(c);
    }
    spg_gpu_commands_free(c);
    if (!ok) {
        spingalett_gpu_model_free_here(m);
        return *why = "out of memory", NULL;
    }
    *why = NULL;
    return m;
}

/* Submits chunk k (its inputs into its slot first); NULL when it cannot. */
static SpgGpuCommands *submit_chunk(SpgGpuModel *m, const float *inputs, uint32_t count, uint32_t k) {
    const uint32_t start = k * m->capacity, n = count - start < m->capacity ? count - start : m->capacity;
    memcpy(m->staging[k & 1u].mapped, inputs + (size_t)start * m->inputs, 4ull * n * m->inputs);
    SpgGpuCommands *c = chunk_commands(m, n, k & 1u);
    return c && spg_gpu_submit(c) ? c : NULL;
}

static bool spingalett_gpu_model_predict_here(SpgGpuModel *m, const float *inputs, uint32_t count, float *outputs) {
    const uint32_t chunks = (count + m->capacity - 1u) / m->capacity;
    SpgGpuCommands *running = submit_chunk(m, inputs, count, 0);
    if (!running) return false;
    for (uint32_t k = 0; k < chunks; k++) {
        /* the next chunk queued while this one runs, then this one's outputs */
        SpgGpuCommands *next = NULL;
        if (k + 1u < chunks && !(next = submit_chunk(m, inputs, count, k + 1u))) {
            spg_gpu_wait(running);
            return false;
        }
        if (!spg_gpu_wait(running)) {
            m->lost = true;
            if (next) spg_gpu_wait(next);
            return false;
        }
        const uint32_t start = k * m->capacity, n = count - start < m->capacity ? count - start : m->capacity;
        float *out = outputs + (size_t)start * m->outputs;
        memcpy(out, (const char *)m->staging[k & 1u].mapped + m->out_at, 4ull * n * m->outputs);
        /* the last layer's activation that needs libm: each sample's outputs, as batched prediction applies it */
        if (m->host_act != ACT_NONE)
            for (uint32_t s = 0; s < n; s++)
                spingalett_engine_activate(out + (size_t)s * m->outputs, m->outputs, m->host_act);
        running = next;
    }
    return true;
}

bool spingalett_gpu_model_predict(SpgGpuModel *m, const float *inputs, uint32_t count, float *outputs) {
    if (!m || m->lost) return false;
    const SpgBackend saved = spg_gpu_using();
    spg_gpu_use(m->backend);
    const bool ok = spingalett_gpu_model_predict_here(m, inputs, count, outputs);
    spg_gpu_use(saved);
    return ok;
}
