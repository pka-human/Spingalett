/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/*
 * Networks on the GPU (Spingalett.Gpu.h): the batch path of Spingalett.Batch.c and
 * Spingalett.Training.c recorded as Vulkan command buffers. Every layer's outputs (and while
 * training, its gradients) have a buffer of `capacity` rows; parameters, their gradients and the
 * optimizer moments live in device memory for the life of the SpgGpuNet, each layer's starting at a
 * multiple of four floats (the padding stays zero), so that kernels read them four at a time.
 *
 * A chunk's commands are recorded once per (samples, gradient scale, first chunk of a step, last
 * chunk, staging slot) and submitted again for every chunk of that kind: what changes from step to
 * step (learning rate, Adam's corrections, the step number of dropout masks) is in a header the
 * host writes into the slot's staging memory with the inputs and targets, copied to the device at
 * the start of the chunk. Two slots alternate, so that the host fills one while the device runs the
 * other.
 */

#include "Spingalett.Gpu.h"
#include "Spingalett.GpuKernels.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define MAX_GROUPS  65535u          /* workgroups in x of element-wise kernels, which loop beyond */
#define COLSUM_ROWS 1024u           /* rows per slice of colsum.comp */
#define SUMSQ_SLICE 4096u           /* sumsq.comp */

/* ------------------------------------------------------------------------- the network */

typedef struct {
    SpgGpuBuffer staging;           /* header | inputs | targets | losses | outputs, host-visible */
    size_t inputs, targets, losses, outputs;    /* their byte offsets */
    /* the chunk's header, inputs (the outputs of layer 0) and targets on the device: a slot's own,
       so that they are copied while the other slot's chunk runs */
    SpgGpuBuffer header, input, target;
    struct { uint64_t key; SpgGpuCommands *commands; } cache[8];
    uint32_t used;                  /* entries of cache */
    SpgGpuCommands *pending;        /* submitted and not yet harvested */
    uint32_t n;                     /* samples of the pending chunk */
    bool train, last;               /* the pending chunk trained, and ended its step */
    float *dest;                    /* where a prediction's outputs go */
} Slot;

struct SpgGpuNet {
    NeuralNetwork *net;
    uint32_t capacity, layers;
    bool training;
    bool bf16;                      /* matrix products in bfloat16 on the matrix units */
    bool lost;                      /* waiting for a chunk failed (the device was lost): nothing it made counts */
    SpgGpuTraining cfg;

    /* parameters: weight layer l's weights at woff[l] of the Wp weight floats, its biases (and
       running statistics) at boff[l] of the Bp bias floats */
    uint64_t *woff, *boff, Wp, Bp;
    SpgGpuBuffer params;            /* weights | biases | running means | running variances */
    SpgGpuBuffer grads;             /* weight gradients | bias gradients */
    SpgGpuBuffer moment1, moment2;  /* optimizer state, laid out like the gradients */
    SpgGpuBuffer *act, *delta, *dmask;  /* per layer: capacity rows (act[0]: the recording slot's input) */
    SpgGpuBuffer gtmp;              /* gradients of layers that feed several, before they are added */
    SpgGpuBuffer targets, header;   /* the recording slot's targets and step header */
    SpgGpuBuffer scalars;           /* the gradient clipping scale */
    SpgGpuBuffer *geo;              /* per conv layer: its geometries (spg_conv_geometry()) */
    SpgConvGeometry **conv;         /* and where they are */
    SpgGpuBuffer *bn;               /* per batch normalization layer: stats (4C) | coefficients (3C) */
    SpgGpuBuffer part;              /* partial sums: colsum slices, split products, norm slices */
    SpgGpuBuffer wt;                /* convolution weights regrouped for data gradients */
    SpgGpuBuffer transfer;          /* host-visible: parameters on their way in or out */
    uint32_t *uses, *pending;       /* consumers of each layer, and those not yet back-propagated */
    /* batch normalization layers (no activation, no dropout) read only by an addition of two layers,
       which applies them as it adds (their own outputs are never stored, and their gradient is the
       addition's): fold[l] = that addition, or 0 */
    uint32_t *fold;

    Slot slots[2];
    uint32_t next;                  /* the slot of the next chunk */
    float step_loss, total_loss;    /* losses harvested so far: of the open step, of closed steps */
};

static inline uint64_t at(const SpgGpuBuffer *b, uint64_t floats) {
    return b->address + 4u * floats;
}

static inline uint32_t groups(uint64_t items, uint32_t per) {
    uint64_t g = (items + per - 1u) / per;
    return g > MAX_GROUPS ? MAX_GROUPS : (uint32_t)g;
}

static inline uint32_t layer_groups(const LayerShape *s) {
    return s->groups ? s->groups : 1u;
}

static inline uint64_t align4(uint64_t n) {
    return (n + 3u) & ~(uint64_t)3u;
}

bool spingalett_gpu_available(void) {
    return spg_gpu_open();
}

const char *spingalett_gpu_name(void) {
    return spg_gpu_device_name();
}

bool spingalett_gpu_supports(const NeuralNetwork *net, const char **why) {
    for (uint32_t l = 0; l < net->layers; l++)
        if ((uint64_t)net->topology[l] * SPINGALETT_BATCH_CHUNK > UINT32_MAX) {
            if (why) *why = "layers of over 2^32 / 2048 values a sample";
            return false;
        }
    for (uint32_t l = 1; l < net->layers; l++) {
        const LayerShape *s = &net->shapes[l];
        if (s->type == LAYER_CONV2D && (s->kernel_h > 255u || s->kernel_w > 255u ||
                                        net->shapes[spingalett_source(net, l)].channels > 65535u ||
                                        s->channels > 65535u)) {
            if (why) *why = "convolution windows over 255 cells or layers of over 65535 channels";
            return false;
        }
        if (s->type == LAYER_CONV2D && s->stride_h * s->stride_w > SPG_MAX_PHASES) {
            if (why) *why = "convolutions whose strides multiply to over 64";
            return false;
        }
    }
    return true;
}

/* Floats a sample takes on the device: outputs of every layer, and while training their gradients
   and dropout masks. */
static uint64_t sample_floats(const NeuralNetwork *net, bool training) {
    uint64_t f = 0;
    for (uint32_t l = 0; l < net->layers; l++) {
        f += net->topology[l];
        if (training && l > 0) f += net->topology[l] * (net->dropout_rates[l] > 0.0f ? 2u : 1u);
    }
    return f + net->topology[net->layers - 1];      /* targets */
}

uint32_t spingalett_gpu_capacity(const NeuralNetwork *net, uint32_t want, bool training) {
    uint64_t memory = spg_gpu_memory();
    if (memory == 0 || want == 0) return 0;
    /* half the device's memory for the chunk's buffers (the parameters and scratch are smaller) */
    uint64_t per = sample_floats(net, training) * 4u, cap = memory / 2u / (per ? per : 1u);
    if (cap > SPINGALETT_BATCH_CHUNK) cap = SPINGALETT_BATCH_CHUNK;
    return cap < want ? (uint32_t)cap : want;
}

uint32_t spingalett_gpu_net_capacity(const SpgGpuNet *g) {
    return g->capacity;
}

bool spingalett_gpu_net_current(const SpgGpuNet *g) {
    return !g->lost && g->bf16 == (spingalett_get_gpu_precision() == PRECISION_BFLOAT16 && spg_gpu_mma_bf16());
}

/* ------------------------------------------------------------------------- transfers */

/* Runs one-off commands (uploads, downloads) after all chunks submitted so far. */
static bool run_once(SpgGpuNet *g, void (*record)(SpgGpuNet *, SpgGpuCommands *, void *), void *ctx) {
    SpgGpuCommands *c = spg_gpu_commands_create();
    if (!c) return false;
    bool ok = spg_gpu_record_begin(c);
    if (ok) {
        spg_gpu_barrier(c);
        record(g, c, ctx);
        spg_gpu_barrier(c);
        ok = spg_gpu_record_end(c) && spg_gpu_submit(c) && spg_gpu_wait(c);
    }
    spg_gpu_commands_free(c);
    return ok;
}

typedef struct { const SpgGpuBuffer *dst; size_t offset, bytes; bool down; } Copy;

static void record_copy(SpgGpuNet *g, SpgGpuCommands *c, void *ctx) {
    const Copy *cp = (const Copy *)ctx;
    if (cp->down) spg_gpu_copy(c, cp->dst, cp->offset, &g->transfer, 0, cp->bytes);
    else spg_gpu_copy(c, &g->transfer, 0, cp->dst, cp->offset, cp->bytes);
}

/* Copies a device buffer's first `bytes` to or from the transfer buffer. */
static bool move(SpgGpuNet *g, const SpgGpuBuffer *dst, size_t bytes, bool down) {
    Copy cp = {dst, 0, bytes, down};
    return run_once(g, record_copy, &cp);
}

static void record_fill(SpgGpuNet *g, SpgGpuCommands *c, void *ctx) {
    (void)g;
    const SpgGpuBuffer *b = (const SpgGpuBuffer *)ctx;
    spg_gpu_fill(c, b, 0, b->size, 0u);
}

static bool zero(SpgGpuNet *g, SpgGpuBuffer *b) {
    return run_once(g, record_fill, b);
}

/* A parameter array of the network (laid out by weight_offsets or bias_offsets) and its padded
   image in the transfer buffer at `base` floats, one way or the other. */
static void pack(const SpgGpuNet *g, float *host, uint64_t base, bool weights, bool down) {
    const NeuralNetwork *net = g->net;
    float *image = (float *)g->transfer.mapped + base;
    for (uint32_t l = 0; l + 1 < net->layers; l++) {
        uint64_t rows = spingalett_weight_rows(net, l), count = weights ? rows * spingalett_weight_row_len(net, l) : rows;
        uint64_t from = weights ? net->weight_offsets[l] : net->bias_offsets[l];
        uint64_t to = base + (weights ? g->woff[l] : g->boff[l]);
        if (count == 0) continue;
        if (down) memcpy(host + from, image - base + to, count * sizeof(float));
        else memcpy(image - base + to, host + from, count * sizeof(float));
    }
}

static void harvest(SpgGpuNet *g, Slot *s);

/* Waits for every chunk in flight, oldest first, and takes their losses. */
static void drain(SpgGpuNet *g) {
    for (uint32_t k = 0; k < 2; k++) {
        Slot *s = &g->slots[(g->next + k) % 2];
        if (s->pending) harvest(g, s);
    }
}

/* Weights and biases of an array pair (parameters, gradients, a moment) to or from the device. */
static bool exchange(SpgGpuNet *g, const SpgGpuBuffer *b, float *weights, float *biases, bool down) {
    const size_t bytes = (g->Wp + g->Bp) * sizeof(float);
    if (!b->buffer) return true;
    if (down && !move(g, b, bytes, true)) return false;
    pack(g, weights, 0, true, down);
    pack(g, biases, g->Wp, false, down);
    return down || move(g, b, bytes, false);
}

bool spingalett_gpu_upload(SpgGpuNet *g) {
    NeuralNetwork *net = g->net;
    drain(g);
    /* the padding is zero (copied with the image, which starts zeroed) */
    memset(g->transfer.mapped, 0, (g->Wp + 3u * g->Bp) * sizeof(float));
    pack(g, net->weights, 0, true, false);
    pack(g, net->biases, g->Wp, false, false);
    pack(g, net->running_mean, g->Wp + g->Bp, false, false);
    pack(g, net->running_var, g->Wp + 2u * g->Bp, false, false);
    return move(g, &g->params, (g->Wp + 3u * g->Bp) * sizeof(float), false) &&
           exchange(g, &g->moment1, net->opt_m_weights, net->opt_m_biases, false) &&
           exchange(g, &g->moment2, net->opt_v_weights, net->opt_v_biases, false);
}

bool spingalett_gpu_download(SpgGpuNet *g) {
    NeuralNetwork *net = g->net;
    drain(g);
    if (g->lost || !move(g, &g->params, (g->Wp + 3u * g->Bp) * sizeof(float), true)) return false;
    pack(g, net->weights, 0, true, true);
    pack(g, net->biases, g->Wp, false, true);
    pack(g, net->running_mean, g->Wp + g->Bp, false, true);
    pack(g, net->running_var, g->Wp + 2u * g->Bp, false, true);
    return exchange(g, &g->grads, net->grad_weights, net->grad_biases, true) &&
           exchange(g, &g->moment1, net->opt_m_weights, net->opt_m_biases, true) &&
           exchange(g, &g->moment2, net->opt_v_weights, net->opt_v_biases, true);
}

/* ------------------------------------------------------------------------- creation */

static bool make_geometry(SpgGpuNet *g, uint32_t l) {
    const NeuralNetwork *net = g->net;
    const LayerShape *in = &net->shapes[spingalett_source(net, l)], *s = &net->shapes[l];
    SpgConvGeometry *info = g->conv[l] = (SpgConvGeometry *)malloc(sizeof *info);
    if (!info || !spg_conv_geometry(NULL, info, in->height, in->width, in->channels, s->height, s->width, s->channels,
                                    layer_groups(s), s->kernel_h, s->kernel_w, s->stride_h, s->stride_w, s->pad_h,
                                    s->pad_w) ||
        info->size * sizeof(uint32_t) > g->transfer.size)
        return false;
    spg_conv_geometry((uint32_t *)g->transfer.mapped, info, in->height, in->width, in->channels, s->height, s->width,
                      s->channels, layer_groups(s), s->kernel_h, s->kernel_w, s->stride_h, s->stride_w, s->pad_h,
                      s->pad_w);
    return spg_gpu_buffer_create(&g->geo[l], info->size * sizeof(uint32_t), false) &&
           move(g, &g->geo[l], info->size * sizeof(uint32_t), false);
}

/* Floats of the partial sums the network's layers need at most. */
static uint64_t part_floats(const NeuralNetwork *net, uint32_t capacity) {
    uint64_t need = 2u * SPINGALETT_BATCH_CHUNK;
    uint64_t norm = (net->total_weights + net->total_biases + 8u * net->layers + SUMSQ_SLICE - 1u) / SUMSQ_SLICE;
    if (norm > need) need = norm;
    for (uint32_t l = 1; l < net->layers; l++) {
        const LayerShape *s = &net->shapes[l];
        uint64_t rows = (uint64_t)capacity * s->height * s->width;
        uint64_t slices = (rows + COLSUM_ROWS - 1u) / COLSUM_ROWS;
        if (slices * 2u * s->channels > need) need = slices * 2u * s->channels;
        if (s->type == LAYER_CONV2D || s->type == LAYER_DENSE) {
            /* the partial products of split weight gradients */
            uint64_t w = (uint64_t)spingalett_weight_rows(net, l - 1) * spingalett_weight_row_len(net, l - 1);
            if (w > SPG_SPLIT_FLOATS / 2u) continue;
            uint32_t slice_k, split = spg_gemm_split(1, 1, (uint32_t)rows, 1, &slice_k);
            uint64_t floats = split * w < SPG_SPLIT_FLOATS ? split * w : SPG_SPLIT_FLOATS;
            if (floats > need) need = floats;
        }
    }
    return need;
}

void spingalett_gpu_net_free(SpgGpuNet *g) {
    if (!g) return;
    drain(g);
    for (uint32_t k = 0; k < 2; k++) {
        for (uint32_t e = 0; e < g->slots[k].used; e++) spg_gpu_commands_free(g->slots[k].cache[e].commands);
        spg_gpu_buffer_free(&g->slots[k].staging);
        spg_gpu_buffer_free(&g->slots[k].header);
        spg_gpu_buffer_free(&g->slots[k].input);
        spg_gpu_buffer_free(&g->slots[k].target);
    }
    if (g->act) memset(&g->act[0], 0, sizeof g->act[0]);  /* a slot's */
    for (uint32_t l = 0; l < g->layers; l++) {
        if (g->fold && g->fold[l] && g->delta) memset(&g->delta[l], 0, sizeof g->delta[l]);   /* the addition's */
    }
    for (uint32_t l = 0; l < g->layers; l++) {
        if (g->conv) free(g->conv[l]);
        if (g->act) spg_gpu_buffer_free(&g->act[l]);
        if (g->delta) spg_gpu_buffer_free(&g->delta[l]);
        if (g->dmask) spg_gpu_buffer_free(&g->dmask[l]);
        if (g->geo) spg_gpu_buffer_free(&g->geo[l]);
        if (g->bn) spg_gpu_buffer_free(&g->bn[l]);
    }
    SpgGpuBuffer *single[] = {&g->params, &g->grads, &g->moment1, &g->moment2, &g->gtmp, &g->scalars, &g->part,
                              &g->wt, &g->transfer};
    for (size_t k = 0; k < sizeof single / sizeof single[0]; k++) spg_gpu_buffer_free(single[k]);
    free(g->act); free(g->delta); free(g->dmask); free(g->geo); free(g->conv); free(g->bn);
    free(g->uses); free(g->pending); free(g->woff); free(g->boff); free(g->fold);
    free(g);
    spg_gemm_release();
}

SpgGpuNet *spingalett_gpu_net_create(NeuralNetwork *net, uint32_t capacity, const SpgGpuTraining *training) {
    if (!spg_gpu_open() || capacity == 0) return NULL;
    SpgGpuNet *g = (SpgGpuNet *)calloc(1, sizeof *g);
    if (!g) return NULL;
    const uint32_t L = net->layers;
    g->net = net;
    g->capacity = capacity;
    g->layers = L;
    g->training = training != NULL;
    g->bf16 = spingalett_get_gpu_precision() == PRECISION_BFLOAT16 && spg_gpu_mma_bf16();
    if (training) g->cfg = *training;
    g->act = (SpgGpuBuffer *)calloc(L, sizeof(SpgGpuBuffer));
    g->delta = (SpgGpuBuffer *)calloc(L, sizeof(SpgGpuBuffer));
    g->dmask = (SpgGpuBuffer *)calloc(L, sizeof(SpgGpuBuffer));
    g->geo = (SpgGpuBuffer *)calloc(L, sizeof(SpgGpuBuffer));
    g->conv = (SpgConvGeometry **)calloc(L, sizeof(SpgConvGeometry *));
    g->bn = (SpgGpuBuffer *)calloc(L, sizeof(SpgGpuBuffer));
    g->uses = (uint32_t *)calloc(L, sizeof(uint32_t));
    g->pending = (uint32_t *)calloc(L, sizeof(uint32_t));
    g->woff = (uint64_t *)calloc(L, sizeof(uint64_t));
    g->boff = (uint64_t *)calloc(L, sizeof(uint64_t));
    g->fold = (uint32_t *)calloc(L, sizeof(uint32_t));
    if (!g->act || !g->delta || !g->dmask || !g->geo || !g->conv || !g->bn || !g->uses || !g->pending ||
        !g->woff || !g->boff || !g->fold)
        goto fail;
    for (uint32_t l = 1; l < L; l++)
        for (uint32_t k = 0; k < spingalett_input_count(net, l); k++) g->uses[spingalett_inputs(net, l)[k]]++;
    for (uint32_t c = 1; c < L; c++) {
        if (net->shapes[c].type != LAYER_ADD || spingalett_input_count(net, c) != 2) continue;
        for (uint32_t k = 2; k-- > 0;) {        /* one input of an addition: the later */
            const uint32_t l = spingalett_inputs(net, c)[k];
            const uint32_t other = spingalett_inputs(net, c)[1 - k];
            if (l != other && net->shapes[l].type == LAYER_BATCH_NORM && net->act_func[l - 1] == ACT_NONE &&
                g->uses[l] == 1 && net->dropout_rates[l] == 0.0f) {
                g->fold[l] = c;
                break;
            }
        }
    }
    /* every layer's parameters at a multiple of four floats */
    for (uint32_t l = 0; l + 1 < L; l++) {
        uint64_t rows = spingalett_weight_rows(net, l);
        g->woff[l] = g->Wp;
        g->boff[l] = g->Bp;
        g->Wp += align4(rows * spingalett_weight_row_len(net, l));
        g->Bp += align4(rows);
    }

    const uint64_t P = g->Wp + 3u * g->Bp;
    const uint32_t out = net->topology[L - 1];
    bool ok = spg_gpu_buffer_create(&g->params, P * 4u, false) &&
              spg_gpu_buffer_create(&g->transfer, P * 4u > (4u << 20) ? P * 4u : (4u << 20), true) &&
              spg_gpu_buffer_create(&g->scalars, 16u, false) &&
              spg_gpu_buffer_create(&g->part, part_floats(net, capacity) * 4u, false);
    uint64_t widest_shared = 0, wt = 0;
    for (uint32_t l = 0; ok && l < L; l++) {
        const LayerShape *s = &net->shapes[l];
        if (l > 0 && !g->fold[l]) ok = spg_gpu_buffer_create(&g->act[l], (uint64_t)capacity * net->topology[l] * 4u, false);
        if (ok && training && l > 0 && !g->fold[l])
            ok = spg_gpu_buffer_create(&g->delta[l], (uint64_t)capacity * net->topology[l] * 4u, false);
        if (ok && training && l > 0 && l + 1 < L && net->dropout_rates[l] > 0.0f)
            ok = spg_gpu_buffer_create(&g->dmask[l], (uint64_t)capacity * net->topology[l] * 4u, false);
        if (l > 0 && g->uses[l] > 1 && net->topology[l] > widest_shared) widest_shared = net->topology[l];
        if (ok && s->type == LAYER_CONV2D) {
            ok = make_geometry(g, l);
            uint64_t w = (uint64_t)spingalett_weight_rows(net, l - 1) * spingalett_weight_row_len(net, l - 1);
            if (w > wt) wt = w;
        }
        if (ok && s->type == LAYER_BATCH_NORM) ok = spg_gpu_buffer_create(&g->bn[l], 7u * s->channels * 4u, false);
    }
    for (uint32_t l = 1; ok && training && l < L; l++)
        if (g->fold[l]) g->delta[l] = g->delta[g->fold[l]];     /* the addition's gradient, shared */
    if (ok && wt > 0 && training) ok = spg_gpu_buffer_create(&g->wt, wt * 4u, false);
    if (ok && training) {
        /* gradients and moments start zeroed: their padding is never written */
        ok = spg_gpu_buffer_create(&g->grads, (g->Wp + g->Bp) * 4u, false) && zero(g, &g->grads);
        if (ok && widest_shared > 0) ok = spg_gpu_buffer_create(&g->gtmp, (uint64_t)capacity * widest_shared * 4u, false);
        OptimizerType o = training->optimizer;
        if (ok && (o == OPTIMIZER_MOMENTUM || o == OPTIMIZER_ADAM || o == OPTIMIZER_ADAMW))
            ok = spg_gpu_buffer_create(&g->moment1, (g->Wp + g->Bp) * 4u, false) && zero(g, &g->moment1);
        if (ok && (o == OPTIMIZER_RMSPROP || o == OPTIMIZER_ADAM || o == OPTIMIZER_ADAMW))
            ok = spg_gpu_buffer_create(&g->moment2, (g->Wp + g->Bp) * 4u, false) && zero(g, &g->moment2);
    }
    /* staging: header, inputs, targets, losses and outputs of a chunk, 16-byte aligned */
    const uint64_t in = net->topology[0];
    for (uint32_t k = 0; ok && k < 2; k++) {
        Slot *s = &g->slots[k];
        s->inputs = SPG_STEP_HEADER * 4u;
        s->targets = s->inputs + (((uint64_t)capacity * in * 4u + 15u) & ~15ull);
        s->losses = s->targets + (((uint64_t)capacity * out * 4u + 15u) & ~15ull);
        s->outputs = s->losses + (((uint64_t)capacity * 4u + 15u) & ~15ull);
        ok = spg_gpu_buffer_create(&s->staging, s->outputs + (uint64_t)capacity * out * 4u, true) &&
             spg_gpu_buffer_create(&s->header, SPG_STEP_HEADER * 4u, false) &&
             spg_gpu_buffer_create(&s->input, (uint64_t)capacity * in * 4u, false) &&
             (!training || spg_gpu_buffer_create(&s->target, (uint64_t)capacity * out * 4u, false));
    }
    if (ok && spingalett_gpu_upload(g)) return g;
fail:
    spingalett_gpu_net_free(g);
    return NULL;
}

/* ------------------------------------------------------------------------- recording */

/*
 * Commands are recorded through a Recorder, which keeps the memory read and written since the last
 * barrier: every dispatch states what it reads and writes, and a barrier is recorded before it only
 * when it reads what an earlier one wrote or writes what an earlier one read or wrote. Independent
 * dispatches, such as a layer's weight gradient and its data gradient, then run side by side.
 * Consecutive dispatches of one nonzero `group` access the same ranges and write disjoint parts of
 * them (the phases of a data gradient): they do not wait for each other. A group belongs to one layer
 * and purpose (group_of()), so that the next layer's dispatches, which read what these wrote, wait.
 */
typedef struct { uint64_t lo, hi; } Range;

#define TRACKED 96u
typedef struct {
    SpgGpuCommands *c;
    Range reads[TRACKED], writes[TRACKED];
    uint32_t nr, nw, group;
} Recorder;

typedef struct {
    Range read[8], write[4];
    uint32_t nr, nw, group;
} Access;

enum { GROUP_SLICES = 1, GROUP_PHASES = 2, GROUP_OPTIMIZER = 3 };
static uint32_t group_of(uint32_t layer, uint32_t purpose) { return 4u * layer + purpose; }

static Range whole(const SpgGpuBuffer *b) { return (Range){b->address, b->address + b->size}; }
static Range span(uint64_t address, uint64_t floats) { return (Range){address, address + 4u * floats}; }

static void reads(Access *a, Range r) { if (r.hi > r.lo) a->read[a->nr++] = r; }
static void writes(Access *a, Range r) { if (r.hi > r.lo) a->write[a->nw++] = r; }

static bool hits(const Range *list, uint32_t n, Range r) {
    for (uint32_t k = 0; k < n; k++)
        if (r.lo < list[k].hi && list[k].lo < r.hi) return true;
    return false;
}

static void barrier(Recorder *r) {
    spg_gpu_barrier(r->c);
    r->nr = r->nw = 0;
    r->group = 0;
}

/* A barrier if the access conflicts with what was recorded since the last one; then it is noted. */
static void need(Recorder *r, const Access *a) {
    bool wait = r->nr + a->nr > TRACKED || r->nw + a->nw > TRACKED;
    if (!(a->group != 0 && a->group == r->group)) {
        for (uint32_t k = 0; k < a->nr && !wait; k++) wait = hits(r->writes, r->nw, a->read[k]);
        for (uint32_t k = 0; k < a->nw && !wait; k++)
            wait = hits(r->writes, r->nw, a->write[k]) || hits(r->reads, r->nr, a->write[k]);
    }
    if (wait) barrier(r);
    memcpy(r->reads + r->nr, a->read, a->nr * sizeof(Range));
    memcpy(r->writes + r->nw, a->write, a->nw * sizeof(Range));
    r->nr += a->nr;
    r->nw += a->nw;
    r->group = a->group;
}

static void kernel(Recorder *r, const Access *a, SpgKernel k, const uint32_t *spec, uint32_t spec_count,
                   const void *push, uint32_t push_size, uint32_t gx, uint32_t gy, uint32_t gz) {
    need(r, a);
    spg_gpu_dispatch(r->c, k, spec, spec_count, push, push_size, gx, gy, gz);
}

static void product(Recorder *r, const Access *a, SpgGemmPush *p, const SpgGemmMode *m) {
    need(r, a);
    /* the matrix units multiply blocks of 16 x 16 (32 values of k a step): products smaller than that in
       a dimension would mostly multiply padding, and stay in single precision (a rule of the shape, so
       that a product always runs the same way) */
    SpgGemmMode mode = *m;
    mode.bf16 = m->bf16 && p->K >= 32u && p->M >= 16u && p->N >= 16u;
    spg_gemm(r->c, p, &mode);
}

/* The parameters of weight layer l: weights, biases, running statistics, and their gradients. */
static uint64_t weight_count(const SpgGpuNet *g, uint32_t l) {
    return (uint64_t)spingalett_weight_rows(g->net, l) * spingalett_weight_row_len(g->net, l);
}
static uint64_t bias_count(const SpgGpuNet *g, uint32_t l) { return spingalett_weight_rows(g->net, l); }
static uint64_t weights_at(const SpgGpuNet *g, uint32_t l) { return at(&g->params, g->woff[l]); }
static uint64_t biases_at(const SpgGpuNet *g, uint32_t l) { return at(&g->params, g->Wp + g->boff[l]); }
static uint64_t running_at(const SpgGpuNet *g, uint32_t l, uint32_t which) {
    return at(&g->params, g->Wp + (1u + which) * g->Bp + g->boff[l]);
}
static uint64_t grad_weights_at(const SpgGpuNet *g, uint32_t l) { return at(&g->grads, g->woff[l]); }
static uint64_t grad_biases_at(const SpgGpuNet *g, uint32_t l) { return at(&g->grads, g->Wp + g->boff[l]); }
static Range weights_of(const SpgGpuNet *g, uint32_t l) { return span(weights_at(g, l), weight_count(g, l)); }
static Range biases_of(const SpgGpuNet *g, uint32_t l) { return span(biases_at(g, l), bias_count(g, l)); }

/* out = scale * (column sums of R rows of C floats at x) + beta out */
static void column_sums(SpgGpuNet *g, Recorder *r, const SpgGpuBuffer *x, uint32_t R, uint32_t C, uint64_t out,
                        float scale, float beta) {
    uint32_t slices = (R + COLSUM_ROWS - 1u) / COLSUM_ROWS, cols = C <= 16u ? 16u : C <= 32u ? 32u : 64u;
    SpgColsumPush cp = {x->address, 0, 0, g->part.address, R, C, COLSUM_ROWS, 0};
    uint32_t spec[2] = {cols, SPG_COLSUM_SUM};
    Access a = {0};
    reads(&a, whole(x));
    writes(&a, span(g->part.address, (uint64_t)slices * C));
    kernel(r, &a, SPG_KERNEL_colsum, spec, 2, &cp, sizeof cp, (C + cols - 1) / cols, slices, 1);
    SpgReducePush rp = {g->part.address, out, C, C, slices, scale, beta, 0};
    Access b = {0};
    reads(&b, span(g->part.address, (uint64_t)slices * C));
    if (beta != 0.0f) reads(&b, span(out, C));
    writes(&b, span(out, C));
    kernel(r, &b, SPG_KERNEL_reduce, NULL, 0, &rp, sizeof rp, (C + 255u) / 256u, 1, 1);
}

static void eltwise(SpgGpuNet *g, Recorder *r, const Access *a, uint32_t op, uint32_t act, uint64_t y, uint64_t x,
                    uint64_t pa, uint64_t pb, uint64_t total, uint32_t n) {
    SpgEltwisePush p = {y, x, pa, pb, g->header.address, (uint32_t)total, n, 0, 0, 0.0f, 0, 0};
    uint32_t spec[2] = {op, act};
    kernel(r, a, SPG_KERNEL_eltwise, spec, 2, &p, sizeof p, groups(total, 256u), 1, 1);
}

/* ---- forward ---- */

static void conv_forward(SpgGpuNet *g, Recorder *r, uint32_t l, uint32_t n, uint32_t act) {
    const NeuralNetwork *net = g->net;
    const uint32_t src = spingalett_source(net, l);
    const LayerShape *in = &net->shapes[src], *s = &net->shapes[l];
    const uint32_t G = layer_groups(s), CG = in->channels / G, OG = s->channels / G;
    const uint32_t K = s->kernel_h * s->kernel_w * CG;
    SpgGemmPush p = {
        .a = g->act[src].address, .b = weights_at(g, l - 1), .c = g->act[l].address, .e0 = biases_at(g, l - 1),
        .geo = g->geo[l].address, .M = n * s->height * s->width, .N = OG, .K = K, .ldb = K, .ldc = s->channels,
        .a_group = CG, .b_group = OG * K, .c_group = OG, .alpha = 1.0f, .flags = SPG_GEMM_BIAS,
    };
    SpgGemmMode m = {SPG_A_CONV, SPG_B_COL, SPG_EPI_BIAS_ACT, act, G, false, CG % 4u == 0 && in->channels % 4u == 0,
                     true, 0, (uint64_t)n * net->topology[l], g->bf16};
    Access a = {0};
    reads(&a, whole(&g->act[src]));
    reads(&a, weights_of(g, l - 1));
    reads(&a, biases_of(g, l - 1));
    reads(&a, whole(&g->geo[l]));
    writes(&a, whole(&g->act[l]));
    product(r, &a, &p, &m);
}

static void dense_forward(SpgGpuNet *g, Recorder *r, uint32_t l, uint32_t n, uint32_t act) {
    const NeuralNetwork *net = g->net;
    const uint32_t src = spingalett_source(net, l), K = net->topology[src], N = net->topology[l];
    SpgGemmPush p = {
        .a = g->act[src].address, .b = weights_at(g, l - 1), .c = g->act[l].address, .e0 = biases_at(g, l - 1),
        .M = n, .N = N, .K = K, .lda = K, .ldb = K, .ldc = N, .alpha = 1.0f, .flags = SPG_GEMM_BIAS,
    };
    SpgGemmMode m = {SPG_A_ROW, SPG_B_COL, SPG_EPI_BIAS_ACT, act, 1, false, true, true, 0, (uint64_t)n * N, g->bf16};
    Access a = {0};
    reads(&a, whole(&g->act[src]));
    reads(&a, weights_of(g, l - 1));
    reads(&a, biases_of(g, l - 1));
    writes(&a, whole(&g->act[l]));
    product(r, &a, &p, &m);
}

/* train: with the batch's statistics (which also move the running ones); otherwise the running statistics,
   as inference and validation use them */
static void bn_forward(SpgGpuNet *g, Recorder *r, uint32_t l, uint32_t n, uint32_t act, bool train) {
    const NeuralNetwork *net = g->net;
    const uint32_t src = spingalett_source(net, l);
    const LayerShape *s = &net->shapes[l];
    const uint32_t C = s->channels, R = n * s->height * s->width, slices = (R + COLSUM_ROWS - 1u) / COLSUM_ROWS;
    const uint64_t stats = g->bn[l].address;
    SpgBnPush bp = {
        .part = g->part.address, .x = g->act[src].address, .gamma = weights_at(g, l - 1), .beta = biases_at(g, l - 1),
        .rmean = running_at(g, l - 1, 0), .rvar = running_at(g, l - 1, 1), .stats = stats, .C = C,
        .slices = slices, .m = (float)R, .eps = s->eps, .momentum = s->momentum,
    };
    uint32_t mode = SPG_BN_INFER;
    Access a = {0};
    reads(&a, weights_of(g, l - 1));
    reads(&a, biases_of(g, l - 1));
    reads(&a, span(running_at(g, l - 1, 0), C));
    reads(&a, span(running_at(g, l - 1, 1), C));
    writes(&a, span(stats, 4u * C));
    if (train) {
        /* the sums of x - x[row 0] and their squares, then the statistics and coefficients */
        uint32_t cols = C <= 16u ? 16u : C <= 32u ? 32u : 64u;
        SpgColsumPush cp = {g->act[src].address, 0, g->act[src].address, g->part.address, R, C, COLSUM_ROWS, 0};
        uint32_t spec[2] = {cols, SPG_COLSUM_SHIFTED};
        Access sums = {0};
        reads(&sums, whole(&g->act[src]));
        writes(&sums, span(g->part.address, 2ull * slices * C));
        kernel(r, &sums, SPG_KERNEL_colsum, spec, 2, &cp, sizeof cp, (C + cols - 1) / cols, slices, 1);
        mode = SPG_BN_TRAIN;
        reads(&a, span(g->part.address, 2ull * slices * C));
        reads(&a, span(g->act[src].address, C));
        writes(&a, span(running_at(g, l - 1, 0), C));
        writes(&a, span(running_at(g, l - 1, 1), C));
    }
    kernel(r, &a, SPG_KERNEL_bn, &mode, 1, &bp, sizeof bp, (C + 63u) / 64u, 1, 1);
    if (g->fold[l]) return;                     /* applied by the addition that reads it */
    Access b = {0};
    reads(&b, whole(&g->act[src]));
    reads(&b, span(stats, 4u * C));
    writes(&b, whole(&g->act[l]));
    eltwise(g, r, &b, SPG_ELT_AFFINE, act, g->act[l].address, g->act[src].address, stats + 8u * C, stats + 12u * C,
            (uint64_t)R * C, C);
}

static void combine_forward(SpgGpuNet *g, Recorder *r, uint32_t l, uint32_t n, uint32_t act) {
    const NeuralNetwork *net = g->net;
    const LayerShape *s = &net->shapes[l];
    const uint32_t count = spingalett_input_count(net, l), *in = spingalett_inputs(net, l);
    const uint32_t cells = s->height * s->width;
    if (s->type == LAYER_GLOBAL_AVG_POOL) {
        const LayerShape *x = &net->shapes[in[0]];
        SpgCombinePush p = {g->act[in[0]].address, g->act[l].address, 0, n, x->height * x->width, x->channels, 0, 0, 0};
        uint32_t spec[3] = {SPG_COMBINE_GAP, act, 0};
        Access a = {0};
        reads(&a, whole(&g->act[in[0]]));
        writes(&a, whole(&g->act[l]));
        kernel(r, &a, SPG_KERNEL_combine, spec, 3, &p, sizeof p, groups((uint64_t)n * x->channels, 256u), 1, 1);
        return;
    }
    if (s->type == LAYER_CONCAT || count == 1) {
        /* each input into its own channels: no waiting between them */
        for (uint32_t k = 0, c0 = 0; k < count; c0 += net->shapes[in[k]].channels, k++) {
            const uint32_t ck = net->shapes[in[k]].channels;
            SpgCombinePush p = {g->act[in[k]].address, g->act[l].address, 0, n, cells, s->channels, c0, ck, 0};
            uint32_t spec[3] = {SPG_COMBINE_SLICE, act, 0};
            Access a = {.group = group_of(l, GROUP_SLICES)};
            for (uint32_t j = 0; j < count; j++) reads(&a, whole(&g->act[in[j]]));
            writes(&a, whole(&g->act[l]));
            kernel(r, &a, SPG_KERNEL_combine, spec, 3, &p, sizeof p, groups((uint64_t)n * cells * ck, 256u), 1, 1);
        }
        return;
    }
    /* an addition of a normalization it applies: act(other + normalized) in one pass */
    for (uint32_t k = 0; k < count; k++) {
        const uint32_t i = in[k];
        if (g->fold[i] != l) continue;
        const uint32_t src = spingalett_source(net, i), other = in[1 - k], C = net->shapes[i].channels;
        const uint64_t stats = g->bn[i].address;
        SpgEltwisePush p = {g->act[l].address, g->act[src].address, stats + 8u * C, stats + 12u * C, 0,
                            n * net->topology[l], C, 0, 0, 0.0f, 0, g->act[other].address};
        uint32_t spec[2] = {SPG_ELT_AFFINE_ADD, act};
        Access a = {0};
        reads(&a, whole(&g->act[src]));
        reads(&a, whole(&g->act[other]));
        reads(&a, span(stats, 4u * C));
        writes(&a, whole(&g->act[l]));
        kernel(r, &a, SPG_KERNEL_eltwise, spec, 2, &p, sizeof p, groups((uint64_t)n * net->topology[l], 256u), 1, 1);
        return;
    }
    /* additions: the first two inputs, then each further one, the activation after the last */
    for (uint32_t k = 1; k < count; k++) {
        SpgCombinePush p = {g->act[in[k == 1 ? 0 : k]].address, g->act[l].address, g->act[in[1]].address, n, cells,
                            s->channels, 0, 0, (k == 1 ? 0u : 1u) | (k + 1 == count ? 2u : 0u)};
        uint32_t spec[3] = {SPG_COMBINE_ADD, act, 0};
        Access a = {0};
        reads(&a, whole(&g->act[in[k == 1 ? 0 : k]]));
        if (k == 1) reads(&a, whole(&g->act[in[1]]));
        else reads(&a, whole(&g->act[l]));
        writes(&a, whole(&g->act[l]));
        kernel(r, &a, SPG_KERNEL_combine, spec, 3, &p, sizeof p, groups((uint64_t)n * net->topology[l], 256u), 1, 1);
    }
}

static void pool_forward(SpgGpuNet *g, Recorder *r, uint32_t l, uint32_t n) {
    const NeuralNetwork *net = g->net;
    const uint32_t src = spingalett_source(net, l);
    const LayerShape *in = &net->shapes[src], *s = &net->shapes[l];
    SpgPoolPush p = {g->act[src].address, g->act[l].address, 0, 0, n, in->height, in->width, in->channels, s->height,
                     s->width, s->kernel_h, s->kernel_w, s->stride_h, s->stride_w, s->pad_h, s->pad_w};
    uint32_t spec[3] = {s->type == LAYER_MAX_POOL2D, 0, ACT_NONE};
    Access a = {0};
    reads(&a, whole(&g->act[src]));
    writes(&a, whole(&g->act[l]));
    kernel(r, &a, SPG_KERNEL_pool, spec, 3, &p, sizeof p, groups((uint64_t)n * net->topology[l], 256u), 1, 1);
}

static void record_forward(SpgGpuNet *g, Recorder *r, uint32_t n, bool train) {
    const NeuralNetwork *net = g->net;
    for (uint32_t l = 1; l < net->layers; l++) {
        const LayerType type = net->shapes[l].type;
        const ActivationFunction act = net->act_func[l - 1];
        const bool masked = train && g->dmask[l].buffer;
        const uint64_t total = (uint64_t)n * net->topology[l];
        /* softmax runs over whole rows after the layer; every other activation in its kernel */
        const uint32_t fused = act == ACT_SOFTMAX ? ACT_NONE : act;
        if (type == LAYER_DENSE) dense_forward(g, r, l, n, fused);
        else if (type == LAYER_CONV2D) conv_forward(g, r, l, n, fused);
        else if (type == LAYER_BATCH_NORM) bn_forward(g, r, l, n, fused, train);
        else if (type == LAYER_ADD || type == LAYER_CONCAT || type == LAYER_GLOBAL_AVG_POOL)
            combine_forward(g, r, l, n, fused);
        else {
            pool_forward(g, r, l, n);
            if (fused != ACT_NONE) {
                Access a = {0};
                reads(&a, whole(&g->act[l]));
                writes(&a, whole(&g->act[l]));
                eltwise(g, r, &a, SPG_ELT_BIAS_ACT, fused, g->act[l].address, 0, 0, 0, total, net->topology[l]);
            }
        }
        if (act == ACT_SOFTMAX) {
            SpgOutputPush p = {g->act[l].address, 0, 0, 0, n, net->topology[l]};
            uint32_t spec[3] = {SPG_OUT_SOFTMAX, ACT_SOFTMAX, 0};
            Access a = {0};
            reads(&a, whole(&g->act[l]));
            writes(&a, whole(&g->act[l]));
            kernel(r, &a, SPG_KERNEL_output, spec, 3, &p, sizeof p, (n + 63u) / 64u, 1, 1);
        }
        if (masked) {
            const float rate = net->dropout_rates[l];
            SpgEltwisePush p = {g->act[l].address, g->dmask[l].address, 0, 0, g->header.address, (uint32_t)total,
                                net->topology[l], l, (uint32_t)((double)rate * 4294967296.0), 1.0f / (1.0f - rate), 0,
                                0};
            uint32_t spec[2] = {SPG_ELT_DROPOUT, act};
            Access a = {0};
            reads(&a, whole(&g->act[l]));
            reads(&a, whole(&g->header));
            writes(&a, whole(&g->act[l]));
            writes(&a, whole(&g->dmask[l]));
            kernel(r, &a, SPG_KERNEL_eltwise, spec, 2, &p, sizeof p, groups(total, 256u), 1, 1);
        }
    }
}

/* ---- backward ---- */

/* The gradient input k of layer cl gets from it, into dst (added to it with accumulate: dense,
   convolution, adding, concatenating and global pooling layers), then times act'(the input's
   outputs) when fused is not ACT_NONE. Returns whether the derivative was applied. */
static bool input_gradient(SpgGpuNet *g, Recorder *r, uint32_t cl, uint32_t k, const SpgGpuBuffer *dst, uint32_t n,
                           uint32_t fused, bool accumulate) {
    const NeuralNetwork *net = g->net;
    const uint32_t *in = spingalett_inputs(net, cl), i = in[k];
    const LayerShape *s = &net->shapes[cl], *x = &net->shapes[i];
    const bool derive = fused != ACT_NONE;
    Access a = {0};
    reads(&a, whole(&g->delta[cl]));
    if (derive || s->type == LAYER_BATCH_NORM || s->type == LAYER_MAX_POOL2D || s->type == LAYER_AVG_POOL2D)
        reads(&a, whole(&g->act[i]));
    if (accumulate) reads(&a, whole(dst));
    writes(&a, whole(dst));
    switch (s->type) {
        case LAYER_DENSE: {         /* dst = delta[cl] W, W stored [next x cur] */
            const uint32_t cur = net->topology[i], next = net->topology[cl];
            SpgGemmPush p = {
                .a = g->delta[cl].address, .b = weights_at(g, cl - 1), .c = dst->address, .e0 = g->act[i].address,
                .M = n, .N = cur, .K = next, .lda = next, .ldb = cur, .ldc = cur, .alpha = 1.0f,
                .beta = accumulate ? 1.0f : 0.0f,
            };
            SpgGemmMode m = {SPG_A_ROW, SPG_B_ROW, derive ? SPG_EPI_DERIV : SPG_EPI_STORE, fused, 1, false, true, true, 0,
                             (uint64_t)n * cur, g->bf16};
            reads(&a, weights_of(g, cl - 1));
            product(r, &a, &p, &m);
            return derive;
        }
        case LAYER_CONV2D: {
            /* the weights regrouped by phase, then a stride-1 product per phase of the stride over the
               input pixels it holds (one group: they write different pixels) */
            const uint32_t G = layer_groups(s), CG = x->channels / G, OG = s->channels / G;
            const uint32_t taps = s->kernel_h * s->kernel_w;
            const SpgConvGeometry *info = g->conv[cl];
            SpgWtransPush wp = {weights_at(g, cl - 1), g->wt.address, g->geo[cl].address + 4u * info->order,
                                G * OG * taps * CG, OG, CG, taps};
            Access t = {0};
            reads(&t, weights_of(g, cl - 1));
            reads(&t, whole(&g->geo[cl]));
            writes(&t, whole(&g->wt));
            kernel(r, &t, SPG_KERNEL_wtrans, NULL, 0, &wp, sizeof wp, groups(wp.total, 256u), 1, 1);
            reads(&a, whole(&g->wt));
            reads(&a, whole(&g->geo[cl]));
            a.group = group_of(cl, GROUP_PHASES);
            for (uint32_t ph = 0; ph < info->phases; ph++) {
                SpgGemmPush p = {
                    .a = g->delta[cl].address, .b = at(&g->wt, (uint64_t)info->phase[ph].first * OG * CG),
                    .c = dst->address, .e0 = g->act[i].address, .geo = g->geo[cl].address + 4u * info->phase[ph].at,
                    .M = n * info->phase[ph].rh * info->phase[ph].rw, .N = CG, .K = info->phase[ph].taps * OG,
                    .ldb = CG, .ldc = x->channels, .a_group = OG, .b_group = taps * OG * CG, .c_group = CG,
                    .alpha = 1.0f, .beta = accumulate ? 1.0f : 0.0f,
                };
                SpgGemmMode m = {SPG_A_CONV, SPG_B_ROW, derive ? SPG_EPI_DERIV : SPG_EPI_STORE, fused, G,
                                 info->phases > 1, OG % 4u == 0 && s->channels % 4u == 0, true, 0,
                                 (uint64_t)n * net->topology[i], g->bf16};
                product(r, &a, &p, &m);
            }
            return derive;
        }
        case LAYER_BATCH_NORM: {    /* the coefficients were left by the backward sums */
            const uint32_t C = s->channels;
            reads(&a, span(g->bn[cl].address + 16u * C, 3u * C));
            eltwise(g, r, &a, SPG_ELT_BDATA, fused, dst->address, g->delta[cl].address, g->bn[cl].address + 16u * C,
                    g->act[i].address, (uint64_t)n * net->topology[cl], C);
            return true;
        }
        case LAYER_ADD:
        case LAYER_CONCAT: {
            uint32_t c0 = 0;
            for (uint32_t j = 0; s->type == LAYER_CONCAT && j < k; j++) c0 += net->shapes[in[j]].channels;
            SpgCombinePush p = {dst->address, g->delta[cl].address, g->act[i].address, n, s->height * s->width,
                                s->channels, c0, x->channels, accumulate ? 1u : 0u};
            uint32_t spec[3] = {SPG_COMBINE_SLICE, fused, 1};
            if (fused != ACT_NONE) reads(&a, whole(&g->act[i]));
            kernel(r, &a, SPG_KERNEL_combine, spec, 3, &p, sizeof p, groups((uint64_t)n * net->topology[i], 256u), 1, 1);
            return true;
        }
        case LAYER_GLOBAL_AVG_POOL: {
            SpgCombinePush p = {dst->address, g->delta[cl].address, g->act[i].address, n, x->height * x->width,
                                x->channels, 0, 0, accumulate ? 1u : 0u};
            uint32_t spec[3] = {SPG_COMBINE_GAP, fused, 1};
            if (fused != ACT_NONE) reads(&a, whole(&g->act[i]));
            kernel(r, &a, SPG_KERNEL_combine, spec, 3, &p, sizeof p, groups((uint64_t)n * net->topology[i], 256u), 1, 1);
            return true;
        }
        default: {                  /* pooling */
            SpgPoolPush p = {g->act[i].address, 0, g->delta[cl].address, dst->address, n, x->height, x->width,
                             x->channels, s->height, s->width, s->kernel_h, s->kernel_w, s->stride_h, s->stride_w,
                             s->pad_h, s->pad_w};
            uint32_t spec[3] = {s->type == LAYER_MAX_POOL2D, 1, fused};
            kernel(r, &a, SPG_KERNEL_pool, spec, 3, &p, sizeof p, groups((uint64_t)n * net->topology[i], 256u), 1, 1);
            return true;
        }
    }
}

/* The sums of batch normalization layer cl's backward pass: its parameters' gradients (scaled and
   added to the step's) and the coefficients of its data gradient. */
static void bn_backward(SpgGpuNet *g, Recorder *r, uint32_t cl, uint32_t n, float scale, float beta) {
    const NeuralNetwork *net = g->net;
    const uint32_t src = spingalett_source(net, cl);
    const LayerShape *s = &net->shapes[cl];
    const uint32_t C = s->channels, R = n * s->height * s->width, slices = (R + COLSUM_ROWS - 1u) / COLSUM_ROWS;
    uint32_t cols = C <= 16u ? 16u : C <= 32u ? 32u : 64u;
    SpgColsumPush cp = {g->act[src].address, g->delta[cl].address, g->bn[cl].address, g->part.address, R, C,
                        COLSUM_ROWS, 0};
    uint32_t spec[2] = {cols, SPG_COLSUM_DY};
    Access a = {0};
    reads(&a, whole(&g->act[src]));
    reads(&a, whole(&g->delta[cl]));
    reads(&a, span(g->bn[cl].address, C));
    writes(&a, span(g->part.address, 2ull * slices * C));
    kernel(r, &a, SPG_KERNEL_colsum, spec, 2, &cp, sizeof cp, (C + cols - 1) / cols, slices, 1);
    SpgBnPush bp = {
        .part = g->part.address, .gamma = weights_at(g, cl - 1), .stats = g->bn[cl].address,
        .coef = g->bn[cl].address + 16u * C, .ggamma = grad_weights_at(g, cl - 1), .gbeta = grad_biases_at(g, cl - 1),
        .C = C, .slices = slices, .m = (float)R, .scale = scale, .beta_g = beta,
    };
    uint32_t mode = SPG_BN_BACKWARD;
    Access b = {0};
    reads(&b, span(g->part.address, 2ull * slices * C));
    reads(&b, weights_of(g, cl - 1));
    reads(&b, span(g->bn[cl].address, 4u * C));
    if (beta != 0.0f) {
        reads(&b, span(grad_weights_at(g, cl - 1), C));
        reads(&b, span(grad_biases_at(g, cl - 1), C));
    }
    writes(&b, span(g->bn[cl].address + 16u * C, 3u * C));
    writes(&b, span(grad_weights_at(g, cl - 1), C));
    writes(&b, span(grad_biases_at(g, cl - 1), C));
    kernel(r, &b, SPG_KERNEL_bn, &mode, 1, &bp, sizeof bp, (C + 63u) / 64u, 1, 1);
}

/* grad = scale * (the chunk's gradient) + beta grad for dense or convolution layer l + 1, whose
   gradient delta[l + 1] is complete. */
static void weight_gradient(SpgGpuNet *g, Recorder *r, uint32_t l, uint32_t n, float scale, float beta) {
    const NeuralNetwork *net = g->net;
    const uint32_t src = spingalett_source(net, l + 1), out_sz = net->topology[l + 1];
    const LayerShape *s = &net->shapes[l + 1], *in = &net->shapes[src];
    SpgGemmPush p = {.c = grad_weights_at(g, l), .alpha = scale, .beta = beta};
    SpgGemmMode m = {SPG_A_COL, SPG_B_ROW, SPG_EPI_STORE, ACT_NONE, 1, false, true, true, 0, 0, g->bf16};
    uint32_t rows, M, N, K;
    Access a = {0};
    reads(&a, whole(&g->delta[l + 1]));
    reads(&a, whole(&g->act[src]));
    if (s->type == LAYER_DENSE) {
        /* gW[out x in] = delta^T act */
        M = out_sz; N = net->topology[src]; K = n; rows = n;
        p.a = g->delta[l + 1].address; p.lda = out_sz;
        p.b = g->act[src].address; p.ldb = N;
    } else {
        /* gW[g OG + f][window] = sum over output pixels of dy[pixel][g OG + f] x window[pixel] */
        m.groups = layer_groups(s);
        const uint32_t CG = in->channels / m.groups, OG = s->channels / m.groups;
        M = OG; N = s->kernel_h * s->kernel_w * CG; K = n * s->height * s->width; rows = K;
        p.a = g->delta[l + 1].address; p.lda = s->channels; p.a_group = OG;
        p.b = g->act[src].address; p.b_group = CG; p.geo = g->geo[l + 1].address;
        m.bmode = SPG_B_CONV;
        m.vec_b = CG % 4u == 0 && in->channels % 4u == 0;
        reads(&a, whole(&g->geo[l + 1]));
    }
    p.M = M; p.N = N; p.K = K; p.ldc = N; p.c_group = M * N;
    m.c_floats = (uint64_t)m.groups * M * N;
    const Range grad = span(grad_weights_at(g, l), weight_count(g, l));
    uint32_t slice_k, slices = spg_gemm_split(M, N, K, m.groups, &slice_k);
    if (slices > 1) {
        /* partial products per slice, added in order */
        const Range part = span(g->part.address, (uint64_t)slices * m.groups * M * N);
        p.c = g->part.address; p.slices = slices; p.slice_k = slice_k;
        m.epi = SPG_EPI_PARTIAL;
        writes(&a, part);
        product(r, &a, &p, &m);
        SpgReducePush rp = {g->part.address, grad_weights_at(g, l), m.groups * M * N, M * N, slices, scale, beta, 0};
        Access b = {0};
        reads(&b, part);
        if (beta != 0.0f) reads(&b, grad);
        writes(&b, grad);
        kernel(r, &b, SPG_KERNEL_reduce, NULL, 0, &rp, sizeof rp, groups(rp.total, 256u), 1, 1);
    } else {
        if (beta != 0.0f) reads(&a, grad);
        writes(&a, grad);
        product(r, &a, &p, &m);
    }
    column_sums(g, r, &g->delta[l + 1], rows, s->type == LAYER_DENSE ? out_sz : s->channels, grad_biases_at(g, l),
                scale, beta);
}

/* Propagates the output deltas back as batch_backprop_hidden() does, each layer's weight gradient
   recorded as soon as its delta is complete (next to the data gradient that reads it too). */
static void record_backward(SpgGpuNet *g, Recorder *r, uint32_t n, float scale, float beta) {
    const NeuralNetwork *net = g->net;
    const uint32_t last = net->layers - 1;
    memcpy(g->pending, g->uses, net->layers * sizeof(uint32_t));
    for (uint32_t cl = last; cl > 0; cl--) {
        const uint32_t count = spingalett_input_count(net, cl), *in = spingalett_inputs(net, cl);
        const LayerType type = net->shapes[cl].type;
        if (type == LAYER_DENSE || type == LAYER_CONV2D) weight_gradient(g, r, cl - 1, n, scale, beta);
        if (type == LAYER_BATCH_NORM) bn_backward(g, r, cl, n, scale, beta);
        for (uint32_t k = 0; k < count; k++) {
            const uint32_t l = in[k];
            if (l == 0 || g->fold[l] == cl) continue;   /* its gradient is the addition's own */
            const ActivationFunction act = net->act_func[l - 1];
            const uint64_t total = (uint64_t)n * net->topology[l];
            const bool masked = g->dmask[l].buffer != NULL;
            Access fix = {0};       /* the derivative, or the mask, applied to delta[l] */
            reads(&fix, whole(&g->delta[l]));
            reads(&fix, whole(masked ? &g->dmask[l] : &g->act[l]));
            writes(&fix, whole(&g->delta[l]));
            if (g->uses[l] > 1) {
                const bool first = g->pending[l] == g->uses[l], done = --g->pending[l] == 0;
                /* the first gradient is written, later ones added (in the kernel's last step: products
                   add in their epilogue, as the sums of the CPU run; batch normalization and pooling
                   through gtmp), the derivative applied with the last one where the kernel can */
                const bool direct = first || type == LAYER_DENSE || type == LAYER_CONV2D || type == LAYER_ADD ||
                                    type == LAYER_CONCAT || type == LAYER_GLOBAL_AVG_POOL;
                const uint32_t fused = direct && done && !masked ? act : ACT_NONE;
                bool applied = input_gradient(g, r, cl, k, direct ? &g->delta[l] : &g->gtmp, n, fused,
                                              direct && !first) && fused != ACT_NONE;
                if (!direct) {
                    Access add = {0};
                    reads(&add, whole(&g->delta[l]));
                    reads(&add, whole(&g->gtmp));
                    writes(&add, whole(&g->delta[l]));
                    eltwise(g, r, &add, SPG_ELT_ADD, ACT_NONE, g->delta[l].address, g->gtmp.address, 0, 0, total, 1);
                }
                if (done && !applied)
                    eltwise(g, r, &fix, masked ? SPG_ELT_MUL : SPG_ELT_DERIV, masked ? ACT_NONE : act,
                            g->delta[l].address, masked ? g->dmask[l].address : g->act[l].address, 0, 0, total, 1);
                continue;
            }
            if (input_gradient(g, r, cl, k, &g->delta[l], n, masked ? ACT_NONE : act, false) && !masked)
                continue;
            eltwise(g, r, &fix, masked ? SPG_ELT_MUL : SPG_ELT_DERIV, masked ? ACT_NONE : act, g->delta[l].address,
                    masked ? g->dmask[l].address : g->act[l].address, 0, 0, total, 1);
        }
    }
}

static void optimizer(SpgGpuNet *g, Recorder *r) {
    const NeuralNetwork *net = g->net;
    const SpgGpuTraining *o = &g->cfg;
    const uint64_t W = g->Wp, B = g->Bp;
    const Range grads = whole(&g->grads);
    if (o->max_grad_norm > 0.0f) {
        /* the gradients scaled to the maximum norm when above it (the padding is zero) */
        uint32_t slices = (uint32_t)((W + B + SUMSQ_SLICE - 1u) / SUMSQ_SLICE);
        SpgSumsqPush sp = {g->grads.address, at(&g->grads, W), g->part.address, g->scalars.address, (uint32_t)W,
                           (uint32_t)B, slices, o->max_grad_norm};
        uint32_t spec = SPG_SUMSQ_PARTIAL;
        Access a = {0};
        reads(&a, grads);
        writes(&a, span(g->part.address, slices));
        kernel(r, &a, SPG_KERNEL_sumsq, &spec, 1, &sp, sizeof sp, groups(slices, 1u), 1, 1);
        spec = SPG_SUMSQ_CLIP;
        Access b = {0};
        reads(&b, span(g->part.address, slices));
        writes(&b, whole(&g->scalars));
        kernel(r, &b, SPG_KERNEL_sumsq, &spec, 1, &sp, sizeof sp, 1, 1, 1);
        Access c = {0};
        reads(&c, grads);
        reads(&c, whole(&g->scalars));
        writes(&c, grads);
        eltwise(g, r, &c, SPG_ELT_SCALE, ACT_NONE, g->grads.address, 0, g->scalars.address, 0, W + B, 1);
    }
    uint32_t spec = o->optimizer;
    SpgOptimPush p = {0, 0, 0, 0, g->header.address, 0, 0.0f, o->momentum, o->beta1, o->beta2, o->epsilon};
    /* every layer's weights and the biases update independently (one group) */
    Access a = {.group = group_of(0, GROUP_OPTIMIZER)};
    reads(&a, grads);
    reads(&a, whole(&g->header));
    writes(&a, whole(&g->params));
    if (g->moment1.buffer) writes(&a, whole(&g->moment1));
    if (g->moment2.buffer) writes(&a, whole(&g->moment2));
    /* weights, with no decay for batch normalization's gamma; biases without decay */
    bool per_layer = false;
    for (uint32_t l = 1; o->decay != 0.0f && l < net->layers; l++)
        if (net->shapes[l].type == LAYER_BATCH_NORM) per_layer = true;
    for (uint32_t l = 0; l + 1 < net->layers; l++) {
        uint64_t w0 = per_layer ? g->woff[l] : 0, count = per_layer ? weight_count(g, l) : W;
        if (count > 0) {
            p.w = at(&g->params, w0); p.g = at(&g->grads, w0); p.n = (uint32_t)count;
            p.m = g->moment1.buffer ? at(&g->moment1, w0) : 0;
            p.v = g->moment2.buffer ? at(&g->moment2, w0) : 0;
            p.decay = per_layer && net->shapes[l + 1].type == LAYER_BATCH_NORM ? 0.0f : o->decay;
            kernel(r, &a, SPG_KERNEL_optim, &spec, 1, &p, sizeof p, groups(count, 256u), 1, 1);
        }
        if (!per_layer) break;
    }
    if (B > 0) {
        p.w = at(&g->params, W); p.g = at(&g->grads, W); p.n = (uint32_t)B; p.decay = 0.0f;
        p.m = g->moment1.buffer ? at(&g->moment1, W) : 0;
        p.v = g->moment2.buffer ? at(&g->moment2, W) : 0;
        kernel(r, &a, SPG_KERNEL_optim, &spec, 1, &p, sizeof p, groups(B, 256u), 1, 1);
    }
}

/* ------------------------------------------------------------------------- chunks */

/* The losses of a finished chunk, added in the CPU's order. */
static void harvest(SpgGpuNet *g, Slot *s) {
    if (!spg_gpu_wait(s->pending)) g->lost = true;
    s->pending = NULL;
    if (g->lost) {
        s->dest = NULL;
        return;
    }
    if (!s->train) {
        if (s->dest)
            memcpy(s->dest, (const char *)s->staging.mapped + s->outputs,
                   (size_t)s->n * g->net->topology[g->net->layers - 1] * sizeof(float));
        s->dest = NULL;
        return;
    }
    const float *losses = (const float *)((const char *)s->staging.mapped + s->losses);
    float sum = 0.0f;
    for (uint32_t k = 0; k < s->n; k++) sum += losses[k];
    g->step_loss += sum;
    if (s->last) {
        g->total_loss += g->step_loss;
        g->step_loss = 0.0f;
    }
}

/* The slot of the next chunk, its previous chunk finished: the older of the two in flight, so that the
   newer one keeps the device busy while the host fills the slot (chunks are harvested in order). */
static Slot *next_slot(SpgGpuNet *g) {
    Slot *s = &g->slots[g->next];
    if (s->pending) harvest(g, s);
    return s;
}

float *spingalett_gpu_chunk_inputs(SpgGpuNet *g, float **targets) {
    Slot *s = next_slot(g);
    if (targets) *targets = (float *)((char *)s->staging.mapped + s->targets);
    return (float *)((char *)s->staging.mapped + s->inputs);
}

/* Drops the slot's cache entry of commands whose recording failed; NULL. */
static SpgGpuCommands *forget(Slot *s, SpgGpuCommands *c) {
    for (uint32_t e = 0; e < s->used; e++)
        if (s->cache[e].commands == c) {
            memmove(s->cache + e, s->cache + e + 1, (s->used - e - 1) * sizeof s->cache[0]);
            s->used--;
            break;
        }
    spg_gpu_commands_free(c);
    return NULL;
}

/* The slot's command buffer for chunks of this kind, recorded on first use. */
static SpgGpuCommands *chunk_commands(SpgGpuNet *g, Slot *s, uint64_t key, uint32_t n, uint32_t count, bool first,
                                      bool last, bool train) {
    for (uint32_t e = 0; e < s->used; e++)
        if (s->cache[e].key == key) return s->cache[e].commands;
    SpgGpuCommands *c = NULL;
    if (s->used < sizeof s->cache / sizeof s->cache[0]) {
        c = spg_gpu_commands_create();
        if (!c) return NULL;
        s->cache[s->used].key = key;
        s->cache[s->used++].commands = c;
    } else {
        /* a rare kind of chunk: the oldest entry is recorded again */
        c = s->cache[0].commands;
        memmove(s->cache, s->cache + 1, (s->used - 1) * sizeof s->cache[0]);
        s->cache[s->used - 1].key = key;
        s->cache[s->used - 1].commands = c;
    }
    const NeuralNetwork *net = g->net;
    const uint32_t in = net->topology[0], out = net->topology[net->layers - 1];
    Recorder *r = spg_gpu_record_begin(c) ? (Recorder *)calloc(1, sizeof *r) : NULL;
    if (!r) return forget(s, c);
    r->c = c;
    /* the chunk's header, inputs and targets from staging into the slot's buffers, which the chunk in
       the other slot does not use: copied while it still runs; then everything after it */
    g->act[0] = s->input;
    g->header = s->header;
    g->targets = s->target;
    spg_gpu_copy(c, &s->staging, 0, &s->header, 0, SPG_STEP_HEADER * 4u);
    spg_gpu_copy(c, &s->staging, s->inputs, &s->input, 0, (size_t)n * in * 4u);
    if (train) spg_gpu_copy(c, &s->staging, s->targets, &s->target, 0, (size_t)n * out * 4u);
    barrier(r);
    record_forward(g, r, n, train);
    const uint32_t L = net->layers - 1;
    if (train) {
        const float scale = 1.0f / (float)count, beta = first ? 0.0f : 1.0f;
        SpgOutputPush p = {g->act[L].address, g->targets.address, g->delta[L].address, s->staging.address + s->losses,
                           n, out};
        uint32_t spec[3] = {SPG_OUT_LOSS, net->act_func[L - 1], net->loss_func};
        Access a = {0};
        reads(&a, whole(&g->act[L]));
        reads(&a, whole(&g->targets));
        writes(&a, whole(&g->delta[L]));
        writes(&a, span(s->staging.address + s->losses, n));
        kernel(r, &a, SPG_KERNEL_output, spec, 3, &p, sizeof p, (n + 63u) / 64u, 1, 1);
        record_backward(g, r, n, scale, beta);
        if (last) optimizer(g, r);
    } else {
        barrier(r);
        spg_gpu_copy(c, &g->act[L], 0, &s->staging, s->outputs, (size_t)n * out * 4u);
    }
    /* everything visible to the host (losses, outputs); the next chunk's copies need not wait */
    spg_gpu_barrier_host(c);
    free(r);
    return spg_gpu_record_end(c) ? c : forget(s, c);
}

bool spingalett_gpu_train_chunk(SpgGpuNet *g, uint32_t n, uint32_t count, uint32_t position, bool first, bool last,
                                const SpgGpuStep *step) {
    if (!g->training || g->lost || n == 0 || n > g->capacity) return false;
    Slot *s = &g->slots[g->next];
    uint32_t header[SPG_STEP_HEADER] = {0};
    memcpy(&header[0], &step->lr, 4);
    memcpy(&header[1], &step->m_factor, 4);
    memcpy(&header[2], &step->v_factor, 4);
    header[3] = (uint32_t)g->cfg.dropout_seed;
    header[4] = (uint32_t)(g->cfg.dropout_seed >> 32);
    header[5] = (uint32_t)step->step;
    header[6] = (uint32_t)(step->step >> 32);
    header[7] = position;
    /* the commands depend on n, on count (gradients scaled by 1 / count) and on where the chunk is in
       its step */
    const uint64_t key = (uint64_t)n | (uint64_t)count << 24 | (first ? 1ull << 56 : 0) | (last ? 1ull << 57 : 0);
    SpgGpuCommands *c = chunk_commands(g, s, key, n, count, first, last, true);
    if (!c) return false;
    memcpy(s->staging.mapped, header, sizeof header);
    if (!spg_gpu_submit(c)) return false;
    s->pending = c;
    s->n = n;
    s->train = true;
    s->last = last;
    g->next = 1 - g->next;
    return true;
}

bool spingalett_gpu_take_loss(SpgGpuNet *g, float *loss) {
    drain(g);
    *loss = g->total_loss;
    g->total_loss = 0.0f;
    return !g->lost;
}

bool spingalett_gpu_predict(SpgGpuNet *g, const float *inputs, float *outputs, uint32_t n) {
    const NeuralNetwork *net = g->net;
    const uint32_t in = net->topology[0], out = net->topology[net->layers - 1];
    /* chunk after chunk, each filled while the device runs the one before, its outputs taken when
       its slot comes round again */
    for (uint32_t start = 0; start < n; start += g->capacity) {
        const uint32_t m = n - start < g->capacity ? n - start : g->capacity;
        float *dst = spingalett_gpu_chunk_inputs(g, NULL);
        memcpy(dst, inputs + (size_t)start * in, (size_t)m * in * sizeof(float));
        Slot *s = &g->slots[g->next];
        SpgGpuCommands *c = chunk_commands(g, s, (uint64_t)m | 1ull << 63, m, m, false, false, false);
        if (!c || !spg_gpu_submit(c)) {
            drain(g);
            return false;
        }
        s->pending = c;
        s->n = m;
        s->train = false;
        s->dest = outputs + (size_t)start * out;
        g->next = 1 - g->next;
    }
    drain(g);
    return !g->lost;
}
