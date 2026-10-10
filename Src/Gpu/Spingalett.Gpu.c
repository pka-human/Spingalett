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
#include "Spingalett.Thread.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define MAX_GROUPS  65535u          /* workgroups in x of element-wise kernels, which loop beyond */
#define COLSUM_ROWS 1024u           /* rows per slice of colsum.comp */
#define SUMSQ_SLICE 4096u           /* sumsq.comp */
#define DW_RUN      4u              /* adjacent pixels a thread of dwconv.comp computes (its PX) */
#define DW_SUMS     128u            /* workgroups, at most, of a depthwise data gradient that sums (SUMS) */
#define SLOTS       3u              /* chunks in flight (Slot): the host fills one while two keep the device busy */

/* ------------------------------------------------------------------------- the network */

typedef struct {
    SpgGpuBuffer staging;           /* header | inputs | targets | losses | outputs | indices, host-visible */
    size_t inputs, targets, losses, outputs, indices;   /* their byte offsets */
    /* the chunk's header, inputs (the outputs of layer 0) and targets on the device: a slot's own,
       so that they are copied while the other slots' chunks run */
    SpgGpuBuffer header, input, target;
    float *host_inputs;             /* inputs kept as bfloat16: the host's floats, rounded on submission */
    bool staged;                    /* the chunk's inputs are on the device already (rounded or written) */
    uint32_t gathered;              /* the chunk's inputs (1), targets (2) gathered from data sets on the GPU */
    bool in_place;                  /* its inputs read where they are: rows first.. of the inputs' data set */
    uint32_t first;
    struct { uint64_t key, where; SpgGpuCommands *commands; } cache[16];  /* where: the inputs read in place */
    uint32_t used;                  /* entries of cache */
    SpgGpuCommands *pending;        /* submitted and not yet harvested */
    uint32_t n;                     /* samples of the pending chunk */
    bool train, last;               /* the pending chunk trained, and ended its step */
    float *dest;                    /* where a prediction's outputs go */
} Slot;

struct SpgGpuNet {
    SpgBackend backend;             /* the backend it was made on (its functions run there) */
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
    SpgGpuBuffer *geo;              /* per (transposed) conv layer: its geometries (spg_conv_geometry()) */
    SpgConvGeometry **conv;         /* and where they are */
    SpgGpuBuffer *bn;               /* per batch normalization layer: stats (4C) | coefficients (3C);
                                       per layer normalization layer (training): mean, 1 / std per cell */
    SpgGpuBuffer part;              /* partial sums: colsum slices, split products, norm slices */
    SpgGpuBuffer wt;                /* convolution weights regrouped for data gradients (conv_spread()) */
    SpgGpuBuffer transfer;          /* host-visible: parameters on their way in or out */
    SpgGpuBuffer wh;                /* the weights as bfloat16, which the products read in bfloat16 when
                                       the chunks are large enough to read them many times (written by
                                       the optimizer with the weights, and after uploads) */
    SpgGpuArena *arena;             /* the memory of all of these but the transfer buffer */
    size_t transfer_bytes;          /* its size, when it is made */
    SpgGpuCommands *once;           /* uploads and downloads */
    uint32_t *uses, *pending;       /* consumers of each layer, and those not yet back-propagated */
    uint8_t *half;                  /* outputs kept in memory as bfloat16 (kept_half()) */
    uint8_t *dhalf;                 /* and their gradients */
    /* batch normalization layers (no activation, no dropout) read only by an addition of two layers,
       which applies them as it adds (their own outputs are never stored, and their gradient is the
       addition's): fold[l] = that addition, or 0 */
    uint32_t *fold;
    /* batch normalization layers (no dropout) read only by a depthwise convolution of channels in fours,
       which applies them to the values it reads (dwconv.comp's PRO): their outputs are never stored, and
       the readers read their inputs (outputs_of()); pro[l] = that convolution, or 0 */
    uint32_t *pro;
    /* dense and convolution layers (no activation) read only by a batch normalization whose
       outputs are stored: in inference their products apply it in their epilogue (SPG_EPI_SCALE_ACT, the
       normalization's coefficients with the layer's biases folded in) and write the normalization's
       outputs, their own never stored; into[l] = that normalization, or 0 */
    uint32_t *into;
    /* the inference coefficients of every batch normalization (bn.comp's INFER and FOLD, at stats + 2C) are those
       of the parameters as they are: computed by an inference chunk, the chunks after it need not compute them
       again, until an upload, a training chunk or a pass writes the parameters or the statistics (ResNet-20 of
       Examples/Benchmark.c infers 1.03 times as fast without its 21 launches a chunk); recording: whether the
       commands being recorded compute them */
    bool coefficients, recording;

    Slot slots[SLOTS];
    uint32_t next;                  /* the slot of the next chunk */
    SpgGpuRows rows;                /* data sets on the GPU the training chunks' rows come from */
    uint64_t version;               /* the network's host_version whose parameters it has (the caller's) */
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

static inline uint64_t align8(uint64_t n) {
    return (n + 7u) & ~(uint64_t)7u;
}

/* The convolution the products of layer l compute: a convolution layer's own, or for a transposed
   convolution the one whose data gradient it is, from the transposed convolution's outputs to its
   inputs (its forward pass is that convolution's data gradient, its data gradient that
   convolution's forward pass, its weight gradient that convolution's with x and dy swapped). */
typedef struct {
    uint32_t in_h, in_w, in_c, out_h, out_w, out_c;     /* the convolution's input and output maps */
    uint32_t G, CG, OG, taps;                           /* groups, channels of a group in and out, taps */
} ConvView;

static ConvView conv_view(const NeuralNetwork *net, uint32_t l);

/* Whether layer l's convolution is depthwise (a group a channel, of fewer filters than a block of the
   matrix units), which dwconv.comp runs: products of a single channel (a group of all the filters, or
   blocks of filters of one channel) run faster on the matrix kernel. */
static bool depthwise(const NeuralNetwork *net, uint32_t l);

static ConvView conv_view(const NeuralNetwork *net, uint32_t l) {
    const LayerShape *x = &net->shapes[spingalett_source(net, l)], *s = &net->shapes[l];
    const bool transposed = s->type == LAYER_CONV_TRANSPOSE2D;
    const LayerShape *a = transposed ? s : x, *b = transposed ? x : s;
    ConvView v = {a->height, a->width, a->channels, b->height, b->width, b->channels, layer_groups(s), 0, 0,
                  s->kernel_h * s->kernel_w};
    v.CG = v.in_c / v.G;
    v.OG = v.out_c / v.G;
    return v;
}

static bool depthwise(const NeuralNetwork *net, uint32_t l) {
    const ConvView v = conv_view(net, l);
    return v.CG == 1u && v.G == v.in_c && v.G > 1u && v.OG < 16u && v.taps <= SPG_DW_TAPS;
}

/* The slices of a depthwise weight gradient (dwconv.comp's WEIGHTS) over the output pixels of n samples:
   the channels (four a thread with vec 4) of a workgroup of 256 threads, the threads a channel (lanes),
   and the units of a slice, some 32 pixels a thread (more when the slices would be more than a dispatch
   has). A unit is a pixel, or with vec 4 and windows of up to 9 taps a run of DW_RUN pixels of a row. */
typedef struct { uint32_t vec, per, lanes, units, rows, slices; } DwSlices;

static DwSlices dw_slices(const ConvView *v, uint64_t n) {
    DwSlices d;
    d.vec = v->OG == 1u && v->in_c % 4u == 0 ? 4u : 1u;
    const uint32_t run = d.vec == 4u && v->taps <= 9u ? DW_RUN : 1u;
    const uint64_t units = n * v->out_h * ((v->out_w + run - 1u) / run);
    const uint32_t quads = v->out_c / d.vec;
    d.per = quads < 256u ? quads : 256u;
    d.lanes = 256u / d.per;
    uint64_t rows = 32u / run * d.lanes;
    if ((units + rows - 1u) / rows > MAX_GROUPS) rows = (units + MAX_GROUPS - 1u) / MAX_GROUPS;
    d.units = (uint32_t)units;
    d.rows = (uint32_t)rows;
    d.slices = (uint32_t)((units + rows - 1u) / rows);
    return d;
}

bool spingalett_gpu_select(ComputeMode mode) {
    const SpgBackend backend = mode == COMPUTE_CUDA ? SPG_BACKEND_CUDA : SPG_BACKEND_VULKAN;
    spg_gpu_use(backend);
    return spg_gpu_built(backend) && spg_gpu_open();
}

const char *spingalett_gpu_name_of(ComputeMode mode) {
    const SpgBackend saved = spg_gpu_using();
    const char *name = spingalett_gpu_select(mode) ? spg_gpu_device_name() : NULL;
    spg_gpu_use(saved);
    return name;
}

bool spingalett_gpu_bf16_of(ComputeMode mode) {
    const SpgBackend saved = spg_gpu_using();
    const bool bf16 = spingalett_gpu_select(mode) && spg_gpu_mma_bf16();
    spg_gpu_use(saved);
    return bf16;
}

bool spingalett_gpu_available(void) {
    return spg_gpu_open();
}

bool spingalett_gpu_bf16(void) {
    return spg_gpu_open() && spg_gpu_mma_bf16();
}

const char *spingalett_gpu_name(void) {
    return spg_gpu_device_name();
}

bool spingalett_gpu_supports(const NeuralNetwork *net, const char **why) {
    for (uint32_t l = 0; l < net->layers; l++)
        if ((uint64_t)net->topology[l] * 64u > INT32_MAX) {
            if (why) *why = "layers of over 2^25 values a sample";
            return false;
        }
    for (uint32_t l = 1; l < net->layers; l++) {
        const LayerShape *s = &net->shapes[l];
        if (!spingalett_filters(s->type)) continue;
        if (s->kernel_h > 255u || s->kernel_w > 255u || net->shapes[spingalett_source(net, l)].channels > 65535u ||
            s->channels > 65535u) {
            if (why) *why = "convolution windows over 255 cells or layers of over 65535 channels";
            return false;
        }
        if (s->stride_h * s->stride_w > SPG_MAX_PHASES) {
            if (why) *why = "convolutions whose strides multiply to over 64";
            return false;
        }
    }
    return true;
}

/* Floats a sample takes on the device: outputs of every layer, and while training their gradients
   and dropout masks; the targets; and the inputs and targets of the slots beyond the second (the
   estimate of 1.0 counted those of one slot, its capacities stay where memory does not bound them). */
static uint64_t sample_floats(const NeuralNetwork *net, bool training) {
    uint64_t f = 0;
    for (uint32_t l = 0; l < net->layers; l++) {
        f += net->topology[l];
        if (training && l > 0) f += net->topology[l] * (net->dropout_rates[l] > 0.0f ? 2u : 1u);
    }
    const uint64_t out = net->topology[net->layers - 1];
    return f + out + (SLOTS - 2u) * (net->topology[0] + (training ? out : 0u));
}

/* Samples of a training chunk, at most: products of 4,096 rows run closer to the matrix units' rate than
   of 2,048, and a step takes half the submissions (full batches of the MLP of Examples/Benchmark.c train
   5% faster in bfloat16 and 6 to 8% in single precision on an RTX 4050 Laptop GPU; 8,192 are no faster).
   Batch normalization takes its statistics over each chunk: networks with it keep the CPU's chunks of
   SPINGALETT_BATCH_CHUNK, and the same groups. */
#define TRAINING_CHUNK 4096u

/* Bytes of the activations of an inference chunk, at most: a chunk whose layers' outputs stay in the
   GPU's cache from one layer to the next runs faster than a larger one (the U-Net of
   Examples/Benchmark.c infers 1.57 times as fast in chunks of 32 MB as in chunks of 2,048 images on an
   RTX 4050 Laptop GPU) and takes less memory to make. */
#define INFERENCE_BYTES (32ull << 20)

/* Bytes of a sample's activations in inference: bfloat16 but the output layer's where products are in
   bfloat16 (kept_half()). */
static uint64_t inference_bytes(const NeuralNetwork *net) {
    const bool half = spingalett_get_gpu_precision() == PRECISION_BFLOAT16 && spg_gpu_mma_bf16() && spg_gpu_bf16_storage();
    uint64_t bytes = 0;
    for (uint32_t l = 0; l < net->layers; l++) bytes += (uint64_t)net->topology[l] * (half && l + 1 < net->layers ? 2u : 4u);
    return bytes ? bytes : 1u;
}

uint32_t spingalett_gpu_capacity(const NeuralNetwork *net, uint32_t want, bool training) {
    uint64_t memory = spg_gpu_memory();
    if (memory == 0 || want == 0) return 0;
    /* half the device's memory for the chunk's buffers (the parameters and scratch are smaller) */
    uint64_t per = sample_floats(net, training) * 4u, cap = memory / 2u / (per ? per : 1u);
    uint32_t chunk = TRAINING_CHUNK;
    for (uint32_t l = 1; l < net->layers; l++)
        if (net->shapes[l].type == LAYER_BATCH_NORM) chunk = SPINGALETT_BATCH_CHUNK;
    if (training && cap > chunk) cap = chunk;
    /* inference in chunks that keep to the cache, of 64 samples at least */
    const uint64_t cached = INFERENCE_BYTES / inference_bytes(net);
    if (!training && cap > cached) cap = cached < 64u ? 64u : cached;
    /* a layer's values of a chunk indexed by kernels with signed 32-bit integers, as SPIR-V indexes */
    uint32_t widest = 1;
    for (uint32_t l = 0; l < net->layers; l++) widest = net->topology[l] > widest ? net->topology[l] : widest;
    if (cap > INT32_MAX / widest) cap = INT32_MAX / widest;
    return cap < want ? (uint32_t)cap : want;
}

static uint32_t spingalett_gpu_net_capacity_here(const SpgGpuNet *g) {
    return g->capacity;
}

uint32_t spingalett_gpu_net_capacity(const SpgGpuNet *g) {
    const SpgBackend saved = spg_gpu_using();
    if (g) spg_gpu_use(g->backend);
    uint32_t result = spingalett_gpu_net_capacity_here(g);
    spg_gpu_use(saved);
    return result;
}

static uint64_t spingalett_gpu_net_version_here(const SpgGpuNet *g) {
    return g->version;
}

uint64_t spingalett_gpu_net_version(const SpgGpuNet *g) {
    const SpgBackend saved = spg_gpu_using();
    if (g) spg_gpu_use(g->backend);
    uint64_t result = spingalett_gpu_net_version_here(g);
    spg_gpu_use(saved);
    return result;
}

static void spingalett_gpu_net_set_version_here(SpgGpuNet *g, uint64_t version) {
    g->version = version;
}

void spingalett_gpu_net_set_version(SpgGpuNet *g, uint64_t version) {
    const SpgBackend saved = spg_gpu_using();
    if (g) spg_gpu_use(g->backend);
    spingalett_gpu_net_set_version_here(g, version);
    spg_gpu_use(saved);
}

static bool spingalett_gpu_net_current_here(const SpgGpuNet *g) {
    return !g->lost && g->bf16 == (spingalett_get_gpu_precision() == PRECISION_BFLOAT16 && spg_gpu_mma_bf16());
}

bool spingalett_gpu_net_current(const SpgGpuNet *g) {
    /* (made on the backend the caller uses, too) */
    const SpgBackend saved = spg_gpu_using();
    if (g) spg_gpu_use(g->backend);
    bool result = g && g->backend == saved && spingalett_gpu_net_current_here(g);
    spg_gpu_use(saved);
    return result;
}

/* ------------------------------------------------------------------------- transfers */

/* The device keeps a transposed convolution's filters as those of the convolution whose data
   gradient it computes (one per input channel, conv_view()): wv[g IG + i][tap][o] = w[g OG + o][tap][i]
   for weight layer l, IG and OG its input and output channels a group. */
static void swap_filters(const NeuralNetwork *net, uint32_t l, float *host, float *dev, bool down) {
    const LayerShape *x = &net->shapes[spingalett_source(net, l + 1)], *s = &net->shapes[l + 1];
    const uint32_t G = layer_groups(s), IG = x->channels / G, OG = s->channels / G, taps = s->kernel_h * s->kernel_w;
    for (uint32_t k = 0; k < G; k++)
        for (uint32_t o = 0; o < OG; o++)
            for (uint32_t t = 0; t < taps; t++)
                for (uint32_t i = 0; i < IG; i++) {
                    const size_t h = (((size_t)k * OG + o) * taps + t) * IG + i;
                    const size_t d = (((size_t)k * IG + i) * taps + t) * OG + o;
                    if (down) host[h] = dev[d];
                    else dev[d] = host[h];
                }
}

/* A parameter array of the network (laid out by weight_offsets or bias_offsets) and its image, each
   layer's part padded to four floats (with zeros, written as the image is), one way or the other. */
static void pack(const SpgGpuNet *g, float *host, float *image, bool weights, bool down) {
    const NeuralNetwork *net = g->net;
    for (uint32_t l = 0; l + 1 < net->layers; l++) {
        uint64_t rows = spingalett_weight_rows(net, l), count = weights ? rows * spingalett_weight_row_len(net, l) : rows;
        uint64_t from = weights ? net->weight_offsets[l] : net->bias_offsets[l];
        uint64_t to = weights ? g->woff[l] : g->boff[l];
        if (count == 0) continue;
        if (weights && net->shapes[l + 1].type == LAYER_CONV_TRANSPOSE2D)
            swap_filters(net, l, host + from, image + to, down);
        else if (down) memcpy(host + from, image + to, count * sizeof(float));
        else memcpy(image + to, host + from, count * sizeof(float));
        /* the padding to the next layer's weights (biases), zeros in memory reused from other arenas too */
        for (uint64_t k = count; !down && k < (weights ? align8(count) : align4(count)); k++) image[to + k] = 0.0f;
    }
}

static void harvest(SpgGpuNet *g, Slot *s);

/* Waits for every chunk in flight, oldest first, and takes their losses. */
static void drain(SpgGpuNet *g) {
    for (uint32_t k = 0; k < SLOTS; k++) {
        Slot *s = &g->slots[(g->next + k) % SLOTS];
        if (s->pending) harvest(g, s);
    }
}

/*
 * Arrays between the host and the device go through the transfer buffer, as many in one submission
 * as it holds: an item is the first `bytes` of a device buffer, from or to the network's arrays w and
 * b (weights and biases of every layer, laid out by pack(); with mean and var, the parameters), or
 * the geometry of conv layer `layer` (uploads), or zeros (uploads: gradients start zeroed, their
 * padding never written), or (uploads) the same buffer of another copy of the network, copied on the
 * device.
 */
typedef struct {
    const SpgGpuBuffer *device;
    size_t bytes;
    enum { ITEM_ARRAYS, ITEM_GEOMETRY, ITEM_ZERO, ITEM_COPY } kind;
    float *w, *b, *mean, *var;
    uint32_t layer;
    const SpgGpuBuffer *from;
} Item;

static void write_geometry(const SpgGpuNet *g, uint32_t l, uint32_t *geo) {
    const LayerShape *s = &g->net->shapes[l];
    const ConvView v = conv_view(g->net, l);
    spg_conv_geometry(geo, g->conv[l], v.in_h, v.in_w, v.in_c, v.out_h, v.out_w, v.out_c, v.G, s->kernel_h,
                      s->kernel_w, s->stride_h, s->stride_w, s->pad_h, s->pad_w);
}

/* The item's image at `at`, written (upload) or read (download). */
static void image(SpgGpuNet *g, const Item *it, void *at, bool down) {
    if (it->kind == ITEM_ZERO) {
        memset(at, 0, it->bytes);
        return;
    }
    if (it->kind == ITEM_GEOMETRY) {
        write_geometry(g, it->layer, (uint32_t *)at);
        return;
    }
    float *base = (float *)at;
    pack(g, it->w, base, true, down);
    pack(g, it->b, base + g->Wp, false, down);
    if (it->mean) {
        pack(g, it->mean, base + g->Wp + g->Bp, false, down);
        pack(g, it->var, base + g->Wp + 2u * g->Bp, false, down);
    }
}

/* Whether the host writes the item's upload straight into the device's memory (zeros are filled by
   the device, which is faster than the host's writes over the bus). */
static bool direct(const Item *it, bool down) {
    return !down && it->device->mapped && it->kind != ITEM_ZERO && it->kind != ITEM_COPY;
}

/* Bytes of the transfer buffer the item takes: direct uploads, zeros and copies take none. */
static size_t staged(const Item *it, bool down) {
    if (direct(it, down) || (!down && (it->kind == ITEM_ZERO || it->kind == ITEM_COPY))) return 0;
    return (it->bytes + 15u) & ~(size_t)15u;
}

/* Runs the items, after all chunks submitted so far (which the caller has waited for when it
   uploads into memory the host writes), then with half the weights' bfloat16 copy from those uploaded.
   An upload is not waited for unless it copies from another network's buffers: the device runs what
   comes after it behind a barrier, and the next transfer waits for it before the host writes what it
   reads. */
static bool transfer(SpgGpuNet *g, const Item *items, uint32_t count, bool down, bool half) {
    SpgGpuCommands *c = g->once;
    if (!spg_gpu_wait(c)) return false;
    bool commands = half, staging = false, copies = false;
    for (uint32_t k = 0; k < count; k++) {
        const Item *it = &items[k];
        if (direct(it, down)) image(g, it, it->device->mapped, false);
        else commands = true;
        staging = staging || staged(it, down) > 0;
        copies = copies || it->kind == ITEM_COPY;
    }
    if (!commands) return true;
    /* the transfer buffer, made on first use where uploads need none */
    if (staging && !g->transfer.buffer && !spg_gpu_buffer_create(&g->transfer, g->transfer_bytes, true)) return false;
    for (uint32_t first = 0, last; first < count; first = last) {
        size_t used = 0;
        for (last = first; last < count && used + staged(&items[last], down) <= (staging ? g->transfer.size : 0u); last++)
            used += staged(&items[last], down);
        if (last == first) return false;
        char *base = (char *)g->transfer.mapped;
        size_t at = 0;
        for (uint32_t k = first; !down && k < last; at += staged(&items[k], down), k++)
            if (staged(&items[k], down)) image(g, &items[k], base + at, false);
        if (!spg_gpu_record_begin(c)) return false;
        spg_gpu_barrier(c);
        at = 0;
        for (uint32_t k = first; k < last; at += staged(&items[k], down), k++) {
            const Item *it = &items[k];
            if (direct(it, down)) continue;
            if (it->kind == ITEM_ZERO) spg_gpu_fill(c, it->device, 0, it->bytes, 0u);
            else if (it->kind == ITEM_COPY) spg_gpu_copy(c, it->from, 0, it->device, 0, it->bytes);
            else if (down) spg_gpu_copy(c, it->device, 0, &g->transfer, at, it->bytes);
            else spg_gpu_copy(c, &g->transfer, at, it->device, 0, it->bytes);
        }
        spg_gpu_barrier(c);
        const bool final = last == count;
        if (final && half) {
            SpgEltwisePush p = {g->wh.address, g->params.address, 0, 0, 0, (uint32_t)g->Wp, 1, 0, 0, 0.0f, 0, 0};
            const uint32_t spec[3] = {SPG_ELT_COPY, ACT_NONE, 1u};      /* y kept as bfloat16 */
            spg_gpu_dispatch(c, SPG_KERNEL_eltwise_h, spec, 3, &p, sizeof p, groups(g->Wp, 256u), 1, 1);
            spg_gpu_barrier(c);
        }
        if (!spg_gpu_record_end(c) || !spg_gpu_submit(c) || ((down || copies || !final) && !spg_gpu_wait(c)))
            return false;
        at = 0;
        for (uint32_t k = first; down && k < last; at += staged(&items[k], down), k++)
            image(g, &items[k], base + at, true);
    }
    return true;
}

static Item arrays(const SpgGpuBuffer *device, size_t floats, float *w, float *b) {
    return (Item){device, floats * sizeof(float), ITEM_ARRAYS, w, b, NULL, NULL, 0, NULL};
}

/* The parameters and moments to or from the device, and the gradients from it; when the network is
   made, the geometries of its conv layers and its gradients zeroed with them. With `from` (uploads),
   the parameters (and moments both have) come from that copy of the network on the device. */
static bool parameters(SpgGpuNet *g, bool down, bool creating, const SpgGpuNet *from) {
    NeuralNetwork *net = g->net;
    g->coefficients = false;
    const uint32_t L = g->layers;
    Item *items = (Item *)malloc((L + 4u) * sizeof(Item));
    if (!items) return false;
    uint32_t n = 0;
    for (uint32_t l = 0; creating && l < L; l++)
        if (g->conv[l])
            items[n++] = (Item){&g->geo[l], g->conv[l]->size * sizeof(uint32_t), ITEM_GEOMETRY, NULL, NULL, NULL, NULL, l,
                                NULL};
    items[n] = arrays(&g->params, g->Wp + 3u * g->Bp, net->weights, net->biases);
    items[n].mean = net->running_mean;
    items[n++].var = net->running_var;
    const size_t pair = g->Wp + g->Bp;
    if (g->grads.buffer && (down || creating)) {
        items[n++] = arrays(&g->grads, pair, net->grad_weights, net->grad_biases);
        if (!down) items[n - 1].kind = ITEM_ZERO;
    }
    if (g->moment1.buffer) items[n++] = arrays(&g->moment1, pair, net->opt_m_weights, net->opt_m_biases);
    if (g->moment2.buffer) items[n++] = arrays(&g->moment2, pair, net->opt_v_weights, net->opt_v_biases);
    /* a network that has taken no step has its moments zero: filled by the device when it is made */
    for (uint32_t k = n - (g->moment1.buffer != NULL) - (g->moment2.buffer != NULL); creating && !down && k < n; k++)
        if (net->time_step == 0) items[k].kind = ITEM_ZERO;
    for (uint32_t k = 0; from && !down && k < n; k++) {
        const SpgGpuBuffer *same = items[k].device == &g->params ? &from->params
                                 : items[k].device == &g->moment1 ? &from->moment1
                                 : items[k].device == &g->moment2 ? &from->moment2 : NULL;
        if (same && same->buffer && items[k].kind == ITEM_ARRAYS) {
            items[k].kind = ITEM_COPY;
            items[k].from = same;
        }
    }
    const bool ok = transfer(g, items, n, down, !down && g->wh.buffer);
    free(items);
    return ok;
}

static bool spingalett_gpu_upload_here(SpgGpuNet *g) {
    drain(g);
    return parameters(g, false, false, NULL);
}

bool spingalett_gpu_upload(SpgGpuNet *g) {
    const SpgBackend saved = spg_gpu_using();
    if (g) spg_gpu_use(g->backend);
    bool result = spingalett_gpu_upload_here(g);
    spg_gpu_use(saved);
    return result;
}

static bool spingalett_gpu_download_here(SpgGpuNet *g) {
    drain(g);
    /* the host's gradients and moments, made when they first come back */
    if ((g->grads.buffer || g->moment1.buffer || g->moment2.buffer) && !spingalett_training_state(g->net)) return false;
    return !g->lost && parameters(g, true, false, NULL);
}

bool spingalett_gpu_download(SpgGpuNet *g) {
    const SpgBackend saved = spg_gpu_using();
    if (g) spg_gpu_use(g->backend);
    bool result = spingalett_gpu_download_here(g);
    spg_gpu_use(saved);
    return result;
}

/* Whether two copies of a network lay its parameters out alike (the layout is the network's). */
static bool alike(const SpgGpuNet *g, const SpgGpuNet *from) {
    return from && from->net == g->net && from->Wp == g->Wp && from->Bp == g->Bp && !from->lost;
}

static bool spingalett_gpu_take_parameters_here(SpgGpuNet *g, SpgGpuNet *from) {
    if (g->backend != from->backend || !alike(g, from)) return false;
    drain(g);
    drain(from);
    return parameters(g, false, false, from);
}

bool spingalett_gpu_take_parameters(SpgGpuNet *g, SpgGpuNet *from) {
    const SpgBackend saved = spg_gpu_using();
    if (g) spg_gpu_use(g->backend);
    bool result = spingalett_gpu_take_parameters_here(g, from);
    spg_gpu_use(saved);
    return result;
}

/* ------------------------------------------------------------------------- creation */

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
        if (s->type == LAYER_BATCH_NORM && 2u * DW_SUMS * s->channels > need)       /* (dw_sums()) */
            need = 2u * DW_SUMS * s->channels;
        if (spingalett_filters(s->type) && depthwise(net, l)) {
            /* dwconv.comp's partial sums of the weight gradient, a slice of the output pixels each */
            const ConvView v = conv_view(net, l);
            const uint64_t floats = (uint64_t)dw_slices(&v, capacity).slices * v.out_c * v.taps;
            if (floats > need) need = floats;
            continue;
        }
        if (s->type == LAYER_DENSE || spingalett_filters(s->type)) {
            /* the partial products of split weight gradients, summed over the pixels of the
               convolution's output (conv_view()) */
            uint64_t w = (uint64_t)spingalett_weight_rows(net, l - 1) * spingalett_weight_row_len(net, l - 1);
            if (w > SPG_SPLIT_FLOATS / 2u) continue;
            if (s->type == LAYER_CONV_TRANSPOSE2D) {
                const LayerShape *x = &net->shapes[spingalett_source(net, l)];
                rows = (uint64_t)capacity * x->height * x->width;
            }
            uint32_t slice_k, split = spg_gemm_split(1, 1, (uint32_t)rows, 1, &slice_k);
            uint64_t floats = split * w < SPG_SPLIT_FLOATS ? split * w : SPG_SPLIT_FLOATS;
            if (floats > need) need = floats;
        }
    }
    return need;
}

/* Whether layer l's outputs are kept as bfloat16: with products in bfloat16, those of every layer but
   the output layer (and a softmax), the network's input among them (rounded by the host), and their
   gradients in training. Products round their operands to bfloat16 anyway; the passes between them
   (normalizations, pooling, additions, concatenations, upsampling, dropout) read and write bfloat16
   and compute in single precision, as PyTorch's autocast keeps activations and their gradients. The
   parameters' gradients (summed in single precision by the products) and the optimizer stay in
   single precision. */
static bool kept_half(const SpgGpuNet *g, uint32_t l) {
    const NeuralNetwork *net = g->net;
    if (!g->bf16 || !spg_gpu_bf16_storage() || l + 1 >= net->layers || g->uses[l] == 0 || g->fold[l]) return false;
    return l == 0 || net->act_func[l - 1] != ACT_SOFTMAX;
}

/* What the readers of layer l's outputs read: those outputs, or the input of a normalization that its
   depthwise reader applies (pro[l]). */
static const SpgGpuBuffer *outputs_of(const SpgGpuNet *g, uint32_t l) {
    return g->pro[l] ? &g->act[spingalett_source(g->net, l)] : &g->act[l];
}

/* The normalization depthwise convolution l applies to what it reads, or 0. */
static uint32_t applied_norm(const SpgGpuNet *g, uint32_t l) {
    const uint32_t src = spingalett_source(g->net, l);
    return g->pro[src] == l ? src : 0u;
}

/* The threads' work of dw_pass() in mode over n samples (vec: channels a thread): runs of DW_RUN pixels of
   a row, of one where they do not apply. */
static uint64_t dw_items(const SpgGpuNet *g, uint32_t l, uint32_t mode, uint32_t n, uint32_t vec) {
    const ConvView v = conv_view(g->net, l);
    const bool apply = mode == SPG_DW_APPLY;
    const uint32_t rows = apply ? v.out_h : v.in_h, width = apply ? v.out_w : v.in_w, C = apply ? v.out_c : v.in_c;
    const uint32_t run = vec == 4u && (apply || DW_RUN % g->net->shapes[l].stride_w == 0) ? DW_RUN : 1u;
    return (uint64_t)n * rows * ((width + run - 1u) / run) * (C / vec);
}

/* The workgroups whose sums (dwconv.comp's SUMS) give normalization bn's backward pass over n samples
   what colsum.comp would, or 0: the data gradient of the depthwise convolution that applies it (pro[]),
   with its derivative, sums as it computes the normalization's gradient. The convolution follows the
   normalization, so that nothing between that gradient and the normalization's backward pass uses part;
   a thread keeps one group of four channels (256 a multiple of channels / 4). */
static uint32_t dw_sums(const SpgGpuNet *g, uint32_t bn, uint32_t n) {
    const NeuralNetwork *net = g->net;
    if (!g->training || !g->pro[bn] || g->pro[bn] != bn + 1u || net->act_func[bn - 1] == ACT_NONE ||
        256u % (net->shapes[bn].channels / 4u) != 0)
        return 0;
    const uint32_t wg = groups(dw_items(g, bn + 1u, SPG_DW_SPREAD, n, 4u), 256u);
    return wg < DW_SUMS ? wg : DW_SUMS;
}

static bool spingalett_gpu_net_reuse_here(SpgGpuNet *g, uint32_t capacity, const SpgGpuTraining *training) {
    const OptimizerType o = training->optimizer;
    if (!g->training || g->lost || g->capacity != capacity || !spingalett_gpu_net_current(g) ||
        ((o == OPTIMIZER_MOMENTUM || o == OPTIMIZER_ADAM || o == OPTIMIZER_ADAMW) && !g->moment1.buffer) ||
        ((o == OPTIMIZER_RMSPROP || o == OPTIMIZER_ADAM || o == OPTIMIZER_ADAMW) && !g->moment2.buffer))
        return false;
    drain(g);
    /* the settings are recorded in the commands of the optimizer (the dropout seed goes in the header) */
    const SpgGpuTraining *old = &g->cfg;
    if (old->optimizer != o || old->decay != training->decay || old->momentum != training->momentum ||
        old->beta1 != training->beta1 || old->beta2 != training->beta2 || old->epsilon != training->epsilon ||
        old->max_grad_norm != training->max_grad_norm)
        for (uint32_t k = 0; k < SLOTS; k++) {
            for (uint32_t e = 0; e < g->slots[k].used; e++) spg_gpu_commands_free(g->slots[k].cache[e].commands);
            g->slots[k].used = 0;
        }
    g->cfg = *training;
    g->step_loss = g->total_loss = 0.0f;
    for (uint32_t k = 0; k < SLOTS; k++) {
        g->slots[k].staged = g->slots[k].in_place = false;
        g->slots[k].gathered = 0;
    }
    return true;
}

bool spingalett_gpu_net_reuse(SpgGpuNet *g, uint32_t capacity, const SpgGpuTraining *training) {
    const SpgBackend saved = spg_gpu_using();
    if (g) spg_gpu_use(g->backend);
    bool result = spingalett_gpu_net_reuse_here(g, capacity, training);
    spg_gpu_use(saved);
    return result;
}

static void spingalett_gpu_net_free_here(SpgGpuNet *g) {
    if (!g) return;
    drain(g);
    for (uint32_t k = 0; k < SLOTS; k++)
        for (uint32_t e = 0; e < g->slots[k].used; e++) spg_gpu_commands_free(g->slots[k].cache[e].commands);
    spg_gpu_commands_free(g->once);
    spg_gpu_arena_free(g->arena);       /* every buffer but */
    spg_gpu_buffer_free(&g->transfer);
    for (uint32_t k = 0; k < SLOTS; k++) spingalett_aligned_free(g->slots[k].host_inputs);
    for (uint32_t l = 0; g->conv && l < g->layers; l++) free(g->conv[l]);
    free(g->act); free(g->delta); free(g->dmask); free(g->geo); free(g->conv); free(g->bn);
    free(g->uses); free(g->pending); free(g->woff); free(g->boff); free(g->fold); free(g->pro); free(g->into); free(g->half); free(g->dhalf);
    free(g);
    spg_gemm_release();
}

void spingalett_gpu_net_free(SpgGpuNet *g) {
    const SpgBackend saved = spg_gpu_using();
    if (g) spg_gpu_use(g->backend);
    spingalett_gpu_net_free_here(g);
    spg_gpu_use(saved);
}

SpgGpuNet *spingalett_gpu_net_create(NeuralNetwork *net, uint32_t capacity, const SpgGpuTraining *training) {
    return spingalett_gpu_net_create_from(net, capacity, training, NULL);
}

SpgGpuNet *spingalett_gpu_net_create_from(NeuralNetwork *net, uint32_t capacity, const SpgGpuTraining *training,
                                          SpgGpuNet *from) {
    if (!spg_gpu_open() || capacity == 0) return NULL;
    if (from && from->backend != spg_gpu_using()) from = NULL;      /* (the parameters from the host then) */
    SpgGpuNet *g = (SpgGpuNet *)calloc(1, sizeof *g);
    if (g) g->backend = spg_gpu_using();
    if (!g) return NULL;
    const uint32_t L = net->layers;
    g->net = net;
    g->capacity = capacity;
    g->layers = L;
    g->training = training != NULL;
    g->version = UINT64_MAX;
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
    g->pro = (uint32_t *)calloc(L, sizeof(uint32_t));
    g->into = (uint32_t *)calloc(L, sizeof(uint32_t));
    g->half = (uint8_t *)calloc(L, 1);
    g->dhalf = (uint8_t *)calloc(L, 1);
    if (!g->act || !g->delta || !g->dmask || !g->geo || !g->conv || !g->bn || !g->uses || !g->pending ||
        !g->woff || !g->boff || !g->fold || !g->pro || !g->into || !g->half || !g->dhalf)
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
    for (uint32_t c = 2; c < L; c++) {
        const uint32_t l = spingalett_source(net, c);
        if (net->shapes[c].type == LAYER_CONV2D && depthwise(net, c) && conv_view(net, c).OG == 1u &&
            net->shapes[l].channels % 4u == 0 && net->shapes[l].type == LAYER_BATCH_NORM && g->uses[l] == 1 &&
            net->dropout_rates[l] == 0.0f && net->act_func[l - 1] != ACT_SOFTMAX && !g->fold[l])
            g->pro[l] = c;
    }
    for (uint32_t l = 2; l < L; l++) {
        const uint32_t c = spingalett_source(net, l);
        const LayerType type = net->shapes[c].type;
        if (net->shapes[l].type == LAYER_BATCH_NORM && !g->fold[l] && !g->pro[l] && c > 0 && g->uses[c] == 1 &&
            net->act_func[c - 1] == ACT_NONE && (type == LAYER_DENSE || type == LAYER_CONV2D))
            g->into[c] = l;
    }
    for (uint32_t l = 0; l < L; l++) {     /* (a pro[] layer has no outputs, but a gradient as any other) */
        g->half[l] = kept_half(g, l) && !g->pro[l];
        g->dhalf[l] = training && l > 0 && kept_half(g, l);
    }
    /* every layer's weights at a multiple of eight floats (a bfloat16 copy's start at 16 bytes), its
       biases at one of four: the same layout in every copy of the network */
    const bool wh = g->bf16 && spg_gpu_bf16_storage() && capacity >= 256u;
    for (uint32_t l = 0; l + 1 < L; l++) {
        uint64_t rows = spingalett_weight_rows(net, l), count = rows * spingalett_weight_row_len(net, l);
        g->woff[l] = g->Wp;
        g->boff[l] = g->Bp;
        g->Wp += align8(count);
        g->Bp += align4(rows);
    }

    /* every buffer asked for, then made at once in a few allocations; what the host writes (the
       parameters and moments it uploads, the geometries, each chunk's header and inputs) straight
       into device memory where it can */
    SpgGpuArena *A = g->arena = spg_gpu_arena_create();
    if (!A) goto fail;
    const SpgMemory written = spg_gpu_host_writes() ? SPG_MEMORY_HOST_WRITES : SPG_MEMORY_DEVICE;
    const uint64_t P = g->Wp + 3u * g->Bp, pair = g->Wp + g->Bp;
    const uint32_t out = net->topology[L - 1];
    spg_gpu_arena_add(A, &g->params, P * 4u, written);
    if (wh) spg_gpu_arena_add(A, &g->wh, g->Wp * 2u, SPG_MEMORY_DEVICE);
    spg_gpu_arena_add(A, &g->scalars, 16u, SPG_MEMORY_DEVICE);
    spg_gpu_arena_add(A, &g->part, part_floats(net, capacity) * 4u, SPG_MEMORY_DEVICE);
    uint64_t widest_shared = 0, wt = 0, geometries = 0, widest_geometry = 0;
    for (uint32_t l = 0; l < L; l++) {
        const LayerShape *s = &net->shapes[l];
        const uint64_t rows = (uint64_t)capacity * net->topology[l] * 4u;
        if (l > 0 && !g->fold[l] && !g->pro[l] && (training || !g->into[l]))
            spg_gpu_arena_add(A, &g->act[l], g->half[l] ? rows / 2u : rows, SPG_MEMORY_DEVICE);
        if (training && l > 0 && !g->fold[l])
            spg_gpu_arena_add(A, &g->delta[l], g->dhalf[l] ? rows / 2u : rows, SPG_MEMORY_DEVICE);
        if (training && l > 0 && l + 1 < L && net->dropout_rates[l] > 0.0f)
            spg_gpu_arena_add(A, &g->dmask[l], rows, SPG_MEMORY_DEVICE);
        if (l > 0 && g->uses[l] > 1 && net->topology[l] > widest_shared) widest_shared = net->topology[l];
        if (spingalett_filters(s->type)) {
            const ConvView v = conv_view(net, l);
            SpgConvGeometry *info = g->conv[l] = (SpgConvGeometry *)malloc(sizeof *info);
            if (!info || !spg_conv_geometry(NULL, info, v.in_h, v.in_w, v.in_c, v.out_h, v.out_w, v.out_c, v.G,
                                            s->kernel_h, s->kernel_w, s->stride_h, s->stride_w, s->pad_h, s->pad_w))
                goto fail;
            const uint64_t bytes = ((uint64_t)info->size * sizeof(uint32_t) + 15u) & ~15ull;
            spg_gpu_arena_add(A, &g->geo[l], bytes, written);
            geometries += bytes;
            if (bytes > widest_geometry) widest_geometry = bytes;
            /* filters regrouped for data gradients, and for the forward passes of transposed convolutions */
            uint64_t w = (uint64_t)spingalett_weight_rows(net, l - 1) * spingalett_weight_row_len(net, l - 1);
            if ((training || s->type == LAYER_CONV_TRANSPOSE2D) && w > wt) wt = w;
        }
        if (s->type == LAYER_BATCH_NORM) spg_gpu_arena_add(A, &g->bn[l], 7u * s->channels * 4u, SPG_MEMORY_DEVICE);
        /* layer normalization: each cell's mean and 1 / std for the backward pass */
        if (training && s->type == LAYER_LAYER_NORM)
            spg_gpu_arena_add(A, &g->bn[l], 2ull * capacity * s->height * s->width * 4u, SPG_MEMORY_DEVICE);
    }
    spg_gpu_arena_add(A, &g->wt, wt * 4u, SPG_MEMORY_DEVICE);
    uint32_t moments = 0;
    if (training) {
        spg_gpu_arena_add(A, &g->grads, pair * 4u, SPG_MEMORY_DEVICE);
        spg_gpu_arena_add(A, &g->gtmp, (uint64_t)capacity * widest_shared * 4u, SPG_MEMORY_DEVICE);
        OptimizerType o = training->optimizer;
        if (o == OPTIMIZER_MOMENTUM || o == OPTIMIZER_ADAM || o == OPTIMIZER_ADAMW) {
            spg_gpu_arena_add(A, &g->moment1, pair * 4u, written);
            moments++;
        }
        if (o == OPTIMIZER_RMSPROP || o == OPTIMIZER_ADAM || o == OPTIMIZER_ADAMW) {
            spg_gpu_arena_add(A, &g->moment2, pair * 4u, written);
            moments++;
        }
    }
    /* the transfer buffer (made when first needed): every array of an upload or download at once, up
       to 64 MB, and at least the largest (parameters and geometries go through it whole) */
    const uint64_t up = geometries + (P + moments * pair) * 4u, down = (P + (moments + (training != NULL)) * pair) * 4u;
    uint64_t room = up > down ? up : down;
    if (room > (64u << 20)) room = 64u << 20;
    if (room < P * 4u) room = P * 4u;
    if (room < widest_geometry) room = widest_geometry;
    g->transfer_bytes = room;
    /* staging: header, inputs (when the host cannot write them straight into the device's memory),
       targets, losses and outputs of a chunk, 16-byte aligned; inputs kept as bfloat16 come from
       floats of the host's */
    const uint64_t in = net->topology[0], input_bytes = (uint64_t)capacity * in * (g->half[0] ? 2u : 4u);
    for (uint32_t k = 0; k < SLOTS; k++) {
        Slot *s = &g->slots[k];
        if (g->half[0] && !(s->host_inputs = (float *)spingalett_aligned_alloc((size_t)capacity * in * sizeof(float))))
            goto fail;
        s->inputs = SPG_STEP_HEADER * 4u;
        s->targets = s->inputs + (written == SPG_MEMORY_DEVICE ? (input_bytes + 15u) & ~15ull : 0u);
        s->losses = s->targets + (((uint64_t)capacity * out * 4u + 15u) & ~15ull);
        s->outputs = s->losses + (((uint64_t)capacity * 4u + 15u) & ~15ull);
        s->indices = s->outputs + (((uint64_t)capacity * out * 4u + 15u) & ~15ull);
        spg_gpu_arena_add(A, &s->staging, s->indices + (uint64_t)capacity * 4u, SPG_MEMORY_HOST);
        spg_gpu_arena_add(A, &s->header, SPG_STEP_HEADER * 4u, written);
        spg_gpu_arena_add(A, &s->input, input_bytes, written);
        if (training) spg_gpu_arena_add(A, &s->target, (uint64_t)capacity * out * 4u, SPG_MEMORY_DEVICE);
    }
    if (!spg_gpu_arena_commit(A) || !(g->once = spg_gpu_commands_create())) goto fail;
    for (uint32_t l = 1; training && l < L; l++)
        if (g->fold[l]) g->delta[l] = g->delta[g->fold[l]];     /* the addition's gradient, shared */
    drain(g);
    if (from) drain(from);
    if (parameters(g, false, true, alike(g, from) ? from : NULL)) return g;
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
    const SpgGpuNet *g;
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

/* Whether address is in the output (or the gradient) of a layer kept as bfloat16. */
static bool in_half(const SpgGpuNet *g, uint64_t address) {
    for (uint32_t l = 0; address && l < g->layers; l++) {
        if (g->half[l] && address >= g->act[l].address && address < g->act[l].address + g->act[l].size) return true;
        if (g->dhalf[l] && address >= g->delta[l].address && address < g->delta[l].address + g->delta[l].size)
            return true;
    }
    return address && g->wh.buffer && address >= g->wh.address && address < g->wh.address + g->wh.size;
}

/* The kernels with variants that keep activations as bfloat16 (SPG_KERNEL_H): the push constant words
   that may hold an activation's address (half.glsl), and the constant_id of their HALF. */
static const struct { uint16_t words; uint8_t id; } half_words[SPG_KERNEL_COUNT] = {
    [SPG_KERNEL_colsum] = {0x7, 2}, [SPG_KERNEL_bn] = {0x2, 1}, [SPG_KERNEL_eltwise] = {0x10F, 2},
    [SPG_KERNEL_pool] = {0xF, 3}, [SPG_KERNEL_combine] = {0x7, 3}, [SPG_KERNEL_upsample] = {0xF, 3},
    [SPG_KERNEL_ln] = {0x7, 3}, [SPG_KERNEL_optim] = {0x100, 1}, [SPG_KERNEL_rows] = {0x2, 2},
    [SPG_KERNEL_dwconv] = {0xD, 11},
};

/* Records a kernel after the barrier it needs; its bfloat16 variant, told which buffers are such, when
   it reads or writes an activation kept as bfloat16 (every constant before HALF given). */
static void kernel(Recorder *r, const Access *a, SpgKernel k, const uint32_t *spec, uint32_t spec_count,
                   const void *push, uint32_t push_size, uint32_t gx, uint32_t gy, uint32_t gz) {
    need(r, a);
    uint32_t mask = 0;
    for (uint32_t w = 0; r->g && w < 16u; w++)
        if ((half_words[k].words >> w & 1u) && 8u * w + 8u <= push_size) {
            uint64_t address;
            memcpy(&address, (const char *)push + 8u * w, sizeof address);
            if (in_half(r->g, address)) mask |= 1u << w;
        }
    if (mask && spec_count == half_words[k].id && spec_count < SPG_SPEC_MAX) {
        uint32_t with[SPG_SPEC_MAX];
        if (spec_count) memcpy(with, spec, spec_count * sizeof(uint32_t));
        with[spec_count] = mask;
        spg_gpu_dispatch(r->c, (SpgKernel)(k + 1), with, spec_count + 1u, push, push_size, gx, gy, gz);
        return;
    }
    spg_gpu_dispatch(r->c, k, spec, spec_count, push, push_size, gx, gy, gz);
}

static void product(Recorder *r, const Access *a, SpgGemmPush *p, const SpgGemmMode *m) {
    need(r, a);
    /* operands and results kept as bfloat16 (only with products in bfloat16) */
    SpgGemmMode mode = *m;
    mode.half = 0;
    if (m->bf16 && r->g) {
        const SpgGpuNet *g = r->g;
        mode.half = (in_half(g, p->a) ? 1u : 0u) | (in_half(g, p->b) ? 2u : 0u) |
                    (m->epi != SPG_EPI_PARTIAL && in_half(g, p->c) ? 4u : 0u) |
                    (m->epi == SPG_EPI_DERIV && in_half(g, p->e0) ? 8u : 0u);
    }
    /* the matrix units multiply blocks of 16 x 16 (32 values of k a step): products smaller than that in
       a dimension would mostly multiply padding, and stay in single precision unless they read or write
       bfloat16 (a rule of the network's shape, so that a product always runs the same way) */
    mode.bf16 = m->bf16 && (mode.half || (p->K >= 32u && p->M >= 16u && p->N >= 16u));
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
/* The weights the products read: the bfloat16 copy where the network keeps one. */
static uint64_t mm_weights_at(const SpgGpuNet *g, uint32_t l) {
    return g->wh.buffer ? g->wh.address + 2u * g->woff[l] : weights_at(g, l);
}
static Range mm_weights_of(const SpgGpuNet *g, uint32_t l) {
    if (!g->wh.buffer) return weights_of(g, l);
    return (Range){mm_weights_at(g, l), mm_weights_at(g, l) + 2u * weight_count(g, l)};
}
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

/* ---- convolutions (conv_view()) ---- */

/* y = act(x convolved + layer l's biases) with SPG_EPI_BIAS_ACT; y = act(x convolved times e0 plus the
   floats after them, a pair per output channel) with SPG_EPI_SCALE_ACT; otherwise y = (beta = 1: +=) x
   convolved, times act'(e0) with SPG_EPI_DERIV (e0 laid out like y). Over n samples, x and y the
   maps of the convolution's input and output: a convolution's forward pass, a transposed
   convolution's data gradient. The access lists what the caller reads and writes besides. */
/* A depthwise convolution's pass of n samples on dwconv.comp (mode SPG_DW_APPLY or SPG_DW_SPREAD), with the
   epilogues of conv_apply() and conv_spread(): from x (the gradient with SPREAD) into y. */
static void dw_pass(SpgGpuNet *g, Recorder *r, Access *a, uint32_t l, uint32_t mode, const SpgGpuBuffer *x,
                    const SpgGpuBuffer *y, uint32_t n, uint32_t epi, uint32_t act, uint64_t e0, float beta) {
    const ConvView v = conv_view(g->net, l);
    const LayerShape *s = &g->net->shapes[l];
    const bool bias = epi == SPG_EPI_BIAS_ACT;
    /* four channels a thread where they come in fours, each its own (og 1), of runs of DW_RUN pixels of a
       row: output pixels (APPLY), or input pixels where the run is a multiple of the stride (SPREAD) */
    const uint32_t vec = v.OG == 1u && v.in_c % 4u == 0 ? 4u : 1u;
    const bool apply = mode == SPG_DW_APPLY;
    /* the normalization the convolution applies: to x of its forward pass, to e0 of its data gradient's
       derivative, which may also sum for the normalization's backward pass (over `sums` workgroups) */
    const uint32_t bn = apply || epi == SPG_EPI_DERIV ? applied_norm(g, l) : 0u;
    const uint32_t sums = !apply && bn ? dw_sums(g, bn, n) : 0u;
    SpgDwconvPush p = {x->address, weights_at(g, l - 1), y->address, bias ? biases_at(g, l - 1) : e0,
                       bn ? g->bn[bn].address : 0, sums ? g->part.address : 0, (uint32_t)dw_items(g, l, mode, n, vec),
                       v.in_h, v.in_w, v.in_c, v.out_h, v.out_w, v.out_c, v.OG, s->pad_h, 0, 0, 0, beta};
    const uint32_t spec[11] = {mode, epi, act, s->kernel_h, s->kernel_w, s->stride_h, s->stride_w, vec, s->pad_w,
                               bn ? 1u + g->net->act_func[bn - 1] : 0u, sums ? 1u : 0u};
    if (bn) reads(a, span(g->bn[bn].address, 4u * v.in_c));
    if (epi == SPG_EPI_SCALE_ACT) reads(a, span(e0, 2ull * v.out_c));
    if (sums) writes(a, span(g->part.address, 2ull * sums * v.in_c));
    reads(a, whole(x));
    reads(a, weights_of(g, l - 1));
    if (bias) reads(a, biases_of(g, l - 1));
    if (beta != 0.0f) reads(a, whole(y));
    writes(a, whole(y));
    kernel(r, a, SPG_KERNEL_dwconv, spec, 11, &p, sizeof p, sums ? sums : groups(p.total, 256u), 1, 1);
}

static void conv_apply(SpgGpuNet *g, Recorder *r, Access *a, uint32_t l, const SpgGpuBuffer *x, const SpgGpuBuffer *y,
                       uint32_t n, uint32_t epi, uint32_t act, uint64_t e0, float beta) {
    if (depthwise(g->net, l)) {
        dw_pass(g, r, a, l, SPG_DW_APPLY, x, y, n, epi, act, e0, beta);
        return;
    }
    const ConvView v = conv_view(g->net, l);
    const uint32_t K = v.taps * v.CG;
    const bool bias = epi == SPG_EPI_BIAS_ACT;
    SpgGemmPush p = {
        .a = x->address, .b = mm_weights_at(g, l - 1), .c = y->address, .e0 = bias ? biases_at(g, l - 1) : e0,
        .e1 = epi == SPG_EPI_SCALE_ACT ? e0 + 4ull * v.out_c : 0,
        .geo = g->geo[l].address, .M = n * v.out_h * v.out_w, .N = v.OG, .K = K, .ldb = K, .ldc = v.out_c,
        .a_group = v.CG, .b_group = v.OG * K, .c_group = v.OG, .alpha = 1.0f, .beta = beta,
        .flags = bias ? SPG_GEMM_BIAS : 0u,
    };
    SpgGemmMode m = {SPG_A_CONV, SPG_B_COL, epi, act, v.G, false, v.CG % 4u == 0 && v.in_c % 4u == 0, true, 0,
                     (uint64_t)n * v.out_h * v.out_w * v.out_c, g->bf16};
    m.wide_a = v.CG % 8u == 0 && v.in_c % 8u == 0;
    m.wide_b = true;
    reads(a, whole(x));
    reads(a, mm_weights_of(g, l - 1));
    if (bias) reads(a, biases_of(g, l - 1));
    if (epi == SPG_EPI_SCALE_ACT) reads(a, span(e0, 2ull * v.out_c));
    reads(a, whole(&g->geo[l]));
    writes(a, whole(y));
    product(r, a, &p, &m);
}

/* The transpose of conv_apply(), with its epilogues: dx (the convolution's input maps) from dy (its
   output maps) through the filters regrouped by phase (wtrans.comp), a stride-1 product per phase of
   the stride over the pixels it holds (one group: they write different pixels; a phase no tap
   reaches gets its epilogue alone). A convolution's data gradient, a transposed convolution's
   forward pass. */
static void conv_spread(SpgGpuNet *g, Recorder *r, Access *a, uint32_t l, const SpgGpuBuffer *dy,
                        const SpgGpuBuffer *dx, uint32_t n, uint32_t epi, uint32_t act, uint64_t e0, float beta) {
    if (depthwise(g->net, l)) {
        dw_pass(g, r, a, l, SPG_DW_SPREAD, dy, dx, n, epi, act, e0, beta);
        return;
    }
    const ConvView v = conv_view(g->net, l);
    const SpgConvGeometry *info = g->conv[l];
    const bool bias = epi == SPG_EPI_BIAS_ACT;
    SpgWtransPush wp = {weights_at(g, l - 1), g->wt.address, g->geo[l].address + 4u * info->order,
                        v.G * v.OG * v.taps * v.CG, v.OG, v.CG, v.taps};
    Access t = {0};
    reads(&t, weights_of(g, l - 1));
    reads(&t, whole(&g->geo[l]));
    writes(&t, whole(&g->wt));
    kernel(r, &t, SPG_KERNEL_wtrans, NULL, 0, &wp, sizeof wp, groups(wp.total, 256u), 1, 1);
    reads(a, whole(dy));
    reads(a, whole(&g->wt));
    reads(a, whole(&g->geo[l]));
    if (bias) reads(a, biases_of(g, l - 1));
    writes(a, whole(dx));
    a->group = group_of(l, GROUP_PHASES);
    for (uint32_t ph = 0; ph < info->phases; ph++) {
        SpgGemmPush p = {
            .a = dy->address, .b = at(&g->wt, (uint64_t)info->phase[ph].first * v.OG * v.CG), .c = dx->address,
            .e0 = bias ? biases_at(g, l - 1) : e0, .geo = g->geo[l].address + 4u * info->phase[ph].at,
            .M = n * info->phase[ph].rh * info->phase[ph].rw, .N = v.CG, .K = info->phase[ph].taps * v.OG,
            .ldb = v.CG, .ldc = v.in_c, .a_group = v.OG, .b_group = v.taps * v.OG * v.CG, .c_group = v.CG,
            .alpha = 1.0f, .beta = beta, .flags = bias ? SPG_GEMM_BIAS : 0u,
        };
        SpgGemmMode m = {SPG_A_CONV, SPG_B_ROW, epi, act, v.G, info->phases > 1, v.OG % 4u == 0 && v.out_c % 4u == 0,
                         true, 0, (uint64_t)n * v.in_h * v.in_w * v.in_c, g->bf16};
        m.wide_a = v.OG % 8u == 0 && v.out_c % 8u == 0;
        m.wide_b = true;
        product(r, a, &p, &m);
    }
}

/* ---- forward ---- */

static void conv_forward(SpgGpuNet *g, Recorder *r, uint32_t l, uint32_t n, uint32_t act) {
    const uint32_t src = spingalett_source(g->net, l);
    Access a = {0};
    if (g->net->shapes[l].type == LAYER_CONV2D)
        conv_apply(g, r, &a, l, outputs_of(g, src), &g->act[l], n, SPG_EPI_BIAS_ACT, act, 0, 0.0f);
    else
        conv_spread(g, r, &a, l, &g->act[src], &g->act[l], n, SPG_EPI_BIAS_ACT, act, 0, 0.0f);
}

static void upsample_forward(SpgGpuNet *g, Recorder *r, uint32_t l, uint32_t n, uint32_t act) {
    const NeuralNetwork *net = g->net;
    const uint32_t src = spingalett_source(net, l);
    const LayerShape *in = &net->shapes[src], *s = &net->shapes[l];
    SpgUpsamplePush p = {g->act[src].address, g->act[l].address, 0, 0, n, in->height, in->width, in->channels,
                         s->stride_h, s->stride_w, 0, 0};
    uint32_t spec[3] = {s->mode == UPSAMPLE_BILINEAR, 0, act};
    Access a = {0};
    reads(&a, whole(&g->act[src]));
    writes(&a, whole(&g->act[l]));
    kernel(r, &a, SPG_KERNEL_upsample, spec, 3, &p, sizeof p, groups((uint64_t)n * net->topology[l], 256u), 1, 1);
}

/* Threads of ln.comp that share a cell: about eight channels each, at most 64. */
static uint32_t ln_threads(uint32_t C) {
    uint32_t t = 1;
    while (t < 64u && 16u * t <= C) t *= 2u;
    return t;
}

/* train: each cell's statistics kept for the backward pass */
static void ln_forward(SpgGpuNet *g, Recorder *r, uint32_t l, uint32_t n, uint32_t act, bool train) {
    const NeuralNetwork *net = g->net;
    const uint32_t src = spingalett_source(net, l);
    const LayerShape *s = &net->shapes[l];
    const uint32_t cells = n * s->height * s->width, T = ln_threads(s->channels);
    const bool stats = train && g->bn[l].buffer;
    SpgLnPush p = {g->act[src].address, g->act[l].address, 0, weights_at(g, l - 1), biases_at(g, l - 1),
                   stats ? g->bn[l].address : 0, cells, s->channels, stats ? 1u : 0u, s->eps};
    uint32_t spec[3] = {T, 0, act};
    Access a = {0};
    reads(&a, whole(&g->act[src]));
    reads(&a, weights_of(g, l - 1));
    reads(&a, biases_of(g, l - 1));
    writes(&a, whole(&g->act[l]));
    if (stats) writes(&a, span(g->bn[l].address, 2ull * cells));
    kernel(r, &a, SPG_KERNEL_ln, spec, 3, &p, sizeof p, groups(cells, 256u / T), 1, 1);
}

/* y = act(dense layer l of x + its biases), or with scales (a pair of floats an output: e0, the scales then
   the shifts) y = act(the product times the scales plus the shifts) */
static void dense_apply(SpgGpuNet *g, Recorder *r, Access *a, uint32_t l, const SpgGpuBuffer *y, uint32_t n,
                        uint32_t act, uint64_t scales) {
    const NeuralNetwork *net = g->net;
    const uint32_t src = spingalett_source(net, l), K = net->topology[src], N = net->topology[l];
    SpgGemmPush p = {
        .a = g->act[src].address, .b = mm_weights_at(g, l - 1), .c = y->address,
        .e0 = scales ? scales : biases_at(g, l - 1), .e1 = scales ? scales + 4ull * N : 0,
        .M = n, .N = N, .K = K, .lda = K, .ldb = K, .ldc = N, .alpha = 1.0f, .flags = scales ? 0u : SPG_GEMM_BIAS,
    };
    SpgGemmMode m = {SPG_A_ROW, SPG_B_COL, scales ? SPG_EPI_SCALE_ACT : SPG_EPI_BIAS_ACT, act, 1, false, true, true, 0,
                     (uint64_t)n * N, g->bf16, 0, true, true};
    reads(a, whole(&g->act[src]));
    reads(a, mm_weights_of(g, l - 1));
    if (scales) reads(a, span(scales, 2ull * N));
    else reads(a, biases_of(g, l - 1));
    writes(a, whole(y));
    product(r, a, &p, &m);
}

static void dense_forward(SpgGpuNet *g, Recorder *r, uint32_t l, uint32_t n, uint32_t act) {
    Access a = {0};
    dense_apply(g, r, &a, l, &g->act[l], n, act, 0);
}

/* train: with the batch's statistics (which also move the running ones); otherwise the running statistics,
   as inference and validation use them, applied by the product of its input where that applies it (into[]):
   the coefficients with the product's biases folded in, then the product (the normalization's input never
   stored; ResNet-20 of Examples/Benchmark.c infers 1.1 times as fast) */
static void bn_forward(SpgGpuNet *g, Recorder *r, uint32_t l, uint32_t n, uint32_t act, bool train) {
    const NeuralNetwork *net = g->net;
    const uint32_t src = spingalett_source(net, l);
    const LayerShape *s = &net->shapes[l];
    const uint32_t C = s->channels, R = n * s->height * s->width, slices = (R + COLSUM_ROWS - 1u) / COLSUM_ROWS;
    const uint64_t stats = g->bn[l].address;
    const bool into = !train && g->into[src] == l;
    SpgBnPush bp = {
        .part = g->part.address, .x = into ? biases_at(g, src - 1) : g->act[src].address, .gamma = weights_at(g, l - 1),
        .beta = biases_at(g, l - 1), .rmean = running_at(g, l - 1, 0), .rvar = running_at(g, l - 1, 1), .stats = stats,
        .C = C, .slices = slices, .m = (float)R, .eps = s->eps, .momentum = s->momentum,
    };
    uint32_t mode = into ? SPG_BN_FOLD : SPG_BN_INFER;
    Access a = {0};
    reads(&a, weights_of(g, l - 1));
    reads(&a, biases_of(g, l - 1));
    reads(&a, span(running_at(g, l - 1, 0), C));
    reads(&a, span(running_at(g, l - 1, 1), C));
    if (into) reads(&a, biases_of(g, src - 1));
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
    if (train || g->recording) kernel(r, &a, SPG_KERNEL_bn, &mode, 1, &bp, sizeof bp, (C + 63u) / 64u, 1, 1);
    if (g->fold[l] || g->pro[l]) return;        /* applied by the addition or the convolution that reads it */
    Access b = {0};
    if (into) {
        if (net->shapes[src].type == LAYER_DENSE) dense_apply(g, r, &b, src, &g->act[l], n, act, stats + 8u * C);
        else conv_apply(g, r, &b, src, outputs_of(g, spingalett_source(net, src)), &g->act[l], n, SPG_EPI_SCALE_ACT,
                        act, stats + 8u * C, 0.0f);
        return;
    }
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

/* The output kernel's dispatch over n rows of `width` outputs: a thread a row, or for rows of more
   than 64 outputs a workgroup (spec[3]). */
static void output_kernel(Recorder *r, const Access *a, uint32_t spec[4], const SpgOutputPush *p) {
    spec[3] = p->n > 64u;
    kernel(r, a, SPG_KERNEL_output, spec, 4, p, sizeof *p, spec[3] ? groups(p->rows, 1u) : (p->rows + 63u) / 64u, 1,
           1);
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
        if (!train && g->into[l]) continue;     /* its product runs with the normalization it applies */
        if (type == LAYER_DENSE) dense_forward(g, r, l, n, fused);
        else if (spingalett_filters(type)) conv_forward(g, r, l, n, fused);
        else if (type == LAYER_BATCH_NORM) bn_forward(g, r, l, n, fused, train);
        else if (type == LAYER_LAYER_NORM) ln_forward(g, r, l, n, fused, train);
        else if (type == LAYER_UPSAMPLE) upsample_forward(g, r, l, n, fused);
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
            uint32_t spec[4] = {SPG_OUT_SOFTMAX, ACT_SOFTMAX, 0, 0};
            Access a = {0};
            reads(&a, whole(&g->act[l]));
            writes(&a, whole(&g->act[l]));
            output_kernel(r, &a, spec, &p);
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
   convolution, transposed convolution, adding, concatenating, global pooling and upsampling layers),
   then times act'(the input's outputs) when fused is not ACT_NONE. Returns whether the derivative
   was applied. */
static bool input_gradient(SpgGpuNet *g, Recorder *r, uint32_t cl, uint32_t k, const SpgGpuBuffer *dst, uint32_t n,
                           uint32_t fused, bool accumulate) {
    const NeuralNetwork *net = g->net;
    const uint32_t *in = spingalett_inputs(net, cl), i = in[k];
    const LayerShape *s = &net->shapes[cl], *x = &net->shapes[i];
    const bool derive = fused != ACT_NONE;
    Access a = {0};
    reads(&a, whole(&g->delta[cl]));
    if (derive || spingalett_normalization(s->type) || s->type == LAYER_MAX_POOL2D || s->type == LAYER_AVG_POOL2D)
        reads(&a, whole(outputs_of(g, i)));
    if (accumulate) reads(&a, whole(dst));
    writes(&a, whole(dst));
    switch (s->type) {
        case LAYER_DENSE: {         /* dst = delta[cl] W, W stored [next x cur] */
            const uint32_t cur = net->topology[i], next = net->topology[cl];
            SpgGemmPush p = {
                .a = g->delta[cl].address, .b = mm_weights_at(g, cl - 1), .c = dst->address, .e0 = g->act[i].address,
                .M = n, .N = cur, .K = next, .lda = next, .ldb = cur, .ldc = cur, .alpha = 1.0f,
                .beta = accumulate ? 1.0f : 0.0f,
            };
            SpgGemmMode m = {SPG_A_ROW, SPG_B_ROW, derive ? SPG_EPI_DERIV : SPG_EPI_STORE, fused, 1, false, true, true, 0,
                             (uint64_t)n * cur, g->bf16, 0, true, true};
            reads(&a, mm_weights_of(g, cl - 1));
            product(r, &a, &p, &m);
            return derive;
        }
        case LAYER_CONV2D:          /* through the convolution's data gradient (conv_spread()) */
            conv_spread(g, r, &a, cl, &g->delta[cl], dst, n, derive ? SPG_EPI_DERIV : SPG_EPI_STORE, fused,
                        outputs_of(g, i)->address, accumulate ? 1.0f : 0.0f);
            return derive;
        case LAYER_CONV_TRANSPOSE2D:    /* the forward pass of the convolution it transposes */
            conv_apply(g, r, &a, cl, &g->delta[cl], dst, n, derive ? SPG_EPI_DERIV : SPG_EPI_STORE, fused,
                       g->act[i].address, accumulate ? 1.0f : 0.0f);
            return derive;
        case LAYER_LAYER_NORM: {
            const uint32_t cells = n * s->height * s->width;
            SpgLnPush p = {g->act[i].address, dst->address, g->delta[cl].address, weights_at(g, cl - 1), 0,
                           g->bn[cl].address, cells, s->channels, 0, 0.0f};
            uint32_t spec[3] = {ln_threads(s->channels), 1, fused};
            reads(&a, weights_of(g, cl - 1));
            reads(&a, span(g->bn[cl].address, 2ull * cells));
            kernel(r, &a, SPG_KERNEL_ln, spec, 3, &p, sizeof p, groups(cells, 256u / spec[0]), 1, 1);
            return true;
        }
        case LAYER_UPSAMPLE: {
            SpgUpsamplePush p = {g->act[i].address, 0, g->delta[cl].address, dst->address, n, x->height, x->width,
                                 x->channels, s->stride_h, s->stride_w, accumulate ? 1u : 0u, 0};
            uint32_t spec[3] = {s->mode == UPSAMPLE_BILINEAR, 1, fused};
            kernel(r, &a, SPG_KERNEL_upsample, spec, 3, &p, sizeof p, groups((uint64_t)n * net->topology[i], 256u), 1, 1);
            return true;
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
            /* windows that tile the input: a thread per window */
            const bool tiled = s->stride_h == s->kernel_h && s->stride_w == s->kernel_w && s->pad_h == 0 &&
                               s->pad_w == 0 && s->height * s->kernel_h == x->height &&
                               s->width * s->kernel_w == x->width;
            uint32_t spec[3] = {s->type == LAYER_MAX_POOL2D, tiled ? 2u : 1u, fused};
            kernel(r, &a, SPG_KERNEL_pool, spec, 3, &p, sizeof p,
                   groups((uint64_t)n * net->topology[tiled ? cl : i], 256u), 1, 1);
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
    const uint32_t C = s->channels, R = n * s->height * s->width, sums = dw_sums(g, cl, n);
    const uint32_t slices = sums ? sums : (R + COLSUM_ROWS - 1u) / COLSUM_ROWS;
    if (!sums) {    /* (otherwise the data gradient that made delta[cl] summed) */
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
    }
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

/* The parameter gradients of layer normalization layer cl: gamma's the column sums of dy xhat, beta's
   of dy (scaled and added to the step's). */
static void ln_backward(SpgGpuNet *g, Recorder *r, uint32_t cl, uint32_t n, float scale, float beta) {
    const NeuralNetwork *net = g->net;
    const uint32_t src = spingalett_source(net, cl);
    const LayerShape *s = &net->shapes[cl];
    const uint32_t C = s->channels, R = n * s->height * s->width, slices = (R + COLSUM_ROWS - 1u) / COLSUM_ROWS;
    const uint32_t cols = C <= 16u ? 16u : C <= 32u ? 32u : 64u;
    SpgColsumPush cp = {g->act[src].address, g->delta[cl].address, g->bn[cl].address, g->part.address, R, C,
                        COLSUM_ROWS, 0};
    uint32_t spec[2] = {cols, SPG_COLSUM_LN};
    Access a = {0};
    reads(&a, whole(&g->act[src]));
    reads(&a, whole(&g->delta[cl]));
    reads(&a, span(g->bn[cl].address, 2ull * R));
    writes(&a, span(g->part.address, 2ull * slices * C));
    kernel(r, &a, SPG_KERNEL_colsum, spec, 2, &cp, sizeof cp, (C + cols - 1) / cols, slices, 1);
    /* the slices added in order: their first C sums to beta's gradient, their last C to gamma's */
    for (uint32_t k = 0; k < 2; k++) {
        const uint64_t out = k ? grad_weights_at(g, cl - 1) : grad_biases_at(g, cl - 1);
        SpgReducePush rp = {g->part.address + (k ? 4u * C : 0u), out, C, 2u * C, slices, scale, beta, 0};
        Access b = {0};
        reads(&b, span(g->part.address, 2ull * slices * C));
        if (beta != 0.0f) reads(&b, span(out, C));
        writes(&b, span(out, C));
        kernel(r, &b, SPG_KERNEL_reduce, NULL, 0, &rp, sizeof rp, (C + 255u) / 256u, 1, 1);
    }
}

/* grad = scale * (the chunk's gradient) + beta grad for dense or (transposed) convolution layer l + 1,
   whose gradient delta[l + 1] is complete. */
static void weight_gradient(SpgGpuNet *g, Recorder *r, uint32_t l, uint32_t n, float scale, float beta) {
    const NeuralNetwork *net = g->net;
    const uint32_t src = spingalett_source(net, l + 1), out_sz = net->topology[l + 1];
    const LayerShape *s = &net->shapes[l + 1];
    SpgGemmPush p = {.c = grad_weights_at(g, l), .alpha = scale, .beta = beta};
    SpgGemmMode m = {SPG_A_COL, SPG_B_ROW, SPG_EPI_STORE, ACT_NONE, 1, false, true, true, 0, 0, g->bf16, 0, true, true};
    uint32_t rows, M, N, K;
    Access a = {0};
    reads(&a, whole(&g->delta[l + 1]));
    reads(&a, whole(outputs_of(g, src)));
    if (s->type != LAYER_DENSE && depthwise(net, l + 1)) {
        /* depthwise: partial sums over slices of the output pixels (dwconv.comp), added in order */
        const ConvView v = conv_view(net, l + 1);
        const bool transposed = s->type == LAYER_CONV_TRANSPOSE2D;
        const SpgGpuBuffer *x = transposed ? &g->delta[l + 1] : outputs_of(g, src);
        const SpgGpuBuffer *dy = transposed ? &g->act[src] : &g->delta[l + 1];
        const DwSlices d = dw_slices(&v, n);
        const uint32_t slices = d.slices;
        const uint32_t bn = transposed ? 0u : applied_norm(g, l + 1);
        SpgDwconvPush p = {x->address, g->part.address, dy->address, 0, bn ? g->bn[bn].address : 0, 0, 0,
                           v.in_h, v.in_w, v.in_c, v.out_h, v.out_w, v.out_c, v.OG, s->pad_h, d.units, d.rows, d.lanes,
                           0.0f};
        const uint32_t spec[11] = {SPG_DW_WEIGHTS, 0, ACT_NONE, s->kernel_h, s->kernel_w, s->stride_h, s->stride_w,
                                   d.vec, s->pad_w, bn ? 1u + net->act_func[bn - 1] : 0u, 0};
        if (bn) reads(&a, span(g->bn[bn].address, 4u * v.in_c));
        const Range part = span(g->part.address, (uint64_t)slices * v.out_c * v.taps);
        writes(&a, part);
        kernel(r, &a, SPG_KERNEL_dwconv, spec, 11, &p, sizeof p, slices, (v.out_c / d.vec + d.per - 1u) / d.per, 1);
        const Range grad = span(grad_weights_at(g, l), weight_count(g, l));
        SpgReducePush rp = {g->part.address, grad_weights_at(g, l), v.out_c * v.taps, v.out_c * v.taps, slices, scale,
                            beta, 0};
        Access b = {0};
        reads(&b, part);
        if (beta != 0.0f) reads(&b, grad);
        writes(&b, grad);
        kernel(r, &b, SPG_KERNEL_reduce, NULL, 0, &rp, sizeof rp, groups(rp.total, 256u), 1, 1);
        column_sums(g, r, &g->delta[l + 1], n * s->height * s->width, s->channels, grad_biases_at(g, l), scale, beta);
        return;
    }
    if (s->type == LAYER_DENSE) {
        /* gW[out x in] = delta^T act */
        M = out_sz; N = net->topology[src]; K = n; rows = n;
        p.a = g->delta[l + 1].address; p.lda = out_sz;
        p.b = g->act[src].address; p.ldb = N;
    } else {
        /* gW[g OG + f][window] = sum over the convolution's output pixels of dy[pixel][g OG + f] x
           window[pixel] (conv_view(): for a transposed convolution, x is its output gradient and dy
           its input) */
        const ConvView v = conv_view(net, l + 1);
        const bool transposed = s->type == LAYER_CONV_TRANSPOSE2D;
        const SpgGpuBuffer *x = transposed ? &g->delta[l + 1] : &g->act[src];
        const SpgGpuBuffer *dy = transposed ? &g->act[src] : &g->delta[l + 1];
        m.groups = v.G;
        M = v.OG; N = v.taps * v.CG; K = n * v.out_h * v.out_w; rows = n * s->height * s->width;
        p.a = dy->address; p.lda = v.out_c; p.a_group = v.OG;
        p.b = x->address; p.b_group = v.CG; p.geo = g->geo[l + 1].address;
        m.bmode = SPG_B_CONV;
        m.vec_b = v.CG % 4u == 0 && v.in_c % 4u == 0;
        m.wide_b = v.CG % 8u == 0 && v.in_c % 8u == 0;
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
        if (type == LAYER_DENSE || spingalett_filters(type)) weight_gradient(g, r, cl - 1, n, scale, beta);
        if (type == LAYER_BATCH_NORM) bn_backward(g, r, cl, n, scale, beta);
        if (type == LAYER_LAYER_NORM) ln_backward(g, r, cl, n, scale, beta);
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
                const bool direct = first || type == LAYER_DENSE || spingalett_filters(type) || type == LAYER_ADD ||
                                    type == LAYER_CONCAT || type == LAYER_GLOBAL_AVG_POOL || type == LAYER_UPSAMPLE;
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
                /* (the derivative of no activation is 1: nothing to apply) */
                if (done && !applied && (masked || act != ACT_NONE))
                    eltwise(g, r, &fix, masked ? SPG_ELT_MUL : SPG_ELT_DERIV, masked ? ACT_NONE : act,
                            g->delta[l].address, masked ? g->dmask[l].address : g->act[l].address, 0, 0, total, 1);
                continue;
            }
            if ((input_gradient(g, r, cl, k, &g->delta[l], n, masked ? ACT_NONE : act, false) && !masked) ||
                (!masked && act == ACT_NONE))
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
    if (g->wh.buffer) writes(&a, whole(&g->wh));
    if (g->moment1.buffer) writes(&a, whole(&g->moment1));
    if (g->moment2.buffer) writes(&a, whole(&g->moment2));
    /* weights, with no decay for the normalizations' gamma; biases without decay */
    bool per_layer = false;
    for (uint32_t l = 1; o->decay != 0.0f && l < net->layers; l++)
        if (spingalett_normalization(net->shapes[l].type)) per_layer = true;
    for (uint32_t l = 0; l + 1 < net->layers; l++) {
        uint64_t w0 = per_layer ? g->woff[l] : 0, count = per_layer ? weight_count(g, l) : W;
        if (count > 0) {
            p.w = at(&g->params, w0); p.g = at(&g->grads, w0); p.n = (uint32_t)count;
            p.wh = g->wh.buffer ? g->wh.address + 2u * w0 : 0;
            p.m = g->moment1.buffer ? at(&g->moment1, w0) : 0;
            p.v = g->moment2.buffer ? at(&g->moment2, w0) : 0;
            p.decay = per_layer && spingalett_normalization(net->shapes[l + 1].type) ? 0.0f : o->decay;
            kernel(r, &a, SPG_KERNEL_optim, &spec, 1, &p, sizeof p, groups(count, 256u), 1, 1);
        }
        if (!per_layer) break;
    }
    if (B > 0) {
        p.w = at(&g->params, W); p.g = at(&g->grads, W); p.n = (uint32_t)B; p.decay = 0.0f; p.wh = 0;
        p.m = g->moment1.buffer ? at(&g->moment1, W) : 0;
        p.v = g->moment2.buffer ? at(&g->moment2, W) : 0;
        kernel(r, &a, SPG_KERNEL_optim, &spec, 1, &p, sizeof p, groups(B, 256u), 1, 1);
    }
}

/* ------------------------------------------------------------------------- data sets on the GPU */

struct SpingalettDeviceData {
    SpgBackend backend;             /* the backend whose memory holds it */
    SpgGpuArena *arena;
    SpgGpuBuffer buffer;            /* count rows of size floats */
    uint32_t count, size;
    /* the rows as bfloat16, rounded as the host rounds inputs kept as such (made on first use, under the
       lock: a set may serve several networks and threads) */
    SpgSignal *lock;
    SpgGpuArena *half_arena;
    SpgGpuBuffer half;
    int half_state;                 /* 0: not made yet, 1: made, -1: could not be */
};

/* One copy between a device buffer and host memory, through a host-visible buffer of its own. */
static bool data_copy(const SpgGpuBuffer *device, size_t offset, void *host, size_t bytes, bool down) {
    SpgGpuBuffer stage;
    if (!spg_gpu_buffer_create(&stage, bytes, true)) return false;
    if (!down) memcpy(stage.mapped, host, bytes);
    SpgGpuCommands *c = spg_gpu_commands_create();
    bool ok = c && spg_gpu_record_begin(c);
    if (ok) {
        spg_gpu_barrier(c);
        if (down) spg_gpu_copy(c, device, offset, &stage, 0, bytes);
        else spg_gpu_copy(c, &stage, 0, device, offset, bytes);
        spg_gpu_barrier(c);
        ok = spg_gpu_record_end(c) && spg_gpu_submit(c) && spg_gpu_wait(c);
    }
    if (ok && down) memcpy(host, stage.mapped, bytes);
    spg_gpu_commands_free(c);
    spg_gpu_buffer_free(&stage);
    return ok;
}

SpingalettDeviceData *spingalett_gpu_data_create(const float *values, uint32_t count, uint32_t size) {
    if (!spg_gpu_open() || count == 0 || size == 0) return NULL;
    SpingalettDeviceData *d = (SpingalettDeviceData *)calloc(1, sizeof *d);
    if (!d || !(d->lock = spg_signal_create()) || !(d->arena = spg_gpu_arena_create())) {
        if (d) spg_signal_free(d->lock);
        free(d);
        return NULL;
    }
    d->backend = spg_gpu_using();
    d->count = count;
    d->size = size;
    const size_t bytes = (size_t)count * size * sizeof(float);
    /* written by the host straight into the device's memory where it can */
    spg_gpu_arena_add(d->arena, &d->buffer, bytes, spg_gpu_host_writes() ? SPG_MEMORY_HOST_WRITES : SPG_MEMORY_DEVICE);
    bool ok = spg_gpu_arena_commit(d->arena);
    if (ok && d->buffer.mapped) memcpy(d->buffer.mapped, values, bytes);
    else if (ok) ok = data_copy(&d->buffer, 0, (void *)values, bytes, false);
    if (!ok) {
        spingalett_gpu_data_free(d);
        return NULL;
    }
    return d;
}

bool spingalett_gpu_data_on(const SpingalettDeviceData *d, const SpgGpuNet *g) {
    return d && g && d->backend == g->backend;
}

uint32_t spingalett_gpu_data_count(const SpingalettDeviceData *d) { return d ? d->count : 0; }
uint32_t spingalett_gpu_data_size(const SpingalettDeviceData *d) { return d ? d->size : 0; }

static void spingalett_gpu_data_free_here(SpingalettDeviceData *d) {
    if (!d) return;
    spg_gpu_arena_free(d->half_arena);
    spg_gpu_arena_free(d->arena);
    spg_signal_free(d->lock);
    free(d);
}

void spingalett_gpu_data_free(SpingalettDeviceData *d) {
    const SpgBackend saved = spg_gpu_using();
    if (d) spg_gpu_use(d->backend);
    spingalett_gpu_data_free_here(d);
    spg_gpu_use(saved);
}

/* The set's rows as bfloat16 (NULL when they cannot be made: no memory, or more values than the
   conversion indexes), for networks that keep their inputs as such: read where they are, as the host's
   rounded inputs would be, by every kind of layer. */
static const SpgGpuBuffer *data_half(const SpingalettDeviceData *cd) {
    SpingalettDeviceData *d = (SpingalettDeviceData *)cd;      /* a cache of the values it holds */
    spg_lock(d->lock);
    const uint64_t values = (uint64_t)d->count * d->size;
    if (d->half_state == 0) {
        d->half_state = -1;
        /* (the conversion indexes the values with signed 32-bit integers, as SPIR-V does) */
        if (values <= INT32_MAX && spg_gpu_bf16_storage() && (d->half_arena = spg_gpu_arena_create())) {
            spg_gpu_arena_add(d->half_arena, &d->half, values * 2u, SPG_MEMORY_DEVICE);
            SpgGpuCommands *c = spg_gpu_arena_commit(d->half_arena) ? spg_gpu_commands_create() : NULL;
            SpgEltwisePush p = {d->half.address, d->buffer.address, 0, 0, 0, (uint32_t)values, 1, 0, 0, 0.0f, 0, 0};
            const uint32_t spec[3] = {SPG_ELT_COPY, ACT_NONE, 1u};    /* y kept as bfloat16 */
            if (c && spg_gpu_record_begin(c)) {
                spg_gpu_barrier(c);
                spg_gpu_dispatch(c, SPG_KERNEL_eltwise_h, spec, 3, &p, sizeof p, groups(values, 256u), 1, 1);
                spg_gpu_barrier(c);
                if (spg_gpu_record_end(c) && spg_gpu_submit(c) && spg_gpu_wait(c)) d->half_state = 1;
            }
            spg_gpu_commands_free(c);
        }
        if (d->half_state < 0) {
            spg_gpu_arena_free(d->half_arena);
            d->half_arena = NULL;
        }
    }
    spg_unlock(d->lock);
    return d->half_state > 0 ? &d->half : NULL;
}

static bool spingalett_gpu_data_read_here(const SpingalettDeviceData *d, uint32_t first, uint32_t count, float *dst) {
    if (!d || (uint64_t)first + count > d->count) return false;
    return count == 0 || data_copy(&d->buffer, (size_t)first * d->size * sizeof(float), dst,
                                   (size_t)count * d->size * sizeof(float), true);
}

bool spingalett_gpu_data_read(const SpingalettDeviceData *d, uint32_t first, uint32_t count, float *dst) {
    const SpgBackend saved = spg_gpu_using();
    if (d) spg_gpu_use(d->backend);
    bool result = spingalett_gpu_data_read_here(d, first, count, dst);
    spg_gpu_use(saved);
    return result;
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

/* The slot of the next chunk, its previous chunk finished: the oldest of those in flight, so that the
   newer ones keep the device busy while the host fills the slot (chunks are harvested in order). With
   two slots the device waited for the host to fill each chunk: the MLP of Examples/Benchmark.c infers
   1.2 times as fast from host arrays with three (an RTX 4050 Laptop GPU, both backends). */
static Slot *next_slot(SpgGpuNet *g) {
    Slot *s = &g->slots[g->next];
    if (s->pending) harvest(g, s);
    return s;
}

/* Where the host writes a slot's step header and inputs: straight into the device's buffers where
   it can, else into staging, from which the slot's commands copy them (copy_in()). */
static void *header_of(Slot *s) {
    return s->header.mapped ? s->header.mapped : s->staging.mapped;
}

static float *inputs_of(Slot *s) {
    if (s->host_inputs) return s->host_inputs;
    return s->input.mapped ? (float *)s->input.mapped : (float *)((char *)s->staging.mapped + s->inputs);
}

/* n inputs rounded to bfloat16, to the nearest, ties to even, as gemm_common.glsl's to_bf16() */
void spingalett_round_bf16(uint16_t *dst, const float *src, size_t n) {
    for (size_t i = 0; i < n; i++) {
        uint32_t u;
        memcpy(&u, &src[i], sizeof u);
        dst[i] = (uint16_t)(src[i] != src[i] ? (u >> 16) | 0x40u : (u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
    }
}

/* The host's inputs of n samples rounded into the device's buffer (or staging), when the inputs are
   kept as bfloat16. */
static void stage_inputs(const SpgGpuNet *g, Slot *s, uint32_t n) {
    if (s->staged || !s->host_inputs) {
        s->staged = false;
        return;
    }
    void *dst = s->input.mapped ? s->input.mapped : (char *)s->staging.mapped + s->inputs;
    spingalett_round_bf16((uint16_t *)dst, s->host_inputs, (size_t)n * g->net->topology[0]);
}

static void copy_in(SpgGpuCommands *c, Slot *s, size_t input_bytes) {
    if (!s->header.mapped) spg_gpu_copy(c, &s->staging, 0, &s->header, 0, SPG_STEP_HEADER * 4u);
    if (!s->input.mapped) spg_gpu_copy(c, &s->staging, s->inputs, &s->input, 0, input_bytes);
}

static float *spingalett_gpu_chunk_inputs_here(SpgGpuNet *g, float **targets) {
    Slot *s = next_slot(g);
    if (targets) *targets = (float *)((char *)s->staging.mapped + s->targets);
    return inputs_of(s);
}

float *spingalett_gpu_chunk_inputs(SpgGpuNet *g, float **targets) {
    const SpgBackend saved = spg_gpu_using();
    if (g) spg_gpu_use(g->backend);
    float * result = spingalett_gpu_chunk_inputs_here(g, targets);
    spg_gpu_use(saved);
    return result;
}

/* Where the host rounds a slot's inputs kept as bfloat16 (straight from the caller's floats): the device's
   buffer where it can write it, else staging, from which the slot's commands copy them. */
static uint16_t *spingalett_gpu_chunk_inputs_bf16_here(SpgGpuNet *g, float **targets) {
    Slot *s = &g->slots[g->next];
    if (!s->host_inputs) return NULL;
    next_slot(g);
    if (targets) *targets = (float *)((char *)s->staging.mapped + s->targets);
    s->staged = true;
    return (uint16_t *)(s->input.mapped ? s->input.mapped : (char *)s->staging.mapped + s->inputs);
}

uint16_t *spingalett_gpu_chunk_inputs_bf16(SpgGpuNet *g, float **targets) {
    const SpgBackend saved = spg_gpu_using();
    if (g) spg_gpu_use(g->backend);
    uint16_t * result = spingalett_gpu_chunk_inputs_bf16_here(g, targets);
    spg_gpu_use(saved);
    return result;
}

static void spingalett_gpu_chunk_ready_here(SpgGpuNet *g, uint32_t n) {
    Slot *s = &g->slots[g->next];
    stage_inputs(g, s, n);
    s->staged = s->host_inputs != NULL;
}

void spingalett_gpu_chunk_ready(SpgGpuNet *g, uint32_t n) {
    const SpgBackend saved = spg_gpu_using();
    if (g) spg_gpu_use(g->backend);
    spingalett_gpu_chunk_ready_here(g, n);
    spg_gpu_use(saved);
}

static void spingalett_gpu_set_rows_here(SpgGpuNet *g, const SpgGpuRows *rows) {
    if (rows) g->rows = *rows;
    else memset(&g->rows, 0, sizeof g->rows);
    /* (a chunk filled ahead for a step that did not come takes nothing from them) */
    for (uint32_t k = 0; k < SLOTS; k++) {
        g->slots[k].gathered = 0;
        g->slots[k].in_place = false;
    }
}

void spingalett_gpu_set_rows(SpgGpuNet *g, const SpgGpuRows *rows) {
    const SpgBackend saved = spg_gpu_using();
    if (g) spg_gpu_use(g->backend);
    spingalett_gpu_set_rows_here(g, rows);
    spg_gpu_use(saved);
}

static uint32_t *spingalett_gpu_chunk_rows_here(SpgGpuNet *g) {
    Slot *s = next_slot(g);
    s->gathered = (g->rows.inputs && !s->in_place ? 1u : 0u) | (g->rows.targets ? 2u : 0u);
    return (uint32_t *)((char *)s->staging.mapped + s->indices);
}

uint32_t *spingalett_gpu_chunk_rows(SpgGpuNet *g) {
    const SpgBackend saved = spg_gpu_using();
    if (g) spg_gpu_use(g->backend);
    uint32_t * result = spingalett_gpu_chunk_rows_here(g);
    spg_gpu_use(saved);
    return result;
}

static bool spingalett_gpu_chunk_in_place_here(SpgGpuNet *g, uint32_t first) {
    if (!g->rows.inputs || g->rows.shift || g->rows.flip || (g->half[0] && !data_half(g->rows.inputs))) return false;
    Slot *s = next_slot(g);
    s->in_place = true;
    s->first = first;
    return true;
}

bool spingalett_gpu_chunk_in_place(SpgGpuNet *g, uint32_t first) {
    const SpgBackend saved = spg_gpu_using();
    if (g) spg_gpu_use(g->backend);
    bool result = spingalett_gpu_chunk_in_place_here(g, first);
    spg_gpu_use(saved);
    return result;
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

/* The slot's command buffer of this key (and address of the inputs read in place, or 0); *fresh when it
   is to be recorded (NULL: out of memory). */
static SpgGpuCommands *cache_entry(Slot *s, uint64_t key, uint64_t where, bool *fresh) {
    *fresh = false;
    for (uint32_t e = 0; e < s->used; e++)
        if (s->cache[e].key == key && s->cache[e].where == where) return s->cache[e].commands;
    *fresh = true;
    SpgGpuCommands *c = NULL;
    if (s->used < sizeof s->cache / sizeof s->cache[0]) {
        c = spg_gpu_commands_create();
        if (!c) return NULL;
        s->used++;
    } else {
        /* a rare kind of chunk: the oldest entry is recorded again */
        c = s->cache[0].commands;
        memmove(s->cache, s->cache + 1, (s->used - 1) * sizeof s->cache[0]);
    }
    s->cache[s->used - 1].key = key;
    s->cache[s->used - 1].where = where;
    s->cache[s->used - 1].commands = c;
    return c;
}

/* Rows of the data sets on the GPU (indices in the slot's staging, the sets' addresses in the step
   header) gathered into the slot's inputs (gathered bit 1) and targets (bit 2). */
static void gather_rows(SpgGpuNet *g, Recorder *r, Slot *s, uint32_t n, uint32_t gathered) {
    const NeuralNetwork *net = g->net;
    const LayerShape *shape = &net->shapes[0];
    for (uint32_t k = 0; k < 2; k++) {
        if (!(gathered >> k & 1u)) continue;
        const SpgGpuBuffer *dst = k ? &s->target : &s->input;
        const uint32_t size = net->topology[k ? net->layers - 1 : 0];
        SpgRowsPush p = {s->staging.address + s->indices, dst->address, s->header.address, n, size,
                         shape->height, shape->width, shape->channels};
        /* STEP_TARGETS or STEP_INPUTS; four values a thread where rows hold a multiple of four */
        const uint32_t spec[2] = {k ? 12u : 10u, size % 4u ? 1u : 4u};
        Access a = {0};
        reads(&a, span(s->staging.address + s->indices, n));
        reads(&a, whole(&s->header));
        writes(&a, whole(dst));
        kernel(r, &a, SPG_KERNEL_rows, spec, 2, &p, sizeof p, groups((uint64_t)n * size / spec[1], 256u), 1, 1);
    }
}

/* The slot's command buffer for chunks of this kind, recorded on first use; the network's inputs those
   of `direct` (rows of a data set on the GPU, in single precision) instead of the slot's when given. */
static SpgGpuCommands *chunk_commands(SpgGpuNet *g, Slot *s, uint64_t key, uint32_t n, uint32_t count, bool first,
                                      bool last, bool train, uint32_t gathered, const SpgGpuBuffer *direct) {
    bool fresh;
    SpgGpuCommands *c = cache_entry(s, key, direct ? direct->address : 0, &fresh);
    if (!c || !fresh) return c;
    const NeuralNetwork *net = g->net;
    const uint32_t in = net->topology[0], out = net->topology[net->layers - 1];
    Recorder *r = spg_gpu_record_begin(c) ? (Recorder *)calloc(1, sizeof *r) : NULL;
    if (!r) return forget(s, c);
    r->c = c;
    r->g = g;
    /* the chunk's header, inputs and targets from staging into the slot's buffers, which the chunks in
       the other slots do not use: copied while they still run; then everything after it */
    g->act[0] = direct ? *direct : s->input;
    g->header = s->header;
    g->targets = s->target;
    /* inputs and targets from staging, or gathered from the GPU's data sets after the header, or read
       where they are */
    copy_in(c, s, gathered & 1u || direct ? 0 : (size_t)n * in * (g->half[0] ? 2u : 4u));
    if (train && !(gathered & 2u)) spg_gpu_copy(c, &s->staging, s->targets, &s->target, 0, (size_t)n * out * 4u);
    /* the gather reads the indices and the header the host wrote, and writes the slot's buffers: while the
       other slots' chunks still run, unless the header is copied first */
    if (gathered) {
        if (!s->header.mapped) barrier(r);
        gather_rows(g, r, s, n, gathered);
    }
    barrier(r);
    record_forward(g, r, n, train);
    const uint32_t L = net->layers - 1;
    if (train) {
        const float scale = 1.0f / (float)count, beta = first ? 0.0f : 1.0f;
        SpgOutputPush p = {g->act[L].address, g->targets.address, g->delta[L].address, s->staging.address + s->losses,
                           n, out};
        uint32_t spec[4] = {SPG_OUT_LOSS, net->act_func[L - 1], net->loss_func, 0};
        Access a = {0};
        reads(&a, whole(&g->act[L]));
        reads(&a, whole(&g->targets));
        writes(&a, whole(&g->delta[L]));
        writes(&a, span(s->staging.address + s->losses, n));
        output_kernel(r, &a, spec, &p);
        record_backward(g, r, n, scale, beta);
        if (last) optimizer(g, r);
    } else {
        barrier(r);
        spg_gpu_copy(c, &g->act[L], 0, &s->staging, s->outputs, (size_t)n * out * 4u);
    }
    /* everything visible to the host (losses, outputs); the next chunk's copies need not wait */
    spg_gpu_barrier_host(c);
    free(r);
    g->act[0] = s->input;
    return spg_gpu_record_end(c) ? c : forget(s, c);
}

/* Rows first .. first + n - 1 of a data set on the GPU (of its bfloat16 copy with half), as a buffer of
   their own. */
static SpgGpuBuffer rows_view(const SpingalettDeviceData *d, uint32_t first, uint32_t n, bool half) {
    SpgGpuBuffer view = half ? d->half : d->buffer;
    const size_t value = half ? 2u : sizeof(float), at = (size_t)first * d->size * value;
    view.address += at;
    view.offset += at;
    view.size = (size_t)n * d->size * value;
    if (view.mapped) view.mapped = (char *)view.mapped + at;
    return view;
}

/* The data sets on the GPU the chunks' rows come from and what is done to them (STEP_INPUTS to
   STEP_SHARE). */
static void rows_header(const SpgGpuRows *rows, uint32_t header[SPG_STEP_HEADER]) {
    const uint64_t in = rows->inputs ? rows->inputs->buffer.address : 0;
    const uint64_t out = rows->targets ? rows->targets->buffer.address : 0;
    header[10] = (uint32_t)in;
    header[11] = (uint32_t)(in >> 32);
    header[12] = (uint32_t)out;
    header[13] = (uint32_t)(out >> 32);
    header[14] = (uint32_t)rows->seed;
    header[15] = (uint32_t)(rows->seed >> 32);
    header[16] = rows->shift | (rows->flip ? 1u << 31 : 0u);
    memcpy(&header[17], &rows->keep, 4);
    memcpy(&header[18], &rows->share, 4);
}

/* The step header (common.glsl's STEP_* entries). */
static void step_header(const SpgGpuNet *g, const SpgGpuStep *step, uint32_t position, float grad_scale,
                        uint32_t header[SPG_STEP_HEADER]) {
    memset(header, 0, SPG_STEP_HEADER * sizeof(uint32_t));
    memcpy(&header[0], &step->lr, 4);
    memcpy(&header[1], &step->m_factor, 4);
    memcpy(&header[2], &step->v_factor, 4);
    header[3] = (uint32_t)g->cfg.dropout_seed;
    header[4] = (uint32_t)(g->cfg.dropout_seed >> 32);
    header[5] = (uint32_t)step->step;
    header[6] = (uint32_t)(step->step >> 32);
    header[7] = position;
    memcpy(&header[9], &grad_scale, 4);
    rows_header(&g->rows, header);
}

static bool spingalett_gpu_train_chunk_here(SpgGpuNet *g, uint32_t n, uint32_t count, uint32_t position, bool first, bool last,
                                const SpgGpuStep *step) {
    if (!g->training || g->lost || n == 0 || n > g->capacity) return false;
    Slot *s = &g->slots[g->next];
    uint32_t header[SPG_STEP_HEADER];
    step_header(g, step, position, 1.0f, header);
    /* the commands depend on n, on count (gradients scaled by 1 / count) and on where the chunk is in
       its step */
    const uint64_t key = (uint64_t)n | (uint64_t)count << 24 | (first ? 1ull << 56 : 0) | (last ? 1ull << 57 : 0) |
                         (uint64_t)s->gathered << 59;
    const SpgGpuBuffer view = s->in_place ? rows_view(g->rows.inputs, s->first, n, g->half[0]) : (SpgGpuBuffer){0};
    SpgGpuCommands *c = chunk_commands(g, s, key, n, count, first, last, true, s->gathered, s->in_place ? &view : NULL);
    if (!c) return false;
    g->coefficients = false;
    memcpy(header_of(s), header, sizeof header);
    if (!(s->gathered & 1u) && !s->in_place) stage_inputs(g, s, n);
    s->gathered = 0;
    s->in_place = false;
    if (!spg_gpu_submit(c)) return false;
    s->pending = c;
    s->n = n;
    s->train = true;
    s->last = last;
    g->next = (g->next + 1u) % SLOTS;
    return true;
}

bool spingalett_gpu_train_chunk(SpgGpuNet *g, uint32_t n, uint32_t count, uint32_t position, bool first, bool last,
                                const SpgGpuStep *step) {
    const SpgBackend saved = spg_gpu_using();
    if (g) spg_gpu_use(g->backend);
    bool result = spingalett_gpu_train_chunk_here(g, n, count, position, first, last, step);
    spg_gpu_use(saved);
    return result;
}

static bool spingalett_gpu_take_loss_here(SpgGpuNet *g, float *loss) {
    drain(g);
    *loss = g->total_loss;
    g->total_loss = 0.0f;
    return !g->lost;
}

bool spingalett_gpu_take_loss(SpgGpuNet *g, float *loss) {
    const SpgBackend saved = spg_gpu_using();
    if (g) spg_gpu_use(g->backend);
    bool result = spingalett_gpu_take_loss_here(g, loss);
    spg_gpu_use(saved);
    return result;
}

static bool spingalett_gpu_predict_here(SpgGpuNet *g, const float *inputs, float *outputs, uint32_t n) {
    return spingalett_gpu_predict_rows(g, inputs, NULL, 0, outputs, n);
}

bool spingalett_gpu_predict(SpgGpuNet *g, const float *inputs, float *outputs, uint32_t n) {
    const SpgBackend saved = spg_gpu_using();
    if (g) spg_gpu_use(g->backend);
    bool result = spingalett_gpu_predict_here(g, inputs, outputs, n);
    spg_gpu_use(saved);
    return result;
}

static bool spingalett_gpu_predict_rows_here(SpgGpuNet *g, const float *inputs, const SpingalettDeviceData *rows, uint32_t first,
                                 float *outputs, uint32_t n) {
    const NeuralNetwork *net = g->net;
    const uint32_t in = net->topology[0], out = net->topology[net->layers - 1];
    /* chunk after chunk, each filled while the device runs those before, its outputs taken when
       its slot comes round again: the caller's floats in chunks of at most 2,048 samples, so that their
       copies overlap with the work on the chunks before (the MLP of Examples/Benchmark.c infers 1.1 times
       as fast in bfloat16 from host arrays in chunks of 2,048 as of 3,637, which its memory allows), a
       data set's rows in whole chunks */
    const uint32_t most = rows || g->capacity < SPINGALETT_BATCH_CHUNK ? g->capacity : SPINGALETT_BATCH_CHUNK;
    for (uint32_t start = 0; start < n; start += most) {
        const uint32_t m = n - start < most ? n - start : most;
        float *dst = spingalett_gpu_chunk_inputs(g, NULL);
        Slot *s = &g->slots[g->next];
        s->staged = false;
        /* the rows of a data set on the GPU read where they are (products round them to bfloat16 as they
           load them), the commands recorded for them kept for the next call; or gathered (and rounded)
           into the slot's inputs */
        const bool gather = rows && g->half[0] && !data_half(rows);
        const SpgGpuBuffer view = rows && !gather ? rows_view(rows, first + start, m, g->half[0]) : (SpgGpuBuffer){0};
        if (gather) {
            uint32_t *index = (uint32_t *)((char *)s->staging.mapped + s->indices), header[SPG_STEP_HEADER] = {0};
            for (uint32_t k = 0; k < m; k++) index[k] = first + start + k;
            rows_header(&(SpgGpuRows){.inputs = rows}, header);
            memcpy(header_of(s), header, sizeof header);
        } else if (rows) {
            /* read where they are */
        } else {
            /* rounded straight from the caller's floats (or copied), on the OpenMP threads for a chunk of a
               megabyte or more (as training fills its chunks) */
            uint16_t *d16 = s->host_inputs ? (uint16_t *)(s->input.mapped ? s->input.mapped
                                                                          : (char *)s->staging.mapped + s->inputs)
                                           : NULL;
            const float *src = inputs + (size_t)start * in;
            SPINGALETT_PARALLEL_FOR((uint64_t)m * in >= (1u << 18) && m > 1,
                for (int64_t r = 0; r < (int64_t)m; r++) {
                    if (d16) spingalett_round_bf16(d16 + (size_t)r * in, src + (size_t)r * in, in);
                    else memcpy(dst + (size_t)r * in, src + (size_t)r * in, in * sizeof(float));
                }
            );
        }
        g->recording = !g->coefficients;
        SpgGpuCommands *c = chunk_commands(g, s, (uint64_t)m | 1ull << 63 | (gather ? 1ull << 59 : 0) |
                                                     (g->recording ? 1ull << 60 : 0), m, m, false, false, false,
                                           gather ? 1u : 0u, rows && !gather ? &view : NULL);
        g->recording = false;
        if (!c || !spg_gpu_submit(c)) {
            drain(g);
            return false;
        }
        g->coefficients = true;
        s->pending = c;
        s->n = m;
        s->train = false;
        s->dest = outputs + (size_t)start * out;
        g->next = (g->next + 1u) % SLOTS;
    }
    drain(g);
    return !g->lost;
}

bool spingalett_gpu_predict_rows(SpgGpuNet *g, const float *inputs, const SpingalettDeviceData *rows, uint32_t first,
                                 float *outputs, uint32_t n) {
    const SpgBackend saved = spg_gpu_using();
    if (g) spg_gpu_use(g->backend);
    bool result = spingalett_gpu_predict_rows_here(g, inputs, rows, first, outputs, n);
    spg_gpu_use(saved);
    return result;
}

/* ------------------------------------------------------------------------- passes (the step API) */

/*
 * The step API (spingalett_trainer_*) runs one pass at a time and waits for it: a forward pass in
 * training mode whose outputs the caller reads, a backward pass from targets or from the gradient of
 * the caller's own loss, which adds the samples' gradients up, and an optimizer step with their mean.
 * They use slot 0 and its cache, recorded once per size like chunks.
 */

enum { PASS_FORWARD = 1, PASS_LOSS = 2, PASS_GRADS = 3, PASS_STEP = 4 };

static SpgGpuCommands *pass_commands(SpgGpuNet *g, uint32_t n, uint32_t kind, bool add) {
    Slot *s = &g->slots[0];
    bool fresh;
    SpgGpuCommands *c = cache_entry(s, (uint64_t)n | 1ull << 62 | (uint64_t)kind << 58 | (add ? 1ull << 57 : 0), 0, &fresh);
    if (!c || !fresh) return c;
    const NeuralNetwork *net = g->net;
    const uint32_t in = net->topology[0], out = net->topology[net->layers - 1], L = net->layers - 1;
    Recorder *r = spg_gpu_record_begin(c) ? (Recorder *)calloc(1, sizeof *r) : NULL;
    if (!r) return forget(s, c);
    r->c = c;
    r->g = g;
    g->act[0] = s->input;
    g->header = s->header;
    g->targets = s->target;
    if (kind == PASS_FORWARD) {
        copy_in(c, s, (size_t)n * in * (g->half[0] ? 2u : 4u));
        barrier(r);
        record_forward(g, r, n, true);
        barrier(r);
        spg_gpu_copy(c, &g->act[L], 0, &s->staging, s->outputs, (size_t)n * out * 4u);
    } else if (kind == PASS_STEP) {
        copy_in(c, s, 0);
        barrier(r);
        /* the mean of the samples' gradients, added up by the backward passes */
        Access a = {0};
        reads(&a, whole(&g->grads));
        reads(&a, whole(&g->header));
        writes(&a, whole(&g->grads));
        eltwise(g, r, &a, SPG_ELT_SCALE, ACT_NONE, g->grads.address, 0, at(&g->header, 9), 0, g->Wp + g->Bp, 1);
        optimizer(g, r);
    } else {
        /* the targets, or dL/d(output) of the caller's loss, then the backward pass */
        spg_gpu_copy(c, &s->staging, s->targets, &s->target, 0, (size_t)n * out * 4u);
        barrier(r);
        SpgOutputPush p = {g->act[L].address, g->targets.address, g->delta[L].address, s->staging.address + s->losses,
                           n, out};
        uint32_t spec[4] = {kind == PASS_GRADS ? SPG_OUT_GRADS : SPG_OUT_LOSS, net->act_func[L - 1], net->loss_func,
                            0};
        Access a = {0};
        reads(&a, whole(&g->act[L]));
        reads(&a, whole(&g->targets));
        writes(&a, whole(&g->delta[L]));
        writes(&a, span(s->staging.address + s->losses, n));
        output_kernel(r, &a, spec, &p);
        record_backward(g, r, n, 1.0f, add ? 1.0f : 0.0f);
    }
    spg_gpu_barrier_host(c);
    free(r);
    return spg_gpu_record_end(c) ? c : forget(s, c);
}

/* Runs a pass's commands with the header given (NULL: the staging's as it is) and waits. */
static bool run_pass(SpgGpuNet *g, SpgGpuCommands *c, const uint32_t *header) {
    Slot *s = &g->slots[0];
    if (!c || g->lost) return false;
    g->coefficients = false;
    if (header) memcpy(header_of(s), header, SPG_STEP_HEADER * sizeof(uint32_t));
    if (!spg_gpu_submit(c)) return false;
    if (!spg_gpu_wait(c)) g->lost = true;
    return !g->lost;
}

static float *spingalett_gpu_pass_buffers_here(SpgGpuNet *g, float **targets) {
    drain(g);
    g->next = 0;
    Slot *s = &g->slots[0];
    if (targets) *targets = (float *)((char *)s->staging.mapped + s->targets);
    return inputs_of(s);
}

float *spingalett_gpu_pass_buffers(SpgGpuNet *g, float **targets) {
    const SpgBackend saved = spg_gpu_using();
    if (g) spg_gpu_use(g->backend);
    float * result = spingalett_gpu_pass_buffers_here(g, targets);
    spg_gpu_use(saved);
    return result;
}

static const float *spingalett_gpu_pass_forward_here(SpgGpuNet *g, uint32_t n, uint32_t position, uint64_t step) {
    if (!g->training || n == 0 || n > g->capacity) return NULL;
    const SpgGpuStep st = {0.0f, 0.0f, 0.0f, step};
    uint32_t header[SPG_STEP_HEADER];
    step_header(g, &st, position, 1.0f, header);
    stage_inputs(g, &g->slots[0], n);
    if (!run_pass(g, pass_commands(g, n, PASS_FORWARD, false), header)) return NULL;
    Slot *s = &g->slots[0];
    return (const float *)((const char *)s->staging.mapped + s->outputs);
}

const float *spingalett_gpu_pass_forward(SpgGpuNet *g, uint32_t n, uint32_t position, uint64_t step) {
    const SpgBackend saved = spg_gpu_using();
    if (g) spg_gpu_use(g->backend);
    const float * result = spingalett_gpu_pass_forward_here(g, n, position, step);
    spg_gpu_use(saved);
    return result;
}

static bool spingalett_gpu_pass_backward_here(SpgGpuNet *g, uint32_t n, bool from_grads, bool add, float *loss) {
    if (!g->training || n == 0 || n > g->capacity) return false;
    if (!run_pass(g, pass_commands(g, n, from_grads ? PASS_GRADS : PASS_LOSS, add), NULL)) return false;
    if (loss) {
        Slot *s = &g->slots[0];
        const float *losses = (const float *)((const char *)s->staging.mapped + s->losses);
        float sum = 0.0f;
        for (uint32_t k = 0; k < n; k++) sum += losses[k];
        *loss = sum;
    }
    return true;
}

bool spingalett_gpu_pass_backward(SpgGpuNet *g, uint32_t n, bool from_grads, bool add, float *loss) {
    const SpgBackend saved = spg_gpu_using();
    if (g) spg_gpu_use(g->backend);
    bool result = spingalett_gpu_pass_backward_here(g, n, from_grads, add, loss);
    spg_gpu_use(saved);
    return result;
}

static bool spingalett_gpu_pass_step_here(SpgGpuNet *g, const SpgGpuTraining *cfg, const SpgGpuStep *step, float grad_scale) {
    if (!g->training) return false;
    /* the optimizer's settings are recorded in its commands: other settings record them again */
    Slot *s = &g->slots[0];
    const uint64_t seed = g->cfg.dropout_seed;
    SpgGpuTraining want = *cfg;
    want.dropout_seed = seed;
    if (memcmp(&want, &g->cfg, sizeof want) != 0) {
        g->cfg = want;
        for (uint32_t e = 0; e < s->used; e++)
            if (s->cache[e].key == ((uint64_t)1u << 62 | (uint64_t)PASS_STEP << 58)) {
                forget(s, s->cache[e].commands);
                break;
            }
    }
    if ((cfg->optimizer == OPTIMIZER_MOMENTUM || cfg->optimizer == OPTIMIZER_ADAM || cfg->optimizer == OPTIMIZER_ADAMW) &&
        !g->moment1.buffer)
        return false;
    if ((cfg->optimizer == OPTIMIZER_RMSPROP || cfg->optimizer == OPTIMIZER_ADAM || cfg->optimizer == OPTIMIZER_ADAMW) &&
        !g->moment2.buffer)
        return false;
    uint32_t header[SPG_STEP_HEADER];
    step_header(g, step, 0, grad_scale, header);
    return run_pass(g, pass_commands(g, 0, PASS_STEP, false), header);
}

bool spingalett_gpu_pass_step(SpgGpuNet *g, const SpgGpuTraining *cfg, const SpgGpuStep *step, float grad_scale) {
    const SpgBackend saved = spg_gpu_using();
    if (g) spg_gpu_use(g->backend);
    bool result = spingalett_gpu_pass_step_here(g, cfg, step, grad_scale);
    spg_gpu_use(saved);
    return result;
}
