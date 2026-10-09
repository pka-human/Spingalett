/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

#include "Spingalett.Private.h"
#include "Spingalett.Gpu.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

#if defined(_OPENMP)
#include <omp.h>
#endif

static void compute_offsets(NeuralNetwork *net) {
    uint64_t n = 0, w = 0, b = 0;
    net->neuron_offsets[0] = 0;
    for (uint32_t l = 0; l < net->layers; l++) {
        n += (uint64_t)net->topology[l];
        net->neuron_offsets[l + 1] = n;
        if (l + 1 < net->layers) {
            net->weight_offsets[l] = w;
            net->bias_offsets[l] = b;
            w += (uint64_t)spingalett_weight_rows(net, l) * spingalett_weight_row_len(net, l);
            b += (uint64_t)spingalett_weight_rows(net, l);
        }
    }
    net->total_neurons = n;
    net->total_weights = w;
    net->total_biases = b;
}

NeuralNetwork *new_spingalett_struct_arguments(NeuralNetworkArgs args) {
    LossFunction loss_func = args.loss_func;
    if ((unsigned)loss_func >= LOSS_COUNT) {
        set_error(SPINGALETT_ERR_INVALID, "Invalid loss function");
        return NULL;
    }
    spingalett_log(LOG_INFO, "Creating new network with %s loss function", loss_func_names[loss_func]);
    NeuralNetwork *net = (NeuralNetwork *)calloc(1, sizeof(NeuralNetwork));
    if (!net) { set_error(SPINGALETT_ERR_ALLOC, "NeuralNetwork calloc failed"); return NULL; }
    net->loss_func = loss_func;
    return net;
}

static bool layer_error(const char *msg) {
    set_error(SPINGALETT_ERR_INVALID, msg);
    spingalett_log(LOG_ERROR, "%s", msg);
    return false;
}

/* The layers a new layer reads: those of args->inputs (input_count of them, or up to the last
   nonzero entry), or the layer before it; false (error set) when they are not earlier layers or do
   not suit the layer's kind. */
static bool layer_inputs(const NeuralNetwork *net, const LayerArgs *args, uint32_t *inputs, uint32_t *count) {
    uint32_t n = args->input_count;
    if (n == 0)
        for (uint32_t k = SPINGALETT_MAX_INPUTS; k > 0; k--)
            if (args->inputs[k - 1] != 0) { n = k; break; }
    if (net->layers == 0) {
        *count = 0;
        return n == 0 ? true : layer_error("The input layer reads no other layers");
    }
    if (n > SPINGALETT_MAX_INPUTS)
        return layer_error("A layer reads at most SPINGALETT_MAX_INPUTS (16) layers");
    if (n == 0) {
        inputs[0] = net->layers - 1;
        *count = 1;
    } else {
        for (uint32_t k = 0; k < n; k++) {
            if (args->inputs[k] >= net->layers)
                return layer_error("A layer can only read layers added before it");
            inputs[k] = args->inputs[k];
        }
        *count = n;
    }
    if (*count > 1 && args->type != LAYER_ADD && args->type != LAYER_CONCAT)
        return layer_error("Only add and concatenation layers read several layers");
    return true;
}

/* The shape of a new layer from its arguments and its inputs; false (error set) when they do not
   fit together. */
static bool layer_shape(const NeuralNetwork *net, const LayerArgs *args, const uint32_t *inputs, uint32_t count,
                        LayerShape *shape, uint32_t *outputs) {
    *shape = (LayerShape){.type = args->type, .height = 1, .width = 1};
    uint64_t units;
    if (net->layers == 0) {
        if (args->type != LAYER_DENSE)
            return layer_error("The first layer is the input layer; give it a size or a shape, not a type");
        if (args->height || args->width || args->channels) {
            shape->height = args->height ? args->height : 1;
            shape->width = args->width ? args->width : 1;
            shape->channels = args->channels ? args->channels : 1;
            units = (uint64_t)shape->height * shape->width * shape->channels;
            if (args->neurons_amount && args->neurons_amount != units)
                return layer_error("Input layer: neurons_amount differs from height x width x channels");
        } else {
            units = args->neurons_amount;
            shape->channels = args->neurons_amount;
        }
    } else if (args->type == LAYER_DENSE) {
        units = args->neurons_amount;
        shape->channels = args->neurons_amount;
    } else if (args->type == LAYER_ADD || args->type == LAYER_CONCAT) {
        /* adding: one shape for all; concatenating: one height and width, the channels summed */
        const LayerShape *first = &net->shapes[inputs[0]];
        uint64_t channels = 0;
        for (uint32_t k = 0; k < count; k++) {
            const LayerShape *in = &net->shapes[inputs[k]];
            if (in->height != first->height || in->width != first->width ||
                (args->type == LAYER_ADD && in->channels != first->channels))
                return layer_error(args->type == LAYER_ADD ? "Added layers must have the same shape"
                                                           : "Concatenated layers must have the same height and width");
            channels += in->channels;
        }
        if (args->type == LAYER_ADD) channels = first->channels;
        if (channels > UINT32_MAX)
            return layer_error("Layers are limited to 2^32 - 1 outputs");
        shape->height = first->height;
        shape->width = first->width;
        shape->channels = (uint32_t)channels;
        units = (uint64_t)first->height * first->width * channels;
    } else if (args->type == LAYER_GLOBAL_AVG_POOL) {
        shape->channels = net->shapes[inputs[0]].channels;
        units = shape->channels;
    } else if (args->type == LAYER_UPSAMPLE) {
        /* every cell into stride_h x stride_w cells */
        const LayerShape *in = &net->shapes[inputs[0]];
        uint32_t sh = args->stride_h ? args->stride_h : args->stride ? args->stride : 2u;
        uint32_t sw = args->stride_w ? args->stride_w : args->stride ? args->stride : 2u;
        if ((unsigned)args->upsample >= UPSAMPLE_MODE_COUNT)
            return layer_error("Invalid upsampling mode");
        if ((uint64_t)in->height * sh > 65535 || (uint64_t)in->width * sw > 65535)
            return layer_error("Layer heights and widths are limited to 65535");
        shape->stride_h = sh;
        shape->stride_w = sw;
        shape->mode = args->upsample;
        shape->height = in->height * sh;
        shape->width = in->width * sw;
        shape->channels = in->channels;
        units = (uint64_t)shape->height * shape->width * shape->channels;
    } else if (args->type == LAYER_LAYER_NORM) {
        /* the input's shape, normalized per cell */
        const LayerShape *in = &net->shapes[inputs[0]];
        if (!(args->epsilon >= 0.0f && args->epsilon < 1.0f))
            return layer_error("Layer normalization needs epsilon in [0, 1)");
        shape->height = in->height;
        shape->width = in->width;
        shape->channels = in->channels;
        shape->eps = args->epsilon > 0.0f ? args->epsilon : 1e-5f;
        units = net->topology[inputs[0]];
    } else if (args->type == LAYER_CONV_TRANSPOSE2D) {
        /* each input cell adds its window (kernel_h x kernel_w) to the output from (y stride - padding,
           x stride - padding) on */
        const LayerShape *in = &net->shapes[inputs[0]];
        uint32_t kh = args->kernel_h ? args->kernel_h : args->kernel;
        uint32_t kw = args->kernel_w ? args->kernel_w : args->kernel;
        uint32_t sh = args->stride_h ? args->stride_h : args->stride ? args->stride : 1u;
        uint32_t sw = args->stride_w ? args->stride_w : args->stride ? args->stride : 1u;
        uint32_t ph = args->padding_h ? args->padding_h : args->padding;
        uint32_t pw = args->padding_w ? args->padding_w : args->padding;
        uint32_t oh = args->output_padding_h ? args->output_padding_h : args->output_padding;
        uint32_t ow = args->output_padding_w ? args->output_padding_w : args->output_padding;
        uint32_t groups = args->groups ? args->groups : 1u;
        if (kh == 0 || kw == 0)
            return layer_error("A transposed convolution needs a kernel size");
        if (kh > 255 || kw > 255 || sh > 255 || sw > 255)
            return layer_error("Transposed convolutions are limited to kernels and strides of 255");
        if (ph >= kh || pw >= kw)
            return layer_error("Padding must be smaller than the kernel");
        if (oh >= sh || ow >= sw)
            return layer_error("Output padding must be smaller than the stride");
        if (args->filters == 0)
            return layer_error("A transposed convolution layer needs filters > 0");
        if (in->channels % groups != 0 || args->filters % groups != 0)
            return layer_error("Convolution groups must divide the input channels and the filters");
        int64_t height = ((int64_t)in->height - 1) * sh - 2 * (int64_t)ph + kh + oh;
        int64_t width = ((int64_t)in->width - 1) * sw - 2 * (int64_t)pw + kw + ow;
        if (height < 1 || width < 1)
            return layer_error("The padding leaves the transposed convolution no output");
        if (height > 65535 || width > 65535)
            return layer_error("Layer heights and widths are limited to 65535");
        shape->groups = groups;
        shape->kernel_h = kh; shape->kernel_w = kw;
        shape->stride_h = sh; shape->stride_w = sw;
        shape->pad_h = ph; shape->pad_w = pw;
        shape->height = (uint32_t)height;
        shape->width = (uint32_t)width;
        shape->channels = args->filters;
        units = (uint64_t)shape->height * shape->width * shape->channels;
    } else if (args->type == LAYER_BATCH_NORM) {
        /* the input's shape, normalized per channel */
        const LayerShape *in = &net->shapes[inputs[0]];
        if (!(args->epsilon >= 0.0f && args->epsilon < 1.0f) || !(args->momentum >= 0.0f && args->momentum <= 1.0f))
            return layer_error("Batch normalization needs epsilon in [0, 1) and momentum in [0, 1]");
        shape->height = in->height;
        shape->width = in->width;
        shape->channels = in->channels;
        shape->eps = args->epsilon > 0.0f ? args->epsilon : 1e-5f;
        shape->momentum = args->momentum > 0.0f ? args->momentum : 0.1f;
        units = net->topology[inputs[0]];
    } else {
        const LayerShape *in = &net->shapes[inputs[0]];
        bool conv = args->type == LAYER_CONV2D;
        uint32_t kh = args->kernel_h ? args->kernel_h : args->kernel;
        uint32_t kw = args->kernel_w ? args->kernel_w : args->kernel;
        if (kh == 0 || kw == 0)
            return layer_error("Convolution and pooling layers need a kernel size");
        uint32_t sh = args->stride_h ? args->stride_h : args->stride ? args->stride : conv ? 1u : kh;
        uint32_t sw = args->stride_w ? args->stride_w : args->stride ? args->stride : conv ? 1u : kw;
        uint32_t ph = args->padding_h ? args->padding_h : args->padding;
        uint32_t pw = args->padding_w ? args->padding_w : args->padding;
        if (kh > 65535 || kw > 65535 || sh > 65535 || sw > 65535)
            return layer_error("Kernel sizes and strides are limited to 65535");
        if (ph >= kh || pw >= kw)
            return layer_error("Padding must be smaller than the kernel");
        if ((uint64_t)in->height + 2u * ph < kh || (uint64_t)in->width + 2u * pw < kw)
            return layer_error("The kernel is larger than the (padded) input");
        if (conv && args->filters == 0)
            return layer_error("A convolution layer needs filters > 0");
        uint32_t groups = conv && args->groups ? args->groups : 1u;
        if (conv && (in->channels % groups != 0 || args->filters % groups != 0))
            return layer_error("Convolution groups must divide the input channels and the filters");
        shape->groups = groups;
        shape->kernel_h = kh; shape->kernel_w = kw;
        shape->stride_h = sh; shape->stride_w = sw;
        shape->pad_h = ph; shape->pad_w = pw;
        shape->height = (uint32_t)(((uint64_t)in->height + 2u * ph - kh) / sh + 1u);
        shape->width = (uint32_t)(((uint64_t)in->width + 2u * pw - kw) / sw + 1u);
        shape->channels = conv ? args->filters : in->channels;
        units = (uint64_t)shape->height * shape->width * shape->channels;
        if (conv && (uint64_t)kh * kw * (in->channels / groups) > UINT32_MAX)
            return layer_error("Convolution windows are limited to 2^32 - 1 inputs");
    }
    if (units == 0)
        return layer_error("A layer needs at least one output (neurons_amount, filters or shape)");
    if (units > UINT32_MAX)
        return layer_error("Layers are limited to 2^32 - 1 outputs");
    if (shape->height > 65535 || shape->width > 65535)
        return layer_error("Layer heights and widths are limited to 65535");
    *outputs = (uint32_t)units;
    return true;
}

/* An array of `need` floats whose first `keep` are those of `old`, the rest zero: `old` itself when
   it holds `cap` >= need floats, a new array of `grown` floats otherwise. */
static float *grow_floats(float *old, uint64_t keep, uint64_t need, uint64_t cap, uint64_t grown) {
    float *p = old;
    if (!old || need > cap) {
        p = (float *)spingalett_aligned_alloc((size_t)grown * sizeof(float));
        if (!p) return NULL;
        if (keep) memcpy(p, old, (size_t)keep * sizeof(float));
    }
    memset(p + keep, 0, (size_t)(need - keep) * sizeof(float));
    return p;
}

/* The capacity an array that must hold `need` floats moves to: half as much again as it had, so that
   building a network layer by layer copies its parameters a few times in all rather than once per
   layer. The room past `need` is never written, which leaves the pages of large arrays unused. */
static uint64_t grown_capacity(uint64_t need, uint64_t cap) {
    uint64_t grown = cap + cap / 2;
    return grown > need ? grown : need;
}

bool spingalett_network_reserve(NeuralNetwork *net, uint64_t neurons, uint64_t weights, uint64_t biases) {
    if (!net || net->layers != 0 || net->neurons) return false;
    if (weights == 0) weights = 1;
    if (biases == 0) biases = 1;
    float *n = (float *)spingalett_aligned_calloc((size_t)neurons + 1, sizeof(float)), *b[3];
    float *w = (float *)spingalett_aligned_calloc((size_t)weights, sizeof(float));
    bool ok = n && w;
    for (int k = 0; k < 3; k++) ok = (b[k] = (float *)spingalett_aligned_calloc((size_t)biases, sizeof(float))) && ok;
    if (!ok) {
        spingalett_aligned_free(n);
        spingalett_aligned_free(w);
        for (int k = 0; k < 3; k++) spingalett_aligned_free(b[k]);
        return false;                   /* the layers then allocate as they are added */
    }
    net->neurons = n;
    net->weights = w;
    net->biases = b[0]; net->running_mean = b[1]; net->running_var = b[2];
    net->cap_neurons = neurons + 1;
    net->cap_weights = weights;
    net->cap_biases = biases;
    return true;
}

bool spingalett_network_hold_gpu(NeuralNetwork *net, bool wait) {
    bool expected = false;
    while (!atomic_compare_exchange_weak(&net->gpu_busy, &expected, true)) {
        if (!wait && expected) return false;
        expected = false;
    }
    return true;
}

void spingalett_network_let_go_gpu(NeuralNetwork *net) {
    atomic_store(&net->gpu_busy, false);
}

void spingalett_network_sync(const NeuralNetwork *cnet) {
    NeuralNetwork *net = (NeuralNetwork *)cnet;         /* the values the network holds do not change */
    if (!net || !atomic_load(&net->gpu_newer)) return;
    /* readers on several threads (predict() may run concurrently, on the kept copy too) bring them back
       once, and not while a predict() runs on it */
    spingalett_network_hold_gpu(net, true);
    if (atomic_load(&net->gpu_newer) && net->gpu_trainer && !spingalett_gpu_download(net->gpu_trainer))
        spingalett_log(LOG_ERROR, "The parameters could not be copied back from the GPU");
    atomic_store(&net->gpu_newer, false);
    spingalett_network_let_go_gpu(net);
}

void spingalett_network_release_gpu(NeuralNetwork *net) {
    if (!net || !net->gpu_kept) return;
    spingalett_network_sync(net);
    spingalett_network_hold_gpu(net, true);
    spingalett_gpu_net_free(net->gpu_trainer);
    net->gpu_trainer = NULL;
    net->gpu_kept = false;
    atomic_store(&net->gpu_newer, false);
    spingalett_network_let_go_gpu(net);
}

void spingalett_network_written(NeuralNetwork *net) {
    if (net) net->param_version++;
}

bool spingalett_training_state(NeuralNetwork *net) {
    if (net->grad_weights) return true;
    const size_t nw = net->cap_weights ? (size_t)net->cap_weights : 1u, nb = net->cap_biases ? (size_t)net->cap_biases : 1u;
    float *w[3], *b[3];
    bool ok = true;
    for (int k = 0; k < 3; k++) {
        ok = (w[k] = (float *)spingalett_aligned_calloc(nw, sizeof(float))) && ok;
        ok = (b[k] = (float *)spingalett_aligned_calloc(nb, sizeof(float))) && ok;
    }
    if (!ok) {
        for (int k = 0; k < 3; k++) { spingalett_aligned_free(w[k]); spingalett_aligned_free(b[k]); }
        set_error(SPINGALETT_ERR_ALLOC, "Allocation of the gradients and optimizer state failed");
        return false;
    }
    net->grad_weights = w[0]; net->opt_m_weights = w[1]; net->opt_v_weights = w[2];
    net->grad_biases = b[0]; net->opt_m_biases = b[1]; net->opt_v_biases = b[2];
    net->cap_weights = nw;
    net->cap_biases = nb;
    return true;
}

bool spingalett_add_layer(LayerArgs args) {
    NeuralNetwork *net = args.net;
    ActivationFunction act_func = args.act_func;
    WeightInitialization wi = args.weight_initialization;
    float dropout_rate = args.dropout_rate;

    if (!net)
        return layer_error("layer: net is NULL");
    if ((unsigned)args.type >= LAYER_TYPE_COUNT)
        return layer_error("Invalid layer type");
    if (!(dropout_rate >= 0.0f && dropout_rate < 1.0f))
        return layer_error("Dropout rate must be in [0, 1)");

    uint32_t inputs[SPINGALETT_MAX_INPUTS], input_count;
    LayerShape shape;
    uint32_t neurons_amount;
    if (!layer_inputs(net, &args, inputs, &input_count) ||
        !layer_shape(net, &args, inputs, input_count, &shape, &neurons_amount))
        return false;
    spingalett_batch_workspace_free(net->forward_ws);     /* made for the old layers */
    net->forward_ws = NULL;
    spingalett_gpu_net_free(atomic_exchange(&net->gpu_predict, NULL));
    spingalett_network_release_gpu(net);                    /* its parameters back first */
    bool pooling = shape.type == LAYER_MAX_POOL2D || shape.type == LAYER_AVG_POOL2D ||
                   shape.type == LAYER_GLOBAL_AVG_POOL || shape.type == LAYER_UPSAMPLE;
    if (pooling) act_func = ACT_NONE;
    static const char *const type_names[] = {"dense", "conv2d", "max_pool2d", "avg_pool2d", "batch_norm", "add",
                                             "concat", "global_avg_pool2d", "conv_transpose2d", "upsample2d",
                                             "layer_norm"};

    uint32_t nl = net->layers + 1;

    if (nl == 1) {
        spingalett_log(LOG_INFO, "Layer #0: input %u x %u x %u", shape.height, shape.width, shape.channels);
        if (dropout_rate > 0.0f) {
            spingalett_log(LOG_WARNING, "Dropout is not applied to the input layer; ignoring rate %g", (double)dropout_rate);
            dropout_rate = 0.0f;
        }
    } else {
        if ((unsigned)act_func >= ACT_COUNT)
            return layer_error("Invalid activation function");
        if ((unsigned)wi >= WEIGHT_INITIALIZATION_COUNT)
            return layer_error("Invalid weight initialization");
        bool chained = input_count == 1 && inputs[0] == net->layers - 1;
        char from[96] = "";
        for (uint32_t k = 0, used = 0; !chained && k < input_count && used + 12 < sizeof from; k++)
            used += (uint32_t)snprintf(from + used, sizeof from - used, "%s%u", k ? ", " : ", reads ", inputs[k]);
        spingalett_log(LOG_INFO, "Layer #%u: %s, output %u x %u x %u, activation %s, dropout %g%s", nl - 1,
                       type_names[shape.type], shape.height, shape.width, shape.channels,
                       act_func_names[act_func], (double)dropout_rate, from);
    }

    /* the new layer's parameters, as compute_offsets() will count them */
    uint32_t prev_neurons = (net->layers > 0) ? net->topology[inputs[0]] : 0;
    uint32_t rows = 0, row_len = 0;
    if (nl > 1 && shape.type == LAYER_DENSE) { rows = neurons_amount; row_len = prev_neurons; }
    if (nl > 1 && spingalett_filters(shape.type)) {
        rows = shape.channels;
        row_len = shape.kernel_h * shape.kernel_w * (net->shapes[inputs[0]].channels / shape.groups);
    }
    if (nl > 1 && spingalett_normalization(shape.type)) { rows = shape.channels; row_len = 1; }
    uint64_t add_w = (uint64_t)rows * row_len;
    uint64_t add_b = rows;

    uint64_t new_tn = net->total_neurons + (uint64_t)neurons_amount;
    uint64_t new_tw = net->total_weights + add_w;
    uint64_t new_tb = net->total_biases  + add_b;

    uint32_t           *t_topo    = (uint32_t *)           malloc(nl * sizeof(uint32_t));
    LayerShape         *t_shapes  = (LayerShape *)         malloc(nl * sizeof(LayerShape));
    uint64_t           *t_noff    = (uint64_t *)           calloc(nl + 1, sizeof(uint64_t));
    uint64_t           *t_woff    = (uint64_t *)           calloc(nl, sizeof(uint64_t));
    uint64_t           *t_boff    = (uint64_t *)           calloc(nl, sizeof(uint64_t));
    float              *t_drop    = (float *)              malloc(nl * sizeof(float));
    uint32_t           *t_ioff    = (uint32_t *)           malloc((nl + 1) * sizeof(uint32_t));
    uint32_t old_inputs = net->layers ? net->input_offsets[net->layers] : 0u;
    uint32_t           *t_ilist   = (uint32_t *)           malloc((old_inputs + input_count + 1) * sizeof(uint32_t));

    /* The big arrays stay where they are when they have room (a loader reserved the final sizes, or
       an earlier move left some); otherwise they move to larger arrays: the old values copied, the
       rest zeroed. Gradients and optimizer state exist once training has started only. */
    ActivationFunction *t_act     = NULL;
    /* sizes of at least 1: pooling layers add no parameters, and a network of pooling layers has none */
    size_t nw = new_tw ? new_tw : 1, nb = new_tb ? new_tb : 1;
    const uint64_t cap_w = nw > net->cap_weights ? grown_capacity(nw, net->cap_weights) : net->cap_weights;
    const uint64_t cap_b = nb > net->cap_biases ? grown_capacity(nb, net->cap_biases) : net->cap_biases;
    const uint64_t cap_n = new_tn > net->cap_neurons ? grown_capacity(new_tn, net->cap_neurons) : net->cap_neurons;
    const bool state = net->grad_weights != NULL;
    float *const old_w[4] = {net->weights, net->grad_weights, net->opt_m_weights, net->opt_v_weights};
    float *const old_b[6] = {net->biases, net->running_mean, net->running_var, net->grad_biases, net->opt_m_biases,
                             net->opt_v_biases};
    float *t_neurons = grow_floats(net->neurons, net->total_neurons, new_tn, net->cap_neurons, cap_n);
    float *t_wa[4] = {NULL, NULL, NULL, NULL}, *t_ba[6] = {NULL, NULL, NULL, NULL, NULL, NULL};
    bool ok = t_topo && t_shapes && t_noff && t_woff && t_boff && t_neurons && t_drop && t_ioff && t_ilist;
    if (nl > 1) {
        t_act = (ActivationFunction *)malloc((nl - 1) * sizeof(ActivationFunction));
        ok = ok && t_act;
        for (int k = 0; k < (state ? 4 : 1); k++)
            ok = ok && (t_wa[k] = grow_floats(old_w[k], net->total_weights, nw, net->cap_weights, cap_w));
        for (int k = 0; k < (state ? 6 : 3); k++)
            ok = ok && (t_ba[k] = grow_floats(old_b[k], net->total_biases, nb, net->cap_biases, cap_b));
    }

    if (!ok) {
        free(t_topo); free(t_shapes); free(t_noff); free(t_woff); free(t_boff); free(t_drop);
        free(t_ioff); free(t_ilist);
        if (t_neurons != net->neurons) spingalett_aligned_free(t_neurons);
        free(t_act);
        for (int k = 0; k < 4; k++) if (t_wa[k] != old_w[k]) spingalett_aligned_free(t_wa[k]);
        for (int k = 0; k < 6; k++) if (t_ba[k] != old_b[k]) spingalett_aligned_free(t_ba[k]);
        set_error(SPINGALETT_ERR_ALLOC, "Layer allocation failed");
        return false;
    }
    float *t_w = t_wa[0], *t_gw = t_wa[1], *t_mw = t_wa[2], *t_vw = t_wa[3];
    float *t_b = t_ba[0], *t_rm = t_ba[1], *t_rv = t_ba[2], *t_gb = t_ba[3], *t_mb = t_ba[4], *t_vb = t_ba[5];

    if (net->layers > 0) {
        memcpy(t_topo, net->topology, net->layers * sizeof(uint32_t));
        memcpy(t_shapes, net->shapes, net->layers * sizeof(LayerShape));
    }
    t_topo[net->layers] = neurons_amount;
    t_shapes[net->layers] = shape;

    if (net->layers > 0)
        memcpy(t_drop, net->dropout_rates, net->layers * sizeof(float));
    t_drop[net->layers] = dropout_rate;

    if (net->layers > 0) {
        memcpy(t_ioff, net->input_offsets, (net->layers + 1) * sizeof(uint32_t));
        memcpy(t_ilist, net->input_list, old_inputs * sizeof(uint32_t));
    } else {
        t_ioff[0] = 0;
    }
    memcpy(t_ilist + old_inputs, inputs, input_count * sizeof(uint32_t));
    t_ioff[nl] = old_inputs + input_count;
    bool graph = net->graph || shape.type == LAYER_ADD || shape.type == LAYER_CONCAT ||
                 shape.type == LAYER_GLOBAL_AVG_POOL || (nl > 1 && !(input_count == 1 && inputs[0] == nl - 2));

    if (nl > 1) {
        if (net->layers > 1)
            memcpy(t_act, net->act_func, (net->layers - 1) * sizeof(ActivationFunction));
        t_act[net->layers - 1] = act_func;

        /* Standard deviations: Glorot sqrt(2 / (fan_in + fan_out)), He sqrt(2 / fan_in),
           LeCun sqrt(1 / fan_in); a filter's fan-in is its window, its fan-out the window times
           the filters of its group. An output of a transposed convolution gathers about its window
           over the strides: that is its fan-in. */
        float fan_in = (float)row_len;
        if (shape.type == LAYER_CONV_TRANSPOSE2D && row_len >= shape.stride_h * shape.stride_w)
            fan_in = (float)row_len / (float)(shape.stride_h * shape.stride_w);
        float fan_out = spingalett_filters(shape.type)
                      ? (float)shape.kernel_h * (float)shape.kernel_w * (float)(rows / shape.groups) : (float)rows;
        float scale = 1.0f;
        if (wi == WEIGHT_INITIALIZATION_XAVIER)
            scale = sqrtf(2.0f / (fan_in + fan_out));
        else if (wi == WEIGHT_INITIALIZATION_HE)
            scale = sqrtf(2.0f / fan_in);
        else if (wi == WEIGHT_INITIALIZATION_LECUN)
            scale = sqrtf(1.0f / fan_in);

        for (uint64_t idx = 0; idx < add_w; idx++) {
            uint64_t pos = net->total_weights + idx;
            if (spingalett_normalization(shape.type))
                t_w[pos] = 1.0f;                        /* gamma; beta starts at 0 */
            else if (wi == WEIGHT_INITIALIZATION_RANDOM)
                t_w[pos] = random_uniform_weight();
            else if (wi != WEIGHT_INITIALIZATION_NONE)
                t_w[pos] = random_normal_weight() * scale;
        }
        if (shape.type == LAYER_BATCH_NORM)
            for (uint64_t idx = 0; idx < add_b; idx++)
                t_rv[net->total_biases + idx] = 1.0f;   /* running variance; the mean starts at 0 */
    }

    free(net->topology);
    free(net->shapes);
    free(net->act_func);
    free(net->dropout_rates);
    free(net->neuron_offsets);
    free(net->weight_offsets);
    free(net->bias_offsets);
    free(net->input_offsets);
    free(net->input_list);
    if (t_neurons != net->neurons) {
        spingalett_aligned_free(net->neurons);
        net->cap_neurons = cap_n;
    }
    if (nl > 1) {
        if (t_w != old_w[0]) net->cap_weights = cap_w;
        if (t_b != old_b[0]) net->cap_biases = cap_b;
        for (int k = 0; k < (state ? 4 : 1); k++) if (t_wa[k] != old_w[k]) spingalett_aligned_free(old_w[k]);
        for (int k = 0; k < (state ? 6 : 3); k++) if (t_ba[k] != old_b[k]) spingalett_aligned_free(old_b[k]);
    } else {                            /* the input layer: arrays reserved for the parameters stay */
        t_w = old_w[0]; t_gw = old_w[1]; t_mw = old_w[2]; t_vw = old_w[3];
        t_b = old_b[0]; t_rm = old_b[1]; t_rv = old_b[2]; t_gb = old_b[3]; t_mb = old_b[4]; t_vb = old_b[5];
    }

    net->layers         = nl;
    net->input_offsets  = t_ioff;
    net->input_list     = t_ilist;
    net->graph          = graph;
    net->topology       = t_topo;
    net->shapes         = t_shapes;
    net->act_func       = t_act;
    net->dropout_rates  = t_drop;
    net->neuron_offsets  = t_noff;
    net->weight_offsets  = t_woff;
    net->bias_offsets    = t_boff;
    net->neurons         = t_neurons;
    net->weights         = t_w;
    net->biases          = t_b;
    net->grad_weights    = t_gw;
    net->grad_biases     = t_gb;
    net->opt_m_weights   = t_mw;
    net->opt_m_biases    = t_mb;
    net->opt_v_weights   = t_vw;
    net->opt_v_biases    = t_vb;
    net->running_mean    = t_rm;
    net->running_var     = t_rv;
    net->total_neurons   = new_tn;
    net->total_weights   = new_tw;
    net->total_biases    = new_tb;

    compute_offsets(net);
    return true;
}

uint32_t layer_struct_arguments(LayerArgs args) {
    return spingalett_add_layer(args) ? args.net->layers - 1 : SPINGALETT_NO_LAYER;
}

bool spingalett_check_graph(const NeuralNetwork *net, const char *who) {
    if (!net->graph) return true;
    /* every layer but the last feeds a later one */
    bool *used = (bool *)calloc(net->layers, sizeof(bool));
    if (!used) {
        set_error(SPINGALETT_ERR_ALLOC, "allocation failed");
        return false;
    }
    for (uint32_t l = 1; l < net->layers; l++)
        for (uint32_t k = 0; k < spingalett_input_count(net, l); k++) used[spingalett_inputs(net, l)[k]] = true;
    uint32_t unused = net->layers;
    for (uint32_t l = 0; l + 1 < net->layers && unused == net->layers; l++)
        if (!used[l]) unused = l;
    free(used);
    if (unused == net->layers) return true;
    char msg[160];
    snprintf(msg, sizeof msg, "%s: the output of layer %u is not used; every layer but the last must feed a later one",
             who, unused);
    set_error(SPINGALETT_ERR_INVALID, msg);
    spingalett_log(LOG_ERROR, "%s", msg);
    return false;
}

LayerArgs spingalett_layer_args(NeuralNetwork *net, uint32_t l) {
    const LayerShape *s = &net->shapes[l];
    LayerArgs a = {0};
    a.net = net;
    a.weight_initialization = WEIGHT_INITIALIZATION_NONE;
    a.dropout_rate = net->dropout_rates[l];
    if (l == 0) {
        a.neurons_amount = net->topology[0];
        a.height = s->height;
        a.width = s->width;
        a.channels = s->channels;
        return a;
    }
    a.type = s->type;
    a.act_func = net->act_func[l - 1];
    a.input_count = spingalett_input_count(net, l);
    memcpy(a.inputs, spingalett_inputs(net, l), a.input_count * sizeof(uint32_t));
    switch (s->type) {
        case LAYER_DENSE:
            a.neurons_amount = net->topology[l];
            break;
        case LAYER_BATCH_NORM:
        case LAYER_LAYER_NORM:
            a.epsilon = s->eps;
            a.momentum = s->momentum;
            break;
        case LAYER_ADD:
        case LAYER_CONCAT:
        case LAYER_GLOBAL_AVG_POOL:
            break;
        case LAYER_UPSAMPLE:
            a.stride_h = s->stride_h; a.stride_w = s->stride_w;
            a.upsample = (UpsampleMode)s->mode;
            break;
        case LAYER_CONV_TRANSPOSE2D: {
            const LayerShape *in = &net->shapes[spingalett_source(net, l)];
            a.output_padding_h = s->height - ((in->height - 1) * s->stride_h - 2 * s->pad_h + s->kernel_h);
            a.output_padding_w = s->width - ((in->width - 1) * s->stride_w - 2 * s->pad_w + s->kernel_w);
        }
            [[fallthrough]];                /* the window */
        default:
            a.filters = spingalett_filters(s->type) ? s->channels : 0u;
            a.groups = spingalett_filters(s->type) ? s->groups : 0u;
            a.kernel_h = s->kernel_h; a.kernel_w = s->kernel_w;
            a.stride_h = s->stride_h; a.stride_w = s->stride_w;
            a.padding_h = s->pad_h; a.padding_w = s->pad_w;
            break;
    }
    return a;
}

uint32_t spingalett_layer_count(const NeuralNetwork *net) {
    return net ? net->layers : 0;
}

bool spingalett_network_layer(const NeuralNetwork *net, uint32_t index, SpingalettNetworkLayer *layer) {
    if (!net || !layer || index >= net->layers) {
        set_error(SPINGALETT_ERR_INVALID, "spingalett_network_layer: NULL argument or index out of range");
        return false;
    }
    const LayerShape *s = &net->shapes[index];
    memset(layer, 0, sizeof *layer);        /* padding bytes too, so equal layers compare equal */
    layer->type = s->type;
    layer->height = s->height;
    layer->width = s->width;
    layer->channels = s->channels;
    layer->outputs = net->topology[index];
    layer->activation = index > 0 ? net->act_func[index - 1] : ACT_NONE;
    layer->dropout_rate = net->dropout_rates[index];
    layer->kernel_h = s->kernel_h;
    layer->kernel_w = s->kernel_w;
    layer->stride_h = s->stride_h;
    layer->stride_w = s->stride_w;
    layer->padding_h = s->pad_h;
    layer->padding_w = s->pad_w;
    layer->groups = spingalett_filters(s->type) ? s->groups : 0u;
    layer->epsilon = s->eps;
    layer->momentum = s->momentum;
    layer->upsample = (UpsampleMode)s->mode;
    if (index > 0) {
        layer->bias_count = spingalett_weight_rows(net, index - 1);
        layer->weight_count = layer->bias_count * spingalett_weight_row_len(net, index - 1);
        layer->input_count = spingalett_input_count(net, index);
        memcpy(layer->inputs, spingalett_inputs(net, index), layer->input_count * sizeof(uint32_t));
    }
    return true;
}

uint32_t spingalett_input_size(const NeuralNetwork *net) {
    return net && net->layers ? net->topology[0] : 0;
}

uint32_t spingalett_output_size(const NeuralNetwork *net) {
    return net && net->layers ? net->topology[net->layers - 1] : 0;
}

uint64_t spingalett_parameter_count(const NeuralNetwork *net) {
    return net ? net->total_weights + net->total_biases : 0;
}

LossFunction spingalett_network_loss(const NeuralNetwork *net) {
    return net ? net->loss_func : LOSS_MSE;
}

uint64_t spingalett_optimizer_steps(const NeuralNetwork *net) {
    return net ? net->time_step : 0;
}

/* The array of `kind` feeding layer index and its length; NULL when out of range (error set, *valid
   false) or when it is a gradient of a network that has not trained yet (*valid true). */
static float *parameter_block(const NeuralNetwork *net, uint32_t index, ParameterKind kind, uint64_t count,
                              const char *who, bool *valid) {
    *valid = false;
    if (!net || index == 0 || index >= net->layers || (unsigned)kind >= PARAM_KIND_COUNT) {
        set_error(SPINGALETT_ERR_INVALID, who);
        return NULL;
    }
    uint64_t rows = spingalett_weight_rows(net, index - 1);
    bool weights = kind == PARAM_WEIGHTS || kind == PARAM_WEIGHT_GRADIENTS;
    bool statistics = kind == PARAM_RUNNING_MEAN || kind == PARAM_RUNNING_VARIANCE;
    uint64_t expected = weights ? rows * spingalett_weight_row_len(net, index - 1) : rows;
    if (count != expected || (statistics && net->shapes[index].type != LAYER_BATCH_NORM)) {
        set_error(SPINGALETT_ERR_INVALID, who);
        return NULL;
    }
    *valid = true;
    uint64_t b = net->bias_offsets[index - 1];
    switch (kind) {
        case PARAM_WEIGHTS:          return net->weights + net->weight_offsets[index - 1];
        case PARAM_BIASES:           return net->biases + b;
        case PARAM_WEIGHT_GRADIENTS: return net->grad_weights ? net->grad_weights + net->weight_offsets[index - 1] : NULL;
        case PARAM_BIAS_GRADIENTS:   return net->grad_biases ? net->grad_biases + b : NULL;
        case PARAM_RUNNING_MEAN:     return net->running_mean + b;
        default:                     return net->running_var + b;
    }
}

bool spingalett_get_parameters(const NeuralNetwork *net, uint32_t index, ParameterKind kind, float *values, uint64_t count) {
    bool valid;
    spingalett_network_sync(net);
    const float *p = parameter_block(net, index, kind, count,
                                     "spingalett_get_parameters: invalid layer, kind or count", &valid);
    if (!valid || (!values && count > 0)) return false;
    if (count && p) memcpy(values, p, (size_t)count * sizeof(float));
    else if (count) memset(values, 0, (size_t)count * sizeof(float));     /* gradients before any training */
    return true;
}

bool spingalett_set_parameters(NeuralNetwork *net, uint32_t index, ParameterKind kind, const float *values, uint64_t count) {
    bool valid;
    float *p = parameter_block(net, index, kind, count, "spingalett_set_parameters: invalid layer, kind or count", &valid);
    if (valid && !p) {
        if (!spingalett_training_state(net)) return false;
        p = parameter_block(net, index, kind, count, "spingalett_set_parameters: invalid layer, kind or count", &valid);
    }
    if (!p || (!values && count > 0)) return false;
    spingalett_network_sync(net);
    if (count) memcpy(p, values, (size_t)count * sizeof(float));
    spingalett_network_written(net);
    return true;
}

bool spingalett_has_dropout(const NeuralNetwork *net) {
    for (uint32_t l = 1; l + 1 < net->layers; l++)
        if (net->dropout_rates[l] > 0.0f) return true;
    return false;
}

float *spingalett_forward_pass(NeuralNetwork *net, const float *input, ComputeMode mode,
                               const DropoutContext *dropout) {
    memcpy(SPINGALETT_LAYER_PTR(net, 0), input, net->topology[0] * sizeof(float));

    for (uint32_t l = 1; l < net->layers; l++) {
        uint32_t prev_size = net->topology[l - 1];
        uint32_t curr_size = net->topology[l];
        const float *W = SPINGALETT_WEIGHT_MTX_PTR(net, l - 1);
        const float *b = net->biases + net->bias_offsets[l - 1];
        const float *x = SPINGALETT_LAYER_PTR(net, l - 1);
        float *y = SPINGALETT_LAYER_PTR(net, l);

        memcpy(y, b, curr_size * sizeof(float));

#if defined(SPINGALETT_HAS_OPENBLAS)
        if (mode == COMPUTE_OPENBLAS) {
            cblas_sgemv(CblasRowMajor, CblasNoTrans,
                        (int)curr_size, (int)prev_size,
                        1.0f, W, (int)prev_size,
                        x, 1, 1.0f, y, 1);
        } else
#endif
        {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if(spingalett_use_omp(mode, (uint64_t)curr_size * prev_size))
#endif
            for (int64_t j = 0; j < (int64_t)curr_size; j++)
                y[j] += spingalett_dot_product(x, W + (uint64_t)j * prev_size, (uint64_t)prev_size);
        }

        apply_activation_batch(y, curr_size, net->act_func[l - 1]);

        if (dropout && l + 1 < net->layers && net->dropout_rates[l] > 0.0f)
            spingalett_dropout_apply(y, dropout->dmask + net->neuron_offsets[l], curr_size,
                                     net->act_func[l - 1], net->dropout_rates[l],
                                     dropout, l, dropout->position);
    }
    (void)mode;

    return SPINGALETT_LAYER_PTR(net, net->layers - 1);
}

float *forward_struct_arguments(ForwardArgs args) {
    NeuralNetwork *net = args.net;
    const float *input = args.input;

    if (!net || !input) {
        set_error(SPINGALETT_ERR_INVALID, "forward: net or input is NULL");
        return NULL;
    }
    spingalett_network_sync(net);

    if (net->layers < 2) {
        set_error(SPINGALETT_ERR_INVALID, "Network must have at least 2 layers for forward pass");
        return NULL;
    }

    ComputeMode mode = resolve_compute_mode();
    if (spingalett_all_dense(net))
        return spingalett_forward_pass(net, input, mode, NULL);

    /* one sample through the batch kernels; the output lands in the network's output slot */
    if (net->forward_ws && net->forward_mode != mode) {
        spingalett_batch_workspace_free(net->forward_ws);
        net->forward_ws = NULL;
    }
    if (!net->forward_ws) {
        if (!spingalett_check_graph(net, "forward"))
            return NULL;
        net->forward_ws = spingalett_batch_workspace_create(net, 1, false, false, mode);
        net->forward_mode = mode;
        if (!net->forward_ws) {
            set_error(SPINGALETT_ERR_ALLOC, "forward: workspace allocation failed");
            return NULL;
        }
    }
    BatchWorkspace *ws = net->forward_ws;
    ws->act[0] = (float *)input;                                        /* read only */
    ws->act[net->layers - 1] = SPINGALETT_LAYER_PTR(net, net->layers - 1);
    spingalett_batch_forward(net, ws, 1, NULL, 0, mode);
    return ws->act[net->layers - 1];
}

void print_parameters(const NeuralNetwork *net) {
    spingalett_network_sync(net);
    if (!net || net->layers < 2) {
        set_error(SPINGALETT_ERR_INVALID, "print_parameters: network must have at least 2 layers");
        return;
    }
    spingalett_log(LOG_INFO, "========== DEBUG NETWORK PARAMETERS ==========");
    spingalett_log(LOG_INFO, "Loss Function: %s", loss_func_names[net->loss_func]);

    for (uint32_t i = 0; i < net->layers - 1; i++) {
        uint32_t rows = spingalett_weight_rows(net, i), cols = spingalett_weight_row_len(net, i);
        uint32_t src = spingalett_source(net, i + 1);
        spingalett_log(LOG_INFO, "[Connection: Layer %u (%u neurons) -> Layer %u (%u neurons), activation function: %s, dropout: %g]",
            src, net->topology[src], i + 1, net->topology[i + 1],
            act_func_names[net->act_func[i]], (double)net->dropout_rates[i + 1]);
        spingalett_log(LOG_INFO, "  Biases of layer %u:", i + 1);
        for (uint32_t k = 0; k < rows; k++)
            spingalett_log(LOG_INFO, "    Row %u bias: %12.6g", k, (double)net->biases[net->bias_offsets[i] + k]);
        spingalett_log(LOG_INFO, "  Weights: %u rows x %u", rows, cols);
        for (uint32_t j = 0; j < rows; j++)
            for (uint32_t k = 0; k < cols; k++)
                spingalett_log(LOG_INFO, "    Row %u, weight %u: %12.6g", j, k,
                               (double)net->weights[net->weight_offsets[i] + (uint64_t)j * cols + k]);
    }

    spingalett_log(LOG_INFO, "================================================");
}

void free_network(NeuralNetwork *net) {
    if (!net) return;
    spingalett_batch_workspace_free(net->forward_ws);
    spingalett_gpu_net_free(atomic_exchange(&net->gpu_predict, NULL));
    if (net->gpu_kept) spingalett_gpu_net_free(net->gpu_trainer);
    spingalett_aligned_free(net->neurons);
    spingalett_aligned_free(net->weights);
    spingalett_aligned_free(net->biases);
    spingalett_aligned_free(net->grad_weights);
    spingalett_aligned_free(net->grad_biases);
    spingalett_aligned_free(net->opt_m_weights);
    spingalett_aligned_free(net->opt_m_biases);
    spingalett_aligned_free(net->opt_v_weights);
    spingalett_aligned_free(net->opt_v_biases);
    spingalett_aligned_free(net->running_mean);
    spingalett_aligned_free(net->running_var);
    free(net->neuron_offsets);
    free(net->weight_offsets);
    free(net->bias_offsets);
    free(net->topology);
    free(net->shapes);
    free(net->act_func);
    free(net->dropout_rates);
    free(net->input_offsets);
    free(net->input_list);
    free(net);
    spingalett_log(LOG_DEBUG, "Memory freed.");
}
