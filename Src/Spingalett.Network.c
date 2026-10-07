/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

#include "Spingalett.Private.h"
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

/* The shape of a new layer from its arguments and the previous layer; false (error set) when they
   do not fit together. */
static bool layer_shape(const NeuralNetwork *net, const LayerArgs *args, LayerShape *shape, uint32_t *outputs) {
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
    } else if (args->type == LAYER_BATCH_NORM) {
        /* the previous layer's shape, normalized per channel */
        const LayerShape *in = &net->shapes[net->layers - 1];
        if (!(args->epsilon >= 0.0f && args->epsilon < 1.0f) || !(args->momentum >= 0.0f && args->momentum <= 1.0f))
            return layer_error("Batch normalization needs epsilon in [0, 1) and momentum in [0, 1]");
        shape->height = in->height;
        shape->width = in->width;
        shape->channels = in->channels;
        shape->eps = args->epsilon > 0.0f ? args->epsilon : 1e-5f;
        shape->momentum = args->momentum > 0.0f ? args->momentum : 0.1f;
        units = net->topology[net->layers - 1];
    } else {
        const LayerShape *in = &net->shapes[net->layers - 1];
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
        if (groups > 1) return layer_error("Grouped convolutions are not supported yet");
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

    LayerShape shape;
    uint32_t neurons_amount;
    if (!layer_shape(net, &args, &shape, &neurons_amount))
        return false;
    spingalett_batch_workspace_free(net->forward_ws);     /* made for the old layers */
    net->forward_ws = NULL;
    bool pooling = shape.type == LAYER_MAX_POOL2D || shape.type == LAYER_AVG_POOL2D;
    if (pooling) act_func = ACT_NONE;
    static const char *const type_names[] = {"dense", "conv2d", "max_pool2d", "avg_pool2d", "batch_norm"};

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
        spingalett_log(LOG_INFO, "Layer #%u: %s, output %u x %u x %u, activation %s, dropout %g", nl - 1,
                       type_names[shape.type], shape.height, shape.width, shape.channels,
                       act_func_names[act_func], (double)dropout_rate);
    }

    /* the new layer's parameters, as compute_offsets() will count them */
    uint32_t prev_neurons = (net->layers > 0) ? net->topology[net->layers - 1] : 0;
    uint32_t rows = 0, row_len = 0;
    if (nl > 1 && shape.type == LAYER_DENSE) { rows = neurons_amount; row_len = prev_neurons; }
    if (nl > 1 && shape.type == LAYER_CONV2D) {
        rows = shape.channels;
        row_len = shape.kernel_h * shape.kernel_w * (net->shapes[net->layers - 1].channels / shape.groups);
    }
    if (nl > 1 && shape.type == LAYER_BATCH_NORM) { rows = shape.channels; row_len = 1; }
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
    float              *t_neurons = (float *)              spingalett_aligned_calloc(new_tn, sizeof(float));
    float              *t_drop    = (float *)              malloc(nl * sizeof(float));

    ActivationFunction *t_act     = NULL;
    float *t_w = NULL, *t_b = NULL;
    float *t_gw = NULL, *t_gb = NULL;
    float *t_mw = NULL, *t_mb = NULL;
    float *t_vw = NULL, *t_vb = NULL;
    float *t_rm = NULL, *t_rv = NULL;

    /* sizes of at least 1: pooling layers add no parameters, and a network of pooling layers has none */
    size_t nw = new_tw ? new_tw : 1, nb = new_tb ? new_tb : 1;
    if (nl > 1) {
        t_act = (ActivationFunction *)malloc((nl - 1) * sizeof(ActivationFunction));
        t_w   = (float *)spingalett_aligned_calloc(nw, sizeof(float));
        t_b   = (float *)spingalett_aligned_calloc(nb, sizeof(float));
        t_gw  = (float *)spingalett_aligned_calloc(nw, sizeof(float));
        t_gb  = (float *)spingalett_aligned_calloc(nb, sizeof(float));
        t_mw  = (float *)spingalett_aligned_calloc(nw, sizeof(float));
        t_mb  = (float *)spingalett_aligned_calloc(nb, sizeof(float));
        t_vw  = (float *)spingalett_aligned_calloc(nw, sizeof(float));
        t_vb  = (float *)spingalett_aligned_calloc(nb, sizeof(float));
        t_rm  = (float *)spingalett_aligned_calloc(nb, sizeof(float));
        t_rv  = (float *)spingalett_aligned_calloc(nb, sizeof(float));
    }

    bool ok = t_topo && t_shapes && t_noff && t_woff && t_boff && t_neurons && t_drop;
    if (nl > 1)
        ok = ok && t_act && t_w && t_b && t_gw && t_gb && t_mw && t_mb && t_vw && t_vb && t_rm && t_rv;

    if (!ok) {
        free(t_topo); free(t_shapes); free(t_noff); free(t_woff); free(t_boff); free(t_drop);
        spingalett_aligned_free(t_neurons);
        if (nl > 1) {
            free(t_act);
            spingalett_aligned_free(t_w);  spingalett_aligned_free(t_b);
            spingalett_aligned_free(t_gw); spingalett_aligned_free(t_gb);
            spingalett_aligned_free(t_mw); spingalett_aligned_free(t_mb);
            spingalett_aligned_free(t_vw); spingalett_aligned_free(t_vb);
            spingalett_aligned_free(t_rm); spingalett_aligned_free(t_rv);
        }
        set_error(SPINGALETT_ERR_ALLOC, "Layer allocation failed");
        return false;
    }

    if (net->layers > 0) {
        memcpy(t_topo, net->topology, net->layers * sizeof(uint32_t));
        memcpy(t_shapes, net->shapes, net->layers * sizeof(LayerShape));
    }
    t_topo[net->layers] = neurons_amount;
    t_shapes[net->layers] = shape;

    if (net->layers > 0)
        memcpy(t_drop, net->dropout_rates, net->layers * sizeof(float));
    t_drop[net->layers] = dropout_rate;

    if (net->total_neurons > 0)
        memcpy(t_neurons, net->neurons, net->total_neurons * sizeof(float));

    if (nl > 1) {
        if (net->layers > 1)
            memcpy(t_act, net->act_func, (net->layers - 1) * sizeof(ActivationFunction));
        t_act[net->layers - 1] = act_func;

        if (net->total_weights > 0) {
            memcpy(t_w,  net->weights,       net->total_weights * sizeof(float));
            memcpy(t_gw, net->grad_weights,  net->total_weights * sizeof(float));
            memcpy(t_mw, net->opt_m_weights, net->total_weights * sizeof(float));
            memcpy(t_vw, net->opt_v_weights, net->total_weights * sizeof(float));
        }
        if (net->total_biases > 0) {
            memcpy(t_b,  net->biases,        net->total_biases * sizeof(float));
            memcpy(t_gb, net->grad_biases,   net->total_biases * sizeof(float));
            memcpy(t_mb, net->opt_m_biases,  net->total_biases * sizeof(float));
            memcpy(t_vb, net->opt_v_biases,  net->total_biases * sizeof(float));
            memcpy(t_rm, net->running_mean,  net->total_biases * sizeof(float));
            memcpy(t_rv, net->running_var,   net->total_biases * sizeof(float));
        }

        /* Standard deviations: Glorot sqrt(2 / (fan_in + fan_out)), He sqrt(2 / fan_in),
           LeCun sqrt(1 / fan_in); a filter's fan-in is its window, its fan-out the window times
           the filters. */
        float fan_in = (float)row_len;
        float fan_out = shape.type == LAYER_CONV2D ? (float)shape.kernel_h * (float)shape.kernel_w * (float)rows : (float)rows;
        float scale = 1.0f;
        if (wi == WEIGHT_INITIALIZATION_XAVIER)
            scale = sqrtf(2.0f / (fan_in + fan_out));
        else if (wi == WEIGHT_INITIALIZATION_HE)
            scale = sqrtf(2.0f / fan_in);
        else if (wi == WEIGHT_INITIALIZATION_LECUN)
            scale = sqrtf(1.0f / fan_in);

        for (uint64_t idx = 0; idx < add_w; idx++) {
            uint64_t pos = net->total_weights + idx;
            if (shape.type == LAYER_BATCH_NORM)
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

    net->layers         = nl;
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

void layer_struct_arguments(LayerArgs args) {
    (void)spingalett_add_layer(args);
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
    switch (s->type) {
        case LAYER_DENSE:
            a.neurons_amount = net->topology[l];
            break;
        case LAYER_BATCH_NORM:
            a.epsilon = s->eps;
            a.momentum = s->momentum;
            break;
        default:
            a.filters = s->type == LAYER_CONV2D ? s->channels : 0u;
            a.groups = s->type == LAYER_CONV2D ? s->groups : 0u;
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
    layer->groups = s->type == LAYER_CONV2D ? s->groups : 0u;
    layer->epsilon = s->eps;
    layer->momentum = s->momentum;
    if (index > 0) {
        layer->bias_count = spingalett_weight_rows(net, index - 1);
        layer->weight_count = layer->bias_count * spingalett_weight_row_len(net, index - 1);
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

/* The array of `kind` feeding layer index and its length; NULL (error set) when out of range. */
static float *parameter_block(const NeuralNetwork *net, uint32_t index, ParameterKind kind, uint64_t count,
                              const char *who) {
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
    uint64_t b = net->bias_offsets[index - 1];
    switch (kind) {
        case PARAM_WEIGHTS:          return net->weights + net->weight_offsets[index - 1];
        case PARAM_BIASES:           return net->biases + b;
        case PARAM_WEIGHT_GRADIENTS: return net->grad_weights + net->weight_offsets[index - 1];
        case PARAM_BIAS_GRADIENTS:   return net->grad_biases + b;
        case PARAM_RUNNING_MEAN:     return net->running_mean + b;
        default:                     return net->running_var + b;
    }
}

bool spingalett_get_parameters(const NeuralNetwork *net, uint32_t index, ParameterKind kind, float *values, uint64_t count) {
    const float *p = parameter_block(net, index, kind, count,
                                     "spingalett_get_parameters: invalid layer, kind or count");
    if (!p || (!values && count > 0)) return false;
    if (count) memcpy(values, p, (size_t)count * sizeof(float));
    return true;
}

bool spingalett_set_parameters(NeuralNetwork *net, uint32_t index, ParameterKind kind, const float *values, uint64_t count) {
    float *p = parameter_block(net, index, kind, count, "spingalett_set_parameters: invalid layer, kind or count");
    if (!p || (!values && count > 0)) return false;
    if (count) memcpy(p, values, (size_t)count * sizeof(float));
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
    if (!net || net->layers < 2) {
        set_error(SPINGALETT_ERR_INVALID, "print_parameters: network must have at least 2 layers");
        return;
    }
    spingalett_log(LOG_INFO, "========== DEBUG NETWORK PARAMETERS ==========");
    spingalett_log(LOG_INFO, "Loss Function: %s", loss_func_names[net->loss_func]);

    for (uint32_t i = 0; i < net->layers - 1; i++) {
        uint32_t rows = spingalett_weight_rows(net, i), cols = spingalett_weight_row_len(net, i);
        spingalett_log(LOG_INFO, "[Connection: Layer %u (%u neurons) -> Layer %u (%u neurons), activation function: %s, dropout: %g]",
            i, net->topology[i], i + 1, net->topology[i + 1],
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
    free(net);
    spingalett_log(LOG_DEBUG, "Memory freed.");
}
