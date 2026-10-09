/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/* The network's representation, private to the library (and its white-box tests). */

#pragma once

#include "Spingalett/Spingalett.h"
#include <stdatomic.h>

/* Layer l's output shape and, for conv and pooling layers, its window over its input (transposed
   convolutions: the window each input cell adds; upsampling: the factors in stride_h, stride_w). */
typedef struct {
    LayerType type;
    uint32_t height, width, channels;
    uint32_t kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w;
    uint32_t groups;                /* conv and transposed conv: channel groups (1 otherwise) */
    float eps, momentum;            /* batch normalization (0 otherwise); layer normalization: eps */
    uint32_t mode;                  /* upsampling: UpsampleMode */
} LayerShape;

/*
 * All parameters live in flat arrays shared by every layer kind: weight layer l (the parameters
 * feeding layer l + 1) owns weights [weight_offsets[l], +weight_count) and biases
 * [bias_offsets[l], +bias_count), with the gradients and optimizer moments laid out alike.
 *
 * Layer l >= 1 reads the layers input_list[input_offsets[l] .. input_offsets[l + 1]), all of them
 * below l, so that the index order is a topological order of the graph; the first of them is its
 * source, whose output shape is the layer's input shape. In a chain every layer reads the one
 * before it.
 */
struct SpingalettNetwork {
    uint32_t layers;
    uint32_t *input_offsets;        /* [layers + 1] */
    uint32_t *input_list;           /* [input_offsets[layers]] */
    bool graph;                     /* some layer reads other layers than the one before it, or is
                                       an add, concatenation or global pooling */
    uint32_t *topology;             /* outputs of each layer: height * width * channels */
    ActivationFunction *act_func;   /* [layers - 1]: the activation of layer l + 1 */
    LayerShape *shapes;             /* [layers] */

    float *weights;
    float *biases;
    float *neurons;                 /* one sample's activations, laid out by neuron_offsets */

    float *grad_weights;
    float *grad_biases;

    float *opt_m_weights;
    float *opt_m_biases;
    float *opt_v_weights;
    float *opt_v_biases;

    uint64_t *neuron_offsets;
    uint64_t *weight_offsets;
    uint64_t *bias_offsets;

    uint64_t total_neurons;
    uint64_t total_weights;
    uint64_t total_biases;
    /* floats the neuron, weight-sized and bias-sized arrays hold (at least the totals): a loader
       reserves the final sizes, so that adding layers does not move the arrays */
    uint64_t cap_neurons, cap_weights, cap_biases;

    uint64_t time_step;
    LossFunction loss_func;

    float *dropout_rates;           /* per layer, applied to its outputs while training */

    /* batch normalization layers: the statistics inference uses, laid out like the biases (zeros
       elsewhere) */
    float *running_mean;
    float *running_var;

    /* forward() of networks with convolution or pooling layers runs the batch kernels on one
       sample, in this workspace (made on first use, for the compute mode it was made for) */
    struct BatchWorkspace *forward_ws;
    ComputeMode forward_mode;

    /* predict() and evaluate() with COMPUTE_VULKAN: the network on the GPU, kept between calls (made
       for the layers as they were, its parameters copied to it when they changed); a call takes it
       and puts it back, so that concurrent calls make their own */
    _Atomic(struct SpgGpuNet *) gpu_predict;

    /* the network's copy on the GPU: a trainer's of the step API, or (gpu_kept) the one train() left,
       which the network owns and the next train() takes again; newer than these arrays while
       gpu_newer (spingalett_network_sync() brings it back). param_version counts writes to the arrays
       on the host, after which a copy takes them again; gpu_version is the one the kept copy has;
       host_version counts every change of the arrays, the copies back from the GPU too (gpu_predict
       holds the parameters of one, spingalett_gpu_net_version()) */
    struct SpgGpuNet *gpu_trainer;
    atomic_bool gpu_newer;
    bool gpu_kept;
    atomic_bool gpu_busy;           /* the kept copy in use (a predict(), a copy back, its release) */
    uint64_t param_version, gpu_version;
    _Atomic(uint64_t) host_version;
};

/* The layers layer l (>= 1) reads, and the first of them (its source). */
static inline uint32_t spingalett_input_count(const NeuralNetwork *net, uint32_t l) {
    return net->input_offsets[l + 1] - net->input_offsets[l];
}

static inline const uint32_t *spingalett_inputs(const NeuralNetwork *net, uint32_t l) {
    return net->input_list + net->input_offsets[l];
}

static inline uint32_t spingalett_source(const NeuralNetwork *net, uint32_t l) {
    return net->input_list[net->input_offsets[l]];
}

/* Weight layer l as a matrix: rows (dense outputs, conv and transposed conv filters, normalization
   channels) of row_len weights (dense inputs, kernel_h x kernel_w x input channels of the filter's
   group, the normalization's gamma), one bias per row; pooling, adding, concatenating and
   upsampling layers have none. A transposed convolution's filter j holds, at (kh, kw, c), the weight
   by which input channel c of its group reaches output channel j at offset (kh, kw) of the input
   cell's window. */
static inline bool spingalett_filters(LayerType type) { return type == LAYER_CONV2D || type == LAYER_CONV_TRANSPOSE2D; }
static inline bool spingalett_normalization(LayerType type) {
    return type == LAYER_BATCH_NORM || type == LAYER_LAYER_NORM;
}

static inline uint32_t spingalett_weight_rows(const NeuralNetwork *net, uint32_t l) {
    const LayerShape *s = &net->shapes[l + 1];
    return s->type == LAYER_DENSE ? net->topology[l + 1]
         : spingalett_filters(s->type) || spingalett_normalization(s->type) ? s->channels : 0u;
}

static inline uint32_t spingalett_weight_row_len(const NeuralNetwork *net, uint32_t l) {
    const LayerShape *s = &net->shapes[l + 1];
    const uint32_t src = spingalett_source(net, l + 1);
    return s->type == LAYER_DENSE ? net->topology[src]
         : spingalett_filters(s->type) ? s->kernel_h * s->kernel_w * (net->shapes[src].channels / s->groups)
         : spingalett_normalization(s->type) ? 1u : 0u;
}

/* Whether every weight layer is dense and reads the one before it (the network of earlier versions). */
static inline bool spingalett_all_dense(const NeuralNetwork *net) {
    if (net->graph) return false;
    for (uint32_t l = 1; l < net->layers; l++)
        if (net->shapes[l].type != LAYER_DENSE) return false;
    return true;
}
