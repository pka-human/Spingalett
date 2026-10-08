/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/* The network's representation, private to the library (and its white-box tests). */

#pragma once

#include "Spingalett/Spingalett.h"

/* Layer l's output shape and, for conv and pooling layers, its window over its input. */
typedef struct {
    LayerType type;
    uint32_t height, width, channels;
    uint32_t kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w;
    uint32_t groups;                /* conv: channel groups (1 otherwise) */
    float eps, momentum;            /* batch normalization (0 otherwise) */
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
struct NeuralNetwork {
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

/* Weight layer l as a matrix: rows (dense outputs, conv filters, batch normalization channels) of
   row_len weights (dense inputs, conv kernel_h x kernel_w x input channels of the filter's group,
   batch normalization's gamma), one bias per row; pooling, adding and concatenating layers have
   none. */
static inline uint32_t spingalett_weight_rows(const NeuralNetwork *net, uint32_t l) {
    const LayerShape *s = &net->shapes[l + 1];
    return s->type == LAYER_DENSE ? net->topology[l + 1]
         : s->type == LAYER_CONV2D || s->type == LAYER_BATCH_NORM ? s->channels : 0u;
}

static inline uint32_t spingalett_weight_row_len(const NeuralNetwork *net, uint32_t l) {
    const LayerShape *s = &net->shapes[l + 1];
    const uint32_t src = spingalett_source(net, l + 1);
    return s->type == LAYER_DENSE ? net->topology[src]
         : s->type == LAYER_CONV2D ? s->kernel_h * s->kernel_w * (net->shapes[src].channels / s->groups)
         : s->type == LAYER_BATCH_NORM ? 1u : 0u;
}

/* Whether every weight layer is dense and reads the one before it (the network of earlier versions). */
static inline bool spingalett_all_dense(const NeuralNetwork *net) {
    if (net->graph) return false;
    for (uint32_t l = 1; l < net->layers; l++)
        if (net->shapes[l].type != LAYER_DENSE) return false;
    return true;
}
