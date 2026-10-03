/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

#pragma once

#include "Spingalett/Spingalett.h"
#include <stdint.h>

#if defined(SPINGALETT_HAS_OPENBLAS)
#include <cblas.h>
#endif

#define SPINGALETT_ERRMSG_MAX 256
#define SPINGALETT_ALIGNMENT  64

/* Multiply-adds below which forking an OpenMP team costs more than it saves. */
#define SPINGALETT_OMP_MIN_WORK 32768u

static inline bool spingalett_use_omp(ComputeMode mode, uint64_t work) {
    return mode == COMPUTE_OPENMP && work >= SPINGALETT_OMP_MIN_WORK;
}

#define SPINGALETT_NEURON(net, l, j)        ((net)->neurons[(net)->neuron_offsets[l] + (uint64_t)(j)])
#define SPINGALETT_LAYER_PTR(net, l)        ((net)->neurons + (net)->neuron_offsets[l])

#define SPINGALETT_WEIGHT(net, l, j, k)     ((net)->weights[(net)->weight_offsets[l] + (uint64_t)(j) * (uint64_t)(net)->topology[l] + (uint64_t)(k)])
#define SPINGALETT_BIAS(net, l, j)          ((net)->biases[(net)->bias_offsets[l] + (uint64_t)(j)])

#define SPINGALETT_WEIGHT_MTX_PTR(net, l)   ((net)->weights + (net)->weight_offsets[l])
#define SPINGALETT_GRAD_W_MTX_PTR(net, l)   ((net)->grad_weights + (net)->weight_offsets[l])

void set_error(int code, const char *msg);

bool spingalett_add_layer(LayerArgs args);
float *spingalett_forward_pass(NeuralNetwork *net, const float *input, ComputeMode mode);

void spingalett_log(LogLevel level, const char *fmt, ...);

void *spingalett_aligned_alloc(size_t size);
void *spingalett_aligned_calloc(size_t count, size_t elem_size);
void  spingalett_aligned_free(void *ptr);

void apply_softmax(float *layer, uint32_t size);
void apply_activation_batch(float *data, uint32_t size, ActivationFunction act);
void apply_activation_bulk(float *data, uint64_t total, ActivationFunction act);
void apply_derivative_batch(float *deriv, const float *act_data, uint64_t total, ActivationFunction act);

void     rng_seed(uint64_t seed);
uint32_t rng_next(void);
float    rng_next_float(void);

float random_uniform_weight(void);
float random_normal_weight(void);

float spingalett_dot_product(const float *restrict a, const float *restrict b, uint64_t n);

void spingalett_shuffle_indices(uint32_t *indices, uint32_t n);

void spingalett_sgd_update(float *restrict W, const float *restrict gW, uint64_t n, float lr, float decay);
void spingalett_momentum_update(float *restrict W, float *restrict mW, const float *restrict gW, uint64_t n,
                                float lr, float momentum, float decay);
void spingalett_rmsprop_update(float *restrict W, float *restrict vW, const float *restrict gW, uint64_t n,
                               float lr, float beta2, float epsilon, float decay);
void spingalett_adam_update(float *restrict W, float *restrict mW, float *restrict vW, const float *restrict gW,
                            uint64_t n, float lr, float beta1, float beta2,
                            float m_factor, float v_factor, float epsilon,
                            float decay, float wd_factor);

void spingalett_vec_scale(float *data, uint64_t n, float scale);
void spingalett_vec_scaled_copy(float *restrict dst, const float *restrict src, uint64_t n, float alpha);
void spingalett_vec_axpy(float *restrict y, const float *restrict x, uint64_t n, float alpha);

float spingalett_clip_grad_norm(NeuralNetwork *net, float max_norm);

ComputeMode resolve_compute_mode(void);

void spingalett_fp_flush_denormals_begin(void);
void spingalett_fp_flush_denormals_end(void);

float compute_sample_loss(const float *output, const float *target,
                          uint32_t output_size, LossFunction loss_func,
                          ActivationFunction output_act);

extern const char * const act_func_names[];
extern const char * const loss_func_names[];
extern const char * const training_strategy_names[];
extern const char * const training_mode_names[];
extern const char * const optimizer_names[];
extern const char * const weight_initialization_names[];
extern const char * const precision_names[];
extern const char * const compute_mode_names[];
