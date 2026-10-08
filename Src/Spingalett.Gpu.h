/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/*
 * Networks on the GPU (Src/Gpu/Spingalett.Gpu.c): a copy of a network's parameters in device memory
 * with the buffers of chunks of up to `capacity` samples, training steps and forward passes recorded
 * as command buffers (once per chunk size) and run without the host in between. The host fills the
 * chunk's inputs and targets in memory the device reads, and reads losses and outputs back; the
 * parameters come back with spingalett_gpu_download().
 *
 * Chunks are submitted in order and run one after the other on the device; the host may fill one
 * while the device runs the previous one. Results are deterministic: every sum is taken in a fixed
 * order, never by atomics.
 *
 * Without the Vulkan backend (SPINGALETT_HAS_VULKAN undefined), every function reports that the GPU
 * is unavailable.
 */

#pragma once

#include "Spingalett.Private.h"

typedef struct SpgGpuNet SpgGpuNet;

/* Fixed for a training run. */
typedef struct {
    OptimizerType optimizer;
    float decay, momentum, beta1, beta2, epsilon;
    float max_grad_norm;
    uint64_t dropout_seed;
} SpgGpuTraining;

/* Per optimizer step. */
typedef struct {
    float lr, m_factor, v_factor;
    uint64_t step;                  /* the step of the samples (dropout masks) */
} SpgGpuStep;

#if defined(SPINGALETT_HAS_VULKAN)

/* Whether a device is usable (opens it on first use), and its name. */
bool spingalett_gpu_available(void);
const char *spingalett_gpu_name(void);
/* Whether the GPU runs every layer of the network (names the first it does not in `why`). */
bool spingalett_gpu_supports(const NeuralNetwork *net, const char **why);
/* The chunk size for up to `want` samples that fits the device's memory (0: not even one). */
uint32_t spingalett_gpu_capacity(const NeuralNetwork *net, uint32_t want, bool training);

/* The network's parameters on the device, for chunks of up to `capacity` samples; with `training`,
   the gradients, optimizer state and outputs of every layer too. NULL when memory runs out. */
SpgGpuNet *spingalett_gpu_net_create(NeuralNetwork *net, uint32_t capacity, const SpgGpuTraining *training);
void spingalett_gpu_net_free(SpgGpuNet *g);
uint32_t spingalett_gpu_net_capacity(const SpgGpuNet *g);
/* Whether it was made for the current GPU precision (spingalett_set_gpu_precision()). */
bool spingalett_gpu_net_current(const SpgGpuNet *g);

/* Copies the network's parameters (weights, biases, running statistics and optimizer moments) to
   the device, or back from it after the work submitted so far. */
bool spingalett_gpu_upload(SpgGpuNet *g);
bool spingalett_gpu_download(SpgGpuNet *g);

/* Where the next chunk's inputs and targets go (rows of the input and output layers), once the
   device is done with the chunk that used that memory before. */
float *spingalett_gpu_chunk_inputs(SpgGpuNet *g, float **targets);
/* Trains on the n samples filled in: forward, loss and backward passes, their gradients scaled by
   1 / count and added to the step's (first: the step's first chunk, which replaces them), then with
   `last` the optimizer step. position: the chunk's first sample within its step (dropout). */
bool spingalett_gpu_train_chunk(SpgGpuNet *g, uint32_t n, uint32_t count, uint32_t position, bool first, bool last,
                                const SpgGpuStep *step);
/* Waits for the chunks submitted so far; *loss = the sum of their steps' losses since the last call,
   added up as the CPU does (samples of a chunk in order, then chunks, then steps). */
bool spingalett_gpu_take_loss(SpgGpuNet *g, float *loss);

/* outputs = the network's outputs for n samples (inference: batch normalization with the running
   statistics, no dropout), in chunks of up to `capacity` that overlap with the copies. */
bool spingalett_gpu_predict(SpgGpuNet *g, const float *inputs, float *outputs, uint32_t n);

/* The step API: one pass at a time, each waited for. The inputs of the next forward pass and the
   targets (or dL/d(outputs) of a loss of the caller's) of the next backward pass go where these point.
   The forward pass trains (batch statistics, dropout of the step and positions from `position`) and
   returns its outputs (valid until the next pass); the backward pass back-propagates it and adds the
   samples' gradients to those since the last step (add) or starts them, *loss = the sum of the
   samples' losses (from targets); the step applies the gradients times grad_scale with the
   optimizer of cfg. */
float *spingalett_gpu_pass_buffers(SpgGpuNet *g, float **targets);
const float *spingalett_gpu_pass_forward(SpgGpuNet *g, uint32_t n, uint32_t position, uint64_t step);
bool spingalett_gpu_pass_backward(SpgGpuNet *g, uint32_t n, bool from_grads, bool add, float *loss);
bool spingalett_gpu_pass_step(SpgGpuNet *g, const SpgGpuTraining *cfg, const SpgGpuStep *step, float grad_scale);

#else

static inline bool spingalett_gpu_available(void) { return false; }
static inline const char *spingalett_gpu_name(void) { return NULL; }
static inline bool spingalett_gpu_supports(const NeuralNetwork *net, const char **why) {
    (void)net;
    if (why) *why = "this build has no GPU backend";
    return false;
}
static inline uint32_t spingalett_gpu_capacity(const NeuralNetwork *net, uint32_t want, bool training) {
    (void)net; (void)want; (void)training;
    return 0;
}
static inline SpgGpuNet *spingalett_gpu_net_create(NeuralNetwork *net, uint32_t capacity, const SpgGpuTraining *t) {
    (void)net; (void)capacity; (void)t;
    return NULL;
}
static inline void spingalett_gpu_net_free(SpgGpuNet *g) { (void)g; }
static inline uint32_t spingalett_gpu_net_capacity(const SpgGpuNet *g) { (void)g; return 0; }
static inline bool spingalett_gpu_net_current(const SpgGpuNet *g) { (void)g; return false; }
static inline bool spingalett_gpu_upload(SpgGpuNet *g) { (void)g; return false; }
static inline bool spingalett_gpu_download(SpgGpuNet *g) { (void)g; return false; }
static inline float *spingalett_gpu_chunk_inputs(SpgGpuNet *g, float **targets) {
    (void)g; (void)targets;
    return NULL;
}
static inline bool spingalett_gpu_train_chunk(SpgGpuNet *g, uint32_t n, uint32_t count, uint32_t position, bool first,
                                              bool last, const SpgGpuStep *step) {
    (void)g; (void)n; (void)count; (void)position; (void)first; (void)last; (void)step;
    return false;
}
static inline bool spingalett_gpu_take_loss(SpgGpuNet *g, float *loss) { (void)g; (void)loss; return false; }
static inline bool spingalett_gpu_predict(SpgGpuNet *g, const float *inputs, float *outputs, uint32_t n) {
    (void)g; (void)inputs; (void)outputs; (void)n;
    return false;
}
static inline float *spingalett_gpu_pass_buffers(SpgGpuNet *g, float **targets) {
    (void)g; (void)targets;
    return NULL;
}
static inline const float *spingalett_gpu_pass_forward(SpgGpuNet *g, uint32_t n, uint32_t position, uint64_t step) {
    (void)g; (void)n; (void)position; (void)step;
    return NULL;
}
static inline bool spingalett_gpu_pass_backward(SpgGpuNet *g, uint32_t n, bool from_grads, bool add, float *loss) {
    (void)g; (void)n; (void)from_grads; (void)add; (void)loss;
    return false;
}
static inline bool spingalett_gpu_pass_step(SpgGpuNet *g, const SpgGpuTraining *cfg, const SpgGpuStep *step,
                                            float grad_scale) {
    (void)g; (void)cfg; (void)step; (void)grad_scale;
    return false;
}

#endif
