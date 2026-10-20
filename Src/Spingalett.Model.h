/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/* Deployment models' internals shared by batched prediction (Spingalett.Model.c) and sessions
   (Spingalett.Session.c): the weights prepared once per owned model, the workspace of a pass, and the
   passes of single layers over a batch. */

#pragma once

#include "Spingalett.Private.h"
#include "Spingalett.Thread.h"
#include <stdatomic.h>

/* A layer's weights in the forms batched prediction runs on, built from the image. */
typedef struct {
    float *dequant;                             /* FP16 / BF16 weights as float */
    int8_t *unpacked;                           /* integer convolution filters transposed (short windows,
                                                   depthwise convolutions), or INT4 / INT2 rows as bytes */
    int32_t *sums;                              /* sums of the integer rows, for the 4 x 4 kernels */
    int8_t *interleaved;                        /* rows interleaved for tiles of samples or pixels
                                                   (spingalett_i8_interleaved_tile) */
    int32_t *tile_sums;                         /* the sums of the interleaved rows */
    float *filters;                             /* a float transposed convolution's filters as those of the
                                                   convolution whose data gradient it is */
    float *panels;                              /* sessions: a float linear layer's filters packed for
                                                   spingalett_conv_rows() */
    bool ready;                                 /* all but a dense layer's interleaved rows are built */
} PreparedLayer;

#define MODEL_CHUNK 1024u                        /* samples per pass through the layers, at most */

typedef struct PredictWorkspace {
    float *act[2];                              /* [chunk x max_width] each; for a graph (format 6) act[0]
                                                   holds every output at chunk times its offset */
    int8_t *xq;                                 /* [chunk x max_int_inputs] */
    float *xs;                                  /* activation scales: a sample's, or a cell's (SlettLayer.cells) */
    int8_t *window;                             /* a quantized convolution window or one pixel's sums, per thread */
    size_t window_stride;
    int8_t *tile;                               /* a tile of activations and its sums, per thread */
    size_t tile_stride, tile_sums;              /* bytes per thread, offset of the sums */
    float *conv;                                /* gathered windows of float convolutions (OpenBLAS) */
    float *norm;                                /* batch normalization coefficients */
    uint8_t *blockq;                            /* the quantized inputs of a layer of a block precision
                                                   (spingalett_block_layer()) */
    SpingalettGemmScratch *gemm;
    /* what the workspace was made for, and its size */
    int threads;
    uint32_t chunk;
    ComputeMode mode;
    bool dense_tiles, sessions;
    size_t bytes;
} PredictWorkspace;

/* What a model made by the library owns: its image, its weights prepared for batched prediction and
   sessions (built by the first call that needs them, under the lock, and only read afterwards), and the
   workspace of an earlier call, which the next call takes if it fits. Models filled in by
   spingalett_model_init own nothing, and prepare their weights on every call (every session). */
typedef struct {
    void *image;
    SpgSignal *lock;
    PreparedLayer *layers;
    _Atomic(PredictWorkspace *) spare;
#if !defined(SPINGALETT_RUNTIME)
    /* the full library on a GPU: the model as a network there (models of float weights), made under the lock by
       the first call that needs it (gpu_state 1; -1: the GPU cannot run it); a model of integer weights as the CPU
       computes it, on each backend (Vulkan, CUDA) the first call there makes it (gpu_exact_state); their calls one at
       a time (gpu_lock) */
    struct SpingalettNetwork *gpu_net;
    struct SpgGpuModel *gpu_exact[2];
    SpgSignal *gpu_lock;
    _Atomic int gpu_state;
    _Atomic int gpu_exact_state[2];
#endif
} ModelOwner;

/* The model's prepared weights: those of an owned model (prepared under its lock), or new ones that
   spingalett_model_release_layers() frees. dense_tiles: for calls with a tile of samples or more;
   sessions: with the panels of the float linear layers. NULL when memory runs out (the error unset). */
PreparedLayer *spingalett_model_layers(const SpingalettModel *model, bool dense_tiles, bool sessions);
void spingalett_model_release_layers(const SpingalettModel *model, PreparedLayer *layers);

/* A workspace for passes of `chunk` samples (sessions: of chunk times the model's cells rows, without the
   scratch of float convolutions and products, which sessions run on their panels); NULL when memory runs
   out. */
PredictWorkspace *spingalett_predict_workspace_create(const SpingalettModel *model, uint32_t chunk, ComputeMode mode,
                                                      bool dense_tiles, bool sessions);
void spingalett_predict_workspace_free(PredictWorkspace *w);

/* One layer over n samples: x [n x inputs] -> y [n x outputs], activation included (input: the network's inputs,
   n rows of input_size, whose tokens attention ignoring padding reads; NULL for other layers; kv: the second
   input of cross-attention, else NULL). A dense layer's interleaved rows are read only in calls with
   dense_tiles. */
void spingalett_predict_layer(const uint8_t *image, const SlettLayer *L, const PreparedLayer *prep, bool dense_tiles,
                              const float *input, uint32_t input_size, const float *x, const float *kv, float *y,
                              uint32_t n, PredictWorkspace *w, ComputeMode mode);
/* An addition, product, concatenation or global pooling over n samples (x[k]: input k, units[k] per
   sample, channels[k] a cell). */
void spingalett_predict_combine(const SlettLayer *L, const float *const *x, const uint32_t *units,
                                const uint32_t *channels, float *y, uint32_t n, ComputeMode mode);

/* Whether a model's first layer is an embedding (its cells are tokens: a language model). */
static inline bool spingalett_slett_language_model(const uint8_t *image) {
    if (slett_get16(image + 6) < 8u) return false;
    SlettLayer first;
    spingalett_slett_layer(image, 0, &first);
    return first.type == LAYER_EMBEDDING;
}

/* A float linear layer of a language model, which sessions run on packed panels. */
static inline bool spingalett_session_panels(const SlettLayer *L, bool language_model) {
    return language_model && L->groups == 1 && L->precision <= PRECISION_BFLOAT16 && slett_conv_pointwise(L);
}
