/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/* .slett images written layer by layer, without a network: the deployment models of language models' files
   (GGUF, Hugging Face's directories), whose weights go into the image as the file stores them (Spingalett.Gguf.c).
   The layers are those of a graph over cells (a sequence of tokens): embeddings, normalizations of each cell,
   linear layers (1 x 1 convolutions of one group), attention, additions, products and concatenations. */

#pragma once

#include "Spingalett.Private.h"

typedef struct {
    LayerType type;                 /* LAYER_EMBEDDING, _RMS_NORM, _LAYER_NORM, _CONV2D (a linear layer), _ATTENTION,
                                       _ADD, _MULTIPLY or _CONCAT */
    ActivationFunction activation;
    PrecisionMode precision;        /* of the weights (embeddings, linear layers; normalizations are FLOAT32) */
    uint32_t input_count;
    uint32_t inputs[SPINGALETT_MAX_INPUTS];     /* layer indices, 0 the network's input */
    uint32_t channels;              /* the output's channels a cell (linear layers, embeddings, attention: heads x
                                       head size); normalizations, additions and products: 0 (their input's) */
    uint32_t vocabulary;            /* embedding: rows of its table */
    bool positions;                 /* embedding: a learned vector a cell */
    uint32_t heads, kv_heads, window;               /* attention */
    float theta;                    /* attention: the rotary embeddings' base, 0 for none */
    bool causal;
    uint32_t rotary;                /* attention (format 10): the values of a head rotated (0: all) */
    bool qk_norm;                   /* attention (format 10): each head's queries and keys normalized (RMS, eps; the
                                       gammas, head values of the queries' then of the keys', its weights) */
    bool freqs;                     /* attention (format 10): rotary frequencies of its own (rotary / 2 floats, its
                                       scales) */
    float score_scale;              /* attention (format 10): the scores' scale (0: 1 / sqrt(head)) */
    float eps;                      /* normalizations */
    uint32_t tied;                  /* linear: the embedding (a layer index) whose table are its weights, or 0 */
} SpgLayerSpec;

typedef struct SpgImageBuilder SpgImageBuilder;

/* A network whose input is `cells` values (token ids: an input of one row of cells), of loss `loss`. */
SpgImageBuilder *spingalett_image_new(uint32_t cells, LossFunction loss);
void spingalett_image_free(SpgImageBuilder *b);
/* Adds a layer after those before it: its index (from 1), 0 when its spec does not fit its inputs (the error set). */
uint32_t spingalett_image_layer(SpgImageBuilder *b, const SpgLayerSpec *spec);
/* The image of the layers added: header, layer table and input lists written, every section of parameters in
   place and zero, an aligned buffer the caller fills (spingalett_image_weights() and the others) and seals; NULL
   with the error set when it does not fit memory. */
uint8_t *spingalett_image_layout(SpgImageBuilder *b, size_t *size);
/* Where layer `index`'s weights (its rows: rows x spingalett_slett_row_bytes(); an attention's gammas), row scales
   (integer precisions; an attention's frequencies) and biases lie in the image, NULL for those it has not; the rows
   and row length of its weights. */
uint8_t *spingalett_image_weights(const SpgImageBuilder *b, uint8_t *image, uint32_t index, uint32_t *rows,
                                  uint32_t *row_len);
float *spingalett_image_scales(const SpgImageBuilder *b, uint8_t *image, uint32_t index);
float *spingalett_image_biases(const SpgImageBuilder *b, uint8_t *image, uint32_t index);
/* Writes the checksums of a filled image. */
void spingalett_image_seal(uint8_t *image, size_t size);
