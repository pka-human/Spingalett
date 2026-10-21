/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/* The layers of block precisions (format 10: GGUF's types) over many cells, for batched prediction
   (Spingalett.Model.c) and sessions (Spingalett.Session.c): the bits of the engine's spingalett_block_dot(). */

#pragma once

#include "Spingalett.Private.h"

/* Bytes of scratch spingalett_block_layer() takes for `items` cells of layer L's inputs on `threads` threads. */
uint64_t spingalett_block_scratch(const SlettLayer *L, uint64_t items, int threads);

/* A dense layer, or a linear layer (a 1 x 1 convolution of one group), of a block precision on `items` cells of its
   row length each: y[i x rows + j] = bias[j] + the dot product of row j with cell i, before activation, each cell
   quantized in the precision's blocks (spingalett_quantize_blocks()); on `threads` threads (1: the calling thread).
   scratch: spingalett_block_scratch() bytes, aligned to 64. */
void spingalett_block_layer(const uint8_t *image, const SlettLayer *L, const float *x, float *y, uint64_t items,
                            uint8_t *scratch, int threads);
