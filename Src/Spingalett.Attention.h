/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/* Attention of sessions over the keys they keep (Spingalett.Session.c), in vectors, with the bits of the engine's
   spingalett_engine_attend(). */

#pragma once

#include "Spingalett.Private.h"

/* Keys kept in blocks of SPG_KEY_BLOCK positions, transposed: value d of the key at position p at
   block[(p / SPG_KEY_BLOCK) head SPG_KEY_BLOCK + d SPG_KEY_BLOCK + p % SPG_KEY_BLOCK]. */
#define SPG_KEY_BLOCK 8u

/* Key p (head values) into its place among blocks of keys. */
static inline void spingalett_key_store(float *blocks, uint32_t head, uint32_t p, const float *key) {
    float *at = blocks + (size_t)(p / SPG_KEY_BLOCK) * head * SPG_KEY_BLOCK + p % SPG_KEY_BLOCK;
    for (uint32_t d = 0; d < head; d++) at[(size_t)d * SPG_KEY_BLOCK] = key[d];
}

/* One query's attention over the keys at positions first .. first + count - 1 of `blocks` and their values (value p at
   values + p value_stride), as spingalett_engine_attend() computes it over the same keys and values: each score the
   sum of q[d] k[d] in the order of d, times scale, the softmax online with spingalett_expf(); into o[head]. */
void spingalett_attend_blocks(const float *q, const float *blocks, const float *values, size_t value_stride,
                              uint32_t first, uint32_t count, uint32_t head, float scale, float *o);
