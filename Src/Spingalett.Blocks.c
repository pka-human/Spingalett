/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/*
 * Layers of block precisions (GGUF's types) over many cells. Each cell is quantized in blocks of the type's
 * size (32 or 256 values: a scale a block, the sums of every 16 codes), then every output is the engine's
 * spingalett_block_dot() of its row with the cell: the integer sums of a block's sub-blocks, scaled by their
 * integer scales and added in integers, then the block's float term added to the row's sum, block after
 * block. The outputs are split between threads by rows, each computed whole by one of them, so that a cell's
 * outputs do not depend on the cells computed with it or on the threads.
 */

#include "Spingalett.Blocks.h"
#include <string.h>

/* Rows of the weights a thread takes at a time. */
#define BLOCK_ROWS 64u

uint64_t spingalett_block_scratch(const SlettLayer *L, uint64_t items) {
    return slett_block_input_bytes(items * L->row_len);
}

void spingalett_block_layer(const uint8_t *image, const SlettLayer *L, const float *x, float *y, uint64_t items,
                            uint8_t *scratch, int threads) {
    const PrecisionMode p = L->precision;
    const uint32_t C = L->row_len, OC = L->rows, S = slett_block_values(p);
    const uint64_t n = items * C;
    int8_t *xq = (int8_t *)scratch;
    float *dx = (float *)(void *)(scratch + slett_align(n));
    int32_t *sums = (int32_t *)(void *)(scratch + slett_align(n) + slett_align(n / 8u));
    const float *bias = (const float *)(const void *)(image + L->biases);
    const uint8_t *weights = image + L->weights;
    const size_t row = (size_t)spingalett_slett_row_bytes(p, C);
    (void)threads;
    SPINGALETT_PARALLEL_FOR_THREADS(threads, threads > 1 && items > 1,
        for (int64_t i = 0; i < (int64_t)items; i++)
            spingalett_quantize_blocks(x + (size_t)i * C, C, S, xq + (size_t)i * C, dx + (size_t)i * (C / S),
                                       sums + (size_t)i * (C / 16u));
    );
    const int64_t blocks = (int64_t)((OC + BLOCK_ROWS - 1u) / BLOCK_ROWS);
    SPINGALETT_PARALLEL_FOR_THREADS(threads, threads > 1,
        for (int64_t b = 0; b < blocks; b++) {
            const uint32_t j0 = (uint32_t)b * BLOCK_ROWS, end = OC - j0 < BLOCK_ROWS ? OC : j0 + BLOCK_ROWS;
            for (uint64_t i = 0; i < items; i++)
                for (uint32_t j = j0; j < end; j++)
                    y[i * OC + j] = bias[j] + spingalett_block_dot(p, weights + j * row, xq + i * C, dx + i * (C / S),
                                                                   sums + i * (C / 16u), C);
        }
    );
}
