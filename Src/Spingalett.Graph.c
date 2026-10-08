/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/*
 * Networks as graphs: the layers that combine the outputs of others (addition, concatenation along
 * the channels) and global average pooling, forward and backward over batches of channels-last
 * tensors, and the plan that lets inference outputs share memory.
 *
 * Every kernel works sample by sample with the same order of operations whatever the batch and the
 * thread count: additions take their inputs in order, pooling sums cells in order, so a sample's
 * result depends on nothing else.
 */

#include "Spingalett.Private.h"
#include <stdlib.h>
#include <string.h>

/* Elementwise work below which these memory-bound passes stay on one thread. */
static inline bool parallel(ComputeMode mode, uint32_t n, uint64_t work) {
    return n > 1 && spingalett_use_omp(mode, work);
}

/* ------------------------------------------------------------------------- memory plan */

typedef struct {
    uint64_t offset, size;
    uint32_t last;              /* the last step that reads it */
} LiveBuffer;

uint64_t spingalett_plan_buffers(uint32_t outputs, const uint64_t *sizes, uint32_t count, const uint32_t *steps,
                                 const uint32_t *const *reads, const uint32_t *read_count, uint64_t align,
                                 uint64_t *offsets) {
    uint32_t *last = (uint32_t *)malloc(((size_t)outputs + 1) * sizeof(uint32_t));
    LiveBuffer *live = (LiveBuffer *)malloc(((size_t)outputs + 1) * sizeof(LiveBuffer));
    if (!last || !live) {
        free(last);
        free(live);
        return UINT64_MAX;
    }
    for (uint32_t t = 0; t < outputs; t++) {
        last[t] = 0;
        offsets[t] = UINT64_MAX;
    }
    for (uint32_t s = 0; s < count; s++) {     /* readers come after their producer */
        last[steps[s]] = s;                    /* an output nothing reads lives for its own step */
        for (uint32_t k = 0; k < read_count[s]; k++) last[reads[s][k]] = s;
    }

    uint64_t total = 0;
    uint32_t n = 0;                            /* live buffers, by increasing offset */
    for (uint32_t s = 0; s < count; s++) {
        uint32_t keep = 0;
        for (uint32_t i = 0; i < n; i++)
            if (live[i].last >= s) live[keep++] = live[i];
        n = keep;
        uint32_t t = steps[s];
        if (sizes[t] == 0) continue;
        uint64_t size = (sizes[t] + align - 1) / align * align, at = 0;
        uint32_t pos = 0;
        /* the lowest gap that holds it */
        for (; pos < n; pos++) {
            if (at + size <= live[pos].offset) break;
            uint64_t end = live[pos].offset + live[pos].size;
            if (end > at) at = end;
        }
        memmove(live + pos + 1, live + pos, (size_t)(n - pos) * sizeof(LiveBuffer));
        live[pos] = (LiveBuffer){at, size, last[t]};
        n++;
        offsets[t] = at;
        if (at + size > total) total = at + size;
    }
    free(last);
    free(live);
    return total;
}

bool spingalett_fused_norm(const NeuralNetwork *net, const uint32_t *uses, uint32_t l) {
    LayerType type = net->shapes[l].type;
    return (type == LAYER_DENSE || type == LAYER_CONV2D) && net->act_func[l - 1] == ACT_NONE && l + 1 < net->layers &&
           net->shapes[l + 1].type == LAYER_BATCH_NORM && spingalett_source(net, l + 1) == l && uses[l] == 1 &&
           net->act_func[l] != ACT_SOFTMAX;
}

uint64_t spingalett_plan_outputs(const NeuralNetwork *net, const uint32_t *uses, uint64_t *offsets) {
    const uint32_t L = net->layers;
    uint64_t *sizes = (uint64_t *)malloc((size_t)L * sizeof(uint64_t));
    uint32_t *steps = (uint32_t *)malloc((size_t)L * sizeof(uint32_t));
    const uint32_t **reads = (const uint32_t **)malloc((size_t)L * sizeof(uint32_t *));
    uint32_t *read_count = (uint32_t *)malloc((size_t)L * sizeof(uint32_t));
    uint64_t total = UINT64_MAX;
    if (sizes && steps && reads && read_count) {
        uint32_t count = 0;
        for (uint32_t t = 0; t < L; t++) sizes[t] = t == 0 || t + 1 == L ? 0u : net->topology[t];
        for (uint32_t l = 1; l < L; l++) {
            /* a fused product computes its normalization's output, from the product's inputs */
            bool fused = spingalett_fused_norm(net, uses, l);
            if (fused) sizes[l] = 0;
            steps[count] = fused ? l + 1 : l;
            reads[count] = spingalett_inputs(net, l);
            read_count[count] = spingalett_input_count(net, l);
            count++;
            if (fused) l++;
        }
        total = spingalett_plan_buffers(L, sizes, count, steps, reads, read_count, 16u, offsets);
    }
    free(sizes);
    free(steps);
    free(reads);
    free(read_count);
    return total;
}

/* ------------------------------------------------------------------------- forward */

void spingalett_add_forward(const float *const *x, uint32_t count, float *y, uint32_t n, uint64_t size,
                            ActivationFunction act, ComputeMode mode) {
    SPINGALETT_PARALLEL_FOR(parallel(mode, n, (uint64_t)n * size * count),
        for (int64_t s = 0; s < (int64_t)n; s++) {
            const uint64_t base = (uint64_t)s * size;
            float *restrict o = y + base;
            const float *restrict a = x[0] + base;
            if (count == 1) {
                memcpy(o, a, (size_t)size * sizeof(float));
            } else {
                const float *restrict b = x[1] + base;
                for (uint64_t i = 0; i < size; i++) o[i] = a[i] + b[i];
                for (uint32_t k = 2; k < count; k++) {
                    const float *restrict c = x[k] + base;
                    for (uint64_t i = 0; i < size; i++) o[i] += c[i];
                }
            }
            apply_activation_batch(o, (uint32_t)size, act);
        }
    );
    (void)mode;
}

/* Copies of a few channels (a cell of a narrow input), kept inline. */
static inline void copy_channels(float *restrict dst, const float *restrict src, uint32_t n) {
    if (n > 16) { memcpy(dst, src, (size_t)n * sizeof(float)); return; }
    for (uint32_t i = 0; i < 16; i++) if (i < n) dst[i] = src[i];
}

void spingalett_concat_forward(const float *const *x, const uint32_t *channels, uint32_t count, float *y, uint32_t n,
                               uint64_t cells, ActivationFunction act, ComputeMode mode) {
    uint32_t C = 0;
    for (uint32_t k = 0; k < count; k++) C += channels[k];
    const uint64_t size = cells * C;
    SPINGALETT_PARALLEL_FOR(parallel(mode, n, (uint64_t)n * size),
        for (int64_t s = 0; s < (int64_t)n; s++) {
            float *o = y + (uint64_t)s * size;
            for (uint32_t k = 0, c0 = 0; k < count; c0 += channels[k], k++) {
                const uint32_t ck = channels[k];
                const float *xs = x[k] + (uint64_t)s * cells * ck;
                for (uint64_t p = 0; p < cells; p++) copy_channels(o + p * C + c0, xs + p * ck, ck);
            }
            apply_activation_batch(o, (uint32_t)size, act);
        }
    );
    (void)mode;
}

void spingalett_global_pool_forward(const float *x, float *y, uint32_t n, uint64_t cells, uint32_t C,
                                    ActivationFunction act, ComputeMode mode) {
    const float inv = 1.0f / (float)cells;
    SPINGALETT_PARALLEL_FOR(parallel(mode, n, (uint64_t)n * cells * C),
        for (int64_t s = 0; s < (int64_t)n; s++) {
            const float *restrict xs = x + (uint64_t)s * cells * C;
            float *restrict o = y + (uint64_t)s * C;
            for (uint32_t c = 0; c < C; c++) o[c] = xs[c];
            for (uint64_t p = 1; p < cells; p++) {
                const float *restrict row = xs + p * C;
                for (uint32_t c = 0; c < C; c++) o[c] += row[c];
            }
            for (uint32_t c = 0; c < C; c++) o[c] *= inv;
            apply_activation_batch(o, C, act);
        }
    );
    (void)mode;
}

/* ------------------------------------------------------------------------- backward */

void spingalett_slice_backward(const float *dy, uint32_t C, uint32_t c0, uint32_t ck, float *dx, const float *x,
                               uint32_t n, uint64_t cells, ActivationFunction act, bool accumulate, ComputeMode mode) {
    const uint64_t size = cells * ck;
    SPINGALETT_PARALLEL_FOR(parallel(mode, n, (uint64_t)n * size),
        for (int64_t s = 0; s < (int64_t)n; s++) {
            const float *restrict g = dy + (uint64_t)s * cells * C + c0;
            float *restrict d = dx + (uint64_t)s * size;
            if (ck == C) {                  /* an addition's input, or the only input */
                if (accumulate) for (uint64_t i = 0; i < size; i++) d[i] += g[i];
                else memcpy(d, g, (size_t)size * sizeof(float));
            } else if (accumulate) {
                for (uint64_t p = 0; p < cells; p++)
                    for (uint32_t c = 0; c < ck; c++) d[p * ck + c] += g[p * C + c];
            } else {
                for (uint64_t p = 0; p < cells; p++) copy_channels(d + p * ck, g + p * C, ck);
            }
            if (act != ACT_NONE) apply_derivative_batch(d, x + (uint64_t)s * size, size, act);
        }
    );
    (void)mode;
}

void spingalett_global_pool_backward(const float *dy, float *dx, const float *x, uint32_t n, uint64_t cells, uint32_t C,
                                     ActivationFunction act, bool accumulate, ComputeMode mode) {
    const float inv = 1.0f / (float)cells;
    const uint64_t size = cells * C;
    SPINGALETT_PARALLEL_FOR(parallel(mode, n, (uint64_t)n * size),
        for (int64_t s = 0; s < (int64_t)n; s++) {
            const float *restrict g = dy + (uint64_t)s * C;
            float *restrict d = dx + (uint64_t)s * size;
            for (uint64_t p = 0; p < cells; p++) {
                float *restrict row = d + p * C;
                if (accumulate) for (uint32_t c = 0; c < C; c++) row[c] += g[c] * inv;
                else for (uint32_t c = 0; c < C; c++) row[c] = g[c] * inv;
            }
            if (act != ACT_NONE) apply_derivative_batch(d, x + (uint64_t)s * size, size, act);
        }
    );
    (void)mode;
}

void spingalett_gradient_sum(float *delta, const float *add, const float *y, const float *dmask, uint32_t n,
                             uint64_t size, ActivationFunction act, bool last, ComputeMode mode) {
    SPINGALETT_PARALLEL_FOR(parallel(mode, n, (uint64_t)n * size),
        for (int64_t s = 0; s < (int64_t)n; s++) {
            const uint64_t base = (uint64_t)s * size;
            float *restrict d = delta + base;
            if (add) {
                const float *restrict a = add + base;
                for (uint64_t i = 0; i < size; i++) d[i] += a[i];
            }
            if (!last) continue;
            if (dmask) spingalett_vec_mul(d, dmask + base, size);
            else apply_derivative_batch(d, y + base, size, act);
        }
    );
    (void)mode;
}
