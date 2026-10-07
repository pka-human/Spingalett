/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/*
 * Convolution and pooling layers over batches of channels-last tensors: n samples of a layer are
 * n rows of height x width x channels floats, element (y, x, c) of a sample at
 * (y * width + x) * channels + c.
 *
 * A convolution is a matrix product. With one row per output pixel holding the input window it
 * reads (im2col: kernel_h runs of kernel_w x channels contiguous floats), the outputs of a batch are
 * rows x window times the transposed weights [filters x window], so the native GEMM computes them.
 * The data gradient is gathered the same way: one row per input pixel holding the output-pixel
 * gradients whose windows cover it, times the weights regrouped per input channel. No pixel is
 * written by two rows, so every pass is parallel without atomics. The weight gradient is the
 * product of the transposed output gradients and the im2col rows. Rows are taken in chunks whose
 * gathered windows fit a bounded buffer.
 */

#include "Spingalett.Private.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#if defined(_OPENMP)
#include <omp.h>
#endif

/* Gathered rows are taken in chunks of at most this many floats (8 MB), at least one row. */
#define CONV_CHUNK_FLOATS (1u << 21)

static inline bool use_omp(ComputeMode mode, uint64_t work) {
    return spingalett_use_omp(mode, work);
}

size_t spingalett_conv_scratch_floats(const NeuralNetwork *net, uint32_t capacity, bool training) {
    size_t need = 0;
    for (uint32_t l = 0; l + 1 < net->layers; l++) {
        const LayerShape *in = &net->shapes[l], *out = &net->shapes[l + 1];
        if (out->type != LAYER_CONV2D) continue;
        size_t fwd = (size_t)out->kernel_h * out->kernel_w * in->channels;     /* per output pixel */
        size_t total = fwd * out->height * out->width * capacity;
        size_t chunk = total < CONV_CHUNK_FLOATS ? total : (fwd > CONV_CHUNK_FLOATS ? fwd : CONV_CHUNK_FLOATS);
        if (chunk > need) need = chunk;
        if (training && l > 0) {
            size_t bwd = (size_t)out->kernel_h * out->kernel_w * out->channels;     /* per input pixel */
            total = bwd * in->height * in->width * capacity;
            chunk = total < CONV_CHUNK_FLOATS ? total : (bwd > CONV_CHUNK_FLOATS ? bwd : CONV_CHUNK_FLOATS);
            /* the regrouped weights of the data gradient come first */
            size_t w = ((size_t)in->channels * bwd + 15u) & ~(size_t)15u;
            if (w + chunk > need) need = w + chunk;
        }
    }
    return need;
}

/* Rows of a chunk: as many as fit CONV_CHUNK_FLOATS with row_len floats each (at least 1). */
static uint64_t chunk_rows(uint64_t total, size_t row_len) {
    uint64_t rows = row_len ? CONV_CHUNK_FLOATS / row_len : total;
    if (rows == 0) rows = 1;
    return rows < total ? rows : total;
}

/* im2col: rows [p0, p0 + rows) of all output pixels of the batch (pixel p is sample p / pixels,
   output position p % pixels), each the kernel_h x kernel_w x channels window it reads. */
static void gather_windows(const float *x, const LayerShape *in, const LayerShape *out, uint64_t p0, uint64_t rows,
                           float *col, ComputeMode mode) {
    const uint32_t H = in->height, W = in->width, C = in->channels, OW = out->width;
    const uint32_t KH = out->kernel_h, KW = out->kernel_w;
    const uint64_t pixels = (uint64_t)out->height * OW, sample = (uint64_t)H * W * C;
    const size_t run = (size_t)KW * C, K = (size_t)KH * run;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if(use_omp(mode, rows * K))
#endif
    for (int64_t r = 0; r < (int64_t)rows; r++) {
        uint64_t p = p0 + (uint64_t)r, n = p / pixels;
        uint32_t pix = (uint32_t)(p % pixels), oh = pix / OW, ow = pix % OW;
        const float *xs = x + n * sample;
        float *dst = col + (size_t)r * K;
        int64_t ih0 = (int64_t)oh * out->stride_h - out->pad_h, iw0 = (int64_t)ow * out->stride_w - out->pad_w;
        /* window columns [a, b) are inside the image */
        int64_t a = iw0 < 0 ? -iw0 : 0, b = iw0 + KW > (int64_t)W ? (int64_t)W - iw0 : KW;
        if (b < a) b = a;
        for (uint32_t kh = 0; kh < KH; kh++, dst += run) {
            int64_t ih = ih0 + kh;
            if (ih < 0 || ih >= (int64_t)H) { memset(dst, 0, run * sizeof(float)); continue; }
            if (a > 0) memset(dst, 0, (size_t)a * C * sizeof(float));
            if (b > a) memcpy(dst + (size_t)a * C, xs + ((size_t)ih * W + (size_t)(iw0 + a)) * C, (size_t)(b - a) * C * sizeof(float));
            if (b < KW) memset(dst + (size_t)b * C, 0, (size_t)(KW - b) * C * sizeof(float));
        }
    }
    (void)mode;
}

/* The transpose of im2col for the data gradient: rows [q0, q0 + rows) of all input pixels of the
   batch, each holding, for every kernel position (kh, kw), the filters' gradients at the output
   pixel whose window puts kh, kw on this input pixel (zeros where there is none). */
static void gather_output_grads(const float *dy, const LayerShape *in, const LayerShape *out, uint64_t q0,
                                uint64_t rows, float *col, ComputeMode mode) {
    const uint32_t W = in->width, OH = out->height, OW = out->width, OC = out->channels;
    const uint32_t KH = out->kernel_h, KW = out->kernel_w, SH = out->stride_h, SW = out->stride_w;
    const uint64_t pixels = (uint64_t)in->height * W, sample = (uint64_t)OH * OW * OC;
    const size_t K = (size_t)KH * KW * OC;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if(use_omp(mode, rows * K))
#endif
    for (int64_t r = 0; r < (int64_t)rows; r++) {
        uint64_t q = q0 + (uint64_t)r, n = q / pixels;
        uint32_t pix = (uint32_t)(q % pixels), ih = pix / W, iw = pix % W;
        const float *ds = dy + n * sample;
        float *dst = col + (size_t)r * K;
        for (uint32_t kh = 0; kh < KH; kh++) {
            int64_t th = (int64_t)ih + out->pad_h - kh;         /* = oh * SH when an output covers it */
            bool row_ok = th >= 0 && th % SH == 0 && th / SH < OH;
            for (uint32_t kw = 0; kw < KW; kw++, dst += OC) {
                int64_t tw = (int64_t)iw + out->pad_w - kw;
                if (row_ok && tw >= 0 && tw % SW == 0 && tw / SW < OW)
                    memcpy(dst, ds + ((size_t)(th / SH) * OW + (size_t)(tw / SW)) * OC, OC * sizeof(float));
                else
                    memset(dst, 0, OC * sizeof(float));
            }
        }
    }
    (void)mode;
}

/* Rows per product when nothing is gathered (pointwise convolutions): the GEMM counts in 32 bits. */
#define POINTWISE_ROWS (1u << 30)

/* A 1 x 1 convolution with stride 1 and no padding reads each input pixel as its window. */
static bool pointwise(const LayerShape *out) {
    return out->kernel_h == 1 && out->kernel_w == 1 && out->stride_h == 1 && out->stride_w == 1 &&
           out->pad_h == 0 && out->pad_w == 0;
}

void spingalett_conv_forward(const NeuralNetwork *net, uint32_t l, const float *x, float *y, uint32_t n,
                             float *scratch, SpingalettGemmScratch *gemm, ComputeMode mode) {
    const LayerShape *in = &net->shapes[l], *out = &net->shapes[l + 1];
    const uint32_t OC = out->channels;
    const size_t K = (size_t)out->kernel_h * out->kernel_w * in->channels;
    const uint64_t total = (uint64_t)n * out->height * out->width;
    const float *Wt = SPINGALETT_WEIGHT_MTX_PTR(net, l), *bias = net->biases + net->bias_offsets[l];

    if (pointwise(out)) {
        for (uint64_t p0 = 0; p0 < total; p0 += POINTWISE_ROWS) {
            uint64_t rows = total - p0 < POINTWISE_ROWS ? total - p0 : POINTWISE_ROWS;
            spingalett_gemm(gemm, mode, false, true, (uint32_t)rows, OC, (uint32_t)K, 1.0f, x + p0 * K, K, Wt, K,
                            0.0f, y + p0 * OC, OC);
        }
    } else {
        uint64_t step = chunk_rows(total, K);
        for (uint64_t p0 = 0; p0 < total; p0 += step) {
            uint64_t rows = total - p0 < step ? total - p0 : step;
            gather_windows(x, in, out, p0, rows, scratch, mode);
            spingalett_gemm(gemm, mode, false, true, (uint32_t)rows, OC, (uint32_t)K, 1.0f, scratch, K, Wt, K,
                            0.0f, y + p0 * OC, OC);
        }
    }
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if(use_omp(mode, total * OC))
#endif
    for (int64_t p = 0; p < (int64_t)total; p++)
        spingalett_vec_axpy(y + (size_t)p * OC, bias, OC, 1.0f);
}

void spingalett_conv_backward_data(const NeuralNetwork *net, uint32_t l, const float *dy, float *dx, uint32_t n,
                                   float *scratch, SpingalettGemmScratch *gemm, ComputeMode mode) {
    const LayerShape *in = &net->shapes[l], *out = &net->shapes[l + 1];
    const uint32_t C = in->channels, OC = out->channels, KH = out->kernel_h, KW = out->kernel_w;
    const size_t K = (size_t)KH * KW * C;
    const uint64_t total = (uint64_t)n * in->height * in->width;
    const float *Wt = SPINGALETT_WEIGHT_MTX_PTR(net, l);

    if (pointwise(out)) {           /* dx = dy * W */
        for (uint64_t q0 = 0; q0 < total; q0 += POINTWISE_ROWS) {
            uint64_t rows = total - q0 < POINTWISE_ROWS ? total - q0 : POINTWISE_ROWS;
            spingalett_gemm(gemm, mode, false, false, (uint32_t)rows, C, OC, 1.0f, dy + q0 * OC, OC, Wt, K,
                            0.0f, dx + q0 * C, C);
        }
        return;
    }
    /* The weights regrouped per input channel: Wr[c][(kh, kw, oc)] = W[oc][(kh, kw, c)], at the
       end of the scratch buffer; the gathered rows use the rest. */
    const size_t Kr = (size_t)KH * KW * OC, wsize = (size_t)C * Kr;
    float *Wr = scratch;
    float *col = scratch + ((wsize + 15u) & ~(size_t)15u);
    for (uint32_t oc = 0; oc < OC; oc++)
        for (size_t s = 0; s < (size_t)KH * KW; s++)
            for (uint32_t c = 0; c < C; c++)
                Wr[(size_t)c * Kr + s * OC + oc] = Wt[(size_t)oc * K + s * C + c];

    uint64_t step = chunk_rows(total, Kr);
    for (uint64_t q0 = 0; q0 < total; q0 += step) {
        uint64_t rows = total - q0 < step ? total - q0 : step;
        gather_output_grads(dy, in, out, q0, rows, col, mode);
        spingalett_gemm(gemm, mode, false, true, (uint32_t)rows, C, (uint32_t)Kr, 1.0f, col, Kr, Wr, Kr,
                        0.0f, dx + q0 * C, C);
    }
}

void spingalett_conv_backward_weights(NeuralNetwork *net, uint32_t l, const float *x, const float *dy, uint32_t n,
                                      float scale, float beta, float *scratch, SpingalettGemmScratch *gemm,
                                      ComputeMode mode) {
    const LayerShape *in = &net->shapes[l], *out = &net->shapes[l + 1];
    const uint32_t OC = out->channels;
    const size_t K = (size_t)out->kernel_h * out->kernel_w * in->channels;
    const uint64_t total = (uint64_t)n * out->height * out->width;
    float *gW = SPINGALETT_GRAD_W_MTX_PTR(net, l), *gB = net->grad_biases + net->bias_offsets[l];

    /* gW[oc x window] = scale * dy^T * windows (+ beta * gW) */
    if (pointwise(out)) {
        for (uint64_t p0 = 0; p0 < total; p0 += POINTWISE_ROWS) {
            uint64_t rows = total - p0 < POINTWISE_ROWS ? total - p0 : POINTWISE_ROWS;
            spingalett_gemm(gemm, mode, true, false, OC, (uint32_t)K, (uint32_t)rows, scale, dy + p0 * OC, OC,
                            x + p0 * K, K, p0 == 0 ? beta : 1.0f, gW, K);
        }
    } else {
        uint64_t step = chunk_rows(total, K);
        for (uint64_t p0 = 0; p0 < total; p0 += step) {
            uint64_t rows = total - p0 < step ? total - p0 : step;
            gather_windows(x, in, out, p0, rows, scratch, mode);
            spingalett_gemm(gemm, mode, true, false, OC, (uint32_t)K, (uint32_t)rows, scale, dy + p0 * OC, OC,
                            scratch, K, p0 == 0 ? beta : 1.0f, gW, K);
        }
    }
    /* gB = scale * column sums of dy (+ beta * gB): blocks of rows summed in float in parallel, the
       block sums added in order in double, so the result does not depend on the thread count */
    const uint64_t block = 64, blocks = (total + block - 1) / block;
    float *part = (float *)malloc((size_t)blocks * OC * sizeof(float));
    double *sum = (double *)calloc(OC, sizeof(double));
    if (!part || !sum) {
        free(part); free(sum);
        set_error(SPINGALETT_ERR_ALLOC, "conv: bias gradient allocation failed");
        return;
    }
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if(use_omp(mode, total * OC))
#endif
    for (int64_t b = 0; b < (int64_t)blocks; b++) {
        float *ps = part + (size_t)b * OC;
        uint64_t p = (uint64_t)b * block, end = p + block < total ? p + block : total;
        memcpy(ps, dy + (size_t)p * OC, OC * sizeof(float));
        for (p++; p < end; p++) spingalett_vec_axpy(ps, dy + (size_t)p * OC, OC, 1.0f);
    }
    for (uint64_t b = 0; b < blocks; b++)
        for (uint32_t oc = 0; oc < OC; oc++) sum[oc] += part[(size_t)b * OC + oc];
    for (uint32_t oc = 0; oc < OC; oc++)
        gB[oc] = (float)(scale * sum[oc]) + (beta == 0.0f ? 0.0f : beta * gB[oc]);
    free(part);
    free(sum);
}

/* ---- pooling: windows are clipped to the image (padding cells are skipped) */

static inline void window_bounds(const LayerShape *in, const LayerShape *out, uint32_t oh, uint32_t ow,
                                 uint32_t *h0, uint32_t *h1, uint32_t *w0, uint32_t *w1) {
    int64_t hs = (int64_t)oh * out->stride_h - out->pad_h, ws = (int64_t)ow * out->stride_w - out->pad_w;
    int64_t he = hs + out->kernel_h, we = ws + out->kernel_w;
    *h0 = (uint32_t)(hs < 0 ? 0 : hs); *h1 = (uint32_t)(he > (int64_t)in->height ? in->height : he);
    *w0 = (uint32_t)(ws < 0 ? 0 : ws); *w1 = (uint32_t)(we > (int64_t)in->width ? in->width : we);
}

void spingalett_pool_forward(const NeuralNetwork *net, uint32_t l, const float *x, float *y, uint32_t n,
                             ComputeMode mode) {
    const LayerShape *in = &net->shapes[l], *out = &net->shapes[l + 1];
    const uint32_t C = in->channels, W = in->width;
    const size_t in_size = net->topology[l], out_size = net->topology[l + 1];
    const bool max = out->type == LAYER_MAX_POOL2D;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if(use_omp(mode, (uint64_t)n * out_size * out->kernel_h * out->kernel_w))
#endif
    for (int64_t s = 0; s < (int64_t)n; s++) {
        const float *xs = x + (size_t)s * in_size;
        float *ys = y + (size_t)s * out_size;
        for (uint32_t oh = 0; oh < out->height; oh++)
            for (uint32_t ow = 0; ow < out->width; ow++) {
                float *o = ys + ((size_t)oh * out->width + ow) * C;
                uint32_t h0, h1, w0, w1;
                window_bounds(in, out, oh, ow, &h0, &h1, &w0, &w1);
                memcpy(o, xs + ((size_t)h0 * W + w0) * C, C * sizeof(float));
                for (uint32_t h = h0; h < h1; h++)
                    for (uint32_t w = (h == h0 ? w0 + 1 : w0); w < w1; w++) {
                        const float *v = xs + ((size_t)h * W + w) * C;
                        if (max) { for (uint32_t c = 0; c < C; c++) if (v[c] > o[c]) o[c] = v[c]; }
                        else     { for (uint32_t c = 0; c < C; c++) o[c] += v[c]; }
                    }
                if (!max) {
                    float inv = 1.0f / (float)((h1 - h0) * (w1 - w0));
                    for (uint32_t c = 0; c < C; c++) o[c] *= inv;
                }
            }
    }
    (void)mode;
}

void spingalett_pool_backward(const NeuralNetwork *net, uint32_t l, const float *x, const float *dy, float *dx,
                              uint32_t n, ComputeMode mode) {
    const LayerShape *in = &net->shapes[l], *out = &net->shapes[l + 1];
    const uint32_t C = in->channels, W = in->width;
    const size_t in_size = net->topology[l], out_size = net->topology[l + 1];
    const bool max = out->type == LAYER_MAX_POOL2D;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if(use_omp(mode, (uint64_t)n * out_size * out->kernel_h * out->kernel_w))
#endif
    for (int64_t s = 0; s < (int64_t)n; s++) {
        const float *xs = x + (size_t)s * in_size, *ds = dy + (size_t)s * out_size;
        float *dxs = dx + (size_t)s * in_size;
        memset(dxs, 0, in_size * sizeof(float));
        for (uint32_t oh = 0; oh < out->height; oh++)
            for (uint32_t ow = 0; ow < out->width; ow++) {
                const float *g = ds + ((size_t)oh * out->width + ow) * C;
                uint32_t h0, h1, w0, w1;
                window_bounds(in, out, oh, ow, &h0, &h1, &w0, &w1);
                if (max) {
                    /* the gradient goes to the cell the forward pass chose: the first maximum */
                    for (uint32_t c = 0; c < C; c++) {
                        size_t best = ((size_t)h0 * W + w0) * C + c;
                        for (uint32_t h = h0; h < h1; h++)
                            for (uint32_t w = (h == h0 ? w0 + 1 : w0); w < w1; w++) {
                                size_t at = ((size_t)h * W + w) * C + c;
                                if (xs[at] > xs[best]) best = at;
                            }
                        dxs[best] += g[c];
                    }
                } else {
                    float inv = 1.0f / (float)((h1 - h0) * (w1 - w0));
                    for (uint32_t h = h0; h < h1; h++)
                        for (uint32_t w = w0; w < w1; w++) {
                            float *d = dxs + ((size_t)h * W + w) * C;
                            for (uint32_t c = 0; c < C; c++) d[c] += g[c] * inv;
                        }
                }
            }
    }
    (void)mode;
}
