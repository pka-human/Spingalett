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
 * product of the transposed output gradients and the im2col rows.
 *
 * The native GEMM reads these rows through operand sources: windows are gathered straight into
 * its packed panels (implicit im2col), which stay in cache, and no row matrix is ever stored. With
 * OpenBLAS, rows are gathered in chunks whose windows fit a bounded buffer.
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

/* Whether rows are gathered by the native GEMM itself (implicit im2col) rather than into memory. */
static inline bool implicit(ComputeMode mode) { return mode != COMPUTE_OPENBLAS; }

size_t spingalett_conv_scratch_floats(const NeuralNetwork *net, uint32_t capacity, bool training, ComputeMode mode) {
    size_t need = 0;
    for (uint32_t l = 0; l + 1 < net->layers; l++) {
        const LayerShape *in = &net->shapes[l], *out = &net->shapes[l + 1];
        if (out->type != LAYER_CONV2D) continue;
        size_t fwd = (size_t)out->kernel_h * out->kernel_w * in->channels;     /* per output pixel */
        if (implicit(mode)) {
            /* implicit im2col: the regrouped weights (data gradient) or a transposed weight
               gradient (narrow windows) */
            size_t w = training ? (size_t)out->channels * (fwd + 1) : 0;
            if (w > need) need = w;
            continue;
        }
        size_t total = fwd * out->height * out->width * capacity;
        size_t chunk = total < CONV_CHUNK_FLOATS ? total : (fwd > CONV_CHUNK_FLOATS ? fwd : CONV_CHUNK_FLOATS);
        if (chunk > need) need = chunk;
        if (training && fwd < 32 && out->channels > fwd) {     /* transposed weight gradient */
            size_t t = (fwd * out->channels + 15u) & ~(size_t)15u;
            if (t + chunk > need) need = t + chunk;
        }
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

/* Copies of short runs (a window row of a single-channel image is a few floats): written so that
   compilers keep them inline instead of calling memcpy. */
static inline void copy_run(float *restrict dst, const float *restrict src, size_t n) {
    if (n > 16) { memcpy(dst, src, n * sizeof(float)); return; }
    for (size_t i = 0; i < 16; i++) if (i < n) dst[i] = src[i];
}

static inline void zero_run(float *dst, size_t n) {
    if (n > 16) { memset(dst, 0, n * sizeof(float)); return; }
    for (size_t i = 0; i < 16; i++) if (i < n) dst[i] = 0.0f;
}

/* im2col: rows [p0, p0 + rows) of all output pixels of the batch (pixel p is sample p / pixels,
   output position p % pixels), each the kernel_h x kernel_w x channels window it reads. Work is
   split by output rows (sample, oh), so positions advance without divisions. */
static void gather_windows(const float *x, const LayerShape *in, const LayerShape *out, uint64_t p0, uint64_t rows,
                           float *col, ComputeMode mode) {
    const uint32_t H = in->height, W = in->width, C = in->channels, OW = out->width;
    const uint32_t KH = out->kernel_h, KW = out->kernel_w;
    const uint64_t sample = (uint64_t)H * W * C, end = p0 + rows;
    const size_t run = (size_t)KW * C, K = (size_t)KH * run;
    const uint64_t seg0 = p0 / OW, seg1 = (end + OW - 1) / OW;          /* output rows touched */
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if(use_omp(mode, rows * K))
#endif
    for (int64_t g = (int64_t)seg0; g < (int64_t)seg1; g++) {
        uint64_t n = (uint64_t)g / out->height;
        uint32_t oh = (uint32_t)((uint64_t)g % out->height);
        uint64_t first = (uint64_t)g * OW, last = first + OW;
        uint32_t ow0 = first < p0 ? (uint32_t)(p0 - first) : 0u, ow1 = last > end ? (uint32_t)(end - first) : OW;
        const float *xs = x + n * sample;
        int64_t ih0 = (int64_t)oh * out->stride_h - out->pad_h;
        for (uint32_t ow = ow0; ow < ow1; ow++) {
            float *dst = col + (size_t)(first + ow - p0) * K;
            int64_t iw0 = (int64_t)ow * out->stride_w - out->pad_w;
            /* window columns [a, b) are inside the image */
            int64_t a = iw0 < 0 ? -iw0 : 0, b = iw0 + KW > (int64_t)W ? (int64_t)W - iw0 : KW;
            if (b < a) b = a;
            for (uint32_t kh = 0; kh < KH; kh++, dst += run) {
                int64_t ih = ih0 + kh;
                if (ih < 0 || ih >= (int64_t)H) { zero_run(dst, run); continue; }
                const float *src = xs + ((size_t)ih * W + (size_t)(iw0 + a)) * C;
                if (a == 0 && b == KW) { copy_run(dst, src, run); continue; }
                zero_run(dst, (size_t)a * C);
                copy_run(dst + (size_t)a * C, src, (size_t)(b - a) * C);
                zero_run(dst + (size_t)b * C, (size_t)(KW - b) * C);
            }
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

/* ---- operand sources for the native GEMM: the same rows, gathered on demand */

typedef struct {
    const float *data;              /* x (windows) or dy (output gradients) */
    const LayerShape *in, *out;
    uint64_t base;                  /* global row of the source's row 0 */
    bool ones;                      /* windows: a column of ones follows the window (column K) */
} RowSource;

/* Elements [col, col + cols) of window rows [row, row + rows) (output pixels). Positions advance
   from row to row and segments from run to run, so the loops divide only once per call. */
static void fill_windows(const void *ctx, uint32_t row, uint32_t rows, uint32_t col, uint32_t cols, float *dst, size_t ld) {
    const RowSource *src = (const RowSource *)ctx;
    const LayerShape *in = src->in, *out = src->out;
    const uint32_t H = in->height, W = in->width, C = in->channels, OH = out->height, OW = out->width;
    const uint32_t KW = out->kernel_w, run = KW * C, K = out->kernel_h * run;
    const bool one = src->ones && col + cols > K;
    const uint32_t end = one ? K : col + cols, kh0 = col / run, rem0 = col - kh0 * run;
    const uint64_t pixels = (uint64_t)OH * OW, sample = (uint64_t)H * W * C, p = src->base + row;
    uint64_t n = p / pixels;
    uint32_t pix = (uint32_t)(p % pixels), oh = pix / OW, ow = pix % OW;
    for (uint32_t r = 0; r < rows; r++, dst += ld) {
        const float *xs = src->data + n * sample;
        int64_t ih0 = (int64_t)oh * out->stride_h - out->pad_h, iw0 = (int64_t)ow * out->stride_w - out->pad_w;
        /* floats [a, b) of each window row are inside the image */
        int64_t a = iw0 < 0 ? -iw0 * C : 0, b = iw0 + KW > (int64_t)W ? ((int64_t)W - iw0) * C : run;
        if (b < a) b = a;
        const float *base = xs + iw0 * (int64_t)C;
        float *d = dst;
        uint32_t kh = kh0, rem = rem0;
        for (uint32_t k = col; k < end; kh++, rem = 0) {
            uint32_t take = run - rem < end - k ? run - rem : end - k;
            int64_t ih = ih0 + kh, lo = rem, hi = (int64_t)rem + take;
            if (ih < 0 || ih >= (int64_t)H || hi <= a || lo >= b) {
                zero_run(d, take);
            } else if (lo >= a && hi <= b) {
                copy_run(d, base + (size_t)ih * W * C + lo, take);
            } else {
                int64_t c0 = lo > a ? lo : a, c1 = hi < b ? hi : b;
                zero_run(d, (size_t)(c0 - lo));
                copy_run(d + (c0 - lo), base + (size_t)ih * W * C + c0, (size_t)(c1 - c0));
                zero_run(d + (c1 - lo), (size_t)(hi - c1));
            }
            d += take;
            k += take;
        }
        if (one) *d = 1.0f;
        if (++ow == OW) { ow = 0; if (++oh == OH) { oh = 0; n++; } }
    }
}

/* Elements [col, col + cols) of output-gradient rows [row, row + rows) (input pixels). */
static void fill_output_grads(const void *ctx, uint32_t row, uint32_t rows, uint32_t col, uint32_t cols, float *dst,
                              size_t ld) {
    const RowSource *src = (const RowSource *)ctx;
    const LayerShape *in = src->in, *out = src->out;
    const uint32_t H = in->height, W = in->width, OH = out->height, OW = out->width, OC = out->channels;
    const uint32_t KW = out->kernel_w, SH = out->stride_h, SW = out->stride_w, end = col + cols;
    const uint32_t kpos0 = col / OC, oc0 = col - kpos0 * OC, kh0 = kpos0 / KW, kw0 = kpos0 - kh0 * KW;
    const uint64_t pixels = (uint64_t)H * W, sample = (uint64_t)OH * OW * OC, q = src->base + row;
    uint64_t n = q / pixels;
    uint32_t pix = (uint32_t)(q % pixels), ih = pix / W, iw = pix % W;
    for (uint32_t r = 0; r < rows; r++, dst += ld) {
        const float *ds = src->data + n * sample;
        float *d = dst;
        uint32_t kh = kh0, kw = kw0, oc = oc0;
        int64_t th = (int64_t)ih + out->pad_h - kh;     /* = oh * SH when an output row covers it */
        bool row_ok = th >= 0 && th % SH == 0 && th / SH < OH;
        for (uint32_t k = col; k < end; oc = 0) {
            uint32_t take = OC - oc < end - k ? OC - oc : end - k;
            int64_t tw = (int64_t)iw + out->pad_w - kw;
            if (row_ok && tw >= 0 && tw % SW == 0 && tw / SW < OW)
                copy_run(d, ds + ((size_t)(th / SH) * OW + (size_t)(tw / SW)) * OC + oc, take);
            else
                zero_run(d, take);
            d += take;
            k += take;
            if (++kw == KW) {
                kw = 0;
                kh++;
                th--;
                row_ok = th >= 0 && th % SH == 0 && th / SH < OH;
            }
        }
        if (++iw == W) { iw = 0; if (++ih == H) { ih = 0; n++; } }
    }
}

static inline bool gemm_parallel(ComputeMode mode, uint64_t m, uint64_t n, uint64_t k) {
    return mode == COMPUTE_OPENMP && m * n * k >= SPINGALETT_GEMM_PARALLEL_WORK;
}

/* A product of gathered rows: on the native kernels with the hooks (sources, epilogue); with
   OpenBLAS from memory, the epilogue then running over all of C. */
static void conv_gemm(SpingalettGemmScratch *gemm, ComputeMode mode, bool trans_a, bool trans_b, uint32_t M,
                      uint32_t N, uint32_t K, float alpha, const float *A, size_t lda, const float *B, size_t ldb,
                      float beta, float *C, size_t ldc, const SpingalettGemmHooks *hooks) {
    if (mode == COMPUTE_OPENBLAS) {
        spingalett_gemm(gemm, mode, trans_a, trans_b, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        if (hooks && hooks->epilogue) hooks->epilogue(hooks->epilogue_ctx, 0, M, 0, N, C, ldc);
        return;
    }
    spingalett_gemm_hooked(gemm, trans_a, trans_b, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc,
                           gemm_parallel(mode, M, N, K), hooks);
}

/* Epilogues: the forward pass adds the bias and applies an element-wise activation; the data
   gradient is multiplied by the derivative of the input layer's activation, read from its output
   (rows from row `base` of the batch). */
typedef struct {
    const float *bias;
    ActivationFunction act;
} BiasActivation;

static void add_bias_activate(const void *ctx, uint32_t row, uint32_t rows, uint32_t col, uint32_t cols, float *c,
                              size_t ldc) {
    const BiasActivation *e = (const BiasActivation *)ctx;
    const float *bias = e->bias + col;
    for (uint32_t r = 0; r < rows; r++) {
        float *cr = c + (size_t)r * ldc;
        for (uint32_t j = 0; j < cols; j++) cr[j] += bias[j];
        if (cols != ldc) apply_activation_bulk(cr, cols, e->act);
    }
    if (cols == ldc) apply_activation_bulk(c, (uint64_t)rows * cols, e->act);
    (void)row;
}

typedef struct {
    const float *y;                 /* the activations of the gradient's layer */
    uint64_t base;
    ActivationFunction act;
} Derivative;

static void multiply_derivative(const void *ctx, uint32_t row, uint32_t rows, uint32_t col, uint32_t cols, float *c,
                                size_t ldc) {
    const Derivative *e = (const Derivative *)ctx;
    const float *y = e->y + (e->base + row) * ldc + col;
    if (cols == ldc) { apply_derivative_batch(c, y, (uint64_t)rows * cols, e->act); return; }
    for (uint32_t r = 0; r < rows; r++) apply_derivative_batch(c + (size_t)r * ldc, y + (size_t)r * ldc, cols, e->act);
}

/* Rows per product when nothing is gathered (pointwise convolutions): the GEMM counts in 32 bits. */
#define POINTWISE_ROWS (1u << 30)

/* A 1 x 1 convolution with stride 1 and no padding reads each input pixel as its window. */
static bool pointwise(const LayerShape *out) {
    return out->kernel_h == 1 && out->kernel_w == 1 && out->stride_h == 1 && out->stride_w == 1 &&
           out->pad_h == 0 && out->pad_w == 0;
}

void spingalett_conv_forward(const NeuralNetwork *net, uint32_t l, const float *x, float *y, uint32_t n,
                             ActivationFunction act, float *scratch, SpingalettGemmScratch *gemm, ComputeMode mode) {
    const LayerShape *in = &net->shapes[l], *out = &net->shapes[l + 1];
    const uint32_t OC = out->channels;
    const size_t K = (size_t)out->kernel_h * out->kernel_w * in->channels;
    const uint64_t total = (uint64_t)n * out->height * out->width;
    const float *Wt = SPINGALETT_WEIGHT_MTX_PTR(net, l);
    BiasActivation epilogue = {net->biases + net->bias_offsets[l], act == ACT_SOFTMAX ? ACT_NONE : act};
    SpingalettGemmHooks hooks = {NULL, NULL, add_bias_activate, &epilogue};

    if (pointwise(out) || implicit(mode)) {
        bool gather = !pointwise(out);
        for (uint64_t p0 = 0; p0 < total; p0 += POINTWISE_ROWS) {
            uint64_t rows = total - p0 < POINTWISE_ROWS ? total - p0 : POINTWISE_ROWS;
            RowSource src = {x, in, out, p0, false};
            SpingalettGemmSource windows = {fill_windows, &src};
            hooks.a = gather ? &windows : NULL;
            conv_gemm(gemm, mode, false, true, (uint32_t)rows, OC, (uint32_t)K, 1.0f, gather ? NULL : x + p0 * K, K,
                      Wt, K, 0.0f, y + p0 * OC, OC, &hooks);
        }
        return;
    }
    uint64_t step = chunk_rows(total, K);
    for (uint64_t p0 = 0; p0 < total; p0 += step) {
        uint64_t rows = total - p0 < step ? total - p0 : step;
        gather_windows(x, in, out, p0, rows, scratch, mode);
        conv_gemm(gemm, mode, false, true, (uint32_t)rows, OC, (uint32_t)K, 1.0f, scratch, K, Wt, K, 0.0f,
                  y + p0 * OC, OC, &hooks);
    }
}

void spingalett_conv_backward_data(const NeuralNetwork *net, uint32_t l, const float *dy, float *dx, uint32_t n,
                                   const float *x, ActivationFunction act, float *scratch,
                                   SpingalettGemmScratch *gemm, ComputeMode mode) {
    const LayerShape *in = &net->shapes[l], *out = &net->shapes[l + 1];
    const uint32_t C = in->channels, OC = out->channels, KH = out->kernel_h, KW = out->kernel_w;
    const size_t K = (size_t)KH * KW * C;
    const uint64_t total = (uint64_t)n * in->height * in->width;
    const float *Wt = SPINGALETT_WEIGHT_MTX_PTR(net, l);
    Derivative epilogue = {x, 0, act};
    SpingalettGemmHooks hooks = {NULL, NULL, act == ACT_NONE ? NULL : multiply_derivative, &epilogue};

    if (pointwise(out)) {           /* dx = dy * W */
        for (uint64_t q0 = 0; q0 < total; q0 += POINTWISE_ROWS) {
            uint64_t rows = total - q0 < POINTWISE_ROWS ? total - q0 : POINTWISE_ROWS;
            epilogue.base = q0;
            conv_gemm(gemm, mode, false, false, (uint32_t)rows, C, OC, 1.0f, dy + q0 * OC, OC, Wt, K, 0.0f,
                      dx + q0 * C, C, &hooks);
        }
        return;
    }
    /* The weights regrouped per input channel, Wr[c][(kh, kw, oc)] = W[oc][(kh, kw, c)], at the
       start of the scratch buffer; gathered rows (OpenBLAS) use the rest. */
    const size_t Kr = (size_t)KH * KW * OC, wsize = (size_t)C * Kr;
    float *Wr = scratch;
    float *col = scratch + ((wsize + 15u) & ~(size_t)15u);
    for (uint32_t oc = 0; oc < OC; oc++)
        for (size_t s = 0; s < (size_t)KH * KW; s++)
            for (uint32_t c = 0; c < C; c++)
                Wr[(size_t)c * Kr + s * OC + oc] = Wt[(size_t)oc * K + s * C + c];

    if (implicit(mode)) {
        for (uint64_t q0 = 0; q0 < total; q0 += POINTWISE_ROWS) {
            uint64_t rows = total - q0 < POINTWISE_ROWS ? total - q0 : POINTWISE_ROWS;
            RowSource src = {dy, in, out, q0, false};
            SpingalettGemmSource grads = {fill_output_grads, &src};
            hooks.a = &grads;
            epilogue.base = q0;
            conv_gemm(gemm, mode, false, true, (uint32_t)rows, C, (uint32_t)Kr, 1.0f, NULL, Kr, Wr, Kr, 0.0f,
                      dx + q0 * C, C, &hooks);
        }
        return;
    }
    uint64_t step = chunk_rows(total, Kr);
    for (uint64_t q0 = 0; q0 < total; q0 += step) {
        uint64_t rows = total - q0 < step ? total - q0 : step;
        gather_output_grads(dy, in, out, q0, rows, col, mode);
        epilogue.base = q0;
        conv_gemm(gemm, mode, false, true, (uint32_t)rows, C, (uint32_t)Kr, 1.0f, col, Kr, Wr, Kr, 0.0f,
                  dx + q0 * C, C, &hooks);
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

    /* gW[oc x window] = scale * dy^T * windows (+ beta * gW). Windows narrower than a GEMM panel
       (a first layer over one channel has 9 weights per filter) are multiplied the other way
       round, T[window x oc] = windows^T * dy, which fills the panels, and T is transposed into gW.
       Gathered by the GEMM, the windows then end in a column of ones, whose row of T is the bias
       gradient: dy is read once. */
    if (K < 32 && OC > K && !pointwise(out)) {
        float *T = scratch, *col = scratch + ((K * OC + 15u) & ~(size_t)15u);
        if (implicit(mode)) {
            for (uint64_t p0 = 0; p0 < total; p0 += POINTWISE_ROWS) {
                uint64_t rows = total - p0 < POINTWISE_ROWS ? total - p0 : POINTWISE_ROWS;
                RowSource src = {x, in, out, p0, true};
                SpingalettGemmSource windows = {fill_windows, &src};
                SpingalettGemmHooks hooks = {&windows, NULL, NULL, NULL};
                conv_gemm(gemm, mode, true, false, (uint32_t)K + 1, OC, (uint32_t)rows, 1.0f, NULL, K + 1,
                          dy + p0 * OC, OC, p0 == 0 ? 0.0f : 1.0f, T, OC, &hooks);
            }
            for (uint32_t oc = 0; oc < OC; oc++) {
                for (size_t k = 0; k < K; k++)
                    gW[(size_t)oc * K + k] = scale * T[k * OC + oc] + (beta == 0.0f ? 0.0f : beta * gW[(size_t)oc * K + k]);
                gB[oc] = scale * T[K * OC + oc] + (beta == 0.0f ? 0.0f : beta * gB[oc]);
            }
            return;
        } else {
            uint64_t step = chunk_rows(total, K);
            for (uint64_t p0 = 0; p0 < total; p0 += step) {
                uint64_t rows = total - p0 < step ? total - p0 : step;
                gather_windows(x, in, out, p0, rows, col, mode);
                spingalett_gemm(gemm, mode, true, false, (uint32_t)K, OC, (uint32_t)rows, 1.0f, col, K, dy + p0 * OC, OC,
                                p0 == 0 ? 0.0f : 1.0f, T, OC);
            }
        }
        for (uint32_t oc = 0; oc < OC; oc++)
            for (size_t k = 0; k < K; k++)
                gW[(size_t)oc * K + k] = scale * T[k * OC + oc] + (beta == 0.0f ? 0.0f : beta * gW[(size_t)oc * K + k]);
    } else if (pointwise(out)) {
        for (uint64_t p0 = 0; p0 < total; p0 += POINTWISE_ROWS) {
            uint64_t rows = total - p0 < POINTWISE_ROWS ? total - p0 : POINTWISE_ROWS;
            spingalett_gemm(gemm, mode, true, false, OC, (uint32_t)K, (uint32_t)rows, scale, dy + p0 * OC, OC,
                            x + p0 * K, K, p0 == 0 ? beta : 1.0f, gW, K);
        }
    } else if (implicit(mode)) {
        for (uint64_t p0 = 0; p0 < total; p0 += POINTWISE_ROWS) {
            uint64_t rows = total - p0 < POINTWISE_ROWS ? total - p0 : POINTWISE_ROWS;
            RowSource src = {x, in, out, p0, false};
            SpingalettGemmSource windows = {fill_windows, &src};
            SpingalettGemmHooks hooks = {NULL, &windows, NULL, NULL};
            conv_gemm(gemm, mode, true, false, OC, (uint32_t)K, (uint32_t)rows, scale, dy + p0 * OC, OC, NULL, K,
                      p0 == 0 ? beta : 1.0f, gW, K, &hooks);
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
        for (p++; p < end; p++) {
            const float *row = dy + (size_t)p * OC;
            for (uint32_t oc = 0; oc < OC; oc++) ps[oc] += row[oc];
        }
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

/* Pooling works on channel blocks of POOL_BLOCK floats held in registers (the last block of a pixel
   may be shorter): the window's cells are visited in order and the first maximum wins. */
#define POOL_BLOCK 16u

static inline void pool_block_forward(const float *restrict xs, float *restrict o, uint32_t W, uint32_t C,
                                      uint32_t h0, uint32_t h1, uint32_t w0, uint32_t w1, uint32_t c0,
                                      uint32_t len, bool max) {
    float acc[POOL_BLOCK];
    const float *first = xs + ((size_t)h0 * W + w0) * C + c0;
    for (uint32_t c = 0; c < POOL_BLOCK; c++) acc[c] = c < len ? first[c] : 0.0f;
    for (uint32_t h = h0; h < h1; h++)
        for (uint32_t w = (h == h0 ? w0 + 1 : w0); w < w1; w++) {
            const float *v = xs + ((size_t)h * W + w) * C + c0;
            if (max) { for (uint32_t c = 0; c < POOL_BLOCK; c++) if (c < len && v[c] > acc[c]) acc[c] = v[c]; }
            else     { for (uint32_t c = 0; c < POOL_BLOCK; c++) if (c < len) acc[c] += v[c]; }
        }
    float inv = max ? 1.0f : 1.0f / (float)((h1 - h0) * (w1 - w0));
    for (uint32_t c = 0; c < POOL_BLOCK; c++) if (c < len) o[c] = acc[c] * inv;
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
                for (uint32_t c0 = 0; c0 < C; c0 += POOL_BLOCK)
                    pool_block_forward(xs, o + c0, W, C, h0, h1, w0, w1, c0,
                                       C - c0 < POOL_BLOCK ? C - c0 : POOL_BLOCK, max);
            }
    }
    (void)mode;
}

/* The gradient of one channel block of one window: max pooling routes g to the first maximum
   (recomputed from the input), average pooling spreads it evenly. With tiling windows each input
   cell is written once (no clearing needed); otherwise contributions are added. */
static inline void pool_block_backward(const float *restrict xs, float *restrict dxs, const float *restrict g,
                                       uint32_t W, uint32_t C, uint32_t h0, uint32_t h1, uint32_t w0, uint32_t w1,
                                       uint32_t c0, uint32_t len, bool max, bool tiles) {
    if (max) {
        float best[POOL_BLOCK];
        uint32_t at[POOL_BLOCK];
        const float *first = xs + ((size_t)h0 * W + w0) * C + c0;
        for (uint32_t c = 0; c < POOL_BLOCK; c++) { best[c] = c < len ? first[c] : 0.0f; at[c] = 0; }
        uint32_t cell = 0;
        for (uint32_t h = h0; h < h1; h++)
            for (uint32_t w = w0; w < w1; w++, cell++) {
                if (cell == 0) continue;
                const float *v = xs + ((size_t)h * W + w) * C + c0;
                for (uint32_t c = 0; c < POOL_BLOCK; c++) {
                    bool more = c < len && v[c] > best[c];
                    best[c] = more ? v[c] : best[c];
                    at[c] = more ? cell : at[c];
                }
            }
        cell = 0;
        for (uint32_t h = h0; h < h1; h++)
            for (uint32_t w = w0; w < w1; w++, cell++) {
                float *d = dxs + ((size_t)h * W + w) * C + c0;
                if (tiles) { for (uint32_t c = 0; c < POOL_BLOCK; c++) if (c < len) d[c] = at[c] == cell ? g[c] : 0.0f; }
                else       { for (uint32_t c = 0; c < POOL_BLOCK; c++) if (c < len && at[c] == cell) d[c] += g[c]; }
            }
    } else {
        float inv = 1.0f / (float)((h1 - h0) * (w1 - w0)), share[POOL_BLOCK];
        for (uint32_t c = 0; c < POOL_BLOCK; c++) share[c] = c < len ? g[c] * inv : 0.0f;
        for (uint32_t h = h0; h < h1; h++)
            for (uint32_t w = w0; w < w1; w++) {
                float *d = dxs + ((size_t)h * W + w) * C + c0;
                if (tiles) { for (uint32_t c = 0; c < POOL_BLOCK; c++) if (c < len) d[c] = share[c]; }
                else       { for (uint32_t c = 0; c < POOL_BLOCK; c++) if (c < len) d[c] += share[c]; }
            }
    }
}

void spingalett_pool_backward(const NeuralNetwork *net, uint32_t l, const float *x, const float *dy, float *dx,
                              uint32_t n, ActivationFunction act, ComputeMode mode) {
    const LayerShape *in = &net->shapes[l], *out = &net->shapes[l + 1];
    const uint32_t C = in->channels, W = in->width;
    const size_t in_size = net->topology[l], out_size = net->topology[l + 1];
    const bool max = out->type == LAYER_MAX_POOL2D;
    /* windows that tile the image exactly (2 x 2 with stride 2 over an even size, say) write
       every input cell once: dx is written directly, without clearing it first */
    const bool tiles = out->stride_h == out->kernel_h && out->stride_w == out->kernel_w && out->pad_h == 0 &&
                       out->pad_w == 0 && out->height * out->kernel_h == in->height && out->width * out->kernel_w == W;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if(use_omp(mode, (uint64_t)n * out_size * out->kernel_h * out->kernel_w))
#endif
    for (int64_t s = 0; s < (int64_t)n; s++) {
        const float *xs = x + (size_t)s * in_size, *ds = dy + (size_t)s * out_size;
        float *dxs = dx + (size_t)s * in_size;
        if (!tiles) memset(dxs, 0, in_size * sizeof(float));
        for (uint32_t oh = 0; oh < out->height; oh++)
            for (uint32_t ow = 0; ow < out->width; ow++) {
                const float *g = ds + ((size_t)oh * out->width + ow) * C;
                uint32_t h0, h1, w0, w1;
                window_bounds(in, out, oh, ow, &h0, &h1, &w0, &w1);
                for (uint32_t c0 = 0; c0 < C; c0 += POOL_BLOCK)
                    pool_block_backward(xs, dxs, g + c0, W, C, h0, h1, w0, w1, c0,
                                        C - c0 < POOL_BLOCK ? C - c0 : POOL_BLOCK, max, tiles);
            }
        apply_derivative_batch(dxs, xs, in_size, act);     /* while the sample is in cache */
    }
    (void)mode;
}
