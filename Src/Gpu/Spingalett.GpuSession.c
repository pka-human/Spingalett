/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/*
 * Sessions of language models on the GPU, straight from the model's image (spingalett_session_feed() in a GPU compute
 * mode, Spingalett.Session.c): its layers as the session runs them on the CPU, the weights where the image keeps them
 * in the device's memory as they are stored (GGUF's block types as blocks, read by bquant and bdot with the CPU's
 * bits; tables of embeddings as the image has them, rows decoded as a token asks for them; floats, float16 and
 * bfloat16 weights of linear layers as floats for the matrix product), the keys and values of every attention layer
 * in caches of every sequence. A step runs rows of tokens through every layer: an embedding (bembed), linear layers
 * (bquant and bdot, or the matrix product with the bias and activation in its epilogue), RMS and layer normalization
 * (grow), additions and products of layers and activations (gew), attention (kvattn: the queries and keys normalized
 * and rotated, the keys and values into the caches, then each query over its sequence's keys); the last layer on the
 * rows whose logits are asked for alone (their inputs gathered first). The commands of a step are recorded once for
 * each number of rows and of logits (the tokens, positions and rows asked for in buffers the host writes) and run
 * again. The logits are the CPU session's to rounding: the block products give its bits, the rest rounds otherwise.
 */

#include "Spingalett.Gpu.h"
#include "Spingalett.GpuKernels.h"
#include "Spingalett.Model.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define KEPT 4u                     /* command lists kept, by their rows and logits */

enum { RUN_EMBED, RUN_BLOCK, RUN_FLOAT, RUN_RMS, RUN_NORM, RUN_ADD, RUN_MUL, RUN_CONCAT, RUN_ATTN };

typedef struct {
    SlettLayer L;
    uint8_t run;
    uint32_t type;                  /* RUN_BLOCK: SPG_B_*; RUN_EMBED: SPG_EMB_* */
    uint32_t stride;                /* bytes a row of weights (RUN_BLOCK, RUN_EMBED) */
    bool packed;                    /* RUN_BLOCK: the weights packed (bmmp, bdotp: image_packed()) */
    uint64_t w, b, g;               /* weights (floats N x K for RUN_FLOAT), biases (or the positions' table), gamma */
    uint32_t inputs[SPINGALETT_MAX_INPUTS];     /* the layers it reads (1 + their index) */
    uint32_t channels[SPINGALETT_MAX_INPUTS];   /* their channels (RUN_CONCAT) */
    uint32_t act_gew;               /* its activation as gew's operator, NONE for none (in the product's epilogue) */
    float alpha;
    uint64_t out;                   /* its output's address */
    uint64_t gate;                  /* RUN_BLOCK: the values its outputs are multiplied by (the product of layers after it,
                                       taken in), or 0 */
    bool skip;                      /* a product of layers its reader's epilogue computes */
    uint32_t pair;                  /* RUN_BLOCK: the other layer of a gated unit whose product this one's computes too
                                       (1 + its index; find_pairs()), or 0 */
    bool paired;                    /* RUN_BLOCK: computed by a later layer's pair */
    uint32_t join;                  /* RUN_BLOCK: the layers after it in its concatenation whose rows its product computes
                                       too (find_joins()), or 0 */
    uint32_t joint;                 /* the rows of that product */
    bool joined;                    /* RUN_BLOCK: computed by another layer's join */
    uint32_t pair_act;              /* the pair's activation (its first rows') */
    uint32_t into;                  /* RUN_BLOCK: the concatenation it writes its columns of (1 + its index), or 0 */
    uint32_t ldy;                   /* RUN_BLOCK: floats between its output's rows (0: its channels) */
    bool gathered;                  /* RUN_CONCAT: its inputs write their columns of it themselves */
    bool folded;                    /* a sum of two layers the RMS normalization after it computes (bnorm) */
    uint32_t tpr, qbv;              /* RMS normalizations by bnorm: grow's threads a row; the blocks of values its
                                       outputs are quantized in for the block layer after it (0: none) */
    bool unwritten;                 /* and its outputs quantized alone (its readers block layers that take them so) */
    /* attention */
    uint64_t keys, values, table, gq, gk;
    uint64_t strides;               /* gflash's strides over the caches (a prompt's attention), or 0 */
    bool half_kv;                   /* the caches of bfloat16 (gflash's: it rounds them so as it reads them) */
    uint32_t heads, kv, head, rot;
    float eps, scale;
} GLayer;

typedef struct {
    uint32_t rows, picked;
    uint64_t runs;                  /* the runs of rows it was recorded for (a hash of their lengths) */
    SpgGpuCommands *c;
} Kept;

struct SpgGpuSession {
    const uint8_t *image;
    uint32_t layers, sequences, max_length, most, vocabulary;
    uint32_t most_logits;           /* rows whose logits a step writes, at most */
    GLayer *L;
    SpgGpuBuffer *buffers;          /* everything allocated, freed together */
    uint32_t buffer_count, buffer_cap;
    uint64_t *uploaded;             /* (image offset, address) pairs: weights uploaded once (tied tables shared) */
    uint32_t uploaded_count, uploaded_cap;
    uint64_t acts, q, xq, dx, xs, gathered, logits, tokens, info, pick;
    SpgGpuBuffer staging;           /* host-visible: tokens, info, picks, then the logits */
    size_t info_at, pick_at, logits_at;
    uint32_t kmax;                  /* inputs of the widest block layer */
    uint32_t run_count, *runs;      /* recording: the step's runs of rows (each of one sequence at consecutive
                                       positions): their first rows, and one past the last */
    uint64_t quantized;             /* recording: the input the quantized rows hold (0: none), in blocks of quantized_bv
                                       (in tiles: quantized_tiled) */
    uint32_t quantized_bv;
    bool quantized_tiled;
    Kept kept[KEPT];
    uint32_t next;
};

#define NONE 0xFFFFFFFFu

static uint64_t buffer_new(SpgGpuSession *s, size_t bytes, bool host) {
    if (s->buffer_count == s->buffer_cap) {
        const uint32_t cap = s->buffer_cap ? 2u * s->buffer_cap : 64u;
        SpgGpuBuffer *grown = (SpgGpuBuffer *)realloc(s->buffers, cap * sizeof(SpgGpuBuffer));
        if (!grown) return 0;
        s->buffers = grown;
        s->buffer_cap = cap;
    }
    SpgGpuBuffer *b = &s->buffers[s->buffer_count];
    if (!spg_gpu_buffer_create(b, bytes ? bytes : 16u, host)) return 0;
    s->buffer_count++;
    return b->address;
}

/* bytes from the host into a new buffer (through a staging buffer, waited for), 16 more after them that kernels may
   read (bmm reads rows of weights in whole words): its address, 0 on failure */
static uint64_t upload(SpgGpuSession *s, const void *data, size_t bytes) {
    const uint64_t at = buffer_new(s, bytes + 16u, false);
    if (!at) return 0;
    const SpgGpuBuffer *dst = &s->buffers[s->buffer_count - 1u];
    SpgGpuBuffer stage;
    if (!spg_gpu_buffer_create(&stage, bytes ? bytes : 16u, true)) return 0;
    memcpy(stage.mapped, data, bytes);
    SpgGpuCommands *c = spg_gpu_commands_create();
    bool ok = c && spg_gpu_record_begin(c);
    if (ok) spg_gpu_copy(c, &stage, 0, dst, 0, bytes ? bytes : 16u);
    ok = ok && spg_gpu_record_end(c) && spg_gpu_submit(c) && spg_gpu_wait(c);
    if (c) spg_gpu_commands_free(c);
    spg_gpu_buffer_free(&stage);
    return ok ? at : 0;
}

/* The image's bytes at offset uploaded once (layers tied to an embedding read its table's) */
static uint64_t image_bytes(SpgGpuSession *s, uint64_t offset, size_t bytes) {
    for (uint32_t k = 0; k < s->uploaded_count; k++)
        if (s->uploaded[2u * k] == offset) return s->uploaded[2u * k + 1u];
    if (s->uploaded_count == s->uploaded_cap) {
        const uint32_t cap = s->uploaded_cap ? 2u * s->uploaded_cap : 64u;
        uint64_t *grown = (uint64_t *)realloc(s->uploaded, 2u * cap * sizeof(uint64_t));
        if (!grown) return 0;
        s->uploaded = grown;
        s->uploaded_cap = cap;
    }
    const uint64_t at = upload(s, s->image + offset, bytes);
    if (!at) return 0;
    s->uploaded[2u * s->uploaded_count] = offset;
    s->uploaded[2u * s->uploaded_count++ + 1u] = at;
    return at;
}

/* The image's rows of N x K values of block type `type` at offset packed (Spingalett.GpuPack.c: the matrix units and
   the loads take whole words and sectors), uploaded once (tied tables shared) */
#define PACKED (1ull << 63)         /* (uploaded[]'s offsets of packed copies) */
static uint64_t image_packed(SpgGpuSession *s, uint64_t offset, uint32_t type, uint32_t N, uint32_t K) {
    for (uint32_t k = 0; k < s->uploaded_count; k++)
        if (s->uploaded[2u * k] == (offset | PACKED)) return s->uploaded[2u * k + 1u];
    if (s->uploaded_count == s->uploaded_cap) {
        const uint32_t cap = s->uploaded_cap ? 2u * s->uploaded_cap : 64u;
        uint64_t *grown = (uint64_t *)realloc(s->uploaded, 2u * cap * sizeof(uint64_t));
        if (!grown) return 0;
        s->uploaded = grown;
        s->uploaded_cap = cap;
    }
    const size_t bytes = spg_packed_bytes(type, N, K);
    uint8_t *packed = (uint8_t *)malloc(bytes);
    if (!packed) return 0;
    spg_pack_blocks(type, s->image + offset, N, K, packed);
    const uint64_t at = upload(s, packed, bytes);
    free(packed);
    if (!at) return 0;
    s->uploaded[2u * s->uploaded_count] = offset | PACKED;
    s->uploaded[2u * s->uploaded_count++ + 1u] = at;
    return at;
}

/* An activation as gew's operator (alpha its slope), NONE for none, false for one gew has not */
static bool gew_act(ActivationFunction act, uint32_t *op, float *alpha) {
    *alpha = 0.0f;
    switch (act) {
        case ACT_NONE: *op = NONE; return true;
        case ACT_RELU: *op = SPG_GEW_RELU; return true;
        case ACT_LEAKY_RELU: *op = SPG_GEW_LEAKY; *alpha = 0.01f; return true;
        case ACT_SIGMOID: *op = SPG_GEW_SIGMOID; return true;
        case ACT_TANH: *op = SPG_GEW_TANH; return true;
        case ACT_GELU: *op = SPG_GEW_GELU; return true;
        case ACT_GELU_TANH: *op = SPG_GEW_GELU_TANH; return true;
        case ACT_SILU: *op = SPG_GEW_SILU; *alpha = 1.0f; return true;
        default: return false;
    }
}

/* The products' epilogues: the activations they take */
static bool epilogue_act(ActivationFunction act) {
    return act == ACT_NONE || act == ACT_RELU || act == ACT_LEAKY_RELU || act == ACT_SIGMOID || act == ACT_TANH ||
           act == ACT_GELU || act == ACT_GELU_TANH || act == ACT_SILU;
}

void spingalett_gpu_session_free(SpgGpuSession *s) {
    if (!s) return;
    for (uint32_t k = 0; k < KEPT; k++)
        if (s->kept[k].c) spg_gpu_commands_free(s->kept[k].c);
    for (uint32_t k = 0; k < s->buffer_count; k++) spg_gpu_buffer_free(&s->buffers[k]);
    spg_gpu_buffer_free(&s->staging);
    free(s->buffers);
    free(s->uploaded);
    free(s->L);
    free(s->runs);
    free(s);
}

/* A layer of block precision a gated unit's pair may take: a linear layer whose weights the session packs */
static bool pairable(const SlettLayer *L) {
    if (L->type != LAYER_DENSE && (L->type != LAYER_CONV2D || !slett_conv_pointwise(L) || L->groups != 1)) return false;
    return spingalett_precision_is_block(L->precision) &&
           spg_packed_bytes(L->precision - PRECISION_Q8_0, L->rows, L->row_len) != 0 && epilogue_act(L->activation);
}

/* Gated units as one product (bmm's and bdot's pairs, block_out()): a product of two layers (layout()'s: its
   activation none, not the last layer, its second input the layer just before it, read by it alone) of two linear
   layers of packed blocks of one input, precision and shape, both read by it alone, one of them at most with an
   activation. The later computes both (pair), the other runs nowhere (paired): its product's output never written,
   nor the two outputs' product. Qwen3-0.6B's prompt on an RTX 4050: the up's 6 MB of outputs a layer neither written
   nor read again. */
static void find_pairs(SpgGpuSession *s) {
    const uint32_t L = s->layers;
    uint32_t *readers = (uint32_t *)calloc(L + 1u, sizeof(uint32_t));
    if (!readers) return;
    for (uint32_t i = 0; i < L; i++)
        for (uint32_t k = 0; k < s->L[i].L.input_count; k++) readers[s->L[i].inputs[k]]++;
    for (uint32_t i = 1; i + 1u < L; i++) {
        GLayer *M = &s->L[i], *G = &s->L[i - 1u];
        if (M->L.type != LAYER_MULTIPLY || M->L.input_count != 2 || M->L.activation != ACT_NONE || readers[i] != 1u)
            continue;
        const uint32_t other = M->inputs[0] == i ? M->inputs[1] : M->inputs[1] == i ? M->inputs[0] : 0;
        if (!other || other == i || readers[other] != 1u) continue;
        GLayer *U = &s->L[other - 1u];
        const SlettLayer *a = &G->L, *b = &U->L;
        if (!pairable(a) || !pairable(b) || a->precision != b->precision || a->row_len != b->row_len ||
            a->rows != b->rows || a->in_c != b->in_c || a->out_c != b->out_c || G->inputs[0] != U->inputs[0] ||
            (a->activation != ACT_NONE && b->activation != ACT_NONE) || U->pair || U->paired)
            continue;
        G->pair = other;
        U->paired = true;
    }
    free(readers);
}

/* Two layers of block precision one product may compute: linear layers of packed blocks of one input, precision, row
   length and activation */
static bool joinable(const GLayer *a, const GLayer *b) {
    return pairable(&a->L) && pairable(&b->L) && a->L.precision == b->L.precision && a->L.row_len == b->L.row_len &&
           a->L.in_c == b->L.in_c && a->inputs[0] == b->inputs[0] && a->L.activation == b->L.activation;
}

/* A concatenation's block layers as one product (attention's queries, keys and values): where every input of a
   concatenation without activation is a packed block layer read by it alone (layout() then gathers it), a run of
   inputs joinable() with the first is computed by the first (join, the count of the others: their rows after its own),
   the others run nowhere (joined). Qwen3-4B's step of a token on an RTX 4050: the keys' and values' products of
   1024 rows each, at 62% of the memory's bandwidth alone, run in the queries'. */
static void find_joins(SpgGpuSession *s) {
    const uint32_t L = s->layers;
    uint32_t *readers = (uint32_t *)calloc(L + 1u, sizeof(uint32_t));
    if (!readers) return;
    for (uint32_t i = 0; i < L; i++)
        for (uint32_t k = 0; k < s->L[i].L.input_count; k++) readers[s->L[i].inputs[k]]++;
    for (uint32_t i = 1; i + 1u < L; i++) {
        const GLayer *C = &s->L[i];
        const uint32_t n = C->L.input_count;
        bool all = C->L.type == LAYER_CONCAT && C->L.activation == ACT_NONE;
        for (uint32_t k = 0; all && k < n; k++) {
            const uint32_t j = C->inputs[k];
            all = j && j <= i && readers[j] == 1u && pairable(&s->L[j - 1u].L) && !s->L[j - 1u].pair &&
                  !s->L[j - 1u].paired;
            for (uint32_t m = 0; all && m < k; m++) all = C->inputs[m] != j;
        }
        for (uint32_t k = 0; all && k < n;) {
            GLayer *G = &s->L[C->inputs[k] - 1u];
            uint32_t m = k + 1u;
            G->joint = G->L.rows;
            for (; m < n && joinable(G, &s->L[C->inputs[m] - 1u]); m++) {
                s->L[C->inputs[m] - 1u].joined = true;
                G->joint += s->L[C->inputs[m] - 1u].L.rows;
            }
            G->join = m - k - 1u;
            k = m;
        }
    }
    free(readers);
}

/* A join's layers as one (find_joins(), G the first): their packed rows one after the other, and so their biases;
   false when memory runs out */
static bool place_join(SpgGpuSession *s, uint32_t i) {
    GLayer *G = &s->L[i];
    const GLayer *C = NULL;
    uint32_t k0 = 0;
    for (uint32_t c = i + 1u; !C && c < s->layers; c++)
        for (uint32_t k = 0; k < s->L[c].L.input_count; k++)
            if (s->L[c].inputs[k] == i + 1u) C = &s->L[c], k0 = k;
    const uint32_t K = G->L.row_len, N = G->joint, type = G->type, row = SPG_PACK_ROW(type, K);
    uint8_t *w = (uint8_t *)malloc((size_t)N * row);
    float *b = (float *)malloc((size_t)N * sizeof(float));
    bool ok = C && w && b;
    for (uint32_t m = 0, at = 0; ok && m <= G->join; m++) {
        const SlettLayer *A = &s->L[C->inputs[k0 + m] - 1u].L;
        spg_pack_blocks(type, s->image + A->weights, A->rows, K, w + (size_t)at * row);
        memcpy(b + at, s->image + A->biases, (size_t)A->rows * sizeof(float));
        at += A->rows;
    }
    G->packed = true;
    G->stride = row;
    G->w = ok ? upload(s, w, (size_t)N * row) : 0;
    G->b = ok && G->w ? upload(s, b, (size_t)N * sizeof(float)) : 0;
    if (K > s->kmax) s->kmax = K;
    free(w);
    free(b);
    return G->w && G->b;
}

/* A gated unit's two block layers as one (G the later, U the other; find_pairs()): their packed rows interleaved, the
   layer with the activation first (row 2j, then the other's row j at 2j + 1), and so their biases; false when memory
   runs out */
static bool place_pair(SpgGpuSession *s, GLayer *G, const GLayer *U) {
    const uint32_t K = G->L.row_len, N = G->L.rows, type = G->type, row = SPG_PACK_ROW(type, K);
    const bool first = G->L.activation != ACT_NONE || U->L.activation == ACT_NONE;
    const SlettLayer *A = first ? &G->L : &U->L, *B = first ? &U->L : &G->L;
    const uint64_t raw = spingalett_slett_row_bytes(G->L.precision, K);
    uint8_t *w = (uint8_t *)malloc((size_t)2u * N * row);
    float *b = (float *)malloc((size_t)2u * N * sizeof(float));
    bool ok = w && b;
    for (uint32_t j = 0; ok && j < N; j++) {
        spg_pack_blocks(type, s->image + A->weights + j * raw, 1u, K, w + (size_t)2u * j * row);
        spg_pack_blocks(type, s->image + B->weights + j * raw, 1u, K, w + (size_t)(2u * j + 1u) * row);
        memcpy(&b[2u * j], s->image + A->biases + (size_t)j * sizeof(float), sizeof(float));
        memcpy(&b[2u * j + 1u], s->image + B->biases + (size_t)j * sizeof(float), sizeof(float));
    }
    G->packed = true;
    G->stride = row;
    G->pair_act = (uint32_t)A->activation;
    G->w = ok ? upload(s, w, (size_t)2u * N * row) : 0;
    G->b = ok && G->w ? upload(s, b, (size_t)2u * N * sizeof(float)) : 0;
    if (K > s->kmax) s->kmax = K;
    free(w);
    free(b);
    return G->w && G->b;
}

/* Layer i's plan and weights on the device; false with *why for a layer the GPU's session does not run. */
static bool place(SpgGpuSession *s, uint32_t i, const char **why) {
    GLayer *G = &s->L[i];
    const SlettLayer *L = &G->L;
    const uint8_t *image = s->image;
    for (uint32_t k = 0; k < L->input_count; k++) G->inputs[k] = spingalett_slett_input(image, L, k);
    G->act_gew = NONE;
    if (!gew_act(L->activation, &G->act_gew, &G->alpha)) return *why = "an activation the GPU's sessions have not", false;
    switch (L->type) {
        case LAYER_EMBEDDING: {
            if (i != 0) return *why = "an embedding past the first layer", false;
            G->run = RUN_EMBED;
            if (L->precision == PRECISION_FLOAT32) G->type = SPG_EMB_F32;
            else if (L->precision == PRECISION_FP16) G->type = SPG_EMB_F16;
            else if (L->precision == PRECISION_BFLOAT16) G->type = SPG_EMB_BF16;
            else if (spingalett_precision_is_block(L->precision)) G->type = SPG_EMB_BLOCK + (L->precision - PRECISION_Q8_0);
            else return *why = "an embedding of integer weights", false;
            G->stride = (uint32_t)spingalett_slett_row_bytes(L->precision, L->row_len);
            const uint32_t t = L->precision - PRECISION_Q8_0;
            if (spingalett_precision_is_block(L->precision) && spg_packed_bytes(t, L->rows, L->row_len)) {
                /* (packed: a table tied to the output layer is that layer's weights) */
                G->type = SPG_EMB_PACKED + t;
                G->stride = SPG_PACK_ROW(t, L->row_len);
                G->w = image_packed(s, L->weights, t, L->rows, L->row_len);
            } else {
                G->w = image_bytes(s, L->weights, (size_t)G->stride * L->rows);
            }
            if (!G->w) return *why = "out of device memory", false;
            if (L->positions) {
                G->b = image_bytes(s, L->biases, (size_t)L->out_h * L->out_w * L->out_c * sizeof(float));
                if (!G->b) return *why = "out of device memory", false;
            }
            return true;
        }
        case LAYER_DENSE:
        case LAYER_CONV2D: {
            if (L->type == LAYER_CONV2D && (!slett_conv_pointwise(L) || L->groups != 1))
                return *why = "a convolution that mixes positions", false;
            const uint32_t K = L->row_len, N = L->rows;
            const bool whole = !G->pair && !G->paired && !G->join && !G->joined;   /* (its own weights and biases) */
            if (whole) G->b = image_bytes(s, L->biases, (size_t)N * sizeof(float));
            if (!G->b && whole) return *why = "out of device memory", false;
            if (spingalett_precision_is_block(L->precision)) {
                G->run = RUN_BLOCK;
                G->act_gew = NONE;          /* (bdot's epilogue applies it) */
                G->type = L->precision - PRECISION_Q8_0;
                G->stride = (uint32_t)spingalett_slett_row_bytes(L->precision, K);
                if (G->paired || G->joined) return true;
                if (G->pair) return place_pair(s, G, &s->L[G->pair - 1u]) || (*why = "out of device memory", false);
                if (G->join) return place_join(s, i) || (*why = "out of device memory", false);
                G->packed = spg_packed_bytes(G->type, N, K) != 0;
                G->w = G->packed ? image_packed(s, L->weights, G->type, N, K) : image_bytes(s, L->weights, (size_t)G->stride * N);
                if (G->packed) G->stride = SPG_PACK_ROW(G->type, K);
                if (K > s->kmax) s->kmax = K;
                return G->w || (*why = "out of device memory", false);
            }
            if (L->precision != PRECISION_FLOAT32 && L->precision != PRECISION_FP16 && L->precision != PRECISION_BFLOAT16)
                return *why = "a linear layer of integer weights", false;
            /* the weights as floats, N x K: the matrix product's B as its forward pass reads it */
            G->run = RUN_FLOAT;
            float *w = (float *)malloc((size_t)N * K * sizeof(float));
            if (!w) return *why = "out of memory", false;
            const uint8_t *src = image + L->weights;
            const uint64_t row = spingalett_slett_row_bytes(L->precision, K);
            for (uint32_t j = 0; j < N; j++)
                for (uint32_t k = 0; k < K; k++) {
                    const uint8_t *e = src + j * row;
                    w[(size_t)j * K + k] = L->precision == PRECISION_FLOAT32 ? ((const float *)(const void *)e)[k]
                                         : L->precision == PRECISION_FP16 ? spingalett_fp16_to_float(slett_get16(e + 2u * k))
                                         : spingalett_bf16_to_float(slett_get16(e + 2u * k));
                }
            G->w = upload(s, w, (size_t)N * K * sizeof(float));
            free(w);
            if (epilogue_act(L->activation)) G->act_gew = NONE;
            return G->w || (*why = "out of device memory", false);
        }
        case LAYER_RMS_NORM:
        case LAYER_LAYER_NORM:
            G->run = L->type == LAYER_RMS_NORM ? RUN_RMS : RUN_NORM;
            G->g = image_bytes(s, L->weights, (size_t)L->out_c * sizeof(float));
            if (L->type == LAYER_LAYER_NORM) G->b = image_bytes(s, L->biases, (size_t)L->out_c * sizeof(float));
            return (G->g && (G->run == RUN_RMS || G->b)) || (*why = "out of device memory", false);
        case LAYER_ADD:
        case LAYER_MULTIPLY:
            G->run = L->type == LAYER_ADD ? RUN_ADD : RUN_MUL;
            for (uint32_t k = 0; k < L->input_count; k++) {
                uint32_t h, w, units;
                spingalett_slett_output_shape(image, G->inputs[k], &h, &w, &units);
                if (units / (h * w) != L->out_c) return *why = "a sum or product of layers that broadcasts", false;
            }
            return true;
        case LAYER_CONCAT:
            G->run = RUN_CONCAT;
            for (uint32_t k = 0; k < L->input_count; k++) {
                uint32_t h, w, units;
                spingalett_slett_output_shape(image, G->inputs[k], &h, &w, &units);
                G->channels[k] = units / (h * w);
            }
            return true;
        case LAYER_ATTENTION: {
            if (!L->causal || L->input_count != 1) return *why = "attention that is not causal self-attention", false;
            SlettAttention a;
            spingalett_engine_attention_of(image, L, NULL, &a);
            G->run = RUN_ATTN;
            G->heads = a.heads, G->kv = a.kv, G->head = a.head, G->rot = a.rot, G->eps = a.eps, G->scale = a.scale;
            if (G->head > 256u || !G->kv || G->heads % G->kv) return *why = "attention heads of over 256 values", false;
            /* (the queries over the caches on the matrix units, gflash, where it has the head's unit: its caches of
               bfloat16, which it reads by asynchronous copies) */
            G->half_kv = (G->head == 32u || G->head == 64u || G->head == 128u) && spg_gpu_mma_bf16() &&
                         spg_gpu_kernel_built(SPG_KERNEL_gflash);
            const size_t floats = (size_t)s->sequences * G->kv * s->max_length * G->head;
            G->keys = buffer_new(s, floats * (G->half_kv ? 2u : sizeof(float)), false);
            G->values = buffer_new(s, floats * (G->half_kv ? 2u : sizeof(float)), false);
            if (!G->keys || !G->values) return *why = "out of device memory for the keys and values", false;
            if (a.gq) {
                G->gq = image_bytes(s, L->weights, 2u * G->head * sizeof(float));
                G->gk = G->gq + G->head * sizeof(float);
                if (!G->gq) return *why = "out of device memory", false;
            }
            if (L->theta > 0.0f) {
                float *table = (float *)malloc((size_t)s->max_length * G->head * sizeof(float));
                if (!table) return *why = "out of memory", false;
                spingalett_engine_rope_table(s->max_length, G->rot, L->theta,
                                             L->freqs ? (const float *)(const void *)(image + L->scales) : NULL, table);
                G->table = upload(s, table, (size_t)s->max_length * (G->rot / 2u) * 2u * sizeof(float));
                free(table);
                if (!G->table) return *why = "out of device memory", false;
            } else {
                G->rot = 0;
            }
            if (G->half_kv) {
                /* gflash's strides */
                const uint32_t D = G->head, H = G->heads, ML = s->max_length;
                const uint32_t st[21] = {0, D, H * D, G->kv * ML * D, ML * D, D, G->kv * ML * D, ML * D, D, 0, D, H * D,
                                         0, 0, 0, 0, 0, 0, 0, 0, 1};
                G->strides = upload(s, st, sizeof st);
            }
            return true;
        }
        default:
            return *why = "a layer the GPU's sessions have not", false;
    }
}

/* The layers' outputs in one buffer, each `most` rows of its channels from an offset of a multiple of 256 bytes (the
   kernels' vector loads), first fit, its room free again after its last reader; a product of two layers whose
   second is the block layer just before it, read by it alone, in that layer's epilogue (a gated unit: up times the
   gate's activation), written where the product's output goes; a concatenation of block layers read by it alone
   written by them, each its columns (rows ldy apart: queries, keys and values of types apart, with no copies). */
static bool layout(SpgGpuSession *s) {
    const uint32_t L = s->layers;
    uint32_t *readers = (uint32_t *)calloc(L + 1u, sizeof(uint32_t)), *last = (uint32_t *)calloc(L + 1u, sizeof(uint32_t));
    uint64_t *at = (uint64_t *)calloc(L, sizeof(uint64_t)), *bytes = (uint64_t *)calloc(L, sizeof(uint64_t));
    uint32_t *from = (uint32_t *)calloc(L, sizeof(uint32_t));
    bool ok = readers && last && at && bytes && from;
    for (uint32_t i = 0; ok && i < L; i++)
        for (uint32_t k = 0; k < s->L[i].L.input_count; k++) {
            readers[s->L[i].inputs[k]]++;
            last[s->L[i].inputs[k]] = i;
        }
    for (uint32_t i = 1; ok && i + 1u < L; i++) {
        GLayer *G = &s->L[i], *P = &s->L[i - 1u];
        if (G->run != RUN_MUL || G->L.input_count != 2 || G->act_gew != NONE) continue;
        for (uint32_t k = 0; k < 2; k++)
            if (G->inputs[k] == i && G->inputs[1u - k] != i && G->inputs[1u - k] && P->run == RUN_BLOCK && readers[i] == 1u) {
                G->skip = true;
                break;
            }
    }
    for (uint32_t i = 1; ok && i + 1u < L; i++) {
        GLayer *G = &s->L[i];
        bool all = G->run == RUN_CONCAT && G->act_gew == NONE;
        for (uint32_t k = 0; all && k < G->L.input_count; k++) {
            const uint32_t j = G->inputs[k];
            all = j && j <= i && s->L[j - 1u].run == RUN_BLOCK && !s->L[j - 1u].paired && !s->L[j - 1u].skip &&
                  readers[j] == 1u && !s->L[j - 1u].into;
            for (uint32_t m = 0; all && m < k; m++) all = G->inputs[m] != j;
            if (all && j < L && s->L[j].skip) all = false;  /* (a product of layers writes where its block layer would) */
        }
        if (!all) continue;
        G->gathered = true;
        for (uint32_t k = 0; k < G->L.input_count; k++) s->L[G->inputs[k] - 1u].into = i + 1u;
    }
    /* the rooms: of each layer's output but the last's and those a fused product writes instead, from the layer that
       writes it (from[]) to the last that reads it, in the order they are taken */
    uint64_t top = 0;
    for (uint32_t i = 0; ok && i + 1u < L; i++) {
        if (i + 2u < L && s->L[i + 1u].skip) continue;     /* (its product's room instead) */
        if (s->L[i].paired || s->L[i].into) continue;       /* (never written; its concatenation's room instead) */
        from[i] = s->L[i].skip ? i - 1u : i;
        if (s->L[i].gathered)                               /* (from its first input on) */
            for (uint32_t k = 0; k < s->L[i].L.input_count; k++)
                if (s->L[i].inputs[k] - 1u < from[i]) from[i] = s->L[i].inputs[k] - 1u;
        bytes[i] = ((uint64_t)s->most * s->L[i].L.out_c * sizeof(float) + 255u) & ~(uint64_t)255u;
        uint64_t place_at = 0;
        for (bool moved = true; moved;) {
            moved = false;
            for (uint32_t j = 0; j < i; j++)
                if (bytes[j] && last[j + 1u] >= from[i] && place_at < at[j] + bytes[j] && at[j] < place_at + bytes[i]) {
                    place_at = at[j] + bytes[j];
                    moved = true;
                }
        }
        at[i] = place_at;
        if (place_at + bytes[i] > top) top = place_at + bytes[i];
    }
    s->acts = ok ? buffer_new(s, top ? top : 256u, false) : 0;
    ok = ok && s->acts;
    for (uint32_t i = 0; ok && i + 1u < L; i++) s->L[i].out = s->acts + at[i];
    for (uint32_t i = 1; ok && i + 1u < L; i++)
        if (s->L[i].skip) {
            GLayer *G = &s->L[i], *P = &s->L[i - 1u];
            const uint32_t other = G->inputs[0] == i ? G->inputs[1] : G->inputs[0];
            P->gate = P->pair ? 0 : s->L[other - 1u].out;
            P->out = G->out;
        }
    for (uint32_t i = 1; ok && i + 1u < L; i++)
        if (s->L[i].gathered)
            for (uint32_t k = 0, at = 0; k < s->L[i].L.input_count; at += s->L[i].channels[k], k++) {
                GLayer *P = &s->L[s->L[i].inputs[k] - 1u];
                P->out = s->L[i].out + (uint64_t)at * sizeof(float);
                P->ldy = s->L[i].L.out_c;
            }
    free(readers);
    free(last);
    free(at);
    free(bytes);
    free(from);
    return ok;
}

static bool reads(const GLayer *G, uint32_t from) {
    for (uint32_t k = 0; k < G->L.input_count; k++)
        if (G->inputs[k] == from) return true;
    return false;
}

/* RMS normalizations in one pass with what is around them (bnorm): the sum of two layers just before one, read by it
   (and by others: it writes the sum), and the quantization of its outputs for a block layer just after it that reads
   them (which then reads them quantized: s->quantized). Where grow takes 32 threads a row or more, and 32 values a
   thread at most. Outputs read by block layers alone, of its blocks, the layers just after it (none between to quantize
   other rows), not the last layer (which reads its rows gathered), are not written at all: Qwen3-4B's prompt of 512 on
   an RTX 4050, 5 MB a normalization. */
static void fuse_norms(SpgGpuSession *s) {
    if (!spg_gpu_kernel_built(SPG_KERNEL_bnorm)) return;
    for (uint32_t i = 0; i < s->layers; i++) {
        GLayer *G = &s->L[i];
        const uint32_t k = G->L.out_c;
        uint32_t tpr = 1;
        while (tpr < 256u && 8u * tpr <= k) tpr *= 2u;
        if (G->run != RUN_RMS || tpr < 32u || k % tpr || k / tpr > 32u || G->act_gew != NONE) continue;
        G->tpr = tpr;
        GLayer *A = i ? &s->L[i - 1u] : NULL;
        if (A && A->run == RUN_ADD && A->L.input_count == 2 && A->act_gew == NONE && G->inputs[0] == i && A->inputs[0] &&
            A->inputs[1] && A->L.out_c == k)
            A->folded = true;
        const GLayer *B = i + 1u < s->layers ? &s->L[i + 1u] : NULL;
        if (B && B->run == RUN_BLOCK && B->inputs[0] == i + 1u && B->L.row_len == k) {
            const uint32_t bv = slett_block_values(B->L.precision);
            if (k % bv == 0u) G->qbv = bv;
        }
        if (G->qbv) {
            uint32_t j = i + 1u;
            while (j + 1u < s->layers && reads(&s->L[j], i + 1u) && s->L[j].run == RUN_BLOCK &&
                   slett_block_values(s->L[j].L.precision) == G->qbv && s->L[j].L.row_len == k &&
                   SPG_BMM_TILED(s->L[j].type) == SPG_BMM_TILED(B->type))
                j++;
            G->unwritten = true;
            for (; G->unwritten && j < s->layers; j++) G->unwritten = !reads(&s->L[j], i + 1u);
        }
    }
}

SpgGpuSession *spingalett_gpu_session_new(const SpingalettModel *model, uint32_t sequences, uint32_t max_length,
                                          uint32_t most, const char **why) {
    *why = NULL;
    if (!spg_gpu_kernel_built(SPG_KERNEL_bembed) || !spg_gpu_kernel_built(SPG_KERNEL_kvattn) ||
        !spg_gpu_kernel_built(SPG_KERNEL_bquant) || !spg_gpu_kernel_built(SPG_KERNEL_bdot) ||
        !spg_gpu_kernel_built(SPG_KERNEL_bmm) ||
        !spg_gpu_kernel_built(SPG_KERNEL_gew) || !spg_gpu_kernel_built(SPG_KERNEL_grow) ||
        !spg_gpu_kernel_built(SPG_KERNEL_ggather) || !spg_gpu_kernel_built(SPG_KERNEL_gcopy))
        return *why = "the backend has not its kernels (CUDA's alone, as yet)", NULL;
    SpgGpuSession *s = (SpgGpuSession *)calloc(1, sizeof *s);
    if (!s) return *why = "out of memory", NULL;
    s->image = (const uint8_t *)model->image;
    s->layers = model->layer_count;
    s->sequences = sequences;
    s->max_length = max_length;
    s->most = most;
    s->L = (GLayer *)calloc(s->layers, sizeof(GLayer));
    s->runs = (uint32_t *)malloc((size_t)(most + 1u) * sizeof(uint32_t));
    if (!s->L || !s->runs) {
        spingalett_gpu_session_free(s);
        return *why = "out of memory", NULL;
    }
    uint32_t widest = 0;
    for (uint32_t i = 0; i < s->layers; i++) {
        GLayer *G = &s->L[i];
        spingalett_slett_layer(s->image, i, &G->L);
        for (uint32_t k = 0; k < G->L.input_count; k++) G->inputs[k] = spingalett_slett_input(s->image, &G->L, k);
    }
    find_pairs(s);
    find_joins(s);
    for (uint32_t i = 0; i < s->layers; i++) {
        GLayer *G = &s->L[i];
        if (G->L.in_c > widest) widest = G->L.in_c;
        if (G->L.out_c > widest) widest = G->L.out_c;
        if (!place(s, i, why)) {
            spingalett_gpu_session_free(s);
            return NULL;
        }
    }
    const GLayer *last = &s->L[s->layers - 1u];
    s->vocabulary = last->L.out_c;
    fuse_norms(s);
    if (!layout(s)) {
        spingalett_gpu_session_free(s);
        return *why = "out of memory", NULL;
    }
    uint32_t qfloats = 0;
    for (uint32_t i = 0; i < s->layers; i++)
        if (s->L[i].run == RUN_ATTN && s->L[i].heads * s->L[i].head > qfloats) qfloats = s->L[i].heads * s->L[i].head;
    const uint32_t kmax = s->kmax ? s->kmax : 16u;
    s->q = buffer_new(s, (size_t)most * (qfloats ? qfloats : 1u) * sizeof(float), false);
    s->xq = buffer_new(s, (size_t)SPG_TILED_BYTES(most, kmax), false);     /* (as many as rows of kmax bytes, or more) */
    s->dx = buffer_new(s, (size_t)most * (kmax / 32u + 1u) * sizeof(float), false);
    s->xs = buffer_new(s, (size_t)most * (kmax / 16u + 1u) * sizeof(int32_t), false);
    /* logits of SPG_SESSION_LOGITS rows a step at most, unless the last layer runs on every row */
    s->most_logits = most < SPG_SESSION_LOGITS || last->run == RUN_ATTN || last->run == RUN_EMBED ? most
                     : SPG_SESSION_LOGITS;
    s->gathered = buffer_new(s, (size_t)s->most_logits * widest * sizeof(float), false);
    s->logits = buffer_new(s, (size_t)s->most_logits * s->vocabulary * sizeof(float), false);
    s->tokens = buffer_new(s, (size_t)most * sizeof(uint32_t), false);
    s->info = buffer_new(s, (size_t)most * 2u * sizeof(uint32_t), false);
    s->pick = buffer_new(s, (size_t)most * sizeof(uint32_t), false);
    s->info_at = (size_t)most * sizeof(uint32_t);
    s->pick_at = s->info_at + (size_t)most * 2u * sizeof(uint32_t);
    s->logits_at = (s->pick_at + (size_t)most * sizeof(uint32_t) + 15u) & ~(size_t)15u;
    if (!s->acts || !s->q || !s->xq || !s->dx || !s->xs || !s->gathered || !s->logits || !s->tokens || !s->info ||
        !s->pick || !spg_gpu_buffer_create(&s->staging, s->logits_at + (size_t)s->most_logits * s->vocabulary * sizeof(float), true)) {
        spingalett_gpu_session_free(s);
        return *why = "out of device memory", NULL;
    }
    return s;
}

uint32_t spingalett_gpu_session_rows(const SpgGpuSession *s) { return s ? s->most : 0; }

/* ------------------------------------------------------------------------- recording a step */

static const SpgGpuBuffer *buffer_at(const SpgGpuSession *s, uint64_t address) {
    for (uint32_t k = 0; k < s->buffer_count; k++)
        if (address >= s->buffers[k].address && address < s->buffers[k].address + s->buffers[k].size) return &s->buffers[k];
    return NULL;
}

static uint32_t groups(uint64_t n, uint32_t per) { return (uint32_t)((n + per - 1u) / per); }

static void gew_flat(SpgGpuCommands *c, uint32_t op, uint64_t y, uint64_t a, uint64_t b, uint32_t total, float alpha) {
    SpgGewPush p;
    memset(&p, 0, sizeof p);
    p.y = y;
    p.a = a;
    p.b = b ? b : a;
    p.c = a;
    p.total = total;
    for (uint32_t d = 0; d < SPG_GEW_AXES; d++) p.dims[d] = 1u;
    p.dims[SPG_GEW_AXES - 1u] = total;
    p.alpha = alpha;
    const uint32_t spec[3] = {op, 0u, 1u | 1u << 2 | 1u << 4};
    spg_gpu_dispatch(c, SPG_KERNEL_gew, spec, 3, &p, sizeof p, groups(total, 256u), 1, 1);
}

/* The address of layer index j's output (1 + a layer's index; the step's rows) */
static uint64_t output_of(const SpgGpuSession *s, uint32_t j) { return s->L[j - 1u].out; }

/* Layer i on `rows` rows: x its inputs' addresses, y its output's */
static void record_layer(SpgGpuSession *s, SpgGpuCommands *c, uint32_t i, uint32_t rows, const uint64_t *x, uint64_t y) {
    const GLayer *G = &s->L[i];
    const SlettLayer *L = &G->L;
    switch (G->run) {
        case RUN_EMBED: {
            SpgBembedPush p = {G->w, s->tokens, s->info, G->b, y, rows, L->out_c, G->stride, G->type, L->rows,
                               L->embed_scale};
            spg_gpu_dispatch(c, SPG_KERNEL_bembed, NULL, 0, &p, sizeof p, groups((uint64_t)rows * L->out_c, 256u), 1, 1);
            break;
        }
        case RUN_BLOCK: {
            /* (an input the block layer before quantized in blocks of as many values: quantized once) */
            const uint32_t K = L->row_len, BV = slett_block_values(L->precision);
            const bool tiled = rows >= SPG_MM_ROWS && SPG_BMM_TILED(G->type);    /* (as bmm reads them) */
            if (s->quantized != x[0] || s->quantized_bv != BV || s->quantized_tiled != tiled) {
                SpgBquantPush qp = {x[0], s->xq, s->dx, s->xs, rows, K, BV, L->in_c, tiled, 0};
                spg_gpu_dispatch(c, SPG_KERNEL_bquant, NULL, 0, &qp, sizeof qp,
                                 groups((uint64_t)rows * (K / BV), SPG_BQUANT_BLOCKS(BV)), 1, 1);
                spg_gpu_barrier(c);
                s->quantized = x[0];
                s->quantized_bv = BV;
                s->quantized_tiled = tiled;
            }
            /* (a gated unit's pair: twice the rows of weights, the outputs of the unit; a join: its layers' rows) */
            const uint32_t act = G->pair ? G->pair_act : (uint32_t)L->activation;
            const uint32_t N = G->pair ? 2u * L->rows : G->join ? G->joint : L->rows;
            if (rows >= SPG_MM_ROWS) {
                /* three rows or more on the matrix units (bmm's 64 rows cost little more than bdot's one: Qwen3-0.6B's
                   layer in Q8_0 on an RTX 4050, 150 us for up to 64 rows, 53 us a row by bdot, 365 us 8 rows) */
                const uint32_t spec[4] = {G->type, G->packed, act, G->pair != 0u};
                SpgBdotPush dp = {G->w, s->xq, s->dx, s->xs, G->b, y, G->gate, rows, N, K, G->stride, 0,
                                  G->ldy ? G->ldy : L->out_c};
                spg_gpu_dispatch(c, SPG_KERNEL_bmm, spec, 4, &dp, sizeof dp, groups(rows, 64u),
                                 groups(N, G->packed ? spg_packed_cols(G->type) : 64u), 1);
                break;
            }
            for (uint32_t r0 = 0; r0 < rows; r0++) {
                const uint32_t spec[5] = {G->type, 1u, act, G->packed, G->pair != 0u};
                SpgBdotPush dp = {G->w, s->xq, s->dx, s->xs, G->b, y, G->gate, 1u, N, K, G->stride, r0,
                                  G->ldy ? G->ldy : L->out_c};
                spg_gpu_dispatch(c, SPG_KERNEL_bdot, spec, 5, &dp, sizeof dp, groups(G->pair ? L->rows : N, 8u), 1, 1);
            }
            break;
        }
        case RUN_FLOAT: {
            SpgGemmPush p;
            memset(&p, 0, sizeof p);
            p.a = x[0];
            p.b = G->w;
            p.c = y;
            p.e0 = G->b;
            p.M = rows;
            p.N = L->rows;
            p.K = L->row_len;
            p.lda = L->in_c;
            p.ldb = L->row_len;
            p.ldc = L->out_c;
            p.alpha = 1.0f;
            p.flags = SPG_GEMM_BIAS;
            /* (its tile chosen by its shape, not timed: the rows of steps vary from one to the next) */
            const bool fused = G->act_gew == NONE;
            SpgGemmMode m = {SPG_A_ROW, SPG_B_COL, SPG_EPI_BIAS_ACT, fused ? (uint32_t)L->activation : (uint32_t)ACT_NONE,
                             1, false, true, true, 0, 0, false, 0, false, false};
            spg_gemm(c, &p, &m);
            break;
        }
        case RUN_RMS:
        case RUN_NORM: {
            if (G->tpr) {
                /* (with the sum before it, the quantization after it) */
                const GLayer *A = i && s->L[i - 1u].folded ? &s->L[i - 1u] : NULL;
                const bool tiled = G->qbv && rows >= SPG_MM_ROWS && SPG_BMM_TILED(s->L[i + 1u].type);  /* (its reader's) */
                SpgBnormPush p = {A ? output_of(s, A->inputs[0]) : x[0], A ? output_of(s, A->inputs[1]) : 0, A ? A->out : 0,
                                  G->g, G->unwritten ? 0 : y, G->qbv ? s->xq : 0, G->qbv ? s->dx : 0,
                                  G->qbv ? s->xs : 0, rows, L->out_c, G->qbv, L->eps, tiled, 0};
                const uint32_t n = L->out_c / G->tpr, spec[2] = {G->tpr, n <= 8u ? 8u : n <= 16u ? 16u : 32u};
                spg_gpu_dispatch(c, SPG_KERNEL_bnorm, spec, 2, &p, sizeof p, groups(rows, 256u / G->tpr), 1, 1);
                if (G->qbv) {
                    s->quantized = y;
                    s->quantized_bv = G->qbv;
                    s->quantized_tiled = tiled;
                }
                break;
            }
            SpgRowPush p;
            memset(&p, 0, sizeof p);
            p.x = x[0];
            p.y = y;
            p.a = G->g;
            p.b = G->run == RUN_NORM ? G->b : G->g;
            p.aux = y;
            p.outer = rows;
            p.len = L->out_c;
            p.inner = 1;
            p.rows = rows;
            p.eps = L->eps;
            p.flags = G->run == RUN_NORM ? SPG_ROW_B : 0u;
            uint32_t tpr = 1;
            while (tpr < 256u && 8u * tpr <= p.len) tpr *= 2u;
            const uint32_t spec[2] = {G->run == RUN_RMS ? SPG_ROW_RMSNORM : SPG_ROW_LAYERNORM, tpr};
            spg_gpu_dispatch(c, SPG_KERNEL_grow, spec, 2, &p, sizeof p, groups(rows, 256u / tpr), 1, 1);
            break;
        }
        case RUN_ADD:
        case RUN_MUL: {
            const uint32_t total = rows * L->out_c, op = G->run == RUN_ADD ? SPG_GEW_ADD : SPG_GEW_MUL;
            if (L->input_count == 1) {
                gew_flat(c, SPG_GEW_COPY, y, x[0], 0, total, 0.0f);
                break;
            }
            gew_flat(c, op, y, x[0], x[1], total, 0.0f);
            for (uint32_t k = 2; k < L->input_count; k++) {
                spg_gpu_barrier(c);
                gew_flat(c, op, y, y, x[k], total, 0.0f);
            }
            break;
        }
        case RUN_CONCAT: {
            /* each input's rows into its columns of the output's */
            for (uint32_t k = 0, at = 0; k < L->input_count; at += G->channels[k], k++) {
                SpgCopyPush p;
                memset(&p, 0, sizeof p);
                p.y = y + (uint64_t)at * sizeof(float);
                p.x = x[k];
                p.total = rows * G->channels[k];
                for (uint32_t d = 0; d < SPG_COPY_AXES; d++) p.dims[d] = 1u;
                p.dims[SPG_COPY_AXES - 2u] = rows;
                p.dims[SPG_COPY_AXES - 1u] = G->channels[k];
                p.sx[SPG_COPY_AXES - 2u] = (int32_t)G->channels[k];
                p.sx[SPG_COPY_AXES - 1u] = 1;
                p.sy[SPG_COPY_AXES - 2u] = (int32_t)L->out_c;
                p.sy[SPG_COPY_AXES - 1u] = 1;
                const uint32_t spec[1] = {4u};
                spg_gpu_dispatch(c, SPG_KERNEL_gcopy, spec, 1, &p, sizeof p, groups(p.total, 256u), 1, 1);
            }
            break;
        }
        case RUN_ATTN: {
            /* (with gflash its queries as bfloat16, as it would round them) */
            SpgKvattnPush p = {x[0], s->q, G->keys, G->values, y, G->table, G->gq, G->gk, s->info, rows, G->heads, G->kv,
                               G->head, G->rot, L->window, s->max_length, SPG_KV_PREP, G->eps, G->scale, G->half_kv,
                               G->strides != 0};
            spg_gpu_dispatch(c, SPG_KERNEL_kvattn, NULL, 0, &p, sizeof p, groups((uint64_t)rows * (G->heads + G->kv), 8u), 1, 1);
            spg_gpu_barrier(c);
            if (G->strides) {
                /* each run of rows (a sequence's tokens at consecutive positions: a prompt, or a token of a sequence) over
                   its caches on the matrix units, whatever rows are with it: the same bits however tokens are fed */
                const uint32_t D = G->head, unit = D;
                for (uint32_t k = 0; k < s->run_count; k++) {
                    const uint32_t r0 = s->runs[k], n = s->runs[k + 1u] - r0;
                    const bool half_rows = unit <= 64u && (uint64_t)((n + 127u) / 128u) * G->heads < 64u;
                    const uint32_t block_rows = unit > 64u || half_rows ? 64u : 128u, spec[2] = {unit, half_rows ? 1u : 2u};
                    SpgGattnPush gp;
                    memset(&gp, 0, sizeof gp);
                    gp.q = s->q + (uint64_t)r0 * G->heads * D * 2u;
                    gp.k = G->keys;
                    gp.v = G->values;
                    gp.y = y + (uint64_t)r0 * L->out_c * sizeof(float);
                    gp.mask = gp.keep = s->q;
                    gp.strides = G->strides;
                    gp.queries = n;
                    gp.d = D;
                    gp.heads = G->heads;
                    gp.scale = G->scale;
                    gp.flags = SPG_GATTN_CACHE | SPG_GATTN_VEC | SPG_GATTN_HALF_Q | SPG_GATTN_HALF_K | SPG_GATTN_HALF_V;
                    gp.info = s->info + (uint64_t)r0 * 2u * sizeof(uint32_t);
                    gp.group = G->heads / G->kv;
                    gp.window = L->window;
                    spg_gpu_dispatch(c, SPG_KERNEL_gflash, spec, 2, &gp, sizeof gp, groups(n, block_rows), G->heads, 1);
                }
                break;
            }
            p.op = SPG_KV_ATTEND;
            spg_gpu_dispatch(c, SPG_KERNEL_kvattn, NULL, 0, &p, sizeof p, rows * G->heads, 1, 1);
            break;
        }
        default:
            break;
    }
    if (G->act_gew != NONE) {
        spg_gpu_barrier(c);
        gew_flat(c, G->act_gew, y, y, 0, rows * L->out_c, G->alpha);
    }
}

/* The commands of a step of `rows` rows, the logits of `picked` of them */
static SpgGpuCommands *record_step(SpgGpuSession *s, uint32_t rows, uint32_t picked) {
    SpgGpuCommands *c = spg_gpu_commands_create();
    if (!c || !spg_gpu_record_begin(c)) {
        if (c) spg_gpu_commands_free(c);
        return NULL;
    }
    const SpgGpuBuffer *tokens = buffer_at(s, s->tokens), *info = buffer_at(s, s->info), *pick = buffer_at(s, s->pick);
    spg_gpu_copy(c, &s->staging, 0, tokens, 0, (size_t)rows * sizeof(uint32_t));
    spg_gpu_copy(c, &s->staging, s->info_at, info, 0, (size_t)rows * 2u * sizeof(uint32_t));
    spg_gpu_copy(c, &s->staging, s->pick_at, pick, 0, (size_t)(picked ? picked : 1u) * sizeof(uint32_t));
    s->quantized = 0;
    for (uint32_t i = 0; i < s->layers; i++) {
        const GLayer *G = &s->L[i];
        if (G->skip || G->folded || G->paired || G->joined || G->gathered) continue;
        spg_gpu_barrier(c);
        if (G->run != RUN_BLOCK) s->quantized = 0;      /* (a layer that may write where the quantized input was) */
        uint64_t x[SPINGALETT_MAX_INPUTS];
        for (uint32_t k = 0; k < G->L.input_count; k++) x[k] = G->inputs[k] ? output_of(s, G->inputs[k]) : 0;
        if (i + 1u < s->layers) {
            record_layer(s, c, i, rows, x, G->out);
            continue;
        }
        if (!picked) break;
        /* the last layer on the picked rows: its inputs gathered, unless every row is picked */
        const bool all = picked == rows || G->run == RUN_ATTN || G->run == RUN_EMBED;
        uint64_t at = s->gathered;
        for (uint32_t k = 0; k < G->L.input_count && !all; k++) {
            uint32_t h, w, units;
            spingalett_slett_output_shape(s->image, G->inputs[k], &h, &w, &units);
            const uint32_t C = units / (h * w);
            SpgGatherPush p = {at, x[k], s->pick, picked * C, C, picked, rows};
            const uint32_t spec[2] = {4u, SPG_T_I32};
            spg_gpu_dispatch(c, SPG_KERNEL_ggather, spec, 2, &p, sizeof p, groups((uint64_t)picked * C, 256u), 1, 1);
            x[k] = at;
            at += (uint64_t)picked * C * sizeof(float);
        }
        if (!all) spg_gpu_barrier(c);
        record_layer(s, c, i, all ? rows : picked, x, s->logits);
        spg_gpu_barrier(c);
        spg_gpu_copy(c, buffer_at(s, s->logits), 0, &s->staging, s->logits_at,
                     (size_t)(all ? rows : picked) * s->vocabulary * sizeof(float));
    }
    spg_gpu_barrier_host(c);
    if (!spg_gpu_record_end(c)) {
        spg_gpu_commands_free(c);
        return NULL;
    }
    return c;
}

bool spingalett_gpu_session_step(SpgGpuSession *s, uint32_t rows, const uint32_t *tokens, const uint32_t *sequence,
                                 const uint32_t *position, uint32_t picked, const uint32_t *pick, float *logits) {
    if (!s || !rows || rows > s->most) return false;
    const GLayer *last = &s->L[s->layers - 1u];
    const bool all = picked == rows || last->run == RUN_ATTN || last->run == RUN_EMBED;
    if ((all && picked ? rows : picked) > s->most_logits) return false;
    uint8_t *stage = (uint8_t *)s->staging.mapped;
    memcpy(stage, tokens, (size_t)rows * sizeof(uint32_t));
    uint32_t *info = (uint32_t *)(void *)(stage + s->info_at);
    for (uint32_t r = 0; r < rows; r++) info[2u * r] = sequence[r], info[2u * r + 1u] = position[r];
    if (picked) memcpy(stage + s->pick_at, pick, (size_t)picked * sizeof(uint32_t));
    /* the runs of rows of one sequence at consecutive positions (attention takes each over its caches) */
    s->run_count = 0;
    uint64_t runs = 1469598103934665603ull;
    for (uint32_t r = 0; r < rows; r++)
        if (!r || sequence[r] != sequence[r - 1u] || position[r] != position[r - 1u] + 1u) {
            s->runs[s->run_count++] = r;
            runs = (runs ^ r) * 1099511628211ull;
        }
    s->runs[s->run_count] = rows;
    SpgGpuCommands *c = NULL;
    for (uint32_t k = 0; k < KEPT && !c; k++)
        if (s->kept[k].c && s->kept[k].rows == rows && s->kept[k].picked == picked && s->kept[k].runs == runs)
            c = s->kept[k].c;
    if (!c) {
        c = record_step(s, rows, picked);
        if (!c) return false;
        Kept *k = &s->kept[s->next];
        if (k->c) spg_gpu_commands_free(k->c);
        k->rows = rows;
        k->picked = picked;
        k->runs = runs;
        k->c = c;
        s->next = (s->next + 1u) % KEPT;
    }
    if (!spg_gpu_submit(c) || !spg_gpu_wait(c)) return false;
    if (picked && logits) {
        const float *got = (const float *)(const void *)(stage + s->logits_at);
        for (uint32_t r = 0; r < picked; r++)
            memcpy(logits + (size_t)r * s->vocabulary, got + (size_t)(all ? pick[r] : r) * s->vocabulary,
                   s->vocabulary * sizeof(float));
    }
    return true;
}

/* Sequence `to` made a copy of the first `length` positions of sequence `from` (its keys and values) */
bool spingalett_gpu_session_copy(SpgGpuSession *s, uint32_t from, uint32_t to, uint32_t length) {
    if (!s || from >= s->sequences || to >= s->sequences) return false;
    if (from == to || !length) return true;
    SpgGpuCommands *c = spg_gpu_commands_create();
    bool ok = c && spg_gpu_record_begin(c);
    for (uint32_t i = 0; ok && i < s->layers; i++) {
        const GLayer *G = &s->L[i];
        if (G->run != RUN_ATTN) continue;
        const size_t per = (size_t)s->max_length * G->head * (G->half_kv ? 2u : sizeof(float)), seq = per * G->kv;
        for (int kv = 0; kv < 2; kv++) {
            const SpgGpuBuffer *b = buffer_at(s, kv ? G->values : G->keys);
            for (uint32_t g = 0; g < G->kv; g++)
                spg_gpu_copy(c, b, (size_t)from * seq + g * per, b, (size_t)to * seq + g * per,
                             (size_t)length * G->head * (G->half_kv ? 2u : sizeof(float)));
        }
    }
    ok = ok && spg_gpu_record_end(c) && spg_gpu_submit(c) && spg_gpu_wait(c);
    if (c) spg_gpu_commands_free(c);
    return ok;
}
