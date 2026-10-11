/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/* What the loaders of language models share (Spingalett.Pretrained.c, Spingalett.Gguf.c): the network of
   a decoder-only transformer, built from its hyper-parameters, its parameters read from a source of
   tensors that each file format provides. */

#pragma once

#include "Spingalett.Private.h"

/* The parts of a transformer by which the formats name their tensors. Those of the blocks come once a
   block; weights are read as PyTorch's nn.Linear holds them, [out][in]. */
typedef enum {
    SPG_PART_EMBED,                 /* [vocabulary][width] */
    SPG_PART_POSITIONS,             /* GPT-2's learned positions, [positions][width] (the first context rows) */
    SPG_PART_NORM, SPG_PART_NORM_BIAS,          /* the last normalization */
    SPG_PART_HEAD,                  /* [vocabulary][width]; absent: the embeddings (tied) */
    SPG_PART_ATTN_NORM, SPG_PART_ATTN_NORM_BIAS,
    SPG_PART_Q, SPG_PART_K, SPG_PART_V,         /* [heads x head size][width], [kv heads x head size][width] */
    SPG_PART_Q_BIAS, SPG_PART_K_BIAS, SPG_PART_V_BIAS,
    SPG_PART_QKV, SPG_PART_QKV_BIAS,            /* GPT-2's queries, keys and values in one */
    SPG_PART_O, SPG_PART_O_BIAS,
    SPG_PART_FFN_NORM, SPG_PART_FFN_NORM_BIAS,
    SPG_PART_GATE, SPG_PART_GATE_BIAS,          /* LLaMA's SwiGLU: down(silu(gate x) * up x) */
    SPG_PART_UP, SPG_PART_UP_BIAS,              /* GPT-2's MLP: down(act(up x)) */
    SPG_PART_DOWN, SPG_PART_DOWN_BIAS,
    SPG_PART_COUNT
} SpgPart;

typedef enum {
    SPG_ARCH_LLAMA,                 /* RMS normalization, rotary grouped-query attention, SwiGLU (LLaMA, Mistral,
                                       Qwen2) */
    SPG_ARCH_GPT2                   /* layer normalization, learned positions, a GELU MLP */
} SpgArch;

typedef struct {
    SpgArch arch;
    uint32_t vocabulary, width, layers;
    uint32_t heads, kv_heads, head_size;
    uint32_t hidden;                /* the MLP's width */
    uint32_t context;               /* tokens a sample: the network's window */
    float eps;                      /* the normalizations' */
    float theta;                    /* rotary embeddings' base (LLaMA) */
    ActivationFunction act;         /* GPT-2's MLP */
} SpgTransformer;

/* Where a format's tensors come from. */
typedef struct SpgTensorSource {
    /* Whether the model has a part (block: the block of a block's part). */
    bool (*has)(struct SpgTensorSource *s, SpgPart part, uint32_t block);
    /* A part's values into dst, rows x cols in row-major order (a vector: one row); false with s->error set
       when the model lacks it, its shape differs or it cannot be decoded. */
    bool (*read)(struct SpgTensorSource *s, SpgPart part, uint32_t block, float *dst, uint32_t rows, uint32_t cols);
    char error[256];
} SpgTensorSource;

/* The network of a transformer, its parameters read from src (its loss the sparse cross-entropy of the next
   token); NULL with the error set. */
NeuralNetwork *spingalett_build_transformer(const SpgTransformer *t, SpgTensorSource *src);

/* Decodes n elements of a type to floats, in chunks on the OpenMP threads when there are many. */
void spingalett_decode_parallel(float *dst, const uint8_t *src, int dtype, uint64_t n);
