/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/* What the tokenizers (Spingalett.Tokenizer.c) offer the rest of the library: a tokenizer made from the
   vocabulary of a GGUF file's metadata (Spingalett.Gguf.c). */

#pragma once

#include "Spingalett/Spingalett.h"

/* A string of a file: bytes, not NUL-terminated. */
typedef struct {
    const char *s;
    uint32_t n;
} SpgString;

/* The tokenizer.ggml.* keys of a GGUF file. */
typedef struct {
    const char *model;              /* tokenizer.ggml.model: "gpt2" (byte-level BPE with merges) or "llama"
                                       (SentencePiece's BPE: scores, byte fallback) */
    const char *pre;                /* tokenizer.ggml.pre (the pre-tokenizer of "gpt2" models), or NULL */
    uint32_t count;                 /* tokens: ids 0 to count - 1 */
    const SpgString *tokens;
    const float *scores;            /* [count] or NULL */
    const int32_t *types;           /* [count] or NULL: 1 normal, 2 unknown, 3 control, 4 user-defined,
                                       5 unused, 6 byte */
    uint32_t merge_count;
    const SpgString *merges;        /* "a b", or NULL */
    uint32_t bos, eos, unk, pad;    /* SPINGALETT_NO_TOKEN when absent */
    bool add_space_prefix;          /* "llama": a space before the text (and after each special token) */
    bool add_bos, add_eos;          /* the template of add_special */
    const char *chat_template;      /* tokenizer.chat_template (not NUL-terminated), or NULL */
    uint32_t chat_template_length;
} SpgTokenizerSpec;

/* NULL with the error set when the vocabulary cannot be used. */
SpingalettTokenizer *spingalett_tokenizer_build(const SpgTokenizerSpec *spec);

/* The tokenizer of a GGUF file in memory (Spingalett.Gguf.c), which keeps nothing of its bytes; NULL with
   the error set. */
SpingalettTokenizer *spingalett_gguf_tokenizer(const uint8_t *data, size_t size);
