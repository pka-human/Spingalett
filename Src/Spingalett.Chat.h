/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/* Chat templates (Spingalett.Chat.c): a conversation rendered by a Jinja template, as transformers renders it.
   What spingalett_chat_render() (Spingalett.Tokenizer.c, which knows the tokenizer's template and tokens) calls. */

#pragma once

#include <stdbool.h>
#include <stdint.h>

typedef struct {
    const char *source;                     /* the template */
    const char *const *roles;               /* [message_count] */
    const char *const *contents;            /* [message_count] (NULL: none) */
    uint32_t message_count;
    bool add_generation_prompt;
    const char *bos_token, *eos_token;      /* (NULL: undefined) */
    const char *variables;                  /* a JSON object of more variables, or NULL */
    uint64_t time;                          /* strftime_now()'s time, seconds since 1970 (0: now) */
} SpgChatRender;

/* The text in *text (malloc'ed, NUL-terminated); its bytes, -1 with the error set (and *text NULL). */
int64_t spingalett_chat_text(const SpgChatRender *chat, char **text);
