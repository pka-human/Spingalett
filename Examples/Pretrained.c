/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/*
 * A pretrained language model continuing a prompt: a Hugging Face model directory (config.json and
 * safetensors weights; LLaMA, Mistral, Qwen2, GPT-2) or a GGUF file, loaded by
 * spingalett_load_pretrained(), and its tokenizer by spingalett_tokenizer_load(). The prompt is encoded
 * with the tokens the tokenizer adds around a text (LLaMA's BOS) and continued a token at a time, each
 * token printed as it comes: the tokens generated so far are decoded together and what is new printed, so
 * that a character split between tokens prints whole. Generation stops at the tokenizer's EOS.
 *
 *   Bin/Pretrained <model directory or .gguf file> ["prompt"] [tokens] [temperature] [context]
 *
 * The defaults: 64 tokens, the most likely token each step (temperature 0), a window of 256 tokens. A
 * temperature draws among the most likely tokens whose probabilities add up to 0.9. SmolLM2-135M
 * (HuggingFaceTB/SmolLM2-135M on the Hugging Face Hub) or one of its GGUF files is a quick start.
 */

#include <Spingalett/Spingalett.Short.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double seconds_now(void) {
    struct timespec t;
    timespec_get(&t, TIME_UTC);
    return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

/* The text of tokens (special ones left out) into *text, grown as needed; its bytes, -1 on error. */
static int64_t decoded(const SpingalettTokenizer *tok, const uint32_t *tokens, size_t count, char **text, size_t *capacity) {
    int64_t size = spingalett_detokenize(.tokenizer = tok, .tokens = tokens, .count = count, .skip_special = true);
    if (size < 0) return -1;
    if ((size_t)size + 1 > *capacity) {
        char *grown = realloc(*text, 2 * ((size_t)size + 1));
        if (!grown) return -1;
        *text = grown;
        *capacity = 2 * ((size_t)size + 1);
    }
    return spingalett_detokenize(.tokenizer = tok, .tokens = tokens, .count = count, .text = *text, .capacity = *capacity,
                                 .skip_special = true);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model directory or .gguf file> [\"prompt\"] [tokens] [temperature] [context]\n", argv[0]);
        return 2;
    }
    const char *path = argv[1], *prompt_text = argc > 2 ? argv[2] : "Once upon a time";
    const uint32_t count = argc > 3 ? (uint32_t)atoi(argv[3]) : 64u;
    const float temperature = argc > 4 ? (float)atof(argv[4]) : 0.0f;
    const uint32_t context = argc > 5 ? (uint32_t)atoi(argv[5]) : 256u;
    spingalett_set_verbose(false);

    double start = seconds_now();
    NeuralNetwork *net = spingalett_load_pretrained(.path = path, .context = context);
    SpingalettTokenizer *tok = net ? spingalett_tokenizer_load(path) : NULL;
    if (!net || !tok) {
        fprintf(stderr, "%s: %s\n", path, spingalett_last_error_message());
        free_network(net);
        return 1;
    }
    fprintf(stderr, "%llu parameters, a window of %u tokens, loaded in %.2f s\n",
            (unsigned long long)spingalett_parameter_count(net), context, seconds_now() - start);

    /* the prompt's tokens, with BOS where the tokenizer adds one, and room for those to come */
    int64_t n = spingalett_tokenize(.tokenizer = tok, .text = prompt_text, .add_special = true);
    uint32_t *tokens = n > 0 ? malloc(((size_t)n + count) * sizeof(uint32_t)) : NULL;
    if (!tokens || spingalett_tokenize(.tokenizer = tok, .text = prompt_text, .tokens = tokens, .capacity = (size_t)n,
                                       .add_special = true) != n) {
        fprintf(stderr, "the prompt cannot be encoded: %s\n", spingalett_last_error_message());
        return 1;
    }
    /* the text is printed as the tokens decode together (a word's space depends on the tokens before it) */
    size_t capacity = 4096;
    char *text = malloc(capacity);
    int64_t size = decoded(tok, tokens, (size_t)n, &text, &capacity);
    size_t printed = size > 0 ? (size_t)size : 0;
    fwrite(text, 1, printed, stdout);
    fflush(stdout);

    /* a token at a time: the window is the prompt and the tokens made so far */
    const SpingalettTokenizerInfo info = spingalett_tokenizer_info(tok);
    uint32_t total = 0;
    start = seconds_now();
    for (; total < count; total++) {
        uint32_t next;
        if (!spingalett_generate(.net = net, .prompt = tokens, .prompt_length = (uint32_t)n + total, .tokens = &next,
                                 .count = 1, .temperature = temperature, .top_p = temperature > 0.0f ? 0.9f : 0.0f))
            break;
        if (next == info.eos) break;
        tokens[n + total] = next;
        /* what the new token adds to the text printed (nothing while a character is split between tokens) */
        size = decoded(tok, tokens, (size_t)n + total + 1, &text, &capacity);
        if (size < 0) break;
        const bool split = size >= 3 && !memcmp(text + size - 3, "\xEF\xBF\xBD", 3);
        if (!split && (size_t)size > printed) {
            fwrite(text + printed, 1, (size_t)size - printed, stdout);
            fflush(stdout);
            printed = (size_t)size;
        }
    }
    const double seconds = seconds_now() - start;
    printf("\n");
    fflush(stdout);
    fprintf(stderr, "%u tokens in %.2f s\n", total, seconds);
    free(text);
    free(tokens);
    spingalett_tokenizer_free(tok);
    free_network(net);
    return 0;
}
