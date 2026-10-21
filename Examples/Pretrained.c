/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/*
 * A pretrained language model continuing a prompt: a GGUF file (LLaMA 1 to 3.x, Mistral, Qwen2 and 2.5, Qwen3,
 * Gemma 3, GPT-2), opened by spingalett_model_open() with its tensors as the file stores them, or a Hugging Face
 * model directory (config.json and safetensors weights; LLaMA, Mistral, Qwen2, GPT-2), loaded by
 * spingalett_load_pretrained() and made a model in single precision; the tokenizer of either by
 * spingalett_tokenizer_load(). The prompt is encoded with the tokens the tokenizer adds around a text (LLaMA's
 * BOS), fed to a session at once, and continued a token at a time, each token printed as it comes: the tokens
 * generated so far are decoded together and what is new printed, so that a character split between tokens
 * prints whole. Generation stops at the tokenizer's EOS.
 *
 *   Bin/Pretrained <model directory or .gguf file> ["prompt"] [tokens] [temperature] [context]
 *
 * The defaults: 64 tokens, the most likely token each step (temperature 0), a window of 1024 tokens. A
 * temperature draws among the most likely tokens whose probabilities add up to 0.9. SmolLM2-135M
 * (HuggingFaceTB/SmolLM2-135M on the Hugging Face Hub) or one of its GGUF files is a quick start.
 */

#include <Spingalett/Spingalett.Short.h>
#include <math.h>
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

/* The next token of the logits: the most likely at temperature 0, else a draw among the most likely tokens whose
   probabilities add up to 0.9. */
static uint32_t pick(const float *logits, uint32_t vocabulary, float temperature, float *p, uint32_t *order) {
    uint32_t best = 0;
    for (uint32_t v = 1; v < vocabulary; v++)
        if (logits[v] > logits[best]) best = v;
    if (temperature <= 0.0f) return best;
    double total = 0.0;
    for (uint32_t v = 0; v < vocabulary; v++) total += p[v] = expf((logits[v] - logits[best]) / temperature);
    for (uint32_t v = 0; v < vocabulary; v++) p[v] = (float)(p[v] / total), order[v] = v;
    /* the most likely first, until they hold 0.9 of the probability */
    uint32_t kept = 0;
    double held = 0.0;
    while (held < 0.9 && kept < vocabulary) {
        uint32_t at = kept;
        for (uint32_t v = kept + 1; v < vocabulary; v++)
            if (p[order[v]] > p[order[at]]) at = v;
        const uint32_t t = order[kept];
        order[kept] = order[at];
        order[at] = t;
        held += p[order[kept++]];
    }
    double r = (double)rand() / ((double)RAND_MAX + 1.0) * held;
    for (uint32_t k = 0; k < kept; k++)
        if ((r -= p[order[k]]) < 0.0) return order[k];
    return order[kept - 1];
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model directory or .gguf file> [\"prompt\"] [tokens] [temperature] [context]\n", argv[0]);
        return 2;
    }
    const char *path = argv[1], *prompt_text = argc > 2 ? argv[2] : "Once upon a time";
    const uint32_t count = argc > 3 ? (uint32_t)atoi(argv[3]) : 64u;
    const float temperature = argc > 4 ? (float)atof(argv[4]) : 0.0f;
    const uint32_t context = argc > 5 ? (uint32_t)atoi(argv[5]) : 1024u;
    spingalett_set_verbose(false);
#if defined(SPINGALETT_HAS_OPENMP)
    spingalett_set_compute_mode(COMPUTE_OPENMP);
#endif
    srand((unsigned)time(NULL));

    /* a GGUF file as it is stored; a directory's network as a model of its weights in single precision */
    double start = seconds_now();
    const size_t length = strlen(path);
    const bool gguf = length > 5 && !strcmp(path + length - 5, ".gguf");
    SpingalettModel *model = NULL;
    if (gguf) {
        model = spingalett_model_open(.path = path, .context = context);
    } else {
        NeuralNetwork *net = spingalett_load_pretrained(.path = path, .context = context);
        model = net ? spingalett_model_from_network(net, PRECISION_FLOAT32) : NULL;
        free_network(net);
    }
    SpingalettTokenizer *tok = model ? spingalett_tokenizer_load(path) : NULL;
    SpingalettSession *s = tok ? spingalett_session_new(model, NULL) : NULL;
    if (!s) {
        fprintf(stderr, "%s: %s\n", path, spingalett_last_error_message());
        spingalett_tokenizer_free(tok);
        spingalett_model_free(model);
        return 1;
    }
    const uint32_t vocabulary = spingalett_session_vocabulary(s);
    fprintf(stderr, "%u layers, a window of %u tokens, loaded in %.2f s\n", model->layer_count, model->input_size,
            seconds_now() - start);

    /* the prompt's tokens, with BOS where the tokenizer adds one, and room for those to come */
    int64_t n = spingalett_tokenize(.tokenizer = tok, .text = prompt_text, .add_special = true);
    uint32_t *tokens = n > 0 ? malloc(((size_t)n + count) * sizeof(uint32_t)) : NULL;
    if (!tokens || spingalett_tokenize(.tokenizer = tok, .text = prompt_text, .tokens = tokens, .capacity = (size_t)n,
                                       .add_special = true) != n || (uint64_t)n + count > model->input_size) {
        fprintf(stderr, "the prompt cannot be encoded, or it and the tokens to come exceed the window: %s\n",
                spingalett_last_error_message());
        return 1;
    }
    /* the text is printed as the tokens decode together (a word's space depends on the tokens before it) */
    size_t capacity = 4096;
    char *text = malloc(capacity);
    int64_t size = decoded(tok, tokens, (size_t)n, &text, &capacity);
    size_t printed = size > 0 ? (size_t)size : 0;
    fwrite(text, 1, printed, stdout);
    fflush(stdout);

    /* the prompt at once, then a token at a time: each fed once, its row through every layer */
    float *logits = malloc((size_t)vocabulary * sizeof(float)), *p = malloc((size_t)vocabulary * sizeof(float));
    uint32_t *order = malloc((size_t)vocabulary * sizeof(uint32_t));
    const SpingalettTokenizerInfo info = spingalett_tokenizer_info(tok);
    start = seconds_now();
    bool ok = spingalett_session_feed(s, &(SpingalettSessionFeed){.tokens = tokens, .count = (uint32_t)n,
                                                                    .logits = logits}, 1);
    const double prompt_seconds = seconds_now() - start;
    uint32_t total = 0;
    start = seconds_now();
    for (; ok && total < count; total++) {
        const uint32_t next = pick(logits, vocabulary, temperature, p, order);
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
        ok = spingalett_session_feed(s, &(SpingalettSessionFeed){.tokens = &tokens[n + total], .count = 1,
                                                                  .logits = logits}, 1);
    }
    const double seconds = seconds_now() - start;
    printf("\n");
    fflush(stdout);
    fprintf(stderr, "a prompt of %lld tokens in %.2f s, %u tokens in %.2f s (%.1f a second)\n", (long long)n,
            prompt_seconds, total, seconds, seconds > 0.0 ? total / seconds : 0.0);
    free(text);
    free(tokens);
    free(logits);
    free(p);
    free(order);
    spingalett_session_free(s);
    spingalett_tokenizer_free(tok);
    spingalett_model_free(model);
    return 0;
}
