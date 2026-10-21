/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/*
 * A chat with a language model in the terminal, with the runtime alone (Spingalett.Runtime.h, libspingalett-runtime):
 * the model of a GGUF file as the file stores it, its tokenizer and its chat template.
 *
 *   Chat <model.gguf> [options]
 *     --system TEXT          the system message
 *     --temperature T        0.7 (0: the most likely token each step)
 *     --top-k K, --top-p P   0, 0.95 (0: all)
 *     --min-p P              0.05
 *     --repeat-penalty R     1.0, over the last 64 tokens
 *     --tokens N             at most N tokens a reply (512)
 *     --context N            the window: the conversation's tokens at most (4096)
 *     --seed S               the draws' seed (the time by default)
 *     --threads N            at most N threads (all cores by default)
 *
 * Each turn renders the whole conversation with the model's template (spingalett_chat_render()) and tokenizes it;
 * the session keeps the tokens it holds from the turns before and is fed only what follows the part they share, so
 * that a turn costs its own tokens. The reply streams as it is drawn (SpingalettSampling's on_token): the tokens so
 * far decoded together, what is new printed, a character split between tokens printed whole. A reply ends at the
 * tokenizer's EOS or a token that ends turns in the common templates. An empty line, or end of input, ends the
 * chat; "/reset" begins a new conversation.
 */

#define SPINGALETT_SHORT_NAMES
#include <Spingalett/Spingalett.Runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#if !defined(TIME_UTC) && defined(_WIN32)
#include <windows.h>
#endif

static double now(void) {
#if defined(TIME_UTC)
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
#elif defined(_WIN32)
    LARGE_INTEGER frequency, counter;
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart / (double)frequency.QuadPart;
#else
    return (double)clock() / CLOCKS_PER_SEC;
#endif
}

/* A growing string. */
typedef struct {
    char *s;
    size_t n, cap;
} Text;

static int text_add(Text *t, const char *s, size_t n) {
    if (t->n + n + 1 > t->cap) {
        size_t cap = t->cap ? t->cap : 256;
        char *grown;
        while (cap < t->n + n + 1) cap *= 2;
        grown = (char *)realloc(t->s, cap);
        if (!grown) return 0;
        t->s = grown;
        t->cap = cap;
    }
    memcpy(t->s + t->n, s, n);
    t->n += n;
    t->s[t->n] = '\0';
    return 1;
}

/* A line of input without its newline into t; 0 at end of input. */
static int read_line(Text *t) {
    char chunk[512];
    t->n = 0;
    if (t->s) t->s[0] = '\0';
    while (fgets(chunk, sizeof chunk, stdin)) {
        size_t n = strlen(chunk);
        int end = n && chunk[n - 1] == '\n';
        if (end) n--;
        if (!text_add(t, chunk, n)) return 0;
        if (end) return 1;
    }
    return t->n > 0;
}

/* The reply as it streams: its tokens, and the bytes of their text already printed. */
typedef struct {
    const SpingalettTokenizer *tok;
    uint32_t *tokens;
    uint32_t count, capacity;
    size_t printed;
    Text text;
} Reply;

static bool on_token(void *user, uint32_t sequence, uint32_t token) {
    Reply *r = (Reply *)user;
    int64_t size;
    (void)sequence;
    if (r->count == r->capacity) return false;
    r->tokens[r->count++] = token;
    size = spingalett_detokenize(.tokenizer = r->tok, .tokens = r->tokens, .count = r->count, .skip_special = true);
    if (size < 0) return true;
    if ((size_t)size + 1 > r->text.cap) {
        char *grown = (char *)realloc(r->text.s, 2 * ((size_t)size + 1));
        if (!grown) return false;
        r->text.s = grown;
        r->text.cap = 2 * ((size_t)size + 1);
    }
    spingalett_detokenize(.tokenizer = r->tok, .tokens = r->tokens, .count = r->count, .text = r->text.s,
                          .capacity = r->text.cap, .skip_special = true);
    r->text.n = (size_t)size;
    /* (nothing while a character is split between tokens: its bytes decode as U+FFFD) */
    if (size >= 3 && !memcmp(r->text.s + size - 3, "\xEF\xBF\xBD", 3)) return true;
    if ((size_t)size > r->printed) {
        fwrite(r->text.s + r->printed, 1, (size_t)size - r->printed, stdout);
        fflush(stdout);
        r->printed = (size_t)size;
    }
    return true;
}

static int usage(const char *self) {
    fprintf(stderr, "usage: %s <model.gguf> [--system TEXT] [--temperature T] [--top-k K] [--top-p P] [--min-p P]\n"
                    "       [--repeat-penalty R] [--tokens N] [--context N] [--seed S] [--threads N]\n", self);
    return 2;
}

int main(int argc, char **argv) {
    const char *path = NULL, *system_message = NULL;
    float temperature = 0.7f, top_p = 0.95f, min_p = 0.05f, repeat = 1.0f;
    uint32_t top_k = 0, max_tokens = 512, context = 4096, threads = 0;
    uint64_t seed = (uint64_t)time(NULL);
    int i;
    for (i = 1; i < argc; i++) {
        const char *a = argv[i], *v = i + 1 < argc ? argv[i + 1] : NULL;
        if (a[0] != '-') { path = a; continue; }
        if (!v) return usage(argv[0]);
        if (!strcmp(a, "--system")) system_message = v;
        else if (!strcmp(a, "--temperature")) temperature = (float)atof(v);
        else if (!strcmp(a, "--top-k")) top_k = (uint32_t)atoi(v);
        else if (!strcmp(a, "--top-p")) top_p = (float)atof(v);
        else if (!strcmp(a, "--min-p")) min_p = (float)atof(v);
        else if (!strcmp(a, "--repeat-penalty")) repeat = (float)atof(v);
        else if (!strcmp(a, "--tokens")) max_tokens = (uint32_t)atoi(v);
        else if (!strcmp(a, "--context")) context = (uint32_t)atoi(v);
        else if (!strcmp(a, "--seed")) seed = (uint64_t)strtoull(v, NULL, 10);
        else if (!strcmp(a, "--threads")) threads = (uint32_t)atoi(v);
        else return usage(argv[0]);
        i++;
    }
    if (!path) return usage(argv[0]);
    spingalett_set_verbose(false);
    spingalett_set_compute_mode(COMPUTE_OPENMP);
    if (threads) spingalett_set_num_threads((int)threads);

    double start = now();
    SpingalettModel *model = spingalett_model_open(.path = path, .context = context);
    SpingalettTokenizer *tok = model ? spingalett_tokenizer_load(path) : NULL;
    SpingalettSession *s = tok ? spingalett_session_new(model, NULL) : NULL;
    if (!s) {
        fprintf(stderr, "%s: %s\n", path, spingalett_last_error_message());
        return 1;
    }
    if (!spingalett_tokenizer_chat_template(tok)) {
        fprintf(stderr, "%s: the model has no chat template\n", path);
        return 1;
    }
    context = model->input_size;
    fprintf(stderr, "%u layers, a window of %u tokens, loaded in %.2f s. An empty line ends the chat, /reset begins "
                    "another.\n", model->layer_count, context, now() - start);

    /* the tokens that end a turn: EOS, and those of the common templates the vocabulary has */
    static const char *const ends[] = {"<|im_end|>", "<|eot_id|>", "<end_of_turn>", "<|end|>", "<|endoftext|>",
                                       "<|end_of_text|>", "</s>", "<｜end▁of▁sentence｜>", "<|user|>", "<|im_start|>"};
    uint32_t stops[16], stop_count = 0;
    SpingalettTokenizerInfo info = spingalett_tokenizer_info(tok);
    if (info.eos != SPINGALETT_NO_TOKEN) stops[stop_count++] = info.eos;
    for (i = 0; i < (int)(sizeof ends / sizeof *ends); i++) {
        uint32_t id = spingalett_tokenizer_id(tok, ends[i]);
        if (id != SPINGALETT_NO_TOKEN) stops[stop_count++] = id;
    }

    /* the conversation, and the tokens the session holds */
    SpingalettChatMessage *messages = NULL;
    uint32_t message_count = 0, held_count = 0;
    uint32_t *held = (uint32_t *)malloc((size_t)context * sizeof(uint32_t)), *prompt = NULL;
    Text line = {0}, rendered = {0};
    Reply reply;
    memset(&reply, 0, sizeof reply);
    reply.tok = tok;
    reply.tokens = (uint32_t *)malloc(((size_t)max_tokens + 1) * sizeof(uint32_t));
    reply.capacity = max_tokens;
    if (!held || !reply.tokens) return 1;
    for (;;) {
        printf("\n> ");
        fflush(stdout);
        if (!read_line(&line) || line.n == 0) break;
        if (!strcmp(line.s, "/reset")) {
            for (i = 0; i < (int)message_count; i++) free((char *)messages[i].content);
            message_count = 0;
            held_count = 0;
            spingalett_session_rewind(s, 0, 0);
            continue;
        }
        SpingalettChatMessage *grown = (SpingalettChatMessage *)realloc(messages, (message_count + 3) * sizeof *messages);
        if (!grown) break;
        messages = grown;
        if (message_count == 0 && system_message) {
            messages[0].role = "system";
            messages[0].content = strcpy((char *)malloc(strlen(system_message) + 1), system_message);
            message_count = 1;
        }
        messages[message_count].role = "user";
        messages[message_count].content = strcpy((char *)malloc(line.n + 1), line.s);
        message_count++;

        /* the conversation's text, and its tokens (the template writes the special tokens) */
        int64_t size = spingalett_chat_render(.tokenizer = tok, .messages = messages, .count = message_count,
                                              .add_generation_prompt = true);
        if (size < 0) { fprintf(stderr, "%s\n", spingalett_last_error_message()); break; }
        if ((size_t)size + 1 > rendered.cap) {
            char *bigger = (char *)realloc(rendered.s, (size_t)size + 1);
            if (!bigger) break;
            rendered.s = bigger;
            rendered.cap = (size_t)size + 1;
        }
        spingalett_chat_render(.tokenizer = tok, .messages = messages, .count = message_count,
                               .add_generation_prompt = true, .text = rendered.s, .capacity = rendered.cap);
        int64_t n = spingalett_tokenize(.tokenizer = tok, .text = rendered.s, .special = true);
        if (n <= 0 || (uint64_t)n + 1 >= context) {
            fprintf(stderr, "the conversation (%lld tokens) fills the window of %u: /reset begins another\n",
                    (long long)n, context);
            message_count--;
            continue;
        }
        free(prompt);
        prompt = (uint32_t *)malloc((size_t)n * sizeof(uint32_t));
        spingalett_tokenize(.tokenizer = tok, .text = rendered.s, .tokens = prompt, .capacity = (size_t)n, .special = true);
        /* what the session holds of it, and the rest fed (one token at least) */
        uint32_t shared = 0;
        while (shared < held_count && shared < (uint32_t)n && held[shared] == prompt[shared]) shared++;
        if (shared == (uint32_t)n) shared--;
        spingalett_session_rewind(s, 0, shared);

        uint32_t room = context - (uint32_t)n, made = 0, fed = (uint32_t)n - shared;
        const uint32_t *start_at = prompt + shared;
        SpingalettSampling sampling;
        memset(&sampling, 0, sizeof sampling);
        sampling.min_p = min_p;
        sampling.repetition_penalty = repeat;
        sampling.on_token = on_token;
        sampling.user = &reply;
        reply.count = 0;
        reply.printed = 0;
        double t = now();
        uint32_t total = spingalett_session_generate(.session = s, .prompts = &start_at, .prompt_lengths = &fed,
                                                     .tokens = reply.tokens, .made = &made,
                                                     .count = room < max_tokens ? room : max_tokens,
                                                     .temperature = temperature, .top_k = top_k, .top_p = top_p,
                                                     .seed = seed++, .stop_tokens = stops, .stop_count = stop_count,
                                                     .sampling = &sampling);
        t = now() - t;
        if (!total) { fprintf(stderr, "\n%s\n", spingalett_last_error_message()); break; }
        fprintf(stderr, "\n[%u tokens fed, %u drawn, %.1f tokens a second]", fed, made, made / t);
        /* the session holds the prompt and the reply; the conversation, the reply's text */
        memcpy(held, prompt, (size_t)n * sizeof(uint32_t));
        held_count = spingalett_session_length(s, 0);
        if (held_count > (uint32_t)n) memcpy(held + n, reply.tokens, (size_t)(held_count - (uint32_t)n) * sizeof(uint32_t));
        messages[message_count].role = "assistant";
        messages[message_count].content = strcpy((char *)malloc(reply.text.n + 1), reply.count && reply.text.s ? reply.text.s : "");
        message_count++;
    }
    printf("\n");
    for (i = 0; i < (int)message_count; i++) free((char *)messages[i].content);
    free(messages);
    free(held);
    free(prompt);
    free(reply.tokens);
    free(reply.text.s);
    free(line.s);
    free(rendered.s);
    spingalett_session_free(s);
    spingalett_tokenizer_free(tok);
    spingalett_model_free(model);
    return 0;
}
