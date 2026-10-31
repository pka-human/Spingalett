/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/*
 * Times a language model's sessions as llama.cpp's llama-bench times its models, on the same GGUF files: a prompt of
 * random tokens fed at once with the last token's logits (pp), then random tokens fed one at a time into an empty
 * sequence (tg). Prints the medians in tokens a second. -a compares the logits of a prompt fed at once with the
 * same prompt fed a token at a time, which sessions keep equal.
 *
 *   LlmBench model.gguf [-d cpu|cuda|vulkan] [-p 512] [-n 128] [-r 5] [-t threads] [-c context] [-m MiB] [-a] [-v]
 *
 * -m bounds the GPU's memory a session takes (a model it does not hold splits between the GPU and the CPU).
 */

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
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
#elif defined(_WIN32)
    LARGE_INTEGER frequency, counter;
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart / (double)frequency.QuadPart;
#else
    return (double)clock() / CLOCKS_PER_SEC;
#endif
}

static int compare(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : x > y;
}

static double median(double *values, int count) {
    qsort(values, (size_t)count, sizeof *values, compare);
    return values[count / 2];
}

/* A generator of the tokens: the same prompt every time, for comparisons between builds. */
static uint32_t next_token(uint64_t *state, uint32_t vocabulary) {
    *state = *state * 6364136223846793005ull + 1442695040888963407ull;
    return (uint32_t)((*state >> 33) % vocabulary);
}

static bool feed(SpingalettSession *s, const uint32_t *tokens, uint32_t count, float *logits) {
    SpingalettSessionFeed f = {.sequence = 0, .tokens = tokens, .count = count, .logits = logits};
    return spingalett_session_feed(s, &f, 1);
}

static int usage(const char *name) {
    fprintf(stderr, "usage: %s model.gguf [-d cpu|cuda|vulkan] [-p prompt] [-n tokens] [-r repeats] [-t threads] "
                    "[-c context] [-m MiB] [-a] [-v]\n", name);
    return 2;
}

int main(int argc, char **argv) {
    const char *path = NULL, *device = "cpu";
    uint32_t prompt = 512, tokens = 128, context = 0;
    uint64_t memory = 0;
    int repeats = 5, threads = 0;
    bool agree = false;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if (!strcmp(a, "-a")) agree = true;
        else if (!strcmp(a, "-v")) spingalett_set_log_level(SPINGALETT_LOG_DEBUG);
        else if (a[0] != '-') path = a;
        else if (i + 1 == argc) return usage(argv[0]);
        else if (!strcmp(a, "-d")) device = argv[++i];
        else if (!strcmp(a, "-p")) prompt = (uint32_t)atoi(argv[++i]);
        else if (!strcmp(a, "-n")) tokens = (uint32_t)atoi(argv[++i]);
        else if (!strcmp(a, "-r")) repeats = atoi(argv[++i]);
        else if (!strcmp(a, "-t")) threads = atoi(argv[++i]);
        else if (!strcmp(a, "-c")) context = (uint32_t)atoi(argv[++i]);
        else if (!strcmp(a, "-m")) memory = (uint64_t)atoi(argv[++i]) << 20;
        else return usage(argv[0]);
    }
    if (!path || repeats < 1 || repeats > 1000) return usage(argv[0]);

    SpingalettComputeMode mode = !strcmp(device, "cuda") ? SPINGALETT_COMPUTE_CUDA
                                 : !strcmp(device, "vulkan") ? SPINGALETT_COMPUTE_VULKAN : SPINGALETT_COMPUTE_OPENMP;
    spingalett_set_compute_mode(mode);
    if (threads > 0) spingalett_set_num_threads((unsigned)threads);

    double start = now();
    SpingalettModel *model = spingalett_model_open(.path = path, .context = context ? context : prompt + tokens + 1);
    SpingalettSessionOptions options;
    memset(&options, 0, sizeof options);
    options.gpu_memory = memory;
    SpingalettSession *s = model ? spingalett_session_new(model, &options) : NULL;
    if (!s) {
        fprintf(stderr, "%s: %s\n", path, spingalett_last_error_message());
        return 1;
    }
    uint32_t vocabulary = spingalett_session_vocabulary(s);
    uint32_t *ids = malloc(sizeof *ids * (size_t)(prompt + tokens + 1));
    float *logits = malloc(sizeof *logits * vocabulary), *other = malloc(sizeof *other * vocabulary);
    double *times = malloc(sizeof *times * (size_t)repeats);
    if (!ids || !logits || !other || !times) return 1;
    uint64_t state = 1;
    for (uint32_t i = 0; i < prompt + tokens + 1; i++) ids[i] = next_token(&state, vocabulary);

    /* The first run makes the kernels and the plans, and is not counted. */
    if (!feed(s, ids, prompt, logits)) goto failed;
    spingalett_session_rewind(s, 0, 0);
    if (!feed(s, ids, 1, logits)) goto failed;
    spingalett_session_rewind(s, 0, 0);
    printf("%s on %s: open and first runs %.2f s\n", path, device, now() - start);

    double pp = 0.0, tg = 0.0;
    if (prompt) {
        for (int r = 0; r < repeats; r++) {
            start = now();
            if (!feed(s, ids, prompt, logits)) goto failed;
            times[r] = now() - start;
            spingalett_session_rewind(s, 0, 0);
        }
        pp = (double)prompt / median(times, repeats);
    }
    if (tokens) {
        for (int r = 0; r < repeats; r++) {
            start = now();
            for (uint32_t i = 0; i < tokens; i++)
                if (!feed(s, ids + i, 1, logits)) goto failed;
            times[r] = now() - start;
            spingalett_session_rewind(s, 0, 0);
        }
        tg = (double)tokens / median(times, repeats);
    }
    printf("pp%u %.1f t/s, tg%u %.1f t/s\n", prompt, pp, tokens, tg);

    if (agree && prompt) {
        if (!feed(s, ids, prompt, logits)) goto failed;
        spingalett_session_rewind(s, 0, 0);
        for (uint32_t i = 0; i < prompt; i++)
            if (!feed(s, ids + i, 1, other)) goto failed;
        spingalett_session_rewind(s, 0, 0);
        float most = 0.0f, largest = 0.0f;
        for (uint32_t v = 0; v < vocabulary; v++) {
            const float d = logits[v] > other[v] ? logits[v] - other[v] : other[v] - logits[v];
            if (d > most) most = d;
            if (logits[v] > largest) largest = logits[v];
        }
        printf("a prompt at once and token by token: ");
        if (memcmp(logits, other, sizeof *logits * vocabulary)) printf("logits up to %.3g apart, the largest %.3g\n", most, largest);
        else printf("the same bits\n");
    }

    free(times);
    free(other);
    free(logits);
    free(ids);
    spingalett_session_free(s);
    spingalett_model_free(model);
    return 0;

failed:
    fprintf(stderr, "%s: %s\n", path, spingalett_last_error_message());
    return 1;
}
