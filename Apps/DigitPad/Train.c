/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/*
 * Trains the DigitPad model: a 784-1024-512-10 MLP on MNIST with on-the-fly augmentation (random
 * rotation, scale, aspect, shear, shift and stroke thickness) fed through a data generator, so
 * the network copes with digits drawn by hand rather than only with scanned MNIST digits.
 * 5,000 training images are held out to select the best epoch; the test set is evaluated once.
 *
 *   DigitPadTrain <mnist-dir> <output.nn> [epochs]
 */

#include "Digits.h"
#include <Spingalett/Spingalett.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define VALIDATION 5000

typedef struct {
    uint32_t count;
    float *images;
    uint8_t *labels;
} Split;

static uint32_t read_be32(FILE *f) {
    unsigned char b[4];
    if (fread(b, 1, 4, f) != 4) return 0;
    return ((uint32_t)b[0] << 24) | ((uint32_t)b[1] << 16) | ((uint32_t)b[2] << 8) | b[3];
}

static bool load_split(const char *dir, const char *images_name, const char *labels_name, Split *s) {
    char path[1024];
    snprintf(path, sizeof path, "%s/%s", dir, images_name);
    FILE *fi = fopen(path, "rb");
    snprintf(path, sizeof path, "%s/%s", dir, labels_name);
    FILE *fl = fopen(path, "rb");
    bool ok = fi && fl && read_be32(fi) == 0x803 && read_be32(fl) == 0x801;
    if (ok) {
        uint32_t n = read_be32(fi), rows = read_be32(fi), cols = read_be32(fi);
        ok = n == read_be32(fl) && rows == 28 && cols == 28;
        unsigned char *pixels = ok ? malloc((size_t)n * DIGIT_PIXELS) : NULL;
        s->count = n;
        s->images = ok ? malloc((size_t)n * DIGIT_PIXELS * sizeof(float)) : NULL;
        s->labels = ok ? malloc(n) : NULL;
        ok = ok && pixels && s->images && s->labels &&
             fread(pixels, 1, (size_t)n * DIGIT_PIXELS, fi) == (size_t)n * DIGIT_PIXELS &&
             fread(s->labels, 1, n, fl) == n;
        for (size_t i = 0; ok && i < (size_t)n * DIGIT_PIXELS; i++)
            s->images[i] = pixels[i] / 255.0f;
        free(pixels);
    }
    if (fi) fclose(fi);
    if (fl) fclose(fl);
    return ok;
}

/* ---- generator: endless stream of augmented training samples, reshuffled every pass ---- */
typedef struct {
    const Split *data;
    uint32_t first, count;          /* training range inside data */
    uint32_t *order, pos;
    DigitRng rng;
} Stream;

static void shuffle(Stream *st) {
    for (uint32_t i = st->count - 1; i > 0; i--) {
        uint32_t j = (uint32_t)(digit_rng_next(&st->rng) % (i + 1));
        uint32_t t = st->order[i]; st->order[i] = st->order[j]; st->order[j] = t;
    }
}

static uint32_t augmented_batch(float *inputs, float *targets, uint32_t requested, void *user) {
    Stream *st = user;
    for (uint32_t s = 0; s < requested; s++) {
        if (st->pos == st->count) { shuffle(st); st->pos = 0; }
        uint32_t idx = st->first + st->order[st->pos++];
        digit_augment(st->data->images + (size_t)idx * DIGIT_PIXELS, inputs + (size_t)s * DIGIT_PIXELS, &st->rng);
        float *t = targets + (size_t)s * 10;
        for (int k = 0; k < 10; k++) t[k] = 0.0f;
        t[st->data->labels[idx]] = 1.0f;
    }
    return requested;
}

/* ---- evaluation ---- */
static float *eval_outputs;

static float accuracy(NeuralNetwork *net, const float *images, const uint8_t *labels, uint32_t n) {
    predict(.net = net, .inputs = images, .sample_count = n, .outputs = eval_outputs);
    uint32_t correct = 0;
    for (uint32_t i = 0; i < n; i++) {
        const float *o = eval_outputs + (size_t)i * 10;
        int best = 0;
        for (int k = 1; k < 10; k++) if (o[k] > o[best]) best = k;
        correct += best == labels[i];
    }
    return (float)correct / (float)n;
}

static const Split *g_train;
static float g_best = 0.0f;
static size_t g_best_epoch = 0;
static const char *g_output;
static double g_start;

static double now(void) {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static bool on_epoch(NeuralNetwork *net, size_t epoch, float loss) {
    uint32_t first = g_train->count - VALIDATION;
    float val = accuracy(net, g_train->images + (size_t)first * DIGIT_PIXELS, g_train->labels + first, VALIDATION);
    bool best = val > g_best;
    if (best) {
        g_best = val;
        g_best_epoch = epoch;
        save_spingalett(.net = net, .filename = g_output, .do_not_save_optimizer = true);
    }
    printf("epoch %3zu  loss %.4f  validation %.2f%%%s  (%.0f s)\n",
           epoch, (double)loss, 100.0 * (double)val, best ? "  *" : "", now() - g_start);
    fflush(stdout);
    return false;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <mnist-dir> <output.nn> [epochs]\n", argv[0]);
        return 1;
    }
    size_t epochs = argc > 3 ? (size_t)strtoul(argv[3], NULL, 10) : 40;
    g_output = argv[2];

    Split train_set = {0}, test_set = {0};
    if (!load_split(argv[1], "train-images-idx3-ubyte", "train-labels-idx1-ubyte", &train_set) ||
        !load_split(argv[1], "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte", &test_set)) {
        fprintf(stderr, "could not read MNIST from %s\n", argv[1]);
        return 1;
    }
    g_train = &train_set;
    eval_outputs = malloc((size_t)test_set.count * 10 * sizeof(float));

    Stream stream = {.data = &train_set, .first = 0, .count = train_set.count - VALIDATION, .rng = {2026}};
    stream.order = malloc(stream.count * sizeof(uint32_t));
    for (uint32_t i = 0; i < stream.count; i++) stream.order[i] = i;
    shuffle(&stream);

    spingalett_set_verbose(false);
#if defined(SPINGALETT_HAS_OPENMP)
    spingalett_set_compute_mode(COMPUTE_OPENMP);
#endif
    spingalett_seed(2026);

    NeuralNetwork *net = new_spingalett(LOSS_CROSS_ENTROPY);
    layer(net, DIGIT_PIXELS);
    layer(net, 1024, ACT_RELU, WEIGHT_INITIALIZATION_HE, .dropout_rate = 0.25f);
    layer(net, 512, ACT_RELU, WEIGHT_INITIALIZATION_HE, .dropout_rate = 0.25f);
    layer(net, 10, ACT_SOFTMAX, WEIGHT_INITIALIZATION_XAVIER);

    LRScheduleParams schedule = {.warmup_epochs = 2, .min_lr = 1e-5f};
    printf("training 784-1024-512-10 on %u augmented samples per epoch, %u held out\n", stream.count, VALIDATION);
    g_start = now();
    train(
        .net = net,
        .training_mode = MODE_GENERATOR_FUNCTION,
        .generator = augmented_batch,
        .generator_data = &stream,
        .sample_count = stream.count,           /* samples per epoch from the endless stream */
        .epochs = epochs,
        .training_strategy = STRATEGY_SMALL_BATCH,
        .batch_size = 128,
        .optimizer_type = OPTIMIZER_ADAMW,
        .learning_rate = 1e-3f,
        .weight_decay = 1e-2f,
        .max_grad_norm = 5.0f,
        .lr_scheduler = spingalett_lr_warmup_cosine,
        .lr_scheduler_data = &schedule,
        .callback = on_epoch,
        .callback_interval = 1
    );
    free_network(net);

    /* evaluate the selected checkpoint on the untouched test set, clean and distorted */
    NeuralNetwork *best = load_spingalett(g_output);
    if (!best) { fprintf(stderr, "could not reload %s\n", g_output); return 1; }
    float test = accuracy(best, test_set.images, test_set.labels, test_set.count);
    float *distorted = malloc((size_t)test_set.count * DIGIT_PIXELS * sizeof(float));
    DigitRng rng = {7};
    for (uint32_t i = 0; i < test_set.count; i++)
        digit_augment(test_set.images + (size_t)i * DIGIT_PIXELS, distorted + (size_t)i * DIGIT_PIXELS, &rng);
    float robust = accuracy(best, distorted, test_set.labels, test_set.count);
    printf("best epoch %zu: validation %.2f%%, test %.2f%%, distorted test %.2f%% (%.0f s)\n",
           g_best_epoch, 100.0 * (double)g_best, 100.0 * (double)test, 100.0 * (double)robust, now() - g_start);

    char info[1100];
    snprintf(info, sizeof info, "%s.info", g_output);
    FILE *f = fopen(info, "w");
    if (f) {
        fprintf(f, "784-1024-512-10 MLP, MNIST test accuracy %.2f%%\n", 100.0 * (double)test);
        fclose(f);
    }
    free_network(best);
    free(distorted);
    free(eval_outputs);
    free(stream.order);
    return 0;
}
