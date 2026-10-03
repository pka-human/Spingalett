/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/*
 * Trains the DigitPad model: a 784-1024-512-10 MLP on MNIST with on-the-fly augmentation (random
 * rotation, scale, aspect, shear, shift and stroke thickness) fed through a data generator, so
 * the network copes with digits drawn by hand rather than only with scanned MNIST digits.
 * 5,000 training images are held out to select the best epoch, whose weights train() restores;
 * the test set is evaluated once.
 *
 *   DigitPadTrain <mnist-dir> <output.slett> [epochs]
 */

#include "Digits.h"
#include <Spingalett/Spingalett.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define VALIDATION 5000

static bool load(const char *dir, const char *images, const char *labels, SpingalettDataset *d) {
    char ipath[1024], lpath[1024];
    snprintf(ipath, sizeof ipath, "%s/%s", dir, images);
    snprintf(lpath, sizeof lpath, "%s/%s", dir, labels);
    return spingalett_load_idx(ipath, lpath, 10, d);
}

/* ---- generator: endless stream of augmented training samples, reshuffled every pass ---- */
typedef struct {
    const SpingalettDataset *data;
    uint32_t *order, pos;
    DigitRng rng;
} Stream;

static void shuffle(Stream *st) {
    for (uint32_t i = st->data->count - 1; i > 0; i--) {
        uint32_t j = (uint32_t)(digit_rng_next(&st->rng) % (i + 1));
        uint32_t t = st->order[i]; st->order[i] = st->order[j]; st->order[j] = t;
    }
}

static uint32_t augmented_batch(float *inputs, float *targets, uint32_t requested, void *user) {
    Stream *st = user;
    for (uint32_t s = 0; s < requested; s++) {
        if (st->pos == st->data->count) { shuffle(st); st->pos = 0; }
        uint32_t idx = st->order[st->pos++];
        digit_augment(st->data->inputs + (size_t)idx * DIGIT_PIXELS, inputs + (size_t)s * DIGIT_PIXELS, &st->rng);
        memcpy(targets + (size_t)s * 10, st->data->targets + (size_t)idx * 10, 10 * sizeof(float));
    }
    return requested;
}

/* ---- progress ---- */
static double now(void) {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static bool on_epoch(NeuralNetwork *net, const TrainProgress *p, void *started) {
    (void)net;
    printf("epoch %3zu  loss %.4f  validation %.2f%%%s  (%.0f s)\n", p->epoch, (double)p->train_loss,
           100.0 * (double)p->validation.accuracy, p->improved ? "  *" : "", now() - *(const double *)started);
    fflush(stdout);
    return false;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <mnist-dir> <output.slett> [epochs]\n", argv[0]);
        return 1;
    }
    size_t epochs = argc > 3 ? (size_t)strtoul(argv[3], NULL, 10) : 40;
    const char *output = argv[2];

    SpingalettDataset train_set, val_set, test_set;
    if (!load(argv[1], "train-images-idx3-ubyte", "train-labels-idx1-ubyte", &train_set) ||
        !load(argv[1], "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte", &test_set) ||
        train_set.input_size != DIGIT_PIXELS || !spingalett_dataset_split(&train_set, VALIDATION, &val_set)) {
        fprintf(stderr, "could not read MNIST from %s: %s\n", argv[1], spingalett_last_error_message());
        return 1;
    }

    Stream stream = {.data = &train_set, .rng = {2026}};
    stream.order = malloc(train_set.count * sizeof(uint32_t));
    for (uint32_t i = 0; i < train_set.count; i++) stream.order[i] = i;
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
    printf("training 784-1024-512-10 on %u augmented samples per epoch, %u held out\n", train_set.count, val_set.count);
    double started = now();
    TrainReport report = train(
        .net = net,
        .training_mode = MODE_GENERATOR_FUNCTION,
        .generator = augmented_batch,
        .generator_data = &stream,
        .sample_count = train_set.count,        /* samples per epoch from the endless stream */
        .epochs = epochs,
        .training_strategy = STRATEGY_SMALL_BATCH,
        .batch_size = 128,
        .optimizer_type = OPTIMIZER_ADAMW,
        .learning_rate = 1e-3f,
        .weight_decay = 1e-2f,
        .max_grad_norm = 5.0f,
        .lr_scheduler = spingalett_lr_warmup_cosine,
        .lr_scheduler_data = &schedule,
        .val_inputs = val_set.inputs,
        .val_targets = val_set.targets,
        .val_count = val_set.count,
        .monitor = MONITOR_VAL_ACCURACY,
        .restore_best_weights = true,
        .callback = on_epoch,
        .callback_data = &started
    );
    if (report.status == TRAIN_FAILED) {
        fprintf(stderr, "training failed: %s\n", spingalett_last_error_message());
        return 1;
    }

    /* the best epoch's weights are back in place: evaluate on the untouched test set, clean and distorted */
    EvalMetrics test = evaluate(.net = net, .inputs = test_set.inputs, .targets = test_set.targets,
                                .sample_count = test_set.count);
    float *distorted = malloc((size_t)test_set.count * DIGIT_PIXELS * sizeof(float));
    DigitRng rng = {7};
    for (uint32_t i = 0; i < test_set.count; i++)
        digit_augment(test_set.inputs + (size_t)i * DIGIT_PIXELS, distorted + (size_t)i * DIGIT_PIXELS, &rng);
    EvalMetrics robust = evaluate(.net = net, .inputs = distorted, .targets = test_set.targets,
                                  .sample_count = test_set.count);
    printf("best epoch %zu: validation %.2f%%, test %.2f%%, distorted test %.2f%% (%.0f s)\n",
           report.best_epoch, 100.0 * (double)report.best_value, 100.0 * (double)test.accuracy,
           100.0 * (double)robust.accuracy, now() - started);

    save_spingalett(.net = net, .filename = output, .do_not_save_optimizer = true);
    char info[1100];
    snprintf(info, sizeof info, "%s.info", output);
    FILE *f = fopen(info, "w");
    if (f) {
        fprintf(f, "784-1024-512-10 MLP, MNIST test accuracy %.2f%%\n", 100.0 * (double)test.accuracy);
        fclose(f);
    }
    free_network(net);
    free(distorted);
    free(stream.order);
    spingalett_dataset_free(&train_set);
    spingalett_dataset_free(&val_set);
    spingalett_dataset_free(&test_set);
    return 0;
}
