/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/*
 * Handwritten-digit classification on MNIST with a 784-256-128-10 multilayer perceptron.
 *
 *   Examples/download_mnist.sh data/mnist      # fetch the four IDX files once
 *   Bin/MNIST data/mnist [epochs] [st|omp|blas]
 *
 * Reports the test accuracy after every epoch and saves the trained model to mnist.nn.
 */

#include <Spingalett/Spingalett.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
    uint32_t count;
    float *images;      /* count x 784, scaled to [0, 1] */
    float *one_hot;     /* count x 10 */
    uint8_t *labels;
} Dataset;

static uint32_t read_be32(FILE *f) {
    unsigned char b[4];
    if (fread(b, 1, 4, f) != 4) return 0;
    return ((uint32_t)b[0] << 24) | ((uint32_t)b[1] << 16) | ((uint32_t)b[2] << 8) | b[3];
}

static FILE *open_in(const char *dir, const char *name) {
    char path[1024];
    snprintf(path, sizeof path, "%s/%s", dir, name);
    FILE *f = fopen(path, "rb");
    if (!f) fprintf(stderr, "cannot open %s\n", path);
    return f;
}

/* Reads an IDX image file (magic 0x803) and its label file (magic 0x801). */
static bool load_split(const char *dir, const char *images_name, const char *labels_name, Dataset *d) {
    FILE *fi = open_in(dir, images_name), *fl = open_in(dir, labels_name);
    bool ok = fi && fl;
    if (ok) {
        uint32_t magic_i = read_be32(fi), n = read_be32(fi), rows = read_be32(fi), cols = read_be32(fi);
        uint32_t magic_l = read_be32(fl), nl = read_be32(fl);
        ok = magic_i == 0x803 && magic_l == 0x801 && n == nl && rows == 28 && cols == 28;
        if (ok) {
            d->count = n;
            d->images = malloc((size_t)n * 784 * sizeof(float));
            d->one_hot = calloc((size_t)n * 10, sizeof(float));
            d->labels = malloc(n);
            unsigned char *pixels = malloc((size_t)n * 784);
            ok = d->images && d->one_hot && d->labels && pixels &&
                 fread(pixels, 1, (size_t)n * 784, fi) == (size_t)n * 784 &&
                 fread(d->labels, 1, n, fl) == n;
            for (size_t i = 0; ok && i < (size_t)n * 784; i++)
                d->images[i] = pixels[i] / 255.0f;
            for (uint32_t i = 0; ok && i < n; i++)
                d->one_hot[(size_t)i * 10 + d->labels[i]] = 1.0f;
            free(pixels);
        }
    }
    if (fi) fclose(fi);
    if (fl) fclose(fl);
    return ok;
}

static void free_split(Dataset *d) {
    free(d->images);
    free(d->one_hot);
    free(d->labels);
}

static Dataset test_set;
static float *test_outputs;
static double train_started;

static double seconds(void) {
    return (double)clock() / CLOCKS_PER_SEC;
}

static double now(void) {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static float accuracy(NeuralNetwork *net, const Dataset *d, float *outputs) {
    predict(.net = net, .inputs = d->images, .sample_count = d->count, .outputs = outputs);
    uint32_t correct = 0;
    for (uint32_t i = 0; i < d->count; i++) {
        const float *o = outputs + (size_t)i * 10;
        int best = 0;
        for (int k = 1; k < 10; k++)
            if (o[k] > o[best]) best = k;
        correct += (best == d->labels[i]);
    }
    return (float)correct / (float)d->count;
}

static bool on_epoch(NeuralNetwork *net, size_t epoch, float loss) {
    float acc = accuracy(net, &test_set, test_outputs);
    printf("epoch %2zu  loss %.4f  test accuracy %.2f%%  (%.1f s)\n",
           epoch, (double)loss, 100.0 * (double)acc, now() - train_started);
    return false;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <mnist-dir> [epochs] [st|omp|blas]\n"
                        "Download the data with Examples/download_mnist.sh <mnist-dir>.\n", argv[0]);
        return 1;
    }
    size_t epochs = argc > 2 ? (size_t)strtoul(argv[2], NULL, 10) : 10;

    ComputeMode mode = COMPUTE_SINGLE_THREADED;
#if defined(SPINGALETT_HAS_OPENMP)
    mode = COMPUTE_OPENMP;
#endif
    if (argc > 3)
        mode = !strcmp(argv[3], "blas") ? COMPUTE_OPENBLAS : !strcmp(argv[3], "omp") ? COMPUTE_OPENMP : COMPUTE_SINGLE_THREADED;

    Dataset train_set = {0};
    if (!load_split(argv[1], "train-images-idx3-ubyte", "train-labels-idx1-ubyte", &train_set) ||
        !load_split(argv[1], "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte", &test_set)) {
        fprintf(stderr, "could not read MNIST from %s (run Examples/download_mnist.sh %s)\n", argv[1], argv[1]);
        return 1;
    }
    printf("MNIST: %u training and %u test images\n", train_set.count, test_set.count);
    test_outputs = malloc((size_t)test_set.count * 10 * sizeof(float));

    spingalett_set_verbose(false);
    spingalett_set_compute_mode(mode);
    spingalett_seed(42);

    NeuralNetwork *net = new_spingalett(LOSS_CROSS_ENTROPY);
    layer(net, 784);
    layer(net, 256, ACT_RELU, WEIGHT_INITIALIZATION_HE, .dropout_rate = 0.2f);
    layer(net, 128, ACT_RELU, WEIGHT_INITIALIZATION_HE);
    layer(net, 10, ACT_SOFTMAX, WEIGHT_INITIALIZATION_XAVIER);

    double cpu0 = seconds();
    train_started = now();
    train(
        .net = net,
        .inputs = train_set.images,
        .targets = train_set.one_hot,
        .sample_count = train_set.count,
        .epochs = epochs,
        .training_strategy = STRATEGY_SMALL_BATCH,
        .batch_size = 128,
        .optimizer_type = OPTIMIZER_ADAMW,
        .learning_rate = 1e-3f,
        .weight_decay = 1e-4f,
        .lr_scheduler = spingalett_lr_cosine_decay,
        .callback = on_epoch,
        .callback_interval = 1
    );
    double wall = now() - train_started, cpu = seconds() - cpu0;

    printf("trained %zu epochs in %.1f s (%.0f samples/s, CPU time %.1f s)\n",
           epochs, wall, (double)train_set.count * (double)epochs / wall, cpu);
    printf("final test accuracy: %.2f%%\n", 100.0 * (double)accuracy(net, &test_set, test_outputs));

    save_spingalett(.net = net, .filename = "mnist.nn", .do_not_save_optimizer = true);
    free_network(net);
    free(test_outputs);
    free_split(&train_set);
    free_split(&test_set);
    return 0;
}
