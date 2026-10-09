/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/*
 * Handwritten-digit classification on MNIST with a small convolutional network:
 *
 *   28 x 28 x 1 -> conv 3x3, 32 filters, ReLU -> max pool 2 -> conv 3x3, 64 filters, ReLU
 *               -> max pool 2 -> dense 128, ReLU, dropout -> dense 10, softmax
 *
 *   Examples/download_mnist.sh data/mnist      # fetch the four IDX files once
 *   Bin/MNIST_CNN data/mnist [epochs] [st|omp|blas]
 *
 * The images go in as they are: an MNIST sample of 784 floats is a 28 x 28 x 1 tensor. Holds out
 * 5,000 training images for validation, keeps the weights of the epoch with the best validation
 * accuracy, reports the test accuracy at the end, shows what quantizing the model for deployment
 * costs in accuracy and gains in size, and saves it to mnist_cnn.slett (format version 4: the
 * inference engine runs it, on microcontrollers too; see Examples/Embedded).
 */

#include <Spingalett/Spingalett.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#if !defined(TIME_UTC) && defined(_WIN32)
#include <windows.h>
#endif

static double now(void) {
#if defined(TIME_UTC)                    /* C11 timespec_get; some Windows C runtimes lack it */
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

static bool load(const char *dir, const char *images, const char *labels, SpingalettDataset *d) {
    char ipath[1024], lpath[1024];
    snprintf(ipath, sizeof ipath, "%s/%s", dir, images);
    snprintf(lpath, sizeof lpath, "%s/%s", dir, labels);
    return spingalett_load_idx(ipath, lpath, 10, d);
}

static bool on_epoch(SpingalettNetwork *net, const SpingalettTrainProgress *p, void *started) {
    (void)net;
    printf("epoch %2zu  loss %.4f  validation loss %.4f  accuracy %.2f%%%s  (%.1f s)\n",
           p->epoch, (double)p->train_loss, (double)p->validation.loss, 100.0 * (double)p->validation.accuracy,
           p->improved ? "  *" : "", now() - *(const double *)started);
    return false;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <mnist-dir> [epochs] [st|omp|blas]\n"
                        "Download the data with Examples/download_mnist.sh <mnist-dir>.\n", argv[0]);
        return 1;
    }
    size_t epochs = argc > 2 ? (size_t)strtoul(argv[2], NULL, 10) : 5;

    SpingalettComputeMode mode = SPINGALETT_COMPUTE_SINGLE_THREADED;
#if defined(SPINGALETT_HAS_OPENMP)
    mode = SPINGALETT_COMPUTE_OPENMP;
#endif
    if (argc > 3)
        mode = !strcmp(argv[3], "blas")  ? SPINGALETT_COMPUTE_OPENBLAS
               : !strcmp(argv[3], "omp") ? SPINGALETT_COMPUTE_OPENMP
                                         : SPINGALETT_COMPUTE_SINGLE_THREADED;

    SpingalettDataset train_set, val_set, test_set;
    if (!load(argv[1], "train-images-idx3-ubyte", "train-labels-idx1-ubyte", &train_set) ||
        !load(argv[1], "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte", &test_set) ||
        !spingalett_dataset_split(&train_set, 5000, &val_set)) {
        fprintf(stderr, "could not read MNIST from %s (run Examples/download_mnist.sh %s): %s\n",
                argv[1], argv[1], spingalett_last_error_message());
        return 1;
    }
    printf("MNIST: %u training, %u validation and %u test images\n", train_set.count, val_set.count, test_set.count);

    spingalett_set_verbose(false);
    spingalett_set_compute_mode(mode);
    spingalett_seed(42);

    SpingalettNetwork *net = spingalett_network_new(SPINGALETT_LOSS_CROSS_ENTROPY);
    spingalett_layer(.net = net, .height = 28, .width = 28, .channels = 1);
    spingalett_conv2d(.net = net, .filters = 32, .kernel = 3, .padding = 1, .act_func = SPINGALETT_ACT_RELU,
                      .weight_initialization = SPINGALETT_INIT_HE);
    spingalett_max_pool2d(.net = net, .kernel = 2);
    spingalett_conv2d(.net = net, .filters = 64, .kernel = 3, .padding = 1, .act_func = SPINGALETT_ACT_RELU,
                      .weight_initialization = SPINGALETT_INIT_HE);
    spingalett_max_pool2d(.net = net, .kernel = 2);
    spingalett_layer(net, 128, SPINGALETT_ACT_RELU, SPINGALETT_INIT_HE, .dropout_rate = 0.3f);
    spingalett_layer(net, 10, SPINGALETT_ACT_SOFTMAX, SPINGALETT_INIT_XAVIER);
    printf("network: %llu parameters\n", (unsigned long long)spingalett_parameter_count(net));

    double started = now();
    SpingalettTrainReport report = spingalett_train(
        .net = net,
        .inputs = train_set.inputs,
        .targets = train_set.targets,
        .sample_count = train_set.count,
        .epochs = epochs,
        .training_strategy = SPINGALETT_STRATEGY_SMALL_BATCH,
        .batch_size = 128,
        .optimizer_type = SPINGALETT_OPTIMIZER_ADAMW,
        .learning_rate = 1e-3f,
        .weight_decay = 1e-4f,
        .lr_scheduler = spingalett_lr_cosine_decay,
        .val_inputs = val_set.inputs,
        .val_targets = val_set.targets,
        .val_count = val_set.count,
        .monitor = SPINGALETT_MONITOR_VAL_ACCURACY,
        .restore_best_weights = true,
        .callback = on_epoch,
        .callback_data = &started
    );
    double wall = now() - started;
    if (report.status == SPINGALETT_TRAIN_FAILED) {
        fprintf(stderr, "training failed: %s\n", spingalett_last_error_message());
        return 1;
    }

    SpingalettEvalMetrics test = spingalett_evaluate(.net = net, .inputs = test_set.inputs, .targets = test_set.targets,
                                                     .sample_count = test_set.count);
    printf("trained %zu epochs in %.1f s (%.0f samples/s); kept epoch %zu (validation accuracy %.2f%%)\n",
           report.epochs_run, wall, (double)train_set.count * (double)report.epochs_run / wall,
           report.best_epoch, 100.0 * (double)report.best_value);
    printf("test accuracy: %.2f%%  (test loss %.4f)\n", 100.0 * (double)test.accuracy, (double)test.loss);

    /* the same network as a deployment model in other precisions (see ModelTool eval) */
    const SpingalettPrecisionMode quantized[] = {SPINGALETT_PRECISION_FP16, SPINGALETT_PRECISION_INT8,
                                                 SPINGALETT_PRECISION_INT4};
    const char *names[] = {"FP16", "INT8", "INT4"};
    for (int q = 0; q < 3; q++) {
        SpingalettModel *model = spingalett_model_from_network(net, quantized[q]);
        if (!model) continue;
        SpingalettEvalMetrics m = spingalett_model_evaluate(model, test_set.inputs, test_set.targets, test_set.count);
        printf("%s model: %zu bytes, test accuracy %.2f%%\n", names[q], model->image_size, 100.0 * (double)m.accuracy);
        spingalett_model_free(model);
    }

    spingalett_save(.net = net, .filename = "mnist_cnn.slett", .do_not_save_optimizer = true);
    spingalett_network_free(net);
    spingalett_dataset_free(&train_set);
    spingalett_dataset_free(&val_set);
    spingalett_dataset_free(&test_set);
    return 0;
}
