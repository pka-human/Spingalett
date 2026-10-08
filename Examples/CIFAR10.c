/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/*
 * Image classification on CIFAR-10 (32 x 32 color images of ten classes) with a convolutional
 * network using batch normalization:
 *
 *   32 x 32 x 3 -> [conv 3x3 -> batch norm, ReLU] x 2, 32 filters -> max pool 2
 *               -> [conv 3x3 -> batch norm, ReLU] x 2, 64 filters -> max pool 2
 *               -> [conv 3x3 -> batch norm, ReLU] x 2, 128 filters -> max pool 2
 *               -> dense 128, batch norm, ReLU, dropout -> dense 10, softmax
 *
 * With "separable", every 3 x 3 convolution after the first is depthwise-separable (a depthwise 3 x 3
 * convolution, then a pointwise 1 x 1 one, each normalized): far fewer parameters and
 * multiply-adds, for some accuracy.
 *
 * With "resnet20" (or resnet32, resnet44, resnet56: 6n + 2 layers), a residual network as He et al.
 * built it for CIFAR-10: a normalized 3 x 3 convolution of 16 filters, then three stages of n blocks
 * of two normalized 3 x 3 convolutions with 16, 32 and 64 filters, each block adding its input to its
 * output before the ReLU (the first block of the second and third stages halves the size with a
 * stride of 2, and its shortcut is a normalized 1 x 1 convolution of stride 2), global average
 * pooling and a dense softmax layer. "wide" doubles the filters. It trains with SGD and momentum
 * (lr 0.1 after a warm-up, cosine decay, weight decay 5e-4) and label smoothing of 0.1.
 *
 *   Examples/download_cifar10.sh data/cifar10      # fetch the binary batches once
 *   Bin/CIFAR10 data/cifar10 [epochs] [separable | resnet20 | resnet32 ... [wide]] [st|omp|blas|gpu]
 *
 * "gpu" trains and evaluates on the GPU (COMPUTE_VULKAN), when the library has the backend and finds
 * a device.
 *
 * Training augments the images with random shifts of up to 4 pixels and mirror images. 5,000 training
 * images are held out to keep the weights of the best epoch; the test accuracy follows, then that of
 * the network as FP16 and INT8 deployment models, in which each normalization is folded into the
 * convolution before it. The network is saved to cifar10.slett.
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

static bool load(const char *dir, bool train, SpingalettDataset *d) {
    char paths[5][1024];
    const char *list[5];
    uint32_t count = train ? 5u : 1u;
    for (uint32_t i = 0; i < count; i++) {
        if (train) snprintf(paths[i], sizeof paths[i], "%s/cifar-10-batches-bin/data_batch_%u.bin", dir, i + 1);
        else       snprintf(paths[i], sizeof paths[i], "%s/cifar-10-batches-bin/test_batch.bin", dir);
        list[i] = paths[i];
    }
    return spingalett_load_cifar(list, count, 10, d);
}

/* A 3 x 3 convolution of `filters` outputs, normalized and rectified; depthwise-separable on request. */
static void conv_block(NeuralNetwork *net, uint32_t filters, bool separable) {
    if (separable) {
        SpingalettNetworkLayer last;
        spingalett_network_layer(net, spingalett_layer_count(net) - 1, &last);
        conv2d(.net = net, .filters = last.channels, .kernel = 3, .padding = 1, .groups = last.channels,
               .act_func = ACT_NONE, .weight_initialization = WEIGHT_INITIALIZATION_HE);
        batch_norm(.net = net, .act_func = ACT_RELU);
        conv2d(.net = net, .filters = filters, .kernel = 1, .act_func = ACT_NONE,
               .weight_initialization = WEIGHT_INITIALIZATION_HE);
    } else {
        conv2d(.net = net, .filters = filters, .kernel = 3, .padding = 1, .act_func = ACT_NONE,
               .weight_initialization = WEIGHT_INITIALIZATION_HE);
    }
    batch_norm(.net = net, .act_func = ACT_RELU);
}

/* A residual block reading x: two normalized 3 x 3 convolutions, the first of the given stride, and
   x itself added before the ReLU (through a normalized 1 x 1 convolution where the shape changes). */
static uint32_t residual_block(NeuralNetwork *net, uint32_t x, uint32_t filters, uint32_t stride) {
    SpingalettNetworkLayer in;
    spingalett_network_layer(net, x, &in);
    conv2d(.net = net, .inputs = {x}, .filters = filters, .kernel = 3, .padding = 1, .stride = stride,
           .act_func = ACT_NONE, .weight_initialization = WEIGHT_INITIALIZATION_HE);
    batch_norm(.net = net, .act_func = ACT_RELU);
    conv2d(.net = net, .filters = filters, .kernel = 3, .padding = 1, .act_func = ACT_NONE,
           .weight_initialization = WEIGHT_INITIALIZATION_HE);
    uint32_t y = batch_norm(.net = net, .act_func = ACT_NONE);
    uint32_t shortcut = x;
    if (stride != 1 || in.channels != filters) {
        conv2d(.net = net, .inputs = {x}, .filters = filters, .kernel = 1, .stride = stride, .act_func = ACT_NONE,
               .weight_initialization = WEIGHT_INITIALIZATION_HE);
        shortcut = batch_norm(.net = net, .act_func = ACT_NONE);
    }
    return add_layers(.net = net, .inputs = {shortcut, y}, .act_func = ACT_RELU);
}

static bool on_epoch(NeuralNetwork *net, const TrainProgress *p, void *started) {
    (void)net;
    printf("epoch %2zu  lr %.5f  loss %.4f  validation loss %.4f  accuracy %.2f%%%s  (%.0f s)\n", p->epoch,
           (double)p->learning_rate, (double)p->train_loss, (double)p->validation.loss,
           100.0 * (double)p->validation.accuracy, p->improved ? "  *" : "", now() - *(const double *)started);
    fflush(stdout);                     /* progress shows when the output goes to a file */
    return false;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <cifar10-dir> [epochs] [separable | resnet20 | resnet32 | resnet44 | resnet56 [wide]] "
                        "[st|omp|blas]\n"
                        "Download the data with Examples/download_cifar10.sh <cifar10-dir>.\n", argv[0]);
        return 1;
    }
    size_t epochs = argc > 2 ? (size_t)strtoul(argv[2], NULL, 10) : 20;
    bool separable = false, wide = false;
    uint32_t depth = 0;                 /* a residual network of this many layers, 0 for none */
    ComputeMode mode = COMPUTE_SINGLE_THREADED;
#if defined(SPINGALETT_HAS_OPENMP)
    mode = COMPUTE_OPENMP;
#endif
    for (int i = 3; i < argc; i++) {
        if (!strcmp(argv[i], "separable")) separable = true;
        else if (!strncmp(argv[i], "resnet", 6)) depth = (uint32_t)strtoul(argv[i] + 6, NULL, 10);
        else if (!strcmp(argv[i], "wide")) wide = true;
        else if (!strcmp(argv[i], "blas")) mode = COMPUTE_OPENBLAS;
        else if (!strcmp(argv[i], "omp")) mode = COMPUTE_OPENMP;
        else if (!strcmp(argv[i], "st")) mode = COMPUTE_SINGLE_THREADED;
        else if (!strcmp(argv[i], "gpu")) mode = COMPUTE_VULKAN;
    }

    SpingalettDataset train_set, val_set, test_set;
    spingalett_seed(42);
    if (!load(argv[1], true, &train_set) || !load(argv[1], false, &test_set)) {
        fprintf(stderr, "could not read CIFAR-10 from %s (run Examples/download_cifar10.sh %s): %s\n",
                argv[1], argv[1], spingalett_last_error_message());
        return 1;
    }
    spingalett_dataset_shuffle(&train_set);
    if (!spingalett_dataset_split(&train_set, 5000, &val_set)) {
        fprintf(stderr, "could not hold out validation images: %s\n", spingalett_last_error_message());
        return 1;
    }
    printf("CIFAR-10: %u training, %u validation and %u test images\n", train_set.count, val_set.count, test_set.count);

    spingalett_set_verbose(false);
    spingalett_set_compute_mode(mode);
    if (mode == COMPUTE_VULKAN) printf("GPU: %s\n", spingalett_gpu_device() ? spingalett_gpu_device() : "none (the CPU)");

    if (depth && (depth < 8 || (depth - 2) % 6 != 0)) {
        fprintf(stderr, "a residual network has 6n + 2 layers (resnet20, resnet32, resnet44, resnet56, ...)\n");
        return 1;
    }
    NeuralNetwork *net = new_spingalett(LOSS_CROSS_ENTROPY);
    layer(.net = net, .height = 32, .width = 32, .channels = 3);
    if (depth) {
        const uint32_t base = wide ? 32u : 16u;
        conv2d(.net = net, .filters = base, .kernel = 3, .padding = 1, .act_func = ACT_NONE,
               .weight_initialization = WEIGHT_INITIALIZATION_HE);
        uint32_t x = batch_norm(.net = net, .act_func = ACT_RELU);
        for (uint32_t stage = 0; stage < 3; stage++)
            for (uint32_t b = 0; b < (depth - 2) / 6; b++)
                x = residual_block(net, x, base << stage, stage > 0 && b == 0 ? 2u : 1u);
        global_avg_pool2d(.net = net);
    } else {
        const uint32_t widths[3] = {32, 64, 128};
        for (int stage = 0; stage < 3; stage++) {
            conv_block(net, widths[stage], separable && stage > 0);
            conv_block(net, widths[stage], separable);
            max_pool2d(.net = net, .kernel = 2);
        }
        layer(net, 128, ACT_NONE, WEIGHT_INITIALIZATION_HE);
        batch_norm(.net = net, .act_func = ACT_RELU, .dropout_rate = 0.3f);
    }
    layer(net, 10, ACT_SOFTMAX, WEIGHT_INITIALIZATION_XAVIER);
    char name[48] = "";
    if (depth) snprintf(name, sizeof name, " (ResNet-%u%s)", depth, wide ? ", wide" : "");
    else if (separable) snprintf(name, sizeof name, " (depthwise-separable)");
    printf("network%s: %u layers, %llu parameters\n", name, spingalett_layer_count(net),
           (unsigned long long)spingalett_parameter_count(net));

    LRScheduleParams schedule = {.warmup_epochs = 1, .min_lr = 1e-5f};
    double started = now();
    TrainReport report = train(
        .net = net,
        .inputs = train_set.inputs,
        .targets = train_set.targets,
        .sample_count = train_set.count,
        .epochs = epochs,
        .training_strategy = STRATEGY_SMALL_BATCH,
        .batch_size = 128,
        .optimizer_type = depth ? OPTIMIZER_MOMENTUM : OPTIMIZER_ADAMW,
        .learning_rate = depth ? 0.1f : 2e-3f,
        .weight_decay = 5e-4f,
        .label_smoothing = depth ? 0.1f : 0.0f,
        .lr_scheduler = spingalett_lr_warmup_cosine,
        .lr_scheduler_data = &schedule,
        .augment_shift = 4,
        .augment_flip = true,
        .val_inputs = val_set.inputs,
        .val_targets = val_set.targets,
        .val_count = val_set.count,
        .monitor = MONITOR_VAL_ACCURACY,
        .restore_best_weights = true,
        .callback = on_epoch,
        .callback_data = &started
    );
    double wall = now() - started;
    if (report.status == TRAIN_FAILED) {
        fprintf(stderr, "training failed: %s\n", spingalett_last_error_message());
        return 1;
    }

    EvalMetrics test = evaluate(.net = net, .inputs = test_set.inputs, .targets = test_set.targets,
                                .sample_count = test_set.count);
    printf("trained %zu epochs in %.0f s (%.0f samples/s); kept epoch %zu (validation accuracy %.2f%%)\n",
           report.epochs_run, wall, (double)train_set.count * (double)report.epochs_run / wall,
           report.best_epoch, 100.0 * (double)report.best_value);
    printf("test accuracy: %.2f%%  (test loss %.4f)\n", 100.0 * (double)test.accuracy, (double)test.loss);

    /* deployment models: batch normalization folded into the convolutions before it */
    const PrecisionMode quantized[] = {PRECISION_FP16, PRECISION_INT8};
    const char *names[] = {"FP16", "INT8"};
    for (int q = 0; q < 2; q++) {
        SpingalettModel *model = spingalett_model_from_network(net, quantized[q]);
        if (!model) continue;
        double t0 = now();
        EvalMetrics m = spingalett_model_evaluate(model, test_set.inputs, test_set.targets, test_set.count);
        printf("%s model: %u layers, %zu bytes, test accuracy %.2f%% (%.0f images/s)\n", names[q], model->layer_count,
               model->image_size, 100.0 * (double)m.accuracy, (double)test_set.count / (now() - t0));
        spingalett_model_free(model);
    }

    save_spingalett(.net = net, .filename = "cifar10.slett", .do_not_save_optimizer = true);
    free_network(net);
    spingalett_dataset_free(&train_set);
    spingalett_dataset_free(&val_set);
    spingalett_dataset_free(&test_set);
    return 0;
}
