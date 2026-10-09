/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/*
 * Semantic segmentation with a U-Net (Ronneberger et al., 2015) on synthetic images: 64 x 64 color
 * images of circles, squares and triangles of random sizes, angles and colors over a shaded, noisy
 * background, every pixel to be labelled with the kind of shape it belongs to (three sigmoid outputs
 * a pixel, trained with binary cross-entropy; none above 1/2 is the background). Color tells the
 * shapes nothing: telling a circle from a square takes the context the contracting path gathers,
 * and the expanding path brings it back to full resolution through transposed convolutions,
 * concatenated with the maps of the same size on the way down:
 *
 *   64 x 64 x 3 -> [conv 3x3, BN, ReLU] x 2, 16 (e1)  -> max pool 2
 *               -> [conv 3x3, BN, ReLU] x 2, 32 (e2)  -> max pool 2
 *               -> [conv 3x3, BN, ReLU] x 2, 64       -> transposed conv 2x2 / 2, 32, ReLU
 *   concat e2   -> [conv 3x3, BN, ReLU] x 2, 32       -> transposed conv 2x2 / 2, 16, ReLU
 *   concat e1   -> [conv 3x3, BN, ReLU] x 2, 16       -> conv 1x1, 3, sigmoid
 *
 * With "bilinear", the expanding path upsamples bilinearly and convolves (3 x 3) instead; with "ln",
 * layer normalization over each pixel's channels takes the place of batch normalization (it learns
 * far more slowly here: a mean IoU of about 0.45 after 12 epochs, against 0.88 with batch
 * normalization).
 *
 *   Bin/Segmentation [epochs] [bilinear] [ln] [st|omp|gpu|bf16]
 *
 * "gpu" trains and predicts on the GPU (SPINGALETT_COMPUTE_VULKAN) when the library has the backend and finds a
 * device; "bf16" too, with the matrix products in bfloat16 on its matrix units where it has them.
 * Every epoch draws 4,096 new images (a generator function); 256 fixed ones keep the weights of the
 * best epoch. The pixel accuracy and each class's intersection over union on 512 test images
 * follow, for the network and for it as FP16 and INT8 deployment models (batch normalization folded
 * into the convolutions before it), then one test image's labels drawn as text. The network is saved
 * to segmentation.slett.
 */

#include <Spingalett/Spingalett.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#if !defined(TIME_UTC) && defined(_WIN32)
#include <windows.h>
#endif

#define SIZE    64u                     /* images of SIZE x SIZE pixels */
#define CLASSES 3u                      /* circle, square, triangle */
#define PIXELS  (SIZE * SIZE)

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

/* ---- the images ---- */

static uint64_t next(uint64_t *state) {     /* xorshift64* */
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return *state * 2685821657736338717ull;
}

static float uniform(uint64_t *state) {
    return (float)(next(state) >> 40) * (1.0f / 16777216.0f);
}

/* One image (SIZE x SIZE x 3, channels last) and its labels (SIZE x SIZE x CLASSES, one-hot or zero):
   two to five shapes, each over those drawn before it. */
static void draw(uint64_t *rng, float *image, float *labels) {
    float base[3], gx[3], gy[3];
    for (int c = 0; c < 3; c++) {
        base[c] = 0.2f + 0.6f * uniform(rng);
        gx[c] = 0.4f * (uniform(rng) - 0.5f);
        gy[c] = 0.4f * (uniform(rng) - 0.5f);
    }
    for (uint32_t y = 0; y < SIZE; y++)
        for (uint32_t x = 0; x < SIZE; x++)
            for (int c = 0; c < 3; c++)
                image[(y * SIZE + x) * 3 + c] = base[c] + gx[c] * ((float)x / SIZE - 0.5f) +
                                                gy[c] * ((float)y / SIZE - 0.5f) + 0.15f * (uniform(rng) - 0.5f);
    memset(labels, 0, PIXELS * CLASSES * sizeof(float));
    const uint32_t shapes = 2u + (uint32_t)(next(rng) % 4u);
    for (uint32_t s = 0; s < shapes; s++) {
        const uint32_t kind = (uint32_t)(next(rng) % CLASSES);
        const float cx = 6.0f + uniform(rng) * (SIZE - 12.0f), cy = 6.0f + uniform(rng) * (SIZE - 12.0f);
        const float r = 5.0f + 8.0f * uniform(rng), angle = 6.2831853f * uniform(rng);
        const float co = cosf(angle), si = sinf(angle);
        float color[3];
        for (int c = 0; c < 3; c++) color[c] = uniform(rng);
        /* the triangle's sides: outward normals at angle + pi + 2 pi k / 3, half its circumradius away */
        float nx[3], ny[3];
        for (int k = 0; k < 3; k++) {
            nx[k] = cosf(angle + 3.1415927f + 2.0943951f * (float)k);
            ny[k] = sinf(angle + 3.1415927f + 2.0943951f * (float)k);
        }
        for (uint32_t y = 0; y < SIZE; y++)
            for (uint32_t x = 0; x < SIZE; x++) {
                const float dx = (float)x + 0.5f - cx, dy = (float)y + 0.5f - cy;
                bool inside;
                if (kind == 0) {
                    inside = dx * dx + dy * dy <= r * r;
                } else if (kind == 1) {
                    const float u = dx * co + dy * si, v = dy * co - dx * si;
                    inside = fabsf(u) <= 0.85f * r && fabsf(v) <= 0.85f * r;
                } else {
                    inside = true;
                    for (int k = 0; k < 3; k++) inside = inside && dx * nx[k] + dy * ny[k] <= 0.5f * r;
                }
                if (!inside) continue;
                float *px = image + (y * SIZE + x) * 3, *lab = labels + (y * SIZE + x) * CLASSES;
                for (int c = 0; c < 3; c++) px[c] = color[c] + 0.15f * (uniform(rng) - 0.5f);
                for (uint32_t k = 0; k < CLASSES; k++) lab[k] = k == kind ? 1.0f : 0.0f;
            }
    }
}

/* SpingalettDataGeneratorFn: new images every time */
static uint32_t generate(float *inputs, float *targets, uint32_t requested, void *rng) {
    for (uint32_t i = 0; i < requested; i++)
        draw((uint64_t *)rng, inputs + (size_t)i * PIXELS * 3, targets + (size_t)i * PIXELS * CLASSES);
    return requested;
}

/* ---- the network ---- */

/* Two 3 x 3 convolutions of `filters` outputs, each normalized and rectified; the last layer's index. */
static uint32_t double_conv(SpingalettNetwork *net, uint32_t filters, bool ln) {
    uint32_t last = 0;
    for (int k = 0; k < 2; k++) {
        spingalett_conv2d(.net = net, .filters = filters, .kernel = 3, .padding = 1, .act_func = SPINGALETT_ACT_NONE,
                          .weight_initialization = SPINGALETT_INIT_HE);
        last = ln ? spingalett_layer_norm(.net = net, .act_func = SPINGALETT_ACT_RELU)
                  : spingalett_batch_norm(.net = net, .act_func = SPINGALETT_ACT_RELU);
    }
    return last;
}

/* Twice the size, `filters` channels: a transposed convolution, or bilinear upsampling and a 3 x 3
   convolution. */
static uint32_t up(SpingalettNetwork *net, uint32_t filters, bool bilinear) {
    if (!bilinear)
        return spingalett_conv_transpose2d(.net = net, .filters = filters, .kernel = 2, .stride = 2,
                                           .act_func = SPINGALETT_ACT_RELU,
                                           .weight_initialization = SPINGALETT_INIT_HE);
    spingalett_upsample2d(.net = net, .stride = 2, .upsample = SPINGALETT_UPSAMPLE_BILINEAR);
    return spingalett_conv2d(.net = net, .filters = filters, .kernel = 3, .padding = 1, .act_func = SPINGALETT_ACT_RELU,
                             .weight_initialization = SPINGALETT_INIT_HE);
}

static SpingalettNetwork *unet(bool bilinear, bool ln) {
    SpingalettNetwork *net = spingalett_network_new(SPINGALETT_LOSS_CROSS_ENTROPY);
    spingalett_layer(.net = net, .height = SIZE, .width = SIZE, .channels = 3);
    uint32_t e1 = double_conv(net, 16, ln);
    spingalett_max_pool2d(.net = net, .kernel = 2);
    uint32_t e2 = double_conv(net, 32, ln);
    spingalett_max_pool2d(.net = net, .kernel = 2);
    double_conv(net, 64, ln);
    uint32_t u = up(net, 32, bilinear);
    spingalett_concat_layers(.net = net, .inputs = {e2, u}, .act_func = SPINGALETT_ACT_NONE);
    double_conv(net, 32, ln);
    u = up(net, 16, bilinear);
    spingalett_concat_layers(.net = net, .inputs = {e1, u}, .act_func = SPINGALETT_ACT_NONE);
    double_conv(net, 16, ln);
    spingalett_conv2d(.net = net, .filters = CLASSES, .kernel = 1, .act_func = SPINGALETT_ACT_SIGMOID,
                      .weight_initialization = SPINGALETT_INIT_XAVIER);
    return net;
}

/* ---- scores ---- */

/* A pixel's label: 1 + the class of its largest output when that is over 1/2, 0 (the background)
   otherwise. */
static uint32_t label(const float *p) {
    uint32_t best = 0;
    for (uint32_t k = 1; k < CLASSES; k++)
        if (p[k] > p[best]) best = k;
    return p[best] > 0.5f ? best + 1u : 0u;
}

/* Pixel accuracy, and the intersection over union of each class's pixels with the true ones. */
static double score(const float *outputs, const float *truth, uint32_t count, double iou[CLASSES]) {
    uint64_t right = 0, both[CLASSES] = {0}, either[CLASSES] = {0};
    for (uint64_t i = 0; i < (uint64_t)count * PIXELS; i++) {
        const uint32_t p = label(outputs + i * CLASSES), t = label(truth + i * CLASSES);
        right += p == t;
        for (uint32_t k = 0; k < CLASSES; k++) {
            both[k] += p == k + 1u && t == k + 1u;
            either[k] += p == k + 1u || t == k + 1u;
        }
    }
    for (uint32_t k = 0; k < CLASSES; k++) iou[k] = either[k] ? (double)both[k] / (double)either[k] : 1.0;
    return (double)right / ((double)count * PIXELS);
}

static void report(const char *name, const float *outputs, const float *truth, uint32_t count, double seconds) {
    double iou[CLASSES];
    const double accuracy = score(outputs, truth, count, iou);
    printf("%-14s pixel accuracy %.2f%%, IoU circle %.3f, square %.3f, triangle %.3f, mean %.3f  (%.0f images/s)\n",
           name, 100.0 * accuracy, iou[0], iou[1], iou[2], (iou[0] + iou[1] + iou[2]) / 3.0, count / seconds);
}

static bool on_epoch(SpingalettNetwork *net, const SpingalettTrainProgress *p, void *started) {
    (void)net;
    printf("epoch %2zu  lr %.5f  loss %.4f  validation loss %.4f%s  (%.0f s)\n", p->epoch, (double)p->learning_rate,
           (double)p->train_loss, (double)p->validation.loss, p->improved ? "  *" : "",
           now() - *(const double *)started);
    fflush(stdout);
    return false;
}

int main(int argc, char **argv) {
    size_t epochs = 12;
    bool bilinear = false, ln = false;
    SpingalettComputeMode mode = SPINGALETT_COMPUTE_SINGLE_THREADED;
#if defined(SPINGALETT_HAS_OPENMP)
    mode = SPINGALETT_COMPUTE_OPENMP;
#endif
    for (int i = 1; i < argc; i++) {
        if (argv[i][0] >= '0' && argv[i][0] <= '9') epochs = (size_t)strtoul(argv[i], NULL, 10);
        else if (!strcmp(argv[i], "bilinear")) bilinear = true;
        else if (!strcmp(argv[i], "ln")) ln = true;
        else if (!strcmp(argv[i], "omp")) mode = SPINGALETT_COMPUTE_OPENMP;
        else if (!strcmp(argv[i], "st")) mode = SPINGALETT_COMPUTE_SINGLE_THREADED;
        else if (!strcmp(argv[i], "gpu")) mode = SPINGALETT_COMPUTE_VULKAN;
        else if (!strcmp(argv[i], "bf16"))
            mode = SPINGALETT_COMPUTE_VULKAN, spingalett_set_gpu_precision(SPINGALETT_PRECISION_BFLOAT16);
        else {
            fprintf(stderr, "usage: %s [epochs] [bilinear] [ln] [st|omp|gpu|bf16]\n", argv[0]);
            return 1;
        }
    }
    spingalett_set_verbose(false);
    spingalett_set_compute_mode(mode);
    if (mode == SPINGALETT_COMPUTE_VULKAN)
        printf("GPU: %s\n", spingalett_gpu_device() ? spingalett_gpu_device() : "none (the CPU)");

    /* fixed validation and test images, from generators of their own */
    const uint32_t val_count = 256, test_count = 512;
    float *val_x = (float *)malloc((size_t)val_count * PIXELS * 3 * sizeof(float));
    float *val_t = (float *)malloc((size_t)val_count * PIXELS * CLASSES * sizeof(float));
    float *test_x = (float *)malloc((size_t)test_count * PIXELS * 3 * sizeof(float));
    float *test_t = (float *)malloc((size_t)test_count * PIXELS * CLASSES * sizeof(float));
    float *test_y = (float *)malloc((size_t)test_count * PIXELS * CLASSES * sizeof(float));
    if (!val_x || !val_t || !test_x || !test_t || !test_y) {
        fprintf(stderr, "out of memory\n");
        return 1;
    }
    uint64_t val_rng = 0x5eed0001, test_rng = 0x5eed0002, train_rng = 0x5eed0003;
    generate(val_x, val_t, val_count, &val_rng);
    generate(test_x, test_t, test_count, &test_rng);

    spingalett_seed(42);
    SpingalettNetwork *net = unet(bilinear, ln);
    printf("U-Net (%s, %s): %u layers, %llu parameters\n", bilinear ? "bilinear upsampling" : "transposed convolutions",
           ln ? "layer normalization" : "batch normalization", spingalett_layer_count(net),
           (unsigned long long)spingalett_parameter_count(net));

    SpingalettLRScheduleParams schedule = {.warmup_epochs = 1, .min_lr = 1e-5f};
    double started = now();
    SpingalettTrainReport result = spingalett_train(
        .net = net,
        .training_mode = SPINGALETT_MODE_GENERATOR_FUNCTION,
        .generator = generate,
        .generator_data = &train_rng,
        .sample_count = 4096,
        .epochs = epochs,
        .training_strategy = SPINGALETT_STRATEGY_SMALL_BATCH,
        .batch_size = 32,
        .optimizer_type = SPINGALETT_OPTIMIZER_ADAMW,
        .learning_rate = 3e-3f,
        .weight_decay = 1e-4f,
        .lr_scheduler = spingalett_lr_warmup_cosine,
        .lr_scheduler_data = &schedule,
        .val_inputs = val_x,
        .val_targets = val_t,
        .val_count = val_count,
        .monitor = SPINGALETT_MONITOR_VAL_LOSS,
        .restore_best_weights = true,
        .callback = on_epoch,
        .callback_data = &started
    );
    const double wall = now() - started;
    if (result.status == SPINGALETT_TRAIN_FAILED) {
        fprintf(stderr, "training failed: %s\n", spingalett_last_error_message());
        return 1;
    }
    printf("trained %zu epochs in %.0f s (%.0f images/s); kept epoch %zu\n", result.epochs_run, wall,
           4096.0 * (double)result.epochs_run / wall, result.best_epoch);

    /* timed the second time: the first chooses the GPU's tiles for these shapes */
    spingalett_predict(.net = net, .inputs = test_x, .outputs = test_y, .sample_count = test_count);
    double t0 = now();
    spingalett_predict(.net = net, .inputs = test_x, .outputs = test_y, .sample_count = test_count);
    report("network", test_y, test_t, test_count, now() - t0);

    /* deployment models, on the CPU */
    const SpingalettPrecisionMode precisions[] = {SPINGALETT_PRECISION_FP16, SPINGALETT_PRECISION_INT8};
    const char *names[] = {"FP16 model", "INT8 model"};
    float *model_y = (float *)malloc((size_t)test_count * PIXELS * CLASSES * sizeof(float));
    for (int q = 0; q < 2 && model_y; q++) {
        SpingalettModel *model = spingalett_model_from_network(net, precisions[q]);
        if (!model) continue;
        t0 = now();
        if (spingalett_model_predict(model, test_x, test_count, model_y))
            report(names[q], model_y, test_t, test_count, now() - t0);
        spingalett_model_free(model);
    }
    free(model_y);

    /* the first test image's labels, every other pixel: truth | prediction */
    static const char marks[] = ".o#^";
    printf("\n%-34s%s\n", "truth (. o # ^)", "prediction");
    for (uint32_t y = 0; y < SIZE; y += 2) {
        char line[2 * (SIZE / 2) + 4];
        size_t at = 0;
        for (int side = 0; side < 2; side++) {
            const float *src = side ? test_y : test_t;
            for (uint32_t x = 0; x < SIZE; x += 2) line[at++] = marks[label(src + (y * SIZE + x) * CLASSES)];
            if (!side) { line[at++] = ' '; line[at++] = ' '; }
        }
        line[at] = '\0';
        printf("%s\n", line);
    }

    spingalett_save(.net = net, .filename = "segmentation.slett", .do_not_save_optimizer = true);
    spingalett_network_free(net);
    free(val_x); free(val_t); free(test_x); free(test_t); free(test_y);
    return 0;
}
