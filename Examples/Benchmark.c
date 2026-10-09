/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/*
 * Throughput of a 784-512-1000-10 MLP (ReLU, softmax + cross-entropy, Adam, lr 1e-3) on 20,000
 * synthetic samples, for every compute backend compiled in:
 *
 *   full batch   5 epochs, one optimizer step per epoch
 *   mini-batch   1 epoch, batches of 64 (313 optimizer steps)
 *   inference    predict() over all samples
 *
 * and then the network as a deployment model in every precision: the latency of one sample with
 * spingalett_model_run (against forward()) and the throughput of spingalett_model_predict.
 *
 * Then a convolutional network, Examples/MNIST_CNN.c's (28 x 28 x 1 -> conv 32 3x3 ReLU -> max pool
 * -> conv 64 3x3 ReLU -> max pool -> dense 128 ReLU, dropout 0.3 -> dense 10 softmax), on 10,000
 * synthetic images: one epoch of mini-batches of 128 with AdamW (lr 1e-3, weight decay 1e-4), and
 * inference; and the same network with batch normalization after each convolution and the hidden
 * dense layer (which then take no activation, the normalizations ReLU).
 *
 * Then ResNet-20 (He et al.'s residual network for CIFAR-10, Examples/CIFAR10.c resnet20) on 4,096
 * synthetic 32 x 32 x 3 images: one epoch of mini-batches of 128 with SGD and momentum, and
 * inference. Then the U-Net of Examples/Segmentation.c (transposed convolutions up, three sigmoid
 * outputs a pixel) on 1,024 synthetic 64 x 64 x 3 images: one epoch of mini-batches of 32 with AdamW
 * (lr 1e-3), and inference.
 *
 * With a usable GPU (spingalett_gpu_device()), every workload also runs with COMPUTE_VULKAN, after a
 * first run that is not timed (it makes the GPU's pipelines and times the matrix products' tiles,
 * once per process), and again with the products in bfloat16 when the GPU has matrix units for them
 * (spingalett_set_gpu_precision()); each on the host's arrays, and on data sets in the GPU's memory
 * (spingalett_device_data_new(), made before the clock starts: the rows marked "data on GPU"). Inference
 * is timed on a second call, as PyTorch's after its warm-up: the first makes the network's copy on the
 * GPU (which predict() keeps while the parameters do not change), the step that moving a PyTorch model
 * to the GPU is.
 *
 * Usage: Benchmark [threads] [gpu] ("gpu": the GPU's rows only, as Examples/benchmark_pytorch.py --cuda
 * runs them). Examples/benchmark_pytorch.py runs the same workloads in PyTorch.
 */

#include <Spingalett/Spingalett.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#if !defined(TIME_UTC) && defined(_WIN32)
#include <windows.h>
#endif

#define INPUT_SIZE  784
#define HIDDEN_1    512
#define HIDDEN_2    1000
#define OUTPUT_SIZE 10
#define SAMPLES     20000
#define EPOCHS      5
#define MINI_BATCH  64

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

static void generate_synthetic_data(float **inputs, float **targets) {
    *inputs  = (float *)malloc((size_t)SAMPLES * INPUT_SIZE  * sizeof(float));
    *targets = (float *)malloc((size_t)SAMPLES * OUTPUT_SIZE * sizeof(float));
    if (!*inputs || !*targets) {
        fprintf(stderr, "Allocation failed\n");
        exit(1);
    }
    for (size_t i = 0; i < (size_t)SAMPLES * INPUT_SIZE; i++)
        (*inputs)[i] = (float)rand() / (float)RAND_MAX;
    /* soft targets: a probability distribution per sample */
    for (size_t s = 0; s < SAMPLES; s++) {
        float sum = 0.0f, *t = *targets + s * OUTPUT_SIZE;
        for (int k = 0; k < OUTPUT_SIZE; k++) sum += t[k] = (float)rand() / (float)RAND_MAX;
        for (int k = 0; k < OUTPUT_SIZE; k++) t[k] /= sum;
    }
}

/* The GPU's rows "data on GPU": samples in data sets in its memory, as PyTorch's --cuda keeps them. */
static bool on_device;

typedef struct { SpingalettDeviceData *x, *t; } Sets;

static Sets sets_of(const float *x, uint32_t in, const float *t, uint32_t out, uint32_t n) {
    if (!on_device) return (Sets){NULL, NULL};
    Sets d = {spingalett_device_data_new(x, n, in), t ? spingalett_device_data_new(t, n, out) : NULL};
    if (!d.x || (t && !d.t)) {
        fprintf(stderr, "Not enough GPU memory for the data sets\n");
        exit(1);
    }
    return d;
}

static void sets_free(Sets d) {
    spingalett_device_data_free(d.x);
    spingalett_device_data_free(d.t);
}

static NeuralNetwork *create_network(void) {
    NeuralNetwork *net = new_spingalett(.loss_func = LOSS_CROSS_ENTROPY);
    layer(.net = net, .neurons_amount = INPUT_SIZE);
    layer(.net = net, .neurons_amount = HIDDEN_1, .act_func = ACT_RELU, .weight_initialization = WEIGHT_INITIALIZATION_HE);
    layer(.net = net, .neurons_amount = HIDDEN_2, .act_func = ACT_RELU, .weight_initialization = WEIGHT_INITIALIZATION_HE);
    layer(.net = net, .neurons_amount = OUTPUT_SIZE, .act_func = ACT_SOFTMAX, .weight_initialization = WEIGHT_INITIALIZATION_XAVIER);
    return net;
}

static double train_throughput(const float *inputs, const float *targets, TrainingStrategy strategy, size_t epochs) {
    NeuralNetwork *net = create_network();
    Sets d = sets_of(inputs, INPUT_SIZE, targets, OUTPUT_SIZE, SAMPLES);
    double start = now();
    train(.net = net, .inputs = d.x ? NULL : inputs, .targets = d.t ? NULL : targets, .device_inputs = d.x,
          .device_targets = d.t, .sample_count = SAMPLES, .epochs = epochs,
          .learning_rate = 0.001f, .optimizer_type = OPTIMIZER_ADAM, .training_strategy = strategy,
          .batch_size = MINI_BATCH);
    double elapsed = now() - start;
    sets_free(d);
    free_network(net);
    return (double)SAMPLES * (double)epochs / elapsed;
}

static double inference_throughput(const float *inputs) {
    NeuralNetwork *net = create_network();
    float *outputs = (float *)malloc((size_t)SAMPLES * OUTPUT_SIZE * sizeof(float));
    Sets d = sets_of(inputs, INPUT_SIZE, NULL, OUTPUT_SIZE, SAMPLES);
    double start = 0.0;
    for (int timed = 0; timed < 2; timed++) {
        start = now();
        predict(.net = net, .inputs = d.x ? NULL : inputs, .device_inputs = d.x, .sample_count = SAMPLES,
                .outputs = outputs);
    }
    double elapsed = now() - start;
    sets_free(d);
    free(outputs);
    free_network(net);
    return SAMPLES / elapsed;
}

#define CNN_SAMPLES 10000
#define CNN_BATCH   128

static NeuralNetwork *create_cnn(bool normalized) {
    ActivationFunction act = normalized ? ACT_NONE : ACT_RELU;
    NeuralNetwork *net = new_spingalett(.loss_func = LOSS_CROSS_ENTROPY);
    layer(.net = net, .height = 28, .width = 28, .channels = 1);
    conv2d(.net = net, .filters = 32, .kernel = 3, .padding = 1, .act_func = act,
           .weight_initialization = WEIGHT_INITIALIZATION_HE);
    if (normalized) batch_norm(.net = net, .act_func = ACT_RELU);
    max_pool2d(.net = net, .kernel = 2);
    conv2d(.net = net, .filters = 64, .kernel = 3, .padding = 1, .act_func = act,
           .weight_initialization = WEIGHT_INITIALIZATION_HE);
    if (normalized) batch_norm(.net = net, .act_func = ACT_RELU);
    max_pool2d(.net = net, .kernel = 2);
    layer(.net = net, .neurons_amount = 128, .act_func = act, .weight_initialization = WEIGHT_INITIALIZATION_HE,
          .dropout_rate = normalized ? 0.0f : 0.3f);
    if (normalized) batch_norm(.net = net, .act_func = ACT_RELU, .dropout_rate = 0.3f);
    layer(.net = net, .neurons_amount = 10, .act_func = ACT_SOFTMAX, .weight_initialization = WEIGHT_INITIALIZATION_XAVIER);
    return net;
}

static void cnn_benchmark(const char *name, ComputeMode mode, bool normalized, const float *images,
                          const float *labels) {
    spingalett_set_compute_mode(mode);
    for (int untimed = mode == COMPUTE_VULKAN; untimed >= 0; untimed--) {   /* the GPU's first run */
    NeuralNetwork *net = create_cnn(normalized);
    Sets d = sets_of(images, INPUT_SIZE, labels, OUTPUT_SIZE, CNN_SAMPLES);
    double start = now();
    train(.net = net, .inputs = d.x ? NULL : images, .targets = d.t ? NULL : labels, .device_inputs = d.x,
          .device_targets = d.t, .sample_count = CNN_SAMPLES, .epochs = 1,
          .learning_rate = 1e-3f, .weight_decay = 1e-4f, .optimizer_type = OPTIMIZER_ADAMW,
          .training_strategy = STRATEGY_SMALL_BATCH, .batch_size = CNN_BATCH);
    double trained = now() - start;
    float *outputs = (float *)malloc((size_t)CNN_SAMPLES * OUTPUT_SIZE * sizeof(float));
    for (int timed = 0; timed < 2; timed++) {
        start = now();
        predict(.net = net, .inputs = d.x ? NULL : images, .device_inputs = d.x, .sample_count = CNN_SAMPLES,
                .outputs = outputs);
    }
    double inferred = now() - start;
    if (!untimed) printf("%-24s %14.0f %14.0f\n", name, CNN_SAMPLES / trained, CNN_SAMPLES / inferred);
    sets_free(d);
    free(outputs);
    free_network(net);
    }
}

#define RESNET_SAMPLES 4096

/* ResNet-20: a residual block reads x, two normalized 3 x 3 convolutions add to it (a 1 x 1
   projection where the shape changes) before the ReLU. */
static uint32_t residual_block(NeuralNetwork *net, uint32_t x, uint32_t filters, uint32_t stride) {
    SpingalettNetworkLayer in;
    spingalett_network_layer(net, x, &in);
    conv2d(.net = net, .inputs = {x}, .filters = filters, .kernel = 3, .padding = 1, .stride = stride,
           .act_func = ACT_NONE, .weight_initialization = WEIGHT_INITIALIZATION_HE);
    batch_norm(.net = net, .act_func = ACT_RELU);
    conv2d(.net = net, .filters = filters, .kernel = 3, .padding = 1, .act_func = ACT_NONE,
           .weight_initialization = WEIGHT_INITIALIZATION_HE);
    uint32_t y = batch_norm(.net = net, .act_func = ACT_NONE), shortcut = x;
    if (stride != 1 || in.channels != filters) {
        conv2d(.net = net, .inputs = {x}, .filters = filters, .kernel = 1, .stride = stride, .act_func = ACT_NONE,
               .weight_initialization = WEIGHT_INITIALIZATION_HE);
        shortcut = batch_norm(.net = net, .act_func = ACT_NONE);
    }
    return add_layers(.net = net, .inputs = {shortcut, y}, .act_func = ACT_RELU);
}

static NeuralNetwork *create_resnet20(void) {
    NeuralNetwork *net = new_spingalett(.loss_func = LOSS_CROSS_ENTROPY);
    layer(.net = net, .height = 32, .width = 32, .channels = 3);
    conv2d(.net = net, .filters = 16, .kernel = 3, .padding = 1, .act_func = ACT_NONE,
           .weight_initialization = WEIGHT_INITIALIZATION_HE);
    uint32_t x = batch_norm(.net = net, .act_func = ACT_RELU);
    for (uint32_t stage = 0; stage < 3; stage++)
        for (uint32_t b = 0; b < 3; b++) x = residual_block(net, x, 16u << stage, stage > 0 && b == 0 ? 2u : 1u);
    global_avg_pool2d(.net = net);
    layer(.net = net, .neurons_amount = 10, .act_func = ACT_SOFTMAX, .weight_initialization = WEIGHT_INITIALIZATION_XAVIER);
    return net;
}

static void resnet_benchmark(const char *name, ComputeMode mode, const float *images, const float *labels) {
    spingalett_set_compute_mode(mode);
    for (int untimed = mode == COMPUTE_VULKAN; untimed >= 0; untimed--) {   /* the GPU's first run */
    NeuralNetwork *net = create_resnet20();
    Sets d = sets_of(images, 3072u, labels, OUTPUT_SIZE, RESNET_SAMPLES);
    double start = now();
    train(.net = net, .inputs = d.x ? NULL : images, .targets = d.t ? NULL : labels, .device_inputs = d.x,
          .device_targets = d.t, .sample_count = RESNET_SAMPLES, .epochs = 1,
          .learning_rate = 0.1f, .momentum = 0.9f, .weight_decay = 5e-4f, .optimizer_type = OPTIMIZER_MOMENTUM,
          .training_strategy = STRATEGY_SMALL_BATCH, .batch_size = CNN_BATCH);
    double trained = now() - start;
    float *outputs = (float *)malloc((size_t)RESNET_SAMPLES * OUTPUT_SIZE * sizeof(float));
    for (int timed = 0; timed < 2; timed++) {
        start = now();
        predict(.net = net, .inputs = d.x ? NULL : images, .device_inputs = d.x, .sample_count = RESNET_SAMPLES,
                .outputs = outputs);
    }
    double inferred = now() - start;
    if (!untimed) printf("%-24s %14.0f %14.0f\n", name, RESNET_SAMPLES / trained, RESNET_SAMPLES / inferred);
    sets_free(d);
    free(outputs);
    free_network(net);
    }
}

#define UNET_SAMPLES 1024
#define UNET_SIZE    64

/* Examples/Segmentation.c's U-Net: two normalized 3 x 3 convolutions a level, transposed
   convolutions up, the maps of the way down concatenated. */
static uint32_t double_conv(NeuralNetwork *net, uint32_t filters) {
    uint32_t last = 0;
    for (int k = 0; k < 2; k++) {
        conv2d(.net = net, .filters = filters, .kernel = 3, .padding = 1, .act_func = ACT_NONE,
               .weight_initialization = WEIGHT_INITIALIZATION_HE);
        last = batch_norm(.net = net, .act_func = ACT_RELU);
    }
    return last;
}

static NeuralNetwork *create_unet(void) {
    NeuralNetwork *net = new_spingalett(.loss_func = LOSS_CROSS_ENTROPY);
    layer(.net = net, .height = UNET_SIZE, .width = UNET_SIZE, .channels = 3);
    uint32_t e1 = double_conv(net, 16);
    max_pool2d(.net = net, .kernel = 2);
    uint32_t e2 = double_conv(net, 32);
    max_pool2d(.net = net, .kernel = 2);
    double_conv(net, 64);
    uint32_t u = conv_transpose2d(.net = net, .filters = 32, .kernel = 2, .stride = 2, .act_func = ACT_RELU,
                                  .weight_initialization = WEIGHT_INITIALIZATION_HE);
    concat_layers(.net = net, .inputs = {e2, u}, .act_func = ACT_NONE);
    double_conv(net, 32);
    u = conv_transpose2d(.net = net, .filters = 16, .kernel = 2, .stride = 2, .act_func = ACT_RELU,
                         .weight_initialization = WEIGHT_INITIALIZATION_HE);
    concat_layers(.net = net, .inputs = {e1, u}, .act_func = ACT_NONE);
    double_conv(net, 16);
    conv2d(.net = net, .filters = 3, .kernel = 1, .act_func = ACT_SIGMOID,
           .weight_initialization = WEIGHT_INITIALIZATION_XAVIER);
    return net;
}

static void unet_benchmark(const char *name, ComputeMode mode, const float *images, const float *masks) {
    spingalett_set_compute_mode(mode);
    for (int untimed = mode == COMPUTE_VULKAN; untimed >= 0; untimed--) {   /* the GPU's first run */
    NeuralNetwork *net = create_unet();
    Sets d = sets_of(images, UNET_SIZE * UNET_SIZE * 3u, masks, UNET_SIZE * UNET_SIZE * 3u, UNET_SAMPLES);
    double start = now();
    train(.net = net, .inputs = d.x ? NULL : images, .targets = d.t ? NULL : masks, .device_inputs = d.x,
          .device_targets = d.t, .sample_count = UNET_SAMPLES, .epochs = 1,
          .learning_rate = 1e-3f, .weight_decay = 1e-4f, .optimizer_type = OPTIMIZER_ADAMW,
          .training_strategy = STRATEGY_SMALL_BATCH, .batch_size = 32);
    double trained = now() - start;
    float *outputs = (float *)malloc((size_t)UNET_SAMPLES * UNET_SIZE * UNET_SIZE * 3 * sizeof(float));
    for (int timed = 0; timed < 2; timed++) {
        start = now();
        predict(.net = net, .inputs = d.x ? NULL : images, .device_inputs = d.x, .sample_count = UNET_SAMPLES,
                .outputs = outputs);
    }
    double inferred = now() - start;
    if (!untimed) printf("%-24s %14.0f %14.0f\n", name, UNET_SAMPLES / trained, UNET_SAMPLES / inferred);
    sets_free(d);
    free(outputs);
    free_network(net);
    }
}

static void run_benchmark(const char *name, ComputeMode mode, const float *inputs, const float *targets) {
    spingalett_set_compute_mode(mode);
    if (mode == COMPUTE_VULKAN) {           /* the GPU's first use, not timed */
        train_throughput(inputs, targets, STRATEGY_FULL_BATCH, 1);
        train_throughput(inputs, targets, STRATEGY_SMALL_BATCH, 1);
        inference_throughput(inputs);
    }
    double full = train_throughput(inputs, targets, STRATEGY_FULL_BATCH, EPOCHS);
    double mini = train_throughput(inputs, targets, STRATEGY_SMALL_BATCH, 1);
    double infer = inference_throughput(inputs);
    printf("%-24s %14.0f %14.0f %14.0f\n", name, full, mini, infer);
}

/* Seconds per call of fn over the samples in turn, for about a third of a second. */
static double latency(int (*fn)(void *ctx, const float *input), void *ctx, const float *inputs) {
    uint32_t runs = 0;
    double start = now(), elapsed;
    do {
        fn(ctx, inputs + (size_t)(runs % SAMPLES) * INPUT_SIZE);
        runs++;
    } while ((elapsed = now() - start) < 0.3);
    return elapsed / runs;
}

static int run_forward(void *net, const float *input) {
    return forward(.net = (NeuralNetwork *)net, .input = input) ? 0 : 1;
}

typedef struct { const SpingalettModel *model; void *workspace; float out[OUTPUT_SIZE]; } RunContext;

static int run_model(void *ctx, const float *input) {
    RunContext *c = (RunContext *)ctx;
    return spingalett_model_run(c->model, input, c->out, c->workspace);
}

static void deployment_benchmark(const float *inputs) {
    static const char *names[] = {"FLOAT32", "FP16", "BFLOAT16", "INT8", "INT4", "INT2"};
    NeuralNetwork *net = create_network();
    float *outputs = (float *)malloc((size_t)SAMPLES * OUTPUT_SIZE * sizeof(float));
    spingalett_set_compute_mode(COMPUTE_SINGLE_THREADED);
    printf("\n%-16s %14s %14s %14s\n", "deployment", "bytes", "one sample us", "batched/s");
    printf("%-16s %14s %14.1f\n", "forward()", "", latency(run_forward, net, inputs) * 1e6);
    for (int p = 0; p < PRECISION_COUNT; p++) {
        SpingalettModel *model = spingalett_model_from_network(net, (PrecisionMode)p);
        if (!model) continue;
        RunContext ctx = {model, malloc(model->workspace_size), {0}};
        double single = latency(run_model, &ctx, inputs);
#if defined(SPINGALETT_HAS_OPENMP)
        spingalett_set_compute_mode(COMPUTE_OPENMP);
#endif
        double batched = 0;
        for (int rep = 0; rep < 3; rep++) {           /* best of three */
            double start = now();
            spingalett_model_predict(model, inputs, SAMPLES, outputs);
            double rate = SAMPLES / (now() - start);
            if (rate > batched) batched = rate;
        }
        spingalett_set_compute_mode(COMPUTE_SINGLE_THREADED);
        printf("%-16s %14zu %14.1f %14.0f\n", names[p], model->image_size, single * 1e6, batched);
        free(ctx.workspace);
        spingalett_model_free(model);
    }
    free(outputs);
    free_network(net);
}

int main(int argc, char **argv) {
    srand(42);
    spingalett_set_verbose(false);

    float *inputs = NULL, *targets = NULL;
    generate_synthetic_data(&inputs, &targets);

    /* More threads than cores oversubscribes the CPU and slows training down. */
    bool cpu = true;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "gpu")) cpu = false;
        else spingalett_set_num_threads((unsigned)strtoul(argv[i], NULL, 10));
    }

    NeuralNetwork *probe = create_network();
    printf("Spingalett %s (%s kernels): %d-%d-%d-%d (%" PRIu64 " parameters), %d samples, threads: ",
           spingalett_version(), spingalett_cpu_kernels(), INPUT_SIZE, HIDDEN_1, HIDDEN_2, OUTPUT_SIZE,
           spingalett_parameter_count(probe), SAMPLES);
    free_network(probe);
    if (spingalett_get_num_threads() > 0) printf("%u", spingalett_get_num_threads());
    else printf("runtime default");
    printf(spingalett_gpu_device() ? ", GPU: %s\n\n" : "\n\n", spingalett_gpu_device());

    printf("%-24s %14s %14s %14s\n", "samples/s", "full batch", "mini-batch 64", "inference");
    if (cpu) {
        run_benchmark("Single-threaded", COMPUTE_SINGLE_THREADED, inputs, targets);
#if defined(SPINGALETT_HAS_OPENMP)
        run_benchmark("OpenMP", COMPUTE_OPENMP, inputs, targets);
#endif
#if defined(SPINGALETT_HAS_OPENBLAS)
        run_benchmark("OpenBLAS", COMPUTE_OPENBLAS, inputs, targets);
#endif
    }
    const char *gpu = spingalett_gpu_device();
    /* and with the matrix products in bfloat16 on its matrix units, where it has them */
    const bool bf16 = gpu && spingalett_set_gpu_precision(PRECISION_BFLOAT16);
    spingalett_set_gpu_precision(PRECISION_FLOAT32);
    for (int dev = 0; gpu && dev < 2; dev++) {      /* the host's arrays, then data sets on the GPU */
        on_device = dev;
        run_benchmark(dev ? "Vulkan GPU, data on GPU" : "Vulkan GPU", COMPUTE_VULKAN, inputs, targets);
        if (bf16) {
            spingalett_set_gpu_precision(PRECISION_BFLOAT16);
            run_benchmark(dev ? "Vulkan bf16, data on GPU" : "Vulkan GPU bf16", COMPUTE_VULKAN, inputs, targets);
            spingalett_set_gpu_precision(PRECISION_FLOAT32);
        }
    }
    on_device = false;
    if (cpu) deployment_benchmark(inputs);

    /* the convolutional network: random images, one-hot labels */
    float *images = (float *)malloc((size_t)CNN_SAMPLES * INPUT_SIZE * sizeof(float));
    float *labels = (float *)calloc((size_t)CNN_SAMPLES * OUTPUT_SIZE, sizeof(float));
    if (!images || !labels) {
        fprintf(stderr, "Allocation failed\n");
        return 1;
    }
    for (size_t i = 0; i < (size_t)CNN_SAMPLES * INPUT_SIZE; i++) images[i] = (float)rand() / (float)RAND_MAX;
    for (size_t s = 0; s < CNN_SAMPLES; s++) labels[s * OUTPUT_SIZE + (size_t)rand() % OUTPUT_SIZE] = 1.0f;
    for (int normalized = 0; normalized < 2; normalized++) {
        NeuralNetwork *cnn = create_cnn(normalized);
        printf("\nconvolutional network (Examples/MNIST_CNN.c)%s, %" PRIu64 " parameters, %d images\n",
               normalized ? " with batch normalization" : "", spingalett_parameter_count(cnn), CNN_SAMPLES);
        free_network(cnn);
        printf("%-24s %14s %14s\n", "samples/s", "training", "inference");
        if (cpu) {
            cnn_benchmark("Single-threaded", COMPUTE_SINGLE_THREADED, normalized, images, labels);
#if defined(SPINGALETT_HAS_OPENMP)
            cnn_benchmark("OpenMP", COMPUTE_OPENMP, normalized, images, labels);
#endif
#if defined(SPINGALETT_HAS_OPENBLAS)
            cnn_benchmark("OpenBLAS", COMPUTE_OPENBLAS, normalized, images, labels);
#endif
        }
        for (int dev = 0; gpu && dev < 2; dev++) {      /* the host's arrays, then data sets on the GPU */
            on_device = dev;
            cnn_benchmark(dev ? "Vulkan GPU, data on GPU" : "Vulkan GPU", COMPUTE_VULKAN, normalized, images, labels);
            if (bf16) {
                spingalett_set_gpu_precision(PRECISION_BFLOAT16);
                cnn_benchmark(dev ? "Vulkan bf16, data on GPU" : "Vulkan GPU bf16", COMPUTE_VULKAN, normalized, images, labels);
                spingalett_set_gpu_precision(PRECISION_FLOAT32);
            }
        }
        on_device = false;
    }

    free(images);
    free(labels);

    /* ResNet-20 on random 32 x 32 x 3 images */
    images = (float *)malloc((size_t)RESNET_SAMPLES * 3072 * sizeof(float));
    labels = (float *)calloc((size_t)RESNET_SAMPLES * OUTPUT_SIZE, sizeof(float));
    if (!images || !labels) {
        fprintf(stderr, "Allocation failed\n");
        return 1;
    }
    for (size_t i = 0; i < (size_t)RESNET_SAMPLES * 3072; i++) images[i] = (float)rand() / (float)RAND_MAX;
    for (size_t s = 0; s < RESNET_SAMPLES; s++) labels[s * OUTPUT_SIZE + (size_t)rand() % OUTPUT_SIZE] = 1.0f;
    NeuralNetwork *resnet = create_resnet20();
    printf("\nResNet-20 (Examples/CIFAR10.c resnet20), %u layers, %" PRIu64 " parameters, %d images\n",
           spingalett_layer_count(resnet), spingalett_parameter_count(resnet), RESNET_SAMPLES);
    free_network(resnet);
    printf("%-24s %14s %14s\n", "samples/s", "training", "inference");
    if (cpu) {
        resnet_benchmark("Single-threaded", COMPUTE_SINGLE_THREADED, images, labels);
#if defined(SPINGALETT_HAS_OPENMP)
        resnet_benchmark("OpenMP", COMPUTE_OPENMP, images, labels);
#endif
    }
    for (int dev = 0; gpu && dev < 2; dev++) {      /* the host's arrays, then data sets on the GPU */
        on_device = dev;
        resnet_benchmark(dev ? "Vulkan GPU, data on GPU" : "Vulkan GPU", COMPUTE_VULKAN, images, labels);
        if (bf16) {
            spingalett_set_gpu_precision(PRECISION_BFLOAT16);
            resnet_benchmark(dev ? "Vulkan bf16, data on GPU" : "Vulkan GPU bf16", COMPUTE_VULKAN, images, labels);
            spingalett_set_gpu_precision(PRECISION_FLOAT32);
        }
    }
    on_device = false;
    free(images);
    free(labels);

    /* the U-Net on random 64 x 64 x 3 images and random masks of three classes */
    const size_t pixels = (size_t)UNET_SAMPLES * UNET_SIZE * UNET_SIZE;
    images = (float *)malloc(pixels * 3 * sizeof(float));
    labels = (float *)malloc(pixels * 3 * sizeof(float));
    if (!images || !labels) {
        fprintf(stderr, "Allocation failed\n");
        return 1;
    }
    for (size_t i = 0; i < pixels * 3; i++) images[i] = (float)rand() / (float)RAND_MAX;
    for (size_t i = 0; i < pixels * 3; i++) labels[i] = (float)(rand() % 2);
    NeuralNetwork *unet = create_unet();
    printf("\nU-Net (Examples/Segmentation.c), %u layers, %" PRIu64 " parameters, %d images of %d x %d\n",
           spingalett_layer_count(unet), spingalett_parameter_count(unet), UNET_SAMPLES, UNET_SIZE, UNET_SIZE);
    free_network(unet);
    printf("%-24s %14s %14s\n", "samples/s", "training", "inference");
    if (cpu) {
        unet_benchmark("Single-threaded", COMPUTE_SINGLE_THREADED, images, labels);
#if defined(SPINGALETT_HAS_OPENMP)
        unet_benchmark("OpenMP", COMPUTE_OPENMP, images, labels);
#endif
    }
    for (int dev = 0; gpu && dev < 2; dev++) {      /* the host's arrays, then data sets on the GPU */
        on_device = dev;
        unet_benchmark(dev ? "Vulkan GPU, data on GPU" : "Vulkan GPU", COMPUTE_VULKAN, images, labels);
        if (bf16) {
            spingalett_set_gpu_precision(PRECISION_BFLOAT16);
            unet_benchmark(dev ? "Vulkan bf16, data on GPU" : "Vulkan GPU bf16", COMPUTE_VULKAN, images, labels);
            spingalett_set_gpu_precision(PRECISION_FLOAT32);
        }
    }
    on_device = false;
    free(images);
    free(labels);
    free(inputs);
    free(targets);
    return 0;
}
