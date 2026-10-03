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
 * Usage: Benchmark [threads]. Examples/benchmark_pytorch.py runs the same workload in PyTorch.
 */

#include <Spingalett/Spingalett.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define INPUT_SIZE  784
#define HIDDEN_1    512
#define HIDDEN_2    1000
#define OUTPUT_SIZE 10
#define SAMPLES     20000
#define EPOCHS      5
#define MINI_BATCH  64

static double now(void) {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
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
    double start = now();
    train(.net = net, .inputs = inputs, .targets = targets, .sample_count = SAMPLES, .epochs = epochs,
          .learning_rate = 0.001f, .optimizer_type = OPTIMIZER_ADAM, .training_strategy = strategy,
          .batch_size = MINI_BATCH);
    double elapsed = now() - start;
    free_network(net);
    return (double)SAMPLES * (double)epochs / elapsed;
}

static double inference_throughput(const float *inputs) {
    NeuralNetwork *net = create_network();
    float *outputs = (float *)malloc((size_t)SAMPLES * OUTPUT_SIZE * sizeof(float));
    double start = now();
    predict(.net = net, .inputs = inputs, .sample_count = SAMPLES, .outputs = outputs);
    double elapsed = now() - start;
    free(outputs);
    free_network(net);
    return SAMPLES / elapsed;
}

static void run_benchmark(const char *name, ComputeMode mode, const float *inputs, const float *targets) {
    spingalett_set_compute_mode(mode);
    double full = train_throughput(inputs, targets, STRATEGY_FULL_BATCH, EPOCHS);
    double mini = train_throughput(inputs, targets, STRATEGY_SMALL_BATCH, 1);
    double infer = inference_throughput(inputs);
    printf("%-16s %14.0f %14.0f %14.0f\n", name, full, mini, infer);
}

int main(int argc, char **argv) {
    srand(42);
    spingalett_set_verbose(false);

    float *inputs = NULL, *targets = NULL;
    generate_synthetic_data(&inputs, &targets);

    /* More threads than cores oversubscribes the CPU and slows training down. */
    if (argc > 1)
        spingalett_set_num_threads((unsigned)strtoul(argv[1], NULL, 10));

    NeuralNetwork *probe = create_network();
    printf("Spingalett %s: %d-%d-%d-%d (%" PRIu64 " parameters), %d samples, threads: ",
           spingalett_version(), INPUT_SIZE, HIDDEN_1, HIDDEN_2, OUTPUT_SIZE,
           probe->total_weights + probe->total_biases, SAMPLES);
    free_network(probe);
    if (spingalett_get_num_threads() > 0) printf("%u\n\n", spingalett_get_num_threads());
    else printf("runtime default\n\n");

    printf("%-16s %14s %14s %14s\n", "samples/s", "full batch", "mini-batch 64", "inference");
    run_benchmark("Single-threaded", COMPUTE_SINGLE_THREADED, inputs, targets);
#if defined(SPINGALETT_HAS_OPENMP)
    run_benchmark("OpenMP", COMPUTE_OPENMP, inputs, targets);
#endif
#if defined(SPINGALETT_HAS_OPENBLAS)
    run_benchmark("OpenBLAS", COMPUTE_OPENBLAS, inputs, targets);
#endif

    free(inputs);
    free(targets);
    return 0;
}
