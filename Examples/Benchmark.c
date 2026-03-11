/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

#include <Spingalett/Spingalett.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#if defined(_WIN32)
    #include <windows.h>
    static double get_wall_time(void) {
        LARGE_INTEGER time, freq;
        if (!QueryPerformanceFrequency(&freq)) return 0;
        if (!QueryPerformanceCounter(&time)) return 0;
        return (double)time.QuadPart / freq.QuadPart;
    }
#else
    #include <sys/time.h>
    static double get_wall_time(void) {
        struct timeval time;
        if (gettimeofday(&time, NULL)) return 0;
        return (double)time.tv_sec + (double)time.tv_usec * 1e-6;
    }
#endif

#define INPUT_SIZE  784
#define HIDDEN_1    512
#define HIDDEN_2    1000
#define OUTPUT_SIZE 10
#define SAMPLES     20000
#define EPOCHS      5

static void generate_synthetic_data(float **inputs, float **targets) {
    printf("Generating %d samples of synthetic data...\n", SAMPLES);
    *inputs  = (float *)malloc(SAMPLES * INPUT_SIZE  * sizeof(float));
    *targets = (float *)malloc(SAMPLES * OUTPUT_SIZE * sizeof(float));

    if (!*inputs || !*targets) {
        fprintf(stderr, "Allocation failed\n");
        exit(1);
    }

    for (int i = 0; i < SAMPLES * INPUT_SIZE; i++)
        (*inputs)[i] = (float)rand() / (float)RAND_MAX;

    for (int i = 0; i < SAMPLES * OUTPUT_SIZE; i++)
        (*targets)[i] = (float)rand() / (float)RAND_MAX;
}

static NeuralNetwork *create_network(void) {
    NeuralNetwork *net = new_spingalett(.loss_func = LOSS_CROSS_ENTROPY);

    layer(.net = net, .neurons_amount = INPUT_SIZE);
    layer(.net = net, .neurons_amount = HIDDEN_1,
        .act_func = ACT_RELU, .weight_initialization = WEIGHT_INITIALIZATION_HE);
    layer(.net = net, .neurons_amount = HIDDEN_2,
        .act_func = ACT_RELU, .weight_initialization = WEIGHT_INITIALIZATION_HE);
    layer(.net = net, .neurons_amount = OUTPUT_SIZE,
        .act_func = ACT_SOFTMAX, .weight_initialization = WEIGHT_INITIALIZATION_XAVIER);

    return net;
}

static void run_benchmark(const char *name, ComputeMode mode,
                        float *inputs, float *targets) {
    printf("\n--- Benchmarking: %s ---\n", name);

    spingalett_set_compute_mode(mode);
    spingalett_set_verbose(false);

    NeuralNetwork *net = create_network();
    printf("Topology: %d -> %d -> %d -> %d\n",
        INPUT_SIZE, HIDDEN_1, HIDDEN_2, OUTPUT_SIZE);
    printf("Total Parameters: %lu\n", net->total_weights + net->total_biases);

    double start = get_wall_time();

    train(
        .net = net,
        .inputs = inputs,
        .targets = targets,
        .sample_count = SAMPLES,
        .epochs = EPOCHS,
        .learning_rate = 0.001f,
        .optimizer_type = OPTIMIZER_ADAM,
        .training_strategy = STRATEGY_FULL_BATCH,
        .training_mode = MODE_ARRAY,
        .report_interval = 0
    );

    double elapsed = get_wall_time() - start;
    printf("Result [%s]: %.4f seconds\n", name, elapsed);
    printf("Speed: %.2f samples/sec\n", (double)(SAMPLES * EPOCHS) / elapsed);

    free_network(net);
}

int main(void) {
    srand(42);

    float *inputs = NULL;
    float *targets = NULL;
    generate_synthetic_data(&inputs, &targets);
    spingalett_set_num_threads(16);
    printf("\nThreads: %d", spingalett_get_num_threads());
    
    run_benchmark("Single Threaded", COMPUTE_SINGLE_THREADED, inputs, targets);

#if defined(SPINGALETT_HAS_OPENMP)
    run_benchmark("OpenMP", COMPUTE_OPENMP, inputs, targets);
#else
    printf("\n[SKIP] OpenMP benchmark skipped (library built without OpenMP)\n");
#endif
//*/
    
#if defined(SPINGALETT_HAS_OPENBLAS)
    run_benchmark("OpenBLAS", COMPUTE_OPENBLAS, inputs, targets);
#else
    printf("\n[SKIP] OpenBLAS benchmark skipped (library built without OpenBLAS)\n");
#endif


    free(inputs);
    free(targets);
    return 0;
}
