/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/
#pragma once

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include "Spingalett.Config.h"

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32) || defined(__CYGWIN__)
#  ifdef SPINGALETT_EXPORTS
#    define SPINGALETT_API __declspec(dllexport)
#  else
#    define SPINGALETT_API __declspec(dllimport)
#  endif
#else
#  define SPINGALETT_API __attribute__((visibility("default")))
#endif

#define SPINGALETT_FORMAT_VERSION 1

typedef enum {
    LOG_DEBUG,
    LOG_INFO,
    LOG_WARNING,
    LOG_ERROR,
    LOG_NONE
} LogLevel;

typedef void (*LogCallback)(LogLevel level, const char *message);

typedef enum {
    ACT_SIGMOID,
    ACT_RELU,
    ACT_TANH,
    ACT_LEAKY_RELU,
    ACT_FOO52,
    ACT_SOFTMAX,
    ACT_NONE,
    ACT_COUNT
} ActivationFunction;

typedef enum {
    LOSS_MSE,
    LOSS_CROSS_ENTROPY,
    LOSS_COUNT
} LossFunction;

typedef enum {
    WEIGHT_INITIALIZATION_RANDOM,
    WEIGHT_INITIALIZATION_XAVIER,
    WEIGHT_INITIALIZATION_HE,
    WEIGHT_INITIALIZATION_NONE,
    WEIGHT_INITIALIZATION_COUNT
} WeightInitialization;

typedef enum {
    MODE_ARRAY,
    MODE_GENERATOR_FUNCTION,
    MODE_COUNT
} TrainingMode;

typedef enum {
    STRATEGY_SAMPLE,
    STRATEGY_FULL_BATCH,
    STRATEGY_SMALL_BATCH,
    STRATEGY_COUNT
} TrainingStrategy;

typedef enum {
    OPTIMIZER_SGD,
    OPTIMIZER_MOMENTUM,
    OPTIMIZER_RMSPROP,
    OPTIMIZER_ADAM,
    OPTIMIZER_ADAMW,
    OPTIMIZER_COUNT
} OptimizerType;

typedef enum {
    COMPUTE_SINGLE_THREADED,
    COMPUTE_OPENMP,
    COMPUTE_OPENBLAS,
    COMPUTE_CUDA,
    COMPUTE_COUNT
} ComputeMode;

typedef enum {
    PRECISION_FLOAT32,
    PRECISION_FP16,
    PRECISION_BFLOAT16,
    PRECISION_INT8,
    PRECISION_INT4,
    PRECISION_INT2,
    PRECISION_COUNT
} PrecisionMode;

typedef enum {
    AUTOSAVE_OFF,
    AUTOSAVE_OVERWRITE,
    AUTOSAVE_NEW_FILES
} AutoSaveMode;

typedef struct {
    uint32_t layers;
    uint32_t *topology;
    ActivationFunction *act_func;

    float *weights;
    float *biases;
    float *neurons;

    float *grad_weights;
    float *grad_biases;

    float *opt_m_weights;
    float *opt_m_biases;
    float *opt_v_weights;
    float *opt_v_biases;

    uint64_t *neuron_offsets;
    uint64_t *weight_offsets;
    uint64_t *bias_offsets;

    uint64_t total_neurons;
    uint64_t total_weights;
    uint64_t total_biases;

    uint64_t time_step;
    LossFunction loss_func;
} NeuralNetwork;

typedef bool (*TrainCallback)(NeuralNetwork *net, size_t epoch, float current_error);

typedef struct {
    LossFunction loss_func;
} NeuralNetworkArgs;

typedef struct {
    NeuralNetwork *net;
    uint32_t neurons_amount;
    ActivationFunction act_func;
    WeightInitialization weight_initialization;
} LayerArgs;

typedef struct {
    NeuralNetwork *net;
    const float *input;
} ForwardArgs;

typedef struct {
    NeuralNetwork *net;
    TrainingMode training_mode;
    TrainingStrategy training_strategy;
    OptimizerType optimizer_type;

    float *inputs;
    float *targets;
    uint32_t sample_count;
    uint32_t batch_size;
    size_t epochs;

    float learning_rate;
    float weight_decay;
    float momentum;
    float beta1;
    float beta2;
    float epsilon;
    float max_grad_norm;

    bool reset_optimizer;
    size_t nan_check_interval;

    size_t report_interval;

    AutoSaveMode autosave_mode;
    size_t autosave_interval;
    const char *autosave_path;
    bool autosave_do_not_save_optimizer;
    PrecisionMode autosave_precision;

    TrainCallback callback;
    size_t callback_interval;
} TrainArgs;

typedef struct {
    NeuralNetwork *net;
    const char *filename;
    bool do_not_save_optimizer;
    PrecisionMode precision;
} SaveArgs;

#define SPINGALETT_OK           0
#define SPINGALETT_ERR_ALLOC    1
#define SPINGALETT_ERR_INVALID  2

SPINGALETT_API int spingalett_last_error_code(void);
SPINGALETT_API const char *spingalett_last_error_message(void);
SPINGALETT_API void spingalett_clear_error(void);

SPINGALETT_API ComputeMode spingalett_get_compute_mode(void);
SPINGALETT_API void spingalett_set_compute_mode(ComputeMode mode);
SPINGALETT_API unsigned spingalett_get_num_threads(void);
SPINGALETT_API void spingalett_set_num_threads(unsigned n);

SPINGALETT_API void spingalett_set_log_callback(LogCallback cb);
SPINGALETT_API void spingalett_set_log_level(LogLevel level);

SPINGALETT_API void spingalett_set_verbose(bool enabled);
SPINGALETT_API bool spingalett_get_verbose(void);

#define new_spingalett(...) new_spingalett_struct_arguments((NeuralNetworkArgs){__VA_ARGS__})
SPINGALETT_API NeuralNetwork *new_spingalett_struct_arguments(NeuralNetworkArgs args);

#define layer(...) layer_struct_arguments((LayerArgs){__VA_ARGS__})
SPINGALETT_API void layer_struct_arguments(LayerArgs args);

SPINGALETT_API float activate(float x, ActivationFunction act_func);
SPINGALETT_API float derivative(float x, ActivationFunction act_func);

#define forward(...) forward_struct_arguments((ForwardArgs){__VA_ARGS__})
SPINGALETT_API float *forward_struct_arguments(ForwardArgs args);

#define train(...) train_struct_arguments((TrainArgs){__VA_ARGS__})
SPINGALETT_API void train_struct_arguments(TrainArgs args);

#define save_spingalett(...) save_spingalett_struct_arguments((SaveArgs){__VA_ARGS__})
SPINGALETT_API void save_spingalett_struct_arguments(SaveArgs args);

SPINGALETT_API NeuralNetwork *load_spingalett(const char *filename);

SPINGALETT_API void print_parameters(NeuralNetwork *net);
SPINGALETT_API void free_network(NeuralNetwork *net);

#ifdef __cplusplus
}
#endif
