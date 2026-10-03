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

#define SPINGALETT_FORMAT_VERSION 2

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
    WEIGHT_INITIALIZATION_RANDOM,   /* uniform in [-1, 1] */
    WEIGHT_INITIALIZATION_XAVIER,   /* Glorot normal: variance 2 / (fan_in + fan_out) */
    WEIGHT_INITIALIZATION_HE,       /* He normal: variance 2 / fan_in */
    WEIGHT_INITIALIZATION_NONE,     /* zeros */
    WEIGHT_INITIALIZATION_LECUN,    /* LeCun normal: variance 1 / fan_in */
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

    float *dropout_rates;           /* per layer, applied to its outputs while training */
} NeuralNetwork;

/* Quantity watched for early stopping and best-epoch selection. */
typedef enum {
    MONITOR_AUTO,                   /* validation loss when validation data is given, else training loss */
    MONITOR_TRAIN_LOSS,
    MONITOR_VAL_LOSS,
    MONITOR_VAL_ACCURACY,
    MONITOR_COUNT
} MonitorMetric;

/* Result of evaluate() and of the per-epoch validation pass. */
typedef struct {
    float loss;                     /* mean over samples of the network's loss (see evaluate()) */
    float accuracy;                 /* fraction of samples whose output argmax matches the target's
                                       argmax; with a single output, both on the same side of 0.5 */
} EvalMetrics;

/* State of a train() call, passed to the epoch callback. */
typedef struct {
    size_t epoch;                   /* epochs completed in this call (1-based) */
    size_t epochs;                  /* TrainArgs.epochs */
    float train_loss;               /* mean training loss of this epoch */
    float learning_rate;            /* learning rate used in this epoch */
    bool has_validation;
    EvalMetrics validation;         /* after this epoch, when has_validation */
    MonitorMetric monitor;          /* the monitored quantity (MONITOR_AUTO resolved) */
    size_t best_epoch;              /* epoch with the best monitored value so far */
    float best_value;
    bool improved;                  /* this epoch is the new best */
} TrainProgress;

/* Called after every callback_interval epochs (and the last one); returning true stops training. */
typedef bool (*TrainCallback)(NeuralNetwork *net, const TrainProgress *progress, void *user_data);

typedef enum {
    TRAIN_FAILED,                   /* invalid arguments, out of memory or a misbehaving generator */
    TRAIN_COMPLETED,                /* all epochs ran */
    TRAIN_EARLY_STOPPED,            /* no improvement for early_stopping_patience epochs */
    TRAIN_INTERRUPTED,              /* the callback returned true */
    TRAIN_DIVERGED,                 /* NaN or Inf parameters found (nan_check_interval) */
    TRAIN_NO_DATA                   /* the generator produced no samples in an epoch */
} TrainStatus;

/* Result of train(). */
typedef struct {
    TrainStatus status;
    size_t epochs_run;
    float train_loss;               /* mean training loss of the last epoch */
    bool has_validation;
    EvalMetrics validation;         /* validation metrics of the last epoch */
    MonitorMetric monitor;
    size_t best_epoch;              /* 0 when no epoch was monitored */
    float best_value;
    bool restored_best;             /* parameters were reset to those of best_epoch */
} TrainReport;

/*
 * Data source for MODE_GENERATOR_FUNCTION. Write up to `requested` samples into `inputs`
 * ([requested x input size], row-major) and `targets` ([requested x output size]) and return
 * how many were written; returning 0 ends the epoch. Shuffling and augmentation are up to the
 * generator. It is called once per mini-batch, once per epoch for full batch (requested =
 * sample_count), and in chunks for per-sample training.
 */
typedef uint32_t (*DataGeneratorFn)(float *inputs, float *targets, uint32_t requested, void *user_data);

/*
 * Learning-rate schedule, called before every epoch. `epoch` is the number of epochs already
 * completed in this train() call (0 for the first), `initial_lr` is TrainArgs.learning_rate.
 * Returns the learning rate for the coming epoch; negative or NaN results are ignored.
 */
typedef float (*LRSchedulerFn)(size_t epoch, size_t total_epochs, float initial_lr, void *user_data);

/* Parameters of the built-in schedulers, passed as lr_scheduler_data (NULL = defaults). */
typedef struct {
    size_t warmup_epochs;   /* linear_warmup, warmup_cosine; 0 = 5% of the run (at least 1) */
    size_t step_size;       /* step_decay: epochs between decays; 0 = a third of the run */
    float  gamma;           /* step_decay: decay factor; 0 = 0.1 */
    float  min_lr;          /* cosine_decay, warmup_cosine: final learning rate; default 0 */
} LRScheduleParams;

typedef struct {
    LossFunction loss_func;
} NeuralNetworkArgs;

typedef struct {
    NeuralNetwork *net;
    uint32_t neurons_amount;
    ActivationFunction act_func;
    WeightInitialization weight_initialization;
    float dropout_rate;             /* [0, 1): inverted dropout on this layer's outputs during
                                       training; ignored on the input and output layers */
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

    const float *inputs;            /* MODE_ARRAY: [sample_count x input size] */
    const float *targets;           /* MODE_ARRAY: [sample_count x output size] */
    DataGeneratorFn generator;      /* MODE_GENERATOR_FUNCTION */
    void *generator_data;
    uint32_t sample_count;          /* MODE_ARRAY: number of samples. Generator: samples per epoch
                                       (0 = until the generator returns 0; required for full batch) */
    uint32_t batch_size;
    bool do_not_shuffle;            /* keep sample order (per-sample and mini-batch training) */
    size_t epochs;

    float learning_rate;
    float weight_decay;
    float momentum;
    float beta1;
    float beta2;
    float epsilon;
    float max_grad_norm;            /* clip the global L2 norm of each step's gradient; 0 = off */

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
    void *callback_data;            /* passed to the callback as user_data */

    LRSchedulerFn lr_scheduler;     /* NULL = constant learning_rate */
    void *lr_scheduler_data;

    /* Validation set, evaluated after every epoch (val_count = 0: none). */
    const float *val_inputs;        /* [val_count x input size] */
    const float *val_targets;       /* [val_count x output size] */
    uint32_t val_count;

    /* Best-epoch tracking, active with validation data, early stopping or restore_best_weights. */
    MonitorMetric monitor;
    size_t early_stopping_patience; /* stop after this many epochs without improvement; 0 = never */
    float early_stopping_min_delta; /* smallest change of the monitored value that counts as one */
    bool restore_best_weights;      /* when training ends, for whatever reason, reset weights and
                                       biases to those of the best epoch (kept in memory) */

    /* OpenBLAS threads used while training (restored afterwards). 0 = auto: one thread when
       each BLAS call is too small to amortize threading (per-sample training, small nets or
       mini-batches), otherwise spingalett_set_num_threads() or OpenBLAS's own default. */
    int blas_num_threads;
} TrainArgs;

typedef struct {
    NeuralNetwork *net;
    const float *inputs;            /* [sample_count x input size] */
    uint32_t sample_count;
    float *outputs;                 /* [sample_count x output size] */
} PredictArgs;

typedef struct {
    NeuralNetwork *net;
    const float *inputs;            /* [sample_count x input size] */
    const float *targets;           /* [sample_count x output size] */
    uint32_t sample_count;
} EvaluateArgs;

typedef struct {
    NeuralNetwork *net;
    const char *filename;
    bool do_not_save_optimizer;
    PrecisionMode precision;
} SaveArgs;

/* Optimizer settings for the low-level training API; zero fields take the train() defaults. */
typedef struct {
    OptimizerType type;
    float learning_rate;            /* 0 = 0.01 */
    float weight_decay;
    float momentum;                 /* 0 = 0.9 */
    float beta1;                    /* 0 = 0.9 */
    float beta2;                    /* 0 = 0.999 */
    float epsilon;                  /* 0 = 1e-8 */
    float max_grad_norm;            /* clip the global L2 norm of the step's gradient; 0 = off */
} OptimizerArgs;

/* Holds the activations of one batch between the calls of the low-level training API. */
typedef struct SpingalettTrainer SpingalettTrainer;

/* An in-memory data set (see spingalett_load_idx / spingalett_load_csv). */
typedef struct {
    uint32_t count;
    uint32_t input_size;
    uint32_t target_size;
    float *inputs;                  /* [count x input_size] */
    float *targets;                 /* [count x target_size] */
} SpingalettDataset;

#define SPINGALETT_OK                   0
#define SPINGALETT_ERR_ALLOC            1   /* out of memory */
#define SPINGALETT_ERR_INVALID          2   /* invalid argument or file contents */
#define SPINGALETT_ERR_FILE_IO          3   /* file could not be opened, read or written, or is truncated */
#define SPINGALETT_ERR_FORMAT_VERSION   4   /* model file written by an unsupported format version */

/* Library version (the header's SPINGALETT_VERSION_* macros describe the headers in use). */
SPINGALETT_API const char *spingalett_version(void);

SPINGALETT_API int spingalett_last_error_code(void);
SPINGALETT_API const char *spingalett_last_error_message(void);
SPINGALETT_API void spingalett_clear_error(void);

SPINGALETT_API ComputeMode spingalett_get_compute_mode(void);
SPINGALETT_API void spingalett_set_compute_mode(ComputeMode mode);
SPINGALETT_API unsigned spingalett_get_num_threads(void);
SPINGALETT_API void spingalett_set_num_threads(unsigned n);

SPINGALETT_API void spingalett_set_log_callback(LogCallback cb);
SPINGALETT_API void spingalett_set_log_level(LogLevel level);

/* Seeds the calling thread's generator (weight init, shuffling, dropout) for reproducible runs. */
SPINGALETT_API void spingalett_seed(uint64_t seed);

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

/* Built-in learning-rate schedules (see LRScheduleParams). */
SPINGALETT_API float spingalett_lr_cosine_decay(size_t epoch, size_t total_epochs, float initial_lr, void *params);
SPINGALETT_API float spingalett_lr_linear_warmup(size_t epoch, size_t total_epochs, float initial_lr, void *params);
SPINGALETT_API float spingalett_lr_step_decay(size_t epoch, size_t total_epochs, float initial_lr, void *params);
SPINGALETT_API float spingalett_lr_warmup_cosine(size_t epoch, size_t total_epochs, float initial_lr, void *params);

/* Batched inference: writes the outputs of all samples. Uses matrix-matrix products on every
   backend, so it is much faster than calling forward() per sample. Returns false on error. */
#define predict(...) predict_struct_arguments((PredictArgs){__VA_ARGS__})
SPINGALETT_API bool predict_struct_arguments(PredictArgs args);

/* Mean loss and accuracy over a data set. The loss is the one train() reports: the sum over the
   outputs of the squared error for LOSS_MSE (whose gradient train() follows up to a factor of 2),
   the cross-entropy for LOSS_CROSS_ENTROPY. On error both metrics are NaN. */
#define evaluate(...) evaluate_struct_arguments((EvaluateArgs){__VA_ARGS__})
SPINGALETT_API EvalMetrics evaluate_struct_arguments(EvaluateArgs args);

#define train(...) train_struct_arguments((TrainArgs){__VA_ARGS__})
SPINGALETT_API TrainReport train_struct_arguments(TrainArgs args);

/*
 * Low-level training, for custom loops and losses. A trainer runs forward and backward passes on
 * batches of up to max_batch samples; backward passes add to the network's gradient (grad_weights,
 * grad_biases hold the sum over the samples since the last step) and a step applies the mean of
 * that sum with the given optimizer, so a step's batch can be split into several backward passes.
 * Optimizer state and the step count live in the network and are shared with train().
 */
SPINGALETT_API SpingalettTrainer *spingalett_trainer_new(NeuralNetwork *net, uint32_t max_batch);
SPINGALETT_API void spingalett_trainer_free(SpingalettTrainer *trainer);
/* Training-mode forward pass (dropout active) over count samples; returns their outputs
   [count x output size], valid until the next forward pass. NULL on error. */
SPINGALETT_API const float *spingalett_trainer_forward(SpingalettTrainer *trainer, const float *inputs, uint32_t count);
/* Back-propagates the network's loss for the last forward pass; returns the summed loss of its
   samples (NaN on error). */
SPINGALETT_API float spingalett_trainer_backward(SpingalettTrainer *trainer, const float *targets);
/* Back-propagates a custom loss: output_grads [count x output size] holds dL/d(output) for each
   sample of the last forward pass (for MSE that is output - target). */
SPINGALETT_API bool spingalett_trainer_backward_output_grads(SpingalettTrainer *trainer, const float *output_grads);
/* Optimizer step with the mean gradient accumulated since the last step, which is then cleared.
   Fails when nothing was accumulated. */
SPINGALETT_API bool spingalett_trainer_step(SpingalettTrainer *trainer, const OptimizerArgs *optimizer);
/* Discards the gradient accumulated since the last step. */
SPINGALETT_API void spingalett_trainer_zero_grad(SpingalettTrainer *trainer);
/* Forward, backward with the network's loss and a step on one batch; returns its mean loss
   (NaN on error). */
SPINGALETT_API float spingalett_train_on_batch(SpingalettTrainer *trainer, const float *inputs, const float *targets,
                                              uint32_t count, const OptimizerArgs *optimizer);

#define save_spingalett(...) save_spingalett_struct_arguments((SaveArgs){__VA_ARGS__})
SPINGALETT_API void save_spingalett_struct_arguments(SaveArgs args);

SPINGALETT_API NeuralNetwork *load_spingalett(const char *filename);

/* Data sets. The readers fill *dataset (free it with spingalett_dataset_free) and return false
   on error, leaving it empty. */

/* An IDX pair, the format of MNIST: images_path holds the samples (any shape, flattened; unsigned
   bytes are scaled to [0, 1], float and double values are kept), labels_path one unsigned-byte
   class label per sample, one-hot encoded over num_classes (0 = largest label + 1). */
SPINGALETT_API bool spingalett_load_idx(const char *images_path, const char *labels_path,
                                        uint32_t num_classes, SpingalettDataset *dataset);
/* A numeric CSV file: comma-separated, unquoted, one sample per line; a first line that is not
   numeric is taken as a header. The last target_columns columns are the targets; with
   num_classes > 0 the single target column holds class indices that are one-hot encoded. */
SPINGALETT_API bool spingalett_load_csv(const char *path, uint32_t target_columns, uint32_t num_classes,
                                        SpingalettDataset *dataset);
/* Shuffles the samples with the library's generator (see spingalett_seed). */
SPINGALETT_API void spingalett_dataset_shuffle(SpingalettDataset *dataset);
/* Moves the last count samples into *tail, e.g. to hold out a validation set. */
SPINGALETT_API bool spingalett_dataset_split(SpingalettDataset *dataset, uint32_t count, SpingalettDataset *tail);
SPINGALETT_API void spingalett_dataset_free(SpingalettDataset *dataset);

SPINGALETT_API void print_parameters(NeuralNetwork *net);
SPINGALETT_API void free_network(NeuralNetwork *net);

#ifdef __cplusplus
}
#endif
