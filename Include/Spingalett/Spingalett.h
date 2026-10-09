/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/
#pragma once

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include "Spingalett.Config.h"
#include "Spingalett.Inference.h"   /* SPINGALETT_API, shared enums, error codes, the inference engine */

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    SPINGALETT_LOG_DEBUG,
    SPINGALETT_LOG_INFO,
    SPINGALETT_LOG_WARNING,
    SPINGALETT_LOG_ERROR,
    SPINGALETT_LOG_NONE
} SpingalettLogLevel;

typedef void (*SpingalettLogCallback)(SpingalettLogLevel level, const char *message);

typedef enum {
    SPINGALETT_WEIGHT_INITIALIZATION_RANDOM,   /* uniform in [-1, 1] */
    SPINGALETT_WEIGHT_INITIALIZATION_XAVIER,   /* Glorot normal: variance 2 / (fan_in + fan_out) */
    SPINGALETT_WEIGHT_INITIALIZATION_HE,       /* He normal: variance 2 / fan_in */
    SPINGALETT_WEIGHT_INITIALIZATION_NONE,     /* zeros */
    SPINGALETT_WEIGHT_INITIALIZATION_LECUN,    /* LeCun normal: variance 1 / fan_in */
    SPINGALETT_WEIGHT_INITIALIZATION_COUNT
} SpingalettWeightInitialization;

typedef enum {
    SPINGALETT_MODE_ARRAY,
    SPINGALETT_MODE_GENERATOR_FUNCTION,
    SPINGALETT_MODE_COUNT
} SpingalettTrainingMode;

typedef enum {
    SPINGALETT_STRATEGY_SAMPLE,
    SPINGALETT_STRATEGY_FULL_BATCH,
    SPINGALETT_STRATEGY_SMALL_BATCH,
    SPINGALETT_STRATEGY_COUNT
} SpingalettTrainingStrategy;

typedef enum {
    SPINGALETT_OPTIMIZER_SGD,
    SPINGALETT_OPTIMIZER_MOMENTUM,
    SPINGALETT_OPTIMIZER_RMSPROP,
    SPINGALETT_OPTIMIZER_ADAM,
    SPINGALETT_OPTIMIZER_ADAMW,
    SPINGALETT_OPTIMIZER_COUNT
} SpingalettOptimizerType;

typedef enum {
    SPINGALETT_COMPUTE_SINGLE_THREADED,
    SPINGALETT_COMPUTE_OPENMP,
    SPINGALETT_COMPUTE_OPENBLAS,
    SPINGALETT_COMPUTE_CUDA,                   /* reserved: falls back to the CPU */
    SPINGALETT_COMPUTE_VULKAN,                 /* the GPU through Vulkan compute (spingalett_gpu_device()); the CPU
                                       parts of training run as with SPINGALETT_COMPUTE_OPENMP */
    SPINGALETT_COMPUTE_COUNT
} SpingalettComputeMode;

typedef enum {
    SPINGALETT_AUTOSAVE_OFF,
    SPINGALETT_AUTOSAVE_OVERWRITE,
    SPINGALETT_AUTOSAVE_NEW_FILES
} SpingalettAutoSaveMode;

/* A network being built or trained. Its contents are private: describe it with
   spingalett_network_layer() and friends, and read or write its parameters with
   spingalett_get_parameters() / spingalett_set_parameters(). */
typedef struct SpingalettNetwork SpingalettNetwork;

/* A data set in the GPU's memory (spingalett_device_data_new()). */
typedef struct SpingalettDeviceData SpingalettDeviceData;

/* Kinds of layers: SpingalettLayerType, in Spingalett.Inference.h (the inference engine runs them all). */

/* Layer `index` of a network (0 is the input layer), see spingalett_network_layer(). */
typedef struct {
    SpingalettLayerType type;                 /* the input layer is SPINGALETT_LAYER_DENSE */
    uint32_t height, width, channels;   /* output shape; a dense layer is 1 x 1 x outputs */
    uint32_t outputs;               /* height * width * channels */
    SpingalettActivationFunction activation;  /* SPINGALETT_ACT_NONE for the input layer and pooling layers */
    float dropout_rate;
    uint32_t kernel_h, kernel_w;    /* conv and pooling: window over the previous layer */
    uint32_t stride_h, stride_w;
    uint32_t padding_h, padding_w;  /* zeros (conv) or ignored cells (pooling) on each side */
    uint64_t weight_count;          /* parameters feeding the layer: dense outputs x inputs, conv */
    uint64_t bias_count;            /* filters x kernel_h x kernel_w x input channels / groups; one
                                       bias per output (dense) or filter (conv); batch normalization:
                                       gamma (weights) and beta (biases), one per channel; none for
                                       the input and pooling layers */
    uint32_t groups;                /* conv: channel groups (1 unless set) */
    float epsilon;                  /* batch normalization: added to the variance */
    float momentum;                 /* batch normalization: weight of each batch in the running
                                       statistics */
    uint32_t input_count;           /* layers it reads (0 for the input layer) */
    uint32_t inputs[SPINGALETT_MAX_INPUTS];     /* their indices, all below `index` */
    SpingalettUpsampleMode upsample;          /* upsampling: how cells are filled (stride_h x stride_w each) */
} SpingalettNetworkLayer;

/* Which parameters spingalett_get_parameters() and spingalett_set_parameters() copy. */
typedef enum {
    SPINGALETT_PARAM_WEIGHTS,
    SPINGALETT_PARAM_BIASES,
    SPINGALETT_PARAM_WEIGHT_GRADIENTS,         /* the gradient left by the last training step or backward pass */
    SPINGALETT_PARAM_BIAS_GRADIENTS,
    SPINGALETT_PARAM_RUNNING_MEAN,             /* batch normalization: the statistics inference uses, one per */
    SPINGALETT_PARAM_RUNNING_VARIANCE,         /* channel (count: the layer's bias_count) */
    SPINGALETT_PARAM_KIND_COUNT
} SpingalettParameterKind;

/* Quantity watched for early stopping and best-epoch selection. */
typedef enum {
    SPINGALETT_MONITOR_AUTO,                   /* validation loss when validation data is given, else training loss */
    SPINGALETT_MONITOR_TRAIN_LOSS,
    SPINGALETT_MONITOR_VAL_LOSS,
    SPINGALETT_MONITOR_VAL_ACCURACY,
    SPINGALETT_MONITOR_COUNT
} SpingalettMonitorMetric;

/* Result of spingalett_evaluate() and of the per-epoch validation pass. */
typedef struct {
    float loss;                     /* mean over samples of the network's loss (see spingalett_evaluate()) */
    float accuracy;                 /* fraction of samples whose output argmax matches the target's
                                       argmax; with a single output, both on the same side of 0.5 */
} SpingalettEvalMetrics;

/* State of a spingalett_train() call, passed to the epoch callback. */
typedef struct {
    size_t epoch;                   /* epochs completed in this call (1-based) */
    size_t epochs;                  /* SpingalettTrainArgs.epochs */
    float train_loss;               /* mean training loss of this epoch */
    float learning_rate;            /* learning rate used in this epoch */
    bool has_validation;
    SpingalettEvalMetrics validation;         /* after this epoch, when has_validation */
    SpingalettMonitorMetric monitor;          /* the monitored quantity (SPINGALETT_MONITOR_AUTO resolved) */
    size_t best_epoch;              /* epoch with the best monitored value so far */
    float best_value;
    bool improved;                  /* this epoch is the new best */
} SpingalettTrainProgress;

/* Called after every callback_interval epochs (and the last one); returning true stops training. */
typedef bool (*SpingalettTrainCallback)(SpingalettNetwork *net, const SpingalettTrainProgress *progress, void *user_data);

typedef enum {
    SPINGALETT_TRAIN_FAILED,                   /* invalid arguments, out of memory or a misbehaving generator */
    SPINGALETT_TRAIN_COMPLETED,                /* all epochs ran */
    SPINGALETT_TRAIN_EARLY_STOPPED,            /* no improvement for early_stopping_patience epochs */
    SPINGALETT_TRAIN_INTERRUPTED,              /* the callback returned true */
    SPINGALETT_TRAIN_DIVERGED,                 /* NaN or Inf parameters found (nan_check_interval) */
    SPINGALETT_TRAIN_NO_DATA                   /* the generator produced no samples in an epoch */
} SpingalettTrainStatus;

/* Result of spingalett_train(). */
typedef struct {
    SpingalettTrainStatus status;
    size_t epochs_run;
    float train_loss;               /* mean training loss of the last epoch */
    bool has_validation;
    SpingalettEvalMetrics validation;         /* validation metrics of the last epoch */
    SpingalettMonitorMetric monitor;
    size_t best_epoch;              /* 0 when no epoch was monitored */
    float best_value;
    bool restored_best;             /* parameters were reset to those of best_epoch */
} SpingalettTrainReport;

/*
 * Data source for SPINGALETT_MODE_GENERATOR_FUNCTION. Write up to `requested` samples into `inputs`
 * ([requested x input size], row-major) and `targets` ([requested x output size]) and return
 * how many were written; returning 0 ends the epoch (a 0 in answer to the first request of an
 * epoch is retried once before training stops). Shuffling and augmentation are up to the
 * generator. It is called once per mini-batch, once per epoch for full batch (requested =
 * sample_count), and in chunks for per-sample training.
 */
typedef uint32_t (*SpingalettDataGeneratorFn)(float *inputs, float *targets, uint32_t requested, void *user_data);

/*
 * Learning-rate schedule, called before every epoch. `epoch` is the number of epochs already
 * completed in this spingalett_train() call (0 for the first), `initial_lr` is SpingalettTrainArgs.learning_rate.
 * Returns the learning rate for the coming epoch; negative or NaN results are ignored.
 */
typedef float (*SpingalettLRSchedulerFn)(size_t epoch, size_t total_epochs, float initial_lr, void *user_data);

/* Parameters of the built-in schedulers, passed as lr_scheduler_data (NULL = defaults). */
typedef struct {
    size_t warmup_epochs;   /* linear_warmup, warmup_cosine; 0 = 5% of the run (at least 1) */
    size_t step_size;       /* step_decay: epochs between decays; 0 = a third of the run */
    float  gamma;           /* step_decay: decay factor; 0 = 0.1 */
    float  min_lr;          /* cosine_decay, warmup_cosine: final learning rate; default 0 */
} SpingalettLRScheduleParams;

typedef struct {
    SpingalettLossFunction loss_func;
} SpingalettNetworkArgs;

typedef struct {
    SpingalettNetwork *net;
    uint32_t neurons_amount;        /* dense: outputs; input layer: its size (or give its shape) */
    SpingalettActivationFunction act_func;    /* dense, conv, batch normalization, add and concatenation layers
                                       (pooling layers have none); 0 is SPINGALETT_ACT_NONE */
    SpingalettWeightInitialization weight_initialization;
    float dropout_rate;             /* [0, 1): inverted dropout on this layer's outputs during
                                       training; ignored on the input and output layers */
    SpingalettLayerType type;                 /* SPINGALETT_LAYER_DENSE unless set; see spingalett_conv2d(), spingalett_max_pool2d(),
                                       spingalett_avg_pool2d(), spingalett_batch_norm(), spingalett_add_layers(), spingalett_concat_layers(),
                                       spingalett_global_avg_pool2d(), spingalett_conv_transpose2d(), spingalett_upsample2d(),
                                       spingalett_layer_norm() */
    uint32_t height, width, channels;   /* input layer: the shape of a sample (channels-last), e.g.
                                       28 x 28 x 1 for MNIST; omitted: 1 x 1 x neurons_amount */
    uint32_t filters;               /* conv and transposed conv: output channels */
    uint32_t kernel;                /* conv and pooling: square window size */
    uint32_t stride;                /* 0 = 1 for conv, the window size for pooling; upsampling: the
                                       factor (cells per input cell along an axis), 0 = 2 */
    uint32_t padding;               /* cells added on each side: zeros for conv, ignored by pooling;
                                       kernel / 2 keeps the size of odd windows with stride 1 */
    uint32_t kernel_h, kernel_w;    /* per-axis overrides of kernel, stride and padding (0 = unset) */
    uint32_t stride_h, stride_w;
    uint32_t padding_h, padding_w;
    uint32_t groups;                /* conv: split input and output channels into this many groups,
                                       each filter seeing the input channels of its group (0 = 1;
                                       the input channels give depthwise convolution) */
    float epsilon;                  /* batch and layer normalization: added to the variance; 0 = 1e-5 */
    float momentum;                 /* batch normalization: running statistics move this far towards
                                       each training batch's; 0 = 0.1 */
    uint32_t inputs[SPINGALETT_MAX_INPUTS];     /* the earlier layers this one reads, by index (as
                                       spingalett_layer() returns it; 0 is the input layer); none: the layer
                                       added just before. Dense, convolution, pooling and batch
                                       normalization layers read one, spingalett_add_layers() and
                                       spingalett_concat_layers() one or more. */
    uint32_t input_count;           /* entries of inputs; 0 counts them up to the last nonzero one,
                                       so give it when the last input is the input layer */
    SpingalettUpsampleMode upsample;          /* upsampling: copies (SPINGALETT_UPSAMPLE_NEAREST, the default) or bilinear */
    uint32_t output_padding;        /* transposed convolution: cells added to the output's bottom and
                                       right (less than the stride), to reach sizes the stride skips */
    uint32_t output_padding_h, output_padding_w;    /* per-axis overrides (0 = unset) */
} SpingalettLayerArgs;

typedef struct {
    SpingalettNetwork *net;
    const float *input;
} SpingalettForwardArgs;

typedef struct {
    SpingalettNetwork *net;
    SpingalettTrainingMode training_mode;
    SpingalettTrainingStrategy training_strategy;
    SpingalettOptimizerType optimizer_type;

    const float *inputs;            /* SPINGALETT_MODE_ARRAY: [sample_count x input size] */
    const float *targets;           /* SPINGALETT_MODE_ARRAY: [sample_count x output size] */
    /* SPINGALETT_MODE_ARRAY: the inputs or the targets (or both) in the GPU's memory instead (their first
       sample_count rows), gathered and augmented there with SPINGALETT_COMPUTE_VULKAN */
    const SpingalettDeviceData *device_inputs;
    const SpingalettDeviceData *device_targets;
    SpingalettDataGeneratorFn generator;      /* SPINGALETT_MODE_GENERATOR_FUNCTION */
    void *generator_data;
    uint32_t sample_count;          /* SPINGALETT_MODE_ARRAY: number of samples. Generator: samples per epoch
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

    SpingalettAutoSaveMode autosave_mode;
    size_t autosave_interval;
    const char *autosave_path;
    bool autosave_do_not_save_optimizer;
    SpingalettPrecisionMode autosave_precision;

    SpingalettTrainCallback callback;
    size_t callback_interval;
    void *callback_data;            /* passed to the callback as user_data */

    SpingalettLRSchedulerFn lr_scheduler;     /* NULL = constant learning_rate */
    void *lr_scheduler_data;

    /* Validation set, evaluated after every epoch (val_count = 0: none). */
    const float *val_inputs;        /* [val_count x input size] */
    const float *val_targets;       /* [val_count x output size] */
    const SpingalettDeviceData *device_val_inputs;   /* or in the GPU's memory */
    const SpingalettDeviceData *device_val_targets;
    uint32_t val_count;

    /* Best-epoch tracking, active with validation data, early stopping or restore_best_weights. */
    SpingalettMonitorMetric monitor;
    size_t early_stopping_patience; /* stop after this many epochs without improvement; 0 = never */
    float early_stopping_min_delta; /* smallest change of the monitored value that counts as one */
    bool restore_best_weights;      /* when training ends, for whatever reason, reset weights and
                                       biases to those of the best epoch (kept in memory) */

    /* OpenBLAS threads used while training (restored afterwards). 0 = auto: one thread when
       each BLAS call is too small to amortize threading (per-sample training, small nets or
       mini-batches), otherwise spingalett_set_num_threads() or OpenBLAS's own default. */
    int blas_num_threads;

    /* Augmentation of image samples (networks whose input layer has a height and width), drawn
       anew for every sample of every step: a shift by up to augment_shift cells along each axis,
       cells shifted in being 0 (random crops of an image padded by augment_shift), and, with
       augment_flip, a mirror image left to right for half of the samples. Validation data is not
       augmented. */
    uint32_t augment_shift;
    bool augment_flip;

    /* Label smoothing: training targets move this far towards the uniform distribution,
       t' = (1 - label_smoothing) t + label_smoothing / outputs (/ 2 for each sigmoid output, its own
       two classes), in [0, 1); the reported training loss is the smoothed targets' and validation
       uses the targets as given. 0.1 is common for classifiers. */
    float label_smoothing;

    /* Reduce on plateau: when the monitored value (see monitor) has not improved for
       lr_plateau_patience epochs, the learning rate (the schedule's, when there is one) is
       multiplied by lr_plateau_factor, in (0, 1), for the rest of the run, again after every
       lr_plateau_patience epochs without improvement, but never below lr_plateau_min_lr.
       lr_plateau_patience 0: off. */
    float lr_plateau_factor;
    size_t lr_plateau_patience;
    float lr_plateau_min_lr;
} SpingalettTrainArgs;

typedef struct {
    SpingalettNetwork *net;
    const float *inputs;            /* [sample_count x input size] */
    const SpingalettDeviceData *device_inputs;       /* or in the GPU's memory (its first rows) */
    uint32_t sample_count;
    float *outputs;                 /* [sample_count x output size] */
} SpingalettPredictArgs;

typedef struct {
    SpingalettNetwork *net;
    const float *inputs;            /* [sample_count x input size] */
    const float *targets;           /* [sample_count x output size] */
    const SpingalettDeviceData *device_inputs;       /* either or both in the GPU's memory instead */
    const SpingalettDeviceData *device_targets;
    uint32_t sample_count;
} SpingalettEvaluateArgs;

typedef struct {
    SpingalettNetwork *net;
    const char *filename;
    bool do_not_save_optimizer;
    SpingalettPrecisionMode precision;
} SpingalettSaveArgs;

/* Optimizer settings for the low-level training API; zero fields take the spingalett_train() defaults. */
typedef struct {
    SpingalettOptimizerType type;
    float learning_rate;            /* 0 = 0.01 */
    float weight_decay;
    float momentum;                 /* 0 = 0.9 */
    float beta1;                    /* 0 = 0.9 */
    float beta2;                    /* 0 = 0.999 */
    float epsilon;                  /* 0 = 1e-8 */
    float max_grad_norm;            /* clip the global L2 norm of the step's gradient; 0 = off */
} SpingalettOptimizerArgs;

/* Holds the activations of one batch between the calls of the low-level training API. */
typedef struct SpingalettTrainer SpingalettTrainer;

/* An in-memory data set (see spingalett_load_idx, spingalett_load_csv, spingalett_load_cifar). */
typedef struct {
    uint32_t count;
    uint32_t input_size;
    uint32_t target_size;
    float *inputs;                  /* [count x input_size] */
    float *targets;                 /* [count x target_size] */
    uint32_t height, width, channels;   /* shape of an input (channels last) when known, as for IDX
                                           images, CIFAR and .slettd files that record it; else 0 */
    char **class_names;             /* target_size class names when known, else NULL (owned by the
                                       data set: see spingalett_dataset_set_class_names) */
} SpingalettDataset;

/* Library version (the header's SPINGALETT_VERSION_* macros describe the headers in use). */
SPINGALETT_API const char *spingalett_version(void);

SPINGALETT_API int spingalett_last_error_code(void);
SPINGALETT_API const char *spingalett_last_error_message(void);
SPINGALETT_API void spingalett_clear_error(void);

/* Instruction set of the matrix-multiplication kernels that training and batched inference use:
   "AVX-512", "AVX2", "AVX", "SSE2", "NEON" or "C". x86-64 libraries built without
   SPINGALETT_NATIVE_ARCH (such as the release binaries) choose AVX-512 or AVX2 kernels at run time
   when the processor has them. */
SPINGALETT_API const char *spingalett_cpu_kernels(void);

SPINGALETT_API SpingalettComputeMode spingalett_get_compute_mode(void);
SPINGALETT_API void spingalett_set_compute_mode(SpingalettComputeMode mode);
/* The name of the GPU that SPINGALETT_COMPUTE_VULKAN uses (opening the device on first call), or NULL when the
   library was built without the Vulkan backend or no device is usable: Vulkan 1.2 with buffer device
   addresses. The first discrete GPU is chosen, else an integrated one; the environment variable
   SPINGALETT_GPU_DEVICE picks one by its index in the Vulkan device list. */
SPINGALETT_API const char *spingalett_gpu_device(void);
/* Precision of the GPU's matrix products: SPINGALETT_PRECISION_FLOAT32 (the default: single precision, as on
   the CPU) or SPINGALETT_PRECISION_BFLOAT16 (the operands rounded to bfloat16, which keeps 8 bits of mantissa, and
   multiplied on the GPU's matrix units with the products added in single precision: faster; the
   layers' outputs but the output layer's, the network's inputs among them, and their gradients are
   kept in memory as bfloat16, the passes between products compute in single precision, and the
   parameters, their gradients and the optimizer stay in single precision, as with PyTorch's
   autocast). Devices without bfloat16
   cooperative matrices keep single precision. Applies from the next spingalett_train(), spingalett_predict() or
   spingalett_evaluate() call, or the next trainer; results stay deterministic. Returns whether the GPU
   multiplies in that precision (false without a device, or for bfloat16 without its matrix units);
   other values are ignored and return false. */
SPINGALETT_API bool spingalett_set_gpu_precision(SpingalettPrecisionMode precision);
SPINGALETT_API SpingalettPrecisionMode spingalett_get_gpu_precision(void);
SPINGALETT_API unsigned spingalett_get_num_threads(void);
SPINGALETT_API void spingalett_set_num_threads(unsigned n);

SPINGALETT_API void spingalett_set_log_callback(SpingalettLogCallback cb);
SPINGALETT_API void spingalett_set_log_level(SpingalettLogLevel level);

/* Seeds the calling thread's generator (weight init, shuffling, dropout) for reproducible runs. */
SPINGALETT_API void spingalett_seed(uint64_t seed);

SPINGALETT_API void spingalett_set_verbose(bool enabled);
SPINGALETT_API bool spingalett_get_verbose(void);

#define spingalett_network_new(...) spingalett_network_new_args((SpingalettNetworkArgs){__VA_ARGS__})
SPINGALETT_API SpingalettNetwork *spingalett_network_new_args(SpingalettNetworkArgs args);

/* Appends a layer and returns its index; the first one is the input layer (index 0). A layer reads
   the one before it unless .inputs names others, so that networks can be graphs: residual blocks,
   branches that are concatenated. The network's output is its last layer, which every other layer
   must feed, directly or through later ones, before it trains or predicts. Errors leave the network
   unchanged, set the error (spingalett_last_error_code()) and return SPINGALETT_NO_LAYER. */
#define SPINGALETT_NO_LAYER UINT32_MAX
#define spingalett_layer(...) spingalett_append_layer((SpingalettLayerArgs){__VA_ARGS__})
#define spingalett_conv2d(...) spingalett_append_layer((SpingalettLayerArgs){.type = SPINGALETT_LAYER_CONV2D, __VA_ARGS__})
#define spingalett_max_pool2d(...) spingalett_append_layer((SpingalettLayerArgs){.type = SPINGALETT_LAYER_MAX_POOL2D, __VA_ARGS__})
#define spingalett_avg_pool2d(...) spingalett_append_layer((SpingalettLayerArgs){.type = SPINGALETT_LAYER_AVG_POOL2D, __VA_ARGS__})
#define spingalett_batch_norm(...) spingalett_append_layer((SpingalettLayerArgs){.type = SPINGALETT_LAYER_BATCH_NORM, __VA_ARGS__})
/* The sum of .inputs (layers of one shape), then .act_func (SPINGALETT_ACT_NONE for the sum alone):
   spingalett_add_layers(.net = net, .inputs = {x, y}, .act_func = SPINGALETT_ACT_RELU) closes a residual block. */
#define spingalett_add_layers(...) spingalett_append_layer((SpingalettLayerArgs){.type = SPINGALETT_LAYER_ADD, __VA_ARGS__})
/* .inputs side by side along the channels (layers of one height and width), then .act_func (SPINGALETT_ACT_NONE
   for none). */
#define spingalett_concat_layers(...) spingalett_append_layer((SpingalettLayerArgs){.type = SPINGALETT_LAYER_CONCAT, __VA_ARGS__})
/* The mean of each channel over its cells (1 x 1 x channels); a pooling layer, without activation. */
#define spingalett_global_avg_pool2d(...) spingalett_append_layer((SpingalettLayerArgs){.type = SPINGALETT_LAYER_GLOBAL_AVG_POOL, __VA_ARGS__})
/* A transposed convolution (upsamples with learned weights: kernel 2, stride 2 doubles the size). */
#define spingalett_conv_transpose2d(...) spingalett_append_layer((SpingalettLayerArgs){.type = SPINGALETT_LAYER_CONV_TRANSPOSE2D, __VA_ARGS__})
/* Upsampling by .stride (default 2), nearest or (.upsample = SPINGALETT_UPSAMPLE_BILINEAR) bilinear. */
#define spingalett_upsample2d(...) spingalett_append_layer((SpingalettLayerArgs){.type = SPINGALETT_LAYER_UPSAMPLE, __VA_ARGS__})
/* Layer normalization over each cell's channels (a dense layer's outputs). */
#define spingalett_layer_norm(...) spingalett_append_layer((SpingalettLayerArgs){.type = SPINGALETT_LAYER_LAYER_NORM, __VA_ARGS__})
SPINGALETT_API uint32_t spingalett_append_layer(SpingalettLayerArgs args);

/* Describing a network. Layers are numbered in the order they were added, every layer after its
   inputs. */
SPINGALETT_API uint32_t spingalett_layer_count(const SpingalettNetwork *net);      /* input layer included */
SPINGALETT_API bool spingalett_network_layer(const SpingalettNetwork *net, uint32_t index, SpingalettNetworkLayer *layer);
SPINGALETT_API uint32_t spingalett_input_size(const SpingalettNetwork *net);
SPINGALETT_API uint32_t spingalett_output_size(const SpingalettNetwork *net);
SPINGALETT_API uint64_t spingalett_parameter_count(const SpingalettNetwork *net);   /* weights and biases */
SPINGALETT_API SpingalettLossFunction spingalett_network_loss(const SpingalettNetwork *net);
SPINGALETT_API uint64_t spingalett_optimizer_steps(const SpingalettNetwork *net);   /* steps taken (Adam's t) */

/* Copies the parameters feeding layer `index` (1 to layers - 1) out of or into the network: count
   must be the layer's weight_count, or its bias_count for biases and the running statistics of
   batch normalization layers. Dense weights are [outputs][inputs], conv weights
   [filters][kernel_h][kernel_w][input channels / groups], batch normalization's weights and biases
   its gamma and beta. Return false (with the error set) otherwise. */
SPINGALETT_API bool spingalett_get_parameters(const SpingalettNetwork *net, uint32_t index, SpingalettParameterKind kind,
                                              float *values, uint64_t count);
SPINGALETT_API bool spingalett_set_parameters(SpingalettNetwork *net, uint32_t index, SpingalettParameterKind kind,
                                              const float *values, uint64_t count);

SPINGALETT_API float spingalett_activate(float x, SpingalettActivationFunction act_func);
SPINGALETT_API float spingalett_derivative(float x, SpingalettActivationFunction act_func);

#define spingalett_forward(...) spingalett_forward_args((SpingalettForwardArgs){__VA_ARGS__})
SPINGALETT_API float *spingalett_forward_args(SpingalettForwardArgs args);

/* Built-in learning-rate schedules (see SpingalettLRScheduleParams). */
SPINGALETT_API float spingalett_lr_cosine_decay(size_t epoch, size_t total_epochs, float initial_lr, void *params);
SPINGALETT_API float spingalett_lr_linear_warmup(size_t epoch, size_t total_epochs, float initial_lr, void *params);
SPINGALETT_API float spingalett_lr_step_decay(size_t epoch, size_t total_epochs, float initial_lr, void *params);
SPINGALETT_API float spingalett_lr_warmup_cosine(size_t epoch, size_t total_epochs, float initial_lr, void *params);

/* Batched inference: writes the outputs of all samples. Uses matrix-matrix products on every
   backend, so it is much faster than calling spingalett_forward() per sample. Returns false on error. */
#define spingalett_predict(...) spingalett_predict_args((SpingalettPredictArgs){__VA_ARGS__})
SPINGALETT_API bool spingalett_predict_args(SpingalettPredictArgs args);

/* Mean loss and accuracy over a data set. The loss is the one spingalett_train() reports: the sum over the
   outputs of the squared error for SPINGALETT_LOSS_MSE (whose gradient spingalett_train() follows up to a factor of 2),
   the cross-entropy for SPINGALETT_LOSS_CROSS_ENTROPY. On error both metrics are NaN. */
#define spingalett_evaluate(...) spingalett_evaluate_args((SpingalettEvaluateArgs){__VA_ARGS__})
SPINGALETT_API SpingalettEvalMetrics spingalett_evaluate_args(SpingalettEvaluateArgs args);

#define spingalett_train(...) spingalett_train_args((SpingalettTrainArgs){__VA_ARGS__})
SPINGALETT_API SpingalettTrainReport spingalett_train_args(SpingalettTrainArgs args);

/*
 * Data sets in the GPU's memory: count rows of size floats copied to the device once, for the
 * device_* fields of spingalett_train(), spingalett_predict() and spingalett_evaluate(). With SPINGALETT_COMPUTE_VULKAN those calls read their
 * chunks on the device, where spingalett_train() gathers, augments and smooths them, instead of copying samples
 * from the host for every pass; on the CPU they copy the rows back first. With SPINGALETT_PRECISION_BFLOAT16 a set
 * makes a copy of its rows as bfloat16 on first use (half its size again), which networks read as the
 * inputs they keep as such. A set's row size is the network's input (or output) size, and it holds at
 * least the call's samples. NULL without a usable GPU, or when its memory runs out. A set may serve
 * several networks and threads at once, and is freed once no call uses it.
 */
SPINGALETT_API SpingalettDeviceData *spingalett_device_data_new(const float *values, uint32_t count, uint32_t size);
SPINGALETT_API void spingalett_device_data_free(SpingalettDeviceData *data);
SPINGALETT_API uint32_t spingalett_device_data_count(const SpingalettDeviceData *data);
SPINGALETT_API uint32_t spingalett_device_data_size(const SpingalettDeviceData *data);
/* Copies rows first .. first + count - 1 to values [count x size]. */
SPINGALETT_API bool spingalett_device_data_read(const SpingalettDeviceData *data, uint32_t first, uint32_t count,
                                               float *values);

/*
 * Low-level training, for custom loops and losses. A trainer runs forward and backward passes on
 * batches of up to max_batch samples; backward passes add to the network's gradient (grad_weights,
 * grad_biases hold the sum over the samples since the last step) and a step applies the mean of
 * that sum with the given optimizer, so a step's batch can be split into several backward passes.
 * Optimizer state and the step count live in the network and are shared with spingalett_train().
 *
 * With SPINGALETT_COMPUTE_VULKAN (when spingalett_trainer_new() is called) the passes run on the GPU, which keeps
 * the parameters, gradients and optimizer state between them: functions that read the network
 * (spingalett_predict(), spingalett_save(), spingalett_get_parameters() and the others) copy them back first,
 * and parameters set on the host go to the GPU before the next forward pass. A trainer is used from
 * one thread at a time, and the network is not read from another thread while it runs a pass.
 */
SPINGALETT_API SpingalettTrainer *spingalett_trainer_new(SpingalettNetwork *net, uint32_t max_batch);
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
SPINGALETT_API bool spingalett_trainer_step(SpingalettTrainer *trainer, const SpingalettOptimizerArgs *optimizer);
/* Discards the gradient accumulated since the last step. */
SPINGALETT_API void spingalett_trainer_zero_grad(SpingalettTrainer *trainer);
/* Label smoothing for spingalett_trainer_backward() and spingalett_train_on_batch(), as
   SpingalettTrainArgs.label_smoothing (0, the default, uses the targets as given). False when out of [0, 1). */
SPINGALETT_API bool spingalett_trainer_set_label_smoothing(SpingalettTrainer *trainer, float label_smoothing);
/* Forward, backward with the network's loss and a step on one batch; returns its mean loss
   (NaN on error). */
SPINGALETT_API float spingalett_train_on_batch(SpingalettTrainer *trainer, const float *inputs, const float *targets,
                                              uint32_t count, const SpingalettOptimizerArgs *optimizer);

#define spingalett_save(...) spingalett_save_args((SpingalettSaveArgs){__VA_ARGS__})
SPINGALETT_API void spingalett_save_args(SpingalettSaveArgs args);

/* Reads a .slett file of any format version (1 to SPINGALETT_FORMAT_VERSION). Quantized weights are
   expanded to float. */
SPINGALETT_API SpingalettNetwork *spingalett_load(const char *filename);

/* The bytes save_spingalett writes, in memory (aligned to 64 bytes, so they also serve as a model
   image for spingalett_model_init). *size receives their count. Release with spingalett_free.
   Returns NULL on error. */
SPINGALETT_API void *spingalett_save_to_memory(const SpingalettNetwork *net, SpingalettPrecisionMode precision,
                                               bool save_optimizer, size_t *size);
/* Reads a .slett file image of any format version from memory; data is not modified and need not
   be aligned. */
SPINGALETT_API SpingalettNetwork *spingalett_load_from_memory(const void *data, size_t size);
/* Releases memory the library returned (spingalett_save_to_memory). */
SPINGALETT_API void spingalett_free(void *ptr);

/*
 * ONNX import: a model of the operators Spingalett has (Conv and ConvTranspose with groups, Gemm,
 * MatMul with a constant right operand, MaxPool, AveragePool, GlobalAveragePool (or ReduceMean over
 * the height and width), BatchNormalization, LayerNormalization over a vector or over a map's
 * channels (between Transposes to channels last and back), Resize and Upsample by integer factors
 * (nearest, or linear between the cells' centres), Add, Concat along the channels, Relu, Sigmoid,
 * Tanh, LeakyRelu with slope 0.01, Softmax over vectors, Flatten, Reshape that flattens, Identity,
 * Dropout, Constant), as a network to train further, save or deploy. The network takes
 * channels-last samples: an input of shape [N, C, H, W] becomes an input layer of H x W x C, so
 * images in NCHW order must be transposed to H, W, C, and maps come out channels last too; a dense
 * layer after a flattened map gets its weight columns reordered to match. Its loss is cross-entropy
 * when the output layer is a softmax or sigmoid, else mean squared error. Errors name the operator
 * or node that cannot be imported. Weights in external data files are read when the model comes
 * from a path (the files in its folder), not from memory.
 */
SPINGALETT_API SpingalettNetwork *spingalett_import_onnx(const char *path);
SPINGALETT_API SpingalettNetwork *spingalett_import_onnx_from_memory(const void *data, size_t size);

/*
 * PyTorch weights into a network of the same architecture: a state dict saved with torch.save
 * (.pt, .pth; also a checkpoint dictionary holding one under "state_dict", "model_state_dict" or
 * "model") or a .safetensors file. The pickle inside torch.save files is read without running any
 * of it: only dictionaries of tensors are recognized. The tensors of each module (the name up to
 * its last dot: "features.0" for "features.0.weight") go, module by module, to the layers with
 * parameters in their order: dense layers take weight [out, in] and bias, convolutions weight
 * [out, in / groups, kh, kw] and bias, transposed convolutions (ConvTranspose2d) weight
 * [in, out / groups, kh, kw] and bias, batch normalizations weight, bias, running_mean and
 * running_var, layer normalizations (LayerNorm over the channels) weight and bias. The modules come
 * in the order of the state dict (torch.save files keep it) or, in
 * safetensors files, which sort names, in natural order of their names ("2" before "10", as
 * nn.Sequential numbers them); `modules` (module_count names, or NULL) gives the order explicitly.
 * Weights are reordered for channels-last data as spingalett_import_onnx() does. On error (a shape
 * that does not fit, names the file does not have) the network is unchanged.
 */
SPINGALETT_API bool spingalett_load_pytorch(SpingalettNetwork *net, const char *path, const char *const *modules,
                                            uint32_t module_count);
SPINGALETT_API bool spingalett_load_pytorch_from_memory(SpingalettNetwork *net, const void *data, size_t size,
                                                        const char *const *modules, uint32_t module_count);

/*
 * Deployment. A SpingalettModel (Spingalett.Inference.h) is a read-only network that computes in the
 * precision its weights are stored in: INT8, INT4 and INT2 layers use integer kernels. The functions
 * below create models that own their image; release them with spingalett_model_free. Models made by
 * spingalett_model_init over a caller's image need no release.
 */

/* A network quantized (or converted) to precision for inference. */
SPINGALETT_API SpingalettModel *spingalett_model_from_network(const SpingalettNetwork *net, SpingalettPrecisionMode precision);
/* Reads a .slett file. Images of format version 3 are used as stored; older ones are converted
   in the precision they were saved in. */
SPINGALETT_API SpingalettModel *spingalett_model_load(const char *path);
/* Same, from an image in memory, which is copied. */
SPINGALETT_API SpingalettModel *spingalett_model_from_memory(const void *data, size_t size);
SPINGALETT_API void spingalett_model_free(SpingalettModel *model);
/* Batched inference: inputs [count x input_size] give outputs [count x output_size]. Uses all
   threads in SPINGALETT_COMPUTE_OPENMP mode. Returns false on error. */
SPINGALETT_API bool spingalett_model_predict(const SpingalettModel *model, const float *inputs, uint32_t count,
                                             float *outputs);
/* Mean loss (the model's loss function, as spingalett_evaluate() computes it) and accuracy over a data set. */
SPINGALETT_API SpingalettEvalMetrics spingalett_model_evaluate(const SpingalettModel *model, const float *inputs,
                                                     const float *targets, uint32_t count);

/*
 * Writes net, in precision and without optimizer state, as a C header for compiling the model into
 * a program or firmware: a 16-byte aligned `static const uint8_t name[]` holding the .slett image,
 * and macros NAME_SIZE, NAME_INPUTS, NAME_OUTPUTS and NAME_WORKSPACE (bytes for
 * spingalett_model_run), where NAME is name in upper case. name must be a C identifier.
 */
SPINGALETT_API bool spingalett_export_c_header(const SpingalettNetwork *net, const char *path, const char *name,
                                               SpingalettPrecisionMode precision);

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
/* CIFAR binary batches, read one after the other: records of label bytes and a 32 x 32 RGB image
   stored plane by plane, read as 32 x 32 x 3 channels-last samples scaled to [0, 1] with one-hot
   targets. num_classes chooses the variant: 10 for CIFAR-10 (data_batch_1.bin ... test_batch.bin,
   one label byte), 100 for CIFAR-100's fine labels or 20 for its coarse ones (train.bin, test.bin,
   two label bytes). */
SPINGALETT_API bool spingalett_load_cifar(const char *const *paths, uint32_t path_count, uint32_t num_classes,
                                          SpingalettDataset *dataset);
/* Gives the data set copies of count class names (count must equal target_size); NULL clears them. */
SPINGALETT_API bool spingalett_dataset_set_class_names(SpingalettDataset *dataset, const char *const *names,
                                                       uint32_t count);
/* Shuffles the samples with the library's generator (see spingalett_seed). */
SPINGALETT_API void spingalett_dataset_shuffle(SpingalettDataset *dataset);
/* Moves the last count samples into *tail, e.g. to hold out a validation set. */
SPINGALETT_API bool spingalett_dataset_split(SpingalettDataset *dataset, uint32_t count, SpingalettDataset *tail);
SPINGALETT_API void spingalett_dataset_free(SpingalettDataset *dataset);

/*
 * .slettd data set files: compact binary storage that loads straight into a SpingalettDataset.
 * Values are stored in the smallest encoding chosen per stream (inputs and each set of targets)
 * and compressed with adaptive context models (a binary range coder, or a faster rANS coder of
 * half-bytes for dense data such as photographs), in independently decodable chunks with CRC-32
 * checksums. Files can also record the input shape, class names and further sets of targets for
 * the same samples (e.g. CIFAR-100's fine and coarse labels). They load whole (chunks decode in
 * parallel with OpenMP), stream into spingalett_train() through a reader that holds a few chunks at a time,
 * or stay in memory in their compact form. See docs/DatasetFormat.md for the layout.
 */
typedef enum {
    SPINGALETT_DATASET_ENCODING_AUTO,          /* the smallest lossless one of U8_UNIT, FP16 and FLOAT32;
                                       one-hot target rows are stored as CLASS */
    SPINGALETT_DATASET_ENCODING_FLOAT32,
    SPINGALETT_DATASET_ENCODING_FP16,          /* IEEE half; lossy unless every value is a half */
    SPINGALETT_DATASET_ENCODING_BFLOAT16,      /* lossy: 8 mantissa bits */
    SPINGALETT_DATASET_ENCODING_U8_UNIT,       /* q / 255 for q in 0..255, exact for 8-bit data in [0, 1] */
    SPINGALETT_DATASET_ENCODING_U8_AFFINE,     /* per feature min + q * (max - min) / 255: lossy 8-bit quantization */
    SPINGALETT_DATASET_ENCODING_CLASS,         /* targets only: the argmax of each row, stored as a class index */
    SPINGALETT_DATASET_ENCODING_COUNT
} SpingalettDatasetEncoding;

/* A further set of targets for the samples of a data set, saved next to its own targets. */
typedef struct {
    const char *name;               /* e.g. "coarse"; NULL for none */
    uint32_t size;                  /* targets per sample (classes, for one-hot rows) */
    const float *targets;           /* [count x size] */
    const char *const *class_names; /* size names, or NULL */
    SpingalettDatasetEncoding encoding;       /* AUTO: the smallest lossless one */
} SpingalettTargetSet;

typedef struct {
    SpingalettDatasetEncoding input_encoding;
    SpingalettDatasetEncoding target_encoding;
    bool no_compression;            /* store the encoded bytes as they are (fastest to read) */
    const char *target_name;        /* name of the data set's own targets (e.g. "fine"), or NULL */
    const SpingalettTargetSet *extra_targets;   /* further sets of targets, or NULL */
    uint32_t extra_target_count;
} SpingalettDatasetSaveOptions;

typedef struct {
    uint32_t count, input_size, target_size;
    SpingalettDatasetEncoding input_encoding, target_encoding;
    uint32_t chunk_count;
    uint64_t file_size;
    uint32_t format_version;
    uint32_t height, width, channels;   /* input shape, 0 when the file does not record it */
    uint32_t target_set_count;      /* sets of targets in the file (at least 1) */
    uint32_t target_set;            /* the set this reader serves: target_size and target_encoding
                                       describe it */
} SpingalettDatasetInfo;

typedef struct {
    bool shuffle;                   /* every pass in a new random order (the library's generator,
                                       drawn when the reader opens: see spingalett_seed) */
    bool in_memory;                 /* decode the file once and keep its values in their compact
                                       form (one byte per 8-bit value, a quarter of float), converted
                                       a batch at a time; passes then shuffle all samples at once
                                       rather than chunk by chunk */
    bool no_prefetch;               /* no background thread. By default one decodes the next chunks
                                       while the caller trains if a processor is left for it (fewer
                                       OpenMP threads than processors); otherwise, and with this
                                       set, chunks decode when they are needed, several at a time on
                                       the OpenMP threads. The samples come in the same order. */
    uint32_t target_set;            /* which set of targets to serve (0: the data set's own) */
} SpingalettDatasetReaderOptions;

/* Streams the samples of a .slettd file chunk by chunk. */
typedef struct SpingalettDatasetReader SpingalettDatasetReader;

/* Writes dataset to path (SPINGALETT_DATASET_EXTENSION is appended when the name has none);
   options NULL = all defaults. */
SPINGALETT_API bool spingalett_save_dataset(const SpingalettDataset *dataset, const char *path,
                                            const SpingalettDatasetSaveOptions *options);
SPINGALETT_API bool spingalett_load_dataset(const char *path, SpingalettDataset *dataset);
/* Same, from a file image in memory (e.g. a const array in flash); data is not modified. */
SPINGALETT_API bool spingalett_load_dataset_from_memory(const void *data, size_t size, SpingalettDataset *dataset);
/* Loads the inputs with set target_set of targets (0 is what spingalett_load_dataset loads). */
SPINGALETT_API bool spingalett_load_dataset_targets(const char *path, uint32_t target_set, SpingalettDataset *dataset);
SPINGALETT_API bool spingalett_load_dataset_from_memory_targets(const void *data, size_t size, uint32_t target_set,
                                                                SpingalettDataset *dataset);

/* A reader keeps a few chunks in memory (decoded to their compact form, converted to float a
   batch at a time), however large the file is. With shuffle, every pass visits the chunks and the
   samples within each chunk in a new random order. Same as spingalett_dataset_open_ex with only
   shuffle set. */
SPINGALETT_API SpingalettDatasetReader *spingalett_dataset_open(const char *path, bool shuffle);
/* options NULL = defaults (in file order, streaming, the first set of targets). */
SPINGALETT_API SpingalettDatasetReader *spingalett_dataset_open_ex(const char *path, const SpingalettDatasetReaderOptions *options);
/* A reader over 8-bit inputs in memory (value q stands for q / 255, as U8_UNIT) and float
   targets: both are copied, the inputs as bytes, and converted to float a batch at a time. */
SPINGALETT_API SpingalettDatasetReader *spingalett_dataset_open_u8(const uint8_t *inputs, const float *targets,
                                                                   uint32_t count, uint32_t input_size,
                                                                   uint32_t target_size, bool shuffle);
/* Names recorded in the file: the name of a set of targets, class `index` of a set, or NULL. */
SPINGALETT_API const char *spingalett_dataset_target_set_name(const SpingalettDatasetReader *reader,
                                                              uint32_t target_set);
SPINGALETT_API const char *spingalett_dataset_class_name(const SpingalettDatasetReader *reader, uint32_t target_set,
                                                         uint32_t index);
/* Targets per sample of a set (0 when there is no such set). */
SPINGALETT_API uint32_t spingalett_dataset_target_set_size(const SpingalettDatasetReader *reader, uint32_t target_set);
SPINGALETT_API void spingalett_dataset_close(SpingalettDatasetReader *reader);
SPINGALETT_API SpingalettDatasetInfo spingalett_dataset_info(const SpingalettDatasetReader *reader);
/* Copies up to max_samples samples; returns how many. Returns 0 once a pass is complete (or on
   error, which is sticky); the next call starts a new pass. */
SPINGALETT_API uint32_t spingalett_dataset_read(SpingalettDatasetReader *reader, float *inputs, float *targets,
                                                uint32_t max_samples);
/* A SpingalettDataGeneratorFn over a reader, for .generator = spingalett_dataset_generator, .generator_data = reader. */
SPINGALETT_API uint32_t spingalett_dataset_generator(float *inputs, float *targets, uint32_t requested, void *reader);

SPINGALETT_API void spingalett_print_network(const SpingalettNetwork *net);
SPINGALETT_API void spingalett_network_free(SpingalettNetwork *net);

#ifdef __cplusplus
}
#endif

/* The names of 0.x, without the prefix (layer(), train(), NeuralNetwork, ACT_RELU, ...), for programs
   written for them; define SPINGALETT_NO_SHORT_NAMES before including this header to leave them out. */
#if !defined(SPINGALETT_NO_SHORT_NAMES)
#include "Spingalett.Short.h"
#endif
