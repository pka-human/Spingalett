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
    LOG_DEBUG,
    LOG_INFO,
    LOG_WARNING,
    LOG_ERROR,
    LOG_NONE
} LogLevel;

typedef void (*LogCallback)(LogLevel level, const char *message);

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
    AUTOSAVE_OFF,
    AUTOSAVE_OVERWRITE,
    AUTOSAVE_NEW_FILES
} AutoSaveMode;

/* A network being built or trained. Its contents are private: describe it with
   spingalett_network_layer() and friends, and read or write its parameters with
   spingalett_get_parameters() / spingalett_set_parameters(). */
typedef struct NeuralNetwork NeuralNetwork;

/* Kinds of layers: LayerType, in Spingalett.Inference.h (the inference engine runs them all). */

/* Layer `index` of a network (0 is the input layer), see spingalett_network_layer(). */
typedef struct {
    LayerType type;                 /* the input layer is LAYER_DENSE */
    uint32_t height, width, channels;   /* output shape; a dense layer is 1 x 1 x outputs */
    uint32_t outputs;               /* height * width * channels */
    ActivationFunction activation;  /* ACT_NONE for the input layer and pooling layers */
    float dropout_rate;
    uint32_t kernel_h, kernel_w;    /* conv and pooling: window over the previous layer */
    uint32_t stride_h, stride_w;
    uint32_t padding_h, padding_w;  /* zeros (conv) or ignored cells (pooling) on each side */
    uint64_t weight_count;          /* parameters feeding the layer: dense outputs x inputs, conv */
    uint64_t bias_count;            /* filters x kernel_h x kernel_w x input channels; one bias per
                                       output (dense) or filter (conv); none for the input and
                                       pooling layers */
} SpingalettNetworkLayer;

/* Which parameters spingalett_get_parameters() and spingalett_set_parameters() copy. */
typedef enum {
    PARAM_WEIGHTS,
    PARAM_BIASES,
    PARAM_WEIGHT_GRADIENTS,         /* the gradient left by the last training step or backward pass */
    PARAM_BIAS_GRADIENTS,
    PARAM_KIND_COUNT
} ParameterKind;

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
 * how many were written; returning 0 ends the epoch (a 0 in answer to the first request of an
 * epoch is retried once before training stops). Shuffling and augmentation are up to the
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
    uint32_t neurons_amount;        /* dense: outputs; input layer: its size (or give its shape) */
    ActivationFunction act_func;    /* dense and conv layers (pooling layers have none) */
    WeightInitialization weight_initialization;
    float dropout_rate;             /* [0, 1): inverted dropout on this layer's outputs during
                                       training; ignored on the input and output layers */
    LayerType type;                 /* LAYER_DENSE unless set; see conv2d(), max_pool2d(), avg_pool2d() */
    uint32_t height, width, channels;   /* input layer: the shape of a sample (channels-last), e.g.
                                       28 x 28 x 1 for MNIST; omitted: 1 x 1 x neurons_amount */
    uint32_t filters;               /* conv: output channels */
    uint32_t kernel;                /* conv and pooling: square window size */
    uint32_t stride;                /* 0 = 1 for conv, the window size for pooling */
    uint32_t padding;               /* cells added on each side: zeros for conv, ignored by pooling;
                                       kernel / 2 keeps the size of odd windows with stride 1 */
    uint32_t kernel_h, kernel_w;    /* per-axis overrides of kernel, stride and padding (0 = unset) */
    uint32_t stride_h, stride_w;
    uint32_t padding_h, padding_w;
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

/* Appends a layer; the first one is the input layer. Errors leave the network unchanged and set
   the error (spingalett_last_error_code()). */
#define layer(...) layer_struct_arguments((LayerArgs){__VA_ARGS__})
#define conv2d(...) layer_struct_arguments((LayerArgs){.type = LAYER_CONV2D, __VA_ARGS__})
#define max_pool2d(...) layer_struct_arguments((LayerArgs){.type = LAYER_MAX_POOL2D, __VA_ARGS__})
#define avg_pool2d(...) layer_struct_arguments((LayerArgs){.type = LAYER_AVG_POOL2D, __VA_ARGS__})
SPINGALETT_API void layer_struct_arguments(LayerArgs args);

/* Describing a network. */
SPINGALETT_API uint32_t spingalett_layer_count(const NeuralNetwork *net);      /* input layer included */
SPINGALETT_API bool spingalett_network_layer(const NeuralNetwork *net, uint32_t index, SpingalettNetworkLayer *layer);
SPINGALETT_API uint32_t spingalett_input_size(const NeuralNetwork *net);
SPINGALETT_API uint32_t spingalett_output_size(const NeuralNetwork *net);
SPINGALETT_API uint64_t spingalett_parameter_count(const NeuralNetwork *net);   /* weights and biases */
SPINGALETT_API LossFunction spingalett_network_loss(const NeuralNetwork *net);
SPINGALETT_API uint64_t spingalett_optimizer_steps(const NeuralNetwork *net);   /* steps taken (Adam's t) */

/* Copies the parameters feeding layer `index` (1 to layers - 1) out of or into the network: count
   must be the layer's weight_count or bias_count. Dense weights are [outputs][inputs], conv weights
   [filters][kernel_h][kernel_w][input channels]. Return false (with the error set) otherwise. */
SPINGALETT_API bool spingalett_get_parameters(const NeuralNetwork *net, uint32_t index, ParameterKind kind,
                                              float *values, uint64_t count);
SPINGALETT_API bool spingalett_set_parameters(NeuralNetwork *net, uint32_t index, ParameterKind kind,
                                              const float *values, uint64_t count);

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

/* Reads a .slett file of any format version (1 to SPINGALETT_FORMAT_VERSION). Quantized weights are
   expanded to float. */
SPINGALETT_API NeuralNetwork *load_spingalett(const char *filename);

/* The bytes save_spingalett writes, in memory (aligned to 64 bytes, so they also serve as a model
   image for spingalett_model_init). *size receives their count. Release with spingalett_free.
   Returns NULL on error. */
SPINGALETT_API void *spingalett_save_to_memory(const NeuralNetwork *net, PrecisionMode precision,
                                               bool save_optimizer, size_t *size);
/* Reads a .slett file image of any format version from memory; data is not modified and need not
   be aligned. */
SPINGALETT_API NeuralNetwork *load_spingalett_from_memory(const void *data, size_t size);
/* Releases memory the library returned (spingalett_save_to_memory). */
SPINGALETT_API void spingalett_free(void *ptr);

/*
 * Deployment. A SpingalettModel (Spingalett.Inference.h) is a read-only network that computes in the
 * precision its weights are stored in: INT8, INT4 and INT2 layers use integer kernels. The functions
 * below create models that own their image; release them with spingalett_model_free. Models made by
 * spingalett_model_init over a caller's image need no release.
 */

/* A network quantized (or converted) to precision for inference. */
SPINGALETT_API SpingalettModel *spingalett_model_from_network(const NeuralNetwork *net, PrecisionMode precision);
/* Reads a .slett file. Images of format version 3 are used as stored; older ones are converted
   in the precision they were saved in. */
SPINGALETT_API SpingalettModel *spingalett_model_load(const char *path);
/* Same, from an image in memory, which is copied. */
SPINGALETT_API SpingalettModel *spingalett_model_from_memory(const void *data, size_t size);
SPINGALETT_API void spingalett_model_free(SpingalettModel *model);
/* Batched inference: inputs [count x input_size] give outputs [count x output_size]. Uses all
   threads in COMPUTE_OPENMP mode. Returns false on error. */
SPINGALETT_API bool spingalett_model_predict(const SpingalettModel *model, const float *inputs, uint32_t count,
                                             float *outputs);
/* Mean loss (the model's loss function, as evaluate() computes it) and accuracy over a data set. */
SPINGALETT_API EvalMetrics spingalett_model_evaluate(const SpingalettModel *model, const float *inputs,
                                                     const float *targets, uint32_t count);

/*
 * Writes net, in precision and without optimizer state, as a C header for compiling the model into
 * a program or firmware: a 16-byte aligned `static const uint8_t name[]` holding the .slett image,
 * and macros NAME_SIZE, NAME_INPUTS, NAME_OUTPUTS and NAME_WORKSPACE (bytes for
 * spingalett_model_run), where NAME is name in upper case. name must be a C identifier.
 */
SPINGALETT_API bool spingalett_export_c_header(const NeuralNetwork *net, const char *path, const char *name,
                                               PrecisionMode precision);

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

/*
 * .slettd data set files: compact binary storage that loads straight into a SpingalettDataset.
 * Values are stored in the smallest encoding chosen per stream (inputs, targets) and compressed
 * with an adaptive context-model range coder, in independently decodable chunks with CRC-32
 * checksums. They can be loaded whole (chunks decode in parallel with OpenMP) or streamed into
 * train() as a generator. See docs/DatasetFormat.md for the layout.
 */
typedef enum {
    DATASET_ENCODING_AUTO,          /* the smallest lossless one of U8_UNIT, FP16 and FLOAT32;
                                       one-hot target rows are stored as CLASS */
    DATASET_ENCODING_FLOAT32,
    DATASET_ENCODING_FP16,          /* IEEE half; lossy unless every value is a half */
    DATASET_ENCODING_BFLOAT16,      /* lossy: 8 mantissa bits */
    DATASET_ENCODING_U8_UNIT,       /* q / 255 for q in 0..255, exact for 8-bit data in [0, 1] */
    DATASET_ENCODING_U8_AFFINE,     /* per feature min + q * (max - min) / 255: lossy 8-bit quantization */
    DATASET_ENCODING_CLASS,         /* targets only: the argmax of each row, stored as a class index */
    DATASET_ENCODING_COUNT
} DatasetEncoding;

typedef struct {
    DatasetEncoding input_encoding;
    DatasetEncoding target_encoding;
    bool no_compression;            /* store the encoded bytes as they are (fastest to read) */
} DatasetSaveOptions;

typedef struct {
    uint32_t count, input_size, target_size;
    DatasetEncoding input_encoding, target_encoding;
    uint32_t chunk_count;
    uint64_t file_size;
} SpingalettDatasetInfo;

/* Streams the samples of a .slettd file chunk by chunk. */
typedef struct SpingalettDatasetReader SpingalettDatasetReader;

/* Writes dataset to path (SPINGALETT_DATASET_EXTENSION is appended when the name has none);
   options NULL = all defaults. */
SPINGALETT_API bool spingalett_save_dataset(const SpingalettDataset *dataset, const char *path,
                                            const DatasetSaveOptions *options);
SPINGALETT_API bool spingalett_load_dataset(const char *path, SpingalettDataset *dataset);
/* Same, from a file image in memory (e.g. a const array in flash); data is not modified. */
SPINGALETT_API bool spingalett_load_dataset_from_memory(const void *data, size_t size, SpingalettDataset *dataset);

/* A reader keeps one decoded chunk in memory. With shuffle, every pass visits the chunks and the
   samples within each chunk in a new random order (library generator, see spingalett_seed). */
SPINGALETT_API SpingalettDatasetReader *spingalett_dataset_open(const char *path, bool shuffle);
SPINGALETT_API void spingalett_dataset_close(SpingalettDatasetReader *reader);
SPINGALETT_API SpingalettDatasetInfo spingalett_dataset_info(const SpingalettDatasetReader *reader);
/* Copies up to max_samples samples; returns how many. Returns 0 once a pass is complete (or on
   error, which is sticky); the next call starts a new pass. */
SPINGALETT_API uint32_t spingalett_dataset_read(SpingalettDatasetReader *reader, float *inputs, float *targets,
                                                uint32_t max_samples);
/* A DataGeneratorFn over a reader, for .generator = spingalett_dataset_generator, .generator_data = reader. */
SPINGALETT_API uint32_t spingalett_dataset_generator(float *inputs, float *targets, uint32_t requested, void *reader);

SPINGALETT_API void print_parameters(const NeuralNetwork *net);
SPINGALETT_API void free_network(NeuralNetwork *net);

#ifdef __cplusplus
}
#endif
