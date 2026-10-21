/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/*
 * libspingalett's C API as PHP's FFI parses it: the declarations of Include/Spingalett/Spingalett.h,
 * Spingalett.Runtime.h and Spingalett.Inference.h without the preprocessor. The function-like macros
 * (spingalett_layer(), spingalett_train(), ...) are left out: their *_args functions take the
 * structures themselves. SPINGALETT_RESERVED is written as 8 and SPINGALETT_MAX_INPUTS as 16; the other
 * macros (error codes, SPINGALETT_NO_LAYER) are constants of the PHP classes. Spingalett\Library loads
 * this file with FFI::cdef() and the library it finds, or FFI::load() through Library::preload(). The
 * tests compare every declaration with the C headers' and every structure's layout with the C
 * compiler's; the documentation is in the C headers.
 */

/* The inference engine (Spingalett.Inference.h). */

typedef enum {
    SPINGALETT_ACT_NONE,
    SPINGALETT_ACT_SIGMOID,
    SPINGALETT_ACT_RELU,
    SPINGALETT_ACT_TANH,
    SPINGALETT_ACT_LEAKY_RELU,
    SPINGALETT_ACT_FOO52,
    SPINGALETT_ACT_SOFTMAX,
    SPINGALETT_ACT_GELU,
    SPINGALETT_ACT_GELU_TANH,
    SPINGALETT_ACT_SILU,
    SPINGALETT_ACT_COUNT
} SpingalettActivationFunction;

typedef enum {
    SPINGALETT_LOSS_MSE,
    SPINGALETT_LOSS_CROSS_ENTROPY,
    SPINGALETT_LOSS_SPARSE_CROSS_ENTROPY,
    SPINGALETT_LOSS_COUNT
} SpingalettLossFunction;

typedef enum {
    SPINGALETT_LAYER_DENSE,
    SPINGALETT_LAYER_CONV2D,
    SPINGALETT_LAYER_MAX_POOL2D,
    SPINGALETT_LAYER_AVG_POOL2D,
    SPINGALETT_LAYER_BATCH_NORM,
    SPINGALETT_LAYER_ADD,
    SPINGALETT_LAYER_CONCAT,
    SPINGALETT_LAYER_GLOBAL_AVG_POOL,
    SPINGALETT_LAYER_CONV_TRANSPOSE2D,
    SPINGALETT_LAYER_UPSAMPLE,
    SPINGALETT_LAYER_LAYER_NORM,
    SPINGALETT_LAYER_EMBEDDING,
    SPINGALETT_LAYER_ATTENTION,
    SPINGALETT_LAYER_RMS_NORM,
    SPINGALETT_LAYER_MULTIPLY,
    SPINGALETT_LAYER_TYPE_COUNT
} SpingalettLayerType;

typedef enum {
    SPINGALETT_UPSAMPLE_NEAREST,
    SPINGALETT_UPSAMPLE_BILINEAR,
    SPINGALETT_UPSAMPLE_MODE_COUNT
} SpingalettUpsampleMode;

typedef enum {
    SPINGALETT_PRECISION_FLOAT32,
    SPINGALETT_PRECISION_FP16,
    SPINGALETT_PRECISION_BFLOAT16,
    SPINGALETT_PRECISION_INT8,
    SPINGALETT_PRECISION_INT4,
    SPINGALETT_PRECISION_INT2,
    SPINGALETT_PRECISION_Q8_0,
    SPINGALETT_PRECISION_Q4_0,
    SPINGALETT_PRECISION_Q4_1,
    SPINGALETT_PRECISION_Q5_0,
    SPINGALETT_PRECISION_Q5_1,
    SPINGALETT_PRECISION_Q2_K,
    SPINGALETT_PRECISION_Q3_K,
    SPINGALETT_PRECISION_Q4_K,
    SPINGALETT_PRECISION_Q5_K,
    SPINGALETT_PRECISION_Q6_K,
    SPINGALETT_PRECISION_IQ4_NL,
    SPINGALETT_PRECISION_IQ4_XS,
    SPINGALETT_PRECISION_COUNT
} SpingalettPrecisionMode;

typedef struct {
    uint32_t input_size;
    uint32_t output_size;
    uint32_t layer_count;
    SpingalettLossFunction loss;
    size_t workspace_size;
    const void *image;
    size_t image_size;
    uint32_t max_width_;
    uint32_t max_int_inputs_;
    size_t conv_scratch_;
    void *owner_;
    size_t activations_;
    uint64_t reserved[8];
} SpingalettModel;

typedef struct {
    SpingalettLayerType type;
    uint32_t inputs;
    uint32_t outputs;
    SpingalettActivationFunction activation;
    SpingalettPrecisionMode precision;
    uint32_t in_height, in_width, in_channels;
    uint32_t height, width, channels;
    uint32_t kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w;
    uint32_t groups;
    float epsilon;
    uint32_t input_count;
    uint32_t input_layers[16];
    SpingalettUpsampleMode upsample;
    uint32_t vocabulary;
    uint32_t heads, kv_heads;
    float rope_theta;
    bool causal;
    bool positions;
    bool ignore_padding;
    uint32_t tied;
    uint32_t window;
    uint32_t token_offset;
    uint64_t reserved[8 - 4];
} SpingalettLayerInfo;

int spingalett_model_init(SpingalettModel *model, const void *image, size_t size);
int spingalett_model_run(const SpingalettModel *model, const float *input, float *output,
                         void *workspace);
bool spingalett_model_layer(const SpingalettModel *model, uint32_t index, SpingalettLayerInfo *info);

/* The runtime (Spingalett.Runtime.h): versions, errors, settings, deployment models. */

typedef enum {
    SPINGALETT_LOG_DEBUG,
    SPINGALETT_LOG_INFO,
    SPINGALETT_LOG_WARNING,
    SPINGALETT_LOG_ERROR,
    SPINGALETT_LOG_NONE
} SpingalettLogLevel;

typedef void (*SpingalettLogCallback)(SpingalettLogLevel level, const char *message);

typedef enum {
    SPINGALETT_COMPUTE_SINGLE_THREADED,
    SPINGALETT_COMPUTE_OPENMP,
    SPINGALETT_COMPUTE_OPENBLAS,
    SPINGALETT_COMPUTE_CUDA,
    SPINGALETT_COMPUTE_VULKAN,
    SPINGALETT_COMPUTE_COUNT
} SpingalettComputeMode;

typedef struct {
    float loss;
    float accuracy;
    uint64_t reserved[8];
} SpingalettEvalMetrics;

const char *spingalett_version(void);
int spingalett_last_error_code(void);
const char *spingalett_last_error_message(void);
void spingalett_clear_error(void);
const char *spingalett_cpu_kernels(void);
SpingalettComputeMode spingalett_get_compute_mode(void);
bool spingalett_set_compute_mode(SpingalettComputeMode mode);
unsigned spingalett_get_num_threads(void);
void spingalett_set_num_threads(unsigned n);
void spingalett_set_log_callback(SpingalettLogCallback cb);
void spingalett_set_log_level(SpingalettLogLevel level);
void spingalett_set_verbose(bool enabled);
bool spingalett_get_verbose(void);
SpingalettModel *spingalett_model_load(const char *path);
typedef struct {
    const char *path;
    uint32_t context;
    uint64_t reserved[8];
} SpingalettModelOpenArgs;
SpingalettModel *spingalett_model_open_args(SpingalettModelOpenArgs args);
SpingalettModel *spingalett_model_from_memory(const void *data, size_t size);
void spingalett_model_free(SpingalettModel *model);
bool spingalett_model_predict(const SpingalettModel *model, const float *inputs, uint32_t count,
                              float *outputs);
SpingalettEvalMetrics spingalett_model_evaluate(const SpingalettModel *model, const float *inputs,
                                                const float *targets, uint32_t count);

typedef struct SpingalettSession SpingalettSession;
typedef struct {
    uint32_t max_length;
    uint32_t sequences;
    uint64_t reserved[8];
} SpingalettSessionOptions;
SpingalettSession *spingalett_session_new(const SpingalettModel *model, const SpingalettSessionOptions *options);
void spingalett_session_free(SpingalettSession *session);
uint32_t spingalett_session_vocabulary(const SpingalettSession *session);
uint32_t spingalett_session_length(const SpingalettSession *session, uint32_t sequence);
typedef struct {
    uint32_t sequence;
    const uint32_t *tokens;
    uint32_t count;
    float *logits;
    bool all;
    uint64_t reserved[8];
} SpingalettSessionFeed;
bool spingalett_session_feed(SpingalettSession *session, const SpingalettSessionFeed *feeds, uint32_t count);
bool spingalett_session_rewind(SpingalettSession *session, uint32_t sequence, uint32_t length);
bool spingalett_session_copy(SpingalettSession *session, uint32_t from, uint32_t to);
typedef struct {
    SpingalettSession *session;
    uint32_t sequences;
    const uint32_t *const *prompts;
    const uint32_t *prompt_lengths;
    uint32_t *tokens;
    uint32_t *made;
    uint32_t count;
    float temperature;
    uint32_t top_k;
    float top_p;
    uint64_t seed;
    const uint32_t *stop_tokens;
    uint32_t stop_count;
    uint64_t reserved[8];
} SpingalettSessionGenerateArgs;
uint32_t spingalett_session_generate_args(SpingalettSessionGenerateArgs args);

/* Networks and training (Spingalett.h). */

typedef enum {
    SPINGALETT_INIT_RANDOM,
    SPINGALETT_INIT_XAVIER,
    SPINGALETT_INIT_HE,
    SPINGALETT_INIT_NONE,
    SPINGALETT_INIT_LECUN,
    SPINGALETT_INIT_COUNT
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
    SPINGALETT_AUTOSAVE_OFF,
    SPINGALETT_AUTOSAVE_OVERWRITE,
    SPINGALETT_AUTOSAVE_NEW_FILES
} SpingalettAutoSaveMode;

typedef struct SpingalettNetwork SpingalettNetwork;

typedef struct SpingalettDeviceData SpingalettDeviceData;

typedef struct {
    SpingalettLayerType type;
    uint32_t height, width, channels;
    uint32_t outputs;
    SpingalettActivationFunction activation;
    float dropout_rate;
    uint32_t kernel_h, kernel_w;
    uint32_t stride_h, stride_w;
    uint32_t padding_h, padding_w;
    uint64_t weight_count;
    uint64_t bias_count;
    uint32_t groups;
    float epsilon;
    float momentum;
    uint32_t input_count;
    uint32_t inputs[16];
    SpingalettUpsampleMode upsample;
    uint32_t vocabulary;
    uint32_t heads, kv_heads;
    float rope_theta;
    bool causal;
    bool positions;
    bool ignore_padding;
    uint32_t tied;
    uint32_t window;
    uint32_t token_offset;
    uint64_t reserved[8 - 4];
} SpingalettNetworkLayer;

typedef enum {
    SPINGALETT_PARAM_WEIGHTS,
    SPINGALETT_PARAM_BIASES,
    SPINGALETT_PARAM_WEIGHT_GRADIENTS,
    SPINGALETT_PARAM_BIAS_GRADIENTS,
    SPINGALETT_PARAM_RUNNING_MEAN,
    SPINGALETT_PARAM_RUNNING_VARIANCE,
    SPINGALETT_PARAM_KIND_COUNT
} SpingalettParameterKind;

typedef enum {
    SPINGALETT_MONITOR_AUTO,
    SPINGALETT_MONITOR_TRAIN_LOSS,
    SPINGALETT_MONITOR_VAL_LOSS,
    SPINGALETT_MONITOR_VAL_ACCURACY,
    SPINGALETT_MONITOR_COUNT
} SpingalettMonitorMetric;

typedef struct {
    size_t epoch;
    size_t epochs;
    float train_loss;
    float learning_rate;
    bool has_validation;
    SpingalettEvalMetrics validation;
    SpingalettMonitorMetric monitor;
    size_t best_epoch;
    float best_value;
    bool improved;
    uint64_t reserved[8];
} SpingalettTrainProgress;

typedef bool (*SpingalettTrainCallback)(SpingalettNetwork *net, const SpingalettTrainProgress *progress, void *user_data);

typedef enum {
    SPINGALETT_TRAIN_FAILED,
    SPINGALETT_TRAIN_COMPLETED,
    SPINGALETT_TRAIN_EARLY_STOPPED,
    SPINGALETT_TRAIN_INTERRUPTED,
    SPINGALETT_TRAIN_DIVERGED,
    SPINGALETT_TRAIN_NO_DATA
} SpingalettTrainStatus;

typedef struct {
    SpingalettTrainStatus status;
    size_t epochs_run;
    float train_loss;
    bool has_validation;
    SpingalettEvalMetrics validation;
    SpingalettMonitorMetric monitor;
    size_t best_epoch;
    float best_value;
    bool restored_best;
    uint64_t reserved[8];
} SpingalettTrainReport;

typedef uint32_t (*SpingalettDataGeneratorFn)(float *inputs, float *targets, uint32_t requested, void *user_data);

typedef float (*SpingalettLRSchedulerFn)(size_t epoch, size_t total_epochs, float initial_lr, void *user_data);

typedef struct {
    size_t warmup_epochs;
    size_t step_size;
    float gamma;
    float min_lr;
    uint64_t reserved[8];
} SpingalettLRScheduleParams;

typedef struct {
    SpingalettLossFunction loss_func;
    uint64_t reserved[8];
} SpingalettNetworkArgs;

typedef struct {
    SpingalettNetwork *net;
    uint32_t neurons_amount;
    SpingalettActivationFunction act_func;
    SpingalettWeightInitialization weight_initialization;
    float dropout_rate;
    SpingalettLayerType type;
    uint32_t height, width, channels;
    uint32_t filters;
    uint32_t kernel;
    uint32_t stride;
    uint32_t padding;
    uint32_t kernel_h, kernel_w;
    uint32_t stride_h, stride_w;
    uint32_t padding_h, padding_w;
    uint32_t groups;
    float epsilon;
    float momentum;
    uint32_t inputs[16];
    uint32_t input_count;
    SpingalettUpsampleMode upsample;
    uint32_t output_padding;
    uint32_t output_padding_h, output_padding_w;
    uint32_t vocabulary;
    uint32_t heads;
    uint32_t kv_heads;
    float rope_theta;
    bool causal;
    bool positions;
    bool ignore_padding;
    uint32_t tied;
    uint32_t window;
    uint32_t token_offset;
    uint32_t token_count;
    uint64_t reserved[8 - 5];
} SpingalettLayerArgs;

typedef struct {
    SpingalettNetwork *net;
    const float *input;
    uint64_t reserved[8];
} SpingalettForwardArgs;

typedef struct {
    SpingalettNetwork *net;
    SpingalettTrainingMode training_mode;
    SpingalettTrainingStrategy training_strategy;
    SpingalettOptimizerType optimizer_type;
    const float *inputs;
    const float *targets;
    const SpingalettDeviceData *device_inputs;
    const SpingalettDeviceData *device_targets;
    SpingalettDataGeneratorFn generator;
    void *generator_data;
    uint32_t sample_count;
    uint32_t batch_size;
    bool do_not_shuffle;
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
    SpingalettAutoSaveMode autosave_mode;
    size_t autosave_interval;
    const char *autosave_path;
    bool autosave_do_not_save_optimizer;
    SpingalettPrecisionMode autosave_precision;
    SpingalettTrainCallback callback;
    size_t callback_interval;
    void *callback_data;
    SpingalettLRSchedulerFn lr_scheduler;
    void *lr_scheduler_data;
    const float *val_inputs;
    const float *val_targets;
    const SpingalettDeviceData *device_val_inputs;
    const SpingalettDeviceData *device_val_targets;
    uint32_t val_count;
    SpingalettMonitorMetric monitor;
    size_t early_stopping_patience;
    float early_stopping_min_delta;
    bool restore_best_weights;
    int blas_num_threads;
    uint32_t augment_shift;
    bool augment_flip;
    float label_smoothing;
    float lr_plateau_factor;
    size_t lr_plateau_patience;
    float lr_plateau_min_lr;
    uint32_t accumulation_steps;
    bool lr_per_step;
    bool recompute;
    uint64_t reserved[8 - 1];
} SpingalettTrainArgs;

typedef struct {
    SpingalettNetwork *net;
    const float *inputs;
    const SpingalettDeviceData *device_inputs;
    uint32_t sample_count;
    float *outputs;
    uint64_t reserved[8];
} SpingalettPredictArgs;

typedef struct {
    SpingalettNetwork *net;
    const float *inputs;
    const float *targets;
    const SpingalettDeviceData *device_inputs;
    const SpingalettDeviceData *device_targets;
    uint32_t sample_count;
    uint64_t reserved[8];
} SpingalettEvaluateArgs;

typedef struct {
    SpingalettNetwork *net;
    const char *filename;
    bool do_not_save_optimizer;
    SpingalettPrecisionMode precision;
    uint64_t reserved[8];
} SpingalettSaveArgs;

typedef struct {
    SpingalettOptimizerType type;
    float learning_rate;
    float weight_decay;
    float momentum;
    float beta1;
    float beta2;
    float epsilon;
    float max_grad_norm;
    uint64_t reserved[8];
} SpingalettOptimizerArgs;

typedef struct SpingalettTrainer SpingalettTrainer;

typedef struct {
    uint32_t count;
    uint32_t input_size;
    uint32_t target_size;
    float *inputs;
    float *targets;
    uint32_t height, width, channels;
    char **class_names;
    uint64_t reserved[8];
} SpingalettDataset;

const char *spingalett_gpu_device(void);
const char *spingalett_cuda_device(void);
bool spingalett_set_gpu_precision(SpingalettPrecisionMode precision);
SpingalettPrecisionMode spingalett_get_gpu_precision(void);
void spingalett_seed(uint64_t seed);

/* Networks: spingalett_append_layer() returns SPINGALETT_NO_LAYER (UINT32_MAX) on error. */
SpingalettNetwork *spingalett_network_new_args(SpingalettNetworkArgs args);
uint32_t spingalett_append_layer(SpingalettLayerArgs args);
uint32_t spingalett_layer_count(const SpingalettNetwork *net);
bool spingalett_network_layer(const SpingalettNetwork *net, uint32_t index, SpingalettNetworkLayer *layer);
uint32_t spingalett_input_size(const SpingalettNetwork *net);
uint32_t spingalett_output_size(const SpingalettNetwork *net);
uint32_t spingalett_target_size(const SpingalettNetwork *net);
uint64_t spingalett_parameter_count(const SpingalettNetwork *net);
SpingalettLossFunction spingalett_network_loss(const SpingalettNetwork *net);
uint64_t spingalett_optimizer_steps(const SpingalettNetwork *net);
bool spingalett_get_parameters(const SpingalettNetwork *net, uint32_t index, SpingalettParameterKind kind,
                               float *values, uint64_t count);
bool spingalett_set_parameters(SpingalettNetwork *net, uint32_t index, SpingalettParameterKind kind,
                               const float *values, uint64_t count);
float spingalett_activate(float x, SpingalettActivationFunction act_func);
float spingalett_derivative(float x, SpingalettActivationFunction act_func);
float *spingalett_forward_args(SpingalettForwardArgs args);
float spingalett_lr_cosine_decay(size_t epoch, size_t total_epochs, float initial_lr, void *params);
float spingalett_lr_linear_warmup(size_t epoch, size_t total_epochs, float initial_lr, void *params);
float spingalett_lr_step_decay(size_t epoch, size_t total_epochs, float initial_lr, void *params);
float spingalett_lr_warmup_cosine(size_t epoch, size_t total_epochs, float initial_lr, void *params);
float spingalett_lr_linear_decay(size_t epoch, size_t total_epochs, float initial_lr, void *params);
float spingalett_lr_warmup_linear(size_t epoch, size_t total_epochs, float initial_lr, void *params);
bool spingalett_predict_args(SpingalettPredictArgs args);
SpingalettEvalMetrics spingalett_evaluate_args(SpingalettEvaluateArgs args);
SpingalettTrainReport spingalett_train_args(SpingalettTrainArgs args);

/* Generation. */
typedef struct {
    SpingalettNetwork *net;
    const uint32_t *prompt;
    uint32_t prompt_length;
    uint32_t *tokens;
    uint32_t count;
    float temperature;
    uint32_t top_k;
    float top_p;
    uint64_t seed;
    const uint32_t *stop_tokens;
    uint32_t stop_count;
    uint64_t reserved[8];
} SpingalettGenerateArgs;

uint32_t spingalett_generate_args(SpingalettGenerateArgs args);

/* Data sets in the GPU's memory. */
SpingalettDeviceData *spingalett_device_data_new(const float *values, uint32_t count, uint32_t size);
void spingalett_device_data_free(SpingalettDeviceData *data);
uint32_t spingalett_device_data_count(const SpingalettDeviceData *data);
uint32_t spingalett_device_data_size(const SpingalettDeviceData *data);
bool spingalett_device_data_read(const SpingalettDeviceData *data, uint32_t first, uint32_t count,
                                 float *values);

/* The low-level training API. */
SpingalettTrainer *spingalett_trainer_new(SpingalettNetwork *net, uint32_t max_batch);
void spingalett_trainer_free(SpingalettTrainer *trainer);
const float *spingalett_trainer_forward(SpingalettTrainer *trainer, const float *inputs, uint32_t count);
float spingalett_trainer_backward(SpingalettTrainer *trainer, const float *targets);
bool spingalett_trainer_backward_output_grads(SpingalettTrainer *trainer, const float *output_grads);
bool spingalett_trainer_step(SpingalettTrainer *trainer, const SpingalettOptimizerArgs *optimizer);
void spingalett_trainer_zero_grad(SpingalettTrainer *trainer);
bool spingalett_trainer_set_label_smoothing(SpingalettTrainer *trainer, float label_smoothing);
float spingalett_train_on_batch(SpingalettTrainer *trainer, const float *inputs, const float *targets,
                                uint32_t count, const SpingalettOptimizerArgs *optimizer);

/* Files, memory images, importers, deployment models. */
bool spingalett_save_args(SpingalettSaveArgs args);
SpingalettNetwork *spingalett_load(const char *filename);
void *spingalett_save_to_memory(const SpingalettNetwork *net, SpingalettPrecisionMode precision,
                                bool save_optimizer, size_t *size);
SpingalettNetwork *spingalett_load_from_memory(const void *data, size_t size);
void spingalett_free(void *ptr);
bool spingalett_save_checkpoint(SpingalettNetwork *net, const char *path);
SpingalettNetwork *spingalett_load_checkpoint(const char *path);
SpingalettNetwork *spingalett_import_onnx(const char *path);
SpingalettNetwork *spingalett_import_onnx_from_memory(const void *data, size_t size);
bool spingalett_load_pytorch(SpingalettNetwork *net, const char *path, const char *const *modules,
                             uint32_t module_count);
bool spingalett_load_pytorch_from_memory(SpingalettNetwork *net, const void *data, size_t size,
                                         const char *const *modules, uint32_t module_count);

SpingalettModel *spingalett_model_from_network(const SpingalettNetwork *net, SpingalettPrecisionMode precision);
bool spingalett_export_c_header(const SpingalettNetwork *net, const char *path, const char *name,
                                SpingalettPrecisionMode precision);

/* Data sets in memory. */
bool spingalett_load_idx(const char *images_path, const char *labels_path,
                         uint32_t num_classes, SpingalettDataset *dataset);
bool spingalett_load_csv(const char *path, uint32_t target_columns, uint32_t num_classes,
                         SpingalettDataset *dataset);
bool spingalett_load_cifar(const char *const *paths, uint32_t path_count, uint32_t num_classes,
                           SpingalettDataset *dataset);
bool spingalett_dataset_set_class_names(SpingalettDataset *dataset, const char *const *names,
                                        uint32_t count);
void spingalett_dataset_shuffle(SpingalettDataset *dataset);
bool spingalett_dataset_split(SpingalettDataset *dataset, uint32_t count, SpingalettDataset *tail);
void spingalett_dataset_free(SpingalettDataset *dataset);

/* .slettd files, their readers and the readers of token files. */
typedef enum {
    SPINGALETT_DATASET_ENCODING_AUTO,
    SPINGALETT_DATASET_ENCODING_FLOAT32,
    SPINGALETT_DATASET_ENCODING_FP16,
    SPINGALETT_DATASET_ENCODING_BFLOAT16,
    SPINGALETT_DATASET_ENCODING_U8_UNIT,
    SPINGALETT_DATASET_ENCODING_U8_AFFINE,
    SPINGALETT_DATASET_ENCODING_CLASS,
    SPINGALETT_DATASET_ENCODING_COUNT
} SpingalettDatasetEncoding;

typedef struct {
    const char *name;
    uint32_t size;
    const float *targets;
    const char *const *class_names;
    SpingalettDatasetEncoding encoding;
    uint64_t reserved[8];
} SpingalettTargetSet;

typedef struct {
    SpingalettDatasetEncoding input_encoding;
    SpingalettDatasetEncoding target_encoding;
    bool no_compression;
    const char *target_name;
    const SpingalettTargetSet *extra_targets;
    uint32_t extra_target_count;
    uint64_t reserved[8];
} SpingalettDatasetSaveOptions;

typedef struct {
    uint32_t count, input_size, target_size;
    SpingalettDatasetEncoding input_encoding, target_encoding;
    uint32_t chunk_count;
    uint64_t file_size;
    uint32_t format_version;
    uint32_t height, width, channels;
    uint32_t target_set_count;
    uint32_t target_set;
    uint64_t reserved[8];
} SpingalettDatasetInfo;

typedef struct {
    bool shuffle;
    bool in_memory;
    bool no_prefetch;
    uint32_t target_set;
    uint64_t reserved[8];
} SpingalettDatasetReaderOptions;

typedef struct SpingalettDatasetReader SpingalettDatasetReader;

bool spingalett_save_dataset(const SpingalettDataset *dataset, const char *path,
                             const SpingalettDatasetSaveOptions *options);
bool spingalett_load_dataset(const char *path, SpingalettDataset *dataset);
bool spingalett_load_dataset_from_memory(const void *data, size_t size, SpingalettDataset *dataset);
bool spingalett_load_dataset_targets(const char *path, uint32_t target_set, SpingalettDataset *dataset);
bool spingalett_load_dataset_from_memory_targets(const void *data, size_t size, uint32_t target_set,
                                                 SpingalettDataset *dataset);
SpingalettDatasetReader *spingalett_dataset_open(const char *path, bool shuffle);
SpingalettDatasetReader *spingalett_dataset_open_ex(const char *path, const SpingalettDatasetReaderOptions *options);
SpingalettDatasetReader *spingalett_dataset_open_u8(const uint8_t *inputs, const float *targets,
                                                    uint32_t count, uint32_t input_size,
                                                    uint32_t target_size, bool shuffle);

typedef struct {
    uint32_t context;
    uint32_t token_bytes;
    uint32_t stride;
    uint64_t offset;
    bool shuffle;
    uint64_t reserved[8];
} SpingalettTokenReaderOptions;

SpingalettDatasetReader *spingalett_dataset_open_tokens(const char *path,
                                                        const SpingalettTokenReaderOptions *options);
const char *spingalett_dataset_target_set_name(const SpingalettDatasetReader *reader,
                                               uint32_t target_set);
const char *spingalett_dataset_class_name(const SpingalettDatasetReader *reader, uint32_t target_set,
                                          uint32_t index);
uint32_t spingalett_dataset_target_set_size(const SpingalettDatasetReader *reader, uint32_t target_set);
void spingalett_dataset_close(SpingalettDatasetReader *reader);
SpingalettDatasetInfo spingalett_dataset_info(const SpingalettDatasetReader *reader);
uint32_t spingalett_dataset_read(SpingalettDatasetReader *reader, float *inputs, float *targets,
                                 uint32_t max_samples);
typedef struct {
    uint64_t seed;
    uint64_t pass;
    uint64_t position;
    uint64_t reserved[8];
} SpingalettDatasetPosition;
bool spingalett_dataset_get_position(const SpingalettDatasetReader *reader, SpingalettDatasetPosition *position);
bool spingalett_dataset_set_position(SpingalettDatasetReader *reader, const SpingalettDatasetPosition *position);
uint32_t spingalett_dataset_generator(float *inputs, float *targets, uint32_t requested, void *reader);

void spingalett_print_network(const SpingalettNetwork *net);
void spingalett_network_free(SpingalettNetwork *net);

typedef struct SpingalettTokenizer SpingalettTokenizer;
typedef struct {
    uint32_t vocabulary;
    uint32_t added;
    uint32_t bos, eos;
    uint32_t pad, unk;
    uint64_t reserved[8];
} SpingalettTokenizerInfo;
SpingalettTokenizer *spingalett_tokenizer_load(const char *path);
SpingalettTokenizer *spingalett_tokenizer_from_memory(const void *data, size_t size);
void spingalett_tokenizer_free(SpingalettTokenizer *tokenizer);
SpingalettTokenizerInfo spingalett_tokenizer_info(const SpingalettTokenizer *tokenizer);
const char *spingalett_tokenizer_token(const SpingalettTokenizer *tokenizer, uint32_t id);
uint32_t spingalett_tokenizer_id(const SpingalettTokenizer *tokenizer, const char *token);
typedef struct {
    const SpingalettTokenizer *tokenizer;
    const char *text;
    size_t length;
    uint32_t *tokens;
    size_t capacity;
    bool special;
    bool add_special;
    uint64_t reserved[8];
} SpingalettTokenizeArgs;
int64_t spingalett_tokenize_args(SpingalettTokenizeArgs args);
typedef struct {
    const SpingalettTokenizer *tokenizer;
    const uint32_t *tokens;
    size_t count;
    char *text;
    size_t capacity;
    bool skip_special;
    uint64_t reserved[8];
} SpingalettDetokenizeArgs;
int64_t spingalett_detokenize_args(SpingalettDetokenizeArgs args);
typedef struct {
    const char *path;
    uint32_t context;
    uint64_t reserved[8];
} SpingalettPretrainedArgs;
SpingalettNetwork *spingalett_load_pretrained_args(SpingalettPretrainedArgs args);
