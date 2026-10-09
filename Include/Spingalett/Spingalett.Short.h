/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/*
 * The names of Spingalett 0.x, without the library's prefix: NeuralNetwork for SpingalettNetwork,
 * ACT_RELU for SPINGALETT_ACT_RELU, layer() for spingalett_layer(), train() for spingalett_train() and
 * so on, so that programs written for them keep compiling: include this header in place of
 * Spingalett.h, or define SPINGALETT_SHORT_NAMES before including that. The engine's names
 * (ActivationFunction, ACT_*, LOSS_*, LAYER_*, UPSAMPLE_*, PRECISION_*) come with
 * Spingalett.Inference.h when SPINGALETT_SHORT_NAMES is defined. They are macros and typedefs in the
 * global namespace (LOG_DEBUG is also <syslog.h>'s, and a function of the program named train() or
 * layer() would be replaced), which is why Spingalett.h leaves them out unless asked.
 */
#pragma once

#if !defined(SPINGALETT_SHORT_NAMES)
#define SPINGALETT_SHORT_NAMES
#endif
#include "Spingalett.h"
#include "Spingalett.Inference.h"   /* its short names, when it was included before this header */

/* types */
typedef SpingalettNetwork NeuralNetwork;
typedef SpingalettLogLevel LogLevel;
typedef SpingalettLogCallback LogCallback;
typedef SpingalettWeightInitialization WeightInitialization;
typedef SpingalettTrainingMode TrainingMode;
typedef SpingalettTrainingStrategy TrainingStrategy;
typedef SpingalettOptimizerType OptimizerType;
typedef SpingalettComputeMode ComputeMode;
typedef SpingalettAutoSaveMode AutoSaveMode;
typedef SpingalettParameterKind ParameterKind;
typedef SpingalettMonitorMetric MonitorMetric;
typedef SpingalettEvalMetrics EvalMetrics;
typedef SpingalettTrainProgress TrainProgress;
typedef SpingalettTrainCallback TrainCallback;
typedef SpingalettTrainStatus TrainStatus;
typedef SpingalettTrainReport TrainReport;
typedef SpingalettDataGeneratorFn DataGeneratorFn;
typedef SpingalettLRSchedulerFn LRSchedulerFn;
typedef SpingalettLRScheduleParams LRScheduleParams;
typedef SpingalettNetworkArgs NeuralNetworkArgs;
typedef SpingalettLayerArgs LayerArgs;
typedef SpingalettForwardArgs ForwardArgs;
typedef SpingalettTrainArgs TrainArgs;
typedef SpingalettPredictArgs PredictArgs;
typedef SpingalettEvaluateArgs EvaluateArgs;
typedef SpingalettSaveArgs SaveArgs;
typedef SpingalettOptimizerArgs OptimizerArgs;
typedef SpingalettDatasetEncoding DatasetEncoding;
typedef SpingalettDatasetSaveOptions DatasetSaveOptions;
typedef SpingalettDatasetReaderOptions DatasetReaderOptions;

/* constants */
#define LOG_DEBUG SPINGALETT_LOG_DEBUG
#define LOG_INFO SPINGALETT_LOG_INFO
#define LOG_WARNING SPINGALETT_LOG_WARNING
#define LOG_ERROR SPINGALETT_LOG_ERROR
#define LOG_NONE SPINGALETT_LOG_NONE
#define WEIGHT_INITIALIZATION_RANDOM SPINGALETT_INIT_RANDOM
#define WEIGHT_INITIALIZATION_XAVIER SPINGALETT_INIT_XAVIER
#define WEIGHT_INITIALIZATION_HE SPINGALETT_INIT_HE
#define WEIGHT_INITIALIZATION_NONE SPINGALETT_INIT_NONE
#define WEIGHT_INITIALIZATION_LECUN SPINGALETT_INIT_LECUN
#define WEIGHT_INITIALIZATION_COUNT SPINGALETT_INIT_COUNT
#define MODE_ARRAY SPINGALETT_MODE_ARRAY
#define MODE_GENERATOR_FUNCTION SPINGALETT_MODE_GENERATOR_FUNCTION
#define MODE_COUNT SPINGALETT_MODE_COUNT
#define STRATEGY_SAMPLE SPINGALETT_STRATEGY_SAMPLE
#define STRATEGY_FULL_BATCH SPINGALETT_STRATEGY_FULL_BATCH
#define STRATEGY_SMALL_BATCH SPINGALETT_STRATEGY_SMALL_BATCH
#define STRATEGY_COUNT SPINGALETT_STRATEGY_COUNT
#define OPTIMIZER_SGD SPINGALETT_OPTIMIZER_SGD
#define OPTIMIZER_MOMENTUM SPINGALETT_OPTIMIZER_MOMENTUM
#define OPTIMIZER_RMSPROP SPINGALETT_OPTIMIZER_RMSPROP
#define OPTIMIZER_ADAM SPINGALETT_OPTIMIZER_ADAM
#define OPTIMIZER_ADAMW SPINGALETT_OPTIMIZER_ADAMW
#define OPTIMIZER_COUNT SPINGALETT_OPTIMIZER_COUNT
#define COMPUTE_SINGLE_THREADED SPINGALETT_COMPUTE_SINGLE_THREADED
#define COMPUTE_OPENMP SPINGALETT_COMPUTE_OPENMP
#define COMPUTE_OPENBLAS SPINGALETT_COMPUTE_OPENBLAS
#define COMPUTE_CUDA SPINGALETT_COMPUTE_CUDA
#define COMPUTE_VULKAN SPINGALETT_COMPUTE_VULKAN
#define COMPUTE_COUNT SPINGALETT_COMPUTE_COUNT
#define AUTOSAVE_OFF SPINGALETT_AUTOSAVE_OFF
#define AUTOSAVE_OVERWRITE SPINGALETT_AUTOSAVE_OVERWRITE
#define AUTOSAVE_NEW_FILES SPINGALETT_AUTOSAVE_NEW_FILES
#define PARAM_WEIGHTS SPINGALETT_PARAM_WEIGHTS
#define PARAM_BIASES SPINGALETT_PARAM_BIASES
#define PARAM_WEIGHT_GRADIENTS SPINGALETT_PARAM_WEIGHT_GRADIENTS
#define PARAM_BIAS_GRADIENTS SPINGALETT_PARAM_BIAS_GRADIENTS
#define PARAM_RUNNING_MEAN SPINGALETT_PARAM_RUNNING_MEAN
#define PARAM_RUNNING_VARIANCE SPINGALETT_PARAM_RUNNING_VARIANCE
#define PARAM_KIND_COUNT SPINGALETT_PARAM_KIND_COUNT
#define MONITOR_AUTO SPINGALETT_MONITOR_AUTO
#define MONITOR_TRAIN_LOSS SPINGALETT_MONITOR_TRAIN_LOSS
#define MONITOR_VAL_LOSS SPINGALETT_MONITOR_VAL_LOSS
#define MONITOR_VAL_ACCURACY SPINGALETT_MONITOR_VAL_ACCURACY
#define MONITOR_COUNT SPINGALETT_MONITOR_COUNT
#define TRAIN_FAILED SPINGALETT_TRAIN_FAILED
#define TRAIN_COMPLETED SPINGALETT_TRAIN_COMPLETED
#define TRAIN_EARLY_STOPPED SPINGALETT_TRAIN_EARLY_STOPPED
#define TRAIN_INTERRUPTED SPINGALETT_TRAIN_INTERRUPTED
#define TRAIN_DIVERGED SPINGALETT_TRAIN_DIVERGED
#define TRAIN_NO_DATA SPINGALETT_TRAIN_NO_DATA
#define DATASET_ENCODING_AUTO SPINGALETT_DATASET_ENCODING_AUTO
#define DATASET_ENCODING_FLOAT32 SPINGALETT_DATASET_ENCODING_FLOAT32
#define DATASET_ENCODING_FP16 SPINGALETT_DATASET_ENCODING_FP16
#define DATASET_ENCODING_BFLOAT16 SPINGALETT_DATASET_ENCODING_BFLOAT16
#define DATASET_ENCODING_U8_UNIT SPINGALETT_DATASET_ENCODING_U8_UNIT
#define DATASET_ENCODING_U8_AFFINE SPINGALETT_DATASET_ENCODING_U8_AFFINE
#define DATASET_ENCODING_CLASS SPINGALETT_DATASET_ENCODING_CLASS
#define DATASET_ENCODING_COUNT SPINGALETT_DATASET_ENCODING_COUNT

/* builders (designated initializers, as with the prefixed ones) */
#define new_spingalett(...) spingalett_network_new(__VA_ARGS__)
#define layer(...) spingalett_layer(__VA_ARGS__)
#define conv2d(...) spingalett_conv2d(__VA_ARGS__)
#define max_pool2d(...) spingalett_max_pool2d(__VA_ARGS__)
#define avg_pool2d(...) spingalett_avg_pool2d(__VA_ARGS__)
#define batch_norm(...) spingalett_batch_norm(__VA_ARGS__)
#define add_layers(...) spingalett_add_layers(__VA_ARGS__)
#define concat_layers(...) spingalett_concat_layers(__VA_ARGS__)
#define global_avg_pool2d(...) spingalett_global_avg_pool2d(__VA_ARGS__)
#define conv_transpose2d(...) spingalett_conv_transpose2d(__VA_ARGS__)
#define upsample2d(...) spingalett_upsample2d(__VA_ARGS__)
#define layer_norm(...) spingalett_layer_norm(__VA_ARGS__)
#define forward(...) spingalett_forward(__VA_ARGS__)
#define predict(...) spingalett_predict(__VA_ARGS__)
#define evaluate(...) spingalett_evaluate(__VA_ARGS__)
#define train(...) spingalett_train(__VA_ARGS__)
#define save_spingalett(...) spingalett_save(__VA_ARGS__)

/* functions (variadic, so that compound literals pass through whole) */
#define new_spingalett_struct_arguments(...) spingalett_network_new_args(__VA_ARGS__)
#define layer_struct_arguments(...) spingalett_append_layer(__VA_ARGS__)
#define forward_struct_arguments(...) spingalett_forward_args(__VA_ARGS__)
#define predict_struct_arguments(...) spingalett_predict_args(__VA_ARGS__)
#define evaluate_struct_arguments(...) spingalett_evaluate_args(__VA_ARGS__)
#define train_struct_arguments(...) spingalett_train_args(__VA_ARGS__)
#define save_spingalett_struct_arguments(...) spingalett_save_args(__VA_ARGS__)
#define activate(...) spingalett_activate(__VA_ARGS__)
#define derivative(...) spingalett_derivative(__VA_ARGS__)
#define load_spingalett(...) spingalett_load(__VA_ARGS__)
#define load_spingalett_from_memory(...) spingalett_load_from_memory(__VA_ARGS__)
#define print_parameters(...) spingalett_print_network(__VA_ARGS__)
#define free_network(...) spingalett_network_free(__VA_ARGS__)
