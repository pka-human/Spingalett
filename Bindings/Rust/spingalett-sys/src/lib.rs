// SPDX-License-Identifier: MIT
// Copyright (c) 2026 pka_human (pka_human@proton.me)

//! Raw declarations of libspingalett's C API (Include/Spingalett/Spingalett.h and the headers it includes):
//! the structures the calls take, laid out as in C (the crate `spingalett`'s tests compare every offset
//! with the C compiler's), the enumerators as constants, and the functions. Use the crate `spingalett`
//! for a safe interface.

#![allow(non_camel_case_types)]

use std::os::raw::{c_char, c_int, c_void};

/// Reserved words at the end of every argument structure (zero them).
pub const RESERVED: usize = 8;
/// Inputs a layer reads at most.
pub const MAX_INPUTS: usize = 16;
/// What spingalett_append_layer() returns on error.
pub const NO_LAYER: u32 = u32::MAX;

pub type ComputeMode = c_int;
pub const COMPUTE_SINGLE_THREADED: ComputeMode = 0;
pub const COMPUTE_OPENMP: ComputeMode = 1;
pub const COMPUTE_OPENBLAS: ComputeMode = 2;
pub const COMPUTE_CUDA: ComputeMode = 3;
pub const COMPUTE_VULKAN: ComputeMode = 4;

pub type ActivationFunction = c_int;
pub const ACT_NONE: ActivationFunction = 0;
pub const ACT_SIGMOID: ActivationFunction = 1;
pub const ACT_RELU: ActivationFunction = 2;
pub const ACT_TANH: ActivationFunction = 3;
pub const ACT_LEAKY_RELU: ActivationFunction = 4;
pub const ACT_FOO52: ActivationFunction = 5;
pub const ACT_SOFTMAX: ActivationFunction = 6;
pub const ACT_GELU: ActivationFunction = 7;
pub const ACT_GELU_TANH: ActivationFunction = 8;
pub const ACT_SILU: ActivationFunction = 9;

pub type LossFunction = c_int;
pub const LOSS_MSE: LossFunction = 0;
pub const LOSS_CROSS_ENTROPY: LossFunction = 1;
pub const LOSS_SPARSE_CROSS_ENTROPY: LossFunction = 2;

pub type LayerType = c_int;
pub const LAYER_DENSE: LayerType = 0;
pub const LAYER_CONV2D: LayerType = 1;
pub const LAYER_MAX_POOL2D: LayerType = 2;
pub const LAYER_AVG_POOL2D: LayerType = 3;
pub const LAYER_BATCH_NORM: LayerType = 4;
pub const LAYER_ADD: LayerType = 5;
pub const LAYER_CONCAT: LayerType = 6;
pub const LAYER_GLOBAL_AVG_POOL: LayerType = 7;
pub const LAYER_CONV_TRANSPOSE2D: LayerType = 8;
pub const LAYER_UPSAMPLE: LayerType = 9;
pub const LAYER_LAYER_NORM: LayerType = 10;
pub const LAYER_EMBEDDING: LayerType = 11;
pub const LAYER_ATTENTION: LayerType = 12;
pub const LAYER_RMS_NORM: LayerType = 13;
pub const LAYER_MULTIPLY: LayerType = 14;

pub type PrecisionMode = c_int;
pub const PRECISION_FLOAT32: PrecisionMode = 0;
pub const PRECISION_FP16: PrecisionMode = 1;
pub const PRECISION_BFLOAT16: PrecisionMode = 2;
pub const PRECISION_INT8: PrecisionMode = 3;
pub const PRECISION_INT4: PrecisionMode = 4;
pub const PRECISION_INT2: PrecisionMode = 5;

pub type OptimizerType = c_int;
pub const OPTIMIZER_SGD: OptimizerType = 0;
pub const OPTIMIZER_MOMENTUM: OptimizerType = 1;
pub const OPTIMIZER_RMSPROP: OptimizerType = 2;
pub const OPTIMIZER_ADAM: OptimizerType = 3;
pub const OPTIMIZER_ADAMW: OptimizerType = 4;

pub type TrainingStrategy = c_int;
pub const STRATEGY_SAMPLE: TrainingStrategy = 0;
pub const STRATEGY_FULL_BATCH: TrainingStrategy = 1;
pub const STRATEGY_SMALL_BATCH: TrainingStrategy = 2;

pub type TrainingMode = c_int;
pub const MODE_ARRAY: TrainingMode = 0;
pub const MODE_GENERATOR_FUNCTION: TrainingMode = 1;

pub type WeightInitialization = c_int;
pub const INIT_RANDOM: WeightInitialization = 0;
pub const INIT_XAVIER: WeightInitialization = 1;
pub const INIT_HE: WeightInitialization = 2;
pub const INIT_NONE: WeightInitialization = 3;
pub const INIT_LECUN: WeightInitialization = 4;

pub type UpsampleMode = c_int;
pub const UPSAMPLE_NEAREST: UpsampleMode = 0;
pub const UPSAMPLE_BILINEAR: UpsampleMode = 1;

pub type ParameterKind = c_int;
pub const PARAM_WEIGHTS: ParameterKind = 0;
pub const PARAM_BIASES: ParameterKind = 1;

pub type TrainStatus = c_int;
pub const TRAIN_FAILED: TrainStatus = 0;
pub const TRAIN_COMPLETED: TrainStatus = 1;
pub const TRAIN_EARLY_STOPPED: TrainStatus = 2;
pub const TRAIN_INTERRUPTED: TrainStatus = 3;
pub const TRAIN_DIVERGED: TrainStatus = 4;
pub const TRAIN_NO_DATA: TrainStatus = 5;

/// A network (opaque).
#[repr(C)]
pub struct Network {
    _private: [u8; 0],
}
/// Rows in the GPU's memory (opaque).
#[repr(C)]
pub struct DeviceData {
    _private: [u8; 0],
}
/// A reader of a data set file (opaque).
#[repr(C)]
pub struct DatasetReader {
    _private: [u8; 0],
}

pub type DataGeneratorFn = Option<unsafe extern "C" fn(*mut f32, *mut f32, u32, *mut c_void) -> u32>;
pub type TrainCallback = Option<unsafe extern "C" fn(*mut Network, *const TrainProgress, *mut c_void) -> bool>;
pub type LRSchedulerFn = Option<unsafe extern "C" fn(usize, usize, f32, *mut c_void) -> f32>;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct EvalMetrics {
    pub loss: f32,
    pub accuracy: f32,
    pub reserved: [u64; RESERVED],
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct NetworkArgs {
    pub loss_func: LossFunction,
    pub reserved: [u64; RESERVED],
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct LayerArgs {
    pub net: *mut Network,
    pub neurons_amount: u32,
    pub act_func: ActivationFunction,
    pub weight_initialization: WeightInitialization,
    pub dropout_rate: f32,
    pub type_: LayerType,
    pub height: u32,
    pub width: u32,
    pub channels: u32,
    pub filters: u32,
    pub kernel: u32,
    pub stride: u32,
    pub padding: u32,
    pub kernel_h: u32,
    pub kernel_w: u32,
    pub stride_h: u32,
    pub stride_w: u32,
    pub padding_h: u32,
    pub padding_w: u32,
    pub groups: u32,
    pub epsilon: f32,
    pub momentum: f32,
    pub inputs: [u32; MAX_INPUTS],
    pub input_count: u32,
    pub upsample: UpsampleMode,
    pub output_padding: u32,
    pub output_padding_h: u32,
    pub output_padding_w: u32,
    pub vocabulary: u32,
    pub heads: u32,
    pub kv_heads: u32,
    pub rope_theta: f32,
    pub causal: bool,
    pub positions: bool,
    pub reserved: [u64; RESERVED - 3],
}

impl Default for LayerArgs {
    fn default() -> Self {
        // (every field zero, as C's designated initializers leave them)
        unsafe { std::mem::zeroed() }
    }
}

/// A layer of a network as spingalett_network_layer() describes it.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct NetworkLayer {
    pub type_: LayerType,
    pub height: u32,
    pub width: u32,
    pub channels: u32,
    pub outputs: u32,
    pub activation: ActivationFunction,
    pub dropout_rate: f32,
    pub kernel_h: u32,
    pub kernel_w: u32,
    pub stride_h: u32,
    pub stride_w: u32,
    pub padding_h: u32,
    pub padding_w: u32,
    pub weight_count: u64,
    pub bias_count: u64,
    pub groups: u32,
    pub epsilon: f32,
    pub momentum: f32,
    pub input_count: u32,
    pub inputs: [u32; MAX_INPUTS],
    pub upsample: UpsampleMode,
    pub vocabulary: u32,
    pub heads: u32,
    pub kv_heads: u32,
    pub rope_theta: f32,
    pub causal: bool,
    pub positions: bool,
    pub reserved: [u64; RESERVED - 2],
}

impl Default for NetworkLayer {
    fn default() -> Self {
        unsafe { std::mem::zeroed() }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct TrainProgress {
    pub epoch: usize,
    pub epochs: usize,
    pub train_loss: f32,
    pub learning_rate: f32,
    pub has_validation: bool,
    pub validation: EvalMetrics,
    pub monitor: c_int,
    pub best_epoch: usize,
    pub best_value: f32,
    pub improved: bool,
    pub reserved: [u64; RESERVED],
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct TrainArgs {
    pub net: *mut Network,
    pub training_mode: TrainingMode,
    pub training_strategy: TrainingStrategy,
    pub optimizer_type: OptimizerType,
    pub inputs: *const f32,
    pub targets: *const f32,
    pub device_inputs: *const DeviceData,
    pub device_targets: *const DeviceData,
    pub generator: DataGeneratorFn,
    pub generator_data: *mut c_void,
    pub sample_count: u32,
    pub batch_size: u32,
    pub do_not_shuffle: bool,
    pub epochs: usize,
    pub learning_rate: f32,
    pub weight_decay: f32,
    pub momentum: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
    pub max_grad_norm: f32,
    pub reset_optimizer: bool,
    pub nan_check_interval: usize,
    pub report_interval: usize,
    pub autosave_mode: c_int,
    pub autosave_interval: usize,
    pub autosave_path: *const c_char,
    pub autosave_do_not_save_optimizer: bool,
    pub autosave_precision: PrecisionMode,
    pub callback: TrainCallback,
    pub callback_interval: usize,
    pub callback_data: *mut c_void,
    pub lr_scheduler: LRSchedulerFn,
    pub lr_scheduler_data: *mut c_void,
    pub val_inputs: *const f32,
    pub val_targets: *const f32,
    pub device_val_inputs: *const DeviceData,
    pub device_val_targets: *const DeviceData,
    pub val_count: u32,
    pub monitor: c_int,
    pub early_stopping_patience: usize,
    pub early_stopping_min_delta: f32,
    pub restore_best_weights: bool,
    pub blas_num_threads: c_int,
    pub augment_shift: u32,
    pub augment_flip: bool,
    pub label_smoothing: f32,
    pub lr_plateau_factor: f32,
    pub lr_plateau_patience: usize,
    pub lr_plateau_min_lr: f32,
    pub reserved: [u64; RESERVED],
}

impl Default for TrainArgs {
    fn default() -> Self {
        unsafe { std::mem::zeroed() }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct TrainReport {
    pub status: TrainStatus,
    pub epochs_run: usize,
    pub train_loss: f32,
    pub has_validation: bool,
    pub validation: EvalMetrics,
    pub monitor: c_int,
    pub best_epoch: usize,
    pub best_value: f32,
    pub restored_best: bool,
    pub reserved: [u64; RESERVED],
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct PredictArgs {
    pub net: *mut Network,
    pub inputs: *const f32,
    pub device_inputs: *const DeviceData,
    pub sample_count: u32,
    pub outputs: *mut f32,
    pub reserved: [u64; RESERVED],
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct EvaluateArgs {
    pub net: *mut Network,
    pub inputs: *const f32,
    pub targets: *const f32,
    pub device_inputs: *const DeviceData,
    pub device_targets: *const DeviceData,
    pub sample_count: u32,
    pub reserved: [u64; RESERVED],
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SaveArgs {
    pub net: *mut Network,
    pub filename: *const c_char,
    pub do_not_save_optimizer: bool,
    pub precision: PrecisionMode,
    pub reserved: [u64; RESERVED],
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct GenerateArgs {
    pub net: *mut Network,
    pub prompt: *const u32,
    pub prompt_length: u32,
    pub tokens: *mut u32,
    pub count: u32,
    pub temperature: f32,
    pub top_k: u32,
    pub top_p: f32,
    pub seed: u64,
    pub stop_tokens: *const u32,
    pub stop_count: u32,
    pub reserved: [u64; RESERVED],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct TokenReaderOptions {
    pub context: u32,
    pub token_bytes: u32,
    pub stride: u32,
    pub offset: u64,
    pub shuffle: bool,
    pub reserved: [u64; RESERVED],
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct DatasetInfo {
    pub count: u32,
    pub input_size: u32,
    pub target_size: u32,
    pub input_encoding: c_int,
    pub target_encoding: c_int,
    pub chunk_count: u32,
    pub file_size: u64,
    pub format_version: u32,
    pub height: u32,
    pub width: u32,
    pub channels: u32,
    pub target_set_count: u32,
    pub target_set: u32,
    pub reserved: [u64; RESERVED],
}

/// A deployment model: the fields up to `image_size` are public, the rest the library's.
#[repr(C)]
#[derive(Debug)]
pub struct Model {
    pub input_size: u32,
    pub output_size: u32,
    pub layer_count: u32,
    pub loss: LossFunction,
    pub workspace_size: usize,
    pub image: *const c_void,
    pub image_size: usize,
    pub max_width_: u32,
    pub max_int_inputs_: u32,
    pub conv_scratch_: usize,
    pub owner_: *mut c_void,
    pub activations_: usize,
    pub reserved: [u64; RESERVED],
}

extern "C" {
    pub fn spingalett_version() -> *const c_char;
    pub fn spingalett_last_error_code() -> c_int;
    pub fn spingalett_last_error_message() -> *const c_char;
    pub fn spingalett_clear_error();
    pub fn spingalett_get_compute_mode() -> ComputeMode;
    pub fn spingalett_set_compute_mode(mode: ComputeMode) -> bool;
    pub fn spingalett_set_num_threads(n: u32);
    pub fn spingalett_set_verbose(enabled: bool);
    pub fn spingalett_gpu_device() -> *const c_char;
    pub fn spingalett_cuda_device() -> *const c_char;
    pub fn spingalett_set_gpu_precision(precision: PrecisionMode) -> bool;
    pub fn spingalett_seed(seed: u64);

    pub fn spingalett_network_new_args(args: NetworkArgs) -> *mut Network;
    pub fn spingalett_network_free(net: *mut Network);
    pub fn spingalett_append_layer(args: LayerArgs) -> u32;
    pub fn spingalett_layer_count(net: *const Network) -> u32;
    pub fn spingalett_network_layer(net: *const Network, index: u32, layer: *mut NetworkLayer) -> bool;
    pub fn spingalett_input_size(net: *const Network) -> u32;
    pub fn spingalett_output_size(net: *const Network) -> u32;
    pub fn spingalett_target_size(net: *const Network) -> u32;
    pub fn spingalett_parameter_count(net: *const Network) -> u64;
    pub fn spingalett_get_parameters(net: *const Network, index: u32, kind: ParameterKind, values: *mut f32,
                                     count: u64) -> bool;
    pub fn spingalett_set_parameters(net: *mut Network, index: u32, kind: ParameterKind, values: *const f32,
                                     count: u64) -> bool;
    pub fn spingalett_train_args(args: TrainArgs) -> TrainReport;
    pub fn spingalett_predict_args(args: PredictArgs) -> bool;
    pub fn spingalett_evaluate_args(args: EvaluateArgs) -> EvalMetrics;
    pub fn spingalett_generate_args(args: GenerateArgs) -> u32;
    pub fn spingalett_save_args(args: SaveArgs) -> bool;
    pub fn spingalett_load(filename: *const c_char) -> *mut Network;

    pub fn spingalett_dataset_open_tokens(path: *const c_char, options: *const TokenReaderOptions) -> *mut DatasetReader;
    pub fn spingalett_dataset_info(reader: *const DatasetReader) -> DatasetInfo;
    pub fn spingalett_dataset_close(reader: *mut DatasetReader);
    pub fn spingalett_dataset_generator(inputs: *mut f32, targets: *mut f32, requested: u32, reader: *mut c_void) -> u32;

    pub fn spingalett_model_from_network(net: *const Network, precision: PrecisionMode) -> *mut Model;
    pub fn spingalett_model_load(path: *const c_char) -> *mut Model;
    pub fn spingalett_model_free(model: *mut Model);
    pub fn spingalett_model_predict(model: *const Model, inputs: *const f32, count: u32, outputs: *mut f32) -> bool;
    pub fn spingalett_model_evaluate(model: *const Model, inputs: *const f32, targets: *const f32, count: u32)
        -> EvalMetrics;
}
