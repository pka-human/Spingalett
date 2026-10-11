// SPDX-License-Identifier: MIT
// Copyright (c) 2026 pka_human (pka_human@proton.me)

//! Spingalett for Rust: neural networks (dense, convolutional and transformer layers, as chains or graphs)
//! trained and run by libspingalett on the CPU or the GPU (CUDA, Vulkan), deployment models from FP32 to
//! INT2, and text generation.
//!
//! ```no_run
//! use spingalett::{Activation, Layer, Loss, Network, TrainOptions};
//! let mut net = Network::new(Loss::Mse)?;
//! net.add(Layer::input(1, 1, 2))?;
//! net.add(Layer::dense(8).activation(Activation::Tanh))?;
//! net.add(Layer::dense(1).activation(Activation::Sigmoid))?;
//! let x = [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0];
//! let y = [0.0, 1.0, 1.0, 0.0];
//! net.train(&x, &y, &TrainOptions { epochs: 2000, learning_rate: 0.5, ..Default::default() })?;
//! println!("{:?}", net.predict(&x)?);
//! # Ok::<(), spingalett::Error>(())
//! ```

use spingalett_sys as sys;
use std::ffi::{CStr, CString};
use std::fmt;
use std::os::raw::c_void;
use std::path::Path;
use std::ptr::{self, NonNull};

pub use spingalett_sys;

/// An error the library reported: its code (SPINGALETT_ERR_*) and message.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Error {
    pub code: i32,
    pub message: String,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} (code {})", self.message, self.code)
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

/// The library's last error, or one with `what` when it set none.
fn last_error(what: &str) -> Error {
    unsafe {
        let code = sys::spingalett_last_error_code();
        let message = sys::spingalett_last_error_message();
        let message = if message.is_null() { String::new() } else { CStr::from_ptr(message).to_string_lossy().into_owned() };
        Error { code: if code != 0 { code } else { -1 }, message: if message.is_empty() { what.to_owned() } else { message } }
    }
}

/// Runs a call after clearing the error, and turns an error it set into Err.
fn checked<T>(what: &str, call: impl FnOnce() -> T) -> Result<T> {
    unsafe { sys::spingalett_clear_error() };
    let result = call();
    if unsafe { sys::spingalett_last_error_code() } != 0 {
        return Err(last_error(what));
    }
    Ok(result)
}

fn c_path(path: &Path) -> Result<CString> {
    CString::new(path.to_string_lossy().as_bytes())
        .map_err(|_| Error { code: -1, message: format!("{} holds a NUL byte", path.display()) })
}

fn invalid(message: String) -> Error {
    Error { code: -1, message }
}

macro_rules! c_enum {
    ($(#[$meta:meta])* $name:ident { $($(#[$vmeta:meta])* $variant:ident = $value:expr),* $(,)? }) => {
        $(#[$meta])*
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        #[repr(i32)]
        pub enum $name { $($(#[$vmeta])* $variant = $value),* }
    };
}

c_enum!(
    /// Activation of a layer's outputs.
    Activation { None = 0, Sigmoid = 1, Relu = 2, Tanh = 3, LeakyRelu = 4, Foo52 = 5, Softmax = 6,
                 /// x Phi(x), with erf
                 Gelu = 7,
                 /// GPT-2's tanh approximation of GELU
                 GeluTanh = 8,
                 /// x sigmoid(x)
                 Silu = 9 }
);
c_enum!(
    /// Loss a network trains with.
    Loss { Mse = 0, CrossEntropy = 1,
           /// one class index a cell (a language model's next tokens)
           SparseCrossEntropy = 2 }
);
c_enum!(
    /// Where the library computes.
    ComputeMode { SingleThreaded = 0, OpenMp = 1, OpenBlas = 2, Cuda = 3, Vulkan = 4 }
);
c_enum!(
    /// Precision of deployment models, and of the GPU's products (Float32 or BFloat16).
    Precision { Float32 = 0, Fp16 = 1, BFloat16 = 2, Int8 = 3, Int4 = 4, Int2 = 5 }
);
c_enum!(
    /// Optimizer of training.
    Optimizer { Sgd = 0, Momentum = 1, RmsProp = 2, Adam = 3, AdamW = 4 }
);
c_enum!(
    /// How training goes through the samples.
    Strategy { Sample = 0, FullBatch = 1, SmallBatch = 2 }
);
c_enum!(
    /// Initialization of a layer's weights.
    Init { Random = 0, Xavier = 1, He = 2, Zeros = 3, LeCun = 4 }
);

/// The library's version ("1.2.0").
pub fn version() -> String {
    unsafe { CStr::from_ptr(sys::spingalett_version()).to_string_lossy().into_owned() }
}

/// Where training and inference run; false when the mode is not available (no such GPU).
pub fn set_compute_mode(mode: ComputeMode) -> bool {
    unsafe { sys::spingalett_set_compute_mode(mode as i32) }
}

/// Precision of the GPU's matrix products (Float32 or BFloat16); false when the GPU has no such products.
pub fn set_gpu_precision(precision: Precision) -> bool {
    unsafe { sys::spingalett_set_gpu_precision(precision as i32) }
}

/// Seeds the calling thread's generator (initialization, shuffling, dropout, draws).
pub fn seed(seed: u64) {
    unsafe { sys::spingalett_seed(seed) }
}

/// The library's informational messages on stdout, or not.
pub fn set_verbose(enabled: bool) {
    unsafe { sys::spingalett_set_verbose(enabled) }
}

/// Threads of the CPU's parallel loops (0: all).
pub fn set_num_threads(threads: u32) {
    unsafe { sys::spingalett_set_num_threads(threads) }
}

fn device(name: *const std::os::raw::c_char) -> Option<String> {
    if name.is_null() { None } else { Some(unsafe { CStr::from_ptr(name) }.to_string_lossy().into_owned()) }
}

/// The name of the GPU that ComputeMode::Cuda uses, if there is one.
pub fn cuda_device() -> Option<String> {
    device(unsafe { sys::spingalett_cuda_device() })
}

/// The name of the GPU that ComputeMode::Vulkan uses, if there is one.
pub fn gpu_device() -> Option<String> {
    device(unsafe { sys::spingalett_gpu_device() })
}

/// A layer to add to a network: a kind with its sizes, then options.
///
/// ```
/// use spingalett::{Activation, Layer};
/// let conv = Layer::conv2d(16, 3).padding(1).activation(Activation::Relu);
/// let attention = Layer::attention(8).kv_heads(2).causal(true).rope_theta(10000.0);
/// ```
#[derive(Clone, Copy, Debug)]
pub struct Layer {
    args: sys::LayerArgs,
}

impl Layer {
    fn of(kind: sys::LayerType) -> Self {
        let mut args = sys::LayerArgs::default();
        args.type_ = kind;
        Layer { args }
    }

    /// The input layer: height x width x channels values a sample (a window of tokens: 1 x 1 x tokens).
    pub fn input(height: u32, width: u32, channels: u32) -> Self {
        let mut l = Layer::of(sys::LAYER_DENSE);
        l.args.height = height;
        l.args.width = width;
        l.args.channels = channels;
        l
    }
    /// A fully connected layer of `neurons` outputs (LeCun initialization, no activation).
    pub fn dense(neurons: u32) -> Self {
        let mut l = Layer::of(sys::LAYER_DENSE);
        l.args.neurons_amount = neurons;
        l.args.weight_initialization = sys::INIT_LECUN;
        l
    }
    /// A 2D convolution: `filters` output channels, `kernel` x `kernel` windows (He initialization, ReLU).
    pub fn conv2d(filters: u32, kernel: u32) -> Self {
        let mut l = Layer::of(sys::LAYER_CONV2D);
        l.args.filters = filters;
        l.args.kernel = kernel;
        l.args.act_func = sys::ACT_RELU;
        l.args.weight_initialization = sys::INIT_HE;
        l
    }
    /// A transposed 2D convolution (upsampling by its stride).
    pub fn conv_transpose2d(filters: u32, kernel: u32) -> Self {
        let mut l = Layer::conv2d(filters, kernel);
        l.args.type_ = sys::LAYER_CONV_TRANSPOSE2D;
        l
    }
    /// The maximum over each `kernel` x `kernel` window.
    pub fn max_pool2d(kernel: u32) -> Self {
        let mut l = Layer::of(sys::LAYER_MAX_POOL2D);
        l.args.kernel = kernel;
        l
    }
    /// The mean over each `kernel` x `kernel` window.
    pub fn avg_pool2d(kernel: u32) -> Self {
        let mut l = Layer::of(sys::LAYER_AVG_POOL2D);
        l.args.kernel = kernel;
        l
    }
    /// The mean of each channel over all cells.
    pub fn global_avg_pool() -> Self {
        Layer::of(sys::LAYER_GLOBAL_AVG_POOL)
    }
    /// Batch normalization of each channel.
    pub fn batch_norm() -> Self {
        Layer::of(sys::LAYER_BATCH_NORM)
    }
    /// Layer normalization of each cell's channels.
    pub fn layer_norm() -> Self {
        Layer::of(sys::LAYER_LAYER_NORM)
    }
    /// RMS normalization of each cell's channels (no biases).
    pub fn rms_norm() -> Self {
        Layer::of(sys::LAYER_RMS_NORM)
    }
    /// The sum of the layers `inputs` names (one shape).
    pub fn add_layers(inputs: &[u32]) -> Self {
        Layer::of(sys::LAYER_ADD).inputs(inputs)
    }
    /// The layers `inputs` names side by side along the channels.
    pub fn concat(inputs: &[u32]) -> Self {
        Layer::of(sys::LAYER_CONCAT).inputs(inputs)
    }
    /// The product of the layers `inputs` names, element by element (SwiGLU's gate).
    pub fn multiply(inputs: &[u32]) -> Self {
        Layer::of(sys::LAYER_MULTIPLY).inputs(inputs)
    }
    /// Each cell's map of `factor` x `factor` cells (nearest neighbour; bilinear with `bilinear`).
    pub fn upsample(factor: u32) -> Self {
        let mut l = Layer::of(sys::LAYER_UPSAMPLE);
        l.args.stride = factor;
        l
    }
    /// A linear map of each cell's channels to `neurons` (a 1 x 1 convolution: a transformer's
    /// projections; LeCun initialization, no activation).
    pub fn linear(neurons: u32) -> Self {
        let mut l = Layer::of(sys::LAYER_CONV2D);
        l.args.filters = neurons;
        l.args.kernel = 1;
        l.args.weight_initialization = sys::INIT_LECUN;
        l
    }
    /// The vectors of `width` values of a vocabulary of `vocabulary` tokens, a token a cell.
    pub fn embedding(vocabulary: u32, width: u32) -> Self {
        let mut l = Layer::of(sys::LAYER_EMBEDDING);
        l.args.vocabulary = vocabulary;
        l.args.neurons_amount = width;
        l.args.weight_initialization = sys::INIT_LECUN;
        l
    }
    /// Attention of `heads` query heads over packed queries, keys and values (a linear layer before it
    /// gives (heads + 2 kv_heads) x head size channels a cell).
    pub fn attention(heads: u32) -> Self {
        let mut l = Layer::of(sys::LAYER_ATTENTION);
        l.args.heads = heads;
        l
    }

    pub fn activation(mut self, activation: Activation) -> Self {
        self.args.act_func = activation as i32;
        self
    }
    pub fn init(mut self, init: Init) -> Self {
        self.args.weight_initialization = init as i32;
        self
    }
    pub fn dropout(mut self, rate: f32) -> Self {
        self.args.dropout_rate = rate;
        self
    }
    pub fn stride(mut self, stride: u32) -> Self {
        self.args.stride = stride;
        self
    }
    pub fn padding(mut self, padding: u32) -> Self {
        self.args.padding = padding;
        self
    }
    pub fn groups(mut self, groups: u32) -> Self {
        self.args.groups = groups;
        self
    }
    pub fn epsilon(mut self, epsilon: f32) -> Self {
        self.args.epsilon = epsilon;
        self
    }
    /// Output cells a transposed convolution adds at the bottom and the right.
    pub fn output_padding(mut self, padding: u32) -> Self {
        self.args.output_padding = padding;
        self
    }
    /// Bilinear upsampling.
    pub fn bilinear(mut self) -> Self {
        self.args.upsample = sys::UPSAMPLE_BILINEAR;
        self
    }
    /// The layers it reads (indices that Network::add returned; 0 the input layer); none: the last one.
    pub fn inputs(mut self, inputs: &[u32]) -> Self {
        let n = inputs.len().min(sys::MAX_INPUTS);
        self.args.inputs[..n].copy_from_slice(&inputs[..n]);
        self.args.input_count = inputs.len() as u32;
        self
    }
    /// Key and value heads of attention (grouped queries; 0: as many as the query heads).
    pub fn kv_heads(mut self, heads: u32) -> Self {
        self.args.kv_heads = heads;
        self
    }
    /// Each query sees itself and the cells before it only (language models).
    pub fn causal(mut self, causal: bool) -> Self {
        self.args.causal = causal;
        self
    }
    /// Rotary position embeddings of base theta (10000 is usual) on the queries and keys.
    pub fn rope_theta(mut self, theta: f32) -> Self {
        self.args.rope_theta = theta;
        self
    }
    /// Learned position vectors added to an embedding's (GPT-2's).
    pub fn positions(mut self, positions: bool) -> Self {
        self.args.positions = positions;
        self
    }
}

/// Options of Network::train() (the defaults: 10 epochs of mini-batches of 32, Adam at 0.001).
#[derive(Clone, Copy, Debug)]
pub struct TrainOptions {
    pub epochs: usize,
    pub batch_size: u32,
    pub strategy: Strategy,
    pub optimizer: Optimizer,
    pub learning_rate: f32,
    pub weight_decay: f32,
    pub momentum: f32,
    pub beta1: f32,
    pub beta2: f32,
    /// Clip of the global norm of each step's gradient (0: off).
    pub max_grad_norm: f32,
    pub label_smoothing: f32,
    /// Keep the samples in order.
    pub no_shuffle: bool,
}

impl Default for TrainOptions {
    fn default() -> Self {
        TrainOptions {
            epochs: 10,
            batch_size: 32,
            strategy: Strategy::SmallBatch,
            optimizer: Optimizer::Adam,
            learning_rate: 1e-3,
            weight_decay: 0.0,
            momentum: 0.0,
            beta1: 0.0,
            beta2: 0.0,
            max_grad_norm: 0.0,
            label_smoothing: 0.0,
            no_shuffle: false,
        }
    }
}

/// How a training run ended.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrainReport {
    pub completed: bool,
    pub epochs_run: usize,
    pub train_loss: f32,
}

/// Mean loss and accuracy over a data set.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Metrics {
    pub loss: f32,
    pub accuracy: f32,
}

/// How Network::generate() picks each token (the default: the most likely one).
#[derive(Clone, Debug, Default)]
pub struct Sampling {
    /// The logits divided by it before the softmax; 0: the most likely token, no draws.
    pub temperature: f32,
    /// Draws among the top_k most likely tokens only (0: all).
    pub top_k: u32,
    /// And among the most likely whose probabilities add up to top_p (0: all).
    pub top_p: f32,
    /// The draws' generator (0: one draw of the library's).
    pub seed: u64,
    /// Tokens that end the generation (returned as the last one).
    pub stop: Vec<u32>,
}

/// A network: its layers, parameters and training state.
pub struct Network {
    ptr: NonNull<sys::Network>,
}

// A network moves between threads; calls on one network must not overlap.
unsafe impl Send for Network {}

impl Network {
    /// An empty network that trains with `loss`; its first layer is the input layer.
    pub fn new(loss: Loss) -> Result<Self> {
        let args = sys::NetworkArgs { loss_func: loss as i32, reserved: [0; sys::RESERVED] };
        let ptr = checked("spingalett_network_new", || unsafe { sys::spingalett_network_new_args(args) })?;
        NonNull::new(ptr).map(|ptr| Network { ptr }).ok_or_else(|| last_error("network allocation failed"))
    }

    /// A network saved with Network::save() (or by any program of the library: .slett files).
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = c_path(path.as_ref())?;
        let ptr = unsafe { sys::spingalett_load(path.as_ptr()) };
        NonNull::new(ptr).map(|ptr| Network { ptr }).ok_or_else(|| last_error("load failed"))
    }

    fn raw(&self) -> *mut sys::Network {
        self.ptr.as_ptr()
    }

    /// Appends a layer; returns its index (what Layer::inputs() of later layers names).
    pub fn add(&mut self, layer: Layer) -> Result<u32> {
        let mut args = layer.args;
        args.net = self.raw();
        let index = checked("spingalett_append_layer", || unsafe { sys::spingalett_append_layer(args) })?;
        if index == sys::NO_LAYER { Err(last_error("the layer does not fit the network")) } else { Ok(index) }
    }

    /// Layers, the input layer included.
    pub fn layer_count(&self) -> u32 {
        unsafe { sys::spingalett_layer_count(self.raw()) }
    }
    pub fn input_size(&self) -> u32 {
        unsafe { sys::spingalett_input_size(self.raw()) }
    }
    pub fn output_size(&self) -> u32 {
        unsafe { sys::spingalett_output_size(self.raw()) }
    }
    /// Targets a sample (the output size, or one class index a cell with the sparse cross-entropy).
    pub fn target_size(&self) -> u32 {
        unsafe { sys::spingalett_target_size(self.raw()) }
    }
    /// Weights and biases.
    pub fn parameter_count(&self) -> u64 {
        unsafe { sys::spingalett_parameter_count(self.raw()) }
    }

    fn samples(&self, values: usize, size: u32, what: &str) -> Result<u32> {
        if size == 0 || values % size as usize != 0 || values / size as usize > u32::MAX as usize {
            return Err(invalid(format!("{what}: {values} values are no whole number of samples of {size}")));
        }
        Ok((values / size as usize) as u32)
    }

    /// Trains on samples in arrays: `inputs` of input_size() values a sample, `targets` of target_size().
    pub fn train(&mut self, inputs: &[f32], targets: &[f32], options: &TrainOptions) -> Result<TrainReport> {
        let n = self.samples(inputs.len(), self.input_size(), "train")?;
        if self.samples(targets.len(), self.target_size(), "train")? != n {
            return Err(invalid("train: as many samples of targets as of inputs".into()));
        }
        let mut args = self.train_args(options);
        args.inputs = inputs.as_ptr();
        args.targets = targets.as_ptr();
        args.sample_count = n;
        self.run(args)
    }

    /// Trains a language model on the windows of a file of token ids (nanoGPT's .bin files of uint16,
    /// llm.c's): each sample input_size() tokens from a multiple of `stride` (0: the window's length) on,
    /// its targets the token after each.
    pub fn train_tokens(&mut self, path: impl AsRef<Path>, stride: u32, options: &TrainOptions) -> Result<TrainReport> {
        let path = c_path(path.as_ref())?;
        let reader_options = sys::TokenReaderOptions {
            context: self.input_size(),
            stride,
            shuffle: !options.no_shuffle,
            ..Default::default()
        };
        let reader = unsafe { sys::spingalett_dataset_open_tokens(path.as_ptr(), &reader_options) };
        if reader.is_null() {
            return Err(last_error("spingalett_dataset_open_tokens failed"));
        }
        let mut args = self.train_args(options);
        args.training_mode = sys::MODE_GENERATOR_FUNCTION;
        args.generator = Some(sys::spingalett_dataset_generator);
        args.generator_data = reader as *mut c_void;
        args.sample_count = unsafe { sys::spingalett_dataset_info(reader) }.count;
        let report = self.run(args);
        unsafe { sys::spingalett_dataset_close(reader) };
        report
    }

    fn train_args(&self, o: &TrainOptions) -> sys::TrainArgs {
        sys::TrainArgs {
            net: self.raw(),
            training_strategy: o.strategy as i32,
            optimizer_type: o.optimizer as i32,
            batch_size: o.batch_size,
            do_not_shuffle: o.no_shuffle,
            epochs: o.epochs,
            learning_rate: o.learning_rate,
            weight_decay: o.weight_decay,
            momentum: o.momentum,
            beta1: o.beta1,
            beta2: o.beta2,
            max_grad_norm: o.max_grad_norm,
            label_smoothing: o.label_smoothing,
            ..Default::default()
        }
    }

    fn run(&mut self, args: sys::TrainArgs) -> Result<TrainReport> {
        let r = unsafe { sys::spingalett_train_args(args) };
        if r.status == sys::TRAIN_FAILED || r.status == sys::TRAIN_NO_DATA {
            return Err(last_error("training failed"));
        }
        Ok(TrainReport { completed: r.status == sys::TRAIN_COMPLETED, epochs_run: r.epochs_run, train_loss: r.train_loss })
    }

    /// The outputs of samples of input_size() values each.
    pub fn predict(&mut self, inputs: &[f32]) -> Result<Vec<f32>> {
        let n = self.samples(inputs.len(), self.input_size(), "predict")?;
        let mut outputs = vec![0.0f32; n as usize * self.output_size() as usize];
        let args = sys::PredictArgs {
            net: self.raw(),
            inputs: inputs.as_ptr(),
            device_inputs: ptr::null(),
            sample_count: n,
            outputs: outputs.as_mut_ptr(),
            reserved: [0; sys::RESERVED],
        };
        if !unsafe { sys::spingalett_predict_args(args) } {
            return Err(last_error("predict failed"));
        }
        Ok(outputs)
    }

    /// Mean loss and accuracy over samples.
    pub fn evaluate(&mut self, inputs: &[f32], targets: &[f32]) -> Result<Metrics> {
        let n = self.samples(inputs.len(), self.input_size(), "evaluate")?;
        if self.samples(targets.len(), self.target_size(), "evaluate")? != n {
            return Err(invalid("evaluate: as many samples of targets as of inputs".into()));
        }
        let args = sys::EvaluateArgs {
            net: self.raw(),
            inputs: inputs.as_ptr(),
            targets: targets.as_ptr(),
            device_inputs: ptr::null(),
            device_targets: ptr::null(),
            sample_count: n,
            reserved: [0; sys::RESERVED],
        };
        let m = unsafe { sys::spingalett_evaluate_args(args) };
        if m.loss.is_nan() { Err(last_error("evaluate failed")) } else { Ok(Metrics { loss: m.loss, accuracy: m.accuracy }) }
    }

    /// Continues a prompt of token ids by `count` tokens with a causal language model (its outputs the
    /// logits of every token of its vocabulary for each input token), a token at a time.
    pub fn generate(&mut self, prompt: &[u32], count: u32, sampling: &Sampling) -> Result<Vec<u32>> {
        if prompt.is_empty() {
            return Err(invalid("generate: a prompt of one token at least".into()));
        }
        let mut tokens = vec![0u32; count as usize];
        let args = sys::GenerateArgs {
            net: self.raw(),
            prompt: prompt.as_ptr(),
            prompt_length: prompt.len() as u32,
            tokens: tokens.as_mut_ptr(),
            count,
            temperature: sampling.temperature,
            top_k: sampling.top_k,
            top_p: sampling.top_p,
            seed: sampling.seed,
            stop_tokens: if sampling.stop.is_empty() { ptr::null() } else { sampling.stop.as_ptr() },
            stop_count: sampling.stop.len() as u32,
            reserved: [0; sys::RESERVED],
        };
        let made = checked("generate", || unsafe { sys::spingalett_generate_args(args) })?;
        tokens.truncate(made as usize);
        Ok(tokens)
    }

    /// Saves the network (precision: of the weights in the file; Float32 keeps the optimizer's state too).
    pub fn save(&self, path: impl AsRef<Path>, precision: Precision) -> Result<()> {
        let path = c_path(path.as_ref())?;
        let args = sys::SaveArgs {
            net: self.raw(),
            filename: path.as_ptr(),
            do_not_save_optimizer: false,
            precision: precision as i32,
            reserved: [0; sys::RESERVED],
        };
        if unsafe { sys::spingalett_save_args(args) } { Ok(()) } else { Err(last_error("save failed")) }
    }

    /// What the library says of layer `layer` (0 the input layer).
    pub fn layer(&self, layer: u32) -> Result<sys::NetworkLayer> {
        let mut info = sys::NetworkLayer::default();
        if unsafe { sys::spingalett_network_layer(self.raw(), layer, &mut info) } {
            Ok(info)
        } else {
            Err(last_error("no such layer"))
        }
    }

    /// The weights of layer `layer`, as the library lays them out (rows of outputs, channels last).
    pub fn weights(&self, layer: u32) -> Result<Vec<f32>> {
        let count = self.layer(layer)?.weight_count;
        self.read(layer, sys::PARAM_WEIGHTS, count)
    }
    /// The biases of layer `layer`.
    pub fn biases(&self, layer: u32) -> Result<Vec<f32>> {
        let count = self.layer(layer)?.bias_count;
        self.read(layer, sys::PARAM_BIASES, count)
    }
    fn read(&self, layer: u32, kind: sys::ParameterKind, count: u64) -> Result<Vec<f32>> {
        let mut values = vec![0.0f32; count as usize];
        if count == 0 || unsafe { sys::spingalett_get_parameters(self.raw(), layer, kind, values.as_mut_ptr(), count) } {
            Ok(values)
        } else {
            Err(last_error("spingalett_get_parameters failed"))
        }
    }

    /// Sets the weights (or biases, `biases`) of layer `layer`.
    pub fn set_parameters(&mut self, layer: u32, biases: bool, values: &[f32]) -> Result<()> {
        let kind = if biases { sys::PARAM_BIASES } else { sys::PARAM_WEIGHTS };
        if unsafe { sys::spingalett_set_parameters(self.raw(), layer, kind, values.as_ptr(), values.len() as u64) } {
            Ok(())
        } else {
            Err(last_error("set_parameters failed"))
        }
    }

    /// The network as a deployment model of `precision`.
    pub fn to_model(&self, precision: Precision) -> Result<Model> {
        let ptr = unsafe { sys::spingalett_model_from_network(self.raw(), precision as i32) };
        NonNull::new(ptr).map(|ptr| Model { ptr }).ok_or_else(|| last_error("spingalett_model_from_network failed"))
    }
}

impl Drop for Network {
    fn drop(&mut self) {
        unsafe { sys::spingalett_network_free(self.raw()) }
    }
}

/// A deployment model: inference only, in the precision of its weights (FP32 down to INT2).
pub struct Model {
    ptr: NonNull<sys::Model>,
}

// Models may be used from several threads at once.
unsafe impl Send for Model {}
unsafe impl Sync for Model {}

impl Model {
    /// A model from a .slett file.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = c_path(path.as_ref())?;
        let ptr = unsafe { sys::spingalett_model_load(path.as_ptr()) };
        NonNull::new(ptr).map(|ptr| Model { ptr }).ok_or_else(|| last_error("spingalett_model_load failed"))
    }
    pub fn input_size(&self) -> u32 {
        unsafe { self.ptr.as_ref() }.input_size
    }
    pub fn output_size(&self) -> u32 {
        unsafe { self.ptr.as_ref() }.output_size
    }
    /// The outputs of samples of input_size() values each.
    pub fn predict(&self, inputs: &[f32]) -> Result<Vec<f32>> {
        let size = self.input_size() as usize;
        if size == 0 || inputs.len() % size != 0 {
            return Err(invalid(format!("predict: {} values are no whole number of samples of {size}", inputs.len())));
        }
        let n = inputs.len() / size;
        let mut outputs = vec![0.0f32; n * self.output_size() as usize];
        if unsafe { sys::spingalett_model_predict(self.ptr.as_ptr(), inputs.as_ptr(), n as u32, outputs.as_mut_ptr()) } {
            Ok(outputs)
        } else {
            Err(last_error("model predict failed"))
        }
    }
}

impl Drop for Model {
    fn drop(&mut self) {
        unsafe { sys::spingalett_model_free(self.ptr.as_ptr()) }
    }
}
