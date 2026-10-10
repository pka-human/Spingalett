// SPDX-License-Identifier: MIT
// Copyright (c) 2026 pka_human (pka_human@proton.me)

// Package spingalett binds Spingalett, a neural-network library in C (libspingalett): dense, convolutional and
// transformer layers as chains or graphs, trained on the CPU or the GPU (CUDA, Vulkan), deployment models from
// FP32 down to INT2, and text generation. The library is found through pkg-config (spingalett.pc, installed
// with it), or CGO_CFLAGS and CGO_LDFLAGS.
//
//	net, _ := spingalett.NewNetwork(spingalett.LossMSE)
//	defer net.Close()
//	net.Add(spingalett.Input(1, 1, 2))
//	net.Add(spingalett.Dense(8).Activation(spingalett.ActTanh))
//	net.Add(spingalett.Dense(1).Activation(spingalett.ActSigmoid))
//	net.Train(x, y, spingalett.TrainOptions{Epochs: 1500, BatchSize: 4, LearningRate: 0.05})
package spingalett

/*
#cgo pkg-config: spingalett
#include <stdlib.h>
#include <Spingalett/Spingalett.h>
*/
import "C"

import (
	"errors"
	"fmt"
	"runtime"
	"unsafe"
)

// Activation of a layer's outputs.
type Activation int32

const (
	ActNone      Activation = C.SPINGALETT_ACT_NONE
	ActSigmoid   Activation = C.SPINGALETT_ACT_SIGMOID
	ActReLU      Activation = C.SPINGALETT_ACT_RELU
	ActTanh      Activation = C.SPINGALETT_ACT_TANH
	ActLeakyReLU Activation = C.SPINGALETT_ACT_LEAKY_RELU
	ActSoftmax   Activation = C.SPINGALETT_ACT_SOFTMAX
	ActGELU      Activation = C.SPINGALETT_ACT_GELU
	ActGELUTanh  Activation = C.SPINGALETT_ACT_GELU_TANH
	ActSiLU      Activation = C.SPINGALETT_ACT_SILU
)

// Loss a network trains with.
type Loss int32

const (
	LossMSE                Loss = C.SPINGALETT_LOSS_MSE
	LossCrossEntropy       Loss = C.SPINGALETT_LOSS_CROSS_ENTROPY
	LossSparseCrossEntropy Loss = C.SPINGALETT_LOSS_SPARSE_CROSS_ENTROPY
)

// ComputeMode says where the library computes.
type ComputeMode int32

const (
	ComputeSingleThreaded ComputeMode = C.SPINGALETT_COMPUTE_SINGLE_THREADED
	ComputeOpenMP         ComputeMode = C.SPINGALETT_COMPUTE_OPENMP
	ComputeOpenBLAS       ComputeMode = C.SPINGALETT_COMPUTE_OPENBLAS
	ComputeCUDA           ComputeMode = C.SPINGALETT_COMPUTE_CUDA
	ComputeVulkan         ComputeMode = C.SPINGALETT_COMPUTE_VULKAN
)

// Precision of deployment models, and of the GPU's products (Float32 or BFloat16).
type Precision int32

const (
	Float32  Precision = C.SPINGALETT_PRECISION_FLOAT32
	FP16     Precision = C.SPINGALETT_PRECISION_FP16
	BFloat16 Precision = C.SPINGALETT_PRECISION_BFLOAT16
	Int8     Precision = C.SPINGALETT_PRECISION_INT8
	Int4     Precision = C.SPINGALETT_PRECISION_INT4
	Int2     Precision = C.SPINGALETT_PRECISION_INT2
)

// Optimizer of training (the zero value: Adam).
type Optimizer int32

const (
	Adam Optimizer = iota
	AdamW
	SGD
	Momentum
	RMSProp
)

var optimizers = [...]C.SpingalettOptimizerType{C.SPINGALETT_OPTIMIZER_ADAM, C.SPINGALETT_OPTIMIZER_ADAMW,
	C.SPINGALETT_OPTIMIZER_SGD, C.SPINGALETT_OPTIMIZER_MOMENTUM, C.SPINGALETT_OPTIMIZER_RMSPROP}

// Strategy says how training goes through the samples (the zero value: mini-batches).
type Strategy int32

const (
	SmallBatch Strategy = iota
	FullBatch
	PerSample
)

var strategies = [...]C.SpingalettTrainingStrategy{C.SPINGALETT_STRATEGY_SMALL_BATCH, C.SPINGALETT_STRATEGY_FULL_BATCH,
	C.SPINGALETT_STRATEGY_SAMPLE}

// Init is the initialization of a layer's weights.
type Init int32

const (
	InitRandom Init = C.SPINGALETT_INIT_RANDOM
	InitXavier Init = C.SPINGALETT_INIT_XAVIER
	InitHe     Init = C.SPINGALETT_INIT_HE
	InitZeros  Init = C.SPINGALETT_INIT_NONE
	InitLeCun  Init = C.SPINGALETT_INIT_LECUN
)

// Error is an error the library reported: its code (SPINGALETT_ERR_*) and message.
type Error struct {
	Code    int
	Message string
}

func (e *Error) Error() string { return fmt.Sprintf("%s (code %d)", e.Message, e.Code) }

func lastError(what string) error {
	code := int(C.spingalett_last_error_code())
	message := ""
	if m := C.spingalett_last_error_message(); m != nil {
		message = C.GoString(m)
	}
	if code == 0 {
		code = -1
	}
	if message == "" {
		message = what
	}
	return &Error{Code: code, Message: message}
}

// Version is the library's version ("1.2.0").
func Version() string { return C.GoString(C.spingalett_version()) }

// SetComputeMode says where training and inference run; false when the mode is not available.
func SetComputeMode(mode ComputeMode) bool {
	return bool(C.spingalett_set_compute_mode(C.SpingalettComputeMode(mode)))
}

// SetGPUPrecision sets the precision of the GPU's products (Float32 or BFloat16); false when it has none such.
func SetGPUPrecision(p Precision) bool {
	return bool(C.spingalett_set_gpu_precision(C.SpingalettPrecisionMode(p)))
}

// Seed seeds the calling thread's generator (initialization, shuffling, dropout, draws).
func Seed(seed uint64) { C.spingalett_seed(C.uint64_t(seed)) }

// SetVerbose turns the library's informational messages on or off.
func SetVerbose(on bool) { C.spingalett_set_verbose(C.bool(on)) }

// CUDADevice is the name of the GPU that ComputeCUDA uses, or "".
func CUDADevice() string {
	if d := C.spingalett_cuda_device(); d != nil {
		return C.GoString(d)
	}
	return ""
}

// Layer is a layer to add to a network: made by a kind's function, then options (each returns the layer).
type Layer struct {
	args C.SpingalettLayerArgs
}

func layerOf(kind C.SpingalettLayerType) Layer {
	var l Layer
	l.args._type = kind
	return l
}

// Input is the input layer: height x width x channels values a sample (a window of tokens: 1 x 1 x tokens).
func Input(height, width, channels uint32) Layer {
	l := layerOf(C.SPINGALETT_LAYER_DENSE)
	l.args.height, l.args.width, l.args.channels = C.uint32_t(height), C.uint32_t(width), C.uint32_t(channels)
	return l
}

// Dense is a fully connected layer (LeCun initialization, no activation).
func Dense(neurons uint32) Layer {
	l := layerOf(C.SPINGALETT_LAYER_DENSE)
	l.args.neurons_amount = C.uint32_t(neurons)
	l.args.weight_initialization = C.SPINGALETT_INIT_LECUN
	return l
}

// Conv2D is a 2D convolution of filters output channels, kernel x kernel windows (He initialization, ReLU).
func Conv2D(filters, kernel uint32) Layer {
	l := layerOf(C.SPINGALETT_LAYER_CONV2D)
	l.args.filters, l.args.kernel = C.uint32_t(filters), C.uint32_t(kernel)
	l.args.act_func = C.SPINGALETT_ACT_RELU
	l.args.weight_initialization = C.SPINGALETT_INIT_HE
	return l
}

// ConvTranspose2D is a transposed 2D convolution (upsampling by its stride).
func ConvTranspose2D(filters, kernel uint32) Layer {
	l := Conv2D(filters, kernel)
	l.args._type = C.SPINGALETT_LAYER_CONV_TRANSPOSE2D
	return l
}

// MaxPool2D is the maximum over each kernel x kernel window.
func MaxPool2D(kernel uint32) Layer {
	l := layerOf(C.SPINGALETT_LAYER_MAX_POOL2D)
	l.args.kernel = C.uint32_t(kernel)
	return l
}

// AvgPool2D is the mean over each kernel x kernel window.
func AvgPool2D(kernel uint32) Layer {
	l := layerOf(C.SPINGALETT_LAYER_AVG_POOL2D)
	l.args.kernel = C.uint32_t(kernel)
	return l
}

// GlobalAvgPool is the mean of each channel over all cells.
func GlobalAvgPool() Layer { return layerOf(C.SPINGALETT_LAYER_GLOBAL_AVG_POOL) }

// BatchNorm normalizes each channel over a batch.
func BatchNorm() Layer { return layerOf(C.SPINGALETT_LAYER_BATCH_NORM) }

// LayerNorm normalizes each cell's channels.
func LayerNorm() Layer { return layerOf(C.SPINGALETT_LAYER_LAYER_NORM) }

// RMSNorm normalizes each cell's channels by their root mean square (no biases).
func RMSNorm() Layer { return layerOf(C.SPINGALETT_LAYER_RMS_NORM) }

// AddLayers is the sum of the layers inputs names.
func AddLayers(inputs ...uint32) Layer { return layerOf(C.SPINGALETT_LAYER_ADD).Inputs(inputs...) }

// Concat is the layers inputs names side by side along the channels.
func Concat(inputs ...uint32) Layer { return layerOf(C.SPINGALETT_LAYER_CONCAT).Inputs(inputs...) }

// Multiply is the product of the layers inputs names, element by element (SwiGLU's gate).
func Multiply(inputs ...uint32) Layer { return layerOf(C.SPINGALETT_LAYER_MULTIPLY).Inputs(inputs...) }

// Upsample makes each cell factor x factor cells (nearest neighbour; Bilinear() for bilinear).
func Upsample(factor uint32) Layer {
	l := layerOf(C.SPINGALETT_LAYER_UPSAMPLE)
	l.args.stride = C.uint32_t(factor)
	return l
}

// Linear maps each cell's channels to neurons (a 1 x 1 convolution: a transformer's projections).
func Linear(neurons uint32) Layer {
	l := layerOf(C.SPINGALETT_LAYER_CONV2D)
	l.args.filters, l.args.kernel = C.uint32_t(neurons), 1
	l.args.weight_initialization = C.SPINGALETT_INIT_LECUN
	return l
}

// Embedding gives the vectors of width values of a vocabulary's tokens, a token a cell.
func Embedding(vocabulary, width uint32) Layer {
	l := layerOf(C.SPINGALETT_LAYER_EMBEDDING)
	l.args.vocabulary, l.args.neurons_amount = C.uint32_t(vocabulary), C.uint32_t(width)
	l.args.weight_initialization = C.SPINGALETT_INIT_LECUN
	return l
}

// Attention is attention of heads query heads over packed queries, keys and values.
func Attention(heads uint32) Layer {
	l := layerOf(C.SPINGALETT_LAYER_ATTENTION)
	l.args.heads = C.uint32_t(heads)
	return l
}

func (l Layer) Activation(a Activation) Layer {
	l.args.act_func = C.SpingalettActivationFunction(a)
	return l
}
func (l Layer) Init(i Init) Layer {
	l.args.weight_initialization = C.SpingalettWeightInitialization(i)
	return l
}
func (l Layer) Dropout(rate float32) Layer   { l.args.dropout_rate = C.float(rate); return l }
func (l Layer) Stride(s uint32) Layer        { l.args.stride = C.uint32_t(s); return l }
func (l Layer) Padding(p uint32) Layer       { l.args.padding = C.uint32_t(p); return l }
func (l Layer) Groups(g uint32) Layer        { l.args.groups = C.uint32_t(g); return l }
func (l Layer) Epsilon(e float32) Layer      { l.args.epsilon = C.float(e); return l }
func (l Layer) OutputPadding(p uint32) Layer { l.args.output_padding = C.uint32_t(p); return l }
func (l Layer) Bilinear() Layer              { l.args.upsample = C.SPINGALETT_UPSAMPLE_BILINEAR; return l }
func (l Layer) KVHeads(h uint32) Layer       { l.args.kv_heads = C.uint32_t(h); return l }
func (l Layer) Causal(c bool) Layer          { l.args.causal = C.bool(c); return l }
func (l Layer) RopeTheta(t float32) Layer    { l.args.rope_theta = C.float(t); return l }
func (l Layer) Positions(p bool) Layer       { l.args.positions = C.bool(p); return l }

// Inputs names the layers it reads (indices Add returned; 0 the input layer); none: the last one.
func (l Layer) Inputs(inputs ...uint32) Layer {
	for k, i := range inputs {
		if k < C.SPINGALETT_MAX_INPUTS {
			l.args.inputs[k] = C.uint32_t(i)
		}
	}
	l.args.input_count = C.uint32_t(len(inputs))
	return l
}

// TrainOptions are the options of Train (zero values: 10 epochs of mini-batches of 32, Adam at 0.001).
type TrainOptions struct {
	Epochs         int
	BatchSize      uint32
	Strategy       Strategy
	Optimizer      Optimizer
	LearningRate   float32
	WeightDecay    float32
	Momentum       float32
	Beta1, Beta2   float32
	MaxGradNorm    float32
	LabelSmoothing float32
	NoShuffle      bool
}

// TrainResult says how a training run ended.
type TrainResult struct {
	Completed bool
	EpochsRun int
	TrainLoss float32
}

// Metrics are a mean loss and accuracy.
type Metrics struct {
	Loss, Accuracy float32
}

// Sampling says how Generate picks each token (zero: the most likely one).
type Sampling struct {
	Temperature float32
	TopK        uint32
	TopP        float32
	Seed        uint64
	Stop        []uint32
}

// Network is a network: its layers, parameters and training state; used by one goroutine at a time.
type Network struct {
	p *C.SpingalettNetwork
}

// NewNetwork makes an empty network that trains with loss; its first layer is the input layer.
func NewNetwork(loss Loss) (*Network, error) {
	var args C.SpingalettNetworkArgs
	args.loss_func = C.SpingalettLossFunction(loss)
	C.spingalett_clear_error()
	p := C.spingalett_network_new_args(args)
	if p == nil {
		return nil, lastError("network allocation failed")
	}
	n := &Network{p: p}
	runtime.SetFinalizer(n, (*Network).Close)
	return n, nil
}

// Load reads a network from a .slett file.
func Load(path string) (*Network, error) {
	cpath := C.CString(path)
	defer C.free(unsafe.Pointer(cpath))
	p := C.spingalett_load(cpath)
	if p == nil {
		return nil, lastError("load failed")
	}
	n := &Network{p: p}
	runtime.SetFinalizer(n, (*Network).Close)
	return n, nil
}

// Close releases the network.
func (n *Network) Close() {
	if n.p != nil {
		C.spingalett_network_free(n.p)
		n.p = nil
	}
}

// Add appends a layer and returns its index (what later layers' Inputs name).
func (n *Network) Add(l Layer) (uint32, error) {
	l.args.net = n.p
	C.spingalett_clear_error()
	index := uint32(C.spingalett_append_layer(l.args))
	if index == ^uint32(0) {
		return 0, lastError("the layer does not fit the network")
	}
	return index, nil
}

func (n *Network) LayerCount() uint32     { return uint32(C.spingalett_layer_count(n.p)) }
func (n *Network) InputSize() uint32      { return uint32(C.spingalett_input_size(n.p)) }
func (n *Network) OutputSize() uint32     { return uint32(C.spingalett_output_size(n.p)) }
func (n *Network) TargetSize() uint32     { return uint32(C.spingalett_target_size(n.p)) }
func (n *Network) ParameterCount() uint64 { return uint64(C.spingalett_parameter_count(n.p)) }

func samples(values int, size uint32, what string) (uint32, error) {
	if size == 0 || values%int(size) != 0 {
		return 0, fmt.Errorf("%s: %d values are no whole number of samples of %d", what, values, size)
	}
	return uint32(values / int(size)), nil
}

func (n *Network) trainArgs(o TrainOptions) C.SpingalettTrainArgs {
	var a C.SpingalettTrainArgs
	a.net = n.p
	if o.Epochs == 0 {
		o.Epochs = 10
	}
	if o.BatchSize == 0 {
		o.BatchSize = 32
	}
	if o.LearningRate == 0 {
		o.LearningRate = 1e-3
	}
	a.training_strategy = strategies[o.Strategy]
	a.optimizer_type = optimizers[o.Optimizer]
	a.batch_size = C.uint32_t(o.BatchSize)
	a.do_not_shuffle = C.bool(o.NoShuffle)
	a.epochs = C.size_t(o.Epochs)
	a.learning_rate = C.float(o.LearningRate)
	a.weight_decay = C.float(o.WeightDecay)
	a.momentum = C.float(o.Momentum)
	a.beta1, a.beta2 = C.float(o.Beta1), C.float(o.Beta2)
	a.max_grad_norm = C.float(o.MaxGradNorm)
	a.label_smoothing = C.float(o.LabelSmoothing)
	return a
}

func run(a C.SpingalettTrainArgs) (TrainResult, error) {
	r := C.spingalett_train_args(a)
	if r.status == C.SPINGALETT_TRAIN_FAILED || r.status == C.SPINGALETT_TRAIN_NO_DATA {
		return TrainResult{}, lastError("training failed")
	}
	return TrainResult{Completed: r.status == C.SPINGALETT_TRAIN_COMPLETED, EpochsRun: int(r.epochs_run),
		TrainLoss: float32(r.train_loss)}, nil
}

// Train trains on samples in slices: InputSize values a sample of inputs, TargetSize of targets.
func (n *Network) Train(inputs, targets []float32, o TrainOptions) (TrainResult, error) {
	count, err := samples(len(inputs), n.InputSize(), "Train")
	if err != nil {
		return TrainResult{}, err
	}
	if t, err := samples(len(targets), n.TargetSize(), "Train"); err != nil || t != count {
		return TrainResult{}, errors.New("Train: as many samples of targets as of inputs")
	}
	var pin runtime.Pinner
	defer pin.Unpin()
	pin.Pin(&inputs[0])
	pin.Pin(&targets[0])
	a := n.trainArgs(o)
	a.inputs = (*C.float)(unsafe.Pointer(&inputs[0]))
	a.targets = (*C.float)(unsafe.Pointer(&targets[0]))
	a.sample_count = C.uint32_t(count)
	return run(a)
}

// TrainTokens trains a language model on the windows of a file of token ids (nanoGPT's .bin, llm.c's): each
// sample InputSize tokens from a multiple of stride (0: the window's length) on, its targets the token after each.
func (n *Network) TrainTokens(path string, stride uint32, o TrainOptions) (TrainResult, error) {
	cpath := C.CString(path)
	defer C.free(unsafe.Pointer(cpath))
	var ro C.SpingalettTokenReaderOptions
	ro.context = C.uint32_t(n.InputSize())
	ro.stride = C.uint32_t(stride)
	ro.shuffle = C.bool(!o.NoShuffle)
	reader := C.spingalett_dataset_open_tokens(cpath, &ro)
	if reader == nil {
		return TrainResult{}, lastError("spingalett_dataset_open_tokens failed")
	}
	defer C.spingalett_dataset_close(reader)
	a := n.trainArgs(o)
	a.training_mode = C.SPINGALETT_MODE_GENERATOR_FUNCTION
	a.generator = C.SpingalettDataGeneratorFn(C.spingalett_dataset_generator)
	a.generator_data = unsafe.Pointer(reader)
	a.sample_count = C.spingalett_dataset_info(reader).count
	return run(a)
}

// Predict gives the outputs of samples of InputSize values each.
func (n *Network) Predict(inputs []float32) ([]float32, error) {
	count, err := samples(len(inputs), n.InputSize(), "Predict")
	if err != nil {
		return nil, err
	}
	outputs := make([]float32, int(count)*int(n.OutputSize()))
	if count == 0 {
		return outputs, nil
	}
	var pin runtime.Pinner
	defer pin.Unpin()
	pin.Pin(&inputs[0])
	pin.Pin(&outputs[0])
	var a C.SpingalettPredictArgs
	a.net = n.p
	a.inputs = (*C.float)(unsafe.Pointer(&inputs[0]))
	a.sample_count = C.uint32_t(count)
	a.outputs = (*C.float)(unsafe.Pointer(&outputs[0]))
	if !C.spingalett_predict_args(a) {
		return nil, lastError("predict failed")
	}
	return outputs, nil
}

// Evaluate gives the mean loss and accuracy over samples.
func (n *Network) Evaluate(inputs, targets []float32) (Metrics, error) {
	count, err := samples(len(inputs), n.InputSize(), "Evaluate")
	if err != nil || count == 0 {
		return Metrics{}, errors.New("Evaluate: no whole samples")
	}
	var pin runtime.Pinner
	defer pin.Unpin()
	pin.Pin(&inputs[0])
	pin.Pin(&targets[0])
	var a C.SpingalettEvaluateArgs
	a.net = n.p
	a.inputs = (*C.float)(unsafe.Pointer(&inputs[0]))
	a.targets = (*C.float)(unsafe.Pointer(&targets[0]))
	a.sample_count = C.uint32_t(count)
	m := C.spingalett_evaluate_args(a)
	if m.loss != m.loss {
		return Metrics{}, lastError("evaluate failed")
	}
	return Metrics{Loss: float32(m.loss), Accuracy: float32(m.accuracy)}, nil
}

// Generate continues a prompt of token ids by count tokens with a causal language model, a token at a time.
func (n *Network) Generate(prompt []uint32, count uint32, s Sampling) ([]uint32, error) {
	if len(prompt) == 0 {
		return nil, errors.New("Generate: a prompt of one token at least")
	}
	tokens := make([]uint32, count+1)
	var pin runtime.Pinner
	defer pin.Unpin()
	pin.Pin(&prompt[0])
	pin.Pin(&tokens[0])
	var a C.SpingalettGenerateArgs
	a.net = n.p
	a.prompt = (*C.uint32_t)(unsafe.Pointer(&prompt[0]))
	a.prompt_length = C.uint32_t(len(prompt))
	a.tokens = (*C.uint32_t)(unsafe.Pointer(&tokens[0]))
	a.count = C.uint32_t(count)
	a.temperature = C.float(s.Temperature)
	a.top_k = C.uint32_t(s.TopK)
	a.top_p = C.float(s.TopP)
	a.seed = C.uint64_t(s.Seed)
	if len(s.Stop) > 0 {
		pin.Pin(&s.Stop[0])
		a.stop_tokens = (*C.uint32_t)(unsafe.Pointer(&s.Stop[0]))
		a.stop_count = C.uint32_t(len(s.Stop))
	}
	C.spingalett_clear_error()
	made := uint32(C.spingalett_generate_args(a))
	if C.spingalett_last_error_code() != 0 {
		return nil, lastError("generate failed")
	}
	return tokens[:made], nil
}

// Save writes the network to a .slett file (precision of the weights in it).
func (n *Network) Save(path string, p Precision) error {
	cpath := C.CString(path)
	defer C.free(unsafe.Pointer(cpath))
	var a C.SpingalettSaveArgs
	a.net = n.p
	a.filename = cpath
	a.precision = C.SpingalettPrecisionMode(p)
	if !C.spingalett_save_args(a) {
		return lastError("save failed")
	}
	return nil
}

func (n *Network) read(layer uint32, kind C.SpingalettParameterKind, biases bool) ([]float32, error) {
	var info C.SpingalettNetworkLayer
	if !C.spingalett_network_layer(n.p, C.uint32_t(layer), &info) {
		return nil, lastError("no such layer")
	}
	count := uint64(info.weight_count)
	if biases {
		count = uint64(info.bias_count)
	}
	values := make([]float32, count)
	if count == 0 {
		return values, nil
	}
	if !C.spingalett_get_parameters(n.p, C.uint32_t(layer), kind, (*C.float)(unsafe.Pointer(&values[0])), C.uint64_t(count)) {
		return nil, lastError("get_parameters failed")
	}
	return values, nil
}

// Weights gives the weights of a layer (rows of outputs, channels last).
func (n *Network) Weights(layer uint32) ([]float32, error) {
	return n.read(layer, C.SPINGALETT_PARAM_WEIGHTS, false)
}

// Biases gives the biases of a layer.
func (n *Network) Biases(layer uint32) ([]float32, error) {
	return n.read(layer, C.SPINGALETT_PARAM_BIASES, true)
}

// SetParameters sets the weights (or the biases) of a layer.
func (n *Network) SetParameters(layer uint32, biases bool, values []float32) error {
	kind := C.SpingalettParameterKind(C.SPINGALETT_PARAM_WEIGHTS)
	if biases {
		kind = C.SPINGALETT_PARAM_BIASES
	}
	var p *C.float
	if len(values) > 0 {
		p = (*C.float)(unsafe.Pointer(&values[0]))
	}
	if !C.spingalett_set_parameters(n.p, C.uint32_t(layer), kind, p, C.uint64_t(len(values))) {
		return lastError("set_parameters failed")
	}
	return nil
}

// ToModel makes a deployment model of the network in a precision.
func (n *Network) ToModel(p Precision) (*Model, error) {
	m := C.spingalett_model_from_network(n.p, C.SpingalettPrecisionMode(p))
	if m == nil {
		return nil, lastError("spingalett_model_from_network failed")
	}
	model := &Model{p: m}
	runtime.SetFinalizer(model, (*Model).Close)
	return model, nil
}

// Model is a deployment model: inference only, in its weights' precision; goroutines may share one.
type Model struct {
	p *C.SpingalettModel
}

// LoadModel reads a model from a .slett file.
func LoadModel(path string) (*Model, error) {
	cpath := C.CString(path)
	defer C.free(unsafe.Pointer(cpath))
	m := C.spingalett_model_load(cpath)
	if m == nil {
		return nil, lastError("spingalett_model_load failed")
	}
	model := &Model{p: m}
	runtime.SetFinalizer(model, (*Model).Close)
	return model, nil
}

// Close releases the model.
func (m *Model) Close() {
	if m.p != nil {
		C.spingalett_model_free(m.p)
		m.p = nil
	}
}

func (m *Model) InputSize() uint32  { return uint32(m.p.input_size) }
func (m *Model) OutputSize() uint32 { return uint32(m.p.output_size) }

// Predict gives the outputs of samples of InputSize values each.
func (m *Model) Predict(inputs []float32) ([]float32, error) {
	count, err := samples(len(inputs), m.InputSize(), "Predict")
	if err != nil || count == 0 {
		return nil, errors.New("Predict: no whole samples")
	}
	outputs := make([]float32, int(count)*int(m.OutputSize()))
	if !C.spingalett_model_predict(m.p, (*C.float)(unsafe.Pointer(&inputs[0])), C.uint32_t(count),
		(*C.float)(unsafe.Pointer(&outputs[0]))) {
		return nil, lastError("model predict failed")
	}
	return outputs, nil
}
