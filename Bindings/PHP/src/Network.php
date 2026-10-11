<?php

// SPDX-License-Identifier: MIT
// Copyright (c) 2026 pka_human (pka_human@proton.me)

declare(strict_types=1);

namespace Spingalett;

use FFI;
use FFI\CData;

/**
 * A network: its layers, parameters and training state. Samples are arrays of floats, flat (the
 * network's input size values a sample, one after the other) or one array a sample; each call copies
 * them into the library's floats once.
 */
final class Network
{
    /** What spingalett_append_layer() returns on error (SPINGALETT_NO_LAYER). */
    private const NO_LAYER = 0xFFFFFFFF;

    // SpingalettTrainStatus
    private const TRAIN_FAILED = 0;
    private const TRAIN_COMPLETED = 1;
    private const TRAIN_NO_DATA = 5;

    private ?CData $net = null;

    /** An empty network that trains with $loss (a Loss constant); its first layer is the input layer. */
    public function __construct(int $loss = Loss::MSE)
    {
        $ffi = Library::ffi();
        $args = $ffi->new('SpingalettNetworkArgs');
        $args->loss_func = $loss;
        $ffi->spingalett_clear_error();
        $net = $ffi->spingalett_network_new_args($args);
        if (Native::isNull($net)) {
            throw SpingalettException::last('network allocation failed');
        }
        $this->net = $net;
    }

    /** A network from a .slett file. */
    public static function load(string $path): self
    {
        $net = Library::ffi()->spingalett_load($path);
        if (Native::isNull($net)) {
            throw SpingalettException::last('load failed');
        }
        $network = (new \ReflectionClass(self::class))->newInstanceWithoutConstructor();
        $network->net = $net;
        return $network;
    }

    /**
     * A pretrained language model: a Hugging Face model directory (config.json and model.safetensors, or its
     * shards) of LLaMA, Mistral, Qwen2 or GPT-2, or a GGUF file; $context tokens a sample (0: the model's
     * positions).
     */
    public static function fromPretrained(string $path, int $context = 0): self
    {
        $ffi = Library::ffi();
        $name = Native::cString($path);
        $args = $ffi->new('SpingalettPretrainedArgs');
        $args->path = Native::pointer($name, 'char');
        $args->context = $context;
        $net = $ffi->spingalett_load_pretrained_args($args);
        if (Native::isNull($net)) {
            throw SpingalettException::last('spingalett_load_pretrained failed');
        }
        $network = (new \ReflectionClass(self::class))->newInstanceWithoutConstructor();
        $network->net = $net;
        return $network;
    }

    /** Releases the network (as its destruction does). */
    public function close(): void
    {
        if ($this->net !== null) {
            Library::ffi()->spingalett_network_free($this->net);
            $this->net = null;
        }
    }

    public function __destruct()
    {
        $this->close();
    }

    /** The SpingalettNetwork * of the library, for the functions of Library::ffi() these methods leave out. */
    public function raw(): CData
    {
        if ($this->net === null) {
            throw new \LogicException('the network is closed');
        }
        return $this->net;
    }

    /** Appends a layer; returns its index, by which later layers name it (Layer::inputs()). */
    public function add(Layer $layer): int
    {
        $ffi = Library::ffi();
        $args = $ffi->new('SpingalettLayerArgs');
        $layer->write($args);
        $args->net = $this->raw();
        $ffi->spingalett_clear_error();
        $index = $ffi->spingalett_append_layer($args);
        if ($index === self::NO_LAYER) {
            throw SpingalettException::last('the layer does not fit the network');
        }
        return $index;
    }

    /** Layers, the input layer included. */
    public function layerCount(): int
    {
        return Library::ffi()->spingalett_layer_count($this->raw());
    }

    public function inputSize(): int
    {
        return Library::ffi()->spingalett_input_size($this->raw());
    }

    public function outputSize(): int
    {
        return Library::ffi()->spingalett_output_size($this->raw());
    }

    /** Targets a sample: the outputs, or one class index a cell with Loss::SPARSE_CROSS_ENTROPY. */
    public function targetSize(): int
    {
        return Library::ffi()->spingalett_target_size($this->raw());
    }

    /** Weights and biases. */
    public function parameterCount(): int
    {
        return Library::ffi()->spingalett_parameter_count($this->raw());
    }

    /** Trains on samples: inputSize() values a sample of inputs, targetSize() of targets. */
    public function train(array $inputs, array $targets, ?TrainOptions $options = null): TrainResult
    {
        $x = Native::floats($inputs, $values);
        $count = Native::samples($values, $this->inputSize(), 'train');
        $y = Native::floats($targets, $values);
        if (Native::samples($values, $this->targetSize(), 'train') !== $count || $count === 0) {
            throw new \InvalidArgumentException('train: as many samples of targets as of inputs, one at least');
        }
        $args = $this->trainArgs($options ?? new TrainOptions());
        $args->inputs = Native::pointer($x, 'float');
        $args->targets = Native::pointer($y, 'float');
        $args->sample_count = $count;
        return self::run($args);
    }

    /**
     * Trains a language model on the windows of a file of token ids (nanoGPT's .bin, llm.c's): each
     * sample inputSize() tokens from a multiple of $stride (0: the window's length) on, its targets the
     * token after each.
     */
    public function trainTokens(string $path, int $stride = 0, ?TrainOptions $options = null): TrainResult
    {
        $options = $options ?? new TrainOptions();
        $ffi = Library::ffi();
        $readerOptions = $ffi->new('SpingalettTokenReaderOptions');
        $readerOptions->context = $this->inputSize();
        $readerOptions->stride = $stride;
        $readerOptions->shuffle = !$options->noShuffle;
        $reader = $ffi->spingalett_dataset_open_tokens($path, FFI::addr($readerOptions));
        if (Native::isNull($reader)) {
            throw SpingalettException::last('spingalett_dataset_open_tokens failed');
        }
        try {
            $args = $this->trainArgs($options);
            $args->training_mode = 1;                   // SPINGALETT_MODE_GENERATOR_FUNCTION
            // (cast to the field's type: PHP 8.0's FFI assigns no function pointer of another declaration)
            $args->generator = $ffi->cast('SpingalettDataGeneratorFn', $ffi->spingalett_dataset_generator);
            $args->generator_data = $reader;
            $args->sample_count = $ffi->spingalett_dataset_info($reader)->count;
            return self::run($args);
        } finally {
            $ffi->spingalett_dataset_close($reader);
        }
    }

    /** The outputs of samples of inputSize() values each: outputSize() values a sample, flat. */
    public function predict(array $inputs): array
    {
        $x = Native::floats($inputs, $values);
        $count = Native::samples($values, $this->inputSize(), 'predict');
        if ($count === 0) {
            return [];
        }
        $outputs = $count * $this->outputSize();
        $y = Native::buffer('float', $outputs);
        $ffi = Library::ffi();
        $args = $ffi->new('SpingalettPredictArgs');
        $args->net = $this->raw();
        $args->inputs = Native::pointer($x, 'float');
        $args->sample_count = $count;
        $args->outputs = Native::pointer($y, 'float');
        if (!$ffi->spingalett_predict_args($args)) {
            throw SpingalettException::last('predict failed');
        }
        return Native::read($y, 'float', $outputs);
    }

    /** The mean loss and accuracy over samples. */
    public function evaluate(array $inputs, array $targets): Metrics
    {
        $x = Native::floats($inputs, $values);
        $count = Native::samples($values, $this->inputSize(), 'evaluate');
        $y = Native::floats($targets, $values);
        if (Native::samples($values, $this->targetSize(), 'evaluate') !== $count || $count === 0) {
            throw new \InvalidArgumentException('evaluate: as many samples of targets as of inputs, one at least');
        }
        $ffi = Library::ffi();
        $args = $ffi->new('SpingalettEvaluateArgs');
        $args->net = $this->raw();
        $args->inputs = Native::pointer($x, 'float');
        $args->targets = Native::pointer($y, 'float');
        $args->sample_count = $count;
        $metrics = $ffi->spingalett_evaluate_args($args);
        if (is_nan($metrics->loss)) {
            throw SpingalettException::last('evaluate failed');
        }
        return new Metrics($metrics->loss, $metrics->accuracy);
    }

    /**
     * Continues a prompt of token ids by $count tokens with a causal language model, a token at a time;
     * returns the tokens generated (fewer when a stop token came).
     */
    public function generate(array $prompt, int $count, ?Sampling $sampling = null): array
    {
        if ($prompt === []) {
            throw new \InvalidArgumentException('generate: a prompt of one token at least');
        }
        $sampling = $sampling ?? new Sampling();
        $ffi = Library::ffi();
        $p = Native::uint32s($prompt, $length);
        $tokens = Native::buffer('uint32_t', $count);
        $args = $ffi->new('SpingalettGenerateArgs');
        $args->net = $this->raw();
        $args->prompt = Native::pointer($p, 'uint32_t');
        $args->prompt_length = $length;
        $args->tokens = Native::pointer($tokens, 'uint32_t');
        $args->count = $count;
        $args->temperature = $sampling->temperature;
        $args->top_k = $sampling->topK;
        $args->top_p = $sampling->topP;
        $args->seed = $sampling->seed;
        if ($sampling->stop !== []) {
            $stop = Native::uint32s($sampling->stop, $stops);
            $args->stop_tokens = Native::pointer($stop, 'uint32_t');
            $args->stop_count = $stops;
        }
        $ffi->spingalett_clear_error();
        $made = $ffi->spingalett_generate_args($args);
        if ($ffi->spingalett_last_error_code() !== 0) {
            throw SpingalettException::last('generate failed');
        }
        return Native::read($tokens, 'uint32_t', $made);
    }

    /** Writes the network to a .slett file, its weights in $precision (a Precision constant). */
    public function save(string $path, int $precision = Precision::FLOAT32): void
    {
        $ffi = Library::ffi();
        $name = Native::cString($path);
        $args = $ffi->new('SpingalettSaveArgs');
        $args->net = $this->raw();
        $args->filename = Native::pointer($name, 'char');
        $args->precision = $precision;
        if (!$ffi->spingalett_save_args($args)) {
            throw SpingalettException::last('save failed');
        }
    }

    /** The weights of layer $layer (rows of outputs, channels last). */
    public function weights(int $layer): array
    {
        return $this->parameters($layer, false);
    }

    /** The biases of layer $layer. */
    public function biases(int $layer): array
    {
        return $this->parameters($layer, true);
    }

    /** Sets the weights (or, with $biases, the biases) of layer $layer. */
    public function setParameters(int $layer, bool $biases, array $values): void
    {
        $v = Native::floats($values, $count);
        if (!Library::ffi()->spingalett_set_parameters($this->raw(), $layer, $biases ? 1 : 0,
                                                       Native::pointer($v, 'float'), $count)) {
            throw SpingalettException::last('set_parameters failed');
        }
    }

    /** The network as a deployment model of $precision (a Precision constant). */
    public function toModel(int $precision = Precision::FLOAT32): Model
    {
        $model = Library::ffi()->spingalett_model_from_network($this->raw(), $precision);
        if (Native::isNull($model)) {
            throw SpingalettException::last('spingalett_model_from_network failed');
        }
        return new Model($model);
    }

    private function parameters(int $layer, bool $biases): array
    {
        $ffi = Library::ffi();
        $info = $ffi->new('SpingalettNetworkLayer');
        if (!$ffi->spingalett_network_layer($this->raw(), $layer, FFI::addr($info))) {
            throw SpingalettException::last('no such layer');
        }
        $count = $biases ? $info->bias_count : $info->weight_count;
        if ($count === 0) {
            return [];
        }
        $values = Native::buffer('float', $count);
        if (!$ffi->spingalett_get_parameters($this->raw(), $layer, $biases ? 1 : 0, Native::pointer($values, 'float'),
                                             $count)) {
            throw SpingalettException::last('get_parameters failed');
        }
        return Native::read($values, 'float', $count);
    }

    private function trainArgs(TrainOptions $o): CData
    {
        $args = Library::ffi()->new('SpingalettTrainArgs');
        $args->net = $this->raw();
        $args->training_strategy = $o->strategy;
        $args->optimizer_type = $o->optimizer;
        $args->batch_size = $o->batchSize;
        $args->do_not_shuffle = $o->noShuffle;
        $args->epochs = $o->epochs;
        $args->learning_rate = $o->learningRate;
        $args->weight_decay = $o->weightDecay;
        $args->momentum = $o->momentum;
        $args->beta1 = $o->beta1;
        $args->beta2 = $o->beta2;
        $args->max_grad_norm = $o->maxGradNorm;
        $args->label_smoothing = $o->labelSmoothing;
        $args->accumulation_steps = $o->accumulationSteps;
        $args->recompute = $o->recompute;
        return $args;
    }

    private static function run(CData $args): TrainResult
    {
        $report = Library::ffi()->spingalett_train_args($args);
        if ($report->status === self::TRAIN_FAILED || $report->status === self::TRAIN_NO_DATA) {
            throw SpingalettException::last('training failed');
        }
        return new TrainResult($report->status === self::TRAIN_COMPLETED, $report->epochs_run, $report->train_loss);
    }
}
