<?php

// SPDX-License-Identifier: MIT
// Copyright (c) 2026 pka_human (pka_human@proton.me)

declare(strict_types=1);

namespace Spingalett;

use FFI\CData;

/**
 * A deployment model: inference only, in the precision of its weights (integer kernels for INT8, INT4
 * and INT2). Made by Network::toModel() or Model::load().
 */
final class Model
{
    private ?CData $model;

    /** @internal A model of the library's SpingalettModel *, which it owns from then on. */
    public function __construct(CData $model)
    {
        $this->model = $model;
    }

    /**
     * A model from a .slett file, or a language model's GGUF file (its tensors as the file stores them, the
     * quantized types as block precisions) with a window of $context tokens (0: the model's positions, 4096 at
     * most).
     */
    public static function load(string $path, int $context = 0): self
    {
        $ffi = Library::ffi();
        $args = $ffi->new('SpingalettModelOpenArgs');
        $bytes = $ffi->new('char[' . (strlen($path) + 1) . ']');
        \FFI::memcpy($bytes, $path . "\0", strlen($path) + 1);
        $args->path = $ffi->cast('const char *', $bytes);
        $args->context = $context;
        $model = $ffi->spingalett_model_open_args($args);
        if (Native::isNull($model)) {
            throw SpingalettException::last('spingalett_model_open_args failed');
        }
        return new self($model);
    }

    /** Releases the model (as its destruction does). */
    public function close(): void
    {
        if ($this->model !== null) {
            Library::ffi()->spingalett_model_free($this->model);
            $this->model = null;
        }
    }

    public function __destruct()
    {
        $this->close();
    }

    /** The SpingalettModel * of the library, for the functions of Library::ffi() these methods leave out. */
    public function raw(): CData
    {
        if ($this->model === null) {
            throw new \LogicException('the model is closed');
        }
        return $this->model;
    }

    public function inputSize(): int
    {
        return $this->raw()->input_size;
    }

    public function outputSize(): int
    {
        return $this->raw()->output_size;
    }

    /** The outputs of samples of inputSize() values each (flat, or an array a sample), flat. */
    public function predict(array $inputs): array
    {
        $x = Native::floats($inputs, $values);
        $count = Native::samples($values, $this->inputSize(), 'predict');
        if ($count === 0) {
            return [];
        }
        $outputs = $count * $this->outputSize();
        $y = Native::buffer('float', $outputs);
        if (!Library::ffi()->spingalett_model_predict($this->raw(), Native::pointer($x, 'float'), $count,
                                                      Native::pointer($y, 'float'))) {
            throw SpingalettException::last('spingalett_model_predict failed');
        }
        return Native::read($y, 'float', $outputs);
    }

    /**
     * A session of this causal language model: $sequences sequences of up to $maxLength tokens (0: the
     * model's window) fed a token at a time (Session).
     */
    public function session(int $maxLength = 0, int $sequences = 1, int $gpuMemory = 0): Session
    {
        return new Session($this, $maxLength, $sequences, $gpuMemory);
    }

    /** The mean loss (the loss the network trained with) and accuracy over samples. */
    public function evaluate(array $inputs, array $targets): Metrics
    {
        $x = Native::floats($inputs, $values);
        $count = Native::samples($values, $this->inputSize(), 'evaluate');
        $y = Native::floats($targets, $values);
        if ($count === 0 || $values === 0 || $values % $count !== 0) {
            throw new \InvalidArgumentException('evaluate: as many samples of targets as of inputs, one at least');
        }
        $metrics = Library::ffi()->spingalett_model_evaluate($this->raw(), Native::pointer($x, 'float'),
                                                             Native::pointer($y, 'float'), $count);
        if (is_nan($metrics->loss)) {
            throw SpingalettException::last('spingalett_model_evaluate failed');
        }
        return new Metrics($metrics->loss, $metrics->accuracy);
    }
}
