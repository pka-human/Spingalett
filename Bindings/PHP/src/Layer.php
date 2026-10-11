<?php

// SPDX-License-Identifier: MIT
// Copyright (c) 2026 pka_human (pka_human@proton.me)

declare(strict_types=1);

namespace Spingalett;

use FFI\CData;

/**
 * A layer to add to a network (Network::add()): made by a kind's method, then options, each of which
 * returns the layer.
 */
final class Layer
{
    // SpingalettLayerType
    private const DENSE = 0;
    private const CONV2D = 1;
    private const MAX_POOL2D = 2;
    private const AVG_POOL2D = 3;
    private const BATCH_NORM = 4;
    private const ADD = 5;
    private const CONCAT = 6;
    private const GLOBAL_AVG_POOL = 7;
    private const CONV_TRANSPOSE2D = 8;
    private const UPSAMPLE = 9;
    private const LAYER_NORM = 10;
    private const EMBEDDING = 11;
    private const ATTENTION = 12;
    private const RMS_NORM = 13;
    private const MULTIPLY = 14;

    /** The most layers a layer reads (SPINGALETT_MAX_INPUTS). */
    public const MAX_INPUTS = 16;

    /** @var array<string, int|float|bool> fields of SpingalettLayerArgs */
    private array $fields;

    /** @var int[] the layers it reads */
    private array $inputs = [];

    private function __construct(int $type, array $fields = [])
    {
        $this->fields = ['type' => $type] + $fields;
    }

    /** The input layer: $height x $width x $channels values a sample (a window of tokens: 1 x 1 x tokens). */
    public static function input(int $height, int $width, int $channels): self
    {
        return new self(self::DENSE, ['height' => $height, 'width' => $width, 'channels' => $channels]);
    }

    /** A fully connected layer (LeCun initialization, no activation). */
    public static function dense(int $neurons): self
    {
        return new self(self::DENSE, ['neurons_amount' => $neurons, 'weight_initialization' => Init::LECUN]);
    }

    /** A 2D convolution: $filters output channels, $kernel x $kernel windows (He initialization, ReLU). */
    public static function conv2d(int $filters, int $kernel): self
    {
        return new self(self::CONV2D, ['filters' => $filters, 'kernel' => $kernel, 'act_func' => Activation::RELU,
                                       'weight_initialization' => Init::HE]);
    }

    /** A transposed 2D convolution (upsampling by its stride; He initialization, ReLU). */
    public static function convTranspose2d(int $filters, int $kernel): self
    {
        $layer = self::conv2d($filters, $kernel);
        $layer->fields['type'] = self::CONV_TRANSPOSE2D;
        return $layer;
    }

    /** The maximum over each $kernel x $kernel window. */
    public static function maxPool2d(int $kernel): self
    {
        return new self(self::MAX_POOL2D, ['kernel' => $kernel]);
    }

    /** The mean over each $kernel x $kernel window. */
    public static function avgPool2d(int $kernel): self
    {
        return new self(self::AVG_POOL2D, ['kernel' => $kernel]);
    }

    /** The mean of each channel over all cells. */
    public static function globalAvgPool(): self
    {
        return new self(self::GLOBAL_AVG_POOL);
    }

    /** Each channel normalized over a batch. */
    public static function batchNorm(): self
    {
        return new self(self::BATCH_NORM);
    }

    /** Each cell's channels normalized. */
    public static function layerNorm(): self
    {
        return new self(self::LAYER_NORM);
    }

    /** Each cell's channels divided by their root mean square (no biases). */
    public static function rmsNorm(): self
    {
        return new self(self::RMS_NORM);
    }

    /** The sum of the layers named (indices Network::add() returned). */
    public static function addLayers(int ...$inputs): self
    {
        return (new self(self::ADD))->inputs(...$inputs);
    }

    /** The layers named side by side along the channels. */
    public static function concat(int ...$inputs): self
    {
        return (new self(self::CONCAT))->inputs(...$inputs);
    }

    /** The product of the layers named, element by element (SwiGLU's gate). */
    public static function multiply(int ...$inputs): self
    {
        return (new self(self::MULTIPLY))->inputs(...$inputs);
    }

    /** Each cell made $factor x $factor cells (nearest neighbour; bilinear() for bilinear). */
    public static function upsample(int $factor): self
    {
        return new self(self::UPSAMPLE, ['stride' => $factor]);
    }

    /** A linear map of each cell's channels to $neurons (a 1 x 1 convolution: a transformer's projections). */
    public static function linear(int $neurons): self
    {
        return new self(self::CONV2D, ['filters' => $neurons, 'kernel' => 1, 'weight_initialization' => Init::LECUN]);
    }

    /** The vectors of $width values of a vocabulary's tokens, a token a cell. */
    public static function embedding(int $vocabulary, int $width): self
    {
        return new self(self::EMBEDDING, ['vocabulary' => $vocabulary, 'neurons_amount' => $width,
                                          'weight_initialization' => Init::LECUN]);
    }

    /** Attention of $heads query heads over the queries, keys and values the layer before packs. */
    public static function attention(int $heads): self
    {
        return new self(self::ATTENTION, ['heads' => $heads]);
    }

    /** The activation of its outputs (an Activation constant). */
    public function activation(int $activation): self
    {
        $this->fields['act_func'] = $activation;
        return $this;
    }

    /** The initialization of its weights (an Init constant). */
    public function init(int $init): self
    {
        $this->fields['weight_initialization'] = $init;
        return $this;
    }

    /** Inverted dropout of its outputs while training, in [0, 1). */
    public function dropout(float $rate): self
    {
        $this->fields['dropout_rate'] = $rate;
        return $this;
    }

    public function stride(int $stride): self
    {
        $this->fields['stride'] = $stride;
        return $this;
    }

    public function padding(int $padding): self
    {
        $this->fields['padding'] = $padding;
        return $this;
    }

    /** Convolution: channel groups (the input channels: depthwise). */
    public function groups(int $groups): self
    {
        $this->fields['groups'] = $groups;
        return $this;
    }

    /** Normalizations: added to the variance (the mean square). */
    public function epsilon(float $epsilon): self
    {
        $this->fields['epsilon'] = $epsilon;
        return $this;
    }

    /** Transposed convolution: cells added to the output's bottom and right. */
    public function outputPadding(int $padding): self
    {
        $this->fields['output_padding'] = $padding;
        return $this;
    }

    /** Upsampling: bilinear interpolation. */
    public function bilinear(): self
    {
        $this->fields['upsample'] = 1;
        return $this;
    }

    /** Attention: key and value heads, dividing the query heads (grouped-query attention). */
    public function kvHeads(int $heads): self
    {
        $this->fields['kv_heads'] = $heads;
        return $this;
    }

    /** Attention: each cell attends to itself and the cells before it only. */
    public function causal(bool $causal = true): self
    {
        $this->fields['causal'] = $causal;
        return $this;
    }

    /** Attention: rotary position embeddings of base $theta (10000 is usual). */
    public function ropeTheta(float $theta): self
    {
        $this->fields['rope_theta'] = $theta;
        return $this;
    }

    /** Embedding: a learned vector added to each cell (absolute position embeddings). */
    public function positions(bool $positions = true): self
    {
        $this->fields['positions'] = $positions;
        return $this;
    }

    /** The layers it reads (indices Network::add() returned; 0 the input layer); none: the one before. */
    public function inputs(int ...$inputs): self
    {
        if (count($inputs) > self::MAX_INPUTS) {
            throw new \InvalidArgumentException('a layer reads at most ' . self::MAX_INPUTS . ' layers');
        }
        $this->inputs = $inputs;
        return $this;
    }

    /** @internal Writes the layer into a zeroed SpingalettLayerArgs. */
    public function write(CData $args): void
    {
        foreach ($this->fields as $field => $value) {
            $args->$field = $value;
        }
        foreach ($this->inputs as $k => $input) {
            $args->inputs[$k] = $input;
        }
        $args->input_count = count($this->inputs);
    }
}
