<?php

// SPDX-License-Identifier: MIT
// Copyright (c) 2026 pka_human (pka_human@proton.me)

declare(strict_types=1);

namespace Spingalett;

/**
 * Activation of a layer's outputs (SpingalettActivationFunction).
 */
final class Activation
{
    public const NONE = 0;
    public const SIGMOID = 1;
    public const RELU = 2;
    public const TANH = 3;
    public const LEAKY_RELU = 4;    // slope 0.01 below zero
    public const FOO52 = 5;         // slope 0.01 below 0 and above 1, identity between
    public const SOFTMAX = 6;       // over the layer's outputs
    public const GELU = 7;
    public const GELU_TANH = 8;     // GELU's tanh approximation (GPT-2's)
    public const SILU = 9;          // x sigmoid(x)

    private function __construct()
    {
    }
}
