<?php

// SPDX-License-Identifier: MIT
// Copyright (c) 2026 pka_human (pka_human@proton.me)

declare(strict_types=1);

namespace Spingalett;

/**
 * The optimizer of training (SpingalettOptimizerType).
 */
final class Optimizer
{
    public const SGD = 0;
    public const MOMENTUM = 1;
    public const RMSPROP = 2;
    public const ADAM = 3;
    public const ADAMW = 4;

    private function __construct()
    {
    }
}
