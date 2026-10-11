<?php

// SPDX-License-Identifier: MIT
// Copyright (c) 2026 pka_human (pka_human@proton.me)

declare(strict_types=1);

namespace Spingalett;

/**
 * The loss a network trains with (SpingalettLossFunction).
 */
final class Loss
{
    public const MSE = 0;
    public const CROSS_ENTROPY = 1;
    public const SPARSE_CROSS_ENTROPY = 2;  // a softmax over each cell's channels against one class index a cell

    private function __construct()
    {
    }
}
