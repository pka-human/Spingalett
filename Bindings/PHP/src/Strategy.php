<?php

// SPDX-License-Identifier: MIT
// Copyright (c) 2026 pka_human (pka_human@proton.me)

declare(strict_types=1);

namespace Spingalett;

/**
 * How training goes through the samples (SpingalettTrainingStrategy).
 */
final class Strategy
{
    public const SAMPLE = 0;        // a step a sample
    public const FULL_BATCH = 1;    // a step an epoch
    public const SMALL_BATCH = 2;   // a step a mini-batch

    private function __construct()
    {
    }
}
