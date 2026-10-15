<?php

// SPDX-License-Identifier: MIT
// Copyright (c) 2026 pka_human (pka_human@proton.me)

declare(strict_types=1);

namespace Spingalett;

/**
 * A mean loss and accuracy over samples.
 */
final class Metrics
{
    public float $loss;
    public float $accuracy;

    public function __construct(float $loss, float $accuracy)
    {
        $this->loss = $loss;
        $this->accuracy = $accuracy;
    }
}
