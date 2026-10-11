<?php

// SPDX-License-Identifier: MIT
// Copyright (c) 2026 pka_human (pka_human@proton.me)

declare(strict_types=1);

namespace Spingalett;

/**
 * How a training run ended.
 */
final class TrainResult
{
    /** All epochs ran. */
    public bool $completed;
    public int $epochsRun;
    /** The mean training loss of the last epoch. */
    public float $trainLoss;

    public function __construct(bool $completed, int $epochsRun, float $trainLoss)
    {
        $this->completed = $completed;
        $this->epochsRun = $epochsRun;
        $this->trainLoss = $trainLoss;
    }
}
