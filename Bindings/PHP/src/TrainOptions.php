<?php

// SPDX-License-Identifier: MIT
// Copyright (c) 2026 pka_human (pka_human@proton.me)

declare(strict_types=1);

namespace Spingalett;

/**
 * The options of Network::train() and Network::trainTokens(): by default 10 epochs of mini-batches of
 * 32, Adam at 0.001. Give them by name (PHP 8: new TrainOptions(epochs: 1500, batchSize: 4)), in this
 * order, or set the properties.
 */
final class TrainOptions
{
    public int $epochs;
    public int $batchSize;
    public float $learningRate;
    /** An Optimizer constant. */
    public int $optimizer;
    /** A Strategy constant. */
    public int $strategy;
    public float $weightDecay;
    /** SGD with momentum and RMSProp: 0 is the library's default. */
    public float $momentum;
    /** Adam and AdamW: 0 is the library's default. */
    public float $beta1;
    public float $beta2;
    /** A clip of the global L2 norm of each step's gradient (0: none). */
    public float $maxGradNorm;
    /** Targets moved this far towards the uniform distribution, in [0, 1). */
    public float $labelSmoothing;
    /** Keeps the samples in their order. */
    public bool $noShuffle;
    /** Gradient accumulation: an optimizer step after every this many mini-batches (0 or 1: every one). */
    public int $accumulationSteps;

    public function __construct(int $epochs = 10, int $batchSize = 32, float $learningRate = 1e-3,
                                int $optimizer = Optimizer::ADAM, int $strategy = Strategy::SMALL_BATCH,
                                float $weightDecay = 0.0, float $momentum = 0.0, float $beta1 = 0.0,
                                float $beta2 = 0.0, float $maxGradNorm = 0.0, float $labelSmoothing = 0.0,
                                bool $noShuffle = false, int $accumulationSteps = 0)
    {
        $this->epochs = $epochs;
        $this->batchSize = $batchSize;
        $this->learningRate = $learningRate;
        $this->optimizer = $optimizer;
        $this->strategy = $strategy;
        $this->weightDecay = $weightDecay;
        $this->momentum = $momentum;
        $this->beta1 = $beta1;
        $this->beta2 = $beta2;
        $this->maxGradNorm = $maxGradNorm;
        $this->labelSmoothing = $labelSmoothing;
        $this->noShuffle = $noShuffle;
        $this->accumulationSteps = $accumulationSteps;
    }
}
