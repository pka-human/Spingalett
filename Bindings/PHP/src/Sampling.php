<?php

// SPDX-License-Identifier: MIT
// Copyright (c) 2026 pka_human (pka_human@proton.me)

declare(strict_types=1);

namespace Spingalett;

/**
 * How Network::generate() picks each token: by default the most likely one. Give the options by name
 * (PHP 8: new Sampling(temperature: 0.8, topK: 40)), in this order, or set the properties.
 */
final class Sampling
{
    /** The logits divided by it before the softmax; 0: the most likely token, no draws. */
    public float $temperature;
    /** Draws among the topK most likely tokens only (0: all). */
    public int $topK;
    /** And among the most likely whose probabilities add up to topP at least (0 or 1: all). */
    public float $topP;
    /** The draws' generator (0: one draw of the library's, Library::seed()). */
    public int $seed;
    /** @var int[] tokens that end the generation (returned as its last one) */
    public array $stop;

    public function __construct(float $temperature = 0.0, int $topK = 0, float $topP = 0.0, int $seed = 0,
                                array $stop = [])
    {
        $this->temperature = $temperature;
        $this->topK = $topK;
        $this->topP = $topP;
        $this->seed = $seed;
        $this->stop = $stop;
    }
}
