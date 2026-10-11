<?php

// SPDX-License-Identifier: MIT
// Copyright (c) 2026 pka_human (pka_human@proton.me)

declare(strict_types=1);

namespace Spingalett;

/**
 * The initialization of a layer's weights (SpingalettWeightInitialization).
 */
final class Init
{
    public const RANDOM = 0;        // uniform in [-1, 1]
    public const XAVIER = 1;        // Glorot normal
    public const HE = 2;            // He normal
    public const ZEROS = 3;         // SPINGALETT_INIT_NONE
    public const LECUN = 4;         // LeCun normal

    private function __construct()
    {
    }
}
