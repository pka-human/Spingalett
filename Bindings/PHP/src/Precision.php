<?php

// SPDX-License-Identifier: MIT
// Copyright (c) 2026 pka_human (pka_human@proton.me)

declare(strict_types=1);

namespace Spingalett;

/**
 * Precision of saved files and deployment models, and of the GPU's products (FLOAT32 or BFLOAT16)
 * (SpingalettPrecisionMode).
 */
final class Precision
{
    public const FLOAT32 = 0;
    public const FP16 = 1;
    public const BFLOAT16 = 2;
    public const INT8 = 3;
    public const INT4 = 4;
    public const INT2 = 5;

    private function __construct()
    {
    }
}
