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
    /* GGUF's block types (format 10): language models' layers */
    public const Q8_0 = 6;
    public const Q4_0 = 7;
    public const Q4_1 = 8;
    public const Q5_0 = 9;
    public const Q5_1 = 10;
    public const Q2_K = 11;
    public const Q3_K = 12;
    public const Q4_K = 13;
    public const Q5_K = 14;
    public const Q6_K = 15;
    public const IQ4_NL = 16;
    public const IQ4_XS = 17;

    private function __construct()
    {
    }
}
