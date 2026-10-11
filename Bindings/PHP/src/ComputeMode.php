<?php

// SPDX-License-Identifier: MIT
// Copyright (c) 2026 pka_human (pka_human@proton.me)

declare(strict_types=1);

namespace Spingalett;

/**
 * Where the library computes (SpingalettComputeMode).
 */
final class ComputeMode
{
    public const SINGLE_THREADED = 0;
    public const OPENMP = 1;
    public const OPENBLAS = 2;
    public const CUDA = 3;          // an NVIDIA GPU (Library::cudaDevice())
    public const VULKAN = 4;        // a GPU through Vulkan compute (Library::gpuDevice())

    private function __construct()
    {
    }
}
