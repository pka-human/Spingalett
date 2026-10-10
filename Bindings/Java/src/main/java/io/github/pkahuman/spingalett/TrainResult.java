// SPDX-License-Identifier: MIT
// Copyright (c) 2026 pka_human (pka_human@proton.me)

package io.github.pkahuman.spingalett;

/** How a training run ended. */
public record TrainResult(boolean completed, long epochsRun, float trainLoss) {}
