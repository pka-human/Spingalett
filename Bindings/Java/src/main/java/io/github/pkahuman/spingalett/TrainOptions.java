// SPDX-License-Identifier: MIT
// Copyright (c) 2026 pka_human (pka_human@proton.me)

package io.github.pkahuman.spingalett;

/** Options of Network.train (the defaults: 10 epochs of mini-batches of 32, Adam at 0.001); setters chain. */
public final class TrainOptions {
    int epochs = 10, batchSize = 32;
    Strategy strategy = Strategy.SMALL_BATCH;
    Optimizer optimizer = Optimizer.ADAM;
    float learningRate = 1e-3f, weightDecay, momentum, beta1, beta2, maxGradNorm, labelSmoothing;
    boolean noShuffle;

    public TrainOptions epochs(int v) { epochs = v; return this; }
    public TrainOptions batchSize(int v) { batchSize = v; return this; }
    public TrainOptions strategy(Strategy v) { strategy = v; return this; }
    public TrainOptions optimizer(Optimizer v) { optimizer = v; return this; }
    public TrainOptions learningRate(float v) { learningRate = v; return this; }
    public TrainOptions weightDecay(float v) { weightDecay = v; return this; }
    public TrainOptions momentum(float v) { momentum = v; return this; }
    public TrainOptions betas(float b1, float b2) { beta1 = b1; beta2 = b2; return this; }
    /** Clip of the global norm of each step's gradient (0: off). */
    public TrainOptions maxGradNorm(float v) { maxGradNorm = v; return this; }
    public TrainOptions labelSmoothing(float v) { labelSmoothing = v; return this; }
    public TrainOptions noShuffle(boolean v) { noShuffle = v; return this; }
}
