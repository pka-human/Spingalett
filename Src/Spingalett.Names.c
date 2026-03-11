/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

#include "Spingalett.Private.h"

const char * const act_func_names[] = {
    "SIGMOID", "RELU", "TANH", "LEAKY_RELU", "FOO52", "SOFTMAX", "NONE"
};

const char * const loss_func_names[] = {
    "MSE", "CROSS_ENTROPY"
};

const char * const training_strategy_names[] = {
    "SGD", "FULL_BATCH", "SMALL_BATCH"
};

const char * const training_mode_names[] = {
    "ARRAY", "GENERATOR_FUNCTION"
};

const char * const optimizer_names[] = {
    "SGD", "MOMENTUM", "RMSPROP", "ADAM", "ADAMW"
};

const char * const weight_initialization_names[] = {
    "RANDOM", "XAVIER", "HE", "NONE"
};

const char * const precision_names[] = {
    "FLOAT32", "FP16", "BF16", "INT8", "INT4", "INT2"
};

const char * const compute_mode_names[] = {
    "SINGLE_THREADED", "OPENMP", "OPENBLAS", "CUDA"
};
