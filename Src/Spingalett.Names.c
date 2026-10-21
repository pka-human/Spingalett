/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

#include "Spingalett.Private.h"

const char * const act_func_names[] = {
    "NONE", "SIGMOID", "RELU", "TANH", "LEAKY_RELU", "FOO52", "SOFTMAX", "GELU", "GELU_TANH", "SILU"
};

const char * const loss_func_names[] = {
    "MSE", "CROSS_ENTROPY", "SPARSE_CROSS_ENTROPY"
};

const char * const training_strategy_names[] = {
    "SAMPLE", "FULL_BATCH", "SMALL_BATCH"
};

const char * const training_mode_names[] = {
    "ARRAY", "GENERATOR_FUNCTION"
};

const char * const optimizer_names[] = {
    "SGD", "MOMENTUM", "RMSPROP", "ADAM", "ADAMW"
};

const char * const weight_initialization_names[] = {
    "RANDOM", "XAVIER", "HE", "NONE", "LECUN"
};

const char * const precision_names[] = {
    "FLOAT32", "FP16", "BF16", "INT8", "INT4", "INT2", "Q8_0", "Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q2_K", "Q3_K", "Q4_K",
    "Q5_K", "Q6_K", "IQ4_NL", "IQ4_XS"
};
_Static_assert(sizeof precision_names / sizeof *precision_names == PRECISION_COUNT, "a name for every precision");

const char * const compute_mode_names[] = {
    "SINGLE_THREADED", "OPENMP", "OPENBLAS", "CUDA", "VULKAN"
};
