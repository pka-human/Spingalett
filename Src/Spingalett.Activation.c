/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

#include "Spingalett.Private.h"
#include <math.h>

float spingalett_activate(float x, ActivationFunction act_func) {
    return spingalett_activate_value(x, act_func);
}

float spingalett_derivative(float x, ActivationFunction act_func) {
    switch (act_func) {
        case ACT_RELU:
            return x > 0.0f ? 1.0f : 0.0f;
        case ACT_LEAKY_RELU:
            return x > 0.0f ? 1.0f : 0.01f;
        case ACT_SIGMOID:
        case ACT_SOFTMAX:
            return x * (1.0f - x);
        case ACT_TANH:
            return 1.0f - (x * x);
        case ACT_FOO52:
            return (x > 1.0f || x < 0.0f) ? 0.01f : 1.0f;
        case ACT_GELU:
        case ACT_GELU_TANH:
        case ACT_SILU:                  /* of the value before the activation */
            return spingalett_input_derivative(x, act_func);
        default:
            return 1.0f;
    }
}


