/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

#include "Spingalett.Private.h"
#include <math.h>

float activate(float x, ActivationFunction act_func) {
    switch (act_func) {
        case ACT_RELU:
            return x > 0.0f ? x : 0.0f;
        case ACT_LEAKY_RELU:
            return x > 0.0f ? x : 0.01f * x;
        case ACT_SIGMOID:
            return 1.0f / (1.0f + expf(-x));
        case ACT_TANH:
            return tanhf(x);
        case ACT_FOO52:
            return x > 1.0f ? 1.0f + 0.01f * (x - 1.0f) : (x < 0.0f ? 0.01f * x : x);
        default:
            return x;
    }
}

float derivative(float x, ActivationFunction act_func) {
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
        default:
            return 1.0f;
    }
}


