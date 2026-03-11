#include "Spingalett.Private.h"
#include <math.h>
#include <float.h>

void apply_softmax(float *layer, uint32_t size) {
    float max_val = -FLT_MAX;
    for (uint16_t i = 0; i < size; i++) {
        if (layer[i] > max_val) max_val = layer[i];
    }

    float sum = 0.0f;
    for (uint16_t i = 0; i < size; i++) {
        layer[i] = expf(layer[i] - max_val);
        sum += layer[i];
    }

    if (sum > 0.0f) {
        float inv = 1.0f / sum;
        for (uint16_t i = 0; i < size; i++) layer[i] *= inv;
    }
}

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


