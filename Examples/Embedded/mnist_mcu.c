/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/*
 * Handwritten digits on a microcontroller with the standalone Spingalett inference engine. The
 * model (mnist_model.h, written by "ModelTool header") and a few MNIST test digits (mnist_samples.h,
 * written by make_samples.py) are compiled into flash; RAM holds the engine's workspace, one input
 * and one output. There is no heap and no file system: spingalett_model_init checks the image in
 * place and spingalett_model_run evaluates it there.
 */

#include <Spingalett/Spingalett.Inference.h>
#include <stdio.h>
#include "mnist_model.h"
#include "mnist_samples.h"

static float workspace[MNIST_MODEL_WORKSPACE / sizeof(float)];
static float input[MNIST_MODEL_INPUTS];
static float output[MNIST_MODEL_OUTPUTS];

static const char *precision_name(PrecisionMode p) {
    static const char *names[] = {"FP32", "FP16", "BF16", "INT8", "INT4", "INT2"};
    return (unsigned)p < PRECISION_COUNT ? names[p] : "?";
}

int main(void) {
    SpingalettModel model;
    int rc = spingalett_model_init(&model, mnist_model, MNIST_MODEL_SIZE);
    if (rc != SPINGALETT_OK) {
        printf("the model image was rejected: error %d\n", rc);
        return 1;
    }
    unsigned long macs = 0;
    printf("model in flash: %u bytes, workspace in RAM: %u bytes\n", (unsigned)model.image_size,
           (unsigned)sizeof workspace);
    for (uint32_t i = 0; i < model.layer_count; i++) {
        SpingalettLayerInfo layer;
        spingalett_model_layer(&model, i, &layer);
        printf("  layer %u: %u -> %u, %s weights\n", (unsigned)i + 1, (unsigned)layer.inputs,
               (unsigned)layer.outputs, precision_name(layer.precision));
        macs += (unsigned long)layer.inputs * layer.outputs;
    }
    printf("%lu multiply-accumulates per digit\n", macs);

    unsigned correct = 0;
    for (unsigned s = 0; s < SAMPLE_COUNT; s++) {
        for (unsigned k = 0; k < MNIST_MODEL_INPUTS; k++) input[k] = (float)sample_pixels[s][k] / 255.0f;
        rc = spingalett_model_run(&model, input, output, workspace);
        if (rc != SPINGALETT_OK) {
            printf("inference failed: error %d\n", rc);
            return 1;
        }
        unsigned best = 0;
        for (unsigned k = 1; k < MNIST_MODEL_OUTPUTS; k++)
            if (output[k] > output[best]) best = k;
        correct += best == sample_labels[s];
        if (s < 10)
            printf("  digit %u: recognised as %u (%.1f%%)\n", (unsigned)sample_labels[s], best, 100.0 * (double)output[best]);
    }
    printf("accuracy: %u of %u test digits (%.1f%%)\n", correct, (unsigned)SAMPLE_COUNT, 100.0 * correct / SAMPLE_COUNT);
    return correct * 100 >= SAMPLE_COUNT * 95u ? 0 : 2;
}
