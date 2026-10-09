/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

#include <Spingalett/Spingalett.h>
#include <stdio.h>
#include <math.h>

static bool on_progress(NeuralNetwork *net, const TrainProgress *progress, void *user_data) {
    (void)net;
    float target_error = *(const float *)user_data;
    printf("  [callback] epoch %zu, error=%.6f\n", progress->epoch, (double)progress->train_loss);
    return progress->train_loss < target_error;
}

int main(void) {
    float inputs[] = {
        0.f, 0.f,
        0.f, 1.f,
        1.f, 0.f,
        1.f, 1.f
    };
    float targets[] = {
        0.f,
        1.f,
        1.f,
        0.f
    };

    /* A fixed seed gives the same initial weights on every run: unseeded, about one start in five
       hundred falls into XOR's local minimum (two of the four cases at 0.5). */
    spingalett_seed(42);

    NeuralNetwork *nn = new_spingalett(.loss_func = LOSS_MSE);

    layer(nn, 2);
    layer(nn, 8, ACT_FOO52, WEIGHT_INITIALIZATION_HE);
    layer(nn, 1, ACT_SIGMOID, WEIGHT_INITIALIZATION_XAVIER);

    float target_error = 1e-5f;
    TrainReport report = train(
        .net = nn,
        .training_mode = MODE_ARRAY,
        .training_strategy = STRATEGY_FULL_BATCH,
        .optimizer_type = OPTIMIZER_ADAMW,
        .inputs = inputs,
        .targets = targets,
        .sample_count = 4,
        .epochs = 10000,
        .learning_rate = 0.01f,
        .weight_decay = 1e-4f,
        .report_interval = 2000,
        .callback = on_progress,
        .callback_interval = 5000,
        .callback_data = &target_error
    );
    printf("trained %zu epochs, final error %.6f\n", report.epochs_run, (double)report.train_loss);

    printf("\n--- XOR Results ---\n");
    int passed = 0;
    for (int i = 0; i < 4; i++) {
        float *out = forward(.net = nn, .input = &inputs[i * 2]);
        float expected = targets[i];
        float err = fabsf(out[0] - expected);
        const char *status = (err < 0.1f) ? "OK" : "FAIL";
        printf("  %s  in=[%.0f, %.0f]  target=%.0f  output=%.4f\n",
               status, (double)inputs[i * 2], (double)inputs[i * 2 + 1],
               (double)expected, (double)out[0]);
        if (err < 0.1f) passed++;
    }
    printf("Passed: %d/4\n", passed);

    save_spingalett(
        .net = nn,
        .filename = "xor_model",
        .precision = PRECISION_FLOAT32
    );

    free_network(nn);

    printf("\n--- Loading saved model ---\n");
    NeuralNetwork *loaded = load_spingalett("xor_model.slett");
    if (loaded) {
        spingalett_set_verbose(false);

        printf("\n--- Loaded model results ---\n");
        for (int i = 0; i < 4; i++) {
            float *out = forward(.net = loaded, .input = &inputs[i * 2]);
            printf("  in=[%.0f, %.0f]  output=%.4f\n",
                   (double)inputs[i * 2], (double)inputs[i * 2 + 1], (double)out[0]);
        }

        spingalett_set_verbose(true);
        free_network(loaded);
    }

    return 0;
}
