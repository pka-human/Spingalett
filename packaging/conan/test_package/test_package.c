/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/* A package's library loads, trains a little XOR network, predicts with it and saves it (xor.slett,
   for test_runtime), and names its GPU. */

#include <Spingalett/Spingalett.h>
#include <stdio.h>

int main(void) {
    static const float inputs[] = {0, 0, 0, 1, 1, 0, 1, 1};
    static const float targets[] = {0, 1, 1, 0};
    float outputs[4] = {0};

    spingalett_seed(1);
    SpingalettNetwork *net = spingalett_network_new(.loss_func = SPINGALETT_LOSS_MSE);
    spingalett_layer(.net = net, .neurons_amount = 2);
    spingalett_layer(.net = net, .neurons_amount = 8, .act_func = SPINGALETT_ACT_TANH);
    spingalett_layer(.net = net, .neurons_amount = 1, .act_func = SPINGALETT_ACT_SIGMOID);
    SpingalettTrainReport report = spingalett_train(.net = net, .inputs = inputs, .targets = targets, .sample_count = 4,
                                                    .training_strategy = SPINGALETT_STRATEGY_FULL_BATCH,
                                                    .optimizer_type = SPINGALETT_OPTIMIZER_ADAM, .epochs = 2000,
                                                    .learning_rate = 0.05f);
    bool ok = report.status == SPINGALETT_TRAIN_COMPLETED &&
              spingalett_predict(.net = net, .inputs = inputs, .sample_count = 4, .outputs = outputs) &&
              spingalett_save(.net = net, .filename = "xor.slett", .precision = SPINGALETT_PRECISION_INT8);
    const char *gpu = spingalett_gpu_device();
    printf("Spingalett %s (%s kernels, GPU: %s): XOR %.2f %.2f %.2f %.2f\n", spingalett_version(),
           spingalett_cpu_kernels(), gpu ? gpu : "none", outputs[0], outputs[1], outputs[2], outputs[3]);
    spingalett_network_free(net);
    return ok ? 0 : 1;
}
