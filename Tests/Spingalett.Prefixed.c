/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/*
 * A program that uses only the prefixed names: the headers leave out the names of 0.x unless asked
 * (Spingalett.Short.h), so that a program may use them for its own (the declarations below, and
 * <syslog.h>'s LOG_DEBUG, would conflict with any that leaked), and the prefixed builders train,
 * predict, save and load a network.
 */

#if __has_include(<syslog.h>)
#include <syslog.h>
#endif
#include <Spingalett/Spingalett.h>
#include <math.h>
#include <stdio.h>

typedef int NeuralNetwork;
typedef double TrainArgs;
typedef char ActivationFunction;
enum { ACT_RELU = 101, LAYER_DENSE = 102, PRECISION_INT8 = 103, LOSS_MSE = 104, TRAIN_COMPLETED = 105 };
static int train(int x) { return x + 1; }
static int layer(int x) { return x + 2; }
static int activate(int x) { return x + 3; }
static int free_network(int x) { return x + 4; }

int main(void) {
    NeuralNetwork mine = train(layer(activate(free_network(0)))) + ACT_RELU + LAYER_DENSE + PRECISION_INT8 + LOSS_MSE +
                         TRAIN_COMPLETED;
    TrainArgs unused = 0.0;
    ActivationFunction c = 0;
    (void)unused;
    (void)c;

    spingalett_set_verbose(false);
    spingalett_seed(42);
    const float x[] = {0, 0, 0, 1, 1, 0, 1, 1}, t[] = {0, 1, 1, 0};
    SpingalettNetwork *net = spingalett_network_new(.loss_func = SPINGALETT_LOSS_MSE);
    spingalett_layer(.net = net, .neurons_amount = 2);
    spingalett_layer(.net = net, .neurons_amount = 8, .act_func = SPINGALETT_ACT_TANH,
                     .weight_initialization = SPINGALETT_INIT_XAVIER);
    spingalett_layer(.net = net, .neurons_amount = 1, .act_func = SPINGALETT_ACT_SIGMOID,
                     .weight_initialization = SPINGALETT_INIT_XAVIER);
    SpingalettTrainReport r = spingalett_train(.net = net, .inputs = x, .targets = t, .sample_count = 4, .epochs = 3000,
                                               .learning_rate = 0.05f, .optimizer_type = SPINGALETT_OPTIMIZER_ADAM,
                                               .training_strategy = SPINGALETT_STRATEGY_FULL_BATCH);
    float y[4];
    size_t size = 0;
    void *image = spingalett_save_to_memory(net, SPINGALETT_PRECISION_FLOAT32, false, &size);
    SpingalettNetwork *back = image ? spingalett_load_from_memory(image, size) : NULL;
    float z[4];
    bool ok = r.status == SPINGALETT_TRAIN_COMPLETED && spingalett_predict(.net = net, .inputs = x, .sample_count = 4,
                                                                           .outputs = y) &&
              back && spingalett_predict(.net = back, .inputs = x, .sample_count = 4, .outputs = z);
    for (int i = 0; ok && i < 4; i++) ok = fabsf(y[i] - t[i]) < 0.1f && y[i] == z[i];
    ok = ok && spingalett_activate(0.0f, SPINGALETT_ACT_SIGMOID) == 0.5f && mine == 525;
    spingalett_free(image);
    spingalett_network_free(back);
    spingalett_network_free(net);
    printf("prefixed names: %s\n", ok ? "ok" : "FAILED");
    return ok ? 0 : 1;
}
