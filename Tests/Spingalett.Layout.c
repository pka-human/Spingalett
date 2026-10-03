/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/* Prints sizeof/offsetof of every public struct the Python bindings mirror with ctypes;
   Tests/test_python_layout.py compares them with the ctypes definitions. */

#include <Spingalett/Spingalett.h>
#include <stddef.h>
#include <stdio.h>

#define SIZE(T)     printf("%s size %zu\n", #T, sizeof(T))
#define FIELD(T, f) printf("%s.%s %zu\n", #T, #f, offsetof(T, f))

int main(void) {
    SIZE(NeuralNetwork);
    FIELD(NeuralNetwork, layers); FIELD(NeuralNetwork, topology); FIELD(NeuralNetwork, act_func);
    FIELD(NeuralNetwork, weights); FIELD(NeuralNetwork, biases); FIELD(NeuralNetwork, neurons);
    FIELD(NeuralNetwork, grad_weights); FIELD(NeuralNetwork, grad_biases);
    FIELD(NeuralNetwork, opt_m_weights); FIELD(NeuralNetwork, opt_m_biases);
    FIELD(NeuralNetwork, opt_v_weights); FIELD(NeuralNetwork, opt_v_biases);
    FIELD(NeuralNetwork, neuron_offsets); FIELD(NeuralNetwork, weight_offsets); FIELD(NeuralNetwork, bias_offsets);
    FIELD(NeuralNetwork, total_neurons); FIELD(NeuralNetwork, total_weights); FIELD(NeuralNetwork, total_biases);
    FIELD(NeuralNetwork, time_step); FIELD(NeuralNetwork, loss_func); FIELD(NeuralNetwork, dropout_rates);

    SIZE(NeuralNetworkArgs); FIELD(NeuralNetworkArgs, loss_func);

    SIZE(LayerArgs);
    FIELD(LayerArgs, net); FIELD(LayerArgs, neurons_amount); FIELD(LayerArgs, act_func);
    FIELD(LayerArgs, weight_initialization); FIELD(LayerArgs, dropout_rate);

    SIZE(ForwardArgs); FIELD(ForwardArgs, net); FIELD(ForwardArgs, input);

    SIZE(TrainArgs);
    FIELD(TrainArgs, net); FIELD(TrainArgs, training_mode); FIELD(TrainArgs, training_strategy);
    FIELD(TrainArgs, optimizer_type); FIELD(TrainArgs, inputs); FIELD(TrainArgs, targets);
    FIELD(TrainArgs, generator); FIELD(TrainArgs, generator_data); FIELD(TrainArgs, sample_count);
    FIELD(TrainArgs, batch_size); FIELD(TrainArgs, do_not_shuffle); FIELD(TrainArgs, epochs);
    FIELD(TrainArgs, learning_rate); FIELD(TrainArgs, weight_decay); FIELD(TrainArgs, momentum);
    FIELD(TrainArgs, beta1); FIELD(TrainArgs, beta2); FIELD(TrainArgs, epsilon); FIELD(TrainArgs, max_grad_norm);
    FIELD(TrainArgs, reset_optimizer); FIELD(TrainArgs, nan_check_interval); FIELD(TrainArgs, report_interval);
    FIELD(TrainArgs, autosave_mode); FIELD(TrainArgs, autosave_interval); FIELD(TrainArgs, autosave_path);
    FIELD(TrainArgs, autosave_do_not_save_optimizer); FIELD(TrainArgs, autosave_precision);
    FIELD(TrainArgs, callback); FIELD(TrainArgs, callback_interval);
    FIELD(TrainArgs, lr_scheduler); FIELD(TrainArgs, lr_scheduler_data); FIELD(TrainArgs, blas_num_threads);

    SIZE(SaveArgs);
    FIELD(SaveArgs, net); FIELD(SaveArgs, filename); FIELD(SaveArgs, do_not_save_optimizer); FIELD(SaveArgs, precision);

    SIZE(LRScheduleParams);
    FIELD(LRScheduleParams, warmup_epochs); FIELD(LRScheduleParams, step_size);
    FIELD(LRScheduleParams, gamma); FIELD(LRScheduleParams, min_lr);
    return 0;
}
