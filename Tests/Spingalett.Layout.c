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
    SIZE(NeuralNetworkArgs); FIELD(NeuralNetworkArgs, loss_func);

    SIZE(LayerArgs);
    FIELD(LayerArgs, net); FIELD(LayerArgs, neurons_amount); FIELD(LayerArgs, act_func);
    FIELD(LayerArgs, weight_initialization); FIELD(LayerArgs, dropout_rate); FIELD(LayerArgs, type);
    FIELD(LayerArgs, height); FIELD(LayerArgs, width); FIELD(LayerArgs, channels); FIELD(LayerArgs, filters);
    FIELD(LayerArgs, kernel); FIELD(LayerArgs, stride); FIELD(LayerArgs, padding);
    FIELD(LayerArgs, kernel_h); FIELD(LayerArgs, kernel_w); FIELD(LayerArgs, stride_h); FIELD(LayerArgs, stride_w);
    FIELD(LayerArgs, padding_h); FIELD(LayerArgs, padding_w); FIELD(LayerArgs, groups); FIELD(LayerArgs, epsilon);
    FIELD(LayerArgs, momentum);

    SIZE(SpingalettNetworkLayer);
    FIELD(SpingalettNetworkLayer, type); FIELD(SpingalettNetworkLayer, height); FIELD(SpingalettNetworkLayer, width);
    FIELD(SpingalettNetworkLayer, channels); FIELD(SpingalettNetworkLayer, outputs); FIELD(SpingalettNetworkLayer, activation);
    FIELD(SpingalettNetworkLayer, dropout_rate); FIELD(SpingalettNetworkLayer, kernel_h); FIELD(SpingalettNetworkLayer, kernel_w);
    FIELD(SpingalettNetworkLayer, stride_h); FIELD(SpingalettNetworkLayer, stride_w); FIELD(SpingalettNetworkLayer, padding_h);
    FIELD(SpingalettNetworkLayer, padding_w); FIELD(SpingalettNetworkLayer, weight_count); FIELD(SpingalettNetworkLayer, bias_count);
    FIELD(SpingalettNetworkLayer, groups); FIELD(SpingalettNetworkLayer, epsilon); FIELD(SpingalettNetworkLayer, momentum);

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
    FIELD(TrainArgs, callback); FIELD(TrainArgs, callback_interval); FIELD(TrainArgs, callback_data);
    FIELD(TrainArgs, lr_scheduler); FIELD(TrainArgs, lr_scheduler_data);
    FIELD(TrainArgs, val_inputs); FIELD(TrainArgs, val_targets); FIELD(TrainArgs, val_count);
    FIELD(TrainArgs, monitor); FIELD(TrainArgs, early_stopping_patience); FIELD(TrainArgs, early_stopping_min_delta);
    FIELD(TrainArgs, restore_best_weights); FIELD(TrainArgs, blas_num_threads); FIELD(TrainArgs, augment_shift);
    FIELD(TrainArgs, augment_flip);

    SIZE(EvalMetrics); FIELD(EvalMetrics, loss); FIELD(EvalMetrics, accuracy);

    SIZE(TrainProgress);
    FIELD(TrainProgress, epoch); FIELD(TrainProgress, epochs); FIELD(TrainProgress, train_loss);
    FIELD(TrainProgress, learning_rate); FIELD(TrainProgress, has_validation); FIELD(TrainProgress, validation);
    FIELD(TrainProgress, monitor); FIELD(TrainProgress, best_epoch); FIELD(TrainProgress, best_value);
    FIELD(TrainProgress, improved);

    SIZE(TrainReport);
    FIELD(TrainReport, status); FIELD(TrainReport, epochs_run); FIELD(TrainReport, train_loss);
    FIELD(TrainReport, has_validation); FIELD(TrainReport, validation); FIELD(TrainReport, monitor);
    FIELD(TrainReport, best_epoch); FIELD(TrainReport, best_value); FIELD(TrainReport, restored_best);

    SIZE(EvaluateArgs);
    FIELD(EvaluateArgs, net); FIELD(EvaluateArgs, inputs); FIELD(EvaluateArgs, targets); FIELD(EvaluateArgs, sample_count);

    SIZE(OptimizerArgs);
    FIELD(OptimizerArgs, type); FIELD(OptimizerArgs, learning_rate); FIELD(OptimizerArgs, weight_decay);
    FIELD(OptimizerArgs, momentum); FIELD(OptimizerArgs, beta1); FIELD(OptimizerArgs, beta2);
    FIELD(OptimizerArgs, epsilon); FIELD(OptimizerArgs, max_grad_norm);

    SIZE(SpingalettDataset);
    FIELD(SpingalettDataset, count); FIELD(SpingalettDataset, input_size); FIELD(SpingalettDataset, target_size);
    FIELD(SpingalettDataset, inputs); FIELD(SpingalettDataset, targets); FIELD(SpingalettDataset, height);
    FIELD(SpingalettDataset, width); FIELD(SpingalettDataset, channels); FIELD(SpingalettDataset, class_names);

    SIZE(SpingalettTargetSet);
    FIELD(SpingalettTargetSet, name); FIELD(SpingalettTargetSet, size); FIELD(SpingalettTargetSet, targets);
    FIELD(SpingalettTargetSet, class_names); FIELD(SpingalettTargetSet, encoding);

    SIZE(DatasetSaveOptions);
    FIELD(DatasetSaveOptions, input_encoding); FIELD(DatasetSaveOptions, target_encoding);
    FIELD(DatasetSaveOptions, no_compression); FIELD(DatasetSaveOptions, target_name);
    FIELD(DatasetSaveOptions, extra_targets); FIELD(DatasetSaveOptions, extra_target_count);

    SIZE(SpingalettDatasetInfo);
    FIELD(SpingalettDatasetInfo, count); FIELD(SpingalettDatasetInfo, input_size); FIELD(SpingalettDatasetInfo, target_size);
    FIELD(SpingalettDatasetInfo, input_encoding); FIELD(SpingalettDatasetInfo, target_encoding);
    FIELD(SpingalettDatasetInfo, chunk_count); FIELD(SpingalettDatasetInfo, file_size);
    FIELD(SpingalettDatasetInfo, format_version); FIELD(SpingalettDatasetInfo, height); FIELD(SpingalettDatasetInfo, width);
    FIELD(SpingalettDatasetInfo, channels); FIELD(SpingalettDatasetInfo, target_set_count);
    FIELD(SpingalettDatasetInfo, target_set);

    SIZE(DatasetReaderOptions);
    FIELD(DatasetReaderOptions, shuffle); FIELD(DatasetReaderOptions, in_memory); FIELD(DatasetReaderOptions, no_prefetch);
    FIELD(DatasetReaderOptions, target_set);

    SIZE(PredictArgs);
    FIELD(PredictArgs, net); FIELD(PredictArgs, inputs); FIELD(PredictArgs, sample_count); FIELD(PredictArgs, outputs);

    SIZE(SaveArgs);
    FIELD(SaveArgs, net); FIELD(SaveArgs, filename); FIELD(SaveArgs, do_not_save_optimizer); FIELD(SaveArgs, precision);

    SIZE(SpingalettModel);
    FIELD(SpingalettModel, input_size); FIELD(SpingalettModel, output_size); FIELD(SpingalettModel, layer_count);
    FIELD(SpingalettModel, loss); FIELD(SpingalettModel, workspace_size); FIELD(SpingalettModel, image);
    FIELD(SpingalettModel, image_size); FIELD(SpingalettModel, max_width_); FIELD(SpingalettModel, max_int_inputs_);
    FIELD(SpingalettModel, conv_scratch_); FIELD(SpingalettModel, owner_);

    SIZE(SpingalettLayerInfo);
    FIELD(SpingalettLayerInfo, type); FIELD(SpingalettLayerInfo, inputs); FIELD(SpingalettLayerInfo, outputs);
    FIELD(SpingalettLayerInfo, activation); FIELD(SpingalettLayerInfo, precision);
    FIELD(SpingalettLayerInfo, in_height); FIELD(SpingalettLayerInfo, in_width); FIELD(SpingalettLayerInfo, in_channels);
    FIELD(SpingalettLayerInfo, height); FIELD(SpingalettLayerInfo, width); FIELD(SpingalettLayerInfo, channels);
    FIELD(SpingalettLayerInfo, kernel_h); FIELD(SpingalettLayerInfo, kernel_w); FIELD(SpingalettLayerInfo, stride_h);
    FIELD(SpingalettLayerInfo, stride_w); FIELD(SpingalettLayerInfo, padding_h); FIELD(SpingalettLayerInfo, padding_w);
    FIELD(SpingalettLayerInfo, groups); FIELD(SpingalettLayerInfo, epsilon);

    SIZE(LRScheduleParams);
    FIELD(LRScheduleParams, warmup_epochs); FIELD(LRScheduleParams, step_size);
    FIELD(LRScheduleParams, gamma); FIELD(LRScheduleParams, min_lr);
    return 0;
}
