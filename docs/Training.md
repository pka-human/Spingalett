# Training

`spingalett_train()` and its options, validation, schedules, generators, a program's own loop, checkpoints and reproducibility.

## Training

`spingalett_train()` takes a `SpingalettTrainArgs` struct. Zero-valued fields use the defaults
below.

| Field | Default | Description |
|---|---|---|
| `inputs`, `targets`, `sample_count` | required | Row-major arrays of `sample_count` samples |
| `device_inputs`, `device_targets` | none | Either or both as data sets in the GPU's memory instead (see [The GPU](GPU.md#the-gpu)) |
| `epochs` | required | Number of passes over the data |
| `training_strategy` | `SPINGALETT_STRATEGY_SAMPLE` | `SPINGALETT_STRATEGY_SAMPLE` (one step per sample), `SPINGALETT_STRATEGY_FULL_BATCH`, `SPINGALETT_STRATEGY_SMALL_BATCH` |
| `batch_size` | 32 | Mini-batch size |
| `accumulation_steps` | 0 (off) | Gradient accumulation: an optimizer step after every this many mini-batches, with the mean gradient of their samples (see below) |
| `do_not_shuffle` | false | Per-sample and mini-batch training reshuffle the samples every epoch unless set |
| `optimizer_type` | `SPINGALETT_OPTIMIZER_SGD` | `SGD`, `MOMENTUM`, `RMSPROP`, `ADAM`, `ADAMW` |
| `learning_rate` | 0.01 | Base learning rate |
| `momentum` | 0.9 | Momentum coefficient |
| `beta1`, `beta2`, `epsilon` | 0.9, 0.999, 1e-8 | Adam/RMSProp moment parameters; epsilon is added to the square root of the second moment, as in PyTorch |
| `weight_decay` | 0 | L2 penalty; decoupled (`w *= 1 - lr * wd`) for AdamW. Biases are not decayed |
| `max_grad_norm` | 0 (off) | Clip the global L2 norm of every step's gradient |
| `lr_scheduler`, `lr_scheduler_data` | none | Learning-rate schedule, see below |
| `lr_per_step` | false | Evaluate the schedule before every optimizer step instead of every epoch |
| `recompute` | false | Activation recomputation on the CPU: less memory for about one more forward pass, the same results (see below) |
| `reset_optimizer` | false | Clear moment estimates and the step counter before training (and begin a new run in place of a checkpoint's) |
| `nan_check_interval` | 0 (off) | Stop when weights become NaN/Inf, checked every N epochs |
| `report_interval` | 0 (off) | Log the training loss every N epochs |
| `callback`, `callback_interval`, `callback_data` | none, 1, NULL | `bool cb(SpingalettNetwork *, const SpingalettTrainProgress *, void *callback_data)`; return `true` to stop |
| `val_inputs`, `val_targets`, `val_count` | none | Validation set, evaluated after every epoch (`device_val_inputs`, `device_val_targets`: in the GPU's memory) |
| `monitor` | `SPINGALETT_MONITOR_AUTO` | Quantity that selects the best epoch: validation loss if there is validation data, else training loss; or `SPINGALETT_MONITOR_TRAIN_LOSS`, `SPINGALETT_MONITOR_VAL_LOSS`, `SPINGALETT_MONITOR_VAL_ACCURACY` |
| `early_stopping_patience`, `early_stopping_min_delta` | 0 (off), 0 | Stop after this many epochs without an improvement larger than `min_delta` |
| `restore_best_weights` | false | End with the weights and biases of the best epoch, kept in memory |
| `autosave_mode`, `autosave_interval`, `autosave_path` | off | Periodic saves (`SPINGALETT_AUTOSAVE_OVERWRITE` or `SPINGALETT_AUTOSAVE_NEW_FILES`, which appends `_epoch_N`): [checkpoints](#checkpoints) in FP32 with the optimizer state |
| `autosave_precision`, `autosave_do_not_save_optimizer` | FP32, false | The saves' format: another precision, or no optimizer state, saves a model |
| `blas_num_threads` | 0 (auto) | OpenBLAS threads during training, see [Backends](GPU.md#backends-and-threading) |
| `augment_shift`, `augment_flip` | 0, false | Image augmentation (the input layer has a height and width): each training sample is shifted by up to `augment_shift` cells along each axis, zeros shifted in, and with `augment_flip` mirrored left to right half of the time; drawn anew for every sample of every step, the same on any number of threads |

The reported loss is averaged over the samples of an epoch: the sum of squared errors per sample
for MSE, and categorical (softmax) or binary (sigmoid) cross-entropy otherwise. Cross-entropy
requires a softmax or sigmoid output layer; `spingalett_train()` rejects other combinations.

Gradient accumulation (`.accumulation_steps = 4` with `.batch_size = 8`) trains on steps of four
mini-batches with the memory of one: the step's samples run in chunks of a mini-batch, and its
gradient is their mean. That is the gradient of one batch of 32 samples computed in chunks of 8, bit
for bit (`spingalett_train()` computes batches larger than its workspace holds that way: a full
batch of 5,000 samples in chunks of 2,048 gives the bits of 3 x 2,048 accumulated); against a batch
of 32 computed in one chunk, only the order of the sums differs, and batch normalization takes its
statistics over each mini-batch. Dropout masks and augmentation are drawn by a sample's position in
its step, as for the larger batch. With a generator, a step is `accumulation_steps` calls of it;
full-batch training ignores the option, its step being the whole set. On the GPU the step's
mini-batches are its chunks in the same way.

Activation recomputation (checkpointing, `.recompute = true`) trades a forward pass for memory: the
workspace keeps through the backward pass the outputs of the layers several layers read (a transformer's
residual stream, one a block), of batch normalizations and of the output layer, and the other layers'
outputs, gradients and values before their activations for one segment between two kept layers at a time
(a block), computing a segment's outputs again just before its backward pass. The weights come out bit for
bit as without it. A GPT-like network of six blocks of 256 channels over 256 tokens trains in chunks of 16
samples with a peak of 141 MB instead of 216 MB (the parameters and AdamW's moments are 80 MB of either),
taking 18.2 s instead of 14.2 s on one thread and 7.8 s instead of 5.6 s on eight (i7-12650H). It applies
to mini-batch and full-batch training on the CPU; the GPU keeps every layer.

Optimizer state (moment estimates and the step counter used for Adam's bias correction) is stored
in the network, so training can be resumed by calling `spingalett_train()` again or after loading a
checkpoint that includes the optimizer state.

`spingalett_train()` returns a `SpingalettTrainReport`: `status` (`SPINGALETT_TRAIN_COMPLETED`,
`SPINGALETT_TRAIN_EARLY_STOPPED`, `SPINGALETT_TRAIN_INTERRUPTED` by the callback,
`SPINGALETT_TRAIN_DIVERGED` on NaN/Inf, `SPINGALETT_TRAIN_NO_DATA` from a generator, or
`SPINGALETT_TRAIN_FAILED` with the reason in `spingalett_last_error_message()`), `epochs_run`, the
last epoch's training loss and validation metrics, and the best epoch with its value.

## Validation and early stopping

With a validation set, `spingalett_train()` evaluates loss and accuracy after every epoch and tracks
the best epoch of the monitored quantity. Early stopping ends training once it has not improved for
`early_stopping_patience` epochs, and `restore_best_weights` resets the parameters to those of the
best epoch however training ended (completion, early stopping, the callback or divergence); the best
parameters are kept in memory, which costs one copy of the weights and biases.

```c
static bool on_epoch(NeuralNetwork *net, const TrainProgress *p, void *log) {
    fprintf(log, "epoch %zu: loss %.4f, val accuracy %.2f%%%s\n", p->epoch, p->train_loss,
            100 * p->validation.accuracy, p->improved ? " (best)" : "");
    return false;
}

TrainReport r = train(
    .net = net, .inputs = x, .targets = y, .sample_count = n, .epochs = 100,
    .training_strategy = STRATEGY_SMALL_BATCH, .optimizer_type = OPTIMIZER_ADAMW,
    .val_inputs = xv, .val_targets = yv, .val_count = nv,
    .monitor = MONITOR_VAL_ACCURACY, .early_stopping_patience = 5,
    .restore_best_weights = true, .callback = on_epoch, .callback_data = stdout);
printf("stopped after %zu epochs, kept epoch %zu\n", r.epochs_run, r.best_epoch);
```

The callback's `SpingalettTrainProgress` carries the epoch, training loss, learning rate, validation
metrics, the best epoch so far and whether this epoch improved on it. Accuracy compares the
argmax of the outputs with that of the targets; with a single output, it checks that both are on
the same side of 0.5.

## Learning-rate schedules

```c
typedef float (*LRSchedulerFn)(size_t epoch, size_t total_epochs, float initial_lr,
                                        void *user_data);
```

The scheduler runs before every epoch; `epoch` counts the epochs already completed (0 for the
first). With `.lr_per_step = true` it runs before every optimizer step instead, `epoch` counting the
steps taken in the call and `total_epochs` the steps of the call (epochs times steps an epoch; with a
generator and no `sample_count`, 0 during the first epoch, then that epoch's steps times the epochs),
as language models are trained: a warm-up of a few thousand steps, then a decay. Six schedules are
built in and take an optional `SpingalettLRScheduleParams` as `user_data` (its `warmup_epochs` and
`step_size` counting steps with `lr_per_step`):

```c
LRScheduleParams schedule = {.warmup_epochs = 10, .min_lr = 1e-5f};
train(.net = net, /* ... */ .learning_rate = 1e-3f,
      .lr_scheduler = spingalett_lr_warmup_cosine, .lr_scheduler_data = &schedule);
```

| Function | Parameters used |
|---|---|
| `spingalett_lr_cosine_decay` | `min_lr` |
| `spingalett_lr_linear_warmup` | `warmup_epochs` (default 5% of the run) |
| `spingalett_lr_step_decay` | `step_size` (default a third of the run), `gamma` (default 0.1) |
| `spingalett_lr_warmup_cosine` | `warmup_epochs`, `min_lr` |
| `spingalett_lr_linear_decay` | `min_lr` |
| `spingalett_lr_warmup_linear` | `warmup_epochs`, `min_lr` |

Reduce on plateau works with or without a schedule: with `.lr_plateau_patience = 5` and
`.lr_plateau_factor = 0.5f`, five epochs without improvement of the monitored value (`monitor`:
the validation loss when there is validation data) halve the learning rate for the rest of the run,
and five more halve it again, never below `.lr_plateau_min_lr`. Label smoothing
(`.label_smoothing = 0.1f`) trains classifiers on targets moved that far towards the uniform
distribution.

## Data generators

For datasets that do not fit in memory, or data produced on the fly, set
`.training_mode = SPINGALETT_MODE_GENERATOR_FUNCTION` and supply a generator:

```c
typedef uint32_t (*DataGeneratorFn)(float *inputs, float *targets, uint32_t requested,
                                    void *user_data);
```

The generator writes up to `requested` samples (row-major) and returns how many it wrote;
returning 0 ends the epoch. It is called once per mini-batch, once per epoch for full-batch
training (requesting `sample_count` samples, which is then required) and in chunks for per-sample
training. In this mode `sample_count` optionally caps the number of samples per epoch, which
allows endless generators. Shuffling and augmentation are the generator's responsibility.

## Custom training loops and losses

`spingalett_train()` covers the usual loops; for anything else, a `SpingalettTrainer` exposes the
steps. `spingalett_trainer_forward()` runs a training-mode forward pass (dropout active) and returns
the outputs; `spingalett_trainer_backward()` back-propagates the network's own loss, and
`spingalett_trainer_backward_output_grads()` a custom one given dL/d(output) per sample. Backward
passes add up the per-sample gradients in the network (readable with `spingalett_get_parameters()`
and `SPINGALETT_PARAM_WEIGHT_GRADIENTS` / `SPINGALETT_PARAM_BIAS_GRADIENTS`), and
`spingalett_trainer_step()` applies their mean with the given optimizer, so one step can span
several backward passes:

```c
SpingalettTrainer *tr = spingalett_trainer_new(net, 64);          /* up to 64 samples per pass */
OptimizerArgs adam = {.type = OPTIMIZER_ADAM, .learning_rate = 1e-3f};
for (size_t s = 0; s < n; s += 64) {
    const float *out = spingalett_trainer_forward(tr, x + s * in_size, 64);
    for (size_t i = 0; i < 64 * out_size; i++)                    /* e.g. a weighted MSE */
        grad[i] = weight[i % out_size] * (out[i] - y[s * out_size + i]);
    spingalett_trainer_backward_output_grads(tr, grad);
    spingalett_trainer_step(tr, &adam);
}
spingalett_trainer_free(tr);
```

`spingalett_train_on_batch()` combines forward, backward with the built-in loss and step. A loop of
it over unshuffled mini-batches reproduces `spingalett_train()` with
`SPINGALETT_STRATEGY_SMALL_BATCH`, including the dropout masks. Optimizer state and the step counter
are the network's, shared with `spingalett_train()`.

## Checkpoints

A checkpoint holds a network with its optimizer state and the state of the `spingalett_train()` run
that trained it, as its last epoch left it, so that a run stopped there (by its callback, by a crash
after an autosave) and carried on trains the same bits as one that never stopped, on any number of
threads:

```c
NeuralNetwork *net = spingalett_load_checkpoint("run.ckpt");   /* NULL: none yet */
if (!net) net = build_network();
SpingalettDatasetReader *r = spingalett_dataset_open("train.slettd", true);
train(.net = net, .epochs = 30,                                /* the run's epochs, not those left */
      .training_mode = MODE_GENERATOR_FUNCTION, .generator = spingalett_dataset_generator,
      .generator_data = r, .training_strategy = STRATEGY_SMALL_BATCH, .batch_size = 128,
      .autosave_mode = AUTOSAVE_OVERWRITE, .autosave_interval = 1, .autosave_path = "run.ckpt");
```

The run's state is what its parameters and moments do not say: the seeds of its dropout masks, of its
augmentations and of its epochs' orders (each epoch's order is drawn from the run's seed and the
epoch's number), Adam's bias corrections as the steps multiplied them, its epochs and steps (which
schedules count), reduce-on-plateau's scale, the best epoch and the patience of early stopping, with
`restore_best_weights` the best epoch's parameters, and the place of the reader it read through
`spingalett_dataset_generator()`, to which the resumed run moves its own reader. The `spingalett_train()`
after `spingalett_load_checkpoint()` is given the run's arguments again, its `.epochs` all the run's
epochs: it runs those left (none for a run that ran them all or stopped early), and its callback,
schedule and best epoch count the run's epochs. The call after it, as every call on a network made or
loaded with `spingalett_load()`, begins a run of its own; so does one with `.reset_optimizer`.
`spingalett_save_checkpoint(net, path)` writes one from a callback or after training; an autosave in
FP32 with the optimizer state is one. A checkpoint is a FLOAT32 image with a record of the run after
it, so `spingalett_load()`, the engine and the runtime read it as a model
([docs/ModelFormat.md](ModelFormat.md#checkpoints)).

## Reproducibility

`spingalett_seed(seed)` seeds the calling thread's generator, which drives weight
initialization, mini-batch shuffling and dropout. With a fixed seed, training is deterministic,
and dropout masks are identical across backends and thread counts. With the built-in kernels,
the trained weights are bit for bit the same on any number of threads and in single-threaded mode
(a test checks it); OpenBLAS agrees up to rounding.
