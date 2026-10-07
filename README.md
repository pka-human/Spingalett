<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="Images/LogoDark.png">
  <source media="(prefers-color-scheme: light)" srcset="Images/LogoLight.png">
  <img alt="Spingalett" src="Images/LogoLight.png">
</picture>

[![CI](https://github.com/pka-human/Spingalett/actions/workflows/ci.yml/badge.svg)](https://github.com/pka-human/Spingalett/actions/workflows/ci.yml)
[![Standard](https://img.shields.io/badge/C-23-blue.svg?style=flat-square)](https://en.wikipedia.org/wiki/C23_(C_standard_revision))
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=flat-square)](LICENSE)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/pka-human/Spingalett)

</div>

Spingalett is a neural-network library written in C23 for training and running fully connected
and convolutional networks on the CPU. It depends only on the C standard library: batch training
and inference run as matrix-matrix products on built-in AVX-512, AVX2, NEON or portable kernels (on
x86-64, chosen for the processor at run time), with OpenMP and OpenBLAS as optional build-time
accelerators; results are the same bits on one thread and many. Networks are declared with C23
designated initializers and all parameters live in flat contiguous arrays. For deployment, a trained network becomes a read-only
model in FP32, FP16, BF16, INT8, INT4 or INT2 that runs with integer kernels where the weights are
integers, in place from memory, a compiled-in array or flash, on desktops and on microcontrollers
alike. Python bindings are included.

## Contents

- [Features](#features)
- [Building](#building)
- [Quick start](#quick-start)
- [Usage](#usage)
- [Deployment](#deployment)
- [Python bindings](#python-bindings)
- [DigitPad demo](#digitpad-demo)
- [Performance](#performance)
- [Project layout](#project-layout)
- [Status and roadmap](#status-and-roadmap)
- [Contributing](#contributing)
- [License](#license)

## Features

| Area | Supported |
|---|---|
| Layers | Fully connected, 2D convolution (any kernel, stride and padding, rectangular windows), max and average pooling; channels-last tensors; optional dropout per layer |
| Activations | Sigmoid, ReLU, Leaky ReLU, Tanh, FOO52, Softmax (output layer), None |
| Losses | Mean squared error, cross-entropy (softmax or sigmoid outputs) |
| Optimizers | SGD, Momentum, RMSProp, Adam, AdamW; L2 or decoupled weight decay |
| Training | Per-sample, full-batch and mini-batch strategies; in-memory arrays or a data generator; validation with early stopping and best-weight restore |
| Custom loops | Public forward / backward / optimizer-step API with custom losses and gradient accumulation |
| Data | `.slettd` data set files (compact, lossless by default, streamable into training), IDX (MNIST) and CSV readers, shuffling, hold-out splits |
| Regularization and stability | Dropout, weight decay, global gradient-norm clipping, NaN/Inf detection |
| Learning-rate schedules | Cosine decay, linear warm-up, step decay, warm-up + cosine, or a custom callback |
| Initialization | Uniform, Glorot (Xavier), He and LeCun normal |
| Inference | Per-sample `forward()`, batched `predict()`, `evaluate()` (loss and accuracy) |
| Deployment | Read-only models, dense and convolutional, in FP32, FP16, BF16, INT8, INT4 or INT2 with per-row scales and int8 x int8 kernels (AVX2/VNNI, SSE2, NEON, Arm DSP; INT4 and INT2 decoded in registers); run in place from memory or flash; C header export; a standalone engine for microcontrollers (one C file, no heap) |
| Backends | Built-in matrix kernels (AVX-512, AVX2/FMA, AVX, NEON, portable C; on x86-64 chosen at run time), single-threaded or OpenMP; OpenBLAS |
| Serialization | `.slett` model files in FP32, FP16, BF16, INT8, INT4 or INT2, optional optimizer state, CRC-32 checksums; to and from memory; versioned format |
| Introspection | Layer descriptions and parameter copies by layer through accessor functions (the network is an opaque handle) |
| Bindings | Python (ctypes + NumPy) |

## Building

Prebuilt libraries for Linux (x86-64 and ARM64), Windows and macOS are attached to every
[release](https://github.com/pka-human/Spingalett/releases): each archive contains the headers,
the shared library, a CMake package, `DatasetTool` and `ModelTool`. Extract one and point CMake at it with
`-DCMAKE_PREFIX_PATH=<directory>`, or compile directly with `-I<dir>/include -L<dir>/lib
-lspingalett`. The x86-64 archives come in a baseline build that runs on any x86-64 CPU and a
`-v3` build for processors with AVX2 and FMA; both pick AVX2 or AVX-512 matrix kernels at run time
when the processor has them, and the `-v3` build also compiles the rest of the library
(activations, optimizers, the inference engine) for AVX2. The Windows DLL ships with import
libraries for MinGW and MSVC. Every package is built with OpenMP; the macOS one (a universal
library for Apple silicon and Intel, macOS 11 or newer) carries LLVM's OpenMP runtime,
`libomp.dylib`, next to the library, since Apple's compilers come without one.

To build from source: requirements: CMake 3.21 or newer and a compiler with C23 support. GCC 13 and Clang 18 are tested
in CI; MSVC 19.36 or newer is expected to work but is not tested. OpenMP and OpenBLAS are
optional.

```bash
# Dependency-free build
cmake -S . -B Build -DCMAKE_BUILD_TYPE=Release
cmake --build Build --parallel

# With OpenMP and OpenBLAS (Debian/Ubuntu: apt install libopenblas-dev)
cmake -S . -B Build -DCMAKE_BUILD_TYPE=Release -DBUILD_WITH_OPENMP=ON -DBUILD_WITH_OPENBLAS=ON
cmake --build Build --parallel

# Run the test suite
ctest --test-dir Build --output-on-failure

# Install headers and the shared library
cmake --install Build --prefix /usr/local
```

The shared library and the example programs are written to `Bin/`, import libraries to `Lib/`.
By default the library is compiled with `-march=native` and therefore tuned for the build
machine; for redistributable binaries, configure with `-DSPINGALETT_NATIVE_ARCH=OFF` and choose
a baseline through `CMAKE_C_FLAGS` (for example `-march=x86-64-v3` for AVX2). On x86-64 with GCC
or Clang, such a build also compiles the matrix kernels for AVX2 and AVX-512 and runs the best set
the processor supports (`spingalett_cpu_kernels()` names it).

| CMake option | Default | Description |
|---|---|---|
| `BUILD_WITH_OPENMP` | `OFF` | Enable the OpenMP backend |
| `BUILD_WITH_OPENBLAS` | `OFF` | Enable the OpenBLAS backend (found via pkg-config or the default library paths) |
| `BUILD_EXAMPLE` | `ON` | Build the programs in `Examples/` |
| `BUILD_TESTS` | `ON` | Build the test suite and register it with CTest |
| `BUILD_APPS` | `OFF` | Build the DigitPad demo (needs SDL2) and its trainer |
| `SPINGALETT_INFERENCE_ONLY` | `OFF` | Build only the inference engine (`Spingalett.Inference.h`) as a static library: no training, file I/O, OpenMP or heap |
| `SPINGALETT_NATIVE_ARCH` | `ON` | Compile with `-march=native`; turn off for binaries that must run on other machines |
| `SPINGALETT_CPU_DISPATCH` | `ON` | Without `-march=native` (GCC, Clang): on x86-64 also build AVX2 and AVX-512 matrix kernels and AVX-512 VNNI and AVX-VNNI integer kernels, on AArch64 Linux integer kernels for the dot product instructions, and choose at run time |
| `SPINGALETT_BIN_DIR` | `<source>/Bin` | Output directory for executables and shared libraries |
| `SPINGALETT_LIB_DIR` | `<source>/Lib` | Output directory for static and import libraries |

An installed Spingalett is found with CMake's `find_package`; projects that vendor the
repository can use `add_subdirectory()` instead. Both provide the target `Spingalett::spingalett`,
which carries the include paths:

```cmake
find_package(Spingalett 0.7 REQUIRED)        # or: add_subdirectory(external/Spingalett)
target_link_libraries(my_app PRIVATE Spingalett::spingalett)
```

Programs that compile the library's sources into themselves, or link it statically, define
`SPINGALETT_STATIC` so that the headers do not declare the API as imported from a DLL.

The headers define `SPINGALETT_VERSION_MAJOR`, `_MINOR`, `_PATCH` and `_STRING`;
`spingalett_version()` returns the version of the library actually loaded.

## Quick start

```c
#include <Spingalett/Spingalett.h>
#include <stdio.h>

int main(void) {
    float inputs[4][2]  = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    float targets[4][1] = {{0}, {1}, {1}, {0}};

    NeuralNetwork *net = new_spingalett(LOSS_MSE);
    layer(net, 2);                                              /* input layer */
    layer(net, 8, ACT_TANH, WEIGHT_INITIALIZATION_XAVIER);
    layer(net, 1, ACT_SIGMOID, WEIGHT_INITIALIZATION_XAVIER);

    train(
        .net = net,
        .inputs = &inputs[0][0],
        .targets = &targets[0][0],
        .sample_count = 4,
        .epochs = 5000,
        .training_strategy = STRATEGY_FULL_BATCH,
        .optimizer_type = OPTIMIZER_ADAM,
        .learning_rate = 0.02f,
        .report_interval = 1000
    );

    for (int i = 0; i < 4; i++) {
        float *out = forward(net, inputs[i]);
        printf("%.0f XOR %.0f = %.4f\n", inputs[i][0], inputs[i][1], out[0]);
    }

    save_spingalett(.net = net, .filename = "xor");             /* writes xor.slett */
    free_network(net);
    return 0;
}
```

Every function that takes many parameters is a macro over a struct, so arguments can be given
positionally (`layer(net, 8, ACT_TANH)`) or by name (`layer(.net = net, .neurons_amount = 8)`).
Fields that are not mentioned are zero, which selects the documented default.

The `Examples/` directory contains this XOR program, an MNIST classifier
(`Examples/download_mnist.sh data/mnist && Bin/MNIST data/mnist`), a convolutional one
(`Bin/MNIST_CNN data/mnist`, about 99% after two epochs) and the benchmark described under
[Performance](#performance).

## Usage

### Defining a network

`new_spingalett(loss)` creates an empty network; each `layer()` call appends a layer, the first
one being the input layer. `LayerArgs` fields:

| Field | Meaning |
|---|---|
| `neurons_amount` | Layer width |
| `act_func` | Activation of this layer (ignored for the input layer) |
| `weight_initialization` | `RANDOM` (uniform in [-1, 1]), `XAVIER` (Glorot normal, variance 2/(fan_in + fan_out)), `HE` (normal, variance 2/fan_in), `LECUN` (normal, variance 1/fan_in), `NONE` (zeros) |
| `dropout_rate` | Probability in [0, 1) of zeroing each output of this layer during training |

Biases start at zero. Softmax is only allowed in the output layer. Leaky ReLU uses a slope of
0.01; FOO52 is the piecewise-linear function `0.01x` for `x < 0`, `x` on `[0, 1]` and
`1 + 0.01(x - 1)` above 1.

Dropout is inverted (kept units are scaled by `1 / (1 - p)`), so inference needs no rescaling.
It is not applied to the input or output layer.

### Convolution and pooling

Data flows through a network as one tensor per sample, `height x width x channels` floats in
channels-last order: element `(y, x, c)` is at `(y * width + x) * channels + c`, so an image stored
row by row with interleaved channels goes in as it is. Give the input layer a shape, then add
layers with `conv2d()`, `max_pool2d()` and `avg_pool2d()`; a dense layer reads whatever precedes it
as a flat vector, so no flattening layer is needed:

```c
NeuralNetwork *net = new_spingalett(LOSS_CROSS_ENTROPY);
layer(.net = net, .height = 28, .width = 28, .channels = 1);                      /* input */
conv2d(.net = net, .filters = 32, .kernel = 3, .padding = 1, .act_func = ACT_RELU,
       .weight_initialization = WEIGHT_INITIALIZATION_HE);                          /* 28 x 28 x 32 */
max_pool2d(.net = net, .kernel = 2);                                                /* 14 x 14 x 32 */
conv2d(.net = net, .filters = 64, .kernel = 3, .padding = 1, .act_func = ACT_RELU,
       .weight_initialization = WEIGHT_INITIALIZATION_HE);                          /* 14 x 14 x 64 */
max_pool2d(.net = net, .kernel = 2);                                                /* 7 x 7 x 64 */
layer(net, 128, ACT_RELU, WEIGHT_INITIALIZATION_HE, .dropout_rate = 0.3f);
layer(net, 10, ACT_SOFTMAX, WEIGHT_INITIALIZATION_XAVIER);
```

| Field | Meaning |
|---|---|
| `height`, `width`, `channels` | Shape of the input layer (its `neurons_amount` may be left 0) |
| `filters` | Output channels of a convolution |
| `kernel`, `kernel_h`, `kernel_w` | Window size; `kernel` sets both, `kernel_h` / `kernel_w` override one axis |
| `stride`, `stride_h`, `stride_w` | Window step: 1 for convolutions and the kernel size for pooling unless set |
| `padding`, `padding_h`, `padding_w` | Zero cells added on both sides of each axis; smaller than the kernel |

An output axis has `(size + 2 padding - kernel) / stride + 1` cells. A convolution's filters are
`kernel_h x kernel_w x input channels` weights each, plus one bias, and take an activation, a weight
initialization (fan-in is the window size) and dropout like dense layers. Pooling has no parameters
and no activation; windows are clipped to the input, so padding cells never count (average pooling
divides by the cells inside), and max pooling passes the gradient to the first maximum. Every
training strategy, optimizer, schedule, the custom training loop and `predict()` work with these
layers as with dense ones.

Convolutions run as matrix products whose input windows are gathered by the matrix kernels
themselves (implicit im2col: no window matrix is stored), with the bias and activation applied to
each tile of the result while it is in cache; weight gradients split their long summation over all
pixels into fixed slots that the threads share. On a single core, `Examples/MNIST_CNN.c` trains 1.5
times as fast as PyTorch and infers 4 times as fast (see [Performance](#performance)).

### Inspecting a network

`NeuralNetwork` is an opaque handle: its layout is private, and accessors describe it.

```c
uint32_t layers = spingalett_layer_count(net);                       /* the input layer included */
SpingalettNetworkLayer d;
spingalett_network_layer(net, 1, &d);   /* type, shape, outputs, activation, dropout, window, parameter counts */

float *w = malloc(d.weight_count * sizeof(float));
spingalett_get_parameters(net, 1, PARAM_WEIGHTS, w, d.weight_count); /* weights feeding layer 1 */
spingalett_set_parameters(net, 1, PARAM_WEIGHTS, w, d.weight_count);
```

Parameters come as `PARAM_WEIGHTS`, `PARAM_BIASES`, `PARAM_WEIGHT_GRADIENTS` and
`PARAM_BIAS_GRADIENTS`. Dense weights are `outputs x inputs` (row `j` holds the weights into unit
`j`); convolution weights are `filters x kernel_h x kernel_w x input channels`. Also available:
`spingalett_input_size()`, `spingalett_output_size()`, `spingalett_parameter_count()`,
`spingalett_network_loss()` and `spingalett_optimizer_steps()`.

### Training

`train()` takes a `TrainArgs` struct. Zero-valued fields use the defaults below.

| Field | Default | Description |
|---|---|---|
| `inputs`, `targets`, `sample_count` | required | Row-major arrays of `sample_count` samples |
| `epochs` | required | Number of passes over the data |
| `training_strategy` | `STRATEGY_SAMPLE` | `STRATEGY_SAMPLE` (one step per sample), `STRATEGY_FULL_BATCH`, `STRATEGY_SMALL_BATCH` |
| `batch_size` | 32 | Mini-batch size |
| `do_not_shuffle` | false | Per-sample and mini-batch training reshuffle the samples every epoch unless set |
| `optimizer_type` | `OPTIMIZER_SGD` | `SGD`, `MOMENTUM`, `RMSPROP`, `ADAM`, `ADAMW` |
| `learning_rate` | 0.01 | Base learning rate |
| `momentum` | 0.9 | Momentum coefficient |
| `beta1`, `beta2`, `epsilon` | 0.9, 0.999, 1e-8 | Adam/RMSProp moment parameters; epsilon is added to the square root of the second moment, as in PyTorch |
| `weight_decay` | 0 | L2 penalty; decoupled (`w *= 1 - lr * wd`) for AdamW. Biases are not decayed |
| `max_grad_norm` | 0 (off) | Clip the global L2 norm of every step's gradient |
| `lr_scheduler`, `lr_scheduler_data` | none | Learning-rate schedule, see below |
| `reset_optimizer` | false | Clear moment estimates and the step counter before training |
| `nan_check_interval` | 0 (off) | Stop when weights become NaN/Inf, checked every N epochs |
| `report_interval` | 0 (off) | Log the training loss every N epochs |
| `callback`, `callback_interval`, `callback_data` | none, 1, NULL | `bool cb(NeuralNetwork *, const TrainProgress *, void *callback_data)`; return `true` to stop |
| `val_inputs`, `val_targets`, `val_count` | none | Validation set, evaluated after every epoch |
| `monitor` | `MONITOR_AUTO` | Quantity that selects the best epoch: validation loss if there is validation data, else training loss; or `MONITOR_TRAIN_LOSS`, `MONITOR_VAL_LOSS`, `MONITOR_VAL_ACCURACY` |
| `early_stopping_patience`, `early_stopping_min_delta` | 0 (off), 0 | Stop after this many epochs without an improvement larger than `min_delta` |
| `restore_best_weights` | false | End with the weights and biases of the best epoch, kept in memory |
| `autosave_mode`, `autosave_interval`, `autosave_path` | off | Periodic checkpoints (`AUTOSAVE_OVERWRITE` or `AUTOSAVE_NEW_FILES`, which appends `_epoch_N`) |
| `autosave_precision`, `autosave_do_not_save_optimizer` | FP32, false | Checkpoint format |
| `blas_num_threads` | 0 (auto) | OpenBLAS threads during training, see [Backends](#backends-and-threading) |

The reported loss is averaged over the samples of an epoch: the sum of squared errors per sample
for MSE, and categorical (softmax) or binary (sigmoid) cross-entropy otherwise. Cross-entropy
requires a softmax or sigmoid output layer; `train()` rejects other combinations.

Optimizer state (moment estimates and the step counter used for Adam's bias correction) is stored
in the network, so training can be resumed by calling `train()` again or after loading a
checkpoint that includes the optimizer state.

`train()` returns a `TrainReport`: `status` (`TRAIN_COMPLETED`, `TRAIN_EARLY_STOPPED`,
`TRAIN_INTERRUPTED` by the callback, `TRAIN_DIVERGED` on NaN/Inf, `TRAIN_NO_DATA` from a
generator, or `TRAIN_FAILED` with the reason in `spingalett_last_error_message()`), `epochs_run`,
the last epoch's training loss and validation metrics, and the best epoch with its value.

### Validation and early stopping

With a validation set, `train()` evaluates loss and accuracy after every epoch and tracks the best
epoch of the monitored quantity. Early stopping ends training once it has not improved for
`early_stopping_patience` epochs, and `restore_best_weights` resets the parameters to those of the
best epoch however training ended (completion, early stopping, the callback or divergence); the
best parameters are kept in memory, which costs one copy of the weights and biases.

```c
static bool on_epoch(NeuralNetwork *net, const TrainProgress *p, void *log) {
    fprintf(log, "epoch %zu: loss %.4f, val accuracy %.2f%%%s\n", p->epoch, p->train_loss,
            100 * p->validation.accuracy, p->improved ? " (best)" : "");
    return false;
}

TrainReport r = train(.net = net, .inputs = x, .targets = y, .sample_count = n, .epochs = 100,
                      .training_strategy = STRATEGY_SMALL_BATCH, .optimizer_type = OPTIMIZER_ADAMW,
                      .val_inputs = xv, .val_targets = yv, .val_count = nv,
                      .monitor = MONITOR_VAL_ACCURACY, .early_stopping_patience = 5,
                      .restore_best_weights = true, .callback = on_epoch, .callback_data = stdout);
printf("stopped after %zu epochs, kept epoch %zu\n", r.epochs_run, r.best_epoch);
```

The callback's `TrainProgress` carries the epoch, training loss, learning rate, validation
metrics, the best epoch so far and whether this epoch improved on it. Accuracy compares the
argmax of the outputs with that of the targets; with a single output, it checks that both are on
the same side of 0.5.
### Learning-rate schedules

```c
typedef float (*LRSchedulerFn)(size_t epoch, size_t total_epochs, float initial_lr, void *user_data);
```

The scheduler runs before every epoch; `epoch` counts the epochs already completed (0 for the
first). Four schedules are built in and take an optional `LRScheduleParams` as `user_data`:

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

### Data generators

For datasets that do not fit in memory, or data produced on the fly, set
`.training_mode = MODE_GENERATOR_FUNCTION` and supply a generator:

```c
typedef uint32_t (*DataGeneratorFn)(float *inputs, float *targets, uint32_t requested, void *user_data);
```

The generator writes up to `requested` samples (row-major) and returns how many it wrote;
returning 0 ends the epoch. It is called once per mini-batch, once per epoch for full-batch
training (requesting `sample_count` samples, which is then required) and in chunks for per-sample
training. In this mode `sample_count` optionally caps the number of samples per epoch, which
allows endless generators. Shuffling and augmentation are the generator's responsibility.

### Custom training loops and losses

`train()` covers the usual loops; for anything else, a `SpingalettTrainer` exposes the steps.
`spingalett_trainer_forward()` runs a training-mode forward pass (dropout active) and returns the
outputs; `spingalett_trainer_backward()` back-propagates the network's own loss, and
`spingalett_trainer_backward_output_grads()` a custom one given dL/d(output) per sample. Backward
passes add up the per-sample gradients in the network (readable with `spingalett_get_parameters()`
and `PARAM_WEIGHT_GRADIENTS` / `PARAM_BIAS_GRADIENTS`), and
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

`spingalett_train_on_batch()` combines forward, backward with the built-in loss and step. A loop
of it over unshuffled mini-batches reproduces `train()` with `STRATEGY_SMALL_BATCH`, including the
dropout masks. Optimizer state and the step counter are the network's, shared with `train()`.

### Data sets

```c
SpingalettDataset train_set, val_set;
spingalett_load_idx("train-images-idx3-ubyte", "train-labels-idx1-ubyte", 10, &train_set);
spingalett_dataset_shuffle(&train_set);                 /* optional, uses spingalett_seed() */
spingalett_dataset_split(&train_set, 5000, &val_set);   /* last 5,000 samples -> val_set */
/* ... train(.inputs = train_set.inputs, .targets = train_set.targets, .sample_count = train_set.count, ...) */
spingalett_dataset_free(&train_set);
spingalett_dataset_free(&val_set);
```

`spingalett_load_idx()` reads the IDX format used by MNIST: unsigned-byte samples of any shape
are flattened and scaled to [0, 1] (float and double files are read unchanged), and labels are
one-hot encoded. `spingalett_load_csv(path, target_columns, num_classes, &d)` reads numeric,
comma-separated files; a non-numeric first line is skipped as a header, the last `target_columns`
columns are the targets, and with `num_classes > 0` a single label column is one-hot encoded.

### Data set files

`.slettd` is Spingalett's own data set format: binary, compact and loaded straight into a
`SpingalettDataset`. By default every stream is stored in the smallest encoding that keeps all
values exact (8-bit `q / 255` for image data, IEEE half, or float32; one-hot targets as class
indices) and compressed with an adaptive context-model range coder that learns which earlier
values predict the next, such as the pixel above in an image. Lossy FP16, BF16 and per-feature
8-bit encodings are available on request. Files consist of independently decodable chunks with
CRC-32 checksums, so they load in parallel with OpenMP and can be streamed into `train()` with only
one chunk in memory:

```c
spingalett_save_dataset(&train_set, "mnist-train", NULL);         /* writes mnist-train.slettd */
SpingalettDataset d;
spingalett_load_dataset("mnist-train.slettd", &d);                 /* bit-identical to train_set */

SpingalettDatasetReader *r = spingalett_dataset_open("mnist-train.slettd", true);   /* shuffled */
train(.net = net, .training_mode = MODE_GENERATOR_FUNCTION, .generator = spingalett_dataset_generator,
      .generator_data = r, .epochs = 10, .training_strategy = STRATEGY_SMALL_BATCH, .batch_size = 128);
spingalett_dataset_close(r);
```

| MNIST training set (60,000 images and labels) | Size |
|---|---:|
| float32 in memory | 190.6 MB |
| IDX files | 47.1 MB |
| images only, `gzip -9` / `xz -9` | 9.7 / 7.9 MB |
| `.slettd` (lossless) | 7.8 MB |

`spingalett_load_dataset_from_memory()` reads a file image already in memory, and
`Bin/DatasetTool` converts IDX and CSV files (`DatasetTool idx <images> <labels> out.slettd`) and
prints a file's layout (`DatasetTool info file.slettd`). The format is specified in
[docs/DatasetFormat.md](docs/DatasetFormat.md).

### Inference

`forward(net, input)` evaluates one sample and returns a pointer to the output layer inside the
network; the buffer is overwritten by the next call. `predict()` evaluates many samples at once
with matrix-matrix products and is the faster choice for more than a handful of inputs:

```c
float *outputs = malloc(count * output_size * sizeof(float));
predict(.net = net, .inputs = inputs, .sample_count = count, .outputs = outputs);   /* false on error */
```

`evaluate()` returns the mean loss (as reported by training) and the accuracy over a data set:

```c
EvalMetrics m = evaluate(.net = net, .inputs = x, .targets = y, .sample_count = n);
```

Dropout is not applied during inference.

### Saving and loading

```c
save_spingalett(.net = net, .filename = "model.slett", .precision = PRECISION_FP16, .do_not_save_optimizer = true);
NeuralNetwork *net = load_spingalett("model.slett");
```

Models are stored in `.slett` files; the extension is appended when the filename has none, and
files written under the former `.nn` name load unchanged. Weights are stored in the selected
precision; biases and, unless disabled, the optimizer state in float:

| Precision | Storage per weight | Notes |
|---|---|---|
| `PRECISION_FLOAT32` | 4 bytes | Lossless |
| `PRECISION_FP16`, `PRECISION_BFLOAT16` | 2 bytes | Round to nearest even |
| `PRECISION_INT8`, `PRECISION_INT4` | 1 byte, 4 bits | Symmetric, one scale per weight row (output unit) |
| `PRECISION_INT2` | 2 bits | Ternary {-scale, 0, +scale}, one scale per row |

The same bytes can be produced and read in memory:

```c
size_t size;
void *image = spingalett_save_to_memory(net, PRECISION_INT8, false, &size);   /* free with spingalett_free */
NeuralNetwork *copy = load_spingalett_from_memory(image, size);
```

The file format is versioned and specified in [docs/ModelFormat.md](docs/ModelFormat.md):
little-endian, with 16-byte aligned sections and CRC-32 checksums, so that a file image can be
executed in place (see [Deployment](#deployment)). Networks with convolution or pooling layers are
saved in version 4 (`SPINGALETT_FORMAT_VERSION`), which adds the layers' kinds and shapes; networks
of dense layers only are still saved in version 3, so that the engines of earlier releases run them.
Files of versions 1 and 2 remain loadable. `load_spingalett()` returns `NULL` for missing, truncated or
corrupt files and for files with an invalid header.

### Backends and threading

```c
spingalett_set_compute_mode(COMPUTE_OPENBLAS);   /* COMPUTE_SINGLE_THREADED, COMPUTE_OPENMP, COMPUTE_OPENBLAS */
spingalett_set_num_threads(8);                   /* 0 = runtime default */
```

Full-batch and mini-batch training and `predict()` process samples in chunks of up to 2048 as
matrix-matrix products (fewer when a sample's activations are large, so that a chunk stays near
64 MB); per-sample training and `forward()` of dense networks use matrix-vector kernels.

- `COMPUTE_SINGLE_THREADED` uses the built-in kernels: AVX-512 or AVX/FMA when the compiler
  targets them, portable C otherwise.
- `COMPUTE_OPENMP` uses the same kernels on all threads; operations that are too small to benefit
  run serially. Threads split matrix products by tiles of the result (and long weight-gradient
  sums by slots fixed by the shape), so results are the same bits on any number of threads and in
  single-threaded mode.
- `COMPUTE_OPENBLAS` delegates matrix products to OpenBLAS. It is usually on par with OpenMP for
  large batches and slower for small ones. During training, `blas_num_threads = 0` uses a single
  OpenBLAS thread when each call is too small to amortize threading and the configured thread
  count otherwise, and restores the caller's setting afterwards; outside `train()`, OpenBLAS uses
  its own thread configuration (`OPENBLAS_NUM_THREADS`).
- A requested backend that was not compiled in falls back to single-threaded with a one-time
  warning. `COMPUTE_CUDA` is reserved and currently falls back as well.

Results agree across backends up to floating-point rounding. For the duration of `train()`,
denormal floats are flushed to zero on the calling thread and the OpenMP workers; the previous
floating-point mode is restored afterwards.

A network must not be used by several threads at the same time. Compute mode, thread count and
logging settings are process-wide; the random generator and the error state are per thread.

### Reproducibility

`spingalett_seed(seed)` seeds the calling thread's generator, which drives weight
initialization, mini-batch shuffling and dropout. With a fixed seed, training is deterministic,
and dropout masks are identical across backends and thread counts. With the built-in kernels,
the trained weights are bit for bit the same on any number of threads and in single-threaded mode
(a test checks it); OpenBLAS agrees up to rounding.

### Errors and logging

Functions report failures through a thread-local error state instead of return codes:

```c
spingalett_clear_error();
NeuralNetwork *net = load_spingalett("model.slett");
if (!net)
    fprintf(stderr, "%s (code %d)\n", spingalett_last_error_message(), spingalett_last_error_code());
```

| Code | Meaning |
|---|---|
| `SPINGALETT_OK` | No error |
| `SPINGALETT_ERR_ALLOC` | Out of memory |
| `SPINGALETT_ERR_INVALID` | Invalid argument or file contents |
| `SPINGALETT_ERR_FILE_IO` | A file could not be opened, read or written, or is truncated |
| `SPINGALETT_ERR_FORMAT_VERSION` | The model file uses an unsupported format version |

The error state is not cleared by successful calls.

Log messages go to stdout (warnings and errors to stderr) unless redirected with
`spingalett_set_log_callback(void (*)(LogLevel, const char *))`. `spingalett_set_log_level()` sets
the minimum level and `spingalett_set_verbose(false)` suppresses everything below warnings.

## Deployment

A `NeuralNetwork` is built for training: float parameters, gradients and optimizer state. To run a
trained network, turn it into a model, a read-only network that keeps its weights in the precision
they are stored in and computes with them:

```c
SpingalettModel *model = spingalett_model_from_network(net, PRECISION_INT8);   /* or spingalett_model_load("model.slett") */
spingalett_model_predict(model, inputs, count, outputs);                         /* batched, multi-threaded */
EvalMetrics m = spingalett_model_evaluate(model, inputs, targets, count);
spingalett_model_free(model);
```

INT8, INT4 and INT2 layers run in integer arithmetic: each layer quantizes its input to 8 bits per
sample (the largest magnitude maps to 127), multiplies it with the weights in 32-bit integers
(AVX2 with VNNI where available, SSE2, NEON with the dot-product extension where available, the Arm
DSP extension, or portable C) and rescales each output by its weight row's scale. With AVX2 and
NEON, INT4 and INT2 codes are decoded in registers, so they run about as fast as INT8 from a half or
a quarter of the memory. FP32, FP16 and
BF16 layers compute in float, converting the weights as they read them. A convolution computes each
output pixel as its filters' dot products with the window it reads (short windows, such as a first
layer over one or three channels, accumulate all filters at once); pooling runs in float. Batched
prediction of integer models computes exactly what single runs compute, on every backend and
platform.

Accuracy on the 10,000 MNIST test images (`ModelTool eval`):

| Network | FP32 | FP16 | INT8 | INT4 | INT2 |
|---|---:|---:|---:|---:|---:|
| 784-256-128-10, `Examples/MNIST.c` (5 epochs) | 97.86% | 97.86% | 97.87% | 97.71% | 95.15% |
| Model size | 941 KB | 471 KB | 238 KB | 121 KB | 62 KB |
| 784-1024-512-10, DigitPad | 99.27% | 99.27% | 99.27% | 99.32% | 79.93% |
| CNN of `Examples/MNIST_CNN.c` (2 epochs) | 98.89% | 98.89% | 98.89% | 98.75% | 97.74% |
| Model size | 1.69 MB | 844 KB | 424 KB | 213 KB | 108 KB |

INT8 and INT4 keep the accuracy of these networks; INT2 (ternary weights without
quantization-aware training) suits small layers. One sample through the 784-512-1000-10 benchmark
network on one thread (`Bin/Benchmark`, Intel Xeon @ 2.80 GHz):

| | `forward()` | FP32 model | FP16 | BF16 | INT8 | INT4 | INT2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Microseconds per sample | 125 | 120 | 74 | 78 | 21 | 20 | 20 |
| Weights | 3.7 MB | 3.7 MB | 1.9 MB | 1.9 MB | 0.9 MB | 0.5 MB | 0.2 MB |

### Running in place

A model is a view of a `.slett` image (format version 3, or 4 with convolutions).
`spingalett_model_init()` checks an image (header, shapes, bounds, CRC-32) and copies nothing; `spingalett_model_run()` evaluates one sample with a
workspace from the caller and allocates nothing either:

```c
SpingalettModel model;                                    /* a plain struct: no release needed */
if (spingalett_model_init(&model, image, size) != SPINGALETT_OK) { /* damaged or wrong image */ }
float *workspace = malloc(model.workspace_size);          /* one per thread */
spingalett_model_run(&model, input, output, workspace);
```

The image can come from `spingalett_save_to_memory()`, a file read into memory, an array compiled
into the program or memory-mapped flash; it must stay valid while the model is used. These two
functions and `spingalett_model_layer()` make up the inference engine, declared in
`Spingalett.Inference.h` (included by `Spingalett.h`).

### Microcontrollers and C headers

```bash
Bin/ModelTool header model.slett model.h my_model --precision int8    # or spingalett_export_c_header()
```

writes the image as a 16-byte aligned `static const uint8_t my_model[]` with the macros
`MY_MODEL_SIZE`, `MY_MODEL_INPUTS`, `MY_MODEL_OUTPUTS` and `MY_MODEL_WORKSPACE`. The engine itself,
`Src/Spingalett.Inference.c` with `Include/Spingalett/Spingalett.Inference.h` and
`Src/Spingalett.Engine.h`, compiles on its own with `-DSPINGALETT_INFERENCE_ONLY`: no training
code, no file I/O, no OpenMP, no heap and no global state, about 12 KB of code on a Cortex-M4
with convolutions and pooling. CMake builds it alone with `-DSPINGALETT_INFERENCE_ONLY=ON`.
[Examples/Embedded](Examples/Embedded) runs the MNIST models on a Cortex-M4F in QEMU with the
weights in flash: the 784-256-128-10 network with 2.8 KB of RAM, the CNN (INT8, 424 KB of flash)
with 208 KB.

### ModelTool

`Examples/ModelTool.c`, installed with the library:

```
ModelTool info model.slett                              layers, precisions, sizes, workspace
ModelTool convert in.slett out.slett --precision int8   any format version, as format 3 or 4
ModelTool header model.slett model.h name [--precision P]
ModelTool eval model.slett data.slettd                  accuracy and loss in every precision
ModelTool eval model.slett images labels                (an IDX pair, such as MNIST)
ModelTool bench model.slett                             latency and throughput in every precision
```

## Python bindings

`Bindings/Python` contains pure-Python bindings built on `ctypes` and NumPy; nothing is compiled
at install time.

```bash
pip install ./Bindings/Python
export SPINGALETT_LIBRARY=$PWD/Bin/libspingalett.so
```

```python
import numpy as np
import spingalett as sg

x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y = np.array([[0], [1], [1], [0]], dtype=np.float32)

with sg.Network(sg.Loss.MSE, [sg.Layer(2),
                              sg.Layer(16, sg.Activation.TANH, sg.Init.XAVIER),
                              sg.Layer(1, sg.Activation.SIGMOID, sg.Init.XAVIER)]) as net:
    net.train(x, y, epochs=3000, optimizer=sg.Optimizer.ADAM, learning_rate=0.02,
              lr_scheduler=sg.CosineDecay())
    print(net.forward(x))
    with net.to_model(sg.Precision.INT8) as model:       # deployment model, integer kernels
        print(model.predict(x), model.size, "bytes")

cnn = sg.Network(sg.Loss.CROSS_ENTROPY, [sg.Input(28, 28, 1), sg.Conv2D(32, 3, padding=1), sg.MaxPool2D(2),
                                         sg.Layer(10, sg.Activation.SOFTMAX, sg.Init.XAVIER)])
print(cnn.layers[1].shape, cnn.get_weights(0).shape)    # (28, 28, 32) (32, 3, 3, 1)
```

See [Bindings/Python/README.md](Bindings/Python/README.md) for the full API.

## DigitPad demo

[Apps/DigitPad](Apps/DigitPad) is a desktop app in which you draw a digit with the mouse and a
Spingalett network classifies it as you draw. Its 784-1024-512-10 model, trained with on-the-fly
augmentation through a data generator, reaches 99.27% MNIST test accuracy. The directory contains
the app, the trainer and scripts that package the app and the model as a self-contained Linux
AppImage and as a Windows zip; every release attaches both.

![DigitPad](Apps/DigitPad/screenshot.png)

## Performance

`Examples/Benchmark.c` measures a 784-512-1000-10 network (925K parameters, ReLU, softmax with
cross-entropy, Adam) on 20,000 synthetic samples: full-batch training (5 epochs), mini-batch
training (batches of 64, one epoch) and batched inference; then the convolutional network of
`Examples/MNIST_CNN.c` (422K parameters) on 10,000 synthetic images: one epoch of mini-batches of
128 with AdamW, and inference. `Examples/benchmark_pytorch.py` runs the same workloads in PyTorch.
Run them with `Bin/Benchmark [threads]` and `python Examples/benchmark_pytorch.py [threads]`.

Samples per second on a 4-vCPU cloud VM (Intel Xeon @ 2.80 GHz, Cascade Lake, AVX-512), medians of
three interleaved runs (the VM is shared and single runs vary by up to 20%). Spingalett 0.7 is
built with GCC 13 and uses its built-in kernels (no BLAS library); PyTorch 2.14.1 is the CPU build
from PyPI (Intel MKL and oneDNN):

| Fully connected network | Threads | Full batch | Mini-batch 64 | Inference |
|---|---:|---:|---:|---:|
| Spingalett | 1 | 18,100 | 11,600 | 52,700 |
| PyTorch | 1 | 14,700 | 5,200 | 40,100 |
| Spingalett (OpenMP) | 4 | 56,700 | 29,300 | 172,700 |
| PyTorch | 4 | 57,300 | 9,200 | 146,900 |

| Convolutional network | Threads | Training | Inference |
|---|---:|---:|---:|
| Spingalett | 1 | 1,720 | 5,550 |
| PyTorch | 1 | 1,100 | 1,390 |
| Spingalett (OpenMP) | 4 | 5,500 | 17,900 |
| PyTorch | 4 | 3,060 | 4,440 |

Spingalett trains the convolutional network 1.6 to 1.8 times as fast and runs it 4 times as fast.
For the fully connected network it trains mini-batches 2.3 to 3.2 times as fast, infers 1.2 to 1.3
times as fast and trains full batches 1.2 times as fast on one thread and as fast on four. The gap
is largest for mini-batches, where fixed per-step costs weigh most.

Against 0.6, the benchmark network runs at the same speed. Output layers of up to 16 units now run
as blocks of dot products instead of mostly idle matrix panels, and products too small to gain from
threads run on one: a 32-64-64-1 regression network trains 7% faster on one thread and 65% faster
on four, and infers 10% and 40% faster.

The x86-64 release packages are compiled for a baseline instruction set and choose AVX2 or AVX-512
matrix kernels at run time (since 0.6). On the same machine, on one thread, the baseline package
trains 3.6 to 4.9 times as fast, and infers 5 times as fast, as with kernels for its baseline
(SSE2), as in 0.5:

| Baseline x86-64 build, one thread | Full batch | Mini-batch 64 | Inference |
|---|---:|---:|---:|
| SSE2 kernels (0.5) | 3,700 | 3,200 | 9,000 |
| AVX-512 kernels chosen at run time (0.6) | 18,200 | 11,400 | 45,000 |

`Examples/MNIST.c` trains a 784-256-128-10 network with dropout (AdamW, cosine schedule,
mini-batches of 128) on 55,000 images, keeps the epoch with the best accuracy on the other 5,000
and reaches 98.2% test accuracy after 10 epochs, which take a few seconds with OpenMP on the same
VM. `Examples/MNIST_CNN.c` reaches 98.9% after two epochs of about 9 seconds each on four threads.

## Project layout

```
Include/Spingalett/   Public headers (Spingalett.h, the inference engine's Spingalett.Inference.h)
                      and the CMake-generated configuration header template
Src/                  Library sources (network, training, kernels, serialization, inference engine, ...)
Examples/             XOR, MNIST (dense and convolutional), throughput benchmark (C and PyTorch
                      counterpart), DatasetTool, ModelTool
Examples/Embedded/    MNIST on a Cortex-M4 (QEMU) with the standalone inference engine
docs/                 File format specifications (models, data sets)
Apps/DigitPad/        Digit-drawing demo app, its trainer and AppImage and Windows packaging
Tests/                Test suite (CTest) and fixtures
Bindings/Python/      Python bindings
cmake/                CMake package and inference-only build helpers
```

## Status and roadmap

Spingalett is at version 0.7; the C API may still change between minor versions (see
[CHANGELOG.md](CHANGELOG.md)), and the shared library's soname carries the minor version
(`libspingalett.so.0.7`). Since 0.7 the network is an opaque handle, so its internal layout can
change without breaking programs. Saved models are versioned and remain loadable; the inference
engine and model format versions 3 and 4 are meant to stay stable from here on.

Planned work, roughly in order:

- 0.8: batch normalization, depthwise and grouped convolutions, a CIFAR-10 example, faster integer
  convolutions on more targets
- 0.9: residual connections (networks as graphs), ONNX import, Python wheels on PyPI, a first GPU
  backend
- 1.0: API freeze, C++ wrapper
- Later: CUDA/cuDNN backend, quantization-aware training, NEON kernels for training, further
  language bindings

## Contributing

Bug reports and pull requests are welcome. Please make sure the test suite passes
(`ctest --test-dir Build --output-on-failure`) for both a minimal build and a build with
`-DBUILD_WITH_OPENMP=ON -DBUILD_WITH_OPENBLAS=ON`, and add tests for new behaviour. CI runs the
suite with GCC and Clang, without `-march=native` (portable kernels) and under AddressSanitizer
and UndefinedBehaviorSanitizer.

## License

Spingalett is released under the [MIT License](LICENSE).
