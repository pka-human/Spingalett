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
networks on the CPU. It depends only on the C standard library; OpenMP and OpenBLAS are optional
build-time accelerators. Networks are declared with C23 designated initializers, all parameters
live in flat contiguous arrays, and models can be saved in reduced precision down to 2 bits per
weight. Python bindings are included.

## Contents

- [Features](#features)
- [Building](#building)
- [Quick start](#quick-start)
- [Usage](#usage)
- [Python bindings](#python-bindings)
- [Performance](#performance)
- [Project layout](#project-layout)
- [Status and roadmap](#status-and-roadmap)
- [Contributing](#contributing)
- [License](#license)

## Features

| Area | Supported |
|---|---|
| Layers | Fully connected, optional dropout per layer |
| Activations | Sigmoid, ReLU, Leaky ReLU, Tanh, FOO52, Softmax (output layer), None |
| Losses | Mean squared error, cross-entropy (softmax or sigmoid outputs) |
| Optimizers | SGD, Momentum, RMSProp, Adam, AdamW; L2 or decoupled weight decay |
| Training | Per-sample, full-batch and mini-batch strategies; in-memory arrays or a data generator |
| Regularization and stability | Dropout, weight decay, global gradient-norm clipping, NaN/Inf detection |
| Learning-rate schedules | Cosine decay, linear warm-up, step decay, warm-up + cosine, or a custom callback |
| Initialization | Uniform, Xavier/LeCun normal, He normal |
| Backends | Single-threaded (AVX/AVX2/FMA), OpenMP, OpenBLAS |
| Serialization | FP32, FP16, BF16, INT8, INT4, INT2; optional optimizer state; versioned format |
| Bindings | Python (ctypes + NumPy) |

## Building

Requirements: CMake 3.21 or newer and a compiler with C23 support. GCC 13 and Clang 18 are tested
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
Release builds use `-march=native`, so binaries are tuned for the build machine.

| CMake option | Default | Description |
|---|---|---|
| `BUILD_WITH_OPENMP` | `OFF` | Enable the OpenMP backend |
| `BUILD_WITH_OPENBLAS` | `OFF` | Enable the OpenBLAS backend (found via pkg-config or the default library paths) |
| `BUILD_EXAMPLE` | `ON` | Build the programs in `Examples/` |
| `BUILD_TESTS` | `ON` | Build the test suite and register it with CTest |
| `SPINGALETT_BIN_DIR` | `<source>/Bin` | Output directory for executables and shared libraries |
| `SPINGALETT_LIB_DIR` | `<source>/Lib` | Output directory for static and import libraries |

Projects that add Spingalett with `add_subdirectory()` only need to link the `spingalett`
target; include paths, including the generated `Spingalett.Config.h`, are propagated.

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

    save_spingalett(.net = net, .filename = "xor");             /* writes xor.nn */
    free_network(net);
    return 0;
}
```

Every function that takes many parameters is a macro over a struct, so arguments can be given
positionally (`layer(net, 8, ACT_TANH)`) or by name (`layer(.net = net, .neurons_amount = 8)`).
Fields that are not mentioned are zero, which selects the documented default.

## Usage

### Defining a network

`new_spingalett(loss)` creates an empty network; each `layer()` call appends a layer, the first
one being the input layer. `LayerArgs` fields:

| Field | Meaning |
|---|---|
| `neurons_amount` | Layer width |
| `act_func` | Activation of this layer (ignored for the input layer) |
| `weight_initialization` | `RANDOM` (uniform in [-1, 1]), `XAVIER` (normal, variance 1/fan_in), `HE` (normal, variance 2/fan_in), `NONE` (zeros) |
| `dropout_rate` | Probability in [0, 1) of zeroing each output of this layer during training |

Biases start at zero. Softmax is only allowed in the output layer. Leaky ReLU uses a slope of
0.01; FOO52 is the piecewise-linear function `0.01x` for `x < 0`, `x` on `[0, 1]` and
`1 + 0.01(x - 1)` above 1.

Dropout is inverted (kept units are scaled by `1 / (1 - p)`), so inference needs no rescaling.
It is not applied to the input or output layer.

### Training

`train()` takes a `TrainArgs` struct. Zero-valued fields use the defaults below.

| Field | Default | Description |
|---|---|---|
| `inputs`, `targets`, `sample_count` | required | Row-major arrays of `sample_count` samples |
| `epochs` | required | Number of passes over the data |
| `training_strategy` | `STRATEGY_SAMPLE` | `STRATEGY_SAMPLE` (one step per sample, in order), `STRATEGY_FULL_BATCH`, `STRATEGY_SMALL_BATCH` (reshuffled every epoch) |
| `batch_size` | 32 | Mini-batch size |
| `optimizer_type` | `OPTIMIZER_SGD` | `SGD`, `MOMENTUM`, `RMSPROP`, `ADAM`, `ADAMW` |
| `learning_rate` | 0.01 | Base learning rate |
| `momentum` | 0.9 | Momentum coefficient |
| `beta1`, `beta2`, `epsilon` | 0.9, 0.999, 1e-8 | Adam/RMSProp moment parameters |
| `weight_decay` | 0 | L2 penalty; decoupled (`w *= 1 - lr * wd`) for AdamW. Biases are not decayed |
| `max_grad_norm` | 0 (off) | Clip the global L2 norm of every step's gradient |
| `lr_scheduler`, `lr_scheduler_data` | none | Learning-rate schedule, see below |
| `reset_optimizer` | false | Clear moment estimates and the step counter before training |
| `nan_check_interval` | 0 (off) | Stop when weights become NaN/Inf, checked every N epochs |
| `report_interval` | 0 (off) | Log the training loss every N epochs |
| `callback`, `callback_interval` | none, 1 | `bool cb(NeuralNetwork *, size_t epoch, float loss)`; return `true` to stop |
| `autosave_mode`, `autosave_interval`, `autosave_path` | off | Periodic checkpoints (`AUTOSAVE_OVERWRITE` or `AUTOSAVE_NEW_FILES`, which appends `_epoch_N`) |
| `autosave_precision`, `autosave_do_not_save_optimizer` | FP32, false | Checkpoint format |
| `blas_num_threads` | 0 (auto) | OpenBLAS threads during training, see [Backends](#backends-and-threading) |

The reported loss is averaged over the samples of an epoch: the sum of squared errors per sample
for MSE, and categorical (softmax) or binary (sigmoid) cross-entropy otherwise.

Optimizer state (moment estimates and the step counter used for Adam's bias correction) is stored
in the network, so training can be resumed by calling `train()` again or after loading a
checkpoint that includes the optimizer state.

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

### Inference

`forward(net, input)` returns a pointer to the output layer inside the network. The buffer is
overwritten by the next call, so copy the values if they need to be kept. Dropout is not applied.

### Saving and loading

```c
save_spingalett(.net = net, .filename = "model.nn", .precision = PRECISION_FP16, .do_not_save_optimizer = true);
NeuralNetwork *net = load_spingalett("model.nn");
```

`.nn` is appended when the filename has no extension. Weights, biases and, unless disabled, the
optimizer state are stored in the selected precision:

| Precision | Storage per value | Notes |
|---|---|---|
| `PRECISION_FLOAT32` | 4 bytes | Lossless |
| `PRECISION_FP16`, `PRECISION_BFLOAT16` | 2 bytes | Round to nearest even |
| `PRECISION_INT8`, `PRECISION_INT4` | 1 byte, 4 bits | Symmetric, one scale per tensor |
| `PRECISION_INT2` | 2 bits | Ternary {-scale, 0, +scale} |

The file format is versioned (`SPINGALETT_FORMAT_VERSION`, currently 2); version 1 files remain
loadable. Values are stored in native byte order. `load_spingalett()` returns `NULL` for missing
or truncated files and for files with an invalid header.

### Backends and threading

```c
spingalett_set_compute_mode(COMPUTE_OPENBLAS);   /* COMPUTE_SINGLE_THREADED, COMPUTE_OPENMP, COMPUTE_OPENBLAS */
spingalett_set_num_threads(8);                   /* 0 = runtime default */
```

- `COMPUTE_SINGLE_THREADED` uses hand-written AVX/AVX2/FMA kernels when the compiler targets them.
- `COMPUTE_OPENMP` parallelizes layers that are large enough to benefit; small layers run serially.
- `COMPUTE_OPENBLAS` runs batch training as matrix-matrix products and is the fastest option for
  full-batch and mini-batch training. With `blas_num_threads = 0`, training uses a single
  OpenBLAS thread when each call is too small to amortize threading (per-sample training, small
  layers or small batches) and the configured thread count otherwise. The caller's OpenBLAS
  setting is restored afterwards.
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
and dropout masks are identical across backends and thread counts.

### Errors and logging

Functions report failures through a thread-local error state instead of return codes:

```c
spingalett_clear_error();
NeuralNetwork *net = load_spingalett("model.nn");
if (!net)
    fprintf(stderr, "%s (code %d)\n", spingalett_last_error_message(), spingalett_last_error_code());
```

Error codes are `SPINGALETT_OK`, `SPINGALETT_ERR_ALLOC` and `SPINGALETT_ERR_INVALID`. The error
state is not cleared by successful calls.

Log messages go to stdout (warnings and errors to stderr) unless redirected with
`spingalett_set_log_callback(void (*)(LogLevel, const char *))`. `spingalett_set_log_level()` sets
the minimum level and `spingalett_set_verbose(false)` suppresses everything below warnings.

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
```

See [Bindings/Python/README.md](Bindings/Python/README.md) for the full API.

## Performance

`Examples/Benchmark.c` trains a 784-512-1000-10 network (925K parameters, ReLU, softmax with
cross-entropy, Adam, full batch) on 20,000 synthetic samples for 5 epochs and reports throughput
for every backend compiled in. Run it with `Bin/Benchmark [threads]`.

The table compares the current code with the previous release, both trained with this
configuration on the same 4-vCPU cloud VM (Intel Xeon @ 2.10 GHz, AVX2, GCC 13,
OpenBLAS 0.3.26):

| Backend | Previous release | Current | Change |
|---|---:|---:|---:|
| Single-threaded | 2,949 samples/s | 3,960 samples/s | +34% |
| OpenMP (4 threads) | 6,910 samples/s | 11,268 samples/s | +63% |
| OpenBLAS | 55,054 samples/s | 78,654 samples/s | +43% |

Per-sample training (`STRATEGY_SAMPLE`, 2,000 samples) of the same network improved from 132 to
1,480 samples/s single-threaded and from 435 to 3,924 samples/s with OpenMP, mainly by flushing
denormals during training and by reusing the vectorized optimizer kernels.

An earlier comparison with PyTorch on an Intel Core i7-12650H, measured with the previous
release, reported 25,349 vs. 22,195 samples/s on one thread and 88,921 vs. 63,068 samples/s on
twelve threads (Spingalett with OpenBLAS vs. PyTorch CPU with Intel MKL).

## Project layout

```
Include/Spingalett/   Public header and the CMake-generated configuration header template
Src/                  Library sources (network, training, SIMD kernels, serialization, ...)
Examples/             XOR example and throughput benchmark
Tests/                Test suite (CTest) and fixtures
Bindings/Python/      Python bindings
```

## Status and roadmap

Spingalett is at version 0.1; the C API and the in-memory `NeuralNetwork` layout may still change
between minor versions. Saved models are versioned and remain loadable.

Planned work, roughly in order:

- A layer abstraction as the basis for further layer types
- Batch normalization, 2D convolution and pooling layers
- A native blocked matrix-multiplication kernel, so the dependency-free build approaches
  OpenBLAS throughput
- CUDA backend

## Contributing

Bug reports and pull requests are welcome. Please make sure the test suite passes
(`ctest --test-dir Build --output-on-failure`) for both a minimal build and a build with
`-DBUILD_WITH_OPENMP=ON -DBUILD_WITH_OPENBLAS=ON`, and add tests for new behaviour. CI runs the
suite with GCC and Clang and under AddressSanitizer and UndefinedBehaviorSanitizer.

## License

Spingalett is released under the [MIT License](LICENSE).
