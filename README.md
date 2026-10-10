<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="Images/LogoDark.png">
  <source media="(prefers-color-scheme: light)" srcset="Images/LogoLight.png">
  <img alt="Spingalett" src="Images/LogoLight.png">
</picture>

[![CI](https://github.com/pka-human/Spingalett/actions/workflows/ci.yml/badge.svg)](https://github.com/pka-human/Spingalett/actions/workflows/ci.yml)
[![Standard](https://img.shields.io/badge/C-23-blue.svg?style=flat-square)](https://en.wikipedia.org/wiki/C23_(C_standard_revision))
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=flat-square)](LICENSE)
[![Ask DeepWiki](https://img.shields.io/badge/Ask-DeepWiki-blue.svg?style=flat-square)](https://deepwiki.com/pka-human/Spingalett)

</div>

Spingalett is a neural-network library written in C23 for training and running fully connected
and convolutional networks, chains of layers or graphs of them (residual connections, concatenated
branches), on the CPU and on GPUs: NVIDIA's through CUDA, with kernels of its own (no CUDA toolkit,
cuBLAS or cuDNN), and others through Vulkan compute. It depends only on the C standard library (the
CUDA driver and the Vulkan loader are opened at run time where there are some): on the CPU, batch
training and inference run as matrix-matrix products on built-in AVX-512, AVX2, NEON or portable kernels (on
x86-64, chosen for the processor at run time), with OpenMP and OpenBLAS as optional build-time
accelerators; results are the same bits on one thread and many. Networks are declared with C23
designated initializers and all parameters live in flat contiguous arrays. For deployment, a trained network becomes a read-only
model in FP32, FP16, BF16, INT8, INT4 or INT2 that runs with integer kernels where the weights are
integers, in place from memory, a compiled-in array or flash, on desktops and on microcontrollers
alike; programs that only run models link the runtime, a library of that part alone. Models come in
from ONNX files and PyTorch weights, and Python bindings (wheels with the library inside) are
included.

## Contents

- [Features](#features)
- [Building](#building)
- [Quick start](#quick-start)
- [Usage](#usage)
- [Deployment](#deployment)
- [Python bindings](#python-bindings)
- [C++](#c)
- [DigitPad demo](#digitpad-demo)
- [Performance](#performance)
- [Project layout](#project-layout)
- [Compatibility](#compatibility)
- [Status and roadmap](#status-and-roadmap)
- [Contributing](#contributing)
- [License](#license)

## Features

| Area | Supported |
|---|---|
| Layers | Fully connected, 2D convolution (any kernel, stride and padding, rectangular windows; grouped and depthwise), transposed 2D convolution (strides, padding, output padding, groups), max, average and global average pooling, nearest and bilinear upsampling, batch and layer normalization, addition and concatenation of layers; channels-last tensors; optional dropout per layer |
| Architectures | Chains of layers, or directed acyclic graphs: any layer reads any earlier ones (residual networks, Inception- and DenseNet-style branches, U-Nets), trained and run with outputs sharing memory at inference |
| Activations | Sigmoid, ReLU, Leaky ReLU, Tanh, FOO52, Softmax (output layer), None |
| Losses | Mean squared error, cross-entropy (softmax or sigmoid outputs) |
| Optimizers | SGD, Momentum, RMSProp, Adam, AdamW; L2 or decoupled weight decay |
| Training | Per-sample, full-batch and mini-batch strategies; in-memory arrays or a data generator; image augmentation (random shifts, mirror images); validation with early stopping and best-weight restore |
| Custom loops | Public forward / backward / optimizer-step API with custom losses and gradient accumulation |
| Data | `.slettd` data set files (compact, lossless by default, streamable into training), IDX (MNIST), CIFAR-10/100 and CSV readers, shuffling, hold-out splits |
| Regularization and stability | Dropout, weight decay, global gradient-norm clipping, NaN/Inf detection |
| Learning-rate schedules | Cosine decay, linear warm-up, step decay, warm-up + cosine, or a custom callback |
| Initialization | Uniform, Glorot (Xavier), He and LeCun normal |
| Inference | Per-sample `spingalett_forward()`, batched `spingalett_predict()`, `spingalett_evaluate()` (loss and accuracy) |
| Deployment | Read-only models, dense and convolutional, in FP32, FP16, BF16, INT8, INT4 or INT2 with per-row scales and int8 x int8 kernels (AVX-512 VNNI, AVX-VNNI, AVX2, SSE2, NEON with or without the dot product extension, Arm DSP; INT4 and INT2 decoded in registers), batch normalization folded into the layer before it; run in place from memory or flash; C header export; the runtime, a library of the models alone (no training code, a quarter of the size); a standalone engine for microcontrollers (one C file, no heap) |
| Backends | Built-in matrix kernels (AVX-512, AVX2/FMA, AVX, NEON, portable C; on x86-64 chosen at run time), single-threaded or OpenMP; OpenBLAS; a GPU for training, `spingalett_predict()` and `spingalett_evaluate()`, deterministic, in single precision or bfloat16 on its matrix units: NVIDIA GPUs (Ampere and later) through CUDA with the library's own kernels, any GPU through Vulkan compute (NVIDIA, AMD, Intel; Apple through MoltenVK) |
| Serialization | `.slett` model files in FP32, FP16, BF16, INT8, INT4 or INT2, optional optimizer state, CRC-32 checksums; to and from memory; versioned format |
| Introspection | Layer descriptions and parameter copies by layer through accessor functions (the network is an opaque handle) |
| Interoperability | ONNX import (`spingalett_import_onnx()`, `ModelTool import`), PyTorch weights from `torch.save` and safetensors files, `Network.from_torch()` in Python |
| Bindings | Python (ctypes + NumPy; wheels with the library inside, typed) |

## Building

Prebuilt libraries for Linux (x86-64 and ARM64), Windows and macOS are attached to every
[release](https://github.com/pka-human/Spingalett/releases): each archive contains the headers,
the shared library, a CMake package, a pkg-config file, `DatasetTool` and `ModelTool`, with the
runtime library next to them; the `spingalett-runtime-*` archives hold the [runtime](#the-runtime)
alone, for programs that only run models. Extract one
and point CMake at it with `-DCMAKE_PREFIX_PATH=<directory>`, or compile directly with
`-I<dir>/include -L<dir>/lib -lspingalett`. The x86-64 archives come in a baseline build that runs on any x86-64 CPU and a
`-v3` build for processors with AVX2 and FMA; both pick AVX2 or AVX-512 matrix kernels at run time
when the processor has them, and the `-v3` build also compiles the rest of the library
(activations, optimizers, the inference engine) for AVX2. The Windows DLL ships with import
libraries for MinGW and MSVC. Every package is built with OpenMP and the Vulkan GPU backend, those for
Linux and Windows with the CUDA backend too; the macOS one (a universal library for Apple silicon and Intel, macOS 11 or newer) carries LLVM's
OpenMP runtime, `libomp.dylib`, next to the library, since Apple's compilers come without one.

To build from source: requirements: CMake 3.21 or newer and a compiler with C23 support. GCC 13 and Clang 18 are tested
in CI; MSVC 19.36 or newer is expected to work but is not tested. OpenMP and OpenBLAS are
optional, and so are the GPU backends: the Vulkan one needs the Vulkan headers and `glslc` (shaderc) to
build, the CUDA one a Clang with the NVPTX target, which compiles its kernels to PTX (no CUDA toolkit:
the driver compiles the PTX on first use); neither needs the GPU's runtime to build, since the library
opens it when a GPU mode is first used.

```bash
# Dependency-free build
cmake -S . -B Build -DCMAKE_BUILD_TYPE=Release
cmake --build Build --parallel

# With OpenMP and OpenBLAS (Debian/Ubuntu: apt install libopenblas-dev)
cmake -S . -B Build -DCMAKE_BUILD_TYPE=Release -DBUILD_WITH_OPENMP=ON -DBUILD_WITH_OPENBLAS=ON
cmake --build Build --parallel

# The GPU backends are built when their compilers are found: Vulkan with glslc and the Vulkan headers
# (Debian/Ubuntu: apt install glslc libvulkan-dev; macOS: brew install shaderc vulkan-headers),
# CUDA with a Clang that targets NVPTX (Debian/Ubuntu: apt install clang)

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
| `SPINGALETT_RUNTIME` | `ON` | Build the [runtime](#the-runtime) library next to the library |
| `SPINGALETT_RUNTIME_ONLY` | `OFF` | Build only the runtime library (`Spingalett.Runtime.h`): deployment models on the CPU, without training, importers or the GPU backend (no `glslc` needed) |
| `SPINGALETT_INFERENCE_ONLY` | `OFF` | Build only the inference engine (`Spingalett.Inference.h`) as a static library: no training, file I/O, OpenMP or heap |
| `SPINGALETT_NATIVE_ARCH` | `ON` | Compile with `-march=native`; turn off for binaries that must run on other machines |
| `SPINGALETT_CPU_DISPATCH` | `ON` | Without `-march=native` (GCC, Clang): on x86-64 also build AVX2 and AVX-512 matrix kernels and AVX-512 VNNI and AVX-VNNI integer kernels, on AArch64 Linux integer kernels for the dot product instructions, and choose at run time |
| `SPINGALETT_VULKAN` | `AUTO` | The Vulkan GPU backend: `AUTO` builds it when `glslc` and the Vulkan headers are found, `ON` requires them, `OFF` leaves it out |
| `SPINGALETT_SPIRV_DIR` | | Compiled shaders (`<kernel>.spv.inc` from another build's `Gpu/` directory) to use where there is no `glslc` |
| `SPINGALETT_CUDA` | `AUTO` | The CUDA GPU backend: `AUTO` builds it when a Clang with the NVPTX target is found (`SPINGALETT_CUDA_CLANG` names one), `ON` requires it, `OFF` leaves it out |
| `SPINGALETT_PTX_DIR` | | Compiled CUDA kernels (`<unit>.ptx.inc` from another build's `Cuda/` directory) to use where there is no such Clang |
| `SPINGALETT_BIN_DIR` | `<source>/Bin` | Output directory for executables and shared libraries |
| `SPINGALETT_LIB_DIR` | `<source>/Lib` | Output directory for static and import libraries |

An installed Spingalett is found with CMake's `find_package`; projects that vendor the
repository can use `add_subdirectory()` instead. Both provide the target `Spingalett::spingalett`,
and `Spingalett::runtime` for the runtime, which carry the include paths:

```cmake
find_package(Spingalett 1.1 REQUIRED)         # or: add_subdirectory(external/Spingalett)
target_link_libraries(my_app PRIVATE Spingalett::spingalett)    # or Spingalett::runtime
```

pkg-config finds it as `spingalett` (`cc app.c $(pkg-config --cflags --libs spingalett)`), and the
runtime as `spingalett-runtime`; the file
locates the installation from its own directory, so an extracted release archive serves too, through
`PKG_CONFIG_PATH=<dir>/lib/pkgconfig`.

The repository carries a [Conan](https://conan.io) recipe and a [vcpkg](https://vcpkg.io) port that
build the library of the checkout (shared, with OpenMP, kernels chosen at run time, the runtime
next to it; the GPU backend as an option, with `glslc` and the Vulkan headers from the package
manager). Both give `find_package(Spingalett)` and pkg-config's `spingalett` and
`spingalett-runtime`; they build with GCC or Clang (MinGW on Windows):

```bash
conan create packaging/conan --build=missing                  # -o "&:vulkan=True": the GPU backend
                                                              # -o "&:runtime_only=True": the runtime alone
vcpkg install spingalett --overlay-ports=packaging/vcpkg      # "spingalett[vulkan]": the GPU backend
```

Programs that compile the library's sources into themselves, or link it statically, define
`SPINGALETT_STATIC` so that the headers do not declare the API as imported from a DLL.

The headers define `SPINGALETT_VERSION_MAJOR`, `_MINOR`, `_PATCH` and `_STRING`;
`spingalett_version()` returns the version of the library actually loaded.

## Quick start

```c
#include <Spingalett/Spingalett.Short.h>
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

**Names.** The examples here and in `Examples/` use the short names of
`Spingalett/Spingalett.Short.h` (`NeuralNetwork`, `layer()`, `train()`, `ACT_RELU`, ...), which read
more easily. The library declares every name with its prefix (`spingalett_` for functions and
builders, `Spingalett` for types, `SPINGALETT_` for constants), and so do the descriptions in this
document and [docs/Reference.md](docs/Reference.md): C has one namespace for everything a program
includes, and names such as `train()`, `layer()` or `LOG_DEBUG` (also `<syslog.h>`'s) would collide
with the program's own and with other libraries'. The short names are macros and typedefs over the
prefixed ones, so a program compiles to the same calls either way: it includes `Spingalett.Short.h`
in place of `Spingalett.h` (or defines `SPINGALETT_SHORT_NAMES` before including that) to use them,
and `Spingalett.h` alone leaves them out. Functions whose names had the prefix in 0.x keep it
(`spingalett_model_load()`, `spingalett_load_idx()`, `spingalett_set_compute_mode()`).

The `Examples/` directory contains this XOR program, an MNIST classifier
(`Examples/download_mnist.sh data/mnist && Bin/MNIST data/mnist`), a convolutional one
(`Bin/MNIST_CNN data/mnist`, about 99% after two epochs), a CIFAR-10 classifier with batch
normalization and augmentation (`Examples/download_cifar10.sh data/cifar10 && Bin/CIFAR10
data/cifar10`) and the benchmark described under [Performance](#performance).
[docs/Tutorial.md](docs/Tutorial.md) goes through these programs in order, and
[docs/Reference.md](docs/Reference.md) lists every declaration of the headers with its comment.

## Usage

### Defining a network

`spingalett_network_new(loss)` creates an empty network; each `spingalett_layer()` call appends a
layer, the first one being the input layer. `SpingalettLayerArgs` fields:

| Field | Meaning |
|---|---|
| `neurons_amount` | Layer width |
| `act_func` | Activation of this layer (ignored for the input layer); left out, `SPINGALETT_ACT_NONE` |
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
row by row with interleaved channels goes in as it is. Give the input layer a shape, then add layers
with `spingalett_conv2d()`, `spingalett_max_pool2d()` and `spingalett_avg_pool2d()`; a dense layer
reads whatever precedes it as a flat vector, so no flattening layer is needed:

```c
NeuralNetwork *net = new_spingalett(LOSS_CROSS_ENTROPY);
layer(.net = net, .height = 28, .width = 28, .channels = 1);                    /* input */
conv2d(.net = net, .filters = 32, .kernel = 3, .padding = 1,                    /* 28 x 28 x 32 */
       .act_func = ACT_RELU, .weight_initialization = WEIGHT_INITIALIZATION_HE);
max_pool2d(.net = net, .kernel = 2);                                            /* 14 x 14 x 32 */
conv2d(.net = net, .filters = 64, .kernel = 3, .padding = 1,                    /* 14 x 14 x 64 */
       .act_func = ACT_RELU, .weight_initialization = WEIGHT_INITIALIZATION_HE);
max_pool2d(.net = net, .kernel = 2);                                            /* 7 x 7 x 64 */
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
| `groups` | Convolutions: split the input channels and the filters into this many groups; each filter sees the input channels of its own group. As many groups as input channels make a depthwise convolution (with any number of filters per channel) |

An output axis has `(size + 2 padding - kernel) / stride + 1` cells. A convolution's filters are
`kernel_h x kernel_w x (input channels / groups)` weights each, plus one bias, and take an
activation, a weight initialization (fan-in is the window size) and dropout like dense layers.
Pooling has no parameters and no activation; windows are clipped to the input, so padding cells
never count (average pooling divides by the cells inside), and max pooling passes the gradient to
the first maximum. Every training strategy, optimizer, schedule, the custom training loop and
`spingalett_predict()` work with these layers as with dense ones.

Convolutions run as indirect matrix products: the matrix kernels read each pixel's window straight
from the image through one pointer per kernel tap, so no window is ever gathered or copied, with the
bias and activation applied to each tile of the result while it is in cache. The data gradient of
strided convolutions runs a phase of input cells at a time, over the taps that cover them only, and
weight gradients split their long summation over all pixels into slots fixed by the shape that the
threads share. Depthwise convolutions are computed directly, all of a pixel's channels at once, and
other grouped ones as one product per group. The network of `Examples/MNIST_CNN.c` trains 1.4 to
2.3 times as fast as in PyTorch and infers 2.6 to 3.9 times as fast, ResNet-20 1.1 to 1.8 and 1.5 to
2.8 times as fast (see [Performance](#performance)).

### Batch normalization

`spingalett_batch_norm()` normalizes the previous layer per channel (per unit after a dense layer):
`y = act(gamma (x - mean) / sqrt(var + epsilon) + beta)`. While training, mean and variance are
those of the batch (of each chunk of up to 2,048 samples in full-batch training), and running
averages of them move `momentum` of the way towards each batch's; `spingalett_predict()`,
`spingalett_forward()`, `spingalett_evaluate()` and deployment models use the running averages.
Gamma starts at 1 and beta at 0; they are the layer's weights and biases for every optimizer, but
weight decay leaves them alone. The convolution or dense layer before a normalization usually takes
no activation:

```c
conv2d(.net = net, .filters = 64, .kernel = 3, .padding = 1,
       .weight_initialization = WEIGHT_INITIALIZATION_HE);                      /* no activation */
batch_norm(.net = net, .act_func = ACT_RELU);                         /* .epsilon 1e-5, .momentum 0.1 */
```

Such a pair costs no more than the convolution alone at inference: `spingalett_predict()` and
`spingalett_forward()` apply the normalization while the convolution's tiles are in cache, and
deployment models (and files saved in other precisions than FP32 without optimizer state) fold it
into the convolution's weights and biases. A normalization of a dense layer needs batches of at
least 2 samples. `SPINGALETT_PARAM_RUNNING_MEAN` and `SPINGALETT_PARAM_RUNNING_VARIANCE` read and
write the statistics, and `restore_best_weights` restores them with the weights.

### Graphs: residual connections and branches

Every builder returns the index of the layer it adds (`SPINGALETT_NO_LAYER` on error), and `.inputs`
names the earlier layers a layer reads; without it a layer reads the one added before it, so chains
are built as before. Two kinds of layers combine others: `spingalett_add_layers()` sums layers of
one shape and `spingalett_concat_layers()` puts layers of one height and width side by side along
the channels, each followed by its `.act_func` (none when left out).
`spingalett_global_avg_pool2d()` averages each channel over all cells. A residual block of ResNet:

```c
uint32_t block(NeuralNetwork *net, uint32_t x, uint32_t filters, uint32_t stride) {
    conv2d(.net = net, .inputs = {x}, .filters = filters, .kernel = 3, .padding = 1,
           .stride = stride, .weight_initialization = WEIGHT_INITIALIZATION_HE);
    batch_norm(.net = net, .act_func = ACT_RELU);
    conv2d(.net = net, .filters = filters, .kernel = 3, .padding = 1,
           .weight_initialization = WEIGHT_INITIALIZATION_HE);
    uint32_t y = batch_norm(.net = net);
    uint32_t shortcut = x;
    if (stride != 1) {                      /* the shortcut changes shape too: a 1 x 1 projection */
        conv2d(.net = net, .inputs = {x}, .filters = filters, .kernel = 1, .stride = stride,
               .weight_initialization = WEIGHT_INITIALIZATION_HE);
        shortcut = batch_norm(.net = net);
    }
    return add_layers(.net = net, .inputs = {shortcut, y}, .act_func = ACT_RELU);
}
```

and Inception-style branches, concatenated:

```c
uint32_t x = layer(.net = net, .height = 32, .width = 32, .channels = 16);
uint32_t a = conv2d(.net = net, .inputs = {x}, .filters = 8, .kernel = 1,
                    .act_func = ACT_RELU);
uint32_t b = conv2d(.net = net, .inputs = {x}, .filters = 8, .kernel = 3, .padding = 1,
                    .act_func = ACT_RELU);
uint32_t c = max_pool2d(.net = net, .inputs = {x}, .kernel = 3, .stride = 1, .padding = 1);
concat_layers(.net = net, .inputs = {a, b, c});                                 /* 32 x 32 x 32 */
```

`.input_count` gives the number of inputs; left 0 it counts `.inputs` up to the last nonzero entry,
so it is needed when the last input is the input layer (`.inputs = {a, 0}, .input_count = 2`). A
layer reads at most `SPINGALETT_MAX_INPUTS` (16) layers. The last layer is the network's output,
and every other layer must feed a later one before the network trains or predicts.

Layers run in the order they were added, which puts each after its inputs. Training follows the
graph backwards: a layer that feeds several gets their gradients in a fixed order, so training is
still the same bits on any number of threads. At inference, outputs share memory where their lives
do not overlap, in `spingalett_predict()`, deployment models and the engine (whose `.slett` files,
format version 6, record where each output lives): a residual network of 16 blocks needs memory for
a few of its widest outputs, not for all of them. `spingalett_network_layer()` reports each layer's
`inputs`, and `ModelTool info` lists them. `Examples/CIFAR10.c` builds ResNet-20 to ResNet-56
(`resnet20`, `resnet32`, ...).

### Transposed convolutions, upsampling and layer normalization

`spingalett_conv_transpose2d()` is the transpose of a convolution: each input cell spreads its
channels over a window of the output through the filters, its windows `stride` cells apart, so a 2 x
2 kernel of stride 2 doubles the height and width. `spingalett_upsample2d()` repeats or interpolates
cells by integer factors, and `spingalett_layer_norm()` normalizes each cell over its channels. With
concatenations they make U-Nets, the expanding path joined to the maps of the same size on the way
down:

```c
uint32_t e = conv2d(.net = net, .filters = 32, .kernel = 3, .padding = 1,
                    .act_func = ACT_RELU);                                      /* 64 x 64 x 32 */
max_pool2d(.net = net, .kernel = 2);                                            /* 32 x 32 x 32 */
conv2d(.net = net, .filters = 64, .kernel = 3, .padding = 1,
       .act_func = ACT_RELU);                                                   /* 32 x 32 x 64 */
uint32_t u = conv_transpose2d(.net = net, .filters = 32, .kernel = 2, .stride = 2,
                              .act_func = ACT_RELU);                            /* 64 x 64 x 32 */
concat_layers(.net = net, .inputs = {e, u});                                    /* 64 x 64 x 64 */
upsample2d(.net = net, .stride = 2,                                             /* 128 x 128 x 64 */
           .upsample = UPSAMPLE_BILINEAR);
layer_norm(.net = net, .act_func = ACT_RELU);                                   /* 128 x 128 x 64 */
```

| Builder | Arguments | Output, along each axis |
|---|---|---|
| `spingalett_conv_transpose2d()` | `filters`, `kernel`, `stride`, `padding`, `output_padding` (each with `_h` and `_w` forms), `groups`, `act_func` | `(size - 1) stride - 2 padding + kernel + output_padding` cells |
| `spingalett_upsample2d()` | `stride` (or `stride_h`, `stride_w`): the factors, 2 by default; `upsample`: `SPINGALETT_UPSAMPLE_NEAREST` (default) or `SPINGALETT_UPSAMPLE_BILINEAR` | `size x factor` cells |
| `spingalett_layer_norm()` | `epsilon` (1e-5), `act_func` | the input's shape |

- A transposed convolution's output padding (less than the stride) adds cells at the end of each
  axis, so that the output can match the input of the convolution it transposes: a 3 x 3
  convolution of stride 2 and padding 1 halves 64 cells to 32, and its transpose with
  `.output_padding = 1` makes 64 again. Its filter `j` holds, at `(kh, kw, c)`, the weight by which
  input channel `c` of its group reaches output channel `j` at offset `(kh, kw)` of the input
  cell's window: `kernel_h x kernel_w x (input channels / groups)` weights an output channel, as a
  convolution's filters have. It runs on the convolution kernels with the passes swapped: its
  forward pass is the data gradient of the convolution it transposes (a phase of the stride at a
  time, over the taps that reach it), its data gradient that convolution's forward pass.
- Bilinear upsampling interpolates between the cells' centres, repeating the cells at the edges
  (PyTorch's `align_corners=False`); its weights come from integers, so every platform computes
  the same ones. Upsampling has no parameters and no activation.
- Layer normalization computes the mean and variance of each cell's channels (a dense layer's
  outputs are a single cell): `y = act(gamma (x - mean) / sqrt(var + epsilon) + beta)`, gamma and
  beta per channel. It has no running statistics, so training and inference compute the same at any
  batch size; weight decay leaves gamma alone. Unlike batch normalization it is not folded into the
  layer before it: deployment models and the engine run it as a layer of its own.
- All three train on the CPU and the GPU, save in `.slett` format version 7, and run in deployment
  models in every precision and in the engine; batch normalization after a transposed convolution
  is folded into it. Float deployment models run transposed convolutions through the same kernels
  as training, integer ones through the engine's pass.

`Examples/Segmentation.c` segments synthetic images of circles, squares and triangles with a U-Net
of transposed convolutions (or bilinear upsampling, or layer normalization): 12 epochs take 15 s on
an RTX 4050 Laptop GPU and reach a mean intersection over union of 0.88, which its INT8 model keeps
(see [Performance](#performance)).

### Importing models: ONNX and PyTorch

`spingalett_import_onnx(path)` (or `_from_memory`) turns an ONNX model into a network that trains,
saves and deploys like any other: convolutions and transposed convolutions with groups, Gemm and
MatMul, max, average and global average pooling, batch normalization, layer normalization (over a
vector, or over a map's channels between Transposes to channels last and back, as LayerNorm2d
modules export it), Resize and Upsample by integer factors (nearest, or linear between the cells'
centres), Add, Concat along the channels, Relu, Sigmoid, Tanh, LeakyRelu, Softmax, Flatten and
flattening Reshape, Identity and Dropout. The reader of the protocol buffer format is the library's
own; an operator it does not have, or a mode of one that it does not compute (a Resize with aligned
corners, say), is named in the error. Spingalett works channels-last, so an ONNX input of shape
`[N, C, H, W]` becomes an input layer of `H x W x C`: transpose NCHW images before feeding them
(`x.transpose(0, 2, 3, 1)` in NumPy), and maps come out channels last too.
Weights are reordered on import, including the columns of a dense layer after a flattened map, so
the network computes what the model does. `ModelTool import model.onnx model.slett` converts a
file; models exported from PyTorch with either exporter agree with PyTorch within 1e-7. Weights kept
in an external file next to the model (`torch.onnx.export` of models over 2 GB) are read from the
model's folder when importing from its path. Files are mapped rather than read, and each tensor goes
into its layer in one pass: on an i7-12650H a 100 MB ResNet-50 imports in 0.09 s and a 500 MB model
in 0.17 s.

PyTorch weights load into a network built with the same layers:
`spingalett_load_pytorch(net, "model.pt", NULL, 0)` reads a state dict saved with `torch.save`
(or a checkpoint dictionary holding one) or a `.safetensors` file. Modules are matched to layers
with parameters in order (the state dict's order, or the natural order of the names in
safetensors files, which sort them; a list of module names gives any order), shapes are checked
and the network is left unchanged when anything does not fit. `ConvTranspose2d` modules (weight
`[in, out / groups, kh, kw]`) load into transposed convolutions, `LayerNorm` modules over the
channels into layer normalizations. The pickle inside `.pt` files is interpreted without running
any of it: only dictionaries of tensors are understood (tensors that are views, such as a
transposed weight, load as well).

```c
NeuralNetwork *net = spingalett_import_onnx("resnet.onnx");      /* NULL on error */
float out[10];
predict(.net = net, .inputs = image_hwc, .sample_count = 1, .outputs = out);
```

### Inspecting a network

`SpingalettNetwork` is an opaque handle: its layout is private, and accessors describe it.

```c
uint32_t layers = spingalett_layer_count(net);              /* the input layer included */
SpingalettNetworkLayer d;
spingalett_network_layer(net, 1, &d);   /* type, shape, outputs, activation, dropout, window, counts */

float *w = malloc(d.weight_count * sizeof(float));
spingalett_get_parameters(net, 1, PARAM_WEIGHTS, w, d.weight_count);               /* feeding layer 1 */
spingalett_set_parameters(net, 1, PARAM_WEIGHTS, w, d.weight_count);
```

Parameters come as `SPINGALETT_PARAM_WEIGHTS`, `SPINGALETT_PARAM_BIASES`,
`SPINGALETT_PARAM_WEIGHT_GRADIENTS`, `SPINGALETT_PARAM_BIAS_GRADIENTS` and, for batch normalization,
`SPINGALETT_PARAM_RUNNING_MEAN` and `SPINGALETT_PARAM_RUNNING_VARIANCE`. Dense weights are
`outputs x inputs` (row `j` holds the weights into unit `j`); convolution weights are
`filters x kernel_h x kernel_w x (input channels / groups)`; batch normalization has gamma as
weights and beta as biases, one per channel. Also available: `spingalett_input_size()`,
`spingalett_output_size()`, `spingalett_parameter_count()`, `spingalett_network_loss()` and
`spingalett_optimizer_steps()`.

### Training

`spingalett_train()` takes a `SpingalettTrainArgs` struct. Zero-valued fields use the defaults
below.

| Field | Default | Description |
|---|---|---|
| `inputs`, `targets`, `sample_count` | required | Row-major arrays of `sample_count` samples |
| `device_inputs`, `device_targets` | none | Either or both as data sets in the GPU's memory instead (see [The GPU](#the-gpu)) |
| `epochs` | required | Number of passes over the data |
| `training_strategy` | `SPINGALETT_STRATEGY_SAMPLE` | `SPINGALETT_STRATEGY_SAMPLE` (one step per sample), `SPINGALETT_STRATEGY_FULL_BATCH`, `SPINGALETT_STRATEGY_SMALL_BATCH` |
| `batch_size` | 32 | Mini-batch size |
| `do_not_shuffle` | false | Per-sample and mini-batch training reshuffle the samples every epoch unless set |
| `optimizer_type` | `SPINGALETT_OPTIMIZER_SGD` | `SGD`, `MOMENTUM`, `RMSPROP`, `ADAM`, `ADAMW` |
| `learning_rate` | 0.01 | Base learning rate |
| `momentum` | 0.9 | Momentum coefficient |
| `beta1`, `beta2`, `epsilon` | 0.9, 0.999, 1e-8 | Adam/RMSProp moment parameters; epsilon is added to the square root of the second moment, as in PyTorch |
| `weight_decay` | 0 | L2 penalty; decoupled (`w *= 1 - lr * wd`) for AdamW. Biases are not decayed |
| `max_grad_norm` | 0 (off) | Clip the global L2 norm of every step's gradient |
| `lr_scheduler`, `lr_scheduler_data` | none | Learning-rate schedule, see below |
| `reset_optimizer` | false | Clear moment estimates and the step counter before training |
| `nan_check_interval` | 0 (off) | Stop when weights become NaN/Inf, checked every N epochs |
| `report_interval` | 0 (off) | Log the training loss every N epochs |
| `callback`, `callback_interval`, `callback_data` | none, 1, NULL | `bool cb(SpingalettNetwork *, const SpingalettTrainProgress *, void *callback_data)`; return `true` to stop |
| `val_inputs`, `val_targets`, `val_count` | none | Validation set, evaluated after every epoch (`device_val_inputs`, `device_val_targets`: in the GPU's memory) |
| `monitor` | `SPINGALETT_MONITOR_AUTO` | Quantity that selects the best epoch: validation loss if there is validation data, else training loss; or `SPINGALETT_MONITOR_TRAIN_LOSS`, `SPINGALETT_MONITOR_VAL_LOSS`, `SPINGALETT_MONITOR_VAL_ACCURACY` |
| `early_stopping_patience`, `early_stopping_min_delta` | 0 (off), 0 | Stop after this many epochs without an improvement larger than `min_delta` |
| `restore_best_weights` | false | End with the weights and biases of the best epoch, kept in memory |
| `autosave_mode`, `autosave_interval`, `autosave_path` | off | Periodic checkpoints (`SPINGALETT_AUTOSAVE_OVERWRITE` or `SPINGALETT_AUTOSAVE_NEW_FILES`, which appends `_epoch_N`) |
| `autosave_precision`, `autosave_do_not_save_optimizer` | FP32, false | Checkpoint format |
| `blas_num_threads` | 0 (auto) | OpenBLAS threads during training, see [Backends](#backends-and-threading) |
| `augment_shift`, `augment_flip` | 0, false | Image augmentation (the input layer has a height and width): each training sample is shifted by up to `augment_shift` cells along each axis, zeros shifted in, and with `augment_flip` mirrored left to right half of the time; drawn anew for every sample of every step, the same on any number of threads |

The reported loss is averaged over the samples of an epoch: the sum of squared errors per sample
for MSE, and categorical (softmax) or binary (sigmoid) cross-entropy otherwise. Cross-entropy
requires a softmax or sigmoid output layer; `spingalett_train()` rejects other combinations.

Optimizer state (moment estimates and the step counter used for Adam's bias correction) is stored
in the network, so training can be resumed by calling `spingalett_train()` again or after loading a
checkpoint that includes the optimizer state.

`spingalett_train()` returns a `SpingalettTrainReport`: `status` (`SPINGALETT_TRAIN_COMPLETED`,
`SPINGALETT_TRAIN_EARLY_STOPPED`, `SPINGALETT_TRAIN_INTERRUPTED` by the callback,
`SPINGALETT_TRAIN_DIVERGED` on NaN/Inf, `SPINGALETT_TRAIN_NO_DATA` from a generator, or
`SPINGALETT_TRAIN_FAILED` with the reason in `spingalett_last_error_message()`), `epochs_run`, the
last epoch's training loss and validation metrics, and the best epoch with its value.

### Validation and early stopping

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
### Learning-rate schedules

```c
typedef float (*LRSchedulerFn)(size_t epoch, size_t total_epochs, float initial_lr,
                                        void *user_data);
```

The scheduler runs before every epoch; `epoch` counts the epochs already completed (0 for the
first). Four schedules are built in and take an optional `SpingalettLRScheduleParams` as
`user_data`:

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

Reduce on plateau works with or without a schedule: with `.lr_plateau_patience = 5` and
`.lr_plateau_factor = 0.5f`, five epochs without improvement of the monitored value (`monitor`:
the validation loss when there is validation data) halve the learning rate for the rest of the run,
and five more halve it again, never below `.lr_plateau_min_lr`. Label smoothing
(`.label_smoothing = 0.1f`) trains classifiers on targets moved that far towards the uniform
distribution.

### Data generators

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

### Custom training loops and losses

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

### Data sets

```c
SpingalettDataset train_set, val_set;
spingalett_load_idx("train-images-idx3-ubyte", "train-labels-idx1-ubyte", 10, &train_set);
spingalett_dataset_shuffle(&train_set);                 /* optional, uses spingalett_seed() */
spingalett_dataset_split(&train_set, 5000, &val_set);   /* last 5,000 samples -> val_set */
/* ... train(.inputs = train_set.inputs, .targets = train_set.targets,
                        .sample_count = train_set.count, ...) */
spingalett_dataset_free(&train_set);
spingalett_dataset_free(&val_set);
```

`spingalett_load_idx()` reads the IDX format used by MNIST: unsigned-byte samples of any shape
are flattened and scaled to [0, 1] (float and double files are read unchanged), and labels are
one-hot encoded. `spingalett_load_csv(path, target_columns, num_classes, &d)` reads numeric,
comma-separated files; a non-numeric first line is skipped as a header, the last `target_columns`
columns are the targets, and with `num_classes > 0` a single label column is one-hot encoded.
`spingalett_load_cifar(paths, count, num_classes, &d)` reads CIFAR binary batches one after the
other as 32 x 32 x 3 channels-last images in [0, 1]: `num_classes` 10 for CIFAR-10, 100 or 20 for
the fine or coarse labels of CIFAR-100.

### Data set files

`.slettd` is Spingalett's own data set format: binary, compact and loaded straight into a
`SpingalettDataset`. By default every stream is stored in the smallest encoding that keeps all
values exact (8-bit `q / 255` for image data, IEEE half, or float32; one-hot targets as class
indices) and compressed with adaptive context models that learn which earlier values predict the
next, such as the pixel above in an image: a binary range coder, or for dense data such as
photographs an rANS coder of half-bytes that decodes two to three times as fast. Lossy FP16, BF16
and per-feature 8-bit encodings are available on request. Files consist of independently
decodable chunks with CRC-32 checksums, and can record the input shape, class names and further
sets of targets for the same samples (CIFAR-100's fine and coarse labels, for example).

```c
spingalett_save_dataset(&train_set, "mnist-train", NULL);         /* writes mnist-train.slettd */
SpingalettDataset d = {0};
spingalett_load_dataset("mnist-train.slettd", &d);                 /* bit-identical to train_set */

SpingalettDatasetReader *r = spingalett_dataset_open("mnist-train.slettd", true);   /* shuffled */
train(.net = net, .training_mode = MODE_GENERATOR_FUNCTION,
      .generator = spingalett_dataset_generator, .generator_data = r, .epochs = 10,
      .training_strategy = STRATEGY_SMALL_BATCH, .batch_size = 128);
spingalett_dataset_close(r);
```

A file can be used in three ways, chosen by its size against the memory at hand:

| Way | Memory | Use it when |
|---|---|---|
| `spingalett_load_dataset()`, then `spingalett_train()` on the arrays | 4 bytes per value | the data fits in memory as float |
| a reader with `.in_memory = true` (`spingalett_dataset_open_ex`) | 1 byte per 8-bit value, 2 per FP16 | it fits in its compact form; passes shuffle all samples, as with arrays |
| a streaming reader (`spingalett_dataset_open`) | a few chunks of about 1 MB | it does not fit; passes shuffle the chunks, and the samples within each chunk |

Loading decodes the chunks in parallel with OpenMP. Readers convert values to float a batch at a
time. A streaming reader decodes the next chunks on a background thread when a processor is free
for it, and otherwise decodes several chunks at a time on the OpenMP threads when it needs them,
since a thread competing with the OpenMP threads for the processors would stall them; the samples
come in the same order either way. With `spingalett_dataset_open_u8()`, 8-bit data already in
memory trains through a reader without a float copy. One epoch of a small CNN on CIFAR-10 (two
convolutions, 4 threads, Xeon @ 2.1 GHz, 4 vCPUs, medians):

| Source of the 50,000 training images | Memory | Time per epoch |
|---|---:|---:|
| float arrays | 616 MB | 4.5 s |
| `.slettd` in memory (8-bit) | 154 MB | 4.6 s |
| `.slettd` streamed | about 10 MB | 5.5 s (0.8: 14.7 s) |

Saving is parallel too: chunks are compressed side by side and written in order (CIFAR-10's
training set: 2.2 s on 4 threads, loading 1.1 s against 2.7 s in 0.8).

| Training set | float32 | Original files | `gzip -9` / `xz -9` of them | `.slettd` (lossless) |
|---|---:|---:|---:|---:|
| MNIST (60,000 images and labels) | 190.6 MB | 47.1 MB (IDX) | 9.7 / 7.9 MB (images only) | 7.8 MB |
| CIFAR-10 (50,000 images and labels) | 616.4 MB | 153.7 MB (binary batches) | 141.7 / 116.3 MB | 112.1 MB |

The shape and class names travel with a `SpingalettDataset` (`height`, `width`, `channels`,
`class_names`, filled by the IDX and CIFAR readers and by `.slettd` files that record them; set
names with `spingalett_dataset_set_class_names()`). Further sets of targets are saved through
`SpingalettDatasetSaveOptions.extra_targets` and loaded with
`spingalett_load_dataset_targets(path, set, &d)` or `SpingalettDatasetReaderOptions.target_set`;
`spingalett_dataset_info()` and `spingalett_dataset_class_name()` describe a reader's file.
`spingalett_load_dataset_from_memory()` reads a file image already in memory.

`Bin/DatasetTool` converts data sets and inspects files:

```sh
DatasetTool idx train-images-idx3-ubyte train-labels-idx1-ubyte mnist-train.slettd
DatasetTool csv data.csv 1 3 iris.slettd             # last column: a label of 3 classes
DatasetTool cifar cifar10-train.slettd data_batch_{1,2,3,4,5}.bin
DatasetTool cifar cifar100-train.slettd train.bin --cifar100      # fine and coarse labels
DatasetTool images photos/ photos.slettd --size 64x64 --rgb       # one subfolder per class
DatasetTool info cifar10-train.slettd                # encodings, shape, sets of targets, class names
DatasetTool verify cifar10-train.slettd              # decodes every chunk, checks every checksum
```

`cifar` reads the class names from `batches.meta.txt` (CIFAR-10) or `fine_label_names.txt` and
`coarse_label_names.txt` (CIFAR-100) next to the batches when they are there. `images` reads PNG,
JPEG, BMP, TGA, GIF, PSD, HDR, PIC and PNM files (with stb_image), names the classes after the
subfolders, and resizes by area averaging. The format is specified in
[docs/DatasetFormat.md](docs/DatasetFormat.md).

### Inference

`spingalett_forward(net, input)` evaluates one sample and returns a pointer to the output layer
inside the network; the buffer is overwritten by the next call. `spingalett_predict()` evaluates
many samples at once with matrix-matrix products and is the faster choice for more than a handful of
inputs:

```c
float *outputs = malloc(count * output_size * sizeof(float));
predict(.net = net, .inputs = inputs, .sample_count = count,                     /* false on error */
        .outputs = outputs);
```

`spingalett_evaluate()` returns the mean loss (as reported by training) and the accuracy over a data
set:

```c
EvalMetrics m = evaluate(.net = net, .inputs = x, .targets = y,
                         .sample_count = n);
```

Both take data sets in the GPU's memory (`device_inputs`, and `device_targets` for
`spingalett_evaluate()`) in place of arrays; see [The GPU](#the-gpu). Dropout is not applied during
inference.

### Saving and loading

```c
save_spingalett(.net = net, .filename = "model.slett", .precision = PRECISION_FP16,
                .do_not_save_optimizer = true);
NeuralNetwork *net = load_spingalett("model.slett");
```

Models are stored in `.slett` files; the extension is appended when the filename has none, and
files written under the former `.nn` name load unchanged. Weights are stored in the selected
precision; biases and, unless disabled, the optimizer state in float:

| Precision | Storage per weight | Notes |
|---|---|---|
| `SPINGALETT_PRECISION_FLOAT32` | 4 bytes | Lossless |
| `SPINGALETT_PRECISION_FP16`, `SPINGALETT_PRECISION_BFLOAT16` | 2 bytes | Round to nearest even |
| `SPINGALETT_PRECISION_INT8`, `SPINGALETT_PRECISION_INT4` | 1 byte, 4 bits | Symmetric, one scale per weight row (output unit) |
| `SPINGALETT_PRECISION_INT2` | 2 bits | Ternary {-scale, 0, +scale}, one scale per row |

The same bytes can be produced and read in memory:

```c
size_t size;
void *image = spingalett_save_to_memory(net, PRECISION_INT8, false, &size);
                                                              /* released by spingalett_free() */
NeuralNetwork *copy = load_spingalett_from_memory(image, size);
```

The file format is versioned and specified in [docs/ModelFormat.md](docs/ModelFormat.md):
little-endian, with 16-byte aligned sections and CRC-32 checksums, so that a file image can be
executed in place (see [Deployment](#deployment)). Networks with convolution or pooling layers are
saved in version 4 (`SPINGALETT_FORMAT_VERSION`), which adds the layers' kinds and shapes; networks
of dense layers only are still saved in version 3, so that the engines of earlier releases run them.
Files of versions 1 and 2 remain loadable. `spingalett_load()` returns `NULL` for missing, truncated
or corrupt files and for files with an invalid header.

### Backends and threading

```c
spingalett_set_compute_mode(COMPUTE_OPENBLAS);             /* or SINGLE_THREADED, OPENMP, CUDA, VULKAN */
spingalett_set_num_threads(8);                             /* 0 = runtime default */
```

Full-batch and mini-batch training and `spingalett_predict()` process samples in chunks of up to
2048 as matrix-matrix products (fewer when a sample's activations are large, so that a chunk stays
near 64 MB); per-sample training and `spingalett_forward()` of dense networks use matrix-vector
kernels.

-  `SPINGALETT_COMPUTE_SINGLE_THREADED` uses the built-in kernels: AVX-512 or AVX/FMA when the
  compiler targets them, portable C otherwise.
-  `SPINGALETT_COMPUTE_OPENMP` uses the same kernels on all threads; operations that are too small
  to benefit run serially. Threads split matrix products by tiles of the result (and long
  weight-gradient sums by slots fixed by the shape), so results are the same bits on any number of
  threads and in single-threaded mode.
-  `SPINGALETT_COMPUTE_OPENBLAS` delegates matrix products to OpenBLAS. It is usually on par with
  OpenMP for large batches and slower for small ones. During training, `blas_num_threads = 0` uses a
  single OpenBLAS thread when each call is too small to amortize threading and the configured thread
  count otherwise, and restores the caller's setting afterwards; outside `spingalett_train()`,
  OpenBLAS uses its own thread configuration (`OPENBLAS_NUM_THREADS`).
-  `SPINGALETT_COMPUTE_CUDA` (an NVIDIA GPU) and `SPINGALETT_COMPUTE_VULKAN` (any GPU) run training,
  the custom-loop API, `spingalett_predict()` and `spingalett_evaluate()` on a GPU
  ([The GPU](#the-gpu)); what stays on the CPU runs as with `SPINGALETT_COMPUTE_OPENMP`.
-  A requested backend that was not compiled in falls back to single-threaded with a one-time
  warning (a GPU mode without a usable device: to the CPU, as above).

Results agree across backends up to floating-point rounding. For the duration of
`spingalett_train()`, denormal floats are flushed to zero on the calling thread and the OpenMP
workers; the previous floating-point mode is restored afterwards.

A network must not be used by several threads at the same time. Compute mode, thread count and
logging settings are process-wide; the random generator and the error state are per thread.

### The GPU

With `SPINGALETT_COMPUTE_CUDA` or `SPINGALETT_COMPUTE_VULKAN`, `spingalett_train()` (full-batch and
mini-batch strategies), the custom-loop API (`spingalett_trainer_*`), `spingalett_predict()` and
`spingalett_evaluate()` run on a GPU. The two modes share one executor, and what follows holds for
both:

-  `SPINGALETT_COMPUTE_CUDA` runs on NVIDIA GPUs of compute capability 8.0 or later (Ampere and
  newer: GeForce RTX 30 to 50, A100, H100) with a driver of CUDA 11.0 or later, on the library's own
  kernels (`Src/Gpu/Cuda`): matrix products in single precision and, in bfloat16, on the tensor
  cores, convolutions as implicit products, depthwise convolutions, normalizations, pooling. They are
  compiled to PTX when the library is built and kept in it compressed; the driver compiles a kernel
  for the GPU when it is first used and keeps it in its cache (`~/.nv/ComputeCache`). Neither the
  CUDA toolkit nor cuBLAS or cuDNN is needed to build or to run. `spingalett_cuda_device()` names the
  GPU, or returns `NULL` without one.
-  `SPINGALETT_COMPUTE_VULKAN` runs through Vulkan compute: NVIDIA, AMD and Intel GPUs on Linux and
  Windows, Apple GPUs through MoltenVK (from the Vulkan SDK, or
  `brew install molten-vk vulkan-loader`). `spingalett_gpu_device()` names the device, or returns
  `NULL` when there is none (it needs Vulkan 1.2 with buffer device addresses).

The library opens the CUDA driver or the Vulkan loader when a mode is first used, so it needs
neither to load or to run on the CPU; without a usable device the mode falls back to the CPU with a
warning. On an NVIDIA GPU the CUDA mode is the faster ([Performance](#performance)).

```c
spingalett_set_compute_mode(COMPUTE_CUDA);                 /* or COMPUTE_VULKAN */
const char *gpu = spingalett_cuda_device();               /* spingalett_gpu_device() for Vulkan */
printf("training on %s\n", gpu ? gpu : "the CPU");
train(.net = net, .inputs = x, .targets = y, .sample_count = n, .epochs = 30,             /* as on the CPU */
      .training_strategy = STRATEGY_SMALL_BATCH, .batch_size = 128);
```

-  Parameters, their gradients and the optimizer's moments stay in GPU memory, and so does the
  network's copy there after `spingalett_train()` returns: functions that read the parameters
  (`spingalett_predict()`, `spingalett_save()`, `spingalett_get_parameters()`, a callback's reads
  during training, and the others) copy them back first, parameters set on the host go to the device
  before the next epoch, and the next `spingalett_train()` of the same batch size and optimizer
  trains on the copy again without copying anything. The network frees it when it is freed, gains a
  layer, trains on the CPU or makes a trainer. The host prepares a batch (gathering, augmentation,
  label smoothing, and writing it into the device's memory where it can) while the GPU trains on the
  one before, and takes the losses back at the end of every epoch.
-  Data sets can stay in the GPU's memory: `spingalett_device_data_new(values, count, size)` copies
  rows of floats there once, and the `device_inputs` and `device_targets` fields of
  `spingalett_train()` (with `device_val_inputs` and `device_val_targets`), `spingalett_predict()`
  and `spingalett_evaluate()` take such sets in place of arrays, either or both. Only the rows'
  indices cross the bus then: the GPU gathers each chunk's rows, augments and smooths them as the
  host would (training on a set gives the bits of training on the host's arrays), and reads them
  where they are when they come in order (inference, full batches). With
  `SPINGALETT_PRECISION_BFLOAT16` a set keeps a bfloat16 copy of its rows, made on first use, half
  its size again. On the CPU those calls copy the rows back first.

  ```c
  SpingalettDeviceData *x = spingalett_device_data_new(images, n, 784), *y = spingalett_device_data_new(labels, n, 10);
  train(.net = net, .device_inputs = x, .device_targets = y, .sample_count = n, .epochs = 30,
        .training_strategy = STRATEGY_SMALL_BATCH, .batch_size = 128, .augment_shift = 2);
  predict(.net = net, .device_inputs = x, .outputs = out, .sample_count = n);
  spingalett_device_data_free(x);
  spingalett_device_data_free(y);
  ```
-  Every kind of layer, activation, loss and optimizer runs on the GPU, with dropout (the same masks
  as on the CPU), gradient clipping and batch normalization. Per-sample training,
  `spingalett_forward()` and deployment models run on the CPU.
-  A trainer (`spingalett_trainer_new()`) made with a GPU mode runs its passes on
  the GPU, one at a time: the forward pass's outputs come back for the caller's loss, and backward
  passes from targets or from the caller's dL/d(output) add up on the device until the step. The
  parameters stay there between passes; functions that read the network (`spingalett_predict()`,
  `spingalett_save()`, `spingalett_get_parameters()` and the others) copy them back first, and
  parameters set on the host go to the GPU before the next forward pass.
-  `spingalett_set_gpu_precision(SPINGALETT_PRECISION_BFLOAT16)` multiplies matrices in bfloat16 on
  the GPU's matrix units (tensor cores: `mma.sync` with CUDA, `VK_KHR_cooperative_matrix` with
  `VK_KHR_shader_bfloat16` with Vulkan), the products added in single precision. The layers' outputs, all but the output layer's (and the
  network's inputs, which the host rounds as it sends them), and their gradients are kept on the GPU
  as bfloat16, the values the products round them to anyway: half the memory and half the bytes
  every pass reads and writes. The passes between products (normalizations, pooling, additions,
  upsampling) compute in single precision, and the parameters, their gradients and the optimizer
  stay in it, as with PyTorch's autocast; products smaller than a block of the matrix units stay in
  single precision unless they read or write bfloat16. Runs stay deterministic. On an RTX 4050
  Laptop GPU ResNet-20 trains 1.7 times as fast as in single precision and reaches the same CIFAR-10
  test accuracy (91.80% after 100 epochs, single precision 91.55%). Vulkan devices without the
  extensions keep single precision.
- Results are deterministic: every sum runs in a fixed order, never through atomics, so a run gives
  the same bits every time on one device and backend. They agree with the CPU's up to rounding: the
  products add in another order, with fused multiply-adds, and batch normalization adds its sums in
  single rather than double precision (the two backends agree with each other the same way).
-  The first products of each shape are timed with a few tile sizes, and the fastest is kept for the
  life of the process: the first `spingalett_train()` of a process takes a few tenths of a second
  longer (ResNet-20 on an RTX 4050 Laptop GPU: 0.3 s), and the very first on a machine some ten
  seconds, while the driver compiles the kernels, which it then keeps on disk (NVIDIA's cache is
  limited in size, which `__GL_SHADER_DISK_CACHE_SIZE` raises). Tiles change the speed, never the
  results; `SPINGALETT_GPU_TUNE=0` estimates them instead.
- Training processes samples in chunks of up to 4,096 that fit in half the GPU's memory; networks
  with batch normalization, which full-batch training normalizes over each chunk, in chunks of up to
  2,048, as on the CPU.
  Inference runs in chunks of at most 32 MB of activations as they are kept (64 samples at least;
  from host arrays 2,048 at most), whose layers' outputs stay in the GPU's cache from one layer to
  the next.
- Where Vulkan lets the host write into all of the GPU's memory (resizable BAR, unified memory), it
  writes each chunk's inputs and the parameters straight into it, instead of the GPU copying them
  from host memory; with CUDA a chunk's inputs are copied on a stream of their own while the GPU
  works on the chunks before. Three chunks are in flight: the host fills one while the GPU works on
  the two before it.
- `SPINGALETT_CUDA_DEVICE=n` picks the n-th device of the CUDA driver's list, `SPINGALETT_GPU_DEVICE=n`
  the n-th of the Vulkan device list instead of the first discrete GPU; `SPINGALETT_GPU_PROFILE=1`
  prints the GPU time per kernel and copy when the process exits (with CUDA timed by events around
  every launch, which add some microseconds to each); `SPINGALETT_GPU_NO_HOST_WRITES=1` has Vulkan
  copy inputs and parameters from host memory, and `SPINGALETT_GPU_NO_BF16_STORAGE=1` keeps
  activations in single precision in its bfloat16 mode.

In Python: `sg.set_compute_mode(sg.ComputeMode.CUDA)` (or `ComputeMode.VULKAN`), `sg.cuda_device()`,
`sg.gpu_device()`, `sg.set_gpu_precision(sg.Precision.BFLOAT16)` and `sg.DeviceData(array)`, which
`spingalett_train()`, `validation_data`, `spingalett_forward()` and `spingalett_evaluate()` take in
place of arrays.

### Reproducibility

`spingalett_seed(seed)` seeds the calling thread's generator, which drives weight
initialization, mini-batch shuffling and dropout. With a fixed seed, training is deterministic,
and dropout masks are identical across backends and thread counts. With the built-in kernels,
the trained weights are bit for bit the same on any number of threads and in single-threaded mode
(a test checks it); OpenBLAS agrees up to rounding.

### Errors and logging

A function that fails returns false, `NULL`, NaN (losses and metrics), `SPINGALETT_NO_LAYER` or a report
whose status is `SPINGALETT_TRAIN_FAILED`, and sets the calling thread's error code and message:

```c
spingalett_clear_error();
NeuralNetwork *net = load_spingalett("model.slett");
if (!net)
    fprintf(stderr, "%s (code %d)\n", spingalett_last_error_message(),
            spingalett_last_error_code());
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
`spingalett_set_log_callback(void (*)(SpingalettLogLevel, const char *))`.
`spingalett_set_log_level()` sets the minimum level and `spingalett_set_verbose(false)` suppresses
everything below warnings.

## Deployment

A `SpingalettNetwork` is built for training: float parameters, and the gradients and optimizer state
it allocates when it first trains (a network loaded or imported only to predict holds its
parameters once). To run a trained network, turn it into a model, a read-only network that keeps its weights in the precision
they are stored in and computes with them:

```c
SpingalettModel *model = spingalett_model_from_network(net, PRECISION_INT8);
                                                  /* or spingalett_model_load("model.slett") */
spingalett_model_predict(model, inputs, count, outputs);         /* batched, multi-threaded */
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

`spingalett_model_predict()` and `spingalett_model_evaluate()` go further on processors with byte
dot-product instructions (AVX-512 VNNI; AVX-VNNI, as in Intel Core processors since the 12th
generation; the Arm dot product extension, as in Apple silicon, Cortex-A76 and later, and AWS
Graviton 2 and later): integer convolutions (with one group) and dense layers run in tiles of 12
pixels or samples against weight rows interleaved for those instructions, every weight read once
for the tile. The release packages carry these kernels and choose them at run time (see
[Performance](#performance) for what they gain).

A model made by the library (`spingalett_model_from_network()`, `spingalett_model_load()`,
`spingalett_model_from_memory()`) prepares what batched prediction runs on once, on the first call
that needs it: FP16 and BF16 weights expanded to float, INT4 and INT2 rows unpacked to bytes,
integer rows interleaved for the tiles. It keeps them, and the last call's buffers, until
`spingalett_model_free()`, so a call on a single sample costs about what `spingalett_model_run()`
does (4 to 12 us for a 784-256-128-10 MLP in any precision, 0.2 us for a tiny one). The prepared
forms take memory next to the image: up to the float size of FP16 layers and four times the packed
size of INT2 layers. A model stays safe to use from several threads at once. Models filled in by
`spingalett_model_init()` over an image of your own prepare on every call. Float layers on one
sample or a few compute through dot products rather than the batched kernels, so their outputs can
differ in the last bits from those of the same samples in a larger batch; integer layers compute
exactly what `spingalett_model_run()` computes, in batches of any size.

Accuracy on the 10,000 test images of MNIST (`ModelTool eval`) and CIFAR-10
(`spingalett_model_evaluate()`):

| Network | FP32 | FP16 | INT8 | INT4 | INT2 |
|---|---:|---:|---:|---:|---:|
| 784-256-128-10, `Examples/MNIST.c` (5 epochs) | 97.86% | 97.86% | 97.87% | 97.71% | 95.15% |
| Model size | 941 KB | 471 KB | 238 KB | 121 KB | 62 KB |
| CNN of `Examples/MNIST_CNN.c` (2 epochs) | 98.89% | 98.89% | 98.89% | 98.75% | 97.74% |
| Model size | 1.69 MB | 844 KB | 424 KB | 213 KB | 108 KB |
| DigitPad's CNN with batch normalization (30 epochs) | 99.58% | 99.58% | 99.56% | 99.51% | 96.76% |
| Model size | 1.88 MB | 937 KB | 471 KB | 237 KB | 120 KB |
| CIFAR-10 network of `Examples/CIFAR10.c` (20 epochs) | 87.79% | 87.79% | 87.82% | 84.87% | 14% |
| Model size | 2.21 MB | 1.10 MB | 555 KB | 280 KB | 143 KB |

INT8 keeps the accuracy of these networks and INT4 nearly so; INT2 (ternary weights without
quantization-aware training) suits small layers. Normalizations are folded into the layer before
them and cost nothing at inference: the deployed CIFAR-10 network has 11 layers where the trained
one has 18.

The 784-512-1000-10 benchmark network on one thread (`Bin/Benchmark`, Intel Xeon @ 2.10 GHz,
Sapphire Rapids), one sample at a time and batched:

| | `spingalett_forward()` | FP32 model | FP16 | BF16 | INT8 | INT4 | INT2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Microseconds per sample | 161 | 150 | 86 | 85 | 28 | 31 | 20 |
| Batched samples per second | | 50,600 | 48,500 | 57,400 | 184,600 | 172,200 | 195,900 |
| Weights | 3.7 MB | 3.7 MB | 1.9 MB | 1.9 MB | 0.9 MB | 0.5 MB | 0.2 MB |

### The runtime

Programs that only run trained models can link the runtime instead of the library.
`libspingalett-runtime`, declared by `Spingalett.Runtime.h`, holds what this section describes:
`.slett` files loaded as models, `spingalett_model_predict()` and `spingalett_model_evaluate()` with
the library's kernels, threads and choice of instruction sets at run time, the engine of
`Spingalett.Inference.h`, errors, logging and the thread settings. It has no networks, training,
data sets, importers or GPU backend, which leaves 510 KB of the full library's 2.0 MB without the
CUDA backend's kernels and 12.6 MB with them (the x86-64 release builds, with the kernels for AVX2 and
AVX-512), and needs only the C library, threads and
OpenMP. It computes the same bits as the library, as fast: its functions are the library's
(`Spingalett.h` declares them through `Spingalett.Runtime.h`), compiled from the same sources with
the same flags; the tests compare the two on models of every kind of layer in every precision, and
alternated in one process on an Intel Core i7-12650H they predict batches of the benchmark's MLP,
the MNIST CNN and ResNet-20, in FP32 and INT8, within 1.5% of each other either way on one thread
and on eight (2.5% on all sixteen, which mix the processor's two kinds of cores). A program written
for the runtime also builds and runs with the full library.

```c
#define SPINGALETT_SHORT_NAMES                         /* COMPUTE_OPENMP, PRECISION_INT8, ... */
#include <Spingalett/Spingalett.Runtime.h>

SpingalettModel *model = spingalett_model_load("model.slett");
if (!model) { fprintf(stderr, "%s\n", spingalett_last_error_message()); return 1; }
spingalett_set_compute_mode(COMPUTE_OPENMP);           /* every core */
spingalett_model_predict(model, inputs, count, outputs);
spingalett_model_free(model);
```

A build with `-DSPINGALETT_RUNTIME_ONLY=ON` makes the runtime alone, without `glslc` or the Vulkan
headers; other builds make it next to the library (`SPINGALETT_RUNTIME`). It installs as the CMake
target `Spingalett::runtime` and the pkg-config package `spingalett-runtime`, with `RunModel`
(`Examples/Runtime/RunModel.c`), which describes a model and times it, or predicts the samples of a
file of floats:

```
RunModel model.slett                        the layers, then the time of one sample and of batches
RunModel model.slett inputs.f32 out.f32     the outputs of raw float samples (classes without out.f32)
```

The runtime reads `.slett` files of format versions 3 to 7, everything Spingalett 0.5 and later
writes. Files of versions 1 and 2 (0.4 and earlier) hold networks rather than images; it refuses
them with `SPINGALETT_ERR_FORMAT_VERSION`, and `ModelTool convert` rewrites them.

| | Library | Runtime | Engine |
|---|---|---|---|
| Header | `Spingalett.h` | `Spingalett.Runtime.h` | `Spingalett.Inference.h` |
| Training, networks, data sets, importers, the GPU | yes | | |
| `.slett` models from FP32 to INT2 | yes | yes | yes |
| Batches, threads, kernels chosen at run time | yes | yes | one sample at a time |
| Heap, files | yes | yes | |
| Size | 12.6 MB (x86-64; 2.0 MB without CUDA) | 510 KB (x86-64) | about 12 KB of code on a Cortex-M4 |

### Running in place

A model is a view of a `.slett` image (format version 3 or later: 4 with convolutions, 6 for graphs, 7 with
transposed convolutions, upsampling or layer normalization).
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
at install time. The wheels carry the library (Linux x86-64 and AArch64, Windows, macOS):

```bash
pip install spingalett                                  # or, from a checkout:
pip install ./Bindings/Python && export SPINGALETT_LIBRARY=$PWD/Bin/libspingalett.so
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

resnet = sg.Network.from_torch(torch_module, torch.rand(1, 3, 32, 32))   # through ONNX, channels last
```

See [Bindings/Python/README.md](Bindings/Python/README.md) for the full API.

## C++

`Spingalett/Spingalett.hpp` is a header-only C++23 interface over the same library: the library's
objects as move-only owners (`spingalett::Network`, `Model`, `Dataset`, `DeviceData`), data as
`std::span`, errors as `std::expected<T, spingalett::Error>` (the library's code and message; nothing
throws), scoped enums, and a fluent `Builder` that makes the network in `build()`:

```cpp
#include <Spingalett/Spingalett.hpp>
namespace sg = spingalett;

auto net = sg::Builder(sg::Loss::CrossEntropy)
               .input(28, 28, 1)
               .conv2d({.filters = 32, .kernel = 3, .padding = 1, .act = sg::Activation::Relu,
                        .init = sg::Init::He})
               .max_pool2d(2)
               .dense(128, sg::Activation::Relu, sg::Init::He)
               .dense(10, sg::Activation::Softmax, sg::Init::Xavier)
               .build();
if (!net) {
    std::cerr << net.error().message << '\n';     /* the library's message: what failed */
    return 1;
}

sg::TrainOptions o;
o.epochs = 2;
o.strategy = sg::Strategy::MiniBatch;
o.batch_size = 128;
o.optimizer = sg::Optimizer::Adam;
o.learning_rate = 1e-3f;
sg::Result<sg::TrainReport> report = net->train(images, labels, o);   /* std::span<const float> */
sg::Result<std::vector<float>> outputs = net->predict(images);
sg::Result<sg::Model> int8 = net->to_model(sg::Precision::Int8);
```

`Builder::from({a, b})` names the inputs of the next layer and `last()` gives the index of the last
one, for residual blocks and branches; `add_layers()` and `concat_layers()` take their inputs
directly. `TrainOptions::raw` carries every other field of `SpingalettTrainArgs`, and `raw()` gives the
C object of a wrapper for whatever the wrapper does not cover.

## DigitPad demo

[Apps/DigitPad](Apps/DigitPad) is a desktop app in which you draw a digit with the mouse and a
Spingalett network classifies it as you draw. Its convolutional network with batch normalization,
trained with on-the-fly augmentation through a data generator, reaches 99.58% MNIST test accuracy
(99.45% on randomly distorted test digits). The directory contains
the app, the trainer and scripts that package the app and the model as a self-contained Linux
AppImage and as a Windows zip; every release attaches both.

![DigitPad](Apps/DigitPad/screenshot.png)

## Performance

`Examples/Benchmark.c` measures a 784-512-1000-10 network (925K parameters, ReLU, softmax with
cross-entropy, Adam) on 20,000 synthetic samples: full-batch training (5 epochs), mini-batch
training (batches of 64, one epoch) and batched inference; then the convolutional network of
`Examples/MNIST_CNN.c` (422K parameters) on 10,000 synthetic images: one epoch of mini-batches of
128 with AdamW, and inference, without and with batch normalization after each convolution and the
hidden dense layer; and ResNet-20 (`Examples/CIFAR10.c resnet20`, 273K parameters, 54 layers) on
4,096 synthetic 32 x 32 x 3 images: one epoch of mini-batches of 128 with SGD and momentum, and
inference; and the U-Net of `Examples/Segmentation.c` (118K parameters, 28 layers: transposed
convolutions, concatenations, three sigmoid outputs a pixel) on 1,024 synthetic 64 x 64 x 3 images:
one epoch of mini-batches of 32 with AdamW, and inference; and a MobileNet-style network (69K
parameters: a 3 x 3 convolution and four depthwise-separable blocks of 64 to 256 channels, each
convolution normalized) on 4,096 synthetic 32 x 32 x 3 images: one epoch of mini-batches of 128 with
SGD and momentum, and inference. `Examples/benchmark_pytorch.py` runs the same workloads in PyTorch. Run them with `Bin/Benchmark [threads]` and
`python Examples/benchmark_pytorch.py [threads]`.

Samples per second on a laptop (Intel Core i7-12650H: 6 performance and 4 efficiency cores, AVX2
and AVX-VNNI, no AVX-512), medians of three interleaved runs. Spingalett 0.10 is built with GCC 16
and uses its built-in kernels (no BLAS library); PyTorch 2.14.1 is the CPU build from PyPI (Intel
MKL and oneDNN). Against 0.9, measured in the same runs, 0.10 trains the convolutional networks 13
to 19% faster and runs them 15 to 27% faster, and leaves the fully connected network as it was;
0.11 to 0.13 leave all of them as they were on the CPU (within 4% of the release before, either
way, in interleaved runs). The U-Net, new in 0.13, is measured with 0.13, the MobileNet-style network,
new in 0.14, with 0.14:

| Fully connected network | Threads | Full batch | Mini-batch 64 | Inference |
|---|---:|---:|---:|---:|
| Spingalett | 1 | 25,700 | 20,200 | 67,100 |
| PyTorch | 1 | 24,000 | 10,800 | 60,800 |
| Spingalett (OpenMP) | 8 | 114,600 | 52,900 | 287,400 |
| PyTorch | 8 | 73,000 | 26,100 | 168,200 |

| Convolutional network | Threads | Training | Inference |
|---|---:|---:|---:|
| Spingalett | 1 | 3,090 | 11,180 |
| PyTorch | 1 | 2,260 | 4,370 |
| Spingalett (OpenMP) | 8 | 14,590 | 50,860 |
| PyTorch | 8 | 6,210 | 13,020 |
| **With batch normalization** | | | |
| Spingalett | 1 | 2,640 | 11,380 |
| PyTorch | 1 | 1,770 | 3,340 |
| Spingalett (OpenMP) | 8 | 11,030 | 55,650 |
| PyTorch | 8 | 4,630 | 10,160 |

| ResNet-20 | Threads | Training | Inference |
|---|---:|---:|---:|
| Spingalett | 1 | 377 | 1,387 |
| PyTorch | 1 | 342 | 900 |
| Spingalett (OpenMP) | 8 | 1,584 | 6,402 |
| PyTorch | 8 | 896 | 2,323 |

| U-Net, 64 x 64 | Threads | Training | Inference |
|---|---:|---:|---:|
| Spingalett | 1 | 146 | 511 |
| PyTorch | 1 | 134 | 292 |
| Spingalett (OpenMP) | 8 | 598 | 2,467 |
| PyTorch | 8 | 365 | 754 |

| MobileNet-style network | Threads | Training | Inference |
|---|---:|---:|---:|
| Spingalett | 1 | 797 | 3,762 |
| PyTorch | 1 | 589 | 1,648 |
| Spingalett (OpenMP) | 8 | 2,765 | 16,010 |
| PyTorch | 8 | 1,377 | 3,808 |

Spingalett trains the convolutional network 1.4 to 2.3 times as fast as PyTorch and runs it 2.6 to
3.9 times as fast; with batch normalization 1.5 to 2.4 times and 3.4 to 5.5 times, since the
normalization runs in the convolution's epilogue at inference. ResNet-20 trains 1.1 to 1.8 times as
fast and runs 1.5 to 2.8 times as fast, the U-Net 1.1 to 1.6 times and 1.7 to 3.3 times, the
MobileNet-style network 1.35 to 2 times and 2.3 to 4.2 times. For the
fully connected network Spingalett trains mini-batches 1.9 to 2 times as fast, full batches 1.1 to
1.6 times and infers 1.1 to 1.7 times as fast. The indirect convolution kernels of 0.10 account for the gains over 0.9: built without them
(`-DSPINGALETT_NO_DIRECT_CONV`), ResNet-20 trains at 234 and 986 samples per second and infers at 958
and 4,798.

On the laptop's GPU (NVIDIA GeForce RTX 4050 Laptop GPU, 6 GB, driver 610.43), Spingalett 1.1 runs
the same workloads with `SPINGALETT_COMPUTE_CUDA` and `SPINGALETT_COMPUTE_VULKAN` (`Bin/Benchmark
gpu`), in single precision and in bfloat16 (`spingalett_set_gpu_precision()`), against PyTorch 2.14.1
with CUDA 13.0, cuBLAS and cuDNN 9.24 (`python Examples/benchmark_pytorch.py --cuda`, with PyTorch's
default TF32 convolutions, `--cuda-fp32`, and `--cuda-bf16`: autocast to bfloat16), medians of three
interleaved runs. The first three columns keep the data in GPU memory from the start: Spingalett's
data sets (`spingalett_device_data_new()`), PyTorch's tensors. The last two take it from host memory:
Spingalett's arrays, each chunk copied over while the GPU works on the ones before, and PyTorch's
pinned tensors, each batch copied as it is used and the outputs of inference copied back
(`--host-data`). Inference is timed after a first call (Spingalett) or a warm-up step (PyTorch):

| On the GPU, samples/s | CUDA, data on GPU (FP32 / bf16) | Vulkan, data on GPU (FP32 / bf16) | PyTorch, data on GPU (TF32 / FP32 / bf16) | CUDA, host data (FP32 / bf16) | PyTorch, host data (TF32 / bf16) |
|---|---:|---:|---:|---:|---:|
| ResNet-20, training | 11,440 / 22,020 | 10,330 / 18,310 | 8,402 / 7,222 / 12,210 | 11,560 / 22,000 | 8,337 / 12,020 |
| ResNet-20, inference | 40,570 / 76,220 | 37,140 / 65,100 | 19,630 / 19,260 / 31,160 | 41,060 / 77,170 | 19,100 / 30,180 |
| Convolutional network, training | 114,700 / 208,800 | 108,300 / 165,800 | 56,000 / 59,150 / 98,210 | 114,900 / 209,600 | 54,690 / 69,970 |
| Convolutional network, inference | 375,700 / 978,300 | 331,300 / 501,600 | 130,600 / 141,400 / 236,800 | 380,600 / 964,200 | 126,400 / 223,200 |
| With batch normalization, training | 79,330 / 150,300 | 74,770 / 119,700 | 47,870 / 48,000 / 83,760 | 79,740 / 151,800 | 47,760 / 62,950 |
| With batch normalization, inference | 374,900 / 775,400 | 312,300 / 462,100 | 106,500 / 113,300 / 195,700 | 368,400 / 784,900 | 103,500 / 184,900 |
| Fully connected network, mini-batch 64 | 361,700 / 531,400 | 313,100 / 443,900 | 76,060 / 80,250 / 68,910 | 380,900 / 546,800 | 67,440 / 56,700 |
| Fully connected network, full batch | 1,255,000 / 3,188,000 | 1,158,000 / 2,864,000 | 1,045,000 / 1,045,000 / 2,541,000 | 1,157,000 / 2,972,000 | 833,800 / 1,551,000 |
| Fully connected network, inference | 3,808,000 / 9,100,000 | 3,272,000 / 8,061,000 | 2,887,000 / 2,952,000 / 6,211,000 | 2,806,000 / 6,549,000 | 1,680,000 / 2,445,000 |
| U-Net, training | 4,243 / 7,925 | 4,008 / 6,855 | 3,289 / 2,998 / 4,975 | 4,143 / 7,633 | 3,232 / 4,345 |
| U-Net, inference | 14,260 / 29,010 | 14,420 / 23,480 | 6,317 / 6,296 / 11,660 | 14,990 / 28,430 | 5,905 / 10,860 |
| MobileNet-style network, training | 14,860 / 32,800 | 14,210 / 29,830 | 9,780 / 8,573 / 22,680 | 14,760 / 32,260 | 9,679 / 22,050 |
| MobileNet-style network, inference | 111,700 / 158,700 | 101,300 / 128,200 | 19,410 / 17,780 / 37,630 | 116,000 / 158,900 | 19,130 / 36,500 |

Spingalett is ahead of PyTorch in every workload, on either backend and in either setting. With the
data in GPU memory, on CUDA in single precision, it trains ResNet-20 1.36 times as fast as PyTorch
with TF32 (1.58 times as fast as PyTorch in single precision) and runs it 2.1 times as fast, trains
the convolutional networks 1.7 to 2 times as fast and runs them 2.9 to 3.5 times as fast, the U-Net
1.3 and 2.3 times, the MobileNet-style network (PyTorch in channels-last, its faster layout for it)
1.5 and 5.8 times, and the fully connected network 1.2 times as fast in full batches, 4.8 times in
mini-batches, and infers 1.3 times as fast. In bfloat16, against PyTorch's autocast: ResNet-20 1.8
and 2.45 times, the convolutional networks 1.8 to 2.1 and 4 to 4.1 times, the U-Net 1.6 and 2.5
times, the MobileNet-style network 1.45 and 4.2 times, the fully connected network 1.25 times in full
batches, 7.7 times in mini-batches and 1.5 times in inference. From host arrays the margins are as
wide or wider (PyTorch copies each batch as it uses it; Spingalett copies a chunk while the GPU works
on the ones before): the fully connected network, whose inference the bus binds, infers 1.7 times as
fast in single precision and 2.7 times in bfloat16.

The CUDA backend is the faster on this GPU: against the Vulkan backend (1.1, the same executor) it
trains 1.05 to 1.16 times as fast in single precision and 1.1 to 1.26 times in bfloat16, and infers
1.1 to 1.2 times as fast in single precision (the U-Net alike) and 1.13 to 1.95 times in bfloat16,
where its products of the tensor cores, its kernel for a network's first convolution (`dconv.cu`)
and its depthwise convolutions pull ahead. Against 0.14 (the Vulkan backend, see the
[CHANGELOG](CHANGELOG.md)), both backends infer networks with batch normalization faster since the
normalization runs in the epilogue of the product it reads (the convolutional network with batch
normalization 1.3 times as fast in bfloat16 on Vulkan), and inference from host arrays keeps three
chunks in flight. In single precision Spingalett trains ResNet-20 on the GPU 7.2 times as fast as
on the eight threads of the CPU.

0.9 took its time out of the calls these workloads do not measure: small batches and single
samples, files, data sets and Python. Same VM, 4 threads, 0.8 against 0.9 (see the
[CHANGELOG](CHANGELOG.md) for more):

| Call | 0.8 | 0.9 |
|---|---:|---:|
| `spingalett_model_predict()`, one sample, 784-256-128-10 MLP, FP32 / FP16 / INT8 | 63 / 403 / 10.5 us | 12 / 8 / 4.4 us |
| the same, a batch of 1024, FP16 / INT8, per sample | 2.04 / 0.54 us | 0.92 / 0.31 us |
| training that MLP with batches of 2 / 8, 4096 samples | 0.57 / 0.093 s | 0.33 / 0.076 s |
| loading a 924,930-parameter FP32 network / INT8 network from memory | 26 / 13 ms | 8.3 / 1.3 ms |
| saving it in FP32 / FP16 | 10.8 / 15.2 ms | 2.5 / 1.4 ms |
| an epoch of a small CNN on CIFAR-10 streamed from a `.slettd` file (from float arrays: 4.7 / 4.5 s) | 14.7 s | 5.5 s |
| Python `spingalett_forward()` of one sample through that MLP | 69 us | 22 us |

Deployment models of the CIFAR-10 network of `Examples/CIFAR10.c` (551K parameters, six
convolutions with batch normalization folded in), `spingalett_model_predict()` on 4,000 test
images, images per second; 0.7's integer kernels against 0.8's (float layers are unchanged):

| CIFAR-10 network | Threads | FP32 | FP16 | INT8 | INT4 |
|---|---:|---:|---:|---:|---:|
| 0.7 integer kernels | 1 | 1,020 | 950 | 810 | 830 |
| 0.8 | 1 | 1,010 | 970 | 2,520 | 2,960 |
| 0.7 integer kernels | 4 | 3,420 | 3,550 | 2,710 | 2,500 |
| 0.8 | 4 | 3,530 | 3,710 | 7,380 | 7,950 |

With the AVX-512 VNNI tiles an INT8 model predicts 3.1 times as fast as before on one thread and
2.7 times on four, and 2.5 and 2.1 times as fast as the FP32 model. One image at a time
(`spingalett_model_run()`), the INT8 network runs 1.7 to 1.8 times as fast as in 0.7 (1,580
against 860 images per second), its convolutions now taking four pixels against four filters at a
time.

The x86-64 release packages are compiled for a baseline instruction set and choose AVX2 or AVX-512
matrix kernels at run time (since 0.6), and AVX-512 VNNI or AVX-VNNI integer kernels (since 0.8).
On the Cascade Lake VM where 0.6 was measured, on one thread, the baseline package trains 3.6 to
4.9 times as fast, and infers 5 times as fast, as with kernels for its baseline (SSE2), as in 0.5:

| Baseline x86-64 build, one thread | Full batch | Mini-batch 64 | Inference |
|---|---:|---:|---:|
| SSE2 kernels (0.5) | 3,700 | 3,200 | 9,000 |
| AVX-512 kernels chosen at run time (0.6) | 18,200 | 11,400 | 45,000 |

`Examples/MNIST.c` trains a 784-256-128-10 network with dropout (AdamW, cosine schedule,
mini-batches of 128) on 55,000 images, keeps the epoch with the best accuracy on the other 5,000
and reaches 98.2% test accuracy after 10 epochs, which take a few seconds with OpenMP on the same
VM. `Examples/MNIST_CNN.c` reaches 98.9% after two epochs of about 9 seconds each on four threads.
`Examples/CIFAR10.c` reaches 87.8% test accuracy after 20 epochs, 20 minutes on four threads of the
Sapphire Rapids VM (760 images per second with augmentation); its INT8 model keeps 87.8%. With
`resnet20` it reaches 91.55% after 100 epochs (12 threads of the i7-12650H, before the indirect
kernels: 93 minutes; FP16 model 91.56%, INT8 91.58% in 281 KB), and the same 91.55% trained on the
RTX 4050 Laptop GPU (`gpu`) in 9 minutes, 91.80% in bfloat16 (`bf16`) in 5 minutes (FP16 model
91.83%, INT8 91.77%); with `resnet32`, 92.39% (INT8 model 92.43% in 480 KB).
`Examples/Segmentation.c` reaches a mean intersection over union of 0.882 on its test images
(circles 0.925, squares 0.866, triangles 0.855; 98.5% of the pixels right) after 12 epochs of 4,096
new images, 15 s on the RTX 4050 and 80 s on the i7-12650H's CPU; its INT8 model keeps 0.878, and
predicts 3,400 images a second on the CPU against 3,100 for the FP32 model.

## Project layout

```
Include/Spingalett/   Public headers (Spingalett.h, the runtime's Spingalett.Runtime.h, the inference
                      engine's Spingalett.Inference.h) and the CMake-generated configuration header
                      template
Src/                  Library sources (network, training, kernels, serialization, inference engine, ...)
Src/Gpu/              The GPU backends: device layer, network executor, Vulkan's compute shaders
                      (Shaders/) and the CUDA kernels (Cuda/)
Examples/             XOR, MNIST (dense and convolutional), CIFAR-10, U-Net segmentation, throughput
                      benchmark (C and PyTorch counterpart), DatasetTool, ModelTool
Examples/Runtime/     RunModel, a program of the runtime alone
Examples/Embedded/    MNIST on a Cortex-M4 (QEMU) with the standalone inference engine
docs/                 File format specifications (models, data sets)
Apps/DigitPad/        Digit-drawing demo app, its trainer and AppImage and Windows packaging
Tests/                Test suite (CTest) and fixtures
Bindings/Python/      Python bindings
cmake/                CMake package, pkg-config, runtime and inference-only build helpers
```

## Compatibility

From 1.0 on, Spingalett follows [semantic versioning](https://semver.org/): a 1.x release keeps
what 1.0 gave, and only 2.0 may take anything away.

- **Source:** a program written for 1.x compiles against every later 1.y. No function, type, struct
  field, enumerator or constant of the headers (`Spingalett.h`, `Spingalett.Runtime.h`,
  `Spingalett.Inference.h`, `Spingalett.Short.h`, `Spingalett.hpp`) is removed, renamed or given
  another meaning (a declaration may move to a header that `Spingalett.h` includes); releases add
  them. A new field takes the place of reserved words and means, at zero, what a program that does
  not know it expects, so the designated initializers of 1.0 keep their meaning.
- **Binary:** a program built against 1.x runs with every later 1.y without being built again: the
  shared library keeps the soname `libspingalett.so.1` (`libspingalett.1.dylib` on macOS), and the
  runtime `libspingalett-runtime.so.1` with the same symbol versions, the structs
  keep their size and the offsets of their fields, enumerators keep their values, and functions keep
  their parameters. On ELF platforms the functions carry symbol versions (`SPINGALETT_1.0`, then one
  per release that adds some), so that the loader refuses a library older than the program needs.
  The test `api.abi` compares every release with the one before.
- **Files:** every 1.x loads the `.slett` files 1.0 loads (formats 1 to 7; the engine runs 3 to 7)
  and `.slettd` files of formats 1 and 2, and writes the oldest format that holds what it saves; a
  later format only adds kinds of layers or coders, so a 1.0 engine runs every model of a later 1.x
  that uses only what it knows.
- **Python:** the `spingalett` package keeps its functions, classes and keyword arguments in the
  same way.
- **Not held:** the values of the `*_COUNT` enumerators and of `SPINGALETT_FORMAT_VERSION`, which
  grow with what a release adds; the text of messages; speed; and the last bits of results. Training
  repeats its bits on one version (on any number of threads, and on one GPU), but a release whose
  kernels add in another order may round differently. Everything outside `Include/` is internal.
- **Deprecation:** what is to go in 2.0 is marked deprecated in a 1.x release, in the header and the
  CHANGELOG, and keeps working until then.

## Status and roadmap

Spingalett is at version 1.1.0. Its API (prefixed names, structs that can grow, zero as every
field's default), its ABI (the soname `libspingalett.so.1`) and its formats (`.slett` 7, `.slettd` 2)
are kept by every 1.x release, as [Compatibility](#compatibility) states. The network is an opaque
handle, so its internal layout can change without breaking programs; saved models are versioned and
remain loadable.

Planned work, roughly in order (details in [ROADMAP.md](ROADMAP.md)):

- 1.1 (released): the runtime, and the CUDA backend of the library's own kernels (no CUDA toolkit,
  cuBLAS or cuDNN), next to Vulkan
- Any release of 1.x, changing no API: the single-precision kernel for convolutions of few
  channels, fewer passes, products in FP16, Winograd convolutions, deployment models on the GPU
- Later: quantization-aware training, NEON kernels for training, further language bindings

## Contributing

Bug reports and pull requests are welcome. Please make sure the test suite passes
(`ctest --test-dir Build --output-on-failure`) for both a minimal build and a build with
`-DBUILD_WITH_OPENMP=ON -DBUILD_WITH_OPENBLAS=ON`, and add tests for new behaviour. CI runs the
suite with GCC and Clang, without `-march=native` (portable kernels) and under AddressSanitizer
and UndefinedBehaviorSanitizer. [AGENTS.md](AGENTS.md) summarizes the layout, the checks and the
invariants the tests enforce, for coding agents and new contributors alike.

## License

Spingalett is released under the [MIT License](LICENSE).
