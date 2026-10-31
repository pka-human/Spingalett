# Inference and deployment

Running networks, saving them, deployment models from FP32 to INT2, the runtime library, microcontrollers and WebAssembly.

## Inference

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
`spingalett_evaluate()`) in place of arrays; see [The GPU](GPU.md#the-gpu). Dropout is not applied during
inference.

## Saving and loading

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

The file format is versioned and specified in [docs/ModelFormat.md](ModelFormat.md):
little-endian, with 16-byte aligned sections and CRC-32 checksums, so that a file image can be
executed in place (see [Deployment](#deployment-models)). Networks with convolution or pooling layers are
saved in version 4 (`SPINGALETT_FORMAT_VERSION`), which adds the layers' kinds and shapes; networks
of dense layers only are still saved in version 3, so that the engines of earlier releases run them.
Files of versions 1 and 2 remain loadable. `spingalett_load()` returns `NULL` for missing, truncated
or corrupt files and for files with an invalid header.

## Deployment models

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
sample (the largest magnitude maps to 127; per token for the linear layers of language models, whose
tokens then do not depend on each other), multiplies it with the weights in 32-bit integers
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
[Performance](Performance.md) for what they gain).

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
exactly what `spingalett_model_run()` computes, in batches of any size. A model of a language model
also runs in sessions, a token at a time with a key-value cache (see
[Transformers and language models](LanguageModels.md#transformers-and-language-models)): the first session packs the
float linear layers' filters once more for its kernels of few rows (the float size of those layers
again), which the model keeps for every session.

In the full library's GPU modes (`SPINGALETT_COMPUTE_CUDA`, `SPINGALETT_COMPUTE_VULKAN`) a model that owns
its image predicts on the GPU, which the first call prepares and later calls reuse. A model of float
weights runs as a network there, its outputs the CPU's to rounding. An integer model (INT8, INT4, INT2)
runs as the CPU computes it, its outputs those of `spingalett_model_run()` bit for bit: each layer's
inputs are quantized on the GPU as the CPU quantizes them (the divisions rounded as IEEE's, the values
to the nearest even integer), its codes multiply them on the matrix kernels as floats or bfloat16, which
hold such integers exactly, in sums of at most 1,040 products of INT8 codes (16,512 of INT4), which stay
below 2^24 where floats are exact integers (longer sums run in slices added as integers), and the outputs
are rescaled in the CPU's order of operations; pooling, additions, products and concatenations of
layers, batch normalization and upsampling follow the CPU's order too. The activations inside the
network must be none, ReLU or leaky ReLU (the last layer's, softmax or sigmoid among them, runs on the
CPU); a model with others, layer normalization or attention keeps the CPU. On an RTX 4050 Laptop GPU,
against all 16 threads of the Intel Core i7-12650H next to it (samples per second, INT8, medians):

| Model (`Examples/Benchmark.c`) | CPU | CUDA | Vulkan |
|---|---:|---:|---:|
| 784-512-1000-10 MLP, 20,000 samples | 658,000 | 2,490,000 | 2,010,000 |
| CNN of MNIST, 10,000 samples | 86,400 | 567,000 | 367,000 |
| ResNet-20, 4,096 images of 32 x 32 | 10,300 | 69,900 | 56,900 |
| MobileNet-style network, 4,096 images | 15,400 | 89,400 | 76,400 |
| U-Net, 1,024 images of 64 x 64 | 3,130 | 15,200 | 14,100 |

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

## The runtime

Programs that only run trained models can link the runtime instead of the library.
`libspingalett-runtime`, declared by `Spingalett.Runtime.h`, holds what this section describes:
`.slett` files loaded as models, `spingalett_model_predict()` and `spingalett_model_evaluate()` with
the library's kernels, threads and choice of instruction sets at run time, sessions of language models
(`spingalett_session_feed()`, generation with a key-value cache), the engine of
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

The runtime reads `.slett` files of format versions 3 to 10, everything Spingalett 0.5 and later
writes, and language models' GGUF files. Files of versions 1 and 2 (0.4 and earlier) hold networks rather than images; it refuses
them with `SPINGALETT_ERR_FORMAT_VERSION`, and `ModelTool convert` rewrites them.

| | Library | Runtime | Engine |
|---|---|---|---|
| Header | `Spingalett.h` | `Spingalett.Runtime.h` | `Spingalett.Inference.h` |
| Training, networks, data sets, importers, the GPU | yes | | |
| `.slett` models from FP32 to INT2 | yes | yes | yes |
| Sessions of language models (key-value cache) | yes | yes | |
| Batches, threads, kernels chosen at run time | yes | yes | one sample at a time |
| Heap, files | yes | yes | |
| Size | 12.6 MB (x86-64; 2.0 MB without CUDA) | 510 KB (x86-64) | about 12 KB of code on a Cortex-M4 |

## Running in place

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

## Microcontrollers and C headers

```bash
Bin/ModelTool header model.slett model.h my_model --precision int8    # or spingalett_export_c_header()
```

writes the image as a 16-byte aligned `static const uint8_t my_model[]` with the macros
`MY_MODEL_SIZE`, `MY_MODEL_INPUTS`, `MY_MODEL_OUTPUTS` and `MY_MODEL_WORKSPACE`. The engine itself,
`Src/Spingalett.Inference.c` with `Include/Spingalett/Spingalett.Inference.h` and
`Src/Spingalett.Engine.h`, compiles on its own with `-DSPINGALETT_INFERENCE_ONLY`: no training
code, no file I/O, no OpenMP, no heap and no global state, about 12 KB of code on a Cortex-M4
with convolutions and pooling. CMake builds it alone with `-DSPINGALETT_INFERENCE_ONLY=ON`.
[Examples/Embedded](../Examples/Embedded) runs the MNIST models on a Cortex-M4F in QEMU with the
weights in flash: the 784-256-128-10 network with 2.8 KB of RAM, the CNN (INT8, 424 KB of flash)
with 208 KB.

## WebAssembly: Node and browsers

[`Bindings/JavaScript`](../Bindings/JavaScript/README.md) builds the engine for WebAssembly with Clang and
`wasm-ld`, without a C library (`sh Bindings/JavaScript/build.sh`; the mathematical functions it calls
round as glibc's do), and wraps it in an npm package `spingalett`: an ES module with TypeScript
declarations for Node 18 and later and for browsers, around a 62 KB module that imports nothing. A
model, of any kind and precision the engine runs, is loaded from its bytes and runs sample by sample
with the native engine's results (integer models bit for bit); language models give the next token's
logits and generate as `spingalett_generate()` does:

```js
import { loadModel } from 'spingalett';

const model = await loadModel(await (await fetch('chargpt.slett')).arrayBuffer());
console.log(model.layers.map((l) => `${l.type} ${l.precision}`));
const tokens = model.generate([...new TextEncoder().encode('ROMEO:')], 200, { temperature: 0.8, topK: 40 });
```

Each release attaches the package (`spingalett-<version>.tgz`, for `npm install`) and the module.

## ModelTool

`Examples/ModelTool.c`, installed with the library:

```
ModelTool info model.slett                              layers, precisions, sizes, workspace
ModelTool convert in.slett out.slett --precision int8   any format version, as format 3 or 4
ModelTool header model.slett model.h name [--precision P]
ModelTool eval model.slett data.slettd                  accuracy and loss in every precision
ModelTool eval model.slett images labels                (an IDX pair, such as MNIST)
ModelTool bench model.slett                             latency and throughput in every precision
```
