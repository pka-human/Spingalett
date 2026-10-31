# Graphs of operators

A network of layers (`Spingalett.h`) and a deployment model (`Spingalett.Runtime.h`) hold the layers the library
defines. A **graph of operators** holds any model built of the operators ONNX defines: values (the graph's inputs,
its constants and what its nodes compute), tensors of up to eight axes whose sizes are numbers or named symbols (a
batch, a sequence's tokens), several named inputs and outputs, and nodes of 155 operators, each with ONNX's
semantics. Graphs come from ONNX files, as PyTorch (`torch.onnx.export`), TensorFlow's and JAX's converters and
other tools write them, and from `.slett` files of formats 11 and 12 (docs/ModelFormat.md), which the library writes. They
run on the CPU and on the GPU, in the runtime too (`libspingalett-runtime`).

## Running a graph

```c
#include <Spingalett/Spingalett.Runtime.h>

SpingalettGraph *g = spingalett_graph_open(.path = "model.onnx");     /* or .data and .size: bytes in memory */
if (!g) { fprintf(stderr, "%s\n", spingalett_last_error_message()); return 1; }

/* what it takes and gives, as it declares them */
for (uint32_t i = 0; i < spingalett_graph_input_count(g); i++) {
    SpingalettTensorInfo in = spingalett_graph_input(g, i);
    printf("%s: rank %u, first size %lld (%s)\n", in.name, in.rank, (long long)in.shape[0],
           in.symbols[0] ? in.symbols[0] : "a number");
}

float pixels[1 * 3 * 224 * 224] = {0};
SpingalettTensor input = {.name = "image", .dtype = SPINGALETT_DTYPE_FLOAT32, .rank = 4,
                          .shape = {1, 3, 224, 224}, .data = pixels};
SpingalettTensor output = {.name = "logits"};       /* NULL names: the graph's inputs and outputs in order */
if (!spingalett_graph_run(.graph = g, .inputs = &input, .input_count = 1, .outputs = &output, .output_count = 1))
    fprintf(stderr, "%s\n", spingalett_last_error_message());
/* output.dtype, .rank, .shape and .data are set; the values live in the graph until its next run */

spingalett_graph_save(g, "model.slett");            /* format 11: opens without parsing ONNX */
spingalett_graph_free(g);
```

In Python:

```python
import numpy as np
import spingalett as sp

graph = sp.Graph.load("model.onnx")
print(graph.inputs)        # [TensorInfo(name='image', dtype=dtype('float32'), shape=(1, 3, 224, 224))]
outputs = graph.run({"image": np.zeros((1, 3, 224, 224), np.float32)})      # a dict of copies, by name
graph.save("model.slett")
```

`Examples/Runtime/RunGraph.c` (`Bin/RunGraph model.onnx [--dim batch=8] [--runs N] [--save model.slett]`) prints a
graph's inputs, outputs and operators and times its runs on random inputs.

A run checks every input against its declared type and shape: a symbol must have one size in all the inputs of a
run, and a number its value. A graph runs one call at a time; calls from several threads wait for each other.
Bytes given to `spingalett_graph_open()` are copied; a file is mapped, and constants of float32 and integer types are
read where they lie in it.

## Values and types

Floats compute in single precision: constants of float16 and bfloat16 become float32 when the graph opens (taking
twice their memory), inputs of those types are accepted as float32 or widened from their bits, `Cast` to them
rounds to their precision and keeps the result in float32, and outputs declared of those types return to them
(rounded to nearest even). ONNX's float64 constants become float32. Integers (int64, int32, int8, uint8) compute in
64 bits and are stored in their types; booleans are bytes of 0 or 1. Strings, sequences, maps and optional values,
16-bit and unsigned 32- and 64-bit integers, float8 and 4-bit types are not held.

## Operators

ONNX's operators of the default domain, with the attributes and inputs of opsets 7 to 28 (an input that an earlier
opset took as an attribute is read from either: Split's sizes, Slice's ranges, Squeeze's, Unsqueeze's and the
reductions' axes, Pad's pads, Clip's bounds, TopK's k), the forms of Softmax, LogSoftmax and Hardmax before opset 13
(over the axes from theirs on) and of Resize before opset 11 (asymmetric coordinates, nearest by floor):

| Group | Operators |
|---|---|
| Element-wise (broadcasting) | Add, Sub, Mul, Div, Pow, Mod, Max, Min, Sum, Mean, PRelu, Where; Equal, Less, LessOrEqual, Greater, GreaterOrEqual; And, Or, Xor, Not; BitwiseAnd, BitwiseOr, BitwiseXor, BitwiseNot, BitShift |
| Unary | Abs, Neg, Exp, Log, Sqrt, Reciprocal, Sin, Cos, Tan, Asin, Acos, Atan, Sinh, Cosh, Asinh, Acosh, Atanh, Tanh, Sigmoid, Erf, Floor, Ceil, Round, Sign, IsNaN, IsInf; Relu, LeakyRelu, Elu, Selu, Celu, Gelu, HardSigmoid, HardSwish, Softplus, Softsign, Mish, ThresholdedRelu, Shrink, Swish, SwiGLU; Clip, Cast, CastLike, Identity, Dropout (inference) |
| Products and convolutions | MatMul, Gemm, Einsum, Conv, ConvTranspose (one to three spatial axes, groups, dilations, padding and auto_pad), MaxPool (and its indices), AveragePool, LpPool, GlobalAveragePool, GlobalMaxPool |
| Normalizations | BatchNormalization (inference), LayerNormalization, RMSNormalization, GroupNormalization, InstanceNormalization, LpNormalization, MeanVarianceNormalization, LRN; Softmax, LogSoftmax, Hardmax |
| Reductions | ReduceSum, ReduceMean, ReduceMax, ReduceMin, ReduceProd, ReduceSumSquare, ReduceL1, ReduceL2, ReduceLogSum, ReduceLogSumExp, ArgMax, ArgMin, TopK, CumSum, CumProd |
| Shapes and data | Reshape, Flatten, Squeeze, Unsqueeze, Transpose, Concat, Split, Slice, Gather, GatherElements, GatherND, ScatterND, ScatterElements, Scatter, TensorScatter, Expand, Tile, Pad, Shape, Size, Range, ConstantOfShape, Constant, Trilu, Resize (nearest, linear and cubic, every coordinate transformation), Upsample, DepthToSpace, SpaceToDepth, NonZero, OneHot, EyeLike, Compress, ReverseSequence; HannWindow, HammingWindow, BlackmanWindow |
| Quantization | QuantizeLinear, DequantizeLinear (per tensor, per axis and blocked; uint8, int8, int32), DynamicQuantizeLinear, MatMulInteger, QLinearMatMul |
| Composite (opset 23 and later) | Attention (three or four axes, grouped heads, a past, boolean or added masks shorter than the keys, causal and sliding windows, soft caps, the scores' outputs), RotaryEmbedding, LinearAttention (linear, gated, delta and gated delta rules), CausalConvWithState |

Not supported: control flow (If, Loop, Scan), sequences, strings, training (losses, optimizers, BatchNormalization's
training mode), random draws, signal transforms (DFT, STFT), detection (NonMaxSuppression, RoiAlign), GridSample,
recurrent layers (RNN, GRU, LSTM), Unique, Resize's antialiasing when downsampling. A file that holds one fails to
open with the operator and the file's opset named. Tests/Data/graph_cases.txt lists ONNX's conformance cases the
`operators` test group leaves out, and why.

## What happens when a graph opens

The nodes are put in an order that runs each after what it reads (files are usually in one), constants are folded
(nodes whose inputs are all constants run once, their outputs kept when they hold at most 2^26 values), nodes that no
output needs are left out, and nodes of one operator on the same inputs with the same attributes become one. Then
chains of nodes whose intermediate values nothing else reads are fused into operators of the library's own or of
ONNX's (`Src/Spingalett.IRFuse.c`), which do the same work in fewer passes over memory:

- **LayerNormalization and RMSNormalization** from their forms written out over the last axis (ReduceMean, Sub,
  Pow, Sqrt, Div or Reciprocal, a scale and a bias), as exporters before opset 17 write them;
- **Gelu and Swish** from the forms exporters write out: `x * 0.5 * (1 + Erf(x / sqrt 2))`, its tanh approximation,
  `x * Sigmoid(x)` and `x * Sigmoid(1.702 x)` (CLIP's quick GELU);
- **Linear**: a MatMul by a constant matrix, an Add of a constant row and an activation (ReLU, sigmoid, tanh, GELU,
  SiLU), each after the first optional, or a Gemm by a constant matrix and row and such an activation: the matrix is
  packed once for the library's row kernel and the bias and activation applied while each tile is in cache. An
  activation after a Reshape of the product goes into the epilogue too. A MatMul or Gemm by a constant matrix packs
  it once as it is;
- **FusedAttention**: MatMul of the queries and the transposed keys (each scaled by a constant or not), a scaling,
  a Where keeping scores (the others a large negative: a causal mask), an Add of a mask, Softmax over the keys (or
  PyTorch's safe softmax, Where(IsNaN(p), 0, p)) and MatMul by the values, Casts to float among them, computed in
  blocks of 64 queries without the scores of all of them in memory; when the heads' transposes surround it,
  (B, S, H, D) to (B, H, S, D) and back, they are taken in;
- **Conv**: a channel's Mul and Add of constants and an inference BatchNormalization after a convolution are
  folded into new filters and a bias, and an activation after it is applied in the convolution's epilogue.

`SPINGALETT_GRAPH_FUSE=0` in the environment leaves the fusions out (to compare with).
`spingalett_graph_node_count()` and `spingalett_graph_node_op()` (Python: `Graph.operators`) give the nodes that
remain. A saved graph keeps its fusions; its constants are written as the source held them (float16 stays 16 bits).

## Graphs on the GPU

In `SPINGALETT_COMPUTE_CUDA` and `SPINGALETT_COMPUTE_VULKAN` a graph's nodes run on the GPU where it has their kernels, the others on the
host with their values moved; the outputs arrive on the host as on the CPU. Nodes of few values stay on the host, whose time
they would not repay (`SPINGALETT_GRAPH_GPU_MIN` sets what counts as few). `SPINGALETT_GRAPH_GPU_REPORT=1` lists the operators the host ran, 3 every
node. The first run with new input shapes makes a plan of memory and commands, and later ones replay it.

`spingalett_set_gpu_precision(SPINGALETT_PRECISION_BFLOAT16)` multiplies on the matrix units: products and attention read and write bfloat16, values
whose readers all can are kept so in memory, and the rest computes in single precision. Results match the CPU's to rounding in single
precision, and within bfloat16's error (3% of an output's largest magnitude in the tests) in bfloat16; a run on one device repeats its bits.

## How graphs compute

Every value is computed by one thread in one fixed order: element-wise operators split their outputs between
threads, normalizations and reductions sum each row in its order, products and convolutions run on the library's
matrix kernels, whose sums are fixed by the shapes. A graph gives the same bits on any number of threads, and a
saved graph the bits of the graph it was saved from. Results match other implementations (PyTorch, ONNX's
reference) up to rounding; in the conformance cases within ONNX's tolerances, in whole models within relative
differences of 1e-5 (Tests/check_graph_models.py).

Products by constant matrices run on `spingalett_conv_rows()` against the matrix packed once (on its first run,
which also takes the time of packing; the packed copy takes as much memory as the matrix). Convolutions of one or two
spatial axes without dilation run on the library's channels-last kernels (indirect products, depthwise ones
directly): the input is transposed to channels last and the output back, the filters reordered once. Tensors take
blocks of a pool the graph keeps between runs, so that runs after the first allocate nothing; shape operators
(Reshape, Flatten, Squeeze, Unsqueeze, Identity) are views of their input.

With `SPINGALETT_GRAPH_PROFILE=1` in the environment, a graph prints, when it is freed, the time spent in each
operator over its runs after the first.

Times of whole models, exported by `torch.onnx` with random weights, against PyTorch 2.14 on the same six
performance cores of an i7-12650H (medians of interleaved runs, `Tests/check_graph_models.py`):

| Model | Graph | PyTorch |
|---|---:|---:|
| ResNet-50, one image of 224 x 224 | 24.4 ms | 21.5 ms |
| ViT-B/16, one image | 80.9 ms | 63.0 ms |
| BERT-base, two sequences of 128 tokens | 97.3 ms | 81.2 ms |
| GPT-2, 64 tokens | 45.3 ms | 36.1 ms |
| CLIP ViT-L/14's text encoder, 77 tokens | 46.1 ms | 29.2 ms |
| Whisper's encoder (tiny), 30 s | 83.7 ms | 68.2 ms |
| Stable Diffusion's VAE decoder, 32 x 32 latents | 1201 ms | 1046 ms |
| A UNet of SDXL's architecture (fewer channels and layers), 32 x 32 latents | 36.8 ms | 34.6 ms |

ONNX files as transformers.js publishes them (Xenova's exports, opsets 11 to 14, their GELU, layer normalization
and attention written out), against ONNX Runtime 1.31 on the same cores (`Tests/check_graph_files.py`, ONNX
Runtime in a process of its own; outputs within 1e-5 of its own, the float16 file's within 2.2e-4, the quantized
one's within 2.6e-2, where one of 49,152 dynamically quantized values rounds the other way):

| File | Graph | ONNX Runtime |
|---|---:|---:|
| all-MiniLM-L6-v2, two sequences of 64 tokens | 8.1 ms | 6.1 ms |
| all-MiniLM-L6-v2, quantized (MatMulInteger) | 16.6 ms | 3.3 ms |
| distilgpt2's decoder, 64 tokens | 35.1 ms | 19.1 ms |
| ViT-B/16 in float16 | 92.4 ms | 60.3 ms |
| DETR-ResNet-50, 480 x 640 | 160.9 ms | 100.9 ms |

Convolutions transpose their input to channels last and their output back (1.6 keeps activations channels-last
between them), and products of 8-bit integers sum 16-bit rows without the library's integer tile kernels: the
largest gaps above.
