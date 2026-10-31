# Models from elsewhere

ONNX files and PyTorch weights as networks that train, and ONNX models run as they are. [Graphs.md](Graphs.md) describes graphs of operators in full.

## Importing models: ONNX and PyTorch

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

Transformers import into the library's own layers (models exported with an example of one sample, or
with a dynamic batch). An input of token ids (`[N, T]`, integers) becomes an input layer of `T`
values and `Gather` from a constant table an embedding, a learned position table added to it
(GPT-2's) its positions; an input `[N, T, C]` becomes a layer of `1 x T` cells. `MatMul` and `Gemm`
by constant weights on the tokens become linear layers; layer normalization and RMS normalization
over each token (`LayerNormalization`, `RMSNormalization`, or written out with `ReduceMean`, `Pow`,
`Sqrt` and `Div`) become theirs; element-wise expressions are recognised as the activations they
compute (GELU as `Gelu` or written out in its erf or tanh form, SiLU as `x * Sigmoid(x)`), and
products of two layers (`Mul`, SwiGLU's gate) become multiplication layers. Attention as exporters
write it out becomes one attention layer: queries, keys and values from one product or three, split
into heads by `Reshape`, `Transpose`, `Split` or `Slice`, the product of queries and keys scaled (on
either side or both, as `scaled_dot_product_attention` exports it), masked causally (`Where`, or an
added triangle of `-inf`), a `Softmax` over the keys, the product with the values, keys and values
repeated for grouped queries (`Expand`, `Tile`), and rotary embeddings (`rotate_half` with tables of
cosines and sines, whose base the importer reads from the tables). The queries', keys' and values'
products become the one linear layer the attention layer reads, a scale other than
`1/sqrt(head size)` folded into the queries' weights. Opset 23's `Attention` imports the same way,
and so do the fused operators of ONNX Runtime's transformer optimizer (`com.microsoft`'s
`Attention`, `MultiHeadAttention`, `SkipLayerNormalization`, `SkipSimplifiedLayerNormalization`,
`FastGelu`, `BiasGelu`). Nodes whose inputs are all constants are computed on import, the shape
arithmetic both exporters leave in graphs among them. GPT-2-, LLaMA- (grouped queries, rotary
embeddings, SwiGLU), Gemma- and BERT-like models and `nn.TransformerEncoderLayer`, exported by
either exporter at opsets 14 to 23, import into the layers the library builds such models of (a
GPT-2 block is 9 layers) and agree with PyTorch to the rounding of single precision (within 3e-6 on
the tests' models); a language model takes the sparse cross-entropy loss, so it trains further and
generates as it is, and a model of GPT-2 small's size (124M parameters, 650 MB) imports in 0.57 s.
What no layer computes is named in the error: a softmax over each token's channels (the library's
softmax spans a layer's outputs), a product of two computed tensors outside an attention, masks or
caches given as inputs, rotary embeddings over part of a head or of interleaved pairs, attention
between two sequences.

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

## Graphs of operators: any ONNX model

Import makes networks of the library's layers, which train. To run a model as it is, whatever its operators, open
it as a graph of operators ([docs/Graphs.md](Graphs.md)): values (inputs, constants, what nodes compute),
tensors of up to eight axes whose sizes may be named symbols (a batch, a sequence's tokens), several named inputs and
outputs, and nodes of 155 operators with ONNX's semantics: arithmetic and functions with broadcasting, products,
convolutions of one to three spatial axes, pooling, the normalizations, reductions, shapes and data, Resize,
quantization, Einsum, opset 23's Attention and RotaryEmbedding, and opset 27's linear attention and causal
convolutions of hybrid models. The tensors stay in ONNX's layout (NCHW images, say).

```c
SpingalettGraph *g = spingalett_graph_open(.path = "unet.onnx");    /* an ONNX file, or a .slett file of format 11 */
SpingalettTensor in[3] = {
    {.name = "sample", .dtype = SPINGALETT_DTYPE_FLOAT32, .rank = 4, .shape = {1, 4, 64, 64}, .data = latents},
    {.name = "timestep", .dtype = SPINGALETT_DTYPE_FLOAT32, .rank = 1, .shape = {1}, .data = &t},
    {.name = "text", .dtype = SPINGALETT_DTYPE_FLOAT32, .rank = 3, .shape = {1, 77, 2048}, .data = text},
};
SpingalettTensor out = {0};                                         /* NULL name: the first output */
if (spingalett_graph_run(.graph = g, .inputs = in, .input_count = 3, .outputs = &out, .output_count = 1))
    use(out.data, out.shape);                                       /* valid until the graph's next run */
spingalett_graph_save(g, "unet.slett");                             /* format 11, mapped when opened */
spingalett_graph_free(g);
```

When a graph opens, its constants are folded, nodes no output needs and repeated ones are left out, and chains of
operators are fused into the library's kernels: a product by a constant matrix with its bias and activation
(packed once for the row kernel of language models), attention written out as products, a mask and a softmax (in
blocks of 64 queries, never all the scores in memory), GELU and SiLU written out, a convolution's activation.
Convolutions run on the library's channels-last kernels. Floats compute in single precision (float16 and bfloat16
widened), integers in their types; every operator sums in a fixed order, so a graph gives the same bits on any
number of threads. ONNX's 1262 conformance cases of single operators that hold these types pass (the `operators`
group), and whole models exported by `torch.onnx` with random weights match PyTorch within 1e-5
(`Tests/check_graph_models.py`): on six performance cores of an i7-12650H, ResNet-50 runs in 24.4 ms (PyTorch: 21.5),
BERT-base on two sequences of 128 tokens in 97 (81), GPT-2 on 64 tokens in 45 (36) and a UNet of SDXL's
architecture in 36.8 (34.6); transformers.js's ONNX files match ONNX Runtime (`Tests/check_graph_files.py`). Control flow (If, Loop, Scan), sequences and strings are not graphs' operators;
`Examples/Runtime/RunGraph.c` prints a graph's inputs, outputs and operators and times it, and
`SPINGALETT_GRAPH_PROFILE=1` prints the time of each operator.
