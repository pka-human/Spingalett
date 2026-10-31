# The .slett model format, versions 3 to 11

`.slett` files store a network: its shape, its parameters in a chosen precision and, optionally,
the optimizer state for resuming training. `spingalett_save()` and `spingalett_save_to_memory()`
write the oldest version that can hold the network: version 3 for networks of dense layers only
(which engines from Spingalett 0.5 on can run), version 4 for networks with convolution or pooling
layers (0.7 on), version 5 for networks with batch normalization or grouped convolutions (0.8 on),
version 6 for graphs: networks with a layer that reads other layers than the one before it, adds or
concatenates several, or pools globally (0.10 on), version 7 for networks with transposed
convolutions, upsampling or layer normalization (0.13 on), version 8 for the layers of transformers
(embeddings, attention, RMS normalization, products of layers), the GELU and SiLU activations and the
sparse cross-entropy loss (1.2 on), version 9 for the training features of language models: dense and
linear layers tied to an embedding's table, attention that ignores padding or reads a sliding window,
cross-attention and embeddings of part of their input (1.3 on), version 10 for layers whose weights are in a
block precision, one of GGUF's quantization types (1.4 on). `spingalett_load()` and `spingalett_load_from_memory()`
read versions 1 to 10. Version 11 (1.5 on) holds a graph of operators instead of a network of layers
(**Graphs of operators**): `spingalett_graph_save()` writes it and `spingalett_graph_open()` reads it; version 12
(1.6 on) is version 11 whose nodes use the attributes of 1.6 that a reader of version 11 would misread (maps
channels last, a group normalization's activation, attention of views), which a writer gives only such graphs.
A checkpoint (`spingalett_save_checkpoint()`, 1.4 on) is an image of one of these versions with a
record of a training run after it, which readers of the image ignore (**Checkpoints**).

Versions 3 to 5 are laid out so that a file image can be used as it is: the inference engine
(`spingalett_model_init()` in `Spingalett.Inference.h`) checks the image and computes directly from
it, so a model can be read into memory, compiled into a program as a byte array, or executed from
flash. This document describes the layout precisely enough to write an independent reader.
Version 4 is version 3 with a larger layer table entry that adds the layer's kind and shape;
version 5 extends the entry again with convolution groups and the constants of batch
normalization; version 6 adds the layers each layer reads and where its output lives in the
engine's workspace; version 7 adds three kinds of layers and the field of the upsampling mode, in
version 6's layout; version 8 adds four kinds of layers, three activations and a loss, in version 6's
layout too (their fields in the existing ones and the reserved bytes 104 to 111); version 9 adds fields
to version 8's entries in bytes that were 0 for their kinds; version 10 adds the codes of the block precisions in
version 9's layout. The differences are marked below.

Batch normalization is folded when a network is saved for deployment: files written in a precision
other than FLOAT32 without optimizer state, and the models of `spingalett_model_from_network()`,
merge every batch normalization that reads a dense or convolution layer without activation which
feeds nothing else into that layer (its weight rows scaled by `gamma / sqrt(var + eps)` and its
biases shifted alike; the layer takes the normalization's activation, and the layers that read the
normalization read it instead). Other normalizations remain layers of their own, so such files can
still be version 3 or 4.

**Stability.** Version 7 is the format of Spingalett 1.0. Later versions only add what earlier
ones cannot express (kinds of layers, precisions, activations, fields in space reserved as zero),
and leave what a version 7 file means unchanged; a 1.x writer writes the oldest version that holds
the network, as now, and every 1.x engine runs every file whose layers it has. Codes of version 7
(activations, layer kinds, precisions) keep their values. Version 8 (Spingalett 1.2), version 9
(Spingalett 1.3) and version 10 (Spingalett 1.4) are such additions. Version 11 (Spingalett 1.5) is a format
of its own after the same magic: its operators' codes keep their values, new operators take new codes, and every
1.x reader of graphs opens every file whose operators it has. Version 12 (Spingalett 1.6) is version 11's layout
with attributes whose meaning a 1.5 reader does not know, so that it refuses the file rather than misread it.

All integers are little-endian. "float" means an IEEE 754 binary32 stored as its bit pattern.
CRC-32 is the IEEE polynomial (0xEDB88320, reflected, initial value and final XOR 0xFFFFFFFF), as
in zlib and PNG.

## Overview

```
header (64 bytes) | layer table (48 bytes per entry in version 3, 64 in version 4, 80 in version 5,
112 in versions 6 to 10) | sections ...
```

A network of `L` layers (the input layer included) has `L - 1` table entries; entry `i` (counted
from 0) describes layer `i + 1` and what feeds it: the output of layer `i` in versions 3 to 5, the
outputs of the layers it names in versions 6 to 10 (always layers before it, so that the table order is an
order in which every layer can be computed after its inputs). Each dense or convolution layer
owns up to four sections: its weights, the per-row scales of integer weights, its biases and its
optimizer state; a batch normalization owns the same four, its scales section holding its running
statistics; pooling, adding, concatenating and upsampling layers own none. From version 6 on, an entry that reads
several layers owns a section listing them. Every section starts at an offset that is a multiple
of 16; the bytes between sections are 0. A writer places the sections in table order, but a reader
must only rely on the offsets.

Data flows through the network as one tensor per sample, `height x width x channels` floats in
channels-last order: element `(y, x, c)` is at `(y * width + x) * channels + c`. A dense layer reads
its input as a flat vector; its output has shape `1 x 1 x out`. In version 3 every layer is dense.

## Header

| Offset | Size | Field |
|---:|---:|---|
| 0 | 6 | magic `SLETTM` |
| 6 | 2 | format version, 3 to 10 |
| 8 | 4 | `L`: number of layers, input layer included (2 to 65536) |
| 12 | 1 | loss function: 0 mean squared error, 1 cross-entropy, 2 sparse cross-entropy (version 8: a softmax over each cell's channels of the output layer against a class index a cell) |
| 13 | 1 | flags: bit 0 set when the file holds optimizer state; other bits 0 |
| 14 | 2 | reserved, 0 |
| 16 | 8 | optimizer time step (number of steps taken; 0 without optimizer state) |
| 24 | 8 | file size in bytes |
| 32 | 4 | versions 4 to 10: height of the input layer (1 to 65535); version 3: reserved, 0 |
| 36 | 4 | versions 4 to 10: width of the input layer (1 to 65535); version 3: reserved, 0 |
| 40 | 8 | versions 6 to 10: bytes of activations, the part of the engine's workspace that holds the layers' outputs (see below); versions 3 to 5: reserved, 0 |
| 48 | 8 | reserved, 0 |
| 56 | 4 | CRC-32 of bytes 64 to file size - 1 (the layer table and all sections) |
| 60 | 4 | CRC-32 of bytes 0 to 59 |

Data after the recorded file size (for example the rest of a flash region) is not part of the
image.

The input layer's channels are its units (`in` of the first entry) divided by height x width.

## Layer table

`L - 1` entries start at offset 64, 48 bytes each in version 3, 64 bytes in version 4, 80 bytes
in version 5 and 112 bytes in versions 6 to 10:

| Offset | Size | Field |
|---:|---:|---|
| 0 | 4 | `in`: units of layer `i` (for the first entry, the network's input size); versions 6 to 10: units of the entry's first input |
| 4 | 4 | `out`: units of layer `i + 1`; in versions 3 to 5 equals `in` of the next entry |
| 8 | 1 | activation of layer `i + 1` (codes below) |
| 9 | 1 | weight precision (codes below) |
| 10 | 1 | versions 4 to 10: kind of layer `i + 1`: 0 dense, 1 convolution, 2 max pooling, 3 average pooling, 4 batch normalization (versions 5 to 10), 5 addition, 6 concatenation, 7 global average pooling (versions 6 to 10), 8 transposed convolution, 9 upsampling, 10 layer normalization (versions 7 to 10), 11 embedding, 12 attention, 13 RMS normalization, 14 multiplication (versions 8 to 10); version 3: reserved, 0 |
| 11 | 1 | reserved, 0 |
| 12 | 4 | dropout rate of layer `i + 1`, a float in [0, 1) (used only when training) |
| 16 | 8 | offset of the weights; 0 for kinds without parameters (pooling, addition, concatenation); version 10: an attention's gammas of its queries' and keys' normalization (flag bit 2) |
| 24 | 8 | offset of the row scales (batch normalization: of its running statistics); 0 for FLOAT32, FP16 and BFLOAT16 dense and convolution layers, and for kinds without parameters; version 10: an attention's rotary frequencies (flag bit 3) |
| 32 | 8 | offset of the biases; 0 for kinds without parameters |
| 40 | 8 | offset of the optimizer state; 0 when the file has none, and for kinds without parameters |
| 48 | 2 | versions 4 to 10: output height of layer `i + 1` |
| 50 | 2 | versions 4 to 10: output width of layer `i + 1` |
| 52 | 2 | versions 4 to 10: kernel height (window rows); version 10: an attention's dimensions of a head rotated (0: all) |
| 54 | 2 | versions 4 to 10: kernel width (window columns) |
| 56 | 2 | versions 4 to 10: vertical stride (version 10: with offset 58, an attention's scale of its scores, a float, 0: `1 / sqrt(head)`) |
| 58 | 2 | versions 4 to 10: horizontal stride |
| 60 | 2 | versions 4 to 10: padding at the top and at the bottom |
| 62 | 2 | versions 4 to 10: padding on the left and on the right |
| 64 | 4 | versions 5 to 10: convolution groups `g` (at least 1, dividing `in_c` and `out_c`; also of transposed convolutions); versions 8 to 10: an attention's heads, an embedding's vocabulary (the rows of its table); 0 for other kinds |
| 68 | 4 | versions 5 to 10: the epsilon of batch (and layer, and RMS) normalization, a float in (0, 1); version 10: of an attention's normalization of its queries and keys (flag bit 2); 0 for other kinds |
| 72 | 4 | versions 5 to 10: batch normalization's momentum, a float in [0, 1] (used only when training); version 9: an attention's sliding window `w`, an integer (0: every key); 0 for other kinds |
| 76 | 4 | versions 7 to 10: upsampling mode, 0 nearest or 1 bilinear; versions 8 to 10: attention's flags (bit 0: causal; version 9: bit 1, padding ignored, with bits 16 to 31 the index of the embedding whose tokens tell it; version 10: bit 2, the queries and keys normalized, bit 3, rotary frequencies of its own; other bits 0), an embedding's flags (bit 0: positions); 0 for other kinds; versions 5 and 6: reserved, 0 |
| 80 | 4 | versions 6 to 10: `k`, the number of layers the entry reads, 1 to 16 (more than 1 only for additions, concatenations and multiplications; version 9: 2 for cross-attention) |
| 84 | 4 | versions 6 to 10: the first of them, a layer index from 0 (the network's input) to `i` |
| 88 | 8 | versions 6 to 10: offset of the input list, `k` 32-bit layer indices (the first equal to the one above, all from 0 to `i`), when `k` is 2 or more; 0 otherwise |
| 96 | 8 | versions 6 to 10: byte offset of the layer's output among the activations (a multiple of 16); 0 for the last layer |
| 104 | 4 | versions 8 to 10: an attention's key and value heads (at least 1, dividing its heads); version 9 also: a dense or convolution layer's tied embedding, the index of an embedding layer before it whose table are its weights (0: weights of its own), an embedding's token offset, the first of its input's values it reads; 0 for other kinds; versions 6 and 7: reserved, 0 |
| 108 | 4 | versions 8 to 10: an attention's base of the rotary position embeddings, a float at least 0 (0: none); version 10: an embedding's scale of its vectors, a float at least 0 (0: none); 0 for other kinds; versions 6 and 7: reserved, 0 |

The input shape of entry `i` is the output shape of entry `i - 1` in versions 4 and 5, and the
output shape of its first input in versions 6 to 10 (the input layer's shape from the header when that is
layer 0): `in_h x in_w x in_c` with `in_c = in / (in_h * in_w)`; the output channels are
`out_c = out / (out_h * out_w)`. Both divisions must be exact.

- Dense: output height and width 1, kernel, stride and padding 0.
- Convolution and pooling: kernel, stride and padding at least 1, 1 and 0, padding smaller than
  the kernel along each axis; the output has `out_h = (in_h + 2 pad_h - kernel_h) / stride_h + 1`
  rows (integer division; at least 1) and likewise `out_w` columns. Window `(oh, ow)` covers input
  rows `oh * stride_h - pad_h` to that plus `kernel_h - 1`, and columns likewise.
- Convolution: `out_c` filters of `kernel_h x kernel_w x (in_c / g)` weights (`g` is 1 in version
  4). The channels split into `g` groups: filter `j` belongs to group `j / (out_c / g)` and reads
  only the input channels of its group, `in_c / g` of them starting at that group times `in_c / g`.
  With `g = in_c` every filter sees a single channel (depthwise convolution).
- Pooling: `out_c = in_c`, activation 6 (none); the precision byte is the file's precision, which
  pooling does not use.
- Batch normalization (versions 5 to 7): the output shape equals the input shape, kernel, stride
  and padding 0, precision 0 (FLOAT32) whatever the file's precision; `out_c` channels, each with
  its gamma (the weights: `rows = out_c`, `n = 1`), its beta (the biases) and its running mean and
  variance (the scales section).
- Addition (version 6): `k` inputs whose outputs all have the entry's output shape, which equals
  the input shape; kernel, stride, padding, groups, epsilon and momentum 0, no sections. With one
  input it passes that input on (to give it an activation of its own).
- Concatenation (version 6): `k` inputs of the entry's output height and width whose channels add
  up to `out_c`; kernel, stride, padding, groups, epsilon and momentum 0, no sections.
- Global average pooling (version 6): output `1 x 1 x in_c`, activation 6 (none); kernel, stride,
  padding, groups, epsilon and momentum 0, no sections.
- Transposed convolution (version 7): kernel and stride at least 1, padding smaller than the kernel,
  groups `g` as for convolutions; the output has `out_h = (in_h - 1) stride_h - 2 pad_h + kernel_h +
  op_h` rows with an output padding `op_h` from 0 to `stride_h - 1` (not stored: what the output height
  leaves), and likewise columns. Sections as for convolutions: `out_c` filters (`rows = out_c`) of
  `kernel_h x kernel_w x (in_c / g)` weights, filter `j` holding at `(kh, kw, c)` the weight by which
  channel `c` of its group's input reaches output channel `j` at offset `(kh, kw)` of an input cell's
  window: input cell `(ih, iw)` adds to output cells from `(ih stride_h - pad_h, iw stride_w - pad_w)`
  on.
- Upsampling (version 7): the strides hold the factors (at least 1), `out_h = in_h stride_h` and
  `out_w = in_w stride_w`, `out_c = in_c`; kernel, padding, groups, epsilon and momentum 0, the mode
  at offset 76, activation 6 (none), no sections.
- Layer normalization (version 7): as batch normalization, without running statistics (no scales
  section) and with momentum 0: `out_c` channels with a gamma and a beta each, epsilon in (0, 1).
- Embedding (version 8): `in` values of the input (a token's index each, as a float), the output
  `in_h` rows of `in_w x in_c` cells of `out_c` channels (a sequence of `n` tokens: `1 x n x out_c`);
  kernel, stride and padding 0, epsilon and momentum 0. Its weights are the table, `rows` = the
  vocabulary (offset 64) rows of `n = out_c` values in the file's precision (with row scales for the
  integer ones); its biases section, when flag bit 0 (offset 76) is set, holds a vector of `out_c`
  floats for each output cell (`out` floats, the positions'), and is absent (offset 0) otherwise.
  Version 9: an embedding of part of its input reads, of an input of one row (`in_h` 1), the `out_w` values
  from its token offset `o` (offset 104) on, `o + out_w` at most `in_w x in_c` (an encoder's and a decoder's
  tokens side by side in one input, an embedding each).
- Attention (version 8): output shape `in_h x in_w x (heads x head)`, the input's channels `(heads +
  2 kv_heads) x head` for a head size `head` (even when the rotary base is not 0); no sections, no
  window, epsilon and momentum 0 (version 9: offset 72 the sliding window, any value). Version 9: with flag
  bit 1, keys of padding are left out: the embedding
  `e` of bits 16 to 31 (a layer before it, reading the network's input) has as many cells as the attention
  has keys, key `j`'s token being value `o + j` of the network's input (`o` the embedding's token offset).
  Cross-attention (version 9): `k` = 2, the first input the queries (`in_c` = `heads x head`), the second, a
  layer other than the network's input, the keys and values of its own cells (`2 kv_heads x head` channels,
  the key heads' then the value heads'), the first input a layer too; neither causal, nor windowed, nor
  rotary (offset 108 0), and its padding that of the second input's cells.
  Version 10: with flag bit 2 each head's queries and keys are RMS-normalized before their rotation (Qwen3's and
  Gemma 3's `q_norm` and `k_norm`): the weights section holds `2 head` floats, the queries' gammas then the keys',
  and offset 68 the epsilon; with flag bit 3 the scales section holds `rot / 2` floats, the rotation's frequencies
  (LLaMA 3's, linear scaling's), in place of `theta^(-2 i / rot)`; offset 52 the dimensions rotated `rot` (even, at
  most the head; 0: the head), offset 56 a float scale of the scores (0: `1 / sqrt(head)`). None of these apply to
  cross-attention.
- RMS normalization (version 8): as layer normalization without beta (no biases section).
- Multiplication (version 8): as addition, `k` inputs of the entry's output shape.
- A dense layer or convolution tied to an embedding (version 9, offset 104 the embedding's layer index `t`,
  from 1 to `i`): a dense layer, or a convolution of a 1 x 1 window, stride 1, no padding and one group,
  whose `rows` (its outputs or filters) are the embedding's vocabulary and whose `n` (its inputs, or the
  input's channels) are a row of the table: its weights and row scales offsets equal the embedding's
  (entry `t - 1`) and so does its precision; the table is stored once. Its biases are its own, and its
  optimizer state holds the moments of its biases alone (`2 x rows` floats: `m`, then `v`), the table's
  being the embedding's.

The precision byte of kinds without parameters is the file's precision, which they do not use.

Activation codes: 0 sigmoid, 1 ReLU, 2 tanh, 3 leaky ReLU (slope 0.01), 4 FOO52, 5 softmax,
6 none; from version 8 also 7 GELU (`x Phi(x)`, `Phi` through erf), 8 GELU's tanh approximation
(`x (1 + tanh(sqrt(2 / pi) (x + 0.044715 x^3))) / 2`) and 9 SiLU (`x / (1 + exp(-x))`).

## Sections

### Weights

`rows` rows of `n` weights each: for a dense layer, `rows = out` (one per output unit) and
`n = in`, row `j` holding the weights into unit `j` (row `j` of the `out x in` matrix `W`, so that
`y = activation(W x + b)`); for a convolution, `rows = out_c` (one per filter) and
`n = kernel_h x kernel_w x (in_c / g)`, row `j` holding filter `j` in window order: weight
`(kh * kernel_w + kw) * (in_c / g) + c` multiplies channel `c` of the filter's group at window row
`kh`, column `kw`; for a batch normalization, `rows = out_c` and `n = 1`: gamma per channel. Rows
are stored back to back without padding; what a row looks like depends on the precision:

| Code | Precision | Row size in bytes | Weight `k` of row `j` |
|---:|---|---|---|
| 0 | FLOAT32 | 4 `n` | the float itself |
| 1 | FP16 | 2 `n` | IEEE binary16 (written with round to nearest even) |
| 2 | BFLOAT16 | 2 `n` | the upper 16 bits of a float (written with round to nearest even) |
| 3 | INT8 | `n` | `q * scale[j]`, `q` a signed byte in -127..127 |
| 4 | INT4 | ceil(`n` / 2) | `q * scale[j]`; `q` is the 4-bit two's complement code in the low nibble of byte `k / 2` for even `k`, the high nibble for odd `k`, in -7..7 |
| 5 | INT2 | ceil(`n` / 4) | `q * scale[j]`; `q` is the 2-bit two's complement code at bits `2 (k mod 4)` of byte `k / 4`: 0 is 0, 1 is +1, 3 is -1 (2 is not written) |
| 6 to 17 | the block precisions (version 10): Q8_0, Q4_0, Q4_1, Q5_0, Q5_1, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, IQ4_NL, IQ4_XS | `n` / block x the block's bytes | **Block precisions** below |

Sections of FLOAT32 weights start at a multiple of 4 and those of FP16 and BFLOAT16 at a multiple
of 2 (all sections are 16-aligned anyway). Each row of INT4 and INT2 weights starts on a byte; the
unused bits of a row's last byte are 0.

### Block precisions

From version 10 the weights of dense layers, embeddings and linear layers (convolutions of a 1 x 1 window, stride
1, no padding and one group) may be in a block precision: GGUF's quantization types, each row stored as llama.cpp
stores a row of a tensor of that type (`block_q4_K` and the others of ggml), so that a GGUF file's tensors are
copied into an image as they are and back. A row of `n` weights, `n` a multiple of the type's block, is `n` /
block blocks back to back; such layers have no row scales (offset 24 is 0), their scales are in the blocks.
`d`, `dmin` and `m` below are IEEE binary16, all multi-byte values little-endian, and weight `k` of a block is:

| Code | Type | Block | Bytes | Layout | Weight `k` |
|---:|---|---:|---:|---|---|
| 6 | Q8_0 | 32 | 34 | `d`, `qs[32]` signed bytes | `d qs[k]` |
| 7 | Q4_0 | 32 | 18 | `d`, `qs[16]` | `d (c - 8)`, `c` the low nibble of `qs[k]` for `k` < 16, the high nibble of `qs[k - 16]` after |
| 8 | Q4_1 | 32 | 20 | `d`, `m`, `qs[16]` | `d c + m`, `c` as Q4_0's |
| 9 | Q5_0 | 32 | 22 | `d`, `qh` (32 bits), `qs[16]` | `d (c - 16)`, `c` Q4_0's nibble plus 16 times bit `k` of `qh` |
| 10 | Q5_1 | 32 | 24 | `d`, `m`, `qh`, `qs[16]` | `d c + m`, `c` as Q5_0's |
| 11 | Q2_K | 256 | 84 | `scales[16]`, `qs[64]`, `d`, `dmin` | for `k = 128 h + 32 j + l` (`l` < 32) the 2-bit code `c` at bits `2 j` of `qs[32 h + l]`, sub-block `s = k / 16`: `d (scales[s] & 15) c - dmin (scales[s] >> 4)` |
| 12 | Q3_K | 256 | 110 | `hmask[32]`, `qs[64]`, `scales[12]`, `d` | `c` as Q2_K's, minus 4 when bit `4 h + j` of `hmask[l]` is 0; `d (sc[s] - 32) c` with the 16 scales of 6 bits `sc` unpacked from `scales` as llama.cpp's `kmask1`/`kmask2` shuffle does |
| 13 | Q4_K | 256 | 144 | `d`, `dmin`, `scales[12]`, `qs[128]` | for `k = 64 i + 32 u + l` the low (`u` 0) or high (`u` 1) nibble `c` of `qs[32 i + l]`, sub-block `s = k / 32`: `d sc[s] c - dmin mn[s]`, the 6-bit `sc` and `mn` of `scales` as `get_scale_min_k4()` reads them |
| 14 | Q5_K | 256 | 176 | `d`, `dmin`, `scales[12]`, `qh[32]`, `qs[128]` | as Q4_K's with `c` plus 16 times bit `s` of `qh[l]` |
| 15 | Q6_K | 256 | 210 | `ql[128]`, `qh[64]`, `scales[16]` signed bytes, `d` | for `k = 128 h + 32 g + l` the low (`g` < 2) or high nibble of `ql[64 h + 32 (g mod 2) + l]` plus 16 times bits `2 g` of `qh[32 h + l]`, minus 32: `d scales[k / 16] c` |
| 16 | IQ4_NL | 32 | 18 | `d`, `qs[16]` | `d v[c]`, `c` as Q4_0's and `v` the table -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113 |
| 17 | IQ4_XS | 256 | 136 | `d`, `scales_h` (16 bits), `scales_l[4]`, `qs[128]` | sub-block `s = k / 32` of IQ4_NL's codes in `qs[16 s ..]`: `d (ls - 32) v[c]`, `ls` the nibble `s mod 2` of `scales_l[s / 2]` plus 16 times bits `2 s` of `scales_h` |

Writers (`spingalett_save()` in one of these precisions) choose the codes and scales as llama.cpp's quantizers do
without an importance matrix, byte for byte. A layer whose rows are not whole blocks of a k-quant or IQ4_XS
takes the type llama.cpp falls back to (Q2_K, Q3_K and IQ4_XS IQ4_NL, Q4_K Q5_0, Q5_K Q5_1, Q6_K Q8_0), and a
layer that takes no block precision (other kinds, rows not of whole blocks of 32) FP16; normalizations stay
FLOAT32.

Writers compute the scales per row (per output unit or filter):

- INT8: `scale = max |w| / 127`, `q = round(w / scale)` (to nearest even)
- INT4: `scale = max |w| / 7`, `q = round(w / scale)` clamped to -7..7
- INT2 (ternary weights): with `t = 0.7 * mean |w|`, `q = +1` for `w > t`, `-1` for `w < -t`, else
  0, and `scale` the mean of `|w|` over the weights with `q != 0`

A row of zeros gets scale 0. A row with a NaN or infinite weight gets scale NaN.

### Row scales

`rows` floats, one per weight row, for INT8, INT4 and INT2 weights.

### Running statistics (batch normalization)

`2 x out_c` floats: the running mean of each channel, then its running variance.

### Biases

`rows` floats (one per output unit or filter), whatever the weight precision.

### Optimizer state

Present when flag bit 0 is set: the first and second moment estimates of the optimizer, as floats,
in this order: `m` of the weights (`rows x n`, in row order), `v` of the weights, `m` of the
biases (`rows`), `v` of the biases. SGD leaves them 0, Momentum uses `m`, RMSProp uses `v`, Adam
and AdamW use both. For a batch normalization the weights are gamma and the biases beta.

## Validation

A reader should reject an image when the magic, the version or either checksum does not match;
when the recorded file size exceeds the data; when `L` is outside 2..65536 or the loss, an
activation, a precision or a layer kind code is unknown; when `in` or `out` is 0, an entry's `in`
differs from the previous entry's `out` or a dropout rate lies outside [0, 1); when a precision code above 5
appears before version 10, or a block precision is the precision of a layer with parameters other than a dense
layer, an embedding or a linear layer, whose rows are not whole blocks, or which has a row scales section; in versions 4 and 5,
when the shapes, windows, groups, normalization constants or section offsets disagree with the
layer kind as described above (a batch normalization in a version 4 file, a kind from 5 to 7 before
version 6, a kind above 7 before version 7, or a kind above 10, an activation code above 6 or loss 2
before version 8, is invalid); when a section offset is below 64, misaligned for its type or extends past
the file size; for integer layers but embeddings, when `n` exceeds 131072 (so that `127 * 127 * n`
fits a 32-bit accumulator); in version 9, when a tied layer names no embedding before it, or one whose
rows, row length, precision, weights or scales differ from its own, or is a convolution of another window,
or an attention's flags have bits other than those above, bit 1 without an embedding, or an embedding that
reads another layer than the input or has another number of cells than its keys, or a cross-attention reads the
network's input or keys and values of other channels, or is causal, windowed or rotary, or an embedding's range
lies beyond its input or its input has several rows; in version 10, when an attention's sections or fields of
version 10 disagree with its flags, its rotated dimensions are odd or more than its head, or a scale is negative or
not finite;
and in versions 6 to 10, when an entry reads a layer that is not before it,
more than 16 layers, several layers without being an addition, concatenation, multiplication or (version 9)
cross-attention, or
layers whose shapes do not
fit, when `in` differs from the units of its first input, or when an output (other than the last
one) does not lie within the activations, is misaligned or overlaps the output of an input of its
own layer.

## Checkpoints

`spingalett_save_checkpoint()` writes the network's image in FLOAT32 with the optimizer state and,
from the file size the header holds (offset 24) on, a record of the training run that trained it, as
its last epoch left it. Readers of the image stop at its file size, so `spingalett_load()`, the engine
and the runtime read a checkpoint as a model; `spingalett_load_checkpoint()` reads the record too, and
the `spingalett_train()` after it carries the run on. The record (integers little-endian, floats IEEE
single precision):

| Offset | Size | Field |
|---:|---:|---|
| 0 | 6 | magic `SLETTR` |
| 6 | 2 | record version, 1 |
| 8 | 4 | flags: bit 0 the run's state (offsets 16 to 151), bit 1 a reader's place (offsets 120 to 143), bit 2 the best epoch's parameters (from offset 152); other bits 0. Without bit 0 the fields to offset 151 are 0 |
| 12 | 4 | reserved, 0 |
| 16 | 8 | the network's optimizer steps at the end of the run's last epoch (the header's step count, offset 16) |
| 24 | 8 | epochs of the run completed |
| 32 | 8 | optimizer steps of the run |
| 40 | 8 | optimizer steps of all the run's epochs, which schedules by step are given (0: not yet known, for a generator without a sample count before its first epoch) |
| 48 | 8 | the seed of the dropout masks |
| 56 | 8 | the seed of the augmentations |
| 64 | 8 | the seed of the epochs' orders (epoch e's order is the permutation of this seed and e) |
| 72 | 4 | Adam's `beta1` raised to the steps, as the steps multiplied it |
| 76 | 4 | `beta2` likewise |
| 80 | 4 | the last learning rate of the schedule |
| 84 | 4 | the scale reduce-on-plateau reached (1 before a reduction) |
| 88 | 4 | the best monitored value (an infinity before the first) |
| 92 | 4 | reserved, 0 |
| 96 | 8 | the run's best epoch (0: none) |
| 104 | 8 | epochs without improvement |
| 112 | 8 | epochs since the plateau's last reduction |
| 120 | 24 | the place of the reader the run read through `spingalett_dataset_generator()`: its seed, pass and position (`SpingalettDatasetPosition`) |
| 144 | 8 | `n`: floats of the best epoch's parameters, 0 without bit 2 |
| 152 | 4 `n` | the best epoch's parameters, held for `restore_best_weights` when that epoch is not the last: the weights, the biases, then the running means and the running variances, laid out as in the network (`n` = weights + 3 x biases) |
| 152 + 4 `n` | 4 | CRC-32 of the record's bytes before it |

A file with bytes after its image that are not a record of this length, version and checksum does not
load as a checkpoint.

## Graphs of operators (versions 11 and 12)

A file of version 11 or 12 holds a graph (`docs/Graphs.md`): values (the graph's inputs, its constants and what its
nodes compute), nodes of operators that read and compute values, and the graph's inputs and outputs. It starts
with the magic and the version of the other versions and lays out the rest as follows.

```
header (128 bytes) | values (112 bytes each) | nodes (32 bytes each) | attributes (24 bytes each) | edges (4 bytes
each) | strings | numbers | constants (each at an offset of a multiple of 64)
```

A writer places the tables in this order; a reader relies only on the offsets. Strings are UTF-8 and end with a
NUL; a string is given by its offset in the strings, and offset 0 holds the empty string.

| Offset | Size | Field |
|---:|---:|---|
| 0 | 6 | magic `SLETTM` |
| 6 | 2 | format version, 11, or 12 where nodes use 1.6's attributes |
| 8 | 4 | `V`: values |
| 12 | 4 | `N`: nodes |
| 16 | 4 | `I`: the graph's inputs |
| 20 | 4 | `O`: the graph's outputs |
| 24 | 8 | file size in bytes |
| 32 | 4 | `S`: symbols (names of sizes, such as a batch) |
| 36 | 4 | `A`: attributes |
| 40 | 4 | the opset: the version of ONNX's default domain whose semantics the operators follow (Softmax and Hardmax before 13 over the axes from theirs on, Resize before 11 by asymmetric coordinates) |
| 44 | 4 | the producer (a string) |
| 48 | 8 | offset of the strings |
| 56 | 8 | bytes of the strings |
| 64 | 8 | offset of the values |
| 72 | 8 | offset of the nodes |
| 80 | 8 | offset of the edges |
| 88 | 8 | offset of the attributes |
| 96 | 8 | offset of the numbers |
| 104 | 8 | bytes of the numbers |
| 112 | 8 | the end of the tables (the constants come after it) |
| 120 | 4 | CRC-32 of bytes 128 to the end of the tables - 1 |
| 124 | 4 | CRC-32 of bytes 0 to 123 |

The checksums cover the header and the tables, not the constants, so that a graph of gigabytes opens without its
constants being read whole.

**Values** (112 bytes each, value `v` at `values + 112 v`):

| Offset | Size | Field |
|---:|---:|---|
| 0 | 4 | name (a string) |
| 4 | 1 | type: ONNX's codes (`SpingalettDType`): 1 float32, 2 uint8, 3 int8, 6 int32, 7 int64, 9 bool (a byte, 0 or 1), 10 float16, 16 bfloat16 |
| 5 | 1 | rank, 0 to 8 |
| 6 | 1 | kind: 0 an input of the graph, 1 a constant, 2 computed by a node |
| 7 | 1 | flags: bit 0 set when the type and shape are known (inputs, constants, declared outputs) |
| 8 | 64 | the sizes of the eight axes (int64; -1 for a size a symbol or nothing gives; axes past the rank -1) |
| 72 | 16 | the symbols of the eight axes (uint16: a symbol's number + 1, 0 for none) |
| 88 | 8 | constants: the offset of the values, a multiple of 64 at or after the end of the tables |
| 96 | 8 | constants: the bytes of the values (the product of the sizes times the type's size) |
| 104 | 8 | reserved, 0 |

A constant's values are in row-major order, little-endian, in its type (float16 and bfloat16 as their bits).

**Nodes** (32 bytes each), in an order in which every node runs after the nodes computing what it reads:

| Offset | Size | Field |
|---:|---:|---|
| 0 | 2 | the operator's code (below) |
| 2 | 2 | reserved, 0 |
| 4 | 4 | name (a string) |
| 8 | 4 | its first edge: its inputs' values, then its outputs', from there in the edges |
| 12 | 4 | inputs |
| 16 | 4 | outputs |
| 20 | 4 | its first attribute |
| 24 | 4 | attributes |
| 28 | 4 | reserved, 0 |

**Edges** are values' numbers (uint32), 0xFFFFFFFF for an optional input left out: the nodes' inputs and outputs,
then the graph's `I` inputs and `O` outputs, then the `S` symbols' names (strings).

**Attributes** (24 bytes each):

| Offset | Size | Field |
|---:|---:|---|
| 0 | 4 | name (a string) |
| 4 | 1 | type: 0 an integer, 1 a float, 2 integers, 3 floats, 4 a string, 5 a tensor |
| 5 | 3 | reserved, 0 |
| 8 | 8 | an integer (int64); a float's bits (in the low 4 bytes); integers and floats: their offset in the numbers (a multiple of 8 and of 4); a string's offset; a tensor's value (a constant) |
| 16 | 4 | integers and floats: how many (int64 and floats each) |
| 20 | 4 | reserved, 0 |

The operators are ONNX's, with ONNX's attributes and inputs, and three of the library's own, which the fusions of
a graph that opens make (`Src/Spingalett.IRFuse.c`): **Linear** (`X`, `W`, an optional bias: `X` (..., K) times
the constant `W` (K x N; with the integer attribute `trans_b` 1, N x K and taken transposed) plus the bias, through
the activation of the integer attribute `activation`, an `ActivationFunction`), **FusedAttention** (`Q`, `K`, `V`, an optional mask: softmax(`Q K^T` times the float
attribute `scale` plus the mask) `V`, the softmax over the keys, rows of no keys 0 with the integer attribute
`safe` 1; with `layout` 0 `Q` (..., S, D), the second input `K^T` (..., D, T) and `V` (..., T, Dv) over broadcast
batches, with `layout` 1 `Q` (B, S, H, D), `K` (B, T, H or 1, D) and `V` (B, T, H or 1, Dv) into (B, S, H, Dv); with
`layout` 2 (version 12) Q, K and V read in their inputs (which may be one value) where the 18 integers `views` put
them: B, H, S, T, D and Dv, then for each of Q, K and V an offset in elements and its strides over B, over H and
between rows (a query's, a key's, a value's), the values of a row contiguous; the output has the integers `shape`,
(B, S, H, Dv) in memory with the integer `heads` 1, else (B, H, S, Dv)), and Conv's integer attribute `activation`,
applied after the convolution and its bias. Version 12 adds the integer attribute `channels_last` 1 of Conv,
MaxPool, AveragePool, GroupNormalization and InstanceNormalization (their maps [N, spatial..., C] in and out, the
filters still ONNX's), and the `activation` of GroupNormalization and InstanceNormalization (after the scale and
bias). Codes:

| | | | | | |
|---|---|---|---|---|---|
| 0 Add | 1 Sub | 2 Mul | 3 Div | 4 Pow | 5 Mod |
| 6 Max | 7 Min | 8 Sum | 9 Mean | 10 Equal | 11 Less |
| 12 LessOrEqual | 13 Greater | 14 GreaterOrEqual | 15 And | 16 Or | 17 Xor |
| 18 PRelu | 19 Where | 20 Abs | 21 Neg | 22 Exp | 23 Log |
| 24 Sqrt | 25 Reciprocal | 26 Sin | 27 Cos | 28 Tanh | 29 Sigmoid |
| 30 Erf | 31 Relu | 32 LeakyRelu | 33 Elu | 34 Selu | 35 Gelu |
| 36 HardSigmoid | 37 HardSwish | 38 Softplus | 39 Softsign | 40 Mish | 41 Floor |
| 42 Ceil | 43 Round | 44 Sign | 45 Not | 46 IsNaN | 47 IsInf |
| 48 Tan | 49 Asin | 50 Acos | 51 Atan | 52 Sinh | 53 Cosh |
| 54 Asinh | 55 Acosh | 56 Atanh | 57 Celu | 58 ThresholdedRelu | 59 Shrink |
| 60 Swish | 61 BitwiseNot | 62 BitwiseAnd | 63 BitwiseOr | 64 BitwiseXor | 65 BitShift |
| 66 SwiGLU | 67 Clip | 68 Cast | 69 CastLike | 70 Identity | 71 Dropout |
| 72 MatMul | 73 Gemm | 74 Conv | 75 ConvTranspose | 76 MaxPool | 77 AveragePool |
| 78 LpPool | 79 GlobalAveragePool | 80 GlobalMaxPool | 81 BatchNormalization | 82 InstanceNormalization | 83 LayerNormalization |
| 84 GroupNormalization | 85 RMSNormalization | 86 LpNormalization | 87 MeanVarianceNormalization | 88 LRN | 89 Softmax |
| 90 LogSoftmax | 91 Hardmax | 92 ReduceSum | 93 ReduceMean | 94 ReduceMax | 95 ReduceMin |
| 96 ReduceProd | 97 ReduceSumSquare | 98 ReduceL1 | 99 ReduceL2 | 100 ReduceLogSumExp | 101 ReduceLogSum |
| 102 ArgMax | 103 ArgMin | 104 TopK | 105 CumSum | 106 CumProd | 107 Reshape |
| 108 Flatten | 109 Squeeze | 110 Unsqueeze | 111 Transpose | 112 Concat | 113 Split |
| 114 Slice | 115 Gather | 116 GatherElements | 117 GatherND | 118 ScatterND | 119 ScatterElements |
| 120 Scatter | 121 TensorScatter | 122 Expand | 123 Tile | 124 Pad | 125 Shape |
| 126 Size | 127 Range | 128 ConstantOfShape | 129 Constant | 130 Trilu | 131 Resize |
| 132 Upsample | 133 DepthToSpace | 134 SpaceToDepth | 135 NonZero | 136 OneHot | 137 EyeLike |
| 138 Compress | 139 ReverseSequence | 140 BlackmanWindow | 141 HannWindow | 142 HammingWindow | 143 Einsum |
| 144 QuantizeLinear | 145 DequantizeLinear | 146 DynamicQuantizeLinear | 147 MatMulInteger | 148 QLinearMatMul | 149 Attention |
| 150 RotaryEmbedding | 151 LinearAttention | 152 CausalConvWithState | 153 Linear | 154 FusedAttention |  |

A reader should reject a file whose magic, version or either checksum does not match; whose recorded size exceeds
the data; whose tables, strings or numbers lie outside the file, or whose strings do not end with a NUL; with a
value of an unknown type, kind, a rank above 8, a size below -1 or a symbol beyond `S`; with a constant whose bytes
differ from its shape's, lie before the end of the tables or outside the file, or whose offset is no multiple of 64;
with a node of an unknown operator, or whose edges or attributes lie beyond the tables, or that reads or writes a
value beyond `V`; with an attribute of an unknown type, numbers outside the numbers or a tensor that is not a
constant; with an input that is not a value of kind 0; and with a graph whose nodes compute a value twice, read a
value that nothing computes, or read each other in a cycle.

## How the inference engine computes

The engine evaluates the layers in order; the output of the last one is the network's output.
Before version 6 every layer reads the output of the one before it, and the outputs alternate
between two buffers of the widest hidden layer's size. From version 6 on every output other than the
last one lives among the activations of the workspace (the header's byte count), at the offset its
entry gives, from the moment its layer runs until the last layer reading it has run; the writer
chooses the offsets so that outputs alive at the same time do not overlap (Spingalett places each
at the lowest offset free for its life, so that a deep residual network needs a few of its widest
outputs, not all of them). Each output takes its units times 4 bytes, rounded up to a multiple of
16.

- FLOAT32, FP16 and BFLOAT16 layers compute `y = b + W x` in float, with the weights converted to
  float as they are read. A tied layer computes as any other with the table as `W` (the logits of a
  language model's head: `y_j = b_j + E_j . x`, `E_j` the row of token `j`).
- INT8, INT4 and INT2 layers quantize their input `x` (`in` floats, the whole tensor) to bytes
  first: `s = max |x| / 127` and `xq = round(x / s)` to nearest even (all `xq` 0 when `x` is all 0),
  then `y[j] = b[j] + (scale[j] * s) * sum_k q[j][k] * xq[k]` with the sum in 32-bit integers.
  In a language model (a model whose first layer is an embedding: its cells are tokens) a convolution
  of 1 x 1 windows, stride 1, no padding and one group (a linear map of each cell) quantizes each
  cell's channels alone instead, with an `s` of their own, so that a token's outputs depend on that
  token only (engines before Spingalett 1.3 quantize the whole tensor there too).
- Layers of a block precision quantize their input in blocks of the type's size (32 or 256 values; a linear
  layer's cells never share a block): each block as INT8 layers quantize a tensor, `dx = max |x| / 127` and
  `xq = round(x / dx)`, with `X_i` the sum of the codes `16 i` to `16 i + 15`. A row's output is then `b[j]`
  plus a float sum over its blocks in order, starting at 0: a block of weight codes `c` (the signed codes,
  offsets applied, of the table above), of sub-blocks of 16 or 32 with integer scales `sc` (1 for the types of
  32; Q3_K's and IQ4_XS's less 32) and minima `mn`, adds `t = (d dx) I1`, plus `(m dx) I2` for the types with
  minima, where `I1 = sum over sub-blocks of sc (sum of c xq)` and `I2 = sum over sub-blocks of mn X` are 32-bit
  integers (`X` a sub-block's sum of codes; `m` is Q4_1's and Q5_1's `m` with `mn` 1, and `-dmin` for Q2_K,
  Q4_K and Q5_K), each product of two floats rounded, `I1` and `I2` converted to float. Every kernel of the
  library computes these bits.
- A convolution computes, for each output pixel `(oh, ow)`, its `out_c` outputs as above with `x`
  the pixel's window: `kernel_h x kernel_w x in_c` values in the order of the filter rows, 0 where
  the window lies in the padding (for integer layers, the window of the quantized input); with
  groups, each filter takes the part of the window that holds its group's channels.
- Batch normalization computes, per channel `c`, `a = gamma / sqrt(var + eps)` and
  `b = beta - mean * a` in float, then `y = a x + b` for every cell of the channel.
- A transposed convolution computes, for each output pixel `(oh, ow)`, its outputs as a convolution
  does with `x` the window of inputs that reach the pixel: for tap `(kh, kw)`, in the order of the
  filter rows, input cell `((oh + pad_h - kh) / stride_h, (ow + pad_w - kw) / stride_w)` when both
  divisions are exact and the cell lies inside the input, 0 otherwise (for integer layers, from the
  quantized input); with groups, each filter takes its group's channels.
- Upsampling copies cell `(oh / stride_h, ow / stride_w)` (mode 0), or interpolates bilinearly
  (mode 1): output row `o` reads input rows `r0 = floor(t)` and `r1 = min(r0 + 1, in_h - 1)` with
  `t = (2 o + 1 - stride_h) / (2 stride_h)` (rows with `t <= 0` read row 0 alone), the second
  weighted `wr = t - r0`, computed as the integer remainder of `2 o + 1 - stride_h` by `2 stride_h`
  divided by `2 stride_h` in float; columns likewise, and per channel
  `y = (1 - wr) ((1 - wc) x00 + wc x01) + wr ((1 - wc) x10 + wc x11)` in float in that order.
- Layer normalization computes, per cell, `mean = (sum over c of x) * (1 / C)` and
  `var = (sum over c of (x - mean)^2) * (1 / C)` in float, the sums in channel order, then
  `y = (x - mean) * (1 / sqrt(var + eps)) * gamma[c] + beta[c]`.
- RMS normalization computes, per cell, `r = 1 / sqrt((sum over c of x^2) * (1 / C) + eps)` in float
  (the sum in channel order), then `y = x * r * gamma[c]`.
- An embedding copies, for each input value `t` (version 9: of its range), row `t` of its table (decoded as
  the weights of a dense layer are) when `0 <= t < vocabulary` (`t` taken as an integer), zeros otherwise,
  multiplies it by its scale when it has one (version 10: Gemma's square root of the width), then adds the
  position's vector of its cell when it has them.
- Attention computes, for each query head `h` (reading key and value head `g = h / (heads /
  kv_heads)`) and each cell `i`, the scores `s_j = (q_i . k_j) / sqrt(head)` over the cells `j` (up to
  `i` when causal) in order, each dot product summed in order, and `y = (sum_j exp(s_j - m) v_j) /
  (sum_j exp(s_j - m))` with the softmax taken online: `m` the largest score so far, the sums scaled
  by `exp(m_old - m_new)` when it grows. With a rotary base `theta > 0`, queries and keys are first
  rotated: for `i` below `half = head / 2`, `x'_i = x_i cos - x_(i + half) sin` and `x'_(i + half) =
  x_(i + half) cos + x_i sin` with the angle `p theta^(-2 i / head)` of the cell's position `p`, its
  cosine and sine computed in double and rounded to float (version 10: the first `rot` values, pairs `i` and
  `i + rot / 2`, with the frequencies of the scales section when it has them, after each head's normalization with
  its gammas when it has one; the scores times the layer's own scale when it has one). The exponentials are
  computed as glibc's `expf` computes them: in double, `exp(x) = 2^(k / 32) 2^(r / 32)` of Arm's optimized
  routines, without fused multiply-adds, rounded to float once. With a sliding window `w > 0` (version 9), cell
  `i` reads the keys `j` with `j + w > i`, and `j < i + w` when not causal (a window of at least the cells is
  no window). Ignoring padding (version 9), the keys whose tokens are negative (not 0 or more) are left out
  of the sums as if they were not there; a query left without a key gives zeros. Cross-attention (version 9)
  computes the same with the queries of its first input and the keys and values of its second, every query
  over every key of its sample.
- A multiplication multiplies its inputs in their order (`x0 * x1`, then `* x2` and so on).
- Pooling takes, per channel, the maximum or the mean of the window's cells inside the input;
  padding cells are not counted (a window of 2 x 2 cells over one row of padding averages 2 cells).
- An addition sums its inputs in their order (`x0 + x1`, then `+ x2` and so on), a concatenation
  puts the channels of each cell of its inputs side by side in their order, and global average
  pooling sums each channel over the cells in order and multiplies the sum by `1 / (in_h * in_w)`.

The layer's activation is then applied to `y`; softmax over the whole layer. The engine needs
scratch beyond the layers' outputs for an attention with rotary embeddings: the rotated keys of a
sample (`cells x kv_heads x head` floats), a query and the table of `cells x head` cosines and
sines.

## Versions 1 and 2

Files written before Spingalett 0.5 have no magic: they start with the format version as a 16-bit
integer (1 or 2) and store everything in the byte order of the machine that wrote them, without
alignment or checksums. After the version come the layer count (32 bits), the loss (8 bits), an
optimizer flag (8 bits), the time step (64 bits, only with the flag), the precision (8 bits), the
`L` layer sizes (32 bits each), `L - 1` activation codes (8 bits each) and, in version 2, `L - 1`
dropout rates (floats). Then, for each weight layer, the weights, `m` and `v` of the weights (with
the flag), the biases, and `m` and `v` of the biases (with the flag), each array in the file's
precision; INT8, INT4 and INT2 arrays are preceded by one float `max |value|` for the whole array
and decode as `q / 127`, `q / 7` and `q` times it. Reading such a file and saving it again gives a
version 3 file (`ModelTool convert`).

Spingalett 0.5 and 0.6 read versions 1 to 3; version 4 appeared in 0.7, version 5 in 0.8,
version 6 in 0.10, version 7 in 0.13, version 8 in 1.2, version 9 in 1.3, version 10 in 1.4, version 11 in 1.5
and version 12 in 1.6.
