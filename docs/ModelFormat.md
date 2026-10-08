# The .slett model format, versions 3 to 6

`.slett` files store a network: its shape, its parameters in a chosen precision and, optionally,
the optimizer state for resuming training. `save_spingalett()` and `spingalett_save_to_memory()`
write the oldest version that can hold the network: version 3 for networks of dense layers only
(which engines from Spingalett 0.5 on can run), version 4 for networks with convolution or pooling
layers (0.7 on), version 5 for networks with batch normalization or grouped convolutions (0.8 on),
version 6 for graphs: networks with a layer that reads other layers than the one before it, adds or
concatenates several, or pools globally (0.10 on). `load_spingalett()` and
`load_spingalett_from_memory()` read versions 1 to 6.

Versions 3 to 5 are laid out so that a file image can be used as it is: the inference engine
(`spingalett_model_init()` in `Spingalett.Inference.h`) checks the image and computes directly from
it, so a model can be read into memory, compiled into a program as a byte array, or executed from
flash. This document describes the layout precisely enough to write an independent reader.
Version 4 is version 3 with a larger layer table entry that adds the layer's kind and shape;
version 5 extends the entry again with convolution groups and the constants of batch
normalization; version 6 adds the layers each layer reads and where its output lives in the
engine's workspace. The differences are marked below.

Batch normalization is folded when a network is saved for deployment: files written in a precision
other than FLOAT32 without optimizer state, and the models of `spingalett_model_from_network()`,
merge every batch normalization that reads a dense or convolution layer without activation which
feeds nothing else into that layer (its weight rows scaled by `gamma / sqrt(var + eps)` and its
biases shifted alike; the layer takes the normalization's activation, and the layers that read the
normalization read it instead). Other normalizations remain layers of their own, so such files can
still be version 3 or 4.

All integers are little-endian. "float" means an IEEE 754 binary32 stored as its bit pattern.
CRC-32 is the IEEE polynomial (0xEDB88320, reflected, initial value and final XOR 0xFFFFFFFF), as
in zlib and PNG.

## Overview

```
header (64 bytes) | layer table (48 bytes per entry in version 3, 64 in version 4, 80 in version 5,
112 in version 6) | sections ...
```

A network of `L` layers (the input layer included) has `L - 1` table entries; entry `i` (counted
from 0) describes layer `i + 1` and what feeds it: the output of layer `i` in versions 3 to 5, the
outputs of the layers it names in version 6 (always layers before it, so that the table order is an
order in which every layer can be computed after its inputs). Each dense or convolution layer
owns up to four sections: its weights, the per-row scales of integer weights, its biases and its
optimizer state; a batch normalization owns the same four, its scales section holding its running
statistics; pooling, adding and concatenating layers own none. In version 6, an entry that reads
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
| 6 | 2 | format version, 3 to 6 |
| 8 | 4 | `L`: number of layers, input layer included (2 to 65536) |
| 12 | 1 | loss function: 0 mean squared error, 1 cross-entropy |
| 13 | 1 | flags: bit 0 set when the file holds optimizer state; other bits 0 |
| 14 | 2 | reserved, 0 |
| 16 | 8 | optimizer time step (number of steps taken; 0 without optimizer state) |
| 24 | 8 | file size in bytes |
| 32 | 4 | versions 4 to 6: height of the input layer (1 to 65535); version 3: reserved, 0 |
| 36 | 4 | versions 4 to 6: width of the input layer (1 to 65535); version 3: reserved, 0 |
| 40 | 8 | version 6: bytes of activations, the part of the engine's workspace that holds the layers' outputs (see below); versions 3 to 5: reserved, 0 |
| 48 | 8 | reserved, 0 |
| 56 | 4 | CRC-32 of bytes 64 to file size - 1 (the layer table and all sections) |
| 60 | 4 | CRC-32 of bytes 0 to 59 |

Data after the recorded file size (for example the rest of a flash region) is not part of the
image.

The input layer's channels are its units (`in` of the first entry) divided by height x width.

## Layer table

`L - 1` entries start at offset 64, 48 bytes each in version 3, 64 bytes in version 4, 80 bytes
in version 5 and 112 bytes in version 6:

| Offset | Size | Field |
|---:|---:|---|
| 0 | 4 | `in`: units of layer `i` (for the first entry, the network's input size); version 6: units of the entry's first input |
| 4 | 4 | `out`: units of layer `i + 1`; in versions 3 to 5 equals `in` of the next entry |
| 8 | 1 | activation of layer `i + 1` (codes below) |
| 9 | 1 | weight precision (codes below) |
| 10 | 1 | versions 4 to 6: kind of layer `i + 1`: 0 dense, 1 convolution, 2 max pooling, 3 average pooling, 4 batch normalization (versions 5 and 6), 5 addition, 6 concatenation, 7 global average pooling (version 6); version 3: reserved, 0 |
| 11 | 1 | reserved, 0 |
| 12 | 4 | dropout rate of layer `i + 1`, a float in [0, 1) (used only when training) |
| 16 | 8 | offset of the weights; 0 for kinds without parameters (pooling, addition, concatenation) |
| 24 | 8 | offset of the row scales (batch normalization: of its running statistics); 0 for FLOAT32, FP16 and BFLOAT16 dense and convolution layers, and for kinds without parameters |
| 32 | 8 | offset of the biases; 0 for kinds without parameters |
| 40 | 8 | offset of the optimizer state; 0 when the file has none, and for kinds without parameters |
| 48 | 2 | versions 4 to 6: output height of layer `i + 1` |
| 50 | 2 | versions 4 to 6: output width of layer `i + 1` |
| 52 | 2 | versions 4 to 6: kernel height (window rows) |
| 54 | 2 | versions 4 to 6: kernel width (window columns) |
| 56 | 2 | versions 4 to 6: vertical stride |
| 58 | 2 | versions 4 to 6: horizontal stride |
| 60 | 2 | versions 4 to 6: padding at the top and at the bottom |
| 62 | 2 | versions 4 to 6: padding on the left and on the right |
| 64 | 4 | versions 5 and 6: convolution groups `g` (at least 1, dividing `in_c` and `out_c`); 0 for other kinds |
| 68 | 4 | versions 5 and 6: batch normalization's epsilon, a float in (0, 1); 0 for other kinds |
| 72 | 4 | versions 5 and 6: batch normalization's momentum, a float in [0, 1] (used only when training); 0 for other kinds |
| 76 | 4 | versions 5 and 6: reserved, 0 |
| 80 | 4 | version 6: `k`, the number of layers the entry reads, 1 to 16 (more than 1 only for additions and concatenations) |
| 84 | 4 | version 6: the first of them, a layer index from 0 (the network's input) to `i` |
| 88 | 8 | version 6: offset of the input list, `k` 32-bit layer indices (the first equal to the one above, all from 0 to `i`), when `k` is 2 or more; 0 otherwise |
| 96 | 8 | version 6: byte offset of the layer's output among the activations (a multiple of 16); 0 for the last layer |
| 104 | 8 | version 6: reserved, 0 |

The input shape of entry `i` is the output shape of entry `i - 1` in versions 4 and 5, and the
output shape of its first input in version 6 (the input layer's shape from the header when that is
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
- Batch normalization (versions 5 and 6): the output shape equals the input shape, kernel, stride
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

The precision byte of kinds without parameters is the file's precision, which they do not use.

Activation codes: 0 sigmoid, 1 ReLU, 2 tanh, 3 leaky ReLU (slope 0.01), 4 FOO52, 5 softmax,
6 none.

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

Sections of FLOAT32 weights start at a multiple of 4 and those of FP16 and BFLOAT16 at a multiple
of 2 (all sections are 16-aligned anyway). Each row of INT4 and INT2 weights starts on a byte; the
unused bits of a row's last byte are 0.

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
differs from the previous entry's `out` or a dropout rate lies outside [0, 1); in versions 4 and 5,
when the shapes, windows, groups, normalization constants or section offsets disagree with the
layer kind as described above (a batch normalization in a version 4 file, or a kind above 4 before
version 6, is invalid); when a section offset is below 64, misaligned for its type or extends past
the file size; for integer layers, when `n` exceeds 131072 (so that `127 * 127 * n` fits a 32-bit
accumulator); and in version 6, when an entry reads a layer that is not before it, more than 16
layers, several layers without being an addition or concatenation, or layers whose shapes do not
fit, when `in` differs from the units of its first input, or when an output (other than the last
one) does not lie within the activations, is misaligned or overlaps the output of an input of its
own layer.

## How the inference engine computes

The engine evaluates the layers in order; the output of the last one is the network's output.
Before version 6 every layer reads the output of the one before it, and the outputs alternate
between two buffers of the widest hidden layer's size. In version 6 every output other than the
last one lives among the activations of the workspace (the header's byte count), at the offset its
entry gives, from the moment its layer runs until the last layer reading it has run; the writer
chooses the offsets so that outputs alive at the same time do not overlap (Spingalett places each
at the lowest offset free for its life, so that a deep residual network needs a few of its widest
outputs, not all of them). Each output takes its units times 4 bytes, rounded up to a multiple of
16.

- FLOAT32, FP16 and BFLOAT16 layers compute `y = b + W x` in float, with the weights converted to
  float as they are read.
- INT8, INT4 and INT2 layers quantize their input `x` (`in` floats, the whole tensor) to bytes
  first: `s = max |x| / 127` and `xq = round(x / s)` to nearest even (all `xq` 0 when `x` is all 0),
  then `y[j] = b[j] + (scale[j] * s) * sum_k q[j][k] * xq[k]` with the sum in 32-bit integers.
- A convolution computes, for each output pixel `(oh, ow)`, its `out_c` outputs as above with `x`
  the pixel's window: `kernel_h x kernel_w x in_c` values in the order of the filter rows, 0 where
  the window lies in the padding (for integer layers, the window of the quantized input); with
  groups, each filter takes the part of the window that holds its group's channels.
- Batch normalization computes, per channel `c`, `a = gamma / sqrt(var + eps)` and
  `b = beta - mean * a` in float, then `y = a x + b` for every cell of the channel.
- Pooling takes, per channel, the maximum or the mean of the window's cells inside the input;
  padding cells are not counted (a window of 2 x 2 cells over one row of padding averages 2 cells).
- An addition sums its inputs in their order (`x0 + x1`, then `+ x2` and so on), a concatenation
  puts the channels of each cell of its inputs side by side in their order, and global average
  pooling sums each channel over the cells in order and multiplies the sum by `1 / (in_h * in_w)`.

The layer's activation is then applied to `y`; softmax over the whole layer.

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

Spingalett 0.5 and 0.6 read versions 1 to 3; version 4 appeared in 0.7, version 5 in 0.8 and
version 6 in 0.10.
