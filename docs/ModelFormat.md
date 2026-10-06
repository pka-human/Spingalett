# The .slett model format, version 3

`.slett` files store a network: its shape, its parameters in a chosen precision and, optionally,
the optimizer state for resuming training. `save_spingalett()` and `spingalett_save_to_memory()`
write version 3; `load_spingalett()` and `load_spingalett_from_memory()` read versions 1 to 3.

Version 3 is laid out so that a file image can be used as it is: the inference engine
(`spingalett_model_init()` in `Spingalett.Inference.h`) checks the image and computes directly from
it, so a model can be read into memory, compiled into a program as a byte array, or executed from
flash. This document describes the layout precisely enough to write an independent reader.

All integers are little-endian. "float" means an IEEE 754 binary32 stored as its bit pattern.
CRC-32 is the IEEE polynomial (0xEDB88320, reflected, initial value and final XOR 0xFFFFFFFF), as
in zlib and PNG.

## Overview

```
header (64 bytes) | layer table (48 bytes per weight layer) | sections ...
```

A network of `L` layers (the input layer included) has `L - 1` weight layers; weight layer `i`
(counted from 0) connects layer `i` to layer `i + 1`. Each weight layer owns up to four sections:
its weights, the per-row scales of integer weights, its biases and its optimizer state. Every
section starts at an offset that is a multiple of 16; the bytes between sections are 0. A writer
places the sections in table order, but a reader must only rely on the offsets.

## Header

| Offset | Size | Field |
|---:|---:|---|
| 0 | 6 | magic `SLETTM` |
| 6 | 2 | format version, 3 |
| 8 | 4 | `L`: number of layers, input layer included (2 to 65536) |
| 12 | 1 | loss function: 0 mean squared error, 1 cross-entropy |
| 13 | 1 | flags: bit 0 set when the file holds optimizer state; other bits 0 |
| 14 | 2 | reserved, 0 |
| 16 | 8 | optimizer time step (number of steps taken; 0 without optimizer state) |
| 24 | 8 | file size in bytes |
| 32 | 24 | reserved, 0 |
| 56 | 4 | CRC-32 of bytes 64 to file size - 1 (the layer table and all sections) |
| 60 | 4 | CRC-32 of bytes 0 to 59 |

Data after the recorded file size (for example the rest of a flash region) is not part of the
image.

## Layer table

`L - 1` entries of 48 bytes start at offset 64:

| Offset | Size | Field |
|---:|---:|---|
| 0 | 4 | `in`: units of layer `i` (for the first entry, the network's input size) |
| 4 | 4 | `out`: units of layer `i + 1`; equals `in` of the next entry |
| 8 | 1 | activation of layer `i + 1` (codes below) |
| 9 | 1 | weight precision (codes below) |
| 10 | 2 | reserved, 0 |
| 12 | 4 | dropout rate of layer `i + 1`, a float in [0, 1) (used only when training) |
| 16 | 8 | offset of the weights |
| 24 | 8 | offset of the row scales; 0 for FLOAT32, FP16 and BFLOAT16 |
| 32 | 8 | offset of the biases |
| 40 | 8 | offset of the optimizer state; 0 when the file has none |

Activation codes: 0 sigmoid, 1 ReLU, 2 tanh, 3 leaky ReLU (slope 0.01), 4 FOO52, 5 softmax,
6 none.

## Sections

### Weights

`out` rows, one per output unit, each holding the `in` weights into that unit (row `j` of the
`out x in` matrix `W`, so that `y = activation(W x + b)`). Rows are stored back to back without
padding; what a row looks like depends on the precision:

| Code | Precision | Row size in bytes | Weight `k` of row `j` |
|---:|---|---|---|
| 0 | FLOAT32 | 4 `in` | the float itself |
| 1 | FP16 | 2 `in` | IEEE binary16 (written with round to nearest even) |
| 2 | BFLOAT16 | 2 `in` | the upper 16 bits of a float (written with round to nearest even) |
| 3 | INT8 | `in` | `q * scale[j]`, `q` a signed byte in -127..127 |
| 4 | INT4 | ceil(`in` / 2) | `q * scale[j]`; `q` is the 4-bit two's complement code in the low nibble of byte `k / 2` for even `k`, the high nibble for odd `k`, in -7..7 |
| 5 | INT2 | ceil(`in` / 4) | `q * scale[j]`; `q` is the 2-bit two's complement code at bits `2 (k mod 4)` of byte `k / 4`: 0 is 0, 1 is +1, 3 is -1 (2 is not written) |

Sections of FLOAT32 weights start at a multiple of 4 and those of FP16 and BFLOAT16 at a multiple
of 2 (all sections are 16-aligned anyway). Each row of INT4 and INT2 weights starts on a byte; the
unused bits of a row's last byte are 0.

Writers compute the scales per row (per output unit):

- INT8: `scale = max |w| / 127`, `q = round(w / scale)` (to nearest even)
- INT4: `scale = max |w| / 7`, `q = round(w / scale)` clamped to -7..7
- INT2 (ternary weights): with `t = 0.7 * mean |w|`, `q = +1` for `w > t`, `-1` for `w < -t`, else
  0, and `scale` the mean of `|w|` over the weights with `q != 0`

A row of zeros gets scale 0. A row with a NaN or infinite weight gets scale NaN.

### Row scales

`out` floats, one per weight row, for INT8, INT4 and INT2 weights.

### Biases

`out` floats, whatever the weight precision.

### Optimizer state

Present when flag bit 0 is set: the first and second moment estimates of the optimizer, as floats,
in this order: `m` of the weights (`out x in`, in row order), `v` of the weights, `m` of the
biases (`out`), `v` of the biases. SGD leaves them 0, Momentum uses `m`, RMSProp uses `v`, Adam and
AdamW use both.

## Validation

A reader should reject an image when the magic, the version or either checksum does not match;
when the recorded file size exceeds the data; when `L` is outside 2..65536 or the loss, an
activation or a precision code is unknown; when `in` or `out` is 0, an entry's `in` differs from
the previous entry's `out` or a dropout rate lies outside [0, 1); when a section offset is below
64, misaligned for its type or extends past the file size; and, for integer layers, when `in`
exceeds 131072 (so that `127 * 127 * in` fits a 32-bit accumulator).

## How the inference engine computes

The engine evaluates the layers in order; the output of the last one is the network's output.

- FLOAT32, FP16 and BFLOAT16 layers compute `y = b + W x` in float, with the weights converted to
  float as they are read.
- INT8, INT4 and INT2 layers quantize their input `x` (`in` floats) to bytes first:
  `s = max |x| / 127` and `xq = round(x / s)` to nearest even (all `xq` 0 when `x` is all 0), then
  `y[j] = b[j] + (scale[j] * s) * sum_k q[j][k] * xq[k]` with the sum in 32-bit integers.

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
