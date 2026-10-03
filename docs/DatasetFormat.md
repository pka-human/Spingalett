# The .slettd data set format, version 1

`.slettd` files store a data set (inputs and targets, one row per sample) compactly and load
straight into `SpingalettDataset`. They are written by `spingalett_save_dataset()` and read by
`spingalett_load_dataset()`, `spingalett_load_dataset_from_memory()` and the streaming reader
(`spingalett_dataset_open()`). This document describes the layout precisely enough to write an
independent reader.

All integers are little-endian. "float" means an IEEE 754 binary32 stored as its bit pattern.

## Overview

```
header (64 bytes) | parameters | chunk index | chunk 0 | chunk 1 | ...
```

The samples are split into chunks of `chunk_samples` consecutive samples (the last one may be
shorter). Every chunk decodes on its own, so readers can stream a file chunk by chunk, visit chunks
in any order and decode them in parallel. Writers aim at about 1 MiB of encoded values per chunk.

## Header

| Offset | Size | Field |
|---:|---:|---|
| 0 | 6 | magic `SLETTD` |
| 6 | 2 | format version, 1 |
| 8 | 4 | `count`: number of samples |
| 12 | 4 | `input_size`: inputs per sample |
| 16 | 4 | `target_size`: targets per sample, as loaded (for `CLASS`, the number of classes) |
| 20 | 1 | input encoding |
| 21 | 1 | target encoding |
| 22 | 1 | 1 if the writer tried compression, 0 if it stored everything (informational) |
| 23 | 1 | reserved, 0 |
| 24 | 4 | `chunk_samples` (at least 1) |
| 28 | 4 | `chunk_count` = ceil(`count` / `chunk_samples`) |
| 32 | 4 | input stride (coder context, at least 1) |
| 36 | 4 | target stride |
| 40 | 8 | file size in bytes |
| 48 | 4 | size of the parameters block in bytes |
| 52 | 8 | reserved, 0 |
| 60 | 4 | CRC-32 of bytes 0 to 59 |

CRC-32 is the IEEE polynomial (0xEDB88320, reflected, initial value and final XOR 0xFFFFFFFF), as
in zlib and PNG.

## Encodings

A stream holds one value per feature (inputs, or targets) per sample, except `CLASS`, which holds
one value per sample.

| Code | Encoding | Bytes per value | Value |
|---:|---|---:|---|
| 1 | `FLOAT32` | 4 | the float itself |
| 2 | `FP16` | 2 | IEEE binary16 (written with round to nearest even) |
| 3 | `BFLOAT16` | 2 | the upper 16 bits of a float (written with round to nearest even) |
| 4 | `U8_UNIT` | 1 | q / 255 for q in 0..255, rounded to float (`(float)(q / 255.0)`) |
| 5 | `U8_AFFINE` | 1 | `min[f] + q * step[f]`, per feature f, computed in float |
| 6 | `CLASS` | 1 if `target_size` <= 256, else 2 | class index c < `target_size`; the row loads as one-hot (1 at c, 0 elsewhere). Targets only |

Writers choose, by default, the smallest encoding that reproduces every value exactly:
`CLASS` for targets whose rows are all one-hot, then `U8_UNIT`, `FP16` and `FLOAT32`. `FP16`,
`BFLOAT16` and `U8_AFFINE` are also available as lossy encodings (`U8_AFFINE` uses the minimum of
each feature and a step of (max - min) / 255).

## Parameters

For each stream (inputs first, then targets) whose encoding is `U8_AFFINE`, two floats per feature:
`min[f], step[f]` for f = 0 .. size - 1. The block is empty otherwise.

## Chunk index

`chunk_count` entries of 20 bytes, followed by a CRC-32 of the parameters block and all entries:

| Offset | Size | Field |
|---:|---:|---|
| 0 | 8 | absolute file offset of the chunk |
| 8 | 4 | size of the chunk's input stream in bytes |
| 12 | 4 | size of the chunk's target stream in bytes |
| 16 | 4 | CRC-32 of the chunk (both streams) |

A chunk is its input stream immediately followed by its target stream.

## Streams

The first byte of a stream is its method: 0 = stored, 1 = coded. The rest is the stream's values
in byte planes: for the n values of the chunk (sample-major, feature-minor), plane k holds byte k
(least significant first) of each value, and planes follow one another. A stored stream is exactly
these `n * bytes_per_value` bytes. A coded stream holds them compressed as described below.
Byte planes keep slowly varying bytes, such as float exponents, together.

## The coder

Coded streams use an adaptive binary range coder (the one LZMA uses) with 11-bit probabilities:

- probabilities start at 1024 (of 2048) and adapt after every bit: `p += (2048 - p) >> 4` after a
  0, `p -= p >> 4` after a 1;
- encoding a bit with probability p of 0: `bound = (range >> 11) * p`; a 0 sets `range = bound`, a
  1 adds `bound` to `low` and subtracts it from `range`; while `range < 2^24`, shift `range` left by
  8 and emit the top byte of `low` with LZMA's carry handling;
- the encoder starts with `low = 0`, `range = 0xFFFFFFFF`, one pending cache byte 0, and ends with
  five shifts. The decoder reads five bytes into `code` (the first is always 0) and renormalizes
  with one byte whenever `range < 2^24`.

Planes are coded one after another with one coder, and the model is reset at the start of every
plane. For byte i of a plane (b), with bytes before the start of the plane taken as 0 and
`s` the stream's stride:

```
p1 = b[i - 1], ps = b[i - s], ps1 = b[i - s - 1]
zero context = (p1 >> 4) << 7 | (ps >> 4) << 3 | ps1 >> 5           (2048 contexts)
tree context = (p1 >> 4) << 4 | ps >> 4                               (256 contexts)
```

The coder first codes whether the byte is non-zero (bit 1) with the zero context's probability.
For a non-zero byte it then codes the 8 bits, most significant first, down a binary tree of 255
probabilities in the tree context: node 1 for the first bit, then `node = node * 2 + bit`.

The stride lets the context see the value one "row" back. Writers pick it per stream by estimating
the coded size of the first chunk for a few candidates (1 to 4, and every divisor d of the values
per sample, with d - 1 and d + 1); for 28 x 28 images this finds the row length, so the pixel
above is part of the context.

## Validation

Readers must check the magic, version, both metadata checksums and every chunk's checksum, that
every chunk lies inside the file, that stored streams have exactly the expected size, and that
class indices are below `target_size`. A coded stream cannot expand more than about 750 times
(adapted probabilities stay between 15/2048 and 2033/2048, so every coded bit costs at least
0.0106 bits), so the reference reader also rejects chunks claiming
more than 4096 decoded bytes per stored byte, which keeps crafted files from requesting huge
buffers.

## Size on MNIST

The 60,000 training images and labels: 47.1 MB as IDX files, 190.6 MB as float32, 9.7 MB for the
images alone with `gzip -9`, 7.9 MB with `xz -9`, and 7.8 MB as `.slettd`, which loads back bit for
bit identical to `spingalett_load_idx()`.
