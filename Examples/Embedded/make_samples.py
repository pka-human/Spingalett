#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Writes the first N MNIST test digits as a C header (mnist_samples.h) for the embedded example.

    make_samples.py <mnist-dir> [N] > mnist_samples.h
"""
import struct
import sys


def main():
    directory = sys.argv[1]
    count = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    with open(f"{directory}/t10k-images-idx3-ubyte", "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 0x803:
            sys.exit("not an IDX image file")
        count = min(count, n)
        pixels = f.read(count * rows * cols)
    with open(f"{directory}/t10k-labels-idx1-ubyte", "rb") as f:
        f.read(8)
        labels = f.read(count)
    size = rows * cols
    out = [f"/* The first {count} MNIST test digits, {rows}x{cols} bytes each. Written by make_samples.py. */",
           f"#define SAMPLE_COUNT {count}u",
           f"static const uint8_t sample_labels[{count}] = {{{', '.join(str(b) for b in labels)}}};",
           f"static const uint8_t sample_pixels[{count}][{size}] = {{"]
    for i in range(count):
        row = pixels[i * size:(i + 1) * size]
        out.append("    {" + ",".join(str(b) for b in row) + "},")
    out.append("};")
    print("\n".join(out))


if __name__ == "__main__":
    main()
