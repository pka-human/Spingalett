#!/bin/sh
# SPDX-License-Identifier: MIT
# Downloads the MNIST IDX files into the given directory (default: data/mnist).
set -eu
dir="${1:-data/mnist}"
base="https://ossci-datasets.s3.amazonaws.com/mnist"
mkdir -p "$dir"
for name in train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte; do
    if [ -f "$dir/$name" ]; then continue; fi
    echo "downloading $name"
    if command -v curl >/dev/null 2>&1; then
        curl -fsSL "$base/$name.gz" -o "$dir/$name.gz"
    else
        wget -q "$base/$name.gz" -O "$dir/$name.gz"
    fi
    gunzip -f "$dir/$name.gz"
done
echo "MNIST is in $dir"
