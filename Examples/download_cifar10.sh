#!/bin/sh
# SPDX-License-Identifier: MIT
# Downloads the CIFAR-10 binary batches into the given directory (default: data/cifar10): they end
# up in <dir>/cifar-10-batches-bin (data_batch_1.bin ... data_batch_5.bin, test_batch.bin).
set -eu
dir="${1:-data/cifar10}"
url="https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"
mkdir -p "$dir"
if [ -f "$dir/cifar-10-batches-bin/test_batch.bin" ]; then
    echo "CIFAR-10 is in $dir/cifar-10-batches-bin"
    exit 0
fi
echo "downloading CIFAR-10 (162 MB)"
if command -v curl >/dev/null 2>&1; then
    curl -fsSL "$url" -o "$dir/cifar-10-binary.tar.gz"
else
    wget -q "$url" -O "$dir/cifar-10-binary.tar.gz"
fi
tar -xzf "$dir/cifar-10-binary.tar.gz" -C "$dir"
rm "$dir/cifar-10-binary.tar.gz"
echo "CIFAR-10 is in $dir/cifar-10-batches-bin"
