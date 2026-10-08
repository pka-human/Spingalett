#!/bin/sh
# SPDX-License-Identifier: MIT
# Prints the GitHub release notes for a version: its CHANGELOG.md section, then the downloads.
#   .github/release-notes.sh 0.4.0 [with-binaries: 1|0]
set -eu
version=$1
binaries=${2:-1}

awk -v v="$version" '
    $0 ~ "^## \\[" v "\\]" { on = 1; next }
    on && /^## \[/ { exit }
    on && /^\[[^]]+\]: / { exit }
    on { print }
' CHANGELOG.md | sed -e '/./,$!d'          # drop leading blank lines

[ "$binaries" = 1 ] || exit 0
cat <<EOF

## Downloads

| File | For |
|---|---|
| \`spingalett-$version-linux-x86_64.tar.gz\` | any x86-64 CPU (AVX2 and AVX-512 matrix kernels chosen at run time); glibc 2.29 or newer and \`libgomp1\` |
| \`spingalett-$version-linux-x86_64-v3.tar.gz\` | x86-64 CPUs with AVX2 and FMA (Intel Haswell, AMD Zen and newer): the rest of the library compiled for AVX2 too |
| \`spingalett-$version-linux-aarch64.tar.gz\` | 64-bit ARM (Raspberry Pi 4 and 5 with a 64-bit OS, AWS Graviton, Ampere); glibc 2.29+ and \`libgomp1\` |
| \`spingalett-$version-windows-x86_64.zip\` | 64-bit Windows (AVX2 and AVX-512 matrix kernels chosen at run time); the MinGW and OpenMP runtime DLLs are included |
| \`spingalett-$version-windows-x86_64-v3.zip\` | the same for CPUs with AVX2 and FMA |
| \`spingalett-$version-macos-universal.tar.gz\` | macOS 11 or newer, Apple silicon and Intel, with LLVM's OpenMP runtime next to the library |
| \`spingalett-$version-py3-none-*.whl\` | the Python package with the library inside, for any Python 3: manylinux 2.28 x86-64 and AArch64, Windows x86-64, macOS universal; \`pip install spingalett\` installs the same from PyPI |
| \`spingalett-$version-inference-engine.zip\` | the standalone inference engine for firmware: three C files, see its README |
| \`DigitPad-$version-x86_64.AppImage\` | the digit-drawing demo with a trained model, for x86-64 Linux with glibc 2.34+ |
| \`DigitPad-$version-windows-x86_64.zip\` | the same demo for 64-bit Windows 10 and 11: unpack it and start \`DigitPad.exe\` |
| \`SHA256SUMS\` | checksums of all files |

Every archive holds \`include/\`, \`lib/\` (with a CMake package: \`find_package(Spingalett $(echo "$version" | cut -d. -f1-2))\`
with \`CMAKE_PREFIX_PATH\` pointing at the extracted directory), \`bin/DatasetTool\`,
\`bin/ModelTool\` (quantize, evaluate, import ONNX models and export models as C headers), LICENSE and CHANGELOG. The Windows archives contain \`bin/libspingalett.dll\` with import libraries for MinGW
(\`lib/libspingalett.dll.a\`) and MSVC (\`lib/spingalett.lib\`). Python users install a wheel
(\`pip install spingalett\`, or one of the files above); the bindings of a source checkout find the
library through \`SPINGALETT_LIBRARY\`. DigitPad is not signed: on Windows, SmartScreen may ask for
**More info**, then **Run anyway**.
EOF
