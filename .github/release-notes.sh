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
| \`spingalett-$version-linux-x86_64.tar.gz\` | any x86-64 CPU; glibc 2.29 or newer and \`libgomp1\` |
| \`spingalett-$version-linux-x86_64-v3.tar.gz\` | x86-64 CPUs with AVX2 and FMA (Intel Haswell, AMD Zen and newer): faster kernels |
| \`spingalett-$version-linux-aarch64.tar.gz\` | 64-bit ARM (Raspberry Pi 4 and 5 with a 64-bit OS, AWS Graviton, Ampere); glibc 2.29+ and \`libgomp1\` |
| \`spingalett-$version-windows-x86_64.zip\` | 64-bit Windows; the MinGW and OpenMP runtime DLLs are included |
| \`spingalett-$version-windows-x86_64-v3.zip\` | the same for CPUs with AVX2 and FMA |
| \`spingalett-$version-macos-universal.tar.gz\` | macOS 11 or newer, Apple silicon and Intel (single-threaded: no OpenMP) |
| \`DigitPad-$version-x86_64.AppImage\` | the digit-drawing demo with a trained model, for x86-64 Linux with glibc 2.34+ |
| \`SHA256SUMS\` | checksums of all files |

Every archive holds \`include/\`, \`lib/\` (with a CMake package: \`find_package(Spingalett $(echo "$version" | cut -d. -f1-2))\`
with \`CMAKE_PREFIX_PATH\` pointing at the extracted directory), \`bin/DatasetTool\`, LICENSE and
CHANGELOG. The Windows archives contain \`bin/libspingalett.dll\` with import libraries for MinGW
(\`lib/libspingalett.dll.a\`) and MSVC (\`lib/spingalett.lib\`). Python users point
\`SPINGALETT_LIBRARY\` at the shared library and install the bindings from the source archive
(\`pip install ./Bindings/Python\`).
EOF
