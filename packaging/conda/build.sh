#!/bin/bash
# SPDX-License-Identifier: MIT
set -euxo pipefail
cmake -S . -B build -G Ninja ${CMAKE_ARGS} -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$PREFIX" \
    -DCMAKE_INSTALL_LIBDIR=lib -DSPINGALETT_NATIVE_ARCH=OFF -DBUILD_WITH_OPENMP=ON -DSPINGALETT_VULKAN=ON \
    -DBUILD_APPS=OFF -DBUILD_TESTS=OFF -DSPINGALETT_BIN_DIR="$SRC_DIR/build/Bin" -DSPINGALETT_LIB_DIR="$SRC_DIR/build/Lib"
cmake --build build --parallel "${CPU_COUNT}"
cmake --install build
