# SPDX-License-Identifier: MIT
# Copyright (c) 2026 pka_human (pka_human@proton.me)
#
# Spingalett in a container: the library with both GPU backends (CUDA through the NVIDIA container toolkit's
# driver, Vulkan through a mounted ICD), its tools, the headers for building against it, and the Python
# bindings. Built from the repository's root:
#
#   docker build -f packaging/docker/Dockerfile -t spingalett .
#   docker run --rm --gpus all spingalett python3 -c "import spingalett as sg; print(sg.cuda_device())"

FROM ubuntu:24.04 AS build
RUN apt-get update && apt-get install -y --no-install-recommends \
        cmake ninja-build gcc g++ libc6-dev clang glslc libvulkan-dev python3 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY . .
RUN cmake -S . -B /build -G Ninja -DCMAKE_BUILD_TYPE=Release -DSPINGALETT_NATIVE_ARCH=OFF -DBUILD_WITH_OPENMP=ON \
        -DSPINGALETT_VULKAN=ON -DSPINGALETT_CUDA=ON -DBUILD_APPS=OFF -DBUILD_TESTS=OFF \
        -DCMAKE_INSTALL_PREFIX=/opt/spingalett -DSPINGALETT_BIN_DIR=/build/Bin -DSPINGALETT_LIB_DIR=/build/Lib \
    && cmake --build /build \
    && cmake --install /build

FROM ubuntu:24.04
LABEL org.opencontainers.image.source="https://github.com/pka-human/Spingalett" \
      org.opencontainers.image.description="Spingalett: neural networks in C (CPU, CUDA, Vulkan), its tools, headers and Python bindings" \
      org.opencontainers.image.licenses="MIT"
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 libvulkan1 python3 python3-numpy \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /opt/spingalett /usr/local
COPY Bindings/Python/spingalett /usr/local/lib/python3/dist-packages/spingalett
RUN ldconfig && echo /usr/local/lib/python3/dist-packages > /usr/lib/python3/dist-packages/spingalett.pth
ENV SPINGALETT_LIBRARY=/usr/local/lib/libspingalett.so \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
CMD ["python3", "-c", "import spingalett as sg; print('Spingalett', sg.library_version())"]
