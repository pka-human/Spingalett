# Installing and building

Prebuilt packages, building from source, the CMake options, and using the library from a project.

Prebuilt libraries for Linux (x86-64 and ARM64), Windows and macOS are attached to every
[release](https://github.com/pka-human/Spingalett/releases): each archive contains the headers,
the shared library, a CMake package, a pkg-config file, `DatasetTool` and `ModelTool`, with the
runtime library next to them; the `spingalett-runtime-*` archives hold the [runtime](Deployment.md#the-runtime)
alone, for programs that only run models. Extract one
and point CMake at it with `-DCMAKE_PREFIX_PATH=<directory>`, or compile directly with
`-I<dir>/include -L<dir>/lib -lspingalett`. The x86-64 archives come in a baseline build that runs on any x86-64 CPU and a
`-v3` build for processors with AVX2 and FMA; both pick AVX2 or AVX-512 matrix kernels at run time
when the processor has them, and the `-v3` build also compiles the rest of the library
(activations, optimizers, the inference engine) for AVX2. The Windows DLL ships with import
libraries for MinGW and MSVC. Every package is built with OpenMP and the Vulkan GPU backend, those for
Linux and Windows with the CUDA backend too; the macOS one (a universal library for Apple silicon and Intel, macOS 11 or newer) carries LLVM's
OpenMP runtime, `libomp.dylib`, next to the library, since Apple's compilers come without one.

To build from source: requirements: CMake 3.21 or newer and a compiler with C23 support. GCC 13 and Clang 18 are tested
in CI; MSVC 19.36 or newer is expected to work but is not tested. OpenMP and OpenBLAS are
optional, and so are the GPU backends: the Vulkan one needs the Vulkan headers and `glslc` (shaderc) to
build, the CUDA one a Clang with the NVPTX target, which compiles its kernels to PTX (no CUDA toolkit:
the driver compiles the PTX on first use); neither needs the GPU's runtime to build, since the library
opens it when a GPU mode is first used.

```bash
# Dependency-free build
cmake -S . -B Build -DCMAKE_BUILD_TYPE=Release
cmake --build Build --parallel

# With OpenMP and OpenBLAS (Debian/Ubuntu: apt install libopenblas-dev)
cmake -S . -B Build -DCMAKE_BUILD_TYPE=Release -DBUILD_WITH_OPENMP=ON -DBUILD_WITH_OPENBLAS=ON
cmake --build Build --parallel

# The GPU backends are built when their compilers are found: Vulkan with glslc and the Vulkan headers
# (Debian/Ubuntu: apt install glslc libvulkan-dev; macOS: brew install shaderc vulkan-headers),
# CUDA with a Clang that targets NVPTX (Debian/Ubuntu: apt install clang)

# Run the test suite
ctest --test-dir Build --output-on-failure

# Install headers and the shared library
cmake --install Build --prefix /usr/local
```

The shared library and the example programs are written to `Bin/`, import libraries to `Lib/`.
By default the library is compiled with `-march=native` and therefore tuned for the build
machine; for redistributable binaries, configure with `-DSPINGALETT_NATIVE_ARCH=OFF` and choose
a baseline through `CMAKE_C_FLAGS` (for example `-march=x86-64-v3` for AVX2). On x86-64 with GCC
or Clang, such a build also compiles the matrix kernels for AVX2 and AVX-512 and runs the best set
the processor supports (`spingalett_cpu_kernels()` names it).

| CMake option | Default | Description |
|---|---|---|
| `BUILD_WITH_OPENMP` | `OFF` | Enable the OpenMP backend |
| `BUILD_WITH_OPENBLAS` | `OFF` | Enable the OpenBLAS backend (found via pkg-config or the default library paths) |
| `BUILD_EXAMPLE` | `ON` | Build the programs in `Examples/` |
| `BUILD_TESTS` | `ON` | Build the test suite and register it with CTest |
| `BUILD_APPS` | `OFF` | Build the DigitPad demo (needs SDL2) and its trainer |
| `SPINGALETT_RUNTIME` | `ON` | Build the [runtime](Deployment.md#the-runtime) library next to the library |
| `SPINGALETT_RUNTIME_ONLY` | `OFF` | Build only the runtime library (`Spingalett.Runtime.h`): deployment models on the CPU, without training, importers or the GPU backend (no `glslc` needed) |
| `SPINGALETT_INFERENCE_ONLY` | `OFF` | Build only the inference engine (`Spingalett.Inference.h`) as a static library: no training, file I/O, OpenMP or heap |
| `SPINGALETT_NATIVE_ARCH` | `ON` | Compile with `-march=native`; turn off for binaries that must run on other machines |
| `SPINGALETT_CPU_DISPATCH` | `ON` | Without `-march=native` (GCC, Clang): on x86-64 also build AVX2 and AVX-512 matrix kernels and AVX-512 VNNI and AVX-VNNI integer kernels, on AArch64 Linux integer kernels for the dot product instructions, and choose at run time |
| `SPINGALETT_VULKAN` | `AUTO` | The Vulkan GPU backend: `AUTO` builds it when `glslc` and the Vulkan headers are found, `ON` requires them, `OFF` leaves it out |
| `SPINGALETT_SPIRV_DIR` | | Compiled shaders (`<kernel>.spv.inc` from another build's `Gpu/` directory) to use where there is no `glslc` |
| `SPINGALETT_CUDA` | `AUTO` | The CUDA GPU backend: `AUTO` builds it when a Clang with the NVPTX target is found (`SPINGALETT_CUDA_CLANG` names one), `ON` requires it, `OFF` leaves it out |
| `SPINGALETT_PTX_DIR` | | Compiled CUDA kernels (`<unit>.ptx.inc` from another build's `Cuda/` directory) to use where there is no such Clang |
| `SPINGALETT_BIN_DIR` | `<source>/Bin` | Output directory for executables and shared libraries |
| `SPINGALETT_LIB_DIR` | `<source>/Lib` | Output directory for static and import libraries |

An installed Spingalett is found with CMake's `find_package`; projects that vendor the
repository can use `add_subdirectory()` instead. Both provide the target `Spingalett::spingalett`,
and `Spingalett::runtime` for the runtime, which carry the include paths:

```cmake
find_package(Spingalett 1.4 REQUIRED)         # or: add_subdirectory(external/Spingalett)
target_link_libraries(my_app PRIVATE Spingalett::spingalett)    # or Spingalett::runtime
```

pkg-config finds it as `spingalett` (`cc app.c $(pkg-config --cflags --libs spingalett)`), and the
runtime as `spingalett-runtime`; the file
locates the installation from its own directory, so an extracted release archive serves too, through
`PKG_CONFIG_PATH=<dir>/lib/pkgconfig`.

Released versions also come as packages:

```bash
pip install spingalett                    # Python: wheels with the library inside (Linux, Windows, macOS)
brew tap pka-human/spingalett https://github.com/pka-human/Spingalett && brew install spingalett
scoop bucket add spingalett https://github.com/pka-human/Spingalett && scoop install spingalett
nix build github:pka-human/Spingalett     # the flake's package; nix develop: a shell to build it in
sudo apt install ./spingalett_1.6.1_amd64.deb        # Debian and Ubuntu; .rpm for Fedora and openSUSE
docker run --rm --gpus all ghcr.io/pka-human/spingalett:1.6.1     # the library, tools and Python, CUDA and Vulkan
```

The `.deb` and `.rpm` packages are assets of each [release](https://github.com/pka-human/Spingalett/releases),
as are the archives for Linux (x86-64, x86-64-v3, AArch64), Windows (x86-64, x86-64-v3) and macOS (universal).
`packaging/aur` holds PKGBUILDs for Arch Linux (`spingalett`, `spingalett-git`; `makepkg -si`), and
`packaging/conda` a recipe for conda-forge.

The repository carries a [Conan](https://conan.io) recipe and a [vcpkg](https://vcpkg.io) port that
build the library of the checkout (shared, with OpenMP, kernels chosen at run time, the runtime
next to it; the GPU backend as an option, with `glslc` and the Vulkan headers from the package
manager). Both give `find_package(Spingalett)` and pkg-config's `spingalett` and
`spingalett-runtime`; they build with GCC or Clang (MinGW on Windows):

```bash
conan create packaging/conan --build=missing                  # -o "&:vulkan=True": the GPU backend
                                                              # -o "&:runtime_only=True": the runtime alone
vcpkg install spingalett --overlay-ports=packaging/vcpkg      # "spingalett[vulkan]": the GPU backend
```

Programs that compile the library's sources into themselves, or link it statically, define
`SPINGALETT_STATIC` so that the headers do not declare the API as imported from a DLL.

The headers define `SPINGALETT_VERSION_MAJOR`, `_MINOR`, `_PATCH` and `_STRING`;
`spingalett_version()` returns the version of the library actually loaded.
