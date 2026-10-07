#!/bin/sh
# SPDX-License-Identifier: MIT
#
# Builds DigitPad-<version>-windows-x86_64.zip: DigitPad.exe with a trained model, libspingalett,
# SDL2 and whatever runtime DLLs they load, in one folder that runs on 64-bit Windows 10 and 11.
#
#   Apps/DigitPad/Package/build-windows-zip.sh [--model FILE.slett] [--epochs N] [--work DIR]
#
#   --model FILE.slett  package this model (and FILE.slett.info, if present) instead of training one
#   --epochs N          epochs for the model trained when --model is not given (default 30)
#   --work DIR          build directory (default build/windows)
#
# Runs in an MSYS2 UCRT64 shell (pacman -S mingw-w64-ucrt-x86_64-{gcc,cmake,ninja,SDL2}), or on
# Linux as a cross build with MinGW-w64 (Debian/Ubuntu: apt-get install mingw-w64), which builds
# SDL2 from source. Without --model, MNIST is downloaded and DigitPadTrain trains a model first.
set -eu

SDL_VERSION=2.32.10
SDL_SHA256=5f5993c530f084535c65a6879e9b26ad441169b3e25d789d83287040a9ca5165

root=$(cd "$(dirname "$0")/../../.." && pwd)
work="$root/build/windows"
model=""
epochs=30
while [ $# -gt 0 ]; do
    case "$1" in
        --model) model=$(cd "$(dirname "$2")" && pwd)/$(basename "$2"); shift 2 ;;
        --epochs) epochs=$2; shift 2 ;;
        --work) work=$2; shift 2 ;;
        -h|--help) sed -n '3,15p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown option: $1 (see --help)" >&2; exit 1 ;;
    esac
done
mkdir -p "$work"
work=$(cd "$work" && pwd)
if [ -n "${MSYSTEM:-}" ]; then
    # native Windows CMake: hand it C:/... paths rather than MSYS ones
    root=$(cygpath -m "$root"); work=$(cygpath -m "$work")
    if [ -n "$model" ]; then model=$(cygpath -m "$model"); fi
fi
version=$(sed -n 's/^project(Spingalett VERSION \([0-9.]*\).*/\1/p' "$root/CMakeLists.txt")
jobs=$(nproc 2>/dev/null || echo 4)
generator=""
if command -v ninja > /dev/null 2>&1; then generator="-G Ninja"; fi
step() { printf '\n==> %s\n' "$*"; }

# ---- 1. toolchain and SDL2
if [ -n "${MSYSTEM:-}" ]; then
    # MSYS2: the compiler, SDL2 and runtime DLLs of the active environment
    [ "$MSYSTEM" != MSYS ] || { echo "run this in a UCRT64 (or CLANG64, MINGW64) shell" >&2; exit 1; }
    [ -d "$MINGW_PREFIX/lib/cmake/SDL2" ] ||
        { echo "SDL2 not found: pacman -S ${MINGW_PACKAGE_PREFIX}-SDL2" >&2; exit 1; }
    cross=false
    objdump=objdump
    dll_dirs=$(cygpath -m "$MINGW_PREFIX/bin")
else
    # Linux: cross build with MinGW-w64, prefering its POSIX thread model
    triple=x86_64-w64-mingw32
    cc=$(command -v $triple-gcc-posix || command -v $triple-gcc || true)
    [ -n "$cc" ] || { echo "MinGW-w64 not found (Debian/Ubuntu: apt-get install mingw-w64)" >&2; exit 1; }
    cross=true
    objdump=$triple-objdump
    toolchain="$work/mingw-w64.cmake"
    cat > "$toolchain" <<EOF
set(CMAKE_SYSTEM_NAME Windows)
set(CMAKE_SYSTEM_PROCESSOR x86_64)
set(CMAKE_C_COMPILER $cc)
set(CMAKE_CXX_COMPILER ${cc%gcc*}g++${cc##*gcc})
set(CMAKE_RC_COMPILER $triple-windres)
set(CMAKE_FIND_ROOT_PATH /usr/$triple)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
EOF
    sdl="$work/sdl2"
    if [ ! -f "$sdl/bin/SDL2.dll" ]; then
        step "building SDL2 $SDL_VERSION for Windows"
        tarball="$work/SDL2-$SDL_VERSION.tar.gz"
        [ -f "$tarball" ] || curl -fL -o "$tarball" \
            "https://github.com/libsdl-org/SDL/releases/download/release-$SDL_VERSION/SDL2-$SDL_VERSION.tar.gz"
        echo "$SDL_SHA256  $tarball" | sha256sum -c -
        tar -xzf "$tarball" -C "$work"
        cmake $generator -S "$work/SDL2-$SDL_VERSION" -B "$work/sdl2-build" -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_TOOLCHAIN_FILE="$toolchain" -DCMAKE_INSTALL_PREFIX="$sdl" \
            -DSDL_SHARED=ON -DSDL_STATIC=OFF -DSDL_TEST=OFF
        cmake --build "$work/sdl2-build" -j "$jobs"
        cmake --install "$work/sdl2-build"
    fi
    dll_dirs="$sdl/bin
$(dirname "$("$cc" -print-libgcc-file-name)")
/usr/$triple/lib
/usr/$triple/bin"
fi

# ---- 2. the model, trained by a native build
if [ -z "$model" ]; then
    model="$work/mnist.slett"
    if [ ! -f "$model" ]; then
        step "training the model ($epochs epochs)"
        sh "$root/Examples/download_mnist.sh" "$work/mnist"
        cmake $generator -S "$root" -B "$work/train-build" -DCMAKE_BUILD_TYPE=Release -DBUILD_APPS=ON \
            -DBUILD_WITH_OPENMP=ON -DBUILD_TESTS=OFF -DBUILD_EXAMPLE=OFF \
            -DSPINGALETT_BIN_DIR="$work/train-build/Bin" -DSPINGALETT_LIB_DIR="$work/train-build/Lib"
        cmake --build "$work/train-build" -j "$jobs" --target DigitPadTrain
        "$work/train-build/Bin/DigitPadTrain" "$work/mnist" "$model" "$epochs"
    fi
fi
[ -f "$model" ] || { echo "model not found: $model" >&2; exit 1; }

# ---- 3. the app: baseline x86-64, no OpenMP or BLAS, libgcc linked in statically
step "building DigitPad"
set -- -S "$root" -B "$work/build" -DCMAKE_BUILD_TYPE=Release -DBUILD_APPS=ON \
    -DSPINGALETT_NATIVE_ARCH=OFF -DBUILD_WITH_OPENMP=OFF -DBUILD_WITH_OPENBLAS=OFF \
    -DBUILD_TESTS=OFF -DBUILD_EXAMPLE=OFF \
    -DCMAKE_EXE_LINKER_FLAGS=-static-libgcc -DCMAKE_SHARED_LINKER_FLAGS=-static-libgcc \
    -DSPINGALETT_BIN_DIR="$work/build/Bin" -DSPINGALETT_LIB_DIR="$work/build/Lib"
if $cross; then set -- "$@" -DCMAKE_TOOLCHAIN_FILE="$toolchain" -DSDL2_DIR="$sdl/lib/cmake/SDL2"; fi
cmake $generator "$@"
cmake --build "$work/build" -j "$jobs" --target DigitPad
[ -f "$work/build/Bin/DigitPad.exe" ] || { echo "DigitPad.exe was not built" >&2; exit 1; }

# ---- 4. the folder: the program, the DLLs it loads that are not part of Windows, the model
name="DigitPad-$version-windows-x86_64"
dist="$work/$name"
step "assembling $name"
rm -rf "${dist:?}"
mkdir -p "$dist/licenses"
cp "$work/build/Bin/DigitPad.exe" "$dist/"
dll_dirs="$work/build/Bin
$dll_dirs"
find_dll() {
    printf '%s\n' "$dll_dirs" | while IFS= read -r dir; do
        found=$(find "$dir" -maxdepth 1 -type f -iname "$1" 2> /dev/null | head -n 1)
        if [ -n "$found" ]; then echo "$found"; break; fi
    done
}
pending=DigitPad.exe
while [ -n "$pending" ]; do
    set -- $pending
    pending=""
    for file in "$@"; do
        for dll in $("$objdump" -p "$dist/$file" | sed -n 's/^[[:space:]]*DLL Name: //p' | tr -d '\r'); do
            [ -z "$(find "$dist" -maxdepth 1 -iname "$dll")" ] || continue
            src=$(find_dll "$dll")
            [ -n "$src" ] || continue                     # a Windows DLL
            cp "$src" "$dist/"
            pending="$pending $(basename "$src")"
        done
    done
done
cp "$model" "$dist/mnist.slett"
if [ -f "$model.info" ]; then cp "$model.info" "$dist/mnist.slett.info"; fi

# licences: Spingalett, SDL2 and the MinGW-w64 runtime libraries that were copied
cp "$root/LICENSE" "$dist/LICENSE.txt"
if $cross; then
    cp "$work/SDL2-$SDL_VERSION/LICENSE.txt" "$dist/licenses/SDL2.txt"
    for f in "$dist"/*.dll; do
        case "$(basename "$f")" in
            libwinpthread*) cp /usr/share/doc/mingw-w64-common/copyright "$dist/licenses/mingw-w64.txt" ;;
            libgcc*|libgomp*|libatomic*|libssp*) cp /usr/share/doc/gcc-mingw-w64-base/copyright "$dist/licenses/gcc.txt" ;;
        esac
    done
else
    # the licence files of the packages that installed the DLLs
    for f in "$dist"/*.dll; do
        package=$(pacman -Qqo "$MINGW_PREFIX/bin/$(basename "$f")" 2> /dev/null || true)
        [ -n "$package" ] || continue
        pacman -Qlq "$package" | grep "/share/licenses/.*[^/]$" | while IFS= read -r file; do
            rel=${file#*/share/licenses/}
            mkdir -p "$dist/licenses/$(dirname "$rel")"
            cp "$file" "$dist/licenses/$rel"
        done
    done
fi

info="a Spingalett network trained on MNIST"
if [ -f "$dist/mnist.slett.info" ]; then info=$(head -n 1 "$dist/mnist.slett.info"); fi
dlls=$(cd "$dist" && for f in *.dll; do
    case "$f" in
        libspingalett*) what="Spingalett, the network library" ;;
        SDL2*) what="SDL2: the window, mouse and keyboard" ;;
        *) what="compiler runtime library" ;;
    esac
    printf '  %-21s %s\n' "$f" "$what"
done)
sed 's/$/\r/' > "$dist/README.txt" <<EOF
DigitPad $version for Windows
$(echo "DigitPad $version for Windows" | sed 's/./=/g')

Draw a digit with the mouse and watch a neural network recognise it as you draw. The network is
a multilayer perceptron trained on the MNIST handwritten digits with Spingalett, a neural-network
library in C: https://github.com/pka-human/Spingalett

Unpack the whole folder anywhere and start DigitPad.exe (64-bit Windows 10 or 11). The program is
not signed, so Windows may first say it protected your PC: choose "More info", then "Run anyway".

  Left mouse button             draw
  Right mouse button            erase
  C, Space, Backspace, Clear    clear the canvas
  Ctrl+Z, U, Undo               undo the last stroke
  Esc                           quit

From a command prompt:

  DigitPad.exe --verbose                     also print every prediction to the console
  DigitPad.exe --classify image.pgm          classify a binary PGM image and exit
  DigitPad.exe other-model.slett             use another model

The model (mnist.slett): $info

Files:
  DigitPad.exe          the program
  mnist.slett           the model; mnist.slett.info is the line shown in the window
$dlls
  LICENSE.txt           Spingalett's licence (MIT)
  licenses\\             the licences of SDL2 (zlib) and of the other DLLs
EOF

out="$work/$name.zip"
rm -f "$out"
(cd "$work" && cmake -E tar cf "$name.zip" --format=zip "$name")
echo
echo "DLLs packaged with DigitPad.exe:"
printf '%s\n' "$dlls"
echo "built $out"
