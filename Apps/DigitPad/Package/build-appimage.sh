#!/bin/sh
# SPDX-License-Identifier: MIT
#
# Builds DigitPad-<version>-<arch>.AppImage: the DigitPad app, a trained model, libspingalett
# and a static SDL2 whose X11/Wayland/OpenGL backends are loaded at run time, so the image
# needs nothing on the host beyond glibc and the desktop's own display libraries.
#
#   Apps/DigitPad/Package/build-appimage.sh [--model FILE.nn] [--epochs N] [--work DIR]
#
#   --model FILE.nn  package this model (and FILE.nn.info, if present) instead of training one
#   --epochs N       epochs for the model trained when --model is not given (default 60)
#   --work DIR       build directory (default build/appimage)
#
# Without --model, MNIST is downloaded and DigitPadTrain trains a model first (OpenMP build,
# a few minutes on a desktop CPU). Requires cmake, a C23 compiler, curl, and the development
# headers SDL2 builds its video backends from (Debian/Ubuntu: apt-get build-dep libsdl2).
set -eu

SDL_VERSION=2.32.10
SDL_SHA256=5f5993c530f084535c65a6879e9b26ad441169b3e25d789d83287040a9ca5165

root=$(cd "$(dirname "$0")/../../.." && pwd)
work="$root/build/appimage"
model=""
epochs=60
while [ $# -gt 0 ]; do
    case "$1" in
        --model) model=$(cd "$(dirname "$2")" && pwd)/$(basename "$2"); shift 2 ;;
        --epochs) epochs=$2; shift 2 ;;
        --work) work=$2; shift 2 ;;
        -h|--help) sed -n '3,16p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown option: $1 (see --help)" >&2; exit 1 ;;
    esac
done
version=$(sed -n 's/^project(Spingalett VERSION \([0-9.]*\).*/\1/p' "$root/CMakeLists.txt")
arch=$(uname -m)
jobs=$(nproc 2>/dev/null || echo 4)
mkdir -p "$work"
work=$(cd "$work" && pwd)
step() { printf '\n==> %s\n' "$*"; }

# ---- 1. static SDL2: video and events only; display libraries are dlopen()ed at run time.
# SDL's own strlcpy/fmod and the compat header keep the image off glibc 2.38-only symbols.
sdl="$work/sdl2"
if [ ! -f "$sdl/lib/libSDL2.a" ]; then
    step "building SDL2 $SDL_VERSION (static)"
    tarball="$work/SDL2-$SDL_VERSION.tar.gz"
    [ -f "$tarball" ] || curl -fL -o "$tarball" \
        "https://github.com/libsdl-org/SDL/releases/download/release-$SDL_VERSION/SDL2-$SDL_VERSION.tar.gz"
    echo "$SDL_SHA256  $tarball" | sha256sum -c -
    tar -xzf "$tarball" -C "$work"
    cmake -S "$work/SDL2-$SDL_VERSION" -B "$work/sdl2-build" -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_FLAGS="-include $root/Apps/DigitPad/Package/glibc-compat.h" \
        -DHAVE_STRLCPY=0 -DHAVE_STRLCAT=0 -DHAVE_WCSLCPY=0 -DHAVE_WCSLCAT=0 -DHAVE_FMOD=0 -DHAVE_FMODF=0 \
        -DCMAKE_INSTALL_PREFIX="$sdl" -DSDL_SHARED=OFF -DSDL_STATIC=ON -DSDL_STATIC_PIC=ON -DSDL_TEST=OFF \
        -DSDL_AUDIO=OFF -DSDL_HAPTIC=OFF -DSDL_JOYSTICK=OFF -DSDL_HIDAPI=OFF -DSDL_SENSOR=OFF \
        -DSDL_VULKAN=OFF -DSDL_KMSDRM=OFF -DSDL_LIBSAMPLERATE=OFF \
        -DSDL_X11_SHARED=ON -DSDL_WAYLAND_SHARED=ON -DSDL_WAYLAND_LIBDECOR_SHARED=ON
    cmake --build "$work/sdl2-build" -j "$jobs"
    cmake --install "$work/sdl2-build"
fi
config="$sdl/include/SDL2/SDL_config.h"
if ! grep -q "define SDL_VIDEO_DRIVER_X11 1" "$config"; then
    echo "SDL2 was built without X11 support: install the X11 development headers and remove $sdl" >&2
    exit 1
fi
grep -q "define SDL_VIDEO_DRIVER_WAYLAND 1" "$config" ||
    echo "warning: SDL2 was built without Wayland support; DigitPad will run through XWayland" >&2

# ---- 2. the model
if [ -z "$model" ]; then
    model="$work/mnist.nn"
    if [ ! -f "$model" ]; then
        step "training the model ($epochs epochs)"
        sh "$root/Examples/download_mnist.sh" "$work/mnist"
        cmake -S "$root" -B "$work/train-build" -DCMAKE_BUILD_TYPE=Release -DBUILD_APPS=ON \
            -DBUILD_WITH_OPENMP=ON -DBUILD_TESTS=OFF -DBUILD_EXAMPLE=OFF \
            -DSPINGALETT_BIN_DIR="$work/train-build/Bin" -DSPINGALETT_LIB_DIR="$work/train-build/Lib"
        cmake --build "$work/train-build" -j "$jobs" --target DigitPadTrain
        "$work/train-build/Bin/DigitPadTrain" "$work/mnist" "$model" "$epochs"
    fi
fi
[ -f "$model" ] || { echo "model not found: $model" >&2; exit 1; }

# ---- 3. portable app build: baseline ISA, no OpenMP/BLAS, static SDL2
step "building DigitPad"
cmake -S "$root" -B "$work/build" -DCMAKE_BUILD_TYPE=Release -DBUILD_APPS=ON \
    -DSPINGALETT_NATIVE_ARCH=OFF -DBUILD_WITH_OPENMP=OFF -DBUILD_WITH_OPENBLAS=OFF \
    -DBUILD_TESTS=OFF -DBUILD_EXAMPLE=OFF -DSDL2_DIR="$sdl/lib/cmake/SDL2" \
    -DCMAKE_INSTALL_PREFIX=/usr -DCMAKE_INSTALL_LIBDIR=lib \
    -DSPINGALETT_BIN_DIR="$work/build/Bin" -DSPINGALETT_LIB_DIR="$work/build/Lib"
cmake --build "$work/build" -j "$jobs" --target DigitPad

# ---- 4. AppDir
step "assembling the AppDir"
pkg="$root/Apps/DigitPad/Package"
appdir="$work/AppDir"
rm -rf "${appdir:?}"
DESTDIR="$appdir" cmake --install "$work/build" >/dev/null
rm -rf "$appdir/usr/include" "$appdir/usr/lib/cmake" "$appdir/usr/lib/libspingalett.so"
share="$appdir/usr/share"
mkdir -p "$share/digitpad" "$share/applications" "$share/icons/hicolor/scalable/apps" "$share/doc/digitpad"
cp "$model" "$share/digitpad/mnist.nn"
if [ -f "$model.info" ]; then cp "$model.info" "$share/digitpad/mnist.nn.info"; fi
cp "$pkg/DigitPad.desktop" "$share/applications/"
cp "$pkg/digitpad.svg" "$share/icons/hicolor/scalable/apps/"
cp "$root/LICENSE" "$share/doc/digitpad/LICENSE"
cp "$work/SDL2-$SDL_VERSION/LICENSE.txt" "$share/doc/digitpad/LICENSE.SDL2" 2>/dev/null || true
cp "$pkg/DigitPad.desktop" "$pkg/digitpad.svg" "$appdir/"
ln -s digitpad.svg "$appdir/.DirIcon"
cp "$pkg/AppRun" "$appdir/AppRun"
chmod 755 "$appdir/AppRun"

echo "shared libraries DigitPad needs from the host:"
readelf -d "$appdir/usr/bin/DigitPad" "$appdir/usr/lib/libspingalett.so.0" |
    sed -n 's/.*(NEEDED).*\[\(.*\)\]/  \1/p' | sort -u | grep -v libspingalett
glibc=$( (objdump -T "$appdir/usr/bin/DigitPad"; objdump -T "$appdir/usr/lib/libspingalett.so.0") |
    grep -o 'GLIBC_[0-9.]*' | sed 's/GLIBC_//' | sort -t. -k1,1n -k2,2n -k3,3n | tail -1)
echo "minimum glibc: $glibc"

# ---- 5. the image
tool=${APPIMAGETOOL:-$(command -v appimagetool || true)}
if [ -z "$tool" ]; then
    tool="$work/appimagetool-$arch.AppImage"
    if [ ! -x "$tool" ]; then
        step "downloading appimagetool"
        curl -fL -o "$tool" "https://github.com/AppImage/appimagetool/releases/download/continuous/appimagetool-$arch.AppImage"
        chmod +x "$tool"
    fi
fi
out="$work/DigitPad-$version-$arch.AppImage"
step "packing $out"
# APPIMAGE_EXTRACT_AND_RUN lets the tool (itself an AppImage) run where FUSE is unavailable
ARCH=$arch APPIMAGE_EXTRACT_AND_RUN=1 "$tool" --no-appstream "$appdir" "$out"
echo "built $out"
