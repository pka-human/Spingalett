#!/bin/sh
# SPDX-License-Identifier: MIT
#
# Builds the embedded example for a Cortex-M4F (QEMU's MPS2 AN386 board) with the standalone
# inference engine, prints its flash and RAM use, and runs it in QEMU.
#
#   Examples/Embedded/run-qemu.sh <model.slett> <mnist-dir> [options]
#
#   --tool PATH        ModelTool (default: Bin/ModelTool)
#   --precision P      precision of the exported model (default int8)
#   --samples N        MNIST test digits compiled in (default 100)
#   --work DIR         build directory (default build/embedded)
#
# Needs arm-none-eabi-gcc with newlib (Debian/Ubuntu: gcc-arm-none-eabi libnewlib-arm-none-eabi),
# qemu-system-arm and python3. The program exits with 0 when at least 95% of the digits are right.
set -eu

here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
tool="$root/Bin/ModelTool"
precision=int8
samples=100
work="$root/build/embedded"
[ $# -ge 2 ] || { sed -n '3,15p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }
model=$1
mnist=$2
shift 2
while [ $# -gt 0 ]; do
    case "$1" in
        --tool) tool=$2; shift 2 ;;
        --precision) precision=$2; shift 2 ;;
        --samples) samples=$2; shift 2 ;;
        --work) work=$2; shift 2 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done
mkdir -p "$work"

"$tool" header "$model" "$work/mnist_model.h" mnist_model --precision "$precision"
python3 "$here/make_samples.py" "$mnist" "$samples" > "$work/mnist_samples.h"

cpu="-mcpu=cortex-m4 -mthumb -mfloat-abi=hard -mfpu=fpv4-sp-d16"
cflags="$cpu -O2 -std=c11 -Wall -Wextra -ffunction-sections -fdata-sections -DSPINGALETT_INFERENCE_ONLY -I$root/Include -I$work"
# the engine alone, for its code size
arm-none-eabi-gcc $cflags -c "$root/Src/Spingalett.Inference.c" -o "$work/Spingalett.Inference.o"
arm-none-eabi-gcc $cflags -specs=nano.specs -specs=rdimon.specs -nostartfiles -T "$here/mps2-an386.ld" \
    -Wl,--gc-sections -u _printf_float \
    "$here/mnist_mcu.c" "$here/startup.c" "$work/Spingalett.Inference.o" -lm -o "$work/mnist_mcu.elf"

echo "inference engine (Spingalett.Inference.c, -O2):"
arm-none-eabi-size "$work/Spingalett.Inference.o"
echo "whole program (text = flash, bss = RAM):"
arm-none-eabi-size "$work/mnist_mcu.elf"

timeout 600 qemu-system-arm -M mps2-an386 -cpu cortex-m4 -nographic -monitor none -serial none \
    -semihosting-config enable=on,target=native -kernel "$work/mnist_mcu.elf"
