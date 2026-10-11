#!/bin/sh
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 pka_human (pka_human@proton.me)
#
# Builds spingalett.wasm next to this script: the inference engine (Src/Spingalett.Inference.c) with the
# package's exports (wasm/Spingalett.Wasm.c) and mathematical functions (wasm/Spingalett.Libm.c), by
# Clang for wasm32 and wasm-ld, without a C library (wasm/include holds what the engine takes from
# <math.h> and <string.h>). The engine is compiled as for the desktop: C99, its portable kernels (which
# the compiler vectorizes with SIMD128 where that keeps the results), no contraction.
#
#   sh build.sh                    CC and WASM_LD choose the tools: CC=clang-18 WASM_LD=wasm-ld-18
set -eu

here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
CC=${CC:-clang}
if [ -z "${WASM_LD:-}" ]; then
    # wasm-ld on the path, the one of the compiler's own installation, or a versioned name (Debian's)
    WASM_LD=$(command -v wasm-ld || true)
    [ -n "$WASM_LD" ] || WASM_LD=$("$CC" -print-prog-name=wasm-ld)
    if ! command -v "$WASM_LD" > /dev/null 2>&1; then
        major=$("$CC" -dumpversion | cut -d. -f1)
        WASM_LD=wasm-ld-$major
    fi
fi
command -v "$WASM_LD" > /dev/null 2>&1 || { echo "build.sh: wasm-ld not found (install lld, or set WASM_LD)" >&2; exit 1; }

out=$here/spingalett.wasm
objects=$(mktemp -d "${TMPDIR:-/tmp}/spingalett-wasm.XXXXXX")
trap 'rm -rf "$objects"' EXIT

flags="--target=wasm32 -std=c99 -O2 -ffreestanding -nostdlib -ffp-contract=off -msimd128 -mbulk-memory
       -mnontrapping-fptoint -Wall -Wextra -Wpedantic -Werror -DSPINGALETT_INFERENCE_ONLY
       -isystem $here/wasm/include -I$root/Include -I$root/Src"
# shellcheck disable=SC2086
"$CC" $flags -c "$root/Src/Spingalett.Inference.c" -o "$objects/engine.o"
# shellcheck disable=SC2086
"$CC" $flags -c "$here/wasm/Spingalett.Libm.c" -o "$objects/libm.o"
# shellcheck disable=SC2086
"$CC" $flags -c "$here/wasm/Spingalett.Wasm.c" -o "$objects/exports.o"

# No entry point and no imports: the module exports its memory and the functions marked in
# Spingalett.Wasm.c. The stack (256 KB) comes first, so that an overflow traps instead of writing over
# the data.
"$WASM_LD" --no-entry --strip-all --gc-sections --stack-first -z stack-size=262144 -O2 \
    -o "$out" "$objects/engine.o" "$objects/libm.o" "$objects/exports.o"
echo "built $out ($(wc -c < "$out" | tr -d ' ') bytes)"
