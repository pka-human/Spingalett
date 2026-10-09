# SPDX-License-Identifier: MIT
# Copyright (c) 2026 pka_human (pka_human@proton.me)
#
# cmake -DIN=<unit.ptx> -DOUT=<unit.ptx.inc> -P PtxToC.cmake: PTX text as a C string literal, a line a
# piece, for the embedded table of cmake/Cuda.cmake.
file(READ ${IN} text)
string(REPLACE "\\" "\\\\" text "${text}")
string(REPLACE "\"" "\\\"" text "${text}")
string(REPLACE "\t" "\\t" text "${text}")
string(REGEX REPLACE "\n" "\\\\n\"\n\"" text "${text}")
file(WRITE ${OUT} "\"${text}\"\n")
