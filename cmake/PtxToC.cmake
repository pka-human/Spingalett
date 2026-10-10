# SPDX-License-Identifier: MIT
# Copyright (c) 2026 pka_human (pka_human@proton.me)
#
# cmake -DIN=<unit.ptx> -DOUT=<unit.ptx.inc> -P PtxToC.cmake: PTX compressed with gzip, as the bytes of a C
# array initializer, for the embedded table of cmake/Cuda.cmake (Spingalett.Gunzip.c expands it). The
# header's time, extra flags and system are written as 0, 0 and 255, so that builds repeat.
file(ARCHIVE_CREATE OUTPUT ${OUT}.gz PATHS ${IN} FORMAT raw COMPRESSION GZip COMPRESSION_LEVEL 9)
file(READ ${OUT}.gz hex HEX)
file(REMOVE ${OUT}.gz)
string(SUBSTRING "${hex}" 0 8 head)
string(SUBSTRING "${hex}" 20 -1 body)
string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," bytes "${head}00000000" "00ff${body}")
string(REGEX REPLACE "((0x..,){24})" "\\1\n" bytes "${bytes}")
file(WRITE ${OUT} "${bytes}\n")
