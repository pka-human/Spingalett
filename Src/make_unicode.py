# SPDX-License-Identifier: MIT
# Copyright (c) 2026 pka_human (pka_human@proton.me)
"""Writes Src/Spingalett.Unicode.h, the Unicode data of the tokenizers (Spingalett.Tokenizer.c), from
Python's unicodedata:

    python Src/make_unicode.py

- the general categories the pre-tokenizers' patterns test (letters by kind, marks, numbers), as the
  first code point of every run of one category;
- canonical normalization (NFC): the combining classes, the canonical decompositions (one step each)
  and the primary composites (the decompositions of two code points that NFC composes again, the
  composition exclusions left out). Hangul syllables are computed, not listed.
"""
import os
import sys
import unicodedata

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Spingalett.Unicode.h")
CATEGORIES = {"Lu": 1, "Ll": 2, "Lt": 3, "Lm": 4, "Lo": 5, "Mn": 6, "Mc": 6, "Me": 6, "Nd": 7, "Nl": 7, "No": 7}
HANGUL = range(0xAC00, 0xAC00 + 11172)


def table(name, ctype, values, per_line):
    lines = [f"static const {ctype} {name}[{len(values)}] = {{"]
    for k in range(0, len(values), per_line):
        lines.append("    " + ", ".join(values[k:k + per_line]) + ",")
    lines.append("};")
    return "\n".join(lines)


def main():
    runs, previous = [], None
    for cp in range(0x110000):
        c = CATEGORIES.get(unicodedata.category(chr(cp)), 0)
        if c != previous:
            runs.append(f"0x{cp << 3 | c:X}")
            previous = c
    classes = [f"0x{cp << 8 | unicodedata.combining(chr(cp)):X}" for cp in range(0x110000)
               if unicodedata.combining(chr(cp))]
    decompositions, composites = [], []
    for cp in range(0x110000):
        if cp in HANGUL:
            continue
        d = unicodedata.decomposition(chr(cp))
        if not d or d.startswith("<"):
            continue
        parts = [int(x, 16) for x in d.split()]
        a, b = parts[0], parts[1] if len(parts) > 1 else 0
        decompositions.append(f"{{0x{cp:X}, 0x{a:X}, 0x{b:X}}}")
        if b and unicodedata.normalize("NFC", chr(a) + chr(b)) == chr(cp):
            composites.append((a, b, cp))
    composites.sort()
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"""/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/* Unicode {unicodedata.unidata_version} data of the tokenizers, written by Src/make_unicode.py: do not edit. */

#pragma once

#include <stdint.h>

/* The general category from each listed code point to the next: (code point << 3) | kind, kinds 0 for
   none of these, 1 Lu, 2 Ll, 3 Lt, 4 Lm, 5 Lo, 6 a mark (M), 7 a number (N). */
{table("unicode_categories", "uint32_t", runs, 8)}

/* Code points of a combining class other than 0: (code point << 8) | class. */
{table("unicode_classes", "uint32_t", classes, 8)}

/* Canonical decompositions, one step: code point, first, second (0 for a singleton). */
{table("unicode_decompositions", "uint32_t", decompositions, 4).replace("uint32_t unicode_decompositions", "uint32_t unicode_decompositions").replace(f"[{len(decompositions)}]", f"[{len(decompositions)}][3]")}

/* Primary composites by their two code points: first, second, composite. */
{table("unicode_composites", "uint32_t", [f"{{0x{a:X}, 0x{b:X}, 0x{c:X}}}" for a, b, c in composites], 4).replace(f"[{len(composites)}]", f"[{len(composites)}][3]")}
""")
    print(f"{OUT}: {len(runs)} category runs, {len(classes)} combining classes, {len(decompositions)} decompositions, "
          f"{len(composites)} composites (Unicode {unicodedata.unidata_version})")


if __name__ == "__main__":
    sys.exit(main())
