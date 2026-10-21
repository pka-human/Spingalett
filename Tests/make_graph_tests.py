# SPDX-License-Identifier: MIT
# Copyright (c) 2026 pka_human (pka_human@proton.me)
"""Writes the cases of the `operators` test group: ONNX's conformance tests of single operators (the cases the onnx
package generates, onnx.backend.test.case.node, whose outputs come from ONNX's reference implementation), into
Tests/Data/graph_cases.bin:

    SPINGALETT_LIBRARY=Bin/libspingalett.so python Tests/make_graph_tests.py     # needs onnx and numpy

and lists in Tests/Data/graph_cases.txt the cases left out and why: operators graphs do not have (control flow,
sequences, strings, training, randomness), types they do not hold (float64 and float8, strings, 16-bit
integers), and cases the library fails. The file keeps the cases the library passes, so that the test group holds
it to them; the library's report of a case it fails is the list's line.

The file holds, little-endian:

    "SGCASES1", u32 cases
    per case: u16 name length, name; u32 model length, the ONNX model; f32 rtol, f32 atol;
              u32 inputs, tensors; u32 outputs, tensors (the reference's)
    per tensor: u16 name length, name; u8 type (SpingalettDType); u8 rank; i64 sizes[rank]; u64 bytes, values

Floats of 64 bits become float32 (graphs compute floats in single precision).
"""

import os
import struct
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "Bindings", "Python"))

import spingalett as sp                                     # noqa: E402
from onnx.backend.test.case import node                     # noqa: E402

TYPES = {np.dtype(np.float32): 1, np.dtype(np.uint8): 2, np.dtype(np.int8): 3, np.dtype(np.int32): 6,
         np.dtype(np.int64): 7, np.dtype(np.bool_): 9, np.dtype(np.float16): 10}
MOST_BYTES = 1 << 18            # a case's tensors at most: the big 3-D poolings are left out


def tolerances(case, ref):
    rtol, atol = max(getattr(case, "rtol", 1e-4), 1e-4), max(getattr(case, "atol", 1e-5), 1e-5)
    if any(np.asarray(r).dtype == np.float16 for r in ref):
        rtol, atol = 1e-2, 1e-2
    return rtol, atol


def as_tensor(value):
    arr = np.asarray(value)
    if arr.dtype == np.float64:
        arr = arr.astype(np.float32)
    if arr.dtype not in TYPES or arr.ndim > 8:
        return None
    return arr if arr.flags.c_contiguous else arr.copy(order="C")     # (a scalar stays a scalar)


def close(got, ref, rtol, atol):
    if got.shape != ref.shape:
        return False
    if ref.dtype.kind == "f":
        return np.allclose(got.astype(np.float64), ref.astype(np.float64), rtol=rtol, atol=atol, equal_nan=True)
    return np.array_equal(got.astype(ref.dtype), ref)


def pack_tensor(name, arr):
    data = arr.tobytes()
    encoded = name.encode()
    return (struct.pack("<H", len(encoded)) + encoded + struct.pack("<BB", TYPES[arr.dtype], arr.ndim) +
            b"".join(struct.pack("<q", d) for d in arr.shape) + struct.pack("<Q", len(data)) + data)


def main():
    kept, left = [], []
    for case in node.collect_testcases(None):
        model = case.model.SerializeToString()
        try:
            graph = sp.Graph.from_bytes(model)
        except sp.SpingalettError as e:
            left.append((case.name, str(e).split(" (code")[0]))
            continue
        records, ok, why = [], True, ""
        for inputs, outputs in case.data_sets:
            names = [i.name for i in case.model.graph.input]
            onames = [o.name for o in case.model.graph.output]
            ins = [(n, as_tensor(v)) for n, v in zip(names, inputs) if v is not None]
            refs = [(n, as_tensor(v)) for n, v in zip(onames, outputs)]
            if any(a is None for _, a in ins + refs):
                ok, why = False, "a type graphs do not hold"
                break
            try:
                got = graph.run(dict(ins))
            except sp.SpingalettError as e:
                ok, why = False, str(e).split(" (code")[0]
                break
            rtol, atol = tolerances(case, [a for _, a in refs])
            bad = [n for n, a in refs if not close(got[n], a, rtol, atol)]
            if bad:
                ok, why = False, "outputs differ from the reference: " + ", ".join(bad)
                break
            records.append((ins, refs, rtol, atol))
        graph.close()
        if not ok:
            left.append((case.name, why))
            continue
        size = sum(a.nbytes for ins, refs, _, _ in records for _, a in ins + refs)
        if size > MOST_BYTES:
            left.append((case.name, f"passes; left out for its size ({size} bytes)"))
            continue
        for k, (ins, refs, rtol, atol) in enumerate(records):
            name = case.name if len(records) == 1 else f"{case.name}#{k}"
            kept.append(struct.pack("<H", len(name.encode())) + name.encode() + struct.pack("<I", len(model)) + model +
                        struct.pack("<ffI", rtol, atol, len(ins)) + b"".join(pack_tensor(n, a) for n, a in ins) +
                        struct.pack("<I", len(refs)) + b"".join(pack_tensor(n, a) for n, a in refs))
    with open(os.path.join(HERE, "Data", "graph_cases.bin"), "wb") as f:
        f.write(b"SGCASES1" + struct.pack("<I", len(kept)) + b"".join(kept))
    with open(os.path.join(HERE, "Data", "graph_cases.txt"), "w") as f:
        f.write(f"# ONNX's conformance cases of single operators that Tests/Data/graph_cases.bin leaves out "
                f"({len(left)}; {len(kept)} kept), written by Tests/make_graph_tests.py\n")
        for name, why in left:
            f.write(f"{name}: {why}\n")
    print(f"{len(kept)} cases kept, {len(left)} left out")


if __name__ == "__main__":
    main()
