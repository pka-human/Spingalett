# SPDX-License-Identifier: MIT
# Copyright (c) 2026 pka_human (pka_human@proton.me)
"""Writes the cases of the `operators` test group: ONNX's conformance tests of single operators (the cases the onnx
package generates, onnx.backend.test.case.node, whose outputs come from ONNX's reference implementation), into
Tests/Data/graph_cases.bin, and the small model of the runtime's test (Tests/Data/graph_small.onnx and .bin):

    SPINGALETT_LIBRARY=Bin/libspingalett.so python Tests/make_graph_tests.py     # needs onnx and numpy

and lists in Tests/Data/graph_cases.txt the cases left out and why: operators graphs do not have (control flow,
sequences, strings, training, randomness), types they do not hold (float64 and float8, strings, 16-bit
integers), and cases the library fails. The file keeps the cases the library passes, so that the test group holds
it to them; the library's report of a case it fails is the list's line.

The file holds, little-endian:

    "SGCASES1", u32 cases
    per case: u16 name length, name; u32 model length, the ONNX model; f32 rtol, f32 atol;
              u32 inputs, tensors; u32 outputs, tensors (the reference's)
    per tensor: u16 name length, name; u8 type (SpingalettDType); u8 rank; i64 sizes[rank]; u64 bytes; zeros to an
                offset of a multiple of 8; the values

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


def pack_tensor(out, name, arr):
    """Appends a tensor to the bytearray out, its values at an offset of a multiple of 8."""
    data = arr.tobytes()
    encoded = name.encode()
    out += (struct.pack("<H", len(encoded)) + encoded + struct.pack("<BB", TYPES[arr.dtype], arr.ndim) +
            b"".join(struct.pack("<q", d) for d in arr.shape) + struct.pack("<Q", len(data)))
    out += b"\0" * (-len(out) % 8)
    out += data


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
            kept.append((name, model, rtol, atol, ins, refs))
    out = bytearray(b"SGCASES1" + struct.pack("<I", len(kept)))
    for name, model, rtol, atol, ins, refs in kept:
        out += struct.pack("<H", len(name.encode())) + name.encode() + struct.pack("<I", len(model)) + model
        out += struct.pack("<ffI", rtol, atol, len(ins))
        for n, a in ins:
            pack_tensor(out, n, a)
        out += struct.pack("<I", len(refs))
        for n, a in refs:
            pack_tensor(out, n, a)
    with open(os.path.join(HERE, "Data", "graph_cases.bin"), "wb") as f:
        f.write(out)
    with open(os.path.join(HERE, "Data", "graph_cases.txt"), "w") as f:
        f.write(f"# ONNX's conformance cases of single operators that Tests/Data/graph_cases.bin leaves out "
                f"({len(left)}; {len(kept)} kept), written by Tests/make_graph_tests.py\n")
        for name, why in left:
            f.write(f"{name}: {why}\n")
    print(f"{len(kept)} cases kept, {len(left)} left out")


def small_model():
    """Tests/Data/graph_small.onnx and .bin for the runtime's test: x (batch, 4) times W plus b, GELU written out (which
    the library fuses), a softmax, and the argmax; the .bin holds u32 rows, x, the softmax and the argmax."""
    from onnx import TensorProto, helper, numpy_helper
    rng = np.random.default_rng(5)
    W = rng.standard_normal((4, 3)).astype(np.float32)
    b = rng.standard_normal(3).astype(np.float32)
    inits = [numpy_helper.from_array(W, "W"), numpy_helper.from_array(b, "b"),
             numpy_helper.from_array(np.array(1.4142135, np.float32), "root2"),
             numpy_helper.from_array(np.array(1.0, np.float32), "one"), numpy_helper.from_array(np.array(0.5, np.float32), "half")]
    nodes = [helper.make_node("MatMul", ["x", "W"], ["xw"]), helper.make_node("Add", ["xw", "b"], ["h"]),
             helper.make_node("Div", ["h", "root2"], ["d"]), helper.make_node("Erf", ["d"], ["e"]),
             helper.make_node("Add", ["e", "one"], ["e1"]), helper.make_node("Mul", ["half", "e1"], ["m"]),
             helper.make_node("Mul", ["h", "m"], ["g"]), helper.make_node("Softmax", ["g"], ["y"], axis=-1),
             helper.make_node("ArgMax", ["g"], ["k"], axis=1, keepdims=0)]
    graph = helper.make_graph(nodes, "small", [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", 4])],
                              [helper.make_tensor_value_info("y", TensorProto.FLOAT, ["batch", 3]),
                               helper.make_tensor_value_info("k", TensorProto.INT64, ["batch"])], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)], producer_name="make_graph_tests")
    with open(os.path.join(HERE, "Data", "graph_small.onnx"), "wb") as f:
        f.write(model.SerializeToString())
    from onnx.reference import ReferenceEvaluator
    x = rng.standard_normal((5, 4)).astype(np.float32)
    y, k = ReferenceEvaluator(model).run(None, {"x": x})
    with open(os.path.join(HERE, "Data", "graph_small.bin"), "wb") as f:
        f.write(struct.pack("<I", 5) + x.tobytes() + y.astype(np.float32).tobytes() + k.astype(np.int64).tobytes())


if __name__ == "__main__":
    small_model()
    main()
