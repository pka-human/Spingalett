# SPDX-License-Identifier: MIT
# Copyright (c) 2026 pka_human (pka_human@proton.me)
"""Writes the cases of the `operators` test group: ONNX's conformance tests of single operators (the cases the onnx
package generates, onnx.backend.test.case.node, whose outputs come from ONNX's reference implementation), into
Tests/Data/graph_cases.bin, small networks of several operators into Tests/Data/graph_blocks.bin (block_cases()), and
the small model of the runtime's test (Tests/Data/graph_small.onnx and .bin):

    SPINGALETT_LIBRARY=Bin/libspingalett.so python Tests/make_graph_tests.py     # needs onnx and numpy
                                                                                 # (--blocks: graph_blocks.bin alone)

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
    write_cases(os.path.join(HERE, "Data", "graph_cases.bin"), kept)
    with open(os.path.join(HERE, "Data", "graph_cases.txt"), "w") as f:
        f.write(f"# ONNX's conformance cases of single operators that Tests/Data/graph_cases.bin leaves out "
                f"({len(left)}; {len(kept)} kept), written by Tests/make_graph_tests.py\n")
        for name, why in left:
            f.write(f"{name}: {why}\n")
    print(f"{len(kept)} cases kept, {len(left)} left out")


def write_cases(path, kept):
    out = bytearray(b"SGCASES1" + struct.pack("<I", len(kept)))
    for name, model, rtol, atol, ins, refs in kept:
        out += struct.pack("<H", len(name.encode())) + name.encode() + struct.pack("<I", len(model)) + model
        out += struct.pack("<ffI", rtol, atol, len(ins))
        for n, a in ins:
            pack_tensor(out, n, a)
        out += struct.pack("<I", len(refs))
        for n, a in refs:
            pack_tensor(out, n, a)
    with open(path, "wb") as f:
        f.write(out)


def block_cases():
    """Tests/Data/graph_blocks.bin, in graph_cases.bin's format: small networks of several operators, which the layout
    pass, the fusions and the GPU's plans see whole (channels-last convolutions and poolings, a convolution of patches,
    a bottleneck block, attention written out, attention of Q, K and V split from one product or gathered from it; on
    the GPU in bfloat16, values kept as such between products)."""
    from onnx import TensorProto, helper, numpy_helper
    from onnx.reference import ReferenceEvaluator
    rng = np.random.default_rng(11)
    cases = []

    def case(name, nodes, inputs, outputs, inits):
        inits = [numpy_helper.from_array(np.asarray(v), k) for k, v in inits.items()]
        graph = helper.make_graph(nodes, name, [helper.make_tensor_value_info(k, TensorProto.FLOAT, v.shape)
                                                for k, v in inputs.items()],
                                  [helper.make_tensor_value_info(k, TensorProto.FLOAT, None) for k in outputs], inits)
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)], producer_name="make_graph_tests")
        refs = ReferenceEvaluator(model).run(None, inputs)
        model = model.SerializeToString()
        refs = [(k, np.asarray(r, np.float32)) for k, r in zip(outputs, refs)]
        got = sp.Graph.from_bytes(model).run(inputs)
        assert all(close(got[k], r, 1e-3, 1e-4) for k, r in refs), name
        cases.append((name, model, 1e-3, 1e-4, list(inputs.items()), refs))

    def w(*shape, scale=None):
        fan = int(np.prod(shape[1:])) if len(shape) > 1 else shape[0]
        return (rng.standard_normal(shape) * (scale or fan ** -0.5)).astype(np.float32)

    def x(*shape):
        return rng.standard_normal(shape).astype(np.float32)

    conv = helper.make_node
    for pool in ("MaxPool", "AveragePool"):
        attrs = dict(kernel_shape=[3, 3], strides=[2, 2], pads=[1, 1, 1, 1]) if pool == "MaxPool" else \
            dict(kernel_shape=[2, 2], strides=[2, 2])
        case(f"conv_{pool.lower()}", [conv("Conv", ["x", "W", "B"], ["c"], pads=[1, 1, 1, 1]),
                                      conv("Relu", ["c"], ["r"]), conv(pool, ["r"], ["y"], **attrs)],
             {"x": x(1, 3, 32, 32)}, ["y", "r"], {"W": w(8, 3, 3, 3), "B": w(8, scale=0.1)})
    case("patches", [conv("Conv", ["x", "W", "B"], ["c"], kernel_shape=[16, 16], strides=[16, 16]),
                     conv("Reshape", ["c", "shape"], ["s"]), conv("Transpose", ["s"], ["t"], perm=[0, 2, 1]),
                     conv("LayerNormalization", ["t", "g", "b"], ["n"], axis=-1),
                     conv("MatMul", ["n", "Wl"], ["m"]), conv("Add", ["m", "bl"], ["y"])],
         {"x": x(1, 3, 64, 64)}, ["y"],
         {"W": w(48, 3, 16, 16), "B": w(48, scale=0.1), "shape": np.array([1, 48, 16], np.int64),
          "g": 1 + w(48, scale=0.1), "b": w(48, scale=0.1), "Wl": w(48, 32), "bl": w(32, scale=0.1)})
    case("bottleneck", [conv("Conv", ["x", "W1", "B1"], ["c1"], kernel_shape=[7, 7], strides=[2, 2], pads=[3] * 4),
                        conv("Relu", ["c1"], ["r1"]),
                        conv("MaxPool", ["r1"], ["p"], kernel_shape=[3, 3], strides=[2, 2], pads=[1] * 4),
                        conv("Conv", ["p", "Wa"], ["a0"]), conv("Relu", ["a0"], ["a"]),
                        conv("Conv", ["a", "Wb"], ["b0"], pads=[1] * 4), conv("Relu", ["b0"], ["b"]),
                        conv("Conv", ["b", "Wc"], ["c"]), conv("Conv", ["p", "Wd"], ["d"]),
                        conv("Add", ["c", "d"], ["s0"]), conv("Relu", ["s0"], ["s"]),
                        conv("GlobalAveragePool", ["s"], ["gp"]), conv("Flatten", ["gp"], ["f"]),
                        conv("Gemm", ["f", "Wf", "bf"], ["y"], transB=1)],
         {"x": x(1, 3, 64, 64)}, ["y"],
         {"W1": w(32, 3, 7, 7), "B1": w(32, scale=0.1), "Wa": w(32, 32, 1, 1), "Wb": w(32, 32, 3, 3),
          "Wc": w(64, 32, 1, 1), "Wd": w(64, 32, 1, 1), "Wf": w(10, 64), "bf": w(10, scale=0.1)})
    heads = [conv("LayerNormalization", ["x", "g", "b"], ["n"], axis=-1)]
    for p, perm in (("q", [0, 2, 1, 3]), ("k", [0, 2, 3, 1]), ("v", [0, 2, 1, 3])):
        heads += [conv("MatMul", ["n", "W" + p], [p + "0"]), conv("Add", [p + "0", "b" + p], [p + "1"]),
                  conv("Reshape", [p + "1", "split"], [p + "2"]), conv("Transpose", [p + "2"], [p], perm=perm)]
    heads += [conv("MatMul", ["q", "k"], ["s0"]), conv("Mul", ["s0", "scale"], ["s"]),
              conv("Softmax", ["s"], ["sm"], axis=-1), conv("MatMul", ["sm", "v"], ["o0"]),
              conv("Transpose", ["o0"], ["o1"], perm=[0, 2, 1, 3]), conv("Reshape", ["o1", "merge"], ["o2"]),
              conv("MatMul", ["o2", "Wo"], ["o3"]), conv("Add", ["o3", "bo"], ["o"]), conv("Add", ["o", "x"], ["y"])]
    case("attention", heads, {"x": x(1, 32, 64)}, ["y"],
         {"g": 1 + w(64, scale=0.1), "b": w(64, scale=0.1), "Wq": w(64, 64), "bq": w(64, scale=0.1),
          "Wk": w(64, 64), "bk": w(64, scale=0.1), "Wv": w(64, 64), "bv": w(64, scale=0.1),
          "split": np.array([1, 32, 2, 32], np.int64), "scale": np.array(32 ** -0.5, np.float32),
          "merge": np.array([1, 32, 64], np.int64), "Wo": w(64, 64), "bo": w(64, scale=0.1)})
    # Q, K and V of one product: split in three as GPT-2's export has them, gathered as torchvision's (sequence first)
    split = [conv("MatMul", ["x", "Wp"], ["p0"]), conv("Add", ["p0", "bp"], ["p"]),
             conv("Split", ["p", "thirds"], ["q0", "k0", "v0"], axis=2)]
    for t, perm in (("q", [0, 2, 1, 3]), ("v", [0, 2, 1, 3]), ("k", [0, 2, 1, 3])):
        split += [conv("Reshape", [t + "0", "heads"], [t + "1"]), conv("Transpose", [t + "1"], [t + "2"], perm=perm)]
    split += [conv("Reshape", ["k2", "merged"], ["k3"]), conv("Transpose", ["k3"], ["k4"], perm=[0, 2, 1]),
              conv("Reshape", ["k4", "kt_shape"], ["kt"]), conv("MatMul", ["q2", "kt"], ["s0"]),
              conv("Mul", ["s0", "scale"], ["s"]), conv("Softmax", ["s"], ["sm"], axis=-1),
              conv("MatMul", ["sm", "v2"], ["o0"]), conv("Transpose", ["o0"], ["o1"], perm=[0, 2, 1, 3]),
              conv("Reshape", ["o1", "back"], ["y"])]
    case("attention_split", split, {"x": x(1, 24, 64)}, ["y"],
         {"Wp": w(64, 192), "bp": w(192, scale=0.1), "thirds": np.array([64, 64, 64], np.int64),
          "heads": np.array([1, 24, 2, 32], np.int64), "merged": np.array([2, 24, 32], np.int64),
          "kt_shape": np.array([1, 2, 32, 24], np.int64), "scale": np.array(32 ** -0.5, np.float32),
          "back": np.array([1, 24, 64], np.int64)})
    gather = [conv("MatMul", ["x", "Wp"], ["p0"]), conv("Add", ["p0", "bp"], ["p"]),
              conv("Reshape", ["p", "packs"], ["p1"]), conv("Transpose", ["p1"], ["p2"], perm=[2, 1, 0, 3])]
    for i, t in enumerate("qkv"):
        gather += [conv("Gather", ["p2", f"i{i}"], [t + "0"], axis=0), conv("Reshape", [t + "0", "heads"], [t + "1"]),
                   conv("Transpose", [t + "1"], [t + "2"], perm=[0, 2, 3, 1] if t == "k" else [0, 2, 1, 3])]
    gather += [conv("MatMul", ["q2", "k2"], ["s0"]), conv("Mul", ["s0", "scale"], ["s"]),
               conv("Softmax", ["s"], ["sm"], axis=-1), conv("MatMul", ["sm", "v2"], ["o0"]),
               conv("Transpose", ["o0"], ["o1"], perm=[2, 0, 1, 3]), conv("Reshape", ["o1", "back"], ["y"])]
    case("attention_gather", gather, {"x": x(17, 1, 64)}, ["y"],
         {"Wp": w(64, 192), "bp": w(192, scale=0.1), "packs": np.array([17, 1, 3, 64], np.int64),
          "i0": np.array(0, np.int64), "i1": np.array(1, np.int64), "i2": np.array(2, np.int64),
          "heads": np.array([1, 17, 2, 32], np.int64), "scale": np.array(32 ** -0.5, np.float32),
          "back": np.array([17, 1, 64], np.int64)})
    write_cases(os.path.join(HERE, "Data", "graph_blocks.bin"), cases)
    print(f"{len(cases)} cases of blocks")


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
    block_cases()
    if "--blocks" not in sys.argv:
        small_model()
        main()
