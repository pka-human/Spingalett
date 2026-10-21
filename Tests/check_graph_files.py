# SPDX-License-Identifier: MIT
# Copyright (c) 2026 pka_human (pka_human@proton.me)
"""ONNX files as other tools wrote them, as graphs of operators against ONNX Runtime: each file is opened by
spingalett.Graph and by onnxruntime, run on the same random inputs (token ids, masks of ones and token types of
zeros for integer inputs, the symbols' sizes given), and the outputs compared (the largest difference relative to the
largest value), both timed (medians):

    SPINGALETT_LIBRARY=Bin/libspingalett.so python Tests/check_graph_files.py model.onnx ... [--dim name=size]
                                                                                       [--threads N]

ONNX Runtime runs in a process of its own: its threads and OpenMP's would contend in one. It needs onnx, onnxruntime
and numpy. Graphs compute floats in single precision, so files of float16 are compared with the tolerance of
float16, and quantized ones with that of a step of their 8-bit values.
"""

import argparse
import os
import statistics
import subprocess
import sys
import tempfile
import time

import numpy as np
import onnx

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "Bindings", "Python"))
import spingalett as sp                                     # noqa: E402

TYPES = {onnx.TensorProto.FLOAT: np.float32, onnx.TensorProto.FLOAT16: np.float16, onnx.TensorProto.INT64: np.int64,
         onnx.TensorProto.INT32: np.int32, onnx.TensorProto.BOOL: np.bool_, onnx.TensorProto.UINT8: np.uint8,
         onnx.TensorProto.INT8: np.int8}

# runs the file in ONNX Runtime on the inputs of an .npz file: its outputs into another, its median time printed
ORT_RUNNER = """
import sys, time, statistics, numpy as np, onnxruntime as ort
path, feeds_file, out_file, threads, runs = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
feeds = dict(np.load(feeds_file))
options = ort.SessionOptions()
if threads:
    options.intra_op_num_threads = threads
session = ort.InferenceSession(path, options, providers=["CPUExecutionProvider"])
outs = session.run(None, feeds)
np.savez(out_file, **{str(k): v for k, v in enumerate(outs)})
times = []
for _ in range(runs):
    t = time.perf_counter()
    session.run(None, feeds)
    times.append(time.perf_counter() - t)
print(statistics.median(times))
"""


def inputs_for(model, dims, rng):
    """Random inputs for the graph's inputs (those without an initializer of the same name)."""
    given = {t.name for t in model.graph.initializer}
    feeds = {}
    for i in model.graph.input:
        if i.name in given:
            continue
        t = i.type.tensor_type
        shape = []
        for d in t.shape.dim:
            if d.HasField("dim_value") and d.dim_value > 0:
                shape.append(d.dim_value)
            else:
                shape.append(dims.get(d.dim_param, dims.get("*", 1)))
        dtype = TYPES[t.elem_type]
        if np.issubdtype(dtype, np.integer):
            value = rng.integers(0, 1000, shape).astype(dtype)
            if "mask" in i.name:
                value = np.ones(shape, dtype)
            elif "type" in i.name:
                value = np.zeros(shape, dtype)
            elif "position" in i.name:
                value = np.broadcast_to(np.arange(shape[-1], dtype=dtype), shape).copy()
        elif dtype == np.bool_:
            value = np.ones(shape, dtype)
        else:
            value = rng.standard_normal(shape).astype(dtype)
        feeds[i.name] = value
    return feeds


def median_time(fn, runs):
    times = []
    for _ in range(runs):
        t = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t)
    return statistics.median(times)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+")
    parser.add_argument("--dim", action="append", default=[])
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--runs", type=int, default=7)
    args = parser.parse_args()
    dims = {k: int(v) for k, v in (d.split("=") for d in args.dim)}
    if args.threads:
        sp.set_num_threads(args.threads)
    sp.set_compute_mode(sp.ComputeMode.OPENMP)
    bad = 0
    for path in args.files:
        name = os.path.basename(os.path.dirname(os.path.dirname(path))) + "/" + os.path.basename(path)
        model = onnx.load(path, load_external_data=False)
        feeds = inputs_for(model, dims, np.random.default_rng(0))
        with tempfile.TemporaryDirectory() as directory:
            feeds_file, out_file = os.path.join(directory, "feeds.npz"), os.path.join(directory, "outs.npz")
            np.savez(feeds_file, **feeds)
            run = subprocess.run([sys.executable, "-c", ORT_RUNNER, path, feeds_file, out_file, str(args.threads),
                                  str(args.runs)], capture_output=True, text=True)
            if run.returncode:
                print(f"{name:52s} ONNX Runtime fails: {run.stderr.strip().splitlines()[-1]}")
                continue
            theirs = float(run.stdout.strip().splitlines()[-1])
            outs = np.load(out_file)
            refs = [outs[str(k)] for k in range(len(outs.files))]
        try:
            t = time.perf_counter()
            graph = sp.Graph.load(path)
            opened = time.perf_counter() - t
            got = graph.run(feeds)
        except sp.SpingalettError as e:
            print(f"{name:52s} FAILS: {e}")
            bad += 1
            continue
        # graphs of float16 compute in single precision where ONNX Runtime computes in half precision, and a
        # quantization (dynamic or not) turns a difference of rounding into a step of its scale
        half = any(t.data_type == onnx.TensorProto.FLOAT16 for t in model.graph.initializer) or \
            any(r.dtype == np.float16 for r in refs)
        quantized = any(n.op_type in ("DynamicQuantizeLinear", "QuantizeLinear") for n in model.graph.node)
        worst = 0.0
        for o, r in zip(model.graph.output, refs):
            g = got[o.name]
            if g.shape != r.shape:
                worst = float("inf")
                print(f"  {o.name}: shape {g.shape}, ONNX Runtime's {r.shape}")
                continue
            if r.dtype.kind == "f":
                worst = max(worst, float(np.max(np.abs(g.astype(np.float64) - r.astype(np.float64)))
                                     / max(float(np.max(np.abs(r.astype(np.float64)))), 1e-30)))
            elif not np.array_equal(g, r):
                worst = float("inf")
        ours = median_time(lambda: graph.run(feeds), args.runs)
        ok = worst < (1e-2 if half else 5e-2 if quantized else 1e-4)
        bad += not ok
        print(f"{name:52s} {'ok ' if ok else 'BAD'} rel. error {worst:.2e}  nodes {len(graph.operators):5d}  "
              f"open {opened * 1e3:7.1f} ms  run {ours * 1e3:8.2f} ms  ONNX Runtime {theirs * 1e3:8.2f} ms")
        graph.close()
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
