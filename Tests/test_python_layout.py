# SPDX-License-Identifier: MIT
"""Checks that the ctypes structures in the Python bindings match the C layout.
Usage: test_python_layout.py <path to SpingalettLayout>   (run through ctest)
"""
import ctypes
import subprocess
import sys

import spingalett as sg

MIRRORS = {
    "NeuralNetworkArgs": sg._NeuralNetworkArgs, "SpingalettNetworkLayer": sg._NetworkLayer,
    "LayerArgs": sg._LayerArgs, "ForwardArgs": sg._ForwardArgs, "TrainArgs": sg._TrainArgs,
    "SaveArgs": sg._SaveArgs, "LRScheduleParams": sg._LRScheduleParams, "PredictArgs": sg._PredictArgs,
    "EvalMetrics": sg._EvalMetrics, "TrainProgress": sg._TrainProgress, "TrainReport": sg._TrainReport,
    "EvaluateArgs": sg._EvaluateArgs, "OptimizerArgs": sg._OptimizerArgs, "SpingalettDataset": sg._Dataset,
    "DatasetSaveOptions": sg._DatasetSaveOptions, "SpingalettDatasetInfo": sg._DatasetInfo,
    "SpingalettModel": sg._Model, "SpingalettLayerInfo": sg._LayerInfo,
}

lines = subprocess.run([sys.argv[1]], check=True, capture_output=True, text=True).stdout.splitlines()
mismatches, seen = [], {name: set() for name in MIRRORS}
for line in lines:
    key, *rest = line.split()
    if rest[0] == "size":
        got, want = ctypes.sizeof(MIRRORS[key]), int(rest[1])
    else:
        struct, field = key.split(".")
        seen[struct].add(field)
        got, want = getattr(MIRRORS[struct], field).offset, int(rest[0])
    if got != want:
        mismatches.append(f"{key}: ctypes {got}, C {want}")

# every ctypes field must have been checked against C (catches fields added only on one side)
for name, mirror in MIRRORS.items():
    extra = {f for f, _ in mirror._fields_} - seen[name]
    if extra:
        mismatches.append(f"{name}: fields not in the C layout dump: {sorted(extra)}")

print(f"{len(lines)} layout checks, {len(mismatches)} mismatches")
for m in mismatches:
    print("MISMATCH", m)
sys.exit(1 if mismatches else 0)
