# SPDX-License-Identifier: MIT
# Copyright (c) 2026 pka_human (pka_human@proton.me)
"""Python bindings for the Spingalett C23 deep learning engine.

Pure ``ctypes`` over the shared library: nothing is compiled at install time, and the wheels on
PyPI carry the library inside the package. The library is located in this order:

1. the ``SPINGALETT_LIBRARY`` environment variable (full path to the library),
2. inside this package (wheels), then the repository's ``Bin/`` directory,
3. the system loader (``LD_LIBRARY_PATH``, ``DYLD_LIBRARY_PATH``, ``PATH``).

Example::

    import numpy as np
    import spingalett as sg

    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    y = np.array([[0], [1], [1], [0]], dtype=np.float32)

    with sg.Network(sg.Loss.MSE, [sg.Layer(2),
                                  sg.Layer(8, sg.Activation.TANH, sg.Init.XAVIER),
                                  sg.Layer(1, sg.Activation.SIGMOID, sg.Init.XAVIER)]) as net:
        net.train(x, y, epochs=5000, optimizer=sg.Optimizer.ADAM, learning_rate=0.02)
        print(net.forward(x))
"""

from __future__ import annotations

import ctypes
import ctypes.util
import dataclasses
import enum
import os
import sys
from ctypes import (CFUNCTYPE, POINTER, Structure, c_bool, c_char_p, c_float, c_int,
                    c_size_t, c_uint32, c_uint64, c_void_p)
from typing import Callable, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

__version__ = "0.10.0"

__all__ = [
    "Activation", "Loss", "Init", "Strategy", "Optimizer", "ComputeMode", "Precision",
    "AutoSave", "LogLevel", "ErrorCode", "Monitor", "TrainStatus", "Layer", "TrainConfig", "Network",
    "LayerType", "Input", "Conv2D", "MaxPool2D", "AvgPool2D", "BatchNorm", "LayerDescription",
    "Model", "LayerInfo",
    "Metrics", "Progress", "TrainResult", "Trainer", "SpingalettError", "load_idx", "load_cifar", "load_csv",
    "DatasetEncoding", "save_dataset", "load_dataset", "dataset_info",
    "CosineDecay", "LinearWarmup", "StepDecay", "WarmupCosine",
    "set_compute_mode", "get_compute_mode", "set_num_threads", "get_num_threads", "cpu_kernels",
    "seed", "set_verbose", "set_log_level", "set_log_callback", "library_path", "library_version",
]


# --------------------------------------------------------------------------- enums
# Values mirror Include/Spingalett/Spingalett.h.

class Activation(enum.IntEnum):
    SIGMOID = 0
    RELU = 1
    TANH = 2
    LEAKY_RELU = 3
    FOO52 = 4
    SOFTMAX = 5
    NONE = 6


class Loss(enum.IntEnum):
    MSE = 0
    CROSS_ENTROPY = 1


class Init(enum.IntEnum):
    RANDOM = 0      # uniform in [-1, 1]
    XAVIER = 1      # Glorot normal, variance 2 / (fan_in + fan_out)
    HE = 2          # He normal, variance 2 / fan_in
    NONE = 3        # zeros
    LECUN = 4       # LeCun normal, variance 1 / fan_in


class Strategy(enum.IntEnum):
    SAMPLE = 0
    FULL_BATCH = 1
    MINI_BATCH = 2


class Optimizer(enum.IntEnum):
    SGD = 0
    MOMENTUM = 1
    RMSPROP = 2
    ADAM = 3
    ADAMW = 4


class ComputeMode(enum.IntEnum):
    SINGLE_THREADED = 0
    OPENMP = 1
    OPENBLAS = 2
    CUDA = 3


class Precision(enum.IntEnum):
    FLOAT32 = 0
    FP16 = 1
    BFLOAT16 = 2
    INT8 = 3
    INT4 = 4
    INT2 = 5


class LayerType(enum.IntEnum):
    DENSE = 0
    CONV2D = 1
    MAX_POOL2D = 2
    AVG_POOL2D = 3
    BATCH_NORM = 4
    ADD = 5
    CONCAT = 6
    GLOBAL_AVG_POOL = 7


MAX_INPUTS = 16     # inputs of a layer, at most (SPINGALETT_MAX_INPUTS)


class AutoSave(enum.IntEnum):
    OFF = 0
    OVERWRITE = 1
    NEW_FILES = 2


class ErrorCode(enum.IntEnum):
    OK = 0
    ALLOC = 1
    INVALID = 2
    FILE_IO = 3
    FORMAT_VERSION = 4


class Monitor(enum.IntEnum):
    AUTO = 0            # validation loss with validation data, else training loss
    TRAIN_LOSS = 1
    VAL_LOSS = 2
    VAL_ACCURACY = 3


class DatasetEncoding(enum.IntEnum):
    AUTO = 0            # smallest lossless; one-hot targets become CLASS
    FLOAT32 = 1
    FP16 = 2
    BFLOAT16 = 3
    U8_UNIT = 4         # q / 255
    U8_AFFINE = 5       # per-feature 8-bit quantization (lossy)
    CLASS = 6           # targets: class index of each row


class TrainStatus(enum.IntEnum):
    FAILED = 0
    COMPLETED = 1
    EARLY_STOPPED = 2
    INTERRUPTED = 3
    DIVERGED = 4
    NO_DATA = 5


class LogLevel(enum.IntEnum):
    DEBUG = 0
    INFO = 1
    WARNING = 2
    ERROR = 3
    NONE = 4


_MODE_ARRAY = 0
_MODE_GENERATOR = 1


# --------------------------------------------------------------------------- C structs
# Field order and types must match Spingalett.h exactly (enums are C ints).

_NetPtr = c_void_p      # NeuralNetwork is opaque


class _NetworkLayer(Structure):
    _fields_ = [
        ("type", c_int),
        ("height", c_uint32),
        ("width", c_uint32),
        ("channels", c_uint32),
        ("outputs", c_uint32),
        ("activation", c_int),
        ("dropout_rate", c_float),
        ("kernel_h", c_uint32),
        ("kernel_w", c_uint32),
        ("stride_h", c_uint32),
        ("stride_w", c_uint32),
        ("padding_h", c_uint32),
        ("padding_w", c_uint32),
        ("weight_count", c_uint64),
        ("bias_count", c_uint64),
        ("groups", c_uint32),
        ("epsilon", c_float),
        ("momentum", c_float),
        ("input_count", c_uint32),
        ("inputs", c_uint32 * MAX_INPUTS),
    ]


class _EvalMetrics(Structure):
    _fields_ = [("loss", c_float), ("accuracy", c_float)]


class _TrainProgress(Structure):
    _fields_ = [
        ("epoch", c_size_t),
        ("epochs", c_size_t),
        ("train_loss", c_float),
        ("learning_rate", c_float),
        ("has_validation", c_bool),
        ("validation", _EvalMetrics),
        ("monitor", c_int),
        ("best_epoch", c_size_t),
        ("best_value", c_float),
        ("improved", c_bool),
    ]


class _TrainReport(Structure):
    _fields_ = [
        ("status", c_int),
        ("epochs_run", c_size_t),
        ("train_loss", c_float),
        ("has_validation", c_bool),
        ("validation", _EvalMetrics),
        ("monitor", c_int),
        ("best_epoch", c_size_t),
        ("best_value", c_float),
        ("restored_best", c_bool),
    ]


_TrainCallbackFn = CFUNCTYPE(c_bool, _NetPtr, POINTER(_TrainProgress), c_void_p)
_LRSchedulerFn = CFUNCTYPE(c_float, c_size_t, c_size_t, c_float, c_void_p)
_LogCallbackFn = CFUNCTYPE(None, c_int, c_char_p)
_DataGeneratorFn = CFUNCTYPE(c_uint32, POINTER(c_float), POINTER(c_float), c_uint32, c_void_p)


class _NeuralNetworkArgs(Structure):
    _fields_ = [("loss_func", c_int)]


class _LayerArgs(Structure):
    _fields_ = [
        ("net", _NetPtr),
        ("neurons_amount", c_uint32),
        ("act_func", c_int),
        ("weight_initialization", c_int),
        ("dropout_rate", c_float),
        ("type", c_int),
        ("height", c_uint32),
        ("width", c_uint32),
        ("channels", c_uint32),
        ("filters", c_uint32),
        ("kernel", c_uint32),
        ("stride", c_uint32),
        ("padding", c_uint32),
        ("kernel_h", c_uint32),
        ("kernel_w", c_uint32),
        ("stride_h", c_uint32),
        ("stride_w", c_uint32),
        ("padding_h", c_uint32),
        ("padding_w", c_uint32),
        ("groups", c_uint32),
        ("epsilon", c_float),
        ("momentum", c_float),
        ("inputs", c_uint32 * MAX_INPUTS),
        ("input_count", c_uint32),
    ]


# Arrays passed in and out of the calls made per batch go by address (c_void_p, from
# ndarray.ctypes.data): building a ctypes float pointer costs about a microsecond more per array.
_FloatArray = c_void_p


class _ForwardArgs(Structure):
    _fields_ = [("net", _NetPtr), ("input", _FloatArray)]


class _TrainArgs(Structure):
    _fields_ = [
        ("net", _NetPtr),
        ("training_mode", c_int),
        ("training_strategy", c_int),
        ("optimizer_type", c_int),
        ("inputs", POINTER(c_float)),
        ("targets", POINTER(c_float)),
        ("generator", _DataGeneratorFn),
        ("generator_data", c_void_p),
        ("sample_count", c_uint32),
        ("batch_size", c_uint32),
        ("do_not_shuffle", c_bool),
        ("epochs", c_size_t),
        ("learning_rate", c_float),
        ("weight_decay", c_float),
        ("momentum", c_float),
        ("beta1", c_float),
        ("beta2", c_float),
        ("epsilon", c_float),
        ("max_grad_norm", c_float),
        ("reset_optimizer", c_bool),
        ("nan_check_interval", c_size_t),
        ("report_interval", c_size_t),
        ("autosave_mode", c_int),
        ("autosave_interval", c_size_t),
        ("autosave_path", c_char_p),
        ("autosave_do_not_save_optimizer", c_bool),
        ("autosave_precision", c_int),
        ("callback", _TrainCallbackFn),
        ("callback_interval", c_size_t),
        ("callback_data", c_void_p),
        ("lr_scheduler", _LRSchedulerFn),
        ("lr_scheduler_data", c_void_p),
        ("val_inputs", POINTER(c_float)),
        ("val_targets", POINTER(c_float)),
        ("val_count", c_uint32),
        ("monitor", c_int),
        ("early_stopping_patience", c_size_t),
        ("early_stopping_min_delta", c_float),
        ("restore_best_weights", c_bool),
        ("blas_num_threads", c_int),
        ("augment_shift", c_uint32),
        ("augment_flip", c_bool),
        ("label_smoothing", c_float),
        ("lr_plateau_factor", c_float),
        ("lr_plateau_patience", c_size_t),
        ("lr_plateau_min_lr", c_float),
    ]


class _PredictArgs(Structure):
    _fields_ = [
        ("net", _NetPtr),
        ("inputs", _FloatArray),
        ("sample_count", c_uint32),
        ("outputs", _FloatArray),
    ]


class _EvaluateArgs(Structure):
    _fields_ = [
        ("net", _NetPtr),
        ("inputs", _FloatArray),
        ("targets", _FloatArray),
        ("sample_count", c_uint32),
    ]


class _OptimizerArgs(Structure):
    _fields_ = [
        ("type", c_int),
        ("learning_rate", c_float),
        ("weight_decay", c_float),
        ("momentum", c_float),
        ("beta1", c_float),
        ("beta2", c_float),
        ("epsilon", c_float),
        ("max_grad_norm", c_float),
    ]


class _Dataset(Structure):
    _fields_ = [
        ("count", c_uint32),
        ("input_size", c_uint32),
        ("target_size", c_uint32),
        ("inputs", POINTER(c_float)),
        ("targets", POINTER(c_float)),
        ("height", c_uint32),
        ("width", c_uint32),
        ("channels", c_uint32),
        ("class_names", POINTER(c_char_p)),
    ]


class _TargetSet(Structure):
    _fields_ = [
        ("name", c_char_p),
        ("size", c_uint32),
        ("targets", POINTER(c_float)),
        ("class_names", POINTER(c_char_p)),
        ("encoding", c_int),
    ]


class _DatasetSaveOptions(Structure):
    _fields_ = [
        ("input_encoding", c_int),
        ("target_encoding", c_int),
        ("no_compression", c_bool),
        ("target_name", c_char_p),
        ("extra_targets", POINTER(_TargetSet)),
        ("extra_target_count", c_uint32),
    ]


class _DatasetInfo(Structure):
    _fields_ = [
        ("count", c_uint32),
        ("input_size", c_uint32),
        ("target_size", c_uint32),
        ("input_encoding", c_int),
        ("target_encoding", c_int),
        ("chunk_count", c_uint32),
        ("file_size", c_uint64),
        ("format_version", c_uint32),
        ("height", c_uint32),
        ("width", c_uint32),
        ("channels", c_uint32),
        ("target_set_count", c_uint32),
        ("target_set", c_uint32),
    ]


class _DatasetReaderOptions(Structure):
    _fields_ = [("shuffle", c_bool), ("in_memory", c_bool), ("no_prefetch", c_bool), ("target_set", c_uint32)]


class _SaveArgs(Structure):
    _fields_ = [
        ("net", _NetPtr),
        ("filename", c_char_p),
        ("do_not_save_optimizer", c_bool),
        ("precision", c_int),
    ]


class _Model(Structure):
    _fields_ = [
        ("input_size", c_uint32),
        ("output_size", c_uint32),
        ("layer_count", c_uint32),
        ("loss", c_int),
        ("workspace_size", c_size_t),
        ("image", c_void_p),
        ("image_size", c_size_t),
        ("max_width_", c_uint32),
        ("max_int_inputs_", c_uint32),
        ("conv_scratch_", c_size_t),
        ("owner_", c_void_p),
        ("activations_", c_size_t),
    ]


_ModelPtr = POINTER(_Model)


class _LayerInfo(Structure):
    _fields_ = [
        ("type", c_int),
        ("inputs", c_uint32),
        ("outputs", c_uint32),
        ("activation", c_int),
        ("precision", c_int),
        ("in_height", c_uint32),
        ("in_width", c_uint32),
        ("in_channels", c_uint32),
        ("height", c_uint32),
        ("width", c_uint32),
        ("channels", c_uint32),
        ("kernel_h", c_uint32),
        ("kernel_w", c_uint32),
        ("stride_h", c_uint32),
        ("stride_w", c_uint32),
        ("padding_h", c_uint32),
        ("padding_w", c_uint32),
        ("groups", c_uint32),
        ("epsilon", c_float),
        ("input_count", c_uint32),
        ("input_layers", c_uint32 * MAX_INPUTS),
    ]


class _LRScheduleParams(Structure):
    _fields_ = [
        ("warmup_epochs", c_size_t),
        ("step_size", c_size_t),
        ("gamma", c_float),
        ("min_lr", c_float),
    ]


# --------------------------------------------------------------------------- library loading

def _library_names() -> List[str]:
    if sys.platform.startswith("win"):
        return ["spingalett.dll", "libspingalett.dll"]
    abi = ".".join(__version__.split(".")[:2])     # the soname carries major.minor before 1.0
    if sys.platform == "darwin":
        return ["libspingalett.dylib", f"libspingalett.{abi}.dylib"]
    return ["libspingalett.so", f"libspingalett.so.{abi}"]


def _load_library() -> ctypes.CDLL:
    env = os.environ.get("SPINGALETT_LIBRARY")
    if env:
        return ctypes.CDLL(env)

    here = os.path.dirname(os.path.abspath(__file__))
    if sys.platform.startswith("win") and hasattr(os, "add_dll_directory"):
        os.add_dll_directory(here)              # the runtime DLLs a wheel carries next to the library
    candidates = [os.path.join(here, n) for n in _library_names()]
    candidates += [os.path.join(here, "..", "..", "..", "Bin", n) for n in _library_names()]   # a checkout
    for path in candidates:
        if os.path.exists(path):
            return ctypes.CDLL(os.path.normpath(path))

    found = ctypes.util.find_library("spingalett")
    errors = []
    for name in ([found] if found else []) + _library_names():
        try:
            return ctypes.CDLL(name)
        except OSError as exc:
            errors.append(str(exc))
    raise ImportError(
        "Could not load the Spingalett shared library. Build it with CMake and either set "
        "SPINGALETT_LIBRARY to its path or put it on the library search path. "
        "Tried: " + "; ".join(errors))


_lib = _load_library()

# The bindings mirror C struct layouts, which may change between minor releases before 1.0:
# refuse to run against a library of another major.minor version instead of corrupting memory.
try:
    _version_fn = _lib.spingalett_version
except AttributeError:
    raise ImportError(f"{_lib._name} predates Spingalett 0.2; rebuild the library") from None
_version_fn.restype = c_char_p
_version_fn.argtypes = []
_LIBRARY_VERSION = _version_fn().decode()
if _LIBRARY_VERSION.split(".")[:2] != __version__.split(".")[:2]:
    raise ImportError(f"spingalett bindings {__version__} require library version "
                      f"{'.'.join(__version__.split('.')[:2])}.x, but {_lib._name} is {_LIBRARY_VERSION}")


def _bind(name, restype, argtypes):
    fn = getattr(_lib, name)
    fn.restype = restype
    fn.argtypes = argtypes
    return fn


_new = _bind("new_spingalett_struct_arguments", _NetPtr, [_NeuralNetworkArgs])
_layer = _bind("layer_struct_arguments", c_uint32, [_LayerArgs])
_layer_count = _bind("spingalett_layer_count", c_uint32, [_NetPtr])
_input_size = _bind("spingalett_input_size", c_uint32, [_NetPtr])
_output_size = _bind("spingalett_output_size", c_uint32, [_NetPtr])
_network_layer = _bind("spingalett_network_layer", c_bool, [_NetPtr, c_uint32, POINTER(_NetworkLayer)])
_parameter_count = _bind("spingalett_parameter_count", c_uint64, [_NetPtr])
_network_loss = _bind("spingalett_network_loss", c_int, [_NetPtr])
_optimizer_steps = _bind("spingalett_optimizer_steps", c_uint64, [_NetPtr])
_get_parameters = _bind("spingalett_get_parameters", c_bool, [_NetPtr, c_uint32, c_int, POINTER(c_float), c_uint64])
_set_parameters = _bind("spingalett_set_parameters", c_bool, [_NetPtr, c_uint32, c_int, POINTER(c_float), c_uint64])
_PARAM_WEIGHTS, _PARAM_BIASES, _PARAM_WEIGHT_GRADIENTS, _PARAM_BIAS_GRADIENTS = 0, 1, 2, 3
_PARAM_RUNNING_MEAN, _PARAM_RUNNING_VARIANCE = 4, 5
_forward = _bind("forward_struct_arguments", c_void_p, [_ForwardArgs])
_predict = _bind("predict_struct_arguments", c_bool, [_PredictArgs])
_train = _bind("train_struct_arguments", _TrainReport, [_TrainArgs])
_evaluate = _bind("evaluate_struct_arguments", _EvalMetrics, [_EvaluateArgs])
_TrainerPtr = c_void_p
_trainer_new = _bind("spingalett_trainer_new", _TrainerPtr, [_NetPtr, c_uint32])
_trainer_free = _bind("spingalett_trainer_free", None, [_TrainerPtr])
_trainer_forward = _bind("spingalett_trainer_forward", c_void_p, [_TrainerPtr, _FloatArray, c_uint32])
_trainer_backward = _bind("spingalett_trainer_backward", c_float, [_TrainerPtr, _FloatArray])
_trainer_backward_grads = _bind("spingalett_trainer_backward_output_grads", c_bool, [_TrainerPtr, _FloatArray])
_trainer_step = _bind("spingalett_trainer_step", c_bool, [_TrainerPtr, POINTER(_OptimizerArgs)])
_trainer_zero_grad = _bind("spingalett_trainer_zero_grad", None, [_TrainerPtr])
_load_idx = _bind("spingalett_load_idx", c_bool, [c_char_p, c_char_p, c_uint32, POINTER(_Dataset)])
_load_cifar = _bind("spingalett_load_cifar", c_bool, [POINTER(c_char_p), c_uint32, c_uint32, POINTER(_Dataset)])
_load_csv = _bind("spingalett_load_csv", c_bool, [c_char_p, c_uint32, c_uint32, POINTER(_Dataset)])
_dataset_free = _bind("spingalett_dataset_free", None, [POINTER(_Dataset)])
_save_dataset = _bind("spingalett_save_dataset", c_bool, [POINTER(_Dataset), c_char_p, POINTER(_DatasetSaveOptions)])
_load_dataset = _bind("spingalett_load_dataset_targets", c_bool, [c_char_p, c_uint32, POINTER(_Dataset)])
_dataset_set_class_names = _bind("spingalett_dataset_set_class_names", c_bool, [POINTER(_Dataset), POINTER(c_char_p), c_uint32])
_dataset_open = _bind("spingalett_dataset_open_ex", c_void_p, [c_char_p, POINTER(_DatasetReaderOptions)])
_dataset_open_u8 = _bind("spingalett_dataset_open_u8", c_void_p,
                         [POINTER(ctypes.c_uint8), POINTER(c_float), c_uint32, c_uint32, c_uint32, c_bool])
_dataset_target_set_name = _bind("spingalett_dataset_target_set_name", c_char_p, [c_void_p, c_uint32])
_dataset_class_name = _bind("spingalett_dataset_class_name", c_char_p, [c_void_p, c_uint32, c_uint32])
_dataset_target_set_size = _bind("spingalett_dataset_target_set_size", c_uint32, [c_void_p, c_uint32])
_dataset_close = _bind("spingalett_dataset_close", None, [c_void_p])
_dataset_info = _bind("spingalett_dataset_info", _DatasetInfo, [c_void_p])
_dataset_generator = _DataGeneratorFn(ctypes.cast(_lib.spingalett_dataset_generator, c_void_p).value)
_save = _bind("save_spingalett_struct_arguments", None, [_SaveArgs])
_save_to_memory = _bind("spingalett_save_to_memory", c_void_p, [_NetPtr, c_int, c_bool, POINTER(c_size_t)])
_load_from_memory = _bind("load_spingalett_from_memory", _NetPtr, [c_char_p, c_size_t])
_free_memory = _bind("spingalett_free", None, [c_void_p])
_export_c_header = _bind("spingalett_export_c_header", c_bool, [_NetPtr, c_char_p, c_char_p, c_int])
_model_from_network = _bind("spingalett_model_from_network", _ModelPtr, [_NetPtr, c_int])
_model_load = _bind("spingalett_model_load", _ModelPtr, [c_char_p])
_model_from_memory = _bind("spingalett_model_from_memory", _ModelPtr, [c_char_p, c_size_t])
_model_free = _bind("spingalett_model_free", None, [_ModelPtr])
_model_layer = _bind("spingalett_model_layer", c_bool, [_ModelPtr, c_uint32, POINTER(_LayerInfo)])
_model_predict = _bind("spingalett_model_predict", c_bool, [_ModelPtr, _FloatArray, c_uint32, _FloatArray])
_model_evaluate = _bind("spingalett_model_evaluate", _EvalMetrics, [_ModelPtr, _FloatArray, _FloatArray, c_uint32])
_load = _bind("load_spingalett", _NetPtr, [c_char_p])
_import_onnx = _bind("spingalett_import_onnx", _NetPtr, [c_char_p])
_import_onnx_from_memory = _bind("spingalett_import_onnx_from_memory", _NetPtr, [c_char_p, c_size_t])
_load_pytorch = _bind("spingalett_load_pytorch", c_bool, [_NetPtr, c_char_p, POINTER(c_char_p), c_uint32])
_load_pytorch_from_memory = _bind("spingalett_load_pytorch_from_memory", c_bool,
                                  [_NetPtr, c_char_p, c_size_t, POINTER(c_char_p), c_uint32])
_free = _bind("free_network", None, [_NetPtr])
_print_parameters = _bind("print_parameters", None, [_NetPtr])

_last_error_code = _bind("spingalett_last_error_code", c_int, [])
_last_error_message = _bind("spingalett_last_error_message", c_char_p, [])
_clear_error = _bind("spingalett_clear_error", None, [])

_get_compute_mode = _bind("spingalett_get_compute_mode", c_int, [])
_set_compute_mode = _bind("spingalett_set_compute_mode", None, [c_int])
_get_num_threads = _bind("spingalett_get_num_threads", ctypes.c_uint, [])
_cpu_kernels = _bind("spingalett_cpu_kernels", c_char_p, [])
_set_num_threads = _bind("spingalett_set_num_threads", None, [ctypes.c_uint])
_set_log_callback = _bind("spingalett_set_log_callback", None, [_LogCallbackFn])
_set_log_level = _bind("spingalett_set_log_level", None, [c_int])
_set_verbose = _bind("spingalett_set_verbose", None, [c_bool])
_seed = _bind("spingalett_seed", None, [c_uint64])


def library_path() -> str:
    """Path of the loaded shared library."""
    return _lib._name


def library_version() -> str:
    """Version of the loaded shared library."""
    return _LIBRARY_VERSION


# --------------------------------------------------------------------------- errors

class SpingalettError(RuntimeError):
    """Raised when a library call reports an error."""

    def __init__(self, code: int, message: str):
        super().__init__(f"{message} (code {code})")
        try:
            self.code = ErrorCode(code)
        except ValueError:
            self.code = code


def _call(fn, *args):
    _clear_error()
    result = fn(*args)
    code = _last_error_code()
    if code != 0:
        raise SpingalettError(code, (_last_error_message() or b"unknown error").decode(errors="replace"))
    return result


def _encode_path(path) -> bytes:
    return os.fsencode(os.fspath(path))


# --------------------------------------------------------------------------- global settings

def set_compute_mode(mode: ComputeMode) -> None:
    _set_compute_mode(int(mode))


def get_compute_mode() -> ComputeMode:
    return ComputeMode(_get_compute_mode())


def set_num_threads(n: int) -> None:
    """Threads for OpenMP (and OpenBLAS when large enough); 0 = runtime default."""
    _set_num_threads(int(n))


def get_num_threads() -> int:
    return int(_get_num_threads())


def cpu_kernels() -> str:
    """Instruction set of the matrix-multiplication kernels in use ("AVX-512", "AVX2", "AVX",
    "SSE2", "NEON" or "C"); x86-64 libraries not built for the build machine choose it at run time."""
    return _cpu_kernels().decode()


def seed(value: int) -> None:
    """Seed the calling thread's generator (weight init, shuffling, dropout)."""
    _seed(int(value) & 0xFFFFFFFFFFFFFFFF)


def set_verbose(enabled: bool) -> None:
    _set_verbose(bool(enabled))


def set_log_level(level: LogLevel) -> None:
    _set_log_level(int(level))


_log_callback_ref = None


def set_log_callback(callback: Optional[Callable[[LogLevel, str], None]]) -> None:
    """Route library log messages to ``callback(level, message)``; ``None`` restores stdout/stderr."""
    global _log_callback_ref
    if callback is None:
        _log_callback_ref = None
        _set_log_callback(_LogCallbackFn())
        return

    def trampoline(level, message):
        try:
            callback(LogLevel(level), message.decode(errors="replace"))
        except Exception:  # never let an exception unwind through C
            pass

    _log_callback_ref = _LogCallbackFn(trampoline)
    _set_log_callback(_log_callback_ref)


# --------------------------------------------------------------------------- schedules

class _BuiltinSchedule:
    _symbol = ""

    def __init__(self, **params):
        self._params = _LRScheduleParams(**params)
        self._fn = _LRSchedulerFn(ctypes.cast(getattr(_lib, self._symbol), c_void_p).value)

    def _as_c(self):
        return self._fn, ctypes.cast(ctypes.pointer(self._params), c_void_p)

    def __call__(self, epoch: int, total_epochs: int, initial_lr: float) -> float:
        return float(self._fn(epoch, total_epochs, initial_lr, ctypes.cast(ctypes.pointer(self._params), c_void_p)))


class CosineDecay(_BuiltinSchedule):
    """Half-cosine from ``learning_rate`` down to ``min_lr`` over the run."""
    _symbol = "spingalett_lr_cosine_decay"

    def __init__(self, min_lr: float = 0.0):
        super().__init__(min_lr=min_lr)


class LinearWarmup(_BuiltinSchedule):
    """Linear ramp over ``warmup_epochs`` (0 = 5% of the run), then constant."""
    _symbol = "spingalett_lr_linear_warmup"

    def __init__(self, warmup_epochs: int = 0):
        super().__init__(warmup_epochs=warmup_epochs)


class StepDecay(_BuiltinSchedule):
    """Multiply by ``gamma`` every ``step_size`` epochs (0 = a third of the run)."""
    _symbol = "spingalett_lr_step_decay"

    def __init__(self, step_size: int = 0, gamma: float = 0.1):
        super().__init__(step_size=step_size, gamma=gamma)


class WarmupCosine(_BuiltinSchedule):
    """Linear warmup followed by cosine decay to ``min_lr``."""
    _symbol = "spingalett_lr_warmup_cosine"

    def __init__(self, warmup_epochs: int = 0, min_lr: float = 0.0):
        super().__init__(warmup_epochs=warmup_epochs, min_lr=min_lr)


Schedule = Union[_BuiltinSchedule, Callable[[int, int, float], float]]


# --------------------------------------------------------------------------- high-level API

@dataclasses.dataclass(frozen=True)
class _NativeGenerator:
    fn: object      # a _DataGeneratorFn pointing into the library
    data: object

# Layer specifications for Network(layers=[...]). ``inputs`` names the earlier layers a layer reads,
# by index (0 is the input layer, negative indices count back from the layer itself, -1 being the
# one before it); None reads the layer before it.
_Inputs = Optional[Union[int, Sequence[int]]]


@dataclasses.dataclass
class Layer:
    """Dense layer description. Activation, init and dropout are ignored for the input layer."""
    neurons: int
    activation: Activation = Activation.SIGMOID
    init: Init = Init.RANDOM
    dropout: float = 0.0
    inputs: _Inputs = None


@dataclasses.dataclass(frozen=True)
class Input:
    """Input layer holding samples of shape (height, width, channels), channels last."""
    height: int
    width: int
    channels: int = 1


@dataclasses.dataclass(frozen=True)
class Conv2D:
    """2D convolution with ``filters`` output channels over ``kernel`` x ``kernel`` windows; with
    ``groups``, each filter sees the input channels of its group only (input channels: depthwise)."""
    filters: int
    kernel: int
    stride: int = 1
    padding: int = 0
    activation: Activation = Activation.RELU
    init: Init = Init.HE
    dropout: float = 0.0
    groups: int = 1
    inputs: _Inputs = None


@dataclasses.dataclass(frozen=True)
class BatchNorm:
    """Batch normalization of the previous layer, per channel, followed by ``activation``."""
    activation: Activation = Activation.NONE
    epsilon: float = 1e-5
    momentum: float = 0.1
    dropout: float = 0.0
    inputs: _Inputs = None


@dataclasses.dataclass(frozen=True)
class MaxPool2D:
    """Maximum over ``kernel`` x ``kernel`` windows (stride: the kernel size by default)."""
    kernel: int
    stride: int = 0
    padding: int = 0
    inputs: _Inputs = None


@dataclasses.dataclass(frozen=True)
class AvgPool2D:
    """Mean over ``kernel`` x ``kernel`` windows (padded cells not counted)."""
    kernel: int
    stride: int = 0
    padding: int = 0
    inputs: _Inputs = None


@dataclasses.dataclass(frozen=True)
class Add:
    """The sum of earlier layers of one shape (a residual connection), then ``activation``."""
    inputs: _Inputs
    activation: Activation = Activation.NONE
    dropout: float = 0.0


@dataclasses.dataclass(frozen=True)
class Concat:
    """Earlier layers of one height and width side by side along the channels, then ``activation``."""
    inputs: _Inputs
    activation: Activation = Activation.NONE
    dropout: float = 0.0


@dataclasses.dataclass(frozen=True)
class GlobalAvgPool:
    """The mean of each channel over all cells: (1, 1, channels), without activation."""
    dropout: float = 0.0
    inputs: _Inputs = None


@dataclasses.dataclass(frozen=True)
class LayerDescription:
    """Layer ``index`` of a :class:`Network` (0 is the input layer)."""
    type: LayerType
    shape: Tuple[int, int, int]     # (height, width, channels); a dense layer is (1, 1, outputs)
    outputs: int
    activation: Activation
    dropout: float
    kernel: Tuple[int, int]
    stride: Tuple[int, int]
    padding: Tuple[int, int]
    weight_count: int
    bias_count: int
    groups: int = 0                 # convolutions: channel groups
    epsilon: float = 0.0            # batch normalization
    momentum: float = 0.0
    inputs: Tuple[int, ...] = ()    # the layers it reads (none for the input layer)


@dataclasses.dataclass(frozen=True)
class Metrics:
    """Mean loss and accuracy over a data set (see :meth:`Network.evaluate`)."""
    loss: float
    accuracy: float


@dataclasses.dataclass(frozen=True)
class Progress:
    """State of a training run, passed to ``TrainConfig.callback``."""
    epoch: int                      # epochs completed (1-based)
    epochs: int
    train_loss: float
    learning_rate: float
    validation: Optional[Metrics]   # after this epoch, when validation data was given
    monitor: Monitor
    best_epoch: int
    best_value: float
    improved: bool


@dataclasses.dataclass(frozen=True)
class TrainResult:
    """Outcome of :meth:`Network.train`."""
    status: TrainStatus
    epochs_run: int
    train_loss: float
    validation: Optional[Metrics]   # metrics of the last epoch
    monitor: Monitor
    best_epoch: int                 # 0 when nothing was monitored
    best_value: float
    restored_best: bool


def _metrics(m: _EvalMetrics) -> Metrics:
    return Metrics(float(m.loss), float(m.accuracy))


@dataclasses.dataclass
class TrainConfig:
    """Training parameters (see TrainArgs in Spingalett.h). Zero means "library default"."""
    epochs: int = 1
    strategy: Strategy = Strategy.FULL_BATCH
    optimizer: Optimizer = Optimizer.ADAM
    learning_rate: float = 0.01
    batch_size: int = 0
    shuffle: bool = True            # reshuffle every epoch (per-sample and mini-batch, array data)
    weight_decay: float = 0.0
    momentum: float = 0.0
    beta1: float = 0.0
    beta2: float = 0.0
    epsilon: float = 0.0
    max_grad_norm: float = 0.0
    reset_optimizer: bool = False
    nan_check_interval: int = 0
    report_interval: int = 0
    lr_scheduler: Optional[Schedule] = None
    callback: Optional[Callable[["Network", Progress], Optional[bool]]] = None
    callback_interval: int = 0
    monitor: Monitor = Monitor.AUTO
    early_stopping_patience: int = 0    # stop after this many epochs without improvement
    early_stopping_min_delta: float = 0.0
    restore_best_weights: bool = False  # end with the parameters of the best epoch
    autosave_mode: AutoSave = AutoSave.OFF
    autosave_interval: int = 0
    autosave_path: Optional[str] = None
    autosave_save_optimizer: bool = True
    autosave_precision: Precision = Precision.FLOAT32
    blas_num_threads: int = 0
    augment_shift: int = 0              # images: random shifts by up to this many cells (zero fill)
    augment_flip: bool = False          # images: mirror left to right half of the time
    label_smoothing: float = 0.0        # targets moved this far towards uniform (0.1 is common)
    lr_plateau_factor: float = 0.0      # reduce on plateau: multiply the learning rate by this...
    lr_plateau_patience: int = 0        # ...after this many epochs without improvement (0: off)
    lr_plateau_min_lr: float = 0.0      # ...but not below this


def _as_float(data) -> np.ndarray:
    """float32 values; uint8 arrays are image bytes, value q read as q / 255 (as the C readers and
    the U8_UNIT encoding read them)."""
    arr = np.asarray(data)
    if arr.dtype == np.uint8:
        return (arr.astype(np.float64) / 255.0).astype(np.float32)
    return np.ascontiguousarray(arr, dtype=np.float32)


def _as_matrix(data, width: int, name: str, images: bool = True) -> np.ndarray:
    """Rows of `width` float32 values; uint8 inputs are image bytes (see _as_float), while targets
    and gradients (images=False) keep their values."""
    arr = np.ascontiguousarray(_as_float(data) if images else np.asarray(data, dtype=np.float32))
    if arr.ndim == 1:
        if arr.size % width != 0:
            raise ValueError(f"{name}: {arr.size} values do not split into rows of {width}")
        arr = arr.reshape(-1, width)
    if arr.ndim != 2 or arr.shape[1] != width:
        raise ValueError(f"{name}: expected shape (n, {width}), got {arr.shape}")
    return arr


def _float_ptr(arr: np.ndarray):
    return arr.ctypes.data_as(POINTER(c_float))


class Network:
    """A Spingalett network. Use as a context manager or call :meth:`close` to free it."""

    def __init__(self, loss: Loss = Loss.MSE, layers: Optional[Iterable[Union[Layer, int]]] = None):
        self._ptr = _call(_new, _NeuralNetworkArgs(int(loss)))
        if not self._ptr:
            raise SpingalettError(-1, "network allocation failed")
        for spec in layers or ():
            if isinstance(spec, Layer):
                self.add_layer(spec.neurons, spec.activation, spec.init, spec.dropout, inputs=spec.inputs)
            elif isinstance(spec, Input):
                self.add_input(spec.height, spec.width, spec.channels)
            elif isinstance(spec, Conv2D):
                self.add_conv2d(spec.filters, spec.kernel, spec.stride, spec.padding, spec.activation, spec.init,
                                spec.dropout, spec.groups, inputs=spec.inputs)
            elif isinstance(spec, BatchNorm):
                self.add_batch_norm(spec.activation, spec.epsilon, spec.momentum, spec.dropout, inputs=spec.inputs)
            elif isinstance(spec, MaxPool2D):
                self.add_max_pool2d(spec.kernel, spec.stride, spec.padding, inputs=spec.inputs)
            elif isinstance(spec, AvgPool2D):
                self.add_avg_pool2d(spec.kernel, spec.stride, spec.padding, inputs=spec.inputs)
            elif isinstance(spec, Add):
                self.add_add(spec.inputs, spec.activation, spec.dropout)
            elif isinstance(spec, Concat):
                self.add_concat(spec.inputs, spec.activation, spec.dropout)
            elif isinstance(spec, GlobalAvgPool):
                self.add_global_avg_pool(spec.dropout, inputs=spec.inputs)
            else:
                self.add_layer(int(spec))

    @classmethod
    def _wrap(cls, ptr) -> "Network":
        net = cls.__new__(cls)
        net._ptr = ptr
        return net

    @classmethod
    def load(cls, path) -> "Network":
        """Load a network saved with :meth:`save` (or the C API)."""
        ptr = _call(_load, _encode_path(path))
        if not ptr:
            raise SpingalettError(-1, f"could not load {path!s}")
        return cls._wrap(ptr)

    # ---- lifetime
    def close(self) -> None:
        ptr = getattr(self, "_ptr", None)
        if ptr:
            self._ptr = None
            _free(ptr)

    def __enter__(self) -> "Network":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:  # interpreter shutdown may have torn down ctypes already
            pass

    @property
    def _net(self):
        """The C handle; raises once the network is closed."""
        if not getattr(self, "_ptr", None):
            raise ValueError("network is closed")
        return self._ptr

    # ---- structure
    def _add(self, inputs: _Inputs = None, **fields) -> "Network":
        args = _LayerArgs(net=self._net, **fields)
        if inputs is not None:
            count = _layer_count(self._net)
            listed = [inputs] if isinstance(inputs, (int, np.integer)) else list(inputs)
            if not 1 <= len(listed) <= MAX_INPUTS:
                raise ValueError(f"a layer reads 1 to {MAX_INPUTS} layers, not {len(listed)}")
            for k, index in enumerate(listed):
                index = int(index)
                if not -count <= index < count:
                    raise IndexError(f"input layer {index} out of range for {count} layers")
                args.inputs[k] = index % count
            args.input_count = len(listed)
        _call(_layer, args)
        return self

    def __len__(self) -> int:
        return int(_layer_count(self._net))

    @property
    def last(self) -> int:
        """Index of the layer added last (the output layer so far): what later layers name in
        ``inputs`` to read it."""
        return len(self) - 1

    def add_layer(self, neurons: int, activation: Activation = Activation.SIGMOID,
                  init: Init = Init.RANDOM, dropout: float = 0.0, inputs: _Inputs = None) -> "Network":
        """Append a dense layer; the first layer added is the input layer. Every layer reads the one
        before it unless ``inputs`` names another (an index, negative ones counting back from the new
        layer: -1 is the layer before it)."""
        return self._add(inputs, neurons_amount=int(neurons), act_func=int(activation), weight_initialization=int(init),
                         dropout_rate=float(dropout))

    def add_input(self, height: int, width: int, channels: int = 1) -> "Network":
        """Make the input layer take samples of shape (height, width, channels), channels last."""
        return self._add(height=int(height), width=int(width), channels=int(channels))

    def add_conv2d(self, filters: int, kernel: int, stride: int = 1, padding: int = 0,
                   activation: Activation = Activation.RELU, init: Init = Init.HE, dropout: float = 0.0,
                   groups: int = 1, inputs: _Inputs = None) -> "Network":
        """Append a 2D convolution: ``filters`` output channels, ``kernel`` x ``kernel`` windows,
        ``padding`` zeros on each side (kernel // 2 keeps the size of odd kernels at stride 1). With
        ``groups``, input channels and filters split into that many groups and each filter sees the
        input channels of its own (groups = input channels: a depthwise convolution)."""
        return self._add(inputs, type=int(LayerType.CONV2D), filters=int(filters), kernel=int(kernel),
                         stride=int(stride), padding=int(padding), act_func=int(activation),
                         weight_initialization=int(init), dropout_rate=float(dropout), groups=int(groups))

    def add_batch_norm(self, activation: Activation = Activation.NONE, epsilon: float = 1e-5, momentum: float = 0.1,
                       dropout: float = 0.0, inputs: _Inputs = None) -> "Network":
        """Append batch normalization of the previous layer, per channel, then ``activation``: while
        training with the batch's mean and variance, otherwise with running averages of them (each
        batch moves them ``momentum`` of the way)."""
        return self._add(inputs, type=int(LayerType.BATCH_NORM), act_func=int(activation), epsilon=float(epsilon),
                         momentum=float(momentum), dropout_rate=float(dropout))

    def add_max_pool2d(self, kernel: int, stride: int = 0, padding: int = 0, inputs: _Inputs = None) -> "Network":
        """Append max pooling over ``kernel`` x ``kernel`` windows (stride 0 = the kernel size)."""
        return self._add(inputs, type=int(LayerType.MAX_POOL2D), kernel=int(kernel), stride=int(stride),
                         padding=int(padding))

    def add_avg_pool2d(self, kernel: int, stride: int = 0, padding: int = 0, inputs: _Inputs = None) -> "Network":
        """Append average pooling over ``kernel`` x ``kernel`` windows (stride 0 = the kernel size)."""
        return self._add(inputs, type=int(LayerType.AVG_POOL2D), kernel=int(kernel), stride=int(stride),
                         padding=int(padding))

    def add_add(self, inputs: _Inputs, activation: Activation = Activation.NONE, dropout: float = 0.0) -> "Network":
        """Append the sum of earlier layers of one shape, then ``activation``: with the input of a
        block and its last layer, a residual connection."""
        return self._add(inputs, type=int(LayerType.ADD), act_func=int(activation), dropout_rate=float(dropout))

    def add_concat(self, inputs: _Inputs, activation: Activation = Activation.NONE, dropout: float = 0.0) -> "Network":
        """Append earlier layers of one height and width side by side along the channels, in the
        order given, then ``activation``."""
        return self._add(inputs, type=int(LayerType.CONCAT), act_func=int(activation), dropout_rate=float(dropout))

    def add_global_avg_pool(self, dropout: float = 0.0, inputs: _Inputs = None) -> "Network":
        """Append the mean of each channel over all cells: shape (1, 1, channels), a pooling layer
        without activation."""
        return self._add(inputs, type=int(LayerType.GLOBAL_AVG_POOL), dropout_rate=float(dropout))

    def layer(self, index: int) -> LayerDescription:
        """Layer ``index`` (0 is the input layer)."""
        info = _NetworkLayer()
        count = _layer_count(self._net)
        if not -count <= index < count:
            raise IndexError(f"layer index {index} out of range for {count} layers")
        _network_layer(self._ptr, index % count, ctypes.byref(info))
        return LayerDescription(LayerType(info.type), (info.height, info.width, info.channels), int(info.outputs),
                                Activation(info.activation), float(info.dropout_rate), (info.kernel_h, info.kernel_w),
                                (info.stride_h, info.stride_w), (info.padding_h, info.padding_w),
                                int(info.weight_count), int(info.bias_count), int(info.groups), float(info.epsilon),
                                float(info.momentum), tuple(int(info.inputs[k]) for k in range(info.input_count)))

    @property
    def layers(self) -> List[LayerDescription]:
        return [self.layer(i) for i in range(_layer_count(self._net))]

    @property
    def topology(self) -> List[int]:
        """Outputs of every layer (height x width x channels)."""
        return [layer.outputs for layer in self.layers]

    @property
    def activations(self) -> List[Activation]:
        return [layer.activation for layer in self.layers[1:]]

    @property
    def dropout_rates(self) -> List[float]:
        return [layer.dropout for layer in self.layers]

    @property
    def loss(self) -> Loss:
        return Loss(_network_loss(self._net))

    @property
    def input_size(self) -> int:
        return int(_input_size(self._net))

    @property
    def output_size(self) -> int:
        return int(_output_size(self._net))

    @property
    def num_parameters(self) -> int:
        return int(_parameter_count(self._net))

    @property
    def time_step(self) -> int:
        """Optimizer steps taken so far (drives Adam's bias correction)."""
        return int(_optimizer_steps(self._net))

    def _connection(self, index: int):
        """(layer fed by weight set ``index``, its description, weight shape)."""
        count = _layer_count(self._net) - 1
        if not -count <= index < count:
            raise IndexError(f"connection index {index} out of range for {count} weight sets")
        i = index % count + 1
        info = self.layer(i)
        if info.type == LayerType.CONV2D:
            shape = (info.shape[2], info.kernel[0], info.kernel[1], self.layer(info.inputs[0]).shape[2] // info.groups)
        elif info.type == LayerType.BATCH_NORM:
            shape = (info.bias_count,)
        else:
            shape = (info.bias_count, info.weight_count // info.bias_count if info.bias_count else 0)
        return i, info, shape

    def _get(self, index: int, kind: int, weights: bool) -> np.ndarray:
        i, info, shape = self._connection(index)
        out = np.empty(shape if weights else (info.bias_count,), dtype=np.float32)
        if out.size:
            _call(_get_parameters, self._ptr, i, kind, _float_ptr(out), out.size)
        return out

    def _set(self, index: int, kind: int, weights: bool, values) -> None:
        i, info, shape = self._connection(index)
        arr = np.ascontiguousarray(values, dtype=np.float32).reshape(shape if weights else (info.bias_count,))
        if arr.size:
            _call(_set_parameters, self._ptr, i, kind, _float_ptr(arr), arr.size)

    def get_weights(self, index: int) -> np.ndarray:
        """Copy of the weights feeding layer ``index + 1``: shape (outputs, inputs) for a dense layer,
        (filters, kernel_h, kernel_w, input channels / groups) for a convolution, gamma (channels,)
        for batch normalization, empty for pooling."""
        return self._get(index, _PARAM_WEIGHTS, True)

    def set_weights(self, index: int, values) -> None:
        self._set(index, _PARAM_WEIGHTS, True, values)

    def get_biases(self, index: int) -> np.ndarray:
        """Copy of the biases of layer ``index + 1`` (one per output or filter)."""
        return self._get(index, _PARAM_BIASES, False)

    def set_biases(self, index: int, values) -> None:
        self._set(index, _PARAM_BIASES, False, values)

    def get_running_statistics(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        """(mean, variance) per channel that batch normalization layer ``index + 1`` uses outside
        training."""
        return self._get(index, _PARAM_RUNNING_MEAN, False), self._get(index, _PARAM_RUNNING_VARIANCE, False)

    def set_running_statistics(self, index: int, mean, variance) -> None:
        self._set(index, _PARAM_RUNNING_MEAN, False, mean)
        self._set(index, _PARAM_RUNNING_VARIANCE, False, variance)

    # ---- inference
    def forward(self, inputs) -> np.ndarray:
        """Run inference. A 1-D input returns one output vector; a 2-D batch returns one row per
        sample and runs as a single batched call (matrix-matrix products on every backend)."""
        n_in, n_out = self.input_size, self.output_size
        arr = _as_float(inputs)
        single = arr.ndim == 1 and arr.size == n_in
        batch = _as_matrix(arr, n_in, "inputs")
        out = np.empty((batch.shape[0], n_out), dtype=np.float32)
        if single:
            ptr = _call(_forward, _ForwardArgs(self._ptr, batch.ctypes.data))
            ctypes.memmove(out.ctypes.data, ptr, n_out * 4)
            return out[0]
        if batch.shape[0]:
            self._net  # raise if closed
            _call(_predict, _PredictArgs(self._ptr, batch.ctypes.data, batch.shape[0], out.ctypes.data))
        return out

    __call__ = forward

    def evaluate(self, inputs, targets) -> Metrics:
        """Mean loss (as reported by training) and accuracy over a data set. Accuracy compares
        the argmax of outputs and targets; with a single output, their side of 0.5."""
        x, y = self._pair(inputs, targets, "")
        return _metrics(_call(_evaluate, _EvaluateArgs(self._ptr, x.ctypes.data, y.ctypes.data, x.shape[0])))

    def _pair(self, inputs, targets, what: str):
        x = _as_matrix(inputs, self.input_size, what + "inputs")
        y = _as_matrix(targets, self.output_size, what + "targets", images=False)
        if x.shape[0] != y.shape[0]:
            raise ValueError(f"{what}inputs have {x.shape[0]} rows but {what}targets have {y.shape[0]}")
        if x.shape[0] == 0:
            raise ValueError(f"no {what or 'training '}samples")
        return x, y

    def get_weight_gradients(self, index: int) -> np.ndarray:
        """Copy of the accumulated gradient of weight set ``index`` (see :class:`Trainer`)."""
        return self._get(index, _PARAM_WEIGHT_GRADIENTS, True)

    def get_bias_gradients(self, index: int) -> np.ndarray:
        return self._get(index, _PARAM_BIAS_GRADIENTS, False)

    # ---- training
    def train(self, inputs, targets, config: Optional[TrainConfig] = None, validation_data=None,
              **overrides) -> TrainResult:
        """Train on ``inputs`` (n, input_size) and ``targets`` (n, output_size).

        Parameters come from ``config`` (a :class:`TrainConfig`) and/or keyword overrides with the
        same names, e.g. ``net.train(x, y, epochs=100, optimizer=Optimizer.ADAMW)``.
        ``validation_data=(x_val, y_val)`` is evaluated after every epoch and drives
        ``early_stopping_patience`` and ``restore_best_weights``.
        ``callback(network, progress)`` receives a :class:`Progress` and may return True to stop;
        ``lr_scheduler`` is a built-in schedule or ``fn(epoch, total_epochs, initial_lr) -> lr``.
        """
        cfg = dataclasses.replace(config or TrainConfig(), **overrides)
        raw = np.asarray(inputs)
        if raw.dtype == np.uint8:
            # image bytes stay bytes: a reader converts a batch at a time (a quarter of the memory)
            xb = np.ascontiguousarray(raw).reshape(-1, self.input_size) if raw.size % self.input_size == 0 else None
            y = _as_matrix(targets, self.output_size, "targets", images=False)
            if xb is None or xb.shape[0] != y.shape[0] or xb.shape[0] == 0:
                raise ValueError(f"inputs {raw.shape} and targets {y.shape} do not match the network")
            reader = _call(_dataset_open_u8, xb.ctypes.data_as(POINTER(ctypes.c_uint8)), _float_ptr(y),
                           xb.shape[0], self.input_size, self.output_size, bool(cfg.shuffle))
            try:
                return self._run(cfg, _MODE_GENERATOR, None, None, int(xb.shape[0]),
                                 _NativeGenerator(_dataset_generator, c_void_p(reader)), validation_data)
            finally:
                _dataset_close(reader)
        x, y = self._pair(inputs, targets, "")
        return self._run(cfg, _MODE_ARRAY, x, y, x.shape[0], None, validation_data)

    def train_from_generator(self, generator: Callable[[np.ndarray, np.ndarray], int],
                             config: Optional[TrainConfig] = None, samples_per_epoch: int = 0,
                             validation_data=None, **overrides) -> TrainResult:
        """Train on data produced on demand.

        ``generator(inputs, targets)`` receives writable float32 arrays of shape
        (requested, input_size) and (requested, output_size), fills the first rows and returns how
        many it wrote; returning 0 ends the epoch. It is called once per mini-batch, once per epoch
        for full batch (``samples_per_epoch`` rows requested; required), and in chunks for
        per-sample training. ``samples_per_epoch`` > 0 also caps the epoch length.
        """
        cfg = dataclasses.replace(config or TrainConfig(), **overrides)
        if cfg.strategy == Strategy.FULL_BATCH and samples_per_epoch <= 0:
            raise ValueError("full-batch training from a generator needs samples_per_epoch")
        return self._run(cfg, _MODE_GENERATOR, None, None, int(samples_per_epoch), generator, validation_data)

    def train_from_file(self, path, config: Optional[TrainConfig] = None, shuffle: bool = True,
                        validation_data=None, in_memory: bool = False, prefetch: bool = True,
                        target_set: int = 0, **overrides) -> TrainResult:
        """Train on a .slettd data set file through the C reader (no Python call per batch).
        By default it streams the file with memory for a few chunks, decoding the next chunks on a
        background thread when a processor is free for it, or several at a time on the OpenMP
        threads otherwise (``prefetch=False``: never on a background thread); ``shuffle`` reorders
        chunks and samples each epoch. ``in_memory`` decodes the file once and keeps its values in
        their compact form (a byte per 8-bit value), shuffling all samples each epoch.
        ``target_set`` picks the set of targets of files that hold several."""
        cfg = dataclasses.replace(config or TrainConfig(), **overrides)
        if int(target_set) < 0:
            raise ValueError(f"target_set must be 0 or more, got {target_set}")
        opts = _DatasetReaderOptions(bool(shuffle), bool(in_memory), not prefetch, int(target_set))
        reader = _call(_dataset_open, _encode_path(path), ctypes.byref(opts))
        try:
            info = _dataset_info(reader)
            if (info.input_size, info.target_size) != (self.input_size, self.output_size):
                raise ValueError(f"{path!s} has {info.input_size} inputs and {info.target_size} targets, "
                                 f"the network {self.input_size} and {self.output_size}")
            return self._run(cfg, _MODE_GENERATOR, None, None, int(info.count),
                             _NativeGenerator(_dataset_generator, c_void_p(reader)), validation_data)
        finally:
            _dataset_close(reader)

    def _run(self, cfg: TrainConfig, mode: int, x, y, sample_count: int, generator,
             validation_data) -> TrainResult:
        pending: List[BaseException] = []
        keep = []  # C callback objects must outlive the call

        c_callback = _TrainCallbackFn()
        if cfg.callback is not None:
            user_cb = cfg.callback

            def on_epoch(_net_ptr, progress_ptr, _ud):
                if pending:
                    return True
                try:
                    p = progress_ptr.contents
                    progress = Progress(int(p.epoch), int(p.epochs), float(p.train_loss), float(p.learning_rate),
                                        _metrics(p.validation) if p.has_validation else None, Monitor(p.monitor),
                                        int(p.best_epoch), float(p.best_value), bool(p.improved))
                    return bool(user_cb(self, progress))
                except BaseException as exc:  # re-raised once train() returns
                    pending.append(exc)
                    return True

            c_callback = _TrainCallbackFn(on_epoch)
            keep.append(c_callback)

        c_sched, c_sched_data = _LRSchedulerFn(), None
        if isinstance(cfg.lr_scheduler, _BuiltinSchedule):
            c_sched, c_sched_data = cfg.lr_scheduler._as_c()
            keep.append(cfg.lr_scheduler)
        elif cfg.lr_scheduler is not None:
            user_sched = cfg.lr_scheduler

            def schedule(epoch, total, initial_lr, _ud):
                if pending:
                    return -1.0
                try:
                    return float(user_sched(int(epoch), int(total), float(initial_lr)))
                except BaseException as exc:
                    pending.append(exc)
                    return -1.0

            c_sched = _LRSchedulerFn(schedule)
            keep.append(c_sched)

        c_gen, c_gen_data = _DataGeneratorFn(), None
        if isinstance(generator, _NativeGenerator):
            c_gen, c_gen_data = generator.fn, generator.data
        elif generator is not None:
            n_in, n_out = self.input_size, self.output_size

            def produce(in_ptr, tg_ptr, requested, _ud):
                if pending:
                    return 0
                try:
                    xs = np.ctypeslib.as_array(in_ptr, shape=(requested, n_in))
                    ys = np.ctypeslib.as_array(tg_ptr, shape=(requested, n_out))
                    count = int(generator(xs, ys))
                    if not 0 <= count <= requested:
                        raise ValueError(f"generator returned {count} for a request of {requested}")
                    return count
                except BaseException as exc:
                    pending.append(exc)
                    return 0

            c_gen = _DataGeneratorFn(produce)
            keep.append(c_gen)

        autosave_path = _encode_path(cfg.autosave_path) if cfg.autosave_path else None
        xv = yv = None
        if validation_data is not None:
            xv, yv = self._pair(validation_data[0], validation_data[1], "validation ")

        args = _TrainArgs(
            net=self._ptr,
            training_mode=mode,
            training_strategy=int(cfg.strategy),
            optimizer_type=int(cfg.optimizer),
            inputs=_float_ptr(x) if x is not None else None,
            targets=_float_ptr(y) if y is not None else None,
            generator=c_gen,
            generator_data=c_gen_data,
            sample_count=sample_count,
            batch_size=int(cfg.batch_size),
            do_not_shuffle=not cfg.shuffle,
            epochs=int(cfg.epochs),
            learning_rate=float(cfg.learning_rate),
            weight_decay=float(cfg.weight_decay),
            momentum=float(cfg.momentum),
            beta1=float(cfg.beta1),
            beta2=float(cfg.beta2),
            epsilon=float(cfg.epsilon),
            max_grad_norm=float(cfg.max_grad_norm),
            reset_optimizer=bool(cfg.reset_optimizer),
            nan_check_interval=int(cfg.nan_check_interval),
            report_interval=int(cfg.report_interval),
            autosave_mode=int(cfg.autosave_mode),
            autosave_interval=int(cfg.autosave_interval),
            autosave_path=autosave_path,
            autosave_do_not_save_optimizer=not cfg.autosave_save_optimizer,
            autosave_precision=int(cfg.autosave_precision),
            callback=c_callback,
            callback_interval=int(cfg.callback_interval),
            lr_scheduler=c_sched,
            lr_scheduler_data=c_sched_data,
            val_inputs=_float_ptr(xv) if xv is not None else None,
            val_targets=_float_ptr(yv) if yv is not None else None,
            val_count=xv.shape[0] if xv is not None else 0,
            monitor=int(cfg.monitor),
            early_stopping_patience=int(cfg.early_stopping_patience),
            early_stopping_min_delta=float(cfg.early_stopping_min_delta),
            restore_best_weights=bool(cfg.restore_best_weights),
            blas_num_threads=int(cfg.blas_num_threads),
            augment_shift=int(cfg.augment_shift),
            augment_flip=bool(cfg.augment_flip),
            label_smoothing=float(cfg.label_smoothing),
            lr_plateau_factor=float(cfg.lr_plateau_factor),
            lr_plateau_patience=int(cfg.lr_plateau_patience),
            lr_plateau_min_lr=float(cfg.lr_plateau_min_lr),
        )
        r = _call(_train, args)
        del keep
        if pending:
            raise pending[0]
        return TrainResult(TrainStatus(r.status), int(r.epochs_run), float(r.train_loss),
                           _metrics(r.validation) if r.has_validation else None, Monitor(r.monitor),
                           int(r.best_epoch), float(r.best_value), bool(r.restored_best))

    # ---- persistence
    def save(self, path, precision: Precision = Precision.FLOAT32, save_optimizer: bool = True) -> None:
        """Save to ``path`` (".slett" is appended when there is no extension)."""
        _call(_save, _SaveArgs(self._ptr, _encode_path(path), not save_optimizer, int(precision)))

    def to_bytes(self, precision: Precision = Precision.FLOAT32, save_optimizer: bool = False) -> bytes:
        """The .slett file :meth:`save` would write, as bytes."""
        size = c_size_t()
        self._net  # raise if closed
        ptr = _call(_save_to_memory, self._ptr, int(precision), bool(save_optimizer), ctypes.byref(size))
        try:
            return ctypes.string_at(ptr, size.value)
        finally:
            _free_memory(ptr)

    @classmethod
    def from_bytes(cls, data) -> "Network":
        """A network from the bytes of a .slett file of any format version."""
        data = bytes(data)
        ptr = _call(_load_from_memory, data, len(data))
        if not ptr:
            raise SpingalettError(-1, "could not load the network")
        return cls._wrap(ptr)

    @classmethod
    def from_onnx(cls, source) -> "Network":
        """A network from an ONNX model: a path, or the model's bytes. It takes channels-last
        samples: an ONNX input [N, C, H, W] becomes an input layer of shape (H, W, C), so NCHW images
        must be transposed (``images.transpose(0, 2, 3, 1)``); see spingalett_import_onnx() for the
        operators it reads."""
        if isinstance(source, (bytes, bytearray, memoryview)):
            data = bytes(source)
            ptr = _call(_import_onnx_from_memory, data, len(data))
        else:
            ptr = _call(_import_onnx, _encode_path(source))
        if not ptr:
            raise SpingalettError(-1, "could not import the ONNX model")
        return cls._wrap(ptr)

    @classmethod
    def from_torch(cls, module, example_input) -> "Network":
        """A network computing what a PyTorch module computes, through an ONNX export in memory
        (``torch.onnx.export``; ``example_input`` is a tensor of the module's input shape, NCHW for
        images). The module is exported in evaluation mode. Like :meth:`from_onnx`, the network takes
        channels-last samples."""
        import io
        import torch
        was_training = module.training
        module.eval()
        try:
            buffer = io.BytesIO()
            try:
                torch.onnx.export(module, (example_input,), buffer, dynamo=False, do_constant_folding=True)
            except TypeError:       # releases without the dynamo argument
                torch.onnx.export(module, (example_input,), buffer, do_constant_folding=True)
        finally:
            module.train(was_training)
        return cls.from_onnx(buffer.getvalue())

    def load_pytorch(self, source, modules: Optional[Sequence[str]] = None) -> "Network":
        """Copies PyTorch weights into this network, which must have the same layers: a state dict
        (``module.state_dict()``, of tensors or NumPy arrays), a torch.save file (.pt, .pth) or a
        .safetensors file, as a path or bytes. The tensors of each module go to the layers with
        parameters in order (see spingalett_load_pytorch()); ``modules`` names the modules in that
        order when the source's order differs. Filters and dense weights are reordered for
        channels-last data. The network is unchanged on error."""
        names = None
        if isinstance(source, dict):
            import json, struct
            header, chunks, offset, order = {}, [], 0, []
            for key, value in source.items():
                if key.endswith("num_batches_tracked"):
                    continue
                if hasattr(value, "detach"):            # a torch tensor
                    value = value.detach().cpu().float().numpy()
                array = np.ascontiguousarray(value, dtype="<f4")
                header[key] = {"dtype": "F32", "shape": list(array.shape), "data_offsets": [offset, offset + array.nbytes]}
                chunks.append(array.tobytes())
                offset += array.nbytes
                module = key.rsplit(".", 1)[0] if "." in key else ""
                if module not in order:
                    order.append(module)
            text = json.dumps(header).encode()
            data = struct.pack("<Q", len(text)) + text + b"".join(chunks)
            names = list(modules) if modules is not None else order
            source = data
        elif modules is not None:
            names = list(modules)
        encoded = (c_char_p * max(1, len(names or ())))(*[n.encode() for n in names or ()])
        count = len(names) if names is not None else 0
        array = encoded if names is not None else None
        if isinstance(source, (bytes, bytearray, memoryview)):
            data = bytes(source)
            _call(_load_pytorch_from_memory, self._net, data, len(data), array, count)
        else:
            _call(_load_pytorch, self._net, _encode_path(source), array, count)
        return self

    # ---- deployment
    def to_model(self, precision: Precision = Precision.INT8) -> "Model":
        """A read-only copy for inference that computes in ``precision`` (see :class:`Model`)."""
        self._net
        return Model._wrap(_call(_model_from_network, self._ptr, int(precision)))

    def export_c_header(self, path, name: str, precision: Precision = Precision.INT8) -> None:
        """Write the network in ``precision`` as a C header for firmware: a ``static const uint8_t
        name[]`` holding the model image and NAME_SIZE, NAME_INPUTS, NAME_OUTPUTS and NAME_WORKSPACE
        macros (see Spingalett.Inference.h)."""
        self._net
        _call(_export_c_header, self._ptr, _encode_path(path), name.encode(), int(precision))

    def print_parameters(self) -> None:
        _call(_print_parameters, self._ptr)

    def __repr__(self) -> str:
        if not getattr(self, "_ptr", None):
            return "<spingalett.Network (closed)>"
        return f"<spingalett.Network {self.topology} loss={self.loss.name} params={self.num_parameters}>"


# --------------------------------------------------------------------------- deployment models

@dataclasses.dataclass(frozen=True)
class LayerInfo:
    """One layer of a :class:`Model` (the input layer is not counted)."""
    inputs: int
    outputs: int
    activation: Activation
    precision: Precision
    type: LayerType = LayerType.DENSE
    shape: Tuple[int, int, int] = (1, 1, 0)     # output (height, width, channels)
    input_shape: Tuple[int, int, int] = (1, 1, 0)
    kernel: Tuple[int, int] = (0, 0)
    stride: Tuple[int, int] = (0, 0)
    padding: Tuple[int, int] = (0, 0)
    groups: int = 0
    epsilon: float = 0.0
    input_layers: Tuple[int, ...] = ()  # network layers it reads (0: the input; i + 1: layer i here)


class Model:
    """A read-only network for inference that computes in the precision its weights are stored in:
    INT8, INT4 and INT2 layers quantize their input to 8 bits per sample and use integer kernels,
    FLOAT32, FP16 and BFLOAT16 layers compute in float. Create one with :meth:`Network.to_model`,
    :meth:`Model.load` or :meth:`Model.from_bytes`; :meth:`to_bytes` gives the .slett image, which
    the standalone C engine runs in place."""

    def __init__(self):
        raise TypeError("use Network.to_model, Model.load or Model.from_bytes")

    @classmethod
    def _wrap(cls, ptr) -> "Model":
        if not ptr:
            raise SpingalettError(-1, "could not create the model")
        model = cls.__new__(cls)
        model._ptr = ptr
        model._sizes = (int(ptr.contents.input_size), int(ptr.contents.output_size))   # read-only
        return model

    @classmethod
    def load(cls, path) -> "Model":
        """Read a .slett file; files of older format versions are converted in their precision."""
        return cls._wrap(_call(_model_load, _encode_path(path)))

    @classmethod
    def from_bytes(cls, data) -> "Model":
        """A model from the bytes of a .slett file (copied)."""
        data = bytes(data)
        return cls._wrap(_call(_model_from_memory, data, len(data)))

    # ---- lifetime
    def close(self) -> None:
        ptr = getattr(self, "_ptr", None)
        if ptr:
            self._ptr = None
            _model_free(ptr)

    def __enter__(self) -> "Model":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    @property
    def _model(self) -> _Model:
        if not getattr(self, "_ptr", None):
            raise ValueError("model is closed")
        return self._ptr.contents

    # ---- description
    @property
    def input_size(self) -> int:
        return int(self._model.input_size)

    @property
    def output_size(self) -> int:
        return int(self._model.output_size)

    @property
    def loss(self) -> Loss:
        return Loss(self._model.loss)

    @property
    def size(self) -> int:
        """Bytes of the model image."""
        return int(self._model.image_size)

    @property
    def workspace_size(self) -> int:
        """Bytes of workspace the C engine's spingalett_model_run needs per thread."""
        return int(self._model.workspace_size)

    @property
    def layers(self) -> List[LayerInfo]:
        out = []
        for i in range(self._model.layer_count):
            info = _LayerInfo()
            _model_layer(self._ptr, i, ctypes.byref(info))
            out.append(LayerInfo(int(info.inputs), int(info.outputs), Activation(info.activation),
                                 Precision(info.precision), type=LayerType(info.type),
                                 shape=(int(info.height), int(info.width), int(info.channels)),
                                 input_shape=(int(info.in_height), int(info.in_width), int(info.in_channels)),
                                 kernel=(int(info.kernel_h), int(info.kernel_w)),
                                 stride=(int(info.stride_h), int(info.stride_w)),
                                 padding=(int(info.padding_h), int(info.padding_w)),
                                 groups=int(info.groups), epsilon=float(info.epsilon),
                                 input_layers=tuple(int(info.input_layers[k]) for k in range(info.input_count))))
        return out

    def to_bytes(self) -> bytes:
        """The model image: a .slett file (format version 3; 4 with convolution or pooling layers; 5 with
        batch normalization that could not be folded, or grouped convolutions)."""
        m = self._model
        return ctypes.string_at(m.image, m.image_size)

    # ---- inference
    def predict(self, inputs) -> np.ndarray:
        """Outputs for one sample (1-D input) or a batch (one row per sample)."""
        self._model  # raise if closed
        n_in, n_out = self._sizes
        arr = _as_float(inputs)
        single = arr.ndim == 1 and arr.size == n_in
        batch = _as_matrix(arr, n_in, "inputs")
        out = np.empty((batch.shape[0], n_out), dtype=np.float32)
        if batch.shape[0]:
            _call(_model_predict, self._ptr, batch.ctypes.data, batch.shape[0], out.ctypes.data)
        return out[0] if single else out

    __call__ = predict

    def evaluate(self, inputs, targets) -> Metrics:
        """Mean loss and accuracy over a data set, as :meth:`Network.evaluate` computes them."""
        x = _as_matrix(inputs, self.input_size, "inputs")
        y = _as_matrix(targets, self.output_size, "targets", images=False)
        if x.shape[0] != y.shape[0] or x.shape[0] == 0:
            raise ValueError(f"inputs have {x.shape[0]} rows and targets {y.shape[0]}; need the same, at least 1")
        return _metrics(_call(_model_evaluate, self._ptr, x.ctypes.data, y.ctypes.data, x.shape[0]))

    def __repr__(self) -> str:
        if not getattr(self, "_ptr", None):
            return "<spingalett.Model (closed)>"
        shape = [self.input_size] + [l.outputs for l in self.layers]
        precisions = sorted({l.precision.name for l in self.layers})
        return f"<spingalett.Model {shape} {'/'.join(precisions)} {self.size} bytes>"


# --------------------------------------------------------------------------- low-level training

class Trainer:
    """Forward, backward and optimizer steps under your control: custom losses, gradient
    accumulation, custom loops. Backward passes add the per-sample gradients up; :meth:`step`
    applies their mean and clears them. Optimizer state lives in the network (shared with
    :meth:`Network.train`).

    ::

        with sg.Trainer(net, max_batch=64) as tr:
            out = tr.forward(xb)
            tr.backward_output_grads(dloss_dout(out, yb))   # or tr.backward(yb) for the built-in loss
            tr.step(optimizer=sg.Optimizer.ADAM, learning_rate=1e-3)
    """

    def __init__(self, network: Network, max_batch: int):
        network._net  # raise if closed
        self._network = network          # keeps the network alive
        self._ptr = _call(_trainer_new, network._ptr, int(max_batch))
        self.max_batch = int(max_batch)
        self._rows = 0

    def close(self) -> None:
        ptr = getattr(self, "_ptr", None)
        if ptr:
            self._ptr = None
            _trainer_free(ptr)

    def __enter__(self) -> "Trainer":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _handle(self):
        if not getattr(self, "_ptr", None):
            raise ValueError("trainer is closed")
        self._network._net  # raise if the network was closed
        return self._ptr

    def forward(self, inputs) -> np.ndarray:
        """Training-mode forward pass (dropout active); returns a copy of the outputs."""
        net = self._network
        x = _as_matrix(inputs, net.input_size, "inputs")
        ptr = _call(_trainer_forward, self._handle(), x.ctypes.data, x.shape[0])
        self._rows = x.shape[0]
        out = np.empty((x.shape[0], net.output_size), dtype=np.float32)
        ctypes.memmove(out.ctypes.data, ptr, out.nbytes)
        return out

    def backward(self, targets) -> float:
        """Back-propagates the network's loss for the last forward pass; returns the summed loss."""
        y = _as_matrix(targets, self._network.output_size, "targets", images=False)
        if y.shape[0] != self._rows:
            raise ValueError(f"targets have {y.shape[0]} rows, the last forward pass had {self._rows}")
        return float(_call(_trainer_backward, self._handle(), y.ctypes.data))

    def backward_output_grads(self, output_grads) -> None:
        """Back-propagates a custom loss given dL/d(output) for every sample of the last forward pass."""
        g = _as_matrix(output_grads, self._network.output_size, "output_grads", images=False)
        if g.shape[0] != self._rows:
            raise ValueError(f"output_grads have {g.shape[0]} rows, the last forward pass had {self._rows}")
        _call(_trainer_backward_grads, self._handle(), g.ctypes.data)

    def step(self, optimizer: Optimizer = Optimizer.ADAM, learning_rate: float = 0.01, weight_decay: float = 0.0,
             momentum: float = 0.0, beta1: float = 0.0, beta2: float = 0.0, epsilon: float = 0.0,
             max_grad_norm: float = 0.0) -> None:
        """Optimizer step with the mean accumulated gradient; zero values take the library defaults."""
        opt = _OptimizerArgs(int(optimizer), learning_rate, weight_decay, momentum, beta1, beta2, epsilon, max_grad_norm)
        _call(_trainer_step, self._handle(), ctypes.byref(opt))

    def zero_grad(self) -> None:
        _trainer_zero_grad(self._handle())

    def train_on_batch(self, inputs, targets, **optimizer) -> float:
        """Forward, backward and step on one batch; returns its mean loss."""
        self.forward(inputs)
        loss = self.backward(targets)
        self.step(**optimizer)
        return loss / self._rows


# --------------------------------------------------------------------------- data sets

def _take_dataset(ds: _Dataset):
    try:
        x = np.ctypeslib.as_array(ds.inputs, shape=(ds.count, ds.input_size)).copy()
        y = np.ctypeslib.as_array(ds.targets, shape=(ds.count, ds.target_size)).copy()
    finally:
        _dataset_free(ctypes.byref(ds))
    return x, y


def _names_array(names, count: int, what: str):
    if names is None:
        return None
    names = [str(n).encode() for n in names]
    if len(names) != count:
        raise ValueError(f"{what}: {len(names)} names for {count} targets")
    return (c_char_p * (count + 1))(*names, None)


def load_idx(images_path, labels_path, num_classes: int = 0):
    """Read an IDX pair (the MNIST format) as ``(inputs, one_hot_targets)`` float32 arrays.
    Unsigned-byte images are scaled to [0, 1]; ``num_classes`` 0 = largest label + 1."""
    ds = _Dataset()
    _call(_load_idx, _encode_path(images_path), _encode_path(labels_path), int(num_classes), ctypes.byref(ds))
    return _take_dataset(ds)


def load_cifar(paths, num_classes: int = 10):
    """Read CIFAR binary batches (one path or several) as ``(inputs, one_hot_targets)``: images of
    32 x 32 x 3 values in [0, 1], channels last. ``num_classes`` 10 reads CIFAR-10, 100 the fine and
    20 the coarse labels of CIFAR-100."""
    if isinstance(paths, (str, bytes, os.PathLike)):
        paths = [paths]
    encoded = [_encode_path(p) for p in paths]
    array = (c_char_p * len(encoded))(*encoded)
    ds = _Dataset()
    _call(_load_cifar, array, len(encoded), int(num_classes), ctypes.byref(ds))
    return _take_dataset(ds)


def load_csv(path, target_columns: int = 1, num_classes: int = 0):
    """Read a numeric CSV file as ``(inputs, targets)``: the last ``target_columns`` columns are
    targets; with ``num_classes`` > 0 the single target column is one-hot encoded."""
    ds = _Dataset()
    _call(_load_csv, _encode_path(path), int(target_columns), int(num_classes), ctypes.byref(ds))
    return _take_dataset(ds)


def save_dataset(path, inputs, targets, input_encoding: DatasetEncoding = DatasetEncoding.AUTO,
                 target_encoding: DatasetEncoding = DatasetEncoding.AUTO, compress: bool = True,
                 shape=None, class_names=None, target_name=None, extra_targets=None) -> None:
    """Write ``(inputs, targets)`` to a .slettd file (".slettd" is appended when there is no
    extension). AUTO picks the smallest lossless encoding; FP16, BFLOAT16 and U8_AFFINE are lossy.
    uint8 inputs are image bytes (q / 255) and are stored as such.

    The file can also record the input ``shape`` (height, width, channels), the ``class_names`` of
    the targets, a ``target_name`` for them, and ``extra_targets``: further sets of targets for
    the same samples, each a dict with "targets" and optionally "name", "class_names" and
    "encoding" (load one with ``load_dataset(path, target_set=k)``)."""
    x = _as_float(inputs)
    y = np.ascontiguousarray(targets, dtype=np.float32)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    elif x.ndim > 2:
        if shape is None and x.ndim in (3, 4):
            shape = tuple(x.shape[1:]) + ((1,) if x.ndim == 3 else ())
        x = x.reshape(x.shape[0], -1)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0] or x.shape[0] == 0:
        raise ValueError(f"inputs {x.shape} and targets {y.shape} must be non-empty matrices with the same rows")
    x = np.ascontiguousarray(x)
    ds = _Dataset(x.shape[0], x.shape[1], y.shape[1], _float_ptr(x), _float_ptr(y))
    if shape is not None:
        h, w, c = (tuple(shape) + (1,))[:3] if len(shape) == 2 else tuple(shape)
        if h * w * c != x.shape[1]:
            raise ValueError(f"shape {tuple(shape)} does not hold {x.shape[1]} inputs")
        ds.height, ds.width, ds.channels = int(h), int(w), int(c)
    keep = []
    names = _names_array(class_names, y.shape[1], "class_names")
    if names is not None:
        keep.append(names)
        ds.class_names = ctypes.cast(names, POINTER(c_char_p))
    sets = []
    for k, extra in enumerate(extra_targets or []):
        yk = np.ascontiguousarray(extra["targets"], dtype=np.float32)
        if yk.ndim == 1:
            yk = yk.reshape(-1, 1)
        if yk.ndim != 2 or yk.shape[0] != x.shape[0]:
            raise ValueError(f"extra_targets[{k}]: expected {x.shape[0]} rows, got {yk.shape}")
        nk = _names_array(extra.get("class_names"), yk.shape[1], f"extra_targets[{k}] class_names")
        keep.extend([yk, nk])
        sets.append(_TargetSet(str(extra["name"]).encode() if extra.get("name") is not None else None, yk.shape[1],
                               _float_ptr(yk), ctypes.cast(nk, POINTER(c_char_p)) if nk is not None else None,
                               int(extra.get("encoding", DatasetEncoding.AUTO))))
    set_array = (_TargetSet * len(sets))(*sets) if sets else None
    opts = _DatasetSaveOptions(int(input_encoding), int(target_encoding), not compress,
                               str(target_name).encode() if target_name is not None else None,
                               ctypes.cast(set_array, POINTER(_TargetSet)) if set_array else None, len(sets))
    _call(_save_dataset, ctypes.byref(ds), _encode_path(path), ctypes.byref(opts))
    del keep


def load_dataset(path, target_set: int = 0):
    """Read a .slettd file as ``(inputs, targets)`` float32 arrays; ``target_set`` picks the set of
    targets of files that hold several (see :func:`dataset_info`)."""
    if int(target_set) < 0:
        raise ValueError(f"target_set must be 0 or more, got {target_set}")
    ds = _Dataset()
    _call(_load_dataset, _encode_path(path), int(target_set), ctypes.byref(ds))
    return _take_dataset(ds)


def dataset_info(path) -> dict:
    """What a .slettd file holds: sample count, sizes, encodings, chunks, file size and format
    version, the input "shape" (height, width, channels, or None) and its "target_sets", each a
    dict with "name", "size" and "class_names" (None when the file records none)."""
    opts = _DatasetReaderOptions(False, False, True, 0)
    reader = _call(_dataset_open, _encode_path(path), ctypes.byref(opts))
    try:
        i = _dataset_info(reader)
        sets = []
        for k in range(i.target_set_count):
            size = _dataset_target_set_size(reader, k)
            name = _dataset_target_set_name(reader, k)
            names = [_dataset_class_name(reader, k, c) for c in range(size)]
            sets.append({"name": name.decode() if name is not None else None, "size": int(size),
                         "class_names": [n.decode() for n in names] if all(n is not None for n in names) else None})
        return {"count": i.count, "input_size": i.input_size, "target_size": i.target_size,
                "input_encoding": DatasetEncoding(i.input_encoding), "target_encoding": DatasetEncoding(i.target_encoding),
                "chunk_count": i.chunk_count, "file_size": i.file_size, "format_version": i.format_version,
                "shape": (i.height, i.width, i.channels) if i.height or i.width or i.channels else None,
                "target_sets": sets}
    finally:
        _dataset_close(reader)
