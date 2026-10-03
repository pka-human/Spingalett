# SPDX-License-Identifier: MIT
# Copyright (c) 2026 pka_human (pka_human@proton.me)
"""Python bindings for the Spingalett C23 deep learning engine.

Pure ``ctypes`` over the shared library: nothing is compiled at install time.
The library is located in this order:

1. the ``SPINGALETT_LIBRARY`` environment variable (full path to the library),
2. next to this module, then the repository's ``Bin/`` directory,
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
from typing import Callable, Iterable, List, Optional, Sequence, Union

import numpy as np

__version__ = "0.4.0"

__all__ = [
    "Activation", "Loss", "Init", "Strategy", "Optimizer", "ComputeMode", "Precision",
    "AutoSave", "LogLevel", "ErrorCode", "Monitor", "TrainStatus", "Layer", "TrainConfig", "Network",
    "Metrics", "Progress", "TrainResult", "Trainer", "SpingalettError", "load_idx", "load_csv",
    "CosineDecay", "LinearWarmup", "StepDecay", "WarmupCosine",
    "set_compute_mode", "get_compute_mode", "set_num_threads", "get_num_threads",
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

class _NeuralNetwork(Structure):
    _fields_ = [
        ("layers", c_uint32),
        ("topology", POINTER(c_uint32)),
        ("act_func", POINTER(c_int)),
        ("weights", POINTER(c_float)),
        ("biases", POINTER(c_float)),
        ("neurons", POINTER(c_float)),
        ("grad_weights", POINTER(c_float)),
        ("grad_biases", POINTER(c_float)),
        ("opt_m_weights", POINTER(c_float)),
        ("opt_m_biases", POINTER(c_float)),
        ("opt_v_weights", POINTER(c_float)),
        ("opt_v_biases", POINTER(c_float)),
        ("neuron_offsets", POINTER(c_uint64)),
        ("weight_offsets", POINTER(c_uint64)),
        ("bias_offsets", POINTER(c_uint64)),
        ("total_neurons", c_uint64),
        ("total_weights", c_uint64),
        ("total_biases", c_uint64),
        ("time_step", c_uint64),
        ("loss_func", c_int),
        ("dropout_rates", POINTER(c_float)),
    ]


_NetPtr = POINTER(_NeuralNetwork)


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
    ]


class _ForwardArgs(Structure):
    _fields_ = [("net", _NetPtr), ("input", POINTER(c_float))]


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
    ]


class _PredictArgs(Structure):
    _fields_ = [
        ("net", _NetPtr),
        ("inputs", POINTER(c_float)),
        ("sample_count", c_uint32),
        ("outputs", POINTER(c_float)),
    ]


class _EvaluateArgs(Structure):
    _fields_ = [
        ("net", _NetPtr),
        ("inputs", POINTER(c_float)),
        ("targets", POINTER(c_float)),
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
    ]


class _SaveArgs(Structure):
    _fields_ = [
        ("net", _NetPtr),
        ("filename", c_char_p),
        ("do_not_save_optimizer", c_bool),
        ("precision", c_int),
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
    candidates = [os.path.join(here, n) for n in _library_names()]
    candidates += [os.path.join(here, "..", "..", "Bin", n) for n in _library_names()]
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
_layer = _bind("layer_struct_arguments", None, [_LayerArgs])
_forward = _bind("forward_struct_arguments", POINTER(c_float), [_ForwardArgs])
_predict = _bind("predict_struct_arguments", c_bool, [_PredictArgs])
_train = _bind("train_struct_arguments", _TrainReport, [_TrainArgs])
_evaluate = _bind("evaluate_struct_arguments", _EvalMetrics, [_EvaluateArgs])
_TrainerPtr = c_void_p
_trainer_new = _bind("spingalett_trainer_new", _TrainerPtr, [_NetPtr, c_uint32])
_trainer_free = _bind("spingalett_trainer_free", None, [_TrainerPtr])
_trainer_forward = _bind("spingalett_trainer_forward", POINTER(c_float), [_TrainerPtr, POINTER(c_float), c_uint32])
_trainer_backward = _bind("spingalett_trainer_backward", c_float, [_TrainerPtr, POINTER(c_float)])
_trainer_backward_grads = _bind("spingalett_trainer_backward_output_grads", c_bool, [_TrainerPtr, POINTER(c_float)])
_trainer_step = _bind("spingalett_trainer_step", c_bool, [_TrainerPtr, POINTER(_OptimizerArgs)])
_trainer_zero_grad = _bind("spingalett_trainer_zero_grad", None, [_TrainerPtr])
_load_idx = _bind("spingalett_load_idx", c_bool, [c_char_p, c_char_p, c_uint32, POINTER(_Dataset)])
_load_csv = _bind("spingalett_load_csv", c_bool, [c_char_p, c_uint32, c_uint32, POINTER(_Dataset)])
_dataset_free = _bind("spingalett_dataset_free", None, [POINTER(_Dataset)])
_save = _bind("save_spingalett_struct_arguments", None, [_SaveArgs])
_load = _bind("load_spingalett", _NetPtr, [c_char_p])
_free = _bind("free_network", None, [_NetPtr])
_print_parameters = _bind("print_parameters", None, [_NetPtr])

_last_error_code = _bind("spingalett_last_error_code", c_int, [])
_last_error_message = _bind("spingalett_last_error_message", c_char_p, [])
_clear_error = _bind("spingalett_clear_error", None, [])

_get_compute_mode = _bind("spingalett_get_compute_mode", c_int, [])
_set_compute_mode = _bind("spingalett_set_compute_mode", None, [c_int])
_get_num_threads = _bind("spingalett_get_num_threads", ctypes.c_uint, [])
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

@dataclasses.dataclass
class Layer:
    """Layer description. Activation, init and dropout are ignored for the input layer."""
    neurons: int
    activation: Activation = Activation.SIGMOID
    init: Init = Init.RANDOM
    dropout: float = 0.0


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


def _as_matrix(data, width: int, name: str) -> np.ndarray:
    arr = np.ascontiguousarray(data, dtype=np.float32)
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
                self.add_layer(spec.neurons, spec.activation, spec.init, spec.dropout)
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
    def _net(self) -> _NeuralNetwork:
        if not getattr(self, "_ptr", None):
            raise ValueError("network is closed")
        return self._ptr.contents

    # ---- structure
    def add_layer(self, neurons: int, activation: Activation = Activation.SIGMOID,
                  init: Init = Init.RANDOM, dropout: float = 0.0) -> "Network":
        """Append a layer; the first layer added is the input layer."""
        self._net  # raise if closed
        _call(_layer, _LayerArgs(self._ptr, int(neurons), int(activation), int(init), float(dropout)))
        return self

    @property
    def topology(self) -> List[int]:
        n = self._net
        return [n.topology[i] for i in range(n.layers)]

    @property
    def activations(self) -> List[Activation]:
        n = self._net
        return [Activation(n.act_func[i]) for i in range(max(n.layers - 1, 0))]

    @property
    def dropout_rates(self) -> List[float]:
        n = self._net
        return [n.dropout_rates[i] for i in range(n.layers)]

    @property
    def loss(self) -> Loss:
        return Loss(self._net.loss_func)

    @property
    def input_size(self) -> int:
        return self.topology[0]

    @property
    def output_size(self) -> int:
        return self.topology[-1]

    @property
    def num_parameters(self) -> int:
        n = self._net
        return int(n.total_weights + n.total_biases)

    @property
    def time_step(self) -> int:
        """Optimizer steps taken so far (drives Adam's bias correction)."""
        return int(self._net.time_step)

    def _check_connection(self, index: int) -> int:
        count = self._net.layers - 1
        if not -count <= index < count:
            raise IndexError(f"connection index {index} out of range for {count} weight matrices")
        return index % count

    def get_weights(self, index: int) -> np.ndarray:
        """Copy of weight matrix ``index`` (layer index -> index + 1), shape (out, in)."""
        i = self._check_connection(index)
        n = self._net
        rows, cols = n.topology[i + 1], n.topology[i]
        base = ctypes.addressof(n.weights.contents) + n.weight_offsets[i] * 4
        return np.ctypeslib.as_array((c_float * (rows * cols)).from_address(base)).reshape(rows, cols).copy()

    def set_weights(self, index: int, values) -> None:
        i = self._check_connection(index)
        n = self._net
        rows, cols = n.topology[i + 1], n.topology[i]
        arr = np.ascontiguousarray(values, dtype=np.float32).reshape(rows, cols)
        ctypes.memmove(ctypes.addressof(n.weights.contents) + n.weight_offsets[i] * 4, arr.ctypes.data, arr.nbytes)

    def get_biases(self, index: int) -> np.ndarray:
        """Copy of the biases of layer ``index + 1``."""
        i = self._check_connection(index)
        n = self._net
        size = n.topology[i + 1]
        base = ctypes.addressof(n.biases.contents) + n.bias_offsets[i] * 4
        return np.ctypeslib.as_array((c_float * size).from_address(base)).copy()

    def set_biases(self, index: int, values) -> None:
        i = self._check_connection(index)
        n = self._net
        arr = np.ascontiguousarray(values, dtype=np.float32).reshape(n.topology[i + 1])
        ctypes.memmove(ctypes.addressof(n.biases.contents) + n.bias_offsets[i] * 4, arr.ctypes.data, arr.nbytes)

    # ---- inference
    def forward(self, inputs) -> np.ndarray:
        """Run inference. A 1-D input returns one output vector; a 2-D batch returns one row per
        sample and runs as a single batched call (matrix-matrix products on every backend)."""
        n_in, n_out = self.input_size, self.output_size
        arr = np.ascontiguousarray(inputs, dtype=np.float32)
        single = arr.ndim == 1 and arr.size == n_in
        batch = _as_matrix(arr, n_in, "inputs")
        out = np.empty((batch.shape[0], n_out), dtype=np.float32)
        if single:
            ptr = _call(_forward, _ForwardArgs(self._ptr, _float_ptr(batch[0])))
            ctypes.memmove(out.ctypes.data, ptr, n_out * 4)
            return out[0]
        if batch.shape[0]:
            self._net  # raise if closed
            _call(_predict, _PredictArgs(self._ptr, _float_ptr(batch), batch.shape[0], _float_ptr(out)))
        return out

    __call__ = forward

    def evaluate(self, inputs, targets) -> Metrics:
        """Mean loss (as reported by training) and accuracy over a data set. Accuracy compares
        the argmax of outputs and targets; with a single output, their side of 0.5."""
        x, y = self._pair(inputs, targets, "")
        return _metrics(_call(_evaluate, _EvaluateArgs(self._ptr, _float_ptr(x), _float_ptr(y), x.shape[0])))

    def _pair(self, inputs, targets, what: str):
        x = _as_matrix(inputs, self.input_size, what + "inputs")
        y = _as_matrix(targets, self.output_size, what + "targets")
        if x.shape[0] != y.shape[0]:
            raise ValueError(f"{what}inputs have {x.shape[0]} rows but {what}targets have {y.shape[0]}")
        if x.shape[0] == 0:
            raise ValueError(f"no {what or 'training '}samples")
        return x, y

    def get_weight_gradients(self, index: int) -> np.ndarray:
        """Copy of the accumulated gradient of weight matrix ``index`` (see :class:`Trainer`)."""
        i = self._check_connection(index)
        n = self._net
        rows, cols = n.topology[i + 1], n.topology[i]
        base = ctypes.addressof(n.grad_weights.contents) + n.weight_offsets[i] * 4
        return np.ctypeslib.as_array((c_float * (rows * cols)).from_address(base)).reshape(rows, cols).copy()

    def get_bias_gradients(self, index: int) -> np.ndarray:
        i = self._check_connection(index)
        n = self._net
        base = ctypes.addressof(n.grad_biases.contents) + n.bias_offsets[i] * 4
        return np.ctypeslib.as_array((c_float * n.topology[i + 1]).from_address(base)).copy()

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

        c_gen = _DataGeneratorFn()
        if generator is not None:
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
            generator_data=None,
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
        """Save to ``path`` (".nn" is appended when there is no extension)."""
        _call(_save, _SaveArgs(self._ptr, _encode_path(path), not save_optimizer, int(precision)))

    def print_parameters(self) -> None:
        _call(_print_parameters, self._ptr)

    def __repr__(self) -> str:
        if not getattr(self, "_ptr", None):
            return "<spingalett.Network (closed)>"
        return f"<spingalett.Network {self.topology} loss={self.loss.name} params={self.num_parameters}>"


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
        ptr = _call(_trainer_forward, self._handle(), _float_ptr(x), x.shape[0])
        self._rows = x.shape[0]
        out = np.empty((x.shape[0], net.output_size), dtype=np.float32)
        ctypes.memmove(out.ctypes.data, ptr, out.nbytes)
        return out

    def backward(self, targets) -> float:
        """Back-propagates the network's loss for the last forward pass; returns the summed loss."""
        y = _as_matrix(targets, self._network.output_size, "targets")
        if y.shape[0] != self._rows:
            raise ValueError(f"targets have {y.shape[0]} rows, the last forward pass had {self._rows}")
        return float(_call(_trainer_backward, self._handle(), _float_ptr(y)))

    def backward_output_grads(self, output_grads) -> None:
        """Back-propagates a custom loss given dL/d(output) for every sample of the last forward pass."""
        g = _as_matrix(output_grads, self._network.output_size, "output_grads")
        if g.shape[0] != self._rows:
            raise ValueError(f"output_grads have {g.shape[0]} rows, the last forward pass had {self._rows}")
        _call(_trainer_backward_grads, self._handle(), _float_ptr(g))

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


def load_idx(images_path, labels_path, num_classes: int = 0):
    """Read an IDX pair (the MNIST format) as ``(inputs, one_hot_targets)`` float32 arrays.
    Unsigned-byte images are scaled to [0, 1]; ``num_classes`` 0 = largest label + 1."""
    ds = _Dataset()
    _call(_load_idx, _encode_path(images_path), _encode_path(labels_path), int(num_classes), ctypes.byref(ds))
    return _take_dataset(ds)


def load_csv(path, target_columns: int = 1, num_classes: int = 0):
    """Read a numeric CSV file as ``(inputs, targets)``: the last ``target_columns`` columns are
    targets; with ``num_classes`` > 0 the single target column is one-hot encoded."""
    ds = _Dataset()
    _call(_load_csv, _encode_path(path), int(target_columns), int(num_classes), ctypes.byref(ds))
    return _take_dataset(ds)
