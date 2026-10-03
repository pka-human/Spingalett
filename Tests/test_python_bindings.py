# SPDX-License-Identifier: MIT
"""Tests for the Python bindings. Run through ctest, or:
    SPINGALETT_LIBRARY=Bin/libspingalett.so PYTHONPATH=Bindings/Python python Tests/test_python_bindings.py
"""
import os, sys, math, tempfile
import numpy as np
import spingalett as sg

sg.set_verbose(False)
failures = []
def check(cond, msg):
    if not cond: failures.append(msg); print("FAIL:", msg)

ACT = {sg.Activation.TANH: np.tanh, sg.Activation.SIGMOID: lambda z: 1 / (1 + np.exp(-z)),
       sg.Activation.RELU: lambda z: np.maximum(z, 0), sg.Activation.NONE: lambda z: z,
       sg.Activation.SOFTMAX: lambda z: np.exp(z - z.max(-1, keepdims=True)) / np.exp(z - z.max(-1, keepdims=True)).sum(-1, keepdims=True)}

def np_forward(net, x):
    a = np.asarray(x, dtype=np.float64)
    for i, act in enumerate(net.activations):
        a = ACT[act](a @ net.get_weights(i).T.astype(np.float64) + net.get_biases(i))
    return a

rng = np.random.default_rng(0)
for mode in (sg.ComputeMode.SINGLE_THREADED, sg.ComputeMode.OPENMP, sg.ComputeMode.OPENBLAS):
    sg.set_compute_mode(mode)
    sg.seed(1)
    net = sg.Network(sg.Loss.CROSS_ENTROPY, [sg.Layer(5), sg.Layer(17, sg.Activation.TANH, sg.Init.XAVIER),
                                             sg.Layer(9, sg.Activation.RELU, sg.Init.HE), sg.Layer(4, sg.Activation.SOFTMAX, sg.Init.XAVIER)])
    check(net.topology == [5, 17, 9, 4] and net.num_parameters == 5*17+17*9+9*4+17+9+4, "topology/params")
    x = rng.normal(size=(32, 5)).astype(np.float32)
    y = np.eye(4, dtype=np.float32)[rng.integers(0, 4, 32)]
    # forward agrees with an independent numpy implementation
    out = net.forward(x)
    check(np.allclose(out, np_forward(net, x), atol=2e-6), f"forward vs numpy (mode {mode.name})")
    check(np.allclose(net.forward(x[3]), out[3]) and net.forward(x[3]).shape == (4,), "single-sample forward")
    # one SGD step with lr=1 = minus the mean gradient; compare to central differences of the numpy loss
    def loss_np():
        p = np_forward(net, x); return float(-(y * np.log(p)).sum(-1).mean())
    w0 = [net.get_weights(i) for i in range(3)]; b0 = [net.get_biases(i) for i in range(3)]
    net.train(x, y, epochs=1, optimizer=sg.Optimizer.SGD, learning_rate=1.0)
    g_analytic = w0[1] - net.get_weights(1)
    for i in range(3): net.set_weights(i, w0[i]); net.set_biases(i, b0[i])
    g_num = np.zeros_like(w0[1])
    for idx in [(0, 0), (3, 7), (8, 16), (5, 2)]:
        for sgn in (1, -1):
            w = w0[1].copy(); w[idx] += sgn * 1e-3; net.set_weights(1, w)
            g_num[idx] += sgn * loss_np() / 2e-3
        net.set_weights(1, w0[1])
        check(abs(g_num[idx] - g_analytic[idx]) < 2e-3 + 3e-2 * abs(g_analytic[idx]), f"gradient {idx} mode {mode.name}: {g_num[idx]} vs {g_analytic[idx]}")
    # training reduces loss
    before = loss_np()
    net.train(x, y, epochs=200, optimizer=sg.Optimizer.ADAM, learning_rate=0.01, strategy=sg.Strategy.MINI_BATCH, batch_size=8)
    check(loss_np() < before * 0.5, f"training reduces loss (mode {mode.name}): {before} -> {loss_np()}")
    net.close()

sg.set_compute_mode(sg.ComputeMode.SINGLE_THREADED)
x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32); y = np.array([[0], [1], [1], [0]], dtype=np.float32)

# callback: early stop + epochs seen
seen = []
with sg.Network(sg.Loss.MSE, [2, sg.Layer(4, sg.Activation.TANH), sg.Layer(1)]) as net:
    net.train(x, y, epochs=100, callback=lambda n, e, err: (seen.append((e, err)), e >= 7)[1], callback_interval=1)
    check([e for e, _ in seen] == list(range(1, 8)), f"early stop at epoch 7: {[e for e, _ in seen]}")
    check(net.time_step == 7, f"time_step after early stop: {net.time_step}")
    # exception inside a callback stops training and is re-raised
    class Boom(Exception): pass
    def bad_cb(n, e, err):
        if e == 3: raise Boom("stop")
    try:
        net.train(x, y, epochs=50, callback=bad_cb); check(False, "exception not propagated")
    except Boom:
        check(net.time_step == 7 + 3, f"training stopped at the failing epoch: {net.time_step}")
    # python lr schedule receives (epoch, total, initial_lr) and its rate is applied
    calls = []
    net.train(x, y, epochs=5, optimizer=sg.Optimizer.SGD, learning_rate=0.5,
              lr_scheduler=lambda e, t, lr: (calls.append((e, t, lr)), 0.0)[1])
    check(calls == [(e, 5, 0.5) for e in range(5)], f"python scheduler calls {calls}")
    w = net.get_weights(0)
    net.train(x, y, epochs=3, optimizer=sg.Optimizer.SGD, lr_scheduler=lambda e, t, lr: 0.0)
    check(np.array_equal(w, net.get_weights(0)), "lr=0 schedule must freeze weights")
    # built-in schedules
    check(abs(sg.CosineDecay()(50, 100, 1.0) - 0.5) < 1e-6 and abs(sg.StepDecay(10, 0.5)(25, 100, 1.0) - 0.25) < 1e-6, "builtin schedule values")
    net.train(x, y, epochs=10, lr_scheduler=sg.WarmupCosine(warmup_epochs=2))
    # validation errors
    for bad in (lambda: net.train(x[:, :1], y, epochs=1), lambda: net.train(x, y[:3], epochs=1), lambda: net.forward([1, 2, 3])):
        try: bad(); check(False, "bad shape accepted")
        except ValueError: pass
    try:
        net.train(x, y, epochs=1, optimizer=99); check(False, "invalid optimizer accepted")
    except (sg.SpingalettError, ValueError): pass

# dropout + save/load + precision
with tempfile.TemporaryDirectory() as d:
    sg.seed(3)
    net = sg.Network(sg.Loss.MSE, [sg.Layer(2), sg.Layer(32, sg.Activation.TANH, sg.Init.XAVIER, dropout=0.25), sg.Layer(1, sg.Activation.SIGMOID, sg.Init.XAVIER)])
    check(net.dropout_rates == [0.0, 0.25, 0.0], f"dropout rates {net.dropout_rates}")
    net.train(x, y, epochs=2000, optimizer=sg.Optimizer.ADAM, learning_rate=0.02)
    out = net.forward(x)
    check(np.all(np.abs(out - y) < 0.25), f"xor with dropout: {out.ravel()}")
    path = os.path.join(d, "model")
    net.save(path)
    with sg.Network.load(path + ".nn") as loaded:
        check(np.array_equal(loaded.forward(x), out) and loaded.dropout_rates == net.dropout_rates and loaded.time_step == net.time_step, "fp32 roundtrip")
    net.save(os.path.join(d, "h.nn"), precision=sg.Precision.FP16, save_optimizer=False)
    with sg.Network.load(os.path.join(d, "h.nn")) as half:
        check(np.allclose(half.get_weights(0), net.get_weights(0).astype(np.float16), atol=0, rtol=0), "fp16 save == numpy float16 rounding")
    net.close()
    try:
        sg.Network.load(os.path.join(d, "missing.nn")); check(False, "missing file loaded")
    except sg.SpingalettError as e:
        check("cannot open" in str(e), f"error message: {e}")

# generator mode
sg.set_compute_mode(sg.ComputeMode.OPENBLAS)
rng2 = np.random.default_rng(7)
gx = rng2.normal(size=(30, 4)).astype(np.float32); gy = np.eye(2, dtype=np.float32)[rng2.integers(0, 2, 30)]
def make(): 
    sg.seed(9)
    return sg.Network(sg.Loss.CROSS_ENTROPY, [4, sg.Layer(10, sg.Activation.TANH, sg.Init.XAVIER, dropout=0.2), sg.Layer(2, sg.Activation.SOFTMAX, sg.Init.XAVIER)])
with make() as a, make() as b:
    sg.seed(3); a.train(gx, gy, epochs=6, optimizer=sg.Optimizer.ADAM)
    def full(xs, ys):
        xs[:30] = gx; ys[:30] = gy; return 30
    sg.seed(3); b.train_from_generator(full, epochs=6, optimizer=sg.Optimizer.ADAM, samples_per_epoch=30)
    check(np.array_equal(a.get_weights(0), b.get_weights(0)) and a.time_step == b.time_step == 6, "generator full batch == array full batch")
    calls = []
    class Stream:
        pos = 0
        def __call__(self, xs, ys):
            calls.append(len(xs))
            n = min(len(xs), 30 - self.pos)
            xs[:n] = gx[self.pos:self.pos + n]; ys[:n] = gy[self.pos:self.pos + n]
            self.pos = self.pos + n if n else 0       # 0 rows ends the epoch; rewind for the next one
            return n
    b.train_from_generator(Stream(), epochs=2, strategy=sg.Strategy.MINI_BATCH, batch_size=8)
    check(calls[:5] == [8, 8, 8, 8, 8] and b.time_step == 6 + 8, f"mini-batch generator calls {calls} steps {b.time_step}")
    def broken(xs, ys): raise KeyError("data source failed")
    try:
        b.train_from_generator(broken, epochs=3, strategy=sg.Strategy.MINI_BATCH); check(False, "generator exception swallowed")
    except KeyError:
        pass
    try:
        b.train_from_generator(full, epochs=1); check(False, "full batch without samples_per_epoch accepted")
    except ValueError:
        pass
sg.set_compute_mode(sg.ComputeMode.SINGLE_THREADED)

# logging callback
msgs = []
sg.set_verbose(True); sg.set_log_callback(lambda lvl, m: msgs.append((lvl, m)))
with sg.Network(sg.Loss.MSE, [2, 3]) as net: pass
sg.set_log_callback(None); sg.set_verbose(False)
check(any("Creating new network" in m for _, m in msgs) and all(isinstance(l, sg.LogLevel) for l, _ in msgs), f"log callback {msgs[:2]}")

# lifetime
net = sg.Network(sg.Loss.MSE, [2, 3]); net.close(); net.close()
try: net.forward([0, 0]); check(False, "closed network usable")
except ValueError: pass
print(repr(sg.Network(sg.Loss.CROSS_ENTROPY, [3, sg.Layer(2, sg.Activation.SOFTMAX)])))
print("library:", sg.library_path())
print("ALL PASSED" if not failures else f"{len(failures)} FAILURES")
sys.exit(1 if failures else 0)
