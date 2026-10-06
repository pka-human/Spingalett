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
    r = net.train(x, y, epochs=100, callback=lambda n, p: (seen.append((p.epoch, p.train_loss)), p.epoch >= 7)[1], callback_interval=1)
    check([e for e, _ in seen] == list(range(1, 8)), f"early stop at epoch 7: {[e for e, _ in seen]}")
    check(net.time_step == 7, f"time_step after early stop: {net.time_step}")
    check(r.status == sg.TrainStatus.INTERRUPTED and r.epochs_run == 7 and abs(r.train_loss - seen[-1][1]) < 1e-7, f"train result {r}")
    # exception inside a callback stops training and is re-raised
    class Boom(Exception): pass
    def bad_cb(n, p):
        if p.epoch == 3: raise Boom("stop")
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

# validation, early stopping, best weights, evaluate
rng3 = np.random.default_rng(11)
vx = rng3.normal(size=(40, 3)).astype(np.float32)
vy = (vx[:, :1] > 0).astype(np.float32)
with sg.Network(sg.Loss.CROSS_ENTROPY, [3, sg.Layer(8, sg.Activation.TANH, sg.Init.XAVIER), sg.Layer(1, sg.Activation.SIGMOID, sg.Init.XAVIER)]) as net:
    snapshots, progress = {}, []
    def keep(n, p):
        progress.append(p); snapshots[p.epoch] = n.get_weights(0)
    # the validation targets are inverted: validation loss gets worse as training improves
    r = net.train(vx, vy, epochs=60, optimizer=sg.Optimizer.ADAM, learning_rate=0.05, validation_data=(vx, 1 - vy),
                  early_stopping_patience=3, restore_best_weights=True, callback=keep)
    check(r.status == sg.TrainStatus.EARLY_STOPPED and r.epochs_run == r.best_epoch + 3 and r.restored_best
          and r.monitor == sg.Monitor.VAL_LOSS, f"early stopping result {r}")
    check(np.array_equal(net.get_weights(0), snapshots[r.best_epoch]), "restored weights are the best epoch's")
    check(all(p.validation is not None for p in progress) and progress[r.best_epoch - 1].improved, "progress validation metrics")
    m = net.evaluate(vx, 1 - vy)
    check(isinstance(m, sg.Metrics) and abs(m.loss - r.best_value) < 1e-5, f"evaluate {m} vs best {r.best_value}")
    acc = float(((net.forward(vx) >= 0.5) == (vy >= 0.5)).mean())
    check(abs(net.evaluate(vx, vy).accuracy - acc) < 1e-6, "evaluate accuracy")
    try:
        net.train(vx, vy, epochs=2, monitor=sg.Monitor.VAL_ACCURACY); check(False, "validation monitor without data")
    except sg.SpingalettError: pass

# low-level trainer: custom loss gradients equal the built-in ones; step matches train()
def small(seed_value):
    sg.seed(seed_value)
    return sg.Network(sg.Loss.CROSS_ENTROPY, [3, sg.Layer(6, sg.Activation.RELU, sg.Init.HE), sg.Layer(2, sg.Activation.SOFTMAX, sg.Init.XAVIER)])
ty = np.eye(2, dtype=np.float32)[(vx[:, 0] > 0).astype(int)]
with small(5) as a, small(5) as b:
    with sg.Trainer(a, max_batch=40) as tr:
        tr.forward(vx); tr.backward(ty)
        g_builtin = a.get_weight_gradients(0)
        tr.zero_grad()
        out = tr.forward(vx); tr.backward_output_grads(-ty / out)
        check(np.allclose(a.get_weight_gradients(0), g_builtin, atol=1e-5), "custom output gradients")
        tr.zero_grad()
        losses = [tr.train_on_batch(vx[i:i + 10], ty[i:i + 10], optimizer=sg.Optimizer.ADAM, learning_rate=0.01) for i in range(0, 40, 10)]
        try:
            tr.backward(ty[:10]); check(False, "backward without forward accepted")
        except sg.SpingalettError: pass
    b.train(vx, ty, epochs=1, optimizer=sg.Optimizer.ADAM, learning_rate=0.01, strategy=sg.Strategy.MINI_BATCH, batch_size=10, shuffle=False)
    check(np.allclose(a.get_weights(0), b.get_weights(0), atol=1e-6) and a.time_step == b.time_step == 4 and all(np.isfinite(losses)),
          "Trainer.train_on_batch == train() mini-batches")

# data set readers
with tempfile.TemporaryDirectory() as d:
    path = os.path.join(d, "data.csv")
    with open(path, "w") as f:
        f.write("x0,x1,label\n" + "".join(f"{i},{i * 2},{i % 2}\n" for i in range(9)))
    cx, cy = sg.load_csv(path, target_columns=1, num_classes=2)
    check(cx.shape == (9, 2) and cy.shape == (9, 2) and cx[4, 1] == 8 and cy[3, 1] == 1, f"load_csv {cx.shape} {cy.shape}")
    try:
        sg.load_csv(os.path.join(d, "missing.csv")); check(False, "missing csv loaded")
    except sg.SpingalettError as e:
        check(e.code == sg.ErrorCode.FILE_IO, "missing csv error code")

# .slettd data set files: exact round trip, encodings, training straight from the file
with tempfile.TemporaryDirectory() as d:
    px = (rng.integers(0, 256, size=(500, 64)) * (rng.random((500, 64)) < 0.3)).astype(np.float32) / 255
    py = np.eye(4, dtype=np.float32)[rng.integers(0, 4, 500)]
    path = os.path.join(d, "set")
    sg.save_dataset(path, px, py)
    info = sg.dataset_info(path + ".slettd")
    lx, ly = sg.load_dataset(path + ".slettd")
    check(np.array_equal(lx, px) and np.array_equal(ly, py), "slettd round trip")
    check(info["input_encoding"] == sg.DatasetEncoding.U8_UNIT and info["target_encoding"] == sg.DatasetEncoding.CLASS
          and info["count"] == 500 and info["file_size"] < px.nbytes // 8, f"slettd info {info}")
    sg.save_dataset(os.path.join(d, "h.slettd"), px * 3.3, py, input_encoding=sg.DatasetEncoding.FP16)
    hx, _ = sg.load_dataset(os.path.join(d, "h.slettd"))
    check(np.array_equal(hx, (px * 3.3).astype(np.float16).astype(np.float32)), "slettd fp16 == numpy float16 rounding")
    with small(8) as a, small(8) as b:
        a.train(px[:, :3], py[:, :2], epochs=2, strategy=sg.Strategy.MINI_BATCH, batch_size=50, shuffle=False, optimizer=sg.Optimizer.ADAM)
        sg.save_dataset(os.path.join(d, "t.slettd"), px[:, :3], py[:, :2])
        r = b.train_from_file(os.path.join(d, "t.slettd"), shuffle=False, epochs=2, strategy=sg.Strategy.MINI_BATCH,
                              batch_size=50, optimizer=sg.Optimizer.ADAM)
        check(r.status == sg.TrainStatus.COMPLETED and np.array_equal(a.get_weights(0), b.get_weights(0)),
              f"train_from_file == train on arrays ({r.status})")
        r = b.train_from_file(os.path.join(d, "t.slettd"), epochs=3, strategy=sg.Strategy.FULL_BATCH)
        check(r.status == sg.TrainStatus.COMPLETED and r.epochs_run == 3, f"train_from_file full batch {r}")
    try:
        sg.load_dataset(os.path.join(d, "missing.slettd")); check(False, "missing data set loaded")
    except sg.SpingalettError as e:
        check(e.code == sg.ErrorCode.FILE_IO, "missing data set error code")

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
    with sg.Network.load(path + ".slett") as loaded:
        check(np.array_equal(loaded.forward(x), out) and loaded.dropout_rates == net.dropout_rates and loaded.time_step == net.time_step, "fp32 roundtrip")
    net.save(os.path.join(d, "h.slett"), precision=sg.Precision.FP16, save_optimizer=False)
    with sg.Network.load(os.path.join(d, "h.slett")) as half:
        check(np.allclose(half.get_weights(0), net.get_weights(0).astype(np.float16), atol=0, rtol=0), "fp16 save == numpy float16 rounding")
    net.close()
    try:
        sg.Network.load(os.path.join(d, "missing.slett")); check(False, "missing file loaded")
    except sg.SpingalettError as e:
        check("cannot open" in str(e) and e.code == sg.ErrorCode.FILE_IO, f"error: {e} code {e.code!r}")
check(sg.library_version() == sg.__version__, f"library {sg.library_version()} vs bindings {sg.__version__}")

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

# deployment models
sg.seed(5)
with sg.Network(sg.Loss.CROSS_ENTROPY, [sg.Layer(12), sg.Layer(20, sg.Activation.RELU, sg.Init.HE),
                                       sg.Layer(5, sg.Activation.SOFTMAX, sg.Init.XAVIER)]) as net:
    x = rng.normal(size=(40, 12)).astype(np.float32)
    ref = net.forward(x)
    blob = net.to_bytes()
    check(blob[:6] == b"SLETTM" and len(blob) % 16 == 0, "to_bytes: format 3 image")
    with sg.Network.from_bytes(blob) as back:
        check(np.array_equal(back.forward(x), ref), "from_bytes round trip")
    for p, tol in ((sg.Precision.FLOAT32, 1e-6), (sg.Precision.FP16, 3e-3), (sg.Precision.INT8, 2e-2), (sg.Precision.INT4, 0.2)):
        with net.to_model(p) as m:
            out = m.predict(x)
            check(m.input_size == 12 and m.output_size == 5 and m.loss == sg.Loss.CROSS_ENTROPY and
                  [l.precision for l in m.layers] == [p, p] and m.layers[0].activation == sg.Activation.RELU,
                  f"model description {m!r}")
            check(np.abs(out - ref).max() < tol, f"model {p.name} vs network: {np.abs(out - ref).max():.2e}")
            check(np.array_equal(m(x[7]), out[7]), "model single sample")
            metrics = m.evaluate(x, ref)
            check(0 <= metrics.accuracy <= 1 and math.isfinite(metrics.loss), "model evaluate")
            with sg.Model.from_bytes(m.to_bytes()) as copy:
                check(np.array_equal(copy.predict(x), out), "model to_bytes / from_bytes")
    with tempfile.TemporaryDirectory() as d:
        net.save(os.path.join(d, "net"), precision=sg.Precision.INT8, save_optimizer=False)
        with sg.Model.load(os.path.join(d, "net.slett")) as m:
            check(m.layers[0].precision == sg.Precision.INT8 and m.size == len(net.to_bytes(sg.Precision.INT8)),
                  "Model.load of an INT8 file")
        net.export_c_header(os.path.join(d, "model.h"), "my_model", sg.Precision.INT4)
        text = open(os.path.join(d, "model.h")).read()
        check("static const uint8_t my_model[MY_MODEL_SIZE]" in text and "#define MY_MODEL_INPUTS 12u" in text, "C header")
        try:
            net.export_c_header(os.path.join(d, "bad.h"), "not-an-identifier"); check(False, "bad header name accepted")
        except sg.SpingalettError as e:
            check(e.code == sg.ErrorCode.INVALID, "bad header name error code")
try:
    sg.Model.from_bytes(b"SLETTM" + bytes(100)); check(False, "corrupt model accepted")
except sg.SpingalettError:
    pass

# lifetime
net = sg.Network(sg.Loss.MSE, [2, 3]); net.close(); net.close()
try: net.forward([0, 0]); check(False, "closed network usable")
except ValueError: pass
print(repr(sg.Network(sg.Loss.CROSS_ENTROPY, [3, sg.Layer(2, sg.Activation.SOFTMAX)])))
print("library:", sg.library_path())
print("ALL PASSED" if not failures else f"{len(failures)} FAILURES")
sys.exit(1 if failures else 0)
