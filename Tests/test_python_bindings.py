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

    # shape, class names and a second set of targets; uint8 images; in-memory and u8 training
    imgs = rng.integers(0, 256, size=(300, 4, 4, 3), dtype=np.uint8)
    labels = rng.integers(0, 3, 300)
    yd = np.eye(3, dtype=np.float32)[labels]
    yp = np.eye(2, dtype=np.float32)[labels % 2]
    meta = os.path.join(d, "meta.slettd")
    sg.save_dataset(meta, imgs, yd, class_names=["cat", "dog", "fox"], target_name="animal",
                    extra_targets=[{"targets": yp, "name": "parity", "class_names": ["even", "odd"]}])
    info = sg.dataset_info(meta)
    check(info["shape"] == (4, 4, 3) and info["format_version"] == 2 and info["input_encoding"] == sg.DatasetEncoding.U8_UNIT
          and info["target_sets"] == [{"name": "animal", "size": 3, "class_names": ["cat", "dog", "fox"]},
                                      {"name": "parity", "size": 2, "class_names": ["even", "odd"]}], f"slettd metadata {info}")
    mx, my = sg.load_dataset(meta)
    check(np.array_equal(mx, imgs.reshape(300, -1).astype(np.float64).__truediv__(255.0).astype(np.float32))
          and np.array_equal(my, yd), "uint8 images round trip as q / 255")
    _, mp = sg.load_dataset(meta, target_set=1)
    for bad in (-1, 2):     # -1 would wrap around to the inputs in C
        try:
            sg.load_dataset(meta, target_set=bad)
            check(False, f"load_dataset(target_set={bad}) did not raise")
        except (ValueError, sg.SpingalettError):
            pass
    check(np.array_equal(mp, yp), "second set of targets")
    def image_net():
        sg.seed(48)
        return sg.Network(sg.Loss.CROSS_ENTROPY, [48, sg.Layer(8, sg.Activation.RELU, sg.Init.HE),
                                                  sg.Layer(3, sg.Activation.SOFTMAX, sg.Init.XAVIER)])
    with image_net() as a, image_net() as b, image_net() as c:
        a.train(mx, yd, epochs=2, strategy=sg.Strategy.MINI_BATCH, batch_size=32, shuffle=False, optimizer=sg.Optimizer.ADAM)
        b.train(imgs.reshape(300, -1), yd, epochs=2, strategy=sg.Strategy.MINI_BATCH, batch_size=32, shuffle=False,
                optimizer=sg.Optimizer.ADAM)
        c.train_from_file(meta, in_memory=True, shuffle=False, epochs=2, strategy=sg.Strategy.MINI_BATCH, batch_size=32,
                          optimizer=sg.Optimizer.ADAM)
        check(np.array_equal(a.get_weights(0), b.get_weights(0)) and np.array_equal(a.get_weights(0), c.get_weights(0)),
              "training on uint8 arrays and in memory == training on floats")
        check(np.array_equal(a.forward(imgs.reshape(300, -1)[:5]), a.forward(mx[:5])), "forward reads uint8 as q / 255")

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
check(sg.cpu_kernels() in ("AVX-512", "AVX2", "AVX", "SSE2", "NEON", "C"), f"cpu kernels {sg.cpu_kernels()!r}")
check(sg.gpu_device() is None or isinstance(sg.gpu_device(), str), f"gpu device {sg.gpu_device()!r}")
sg.set_gpu_precision(sg.Precision.BFLOAT16)
check(sg.get_gpu_precision() == sg.Precision.BFLOAT16, "gpu precision bfloat16")
sg.set_gpu_precision(sg.Precision.FLOAT32)

# the GPU (or, without one, the CPU it falls back to) trains and predicts as the CPU does
gx = np.random.default_rng(3).normal(size=(64, 6)).astype(np.float32)
gy = np.eye(3, dtype=np.float32)[np.arange(64) % 3]
results = []
for mode in (sg.ComputeMode.OPENMP, sg.ComputeMode.VULKAN):
    sg.set_compute_mode(mode)
    sg.seed(4)
    with sg.Network(sg.Loss.CROSS_ENTROPY, [6, sg.Layer(8, sg.Activation.RELU), sg.Layer(3, sg.Activation.SOFTMAX)]) as gn:
        gn.train(gx, gy, epochs=3, strategy=sg.Strategy.MINI_BATCH, batch_size=16, shuffle=False)
        results.append(gn.forward(gx))
check(sg.get_compute_mode() == sg.ComputeMode.VULKAN, "compute mode VULKAN")
check(np.allclose(results[0], results[1], rtol=1e-4, atol=1e-6), "GPU training == CPU training")

# data sets in the GPU's memory: the same bits as arrays, on the GPU and on the CPU
if sg.gpu_device() is not None:
    with sg.DeviceData(gx) as dx, sg.DeviceData(gy) as dy:
        check(len(dx) == 64 and dx.shape == (64, 6) and np.array_equal(dx.numpy(), gx)
              and np.array_equal(dy.numpy(5, 2), gy[5:7]), f"device data {dx!r}")
        for mode in (sg.ComputeMode.VULKAN, sg.ComputeMode.OPENMP):
            sg.set_compute_mode(mode)
            outs, metrics = [], []
            for xs, ys in ((gx, gy), (dx, dy), (dx, gy)):
                sg.seed(4)
                with sg.Network(sg.Loss.CROSS_ENTROPY, [6, sg.Layer(8, sg.Activation.RELU),
                                                       sg.Layer(3, sg.Activation.SOFTMAX)]) as gn:
                    gn.train(xs, ys, epochs=3, strategy=sg.Strategy.MINI_BATCH, batch_size=16, label_smoothing=0.1,
                             validation_data=(gx if ys is dy else dx, dy))
                    outs.append(gn.forward(xs))
                    metrics.append(gn.evaluate(xs, ys))
            check(all(np.array_equal(outs[0], o) for o in outs) and metrics[0] == metrics[1] == metrics[2],
                  f"{mode.name}: training on DeviceData == on arrays")
        try:
            with sg.Network(sg.Loss.MSE, [5, 3]) as gn:
                gn.forward(dx); check(False, "DeviceData of another row size accepted")
        except ValueError:
            pass
    check("closed" in repr(dx), "DeviceData closed by with")
    with sg.DeviceData(np.array([[0, 51, 255]], dtype=np.uint8)) as du:
        check(np.array_equal(du.numpy(), np.array([[0, 51, 255]], dtype=np.float32) / np.float32(255)) or
              np.allclose(du.numpy(), [[0.0, 0.2, 1.0]], rtol=0, atol=1e-7), f"DeviceData of image bytes {du.numpy()}")
sg.set_compute_mode(sg.ComputeMode.OPENMP)

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
            # integer models compute a sample exactly as in a batch; float ones may round differently
            # (one sample or a few run through dot products)
            single = m(x[7])
            check(np.array_equal(single, out[7]) if p in (sg.Precision.INT8, sg.Precision.INT4)
                  else np.allclose(single, out[7], rtol=1e-5, atol=1e-6), "model single sample")
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

# convolution and pooling: forward against numpy, descriptions, parameters, training, files, models
def np_conv(x, w, b, stride, pad):           # x (H, W, C), w (F, kh, kw, C), channels last
    H, W, C = x.shape; F, kh, kw, _ = w.shape
    xp = np.zeros((H + 2 * pad, W + 2 * pad, C)); xp[pad:pad + H, pad:pad + W] = x
    oh, ow = (H + 2 * pad - kh) // stride + 1, (W + 2 * pad - kw) // stride + 1
    out = np.empty((oh, ow, F))
    for i in range(oh):
        for j in range(ow):
            win = xp[i * stride:i * stride + kh, j * stride:j * stride + kw]
            out[i, j] = np.tensordot(w, win, axes=([1, 2, 3], [0, 1, 2])) + b
    return out

def np_pool(x, k, stride, pad, op):          # windows clipped to the image: padding is not counted
    H, W, C = x.shape
    oh, ow = (H + 2 * pad - k) // stride + 1, (W + 2 * pad - k) // stride + 1
    out = np.empty((oh, ow, C))
    for i in range(oh):
        for j in range(ow):
            h0, w0 = max(i * stride - pad, 0), max(j * stride - pad, 0)
            win = x[h0:min(i * stride - pad + k, H), w0:min(j * stride - pad + k, W)]
            out[i, j] = win.max((0, 1)) if op == "max" else win.mean((0, 1))
    return out

sg.seed(5)
cnn = sg.Network(sg.Loss.CROSS_ENTROPY, [sg.Input(8, 7, 2), sg.Conv2D(5, 3, padding=1),
                                         sg.MaxPool2D(2), sg.Conv2D(4, 2, stride=1, activation=sg.Activation.TANH),
                                         sg.AvgPool2D(2, stride=1, padding=1), sg.Layer(3, sg.Activation.SOFTMAX, sg.Init.XAVIER)])
L = cnn.layers
check(L[0].shape == (8, 7, 2) and L[1].type == sg.LayerType.CONV2D and L[1].shape == (8, 7, 5) and L[1].kernel == (3, 3) and
      L[2].type == sg.LayerType.MAX_POOL2D and L[2].shape == (4, 3, 5) and L[2].weight_count == 0 and
      L[3].shape == (3, 2, 4) and L[4].shape == (4, 3, 4) and L[4].padding == (1, 1) and cnn.topology[-1] == 3,
      f"conv layer descriptions: {L}")
check(cnn.get_weights(0).shape == (5, 3, 3, 2) and cnn.get_weights(1).shape == (0, 0) and cnn.get_biases(1).shape == (0,) and
      cnn.get_weights(2).shape == (4, 2, 2, 5) and cnn.get_weights(4).shape == (3, 48), "conv weight shapes")
w = rng.normal(size=(5, 3, 3, 2)).astype(np.float32); cnn.set_weights(0, w)
check(np.array_equal(cnn.get_weights(0), w), "conv set/get weights")

def np_cnn(net, x):
    out = []
    for s in x:
        a = s.reshape(8, 7, 2).astype(np.float64)
        a = np.maximum(np_conv(a, net.get_weights(0), net.get_biases(0), 1, 1), 0)
        a = np_pool(a, 2, 2, 0, "max")
        a = np.tanh(np_conv(a, net.get_weights(2), net.get_biases(2), 1, 0))
        a = np_pool(a, 2, 1, 1, "avg")
        out.append(ACT[sg.Activation.SOFTMAX](net.get_weights(4).astype(np.float64) @ a.reshape(-1) + net.get_biases(4)))
    return np.array(out)

xc = rng.normal(size=(6, 8 * 7 * 2)).astype(np.float32)
for mode in (sg.ComputeMode.SINGLE_THREADED, sg.ComputeMode.OPENMP):
    sg.set_compute_mode(mode)
    check(np.allclose(cnn.forward(xc), np_cnn(cnn, xc), atol=1e-5), f"conv forward vs numpy ({mode.name})")
    check(np.allclose(cnn.forward(xc[2]), np_cnn(cnn, xc[2:3])[0], atol=1e-5), f"conv single-sample forward ({mode.name})")
sg.set_compute_mode(sg.ComputeMode.SINGLE_THREADED)

# learns vertical against horizontal bars
def bars(n):
    xs = np.zeros((n, 8, 7, 2), np.float32); ys = np.zeros((n, 3), np.float32)
    for i in range(n):
        k = rng.integers(0, 3)
        if k == 0: xs[i, rng.integers(0, 8), :, rng.integers(0, 2)] = 1
        elif k == 1: xs[i, :, rng.integers(0, 7), rng.integers(0, 2)] = 1
        else: xs[i, rng.integers(0, 7):, rng.integers(0, 6)] = 0.5
        ys[i, k] = 1
    return xs.reshape(n, -1), ys
xb, yb = bars(240)
cnn.train(xb, yb, epochs=30, optimizer=sg.Optimizer.ADAM, learning_rate=0.01, strategy=sg.Strategy.MINI_BATCH, batch_size=16)
xt, yt = bars(120)
acc = cnn.evaluate(xt, yt).accuracy
check(acc >= 0.9, f"conv network learns bars: accuracy {acc}")

with tempfile.TemporaryDirectory() as d:
    path = os.path.join(d, "cnn.slett")
    cnn.save(path)
    data = open(path, "rb").read()
    check(data[6] == 4, "conv networks are saved as format version 4")
    with sg.Network.load(path) as back:
        check(back.layers == cnn.layers and all(np.array_equal(back.get_weights(i), cnn.get_weights(i)) for i in range(5)),
              "conv save/load round trip")
    with cnn.to_model(sg.Precision.INT8) as m:
        info = m.layers
        check(info[0].type == sg.LayerType.CONV2D and info[0].shape == (8, 7, 5) and info[0].input_shape == (8, 7, 2) and
              info[1].type == sg.LayerType.MAX_POOL2D and info[1].kernel == (2, 2) and info[4].type == sg.LayerType.DENSE,
              f"conv model layer info: {info}")
        p = m.predict(xt)
        check(np.array_equal(p[5], m(xt[5])), "conv model: batched equals single")
        check(np.abs(p - cnn.forward(xt)).max() < 0.05 and (p.argmax(1) == cnn.forward(xt).argmax(1)).mean() > 0.95,
              "INT8 conv model close to the network")

# batch normalization: descriptions, statistics, inference against numpy, training, files, folded models
sg.seed(7)
bn = sg.Network(sg.Loss.CROSS_ENTROPY, [sg.Input(8, 7, 2), sg.Conv2D(5, 3, padding=1, activation=sg.Activation.NONE),
                                        sg.BatchNorm(sg.Activation.RELU), sg.MaxPool2D(2),
                                        sg.BatchNorm(sg.Activation.TANH, epsilon=1e-3, momentum=0.2),
                                        sg.Layer(3, sg.Activation.SOFTMAX, sg.Init.XAVIER)])
L = bn.layers
check(L[2].type == sg.LayerType.BATCH_NORM and L[2].shape == (8, 7, 5) and L[2].weight_count == 5 and L[2].bias_count == 5 and
      abs(L[4].epsilon - 1e-3) < 1e-9 and abs(L[4].momentum - 0.2) < 1e-7 and L[1].groups == 1,
      f"batch norm layer descriptions: {L}")
mean, var = bn.get_running_statistics(1)
check(bn.get_weights(1).shape == (5,) and np.all(bn.get_weights(1) == 1) and np.all(mean == 0) and np.all(var == 1),
      "batch norm initial gamma and statistics")
bn.set_running_statistics(1, rng.normal(size=5), 0.5 + rng.random(5))
bn.set_running_statistics(3, rng.normal(size=5), 0.5 + rng.random(5))
bn.set_weights(1, 0.5 + rng.random(5)); bn.set_biases(1, rng.normal(size=5) * 0.1)

def np_bn(a, net, i, eps):
    m, v = net.get_running_statistics(i)
    return (a - m) / np.sqrt(v + eps) * net.get_weights(i) + net.get_biases(i)

def np_bn_cnn(net, x):
    out = []
    for s in x:
        a = np_conv(s.reshape(8, 7, 2).astype(np.float64), net.get_weights(0), net.get_biases(0), 1, 1)
        a = np.maximum(np_bn(a, net, 1, 1e-5), 0)
        a = np.tanh(np_bn(np_pool(a, 2, 2, 0, "max"), net, 3, 1e-3))
        out.append(ACT[sg.Activation.SOFTMAX](net.get_weights(4).astype(np.float64) @ a.reshape(-1) + net.get_biases(4)))
    return np.array(out)

check(np.allclose(bn.forward(xc), np_bn_cnn(bn, xc), atol=1e-5), "batch norm inference vs numpy")
r = bn.train(xb, yb, epochs=20, optimizer=sg.Optimizer.ADAM, learning_rate=0.02, strategy=sg.Strategy.MINI_BATCH,
             batch_size=16)
acc = bn.evaluate(xt, yt).accuracy
check(acc >= 0.9 and not np.allclose(bn.get_running_statistics(1)[0], mean), f"batch-normalized network learns bars: {acc}")
check(np.allclose(bn.forward(xc), np_bn_cnn(bn, xc), atol=1e-5), "batch norm inference vs numpy after training")
with tempfile.TemporaryDirectory() as d:
    path = os.path.join(d, "bn.slett")
    bn.save(path)
    check(open(path, "rb").read()[6] == 5, "batch-normalized networks are saved as format version 5")
    with sg.Network.load(path) as back:
        check(back.layers == bn.layers and np.array_equal(back.get_running_statistics(3)[1], bn.get_running_statistics(3)[1]),
              "batch norm save/load round trip")
    with bn.to_model(sg.Precision.FLOAT32) as m:
        kinds = [l.type for l in m.layers]
        check(kinds == [sg.LayerType.CONV2D, sg.LayerType.MAX_POOL2D, sg.LayerType.BATCH_NORM, sg.LayerType.DENSE] and
              m.layers[0].activation == sg.Activation.RELU and abs(m.layers[2].epsilon - 1e-3) < 1e-9,
              f"folded model layers: {m.layers}")
        check(np.abs(m.predict(xt) - bn.forward(xt)).max() < 1e-4, "folded model matches the network")

# grouped and depthwise convolutions against numpy, round trip, models
def np_group_conv(x, w, b, stride, pad, groups):   # w (F, kh, kw, C / groups)
    C, F = x.shape[2], w.shape[0]
    cg, fg = C // groups, F // groups
    return np.concatenate([np_conv(x[:, :, g * cg:(g + 1) * cg], w[g * fg:(g + 1) * fg], b[g * fg:(g + 1) * fg], stride, pad)
                           for g in range(groups)], axis=2)

sg.seed(11)
gnet = sg.Network(sg.Loss.MSE, [sg.Input(7, 6, 4), sg.Conv2D(8, 3, padding=1, activation=sg.Activation.TANH, groups=2),
                                sg.Conv2D(16, 3, stride=2, activation=sg.Activation.RELU, groups=8),
                                sg.Layer(2, sg.Activation.NONE, sg.Init.XAVIER)])
check(gnet.layers[1].groups == 2 and gnet.get_weights(0).shape == (8, 3, 3, 2) and gnet.get_weights(1).shape == (16, 3, 3, 1),
      f"grouped conv descriptions and weight shapes: {gnet.layers}")

def np_gnet(net, x):
    out = []
    for s in x:
        a = np.tanh(np_group_conv(s.reshape(7, 6, 4).astype(np.float64), net.get_weights(0), net.get_biases(0), 1, 1, 2))
        a = np.maximum(np_group_conv(a, net.get_weights(1), net.get_biases(1), 2, 0, 8), 0)
        out.append(net.get_weights(2).astype(np.float64) @ a.reshape(-1) + net.get_biases(2))
    return np.array(out)

xg = rng.normal(size=(5, 7 * 6 * 4)).astype(np.float32)
for mode in (sg.ComputeMode.SINGLE_THREADED, sg.ComputeMode.OPENMP):
    sg.set_compute_mode(mode)
    check(np.allclose(gnet.forward(xg), np_gnet(gnet, xg), atol=1e-5), f"grouped conv forward vs numpy ({mode.name})")
sg.set_compute_mode(sg.ComputeMode.SINGLE_THREADED)
yg = rng.normal(size=(5, 2)).astype(np.float32)
before = gnet.evaluate(xg, yg).loss
gnet.train(xg, yg, epochs=30, optimizer=sg.Optimizer.ADAM, learning_rate=0.01)
check(gnet.evaluate(xg, yg).loss < before * 0.5, "grouped network trains")
with tempfile.TemporaryDirectory() as d:
    path = os.path.join(d, "grouped.slett")
    gnet.save(path)
    check(open(path, "rb").read()[6] == 5, "grouped networks are saved as format version 5")
    with sg.Network.load(path) as back:
        check(back.layers == gnet.layers and np.array_equal(back.forward(xg), gnet.forward(xg)), "grouped save/load round trip")
    with gnet.to_model(sg.Precision.INT8) as m:
        check(m.layers[0].groups == 2 and m.layers[1].groups == 8, f"grouped model layers: {m.layers}")
        p = m.predict(xg)
        check(np.array_equal(p[3], m(xg[3])) and np.abs(p - gnet.forward(xg)).max() < 0.1, "grouped INT8 model")

# graphs: a residual block and branches that are concatenated, checked against numpy
def np_graph(net, x):
    out = []
    for sample in x:
        acts = [sample.reshape(net.layer(0).shape)]
        for i in range(1, len(net)):
            d = net.layer(i)
            src = acts[d.inputs[0]]
            if d.type == sg.LayerType.CONV2D:
                w = net.get_weights(i - 1)
                v = np_conv(src, w, net.get_biases(i - 1), d.stride[0], d.padding[0])
            elif d.type == sg.LayerType.ADD:
                v = sum(acts[j] for j in d.inputs)
            elif d.type == sg.LayerType.CONCAT:
                v = np.concatenate([acts[j] for j in d.inputs], axis=2)
            elif d.type == sg.LayerType.GLOBAL_AVG_POOL:
                v = src.mean(axis=(0, 1)).reshape(1, 1, -1)
            else:
                v = (net.get_weights(i - 1) @ src.reshape(-1) + net.get_biases(i - 1)).reshape(1, 1, -1)
            if d.activation == sg.Activation.RELU: v = np.maximum(v, 0)
            elif d.activation == sg.Activation.TANH: v = np.tanh(v)
            elif d.activation == sg.Activation.SOFTMAX: v = np.exp(v - v.max()); v = v / v.sum()
            acts.append(v)
        out.append(acts[-1].reshape(-1))
    return np.array(out)

rnet = sg.Network(sg.Loss.CROSS_ENTROPY)
rnet.add_input(6, 6, 2)
stem = rnet.add_conv2d(4, 3, padding=1, activation=sg.Activation.RELU).last
rnet.add_conv2d(4, 3, padding=1, activation=sg.Activation.TANH)
rnet.add_conv2d(4, 3, padding=1, activation=sg.Activation.NONE)
block = rnet.add_add([stem, -1], activation=sg.Activation.RELU).last
rnet.add_conv2d(3, 1, activation=sg.Activation.TANH, inputs=stem)
rnet.add_concat([block, -1])
rnet.add_global_avg_pool()
rnet.add_layer(3, sg.Activation.SOFTMAX, sg.Init.XAVIER)
check(len(rnet) == 9 and rnet.layer(4).inputs == (1, 3) and rnet.layer(6).inputs == (4, 5) and
      rnet.layer(5).inputs == (1,) and rnet.layer(6).shape == (6, 6, 7), f"graph layers: {rnet.layers}")
xr = rng.normal(size=(6, 72)).astype(np.float32)
check(np.allclose(rnet.forward(xr), np_graph(rnet, xr), atol=1e-5), "graph forward vs numpy")
yr = np.eye(3, dtype=np.float32)[np.arange(6) % 3]
before = rnet.evaluate(xr, yr).loss
rnet.train(xr, yr, epochs=40, optimizer=sg.Optimizer.ADAM, learning_rate=0.02)
check(rnet.evaluate(xr, yr).loss < before * 0.5, "graph network trains")
with tempfile.TemporaryDirectory() as d:
    path = os.path.join(d, "graph.slett")
    rnet.save(path)
    check(open(path, "rb").read()[6] == 6, "graphs are saved as format version 6")
    with sg.Network.load(path) as back:
        check(back.layers == rnet.layers and np.array_equal(back.forward(xr), rnet.forward(xr)), "graph save/load")
    with rnet.to_model(sg.Precision.INT8) as m:
        check(m.layers[3].type == sg.LayerType.ADD and m.layers[3].input_layers == (1, 3), f"graph model: {m.layers[3]}")
        p = m.predict(xr)
        check(np.array_equal(p[2], m(xr[2])) and np.abs(p - rnet.forward(xr)).max() < 0.1, "graph INT8 model")
spec = sg.Network(sg.Loss.MSE, [sg.Input(4, 4, 1), sg.Conv2D(1, 3, padding=1, activation=sg.Activation.NONE),
                                sg.Add([-1, 0]), sg.GlobalAvgPool(), sg.Layer(1, sg.Activation.NONE)])
check(spec.layer(2).type == sg.LayerType.ADD and spec.layer(2).inputs == (1, 0), f"graph from specs: {spec.layer(2)}")
for bad in ([50], [1, 2, 3] * 6):
    try:
        rnet.add_add(bad); check(False, f"inputs {bad} accepted")
    except (IndexError, ValueError):
        pass
try:
    sg.Network(sg.Loss.MSE, [sg.Input(4, 4, 2), sg.Conv2D(2, 3), sg.Add([0, 1])]); check(False, "mismatched add accepted")
except sg.SpingalettError:
    pass

# ONNX files and PyTorch weights (the files of Tests/Data), and PyTorch itself where it is installed
data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data")
with open(os.path.join(data_dir, "onnx_resnet.bin"), "rb") as f:
    head = np.frombuffer(f.read(16), dtype="<u4")
    count, n_in, n_out = int(head[1]), int(head[2]), int(head[3])
    xo = np.frombuffer(f.read(count * n_in * 4), dtype="<f4").reshape(count, n_in)
    yo = np.frombuffer(f.read(count * n_out * 4), dtype="<f4").reshape(count, n_out)
onnx_net = sg.Network.from_onnx(os.path.join(data_dir, "onnx_resnet.onnx"))
check(np.abs(onnx_net.forward(xo) - yo).max() < 1e-5 and onnx_net.loss == sg.Loss.CROSS_ENTROPY, "ONNX import")
with open(os.path.join(data_dir, "onnx_resnet.onnx"), "rb") as f:
    check(np.array_equal(sg.Network.from_onnx(f.read()).forward(xo), onnx_net.forward(xo)), "ONNX import from bytes")
try:
    sg.Network.from_onnx(os.path.join(data_dir, "onnx_unsupported.onnx")); check(False, "Resize imported")
except sg.SpingalettError as e:
    check("Resize" in str(e), f"unsupported operator message: {e}")
torch_like = sg.Network(sg.Loss.MSE, [sg.Input(8, 8, 3), sg.Conv2D(8, 3, padding=1, activation=sg.Activation.NONE),
                                      sg.BatchNorm(sg.Activation.RELU), sg.MaxPool2D(2),
                                      sg.Conv2D(8, 3, padding=1, groups=4, activation=sg.Activation.NONE),
                                      sg.BatchNorm(sg.Activation.RELU), sg.Layer(16, sg.Activation.TANH),
                                      sg.Layer(4, sg.Activation.NONE)])
with open(os.path.join(data_dir, "torch_cnn.bin"), "rb") as f:
    head = np.frombuffer(f.read(16), dtype="<u4")
    count, n_in, n_out = int(head[1]), int(head[2]), int(head[3])
    xt = np.frombuffer(f.read(count * n_in * 4), dtype="<f4").reshape(count, n_in)
    yt = np.frombuffer(f.read(count * n_out * 4), dtype="<f4").reshape(count, n_out)
for name in ("torch_cnn.pt", "torch_cnn.safetensors"):
    torch_like.load_pytorch(os.path.join(data_dir, name))
    check(np.abs(torch_like.forward(xt) - yt).max() < 1e-5, f"PyTorch weights from {name}")
try:
    import torch
except ImportError:
    torch = None
if torch is not None:
    torch.manual_seed(3)
    tm = torch.nn.Sequential(torch.nn.Conv2d(3, 8, 3, padding=1), torch.nn.BatchNorm2d(8), torch.nn.ReLU(),
                             torch.nn.Flatten(), torch.nn.Linear(8 * 6 * 6, 5), torch.nn.Softmax(1)).eval()
    tx = torch.rand(4, 3, 6, 6)
    with torch.no_grad():
        ty = tm(tx).numpy()
    hwc = tx.numpy().transpose(0, 2, 3, 1).reshape(4, -1)
    check(np.abs(sg.Network.from_torch(tm, tx[:1]).forward(hwc) - ty).max() < 1e-5, "Network.from_torch")
    same = sg.Network(sg.Loss.CROSS_ENTROPY, [sg.Input(6, 6, 3), sg.Conv2D(8, 3, padding=1, activation=sg.Activation.NONE),
                                              sg.BatchNorm(sg.Activation.RELU), sg.Layer(5, sg.Activation.SOFTMAX)])
    check(np.abs(same.load_pytorch(tm.state_dict()).forward(hwc) - ty).max() < 1e-5, "load_pytorch(state_dict)")

# transposed convolutions, upsampling and layer normalization: shapes, weights, a state dict
up = sg.Network(sg.Loss.MSE, [sg.Input(8, 8, 3), sg.Conv2D(8, 3, stride=2, padding=1),
                              sg.ConvTranspose2D(6, 3, stride=2, padding=1, output_padding=1, groups=2,
                                                 activation=sg.Activation.NONE),
                              sg.LayerNorm(sg.Activation.TANH), sg.GlobalAvgPool(), sg.Layer(4, sg.Activation.NONE)])
check([l.shape for l in up.layers][1:4] == [(4, 4, 8), (8, 8, 6), (8, 8, 6)], "transposed convolution shapes")
check(up.get_weights(1).shape == (6, 3, 3, 4) and up.get_weights(2).shape == (6,), "transposed and layer norm weights")
with open(os.path.join(data_dir, "torch_transposed.bin"), "rb") as f:
    head = np.frombuffer(f.read(16), dtype="<u4")
    count, n_in, n_out = int(head[1]), int(head[2]), int(head[3])
    xt = np.frombuffer(f.read(count * n_in * 4), dtype="<f4").reshape(count, n_in)
    yt = np.frombuffer(f.read(count * n_out * 4), dtype="<f4").reshape(count, n_out)
up.load_pytorch(os.path.join(data_dir, "torch_transposed.pt"))
check(np.abs(up.forward(xt) - yt).max() < 1e-5, "PyTorch weights of a ConvTranspose2d and a LayerNorm")
w = up.get_weights(1); up.set_weights(1, w); check(np.array_equal(up.get_weights(1), w), "transposed weights round trip")
ups = sg.Network(sg.Loss.MSE, [sg.Input(2, 3, 1), sg.Upsample2D(2), sg.Upsample2D(3, sg.Upsample.BILINEAR)])
check(ups.layers[-1].shape == (12, 18, 1) and ups.forward(np.ones(6)).shape == (216,), "upsampling shapes")

# the library versions the bindings accept: major.minor before 1.0, then the major and a minor at least theirs
check(sg._compatible("0.14.2", "0.14.0") and not sg._compatible("0.13.1", "0.14.0"), "versions of 0.x")
check(sg._compatible("1.0.0", "1.0.0") and sg._compatible("1.3.0", "1.2.5") and not sg._compatible("1.1.0", "1.2.0")
      and not sg._compatible("2.0.0", "1.2.0"), "versions of 1.x")

# lifetime
net = sg.Network(sg.Loss.MSE, [2, 3]); net.close(); net.close()
try: net.forward([0, 0]); check(False, "closed network usable")
except ValueError: pass
print(repr(sg.Network(sg.Loss.CROSS_ENTROPY, [3, sg.Layer(2, sg.Activation.SOFTMAX)])))
print("library:", sg.library_path())
print("ALL PASSED" if not failures else f"{len(failures)} FAILURES")
sys.exit(1 if failures else 0)
