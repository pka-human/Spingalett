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
    check(abs(sg.LinearDecay(min_lr=0.1)(50, 100, 1.0) - 0.55) < 1e-6 and
          abs(sg.WarmupLinear(4, 0.1)(54, 104, 1.0) - 0.55) < 1e-6, "linear schedule values")
    # schedules by step, with gradient accumulation: steps of 2 mini-batches of 1 over the 4 samples, 3 epochs
    calls, steps = [], net.time_step
    net.train(x, y, epochs=3, strategy=sg.Strategy.MINI_BATCH, batch_size=1, accumulation_steps=2, lr_per_step=True,
              lr_scheduler=lambda e, t, lr: (calls.append((e, t)), lr)[1])
    check(calls == [(e, 6) for e in range(6)] and net.time_step == steps + 6, f"schedule by step {calls}")
    net.train(x, y, epochs=2, strategy=sg.Strategy.MINI_BATCH, batch_size=2, accumulation_steps=2,
              lr_per_step=True, lr_scheduler=sg.WarmupLinear(warmup_epochs=1))
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
# checkpoints: a run stopped by its callback, or by the end of the process after an autosave, and carried on
# trains the bits of one that never stopped; train_from_file's reader resumes where the run's stood
with tempfile.TemporaryDirectory() as d:
    ck = os.path.join(d, "run.ckpt")
    def drop_net():
        sg.seed(13)
        return sg.Network(sg.Loss.CROSS_ENTROPY, [3, sg.Layer(8, sg.Activation.RELU, sg.Init.HE, dropout=0.2),
                                                  sg.Layer(2, sg.Activation.SOFTMAX, sg.Init.XAVIER)])
    cfg = dict(epochs=5, strategy=sg.Strategy.MINI_BATCH, batch_size=8, optimizer=sg.Optimizer.ADAM, learning_rate=0.02)
    def stop_at(epoch, path=None):
        def cb(n, p):
            if p.epoch == epoch and path:
                n.save_checkpoint(path)
            return p.epoch == epoch
        return cb
    data = os.path.join(d, "t.slettd")
    sg.save_dataset(data, vx, ty)
    for name, run in (("arrays", lambda n, **k: n.train(vx, ty, **k)), ("file", lambda n, **k: n.train_from_file(data, **k))):
        with drop_net() as a:
            a.train(vx, ty, **cfg) if name == "arrays" else a.train_from_file(data, **cfg)
            want, steps = a.get_weights(0), a.time_step
        with drop_net() as b:
            if name == "arrays":
                r = run(b, callback=stop_at(2, ck), **cfg)
            else:
                r = run(b, callback=stop_at(3), autosave_mode=sg.AutoSave.OVERWRITE, autosave_interval=1,
                        autosave_path=ck, **cfg)
            check(r.status == sg.TrainStatus.INTERRUPTED, f"checkpoint: the stopped run ({name}) {r}")
        sg.seed(99)
        with sg.Network.load_checkpoint(ck) as c:
            r = run(c, **cfg)
            check(r.epochs_run == (3 if name == "arrays" else 2) and c.time_step == steps and
                  np.array_equal(c.get_weights(0), want), f"checkpoint: resumed ({name}) == never stopped {r}")
    with sg.Network.load_checkpoint(ck) as c:
        check(c.train(vx, ty, **cfg).epochs_run == 2, "checkpoint: a resumed run of the epochs left")
        check(c.train(vx, ty, **cfg).epochs_run == 5, "checkpoint: the call after it is a run of its own")
    with sg.Network.load(ck) as m:
        check(m.time_step > 0 and m.get_weights(0).shape == (8, 3), "checkpoint: read as a model")
check(sg.library_version() == sg.__version__, f"library {sg.library_version()} vs bindings {sg.__version__}")
check(sg.cpu_kernels() in ("AVX-512", "AVX2", "AVX", "SSE2", "NEON", "C"), f"cpu kernels {sg.cpu_kernels()!r}")
check(sg.gpu_device() is None or isinstance(sg.gpu_device(), str), f"gpu device {sg.gpu_device()!r}")
check(sg.cuda_device() is None or isinstance(sg.cuda_device(), str), f"cuda device {sg.cuda_device()!r}")
sg.set_gpu_precision(sg.Precision.BFLOAT16)
check(sg.get_gpu_precision() == sg.Precision.BFLOAT16, "gpu precision bfloat16")
sg.set_gpu_precision(sg.Precision.FLOAT32)

# the GPUs (or, without one, the CPU they fall back to) train and predict as the CPU does
gx = np.random.default_rng(3).normal(size=(64, 6)).astype(np.float32)
gy = np.eye(3, dtype=np.float32)[np.arange(64) % 3]
results = []
for mode in (sg.ComputeMode.OPENMP, sg.ComputeMode.CUDA, sg.ComputeMode.VULKAN):
    sg.set_compute_mode(mode)
    sg.seed(4)
    with sg.Network(sg.Loss.CROSS_ENTROPY, [6, sg.Layer(8, sg.Activation.RELU), sg.Layer(3, sg.Activation.SOFTMAX)]) as gn:
        gn.train(gx, gy, epochs=3, strategy=sg.Strategy.MINI_BATCH, batch_size=16, shuffle=False)
        results.append(gn.forward(gx))
check(sg.get_compute_mode() == sg.ComputeMode.VULKAN, "compute mode VULKAN")
check(all(np.allclose(results[0], r, rtol=1e-4, atol=1e-6) for r in results[1:]), "GPU training == CPU training")

# data sets in the GPU's memory: the same bits as arrays, on the GPUs (a Vulkan device's set copied through the
# host for CUDA) and on the CPU
if sg.gpu_device() is not None:
    with sg.DeviceData(gx) as dx, sg.DeviceData(gy) as dy:
        check(len(dx) == 64 and dx.shape == (64, 6) and np.array_equal(dx.numpy(), gx)
              and np.array_equal(dy.numpy(5, 2), gy[5:7]), f"device data {dx!r}")
        for mode in (sg.ComputeMode.VULKAN, sg.ComputeMode.CUDA, sg.ComputeMode.OPENMP):
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

# GPT-2 (Hugging Face's Conv1D state dict, nanoGPT's nn.Linear one) and LLaMA-like models (grouped queries, rotary
# embeddings, SwiGLU, Qwen2's biases) against references written in PyTorch
if torch is not None:
    import torch.nn.functional as tf

    def ref_gpt2(sd, tokens, heads, conv1d, act):
        T, d = len(tokens), sd["transformer.wte.weight"].shape[1]
        lin = lambda x, n: x @ sd[n + ".weight"] + sd[n + ".bias"] if conv1d else x @ sd[n + ".weight"].T + sd[n + ".bias"]
        x = sd["transformer.wte.weight"][tokens] + sd["transformer.wpe.weight"][:T]
        layers = 1 + max(int(k.split(".")[2]) for k in sd if k.startswith("transformer.h."))
        for i in range(layers):
            h = f"transformer.h.{i}."
            y = tf.layer_norm(x, (d,), sd[h + "ln_1.weight"], sd[h + "ln_1.bias"], 1e-5)
            q, k, v = lin(y, h + "attn.c_attn").split(d, -1)
            q, k, v = (t.view(T, heads, d // heads).transpose(0, 1) for t in (q, k, v))
            a = tf.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(0, 1).reshape(T, d)
            x = x + lin(a, h + "attn.c_proj")
            y = tf.layer_norm(x, (d,), sd[h + "ln_2.weight"], sd[h + "ln_2.bias"], 1e-5)
            x = x + lin(act(lin(y, h + "mlp.c_fc")), h + "mlp.c_proj")
        x = tf.layer_norm(x, (d,), sd["transformer.ln_f.weight"], sd["transformer.ln_f.bias"], 1e-5)
        return x @ sd["transformer.wte.weight"].T

    def ref_llama(sd, tokens, heads, kv, theta, eps):
        T, d = len(tokens), sd["model.embed_tokens.weight"].shape[1]
        hd = d // heads
        rms = lambda x, w: x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w
        inv = 1.0 / theta ** (torch.arange(0, hd, 2, dtype=torch.float64) / hd)
        f = torch.outer(torch.arange(T, dtype=torch.float64), inv)
        cos, sin = torch.cat([f, f], -1).cos().float(), torch.cat([f, f], -1).sin().float()
        rot = lambda x: torch.cat([-x[..., hd // 2:], x[..., :hd // 2]], -1)
        lin = lambda x, n: x @ sd[n + ".weight"].T + (sd[n + ".bias"] if n + ".bias" in sd else 0)
        x = sd["model.embed_tokens.weight"][tokens]
        layers = 1 + max(int(k.split(".")[2]) for k in sd if k.startswith("model.layers."))
        for i in range(layers):
            h = f"model.layers.{i}."
            y = rms(x, sd[h + "input_layernorm.weight"])
            q = lin(y, h + "self_attn.q_proj").view(T, heads, hd).transpose(0, 1)
            k = lin(y, h + "self_attn.k_proj").view(T, kv, hd).transpose(0, 1)
            v = lin(y, h + "self_attn.v_proj").view(T, kv, hd).transpose(0, 1)
            q, k = q * cos + rot(q) * sin, k * cos + rot(k) * sin
            k, v = k.repeat_interleave(heads // kv, 0), v.repeat_interleave(heads // kv, 0)
            a = tf.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(0, 1).reshape(T, d)
            x = x + lin(a, h + "self_attn.o_proj")
            y = rms(x, sd[h + "post_attention_layernorm.weight"])
            x = x + lin(tf.silu(lin(y, h + "mlp.gate_proj")) * lin(y, h + "mlp.up_proj"), h + "mlp.down_proj")
        return rms(x, sd["model.norm.weight"]) @ sd.get("lm_head.weight", sd["model.embed_tokens.weight"]).T

    torch.manual_seed(11)
    V, T, d, heads = 50, 12, 32, 4
    for conv1d in (True, False):
        r = lambda *shape: torch.randn(*shape) * 0.2
        sd = {"transformer.wte.weight": r(V, d), "transformer.wpe.weight": r(20, d),
              "transformer.ln_f.weight": 1 + r(d), "transformer.ln_f.bias": r(d)}
        for i in range(2):
            h = f"transformer.h.{i}."
            for name, (a, b) in {"attn.c_attn": (d, 3 * d), "attn.c_proj": (d, d), "mlp.c_fc": (d, 4 * d),
                                 "mlp.c_proj": (4 * d, d)}.items():
                sd[h + name + ".weight"] = r(a, b) if conv1d else r(b, a)
                sd[h + name + ".bias"] = r(b)
            for n in ("ln_1", "ln_2"):
                sd[h + n + ".weight"], sd[h + n + ".bias"] = 1 + r(d), r(d)
        tokens = torch.randint(0, V, (T,))
        act = (lambda x: tf.gelu(x, approximate="tanh")) if conv1d else tf.gelu
        want = ref_gpt2(sd, tokens, heads, conv1d, act).numpy()
        gpt = sg.gpt2_from_state_dict(sd, heads, context=T)
        got = gpt.forward(tokens.numpy().astype(np.float32)).reshape(T, V)
        check(np.abs(got - want).max() < 2e-4, f"gpt2_from_state_dict ({'Conv1D' if conv1d else 'Linear'}): "
              f"{np.abs(got - want).max():.2e} from PyTorch")
        # the head tied to the token embeddings: one table, trained by both, saved and loaded so
        check(gpt.layer(-1).tied == 1 and gpt.layer(-1).weight_count == 0, f"gpt2 head tied: {gpt.layer(-1)}")
        back = sg.Network.from_bytes(gpt.to_bytes())
        check(back.layer(-1).tied == 1 and np.array_equal(back.forward(tokens.numpy().astype(np.float32)).reshape(T, V), got),
              "tied gpt2 saved and loaded")
        model = gpt.to_model(sg.Precision.FLOAT32)
        check(model.layers[-1].tied == 1 and np.abs(model.predict(tokens.numpy().astype(np.float32)).reshape(T, V) - got).max() < 1e-4,
              "tied gpt2 as a model")
    for qwen in (False, True):
        kv, ff, theta = 2, 48, 500000.0
        sd = {"model.embed_tokens.weight": r(V, d), "model.norm.weight": 1 + r(d), "lm_head.weight": r(V, d)}
        for i in range(2):
            h = f"model.layers.{i}."
            sd.update({h + "self_attn.q_proj.weight": r(d, d), h + "self_attn.k_proj.weight": r(kv * d // heads, d),
                       h + "self_attn.v_proj.weight": r(kv * d // heads, d), h + "self_attn.o_proj.weight": r(d, d),
                       h + "mlp.gate_proj.weight": r(ff, d), h + "mlp.up_proj.weight": r(ff, d),
                       h + "mlp.down_proj.weight": r(d, ff), h + "input_layernorm.weight": 1 + r(d),
                       h + "post_attention_layernorm.weight": 1 + r(d)})
            if qwen:
                for n, m in (("q", d), ("k", kv * d // heads), ("v", kv * d // heads)):
                    sd[h + f"self_attn.{n}_proj.bias"] = r(m)
        if qwen:
            del sd["lm_head.weight"]        # (tie_word_embeddings: no head of its own)
        tokens = torch.randint(0, V, (T,))
        want = ref_llama(sd, tokens, heads, kv, theta, 1e-6).numpy()
        lm = sg.llama_from_state_dict(sd, heads, kv, context=T, rope_theta=theta)
        check(lm.layer(-1).tied == (1 if qwen else 0), f"llama head tied {qwen}: {lm.layer(-1).tied}")
        got = lm.forward(tokens.numpy().astype(np.float32)).reshape(T, V)
        check(np.abs(got - want).max() < 2e-4, f"llama_from_state_dict ({'with' if qwen else 'no'} biases): "
              f"{np.abs(got - want).max():.2e} from PyTorch")

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

# a transformer: a token file's windows (nanoGPT's uint16), the sequence learned, its continuation generated
import tempfile
T, V = 16, 13
lm = sg.Network(sg.Loss.SPARSE_CROSS_ENTROPY)
lm.add_layer(T)
lm.add_embedding(V, 16); h = lm.last
lm.add_rms_norm(); lm.add_linear(48)
lm.add_attention(heads=4, kv_heads=2, rope_theta=10000.0, causal=True)
lm.add_linear(16); lm.add_add([h, lm.last])
lm.add_linear(V)
check(lm.target_size == T and lm.output_size == T * V, "language model sizes")
tokens_path = os.path.join(tempfile.mkdtemp(), "tokens.bin")
np.array([i % V for i in range(3000)], dtype=np.uint16).tofile(tokens_path)
sg.seed(3)
result = lm.train_from_tokens(tokens_path, epochs=12, learning_rate=1e-2, optimizer=sg.Optimizer.ADAM, batch_size=32,
                              stride=5)
check(result.status == sg.TrainStatus.COMPLETED and result.train_loss < 0.5, f"train_from_tokens ({result.train_loss:.3f})")
check(lm.generate([3, 4, 5], 20) == [(6 + i) % V for i in range(20)], "generate: the greedy continuation")
drawn = lm.generate([3, 4, 5], 20, temperature=1.0, top_p=0.9, seed=8)
check(drawn == lm.generate([3, 4, 5], 20, temperature=1.0, top_p=0.9, seed=8) and len(drawn) == 20,
      "generate: draws repeat with their seed")
check(lm.generate([3, 4, 5], 20, stop=[9]) == [6, 7, 8, 9], "generate: stop tokens")
os.remove(tokens_path)

# sequences of different lengths: attention ignoring the keys of padding (-1), its outputs at the real tokens
# those of the sequence alone (the padding's values do not matter: here, the network's cells past the sequence)
enc = sg.Network(sg.Loss.SPARSE_CROSS_ENTROPY, [8, sg.Embedding(V, 8, positions=True), sg.Linear(24),
                                                sg.Attention(heads=2, ignore_padding=True), sg.Linear(V)])
check(enc.layer(3).ignore_padding and enc.to_model(sg.Precision.FLOAT32).layers[2].ignore_padding,
      "ignore_padding described")
short = np.array([4, 7, 1, -1, -1, -1, -1, -1], dtype=np.float32)
out = enc.forward(short).reshape(8, V)
enc.set_biases(0, np.concatenate([enc.get_biases(0)[:24], np.zeros(40, np.float32)]))   # the padding's positions
check(np.allclose(enc.forward(short).reshape(8, V)[:3], out[:3], atol=1e-6), "padding ignored by attention")
r = enc.train(np.stack([short, np.arange(8, dtype=np.float32) % V]), np.stack([short, np.arange(8, dtype=np.float32) % V]),
              epochs=2, strategy=sg.Strategy.FULL_BATCH)
check(r.status == sg.TrainStatus.COMPLETED, "a padded batch trains")

# a sliding window: with queries and keys of zeros and the positions' one-hot vectors as values, a cell's outputs
# are the mean of the keys' it attends to (itself and the one before it)
win = sg.Network(sg.Loss.MSE, [6, sg.Embedding(6, 18), sg.Attention(heads=1, causal=True, window=2)])
table = np.zeros((6, 18), np.float32)
table[np.arange(6), 12 + np.arange(6)] = 1.0
win.set_weights(0, table.ravel())
expect = np.array([[1.0 if j == i or (j == i - 1 and i > 0) else 0.0 for j in range(6)] for i in range(6)])
expect /= expect.sum(axis=1, keepdims=True)
check(win.layer(2).window == 2 and win.to_model(sg.Precision.FLOAT32).layers[1].window == 2, "window described")
check(np.allclose(win.forward(np.arange(6, dtype=np.float32)).reshape(6, 6), expect, atol=1e-6), "window attended to")

# an encoder-decoder: 6 source tokens then 5 target tokens in one input, cross-attention from the decoder to the
# encoder; it learns to copy the source's first tokens, saves and loads, and deploys
S, Q = 6, 5
seq = sg.Network(sg.Loss.SPARSE_CROSS_ENTROPY, [S + Q])
seq.add_embedding(V, 16, positions=True, token_count=S).add_linear(48).add_attention(heads=2).add_layer_norm()
encoder = seq.last
seq.add_linear(32, inputs=[encoder])
kv = seq.last
seq.add_embedding(V, 16, positions=True, token_offset=S, inputs=[0]).add_linear(48)
seq.add_attention(heads=2, causal=True).add_linear(16)
seq.add_attention(heads=2, inputs=[seq.last, kv]).add_linear(V)
check(seq.layer(1).token_offset == 0 and seq.layer(1).shape[1] == S and seq.layer(6).token_offset == S and
      seq.layer(6).shape[1] == Q and seq.layer(10).inputs == (9, kv), "an encoder-decoder described")
rng = np.random.default_rng(3)
src = rng.integers(0, V, (256, S)).astype(np.float32)
xs = np.concatenate([src, np.zeros((256, Q), np.float32)], axis=1)
xs[:, S + 1:] = src[:, :Q - 1]                      # the target's tokens so far: the source's, shifted
r = seq.train(xs, src[:, :Q], epochs=100, batch_size=32, strategy=sg.Strategy.MINI_BATCH, learning_rate=1e-2,
              optimizer=sg.Optimizer.ADAM)
m = seq.evaluate(xs, src[:, :Q])
check(r.status == sg.TrainStatus.COMPLETED and m.accuracy > 0.9, f"an encoder-decoder learns to copy ({m.accuracy:.3f})")
back = sg.Network.from_bytes(seq.to_bytes())
check(np.array_equal(back.forward(xs[:4]), seq.forward(xs[:4])) and back.layer(6).token_offset == S,
      "an encoder-decoder saved and loaded")
model = seq.to_model(sg.Precision.FLOAT32)
check(np.allclose(model.predict(xs[:4]), seq.forward(xs[:4]), atol=1e-5) and model.layers[5].token_offset == S,
      "an encoder-decoder deployed")
# activation recomputation trains to the same weights
copies = [sg.Network.from_bytes(seq.to_bytes()) for _ in range(2)]
for net, again in zip(copies, (False, True)):
    sg.seed(4)
    net.train(xs[:64], src[:64, :Q], epochs=2, batch_size=16, strategy=sg.Strategy.MINI_BATCH, learning_rate=1e-3,
              optimizer=sg.Optimizer.ADAMW, recompute=again)
check(all(np.array_equal(copies[0].get_weights(i), copies[1].get_weights(i)) for i in range(len(seq) - 1)),
      "activation recomputation: the same weights")
# sessions: the model a token at a time with a key-value cache, the logits of its batched prediction bit for bit
for precision in (sg.Precision.FLOAT32, sg.Precision.INT8):
    model = lm.to_model(precision)
    window = [(5 + i) % V for i in range(T)]
    whole = model.predict(np.array(window, dtype=np.float32)).reshape(T, V)
    with model.session(sequences=3) as s:
        check(s.vocabulary == V and s.length() == 0, "session: vocabulary and length")
        alone = np.array([s.feed([t]) for t in window])
        check(np.array_equal(alone, whole) and s.length() == T, f"session {precision.name}: a token at a time")
        s.rewind(4)
        rest, both = s.feed_many([(0, window[4:]), (1, window)], all=True)
        check(np.array_equal(rest, whole[4:]) and np.array_equal(both, whole),
              f"session {precision.name}: two sequences in one call, after a rewind")
        s.copy(1, 2); s.rewind(9, sequence=2)
        check(np.array_equal(s.feed(window[9:], sequence=2), whole[-1]), f"session {precision.name}: a copy")
        for b in range(3): s.rewind(0, sequence=b)
        together = s.generate([[3, 4, 5], [7], [1, 2]], 8, temperature=0.8, top_k=5, seed=21)
    alone = []
    for b, prompt in enumerate([[3, 4, 5], [7], [1, 2]]):
        with model.session() as one:
            alone += one.generate([prompt], 8, temperature=0.8, top_k=5, seed=21 + b)
    check(together == alone and all(len(t) == 8 for t in together), f"session {precision.name}: batched generation")
    with model.session() as s:
        check(s.generate([[3, 4, 5]], 20)[0] == [(6 + i) % V for i in range(20)][:T - 3 + 1],
              f"session {precision.name}: greedy generation stops after a full sequence")
try:
    sg.Network(sg.Loss.MSE, [4, 3]).to_model(sg.Precision.FLOAT32).session(); check(False, "a session of an MLP")
except sg.SpingalettError as e:
    check("causal language model" in str(e), "a session of an MLP is refused")
# tokenizers: the ids and texts of Hugging Face's tokenizers (Tests/make_tokenizer_tests.py), and of sentencepiece for a
# GGUF vocabulary
def expectations(name):
    lines = [line.rstrip("\n") for line in open(os.path.join(data_dir, name), encoding="ascii") if not line.startswith("#")]
    unhex = lambda h: "" if h == "-" else bytes.fromhex(h).decode("utf-8")
    for k in range(0, len(lines) - 3, 4):
        special, add, text = lines[k].split()
        ids = [int(x) for x in lines[k + 1].split()[1:]]
        yield bool(int(special)), bool(int(add)), unhex(text), ids, \
            None if lines[k + 2] == "*" else unhex(lines[k + 2]), unhex(lines[k + 3])


for tokenizer_file, expected_file in [(f"tokenizer_{n}.json", f"tokenizer_{n}.txt") for n in
                                      ("gpt2", "llama3", "qwen2", "o200k", "spm", "metaspace")] + \
                                     [("gguf_llama.gguf", "tokenizer_llama3.txt"), ("gguf_qwen2.gguf", "tokenizer_qwen2.txt"),
                                      ("gguf_spm.gguf", "gguf_spm.txt")]:
    with sg.Tokenizer(os.path.join(data_dir, tokenizer_file)) as tok:
        wrong = 0
        for special, add, text, ids, decoded, decoded_skip in expectations(expected_file):
            wrong += tok.encode(text, special=special, add_special=add) != ids
            wrong += (decoded is not None and tok.decode(ids) != decoded) or tok.decode(ids, skip_special=True) != decoded_skip
        check(wrong == 0, f"Tokenizer({tokenizer_file}): {wrong} encodings or decodings differ from the reference's")
with open(os.path.join(data_dir, "tokenizer_llama3.json"), "rb") as f:
    tok = sg.Tokenizer.from_bytes(f.read())
check(tok.bos == 420 and tok.eos == 421 and tok.unk is None and len(tok) == 425 and tok.added == 5, f"tokenizer info {tok!r}")
check(tok.id("<|begin_of_text|>") == 420 and tok.token(420) == "<|begin_of_text|>" and tok.id("no such token") is None
      and tok.token(10 ** 6) is None, "tokens by id and by name")
check(tok.encode("<|begin_of_text|>", special=True) == [420] and len(tok.encode("<|begin_of_text|>")) > 1,
      "special tokens only when asked")
long_text = "Tokenizers cut text into pieces, " * 300
check(tok.decode(tok.encode(long_text)) == long_text, "a long text encoded and decoded")
tok.close()
try:
    sg.Tokenizer(os.path.join(data_dir, "no_such_tokenizer.json"))
    check(False, "a missing tokenizer loaded")
except sg.SpingalettError:
    pass

# pretrained models: Hugging Face's directories and GGUF files against PyTorch's logits (Tests/make_pretrained_tests.py)
def window(name):
    with open(os.path.join(data_dir, name + ".bin"), "rb") as f:
        head = np.frombuffer(f.read(16), dtype="<u4")
        t, n = int(head[2]), int(head[3])
        return np.frombuffer(f.read(4 * t), dtype="<f4"), np.frombuffer(f.read(4 * n), dtype="<f4")


for path, bin_name in [("hf_llama", "hf_llama"), ("hf_qwen2", "hf_qwen2"), ("hf_mistral", "hf_mistral"), ("hf_gpt2", "hf_gpt2"),
                       ("gguf_llama.gguf", "hf_llama"), ("gguf_qwen2.gguf", "hf_qwen2"), ("gguf_gpt2.gguf", "hf_gpt2"),
                       ("gguf_quant.gguf", "gguf_quant")]:
    tokens, logits = window(bin_name)
    with sg.Network.from_pretrained(os.path.join(data_dir, path), context=len(tokens)) as lm:
        got = lm.forward(tokens)
        scale = max(1.0, float(np.abs(logits).max()))
        check(lm.loss == sg.Loss.SPARSE_CROSS_ENTROPY and np.abs(got - logits).max() < 1e-4 * scale,
              f"Network.from_pretrained({path}): {np.abs(got - logits).max():.2e} from PyTorch")
# GGUF files as deployment models (Qwen3's normalized queries and keys, LLaMA 3's rotary frequencies, matrices of
# fifteen types computed in their blocks), a session's logits batched prediction's
for path, bin_name, tolerance in [("gguf_qwen3.gguf", "hf_qwen3", 1e-4), ("gguf_llama3.gguf", "hf_llama3", 1e-4),
                                  ("gguf_gemma3.gguf", "hf_gemma3", 1e-4),
                                  ("gguf_quant.gguf", "gguf_quant", 0.03)]:
    tokens, logits = window(bin_name)
    with sg.Model.load(os.path.join(data_dir, path), context=len(tokens)) as model:
        got = model.predict(tokens[None, :])[0]
        error = np.abs(got - logits).max() if tolerance < 1e-3 else \
            np.sqrt(((got - logits) ** 2).sum() / (logits ** 2).sum())
        with model.session() as s:
            fed = s.feed(tokens.astype(int).tolist(), all=True)
        check(error < tolerance and np.array_equal(np.asarray(fed).reshape(-1), got),
              f"Model.load({path}): {error:.2e} from PyTorch, or the session's logits differ")
with sg.Tokenizer(os.path.join(data_dir, "hf_llama")) as tok:
    tokens, _ = window("hf_llama")
    check(tok.encode("The quick brown fox jumps over the lazy dog, it's 1234 times!")[:len(tokens)] == tokens.astype(int).tolist()
          and tok.eos == 424, "the tokenizer of a model directory")
try:
    sg.Network.from_pretrained(os.path.join(data_dir, "no_such_model"))
    check(False, "a missing model loaded")
except sg.SpingalettError:
    pass
# the C loader against the 1.2 loader of state dicts, where PyTorch and safetensors are installed: the same network
try:
    from safetensors.torch import load_file as load_safetensors
except ImportError:
    load_safetensors = None
if torch is not None and load_safetensors is not None:
    d = os.path.join(data_dir, "hf_llama")
    sd = {}
    for f in sorted(os.listdir(d)):
        if f.endswith(".safetensors"):
            sd.update(load_safetensors(os.path.join(d, f)))
    tokens, _ = window("hf_llama")
    py = sg.llama_from_state_dict(sd, 4, 2, context=len(tokens), rope_theta=500000.0, epsilon=1e-5).forward(tokens)
    check(np.array_equal(py, sg.Network.from_pretrained(d, context=len(tokens)).forward(tokens)),
          "Network.from_pretrained and llama_from_state_dict give the same logits")

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
