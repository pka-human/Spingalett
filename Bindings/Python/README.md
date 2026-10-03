# Spingalett for Python

Thin `ctypes` bindings over the Spingalett shared library: no compiler is needed to install them,
only NumPy and a built `libspingalett` (`.so` / `.dylib` / `.dll`).

```bash
cmake -S . -B Build -DCMAKE_BUILD_TYPE=Release -DBUILD_WITH_OPENBLAS=ON
cmake --build Build --parallel            # produces Bin/libspingalett.so
pip install ./Bindings/Python
export SPINGALETT_LIBRARY=$PWD/Bin/libspingalett.so   # or put Bin/ on the library path
```

Without `SPINGALETT_LIBRARY` the module looks next to itself, then in the repository's `Bin/`
directory (when imported from a checkout), then on the system library path.

```python
import numpy as np
import spingalett as sg

x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y = np.array([[0], [1], [1], [0]], dtype=np.float32)

sg.set_compute_mode(sg.ComputeMode.OPENBLAS)   # falls back to single-threaded if unavailable
sg.seed(42)

layers = [
    sg.Layer(2),
    sg.Layer(16, sg.Activation.TANH, sg.Init.XAVIER, dropout=0.1),
    sg.Layer(1, sg.Activation.SIGMOID, sg.Init.XAVIER),
]
with sg.Network(sg.Loss.MSE, layers) as net:
    net.train(x, y, epochs=3000, optimizer=sg.Optimizer.ADAMW, learning_rate=0.02,
              weight_decay=1e-4, lr_scheduler=sg.WarmupCosine(warmup_epochs=100),
              callback=lambda net, progress: progress.train_loss < 1e-4, callback_interval=100)
    print(net.forward(x))          # one row per sample
    net.save("xor", precision=sg.Precision.FP16)

# validation, early stopping and the best epoch's weights
x_train, y_train = sg.load_idx("train-images-idx3-ubyte", "train-labels-idx1-ubyte", num_classes=10)
with sg.Network(sg.Loss.CROSS_ENTROPY, [784, sg.Layer(128, sg.Activation.RELU, sg.Init.HE),
                                        sg.Layer(10, sg.Activation.SOFTMAX, sg.Init.XAVIER)]) as net:
    result = net.train(x_train[:55000], y_train[:55000], epochs=30, strategy=sg.Strategy.MINI_BATCH,
                       batch_size=128, optimizer=sg.Optimizer.ADAMW, learning_rate=1e-3,
                       validation_data=(x_train[55000:], y_train[55000:]), monitor=sg.Monitor.VAL_ACCURACY,
                       early_stopping_patience=3, restore_best_weights=True)
    print(result.status.name, result.best_epoch, net.evaluate(x_train[55000:], y_train[55000:]))

with sg.Network.load("xor.slett") as net:
    print(net.topology, net.forward([1, 0]))
```

| API | Notes |
|---|---|
| `Network(loss, layers)` / `add_layer(...)` | first layer is the input layer |
| `forward(x)` | 1-D input -> vector, 2-D batch -> matrix (one batched `predict()` call; results are copies) |
| `train(x, y, config=None, validation_data=None, **overrides)` | fields of `TrainConfig` (e.g. `epochs`, `strategy`, `batch_size`, `shuffle`, `early_stopping_patience`, `restore_best_weights`); returns a `TrainResult`; `callback(network, progress)` gets a `Progress`; exceptions raised in callbacks stop training and are re-raised |
| `train_from_generator(fn, samples_per_epoch=0, validation_data=None, ...)` | `fn(inputs, targets)` fills the given arrays and returns the number of rows; 0 ends the epoch |
| `evaluate(x, y)` | `Metrics(loss, accuracy)` |
| `Trainer(net, max_batch)` | `forward(x)`, `backward(y)`, `backward_output_grads(dl_dout)` for custom losses, `step(optimizer=..., learning_rate=...)`, `zero_grad()`, `train_on_batch(x, y, ...)`; gradients via `net.get_weight_gradients(i)` / `get_bias_gradients(i)` |
| `load_idx(images, labels, num_classes=0)`, `load_csv(path, target_columns=1, num_classes=0)` | return `(inputs, targets)` float32 arrays |
| `save_dataset(path, x, y, input_encoding=AUTO, target_encoding=AUTO, compress=True)`, `load_dataset(path)`, `dataset_info(path)` | `.slettd` data set files |
| `train_from_file(path, shuffle=True, ...)` | trains on a `.slettd` file streamed by the C reader, one chunk in memory |
| `CosineDecay`, `LinearWarmup`, `StepDecay`, `WarmupCosine` | built-in schedules; any `fn(epoch, total, initial_lr)` works too |
| `get_weights(i)`, `set_weights(i, w)`, `get_biases(i)`, `set_biases(i, b)` | weight matrix `i` connects layer `i` to `i + 1`, shape `(out, in)` |
| `save(path, precision, save_optimizer)`, `Network.load(path)` | `.slett` files, shared with the C API |
| `set_compute_mode`, `set_num_threads`, `seed`, `set_verbose`, `set_log_level`, `set_log_callback` | process-wide settings |

Library errors raise `SpingalettError`, whose `code` is an `ErrorCode`. The bindings check on import
that the library has the same major.minor version (`library_version()`), because they mirror its
struct layouts. A `Network` is not thread-safe: use one per thread.
