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

# a convolutional network: images as rows of height x width x channels floats (channels last)
cnn = sg.Network(sg.Loss.CROSS_ENTROPY, [
    sg.Input(28, 28, 1),
    sg.Conv2D(32, 3, padding=1), sg.MaxPool2D(2),             # ReLU and He initialization by default
    sg.Conv2D(64, 3, padding=1), sg.MaxPool2D(2),
    sg.Layer(128, sg.Activation.RELU, sg.Init.HE, dropout=0.3),
    sg.Layer(10, sg.Activation.SOFTMAX, sg.Init.XAVIER),
])
cnn.train(x_train[:55000], y_train[:55000], epochs=2, strategy=sg.Strategy.MINI_BATCH, batch_size=128,
          optimizer=sg.Optimizer.ADAMW, learning_rate=1e-3)
print([(l.type.name, l.shape) for l in cnn.layers], cnn.get_weights(0).shape)   # filters x 3 x 3 x 1

# batch normalization, a depthwise-separable block and image augmentation on CIFAR-10
x_cifar, y_cifar = sg.load_cifar([f"cifar-10-batches-bin/data_batch_{i}.bin" for i in range(1, 6)])
bn = sg.Network(sg.Loss.CROSS_ENTROPY, [
    sg.Input(32, 32, 3),
    sg.Conv2D(32, 3, padding=1, activation=sg.Activation.NONE), sg.BatchNorm(sg.Activation.RELU),
    sg.Conv2D(32, 3, padding=1, groups=32, activation=sg.Activation.NONE), sg.BatchNorm(sg.Activation.RELU),
    sg.Conv2D(64, 1, activation=sg.Activation.NONE), sg.BatchNorm(sg.Activation.RELU), sg.MaxPool2D(2),
    sg.Layer(10, sg.Activation.SOFTMAX, sg.Init.XAVIER),
])
bn.train(x_cifar, y_cifar, epochs=5, strategy=sg.Strategy.MINI_BATCH, batch_size=128,
         optimizer=sg.Optimizer.ADAMW, learning_rate=2e-3, augment_shift=4, augment_flip=True)

with sg.Network.load("xor.slett") as net:
    print(net.topology, net.forward([1, 0]))
    # deployment: a read-only INT8 model with integer kernels, and a C header for firmware
    with net.to_model(sg.Precision.INT8) as model:
        print(model.predict([1, 0]), model.layers, model.size)
    net.export_c_header("xor_model.h", "xor_model", sg.Precision.INT8)
```

| API | Notes |
|---|---|
| `Network(loss, layers)` / `add_layer(...)` | first layer is the input layer; `layers` may mix `Layer`, `Input`, `Conv2D`, `MaxPool2D`, `AvgPool2D`, `BatchNorm` and plain widths |
| `add_input(h, w, c)`, `add_conv2d(filters, kernel, stride=1, padding=0, activation=RELU, init=HE, dropout=0, groups=1)`, `add_max_pool2d(kernel, stride=0, padding=0)`, `add_avg_pool2d(...)` | convolution (grouped or depthwise with `groups`) and pooling over channels-last tensors; pooling's stride defaults to the kernel size |
| `add_batch_norm(activation=NONE, epsilon=1e-5, momentum=0.1, dropout=0)` | per-channel normalization of the previous layer: batch statistics while training, running averages otherwise; `get_running_statistics(i)` / `set_running_statistics(i, mean, var)` |
| `layers`, `layer(i)`, `topology` | `LayerDescription(type, shape, outputs, activation, dropout, kernel, stride, padding, weight_count, bias_count, groups, epsilon, momentum)` per layer |
| `forward(x)` | 1-D input -> vector, 2-D batch -> matrix (one batched `predict()` call; results are copies) |
| `train(x, y, config=None, validation_data=None, **overrides)` | fields of `TrainConfig` (e.g. `epochs`, `strategy`, `batch_size`, `shuffle`, `early_stopping_patience`, `restore_best_weights`, `augment_shift`, `augment_flip`); returns a `TrainResult`; `callback(network, progress)` gets a `Progress`; exceptions raised in callbacks stop training and are re-raised |
| `train_from_generator(fn, samples_per_epoch=0, validation_data=None, ...)` | `fn(inputs, targets)` fills the given arrays and returns the number of rows; 0 ends the epoch |
| `evaluate(x, y)` | `Metrics(loss, accuracy)` |
| `Trainer(net, max_batch)` | `forward(x)`, `backward(y)`, `backward_output_grads(dl_dout)` for custom losses, `step(optimizer=..., learning_rate=...)`, `zero_grad()`, `train_on_batch(x, y, ...)`; gradients via `net.get_weight_gradients(i)` / `get_bias_gradients(i)` |
| `load_idx(images, labels, num_classes=0)`, `load_cifar(paths, num_classes=10)`, `load_csv(path, target_columns=1, num_classes=0)` | return `(inputs, targets)` float32 arrays; CIFAR images as 32 x 32 x 3, channels last |
| `save_dataset(path, x, y, input_encoding=AUTO, target_encoding=AUTO, compress=True)`, `load_dataset(path)`, `dataset_info(path)` | `.slettd` data set files |
| `train_from_file(path, shuffle=True, ...)` | trains on a `.slettd` file streamed by the C reader, one chunk in memory |
| `CosineDecay`, `LinearWarmup`, `StepDecay`, `WarmupCosine` | built-in schedules; any `fn(epoch, total, initial_lr)` works too |
| `get_weights(i)`, `set_weights(i, w)`, `get_biases(i)`, `set_biases(i, b)` | weights `i` feed layer `i + 1`: shape `(out, in)` for a dense layer, `(filters, kernel_h, kernel_w, in_channels / groups)` for a convolution, gamma `(channels,)` for batch normalization (beta are its biases), empty for pooling |
| `save(path, precision, save_optimizer)`, `Network.load(path)` | `.slett` files, shared with the C API |
| `to_bytes(precision, save_optimizer=False)`, `Network.from_bytes(data)` | the same files as `bytes` |
| `to_model(precision=INT8)` | a `Model`: read-only, computes in its precision (integer kernels for INT8, INT4, INT2) |
| `Model.load(path)`, `Model.from_bytes(data)`, `model.to_bytes()` | models from and to `.slett` files of any version |
| `model.predict(x)` / `model(x)`, `model.evaluate(x, y)` | 1-D input -> vector, 2-D batch -> matrix; `Metrics(loss, accuracy)` |
| `model.layers`, `input_size`, `output_size`, `size`, `workspace_size` | `LayerInfo(inputs, outputs, activation, precision, type, shape, input_shape, kernel, stride, padding, groups, epsilon)` per layer (normalizations after a dense or convolution layer are folded into it); image and C workspace bytes |
| `export_c_header(path, name, precision=INT8)` | the model as a C header for the standalone engine (`Spingalett.Inference.h`) |
| `set_compute_mode`, `set_num_threads`, `seed`, `set_verbose`, `set_log_level`, `set_log_callback` | process-wide settings |
| `cpu_kernels()`, `library_version()`, `library_path()` | the matrix kernels in use (`"AVX-512"`, `"AVX2"`, ...), the loaded library |

Library errors raise `SpingalettError`, whose `code` is an `ErrorCode`. The bindings check on import
that the library has the same major.minor version (`library_version()`), because they mirror its
struct layouts. A `Network` is not thread-safe: use one per thread. A `Model` is read-only and can
be shared between threads.
