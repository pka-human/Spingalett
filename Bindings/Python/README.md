# Spingalett for Python

Thin `ctypes` bindings over the Spingalett shared library: no compiler is needed to install them,
only NumPy and `libspingalett` (`.so` / `.dylib` / `.dll`). The wheels on PyPI and on the GitHub
releases carry the library inside (Linux x86-64 and AArch64 with glibc 2.28 or newer, Windows
x86-64, macOS 11 and newer, for any Python 3), with OpenMP:

```bash
pip install spingalett
```

From a source checkout, build the library and point the bindings at it:

```bash
cmake -S . -B Build -DCMAKE_BUILD_TYPE=Release -DBUILD_WITH_OPENMP=ON
cmake --build Build --parallel            # produces Bin/libspingalett.so
pip install ./Bindings/Python
export SPINGALETT_LIBRARY=$PWD/Bin/libspingalett.so   # or put Bin/ on the library path
```

Without `SPINGALETT_LIBRARY` the package looks inside itself (wheels), then in the repository's
`Bin/` directory (when imported from a checkout), then on the system library path. The package is
typed (`py.typed`): editors and type checkers read its annotations.

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

# a residual block: inputs= names earlier layers, add_add / add_concat combine them
res = sg.Network(sg.Loss.CROSS_ENTROPY)
x = res.add_input(32, 32, 3).add_conv2d(16, 3, padding=1, activation=sg.Activation.NONE) \
       .add_batch_norm(sg.Activation.RELU).last
res.add_conv2d(16, 3, padding=1, activation=sg.Activation.NONE).add_batch_norm(sg.Activation.RELU)
res.add_conv2d(16, 3, padding=1, activation=sg.Activation.NONE).add_batch_norm()
res.add_add([x, -1], activation=sg.Activation.RELU)       # -1: the layer before
res.add_global_avg_pool().add_layer(10, sg.Activation.SOFTMAX, sg.Init.XAVIER)

# a U-Net: transposed convolutions up, concatenated with the maps on the way down; a sigmoid a pixel
unet = sg.Network(sg.Loss.CROSS_ENTROPY)
e = unet.add_input(64, 64, 3).add_conv2d(16, 3, padding=1).last                  # 64 x 64 x 16
unet.add_max_pool2d(2).add_conv2d(32, 3, padding=1)                                 # 32 x 32 x 32
u = unet.add_conv_transpose2d(16, 2, stride=2).last                                 # 64 x 64 x 16
unet.add_concat([e, u]).add_layer_norm(sg.Activation.RELU)                          # 64 x 64 x 32
unet.add_upsample(2, sg.Upsample.BILINEAR).add_conv2d(1, 1, activation=sg.Activation.SIGMOID)

# PyTorch: a module through ONNX in memory, or a state dict into a network of the same layers
import torch
model = torch.nn.Sequential(torch.nn.Conv2d(3, 8, 3, padding=1), torch.nn.ReLU(), torch.nn.Flatten(),
                            torch.nn.Linear(8 * 32 * 32, 10))
net = sg.Network.from_torch(model, torch.rand(1, 3, 32, 32))     # takes (32, 32, 3) channels-last images
same = sg.Network(sg.Loss.MSE, [sg.Input(32, 32, 3), sg.Conv2D(8, 3, padding=1), sg.Layer(10, sg.Activation.NONE)])
same.load_pytorch(model.state_dict())                              # or "model.pt", "model.safetensors"
onnx = sg.Network.from_onnx("model.onnx")

with sg.Network.load("xor.slett") as net:
    print(net.topology, net.forward([1, 0]))
    # deployment: a read-only INT8 model with integer kernels, and a C header for firmware
    with net.to_model(sg.Precision.INT8) as model:
        print(model.predict([1, 0]), model.layers, model.size)
    net.export_c_header("xor_model.h", "xor_model", sg.Precision.INT8)
```

| API | Notes |
|---|---|
| `Network(loss, layers)` / `add_layer(...)` | first layer is the input layer; `layers` may mix `Layer`, `Input`, `Conv2D`, `ConvTranspose2D`, `MaxPool2D`, `AvgPool2D`, `Upsample2D`, `BatchNorm`, `LayerNorm`, `Add`, `Concat`, `GlobalAvgPool` and plain widths |
| `inputs=` on every `add_*`, `last`, `len(net)` | a layer reads the one before it, or the earlier layers `inputs` names (indices; negative ones count back from the new layer); `last` is the index of the layer added last |
| `add_add(inputs, activation=NONE, dropout=0)`, `add_concat(inputs, ...)`, `add_global_avg_pool(dropout=0, inputs=None)` | the sum of layers of one shape (residual connections), layers side by side along the channels, the mean of each channel |
| `Network.from_onnx(path or bytes)`, `Network.from_torch(module, example_input)` | ONNX models (and PyTorch modules, exported to ONNX in memory) as networks with channels-last inputs: transpose NCHW images with `x.transpose(0, 2, 3, 1)` |
| `load_pytorch(state_dict, path or bytes, modules=None)` | PyTorch weights (`state_dict()`, `torch.save` or safetensors files) into a network of the same layers, reordered for channels-last data; the network is unchanged on error |
| `add_input(h, w, c)`, `add_conv2d(filters, kernel, stride=1, padding=0, activation=RELU, init=HE, dropout=0, groups=1)`, `add_max_pool2d(kernel, stride=0, padding=0)`, `add_avg_pool2d(...)` | convolution (grouped or depthwise with `groups`) and pooling over channels-last tensors; pooling's stride defaults to the kernel size |
| `add_batch_norm(activation=NONE, epsilon=1e-5, momentum=0.1, dropout=0)` | per-channel normalization of the previous layer: batch statistics while training, running averages otherwise; `get_running_statistics(i)` / `set_running_statistics(i, mean, var)` |
| `add_conv_transpose2d(filters, kernel, stride=1, padding=0, output_padding=0, activation=RELU, init=HE, dropout=0, groups=1)`, `add_upsample(factor=2, mode=Upsample.NEAREST)`, `add_layer_norm(activation=NONE, epsilon=1e-5)` | transposed convolution ((in - 1) stride - 2 padding + kernel + output_padding cells an axis), nearest or bilinear upsampling (PyTorch's `align_corners=False`), normalization of each cell over its channels (a dense layer's outputs are one cell) |
| `layers`, `layer(i)`, `topology` | `LayerDescription(type, shape, outputs, activation, dropout, kernel, stride, padding, weight_count, bias_count, groups, epsilon, momentum, inputs)` per layer |
| `forward(x)` | 1-D input -> vector, 2-D batch -> matrix (one batched `predict()` call; results are copies) |
| `train(x, y, config=None, validation_data=None, **overrides)` | fields of `TrainConfig` (e.g. `epochs`, `strategy`, `batch_size`, `shuffle`, `early_stopping_patience`, `restore_best_weights`, `augment_shift`, `augment_flip`, `label_smoothing`); returns a `TrainResult`; `callback(network, progress)` gets a `Progress`; exceptions raised in callbacks stop training and are re-raised |
| `train_from_generator(fn, samples_per_epoch=0, validation_data=None, ...)` | `fn(inputs, targets)` fills the given arrays and returns the number of rows; 0 ends the epoch |
| `evaluate(x, y)` | `Metrics(loss, accuracy)` |
| `Trainer(net, max_batch)` | `forward(x)`, `backward(y)`, `backward_output_grads(dl_dout)` for custom losses, `step(optimizer=..., learning_rate=...)`, `zero_grad()`, `train_on_batch(x, y, ...)`; gradients via `net.get_weight_gradients(i)` / `get_bias_gradients(i)`; on the GPU when made with `ComputeMode.VULKAN` or `ComputeMode.CUDA` |
| `load_idx(images, labels, num_classes=0)`, `load_cifar(paths, num_classes=10)`, `load_csv(path, target_columns=1, num_classes=0)` | return `(inputs, targets)` float32 arrays; CIFAR images as 32 x 32 x 3, channels last |
| `save_dataset(path, x, y, input_encoding=AUTO, target_encoding=AUTO, compress=True, shape=None, class_names=None, target_name=None, extra_targets=None)`, `load_dataset(path, target_set=0)`, `dataset_info(path)` | `.slettd` data set files, optionally with the input shape, class names and further sets of targets (`extra_targets`: dicts with `"targets"` and optionally `"name"`, `"class_names"`, `"encoding"`); `dataset_info` returns the counts, encodings, `shape` and `target_sets` |
| `train_from_file(path, shuffle=True, in_memory=False, prefetch=True, target_set=0, ...)` | trains on a `.slettd` file through the C reader: streamed with a few chunks in memory, or with `in_memory=True` decoded once and kept in its compact form (a byte per 8-bit value); `prefetch=False` keeps decoding off a background thread |
| `CosineDecay`, `LinearWarmup`, `StepDecay`, `WarmupCosine` | built-in schedules; any `fn(epoch, total, initial_lr)` works too |
| `get_weights(i)`, `set_weights(i, w)`, `get_biases(i)`, `set_biases(i, b)` | weights `i` feed layer `i + 1`: shape `(out, in)` for a dense layer, `(filters, kernel_h, kernel_w, in_channels / groups)` for a convolution or a transposed one, gamma `(channels,)` for batch and layer normalization (beta are its biases), empty for pooling and upsampling |
| `save(path, precision, save_optimizer)`, `Network.load(path)` | `.slett` files, shared with the C API |
| `to_bytes(precision, save_optimizer=False)`, `Network.from_bytes(data)` | the same files as `bytes` |
| `to_model(precision=INT8)` | a `Model`: read-only, computes in its precision (integer kernels for INT8, INT4, INT2) |
| `Model.load(path)`, `Model.from_bytes(data)`, `model.to_bytes()` | models from and to `.slett` files of any version |
| `model.predict(x)` / `model(x)`, `model.evaluate(x, y)` | 1-D input -> vector, 2-D batch -> matrix; `Metrics(loss, accuracy)` |
| `model.layers`, `input_size`, `output_size`, `size`, `workspace_size` | `LayerInfo(inputs, outputs, activation, precision, type, shape, input_shape, kernel, stride, padding, groups, epsilon, input_layers)` per layer (normalizations after a dense or convolution layer are folded into it); image and C workspace bytes |
| `export_c_header(path, name, precision=INT8)` | the model as a C header for the standalone engine (`Spingalett.Inference.h`) |
| `set_compute_mode`, `set_num_threads`, `seed`, `set_verbose`, `set_log_level`, `set_log_callback` | process-wide settings; `ComputeMode.CUDA` (NVIDIA) and `ComputeMode.VULKAN` train and predict on the GPU |
| `DeviceData(values)` | a data set in the GPU's memory (rows of float32, an array `(n, ...)` flattened to a row a sample), copied there once; `train()`, `validation_data`, `forward()` and `evaluate()` take it in place of an array (inputs, targets or both), and with the GPU modes gather, augment and smooth their batches on the device; `len(d)`, `d.shape`, `d.numpy(first=0, count=None)`, `close()` or a `with` block |
| `set_gpu_precision(Precision.BFLOAT16)`, `get_gpu_precision()` | the GPU's matrix products in bfloat16 on its matrix units, or `FLOAT32` (the default); returns whether the GPU multiplies in that precision |
| `cpu_kernels()`, `cuda_device()`, `gpu_device()`, `library_version()`, `library_path()` | the matrix kernels in use (`"AVX-512"`, `"AVX2"`, ...), the GPUs `CUDA` and `VULKAN` use (`None` without one), the loaded library |

`uint8` arrays stand for 8-bit images: value q means q / 255 in `train()`, `forward()`,
`predict()`, `evaluate()` and `save_dataset()`. `train()` keeps such inputs as bytes, a quarter of
the memory of float32, and converts them a batch at a time, with the same result as training on
`x / 255` in float32.

Library errors raise `SpingalettError`, whose `code` is an `ErrorCode`. The bindings check on import
that the library can serve them (`library_version()`), because they mirror its struct layouts: the
same major version and at least their own minor one (before 1.0, the same major.minor version). A
`Network` is not thread-safe: use one per thread. A `Model` is read-only and can be shared between
threads.
