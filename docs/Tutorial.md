# A path through Spingalett

Six programs of `Examples/`, each adding to the one before: a network and its training, real data
with validation, convolutions, batch normalization with augmentation and a GPU, a graph for
segmentation, and a model on a microcontroller. Every step names the program to run, what to read
in it, and what it does. The [README](../README.md) documents each part in full, the
[API reference](Reference.md) every declaration.

Build the library and the examples once:

```bash
cmake -S . -B Build -DCMAKE_BUILD_TYPE=Release -DBUILD_WITH_OPENMP=ON
cmake --build Build --parallel
```

The programs land in `Bin/`. Steps 2 and 3 need MNIST (`Examples/download_mnist.sh data/mnist`),
step 4 CIFAR-10 (`Examples/download_cifar10.sh data/cifar10`).

## 1. A network and its training: `Examples/XOR.c`

```bash
Bin/XOR
```

A network is made by `spingalett_network_new()` with its loss, then grown one layer at a time by
`spingalett_layer()`, the first layer being the input. `spingalett_train()` takes the samples as
arrays (`.inputs`, `.targets`, `.sample_count`) and everything else by name; fields left out are
zero, which selects each documented default (no activation, the first optimizer, ...). The
program trains full batches with AdamW, stops early through a callback, runs the network on each
case with `spingalett_forward()`, saves it with `spingalett_save()` and loads it back with
`spingalett_load()`.

Read in it: the builders' named arguments (`.act_func = SPINGALETT_ACT_SIGMOID`), the callback
(`.callback`, `.callback_interval`), and `spingalett_seed()`, which fixes the initial weights.

## 2. Real data and validation: `Examples/MNIST.c`

```bash
Bin/MNIST data/mnist 10 omp
```

A 784-256-128-10 network on handwritten digits, about 98.2% on the test set after 10 epochs.
`spingalett_load_idx()` reads the IDX files into a `SpingalettDataset`, `spingalett_dataset_split()`
holds out 5,000 images for validation, and `spingalett_train()` evaluates them after every epoch
(`.val_inputs`, `.val_targets`, `.val_count`), keeping the weights of the best epoch
(`.restore_best_weights`). `spingalett_evaluate()` gives the test loss and accuracy, and
`spingalett_model_from_network()` makes deployment models in FP16, INT8 and INT4 to show what each
costs in accuracy and saves in size.

Read in it: mini-batches (`SPINGALETT_STRATEGY_SMALL_BATCH`, `.batch_size`), the compute mode
(`spingalett_set_compute_mode()`: one thread, OpenMP, OpenBLAS), and the report `spingalett_train()`
returns.

## 3. Convolutions: `Examples/MNIST_CNN.c`

```bash
Bin/MNIST_CNN data/mnist 2
```

The same digits as 28 x 28 x 1 tensors through two convolutions and max pooling: 98.9% after two
epochs. The input layer takes its shape (`.height`, `.width`, `.channels`); `spingalett_conv2d()` and
`spingalett_max_pool2d()` add the layers, and a dense layer reads what comes before it as a flat
vector. Dropout (`.dropout_rate`) regularizes the dense layer. The model it saves is format
version 4, which the inference engine runs, on a microcontroller too (step 6).

Read in it: the layers' windows (`.kernel`, `.padding`, `.stride`), He initialization for ReLU
layers (`SPINGALETT_INIT_HE`), and the deployment models of a convolutional network.

## 4. Batch normalization, augmentation and the GPU: `Examples/CIFAR10.c`

```bash
Bin/CIFAR10 data/cifar10 30 gpu                 # or omp; resnet20 for a residual network
```

Color images of ten classes through normalized convolutions (`spingalett_batch_norm()` after each
convolution without activation), trained on images shifted and mirrored anew every epoch
(`.augment_shift`, `.augment_flip`), with label smoothing for the residual networks. With `gpu`
(`SPINGALETT_COMPUTE_VULKAN`) it trains and evaluates on the GPU, with `bf16` in bfloat16 on its
matrix units; ResNet-20 reaches 91.55% in single precision. Deployment models fold every
normalization into the convolution before it.

Read in it: `resnet20`'s blocks (`spingalett_add_layers()` closing a residual connection, `.inputs`
naming the layers a layer reads), the learning-rate schedule (`.lr_scheduler`), and
`spingalett_set_gpu_precision()`.

## 5. A graph for segmentation: `Examples/Segmentation.c`

```bash
Bin/Segmentation 12 gpu
```

A U-Net labels every pixel of synthetic images with the shape it belongs to (a mean intersection over
union of 0.88). The contracting path's maps are concatenated (`spingalett_concat_layers()`) with those
of the expanding path, which `spingalett_conv_transpose2d()` (or `spingalett_upsample2d()` and a
convolution, with `bilinear`) brings back to full size. The images are drawn anew for every epoch by
a generator function (`SPINGALETT_MODE_GENERATOR_FUNCTION`, `.generator`), and `spingalett_predict()`
labels the test images in batches.

Read in it: the indices the builders return, kept to name the inputs of later layers; the sigmoid
outputs a pixel with binary cross-entropy; and the INT8 model, which keeps the accuracy.

## 6. A model on a microcontroller: `Examples/Embedded`

```bash
Examples/Embedded/run-qemu.sh mnist.slett data/mnist            # after step 2
```

`ModelTool header` exports the trained network, quantized to INT8, as a C array (or
`spingalett_export_c_header()` from a program), which `mnist_mcu.c` compiles into flash with the
inference engine alone (`Src/Spingalett.Inference.c` and `Spingalett.Inference.h`: no heap, no I/O).
`spingalett_model_init()` checks the image in place and `spingalett_model_run()` evaluates a digit in
2.8 KB of workspace; the script runs it on a Cortex-M4 in QEMU. Its
[README](../Examples/Embedded/README.md) lists the sizes and accuracies of each precision.

## Further

- C++: `Spingalett/Spingalett.hpp` wraps the same calls in owning types, `std::span` and
  `std::expected` (README, section C++).
- Python: `pip install spingalett` gives the same library to NumPy programs
  ([Bindings/Python](../Bindings/Python/README.md)).
- Models from elsewhere: `spingalett_import_onnx()` and `spingalett_load_pytorch()` (README,
  Importing models).
- Files of training data: `.slettd` data sets, streamed by a reader (README, Data set files).
