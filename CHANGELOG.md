# Changelog

All notable changes to this project are documented in this file. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/); before 1.0, a minor release may contain breaking
changes, which are listed under **Changed**.

## [0.9.0] - Unreleased

"Bottlenecks": every part of the library profiled and its slow paths removed, from data set files
through training and inference to model files and the Python bindings. Measurements on a 4-vCPU
Xeon @ 2.1 GHz with AVX-512, 4 threads unless noted.

### Added
- `ROADMAP.md` (plans for the next releases, 1.0 and later) and `AGENTS.md` (layout, checks and
  invariants for coding agents and contributors).
- `.slettd` format version 2 (see `docs/DatasetFormat.md`): metadata records for the input shape,
  names of sets of targets, class names, and further sets of targets for the same samples (such as
  CIFAR-100's fine and coarse labels); and an rANS coder of half-bytes that decodes dense data such
  as photographs two to three times as fast as the binary range coder at about the same size.
  The writer chooses the coder per stream, and still writes version 1 for data sets that need
  neither. CIFAR-10's training set takes 112.1 MB (`xz -9` of the original batches: 116.3 MB).
- `SpingalettDataset.height`, `.width`, `.channels` and `.class_names`, filled by
  `spingalett_load_idx()`, `spingalett_load_cifar()` (class names from `batches.meta.txt`,
  `fine_label_names.txt` or `coarse_label_names.txt` next to the batches) and `.slettd` files, kept
  by `spingalett_dataset_split()`; `spingalett_dataset_set_class_names()`.
- `DatasetSaveOptions.target_name`, `.extra_targets` and `.extra_target_count`
  (`SpingalettTargetSet`), `spingalett_load_dataset_targets()` and
  `spingalett_load_dataset_from_memory_targets()`; `SpingalettDatasetInfo.format_version`,
  `.height`, `.width`, `.channels`, `.target_set_count` and `.target_set`;
  `spingalett_dataset_target_set_name()`, `spingalett_dataset_class_name()` and
  `spingalett_dataset_target_set_size()`.
- `spingalett_dataset_open_ex()` with `DatasetReaderOptions`: `in_memory` decodes the file once and
  keeps its values in their compact form (a byte per 8-bit value, a quarter of float32) with
  every pass shuffling all samples; `target_set` serves another set of targets; `no_prefetch`
  keeps decoding off a background thread. `spingalett_dataset_open_u8()` trains from 8-bit data in
  memory without a float copy.
- `DatasetTool cifar` (CIFAR-10, and CIFAR-100 with both sets of labels), `DatasetTool images`
  (one subfolder per class; PNG, JPEG, BMP and other formats through stb_image, optional resizing)
  and `DatasetTool verify`; `DatasetTool info` prints the shape, sets of targets and class names.
- Python: `save_dataset(..., shape, class_names, target_name, extra_targets)`,
  `load_dataset(path, target_set)`, the shape and sets of targets in `dataset_info()`, and
  `train_from_file(..., in_memory, prefetch, target_set)`.

### Changed
- Deployment models made by the library (`spingalett_model_from_network()`, `_load()`,
  `_from_memory()`) prepare what batched prediction runs on once, on the first call that needs it:
  FP16 and BF16 weights expanded to float, INT4 and INT2 rows unpacked, integer rows interleaved
  for the tile kernels, transposed filters and row sums. Before, `spingalett_model_predict()` built
  them, and all of its buffers, on every call; it now also keeps the last call's workspace for the
  next one. Models stay safe to share between threads (the preparation takes a lock, the workspace
  an atomic exchange); models filled in by `spingalett_model_init()` own nothing and prepare on
  every call as before. The prepared forms take memory next to the image: up to the float size of
  FP16 layers and four times the packed size of INT2 layers. `predict()` of one sample of a
  784-256-128-10 MLP: FP32 63 -> 12 us, FP16 403 -> 8 us, INT8 10.5 -> 4.4 us, INT4 27 -> 4.2 us;
  batches of 1024: FP16 2.04 -> 0.92 us, INT8 0.54 -> 0.31 us per sample.
- Matrix products of at most 8 rows (4 with AVX2, 2 without AVX) from row-major operands, such as
  a dense layer on one sample or a few, take dot products instead of packing all of the weights:
  the FP32 MLP above predicts 8 samples in 2.2 us each instead of 7.9, and training it with batches
  of 2, 4 and 8 takes 0.33, 0.175 and 0.076 s per 4096 samples instead of 0.57, 0.28 and 0.093.
  Float outputs of one sample or a few may now differ in the last bits from those of the same
  samples in a larger batch; integer models still compute exactly what single runs compute, and
  results still do not depend on the thread count.
- Loops that would run on one thread no longer enter an OpenMP parallel region (whose false `if`
  clause still cost about 0.2 us), and integer layers start threads from 2^19 multiply-adds instead
  of 2^15: `predict()` of one sample of a 4-8-8-2 INT8 model takes 0.21 us instead of 2.19.
- Pooling (forward and backward) runs a whole block of 16 channels without a test per channel, so
  that it vectorizes; depthwise convolutions with one filter per channel and 16, 32 or 64 channels
  sum their taps in registers, forward and for the data gradient; the tile kernels' epilogue, window
  gathering and the activation maximum of integer layers lost their overheads. Results are the same
  bits as before. Batched INT8 prediction of a small CIFAR-sized CNN: 36.6 -> 33.3 us per sample;
  training a depthwise-separable block on 32 x 32 x 3 images: 3017 -> 3514 samples/s, a small CNN
  with batch normalization 7188 -> 7529.
- A mini-batch step gathers and augments its samples on the OpenMP threads.
- Model files load and save three to ten times as fast: CRC-32 eight bytes at a time in the library
  (the engine alone keeps its byte-wise loop and 1 KB table), FP16 conversion eight values at a
  time with F16C (the same halves as the portable conversion for every value but NaN, which keeps
  the portable path), and networks built without reallocating every parameter array for each layer
  (adding a layer copied and zeroed all of them; loaders now reserve the final sizes). Files are
  byte for byte the same. A 924,930-parameter MLP: FP32 save 10.8 -> 2.5 ms, load 26 -> 8.3 ms, as
  a model 9.5 -> 2.4 ms; FP16 save 15.2 -> 1.4 ms; INT8 load 13 -> 1.3 ms.
- Python: `Network.input_size` / `output_size` read the sizes instead of describing every layer,
  and per-batch calls pass arrays by address: `forward()` of a 4-8-2 network 42.6 -> 7.8 us, of the
  MLP above on one sample 69 -> 22 us, `Model.predict()` of one sample 9.1 -> 5.4 us,
  `Network.from_bytes()` of that MLP 6.2 -> 0.75 ms.
- Streaming readers decode the next chunks on a background thread when a processor is free for it
  (fewer OpenMP threads than processors), and otherwise several chunks at a time on the OpenMP
  threads when they are needed: a thread competing with the OpenMP threads for the processors
  stalls their barriers. Chunks stay in their compact form until a batch is read, and batches
  convert to float on the OpenMP threads. One epoch of a small CNN on CIFAR-10 streamed from a
  `.slettd` file: 14.7 s with 0.8, 5.5 s now (4.5 s from float arrays, 4.6 s from the file in
  memory).
- A reader's shuffled order is drawn from a seed taken when it opens and from the number of the
  pass or chunk, so it is the same with or without a background thread and on any number of
  threads; it differs from 0.8's order for the same `spingalett_seed()`.
- `spingalett_save_dataset()` compresses chunks in parallel (CIFAR-10's training set: 12.4 s
  before, 2.2 s on 4 threads; MNIST: 1.11 s before, 0.47 s).
- `SpingalettDataset` has new fields: a data set built by hand must be zero-initialized
  (`SpingalettDataset d = {0};`), because `spingalett_save_dataset()` reads them and
  `spingalett_dataset_free()` frees `class_names`. The loaders initialize every field.
- `SpingalettDatasetInfo` and `DatasetSaveOptions` have new fields; zero-initialized options keep
  the previous behavior.
- Python: `uint8` inputs are 8-bit images, value q read as q / 255, in `train()`, `forward()`,
  `predict()`, `evaluate()` and `save_dataset()` (before, they were converted to floats 0 to 255).
  `train()` keeps them as bytes and converts a batch at a time. Targets keep their values.
- `DigitPadTrain` prints each epoch's time and the time since training began.

## [0.8.0] - Unreleased

Deeper convolutional networks: batch normalization, grouped and depthwise convolutions, image
augmentation and a CIFAR-10 example.

### Added
- `batch_norm()` (`LAYER_BATCH_NORM`): per-channel normalization of the previous layer with the
  batch's statistics while training and running averages of them (`.momentum`, default 0.1;
  `.epsilon`, default 1e-5) in `predict()`, `forward()`, `evaluate()` and models; an activation
  and dropout of its own. Gamma and beta are the layer's weights and biases for every optimizer,
  clipping, the step API and `spingalett_get_parameters()`; weight decay leaves them alone.
  `PARAM_RUNNING_MEAN` and `PARAM_RUNNING_VARIANCE` read and write the statistics, which
  `restore_best_weights` restores with the weights. Training stays bit-identical on any number of
  threads; gradients are checked numerically after convolutions, dense layers, the input and
  pooling.
- Inference with a normalization after a dense or convolution layer costs no more than the layer
  alone: `predict()` and `forward()` apply it in the layer's matrix-product epilogue, and deployment
  models (`spingalett_model_from_network()`, and files saved in other precisions than FP32 without
  optimizer state) fold it into the layer's weights and biases. Other normalizations remain layers
  that the inference engine runs in every precision.
- Grouped convolutions: `conv2d(.groups = g)` splits input channels and filters into `g` groups;
  as many groups as input channels make a depthwise convolution, with any number of filters per
  channel. Depthwise convolutions run directly (all of a pixel's channels at once), other groups as
  one matrix product per group, in training and in the engine (integer results bit-exact against a
  reference, batched identical to single runs).
- `.slett` format version 5 (written only for networks that need it): layer table entries of 80
  bytes adding convolution groups and the normalization's epsilon and momentum; normalizations
  store gamma, beta and the running statistics in FLOAT32 (docs/ModelFormat.md).
- Image augmentation in `train()`: `.augment_shift` (random shifts by up to that many cells, zeros
  shifted in) and `.augment_flip` (mirror images for half of the samples), drawn per sample and
  step from a seed, so results do not depend on the thread count.
- `spingalett_load_cifar()` reads CIFAR-10 and CIFAR-100 binary batches as 32 x 32 x 3
  channels-last images.
- `Examples/CIFAR10.c` and `Examples/download_cifar10.sh`: a batch-normalized network with
  augmentation, optionally depthwise-separable, evaluated as FP16 and INT8 models.
- Python: `BatchNorm`, `add_batch_norm()`, `get_running_statistics()` / `set_running_statistics()`,
  `Conv2D(groups=...)`, `load_cifar()`, `TrainConfig.augment_shift` / `augment_flip`.
- Batched integer inference on processors with byte dot-product instructions:
  `spingalett_model_predict()` and `spingalett_model_evaluate()` run integer convolutions (with one
  group) and dense layers (from 12 samples) in tiles of 12 pixels or samples against weight rows
  interleaved four bytes at a time, with kernels for AVX-512 VNNI, AVX-VNNI and the Arm dot
  product extension (Src/Spingalett.Int8Tiles.c). Builds without `-march=native` carry them and
  choose at run time: the x86-64 packages, and on AArch64 Linux. The INT8 CIFAR-10 network
  predicts 3.1 times as fast as in 0.7 on one thread (2,520 images/s) and 2.7 times on four, 2.5
  and 2.1 times as fast as its FP32 model. Results are unchanged (integer sums are exact), and
  still identical to single runs.
- The macOS package is built with OpenMP and carries LLVM's OpenMP runtime (`libomp.dylib`, with
  its license) next to the library, which finds it through `@loader_path`.
- `Bin/Benchmark` and `Examples/benchmark_pytorch.py` also measure the convolutional network with
  batch normalization: Spingalett trains it 1.5 to 2 times as fast as PyTorch and infers it 4.5 to
  5.6 times as fast.

### Changed
- `SPINGALETT_FORMAT_VERSION` is 5. `LayerArgs`, `TrainArgs`, `SpingalettNetworkLayer` and
  `SpingalettLayerInfo` gained fields, and `ParameterKind` gained two values: rebuild programs
  against the new headers.
- DigitPad's model is a batch-normalized convolutional network: 99.58% MNIST test accuracy and
  99.45% on randomly distorted digits (the MLP: 99.27% and 98.44%).
- The inference engine's INT8 convolutions take four pixels against four filters at a time, with
  the filters' sums computed once per layer on AVX-512 VNNI: one CIFAR-10 image through
  `spingalett_model_run()` takes 1.7 to 1.8 times less time. GCC's partial redundancy elimination
  is off for the integer dot-product kernels, where it copied the sums through memory on every
  step. The engine's scratch for an INT8 convolution grows by 4 bytes per filter.
- The shared library's soname is `libspingalett.so.0.8`.

## [0.7.0] - 2026-10-07

Convolutional networks: training, saving, and deployment down to microcontrollers, with the
network becoming an opaque handle.

### Added
- Layer types (`LayerType`): `conv2d()` (2D convolution with any kernel, stride and padding, also
  rectangular), `max_pool2d()` and `avg_pool2d()` next to dense layers. Tensors are channels-last
  (`height x width x channels` per sample); the input layer takes a shape (`.height`, `.width`,
  `.channels`), and a dense layer reads what precedes it as a flat vector. Every training strategy,
  optimizer, schedule, dropout, the step API, `predict()` and `evaluate()` work with them. Gradients
  are checked numerically for every layer kind, activation and compute mode.
- Convolutions run as matrix products whose input windows the matrix kernels gather themselves
  (implicit im2col), with the bias and activation applied while each tile of the result is in
  cache; weight gradients are split over threads in slots fixed by the shape. On one core the
  convolutional network of `Examples/MNIST_CNN.c` trains 1.6 times and infers 4 times as fast as
  PyTorch (1.8 and 4 times on four cores).
- Accessors for the opaque network: `spingalett_layer_count()`, `spingalett_network_layer()`
  (`SpingalettNetworkLayer`: type, shape, outputs, activation, dropout, window, parameter counts),
  `spingalett_input_size()`, `spingalett_output_size()`, `spingalett_parameter_count()`,
  `spingalett_network_loss()`, `spingalett_optimizer_steps()`, and `spingalett_get_parameters()` /
  `spingalett_set_parameters()` for the weights, biases and gradients of a layer (`ParameterKind`).
- `.slett` format version 4 for networks with convolution or pooling layers: 64-byte layer table
  entries with the layer's kind, shape, window, stride and padding, the input shape in the header
  (docs/ModelFormat.md). Networks of dense layers are still written as version 3.
- The inference engine and deployment models run convolutions and pooling in every precision:
  integer convolutions quantize each sample's input once and take integer dot products of the
  filters with each window; windows shorter than 32 values (a first layer over one or three
  channels) accumulate all filters at once. Batched prediction of integer models remains
  identical to single runs. The MNIST CNN keeps 98.9% test accuracy in INT8 (424 KB) and 98.75% in
  INT4, and runs on the Cortex-M4 example with 208 KB of RAM. `SpingalettLayerInfo` describes
  kinds, input and output shapes and windows; `ModelTool info` prints them.
- Python: `Input`, `Conv2D`, `MaxPool2D`, `AvgPool2D` layer specs, `add_input()`, `add_conv2d()`,
  `add_max_pool2d()`, `add_avg_pool2d()`, `Network.layers` / `layer(i)` (`LayerDescription`),
  convolution weights as `(filters, kernel_h, kernel_w, channels)` arrays, `LayerType`, and model
  layer shapes in `LayerInfo`.
- `Examples/MNIST_CNN.c`: a two-convolution MNIST network (about 99% test accuracy after two
  epochs), evaluated as FP16, INT8 and INT4 models. `Bin/Benchmark` and
  `Examples/benchmark_pytorch.py` measure it too.
- A test that trains a convolutional and a dense network on 1, 3 and 4 threads and single-threaded
  and requires identical weights.

### Changed
- **Breaking:** `NeuralNetwork` is an opaque type; read networks through the accessors above
  (`net->topology`, `net->weights` and the other fields are gone from the public header).
  `LayerArgs` gained the fields of convolution and pooling layers, `SpingalettLayerInfo` and
  `SpingalettModel` gained fields, and `LayerType` moved to `Spingalett.Inference.h` (included by
  `Spingalett.h`): rebuild programs against the new headers. `SPINGALETT_FORMAT_VERSION` is 4.
- Products with at most 16 columns from row-major operands (output layers of a few units) run as
  blocks of dot products instead of mostly idle matrix panels: a 64 x 10 x 1000 product takes 23
  instead of 49 us on one thread and 9 instead of 58 us on four.
- The matrix kernels decide whether to use threads from the size and kind of each product
  (products below about 10 us of work run on one thread). A 32-64-64-1 regression network trains
  65% faster with OpenMP (7% faster on one thread) and infers 40% faster (10%).
- A dense layer's bias and activation run in its matrix product's epilogue. Epilogues apply
  activations row by row, so results do not depend on how threads split a product: with the
  built-in kernels, training gives the same bits on any number of threads and single-threaded.
- K blocks of the matrix kernels have equal sizes (288 runs as 2 x 144 rather than 256 + 32), and
  when the whole right operand fits the pack buffer, each tile of the result runs through all K
  blocks while it is in cache. The 784-512-1000-10 benchmark network runs as fast as in 0.6.
- Batch training and `predict()` take fewer samples per chunk when a sample's activations are
  large, so that convolutional networks need tens rather than hundreds of megabytes.
- The shared library's soname is `libspingalett.so.0.7`.

## [0.6.0] - 2026-10-07

A performance release: the same API and file formats, faster kernels.

### Added
- Run-time choice of matrix kernels on x86-64. Libraries built without `-march=native` with GCC
  or Clang (the release packages among them) also contain the GEMM kernels for AVX2+FMA and for
  AVX-512 and run the best set the processor supports. On a Cascade Lake core the baseline x86-64
  package now trains 3.6 to 4.9 times as fast and infers 5 times as fast. The CMake option
  `SPINGALETT_CPU_DISPATCH` (on by default) controls it.
- `spingalett_cpu_kernels()` names the matrix kernels in use (`"AVX-512"`, `"AVX2"`, `"AVX"`,
  `"SSE2"`, `"NEON"` or `"C"`); Python: `cpu_kernels()`. `Bin/Benchmark` prints it.

### Changed
- GEMM: the last row panel of a block runs a kernel of 8 or 4 rows (AVX-512) or 4 or 2 rows (AVX)
  instead of computing zero rows; transposed operands are packed with SIMD transposes; a call is
  one OpenMP parallel region, and products with few rows (mini-batches) pack their shared operand
  once for all threads, which then split the columns. A 64-row product runs 35% faster on one
  thread and 55% faster on four; mini-batch training (784-512-1000-10, batches of 64, Adam) is 36%
  faster on four threads and 5% faster on one. Threads still split the result only by rows and
  columns, so results do not depend on the thread count.
- Batch training applies the optimizer to each layer as soon as its gradient is complete, while
  the gradient is still in cache (steps with gradient clipping keep the old order).
- Inference engine: INT4 and INT2 codes are decoded in registers with AVX2 and NEON, against
  activations permuted once per layer into the order the codes are decoded in; four weight rows
  share every activation load in all precisions. One sample through 784-512-1000-10 takes 18 us in
  INT4 (61 before) and 17 us in INT2 (177 before); through 784-256-128-10, 5 us in INT4 and INT2
  (17 and 44 before) and 10.5 us in FP16 and BF16 (16 and 18 before). Results are unchanged: the
  integer kernels remain bit-exact and batched prediction still equals single runs.
- The shared library's soname is `libspingalett.so.0.6`.

## [0.5.0] - 2026-10-06

### Added
- Deployment models (`SpingalettModel`): read-only networks that keep their weights in the
  precision they are stored in and compute with them. INT8, INT4 and INT2 layers quantize their
  input to 8 bits per sample and accumulate int8 x int8 products in 32-bit integers, with kernels
  for AVX2 (with VNNI where the compiler targets it), SSE2, NEON (with the dot-product extension
  where available), the Arm DSP extension and portable C; FP32, FP16 and BF16 layers compute in
  float. One sample through the 784-512-1000-10 benchmark network takes 20 us in INT8 against
  131 us with `forward()`; INT8 and INT4 keep the MNIST accuracy of the example networks.
  `spingalett_model_from_network()`, `spingalett_model_load()`, `spingalett_model_from_memory()`,
  `spingalett_model_predict()` (batched, OpenMP; identical to single runs),
  `spingalett_model_evaluate()` and `spingalett_model_free()`.
- The inference engine, `Spingalett.Inference.h` with `Src/Spingalett.Inference.c`: checks a
  `.slett` image (`spingalett_model_init()`) and runs it in place (`spingalett_model_run()`) with a
  workspace from the caller, without allocation, I/O or global state, so images can live in flash.
  Compiled with `-DSPINGALETT_INFERENCE_ONLY` it needs nothing beyond `memcpy`, `memset`, `expf`,
  `tanhf` and `lrintf` (about 6 KB of code on a Cortex-M4); the CMake option
  `SPINGALETT_INFERENCE_ONLY` builds the library as the engine alone.
- `spingalett_save_to_memory()`, `load_spingalett_from_memory()` and `spingalett_free()`.
- `spingalett_export_c_header()`: a model as a C header (aligned byte array plus size, input,
  output and workspace macros) for compiling it into firmware.
- `ModelTool` (`Examples/ModelTool.c`, installed with the library): `info`, `convert`, `header`,
  `eval` (accuracy in every precision) and `bench`.
- `Examples/Embedded`: MNIST on a Cortex-M4F under QEMU, the INT8 model in flash and 2.8 KB of RAM.
- `docs/ModelFormat.md`, the specification of the model format.
- Python: `Model` (`load`, `from_bytes`, `predict`, `evaluate`, `layers`, `to_bytes`),
  `Network.to_model()`, `Network.to_bytes()`, `Network.from_bytes()` and
  `Network.export_c_header()`.
- `Examples/MNIST.c` reports the accuracy of the trained network in FP16, INT8 and INT4;
  `Examples/Benchmark.c` measures deployment models in every precision.

### Changed
- Model files are written in format version 3: a magic number, a layer table and 16-byte aligned
  sections, little-endian, with CRC-32 checksums of the header and the contents. Files of versions
  1 and 2 still load; 0.4 cannot read files written by 0.5.
- Integer precisions store one scale per weight row instead of one per tensor; INT2 uses ternary
  thresholds (0.7 of the mean magnitude per row) with the mean magnitude of the kept weights as the
  scale. Biases are always stored in float, and so is the optimizer state.
- `load_spingalett()` reads the file into memory and parses it there; damaged files fail with
  `SPINGALETT_ERR_INVALID` (checksum or layout) or `SPINGALETT_ERR_FILE_IO` (truncated).
- `ActivationFunction`, `LossFunction`, `PrecisionMode`, the error codes and `SPINGALETT_API` are
  defined in `Spingalett.Inference.h`, which `Spingalett.h` includes.
- The shared library's soname is `libspingalett.so.0.5`.

### Fixed
- Data set files opened from several threads at once raced on the CRC-32 table, which was built on
  first use; it is now a constant shared with the model format.

## [0.4.1] - 2026-10-03

### Added
- DigitPad for Windows. `Apps/DigitPad/Package/build-windows-zip.sh` builds
  `DigitPad-<version>-windows-x86_64.zip`: `DigitPad.exe`, the trained model, `libspingalett.dll`,
  `SDL2.dll` and the runtime DLLs they load (collected from the import tables), a `README.txt` and
  the licences. It runs in an MSYS2 UCRT64 shell or cross-compiles on Linux with MinGW-w64.
  Releases attach the zip next to the AppImage.
- `DigitPad.exe` is a GUI program with an icon, version information and a manifest that selects
  UTF-8 as the process code page, so the model loads from folders with non-ASCII names. It prints
  `--help`, `--classify` and `--verbose` output to the console it was started from.
- `Apps/DigitPad/Package/test-windows-zip.ps1`: unpacks the zip into a folder with a non-ASCII
  name, classifies `seven.pgm`, then opens the window, draws a 7 with the mouse and checks the
  prediction.

### Changed
- The release workflow trains the DigitPad model once and packages the same model in the AppImage
  and the Windows zip. Both are tested by classifying `seven.pgm`.
- A release published for an older tag through `workflow_dispatch` is no longer marked as the
  latest release.
- DigitPad decides whether to double its window from the usable display area instead of the
  display mode.

### Fixed
- `DigitPadTrain` exits with an error when it cannot write the model, instead of reporting
  success.
- DigitPad `--classify` reports errors on stderr only, without opening a message box. The footer
  shows the model's file name after a `\` as well as a `/`.
- DigitPad links `SDL2main` before SDL2, which MinGW requires.

## [0.4.0] - 2026-10-03

### Added
- Validation during training: `val_inputs`, `val_targets` and `val_count` in `TrainArgs` are
  evaluated after every epoch. The best epoch of a monitored quantity (`monitor`: validation
  loss or accuracy, or training loss) is tracked; `early_stopping_patience` and
  `early_stopping_min_delta` stop training when it no longer improves, and
  `restore_best_weights` ends training with the best epoch's parameters, kept in memory.
- `evaluate()`: mean loss and accuracy over a data set.
- A low-level training API for custom loops and losses: `spingalett_trainer_new()`,
  `spingalett_trainer_forward()`, `spingalett_trainer_backward()` (built-in loss),
  `spingalett_trainer_backward_output_grads()` (custom loss from dL/d(output)),
  `spingalett_trainer_step()` with `OptimizerArgs`, `spingalett_trainer_zero_grad()` and
  `spingalett_train_on_batch()`. Backward passes accumulate, so one step can span several.
- Data sets: `spingalett_load_idx()` (MNIST format), `spingalett_load_csv()`,
  `spingalett_dataset_shuffle()`, `spingalett_dataset_split()` and `spingalett_dataset_free()`.
- Python: `validation_data`, `monitor`, `early_stopping_patience`, `early_stopping_min_delta` and
  `restore_best_weights` for `Network.train()`, `Network.evaluate()`, `Trainer`,
  `Network.get_weight_gradients()` / `get_bias_gradients()`, `load_idx()` and `load_csv()`.
- `.slettd` data set files: `spingalett_save_dataset()`, `spingalett_load_dataset()`,
  `spingalett_load_dataset_from_memory()` and a streaming reader (`spingalett_dataset_open()`,
  `spingalett_dataset_read()`, and `spingalett_dataset_generator()` for `train()`). Values are kept
  in the smallest lossless encoding (8-bit, half, float32, class indices for one-hot targets) or a
  requested lossy one, and compressed with an adaptive context-model range coder in independently
  decodable chunks with CRC-32 checksums: the MNIST training set takes 7.8 MB (IDX: 47.1 MB).
  Specified in `docs/DatasetFormat.md`; `Examples/DatasetTool.c` converts IDX and CSV files. Python:
  `save_dataset()`, `load_dataset()`, `dataset_info()` and `Network.train_from_file()`.
- Prebuilt release archives for Linux x86-64 (baseline and AVX2/FMA builds) and ARM64, Windows
  x86-64 (MinGW-built DLL with MinGW and MSVC import libraries) and macOS (universal), built,
  tested and published by `.github/workflows/release.yml` for every tag. `DatasetTool` is
  installed with the library.
- `SPINGALETT_STATIC`: define it when compiling the sources into a program or a static library.
- DigitPad (`Apps/DigitPad`, CMake option `BUILD_APPS`): a desktop app that classifies digits
  drawn with the mouse, its trainer (99.27% MNIST test accuracy with on-the-fly augmentation) and
  a script that packages app and model as a Linux AppImage.

### Changed
- The epoch callback is now `bool (*)(NeuralNetwork *, const TrainProgress *, void *user_data)`:
  `TrainProgress` carries the epoch, training loss, learning rate, validation metrics and the
  best epoch so far, and `TrainArgs.callback_data` is passed as `user_data`. In Python the
  callback is `callback(network, progress)`.
- `train()` returns a `TrainReport` (status, epochs run, last losses and metrics, best epoch);
  in Python a `TrainResult`.
- The training loss is computed in every epoch, not only in reported ones.
- Model files use the extension `.slett` (`SPINGALETT_MODEL_EXTENSION`), appended when a file name
  has none; files saved as `.nn` load as before.
- A generator that answers 0 to the first request of an epoch is asked once more before training
  stops, so generators that mark the end of each pass with a 0 also work with `sample_count`.
- The shared library's soname carries the minor version while the major version is 0
  (`libspingalett.so.0.4`), because 0.x minor releases are not ABI compatible.
- `Examples/MNIST.c` holds out 5,000 training images for validation, keeps the best epoch and
  evaluates the test set once.

### Fixed
- Windows builds with MinGW: aligned buffers came from `aligned_alloc`, which the Windows C
  runtime does not provide, and were released with `free()`; they now use `_aligned_malloc` and
  `_aligned_free`, as MSVC builds already did.
- The examples measure time with a fallback where C11 `timespec_get` is missing (some Windows C
  runtimes).

## [0.3.0] - 2026-10-03

### Added
- Native single-precision matrix multiplication with AVX-512, AVX/FMA and portable C kernels and
  OpenMP parallelism. Batch training and inference no longer need OpenBLAS to run as
  matrix-matrix products.
- `predict()`: batched inference over many samples (Python: `Network.forward` on a 2-D array).
- `Examples/MNIST.c` (with `Examples/download_mnist.sh`): an MLP reaching about 98% test
  accuracy in a few seconds.
- `Examples/benchmark_pytorch.py`, the PyTorch counterpart of `Examples/Benchmark.c`.

### Changed
- Full-batch and mini-batch training use matrix-matrix products in every compute mode; OpenBLAS
  is one of two GEMM providers. Without OpenBLAS, full-batch training of the benchmark network is
  about 7x faster than in 0.2 and on par with OpenBLAS when using OpenMP.
- Large batches are processed in chunks of 2048 samples whose gradients are accumulated, which
  bounds the memory used by full-batch training on large datasets.
- `Examples/Benchmark.c` reports full-batch, mini-batch and inference throughput per backend.

## [0.2.0] - 2026-10-03

### Added
- Dropout: `LayerArgs.dropout_rate` (inverted dropout on hidden layers, all backends and strategies).
- Learning-rate schedules: `TrainArgs.lr_scheduler` / `lr_scheduler_data` with the built-ins
  `spingalett_lr_cosine_decay`, `_linear_warmup`, `_step_decay` and `_warmup_cosine`.
- Data generators: `MODE_GENERATOR_FUNCTION` with `TrainArgs.generator` / `generator_data`.
- Gradient clipping (`max_grad_norm`) for per-sample training; it previously applied to batch
  training only.
- `TrainArgs.blas_num_threads` to control OpenBLAS threads during training (automatic by default).
- `TrainArgs.do_not_shuffle`, `WEIGHT_INITIALIZATION_LECUN`.
- `spingalett_seed()`, `spingalett_version()` and the `SPINGALETT_VERSION_*` macros.
- Error codes `SPINGALETT_ERR_FILE_IO` and `SPINGALETT_ERR_FORMAT_VERSION`.
- Python bindings in `Bindings/Python`.
- Test suite (`ctest`) and GitHub Actions CI, including sanitizer and portable (non-AVX) builds.
- CMake package configuration (`find_package(Spingalett)`, target `Spingalett::spingalett`) and
  the options `BUILD_TESTS`, `SPINGALETT_NATIVE_ARCH`, `SPINGALETT_BIN_DIR`, `SPINGALETT_LIB_DIR`.

### Changed
- `WEIGHT_INITIALIZATION_XAVIER` is Glorot normal (variance 2 / (fan_in + fan_out)). The previous
  behavior (variance 1 / fan_in) is `WEIGHT_INITIALIZATION_LECUN`.
- Adam, AdamW and RMSProp add epsilon to the square root of the second moment (`sqrt(v) + eps`),
  as PyTorch and TensorFlow do, instead of `sqrt(v + eps)`.
- Per-sample and mini-batch training reshuffle the samples every epoch (`do_not_shuffle` restores
  the previous order).
- Weight decay is an L2 term for SGD, Momentum, RMSProp and Adam in every strategy. Batch RMSProp
  previously used a decoupled term. AdamW keeps decoupled decay.
- Cross-entropy loss requires a softmax or sigmoid output layer; other outputs are rejected
  instead of training with an incorrect gradient.
- Model files use format version 2 (per-layer dropout rates); version 1 files remain loadable.
- `load_spingalett()` returns `NULL` for truncated files.
- `TrainArgs.inputs` and `targets` are `const float *`. `NeuralNetwork`, `LayerArgs` and
  `TrainArgs` gained fields; code using designated initializers or positional arguments keeps
  compiling, but binaries must be rebuilt.
- Training is substantially faster: denormals are flushed to zero while training, the optimizers
  use shared AVX kernels in every strategy, tanh/sigmoid/softmax are vectorized, and OpenMP
  only parallelizes layers that are large enough (per-sample training up to 11x, OpenBLAS
  full-batch training +43%).

### Fixed
- Adam/AdamW bias correction restarted on every `train()` call and after loading a checkpoint.
- Truncated files were loaded as partially initialized networks; an unrelated earlier error made
  `load_spingalett()` fail.
- 32-bit overflow of weight counts in the serializer.
- Softmax output layers wider than 65,535 neurons looped forever.
- FP16 and BF16 conversion truncated instead of rounding to nearest even.
- Stale gradients after switching compute backends between `train()` calls.
- Invalid training mode, strategy or optimizer values were used as array indices.
- The compute-mode fallback warning was logged on every `forward()` call.
- The `Benchmark` example oversubscribed CPUs with a hard-coded thread count.

## [0.1.0]

Initial release.

[0.7.0]: https://github.com/pka-human/Spingalett/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/pka-human/Spingalett/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/pka-human/Spingalett/compare/v0.4.1...v0.5.0
[0.4.1]: https://github.com/pka-human/Spingalett/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/pka-human/Spingalett/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/pka-human/Spingalett/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/pka-human/Spingalett/compare/0a1dd16...v0.2.0
