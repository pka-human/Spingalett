# Changelog

All notable changes to this project are documented in this file. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/); before 1.0, a minor release may contain breaking
changes, which are listed under **Changed**.

## [Unreleased]

### Changed
- `train()` on the GPU keeps the network's parameters on the device between epochs instead of
  copying them (and the gradients and optimizer moments) back after each: they come back where they
  are read (a callback, an autosave, the NaN check, the best epoch's copy) and when training ends.
  The 784-512-1000-10 MLP of `Examples/Benchmark.c` trains full batches 22% faster on an RTX 4050
  Laptop GPU (5 epochs: 553,000 to 673,000 samples per second).

- A network on the GPU takes its buffers from a few allocations instead of one each, uploads and
  downloads its arrays in one submission instead of one per array, and, where the host can write
  into all of the device's memory (resizable BAR, unified memory), the host writes each chunk's
  inputs and the parameters straight into it instead of the device copying them over from host
  memory. The MLP trains full batches 1.2 times as fast again (5 epochs: 673,000 to 831,000-876,000
  samples per second; an epoch 25.6 to 21.7 ms) and infers 1.5 times as fast (1,150,000 to
  1,800,000 samples per second). `SPINGALETT_GPU_NO_HOST_WRITES=1` keeps the copies.

- The GPU's matrix products choose their tiles by the device's own timestamps, in rounds that run
  every candidate once in one submission (a round of single runs over up to 16 candidates, then
  four rounds of the eight fastest, each candidate's fastest round kept), after keeping an idle
  device busy for 25 ms to raise its clocks. The tile chosen is now within the noise of the
  fastest on the products of `SpingalettGpuTests bench` (it was up to 25% slower: the MLP's
  512 -> 1000 forward pass 415 us against 331), and timing takes half as long (the 255 products of
  `Bin/Benchmark gpu`: 10.7 to 5.4 s, once per process).

- With `PRECISION_BFLOAT16`, the GPU keeps the layers' outputs (all but the output layer's, and the
  network's inputs) as bfloat16, the values the products round them to anyway, as PyTorch's autocast
  does: half the memory and half the bytes every pass moves. The matrix units' kernel reads them
  eight to a load and moves their bits to shared memory as they are; normalizations, pooling,
  additions, concatenations, upsampling and dropout read and write bfloat16 and compute in single
  precision (variants of their kernels built with 16-bit storage, which only devices with bfloat16
  matrix units are asked for); the host rounds the inputs as it sends them, halving the bytes.
  Parameters, gradients and the optimizer stay in single precision. On an RTX 4050 Laptop GPU, in
  bfloat16: ResNet-20 trains at 14,300 samples per second instead of 11,500 and infers at 51,100
  instead of 29,300, the U-Net 5,260 and 12,500 instead of 4,190 and 7,670, the MLP infers at
  2,940,000 instead of 1,840,000; CIFAR-10 with `Examples/CIFAR10.c resnet20` reaches the same test
  accuracy after 10 epochs (85.10% against 85.04%), 24% faster. `SPINGALETT_GPU_NO_BF16_STORAGE=1`
  keeps them in single precision.

- The GPU's pooling backward pass takes windows that tile the input (stride the window, no padding,
  as 2 x 2 max pooling of stride 2) a thread per window: it finds the maximum once and writes the
  window's cells, instead of every cell searching its window again. 5.3 times as fast on the MNIST
  CNN (322 to 60 us a pass), which trains 12% faster in single precision (81,900 to 91,600 samples
  per second) and 26% faster in bfloat16 (95,000 to 120,000); with batch normalization 10% and 16%.

### Fixed
- Parameters a training callback wrote on the GPU (`spingalett_set_parameters()`) were ignored by
  the epochs after it: they now go to the device before the next epoch.

## [0.13.0] - 2026-10-09

"Layers": transposed convolutions, upsampling and layer normalization, for U-Nets and more of ONNX,
on the CPU, the GPU, in files, the engine and deployment models, imported from ONNX and PyTorch.

### Added
- Transposed 2D convolutions: `conv_transpose2d()` (`LAYER_CONV_TRANSPOSE2D`; Python
  `ConvTranspose2D`, `add_conv_transpose2d()`) with strides, padding, output padding
  (`LayerArgs.output_padding`, `output_padding_h`, `output_padding_w`) and groups; the output has
  `(in - 1) stride - 2 padding + kernel + output_padding` cells an axis. A filter per output
  channel, as a convolution's. The passes are those of the convolution transposed, swapped: the
  forward pass is its data gradient, the data gradient its forward pass, the weight gradient its
  weight gradient with input and output gradient exchanged, on the CPU's kernels and the GPU's.
- Upsampling: `upsample2d()` (`LAYER_UPSAMPLE`, `UpsampleMode`; Python `Upsample2D`,
  `add_upsample()`) by integer factors (`stride_h`, `stride_w`, default 2), `UPSAMPLE_NEAREST` or
  `UPSAMPLE_BILINEAR` (the cells' centres aligned and the edges repeated, PyTorch's
  `align_corners=False`; weights from integers, the same on every platform).
- Layer normalization: `layer_norm()` (`LAYER_LAYER_NORM`; Python `LayerNorm`, `add_layer_norm()`)
  over each cell's channels (a dense layer's outputs are one cell), gamma and beta per channel, no
  running statistics, no weight decay on gamma.
- All three train on the CPU and the GPU (deterministic, the GPU's results within rounding of the
  CPU's), save in `.slett` format version 7 (`docs/ModelFormat.md`; networks without them keep
  their versions), and run in the engine and in deployment models in every precision, batch
  normalization folded into a transposed convolution before it. Float deployment models run
  transposed convolutions through the convolution kernels in batches, integer ones through the
  engine's pass, which they reproduce exactly.
- ONNX import: `ConvTranspose` (groups, strides, padding, output padding), `Resize` and `Upsample`
  by constant integer factors (nearest where it reads cell floor(o / factor): asymmetric
  coordinates rounded down or half-pixel ones rounded; linear with half-pixel coordinates), and
  `LayerNormalization` over a vector or over a map's channels between a `Transpose` to channels last
  and one back (LayerNorm2d). A U-Net exported by both PyTorch exporters and a network with both
  kinds of layer normalization import within 6e-8 of PyTorch's outputs.
- PyTorch weights: `ConvTranspose2d` (weight `[in, out / groups, kh, kw]`) and `LayerNorm` modules.
- `Examples/Segmentation.c`: a U-Net segments synthetic 64 x 64 images of circles, squares and
  triangles (a sigmoid a pixel and class, new images every epoch from a generator; `bilinear`,
  `ln`, `gpu`, `bf16`). 12 epochs on an RTX 4050 Laptop GPU: 15 s, mean IoU 0.882, pixel accuracy
  98.5%; the INT8 model keeps 0.878.
- `Bin/Benchmark` and `benchmark_pytorch.py` time that U-Net.

### Changed
- `LayerType` has three new values before `LAYER_TYPE_COUNT`; `LayerArgs`, `SpingalettNetworkLayer`
  and `SpingalettLayerInfo` have new fields (output padding, the upsampling mode), mirrored in the
  Python structures. `SPINGALETT_FORMAT_VERSION` is 7.
- GPU: the output layer's kernel takes rows of more than 64 outputs a workgroup each (sums in a fixed
  tree): the U-Net's loss kernel takes 45 us instead of 2,548 us a batch; narrower rows keep their
  thread a row and their results.
- Float deployment models run transposed convolutions in batches through the convolution kernels
  (they ran sample by sample through the engine): the U-Net's FP32 model predicts 3,150 images a
  second instead of 2,440, its FP16 model 3,190 instead of 2,140 (i7-12650H, 16 threads).
- Nothing slower: interleaved runs of `Bin/Benchmark` on the i7-12650H and its RTX 4050 Laptop GPU,
  0.12.0 against 0.13.0 (three each, one and eight threads), agree within 4% on every CPU workload
  and 5% on every GPU one, either way.
- The U-Net, images per second, training and inference (medians of three interleaved runs): one
  thread 146 and 511 (PyTorch 2.14: 134 and 292), eight threads 598 and 2,467 (365 and 754); on the
  GPU 3,279 and 6,552 (PyTorch with TF32: 3,283 and 6,329; in single precision 3,059 and 6,564),
  with bfloat16 products 3,707 and 6,880 (PyTorch's autocast, which keeps activations in bfloat16:
  4,950 and 11,918).

### Fixed
- The cross-entropy of sigmoid outputs saturated at 1 was NaN (a target of 1) or infinite (a target
  of 0), on the CPU and the GPU: the reported losses, validation and anything monitoring them went
  NaN once per-pixel sigmoids saturated. The bound on 1 - o is now taken through `fmaxf()`, which
  reassociation cannot fold back into a comparison of o.
- The header's comment on ONNX import said external data files were not read (0.12 reads them).

## [0.12.0] - 2026-10-09

"Tensor cores": the GPU's matrix units in bfloat16, the custom-loop API on the GPU, fixes from a
review of the GPU backend; model import ten to a hundred times as fast; networks that predict hold
their parameters once.

### Added
- `spingalett_set_gpu_precision(PRECISION_BFLOAT16)` (Python `set_gpu_precision()`): the GPU's matrix
  products in bfloat16 on its matrix units (tensor cores; `VK_KHR_cooperative_matrix` with
  `VK_KHR_shader_bfloat16`), the products added in single precision; the parameters, the optimizer
  and batch normalization stay in single precision, and so do products smaller than a block of the
  matrix units. Runs stay deterministic. It returns whether the GPU multiplies in the precision set.
  The tiles of the matrix units are many subgroups with one or two accumulators each, all of them
  timed on first use. On an RTX 4050 Laptop GPU (`Bin/Benchmark gpu`, medians of three interleaved
  runs, samples per second, single precision then bfloat16): ResNet-20 trains at 8,750 and 9,960 and
  infers at 22,630 and 25,300; the MNIST CNN trains at 73,680 and 85,950 and infers at 219,600 and
  257,300; the 784-512-1000-10 MLP trains mini-batches of 64 at 225,500 and 264,100. ResNet-20
  reaches 91.61% test accuracy on CIFAR-10 in bfloat16 (`Examples/CIFAR10.c bf16`), 91.55% in single
  precision. PyTorch 2.14 with autocast to bfloat16 on the same GPU (`benchmark_pytorch.py
  --cuda-bf16`, new) trains ResNet-20 at 12,160 and the MNIST CNN at 99,120.
- The custom-loop API on the GPU: a trainer made with `COMPUTE_VULKAN` (`spingalett_trainer_new()`,
  Python `Trainer`) runs its forward, backward and optimizer passes there, the outputs coming back
  for the caller's loss and gradients adding up on the device until the step; the parameters stay
  on the GPU between passes, and functions that read the network copy them back first. ResNet-20 in
  batches of 128 through `spingalett_train_on_batch()`: 7,372 samples/s, as fast as `train()`, against
  972 on the CPU.
- ONNX models whose weights are kept in external files (`torch.onnx.export` of models over 2 GB,
  `onnx.save_model(..., save_as_external_data=True)`) import from their path; the files must lie in
  the model's folder.
- PyTorch weights stored as views (a transposed or permuted tensor in a state dict) and 32-bit
  integer tensors load.
- `Bin/Benchmark gpu` runs the GPU's rows only, and the GPU's rows run again in bfloat16 where the
  GPU has matrix units; `SpingalettGpuTests bench` lists every tile's time with
  `SPINGALETT_BENCH_TILES` set.

### Changed
- Networks allocate their gradients and optimizer state when they first train (`train()`,
  `spingalett_trainer_new()`, a file with optimizer state, or setting a gradient), so that networks
  loaded or imported to predict hold their parameters once instead of four times. Parameter arrays
  grow by half again when a layer does not fit, instead of to the exact size every time: building a
  network of 1000 layers, which copied its parameters once per layer, takes 0.02 s instead of 5.6 s.
- ONNX import and PyTorch weights: files are mapped instead of read, and each tensor is converted
  straight into its layer in one pass (no float copy of every initializer, no table of indices as
  large as the tensor, no copy of the network's parameters to restore on error: everything is
  checked before anything is written); names resolve through hash tables. On an i7-12650H:
  ResNet-50 (100 MB) imports in 0.085 s instead of 1.02 s, peaking at 360 MB instead of 1139 MB; a
  126M-parameter VGG head (504 MB) in 0.17 s instead of 1.94 s, peaking at 1271 MB instead of
  4747 MB, and its state dict loads in 0.08 s instead of 1.0 s.
- GPU: the executor waits for the older of the two chunks in flight before filling the next one
  (it waited for both), and a chunk's last barrier no longer holds back the next one's copies; with
  the other changes the MLP trains 7 to 11% faster on the RTX 4050 Laptop GPU and infers 19% faster,
  and the MNIST CNN trains 5% faster (`Bin/Benchmark`, against 0.11).
- GPU: matrix products of more row tiles than a dispatch may have go in parts, tiles whose columns
  would exceed the device's limit are not chosen, and the gradient norm's sums loop over at most
  65535 workgroups (Intel and lavapipe allow 65535 workgroups in x; every device in y).
- GPU: tanh takes an odd polynomial below |x| = 0.3, as the CPU's vector kernel does (the formula
  through exp lost its relative precision there); the matrix kernel's accumulators are `precise`.
- Building: a glslc too old for the bfloat16 kernel (Ubuntu 24.04's) builds the library without it,
  which then multiplies in single precision; release packages and wheels take shaders compiled once
  with a current shaderc, checked by spirv-val.
- Nothing on the CPU: interleaved runs of `Bin/Benchmark` on the i7-12650H, 0.11.0 against 0.12.0,
  agree within 3% on every workload.

### Fixed
- GPU: validation during training ran batch normalization with the validation chunk's own
  statistics and moved the running statistics towards the validation data; it uses the running
  statistics now, as the CPU does.
- GPU: a concatenation (or a one-input addition, which ONNX import makes for activations it cannot
  fuse) that directly followed another could read its input without a barrier.
- GPU: a chunk whose command recording failed stayed cached and was submitted half-recorded by the
  next call of its size; a lost device made predictions and losses read garbage as success.
- GPU: the bfloat16 kernel used the Vulkan memory model and 16-bit floats without the device
  features enabled (NVIDIA tolerates it, a stricter driver may refuse the pipeline); it takes its
  subgroup from `gl_SubgroupID` and asks for complete subgroups where the device allows.
- GPU: the 32-bit index check skipped the input layer; devices were asked for Vulkan 1.2 features
  before their version was checked; the pipeline cache passed NULL to `memcpy`.
- README: the DeepWiki badge (its image server refuses GitHub's image proxy) and the introduction,
  which still said the library ran on the CPU only.

## [0.11.0] - 2026-10-08

"GPU": training and inference on a GPU through Vulkan compute, deterministic, with every kind of
layer the CPU runs.

### Added
- `COMPUTE_VULKAN`: `train()` (full-batch and mini-batch strategies), `predict()` and `evaluate()`
  on a GPU through Vulkan compute (NVIDIA, AMD and Intel GPUs; Apple GPUs through MoltenVK);
  `spingalett_gpu_device()` names the device (the first discrete GPU, or `SPINGALETT_GPU_DEVICE`).
  The Vulkan loader is opened at run time: the library loads and runs on the CPU without Vulkan, and
  without a usable device the mode falls back to the CPU with a warning. Every kind of layer,
  activation, loss and optimizer runs on the GPU, with dropout (the CPU's masks), gradient clipping,
  label smoothing, augmentation (on the host, overlapped), validation and early stopping; the
  parameters stay on the device for a `train()` call and come back at the end of every epoch.
  Sums run in fixed orders, so GPU runs repeat bit for bit. Python: `ComputeMode.VULKAN`,
  `gpu_device()`. Measured on an RTX 4050 Laptop GPU (`Bin/Benchmark`, medians of three runs,
  samples per second; in parentheses the eight threads of the i7-12650H, then PyTorch 2.14 with CUDA
  and cuDNN on the same GPU, with its default TF32 and in single precision): ResNet-20 trains at
  8,900 (1,573; 8,508 and 7,139) and infers at 22,892 (6,163; 19,742 and 19,409); the MNIST CNN
  trains at 62,926 (14,298; 56,906 and 59,993) and infers at 227,667 (51,250; 132,016 and 143,565);
  with batch normalization 48,387 (10,791; 48,481 and 48,477) and 149,371 (53,411; 107,254 and
  114,126); the 784-512-1000-10 MLP trains mini-batches of 64 at 188,642 (52,688; 80,581 and
  80,413), full batches at 497,191 (112,991; about 1,050,000) and infers at 967,009 (284,002; about
  2,960,000, its data already on the GPU).
- The GPU's matrix kernel serves dense layers, convolutions (windows read through a tap table,
  groups), data gradients (one stride-1 product per phase of a stride) and weight gradients (sums
  split into slices added in a fixed order), with vector loads along each operand's contiguous axis;
  the tile of each product shape is chosen by timing up to eight candidates on first use (about a
  second for ResNet-20; `SPINGALETT_GPU_TUNE=0`: estimated), which cannot change a result since every
  tile adds in the same order.
  `SPINGALETT_GPU_PROFILE=1` prints the GPU time per kernel at exit.
- `Examples/CIFAR10.c gpu` trains on the GPU: ResNet-20 reaches the same 91.55% test accuracy as on
  the CPU, in 9 minutes for 100 epochs (93 on twelve threads of the i7-12650H).
- Building: `SPINGALETT_VULKAN` (`AUTO`, `ON`, `OFF`) needs `glslc` and the Vulkan headers;
  `SPINGALETT_SPIRV_DIR` takes another build's compiled shaders. Release packages and wheels carry
  the backend.
- Tests: `SpingalettGpuTests` checks the matrix kernel in every mode and tile against double
  precision and that tiles agree bit for bit; the `gpu` group trains networks with every kind of
  layer on the GPU and the CPU and compares them. CI runs both on Mesa's lavapipe.
- `Examples/benchmark_pytorch.py --cuda` (PyTorch's defaults) and `--cuda-fp32` (no TF32).
- `Examples/CIFAR10.c resnet32` reaches 92.39% test accuracy in 100 epochs (INT8 model 92.43%,
  480 KB).

### Changed
- `ComputeMode` has a new value, `COMPUTE_VULKAN`, before `COMPUTE_COUNT`.
- Nothing on the CPU: interleaved runs of `Bin/Benchmark` on the i7-12650H, 0.10.0 against 0.11.0,
  agree within 2% on every workload and thread count (the runs of either vary by more).

### Fixed
- README: a repeated phrase in the performance section.

## [0.10.0] - 2026-10-08

"Graphs": networks become directed acyclic graphs of layers (residual connections, concatenated
branches), models come in from ONNX and PyTorch, the Python package goes to PyPI with the library
inside, and convolutions read their windows straight from the image.

### Added
- Networks as graphs: `LayerArgs.inputs` and `.input_count` name the earlier layers a layer reads
  (by default the one before it, so chains are built as before), and `layer()` and the other
  builders return the new layer's index (`SPINGALETT_NO_LAYER` on error). New kinds of layers:
  `LAYER_ADD` (`add_layers()`, the sum of layers of one shape: residual connections),
  `LAYER_CONCAT` (`concat_layers()`, layers side by side along the channels) and
  `LAYER_GLOBAL_AVG_POOL` (`global_avg_pool2d()`). A layer reads at most `SPINGALETT_MAX_INPUTS`
  (16) layers; the last layer is the output, and every other layer must feed a later one before a
  network trains or predicts. Training, the step API, `predict()`, `forward()`, `evaluate()`,
  deployment models in every precision and the inference engine run graphs; training stays
  bit-identical on any number of threads (a layer read by several gets their gradients in a fixed
  order). `SpingalettNetworkLayer` and `SpingalettLayerInfo` list a layer's inputs.
- `.slett` format version 6 (see `docs/ModelFormat.md`): each layer's inputs and where its output
  lives among the engine's activations, planned by the writer so that outputs whose lives do not
  overlap share memory. Chains are still written as versions 3 to 5. Batch normalizations fold into
  products that feed nothing else.
- ONNX import: `spingalett_import_onnx()` and `_from_memory()`, `ModelTool import`, Python
  `Network.from_onnx()` and `Network.from_torch()` (through an ONNX export in memory). A protocol
  buffer reader of the library's own; Conv (groups, strides, symmetric and same padding), Gemm,
  MatMul with a constant operand (a following Add becomes the bias), MaxPool, AveragePool,
  GlobalAveragePool and ReduceMean over the height and width, BatchNormalization, Add, Concat along
  the channels, Relu, Sigmoid, Tanh, LeakyRelu, Softmax, Flatten, flattening Reshape, Identity,
  Dropout and Constant; weights reordered for channels-last data; errors name the operator and node.
  Models exported from PyTorch 2.14 with either exporter agree with PyTorch within 1e-7.
- PyTorch weights: `spingalett_load_pytorch()` and `_from_memory()` copy a state dict saved with
  `torch.save` (or a checkpoint holding one) or a safetensors file into a network of the same
  layers, module by module in order (or in an explicit order), checking shapes and leaving the
  network unchanged on error. The pickle inside `.pt` files is interpreted without running any of
  it. Python: `Network.load_pytorch()` also takes a state dict directly.
- Label smoothing: `TrainArgs.label_smoothing`, `spingalett_trainer_set_label_smoothing()`, Python
  `TrainConfig.label_smoothing`.
- Reduce on plateau: `TrainArgs.lr_plateau_factor`, `.lr_plateau_patience` and `.lr_plateau_min_lr`
  scale the learning rate (the schedule's, when there is one) after epochs without improvement of
  the monitored value; Python `TrainConfig` has the same fields.
- Python wheels with the library inside (manylinux 2.28 x86-64 and AArch64, Windows x86-64, macOS
  universal2, for any Python 3), built and tested by the release workflow, attached to releases and
  uploaded to PyPI by trusted publishing. The bindings are a typed package (`py.typed`).
- `Examples/CIFAR10.c` builds ResNet-20 to ResNet-56 (`resnet20`, `resnet32`, ..., `wide`), trained
  with SGD, momentum, a cosine schedule and label smoothing: ResNet-20 reaches 91.55% test accuracy
  in 100 epochs (INT8 model 91.58%, 281 KB); `Bin/Benchmark` and `Examples/benchmark_pytorch.py`
  measure ResNet-20 (on the i7-12650H, 1.1 to 1.8 times PyTorch's training speed and 1.5 to 2.8
  times its inference speed).
- Tests: the `graph` group (gradient checks of residual, projected, branching and multiply-read
  layers, determinism, inference in every precision, files, folding, memory, validation) and the
  `onnx` group (models and state dicts exported from PyTorch, written by
  `Tests/make_test_models.py`, unsupported operators, truncated files); the importers ran 24,000
  mutated files under AddressSanitizer and UBSan.

### Changed
- Convolutions run as indirect matrix products: the matrix kernels read each pixel's window from the
  image through one pointer per kernel tap (Dukhan's indirect convolution) instead of gathering
  windows and transposing them into panels, for the forward pass, the data gradient (of strided
  convolutions a phase of input cells at a time, over the taps that cover them only: a quarter of
  the products of a 3 x 3 kernel with stride 2) and the weight gradient (tiles of filters by window
  elements summed over the pixels in registers, in slots fixed by the shape). Threads take tiles
  dynamically, which keeps the performance cores of hybrid processors busy. Measured on an
  i7-12650H (medians of three interleaved runs of `Bin/Benchmark`, samples per second, 0.9 ->
  0.10): the MNIST CNN trains at 2,615 -> 3,090 on one thread and 12,230 -> 14,591 on eight and
  infers at 9,174 -> 11,183 and 44,227 -> 50,862; with batch normalization 2,259 -> 2,639,
  9,745 -> 11,027, 9,205 -> 11,378 and 43,980 -> 55,646. ResNet-20 trains at 234 -> 377 and
  986 -> 1,584 and infers at 958 -> 1,387 and 4,798 -> 6,402 (against 0.10 built without the new
  kernels, `-DSPINGALETT_NO_DIRECT_CONV`). Dense networks and deployment models are unchanged.
  Float results of convolutions change in the last bits (another summation order); training
  remains the same bits on any number of threads. Depthwise convolutions, kernels of more than 64
  taps and OpenBLAS keep the previous products.
- `layer_struct_arguments()` returns the layer's index (it returned nothing).
- `LayerArgs`, `SpingalettNetworkLayer`, `SpingalettLayerInfo`, `SpingalettModel` and `TrainArgs`
  have new fields; zero-initialized arguments keep the previous behavior.
- Inference outputs share memory where their lives do not overlap: `predict()` needs a few of the
  widest layers per sample instead of all of them, which also makes it faster.
- The Python bindings are a package (`Bindings/Python/spingalett/`); `import spingalett` is
  unchanged.

### Fixed
- Release builds with GCC 16 crashed in `evaluate()` (and four test groups failed): with LTO, GCC
  16.1 dropped the stack realignment of a function that inlined the thread-local error state and
  still read its arguments through it. `set_error()` now stays out of line.

## [0.9.0] - 2026-10-08

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

### Fixed
- `.slettd` readers reject index entries whose offset plus size wraps around (they read before the
  file's buffer), and metadata records or set numbers of 2^32 - 1, which plus one named the inputs
  (a name record shorter than 4 bytes scanned past the metadata). `spingalett_save_dataset()`
  rejects `extra_target_count` of 255 or more instead of overflowing the set count. The Python
  bindings reject a negative `target_set`.
- Streaming and loading `.slettd` files of 2 GB or more on Windows, where `long` file offsets have
  32 bits.
- Threads opening data sets at the same time no longer race to build the table of 8-bit values.
- Builds with `SPINGALETT_PORTABLE_KERNELS` and without `-march=native` link again (broken in 0.8).

## [0.8.0] - 2026-10-08

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

[0.13.0]: https://github.com/pka-human/Spingalett/compare/v0.12.0...v0.13.0
[0.12.0]: https://github.com/pka-human/Spingalett/compare/v0.11.0...v0.12.0
[0.11.0]: https://github.com/pka-human/Spingalett/compare/v0.10.0...v0.11.0
[0.10.0]: https://github.com/pka-human/Spingalett/compare/v0.9.0...v0.10.0
[0.9.0]: https://github.com/pka-human/Spingalett/compare/v0.8.0...v0.9.0
[0.8.0]: https://github.com/pka-human/Spingalett/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/pka-human/Spingalett/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/pka-human/Spingalett/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/pka-human/Spingalett/compare/v0.4.1...v0.5.0
[0.4.1]: https://github.com/pka-human/Spingalett/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/pka-human/Spingalett/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/pka-human/Spingalett/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/pka-human/Spingalett/compare/0a1dd16...v0.2.0
