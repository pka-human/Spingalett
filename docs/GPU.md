# Backends and the GPU

The CPU's kernels and threads, and everything that runs on NVIDIA's GPUs through CUDA and on any GPU through Vulkan.

## Backends and threading

```c
spingalett_set_compute_mode(COMPUTE_OPENBLAS);             /* or SINGLE_THREADED, OPENMP, CUDA, VULKAN */
spingalett_set_num_threads(8);                             /* 0 = runtime default */
```

Full-batch and mini-batch training and `spingalett_predict()` process samples in chunks of up to
2048 as matrix-matrix products (fewer when a sample's activations are large, so that a chunk stays
near 64 MB); per-sample training and `spingalett_forward()` of dense networks use matrix-vector
kernels.

-  `SPINGALETT_COMPUTE_SINGLE_THREADED` uses the built-in kernels: AVX-512 or AVX/FMA when the
  compiler targets them, portable C otherwise.
-  `SPINGALETT_COMPUTE_OPENMP` uses the same kernels on all threads; operations that are too small
  to benefit run serially. Threads split matrix products by tiles of the result (and long
  weight-gradient sums by slots fixed by the shape), so results are the same bits on any number of
  threads and in single-threaded mode.
-  `SPINGALETT_COMPUTE_OPENBLAS` delegates matrix products to OpenBLAS. It is usually on par with
  OpenMP for large batches and slower for small ones. During training, `blas_num_threads = 0` uses a
  single OpenBLAS thread when each call is too small to amortize threading and the configured thread
  count otherwise, and restores the caller's setting afterwards; outside `spingalett_train()`,
  OpenBLAS uses its own thread configuration (`OPENBLAS_NUM_THREADS`).
-  `SPINGALETT_COMPUTE_CUDA` (an NVIDIA GPU) and `SPINGALETT_COMPUTE_VULKAN` (any GPU) run training,
  the custom-loop API, `spingalett_predict()` and `spingalett_evaluate()` on a GPU
  ([The GPU](#the-gpu)); what stays on the CPU runs as with `SPINGALETT_COMPUTE_OPENMP`.
-  A requested backend that was not compiled in falls back to single-threaded with a one-time
  warning (a GPU mode without a usable device: to the CPU, as above).

Results agree across backends up to floating-point rounding. For the duration of
`spingalett_train()`, denormal floats are flushed to zero on the calling thread and the OpenMP
workers; the previous floating-point mode is restored afterwards.

A network must not be used by several threads at the same time. Compute mode, thread count and
logging settings are process-wide; the random generator and the error state are per thread.

## The GPU

With `SPINGALETT_COMPUTE_CUDA` or `SPINGALETT_COMPUTE_VULKAN`, `spingalett_train()` (full-batch and
mini-batch strategies), the custom-loop API (`spingalett_trainer_*`), `spingalett_predict()` and
`spingalett_evaluate()` run on a GPU. The two modes share one executor, and what follows holds for
both:

-  `SPINGALETT_COMPUTE_CUDA` runs on NVIDIA GPUs of compute capability 8.0 or later (Ampere and
  newer: GeForce RTX 30 to 50, A100, H100) with a driver of CUDA 11.0 or later, on the library's own
  kernels (`Src/Gpu/Cuda`): matrix products in single precision and, in bfloat16, on the tensor
  cores, convolutions as implicit products, depthwise convolutions, normalizations, pooling. They are
  compiled to PTX when the library is built and kept in it compressed; the driver compiles a kernel
  for the GPU when it is first used and keeps it in its cache (`~/.nv/ComputeCache`). Neither the
  CUDA toolkit nor cuBLAS or cuDNN is needed to build or to run. `spingalett_cuda_device()` names the
  GPU, or returns `NULL` without one.
-  `SPINGALETT_COMPUTE_VULKAN` runs through Vulkan compute: NVIDIA, AMD and Intel GPUs on Linux and
  Windows, Apple GPUs through MoltenVK (from the Vulkan SDK, or
  `brew install molten-vk vulkan-loader`). `spingalett_gpu_device()` names the device, or returns
  `NULL` when there is none (it needs Vulkan 1.2 with buffer device addresses).

The library opens the CUDA driver or the Vulkan loader when a mode is first used, so it needs
neither to load or to run on the CPU; without a usable device the mode falls back to the CPU with a
warning. On an NVIDIA GPU the CUDA mode is the faster ([Performance](Performance.md)).

```c
spingalett_set_compute_mode(COMPUTE_CUDA);                 /* or COMPUTE_VULKAN */
const char *gpu = spingalett_cuda_device();               /* spingalett_gpu_device() for Vulkan */
printf("training on %s\n", gpu ? gpu : "the CPU");
train(.net = net, .inputs = x, .targets = y, .sample_count = n, .epochs = 30,             /* as on the CPU */
      .training_strategy = STRATEGY_SMALL_BATCH, .batch_size = 128);
```

-  Parameters, their gradients and the optimizer's moments stay in GPU memory, and so does the
  network's copy there after `spingalett_train()` returns: functions that read the parameters
  (`spingalett_predict()`, `spingalett_save()`, `spingalett_get_parameters()`, a callback's reads
  during training, and the others) copy them back first, parameters set on the host go to the device
  before the next epoch, and the next `spingalett_train()` of the same batch size and optimizer
  trains on the copy again without copying anything. The network frees it when it is freed, gains a
  layer, trains on the CPU or makes a trainer. The host prepares a batch (gathering, augmentation,
  label smoothing, and writing it into the device's memory where it can) while the GPU trains on the
  one before, and takes the losses back at the end of every epoch.
-  Data sets can stay in the GPU's memory: `spingalett_device_data_new(values, count, size)` copies
  rows of floats there once, and the `device_inputs` and `device_targets` fields of
  `spingalett_train()` (with `device_val_inputs` and `device_val_targets`), `spingalett_predict()`
  and `spingalett_evaluate()` take such sets in place of arrays, either or both. Only the rows'
  indices cross the bus then: the GPU gathers each chunk's rows, augments and smooths them as the
  host would (training on a set gives the bits of training on the host's arrays), and reads them
  where they are when they come in order (inference, full batches). With
  `SPINGALETT_PRECISION_BFLOAT16` a set keeps a bfloat16 copy of its rows, made on first use, half
  its size again. On the CPU those calls copy the rows back first.

  ```c
  SpingalettDeviceData *x = spingalett_device_data_new(images, n, 784), *y = spingalett_device_data_new(labels, n, 10);
  train(.net = net, .device_inputs = x, .device_targets = y, .sample_count = n, .epochs = 30,
        .training_strategy = STRATEGY_SMALL_BATCH, .batch_size = 128, .augment_shift = 2);
  predict(.net = net, .device_inputs = x, .outputs = out, .sample_count = n);
  spingalett_device_data_free(x);
  spingalett_device_data_free(y);
  ```
-  Every kind of layer, activation, loss and optimizer runs on the GPU, with dropout (the same masks
  as on the CPU), gradient clipping and batch normalization. Per-sample training,
  `spingalett_forward()` and deployment models run on the CPU.
-  A trainer (`spingalett_trainer_new()`) made with a GPU mode runs its passes on
  the GPU, one at a time: the forward pass's outputs come back for the caller's loss, and backward
  passes from targets or from the caller's dL/d(output) add up on the device until the step. The
  parameters stay there between passes; functions that read the network (`spingalett_predict()`,
  `spingalett_save()`, `spingalett_get_parameters()` and the others) copy them back first, and
  parameters set on the host go to the GPU before the next forward pass.
-  `spingalett_set_gpu_precision(SPINGALETT_PRECISION_BFLOAT16)` multiplies matrices in bfloat16 on
  the GPU's matrix units (tensor cores: `mma.sync` with CUDA, `VK_KHR_cooperative_matrix` with
  `VK_KHR_shader_bfloat16` with Vulkan), the products added in single precision. The layers' outputs, all but the output layer's (and the
  network's inputs, which the host rounds as it sends them), and their gradients are kept on the GPU
  as bfloat16, the values the products round them to anyway: half the memory and half the bytes
  every pass reads and writes. The passes between products (normalizations, pooling, additions,
  upsampling) compute in single precision, and the parameters, their gradients and the optimizer
  stay in it, as with PyTorch's autocast; products smaller than a block of the matrix units stay in
  single precision unless they read or write bfloat16. Runs stay deterministic. On an RTX 4050
  Laptop GPU ResNet-20 trains 1.7 (Vulkan) to 1.9 times (CUDA) as fast as in single precision and
  reaches the same CIFAR-10 test accuracy (91.80% after 100 epochs, single precision 91.55%). Vulkan
  devices without the extensions keep single precision.
- Results are deterministic: every sum runs in a fixed order, never through atomics, so a run gives
  the same bits every time on one device and backend. They agree with the CPU's up to rounding: the
  products add in another order, with fused multiply-adds, and batch normalization adds its sums in
  single rather than double precision (the two backends agree with each other the same way).
-  The first products of each shape are timed with a few tile sizes, and the fastest is kept for the
  life of the process: the first `spingalett_train()` of a process takes a few tenths of a second
  longer (ResNet-20 on an RTX 4050 Laptop GPU: 0.3 s), and the very first on a machine some ten
  seconds, while the driver compiles the kernels, which it then keeps on disk (NVIDIA's cache is
  limited in size, which `__GL_SHADER_DISK_CACHE_SIZE` raises). Tiles change the speed, never the
  results; `SPINGALETT_GPU_TUNE=0` estimates them instead.
- Training processes samples in chunks of up to 4,096 that fit in half the GPU's memory; networks
  with batch normalization, which full-batch training normalizes over each chunk, in chunks of up to
  2,048, as on the CPU.
  Inference runs in chunks of at most 32 MB of activations as they are kept (64 samples at least;
  from host arrays 2,048 at most), whose layers' outputs stay in the GPU's cache from one layer to
  the next.
- Where Vulkan lets the host write into all of the GPU's memory (resizable BAR, unified memory), it
  writes each chunk's inputs and the parameters straight into it, instead of the GPU copying them
  from host memory; with CUDA a chunk's inputs are copied on a stream of their own while the GPU
  works on the chunks before. Three chunks are in flight: the host fills one while the GPU works on
  the two before it.
- `SPINGALETT_CUDA_DEVICE=n` picks the n-th device of the CUDA driver's list, `SPINGALETT_GPU_DEVICE=n`
  the n-th of the Vulkan device list instead of the first discrete GPU; `SPINGALETT_GPU_PROFILE=1`
  prints the GPU time per kernel and copy when the process exits (with CUDA timed by events around
  every launch, which add some microseconds to each); `SPINGALETT_GPU_NO_HOST_WRITES=1` has Vulkan
  copy inputs and parameters from host memory, and `SPINGALETT_GPU_NO_BF16_STORAGE=1` keeps
  activations in single precision in its bfloat16 mode.

In Python: `sg.set_compute_mode(sg.ComputeMode.CUDA)` (or `ComputeMode.VULKAN`), `sg.cuda_device()`,
`sg.gpu_device()`, `sg.set_gpu_precision(sg.Precision.BFLOAT16)` and `sg.DeviceData(array)`, which
`spingalett_train()`, `validation_data`, `spingalett_forward()` and `spingalett_evaluate()` take in
place of arrays.
