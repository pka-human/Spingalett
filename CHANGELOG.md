# Changelog

All notable changes to this project are documented in this file. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/); before 1.0, a minor release may contain breaking
changes, which are listed under **Changed**.

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

[0.5.0]: https://github.com/pka-human/Spingalett/compare/v0.4.1...v0.5.0
[0.4.1]: https://github.com/pka-human/Spingalett/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/pka-human/Spingalett/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/pka-human/Spingalett/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/pka-human/Spingalett/compare/0a1dd16...v0.2.0
