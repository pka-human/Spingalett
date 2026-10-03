# Changelog

All notable changes to this project are documented in this file. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/); before 1.0, a minor release may contain breaking
changes, which are listed under **Changed**.

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

[0.2.0]: https://github.com/pka-human/Spingalett/compare/0a1dd16...v0.2.0
