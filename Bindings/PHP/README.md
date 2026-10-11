# Spingalett for PHP

Bindings of [Spingalett](https://github.com/pka-human/Spingalett), a neural-network library in C, through
PHP's FFI extension (PHP 7.4 and later; nothing to compile): dense, convolutional and transformer layers as
chains or graphs, trained on the CPU or the GPU (CUDA, Vulkan), deployment models from FP32 down to INT2,
and text generation. `spingalett.h` holds the library's C declarations as FFI parses them (no macros: the
`*_args` functions take the structures).

```sh
composer require pka-human/spingalett
```

The classes call the shared library (`libspingalett.so.1`, `libspingalett.1.dylib`, `spingalett.dll`;
version 1.2 or later) that the file `SPINGALETT_LIBRARY` names, else the one in the directory
`SPINGALETT_LIB_DIR` names, else the one the system's loader finds (an installed copy, `LD_LIBRARY_PATH`,
`DYLD_LIBRARY_PATH`, `PATH`); `Library::load($path)` names it in code. The FFI extension must be loaded
(`extension=ffi`). Its setting `ffi.enable=preload`, the default, allows FFI on the command line and in
preloaded scripts only: a server either preloads the declarations (`opcache.preload` of a script that calls
`Spingalett\Library::preload()`; requests then use them without parsing them again) or sets
`ffi.enable=true`. Without Composer, `require 'Bindings/PHP/autoload.php';`.

```php
use Spingalett\{Activation, ComputeMode, Layer, Library, Loss, Network, Sampling, TrainOptions};

// XOR
$net = new Network(Loss::MSE);
$net->add(Layer::input(1, 1, 2));
$net->add(Layer::dense(8)->activation(Activation::TANH));
$net->add(Layer::dense(1)->activation(Activation::SIGMOID));
$x = [[0, 0], [0, 1], [1, 0], [1, 1]];                  // or flat: [0, 0, 0, 1, 1, 0, 1, 1]
$net->train($x, [0, 1, 1, 0], new TrainOptions(epochs: 1500, batchSize: 4, learningRate: 0.05));
print_r($net->predict($x));

// a small LLaMA-like language model: trained on a token file (nanoGPT's .bin), then generating
$lm = new Network(Loss::SPARSE_CROSS_ENTROPY);
$lm->add(Layer::input(1, 1, 128));                      // a window of 128 tokens
$h = $lm->add(Layer::embedding(256, 128));
$lm->add(Layer::rmsNorm());
$lm->add(Layer::linear(3 * 128));
$lm->add(Layer::attention(4)->causal()->ropeTheta(10000));
$a = $lm->add(Layer::linear(128));
$lm->add(Layer::addLayers($h, $a));
$lm->add(Layer::linear(256));
Library::setComputeMode(ComputeMode::CUDA);             // when there is an NVIDIA GPU
$lm->trainTokens('train.bin', 64, new TrainOptions(epochs: 5));
$tokens = $lm->generate([72, 101], 100, new Sampling(temperature: 0.8, topK: 40));
```

Options go by name from PHP 8; PHP 7.4 gives them in order (`new TrainOptions(1500, 4, 0.05)`) or sets the
properties. Samples are arrays of numbers, flat or one array a sample, copied into the library's floats once
a call; outputs come back flat. `Network::save()`, `Network::load()`, `Model::load()` and
`Network::toModel(Precision::INT8)` handle files and deployment models, and `Model::session()` a causal
language model's session, fed a token at a time with the keys and values of earlier tokens kept
(`feed()`, `rewind()`, `copy()`, `generate()` of several sequences at once). `Network::fromPretrained()`
loads a Hugging Face directory of LLaMA, Mistral, Qwen2 or GPT-2, or a GGUF file, and `Tokenizer` encodes and
decodes text with a `tokenizer.json` file, a model directory's or a GGUF file's vocabulary; `weights()`, `biases()` and
`setParameters()` the parameters. `Library::ffi()` gives the C functions themselves, and `raw()` the C
object of a network or model, for what the classes leave out. Errors of the library throw
`SpingalettException` (its code is the library's error code), wrong arguments `InvalidArgumentException`.

Tests: `php -d extension=ffi tests/run.php` with the library where the classes find it;
`SPINGALETT_LAYOUT=/path/to/Bin/SpingalettLayout` also compares every structure's layout with the C
compiler's (PHP 8.1 and later).
