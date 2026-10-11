# Spingalett for JavaScript

Spingalett's inference engine compiled to WebAssembly, for Node and browsers: `.slett` models of every
kind the engine runs (dense and convolutional networks, graphs, transformers and language models) in
FP32, FP16, BF16, INT8, INT4 or INT2, a sample at a time, with the results of the native engine
(integer models bit for bit). The package is an ES module without dependencies, with TypeScript
declarations; `spingalett.wasm` (62 KB) imports nothing. It runs on Node 18 and later and in browsers
with WebAssembly SIMD (Chrome 91, Firefox 89, Safari 16.4 and later).

Models come from the library, its bindings or `ModelTool` (`spingalett_save()`, `Network.save()`,
`ModelTool convert model.slett model-int8.slett --precision int8`, `ModelTool import model.onnx`).

## Installing

```bash
npm install ./spingalett-<version>.tgz          # the package attached to each release
sh Bindings/JavaScript/build.sh                 # or from a checkout: builds spingalett.wasm next to index.js
```

## Node

```js
import { readFile } from 'node:fs/promises';
import { loadModel } from 'spingalett';

const model = await loadModel(await readFile('mnist.slett'));
console.log(model.inputSize, model.outputSize, model.layers.map((l) => `${l.type} ${l.precision}`));
const probabilities = model.run(pixels);        // 784 values (an array or a Float32Array) -> Float32Array(10)
const all = model.runBatch(images);             // count x 784 values -> count x 10
model.free();
```

## Browsers

```html
<script type="module">
  import { loadEngine } from './node_modules/spingalett/index.js';

  const engine = await loadEngine();             // spingalett.wasm next to index.js, through fetch()
  const model = engine.loadModel(await (await fetch('model.slett')).arrayBuffer());
  const y = model.run(new Float32Array(model.inputSize));
</script>
```

`loadEngine()` takes the module from anywhere else too: a URL or a URL string
(`loadEngine('/assets/spingalett.wasm')`), a path in Node, its bytes, a `fetch()` response, or a
`WebAssembly.Module` compiled already. Served as `application/wasm`, it compiles while it downloads.
Bundlers that understand `new URL('./spingalett.wasm', import.meta.url)` (Vite, webpack 5, Parcel) copy
it with the bundle; the package also exports it as `spingalett/spingalett.wasm`.

## Language models

A language model (an embedding first, the logits of a vocabulary for each position: the networks of
`Examples/CharGPT.c`, `gpt2_from_state_dict()` and `llama_from_state_dict()`) gives the logits of the
next token and continues a prompt as `spingalett_generate()` does, with the same choice of tokens:

```js
const lm = await loadModel(await readFile('chargpt.slett'));      // Examples/CharGPT.c: bytes as tokens
const prompt = [...new TextEncoder().encode('ROMEO:')];
const tokens = lm.generate(prompt, 200, { temperature: 0.8, topK: 40, seed: 1 });
console.log(new TextDecoder().decode(new Uint8Array(tokens)));

const logits = lm.logits(prompt);               // Float32Array(lm.vocabulary): the token after the prompt
```

The window holds the last `contextLength` tokens from its first position on and no token (-1) after
them; each token runs the whole window (the engine keeps no cache of keys and values).
`generate(prompt, count, options)` takes `temperature` (0: the most likely token), `topK`, `topP`,
`seed` (a number or a BigInt; 0 draws one), `stop` (tokens that end it) and `onToken(token, index)`,
called with each token as it comes (returning `false` ends it).

## API

| | |
|---|---|
| `loadEngine(source?)` | the module, compiled and instantiated: an `Engine` with a memory of its own |
| `loadModel(bytes, { engine, wasm }?)` | a model in `engine`, or in a default engine loaded on first use |
| `engine.loadModel(bytes)` | checks a `.slett` image (formats 3 to 8, CRC-32 checksums) and opens a copy of it |
| `engine.formatVersion`, `engine.memorySize` | the newest format the engine reads; the bytes of its memory |
| `model.inputSize`, `outputSize`, `layerCount` | values a sample has in and out; weight layers |
| `model.loss`, `formatVersion`, `imageSize`, `workspaceSize` | the loss it was trained with, its file, its workspace |
| `model.layers` | each weight layer as `spingalett_model_layer()` describes it: `type`, `inputs`, `outputs`, `activation`, `precision`, `inputShape` and `outputShape` (height, width, channels), `kernel`, `stride`, `padding`, `groups`, `epsilon`, `inputLayers`, `upsample`, `vocabulary`, `positions`, `heads`, `kvHeads`, `ropeTheta`, `causal` |
| `model.run(input, output?)` | one sample: `inputSize` values in, a `Float32Array` of `outputSize` out |
| `model.runBatch(inputs, output?)` | samples one after the other (or an array of samples), run one by one |
| `model.isLanguageModel`, `contextLength`, `vocabulary` | whether the model is a language model; its window; its vocabulary |
| `model.logits(tokens)`, `model.generate(prompt, count, options?)` | the next token's logits; a continuation |
| `model.free()` | releases the model's memory in the engine |

Enumerations are strings: layer types (`'dense'`, `'conv2d'`, `'attention'`, ...), activations
(`'relu'`, `'gelu_tanh'`, ...), precisions (`'fp32'`, `'int8'`, ...), losses (`'cross_entropy'`, ...). Images the engine
refuses throw a `SpingalettError` whose `code` is the engine's (`SPINGALETT_ERR_*`: 2 a damaged image,
3 a truncated one, 4 a format version it does not read); wrong sizes throw a `RangeError`.

Models live in the engine's memory, which grows as they are loaded and is reused as they are freed
(WebAssembly memory never shrinks); `free()` releases a model at once, and a model collected without it
is released then. An engine runs one sample at a time: for several at once, load an engine in each
worker (`Worker`, `worker_threads`), all of them from one compiled `WebAssembly.Module`.

## Results and speed

The module is the engine of `Src/Spingalett.Inference.c` as the desktop builds it: C99, its portable
kernels (which Clang vectorizes with SIMD128 where that keeps their results), IEEE arithmetic in a fixed
order without contraction. The mathematical functions it needs come from `wasm/Spingalett.Libm.c`:
`expf` is glibc's algorithm, `tanhf` and `erff` are correctly rounded, as glibc's are from 2.41 on;
over all 2^32 floats they give glibc 2.43's results (but for two `expf` results that glibc's variant for
processors with FMA rounds otherwise). So integer models give the native engine's outputs bit for bit,
and float models the same up to the order in which its SIMD kernels add (a few units in the last place).

On an Intel i7-12650H in Node 26, a sample takes (the native engine compiled with `-march=native` in
brackets):

| Model | FP32 | INT8 |
|---|---:|---:|
| 784-512-512-10 dense network | 0.10 ms (0.05) | 0.09 ms (0.01) |
| `Examples/MNIST_CNN.c`'s network | 0.78 ms (0.29) | 0.84 ms (0.14) |
| CharGPT's GPT (3.3 million parameters), a token from a window of 128 | 70 ms (34) | 75 ms (20) |

## Tests

```bash
cd Bindings/JavaScript && sh build.sh && npm test
```

`test/engine.test.mjs` tests the API (loading from files, URLs, bytes and over HTTP, layers, errors,
memory, generation); `test/native.test.mjs` compares the module with the native engine: with
`test/references` (the outputs of the models of earlier releases in `Tests/Data/runtime` and of the
language models of `test/models`, written by `test/make_references.mjs`, and tokens it generated),
integer models bit for bit and float ones within 1e-5 of each sample's largest output; and, where a C
compiler is at hand (`CC`), with the same code built natively, which must give every model's outputs
and every generation bit for bit. `test/make_models.py` writes the language models (a GPT and a LLaMA
of 28 characters, trained to continue a phrase).

`build.sh` needs Clang with the `wasm32` target and `wasm-ld` (LLD), and no C library: `wasm/include`
holds the part of `<math.h>` and `<string.h>` the engine uses. It takes `CC` and `WASM_LD` (Debian's
`wasm-ld-18`, for instance); CI builds the module with Ubuntu 24.04's Clang and LLD 18.

## License

MIT, as Spingalett.
