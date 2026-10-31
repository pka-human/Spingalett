# Bindings

Using the library from Python, C++, Rust, .NET, Go, Java and PHP. JavaScript runs the inference engine itself: see [WebAssembly](Deployment.md#webassembly-node-and-browsers).

## Python bindings

`Bindings/Python` contains pure-Python bindings built on `ctypes` and NumPy; nothing is compiled
at install time. The wheels carry the library (Linux x86-64 and AArch64, Windows, macOS):

```bash
pip install spingalett                                  # or, from a checkout:
pip install ./Bindings/Python && export SPINGALETT_LIBRARY=$PWD/Bin/libspingalett.so
```

```python
import numpy as np
import spingalett as sg

x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y = np.array([[0], [1], [1], [0]], dtype=np.float32)

with sg.Network(sg.Loss.MSE, [sg.Layer(2),
                              sg.Layer(16, sg.Activation.TANH, sg.Init.XAVIER),
                              sg.Layer(1, sg.Activation.SIGMOID, sg.Init.XAVIER)]) as net:
    net.train(x, y, epochs=3000, optimizer=sg.Optimizer.ADAM, learning_rate=0.02,
              lr_scheduler=sg.CosineDecay())
    print(net.forward(x))
    with net.to_model(sg.Precision.INT8) as model:       # deployment model, integer kernels
        print(model.predict(x), model.size, "bytes")

cnn = sg.Network(sg.Loss.CROSS_ENTROPY, [sg.Input(28, 28, 1), sg.Conv2D(32, 3, padding=1), sg.MaxPool2D(2),
                                         sg.Layer(10, sg.Activation.SOFTMAX, sg.Init.XAVIER)])
print(cnn.layers[1].shape, cnn.get_weights(0).shape)    # (28, 28, 32) (32, 3, 3, 1)

resnet = sg.Network.from_torch(torch_module, torch.rand(1, 3, 32, 32))   # through ONNX, channels last

gpt = sg.gpt2_from_state_dict(GPT2LMHeadModel.from_pretrained("gpt2").state_dict(), heads=12, context=256)
gpt.train_from_tokens("train.bin", epochs=1, batch_size=8, optimizer=sg.Optimizer.ADAMW, learning_rate=3e-4)
print(gpt.generate([15496, 11, 314], 50, temperature=0.8, top_k=40))

graph = sg.Graph.load("model.onnx")                    # any ONNX model, as a graph of operators
print(graph.inputs, graph.run({"image": np.zeros((1, 3, 224, 224), np.float32)}))
vision = sg.Graph.from_torch(torch_module, torch.rand(1, 3, 224, 224))     # through torch.onnx

lm = sg.Network.from_pretrained("SmolLM2-135M.Q4_K_M.gguf", context=512)    # or a Hugging Face directory
tok = sg.Tokenizer("SmolLM2-135M.Q4_K_M.gguf")
print(tok.decode(lm.generate(tok.encode("The three primary colors are"), 24)))
```

See [Bindings/Python/README.md](../Bindings/Python/README.md) for the full API.

## C++

`Spingalett/Spingalett.hpp` is a header-only C++23 interface over the same library: the library's
objects as move-only owners (`spingalett::Network`, `Model`, `Graph`, `Tokenizer`, `Dataset`, `DeviceData`), data as
`std::span`, errors as `std::expected<T, spingalett::Error>` (the library's code and message; nothing
throws), scoped enums, and a fluent `Builder` that makes the network in `build()`:

```cpp
#include <Spingalett/Spingalett.hpp>
namespace sg = spingalett;

auto net = sg::Builder(sg::Loss::CrossEntropy)
               .input(28, 28, 1)
               .conv2d({.filters = 32, .kernel = 3, .padding = 1, .act = sg::Activation::Relu,
                        .init = sg::Init::He})
               .max_pool2d(2)
               .dense(128, sg::Activation::Relu, sg::Init::He)
               .dense(10, sg::Activation::Softmax, sg::Init::Xavier)
               .build();
if (!net) {
    std::cerr << net.error().message << '\n';     /* the library's message: what failed */
    return 1;
}

sg::TrainOptions o;
o.epochs = 2;
o.strategy = sg::Strategy::MiniBatch;
o.batch_size = 128;
o.optimizer = sg::Optimizer::Adam;
o.learning_rate = 1e-3f;
sg::Result<sg::TrainReport> report = net->train(images, labels, o);   /* std::span<const float> */
sg::Result<std::vector<float>> outputs = net->predict(images);
sg::Result<sg::Model> int8 = net->to_model(sg::Precision::Int8);
```

`Builder::from({a, b})` names the inputs of the next layer and `last()` gives the index of the last
one, for residual blocks and branches; `add_layers()` and `concat_layers()` take their inputs
directly. `linear()`, `embedding()`, `attention()`, `rms_norm()` and `multiply_layers()` build
transformers; `Network::train_tokens()` trains on a token file and `generate()` continues a prompt
(`sg::Sampling`). `Network::from_pretrained(path, context)` loads a Hugging Face model directory or a
GGUF file, and `sg::Tokenizer::load(path)` its tokenizer (`encode()`, `decode()`, `token()`, `id()`,
`info()`). `TrainOptions::raw` carries every other field of `SpingalettTrainArgs`, and `raw()`
gives the C object of a wrapper for whatever the wrapper does not cover.

## Bindings for other languages

`Bindings/` holds bindings for Rust, .NET, Go, Java and PHP over the shared library: networks of every
kind of layer built in each language's style, training on arrays or token files, prediction,
evaluation, generation, parameters, files and deployment models, pretrained language models (Hugging
Face directories, GGUF files) and their tokenizers. Each one's tests train XOR and a small LLaMA-like
model whose generation they check, check the tokenizers and pretrained models against the C tests'
expectations, and compare every structure the binding lays out with the C compiler's layout (CI runs
them against the installed library):

| Language | Directory | Interface | Builds with |
|---|---|---|---|
| Rust | [`Bindings/Rust`](../Bindings/Rust/README.md) | crates `spingalett-sys` (declarations) and `spingalett` (safe) | cargo; `SPINGALETT_LIB_DIR` |
| C# (.NET 8+) | [`Bindings/CSharp`](../Bindings/CSharp/README.md) | package `Spingalett` (`LibraryImport`) | dotnet |
| Go | [`Bindings/Go`](../Bindings/Go/README.md) | module `github.com/pka-human/Spingalett/Bindings/Go` (cgo) | go; pkg-config |
| Java (22+) | [`Bindings/Java`](../Bindings/Java/README.md) | `io.github.pkahuman.spingalett` (the foreign function and memory API) | Maven |
| PHP (8.0+) | [`Bindings/PHP`](../Bindings/PHP/README.md) | package `pka-human/spingalett` (the FFI extension; `spingalett.h`, the C declarations) | Composer; `SPINGALETT_LIB_DIR` |

```rust
let mut lm = Network::new(Loss::SparseCrossEntropy)?;
lm.add(Layer::input(1, 1, 128))?;
let h = lm.add(Layer::embedding(256, 128))?;
lm.add(Layer::rms_norm())?;
lm.add(Layer::linear(3 * 128))?;
lm.add(Layer::attention(4).causal(true).rope_theta(10000.0))?;
let a = lm.add(Layer::linear(128))?;
lm.add(Layer::add_layers(&[h, a]))?;
lm.add(Layer::linear(256))?;
lm.train_tokens("train.bin", 64, &TrainOptions { epochs: 5, ..Default::default() })?;
let text = lm.generate(&[72, 101], 100, &Sampling { temperature: 0.8, top_k: 40, ..Default::default() })?;
```

JavaScript runs the inference engine itself, compiled to WebAssembly: see
[WebAssembly: Node and browsers](Deployment.md#webassembly-node-and-browsers).
