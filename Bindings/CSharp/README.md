# Spingalett for .NET

Bindings of [Spingalett](https://github.com/pka-human/Spingalett), a neural-network library in C, for .NET 8
and later (`LibraryImport`, no reflection at run time): dense, convolutional and transformer layers as
chains or graphs, trained on the CPU or the GPU (CUDA, Vulkan), deployment models from FP32 down to INT2,
and text generation.

The package calls the shared library `spingalett` (`libspingalett.so`, `spingalett.dll`,
`libspingalett.dylib`, version 1.2 or later), which the system's loader must find: an installed copy, a
directory on `LD_LIBRARY_PATH` (`PATH` on Windows), or next to the program.

```csharp
using Spingalett;

// XOR
using var net = new Network(Loss.Mse);
net.Add(Layer.Input(1, 1, 2));
net.Add(Layer.Dense(8).WithActivation(Activation.Tanh));
net.Add(Layer.Dense(1).WithActivation(Activation.Sigmoid));
float[] x = { 0, 0, 0, 1, 1, 0, 1, 1 }, y = { 0, 1, 1, 0 };
net.Train(x, y, new TrainOptions { Epochs = 1500, BatchSize = 4, LearningRate = 0.05f });
Console.WriteLine(string.Join(" ", net.Predict(x)));

// a small LLaMA-like language model: trained on a token file (nanoGPT's .bin), then generating
using var lm = new Network(Loss.SparseCrossEntropy);
lm.Add(Layer.Input(1, 1, 128));                         // a window of 128 tokens
uint h = lm.Add(Layer.Embedding(256, 128));
lm.Add(Layer.RmsNorm());
lm.Add(Layer.Linear(3 * 128));
lm.Add(Layer.Attention(4).Causal().RopeTheta(10000));
uint a = lm.Add(Layer.Linear(128));
lm.Add(Layer.AddLayers(h, a));
lm.Add(Layer.Linear(256));
Library.SetComputeMode(ComputeMode.Cuda);               // when there is an NVIDIA GPU
lm.TrainTokens("train.bin", stride: 64, new TrainOptions { Epochs = 5 });
uint[] tokens = lm.Generate(new uint[] { 72, 101 }, 100, new Sampling { Temperature = 0.8f, TopK = 40 });
```

Pretrained language models and their tokenizers (library 1.3 or later): `Network.FromPretrained` reads a
Hugging Face model directory (`config.json` and safetensors weights; LLaMA, Mistral, Qwen2, GPT-2) or a GGUF
file (its weights dequantized from F32 to the k-quants), and `Tokenizer` a `tokenizer.json`, a model directory
or a GGUF file, encoding and decoding as Hugging Face's `tokenizers` does:

```csharp
using var model = Network.FromPretrained("SmolLM2-135M.Q4_K_M.gguf", context: 512);   // or a model directory
using var tok = Tokenizer.Load("SmolLM2-135M.Q4_K_M.gguf");
uint[] prompt = tok.Encode("<|im_start|>user\nHi!<|im_end|>\n<|im_start|>assistant\n", special: true);
uint[] reply = model.Generate(prompt, 64, new Sampling { Temperature = 0.7f, Stop = new[] { tok.Id("<|im_end|>")!.Value } });
Console.WriteLine(tok.Decode(reply));
```

`Encode(text, special, addSpecial)`: `special` turns special tokens written in the text into their ids (by
default they are text), `addSpecial` adds the tokens the tokenizer puts around a text (LLaMA's BOS);
`Decode(ids, skipSpecial)`, `Token(id)`, `Id(token)`, `Vocabulary`, `Bos`, `Eos`, `Pad` and `Unk` (null when
the tokenizer has none) complete it.

A `Network` is used by one thread at a time; a `Model` (`Network.ToModel`, `Model.Load`) and a `Tokenizer` may
be shared.

Tests: `dotnet run --project Spingalett.Tests` with the library on `LD_LIBRARY_PATH`;
`SPINGALETT_LAYOUT=/path/to/Bin/SpingalettLayout` also compares every structure's layout with the C
compiler's.
