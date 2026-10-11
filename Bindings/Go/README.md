# Spingalett for Go

Bindings of [Spingalett](https://github.com/pka-human/Spingalett), a neural-network library in C, through cgo:
dense, convolutional and transformer layers as chains or graphs, trained on the CPU or the GPU (CUDA,
Vulkan), deployment models from FP32 down to INT2, and text generation. cgo reads the library's own
header, so the structures are C's.

```sh
go get github.com/pka-human/Spingalett/Bindings/Go
```

The package finds the library through pkg-config (`spingalett.pc`, installed with it; a release package
or `cmake --install`): set `PKG_CONFIG_PATH` to its `lib/pkgconfig` when it is not in a default place,
and let the loader find the shared library at run time (`LD_LIBRARY_PATH`).

```go
package main

import (
	"fmt"

	sg "github.com/pka-human/Spingalett/Bindings/Go"
)

func main() {
	net, _ := sg.NewNetwork(sg.LossMSE)
	defer net.Close()
	net.Add(sg.Input(1, 1, 2))
	net.Add(sg.Dense(8).Activation(sg.ActTanh))
	net.Add(sg.Dense(1).Activation(sg.ActSigmoid))
	x := []float32{0, 0, 0, 1, 1, 0, 1, 1}
	y := []float32{0, 1, 1, 0}
	net.Train(x, y, sg.TrainOptions{Epochs: 1500, BatchSize: 4, LearningRate: 0.05})
	out, _ := net.Predict(x)
	fmt.Println(out)

	// a small LLaMA-like language model trained on a token file (nanoGPT's .bin), then generating
	lm, _ := sg.NewNetwork(sg.LossSparseCrossEntropy)
	defer lm.Close()
	lm.Add(sg.Input(1, 1, 128))
	h, _ := lm.Add(sg.Embedding(256, 128))
	lm.Add(sg.RMSNorm())
	lm.Add(sg.Linear(3 * 128))
	lm.Add(sg.Attention(4).Causal(true).RopeTheta(10000))
	a, _ := lm.Add(sg.Linear(128))
	lm.Add(sg.AddLayers(h, a))
	lm.Add(sg.Linear(256))
	sg.SetComputeMode(sg.ComputeCUDA) // when there is an NVIDIA GPU
	lm.TrainTokens("train.bin", 64, sg.TrainOptions{Epochs: 5})
	tokens, _ := lm.Generate([]uint32{72, 101}, 100, sg.Sampling{Temperature: 0.8, TopK: 40})
	fmt.Println(tokens)
}
```

A `Network` is used by one goroutine at a time; a `Model` may be shared.

## Pretrained language models and tokenizers

`LoadPretrained` makes a network of a Hugging Face model directory (`config.json` and safetensors weights,
sharded or not; LLaMA, Mistral, Qwen2, GPT-2) or of a GGUF file (quantized weights dequantized to floats),
and `LoadTokenizer` reads its tokenizer (a `tokenizer.json`, the directory, or the GGUF file), which
encodes and decodes as Hugging Face's `tokenizers` library does:

```go
lm, err := sg.LoadPretrained("models/SmolLM2-135M", 512) // or "SmolLM2-135M.Q4_K_M.gguf"
if err != nil {
	log.Fatal(err)
}
defer lm.Close()
tok, _ := sg.LoadTokenizer("models/SmolLM2-135M")
defer tok.Close()
prompt, _ := tok.Encode("<|im_start|>user\nHello!<|im_end|>\n<|im_start|>assistant\n", true, false)
end, _ := tok.ID("<|im_end|>")
reply, _ := lm.Generate(prompt, 64, sg.Sampling{Temperature: 0.7, TopP: 0.9, Stop: []uint32{end}})
text, _ := tok.Decode(reply, true)
fmt.Println(text)
```

`Encode(text, special, addSpecial)`: with `special`, special tokens written in the text become their ids
(otherwise they are text, so that text from users cannot forge them); `addSpecial` adds the tokens the
tokenizer puts around a text (LLaMA's BOS). `Token`, `ID`, `Vocabulary`, `BOS` and `EOS` describe the
vocabulary. A `Tokenizer` may be shared by goroutines.
