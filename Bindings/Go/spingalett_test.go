// SPDX-License-Identifier: MIT
// Copyright (c) 2026 pka_human (pka_human@proton.me)

package spingalett

import (
	"bufio"
	"encoding/binary"
	"encoding/hex"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"strconv"
	"strings"
	"testing"
)

// The repository's test data: tokenizers and models of the C test groups tokenizer and pretrained.
var testData = filepath.Join("..", "..", "Tests", "Data")

func unhex(t *testing.T, s string) string {
	if s == "-" {
		return ""
	}
	b, err := hex.DecodeString(s)
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}

// TestTokenizer checks every case of tokenizer_llama3.txt (Hugging Face's ids and texts, written by
// Tests/make_tokenizer_tests.py) on the tokenizer.json and on the GGUF file of the same vocabulary.
func TestTokenizer(t *testing.T) {
	for _, file := range []string{"tokenizer_llama3.json", "gguf_llama.gguf"} {
		tok, err := LoadTokenizer(filepath.Join(testData, file))
		if err != nil {
			t.Fatal(err)
		}
		f, err := os.Open(filepath.Join(testData, "tokenizer_llama3.txt"))
		if err != nil {
			t.Fatal(err)
		}
		var lines []string
		scanner := bufio.NewScanner(f)
		scanner.Buffer(make([]byte, 1<<20), 1<<20)
		for scanner.Scan() {
			if !strings.HasPrefix(scanner.Text(), "#") {
				lines = append(lines, scanner.Text())
			}
		}
		f.Close()
		cases := 0
		for k := 0; k+3 < len(lines); k += 4 {
			head := strings.Fields(lines[k])
			text := unhex(t, head[2])
			special, add := head[0] == "1", head[1] == "1"
			var want []uint32
			for _, field := range strings.Fields(lines[k+1])[1:] {
				v, _ := strconv.ParseUint(field, 10, 32)
				want = append(want, uint32(v))
			}
			got, err := tok.Encode(text, special, add)
			if err != nil || len(got) != len(want) || (len(want) > 0 && !reflect.DeepEqual(got, want)) {
				t.Fatalf("%s: %q gives %v, not %v (%v)", file, text, got, want, err)
			}
			for skip, line := range lines[k+2 : k+4] {
				if s, err := tok.Decode(want, skip == 1); err != nil || s != unhex(t, line) {
					t.Fatalf("%s: %v decode to %q (%v)", file, want, s, err)
				}
			}
			cases++
		}
		if cases < 100 {
			t.Fatalf("%s: %d cases", file, cases)
		}
		bos, hasBOS := tok.BOS()
		eos, hasEOS := tok.EOS()
		wantEOS := uint32(421)
		if file == "gguf_llama.gguf" {
			wantEOS = 424
		}
		if !hasBOS || bos != 420 || !hasEOS || eos != wantEOS || tok.Vocabulary() != 425 {
			t.Fatalf("%s: BOS %d %v, EOS %d %v, vocabulary %d", file, bos, hasBOS, eos, hasEOS, tok.Vocabulary())
		}
		if id, ok := tok.ID("<|begin_of_text|>"); !ok || id != 420 {
			t.Fatal("a special token by name")
		}
		if s, ok := tok.Token(420); !ok || s != "<|begin_of_text|>" {
			t.Fatal("a special token by id")
		}
		if _, ok := tok.ID("no such token"); ok {
			t.Fatal("a token that does not exist")
		}
		long := strings.Repeat("Tokenizers cut text into pieces, ", 200)
		if ids, _ := tok.Encode(long, false, false); len(ids) == 0 {
			t.Fatal("a long text")
		} else if s, _ := tok.Decode(ids, false); s != long {
			t.Fatal("a long text round trip")
		}
		tok.Close()
	}
	data, err := os.ReadFile(filepath.Join(testData, "tokenizer_llama3.json"))
	if err != nil {
		t.Fatal(err)
	}
	tok, err := TokenizerFromBytes(data)
	if err != nil {
		t.Fatal(err)
	}
	defer tok.Close()
	if ids, _ := tok.Encode("<|begin_of_text|>", true, false); !reflect.DeepEqual(ids, []uint32{420}) {
		t.Fatalf("special tokens when asked: %v", ids)
	}
	if ids, _ := tok.Encode("<|begin_of_text|>", false, false); len(ids) < 2 {
		t.Fatalf("special tokens as text: %v", ids)
	}
	if _, err := LoadTokenizer(filepath.Join(testData, "no_such_tokenizer.json")); err == nil {
		t.Fatal("a missing tokenizer")
	}
}

// window reads NAME.bin: "SPGT", 1, the window's tokens and logits (T, T x vocabulary), then both as floats.
func window(t *testing.T, name string) ([]float32, []float32) {
	data, err := os.ReadFile(filepath.Join(testData, name+".bin"))
	if err != nil || len(data) < 16 || string(data[:4]) != "SPGT" {
		t.Fatalf("%s.bin: %v", name, err)
	}
	T, n := binary.LittleEndian.Uint32(data[8:]), binary.LittleEndian.Uint32(data[12:])
	floats := make([]float32, T+n)
	for i := range floats {
		floats[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[16+4*i:]))
	}
	return floats[:T], floats[T:]
}

// TestPretrained checks a Hugging Face directory and the GGUF file of its weights against PyTorch's logits
// (Tests/make_pretrained_tests.py), and the directory's tokenizer.
func TestPretrained(t *testing.T) {
	SetVerbose(false)
	tokens, logits := window(t, "hf_llama")
	for _, path := range []string{"hf_llama", "gguf_llama.gguf"} {
		lm, err := LoadPretrained(filepath.Join(testData, path), uint32(len(tokens)))
		if err != nil {
			t.Fatal(err)
		}
		out, err := lm.Predict(tokens)
		if err != nil || len(out) != len(logits) {
			t.Fatalf("%s: %v", path, err)
		}
		worst := 0.0
		for i := range out {
			worst = math.Max(worst, math.Abs(float64(out[i]-logits[i])))
		}
		if worst > 1e-4 {
			t.Fatalf("%s: logits %g from PyTorch's", path, worst)
		}
		lm.Close()
	}
	tok, err := LoadTokenizer(filepath.Join(testData, "hf_llama"))
	if err != nil {
		t.Fatal(err)
	}
	defer tok.Close()
	ids, err := tok.Encode("The quick brown fox jumps over the lazy dog, it's 1234 times!", false, false)
	if err != nil || len(ids) < len(tokens) {
		t.Fatal(err)
	}
	for i := range tokens {
		if float32(ids[i]) != tokens[i] {
			t.Fatalf("the directory's tokenizer: %v", ids)
		}
	}
	if eos, ok := tok.EOS(); !ok || eos != 424 {
		t.Fatalf("the EOS of tokenizer_config.json: %d", eos)
	}
	if _, err := LoadPretrained(filepath.Join(testData, "no_such_model"), 0); err == nil {
		t.Fatal("a missing model")
	}
}

func TestVersion(t *testing.T) {
	if !strings.HasPrefix(Version(), "1.") {
		t.Fatalf("version %q", Version())
	}
}

func xor(t *testing.T) (*Network, []float32, []float32) {
	SetVerbose(false)
	Seed(7)
	net, err := NewNetwork(LossMSE)
	if err != nil {
		t.Fatal(err)
	}
	for _, l := range []Layer{Input(1, 1, 2), Dense(8).Activation(ActTanh), Dense(1).Activation(ActSigmoid)} {
		if _, err := net.Add(l); err != nil {
			t.Fatal(err)
		}
	}
	return net, []float32{0, 0, 0, 1, 1, 0, 1, 1}, []float32{0, 1, 1, 0}
}

func TestTrainPredict(t *testing.T) {
	net, x, y := xor(t)
	defer net.Close()
	if net.InputSize() != 2 || net.OutputSize() != 1 || net.LayerCount() != 3 {
		t.Fatal("sizes")
	}
	r, err := net.Train(x, y, TrainOptions{Epochs: 1500, BatchSize: 4, LearningRate: 0.05})
	if err != nil || !r.Completed || r.TrainLoss > 0.05 {
		t.Fatalf("train %+v %v", r, err)
	}
	out, err := net.Predict(x)
	if err != nil {
		t.Fatal(err)
	}
	for i := range y {
		if math.Abs(float64(out[i]-y[i])) > 0.3 {
			t.Fatalf("predict %v", out)
		}
	}
	if m, err := net.Evaluate(x, y); err != nil || m.Loss > 0.05 {
		t.Fatalf("evaluate %+v %v", m, err)
	}
	w, _ := net.Weights(1)
	b, _ := net.Biases(1)
	if len(w) != 16 || len(b) != 8 {
		t.Fatalf("parameters %d %d", len(w), len(b))
	}
	if _, err := net.Predict(x[:3]); err == nil {
		t.Fatal("a partial sample")
	}
}

func TestFilesModels(t *testing.T) {
	net, x, y := xor(t)
	defer net.Close()
	if _, err := net.Train(x, y, TrainOptions{Epochs: 50, BatchSize: 4}); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "net.slett")
	if err := net.Save(path, Float32); err != nil {
		t.Fatal(err)
	}
	back, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	defer back.Close()
	a, _ := back.Predict(x)
	b, _ := net.Predict(x)
	if !reflect.DeepEqual(a, b) {
		t.Fatalf("round trip %v %v", a, b)
	}
	model, err := LoadModel(path)
	if err != nil {
		t.Fatal(err)
	}
	defer model.Close()
	c, err := model.Predict(x)
	if err != nil || model.InputSize() != 2 {
		t.Fatal(err)
	}
	for i := range c {
		if math.Abs(float64(c[i]-b[i])) > 1e-5 {
			t.Fatalf("model %v %v", c, b)
		}
	}
	int8, err := net.ToModel(Int8)
	if err != nil {
		t.Fatal(err)
	}
	defer int8.Close()
	if _, err := Load("/nonexistent/net.slett"); err == nil {
		t.Fatal("a missing file")
	}
	_ = os.Remove(path)
}

func TestTransformerGenerates(t *testing.T) {
	SetVerbose(false)
	Seed(3)
	const T, V = 16, 13
	lm, err := NewNetwork(LossSparseCrossEntropy)
	if err != nil {
		t.Fatal(err)
	}
	defer lm.Close()
	add := func(l Layer) uint32 {
		i, err := lm.Add(l)
		if err != nil {
			t.Fatal(err)
		}
		return i
	}
	add(Input(1, 1, T))
	h := add(Embedding(V, 16))
	add(RMSNorm())
	add(Linear(48))
	add(Attention(4).KVHeads(2).Causal(true).RopeTheta(10000))
	a := add(Linear(16))
	add(AddLayers(h, a))
	add(Linear(V))
	if lm.TargetSize() != T || lm.OutputSize() != T*V {
		t.Fatal("language model sizes")
	}
	var x, y []float32
	for s := uint32(0); s < 256; s++ {
		start := (s*7 + 3) % V
		for i := uint32(0); i < T; i++ {
			x = append(x, float32((start+i)%V))
			y = append(y, float32((start+i+1)%V))
		}
	}
	r, err := lm.Train(x, y, TrainOptions{Epochs: 30, BatchSize: 32, LearningRate: 1e-2})
	if err != nil || r.TrainLoss > 0.3 {
		t.Fatalf("train %+v %v", r, err)
	}
	greedy, err := lm.Generate([]uint32{3, 4, 5}, 20, Sampling{})
	if err != nil {
		t.Fatal(err)
	}
	for i, tok := range greedy {
		if tok != uint32((6+i)%V) {
			t.Fatalf("greedy %v", greedy)
		}
	}
	d := Sampling{Temperature: 1, TopP: 0.9, Seed: 8}
	p, _ := lm.Generate([]uint32{3, 4, 5}, 20, d)
	q, _ := lm.Generate([]uint32{3, 4, 5}, 20, d)
	if !reflect.DeepEqual(p, q) || len(p) != 20 {
		t.Fatal("draws repeat with their seed")
	}
	if s, _ := lm.Generate([]uint32{3, 4, 5}, 20, Sampling{Stop: []uint32{9}}); !reflect.DeepEqual(s, []uint32{6, 7, 8, 9}) {
		t.Fatalf("stop %v", s)
	}
}
