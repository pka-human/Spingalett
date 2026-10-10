// SPDX-License-Identifier: MIT
// Copyright (c) 2026 pka_human (pka_human@proton.me)

package spingalett

import (
	"math"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

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
