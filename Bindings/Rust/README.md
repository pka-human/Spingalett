# Spingalett for Rust

Safe bindings of [Spingalett](https://github.com/pka-human/Spingalett), a neural-network library in C:
dense, convolutional and transformer layers as chains or graphs, trained on the CPU or the GPU (CUDA,
Vulkan), deployment models from FP32 down to INT2, and text generation. Two crates:

- `spingalett-sys`: the C API's declarations (structures laid out as in C; the tests compare every
  offset with the C compiler's);
- `spingalett`: `Network`, `Layer`, `Model`, training on arrays or token files, prediction, generation.

## Building

The crates link `libspingalett` (version 1.2 or later). Install the library (a release package, a
distribution package, or `cmake --install` of a build), or point `SPINGALETT_LIB_DIR` at the directory
that holds it; at run time the system's loader must find it (an installed copy, or `LD_LIBRARY_PATH`).

```sh
SPINGALETT_LIB_DIR=/path/to/Spingalett/Bin cargo build
LD_LIBRARY_PATH=/path/to/Spingalett/Bin cargo test
```

`SPINGALETT_LAYOUT=/path/to/Spingalett/Bin/SpingalettLayout` makes the tests check the structures'
layout against the C compiler's.

## Example

```rust
use spingalett::{Activation, Layer, Loss, Network, Sampling, TrainOptions};

fn main() -> Result<(), spingalett::Error> {
    // XOR
    let mut net = Network::new(Loss::Mse)?;
    net.add(Layer::input(1, 1, 2))?;
    net.add(Layer::dense(8).activation(Activation::Tanh))?;
    net.add(Layer::dense(1).activation(Activation::Sigmoid))?;
    let x = [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0];
    let y = [0.0, 1.0, 1.0, 0.0];
    net.train(&x, &y, &TrainOptions { epochs: 1500, batch_size: 4, learning_rate: 0.05, ..Default::default() })?;
    println!("{:?}", net.predict(&x)?);

    // a small LLaMA-like language model: train on a token file (nanoGPT's .bin), then generate
    let mut lm = Network::new(Loss::SparseCrossEntropy)?;
    lm.add(Layer::input(1, 1, 128))?;                       // a window of 128 tokens
    let h = lm.add(Layer::embedding(256, 128))?;
    lm.add(Layer::rms_norm())?;
    lm.add(Layer::linear(3 * 128))?;
    lm.add(Layer::attention(4).causal(true).rope_theta(10000.0))?;
    let a = lm.add(Layer::linear(128))?;
    lm.add(Layer::add_layers(&[h, a]))?;
    lm.add(Layer::linear(256))?;
    spingalett::set_compute_mode(spingalett::ComputeMode::Cuda); // when there is an NVIDIA GPU
    lm.train_tokens("train.bin", 64, &TrainOptions { epochs: 5, ..Default::default() })?;
    let tokens = lm.generate(&[72, 101], 100, &Sampling { temperature: 0.8, top_k: 40, ..Default::default() })?;
    println!("{tokens:?}");
    Ok(())
}
```

A `Network` moves between threads but is used by one at a time; a `Model` may be shared by several.
