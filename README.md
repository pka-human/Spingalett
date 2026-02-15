<div align="center">

<img src="Logo.png" alt="Spingalett Logo"/>

#

[![Standard](https://img.shields.io/badge/C-23-blue.svg?style=flat-square&logo=c)](https://en.wikipedia.org/wiki/C23_(C_standard_revision))
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=flat-square)](LICENSE)
[![Dependencies](https://img.shields.io/badge/Dependencies-Zero_Required-red.svg?style=flat-square)]()
[![CUDA](https://img.shields.io/badge/CUDA-Supported-76b900.svg?style=flat-square&logo=nvidia)]()

<p align="center">
  <strong>Modern C23 Deep Learning Engine: Zero required dependencies, flat memory architecture, and declarative API designed for maximum performance on any hardware.</strong>
</p>

</div>

## ⚡ Why Spingalett?

Machine learning shouldn't require gigabytes of Python dependencies or a virtual environment just to run a simple model. **Spingalett** goes back to the roots of high-performance computing.
It is a lightweight, user-friendly library written in **Pure C23** that outperforms major frameworks like TensorFlow and PyTorch **up to (TODO: Take an accurate measurement)%** due to reduced overhead and optimized memory management.

### Key Features

*   🚀 **Zero Hard Dependencies:** The core engine runs on pure standard C (libc only). No Python, no NumPy, no bloat.
*   ⚡ **Hybrid Backend:** Seamlessly switch between optimized CPU and NVIDIA GPU (**cuBLAS / cuDNN**) for extreme performance.
*   🧠 **Flat Memory Architecture:** Uses a custom "Flattening" technique for tensors, maximizing CPU cache coherency and SIMD usage.
*   💎 **Modern C23 API:** Uses designated initializers (`.field = value`) for a clean, declarative syntax without the performance cost.

## 🛠 Quick Start

Defining and training a network in C has never looked this clean.

```c
#include "Spingalett/Spingalett.h"

int main() {
    // 1. Create Network
    NeuralNetwork *nn = new_spingalett(LOSS_MSE);

    // 2. Define Architecture (Declarative Style)
    layer(nn, 2); // Input layer
    layer(nn, 4, RELU, WEIGHT_INITIALIZATION_HE); // Hidden layer
    layer(nn, 1, CT_SIGMOID, WEIGHT_INITIALIZATION_XAVIER); // Output layer

    // 3. Train
    train(
        .net = nn,
        .training_strategy = STRATEGY_FULL_BATCH,
        .optimizer_type = OPTIMIZER_ADAMW,
        .inputs = (float*)patterns,
        .targets = (float*)targets,
        .epochs = 30000,
        .learning_rate = 0.01f,
        .weight_decay = 0.0001f
    );

    free_network(nn);
    return 0;
}
```

## 📦 Building

TODO: WRITE HOW TO BUILD THIS SHIT

## 💬 Philosophy: Why C in 2026?

**Stability. Portability. Control.**
While Rust is great and Python is convenient, C remains the *lingua franca* of computing. 
*   **Embed everywhere:** Run your model on a $5 microcontroller or a massive supercomputer without significantly changing the codebase.
*   **Stable ABI:** Easily wrap Spingalett in Python (`ctypes`), Go, Lua, Rust or C++ if needed.
*   **Instant Compilation:** No waiting for `cargo` to download the internet.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

<div align="center">
  <sub>Built with ❤️ and pointers by <a href="https://github.com/pka-human">pka_human</a> and contributors</sub>
</div>

