<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="Images/LogoDark.png">
  <source media="(prefers-color-scheme: light)" srcset="Images/LogoLight.png">
  <img alt="Spingalett Logo" src="Images/LogoLight.png">
</picture>

#

[![Standard](https://img.shields.io/badge/C-23-blue.svg?style=flat-square)](https://en.wikipedia.org/wiki/C23_(C_standard_revision))
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=flat-square)](LICENSE)
[![Dependencies](https://img.shields.io/badge/Dependencies-Zero_Required-red.svg?style=flat-square)]()
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/pka-human/Spingalett)
<!--[![CUDA](https://img.shields.io/badge/CUDA-Supported-76b900.svg?style=flat-square&logo=nvidia)]()-->

<p>
Modern C23 Deep Learning Engine: Architectural austerity, linear memory topology, and a declarative interface engineered for raw throughput.
</p>

<!--
<a href="https://www.star-history.com/#pka-human/Spingalett&type=date&legend=bottom-right">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=pka-human/Spingalett&type=date&theme=dark&legend=bottom-right" />
    <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=pka-human/Spingalett&type=date&legend=bottom-right" />
    <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=pka-human/Spingalett&type=date&legend=bottom-right" />
  </picture>
</a>
-->

</div>

## Rationale

The contemporary machine learning landscape has devolved into a bloated dependency hell. **Spingalett** rejects the premise that one requires gigabytes of Python site-packages or a fragile virtual environment to execute a matrix multiplication.

This library is a return to the fundamentals of high-performance computing. It is authored in **Pure C23**, deliberately bypassing the overhead of major frameworks like TensorFlow and PyTorch to deliver superior throughput. It is designed for engineers who prefer manual memory management over garbage-collected hand-holding.

### Core Competencies

*   **Dependency Asceticism:** The engine operates strictly on standard C (libc). OpenMP, OpenBLAS, and CUDA are optional compile-time accelerators — never requirements.
*   **Hybrid Compute Backend:** Seamless interoperability between hand-tuned AVX/FMA SIMD routines, OpenBLAS matrix kernels, and OpenMP parallelism. *Note: NVIDIA GPU acceleration (cuBLAS) is currently a work in progress, but the CPU path is already faster than PyTorch.*
*   **Linearized Memory Model:** Implements a strict flattening protocol for all network tensors (weights, biases, gradients, optimizer state), forcing data locality to maximize CPU cache coherency and SIMD vectorization efficiency.
*   **Modern C23 Syntax:** Leverages designated initializers (`.field = value`) and compound literals to provide a declarative, pseudo-high-level API without sacrificing the raw speed of the underlying binary.
*   **Quantized Serialization:** Save and load models in FP32, FP16, BF16, INT8, INT4, or INT2 precision with a versioned binary format.

## ⚡ Performance vs PyTorch

We don't just claim to be fast; we measure it. In a direct head-to-head CPU benchmark training a deep MLP (784 → 512 → 1000 → 10, ~925K parameters) using Adam and Cross-Entropy, **Spingalett outperforms PyTorch across the board**. 

| Engine | Threads | Throughput (samples/sec) | Relative Speed |
|---|---|---|---|
| PyTorch CPU *(Intel MKL)* | 1 | 22,195 | Baseline |
| **Spingalett** *(OpenBLAS)* | **1** | **25,349** | **+14.2%** |
| PyTorch CPU *(Intel MKL)* | 12 | 63,068 | Baseline |
| **Spingalett** *(OpenBLAS)* | **12** | **88,921** | **+41.0%** |

*Tested on a **12th Gen Intel(R) Core(TM) i7-12650H**.*

## 🛠 Usage Specification

Network definition in C23 achieves a syntactical elegance previously reserved for interpreted languages, yet retains the ruthless efficiency of compiled code.

```c
#include <Spingalett/Spingalett.h>
#include <stdio.h>

int main(void) {
    float patterns[4][2] = { {0.f, 0.f}, {0.f, 1.f}, {1.f, 0.f}, {1.f, 1.f} };
    float targets[4][1]  = { {0.f}, {1.f}, {1.f}, {0.f} };

    // 1. Instantiation
    NeuralNetwork *nn = new_spingalett(LOSS_MSE);

    // 2. Architecture Definition (Positional or Declarative)
    layer(nn, 2); 
    layer(nn, 4, ACT_FOO52, WEIGHT_INITIALIZATION_HE);
    layer(nn, 1, ACT_SIGMOID, WEIGHT_INITIALIZATION_XAVIER); 

    // 3. Execution (Training Loop using C23 designated initializers)
    train(
        .net = nn,
        .training_mode = MODE_ARRAY,
        .training_strategy = STRATEGY_FULL_BATCH,
        .optimizer_type = OPTIMIZER_ADAMW,
        .inputs = (float*)patterns,
        .targets = (float*)targets,
        .sample_count = 4,
        .epochs = 10000,
        .learning_rate = 0.01f,
        .weight_decay = 0.0001f,
        .report_interval = 2000
    );

    // 4. Inference
    printf("\n--- TESTING FOO52 ---\n");
    for (int i = 0; i < 4; i++) {
        float *out = forward(nn, patterns[i]);
        printf("In: %.0f %.0f  Tgt: %.0f  Out: %.4f\n", 
            patterns[i][0], patterns[i][1], targets[i][0], out[0]);
    }

    free_network(nn);
    return 0;
}
```

### Features at a Glance

| Category | Options |
|---|---|
| **Activations** | Sigmoid, ReLU, Tanh, Leaky ReLU, FOO52, Softmax, None |
| **Loss Functions** | MSE, Cross-Entropy |
| **Optimizers** | SGD, Momentum, RMSProp, Adam, AdamW |
| **Training Strategies** | Online (per-sample), Full Batch, Mini-Batch |
| **Weight Init** | Random, Xavier/Glorot, He/Kaiming |
| **Compute Backends** | Single-threaded, OpenMP, OpenBLAS, CUDA (WIP) |
| **Serialization** | FP32, FP16, BF16, INT8, INT4, INT2 — versioned format |

## Building

Requires **CMake ≥ 3.21** and a **C23-capable compiler** (GCC 14+, Clang 18+, MSVC 19.36+).

```bash
# Set compiler
export CC=clang
export CXX=clang++

# Minimal build (zero dependencies)
cmake -S . -B Build -DCMAKE_BUILD_TYPE=Release
cmake --build Build --parallel

# Full build (OpenMP + OpenBLAS)
cmake -S . -B Build -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_WITH_OPENMP=ON \
      -DBUILD_WITH_OPENBLAS=ON
cmake --build Build --parallel
```

### CMake Options

| Option | Default | Description |
|---|---|---|
| `BUILD_WITH_OPENMP` | `OFF` | Enable OpenMP multi-threading |
| `BUILD_WITH_OPENBLAS` | `OFF` | Enable OpenBLAS for matrix operations |
| `BUILD_EXAMPLE` | `ON` | Build example programs from `Examples/` |

## Why C in 2026?

**Stability. Portability. Uncompromising Control.**

While the industry fetishizes the safety of Rust and the convenience of Python, C remains the immutable substrate of computation.

*   **Embed everywhere:** Execute your inference on a $2 microcontroller or a supercluster without refactoring.
*   **ABI Stability:** Spingalett exposes a raw C ABI, making it trivial to wrap in Python (`ctypes`), Go, Rust, or whatever trendy language happens to be popular next week.
*   **Instant Compilation:** We do not require `cargo` to download half the internet to compile a "Hello, World!".

## Licensing

This intellectual property is released under the **MIT License**. Refer to the [LICENSE](LICENSE) file for legal specifics. Use it however you want.

<div align="center">
  <sub>Built with ❤️ and pointers by <a href="https://github.com/pka-human">pka_human</a> and contributors</sub>
</div>
