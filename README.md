<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="Images/LogoDark.png">
  <source media="(prefers-color-scheme: light)" srcset="Images/LogoLight.png">
  <img alt="Spingalett Logo" src="Images/LogoLight.png">
</picture>

#

[![Standard](https://img.shields.io/badge/C-23-blue.svg?style=flat-square&logo=c)](https://en.wikipedia.org/wiki/C23_(C_standard_revision))
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=flat-square)](LICENSE)
[![Dependencies](https://img.shields.io/badge/Dependencies-Zero_Required-red.svg?style=flat-square)]()
[![CUDA](https://img.shields.io/badge/CUDA-Supported-76b900.svg?style=flat-square&logo=nvidia)]()

<p>
Modern C23 Deep Learning Engine: Architectural austerity, linear memory topology, and a declarative interface engineered for raw throughput.
</p>

<!--<a href="https://www.star-history.com/#pka-human/Spingalett&type=date&legend=bottom-right">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=pka-human/Spingalett&type=date&theme=dark&legend=bottom-right" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=pka-human/Spingalett&type=date&legend=bottom-right" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=pka-human/Spingalett&type=date&legend=bottom-right" />
 </picture>
</a>-->

</div>

## Rationale

The contemporary machine learning landscape has devolved into a bloated dependency hell. **Spingalett** rejects the premise that one requires gigabytes of Python site-packages or a fragile virtual environment to execute a matrix multiplication.

This library is a return to the fundamentals of high-performance computing. It is authored in **Pure C23**, deliberately bypassing the overhead of major frameworks like TensorFlow and PyTorch to deliver superior throughput. It is designed for engineers who prefer manual memory management over garbage-collected hand-holding.

### Core Competencies

*   **Dependency Asceticism:** The engine operates strictly on standard C (libc). We do not tolerate external bloatware.
*   **Hybrid Compute Backend:** Provides seamless interoperability between optimized CPU routines and NVIDIA GPU acceleration (**cuBLAS / cuDNN**), allowing for context-aware performance scaling.
*   **Linearized Memory Model:** Implements a strict "Flattening" protocol for tensors, forcing data locality to maximize CPU cache coherency and SIMD vectorization efficiency.
*   **Modern C23 Syntax:** Leverages designated initializers (`.field = value`) to provide a declarative, pseudo-high-level API without sacrificing the raw speed of the underlying binary.

## 🛠 Usage Specification

Network definition in C23 now achieves a syntactical elegance previously reserved for interpreted languages, yet retains the ruthless efficiency of compiled code.

```c
#include "Spingalett/Spingalett.h"

int main() {
    // 1. Instantiation
    NeuralNetwork *nn = new_spingalett(LOSS_MSE);

    // 2. Architecture Definition (Declarative)
    layer(nn, 2); // Input vector
    layer(nn, 4, RELU, WEIGHT_INITIALIZATION_HE); // Hidden manifold
    layer(nn, 1, CT_SIGMOID, WEIGHT_INITIALIZATION_XAVIER); // Output scalar

    // 3. Execution (Training Loop)
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
## Building

TODO: WRITE HOW TO BUILD THIS SHIT

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
