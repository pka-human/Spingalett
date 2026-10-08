# Spingalett on a microcontroller

This example recognises handwritten digits on a Cortex-M4F with the standalone Spingalett
inference engine. The MNIST model, quantized to INT8, and 100 test digits are compiled into flash;
the program needs 2.8 KB of RAM for the engine's workspace and no heap. It runs on QEMU's MPS2
AN386 board, so it can be tried without hardware:

```
model in flash: 238128 bytes, workspace in RAM: 2832 bytes
  layer 1: dense 784 -> 256, INT8 weights
  layer 2: dense 256 -> 128, INT8 weights
  layer 3: dense 128 -> 10, INT8 weights
234752 multiply-accumulates per digit
  digit 7: recognised as 7 (99.9%)
  digit 2: recognised as 2 (99.7%)
  ...
accuracy: 99 of 100 test digits (99.0%)
```

## Running it

```bash
# on the host: the library, ModelTool and a trained model
cmake -S . -B Build -DCMAKE_BUILD_TYPE=Release -DBUILD_WITH_OPENMP=ON && cmake --build Build
sh Examples/download_mnist.sh data/mnist
Bin/MNIST data/mnist 5                       # writes mnist.slett

# cross-compile for the Cortex-M4 and run it in QEMU
sudo apt-get install gcc-arm-none-eabi libnewlib-arm-none-eabi qemu-system-arm
Examples/Embedded/run-qemu.sh mnist.slett data/mnist            # --precision int4, fp16, ...
```

`run-qemu.sh` exports the model with `ModelTool header`, writes the test digits with
`make_samples.py`, compiles `mnist_mcu.c`, the start-up code and `Src/Spingalett.Inference.c` with
`arm-none-eabi-gcc`, prints the code and data sizes and runs the program in `qemu-system-arm`. The
program prints through semihosting and exits with status 0 when at least 95 of the 100 digits are
recognised.

| Model | Flash for the model | Workspace | Accuracy on the 100 digits |
|---|---:|---:|---:|
| FP16 | 471 KB | 2.0 KB | 99% |
| INT8 | 238 KB | 2.8 KB | 99% |
| INT4 | 121 KB | 2.8 KB | 99% |
| INT2 | 62 KB | 2.8 KB | 99% |

On all 10,000 MNIST test images (`ModelTool eval` prints the table), this 784-256-128-10 network
scores the same in FP32, FP16 and INT8 and 0.1 to 0.2% lower in INT4; INT2 costs it 2 to 3%. Wider
networks lose more in INT2 without quantization-aware training (DigitPad's 784-1024-512-10 model
keeps 99.3% in INT8 and INT4 and drops to 80% in INT2), so INT2 suits small models.

The convolutional network of `Examples/MNIST_CNN.c` runs the same way
(`Bin/MNIST_CNN data/mnist 2`, then `run-qemu.sh mnist_cnn.slett data/mnist`). Its activations
need a larger workspace, and the window of the first convolution is short, so it accumulates all 32
filters at once per pixel:

```
model in flash: 423744 bytes, workspace in RAM: 208384 bytes
  layer 1: convolution 3x3, 32 filters, output 28x28, INT8 weights
  layer 2: max pooling 2x2, output 14x14x32
  layer 3: convolution 3x3, 64 filters, output 14x14, INT8 weights
  layer 4: max pooling 2x2, output 7x7x64
  layer 5: dense 3136 -> 128, INT8 weights
  layer 6: dense 128 -> 10, INT8 weights
4241152 multiply-accumulates per digit
  ...
accuracy: 99 of 100 test digits (99.0%)
```

On the 10,000 test images it scores 98.9% in FP32, FP16 and INT8 and 98.75% in INT4.

## Using the engine in your firmware

1. Copy `Include/Spingalett/Spingalett.Inference.h` and `Src/Spingalett.Inference.c` (and
   `Src/Spingalett.Engine.h`, which it includes) into the project, put the `Include` directory on
   the include path and compile with `-DSPINGALETT_INFERENCE_ONLY`. The engine calls only
   `memcpy`, `memset`, `memcmp`, `expf`, `tanhf` and `lrintf`; it takes about 12 KB of flash on a
   Cortex-M4 at `-O2`, convolution and pooling included.
2. Export the trained model: `ModelTool header model.slett model.h my_model --precision int8`
   (or `spingalett_export_c_header()` from your own program). The header holds a 16-byte aligned
   `static const uint8_t my_model[]` and the macros `MY_MODEL_SIZE`, `MY_MODEL_INPUTS`,
   `MY_MODEL_OUTPUTS` and `MY_MODEL_WORKSPACE`.
3. Run it:

```c
#include <Spingalett/Spingalett.Inference.h>
#include "model.h"

static SpingalettModel model;
static float workspace[MY_MODEL_WORKSPACE / sizeof(float)];

void setup(void) {
    if (spingalett_model_init(&model, my_model, MY_MODEL_SIZE) != SPINGALETT_OK) {
        /* the image is damaged: its checksums are checked here */
    }
}

void classify(const float input[MY_MODEL_INPUTS], float output[MY_MODEL_OUTPUTS]) {
    spingalett_model_run(&model, input, output, workspace);
}
```

`spingalett_model_init` checks the image once (header, layer table, bounds and CRC-32) and
`spingalett_model_run` evaluates it in place, so the weights are read straight from flash. The
model is read-only: several tasks can share it, each with its own workspace. On cores with the
DSP extension (Cortex-M4, M7, M33) the INT8 dot products use `SMLAD`, two multiply-accumulates
per instruction.

Graphs (residual blocks, concatenated branches; `.slett` format version 6) run the same way. The
file records where each layer's output lives in the workspace, planned when the model was saved so
that outputs alive at the same time do not overlap: the workspace holds a few of the network's
widest outputs, not all of them, however deep the network is.

## Files

| File | Contents |
|---|---|
| `mnist_mcu.c` | the program: checks the model, classifies the digits, prints the accuracy |
| `startup.c` | vector table, `.data` and `.bss` set-up, FPU enable, semihosting |
| `mps2-an386.ld` | linker script for the board's 4 MB of code memory and 4 MB of RAM |
| `make_samples.py` | writes MNIST test digits as a C header |
| `run-qemu.sh` | export, build, size report and QEMU run |
