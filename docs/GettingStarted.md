# Getting started

The first program, how the library names things, the examples, and how errors and messages reach a program.

## Quick start

```c
#include <Spingalett/Spingalett.Short.h>
#include <stdio.h>

int main(void) {
    float inputs[4][2]  = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    float targets[4][1] = {{0}, {1}, {1}, {0}};

    NeuralNetwork *net = new_spingalett(LOSS_MSE);
    layer(net, 2);                                              /* input layer */
    layer(net, 8, ACT_TANH, WEIGHT_INITIALIZATION_XAVIER);
    layer(net, 1, ACT_SIGMOID, WEIGHT_INITIALIZATION_XAVIER);

    train(
        .net = net,
        .inputs = &inputs[0][0],
        .targets = &targets[0][0],
        .sample_count = 4,
        .epochs = 5000,
        .training_strategy = STRATEGY_FULL_BATCH,
        .optimizer_type = OPTIMIZER_ADAM,
        .learning_rate = 0.02f,
        .report_interval = 1000
    );

    for (int i = 0; i < 4; i++) {
        float *out = forward(net, inputs[i]);
        printf("%.0f XOR %.0f = %.4f\n", inputs[i][0], inputs[i][1], out[0]);
    }

    save_spingalett(.net = net, .filename = "xor");             /* writes xor.slett */
    free_network(net);
    return 0;
}
```

Every function that takes many parameters is a macro over a struct, so arguments can be given
positionally (`layer(net, 8, ACT_TANH)`) or by name (`layer(.net = net, .neurons_amount = 8)`).
Fields that are not mentioned are zero, which selects the documented default.

**Names.** The examples here and in `Examples/` use the short names of
`Spingalett/Spingalett.Short.h` (`NeuralNetwork`, `layer()`, `train()`, `ACT_RELU`, ...), which read
more easily. The library declares every name with its prefix (`spingalett_` for functions and
builders, `Spingalett` for types, `SPINGALETT_` for constants), and so do the descriptions in these
documents and [docs/Reference.md](Reference.md): C has one namespace for everything a program
includes, and names such as `train()`, `layer()` or `LOG_DEBUG` (also `<syslog.h>`'s) would collide
with the program's own and with other libraries'. The short names are macros and typedefs over the
prefixed ones, so a program compiles to the same calls either way: it includes `Spingalett.Short.h`
in place of `Spingalett.h` (or defines `SPINGALETT_SHORT_NAMES` before including that) to use them,
and `Spingalett.h` alone leaves them out. Functions whose names had the prefix in 0.x keep it
(`spingalett_model_load()`, `spingalett_load_idx()`, `spingalett_set_compute_mode()`).

The `Examples/` directory contains this XOR program, an MNIST classifier
(`Examples/download_mnist.sh data/mnist && Bin/MNIST data/mnist`), a convolutional one
(`Bin/MNIST_CNN data/mnist`, about 99% after two epochs), a CIFAR-10 classifier with batch
normalization and augmentation (`Examples/download_cifar10.sh data/cifar10 && Bin/CIFAR10
data/cifar10`) and the benchmark described under [Performance](Performance.md).
[docs/Tutorial.md](Tutorial.md) goes through these programs in order, and
[docs/Reference.md](Reference.md) lists every declaration of the headers with its comment.

## Errors and logging

A function that fails returns false, `NULL`, NaN (losses and metrics), `SPINGALETT_NO_LAYER` or a report
whose status is `SPINGALETT_TRAIN_FAILED`, and sets the calling thread's error code and message:

```c
spingalett_clear_error();
NeuralNetwork *net = load_spingalett("model.slett");
if (!net)
    fprintf(stderr, "%s (code %d)\n", spingalett_last_error_message(),
            spingalett_last_error_code());
```

| Code | Meaning |
|---|---|
| `SPINGALETT_OK` | No error |
| `SPINGALETT_ERR_ALLOC` | Out of memory |
| `SPINGALETT_ERR_INVALID` | Invalid argument or file contents |
| `SPINGALETT_ERR_FILE_IO` | A file could not be opened, read or written, or is truncated |
| `SPINGALETT_ERR_FORMAT_VERSION` | The model file uses an unsupported format version |

The error state is not cleared by successful calls.

Log messages go to stdout (warnings and errors to stderr) unless redirected with
`spingalett_set_log_callback(void (*)(SpingalettLogLevel, const char *))`.
`spingalett_set_log_level()` sets the minimum level and `spingalett_set_verbose(false)` suppresses
everything below warnings.
