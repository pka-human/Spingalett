# Spingalett for Java

Bindings of [Spingalett](https://github.com/pka-human/Spingalett), a neural-network library in C, through the
foreign function and memory API (Java 22 and later; no JNI, no generated code): dense, convolutional and
transformer layers as chains or graphs, trained on the CPU or the GPU (CUDA, Vulkan), deployment models from
FP32 down to INT2, and text generation.

The classes call the shared library `spingalett` (`libspingalett.so`, `spingalett.dll`, `libspingalett.dylib`,
version 1.2 or later) where the system's loader finds it (an installed copy, `LD_LIBRARY_PATH`, `PATH`), or
the file the system property `spingalett.library` names. Run with `--enable-native-access=ALL-UNNAMED` (or
your module's name).

```java
import io.github.pkahuman.spingalett.*;

try (Network net = new Network(Loss.MSE)) {             // XOR
    net.add(Layer.input(1, 1, 2));
    net.add(Layer.dense(8).activation(Activation.TANH));
    net.add(Layer.dense(1).activation(Activation.SIGMOID));
    float[] x = {0, 0, 0, 1, 1, 0, 1, 1}, y = {0, 1, 1, 0};
    net.train(x, y, new TrainOptions().epochs(1500).batchSize(4).learningRate(0.05f));
    System.out.println(java.util.Arrays.toString(net.predict(x)));
}

try (Network lm = new Network(Loss.SPARSE_CROSS_ENTROPY)) {   // a small LLaMA-like language model
    lm.add(Layer.input(1, 1, 128));                     // a window of 128 tokens
    int h = lm.add(Layer.embedding(256, 128));
    lm.add(Layer.rmsNorm());
    lm.add(Layer.linear(3 * 128));
    lm.add(Layer.attention(4).causal(true).ropeTheta(10000f));
    int a = lm.add(Layer.linear(128));
    lm.add(Layer.addLayers(h, a));
    lm.add(Layer.linear(256));
    Spingalett.setComputeMode(ComputeMode.CUDA);        // when there is an NVIDIA GPU
    lm.trainTokens(java.nio.file.Path.of("train.bin"), 64, new TrainOptions().epochs(5));
    int[] tokens = lm.generate(new int[] {72, 101}, 100, new Sampling().temperature(0.8f).topK(40));
}
```

A `Network` is used by one thread at a time; a `Model` may be shared. Tests: `mvn test` with the library on
`LD_LIBRARY_PATH`; `SPINGALETT_LAYOUT=/path/to/Bin/SpingalettLayout` also compares every structure's layout
with the C compiler's.
