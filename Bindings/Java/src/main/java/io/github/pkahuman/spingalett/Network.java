// SPDX-License-Identifier: MIT
// Copyright (c) 2026 pka_human (pka_human@proton.me)

package io.github.pkahuman.spingalett;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SegmentAllocator;
import java.lang.foreign.ValueLayout;
import java.nio.file.Path;

/** A network: its layers, parameters and training state; used by one thread at a time. */
public final class Network implements AutoCloseable {
    private MemorySegment ptr;

    private Network(MemorySegment ptr) { this.ptr = ptr; }

    /** An empty network that trains with loss; its first layer is the input layer. */
    public Network(Loss loss) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment args = arena.allocate(Native.NETWORK_ARGS);
            Native.set(args, Native.NETWORK_ARGS, "loss_func", loss.value);
            SpingalettException.clear();
            ptr = (MemorySegment) Native.NETWORK_NEW.invokeExact(args);
        } catch (Throwable t) {
            throw Native.rethrow(t);
        }
        if (ptr.equals(MemorySegment.NULL)) throw SpingalettException.last("network allocation failed");
    }

    /** A network from a .slett file. */
    public static Network load(Path path) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment p = (MemorySegment) Native.LOAD.invokeExact(arena.allocateFrom(path.toString()));
            if (p.equals(MemorySegment.NULL)) throw SpingalettException.last("load failed");
            return new Network(p);
        } catch (Throwable t) {
            throw Native.rethrow(t);
        }
    }

    MemorySegment ptr() {
        if (ptr == null) throw new IllegalStateException("closed network");
        return ptr;
    }

    /** Appends a layer; returns its index (what later layers' inputs name). */
    public int add(Layer layer) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment args = arena.allocate(Native.LAYER_ARGS);
            layer.write(args, ptr());
            SpingalettException.clear();
            int index = (int) Native.APPEND_LAYER.invokeExact(args);
            if (index == -1) throw SpingalettException.last("the layer does not fit the network");
            return index;
        } catch (Throwable t) {
            throw Native.rethrow(t);
        }
    }

    private int size(java.lang.invoke.MethodHandle h) {
        try {
            return (int) h.invokeExact(ptr());
        } catch (Throwable t) {
            throw Native.rethrow(t);
        }
    }

    public int layerCount() { return size(Native.LAYER_COUNT); }
    public int inputSize() { return size(Native.INPUT_SIZE); }
    public int outputSize() { return size(Native.OUTPUT_SIZE); }
    /** Targets a sample (the outputs, or one class index a cell with the sparse cross-entropy). */
    public int targetSize() { return size(Native.TARGET_SIZE); }
    public long parameterCount() {
        try {
            return (long) Native.PARAMETER_COUNT.invokeExact(ptr());
        } catch (Throwable t) {
            throw Native.rethrow(t);
        }
    }

    private static int samples(int values, int size, String what) {
        if (size == 0 || values % size != 0)
            throw new IllegalArgumentException(what + ": " + values + " values are no whole number of samples of " + size);
        return values / size;
    }

    private MemorySegment trainArgs(Arena arena, TrainOptions o) {
        var T = Native.TRAIN_ARGS;
        MemorySegment a = arena.allocate(T);
        Native.set(a, T, "net", ptr());
        Native.set(a, T, "training_strategy", o.strategy.value);
        Native.set(a, T, "optimizer_type", o.optimizer.value);
        Native.set(a, T, "batch_size", o.batchSize);
        Native.set(a, T, "do_not_shuffle", (byte) (o.noShuffle ? 1 : 0));
        Native.set(a, T, "epochs", (long) o.epochs);
        Native.set(a, T, "learning_rate", o.learningRate);
        Native.set(a, T, "weight_decay", o.weightDecay);
        Native.set(a, T, "momentum", o.momentum);
        Native.set(a, T, "beta1", o.beta1);
        Native.set(a, T, "beta2", o.beta2);
        Native.set(a, T, "max_grad_norm", o.maxGradNorm);
        Native.set(a, T, "label_smoothing", o.labelSmoothing);
        return a;
    }

    private static TrainResult run(Arena arena, MemorySegment args) {
        MemorySegment r;
        try {
            r = (MemorySegment) Native.TRAIN.invokeExact((SegmentAllocator) arena, args);
        } catch (Throwable t) {
            throw Native.rethrow(t);
        }
        int status = (int) Native.get(r, Native.TRAIN_REPORT, "status");
        if (status == 0 || status == 5) throw SpingalettException.last("training failed");
        return new TrainResult(status == 1, (long) Native.get(r, Native.TRAIN_REPORT, "epochs_run"),
                               (float) Native.get(r, Native.TRAIN_REPORT, "train_loss"));
    }

    /** Trains on samples in arrays: inputSize() values a sample of inputs, targetSize() of targets. */
    public TrainResult train(float[] inputs, float[] targets, TrainOptions options) {
        int n = samples(inputs.length, inputSize(), "train");
        if (samples(targets.length, targetSize(), "train") != n)
            throw new IllegalArgumentException("as many samples of targets as of inputs");
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment a = trainArgs(arena, options);
            Native.set(a, Native.TRAIN_ARGS, "inputs", arena.allocateFrom(ValueLayout.JAVA_FLOAT, inputs));
            Native.set(a, Native.TRAIN_ARGS, "targets", arena.allocateFrom(ValueLayout.JAVA_FLOAT, targets));
            Native.set(a, Native.TRAIN_ARGS, "sample_count", n);
            return run(arena, a);
        }
    }

    /**
     * Trains a language model on the windows of a file of token ids (nanoGPT's .bin, llm.c's): each sample
     * inputSize() tokens from a multiple of stride (0: the window's length) on, its targets the token after each.
     */
    public TrainResult trainTokens(Path path, int stride, TrainOptions options) {
        try (Arena arena = Arena.ofConfined()) {
            var R = Native.TOKEN_READER_OPTIONS;
            MemorySegment ro = arena.allocate(R);
            Native.set(ro, R, "context", inputSize());
            Native.set(ro, R, "stride", stride);
            Native.set(ro, R, "shuffle", (byte) (options.noShuffle ? 0 : 1));
            MemorySegment reader = (MemorySegment) Native.DATASET_OPEN_TOKENS.invokeExact(arena.allocateFrom(path.toString()), ro);
            if (reader.equals(MemorySegment.NULL)) throw SpingalettException.last("spingalett_dataset_open_tokens failed");
            try {
                MemorySegment info = (MemorySegment) Native.DATASET_INFO_OF.invokeExact((SegmentAllocator) arena, reader);
                MemorySegment a = trainArgs(arena, options);
                Native.set(a, Native.TRAIN_ARGS, "training_mode", 1);
                Native.set(a, Native.TRAIN_ARGS, "generator", Native.DATASET_GENERATOR);
                Native.set(a, Native.TRAIN_ARGS, "generator_data", reader);
                Native.set(a, Native.TRAIN_ARGS, "sample_count", (int) Native.get(info, Native.DATASET_INFO, "count"));
                return run(arena, a);
            } finally {
                Native.DATASET_CLOSE.invokeExact(reader);
            }
        } catch (Throwable t) {
            throw Native.rethrow(t);
        }
    }

    /** The outputs of samples of inputSize() values each. */
    public float[] predict(float[] inputs) {
        int n = samples(inputs.length, inputSize(), "predict");
        int out = outputSize();
        try (Arena arena = Arena.ofConfined()) {
            var P = Native.PREDICT_ARGS;
            MemorySegment y = arena.allocate(ValueLayout.JAVA_FLOAT, (long) n * out);
            MemorySegment a = arena.allocate(P);
            Native.set(a, P, "net", ptr());
            Native.set(a, P, "inputs", arena.allocateFrom(ValueLayout.JAVA_FLOAT, inputs));
            Native.set(a, P, "sample_count", n);
            Native.set(a, P, "outputs", y);
            if (!(boolean) Native.PREDICT.invokeExact(a)) throw SpingalettException.last("predict failed");
            return y.toArray(ValueLayout.JAVA_FLOAT);
        } catch (Throwable t) {
            throw Native.rethrow(t);
        }
    }

    /** Mean loss and accuracy over samples. */
    public Metrics evaluate(float[] inputs, float[] targets) {
        int n = samples(inputs.length, inputSize(), "evaluate");
        if (samples(targets.length, targetSize(), "evaluate") != n)
            throw new IllegalArgumentException("as many samples of targets as of inputs");
        try (Arena arena = Arena.ofConfined()) {
            var E = Native.EVALUATE_ARGS;
            MemorySegment a = arena.allocate(E);
            Native.set(a, E, "net", ptr());
            Native.set(a, E, "inputs", arena.allocateFrom(ValueLayout.JAVA_FLOAT, inputs));
            Native.set(a, E, "targets", arena.allocateFrom(ValueLayout.JAVA_FLOAT, targets));
            Native.set(a, E, "sample_count", n);
            MemorySegment m = (MemorySegment) Native.EVALUATE.invokeExact((SegmentAllocator) arena, a);
            float loss = (float) Native.get(m, Native.EVAL_METRICS, "loss");
            if (Float.isNaN(loss)) throw SpingalettException.last("evaluate failed");
            return new Metrics(loss, (float) Native.get(m, Native.EVAL_METRICS, "accuracy"));
        } catch (Throwable t) {
            throw Native.rethrow(t);
        }
    }

    /** Continues a prompt of token ids by count tokens with a causal language model, a token at a time. */
    public int[] generate(int[] prompt, int count, Sampling sampling) {
        if (prompt.length == 0) throw new IllegalArgumentException("a prompt of one token at least");
        try (Arena arena = Arena.ofConfined()) {
            var G = Native.GENERATE_ARGS;
            MemorySegment tokens = arena.allocate(ValueLayout.JAVA_INT, Math.max(count, 1));
            MemorySegment a = arena.allocate(G);
            Native.set(a, G, "net", ptr());
            Native.set(a, G, "prompt", arena.allocateFrom(ValueLayout.JAVA_INT, prompt));
            Native.set(a, G, "prompt_length", prompt.length);
            Native.set(a, G, "tokens", tokens);
            Native.set(a, G, "count", count);
            Native.set(a, G, "temperature", sampling.temperature);
            Native.set(a, G, "top_k", sampling.topK);
            Native.set(a, G, "top_p", sampling.topP);
            Native.set(a, G, "seed", sampling.seed);
            if (sampling.stop.length > 0) {
                Native.set(a, G, "stop_tokens", arena.allocateFrom(ValueLayout.JAVA_INT, sampling.stop));
                Native.set(a, G, "stop_count", sampling.stop.length);
            }
            SpingalettException.clear();
            int made = (int) Native.GENERATE.invokeExact(a);
            if (SpingalettException.code0() != 0) throw SpingalettException.last("generate failed");
            return java.util.Arrays.copyOf(tokens.toArray(ValueLayout.JAVA_INT), made);
        } catch (Throwable t) {
            throw Native.rethrow(t);
        }
    }

    /** Saves the network (precision of the weights in the file). */
    public void save(Path path, Precision precision) {
        try (Arena arena = Arena.ofConfined()) {
            var S = Native.SAVE_ARGS;
            MemorySegment a = arena.allocate(S);
            Native.set(a, S, "net", ptr());
            Native.set(a, S, "filename", arena.allocateFrom(path.toString()));
            Native.set(a, S, "precision", precision.value);
            if (!(boolean) Native.SAVE.invokeExact(a)) throw SpingalettException.last("save failed");
        } catch (Throwable t) {
            throw Native.rethrow(t);
        }
    }

    private float[] read(int layer, int kind) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment info = arena.allocate(Native.NETWORK_LAYER);
            if (!(boolean) Native.NETWORK_LAYER_INFO.invokeExact(ptr(), layer, info)) throw SpingalettException.last("no such layer");
            long count = (long) Native.get(info, Native.NETWORK_LAYER, kind == 0 ? "weight_count" : "bias_count");
            if (count == 0) return new float[0];
            MemorySegment v = arena.allocate(ValueLayout.JAVA_FLOAT, count);
            if (!(boolean) Native.GET_PARAMETERS.invokeExact(ptr(), layer, kind, v, count))
                throw SpingalettException.last("get_parameters failed");
            return v.toArray(ValueLayout.JAVA_FLOAT);
        } catch (Throwable t) {
            throw Native.rethrow(t);
        }
    }

    /** The weights of a layer (rows of outputs, channels last). */
    public float[] weights(int layer) { return read(layer, 0); }
    public float[] biases(int layer) { return read(layer, 1); }

    public void setParameters(int layer, boolean biases, float[] values) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment v = arena.allocateFrom(ValueLayout.JAVA_FLOAT, values);
            if (!(boolean) Native.SET_PARAMETERS.invokeExact(ptr(), layer, biases ? 1 : 0, v, (long) values.length))
                throw SpingalettException.last("set_parameters failed");
        } catch (Throwable t) {
            throw Native.rethrow(t);
        }
    }

    /** The network as a deployment model of a precision. */
    public Model toModel(Precision precision) {
        try {
            MemorySegment m = (MemorySegment) Native.MODEL_FROM_NETWORK.invokeExact(ptr(), precision.value);
            if (m.equals(MemorySegment.NULL)) throw SpingalettException.last("spingalett_model_from_network failed");
            return new Model(m);
        } catch (Throwable t) {
            throw Native.rethrow(t);
        }
    }

    @Override
    public void close() {
        if (ptr == null) return;
        try {
            Native.NETWORK_FREE.invokeExact(ptr);
        } catch (Throwable t) {
            throw Native.rethrow(t);
        }
        ptr = null;
    }
}
