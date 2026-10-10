// SPDX-License-Identifier: MIT
// Copyright (c) 2026 pka_human (pka_human@proton.me)

package io.github.pkahuman.spingalett;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.file.Path;

/** A deployment model: inference only, in its weights' precision; threads may share one. */
public final class Model implements AutoCloseable {
    private MemorySegment ptr;

    Model(MemorySegment ptr) { this.ptr = ptr.reinterpret(Native.MODEL.byteSize()); }

    /** A model from a .slett file. */
    public static Model load(Path path) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment m = (MemorySegment) Native.MODEL_LOAD.invokeExact(arena.allocateFrom(path.toString()));
            if (m.equals(MemorySegment.NULL)) throw SpingalettException.last("spingalett_model_load failed");
            return new Model(m);
        } catch (Throwable t) {
            throw Native.rethrow(t);
        }
    }

    private MemorySegment ptr() {
        if (ptr == null) throw new IllegalStateException("closed model");
        return ptr;
    }

    public int inputSize() { return (int) Native.get(ptr(), Native.MODEL, "input_size"); }
    public int outputSize() { return (int) Native.get(ptr(), Native.MODEL, "output_size"); }

    /** The outputs of samples of inputSize() values each. */
    public float[] predict(float[] inputs) {
        int size = inputSize();
        if (size == 0 || inputs.length % size != 0)
            throw new IllegalArgumentException(inputs.length + " values are no whole number of samples of " + size);
        int n = inputs.length / size;
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment y = arena.allocate(ValueLayout.JAVA_FLOAT, (long) n * outputSize());
            MemorySegment x = arena.allocateFrom(ValueLayout.JAVA_FLOAT, inputs);
            if (!(boolean) Native.MODEL_PREDICT.invokeExact(ptr(), x, n, y)) throw SpingalettException.last("model predict failed");
            return y.toArray(ValueLayout.JAVA_FLOAT);
        } catch (Throwable t) {
            throw Native.rethrow(t);
        }
    }

    @Override
    public void close() {
        if (ptr == null) return;
        try {
            Native.MODEL_FREE.invokeExact(ptr);
        } catch (Throwable t) {
            throw Native.rethrow(t);
        }
        ptr = null;
    }
}
