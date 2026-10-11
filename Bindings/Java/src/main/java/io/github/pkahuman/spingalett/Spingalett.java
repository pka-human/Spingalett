// SPDX-License-Identifier: MIT
// Copyright (c) 2026 pka_human (pka_human@proton.me)

package io.github.pkahuman.spingalett;

import java.lang.foreign.MemorySegment;

/**
 * Spingalett for Java: networks with dense, convolutional and transformer layers, trained by libspingalett on the
 * CPU or the GPU (CUDA, Vulkan), deployment models from FP32 to INT2, and text generation. These are the library's
 * settings; see {@link Network}, {@link Layer} and {@link Model}. Run with --enable-native-access=ALL-UNNAMED (or
 * the module's name).
 */
public final class Spingalett {
    private Spingalett() {}

    /** The library's version ("1.2.0"). */
    public static String version() {
        try {
            return Native.string((MemorySegment) Native.VERSION.invokeExact());
        } catch (Throwable t) {
            throw Native.rethrow(t);
        }
    }

    /** Where training and inference run; false when the mode is not available. */
    public static boolean setComputeMode(ComputeMode mode) {
        try {
            return (boolean) Native.SET_COMPUTE_MODE.invokeExact(mode.value);
        } catch (Throwable t) {
            throw Native.rethrow(t);
        }
    }

    /** Precision of the GPU's products (FLOAT32 or BFLOAT16); false when the GPU has none such. */
    public static boolean setGpuPrecision(Precision precision) {
        try {
            return (boolean) Native.SET_GPU_PRECISION.invokeExact(precision.value);
        } catch (Throwable t) {
            throw Native.rethrow(t);
        }
    }

    /** Seeds the calling thread's generator (initialization, shuffling, dropout, draws). */
    public static void seed(long seed) {
        try {
            Native.SEED.invokeExact(seed);
        } catch (Throwable t) {
            throw Native.rethrow(t);
        }
    }

    public static void setVerbose(boolean enabled) {
        try {
            Native.SET_VERBOSE.invokeExact(enabled);
        } catch (Throwable t) {
            throw Native.rethrow(t);
        }
    }

    /** Threads of the CPU's parallel loops (0: all). */
    public static void setThreads(int threads) {
        try {
            Native.SET_NUM_THREADS.invokeExact(threads);
        } catch (Throwable t) {
            throw Native.rethrow(t);
        }
    }

    /** The GPU that ComputeMode.CUDA uses, or null. */
    public static String cudaDevice() {
        try {
            return Native.string((MemorySegment) Native.CUDA_DEVICE.invokeExact());
        } catch (Throwable t) {
            throw Native.rethrow(t);
        }
    }

    /** The GPU that ComputeMode.VULKAN uses, or null. */
    public static String gpuDevice() {
        try {
            return Native.string((MemorySegment) Native.GPU_DEVICE.invokeExact());
        } catch (Throwable t) {
            throw Native.rethrow(t);
        }
    }
}
