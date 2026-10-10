// SPDX-License-Identifier: MIT
// Copyright (c) 2026 pka_human (pka_human@proton.me)

package io.github.pkahuman.spingalett;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/** A layer to add to a network: made by a kind's method, then options (each returns the layer). */
public final class Layer {
    int type, neurons, act, init, height, width, channels, filters, kernel, stride, padding, groups, upsample;
    int outputPadding, vocabulary, heads, kvHeads;
    float dropout, epsilon, ropeTheta;
    boolean causal, positions;
    int[] inputs = new int[0];

    private Layer(int type) { this.type = type; }

    /** The input layer: height x width x channels values a sample (a window of tokens: 1 x 1 x tokens). */
    public static Layer input(int height, int width, int channels) {
        Layer l = new Layer(0);
        l.height = height;
        l.width = width;
        l.channels = channels;
        return l;
    }
    /** A fully connected layer (LeCun initialization, no activation). */
    public static Layer dense(int neurons) {
        Layer l = new Layer(0);
        l.neurons = neurons;
        l.init = Init.LECUN.value;
        return l;
    }
    /** A 2D convolution: filters output channels, kernel x kernel windows (He initialization, ReLU). */
    public static Layer conv2d(int filters, int kernel) {
        Layer l = new Layer(1);
        l.filters = filters;
        l.kernel = kernel;
        l.act = Activation.RELU.value;
        l.init = Init.HE.value;
        return l;
    }
    /** A transposed 2D convolution (upsampling by its stride). */
    public static Layer convTranspose2d(int filters, int kernel) {
        Layer l = conv2d(filters, kernel);
        l.type = 8;
        return l;
    }
    public static Layer maxPool2d(int kernel) { Layer l = new Layer(2); l.kernel = kernel; return l; }
    public static Layer avgPool2d(int kernel) { Layer l = new Layer(3); l.kernel = kernel; return l; }
    public static Layer globalAvgPool() { return new Layer(7); }
    public static Layer batchNorm() { return new Layer(4); }
    public static Layer layerNorm() { return new Layer(10); }
    /** RMS normalization of each cell's channels (no biases). */
    public static Layer rmsNorm() { return new Layer(13); }
    public static Layer addLayers(int... inputs) { return new Layer(5).inputs(inputs); }
    public static Layer concat(int... inputs) { return new Layer(6).inputs(inputs); }
    /** The product of the layers named, element by element (SwiGLU's gate). */
    public static Layer multiply(int... inputs) { return new Layer(14).inputs(inputs); }
    public static Layer upsample(int factor) { Layer l = new Layer(9); l.stride = factor; return l; }
    /** A linear map of each cell's channels (a 1 x 1 convolution: a transformer's projections). */
    public static Layer linear(int neurons) {
        Layer l = new Layer(1);
        l.filters = neurons;
        l.kernel = 1;
        l.init = Init.LECUN.value;
        return l;
    }
    /** The vectors of width values of a vocabulary's tokens, a token a cell. */
    public static Layer embedding(int vocabulary, int width) {
        Layer l = new Layer(11);
        l.vocabulary = vocabulary;
        l.neurons = width;
        l.init = Init.LECUN.value;
        return l;
    }
    /** Attention of heads query heads over packed queries, keys and values. */
    public static Layer attention(int heads) { Layer l = new Layer(12); l.heads = heads; return l; }

    public Layer activation(Activation a) { act = a.value; return this; }
    public Layer init(Init i) { init = i.value; return this; }
    public Layer dropout(float rate) { dropout = rate; return this; }
    public Layer stride(int s) { stride = s; return this; }
    public Layer padding(int p) { padding = p; return this; }
    public Layer groups(int g) { groups = g; return this; }
    public Layer epsilon(float e) { epsilon = e; return this; }
    public Layer outputPadding(int p) { outputPadding = p; return this; }
    public Layer bilinear() { upsample = 1; return this; }
    public Layer kvHeads(int h) { kvHeads = h; return this; }
    public Layer causal(boolean c) { causal = c; return this; }
    public Layer ropeTheta(float t) { ropeTheta = t; return this; }
    public Layer positions(boolean p) { positions = p; return this; }
    /** The layers it reads (indices that Network.add returned; 0 the input layer); none: the last one. */
    public Layer inputs(int... names) {
        if (names.length > Native.MAX_INPUTS) throw new IllegalArgumentException("a layer reads at most 16 layers");
        inputs = names.clone();
        return this;
    }

    /** The layer's arguments in a zeroed structure. */
    void write(MemorySegment s, MemorySegment net) {
        var L = Native.LAYER_ARGS;
        Native.set(s, L, "net", net);
        Native.set(s, L, "neurons_amount", neurons);
        Native.set(s, L, "act_func", act);
        Native.set(s, L, "weight_initialization", init);
        Native.set(s, L, "dropout_rate", dropout);
        Native.set(s, L, "type", type);
        Native.set(s, L, "height", height);
        Native.set(s, L, "width", width);
        Native.set(s, L, "channels", channels);
        Native.set(s, L, "filters", filters);
        Native.set(s, L, "kernel", kernel);
        Native.set(s, L, "stride", stride);
        Native.set(s, L, "padding", padding);
        Native.set(s, L, "groups", groups);
        Native.set(s, L, "epsilon", epsilon);
        Native.set(s, L, "upsample", upsample);
        Native.set(s, L, "output_padding", outputPadding);
        Native.set(s, L, "vocabulary", vocabulary);
        Native.set(s, L, "heads", heads);
        Native.set(s, L, "kv_heads", kvHeads);
        Native.set(s, L, "rope_theta", ropeTheta);
        Native.set(s, L, "causal", (byte) (causal ? 1 : 0));
        Native.set(s, L, "positions", (byte) (positions ? 1 : 0));
        long at = L.byteOffset(java.lang.foreign.MemoryLayout.PathElement.groupElement("inputs"));
        for (int k = 0; k < inputs.length; k++) s.set(ValueLayout.JAVA_INT, at + 4L * k, inputs[k]);
        Native.set(s, L, "input_count", inputs.length);
    }
}
