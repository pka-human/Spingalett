// SPDX-License-Identifier: MIT
// Copyright (c) 2026 pka_human (pka_human@proton.me)

package io.github.pkahuman.spingalett;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.StructLayout;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The bindings against the library: training, prediction, models, a transformer and its generation, and (with
 * SPINGALETT_LAYOUT naming Bin/SpingalettLayout) the structures' layout against the C compiler's.
 */
class SpingalettTest {
    static final float[] X = {0, 0, 0, 1, 1, 0, 1, 1}, Y = {0, 1, 1, 0};

    static Network xor() {
        Spingalett.setVerbose(false);
        Spingalett.seed(7);
        Network net = new Network(Loss.MSE);
        net.add(Layer.input(1, 1, 2));
        net.add(Layer.dense(8).activation(Activation.TANH));
        net.add(Layer.dense(1).activation(Activation.SIGMOID));
        return net;
    }

    @Test
    void version() {
        assertTrue(Spingalett.version().startsWith("1."), Spingalett.version());
    }

    @Test
    void trainsAndPredicts() {
        try (Network net = xor()) {
            assertEquals(2, net.inputSize());
            assertEquals(3, net.layerCount());
            TrainResult r = net.train(X, Y, new TrainOptions().epochs(1500).batchSize(4).learningRate(0.05f));
            assertTrue(r.completed() && r.trainLoss() < 0.05f, r.toString());
            float[] out = net.predict(X);
            for (int i = 0; i < Y.length; i++) assertTrue(Math.abs(out[i] - Y[i]) < 0.3f);
            assertTrue(net.evaluate(X, Y).loss() < 0.05f);
            assertEquals(16, net.weights(1).length);
            assertEquals(8, net.biases(1).length);
            assertThrows(IllegalArgumentException.class, () -> net.predict(new float[3]));
        }
    }

    @Test
    void filesAndModels(@TempDir Path dir) {
        try (Network net = xor()) {
            net.train(X, Y, new TrainOptions().epochs(50).batchSize(4));
            Path path = dir.resolve("net.slett");
            net.save(path, Precision.FLOAT32);
            float[] out = net.predict(X);
            try (Network back = Network.load(path)) {
                assertArrayEquals(out, back.predict(X));
            }
            try (Model model = Model.load(path)) {
                assertEquals(2, model.inputSize());
                assertArrayEquals(out, model.predict(X), 1e-5f);
            }
            try (Model int8 = net.toModel(Precision.INT8)) {
                assertEquals(4, int8.predict(X).length);
            }
        }
        assertThrows(SpingalettException.class, () -> Network.load(Path.of("/nonexistent/net.slett")));
    }

    @Test
    void transformerGenerates() {
        Spingalett.setVerbose(false);
        Spingalett.seed(3);
        final int T = 16, V = 13;
        try (Network lm = new Network(Loss.SPARSE_CROSS_ENTROPY)) {
            lm.add(Layer.input(1, 1, T));
            int h = lm.add(Layer.embedding(V, 16));
            lm.add(Layer.rmsNorm());
            lm.add(Layer.linear(48));
            lm.add(Layer.attention(4).kvHeads(2).causal(true).ropeTheta(10000f));
            int a = lm.add(Layer.linear(16));
            lm.add(Layer.addLayers(h, a));
            lm.add(Layer.linear(V));
            assertEquals(T, lm.targetSize());
            assertEquals(T * V, lm.outputSize());
            float[] x = new float[256 * T], y = new float[256 * T];
            for (int s = 0; s < 256; s++)
                for (int t = 0; t < T; t++) {
                    int start = (s * 7 + 3) % V;
                    x[s * T + t] = (start + t) % V;
                    y[s * T + t] = (start + t + 1) % V;
                }
            TrainResult r = lm.train(x, y, new TrainOptions().epochs(30).batchSize(32).learningRate(1e-2f));
            assertTrue(r.trainLoss() < 0.3f, r.toString());
            int[] expected = new int[20];
            for (int i = 0; i < 20; i++) expected[i] = (6 + i) % V;
            assertArrayEquals(expected, lm.generate(new int[] {3, 4, 5}, 20, new Sampling()));
            Sampling drawn = new Sampling().temperature(1).topP(0.9f).seed(8);
            assertArrayEquals(lm.generate(new int[] {3, 4, 5}, 20, drawn), lm.generate(new int[] {3, 4, 5}, 20, drawn));
            assertArrayEquals(new int[] {6, 7, 8, 9}, lm.generate(new int[] {3, 4, 5}, 20, new Sampling().stop(9)));
        }
    }

    @Test
    void layoutMatchesC() throws Exception {
        String tool = System.getenv("SPINGALETT_LAYOUT");
        if (tool == null) {
            System.out.println("SPINGALETT_LAYOUT not set: layout not checked");
            return;
        }
        Map<String, Long> c = new HashMap<>();
        Process p = new ProcessBuilder(tool).start();
        try (BufferedReader in = new BufferedReader(new InputStreamReader(p.getInputStream()))) {
            for (String line; (line = in.readLine()) != null; ) {
                String[] parts = line.trim().split("\\s+");
                if (parts.length == 3 && parts[1].equals("size")) c.put(parts[0] + " size", Long.parseLong(parts[2]));
                else if (parts.length == 2) c.put(parts[0], Long.parseLong(parts[1]));
            }
        }
        p.waitFor();
        List<String> bad = new ArrayList<>();
        Object[][] structs = {
            {"NeuralNetworkArgs", Native.NETWORK_ARGS}, {"LayerArgs", Native.LAYER_ARGS}, {"TrainArgs", Native.TRAIN_ARGS},
            {"TrainReport", Native.TRAIN_REPORT}, {"EvalMetrics", Native.EVAL_METRICS}, {"PredictArgs", Native.PREDICT_ARGS},
            {"EvaluateArgs", Native.EVALUATE_ARGS}, {"SaveArgs", Native.SAVE_ARGS},
            {"SpingalettGenerateArgs", Native.GENERATE_ARGS}, {"SpingalettTokenReaderOptions", Native.TOKEN_READER_OPTIONS},
            {"SpingalettDatasetInfo", Native.DATASET_INFO}, {"SpingalettNetworkLayer", Native.NETWORK_LAYER},
            {"SpingalettModel", Native.MODEL},
        };
        for (Object[] s : structs) {
            String name = (String) s[0];
            StructLayout layout = (StructLayout) s[1];
            Long size = c.get(name + " size");
            if (size == null || size != layout.byteSize()) bad.add(name + " size " + layout.byteSize() + ", C " + size);
            for (MemoryLayout m : layout.memberLayouts()) {
                if (m.name().isEmpty() || m.name().get().endsWith("_")) continue;
                String field = m.name().get();
                long off = layout.byteOffset(MemoryLayout.PathElement.groupElement(field));
                Long want = c.get(name + "." + field);
                if (want == null || want != off) bad.add(name + "." + field + " at " + off + ", C " + want);
            }
        }
        assertTrue(bad.isEmpty(), String.join("\n", bad));
    }
}
