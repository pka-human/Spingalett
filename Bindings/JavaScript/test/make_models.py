# SPDX-License-Identifier: MIT
"""Writes the language models of the JavaScript package's tests into test/models:

    python Bindings/JavaScript/test/make_models.py      # needs the library (Bin/) and numpy

Two small causal transformers over a vocabulary of 28 characters (ALPHABET), trained for a moment to
continue a phrase: gpt (learned positions, layer normalization, an MLP with GELU in its first block and
GELU's tanh approximation in its second) and llama (rotary embeddings, grouped-query attention, RMS
normalization, a SwiGLU MLP), each saved in FLOAT32 and INT8: gpt_fp32.slett, gpt_int8.slett,
llama_fp32.slett and llama_int8.slett. Seeded; the files change only when this script or the library's
training does.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "Python"))
import spingalett as sg  # noqa: E402

ALPHABET = " .abcdefghijklmnopqrstuvwxyz"
TEXT = "the quick brown fox jumps over the lazy dog. "
CONTEXT, WIDTH = 8, 16


def windows():
    tokens = [ALPHABET.index(c) for c in TEXT * 3]
    x = np.array([tokens[i:i + CONTEXT] for i in range(len(TEXT))], dtype=np.float32)
    y = np.array([tokens[i + 1:i + CONTEXT + 1] for i in range(len(TEXT))], dtype=np.float32)
    return x, y


def gpt():
    net = sg.Network(sg.Loss.SPARSE_CROSS_ENTROPY)
    net.add_layer(CONTEXT)
    net.add_embedding(len(ALPHABET), WIDTH, positions=True)
    x = net.last
    for act in (sg.Activation.GELU, sg.Activation.GELU_TANH):
        net.add_layer_norm()
        net.add_linear(3 * WIDTH)
        net.add_attention(heads=2, causal=True)
        net.add_linear(WIDTH)
        net.add_add([x, net.last])
        x = net.last
        net.add_layer_norm()
        net.add_linear(2 * WIDTH, activation=act)
        net.add_linear(WIDTH)
        net.add_add([x, net.last])
        x = net.last
    net.add_layer_norm()
    net.add_linear(len(ALPHABET))
    return net


def llama():
    heads, kv_heads, head, hidden = 2, 1, WIDTH // 2, 2 * WIDTH
    net = sg.Network(sg.Loss.SPARSE_CROSS_ENTROPY)
    net.add_layer(CONTEXT)
    net.add_embedding(len(ALPHABET), WIDTH)
    x = net.last
    for _ in range(2):
        net.add_rms_norm()
        net.add_linear((heads + 2 * kv_heads) * head)
        net.add_attention(heads=heads, kv_heads=kv_heads, rope_theta=10000.0, causal=True)
        net.add_linear(WIDTH)
        net.add_add([x, net.last])
        x = net.last
        net.add_rms_norm()
        normed = net.last
        net.add_linear(hidden, activation=sg.Activation.SILU)
        gate = net.last
        net.add_linear(hidden, inputs=normed)
        net.add_multiply([gate, net.last])
        net.add_linear(WIDTH)
        net.add_add([x, net.last])
        x = net.last
    net.add_rms_norm()
    net.add_linear(len(ALPHABET))
    return net


def main():
    out = os.path.join(HERE, "models")
    os.makedirs(out, exist_ok=True)
    sg.set_verbose(False)
    x, y = windows()
    for name, make in (("gpt", gpt), ("llama", llama)):
        sg.seed(2026)
        with make() as net:
            result = net.train(x, y, epochs=150, batch_size=15, optimizer=sg.Optimizer.ADAM, learning_rate=0.01)
            print(f"{name}: loss {result.train_loss:.3f}")
            for suffix, precision in (("fp32", sg.Precision.FLOAT32), ("int8", sg.Precision.INT8)):
                net.save(os.path.join(out, f"{name}_{suffix}.slett"), precision=precision, save_optimizer=False)


if __name__ == "__main__":
    main()
