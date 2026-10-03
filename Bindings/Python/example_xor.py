# SPDX-License-Identifier: MIT
"""XOR with the Python bindings. Run from the repository root after building the library:
    python Bindings/Python/example_xor.py
"""
import numpy as np
import spingalett as sg

x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y = np.array([[0], [1], [1], [0]], dtype=np.float32)

sg.set_verbose(False)
sg.seed(42)

with sg.Network(sg.Loss.MSE, [sg.Layer(2),
                              sg.Layer(8, sg.Activation.FOO52, sg.Init.HE),
                              sg.Layer(1, sg.Activation.SIGMOID, sg.Init.XAVIER)]) as net:
    net.train(x, y, epochs=10000, optimizer=sg.Optimizer.ADAMW,
              learning_rate=0.01, weight_decay=1e-4)
    for row, target, out in zip(x, y, net.forward(x)):
        print(f"in={row.tolist()}  target={target[0]:.0f}  output={out[0]:.4f}")
