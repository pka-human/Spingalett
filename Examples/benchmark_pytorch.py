# SPDX-License-Identifier: MIT
"""PyTorch counterpart of Examples/Benchmark.c: same network, data shape, optimizer and schedule.

    python Examples/benchmark_pytorch.py [threads]

784-512-1000-10 MLP (ReLU, softmax + cross-entropy on soft targets, Adam, lr 1e-3), 20,000
synthetic samples: full batch (5 epochs), mini-batches of 64 (1 epoch) and inference. Each
measurement runs on a fresh model after one untimed warm-up step, so lazy initialization inside
PyTorch is not counted.
"""
import sys
import time

import torch
from torch import nn

SAMPLES, EPOCHS, MINI_BATCH = 20000, 5, 64


def make_model():
    return nn.Sequential(nn.Linear(784, 512), nn.ReLU(), nn.Linear(512, 1000), nn.ReLU(), nn.Linear(1000, 10))


def train_throughput(x, y, batch, epochs):
    model = make_model()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()          # applies softmax; accepts probability targets

    def step(xb, yb):
        opt.zero_grad(set_to_none=True)
        loss_fn(model(xb), yb).backward()
        opt.step()

    step(x[:batch], y[:batch])               # warm-up
    start = time.perf_counter()
    for _ in range(epochs):
        perm = torch.randperm(SAMPLES) if batch < SAMPLES else None
        for i in range(0, SAMPLES, batch):
            idx = perm[i:i + batch] if perm is not None else slice(None)
            step(x[idx], y[idx])
    return SAMPLES * epochs / (time.perf_counter() - start)


def inference_throughput(x):
    model = make_model().eval()
    with torch.inference_mode():
        torch.softmax(model(x[:64]), dim=1)  # warm-up
        start = time.perf_counter()
        torch.softmax(model(x), dim=1)
        return SAMPLES / (time.perf_counter() - start)


def main():
    if len(sys.argv) > 1:
        torch.set_num_threads(int(sys.argv[1]))
    torch.manual_seed(42)
    x = torch.rand(SAMPLES, 784)
    y = torch.rand(SAMPLES, 10)
    y /= y.sum(dim=1, keepdim=True)

    print(f"PyTorch {torch.__version__}, threads: {torch.get_num_threads()}\n")
    print(f"{'samples/s':<16} {'full batch':>14} {'mini-batch 64':>14} {'inference':>14}")
    full = train_throughput(x, y, SAMPLES, EPOCHS)
    mini = train_throughput(x, y, MINI_BATCH, 1)
    infer = inference_throughput(x)
    print(f"{'PyTorch':<16} {full:14.0f} {mini:14.0f} {infer:14.0f}")


if __name__ == "__main__":
    main()
