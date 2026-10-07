# SPDX-License-Identifier: MIT
"""PyTorch counterpart of Examples/Benchmark.c: same networks, data shapes, optimizers and schedules.

    python Examples/benchmark_pytorch.py [threads]

784-512-1000-10 MLP (ReLU, softmax + cross-entropy on soft targets, Adam, lr 1e-3), 20,000
synthetic samples: full batch (5 epochs), mini-batches of 64 (1 epoch) and inference. Then the
convolutional network of Examples/MNIST_CNN.c (channels first, as PyTorch prefers) on 10,000
synthetic 28 x 28 images: one epoch of mini-batches of 128 with AdamW, and inference in batches of
1,000. Each measurement runs on a fresh model after one untimed warm-up step, so lazy
initialization inside PyTorch is not counted.
"""
import sys
import time

import torch
from torch import nn

SAMPLES, EPOCHS, MINI_BATCH = 20000, 5, 64
CNN_SAMPLES, CNN_BATCH = 10000, 128


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


def make_cnn():
    return nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                         nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Flatten(),
                         nn.Linear(3136, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 10))


def cnn_throughput(x, y):
    model = make_cnn()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    def step(xb, yb):
        opt.zero_grad(set_to_none=True)
        loss_fn(model(xb), yb).backward()
        opt.step()

    step(x[:CNN_BATCH], y[:CNN_BATCH])       # warm-up
    start = time.perf_counter()
    perm = torch.randperm(CNN_SAMPLES)
    for i in range(0, CNN_SAMPLES, CNN_BATCH):
        idx = perm[i:i + CNN_BATCH]
        step(x[idx], y[idx])
    train = CNN_SAMPLES / (time.perf_counter() - start)
    model.eval()
    with torch.inference_mode():
        torch.softmax(model(x[:64]), dim=1)  # warm-up
        start = time.perf_counter()
        for i in range(0, CNN_SAMPLES, 1000):
            torch.softmax(model(x[i:i + 1000]), dim=1)
        infer = CNN_SAMPLES / (time.perf_counter() - start)
    return train, infer


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

    images = torch.rand(CNN_SAMPLES, 1, 28, 28)
    labels = torch.randint(0, 10, (CNN_SAMPLES,))
    print(f"\nconvolutional network (Examples/MNIST_CNN.c), {CNN_SAMPLES} images")
    print(f"{'samples/s':<16} {'training':>14} {'inference':>14}")
    train, infer = cnn_throughput(images, labels)
    print(f"{'PyTorch':<16} {train:14.0f} {infer:14.0f}")


if __name__ == "__main__":
    main()
