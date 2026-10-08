# SPDX-License-Identifier: MIT
"""PyTorch counterpart of Examples/Benchmark.c: same networks, data shapes, optimizers and schedules.

    python Examples/benchmark_pytorch.py [threads]

784-512-1000-10 MLP (ReLU, softmax + cross-entropy on soft targets, Adam, lr 1e-3), 20,000
synthetic samples: full batch (5 epochs), mini-batches of 64 (1 epoch) and inference. Then the
convolutional network of Examples/MNIST_CNN.c (channels first, as PyTorch prefers) on 10,000
synthetic 28 x 28 images: one epoch of mini-batches of 128 with AdamW, and inference in batches of
1,000; and the same network with batch normalization after each convolution and the hidden dense
layer. Then ResNet-20 (Examples/CIFAR10.c resnet20) on 4,096 synthetic 32 x 32 images: one epoch
of mini-batches of 128 with SGD and momentum, and inference in batches of 1,000. Each measurement
runs on a fresh model after one untimed warm-up step, so lazy initialization inside PyTorch is not
counted.
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


def make_cnn(normalized=False):
    def norm(n, two_d=True):
        return [nn.BatchNorm2d(n) if two_d else nn.BatchNorm1d(n)] if normalized else []
    return nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), *norm(32), nn.ReLU(), nn.MaxPool2d(2),
                         nn.Conv2d(32, 64, 3, padding=1), *norm(64), nn.ReLU(), nn.MaxPool2d(2), nn.Flatten(),
                         nn.Linear(3136, 128), *norm(128, False), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 10))


def cnn_throughput(x, y, normalized=False):
    model = make_cnn(normalized)
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


RESNET_SAMPLES = 4096


class Block(nn.Module):
    """A residual block: two normalized 3 x 3 convolutions added to the input (a 1 x 1 projection
    where the shape changes), then ReLU."""

    def __init__(self, cin, cout, stride):
        super().__init__()
        self.c1, self.b1 = nn.Conv2d(cin, cout, 3, stride, 1), nn.BatchNorm2d(cout)
        self.c2, self.b2 = nn.Conv2d(cout, cout, 3, 1, 1), nn.BatchNorm2d(cout)
        self.shortcut = (nn.Sequential(nn.Conv2d(cin, cout, 1, stride), nn.BatchNorm2d(cout))
                         if stride != 1 or cin != cout else nn.Identity())

    def forward(self, x):
        return torch.relu(self.b2(self.c2(torch.relu(self.b1(self.c1(x))))) + self.shortcut(x))


def make_resnet20():
    layers, cin = [nn.Conv2d(3, 16, 3, 1, 1), nn.BatchNorm2d(16), nn.ReLU()], 16
    for stage in range(3):
        for b in range(3):
            layers.append(Block(cin, 16 << stage, 2 if stage > 0 and b == 0 else 1))
            cin = 16 << stage
    return nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(cin, 10))


def resnet_throughput(x, y):
    model = make_resnet20()
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    loss_fn = nn.CrossEntropyLoss()

    def step(xb, yb):
        opt.zero_grad(set_to_none=True)
        loss_fn(model(xb), yb).backward()
        opt.step()

    step(x[:CNN_BATCH], y[:CNN_BATCH])       # warm-up
    start = time.perf_counter()
    perm = torch.randperm(RESNET_SAMPLES)
    for i in range(0, RESNET_SAMPLES, CNN_BATCH):
        idx = perm[i:i + CNN_BATCH]
        step(x[idx], y[idx])
    train = RESNET_SAMPLES / (time.perf_counter() - start)
    model.eval()
    with torch.inference_mode():
        torch.softmax(model(x[:64]), dim=1)  # warm-up
        start = time.perf_counter()
        for i in range(0, RESNET_SAMPLES, 1000):
            torch.softmax(model(x[i:i + 1000]), dim=1)
        infer = RESNET_SAMPLES / (time.perf_counter() - start)
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
    for normalized in (False, True):
        print(f"\nconvolutional network (Examples/MNIST_CNN.c){' with batch normalization' if normalized else ''}, "
              f"{CNN_SAMPLES} images")
        print(f"{'samples/s':<16} {'training':>14} {'inference':>14}")
        train, infer = cnn_throughput(images, labels, normalized)
        print(f"{'PyTorch':<16} {train:14.0f} {infer:14.0f}")

    print(f"\nResNet-20 (Examples/CIFAR10.c resnet20), {RESNET_SAMPLES} images")
    print(f"{'samples/s':<16} {'training':>14} {'inference':>14}")
    train, infer = resnet_throughput(torch.rand(RESNET_SAMPLES, 3, 32, 32), torch.randint(0, 10, (RESNET_SAMPLES,)))
    print(f"{'PyTorch':<16} {train:14.0f} {infer:14.0f}")


if __name__ == "__main__":
    main()
