# SPDX-License-Identifier: MIT
"""PyTorch counterpart of Examples/Benchmark.c: same networks, data shapes, optimizers and schedules.

    python Examples/benchmark_pytorch.py [threads] [--cuda | --cuda-fp32 | --cuda-bf16] [--host-data]

784-512-1000-10 MLP (ReLU, softmax + cross-entropy on soft targets, Adam, lr 1e-3), 20,000
synthetic samples: full batch (5 epochs), mini-batches of 64 (1 epoch) and inference. Then the
convolutional network of Examples/MNIST_CNN.c (channels first, as PyTorch prefers) on 10,000
synthetic 28 x 28 images: one epoch of mini-batches of 128 with AdamW, and inference in batches of
1,000; and the same network with batch normalization after each convolution and the hidden dense
layer. Then ResNet-20 (Examples/CIFAR10.c resnet20) on 4,096 synthetic 32 x 32 images: one epoch
of mini-batches of 128 with SGD and momentum, and inference in batches of 1,000. Then the U-Net of
Examples/Segmentation.c on 1,024 synthetic 64 x 64 images with three sigmoid outputs a pixel (binary
cross-entropy): one epoch of mini-batches of 32 with AdamW, and inference in batches of 256. Each measurement
runs on a fresh model after one untimed warm-up step, so lazy initialization inside PyTorch is not
counted.

--cuda runs everything on the GPU (data in device memory, times taken after synchronizing), with
PyTorch's defaults: convolutions in cuDNN may use TF32 tensor cores, which round the inputs of
products to 10 bits of mantissa. --cuda-fp32 turns TF32 off, so that products are in single
precision throughout, as Spingalett's are. --cuda-bf16 runs the forward passes under autocast to
bfloat16, the counterpart of spingalett_set_gpu_precision(PRECISION_BFLOAT16).

--host-data keeps the data in the host's memory (pinned), as Spingalett's functions take it: each
batch is copied to the GPU as it is used (mini-batches gathered on the host first, the way a plain
training loop does), and the outputs of inference copied back.
"""
import contextlib
import sys
import time

import torch
from torch import nn

SAMPLES, EPOCHS, MINI_BATCH = 20000, 5, 64
CNN_SAMPLES, CNN_BATCH = 10000, 128
DEVICE = "cpu"
BF16 = False
HOST = False


def to_dev(t):
    """A batch on the device it trains on: copied there with --host-data."""
    return t.to(DEVICE, non_blocking=True) if HOST else t


def back(t):
    """Outputs where the caller reads them: the host with --host-data."""
    return t.cpu() if HOST else t


def perm_of(n):
    return torch.randperm(n, device="cpu" if HOST else DEVICE)


def amp():
    """Autocast to bfloat16 with --cuda-bf16, nothing otherwise."""
    return torch.autocast("cuda", dtype=torch.bfloat16) if BF16 else contextlib.nullcontext()


def sync():
    if DEVICE != "cpu":
        torch.cuda.synchronize()


def clock():
    sync()
    return time.perf_counter()


def make_model():
    return nn.Sequential(nn.Linear(784, 512), nn.ReLU(), nn.Linear(512, 1000), nn.ReLU(), nn.Linear(1000, 10))


def train_throughput(x, y, batch, epochs):
    model = make_model().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()          # applies softmax; accepts probability targets

    def step(xb, yb):
        opt.zero_grad(set_to_none=True)
        with amp():
            loss = loss_fn(model(xb), yb)
        loss.backward()
        opt.step()

    step(to_dev(x[:batch]), to_dev(y[:batch]))  # warm-up
    start = clock()
    for _ in range(epochs):
        perm = perm_of(SAMPLES) if batch < SAMPLES else None
        for i in range(0, SAMPLES, batch):
            idx = perm[i:i + batch] if perm is not None else slice(None)
            step(to_dev(x[idx]), to_dev(y[idx]))
    return SAMPLES * epochs / (clock() - start)


def inference_throughput(x):
    model = make_model().to(DEVICE).eval()
    with torch.inference_mode(), amp():
        torch.softmax(model(to_dev(x[:64])), dim=1)  # warm-up
        start = clock()
        back(torch.softmax(model(to_dev(x)), dim=1))
        return SAMPLES / (clock() - start)


def make_cnn(normalized=False):
    def norm(n, two_d=True):
        return [nn.BatchNorm2d(n) if two_d else nn.BatchNorm1d(n)] if normalized else []
    return nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), *norm(32), nn.ReLU(), nn.MaxPool2d(2),
                         nn.Conv2d(32, 64, 3, padding=1), *norm(64), nn.ReLU(), nn.MaxPool2d(2), nn.Flatten(),
                         nn.Linear(3136, 128), *norm(128, False), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 10))


def cnn_throughput(x, y, normalized=False):
    model = make_cnn(normalized).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    def step(xb, yb):
        opt.zero_grad(set_to_none=True)
        with amp():
            loss = loss_fn(model(xb), yb)
        loss.backward()
        opt.step()

    step(to_dev(x[:CNN_BATCH]), to_dev(y[:CNN_BATCH]))  # warm-up
    start = clock()
    perm = perm_of(CNN_SAMPLES)
    for i in range(0, CNN_SAMPLES, CNN_BATCH):
        idx = perm[i:i + CNN_BATCH]
        step(to_dev(x[idx]), to_dev(y[idx]))
    train = CNN_SAMPLES / (clock() - start)
    model.eval()
    with torch.inference_mode(), amp():
        torch.softmax(model(to_dev(x[:64])), dim=1)  # warm-up
        start = clock()
        for i in range(0, CNN_SAMPLES, 1000):
            back(torch.softmax(model(to_dev(x[i:i + 1000])), dim=1))
        infer = CNN_SAMPLES / (clock() - start)
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
    model = make_resnet20().to(DEVICE)
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    loss_fn = nn.CrossEntropyLoss()

    def step(xb, yb):
        opt.zero_grad(set_to_none=True)
        with amp():
            loss = loss_fn(model(xb), yb)
        loss.backward()
        opt.step()

    step(to_dev(x[:CNN_BATCH]), to_dev(y[:CNN_BATCH]))  # warm-up
    start = clock()
    perm = perm_of(RESNET_SAMPLES)
    for i in range(0, RESNET_SAMPLES, CNN_BATCH):
        idx = perm[i:i + CNN_BATCH]
        step(to_dev(x[idx]), to_dev(y[idx]))
    train = RESNET_SAMPLES / (clock() - start)
    model.eval()
    with torch.inference_mode(), amp():
        torch.softmax(model(to_dev(x[:64])), dim=1)  # warm-up
        start = clock()
        for i in range(0, RESNET_SAMPLES, 1000):
            back(torch.softmax(model(to_dev(x[i:i + 1000])), dim=1))
        infer = RESNET_SAMPLES / (clock() - start)
    return train, infer


UNET_SAMPLES, UNET_BATCH = 1024, 32


class UNet(nn.Module):
    """Examples/Segmentation.c's U-Net: two normalized 3 x 3 convolutions a level, transposed
    convolutions up, the maps of the way down concatenated."""

    @staticmethod
    def double(cin, cout):
        return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(),
                             nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU())

    def __init__(self):
        super().__init__()
        self.e1, self.e2, self.mid = self.double(3, 16), self.double(16, 32), self.double(32, 64)
        self.u2, self.d2 = nn.ConvTranspose2d(64, 32, 2, 2), self.double(64, 32)
        self.u1, self.d1 = nn.ConvTranspose2d(32, 16, 2, 2), self.double(32, 16)
        self.out = nn.Conv2d(16, 3, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(nn.functional.max_pool2d(e1, 2))
        m = self.mid(nn.functional.max_pool2d(e2, 2))
        d2 = self.d2(torch.cat([e2, torch.relu(self.u2(m))], dim=1))
        d1 = self.d1(torch.cat([e1, torch.relu(self.u1(d2))], dim=1))
        return self.out(d1)                 # logits: the loss applies the sigmoid


def unet_throughput(x, y):
    model = UNet().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    def step(xb, yb):
        opt.zero_grad(set_to_none=True)
        with amp():
            loss = loss_fn(model(xb), yb)
        loss.backward()
        opt.step()

    step(to_dev(x[:UNET_BATCH]), to_dev(y[:UNET_BATCH]))  # warm-up
    start = clock()
    perm = perm_of(UNET_SAMPLES)
    for i in range(0, UNET_SAMPLES, UNET_BATCH):
        idx = perm[i:i + UNET_BATCH]
        step(to_dev(x[idx]), to_dev(y[idx]))
    train = UNET_SAMPLES / (clock() - start)
    model.eval()
    with torch.inference_mode(), amp():
        torch.sigmoid(model(to_dev(x[:32])))  # warm-up
        start = clock()
        for i in range(0, UNET_SAMPLES, 256):
            back(torch.sigmoid(model(to_dev(x[i:i + 256]))))
        infer = UNET_SAMPLES / (clock() - start)
    return train, infer


def data(t):
    """Training data where the run keeps it: the device, or (pinned) the host with --host-data."""
    return t.pin_memory() if HOST else t.to(DEVICE)


def main():
    global DEVICE, BF16, HOST
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if args:
        torch.set_num_threads(int(args[0]))
    if "--cuda" in sys.argv or "--cuda-fp32" in sys.argv or "--cuda-bf16" in sys.argv:
        DEVICE = "cuda"
        BF16 = "--cuda-bf16" in sys.argv
        torch.backends.cudnn.allow_tf32 = "--cuda-fp32" not in sys.argv
        torch.backends.cuda.matmul.allow_tf32 = "--cuda-fp32" not in sys.argv and torch.backends.cuda.matmul.allow_tf32
        HOST = "--host-data" in sys.argv
    torch.manual_seed(42)
    x = torch.rand(SAMPLES, 784)
    y = torch.rand(SAMPLES, 10)
    y /= y.sum(dim=1, keepdim=True)
    x, y = data(x), data(y)

    where = f"threads: {torch.get_num_threads()}"
    if DEVICE != "cpu":
        where = f"GPU: {torch.cuda.get_device_name(0)}, TF32 convolutions {'on' if torch.backends.cudnn.allow_tf32 else 'off'}"
        if BF16:
            where += ", autocast to bfloat16"
        if HOST:
            where += ", data in host memory"
    print(f"PyTorch {torch.__version__}, {where}\n")
    print(f"{'samples/s':<16} {'full batch':>14} {'mini-batch 64':>14} {'inference':>14}")
    full = train_throughput(x, y, SAMPLES, EPOCHS)
    mini = train_throughput(x, y, MINI_BATCH, 1)
    infer = inference_throughput(x)
    print(f"{'PyTorch':<16} {full:14.0f} {mini:14.0f} {infer:14.0f}")

    images = data(torch.rand(CNN_SAMPLES, 1, 28, 28))
    labels = data(torch.randint(0, 10, (CNN_SAMPLES,)))
    for normalized in (False, True):
        print(f"\nconvolutional network (Examples/MNIST_CNN.c){' with batch normalization' if normalized else ''}, "
              f"{CNN_SAMPLES} images")
        print(f"{'samples/s':<16} {'training':>14} {'inference':>14}")
        train, infer = cnn_throughput(images, labels, normalized)
        print(f"{'PyTorch':<16} {train:14.0f} {infer:14.0f}")

    print(f"\nResNet-20 (Examples/CIFAR10.c resnet20), {RESNET_SAMPLES} images")
    print(f"{'samples/s':<16} {'training':>14} {'inference':>14}")
    train, infer = resnet_throughput(data(torch.rand(RESNET_SAMPLES, 3, 32, 32)),
                                     data(torch.randint(0, 10, (RESNET_SAMPLES,))))
    print(f"{'PyTorch':<16} {train:14.0f} {infer:14.0f}")

    print(f"\nU-Net (Examples/Segmentation.c), {UNET_SAMPLES} images of 64 x 64")
    print(f"{'samples/s':<16} {'training':>14} {'inference':>14}")
    train, infer = unet_throughput(data(torch.rand(UNET_SAMPLES, 3, 64, 64)),
                                   data(torch.randint(0, 2, (UNET_SAMPLES, 3, 64, 64)).float()))
    print(f"{'PyTorch':<16} {train:14.0f} {infer:14.0f}")


if __name__ == "__main__":
    main()
