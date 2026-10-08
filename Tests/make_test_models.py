# SPDX-License-Identifier: MIT
"""Writes the PyTorch models of the `onnx` test group into Tests/Data, with sample inputs and
PyTorch's outputs for them:

    python Tests/make_test_models.py       # needs torch, onnxscript (the newer exporter) and safetensors

ONNX models exported from PyTorch: NAME.onnx. State dicts: torch_cnn.pt (torch.save) and
torch_cnn.safetensors, of the network the test builds by hand. For each, NAME.bin holds "SPGT", then
as 32-bit little-endian integers the sample count, the floats per input and per output, then the
inputs (channels last, as Spingalett reads them) and the expected outputs as floats. The models are
small, seeded and fixed, so the files only change when this script does.
"""
import io
import os
import struct

import numpy as np
import torch
from torch import nn

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data")


class Residual(nn.Module):
    """A stem, a residual block, a strided block with a projection, branches concatenated, pooling."""

    def __init__(self, c=6):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, c, 3, padding=1), nn.BatchNorm2d(c), nn.ReLU())
        self.c1, self.b1 = nn.Conv2d(c, c, 3, padding=1, bias=False), nn.BatchNorm2d(c)
        self.c2, self.b2 = nn.Conv2d(c, c, 3, padding=1, bias=False), nn.BatchNorm2d(c)
        self.down, self.proj = nn.Conv2d(c, 2 * c, 3, stride=2, padding=1), nn.Conv2d(c, 2 * c, 1, stride=2)
        self.branch = nn.Conv2d(2 * c, 4, 1)
        self.head = nn.Linear(2 * c + 4, 5)

    def forward(self, x):
        x = self.stem(x)
        x = torch.relu(self.b2(self.c2(torch.relu(self.b1(self.c1(x))))) + x)
        y = torch.relu(self.down(x) + self.proj(x))
        z = torch.cat([y, torch.tanh(self.branch(y))], 1)
        return torch.softmax(self.head(torch.flatten(nn.functional.adaptive_avg_pool2d(z, 1), 1)), 1)


class Affine(nn.Module):
    """x @ W + b written out (MatMul and Add), then a sigmoid."""

    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.randn(7, 4) * 0.5)
        self.b = nn.Parameter(torch.randn(4) * 0.1)

    def forward(self, x):
        return torch.sigmoid(x @ self.w + self.b)


class Upsampled(nn.Module):
    """An operator the importer does not have (Resize), for the error message."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 2, 1)

    def forward(self, x):
        return nn.functional.interpolate(self.conv(x), scale_factor=2.0, mode="nearest")


def models():
    torch.manual_seed(7)
    yield "onnx_mlp", nn.Sequential(nn.Linear(10, 16), nn.Tanh(), nn.Linear(16, 8), nn.LeakyReLU(0.01),
                                    nn.Linear(8, 3), nn.Softmax(1)), (10,), dict(dynamo=False)
    yield "onnx_cnn", nn.Sequential(nn.Conv2d(3, 6, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                                    nn.Conv2d(6, 8, 3, groups=2), nn.Sigmoid(), nn.AvgPool2d(2, stride=1),
                                    nn.Flatten(), nn.Linear(8 * 2 * 2, 4)), (3, 10, 10), dict(dynamo=True)
    yield "onnx_resnet", Residual(), (3, 8, 8), dict(dynamo=False, do_constant_folding=False)
    yield "onnx_resnet_dynamo", Residual(), (3, 8, 8), dict(dynamo=True)
    yield "onnx_matmul", Affine(), (7,), dict(dynamo=False)
    yield "onnx_unsupported", Upsampled(), (2, 4, 4), dict(dynamo=False)


class Weights(nn.Module):
    """The network of the state dict tests: modules numbered past 10, a convolution without bias."""

    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(), nn.MaxPool2d(2),
                                  nn.Conv2d(8, 8, 3, padding=1, groups=4, bias=False), nn.BatchNorm2d(8), nn.ReLU(),
                                  *[nn.Identity() for _ in range(5)], nn.Flatten(), nn.Linear(8 * 4 * 4, 16),
                                  nn.Tanh(), nn.Linear(16, 4))

    def forward(self, x):
        return self.body(x)


def write_expected(name, x, y):
    inputs = x.numpy().transpose(0, 2, 3, 1) if x.dim() == 4 else x.numpy()
    outputs = y.numpy().reshape(x.shape[0], -1)
    with open(os.path.join(OUT, name + ".bin"), "wb") as f:
        f.write(b"SPGT" + struct.pack("<3I", x.shape[0], inputs[0].size, outputs.shape[1]))
        f.write(np.ascontiguousarray(inputs, dtype="<f4").tobytes())
        f.write(np.ascontiguousarray(outputs, dtype="<f4").tobytes())


def randomize_statistics(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):           # statistics other than the initial ones
            m.running_mean.uniform_(-0.5, 0.5)
            m.running_var.uniform_(0.5, 2.0)
            m.weight.data.uniform_(0.5, 1.5)
            m.bias.data.uniform_(-0.2, 0.2)


def main():
    for name, model, shape, options in models():
        model.eval()
        randomize_statistics(model)
        x = torch.rand(4, *shape)
        with torch.no_grad():
            y = model(x)
        buffer = io.BytesIO()
        torch.onnx.export(model, (x[:1],), buffer, **options)
        with open(os.path.join(OUT, name + ".onnx"), "wb") as f:
            f.write(buffer.getvalue())
        write_expected(name, x, y)
        print(f"{name}: {len(buffer.getvalue())} bytes")

    from safetensors.torch import save_file
    torch.manual_seed(11)
    model = Weights().eval()
    randomize_statistics(model)
    x = torch.rand(3, 3, 8, 8)
    with torch.no_grad():
        y = model(x)
    torch.save(model.state_dict(), os.path.join(OUT, "torch_cnn.pt"))
    save_file({k: v.contiguous() for k, v in model.state_dict().items()}, os.path.join(OUT, "torch_cnn.safetensors"))
    write_expected("torch_cnn", x, y)
    print("torch_cnn: .pt and .safetensors")


if __name__ == "__main__":
    main()
