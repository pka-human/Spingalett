# SPDX-License-Identifier: MIT
"""Writes the PyTorch models of the `onnx` test group into Tests/Data, with sample inputs and
PyTorch's outputs for them:

    python Tests/make_test_models.py       # needs torch, onnx, onnxscript (the newer exporter) and safetensors

ONNX models exported from PyTorch: NAME.onnx (onnx_external.onnx: onnx_cnn's weights in
onnx_external.data). State dicts: torch_cnn.pt (torch.save; torch_cnn_strided.pt with tensors that
are views) and torch_cnn.safetensors, and torch_transposed.pt, of the networks the tests build by
hand. For each, NAME.bin holds "SPGT", then as 32-bit little-endian integers the sample count, the
floats per input and per output, then the inputs and the expected outputs as floats (maps channels
last, as Spingalett reads and writes them). The models are small, seeded and fixed, so the files only
change when this script does.
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
    """A Resize the importer does not have (corners aligned), for the error message."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 2, 1)

    def forward(self, x):
        return nn.functional.interpolate(self.conv(x), scale_factor=2.0, mode="bilinear", align_corners=True)


class UNet(nn.Module):
    """Down by pooling, up by transposed convolutions (one grouped, with padding and output padding)
    concatenated with the maps on the way down, nearest and bilinear Resize; maps out."""

    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv2d(3, 6, 3, padding=1), nn.ReLU())
        self.down = nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(6, 8, 3, padding=1), nn.ReLU())
        self.up = nn.ConvTranspose2d(8, 6, 2, stride=2)
        self.up2 = nn.ConvTranspose2d(8, 4, 3, stride=2, padding=1, output_padding=1, groups=2)
        self.mix = nn.Conv2d(16, 4, 3, padding=1)
        self.head = nn.Conv2d(4, 2, 1)

    def forward(self, x):
        e = self.enc(x)
        d = self.down(e)
        z = torch.cat([e, torch.relu(self.up(d)), torch.tanh(self.up2(d))], 1)
        y = self.mix(z)
        near = nn.functional.interpolate(nn.functional.max_pool2d(y, 2), scale_factor=2, mode="nearest")
        lin = nn.functional.interpolate(nn.functional.avg_pool2d(y, 2), scale_factor=2, mode="bilinear",
                                        align_corners=False)
        return torch.sigmoid(self.head(near + lin))


class Normed(nn.Module):
    """Layer normalization over a map's channels (between permutations, as LayerNorm2d does it) and
    over a vector."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)
        self.ln2d = nn.LayerNorm(8)
        self.fc = nn.Linear(8 * 6 * 6, 16)
        self.ln = nn.LayerNorm(16)
        self.out = nn.Linear(16, 4)

    def forward(self, x):
        y = self.ln2d(self.conv(x).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        y = torch.tanh(self.ln(self.fc(torch.flatten(torch.relu(y), 1))))
        return torch.softmax(self.out(y), 1)


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
    yield "onnx_unet", UNet(), (3, 8, 8), dict(dynamo=False)
    yield "onnx_unet_dynamo", UNet(), (3, 8, 8), dict(dynamo=True)
    yield "onnx_layernorm", Normed(), (3, 6, 6), dict(dynamo=False, opset_version=17)
    yield "onnx_layernorm_dynamo", Normed(), (3, 6, 6), dict(dynamo=True)


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


class Transposed(nn.Module):
    """The network of the second state dict test: a grouped transposed convolution with padding and
    output padding, layer normalization over the channels."""

    def __init__(self):
        super().__init__()
        self.down = nn.Conv2d(3, 8, 3, stride=2, padding=1)
        self.up = nn.ConvTranspose2d(8, 6, 3, stride=2, padding=1, output_padding=1, groups=2)
        self.norm = nn.LayerNorm(6)
        self.head = nn.Linear(6, 4)

    def forward(self, x):
        y = self.up(torch.relu(self.down(x)))
        y = torch.tanh(self.norm(y.permute(0, 2, 3, 1)))
        return self.head(y.mean((1, 2)))


def write_expected(name, x, y):
    inputs = x.numpy().transpose(0, 2, 3, 1) if x.dim() == 4 else x.numpy()
    outputs = (y.numpy().transpose(0, 2, 3, 1) if y.dim() == 4 else y.numpy()).reshape(x.shape[0], -1)
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
    # tensors that are views: a transposed dense weight, filters with permuted strides
    state = model.state_dict()
    state["body.13.weight"] = state["body.13.weight"].t().contiguous().t()
    state["body.0.weight"] = state["body.0.weight"].permute(0, 2, 3, 1).contiguous().permute(0, 3, 1, 2)
    torch.save(state, os.path.join(OUT, "torch_cnn_strided.pt"))
    write_expected("torch_cnn", x, y)
    print("torch_cnn: .pt, .safetensors and strided .pt")

    torch.manual_seed(13)
    model = Transposed().eval()
    for m in model.modules():
        if isinstance(m, nn.LayerNorm):             # parameters other than the initial ones
            m.weight.data.uniform_(0.5, 1.5)
            m.bias.data.uniform_(-0.2, 0.2)
    x = torch.rand(3, 3, 8, 8)
    with torch.no_grad():
        y = model(x)
    torch.save(model.state_dict(), os.path.join(OUT, "torch_transposed.pt"))
    write_expected("torch_transposed", x, y)
    print("torch_transposed: .pt")

    # the CNN with its weights in a file of their own (external data), and one naming a file outside
    # its folder
    import onnx
    m = onnx.load(os.path.join(OUT, "onnx_cnn.onnx"))
    onnx.save_model(m, os.path.join(OUT, "onnx_external.onnx"), save_as_external_data=True,
                    all_tensors_to_one_file=True, location="onnx_external.data", size_threshold=0)
    for t in m.graph.initializer:
        for e in t.external_data:
            if e.key == "location":
                e.value = "../onnx_external.data"
    onnx.save_model(m, os.path.join(OUT, "onnx_escape.onnx"))
    print("onnx_external: .onnx and .data; onnx_escape")


if __name__ == "__main__":
    main()
