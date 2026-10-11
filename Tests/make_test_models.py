# SPDX-License-Identifier: MIT
"""Writes the PyTorch models of the `onnx` test group into Tests/Data, with sample inputs and
PyTorch's outputs for them:

    python Tests/make_test_models.py       # needs torch, onnx, onnxscript (the newer exporter) and safetensors

ONNX models exported from PyTorch: NAME.onnx (onnx_external.onnx: onnx_cnn's weights in
onnx_external.data). State dicts: torch_cnn.pt (torch.save; torch_cnn_strided.pt with tensors that
are views) and torch_cnn.safetensors, and torch_transposed.pt, of the networks the tests build by
hand. For each, NAME.bin holds "SPGT", then as 32-bit little-endian integers the sample count, the
floats per input and per output, then the inputs and the expected outputs as floats (maps channels
last, as Spingalett reads and writes them; token indices as floats). The models are small, seeded and
fixed, so the files only change when this script does.

Transformers (onnx_gpt*, onnx_llama*, onnx_encoder*) are exported by both exporters and at the operator
sets that change how they are written (14: layer normalization and GELU written out; 20: their
operators; 23: RMSNormalization and Attention); the newer exporter's debugging metadata (stack traces,
shapes of intermediate values) is removed from their files, which leaves the graphs as written and the
files a fifth of the size. onnx_ort_fused is a graph of ONNX Runtime's fused operators, built by hand.
onnx_tokens_softmax and onnx_gram hold what the importer refuses.
"""
import io
import math
import os
import struct

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

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


class GptAttention(nn.Module):
    """nanoGPT's causal self-attention: queries, keys and values of one product, split into heads, the
    scores scaled and masked by a lower-triangular buffer, a softmax, the output projection."""

    def __init__(self, d, heads, context):
        super().__init__()
        self.qkv, self.proj, self.heads = nn.Linear(d, 3 * d), nn.Linear(d, d), heads
        self.register_buffer("mask", torch.tril(torch.ones(context, context)).view(1, 1, context, context))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q, k, v = (t.view(B, T, self.heads, C // self.heads).transpose(1, 2) for t in (q, k, v))
        scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        scores = scores.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        y = torch.softmax(scores, dim=-1) @ v
        return self.proj(y.transpose(1, 2).contiguous().view(B, T, C))


class GptBlock(nn.Module):
    def __init__(self, d, heads, context, gelu):
        super().__init__()
        self.ln1, self.attn, self.ln2 = nn.LayerNorm(d), GptAttention(d, heads, context), nn.LayerNorm(d)
        self.fc, self.out, self.gelu = nn.Linear(d, 4 * d), nn.Linear(4 * d, d), gelu

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.out(F.gelu(self.fc(self.ln2(x)), approximate=self.gelu))


class Gpt(nn.Module):
    """A GPT-2-like language model: token embeddings and learned positions, blocks of layer
    normalization, causal attention and a GELU MLP (GPT-2's tanh form in the first block, the erf form
    in the second), a final normalization and a head tied to the token table."""

    def __init__(self, vocabulary=11, d=8, heads=2, context=6):
        super().__init__()
        self.tokens, self.positions = nn.Embedding(vocabulary, d), nn.Embedding(context, d)
        self.blocks = nn.ModuleList([GptBlock(d, heads, context, g) for g in ("tanh", "none")])
        self.norm, self.head = nn.LayerNorm(d), nn.Linear(d, vocabulary, bias=False)
        self.head.weight = self.tokens.weight

    def forward(self, idx):
        x = self.tokens(idx) + self.positions(torch.arange(idx.shape[1], device=idx.device))
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))


class RMSNorm(nn.Module):
    """LLaMA's RMS normalization, written out (the TorchScript exporter has no nn.RMSNorm)."""

    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight, self.eps = nn.Parameter(torch.ones(d)), eps

    def forward(self, x):
        return self.weight * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps))


def rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


class LlamaAttention(nn.Module):
    """Hugging Face's LLaMA attention: queries, keys and values of their own products, rotary
    embeddings (rotate_half, cosine and sine tables), grouped-query heads (the keys and values repeated
    by repeat_kv, or by the attention itself), causal scaled dot-product attention."""

    def __init__(self, d, heads, kv_heads, native):
        super().__init__()
        self.heads, self.kv_heads, self.head, self.native = heads, kv_heads, d // heads, native
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, kv_heads * self.head, bias=False)
        self.v = nn.Linear(d, kv_heads * self.head, bias=False)
        self.o = nn.Linear(d, d, bias=False)

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        q = self.q(x).view(B, T, self.heads, self.head).transpose(1, 2)
        k = self.k(x).view(B, T, self.kv_heads, self.head).transpose(1, 2)
        v = self.v(x).view(B, T, self.kv_heads, self.head).transpose(1, 2)
        q, k = q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin
        if self.native:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        else:
            groups = self.heads // self.kv_heads
            k = k[:, :, None].expand(B, self.kv_heads, groups, T, self.head).reshape(B, self.heads, T, self.head)
            v = v[:, :, None].expand(B, self.kv_heads, groups, T, self.head).reshape(B, self.heads, T, self.head)
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o(y.transpose(1, 2).reshape(B, T, -1))


class LlamaBlock(nn.Module):
    def __init__(self, d, heads, kv_heads, hidden, native):
        super().__init__()
        norm = (lambda: nn.RMSNorm(d, eps=1e-6)) if native else (lambda: RMSNorm(d))
        self.n1, self.attn, self.n2 = norm(), LlamaAttention(d, heads, kv_heads, native), norm()
        self.gate, self.up = nn.Linear(d, hidden, bias=False), nn.Linear(d, hidden, bias=False)
        self.down = nn.Linear(hidden, d, bias=False)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.n1(x), cos, sin)
        h = self.n2(x)
        return x + self.down(F.silu(self.gate(h)) * self.up(h))


class Llama(nn.Module):
    """A LLaMA-like language model: RMS normalization, grouped-query causal attention with rotary
    embeddings of base 10000, a SwiGLU MLP; native: nn.RMSNorm and the attention's own grouping."""

    def __init__(self, vocabulary=13, d=16, heads=4, kv_heads=2, hidden=24, context=6, native=False):
        super().__init__()
        self.tokens = nn.Embedding(vocabulary, d)
        self.blocks = nn.ModuleList([LlamaBlock(d, heads, kv_heads, hidden, native) for _ in range(2)])
        self.norm = nn.RMSNorm(d, eps=1e-6) if native else RMSNorm(d)
        self.head = nn.Linear(d, vocabulary, bias=False)
        head = d // heads
        angles = torch.outer(torch.arange(context).float(), 1.0 / 10000.0 ** (torch.arange(0, head, 2).float() / head))
        self.register_buffer("cos", torch.cat([angles, angles], -1).cos(), persistent=False)
        self.register_buffer("sin", torch.cat([angles, angles], -1).sin(), persistent=False)

    def forward(self, idx):
        T = idx.shape[1]
        x = self.tokens(idx)
        for block in self.blocks:
            x = block(x, self.cos[:T], self.sin[:T])
        return self.head(self.norm(x))


class Encoder(nn.Module):
    """A transformer encoder over vectors: a projection, nn.TransformerEncoderLayer (PyTorch's
    multi-head attention without a mask, a ReLU MLP, normalization after each), the tokens averaged,
    a head."""

    def __init__(self, features=5, d=8, heads=2, classes=3):
        super().__init__()
        self.inp = nn.Linear(features, d)
        self.layer = nn.TransformerEncoderLayer(d, heads, 16, dropout=0.0, batch_first=True)
        self.head = nn.Linear(d, classes)

    def forward(self, x):
        return self.head(self.layer(self.inp(x)).mean(1))


class TokensSoftmax(nn.Module):
    """A softmax over each token's channels, which no layer computes (refused)."""

    def __init__(self):
        super().__init__()
        self.tokens, self.head = nn.Embedding(7, 4), nn.Linear(4, 7)

    def forward(self, idx):
        return torch.softmax(self.head(self.tokens(idx)), -1)


class Gram(nn.Module):
    """A product of two computed tensors outside an attention (refused)."""

    def __init__(self):
        super().__init__()
        self.tokens = nn.Embedding(7, 4)

    def forward(self, idx):
        x = self.tokens(idx)
        return x @ x.transpose(1, 2)


def transformers():
    torch.manual_seed(17)
    gpt, llama, native, encoder = Gpt(), Llama(), Llama(native=True), Encoder()
    tokens = lambda m: torch.randint(0, m.tokens.num_embeddings, (3, 6))
    vectors = lambda m: torch.randn(3, 6, 5)
    yield "onnx_gpt", gpt, tokens, dict(dynamo=False)
    yield "onnx_gpt_dynamo", gpt, tokens, dict(dynamo=True)
    yield "onnx_gpt_opset14", gpt, tokens, dict(dynamo=False, opset_version=14)
    yield "onnx_llama", llama, tokens, dict(dynamo=False)
    yield "onnx_llama_dynamo", llama, tokens, dict(dynamo=True)
    yield "onnx_llama_opset23", native, tokens, dict(dynamo=True, opset_version=23)
    yield "onnx_encoder", encoder, vectors, dict(dynamo=False)
    yield "onnx_encoder_dynamo", encoder, vectors, dict(dynamo=True)
    yield "onnx_tokens_softmax", TokensSoftmax(), tokens, dict(dynamo=False)
    yield "onnx_gram", Gram(), tokens, dict(dynamo=False)


def ort_fused(T=6, V=11, d=8, heads=2):
    """A graph of the fused operators ONNX Runtime's transformer optimizer writes (com.microsoft's
    Attention, SkipLayerNormalization, FastGelu, SkipSimplifiedLayerNormalization, MultiHeadAttention,
    BiasGelu), built by hand, and its outputs computed in PyTorch: the model, the token inputs, the
    expected outputs."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    torch.manual_seed(19)
    r = lambda *shape, s=0.4: torch.randn(*shape) * s
    w = dict(wte=r(V, d, s=1.0), wpe=r(T, d), g0=1 + r(d, s=0.2), b0=r(d, s=0.1), wqkv=r(d, 3 * d), bqkv=r(3 * d, s=0.1),
             wp=r(d, d), bp=r(d, s=0.1), g1=1 + r(d, s=0.2), b1=r(d, s=0.1), wf=r(d, 2 * d), bf=r(2 * d, s=0.1),
             wo=r(2 * d, d), bo=r(d, s=0.1), g2=1 + r(d, s=0.2), wq=r(d, d), wk=r(d, d), wv=r(d, d), bm=r(3 * d, s=0.1),
             w2=r(d, d), b2=r(d, s=0.1), wh=r(d, V))
    ms = dict(domain="com.microsoft")
    nodes = [
        helper.make_node("Gather", ["wte", "idx"], ["e"]),
        helper.make_node("Add", ["e", "wpe"], ["x0"]),
        helper.make_node("LayerNormalization", ["x0", "g0", "b0"], ["h"], axis=-1, epsilon=1e-5),
        helper.make_node("Attention", ["h", "wqkv", "bqkv"], ["a"], num_heads=heads, unidirectional=1, **ms),
        helper.make_node("MatMul", ["a", "wp"], ["p"]),
        helper.make_node("SkipLayerNormalization", ["p", "x0", "g1", "b1", "bp"], ["n1", "", "", "s1"], epsilon=1e-5, **ms),
        helper.make_node("MatMul", ["n1", "wf"], ["f"]),
        helper.make_node("FastGelu", ["f", "bf"], ["g"], **ms),
        helper.make_node("MatMul", ["g", "wo"], ["o"]),
        helper.make_node("SkipSimplifiedLayerNormalization", ["o", "s1", "g2", "bo"], ["y", "", "", "s2"], epsilon=1e-6, **ms),
        helper.make_node("MatMul", ["y", "wq"], ["q"]),
        helper.make_node("MatMul", ["y", "wk"], ["k"]),
        helper.make_node("MatMul", ["y", "wv"], ["v"]),
        helper.make_node("MultiHeadAttention", ["q", "k", "v", "bm"], ["m"], num_heads=heads, unidirectional=1, **ms),
        helper.make_node("MatMul", ["m", "w2"], ["m2"]),
        helper.make_node("BiasGelu", ["m2", "b2"], ["z"], **ms),
        helper.make_node("Add", ["z", "s2"], ["zs"]),
        helper.make_node("MatMul", ["zs", "wh"], ["logits"]),
    ]
    graph = helper.make_graph(nodes, "ort_fused", [helper.make_tensor_value_info("idx", TensorProto.INT64, [1, T])],
                              [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, T, V])],
                              [numpy_helper.from_array(t.numpy(), k) for k, t in w.items()])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("com.microsoft", 1)])

    def attention(q, k, v):
        B, S, C = q.shape
        q, k, v = (t.view(B, S, heads, C // heads).transpose(1, 2) for t in (q, k, v))
        return F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, S, C)

    def rms(x, g, eps):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * g

    idx = torch.randint(0, V, (3, T))
    x0 = w["wte"][idx] + w["wpe"]
    h = F.layer_norm(x0, (d,), w["g0"], w["b0"], 1e-5)
    qkv = h @ w["wqkv"] + w["bqkv"]
    s1 = attention(*qkv.split(d, -1)) @ w["wp"] + w["bp"] + x0
    n1 = F.layer_norm(s1, (d,), w["g1"], w["b1"], 1e-5)
    s2 = F.gelu(n1 @ w["wf"] + w["bf"], approximate="tanh") @ w["wo"] + w["bo"] + s1
    y = rms(s2, w["g2"], 1e-6)
    bq, bk, bv = w["bm"].split(d)
    m = attention(y @ w["wq"] + bq, y @ w["wk"] + bk, y @ w["wv"] + bv)
    logits = (F.gelu(m @ w["w2"] + w["b2"]) + s2) @ w["wh"]
    return model.SerializeToString(), idx, logits


def randomize_norms(model):
    for m in model.modules():
        if isinstance(m, (nn.LayerNorm, nn.RMSNorm, RMSNorm)):     # parameters other than the initial ones
            m.weight.data.uniform_(0.5, 1.5)
            if getattr(m, "bias", None) is not None:
                m.bias.data.uniform_(-0.2, 0.2)


def strip_metadata(data):
    """The model without the exporter's debugging metadata: doc strings, node metadata, value_info."""
    import onnx
    m = onnx.load_from_string(data)
    del m.graph.value_info[:]
    for n in m.graph.node:
        n.doc_string = ""
        del n.metadata_props[:]
    m.graph.doc_string = ""
    del m.metadata_props[:]
    return m.SerializeToString()


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

    for name, model, make_input, options in transformers():
        model.eval()
        randomize_norms(model)
        x = make_input(model)
        with torch.no_grad():
            y = model(x)
        buffer = io.BytesIO()
        torch.onnx.export(model, (x[:1],), buffer, **options)
        data = strip_metadata(buffer.getvalue()) if options["dynamo"] else buffer.getvalue()
        with open(os.path.join(OUT, name + ".onnx"), "wb") as f:
            f.write(data)
        write_expected(name, x, y)
        print(f"{name}: {len(data)} bytes")
    data, x, y = ort_fused()
    with open(os.path.join(OUT, "onnx_ort_fused.onnx"), "wb") as f:
        f.write(data)
    write_expected("onnx_ort_fused", x, y)
    print(f"onnx_ort_fused: {len(data)} bytes")

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
