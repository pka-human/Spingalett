# SPDX-License-Identifier: MIT
# Copyright (c) 2026 pka_human (pka_human@proton.me)
"""Whole models as graphs of operators against PyTorch: each model, with random weights in its published architecture
(no downloads), exported by torch.onnx, opened by spingalett.Graph and run on the same inputs; the outputs are compared
(the largest difference relative to the largest reference value), as are the graph's outputs with and without its
fusions, and both are timed (medians of interleaved runs on the threads given):

    SPINGALETT_LIBRARY=Bin/libspingalett.so python Tests/check_graph_models.py [names...] [--threads N] [--keep DIR]

It needs torch, transformers, diffusers and torchvision. The models: the text encoder of Stable Diffusion (CLIP
ViT-L/14's), Stable Diffusion's VAE decoder, a UNet of SDXL's architecture (its blocks, attention and conditioning,
fewer channels and layers), ResNet-50, ViT-B/16, BERT-base, GPT-2 (its logits over a causal window) and Whisper's
encoder (tiny). A graph also goes through a .slett file of format 11 and must give the same bits.
"""

import argparse
import os
import shutil
import statistics
import sys
import tempfile
import time
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "Bindings", "Python"))
import spingalett as sp                                     # noqa: E402

torch.manual_seed(0)


def clip_text():
    from transformers import CLIPTextConfig, CLIPTextModel
    model = CLIPTextModel(CLIPTextConfig(vocab_size=49408, hidden_size=768, intermediate_size=3072, num_hidden_layers=12,
                                         num_attention_heads=12, max_position_embeddings=77, hidden_act="quick_gelu"))
    ids = torch.randint(0, 49408, (1, 77))

    class Wrap(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.m = model

        def forward(self, input_ids):
            out = self.m(input_ids=input_ids)
            return out.last_hidden_state, out.pooler_output
    return Wrap(), (ids,), ["input_ids"]


def vae_decoder():
    from diffusers import AutoencoderKL
    vae = AutoencoderKL(in_channels=3, out_channels=3, latent_channels=4, block_out_channels=(128, 256, 512, 512),
                        down_block_types=("DownEncoderBlock2D",) * 4, up_block_types=("UpDecoderBlock2D",) * 4,
                        layers_per_block=2, norm_num_groups=32, sample_size=512)

    class Wrap(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.vae = vae

        def forward(self, latent):
            return self.vae.decode(latent).sample
    return Wrap(), (torch.randn(1, 4, 32, 32),), ["latent"]


def sdxl_unet():
    from diffusers import UNet2DConditionModel
    unet = UNet2DConditionModel(
        sample_size=32, in_channels=4, out_channels=4, layers_per_block=2,
        block_out_channels=(64, 128, 256), down_block_types=("DownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D"),
        up_block_types=("CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D"), attention_head_dim=(2, 4, 8),
        transformer_layers_per_block=(1, 2, 2), cross_attention_dim=128, use_linear_projection=True,
        addition_embed_type="text_time", addition_time_embed_dim=32, projection_class_embeddings_input_dim=6 * 32 + 128,
        norm_num_groups=32)

    class Wrap(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.unet = unet

        def forward(self, sample, timestep, text, pooled, time_ids):
            return self.unet(sample, timestep, encoder_hidden_states=text,
                             added_cond_kwargs={"text_embeds": pooled, "time_ids": time_ids}).sample
    inputs = (torch.randn(1, 4, 32, 32), torch.tensor([500.0]), torch.randn(1, 77, 128), torch.randn(1, 128),
              torch.tensor([[256.0, 256.0, 0.0, 0.0, 256.0, 256.0]]))
    return Wrap(), inputs, ["sample", "timestep", "text", "pooled", "time_ids"]


def resnet50():
    import torchvision
    return torchvision.models.resnet50().eval(), (torch.randn(1, 3, 224, 224),), ["image"]


def vit_b16():
    import torchvision
    model = torchvision.models.vit_b_16().eval()
    torch.nn.init.normal_(model.heads.head.weight, std=0.02)       # (torchvision starts the head at zeros)
    return model, (torch.randn(1, 3, 224, 224),), ["image"]


def bert_base():
    from transformers import BertConfig, BertModel
    model = BertModel(BertConfig())
    ids = torch.randint(0, 30522, (2, 128))
    mask = torch.ones(2, 128, dtype=torch.int64)
    mask[1, 100:] = 0

    class Wrap(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.m = model

        def forward(self, input_ids, attention_mask):
            return self.m(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
    return Wrap(), (ids, mask), ["input_ids", "attention_mask"]


def gpt2():
    from transformers import GPT2Config, GPT2LMHeadModel
    model = GPT2LMHeadModel(GPT2Config())

    class Wrap(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.m = model

        def forward(self, input_ids):
            return self.m(input_ids=input_ids, use_cache=False).logits
    return Wrap(), (torch.randint(0, 50257, (1, 64)),), ["input_ids"]


def whisper_encoder():
    from transformers import WhisperConfig, WhisperModel
    model = WhisperModel(WhisperConfig(d_model=384, encoder_layers=4, decoder_layers=4, encoder_attention_heads=6,
                                       decoder_attention_heads=6, encoder_ffn_dim=1536, decoder_ffn_dim=1536))

    class Wrap(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.e = model.encoder

        def forward(self, features):
            return self.e(features).last_hidden_state
    return Wrap(), (torch.randn(1, 80, 3000),), ["features"]


MODELS = {"clip_text": clip_text, "vae_decoder": vae_decoder, "sdxl_unet": sdxl_unet, "resnet50": resnet50,
          "vit_b16": vit_b16, "bert_base": bert_base, "gpt2": gpt2, "whisper_encoder": whisper_encoder}


def export(model, inputs, names, path):
    model.eval()
    with torch.no_grad():
        torch.onnx.export(model, inputs, path, input_names=names, dynamo=True, external_data=False, opset_version=18,
                          optimize=True, verbose=False)


def median_time(fn, runs):
    times = []
    for _ in range(runs):
        t = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t)
    return statistics.median(times)


def check(name, threads, keep, runs):
    model, inputs, names = MODELS[name]()
    model.eval()
    directory = keep or tempfile.mkdtemp()
    try:
        return check_in(name, model, inputs, names, directory, runs)
    finally:
        if not keep:
            shutil.rmtree(directory, ignore_errors=True)


def check_in(name, model, inputs, names, directory, runs):
    with torch.no_grad():
        ref = model(*inputs)
    refs = [ref] if isinstance(ref, torch.Tensor) else list(ref)
    refs = [r.numpy() for r in refs]
    path = os.path.join(directory, name + ".onnx")
    export(model, inputs, names, path)
    t = time.perf_counter()
    graph = sp.Graph.load(path)
    opened = time.perf_counter() - t
    feeds = [x.numpy() for x in inputs]
    out = graph.run(feeds)
    got = [out[o.name] for o in graph.outputs]
    worst = 0.0
    for g, r in zip(got, refs):
        if g.shape != r.shape:
            print(f"{name}: an output of shape {g.shape}, PyTorch's {r.shape}")
            return False
        worst = max(worst, float(np.max(np.abs(g.astype(np.float64) - r)) / max(np.max(np.abs(r)), 1e-30)))
    # the fusions against the graph without them
    os.environ["SPINGALETT_GRAPH_FUSE"] = "0"
    plain = sp.Graph.load(path)
    del os.environ["SPINGALETT_GRAPH_FUSE"]
    unfused = plain.run(feeds)
    fusion = max(float(np.max(np.abs(unfused[o.name].astype(np.float64) - g)) / max(np.max(np.abs(g)), 1e-30))
                 for o, g in zip(graph.outputs, got))
    plain.close()
    # the same bits through a file of format 11, and again on one thread
    slett = os.path.join(directory, name + ".slett")
    graph.save(slett)
    again = sp.Graph.load(slett).run(feeds)
    same = all(np.array_equal(again[o.name], g) for o, g in zip(graph.outputs, got))
    sp.set_compute_mode(sp.ComputeMode.SINGLE_THREADED)
    single = graph.run(feeds)
    sp.set_compute_mode(sp.ComputeMode.OPENMP)
    same_threads = all(np.array_equal(single[o.name], g) for o, g in zip(graph.outputs, got))
    # interleaved timing
    ours, theirs = [], []
    with torch.no_grad():
        for _ in range(runs):
            ours.append(median_time(lambda: graph.run(feeds), 1))
            theirs.append(median_time(lambda: model(*inputs), 1))
    ok = worst < 1e-3 and fusion < 1e-3 and same and same_threads
    print(f"{name:16s} {'ok ' if ok else 'BAD'} rel. error {worst:.2e}  fused/not {fusion:.1e}  "
          f"file {'same' if same else 'DIFFERS'}  "
          f"threads {'same' if same_threads else 'DIFFER'}  nodes {len(graph.operators):5d}  open {opened * 1000:7.1f} ms  "
          f"run {statistics.median(ours) * 1000:9.2f} ms  PyTorch {statistics.median(theirs) * 1000:9.2f} ms")
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs="*")
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--keep", default=None)
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
        sp.set_num_threads(args.threads)
    sp.set_compute_mode(sp.ComputeMode.OPENMP)
    if args.keep:
        os.makedirs(args.keep, exist_ok=True)
    bad = [n for n in (args.names or MODELS) if not check(n, args.threads, args.keep, args.runs)]
    if bad:
        print("failed:", ", ".join(bad))
        sys.exit(1)


if __name__ == "__main__":
    main()
