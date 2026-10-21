# SPDX-License-Identifier: MIT
# Copyright (c) 2026 pka_human (pka_human@proton.me)
"""Writes the language models of the `pretrained` test group into Tests/Data: tiny random checkpoints in
Hugging Face's form and GGUF files, with the logits a PyTorch reference computes for a window of tokens:

    python Tests/make_pretrained_tests.py      # needs torch, safetensors, numpy, tokenizers and sentencepiece
                                               # (and checks its dequantization with gguf's where installed)

- hf_llama/: LLaMA with grouped-query attention (4 query heads, 2 key and value heads), rotary embeddings
  of base 500000, weights in bfloat16 in two shards and model.safetensors.index.json, LLaMA 3's form of
  tokenizer (tokenizer_llama3.json) and a tokenizer_config.json;
- hf_qwen2/: Qwen2 with the biases of its queries, keys and values, one key and value head, tied embeddings
  (no lm_head), float16, Qwen2's form of tokenizer;
- hf_mistral/: Mistral with a sliding window (16 tokens) and a head size (8) other than the width over the
  heads, float32, without a tokenizer;
- hf_gpt2/: GPT-2 with its Conv1D weights ([in][out]), the causal mask buffers of old checkpoints, tied
  embeddings, float32, GPT-2's form of tokenizer;
- hf_qwen3/: Qwen3 with each head's queries and keys RMS-normalized, a head size (12) other than the width over
  the heads, tied embeddings, float16, without a tokenizer;
- hf_llama3/: LLaMA with LLaMA 3's scaled rotary embeddings (rope_scaling of rope_type "llama3", its original
  context short enough that the window sees every regime of its frequencies), bfloat16, without a tokenizer;
- hf_gemma3/: Gemma 3 by transformers (seven layers: five windowed of four tokens and of their own rotary base, one
  attending to every key with linear scaling of its frequencies, one windowed), its logits transformers'.

NAME.bin holds "SPGT", then as 32-bit little-endian integers 1, the tokens and the logits a window has (T
and T x vocabulary), then the tokens (as floats) and the logits. The tokens are those the directory's
tokenizer gives TEXT, cut to the window (the first ones, for the model without a tokenizer).

GGUF files, written by the small writer below as llama.cpp's converter writes them:

- gguf_llama.gguf: hf_llama's weights (queries and keys permuted into llama.cpp's order, matrices in
  bfloat16, vectors in float32) and its tokenizer as a vocabulary of the "gpt2" model ("llama-bpe");
- gguf_qwen2.gguf: hf_qwen2's (float16, its biases, no output.weight) and its tokenizer ("qwen2");
- gguf_gpt2.gguf: hf_gpt2's (its Conv1D matrices transposed to [out][in], learned positions), no vocabulary;
- gguf_qwen3.gguf: hf_qwen3's (float16, attn_q_norm and attn_k_norm), architecture "qwen3";
- gguf_llama3.gguf: hf_llama3's (bfloat16, queries and keys permuted) with rope_freqs.weight, the factors llama.cpp's
  converter computes;
- gguf_gemma3.gguf: hf_gemma3's (float32, its normalizations' weights plus 1 as llama.cpp's converter stores them),
  architecture "gemma3";
- gguf_quant.gguf (version 2): a LLaMA of width 256 whose sixteen matrices are random blocks of fifteen
  types, from F32 to the k-quants and IQ4_NL, IQ4_XS; gguf_quant.bin the logits of its weights as
  llama.cpp's formulas dequantize them (the numpy functions below, checked against the gguf package's
  where it is installed), gguf_quant.txt each matrix's sums;
- gguf_spm.gguf: a vocabulary of SentencePiece's BPE trained by the sentencepiece library (the "llama"
  model: scores, byte tokens), without tensors; gguf_spm.txt the ids sentencepiece gives a set of texts.

The files are seeded and fixed, so they change only when this script does.
"""
import json
import math
import os
import shutil
import struct

import numpy as np
import torch
import torch.nn.functional as tf
from safetensors.torch import save_file

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data")
TEXT = "The quick brown fox jumps over the lazy dog, it's 1234 times!"
T = 10


def rounded(x, dtype):
    return x.to(dtype).to(torch.float32)


def llama_weights(cfg, dtype, biases, tied, gen):
    V, d, F = cfg["vocab_size"], cfg["hidden_size"], cfg["intermediate_size"]
    H, KV = cfg["num_attention_heads"], cfg["num_key_value_heads"]
    hd = cfg.get("head_dim", d // H)
    r = lambda *shape: torch.randn(*shape, generator=gen) * 0.2
    sd = {"model.embed_tokens.weight": r(V, d), "model.norm.weight": 1 + r(d)}
    if not tied:
        sd["lm_head.weight"] = r(V, d)
    for i in range(cfg["num_hidden_layers"]):
        h = f"model.layers.{i}."
        sd.update({h + "self_attn.q_proj.weight": r(H * hd, d), h + "self_attn.k_proj.weight": r(KV * hd, d),
                   h + "self_attn.v_proj.weight": r(KV * hd, d), h + "self_attn.o_proj.weight": r(d, H * hd),
                   h + "mlp.gate_proj.weight": r(F, d), h + "mlp.up_proj.weight": r(F, d), h + "mlp.down_proj.weight": r(d, F),
                   h + "input_layernorm.weight": 1 + r(d), h + "post_attention_layernorm.weight": 1 + r(d)})
        if biases:
            for n, m in (("q", H * hd), ("k", KV * hd), ("v", KV * hd)):
                sd[h + f"self_attn.{n}_proj.bias"] = r(m)
        if cfg["model_type"] == "qwen3":
            sd[h + "self_attn.q_norm.weight"], sd[h + "self_attn.k_norm.weight"] = 1 + r(hd), 1 + r(hd)
    return {k: v.to(dtype).contiguous() for k, v in sd.items()}


def llama3_frequencies(cfg):
    """LLaMA 3's rotary frequencies (transformers' _compute_llama3_parameters(), in double precision) and the factors
    llama.cpp's converter writes in rope_freqs.weight (its generate_extra_tensors(): the base frequency over each)."""
    hd = cfg.get("head_dim", cfg["hidden_size"] // cfg["num_attention_heads"])
    base = 1.0 / cfg["rope_theta"] ** (torch.arange(0, hd, 2, dtype=torch.float64) / hd)
    sc = cfg["rope_scaling"]
    factor, low, high, old = sc["factor"], sc["low_freq_factor"], sc["high_freq_factor"], sc["original_max_position_embeddings"]
    inv, factors = [], []
    for f in base.tolist():
        wavelen = 2 * math.pi / f
        if wavelen < old / high:
            inv.append(f), factors.append(1.0)
        elif wavelen > old / low:
            inv.append(f / factor), factors.append(factor)
        else:
            smooth = (old / wavelen - low) / (high - low)
            inv.append((1 - smooth) * f / factor + smooth * f), factors.append(1 / ((1 - smooth) / factor + smooth))
    return torch.tensor(inv, dtype=torch.float64), np.array(factors, dtype=np.float32)


def ref_llama(sd, cfg, tokens):
    sd = {k: v.to(torch.float32) for k, v in sd.items()}
    n, d = len(tokens), cfg["hidden_size"]
    H, KV = cfg["num_attention_heads"], cfg["num_key_value_heads"]
    hd, eps, theta = cfg.get("head_dim", d // H), cfg["rms_norm_eps"], cfg["rope_theta"]
    rms = lambda x, w: x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w
    inv = 1.0 / theta ** (torch.arange(0, hd, 2, dtype=torch.float64) / hd)
    if cfg.get("rope_scaling"):
        inv = llama3_frequencies(cfg)[0]
    f = torch.outer(torch.arange(n, dtype=torch.float64), inv)
    cos, sin = torch.cat([f, f], -1).cos().float(), torch.cat([f, f], -1).sin().float()
    rot = lambda x: torch.cat([-x[..., hd // 2:], x[..., :hd // 2]], -1)
    lin = lambda x, name: x @ sd[name + ".weight"].T + (sd[name + ".bias"] if name + ".bias" in sd else 0)
    x = sd["model.embed_tokens.weight"][tokens]
    for i in range(cfg["num_hidden_layers"]):
        h = f"model.layers.{i}."
        y = rms(x, sd[h + "input_layernorm.weight"])
        q = lin(y, h + "self_attn.q_proj").view(n, H, hd).transpose(0, 1)
        k = lin(y, h + "self_attn.k_proj").view(n, KV, hd).transpose(0, 1)
        v = lin(y, h + "self_attn.v_proj").view(n, KV, hd).transpose(0, 1)
        if h + "self_attn.q_norm.weight" in sd:             # Qwen3: each head's queries and keys normalized
            q, k = rms(q, sd[h + "self_attn.q_norm.weight"]), rms(k, sd[h + "self_attn.k_norm.weight"])
        q, k = q * cos + rot(q) * sin, k * cos + rot(k) * sin
        k, v = k.repeat_interleave(H // KV, 0), v.repeat_interleave(H // KV, 0)
        a = tf.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(0, 1).reshape(n, H * hd)
        x = x + lin(a, h + "self_attn.o_proj")
        y = rms(x, sd[h + "post_attention_layernorm.weight"])
        x = x + lin(tf.silu(lin(y, h + "mlp.gate_proj")) * lin(y, h + "mlp.up_proj"), h + "mlp.down_proj")
    head = sd.get("lm_head.weight", sd["model.embed_tokens.weight"])
    return rms(x, sd["model.norm.weight"]) @ head.T


def gpt2_weights(cfg, gen):
    V, d, P = cfg["vocab_size"], cfg["n_embd"], cfg["n_positions"]
    r = lambda *shape: torch.randn(*shape, generator=gen) * 0.2
    sd = {"wte.weight": r(V, d), "wpe.weight": r(P, d), "ln_f.weight": 1 + r(d), "ln_f.bias": r(d)}
    for i in range(cfg["n_layer"]):
        h = f"h.{i}."
        for name, (a, b) in {"attn.c_attn": (d, 3 * d), "attn.c_proj": (d, d), "mlp.c_fc": (d, 4 * d),
                             "mlp.c_proj": (4 * d, d)}.items():
            sd[h + name + ".weight"], sd[h + name + ".bias"] = r(a, b), r(b)      # Conv1D: [in][out]
        for name in ("ln_1", "ln_2"):
            sd[h + name + ".weight"], sd[h + name + ".bias"] = 1 + r(d), r(d)
        sd[h + "attn.bias"] = torch.tril(torch.ones(P, P)).view(1, 1, P, P)      # the mask buffer
    return {k: v.contiguous() for k, v in sd.items()}


def ref_gpt2(sd, cfg, tokens):
    n, d, H = len(tokens), cfg["n_embd"], cfg["n_head"]
    lin = lambda x, name: x @ sd[name + ".weight"] + sd[name + ".bias"]
    x = sd["wte.weight"][tokens] + sd["wpe.weight"][:n]
    for i in range(cfg["n_layer"]):
        h = f"h.{i}."
        y = tf.layer_norm(x, (d,), sd[h + "ln_1.weight"], sd[h + "ln_1.bias"], cfg["layer_norm_epsilon"])
        q, k, v = lin(y, h + "attn.c_attn").split(d, -1)
        q, k, v = (t.view(n, H, d // H).transpose(0, 1) for t in (q, k, v))
        a = tf.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(0, 1).reshape(n, d)
        x = x + lin(a, h + "attn.c_proj")
        y = tf.layer_norm(x, (d,), sd[h + "ln_2.weight"], sd[h + "ln_2.bias"], cfg["layer_norm_epsilon"])
        x = x + lin(tf.gelu(lin(y, h + "mlp.c_fc"), approximate="tanh"), h + "mlp.c_proj")
    x = tf.layer_norm(x, (d,), sd["ln_f.weight"], sd["ln_f.bias"], cfg["layer_norm_epsilon"])
    return x @ sd["wte.weight"].T


def tokens_of(name, vocabulary):
    """The tokens of TEXT by the directory's tokenizer (Hugging Face's tokenizers, when installed; else the ids
    the tokenizer group's expectations hold), the first T."""
    path = os.path.join(DATA, name, "tokenizer.json")
    if not os.path.exists(path):
        return list(range(3, 3 + T))
    from tokenizers import Tokenizer
    ids = Tokenizer.from_file(path).encode(TEXT, add_special_tokens=False).ids[:T]
    assert len(ids) == T and max(ids) < vocabulary
    return ids


def write(name, config, sd, logits_fn, tokenizer=None, shards=1, tokenizer_config=None):
    d = os.path.join(DATA, name)
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    with open(os.path.join(d, "config.json"), "w", newline="\n") as f:
        json.dump(config, f, indent=2)
    if tokenizer:
        shutil.copyfile(os.path.join(DATA, tokenizer), os.path.join(d, "tokenizer.json"))
    if tokenizer_config:
        with open(os.path.join(d, "tokenizer_config.json"), "w", newline="\n") as f:
            json.dump(tokenizer_config, f, indent=2)
    names = sorted(sd)
    if shards == 1:
        save_file(sd, os.path.join(d, "model.safetensors"), metadata={"format": "pt"})
    else:
        index = {"metadata": {"total_size": sum(t.numel() * t.element_size() for t in sd.values())}, "weight_map": {}}
        for s in range(shards):
            file = f"model-{s + 1:05d}-of-{shards:05d}.safetensors"
            part = {k: sd[k] for k in names[s::shards]}
            save_file(part, os.path.join(d, file), metadata={"format": "pt"})
            index["weight_map"].update({k: file for k in part})
        with open(os.path.join(d, "model.safetensors.index.json"), "w", newline="\n") as f:
            json.dump(index, f, indent=2)
    vocabulary = config["vocab_size"]
    tokens = tokens_of(name, vocabulary)
    with torch.no_grad():
        logits = logits_fn(sd, config, torch.tensor(tokens)).numpy().astype(np.float32)
    with open(os.path.join(DATA, name + ".bin"), "wb") as f:
        f.write(b"SPGT" + struct.pack("<3I", 1, T, T * vocabulary))
        f.write(np.array(tokens, dtype=np.float32).tobytes() + logits.tobytes())
    size = sum(os.path.getsize(os.path.join(d, x)) for x in os.listdir(d))
    print(f"{name}: {len(sd)} tensors, {size} bytes, logits in [{logits.min():.2f}, {logits.max():.2f}]")


# ---------------------------------------------------------------------------------------------------- GGUF

GGUF_UINT32, GGUF_INT32, GGUF_FLOAT32, GGUF_BOOL, GGUF_STRING, GGUF_ARRAY = 4, 5, 6, 7, 8, 9
# ggml's types: id, values a block, bytes a block
GGML = {"F32": (0, 1, 4), "F16": (1, 1, 2), "Q4_0": (2, 32, 18), "Q4_1": (3, 32, 20), "Q5_0": (6, 32, 22),
        "Q5_1": (7, 32, 24), "Q8_0": (8, 32, 34), "Q2_K": (10, 256, 84), "Q3_K": (11, 256, 110), "Q4_K": (12, 256, 144),
        "Q5_K": (13, 256, 176), "Q6_K": (14, 256, 210), "IQ4_NL": (20, 32, 18), "IQ4_XS": (23, 256, 136),
        "BF16": (30, 1, 2)}
IQ4 = np.array([-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113], dtype=np.float32)


def gguf_string(s):
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def gguf_value(kind, value):
    if kind == GGUF_STRING:
        return gguf_string(value)
    if kind == GGUF_ARRAY:
        item, items = value
        return struct.pack("<IQ", item, len(items)) + b"".join(gguf_value(item, v) for v in items)
    return struct.pack({GGUF_UINT32: "<I", GGUF_INT32: "<i", GGUF_FLOAT32: "<f", GGUF_BOOL: "<?"}[kind], value)


def gguf_file(path, metadata, tensors, version=3, alignment=32):
    """A GGUF file: metadata [(key, type, value)], tensors [(name, dims innermost first, ggml type, bytes)]."""
    out = bytearray(b"GGUF" + struct.pack("<IQQ", version, len(tensors), len(metadata)))
    for key, kind, value in metadata:
        out += gguf_string(key) + struct.pack("<I", kind) + gguf_value(kind, value)
    offset = 0
    for name, dims, kind, data in tensors:
        out += gguf_string(name) + struct.pack("<I", len(dims)) + struct.pack(f"<{len(dims)}Q", *dims)
        out += struct.pack("<IQ", GGML[kind][0], offset)
        offset += (len(data) + alignment - 1) // alignment * alignment
    out += b"\0" * (-len(out) % alignment)
    for _, _, _, data in tensors:
        out += data + b"\0" * (-len(data) % alignment)
    with open(path, "wb") as f:
        f.write(out)
    return len(out)


def half(raw):
    return np.frombuffer(raw.tobytes(), dtype="<f2").astype(np.float32)


def dequantize(data, kind, n):
    """n values of a ggml type: llama.cpp's dequantize_row_*() in float32, in its order of operations."""
    if kind in ("F32", "F16", "BF16"):
        if kind == "BF16":
            return (np.frombuffer(data, dtype="<u2").astype(np.uint32) << 16).view(np.float32)
        return np.frombuffer(data, dtype="<f4" if kind == "F32" else "<f2").astype(np.float32)
    _, block, size = GGML[kind]
    b = np.frombuffer(data, dtype=np.uint8).reshape(-1, size)
    f32 = np.float32
    if kind in ("Q4_0", "Q4_1", "IQ4_NL"):
        d = half(b[:, 0:2])[:, None]
        qs = b[:, 4:20] if kind == "Q4_1" else b[:, 2:18]
        q = np.concatenate([qs & 0xF, qs >> 4], 1)
        if kind == "IQ4_NL":
            return (d * IQ4[q]).reshape(-1)
        if kind == "Q4_0":
            return ((q.astype(f32) - 8) * d).reshape(-1)
        return (q.astype(f32) * d + half(b[:, 2:4])[:, None]).reshape(-1)
    if kind in ("Q5_0", "Q5_1"):
        d = half(b[:, 0:2])[:, None]
        at = 2 if kind == "Q5_0" else 4
        qh = b[:, at:at + 4].copy().view("<u4")[:, 0].astype(np.uint32)
        qs = b[:, at + 4:at + 20]
        j = np.arange(16, dtype=np.uint32)
        h0 = ((qh[:, None] >> j) << 4) & 0x10
        h1 = (qh[:, None] >> (j + 12)) & 0x10
        q = np.concatenate([(qs & 0xF) | h0, (qs >> 4) | h1], 1).astype(f32)
        if kind == "Q5_0":
            return ((q - 16) * d).reshape(-1)
        return (q * d + half(b[:, 2:4])[:, None]).reshape(-1)
    if kind == "Q8_0":
        return (b[:, 2:34].view(np.int8).astype(f32) * half(b[:, 0:2])[:, None]).reshape(-1)
    out = np.zeros((b.shape[0], 256), dtype=f32)
    if kind == "Q2_K":
        sc, qs, d, dmin = b[:, :16], b[:, 16:80], half(b[:, 80:82]), half(b[:, 82:84])
        y, is_ = 0, 0
        for nb in range(2):
            q = qs[:, nb * 32:nb * 32 + 32]
            for j in range(4):
                for half_ in range(2):
                    s = sc[:, is_]; is_ += 1
                    dl, ml = d * (s & 0xF).astype(f32), dmin * (s >> 4).astype(f32)
                    out[:, y:y + 16] = dl[:, None] * ((q[:, half_ * 16:half_ * 16 + 16] >> (2 * j)) & 3).astype(f32) - ml[:, None]
                    y += 16
        return out.reshape(-1)
    if kind == "Q3_K":
        hm, qs, d = b[:, :32], b[:, 32:96], half(b[:, 108:110])
        aux = b[:, 96:108].copy().view("<u4").astype(np.uint32)
        k1, k2, tmp = np.uint32(0x03030303), np.uint32(0x0f0f0f0f), aux[:, 2].copy()
        a = np.stack([(aux[:, 0] & k2) | (((tmp >> 0) & k1) << 4), (aux[:, 1] & k2) | (((tmp >> 2) & k1) << 4),
                      ((aux[:, 0] >> 4) & k2) | (((tmp >> 4) & k1) << 4), ((aux[:, 1] >> 4) & k2) | (((tmp >> 6) & k1) << 4)], 1)
        scales = a.astype("<u4").view(np.uint8).view(np.int8).astype(np.int32)
        y, is_, m = 0, 0, 1
        for nb in range(2):
            q = qs[:, nb * 32:nb * 32 + 32]
            for j in range(4):
                for half_ in range(2):
                    dl = d * (scales[:, is_] - 32).astype(f32); is_ += 1
                    lo = ((q[:, half_ * 16:half_ * 16 + 16] >> (2 * j)) & 3).astype(np.int32)
                    hi = np.where(hm[:, half_ * 16:half_ * 16 + 16] & m, 0, 4)
                    out[:, y:y + 16] = dl[:, None] * (lo - hi).astype(f32)
                    y += 16
                m <<= 1
        return out.reshape(-1)

    def scale_min(sc, j):
        if j < 4:
            return sc[:, j] & 63, sc[:, j + 4] & 63
        return (sc[:, j + 4] & 0xF) | ((sc[:, j - 4] >> 6) << 4), (sc[:, j + 4] >> 4) | ((sc[:, j] >> 6) << 4)

    if kind in ("Q4_K", "Q5_K"):
        d, dmin, sc = half(b[:, 0:2]), half(b[:, 2:4]), b[:, 4:16]
        qh, ql = (b[:, 16:48], b[:, 48:176]) if kind == "Q5_K" else (None, b[:, 16:144])
        y = 0
        for j in range(4):
            q = ql[:, j * 32:j * 32 + 32]
            for half_ in range(2):
                s, mn = scale_min(sc, 2 * j + half_)
                dl, ml = d * s.astype(f32), dmin * mn.astype(f32)
                v = (q >> 4) if half_ else (q & 0xF)
                if qh is not None:
                    v = v + np.where(qh & (1 << (2 * j + half_)), 16, 0)
                out[:, y:y + 32] = dl[:, None] * v.astype(f32) - ml[:, None]
                y += 32
        return out.reshape(-1)
    if kind == "Q6_K":
        ql, qh, sc, d = b[:, :128], b[:, 128:192], b[:, 192:208].view(np.int8).astype(f32), half(b[:, 208:210])
        for nb in range(2):
            l_, h_, s_ = ql[:, nb * 64:nb * 64 + 64], qh[:, nb * 32:nb * 32 + 32], sc[:, nb * 8:nb * 8 + 8]
            for part, (lo, shift) in enumerate(((l_[:, :32] & 0xF, 0), (l_[:, 32:] & 0xF, 2), (l_[:, :32] >> 4, 4),
                                                (l_[:, 32:] >> 4, 6))):
                q = (lo | (((h_ >> shift) & 3) << 4)).astype(np.int32) - 32
                for half_ in range(2):
                    out[:, nb * 128 + part * 32 + half_ * 16:nb * 128 + part * 32 + half_ * 16 + 16] = \
                        (d * s_[:, 2 * part + half_])[:, None] * q[:, half_ * 16:half_ * 16 + 16].astype(f32)
        return out.reshape(-1)
    # IQ4_XS
    d, sh, sl, qs = half(b[:, 0:2]), b[:, 2:4].copy().view("<u2")[:, 0].astype(np.int32), b[:, 4:8], b[:, 8:136]
    for ib in range(8):
        ls = ((sl[:, ib // 2] >> (4 * (ib % 2))) & 0xF).astype(np.int32) | (((sh >> (2 * ib)) & 3) << 4)
        dl = d * (ls - 32).astype(f32)
        q = qs[:, ib * 16:ib * 16 + 16]
        out[:, ib * 32:ib * 32 + 16] = dl[:, None] * IQ4[q & 0xF]
        out[:, ib * 32 + 16:ib * 32 + 32] = dl[:, None] * IQ4[q >> 4]
    return out.reshape(-1)


# where each type keeps its half-precision scales (byte offset, low, high)
SCALES = {"Q4_0": [(0, 0.005, 0.03)], "Q4_1": [(0, 0.005, 0.03), (2, -0.2, 0.0)], "Q5_0": [(0, 0.003, 0.015)],
          "Q5_1": [(0, 0.003, 0.015), (2, -0.2, 0.0)], "Q8_0": [(0, 0.0005, 0.002)], "Q2_K": [(80, 0.002, 0.01), (82, 0.002, 0.01)],
          "Q3_K": [(108, 0.001, 0.003)], "Q4_K": [(0, 0.0001, 0.0005), (2, 0.0001, 0.0005)],
          "Q5_K": [(0, 0.00005, 0.00025), (2, 0.00005, 0.00025)], "Q6_K": [(208, 0.00002, 0.0001)],
          "IQ4_NL": [(0, 0.0005, 0.002)], "IQ4_XS": [(0, 0.00002, 0.0001)]}


def quantized(rows, cols, kind, rng):
    """Random blocks of a type for a [rows][cols] matrix (random quants and scales, half-precision fields set to
    sizes that keep the weights below 1), and its values."""
    if kind in ("F32", "F16", "BF16"):
        w = (rng.standard_normal((rows, cols)) * 0.2).astype(np.float32)
        if kind == "F32":
            data = w.astype("<f4").tobytes()
        elif kind == "F16":
            data = w.astype("<f2").tobytes()
        else:
            data = (w.view(np.uint32) >> 16).astype("<u2").tobytes()
        return data, dequantize(data, kind, rows * cols).reshape(rows, cols)
    _, block, size = GGML[kind]
    raw = rng.integers(0, 256, size=(rows * cols // block, size), dtype=np.uint8)
    for at, lo, hi in SCALES[kind]:
        raw[:, at:at + 2] = rng.uniform(lo, hi, raw.shape[0]).astype("<f2").view(np.uint8).reshape(-1, 2)
    data = raw.tobytes()
    return data, dequantize(data, kind, rows * cols).reshape(rows, cols)


def permute(w, heads):
    """llama.cpp's order of a head's rows (its converter's permute())."""
    return w.reshape(heads, 2, w.shape[0] // heads // 2, *w.shape[1:]).swapaxes(1, 2).reshape(w.shape)


def unpermute(w, heads):
    return w.reshape(heads, w.shape[0] // heads // 2, 2, *w.shape[1:]).swapaxes(1, 2).reshape(w.shape)


def llama_metadata(cfg, name, arch="llama"):
    H = cfg["num_attention_heads"]
    head = [(f"{arch}.attention.key_length", GGUF_UINT32, cfg["head_dim"]), (f"{arch}.attention.value_length", GGUF_UINT32,
            cfg["head_dim"])] if "head_dim" in cfg else []
    return head + [("general.architecture", GGUF_STRING, arch), ("general.name", GGUF_STRING, name),
                   ("general.alignment", GGUF_UINT32, 32), (f"{arch}.context_length", GGUF_UINT32, cfg["max_position_embeddings"]),
                   (f"{arch}.embedding_length", GGUF_UINT32, cfg["hidden_size"]),
                   (f"{arch}.block_count", GGUF_UINT32, cfg["num_hidden_layers"]),
                   (f"{arch}.feed_forward_length", GGUF_UINT32, cfg["intermediate_size"]),
                   (f"{arch}.attention.head_count", GGUF_UINT32, H),
                   (f"{arch}.attention.head_count_kv", GGUF_UINT32, cfg["num_key_value_heads"]),
                   (f"{arch}.attention.layer_norm_rms_epsilon", GGUF_FLOAT32, cfg["rms_norm_eps"]),
                   (f"{arch}.rope.freq_base", GGUF_FLOAT32, cfg["rope_theta"]),
                   (f"{arch}.rope.dimension_count", GGUF_UINT32, cfg.get("head_dim", cfg["hidden_size"] // H))]


def vocabulary_metadata(tokenizer_json, pre, bos, eos):
    """A tokenizer.json of byte-level BPE as the vocabulary of GGUF's "gpt2" model: tokens by id, the added ones control
    tokens (special) or user-defined ones, the merges as "a b" strings."""
    tok = json.load(open(os.path.join(DATA, tokenizer_json), encoding="utf-8"))
    vocab = {i: t for t, i in tok["model"]["vocab"].items()}
    types = {i: 1 for i in vocab}
    for a in tok["added_tokens"]:
        vocab[a["id"]], types[a["id"]] = a["content"], 3 if a["special"] else 4
    count = max(vocab) + 1
    merges = [m if isinstance(m, str) else " ".join(m) for m in tok["model"]["merges"]]
    meta = [("tokenizer.ggml.model", GGUF_STRING, "gpt2"), ("tokenizer.ggml.pre", GGUF_STRING, pre),
            ("tokenizer.ggml.tokens", GGUF_ARRAY, (GGUF_STRING, [vocab[i] for i in range(count)])),
            ("tokenizer.ggml.token_type", GGUF_ARRAY, (GGUF_INT32, [types[i] for i in range(count)])),
            ("tokenizer.ggml.merges", GGUF_ARRAY, (GGUF_STRING, merges))]
    if bos is not None:
        meta.append(("tokenizer.ggml.bos_token_id", GGUF_UINT32, bos))
    meta.append(("tokenizer.ggml.eos_token_id", GGUF_UINT32, eos))
    meta.append(("tokenizer.ggml.add_bos_token", GGUF_BOOL, bos is not None))
    return meta


def tensor_bytes(w, kind):
    if kind == "BF16":
        return (w.view(np.uint32) >> 16).astype("<u2").tobytes()
    return w.astype("<f2" if kind == "F16" else "<f4").tobytes()


def gguf_from_hf(file, arch, cfg, sd, matrices, vocabulary=None):
    """A checkpoint's weights in a GGUF file, as llama.cpp's converter writes them: LLaMA's queries and keys permuted
    (not Qwen2's), GPT-2's Conv1D matrices transposed to [out][in] and its mask buffers left out; matrices in a
    lossless type for the checkpoint's values, vectors in float32; a vocabulary when given."""
    names = {"model.embed_tokens.weight": "token_embd.weight", "model.norm.weight": "output_norm.weight",
             "lm_head.weight": "output.weight", "wte.weight": "token_embd.weight", "wpe.weight": "position_embd.weight",
             "ln_f.weight": "output_norm.weight", "ln_f.bias": "output_norm.bias"}
    parts = {"input_layernorm": "attn_norm", "self_attn.q_proj": "attn_q", "self_attn.k_proj": "attn_k",
             "self_attn.v_proj": "attn_v", "self_attn.o_proj": "attn_output", "post_attention_layernorm": "ffn_norm",
             "mlp.gate_proj": "ffn_gate", "mlp.up_proj": "ffn_up", "mlp.down_proj": "ffn_down", "ln_1": "attn_norm",
             "attn.c_attn": "attn_qkv", "attn.c_proj": "attn_output", "ln_2": "ffn_norm", "mlp.c_fc": "ffn_up",
             "mlp.c_proj": "ffn_down", "self_attn.q_norm": "attn_q_norm", "self_attn.k_norm": "attn_k_norm"}
    tensors = []
    for key in sorted(sd):
        w = sd[key].to(torch.float32).numpy()
        if key in names:
            name = names[key]
        else:
            layer, part = key.split(".", 2)[1:] if arch == "gpt2" else key[len("model.layers."):].split(".", 1)
            part, suffix = part.rsplit(".", 1)
            if part == "attn":                                  # GPT-2's causal mask buffer
                continue
            name = f"blk.{layer}.{parts[part]}.{suffix}"
            if arch == "llama" and part == "self_attn.q_proj":
                w = permute(w, cfg["num_attention_heads"])
            elif arch == "llama" and part == "self_attn.k_proj":
                w = permute(w, cfg["num_key_value_heads"])
            elif arch == "gpt2" and w.ndim == 2:
                w = np.ascontiguousarray(w.T)                   # Conv1D [in][out] to [out][in]
        kind = matrices if w.ndim == 2 else "F32"
        tensors.append((name, list(w.shape[::-1]), kind, tensor_bytes(w, kind)))
    if arch == "gpt2":
        metadata = [("general.architecture", GGUF_STRING, "gpt2"), ("general.alignment", GGUF_UINT32, 32),
                    ("gpt2.context_length", GGUF_UINT32, cfg["n_positions"]), ("gpt2.embedding_length", GGUF_UINT32, cfg["n_embd"]),
                    ("gpt2.block_count", GGUF_UINT32, cfg["n_layer"]),
                    ("gpt2.feed_forward_length", GGUF_UINT32, 4 * cfg["n_embd"]),
                    ("gpt2.attention.head_count", GGUF_UINT32, cfg["n_head"]),
                    ("gpt2.attention.layer_norm_epsilon", GGUF_FLOAT32, cfg["layer_norm_epsilon"])]
    else:
        metadata = llama_metadata(cfg, file.split(".")[0].replace("gguf_", "hf_"), arch)
    metadata += vocabulary_metadata(*vocabulary) if vocabulary else []
    if cfg.get("rope_scaling"):                         # LLaMA 3's factors, as the converter adds them
        factors = llama3_frequencies(cfg)[1]
        tensors.append(("rope_freqs.weight", [len(factors)], "F32", factors.astype("<f4").tobytes()))
    size = gguf_file(os.path.join(DATA, file), metadata, tensors)
    print(f"{file}: {len(tensors)} tensors, {size} bytes")


# the quantized model's matrices: its type, and whether llama.cpp permutes its rows
QUANT_TYPES = {"token_embd": "Q4_K", "output": "Q6_K",
               "blk.0.attn_q": "Q2_K", "blk.0.attn_k": "F32", "blk.0.attn_v": "F16", "blk.0.ffn_gate": "Q5_K",
               "blk.0.ffn_up": "Q3_K", "blk.0.attn_output": "Q4_0", "blk.0.ffn_down": "Q5_0",
               "blk.1.attn_q": "IQ4_XS", "blk.1.attn_k": "BF16", "blk.1.attn_v": "Q4_1", "blk.1.ffn_gate": "Q4_K",
               "blk.1.ffn_up": "Q8_0", "blk.1.attn_output": "Q5_1", "blk.1.ffn_down": "IQ4_NL"}


def gguf_quantized():
    """A LLaMA of width 256 (the k-quants' blocks) whose matrices are of fifteen types (GGUF version 2, no
    vocabulary); the logits of its dequantized weights (queries and keys back in Hugging Face's order) by the
    PyTorch reference, and per matrix the sums the test checks its dequantization by."""
    cfg = {"vocab_size": 32, "hidden_size": 256, "intermediate_size": 32, "num_hidden_layers": 2, "num_attention_heads": 2,
           "num_key_value_heads": 1, "head_dim": 16, "rms_norm_eps": 1e-5, "rope_theta": 10000.0,
           "max_position_embeddings": 64}
    V, d, F, H, KV, hd = 32, 256, 32, 2, 1, 16
    rng = np.random.default_rng(77)
    shapes = {"token_embd": (V, d), "output": (V, d), "attn_q": (H * hd, d), "attn_k": (KV * hd, d), "attn_v": (KV * hd, d),
              "attn_output": (d, H * hd), "ffn_gate": (F, d), "ffn_up": (F, d), "ffn_down": (d, F)}
    hf = {"token_embd": "model.embed_tokens", "output": "lm_head", "attn_q": "self_attn.q_proj", "attn_k": "self_attn.k_proj",
          "attn_v": "self_attn.v_proj", "attn_output": "self_attn.o_proj", "ffn_gate": "mlp.gate_proj", "ffn_up": "mlp.up_proj",
          "ffn_down": "mlp.down_proj"}
    tensors, sd, sums = [], {}, []
    for name, kind in QUANT_TYPES.items():
        part = name.split(".")[-1]
        rows, cols = shapes[part]
        data, w = quantized(rows, cols, kind, rng)
        tensors.append((name + ".weight", [cols, rows], kind, data))
        if part == "attn_q":
            w = unpermute(w, H)
        elif part == "attn_k":
            w = unpermute(w, KV)
        key = (hf[part] if "." not in name else f"model.layers.{name.split('.')[1]}.{hf[part]}") + ".weight"
        sd[key] = torch.from_numpy(w.copy())
    for name in ("output_norm", "blk.0.attn_norm", "blk.0.ffn_norm", "blk.1.attn_norm", "blk.1.ffn_norm"):
        w = (1 + rng.standard_normal(d) * 0.1).astype(np.float32)
        tensors.append((name + ".weight", [d], "F32", w.tobytes()))
        key = "model.norm.weight" if name == "output_norm" else \
            f"model.layers.{name.split('.')[1]}.{'input_layernorm' if 'attn' in name else 'post_attention_layernorm'}.weight"
        sd[key] = torch.from_numpy(w)
    size = gguf_file(os.path.join(DATA, "gguf_quant.gguf"), llama_metadata(cfg, "quantized"), tensors, version=2)
    tokens = list(range(1, 1 + T))
    with torch.no_grad():
        logits = ref_llama(sd, cfg, torch.tensor(tokens)).numpy().astype(np.float32)
    with open(os.path.join(DATA, "gguf_quant.bin"), "wb") as f:
        f.write(b"SPGT" + struct.pack("<3I", 1, T, T * V))
        f.write(np.array(tokens, dtype=np.float32).tobytes() + logits.tobytes())
    # the sums of each matrix as the network holds it: layer, then the sum, the sum of squares and the sum weighted by
    # (row + 1) (col + 1) (which sees the order of rows and values), in double precision value after value in the
    # order of the rows, as the test adds them, so that they agree exactly when the weights do
    lines = ["# gguf_quant.gguf: layer index, first row of the matrix in the layer, type, rows, cols, sum, sum of squares, "
             "weighted sum"]
    layer_of = {"token_embd": 1, "output": 2 + 11 * 2 + 1}
    for name, kind in QUANT_TYPES.items():
        part = name.split(".")[-1]
        if "." in name:
            b = int(name.split(".")[1])
            layer = 2 + 11 * b + {"attn_q": 1, "attn_k": 1, "attn_v": 1, "attn_output": 3, "ffn_gate": 6, "ffn_up": 7,
                                  "ffn_down": 9}[part]
            key = f"model.layers.{b}.{hf[part]}.weight"
        else:
            layer, key = layer_of[part], hf[part] + ".weight"
        first = {"attn_q": 0, "attn_k": H * hd, "attn_v": H * hd + KV * hd}.get(part, 0)
        w = sd[key].numpy()
        s = s2 = sw = 0.0
        for r, row in enumerate(w.astype(np.float64).tolist()):
            for c, v in enumerate(row):
                s += v
                s2 += v * v
                sw += v * (r + 1.0) * (c + 1.0)
        lines.append(f"{layer} {first} {kind} {w.shape[0]} {w.shape[1]} {s!r} {s2!r} {sw!r}")
    with open(os.path.join(DATA, "gguf_quant.txt"), "w", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    try:                                    # gguf-py's dequantization, where it is installed: the reference's own check
        from gguf.quants import dequantize as gguf_dequantize
        from gguf.constants import GGMLQuantizationType
        for name, dims, kind, data in tensors:
            want = gguf_dequantize(np.frombuffer(data, dtype=np.uint8), GGMLQuantizationType(GGML[kind][0])).reshape(-1)
            assert np.array_equal(want.astype(np.float32), dequantize(data, kind, len(want))), name
        print("  (the dequantization agrees with gguf-py's on every type)")
    except ImportError:
        pass
    print(f"gguf_quant.gguf: {len(tensors)} tensors of {len(set(QUANT_TYPES.values()))} types, {size} bytes")


SPM_TEXTS = ["", "Hello world", " Hello", "Hello ", "  two  spaces", "It's 12345 café naïve", "Tabs\tand\nnewlines\n",
             "你好，世界！", "Emoji 😀 👍🏽", "unknown ꙮ ǅ chars", "<s>Hello</s>", "a<s>b", "Hello <s> world </s>",
             "<unk> x", "xxxxxxxxxxxxxxxx", "The quick brown fox jumps over the lazy dog."]


def gguf_sentencepiece():
    """A vocabulary of SentencePiece's BPE (trained by the sentencepiece library, with byte fallback) in a GGUF file
    of no tensors, and the ids and texts sentencepiece gives the texts (the special tokens found in the text and each
    piece between them encoded with its own space prefix, as llama.cpp does)."""
    import io
    import re
    import sentencepiece as spm
    model = io.BytesIO()
    lines = [line for text in CORPUS_SPM for line in text.split("\n") if line]
    spm.SentencePieceTrainer.train(sentence_iterator=iter(lines), model_writer=model, model_type="bpe", vocab_size=330,
                                   byte_fallback=True, character_coverage=0.9995, normalization_rule_name="identity",
                                   remove_extra_whitespaces=False, split_digits=True, bos_id=1, eos_id=2, unk_id=0,
                                   minloglevel=2)
    sp = spm.SentencePieceProcessor(model_proto=model.getvalue())
    count = sp.get_piece_size()
    kinds = [2 if sp.is_unknown(i) else 3 if sp.is_control(i) else 6 if sp.is_byte(i) else 5 if sp.is_unused(i) else 1
             for i in range(count)]
    metadata = [("general.architecture", GGUF_STRING, "llama"), ("general.alignment", GGUF_UINT32, 32),
                ("tokenizer.ggml.model", GGUF_STRING, "llama"),
                ("tokenizer.ggml.tokens", GGUF_ARRAY, (GGUF_STRING, [sp.id_to_piece(i) for i in range(count)])),
                ("tokenizer.ggml.scores", GGUF_ARRAY, (GGUF_FLOAT32, [sp.get_score(i) for i in range(count)])),
                ("tokenizer.ggml.token_type", GGUF_ARRAY, (GGUF_INT32, kinds)),
                ("tokenizer.ggml.bos_token_id", GGUF_UINT32, 1), ("tokenizer.ggml.eos_token_id", GGUF_UINT32, 2),
                ("tokenizer.ggml.unknown_token_id", GGUF_UINT32, 0), ("tokenizer.ggml.add_bos_token", GGUF_BOOL, True),
                ("tokenizer.ggml.add_space_prefix", GGUF_BOOL, True)]
    size = gguf_file(os.path.join(DATA, "gguf_spm.gguf"), metadata, [])
    specials = {"<s>": 1, "</s>": 2, "<unk>": 0}
    out = ["# gguf_spm.gguf: special add text, then the ids, the text decoded, and without special tokens"]
    for special in (1, 0):
        for add in (0, 1):
            for text in SPM_TEXTS:
                pieces = re.split("(<s>|</s>|<unk>)", text) if special else [text]
                ids = [1] if add else []
                for p in pieces:
                    if p in specials and special:
                        ids.append(specials[p])
                    elif p:
                        ids += sp.encode(p)
                # sentencepiece decodes without the special tokens; "*": the text with them is not checked
                decoded = sp.decode([i for i in ids if i not in (0, 1, 2)])
                hexed = [s.encode("utf-8").hex() or "-" for s in (text, decoded)]
                out += [f"{special} {add} {hexed[0]}", " ".join(map(str, [len(ids)] + ids)), "*", hexed[1]]
    with open(os.path.join(DATA, "gguf_spm.txt"), "w", newline="\n") as f:
        f.write("\n".join(out) + "\n")
    print(f"gguf_spm.gguf: {count} tokens, {size} bytes")


CORPUS_SPM = [
    "The quick brown fox jumps over the lazy dog. It's a dog's life; we'll see what they've done.",
    "Tokenizers cut text into pieces: words, numbers like 12345 and 3.14159, and punctuation!",
    "Le café est très bon, naïve résumé, straße. Hello world, hello there, two  spaces and tabs\tand lines.",
    "你好，世界！Привет, мир! Emoji are bytes too.",
] * 20


def gemma3():
    """Gemma 3 by transformers, its GGUF file as llama.cpp's converter writes it, and its logits by transformers."""
    import transformers
    torch.manual_seed(2027)
    cfg = transformers.Gemma3TextConfig(
        vocab_size=64, hidden_size=32, intermediate_size=48, num_hidden_layers=7, num_attention_heads=2,
        num_key_value_heads=1, head_dim=16, sliding_window=4, max_position_embeddings=64, query_pre_attn_scalar=16,
        rms_norm_eps=1e-6, tie_word_embeddings=True,
        rope_parameters={"sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
                         "full_attention": {"rope_type": "linear", "factor": 8.0, "rope_theta": 1000000.0}})
    model = transformers.Gemma3ForCausalLM(cfg).eval()
    with torch.no_grad():
        for name, p in model.named_parameters():
            p.copy_(torch.randn(p.shape) * (0.3 if "norm" in name else 0.2))
    d = os.path.join(DATA, "hf_gemma3")
    shutil.rmtree(d, ignore_errors=True)
    model.save_pretrained(d, safe_serialization=True)
    tokens = list(range(3, 3 + T))
    with torch.no_grad():
        logits = model(torch.tensor(tokens)[None]).logits[0].numpy().astype(np.float32)
    with open(os.path.join(DATA, "hf_gemma3.bin"), "wb") as f:
        f.write(b"SPGT" + struct.pack("<3I", 1, T, T * cfg.vocab_size))
        f.write(np.array(tokens, dtype=np.float32).tobytes() + logits.tobytes())
    parts = {"input_layernorm": "attn_norm", "self_attn.q_proj": "attn_q", "self_attn.k_proj": "attn_k",
             "self_attn.v_proj": "attn_v", "self_attn.o_proj": "attn_output", "self_attn.q_norm": "attn_q_norm",
             "self_attn.k_norm": "attn_k_norm", "post_attention_layernorm": "post_attention_norm",
             "pre_feedforward_layernorm": "ffn_norm", "post_feedforward_layernorm": "post_ffw_norm",
             "mlp.gate_proj": "ffn_gate", "mlp.up_proj": "ffn_up", "mlp.down_proj": "ffn_down"}
    tensors = []
    for key, value in sorted(model.state_dict().items()):
        w = value.to(torch.float32).numpy()
        if key == "model.embed_tokens.weight":
            name = "token_embd.weight"
        elif key == "model.norm.weight":
            name = "output_norm.weight"
        elif key == "lm_head.weight":
            continue                                        # tied
        else:
            layer, part = key[len("model.layers."):].split(".", 1)
            part, suffix = part.rsplit(".", 1)
            name = f"blk.{layer}.{parts[part]}.{suffix}"
        if name.endswith("norm.weight"):                    # Gemma3RMSNorm's (1 + w), as the converter stores it
            w = w + 1.0
        tensors.append((name, list(w.shape[::-1]), "F32", w.astype("<f4").tobytes()))
    a = "gemma3"
    metadata = [("general.architecture", GGUF_STRING, a), ("general.alignment", GGUF_UINT32, 32),
                (f"{a}.context_length", GGUF_UINT32, 64), (f"{a}.embedding_length", GGUF_UINT32, 32),
                (f"{a}.block_count", GGUF_UINT32, 7), (f"{a}.feed_forward_length", GGUF_UINT32, 48),
                (f"{a}.attention.head_count", GGUF_UINT32, 2), (f"{a}.attention.head_count_kv", GGUF_UINT32, 1),
                (f"{a}.attention.key_length", GGUF_UINT32, 16), (f"{a}.attention.value_length", GGUF_UINT32, 16),
                (f"{a}.attention.layer_norm_rms_epsilon", GGUF_FLOAT32, 1e-6), (f"{a}.rope.freq_base", GGUF_FLOAT32, 1e6),
                (f"{a}.rope.freq_base_swa", GGUF_FLOAT32, 1e4), (f"{a}.attention.sliding_window", GGUF_UINT32, 4),
                (f"{a}.rope.scaling.type", GGUF_STRING, "linear"), (f"{a}.rope.scaling.factor", GGUF_FLOAT32, 8.0)]
    size = gguf_file(os.path.join(DATA, "gguf_gemma3.gguf"), metadata, tensors)
    print(f"hf_gemma3, gguf_gemma3.gguf: {len(tensors)} tensors, {size} bytes, logits in [{logits.min():.2f}, "
          f"{logits.max():.2f}]")


def main():
    gen = torch.Generator().manual_seed(2024)
    llama = {"architectures": ["LlamaForCausalLM"], "model_type": "llama", "vocab_size": 425, "hidden_size": 16,
             "intermediate_size": 24, "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2,
             "rms_norm_eps": 1e-5, "rope_theta": 500000.0, "max_position_embeddings": 64, "hidden_act": "silu",
             "tie_word_embeddings": False, "torch_dtype": "bfloat16", "rope_scaling": None}
    llama_sd = llama_weights(llama, torch.bfloat16, False, False, gen)
    write("hf_llama", llama, llama_sd, ref_llama, "tokenizer_llama3.json",
          shards=2, tokenizer_config={"bos_token": "<|begin_of_text|>", "eos_token": "<|eot_id|>"})
    qwen2 = {"architectures": ["Qwen2ForCausalLM"], "model_type": "qwen2", "vocab_size": 423, "hidden_size": 16,
             "intermediate_size": 24, "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 1,
             "rms_norm_eps": 1e-6, "rope_theta": 1000000.0, "max_position_embeddings": 32, "hidden_act": "silu",
             "tie_word_embeddings": True, "use_sliding_window": False, "sliding_window": 32, "torch_dtype": "float16"}
    qwen2_sd = llama_weights(qwen2, torch.float16, True, True, gen)
    write("hf_qwen2", qwen2, qwen2_sd, ref_llama, "tokenizer_qwen2.json")
    mistral = {"architectures": ["MistralForCausalLM"], "model_type": "mistral", "vocab_size": 64, "hidden_size": 16,
               "intermediate_size": 24, "num_hidden_layers": 1, "num_attention_heads": 4, "num_key_value_heads": 2,
               "head_dim": 8, "rms_norm_eps": 1e-5, "rope_theta": 10000.0, "max_position_embeddings": 128,
               "sliding_window": 16, "hidden_act": "silu", "tie_word_embeddings": False, "torch_dtype": "float32"}
    write("hf_mistral", mistral, llama_weights(mistral, torch.float32, False, False, gen), ref_llama)
    gpt2 = {"architectures": ["GPT2LMHeadModel"], "model_type": "gpt2", "vocab_size": 421, "n_embd": 16, "n_layer": 1,
            "n_head": 4, "n_positions": 32, "n_ctx": 32, "layer_norm_epsilon": 1e-5, "activation_function": "gelu_new",
            "scale_attn_weights": True, "torch_dtype": "float32"}
    gpt2_sd = gpt2_weights(gpt2, gen)
    write("hf_gpt2", gpt2, gpt2_sd, ref_gpt2, "tokenizer_gpt2.json")
    gguf_from_hf("gguf_llama.gguf", "llama", llama, llama_sd, "BF16", ("tokenizer_llama3.json", "llama-bpe", 420, 424))
    gguf_from_hf("gguf_qwen2.gguf", "qwen2", qwen2, qwen2_sd, "F16", ("tokenizer_qwen2.json", "qwen2", None, 420))
    gguf_from_hf("gguf_gpt2.gguf", "gpt2", gpt2, gpt2_sd, "F32")
    gguf_quantized()
    gguf_sentencepiece()
    gen = torch.Generator().manual_seed(2026)
    qwen3 = {"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3", "vocab_size": 96, "hidden_size": 16,
             "intermediate_size": 24, "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2,
             "head_dim": 12, "rms_norm_eps": 1e-6, "rope_theta": 1000000.0, "max_position_embeddings": 64,
             "hidden_act": "silu", "tie_word_embeddings": True, "attention_bias": False, "torch_dtype": "float16"}
    qwen3_sd = llama_weights(qwen3, torch.float16, False, True, gen)
    write("hf_qwen3", qwen3, qwen3_sd, ref_llama)
    llama3 = {"architectures": ["LlamaForCausalLM"], "model_type": "llama", "vocab_size": 80, "hidden_size": 32,
              "intermediate_size": 24, "num_hidden_layers": 2, "num_attention_heads": 2, "num_key_value_heads": 1,
              "rms_norm_eps": 1e-5, "rope_theta": 10000.0, "max_position_embeddings": 64, "hidden_act": "silu",
              "tie_word_embeddings": False, "torch_dtype": "bfloat16",
              "rope_scaling": {"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0,
                               "original_max_position_embeddings": 32}}
    llama3_sd = llama_weights(llama3, torch.bfloat16, False, False, gen)
    write("hf_llama3", llama3, llama3_sd, ref_llama)
    gguf_from_hf("gguf_qwen3.gguf", "qwen3", qwen3, qwen3_sd, "F16")
    gguf_from_hf("gguf_llama3.gguf", "llama", llama3, llama3_sd, "BF16")
    gemma3()
    try:                                    # transformers' models, where it is installed: the reference's own check
        import transformers
        for name in ("hf_qwen3", "hf_llama3", "hf_llama", "hf_qwen2"):
            model = transformers.AutoModelForCausalLM.from_pretrained(os.path.join(DATA, name), torch_dtype=torch.float32)
            raw = open(os.path.join(DATA, name + ".bin"), "rb").read()
            n = struct.unpack("<I", raw[8:12])[0]
            tokens = np.frombuffer(raw[16:16 + 4 * n], dtype=np.float32).astype(np.int64)
            want = np.frombuffer(raw[16 + 4 * n:], dtype=np.float32).reshape(n, -1)
            with torch.no_grad():
                got = model(torch.tensor(tokens)[None]).logits[0].numpy()
            assert np.abs(got - want).max() < 1e-4, (name, np.abs(got - want).max())
        print("  (the references agree with transformers' models)")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
