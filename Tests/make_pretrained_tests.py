# SPDX-License-Identifier: MIT
# Copyright (c) 2026 pka_human (pka_human@proton.me)
"""Writes the language models of the `pretrained` test group into Tests/Data: tiny random checkpoints in
Hugging Face's form, with the logits a PyTorch reference computes for a window of tokens:

    python Tests/make_pretrained_tests.py      # needs torch, safetensors and numpy

- hf_llama/: LLaMA with grouped-query attention (4 query heads, 2 key and value heads), rotary embeddings
  of base 500000, weights in bfloat16 in two shards and model.safetensors.index.json, LLaMA 3's form of
  tokenizer (tokenizer_llama3.json) and a tokenizer_config.json;
- hf_qwen2/: Qwen2 with the biases of its queries, keys and values, one key and value head, tied embeddings
  (no lm_head), float16, Qwen2's form of tokenizer;
- hf_mistral/: Mistral with a sliding window (16 tokens) and a head size (8) other than the width over the
  heads, float32, without a tokenizer;
- hf_gpt2/: GPT-2 with its Conv1D weights ([in][out]), the causal mask buffers of old checkpoints, tied
  embeddings, float32, GPT-2's form of tokenizer.

NAME.bin holds "SPGT", then as 32-bit little-endian integers 1, the tokens and the logits a window has (T
and T x vocabulary), then the tokens (as floats) and the logits. The tokens are those the directory's
tokenizer gives TEXT, cut to the window (the first ones, for the model without a tokenizer). The files are
seeded and fixed, so they change only when this script does.
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
    return {k: v.to(dtype).contiguous() for k, v in sd.items()}


def ref_llama(sd, cfg, tokens):
    sd = {k: v.to(torch.float32) for k, v in sd.items()}
    n, d = len(tokens), cfg["hidden_size"]
    H, KV = cfg["num_attention_heads"], cfg["num_key_value_heads"]
    hd, eps, theta = cfg.get("head_dim", d // H), cfg["rms_norm_eps"], cfg["rope_theta"]
    rms = lambda x, w: x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w
    inv = 1.0 / theta ** (torch.arange(0, hd, 2, dtype=torch.float64) / hd)
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


def main():
    gen = torch.Generator().manual_seed(2024)
    llama = {"architectures": ["LlamaForCausalLM"], "model_type": "llama", "vocab_size": 425, "hidden_size": 16,
             "intermediate_size": 24, "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2,
             "rms_norm_eps": 1e-5, "rope_theta": 500000.0, "max_position_embeddings": 64, "hidden_act": "silu",
             "tie_word_embeddings": False, "torch_dtype": "bfloat16", "rope_scaling": None}
    write("hf_llama", llama, llama_weights(llama, torch.bfloat16, False, False, gen), ref_llama, "tokenizer_llama3.json",
          shards=2, tokenizer_config={"bos_token": "<|begin_of_text|>", "eos_token": "<|eot_id|>"})
    qwen2 = {"architectures": ["Qwen2ForCausalLM"], "model_type": "qwen2", "vocab_size": 423, "hidden_size": 16,
             "intermediate_size": 24, "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 1,
             "rms_norm_eps": 1e-6, "rope_theta": 1000000.0, "max_position_embeddings": 32, "hidden_act": "silu",
             "tie_word_embeddings": True, "use_sliding_window": False, "sliding_window": 32, "torch_dtype": "float16"}
    write("hf_qwen2", qwen2, llama_weights(qwen2, torch.float16, True, True, gen), ref_llama, "tokenizer_qwen2.json")
    mistral = {"architectures": ["MistralForCausalLM"], "model_type": "mistral", "vocab_size": 64, "hidden_size": 16,
               "intermediate_size": 24, "num_hidden_layers": 1, "num_attention_heads": 4, "num_key_value_heads": 2,
               "head_dim": 8, "rms_norm_eps": 1e-5, "rope_theta": 10000.0, "max_position_embeddings": 128,
               "sliding_window": 16, "hidden_act": "silu", "tie_word_embeddings": False, "torch_dtype": "float32"}
    write("hf_mistral", mistral, llama_weights(mistral, torch.float32, False, False, gen), ref_llama)
    gpt2 = {"architectures": ["GPT2LMHeadModel"], "model_type": "gpt2", "vocab_size": 421, "n_embd": 16, "n_layer": 1,
            "n_head": 4, "n_positions": 32, "n_ctx": 32, "layer_norm_epsilon": 1e-5, "activation_function": "gelu_new",
            "scale_attn_weights": True, "torch_dtype": "float32"}
    write("hf_gpt2", gpt2, gpt2_weights(gpt2, gen), ref_gpt2, "tokenizer_gpt2.json")


if __name__ == "__main__":
    main()
