# SPDX-License-Identifier: MIT
# Copyright (c) 2026 pka_human (pka_human@proton.me)
"""Writes the tokenizers of the `tokenizer` test group into Tests/Data, with the token ids Hugging Face's
tokenizers library gives for a set of texts:

    python Tests/make_tokenizer_tests.py        # needs tokenizers

Each tokenizer is trained on the small corpus below and has the configuration of a family of real ones:

- tokenizer_gpt2.json: GPT-2's byte-level BPE (ByteLevel with its regular expression), runs of digits
  apart, <|endoftext|>, an added token that is not special;
- tokenizer_llama3.json: LLaMA 3's (tiktoken's cl100k expression, then ByteLevel; ignore_merges; the
  special tokens after the model's vocabulary; a template that adds <|begin_of_text|>), merges stored as
  pairs;
- tokenizer_qwen2.json: Qwen2's (NFC, the expression with single digits), chat tokens;
- tokenizer_o200k.json: GPT-4o's o200k expression, a Split on a string merged with the next piece, a
  Strip normalizer;
- tokenizer_spm.json: LLaMA 2's SentencePiece conversion (Prepend and Replace of U+2581, byte fallback to
  <0xXX> tokens, unknown tokens fused; Replace, ByteFallback, Fuse and Strip decoding), merges stored as
  strings, an added token with rstrip;
- tokenizer_metaspace.json: Mistral's (Metaspace with prepend_scheme "first", split here; its decoder),
  newlines removed by a Split.

tokenizer_NAME.txt lists, for each text and each choice of special (special tokens in the text become
their ids) and add_special (the template's tokens), the text as hex, the ids, and the text decoded with and
without the special tokens (as hex; "-" for an empty one). The files change only when this script does.
"""
import json
import os
import random

from tokenizers import AddedToken, Regex, Tokenizer, decoders, models, normalizers, pre_tokenizers, processors, trainers

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data")

GPT2 = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
CL100K = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
QWEN2 = CL100K.replace(r"\p{N}{1,3}", r"\p{N}")
O200K = (r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|"
         r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|"
         r"\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+")

CORPUS = [
    "The quick brown fox jumps over the lazy dog. It's a dog's life; we'll see what they've done.",
    "Tokenizers cut text into pieces: words, numbers like 12345 and 3.14159, and punctuation!",
    "def train(net, data):\n    for epoch in range(10):\n        loss = step(net, data)\n    return loss\n",
    "Le café est très bon, naïve résumé, Ærøskøbing, straße. Él dijo: «¡hola!»",
    "Привет, мир! Ελληνικά γράμματα. 你好，世界！こんにちは。안녕하세요.",
    "Emoji 😀👍🏽 and symbols ∑ ∞ ≈ → are bytes too. مرحبا بالعالم. नमस्ते दुनिया.",
    "THE QUICK BROWN FOX, I'M HERE, YOU'D KNOW. Numbers 1 22 333 4444 55555 666666.",
    "a/b/c path//to/file -- dashes -- and \t tabs \r\n windows lines\n\n\nblank lines",
]
TEXTS = [
    "", " ", "Hello world", " Hello", "Hello ", "  two  spaces  ", "It's we'll they've I'M you'D 'ſ 'x",
    "Numbers 1 12 123 1234 12345 3.14 ½ ② Ⅻ", "Tabs\tand\nnewlines\r\n\n  \n end", "Spaces     　 ideographic nbsp",
    "café naïve straße Œuvre", "Combining café naïve 각 Å", "Emoji 😀 👍🏽 👨‍👩‍👧 🇫🇷",
    "你好，世界！こんにちは 안녕하세요", "مرحبا שלום नमस्ते สวัสดี", "Γειά σου Привет", "def f(x):\n    return x**2  # c",
    "path a/b//c --dash-- x", "<|endoftext|>", "Hello<|endoftext|>world", "<s>Hello</s>", "<s> Hello </s>",
    "<|im_start|>user\nHi<|im_end|>\n", "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nHi<|eot_id|>",
    "x" * 100, " " * 40, "😀" * 10, "Mark ́abc ǅungla ʰʱ", "zzzqqq unknown ꙮ chars", "<unk>", "hello world<sep>  next",
]


def corpus():
    rng = random.Random(5)
    words = " ".join(CORPUS).split()
    lines = list(CORPUS)
    for _ in range(300):
        lines.append(" ".join(rng.choice(words) for _ in range(rng.randint(3, 12))))
    return lines


def byte_level(name, pre, vocab_size=420, ignore_merges=False, normalizer=None, specials=(), template=None,
               after_specials=(), added=()):
    tok = Tokenizer(models.BPE(ignore_merges=ignore_merges))
    if normalizer is not None:
        tok.normalizer = normalizer
    tok.pre_tokenizer = pre
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=list(specials), show_progress=False,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(corpus(), trainer)
    if after_specials:
        tok.add_special_tokens(list(after_specials))
    if added:
        tok.add_tokens(list(added))
    steps = [processors.ByteLevel(trim_offsets=False)]
    if template:
        steps.append(processors.TemplateProcessing(single=f"{template} $A", special_tokens=[(template, tok.token_to_id(template))]))
    tok.post_processor = processors.Sequence(steps) if len(steps) > 1 else steps[0]
    return name, tok


def sentencepiece(name, normalizer, pre, decoder, specials, added=(), rstrip=None):
    """A BPE with byte fallback whose vocabulary holds the special tokens, the 256 <0xXX> tokens and the trained
    ones, in that order (LLaMA 2's layout); merges written as "a b" strings."""
    tok = Tokenizer(models.BPE(unk_token="<unk>", byte_fallback=True, fuse_unk=True))
    tok.normalizer = normalizer
    if pre is not None:
        tok.pre_tokenizer = pre
    trainer = trainers.BpeTrainer(vocab_size=400, special_tokens=list(specials), show_progress=False)
    tok.train_from_iterator(corpus(), trainer)
    data = json.loads(tok.to_str())
    trained = sorted(data["model"]["vocab"].items(), key=lambda kv: kv[1])
    vocab = {}
    for s in specials:
        vocab[s] = len(vocab)
    for b in range(256):
        vocab[f"<0x{b:02X}>"] = len(vocab)
    for token, _ in trained:
        if token not in vocab:
            vocab[token] = len(vocab)
    data["model"]["vocab"] = vocab
    data["model"]["merges"] = [m if isinstance(m, str) else " ".join(m) for m in data["model"]["merges"]]
    for a in data["added_tokens"]:
        a["id"] = vocab[a["content"]]
        a["normalized"] = False
    tok = Tokenizer.from_str(json.dumps(data))
    tok.decoder = decoder
    tok.post_processor = processors.TemplateProcessing(single="<s> $A", special_tokens=[("<s>", vocab["<s>"])])
    if added:
        tok.add_tokens(list(added))
    if rstrip:
        tok.add_special_tokens([AddedToken(rstrip, rstrip=True, normalized=False, special=True)])
    return name, tok


def tokenizers():
    meta = pre_tokenizers.Metaspace(replacement="▁", prepend_scheme="first", split=True)
    spm_decoder = decoders.Sequence([decoders.Replace("▁", " "), decoders.ByteFallback(), decoders.Fuse(),
                                     decoders.Strip(" ", 1, 0)])
    return [
        byte_level("gpt2", pre_tokenizers.Sequence([pre_tokenizers.Digits(individual_digits=False),
                                                    pre_tokenizers.ByteLevel(add_prefix_space=False)]),
                   specials=["<|endoftext|>"], added=[AddedToken(" hello world", normalized=True)]),
        byte_level("llama3", pre_tokenizers.Sequence([pre_tokenizers.Split(Regex(CL100K), "isolated"),
                                                      pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)]),
                   ignore_merges=True, after_specials=["<|begin_of_text|>", "<|end_of_text|>", "<|start_header_id|>",
                                                       "<|end_header_id|>", "<|eot_id|>"], template="<|begin_of_text|>"),
        byte_level("qwen2", pre_tokenizers.Sequence([pre_tokenizers.Split(Regex(QWEN2), "isolated"),
                                                     pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)]),
                   normalizer=normalizers.NFC(), after_specials=["<|endoftext|>", "<|im_start|>", "<|im_end|>"]),
        byte_level("o200k", pre_tokenizers.Sequence([pre_tokenizers.Split("--", "merged_with_next"),
                                                     pre_tokenizers.Split(Regex(O200K), "isolated"),
                                                     pre_tokenizers.ByteLevel(add_prefix_space=True, use_regex=False)]),
                   normalizer=normalizers.Sequence([normalizers.NFC(), normalizers.Strip()]), specials=["<|endoftext|>"]),
        sentencepiece("spm", normalizers.Sequence([normalizers.Prepend("▁"), normalizers.Replace(" ", "▁")]), None,
                      spm_decoder, ["<unk>", "<s>", "</s>"], added=[AddedToken("<sep>", normalized=False)], rstrip="<|im_end|>"),
        sentencepiece("metaspace", normalizers.Sequence([]), pre_tokenizers.Sequence([pre_tokenizers.Split("\n", "removed"), meta]),
                      decoders.Metaspace(replacement="▁", prepend_scheme="first"), ["<unk>", "<s>", "</s>"]),
    ]


def main():
    for name, tok in tokenizers():
        path = os.path.join(OUT, f"tokenizer_{name}.json")
        tok.save(path, pretty=False)
        if name == "spm":                   # merges as "a b" strings, as older files store them
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            data["model"]["merges"] = [" ".join(m) for m in data["model"]["merges"]]
            with open(path, "w", encoding="utf-8", newline="\n") as f:
                json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
        tok = Tokenizer.from_file(path)
        lines = [f"# tokenizer_{name}.json: special add text, then the ids, the text decoded, and without special tokens"]
        for special in (True, False):
            tok.encode_special_tokens = not special
            for add in (False, True):
                for text in TEXTS:
                    ids = tok.encode(text, add_special_tokens=add).ids
                    hexed = [s.encode("utf-8").hex() or "-" for s in
                             (text, tok.decode(ids, skip_special_tokens=False), tok.decode(ids, skip_special_tokens=True))]
                    lines += [f"{int(special)} {int(add)} {hexed[0]}", " ".join(map(str, [len(ids)] + ids)), hexed[1], hexed[2]]
        with open(os.path.join(OUT, f"tokenizer_{name}.txt"), "w", newline="\n") as f:
            f.write("\n".join(lines) + "\n")
        print(f"tokenizer_{name}.json: {tok.get_vocab_size()} tokens, {os.path.getsize(path)} bytes")


if __name__ == "__main__":
    main()
