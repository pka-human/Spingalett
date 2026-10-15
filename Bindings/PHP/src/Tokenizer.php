<?php

// SPDX-License-Identifier: MIT
// Copyright (c) 2026 pka_human (pka_human@proton.me)

declare(strict_types=1);

namespace Spingalett;

use FFI\CData;

/**
 * A tokenizer of a Hugging Face tokenizer.json file (byte-level BPE as GPT-2's and tiktoken's encodings,
 * SentencePiece-style BPE with byte fallback), of a model directory holding one, or of a GGUF file's
 * vocabulary: text to token ids and back, in the library.
 */
final class Tokenizer
{
    /** The id of no token (bos(), eos(), pad(), unk() and id() when there is none). */
    public const NO_TOKEN = 0xFFFFFFFF;

    private ?CData $tokenizer;

    private function __construct(CData $tokenizer)
    {
        $this->tokenizer = $tokenizer;
    }

    /** A tokenizer.json file, a model directory with one, or a GGUF file. */
    public static function load(string $path): self
    {
        $tokenizer = Library::ffi()->spingalett_tokenizer_load($path);
        if (Native::isNull($tokenizer)) {
            throw SpingalettException::last('spingalett_tokenizer_load failed');
        }
        return new self($tokenizer);
    }

    /** The contents of a tokenizer.json file or of a GGUF file. */
    public static function fromBytes(string $data): self
    {
        $ffi = Library::ffi();
        $bytes = $ffi->new('char[' . max(1, strlen($data)) . ']');
        \FFI::memcpy($bytes, $data, strlen($data));
        $tokenizer = $ffi->spingalett_tokenizer_from_memory(\FFI::addr($bytes), strlen($data));
        if (Native::isNull($tokenizer)) {
            throw SpingalettException::last('spingalett_tokenizer_from_memory failed');
        }
        return new self($tokenizer);
    }

    /** Releases the tokenizer (as its destruction does). */
    public function close(): void
    {
        if ($this->tokenizer !== null) {
            Library::ffi()->spingalett_tokenizer_free($this->tokenizer);
            $this->tokenizer = null;
        }
    }

    public function __destruct()
    {
        $this->close();
    }

    /** The SpingalettTokenizer * of the library. */
    public function raw(): CData
    {
        if ($this->tokenizer === null) {
            throw new \LogicException('the tokenizer is closed');
        }
        return $this->tokenizer;
    }

    /** Token ids: the largest one + 1. */
    public function vocabulary(): int
    {
        return Library::ffi()->spingalett_tokenizer_info($this->raw())->vocabulary;
    }

    /** The token beginning a text (NO_TOKEN when unknown), and those ending it, padding and of what the vocabulary lacks. */
    public function bos(): int
    {
        return Library::ffi()->spingalett_tokenizer_info($this->raw())->bos;
    }

    public function eos(): int
    {
        return Library::ffi()->spingalett_tokenizer_info($this->raw())->eos;
    }

    public function pad(): int
    {
        return Library::ffi()->spingalett_tokenizer_info($this->raw())->pad;
    }

    public function unk(): int
    {
        return Library::ffi()->spingalett_tokenizer_info($this->raw())->unk;
    }

    /** A token's text as the vocabulary writes it ("Ġthe", "▁the"), null for an id of no token. */
    public function token(int $id): ?string
    {
        return Native::string(Library::ffi()->spingalett_tokenizer_token($this->raw(), $id));
    }

    /** The id of a token of the vocabulary, NO_TOKEN for none. */
    public function id(string $token): int
    {
        return Library::ffi()->spingalett_tokenizer_id($this->raw(), $token);
    }

    /**
     * The ids of a UTF-8 text: with $special, special tokens written in it ("<|im_start|>") become their ids
     * (otherwise they are text, so that text from users cannot forge them); with $addSpecial, the tokens the
     * tokenizer adds around a text (LLaMA's BOS).
     *
     * @return int[]
     */
    public function encode(string $text, bool $special = false, bool $addSpecial = false): array
    {
        $ffi = Library::ffi();
        $bytes = $ffi->new('char[' . max(1, strlen($text)) . ']');
        \FFI::memcpy($bytes, $text, strlen($text));
        $args = $ffi->new('SpingalettTokenizeArgs');
        $args->tokenizer = $this->raw();
        $args->text = $ffi->cast('const char *', \FFI::addr($bytes));
        $args->length = strlen($text);
        $args->special = $special;
        $args->add_special = $addSpecial;
        if ($text === '' && !$addSpecial) {
            return [];
        }
        $count = $ffi->spingalett_tokenize_args($args);
        if ($count < 0) {
            throw SpingalettException::last('spingalett_tokenize failed');
        }
        if ($count === 0) {
            return [];
        }
        $tokens = Native::buffer('uint32_t', $count);
        $args->tokens = Native::pointer($tokens, 'uint32_t');
        $args->capacity = $count;
        if ($ffi->spingalett_tokenize_args($args) !== $count) {
            throw SpingalettException::last('spingalett_tokenize failed');
        }
        return Native::read($tokens, 'uint32_t', $count);
    }

    /** The UTF-8 text of token ids ($skipSpecial: special tokens left out). */
    public function decode(array $tokens, bool $skipSpecial = false): string
    {
        $ffi = Library::ffi();
        $ids = Native::uint32s($tokens, $count);
        $args = $ffi->new('SpingalettDetokenizeArgs');
        $args->tokenizer = $this->raw();
        $args->tokens = Native::pointer($ids, 'uint32_t');
        $args->count = $count;
        $args->skip_special = $skipSpecial;
        $length = $ffi->spingalett_detokenize_args($args);
        if ($length < 0) {
            throw SpingalettException::last('spingalett_detokenize failed');
        }
        $text = $ffi->new('char[' . ($length + 1) . ']');
        $args->text = $ffi->cast('char *', \FFI::addr($text));
        $args->capacity = $length + 1;
        if ($ffi->spingalett_detokenize_args($args) !== $length) {
            throw SpingalettException::last('spingalett_detokenize failed');
        }
        return \FFI::string($text, $length);
    }
}
