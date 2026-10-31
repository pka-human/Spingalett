<?php

// SPDX-License-Identifier: MIT
// Copyright (c) 2026 pka_human (pka_human@proton.me)

declare(strict_types=1);

namespace Spingalett;

use FFI\CData;

/**
 * A session of a causal language model (Model::session()): sequences fed a token at a time, the keys and
 * values of the tokens before kept, so that each new token costs one row through the model. The logits are
 * those Model::predict() gives each position of the whole window, bit for bit.
 */
final class Session
{
    private ?CData $session;
    /** @var Model (kept: the model outlives its sessions) */
    private Model $model;

    /** A session on $model of $sequences sequences of up to $maxLength tokens (0: the model's window), taking up to
        $gpuMemory bytes of the GPU's memory in a GPU mode (0: what it has free; a model it does not hold splits with the CPU). */
    public function __construct(Model $model, int $maxLength = 0, int $sequences = 1, int $gpuMemory = 0)
    {
        $ffi = Library::ffi();
        $options = $ffi->new('SpingalettSessionOptions');
        $options->max_length = $maxLength;
        $options->sequences = $sequences;
        $options->gpu_memory = $gpuMemory;
        $session = $ffi->spingalett_session_new($model->raw(), \FFI::addr($options));
        if (Native::isNull($session)) {
            throw SpingalettException::last('spingalett_session_new failed');
        }
        $this->session = $session;
        $this->model = $model;
    }

    /** Releases the session (as its destruction does). */
    public function close(): void
    {
        if ($this->session !== null) {
            Library::ffi()->spingalett_session_free($this->session);
            $this->session = null;
        }
    }

    public function __destruct()
    {
        $this->close();
    }

    /** The SpingalettSession * of the library. */
    public function raw(): CData
    {
        if ($this->session === null) {
            throw new \LogicException('the session is closed');
        }
        return $this->session;
    }

    /** Logits a position has: the size of the vocabulary. */
    public function vocabulary(): int
    {
        return Library::ffi()->spingalett_session_vocabulary($this->raw());
    }

    /** Tokens a sequence holds. */
    public function length(int $sequence = 0): int
    {
        return Library::ffi()->spingalett_session_length($this->raw(), $sequence);
    }

    /**
     * Appends tokens to a sequence and returns the logits of the token after the last of them (with $all, of
     * the token after each of them, flat).
     */
    public function feed(array $tokens, int $sequence = 0, bool $all = false): array
    {
        if ($tokens === []) {
            throw new \InvalidArgumentException('feed: one token at least');
        }
        $ffi = Library::ffi();
        $t = Native::uint32s($tokens, $count);
        $vocabulary = $this->vocabulary();
        $logits = Native::buffer('float', ($all ? $count : 1) * $vocabulary);
        $feed = $ffi->new('SpingalettSessionFeed');
        $feed->sequence = $sequence;
        $feed->tokens = Native::pointer($t, 'uint32_t');
        $feed->count = $count;
        $feed->logits = Native::pointer($logits, 'float');
        $feed->all = $all;
        if (!$ffi->spingalett_session_feed($this->raw(), \FFI::addr($feed), 1)) {
            throw SpingalettException::last('spingalett_session_feed failed');
        }
        return Native::read($logits, 'float', ($all ? $count : 1) * $vocabulary);
    }

    /** Truncates a sequence to its first $length tokens. */
    public function rewind(int $length, int $sequence = 0): void
    {
        if (!Library::ffi()->spingalett_session_rewind($this->raw(), $sequence, $length)) {
            throw SpingalettException::last('spingalett_session_rewind failed');
        }
    }

    /** Makes sequence $to a copy of sequence $from. */
    public function copy(int $from, int $to): void
    {
        if (!Library::ffi()->spingalett_session_copy($this->raw(), $from, $to)) {
            throw SpingalettException::last('spingalett_session_copy failed');
        }
    }

    /**
     * Continues sequences 0 .. count($prompts) - 1 together, each from its prompt (one token at least), each
     * drawing as Network::generate() draws (sequence b with seed + b); returns the tokens each generated.
     *
     * @param int[][] $prompts
     * @return int[][]
     */
    public function generate(array $prompts, int $count, ?Sampling $sampling = null): array
    {
        $sequences = count($prompts);
        if ($sequences === 0) {
            throw new \InvalidArgumentException('generate: one prompt at least');
        }
        $sampling = $sampling ?? new Sampling();
        $ffi = Library::ffi();
        $kept = [];
        $pointers = $ffi->new("uint32_t *[$sequences]");
        $counts = [];
        foreach (array_values($prompts) as $b => $prompt) {
            if ($prompt === []) {
                throw new \InvalidArgumentException('generate: a prompt of one token at least');
            }
            $kept[$b] = Native::uint32s($prompt, $length);
            $pointers[$b] = Native::pointer($kept[$b], 'uint32_t');
            $counts[] = $length;
        }
        $lengths = Native::uint32s($counts);
        $tokens = Native::buffer('uint32_t', max(1, $sequences * $count));
        $made = Native::buffer('uint32_t', $sequences);
        $args = $ffi->new('SpingalettSessionGenerateArgs');
        $args->session = $this->raw();
        $args->sequences = $sequences;
        $args->prompts = $ffi->cast('const uint32_t *const *', \FFI::addr($pointers[0]));
        $args->prompt_lengths = Native::pointer($lengths, 'uint32_t');
        $args->tokens = Native::pointer($tokens, 'uint32_t');
        $args->made = Native::pointer($made, 'uint32_t');
        $args->count = $count;
        $args->temperature = $sampling->temperature;
        $args->top_k = $sampling->topK;
        $args->top_p = $sampling->topP;
        $args->seed = $sampling->seed;
        if ($sampling->stop !== []) {
            $stop = Native::uint32s($sampling->stop, $stops);
            $args->stop_tokens = Native::pointer($stop, 'uint32_t');
            $args->stop_count = $stops;
        }
        $ffi->spingalett_clear_error();
        $ffi->spingalett_session_generate_args($args);
        if ($ffi->spingalett_last_error_code() !== 0) {
            throw SpingalettException::last('spingalett_session_generate failed');
        }
        $all = Native::read($tokens, 'uint32_t', $sequences * $count);
        $each = Native::read($made, 'uint32_t', $sequences);
        $out = [];
        for ($b = 0; $b < $sequences; $b++) {
            $out[] = array_slice($all, $b * $count, $each[$b]);
        }
        return $out;
    }
}
