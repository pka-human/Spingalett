// SPDX-License-Identifier: MIT
// Copyright (c) 2026 pka_human (pka_human@proton.me)

package io.github.pkahuman.spingalett;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SegmentAllocator;
import java.lang.foreign.ValueLayout;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.OptionalInt;

/**
 * The tokenizer of a language model (spingalett_tokenizer_load()): a Hugging Face tokenizer.json file, a model
 * directory holding one (its tokenizer_config.json names BOS and EOS), a GGUF file, or the bytes of a tokenizer.json
 * or GGUF file. Byte-level BPE (GPT-2, SmolLM), tiktoken's encodings (LLaMA 3, Qwen2) and SentencePiece's BPE
 * (LLaMA 2, Mistral, Phi-3), encoding and decoding as Hugging Face's tokenizers library does. A tokenizer is
 * read-only: threads may share it.
 */
public final class Tokenizer implements AutoCloseable {
    /** The id no token has (SPINGALETT_NO_TOKEN, 0xFFFFFFFF). */
    public static final int NO_TOKEN = -1;

    private MemorySegment ptr;

    private Tokenizer(MemorySegment ptr) { this.ptr = ptr; }

    /** A tokenizer.json file, a model directory holding one, or a GGUF file. */
    public static Tokenizer load(Path path) {
        try (Arena arena = Arena.ofConfined()) {
            SpingalettException.clear();
            MemorySegment p = (MemorySegment) Native.LanguageModels.TOKENIZER_LOAD.invokeExact(arena.allocateFrom(path.toString()));
            if (p.equals(MemorySegment.NULL)) throw SpingalettException.last("spingalett_tokenizer_load failed");
            return new Tokenizer(p);
        } catch (Throwable t) {
            throw Native.rethrow(t);
        }
    }

    /** The bytes of a tokenizer.json or GGUF file, which are not kept. */
    public static Tokenizer fromBytes(byte[] data) {
        try (Arena arena = Arena.ofConfined()) {
            SpingalettException.clear();
            MemorySegment p = (MemorySegment) Native.LanguageModels.TOKENIZER_FROM_MEMORY.invokeExact(
                    arena.allocateFrom(ValueLayout.JAVA_BYTE, data), (long) data.length);
            if (p.equals(MemorySegment.NULL)) throw SpingalettException.last("spingalett_tokenizer_from_memory failed");
            return new Tokenizer(p);
        } catch (Throwable t) {
            throw Native.rethrow(t);
        }
    }

    MemorySegment ptr() {
        if (ptr == null) throw new IllegalStateException("closed tokenizer");
        return ptr;
    }

    /**
     * The ids of a text. special: special tokens written in it ("&lt;|im_start|&gt;") become their ids (otherwise
     * they are text like any other, so that text from users cannot forge them); addSpecial: the tokens the tokenizer
     * adds around a text (LLaMA's BOS).
     */
    public int[] encode(String text, boolean special, boolean addSpecial) {
        byte[] bytes = text.getBytes(StandardCharsets.UTF_8);
        long capacity = bytes.length + 16L;
        try (Arena arena = Arena.ofConfined()) {
            /* the bytes and a NUL after them: the library takes an empty length as "up to the NUL" */
            MemorySegment s = arena.allocate(bytes.length + 1L);
            MemorySegment.copy(bytes, 0, s, ValueLayout.JAVA_BYTE, 0, bytes.length);
            var A = Native.TOKENIZE_ARGS;
            for (;;) {
                MemorySegment ids = arena.allocate(ValueLayout.JAVA_INT, capacity);
                MemorySegment a = arena.allocate(A);
                Native.set(a, A, "tokenizer", ptr());
                Native.set(a, A, "text", s);
                Native.set(a, A, "length", (long) bytes.length);
                Native.set(a, A, "tokens", ids);
                Native.set(a, A, "capacity", capacity);
                Native.set(a, A, "special", (byte) (special ? 1 : 0));
                Native.set(a, A, "add_special", (byte) (addSpecial ? 1 : 0));
                long count = (long) Native.LanguageModels.TOKENIZE.invokeExact(a);
                if (count < 0) throw SpingalettException.last("spingalett_tokenize failed");
                if (count <= capacity) return Arrays.copyOf(ids.toArray(ValueLayout.JAVA_INT), (int) count);
                capacity = count;                       /* more tokens than the guess: again, with room for all */
            }
        } catch (Throwable t) {
            throw Native.rethrow(t);
        }
    }

    /** The text of ids (skipSpecial: without the special tokens); bytes that are no UTF-8 become U+FFFD. */
    public String decode(int[] ids, boolean skipSpecial) {
        long capacity = 4L * ids.length + 64;
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment tokens = arena.allocateFrom(ValueLayout.JAVA_INT, ids.length == 0 ? new int[1] : ids);
            var D = Native.DETOKENIZE_ARGS;
            for (;;) {
                MemorySegment text = arena.allocate(capacity);
                MemorySegment a = arena.allocate(D);
                Native.set(a, D, "tokenizer", ptr());
                Native.set(a, D, "tokens", tokens);
                Native.set(a, D, "count", (long) ids.length);
                Native.set(a, D, "text", text);
                Native.set(a, D, "capacity", capacity);
                Native.set(a, D, "skip_special", (byte) (skipSpecial ? 1 : 0));
                long size = (long) Native.LanguageModels.DETOKENIZE.invokeExact(a);
                if (size < 0) throw SpingalettException.last("spingalett_detokenize failed");
                if (size < capacity)
                    return new String(text.asSlice(0, size).toArray(ValueLayout.JAVA_BYTE), StandardCharsets.UTF_8);
                capacity = size + 1;                    /* the text and its NUL */
            }
        } catch (Throwable t) {
            throw Native.rethrow(t);
        }
    }

    /** The token of an id as the vocabulary writes it ("Ġthe", "▁the", "&lt;|im_start|&gt;"), null for no token. */
    public String token(int id) {
        try {
            return Native.string((MemorySegment) Native.LanguageModels.TOKENIZER_TOKEN.invokeExact(ptr(), id));
        } catch (Throwable t) {
            throw Native.rethrow(t);
        }
    }

    /** The id of a token as the vocabulary writes it (added tokens first), empty when there is none. */
    public OptionalInt id(String token) {
        try (Arena arena = Arena.ofConfined()) {
            int id = (int) Native.LanguageModels.TOKENIZER_ID.invokeExact(ptr(), arena.allocateFrom(token));
            return id == NO_TOKEN ? OptionalInt.empty() : OptionalInt.of(id);
        } catch (Throwable t) {
            throw Native.rethrow(t);
        }
    }

    private int info(String field) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment info = (MemorySegment) Native.LanguageModels.TOKENIZER_INFO_OF.invokeExact((SegmentAllocator) arena, ptr());
            return (int) Native.get(info, Native.TOKENIZER_INFO, field);
        } catch (Throwable t) {
            throw Native.rethrow(t);
        }
    }

    private OptionalInt token(String field) {
        int id = info(field);
        return id == NO_TOKEN ? OptionalInt.empty() : OptionalInt.of(id);
    }

    /** Token ids: the largest one + 1 (the model's tokens and the added ones). */
    public int vocabulary() { return info("vocabulary"); }
    /** Added tokens (special ones and others). */
    public int added() { return info("added"); }
    /** The token beginning a text, when the tokenizer names one. */
    public OptionalInt bos() { return token("bos"); }
    /** The token ending a text, when the tokenizer names one. */
    public OptionalInt eos() { return token("eos"); }
    public OptionalInt pad() { return token("pad"); }
    public OptionalInt unk() { return token("unk"); }

    @Override
    public void close() {
        if (ptr == null) return;
        try {
            Native.LanguageModels.TOKENIZER_FREE.invokeExact(ptr);
        } catch (Throwable t) {
            throw Native.rethrow(t);
        }
        ptr = null;
    }
}
