// SPDX-License-Identifier: MIT
// Copyright (c) 2026 pka_human (pka_human@proton.me)

package io.github.pkahuman.spingalett;

import java.lang.foreign.MemorySegment;

/** An error the library reported: its code (SPINGALETT_ERR_*) and message. */
public final class SpingalettException extends RuntimeException {
    private final int code;

    public SpingalettException(int code, String message) {
        super(message + " (code " + code + ")");
        this.code = code;
    }

    public int code() { return code; }

    /** The library's last error, or one saying `what` when it set none. */
    static SpingalettException last(String what) {
        try {
            int code = (int) Native.LAST_ERROR_CODE.invokeExact();
            String message = Native.string((MemorySegment) Native.LAST_ERROR_MESSAGE.invokeExact());
            return new SpingalettException(code != 0 ? code : -1, message == null || message.isEmpty() ? what : message);
        } catch (Throwable t) {
            throw Native.rethrow(t);
        }
    }

    static void clear() {
        try {
            Native.CLEAR_ERROR.invokeExact();
        } catch (Throwable t) {
            throw Native.rethrow(t);
        }
    }

    static int code0() {
        try {
            return (int) Native.LAST_ERROR_CODE.invokeExact();
        } catch (Throwable t) {
            throw Native.rethrow(t);
        }
    }
}
