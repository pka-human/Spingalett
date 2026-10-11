<?php

// SPDX-License-Identifier: MIT
// Copyright (c) 2026 pka_human (pka_human@proton.me)

declare(strict_types=1);

namespace Spingalett;

/**
 * An error the library reported: its message, and as the exception's code its error code (one of the
 * constants below), or -1 for errors without one (the library not found).
 */
class SpingalettException extends \RuntimeException
{
    public const ALLOC = 1;             // out of memory
    public const INVALID = 2;           // invalid argument or file contents
    public const FILE_IO = 3;           // a file could not be opened, read or written, or is truncated
    public const FORMAT_VERSION = 4;    // a model file of an unsupported format version

    /** The calling thread's last error, or one saying $what when the library set none. */
    public static function last(string $what): self
    {
        $ffi = Library::ffi();
        $code = $ffi->spingalett_last_error_code();
        $message = Native::string($ffi->spingalett_last_error_message());
        return new self($message === null || $message === '' ? $what : $message, $code !== 0 ? $code : -1);
    }
}
