<?php

// SPDX-License-Identifier: MIT
// Copyright (c) 2026 pka_human (pka_human@proton.me)

declare(strict_types=1);

namespace Spingalett;

use FFI;
use FFI\CData;

/**
 * @internal What the classes share: PHP arrays copied into the library's numbers and back, C strings
 * and pointers.
 *
 * A buffer of values is a C array of rows of up to ROW values (float[rows][row] or uint32_t[rows][row]),
 * which pack() and unpack() fill and read a row at a time: faster than a value at a time through FFI,
 * and without passing the whole array to pack() as arguments, which would take as much memory again
 * as the array. The library sees the rows as one array (Native::pointer()).
 */
final class Native
{
    private const ROW = 16384;

    private function __construct()
    {
    }

    /** A zeroed buffer of $count values of $type ("float" or "uint32_t"). */
    public static function buffer(string $type, int $count): CData
    {
        $row = max(1, min($count, self::ROW));
        $rows = max(1, intdiv($count + $row - 1, $row));
        return Library::ffi()->new("{$type}[$rows][$row]");
    }

    /**
     * A buffer of floats holding $values: numbers, or arrays of numbers (a sample each), which are
     * concatenated. $count receives how many values it holds.
     */
    public static function floats(array $values, ?int &$count = null): CData
    {
        if ($values !== [] && is_array(reset($values))) {
            $values = array_merge(...array_values($values));
        }
        return self::fill('float', 'f*', $values, $count);
    }

    /** A buffer of uint32_t holding $values (token ids); $count receives how many. */
    public static function uint32s(array $values, ?int &$count = null): CData
    {
        return self::fill('uint32_t', 'L*', $values, $count);
    }

    private static function fill(string $type, string $format, array $values, ?int &$count): CData
    {
        $count = count($values);
        $buffer = self::buffer($type, $count);
        if ($count === 0) {
            return $buffer;
        }
        // pack() takes the values as arguments, positional ones: the keys of a list.
        if (array_key_last($values) !== $count - 1) {
            $values = array_values($values);
        }
        $row = min($count, self::ROW);
        for ($k = 0; $k * $row < $count; $k++) {
            $bytes = pack($format, ...($count === $row ? $values : array_slice($values, $k * $row, $row)));
            $to = $buffer[$k];                  // a variable: FFI::memcpy() takes its arguments by reference
            FFI::memcpy($to, $bytes, strlen($bytes));
        }
        return $buffer;
    }

    /** The first $count values of a buffer of $type, as floats or integers. */
    public static function read(CData $buffer, string $type, int $count): array
    {
        $format = $type === 'float' ? 'f*' : 'L*';
        $first = $buffer[0];
        $row = intdiv(FFI::sizeof($first), 4);
        if ($count <= $row) {
            return $count > 0 ? array_values(unpack($format, FFI::string($first, 4 * $count))) : [];
        }
        $values = [];
        for ($k = 0; $k * $row < $count; $k++) {
            $from = $buffer[$k];
            array_push($values, ...unpack($format, FFI::string($from, 4 * min($row, $count - $k * $row))));
        }
        return $values;
    }

    /**
     * A pointer to the first value of a buffer (or the first character of a C string), for the
     * structures' fields. The buffer must outlive its use.
     */
    public static function pointer(CData $buffer, string $type): CData
    {
        // A cast of the buffer's address, from pointer to pointer: PHP 8.5's FFI crashes on pointers
        // cast from multidimensional arrays themselves.
        return Library::ffi()->cast("$type *", FFI::addr($buffer));
    }

    /** $text as a C string (a buffer of its characters and a zero). */
    public static function cString(string $text): CData
    {
        $buffer = Library::ffi()->new('char[' . (strlen($text) + 1) . ']');
        if ($text !== '') {
            FFI::memcpy($buffer, $text, strlen($text));
        }
        return $buffer;
    }

    /** What a function returning const char * gave: PHP's FFI returns such strings as PHP strings. */
    public static function string($value): ?string
    {
        if ($value === null || is_string($value)) {
            return $value;
        }
        return FFI::isNull($value) ? null : FFI::string($value);
    }

    /** Whether a pointer a function returned is NULL (PHP's FFI returns NULL pointers as null). */
    public static function isNull($pointer): bool
    {
        return $pointer === null || FFI::isNull($pointer);
    }

    /** Samples of $size values in $values values, which must be a whole number of them. */
    public static function samples(int $values, int $size, string $what): int
    {
        if ($size === 0 || $values % $size !== 0) {
            throw new \InvalidArgumentException("$what: $values values are no whole number of samples of $size");
        }
        return intdiv($values, $size);
    }
}
