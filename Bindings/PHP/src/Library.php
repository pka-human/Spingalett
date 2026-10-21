<?php

// SPDX-License-Identifier: MIT
// Copyright (c) 2026 pka_human (pka_human@proton.me)

declare(strict_types=1);

namespace Spingalett;

use FFI;

/**
 * The library: its C declarations (spingalett.h) loaded through PHP's FFI extension, and its settings.
 *
 * The shared library (version 1.2 or later of 1.x) is the file SPINGALETT_LIBRARY names, else the one in
 * the directory SPINGALETT_LIB_DIR names, else the one the system's loader finds: an installed copy, or
 * a directory on LD_LIBRARY_PATH (DYLD_LIBRARY_PATH on macOS, PATH on Windows). Under opcache.preload,
 * Library::preload() loads the declarations once for all requests; with ffi.enable=preload, PHP's
 * default, that is the only way to use FFI outside the command line.
 */
final class Library
{
    /** The FFI scope of the preloaded declarations. */
    public const SCOPE = 'SPINGALETT';

    /** The C declarations. */
    public const HEADER = __DIR__ . '/../spingalett.h';

    private static ?FFI $ffi = null;

    private function __construct()
    {
    }

    /** The library's functions and types: preloaded, else loaded from the library found on first use. */
    public static function ffi(): FFI
    {
        if (self::$ffi === null) {
            self::extension();
            self::$ffi = self::scope() ?? self::open(self::candidates());
        }
        return self::$ffi;
    }

    /** Uses the library at $path (a file, or a name the system's loader finds) from now on. */
    public static function load(string $path): FFI
    {
        self::extension();
        return self::$ffi = self::open([$path]);
    }

    /**
     * Loads the declarations into the scope SPINGALETT with FFI::load(), for a script that
     * opcache.preload runs: the requests that follow use them without loading them again. $path as for
     * load(); by default the library found as described above.
     */
    public static function preload(?string $path = null): void
    {
        self::extension();
        if ($path === null) {
            $path = self::find();
        }
        // FFI::load() reads the scope and the library from directives at the top of a file.
        $file = tempnam(sys_get_temp_dir(), 'spingalett');
        if ($file === false) {
            throw new SpingalettException('Library::preload(): no temporary file', -1);
        }
        try {
            file_put_contents($file, '#define FFI_SCOPE "' . self::SCOPE . "\"\n#define FFI_LIB \"$path\"\n"
                                     . self::declarations());
            $ffi = FFI::load($file);
        } finally {
            unlink($file);
        }
        if ($ffi === null) {
            throw new SpingalettException("Library::preload(): FFI::load() of $path failed", -1);
        }
        self::$ffi = self::checked($ffi, $path);
    }

    /** The library's version ("1.4.1"). */
    public static function version(): string
    {
        return (string) Native::string(self::ffi()->spingalett_version());
    }

    /** Where training and inference run (a ComputeMode constant); false when the mode is not available. */
    public static function setComputeMode(int $mode): bool
    {
        return self::ffi()->spingalett_set_compute_mode($mode);
    }

    /** The compute mode (a ComputeMode constant). */
    public static function computeMode(): int
    {
        return self::ffi()->spingalett_get_compute_mode();
    }

    /**
     * The precision of the GPU's products (Precision::FLOAT32 or Precision::BFLOAT16); false when the
     * GPU has none such.
     */
    public static function setGpuPrecision(int $precision): bool
    {
        return self::ffi()->spingalett_set_gpu_precision($precision);
    }

    /** Seeds the calling thread's generator (initialization, shuffling, dropout, draws). */
    public static function seed(int $seed): void
    {
        self::ffi()->spingalett_seed($seed);
    }

    /** Turns the library's informational messages on or off. */
    public static function setVerbose(bool $enabled): void
    {
        self::ffi()->spingalett_set_verbose($enabled);
    }

    /** Threads of ComputeMode::OPENMP (0, the default: one per core). */
    public static function setThreads(int $threads): void
    {
        self::ffi()->spingalett_set_num_threads($threads);
    }

    /** The GPU that ComputeMode::CUDA uses, or null. */
    public static function cudaDevice(): ?string
    {
        return Native::string(self::ffi()->spingalett_cuda_device());
    }

    /** The GPU that ComputeMode::VULKAN uses, or null. */
    public static function gpuDevice(): ?string
    {
        return Native::string(self::ffi()->spingalett_gpu_device());
    }

    /** The instruction set of the CPU's matrix kernels ("AVX-512", "AVX2", "NEON", ...). */
    public static function cpuKernels(): string
    {
        return (string) Native::string(self::ffi()->spingalett_cpu_kernels());
    }

    private static function extension(): void
    {
        if (!extension_loaded('ffi')) {
            throw new SpingalettException('PHP\'s FFI extension is not loaded (extension=ffi)', -1);
        }
    }

    /** The preloaded declarations, if there are some. */
    private static function scope(): ?FFI
    {
        try {
            return FFI::scope(self::SCOPE);
        } catch (FFI\Exception $e) {
            return null;
        }
    }

    /** The files or names to try, in order. */
    private static function candidates(): array
    {
        $file = getenv('SPINGALETT_LIBRARY');
        if ($file !== false && $file !== '') {
            return [$file];
        }
        switch (PHP_OS_FAMILY) {
            case 'Windows':
                $names = ['spingalett.dll', 'libspingalett.dll'];
                break;
            case 'Darwin':
                $names = ['libspingalett.1.dylib', 'libspingalett.dylib'];
                break;
            default:
                $names = ['libspingalett.so.1', 'libspingalett.so'];
        }
        $dir = getenv('SPINGALETT_LIB_DIR');
        if ($dir === false || $dir === '') {
            return $names;
        }
        $files = [];
        foreach ($names as $name) {
            $path = rtrim($dir, '/\\') . DIRECTORY_SEPARATOR . $name;
            if (is_file($path)) {
                $files[] = $path;
            }
        }
        if ($files === []) {
            throw new SpingalettException("SPINGALETT_LIB_DIR ($dir) holds no " . implode(' or ', $names), -1);
        }
        return $files;
    }

    /** The first candidate the system's loader opens. */
    private static function find(): string
    {
        $errors = [];
        foreach (self::candidates() as $path) {
            try {
                FFI::cdef('', $path);
                return $path;
            } catch (FFI\Exception $e) {
                $errors[] = $e->getMessage();
            }
        }
        throw self::notFound($errors);
    }

    /** The declarations bound to the first of $paths that loads and has them all. */
    private static function open(array $paths): FFI
    {
        $errors = [];
        foreach ($paths as $path) {
            try {
                $ffi = FFI::cdef(self::declarations(), $path);
            } catch (FFI\Exception $e) {
                $errors[] = $e->getMessage();
                continue;
            }
            return self::checked($ffi, $path);
        }
        throw self::notFound($errors);
    }

    private static function notFound(array $errors): SpingalettException
    {
        return new SpingalettException('libspingalett 1.2 or later could not be loaded (set SPINGALETT_LIBRARY to '
                                       . 'the file or SPINGALETT_LIB_DIR to its directory, or install it where the '
                                       . 'system\'s loader finds it): ' . implode('; ', $errors), -1);
    }

    /** $ffi, when the library is 1.2 or a later 1.x, whose structures and functions these are. */
    private static function checked(FFI $ffi, string $path): FFI
    {
        $version = (string) Native::string($ffi->spingalett_version());
        $parts = array_map('intval', explode('.', $version));
        if ($parts[0] !== 1 || ($parts[1] ?? 0) < 2) {
            throw new SpingalettException("libspingalett $version ($path): the bindings need 1.2 or a later 1.x", -1);
        }
        return $ffi;
    }

    private static function declarations(): string
    {
        $code = @file_get_contents(self::HEADER);
        if ($code === false) {
            throw new SpingalettException('the declarations ' . self::HEADER . ' could not be read', -1);
        }
        return $code;
    }
}
