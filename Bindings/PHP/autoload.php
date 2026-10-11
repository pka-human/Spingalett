<?php

// SPDX-License-Identifier: MIT
// Copyright (c) 2026 pka_human (pka_human@proton.me)

// The classes' autoloader for programs without Composer (Composer's is composer.json's PSR-4 mapping).

spl_autoload_register(static function (string $class): void {
    if (strncmp($class, 'Spingalett\\', 11) === 0) {
        $file = __DIR__ . '/src/' . str_replace('\\', '/', substr($class, 11)) . '.php';
        if (is_file($file)) {
            require $file;
        }
    }
});
