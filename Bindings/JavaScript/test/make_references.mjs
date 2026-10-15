// SPDX-License-Identifier: MIT
// Copyright (c) 2026 pka_human (pka_human@proton.me)

/*
 * Writes test/references: the native engine's outputs ('system' in native.mjs: the host's kernels and C
 * library) for the samples of every model of native.mjs, as NAME.out (floats, little-endian), and the
 * tokens of its generations, with the platform that made them, in references.json. The tests hold the
 * WebAssembly module to them: integer models bit for bit, float ones within rounding.
 *
 *   node test/make_references.mjs              # needs a C compiler (CC, else cc)
 *
 * The references were made on x86-64 Linux with glibc 2.43, whose expf, tanhf and erff the module's
 * mathematical functions reproduce (see wasm/Spingalett.Libm.c); remake them only on such a system.
 */

import { execFileSync } from 'node:child_process';
import { writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { buildNative, compiler, GENERATIONS, HERE, MODELS, generateNative, runNative } from './native.mjs';

const native = buildNative('system');
if (!native) {
    console.error('make_references.mjs: no C compiler (set CC)');
    process.exit(1);
}
try {
    const models = [];
    for (const { name, path, count } of MODELS) {
        const y = runNative(native, path, count);
        writeFileSync(join(HERE, 'references', `${name}.out`), new Uint8Array(y.buffer, y.byteOffset, y.byteLength));
        models.push({ name, count, outputs: y.length });
    }
    const generations = GENERATIONS.map((g) =>
        ({ ...g, tokens: generateNative(native, MODELS.find((m) => m.name === g.model).path, g) }));
    const cc = execFileSync(compiler(), ['--version'], { encoding: 'utf8' }).split('\n')[0];
    // one model or generation a line
    const lines = (items) => items.map((item) => `  ${JSON.stringify(item)}`).join(',\n');
    writeFileSync(join(HERE, 'references', 'references.json'),
        `{\n "platform": ${JSON.stringify(`${process.platform} ${process.arch}`)},\n "compiler": ${JSON.stringify(cc)},\n` +
        ` "models": [\n${lines(models)}\n ],\n "generations": [\n${lines(generations)}\n ]\n}\n`);
    console.log(`references of ${models.length} models and ${generations.length} generations written`);
} finally {
    native.dispose();
}
