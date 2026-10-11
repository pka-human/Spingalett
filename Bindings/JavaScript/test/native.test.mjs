// SPDX-License-Identifier: MIT
// Copyright (c) 2026 pka_human (pka_human@proton.me)

/*
 * The WebAssembly module against the native engine: the references of test/references (made by the
 * native engine with the host's kernels and glibc, make_references.mjs) for integer models bit for bit
 * and float ones within rounding, and the same code built natively (portable kernels, the package's
 * mathematical functions) bit for bit for every model, when a C compiler is at hand.
 */

import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import test from 'node:test';
import { loadEngine } from '../index.js';
import { buildNative, compare, compiler, GENERATIONS, HERE, isInteger, MODELS, generateNative, runNative, samples }
    from './native.mjs';

const engine = await loadEngine();
const references = JSON.parse(readFileSync(join(HERE, 'references', 'references.json'), 'utf8'));

/* the module's outputs of the model's samples */
function outputs({ path, count }) {
    const model = engine.loadModel(readFileSync(path));
    try {
        return { y: model.runBatch(samples(model, count)), integer: isInteger(model), outputSize: model.outputSize };
    } finally {
        model.free();
    }
}

function generate(g) {
    const model = engine.loadModel(readFileSync(MODELS.find((m) => m.name === g.model).path));
    try {
        return model.generate(g.prompt, g.count, g);
    } finally {
        model.free();
    }
}

test('the native engine\'s outputs: integer models bit for bit, float ones within rounding', (t) => {
    assert.equal(references.models.length, MODELS.length, 'a reference for every model (make_references.mjs)');
    for (const m of MODELS) {
        const bytes = readFileSync(join(HERE, 'references', `${m.name}.out`));
        const reference = new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
        const { y, integer, outputSize } = outputs(m);
        assert.equal(y.length, reference.length, `${m.name}: outputs`);
        const { worst, differ } = compare(y, reference, outputSize);
        t.diagnostic(`${m.name.padEnd(12)} ${integer ? (differ ? `${differ} values differ` : 'bit-exact')
            : `${differ} of ${y.length} values differ, by ${worst.toExponential(1)} at most`}`);
        if (integer) assert.equal(differ, 0, `${m.name}: an integer model's outputs differ from the native engine's`);
        else assert.ok(worst < 1e-5, `${m.name}: relative difference ${worst} from the native engine`);
    }
});

test('generation chooses the native engine\'s tokens', () => {
    for (const g of references.generations) assert.deepEqual(generate(g), g.tokens, `${g.model} from ${g.prompt}`);
});

test('built natively, the module\'s code gives its bits', { skip: !compiler() && 'no C compiler' }, (t) => {
    const native = buildNative('portable');
    try {
        for (const m of MODELS) {
            const { y, outputSize } = outputs(m);
            const { worst, differ } = compare(y, runNative(native, m.path, m.count), outputSize);
            assert.equal(differ, 0, `${m.name}: ${differ} values differ, by ${worst} at most`);
        }
        for (const g of GENERATIONS) assert.deepEqual(generate(g), generateNative(native, MODELS.find((m) => m.name === g.model).path, g));
        t.diagnostic(`${MODELS.length} models and ${GENERATIONS.length} generations bit for bit`);
    } finally {
        native.dispose();
    }
});
