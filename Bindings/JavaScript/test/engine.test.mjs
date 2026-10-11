// SPDX-License-Identifier: MIT
// Copyright (c) 2026 pka_human (pka_human@proton.me)

/* The package's API: loading the module and models, what models report, runs, batches, errors, memory,
   and the language models' logits and generation. */

import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createServer } from 'node:http';
import { join } from 'node:path';
import test from 'node:test';
import { pathToFileURL } from 'node:url';
import { Engine, loadEngine, loadModel, Model, SpingalettError } from '../index.js';
import { HERE, PACKAGE, ROOT, samples, tokens } from './native.mjs';

const runtime = (name) => readFileSync(join(ROOT, 'Tests', 'Data', 'runtime', `${name}.slett`));
const language = (name) => readFileSync(join(HERE, 'models', `${name}.slett`));
const wasmPath = join(PACKAGE, 'spingalett.wasm');
const engine = await loadEngine();

test('the module loads from its default place, a path, a URL, bytes and a compiled module', async () => {
    const bytes = readFileSync(wasmPath);
    const sources = [undefined, wasmPath, pathToFileURL(wasmPath), pathToFileURL(wasmPath).href, bytes,
        bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength), new WebAssembly.Module(bytes)];
    const reference = engine.loadModel(runtime('mlp_int8')).run(samples({ inputSize: 64, layers: [{}] }, 1));
    for (const source of sources) {
        const e = await loadEngine(source);
        assert.ok(e instanceof Engine);
        assert.equal(e.formatVersion, 8);
        assert.deepEqual(e.loadModel(runtime('mlp_int8')).run(samples({ inputSize: 64, layers: [{}] }, 1)), reference);
    }
    await assert.rejects(loadEngine(42), TypeError);
    await assert.rejects(loadEngine(join(HERE, 'no-such.wasm')));
    const model = await loadModel(runtime('mlp_int8'));
    assert.ok(model instanceof Model);
    model.free();
    assert.deepEqual(WebAssembly.Module.imports(new WebAssembly.Module(bytes)), [], 'the module imports nothing');
});

test('in browsers\' way: the module and a model fetched over HTTP', async () => {
    const files = { '/spingalett.wasm': [readFileSync(wasmPath), 'application/wasm'],
        '/plain.wasm': [readFileSync(wasmPath), 'application/octet-stream'], '/model.slett': [runtime('graph1_fp16'), ''] };
    const server = createServer((request, response) => {
        const file = files[request.url];
        response.writeHead(file ? 200 : 404, file?.[1] ? { 'Content-Type': file[1] } : {});
        response.end(file?.[0]);
    });
    await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
    const base = `http://127.0.0.1:${server.address().port}`;
    try {
        for (const source of [`${base}/spingalett.wasm`, new URL(`${base}/plain.wasm`), fetch(`${base}/spingalett.wasm`),
            await fetch(`${base}/plain.wasm`)]) {
            const e = await loadEngine(source);
            const model = e.loadModel(await (await fetch(`${base}/model.slett`)).arrayBuffer());
            assert.equal(model.layers.length, 6);
            model.free();
        }
        await assert.rejects(loadEngine(`${base}/missing.wasm`), /HTTP 404/);
    } finally {
        server.closeAllConnections();
        server.close();
    }
});

test('a model reports its sizes, loss, format and layers', () => {
    const mlp = engine.loadModel(runtime('mlp_int8'));
    assert.equal(mlp.inputSize, 64);
    assert.equal(mlp.outputSize, 10);
    assert.equal(mlp.layerCount, 4);
    assert.equal(mlp.loss, 'cross_entropy');
    assert.equal(mlp.formatVersion, 3);
    assert.equal(mlp.imageSize, runtime('mlp_int8').length);
    assert.ok(mlp.workspaceSize > 0);
    assert.deepEqual(mlp.layers.map((l) => [l.type, l.inputs, l.outputs, l.activation, l.precision]), [
        ['dense', 64, 48, 'relu', 'int8'], ['dense', 48, 37, 'tanh', 'int8'], ['dense', 37, 29, 'leaky_relu', 'int8'],
        ['dense', 29, 10, 'softmax', 'int8']]);
    assert.deepEqual(mlp.layers[2].inputLayers, [2]);
    assert.equal(mlp.isLanguageModel, false);
    mlp.free();

    const cnn = engine.loadModel(runtime('cnn_fp32'));
    const conv = cnn.layers[2];
    assert.deepEqual([conv.type, conv.inputShape, conv.outputShape, conv.kernel, conv.stride, conv.padding, conv.groups],
        ['conv2d', [4, 3, 6], [3, 2, 5], [2, 3], [1, 2], [0, 1], 1]);
    assert.ok(Object.isFrozen(cnn.layers) && Object.isFrozen(conv));
    cnn.free();

    const graph = engine.loadModel(runtime('graph5_fp32'));
    assert.equal(graph.formatVersion, 7);
    assert.deepEqual(graph.layers.map((l) => l.type), ['conv2d', 'conv2d', 'conv_transpose2d', 'avg_pool2d', 'upsample',
        'concat', 'conv2d', 'dense']);
    assert.deepEqual(graph.layers[5].inputLayers, [1, 3, 5]);
    graph.free();

    const norm = engine.loadModel(runtime('norm1_fp32'));
    assert.equal(norm.layers[0].type, 'batch_norm');
    assert.equal(norm.layers[0].epsilon, Math.fround(0.001));
    norm.free();

    const llama = engine.loadModel(language('llama_int8'));
    const attention = llama.layers.find((l) => l.type === 'attention');
    assert.deepEqual([attention.heads, attention.kvHeads, attention.ropeTheta, attention.causal], [2, 1, 10000, true]);
    assert.deepEqual([llama.formatVersion, llama.loss, llama.layers[0].type, llama.layers[0].vocabulary,
        llama.layers[0].positions], [8, 'sparse_cross_entropy', 'embedding', 28, false]);
    assert.deepEqual(llama.layers.filter((l) => l.activation === 'silu').length, 2);
    llama.free();
    const gpt = engine.loadModel(language('gpt_fp32'));
    assert.equal(gpt.layers[0].positions, true);
    assert.deepEqual(gpt.layers.filter((l) => l.type === 'layer_norm').length, 5);
    assert.deepEqual(gpt.layers.filter((l) => l.activation.startsWith('gelu')).map((l) => l.activation),
        ['gelu', 'gelu_tanh']);
    gpt.free();
});

test('runs take arrays or Float32Arrays, write into a given output, and batches loop over samples', () => {
    const model = engine.loadModel(runtime('cnn_int8'));
    const x = samples(model, 5), n = model.inputSize, m = model.outputSize;
    const first = model.run(x.subarray(0, n));
    assert.ok(first instanceof Float32Array && first.length === m);
    assert.deepEqual(model.run(Array.from(x.subarray(0, n))), first);
    const out = new Float32Array(m);
    assert.equal(model.run(x.subarray(0, n), out), out);
    assert.deepEqual(out, first);
    const batch = model.runBatch(x);
    assert.equal(batch.length, 5 * m);
    for (let s = 0; s < 5; s++)
        assert.deepEqual(batch.subarray(s * m, (s + 1) * m), model.run(x.subarray(s * n, (s + 1) * n)));
    const nested = [0, 1, 2, 3, 4].map((s) => Array.from(x.subarray(s * n, (s + 1) * n)));
    assert.deepEqual(model.runBatch(nested), batch);
    assert.deepEqual(model.runBatch(Array.from(x)), batch);
    assert.throws(() => model.run(x.subarray(0, n - 1)), RangeError);
    assert.throws(() => model.run(x.subarray(0, n), new Float32Array(m - 1)), RangeError);
    assert.throws(() => model.runBatch(x.subarray(0, n + 1)), RangeError);
    model.free();
    model.free();
    assert.throws(() => model.run(x.subarray(0, n)), /freed/);
});

test('damaged, truncated and unknown images are refused with the engine\'s error codes', () => {
    const image = new Uint8Array(runtime('mlp_int8'));
    const damaged = image.slice();
    damaged[damaged.length - 100] ^= 1;
    const newer = image.slice();
    newer[6] = 9;
    const cases = [[damaged, 2], [image.slice(0, image.length - 16), 3], [newer, 4], [new Uint8Array(64), 2],
        [new Uint8Array(0), 2]];
    for (const [bytes, code] of cases) {
        assert.throws(() => engine.loadModel(bytes), (error) => error instanceof SpingalettError && error.code === code);
    }
    assert.throws(() => engine.loadModel('mlp.slett'), TypeError);
    // a view into a larger buffer, at an offset that is no multiple of 4
    const padded = new Uint8Array(image.length + 3);
    padded.set(image, 3);
    const model = engine.loadModel(padded.subarray(3));
    assert.equal(model.inputSize, 64);
    model.free();
});

test('freed models give their memory back; models of one engine run side by side', () => {
    const bytes = runtime('resnet_int8');
    engine.loadModel(bytes).free();
    const size = engine.memorySize;
    for (let i = 0; i < 50; i++) engine.loadModel(bytes).free();
    assert.equal(engine.memorySize, size, 'memory grows when models are loaded and freed again');

    const a = engine.loadModel(runtime('graph6_int8')), b = engine.loadModel(runtime('graph7_bf16'));
    const xa = samples(a, 3), xb = samples(b, 3);
    const ya = a.runBatch(xa), yb = b.runBatch(xb);
    const sample = (model, x, s) => x.subarray(s * model.inputSize, (s + 1) * model.inputSize);
    const outputs = (model, y, s) => y.subarray(s * model.outputSize, (s + 1) * model.outputSize);
    for (let s = 0; s < 3; s++) {
        assert.deepEqual(b.run(sample(b, xb, s)), outputs(b, yb, s));
        assert.deepEqual(a.run(sample(a, xa, s)), outputs(a, ya, s));
    }
    a.free();
    b.free();
});

test('a language model\'s logits are its window\'s last position, as spingalett_generate() fills it', () => {
    const model = engine.loadModel(language('gpt_int8'));
    assert.deepEqual([model.isLanguageModel, model.contextLength, model.vocabulary], [true, 8, 28]);
    const prompt = tokens('the q');
    const window = new Float32Array(8).fill(-1);
    window.set(prompt);
    assert.deepEqual(model.logits(prompt), model.run(window).slice(4 * 28, 5 * 28));
    const long = tokens('the quick brown');
    assert.deepEqual(model.logits(long), model.run(long.slice(-8)).slice(7 * 28));
    assert.throws(() => model.logits([]), RangeError);
    const mlp = engine.loadModel(runtime('mlp_int8'));
    assert.throws(() => mlp.logits([1]), /no language model/);
    assert.throws(() => mlp.generate([1], 1), /no language model/);
    mlp.free();
    model.free();
});

test('generation: greedy, seeded draws, top-k, stop tokens and onToken', () => {
    const model = engine.loadModel(language('llama_fp32'));
    const prompt = tokens('the quick');
    const greedy = model.generate(prompt, 16);
    const sequence = [...prompt];
    for (let i = 0; i < 16; i++) {
        const logits = model.logits(sequence);
        sequence.push(logits.indexOf(Math.max(...logits)));
    }
    assert.deepEqual(greedy, sequence.slice(prompt.length));
    assert.equal(greedy.map((t) => ' .abcdefghijklmnopqrstuvwxyz'[t]).join(''), ' brown fox jumps');

    const hot = { temperature: 5, seed: 12345 };
    const drawn = model.generate(prompt, 32, hot);
    assert.deepEqual(model.generate(prompt, 32, hot), drawn, 'the same seed draws the same tokens');
    assert.deepEqual(model.generate(prompt, 32, { ...hot, seed: 12345n }), drawn, 'seeds as BigInt');
    assert.notDeepEqual(model.generate(prompt, 32, { ...hot, seed: 54321 }), drawn);
    assert.notDeepEqual(drawn, model.generate(prompt, 32));
    assert.deepEqual(model.generate(prompt, 16, { ...hot, topK: 1 }), greedy, 'top-k 1 is greedy');
    assert.deepEqual(model.generate(prompt, 16, { ...hot, topP: 1e-6 }), greedy, 'a tiny top-p is greedy');
    assert.equal(model.generate(prompt, 32, { temperature: 1 }).length, 32, 'a random seed');

    const space = tokens(' ')[0];
    assert.deepEqual(model.generate(prompt, 16, { stop: [space] }), [space]);
    const seen = [];
    const onToken = (t, i) => {
        seen.push([t, i]);
        return i < 2;
    };
    assert.deepEqual(model.generate(prompt, 16, { onToken }), greedy.slice(0, 3));
    assert.deepEqual(seen, greedy.slice(0, 3).map((t, i) => [t, i]));
    assert.throws(() => model.generate([], 4), RangeError);
    model.free();
});

test('the package\'s version is the library\'s', () => {
    const library = readFileSync(join(ROOT, 'CMakeLists.txt'), 'utf8').match(/project\(Spingalett VERSION ([0-9.]+)/)[1];
    assert.equal(JSON.parse(readFileSync(join(PACKAGE, 'package.json'), 'utf8')).version, library);
});
