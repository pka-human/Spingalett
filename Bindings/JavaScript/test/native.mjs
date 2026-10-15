// SPDX-License-Identifier: MIT
// Copyright (c) 2026 pka_human (pka_human@proton.me)

/*
 * What the tests and make_references.mjs share: the models they run, their samples (as test/native.c
 * makes them), and the native engine, test/native.c with Src/Spingalett.Inference.c built by the host's
 * C compiler (CC, else cc), as the CMake engine target builds it:
 *
 *   'system'    the engine's kernels for the host and the C library's mathematical functions: the native
 *               engine, whose results test/references holds
 *   'portable'  the portable kernels and the package's mathematical functions (wasm/Spingalett.Libm.c):
 *               the code the WebAssembly module runs, compiled for the host, which must give its bits
 */

import { execFileSync } from 'node:child_process';
import { mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

export const HERE = dirname(fileURLToPath(import.meta.url));
export const PACKAGE = join(HERE, '..');
export const ROOT = join(HERE, '..', '..', '..');

const RUNTIME = ['mlp_fp16', 'mlp_int8', 'mlp_int4', 'mlp_int2', 'cnn_fp32', 'cnn_int8', 'conv9_int8', 'norm1_fp32',
    'norm3_int4', 'conv5_bf16', 'conv7_int2', 'conv8_int8', 'graph0_int8', 'graph1_fp16', 'graph3_fp32', 'resnet_int8',
    'graph5_fp32', 'graph5_int4', 'graph6_int8', 'graph7_bf16'];

/* The models of earlier releases (Tests/Data/runtime) and the language models of test/models, with the
   samples each runs. */
export const MODELS = [
    ...RUNTIME.map((name) => ({ name, path: join(ROOT, 'Tests', 'Data', 'runtime', `${name}.slett`), count: 16 })),
    ...['gpt_fp32', 'gpt_int8', 'llama_fp32', 'llama_int8'].map((name) =>
        ({ name, path: join(HERE, 'models', `${name}.slett`), count: 8 })),
];

/* Generations the tests repeat: prompts of the language models' alphabet (make_models.py). */
const ALPHABET = ' .abcdefghijklmnopqrstuvwxyz';
export const tokens = (text) => [...text].map((c) => ALPHABET.indexOf(c));
export const GENERATIONS = [
    { model: 'gpt_int8', prompt: tokens('the q'), count: 24, temperature: 0, topK: 0, topP: 0, seed: 1 },
    { model: 'llama_int8', prompt: tokens('over the'), count: 24, temperature: 0, topK: 0, topP: 0, seed: 1 },
    { model: 'gpt_fp32', prompt: tokens('jumps'), count: 24, temperature: 2.5, topK: 12, topP: 0.95, seed: 42 },
    { model: 'llama_int8', prompt: tokens('a'), count: 24, temperature: 3, topK: 0, topP: 0, seed: 7 },
    { model: 'llama_fp32', prompt: tokens('lazy'), count: 24, temperature: 1.5, topK: 5, topP: 0, seed: 2026 },
];

/** True when the model counts as an integer model: its first layer's weights are integers. */
export const isInteger = (model) => model.layers[0].precision.startsWith('int');

/** `count` samples for the model, one after the other, as test/native.c makes them. */
export function samples(model, count) {
    let state = 2026;
    const next = () => {
        state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
        return state >>> 8;
    };
    const n = model.inputSize, first = model.layers[0];
    const vocabulary = first.type === 'embedding' ? first.vocabulary : 0;
    const x = new Float32Array(count * n);
    for (let s = 0; s < count; s++) {
        const sample = x.subarray(s * n, (s + 1) * n);
        if (vocabulary) {
            const length = 1 + (next() % n);
            for (let i = 0; i < n; i++) sample[i] = i < length ? next() % vocabulary : -1;
        } else {
            for (let i = 0; i < n; i++) sample[i] = next() / 8388608 - 1;
        }
    }
    return x;
}

/** The C compiler's command, or null when there is none. */
export function compiler() {
    const cc = process.env.CC || 'cc';
    try {
        execFileSync(cc, ['--version'], { stdio: 'ignore' });
        return cc;
    } catch {
        return null;
    }
}

/** Builds the native engine of `kind` ('system' or 'portable'); returns { exe, dispose }, or null without a
    C compiler. */
export function buildNative(kind) {
    const cc = compiler();
    if (!cc) return null;
    const dir = mkdtempSync(join(tmpdir(), 'spingalett-native-'));
    const exe = join(dir, process.platform === 'win32' ? 'native.exe' : 'native');
    const args = ['-std=c99', '-O2', '-ffp-contract=off', '-DSPINGALETT_INFERENCE_ONLY', `-I${join(ROOT, 'Include')}`,
        `-I${join(ROOT, 'Src')}`, join(ROOT, 'Src', 'Spingalett.Inference.c'), join(HERE, 'native.c')];
    if (kind === 'portable')
        args.push('-DSPINGALETT_PORTABLE_KERNELS', '-fno-builtin', join(PACKAGE, 'wasm', 'Spingalett.Libm.c'));
    args.push('-o', exe, '-lm');
    execFileSync(cc, args, { stdio: ['ignore', 'inherit', 'inherit'] });
    return { exe, dir, dispose: () => rmSync(dir, { recursive: true, force: true }) };
}

/** The native engine's outputs of the model's samples. */
export function runNative(native, path, count) {
    const out = join(native.dir, 'outputs');
    execFileSync(native.exe, ['run', path, String(count), out]);
    const bytes = readFileSync(out);
    return new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
}

/** The tokens the native engine generates. */
export function generateNative(native, path, g) {
    const args = ['generate', path, g.count, g.temperature, g.topK, g.topP, g.seed, ...g.prompt].map(String);
    const text = execFileSync(native.exe, args, { encoding: 'utf8' }).trim();
    return text ? text.split(' ').map(Number) : [];
}

/** How far y is from the reference: the largest difference of an output from its reference relative to the
    largest magnitude of its sample's references (at least 1e-3), and how many outputs differ at all. */
export function compare(y, reference, outputSize) {
    let worst = 0, differ = 0;
    for (let s = 0; s < reference.length; s += outputSize) {
        let scale = 1e-3, d = 0;
        for (let i = s; i < s + outputSize; i++) {
            scale = Math.max(scale, Math.abs(reference[i]));
            if (Object.is(y[i], reference[i])) continue;
            differ++;
            d = Math.max(d, Number.isNaN(y[i] - reference[i]) ? Infinity : Math.abs(y[i] - reference[i]));
        }
        worst = Math.max(worst, d / scale);
    }
    return { worst, differ };
}
