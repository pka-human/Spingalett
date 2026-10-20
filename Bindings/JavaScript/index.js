// SPDX-License-Identifier: MIT
// Copyright (c) 2026 pka_human (pka_human@proton.me)

/*
 * Spingalett's inference engine (Src/Spingalett.Inference.c) compiled to WebAssembly, for Node and
 * browsers: .slett models of any precision (FP32 down to INT2) run sample by sample, as the engine runs
 * them on a microcontroller, with the same results (integer models bit for bit). The module
 * (spingalett.wasm) imports nothing; models and their workspaces live in its memory, which grows as
 * models are loaded and is reused as they are freed.
 */

const LAYER_TYPES = ['dense', 'conv2d', 'max_pool2d', 'avg_pool2d', 'batch_norm', 'add', 'concat', 'global_avg_pool',
    'conv_transpose2d', 'upsample', 'layer_norm', 'embedding', 'attention', 'rms_norm', 'multiply'];
const ACTIVATIONS = ['none', 'sigmoid', 'relu', 'tanh', 'leaky_relu', 'foo52', 'softmax', 'gelu', 'gelu_tanh', 'silu'];
const PRECISIONS = ['fp32', 'fp16', 'bf16', 'int8', 'int4', 'int2', 'q8_0', 'q4_0', 'q4_1', 'q5_0', 'q5_1', 'q2_k', 'q3_k', 'q4_k', 'q5_k', 'q6_k', 'iq4_nl', 'iq4_xs'];
const LOSSES = ['mse', 'cross_entropy', 'sparse_cross_entropy'];
const UPSAMPLE = ['nearest', 'bilinear'];

/* Spingalett.Inference.h's error codes */
const ERRORS = {
    1: 'out of memory',
    2: 'invalid argument or model contents (a damaged image, a checksum mismatch or an unsupported layer)',
    3: 'the image is truncated',
    4: 'unsupported .slett format version (this engine reads versions 3 to 9)',
};

/** An error of the engine; `code` is one of Spingalett.Inference.h's SPINGALETT_ERR_* codes. */
export class SpingalettError extends Error {
    constructor(code, context) {
        super(`${context}: ${ERRORS[code] ?? `error ${code}`}`);
        this.name = 'SpingalettError';
        this.code = code;
    }
}

const isNode = typeof process !== 'undefined' && process.versions != null && process.versions.node != null;

/* Node's file reading, imported only where it is needed (bundlers leave the specifier alone) */
async function readFile(url) {
    const fs = 'node:fs/promises';
    const { readFile: read } = await import(/* webpackIgnore: true */ /* @vite-ignore */ fs);
    return read(url);
}

async function compile(source) {
    if (source instanceof WebAssembly.Module) return source;
    if (source == null) source = new URL('./spingalett.wasm', import.meta.url);
    if (typeof source === 'string') {
        // an absolute URL, or else a path (Node) or a URL relative to the page (browsers)
        if (/^[a-z][a-z0-9+.-]*:/i.test(source) && !/^[a-z]:[\\/]/i.test(source)) source = new URL(source);
        else if (isNode) return WebAssembly.compile(await readFile(source));
        else source = new URL(source, globalThis.location?.href);
    }
    if (source instanceof URL) {
        if (source.protocol === 'file:') return WebAssembly.compile(await readFile(source));
        source = fetch(source);
    }
    if (typeof Response !== 'undefined' && (source instanceof Response || source instanceof Promise)) {
        const response = await source;
        if (!response.ok) throw new Error(`spingalett: ${response.url}: HTTP ${response.status}`);
        if (WebAssembly.compileStreaming && response.headers.get('Content-Type') === 'application/wasm')
            return WebAssembly.compileStreaming(response);
        return WebAssembly.compile(await response.arrayBuffer());
    }
    if (source instanceof ArrayBuffer || ArrayBuffer.isView(source)) return WebAssembly.compile(source);
    throw new TypeError('spingalett: the module comes as a URL, a path, bytes, a Response or a WebAssembly.Module');
}

/**
 * Loads the WebAssembly module: by default the package's spingalett.wasm (next to this file), or the
 * one given as a URL, a path (Node), a URL string, bytes, a fetch() Response or a compiled
 * WebAssembly.Module. Each engine has a memory of its own; models of one engine run one at a time.
 */
export async function loadEngine(source) {
    const module = await compile(source);
    const instance = await WebAssembly.instantiate(module, {});
    return new Engine(instance);
}

let defaultEngine = null;

/** Loads a model (a .slett image) into the default engine, loaded on first use from `options.wasm`. */
export async function loadModel(bytes, options = {}) {
    if (options.engine) return options.engine.loadModel(bytes);
    if (!defaultEngine) {
        defaultEngine = loadEngine(options.wasm).catch((error) => {
            defaultEngine = null;                       // tried again on the next call
            throw error;
        });
    }
    return (await defaultEngine).loadModel(bytes);
}

/* Releases the memory of models that were not freed before they were collected. */
const finalizer = typeof FinalizationRegistry !== 'undefined'
    ? new FinalizationRegistry(({ engine, handle, image }) => engine._release(handle, image)) : null;

export class Engine {
    #exports;
    #buffer = null;
    #f32 = null;
    #u32 = null;
    #u8 = null;

    constructor(instance) {
        this.#exports = instance.exports;
    }

    /** The newest .slett format version the engine reads. */
    get formatVersion() {
        return this.#exports.spg_format_version();
    }

    /** Bytes of the module's memory (it grows as models are loaded, and never shrinks). */
    get memorySize() {
        return this.#exports.memory.buffer.byteLength;
    }

    /* views of the memory, made again after it grows */
    _views() {
        const buffer = this.#exports.memory.buffer;
        if (buffer !== this.#buffer) {
            this.#buffer = buffer;
            this.#f32 = new Float32Array(buffer);
            this.#u32 = new Uint32Array(buffer);
            this.#u8 = new Uint8Array(buffer);
        }
        return { f32: this.#f32, u32: this.#u32, u8: this.#u8 };
    }

    _call(name, ...args) {
        return this.#exports[name](...args);
    }

    _release(handle, image) {
        this.#exports.spg_model_free(handle);
        this.#exports.spg_free(image);
    }

    /**
     * Checks a .slett image (format version 3 to 9: header, layer table, bounds and CRC-32 checksums)
     * and opens it: the image is copied into the engine's memory, with the model's workspace.
     */
    loadModel(bytes) {
        const data = bytes instanceof ArrayBuffer ? new Uint8Array(bytes)
            : ArrayBuffer.isView(bytes) ? new Uint8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength) : null;
        if (!data) throw new TypeError('spingalett: a model comes as an ArrayBuffer or a typed array of its bytes');
        const e = this.#exports;
        const image = e.spg_alloc(Math.max(data.length, 1)) >>> 0;
        if (!image) throw new SpingalettError(1, 'loading a model');
        this._views().u8.set(data, image);
        const handle = e.spg_model_new(image, data.length) >>> 0;
        if (!handle) {
            const code = e.spg_status();
            e.spg_free(image);
            throw new SpingalettError(code, 'loading a model');
        }
        return new Model(this, handle, image, data.length >= 8 ? data[6] | (data[7] << 8) : 0);
    }
}

function layerInfo(w, f, index) {
    const count = w[19];
    return Object.freeze({
        index,
        type: LAYER_TYPES[w[0]] ?? w[0],
        inputs: w[1],
        outputs: w[2],
        activation: ACTIVATIONS[w[3]] ?? w[3],
        precision: PRECISIONS[w[4]] ?? w[4],
        inputShape: Object.freeze([w[5], w[6], w[7]]),
        outputShape: Object.freeze([w[8], w[9], w[10]]),
        kernel: Object.freeze([w[11], w[12]]),
        stride: Object.freeze([w[13], w[14]]),
        padding: Object.freeze([w[15], w[16]]),
        groups: w[17],
        epsilon: f[18],
        inputLayers: Object.freeze(Array.from(w.subarray(27, 27 + count))),
        upsample: UPSAMPLE[w[20]] ?? w[20],
        vocabulary: w[21],
        heads: w[22],
        kvHeads: w[23],
        ropeTheta: f[24],
        causal: w[25] !== 0,
        positions: w[26] !== 0,
    });
}

const MASK64 = (1n << 64n) - 1n;

function splitmix64(state) {
    state.s = (state.s + 0x9E3779B97F4A7C15n) & MASK64;
    let z = state.s;
    z = ((z ^ (z >> 30n)) * 0xBF58476D1CE4E5B9n) & MASK64;
    z = ((z ^ (z >> 27n)) * 0x94D049BB133111EBn) & MASK64;
    return z ^ (z >> 31n);
}

function randomSeed() {
    const words = new Uint32Array(2);
    if (globalThis.crypto?.getRandomValues) globalThis.crypto.getRandomValues(words);
    else for (let i = 0; i < 2; i++) words[i] = Math.floor(Math.random() * 2 ** 32);
    return (BigInt(words[0]) << 32n) | BigInt(words[1]) | 1n;
}

/* The next token from a row of logits, as spingalett_generate() chooses it (Src/Spingalett.Generate.c):
   the most likely, or a draw from the softmax of the logits over the temperature among the top_k most
   likely and the most likely whose probabilities add up to top_p, in an order fixed by the logits. */
function nextToken(logits, temperature, topK, topP, state) {
    const V = logits.length;
    let best = 0;
    for (let v = 1; v < V; v++) if (logits[v] > logits[best]) best = v;
    if (!(temperature > 0)) return best;
    const inv = 1.0 / temperature, top = logits[best];
    const c = new Array(V);
    for (let v = 0; v < V; v++) c[v] = { p: Math.fround(Math.exp((logits[v] - top) * inv)), token: v };
    c.sort((a, b) => (a.p !== b.p ? (a.p > b.p ? -1 : 1) : a.token - b.token));
    let kept = topK && topK < V ? topK : V;
    let total = 0;
    for (let k = 0; k < kept; k++) total += c[k].p;
    if (topP > 0 && topP < 1) {
        let sum = 0;
        for (let k = 0; k < kept; k++) {
            sum += c[k].p;
            if (sum >= topP * total) {
                kept = k + 1;
                total = sum;
                break;
            }
        }
    }
    const u = Number(splitmix64(state) >> 11n) * (1.0 / 9007199254740992.0) * total;
    let sum = 0;
    for (let k = 0; k < kept; k++) {
        sum += c[k].p;
        if (u < sum) return c[k].token;
    }
    return c[kept - 1].token;
}

/** A model open in an engine. Free it with free() when done (models collected unfreed are released). */
export class Model {
    #engine;
    #handle;
    #image;
    #info;
    #layers = null;

    constructor(engine, handle, image, formatVersion) {
        this.#engine = engine;
        this.#handle = handle;
        this.#image = image;
        const w = engine._views().u32, at = (engine._call('spg_model_info', handle) >>> 0) / 4;
        this.#info = {
            inputSize: w[at], outputSize: w[at + 1], layerCount: w[at + 2], loss: LOSSES[w[at + 3]] ?? w[at + 3],
            workspaceSize: w[at + 4], imageSize: w[at + 5], formatVersion,
        };
        finalizer?.register(this, { engine, handle, image }, this);
    }

    #open() {
        if (!this.#handle) throw new Error('spingalett: the model has been freed');
        return this.#engine._views();
    }

    /** Inputs a sample has (height x width x channels values, channels last). */
    get inputSize() { return this.#info.inputSize; }
    /** Outputs a sample has. */
    get outputSize() { return this.#info.outputSize; }
    /** Weight layers: the network's layers but its input. */
    get layerCount() { return this.#info.layerCount; }
    /** The loss the network was trained with: 'mse', 'cross_entropy' or 'sparse_cross_entropy'. */
    get loss() { return this.#info.loss; }
    /** Bytes of the workspace the engine keeps for the model. */
    get workspaceSize() { return this.#info.workspaceSize; }
    /** Bytes of the .slett image. */
    get imageSize() { return this.#info.imageSize; }
    /** The image's .slett format version (3 to 9). */
    get formatVersion() { return this.#info.formatVersion; }

    /** The weight layers (spingalett_model_layer()), first hidden layer first. */
    get layers() {
        if (!this.#layers) {
            const layers = [];
            for (let i = 0; i < this.layerCount; i++) {
                const { u32 } = this.#open();
                const at = (this.#engine._call('spg_model_layer', this.#handle, i) >>> 0) / 4;
                const words = u32.subarray(at, at + 43);
                layers.push(layerInfo(words, new Float32Array(words.buffer, words.byteOffset, 43), i));
            }
            this.#layers = Object.freeze(layers);
        }
        return this.#layers;
    }

    /**
     * Runs one sample: `input` holds inputSize values (a Float32Array or any array of numbers); the
     * outputSize outputs go to `output` (a Float32Array), or to a new one, which is returned.
     */
    run(input, output) {
        const { inputSize, outputSize } = this.#info;
        if (!input || input.length !== inputSize)
            throw new RangeError(`spingalett: the model takes ${inputSize} inputs a sample, not ${input?.length}`);
        if (output === undefined) output = new Float32Array(outputSize);
        else if (!(output instanceof Float32Array) || output.length < outputSize)
            throw new RangeError(`spingalett: the outputs need a Float32Array of ${outputSize} values`);
        const { f32 } = this.#open();
        const e = this.#engine;
        const x = (e._call('spg_model_input', this.#handle) >>> 0) / 4;
        const y = (e._call('spg_model_output', this.#handle) >>> 0) / 4;
        f32.set(input, x);
        const rc = e._call('spg_model_run', this.#handle);
        if (rc !== 0) throw new SpingalettError(rc, 'running a sample');
        output.set(f32.subarray(y, y + outputSize));
        return output;
    }

    /**
     * Runs a batch sample by sample: `inputs` holds the samples one after the other (count x inputSize
     * values) or is an array of samples. Returns their outputs one after the other (count x outputSize),
     * in `output` when it is given.
     */
    runBatch(inputs, output) {
        const { inputSize, outputSize } = this.#info;
        const nested = inputs.length > 0 && typeof inputs[0] === 'object';
        const count = nested ? inputs.length : inputs.length / inputSize;
        if (!Number.isInteger(count))
            throw new RangeError(`spingalett: ${inputs.length} inputs are no whole number of samples of ${inputSize}`);
        if (output === undefined) output = new Float32Array(count * outputSize);
        else if (!(output instanceof Float32Array) || output.length < count * outputSize)
            throw new RangeError(`spingalett: the outputs need a Float32Array of ${count * outputSize} values`);
        for (let s = 0; s < count; s++) {
            const sample = nested ? inputs[s] : inputs.subarray ? inputs.subarray(s * inputSize, (s + 1) * inputSize)
                : Array.prototype.slice.call(inputs, s * inputSize, (s + 1) * inputSize);
            this.run(sample, output.subarray(s * outputSize, (s + 1) * outputSize));
        }
        return output;
    }

    /** True for a language model: an embedding first, and the logits of a vocabulary for each input token. */
    get isLanguageModel() {
        return this.layerCount > 0 && this.layers[0].type === 'embedding' && this.outputSize % this.inputSize === 0;
    }

    /** A language model's window: the tokens a run reads. */
    get contextLength() { return this.inputSize; }

    /** A language model's vocabulary: the logits it gives for each token. */
    get vocabulary() { return this.outputSize / this.inputSize; }

    #languageModel() {
        if (!this.isLanguageModel)
            throw new Error('spingalett: the model is no language model (an embedding first, logits for every token)');
    }

    /**
     * A language model's logits of the token after `tokens` (token ids): the window holds their last
     * contextLength tokens from its first position on, and the positions after them no token (-1), as
     * spingalett_generate() fills it. Returns a Float32Array of `vocabulary` values.
     */
    logits(tokens) {
        this.#languageModel();
        const T = this.contextLength, V = this.vocabulary, n = Math.min(tokens.length, T);
        if (n === 0) throw new RangeError('spingalett: logits need one token at least');
        const window = new Float32Array(T).fill(-1);
        for (let i = 0; i < n; i++) window[i] = tokens[tokens.length - n + i];
        const out = this.run(window);
        return out.slice((n - 1) * V, n * V);
    }

    /**
     * Continues `prompt` (token ids) by `count` tokens, a token at a time, as spingalett_generate() does:
     * the most likely token (temperature 0), or a draw from the softmax of the logits over `temperature`
     * among the `topK` most likely and those whose probabilities add up to `topP`. `seed` (a number or
     * a BigInt; 0: a random one) makes draws repeatable, a token of `stop` ends the generation (it is
     * returned last), and `onToken(token, index)` sees each token as it comes (false stops). Returns the
     * new tokens.
     */
    generate(prompt, count, { temperature = 0, topK = 0, topP = 0, seed = 0, stop = [], onToken } = {}) {
        this.#languageModel();
        if (!prompt || prompt.length === 0)
            throw new RangeError('spingalett: generate needs a prompt of one token at least');
        const state = { s: BigInt(seed) & MASK64 };
        if (state.s === 0n) state.s = randomSeed();
        const t = Math.fround(temperature), p = Math.fround(topP), stops = new Set(stop);
        const sequence = Array.from(prompt), made = [];
        while (made.length < count) {
            const token = nextToken(this.logits(sequence), t, topK, p, state);
            sequence.push(token);
            made.push(token);
            if (onToken && onToken(token, made.length - 1) === false) break;
            if (stops.has(token)) break;
        }
        return made;
    }

    /** Releases the model's memory in the engine; the model cannot run afterwards. */
    free() {
        if (!this.#handle) return;
        finalizer?.unregister(this);
        this.#engine._release(this.#handle, this.#image);
        this.#handle = 0;
        this.#image = 0;
    }
}

/* `using model = engine.loadModel(bytes)` frees the model at the end of the block, where the language
   has explicit resource management. */
if (typeof Symbol.dispose === 'symbol') Model.prototype[Symbol.dispose] = function dispose() { this.free(); };
