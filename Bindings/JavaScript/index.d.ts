// SPDX-License-Identifier: MIT
// Copyright (c) 2026 pka_human (pka_human@proton.me)

/** Spingalett's inference engine in WebAssembly: .slett models run sample by sample, as on a microcontroller. */

/** Where the WebAssembly module comes from: a URL, a path (Node) or a URL string, its bytes, a fetch() response
    (or its promise), or the module compiled. */
export type WasmSource =
    | URL
    | string
    | ArrayBuffer
    | ArrayBufferView
    | Response
    | Promise<Response>
    | WebAssembly.Module;

/** A .slett image: a file's bytes. */
export type ModelBytes = ArrayBuffer | ArrayBufferView;

export type LayerType =
    | 'dense' | 'conv2d' | 'max_pool2d' | 'avg_pool2d' | 'batch_norm' | 'add' | 'concat' | 'global_avg_pool'
    | 'conv_transpose2d' | 'upsample' | 'layer_norm' | 'embedding' | 'attention' | 'rms_norm' | 'multiply';
export type Activation =
    | 'none' | 'sigmoid' | 'relu' | 'tanh' | 'leaky_relu' | 'foo52' | 'softmax' | 'gelu' | 'gelu_tanh' | 'silu';
export type Precision = 'fp32' | 'fp16' | 'bf16' | 'int8' | 'int4' | 'int2' | 'q8_0' | 'q4_0' | 'q4_1' | 'q5_0' | 'q5_1' | 'q2_k' | 'q3_k' | 'q4_k' | 'q5_k' | 'q6_k' | 'iq4_nl' | 'iq4_xs';
export type Loss = 'mse' | 'cross_entropy' | 'sparse_cross_entropy';

/** A weight layer, as spingalett_model_layer() describes it. Shapes are [height, width, channels]
    (channels last), windows [height, width]. */
export interface LayerInfo {
    /** Its index among the weight layers (0: the first hidden layer). */
    readonly index: number;
    readonly type: LayerType;
    /** Units of its input and of its output. */
    readonly inputs: number;
    readonly outputs: number;
    readonly activation: Activation;
    /** How its weights are stored and computed with (layers without weights report what the file holds). */
    readonly precision: Precision;
    readonly inputShape: readonly [number, number, number];
    readonly outputShape: readonly [number, number, number];
    /** Convolutions and pooling: the window, its stride and padding ([0, 0] for other layers). */
    readonly kernel: readonly [number, number];
    readonly stride: readonly [number, number];
    readonly padding: readonly [number, number];
    /** Convolutions: channel groups. */
    readonly groups: number;
    /** Normalizations: added to the variance (or the mean square). */
    readonly epsilon: number;
    /** The layers it reads: 0 is the network's input, i + 1 the output of weight layer i. */
    readonly inputLayers: readonly number[];
    /** Upsampling: how cells are filled. */
    readonly upsample: 'nearest' | 'bilinear';
    /** Embeddings: rows of the table (tokens), and whether a learned vector a position is added. */
    readonly vocabulary: number;
    readonly positions: boolean;
    /** Attention: query heads, key and value heads, the rotary embeddings' base (0: none), causal masking. */
    readonly heads: number;
    readonly kvHeads: number;
    readonly ropeTheta: number;
    readonly causal: boolean;
}

/** Options of Model.generate(), as spingalett_generate() takes them. */
export interface GenerateOptions {
    /** 0 (the default): the most likely token; above 0, a draw from the softmax of the logits over it. */
    temperature?: number;
    /** Draw among the topK most likely tokens only (0: all). */
    topK?: number;
    /** Draw among the most likely tokens whose probabilities add up to topP (0 or 1: all). */
    topP?: number;
    /** Makes draws repeatable (0: a random seed). */
    seed?: number | bigint;
    /** Tokens that end the generation (returned last). */
    stop?: Iterable<number>;
    /** Sees each token as it is chosen; returning false ends the generation. */
    onToken?: (token: number, index: number) => boolean | void;
}

/** An error of the engine, with one of Spingalett.Inference.h's error codes: 1 out of memory, 2 an invalid
    image (damaged, a checksum mismatch, an unsupported layer), 3 a truncated image, 4 an unsupported format
    version. */
export class SpingalettError extends Error {
    readonly code: number;
}

/** An instance of the WebAssembly module, with a memory of its own that holds its models. */
export class Engine {
    private constructor();
    /** The newest .slett format version the engine reads (8). */
    readonly formatVersion: number;
    /** Bytes of the module's memory: it grows as models are loaded, and is reused as they are freed. */
    readonly memorySize: number;
    /** Checks a .slett image (format 3 to 9: header, layer table, bounds and CRC-32 checksums), copies it
        into the engine's memory and opens it. Throws a SpingalettError for an image it refuses. */
    loadModel(bytes: ModelBytes): Model;
}

/** A model open in an engine. */
export class Model {
    private constructor();
    /** Inputs a sample has: height x width x channels values, channels last. */
    readonly inputSize: number;
    /** Outputs a sample has. */
    readonly outputSize: number;
    /** Weight layers: the network's layers but its input. */
    readonly layerCount: number;
    /** The loss the network was trained with. */
    readonly loss: Loss;
    /** Bytes of the workspace the engine keeps for the model. */
    readonly workspaceSize: number;
    /** Bytes of the image. */
    readonly imageSize: number;
    /** The image's .slett format version (3 to 9). */
    readonly formatVersion: number;
    /** The weight layers, first hidden layer first. */
    readonly layers: readonly LayerInfo[];

    /** Runs one sample of inputSize values; the outputSize outputs go to `output`, or to a new array. */
    run(input: ArrayLike<number>, output?: Float32Array): Float32Array;
    /** Runs samples one by one: `inputs` holds them one after the other (count x inputSize values) or is an
        array of samples. Returns their outputs one after the other (count x outputSize values). */
    runBatch(inputs: ArrayLike<number> | ArrayLike<ArrayLike<number>>, output?: Float32Array): Float32Array;

    /** True for a language model: an embedding first, and logits of a vocabulary for each input token. */
    readonly isLanguageModel: boolean;
    /** A language model's window of tokens (inputSize). */
    readonly contextLength: number;
    /** A language model's vocabulary (outputSize / inputSize). */
    readonly vocabulary: number;
    /** The logits of the token after `tokens`: their last contextLength tokens fill the window from its first
        position on, no token (-1) the positions after them, and the last token's position gives the logits. */
    logits(tokens: ArrayLike<number>): Float32Array;
    /** Continues `prompt` by `count` tokens, as spingalett_generate() does; returns the new tokens. */
    generate(prompt: ArrayLike<number>, count: number, options?: GenerateOptions): number[];

    /** Releases the model's memory in the engine (models collected unfreed are released too). */
    free(): void;
}

/** Loads the WebAssembly module: by default spingalett.wasm next to the package's index.js. */
export function loadEngine(source?: WasmSource): Promise<Engine>;

/** Loads a model into `options.engine`, or into a default engine, loaded on first use from `options.wasm`. */
export function loadModel(bytes: ModelBytes, options?: { engine?: Engine; wasm?: WasmSource }): Promise<Model>;
