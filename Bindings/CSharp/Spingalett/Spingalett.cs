// SPDX-License-Identifier: MIT
// Copyright (c) 2026 pka_human (pka_human@proton.me)

// Spingalett for .NET: networks with dense, convolutional and transformer layers, trained by libspingalett on
// the CPU or the GPU (CUDA, Vulkan), deployment models from FP32 to INT2, and text generation.

using System.Runtime.InteropServices;
using Spingalett.Native;

namespace Spingalett;

public enum Activation { None = 0, Sigmoid = 1, Relu = 2, Tanh = 3, LeakyRelu = 4, Foo52 = 5, Softmax = 6, Gelu = 7, GeluTanh = 8, Silu = 9 }
public enum Loss { Mse = 0, CrossEntropy = 1, SparseCrossEntropy = 2 }
public enum ComputeMode { SingleThreaded = 0, OpenMp = 1, OpenBlas = 2, Cuda = 3, Vulkan = 4 }
public enum Precision { Float32 = 0, Fp16 = 1, BFloat16 = 2, Int8 = 3, Int4 = 4, Int2 = 5 }
public enum Optimizer { Sgd = 0, Momentum = 1, RmsProp = 2, Adam = 3, AdamW = 4 }
public enum Strategy { Sample = 0, FullBatch = 1, SmallBatch = 2 }
public enum Init { Random = 0, Xavier = 1, He = 2, Zeros = 3, LeCun = 4 }

/// <summary>An error the library reported: its code (SPINGALETT_ERR_*) and message.</summary>
public sealed class SpingalettException : Exception
{
    public int Code { get; }
    public SpingalettException(int code, string message) : base($"{message} (code {code})") { Code = code; }

    internal static SpingalettException Last(string what)
    {
        int code = Lib.spingalett_last_error_code();
        string? message = Marshal.PtrToStringUTF8(Lib.spingalett_last_error_message());
        return new SpingalettException(code != 0 ? code : -1, string.IsNullOrEmpty(message) ? what : message);
    }
}

/// <summary>The library's settings.</summary>
public static class Library
{
    public static string Version => Marshal.PtrToStringUTF8(Lib.spingalett_version()) ?? "";
    /// <summary>Where training and inference run; false when the mode is not available.</summary>
    public static bool SetComputeMode(ComputeMode mode) => Lib.spingalett_set_compute_mode((int)mode);
    /// <summary>Precision of the GPU's products (Float32 or BFloat16); false when the GPU has none such.</summary>
    public static bool SetGpuPrecision(Precision precision) => Lib.spingalett_set_gpu_precision((int)precision);
    public static void Seed(ulong seed) => Lib.spingalett_seed(seed);
    public static void SetVerbose(bool enabled) => Lib.spingalett_set_verbose(enabled);
    public static void SetThreads(uint threads) => Lib.spingalett_set_num_threads(threads);
    public static string? CudaDevice => Marshal.PtrToStringUTF8(Lib.spingalett_cuda_device());
    public static string? GpuDevice => Marshal.PtrToStringUTF8(Lib.spingalett_gpu_device());
}

/// <summary>A layer to add to a network: a kind with its sizes, then options (each returns the layer).</summary>
public sealed unsafe class Layer
{
    internal LayerArgs Args;

    Layer(int type) { Args.type = type; }

    public static Layer Input(uint height, uint width, uint channels) =>
        new Layer(0) { Args = { height = height, width = width, channels = channels } };
    public static Layer Dense(uint neurons) => new Layer(0) { Args = { neurons_amount = neurons, weight_initialization = (int)Init.LeCun } };
    public static Layer Conv2D(uint filters, uint kernel) =>
        new Layer(1) { Args = { filters = filters, kernel = kernel, act_func = (int)Activation.Relu, weight_initialization = (int)Init.He } };
    public static Layer ConvTranspose2D(uint filters, uint kernel)
    {
        var l = Conv2D(filters, kernel);
        l.Args.type = 8;
        return l;
    }
    public static Layer MaxPool2D(uint kernel) => new Layer(2) { Args = { kernel = kernel } };
    public static Layer AvgPool2D(uint kernel) => new Layer(3) { Args = { kernel = kernel } };
    public static Layer GlobalAvgPool() => new Layer(7);
    public static Layer BatchNorm() => new Layer(4);
    public static Layer LayerNorm() => new Layer(10);
    public static Layer RmsNorm() => new Layer(13);
    public static Layer AddLayers(params uint[] inputs) => new Layer(5).Inputs(inputs);
    public static Layer Concat(params uint[] inputs) => new Layer(6).Inputs(inputs);
    public static Layer Multiply(params uint[] inputs) => new Layer(14).Inputs(inputs);
    public static Layer Upsample(uint factor) => new Layer(9) { Args = { stride = factor } };
    /// <summary>A linear map of each cell's channels (a 1 x 1 convolution: a transformer's projections).</summary>
    public static Layer Linear(uint neurons) =>
        new Layer(1) { Args = { filters = neurons, kernel = 1, weight_initialization = (int)Init.LeCun } };
    /// <summary>The vectors of `width` values of a vocabulary's tokens, a token a cell.</summary>
    public static Layer Embedding(uint vocabulary, uint width) =>
        new Layer(11) { Args = { vocabulary = vocabulary, neurons_amount = width, weight_initialization = (int)Init.LeCun } };
    /// <summary>Attention of `heads` query heads over packed queries, keys and values.</summary>
    public static Layer Attention(uint heads) => new Layer(12) { Args = { heads = heads } };

    public Layer WithActivation(Activation activation) { Args.act_func = (int)activation; return this; }
    public Layer WithInit(Init init) { Args.weight_initialization = (int)init; return this; }
    public Layer Dropout(float rate) { Args.dropout_rate = rate; return this; }
    public Layer Stride(uint stride) { Args.stride = stride; return this; }
    public Layer Padding(uint padding) { Args.padding = padding; return this; }
    public Layer Groups(uint groups) { Args.groups = groups; return this; }
    public Layer Epsilon(float epsilon) { Args.epsilon = epsilon; return this; }
    public Layer OutputPadding(uint padding) { Args.output_padding = padding; return this; }
    public Layer Bilinear() { Args.upsample = 1; return this; }
    public Layer KvHeads(uint heads) { Args.kv_heads = heads; return this; }
    public Layer Causal(bool causal = true) { Args.causal = (byte)(causal ? 1 : 0); return this; }
    public Layer RopeTheta(float theta) { Args.rope_theta = theta; return this; }
    public Layer Positions(bool positions = true) { Args.positions = (byte)(positions ? 1 : 0); return this; }
    /// <summary>The layers it reads (indices that Network.Add returned); none: the last one.</summary>
    public Layer Inputs(params uint[] inputs)
    {
        if (inputs.Length > Const.MaxInputs) throw new ArgumentException($"a layer reads at most {Const.MaxInputs} layers");
        for (int k = 0; k < inputs.Length; k++) Args.inputs[k] = inputs[k];
        Args.input_count = (uint)inputs.Length;
        return this;
    }
}

/// <summary>Options of Network.Train (the defaults: 10 epochs of mini-batches of 32, Adam at 0.001).</summary>
public sealed record TrainOptions
{
    public int Epochs { get; init; } = 10;
    public uint BatchSize { get; init; } = 32;
    public Strategy Strategy { get; init; } = Strategy.SmallBatch;
    public Optimizer Optimizer { get; init; } = Optimizer.Adam;
    public float LearningRate { get; init; } = 1e-3f;
    public float WeightDecay { get; init; }
    public float Momentum { get; init; }
    public float Beta1 { get; init; }
    public float Beta2 { get; init; }
    public float MaxGradNorm { get; init; }
    public float LabelSmoothing { get; init; }
    public bool NoShuffle { get; init; }
}

public readonly record struct TrainResult(bool Completed, int EpochsRun, float TrainLoss);
public readonly record struct Metrics(float Loss, float Accuracy);

/// <summary>How Network.Generate picks each token (the defaults: the most likely one).</summary>
public sealed record Sampling
{
    public float Temperature { get; init; }
    public uint TopK { get; init; }
    public float TopP { get; init; }
    public ulong Seed { get; init; }
    public uint[] Stop { get; init; } = Array.Empty<uint>();
}

/// <summary>A network: its layers, parameters and training state. Used by one thread at a time.</summary>
public sealed unsafe class Network : IDisposable
{
    nint _ptr;

    Network(nint ptr) { _ptr = ptr; }

    /// <summary>An empty network that trains with `loss`; its first layer is the input layer.</summary>
    public Network(Loss loss)
    {
        var args = new NetworkArgs { loss_func = (int)loss };
        Lib.spingalett_clear_error();
        _ptr = Lib.spingalett_network_new_args(args);
        if (_ptr == 0) throw SpingalettException.Last("network allocation failed");
    }

    /// <summary>A network from a .slett file.</summary>
    public static Network Load(string path)
    {
        nint ptr = Lib.spingalett_load(path);
        return ptr != 0 ? new Network(ptr) : throw SpingalettException.Last("load failed");
    }

    nint Ptr => _ptr != 0 ? _ptr : throw new ObjectDisposedException(nameof(Network));

    /// <summary>Appends a layer; returns its index (what later layers' Inputs name).</summary>
    public uint Add(Layer layer)
    {
        var args = layer.Args;
        args.net = Ptr;
        Lib.spingalett_clear_error();
        uint index = Lib.spingalett_append_layer(args);
        return index != Const.NoLayer ? index : throw SpingalettException.Last("the layer does not fit the network");
    }

    public uint LayerCount => Lib.spingalett_layer_count(Ptr);
    public uint InputSize => Lib.spingalett_input_size(Ptr);
    public uint OutputSize => Lib.spingalett_output_size(Ptr);
    /// <summary>Targets a sample (the outputs, or one class index a cell with the sparse cross-entropy).</summary>
    public uint TargetSize => Lib.spingalett_target_size(Ptr);
    public ulong ParameterCount => Lib.spingalett_parameter_count(Ptr);

    static uint Samples(int values, uint size, string what) =>
        size != 0 && values % size == 0 ? (uint)(values / size) : throw new ArgumentException($"{what}: {values} values are no whole number of samples of {size}");

    TrainArgs Options(TrainOptions o) => new TrainArgs
    {
        net = Ptr,
        training_strategy = (int)o.Strategy,
        optimizer_type = (int)o.Optimizer,
        batch_size = o.BatchSize,
        do_not_shuffle = (byte)(o.NoShuffle ? 1 : 0),
        epochs = (nuint)o.Epochs,
        learning_rate = o.LearningRate,
        weight_decay = o.WeightDecay,
        momentum = o.Momentum,
        beta1 = o.Beta1,
        beta2 = o.Beta2,
        max_grad_norm = o.MaxGradNorm,
        label_smoothing = o.LabelSmoothing,
    };

    static TrainResult Run(TrainArgs args)
    {
        var r = Lib.spingalett_train_args(args);
        if (r.status == 0 || r.status == 5) throw SpingalettException.Last("training failed");
        return new TrainResult(r.status == 1, (int)r.epochs_run, r.train_loss);
    }

    /// <summary>Trains on samples in arrays: InputSize values a sample of inputs, TargetSize of targets.</summary>
    public TrainResult Train(float[] inputs, float[] targets, TrainOptions? options = null)
    {
        uint n = Samples(inputs.Length, InputSize, "Train");
        if (Samples(targets.Length, TargetSize, "Train") != n) throw new ArgumentException("as many samples of targets as of inputs");
        fixed (float* x = inputs, y = targets)
        {
            var args = Options(options ?? new TrainOptions());
            args.inputs = (nint)x;
            args.targets = (nint)y;
            args.sample_count = n;
            return Run(args);
        }
    }

    /// <summary>Trains a language model on the windows of a file of token ids (nanoGPT's .bin, llm.c's).</summary>
    public TrainResult TrainTokens(string path, uint stride = 0, TrainOptions? options = null)
    {
        var o = options ?? new TrainOptions();
        var reader_options = new TokenReaderOptions { context = InputSize, stride = stride, shuffle = (byte)(o.NoShuffle ? 0 : 1) };
        nint reader = Lib.spingalett_dataset_open_tokens(path, &reader_options);
        if (reader == 0) throw SpingalettException.Last("spingalett_dataset_open_tokens failed");
        try
        {
            var args = Options(o);
            args.training_mode = 1;
            args.generator = Lib.DatasetGenerator;
            args.generator_data = reader;
            args.sample_count = Lib.spingalett_dataset_info(reader).count;
            return Run(args);
        }
        finally
        {
            Lib.spingalett_dataset_close(reader);
        }
    }

    /// <summary>The outputs of samples of InputSize values each.</summary>
    public float[] Predict(float[] inputs)
    {
        uint n = Samples(inputs.Length, InputSize, "Predict");
        var outputs = new float[n * OutputSize];
        fixed (float* x = inputs, y = outputs)
        {
            var args = new PredictArgs { net = Ptr, inputs = (nint)x, sample_count = n, outputs = (nint)y };
            if (!Lib.spingalett_predict_args(args)) throw SpingalettException.Last("predict failed");
        }
        return outputs;
    }

    /// <summary>Mean loss and accuracy over samples.</summary>
    public Metrics Evaluate(float[] inputs, float[] targets)
    {
        uint n = Samples(inputs.Length, InputSize, "Evaluate");
        if (Samples(targets.Length, TargetSize, "Evaluate") != n) throw new ArgumentException("as many samples of targets as of inputs");
        fixed (float* x = inputs, y = targets)
        {
            var m = Lib.spingalett_evaluate_args(new EvaluateArgs { net = Ptr, inputs = (nint)x, targets = (nint)y, sample_count = n });
            return float.IsNaN(m.loss) ? throw SpingalettException.Last("evaluate failed") : new Metrics(m.loss, m.accuracy);
        }
    }

    /// <summary>Continues a prompt of token ids by `count` tokens with a causal language model.</summary>
    public uint[] Generate(uint[] prompt, uint count, Sampling? sampling = null)
    {
        if (prompt.Length == 0) throw new ArgumentException("a prompt of one token at least");
        var s = sampling ?? new Sampling();
        var tokens = new uint[count];
        fixed (uint* p = prompt, t = tokens, stop = s.Stop)
        {
            var args = new GenerateArgs
            {
                net = Ptr, prompt = (nint)p, prompt_length = (uint)prompt.Length, tokens = (nint)t, count = count,
                temperature = s.Temperature, top_k = s.TopK, top_p = s.TopP, seed = s.Seed,
                stop_tokens = s.Stop.Length > 0 ? (nint)stop : 0, stop_count = (uint)s.Stop.Length,
            };
            Lib.spingalett_clear_error();
            uint made = Lib.spingalett_generate_args(args);
            if (Lib.spingalett_last_error_code() != 0) throw SpingalettException.Last("generate failed");
            return tokens[..(int)made];
        }
    }

    /// <summary>Saves the network (precision of the weights in the file).</summary>
    public void Save(string path, Precision precision = Precision.Float32)
    {
        nint name = Marshal.StringToCoTaskMemUTF8(path);
        try
        {
            if (!Lib.spingalett_save_args(new SaveArgs { net = Ptr, filename = name, precision = (int)precision }))
                throw SpingalettException.Last("save failed");
        }
        finally
        {
            Marshal.FreeCoTaskMem(name);
        }
    }

    NetworkLayer Describe(uint layer)
    {
        NetworkLayer info;
        return Lib.spingalett_network_layer(Ptr, layer, &info) ? info : throw SpingalettException.Last("no such layer");
    }

    float[] Read(uint layer, int kind, ulong count)
    {
        var values = new float[count];
        fixed (float* v = values)
            if (count > 0 && !Lib.spingalett_get_parameters(Ptr, layer, kind, v, count)) throw SpingalettException.Last("get_parameters failed");
        return values;
    }

    /// <summary>The weights of layer `layer` (rows of outputs, channels last).</summary>
    public float[] Weights(uint layer) => Read(layer, 0, Describe(layer).weight_count);
    public float[] Biases(uint layer) => Read(layer, 1, Describe(layer).bias_count);

    public void SetParameters(uint layer, bool biases, float[] values)
    {
        fixed (float* v = values)
            if (!Lib.spingalett_set_parameters(Ptr, layer, biases ? 1 : 0, v, (ulong)values.Length)) throw SpingalettException.Last("set_parameters failed");
    }

    /// <summary>The network as a deployment model of `precision`.</summary>
    public Model ToModel(Precision precision)
    {
        nint model = Lib.spingalett_model_from_network(Ptr, (int)precision);
        return model != 0 ? new Model(model) : throw SpingalettException.Last("spingalett_model_from_network failed");
    }

    public void Dispose()
    {
        if (_ptr != 0) Lib.spingalett_network_free(_ptr);
        _ptr = 0;
        GC.SuppressFinalize(this);
    }

    ~Network() { if (_ptr != 0) Lib.spingalett_network_free(_ptr); }
}

/// <summary>A deployment model: inference only, in its weights' precision; several threads may share one.</summary>
public sealed unsafe class Model : IDisposable
{
    nint _ptr;

    internal Model(nint ptr) { _ptr = ptr; }

    public static Model Load(string path)
    {
        nint ptr = Lib.spingalett_model_load(path);
        return ptr != 0 ? new Model(ptr) : throw SpingalettException.Last("spingalett_model_load failed");
    }

    nint Ptr => _ptr != 0 ? _ptr : throw new ObjectDisposedException(nameof(Model));
    public uint InputSize => ((Native.Model*)Ptr)->input_size;
    public uint OutputSize => ((Native.Model*)Ptr)->output_size;

    public float[] Predict(float[] inputs)
    {
        uint size = InputSize;
        if (size == 0 || inputs.Length % size != 0) throw new ArgumentException($"{inputs.Length} values are no whole number of samples of {size}");
        uint n = (uint)(inputs.Length / size);
        var outputs = new float[n * OutputSize];
        fixed (float* x = inputs, y = outputs)
            if (!Lib.spingalett_model_predict(Ptr, x, n, y)) throw SpingalettException.Last("model predict failed");
        return outputs;
    }

    public void Dispose()
    {
        if (_ptr != 0) Lib.spingalett_model_free(_ptr);
        _ptr = 0;
        GC.SuppressFinalize(this);
    }

    ~Model() { if (_ptr != 0) Lib.spingalett_model_free(_ptr); }
}
