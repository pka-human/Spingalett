// SPDX-License-Identifier: MIT
// Copyright (c) 2026 pka_human (pka_human@proton.me)

// The C API of libspingalett that the bindings use (Include/Spingalett/Spingalett.h): the argument
// structures laid out as in C (the tests compare every offset with the C compiler's; C's bool is a byte
// here), and the functions.

using System.Runtime.InteropServices;

namespace Spingalett.Native;

public static class Const
{
    public const int Reserved = 8;
    public const int MaxInputs = 16;
    public const uint NoLayer = uint.MaxValue;
}

[StructLayout(LayoutKind.Sequential)]
public unsafe struct EvalMetrics
{
    public float loss;
    public float accuracy;
    public fixed ulong reserved[Const.Reserved];
}

[StructLayout(LayoutKind.Sequential)]
public unsafe struct NetworkArgs
{
    public int loss_func;
    public fixed ulong reserved[Const.Reserved];
}

[StructLayout(LayoutKind.Sequential)]
public unsafe struct LayerArgs
{
    public nint net;
    public uint neurons_amount;
    public int act_func;
    public int weight_initialization;
    public float dropout_rate;
    public int type;
    public uint height, width, channels;
    public uint filters;
    public uint kernel;
    public uint stride;
    public uint padding;
    public uint kernel_h, kernel_w;
    public uint stride_h, stride_w;
    public uint padding_h, padding_w;
    public uint groups;
    public float epsilon;
    public float momentum;
    public fixed uint inputs[Const.MaxInputs];
    public uint input_count;
    public int upsample;
    public uint output_padding;
    public uint output_padding_h, output_padding_w;
    public uint vocabulary;
    public uint heads;
    public uint kv_heads;
    public float rope_theta;
    public byte causal;
    public byte positions;
    public fixed ulong reserved[Const.Reserved - 3];
}

[StructLayout(LayoutKind.Sequential)]
public unsafe struct NetworkLayer
{
    public int type;
    public uint height, width, channels;
    public uint outputs;
    public int activation;
    public float dropout_rate;
    public uint kernel_h, kernel_w;
    public uint stride_h, stride_w;
    public uint padding_h, padding_w;
    public ulong weight_count;
    public ulong bias_count;
    public uint groups;
    public float epsilon;
    public float momentum;
    public uint input_count;
    public fixed uint inputs[Const.MaxInputs];
    public int upsample;
    public uint vocabulary;
    public uint heads, kv_heads;
    public float rope_theta;
    public byte causal;
    public byte positions;
    public fixed ulong reserved[Const.Reserved - 2];
}

[StructLayout(LayoutKind.Sequential)]
public unsafe struct TrainArgs
{
    public nint net;
    public int training_mode;
    public int training_strategy;
    public int optimizer_type;
    public nint inputs;
    public nint targets;
    public nint device_inputs;
    public nint device_targets;
    public nint generator;
    public nint generator_data;
    public uint sample_count;
    public uint batch_size;
    public byte do_not_shuffle;
    public nuint epochs;
    public float learning_rate;
    public float weight_decay;
    public float momentum;
    public float beta1;
    public float beta2;
    public float epsilon;
    public float max_grad_norm;
    public byte reset_optimizer;
    public nuint nan_check_interval;
    public nuint report_interval;
    public int autosave_mode;
    public nuint autosave_interval;
    public nint autosave_path;
    public byte autosave_do_not_save_optimizer;
    public int autosave_precision;
    public nint callback;
    public nuint callback_interval;
    public nint callback_data;
    public nint lr_scheduler;
    public nint lr_scheduler_data;
    public nint val_inputs;
    public nint val_targets;
    public nint device_val_inputs;
    public nint device_val_targets;
    public uint val_count;
    public int monitor;
    public nuint early_stopping_patience;
    public float early_stopping_min_delta;
    public byte restore_best_weights;
    public int blas_num_threads;
    public uint augment_shift;
    public byte augment_flip;
    public float label_smoothing;
    public float lr_plateau_factor;
    public nuint lr_plateau_patience;
    public float lr_plateau_min_lr;
    public fixed ulong reserved[Const.Reserved];
}

[StructLayout(LayoutKind.Sequential)]
public unsafe struct TrainReport
{
    public int status;
    public nuint epochs_run;
    public float train_loss;
    public byte has_validation;
    public EvalMetrics validation;
    public int monitor;
    public nuint best_epoch;
    public float best_value;
    public byte restored_best;
    public fixed ulong reserved[Const.Reserved];
}

[StructLayout(LayoutKind.Sequential)]
public unsafe struct PredictArgs
{
    public nint net;
    public nint inputs;
    public nint device_inputs;
    public uint sample_count;
    public nint outputs;
    public fixed ulong reserved[Const.Reserved];
}

[StructLayout(LayoutKind.Sequential)]
public unsafe struct EvaluateArgs
{
    public nint net;
    public nint inputs;
    public nint targets;
    public nint device_inputs;
    public nint device_targets;
    public uint sample_count;
    public fixed ulong reserved[Const.Reserved];
}

[StructLayout(LayoutKind.Sequential)]
public unsafe struct SaveArgs
{
    public nint net;
    public nint filename;
    public byte do_not_save_optimizer;
    public int precision;
    public fixed ulong reserved[Const.Reserved];
}

[StructLayout(LayoutKind.Sequential)]
public unsafe struct GenerateArgs
{
    public nint net;
    public nint prompt;
    public uint prompt_length;
    public nint tokens;
    public uint count;
    public float temperature;
    public uint top_k;
    public float top_p;
    public ulong seed;
    public nint stop_tokens;
    public uint stop_count;
    public fixed ulong reserved[Const.Reserved];
}

[StructLayout(LayoutKind.Sequential)]
public unsafe struct TokenReaderOptions
{
    public uint context;
    public uint token_bytes;
    public uint stride;
    public ulong offset;
    public byte shuffle;
    public fixed ulong reserved[Const.Reserved];
}

[StructLayout(LayoutKind.Sequential)]
public unsafe struct DatasetInfo
{
    public uint count, input_size, target_size;
    public int input_encoding, target_encoding;
    public uint chunk_count;
    public ulong file_size;
    public uint format_version;
    public uint height, width, channels;
    public uint target_set_count;
    public uint target_set;
    public fixed ulong reserved[Const.Reserved];
}

/// <summary>A deployment model's public fields (the rest is the library's).</summary>
[StructLayout(LayoutKind.Sequential)]
public unsafe struct Model
{
    public uint input_size;
    public uint output_size;
    public uint layer_count;
    public int loss;
    public nuint workspace_size;
    public nint image;
    public nuint image_size;
    public uint max_width_;
    public uint max_int_inputs_;
    public nuint conv_scratch_;
    public nint owner_;
    public nuint activations_;
    public fixed ulong reserved[Const.Reserved];
}

public static unsafe partial class Lib
{
    const string Name = "spingalett";

    [LibraryImport(Name)] public static partial nint spingalett_version();
    [LibraryImport(Name)] public static partial int spingalett_last_error_code();
    [LibraryImport(Name)] public static partial nint spingalett_last_error_message();
    [LibraryImport(Name)] public static partial void spingalett_clear_error();
    [LibraryImport(Name)] [return: MarshalAs(UnmanagedType.U1)] public static partial bool spingalett_set_compute_mode(int mode);
    [LibraryImport(Name)] public static partial void spingalett_set_num_threads(uint n);
    [LibraryImport(Name)] public static partial void spingalett_set_verbose([MarshalAs(UnmanagedType.U1)] bool enabled);
    [LibraryImport(Name)] public static partial nint spingalett_gpu_device();
    [LibraryImport(Name)] public static partial nint spingalett_cuda_device();
    [LibraryImport(Name)] [return: MarshalAs(UnmanagedType.U1)] public static partial bool spingalett_set_gpu_precision(int precision);
    [LibraryImport(Name)] public static partial void spingalett_seed(ulong seed);

    [LibraryImport(Name)] public static partial nint spingalett_network_new_args(NetworkArgs args);
    [LibraryImport(Name)] public static partial void spingalett_network_free(nint net);
    [LibraryImport(Name)] public static partial uint spingalett_append_layer(LayerArgs args);
    [LibraryImport(Name)] public static partial uint spingalett_layer_count(nint net);
    [LibraryImport(Name)] [return: MarshalAs(UnmanagedType.U1)] public static partial bool spingalett_network_layer(nint net, uint index, NetworkLayer* layer);
    [LibraryImport(Name)] public static partial uint spingalett_input_size(nint net);
    [LibraryImport(Name)] public static partial uint spingalett_output_size(nint net);
    [LibraryImport(Name)] public static partial uint spingalett_target_size(nint net);
    [LibraryImport(Name)] public static partial ulong spingalett_parameter_count(nint net);
    [LibraryImport(Name)] [return: MarshalAs(UnmanagedType.U1)] public static partial bool spingalett_get_parameters(nint net, uint index, int kind, float* values, ulong count);
    [LibraryImport(Name)] [return: MarshalAs(UnmanagedType.U1)] public static partial bool spingalett_set_parameters(nint net, uint index, int kind, float* values, ulong count);
    [LibraryImport(Name)] public static partial TrainReport spingalett_train_args(TrainArgs args);
    [LibraryImport(Name)] [return: MarshalAs(UnmanagedType.U1)] public static partial bool spingalett_predict_args(PredictArgs args);
    [LibraryImport(Name)] public static partial EvalMetrics spingalett_evaluate_args(EvaluateArgs args);
    [LibraryImport(Name)] public static partial uint spingalett_generate_args(GenerateArgs args);
    [LibraryImport(Name)] [return: MarshalAs(UnmanagedType.U1)] public static partial bool spingalett_save_args(SaveArgs args);
    [LibraryImport(Name, StringMarshalling = StringMarshalling.Utf8)] public static partial nint spingalett_load(string filename);

    [LibraryImport(Name, StringMarshalling = StringMarshalling.Utf8)] public static partial nint spingalett_dataset_open_tokens(string path, TokenReaderOptions* options);
    [LibraryImport(Name)] public static partial DatasetInfo spingalett_dataset_info(nint reader);
    [LibraryImport(Name)] public static partial void spingalett_dataset_close(nint reader);

    [LibraryImport(Name)] public static partial nint spingalett_model_from_network(nint net, int precision);
    [LibraryImport(Name, StringMarshalling = StringMarshalling.Utf8)] public static partial nint spingalett_model_load(string path);
    [LibraryImport(Name)] public static partial void spingalett_model_free(nint model);
    [LibraryImport(Name)] [return: MarshalAs(UnmanagedType.U1)] public static partial bool spingalett_model_predict(nint model, float* inputs, uint count, float* outputs);

    /// <summary>spingalett_dataset_generator's address, for TrainArgs.generator.</summary>
    public static nint DatasetGenerator =>
        NativeLibrary.GetExport(NativeLibrary.Load(Name, typeof(Lib).Assembly, null), "spingalett_dataset_generator");
}
