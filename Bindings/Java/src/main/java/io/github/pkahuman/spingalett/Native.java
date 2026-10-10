// SPDX-License-Identifier: MIT
// Copyright (c) 2026 pka_human (pka_human@proton.me)

package io.github.pkahuman.spingalett;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.StructLayout;
import java.lang.foreign.SymbolLookup;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.VarHandle;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_BOOLEAN;
import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

/**
 * The C API of libspingalett that the bindings use: the argument structures laid out as C lays them out
 * (fields at their natural alignment; the tests compare every offset with the C compiler's), and the functions.
 * The library is "spingalett" on the loader's paths (LD_LIBRARY_PATH, PATH), or the file the system property
 * spingalett.library names.
 */
final class Native {
    private Native() {}

    static final int RESERVED = 8;
    static final int MAX_INPUTS = 16;

    private static MemoryLayout i(String name) { return JAVA_INT.withName(name); }
    private static MemoryLayout f(String name) { return JAVA_FLOAT.withName(name); }
    private static MemoryLayout l(String name) { return JAVA_LONG.withName(name); }
    private static MemoryLayout a(String name) { return ADDRESS.withName(name); }
    private static MemoryLayout b(String name) { return JAVA_BYTE.withName(name); }
    private static MemoryLayout reserved(int words) { return MemoryLayout.sequenceLayout(words, JAVA_LONG).withName("reserved"); }

    /** A structure of these fields as C lays them out: each at its alignment, the size a multiple of the largest. */
    static StructLayout struct(MemoryLayout... fields) {
        List<MemoryLayout> out = new ArrayList<>();
        long offset = 0, align = 1;
        for (MemoryLayout field : fields) {
            long at = field.byteAlignment(), pad = (at - offset % at) % at;
            if (pad > 0) out.add(MemoryLayout.paddingLayout(pad));
            out.add(field);
            offset += pad + field.byteSize();
            align = Math.max(align, at);
        }
        long tail = (align - offset % align) % align;
        if (tail > 0) out.add(MemoryLayout.paddingLayout(tail));
        return MemoryLayout.structLayout(out.toArray(new MemoryLayout[0]));
    }

    static final StructLayout EVAL_METRICS = struct(f("loss"), f("accuracy"), reserved(RESERVED));
    static final StructLayout NETWORK_ARGS = struct(i("loss_func"), reserved(RESERVED));
    static final StructLayout LAYER_ARGS = struct(a("net"), i("neurons_amount"), i("act_func"), i("weight_initialization"),
            f("dropout_rate"), i("type"), i("height"), i("width"), i("channels"), i("filters"), i("kernel"), i("stride"),
            i("padding"), i("kernel_h"), i("kernel_w"), i("stride_h"), i("stride_w"), i("padding_h"), i("padding_w"),
            i("groups"), f("epsilon"), f("momentum"), MemoryLayout.sequenceLayout(MAX_INPUTS, JAVA_INT).withName("inputs"),
            i("input_count"), i("upsample"), i("output_padding"), i("output_padding_h"), i("output_padding_w"),
            i("vocabulary"), i("heads"), i("kv_heads"), f("rope_theta"), b("causal"), b("positions"), reserved(RESERVED - 3));
    static final StructLayout NETWORK_LAYER = struct(i("type"), i("height"), i("width"), i("channels"), i("outputs"),
            i("activation"), f("dropout_rate"), i("kernel_h"), i("kernel_w"), i("stride_h"), i("stride_w"), i("padding_h"),
            i("padding_w"), l("weight_count"), l("bias_count"), i("groups"), f("epsilon"), f("momentum"), i("input_count"),
            MemoryLayout.sequenceLayout(MAX_INPUTS, JAVA_INT).withName("inputs"), i("upsample"), i("vocabulary"), i("heads"),
            i("kv_heads"), f("rope_theta"), b("causal"), b("positions"), reserved(RESERVED - 2));
    static final StructLayout TRAIN_ARGS = struct(a("net"), i("training_mode"), i("training_strategy"),
            i("optimizer_type"), a("inputs"), a("targets"), a("device_inputs"), a("device_targets"), a("generator"),
            a("generator_data"), i("sample_count"), i("batch_size"), b("do_not_shuffle"), l("epochs"), f("learning_rate"),
            f("weight_decay"), f("momentum"), f("beta1"), f("beta2"), f("epsilon"), f("max_grad_norm"), b("reset_optimizer"),
            l("nan_check_interval"), l("report_interval"), i("autosave_mode"), l("autosave_interval"), a("autosave_path"),
            b("autosave_do_not_save_optimizer"), i("autosave_precision"), a("callback"), l("callback_interval"),
            a("callback_data"), a("lr_scheduler"), a("lr_scheduler_data"), a("val_inputs"), a("val_targets"),
            a("device_val_inputs"), a("device_val_targets"), i("val_count"), i("monitor"), l("early_stopping_patience"),
            f("early_stopping_min_delta"), b("restore_best_weights"), i("blas_num_threads"), i("augment_shift"),
            b("augment_flip"), f("label_smoothing"), f("lr_plateau_factor"), l("lr_plateau_patience"),
            f("lr_plateau_min_lr"), reserved(RESERVED));
    static final StructLayout TRAIN_REPORT = struct(i("status"), l("epochs_run"), f("train_loss"), b("has_validation"),
            EVAL_METRICS.withName("validation"), i("monitor"), l("best_epoch"), f("best_value"), b("restored_best"),
            reserved(RESERVED));
    static final StructLayout PREDICT_ARGS = struct(a("net"), a("inputs"), a("device_inputs"), i("sample_count"),
            a("outputs"), reserved(RESERVED));
    static final StructLayout EVALUATE_ARGS = struct(a("net"), a("inputs"), a("targets"), a("device_inputs"),
            a("device_targets"), i("sample_count"), reserved(RESERVED));
    static final StructLayout SAVE_ARGS = struct(a("net"), a("filename"), b("do_not_save_optimizer"), i("precision"),
            reserved(RESERVED));
    static final StructLayout GENERATE_ARGS = struct(a("net"), a("prompt"), i("prompt_length"), a("tokens"), i("count"),
            f("temperature"), i("top_k"), f("top_p"), l("seed"), a("stop_tokens"), i("stop_count"), reserved(RESERVED));
    static final StructLayout TOKEN_READER_OPTIONS = struct(i("context"), i("token_bytes"), i("stride"), l("offset"),
            b("shuffle"), reserved(RESERVED));
    static final StructLayout DATASET_INFO = struct(i("count"), i("input_size"), i("target_size"), i("input_encoding"),
            i("target_encoding"), i("chunk_count"), l("file_size"), i("format_version"), i("height"), i("width"),
            i("channels"), i("target_set_count"), i("target_set"), reserved(RESERVED));
    /** The public fields of a deployment model, which the library's own follow. */
    static final StructLayout MODEL = struct(i("input_size"), i("output_size"), i("layer_count"), i("loss"),
            l("workspace_size"), a("image"), l("image_size"), i("max_width_"), i("max_int_inputs_"), l("conv_scratch_"),
            a("owner_"), l("activations_"), reserved(RESERVED));

    private static final Map<String, VarHandle> HANDLES = new ConcurrentHashMap<>();

    private static VarHandle handle(StructLayout layout, String field) {
        return HANDLES.computeIfAbsent(System.identityHashCode(layout) + "." + field,
                k -> layout.varHandle(MemoryLayout.PathElement.groupElement(field)));
    }

    static void set(MemorySegment s, StructLayout layout, String field, Object value) {
        handle(layout, field).set(s, 0L, value);
    }

    static Object get(MemorySegment s, StructLayout layout, String field) {
        return handle(layout, field).get(s, 0L);
    }

    static final Linker LINKER = Linker.nativeLinker();
    static final SymbolLookup LIBRARY;

    static {
        String path = System.getProperty("spingalett.library");
        LIBRARY = path != null ? SymbolLookup.libraryLookup(java.nio.file.Path.of(path), Arena.global())
                               : SymbolLookup.libraryLookup(System.mapLibraryName("spingalett"), Arena.global());
    }

    static MemorySegment symbol(String name) {
        return LIBRARY.find(name).orElseThrow(() -> new UnsatisfiedLinkError("libspingalett has no " + name));
    }

    private static MethodHandle fn(String name, MemoryLayout result, MemoryLayout... args) {
        FunctionDescriptor d = result == null ? FunctionDescriptor.ofVoid(args) : FunctionDescriptor.of(result, args);
        return LINKER.downcallHandle(symbol(name), d);
    }

    static final MethodHandle VERSION = fn("spingalett_version", ADDRESS);
    static final MethodHandle LAST_ERROR_CODE = fn("spingalett_last_error_code", JAVA_INT);
    static final MethodHandle LAST_ERROR_MESSAGE = fn("spingalett_last_error_message", ADDRESS);
    static final MethodHandle CLEAR_ERROR = fn("spingalett_clear_error", null);
    static final MethodHandle SET_COMPUTE_MODE = fn("spingalett_set_compute_mode", JAVA_BOOLEAN, JAVA_INT);
    static final MethodHandle SET_NUM_THREADS = fn("spingalett_set_num_threads", null, JAVA_INT);
    static final MethodHandle SET_VERBOSE = fn("spingalett_set_verbose", null, JAVA_BOOLEAN);
    static final MethodHandle CUDA_DEVICE = fn("spingalett_cuda_device", ADDRESS);
    static final MethodHandle GPU_DEVICE = fn("spingalett_gpu_device", ADDRESS);
    static final MethodHandle SET_GPU_PRECISION = fn("spingalett_set_gpu_precision", JAVA_BOOLEAN, JAVA_INT);
    static final MethodHandle SEED = fn("spingalett_seed", null, JAVA_LONG);

    static final MethodHandle NETWORK_NEW = fn("spingalett_network_new_args", ADDRESS, NETWORK_ARGS);
    static final MethodHandle NETWORK_FREE = fn("spingalett_network_free", null, ADDRESS);
    static final MethodHandle APPEND_LAYER = fn("spingalett_append_layer", JAVA_INT, LAYER_ARGS);
    static final MethodHandle LAYER_COUNT = fn("spingalett_layer_count", JAVA_INT, ADDRESS);
    static final MethodHandle NETWORK_LAYER_INFO = fn("spingalett_network_layer", JAVA_BOOLEAN, ADDRESS, JAVA_INT, ADDRESS);
    static final MethodHandle INPUT_SIZE = fn("spingalett_input_size", JAVA_INT, ADDRESS);
    static final MethodHandle OUTPUT_SIZE = fn("spingalett_output_size", JAVA_INT, ADDRESS);
    static final MethodHandle TARGET_SIZE = fn("spingalett_target_size", JAVA_INT, ADDRESS);
    static final MethodHandle PARAMETER_COUNT = fn("spingalett_parameter_count", JAVA_LONG, ADDRESS);
    static final MethodHandle GET_PARAMETERS = fn("spingalett_get_parameters", JAVA_BOOLEAN, ADDRESS, JAVA_INT, JAVA_INT,
                                                  ADDRESS, JAVA_LONG);
    static final MethodHandle SET_PARAMETERS = fn("spingalett_set_parameters", JAVA_BOOLEAN, ADDRESS, JAVA_INT, JAVA_INT,
                                                  ADDRESS, JAVA_LONG);
    static final MethodHandle TRAIN = fn("spingalett_train_args", TRAIN_REPORT, TRAIN_ARGS);
    static final MethodHandle PREDICT = fn("spingalett_predict_args", JAVA_BOOLEAN, PREDICT_ARGS);
    static final MethodHandle EVALUATE = fn("spingalett_evaluate_args", EVAL_METRICS, EVALUATE_ARGS);
    static final MethodHandle GENERATE = fn("spingalett_generate_args", JAVA_INT, GENERATE_ARGS);
    static final MethodHandle SAVE = fn("spingalett_save_args", JAVA_BOOLEAN, SAVE_ARGS);
    static final MethodHandle LOAD = fn("spingalett_load", ADDRESS, ADDRESS);

    static final MethodHandle DATASET_OPEN_TOKENS = fn("spingalett_dataset_open_tokens", ADDRESS, ADDRESS, ADDRESS);
    static final MethodHandle DATASET_INFO_OF = fn("spingalett_dataset_info", DATASET_INFO, ADDRESS);
    static final MethodHandle DATASET_CLOSE = fn("spingalett_dataset_close", null, ADDRESS);
    static final MemorySegment DATASET_GENERATOR = symbol("spingalett_dataset_generator");

    static final MethodHandle MODEL_FROM_NETWORK = fn("spingalett_model_from_network", ADDRESS, ADDRESS, JAVA_INT);
    static final MethodHandle MODEL_LOAD = fn("spingalett_model_load", ADDRESS, ADDRESS);
    static final MethodHandle MODEL_FREE = fn("spingalett_model_free", null, ADDRESS);
    static final MethodHandle MODEL_PREDICT = fn("spingalett_model_predict", JAVA_BOOLEAN, ADDRESS, ADDRESS, JAVA_INT, ADDRESS);

    /** A C string's text, or null for a null pointer. */
    static String string(MemorySegment p) {
        return p.equals(MemorySegment.NULL) ? null : p.reinterpret(Long.MAX_VALUE).getString(0);
    }

    /** Rethrows what a downcall threw (nothing but errors of the call itself). */
    static RuntimeException rethrow(Throwable t) {
        if (t instanceof RuntimeException r) return r;
        if (t instanceof Error e) throw e;
        return new IllegalStateException(t);
    }

    static ValueLayout.OfFloat floats() { return JAVA_FLOAT; }
}
