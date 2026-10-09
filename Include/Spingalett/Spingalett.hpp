/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/*
 * C++ interface to Spingalett, header-only (C++23): the library's objects as move-only owners
 * (Network, Model, Dataset, DeviceData), data as std::span, errors as std::expected<T, Error> (the
 * library's code and message; nothing throws), scoped enums, and networks built by a fluent Builder
 * that adds its layers when build() is called:
 *
 *     auto net = spingalett::Builder(spingalett::Loss::CrossEntropy)
 *                    .input(28, 28, 1)
 *                    .conv2d({.filters = 32, .kernel = 3, .padding = 1, .act = spingalett::Activation::Relu})
 *                    .max_pool2d(2)
 *                    .dense(128, spingalett::Activation::Relu)
 *                    .dense(10, spingalett::Activation::Softmax)
 *                    .build();
 *     if (!net) std::cerr << net.error().message << '\n';
 *
 * What it does not cover is a call away: raw() gives the C object, and TrainOptions::raw carries every
 * field of SpingalettTrainArgs. The short names of the C header (Spingalett.Short.h) would replace
 * the methods of the same names (train(), predict()), so this header does not go with them.
 */
#pragma once

#if defined(SPINGALETT_SHORT_NAMES)
#error "Spingalett.hpp cannot be used with the short names of Spingalett.Short.h (SPINGALETT_SHORT_NAMES)"
#endif
#include "Spingalett.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <initializer_list>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace spingalett {

/* ---------------------------------------------------------------- errors */

/* A failure: SPINGALETT_ERR_* and what failed. */
struct Error {
    int code = SPINGALETT_ERR_INVALID;
    std::string message;
};

template <class T> using Result = std::expected<T, Error>;

/* The calling thread's error as the library left it (a fallback when it set none). */
inline Error last_error(const char *otherwise = "unknown error") {
    const int code = spingalett_last_error_code();
    const char *message = spingalett_last_error_message();
    return Error{code ? code : SPINGALETT_ERR_INVALID, message && *message ? message : otherwise};
}

inline std::unexpected<Error> failure(const char *otherwise = "unknown error") {
    return std::unexpected(last_error(otherwise));
}

inline std::unexpected<Error> invalid(const char *message) {
    return std::unexpected(Error{SPINGALETT_ERR_INVALID, message});
}

/* ---------------------------------------------------------------- enumerations */

enum class Activation : int {
    None = SPINGALETT_ACT_NONE, Sigmoid = SPINGALETT_ACT_SIGMOID, Relu = SPINGALETT_ACT_RELU,
    Tanh = SPINGALETT_ACT_TANH, LeakyRelu = SPINGALETT_ACT_LEAKY_RELU, Foo52 = SPINGALETT_ACT_FOO52,
    Softmax = SPINGALETT_ACT_SOFTMAX,
};
enum class Loss : int { Mse = SPINGALETT_LOSS_MSE, CrossEntropy = SPINGALETT_LOSS_CROSS_ENTROPY };
enum class Init : int {
    Random = SPINGALETT_INIT_RANDOM, Xavier = SPINGALETT_INIT_XAVIER,
    He = SPINGALETT_INIT_HE, Zeros = SPINGALETT_INIT_NONE,
    Lecun = SPINGALETT_INIT_LECUN,
};
enum class Optimizer : int {
    Sgd = SPINGALETT_OPTIMIZER_SGD, Momentum = SPINGALETT_OPTIMIZER_MOMENTUM, RmsProp = SPINGALETT_OPTIMIZER_RMSPROP,
    Adam = SPINGALETT_OPTIMIZER_ADAM, AdamW = SPINGALETT_OPTIMIZER_ADAMW,
};
enum class Strategy : int {
    Sample = SPINGALETT_STRATEGY_SAMPLE, FullBatch = SPINGALETT_STRATEGY_FULL_BATCH,
    MiniBatch = SPINGALETT_STRATEGY_SMALL_BATCH,
};
enum class Precision : int {
    Float32 = SPINGALETT_PRECISION_FLOAT32, Fp16 = SPINGALETT_PRECISION_FP16, BFloat16 = SPINGALETT_PRECISION_BFLOAT16,
    Int8 = SPINGALETT_PRECISION_INT8, Int4 = SPINGALETT_PRECISION_INT4, Int2 = SPINGALETT_PRECISION_INT2,
};
enum class Compute : int {
    SingleThreaded = SPINGALETT_COMPUTE_SINGLE_THREADED, OpenMP = SPINGALETT_COMPUTE_OPENMP,
    OpenBLAS = SPINGALETT_COMPUTE_OPENBLAS, Vulkan = SPINGALETT_COMPUTE_VULKAN,
};
enum class Upsample : int { Nearest = SPINGALETT_UPSAMPLE_NEAREST, Bilinear = SPINGALETT_UPSAMPLE_BILINEAR };

using EvalMetrics = SpingalettEvalMetrics;
using TrainReport = SpingalettTrainReport;
using LayerInfo = SpingalettNetworkLayer;

/* ---------------------------------------------------------------- settings */

inline const char *version() { return spingalett_version(); }
inline void seed(std::uint64_t value) { spingalett_seed(value); }
inline void set_verbose(bool on) { spingalett_set_verbose(on); }
inline void set_num_threads(unsigned n) { spingalett_set_num_threads(n); }
inline Result<void> set_compute_mode(Compute mode) {
    spingalett_clear_error();
    if (!spingalett_set_compute_mode(static_cast<SpingalettComputeMode>(mode))) return failure();
    return {};
}
/* The GPU's name, or empty without a usable device. */
inline std::string gpu_device() {
    const char *name = spingalett_gpu_device();
    return name ? name : "";
}
/* Whether the GPU multiplies in that precision from the next call on. */
inline bool set_gpu_precision(Precision precision) {
    return spingalett_set_gpu_precision(static_cast<SpingalettPrecisionMode>(precision));
}

/* ---------------------------------------------------------------- data */

/* A data set in host memory (SpingalettDataset), as the readers load it. */
class Dataset {
public:
    Dataset() = default;
    Dataset(Dataset &&other) noexcept : d_(std::exchange(other.d_, SpingalettDataset{})) {}
    Dataset &operator=(Dataset &&other) noexcept {
        if (this != &other) {
            spingalett_dataset_free(&d_);
            d_ = std::exchange(other.d_, SpingalettDataset{});
        }
        return *this;
    }
    Dataset(const Dataset &) = delete;
    Dataset &operator=(const Dataset &) = delete;
    ~Dataset() { spingalett_dataset_free(&d_); }

    /* a .slettd file */
    static Result<Dataset> load(const std::string &path) {
        Dataset d;
        spingalett_clear_error();
        if (!spingalett_load_dataset(path.c_str(), &d.d_)) return failure("cannot load the data set");
        return d;
    }
    static Result<Dataset> load_idx(const std::string &images, const std::string &labels, std::uint32_t classes = 0) {
        Dataset d;
        spingalett_clear_error();
        if (!spingalett_load_idx(images.c_str(), labels.c_str(), classes, &d.d_)) return failure("cannot load the IDX files");
        return d;
    }
    static Result<Dataset> load_csv(const std::string &path, std::uint32_t target_columns, std::uint32_t classes = 0) {
        Dataset d;
        spingalett_clear_error();
        if (!spingalett_load_csv(path.c_str(), target_columns, classes, &d.d_)) return failure("cannot load the CSV file");
        return d;
    }
    static Result<Dataset> load_cifar(const std::vector<std::string> &paths, std::uint32_t classes = 10) {
        std::vector<const char *> names;
        for (const std::string &p : paths) names.push_back(p.c_str());
        Dataset d;
        spingalett_clear_error();
        if (!spingalett_load_cifar(names.data(), static_cast<std::uint32_t>(names.size()), classes, &d.d_))
            return failure("cannot load the CIFAR files");
        return d;
    }

    Result<void> save(const std::string &path) const {
        spingalett_clear_error();
        if (!spingalett_save_dataset(&d_, path.c_str(), nullptr)) return failure("cannot save the data set");
        return {};
    }
    void shuffle() { spingalett_dataset_shuffle(&d_); }
    /* the last count samples, moved out (a validation set) */
    Result<Dataset> split(std::uint32_t count) {
        Dataset tail;
        spingalett_clear_error();
        if (!spingalett_dataset_split(&d_, count, &tail.d_)) return failure("cannot split the data set");
        return tail;
    }

    std::uint32_t count() const { return d_.count; }
    std::uint32_t input_size() const { return d_.input_size; }
    std::uint32_t target_size() const { return d_.target_size; }
    std::span<const float> inputs() const { return {d_.inputs, std::size_t(d_.count) * d_.input_size}; }
    std::span<const float> targets() const { return {d_.targets, std::size_t(d_.count) * d_.target_size}; }
    const SpingalettDataset &raw() const { return d_; }
    SpingalettDataset &raw() { return d_; }

private:
    SpingalettDataset d_{};
};

/* Rows of floats in the GPU's memory (spingalett_device_data_new()). */
class DeviceData {
public:
    DeviceData(DeviceData &&other) noexcept : d_(std::exchange(other.d_, nullptr)) {}
    DeviceData &operator=(DeviceData &&other) noexcept {
        if (this != &other) {
            spingalett_device_data_free(d_);
            d_ = std::exchange(other.d_, nullptr);
        }
        return *this;
    }
    DeviceData(const DeviceData &) = delete;
    DeviceData &operator=(const DeviceData &) = delete;
    ~DeviceData() { spingalett_device_data_free(d_); }

    /* count rows of size floats from values (count * size of them) */
    static Result<DeviceData> create(std::span<const float> values, std::uint32_t size) {
        if (size == 0 || values.size() % size != 0) return invalid("DeviceData: values are not whole rows");
        spingalett_clear_error();
        SpingalettDeviceData *d =
            spingalett_device_data_new(values.data(), static_cast<std::uint32_t>(values.size() / size), size);
        if (!d) return failure("no usable GPU, or its memory ran out");
        return DeviceData(d);
    }
    std::uint32_t count() const { return spingalett_device_data_count(d_); }
    std::uint32_t size() const { return spingalett_device_data_size(d_); }
    const SpingalettDeviceData *raw() const { return d_; }

private:
    explicit DeviceData(SpingalettDeviceData *d) : d_(d) {}
    SpingalettDeviceData *d_ = nullptr;
};

/* ---------------------------------------------------------------- deployment */

/* A read-only network computing in the precision of its weights (SpingalettModel). */
class Model {
public:
    Model(Model &&other) noexcept : m_(std::exchange(other.m_, nullptr)) {}
    Model &operator=(Model &&other) noexcept {
        if (this != &other) {
            spingalett_model_free(m_);
            m_ = std::exchange(other.m_, nullptr);
        }
        return *this;
    }
    Model(const Model &) = delete;
    Model &operator=(const Model &) = delete;
    ~Model() { spingalett_model_free(m_); }

    static Result<Model> load(const std::string &path) {
        spingalett_clear_error();
        SpingalettModel *m = spingalett_model_load(path.c_str());
        if (!m) return failure("cannot load the model");
        return Model(m);
    }
    static Result<Model> from_bytes(std::span<const std::byte> image) {
        spingalett_clear_error();
        SpingalettModel *m = spingalett_model_from_memory(image.data(), image.size());
        if (!m) return failure("invalid model image");
        return Model(m);
    }
    /* takes ownership of a model the library made */
    static Model adopt(SpingalettModel *m) { return Model(m); }

    std::uint32_t input_size() const { return m_->input_size; }
    std::uint32_t output_size() const { return m_->output_size; }
    std::uint32_t layer_count() const { return m_->layer_count; }

    /* outputs [samples x output size] of inputs [samples x input size] */
    Result<void> predict(std::span<const float> inputs, std::span<float> outputs) const {
        const std::uint32_t n = samples(inputs.size(), input_size());
        if (n == 0 || outputs.size() != std::size_t(n) * output_size())
            return invalid("Model::predict: inputs and outputs are not of whole samples of one count");
        spingalett_clear_error();
        if (!spingalett_model_predict(m_, inputs.data(), n, outputs.data())) return failure();
        return {};
    }
    Result<std::vector<float>> predict(std::span<const float> inputs) const {
        std::vector<float> out(inputs.size() / (input_size() ? input_size() : 1u) * output_size());
        Result<void> r = predict(inputs, out);
        if (!r) return std::unexpected(r.error());
        return out;
    }
    Result<EvalMetrics> evaluate(std::span<const float> inputs, std::span<const float> targets) const {
        const std::uint32_t n = samples(inputs.size(), input_size());
        if (n == 0 || targets.size() != std::size_t(n) * output_size())
            return invalid("Model::evaluate: inputs and targets are not of whole samples of one count");
        spingalett_clear_error();
        EvalMetrics m = spingalett_model_evaluate(m_, inputs.data(), targets.data(), n);
        if (std::isnan(m.loss)) return failure();
        return m;
    }
    const SpingalettModel *raw() const { return m_; }

private:
    explicit Model(SpingalettModel *m) : m_(m) {}
    static std::uint32_t samples(std::size_t values, std::uint32_t per) {
        return per && values % per == 0 ? static_cast<std::uint32_t>(values / per) : 0u;
    }
    SpingalettModel *m_ = nullptr;
};

/* ---------------------------------------------------------------- networks */

/* train()'s options as typed fields; raw carries every other field of SpingalettTrainArgs (its net,
   data, strategy, optimizer, epochs, batch size, learning rate and validation set are set from the
   fields below and the call's arguments). Zero means the library's default, as in C. */
struct TrainOptions {
    std::size_t epochs = 1;
    Strategy strategy = Strategy::Sample;
    std::uint32_t batch_size = 0;
    Optimizer optimizer = Optimizer::Sgd;
    float learning_rate = 0.0f;
    std::span<const float> val_inputs{};
    std::span<const float> val_targets{};
    SpingalettTrainArgs raw{};
};

class Network;

/* A dense, convolution or transposed convolution layer's settings for Builder. */
struct Conv2D {
    std::uint32_t filters = 0;
    std::uint32_t kernel = 0;
    std::uint32_t stride = 0;       /* 0: 1 */
    std::uint32_t padding = 0;
    std::uint32_t groups = 0;       /* 0: 1; the input's channels: depthwise */
    Activation act = Activation::None;
    Init init = Init::Random;
    float dropout = 0.0f;
};

/*
 * Collects layers and makes the network in build(): the first layer is the input layer, every
 * layer reads the one before it unless from() names others (by index: 0 is the input layer,
 * last() the layer added last), and build() reports the first layer the library refused.
 */
class Builder {
public:
    explicit Builder(Loss loss = Loss::Mse) : loss_(loss) {}

    Builder &input(std::uint32_t size) { return add(args(SPINGALETT_LAYER_DENSE, Activation::None, size)); }
    Builder &input(std::uint32_t height, std::uint32_t width, std::uint32_t channels) {
        SpingalettLayerArgs a = args(SPINGALETT_LAYER_DENSE, Activation::None, 0);
        a.height = height;
        a.width = width;
        a.channels = channels;
        return add(a);
    }
    Builder &dense(std::uint32_t units, Activation act = Activation::None, Init init = Init::Random, float dropout = 0.0f) {
        SpingalettLayerArgs a = args(SPINGALETT_LAYER_DENSE, act, units);
        a.weight_initialization = static_cast<SpingalettWeightInitialization>(init);
        a.dropout_rate = dropout;
        return add(a);
    }
    Builder &conv2d(const Conv2D &c) { return add(conv(SPINGALETT_LAYER_CONV2D, c)); }
    Builder &conv_transpose2d(const Conv2D &c) { return add(conv(SPINGALETT_LAYER_CONV_TRANSPOSE2D, c)); }
    Builder &max_pool2d(std::uint32_t kernel, std::uint32_t stride = 0, std::uint32_t padding = 0) {
        return add(pool(SPINGALETT_LAYER_MAX_POOL2D, kernel, stride, padding));
    }
    Builder &avg_pool2d(std::uint32_t kernel, std::uint32_t stride = 0, std::uint32_t padding = 0) {
        return add(pool(SPINGALETT_LAYER_AVG_POOL2D, kernel, stride, padding));
    }
    Builder &global_avg_pool2d() { return add(args(SPINGALETT_LAYER_GLOBAL_AVG_POOL, Activation::None, 0)); }
    Builder &batch_norm(Activation act = Activation::None) { return add(args(SPINGALETT_LAYER_BATCH_NORM, act, 0)); }
    Builder &layer_norm(Activation act = Activation::None) { return add(args(SPINGALETT_LAYER_LAYER_NORM, act, 0)); }
    Builder &upsample2d(std::uint32_t factor = 2, Upsample mode = Upsample::Nearest) {
        SpingalettLayerArgs a = args(SPINGALETT_LAYER_UPSAMPLE, Activation::None, 0);
        a.stride = factor;
        a.upsample = static_cast<SpingalettUpsampleMode>(mode);
        return add(a);
    }
    /* the sum of layers of one shape, the layers side by side along the channels */
    Builder &add_layers(std::initializer_list<std::uint32_t> layers, Activation act = Activation::None) {
        return from(layers).add(args(SPINGALETT_LAYER_ADD, act, 0));
    }
    Builder &concat_layers(std::initializer_list<std::uint32_t> layers, Activation act = Activation::None) {
        return from(layers).add(args(SPINGALETT_LAYER_CONCAT, act, 0));
    }
    /* any layer, as the C builders describe it (its net and inputs are set by the builder) */
    Builder &layer(const SpingalettLayerArgs &a) { return add(a); }

    /* the layers the next one reads */
    Builder &from(std::initializer_list<std::uint32_t> layers) {
        next_.assign(layers.begin(), layers.end());
        return *this;
    }
    /* index of the layer added last */
    std::uint32_t last() const { return static_cast<std::uint32_t>(layers_.size()) - 1u; }

    inline Result<Network> build() const;

private:
    struct Pending {
        SpingalettLayerArgs args;
        std::vector<std::uint32_t> inputs;
    };
    static SpingalettLayerArgs args(SpingalettLayerType type, Activation act, std::uint32_t units) {
        SpingalettLayerArgs a{};
        a.type = type;
        a.act_func = static_cast<SpingalettActivationFunction>(act);
        a.neurons_amount = units;
        return a;
    }
    static SpingalettLayerArgs conv(SpingalettLayerType type, const Conv2D &c) {
        SpingalettLayerArgs a = args(type, c.act, 0);
        a.filters = c.filters;
        a.kernel = c.kernel;
        a.stride = c.stride;
        a.padding = c.padding;
        a.groups = c.groups;
        a.weight_initialization = static_cast<SpingalettWeightInitialization>(c.init);
        a.dropout_rate = c.dropout;
        return a;
    }
    static SpingalettLayerArgs pool(SpingalettLayerType type, std::uint32_t kernel, std::uint32_t stride, std::uint32_t padding) {
        SpingalettLayerArgs a = args(type, Activation::None, 0);
        a.kernel = kernel;
        a.stride = stride;
        a.padding = padding;
        return a;
    }
    Builder &add(const SpingalettLayerArgs &a) {
        layers_.push_back(Pending{a, std::move(next_)});
        next_.clear();
        return *this;
    }
    Loss loss_;
    std::vector<Pending> layers_;
    std::vector<std::uint32_t> next_;
};

/* A network to train, run, save or deploy (SpingalettNetwork). */
class Network {
public:
    Network(Network &&other) noexcept : n_(std::exchange(other.n_, nullptr)) {}
    Network &operator=(Network &&other) noexcept {
        if (this != &other) {
            spingalett_network_free(n_);
            n_ = std::exchange(other.n_, nullptr);
        }
        return *this;
    }
    Network(const Network &) = delete;
    Network &operator=(const Network &) = delete;
    ~Network() { spingalett_network_free(n_); }

    /* an empty network (Builder makes whole ones) */
    static Result<Network> create(Loss loss = Loss::Mse) {
        SpingalettNetworkArgs a{};
        a.loss_func = static_cast<SpingalettLossFunction>(loss);
        spingalett_clear_error();
        SpingalettNetwork *n = spingalett_network_new_args(a);
        if (!n) return failure("cannot make the network");
        return Network(n);
    }
    static Result<Network> load(const std::string &path) {
        spingalett_clear_error();
        SpingalettNetwork *n = spingalett_load(path.c_str());
        if (!n) return failure("cannot load the network");
        return Network(n);
    }
    static Result<Network> from_bytes(std::span<const std::byte> image) {
        spingalett_clear_error();
        SpingalettNetwork *n = spingalett_load_from_memory(image.data(), image.size());
        if (!n) return failure("invalid network image");
        return Network(n);
    }
    static Result<Network> import_onnx(const std::string &path) {
        spingalett_clear_error();
        SpingalettNetwork *n = spingalett_import_onnx(path.c_str());
        if (!n) return failure("cannot import the model");
        return Network(n);
    }
    /* takes ownership of a network the library made */
    static Network adopt(SpingalettNetwork *n) { return Network(n); }

    /* appends a layer described as for the C builders; its index */
    Result<std::uint32_t> append(SpingalettLayerArgs a) {
        a.net = n_;
        spingalett_clear_error();
        const std::uint32_t index = spingalett_append_layer(a);
        if (index == SPINGALETT_NO_LAYER) return failure("the layer was refused");
        return index;
    }

    std::uint32_t input_size() const { return spingalett_input_size(n_); }
    std::uint32_t output_size() const { return spingalett_output_size(n_); }
    std::uint32_t layer_count() const { return spingalett_layer_count(n_); }
    std::uint64_t parameter_count() const { return spingalett_parameter_count(n_); }
    Result<LayerInfo> layer(std::uint32_t index) const {
        LayerInfo info{};
        spingalett_clear_error();
        if (!spingalett_network_layer(n_, index, &info)) return failure();
        return info;
    }

    Result<TrainReport> train(std::span<const float> inputs, std::span<const float> targets,
                              const TrainOptions &o = {}) {
        const std::uint32_t n = samples(inputs.size(), input_size());
        if (n == 0 || targets.size() != std::size_t(n) * output_size())
            return invalid("Network::train: inputs and targets are not of whole samples of one count");
        SpingalettTrainArgs a = arguments(o);
        a.inputs = inputs.data();
        a.targets = targets.data();
        a.sample_count = n;
        return run(a, o);
    }
    Result<TrainReport> train(const Dataset &data, const TrainOptions &o = {}) {
        return train(data.inputs(), data.targets(), o);
    }
    /* from data sets on the GPU, their first rows */
    Result<TrainReport> train(const DeviceData &inputs, const DeviceData &targets, const TrainOptions &o = {}) {
        SpingalettTrainArgs a = arguments(o);
        a.device_inputs = inputs.raw();
        a.device_targets = targets.raw();
        a.sample_count = inputs.count();
        return run(a, o);
    }

    Result<void> predict(std::span<const float> inputs, std::span<float> outputs) {
        const std::uint32_t n = samples(inputs.size(), input_size());
        if (n == 0 || outputs.size() != std::size_t(n) * output_size())
            return invalid("Network::predict: inputs and outputs are not of whole samples of one count");
        SpingalettPredictArgs a{};
        a.net = n_;
        a.inputs = inputs.data();
        a.sample_count = n;
        a.outputs = outputs.data();
        spingalett_clear_error();
        if (!spingalett_predict_args(a)) return failure();
        return {};
    }
    Result<std::vector<float>> predict(std::span<const float> inputs) {
        std::vector<float> out(inputs.size() / (input_size() ? input_size() : 1u) * output_size());
        Result<void> r = predict(inputs, out);
        if (!r) return std::unexpected(r.error());
        return out;
    }
    Result<EvalMetrics> evaluate(std::span<const float> inputs, std::span<const float> targets) {
        const std::uint32_t n = samples(inputs.size(), input_size());
        if (n == 0 || targets.size() != std::size_t(n) * output_size())
            return invalid("Network::evaluate: inputs and targets are not of whole samples of one count");
        SpingalettEvaluateArgs a{};
        a.net = n_;
        a.inputs = inputs.data();
        a.targets = targets.data();
        a.sample_count = n;
        spingalett_clear_error();
        EvalMetrics m = spingalett_evaluate_args(a);
        if (std::isnan(m.loss)) return failure();
        return m;
    }

    Result<void> save(const std::string &path, Precision precision = Precision::Float32, bool optimizer = true) const {
        SpingalettSaveArgs a{};
        a.net = n_;
        a.filename = path.c_str();
        a.do_not_save_optimizer = !optimizer;
        a.precision = static_cast<SpingalettPrecisionMode>(precision);
        spingalett_clear_error();
        if (!spingalett_save_args(a)) return failure();
        return {};
    }
    Result<std::vector<std::byte>> to_bytes(Precision precision = Precision::Float32, bool optimizer = false) const {
        std::size_t size = 0;
        spingalett_clear_error();
        void *image = spingalett_save_to_memory(n_, static_cast<SpingalettPrecisionMode>(precision), optimizer, &size);
        if (!image) return failure();
        const std::byte *bytes = static_cast<const std::byte *>(image);
        std::vector<std::byte> out(bytes, bytes + size);
        spingalett_free(image);
        return out;
    }
    /* a deployment model of the network in precision */
    Result<Model> to_model(Precision precision = Precision::Float32) const {
        spingalett_clear_error();
        SpingalettModel *m = spingalett_model_from_network(n_, static_cast<SpingalettPrecisionMode>(precision));
        if (!m) return failure();
        return Model::adopt(m);
    }

    SpingalettNetwork *raw() const { return n_; }

private:
    explicit Network(SpingalettNetwork *n) : n_(n) {}
    static std::uint32_t samples(std::size_t values, std::uint32_t per) {
        return per && values % per == 0 ? static_cast<std::uint32_t>(values / per) : 0u;
    }
    SpingalettTrainArgs arguments(const TrainOptions &o) const {
        SpingalettTrainArgs a = o.raw;
        a.net = n_;
        a.epochs = o.epochs;
        a.training_strategy = static_cast<SpingalettTrainingStrategy>(o.strategy);
        a.batch_size = o.batch_size;
        a.optimizer_type = static_cast<SpingalettOptimizerType>(o.optimizer);
        a.learning_rate = o.learning_rate;
        return a;
    }
    Result<TrainReport> run(SpingalettTrainArgs &a, const TrainOptions &o) {
        if (!o.val_inputs.empty()) {
            const std::uint32_t v = samples(o.val_inputs.size(), input_size());
            if (v == 0 || o.val_targets.size() != std::size_t(v) * output_size())
                return invalid("Network::train: the validation set is not of whole samples of one count");
            a.val_inputs = o.val_inputs.data();
            a.val_targets = o.val_targets.data();
            a.val_count = v;
        }
        spingalett_clear_error();
        TrainReport r = spingalett_train_args(a);
        if (r.status == SPINGALETT_TRAIN_FAILED) return failure("training failed");
        return r;
    }
    SpingalettNetwork *n_ = nullptr;
};

inline Result<Network> Builder::build() const {
    Result<Network> net = Network::create(loss_);
    if (!net) return net;
    for (const Pending &p : layers_) {
        SpingalettLayerArgs a = p.args;
        if (p.inputs.size() > SPINGALETT_MAX_INPUTS) return invalid("Builder: more inputs than a layer takes");
        for (std::size_t k = 0; k < p.inputs.size(); k++) a.inputs[k] = p.inputs[k];
        a.input_count = static_cast<std::uint32_t>(p.inputs.size());
        Result<std::uint32_t> index = net->append(a);
        if (!index) return std::unexpected(index.error());
    }
    return net;
}

} // namespace spingalett
