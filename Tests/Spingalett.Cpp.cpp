/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/*
 * The C++ interface (Spingalett.hpp): a residual network made by Builder, trained, run, evaluated,
 * copied through its bytes and deployed, against the C functions' results; errors as values.
 */

#include <Spingalett/Spingalett.hpp>

#include <cstdio>
#include <cstdlib>
#include <vector>

static int failures = 0;

static void check(bool ok, const char *what) {
    std::printf("  %s %s\n", ok ? "ok    " : "FAILED", what);
    failures += ok ? 0 : 1;
}

int main() {
    namespace sg = spingalett;
    sg::set_verbose(false);
    sg::seed(7);
    (void)sg::set_compute_mode(sg::Compute::SingleThreaded);

    /* 6 x 6 x 2 images, one of 3 classes each */
    const std::uint32_t n = 48, in = 6 * 6 * 2, out = 3;
    std::vector<float> x(n * in), t(n * out, 0.0f);
    std::srand(3);
    for (float &v : x) v = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
    for (std::uint32_t s = 0; s < n; s++) t[s * out + s % out] = 1.0f;

    sg::Builder b(sg::Loss::CrossEntropy);
    b.input(6, 6, 2).conv2d({.filters = 4, .kernel = 3, .padding = 1, .act = sg::Activation::Relu, .init = sg::Init::He});
    const std::uint32_t skip = b.last();                                       /* 1 */
    b.conv2d({.filters = 4, .kernel = 3, .padding = 1, .init = sg::Init::He}).batch_norm();
    const std::uint32_t normalized = b.last();                                 /* 3 */
    b.add_layers({skip, normalized}, sg::Activation::Relu).max_pool2d(2).dense(out, sg::Activation::Softmax, sg::Init::Xavier);
    sg::Result<sg::Network> made = b.build();
    check(made.has_value(), "Builder::build() makes the network");
    if (!made) {
        std::printf("    %s\n", made.error().message.c_str());
        return 1;
    }
    sg::Network net = std::move(*made);
    check(net.layer_count() == 7 && net.input_size() == in && net.output_size() == out, "its shape");
    sg::Result<sg::LayerInfo> add = net.layer(4);
    check(add && add->type == SPINGALETT_LAYER_ADD && add->input_count == 2 && add->inputs[0] == skip,
          "the addition reads the skipped layer");

    sg::TrainOptions o;
    o.epochs = 3;
    o.strategy = sg::Strategy::MiniBatch;
    o.batch_size = 16;
    o.optimizer = sg::Optimizer::Adam;
    o.learning_rate = 0.01f;
    o.val_inputs = std::span<const float>(x).first(16 * in);
    o.val_targets = std::span<const float>(t).first(16 * out);
    sg::Result<sg::TrainReport> r = net.train(x, t, o);
    check(r && r->status == SPINGALETT_TRAIN_COMPLETED && r->epochs_run == 3 && r->has_validation, "train()");

    sg::Result<std::vector<float>> y = net.predict(x);
    std::vector<float> c(n * out);
    SpingalettPredictArgs pa{};
    pa.net = net.raw();
    pa.inputs = x.data();
    pa.sample_count = n;
    pa.outputs = c.data();
    check(y && spingalett_predict_args(pa) && *y == c, "predict() as the C function predicts");

    sg::Result<sg::EvalMetrics> m = net.evaluate(x, t);
    check(m && m->loss > 0.0f && m->accuracy >= 0.0f && m->accuracy <= 1.0f, "evaluate()");

    sg::Result<std::vector<std::byte>> bytes = net.to_bytes();
    sg::Result<sg::Network> back = bytes ? sg::Network::from_bytes(*bytes) : std::unexpected(bytes.error());
    sg::Result<std::vector<float>> z = back ? back->predict(x) : std::unexpected(back.error());
    check(z && *z == *y, "a copy through its bytes predicts the same");

    sg::Result<sg::Model> model = net.to_model(sg::Precision::Float32);
    sg::Result<std::vector<float>> w = model ? model->predict(x) : std::unexpected(model.error());
    bool close = w.has_value();
    for (std::size_t i = 0; close && i < w->size(); i++) close = std::abs((*w)[i] - (*y)[i]) < 1e-4f;
    check(close, "the deployment model predicts within 1e-4");

    /* a LLaMA-like language model: learns to count, then generates the continuation */
    {
        const std::uint32_t T = 16, V = 13;
        sg::seed(3);
        sg::Builder lm(sg::Loss::SparseCrossEntropy);
        lm.input(T).embedding(V, 16);
        const std::uint32_t h = lm.last();
        lm.rms_norm().linear(48).attention(4, true, 2, 10000.0f).linear(16);
        lm.add_layers({h, lm.last()}).linear(V);
        sg::Result<sg::Network> model_lm = lm.build();
        std::vector<float> tx, ty;
        for (std::uint32_t smp = 0; smp < 256; smp++)
            for (std::uint32_t i = 0; i < T; i++) {
                const std::uint32_t start = (smp * 7u + 3u) % V;
                tx.push_back(float((start + i) % V));
                ty.push_back(float((start + i + 1u) % V));
            }
        sg::TrainOptions lo;
        lo.epochs = 30;
        lo.strategy = sg::Strategy::MiniBatch;
        lo.batch_size = 32;
        lo.optimizer = sg::Optimizer::Adam;
        lo.learning_rate = 1e-2f;
        sg::Result<sg::TrainReport> lr = model_lm ? model_lm->train(tx, ty, lo) : std::unexpected(model_lm.error());
        check(lr && lr->train_loss < 0.3f && model_lm->target_size() == T, "a language model trains");
        const std::uint32_t prompt[3] = {3, 4, 5};
        sg::Result<std::vector<std::uint32_t>> g = lr ? model_lm->generate(prompt, 20) : std::unexpected(lr.error());
        bool counts = g && g->size() == 20;
        for (std::uint32_t i = 0; counts && i < 20; i++) counts = (*g)[i] == (6u + i) % V;
        check(counts, "generate() continues the count");
        sg::Sampling stop;
        stop.stop = {9};
        sg::Result<std::vector<std::uint32_t>> st = lr ? model_lm->generate(prompt, 20, stop) : std::unexpected(lr.error());
        check(st && *st == std::vector<std::uint32_t>{6, 7, 8, 9}, "generate() stops at a stop token");
    }

    /* errors are values */
    sg::Result<std::vector<float>> partial = net.predict(std::span<const float>(x).first(in + 1));
    check(!partial && partial.error().code == SPINGALETT_ERR_INVALID, "a partial sample is an error");
    sg::Result<sg::Network> missing = sg::Network::load("no/such/file.slett");
    check(!missing && missing.error().code == SPINGALETT_ERR_FILE_IO && !missing.error().message.empty(),
          "a missing file is an error with the library's code and message");
    sg::Builder wrong;
    wrong.input(4).add_layers({0, 7});
    sg::Result<sg::Network> refused = wrong.build();
    check(!refused && !refused.error().message.empty(), "build() reports a layer the library refuses");
    check(!sg::set_compute_mode(static_cast<sg::Compute>(99)), "an unknown compute mode is an error");

    std::printf("%s (%d failures)\n", failures ? "FAILED" : "ALL PASSED", failures);
    return failures ? 1 : 0;
}
