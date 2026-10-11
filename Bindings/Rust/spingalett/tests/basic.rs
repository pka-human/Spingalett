// SPDX-License-Identifier: MIT
// Copyright (c) 2026 pka_human (pka_human@proton.me)

//! The bindings against the library: training, prediction, models, a transformer and its generation, and
//! (with SPINGALETT_LAYOUT naming Bin/SpingalettLayout) the structures' layout against the C compiler's.

use spingalett::{self as sg, Activation, Layer, Loss, Network, Precision, Sampling, TrainOptions};
use std::collections::HashMap;
use std::mem::{offset_of, size_of};

#[test]
fn version() {
    assert!(sg::version().starts_with("1."), "version {}", sg::version());
}

fn xor() -> (Network, Vec<f32>, Vec<f32>) {
    sg::set_verbose(false);
    sg::seed(7);
    let mut net = Network::new(Loss::Mse).unwrap();
    net.add(Layer::input(1, 1, 2)).unwrap();
    net.add(Layer::dense(8).activation(Activation::Tanh)).unwrap();
    net.add(Layer::dense(1).activation(Activation::Sigmoid)).unwrap();
    let x = vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0];
    let y = vec![0.0, 1.0, 1.0, 0.0];
    (net, x, y)
}

#[test]
fn trains_and_predicts() {
    let (mut net, x, y) = xor();
    assert_eq!((net.input_size(), net.output_size(), net.layer_count()), (2, 1, 3));
    let options = TrainOptions { epochs: 1500, batch_size: 4, learning_rate: 0.05, ..Default::default() };
    let report = net.train(&x, &y, &options).unwrap();
    assert!(report.completed && report.train_loss < 0.05, "{report:?}");
    let out = net.predict(&x).unwrap();
    for (o, t) in out.iter().zip(&y) {
        assert!((o - t).abs() < 0.3, "{out:?}");
    }
    let m = net.evaluate(&x, &y).unwrap();
    assert!(m.loss < 0.05, "{m:?}");
    assert_eq!(net.weights(1).unwrap().len(), 16);
    assert_eq!(net.biases(1).unwrap().len(), 8);
    assert!(net.predict(&x[..3]).is_err(), "a partial sample");
}

#[test]
fn models_and_files() {
    let (mut net, x, y) = xor();
    net.train(&x, &y, &TrainOptions { epochs: 50, batch_size: 4, ..Default::default() }).unwrap();
    let path = std::env::temp_dir().join(format!("spingalett_rust_{}.slett", std::process::id()));
    net.save(&path, Precision::Float32).unwrap();
    let mut back = Network::load(&path).unwrap();
    assert_eq!(back.predict(&x).unwrap(), net.predict(&x).unwrap());
    let model = sg::Model::load(&path).unwrap();
    assert_eq!((model.input_size(), model.output_size()), (2, 1));
    let a = model.predict(&x).unwrap();
    let b = net.predict(&x).unwrap();
    assert!(a.iter().zip(&b).all(|(p, q)| (p - q).abs() < 1e-5), "{a:?} {b:?}");
    let int8 = net.to_model(Precision::Int8).unwrap();
    assert_eq!(int8.predict(&x).unwrap().len(), 4);
    std::fs::remove_file(&path).unwrap();
    assert!(Network::load("/nonexistent/net.slett").is_err());
}

#[test]
fn transformer_generates() {
    sg::set_verbose(false);
    sg::seed(3);
    const T: u32 = 16;
    const V: u32 = 13;
    let mut net = Network::new(Loss::SparseCrossEntropy).unwrap();
    net.add(Layer::input(1, 1, T)).unwrap();
    let h = net.add(Layer::embedding(V, 16)).unwrap();
    net.add(Layer::rms_norm()).unwrap();
    net.add(Layer::linear(48)).unwrap();
    net.add(Layer::attention(4).kv_heads(2).causal(true).rope_theta(10000.0)).unwrap();
    let a = net.add(Layer::linear(16)).unwrap();
    net.add(Layer::add_layers(&[h, a])).unwrap();
    net.add(Layer::linear(V)).unwrap();
    assert_eq!((net.target_size(), net.output_size()), (T, T * V));
    // sequences counting up from random starts
    let n = 256;
    let (mut x, mut y) = (Vec::new(), Vec::new());
    for s in 0..n {
        let start = (s * 7 + 3) % V;
        for t in 0..T {
            x.push(((start + t) % V) as f32);
            y.push(((start + t + 1) % V) as f32);
        }
    }
    let options = TrainOptions { epochs: 30, batch_size: 32, learning_rate: 1e-2, ..Default::default() };
    let report = net.train(&x, &y, &options).unwrap();
    assert!(report.train_loss < 0.3, "{report:?}");
    let greedy = net.generate(&[3, 4, 5], 20, &Sampling::default()).unwrap();
    assert_eq!(greedy, (0..20).map(|i| (6 + i) % V).collect::<Vec<_>>());
    let drawn = Sampling { temperature: 1.0, top_p: 0.9, seed: 8, ..Default::default() };
    assert_eq!(net.generate(&[3, 4, 5], 20, &drawn).unwrap(), net.generate(&[3, 4, 5], 20, &drawn).unwrap());
    let stop = Sampling { stop: vec![9], ..Default::default() };
    assert_eq!(net.generate(&[3, 4, 5], 20, &stop).unwrap(), vec![6, 7, 8, 9]);
}

/// SpingalettLayout's dump: "Struct size N" and "Struct.field offset".
fn layout() -> Option<HashMap<String, usize>> {
    let tool = std::env::var("SPINGALETT_LAYOUT").ok()?;
    let out = std::process::Command::new(tool).output().expect("SPINGALETT_LAYOUT runs");
    let mut map = HashMap::new();
    for line in String::from_utf8_lossy(&out.stdout).lines() {
        let parts: Vec<&str> = line.split_whitespace().collect();
        match parts.as_slice() {
            [key, "size", n] => map.insert(format!("{key} size"), n.parse().unwrap()),
            [key, n] => map.insert(key.to_string(), n.parse().unwrap()),
            _ => None,
        };
    }
    Some(map)
}

#[test]
fn layout_matches_c() {
    let Some(c) = layout() else {
        eprintln!("SPINGALETT_LAYOUT not set: layout not checked");
        return;
    };
    let mut bad = Vec::new();
    let mut check = |key: &str, rust: usize| match c.get(key) {
        Some(&want) if want == rust => {}
        other => bad.push(format!("{key}: Rust {rust}, C {other:?}")),
    };
    macro_rules! fields {
        ($c:literal, $t:ty, [$($f:ident = $name:literal),* $(,)?]) => {
            check(concat!($c, " size"), size_of::<$t>());
            $(check(concat!($c, ".", $name), offset_of!($t, $f));)*
        };
    }
    use sg::spingalett_sys as sys;
    fields!("NeuralNetworkArgs", sys::NetworkArgs, [loss_func = "loss_func", reserved = "reserved"]);
    fields!("LayerArgs", sys::LayerArgs, [
        net = "net", neurons_amount = "neurons_amount", act_func = "act_func",
        weight_initialization = "weight_initialization", dropout_rate = "dropout_rate", type_ = "type",
        height = "height", width = "width", channels = "channels", filters = "filters", kernel = "kernel",
        stride = "stride", padding = "padding", kernel_h = "kernel_h", kernel_w = "kernel_w", stride_h = "stride_h",
        stride_w = "stride_w", padding_h = "padding_h", padding_w = "padding_w", groups = "groups",
        epsilon = "epsilon", momentum = "momentum", inputs = "inputs", input_count = "input_count",
        upsample = "upsample", output_padding = "output_padding", output_padding_h = "output_padding_h",
        output_padding_w = "output_padding_w", vocabulary = "vocabulary", heads = "heads", kv_heads = "kv_heads",
        rope_theta = "rope_theta", causal = "causal", positions = "positions", reserved = "reserved",
    ]);
    fields!("TrainArgs", sys::TrainArgs, [
        net = "net", training_mode = "training_mode", training_strategy = "training_strategy",
        optimizer_type = "optimizer_type", inputs = "inputs", targets = "targets", device_inputs = "device_inputs",
        device_targets = "device_targets", generator = "generator", generator_data = "generator_data",
        sample_count = "sample_count", batch_size = "batch_size", do_not_shuffle = "do_not_shuffle",
        epochs = "epochs", learning_rate = "learning_rate", weight_decay = "weight_decay", momentum = "momentum",
        beta1 = "beta1", beta2 = "beta2", epsilon = "epsilon", max_grad_norm = "max_grad_norm",
        reset_optimizer = "reset_optimizer", nan_check_interval = "nan_check_interval",
        report_interval = "report_interval", autosave_mode = "autosave_mode", autosave_interval = "autosave_interval",
        autosave_path = "autosave_path", autosave_do_not_save_optimizer = "autosave_do_not_save_optimizer",
        autosave_precision = "autosave_precision", callback = "callback", callback_interval = "callback_interval",
        callback_data = "callback_data", lr_scheduler = "lr_scheduler", lr_scheduler_data = "lr_scheduler_data",
        val_inputs = "val_inputs", val_targets = "val_targets", device_val_inputs = "device_val_inputs",
        device_val_targets = "device_val_targets", val_count = "val_count", monitor = "monitor",
        early_stopping_patience = "early_stopping_patience", early_stopping_min_delta = "early_stopping_min_delta",
        restore_best_weights = "restore_best_weights", blas_num_threads = "blas_num_threads",
        augment_shift = "augment_shift", augment_flip = "augment_flip", label_smoothing = "label_smoothing",
        lr_plateau_factor = "lr_plateau_factor", lr_plateau_patience = "lr_plateau_patience",
        lr_plateau_min_lr = "lr_plateau_min_lr", reserved = "reserved",
    ]);
    fields!("TrainReport", sys::TrainReport, [
        status = "status", epochs_run = "epochs_run", train_loss = "train_loss", has_validation = "has_validation",
        validation = "validation", monitor = "monitor", best_epoch = "best_epoch", best_value = "best_value",
        restored_best = "restored_best", reserved = "reserved",
    ]);
    fields!("TrainProgress", sys::TrainProgress, [
        epoch = "epoch", epochs = "epochs", train_loss = "train_loss", learning_rate = "learning_rate",
        has_validation = "has_validation", validation = "validation", monitor = "monitor", best_epoch = "best_epoch",
        best_value = "best_value", improved = "improved", reserved = "reserved",
    ]);
    fields!("EvalMetrics", sys::EvalMetrics, [loss = "loss", accuracy = "accuracy", reserved = "reserved"]);
    fields!("PredictArgs", sys::PredictArgs, [
        net = "net", inputs = "inputs", device_inputs = "device_inputs", sample_count = "sample_count",
        outputs = "outputs", reserved = "reserved",
    ]);
    fields!("EvaluateArgs", sys::EvaluateArgs, [
        net = "net", inputs = "inputs", targets = "targets", device_inputs = "device_inputs",
        device_targets = "device_targets", sample_count = "sample_count", reserved = "reserved",
    ]);
    fields!("SaveArgs", sys::SaveArgs, [
        net = "net", filename = "filename", do_not_save_optimizer = "do_not_save_optimizer", precision = "precision",
        reserved = "reserved",
    ]);
    fields!("SpingalettGenerateArgs", sys::GenerateArgs, [
        net = "net", prompt = "prompt", prompt_length = "prompt_length", tokens = "tokens", count = "count",
        temperature = "temperature", top_k = "top_k", top_p = "top_p", seed = "seed", stop_tokens = "stop_tokens",
        stop_count = "stop_count", reserved = "reserved",
    ]);
    fields!("SpingalettTokenReaderOptions", sys::TokenReaderOptions, [
        context = "context", token_bytes = "token_bytes", stride = "stride", offset = "offset", shuffle = "shuffle",
        reserved = "reserved",
    ]);
    fields!("SpingalettDatasetInfo", sys::DatasetInfo, [
        count = "count", input_size = "input_size", target_size = "target_size", input_encoding = "input_encoding",
        target_encoding = "target_encoding", chunk_count = "chunk_count", file_size = "file_size",
        format_version = "format_version", height = "height", width = "width", channels = "channels",
        target_set_count = "target_set_count", target_set = "target_set", reserved = "reserved",
    ]);
    fields!("SpingalettNetworkLayer", sys::NetworkLayer, [
        type_ = "type", height = "height", width = "width", channels = "channels", outputs = "outputs",
        activation = "activation", dropout_rate = "dropout_rate", kernel_h = "kernel_h", kernel_w = "kernel_w",
        stride_h = "stride_h", stride_w = "stride_w", padding_h = "padding_h", padding_w = "padding_w",
        weight_count = "weight_count", bias_count = "bias_count", groups = "groups", epsilon = "epsilon",
        momentum = "momentum", input_count = "input_count", inputs = "inputs", upsample = "upsample",
        vocabulary = "vocabulary", heads = "heads", kv_heads = "kv_heads", rope_theta = "rope_theta",
        causal = "causal", positions = "positions", reserved = "reserved",
    ]);
    fields!("SpingalettModel", sys::Model, [
        input_size = "input_size", output_size = "output_size", layer_count = "layer_count", loss = "loss",
        workspace_size = "workspace_size", image = "image", image_size = "image_size", reserved = "reserved",
    ]);
    assert!(bad.is_empty(), "layout mismatches:\n{}", bad.join("\n"));
}
