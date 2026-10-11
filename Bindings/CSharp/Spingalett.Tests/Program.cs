// SPDX-License-Identifier: MIT
// Copyright (c) 2026 pka_human (pka_human@proton.me)

// The bindings against the library: training, prediction, models, a transformer and its generation, and (with
// SPINGALETT_LAYOUT naming Bin/SpingalettLayout) the structures' layout against the C compiler's. Exit code 0
// when everything holds.

using System.Diagnostics;
using System.Runtime.InteropServices;
using Spingalett;

int failures = 0;
void Check(bool ok, string what)
{
    if (!ok) { failures++; Console.WriteLine($"  FAIL: {what}"); }
}

Library.SetVerbose(false);
Check(Library.Version.StartsWith("1."), $"version {Library.Version}");

// XOR
Library.Seed(7);
float[] x = { 0, 0, 0, 1, 1, 0, 1, 1 }, y = { 0, 1, 1, 0 };
using (var net = new Network(Loss.Mse))
{
    net.Add(Layer.Input(1, 1, 2));
    net.Add(Layer.Dense(8).WithActivation(Activation.Tanh));
    net.Add(Layer.Dense(1).WithActivation(Activation.Sigmoid));
    Check(net.InputSize == 2 && net.OutputSize == 1 && net.LayerCount == 3, "sizes");
    var r = net.Train(x, y, new TrainOptions { Epochs = 1500, BatchSize = 4, LearningRate = 0.05f });
    Check(r.Completed && r.TrainLoss < 0.05f, $"XOR trains ({r.TrainLoss})");
    var o = net.Predict(x);
    Check(o.Zip(y).All(p => Math.Abs(p.First - p.Second) < 0.3f), "XOR predicts");
    Check(net.Evaluate(x, y).Loss < 0.05f, "evaluate");
    Check(net.Weights(1).Length == 16 && net.Biases(1).Length == 8, "parameters");
    string path = Path.Combine(Path.GetTempPath(), $"spingalett_cs_{Environment.ProcessId}.slett");
    net.Save(path);
    using (var back = Network.Load(path)) Check(back.Predict(x).SequenceEqual(o), "file round trip");
    using (var model = Model.Load(path))
        Check(model.InputSize == 2 && model.Predict(x).Zip(o).All(p => Math.Abs(p.First - p.Second) < 1e-5f), "model");
    using (var int8 = net.ToModel(Precision.Int8)) Check(int8.Predict(x).Length == 4, "INT8 model");
    File.Delete(path);
    try { net.Predict(new float[3]); Check(false, "a partial sample"); } catch (ArgumentException) { }
}
try { Network.Load("/nonexistent/net.slett"); Check(false, "a missing file"); } catch (SpingalettException) { }

// a small LLaMA-like model learns to count, then generates
Library.Seed(3);
const uint T = 16, V = 13;
using (var lm = new Network(Loss.SparseCrossEntropy))
{
    lm.Add(Layer.Input(1, 1, T));
    uint h = lm.Add(Layer.Embedding(V, 16));
    lm.Add(Layer.RmsNorm());
    lm.Add(Layer.Linear(48));
    lm.Add(Layer.Attention(4).KvHeads(2).Causal().RopeTheta(10000));
    uint a = lm.Add(Layer.Linear(16));
    lm.Add(Layer.AddLayers(h, a));
    lm.Add(Layer.Linear(V));
    Check(lm.TargetSize == T && lm.OutputSize == T * V, "language model sizes");
    var tx = new List<float>();
    var ty = new List<float>();
    for (uint s = 0; s < 256; s++)
        for (uint t = 0; t < T; t++)
        {
            uint start = (s * 7 + 3) % V;
            tx.Add((start + t) % V);
            ty.Add((start + t + 1) % V);
        }
    var r = lm.Train(tx.ToArray(), ty.ToArray(), new TrainOptions { Epochs = 30, BatchSize = 32, LearningRate = 1e-2f });
    Check(r.TrainLoss < 0.3f, $"the language model learns ({r.TrainLoss})");
    var greedy = lm.Generate(new uint[] { 3, 4, 5 }, 20);
    Check(greedy.SequenceEqual(Enumerable.Range(0, 20).Select(i => (uint)((6 + i) % V))), "greedy generation");
    var drawn = new Sampling { Temperature = 1, TopP = 0.9f, Seed = 8 };
    Check(lm.Generate(new uint[] { 3, 4, 5 }, 20, drawn).SequenceEqual(lm.Generate(new uint[] { 3, 4, 5 }, 20, drawn)), "draws repeat");
    Check(lm.Generate(new uint[] { 3, 4, 5 }, 20, new Sampling { Stop = new uint[] { 9 } }).SequenceEqual(new uint[] { 6, 7, 8, 9 }), "stop tokens");
}

// tokenizers and pretrained language models: the files of Tests/Data, when the tests run within the repository
string? data = null;
for (var dir = new DirectoryInfo(Directory.GetCurrentDirectory()); dir != null && data == null; dir = dir.Parent)
    if (File.Exists(Path.Combine(dir.FullName, "Tests", "Data", "tokenizer_llama3.json"))) data = Path.Combine(dir.FullName, "Tests", "Data");
if (data is null)
{
    Console.WriteLine("Tests/Data not found: tokenizers and pretrained models not checked");
}
else
{
    // every case of Hugging Face's ids and texts (Tests/make_tokenizer_tests.py), by the tokenizer.json and its GGUF form
    static string Unhex(string hex) => hex == "-" ? "" : System.Text.Encoding.UTF8.GetString(Convert.FromHexString(hex));
    var lines = File.ReadAllLines(Path.Combine(data, "tokenizer_llama3.txt")).Where(l => !l.StartsWith('#')).ToArray();
    foreach (var file in new[] { "tokenizer_llama3.json", "gguf_llama.gguf" })
    {
        using var tok = Tokenizer.Load(Path.Combine(data, file));
        int wrong = 0, cases = 0;
        for (int k = 0; k + 3 < lines.Length; k += 4, cases++)
        {
            var head = lines[k].Split(' ');
            string text = Unhex(head[2]);
            uint[] ids = lines[k + 1].Split(' ', StringSplitOptions.RemoveEmptyEntries).Skip(1).Select(uint.Parse).ToArray();
            if (!tok.Encode(text, head[0] == "1", head[1] == "1").SequenceEqual(ids)) wrong++;
            if (tok.Decode(ids) != Unhex(lines[k + 2]) || tok.Decode(ids, skipSpecial: true) != Unhex(lines[k + 3])) wrong++;
        }
        Check(cases >= 100 && wrong == 0, $"{file}: {wrong} of {cases} texts encoded or decoded otherwise than by Hugging Face's");
    }
    using (var tok = Tokenizer.FromBytes(File.ReadAllBytes(Path.Combine(data, "tokenizer_llama3.json"))))
    {
        Check(tok.Bos == 420 && tok.Eos == 421 && tok.Vocabulary == 425 && tok.Added == 5 && tok.Unk is null, "BOS, EOS, vocabulary");
        Check(tok.Id("<|begin_of_text|>") == 420 && tok.Token(420) == "<|begin_of_text|>" && tok.Id("no such token") is null &&
              tok.Token(1u << 30) is null, "tokens by id and by name");
        Check(tok.Encode("<|begin_of_text|>", special: true).SequenceEqual(new uint[] { 420 }) && tok.Encode("<|begin_of_text|>").Length > 1,
              "special tokens only when asked");
        string longText = string.Concat(Enumerable.Repeat("Tokenizers cut text into pieces, ", 300));
        Check(tok.Decode(tok.Encode(longText)) == longText, "a long text encoded and decoded");
    }
    try { Tokenizer.Load(Path.Combine(data, "no_such_tokenizer.json")); Check(false, "a missing tokenizer"); } catch (SpingalettException) { }

    // a Hugging Face directory and its weights in a GGUF file against PyTorch's logits (Tests/make_pretrained_tests.py)
    var bin = File.ReadAllBytes(Path.Combine(data, "hf_llama.bin"));
    uint window = BitConverter.ToUInt32(bin, 8), count = BitConverter.ToUInt32(bin, 12);
    var tokens = new float[window];
    var logits = new float[count];
    Buffer.BlockCopy(bin, 16, tokens, 0, (int)window * 4);
    Buffer.BlockCopy(bin, 16 + (int)window * 4, logits, 0, (int)count * 4);
    foreach (var model in new[] { "hf_llama", "gguf_llama.gguf" })
    {
        using var lm = Network.FromPretrained(Path.Combine(data, model), window);
        var got = lm.Predict(tokens);
        float worst = got.Length == logits.Length ? got.Zip(logits).Max(p => Math.Abs(p.First - p.Second)) : float.PositiveInfinity;
        Check(lm.TargetSize == window && worst < 1e-4f, $"{model}: logits {worst} from PyTorch's");
    }
    using (var tok = Tokenizer.Load(Path.Combine(data, "hf_llama")))
        Check(tok.Encode("The quick brown fox jumps over the lazy dog, it's 1234 times!").Take((int)window).Select(i => (float)i)
                 .SequenceEqual(tokens) && tok.Bos == 420 && tok.Eos == 424, "the tokenizer of a model directory");
    try { Network.FromPretrained(Path.Combine(data, "no_such_model")); Check(false, "a missing model"); } catch (SpingalettException) { }
}

// the structures' layout against the C compiler's
string? tool = Environment.GetEnvironmentVariable("SPINGALETT_LAYOUT");
if (tool is null)
{
    Console.WriteLine("SPINGALETT_LAYOUT not set: layout not checked");
}
else
{
    var c = new Dictionary<string, long>();
    var run = Process.Start(new ProcessStartInfo(tool) { RedirectStandardOutput = true })!;
    foreach (var line in run.StandardOutput.ReadToEnd().Split('\n', StringSplitOptions.RemoveEmptyEntries))
    {
        var parts = line.Split(' ', StringSplitOptions.RemoveEmptyEntries);
        if (parts.Length == 3 && parts[1] == "size") c[parts[0] + " size"] = long.Parse(parts[2]);
        else if (parts.Length == 2) c[parts[0]] = long.Parse(parts[1]);
    }
    run.WaitForExit();
    void Layout(string name, Type type, params string[] fields)
    {
        Check(c.TryGetValue(name + " size", out long size) && size == Marshal.SizeOf(type), $"{name} size {Marshal.SizeOf(type)}");
        foreach (var f in fields)
        {
            string cname = f == "type_" ? "type" : f;
            long off = (long)Marshal.OffsetOf(type, f);
            Check(c.TryGetValue($"{name}.{cname}", out long want) && want == off, $"{name}.{cname} at {off}");
        }
    }
    string[] All(Type t) => t.GetFields().Select(f => f.Name).Where(n => !n.EndsWith('_')).ToArray();
    Layout("NeuralNetworkArgs", typeof(Spingalett.Native.NetworkArgs), All(typeof(Spingalett.Native.NetworkArgs)));
    Layout("LayerArgs", typeof(Spingalett.Native.LayerArgs), All(typeof(Spingalett.Native.LayerArgs)));
    Layout("TrainArgs", typeof(Spingalett.Native.TrainArgs), All(typeof(Spingalett.Native.TrainArgs)));
    Layout("TrainReport", typeof(Spingalett.Native.TrainReport), All(typeof(Spingalett.Native.TrainReport)));
    Layout("EvalMetrics", typeof(Spingalett.Native.EvalMetrics), All(typeof(Spingalett.Native.EvalMetrics)));
    Layout("PredictArgs", typeof(Spingalett.Native.PredictArgs), All(typeof(Spingalett.Native.PredictArgs)));
    Layout("EvaluateArgs", typeof(Spingalett.Native.EvaluateArgs), All(typeof(Spingalett.Native.EvaluateArgs)));
    Layout("SaveArgs", typeof(Spingalett.Native.SaveArgs), All(typeof(Spingalett.Native.SaveArgs)));
    Layout("SpingalettGenerateArgs", typeof(Spingalett.Native.GenerateArgs), All(typeof(Spingalett.Native.GenerateArgs)));
    Layout("SpingalettTokenReaderOptions", typeof(Spingalett.Native.TokenReaderOptions), All(typeof(Spingalett.Native.TokenReaderOptions)));
    Layout("SpingalettDatasetInfo", typeof(Spingalett.Native.DatasetInfo), All(typeof(Spingalett.Native.DatasetInfo)));
    Layout("SpingalettNetworkLayer", typeof(Spingalett.Native.NetworkLayer), All(typeof(Spingalett.Native.NetworkLayer)));
    Layout("SpingalettTokenizerInfo", typeof(Spingalett.Native.TokenizerInfo), All(typeof(Spingalett.Native.TokenizerInfo)));
    Layout("SpingalettTokenizeArgs", typeof(Spingalett.Native.TokenizeArgs), All(typeof(Spingalett.Native.TokenizeArgs)));
    Layout("SpingalettDetokenizeArgs", typeof(Spingalett.Native.DetokenizeArgs), All(typeof(Spingalett.Native.DetokenizeArgs)));
    Layout("SpingalettPretrainedArgs", typeof(Spingalett.Native.PretrainedArgs), All(typeof(Spingalett.Native.PretrainedArgs)));
    Layout("SpingalettModel", typeof(Spingalett.Native.Model), "input_size", "output_size", "layer_count", "loss",
           "workspace_size", "image", "image_size", "reserved");
}

Console.WriteLine(failures == 0 ? "ALL PASSED" : $"{failures} FAILURES");
return failures == 0 ? 0 : 1;
