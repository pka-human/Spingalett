<?php

// SPDX-License-Identifier: MIT
// Copyright (c) 2026 pka_human (pka_human@proton.me)

// The bindings against the library: training, prediction, files, models, every kind of layer, a small
// LLaMA-like model and its generation, training on a token file, preloading, the declarations of
// spingalett.h against the C headers (in a checkout of the repository) and, with SPINGALETT_LAYOUT naming
// Bin/SpingalettLayout, every structure's layout against the C compiler's (PHP 8.1 and later, whose FFI
// gives the offsets). Exits with 1 when a check fails.
//
//     php -d extension=ffi tests/run.php

declare(strict_types=1);

require __DIR__ . '/../autoload.php';

use Spingalett\Activation;
use Spingalett\ComputeMode;
use Spingalett\Init;
use Spingalett\Layer;
use Spingalett\Library;
use Spingalett\Loss;
use Spingalett\Model;
use Spingalett\Network;
use Spingalett\Optimizer;
use Spingalett\Precision;
use Spingalett\Sampling;
use Spingalett\SpingalettException;
use Spingalett\Strategy;
use Spingalett\Tokenizer;
use Spingalett\TrainOptions;

$failures = 0;

function check(bool $ok, string $what): void
{
    global $failures;
    if (!$ok) {
        $failures++;
        echo "  FAIL: $what\n";
    }
}

function test(string $name, callable $body): void
{
    echo "$name\n";
    try {
        $body();
    } catch (Throwable $e) {
        check(false, get_class($e) . ': ' . $e->getMessage() . ' at ' . basename($e->getFile()) . ':' . $e->getLine());
    }
}

/** Whether $body throws an instance of $class. */
function throws(string $class, callable $body): bool
{
    try {
        $body();
    } catch (Throwable $e) {
        return $e instanceof $class;
    }
    return false;
}

function close(array $a, array $b, float $tolerance): bool
{
    if (count($a) !== count($b)) {
        return false;
    }
    foreach ($a as $i => $v) {
        if (abs($v - $b[$i]) > $tolerance) {
            return false;
        }
    }
    return true;
}

const X = [0, 0, 0, 1, 1, 0, 1, 1];
const Y = [0, 1, 1, 0];

function xor_network(): Network
{
    Library::setVerbose(false);
    Library::seed(7);
    $net = new Network(Loss::MSE);
    $net->add(Layer::input(1, 1, 2));
    $net->add(Layer::dense(8)->activation(Activation::TANH));
    $net->add(Layer::dense(1)->activation(Activation::SIGMOID));
    return $net;
}

/** A small LLaMA-like language model: embeddings, RMS normalization, grouped-query attention with RoPE. */
function language_model(int $context, int $vocabulary): Network
{
    $lm = new Network(Loss::SPARSE_CROSS_ENTROPY);
    $lm->add(Layer::input(1, 1, $context));
    $h = $lm->add(Layer::embedding($vocabulary, 16));
    $lm->add(Layer::rmsNorm());
    $lm->add(Layer::linear(48));
    $lm->add(Layer::attention(4)->kvHeads(2)->causal()->ropeTheta(10000));
    $a = $lm->add(Layer::linear(16));
    $lm->add(Layer::addLayers($h, $a));
    $lm->add(Layer::linear($vocabulary));
    return $lm;
}

function temporary(string $name): string
{
    return sys_get_temp_dir() . DIRECTORY_SEPARATOR . 'spingalett_php_' . getmypid() . '_' . $name;
}

test('version', function (): void {
    check(strncmp(Library::version(), '1.', 2) === 0, 'version ' . Library::version());
    check(Library::cpuKernels() !== '', 'CPU kernels');
});

test('constants against the enumerators', function (): void {
    $ffi = Library::ffi();
    $prefixes = [Activation::class => 'ACT', Loss::class => 'LOSS', ComputeMode::class => 'COMPUTE',
                 Precision::class => 'PRECISION', Optimizer::class => 'OPTIMIZER', Strategy::class => 'STRATEGY',
                 Init::class => 'INIT'];
    foreach ($prefixes as $class => $prefix) {
        foreach ((new ReflectionClass($class))->getConstants() as $name => $value) {
            $c = 'SPINGALETT_' . $prefix . '_' . ($class === Init::class && $name === 'ZEROS' ? 'NONE' : $name);
            check($ffi->$c === $value, "$class::$name is $value, $c " . $ffi->$c);
        }
    }
});

test('XOR trains and predicts', function (): void {
    $net = xor_network();
    check($net->inputSize() === 2 && $net->outputSize() === 1 && $net->layerCount() === 3, 'sizes');
    check($net->parameterCount() === 2 * 8 + 8 + 8 + 1, 'parameters ' . $net->parameterCount());
    $r = $net->train(X, Y, new TrainOptions(1500, 4, 0.05));
    check($r->completed && $r->epochsRun === 1500 && $r->trainLoss < 0.05, "training ($r->trainLoss)");
    $out = $net->predict(X);
    check(close($out, Y, 0.3), 'predictions ' . json_encode($out));
    check($net->predict([[0, 0], [0, 1], [1, 0], [1, 1]]) === $out, 'a sample an array');
    check($net->evaluate(X, Y)->loss < 0.05, 'evaluation');
    check(count($net->weights(1)) === 16 && count($net->biases(1)) === 8, 'weights and biases');
    check(throws(InvalidArgumentException::class, function () use ($net): void {
        $net->predict([0, 0, 1]);
    }), 'a partial sample');

    // more values than a row of the buffers holds
    $many = array_merge(...array_fill(0, 10000, X));
    check(close($net->predict($many), array_merge(...array_fill(0, 10000, $out)), 1e-6), 'buffers of several rows');

    $weights = array_map(function (int $i): float {
        return $i / 4.0 - 2.0;
    }, range(0, 15));
    $net->setParameters(1, false, $weights);
    check($net->weights(1) === $weights, 'set parameters');
    check(throws(SpingalettException::class, function () use ($net): void {
        $net->setParameters(1, true, [1, 2, 3]);
    }), 'parameters of the wrong count');
});

test('files and models', function (): void {
    $net = xor_network();
    $net->train(X, Y, new TrainOptions(50, 4));
    $path = temporary('net.slett');
    $net->save($path);
    $out = $net->predict(X);
    check(Network::load($path)->predict(X) === $out, 'a file round trip');
    $model = Model::load($path);
    check($model->inputSize() === 2 && $model->outputSize() === 1, 'model sizes');
    check(close($model->predict(X), $out, 1e-5), 'model predictions');
    $m = $model->evaluate(X, Y);
    check(abs($m->loss - $net->evaluate(X, Y)->loss) < 1e-5, 'model evaluation');
    $int8 = $net->toModel(Precision::INT8);
    check(count($int8->predict(X)) === 4, 'an INT8 model');
    $model->close();
    unlink($path);
    $e = null;
    try {
        Network::load('/nonexistent/net.slett');
    } catch (SpingalettException $caught) {
        $e = $caught;
    }
    check($e !== null && $e->getCode() === SpingalettException::FILE_IO, 'a missing file');
});

test('every kind of layer', function (): void {
    Library::seed(5);
    $net = new Network(Loss::CROSS_ENTROPY);
    $net->add(Layer::input(8, 8, 2));
    $conv = $net->add(Layer::conv2d(4, 3)->padding(1));                           // 8 x 8 x 4
    $bn = $net->add(Layer::batchNorm());
    $net->add(Layer::maxPool2d(2));                                               // 4 x 4 x 4
    $up = $net->add(Layer::convTranspose2d(4, 2)->stride(2));                     // 8 x 8 x 4
    $sum = $net->add(Layer::addLayers($bn, $up));
    $cat = $net->add(Layer::concat($conv, $sum));                                 // 8 x 8 x 8
    $ln = $net->add(Layer::layerNorm());
    $net->add(Layer::multiply($cat, $ln));
    $net->add(Layer::avgPool2d(2));                                               // 4 x 4 x 8
    $net->add(Layer::upsample(2)->bilinear());                                    // 8 x 8 x 8
    $net->add(Layer::rmsNorm()->epsilon(1e-6));
    $net->add(Layer::conv2d(2, 1)->groups(2)->init(Init::XAVIER)->dropout(0.1));  // 8 x 8 x 2
    $net->add(Layer::globalAvgPool());
    $net->add(Layer::dense(3)->activation(Activation::SOFTMAX));
    $ffi = Library::ffi();
    $kinds = ['DENSE', 'CONV2D', 'BATCH_NORM', 'MAX_POOL2D', 'CONV_TRANSPOSE2D', 'ADD', 'CONCAT', 'LAYER_NORM',
              'MULTIPLY', 'AVG_POOL2D', 'UPSAMPLE', 'RMS_NORM', 'CONV2D', 'GLOBAL_AVG_POOL', 'DENSE'];
    check($net->layerCount() === count($kinds) && $net->outputSize() === 3, 'layers and outputs');
    $info = $ffi->new('SpingalettNetworkLayer');
    foreach ($kinds as $i => $kind) {
        $ffi->spingalett_network_layer($net->raw(), $i, FFI::addr($info));
        check($info->type === $ffi->{"SPINGALETT_LAYER_$kind"}, "layer $i is $kind");
        if ($kind === 'UPSAMPLE') {
            check($info->upsample === $ffi->SPINGALETT_UPSAMPLE_BILINEAR, 'bilinear');
        }
        if ($i === 12) {
            check($info->groups === 2, 'groups');
        }
    }
    $x = [];
    $y = [];
    for ($s = 0; $s < 16; $s++) {
        for ($i = 0; $i < 128; $i++) {
            $x[] = sin($s * 1.7 + $i * 0.3);
        }
        array_push($y, ...[$s % 3 === 0 ? 1 : 0, $s % 3 === 1 ? 1 : 0, $s % 3 === 2 ? 1 : 0]);
    }
    $r = $net->train($x, $y, new TrainOptions(2, 8, 1e-3, Optimizer::ADAMW, Strategy::SMALL_BATCH, 0.01));
    check($r->completed && is_finite($r->trainLoss), 'training');
    check(count($net->predict($x)) === 16 * 3, 'predictions');
});

test('a LLaMA-like model learns to count, then generates', function (): void {
    Library::seed(3);
    $t = 16;
    $v = 13;
    $lm = language_model($t, $v);
    check($lm->targetSize() === $t && $lm->outputSize() === $t * $v, 'language model sizes');
    $x = [];
    $y = [];
    for ($s = 0; $s < 256; $s++) {
        $start = ($s * 7 + 3) % $v;
        for ($i = 0; $i < $t; $i++) {
            $x[] = ($start + $i) % $v;
            $y[] = ($start + $i + 1) % $v;
        }
    }
    $r = $lm->train($x, $y, new TrainOptions(30, 32, 1e-2));
    check($r->trainLoss < 0.3, "the language model learns ($r->trainLoss)");
    $greedy = $lm->generate([3, 4, 5], 20);
    check($greedy === array_map(function (int $i) use ($v): int {
        return (6 + $i) % $v;
    }, range(0, 19)), 'greedy generation ' . json_encode($greedy));
    $drawn = new Sampling(1.0, 0, 0.9, 8);
    $p = $lm->generate([3, 4, 5], 20, $drawn);
    check(count($p) === 20 && $p === $lm->generate([3, 4, 5], 20, $drawn), 'draws repeat with their seed');
    check($lm->generate([3, 4, 5], 20, new Sampling(0.0, 0, 0.0, 0, [9])) === [6, 7, 8, 9], 'stop tokens');
    check(throws(InvalidArgumentException::class, function () use ($lm): void {
        $lm->generate([], 5);
    }), 'an empty prompt');
});

test('a session of the LLaMA-like model gives the window\'s logits, a token at a time', function (): void {
    Library::seed(5);
    $t = 16;
    $v = 13;
    $lm = language_model($t, $v);
    foreach ([Precision::FLOAT32, Precision::INT8] as $precision) {
        $model = $lm->toModel($precision);
        $tokens = [3, 4, 5, 6, 7, 8, 9, 10];
        $window = array_merge($tokens, array_fill(0, $t - count($tokens), 0));
        $whole = $model->predict($window);
        $session = $model->session(0, 2);
        check($session->vocabulary() === $v, 'vocabulary');
        $same = true;
        foreach ($tokens as $i => $token) {
            $same = $same && $session->feed([$token]) === array_slice($whole, $i * $v, $v);
        }
        check($same && $session->length() === count($tokens), "fed a token at a time ($precision)");
        $all = $session->feed([1, 2], 1, true);
        $two = $model->predict(array_merge([1, 2], array_fill(0, $t - 2, 0)));
        check($all === array_slice($two, 0, 2 * $v), "a prompt at once ($precision)");
        $session->rewind(1, 1);
        $session->copy(0, 1);
        check($session->length(1) === count($tokens), 'rewind and copy');
        $session->rewind(0, 0);
        $session->rewind(0, 1);
        $batch = $session->generate([[3, 4, 5], [7]], 5);
        $alone = $model->session()->generate([[3, 4, 5]], 5);
        check(count($batch) === 2 && $batch[0] === $alone[0] && count($batch[1]) === 5, 'batched generation');
    }
    check(throws(SpingalettException::class, function (): void {
        xor_network()->toModel()->session();
    }), 'a model that is no language model');
});

test('training on a token file', function (): void {
    Library::seed(11);
    $v = 13;
    $tokens = [];
    for ($i = 0; $i < 4096; $i++) {
        $tokens[] = $i % $v;
    }
    $path = temporary('tokens.bin');
    file_put_contents($path, pack('v*', ...$tokens));      // nanoGPT's .bin: uint16, little-endian
    $lm = language_model(16, $v);
    $r = $lm->trainTokens($path, 4, new TrainOptions(8, 32, 1e-2));
    unlink($path);
    check($r->completed && $r->epochsRun === 8 && $r->trainLoss < 0.3, "training ($r->trainLoss)");
    check($lm->generate([3, 4, 5], 8) === [6, 7, 8, 9, 10, 11, 12, 0], 'generation');
    check(throws(SpingalettException::class, function () use ($lm): void {
        $lm->trainTokens('/nonexistent/tokens.bin');
    }), 'a missing token file');
});

test('tokenizers give the ids and texts of Hugging Face\'s', function (): void {
    $data = __DIR__ . '/../../../Tests/Data/';
    if (!is_file($data . 'tokenizer_gpt2.json')) {
        echo "  the test data are not here: tokenizers not checked\n";
        return;
    }
    // Tests/make_tokenizer_tests.py: "special add text", the ids, the text decoded ("*": not checked) and
    // without special tokens; texts in hexadecimal, "-" for none
    $unhex = function (string $h): string {
        return $h === '-' ? '' : (string)hex2bin($h);
    };
    $files = [];
    foreach (['gpt2', 'llama3', 'qwen2', 'o200k', 'spm', 'metaspace'] as $n) {
        $files["tokenizer_$n.json"] = "tokenizer_$n.txt";
    }
    $files['gguf_llama.gguf'] = 'tokenizer_llama3.txt';
    $files['gguf_spm.gguf'] = 'gguf_spm.txt';
    foreach ($files as $file => $expected) {
        $tok = Tokenizer::load($data . $file);
        $lines = array_values(array_filter(file($data . $expected, FILE_IGNORE_NEW_LINES), function (string $l): bool {
            return $l !== '' && $l[0] !== '#';
        }));
        $wrong = 0;
        for ($k = 0; $k + 3 < count($lines); $k += 4) {
            [$special, $add, $text] = explode(' ', $lines[$k]);
            $ids = array_map('intval', array_slice(explode(' ', $lines[$k + 1]), 1));
            $got = $tok->encode($unhex($text), $special === '1', $add === '1');
            $ok = $got === $ids && ($lines[$k + 2] === '*' || $tok->decode($ids) === $unhex($lines[$k + 2])) &&
                  $tok->decode($ids, true) === $unhex($lines[$k + 3]);
            $wrong += $ok ? 0 : 1;
        }
        check($wrong === 0 && $k > 0, "$file: $wrong of " . ($k / 4) . ' texts wrong');
        // (texts come back as they were where the decoder inverts the pre-tokenizer: byte-level ones)
        $long = str_repeat('Tokenizers cut text into pieces, ', 200);
        if (strpos($file, 'o200k') === false && strpos($file, 'spm') === false && strpos($file, 'metaspace') === false) {
            check($tok->decode($tok->encode($long)) === $long, "$file: a long text round trip");
        }
        $tok->close();
    }
    $tok = Tokenizer::fromBytes((string)file_get_contents($data . 'tokenizer_llama3.json'));
    check($tok->encode('<|begin_of_text|>', true) === [420], 'special tokens when asked');
    check(count($tok->encode('<|begin_of_text|>')) >= 2, 'special tokens as text');
    check($tok->id($tok->token(300) ?? '') === 300 && $tok->token(1 << 30) === null, 'tokens and ids');
    check(throws(SpingalettException::class, function () use ($data): void {
        Tokenizer::load($data . 'no_such_tokenizer.json');
    }), 'a missing tokenizer');
});

test('pretrained models give PyTorch\'s logits', function (): void {
    $data = __DIR__ . '/../../../Tests/Data/';
    if (!is_file($data . 'hf_llama.bin')) {
        echo "  the test data are not here: pretrained models not checked\n";
        return;
    }
    // hf_llama.bin (Tests/make_pretrained_tests.py): "SPGT", 1, T, T x vocabulary, then the tokens and logits as
    // floats
    $bin = (string)file_get_contents($data . 'hf_llama.bin');
    [, $t, $n] = array_values(unpack('V3', $bin, 4));
    $floats = array_values(unpack('g*', substr($bin, 16)));
    $tokens = array_slice($floats, 0, $t);
    $logits = array_slice($floats, $t, $n);
    foreach (['hf_llama', 'gguf_llama.gguf'] as $path) {
        $lm = Network::fromPretrained($data . $path, $t);
        check(close($lm->predict($tokens), $logits, 1e-4), "$path: the logits of PyTorch");
        $lm->close();
    }
    $tok = Tokenizer::load($data . 'hf_llama');
    $ids = $tok->encode("The quick brown fox jumps over the lazy dog, it's 1234 times!");
    check(array_slice($ids, 0, $t) === array_map('intval', $tokens), 'the directory\'s tokenizer');
    check($tok->eos() === 424, 'the EOS of tokenizer_config.json');
    check(throws(SpingalettException::class, function () use ($data): void {
        Network::fromPretrained($data . 'no_such_model');
    }), 'a missing model');
});

test('declarations against the C headers', function (): void {
    $include = __DIR__ . '/../../../Include/Spingalett/';
    if (!is_file($include . 'Spingalett.h')) {
        echo "  the C headers are not here: declarations not checked\n";
        return;
    }
    // Comments, preprocessor lines and spaces out; the macros spingalett.h writes as numbers replaced.
    $normal = function (string $code): string {
        $code = preg_replace(['~/\*.*?\*/~s', '~//[^\n]*~', '~^\s*#.*$~m'], ' ', $code);
        $code = str_replace(['SPINGALETT_API', 'SPINGALETT_RESERVED', 'SPINGALETT_MAX_INPUTS'], ['', '8', '16'], $code);
        return preg_replace('~\s+~', '', $code);
    };
    $c = '';
    foreach (['Spingalett.Inference.h', 'Spingalett.Runtime.h', 'Spingalett.h'] as $header) {
        $c .= explode('#if defined(SPINGALETT_SHORT_NAMES)', file_get_contents($include . $header))[0];
    }
    $header = file_get_contents(Library::HEADER);
    $ours = $normal($header);
    $theirs = $normal($c);
    // every declaration of spingalett.h is one of the C headers'
    $depth = 0;
    $start = 0;
    $count = 0;
    for ($i = 0, $n = strlen($ours); $i < $n; $i++) {
        $depth += $ours[$i] === '{' ? 1 : ($ours[$i] === '}' ? -1 : 0);
        if ($ours[$i] === ';' && $depth === 0) {
            $declaration = substr($ours, $start, $i + 1 - $start);
            check(strpos($theirs, $declaration) !== false, "not in the C headers: $declaration");
            $start = $i + 1;
            $count++;
        }
    }
    check($count > 100, "$count declarations");
    // and every function and type of the C headers is declared
    preg_match_all('~(spingalett_\w+)\(~', $theirs, $functions);
    preg_match_all('~}(Spingalett\w+);|typedefstruct(Spingalett\w+)\2;|\(\*(Spingalett\w+)\)~', $theirs, $types);
    $names = array_merge($functions[1], array_filter(array_merge($types[1], $types[2], $types[3])));
    check(count($names) > 120, count($names) . ' functions and types');
    $plain = preg_replace('~/\*.*?\*/~s', ' ', $header);
    foreach ($names as $name) {
        check(preg_match('~\b' . $name . '\b~', $plain) === 1, "$name is not declared");
    }
});

test('structures against the C compiler\'s layout', function (): void {
    $tool = getenv('SPINGALETT_LAYOUT');
    if ($tool === false || $tool === '') {
        echo "  SPINGALETT_LAYOUT not set: layout not checked\n";
        return;
    }
    if (!method_exists(FFI\CType::class, 'getStructFieldOffset')) {
        echo '  PHP ' . PHP_VERSION . "'s FFI gives no offsets (8.1 does): layout not checked\n";
        return;
    }
    exec(escapeshellarg($tool), $lines, $status);
    check($status === 0 && $lines !== [], 'SpingalettLayout runs');
    $c = [];
    foreach ($lines as $line) {
        $parts = preg_split('~\s+~', trim($line));
        if (count($parts) === 3 && $parts[1] === 'size') {
            $c["$parts[0] size"] = (int) $parts[2];
        } elseif (count($parts) === 2) {
            $c[$parts[0]] = (int) $parts[1];
        }
    }
    // The names of 0.x that SpingalettLayout prints for some of them.
    $short = ['SpingalettNetworkArgs' => 'NeuralNetworkArgs', 'SpingalettLayerArgs' => 'LayerArgs',
              'SpingalettForwardArgs' => 'ForwardArgs', 'SpingalettTrainArgs' => 'TrainArgs',
              'SpingalettEvalMetrics' => 'EvalMetrics', 'SpingalettTrainProgress' => 'TrainProgress',
              'SpingalettTrainReport' => 'TrainReport', 'SpingalettEvaluateArgs' => 'EvaluateArgs',
              'SpingalettOptimizerArgs' => 'OptimizerArgs', 'SpingalettDatasetSaveOptions' => 'DatasetSaveOptions',
              'SpingalettDatasetReaderOptions' => 'DatasetReaderOptions', 'SpingalettPredictArgs' => 'PredictArgs',
              'SpingalettSaveArgs' => 'SaveArgs', 'SpingalettLRScheduleParams' => 'LRScheduleParams'];
    $header = preg_replace('~/\*.*?\*/~s', '', file_get_contents(Library::HEADER));
    preg_match_all('~typedef\s+struct\s*\{[^}]*\}\s*(\w+)\s*;~', $header, $structs);
    $names = [];
    foreach ($structs[1] as $struct) {
        $names[$struct] = $short[$struct] ?? $struct;
    }
    foreach (array_keys($c) as $key) {
        if (substr($key, -5) === ' size') {
            check(in_array(substr($key, 0, -5), $names, true), "spingalett.h has no $key");
        }
    }
    $ffi = Library::ffi();
    $checked = 0;
    foreach ($names as $struct => $name) {
        $type = $ffi->type($struct);
        check(($c["$name size"] ?? null) === $type->getSize(), "$struct: size " . $type->getSize() . ', C '
                                                                . ($c["$name size"] ?? 'none'));
        $fields = $type->getStructFieldNames();
        foreach ($fields as $field) {
            $offset = $type->getStructFieldOffset($field);
            check(($c["$name.$field"] ?? null) === $offset, "$struct.$field at $offset, C "
                                                            . ($c["$name.$field"] ?? 'none'));
            $checked++;
        }
        foreach (array_keys($c) as $key) {
            if (strncmp($key, "$name.", strlen($name) + 1) === 0) {
                check(in_array(substr($key, strlen($name) + 1), $fields, true), "$struct has no $key");
            }
        }
    }
    check(count($names) >= 22 && $checked > 250, count($names) . " structures, $checked fields");
});

// Last: the declarations loaded again, as opcache.preload would, replace the ones the tests above used.
test('preloading', function (): void {
    Library::preload();
    $net = xor_network();
    $r = $net->train(X, Y, new TrainOptions(1500, 4, 0.05));
    check($r->trainLoss < 0.05 && close($net->predict(X), Y, 0.3), 'XOR on the preloaded declarations');
});

echo $failures === 0 ? "ALL PASSED\n" : "$failures FAILURES\n";
exit($failures === 0 ? 0 : 1);
