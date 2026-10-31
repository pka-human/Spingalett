# Performance

What was measured, how, on which machine, and against what. The numbers name the release that measured them.

`Examples/Benchmark.c` measures a 784-512-1000-10 network (925K parameters, ReLU, softmax with
cross-entropy, Adam) on 20,000 synthetic samples: full-batch training (5 epochs), mini-batch
training (batches of 64, one epoch) and batched inference; then the convolutional network of
`Examples/MNIST_CNN.c` (422K parameters) on 10,000 synthetic images: one epoch of mini-batches of
128 with AdamW, and inference, without and with batch normalization after each convolution and the
hidden dense layer; and ResNet-20 (`Examples/CIFAR10.c resnet20`, 273K parameters, 54 layers) on
4,096 synthetic 32 x 32 x 3 images: one epoch of mini-batches of 128 with SGD and momentum, and
inference; and the U-Net of `Examples/Segmentation.c` (118K parameters, 28 layers: transposed
convolutions, concatenations, three sigmoid outputs a pixel) on 1,024 synthetic 64 x 64 x 3 images:
one epoch of mini-batches of 32 with AdamW, and inference; and a MobileNet-style network (69K
parameters: a 3 x 3 convolution and four depthwise-separable blocks of 64 to 256 channels, each
convolution normalized) on 4,096 synthetic 32 x 32 x 3 images: one epoch of mini-batches of 128 with
SGD and momentum, and inference; and a GPT (nanoGPT's character-level configuration: six blocks of
layer normalization, causal attention of six heads and a GELU MLP, width 384, learned positions, a
context of 256 tokens of a vocabulary of 256; 10.9M parameters) on random tokens: one epoch of
mini-batches of 64 sequences with AdamW over 1,024 sequences (64 on the CPU), and the logits of every
position of half of them, in tokens per second. `Examples/benchmark_pytorch.py` runs the same
workloads in PyTorch. Run them with `Bin/Benchmark [threads]` and `python
Examples/benchmark_pytorch.py [threads]`.

Samples per second on a laptop (Intel Core i7-12650H: 6 performance and 4 efficiency cores, AVX2 and
AVX-VNNI, no AVX-512), medians of three interleaved runs. Spingalett 0.10 is built with GCC 16 and
uses its built-in kernels (no BLAS library); PyTorch 2.14.1 is the CPU build from PyPI (Intel MKL and
oneDNN). Against 0.9, measured in the same runs, 0.10 trains the convolutional networks 13 to 19%
faster and runs them 15 to 27% faster, and leaves the fully connected network as it was; 0.11 to 0.13
leave all of them as they were on the CPU (within 4% of the release before, either way, in
interleaved runs), and 1.2, 1.3 and 1.4 leave them as the release before did (within 3% either way, in
interleaved runs; 1.4 against 1.3 on one thread and on six performance cores). The U-Net, new in 0.13, is measured with 0.13, the MobileNet-style network, new in
0.14, with 0.14, the GPT, whose training 1.3 made 1.4 times as fast on one thread and 3.2 times on
eight, with 1.3:

| Fully connected network | Threads | Full batch | Mini-batch 64 | Inference |
|---|---:|---:|---:|---:|
| Spingalett | 1 | 25,700 | 20,200 | 67,100 |
| PyTorch | 1 | 24,000 | 10,800 | 60,800 |
| Spingalett (OpenMP) | 8 | 114,600 | 52,900 | 287,400 |
| PyTorch | 8 | 73,000 | 26,100 | 168,200 |

| Convolutional network | Threads | Training | Inference |
|---|---:|---:|---:|
| Spingalett | 1 | 3,090 | 11,180 |
| PyTorch | 1 | 2,260 | 4,370 |
| Spingalett (OpenMP) | 8 | 14,590 | 50,860 |
| PyTorch | 8 | 6,210 | 13,020 |
| **With batch normalization** | | | |
| Spingalett | 1 | 2,640 | 11,380 |
| PyTorch | 1 | 1,770 | 3,340 |
| Spingalett (OpenMP) | 8 | 11,030 | 55,650 |
| PyTorch | 8 | 4,630 | 10,160 |

| ResNet-20 | Threads | Training | Inference |
|---|---:|---:|---:|
| Spingalett | 1 | 377 | 1,387 |
| PyTorch | 1 | 342 | 900 |
| Spingalett (OpenMP) | 8 | 1,584 | 6,402 |
| PyTorch | 8 | 896 | 2,323 |

| U-Net, 64 x 64 | Threads | Training | Inference |
|---|---:|---:|---:|
| Spingalett | 1 | 146 | 511 |
| PyTorch | 1 | 134 | 292 |
| Spingalett (OpenMP) | 8 | 598 | 2,467 |
| PyTorch | 8 | 365 | 754 |

| MobileNet-style network | Threads | Training | Inference |
|---|---:|---:|---:|
| Spingalett | 1 | 797 | 3,762 |
| PyTorch | 1 | 589 | 1,648 |
| Spingalett (OpenMP) | 8 | 2,765 | 16,010 |
| PyTorch | 8 | 1,377 | 3,808 |

| GPT, tokens/s | Threads | Training | Inference |
|---|---:|---:|---:|
| Spingalett | 1 | 1,616 | 3,893 |
| PyTorch | 1 | 1,534 | 591 |
| Spingalett (OpenMP) | 8 | 7,065 | 19,043 |
| PyTorch | 8 | 4,314 | 1,679 |

Spingalett trains the convolutional network 1.4 to 2.3 times as fast as PyTorch and runs it 2.6 to
3.9 times as fast; with batch normalization 1.5 to 2.4 times and 3.4 to 5.5 times, since the
normalization runs in the convolution's epilogue at inference. ResNet-20 trains 1.1 to 1.8 times as
fast and runs 1.5 to 2.8 times as fast, the U-Net 1.1 to 1.6 times and 1.7 to 3.3 times, the
MobileNet-style network 1.35 to 2 times and 2.3 to 4.2 times, the GPT 1.05 to 1.64 times and 6.6 to
11.3 times. For the fully connected network Spingalett trains mini-batches 1.9 to 2 times as fast,
full batches 1.1 to 1.6 times and infers 1.1 to 1.7 times as fast. The indirect convolution kernels
of 0.10 account for the gains over 0.9: built without them (`-DSPINGALETT_NO_DIRECT_CONV`), ResNet-20
trains at 234 and 986 samples per second and infers at 958 and 4,798.

On the laptop's GPU (NVIDIA GeForce RTX 4050 Laptop GPU, 6 GB, driver 610.43), Spingalett 1.3 runs
the same workloads with `SPINGALETT_COMPUTE_CUDA` and `SPINGALETT_COMPUTE_VULKAN` (`Bin/Benchmark
gpu`), in single precision and in bfloat16 (`spingalett_set_gpu_precision()`), against PyTorch 2.14.1
with CUDA 13.0, cuBLAS and cuDNN 9.24, as it runs by default (`python Examples/benchmark_pytorch.py
--cuda`, with TF32 convolutions; `--cuda-fp32` in single precision; `--cuda-bf16`, autocast to
bfloat16) and at its fastest (`--max`: `torch.compile` with `max-autotune` and CUDA graphs,
`cudnn.benchmark`, channels-last tensors, fused optimizers). The figures are medians of three rounds.
The laptop's GPU slows down by up to 12% as it warms, so each block of runs (Spingalett on CUDA,
PyTorch by default, PyTorch at its fastest, Spingalett on Vulkan) starts with the GPU at 60 °C, and
the blocks' order changes from round to round. The GPT, which comes last in a block, is measured on
its own (`Bin/Benchmark gpu cuda bf16 gpt`, `benchmark_pytorch.py --cuda-bf16 --max --gpt`), each run
from the GPU at 60 °C: a precision of a backend a run, PyTorch a configuration a run. The fully
connected network's mini-batches vary by up to 8% from run to run, in 1.2 as in 1.3. The first four
columns keep the data in GPU memory from the start: Spingalett's data sets
(`spingalett_device_data_new()`), PyTorch's tensors. The last two take it from host memory:
Spingalett's arrays, each chunk copied over while the GPU works on the ones before, and PyTorch's
pinned tensors, each batch copied as it is used and the outputs of inference copied back
(`--host-data`). Inference is timed after a first call (Spingalett) or a warm-up step (PyTorch):

| On the GPU, samples/s | CUDA, data on GPU (FP32 / bf16) | Vulkan, data on GPU (FP32 / bf16) | PyTorch, data on GPU (TF32 / FP32 / bf16) | PyTorch `--max`, data on GPU (TF32 / FP32 / bf16) | CUDA, host data (FP32 / bf16) | PyTorch, host data (TF32 / bf16) |
|---|---:|---:|---:|---:|---:|---:|
| ResNet-20, training | 11,840 / 22,640 | 10,830 / 18,690 | 8,528 / 7,713 / 12,320 | 9,808 / 8,917 / 20,410 | 11,790 / 22,620 | 8,457 / 12,160 |
| ResNet-20, inference | 43,330 / 89,300 | 40,200 / 75,680 | 19,860 / 19,790 / 31,630 | 33,820 / 23,250 / 73,570 | 43,030 / 89,160 | 19,470 / 30,690 |
| Convolutional network, training | 117,400 / 219,400 | 110,000 / 169,600 | 62,050 / 61,210 / 102,500 | 92,160 / 88,530 / 117,000 | 117,900 / 222,000 | 57,390 / 99,220 |
| Convolutional network, inference | 389,500 / 995,400 | 345,300 / 506,900 | 132,700 / 144,200 / 238,200 | 214,600 / 180,000 / 427,400 | 385,900 / 982,500 | 128,300 / 223,900 |
| With batch normalization, training | 80,970 / 153,600 | 75,880 / 122,400 | 48,480 / 48,860 / 83,470 | 66,210 / 63,840 / 107,400 | 80,720 / 154,700 | 48,160 / 57,900 |
| With batch normalization, inference | 376,100 / 938,000 | 318,400 / 478,700 | 107,400 / 114,400 / 198,200 | 214,000 / 180,100 / 427,100 | 379,500 / 947,800 | 104,300 / 188,000 |
| Fully connected network, mini-batch 64 | 349,700 / 531,800 | 322,500 / 443,400 | 88,040 / 79,800 / 70,040 | 87,730 / 113,400 / 129,400 | 383,000 / 557,800 | 91,530 / 73,800 |
| Fully connected network, full batch | 1,313,000 / 3,270,000 | 1,285,000 / 2,903,000 | 1,049,000 / 1,055,000 / 2,546,000 | 1,016,000 / 1,021,000 / 2,393,000 | 1,228,000 / 3,023,000 | 844,600 / 1,573,000 |
| Fully connected network, inference | 3,806,000 / 10,174,000 | 3,621,000 / 8,085,000 | 2,958,000 / 2,958,000 / 6,281,000 | 2,616,000 / 2,626,000 / 5,243,000 | 2,990,000 / 6,472,000 | 1,709,000 / 2,465,000 |
| U-Net, training | 4,546 / 8,230 | 4,193 / 7,092 | 3,339 / 3,113 / 5,014 | 4,092 / 3,599 / 8,028 | 4,492 / 8,254 | 3,257 / 4,586 |
| U-Net, inference | 15,340 / 31,820 | 15,220 / 25,760 | 6,467 / 6,711 / 12,000 | 11,200 / 7,813 / 26,780 | 15,310 / 31,620 | 6,113 / 11,070 |
| MobileNet-style network, training | 15,170 / 33,430 | 14,700 / 31,850 | 9,895 / 8,692 / 22,910 | 9,116 / 8,232 / 21,980 | 14,960 / 33,350 | 9,773 / 22,280 |
| MobileNet-style network, inference | 120,000 / 183,800 | 116,700 / 163,500 | 19,590 / 17,900 / 38,920 | 40,700 / 40,740 / 78,000 | 118,800 / 182,200 | 19,210 / 37,430 |
| GPT, training (tokens/s) | 69,230 / 197,600 | 68,330 / 181,300 | 64,030 / 64,090 / 156,800 | 66,460 / 66,410 / 193,000 | 73,560 / 198,700 | 64,060 / 156,800 |
| GPT, inference (tokens/s) | 224,200 / 684,800 | 217,600 / 583,100 | 208,100 / 208,400 / 517,800 | 207,400 / 207,200 / 655,000 | 231,800 / 687,300 | 203,200 / 497,400 |

On CUDA Spingalett is ahead of PyTorch at its fastest in every workload. With the data in GPU memory,
in single precision, against the fastest of PyTorch's runs in single precision or with TF32
convolutions, it trains ResNet-20 1.21 times as fast and runs it 1.28 times as fast, the
convolutional networks 1.2 to 1.3 and 1.8 times, the U-Net 1.11 and 1.37 times, the MobileNet-style
network 1.5 and 2.9 times, the GPT 1.04 and 1.08 times, and the fully connected network 1.24 times as
fast in full batches and 3.1 times in mini-batches, and infers 1.29 times as fast. In bfloat16,
against PyTorch's fastest run in bfloat16: ResNet-20 1.11 and 1.21 times, the convolutional networks
1.4 to 1.9 and 2.2 to 2.3 times, the U-Net 1.03 and 1.19 times, the MobileNet-style network 1.46 and
2.4 times, the GPT 1.02 and 1.05 times (where 1.2.0, measured the same way, trained at 0.94 times and
inferred at 0.89 times the speed of PyTorch compiled), the fully connected network 1.28 times in full
batches, 4.1 times in mini-batches and 1.6 times in inference. Against PyTorch by default the margins
are wider: in single precision ResNet-20 trains 1.39 times as fast as PyTorch with TF32 (1.53 times
as fast as in single precision) and infers 2.18 times as fast, and the MobileNet-style network infers
6.1 times as fast; in bfloat16 ResNet-20 trains 1.84 times as fast and infers 2.8 times as fast, the
GPT trains 1.26 times as fast and infers 1.32 times as fast, and the fully connected network trains
mini-batches 7.6 times as fast. From host arrays the margins are as wide or wider (PyTorch copies
each batch as it uses it; Spingalett copies a chunk while the GPU works on the ones before): the
fully connected network, whose inference the bus binds, infers 1.75 times as fast in single precision
and 2.6 times in bfloat16.

The CUDA backend is the faster on this GPU: against the Vulkan backend (the same executor) it trains
the other networks 1.02 to 1.09 times as fast in single precision and 1.05 to 1.29 times in bfloat16,
and infers them 1.01 to 1.18 times as fast in single precision and 1.12 to 1.96 times in bfloat16,
where its products of the tensor cores, its kernel for a network's first convolution (`dconv.cu`) and
its depthwise convolutions pull ahead. The GPT, whose attention runs on the matrix units in bfloat16
and in register tiles in single precision on both backends since 1.3, trains 1.01 times as fast on
CUDA in single precision and 1.09 times in bfloat16, and infers 1.03 and 1.17 times as fast: CUDA's
products of 128 x 128 outputs over three stages (`gemm_pipe.cu`) are ahead of Vulkan's on cooperative
matrices. Against 1.2.0, measured in the same way (both built alike, in interleaved runs of every
workload), 1.3 trains the GPT on Vulkan 3.1 to 3.2 times as fast in bfloat16 and 1.9 times in single
precision and infers it 2.6 to 2.7 and 1.6 times as fast, on CUDA 1.10 and 1.04 times as fast and
infers it 1.12 to 1.14 and 1.03 to 1.05 times as fast, and runs the other workloads as 1.2 did:
within 3% either way, but for a few rows that moved by up to 6% either way in those runs as the GPU
warmed, of which the slower ran within 2.5% of 1.2 in runs of their workloads alone. In single
precision Spingalett trains ResNet-20 on the GPU 7.4 times as fast as on the eight threads of the
CPU.

0.9 took its time out of the calls these workloads do not measure: small batches and single
samples, files, data sets and Python. Same VM, 4 threads, 0.8 against 0.9 (see the
[CHANGELOG](../CHANGELOG.md) for more):

| Call | 0.8 | 0.9 |
|---|---:|---:|
| `spingalett_model_predict()`, one sample, 784-256-128-10 MLP, FP32 / FP16 / INT8 | 63 / 403 / 10.5 us | 12 / 8 / 4.4 us |
| the same, a batch of 1024, FP16 / INT8, per sample | 2.04 / 0.54 us | 0.92 / 0.31 us |
| training that MLP with batches of 2 / 8, 4096 samples | 0.57 / 0.093 s | 0.33 / 0.076 s |
| loading a 924,930-parameter FP32 network / INT8 network from memory | 26 / 13 ms | 8.3 / 1.3 ms |
| saving it in FP32 / FP16 | 10.8 / 15.2 ms | 2.5 / 1.4 ms |
| an epoch of a small CNN on CIFAR-10 streamed from a `.slettd` file (from float arrays: 4.7 / 4.5 s) | 14.7 s | 5.5 s |
| Python `spingalett_forward()` of one sample through that MLP | 69 us | 22 us |

Deployment models of the CIFAR-10 network of `Examples/CIFAR10.c` (551K parameters, six
convolutions with batch normalization folded in), `spingalett_model_predict()` on 4,000 test
images, images per second; 0.7's integer kernels against 0.8's (float layers are unchanged):

| CIFAR-10 network | Threads | FP32 | FP16 | INT8 | INT4 |
|---|---:|---:|---:|---:|---:|
| 0.7 integer kernels | 1 | 1,020 | 950 | 810 | 830 |
| 0.8 | 1 | 1,010 | 970 | 2,520 | 2,960 |
| 0.7 integer kernels | 4 | 3,420 | 3,550 | 2,710 | 2,500 |
| 0.8 | 4 | 3,530 | 3,710 | 7,380 | 7,950 |

With the AVX-512 VNNI tiles an INT8 model predicts 3.1 times as fast as before on one thread and
2.7 times on four, and 2.5 and 2.1 times as fast as the FP32 model. One image at a time
(`spingalett_model_run()`), the INT8 network runs 1.7 to 1.8 times as fast as in 0.7 (1,580
against 860 images per second), its convolutions now taking four pixels against four filters at a
time.

The x86-64 release packages are compiled for a baseline instruction set and choose AVX2 or AVX-512
matrix kernels at run time (since 0.6), and AVX-512 VNNI or AVX-VNNI integer kernels (since 0.8).
On the Cascade Lake VM where 0.6 was measured, on one thread, the baseline package trains 3.6 to
4.9 times as fast, and infers 5 times as fast, as with kernels for its baseline (SSE2), as in 0.5:

| Baseline x86-64 build, one thread | Full batch | Mini-batch 64 | Inference |
|---|---:|---:|---:|
| SSE2 kernels (0.5) | 3,700 | 3,200 | 9,000 |
| AVX-512 kernels chosen at run time (0.6) | 18,200 | 11,400 | 45,000 |

Language models' GGUF files as they are stored (`spingalett_model_open()`), against llama.cpp on the same
files: a session fed a prompt of 512 tokens at once, then 128 tokens one at a time from an empty sequence, as
`llama-bench -p 512 -n 128` measures them (llama.cpp at commit 1242be6 of 2026-10-21, built with GCC 16 for the
native processor, its CPU backend with OpenMP), tokens per second on the six performance cores of the i7-12650H
(both pinned to them), medians of three interleaved runs:

| Model, type | Prompt of 512: Spingalett | llama.cpp | Generation of 128: Spingalett | llama.cpp |
|---|---:|---:|---:|---:|
| Qwen2.5-0.5B-Instruct, Q4_0 | 1,462 | 866 | 130.9 | 130.2 |
| Qwen2.5-0.5B-Instruct, Q4_K_M | 1,384 | 476 | 119.9 | 109.3 |
| Qwen2.5-0.5B-Instruct, Q8_0 | 1,320 | 563 | 94.8 | 93.2 |
| Qwen3-0.6B, Q8_0 | 811 | 413 | 80.2 | 77.5 |
| Llama-3.2-1B-Instruct, Q4_K_M | 478 | 292 | 60.1 | 59.3 |
| Llama-3.2-1B-Instruct, Q8_0 | 541 | 210 | 43.1 | 42.9 |
| Gemma-3-1B-it, Q4_K_M | 635 | 219 | 58.9 | 56.4 |

Prompts run on the integer tiles of the block precisions (two groups of eight tokens against four rows a pass)
and on attention split between the threads by its work; generation reads the weights at the memory's speed,
where both libraries are bound. The x86-64 packages, built for a baseline, choose the same kernels at run time
and run as fast (Qwen2.5-0.5B Q4_K_M, medians of five interleaved runs: 1,431 and 118.7 tokens a second against
the native build's 1,441 and 118.8).

## Language models on the GPU

Tokens a second on an RTX 4050 Laptop GPU with an i7-12650H, the GPU at 60 C or less before each run, against llama.cpp's
Vulkan build (1242be6; `llama-bench -p 512 -n 128`), measured the same day, with `Bin/LlmBench model.gguf -d cuda` and `-d vulkan`.
llama.cpp's CUDA build was not measured:

| Model | CUDA prompt | Vulkan prompt | llama.cpp | CUDA tokens | Vulkan tokens | llama.cpp |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-0.6B Q8_0 | 16,447 | 6,078 | 12,120 | 222 | 157 | 207 |
| Qwen3-1.7B Q8_0 | 8,117 | 2,321 | 6,543 | 90.7 | 73.6 | 90.2 |
| Qwen3-4B Q4_K_M | 3,601 | 789 | 2,974 | 64.9 | 52.5 | 65.2 |
| Llama-3.2-1B Q4_K_M | 12,682 | 2,717 | 10,473 | 196 | 153 | 198 |
| Qwen2.5-0.5B Q4_K_M | 22,215 | 6,243 | 16,265 | 326 | 189 | 290 |

A model larger than the memory it is given splits with the CPU. Qwen3-4B Q4_K_M, prompt / tokens: 2 GB of the GPU 2,465 / 34.0, 1.2 GB
1,771 / 26.5, 0.7 GB 1,502 / 19.2; llama.cpp with 30, 20, 10 and 0 of its 37 layers on the GPU 1,816 / 41.0, 1,490 / 30.9, 1,258 / 25.2,
1,062 / 19.3. Qwen3-8B Q4_K_M with what a 6 GB card has free 1,946 / 29.6; llama.cpp, which fits it whole, 1,705 / 35.6.

## Graphs of operators on the GPU

Milliseconds on the same GPU, medians, against PyTorch 2.14.1. Spingalett's times take the inputs from and the outputs to the host;
PyTorch's have its tensors already on the GPU, then, after the slash, on the host too:

| Model | bfloat16 | PyTorch fp16 autocast | Single precision | PyTorch fp32 |
|---|---:|---:|---:|---:|
| ResNet-50 | 1.86 | 1.90 | 3.50 | 3.71 |
| BERT-base, 2 x 128 tokens | 3.40 | 3.79 / 3.94 | 10.1 | 9.0 |
| GPT-2, 64 tokens | 3.76 | 3.03 / 4.19 | 7.2 | 4.7 |
| CLIP's text encoder | 1.86 | 1.94 | 4.74 | 5.13 |
| Whisper's encoder | 3.51 | 3.04 | 8.59 | 9.74 |
| ViT-B/16 | 3.47 | 3.30 | 10.2 | 8.6 |
| A UNet of SDXL's architecture | 4.01 | 13.6 | 8.05 | 11.45 |
| Stable Diffusion's VAE decoder | 54 | 58 | 138 | 80 |

## Examples' accuracy

`Examples/MNIST.c` trains a 784-256-128-10 network with dropout (AdamW, cosine schedule,
mini-batches of 128) on 55,000 images, keeps the epoch with the best accuracy on the other 5,000
and reaches 98.2% test accuracy after 10 epochs, which take a few seconds with OpenMP on the same
VM. `Examples/MNIST_CNN.c` reaches 98.9% after two epochs of about 9 seconds each on four threads.
`Examples/CIFAR10.c` reaches 87.8% test accuracy after 20 epochs, 20 minutes on four threads of the
Sapphire Rapids VM (760 images per second with augmentation); its INT8 model keeps 87.8%. With
`resnet20` it reaches 91.55% after 100 epochs (12 threads of the i7-12650H, before the indirect
kernels: 93 minutes; FP16 model 91.56%, INT8 91.58% in 281 KB), and the same 91.55% trained on the
RTX 4050 Laptop GPU (`gpu`) in 9 minutes, 91.80% in bfloat16 (`bf16`) in 5 minutes (FP16 model
91.83%, INT8 91.77%); with `resnet32`, 92.39% (INT8 model 92.43% in 480 KB).
`Examples/Segmentation.c` reaches a mean intersection over union of 0.882 on its test images
(circles 0.925, squares 0.866, triangles 0.855; 98.5% of the pixels right) after 12 epochs of 4,096
new images, 15 s on the RTX 4050 and 80 s on the i7-12650H's CPU; its INT8 model keeps 0.878, and
predicts 3,400 images a second on the CPU against 3,100 for the FP32 model.
