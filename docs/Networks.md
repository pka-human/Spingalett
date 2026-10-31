# Networks

Declaring networks of layers: dense and convolutional layers, normalization, graphs of layers, and reading a network back.

## Defining a network

`spingalett_network_new(loss)` creates an empty network; each `spingalett_layer()` call appends a
layer, the first one being the input layer. `SpingalettLayerArgs` fields:

| Field | Meaning |
|---|---|
| `neurons_amount` | Layer width |
| `act_func` | Activation of this layer (ignored for the input layer); left out, `SPINGALETT_ACT_NONE` |
| `weight_initialization` | `RANDOM` (uniform in [-1, 1]), `XAVIER` (Glorot normal, variance 2/(fan_in + fan_out)), `HE` (normal, variance 2/fan_in), `LECUN` (normal, variance 1/fan_in), `NONE` (zeros) |
| `dropout_rate` | Probability in [0, 1) of zeroing each output of this layer during training |

Biases start at zero. Softmax is only allowed in the output layer. Leaky ReLU uses a slope of
0.01; FOO52 is the piecewise-linear function `0.01x` for `x < 0`, `x` on `[0, 1]` and
`1 + 0.01(x - 1)` above 1.

Dropout is inverted (kept units are scaled by `1 / (1 - p)`), so inference needs no rescaling.
It is not applied to the input or output layer.

## Convolution and pooling

Data flows through a network as one tensor per sample, `height x width x channels` floats in
channels-last order: element `(y, x, c)` is at `(y * width + x) * channels + c`, so an image stored
row by row with interleaved channels goes in as it is. Give the input layer a shape, then add layers
with `spingalett_conv2d()`, `spingalett_max_pool2d()` and `spingalett_avg_pool2d()`; a dense layer
reads whatever precedes it as a flat vector, so no flattening layer is needed:

```c
NeuralNetwork *net = new_spingalett(LOSS_CROSS_ENTROPY);
layer(.net = net, .height = 28, .width = 28, .channels = 1);                    /* input */
conv2d(.net = net, .filters = 32, .kernel = 3, .padding = 1,                    /* 28 x 28 x 32 */
       .act_func = ACT_RELU, .weight_initialization = WEIGHT_INITIALIZATION_HE);
max_pool2d(.net = net, .kernel = 2);                                            /* 14 x 14 x 32 */
conv2d(.net = net, .filters = 64, .kernel = 3, .padding = 1,                    /* 14 x 14 x 64 */
       .act_func = ACT_RELU, .weight_initialization = WEIGHT_INITIALIZATION_HE);
max_pool2d(.net = net, .kernel = 2);                                            /* 7 x 7 x 64 */
layer(net, 128, ACT_RELU, WEIGHT_INITIALIZATION_HE, .dropout_rate = 0.3f);
layer(net, 10, ACT_SOFTMAX, WEIGHT_INITIALIZATION_XAVIER);
```

| Field | Meaning |
|---|---|
| `height`, `width`, `channels` | Shape of the input layer (its `neurons_amount` may be left 0) |
| `filters` | Output channels of a convolution |
| `kernel`, `kernel_h`, `kernel_w` | Window size; `kernel` sets both, `kernel_h` / `kernel_w` override one axis |
| `stride`, `stride_h`, `stride_w` | Window step: 1 for convolutions and the kernel size for pooling unless set |
| `padding`, `padding_h`, `padding_w` | Zero cells added on both sides of each axis; smaller than the kernel |
| `groups` | Convolutions: split the input channels and the filters into this many groups; each filter sees the input channels of its own group. As many groups as input channels make a depthwise convolution (with any number of filters per channel) |

An output axis has `(size + 2 padding - kernel) / stride + 1` cells. A convolution's filters are
`kernel_h x kernel_w x (input channels / groups)` weights each, plus one bias, and take an
activation, a weight initialization (fan-in is the window size) and dropout like dense layers.
Pooling has no parameters and no activation; windows are clipped to the input, so padding cells
never count (average pooling divides by the cells inside), and max pooling passes the gradient to
the first maximum. Every training strategy, optimizer, schedule, the custom training loop and
`spingalett_predict()` work with these layers as with dense ones.

Convolutions run as indirect matrix products: the matrix kernels read each pixel's window straight
from the image through one pointer per kernel tap, so no window is ever gathered or copied, with the
bias and activation applied to each tile of the result while it is in cache. The data gradient of
strided convolutions runs a phase of input cells at a time, over the taps that cover them only, and
weight gradients split their long summation over all pixels into slots fixed by the shape that the
threads share. Depthwise convolutions are computed directly, all of a pixel's channels at once, and
other grouped ones as one product per group. The network of `Examples/MNIST_CNN.c` trains 1.4 to
2.3 times as fast as in PyTorch and infers 2.6 to 3.9 times as fast, ResNet-20 1.1 to 1.8 and 1.5 to
2.8 times as fast (see [Performance](Performance.md)).

## Batch normalization

`spingalett_batch_norm()` normalizes the previous layer per channel (per unit after a dense layer):
`y = act(gamma (x - mean) / sqrt(var + epsilon) + beta)`. While training, mean and variance are
those of the batch (of each chunk of up to 2,048 samples in full-batch training), and running
averages of them move `momentum` of the way towards each batch's; `spingalett_predict()`,
`spingalett_forward()`, `spingalett_evaluate()` and deployment models use the running averages.
Gamma starts at 1 and beta at 0; they are the layer's weights and biases for every optimizer, but
weight decay leaves them alone. The convolution or dense layer before a normalization usually takes
no activation:

```c
conv2d(.net = net, .filters = 64, .kernel = 3, .padding = 1,
       .weight_initialization = WEIGHT_INITIALIZATION_HE);                      /* no activation */
batch_norm(.net = net, .act_func = ACT_RELU);                         /* .epsilon 1e-5, .momentum 0.1 */
```

Such a pair costs no more than the convolution alone at inference: `spingalett_predict()` and
`spingalett_forward()` apply the normalization while the convolution's tiles are in cache, and
deployment models (and files saved in other precisions than FP32 without optimizer state) fold it
into the convolution's weights and biases. A normalization of a dense layer needs batches of at
least 2 samples. `SPINGALETT_PARAM_RUNNING_MEAN` and `SPINGALETT_PARAM_RUNNING_VARIANCE` read and
write the statistics, and `restore_best_weights` restores them with the weights.

## Graphs: residual connections and branches

Every builder returns the index of the layer it adds (`SPINGALETT_NO_LAYER` on error), and `.inputs`
names the earlier layers a layer reads; without it a layer reads the one added before it, so chains
are built as before. Two kinds of layers combine others: `spingalett_add_layers()` sums layers of
one shape and `spingalett_concat_layers()` puts layers of one height and width side by side along
the channels, each followed by its `.act_func` (none when left out).
`spingalett_global_avg_pool2d()` averages each channel over all cells. A residual block of ResNet:

```c
uint32_t block(NeuralNetwork *net, uint32_t x, uint32_t filters, uint32_t stride) {
    conv2d(.net = net, .inputs = {x}, .filters = filters, .kernel = 3, .padding = 1,
           .stride = stride, .weight_initialization = WEIGHT_INITIALIZATION_HE);
    batch_norm(.net = net, .act_func = ACT_RELU);
    conv2d(.net = net, .filters = filters, .kernel = 3, .padding = 1,
           .weight_initialization = WEIGHT_INITIALIZATION_HE);
    uint32_t y = batch_norm(.net = net);
    uint32_t shortcut = x;
    if (stride != 1) {                      /* the shortcut changes shape too: a 1 x 1 projection */
        conv2d(.net = net, .inputs = {x}, .filters = filters, .kernel = 1, .stride = stride,
               .weight_initialization = WEIGHT_INITIALIZATION_HE);
        shortcut = batch_norm(.net = net);
    }
    return add_layers(.net = net, .inputs = {shortcut, y}, .act_func = ACT_RELU);
}
```

and Inception-style branches, concatenated:

```c
uint32_t x = layer(.net = net, .height = 32, .width = 32, .channels = 16);
uint32_t a = conv2d(.net = net, .inputs = {x}, .filters = 8, .kernel = 1,
                    .act_func = ACT_RELU);
uint32_t b = conv2d(.net = net, .inputs = {x}, .filters = 8, .kernel = 3, .padding = 1,
                    .act_func = ACT_RELU);
uint32_t c = max_pool2d(.net = net, .inputs = {x}, .kernel = 3, .stride = 1, .padding = 1);
concat_layers(.net = net, .inputs = {a, b, c});                                 /* 32 x 32 x 32 */
```

`.input_count` gives the number of inputs; left 0 it counts `.inputs` up to the last nonzero entry,
so it is needed when the last input is the input layer (`.inputs = {a, 0}, .input_count = 2`). A
layer reads at most `SPINGALETT_MAX_INPUTS` (16) layers. The last layer is the network's output,
and every other layer must feed a later one before the network trains or predicts.

Layers run in the order they were added, which puts each after its inputs. Training follows the
graph backwards: a layer that feeds several gets their gradients in a fixed order, so training is
still the same bits on any number of threads. At inference, outputs share memory where their lives
do not overlap, in `spingalett_predict()`, deployment models and the engine (whose `.slett` files,
format version 6, record where each output lives): a residual network of 16 blocks needs memory for
a few of its widest outputs, not for all of them. `spingalett_network_layer()` reports each layer's
`inputs`, and `ModelTool info` lists them. `Examples/CIFAR10.c` builds ResNet-20 to ResNet-56
(`resnet20`, `resnet32`, ...).

## Transposed convolutions, upsampling and layer normalization

`spingalett_conv_transpose2d()` is the transpose of a convolution: each input cell spreads its
channels over a window of the output through the filters, its windows `stride` cells apart, so a 2 x
2 kernel of stride 2 doubles the height and width. `spingalett_upsample2d()` repeats or interpolates
cells by integer factors, and `spingalett_layer_norm()` normalizes each cell over its channels. With
concatenations they make U-Nets, the expanding path joined to the maps of the same size on the way
down:

```c
uint32_t e = conv2d(.net = net, .filters = 32, .kernel = 3, .padding = 1,
                    .act_func = ACT_RELU);                                      /* 64 x 64 x 32 */
max_pool2d(.net = net, .kernel = 2);                                            /* 32 x 32 x 32 */
conv2d(.net = net, .filters = 64, .kernel = 3, .padding = 1,
       .act_func = ACT_RELU);                                                   /* 32 x 32 x 64 */
uint32_t u = conv_transpose2d(.net = net, .filters = 32, .kernel = 2, .stride = 2,
                              .act_func = ACT_RELU);                            /* 64 x 64 x 32 */
concat_layers(.net = net, .inputs = {e, u});                                    /* 64 x 64 x 64 */
upsample2d(.net = net, .stride = 2,                                             /* 128 x 128 x 64 */
           .upsample = UPSAMPLE_BILINEAR);
layer_norm(.net = net, .act_func = ACT_RELU);                                   /* 128 x 128 x 64 */
```

| Builder | Arguments | Output, along each axis |
|---|---|---|
| `spingalett_conv_transpose2d()` | `filters`, `kernel`, `stride`, `padding`, `output_padding` (each with `_h` and `_w` forms), `groups`, `act_func` | `(size - 1) stride - 2 padding + kernel + output_padding` cells |
| `spingalett_upsample2d()` | `stride` (or `stride_h`, `stride_w`): the factors, 2 by default; `upsample`: `SPINGALETT_UPSAMPLE_NEAREST` (default) or `SPINGALETT_UPSAMPLE_BILINEAR` | `size x factor` cells |
| `spingalett_layer_norm()` | `epsilon` (1e-5), `act_func` | the input's shape |

- A transposed convolution's output padding (less than the stride) adds cells at the end of each
  axis, so that the output can match the input of the convolution it transposes: a 3 x 3
  convolution of stride 2 and padding 1 halves 64 cells to 32, and its transpose with
  `.output_padding = 1` makes 64 again. Its filter `j` holds, at `(kh, kw, c)`, the weight by which
  input channel `c` of its group reaches output channel `j` at offset `(kh, kw)` of the input
  cell's window: `kernel_h x kernel_w x (input channels / groups)` weights an output channel, as a
  convolution's filters have. It runs on the convolution kernels with the passes swapped: its
  forward pass is the data gradient of the convolution it transposes (a phase of the stride at a
  time, over the taps that reach it), its data gradient that convolution's forward pass.
- Bilinear upsampling interpolates between the cells' centres, repeating the cells at the edges
  (PyTorch's `align_corners=False`); its weights come from integers, so every platform computes
  the same ones. Upsampling has no parameters and no activation.
- Layer normalization computes the mean and variance of each cell's channels (a dense layer's
  outputs are a single cell): `y = act(gamma (x - mean) / sqrt(var + epsilon) + beta)`, gamma and
  beta per channel. It has no running statistics, so training and inference compute the same at any
  batch size; weight decay leaves gamma alone. Unlike batch normalization it is not folded into the
  layer before it: deployment models and the engine run it as a layer of its own.
- All three train on the CPU and the GPU, save in `.slett` format version 7, and run in deployment
  models in every precision and in the engine; batch normalization after a transposed convolution
  is folded into it. Float deployment models run transposed convolutions through the same kernels
  as training, integer ones through the engine's pass.

`Examples/Segmentation.c` segments synthetic images of circles, squares and triangles with a U-Net
of transposed convolutions (or bilinear upsampling, or layer normalization): 12 epochs take 15 s on
an RTX 4050 Laptop GPU and reach a mean intersection over union of 0.88, which its INT8 model keeps
(see [Performance](Performance.md)).

## Inspecting a network

`SpingalettNetwork` is an opaque handle: its layout is private, and accessors describe it.

```c
uint32_t layers = spingalett_layer_count(net);              /* the input layer included */
SpingalettNetworkLayer d;
spingalett_network_layer(net, 1, &d);   /* type, shape, outputs, activation, dropout, window, counts */

float *w = malloc(d.weight_count * sizeof(float));
spingalett_get_parameters(net, 1, PARAM_WEIGHTS, w, d.weight_count);               /* feeding layer 1 */
spingalett_set_parameters(net, 1, PARAM_WEIGHTS, w, d.weight_count);
```

Parameters come as `SPINGALETT_PARAM_WEIGHTS`, `SPINGALETT_PARAM_BIASES`,
`SPINGALETT_PARAM_WEIGHT_GRADIENTS`, `SPINGALETT_PARAM_BIAS_GRADIENTS` and, for batch normalization,
`SPINGALETT_PARAM_RUNNING_MEAN` and `SPINGALETT_PARAM_RUNNING_VARIANCE`. Dense weights are
`outputs x inputs` (row `j` holds the weights into unit `j`); convolution weights are
`filters x kernel_h x kernel_w x (input channels / groups)`; batch normalization has gamma as
weights and beta as biases, one per channel. Also available: `spingalett_input_size()`,
`spingalett_output_size()`, `spingalett_parameter_count()`, `spingalett_network_loss()` and
`spingalett_optimizer_steps()`.
