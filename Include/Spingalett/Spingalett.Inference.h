/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/*
 * Spingalett inference engine.
 *
 * Runs a network stored as a .slett image (format version 3; 4 for networks with convolution and
 * pooling layers; 5 for networks with batch normalization or grouped convolutions; 6 for networks
 * whose layers read other layers than the one before them, such as residual connections; 7 for
 * transposed convolutions, upsampling and layer normalization; 8 for the layers and activations of
 * transformers: embeddings, attention, RMS normalization, products of layers, GELU and SiLU) in place,
 * in the precision its
 * parameters were saved in. The image can be a file read into memory, a const array
 * compiled into the program (see spingalett_export_c_header) or a region of flash. The engine allocates nothing,
 * does no I/O and keeps no global state: apart from the image it only needs a workspace from the
 * caller, so it runs on microcontrollers as well as on desktops, and any number of threads can share
 * one model.
 *
 * This header and Src/Spingalett.Inference.c are self-contained. Compiled together with
 * -DSPINGALETT_INFERENCE_ONLY (and Include/ on the include path) they give inference without the
 * rest of the library: no training code, no file I/O, no OpenMP, no heap. The full library includes
 * this header from Spingalett.h and adds loading, quantization and batched inference on top of it.
 *
 * Layers whose weights are stored as INT8, INT4 or INT2 run in integer arithmetic: the layer's input
 * is quantized to 8 bits with a scale chosen per sample (its largest magnitude maps to 127), the dot
 * products with the weights accumulate in 32-bit integers, and each output is rescaled by its weight
 * row's scale. FLOAT32, FP16 and BFLOAT16 layers run in float. Convolutions compute each output
 * pixel as such dot products of the filters with the window it reads; pooling and batch
 * normalization run in float, and so do the layers that add or concatenate the outputs of others.
 */
#ifndef SPINGALETT_INFERENCE_H
#define SPINGALETT_INFERENCE_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#if !defined(SPINGALETT_API)
#  if defined(SPINGALETT_STATIC) || defined(SPINGALETT_INFERENCE_ONLY)  /* static library or compiled in */
#    define SPINGALETT_API
#  elif defined(_WIN32) || defined(__CYGWIN__)
#    ifdef SPINGALETT_EXPORTS
#      define SPINGALETT_API __declspec(dllexport)
#    else
#      define SPINGALETT_API __declspec(dllimport)
#    endif
#  else
#    define SPINGALETT_API __attribute__((visibility("default")))
#  endif
#endif

/* Newest version of the .slett model format. spingalett_save() writes the oldest version that holds
   the network: 3 for dense layers only, 4 with convolution or pooling layers, 5 with batch
   normalization or grouped convolutions, 6 for graphs (layers that read other layers than the one
   before them, add or concatenate several, or pool globally), 7 with transposed convolutions,
   upsampling or layer normalization, 8 with embeddings, attention, RMS normalization, products of
   layers, the GELU and SiLU activations or the sparse cross-entropy loss, so that engines of earlier
   releases still run what they can; versions 1 and 2 still load. */
#define SPINGALETT_FORMAT_VERSION 8

/* Most inputs a layer can have (SPINGALETT_LAYER_ADD, SPINGALETT_LAYER_CONCAT and SPINGALETT_LAYER_MULTIPLY read
   several). */
#define SPINGALETT_MAX_INPUTS 16

/* File name extensions: models (spingalett_save() appends it when the name has none) and data sets. */
#define SPINGALETT_MODEL_EXTENSION   ".slett"
#define SPINGALETT_DATASET_EXTENSION ".slettd"

/* Every public struct ends with SPINGALETT_RESERVED 64-bit words of reserved space, zero: room for the
   fields of later 1.x releases, which programs built before them leave zero (the builders' designated
   initializers zero what they do not name), so that their meaning then is what zero means. Structs the
   library fills have them zeroed. */
#define SPINGALETT_RESERVED 8

#define SPINGALETT_OK                   0
#define SPINGALETT_ERR_ALLOC            1   /* out of memory */
#define SPINGALETT_ERR_INVALID          2   /* invalid argument or file contents */
#define SPINGALETT_ERR_FILE_IO          3   /* file could not be opened, read or written, or is truncated */
#define SPINGALETT_ERR_FORMAT_VERSION   4   /* model file written by an unsupported format version */

/* Activations (0 is none). .slett files keep codes of their own (docs/ModelFormat.md). */
typedef enum {
    SPINGALETT_ACT_NONE,
    SPINGALETT_ACT_SIGMOID,
    SPINGALETT_ACT_RELU,
    SPINGALETT_ACT_TANH,
    SPINGALETT_ACT_LEAKY_RELU,                 /* slope 0.01 below zero */
    SPINGALETT_ACT_FOO52,                      /* slope 0.01 below 0 and above 1, identity between */
    SPINGALETT_ACT_SOFTMAX,                    /* over the layer's outputs */
    /* The activations below are not functions of their outputs alone: training keeps the layer's
       values before them (as many again) for their derivatives. */
    SPINGALETT_ACT_GELU,                       /* x Phi(x), Phi the standard normal distribution (erf) */
    SPINGALETT_ACT_GELU_TANH,                  /* GELU's tanh approximation (GPT-2's):
                                       x (1 + tanh(sqrt(2 / pi) (x + 0.044715 x^3))) / 2 */
    SPINGALETT_ACT_SILU,                       /* x sigmoid(x) (swish) */
    SPINGALETT_ACT_COUNT
} SpingalettActivationFunction;

typedef enum {
    SPINGALETT_LOSS_MSE,
    SPINGALETT_LOSS_CROSS_ENTROPY,
    /* Cross-entropy of a softmax over each cell's channels of the output layer, whose activation is
       SPINGALETT_ACT_NONE (its outputs are logits), against one target a cell: the index of its class
       (the next token of a language model), a negative one for none (padding). A sample's loss is the
       mean over its cells with targets; its targets are the output's height x width values
       (spingalett_target_size()). */
    SPINGALETT_LOSS_SPARSE_CROSS_ENTROPY,
    SPINGALETT_LOSS_COUNT
} SpingalettLossFunction;

/*
 * Kinds of layers. Data flows through a network as one tensor per sample, height x width x channels
 * in channels-last order (element (y, x, c) at (y * width + x) * channels + c); a dense layer reads
 * it as a flat vector, so a dense layer after convolutions needs no flattening. A layer reads the
 * layer before it unless it names its inputs: any earlier layers, which makes the network a directed
 * acyclic graph whose last layer is the output.
 */
typedef enum {
    SPINGALETT_LAYER_DENSE,                    /* fully connected: neurons_amount outputs */
    SPINGALETT_LAYER_CONV2D,                   /* 2D convolution: `filters` output channels, kernel windows */
    SPINGALETT_LAYER_MAX_POOL2D,               /* maximum over each window, per channel */
    SPINGALETT_LAYER_AVG_POOL2D,               /* mean over each window (padded cells not counted), per channel */
    SPINGALETT_LAYER_BATCH_NORM,               /* per channel: gamma (x - mean) / sqrt(variance + epsilon) + beta,
                                       with the batch's statistics while training and running
                                       averages of them otherwise */
    SPINGALETT_LAYER_ADD,                      /* the sum of its inputs, which share one shape (residual
                                       connections); with one input, the input itself */
    SPINGALETT_LAYER_CONCAT,                   /* its inputs side by side along the channels, in the order
                                       given; they share height and width */
    SPINGALETT_LAYER_GLOBAL_AVG_POOL,          /* the mean of each channel over all cells: 1 x 1 x channels (no
                                       activation, like the other pooling layers) */
    SPINGALETT_LAYER_CONV_TRANSPOSE2D,         /* transposed 2D convolution, a convolution's data gradient run
                                       forward: `filters` output channels; each input cell adds its
                                       window of weights to the output cells (in - 1) stride - padding
                                       on, so that the output has (in - 1) stride - 2 padding + kernel +
                                       output_padding cells along each axis */
    SPINGALETT_LAYER_UPSAMPLE,                 /* each cell repeated (nearest) or interpolated (bilinear) into
                                       stride_h x stride_w cells, per channel (no parameters, no
                                       activation) */
    SPINGALETT_LAYER_LAYER_NORM,               /* per cell: gamma (x - mean) / sqrt(variance + epsilon) + beta over
                                       its channels, with the cell's own statistics (a dense layer is
                                       one cell): parameters per channel, no running statistics */
    SPINGALETT_LAYER_EMBEDDING,                /* each value of its input the index of a row of a table of
                                       `vocabulary` rows (tokens) of `outputs` values: an input of h x w
                                       x c values gives h x (w c) x outputs, a sequence of n tokens 1 x n
                                       x outputs; indices outside the table give zeros. With positions,
                                       a learned vector a cell is added (absolute position embeddings).
                                       Its input gets no gradient */
    SPINGALETT_LAYER_ATTENTION,                /* multi-head scaled dot-product attention between the cells of its
                                       input, whose channels hold each cell's queries, keys and values
                                       side by side: heads query vectors, then kv_heads key vectors and
                                       kv_heads value vectors (grouped-query attention: query head h
                                       reads key and value head h / (heads / kv_heads)), all of one head
                                       size; the output holds the heads' results side by side (heads x
                                       head size channels a cell). Scores q.k / sqrt(head size), softmax
                                       over the keys (with causal, of the cell itself and the cells
                                       before it), the values weighted by it; with rope_theta, queries
                                       and keys rotated by their cell's position first (rotary position
                                       embeddings, the halves of a head paired). No parameters */
    SPINGALETT_LAYER_RMS_NORM,                 /* per cell: gamma x / sqrt(mean of x^2 + epsilon) over its channels
                                       (root mean square normalization): a gamma per channel, no beta */
    SPINGALETT_LAYER_MULTIPLY,                 /* the element-wise product of its inputs, which share one shape
                                       (gated units: SwiGLU multiplies silu(x W) by x V) */
    SPINGALETT_LAYER_TYPE_COUNT
} SpingalettLayerType;

/* How SPINGALETT_LAYER_UPSAMPLE fills its cells: copies of the input cell, or bilinear interpolation of the
   four nearest input cells with their centres aligned (PyTorch's align_corners=False, ONNX Resize
   with half_pixel), the edges repeated. */
typedef enum {
    SPINGALETT_UPSAMPLE_NEAREST,
    SPINGALETT_UPSAMPLE_BILINEAR,
    SPINGALETT_UPSAMPLE_MODE_COUNT
} SpingalettUpsampleMode;

/* How parameters are stored: in .slett files, and as the weights a model computes with. The
   integer precisions keep one scale per weight row (output unit). */
typedef enum {
    SPINGALETT_PRECISION_FLOAT32,
    SPINGALETT_PRECISION_FP16,                 /* IEEE half */
    SPINGALETT_PRECISION_BFLOAT16,             /* upper 16 bits of a float */
    SPINGALETT_PRECISION_INT8,                 /* q * scale, q in -127..127 */
    SPINGALETT_PRECISION_INT4,                 /* q * scale, q in -7..7 */
    SPINGALETT_PRECISION_INT2,                 /* ternary: q * scale, q in {-1, 0, 1} */
    SPINGALETT_PRECISION_COUNT
} SpingalettPrecisionMode;

/*
 * A network over a .slett image, filled in by spingalett_model_init. Treat the fields as read-only;
 * the struct holds no resources of its own (models from the full library's spingalett_model_load
 * and spingalett_model_from_network own their image and are released with spingalett_model_free).
 */
typedef struct {
    uint32_t input_size;            /* inputs per sample */
    uint32_t output_size;           /* outputs per sample */
    uint32_t layer_count;           /* weight layers: the network's layers minus the input layer */
    SpingalettLossFunction loss;              /* loss the network was trained with */
    size_t workspace_size;          /* bytes of workspace spingalett_model_run needs */
    const void *image;              /* the .slett image */
    size_t image_size;              /* its size as recorded in its header */
    uint32_t max_width_;            /* private: widest hidden layer */
    uint32_t max_int_inputs_;       /* private: widest input of an integer layer */
    size_t conv_scratch_;           /* private: bytes of convolution scratch */
    void *owner_;                   /* private: memory released by spingalett_model_free */
    size_t activations_;            /* private: bytes of the layers' outputs in the workspace */
    uint64_t reserved[SPINGALETT_RESERVED];
} SpingalettModel;

typedef struct {
    SpingalettLayerType type;
    uint32_t inputs;                /* units of the layer's input: height x width x channels */
    uint32_t outputs;               /* units of its output */
    SpingalettActivationFunction activation;
    SpingalettPrecisionMode precision;        /* how this layer's weights are stored and computed with */
    uint32_t in_height, in_width, in_channels;      /* the input's shape */
    uint32_t height, width, channels;               /* the output's shape (1 x 1 x outputs for dense) */
    uint32_t kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w;  /* conv and pooling, else 0 */
    uint32_t groups;                /* conv: channel groups (1: every filter sees every input channel) */
    float epsilon;                  /* batch normalization: added to the variance */
    uint32_t input_count;           /* layers it reads (several for SPINGALETT_LAYER_ADD, SPINGALETT_LAYER_CONCAT and
                                       SPINGALETT_LAYER_MULTIPLY) */
    uint32_t input_layers[SPINGALETT_MAX_INPUTS];   /* their indices in the network: 0 is the input,
                                       i + 1 the output of weight layer i; in_height, in_width and
                                       in_channels describe the first */
    SpingalettUpsampleMode upsample;          /* upsampling: how cells are filled (stride_h x stride_w each) */
    uint32_t vocabulary;            /* embedding: rows of its table (0 for other kinds) */
    uint32_t heads, kv_heads;       /* attention: query heads, key and value heads (0 for other kinds) */
    float rope_theta;               /* attention: base of the rotary position embeddings, 0 for none */
    bool causal;                    /* attention: each cell attends to itself and the cells before it */
    bool positions;                 /* embedding: a learned vector added a cell */
    uint64_t reserved[SPINGALETT_RESERVED - 2];     /* (vocabulary took the padding before them) */
} SpingalettLayerInfo;

/*
 * Checks a .slett image (format version 3 to 8: header, layer table, shapes, bounds and CRC-32
 * checksums) and fills *model. Nothing is copied, so the image must stay valid and unchanged while the model is in
 * use, and its address must be a multiple of 4. size may exceed the image (e.g. a flash region).
 * Returns SPINGALETT_OK or an error code (SPINGALETT_ERR_FORMAT_VERSION for images of other format
 * versions, which the full library's spingalett_model_load converts).
 */
SPINGALETT_API int spingalett_model_init(SpingalettModel *model, const void *image, size_t size);

/*
 * Runs one sample: input [input_size] gives output [output_size]. workspace holds
 * model->workspace_size bytes at an address that is a multiple of 4 (a static float array, or
 * malloc'd memory) and is used as scratch; give every thread that runs a model concurrently its
 * own. With the full library, workspace may be NULL and a temporary one is allocated.
 * Returns SPINGALETT_OK or an error code.
 */
SPINGALETT_API int spingalett_model_run(const SpingalettModel *model, const float *input, float *output,
                                        void *workspace);

/* Describes weight layer `index` (0 is the first hidden layer). Returns false when index is out of
   range. */
SPINGALETT_API bool spingalett_model_layer(const SpingalettModel *model, uint32_t index, SpingalettLayerInfo *info);


#ifdef __cplusplus
}
#endif

#endif /* SPINGALETT_INFERENCE_H */

/* The engine's names of 0.x, without the prefix, when SPINGALETT_SHORT_NAMES is defined (by
   Spingalett.Short.h, which has the rest of the library's). Outside the include guard, so that the
   header gives them when it is included again after that. */
#if defined(SPINGALETT_SHORT_NAMES) && !defined(SPINGALETT_INFERENCE_SHORT_NAMES)
#define SPINGALETT_INFERENCE_SHORT_NAMES
typedef SpingalettActivationFunction ActivationFunction;
typedef SpingalettLossFunction LossFunction;
typedef SpingalettLayerType LayerType;
typedef SpingalettUpsampleMode UpsampleMode;
typedef SpingalettPrecisionMode PrecisionMode;
#define ACT_NONE SPINGALETT_ACT_NONE
#define ACT_SIGMOID SPINGALETT_ACT_SIGMOID
#define ACT_RELU SPINGALETT_ACT_RELU
#define ACT_TANH SPINGALETT_ACT_TANH
#define ACT_LEAKY_RELU SPINGALETT_ACT_LEAKY_RELU
#define ACT_FOO52 SPINGALETT_ACT_FOO52
#define ACT_SOFTMAX SPINGALETT_ACT_SOFTMAX
#define ACT_GELU SPINGALETT_ACT_GELU
#define ACT_GELU_TANH SPINGALETT_ACT_GELU_TANH
#define ACT_SILU SPINGALETT_ACT_SILU
#define ACT_COUNT SPINGALETT_ACT_COUNT
#define LOSS_MSE SPINGALETT_LOSS_MSE
#define LOSS_CROSS_ENTROPY SPINGALETT_LOSS_CROSS_ENTROPY
#define LOSS_SPARSE_CROSS_ENTROPY SPINGALETT_LOSS_SPARSE_CROSS_ENTROPY
#define LOSS_COUNT SPINGALETT_LOSS_COUNT
#define LAYER_DENSE SPINGALETT_LAYER_DENSE
#define LAYER_CONV2D SPINGALETT_LAYER_CONV2D
#define LAYER_MAX_POOL2D SPINGALETT_LAYER_MAX_POOL2D
#define LAYER_AVG_POOL2D SPINGALETT_LAYER_AVG_POOL2D
#define LAYER_BATCH_NORM SPINGALETT_LAYER_BATCH_NORM
#define LAYER_ADD SPINGALETT_LAYER_ADD
#define LAYER_CONCAT SPINGALETT_LAYER_CONCAT
#define LAYER_GLOBAL_AVG_POOL SPINGALETT_LAYER_GLOBAL_AVG_POOL
#define LAYER_CONV_TRANSPOSE2D SPINGALETT_LAYER_CONV_TRANSPOSE2D
#define LAYER_UPSAMPLE SPINGALETT_LAYER_UPSAMPLE
#define LAYER_LAYER_NORM SPINGALETT_LAYER_LAYER_NORM
#define LAYER_EMBEDDING SPINGALETT_LAYER_EMBEDDING
#define LAYER_ATTENTION SPINGALETT_LAYER_ATTENTION
#define LAYER_RMS_NORM SPINGALETT_LAYER_RMS_NORM
#define LAYER_MULTIPLY SPINGALETT_LAYER_MULTIPLY
#define LAYER_TYPE_COUNT SPINGALETT_LAYER_TYPE_COUNT
#define UPSAMPLE_NEAREST SPINGALETT_UPSAMPLE_NEAREST
#define UPSAMPLE_BILINEAR SPINGALETT_UPSAMPLE_BILINEAR
#define UPSAMPLE_MODE_COUNT SPINGALETT_UPSAMPLE_MODE_COUNT
#define PRECISION_FLOAT32 SPINGALETT_PRECISION_FLOAT32
#define PRECISION_FP16 SPINGALETT_PRECISION_FP16
#define PRECISION_BFLOAT16 SPINGALETT_PRECISION_BFLOAT16
#define PRECISION_INT8 SPINGALETT_PRECISION_INT8
#define PRECISION_INT4 SPINGALETT_PRECISION_INT4
#define PRECISION_INT2 SPINGALETT_PRECISION_INT2
#define PRECISION_COUNT SPINGALETT_PRECISION_COUNT
#endif
