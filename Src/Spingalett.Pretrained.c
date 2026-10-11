/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/*
 * Language models from the files they come in (spingalett_load_pretrained()): Hugging Face's model
 * directories, a config.json and the weights in safetensors files (one, or the shards that
 * model.safetensors.index.json lists), of LLaMA, Mistral, Qwen2 and GPT-2, as networks of the library's
 * layers. The networks are those of the Python bindings' llama_from_state_dict() and
 * gpt2_from_state_dict(): RMS normalization, the queries, keys and values of grouped-query attention in
 * one linear layer before rotary attention (rotate_half, as Hugging Face stores the weights), SwiGLU as
 * the product of a SiLU layer and a linear one; GPT-2's layer normalization, learned positions and its
 * Conv1D weights ([in][out]) transposed. A missing lm_head (tied embeddings) is a copy of the embeddings.
 *
 * The network's arrays are reserved at their final sizes before the layers are added, so that a model
 * of several gigabytes is never copied, and each tensor is decoded (F32, F16, BF16) straight into its
 * layer's parameters from the mapped file, in chunks on the OpenMP threads.
 */

#include "Spingalett.Pretrained.h"
#include "Spingalett.Json.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------------- the network */

void spingalett_decode_parallel(float *dst, const uint8_t *src, int dtype, uint64_t n) {
    const uint64_t chunk = 1u << 20;
    const int64_t chunks = (int64_t)((n + chunk - 1) / chunk);
    const size_t esize = spingalett_dtype_size(dtype);
    SPINGALETT_PARALLEL_FOR(chunks > 1,
        for (int64_t c = 0; c < chunks; c++) {
            const uint64_t first = (uint64_t)c * chunk, count = n - first < chunk ? n - first : chunk;
            spingalett_decode(dst + first, src + first * esize, dtype, (size_t)count);
        }
    );
}

/* The layers of a block that hold parameters. */
typedef struct { uint32_t attn_norm, qkv, o, ffn_norm, gate, up, down; } Block;

static bool part_into(SpgTensorSource *src, SpgPart part, uint32_t block, float *dst, uint32_t rows, uint32_t cols) {
    if (src->read(src, part, block, dst, rows, cols)) return true;
    set_error(SPINGALETT_ERR_INVALID, src->error);
    return false;
}

/* An optional bias: read, or zeros. */
static bool bias_into(SpgTensorSource *src, SpgPart part, uint32_t block, float *dst, uint32_t rows) {
    if (src->has(src, part, block)) return part_into(src, part, block, dst, 1, rows);
    memset(dst, 0, (size_t)rows * sizeof(float));
    return true;
}

NeuralNetwork *spingalett_build_transformer(const SpgTransformer *t, SpgTensorSource *src) {
    const bool gpt2 = t->arch == SPG_ARCH_GPT2;
    const uint64_t C = t->context, W = t->width, V = t->vocabulary, F = t->hidden, L = t->layers;
    const uint64_t Q = (uint64_t)t->heads * t->head_size, KV = (uint64_t)t->kv_heads * t->head_size;
    if (!C || C > 65535 || !W || !V || V > (1u << 24) || !L || !t->heads || !t->kv_heads || t->heads % t->kv_heads ||
        !t->head_size || !F || (!gpt2 && t->head_size % 2)) {
        set_error(SPINGALETT_ERR_INVALID, "pretrained: the model's hyper-parameters are invalid (or a context beyond "
                  "65535 tokens, a vocabulary beyond 2^24)");
        return NULL;
    }
    /* the final sizes, so that the arrays are never moved */
    uint64_t neurons = C + C * W + L * C * (gpt2 ? 10 * W + F : 6 * W + Q + 2 * KV + Q + 3 * F) + C * W + C * V;
    uint64_t weights = V * W + L * (2 * W + (Q + 2 * KV) * W + W * Q + (gpt2 ? 2 : 3) * F * W) + W + V * W;
    uint64_t biases = (gpt2 ? C * W : 0) + L * ((Q + 2 * KV) + W + (gpt2 ? 3 * W + F : 2 * F + W)) + (gpt2 ? W : 0) + V;
    NeuralNetwork *net = new_spingalett(.loss_func = LOSS_SPARSE_CROSS_ENTROPY);
    if (!net) return NULL;
    (void)spingalett_network_reserve(net, neurons, weights, biases);
    Block *blocks = (Block *)calloc(L, sizeof(Block));
    bool ok = blocks != NULL;
    const WeightInitialization none = WEIGHT_INITIALIZATION_NONE;
    uint32_t embed = 0, norm = 0, head = 0;
    ok = ok && layer(.net = net, .neurons_amount = (uint32_t)C) != SPINGALETT_NO_LAYER;
    if (ok) embed = embedding(.net = net, .vocabulary = (uint32_t)V, .neurons_amount = (uint32_t)W, .positions = gpt2,
                              .weight_initialization = none);
    uint32_t x = embed;
    for (uint32_t b = 0; ok && b < L; b++) {
        Block *k = &blocks[b];
        k->attn_norm = gpt2 ? layer_norm(.net = net, .epsilon = t->eps) : rms_norm(.net = net, .epsilon = t->eps);
        k->qkv = linear(.net = net, .neurons_amount = (uint32_t)(Q + 2 * KV), .weight_initialization = none);
        attention(.net = net, .heads = t->heads, .kv_heads = t->kv_heads, .causal = true,
                  .rope_theta = gpt2 ? 0.0f : t->theta);
        k->o = linear(.net = net, .neurons_amount = (uint32_t)W, .weight_initialization = none);
        x = add_layers(.net = net, .inputs = {x, k->o});
        k->ffn_norm = gpt2 ? layer_norm(.net = net, .epsilon = t->eps) : rms_norm(.net = net, .epsilon = t->eps);
        if (gpt2) {
            k->up = linear(.net = net, .neurons_amount = (uint32_t)F, .act_func = t->act, .weight_initialization = none);
        } else {
            k->gate = linear(.net = net, .neurons_amount = (uint32_t)F, .act_func = ACT_SILU, .weight_initialization = none);
            k->up = linear(.net = net, .inputs = {k->ffn_norm}, .neurons_amount = (uint32_t)F, .weight_initialization = none);
            multiply_layers(.net = net, .inputs = {k->gate, k->up});
        }
        k->down = linear(.net = net, .neurons_amount = (uint32_t)W, .weight_initialization = none);
        x = add_layers(.net = net, .inputs = {x, k->down});
        ok = x != SPINGALETT_NO_LAYER && k->attn_norm != SPINGALETT_NO_LAYER && k->qkv != SPINGALETT_NO_LAYER &&
             k->o != SPINGALETT_NO_LAYER && k->ffn_norm != SPINGALETT_NO_LAYER && k->up != SPINGALETT_NO_LAYER &&
             k->down != SPINGALETT_NO_LAYER;
    }
    if (ok) {
        norm = gpt2 ? layer_norm(.net = net, .epsilon = t->eps) : rms_norm(.net = net, .epsilon = t->eps);
        head = linear(.net = net, .neurons_amount = (uint32_t)V, .weight_initialization = none);
        ok = embed != SPINGALETT_NO_LAYER && norm != SPINGALETT_NO_LAYER && head != SPINGALETT_NO_LAYER;
    }
    /* the parameters, part by part */
#define WT(l) (net->weights + net->weight_offsets[(l) - 1])
#define BS(l) (net->biases + net->bias_offsets[(l) - 1])
    ok = ok && part_into(src, SPG_PART_EMBED, 0, WT(embed), (uint32_t)V, (uint32_t)W);
    if (ok && gpt2) ok = part_into(src, SPG_PART_POSITIONS, 0, BS(embed), (uint32_t)C, (uint32_t)W);
    for (uint32_t b = 0; ok && b < L; b++) {
        const Block *k = &blocks[b];
        ok = part_into(src, SPG_PART_ATTN_NORM, b, WT(k->attn_norm), 1, (uint32_t)W) &&
             (!gpt2 || bias_into(src, SPG_PART_ATTN_NORM_BIAS, b, BS(k->attn_norm), (uint32_t)W)) &&
             part_into(src, SPG_PART_FFN_NORM, b, WT(k->ffn_norm), 1, (uint32_t)W) &&
             (!gpt2 || bias_into(src, SPG_PART_FFN_NORM_BIAS, b, BS(k->ffn_norm), (uint32_t)W));
        if (ok && gpt2) {
            ok = part_into(src, SPG_PART_QKV, b, WT(k->qkv), (uint32_t)(3 * W), (uint32_t)W) &&
                 bias_into(src, SPG_PART_QKV_BIAS, b, BS(k->qkv), (uint32_t)(3 * W));
        } else if (ok) {
            float *w = WT(k->qkv), *bq = BS(k->qkv);
            ok = part_into(src, SPG_PART_Q, b, w, (uint32_t)Q, (uint32_t)W) &&
                 part_into(src, SPG_PART_K, b, w + Q * W, (uint32_t)KV, (uint32_t)W) &&
                 part_into(src, SPG_PART_V, b, w + (Q + KV) * W, (uint32_t)KV, (uint32_t)W) &&
                 bias_into(src, SPG_PART_Q_BIAS, b, bq, (uint32_t)Q) && bias_into(src, SPG_PART_K_BIAS, b, bq + Q, (uint32_t)KV) &&
                 bias_into(src, SPG_PART_V_BIAS, b, bq + Q + KV, (uint32_t)KV) &&
                 part_into(src, SPG_PART_GATE, b, WT(k->gate), (uint32_t)F, (uint32_t)W) &&
                 bias_into(src, SPG_PART_GATE_BIAS, b, BS(k->gate), (uint32_t)F);
        }
        ok = ok && part_into(src, SPG_PART_O, b, WT(k->o), (uint32_t)W, (uint32_t)Q) &&
             bias_into(src, SPG_PART_O_BIAS, b, BS(k->o), (uint32_t)W) &&
             part_into(src, SPG_PART_UP, b, WT(k->up), (uint32_t)F, (uint32_t)W) &&
             bias_into(src, SPG_PART_UP_BIAS, b, BS(k->up), (uint32_t)F) &&
             part_into(src, SPG_PART_DOWN, b, WT(k->down), (uint32_t)W, (uint32_t)F) &&
             bias_into(src, SPG_PART_DOWN_BIAS, b, BS(k->down), (uint32_t)W);
    }
    ok = ok && part_into(src, SPG_PART_NORM, 0, WT(norm), 1, (uint32_t)W) &&
         (!gpt2 || bias_into(src, SPG_PART_NORM_BIAS, 0, BS(norm), (uint32_t)W));
    if (ok && src->has(src, SPG_PART_HEAD, 0)) ok = part_into(src, SPG_PART_HEAD, 0, WT(head), (uint32_t)V, (uint32_t)W);
    else if (ok) memcpy(WT(head), WT(embed), (size_t)(V * W) * sizeof(float));     /* tied to the embeddings */
    if (ok) memset(BS(head), 0, (size_t)V * sizeof(float));
#undef WT
#undef BS
    free(blocks);
    if (!ok) {
        free_network(net);
        return NULL;
    }
    spingalett_network_written(net);
    return net;
}

/* ------------------------------------------------------------------------- Hugging Face's directories */

/* The tensors of a checkpoint's safetensors files, by Hugging Face's names. */
typedef struct {
    SpgTensorSource base;
    SpgArch arch;
    char prefix[64];                /* of the model's names ("model.", "transformer.", "") */
    bool conv1d;                    /* GPT-2's weights stored [in][out] */
    SpgTensor *tensors;
    size_t count;
} HfSource;

static void part_name(const HfSource *h, SpgPart part, uint32_t block, char *name, size_t cap) {
    static const char *const llama[SPG_PART_COUNT] = {
        [SPG_PART_EMBED] = "embed_tokens.weight", [SPG_PART_NORM] = "norm.weight", [SPG_PART_HEAD] = "lm_head.weight",
        [SPG_PART_ATTN_NORM] = "input_layernorm.weight", [SPG_PART_Q] = "self_attn.q_proj.weight",
        [SPG_PART_K] = "self_attn.k_proj.weight", [SPG_PART_V] = "self_attn.v_proj.weight",
        [SPG_PART_Q_BIAS] = "self_attn.q_proj.bias", [SPG_PART_K_BIAS] = "self_attn.k_proj.bias",
        [SPG_PART_V_BIAS] = "self_attn.v_proj.bias", [SPG_PART_O] = "self_attn.o_proj.weight",
        [SPG_PART_O_BIAS] = "self_attn.o_proj.bias", [SPG_PART_FFN_NORM] = "post_attention_layernorm.weight",
        [SPG_PART_GATE] = "mlp.gate_proj.weight", [SPG_PART_GATE_BIAS] = "mlp.gate_proj.bias",
        [SPG_PART_UP] = "mlp.up_proj.weight", [SPG_PART_UP_BIAS] = "mlp.up_proj.bias",
        [SPG_PART_DOWN] = "mlp.down_proj.weight", [SPG_PART_DOWN_BIAS] = "mlp.down_proj.bias",
    };
    static const char *const gpt2[SPG_PART_COUNT] = {
        [SPG_PART_EMBED] = "wte.weight", [SPG_PART_POSITIONS] = "wpe.weight", [SPG_PART_NORM] = "ln_f.weight",
        [SPG_PART_NORM_BIAS] = "ln_f.bias", [SPG_PART_HEAD] = "lm_head.weight", [SPG_PART_ATTN_NORM] = "ln_1.weight",
        [SPG_PART_ATTN_NORM_BIAS] = "ln_1.bias", [SPG_PART_QKV] = "attn.c_attn.weight", [SPG_PART_QKV_BIAS] = "attn.c_attn.bias",
        [SPG_PART_O] = "attn.c_proj.weight", [SPG_PART_O_BIAS] = "attn.c_proj.bias", [SPG_PART_FFN_NORM] = "ln_2.weight",
        [SPG_PART_FFN_NORM_BIAS] = "ln_2.bias", [SPG_PART_UP] = "mlp.c_fc.weight", [SPG_PART_UP_BIAS] = "mlp.c_fc.bias",
        [SPG_PART_DOWN] = "mlp.c_proj.weight", [SPG_PART_DOWN_BIAS] = "mlp.c_proj.bias",
    };
    const char *suffix = (h->arch == SPG_ARCH_GPT2 ? gpt2 : llama)[part];
    if (!suffix) { name[0] = '\0'; return; }
    if (part == SPG_PART_HEAD) snprintf(name, cap, "%s", suffix);
    else if (part <= SPG_PART_HEAD) snprintf(name, cap, "%s%s", h->prefix, suffix);
    else snprintf(name, cap, "%s%s.%u.%s", h->prefix, h->arch == SPG_ARCH_GPT2 ? "h" : "layers", block, suffix);
}

static const SpgTensor *find_tensor(const HfSource *h, const char *name) {
    const size_t n = strlen(name);
    if (!n) return NULL;
    for (size_t k = 0; k < h->count; k++)
        if (h->tensors[k].name_len == n && !memcmp(h->tensors[k].name, name, n)) return &h->tensors[k];
    return NULL;
}

static bool hf_has(SpgTensorSource *s, SpgPart part, uint32_t block) {
    const HfSource *h = (const HfSource *)s;
    char name[160];
    part_name(h, part, block, name, sizeof name);
    return find_tensor(h, name) != NULL;
}

static bool hf_read(SpgTensorSource *s, SpgPart part, uint32_t block, float *dst, uint32_t rows, uint32_t cols) {
    HfSource *h = (HfSource *)s;
    char name[160];
    part_name(h, part, block, name, sizeof name);
    const SpgTensor *t = find_tensor(h, name);
    if (!t) {
        snprintf(s->error, sizeof s->error, "pretrained: the checkpoint has no %s", name[0] ? name : "such tensor");
        return false;
    }
    const bool vector = rows == 1 && part != SPG_PART_EMBED && part != SPG_PART_HEAD;
    /* GPT-2's Conv1D matrices are stored [in][out]; its learned positions may be more than the context */
    const bool transposed = h->conv1d && !vector && (part == SPG_PART_QKV || part == SPG_PART_O || part == SPG_PART_UP ||
                                                    part == SPG_PART_DOWN);
    const int64_t want0 = transposed ? cols : rows, want1 = transposed ? rows : cols;
    bool fits = vector ? t->ndims == 1 && t->dims[0] == cols
              : t->ndims == 2 && t->dims[1] == want1 &&
                (t->dims[0] == want0 || (part == SPG_PART_POSITIONS && t->dims[0] >= want0));
    if (!fits) {
        char dims[64] = "", want[32];
        for (int d = 0, used = 0; d < t->ndims && d < 8 && used < 56; d++)
            used += snprintf(dims + used, sizeof dims - (size_t)used, "%s%lld", d ? ", " : "", (long long)t->dims[d]);
        if (vector) snprintf(want, sizeof want, "%u", cols);
        else snprintf(want, sizeof want, "%u, %u", (unsigned)want0, (unsigned)want1);
        snprintf(s->error, sizeof s->error, "pretrained: %.100s has shape [%.40s], not [%s]", name, dims, want);
        return false;
    }
    if (!spingalett_dtype_size(t->dtype)) {
        snprintf(s->error, sizeof s->error, "pretrained: %.100s is of a type that is not F32, F16, BF16, F64 or an "
                 "integer", name);
        return false;
    }
    const uint64_t count = (uint64_t)rows * cols;
    if (!transposed) {
        spingalett_decode_parallel(dst, t->data, t->dtype, count);
        return true;
    }
    float *scratch = (float *)malloc(spingalett_dense_scratch(rows, cols, true) * sizeof(float));
    if (!scratch) { snprintf(s->error, sizeof s->error, "pretrained: out of memory"); return false; }
    spingalett_import_dense(dst, t->data, t->dtype, rows, cols, cols, 1, true, 1.0f, scratch);
    free(scratch);
    return true;
}

/* A config's integer (0 when it is missing or not one) and number. */
static uint32_t config_int(const SpgJson *j, const char *key) {
    int64_t v;
    return spingalett_json_integer(j, spingalett_json_member(j, SPG_JSON_ROOT, key), 0, UINT32_MAX, &v) ? (uint32_t)v : 0;
}

static float config_float(const SpgJson *j, const char *key, float fallback) {
    double v;
    return spingalett_json_number(j, spingalett_json_member(j, SPG_JSON_ROOT, key), &v) ? (float)v : fallback;
}

static char *join(const char *dir, const char *name) {
    size_t n = strlen(dir) + strlen(name) + 2;
    char *p = (char *)malloc(n);
    if (p) snprintf(p, n, "%s/%s", dir, name);
    return p;
}

static bool exists(const char *path) {
    FILE *f = fopen(path, "rb");
    if (f) fclose(f);
    return f != NULL;
}

/* The hyper-parameters of config.json; false with the error set for a model of another kind. */
static bool read_config(const SpgJson *j, SpgTransformer *t, uint32_t context, uint32_t *window) {
    char type[32] = "", act[32] = "", msg[200];
    uint32_t v = spingalett_json_member(j, SPG_JSON_ROOT, "model_type");
    if (v && j->values[v].type == SPG_JSON_STRING && j->values[v].count < sizeof type) spingalett_json_copy(j, v, type);
    memset(t, 0, sizeof *t);
    *window = 0;
    if (!strcmp(type, "gpt2")) {
        t->arch = SPG_ARCH_GPT2;
        t->width = config_int(j, "n_embd");
        t->layers = config_int(j, "n_layer");
        t->heads = t->kv_heads = config_int(j, "n_head");
        t->head_size = t->heads ? t->width / t->heads : 0;
        t->hidden = config_int(j, "n_inner") ? config_int(j, "n_inner") : 4 * t->width;
        t->eps = config_float(j, "layer_norm_epsilon", 1e-5f);
        uint32_t positions = config_int(j, "n_positions") ? config_int(j, "n_positions") : config_int(j, "n_ctx");
        t->context = context ? context : positions;
        if (context > positions) {
            snprintf(msg, sizeof msg, "pretrained: a context of %u tokens, beyond GPT-2's %u positions", context, positions);
            set_error(SPINGALETT_ERR_INVALID, msg);
            return false;
        }
        v = spingalett_json_member(j, SPG_JSON_ROOT, "activation_function");
        if (v && j->values[v].count < sizeof act) spingalett_json_copy(j, v, act);
        t->act = !act[0] || !strcmp(act, "gelu_new") || !strcmp(act, "gelu_pytorch_tanh") || !strcmp(act, "gelu_fast") ? ACT_GELU_TANH
               : !strcmp(act, "gelu") ? ACT_GELU : !strcmp(act, "relu") ? ACT_RELU : !strcmp(act, "silu") ? ACT_SILU : ACT_COUNT;
        bool scaled = true, by_layer = false;
        (void)spingalett_json_bool(j, spingalett_json_member(j, SPG_JSON_ROOT, "scale_attn_weights"), &scaled);
        (void)spingalett_json_bool(j, spingalett_json_member(j, SPG_JSON_ROOT, "scale_attn_by_inverse_layer_idx"), &by_layer);
        if (t->act == ACT_COUNT || !scaled || by_layer) {
            snprintf(msg, sizeof msg, "pretrained: GPT-2's %s is not supported", t->act == ACT_COUNT ? "activation function"
                     : "attention without its 1 / sqrt(head size) scale");
            set_error(SPINGALETT_ERR_INVALID, msg);
            return false;
        }
        return true;
    }
    if (strcmp(type, "llama") && strcmp(type, "mistral") && strcmp(type, "qwen2")) {
        snprintf(msg, sizeof msg, "pretrained: the model type '%s' is not supported (llama, mistral, qwen2 and gpt2 are)",
                 type[0] ? type : "(none)");
        set_error(SPINGALETT_ERR_INVALID, msg);
        return false;
    }
    t->arch = SPG_ARCH_LLAMA;
    t->width = config_int(j, "hidden_size");
    t->hidden = config_int(j, "intermediate_size");
    t->layers = config_int(j, "num_hidden_layers");
    t->heads = config_int(j, "num_attention_heads");
    t->kv_heads = config_int(j, "num_key_value_heads") ? config_int(j, "num_key_value_heads") : t->heads;
    t->head_size = config_int(j, "head_dim") ? config_int(j, "head_dim") : t->heads ? t->width / t->heads : 0;
    t->eps = config_float(j, "rms_norm_eps", 1e-6f);
    t->theta = config_float(j, "rope_theta", 10000.0f);
    uint32_t positions = config_int(j, "max_position_embeddings");
    t->context = context ? context : positions && positions < 2048 ? positions : 2048;
    v = spingalett_json_member(j, SPG_JSON_ROOT, "hidden_act");
    if (v && j->values[v].count < sizeof act) spingalett_json_copy(j, v, act);
    if (act[0] && strcmp(act, "silu")) {
        snprintf(msg, sizeof msg, "pretrained: the activation '%s' is not supported (silu is)", act);
        set_error(SPINGALETT_ERR_INVALID, msg);
        return false;
    }
    /* rotary embeddings of another kind (scaled frequencies) or on part of a head */
    uint32_t scaling = spingalett_json_member(j, SPG_JSON_ROOT, "rope_scaling");
    if (scaling && j->values[scaling].type == SPG_JSON_OBJECT) {
        uint32_t kind = spingalett_json_member(j, scaling, "rope_type");
        if (!kind) kind = spingalett_json_member(j, scaling, "type");
        char name[32] = "?";
        if (kind && j->values[kind].count < sizeof name) spingalett_json_copy(j, kind, name);
        if (strcmp(name, "default")) {
            snprintf(msg, sizeof msg, "pretrained: rotary embeddings with '%s' scaling are not supported", name);
            set_error(SPINGALETT_ERR_INVALID, msg);
            return false;
        }
    }
    if (config_float(j, "partial_rotary_factor", 1.0f) != 1.0f) {
        set_error(SPINGALETT_ERR_INVALID, "pretrained: rotary embeddings on part of a head are not supported");
        return false;
    }
    /* Mistral's (and Qwen2's, when it uses one) sliding window */
    bool sliding = !strcmp(type, "mistral");
    (void)spingalett_json_bool(j, spingalett_json_member(j, SPG_JSON_ROOT, "use_sliding_window"), &sliding);
    if (sliding) *window = config_int(j, "sliding_window");
    return true;
}

/* The weights of a directory: model.safetensors, or the shards of model.safetensors.index.json. */
typedef struct {
    SpgFileView *files;
    uint32_t count;
} Shards;

static void close_shards(Shards *s) {
    for (uint32_t k = 0; k < s->count; k++) spingalett_file_close(&s->files[k]);
    free(s->files);
    s->files = NULL;
    s->count = 0;
}

static bool open_shard(Shards *s, const char *dir, const char *name, HfSource *h) {
    char *path = join(dir, name);
    SpgFileView *files = path ? (SpgFileView *)realloc(s->files, ((size_t)s->count + 1) * sizeof(SpgFileView)) : NULL;
    if (!files) { free(path); set_error(SPINGALETT_ERR_ALLOC, "pretrained: out of memory"); return false; }
    s->files = files;
    bool ok = spingalett_file_open(&s->files[s->count], path);
    free(path);
    if (!ok) return false;
    s->count++;
    SpgTensor *tensors = NULL;
    size_t count = 0;
    char error[256];
    const SpgFileView *f = &s->files[s->count - 1];
    if (!spingalett_safetensors(f->data, f->size, &tensors, &count, error, sizeof error)) {
        char msg[320];
        snprintf(msg, sizeof msg, "pretrained: %s: %s", name, error);
        set_error(SPINGALETT_ERR_INVALID, msg);
        return false;
    }
    SpgTensor *all = count ? (SpgTensor *)realloc(h->tensors, (h->count + count) * sizeof(SpgTensor)) : h->tensors;
    if (count && !all) { free(tensors); set_error(SPINGALETT_ERR_ALLOC, "pretrained: out of memory"); return false; }
    h->tensors = all;
    if (count) memcpy(h->tensors + h->count, tensors, count * sizeof(SpgTensor));
    h->count += count;
    free(tensors);
    return true;
}

static bool open_weights(const char *dir, Shards *shards, HfSource *h) {
    char *single = join(dir, "model.safetensors"), *index = join(dir, "model.safetensors.index.json");
    bool ok = single && index;
    if (!ok) set_error(SPINGALETT_ERR_ALLOC, "pretrained: out of memory");
    if (ok && exists(single)) {
        ok = open_shard(shards, dir, "model.safetensors", h);
    } else if (ok && exists(index)) {
        SpgFileView file;
        SpgJson j;
        ok = spingalett_file_open(&file, index);
        if (ok && !spingalett_json_parse(&j, (const char *)file.data, file.size)) {
            char msg[220];
            snprintf(msg, sizeof msg, "pretrained: model.safetensors.index.json: %s", j.error);
            set_error(SPINGALETT_ERR_INVALID, msg);
            spingalett_file_close(&file);
            ok = false;
        } else if (ok) {
            /* each shard the weight map names, once */
            uint32_t map = spingalett_json_member(&j, SPG_JSON_ROOT, "weight_map");
            ok = map && j.values[map].type == SPG_JSON_OBJECT;
            if (!ok) set_error(SPINGALETT_ERR_INVALID, "pretrained: model.safetensors.index.json has no weight_map");
            for (uint32_t k = spingalett_json_first(&j, map); ok && k && k < j.values[map].next; k = j.values[k + 1].next) {
                bool seen = false;
                for (uint32_t e = spingalett_json_first(&j, map); e < k && !seen; e = j.values[e + 1].next)
                    seen = j.values[e + 1].count == j.values[k + 1].count && j.values[e + 1].type == SPG_JSON_STRING &&
                           j.values[k + 1].type == SPG_JSON_STRING && j.values[e + 1].length == j.values[k + 1].length &&
                           !memcmp(j.text + j.values[e + 1].start, j.text + j.values[k + 1].start, j.values[k + 1].length);
                if (seen) continue;
                char *name = spingalett_json_strdup(&j, k + 1, NULL);
                if (!name || strchr(name, '/') || strchr(name, '\\') || !strcmp(name, "..")) {
                    set_error(SPINGALETT_ERR_INVALID, "pretrained: the weight map names a shard outside the directory");
                    ok = false;
                } else {
                    ok = open_shard(shards, dir, name, h);
                }
                free(name);
            }
            spingalett_json_free(&j);
            spingalett_file_close(&file);
        }
    } else if (ok) {
        set_error(SPINGALETT_ERR_FILE_IO, "pretrained: the directory has no model.safetensors or "
                  "model.safetensors.index.json");
        ok = false;
    }
    free(single);
    free(index);
    return ok;
}

static NeuralNetwork *load_directory(const char *dir, uint32_t context) {
    char *config_path = join(dir, "config.json");
    SpgFileView file;
    if (!config_path || !spingalett_file_open(&file, config_path)) {
        free(config_path);
        return NULL;
    }
    free(config_path);
    SpgJson j;
    if (!spingalett_json_parse(&j, (const char *)file.data, file.size)) {
        char msg[200];
        snprintf(msg, sizeof msg, "pretrained: config.json: %s", j.error);
        set_error(SPINGALETT_ERR_INVALID, msg);
        spingalett_file_close(&file);
        return NULL;
    }
    SpgTransformer t;
    uint32_t window = 0;
    bool ok = read_config(&j, &t, context, &window);
    spingalett_json_free(&j);
    spingalett_file_close(&file);
    if (!ok) return NULL;
    HfSource h = {.base = {.has = hf_has, .read = hf_read}, .arch = t.arch};
    Shards shards = {0};
    NeuralNetwork *net = NULL;
    if (open_weights(dir, &shards, &h)) {
        /* the prefix of the model's names, the vocabulary from the embeddings, GPT-2's layout */
        const char *embed = t.arch == SPG_ARCH_GPT2 ? "wte.weight" : "embed_tokens.weight";
        const size_t en = strlen(embed);
        const SpgTensor *e = NULL;
        for (size_t k = 0; k < h.count && !e; k++)
            if (h.tensors[k].name_len >= en && !memcmp(h.tensors[k].name + h.tensors[k].name_len - en, embed, en) &&
                h.tensors[k].name_len - en < sizeof h.prefix && h.tensors[k].ndims == 2)
                e = &h.tensors[k];
        if (!e) {
            char msg[100];
            snprintf(msg, sizeof msg, "pretrained: the checkpoint has no %s", embed);
            set_error(SPINGALETT_ERR_INVALID, msg);
        } else {
            memcpy(h.prefix, e->name, e->name_len - en);
            h.prefix[e->name_len - en] = '\0';
            t.vocabulary = e->dims[0] > 0 && e->dims[0] < (1 << 24) ? (uint32_t)e->dims[0] : 0;
            if (t.arch == SPG_ARCH_GPT2) {
                char name[160];
                part_name(&h, SPG_PART_QKV, 0, name, sizeof name);
                const SpgTensor *qkv = find_tensor(&h, name);
                h.conv1d = qkv && qkv->ndims == 2 && qkv->dims[0] == t.width && qkv->dims[1] == 3 * (int64_t)t.width;
            }
            net = spingalett_build_transformer(&t, &h.base);
            if (net && window && t.context > window)
                spingalett_log(LOG_WARNING, "pretrained: the model attends to the last %u tokens (a sliding window); "
                               "the network attends to all %u of its context", window, t.context);
        }
    }
    free(h.tensors);
    close_shards(&shards);
    return net;
}

SpingalettNetwork *spingalett_load_pretrained_args(SpingalettPretrainedArgs args) {
    if (!args.path) {
        set_error(SPINGALETT_ERR_INVALID, "pretrained: path is NULL");
        return NULL;
    }
    char *config = join(args.path, "config.json");
    const bool directory = config && exists(config);
    free(config);
    if (directory) return load_directory(args.path, args.context);
    set_error(SPINGALETT_ERR_FILE_IO, "pretrained: the path is no model directory (config.json)");
    return NULL;
}
