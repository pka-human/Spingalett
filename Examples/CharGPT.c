/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/*
 * A character-level GPT, as nanoGPT's shakespeare_char, trained on any text file and sampled. The
 * text's bytes are the tokens (a vocabulary of 256), written once to a token file next to the text
 * (<text>.tokens, uint16 as nanoGPT's prepare.py writes them), which spingalett_dataset_open_tokens()
 * reads in windows of 128 bytes, a window every 32 bytes, in a new order every epoch. The model:
 *
 *   embedding of 256 bytes and 128 learned positions, width 256
 *   4 x [layer norm, linear 768 (queries, keys, values of 4 heads of 64), causal attention, linear 256,
 *        add; layer norm, linear 1024 with GELU, linear 256, add]
 *   layer norm, linear 256: each position's logits of the next byte (sparse cross-entropy)
 *
 * about 3.3 million parameters, trained with AdamW on mini-batches of 32 windows. After every epoch
 * the mean loss and 200 bytes generated from the prompt (temperature 0.8, the 40 most likely bytes);
 * at the end 600 bytes, and the network saved to chargpt.slett.
 *
 *   Bin/CharGPT <text file> [epochs] [cpu|gpu|cuda|bf16] ["prompt"]
 *
 * "gpu" trains on the GPU through Vulkan, "cuda" through CUDA, "bf16" with the products in bfloat16
 * (CUDA when it finds a device, else Vulkan); the default is the CPU. Any text of a few hundred
 * kilobytes or more learns words within a few epochs: Shakespeare's plays (nanoGPT's input.txt, 1.1 MB)
 * write lines in their form after ten.
 */

#include <Spingalett/Spingalett.Short.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CONTEXT 128u
#define WIDTH   256u
#define HEADS   4u
#define LAYERS  4u
#define VOCAB   256u
#define STRIDE  32u
#define BATCH   32u

static NeuralNetwork *create_gpt(void) {
    NeuralNetwork *net = new_spingalett(.loss_func = LOSS_SPARSE_CROSS_ENTROPY);
    layer(.net = net, .neurons_amount = CONTEXT);
    uint32_t x = embedding(.net = net, .vocabulary = VOCAB, .neurons_amount = WIDTH, .positions = true);
    for (uint32_t b = 0; b < LAYERS; b++) {
        layer_norm(.net = net);
        linear(.net = net, .neurons_amount = 3u * WIDTH);
        attention(.net = net, .heads = HEADS, .causal = true);
        uint32_t a = linear(.net = net, .neurons_amount = WIDTH);
        x = add_layers(.net = net, .inputs = {x, a});
        layer_norm(.net = net);
        linear(.net = net, .neurons_amount = 4u * WIDTH, .act_func = ACT_GELU);
        a = linear(.net = net, .neurons_amount = WIDTH);
        x = add_layers(.net = net, .inputs = {x, a});
    }
    layer_norm(.net = net);
    linear(.net = net, .neurons_amount = VOCAB);
    return net;
}

/* The text's bytes as uint16 tokens in path; its length, 0 on failure. */
static size_t write_tokens(const char *text, const char *path, uint32_t **prompt_from) {
    FILE *in = fopen(text, "rb");
    if (!in) return 0;
    FILE *out = fopen(path, "wb");
    if (!out) {
        fclose(in);
        return 0;
    }
    size_t count = 0;
    int c;
    static uint32_t first[CONTEXT];
    while ((c = fgetc(in)) != EOF) {
        const unsigned char pair[2] = {(unsigned char)c, 0};
        fwrite(pair, 1, 2, out);
        if (count < CONTEXT) first[count] = (uint32_t)c;
        count++;
    }
    fclose(in);
    fclose(out);
    *prompt_from = first;
    return count;
}

static void sample(NeuralNetwork *net, const uint32_t *prompt, uint32_t length, uint32_t count) {
    uint32_t *tokens = malloc(count * sizeof(uint32_t));
    const uint32_t made = spingalett_generate(.net = net, .prompt = prompt, .prompt_length = length, .tokens = tokens,
                                              .count = count, .temperature = 0.8f, .top_k = 40);
    for (uint32_t i = 0; i < length; i++) putchar((int)prompt[i]);
    for (uint32_t i = 0; i < made; i++) putchar((int)tokens[i]);
    printf("\n");
    free(tokens);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <text file> [epochs] [cpu|gpu|cuda|bf16] [\"prompt\"]\n", argv[0]);
        return 1;
    }
    size_t epochs = 10;
    ComputeMode mode = COMPUTE_OPENMP;
    bool bf16 = false;
    const char *prompt_text = NULL;
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "cpu")) mode = COMPUTE_OPENMP;
        else if (!strcmp(argv[i], "gpu")) mode = COMPUTE_VULKAN;
        else if (!strcmp(argv[i], "cuda")) mode = COMPUTE_CUDA;
        else if (!strcmp(argv[i], "bf16")) bf16 = true, mode = spingalett_cuda_device() ? COMPUTE_CUDA : COMPUTE_VULKAN;
        else if (atoi(argv[i]) > 0) epochs = (size_t)atoi(argv[i]);
        else prompt_text = argv[i];
    }
    spingalett_set_verbose(false);
    spingalett_seed(1);
    spingalett_set_compute_mode(mode);
    if (bf16 && !spingalett_set_gpu_precision(PRECISION_BFLOAT16))
        fprintf(stderr, "bfloat16 products are not available here: single precision\n");

    /* the text as a token file, read in windows */
    char path[4096];
    snprintf(path, sizeof path, "%s.tokens", argv[1]);
    uint32_t *first = NULL;
    const size_t length = write_tokens(argv[1], path, &first);
    if (length < CONTEXT + 1u) {
        fprintf(stderr, "%s: no text, or shorter than %u bytes\n", argv[1], CONTEXT + 1u);
        return 1;
    }
    SpingalettTokenReaderOptions options = {.context = CONTEXT, .stride = STRIDE, .shuffle = true};
    SpingalettDatasetReader *reader = spingalett_dataset_open_tokens(path, &options);
    if (!reader) {
        fprintf(stderr, "%s\n", spingalett_last_error_message());
        return 1;
    }
    const uint32_t windows = spingalett_dataset_info(reader).count;

    /* the prompt: the one given, else the text's first line (at most the window) */
    uint32_t prompt[CONTEXT], prompt_length = 0;
    if (prompt_text) {
        for (; prompt_text[prompt_length] && prompt_length < CONTEXT; prompt_length++)
            prompt[prompt_length] = (unsigned char)prompt_text[prompt_length];
    } else {
        for (; prompt_length < CONTEXT && prompt_length < length && first[prompt_length] != '\n'; prompt_length++)
            prompt[prompt_length] = first[prompt_length];
    }
    if (prompt_length == 0) prompt[prompt_length++] = '\n';

    NeuralNetwork *net = create_gpt();
    printf("CharGPT: %zu bytes of %s, %u windows of %u bytes an epoch, %llu parameters, %s%s\n", length, argv[1],
           windows, CONTEXT, (unsigned long long)spingalett_parameter_count(net),
           mode == COMPUTE_CUDA ? "CUDA" : mode == COMPUTE_VULKAN ? "Vulkan" : "CPU", bf16 ? " (bfloat16)" : "");
    for (size_t e = 0; e < epochs; e++) {
        TrainReport r = train(.net = net, .training_mode = MODE_GENERATOR_FUNCTION,
                              .generator = spingalett_dataset_generator, .generator_data = reader,
                              .sample_count = windows, .epochs = 1, .batch_size = BATCH,
                              .training_strategy = STRATEGY_SMALL_BATCH, .optimizer_type = OPTIMIZER_ADAMW,
                              .learning_rate = 6e-4f, .weight_decay = 0.1f, .beta2 = 0.99f, .max_grad_norm = 1.0f);
        if (r.status != TRAIN_COMPLETED) {
            fprintf(stderr, "training failed: %s\n", spingalett_last_error_message());
            return 1;
        }
        printf("\nepoch %zu: loss %.3f\n", e + 1, r.train_loss);
        sample(net, prompt, prompt_length, 200);
    }
    printf("\n--- 600 bytes ---\n");
    sample(net, prompt, prompt_length, 600);
    save_spingalett(.net = net, .filename = "chargpt.slett");
    spingalett_dataset_close(reader);
    free_network(net);
    remove(path);
    return 0;
}
