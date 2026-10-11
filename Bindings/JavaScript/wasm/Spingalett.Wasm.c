/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/*
 * The WebAssembly module of the JavaScript package: the inference engine (Src/Spingalett.Inference.c,
 * linked with this file) and the functions it exports, a thin layer over Spingalett.Inference.h. The
 * module imports nothing and owns its memory: images, models with their workspaces and the buffers of
 * their inputs and outputs live in blocks of a first-fit allocator over the memory above the stack and
 * the data, which grows as they are allocated (WebAssembly memory never shrinks; freed blocks are
 * reused). JavaScript copies a model's image into a block of spg_alloc() and opens it with
 * spg_model_new(); what it reads back comes as 32-bit words in a static array, in the order
 * spg_model_info() and spg_model_layer() document.
 */

#include <Spingalett/Spingalett.Inference.h>

#define EXPORT(name) __attribute__((export_name(#name)))

/* ------------------------------------------------------------------------- memory */

extern unsigned char __heap_base;       /* from wasm-ld: the first byte after the stack and the data */

#define ALIGN 16u                       /* of every block's payload */
#define HEADER 16u                      /* bytes before it: the block's size and, while free, the next one */
#define PAGE 65536u

typedef struct Block {
    size_t size;                        /* bytes of the block, header included: a multiple of ALIGN */
    struct Block *next;                 /* while free: the next free block up */
} Block;

static Block *free_list;                /* the free blocks in address order, none adjacent */
static unsigned char *heap_top;         /* the end of the blocks made so far */

static size_t round_up(size_t n) { return (n + (ALIGN - 1u)) & ~(size_t)(ALIGN - 1u); }
static unsigned char *end_of(Block *b) { return (unsigned char *)b + b->size; }

/* n bytes at a multiple of 16, or NULL when the memory cannot grow that far */
EXPORT(spg_alloc) void *spg_alloc(size_t n) {
    if (n > SIZE_MAX - 2u * ALIGN - HEADER) return NULL;
    const size_t need = round_up(n) + HEADER;
    for (Block **link = &free_list; *link; link = &(*link)->next) {     /* first fit */
        Block *b = *link;
        if (b->size < need) continue;
        if (b->size - need >= HEADER + ALIGN) {          /* the rest stays free */
            Block *rest = (Block *)(void *)((unsigned char *)b + need);
            rest->size = b->size - need;
            rest->next = b->next;
            *link = rest;
            b->size = need;
        } else {
            *link = b->next;
        }
        return (unsigned char *)b + HEADER;
    }
    if (!heap_top) heap_top = (unsigned char *)round_up((size_t)&__heap_base);
    const size_t top = (size_t)heap_top;
    if (need > SIZE_MAX - top) return NULL;
    const uint64_t end = (uint64_t)top + need, have = (uint64_t)__builtin_wasm_memory_size(0) * PAGE;
    if (end > have) {
        uint64_t pages = (end - have + PAGE - 1u) / PAGE;
        if (pages < 16u) pages = 16u;                    /* a megabyte at least, not a page per block */
        if (__builtin_wasm_memory_grow(0, (size_t)pages) == (size_t)-1 &&
            __builtin_wasm_memory_grow(0, (size_t)((end - have + PAGE - 1u) / PAGE)) == (size_t)-1)
            return NULL;
    }
    Block *b = (Block *)(void *)heap_top;
    b->size = need;
    heap_top += need;
    return (unsigned char *)b + HEADER;
}

/* Releases a block of spg_alloc() (NULL: nothing): it joins its free neighbours, and the top when it
   ends there. */
EXPORT(spg_free) void spg_free(void *p) {
    if (!p) return;
    Block *b = (Block *)(void *)((unsigned char *)p - HEADER);
    Block **link = &free_list, **prev_link = NULL;
    while (*link && *link < b) {
        prev_link = link;
        link = &(*link)->next;
    }
    b->next = *link;
    *link = b;
    if (b->next && end_of(b) == (unsigned char *)b->next) {
        b->size += b->next->size;
        b->next = b->next->next;
    }
    if (prev_link && end_of(*prev_link) == (unsigned char *)b) {
        (*prev_link)->size += b->size;
        (*prev_link)->next = b->next;
        b = *prev_link;
        link = prev_link;
    }
    if (!b->next && end_of(b) == heap_top) {
        heap_top = (unsigned char *)b;
        *link = NULL;
    }
}

/* ------------------------------------------------------------------------- models */

/* A model, its workspace and its input and output, in one block. */
typedef struct {
    SpingalettModel model;
    void *workspace;
    float *input, *output;
} Model;

static int status;                      /* the error code of the last call that failed */
static uint32_t words[64];              /* numbers for JavaScript to read */

EXPORT(spg_status) int spg_status(void) { return status; }

EXPORT(spg_format_version) int spg_format_version(void) { return SPINGALETT_FORMAT_VERSION; }

/* Checks the image of `size` bytes at `image` (a block of spg_alloc(), which must stay unchanged while
   the model is open) and opens it: the model with its workspace and buffers, or NULL with the error
   code in spg_status(). */
EXPORT(spg_model_new) Model *spg_model_new(const void *image, size_t size) {
    SpingalettModel model;
    const int rc = spingalett_model_init(&model, image, size);
    if (rc != SPINGALETT_OK) {
        status = rc;
        return NULL;
    }
    const uint64_t head = round_up(sizeof(Model)), workspace = (model.workspace_size + 15u) & ~(uint64_t)15u;
    const uint64_t in = ((uint64_t)model.input_size * 4u + 15u) & ~(uint64_t)15u;
    const uint64_t out = ((uint64_t)model.output_size * 4u + 15u) & ~(uint64_t)15u;
    const uint64_t total = head + workspace + in + out;
    unsigned char *block = total <= SIZE_MAX ? (unsigned char *)spg_alloc((size_t)total) : NULL;
    if (!block) {
        status = SPINGALETT_ERR_ALLOC;
        return NULL;
    }
    Model *m = (Model *)(void *)block;
    m->model = model;
    m->workspace = block + head;
    m->input = (float *)(void *)(block + head + workspace);
    m->output = (float *)(void *)(block + head + workspace + in);
    return m;
}

EXPORT(spg_model_free) void spg_model_free(Model *m) { spg_free(m); }

/* Where JavaScript writes a sample and reads its outputs. */
EXPORT(spg_model_input) float *spg_model_input(Model *m) { return m->input; }
EXPORT(spg_model_output) float *spg_model_output(Model *m) { return m->output; }

/* Runs the sample in the model's input into its output: SPINGALETT_OK or an error code. */
EXPORT(spg_model_run) int spg_model_run(Model *m) {
    return spingalett_model_run(&m->model, m->input, m->output, m->workspace);
}

/* The model's input_size, output_size, layer_count, loss, workspace_size and image_size, in words[0..5]. */
EXPORT(spg_model_info) uint32_t *spg_model_info(Model *m) {
    words[0] = m->model.input_size;
    words[1] = m->model.output_size;
    words[2] = m->model.layer_count;
    words[3] = (uint32_t)m->model.loss;
    words[4] = (uint32_t)m->model.workspace_size;
    words[5] = (uint32_t)m->model.image_size;
    return words;
}

/* Weight layer `index` (spingalett_model_layer()) in words[0..42], or NULL past the last: type, inputs,
   outputs, activation, precision, in_height, in_width, in_channels, height, width, channels, kernel_h,
   kernel_w, stride_h, stride_w, padding_h, padding_w, groups, epsilon (a float's bits), input_count,
   upsample, vocabulary, heads, kv_heads, rope_theta (a float's bits), causal, positions, then the
   input_count indices of input_layers. */
EXPORT(spg_model_layer) uint32_t *spg_model_layer(Model *m, uint32_t index) {
    SpingalettLayerInfo info;
    if (!spingalett_model_layer(&m->model, index, &info)) return NULL;
    const uint32_t fields[27] = {
        (uint32_t)info.type, info.inputs, info.outputs, (uint32_t)info.activation, (uint32_t)info.precision,
        info.in_height, info.in_width, info.in_channels, info.height, info.width, info.channels,
        info.kernel_h, info.kernel_w, info.stride_h, info.stride_w, info.padding_h, info.padding_w, info.groups,
        0u, info.input_count, (uint32_t)info.upsample, info.vocabulary, info.heads, info.kv_heads, 0u,
        info.causal ? 1u : 0u, info.positions ? 1u : 0u,
    };
    for (uint32_t k = 0; k < 27u; k++) words[k] = fields[k];
    __builtin_memcpy(&words[18], &info.epsilon, 4);
    __builtin_memcpy(&words[24], &info.rope_theta, 4);
    for (uint32_t k = 0; k < SPINGALETT_MAX_INPUTS; k++) words[27u + k] = k < info.input_count ? info.input_layers[k] : 0u;
    return words;
}
