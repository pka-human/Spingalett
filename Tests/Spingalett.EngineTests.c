/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/*
 * Tests of the standalone inference engine: built from Src/Spingalett.Inference.c alone with
 * SPINGALETT_INFERENCE_ONLY, as on a microcontroller. A model with one layer per precision is
 * assembled byte by byte as docs/ModelFormat.md describes, so the test also pins the format; with
 * SPINGALETT_TEST_HEADERS the models exported as C headers by the full library run as well.
 */

#include <Spingalett/Spingalett.Inference.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;
#define CHECK(cond, ...) do { if (!(cond)) { failures++; printf("  FAIL: "); printf(__VA_ARGS__); printf("\n"); } } while (0)

/* ---------------------------------------------------------------- a hand-made image */

static uint32_t crc32_ieee(const uint8_t *p, size_t n) {
    uint32_t c = 0xFFFFFFFFu;
    for (size_t i = 0; i < n; i++) {
        c ^= p[i];
        for (int k = 0; k < 8; k++) c = (c >> 1) ^ (0xEDB88320u & (0u - (c & 1u)));
    }
    return ~c;
}
static void put16(uint8_t *p, uint32_t v) { p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8); }
static void put32(uint8_t *p, uint32_t v) { for (int i = 0; i < 4; i++) p[i] = (uint8_t)(v >> (8 * i)); }
static void put64(uint8_t *p, uint64_t v) { put32(p, (uint32_t)v); put32(p + 4, (uint32_t)(v >> 32)); }
static void putf(uint8_t *p, float f) { uint32_t u; memcpy(&u, &f, 4); put32(p, u); }
static size_t align16(size_t x) { return (x + 15) & ~(size_t)15; }

/* An activation's code in .slett files (docs/ModelFormat.md): 0 sigmoid, 1 ReLU, 2 tanh, 3 leaky ReLU,
   4 FOO52, 5 softmax, 6 none. */
static uint8_t act_code(ActivationFunction act) {
    switch (act) {
        case ACT_SIGMOID:    return 0;
        case ACT_RELU:       return 1;
        case ACT_TANH:       return 2;
        case ACT_LEAKY_RELU: return 3;
        case ACT_FOO52:      return 4;
        case ACT_SOFTMAX:    return 5;
        default:             return 6;
    }
}

typedef struct {
    uint32_t in, out;
    ActivationFunction act;
    PrecisionMode precision;
    double w[6][6];             /* the values the stored weights stand for */
    int q[6][6];                /* integer codes (integer precisions) */
    float scale[6];
    float bias[6];
} Spec;

/* One layer per precision; float weights are exactly representable in their format. */
static const Spec specs[] = {
    {5, 3, ACT_RELU, PRECISION_INT8, .q = {{127, -3, 40, 0, -127}, {5, 6, -7, 8, -9}, {0, 0, 0, 0, 0}},
     .scale = {0.01f, 0.125f, 0.0f}, .bias = {0.25f, -0.5f, 0.75f}},
    {3, 4, ACT_TANH, PRECISION_INT4, .q = {{7, -7, 1}, {-1, 2, -3}, {4, 0, -5}, {6, 6, 6}},
     .scale = {0.1f, 0.2f, 0.05f, 0.01f}, .bias = {0.0f, 0.1f, -0.1f, 0.2f}},
    {4, 3, ACT_LEAKY_RELU, PRECISION_FP16, .w = {{0.5, -1.25, 2.0, 0.375}, {-0.75, 0.0625, 1.5, -2.5}, {1.0, 1.0, -1.0, 0.125}},
     .bias = {0.0f, -0.25f, 0.5f}},
    {3, 3, ACT_SIGMOID, PRECISION_BFLOAT16, .w = {{0.5, -0.25, 1.0}, {2.0, 0.75, -1.5}, {-0.125, 0.0, 0.5}},
     .bias = {0.1f, 0.2f, 0.3f}},
    {3, 2, ACT_NONE, PRECISION_FLOAT32, .w = {{0.3, -0.7, 1.1}, {-1.3, 0.9, 0.2}}, .bias = {0.05f, -0.05f}},
    {2, 2, ACT_SOFTMAX, PRECISION_INT2, .q = {{1, -1}, {0, 1}}, .scale = {0.6f, 1.4f}, .bias = {0.0f, 0.1f}},
};
#define LAYERS ((uint32_t)(sizeof specs / sizeof *specs))

static uint16_t half_bits(double v) {                  /* exact for the values above */
    if (v == 0) return 0;
    int e;
    double m = frexp(fabs(v), &e);                       /* v = m * 2^e, m in [0.5, 1) */
    return (uint16_t)((v < 0 ? 0x8000 : 0) | (uint32_t)(e + 14) << 10 | (uint32_t)((m * 2 - 1) * 1024));
}

static uint8_t *build_image(size_t *size) {
    size_t pos = align16(64 + 48 * LAYERS), off[LAYERS][3];
    for (uint32_t l = 0; l < LAYERS; l++) {
        const Spec *s = &specs[l];
        size_t row = s->precision == PRECISION_FLOAT32 ? 4u * s->in : s->precision <= PRECISION_BFLOAT16 ? 2u * s->in
                   : s->precision == PRECISION_INT8 ? s->in : s->precision == PRECISION_INT4 ? (s->in + 1) / 2 : (s->in + 3) / 4;
        off[l][0] = pos; pos = align16(pos + row * s->out);
        off[l][1] = 0;
        if (s->precision >= PRECISION_INT8) { off[l][1] = pos; pos = align16(pos + 4u * s->out); }
        off[l][2] = pos; pos = align16(pos + 4u * s->out);
    }
    uint8_t *img = calloc(1, pos);
    memcpy(img, "SLETTM", 6);
    put16(img + 6, 3);
    put32(img + 8, LAYERS + 1);
    img[12] = LOSS_CROSS_ENTROPY;
    put64(img + 24, pos);
    for (uint32_t l = 0; l < LAYERS; l++) {
        const Spec *s = &specs[l];
        uint8_t *e = img + 64 + 48 * l;
        put32(e, s->in); put32(e + 4, s->out);
        e[8] = act_code(s->act); e[9] = (uint8_t)s->precision;
        put64(e + 16, off[l][0]); put64(e + 24, off[l][1]); put64(e + 32, off[l][2]);
        for (uint32_t j = 0; j < s->out; j++) {
            putf(img + off[l][2] + 4 * j, s->bias[j]);
            if (off[l][1]) putf(img + off[l][1] + 4 * j, s->scale[j]);
            for (uint32_t k = 0; k < s->in; k++) {
                uint8_t *w = img + off[l][0];
                switch (s->precision) {
                    case PRECISION_FLOAT32:  putf(w + 4 * (j * s->in + k), (float)s->w[j][k]); break;
                    case PRECISION_FP16:     put16(w + 2 * (j * s->in + k), half_bits(s->w[j][k])); break;
                    case PRECISION_BFLOAT16: { float f = (float)s->w[j][k]; uint32_t u; memcpy(&u, &f, 4); put16(w + 2 * (j * s->in + k), u >> 16); break; }
                    case PRECISION_INT8:     w[j * s->in + k] = (uint8_t)(int8_t)s->q[j][k]; break;
                    case PRECISION_INT4:     w[j * ((s->in + 1) / 2) + k / 2] |= (uint8_t)((s->q[j][k] & 15) << (4 * (k & 1))); break;
                    default:                 w[j * ((s->in + 3) / 4) + k / 4] |= (uint8_t)((s->q[j][k] & 3) << (2 * (k & 3))); break;
                }
            }
        }
    }
    put32(img + 56, crc32_ieee(img + 64, pos - 64));
    put32(img + 60, crc32_ieee(img, 60));
    *size = pos;
    return img;
}

/* The engine's arithmetic, written out: integer layers quantize their input (scale max|x| / 127,
   nearest even) and accumulate exactly; float layers in double. */
static void reference(const float *input, float *output) {
    float x[6], y[6];
    memcpy(x, input, specs[0].in * sizeof(float));
    for (uint32_t l = 0; l < LAYERS; l++) {
        const Spec *s = &specs[l];
        if (s->precision >= PRECISION_INT8) {
            float amax = 0;
            for (uint32_t k = 0; k < s->in; k++) if (fabsf(x[k]) > amax) amax = fabsf(x[k]);
            float a = amax / 127.0f, inv = amax > 0 ? 127.0f / amax : 0;
            for (uint32_t j = 0; j < s->out; j++) {
                long acc = 0;
                for (uint32_t k = 0; k < s->in; k++) acc += s->q[j][k] * lrintf(x[k] * inv);
                y[j] = s->bias[j] + (s->scale[j] * a) * (float)acc;
            }
        } else {
            for (uint32_t j = 0; j < s->out; j++) {
                double acc = s->bias[j];
                for (uint32_t k = 0; k < s->in; k++) acc += s->w[j][k] * x[k];
                y[j] = (float)acc;
            }
        }
        double sum = 0, m = -1e30;
        for (uint32_t j = 0; j < s->out; j++) m = y[j] > m ? y[j] : m;
        for (uint32_t j = 0; j < s->out; j++) {
            double v = y[j];
            switch (s->act) {
                case ACT_RELU:       v = v > 0 ? v : 0; break;
                case ACT_LEAKY_RELU: v = v > 0 ? v : 0.01 * v; break;
                case ACT_TANH:       v = tanh(v); break;
                case ACT_SIGMOID:    v = 1 / (1 + exp(-v)); break;
                case ACT_SOFTMAX:    v = exp(v - m); sum += v; break;
                default: break;
            }
            y[j] = (float)v;
        }
        if (s->act == ACT_SOFTMAX) for (uint32_t j = 0; j < s->out; j++) y[j] = (float)(y[j] / sum);
        memcpy(x, y, s->out * sizeof(float));
    }
    memcpy(output, x, specs[LAYERS - 1].out * sizeof(float));
}

static void hand_made_model(void) {
    size_t size;
    uint8_t *img = build_image(&size);
    SpingalettModel m;
    int rc = spingalett_model_init(&m, img, size);
    CHECK(rc == SPINGALETT_OK, "hand-made image rejected (%d)", rc);
    if (rc != SPINGALETT_OK) { free(img); return; }
    CHECK(m.input_size == 5 && m.output_size == 2 && m.layer_count == LAYERS && m.loss == LOSS_CROSS_ENTROPY &&
          m.image_size == size && m.workspace_size >= 16, "model fields");
    for (uint32_t l = 0; l < LAYERS; l++) {
        SpingalettLayerInfo info;
        CHECK(spingalett_model_layer(&m, l, &info) && info.inputs == specs[l].in && info.outputs == specs[l].out &&
              info.activation == specs[l].act && info.precision == specs[l].precision, "layer %u info", l);
    }

    void *ws = malloc(m.workspace_size);               /* exactly the stated size */
    static const float inputs[][5] = {
        {1.0f, -2.0f, 0.5f, 3.0f, -0.25f}, {0, 0, 0, 0, 0}, {-1, -1, -1, -1, -1}, {100, 0.001f, -50, 7, 1e-3f},
        {0.5f, 0.5f, 0.5f, 0.5f, 0.5f},                 /* ties: x * 127 / max = 127 exactly */
    };
    float worst = 0;
    for (size_t t = 0; t < sizeof inputs / sizeof *inputs; t++) {
        float out[2], ref[2];
        rc = spingalett_model_run(&m, inputs[t], out, ws);
        reference(inputs[t], ref);
        CHECK(rc == SPINGALETT_OK, "run %zu failed (%d)", t, rc);
        for (int k = 0; k < 2; k++) {
            float d = fabsf(out[k] - ref[k]);
            if (d > worst) worst = d;
        }
    }
    printf("  hand-made model, one layer per precision: max |engine - reference| = %.1e\n", worst);
    CHECK(worst < 1e-6f, "outputs differ from the reference by %.3e", worst);

    float out[2];
    CHECK(spingalett_model_run(&m, inputs[0], out, NULL) == SPINGALETT_ERR_INVALID, "NULL workspace must fail");
    CHECK(spingalett_model_run(&m, inputs[0], out, (char *)ws + 1) == SPINGALETT_ERR_INVALID, "misaligned workspace must fail");
    CHECK(spingalett_model_run(NULL, inputs[0], out, ws) == SPINGALETT_ERR_INVALID, "NULL model must fail");
    free(ws);

    /* damage: magic, version, checksum, truncation, a layer inconsistent with its neighbour */
    uint8_t *bad = malloc(size);
    memcpy(bad, img, size); bad[0] = 'X';
    CHECK(spingalett_model_init(&m, bad, size) == SPINGALETT_ERR_INVALID, "bad magic");
    memcpy(bad, img, size); bad[6] = 2;
    CHECK(spingalett_model_init(&m, bad, size) == SPINGALETT_ERR_FORMAT_VERSION, "other version");
    memcpy(bad, img, size); bad[size - 20] ^= 1;
    CHECK(spingalett_model_init(&m, bad, size) == SPINGALETT_ERR_INVALID, "payload checksum");
    CHECK(spingalett_model_init(&m, img, size - 1) == SPINGALETT_ERR_FILE_IO, "truncated image");
    memcpy(bad, img, size); put32(bad + 64 + 48, 4);
    put32(bad + 56, crc32_ieee(bad + 64, size - 64)); put32(bad + 60, crc32_ieee(bad, 60));
    CHECK(spingalett_model_init(&m, bad, size) == SPINGALETT_ERR_INVALID, "layer inputs differ from previous outputs");
    free(bad);
    free(img);
}

/* ---------------------------------------------------------------- models exported as C headers */

#if defined(SPINGALETT_TEST_HEADERS)
#include "test_model_int8.h"
#include "test_model_int4.h"
#include "test_model_fp16.h"
#include "test_model_conv_int8.h"
#include "test_model_conv_f32.h"
#include "test_model_norm_int8.h"
#include "test_model_expected.h"

static void exported_headers(void) {
    struct { const uint8_t *image; size_t size; uint32_t in, out; size_t ws; const float *inputs, *expected; const char *name; } m[] = {
        {test_model_int8, TEST_MODEL_INT8_SIZE, TEST_MODEL_INT8_INPUTS, TEST_MODEL_INT8_OUTPUTS, TEST_MODEL_INT8_WORKSPACE,
         test_inputs[0], expected_int8, "INT8"},
        {test_model_int4, TEST_MODEL_INT4_SIZE, TEST_MODEL_INT4_INPUTS, TEST_MODEL_INT4_OUTPUTS, TEST_MODEL_INT4_WORKSPACE,
         test_inputs[0], expected_int4, "INT4"},
        {test_model_fp16, TEST_MODEL_FP16_SIZE, TEST_MODEL_FP16_INPUTS, TEST_MODEL_FP16_OUTPUTS, TEST_MODEL_FP16_WORKSPACE,
         test_inputs[0], expected_fp16, "FP16"},
        {test_model_conv_int8, TEST_MODEL_CONV_INT8_SIZE, TEST_MODEL_CONV_INT8_INPUTS, TEST_MODEL_CONV_INT8_OUTPUTS,
         TEST_MODEL_CONV_INT8_WORKSPACE, test_conv_inputs[0], expected_conv_int8, "convolution INT8"},
        {test_model_conv_f32, TEST_MODEL_CONV_F32_SIZE, TEST_MODEL_CONV_F32_INPUTS, TEST_MODEL_CONV_F32_OUTPUTS,
         TEST_MODEL_CONV_F32_WORKSPACE, test_conv_inputs[0], expected_conv_f32, "convolution FLOAT32"},
        {test_model_norm_int8, TEST_MODEL_NORM_INT8_SIZE, TEST_MODEL_NORM_INT8_INPUTS, TEST_MODEL_NORM_INT8_OUTPUTS,
         TEST_MODEL_NORM_INT8_WORKSPACE, test_conv_inputs[0], expected_norm_int8, "batch-normalized INT8"},
    };
    static float workspace[(TEST_MODEL_INT8_WORKSPACE + TEST_MODEL_FP16_WORKSPACE + TEST_MODEL_CONV_INT8_WORKSPACE +
                            TEST_MODEL_CONV_F32_WORKSPACE + TEST_MODEL_NORM_INT8_WORKSPACE) / sizeof(float)];
    for (int i = 0; i < (int)(sizeof m / sizeof *m); i++) {
        SpingalettModel model;
        int rc = spingalett_model_init(&model, m[i].image, m[i].size);
        CHECK(rc == SPINGALETT_OK && model.input_size == m[i].in && model.output_size == m[i].out &&
              model.workspace_size == m[i].ws && model.workspace_size <= sizeof workspace, "%s header model", m[i].name);
        if (rc != SPINGALETT_OK) continue;
        float worst = 0, out[16];
        for (int s = 0; s < TEST_SAMPLES; s++) {
            spingalett_model_run(&model, m[i].inputs + (size_t)s * m[i].in, out, workspace);
            for (uint32_t k = 0; k < model.output_size; k++) {
                float d = fabsf(out[k] - m[i].expected[s * model.output_size + k]);
                if (d > worst) worst = d;
            }
        }
        printf("  %s model from its C header: max |standalone - library| = %.1e\n", m[i].name, worst);
        CHECK(worst < 1e-5f, "%s header outputs differ by %.3e", m[i].name, worst);
    }
}
#endif

int main(void) {
    printf("[standalone engine]\n");
    hand_made_model();
#if defined(SPINGALETT_TEST_HEADERS)
    exported_headers();
#endif
    printf("%s (%d failures)\n", failures ? "FAILED" : "ALL PASSED", failures);
    return failures != 0;
}
