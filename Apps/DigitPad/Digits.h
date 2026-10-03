/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/* Image helpers shared by the DigitPad app and its trainer: MNIST-style normalization of a
   drawing and random geometric/stroke augmentation of 28x28 digits. Header-only. */

#pragma once

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#define DIGIT_SIDE 28
#define DIGIT_PIXELS (DIGIT_SIDE * DIGIT_SIDE)

/* ---------------------------------------------------------------- normalization */

/* Integral of the piecewise-constant image over [0, x) x [0, y) for fractional x, y, from a
   summed-area table with (w + 1) x (h + 1) entries. Bilinear interpolation is exact here. */
static inline double digit_integral(const double *sat, int w, int h, double x, double y) {
    if (x <= 0 || y <= 0) return 0.0;
    if (x > w) x = w;
    if (y > h) y = h;
    int x0 = (int)x, y0 = (int)y;
    int x1 = x0 < w ? x0 + 1 : x0, y1 = y0 < h ? y0 + 1 : y0;
    double fx = x - x0, fy = y - y0;
    size_t stride = (size_t)w + 1;
    double a = sat[(size_t)y0 * stride + x0], b = sat[(size_t)y0 * stride + x1];
    double c = sat[(size_t)y1 * stride + x0], d = sat[(size_t)y1 * stride + x1];
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy;
}

/*
 * Converts a drawing (w x h intensities in [0, 1]) into an MNIST-style 28x28 input the way the
 * dataset was built: the ink's bounding box is scaled, keeping its aspect ratio, to fit a 20x20
 * box using area averaging (anti-aliasing), then the result is moved so that its center of mass
 * sits at the center of the 28x28 field. `sat` is scratch space for (w + 1) * (h + 1) doubles.
 * Returns false when there is no ink.
 */
static inline bool digit_normalize(const float *img, int w, int h, double *sat, float out[DIGIT_PIXELS]) {
    const float ink = 0.05f;
    int x0 = w, y0 = h, x1 = -1, y1 = -1;
    size_t stride = (size_t)w + 1;
    memset(sat, 0, stride * sizeof(double));
    for (int y = 0; y < h; y++) {
        double row = 0;
        sat[(size_t)(y + 1) * stride] = 0;
        for (int x = 0; x < w; x++) {
            float v = img[(size_t)y * w + x];
            row += v;
            sat[(size_t)(y + 1) * stride + x + 1] = sat[(size_t)y * stride + x + 1] + row;
            if (v > ink) {
                if (x < x0) x0 = x;
                if (x > x1) x1 = x;
                if (y < y0) y0 = y;
                if (y > y1) y1 = y;
            }
        }
    }
    memset(out, 0, DIGIT_PIXELS * sizeof(float));
    if (x1 < 0) return false;

    double bw = x1 - x0 + 1, bh = y1 - y0 + 1;
    double side = bw > bh ? bw : bh;                /* source pixels per 20 target pixels */
    double step = side / 20.0;                      /* source pixels per target pixel */
    /* the square of `side` source pixels, centered on the bounding box, maps onto 20x20 */
    double sx0 = x0 + bw / 2 - side / 2, sy0 = y0 + bh / 2 - side / 2;

    float box[20 * 20];
    double mass = 0, mx = 0, my = 0;
    for (int ty = 0; ty < 20; ty++)
        for (int tx = 0; tx < 20; tx++) {
            double ax = sx0 + tx * step, ay = sy0 + ty * step;
            double s = digit_integral(sat, w, h, ax + step, ay + step) - digit_integral(sat, w, h, ax, ay + step)
                     - digit_integral(sat, w, h, ax + step, ay) + digit_integral(sat, w, h, ax, ay);
            float v = (float)(s / (step * step));
            if (v > 1.0f) v = 1.0f;
            if (v < 0.0f) v = 0.0f;
            box[ty * 20 + tx] = v;
            mass += v; mx += v * (tx + 0.5); my += v * (ty + 0.5);
        }
    if (mass <= 0) return false;

    /* place the 20x20 box so that its center of mass lands on (14, 14) */
    int ox = (int)lround(14.0 - mx / mass), oy = (int)lround(14.0 - my / mass);
    for (int ty = 0; ty < 20; ty++)
        for (int tx = 0; tx < 20; tx++) {
            int x = tx + ox, y = ty + oy;
            if (x >= 0 && x < DIGIT_SIDE && y >= 0 && y < DIGIT_SIDE)
                out[y * DIGIT_SIDE + x] = box[ty * 20 + tx];
        }
    return true;
}

/* ---------------------------------------------------------------- augmentation */

typedef struct { uint64_t s; } DigitRng;

static inline uint64_t digit_rng_next(DigitRng *r) {      /* splitmix64 */
    uint64_t z = (r->s += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

static inline float digit_rng_uniform(DigitRng *r, float lo, float hi) {
    return lo + (hi - lo) * (float)((digit_rng_next(r) >> 40) * (1.0 / 16777216.0));
}

static inline float digit_bilinear(const float *img, float x, float y) {
    /* pixel centers at integer + 0.5 */
    x -= 0.5f; y -= 0.5f;
    int ix = (int)floorf(x), iy = (int)floorf(y);
    float fx = x - ix, fy = y - iy, v = 0;
    for (int dy = 0; dy < 2; dy++)
        for (int dx = 0; dx < 2; dx++) {
            int px = ix + dx, py = iy + dy;
            if (px < 0 || py < 0 || px >= DIGIT_SIDE || py >= DIGIT_SIDE) continue;
            v += img[py * DIGIT_SIDE + px] * (dx ? fx : 1 - fx) * (dy ? fy : 1 - fy);
        }
    return v;
}

/*
 * Random distortion of a 28x28 digit, covering what differs between MNIST and digits drawn with
 * a mouse: rotation, scale and aspect, shear, position and stroke thickness.
 */
static inline void digit_augment(const float in[DIGIT_PIXELS], float out[DIGIT_PIXELS], DigitRng *r) {
    const float pi = 3.14159265f;
    float angle = digit_rng_uniform(r, -15.0f, 15.0f) * pi / 180.0f;
    float scale = digit_rng_uniform(r, 0.80f, 1.15f), aspect = digit_rng_uniform(r, -0.12f, 0.12f);
    float shear = digit_rng_uniform(r, -0.25f, 0.25f);
    float tx = digit_rng_uniform(r, -2.5f, 2.5f), ty = digit_rng_uniform(r, -2.5f, 2.5f);

    /* forward map A = R * Shear * Scale around the center; sample with its inverse */
    float sx = scale * (1 + aspect), sy = scale * (1 - aspect);
    float c = cosf(angle), s = sinf(angle);
    float a00 = c * sx, a01 = (c * shear - s) * sy, a10 = s * sx, a11 = (s * shear + c) * sy;
    float det = a00 * a11 - a01 * a10;
    float i00 = a11 / det, i01 = -a01 / det, i10 = -a10 / det, i11 = a00 / det;

    float warped[DIGIT_PIXELS];
    for (int y = 0; y < DIGIT_SIDE; y++)
        for (int x = 0; x < DIGIT_SIDE; x++) {
            float u = x + 0.5f - 14.0f - tx, v = y + 0.5f - 14.0f - ty;
            warped[y * DIGIT_SIDE + x] = digit_bilinear(in, i00 * u + i01 * v + 14.0f, i10 * u + i11 * v + 14.0f);
        }

    /* stroke thickness: dilate (thicker) or soften-erode (thinner) with a 3x3 neighbourhood */
    float mode = digit_rng_uniform(r, 0.0f, 1.0f);
    if (mode < 0.30f || mode > 0.85f) {
        bool dilate = mode < 0.30f;
        for (int y = 0; y < DIGIT_SIDE; y++)
            for (int x = 0; x < DIGIT_SIDE; x++) {
                float m = warped[y * DIGIT_SIDE + x];
                for (int dy = -1; dy <= 1; dy++)
                    for (int dx = -1; dx <= 1; dx++) {
                        int px = x + dx, py = y + dy;
                        float v = (px < 0 || py < 0 || px >= DIGIT_SIDE || py >= DIGIT_SIDE) ? 0.0f : warped[py * DIGIT_SIDE + px];
                        m = dilate ? (v > m ? v : m) : (v < m ? v : m);
                    }
                out[y * DIGIT_SIDE + x] = dilate ? m : 0.5f * (m + warped[y * DIGIT_SIDE + x]);
            }
    } else {
        memcpy(out, warped, sizeof warped);
    }
    for (int i = 0; i < DIGIT_PIXELS; i++)
        if (out[i] > 1.0f) out[i] = 1.0f;
}
