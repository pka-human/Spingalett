/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/* What the engine (Src/Spingalett.Inference.c with SPINGALETT_INFERENCE_ONLY) takes from <math.h>, for
   its WebAssembly build, which has no C library: the functions WebAssembly has instructions for inline,
   the others in Spingalett.Libm.c. */

#ifndef SPINGALETT_WASM_MATH_H
#define SPINGALETT_WASM_MATH_H

#define NAN      (__builtin_nanf(""))
#define INFINITY (__builtin_inff())

static inline float sqrtf(float x) { return __builtin_sqrtf(x); }          /* f32.sqrt */
static inline float fabsf(float x) { return __builtin_fabsf(x); }          /* f32.abs */
static inline long lrintf(float x) { return (long)__builtin_rintf(x); }    /* f32.nearest: ties to even */

float expf(float x);
float tanhf(float x);
float erff(float x);
double pow(double x, double y);
double sin(double x);
double cos(double x);

#endif
