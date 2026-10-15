/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

#include "Spingalett.Private.h"
#include <math.h>
#include <time.h>
#include <stdbool.h>

static _Thread_local uint32_t rng_state[4];
static _Thread_local bool     rng_initialized = false;

static inline uint32_t rotl32(uint32_t x, int k) {
    return (x << k) | (x >> (32 - k));
}

void rng_seed(uint64_t seed) {
    for (int i = 0; i < 4; i++) {
        seed += 0x9e3779b97f4a7c15ULL;
        uint64_t z = seed;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        z = z ^ (z >> 31);
        rng_state[i] = (uint32_t)z;
    }
    if (rng_state[0] == 0 && rng_state[1] == 0 &&
        rng_state[2] == 0 && rng_state[3] == 0)
        rng_state[0] = 1;
    rng_initialized = true;
}

static void rng_ensure_seeded(void) {
    if (!rng_initialized) {
        uint64_t seed = (uint64_t)time(NULL);
        seed ^= (uint64_t)clock() * 6364136223846793005ULL;
        volatile int stack_var = 0;
        seed ^= (uint64_t)(uintptr_t)&stack_var;
        rng_seed(seed);
    }
}

uint32_t rng_next(void) {
    rng_ensure_seeded();
    uint32_t result = rng_state[0] + rng_state[3];
    uint32_t t = rng_state[1] << 9;

    rng_state[2] ^= rng_state[0];
    rng_state[3] ^= rng_state[1];
    rng_state[1] ^= rng_state[2];
    rng_state[0] ^= rng_state[3];
    rng_state[2] ^= t;
    rng_state[3]  = rotl32(rng_state[3], 11);

    return result;
}

uint64_t rng_next64(void) {
    uint64_t hi = rng_next();
    return (hi << 32) | rng_next();
}

void spingalett_seed(uint64_t seed) {
    rng_seed(seed);
}

float rng_next_float(void) {
    return (float)(rng_next() >> 8) / 16777216.0f;
}

float random_uniform_weight(void) {
    return rng_next_float() * 2.0f - 1.0f;
}

float random_normal_weight(void) {
    float u1 = rng_next_float();
    float u2 = rng_next_float();
    if (u1 < 1e-6f) u1 = 1e-6f;
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265f * u2);
}

static inline uint64_t mix64(uint64_t z) {
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

void spingalett_permutation(uint32_t *order, uint32_t n, uint64_t seed, uint64_t index) {
    for (uint32_t i = 0; i < n; i++) order[i] = i;
    uint64_t state = mix64(seed ^ mix64(index + 0x9E3779B97F4A7C15ull));
    for (uint32_t i = n; i > 1; i--) {
        uint32_t j = (uint32_t)(((mix64(state += 0x9E3779B97F4A7C15ull) >> 32) * i) >> 32), t = order[i - 1];
        order[i - 1] = order[j];
        order[j] = t;
    }
}
