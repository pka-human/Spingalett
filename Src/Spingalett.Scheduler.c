/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

#include "Spingalett.Private.h"
#include <math.h>

#define SPINGALETT_PI 3.14159265358979323846

static const LRScheduleParams default_params = {0};

static const LRScheduleParams *params_or_default(void *params) {
    return params ? (const LRScheduleParams *)params : &default_params;
}

static size_t warmup_epochs(const LRScheduleParams *p, size_t total) {
    size_t w = p->warmup_epochs ? p->warmup_epochs : total / 20;
    return w ? w : 1;
}

/* Half-cosine from initial_lr (progress 0) towards min_lr (progress 1). */
static float cosine(float initial_lr, float min_lr, double progress) {
    if (progress > 1.0) progress = 1.0;
    return min_lr + (initial_lr - min_lr) * (float)(0.5 * (1.0 + cos(SPINGALETT_PI * progress)));
}

float spingalett_lr_cosine_decay(size_t epoch, size_t total_epochs, float initial_lr, void *params) {
    const LRScheduleParams *p = params_or_default(params);
    if (total_epochs == 0) return initial_lr;
    return cosine(initial_lr, p->min_lr, (double)epoch / (double)total_epochs);
}

float spingalett_lr_linear_warmup(size_t epoch, size_t total_epochs, float initial_lr, void *params) {
    size_t w = warmup_epochs(params_or_default(params), total_epochs);
    if (epoch >= w) return initial_lr;
    return initial_lr * (float)(epoch + 1) / (float)w;
}

float spingalett_lr_step_decay(size_t epoch, size_t total_epochs, float initial_lr, void *params) {
    const LRScheduleParams *p = params_or_default(params);
    size_t step = p->step_size ? p->step_size : total_epochs / 3;
    float gamma = p->gamma > 0.0f ? p->gamma : 0.1f;
    if (step == 0) step = 1;
    return initial_lr * powf(gamma, (float)(epoch / step));
}

float spingalett_lr_warmup_cosine(size_t epoch, size_t total_epochs, float initial_lr, void *params) {
    const LRScheduleParams *p = params_or_default(params);
    size_t w = warmup_epochs(p, total_epochs);
    if (epoch < w) return initial_lr * (float)(epoch + 1) / (float)w;
    if (total_epochs <= w) return initial_lr;
    return cosine(initial_lr, p->min_lr, (double)(epoch - w) / (double)(total_epochs - w));
}
