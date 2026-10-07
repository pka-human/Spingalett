/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/* Background threads: POSIX threads, or the Win32 API (also with MinGW). */

#include "Spingalett.Thread.h"
#include <stdlib.h>

#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>

struct SpgThread {
    HANDLE handle;
    void (*fn)(void *);
    void *arg;
};

struct SpgSignal {
    CRITICAL_SECTION lock;
    CONDITION_VARIABLE cond;
};

static DWORD WINAPI thread_main(LPVOID p) {
    SpgThread *t = (SpgThread *)p;
    t->fn(t->arg);
    return 0;
}

SpgThread *spg_thread_start(void (*fn)(void *), void *arg) {
    SpgThread *t = (SpgThread *)malloc(sizeof *t);
    if (!t) return NULL;
    t->fn = fn;
    t->arg = arg;
    t->handle = CreateThread(NULL, 0, thread_main, t, 0, NULL);
    if (!t->handle) { free(t); return NULL; }
    return t;
}

void spg_thread_join(SpgThread *t) {
    if (!t) return;
    WaitForSingleObject(t->handle, INFINITE);
    CloseHandle(t->handle);
    free(t);
}

SpgSignal *spg_signal_create(void) {
    SpgSignal *s = (SpgSignal *)malloc(sizeof *s);
    if (!s) return NULL;
    InitializeCriticalSection(&s->lock);
    InitializeConditionVariable(&s->cond);
    return s;
}

void spg_signal_free(SpgSignal *s) {
    if (!s) return;
    DeleteCriticalSection(&s->lock);
    free(s);
}

void spg_lock(SpgSignal *s) { EnterCriticalSection(&s->lock); }
void spg_unlock(SpgSignal *s) { LeaveCriticalSection(&s->lock); }
void spg_wait(SpgSignal *s) { SleepConditionVariableCS(&s->cond, &s->lock, INFINITE); }
void spg_wake(SpgSignal *s) { WakeAllConditionVariable(&s->cond); }

#else
#  include <pthread.h>

struct SpgThread {
    pthread_t handle;
    void (*fn)(void *);
    void *arg;
};

struct SpgSignal {
    pthread_mutex_t lock;
    pthread_cond_t cond;
};

static void *thread_main(void *p) {
    SpgThread *t = (SpgThread *)p;
    t->fn(t->arg);
    return NULL;
}

SpgThread *spg_thread_start(void (*fn)(void *), void *arg) {
    SpgThread *t = (SpgThread *)malloc(sizeof *t);
    if (!t) return NULL;
    t->fn = fn;
    t->arg = arg;
    if (pthread_create(&t->handle, NULL, thread_main, t) != 0) { free(t); return NULL; }
    return t;
}

void spg_thread_join(SpgThread *t) {
    if (!t) return;
    pthread_join(t->handle, NULL);
    free(t);
}

SpgSignal *spg_signal_create(void) {
    SpgSignal *s = (SpgSignal *)malloc(sizeof *s);
    if (!s) return NULL;
    if (pthread_mutex_init(&s->lock, NULL) != 0) { free(s); return NULL; }
    if (pthread_cond_init(&s->cond, NULL) != 0) { pthread_mutex_destroy(&s->lock); free(s); return NULL; }
    return s;
}

void spg_signal_free(SpgSignal *s) {
    if (!s) return;
    pthread_cond_destroy(&s->cond);
    pthread_mutex_destroy(&s->lock);
    free(s);
}

void spg_lock(SpgSignal *s) { pthread_mutex_lock(&s->lock); }
void spg_unlock(SpgSignal *s) { pthread_mutex_unlock(&s->lock); }
void spg_wait(SpgSignal *s) { pthread_cond_wait(&s->cond, &s->lock); }
void spg_wake(SpgSignal *s) { pthread_cond_broadcast(&s->cond); }
#endif
