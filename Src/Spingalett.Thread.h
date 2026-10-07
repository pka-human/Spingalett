/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/* A background thread with a lock and a condition to wait on: POSIX threads, or the Win32 API.
   Used by the data set reader to decode chunks ahead of the caller. */

#ifndef SPINGALETT_THREAD_H
#define SPINGALETT_THREAD_H

#include <stdbool.h>

typedef struct SpgThread SpgThread;     /* a running thread */
typedef struct SpgSignal SpgSignal;     /* a mutex and a condition variable */

/* Starts fn(arg) on a new thread; NULL when the system cannot create one. */
SpgThread *spg_thread_start(void (*fn)(void *), void *arg);
/* Waits for the thread to return and releases it. */
void spg_thread_join(SpgThread *thread);

SpgSignal *spg_signal_create(void);
void spg_signal_free(SpgSignal *signal);
void spg_lock(SpgSignal *signal);
void spg_unlock(SpgSignal *signal);
/* Releases the lock while it waits for spg_wake, then takes it again (wake-ups may be spurious). */
void spg_wait(SpgSignal *signal);
/* Wakes every waiter. */
void spg_wake(SpgSignal *signal);

#endif
