// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "src/xnnpack/init-once.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include "src/xnnpack/common.h"

#ifdef _WIN32
  #include <windows.h>
#else
  #include <pthread.h>
#endif

// Initialization guards keep track of the config generation they were
// initialized on.
//
// If the generation stored in the guard is different from this value it means
// the initialization needs to be run again.
//
// This is initialized to 1 to invalidate guards that are initialized to 0.
uint32_t xnn_init_generation = 1;

#if XNN_PLATFORM_WINDOWS || XNN_HAS_PTHREADS
void xnn_init_once_impl(struct xnn_init_guard* guard, XNN_ONCE_LOCK_TYPE* lock, void (*init_fn)(void)) {
#if XNN_PLATFORM_WINDOWS
  AcquireSRWLockExclusive(lock);
#elif XNN_HAS_PTHREADS
  pthread_mutex_lock(lock);
#endif

  if (guard->generation != xnn_init_generation) {
    init_fn();
    guard->generation = xnn_init_generation;
  }

#if XNN_PLATFORM_WINDOWS
  ReleaseSRWLockExclusive(lock);
#elif XNN_HAS_PTHREADS
  pthread_mutex_unlock(lock);
#endif
}
#else
void xnn_init_once_impl(struct xnn_init_guard* guard, void (*init_fn)(void)) {
  if (guard->generation != xnn_init_generation) {
    guard->generation = xnn_init_generation;
    init_fn();
  }
}
#endif

void xnn_reset_all_init_guards(void) {
  // We increment twice when the unsigned wraps over to 0 because uninitialized
  // guards have their `generation` set to 0.
  (void)(++xnn_init_generation || ++xnn_init_generation);
}
