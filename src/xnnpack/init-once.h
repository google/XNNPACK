// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_SRC_XNNPACK_INIT_ONCE_H_
#define XNNPACK_SRC_XNNPACK_INIT_ONCE_H_

#include <stdint.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <pthread.h>
#include <stdbool.h>
#endif

#include "src/xnnpack/common.h"

#ifdef __cplusplus
extern "C" {
#endif

#if XNN_PLATFORM_WINDOWS
  #define XNN_ONCE_LOCK_TYPE SRWLOCK
  #define XNN_ONCE_LOCK_INIT SRWLOCK_INIT
#elif XNN_HAS_PTHREADS
  #define XNN_ONCE_LOCK_TYPE pthread_mutex_t
  #define XNN_ONCE_LOCK_INIT PTHREAD_MUTEX_INITIALIZER
#endif

struct xnn_init_guard {
  // We have a global configuration generation that is changed every time
  // `xnn_reset_all_init_guards` is called. This is equal to the generation
  // value at the time the initialization was run and checked against to check
  // if the initialization needs to be run again.
  uint32_t generation;
};

#define XNN_INIT_GUARD_INIT {0}

XNN_INTERNAL void xnn_reset_all_init_guards();

#if XNN_PLATFORM_WINDOWS || XNN_HAS_PTHREADS
XNN_INTERNAL void xnn_init_once_impl(struct xnn_init_guard* guard,
                                     XNN_ONCE_LOCK_TYPE* lock,
                                     void (*init_fn)(void));
#else
XNN_INTERNAL void xnn_init_once_impl(struct xnn_init_guard* guard,
                                     void (*init_fn)(void));
#endif

#if XNN_PLATFORM_WINDOWS || XNN_HAS_PTHREADS
#define XNN_INIT_ONCE_GUARD(name)                             \
  static void init_##name##_config(void);                     \
  static XNN_ONCE_LOCK_TYPE name##_lock = XNN_ONCE_LOCK_INIT; \
  static struct xnn_init_guard name##_guard = XNN_INIT_GUARD_INIT

#define XNN_INIT_ONCE(name) \
  xnn_init_once_impl(&name##_guard, &name##_lock, &init_##name##_config)
#else
#define XNN_INIT_ONCE_GUARD(name)         \
  static void init_##name##_config(void); \
  static struct xnn_init_guard name##_guard = XNN_INIT_GUARD_INIT

#define XNN_INIT_ONCE(name) \
  xnn_init_once_impl(&name##_guard, &init_##name##_config)
#endif

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // XNNPACK_SRC_XNNPACK_INIT_ONCE_H_
