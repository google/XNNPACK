// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include "xnnpack.h"
#include "xnnpack/common.h"

#if XNN_PLATFORM_WINDOWS
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#elif XNN_PLATFORM_MACOS || XNN_PLATFORM_IOS
#include <dispatch/dispatch.h>
#else
#include <pthread.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

struct xnn_mutex {
#if XNN_PLATFORM_WINDOWS
  HANDLE handle;
#elif XNN_PLATFORM_MACOS || XNN_PLATFORM_IOS
  dispatch_semaphore_t semaphore;
#elif XNN_PLATFORM_WEB && !defined(__EMSCRIPTEN_PTHREADS__)
  char _; // Dummy member variable to comply with the C standard
#else
  pthread_mutex_t mutex;
#endif
};

enum xnn_status xnn_mutex_init(struct xnn_mutex* mutex);
enum xnn_status xnn_mutex_lock(struct xnn_mutex* mutex);
enum xnn_status xnn_mutex_unlock(struct xnn_mutex* mutex);
enum xnn_status xnn_mutex_destroy(struct xnn_mutex* mutex);

#ifdef __cplusplus
} // extern "C"
#endif
