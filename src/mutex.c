// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <string.h>

#include "xnnpack.h"
#include "xnnpack/common.h"
#include "xnnpack/log.h"
#include "xnnpack/mutex.h"

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

enum xnn_status xnn_mutex_init(struct xnn_mutex* mutex) {
#if XNN_PLATFORM_WINDOWS
  mutex->handle = CreateMutexW(
      /* security attributes */ NULL,
      /* initially owned */ FALSE,
      /* name */ NULL);
  if (mutex->handle == NULL) {
    xnn_log_error("failed to initialize mutex, error code: %" PRIu32, (uint32_t) GetLastError());
    return xnn_status_out_of_memory;
  }
#elif XNN_PLATFORM_MACOS || XNN_PLATFORM_IOS
  mutex->semaphore = dispatch_semaphore_create(1);
  if (mutex->semaphore == NULL) {
    xnn_log_error("failed to initialize mutex");
    return xnn_status_out_of_memory;
  }
#elif !XNN_PLATFORM_WEB || defined(__EMSCRIPTEN_PTHREADS__)
  const int ret = pthread_mutex_init(&mutex->mutex, NULL);
  if (ret != 0) {
    xnn_log_error("failed to initialize mutex, error code: %d", ret);
    return xnn_status_out_of_memory;
  }
#endif
  return xnn_status_success;
}

enum xnn_status xnn_mutex_lock(struct xnn_mutex* mutex) {
#if XNN_PLATFORM_WINDOWS
  const DWORD wait_result = WaitForSingleObject(mutex->handle, INFINITE);
  if (WAIT_OBJECT_0 != wait_result) {
    xnn_log_error("failed to lock mutex, error code: %" PRIu32, (uint32_t) wait_result);
    return xnn_status_invalid_state;
  }
#elif XNN_PLATFORM_MACOS || XNN_PLATFORM_IOS
  const int wait_result = dispatch_semaphore_wait(mutex->semaphore, DISPATCH_TIME_FOREVER);
  if (0 != wait_result) {
    xnn_log_error("failed to lock mutex, error code: %d", wait_result);
    return xnn_status_invalid_state;
  }
#elif !XNN_PLATFORM_WEB || defined(__EMSCRIPTEN_PTHREADS__)
  const int ret = pthread_mutex_lock(&mutex->mutex);
  if (ret != 0) {
    xnn_log_error("failed to lock mutex, error code: %d", ret);
    return xnn_status_invalid_state;
  }
#endif
  return xnn_status_success;
}

enum xnn_status xnn_mutex_unlock(struct xnn_mutex* mutex) {
#if XNN_PLATFORM_WINDOWS
  if (ReleaseMutex(mutex->handle) == 0) {
    xnn_log_error("failed to unlock mutex, error code: %" PRIu32, (uint32_t) GetLastError());
    return xnn_status_invalid_state;
  }
#elif XNN_PLATFORM_MACOS || XNN_PLATFORM_IOS
  dispatch_semaphore_signal(mutex->semaphore);
#elif !XNN_PLATFORM_WEB || defined(__EMSCRIPTEN_PTHREADS__)
  const int ret = pthread_mutex_unlock(&mutex->mutex);
  if (ret != 0) {
    xnn_log_error("failed to unlock mutex, error code: %d", ret);
    return xnn_status_invalid_state;
  }
#endif
  return xnn_status_success;
}

enum xnn_status xnn_mutex_destroy(struct xnn_mutex* mutex) {
#if XNN_PLATFORM_WINDOWS
  if (CloseHandle(mutex->handle) == 0) {
    xnn_log_error("failed to destroy mutex, error code: %" PRIu32, (uint32_t) GetLastError());
    return xnn_status_invalid_state;
  }
#elif XNN_PLATFORM_MACOS || XNN_PLATFORM_IOS
  dispatch_release(mutex->semaphore);
#elif !XNN_PLATFORM_WEB || defined(__EMSCRIPTEN_PTHREADS__)
  const int ret = pthread_mutex_destroy(&mutex->mutex);
  if (ret != 0) {
    xnn_log_error("failed to destroy mutex, error code: %d", ret);
    return xnn_status_invalid_state;
  }
#endif
  memset(mutex, 0, sizeof(struct xnn_mutex));
  return xnn_status_success;
}
