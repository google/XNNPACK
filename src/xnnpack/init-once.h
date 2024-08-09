// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#ifdef _WIN32
  #include <windows.h>
#else
  #include <pthread.h>
#endif

#include "xnnpack.h"
#include "xnnpack/common.h"

#if XNN_PLATFORM_WINDOWS

  #define XNN_INIT_ONCE_GUARD(name) \
    static void init_##name##_config(void); \
    static BOOL CALLBACK name##_windows_wrapper(PINIT_ONCE init_once, PVOID parameter, PVOID* context) { \
      init_##name##_config(); \
      return TRUE; \
    } \
    static INIT_ONCE name##_guard = INIT_ONCE_STATIC_INIT /* no semicolon */

  #define XNN_INIT_ONCE(name) \
    InitOnceExecuteOnce(&name##_guard, &name##_windows_wrapper, NULL, NULL) /* no semicolon */

#else

  #define XNN_INIT_ONCE_GUARD(name) \
    static void init_##name##_config(void); \
    static pthread_once_t name##_guard = PTHREAD_ONCE_INIT /* no semicolon */

  #define XNN_INIT_ONCE(name) \
    pthread_once(&name##_guard, &init_##name##_config) /* no semicolon */

#endif
