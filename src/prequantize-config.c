// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>

#ifdef _WIN32
  #include <windows.h>
#else
  #include <pthread.h>
#endif

#include <xnnpack/common.h>
#include <xnnpack/config.h>
#include <xnnpack/microparams-init.h>
#include <xnnpack/prequantization.h>

static struct xnn_prequantize_config f32_qd8_prequantize_config = {0};

#if XNN_PLATFORM_WINDOWS
  static INIT_ONCE init_guard_f32_qd8_prequantize = INIT_ONCE_STATIC_INIT;
#else
  static pthread_once_t init_guard_f32_qd8_prequantize = PTHREAD_ONCE_INIT;
#endif

static void init_f32_qd8_prequantize_config(void) {
  f32_qd8_prequantize_config = (struct xnn_prequantize_config) {
    .ukernel = (xnn_f32_prequantize_ukernel_fn) xnn_f32_qd8_asymmetric_quantization_params,
  };
}

#if XNN_PLATFORM_WINDOWS
  static BOOL CALLBACK init_f32_qd8_prequantization_config_windows(PINIT_ONCE init_once, PVOID parameter, PVOID* context) {
    init_f32_qd8_prequantize_config();
    return TRUE;
  }
#endif

const struct xnn_prequantize_config* xnn_init_f32_qd8_prequantize_config() {
  #if XNN_PLATFORM_WINDOWS
    InitOnceExecuteOnce(&init_guard_f32_qd8_prequantize, &init_f32_qd8_prequantization_config_windows, NULL, NULL);
  #else
    pthread_once(&init_guard_f32_qd8_prequantize, &init_f32_qd8_prequantize_config);
  #endif
  return &f32_qd8_prequantize_config;
}
