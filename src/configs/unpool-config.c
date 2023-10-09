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
#include <xnnpack/unpool.h>


static struct xnn_unpool_config x32_unpool_config = {0};

#if XNN_PLATFORM_WINDOWS
  static INIT_ONCE init_guard_x32_unpool = INIT_ONCE_STATIC_INIT;
#else
  static pthread_once_t init_guard_x32_unpool = PTHREAD_ONCE_INIT;
#endif

static void init_x32_unpool_config(void) {
  #if XNN_ARCH_ARM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon) {
      x32_unpool_config.unpool = (xnn_unpool_ukernel_fn) xnn_x32_unpool_ukernel__neon;
    } else if (!XNN_PLATFORM_MOBILE) {
      x32_unpool_config.unpool = (xnn_unpool_ukernel_fn) xnn_x32_unpool_ukernel__scalar;
    }
  #elif XNN_ARCH_ARM64
    x32_unpool_config.unpool = (xnn_unpool_ukernel_fn) xnn_x32_unpool_ukernel__neon;
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    x32_unpool_config.unpool = (xnn_unpool_ukernel_fn) xnn_x32_unpool_ukernel__sse2;
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    x32_unpool_config.unpool = (xnn_unpool_ukernel_fn) xnn_x32_unpool_ukernel__wasmsimd;
  #elif XNN_ARCH_WASM
    x32_unpool_config.unpool = (xnn_unpool_ukernel_fn) xnn_x32_unpool_ukernel__scalar;
  #elif XNN_ARCH_RISCV
    x32_unpool_config.unpool = (xnn_unpool_ukernel_fn) xnn_x32_unpool_ukernel__scalar;
  #elif XNN_ARCH_PPC64
    x32_unpool_config.unpool = (xnn_unpool_ukernel_fn) xnn_x32_unpool_ukernel__scalar;
  #endif
}

#if XNN_PLATFORM_WINDOWS
  static BOOL CALLBACK init_x32_unpool_config_windows(PINIT_ONCE init_once, PVOID parameter, PVOID* context) {
    init_x32_unpool_config();
    return TRUE;
  }
#endif

const struct xnn_unpool_config* xnn_init_x32_unpool_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  #if XNN_PLATFORM_WINDOWS
    InitOnceExecuteOnce(&init_guard_x32_unpool, &init_x32_unpool_config_windows, NULL, NULL);
  #else
    pthread_once(&init_guard_x32_unpool, &init_x32_unpool_config);
  #endif
  return &x32_unpool_config;
}
