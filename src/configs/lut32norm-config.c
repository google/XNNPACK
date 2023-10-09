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
#include <xnnpack/lut.h>


static struct xnn_lut32norm_config u8_lut32norm_config = {0};

#if XNN_PLATFORM_WINDOWS
  static INIT_ONCE init_guard_u8_lut32norm = INIT_ONCE_STATIC_INIT;
#else
  static pthread_once_t init_guard_u8_lut32norm = PTHREAD_ONCE_INIT;
#endif

static void init_u8_lut32norm_config(void) {
  #if XNN_ARCH_ARM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon) {
      u8_lut32norm_config.lut32norm = xnn_u8_lut32norm_ukernel__scalar;
    } else if (!XNN_PLATFORM_MOBILE) {
      u8_lut32norm_config.lut32norm = xnn_u8_lut32norm_ukernel__scalar;
    }
  #elif XNN_ARCH_ARM64
    u8_lut32norm_config.lut32norm = xnn_u8_lut32norm_ukernel__scalar;
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    u8_lut32norm_config.lut32norm = xnn_u8_lut32norm_ukernel__scalar;
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    u8_lut32norm_config.lut32norm = xnn_u8_lut32norm_ukernel__scalar;
  #elif XNN_ARCH_WASM
    u8_lut32norm_config.lut32norm = xnn_u8_lut32norm_ukernel__scalar;
  #elif XNN_ARCH_RISCV
    u8_lut32norm_config.lut32norm = xnn_u8_lut32norm_ukernel__scalar;
  #elif XNN_ARCH_PPC64
    u8_lut32norm_config.lut32norm = xnn_u8_lut32norm_ukernel__scalar;
  #endif
}

#if XNN_PLATFORM_WINDOWS
  static BOOL CALLBACK init_u8_lut32norm_config_windows(PINIT_ONCE init_once, PVOID parameter, PVOID* context) {
    init_u8_lut32norm_config();
    return TRUE;
  }
#endif

const struct xnn_lut32norm_config* xnn_init_u8_lut32norm_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  #if XNN_PLATFORM_WINDOWS
    InitOnceExecuteOnce(&init_guard_u8_lut32norm, &init_u8_lut32norm_config_windows, NULL, NULL);
  #else
    pthread_once(&init_guard_u8_lut32norm, &init_u8_lut32norm_config);
  #endif
  return &u8_lut32norm_config;
}
