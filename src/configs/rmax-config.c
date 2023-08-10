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
#include <xnnpack/rmax.h>


static struct xnn_rmax_config f16_rmax_config = {0};
static struct xnn_rmax_config f32_rmax_config = {0};
static struct xnn_rmax_config u8_rmax_config = {0};

#if XNN_PLATFORM_WINDOWS
  static INIT_ONCE init_guard_f16_rmax = INIT_ONCE_STATIC_INIT;
  static INIT_ONCE init_guard_f32_rmax = INIT_ONCE_STATIC_INIT;
  static INIT_ONCE init_guard_u8_rmax = INIT_ONCE_STATIC_INIT;
#else
  static pthread_once_t init_guard_f16_rmax = PTHREAD_ONCE_INIT;
  static pthread_once_t init_guard_f32_rmax = PTHREAD_ONCE_INIT;
  static pthread_once_t init_guard_u8_rmax = PTHREAD_ONCE_INIT;
#endif

static void init_f16_rmax_config(void) {
  #if XNN_ARCH_ARM && XNN_ENABLE_ARM_FP16_VECTOR && XNN_ENABLE_ARM_FP16_SCALAR
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon_fp16_arith) {
      f16_rmax_config.ukernel = (xnn_rmax_ukernel_fn) xnn_f16_rmax_ukernel__neonfp16arith;
    }
  #elif XNN_ARCH_ARM64 && XNN_ENABLE_ARM_FP16_VECTOR
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon_fp16_arith) {
      f16_rmax_config.ukernel = (xnn_rmax_ukernel_fn) xnn_f16_rmax_ukernel__neonfp16arith;
    }
  #elif (XNN_ARCH_X86 || XNN_ARCH_X86_64) && !XNN_PLATFORM_MOBILE
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_x86_avx2) {
      f16_rmax_config.ukernel = (xnn_rmax_ukernel_fn) xnn_f16_rmax_ukernel__f16c;
    }
  #endif
}

static void init_f32_rmax_config(void) {
  #if XNN_ARCH_ARM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon) {
      f32_rmax_config.ukernel = (xnn_rmax_ukernel_fn) xnn_f32_rmax_ukernel__neon;
    } else if (!XNN_PLATFORM_MOBILE) {
      f32_rmax_config.ukernel = (xnn_rmax_ukernel_fn) xnn_f32_rmax_ukernel__scalar;
    }
  #elif XNN_ARCH_ARM64
    f32_rmax_config.ukernel = (xnn_rmax_ukernel_fn) xnn_f32_rmax_ukernel__neon;
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    f32_rmax_config.ukernel = (xnn_rmax_ukernel_fn) xnn_f32_rmax_ukernel__sse;
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->is_x86) {
      f32_rmax_config.ukernel = (xnn_rmax_ukernel_fn) xnn_f32_rmax_ukernel__wasmsimd_x86;
    } else {
      f32_rmax_config.ukernel = (xnn_rmax_ukernel_fn) xnn_f32_rmax_ukernel__wasmsimd_arm;
    }
  #elif XNN_ARCH_WASM
    f32_rmax_config.ukernel = (xnn_rmax_ukernel_fn) xnn_f32_rmax_ukernel__scalar;
  #elif XNN_ARCH_RISCV
    f32_rmax_config.ukernel = (xnn_rmax_ukernel_fn) xnn_f32_rmax_ukernel__scalar;
  #endif
}

static void init_u8_rmax_config(void) {
  #if XNN_ARCH_ARM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon) {
      u8_rmax_config.ukernel = (xnn_rmax_ukernel_fn) xnn_u8_rmax_ukernel__neon;
    } else if (!XNN_PLATFORM_MOBILE) {
      u8_rmax_config.ukernel = (xnn_rmax_ukernel_fn) xnn_u8_rmax_ukernel__scalar;
    }
  #elif XNN_ARCH_ARM64
    u8_rmax_config.ukernel = (xnn_rmax_ukernel_fn) xnn_u8_rmax_ukernel__neon;
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    u8_rmax_config.ukernel = (xnn_rmax_ukernel_fn) xnn_u8_rmax_ukernel__sse2;
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    u8_rmax_config.ukernel = (xnn_rmax_ukernel_fn) xnn_u8_rmax_ukernel__scalar;
  #elif XNN_ARCH_WASM
    u8_rmax_config.ukernel = (xnn_rmax_ukernel_fn) xnn_u8_rmax_ukernel__scalar;
  #elif XNN_ARCH_RISCV
    u8_rmax_config.ukernel = (xnn_rmax_ukernel_fn) xnn_u8_rmax_ukernel__scalar;
  #endif
}

#if XNN_PLATFORM_WINDOWS
  static BOOL CALLBACK init_f16_rmax_config_windows(PINIT_ONCE init_once, PVOID parameter, PVOID* context) {
    init_f16_rmax_config();
    return TRUE;
  }

  static BOOL CALLBACK init_f32_rmax_config_windows(PINIT_ONCE init_once, PVOID parameter, PVOID* context) {
    init_f32_rmax_config();
    return TRUE;
  }

  static BOOL CALLBACK init_u8_rmax_config_windows(PINIT_ONCE init_once, PVOID parameter, PVOID* context) {
    init_u8_rmax_config();
    return TRUE;
  }
#endif

const struct xnn_rmax_config* xnn_init_f16_rmax_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL || !xnn_is_f16_compatible_config(hardware_config)) {
    return NULL;
  }
  #if XNN_PLATFORM_WINDOWS
    InitOnceExecuteOnce(&init_guard_f16_rmax, &init_f16_rmax_config_windows, NULL, NULL);
  #else
    pthread_once(&init_guard_f16_rmax, &init_f16_rmax_config);
  #endif
  return &f16_rmax_config;
}

const struct xnn_rmax_config* xnn_init_f32_rmax_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  #if XNN_PLATFORM_WINDOWS
    InitOnceExecuteOnce(&init_guard_f32_rmax, &init_f32_rmax_config_windows, NULL, NULL);
  #else
    pthread_once(&init_guard_f32_rmax, &init_f32_rmax_config);
  #endif
  return &f32_rmax_config;
}

const struct xnn_rmax_config* xnn_init_u8_rmax_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  #if XNN_PLATFORM_WINDOWS
    InitOnceExecuteOnce(&init_guard_u8_rmax, &init_u8_rmax_config_windows, NULL, NULL);
  #else
    pthread_once(&init_guard_u8_rmax, &init_u8_rmax_config);
  #endif
  return &u8_rmax_config;
}
