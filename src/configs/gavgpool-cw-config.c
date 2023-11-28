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
#include <xnnpack/gavgpool.h>


static struct xnn_gavgpool_cw_config f16_gavgpool_cw_config = {0};
static struct xnn_gavgpool_cw_config f32_gavgpool_cw_config = {0};

#if XNN_PLATFORM_WINDOWS
  static INIT_ONCE init_guard_f16_gavgpool_cw = INIT_ONCE_STATIC_INIT;
  static INIT_ONCE init_guard_f32_gavgpool_cw = INIT_ONCE_STATIC_INIT;
#else
  static pthread_once_t init_guard_f16_gavgpool_cw = PTHREAD_ONCE_INIT;
  static pthread_once_t init_guard_f32_gavgpool_cw = PTHREAD_ONCE_INIT;
#endif

static void init_f16_gavgpool_cw_config(void) {
  #if XNN_ARCH_ARM && XNN_ENABLE_ARM_FP16_VECTOR && XNN_ENABLE_ARM_FP16_SCALAR
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon_fp16_arith) {
      f16_gavgpool_cw_config.ukernel = (xnn_gavgpool_cw_ukernel_fn) xnn_f16_gavgpool_cw_ukernel__neonfp16arith_u8;
      f16_gavgpool_cw_config.init.f16 = xnn_init_f16_gavgpool_neonfp16arith_params;
      f16_gavgpool_cw_config.update.f16 = xnn_update_f16_gavgpool_neonfp16arith_params;
      f16_gavgpool_cw_config.pixel_tile = 8;
    }
  #elif XNN_ARCH_ARM64 && XNN_ENABLE_ARM_FP16_VECTOR
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon_fp16_arith) {
      f16_gavgpool_cw_config.ukernel = (xnn_gavgpool_cw_ukernel_fn) xnn_f16_gavgpool_cw_ukernel__neonfp16arith_u8;
      f16_gavgpool_cw_config.init.f16 = xnn_init_f16_gavgpool_neonfp16arith_params;
      f16_gavgpool_cw_config.update.f16 = xnn_update_f16_gavgpool_neonfp16arith_params;
      f16_gavgpool_cw_config.pixel_tile = 8;
    }
  #endif
}

static void init_f32_gavgpool_cw_config(void) {
  #if XNN_ARCH_ARM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon) {
      f32_gavgpool_cw_config.ukernel = (xnn_gavgpool_cw_ukernel_fn) xnn_f32_gavgpool_cw_ukernel__neon_u4;
      f32_gavgpool_cw_config.pixel_tile = 4;
    } else if (!XNN_PLATFORM_MOBILE) {
      f32_gavgpool_cw_config.ukernel = (xnn_gavgpool_cw_ukernel_fn) xnn_f32_gavgpool_cw_ukernel__scalar_u1;
      f32_gavgpool_cw_config.pixel_tile = 1;
    }
    f32_gavgpool_cw_config.init.f32 = xnn_init_f32_gavgpool_neon_params;
    f32_gavgpool_cw_config.update.f32 = xnn_update_f32_gavgpool_params;
  #elif XNN_ARCH_ARM64
    f32_gavgpool_cw_config.ukernel = (xnn_gavgpool_cw_ukernel_fn) xnn_f32_gavgpool_cw_ukernel__neon_u4;
    f32_gavgpool_cw_config.pixel_tile = 4;
    f32_gavgpool_cw_config.init.f32 = xnn_init_f32_gavgpool_neon_params;
    f32_gavgpool_cw_config.update.f32 = xnn_update_f32_gavgpool_params;
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    f32_gavgpool_cw_config.ukernel = (xnn_gavgpool_cw_ukernel_fn) xnn_f32_gavgpool_cw_ukernel__sse_u4;
    f32_gavgpool_cw_config.pixel_tile = 4;
    f32_gavgpool_cw_config.init.f32 = xnn_init_f32_gavgpool_sse_params;
    f32_gavgpool_cw_config.update.f32 = xnn_update_f32_gavgpool_params;
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->is_x86) {
      f32_gavgpool_cw_config.ukernel = (xnn_gavgpool_cw_ukernel_fn) xnn_f32_gavgpool_cw_ukernel__wasmsimd_x86_u4;
      f32_gavgpool_cw_config.pixel_tile = 4;
    } else {
      f32_gavgpool_cw_config.ukernel = (xnn_gavgpool_cw_ukernel_fn) xnn_f32_gavgpool_cw_ukernel__wasmsimd_arm_u4;
      f32_gavgpool_cw_config.pixel_tile = 4;
    }
    f32_gavgpool_cw_config.init.f32 = xnn_init_f32_gavgpool_scalar_params;
    f32_gavgpool_cw_config.update.f32 = xnn_update_f32_gavgpool_params;
  #elif XNN_ARCH_WASM
    f32_gavgpool_cw_config.ukernel = (xnn_gavgpool_cw_ukernel_fn) xnn_f32_gavgpool_cw_ukernel__scalar_u1;
    f32_gavgpool_cw_config.pixel_tile = 1;
    f32_gavgpool_cw_config.init.f32 = xnn_init_f32_gavgpool_scalar_params;
    f32_gavgpool_cw_config.update.f32 = xnn_update_f32_gavgpool_params;
  #elif XNN_ARCH_RISCV
    f32_gavgpool_cw_config.ukernel = (xnn_gavgpool_cw_ukernel_fn) xnn_f32_gavgpool_cw_ukernel__scalar_u1;
    f32_gavgpool_cw_config.pixel_tile = 1;
    f32_gavgpool_cw_config.init.f32 = xnn_init_f32_gavgpool_scalar_params;
    f32_gavgpool_cw_config.update.f32 = xnn_update_f32_gavgpool_params;
  #elif XNN_ARCH_PPC64
    f32_gavgpool_cw_config.ukernel = (xnn_gavgpool_cw_ukernel_fn) xnn_f32_gavgpool_cw_ukernel__scalar_u1;
    f32_gavgpool_cw_config.pixel_tile = 1;
    f32_gavgpool_cw_config.init.f32 = xnn_init_f32_gavgpool_scalar_params;
    f32_gavgpool_cw_config.update.f32 = xnn_update_f32_gavgpool_params;
  #endif
}

#if XNN_PLATFORM_WINDOWS
  static BOOL CALLBACK init_f16_gavgpool_cw_config_windows(PINIT_ONCE init_once, PVOID parameter, PVOID* context) {
    init_f16_gavgpool_cw_config();
    return TRUE;
  }

  static BOOL CALLBACK init_f32_gavgpool_cw_config_windows(PINIT_ONCE init_once, PVOID parameter, PVOID* context) {
    init_f32_gavgpool_cw_config();
    return TRUE;
  }
#endif

const struct xnn_gavgpool_cw_config* xnn_init_f16_gavgpool_cw_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL || !xnn_is_f16_chw_compatible_config(hardware_config)) {
    return NULL;
  }
  #if XNN_PLATFORM_WINDOWS
    InitOnceExecuteOnce(&init_guard_f16_gavgpool_cw, &init_f16_gavgpool_cw_config_windows, NULL, NULL);
  #else
    pthread_once(&init_guard_f16_gavgpool_cw, &init_f16_gavgpool_cw_config);
  #endif
  return &f16_gavgpool_cw_config;
}

const struct xnn_gavgpool_cw_config* xnn_init_f32_gavgpool_cw_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL || !xnn_is_chw_compatible_config(hardware_config)) {
    return NULL;
  }
  #if XNN_PLATFORM_WINDOWS
    InitOnceExecuteOnce(&init_guard_f32_gavgpool_cw, &init_f32_gavgpool_cw_config_windows, NULL, NULL);
  #else
    pthread_once(&init_guard_f32_gavgpool_cw, &init_f32_gavgpool_cw_config);
  #endif
  return &f32_gavgpool_cw_config;
}
