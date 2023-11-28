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
#include <xnnpack/prelu.h>


static struct xnn_prelu_config f16_prelu_config = {0};
static struct xnn_prelu_config f32_prelu_config = {0};

#if XNN_PLATFORM_WINDOWS
  static INIT_ONCE init_guard_f16_prelu = INIT_ONCE_STATIC_INIT;
  static INIT_ONCE init_guard_f32_prelu = INIT_ONCE_STATIC_INIT;
#else
  static pthread_once_t init_guard_f16_prelu = PTHREAD_ONCE_INIT;
  static pthread_once_t init_guard_f32_prelu = PTHREAD_ONCE_INIT;
#endif

static void init_f16_prelu_config(void) {
  #if XNN_ARCH_ARM && XNN_ENABLE_ARM_FP16_VECTOR && XNN_ENABLE_ARM_FP16_SCALAR
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon_fp16_arith) {
      f16_prelu_config.ukernel = (xnn_prelu_ukernel_fn) xnn_f16_prelu_ukernel__neonfp16arith_2x16;
      f16_prelu_config.row_tile = 2;
      f16_prelu_config.channel_tile = 16;
    }
  #elif XNN_ARCH_ARM64 && XNN_ENABLE_ARM_FP16_VECTOR
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon_fp16_arith) {
      f16_prelu_config.ukernel = (xnn_prelu_ukernel_fn) xnn_f16_prelu_ukernel__neonfp16arith_2x16;
      f16_prelu_config.row_tile = 2;
      f16_prelu_config.channel_tile = 16;
    }
  #elif (XNN_ARCH_X86 || XNN_ARCH_X86_64) && !XNN_PLATFORM_MOBILE
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_x86_avx2) {
      f16_prelu_config.ukernel = (xnn_prelu_ukernel_fn) xnn_f16_prelu_ukernel__f16c_2x16;
      f16_prelu_config.row_tile = 2;
      f16_prelu_config.channel_tile = 16;
    }
  #endif
}

static void init_f32_prelu_config(void) {
  #if XNN_ARCH_ARM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon) {
      f32_prelu_config.ukernel = (xnn_prelu_ukernel_fn) xnn_f32_prelu_ukernel__neon_2x8;
      f32_prelu_config.row_tile = 2;
      f32_prelu_config.channel_tile = 8;
    } else if (!XNN_PLATFORM_MOBILE) {
      f32_prelu_config.ukernel = (xnn_prelu_ukernel_fn) xnn_f32_prelu_ukernel__scalar_2x4;
      f32_prelu_config.row_tile = 4;
      f32_prelu_config.channel_tile = 4;
    }
  #elif XNN_ARCH_ARM64
    f32_prelu_config.ukernel = (xnn_prelu_ukernel_fn) xnn_f32_prelu_ukernel__neon_2x8;
    f32_prelu_config.row_tile = 2;
    f32_prelu_config.channel_tile = 8;
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (!XNN_PLATFORM_MOBILE && hardware_config->use_x86_avx512f) {
      f32_prelu_config.ukernel = (xnn_prelu_ukernel_fn) xnn_f32_prelu_ukernel__avx512f_2x16;
      f32_prelu_config.row_tile = 2;
      f32_prelu_config.channel_tile = 16;
    } else if (hardware_config->use_x86_avx) {
      f32_prelu_config.ukernel = (xnn_prelu_ukernel_fn) xnn_f32_prelu_ukernel__avx_2x16;
      f32_prelu_config.row_tile = 2;
      f32_prelu_config.channel_tile = 16;
    } else if (hardware_config->use_x86_sse4_1) {
      f32_prelu_config.ukernel = (xnn_prelu_ukernel_fn) xnn_f32_prelu_ukernel__sse41_2x8;
      f32_prelu_config.row_tile = 2;
      f32_prelu_config.channel_tile = 8;
    } else {
      f32_prelu_config.ukernel = (xnn_prelu_ukernel_fn) xnn_f32_prelu_ukernel__sse2_2x8;
      f32_prelu_config.row_tile = 2;
      f32_prelu_config.channel_tile = 8;
    }
  #elif XNN_ARCH_WASMRELAXEDSIMD
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->is_x86) {
      f32_prelu_config.ukernel = (xnn_prelu_ukernel_fn) xnn_f32_prelu_ukernel__wasmrelaxedsimd_iminmax_2x4;
      f32_prelu_config.row_tile = 2;
      f32_prelu_config.channel_tile = 4;
    } else {
      f32_prelu_config.ukernel = (xnn_prelu_ukernel_fn) xnn_f32_prelu_ukernel__wasmrelaxedsimd_laneselect_2x4;
      f32_prelu_config.row_tile = 2;
      f32_prelu_config.channel_tile = 4;
    }
  #elif XNN_ARCH_WASMSIMD
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->is_x86) {
      f32_prelu_config.ukernel = (xnn_prelu_ukernel_fn) xnn_f32_prelu_ukernel__wasmsimd_iminmax_2x8;
      f32_prelu_config.row_tile = 2;
      f32_prelu_config.channel_tile = 8;
    } else {
      f32_prelu_config.ukernel = (xnn_prelu_ukernel_fn) xnn_f32_prelu_ukernel__wasmsimd_laneselect_2x8;
      f32_prelu_config.row_tile = 2;
      f32_prelu_config.channel_tile = 8;
    }
  #elif XNN_ARCH_WASM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->is_x86) {
      f32_prelu_config.ukernel = (xnn_prelu_ukernel_fn) xnn_f32_prelu_ukernel__scalar_2x4;
      f32_prelu_config.row_tile = 2;
      f32_prelu_config.channel_tile = 4;
    } else {
      f32_prelu_config.ukernel = (xnn_prelu_ukernel_fn) xnn_f32_prelu_ukernel__wasm_2x4;
      f32_prelu_config.row_tile = 2;
      f32_prelu_config.channel_tile = 4;
    }
  #elif XNN_ARCH_RISCV
    f32_prelu_config.ukernel = (xnn_prelu_ukernel_fn) xnn_f32_prelu_ukernel__scalar_2x4;
    f32_prelu_config.row_tile = 4;
    f32_prelu_config.channel_tile = 4;
  #elif XNN_ARCH_PPC64
    f32_prelu_config.ukernel = (xnn_prelu_ukernel_fn) xnn_f32_prelu_ukernel__scalar_2x4;
    f32_prelu_config.row_tile = 4;
    f32_prelu_config.channel_tile = 4;
  #endif
}

#if XNN_PLATFORM_WINDOWS
  static BOOL CALLBACK init_f16_prelu_config_windows(PINIT_ONCE init_once, PVOID parameter, PVOID* context) {
    init_f16_prelu_config();
    return TRUE;
  }

  static BOOL CALLBACK init_f32_prelu_config_windows(PINIT_ONCE init_once, PVOID parameter, PVOID* context) {
    init_f32_prelu_config();
    return TRUE;
  }
#endif

const struct xnn_prelu_config* xnn_init_f16_prelu_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL || !xnn_is_f16_compatible_config(hardware_config)) {
    return NULL;
  }
  #if XNN_PLATFORM_WINDOWS
    InitOnceExecuteOnce(&init_guard_f16_prelu, &init_f16_prelu_config_windows, NULL, NULL);
  #else
    pthread_once(&init_guard_f16_prelu, &init_f16_prelu_config);
  #endif
  return &f16_prelu_config;
}

const struct xnn_prelu_config* xnn_init_f32_prelu_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  #if XNN_PLATFORM_WINDOWS
    InitOnceExecuteOnce(&init_guard_f32_prelu, &init_f32_prelu_config_windows, NULL, NULL);
  #else
    pthread_once(&init_guard_f32_prelu, &init_f32_prelu_config);
  #endif
  return &f32_prelu_config;
}
