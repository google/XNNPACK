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
#include <xnnpack/pavgpool.h>


static struct xnn_pavgpool_config f16_pavgpool_config = {0};
static struct xnn_pavgpool_config f32_pavgpool_config = {0};

#if XNN_PLATFORM_WINDOWS
  static INIT_ONCE init_guard_f16_pavgpool = INIT_ONCE_STATIC_INIT;
  static INIT_ONCE init_guard_f32_pavgpool = INIT_ONCE_STATIC_INIT;
#else
  static pthread_once_t init_guard_f16_pavgpool = PTHREAD_ONCE_INIT;
  static pthread_once_t init_guard_f32_pavgpool = PTHREAD_ONCE_INIT;
#endif

static void init_f16_pavgpool_config(void) {
  #if XNN_ARCH_ARM && XNN_ENABLE_ARM_FP16_VECTOR && XNN_ENABLE_ARM_FP16_SCALAR
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon_fp16_arith) {
      f16_pavgpool_config.unipass = (xnn_pavgpool_unipass_ukernel_fn) xnn_f16_pavgpool_minmax_ukernel_9x__neonfp16arith_c8;
      f16_pavgpool_config.multipass = (xnn_pavgpool_multipass_ukernel_fn) xnn_f16_pavgpool_minmax_ukernel_9p8x__neonfp16arith_c8;
      f16_pavgpool_config.init.f16 = xnn_init_f16_minmax_fp16arith_params;
      f16_pavgpool_config.primary_tile = 9;
      f16_pavgpool_config.incremental_tile = 8;
      f16_pavgpool_config.channel_tile = 8;
    }
  #elif XNN_ARCH_ARM64 && XNN_ENABLE_ARM_FP16_VECTOR
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon_fp16_arith) {
      f16_pavgpool_config.unipass = (xnn_pavgpool_unipass_ukernel_fn) xnn_f16_pavgpool_minmax_ukernel_9x__neonfp16arith_c8;
      f16_pavgpool_config.multipass = (xnn_pavgpool_multipass_ukernel_fn) xnn_f16_pavgpool_minmax_ukernel_9p8x__neonfp16arith_c8;
      f16_pavgpool_config.init.f16 = xnn_init_f16_minmax_fp16arith_params;
      f16_pavgpool_config.primary_tile = 9;
      f16_pavgpool_config.incremental_tile = 8;
      f16_pavgpool_config.channel_tile = 8;
    }
  #elif (XNN_ARCH_X86 || XNN_ARCH_X86_64) && !XNN_PLATFORM_MOBILE
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_x86_avx2) {
      f16_pavgpool_config.unipass = (xnn_pavgpool_unipass_ukernel_fn) xnn_f16_pavgpool_minmax_ukernel_9x__avx2_c8;
      f16_pavgpool_config.multipass = (xnn_pavgpool_multipass_ukernel_fn) xnn_f16_pavgpool_minmax_ukernel_9p8x__avx2_c8;
      f16_pavgpool_config.init.f16 = xnn_init_f16_minmax_avx_params;
      f16_pavgpool_config.primary_tile = 9;
      f16_pavgpool_config.incremental_tile = 8;
      f16_pavgpool_config.channel_tile = 8;
    }
  #endif
}

static void init_f32_pavgpool_config(void) {
  #if XNN_ARCH_ARM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon) {
      f32_pavgpool_config.unipass = (xnn_pavgpool_unipass_ukernel_fn) xnn_f32_pavgpool_minmax_ukernel_9x__neon_c4;
      f32_pavgpool_config.multipass = (xnn_pavgpool_multipass_ukernel_fn) xnn_f32_pavgpool_minmax_ukernel_9p8x__neon_c4;
      f32_pavgpool_config.init.f32 = xnn_init_f32_minmax_scalar_params;
      f32_pavgpool_config.primary_tile = 9;
      f32_pavgpool_config.incremental_tile = 8;
      f32_pavgpool_config.channel_tile = 4;
    } else if (!XNN_PLATFORM_MOBILE) {
      f32_pavgpool_config.unipass = (xnn_pavgpool_unipass_ukernel_fn) xnn_f32_pavgpool_minmax_ukernel_9x__scalar_c1;
      f32_pavgpool_config.multipass = (xnn_pavgpool_multipass_ukernel_fn) xnn_f32_pavgpool_minmax_ukernel_9p8x__scalar_c1;
      f32_pavgpool_config.init.f32 = xnn_init_f32_minmax_scalar_params;
      f32_pavgpool_config.primary_tile = 9;
      f32_pavgpool_config.incremental_tile = 8;
      f32_pavgpool_config.channel_tile = 1;
    }
  #elif XNN_ARCH_ARM64
    f32_pavgpool_config.unipass = (xnn_pavgpool_unipass_ukernel_fn) xnn_f32_pavgpool_minmax_ukernel_9x__neon_c4;
    f32_pavgpool_config.multipass = (xnn_pavgpool_multipass_ukernel_fn) xnn_f32_pavgpool_minmax_ukernel_9p8x__neon_c4;
    f32_pavgpool_config.init.f32 = xnn_init_f32_minmax_scalar_params;
    f32_pavgpool_config.primary_tile = 9;
    f32_pavgpool_config.incremental_tile = 8;
    f32_pavgpool_config.channel_tile = 4;
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    f32_pavgpool_config.unipass = (xnn_pavgpool_unipass_ukernel_fn) xnn_f32_pavgpool_minmax_ukernel_9x__sse_c4;
    f32_pavgpool_config.multipass = (xnn_pavgpool_multipass_ukernel_fn) xnn_f32_pavgpool_minmax_ukernel_9p8x__sse_c4;
    f32_pavgpool_config.init.f32 = xnn_init_f32_minmax_sse_params;
    f32_pavgpool_config.primary_tile = 9;
    f32_pavgpool_config.incremental_tile = 8;
    f32_pavgpool_config.channel_tile = 4;
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->is_x86) {
      f32_pavgpool_config.unipass = (xnn_pavgpool_unipass_ukernel_fn) xnn_f32_pavgpool_minmax_ukernel_9x__wasmsimd_x86_c4;
      f32_pavgpool_config.multipass = (xnn_pavgpool_multipass_ukernel_fn) xnn_f32_pavgpool_minmax_ukernel_9p8x__wasmsimd_x86_c4;
      f32_pavgpool_config.init.f32 = xnn_init_f32_minmax_wasmsimd_params;
      f32_pavgpool_config.primary_tile = 9;
      f32_pavgpool_config.incremental_tile = 8;
      f32_pavgpool_config.channel_tile = 4;
    } else {
      f32_pavgpool_config.unipass = (xnn_pavgpool_unipass_ukernel_fn) xnn_f32_pavgpool_minmax_ukernel_9x__wasmsimd_arm_c4;
      f32_pavgpool_config.multipass = (xnn_pavgpool_multipass_ukernel_fn) xnn_f32_pavgpool_minmax_ukernel_9p8x__wasmsimd_arm_c4;
      f32_pavgpool_config.init.f32 = xnn_init_f32_minmax_wasmsimd_params;
      f32_pavgpool_config.primary_tile = 9;
      f32_pavgpool_config.incremental_tile = 8;
      f32_pavgpool_config.channel_tile = 4;
    }
  #elif XNN_ARCH_WASM
    f32_pavgpool_config.unipass = (xnn_pavgpool_unipass_ukernel_fn) xnn_f32_pavgpool_minmax_ukernel_9x__wasm_c1;
    f32_pavgpool_config.multipass = (xnn_pavgpool_multipass_ukernel_fn) xnn_f32_pavgpool_minmax_ukernel_9p8x__wasm_c1;
    f32_pavgpool_config.init.f32 = xnn_init_f32_minmax_scalar_params;
    f32_pavgpool_config.primary_tile = 9;
    f32_pavgpool_config.incremental_tile = 8;
    f32_pavgpool_config.channel_tile = 1;
  #else
    f32_pavgpool_config.unipass = (xnn_pavgpool_unipass_ukernel_fn) xnn_f32_pavgpool_minmax_ukernel_9x__scalar_c1;
    f32_pavgpool_config.multipass = (xnn_pavgpool_multipass_ukernel_fn) xnn_f32_pavgpool_minmax_ukernel_9p8x__scalar_c1;
    f32_pavgpool_config.init.f32 = xnn_init_f32_minmax_scalar_params;
    f32_pavgpool_config.primary_tile = 9;
    f32_pavgpool_config.incremental_tile = 8;
    f32_pavgpool_config.channel_tile = 1;
  #endif
}

#if XNN_PLATFORM_WINDOWS
  static BOOL CALLBACK init_f16_pavgpool_config_windows(PINIT_ONCE init_once, PVOID parameter, PVOID* context) {
    init_f16_pavgpool_config();
    return TRUE;
  }

  static BOOL CALLBACK init_f32_pavgpool_config_windows(PINIT_ONCE init_once, PVOID parameter, PVOID* context) {
    init_f32_pavgpool_config();
    return TRUE;
  }
#endif

const struct xnn_pavgpool_config* xnn_init_f16_pavgpool_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL || !xnn_is_f16_compatible_config(hardware_config)) {
    return NULL;
  }
  #if XNN_PLATFORM_WINDOWS
    InitOnceExecuteOnce(&init_guard_f16_pavgpool, &init_f16_pavgpool_config_windows, NULL, NULL);
  #else
    pthread_once(&init_guard_f16_pavgpool, &init_f16_pavgpool_config);
  #endif
  return &f16_pavgpool_config;
}

const struct xnn_pavgpool_config* xnn_init_f32_pavgpool_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  #if XNN_PLATFORM_WINDOWS
    InitOnceExecuteOnce(&init_guard_f32_pavgpool, &init_f32_pavgpool_config_windows, NULL, NULL);
  #else
    pthread_once(&init_guard_f32_pavgpool, &init_f32_pavgpool_config);
  #endif
  return &f32_pavgpool_config;
}
