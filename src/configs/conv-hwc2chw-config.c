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
#include <xnnpack/conv.h>


static struct xnn_conv_hwc2chw_config f16_conv_hwc2chw_3x3c3s2_config = {0};
static struct xnn_conv_hwc2chw_config f32_conv_hwc2chw_3x3c3s2_config = {0};

#if XNN_PLATFORM_WINDOWS
  static INIT_ONCE init_guard_f16_conv_hwc2chw_3x3c3s2 = INIT_ONCE_STATIC_INIT;
  static INIT_ONCE init_guard_f32_conv_hwc2chw_3x3c3s2 = INIT_ONCE_STATIC_INIT;
#else
  static pthread_once_t init_guard_f16_conv_hwc2chw_3x3c3s2 = PTHREAD_ONCE_INIT;
  static pthread_once_t init_guard_f32_conv_hwc2chw_3x3c3s2 = PTHREAD_ONCE_INIT;
#endif

static void init_f16_conv_hwc2chw_3x3c3s2_config(void) {
  #if XNN_ARCH_ARM && XNN_ENABLE_ARM_FP16_VECTOR && XNN_ENABLE_ARM_FP16_SCALAR
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon_fp16_arith) {
      f16_conv_hwc2chw_3x3c3s2_config.ukernel_with_symm_padding =
        (xnn_conv_hwc2chw_ukernel_fn) xnn_f16_conv_hwc2chw_ukernel_3x3s2p1c3x4__neonfp16arith_2x2;
      f16_conv_hwc2chw_3x3c3s2_config.init.f16 = xnn_init_f16_minmax_fp16arith_params;
      f16_conv_hwc2chw_3x3c3s2_config.output_channel_tile = 4;
      f16_conv_hwc2chw_3x3c3s2_config.output_height_tile = 2;
      f16_conv_hwc2chw_3x3c3s2_config.output_width_tile = 2;
    }
  #elif XNN_ARCH_ARM64 && XNN_ENABLE_ARM_FP16_VECTOR
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon_fp16_arith) {
      f16_conv_hwc2chw_3x3c3s2_config.ukernel_with_symm_padding =
        (xnn_conv_hwc2chw_ukernel_fn) xnn_f16_conv_hwc2chw_ukernel_3x3s2p1c3x4__neonfp16arith_2x2;
      f16_conv_hwc2chw_3x3c3s2_config.init.f16 = xnn_init_f16_minmax_fp16arith_params;
      f16_conv_hwc2chw_3x3c3s2_config.output_channel_tile = 4;
      f16_conv_hwc2chw_3x3c3s2_config.output_height_tile = 2;
      f16_conv_hwc2chw_3x3c3s2_config.output_width_tile = 2;
    }
  #endif
}

static void init_f32_conv_hwc2chw_3x3c3s2_config(void) {
  #if XNN_ARCH_ARM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon) {
      f32_conv_hwc2chw_3x3c3s2_config.ukernel_with_symm_padding =
        (xnn_conv_hwc2chw_ukernel_fn) xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x4__neon_2x2;
      f32_conv_hwc2chw_3x3c3s2_config.init.f32 = xnn_init_f32_minmax_scalar_params;
      f32_conv_hwc2chw_3x3c3s2_config.output_channel_tile = 4;
      f32_conv_hwc2chw_3x3c3s2_config.output_height_tile = 2;
      f32_conv_hwc2chw_3x3c3s2_config.output_width_tile = 2;
    } else if (!XNN_PLATFORM_MOBILE) {
      f32_conv_hwc2chw_3x3c3s2_config.ukernel_with_symm_padding =
        (xnn_conv_hwc2chw_ukernel_fn) xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x4__scalar_1x1;
      f32_conv_hwc2chw_3x3c3s2_config.init.f32 = xnn_init_f32_minmax_scalar_params;
      f32_conv_hwc2chw_3x3c3s2_config.output_channel_tile = 4;
      f32_conv_hwc2chw_3x3c3s2_config.output_height_tile = 1;
      f32_conv_hwc2chw_3x3c3s2_config.output_width_tile = 1;
    }
  #elif XNN_ARCH_ARM64
    f32_conv_hwc2chw_3x3c3s2_config.ukernel_with_symm_padding =
      (xnn_conv_hwc2chw_ukernel_fn) xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x4__aarch64_neonfma_2x2;
    f32_conv_hwc2chw_3x3c3s2_config.init.f32 = xnn_init_f32_minmax_scalar_params;
    f32_conv_hwc2chw_3x3c3s2_config.output_channel_tile = 4;
    f32_conv_hwc2chw_3x3c3s2_config.output_height_tile = 2;
    f32_conv_hwc2chw_3x3c3s2_config.output_width_tile = 2;
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    f32_conv_hwc2chw_3x3c3s2_config.ukernel_with_symm_padding =
      (xnn_conv_hwc2chw_ukernel_fn) xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x4__sse_2x2;
    f32_conv_hwc2chw_3x3c3s2_config.init.f32 = xnn_init_f32_minmax_sse_params;
    f32_conv_hwc2chw_3x3c3s2_config.output_channel_tile = 4;
    f32_conv_hwc2chw_3x3c3s2_config.output_height_tile = 2;
    f32_conv_hwc2chw_3x3c3s2_config.output_width_tile = 2;
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    f32_conv_hwc2chw_3x3c3s2_config.ukernel_with_symm_padding =
      (xnn_conv_hwc2chw_ukernel_fn) xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x4__wasmsimd_2x2;
    f32_conv_hwc2chw_3x3c3s2_config.init.f32 = xnn_init_f32_minmax_wasmsimd_params;
    f32_conv_hwc2chw_3x3c3s2_config.output_channel_tile = 4;
    f32_conv_hwc2chw_3x3c3s2_config.output_height_tile = 2;
    f32_conv_hwc2chw_3x3c3s2_config.output_width_tile = 2;
  #elif XNN_ARCH_WASM
    f32_conv_hwc2chw_3x3c3s2_config.ukernel_with_symm_padding =
      (xnn_conv_hwc2chw_ukernel_fn) xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x4__scalar_1x1;
    f32_conv_hwc2chw_3x3c3s2_config.init.f32 = xnn_init_f32_minmax_scalar_params;
    f32_conv_hwc2chw_3x3c3s2_config.output_channel_tile = 4;
    f32_conv_hwc2chw_3x3c3s2_config.output_height_tile = 1;
    f32_conv_hwc2chw_3x3c3s2_config.output_width_tile = 1;
  #elif XNN_ARCH_RISCV
    f32_conv_hwc2chw_3x3c3s2_config.ukernel_with_symm_padding =
      (xnn_conv_hwc2chw_ukernel_fn) xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x4__scalar_1x1;
    f32_conv_hwc2chw_3x3c3s2_config.init.f32 = xnn_init_f32_minmax_scalar_params;
    f32_conv_hwc2chw_3x3c3s2_config.output_channel_tile = 4;
    f32_conv_hwc2chw_3x3c3s2_config.output_height_tile = 1;
    f32_conv_hwc2chw_3x3c3s2_config.output_width_tile = 1;
  #elif XNN_ARCH_PPC64
    f32_conv_hwc2chw_3x3c3s2_config.ukernel_with_symm_padding =
      (xnn_conv_hwc2chw_ukernel_fn) xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x4__scalar_1x1;
    f32_conv_hwc2chw_3x3c3s2_config.init.f32 = xnn_init_f32_minmax_scalar_params;
    f32_conv_hwc2chw_3x3c3s2_config.output_channel_tile = 4;
    f32_conv_hwc2chw_3x3c3s2_config.output_height_tile = 1;
    f32_conv_hwc2chw_3x3c3s2_config.output_width_tile = 1;
  #endif
}

#if XNN_PLATFORM_WINDOWS
  static BOOL CALLBACK init_f16_conv_hwc2chw_3x3c3s2_config_windows(PINIT_ONCE init_once, PVOID parameter, PVOID* context) {
    init_f16_conv_hwc2chw_3x3c3s2_config();
    return TRUE;
  }

  static BOOL CALLBACK init_f32_conv_hwc2chw_3x3c3s2_config_windows(PINIT_ONCE init_once, PVOID parameter, PVOID* context) {
    init_f32_conv_hwc2chw_3x3c3s2_config();
    return TRUE;
  }
#endif

const struct xnn_conv_hwc2chw_config* xnn_init_f16_conv_hwc2chw_3x3c3s2_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL || !xnn_is_f16_chw_compatible_config(hardware_config)) {
    return NULL;
  }
  #if XNN_PLATFORM_WINDOWS
    InitOnceExecuteOnce(&init_guard_f16_conv_hwc2chw_3x3c3s2, &init_f16_conv_hwc2chw_3x3c3s2_config_windows, NULL, NULL);
  #else
    pthread_once(&init_guard_f16_conv_hwc2chw_3x3c3s2, &init_f16_conv_hwc2chw_3x3c3s2_config);
  #endif
  return &f16_conv_hwc2chw_3x3c3s2_config;
}

const struct xnn_conv_hwc2chw_config* xnn_init_f32_conv_hwc2chw_3x3c3s2_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL || !xnn_is_chw_compatible_config(hardware_config)) {
    return NULL;
  }
  #if XNN_PLATFORM_WINDOWS
    InitOnceExecuteOnce(&init_guard_f32_conv_hwc2chw_3x3c3s2, &init_f32_conv_hwc2chw_3x3c3s2_config_windows, NULL, NULL);
  #else
    pthread_once(&init_guard_f32_conv_hwc2chw_3x3c3s2, &init_f32_conv_hwc2chw_3x3c3s2_config);
  #endif
  return &f32_conv_hwc2chw_3x3c3s2_config;
}
