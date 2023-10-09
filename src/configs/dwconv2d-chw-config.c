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
#include <xnnpack/dwconv.h>


static struct xnn_dwconv2d_chw_config f16_dwconv2d_chw_config = {0};
static struct xnn_dwconv2d_chw_config f32_dwconv2d_chw_config = {0};

#if XNN_PLATFORM_WINDOWS
  static INIT_ONCE init_guard_f16_dwconv2d_chw = INIT_ONCE_STATIC_INIT;
  static INIT_ONCE init_guard_f32_dwconv2d_chw = INIT_ONCE_STATIC_INIT;
#else
  static pthread_once_t init_guard_f16_dwconv2d_chw = PTHREAD_ONCE_INIT;
  static pthread_once_t init_guard_f32_dwconv2d_chw = PTHREAD_ONCE_INIT;
#endif

static void init_f16_dwconv2d_chw_config(void) {
  #if XNN_ARCH_ARM && XNN_ENABLE_ARM_FP16_VECTOR && XNN_ENABLE_ARM_FP16_SCALAR
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon_fp16_arith) {
      f16_dwconv2d_chw_config.dwconv2d_chw_3x3.ukernel = (xnn_dwconv2d_chw_ukernel_fn) xnn_f16_dwconv2d_chw_ukernel_3x3p1__neonfp16arith_2x8;
      f16_dwconv2d_chw_config.dwconv2d_chw_3x3.init.f16 = xnn_init_f16_chw_neonfp16arith_stride1_params;
      f16_dwconv2d_chw_config.dwconv2d_chw_3x3.update.f16 = xnn_update_f16_chw_neonfp16arith_stride1_params;
      f16_dwconv2d_chw_config.dwconv2d_chw_3x3.output_height_tile = 2;
      f16_dwconv2d_chw_config.dwconv2d_chw_3x3.output_width_tile = 8;

      f16_dwconv2d_chw_config.dwconv2d_chw_3x3s2.ukernel = (xnn_dwconv2d_chw_ukernel_fn) xnn_f16_dwconv2d_chw_ukernel_3x3s2p1__neonfp16arith_1x8;
      f16_dwconv2d_chw_config.dwconv2d_chw_3x3s2.init.f16 = xnn_init_f16_chw_neonfp16arith_stride2_params;
      f16_dwconv2d_chw_config.dwconv2d_chw_3x3s2.update.f16 = xnn_update_f16_chw_neonfp16arith_stride2_params;
      f16_dwconv2d_chw_config.dwconv2d_chw_3x3s2.output_height_tile = 1;
      f16_dwconv2d_chw_config.dwconv2d_chw_3x3s2.output_width_tile = 8;

      f16_dwconv2d_chw_config.dwconv2d_chw_5x5.ukernel = (xnn_dwconv2d_chw_ukernel_fn) xnn_f16_dwconv2d_chw_ukernel_5x5p2__neonfp16arith_1x8;
      f16_dwconv2d_chw_config.dwconv2d_chw_5x5.init.f16 = xnn_init_f16_chw_neonfp16arith_stride1_params;
      f16_dwconv2d_chw_config.dwconv2d_chw_5x5.update.f16 = xnn_update_f16_chw_neonfp16arith_stride1_params;
      f16_dwconv2d_chw_config.dwconv2d_chw_5x5.output_height_tile = 1;
      f16_dwconv2d_chw_config.dwconv2d_chw_5x5.output_width_tile = 8;

      f16_dwconv2d_chw_config.dwconv2d_chw_5x5s2.ukernel = (xnn_dwconv2d_chw_ukernel_fn) xnn_f16_dwconv2d_chw_ukernel_5x5s2p2__neonfp16arith_1x8;
      f16_dwconv2d_chw_config.dwconv2d_chw_5x5s2.init.f16 = xnn_init_f16_chw_neonfp16arith_stride2_params;
      f16_dwconv2d_chw_config.dwconv2d_chw_5x5s2.update.f16 = xnn_update_f16_chw_neonfp16arith_stride2_params;
      f16_dwconv2d_chw_config.dwconv2d_chw_5x5s2.output_height_tile = 1;
      f16_dwconv2d_chw_config.dwconv2d_chw_5x5s2.output_width_tile = 8;
    }
  #elif XNN_ARCH_ARM64 && XNN_ENABLE_ARM_FP16_VECTOR
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon_fp16_arith) {
      f16_dwconv2d_chw_config.dwconv2d_chw_3x3.ukernel = (xnn_dwconv2d_chw_ukernel_fn) xnn_f16_dwconv2d_chw_ukernel_3x3p1__neonfp16arith_2x8;
      f16_dwconv2d_chw_config.dwconv2d_chw_3x3.init.f16 = xnn_init_f16_chw_neonfp16arith_stride1_params;
      f16_dwconv2d_chw_config.dwconv2d_chw_3x3.update.f16 = xnn_update_f16_chw_neonfp16arith_stride1_params;
      f16_dwconv2d_chw_config.dwconv2d_chw_3x3.output_height_tile = 2;
      f16_dwconv2d_chw_config.dwconv2d_chw_3x3.output_width_tile = 8;

      f16_dwconv2d_chw_config.dwconv2d_chw_3x3s2.ukernel = (xnn_dwconv2d_chw_ukernel_fn) xnn_f16_dwconv2d_chw_ukernel_3x3s2p1__neonfp16arith_1x8;
      f16_dwconv2d_chw_config.dwconv2d_chw_3x3s2.init.f16 = xnn_init_f16_chw_neonfp16arith_stride2_params;
      f16_dwconv2d_chw_config.dwconv2d_chw_3x3s2.update.f16 = xnn_update_f16_chw_neonfp16arith_stride2_params;
      f16_dwconv2d_chw_config.dwconv2d_chw_3x3s2.output_height_tile = 1;
      f16_dwconv2d_chw_config.dwconv2d_chw_3x3s2.output_width_tile = 8;

      f16_dwconv2d_chw_config.dwconv2d_chw_5x5.ukernel = (xnn_dwconv2d_chw_ukernel_fn) xnn_f16_dwconv2d_chw_ukernel_5x5p2__neonfp16arith_1x8;
      f16_dwconv2d_chw_config.dwconv2d_chw_5x5.init.f16 = xnn_init_f16_chw_neonfp16arith_stride1_params;
      f16_dwconv2d_chw_config.dwconv2d_chw_5x5.update.f16 = xnn_update_f16_chw_neonfp16arith_stride1_params;
      f16_dwconv2d_chw_config.dwconv2d_chw_5x5.output_height_tile = 1;
      f16_dwconv2d_chw_config.dwconv2d_chw_5x5.output_width_tile = 8;

      f16_dwconv2d_chw_config.dwconv2d_chw_5x5s2.ukernel = (xnn_dwconv2d_chw_ukernel_fn) xnn_f16_dwconv2d_chw_ukernel_5x5s2p2__neonfp16arith_1x8;
      f16_dwconv2d_chw_config.dwconv2d_chw_5x5s2.init.f16 = xnn_init_f16_chw_neonfp16arith_stride2_params;
      f16_dwconv2d_chw_config.dwconv2d_chw_5x5s2.update.f16 = xnn_update_f16_chw_neonfp16arith_stride2_params;
      f16_dwconv2d_chw_config.dwconv2d_chw_5x5s2.output_height_tile = 1;
      f16_dwconv2d_chw_config.dwconv2d_chw_5x5s2.output_width_tile = 8;
    }
  #endif
}

static void init_f32_dwconv2d_chw_config(void) {
  #if XNN_ARCH_ARM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon) {
      f32_dwconv2d_chw_config.dwconv2d_chw_3x3.ukernel = (xnn_dwconv2d_chw_ukernel_fn) xnn_f32_dwconv2d_chw_ukernel_3x3p1__neon_2x4;
      f32_dwconv2d_chw_config.dwconv2d_chw_3x3.init.f32 = xnn_init_f32_chw_neon_stride1_params;
      f32_dwconv2d_chw_config.dwconv2d_chw_3x3.update.f32 = xnn_update_f32_chw_neon_stride1_params;
      f32_dwconv2d_chw_config.dwconv2d_chw_3x3.output_height_tile = 2;
      f32_dwconv2d_chw_config.dwconv2d_chw_3x3.output_width_tile = 4;

      f32_dwconv2d_chw_config.dwconv2d_chw_3x3s2.ukernel = (xnn_dwconv2d_chw_ukernel_fn) xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__neon_1x4;
      f32_dwconv2d_chw_config.dwconv2d_chw_3x3s2.init.f32 = xnn_init_f32_chw_neon_stride2_params;
      f32_dwconv2d_chw_config.dwconv2d_chw_3x3s2.update.f32 = xnn_update_f32_chw_neon_stride2_params;
      f32_dwconv2d_chw_config.dwconv2d_chw_3x3s2.output_height_tile = 1;
      f32_dwconv2d_chw_config.dwconv2d_chw_3x3s2.output_width_tile = 4;

      f32_dwconv2d_chw_config.dwconv2d_chw_5x5.ukernel = (xnn_dwconv2d_chw_ukernel_fn) xnn_f32_dwconv2d_chw_ukernel_5x5p2__neon_1x4;
      f32_dwconv2d_chw_config.dwconv2d_chw_5x5.init.f32 = xnn_init_f32_chw_neon_stride1_params;
      f32_dwconv2d_chw_config.dwconv2d_chw_5x5.update.f32 = xnn_update_f32_chw_neon_stride1_params;
      f32_dwconv2d_chw_config.dwconv2d_chw_5x5.output_height_tile = 1;
      f32_dwconv2d_chw_config.dwconv2d_chw_5x5.output_width_tile = 4;

      f32_dwconv2d_chw_config.dwconv2d_chw_5x5s2.ukernel = (xnn_dwconv2d_chw_ukernel_fn) xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__neon_1x4;
      f32_dwconv2d_chw_config.dwconv2d_chw_5x5s2.init.f32 = xnn_init_f32_chw_neon_stride2_params;
      f32_dwconv2d_chw_config.dwconv2d_chw_5x5s2.update.f32 = xnn_update_f32_chw_neon_stride2_params;
      f32_dwconv2d_chw_config.dwconv2d_chw_5x5s2.output_height_tile = 1;
      f32_dwconv2d_chw_config.dwconv2d_chw_5x5s2.output_width_tile = 4;
    } else if (!XNN_PLATFORM_MOBILE) {
      f32_dwconv2d_chw_config.dwconv2d_chw_3x3.ukernel = (xnn_dwconv2d_chw_ukernel_fn) xnn_f32_dwconv2d_chw_ukernel_3x3p1__scalar_4x1;
      f32_dwconv2d_chw_config.dwconv2d_chw_3x3.init.f32 = xnn_init_f32_chw_scalar_params;
      f32_dwconv2d_chw_config.dwconv2d_chw_3x3.output_height_tile = 4;
      f32_dwconv2d_chw_config.dwconv2d_chw_3x3.output_width_tile = 1;

      f32_dwconv2d_chw_config.dwconv2d_chw_3x3s2.ukernel = (xnn_dwconv2d_chw_ukernel_fn) xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__scalar_2x1_acc2;
      f32_dwconv2d_chw_config.dwconv2d_chw_3x3s2.init.f32 = xnn_init_f32_chw_scalar_params;
      f32_dwconv2d_chw_config.dwconv2d_chw_3x3s2.output_height_tile = 2;
      f32_dwconv2d_chw_config.dwconv2d_chw_3x3s2.output_width_tile = 1;

      f32_dwconv2d_chw_config.dwconv2d_chw_5x5.ukernel = (xnn_dwconv2d_chw_ukernel_fn) xnn_f32_dwconv2d_chw_ukernel_5x5p2__scalar_2x1_acc2;
      f32_dwconv2d_chw_config.dwconv2d_chw_5x5.init.f32 = xnn_init_f32_chw_scalar_params;
      f32_dwconv2d_chw_config.dwconv2d_chw_5x5.output_height_tile = 2;
      f32_dwconv2d_chw_config.dwconv2d_chw_5x5.output_width_tile = 1;

      f32_dwconv2d_chw_config.dwconv2d_chw_5x5s2.ukernel = (xnn_dwconv2d_chw_ukernel_fn) xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__scalar_2x1_acc2;
      f32_dwconv2d_chw_config.dwconv2d_chw_5x5s2.init.f32 = xnn_init_f32_chw_scalar_params;
      f32_dwconv2d_chw_config.dwconv2d_chw_5x5s2.output_height_tile = 2;
      f32_dwconv2d_chw_config.dwconv2d_chw_5x5s2.output_width_tile = 1;
    }
  #elif XNN_ARCH_ARM64
    f32_dwconv2d_chw_config.dwconv2d_chw_3x3.ukernel = (xnn_dwconv2d_chw_ukernel_fn) xnn_f32_dwconv2d_chw_ukernel_3x3p1__aarch64_neonfma_3x4;
    f32_dwconv2d_chw_config.dwconv2d_chw_3x3.init.f32 = xnn_init_f32_chw_neon_stride1_params;
    f32_dwconv2d_chw_config.dwconv2d_chw_3x3.update.f32 = xnn_update_f32_chw_neon_stride1_params;
    f32_dwconv2d_chw_config.dwconv2d_chw_3x3.output_height_tile = 3;
    f32_dwconv2d_chw_config.dwconv2d_chw_3x3.output_width_tile = 4;

    f32_dwconv2d_chw_config.dwconv2d_chw_3x3s2.ukernel = (xnn_dwconv2d_chw_ukernel_fn) xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__aarch64_neonfma_2x4_acc2;
    f32_dwconv2d_chw_config.dwconv2d_chw_3x3s2.init.f32 = xnn_init_f32_chw_neon_stride2_params;
    f32_dwconv2d_chw_config.dwconv2d_chw_3x3s2.update.f32 = xnn_update_f32_chw_neon_stride2_params;
    f32_dwconv2d_chw_config.dwconv2d_chw_3x3s2.output_height_tile = 2;
    f32_dwconv2d_chw_config.dwconv2d_chw_3x3s2.output_width_tile = 4;

    f32_dwconv2d_chw_config.dwconv2d_chw_5x5.ukernel = (xnn_dwconv2d_chw_ukernel_fn) xnn_f32_dwconv2d_chw_ukernel_5x5p2__aarch64_neonfma_4x4;
    f32_dwconv2d_chw_config.dwconv2d_chw_5x5.init.f32 = xnn_init_f32_chw_neon_stride1_params;
    f32_dwconv2d_chw_config.dwconv2d_chw_5x5.update.f32 = xnn_update_f32_chw_neon_stride1_params;
    f32_dwconv2d_chw_config.dwconv2d_chw_5x5.output_height_tile = 4;
    f32_dwconv2d_chw_config.dwconv2d_chw_5x5.output_width_tile = 4;

    f32_dwconv2d_chw_config.dwconv2d_chw_5x5s2.ukernel = (xnn_dwconv2d_chw_ukernel_fn) xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__aarch64_neonfma_1x4_acc2;
    f32_dwconv2d_chw_config.dwconv2d_chw_5x5s2.init.f32 = xnn_init_f32_chw_neon_stride2_params;
    f32_dwconv2d_chw_config.dwconv2d_chw_5x5s2.update.f32 = xnn_update_f32_chw_neon_stride2_params;
    f32_dwconv2d_chw_config.dwconv2d_chw_5x5s2.output_height_tile = 1;
    f32_dwconv2d_chw_config.dwconv2d_chw_5x5s2.output_width_tile = 4;
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_x86_ssse3) {
      f32_dwconv2d_chw_config.dwconv2d_chw_3x3.ukernel = (xnn_dwconv2d_chw_ukernel_fn) xnn_f32_dwconv2d_chw_ukernel_3x3p1__ssse3_2x4_acc2;
      f32_dwconv2d_chw_config.dwconv2d_chw_3x3.init.f32 = xnn_init_f32_chw_sse_stride1_params;
      f32_dwconv2d_chw_config.dwconv2d_chw_3x3.update.f32 = xnn_update_f32_chw_sse_stride1_params;
      f32_dwconv2d_chw_config.dwconv2d_chw_3x3.output_height_tile = 2;
      f32_dwconv2d_chw_config.dwconv2d_chw_3x3.output_width_tile = 4;
    } else {
      f32_dwconv2d_chw_config.dwconv2d_chw_3x3.ukernel = (xnn_dwconv2d_chw_ukernel_fn) xnn_f32_dwconv2d_chw_ukernel_3x3p1__sse_2x4_acc2;
      f32_dwconv2d_chw_config.dwconv2d_chw_3x3.init.f32 = xnn_init_f32_chw_sse_stride1_params;
      f32_dwconv2d_chw_config.dwconv2d_chw_3x3.update.f32 = xnn_update_f32_chw_sse_stride1_params;
      f32_dwconv2d_chw_config.dwconv2d_chw_3x3.output_height_tile = 2;
      f32_dwconv2d_chw_config.dwconv2d_chw_3x3.output_width_tile = 4;
    }

    f32_dwconv2d_chw_config.dwconv2d_chw_3x3s2.ukernel = (xnn_dwconv2d_chw_ukernel_fn) xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__sse_1x4_acc3;
    f32_dwconv2d_chw_config.dwconv2d_chw_3x3s2.init.f32 = xnn_init_f32_chw_sse_stride2_params;
    f32_dwconv2d_chw_config.dwconv2d_chw_3x3s2.update.f32 = xnn_update_f32_chw_sse_stride2_params;
    f32_dwconv2d_chw_config.dwconv2d_chw_3x3s2.output_height_tile = 1;
    f32_dwconv2d_chw_config.dwconv2d_chw_3x3s2.output_width_tile = 4;

    f32_dwconv2d_chw_config.dwconv2d_chw_5x5.ukernel = (xnn_dwconv2d_chw_ukernel_fn) xnn_f32_dwconv2d_chw_ukernel_5x5p2__sse_4x4;
    f32_dwconv2d_chw_config.dwconv2d_chw_5x5.init.f32 = xnn_init_f32_chw_sse_stride1_params;
    f32_dwconv2d_chw_config.dwconv2d_chw_5x5.update.f32 = xnn_update_f32_chw_sse_stride1_params;
    f32_dwconv2d_chw_config.dwconv2d_chw_5x5.output_height_tile = 4;
    f32_dwconv2d_chw_config.dwconv2d_chw_5x5.output_width_tile = 4;

    f32_dwconv2d_chw_config.dwconv2d_chw_5x5s2.ukernel = (xnn_dwconv2d_chw_ukernel_fn) xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__sse_2x4;
    f32_dwconv2d_chw_config.dwconv2d_chw_5x5s2.init.f32 = xnn_init_f32_chw_sse_stride2_params;
    f32_dwconv2d_chw_config.dwconv2d_chw_5x5s2.update.f32 = xnn_update_f32_chw_sse_stride2_params;
    f32_dwconv2d_chw_config.dwconv2d_chw_5x5s2.output_height_tile = 2;
    f32_dwconv2d_chw_config.dwconv2d_chw_5x5s2.output_width_tile = 4;
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->is_x86) {
      f32_dwconv2d_chw_config.dwconv2d_chw_3x3.ukernel = (xnn_dwconv2d_chw_ukernel_fn) xnn_f32_dwconv2d_chw_ukernel_3x3p1__wasmsimd_x86_loadsplat_2x4;
      f32_dwconv2d_chw_config.dwconv2d_chw_3x3.init.f32 = xnn_init_f32_chw_wasmsimd_stride1_params;
      f32_dwconv2d_chw_config.dwconv2d_chw_3x3.update.f32 = xnn_update_f32_chw_wasmsimd_stride1_params;
      f32_dwconv2d_chw_config.dwconv2d_chw_3x3.output_height_tile = 2;
      f32_dwconv2d_chw_config.dwconv2d_chw_3x3.output_width_tile = 4;

      f32_dwconv2d_chw_config.dwconv2d_chw_3x3s2.ukernel = (xnn_dwconv2d_chw_ukernel_fn) xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__wasmsimd_x86_splat_1x4_acc2;
      f32_dwconv2d_chw_config.dwconv2d_chw_3x3s2.init.f32 = xnn_init_f32_chw_wasmsimd_stride2_params;
      f32_dwconv2d_chw_config.dwconv2d_chw_3x3s2.update.f32 = xnn_update_f32_chw_wasmsimd_stride2_params;
      f32_dwconv2d_chw_config.dwconv2d_chw_3x3s2.output_height_tile = 1;
      f32_dwconv2d_chw_config.dwconv2d_chw_3x3s2.output_width_tile = 4;

      f32_dwconv2d_chw_config.dwconv2d_chw_5x5.ukernel = (xnn_dwconv2d_chw_ukernel_fn) xnn_f32_dwconv2d_chw_ukernel_5x5p2__wasmsimd_x86_splat_3x4;
      f32_dwconv2d_chw_config.dwconv2d_chw_5x5.init.f32 = xnn_init_f32_chw_wasmsimd_stride1_params;
      f32_dwconv2d_chw_config.dwconv2d_chw_5x5.update.f32 = xnn_update_f32_chw_wasmsimd_stride1_params;
      f32_dwconv2d_chw_config.dwconv2d_chw_5x5.output_height_tile = 3;
      f32_dwconv2d_chw_config.dwconv2d_chw_5x5.output_width_tile = 4;

      f32_dwconv2d_chw_config.dwconv2d_chw_5x5s2.ukernel = (xnn_dwconv2d_chw_ukernel_fn) xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__wasmsimd_x86_splat_1x4_acc2;
      f32_dwconv2d_chw_config.dwconv2d_chw_5x5s2.init.f32 = xnn_init_f32_chw_wasmsimd_stride2_params;
      f32_dwconv2d_chw_config.dwconv2d_chw_5x5s2.update.f32 = xnn_update_f32_chw_wasmsimd_stride2_params;
      f32_dwconv2d_chw_config.dwconv2d_chw_5x5s2.output_height_tile = 1;
      f32_dwconv2d_chw_config.dwconv2d_chw_5x5s2.output_width_tile = 4;
    } else {
      f32_dwconv2d_chw_config.dwconv2d_chw_3x3.ukernel = (xnn_dwconv2d_chw_ukernel_fn) xnn_f32_dwconv2d_chw_ukernel_3x3p1__wasmsimd_arm_loadsplat_2x4;
      f32_dwconv2d_chw_config.dwconv2d_chw_3x3.init.f32 = xnn_init_f32_chw_wasmsimd_stride1_params;
      f32_dwconv2d_chw_config.dwconv2d_chw_3x3.update.f32 = xnn_update_f32_chw_wasmsimd_stride1_params;
      f32_dwconv2d_chw_config.dwconv2d_chw_3x3.output_height_tile = 2;
      f32_dwconv2d_chw_config.dwconv2d_chw_3x3.output_width_tile = 4;

      f32_dwconv2d_chw_config.dwconv2d_chw_3x3s2.ukernel = (xnn_dwconv2d_chw_ukernel_fn) xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__wasmsimd_arm_splat_1x4_acc4;
      f32_dwconv2d_chw_config.dwconv2d_chw_3x3s2.init.f32 = xnn_init_f32_chw_wasmsimd_stride2_params;
      f32_dwconv2d_chw_config.dwconv2d_chw_3x3s2.update.f32 = xnn_update_f32_chw_wasmsimd_stride2_params;
      f32_dwconv2d_chw_config.dwconv2d_chw_3x3s2.output_height_tile = 1;
      f32_dwconv2d_chw_config.dwconv2d_chw_3x3s2.output_width_tile = 4;

      f32_dwconv2d_chw_config.dwconv2d_chw_5x5.ukernel = (xnn_dwconv2d_chw_ukernel_fn) xnn_f32_dwconv2d_chw_ukernel_5x5p2__wasmsimd_arm_splat_3x4;
      f32_dwconv2d_chw_config.dwconv2d_chw_5x5.init.f32 = xnn_init_f32_chw_wasmsimd_stride1_params;
      f32_dwconv2d_chw_config.dwconv2d_chw_5x5.update.f32 = xnn_update_f32_chw_wasmsimd_stride1_params;
      f32_dwconv2d_chw_config.dwconv2d_chw_5x5.output_height_tile = 3;
      f32_dwconv2d_chw_config.dwconv2d_chw_5x5.output_width_tile = 4;

      f32_dwconv2d_chw_config.dwconv2d_chw_5x5s2.ukernel = (xnn_dwconv2d_chw_ukernel_fn) xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__wasmsimd_arm_splat_1x4_acc2;
      f32_dwconv2d_chw_config.dwconv2d_chw_5x5s2.init.f32 = xnn_init_f32_chw_wasmsimd_stride2_params;
      f32_dwconv2d_chw_config.dwconv2d_chw_5x5s2.update.f32 = xnn_update_f32_chw_wasmsimd_stride2_params;
      f32_dwconv2d_chw_config.dwconv2d_chw_5x5s2.output_height_tile = 1;
      f32_dwconv2d_chw_config.dwconv2d_chw_5x5s2.output_width_tile = 4;
    }
  #elif XNN_ARCH_WASM
    f32_dwconv2d_chw_config.dwconv2d_chw_3x3.ukernel = (xnn_dwconv2d_chw_ukernel_fn) xnn_f32_dwconv2d_chw_ukernel_3x3p1__scalar_2x1_acc2;
    f32_dwconv2d_chw_config.dwconv2d_chw_3x3.init.f32 = xnn_init_f32_chw_scalar_params;
    f32_dwconv2d_chw_config.dwconv2d_chw_3x3.output_height_tile = 2;
    f32_dwconv2d_chw_config.dwconv2d_chw_3x3.output_width_tile = 1;

    f32_dwconv2d_chw_config.dwconv2d_chw_3x3s2.ukernel = (xnn_dwconv2d_chw_ukernel_fn) xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__scalar_1x1_acc2;
    f32_dwconv2d_chw_config.dwconv2d_chw_3x3s2.init.f32 = xnn_init_f32_chw_scalar_params;
    f32_dwconv2d_chw_config.dwconv2d_chw_3x3s2.output_height_tile = 1;
    f32_dwconv2d_chw_config.dwconv2d_chw_3x3s2.output_width_tile = 1;

    f32_dwconv2d_chw_config.dwconv2d_chw_5x5.ukernel = (xnn_dwconv2d_chw_ukernel_fn) xnn_f32_dwconv2d_chw_ukernel_5x5p2__scalar_1x1_acc5;
    f32_dwconv2d_chw_config.dwconv2d_chw_5x5.init.f32 = xnn_init_f32_chw_scalar_params;
    f32_dwconv2d_chw_config.dwconv2d_chw_5x5.output_height_tile = 1;
    f32_dwconv2d_chw_config.dwconv2d_chw_5x5.output_width_tile = 1;

    f32_dwconv2d_chw_config.dwconv2d_chw_5x5s2.ukernel = (xnn_dwconv2d_chw_ukernel_fn) xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__scalar_1x1_acc5;
    f32_dwconv2d_chw_config.dwconv2d_chw_5x5s2.init.f32 = xnn_init_f32_chw_scalar_params;
    f32_dwconv2d_chw_config.dwconv2d_chw_5x5s2.output_height_tile = 1;
    f32_dwconv2d_chw_config.dwconv2d_chw_5x5s2.output_width_tile = 1;
  #elif XNN_ARCH_RISCV
    f32_dwconv2d_chw_config.dwconv2d_chw_3x3.ukernel = (xnn_dwconv2d_chw_ukernel_fn) xnn_f32_dwconv2d_chw_ukernel_3x3p1__scalar_2x1_acc2;
    f32_dwconv2d_chw_config.dwconv2d_chw_3x3.init.f32 = xnn_init_f32_chw_scalar_params;
    f32_dwconv2d_chw_config.dwconv2d_chw_3x3.output_height_tile = 2;
    f32_dwconv2d_chw_config.dwconv2d_chw_3x3.output_width_tile = 1;

    f32_dwconv2d_chw_config.dwconv2d_chw_3x3s2.ukernel = (xnn_dwconv2d_chw_ukernel_fn) xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__scalar_1x1_acc2;
    f32_dwconv2d_chw_config.dwconv2d_chw_3x3s2.init.f32 = xnn_init_f32_chw_scalar_params;
    f32_dwconv2d_chw_config.dwconv2d_chw_3x3s2.output_height_tile = 1;
    f32_dwconv2d_chw_config.dwconv2d_chw_3x3s2.output_width_tile = 1;

    f32_dwconv2d_chw_config.dwconv2d_chw_5x5.ukernel = (xnn_dwconv2d_chw_ukernel_fn) xnn_f32_dwconv2d_chw_ukernel_5x5p2__scalar_1x1_acc5;
    f32_dwconv2d_chw_config.dwconv2d_chw_5x5.init.f32 = xnn_init_f32_chw_scalar_params;
    f32_dwconv2d_chw_config.dwconv2d_chw_5x5.output_height_tile = 1;
    f32_dwconv2d_chw_config.dwconv2d_chw_5x5.output_width_tile = 1;

    f32_dwconv2d_chw_config.dwconv2d_chw_5x5s2.ukernel = (xnn_dwconv2d_chw_ukernel_fn) xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__scalar_1x1_acc5;
    f32_dwconv2d_chw_config.dwconv2d_chw_5x5s2.init.f32 = xnn_init_f32_chw_scalar_params;
    f32_dwconv2d_chw_config.dwconv2d_chw_5x5s2.output_height_tile = 1;
    f32_dwconv2d_chw_config.dwconv2d_chw_5x5s2.output_width_tile = 1;
  #elif XNN_ARCH_PPC64
    f32_dwconv2d_chw_config.dwconv2d_chw_3x3.ukernel = (xnn_dwconv2d_chw_ukernel_fn) xnn_f32_dwconv2d_chw_ukernel_3x3p1__scalar_2x1_acc2;
    f32_dwconv2d_chw_config.dwconv2d_chw_3x3.init.f32 = xnn_init_f32_chw_scalar_params;
    f32_dwconv2d_chw_config.dwconv2d_chw_3x3.output_height_tile = 2;
    f32_dwconv2d_chw_config.dwconv2d_chw_3x3.output_width_tile = 1;

    f32_dwconv2d_chw_config.dwconv2d_chw_3x3s2.ukernel = (xnn_dwconv2d_chw_ukernel_fn) xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__scalar_1x1_acc2;
    f32_dwconv2d_chw_config.dwconv2d_chw_3x3s2.init.f32 = xnn_init_f32_chw_scalar_params;
    f32_dwconv2d_chw_config.dwconv2d_chw_3x3s2.output_height_tile = 1;
    f32_dwconv2d_chw_config.dwconv2d_chw_3x3s2.output_width_tile = 1;

    f32_dwconv2d_chw_config.dwconv2d_chw_5x5.ukernel = (xnn_dwconv2d_chw_ukernel_fn) xnn_f32_dwconv2d_chw_ukernel_5x5p2__scalar_1x1_acc5;
    f32_dwconv2d_chw_config.dwconv2d_chw_5x5.init.f32 = xnn_init_f32_chw_scalar_params;
    f32_dwconv2d_chw_config.dwconv2d_chw_5x5.output_height_tile = 1;
    f32_dwconv2d_chw_config.dwconv2d_chw_5x5.output_width_tile = 1;

    f32_dwconv2d_chw_config.dwconv2d_chw_5x5s2.ukernel = (xnn_dwconv2d_chw_ukernel_fn) xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__scalar_1x1_acc5;
    f32_dwconv2d_chw_config.dwconv2d_chw_5x5s2.init.f32 = xnn_init_f32_chw_scalar_params;
    f32_dwconv2d_chw_config.dwconv2d_chw_5x5s2.output_height_tile = 1;
    f32_dwconv2d_chw_config.dwconv2d_chw_5x5s2.output_width_tile = 1;
  #endif
}

#if XNN_PLATFORM_WINDOWS
  static BOOL CALLBACK init_f16_dwconv2d_chw_config_windows(PINIT_ONCE init_once, PVOID parameter, PVOID* context) {
    init_f16_dwconv2d_chw_config();
    return TRUE;
  }

  static BOOL CALLBACK init_f32_dwconv2d_chw_config_windows(PINIT_ONCE init_once, PVOID parameter, PVOID* context) {
    init_f32_dwconv2d_chw_config();
    return TRUE;
  }
#endif

const struct xnn_dwconv2d_chw_config* xnn_init_f16_dwconv2d_chw_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL || !xnn_is_f16_chw_compatible_config(hardware_config)) {
    return NULL;
  }
  #if XNN_PLATFORM_WINDOWS
    InitOnceExecuteOnce(&init_guard_f16_dwconv2d_chw, &init_f16_dwconv2d_chw_config_windows, NULL, NULL);
  #else
    pthread_once(&init_guard_f16_dwconv2d_chw, &init_f16_dwconv2d_chw_config);
  #endif
  return &f16_dwconv2d_chw_config;
}

const struct xnn_dwconv2d_chw_config* xnn_init_f32_dwconv2d_chw_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL || !xnn_is_chw_compatible_config(hardware_config)) {
    return NULL;
  }
  #if XNN_PLATFORM_WINDOWS
    InitOnceExecuteOnce(&init_guard_f32_dwconv2d_chw, &init_f32_dwconv2d_chw_config_windows, NULL, NULL);
  #else
    pthread_once(&init_guard_f32_dwconv2d_chw, &init_f32_dwconv2d_chw_config);
  #endif
  return &f32_dwconv2d_chw_config;
}
