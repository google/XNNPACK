// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>

#include "xnnpack/common.h"
#include "xnnpack/config.h"
#include "xnnpack/init-once.h"
#include "xnnpack/microfnptr.h"
#include "xnnpack/microparams-init.h"
#include "xnnpack/vmulcaddc.h"

static struct xnn_vmulcaddc_config f16_vmulcaddc_config = {0};
static struct xnn_vmulcaddc_config f32_vmulcaddc_config = {0};

XNN_INIT_ONCE_GUARD(f16_vmulcaddc);
XNN_INIT_ONCE_GUARD(f32_vmulcaddc);

static void init_f16_vmulcaddc_config(void) {
  #if XNN_ARCH_ARM && XNN_ENABLE_ARM_FP16_VECTOR && XNN_ENABLE_ARM_FP16_SCALAR
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon_fp16_arith) {
      f16_vmulcaddc_config.ukernel = (xnn_vmulcaddc_ukernel_fn) xnn_f16_vmulcaddc_minmax_ukernel_c8__neonfp16arith_2x;
      f16_vmulcaddc_config.init.f16 = xnn_init_f16_minmax_fp16arith_params;
      f16_vmulcaddc_config.channel_tile = 8;
      f16_vmulcaddc_config.row_tile = 2;
    }
  #elif XNN_ARCH_ARM64 && XNN_ENABLE_ARM_FP16_VECTOR
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon_fp16_arith) {
      f16_vmulcaddc_config.ukernel = (xnn_vmulcaddc_ukernel_fn) xnn_f16_vmulcaddc_minmax_ukernel_c8__neonfp16arith_2x;
      f16_vmulcaddc_config.init.f16 = xnn_init_f16_minmax_fp16arith_params;
      f16_vmulcaddc_config.channel_tile = 8;
      f16_vmulcaddc_config.row_tile = 2;
    }
  #elif (XNN_ARCH_X86 || XNN_ARCH_X86_64) && !XNN_PLATFORM_MOBILE
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_x86_avx2) {
      f16_vmulcaddc_config.ukernel = (xnn_vmulcaddc_ukernel_fn) xnn_f16_vmulcaddc_minmax_ukernel_c8__fma3_2x;
      f16_vmulcaddc_config.init.f16 = xnn_init_f16_minmax_avx_params;
      f16_vmulcaddc_config.channel_tile = 8;
      f16_vmulcaddc_config.row_tile = 2;
    }
  #endif
}

static void init_f32_vmulcaddc_config(void) {
  #if XNN_ARCH_ARM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon) {
      f32_vmulcaddc_config.ukernel = (xnn_vmulcaddc_ukernel_fn) xnn_f32_vmulcaddc_minmax_ukernel_c4__neon_2x;
      f32_vmulcaddc_config.init.f32 = xnn_init_f32_minmax_scalar_params;
      f32_vmulcaddc_config.channel_tile = 4;
      f32_vmulcaddc_config.row_tile = 2;
    } else if (!XNN_PLATFORM_MOBILE) {
      f32_vmulcaddc_config.ukernel = (xnn_vmulcaddc_ukernel_fn) xnn_f32_vmulcaddc_minmax_ukernel_c1__scalar_2x;
      f32_vmulcaddc_config.init.f32 = xnn_init_f32_minmax_scalar_params;
      f32_vmulcaddc_config.channel_tile = 1;
      f32_vmulcaddc_config.row_tile = 2;
    }
  #elif XNN_ARCH_ARM64
    f32_vmulcaddc_config.ukernel = (xnn_vmulcaddc_ukernel_fn) xnn_f32_vmulcaddc_minmax_ukernel_c4__neonfma_2x;
    f32_vmulcaddc_config.init.f32 = xnn_init_f32_minmax_scalar_params;
    f32_vmulcaddc_config.channel_tile = 4;
    f32_vmulcaddc_config.row_tile = 2;
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    f32_vmulcaddc_config.ukernel = (xnn_vmulcaddc_ukernel_fn) xnn_f32_vmulcaddc_minmax_ukernel_c4__sse_2x;
    f32_vmulcaddc_config.init.f32 = xnn_init_f32_minmax_sse_params;
    f32_vmulcaddc_config.channel_tile = 4;
    f32_vmulcaddc_config.row_tile = 2;
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    #if XNN_ARCH_WASMRELAXEDSIMD
      f32_vmulcaddc_config.ukernel = (xnn_vmulcaddc_ukernel_fn) xnn_f32_vmulcaddc_minmax_ukernel_c4__wasmrelaxedsimd_fma_2x;
      f32_vmulcaddc_config.init.f32 = xnn_init_f32_minmax_wasmsimd_params;
      f32_vmulcaddc_config.channel_tile = 4;
      f32_vmulcaddc_config.row_tile = 2;
    #else
      const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
      assert(hardware_config != NULL);
      if (hardware_config->is_x86) {
        f32_vmulcaddc_config.ukernel = (xnn_vmulcaddc_ukernel_fn) xnn_f32_vmulcaddc_minmax_ukernel_c4__wasmsimd_x86_2x;
        f32_vmulcaddc_config.init.f32 = xnn_init_f32_minmax_wasmsimd_params;
        f32_vmulcaddc_config.channel_tile = 4;
        f32_vmulcaddc_config.row_tile = 2;
      } else {
        f32_vmulcaddc_config.ukernel = (xnn_vmulcaddc_ukernel_fn) xnn_f32_vmulcaddc_minmax_ukernel_c4__wasmsimd_arm_2x;
        f32_vmulcaddc_config.init.f32 = xnn_init_f32_minmax_wasmsimd_params;
        f32_vmulcaddc_config.channel_tile = 4;
        f32_vmulcaddc_config.row_tile = 2;
      }
    #endif
  #elif XNN_ARCH_WASM
    f32_vmulcaddc_config.ukernel = (xnn_vmulcaddc_ukernel_fn) xnn_f32_vmulcaddc_minmax_ukernel_c1__wasm_2x;
    f32_vmulcaddc_config.init.f32 = xnn_init_f32_minmax_scalar_params;
    f32_vmulcaddc_config.channel_tile = 1;
    f32_vmulcaddc_config.row_tile = 2;
  #else
    f32_vmulcaddc_config.ukernel = (xnn_vmulcaddc_ukernel_fn) xnn_f32_vmulcaddc_minmax_ukernel_c1__scalar_2x;
    f32_vmulcaddc_config.init.f32 = xnn_init_f32_minmax_scalar_params;
    f32_vmulcaddc_config.channel_tile = 1;
    f32_vmulcaddc_config.row_tile = 2;
  #endif
}

const struct xnn_vmulcaddc_config* xnn_init_f16_vmulcaddc_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL || !xnn_is_f16_compatible_config(hardware_config)) {
    return NULL;
  }
  XNN_INIT_ONCE(f16_vmulcaddc);
  return &f16_vmulcaddc_config;
}

const struct xnn_vmulcaddc_config* xnn_init_f32_vmulcaddc_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  XNN_INIT_ONCE(f32_vmulcaddc);
  return &f32_vmulcaddc_config;
}
