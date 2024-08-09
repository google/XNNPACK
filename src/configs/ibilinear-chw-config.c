// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>

#include "xnnpack/common.h"
#include "xnnpack/config.h"
#include "xnnpack/ibilinear.h"
#include "xnnpack/init-once.h"
#include "xnnpack/microfnptr.h"

static struct xnn_ibilinear_chw_config f16_ibilinear_chw_config = {0};
static struct xnn_ibilinear_chw_config f32_ibilinear_chw_config = {0};

XNN_INIT_ONCE_GUARD(f16_ibilinear_chw);
XNN_INIT_ONCE_GUARD(f32_ibilinear_chw);

static void init_f16_ibilinear_chw_config(void) {
  #if XNN_ARCH_ARM && XNN_ENABLE_ARM_FP16_VECTOR && XNN_ENABLE_ARM_FP16_SCALAR
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon_fp16_arith) {
      f16_ibilinear_chw_config.ukernel = (xnn_ibilinear_chw_ukernel_fn) xnn_f16_ibilinear_chw_ukernel__neonfp16arith_p8;
      f16_ibilinear_chw_config.channel_tile = 1;
      f16_ibilinear_chw_config.pixel_tile = 8;
    }
  #elif XNN_ARCH_ARM64 && XNN_ENABLE_ARM_FP16_VECTOR
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon_fp16_arith) {
      f16_ibilinear_chw_config.ukernel = (xnn_ibilinear_chw_ukernel_fn) xnn_f16_ibilinear_chw_ukernel__neonfp16arith_p8;
      f16_ibilinear_chw_config.channel_tile = 1;
      f16_ibilinear_chw_config.pixel_tile = 8;
    }
  #endif
}

static void init_f32_ibilinear_chw_config(void) {
  #if XNN_ARCH_ARM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon) {
      f32_ibilinear_chw_config.ukernel = (xnn_ibilinear_chw_ukernel_fn) xnn_f32_ibilinear_chw_ukernel__neon_p8;
      f32_ibilinear_chw_config.channel_tile = 1;
      f32_ibilinear_chw_config.pixel_tile = 8;
    } else if (!XNN_PLATFORM_MOBILE) {
      f32_ibilinear_chw_config.ukernel = (xnn_ibilinear_chw_ukernel_fn) xnn_f32_ibilinear_chw_ukernel__scalar_p4;
      f32_ibilinear_chw_config.channel_tile = 1;
      f32_ibilinear_chw_config.pixel_tile = 4;
    }
  #elif XNN_ARCH_ARM64
    f32_ibilinear_chw_config.ukernel = (xnn_ibilinear_chw_ukernel_fn) xnn_f32_ibilinear_chw_ukernel__neonfma_p8;
    f32_ibilinear_chw_config.channel_tile = 1;
    f32_ibilinear_chw_config.pixel_tile = 8;
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    f32_ibilinear_chw_config.ukernel = (xnn_ibilinear_chw_ukernel_fn) xnn_f32_ibilinear_chw_ukernel__sse_p8;
    f32_ibilinear_chw_config.channel_tile = 1;
    f32_ibilinear_chw_config.pixel_tile = 8;
  #elif XNN_ARCH_WASMRELAXEDSIMD || XNN_ARCH_WASMSIMD
    f32_ibilinear_chw_config.ukernel = (xnn_ibilinear_chw_ukernel_fn) xnn_f32_ibilinear_chw_ukernel__wasmsimd_p8;
    f32_ibilinear_chw_config.channel_tile = 1;
    f32_ibilinear_chw_config.pixel_tile = 8;
  #else
    f32_ibilinear_chw_config.ukernel = (xnn_ibilinear_chw_ukernel_fn) xnn_f32_ibilinear_chw_ukernel__scalar_p4;
    f32_ibilinear_chw_config.channel_tile = 1;
    f32_ibilinear_chw_config.pixel_tile = 4;
  #endif
}

const struct xnn_ibilinear_chw_config* xnn_init_f16_ibilinear_chw_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL || !xnn_is_f16_chw_compatible_config(hardware_config)) {
    return NULL;
  }
  XNN_INIT_ONCE(f16_ibilinear_chw);
  return &f16_ibilinear_chw_config;
}

const struct xnn_ibilinear_chw_config* xnn_init_f32_ibilinear_chw_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL || !xnn_is_chw_compatible_config(hardware_config)) {
    return NULL;
  }
  XNN_INIT_ONCE(f32_ibilinear_chw);
  return &f32_ibilinear_chw_config;
}
