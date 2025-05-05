// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/config.h"
#include "src/xnnpack/ibilinear.h"
#include "src/xnnpack/init-once.h"
#include "src/xnnpack/indirection.h"
#include "src/xnnpack/microfnptr.h"

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
    }
  #elif XNN_ARCH_ARM64 && XNN_ENABLE_ARM_FP16_VECTOR
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon_fp16_arith) {
      f16_ibilinear_chw_config.ukernel = (xnn_ibilinear_chw_ukernel_fn) xnn_f16_ibilinear_chw_ukernel__neonfp16arith_p8;
      f16_ibilinear_chw_config.channel_tile = 1;
    }
  #endif
  f16_ibilinear_chw_config.log2_data_element_size = XNN_LOG2_SIZEOF_HALF;
  f16_ibilinear_chw_config.log2_weight_element_size = XNN_LOG2_SIZEOF_HALF;
  f16_ibilinear_chw_config.indirection_init =
      (xnn_indirection_init_resize_bilinear2d_chw_fn) xnn_indirection_init_resize_bilinear2d_chw_f16;
}

static void init_f32_ibilinear_chw_config(void) {
  #if XNN_ARCH_ARM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon) {
      f32_ibilinear_chw_config.ukernel = (xnn_ibilinear_chw_ukernel_fn) xnn_f32_ibilinear_chw_ukernel__neon_p8;
      f32_ibilinear_chw_config.channel_tile = 1;
    } else if (!XNN_PLATFORM_MOBILE) {
      f32_ibilinear_chw_config.ukernel = (xnn_ibilinear_chw_ukernel_fn) xnn_f32_ibilinear_chw_ukernel__scalar_p4;
      f32_ibilinear_chw_config.channel_tile = 1;
    }
  #elif XNN_ARCH_ARM64
    f32_ibilinear_chw_config.ukernel = (xnn_ibilinear_chw_ukernel_fn) xnn_f32_ibilinear_chw_ukernel__neonfma_p8;
    f32_ibilinear_chw_config.channel_tile = 1;
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    f32_ibilinear_chw_config.ukernel = (xnn_ibilinear_chw_ukernel_fn) xnn_f32_ibilinear_chw_ukernel__sse_p8;
    f32_ibilinear_chw_config.channel_tile = 1;
  #elif XNN_ARCH_WASMRELAXEDSIMD || XNN_ARCH_WASMSIMD
    f32_ibilinear_chw_config.ukernel = (xnn_ibilinear_chw_ukernel_fn) xnn_f32_ibilinear_chw_ukernel__wasmsimd_p8;
    f32_ibilinear_chw_config.channel_tile = 1;
  #else
    f32_ibilinear_chw_config.ukernel = (xnn_ibilinear_chw_ukernel_fn) xnn_f32_ibilinear_chw_ukernel__scalar_p4;
    f32_ibilinear_chw_config.channel_tile = 1;
  #endif
  f32_ibilinear_chw_config.log2_data_element_size = XNN_LOG2_SIZEOF_FLOAT;
  f32_ibilinear_chw_config.log2_weight_element_size = XNN_LOG2_SIZEOF_FLOAT;
  f32_ibilinear_chw_config.indirection_init =
      (xnn_indirection_init_resize_bilinear2d_chw_fn) xnn_indirection_init_resize_bilinear2d_chw_f32;
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
