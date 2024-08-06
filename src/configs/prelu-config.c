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
#include "xnnpack/prelu.h"

static struct xnn_prelu_config f16_prelu_config = {0};
static struct xnn_prelu_config f32_prelu_config = {0};

XNN_INIT_ONCE_GUARD(f16_prelu);
XNN_INIT_ONCE_GUARD(f32_prelu);

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
    if (hardware_config->use_x86_f16c) {
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
  #else
    f32_prelu_config.ukernel = (xnn_prelu_ukernel_fn) xnn_f32_prelu_ukernel__scalar_2x4;
    f32_prelu_config.row_tile = 4;
    f32_prelu_config.channel_tile = 4;
  #endif
}

const struct xnn_prelu_config* xnn_init_f16_prelu_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL || !xnn_is_f16_compatible_config(hardware_config)) {
    return NULL;
  }
  XNN_INIT_ONCE(f16_prelu);
  return &f16_prelu_config;
}

const struct xnn_prelu_config* xnn_init_f32_prelu_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  XNN_INIT_ONCE(f32_prelu);
  return &f32_prelu_config;
}
