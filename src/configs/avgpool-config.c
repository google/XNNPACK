// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>

#include "src/xnnpack/avgpool.h"
#include "src/xnnpack/common.h"
#include "src/xnnpack/config.h"
#include "src/xnnpack/init-once.h"
#include "src/xnnpack/microfnptr.h"
#include "src/xnnpack/microparams-init.h"

static struct xnn_avgpool_config f16_avgpool_config = {0};
static struct xnn_avgpool_config f32_avgpool_config = {0};

XNN_INIT_ONCE_GUARD(f16_avgpool);
XNN_INIT_ONCE_GUARD(f32_avgpool);

static void init_f16_avgpool_config(void) {
  #if (XNN_ARCH_ARM || XNN_ARCH_ARM64) && XNN_ENABLE_ARM_FP16_VECTOR
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon_fp16_arith) {
      f16_avgpool_config.ukernel = (xnn_avgpool_ukernel_fn) xnn_f16_avgpool_minmax_ukernel_9p__neonfp16arith_u8;
      f16_avgpool_config.init.f16 = xnn_init_f16_scaleminmax_scalar_params;
      f16_avgpool_config.primary_tile = 9;
      f16_avgpool_config.channel_tile = 8;
    }
  #elif (XNN_ARCH_X86 || XNN_ARCH_X86_64) && !XNN_PLATFORM_MOBILE
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_x86_f16c) {
      f16_avgpool_config.ukernel = (xnn_avgpool_ukernel_fn) xnn_f16_avgpool_minmax_ukernel_9p__f16c_u8;
      f16_avgpool_config.init.f16 = xnn_init_f16_scaleminmax_scalar_params;
      f16_avgpool_config.primary_tile = 9;
      f16_avgpool_config.channel_tile = 8;
    }
  #endif
}

static void init_f32_avgpool_config(void) {
  #if XNN_ARCH_ARM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon) {
      f32_avgpool_config.ukernel = (xnn_avgpool_ukernel_fn) xnn_f32_avgpool_minmax_ukernel_9p__neon_u4;
      f32_avgpool_config.init.f32 = xnn_init_f32_scaleminmax_scalar_params;
      f32_avgpool_config.primary_tile = 9;
      f32_avgpool_config.channel_tile = 4;
    } else if (!XNN_PLATFORM_MOBILE) {
      f32_avgpool_config.ukernel = (xnn_avgpool_ukernel_fn) xnn_f32_avgpool_minmax_ukernel_9p__scalar_u1;
      f32_avgpool_config.init.f32 = xnn_init_f32_scaleminmax_scalar_params;
      f32_avgpool_config.primary_tile = 9;
      f32_avgpool_config.channel_tile = 1;
    }
  #elif XNN_ARCH_ARM64
    f32_avgpool_config.ukernel = (xnn_avgpool_ukernel_fn) xnn_f32_avgpool_minmax_ukernel_9p__neon_u4;
    f32_avgpool_config.init.f32 = xnn_init_f32_scaleminmax_scalar_params;
    f32_avgpool_config.primary_tile = 9;
    f32_avgpool_config.channel_tile = 4;
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    f32_avgpool_config.ukernel = (xnn_avgpool_ukernel_fn) xnn_f32_avgpool_minmax_ukernel_9p__sse2_u4;
    f32_avgpool_config.init.f32 = xnn_init_f32_scaleminmax_scalar_params;
    f32_avgpool_config.primary_tile = 9;
    f32_avgpool_config.channel_tile = 4;
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    f32_avgpool_config.ukernel = (xnn_avgpool_ukernel_fn) xnn_f32_avgpool_minmax_ukernel_9p__wasmsimd_u4;
    f32_avgpool_config.init.f32 = xnn_init_f32_scaleminmax_scalar_params;
    f32_avgpool_config.primary_tile = 9;
    f32_avgpool_config.channel_tile = 4;
  #else
    f32_avgpool_config.ukernel = (xnn_avgpool_ukernel_fn) xnn_f32_avgpool_minmax_ukernel_9p__scalar_u1;
    f32_avgpool_config.init.f32 = xnn_init_f32_scaleminmax_scalar_params;
    f32_avgpool_config.primary_tile = 9;
    f32_avgpool_config.channel_tile = 1;
  #endif
}

const struct xnn_avgpool_config* xnn_init_f16_avgpool_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL || !xnn_is_f16_compatible_config(hardware_config)) {
    return NULL;
  }
  XNN_INIT_ONCE(f16_avgpool);
  return &f16_avgpool_config;
}

const struct xnn_avgpool_config* xnn_init_f32_avgpool_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  XNN_INIT_ONCE(f32_avgpool);
  return &f32_avgpool_config;
}
