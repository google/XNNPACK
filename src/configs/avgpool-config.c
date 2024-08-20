// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>

#include "xnnpack/avgpool.h"
#include "xnnpack/common.h"
#include "xnnpack/config.h"
#include "xnnpack/init-once.h"
#include "xnnpack/microfnptr.h"
#include "xnnpack/microparams-init.h"

static struct xnn_avgpool_config f16_avgpool_config = {0};
static struct xnn_avgpool_config f32_avgpool_config = {0};
static struct xnn_avgpool_config qu8_avgpool_config = {0};

XNN_INIT_ONCE_GUARD(f16_avgpool);
XNN_INIT_ONCE_GUARD(f32_avgpool);
XNN_INIT_ONCE_GUARD(qu8_avgpool);

static void init_f16_avgpool_config(void) {
  #if XNN_ARCH_ARM && XNN_ENABLE_ARM_FP16_VECTOR && XNN_ENABLE_ARM_FP16_SCALAR
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon_fp16_arith) {
      f16_avgpool_config.unipass = (xnn_avgpool_unipass_ukernel_fn) xnn_f16_avgpool_minmax_ukernel_9x__neonfp16arith_c8;
      f16_avgpool_config.multipass = (xnn_avgpool_multipass_ukernel_fn) xnn_f16_avgpool_minmax_ukernel_9p8x__neonfp16arith_c8;
      f16_avgpool_config.init.f16 = xnn_init_f16_scaleminmax_fp16arith_params;
      f16_avgpool_config.primary_tile = 9;
      f16_avgpool_config.incremental_tile = 8;
      f16_avgpool_config.channel_tile = 8;
    }
  #elif XNN_ARCH_ARM64 && XNN_ENABLE_ARM_FP16_VECTOR
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon_fp16_arith) {
      f16_avgpool_config.unipass = (xnn_avgpool_unipass_ukernel_fn) xnn_f16_avgpool_minmax_ukernel_9x__neonfp16arith_c8;
      f16_avgpool_config.multipass = (xnn_avgpool_multipass_ukernel_fn) xnn_f16_avgpool_minmax_ukernel_9p8x__neonfp16arith_c8;
      f16_avgpool_config.init.f16 = xnn_init_f16_scaleminmax_fp16arith_params;
      f16_avgpool_config.primary_tile = 9;
      f16_avgpool_config.incremental_tile = 8;
      f16_avgpool_config.channel_tile = 8;
    }
  #elif (XNN_ARCH_X86 || XNN_ARCH_X86_64) && !XNN_PLATFORM_MOBILE
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_x86_f16c) {
      f16_avgpool_config.unipass = (xnn_avgpool_unipass_ukernel_fn) xnn_f16_avgpool_minmax_ukernel_9x__f16c_c8;
      f16_avgpool_config.multipass = (xnn_avgpool_multipass_ukernel_fn) xnn_f16_avgpool_minmax_ukernel_9p8x__f16c_c8;
      f16_avgpool_config.init.f16 = xnn_init_f16_scaleminmax_avx_params;
      f16_avgpool_config.primary_tile = 9;
      f16_avgpool_config.incremental_tile = 8;
      f16_avgpool_config.channel_tile = 8;
    }
  #endif
}

static void init_f32_avgpool_config(void) {
  #if XNN_ARCH_ARM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon) {
      f32_avgpool_config.unipass = (xnn_avgpool_unipass_ukernel_fn) xnn_f32_avgpool_minmax_ukernel_9x__neon_c4;
      f32_avgpool_config.multipass = (xnn_avgpool_multipass_ukernel_fn) xnn_f32_avgpool_minmax_ukernel_9p8x__neon_c4;
      f32_avgpool_config.init.f32 = xnn_init_f32_scaleminmax_scalar_params;
      f32_avgpool_config.primary_tile = 9;
      f32_avgpool_config.incremental_tile = 8;
      f32_avgpool_config.channel_tile = 4;
    } else if (!XNN_PLATFORM_MOBILE) {
      f32_avgpool_config.unipass = (xnn_avgpool_unipass_ukernel_fn) xnn_f32_avgpool_minmax_ukernel_9x__scalar_c1;
      f32_avgpool_config.multipass = (xnn_avgpool_multipass_ukernel_fn) xnn_f32_avgpool_minmax_ukernel_9p8x__scalar_c1;
      f32_avgpool_config.init.f32 = xnn_init_f32_scaleminmax_scalar_params;
      f32_avgpool_config.primary_tile = 9;
      f32_avgpool_config.incremental_tile = 8;
      f32_avgpool_config.channel_tile = 1;
    }
  #elif XNN_ARCH_ARM64
    f32_avgpool_config.unipass = (xnn_avgpool_unipass_ukernel_fn) xnn_f32_avgpool_minmax_ukernel_9x__neon_c4;
    f32_avgpool_config.multipass = (xnn_avgpool_multipass_ukernel_fn) xnn_f32_avgpool_minmax_ukernel_9p8x__neon_c4;
    f32_avgpool_config.init.f32 = xnn_init_f32_scaleminmax_scalar_params;
    f32_avgpool_config.primary_tile = 9;
    f32_avgpool_config.incremental_tile = 8;
    f32_avgpool_config.channel_tile = 4;
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    f32_avgpool_config.unipass = (xnn_avgpool_unipass_ukernel_fn) xnn_f32_avgpool_minmax_ukernel_9x__sse_c4;
    f32_avgpool_config.multipass = (xnn_avgpool_multipass_ukernel_fn) xnn_f32_avgpool_minmax_ukernel_9p8x__sse_c4;
    f32_avgpool_config.init.f32 = xnn_init_f32_scaleminmax_sse_params;
    f32_avgpool_config.primary_tile = 9;
    f32_avgpool_config.incremental_tile = 8;
    f32_avgpool_config.channel_tile = 4;
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->is_x86) {
      f32_avgpool_config.unipass = (xnn_avgpool_unipass_ukernel_fn) xnn_f32_avgpool_minmax_ukernel_9x__wasmsimd_x86_c4;
      f32_avgpool_config.multipass = (xnn_avgpool_multipass_ukernel_fn) xnn_f32_avgpool_minmax_ukernel_9p8x__wasmsimd_x86_c4;
      f32_avgpool_config.init.f32 = xnn_init_f32_scaleminmax_scalar_params;
      f32_avgpool_config.primary_tile = 9;
      f32_avgpool_config.incremental_tile = 8;
      f32_avgpool_config.channel_tile = 4;
    } else {
      f32_avgpool_config.unipass = (xnn_avgpool_unipass_ukernel_fn) xnn_f32_avgpool_minmax_ukernel_9x__wasmsimd_arm_c4;
      f32_avgpool_config.multipass = (xnn_avgpool_multipass_ukernel_fn) xnn_f32_avgpool_minmax_ukernel_9p8x__wasmsimd_arm_c4;
      f32_avgpool_config.init.f32 = xnn_init_f32_scaleminmax_scalar_params;
      f32_avgpool_config.primary_tile = 9;
      f32_avgpool_config.incremental_tile = 8;
      f32_avgpool_config.channel_tile = 4;
    }
  #elif XNN_ARCH_WASM
    f32_avgpool_config.unipass = (xnn_avgpool_unipass_ukernel_fn) xnn_f32_avgpool_minmax_ukernel_9x__wasm_c1;
    f32_avgpool_config.multipass = (xnn_avgpool_multipass_ukernel_fn) xnn_f32_avgpool_minmax_ukernel_9p8x__wasm_c1;
    f32_avgpool_config.init.f32 = xnn_init_f32_scaleminmax_scalar_params;
    f32_avgpool_config.primary_tile = 9;
    f32_avgpool_config.incremental_tile = 8;
    f32_avgpool_config.channel_tile = 1;
  #elif XNN_ARCH_RISCV && XNN_ENABLE_RISCV_VECTOR
    f32_avgpool_config.unipass = (xnn_avgpool_unipass_ukernel_fn) xnn_f32_avgpool_minmax_ukernel_9x__rvv_c2v;
    f32_avgpool_config.multipass = (xnn_avgpool_multipass_ukernel_fn) xnn_f32_avgpool_minmax_ukernel_9p8x__rvv_c2v;
    f32_avgpool_config.init.f32 = xnn_init_f32_scaleminmax_scalar_params;
    f32_avgpool_config.primary_tile = 9;
    f32_avgpool_config.incremental_tile = 8;
    f32_avgpool_config.channel_tile = 4;
  #else
    f32_avgpool_config.unipass = (xnn_avgpool_unipass_ukernel_fn) xnn_f32_avgpool_minmax_ukernel_9x__scalar_c1;
    f32_avgpool_config.multipass = (xnn_avgpool_multipass_ukernel_fn) xnn_f32_avgpool_minmax_ukernel_9p8x__scalar_c1;
    f32_avgpool_config.init.f32 = xnn_init_f32_scaleminmax_scalar_params;
    f32_avgpool_config.primary_tile = 9;
    f32_avgpool_config.incremental_tile = 8;
    f32_avgpool_config.channel_tile = 1;
  #endif
}

static void init_qu8_avgpool_config(void) {
  #if XNN_ARCH_ARM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon) {
      qu8_avgpool_config.unipass = (xnn_avgpool_unipass_ukernel_fn) xnn_qu8_avgpool_minmax_fp32_ukernel_9x__neon_c8;
      qu8_avgpool_config.multipass = (xnn_avgpool_multipass_ukernel_fn) xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__neon_c8;
      qu8_avgpool_config.init.qu8 = xnn_init_qu8_avgpool_minmax_fp32_neon_params;
      qu8_avgpool_config.primary_tile = 9;
      qu8_avgpool_config.incremental_tile = 8;
      qu8_avgpool_config.channel_tile = 8;
    } else if (!XNN_PLATFORM_MOBILE) {
      qu8_avgpool_config.unipass = (xnn_avgpool_unipass_ukernel_fn) xnn_qu8_avgpool_minmax_fp32_ukernel_9x__scalar_imagic_c1;
      qu8_avgpool_config.multipass = (xnn_avgpool_multipass_ukernel_fn) xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__scalar_imagic_c1;
      qu8_avgpool_config.init.qu8 = xnn_init_qu8_avgpool_minmax_fp32_scalar_imagic_params;
      qu8_avgpool_config.primary_tile = 9;
      qu8_avgpool_config.incremental_tile = 8;
      qu8_avgpool_config.channel_tile = 1;
    }
  #elif XNN_ARCH_ARM64
    qu8_avgpool_config.unipass = (xnn_avgpool_unipass_ukernel_fn) xnn_qu8_avgpool_minmax_fp32_ukernel_9x__neon_c8;
    qu8_avgpool_config.multipass = (xnn_avgpool_multipass_ukernel_fn) xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__neon_c8;
    qu8_avgpool_config.init.qu8 = xnn_init_qu8_avgpool_minmax_fp32_neon_params;
    qu8_avgpool_config.primary_tile = 9;
    qu8_avgpool_config.incremental_tile = 8;
    qu8_avgpool_config.channel_tile = 8;
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    qu8_avgpool_config.unipass = (xnn_avgpool_unipass_ukernel_fn) xnn_qu8_avgpool_minmax_fp32_ukernel_9x__sse2_c8;
    qu8_avgpool_config.multipass = (xnn_avgpool_multipass_ukernel_fn) xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__sse2_c8;
    qu8_avgpool_config.init.qu8 = xnn_init_qu8_avgpool_minmax_fp32_sse2_params;
    qu8_avgpool_config.primary_tile = 9;
    qu8_avgpool_config.incremental_tile = 8;
    qu8_avgpool_config.channel_tile = 8;
  #else
    qu8_avgpool_config.unipass = (xnn_avgpool_unipass_ukernel_fn) xnn_qu8_avgpool_minmax_fp32_ukernel_9x__scalar_imagic_c1;
    qu8_avgpool_config.multipass = (xnn_avgpool_multipass_ukernel_fn) xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__scalar_imagic_c1;
    qu8_avgpool_config.init.qu8 = xnn_init_qu8_avgpool_minmax_fp32_scalar_imagic_params;
    qu8_avgpool_config.primary_tile = 9;
    qu8_avgpool_config.incremental_tile = 8;
    qu8_avgpool_config.channel_tile = 1;
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

const struct xnn_avgpool_config* xnn_init_qu8_avgpool_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  XNN_INIT_ONCE(qu8_avgpool);
  return &qu8_avgpool_config;
}
