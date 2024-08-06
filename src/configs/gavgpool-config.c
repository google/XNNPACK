// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>

#include "xnnpack/common.h"
#include "xnnpack/config.h"
#include "xnnpack/gavgpool.h"
#include "xnnpack/init-once.h"
#include "xnnpack/microfnptr.h"
#include "xnnpack/microparams-init.h"

static struct xnn_gavgpool_config f16_gavgpool_config = {0};
static struct xnn_gavgpool_config f32_gavgpool_config = {0};
static struct xnn_gavgpool_config qs8_gavgpool_config = {0};
static struct xnn_gavgpool_config qu8_gavgpool_config = {0};

XNN_INIT_ONCE_GUARD(f16_gavgpool);
XNN_INIT_ONCE_GUARD(f32_gavgpool);
XNN_INIT_ONCE_GUARD(qs8_gavgpool);
XNN_INIT_ONCE_GUARD(qu8_gavgpool);

static void init_f16_gavgpool_config(void) {
  #if XNN_ARCH_ARM && XNN_ENABLE_ARM_FP16_VECTOR && XNN_ENABLE_ARM_FP16_SCALAR
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon_fp16_arith) {
      f16_gavgpool_config.unipass = (xnn_gavgpool_unipass_ukernel_fn) xnn_f16_gavgpool_minmax_ukernel_7x__neonfp16arith_c8;
      f16_gavgpool_config.multipass = (xnn_gavgpool_multipass_ukernel_fn) xnn_f16_gavgpool_minmax_ukernel_7p7x__neonfp16arith_c8;
      f16_gavgpool_config.init.f16 = xnn_init_f16_scaleminmax_fp16arith_params;
      f16_gavgpool_config.update.f16 = xnn_update_f16_scaleminmax_fp16arith_params;
      f16_gavgpool_config.row_tile = 7;
      f16_gavgpool_config.channel_tile = 8;
    }
  #elif XNN_ARCH_ARM64 && XNN_ENABLE_ARM_FP16_VECTOR
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon_fp16_arith) {
      f16_gavgpool_config.unipass = (xnn_gavgpool_unipass_ukernel_fn) xnn_f16_gavgpool_minmax_ukernel_7x__neonfp16arith_c8;
      f16_gavgpool_config.multipass = (xnn_gavgpool_multipass_ukernel_fn) xnn_f16_gavgpool_minmax_ukernel_7p7x__neonfp16arith_c8;
      f16_gavgpool_config.init.f16 = xnn_init_f16_scaleminmax_fp16arith_params;
      f16_gavgpool_config.update.f16 = xnn_update_f16_scaleminmax_fp16arith_params;
      f16_gavgpool_config.row_tile = 7;
      f16_gavgpool_config.channel_tile = 8;
    }
  #elif (XNN_ARCH_X86 || XNN_ARCH_X86_64) && !XNN_PLATFORM_MOBILE
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_x86_f16c) {
      f16_gavgpool_config.unipass = (xnn_gavgpool_unipass_ukernel_fn) xnn_f16_gavgpool_minmax_ukernel_7x__f16c_c8;
      f16_gavgpool_config.multipass = (xnn_gavgpool_multipass_ukernel_fn) xnn_f16_gavgpool_minmax_ukernel_7p7x__f16c_c8;
      f16_gavgpool_config.init.f16 = xnn_init_f16_scaleminmax_avx_params;
      f16_gavgpool_config.update.f16 = xnn_update_f16_scaleminmax_avx_params;
      f16_gavgpool_config.row_tile = 7;
      f16_gavgpool_config.channel_tile = 8;
    }
  #endif
}

static void init_f32_gavgpool_config(void) {
  #if XNN_ARCH_ARM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon) {
      f32_gavgpool_config.unipass = (xnn_gavgpool_unipass_ukernel_fn) xnn_f32_gavgpool_minmax_ukernel_7x__neon_c4;
      f32_gavgpool_config.multipass = (xnn_gavgpool_multipass_ukernel_fn) xnn_f32_gavgpool_minmax_ukernel_7p7x__neon_c4;
      f32_gavgpool_config.init.f32 = xnn_init_f32_scaleminmax_scalar_params;
      f32_gavgpool_config.update.f32 = xnn_update_f32_scaleminmax_scalar_params;
      f32_gavgpool_config.row_tile = 7;
      f32_gavgpool_config.channel_tile = 4;
    } else if (!XNN_PLATFORM_MOBILE) {
      f32_gavgpool_config.unipass = (xnn_gavgpool_unipass_ukernel_fn) xnn_f32_gavgpool_minmax_ukernel_7x__scalar_c1;
      f32_gavgpool_config.multipass = (xnn_gavgpool_multipass_ukernel_fn) xnn_f32_gavgpool_minmax_ukernel_7p7x__scalar_c1;
      f32_gavgpool_config.init.f32 = xnn_init_f32_scaleminmax_scalar_params;
      f32_gavgpool_config.update.f32 = xnn_update_f32_scaleminmax_scalar_params;
      f32_gavgpool_config.row_tile = 7;
      f32_gavgpool_config.channel_tile = 1;
    }
  #elif XNN_ARCH_ARM64
    f32_gavgpool_config.unipass = (xnn_gavgpool_unipass_ukernel_fn) xnn_f32_gavgpool_minmax_ukernel_7x__neon_c4;
    f32_gavgpool_config.multipass = (xnn_gavgpool_multipass_ukernel_fn) xnn_f32_gavgpool_minmax_ukernel_7p7x__neon_c4;
    f32_gavgpool_config.init.f32 = xnn_init_f32_scaleminmax_scalar_params;
    f32_gavgpool_config.update.f32 = xnn_update_f32_scaleminmax_scalar_params;
    f32_gavgpool_config.row_tile = 7;
    f32_gavgpool_config.channel_tile = 4;
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    f32_gavgpool_config.unipass = (xnn_gavgpool_unipass_ukernel_fn) xnn_f32_gavgpool_minmax_ukernel_7x__sse_c4;
    f32_gavgpool_config.multipass = (xnn_gavgpool_multipass_ukernel_fn) xnn_f32_gavgpool_minmax_ukernel_7p7x__sse_c4;
    f32_gavgpool_config.init.f32 = xnn_init_f32_scaleminmax_sse_params;
    f32_gavgpool_config.update.f32 = xnn_update_f32_scaleminmax_sse_params;
    f32_gavgpool_config.row_tile = 7;
    f32_gavgpool_config.channel_tile = 4;
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->is_x86) {
      f32_gavgpool_config.unipass = (xnn_gavgpool_unipass_ukernel_fn) xnn_f32_gavgpool_minmax_ukernel_7x__wasmsimd_x86_c4;
      f32_gavgpool_config.multipass = (xnn_gavgpool_multipass_ukernel_fn) xnn_f32_gavgpool_minmax_ukernel_7p7x__wasmsimd_x86_c4;
      f32_gavgpool_config.init.f32 = xnn_init_f32_scaleminmax_scalar_params;
      f32_gavgpool_config.update.f32 = xnn_update_f32_scaleminmax_scalar_params;
      f32_gavgpool_config.row_tile = 7;
      f32_gavgpool_config.channel_tile = 4;
    } else {
      f32_gavgpool_config.unipass = (xnn_gavgpool_unipass_ukernel_fn) xnn_f32_gavgpool_minmax_ukernel_7x__wasmsimd_arm_c4;
      f32_gavgpool_config.multipass = (xnn_gavgpool_multipass_ukernel_fn) xnn_f32_gavgpool_minmax_ukernel_7p7x__wasmsimd_arm_c4;
      f32_gavgpool_config.init.f32 = xnn_init_f32_scaleminmax_scalar_params;
      f32_gavgpool_config.update.f32 = xnn_update_f32_scaleminmax_scalar_params;
      f32_gavgpool_config.row_tile = 7;
      f32_gavgpool_config.channel_tile = 4;
    }
  #elif XNN_ARCH_WASM
    f32_gavgpool_config.unipass = (xnn_gavgpool_unipass_ukernel_fn) xnn_f32_gavgpool_minmax_ukernel_7x__wasm_c1;
    f32_gavgpool_config.multipass = (xnn_gavgpool_multipass_ukernel_fn) xnn_f32_gavgpool_minmax_ukernel_7p7x__wasm_c1;
    f32_gavgpool_config.init.f32 = xnn_init_f32_scaleminmax_scalar_params;
    f32_gavgpool_config.update.f32 = xnn_update_f32_scaleminmax_scalar_params;
    f32_gavgpool_config.row_tile = 7;
    f32_gavgpool_config.channel_tile = 1;
  #elif XNN_ARCH_RISCV && XNN_ENABLE_RISCV_VECTOR
    f32_gavgpool_config.unipass = (xnn_gavgpool_unipass_ukernel_fn) xnn_f32_gavgpool_minmax_ukernel_7x__rvv_c2v;
    f32_gavgpool_config.multipass = (xnn_gavgpool_multipass_ukernel_fn) xnn_f32_gavgpool_minmax_ukernel_7p7x__rvv_c2v;
    f32_gavgpool_config.init.f32 = xnn_init_f32_scaleminmax_scalar_params;
    f32_gavgpool_config.update.f32 = xnn_update_f32_scaleminmax_scalar_params;
    f32_gavgpool_config.row_tile = 7;
    f32_gavgpool_config.channel_tile = 16;
  #else
    f32_gavgpool_config.unipass = (xnn_gavgpool_unipass_ukernel_fn) xnn_f32_gavgpool_minmax_ukernel_7x__scalar_c1;
    f32_gavgpool_config.multipass = (xnn_gavgpool_multipass_ukernel_fn) xnn_f32_gavgpool_minmax_ukernel_7p7x__scalar_c1;
    f32_gavgpool_config.init.f32 = xnn_init_f32_scaleminmax_scalar_params;
    f32_gavgpool_config.update.f32 = xnn_update_f32_scaleminmax_scalar_params;
    f32_gavgpool_config.row_tile = 7;
    f32_gavgpool_config.channel_tile = 1;
  #endif
}

static void init_qs8_gavgpool_config(void) {
  #if XNN_ARCH_ARM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon) {
      qs8_gavgpool_config.unipass = (xnn_gavgpool_unipass_ukernel_fn) xnn_qs8_gavgpool_minmax_rndnu_ukernel_7x__neon_c8;
      qs8_gavgpool_config.multipass = (xnn_gavgpool_multipass_ukernel_fn) xnn_qs8_gavgpool_minmax_rndnu_ukernel_7p7x__neon_c8;
      qs8_gavgpool_config.init.qs8 = xnn_init_qs8_avgpool_minmax_rndnu_neon_params;
      qs8_gavgpool_config.update.qs8 = xnn_update_qs8_avgpool_minmax_rndnu_neon_params;
      qs8_gavgpool_config.row_tile = 7;
      qs8_gavgpool_config.channel_tile = 8;
    } else if (!XNN_PLATFORM_MOBILE) {
      qs8_gavgpool_config.unipass = (xnn_gavgpool_unipass_ukernel_fn) xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__scalar_imagic_c1;
      qs8_gavgpool_config.multipass = (xnn_gavgpool_multipass_ukernel_fn) xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_imagic_c1;
      qs8_gavgpool_config.init.qs8 = xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params;
      qs8_gavgpool_config.update.qs8 = xnn_update_qs8_avgpool_minmax_fp32_scalar_imagic_params;
      qs8_gavgpool_config.row_tile = 7;
      qs8_gavgpool_config.channel_tile = 1;
    }
  #elif XNN_ARCH_ARM64
    qs8_gavgpool_config.unipass = (xnn_gavgpool_unipass_ukernel_fn) xnn_qs8_gavgpool_minmax_rndnu_ukernel_7x__neon_c8;
    qs8_gavgpool_config.multipass = (xnn_gavgpool_multipass_ukernel_fn) xnn_qs8_gavgpool_minmax_rndnu_ukernel_7p7x__neon_c8;
    qs8_gavgpool_config.init.qs8 = xnn_init_qs8_avgpool_minmax_rndnu_neon_params;
    qs8_gavgpool_config.update.qs8 = xnn_update_qs8_avgpool_minmax_rndnu_neon_params;
    qs8_gavgpool_config.row_tile = 7;
    qs8_gavgpool_config.channel_tile = 8;
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_x86_sse4_1) {
      qs8_gavgpool_config.unipass = (xnn_gavgpool_unipass_ukernel_fn) xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__sse41_c8;
      qs8_gavgpool_config.multipass = (xnn_gavgpool_multipass_ukernel_fn) xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse41_c8;
      qs8_gavgpool_config.init.qs8 = xnn_init_qs8_avgpool_minmax_fp32_sse4_params;
      qs8_gavgpool_config.update.qs8 = xnn_update_qs8_avgpool_minmax_fp32_sse4_params;
      qs8_gavgpool_config.row_tile = 7;
      qs8_gavgpool_config.channel_tile = 8;
    } else {
      qs8_gavgpool_config.unipass = (xnn_gavgpool_unipass_ukernel_fn) xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__sse2_c8;
      qs8_gavgpool_config.multipass = (xnn_gavgpool_multipass_ukernel_fn) xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse2_c8;
      qs8_gavgpool_config.init.qs8 = xnn_init_qs8_avgpool_minmax_fp32_sse2_params;
      qs8_gavgpool_config.update.qs8 = xnn_update_qs8_avgpool_minmax_fp32_sse2_params;
      qs8_gavgpool_config.row_tile = 7;
      qs8_gavgpool_config.channel_tile = 8;
    }
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    qs8_gavgpool_config.unipass = (xnn_gavgpool_unipass_ukernel_fn) xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__wasmsimd_c16;
    qs8_gavgpool_config.multipass = (xnn_gavgpool_multipass_ukernel_fn) xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__wasmsimd_c16;
    qs8_gavgpool_config.init.qs8 = xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params;
    qs8_gavgpool_config.update.qs8 = xnn_update_qs8_avgpool_minmax_fp32_wasmsimd_params;
    qs8_gavgpool_config.row_tile = 7;
    qs8_gavgpool_config.channel_tile = 16;
  #elif XNN_ARCH_WASM
    qs8_gavgpool_config.unipass = (xnn_gavgpool_unipass_ukernel_fn) xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__scalar_imagic_c4;
    qs8_gavgpool_config.multipass = (xnn_gavgpool_multipass_ukernel_fn) xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_imagic_c4;
    qs8_gavgpool_config.init.qs8 = xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params;
    qs8_gavgpool_config.update.qs8 = xnn_update_qs8_avgpool_minmax_fp32_scalar_imagic_params;
    qs8_gavgpool_config.row_tile = 7;
    qs8_gavgpool_config.channel_tile = 4;
  #else
    qs8_gavgpool_config.unipass = (xnn_gavgpool_unipass_ukernel_fn) xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__scalar_imagic_c1;
    qs8_gavgpool_config.multipass = (xnn_gavgpool_multipass_ukernel_fn) xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_imagic_c1;
    qs8_gavgpool_config.init.qs8 = xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params;
    qs8_gavgpool_config.update.qs8 = xnn_update_qs8_avgpool_minmax_fp32_scalar_imagic_params;
    qs8_gavgpool_config.row_tile = 7;
    qs8_gavgpool_config.channel_tile = 1;
  #endif
}

static void init_qu8_gavgpool_config(void) {
  #if XNN_ARCH_ARM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon) {
      qu8_gavgpool_config.unipass = (xnn_gavgpool_unipass_ukernel_fn) xnn_qu8_gavgpool_minmax_rndnu_ukernel_7x__neon_c8;
      qu8_gavgpool_config.multipass = (xnn_gavgpool_multipass_ukernel_fn) xnn_qu8_gavgpool_minmax_rndnu_ukernel_7p7x__neon_c8;
      qu8_gavgpool_config.init.qu8 = xnn_init_qu8_avgpool_minmax_rndnu_neon_params;
      qu8_gavgpool_config.update.qu8 = xnn_update_qu8_avgpool_minmax_rndnu_neon_params;
      qu8_gavgpool_config.row_tile = 7;
      qu8_gavgpool_config.channel_tile = 8;
    } else if (!XNN_PLATFORM_MOBILE) {
      qu8_gavgpool_config.unipass = (xnn_gavgpool_unipass_ukernel_fn) xnn_qu8_gavgpool_minmax_fp32_ukernel_7x__scalar_imagic_c1;
      qu8_gavgpool_config.multipass = (xnn_gavgpool_multipass_ukernel_fn) xnn_qu8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_imagic_c1;
      qu8_gavgpool_config.init.qu8 = xnn_init_qu8_avgpool_minmax_fp32_scalar_imagic_params;
      qu8_gavgpool_config.update.qu8 = xnn_update_qu8_avgpool_minmax_fp32_scalar_imagic_params;
      qu8_gavgpool_config.row_tile = 7;
      qu8_gavgpool_config.channel_tile = 1;
    }
  #elif XNN_ARCH_ARM64
    qu8_gavgpool_config.unipass = (xnn_gavgpool_unipass_ukernel_fn) xnn_qu8_gavgpool_minmax_rndnu_ukernel_7x__neon_c8;
    qu8_gavgpool_config.multipass = (xnn_gavgpool_multipass_ukernel_fn) xnn_qu8_gavgpool_minmax_rndnu_ukernel_7p7x__neon_c8;
    qu8_gavgpool_config.init.qu8 = xnn_init_qu8_avgpool_minmax_rndnu_neon_params;
    qu8_gavgpool_config.update.qu8 = xnn_update_qu8_avgpool_minmax_rndnu_neon_params;
    qu8_gavgpool_config.row_tile = 7;
    qu8_gavgpool_config.channel_tile = 8;
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_x86_sse4_1) {
      qu8_gavgpool_config.unipass = (xnn_gavgpool_unipass_ukernel_fn) xnn_qu8_gavgpool_minmax_fp32_ukernel_7x__sse41_c8;
      qu8_gavgpool_config.multipass = (xnn_gavgpool_multipass_ukernel_fn) xnn_qu8_gavgpool_minmax_fp32_ukernel_7p7x__sse41_c8;
      qu8_gavgpool_config.init.qu8 = xnn_init_qu8_avgpool_minmax_fp32_sse4_params;
      qu8_gavgpool_config.update.qu8 = xnn_update_qu8_avgpool_minmax_fp32_sse4_params;
      qu8_gavgpool_config.row_tile = 7;
      qu8_gavgpool_config.channel_tile = 8;
    } else {
      qu8_gavgpool_config.unipass = (xnn_gavgpool_unipass_ukernel_fn) xnn_qu8_gavgpool_minmax_fp32_ukernel_7x__sse2_c8;
      qu8_gavgpool_config.multipass = (xnn_gavgpool_multipass_ukernel_fn) xnn_qu8_gavgpool_minmax_fp32_ukernel_7p7x__sse2_c8;
      qu8_gavgpool_config.init.qu8 = xnn_init_qu8_avgpool_minmax_fp32_sse2_params;
      qu8_gavgpool_config.update.qu8 = xnn_update_qu8_avgpool_minmax_fp32_sse2_params;
      qu8_gavgpool_config.row_tile = 7;
      qu8_gavgpool_config.channel_tile = 8;
    }
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    qu8_gavgpool_config.unipass = (xnn_gavgpool_unipass_ukernel_fn) xnn_qu8_gavgpool_minmax_fp32_ukernel_7x__wasmsimd_c16;
    qu8_gavgpool_config.multipass = (xnn_gavgpool_multipass_ukernel_fn) xnn_qu8_gavgpool_minmax_fp32_ukernel_7p7x__wasmsimd_c16;
    qu8_gavgpool_config.init.qu8 = xnn_init_qu8_avgpool_minmax_fp32_wasmsimd_params;
    qu8_gavgpool_config.update.qu8 = xnn_update_qu8_avgpool_minmax_fp32_wasmsimd_params;
    qu8_gavgpool_config.row_tile = 7;
    qu8_gavgpool_config.channel_tile = 16;
  #elif XNN_ARCH_WASM
    qu8_gavgpool_config.unipass = (xnn_gavgpool_unipass_ukernel_fn) xnn_qu8_gavgpool_minmax_fp32_ukernel_7x__scalar_imagic_c4;
    qu8_gavgpool_config.multipass = (xnn_gavgpool_multipass_ukernel_fn) xnn_qu8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_imagic_c4;
    qu8_gavgpool_config.init.qu8 = xnn_init_qu8_avgpool_minmax_fp32_scalar_imagic_params;
    qu8_gavgpool_config.update.qu8 = xnn_update_qu8_avgpool_minmax_fp32_scalar_imagic_params;
    qu8_gavgpool_config.row_tile = 7;
    qu8_gavgpool_config.channel_tile = 4;
  #else
    qu8_gavgpool_config.unipass = (xnn_gavgpool_unipass_ukernel_fn) xnn_qu8_gavgpool_minmax_fp32_ukernel_7x__scalar_imagic_c1;
    qu8_gavgpool_config.multipass = (xnn_gavgpool_multipass_ukernel_fn) xnn_qu8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_imagic_c1;
    qu8_gavgpool_config.init.qu8 = xnn_init_qu8_avgpool_minmax_fp32_scalar_imagic_params;
    qu8_gavgpool_config.update.qu8 = xnn_update_qu8_avgpool_minmax_fp32_scalar_imagic_params;
    qu8_gavgpool_config.row_tile = 7;
    qu8_gavgpool_config.channel_tile = 1;
  #endif
}

const struct xnn_gavgpool_config* xnn_init_f16_gavgpool_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL || !xnn_is_f16_compatible_config(hardware_config)) {
    return NULL;
  }
  XNN_INIT_ONCE(f16_gavgpool);
  return &f16_gavgpool_config;
}

const struct xnn_gavgpool_config* xnn_init_f32_gavgpool_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  XNN_INIT_ONCE(f32_gavgpool);
  return &f32_gavgpool_config;
}

const struct xnn_gavgpool_config* xnn_init_qs8_gavgpool_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  XNN_INIT_ONCE(qs8_gavgpool);
  return &qs8_gavgpool_config;
}

const struct xnn_gavgpool_config* xnn_init_qu8_gavgpool_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  XNN_INIT_ONCE(qu8_gavgpool);
  return &qu8_gavgpool_config;
}
