// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>

#include "xnnpack/common.h"
#include "xnnpack/config.h"
#include "xnnpack/init-once.h"
#include "xnnpack/maxpool.h"
#include "xnnpack/microfnptr.h"
#include "xnnpack/microparams-init.h"

static struct xnn_maxpool_config f16_maxpool_config = {0};
static struct xnn_maxpool_config f32_maxpool_config = {0};
static struct xnn_maxpool_config s8_maxpool_config = {0};
static struct xnn_maxpool_config u8_maxpool_config = {0};

XNN_INIT_ONCE_GUARD(f16_maxpool);
XNN_INIT_ONCE_GUARD(f32_maxpool);
XNN_INIT_ONCE_GUARD(s8_maxpool);
XNN_INIT_ONCE_GUARD(u8_maxpool);

static void init_f16_maxpool_config(void) {
  #if XNN_ARCH_ARM && XNN_ENABLE_ARM_FP16_VECTOR && XNN_ENABLE_ARM_FP16_SCALAR
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon_fp16_arith) {
      f16_maxpool_config.ukernel = (xnn_maxpool_ukernel_fn) xnn_f16_maxpool_minmax_ukernel_9p8x__neonfp16arith_c8;
      f16_maxpool_config.init.f16 = xnn_init_f16_minmax_fp16arith_params;
      f16_maxpool_config.first_pass_tile_size = 9;
      f16_maxpool_config.remainder_pass_tile_size = 8;
    }
  #elif XNN_ARCH_ARM64 && XNN_ENABLE_ARM_FP16_VECTOR
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon_fp16_arith) {
      f16_maxpool_config.ukernel = (xnn_maxpool_ukernel_fn) xnn_f16_maxpool_minmax_ukernel_9p8x__neonfp16arith_c8;
      f16_maxpool_config.init.f16 = xnn_init_f16_minmax_fp16arith_params;
      f16_maxpool_config.first_pass_tile_size = 9;
      f16_maxpool_config.remainder_pass_tile_size = 8;
    }
  #elif (XNN_ARCH_X86 || XNN_ARCH_X86_64) && !XNN_PLATFORM_MOBILE
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_x86_f16c) {
      f16_maxpool_config.ukernel = (xnn_maxpool_ukernel_fn) xnn_f16_maxpool_minmax_ukernel_9p8x__f16c_c8;
      f16_maxpool_config.init.f16 = xnn_init_f16_minmax_avx_params;
      f16_maxpool_config.first_pass_tile_size = 9;
      f16_maxpool_config.remainder_pass_tile_size = 8;
    }
  #endif
}

static void init_f32_maxpool_config(void) {
  #if XNN_ARCH_ARM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon) {
      f32_maxpool_config.ukernel = (xnn_maxpool_ukernel_fn) xnn_f32_maxpool_minmax_ukernel_9p8x__neon_c4;
      f32_maxpool_config.init.f32 = xnn_init_f32_minmax_scalar_params;
      f32_maxpool_config.first_pass_tile_size = 9;
      f32_maxpool_config.remainder_pass_tile_size = 8;
    } else if (!XNN_PLATFORM_MOBILE) {
      f32_maxpool_config.ukernel = (xnn_maxpool_ukernel_fn) xnn_f32_maxpool_minmax_ukernel_9p8x__scalar_c1;
      f32_maxpool_config.init.f32 = xnn_init_f32_minmax_scalar_params;
      f32_maxpool_config.first_pass_tile_size = 9;
      f32_maxpool_config.remainder_pass_tile_size = 8;
    }
  #elif XNN_ARCH_ARM64
    f32_maxpool_config.ukernel = (xnn_maxpool_ukernel_fn) xnn_f32_maxpool_minmax_ukernel_9p8x__neon_c4;
    f32_maxpool_config.init.f32 = xnn_init_f32_minmax_scalar_params;
    f32_maxpool_config.first_pass_tile_size = 9;
    f32_maxpool_config.remainder_pass_tile_size = 8;
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    f32_maxpool_config.ukernel = (xnn_maxpool_ukernel_fn) xnn_f32_maxpool_minmax_ukernel_9p8x__sse_c4;
    f32_maxpool_config.init.f32 = xnn_init_f32_minmax_sse_params;
    f32_maxpool_config.first_pass_tile_size = 9;
    f32_maxpool_config.remainder_pass_tile_size = 8;
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->is_x86) {
      f32_maxpool_config.ukernel = (xnn_maxpool_ukernel_fn) xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_x86_c4;
      f32_maxpool_config.init.f32 = xnn_init_f32_minmax_wasmsimd_params;
      f32_maxpool_config.first_pass_tile_size = 9;
      f32_maxpool_config.remainder_pass_tile_size = 8;
    } else {
      f32_maxpool_config.ukernel = (xnn_maxpool_ukernel_fn) xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_arm_c4;
      f32_maxpool_config.init.f32 = xnn_init_f32_minmax_wasmsimd_params;
      f32_maxpool_config.first_pass_tile_size = 9;
      f32_maxpool_config.remainder_pass_tile_size = 8;
    }
  #elif XNN_ARCH_WASM
    f32_maxpool_config.ukernel = (xnn_maxpool_ukernel_fn) xnn_f32_maxpool_minmax_ukernel_9p8x__wasm_c1;
    f32_maxpool_config.init.f32 = xnn_init_f32_minmax_scalar_params;
    f32_maxpool_config.first_pass_tile_size = 9;
    f32_maxpool_config.remainder_pass_tile_size = 8;
  #elif XNN_ARCH_RISCV && XNN_ENABLE_RISCV_VECTOR
    f32_maxpool_config.ukernel = (xnn_maxpool_ukernel_fn) xnn_f32_maxpool_minmax_ukernel_9p8x__rvv_c2v;
    f32_maxpool_config.init.f32 = xnn_init_f32_minmax_scalar_params;
    f32_maxpool_config.first_pass_tile_size = 9;
    f32_maxpool_config.remainder_pass_tile_size = 8;
  #else
    f32_maxpool_config.ukernel = (xnn_maxpool_ukernel_fn) xnn_f32_maxpool_minmax_ukernel_9p8x__scalar_c1;
    f32_maxpool_config.init.f32 = xnn_init_f32_minmax_scalar_params;
    f32_maxpool_config.first_pass_tile_size = 9;
    f32_maxpool_config.remainder_pass_tile_size = 8;
  #endif
}

static void init_s8_maxpool_config(void) {
  #if XNN_ARCH_ARM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon) {
      s8_maxpool_config.ukernel = (xnn_maxpool_ukernel_fn) xnn_s8_maxpool_minmax_ukernel_9p8x__neon_c16;
      s8_maxpool_config.init.s8 = xnn_init_s8_minmax_scalar_params;
      s8_maxpool_config.first_pass_tile_size = 9;
      s8_maxpool_config.remainder_pass_tile_size = 8;
    } else if (!XNN_PLATFORM_MOBILE) {
      s8_maxpool_config.ukernel = (xnn_maxpool_ukernel_fn) xnn_s8_maxpool_minmax_ukernel_9p8x__scalar_c1;
      s8_maxpool_config.init.s8 = xnn_init_s8_minmax_scalar_params;
      s8_maxpool_config.first_pass_tile_size = 9;
      s8_maxpool_config.remainder_pass_tile_size = 8;
    }
  #elif XNN_ARCH_ARM64
    s8_maxpool_config.ukernel = (xnn_maxpool_ukernel_fn) xnn_s8_maxpool_minmax_ukernel_9p8x__neon_c16;
    s8_maxpool_config.init.s8 = xnn_init_s8_minmax_scalar_params;
    s8_maxpool_config.first_pass_tile_size = 9;
    s8_maxpool_config.remainder_pass_tile_size = 8;
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_x86_sse4_1) {
      s8_maxpool_config.ukernel = (xnn_maxpool_ukernel_fn) xnn_s8_maxpool_minmax_ukernel_9p8x__sse41_c16;
      s8_maxpool_config.init.s8 = xnn_init_s8_minmax_scalar_params;
      s8_maxpool_config.first_pass_tile_size = 9;
      s8_maxpool_config.remainder_pass_tile_size = 8;
    } else {
      s8_maxpool_config.ukernel = (xnn_maxpool_ukernel_fn) xnn_s8_maxpool_minmax_ukernel_9p8x__sse2_c16;
      s8_maxpool_config.init.s8 = xnn_init_s8_minmax_scalar_params;
      s8_maxpool_config.first_pass_tile_size = 9;
      s8_maxpool_config.remainder_pass_tile_size = 8;
    }
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    s8_maxpool_config.ukernel = (xnn_maxpool_ukernel_fn) xnn_s8_maxpool_minmax_ukernel_9p8x__wasmsimd_c16;
    s8_maxpool_config.init.s8 = xnn_init_s8_minmax_scalar_params;
    s8_maxpool_config.first_pass_tile_size = 9;
    s8_maxpool_config.remainder_pass_tile_size = 8;
  #else
    s8_maxpool_config.ukernel = (xnn_maxpool_ukernel_fn) xnn_s8_maxpool_minmax_ukernel_9p8x__scalar_c1;
    s8_maxpool_config.init.s8 = xnn_init_s8_minmax_scalar_params;
    s8_maxpool_config.first_pass_tile_size = 9;
    s8_maxpool_config.remainder_pass_tile_size = 8;
  #endif
}

static void init_u8_maxpool_config(void) {
  #if XNN_ARCH_ARM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon) {
      u8_maxpool_config.ukernel = (xnn_maxpool_ukernel_fn) xnn_u8_maxpool_minmax_ukernel_9p8x__neon_c16;
      u8_maxpool_config.init.u8 = xnn_init_u8_minmax_scalar_params;
      u8_maxpool_config.first_pass_tile_size = 9;
      u8_maxpool_config.remainder_pass_tile_size = 8;
    } else if (!XNN_PLATFORM_MOBILE) {
      u8_maxpool_config.ukernel = (xnn_maxpool_ukernel_fn) xnn_u8_maxpool_minmax_ukernel_9p8x__scalar_c1;
      u8_maxpool_config.init.u8 = xnn_init_u8_minmax_scalar_params;
      u8_maxpool_config.first_pass_tile_size = 9;
      u8_maxpool_config.remainder_pass_tile_size = 8;
    }
  #elif XNN_ARCH_ARM64
    u8_maxpool_config.ukernel = (xnn_maxpool_ukernel_fn) xnn_u8_maxpool_minmax_ukernel_9p8x__neon_c16;
    u8_maxpool_config.init.u8 = xnn_init_u8_minmax_scalar_params;
    u8_maxpool_config.first_pass_tile_size = 9;
    u8_maxpool_config.remainder_pass_tile_size = 8;
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    u8_maxpool_config.ukernel = (xnn_maxpool_ukernel_fn) xnn_u8_maxpool_minmax_ukernel_9p8x__sse2_c16;
    u8_maxpool_config.init.u8 = xnn_init_u8_minmax_scalar_params;
    u8_maxpool_config.first_pass_tile_size = 9;
    u8_maxpool_config.remainder_pass_tile_size = 8;
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    u8_maxpool_config.ukernel = (xnn_maxpool_ukernel_fn) xnn_u8_maxpool_minmax_ukernel_9p8x__wasmsimd_c16;
    u8_maxpool_config.init.u8 = xnn_init_u8_minmax_scalar_params;
    u8_maxpool_config.first_pass_tile_size = 9;
    u8_maxpool_config.remainder_pass_tile_size = 8;
  #else
    u8_maxpool_config.ukernel = (xnn_maxpool_ukernel_fn) xnn_u8_maxpool_minmax_ukernel_9p8x__scalar_c1;
    u8_maxpool_config.init.u8 = xnn_init_u8_minmax_scalar_params;
    u8_maxpool_config.first_pass_tile_size = 9;
    u8_maxpool_config.remainder_pass_tile_size = 8;
  #endif
}

const struct xnn_maxpool_config* xnn_init_f16_maxpool_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL || !xnn_is_f16_compatible_config(hardware_config)) {
    return NULL;
  }
  XNN_INIT_ONCE(f16_maxpool);
  return &f16_maxpool_config;
}

const struct xnn_maxpool_config* xnn_init_f32_maxpool_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  XNN_INIT_ONCE(f32_maxpool);
  return &f32_maxpool_config;
}

const struct xnn_maxpool_config* xnn_init_s8_maxpool_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  XNN_INIT_ONCE(s8_maxpool);
  return &s8_maxpool_config;
}

const struct xnn_maxpool_config* xnn_init_u8_maxpool_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  XNN_INIT_ONCE(u8_maxpool);
  return &u8_maxpool_config;
}
