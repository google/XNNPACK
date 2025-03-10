// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/config.h"
#include "src/xnnpack/init-once.h"
#include "src/xnnpack/maxpool.h"
#include "src/xnnpack/microfnptr.h"
#include "src/xnnpack/microparams-init.h"

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
      f16_maxpool_config.ukernel = (xnn_maxpool_ukernel_fn) xnn_f16_maxpool_minmax_ukernel_9p__neonfp16arith_u8;
      f16_maxpool_config.init.f16 = xnn_init_f16_minmax_scalar_params;
    }
  #elif XNN_ARCH_ARM64 && XNN_ENABLE_ARM_FP16_VECTOR
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon_fp16_arith) {
      f16_maxpool_config.ukernel = (xnn_maxpool_ukernel_fn) xnn_f16_maxpool_minmax_ukernel_9p__neonfp16arith_u8;
      f16_maxpool_config.init.f16 = xnn_init_f16_minmax_scalar_params;
    }
  #elif (XNN_ARCH_X86 || XNN_ARCH_X86_64) && !XNN_PLATFORM_MOBILE
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_x86_avx2) {
      f16_maxpool_config.ukernel = (xnn_maxpool_ukernel_fn) xnn_f16_maxpool_minmax_ukernel_9p__avx2_u16;
      f16_maxpool_config.init.f16 = xnn_init_f16_minmax_scalar_params;
    } else {
      f16_maxpool_config.ukernel = (xnn_maxpool_ukernel_fn) xnn_f16_maxpool_minmax_ukernel_9p__sse41_u8;
      f16_maxpool_config.init.f16 = xnn_init_f16_minmax_scalar_params;
    }
  #endif
}

static void init_f32_maxpool_config(void) {
  #if XNN_ARCH_ARM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon) {
      f32_maxpool_config.ukernel = (xnn_maxpool_ukernel_fn) xnn_f32_maxpool_minmax_ukernel_9p__neon_u4;
      f32_maxpool_config.init.f32 = xnn_init_f32_minmax_scalar_params;
    } else if (!XNN_PLATFORM_MOBILE) {
      f32_maxpool_config.ukernel = (xnn_maxpool_ukernel_fn) xnn_f32_maxpool_minmax_ukernel_9p__scalar_u1;
      f32_maxpool_config.init.f32 = xnn_init_f32_minmax_scalar_params;
    }
  #elif XNN_ARCH_ARM64
    f32_maxpool_config.ukernel = (xnn_maxpool_ukernel_fn) xnn_f32_maxpool_minmax_ukernel_9p__neon_u4;
    f32_maxpool_config.init.f32 = xnn_init_f32_minmax_scalar_params;
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    f32_maxpool_config.ukernel = (xnn_maxpool_ukernel_fn) xnn_f32_maxpool_minmax_ukernel_9p__sse2_u4;
    f32_maxpool_config.init.f32 = xnn_init_f32_minmax_scalar_params;
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    f32_maxpool_config.ukernel = (xnn_maxpool_ukernel_fn) xnn_f32_maxpool_minmax_ukernel_9p__wasmsimd_u4;
    f32_maxpool_config.init.f32 = xnn_init_f32_minmax_scalar_params;
  #elif XNN_ARCH_WASM
    f32_maxpool_config.ukernel = (xnn_maxpool_ukernel_fn) xnn_f32_maxpool_minmax_ukernel_9p__scalar_u1;
    f32_maxpool_config.init.f32 = xnn_init_f32_minmax_scalar_params;
  #elif XNN_ARCH_RISCV && XNN_ENABLE_RISCV_VECTOR
    f32_maxpool_config.ukernel = (xnn_maxpool_ukernel_fn) xnn_f32_maxpool_minmax_ukernel_9p__rvv_u2v;
    f32_maxpool_config.init.f32 = xnn_init_f32_minmax_scalar_params;
  #else
    f32_maxpool_config.ukernel = (xnn_maxpool_ukernel_fn) xnn_f32_maxpool_minmax_ukernel_9p__scalar_u1;
    f32_maxpool_config.init.f32 = xnn_init_f32_minmax_scalar_params;
  #endif
}

static void init_s8_maxpool_config(void) {
  #if XNN_ARCH_ARM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon) {
      s8_maxpool_config.ukernel = (xnn_maxpool_ukernel_fn) xnn_s8_maxpool_minmax_ukernel_9p__neon_u16;
      s8_maxpool_config.init.s8 = xnn_init_s8_minmax_scalar_params;
    } else if (!XNN_PLATFORM_MOBILE) {
      s8_maxpool_config.ukernel = (xnn_maxpool_ukernel_fn) xnn_s8_maxpool_minmax_ukernel_9p__scalar_u1;
      s8_maxpool_config.init.s8 = xnn_init_s8_minmax_scalar_params;
    }
  #elif XNN_ARCH_ARM64
    s8_maxpool_config.ukernel = (xnn_maxpool_ukernel_fn) xnn_s8_maxpool_minmax_ukernel_9p__neon_u16;
    s8_maxpool_config.init.s8 = xnn_init_s8_minmax_scalar_params;
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_x86_sse4_1) {
      s8_maxpool_config.ukernel = (xnn_maxpool_ukernel_fn) xnn_s8_maxpool_minmax_ukernel_9p__sse41_u16;
      s8_maxpool_config.init.s8 = xnn_init_s8_minmax_scalar_params;
    } else {
      s8_maxpool_config.ukernel = (xnn_maxpool_ukernel_fn) xnn_s8_maxpool_minmax_ukernel_9p__scalar_u1;
      s8_maxpool_config.init.s8 = xnn_init_s8_minmax_scalar_params;
    }
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    s8_maxpool_config.ukernel = (xnn_maxpool_ukernel_fn) xnn_s8_maxpool_minmax_ukernel_9p__wasmsimd_u16;
    s8_maxpool_config.init.s8 = xnn_init_s8_minmax_scalar_params;
  #else
    s8_maxpool_config.ukernel = (xnn_maxpool_ukernel_fn) xnn_s8_maxpool_minmax_ukernel_9p__scalar_u1;
    s8_maxpool_config.init.s8 = xnn_init_s8_minmax_scalar_params;
  #endif
}

static void init_u8_maxpool_config(void) {
  #if XNN_ARCH_ARM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon) {
      u8_maxpool_config.ukernel = (xnn_maxpool_ukernel_fn) xnn_u8_maxpool_minmax_ukernel_9p__neon_u16;
      u8_maxpool_config.init.u8 = xnn_init_u8_minmax_scalar_params;
    } else if (!XNN_PLATFORM_MOBILE) {
      u8_maxpool_config.ukernel = (xnn_maxpool_ukernel_fn) xnn_u8_maxpool_minmax_ukernel_9p__scalar_u1;
      u8_maxpool_config.init.u8 = xnn_init_u8_minmax_scalar_params;
    }
  #elif XNN_ARCH_ARM64
    u8_maxpool_config.ukernel = (xnn_maxpool_ukernel_fn) xnn_u8_maxpool_minmax_ukernel_9p__neon_u16;
    u8_maxpool_config.init.u8 = xnn_init_u8_minmax_scalar_params;
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    u8_maxpool_config.ukernel = (xnn_maxpool_ukernel_fn) xnn_u8_maxpool_minmax_ukernel_9p__sse2_u16;
    u8_maxpool_config.init.u8 = xnn_init_u8_minmax_scalar_params;
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    u8_maxpool_config.ukernel = (xnn_maxpool_ukernel_fn) xnn_u8_maxpool_minmax_ukernel_9p__wasmsimd_u16;
    u8_maxpool_config.init.u8 = xnn_init_u8_minmax_scalar_params;
  #else
    u8_maxpool_config.ukernel = (xnn_maxpool_ukernel_fn) xnn_u8_maxpool_minmax_ukernel_9p__scalar_u1;
    u8_maxpool_config.init.u8 = xnn_init_u8_minmax_scalar_params;
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
