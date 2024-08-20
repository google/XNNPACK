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
#include "xnnpack/reduce.h"

static struct xnn_rmax_config f16_rmax_config = {0};
static struct xnn_rmax_config f32_rmax_config = {0};
static struct xnn_rmax_config u8_rmax_config = {0};

XNN_INIT_ONCE_GUARD(f16_rmax);
XNN_INIT_ONCE_GUARD(f32_rmax);
XNN_INIT_ONCE_GUARD(u8_rmax);

static void init_f16_rmax_config(void) {
  #if XNN_ARCH_ARM && XNN_ENABLE_ARM_FP16_VECTOR && XNN_ENABLE_ARM_FP16_SCALAR
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon_fp16_arith) {
      f16_rmax_config.ukernel = (xnn_rmax_ukernel_fn) xnn_f16_rmax_ukernel__neonfp16arith_u32_acc4;
    }
  #elif XNN_ARCH_ARM64 && XNN_ENABLE_ARM_FP16_VECTOR
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon_fp16_arith) {
      f16_rmax_config.ukernel = (xnn_rmax_ukernel_fn) xnn_f16_rmax_ukernel__neonfp16arith_u32_acc4;
    }
  #elif (XNN_ARCH_X86 || XNN_ARCH_X86_64) && !XNN_PLATFORM_MOBILE
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    #if XNN_ENABLE_AVX512FP16
      if (hardware_config->use_x86_avx512fp16) {
        f16_rmax_config.ukernel = (xnn_rmax_ukernel_fn) xnn_f16_rmax_ukernel__avx512fp16_u128_acc4;
      } else
    #endif
    if (hardware_config->use_x86_avx512skx) {
      f16_rmax_config.ukernel = (xnn_rmax_ukernel_fn) xnn_f16_rmax_ukernel__avx512skx_u64_acc4;
    } else if (hardware_config->use_x86_f16c) {
      f16_rmax_config.ukernel = (xnn_rmax_ukernel_fn) xnn_f16_rmax_ukernel__f16c_u32;
    }
  #else
    f16_rmax_config.ukernel = (xnn_rmax_ukernel_fn) xnn_f16_rmax_ukernel__scalar_u2_acc2;
  #endif
}

static void init_f32_rmax_config(void) {
  #if XNN_ARCH_ARM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon) {
      f32_rmax_config.ukernel = (xnn_rmax_ukernel_fn) xnn_f32_rmax_ukernel__neon_u16_acc4;
    } else if (!XNN_PLATFORM_MOBILE) {
      f32_rmax_config.ukernel = (xnn_rmax_ukernel_fn) xnn_f32_rmax_ukernel__scalar_u4_acc4;
    }
  #elif XNN_ARCH_ARM64
    f32_rmax_config.ukernel = (xnn_rmax_ukernel_fn) xnn_f32_rmax_ukernel__neon_u16_acc4;
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_x86_avx512f) {
      f32_rmax_config.ukernel = (xnn_rmax_ukernel_fn) xnn_f32_rmax_ukernel__avx512f_u64_acc4;
    } else if (hardware_config->use_x86_avx) {
      f32_rmax_config.ukernel = (xnn_rmax_ukernel_fn) xnn_f32_rmax_ukernel__avx_u32_acc4;
    } else {
      f32_rmax_config.ukernel = (xnn_rmax_ukernel_fn) xnn_f32_rmax_ukernel__sse_u16_acc4;
    }
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    f32_rmax_config.ukernel = (xnn_rmax_ukernel_fn) xnn_f32_rmax_ukernel__wasmsimd_pminmax_u16_acc4;
  #elif XNN_ARCH_WASM
    f32_rmax_config.ukernel = (xnn_rmax_ukernel_fn) xnn_f32_rmax_ukernel__scalar_u4_acc4;
  #elif XNN_ARCH_RISCV
    #if XNN_ENABLE_RISCV_VECTOR
      f32_rmax_config.ukernel = (xnn_rmax_ukernel_fn) xnn_f32_rmax_ukernel__rvv_u8v;
    #else
      f32_rmax_config.ukernel = (xnn_rmax_ukernel_fn) xnn_f32_rmax_ukernel__scalar_u4_acc4;
    #endif
  #else
    f32_rmax_config.ukernel = (xnn_rmax_ukernel_fn) xnn_f32_rmax_ukernel__scalar_u4_acc4;
  #endif
}

static void init_u8_rmax_config(void) {
  #if XNN_ARCH_ARM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon) {
      u8_rmax_config.ukernel = (xnn_rmax_ukernel_fn) xnn_u8_rmax_ukernel__neon_u16;
    } else if (!XNN_PLATFORM_MOBILE) {
      u8_rmax_config.ukernel = (xnn_rmax_ukernel_fn) xnn_u8_rmax_ukernel__scalar_u2;
    }
  #elif XNN_ARCH_ARM64
    u8_rmax_config.ukernel = (xnn_rmax_ukernel_fn) xnn_u8_rmax_ukernel__neon_u16;
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    u8_rmax_config.ukernel = (xnn_rmax_ukernel_fn) xnn_u8_rmax_ukernel__sse2_u16;
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    u8_rmax_config.ukernel = (xnn_rmax_ukernel_fn) xnn_u8_rmax_ukernel__scalar_u2;
  #else
    u8_rmax_config.ukernel = (xnn_rmax_ukernel_fn) xnn_u8_rmax_ukernel__scalar_u2;
  #endif

}

const struct xnn_rmax_config* xnn_init_f16_rmax_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL || !xnn_is_f16_compatible_config(hardware_config)) {
    return NULL;
  }
  XNN_INIT_ONCE(f16_rmax);
  return &f16_rmax_config;
}

const struct xnn_rmax_config* xnn_init_f32_rmax_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  XNN_INIT_ONCE(f32_rmax);
  return &f32_rmax_config;
}

const struct xnn_rmax_config* xnn_init_u8_rmax_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  XNN_INIT_ONCE(u8_rmax);
  return &u8_rmax_config;
}
