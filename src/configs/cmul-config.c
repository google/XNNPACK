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
#include "xnnpack/vbinary.h"

#if (XNN_ARCH_ARM || XNN_ARCH_ARM64) && XNN_ENABLE_ARM_FP16_VECTOR
  static struct xnn_cmul_config f16_cmul_config = {0};
#endif
static struct xnn_cmul_config f32_cmul_config = {0};


#if (XNN_ARCH_ARM || XNN_ARCH_ARM64) && XNN_ENABLE_ARM_FP16_VECTOR
  XNN_INIT_ONCE_GUARD(f16_cmul);
#endif
XNN_INIT_ONCE_GUARD(f32_cmul);


#if (XNN_ARCH_ARM || XNN_ARCH_ARM64) && XNN_ENABLE_ARM_FP16_VECTOR
  static void init_f16_cmul_config(void) {
      f16_cmul_config.ukernel = (xnn_vbinary_ukernel_fn) xnn_f16_vcmul_ukernel__neonfp16arith_u16;
  }
#endif

static void init_f32_cmul_config(void) {
  #if XNN_ARCH_ARM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon) {
      f32_cmul_config.ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vcmul_ukernel__neon_u8;
    } else if (!XNN_PLATFORM_MOBILE) {
      f32_cmul_config.ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vcmul_ukernel__scalar_u4;
    }
  #elif XNN_ARCH_ARM64
    f32_cmul_config.ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vcmul_ukernel__neon_u8;
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    #if XNN_ENABLE_AVX512F
      if (!XNN_PLATFORM_MOBILE && hardware_config->use_x86_avx512f) {
        f32_cmul_config.ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vcmul_ukernel__avx512f_u32;
      } else
    #endif
    if (hardware_config->use_x86_fma3) {
      f32_cmul_config.ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vcmul_ukernel__fma3_u16;
    } else {
      f32_cmul_config.ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vcmul_ukernel__sse_u8;
    }
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    f32_cmul_config.ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vcmul_ukernel__wasmsimd_u8;
  #elif XNN_ARCH_RISCV && XNN_ENABLE_RISCV_VECTOR
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    f32_cmul_config.ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vcmul_ukernel__rvv_u2v;
  #else
    f32_cmul_config.ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vcmul_ukernel__scalar_u4;
  #endif
}

const struct xnn_cmul_config* xnn_init_f16_cmul_config() {
  #if (XNN_ARCH_ARM || XNN_ARCH_ARM64) && XNN_ENABLE_ARM_FP16_VECTOR
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    if (hardware_config == NULL || !xnn_is_f16_compatible_config(hardware_config)) {
      return NULL;
    }
    XNN_INIT_ONCE(f16_cmul);
    return &f16_cmul_config;
  #else
    return NULL;
  #endif
}

const struct xnn_cmul_config* xnn_init_f32_cmul_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  XNN_INIT_ONCE(f32_cmul);
  return &f32_cmul_config;
}
