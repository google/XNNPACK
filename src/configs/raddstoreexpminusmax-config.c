// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/config.h"
#include "src/xnnpack/init-once.h"
#include "src/xnnpack/microfnptr.h"
#include "src/xnnpack/microparams-init.h"
#include "src/xnnpack/raddstoreexpminusmax.h"

static struct xnn_raddstoreexpminusmax_config f16_raddstoreexpminusmax_config = {0};
static struct xnn_raddstoreexpminusmax_config f32_raddstoreexpminusmax_config = {0};

XNN_INIT_ONCE_GUARD(f16_raddstoreexpminusmax);
XNN_INIT_ONCE_GUARD(f32_raddstoreexpminusmax);

static void init_f16_raddstoreexpminusmax_config(void) {
  #if XNN_ARCH_ARM && XNN_ENABLE_ARM_FP16_VECTOR && XNN_ENABLE_ARM_FP16_SCALAR
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon_fp16_arith) {
      f16_raddstoreexpminusmax_config.ukernel = (xnn_raddstoreexpminusmax_ukernel_fn) xnn_f16_raddstoreexpminusmax_ukernel__neonfp16arith_rr2_p2_u32;
    }
  #elif XNN_ARCH_ARM64 && XNN_ENABLE_ARM_FP16_VECTOR
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon_fp16_arith) {
      f16_raddstoreexpminusmax_config.ukernel = (xnn_raddstoreexpminusmax_ukernel_fn) xnn_f16_raddstoreexpminusmax_ukernel__neonfp16arith_rr2_p2_u32;
    }
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_x86_avx2) {
      f16_raddstoreexpminusmax_config.ukernel = (xnn_raddstoreexpminusmax_ukernel_fn) xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_u32;
    }
  #endif
}

static void init_f32_raddstoreexpminusmax_config(void) {
  #if XNN_ARCH_ARM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon) {
      f32_raddstoreexpminusmax_config.ukernel =
        (xnn_raddstoreexpminusmax_ukernel_fn) xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_lut64_p2_u16_acc2;
    } else {
      f32_raddstoreexpminusmax_config.ukernel =
        (xnn_raddstoreexpminusmax_ukernel_fn) xnn_f32_raddstoreexpminusmax_ukernel__scalar_rr2_p5_u4_acc2;
    }
  #elif XNN_ARCH_ARM64
    f32_raddstoreexpminusmax_config.ukernel =
      (xnn_raddstoreexpminusmax_ukernel_fn) xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_lut64_p2_u16_acc2;
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    #if XNN_ENABLE_AVX512F
      if (hardware_config->use_x86_avx512f) {
        f32_raddstoreexpminusmax_config.ukernel = (xnn_raddstoreexpminusmax_ukernel_fn) xnn_f32_raddstoreexpminusmax_ukernel__avx512f_rr2_p5_u64_acc2;
      } else
    #endif
    #if XNN_ENABLE_AVX256SKX
      if (hardware_config->use_x86_avx256skx) {
        f32_raddstoreexpminusmax_config.ukernel = (xnn_raddstoreexpminusmax_ukernel_fn) xnn_f32_raddstoreexpminusmax_ukernel__avx256skx_rr2_p5_u32_acc2;
      } else
    #endif
    if (hardware_config->use_x86_avx2) {
      f32_raddstoreexpminusmax_config.ukernel = (xnn_raddstoreexpminusmax_ukernel_fn) xnn_f32_raddstoreexpminusmax_ukernel__avx2_rr2_p5_u32_acc2;
    } else {
      f32_raddstoreexpminusmax_config.ukernel = (xnn_raddstoreexpminusmax_ukernel_fn) xnn_f32_raddstoreexpminusmax_ukernel__sse2_rr2_p5_u16_acc2;
    }
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    #if XNN_ARCH_WASMRELAXEDSIMD
      f32_raddstoreexpminusmax_config.ukernel =
        (xnn_raddstoreexpminusmax_ukernel_fn) xnn_f32_raddstoreexpminusmax_ukernel__wasmrelaxedsimd_rr2_p5_u16_acc2;
    #else
      f32_raddstoreexpminusmax_config.ukernel =
        (xnn_raddstoreexpminusmax_ukernel_fn) xnn_f32_raddstoreexpminusmax_ukernel__wasmsimd_rr2_p5_u16_acc2;
    #endif
  #elif XNN_ARCH_RISCV && XNN_ENABLE_RISCV_VECTOR
    f32_raddstoreexpminusmax_config.ukernel = (xnn_raddstoreexpminusmax_ukernel_fn) xnn_f32_raddstoreexpminusmax_ukernel__rvv_rr2_p6_u4v;
  #elif XNN_ARCH_HEXAGON && XNN_ENABLE_HVX
    f32_raddstoreexpminusmax_config.ukernel = (xnn_raddstoreexpminusmax_ukernel_fn) xnn_f32_raddstoreexpminusmax_ukernel__hvx_rr2_p5_u128_acc2;
  #else
    f32_raddstoreexpminusmax_config.ukernel = (xnn_raddstoreexpminusmax_ukernel_fn) xnn_f32_raddstoreexpminusmax_ukernel__scalar_rr2_p5_u4_acc2;
  #endif
}

static bool is_f16_compatible_config(const struct xnn_hardware_config* hardware_config) {
  #if (XNN_ARCH_ARM && XNN_ENABLE_ARM_FP16_VECTOR && XNN_ENABLE_ARM_FP16_SCALAR) || (XNN_ARCH_ARM64 && XNN_ENABLE_ARM_FP16_VECTOR)
    return hardware_config->use_arm_neon_fp16_arith;
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    return hardware_config->use_x86_avx2;
  #else
    return false;
  #endif
}

const struct xnn_raddstoreexpminusmax_config* xnn_init_f16_raddstoreexpminusmax_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL || !is_f16_compatible_config(hardware_config)) {
    return NULL;
  }
  XNN_INIT_ONCE(f16_raddstoreexpminusmax);
  return &f16_raddstoreexpminusmax_config;
}

const struct xnn_raddstoreexpminusmax_config* xnn_init_f32_raddstoreexpminusmax_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  XNN_INIT_ONCE(f32_raddstoreexpminusmax);
  return &f32_raddstoreexpminusmax_config;
}
