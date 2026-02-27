// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "include/xnnpack.h"
#include "src/xnnpack/common.h"
#include "src/xnnpack/config-types.h"
#include "src/xnnpack/config.h"
#include "src/xnnpack/hardware-config.h"
#include "src/xnnpack/init-once.h"
#include "src/xnnpack/microfnptr.h"
#include "src/xnnpack/log.h"
#include "src/xnnpack/raddstoreexpminusmax.h"

static const int default_config = 0;
static const int consistent_config = 1;

static struct xnn_raddstoreexpminusmax_config f16_raddstoreexpminusmax_config = {0};
static struct xnn_raddstoreexpminusmax_config f32_raddstoreexpminusmax_config[2] = {0};

XNN_INIT_ONCE_GUARD(f16_raddstoreexpminusmax);
XNN_INIT_ONCE_GUARD(f32_raddstoreexpminusmax);

// Macros to log the microkernel names if and when they are registered.
#define XNN_INIT_RADDSTOREEXPMINUSMAX_UKERNEL(ukernel) \
  (xnn_raddstoreexpminusmax_ukernel_fn) ukernel;       \
  xnn_log_info("Using raddstoreexpminusmax microkernel '%s'.", #ukernel);

static void init_f16_raddstoreexpminusmax_config(void) {
  #if XNN_ENABLE_ARM_FP16_SCALAR && XNN_ENABLE_ARM_FP16_VECTOR && XNN_ARCH_ARM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    (void) hardware_config;  // May be unused.
    if (hardware_config->arch_flags & xnn_arch_arm_neon_fp16_arith) {
      f16_raddstoreexpminusmax_config.ukernel = XNN_INIT_RADDSTOREEXPMINUSMAX_UKERNEL(xnn_f16_raddstoreexpminusmax_ukernel__neonfp16arith_rr2_p2_u32_acc2);
    }
  #elif XNN_ENABLE_ARM_FP16_VECTOR && XNN_ARCH_ARM64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    (void) hardware_config;  // May be unused.
    if (hardware_config->arch_flags & xnn_arch_arm_neon_fp16_arith) {
      f16_raddstoreexpminusmax_config.ukernel = XNN_INIT_RADDSTOREEXPMINUSMAX_UKERNEL(xnn_f16_raddstoreexpminusmax_ukernel__neonfp16arith_rr2_p2_u32_acc2);
    }
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    (void) hardware_config;  // May be unused.
    #if XNN_ENABLE_AVX2
      if (hardware_config->arch_flags & xnn_arch_x86_avx2) {
        f16_raddstoreexpminusmax_config.ukernel = XNN_INIT_RADDSTOREEXPMINUSMAX_UKERNEL(xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_u32);
      }
    #endif
  #endif
}

static void init_f32_raddstoreexpminusmax_config_impl(struct xnn_raddstoreexpminusmax_config* config, bool consistent_arithmetic) {
  #if XNN_ARCH_ARM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    (void) hardware_config;  // May be unused.
    if (hardware_config->arch_flags & xnn_arch_arm_neon) {
      config->ukernel = XNN_INIT_RADDSTOREEXPMINUSMAX_UKERNEL(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_lut64_p2_u16_acc2);
    } else {
      config->ukernel = XNN_INIT_RADDSTOREEXPMINUSMAX_UKERNEL(xnn_f32_raddstoreexpminusmax_ukernel__scalar_rr2_p5_u4_acc2);
    }
  #elif XNN_ARCH_ARM64
    config->ukernel =
      XNN_INIT_RADDSTOREEXPMINUSMAX_UKERNEL(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_lut64_p2_u16_acc2);
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    (void) hardware_config;  // May be unused.
    #if XNN_ENABLE_AVX512F
      if (!consistent_arithmetic && (hardware_config->arch_flags & xnn_arch_x86_avx512f)) {
        config->ukernel = XNN_INIT_RADDSTOREEXPMINUSMAX_UKERNEL(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_rr2_p5_u64_acc2);
      } else
    #endif
    #if XNN_ENABLE_AVX256SKX
      if (!consistent_arithmetic && (hardware_config->arch_flags & xnn_arch_x86_avx256skx)) {
        config->ukernel = XNN_INIT_RADDSTOREEXPMINUSMAX_UKERNEL(xnn_f32_raddstoreexpminusmax_ukernel__avx256skx_rr2_p5_u32_acc2);
      } else
    #endif
    #if XNN_ENABLE_AVX2
      if (!consistent_arithmetic && (hardware_config->arch_flags & xnn_arch_x86_avx2)) {
        config->ukernel = XNN_INIT_RADDSTOREEXPMINUSMAX_UKERNEL(xnn_f32_raddstoreexpminusmax_ukernel__avx2_rr2_p5_u32_acc2);
      } else
    #endif
    #if XNN_ENABLE_SSE2
      if (hardware_config->arch_flags & xnn_arch_x86_sse2) {
        config->ukernel = XNN_INIT_RADDSTOREEXPMINUSMAX_UKERNEL(xnn_f32_raddstoreexpminusmax_ukernel__sse2_rr2_p5_u16_acc2);
      } else
    #endif
    {
      config->ukernel = XNN_INIT_RADDSTOREEXPMINUSMAX_UKERNEL(xnn_f32_raddstoreexpminusmax_ukernel__scalar_rr2_p5_u4_acc2);
    }
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    #if XNN_ARCH_WASMRELAXEDSIMD
      config->ukernel = XNN_INIT_RADDSTOREEXPMINUSMAX_UKERNEL(xnn_f32_raddstoreexpminusmax_ukernel__wasmrelaxedsimd_rr2_p5_u16_acc2);
    #else
      config->ukernel = XNN_INIT_RADDSTOREEXPMINUSMAX_UKERNEL(xnn_f32_raddstoreexpminusmax_ukernel__wasmsimd_rr2_p5_u16_acc2);
    #endif
  #elif XNN_ARCH_RISCV && XNN_ENABLE_RISCV_VECTOR
    config->ukernel = XNN_INIT_RADDSTOREEXPMINUSMAX_UKERNEL(xnn_f32_raddstoreexpminusmax_ukernel__rvv_rr2_p6_u4v);
  #elif XNN_ARCH_HEXAGON && XNN_ENABLE_HVX
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->arch_flags & xnn_arch_hvx) {
      config->ukernel = XNN_INIT_RADDSTOREEXPMINUSMAX_UKERNEL(xnn_f32_raddstoreexpminusmax_ukernel__hvx_rr2_p5_u128_acc2);
    }
  #else
    config->ukernel = XNN_INIT_RADDSTOREEXPMINUSMAX_UKERNEL(xnn_f32_raddstoreexpminusmax_ukernel__scalar_rr2_p5_u4_acc2);
  #endif
}

static void init_f32_raddstoreexpminusmax_config(void) {
  init_f32_raddstoreexpminusmax_config_impl(&f32_raddstoreexpminusmax_config[default_config], false);
  init_f32_raddstoreexpminusmax_config_impl(&f32_raddstoreexpminusmax_config[consistent_config], true);
}

static bool is_f16_compatible_config(const struct xnn_hardware_config* hardware_config) {
  #if (XNN_ARCH_ARM && XNN_ENABLE_ARM_FP16_VECTOR && XNN_ENABLE_ARM_FP16_SCALAR) || (XNN_ENABLE_ARM_FP16_VECTOR && XNN_ARCH_ARM64)
    return (hardware_config->arch_flags & xnn_arch_arm_neon_fp16_arith);
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    return (hardware_config->arch_flags & xnn_arch_x86_avx2);
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

const struct xnn_raddstoreexpminusmax_config* xnn_init_f32_raddstoreexpminusmax_config(uint32_t flags) {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  XNN_INIT_ONCE(f32_raddstoreexpminusmax);
  if (flags & XNN_FLAG_SLOW_CONSISTENT_ARITHMETIC) {
    return &f32_raddstoreexpminusmax_config[consistent_config];
  } else {
    return &f32_raddstoreexpminusmax_config[default_config];
  }
}
