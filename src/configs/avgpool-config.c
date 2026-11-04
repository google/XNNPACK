// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>

#include "src/xnnpack/avgpool.h"
#include "src/xnnpack/common.h"
#include "src/xnnpack/config-types.h"
#include "src/xnnpack/config.h"
#include "src/xnnpack/hardware-config.h"
#include "src/xnnpack/init-once.h"
#include "src/xnnpack/log.h"
#include "src/xnnpack/microfnptr.h"
#include "src/xnnpack/microparams-init.h"

static struct xnn_avgpool_config f16_avgpool_config = {0};
static struct xnn_avgpool_config f32_avgpool_config = {0};
static struct xnn_avgpool_config qs8_avgpool_config = {0};
static struct xnn_avgpool_config qu8_avgpool_config = {0};

XNN_INIT_ONCE_GUARD(f16_avgpool);
XNN_INIT_ONCE_GUARD(f32_avgpool);
XNN_INIT_ONCE_GUARD(qs8_avgpool);
XNN_INIT_ONCE_GUARD(qu8_avgpool);

// Macros to log the microkernel names if and when they are registered.
#define XNN_INIT_AVGPOOL_UKERNEL(ukernel) \
  (xnn_avgpool_ukernel_fn) ukernel;      \
  xnn_log_info("Using avgpool microkernel '%s'.", #ukernel);

static void init_f16_avgpool_config(void) {
  f16_avgpool_config.ukernel = XNN_INIT_AVGPOOL_UKERNEL(xnn_f16_f32acc_avgpool_minmax_ukernel_9p__scalar_u1);
  f16_avgpool_config.init.f16 = xnn_init_f16_scaleminmax_scalar_params;
  f16_avgpool_config.primary_tile = 9;
  f16_avgpool_config.channel_tile = 1;

  #if XNN_ARCH_ARM || XNN_ARCH_ARM64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    (void) hardware_config;  // May be unused.
    #if XNN_ENABLE_ARM_FP16_VECTOR
      if (hardware_config->arch_flags & xnn_arch_arm_neon_fp16_arith) {
        f16_avgpool_config.ukernel = XNN_INIT_AVGPOOL_UKERNEL(xnn_f16_avgpool_minmax_ukernel_9p__neonfp16arith_u8);
        f16_avgpool_config.init.f16 = xnn_init_f16_scaleminmax_scalar_params;
        f16_avgpool_config.primary_tile = 9;
        f16_avgpool_config.channel_tile = 8;
      } else
    #endif
    if (hardware_config->arch_flags & xnn_arch_arm_neon_fp16) {
      f16_avgpool_config.ukernel = XNN_INIT_AVGPOOL_UKERNEL(xnn_f16_f32acc_avgpool_minmax_ukernel_9p__neonfp16_u4);
      f16_avgpool_config.init.f16 = xnn_init_f16_scaleminmax_scalar_params;
      f16_avgpool_config.primary_tile = 9;
      f16_avgpool_config.channel_tile = 4;
    }
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    (void) hardware_config;  // May be unused.
    #if XNN_ENABLE_AVX512FP16
      if (hardware_config->arch_flags & xnn_arch_x86_avx512fp16) {
        f16_avgpool_config.ukernel = XNN_INIT_AVGPOOL_UKERNEL(xnn_f16_avgpool_minmax_ukernel_9p__avx512fp16_u32);
        f16_avgpool_config.init.f16 = xnn_init_f16_scaleminmax_scalar_params;
        f16_avgpool_config.primary_tile = 9;
        f16_avgpool_config.channel_tile = 32;
      } else
    #endif
    #if XNN_ENABLE_AVX512F
      if (hardware_config->arch_flags & xnn_arch_x86_avx512f) {
        f16_avgpool_config.ukernel = XNN_INIT_AVGPOOL_UKERNEL(xnn_f16_f32acc_avgpool_minmax_ukernel_9p__avx512f_u16);
        f16_avgpool_config.init.f16 = xnn_init_f16_scaleminmax_scalar_params;
        f16_avgpool_config.primary_tile = 9;
        f16_avgpool_config.channel_tile = 16;
      } else
    #endif
    #if XNN_ENABLE_F16C
      if (hardware_config->arch_flags & xnn_arch_x86_f16c) {
        f16_avgpool_config.ukernel = XNN_INIT_AVGPOOL_UKERNEL(xnn_f16_avgpool_minmax_ukernel_9p__f16c_u8);
        f16_avgpool_config.init.f16 = xnn_init_f16_scaleminmax_scalar_params;
        f16_avgpool_config.primary_tile = 9;
        f16_avgpool_config.channel_tile = 8;
      } else
    #endif
    ;  // no f16 support
  #elif XNN_ARCH_RISCV && XNN_ENABLE_RISCV_FP16_VECTOR
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    (void) hardware_config;  // May be unused.
    if (hardware_config->arch_flags & xnn_arch_riscv_vector_fp16_arith) {
      f16_avgpool_config.ukernel = XNN_INIT_AVGPOOL_UKERNEL(xnn_f16_avgpool_minmax_ukernel_9p__rvvfp16arith_u2v);
      f16_avgpool_config.init.f16 = xnn_init_f16_scaleminmax_scalar_params;
      f16_avgpool_config.primary_tile = 9;
      f16_avgpool_config.channel_tile = 2 * hardware_config->vlenb / sizeof(xnn_float16);
    }
  #elif XNN_ARCH_HEXAGON && XNN_ENABLE_HVX
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    (void) hardware_config;  // May be unused.
    if (hardware_config->arch_flags & xnn_arch_hvx) {
      f16_avgpool_config.ukernel = XNN_INIT_AVGPOOL_UKERNEL(xnn_f16_f32acc_avgpool_minmax_ukernel_9p__hvx_u32);
      f16_avgpool_config.init.f16 = xnn_init_f16_scaleminmax_scalar_params;
      f16_avgpool_config.primary_tile = 9;
      f16_avgpool_config.channel_tile = 32;
    }
  #elif XNN_ARCH_WASMRELAXEDSIMDFP16
    f16_avgpool_config.ukernel = XNN_INIT_AVGPOOL_UKERNEL(xnn_f16_avgpool_minmax_ukernel_9p__wasmrelaxedsimdfp16_u8);
    f16_avgpool_config.init.f16 = xnn_init_f16_scaleminmax_scalar_params;
    f16_avgpool_config.primary_tile = 9;
    f16_avgpool_config.channel_tile = 8;
  #elif XNN_ARCH_WASMRELAXEDSIMD
    f16_avgpool_config.ukernel = XNN_INIT_AVGPOOL_UKERNEL(xnn_f16_f32acc_avgpool_minmax_ukernel_9p__wasmrelaxedsimd_u4);
    f16_avgpool_config.init.f16 = xnn_init_f16_scaleminmax_scalar_params;
    f16_avgpool_config.primary_tile = 9;
    f16_avgpool_config.channel_tile = 4;
  #elif XNN_ARCH_WASMSIMD
    f16_avgpool_config.ukernel = XNN_INIT_AVGPOOL_UKERNEL(xnn_f16_f32acc_avgpool_minmax_ukernel_9p__wasmsimd_u4);
    f16_avgpool_config.init.f16 = xnn_init_f16_scaleminmax_scalar_params;
    f16_avgpool_config.primary_tile = 9;
    f16_avgpool_config.channel_tile = 4;
  #endif
}

static void init_f32_avgpool_config(void) {
  #if XNN_ARCH_ARM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    (void) hardware_config;  // May be unused.
    if (hardware_config->arch_flags & xnn_arch_arm_neon) {
      f32_avgpool_config.ukernel = XNN_INIT_AVGPOOL_UKERNEL(xnn_f32_avgpool_minmax_ukernel_9p__neon_u4);
      f32_avgpool_config.init.f32 = xnn_init_f32_scaleminmax_scalar_params;
      f32_avgpool_config.primary_tile = 9;
      f32_avgpool_config.channel_tile = 4;
    } else {
      f32_avgpool_config.ukernel = XNN_INIT_AVGPOOL_UKERNEL(xnn_f32_avgpool_minmax_ukernel_9p__scalar_u1);
      f32_avgpool_config.init.f32 = xnn_init_f32_scaleminmax_scalar_params;
      f32_avgpool_config.primary_tile = 9;
      f32_avgpool_config.channel_tile = 1;
    }
  #elif XNN_ARCH_ARM64
    f32_avgpool_config.ukernel = XNN_INIT_AVGPOOL_UKERNEL(xnn_f32_avgpool_minmax_ukernel_9p__neon_u4);
    f32_avgpool_config.init.f32 = xnn_init_f32_scaleminmax_scalar_params;
    f32_avgpool_config.primary_tile = 9;
    f32_avgpool_config.channel_tile = 4;
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    (void) hardware_config;  // May be unused.
    #if XNN_ENABLE_AVX512F
      if (hardware_config->arch_flags & xnn_arch_x86_avx512f) {
        f32_avgpool_config.ukernel = XNN_INIT_AVGPOOL_UKERNEL(xnn_f32_avgpool_minmax_ukernel_9p__avx512f_u16);
        f32_avgpool_config.init.f32 = xnn_init_f32_scaleminmax_scalar_params;
        f32_avgpool_config.primary_tile = 9;
        f32_avgpool_config.channel_tile = 16;
      } else
    #endif
    #if XNN_ENABLE_AVX
      if (hardware_config->arch_flags & xnn_arch_x86_avx) {
        f32_avgpool_config.ukernel = XNN_INIT_AVGPOOL_UKERNEL(xnn_f32_avgpool_minmax_ukernel_9p__avx_u8);
        f32_avgpool_config.init.f32 = xnn_init_f32_scaleminmax_scalar_params;
        f32_avgpool_config.primary_tile = 9;
        f32_avgpool_config.channel_tile = 8;
      } else
    #endif
    #if XNN_ENABLE_SSE2
      if (hardware_config->arch_flags & xnn_arch_x86_sse2) {
        f32_avgpool_config.ukernel = XNN_INIT_AVGPOOL_UKERNEL(xnn_f32_avgpool_minmax_ukernel_9p__sse2_u4);
        f32_avgpool_config.init.f32 = xnn_init_f32_scaleminmax_scalar_params;
        f32_avgpool_config.primary_tile = 9;
        f32_avgpool_config.channel_tile = 4;
      } else
    #endif
    {
      f32_avgpool_config.ukernel = XNN_INIT_AVGPOOL_UKERNEL(xnn_f32_avgpool_minmax_ukernel_9p__scalar_u1);
      f32_avgpool_config.init.f32 = xnn_init_f32_scaleminmax_scalar_params;
      f32_avgpool_config.primary_tile = 9;
      f32_avgpool_config.channel_tile = 1;
    }
  #elif XNN_ARCH_WASMRELAXEDSIMD
    f32_avgpool_config.ukernel = XNN_INIT_AVGPOOL_UKERNEL(xnn_f32_avgpool_minmax_ukernel_9p__wasmrelaxedsimd_u4);
    f32_avgpool_config.init.f32 = xnn_init_f32_scaleminmax_scalar_params;
    f32_avgpool_config.primary_tile = 9;
    f32_avgpool_config.channel_tile = 4;
  #elif XNN_ARCH_WASMSIMD
    f32_avgpool_config.ukernel = XNN_INIT_AVGPOOL_UKERNEL(xnn_f32_avgpool_minmax_ukernel_9p__wasmsimd_u4);
    f32_avgpool_config.init.f32 = xnn_init_f32_scaleminmax_scalar_params;
    f32_avgpool_config.primary_tile = 9;
    f32_avgpool_config.channel_tile = 4;
  #elif XNN_ARCH_HEXAGON && XNN_ENABLE_HVX
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->arch_flags & xnn_arch_hvx) {
      f32_avgpool_config.ukernel = XNN_INIT_AVGPOOL_UKERNEL(xnn_f32_avgpool_minmax_ukernel_9p__hvx_u32);
      f32_avgpool_config.init.f32 = xnn_init_f32_scaleminmax_scalar_params;
      f32_avgpool_config.primary_tile = 9;
      f32_avgpool_config.channel_tile = 32;
    }
  #elif XNN_ARCH_RISCV && XNN_ENABLE_RISCV_VECTOR
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    f32_avgpool_config.ukernel = XNN_INIT_AVGPOOL_UKERNEL(xnn_f32_avgpool_minmax_ukernel_9p__rvv_u2v);
    f32_avgpool_config.init.f32 = xnn_init_f32_scaleminmax_scalar_params;
    f32_avgpool_config.primary_tile = 9;
    f32_avgpool_config.channel_tile = 2 * hardware_config->vlenb / sizeof(float);
  #else
    f32_avgpool_config.ukernel = XNN_INIT_AVGPOOL_UKERNEL(xnn_f32_avgpool_minmax_ukernel_9p__scalar_u1);
    f32_avgpool_config.init.f32 = xnn_init_f32_scaleminmax_scalar_params;
    f32_avgpool_config.primary_tile = 9;
    f32_avgpool_config.channel_tile = 1;
  #endif
}

const struct xnn_avgpool_config* xnn_init_f16_avgpool_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
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

static void init_qs8_avgpool_config(void) {
  #if XNN_ARCH_ARM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    (void) hardware_config;  // May be unused.
    if (hardware_config->arch_flags & xnn_arch_arm_neon) {
      qs8_avgpool_config.ukernel = XNN_INIT_AVGPOOL_UKERNEL(xnn_qs8_avgpool_minmax_fp32_ukernel_9p__neon_u16);
      qs8_avgpool_config.init.qs8 = xnn_init_qs8_avgpool_minmax_scalar_params;
      qs8_avgpool_config.primary_tile = 9;
      qs8_avgpool_config.channel_tile = 16;
    } else {
      qs8_avgpool_config.ukernel = XNN_INIT_AVGPOOL_UKERNEL(xnn_qs8_avgpool_minmax_fp32_ukernel_9p__scalar_fmagic_u1);
      qs8_avgpool_config.init.qs8 = xnn_init_qs8_avgpool_minmax_scalar_params;
      qs8_avgpool_config.primary_tile = 9;
      qs8_avgpool_config.channel_tile = 1;
    }
  #elif XNN_ARCH_ARM64
    qs8_avgpool_config.ukernel = XNN_INIT_AVGPOOL_UKERNEL(xnn_qs8_avgpool_minmax_fp32_ukernel_9p__neon_u16);
    qs8_avgpool_config.init.qs8 = xnn_init_qs8_avgpool_minmax_scalar_params;
    qs8_avgpool_config.primary_tile = 9;
    qs8_avgpool_config.channel_tile = 16;
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    (void) hardware_config;  // May be unused.
    #if XNN_ENABLE_AVX512SKX
      if (hardware_config->arch_flags & xnn_arch_x86_avx512skx) {
        qs8_avgpool_config.ukernel = XNN_INIT_AVGPOOL_UKERNEL(xnn_qs8_avgpool_minmax_fp32_ukernel_9p__avx512skx_u32);
        qs8_avgpool_config.init.qs8 = xnn_init_qs8_avgpool_minmax_scalar_params;
        qs8_avgpool_config.primary_tile = 9;
        qs8_avgpool_config.channel_tile = 32;
      } else
    #endif
    #if XNN_ENABLE_AVX2
      if (hardware_config->arch_flags & xnn_arch_x86_avx2) {
        qs8_avgpool_config.ukernel = XNN_INIT_AVGPOOL_UKERNEL(xnn_qs8_avgpool_minmax_fp32_ukernel_9p__avx2_u32);
        qs8_avgpool_config.init.qs8 = xnn_init_qs8_avgpool_minmax_scalar_params;
        qs8_avgpool_config.primary_tile = 9;
        qs8_avgpool_config.channel_tile = 32;
      } else
    #endif
    #if XNN_ENABLE_SSE41
      if (hardware_config->arch_flags & xnn_arch_x86_sse4_1) {
        qs8_avgpool_config.ukernel = XNN_INIT_AVGPOOL_UKERNEL(xnn_qs8_avgpool_minmax_fp32_ukernel_9p__sse41_u16);
        qs8_avgpool_config.init.qs8 = xnn_init_qs8_avgpool_minmax_scalar_params;
        qs8_avgpool_config.primary_tile = 9;
        qs8_avgpool_config.channel_tile = 16;
      } else
    #endif
    {
      qs8_avgpool_config.ukernel = XNN_INIT_AVGPOOL_UKERNEL(xnn_qs8_avgpool_minmax_fp32_ukernel_9p__scalar_fmagic_u1);
      qs8_avgpool_config.init.qs8 = xnn_init_qs8_avgpool_minmax_scalar_params;
      qs8_avgpool_config.primary_tile = 9;
      qs8_avgpool_config.channel_tile = 1;
    }
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    qs8_avgpool_config.ukernel = XNN_INIT_AVGPOOL_UKERNEL(xnn_qs8_avgpool_minmax_fp32_ukernel_9p__wasmsimd_u16);
    qs8_avgpool_config.init.qs8 = xnn_init_qs8_avgpool_minmax_scalar_params;
    qs8_avgpool_config.primary_tile = 9;
    qs8_avgpool_config.channel_tile = 16;
  #elif XNN_ARCH_HEXAGON && XNN_ENABLE_HVX
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->arch_flags & xnn_arch_hvx) {
      qs8_avgpool_config.ukernel = XNN_INIT_AVGPOOL_UKERNEL(xnn_qs8_avgpool_minmax_fp32_ukernel_9p__hvx_u32);
      qs8_avgpool_config.init.qs8 = xnn_init_qs8_avgpool_minmax_scalar_params;
      qs8_avgpool_config.primary_tile = 9;
      qs8_avgpool_config.channel_tile = 32;
    }
  #elif XNN_ARCH_RISCV && XNN_ENABLE_RISCV_VECTOR
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    qs8_avgpool_config.ukernel = XNN_INIT_AVGPOOL_UKERNEL(xnn_qs8_avgpool_minmax_fp32_ukernel_9p__rvv_u2v);
    qs8_avgpool_config.init.qs8 = xnn_init_qs8_avgpool_minmax_scalar_params;
    qs8_avgpool_config.primary_tile = 9;
    qs8_avgpool_config.channel_tile = 2 * hardware_config->vlenb / sizeof(int8_t);
  #else
    qs8_avgpool_config.ukernel = XNN_INIT_AVGPOOL_UKERNEL(xnn_qs8_avgpool_minmax_fp32_ukernel_9p__scalar_fmagic_u1);
    qs8_avgpool_config.init.qs8 = xnn_init_qs8_avgpool_minmax_scalar_params;
    qs8_avgpool_config.primary_tile = 9;
    qs8_avgpool_config.channel_tile = 1;
  #endif
}

static void init_qu8_avgpool_config(void) {
  #if XNN_ARCH_ARM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    (void) hardware_config;  // May be unused.
    if (hardware_config->arch_flags & xnn_arch_arm_neon) {
      qu8_avgpool_config.ukernel = XNN_INIT_AVGPOOL_UKERNEL(xnn_qu8_avgpool_minmax_fp32_ukernel_9p__neon_u16);
      qu8_avgpool_config.init.qu8 = xnn_init_qu8_avgpool_minmax_scalar_params;
      qu8_avgpool_config.primary_tile = 9;
      qu8_avgpool_config.channel_tile = 16;
    } else {
      qu8_avgpool_config.ukernel = XNN_INIT_AVGPOOL_UKERNEL(xnn_qu8_avgpool_minmax_fp32_ukernel_9p__scalar_fmagic_u1);
      qu8_avgpool_config.init.qu8 = xnn_init_qu8_avgpool_minmax_scalar_params;
      qu8_avgpool_config.primary_tile = 9;
      qu8_avgpool_config.channel_tile = 1;
    }
  #elif XNN_ARCH_ARM64
    qu8_avgpool_config.ukernel = XNN_INIT_AVGPOOL_UKERNEL(xnn_qu8_avgpool_minmax_fp32_ukernel_9p__neon_u16);
    qu8_avgpool_config.init.qu8 = xnn_init_qu8_avgpool_minmax_scalar_params;
    qu8_avgpool_config.primary_tile = 9;
    qu8_avgpool_config.channel_tile = 16;
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    (void) hardware_config;  // May be unused.
    #if XNN_ENABLE_AVX512SKX
      if (hardware_config->arch_flags & xnn_arch_x86_avx512skx) {
        qu8_avgpool_config.ukernel = XNN_INIT_AVGPOOL_UKERNEL(xnn_qu8_avgpool_minmax_fp32_ukernel_9p__avx512skx_u32);
        qu8_avgpool_config.init.qu8 = xnn_init_qu8_avgpool_minmax_scalar_params;
        qu8_avgpool_config.primary_tile = 9;
        qu8_avgpool_config.channel_tile = 32;
      } else
    #endif
    #if XNN_ENABLE_AVX2
      if (hardware_config->arch_flags & xnn_arch_x86_avx2) {
        qu8_avgpool_config.ukernel = XNN_INIT_AVGPOOL_UKERNEL(xnn_qu8_avgpool_minmax_fp32_ukernel_9p__avx2_u32);
        qu8_avgpool_config.init.qu8 = xnn_init_qu8_avgpool_minmax_scalar_params;
        qu8_avgpool_config.primary_tile = 9;
        qu8_avgpool_config.channel_tile = 32;
      } else
    #endif
    #if XNN_ENABLE_SSE41
      if (hardware_config->arch_flags & xnn_arch_x86_sse4_1) {
        qu8_avgpool_config.ukernel = XNN_INIT_AVGPOOL_UKERNEL(xnn_qu8_avgpool_minmax_fp32_ukernel_9p__sse41_u16);
        qu8_avgpool_config.init.qu8 = xnn_init_qu8_avgpool_minmax_scalar_params;
        qu8_avgpool_config.primary_tile = 9;
        qu8_avgpool_config.channel_tile = 16;
      } else
    #endif
    {
      qu8_avgpool_config.ukernel = XNN_INIT_AVGPOOL_UKERNEL(xnn_qu8_avgpool_minmax_fp32_ukernel_9p__scalar_fmagic_u1);
      qu8_avgpool_config.init.qu8 = xnn_init_qu8_avgpool_minmax_scalar_params;
      qu8_avgpool_config.primary_tile = 9;
      qu8_avgpool_config.channel_tile = 1;
    }
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    qu8_avgpool_config.ukernel = XNN_INIT_AVGPOOL_UKERNEL(xnn_qu8_avgpool_minmax_fp32_ukernel_9p__wasmsimd_u16);
    qu8_avgpool_config.init.qu8 = xnn_init_qu8_avgpool_minmax_scalar_params;
    qu8_avgpool_config.primary_tile = 9;
    qu8_avgpool_config.channel_tile = 16;
  #elif XNN_ARCH_HEXAGON && XNN_ENABLE_HVX
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->arch_flags & xnn_arch_hvx) {
      qu8_avgpool_config.ukernel = XNN_INIT_AVGPOOL_UKERNEL(xnn_qu8_avgpool_minmax_fp32_ukernel_9p__hvx_u32);
      qu8_avgpool_config.init.qu8 = xnn_init_qu8_avgpool_minmax_scalar_params;
      qu8_avgpool_config.primary_tile = 9;
      qu8_avgpool_config.channel_tile = 32;
    }
  #elif XNN_ARCH_RISCV && XNN_ENABLE_RISCV_VECTOR
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    qu8_avgpool_config.ukernel = XNN_INIT_AVGPOOL_UKERNEL(xnn_qu8_avgpool_minmax_fp32_ukernel_9p__rvv_u2v);
    qu8_avgpool_config.init.qu8 = xnn_init_qu8_avgpool_minmax_scalar_params;
    qu8_avgpool_config.primary_tile = 9;
    qu8_avgpool_config.channel_tile = 2 * hardware_config->vlenb / sizeof(uint8_t);
  #else
    qu8_avgpool_config.ukernel = XNN_INIT_AVGPOOL_UKERNEL(xnn_qu8_avgpool_minmax_fp32_ukernel_9p__scalar_fmagic_u1);
    qu8_avgpool_config.init.qu8 = xnn_init_qu8_avgpool_minmax_scalar_params;
    qu8_avgpool_config.primary_tile = 9;
    qu8_avgpool_config.channel_tile = 1;
  #endif
}

const struct xnn_avgpool_config* xnn_init_qs8_avgpool_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  XNN_INIT_ONCE(qs8_avgpool);
  return &qs8_avgpool_config;
}

const struct xnn_avgpool_config* xnn_init_qu8_avgpool_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  XNN_INIT_ONCE(qu8_avgpool);
  return &qu8_avgpool_config;
}
