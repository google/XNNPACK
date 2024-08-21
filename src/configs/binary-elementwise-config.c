// Copyright 2022 Google LLC
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
#include "xnnpack/vbinary.h"

static struct xnn_binary_elementwise_config f16_vadd_config = {0};
static struct xnn_binary_elementwise_config f16_vdiv_config = {0};
static struct xnn_binary_elementwise_config f16_vmax_config = {0};
static struct xnn_binary_elementwise_config f16_vmin_config = {0};
static struct xnn_binary_elementwise_config f16_vmul_config = {0};
static struct xnn_binary_elementwise_config f16_vsub_config = {0};
static struct xnn_binary_elementwise_config f16_vsqrdiff_config = {0};

static struct xnn_binary_elementwise_config f32_vadd_config = {0};
static struct xnn_binary_elementwise_config f32_vcopysign_config = {0};
static struct xnn_binary_elementwise_config f32_vdiv_config = {0};
static struct xnn_binary_elementwise_config f32_vmax_config = {0};
static struct xnn_binary_elementwise_config f32_vmin_config = {0};
static struct xnn_binary_elementwise_config f32_vmul_config = {0};
static struct xnn_binary_elementwise_config f32_vsub_config = {0};
static struct xnn_binary_elementwise_config f32_vsqrdiff_config = {0};


static struct xnn_binary_elementwise_config s32_vmul_config = {0};

static struct xnn_binary_elementwise_config qs8_vadd_config = {0};
static struct xnn_binary_elementwise_config qs8_vmul_config = {0};

static struct xnn_binary_elementwise_config qu8_vadd_config = {0};
static struct xnn_binary_elementwise_config qu8_vmul_config = {0};

XNN_INIT_ONCE_GUARD(f16_vadd);
XNN_INIT_ONCE_GUARD(f16_vdiv);
XNN_INIT_ONCE_GUARD(f16_vmax);
XNN_INIT_ONCE_GUARD(f16_vmin);
XNN_INIT_ONCE_GUARD(f16_vmul);
XNN_INIT_ONCE_GUARD(f16_vsub);
XNN_INIT_ONCE_GUARD(f16_vsqrdiff);
XNN_INIT_ONCE_GUARD(f32_vadd);
XNN_INIT_ONCE_GUARD(f32_vcopysign);
XNN_INIT_ONCE_GUARD(f32_vdiv);
XNN_INIT_ONCE_GUARD(f32_vmax);
XNN_INIT_ONCE_GUARD(f32_vmin);
XNN_INIT_ONCE_GUARD(f32_vmul);
XNN_INIT_ONCE_GUARD(f32_vsub);
XNN_INIT_ONCE_GUARD(f32_vsqrdiff);
XNN_INIT_ONCE_GUARD(s32_vmul);
XNN_INIT_ONCE_GUARD(qs8_vadd);
XNN_INIT_ONCE_GUARD(qs8_vmul);
XNN_INIT_ONCE_GUARD(qu8_vadd);
XNN_INIT_ONCE_GUARD(qu8_vmul);


static void init_f16_vadd_config(void) {
  #if XNN_ARCH_ARM && XNN_ENABLE_ARM_FP16_VECTOR && XNN_ENABLE_ARM_FP16_SCALAR
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon_fp16_arith) {
      f16_vadd_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_f16_vadd_minmax_ukernel__neonfp16arith_u16;
      f16_vadd_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f16_vaddc_minmax_ukernel__neonfp16arith_u16;
      f16_vadd_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f16_vaddc_minmax_ukernel__neonfp16arith_u16;
      f16_vadd_config.init.f16_minmax = xnn_init_f16_minmax_fp16arith_params;
      f16_vadd_config.minmax.element_tile = 16;
    }
  #elif XNN_ARCH_ARM64 && XNN_ENABLE_ARM_FP16_VECTOR
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon_fp16_arith) {
      f16_vadd_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_f16_vadd_minmax_ukernel__neonfp16arith_u16;
      f16_vadd_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f16_vaddc_minmax_ukernel__neonfp16arith_u16;
      f16_vadd_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f16_vaddc_minmax_ukernel__neonfp16arith_u16;
      f16_vadd_config.init.f16_minmax = xnn_init_f16_minmax_fp16arith_params;
      f16_vadd_config.minmax.element_tile = 16;
    }
  #elif (XNN_ARCH_X86 || XNN_ARCH_X86_64) && !XNN_PLATFORM_MOBILE
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    #if XNN_ENABLE_AVX512FP16
      if (hardware_config->use_x86_avx512fp16) {
        f16_vadd_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_f16_vadd_minmax_ukernel__avx512fp16_u64;
        f16_vadd_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f16_vaddc_minmax_ukernel__avx512fp16_u64;
        f16_vadd_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f16_vaddc_minmax_ukernel__avx512fp16_u64;
        f16_vadd_config.init.f16_minmax = xnn_init_f16_minmax_fp16arith_params;
        f16_vadd_config.minmax.element_tile = 64;
      } else
    #endif
    if (hardware_config->use_x86_f16c) {
      f16_vadd_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_f16_vadd_minmax_ukernel__f16c_u16;
      f16_vadd_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f16_vaddc_minmax_ukernel__f16c_u16;
      f16_vadd_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f16_vaddc_minmax_ukernel__f16c_u16;
      f16_vadd_config.init.f16_minmax = xnn_init_f16_minmax_avx_params;
      f16_vadd_config.minmax.element_tile = 16;
    }
  #endif
}

static void init_f16_vdiv_config(void) {
  #if XNN_ARCH_ARM && XNN_ENABLE_ARM_FP16_VECTOR && XNN_ENABLE_ARM_FP16_SCALAR
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon_fp16_arith) {
      f16_vdiv_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_f16_vdiv_minmax_ukernel__fp16arith_u2;
      f16_vdiv_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f16_vdivc_minmax_ukernel__fp16arith_u2;
      f16_vdiv_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f16_vrdivc_minmax_ukernel__fp16arith_u2;
      f16_vdiv_config.init.f16_minmax = xnn_init_f16_minmax_fp16arith_params;
      f16_vdiv_config.minmax.element_tile = 2;
    }
  #elif XNN_ARCH_ARM64 && XNN_ENABLE_ARM_FP16_VECTOR
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon_fp16_arith) {
      f16_vdiv_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_f16_vdiv_minmax_ukernel__aarch64_neonfp16arith_u8;
      f16_vdiv_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f16_vdivc_minmax_ukernel__aarch64_neonfp16arith_u8;
      f16_vdiv_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f16_vrdivc_minmax_ukernel__aarch64_neonfp16arith_u8;
      f16_vdiv_config.init.f16_minmax = xnn_init_f16_minmax_fp16arith_params;
      f16_vdiv_config.minmax.element_tile = 8;
    }
  #elif (XNN_ARCH_X86 || XNN_ARCH_X86_64) && !XNN_PLATFORM_MOBILE
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    #if XNN_ENABLE_AVX512FP16
      if (hardware_config->use_x86_avx512fp16) {
        f16_vdiv_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_f16_vdiv_minmax_ukernel__avx512fp16_u64;
        f16_vdiv_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f16_vdivc_minmax_ukernel__avx512fp16_u64;
        f16_vdiv_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f16_vrdivc_minmax_ukernel__avx512fp16_u64;
        f16_vdiv_config.init.f16_minmax = xnn_init_f16_minmax_fp16arith_params;
        f16_vdiv_config.minmax.element_tile = 64;
      } else
    #endif
    if (hardware_config->use_x86_f16c) {
      f16_vdiv_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_f16_vdiv_minmax_ukernel__f16c_u8;
      f16_vdiv_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f16_vdivc_minmax_ukernel__f16c_u8;
      f16_vdiv_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f16_vrdivc_minmax_ukernel__f16c_u8;
      f16_vdiv_config.init.f16_minmax = xnn_init_f16_minmax_avx_params;
      f16_vdiv_config.minmax.element_tile = 8;
    }
  #endif
}

static void init_f16_vmax_config(void) {
  #if XNN_ARCH_ARM && XNN_ENABLE_ARM_FP16_VECTOR && XNN_ENABLE_ARM_FP16_SCALAR
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon_fp16_arith) {
      f16_vmax_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_f16_vmax_ukernel__neonfp16arith_u16;
      f16_vmax_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f16_vmaxc_ukernel__neonfp16arith_u16;
      f16_vmax_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f16_vmaxc_ukernel__neonfp16arith_u16;
      f16_vmax_config.minmax.element_tile = 16;
    }
  #elif XNN_ARCH_ARM64 && XNN_ENABLE_ARM_FP16_VECTOR
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon_fp16_arith) {
      f16_vmax_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_f16_vmax_ukernel__neonfp16arith_u16;
      f16_vmax_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f16_vmaxc_ukernel__neonfp16arith_u16;
      f16_vmax_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f16_vmaxc_ukernel__neonfp16arith_u16;
      f16_vmax_config.minmax.element_tile = 16;
    }
  #elif (XNN_ARCH_X86 || XNN_ARCH_X86_64) && !XNN_PLATFORM_MOBILE
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    #if XNN_ENABLE_AVX512FP16
      if (hardware_config->use_x86_avx512fp16) {
        f16_vmax_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_f16_vmax_ukernel__avx512fp16_u64;
        f16_vmax_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f16_vmaxc_ukernel__avx512fp16_u64;
        f16_vmax_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f16_vmaxc_ukernel__avx512fp16_u64;
        f16_vmax_config.init.f16_minmax = xnn_init_f16_minmax_fp16arith_params;
        f16_vmax_config.minmax.element_tile = 64;
      } else
    #endif
    if (hardware_config->use_x86_f16c) {
      f16_vmax_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_f16_vmax_ukernel__f16c_u16;
      f16_vmax_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f16_vmaxc_ukernel__f16c_u16;
      f16_vmax_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f16_vmaxc_ukernel__f16c_u16;
      f16_vmax_config.minmax.element_tile = 16;
    }
  #endif
}

static void init_f16_vmin_config(void) {
  #if XNN_ARCH_ARM && XNN_ENABLE_ARM_FP16_VECTOR && XNN_ENABLE_ARM_FP16_SCALAR
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon_fp16_arith) {
      f16_vmin_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_f16_vmin_ukernel__neonfp16arith_u16;
      f16_vmin_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f16_vminc_ukernel__neonfp16arith_u16;
      f16_vmin_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f16_vminc_ukernel__neonfp16arith_u16;
      f16_vmin_config.minmax.element_tile = 16;
    }
  #elif XNN_ARCH_ARM64 && XNN_ENABLE_ARM_FP16_VECTOR
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon_fp16_arith) {
      f16_vmin_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_f16_vmin_ukernel__neonfp16arith_u16;
      f16_vmin_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f16_vminc_ukernel__neonfp16arith_u16;
      f16_vmin_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f16_vminc_ukernel__neonfp16arith_u16;
      f16_vmin_config.minmax.element_tile = 16;
    }
  #elif (XNN_ARCH_X86 || XNN_ARCH_X86_64) && !XNN_PLATFORM_MOBILE
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    #if XNN_ENABLE_AVX512FP16
      if (hardware_config->use_x86_avx512fp16) {
        f16_vmin_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_f16_vmin_ukernel__avx512fp16_u64;
        f16_vmin_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f16_vminc_ukernel__avx512fp16_u64;
        f16_vmin_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f16_vminc_ukernel__avx512fp16_u64;
        f16_vmin_config.init.f16_minmax = xnn_init_f16_minmax_fp16arith_params;
        f16_vmin_config.minmax.element_tile = 64;
      } else
    #endif
    if (hardware_config->use_x86_f16c) {
      f16_vmin_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_f16_vmin_ukernel__f16c_u16;
      f16_vmin_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f16_vminc_ukernel__f16c_u16;
      f16_vmin_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f16_vminc_ukernel__f16c_u16;
      f16_vmin_config.minmax.element_tile = 16;
    }
  #endif
}

static void init_f16_vmul_config(void) {
  #if XNN_ARCH_ARM && XNN_ENABLE_ARM_FP16_VECTOR && XNN_ENABLE_ARM_FP16_SCALAR
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon_fp16_arith) {
      f16_vmul_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_f16_vmul_minmax_ukernel__neonfp16arith_u16;
      f16_vmul_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f16_vmulc_minmax_ukernel__neonfp16arith_u16;
      f16_vmul_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f16_vmulc_minmax_ukernel__neonfp16arith_u16;
      f16_vmul_config.init.f16_minmax = xnn_init_f16_minmax_fp16arith_params;
      f16_vmul_config.minmax.element_tile = 16;
    }
  #elif XNN_ARCH_ARM64 && XNN_ENABLE_ARM_FP16_VECTOR
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon_fp16_arith) {
      f16_vmul_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_f16_vmul_minmax_ukernel__neonfp16arith_u16;
      f16_vmul_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f16_vmulc_minmax_ukernel__neonfp16arith_u16;
      f16_vmul_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f16_vmulc_minmax_ukernel__neonfp16arith_u16;
      f16_vmul_config.init.f16_minmax = xnn_init_f16_minmax_fp16arith_params;
      f16_vmul_config.minmax.element_tile = 16;
    }
  #elif (XNN_ARCH_X86 || XNN_ARCH_X86_64) && !XNN_PLATFORM_MOBILE
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    #if XNN_ENABLE_AVX512FP16
      if (hardware_config->use_x86_avx512fp16) {
        f16_vmul_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_f16_vmul_minmax_ukernel__avx512fp16_u64;
        f16_vmul_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f16_vmulc_minmax_ukernel__avx512fp16_u64;
        f16_vmul_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f16_vmulc_minmax_ukernel__avx512fp16_u64;
        f16_vmul_config.init.f16_minmax = xnn_init_f16_minmax_fp16arith_params;
        f16_vmul_config.minmax.element_tile = 64;
      } else
    #endif
    if (hardware_config->use_x86_f16c) {
      f16_vmul_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_f16_vmul_minmax_ukernel__f16c_u16;
      f16_vmul_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f16_vmulc_minmax_ukernel__f16c_u16;
      f16_vmul_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f16_vmulc_minmax_ukernel__f16c_u16;
      f16_vmul_config.init.f16_minmax = xnn_init_f16_minmax_avx_params;
      f16_vmul_config.minmax.element_tile = 16;
    }
  #endif
}

static void init_f16_vsub_config(void) {
  #if XNN_ARCH_ARM && XNN_ENABLE_ARM_FP16_VECTOR && XNN_ENABLE_ARM_FP16_SCALAR
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon_fp16_arith) {
      f16_vsub_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_f16_vsub_minmax_ukernel__neonfp16arith_u16;
      f16_vsub_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f16_vsubc_minmax_ukernel__neonfp16arith_u16;
      f16_vsub_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f16_vrsubc_minmax_ukernel__neonfp16arith_u16;
      f16_vsub_config.init.f16_minmax = xnn_init_f16_minmax_fp16arith_params;
      f16_vsub_config.minmax.element_tile = 16;
    }
  #elif XNN_ARCH_ARM64 && XNN_ENABLE_ARM_FP16_VECTOR
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon_fp16_arith) {
      f16_vsub_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_f16_vsub_minmax_ukernel__neonfp16arith_u16;
      f16_vsub_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f16_vsubc_minmax_ukernel__neonfp16arith_u16;
      f16_vsub_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f16_vrsubc_minmax_ukernel__neonfp16arith_u16;
      f16_vsub_config.init.f16_minmax = xnn_init_f16_minmax_fp16arith_params;
      f16_vsub_config.minmax.element_tile = 16;
    }
  #elif (XNN_ARCH_X86 || XNN_ARCH_X86_64) && !XNN_PLATFORM_MOBILE
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    #if XNN_ENABLE_AVX512FP16
      if (hardware_config->use_x86_avx512fp16) {
        f16_vsub_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_f16_vsub_minmax_ukernel__avx512fp16_u64;
        f16_vsub_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f16_vsubc_minmax_ukernel__avx512fp16_u64;
        f16_vsub_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f16_vrsubc_minmax_ukernel__avx512fp16_u64;
        f16_vsub_config.init.f16_minmax = xnn_init_f16_minmax_fp16arith_params;
        f16_vsub_config.minmax.element_tile = 64;
      } else
    #endif
    if (hardware_config->use_x86_f16c) {
      f16_vsub_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_f16_vsub_minmax_ukernel__f16c_u16;
      f16_vsub_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f16_vsubc_minmax_ukernel__f16c_u16;
      f16_vsub_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f16_vrsubc_minmax_ukernel__f16c_u16;
      f16_vsub_config.init.f16_minmax = xnn_init_f16_minmax_avx_params;
      f16_vsub_config.minmax.element_tile = 16;
    }
  #endif
}

static void init_f16_vsqrdiff_config(void) {
  #if XNN_ARCH_ARM && XNN_ENABLE_ARM_FP16_VECTOR && XNN_ENABLE_ARM_FP16_SCALAR
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon_fp16_arith) {
      f16_vsqrdiff_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_f16_vsqrdiff_ukernel__neonfp16arith_u16;
      f16_vsqrdiff_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f16_vsqrdiffc_ukernel__neonfp16arith_u16;
      f16_vsqrdiff_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f16_vsqrdiffc_ukernel__neonfp16arith_u16;
      f16_vsqrdiff_config.init.f16_minmax = xnn_init_f16_minmax_fp16arith_params;
      f16_vsqrdiff_config.minmax.element_tile = 16;
    }
  #elif XNN_ARCH_ARM64 && XNN_ENABLE_ARM_FP16_VECTOR
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon_fp16_arith) {
      f16_vsqrdiff_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_f16_vsqrdiff_ukernel__neonfp16arith_u16;
      f16_vsqrdiff_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f16_vsqrdiffc_ukernel__neonfp16arith_u16;
      f16_vsqrdiff_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f16_vsqrdiffc_ukernel__neonfp16arith_u16;
      f16_vsqrdiff_config.init.f16_minmax = xnn_init_f16_minmax_fp16arith_params;
      f16_vsqrdiff_config.minmax.element_tile = 16;
    }
  #elif (XNN_ARCH_X86 || XNN_ARCH_X86_64) && !XNN_PLATFORM_MOBILE
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    #if XNN_ENABLE_AVX512FP16
      if (hardware_config->use_x86_avx512fp16) {
        f16_vsqrdiff_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_f16_vsqrdiff_ukernel__avx512fp16_u64;
        f16_vsqrdiff_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f16_vsqrdiffc_ukernel__avx512fp16_u64;
        f16_vsqrdiff_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f16_vsqrdiffc_ukernel__avx512fp16_u64;
        f16_vsqrdiff_config.init.f16_minmax = xnn_init_f16_minmax_fp16arith_params;
        f16_vsqrdiff_config.minmax.element_tile = 64;
      } else
    #endif
    if (hardware_config->use_x86_f16c) {
      f16_vsqrdiff_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_f16_vsqrdiff_ukernel__f16c_u16;
      f16_vsqrdiff_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f16_vsqrdiffc_ukernel__f16c_u16;
      f16_vsqrdiff_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f16_vsqrdiffc_ukernel__f16c_u16;
      f16_vsqrdiff_config.init.f16_minmax = xnn_init_f16_minmax_avx_params;
      f16_vsqrdiff_config.minmax.element_tile = 16;
    }
  #endif
}

static void init_f32_vadd_config(void) {
  #if XNN_ARCH_ARM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon){
      f32_vadd_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vadd_minmax_ukernel__neon_u8;
      f32_vadd_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vaddc_minmax_ukernel__neon_u8;
      f32_vadd_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vaddc_minmax_ukernel__neon_u8;
      f32_vadd_config.init.f32_minmax = xnn_init_f32_minmax_scalar_params;
      f32_vadd_config.minmax.element_tile = 8;
    } else if (!XNN_PLATFORM_MOBILE) {
      f32_vadd_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vadd_minmax_ukernel__scalar_u8;
      f32_vadd_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vaddc_minmax_ukernel__scalar_u8;
      f32_vadd_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vaddc_minmax_ukernel__scalar_u8;
      f32_vadd_config.init.f32_minmax = xnn_init_f32_minmax_scalar_params;
      f32_vadd_config.minmax.element_tile = 8;
    }
  #elif XNN_ARCH_ARM64
    f32_vadd_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vadd_minmax_ukernel__neon_u8;
    f32_vadd_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vaddc_minmax_ukernel__neon_u8;
    f32_vadd_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vaddc_minmax_ukernel__neon_u8;
    f32_vadd_config.init.f32_minmax = xnn_init_f32_minmax_scalar_params;
    f32_vadd_config.minmax.element_tile = 8;
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (!XNN_PLATFORM_MOBILE && hardware_config->use_x86_avx512f) {
      f32_vadd_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vadd_minmax_ukernel__avx512f_u32;
      f32_vadd_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vaddc_minmax_ukernel__avx512f_u32;
      f32_vadd_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vaddc_minmax_ukernel__avx512f_u32;
      f32_vadd_config.init.f32_minmax = xnn_init_f32_minmax_scalar_params;
      f32_vadd_config.minmax.element_tile = 32;
    } else if (hardware_config->use_x86_avx) {
      f32_vadd_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vadd_minmax_ukernel__avx_u16;
      f32_vadd_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vaddc_minmax_ukernel__avx_u16;
      f32_vadd_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vaddc_minmax_ukernel__avx_u16;
      f32_vadd_config.init.f32_minmax = xnn_init_f32_minmax_avx_params;
      f32_vadd_config.minmax.element_tile = 16;
    } else {
      f32_vadd_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vadd_minmax_ukernel__sse_u8;
      f32_vadd_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vaddc_minmax_ukernel__sse_u8;
      f32_vadd_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vaddc_minmax_ukernel__sse_u8;
      f32_vadd_config.init.f32_minmax = xnn_init_f32_minmax_sse_params;
      f32_vadd_config.minmax.element_tile = 8;
    }
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->is_x86) {
      f32_vadd_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vadd_minmax_ukernel__wasmsimd_x86_u16;
      f32_vadd_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vaddc_minmax_ukernel__wasmsimd_x86_u16;
      f32_vadd_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vaddc_minmax_ukernel__wasmsimd_x86_u16;
      f32_vadd_config.linear.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vadd_ukernel__wasmsimd_u16;
      f32_vadd_config.linear.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vaddc_ukernel__wasmsimd_u16;
      f32_vadd_config.linear.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vaddc_ukernel__wasmsimd_u16;
      f32_vadd_config.init.f32_minmax = xnn_init_f32_minmax_wasmsimd_params;
      f32_vadd_config.minmax.element_tile = 16;
      f32_vadd_config.linear.element_tile = 16;
    } else {
      f32_vadd_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vadd_minmax_ukernel__wasmsimd_arm_u16;
      f32_vadd_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vaddc_minmax_ukernel__wasmsimd_arm_u16;
      f32_vadd_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vaddc_minmax_ukernel__wasmsimd_arm_u16;
      f32_vadd_config.linear.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vadd_ukernel__wasmsimd_u16;
      f32_vadd_config.linear.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vaddc_ukernel__wasmsimd_u16;
      f32_vadd_config.linear.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vaddc_ukernel__wasmsimd_u16;
      f32_vadd_config.init.f32_minmax = xnn_init_f32_minmax_wasmsimd_params;
      f32_vadd_config.minmax.element_tile = 16;
      f32_vadd_config.linear.element_tile = 16;
    }
  #elif XNN_ARCH_WASM
    f32_vadd_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vadd_minmax_ukernel__wasm_u8;
    f32_vadd_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vaddc_minmax_ukernel__wasm_u8;
    f32_vadd_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vaddc_minmax_ukernel__wasm_u8;
    f32_vadd_config.init.f32_minmax = xnn_init_f32_minmax_scalar_params;
    f32_vadd_config.minmax.element_tile = 8;
  #elif XNN_ARCH_RISCV && XNN_ENABLE_RISCV_VECTOR
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    f32_vadd_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vadd_minmax_ukernel__rvv_u8v;
    f32_vadd_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vaddc_minmax_ukernel__rvv_u8v;
    f32_vadd_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vaddc_minmax_ukernel__rvv_u8v;
    f32_vadd_config.init.f32_minmax = xnn_init_f32_minmax_scalar_params;
    f32_vadd_config.minmax.element_tile = hardware_config->vlenb * 2;  // VLENB * (8 / sizeof(float))
  #else
    f32_vadd_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vadd_minmax_ukernel__scalar_u8;
    f32_vadd_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vaddc_minmax_ukernel__scalar_u8;
    f32_vadd_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vaddc_minmax_ukernel__scalar_u8;
    f32_vadd_config.init.f32_minmax = xnn_init_f32_minmax_scalar_params;
    f32_vadd_config.minmax.element_tile = 8;
  #endif
}

static void init_f32_vcopysign_config(void) {
  #if XNN_ARCH_ARM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon){
      f32_vcopysign_config.linear.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vcopysign_ukernel__neon_u8;
      f32_vcopysign_config.linear.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vcopysignc_ukernel__neon_u8;
      f32_vcopysign_config.linear.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vrcopysignc_ukernel__neon_u8;
      f32_vcopysign_config.linear.element_tile = 2;
    } else if (!XNN_PLATFORM_MOBILE) {
      f32_vcopysign_config.linear.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vcopysign_ukernel__scalar_u2;
      f32_vcopysign_config.linear.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vcopysignc_ukernel__scalar_u2;
      f32_vcopysign_config.linear.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vrcopysignc_ukernel__scalar_u2;
      f32_vcopysign_config.linear.element_tile = 2;
    }
  #elif XNN_ARCH_ARM64
    f32_vcopysign_config.linear.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vcopysign_ukernel__neon_u8;
    f32_vcopysign_config.linear.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vcopysignc_ukernel__neon_u8;
    f32_vcopysign_config.linear.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vrcopysignc_ukernel__neon_u8;
    f32_vcopysign_config.linear.element_tile = 8;
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (!XNN_PLATFORM_MOBILE && hardware_config->use_x86_avx512f) {
      f32_vcopysign_config.linear.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vcopysign_ukernel__avx512f_u32;
      f32_vcopysign_config.linear.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vcopysignc_ukernel__avx512f_u32;
      f32_vcopysign_config.linear.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vrcopysignc_ukernel__avx512f_u32;
      f32_vcopysign_config.linear.element_tile = 32;
    } else if (hardware_config->use_x86_avx) {
      f32_vcopysign_config.linear.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vcopysign_ukernel__avx_u16;
      f32_vcopysign_config.linear.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vcopysignc_ukernel__avx_u16;
      f32_vcopysign_config.linear.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vrcopysignc_ukernel__avx_u16;
      f32_vcopysign_config.linear.element_tile = 16;
    } else {
      f32_vcopysign_config.linear.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vcopysign_ukernel__sse2_u8;
      f32_vcopysign_config.linear.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vcopysignc_ukernel__sse2_u8;
      f32_vcopysign_config.linear.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vrcopysignc_ukernel__sse2_u8;
      f32_vcopysign_config.linear.element_tile = 8;
    }
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    f32_vcopysign_config.linear.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vcopysign_ukernel__wasmsimd_u16;
    f32_vcopysign_config.linear.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vcopysignc_ukernel__wasmsimd_u16;
    f32_vcopysign_config.linear.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vrcopysignc_ukernel__wasmsimd_u16;
    f32_vcopysign_config.linear.element_tile = 16;
    f32_vcopysign_config.linear.element_tile = 16;
  #else
    f32_vcopysign_config.linear.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vcopysign_ukernel__scalar_u2;
    f32_vcopysign_config.linear.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vcopysignc_ukernel__scalar_u2;
    f32_vcopysign_config.linear.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vrcopysignc_ukernel__scalar_u2;
    f32_vcopysign_config.linear.element_tile = 2;
  #endif
}


static void init_s32_vmul_config(void) {
  #if XNN_ARCH_ARM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon) {
      s32_vmul_config.linear.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_s32_vmul_ukernel__neon_u8;
      s32_vmul_config.linear.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_s32_vmulc_ukernel__neon_u8;
      s32_vmul_config.linear.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_s32_vmulc_ukernel__neon_u8;
      s32_vmul_config.linear.element_tile = 8;
    }
    else if (!XNN_PLATFORM_MOBILE) {
      s32_vmul_config.linear.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_s32_vmul_ukernel__scalar_u2;
      s32_vmul_config.linear.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_s32_vmulc_ukernel__scalar_u2;
      s32_vmul_config.linear.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_s32_vmulc_ukernel__scalar_u2;
      s32_vmul_config.linear.element_tile = 2;
    }
  #elif XNN_ARCH_ARM64
    s32_vmul_config.linear.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_s32_vmul_ukernel__neon_u8;
    s32_vmul_config.linear.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_s32_vmulc_ukernel__neon_u8;
    s32_vmul_config.linear.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_s32_vmulc_ukernel__neon_u8;
    s32_vmul_config.linear.element_tile = 8;
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (!XNN_PLATFORM_MOBILE && hardware_config->use_x86_avx512f) {
      s32_vmul_config.linear.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_s32_vmul_ukernel__avx512f_u32;
      s32_vmul_config.linear.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_s32_vmulc_ukernel__avx512f_u32;
      s32_vmul_config.linear.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_s32_vmulc_ukernel__avx512f_u32;
      s32_vmul_config.linear.element_tile = 32;
    }
    else if (hardware_config->use_x86_avx2) {
      s32_vmul_config.linear.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_s32_vmul_ukernel__avx2_u16;
      s32_vmul_config.linear.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_s32_vmulc_ukernel__avx2_u16;
      s32_vmul_config.linear.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_s32_vmulc_ukernel__avx2_u16;
      s32_vmul_config.linear.element_tile = 16;
    }
    else {
      s32_vmul_config.linear.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_s32_vmul_ukernel__sse41_u8;
      s32_vmul_config.linear.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_s32_vmulc_ukernel__sse41_u8;
      s32_vmul_config.linear.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_s32_vmulc_ukernel__sse41_u8;
      s32_vmul_config.linear.element_tile = 8;
    }
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    s32_vmul_config.linear.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_s32_vmul_ukernel__wasmsimd_u16;
    s32_vmul_config.linear.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_s32_vmulc_ukernel__wasmsimd_u16;
    s32_vmul_config.linear.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_s32_vmulc_ukernel__wasmsimd_u16;
    s32_vmul_config.linear.element_tile = 16;
  #else
    s32_vmul_config.linear.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_s32_vmul_ukernel__scalar_u2;
    s32_vmul_config.linear.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_s32_vmulc_ukernel__scalar_u2;
    s32_vmul_config.linear.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_s32_vmulc_ukernel__scalar_u2;
    s32_vmul_config.linear.element_tile = 2;
  #endif
}

static void init_f32_vdiv_config(void) {
  #if XNN_ARCH_ARM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon){
      f32_vdiv_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vdiv_minmax_ukernel__scalar_u2;
      f32_vdiv_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vdivc_minmax_ukernel__scalar_u2;
      f32_vdiv_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vrdivc_minmax_ukernel__scalar_u2;
      f32_vdiv_config.init.f32_minmax = xnn_init_f32_minmax_scalar_params;
      f32_vdiv_config.minmax.element_tile = 2;
    } else if (!XNN_PLATFORM_MOBILE) {
      f32_vdiv_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vdiv_minmax_ukernel__scalar_u2;
      f32_vdiv_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vdivc_minmax_ukernel__scalar_u2;
      f32_vdiv_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vrdivc_minmax_ukernel__scalar_u2;
      f32_vdiv_config.init.f32_minmax = xnn_init_f32_minmax_scalar_params;
      f32_vdiv_config.minmax.element_tile = 2;
    }
  #elif XNN_ARCH_ARM64
    f32_vdiv_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vdiv_minmax_ukernel__aarch64_neon_u8;
    f32_vdiv_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vdivc_minmax_ukernel__aarch64_neon_u8;
    f32_vdiv_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vrdivc_minmax_ukernel__aarch64_neon_u8;
    f32_vdiv_config.init.f32_minmax = xnn_init_f32_minmax_scalar_params;
    f32_vdiv_config.minmax.element_tile = 8;
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (!XNN_PLATFORM_MOBILE && hardware_config->use_x86_avx512f) {
      f32_vdiv_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vdiv_minmax_ukernel__avx512f_u32;
      f32_vdiv_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vdivc_minmax_ukernel__avx512f_u32;
      f32_vdiv_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vrdivc_minmax_ukernel__avx512f_u32;
      f32_vdiv_config.init.f32_minmax = xnn_init_f32_minmax_scalar_params;
      f32_vdiv_config.minmax.element_tile = 32;
    } else if (hardware_config->use_x86_avx) {
      f32_vdiv_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vdiv_minmax_ukernel__avx_u16;
      f32_vdiv_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vdivc_minmax_ukernel__avx_u16;
      f32_vdiv_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vrdivc_minmax_ukernel__avx_u16;
      f32_vdiv_config.init.f32_minmax = xnn_init_f32_minmax_avx_params;
      f32_vdiv_config.minmax.element_tile = 16;
    } else {
      f32_vdiv_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vdiv_minmax_ukernel__sse_u8;
      f32_vdiv_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vdivc_minmax_ukernel__sse_u8;
      f32_vdiv_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vrdivc_minmax_ukernel__sse_u8;
      f32_vdiv_config.init.f32_minmax = xnn_init_f32_minmax_sse_params;
      f32_vdiv_config.minmax.element_tile = 8;
    }
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->is_x86) {
      f32_vdiv_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vdiv_minmax_ukernel__wasmsimd_x86_u16;
      f32_vdiv_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vdivc_minmax_ukernel__wasmsimd_x86_u16;
      f32_vdiv_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vrdivc_minmax_ukernel__wasmsimd_x86_u16;
      f32_vdiv_config.linear.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vdiv_ukernel__wasmsimd_u16;
      f32_vdiv_config.linear.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vdivc_ukernel__wasmsimd_u16;
      f32_vdiv_config.linear.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vrdivc_ukernel__wasmsimd_u16;
      f32_vdiv_config.init.f32_minmax = xnn_init_f32_minmax_wasmsimd_params;
      f32_vdiv_config.minmax.element_tile = 16;
      f32_vdiv_config.linear.element_tile = 16;
    } else {
      f32_vdiv_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vdiv_minmax_ukernel__wasmsimd_arm_u16;
      f32_vdiv_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vdivc_minmax_ukernel__wasmsimd_arm_u16;
      f32_vdiv_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vrdivc_minmax_ukernel__wasmsimd_arm_u16;
      f32_vdiv_config.linear.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vdiv_ukernel__wasmsimd_u16;
      f32_vdiv_config.linear.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vdivc_ukernel__wasmsimd_u16;
      f32_vdiv_config.linear.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vrdivc_ukernel__wasmsimd_u16;
      f32_vdiv_config.init.f32_minmax = xnn_init_f32_minmax_wasmsimd_params;
      f32_vdiv_config.minmax.element_tile = 16;
      f32_vdiv_config.linear.element_tile = 16;
    }
  #elif XNN_ARCH_WASM
    f32_vdiv_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vdiv_minmax_ukernel__wasm_u8;
    f32_vdiv_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vdivc_minmax_ukernel__wasm_u8;
    f32_vdiv_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vrdivc_minmax_ukernel__wasm_u8;
    f32_vdiv_config.init.f32_minmax = xnn_init_f32_minmax_scalar_params;
    f32_vdiv_config.minmax.element_tile = 8;
  #elif XNN_ARCH_RISCV && XNN_ENABLE_RISCV_VECTOR
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    f32_vdiv_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vdiv_minmax_ukernel__rvv_u8v;
    f32_vdiv_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vdivc_minmax_ukernel__rvv_u8v;
    f32_vdiv_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vrdivc_minmax_ukernel__rvv_u8v;
    f32_vdiv_config.init.f32_minmax = xnn_init_f32_minmax_scalar_params;
    f32_vdiv_config.minmax.element_tile = hardware_config->vlenb * 2;  // VLENB * (8 / sizeof(float))
  #else
    f32_vdiv_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vdiv_minmax_ukernel__scalar_u2;
    f32_vdiv_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vdivc_minmax_ukernel__scalar_u2;
    f32_vdiv_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vrdivc_minmax_ukernel__scalar_u2;
    f32_vdiv_config.init.f32_minmax = xnn_init_f32_minmax_scalar_params;
    f32_vdiv_config.minmax.element_tile = 2;
  #endif
}

static void init_f32_vmax_config(void) {
  #if XNN_ARCH_ARM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon){
      f32_vmax_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vmax_ukernel__neon_u8;
      f32_vmax_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vmaxc_ukernel__neon_u8;
      f32_vmax_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vmaxc_ukernel__neon_u8;
      f32_vmax_config.minmax.element_tile = 8;
    } else if (!XNN_PLATFORM_MOBILE) {
      f32_vmax_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vmax_ukernel__scalar_u8;
      f32_vmax_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vmaxc_ukernel__scalar_u8;
      f32_vmax_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vmaxc_ukernel__scalar_u8;
      f32_vmax_config.minmax.element_tile = 8;
    }
  #elif XNN_ARCH_ARM64
    f32_vmax_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vmax_ukernel__neon_u8;
    f32_vmax_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vmaxc_ukernel__neon_u8;
    f32_vmax_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vmaxc_ukernel__neon_u8;
    f32_vmax_config.minmax.element_tile = 8;
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (!XNN_PLATFORM_MOBILE && hardware_config->use_x86_avx512f) {
      f32_vmax_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vmax_ukernel__avx512f_u32;
      f32_vmax_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vmaxc_ukernel__avx512f_u32;
      f32_vmax_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vmaxc_ukernel__avx512f_u32;
      f32_vmax_config.minmax.element_tile = 32;
    } else if (hardware_config->use_x86_avx) {
      f32_vmax_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vmax_ukernel__avx_u16;
      f32_vmax_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vmaxc_ukernel__avx_u16;
      f32_vmax_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vmaxc_ukernel__avx_u16;
      f32_vmax_config.minmax.element_tile = 16;
    } else {
      f32_vmax_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vmax_ukernel__sse_u8;
      f32_vmax_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vmaxc_ukernel__sse_u8;
      f32_vmax_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vmaxc_ukernel__sse_u8;
      f32_vmax_config.minmax.element_tile = 8;
    }
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->is_x86) {
      f32_vmax_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vmax_ukernel__wasmsimd_x86_u16;
      f32_vmax_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vmaxc_ukernel__wasmsimd_x86_u16;
      f32_vmax_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vmaxc_ukernel__wasmsimd_x86_u16;
      f32_vmax_config.minmax.element_tile = 16;
    } else {
      f32_vmax_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vmax_ukernel__wasmsimd_arm_u16;
      f32_vmax_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vmaxc_ukernel__wasmsimd_arm_u16;
      f32_vmax_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vmaxc_ukernel__wasmsimd_arm_u16;
      f32_vmax_config.minmax.element_tile = 16;
    }
  #elif XNN_ARCH_WASM
    f32_vmax_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vmax_ukernel__wasm_u8;
    f32_vmax_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vmaxc_ukernel__wasm_u8;
    f32_vmax_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vmaxc_ukernel__wasm_u8;
    f32_vmax_config.minmax.element_tile = 8;
  #elif XNN_ARCH_RISCV && XNN_ENABLE_RISCV_VECTOR
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    f32_vmax_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vmax_ukernel__rvv_u8v;
    f32_vmax_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vmaxc_ukernel__rvv_u8v;
    f32_vmax_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vmaxc_ukernel__rvv_u8v;
    f32_vmax_config.minmax.element_tile = hardware_config->vlenb * 2;  // VLENB * (8 / sizeof(float))
  #else
    f32_vmax_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vmax_ukernel__scalar_u8;
    f32_vmax_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vmaxc_ukernel__scalar_u8;
    f32_vmax_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vmaxc_ukernel__scalar_u8;
    f32_vmax_config.minmax.element_tile = 8;
  #endif
}

static void init_f32_vmin_config(void) {
  #if XNN_ARCH_ARM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon){
      f32_vmin_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vmin_ukernel__neon_u8;
      f32_vmin_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vminc_ukernel__neon_u8;
      f32_vmin_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vminc_ukernel__neon_u8;
      f32_vmin_config.minmax.element_tile = 8;
    } else if (!XNN_PLATFORM_MOBILE) {
      f32_vmin_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vmin_ukernel__scalar_u8;
      f32_vmin_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vminc_ukernel__scalar_u8;
      f32_vmin_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vminc_ukernel__scalar_u8;
      f32_vmin_config.minmax.element_tile = 8;
    }
  #elif XNN_ARCH_ARM64
    f32_vmin_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vmin_ukernel__neon_u8;
    f32_vmin_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vminc_ukernel__neon_u8;
    f32_vmin_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vminc_ukernel__neon_u8;
    f32_vmin_config.minmax.element_tile = 8;
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (!XNN_PLATFORM_MOBILE && hardware_config->use_x86_avx512f) {
      f32_vmin_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vmin_ukernel__avx512f_u32;
      f32_vmin_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vminc_ukernel__avx512f_u32;
      f32_vmin_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vminc_ukernel__avx512f_u32;
      f32_vmin_config.minmax.element_tile = 32;
    } else if (hardware_config->use_x86_avx) {
      f32_vmin_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vmin_ukernel__avx_u16;
      f32_vmin_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vminc_ukernel__avx_u16;
      f32_vmin_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vminc_ukernel__avx_u16;
      f32_vmin_config.minmax.element_tile = 16;
    } else {
      f32_vmin_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vmin_ukernel__sse_u8;
      f32_vmin_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vminc_ukernel__sse_u8;
      f32_vmin_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vminc_ukernel__sse_u8;
      f32_vmin_config.minmax.element_tile = 8;
    }
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->is_x86) {
      f32_vmin_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vmin_ukernel__wasmsimd_x86_u16;
      f32_vmin_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vminc_ukernel__wasmsimd_x86_u16;
      f32_vmin_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vminc_ukernel__wasmsimd_x86_u16;
      f32_vmin_config.minmax.element_tile = 16;
    } else {
      f32_vmin_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vmin_ukernel__wasmsimd_arm_u16;
      f32_vmin_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vminc_ukernel__wasmsimd_arm_u16;
      f32_vmin_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vminc_ukernel__wasmsimd_arm_u16;
      f32_vmin_config.minmax.element_tile = 16;
    }
  #elif XNN_ARCH_WASM
    f32_vmin_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vmin_ukernel__wasm_u8;
    f32_vmin_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vminc_ukernel__wasm_u8;
    f32_vmin_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vminc_ukernel__wasm_u8;
    f32_vmin_config.minmax.element_tile = 8;
  #elif XNN_ARCH_RISCV && XNN_ENABLE_RISCV_VECTOR
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    f32_vmin_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vmin_ukernel__rvv_u8v;
    f32_vmin_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vminc_ukernel__rvv_u8v;
    f32_vmin_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vminc_ukernel__rvv_u8v;
    f32_vmin_config.minmax.element_tile = hardware_config->vlenb * 2;  // VLENB * (8 / sizeof(float))
  #else
    f32_vmin_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vmin_ukernel__scalar_u8;
    f32_vmin_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vminc_ukernel__scalar_u8;
    f32_vmin_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vminc_ukernel__scalar_u8;
    f32_vmin_config.minmax.element_tile = 8;
  #endif
}

static void init_f32_vmul_config(void) {
  #if XNN_ARCH_ARM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon){
      f32_vmul_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vmul_minmax_ukernel__neon_u8;
      f32_vmul_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vmulc_minmax_ukernel__neon_u8;
      f32_vmul_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vmulc_minmax_ukernel__neon_u8;
      f32_vmul_config.init.f32_minmax = xnn_init_f32_minmax_scalar_params;
      f32_vmul_config.minmax.element_tile = 8;
    } else if (!XNN_PLATFORM_MOBILE) {
      f32_vmul_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vmul_minmax_ukernel__scalar_u8;
      f32_vmul_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vmulc_minmax_ukernel__scalar_u8;
      f32_vmul_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vmulc_minmax_ukernel__scalar_u8;
      f32_vmul_config.init.f32_minmax = xnn_init_f32_minmax_scalar_params;
      f32_vmul_config.minmax.element_tile = 8;
    }
  #elif XNN_ARCH_ARM64
    f32_vmul_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vmul_minmax_ukernel__neon_u8;
    f32_vmul_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vmulc_minmax_ukernel__neon_u8;
    f32_vmul_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vmulc_minmax_ukernel__neon_u8;
    f32_vmul_config.init.f32_minmax = xnn_init_f32_minmax_scalar_params;
    f32_vmul_config.minmax.element_tile = 8;
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (!XNN_PLATFORM_MOBILE && hardware_config->use_x86_avx512f) {
      f32_vmul_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vmul_minmax_ukernel__avx512f_u32;
      f32_vmul_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vmulc_minmax_ukernel__avx512f_u32;
      f32_vmul_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vmulc_minmax_ukernel__avx512f_u32;
      f32_vmul_config.init.f32_minmax = xnn_init_f32_minmax_scalar_params;
      f32_vmul_config.minmax.element_tile = 32;
    } else if (hardware_config->use_x86_avx) {
      f32_vmul_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vmul_minmax_ukernel__avx_u16;
      f32_vmul_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vmulc_minmax_ukernel__avx_u16;
      f32_vmul_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vmulc_minmax_ukernel__avx_u16;
      f32_vmul_config.init.f32_minmax = xnn_init_f32_minmax_avx_params;
      f32_vmul_config.minmax.element_tile = 16;
    } else {
      f32_vmul_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vmul_minmax_ukernel__sse_u8;
      f32_vmul_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vmulc_minmax_ukernel__sse_u8;
      f32_vmul_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vmulc_minmax_ukernel__sse_u8;
      f32_vmul_config.init.f32_minmax = xnn_init_f32_minmax_sse_params;
      f32_vmul_config.minmax.element_tile = 8;
    }
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->is_x86) {
      f32_vmul_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vmul_minmax_ukernel__wasmsimd_x86_u16;
      f32_vmul_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vmulc_minmax_ukernel__wasmsimd_x86_u16;
      f32_vmul_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vmulc_minmax_ukernel__wasmsimd_x86_u16;
      f32_vmul_config.linear.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vmul_ukernel__wasmsimd_u16;
      f32_vmul_config.linear.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vmulc_ukernel__wasmsimd_u16;
      f32_vmul_config.linear.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vmulc_ukernel__wasmsimd_u16;
      f32_vmul_config.init.f32_minmax = xnn_init_f32_minmax_wasmsimd_params;
      f32_vmul_config.minmax.element_tile = 16;
      f32_vmul_config.linear.element_tile = 16;
    } else {
      f32_vmul_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vmul_minmax_ukernel__wasmsimd_arm_u16;
      f32_vmul_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vmulc_minmax_ukernel__wasmsimd_arm_u16;
      f32_vmul_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vmulc_minmax_ukernel__wasmsimd_arm_u16;
      f32_vmul_config.linear.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vmul_ukernel__wasmsimd_u16;
      f32_vmul_config.linear.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vmulc_ukernel__wasmsimd_u16;
      f32_vmul_config.linear.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vmulc_ukernel__wasmsimd_u16;
      f32_vmul_config.init.f32_minmax = xnn_init_f32_minmax_wasmsimd_params;
      f32_vmul_config.minmax.element_tile = 16;
      f32_vmul_config.linear.element_tile = 16;
    }
  #elif XNN_ARCH_WASM
    f32_vmul_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vmul_minmax_ukernel__wasm_u8;
    f32_vmul_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vmulc_minmax_ukernel__wasm_u8;
    f32_vmul_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vmulc_minmax_ukernel__wasm_u8;
    f32_vmul_config.init.f32_minmax = xnn_init_f32_minmax_scalar_params;
    f32_vmul_config.minmax.element_tile = 8;
  #elif XNN_ARCH_RISCV && XNN_ENABLE_RISCV_VECTOR
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    f32_vmul_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vmul_minmax_ukernel__rvv_u8v;
    f32_vmul_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vmulc_minmax_ukernel__rvv_u8v;
    f32_vmul_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vmulc_minmax_ukernel__rvv_u8v;
    f32_vmul_config.init.f32_minmax = xnn_init_f32_minmax_scalar_params;
    f32_vmul_config.minmax.element_tile = hardware_config->vlenb * 2;  // VLENB * (8 / sizeof(float))
  #else
    f32_vmul_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vmul_minmax_ukernel__scalar_u8;
    f32_vmul_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vmulc_minmax_ukernel__scalar_u8;
    f32_vmul_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vmulc_minmax_ukernel__scalar_u8;
    f32_vmul_config.init.f32_minmax = xnn_init_f32_minmax_scalar_params;
    f32_vmul_config.minmax.element_tile = 8;
  #endif
}

static void init_f32_vsub_config(void) {
  #if XNN_ARCH_ARM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon){
      f32_vsub_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vsub_minmax_ukernel__neon_u8;
      f32_vsub_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vsubc_minmax_ukernel__neon_u8;
      f32_vsub_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vrsubc_minmax_ukernel__neon_u8;
      f32_vsub_config.init.f32_minmax = xnn_init_f32_minmax_scalar_params;
      f32_vsub_config.minmax.element_tile = 8;
    } else if (!XNN_PLATFORM_MOBILE) {
      f32_vsub_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vsub_minmax_ukernel__scalar_u8;
      f32_vsub_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vsubc_minmax_ukernel__scalar_u8;
      f32_vsub_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vrsubc_minmax_ukernel__scalar_u8;
      f32_vsub_config.init.f32_minmax = xnn_init_f32_minmax_scalar_params;
      f32_vsub_config.minmax.element_tile = 8;
    }
  #elif XNN_ARCH_ARM64
    f32_vsub_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vsub_minmax_ukernel__neon_u8;
    f32_vsub_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vsubc_minmax_ukernel__neon_u8;
    f32_vsub_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vrsubc_minmax_ukernel__neon_u8;
    f32_vsub_config.init.f32_minmax = xnn_init_f32_minmax_scalar_params;
    f32_vsub_config.minmax.element_tile = 8;
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (!XNN_PLATFORM_MOBILE && hardware_config->use_x86_avx512f) {
      f32_vsub_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vsub_minmax_ukernel__avx512f_u32;
      f32_vsub_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vsubc_minmax_ukernel__avx512f_u32;
      f32_vsub_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vrsubc_minmax_ukernel__avx512f_u32;
      f32_vsub_config.init.f32_minmax = xnn_init_f32_minmax_scalar_params;
      f32_vsub_config.minmax.element_tile = 32;
    } else if (hardware_config->use_x86_avx) {
      f32_vsub_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vsub_minmax_ukernel__avx_u16;
      f32_vsub_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vsubc_minmax_ukernel__avx_u16;
      f32_vsub_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vrsubc_minmax_ukernel__avx_u16;
      f32_vsub_config.init.f32_minmax = xnn_init_f32_minmax_avx_params;
      f32_vsub_config.minmax.element_tile = 16;
    } else {
      f32_vsub_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vsub_minmax_ukernel__sse_u8;
      f32_vsub_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vsubc_minmax_ukernel__sse_u8;
      f32_vsub_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vrsubc_minmax_ukernel__sse_u8;
      f32_vsub_config.init.f32_minmax = xnn_init_f32_minmax_sse_params;
      f32_vsub_config.minmax.element_tile = 8;
    }
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->is_x86) {
      f32_vsub_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vsub_minmax_ukernel__wasmsimd_x86_u16;
      f32_vsub_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vsubc_minmax_ukernel__wasmsimd_x86_u16;
      f32_vsub_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vrsubc_minmax_ukernel__wasmsimd_x86_u16;
      f32_vsub_config.linear.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vsub_ukernel__wasmsimd_u16;
      f32_vsub_config.linear.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vsubc_ukernel__wasmsimd_u16;
      f32_vsub_config.linear.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vrsubc_ukernel__wasmsimd_u16;
      f32_vsub_config.init.f32_minmax = xnn_init_f32_minmax_wasmsimd_params;
      f32_vsub_config.minmax.element_tile = 16;
      f32_vsub_config.linear.element_tile = 16;
    } else {
      f32_vsub_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vsub_minmax_ukernel__wasmsimd_arm_u16;
      f32_vsub_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vsubc_minmax_ukernel__wasmsimd_arm_u16;
      f32_vsub_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vrsubc_minmax_ukernel__wasmsimd_arm_u16;
      f32_vsub_config.linear.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vsub_ukernel__wasmsimd_u16;
      f32_vsub_config.linear.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vsubc_ukernel__wasmsimd_u16;
      f32_vsub_config.linear.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vrsubc_ukernel__wasmsimd_u16;
      f32_vsub_config.init.f32_minmax = xnn_init_f32_minmax_wasmsimd_params;
      f32_vsub_config.minmax.element_tile = 16;
      f32_vsub_config.linear.element_tile = 16;
    }
  #elif XNN_ARCH_WASM
    f32_vsub_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vsub_minmax_ukernel__wasm_u8;
    f32_vsub_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vsubc_minmax_ukernel__wasm_u8;
    f32_vsub_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vrsubc_minmax_ukernel__wasm_u8;
    f32_vsub_config.init.f32_minmax = xnn_init_f32_minmax_scalar_params;
    f32_vsub_config.minmax.element_tile = 8;
  #elif XNN_ARCH_RISCV && XNN_ENABLE_RISCV_VECTOR
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    f32_vsub_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vsub_minmax_ukernel__rvv_u8v;
    f32_vsub_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vsubc_minmax_ukernel__rvv_u8v;
    f32_vsub_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vrsubc_minmax_ukernel__rvv_u8v;
    f32_vsub_config.init.f32_minmax = xnn_init_f32_minmax_scalar_params;
    f32_vsub_config.minmax.element_tile = hardware_config->vlenb * 2;  // VLENB * (8 / sizeof(float))
  #else
    f32_vsub_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vsub_minmax_ukernel__scalar_u8;
    f32_vsub_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vsubc_minmax_ukernel__scalar_u8;
    f32_vsub_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vrsubc_minmax_ukernel__scalar_u8;
    f32_vsub_config.init.f32_minmax = xnn_init_f32_minmax_scalar_params;
    f32_vsub_config.minmax.element_tile = 8;
  #endif
}

static void init_f32_vsqrdiff_config(void) {
  #if XNN_ARCH_ARM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon){
      f32_vsqrdiff_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vsqrdiff_ukernel__neon_u8;
      f32_vsqrdiff_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vsqrdiffc_ukernel__neon_u8;
      f32_vsqrdiff_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vsqrdiffc_ukernel__neon_u8;
      f32_vsqrdiff_config.minmax.element_tile = 8;
    } else if (!XNN_PLATFORM_MOBILE) {
      f32_vsqrdiff_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vsqrdiff_ukernel__scalar_u8;
      f32_vsqrdiff_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vsqrdiffc_ukernel__scalar_u8;
      f32_vsqrdiff_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vsqrdiffc_ukernel__scalar_u8;
      f32_vsqrdiff_config.minmax.element_tile = 8;
    }
  #elif XNN_ARCH_ARM64
    f32_vsqrdiff_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vsqrdiff_ukernel__neon_u8;
    f32_vsqrdiff_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vsqrdiffc_ukernel__neon_u8;
    f32_vsqrdiff_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vsqrdiffc_ukernel__neon_u8;
    f32_vsqrdiff_config.minmax.element_tile = 8;
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (!XNN_PLATFORM_MOBILE && hardware_config->use_x86_avx512f) {
      f32_vsqrdiff_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vsqrdiff_ukernel__avx512f_u32;
      f32_vsqrdiff_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vsqrdiffc_ukernel__avx512f_u32;
      f32_vsqrdiff_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vsqrdiffc_ukernel__avx512f_u32;
      f32_vsqrdiff_config.minmax.element_tile = 32;
    } else if (hardware_config->use_x86_avx) {
      f32_vsqrdiff_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vsqrdiff_ukernel__avx_u16;
      f32_vsqrdiff_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vsqrdiffc_ukernel__avx_u16;
      f32_vsqrdiff_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vsqrdiffc_ukernel__avx_u16;
      f32_vsqrdiff_config.minmax.element_tile = 16;
    } else {
      f32_vsqrdiff_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vsqrdiff_ukernel__sse_u8;
      f32_vsqrdiff_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vsqrdiffc_ukernel__sse_u8;
      f32_vsqrdiff_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vsqrdiffc_ukernel__sse_u8;
      f32_vsqrdiff_config.minmax.element_tile = 8;
    }
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    f32_vsqrdiff_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vsqrdiff_ukernel__wasmsimd_u16;
    f32_vsqrdiff_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vsqrdiffc_ukernel__wasmsimd_u16;
    f32_vsqrdiff_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vsqrdiffc_ukernel__wasmsimd_u16;
    f32_vsqrdiff_config.minmax.element_tile = 16;
  #elif XNN_ARCH_RISCV && XNN_ENABLE_RISCV_VECTOR
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    f32_vsqrdiff_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vsqrdiff_ukernel__rvv_u8v;
    f32_vsqrdiff_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vsqrdiffc_ukernel__rvv_u8v;
    f32_vsqrdiff_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vsqrdiffc_ukernel__rvv_u8v;
    f32_vsqrdiff_config.minmax.element_tile = hardware_config->vlenb * 2;  // VLENB * (8 / sizeof(float))
  #else
    f32_vsqrdiff_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vsqrdiff_ukernel__scalar_u8;
    f32_vsqrdiff_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vsqrdiffc_ukernel__scalar_u8;
    f32_vsqrdiff_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vsqrdiffc_ukernel__scalar_u8;
    f32_vsqrdiff_config.minmax.element_tile = 8;
  #endif
}

static void init_qs8_vadd_config(void) {
  #if XNN_ARCH_ARM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon){
      qs8_vadd_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_qs8_vadd_minmax_ukernel__neon_ld64_u16;
      qs8_vadd_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qs8_vaddc_minmax_ukernel__neon_ld64_u16;
      qs8_vadd_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qs8_vaddc_minmax_ukernel__neon_ld64_u16;
      qs8_vadd_config.init.qs8_add = xnn_init_qs8_add_minmax_scalar_params;
      qs8_vadd_config.minmax.element_tile = 16;
    } else if (!XNN_PLATFORM_MOBILE) {
      qs8_vadd_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_qs8_vadd_minmax_ukernel__scalar_u1;
      qs8_vadd_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qs8_vaddc_minmax_ukernel__scalar_u1;
      qs8_vadd_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qs8_vaddc_minmax_ukernel__scalar_u1;
      qs8_vadd_config.init.qs8_add = xnn_init_qs8_add_minmax_scalar_params;
      qs8_vadd_config.minmax.element_tile = 1;
    }
  #elif XNN_ARCH_ARM64
    qs8_vadd_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_qs8_vadd_minmax_ukernel__neon_ld64_u32;
    qs8_vadd_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qs8_vaddc_minmax_ukernel__neon_ld64_u32;
    qs8_vadd_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qs8_vaddc_minmax_ukernel__neon_ld64_u32;
    qs8_vadd_config.init.qs8_add = xnn_init_qs8_add_minmax_scalar_params;
    qs8_vadd_config.minmax.element_tile = 32;
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (!XNN_PLATFORM_MOBILE && hardware_config->use_x86_avx512skx) {
      qs8_vadd_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_qs8_vadd_minmax_ukernel__avx512skx_mul32_ld128_u16;
      qs8_vadd_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qs8_vaddc_minmax_ukernel__avx512skx_mul32_ld128_u16;
      qs8_vadd_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qs8_vaddc_minmax_ukernel__avx512skx_mul32_ld128_u16;
      qs8_vadd_config.init.qs8_add = xnn_init_qs8_add_minmax_scalar_params;
      qs8_vadd_config.minmax.element_tile = 16;
    } else if (hardware_config->use_x86_avx2) {
      qs8_vadd_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_qs8_vadd_minmax_ukernel__avx2_mul32_ld64_u16;
      qs8_vadd_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qs8_vaddc_minmax_ukernel__avx2_mul32_ld64_u16;
      qs8_vadd_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qs8_vaddc_minmax_ukernel__avx2_mul32_ld64_u16;
      qs8_vadd_config.init.qs8_add = xnn_init_qs8_add_minmax_scalar_params;
      qs8_vadd_config.minmax.element_tile = 16;
    } else if (hardware_config->use_x86_avx) {
      qs8_vadd_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_qs8_vadd_minmax_ukernel__avx_mul32_ld32_u8;
      qs8_vadd_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qs8_vaddc_minmax_ukernel__avx_mul32_ld32_u8;
      qs8_vadd_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qs8_vaddc_minmax_ukernel__avx_mul32_ld32_u8;
      qs8_vadd_config.init.qs8_add = xnn_init_qs8_add_minmax_scalar_params;
      qs8_vadd_config.minmax.element_tile = 8;
    } else if (hardware_config->use_x86_sse4_1) {
      qs8_vadd_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_qs8_vadd_minmax_ukernel__sse41_mul16_ld64_u8;
      qs8_vadd_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qs8_vaddc_minmax_ukernel__sse41_mul16_ld64_u8;
      qs8_vadd_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qs8_vaddc_minmax_ukernel__sse41_mul16_ld64_u8;
      qs8_vadd_config.init.qs8_add = xnn_init_qs8_add_minmax_scalar_params;
      qs8_vadd_config.minmax.element_tile = 8;
    } else {
      qs8_vadd_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_qs8_vadd_minmax_ukernel__sse2_mul16_ld64_u8;
      qs8_vadd_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qs8_vaddc_minmax_ukernel__sse2_mul16_ld64_u8;
      qs8_vadd_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qs8_vaddc_minmax_ukernel__sse2_mul16_ld64_u8;
      qs8_vadd_config.init.qs8_add = xnn_init_qs8_add_minmax_scalar_params;
      qs8_vadd_config.minmax.element_tile = 8;
    }
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    qs8_vadd_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_qs8_vadd_minmax_ukernel__wasmsimd_u32;
    qs8_vadd_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qs8_vaddc_minmax_ukernel__wasmsimd_u32;
    qs8_vadd_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qs8_vaddc_minmax_ukernel__wasmsimd_u32;
    qs8_vadd_config.init.qs8_add = xnn_init_qs8_add_minmax_scalar_params;
    qs8_vadd_config.minmax.element_tile = 32;
  #else
    qs8_vadd_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_qs8_vadd_minmax_ukernel__scalar_u4;
    qs8_vadd_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qs8_vaddc_minmax_ukernel__scalar_u4;
    qs8_vadd_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qs8_vaddc_minmax_ukernel__scalar_u4;
    qs8_vadd_config.init.qs8_add = xnn_init_qs8_add_minmax_scalar_params;
    qs8_vadd_config.minmax.element_tile = 4;
  #endif
}

static void init_qs8_vmul_config(void) {
  #if XNN_ARCH_ARM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon){
      qs8_vmul_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_qs8_vmul_minmax_rndnu_ukernel__neon_ld64_u16;
      qs8_vmul_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qs8_vmulc_minmax_rndnu_ukernel__neon_ld64_u16;
      qs8_vmul_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qs8_vmulc_minmax_rndnu_ukernel__neon_ld64_u16;
      qs8_vmul_config.init.qs8_mul = xnn_init_qs8_mul_minmax_rndnu_neon_params;
      qs8_vmul_config.minmax.element_tile = 16;
    } else if (!XNN_PLATFORM_MOBILE) {
      qs8_vmul_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_qs8_vmul_minmax_fp32_ukernel__scalar_u4;
      qs8_vmul_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qs8_vmulc_minmax_fp32_ukernel__scalar_u4;
      qs8_vmul_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qs8_vmulc_minmax_fp32_ukernel__scalar_u4;
      qs8_vmul_config.init.qs8_mul = xnn_init_qs8_mul_minmax_scalar_params;
      qs8_vmul_config.minmax.element_tile = 4;
    }
  #elif XNN_ARCH_ARM64
    qs8_vmul_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_qs8_vmul_minmax_rndnu_ukernel__neon_ld64_u16;
    qs8_vmul_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qs8_vmulc_minmax_rndnu_ukernel__neon_ld64_u16;
    qs8_vmul_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qs8_vmulc_minmax_rndnu_ukernel__neon_ld64_u16;
    qs8_vmul_config.init.qs8_mul = xnn_init_qs8_mul_minmax_rndnu_neon_params;
    qs8_vmul_config.minmax.element_tile = 16;
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_x86_avx) {
      qs8_vmul_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_qs8_vmul_minmax_fp32_ukernel__avx_mul16_ld64_u16;
      qs8_vmul_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qs8_vmulc_minmax_fp32_ukernel__avx_mul16_ld64_u16;
      qs8_vmul_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qs8_vmulc_minmax_fp32_ukernel__avx_mul16_ld64_u16;
      qs8_vmul_config.init.qs8_mul = xnn_init_qs8_mul_minmax_scalar_params;
      qs8_vmul_config.minmax.element_tile = 16;
    } else if (hardware_config->use_x86_sse4_1) {
      qs8_vmul_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_qs8_vmul_minmax_fp32_ukernel__sse41_mul16_ld64_u16;
      qs8_vmul_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qs8_vmulc_minmax_fp32_ukernel__sse41_mul16_ld64_u16;
      qs8_vmul_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qs8_vmulc_minmax_fp32_ukernel__sse41_mul16_ld64_u16;
      qs8_vmul_config.init.qs8_mul = xnn_init_qs8_mul_minmax_scalar_params;
      qs8_vmul_config.minmax.element_tile = 16;
    } else {
      qs8_vmul_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_qs8_vmul_minmax_fp32_ukernel__sse2_mul16_ld64_u8;
      qs8_vmul_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qs8_vmulc_minmax_fp32_ukernel__sse2_mul16_ld64_u8;
      qs8_vmul_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qs8_vmulc_minmax_fp32_ukernel__sse2_mul16_ld64_u8;
      qs8_vmul_config.init.qs8_mul = xnn_init_qs8_mul_minmax_scalar_params;
      qs8_vmul_config.minmax.element_tile = 8;
    }
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    qs8_vmul_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_qs8_vmul_minmax_fp32_ukernel__wasmsimd_mul32_ld64_u8;
    qs8_vmul_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qs8_vmulc_minmax_fp32_ukernel__wasmsimd_mul32_ld64_u8;
    qs8_vmul_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qs8_vmulc_minmax_fp32_ukernel__wasmsimd_mul32_ld64_u8;
    qs8_vmul_config.init.qs8_mul = xnn_init_qs8_mul_minmax_scalar_params;
    qs8_vmul_config.minmax.element_tile = 8;
  #elif XNN_ARCH_RISCV && XNN_ENABLE_RISCV_VECTOR
    qs8_vmul_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_qs8_vmul_minmax_fp32_ukernel__rvv_u2v;
    qs8_vmul_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qs8_vmulc_minmax_fp32_ukernel__rvv_u2v;
    qs8_vmul_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qs8_vmulc_minmax_fp32_ukernel__rvv_u2v;
    qs8_vmul_config.init.qs8_mul = xnn_init_qs8_mul_minmax_scalar_params;
    qs8_vmul_config.minmax.element_tile = 2;
  #else
    qs8_vmul_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_qs8_vmul_minmax_fp32_ukernel__scalar_u4;
    qs8_vmul_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qs8_vmulc_minmax_fp32_ukernel__scalar_u4;
    qs8_vmul_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qs8_vmulc_minmax_fp32_ukernel__scalar_u4;
    qs8_vmul_config.init.qs8_mul = xnn_init_qs8_mul_minmax_scalar_params;
    qs8_vmul_config.minmax.element_tile = 4;
  #endif
}

static void init_qu8_vadd_config(void) {
  #if XNN_ARCH_ARM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon){
      qu8_vadd_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_qu8_vadd_minmax_ukernel__neon_ld64_u16;
      qu8_vadd_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qu8_vaddc_minmax_ukernel__neon_ld64_u16;
      qu8_vadd_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qu8_vaddc_minmax_ukernel__neon_ld64_u16;
      qu8_vadd_config.init.qu8_add = xnn_init_qu8_add_minmax_scalar_params;
      qu8_vadd_config.minmax.element_tile = 8;
    } else if (!XNN_PLATFORM_MOBILE) {
      qu8_vadd_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_qu8_vadd_minmax_ukernel__scalar_u1;
      qu8_vadd_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qu8_vaddc_minmax_ukernel__scalar_u1;
      qu8_vadd_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qu8_vaddc_minmax_ukernel__scalar_u1;
      qu8_vadd_config.init.qu8_add = xnn_init_qu8_add_minmax_scalar_params;
      qu8_vadd_config.minmax.element_tile = 1;
    }
  #elif XNN_ARCH_ARM64
    qu8_vadd_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_qu8_vadd_minmax_ukernel__neon_ld64_u32;
    qu8_vadd_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qu8_vaddc_minmax_ukernel__neon_ld64_u32;
    qu8_vadd_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qu8_vaddc_minmax_ukernel__neon_ld64_u32;
    qu8_vadd_config.init.qu8_add = xnn_init_qu8_add_minmax_scalar_params;
    qu8_vadd_config.minmax.element_tile = 8;
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (!XNN_PLATFORM_MOBILE && hardware_config->use_x86_avx512skx) {
      qu8_vadd_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_qu8_vadd_minmax_ukernel__avx512skx_mul32_ld128_u16;
      qu8_vadd_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qu8_vaddc_minmax_ukernel__avx512skx_mul32_ld128_u16;
      qu8_vadd_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qu8_vaddc_minmax_ukernel__avx512skx_mul32_ld128_u16;
      qu8_vadd_config.init.qu8_add = xnn_init_qu8_add_minmax_scalar_params;
      qu8_vadd_config.minmax.element_tile = 16;
    } else if (hardware_config->use_x86_avx2) {
      qu8_vadd_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_qu8_vadd_minmax_ukernel__avx2_mul32_ld64_u16;
      qu8_vadd_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qu8_vaddc_minmax_ukernel__avx2_mul32_ld64_u16;
      qu8_vadd_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qu8_vaddc_minmax_ukernel__avx2_mul32_ld64_u16;
      qu8_vadd_config.init.qu8_add = xnn_init_qu8_add_minmax_scalar_params;
      qu8_vadd_config.minmax.element_tile = 16;
    } else if (hardware_config->use_x86_avx) {
      qu8_vadd_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_qu8_vadd_minmax_ukernel__avx_mul32_ld32_u8;
      qu8_vadd_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qu8_vaddc_minmax_ukernel__avx_mul32_ld32_u8;
      qu8_vadd_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qu8_vaddc_minmax_ukernel__avx_mul32_ld32_u8;
      qu8_vadd_config.init.qu8_add = xnn_init_qu8_add_minmax_scalar_params;
      qu8_vadd_config.minmax.element_tile = 8;
    } else if (hardware_config->use_x86_sse4_1) {
      qu8_vadd_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_qu8_vadd_minmax_ukernel__sse41_mul16_ld64_u8;
      qu8_vadd_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qu8_vaddc_minmax_ukernel__sse41_mul16_ld64_u8;
      qu8_vadd_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qu8_vaddc_minmax_ukernel__sse41_mul16_ld64_u8;
      qu8_vadd_config.init.qu8_add = xnn_init_qu8_add_minmax_scalar_params;
      qu8_vadd_config.minmax.element_tile = 8;
    } else {
      qu8_vadd_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_qu8_vadd_minmax_ukernel__sse2_mul16_ld64_u8;
      qu8_vadd_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qu8_vaddc_minmax_ukernel__sse2_mul16_ld64_u8;
      qu8_vadd_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qu8_vaddc_minmax_ukernel__sse2_mul16_ld64_u8;
      qu8_vadd_config.init.qu8_add = xnn_init_qu8_add_minmax_scalar_params;
      qu8_vadd_config.minmax.element_tile = 8;
    }
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    qu8_vadd_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_qu8_vadd_minmax_ukernel__wasmsimd_u32;
    qu8_vadd_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qu8_vaddc_minmax_ukernel__wasmsimd_u32;
    qu8_vadd_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qu8_vaddc_minmax_ukernel__wasmsimd_u32;
    qu8_vadd_config.init.qu8_add = xnn_init_qu8_add_minmax_scalar_params;
    qu8_vadd_config.minmax.element_tile = 32;
  #else
    qu8_vadd_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_qu8_vadd_minmax_ukernel__scalar_u4;
    qu8_vadd_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qu8_vaddc_minmax_ukernel__scalar_u4;
    qu8_vadd_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qu8_vaddc_minmax_ukernel__scalar_u4;
    qu8_vadd_config.init.qu8_add = xnn_init_qu8_add_minmax_scalar_params;
    qu8_vadd_config.minmax.element_tile = 4;
  #endif
}

static void init_qu8_vmul_config(void) {
  #if XNN_ARCH_ARM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon){
      qu8_vmul_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_qu8_vmul_minmax_rndnu_ukernel__neon_ld64_u16;
      qu8_vmul_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qu8_vmulc_minmax_rndnu_ukernel__neon_ld64_u16;
      qu8_vmul_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qu8_vmulc_minmax_rndnu_ukernel__neon_ld64_u16;
      qu8_vmul_config.init.qu8_mul = xnn_init_qu8_mul_minmax_rndnu_neon_params;
      qu8_vmul_config.minmax.element_tile = 16;
    } else if (!XNN_PLATFORM_MOBILE) {
      qu8_vmul_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_qu8_vmul_minmax_fp32_ukernel__scalar_u4;
      qu8_vmul_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qu8_vmulc_minmax_fp32_ukernel__scalar_u4;
      qu8_vmul_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qu8_vmulc_minmax_fp32_ukernel__scalar_u4;
      qu8_vmul_config.init.qu8_mul = xnn_init_qu8_mul_minmax_scalar_params;
      qu8_vmul_config.minmax.element_tile = 4;
    }
  #elif XNN_ARCH_ARM64
    qu8_vmul_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_qu8_vmul_minmax_rndnu_ukernel__neon_ld64_u16;
    qu8_vmul_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qu8_vmulc_minmax_rndnu_ukernel__neon_ld64_u16;
    qu8_vmul_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qu8_vmulc_minmax_rndnu_ukernel__neon_ld64_u16;
    qu8_vmul_config.init.qu8_mul = xnn_init_qu8_mul_minmax_rndnu_neon_params;
    qu8_vmul_config.minmax.element_tile = 16;
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_x86_avx) {
      qu8_vmul_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_qu8_vmul_minmax_fp32_ukernel__avx_mul16_ld64_u16;
      qu8_vmul_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qu8_vmulc_minmax_fp32_ukernel__avx_mul16_ld64_u16;
      qu8_vmul_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qu8_vmulc_minmax_fp32_ukernel__avx_mul16_ld64_u16;
      qu8_vmul_config.init.qu8_mul = xnn_init_qu8_mul_minmax_scalar_params;
      qu8_vmul_config.minmax.element_tile = 16;
    } else if (hardware_config->use_x86_sse4_1) {
      qu8_vmul_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_qu8_vmul_minmax_fp32_ukernel__sse41_mul16_ld64_u16;
      qu8_vmul_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qu8_vmulc_minmax_fp32_ukernel__sse41_mul16_ld64_u16;
      qu8_vmul_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qu8_vmulc_minmax_fp32_ukernel__sse41_mul16_ld64_u16;
      qu8_vmul_config.init.qu8_mul = xnn_init_qu8_mul_minmax_scalar_params;
      qu8_vmul_config.minmax.element_tile = 16;
    } else {
      qu8_vmul_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_qu8_vmul_minmax_fp32_ukernel__sse2_mul16_ld64_u8;
      qu8_vmul_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qu8_vmulc_minmax_fp32_ukernel__sse2_mul16_ld64_u8;
      qu8_vmul_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qu8_vmulc_minmax_fp32_ukernel__sse2_mul16_ld64_u8;
      qu8_vmul_config.init.qu8_mul = xnn_init_qu8_mul_minmax_scalar_params;
      qu8_vmul_config.minmax.element_tile = 8;
    }
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    qu8_vmul_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_qu8_vmul_minmax_fp32_ukernel__wasmsimd_mul32_ld64_u8;
    qu8_vmul_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qu8_vmulc_minmax_fp32_ukernel__wasmsimd_mul32_ld64_u8;
    qu8_vmul_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qu8_vmulc_minmax_fp32_ukernel__wasmsimd_mul32_ld64_u8;
    qu8_vmul_config.init.qu8_mul = xnn_init_qu8_mul_minmax_scalar_params;
    qu8_vmul_config.minmax.element_tile = 8;
  #elif XNN_ARCH_RISCV && XNN_ENABLE_RISCV_VECTOR
    qu8_vmul_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_qu8_vmul_minmax_fp32_ukernel__rvv_u2v;
    qu8_vmul_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qu8_vmulc_minmax_fp32_ukernel__rvv_u2v;
    qu8_vmul_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qu8_vmulc_minmax_fp32_ukernel__rvv_u2v;
    qu8_vmul_config.init.qu8_mul = xnn_init_qu8_mul_minmax_scalar_params;
    qu8_vmul_config.minmax.element_tile = 2;
  #else
    qu8_vmul_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_qu8_vmul_minmax_fp32_ukernel__scalar_u4;
    qu8_vmul_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qu8_vmulc_minmax_fp32_ukernel__scalar_u4;
    qu8_vmul_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qu8_vmulc_minmax_fp32_ukernel__scalar_u4;
    qu8_vmul_config.init.qu8_mul = xnn_init_qu8_mul_minmax_scalar_params;
    qu8_vmul_config.minmax.element_tile = 4;
  #endif
}

const struct xnn_binary_elementwise_config* xnn_init_f16_vadd_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL || !xnn_is_f16_compatible_config(hardware_config)) {
    return NULL;
  }
  XNN_INIT_ONCE(f16_vadd);
  return &f16_vadd_config;
}

const struct xnn_binary_elementwise_config* xnn_init_f16_vdiv_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL || !xnn_is_f16_compatible_config(hardware_config)) {
    return NULL;
  }
  XNN_INIT_ONCE(f16_vdiv);
  return &f16_vdiv_config;
}

const struct xnn_binary_elementwise_config* xnn_init_f16_vmax_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL || !xnn_is_f16_compatible_config(hardware_config)) {
    return NULL;
  }
  XNN_INIT_ONCE(f16_vmax);
  return &f16_vmax_config;
}

const struct xnn_binary_elementwise_config* xnn_init_f16_vmin_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL || !xnn_is_f16_compatible_config(hardware_config)) {
    return NULL;
  }
  XNN_INIT_ONCE(f16_vmin);
  return &f16_vmin_config;
}

const struct xnn_binary_elementwise_config* xnn_init_f16_vmul_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL || !xnn_is_f16_compatible_config(hardware_config)) {
    return NULL;
  }
  XNN_INIT_ONCE(f16_vmul);
  return &f16_vmul_config;
}

const struct xnn_binary_elementwise_config* xnn_init_f16_vsub_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL || !xnn_is_f16_compatible_config(hardware_config)) {
    return NULL;
  }
  XNN_INIT_ONCE(f16_vsub);
  return &f16_vsub_config;
}

const struct xnn_binary_elementwise_config* xnn_init_f16_vsqrdiff_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL || !xnn_is_f16_compatible_config(hardware_config)) {
    return NULL;
  }
  XNN_INIT_ONCE(f16_vsqrdiff);
  return &f16_vsqrdiff_config;
}

const struct xnn_binary_elementwise_config* xnn_init_f32_vadd_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  XNN_INIT_ONCE(f32_vadd);
  return &f32_vadd_config;
}

const struct xnn_binary_elementwise_config* xnn_init_f32_vcopysign_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  XNN_INIT_ONCE(f32_vcopysign);
  return &f32_vcopysign_config;
}

const struct xnn_binary_elementwise_config* xnn_init_s32_vmul_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  XNN_INIT_ONCE(s32_vmul);
  return &s32_vmul_config;
}

const struct xnn_binary_elementwise_config* xnn_init_f32_vdiv_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  XNN_INIT_ONCE(f32_vdiv);
  return &f32_vdiv_config;
}

const struct xnn_binary_elementwise_config* xnn_init_f32_vmax_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  XNN_INIT_ONCE(f32_vmax);
  return &f32_vmax_config;
}

const struct xnn_binary_elementwise_config* xnn_init_f32_vmin_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  XNN_INIT_ONCE(f32_vmin);
  return &f32_vmin_config;
}

const struct xnn_binary_elementwise_config* xnn_init_f32_vmul_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  XNN_INIT_ONCE(f32_vmul);
  return &f32_vmul_config;
}

const struct xnn_binary_elementwise_config* xnn_init_f32_vsub_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  XNN_INIT_ONCE(f32_vsub);
  return &f32_vsub_config;
}

const struct xnn_binary_elementwise_config* xnn_init_f32_vsqrdiff_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  XNN_INIT_ONCE(f32_vsqrdiff);
  return &f32_vsqrdiff_config;
}

const struct xnn_binary_elementwise_config* xnn_init_qs8_vadd_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  XNN_INIT_ONCE(qs8_vadd);
  return &qs8_vadd_config;
}

const struct xnn_binary_elementwise_config* xnn_init_qs8_vmul_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  XNN_INIT_ONCE(qs8_vmul);
  return &qs8_vmul_config;
}

const struct xnn_binary_elementwise_config* xnn_init_qu8_vadd_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  XNN_INIT_ONCE(qu8_vadd);
  return &qu8_vadd_config;
}

const struct xnn_binary_elementwise_config* xnn_init_qu8_vmul_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  XNN_INIT_ONCE(qu8_vmul);
  return &qu8_vmul_config;
}
