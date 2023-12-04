// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>

#ifdef _WIN32
  #include <windows.h>
#else
  #include <pthread.h>
#endif

#include <xnnpack/common.h>
#include <xnnpack/config.h>
#include <xnnpack/microparams-init.h>
#include <xnnpack/vbinary.h>

static struct xnn_binary_elementwise_config f16_vadd_config = {0};
static struct xnn_binary_elementwise_config f16_vdiv_config = {0};
static struct xnn_binary_elementwise_config f16_vmax_config = {0};
static struct xnn_binary_elementwise_config f16_vmin_config = {0};
static struct xnn_binary_elementwise_config f16_vmul_config = {0};
static struct xnn_binary_elementwise_config f16_vsub_config = {0};
static struct xnn_binary_elementwise_config f16_vsqrdiff_config = {0};

static struct xnn_binary_elementwise_config f32_vadd_config = {0};
static struct xnn_binary_elementwise_config f32_vdiv_config = {0};
static struct xnn_binary_elementwise_config f32_vmax_config = {0};
static struct xnn_binary_elementwise_config f32_vmin_config = {0};
static struct xnn_binary_elementwise_config f32_vmul_config = {0};
static struct xnn_binary_elementwise_config f32_vsub_config = {0};
static struct xnn_binary_elementwise_config f32_vsqrdiff_config = {0};

static struct xnn_binary_elementwise_config qs8_vadd_config = {0};
static struct xnn_binary_elementwise_config qs8_vmul_config = {0};

static struct xnn_binary_elementwise_config qu8_vadd_config = {0};
static struct xnn_binary_elementwise_config qu8_vmul_config = {0};

#if XNN_PLATFORM_WINDOWS
  static INIT_ONCE init_guard_f16_vadd = INIT_ONCE_STATIC_INIT;
  static INIT_ONCE init_guard_f16_vdiv = INIT_ONCE_STATIC_INIT;
  static INIT_ONCE init_guard_f16_vmax = INIT_ONCE_STATIC_INIT;
  static INIT_ONCE init_guard_f16_vmin = INIT_ONCE_STATIC_INIT;
  static INIT_ONCE init_guard_f16_vmul = INIT_ONCE_STATIC_INIT;
  static INIT_ONCE init_guard_f16_vsub = INIT_ONCE_STATIC_INIT;
  static INIT_ONCE init_guard_f16_vsqrdiff = INIT_ONCE_STATIC_INIT;
  static INIT_ONCE init_guard_f32_vadd = INIT_ONCE_STATIC_INIT;
  static INIT_ONCE init_guard_f32_vdiv = INIT_ONCE_STATIC_INIT;
  static INIT_ONCE init_guard_f32_vmax = INIT_ONCE_STATIC_INIT;
  static INIT_ONCE init_guard_f32_vmin = INIT_ONCE_STATIC_INIT;
  static INIT_ONCE init_guard_f32_vmul = INIT_ONCE_STATIC_INIT;
  static INIT_ONCE init_guard_f32_vsub = INIT_ONCE_STATIC_INIT;
  static INIT_ONCE init_guard_f32_vsqrdiff = INIT_ONCE_STATIC_INIT;
  static INIT_ONCE init_guard_qs8_vadd = INIT_ONCE_STATIC_INIT;
  static INIT_ONCE init_guard_qs8_vmul = INIT_ONCE_STATIC_INIT;
  static INIT_ONCE init_guard_qu8_vadd = INIT_ONCE_STATIC_INIT;
  static INIT_ONCE init_guard_qu8_vmul = INIT_ONCE_STATIC_INIT;
#else
  static pthread_once_t init_guard_f16_vadd = PTHREAD_ONCE_INIT;
  static pthread_once_t init_guard_f16_vdiv = PTHREAD_ONCE_INIT;
  static pthread_once_t init_guard_f16_vmax = PTHREAD_ONCE_INIT;
  static pthread_once_t init_guard_f16_vmin = PTHREAD_ONCE_INIT;
  static pthread_once_t init_guard_f16_vmul = PTHREAD_ONCE_INIT;
  static pthread_once_t init_guard_f16_vsub = PTHREAD_ONCE_INIT;
  static pthread_once_t init_guard_f16_vsqrdiff = PTHREAD_ONCE_INIT;
  static pthread_once_t init_guard_f32_vadd = PTHREAD_ONCE_INIT;
  static pthread_once_t init_guard_f32_vdiv = PTHREAD_ONCE_INIT;
  static pthread_once_t init_guard_f32_vmax = PTHREAD_ONCE_INIT;
  static pthread_once_t init_guard_f32_vmin = PTHREAD_ONCE_INIT;
  static pthread_once_t init_guard_f32_vmul = PTHREAD_ONCE_INIT;
  static pthread_once_t init_guard_f32_vsub = PTHREAD_ONCE_INIT;
  static pthread_once_t init_guard_f32_vsqrdiff = PTHREAD_ONCE_INIT;
  static pthread_once_t init_guard_qs8_vadd = PTHREAD_ONCE_INIT;
  static pthread_once_t init_guard_qs8_vmul = PTHREAD_ONCE_INIT;
  static pthread_once_t init_guard_qu8_vadd = PTHREAD_ONCE_INIT;
  static pthread_once_t init_guard_qu8_vmul = PTHREAD_ONCE_INIT;
#endif


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
    if (hardware_config->use_x86_avx2) {
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
    if (hardware_config->use_x86_avx2) {
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
    if (hardware_config->use_x86_avx2) {
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
    if (hardware_config->use_x86_avx2) {
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
    if (hardware_config->use_x86_avx2) {
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
    if (hardware_config->use_x86_avx2) {
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
    if (hardware_config->use_x86_avx2) {
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
  #elif XNN_ARCH_RISCV
    f32_vadd_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vadd_minmax_ukernel__scalar_u8;
    f32_vadd_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vaddc_minmax_ukernel__scalar_u8;
    f32_vadd_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vaddc_minmax_ukernel__scalar_u8;
    f32_vadd_config.init.f32_minmax = xnn_init_f32_minmax_scalar_params;
    f32_vadd_config.minmax.element_tile = 8;
  #elif XNN_ARCH_PPC64
    f32_vadd_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vadd_minmax_ukernel__scalar_u8;
    f32_vadd_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vaddc_minmax_ukernel__scalar_u8;
    f32_vadd_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vaddc_minmax_ukernel__scalar_u8;
    f32_vadd_config.init.f32_minmax = xnn_init_f32_minmax_scalar_params;
    f32_vadd_config.minmax.element_tile = 8;
  #else
    #error "Unsupported architecture"
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
  #elif XNN_ARCH_RISCV
    f32_vdiv_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vdiv_minmax_ukernel__scalar_u2;
    f32_vdiv_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vdivc_minmax_ukernel__scalar_u2;
    f32_vdiv_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vrdivc_minmax_ukernel__scalar_u2;
    f32_vdiv_config.init.f32_minmax = xnn_init_f32_minmax_scalar_params;
    f32_vdiv_config.minmax.element_tile = 2;
  #elif XNN_ARCH_PPC64
    f32_vdiv_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vdiv_minmax_ukernel__scalar_u2;
    f32_vdiv_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vdivc_minmax_ukernel__scalar_u2;
    f32_vdiv_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vrdivc_minmax_ukernel__scalar_u2;
    f32_vdiv_config.init.f32_minmax = xnn_init_f32_minmax_scalar_params;
    f32_vdiv_config.minmax.element_tile = 2;
  #else
    #error "Unsupported architecture"
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
      f32_vmax_config.init.f32_default = xnn_init_f32_default_avx_params;
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
  #elif XNN_ARCH_RISCV
    f32_vmax_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vmax_ukernel__scalar_u8;
    f32_vmax_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vmaxc_ukernel__scalar_u8;
    f32_vmax_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vmaxc_ukernel__scalar_u8;
    f32_vmax_config.minmax.element_tile = 8;
  #elif XNN_ARCH_PPC64
    f32_vmax_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vmax_ukernel__scalar_u8;
    f32_vmax_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vmaxc_ukernel__scalar_u8;
    f32_vmax_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vmaxc_ukernel__scalar_u8;
    f32_vmax_config.minmax.element_tile = 8;
  #else
    #error "Unsupported architecture"
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
      f32_vmin_config.init.f32_default = xnn_init_f32_default_avx_params;
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
  #elif XNN_ARCH_RISCV
    f32_vmin_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vmin_ukernel__scalar_u8;
    f32_vmin_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vminc_ukernel__scalar_u8;
    f32_vmin_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vminc_ukernel__scalar_u8;
    f32_vmin_config.minmax.element_tile = 8;
  #elif XNN_ARCH_PPC64
    f32_vmin_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vmin_ukernel__scalar_u8;
    f32_vmin_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vminc_ukernel__scalar_u8;
    f32_vmin_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vminc_ukernel__scalar_u8;
    f32_vmin_config.minmax.element_tile = 8;
  #else
    #error "Unsupported architecture"
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
  #elif XNN_ARCH_RISCV
    f32_vmul_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vmul_minmax_ukernel__scalar_u8;
    f32_vmul_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vmulc_minmax_ukernel__scalar_u8;
    f32_vmul_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vmulc_minmax_ukernel__scalar_u8;
    f32_vmul_config.init.f32_minmax = xnn_init_f32_minmax_scalar_params;
    f32_vmul_config.minmax.element_tile = 8;
  #elif XNN_ARCH_PPC64
    f32_vmul_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vmul_minmax_ukernel__scalar_u8;
    f32_vmul_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vmulc_minmax_ukernel__scalar_u8;
    f32_vmul_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vmulc_minmax_ukernel__scalar_u8;
    f32_vmul_config.init.f32_minmax = xnn_init_f32_minmax_scalar_params;
    f32_vmul_config.minmax.element_tile = 8;
  #else
    #error "Unsupported architecture"
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
  #elif XNN_ARCH_RISCV
    f32_vsub_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vsub_minmax_ukernel__scalar_u8;
    f32_vsub_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vsubc_minmax_ukernel__scalar_u8;
    f32_vsub_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vrsubc_minmax_ukernel__scalar_u8;
    f32_vsub_config.init.f32_minmax = xnn_init_f32_minmax_scalar_params;
    f32_vsub_config.minmax.element_tile = 8;
  #elif XNN_ARCH_PPC64
    f32_vsub_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vsub_minmax_ukernel__scalar_u8;
    f32_vsub_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vsubc_minmax_ukernel__scalar_u8;
    f32_vsub_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vrsubc_minmax_ukernel__scalar_u8;
    f32_vsub_config.init.f32_minmax = xnn_init_f32_minmax_scalar_params;
    f32_vsub_config.minmax.element_tile = 8;
  #else
    #error "Unsupported architecture"
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
      f32_vsqrdiff_config.init.f32_default = xnn_init_f32_default_avx_params;
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
  #elif XNN_ARCH_WASM
    f32_vsqrdiff_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vsqrdiff_ukernel__scalar_u8;
    f32_vsqrdiff_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vsqrdiffc_ukernel__scalar_u8;
    f32_vsqrdiff_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vsqrdiffc_ukernel__scalar_u8;
    f32_vsqrdiff_config.minmax.element_tile = 8;
  #elif XNN_ARCH_RISCV
    f32_vsqrdiff_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vsqrdiff_ukernel__scalar_u8;
    f32_vsqrdiff_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vsqrdiffc_ukernel__scalar_u8;
    f32_vsqrdiff_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vsqrdiffc_ukernel__scalar_u8;
    f32_vsqrdiff_config.minmax.element_tile = 8;
  #elif XNN_ARCH_PPC64
    f32_vsqrdiff_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vsqrdiff_ukernel__scalar_u8;
    f32_vsqrdiff_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vsqrdiffc_ukernel__scalar_u8;
    f32_vsqrdiff_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_f32_vsqrdiffc_ukernel__scalar_u8;
    f32_vsqrdiff_config.minmax.element_tile = 8;
  #else
    #error "Unsupported architecture"
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
      qs8_vadd_config.init.qs8_add = xnn_init_qs8_add_minmax_neon_params;
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
    qs8_vadd_config.init.qs8_add = xnn_init_qs8_add_minmax_neon_params;
    qs8_vadd_config.minmax.element_tile = 32;
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (!XNN_PLATFORM_MOBILE && hardware_config->use_x86_avx512skx) {
      qs8_vadd_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_qs8_vadd_minmax_ukernel__avx512skx_mul32_ld128_u16;
      qs8_vadd_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qs8_vaddc_minmax_ukernel__avx512skx_mul32_ld128_u16;
      qs8_vadd_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qs8_vaddc_minmax_ukernel__avx512skx_mul32_ld128_u16;
      qs8_vadd_config.init.qs8_add = xnn_init_qs8_add_minmax_avx512_params;
      qs8_vadd_config.minmax.element_tile = 16;
    } else if (hardware_config->use_x86_xop) {
      qs8_vadd_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_qs8_vadd_minmax_ukernel__xop_mul32_ld32_u8;
      qs8_vadd_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qs8_vaddc_minmax_ukernel__xop_mul32_ld32_u8;
      qs8_vadd_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qs8_vaddc_minmax_ukernel__xop_mul32_ld32_u8;
      qs8_vadd_config.init.qs8_add = xnn_init_qs8_add_minmax_sse4_mul32_params;
      qs8_vadd_config.minmax.element_tile = 8;
    } else if (hardware_config->use_x86_avx2) {
      qs8_vadd_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_qs8_vadd_minmax_ukernel__avx2_mul32_ld64_u16;
      qs8_vadd_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qs8_vaddc_minmax_ukernel__avx2_mul32_ld64_u16;
      qs8_vadd_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qs8_vaddc_minmax_ukernel__avx2_mul32_ld64_u16;
      qs8_vadd_config.init.qs8_add = xnn_init_qs8_add_minmax_avx2_params;
      qs8_vadd_config.minmax.element_tile = 16;
    } else if (hardware_config->use_x86_avx) {
      qs8_vadd_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_qs8_vadd_minmax_ukernel__avx_mul32_ld32_u8;
      qs8_vadd_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qs8_vaddc_minmax_ukernel__avx_mul32_ld32_u8;
      qs8_vadd_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qs8_vaddc_minmax_ukernel__avx_mul32_ld32_u8;
      qs8_vadd_config.init.qs8_add = xnn_init_qs8_add_minmax_sse4_mul32_params;
      qs8_vadd_config.minmax.element_tile = 8;
    } else if (hardware_config->use_x86_sse4_1) {
      qs8_vadd_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_qs8_vadd_minmax_ukernel__sse41_mul16_ld64_u8;
      qs8_vadd_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qs8_vaddc_minmax_ukernel__sse41_mul16_ld64_u8;
      qs8_vadd_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qs8_vaddc_minmax_ukernel__sse41_mul16_ld64_u8;
      qs8_vadd_config.init.qs8_add = xnn_init_qs8_add_minmax_sse4_mul16_params;
      qs8_vadd_config.minmax.element_tile = 8;
    } else {
      qs8_vadd_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_qs8_vadd_minmax_ukernel__sse2_mul16_ld64_u8;
      qs8_vadd_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qs8_vaddc_minmax_ukernel__sse2_mul16_ld64_u8;
      qs8_vadd_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qs8_vaddc_minmax_ukernel__sse2_mul16_ld64_u8;
      qs8_vadd_config.init.qs8_add = xnn_init_qs8_add_minmax_sse2_params;
      qs8_vadd_config.minmax.element_tile = 8;
    }
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    qs8_vadd_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_qs8_vadd_minmax_ukernel__wasmsimd_u32;
    qs8_vadd_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qs8_vaddc_minmax_ukernel__wasmsimd_u32;
    qs8_vadd_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qs8_vaddc_minmax_ukernel__wasmsimd_u32;
    qs8_vadd_config.init.qs8_add = xnn_init_qs8_add_minmax_wasmsimd_params;
    qs8_vadd_config.minmax.element_tile = 32;
  #elif XNN_ARCH_WASM
    qs8_vadd_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_qs8_vadd_minmax_ukernel__scalar_u4;
    qs8_vadd_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qs8_vaddc_minmax_ukernel__scalar_u4;
    qs8_vadd_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qs8_vaddc_minmax_ukernel__scalar_u4;
    qs8_vadd_config.init.qs8_add = xnn_init_qs8_add_minmax_scalar_params;
    qs8_vadd_config.minmax.element_tile = 4;
  #elif XNN_ARCH_RISCV
    qs8_vadd_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_qs8_vadd_minmax_ukernel__scalar_u4;
    qs8_vadd_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qs8_vaddc_minmax_ukernel__scalar_u4;
    qs8_vadd_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qs8_vaddc_minmax_ukernel__scalar_u4;
    qs8_vadd_config.init.qs8_add = xnn_init_qs8_add_minmax_scalar_params;
    qs8_vadd_config.minmax.element_tile = 4;
  #elif XNN_ARCH_PPC64
    qs8_vadd_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_qs8_vadd_minmax_ukernel__scalar_u4;
    qs8_vadd_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qs8_vaddc_minmax_ukernel__scalar_u4;
    qs8_vadd_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qs8_vaddc_minmax_ukernel__scalar_u4;
    qs8_vadd_config.init.qs8_add = xnn_init_qs8_add_minmax_scalar_params;
    qs8_vadd_config.minmax.element_tile = 4;
  #else
    #error "Unsupported architecture"
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
      qs8_vmul_config.init.qs8_mul = xnn_init_qs8_mul_minmax_fp32_scalar_params;
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
      qs8_vmul_config.init.qs8_mul = xnn_init_qs8_mul_minmax_fp32_sse4_params;
      qs8_vmul_config.minmax.element_tile = 16;
    } else if (hardware_config->use_x86_sse4_1) {
      qs8_vmul_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_qs8_vmul_minmax_fp32_ukernel__sse41_mul16_ld64_u16;
      qs8_vmul_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qs8_vmulc_minmax_fp32_ukernel__sse41_mul16_ld64_u16;
      qs8_vmul_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qs8_vmulc_minmax_fp32_ukernel__sse41_mul16_ld64_u16;
      qs8_vmul_config.init.qs8_mul = xnn_init_qs8_mul_minmax_fp32_sse4_params;
      qs8_vmul_config.minmax.element_tile = 16;
    } else {
      qs8_vmul_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_qs8_vmul_minmax_fp32_ukernel__sse2_mul16_ld64_u8;
      qs8_vmul_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qs8_vmulc_minmax_fp32_ukernel__sse2_mul16_ld64_u8;
      qs8_vmul_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qs8_vmulc_minmax_fp32_ukernel__sse2_mul16_ld64_u8;
      qs8_vmul_config.init.qs8_mul = xnn_init_qs8_mul_minmax_fp32_sse2_params;
      qs8_vmul_config.minmax.element_tile = 8;
    }
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    qs8_vmul_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_qs8_vmul_minmax_fp32_ukernel__wasmsimd_mul32_ld64_u8;
    qs8_vmul_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qs8_vmulc_minmax_fp32_ukernel__wasmsimd_mul32_ld64_u8;
    qs8_vmul_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qs8_vmulc_minmax_fp32_ukernel__wasmsimd_mul32_ld64_u8;
    qs8_vmul_config.init.qs8_mul = xnn_init_qs8_mul_minmax_fp32_wasmsimd_params;
    qs8_vmul_config.minmax.element_tile = 8;
  #elif XNN_ARCH_WASM
    qs8_vmul_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_qs8_vmul_minmax_fp32_ukernel__scalar_u4;
    qs8_vmul_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qs8_vmulc_minmax_fp32_ukernel__scalar_u4;
    qs8_vmul_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qs8_vmulc_minmax_fp32_ukernel__scalar_u4;
    qs8_vmul_config.init.qs8_mul = xnn_init_qs8_mul_minmax_fp32_scalar_params;
    qs8_vmul_config.minmax.element_tile = 4;
  #elif XNN_ARCH_RISCV
    #if XNN_ENABLE_RISCV_VECTOR
      qs8_vmul_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_qs8_vmul_minmax_fp32_ukernel__rvv_u2v;
      qs8_vmul_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qs8_vmulc_minmax_fp32_ukernel__rvv_u2v;
      qs8_vmul_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qs8_vmulc_minmax_fp32_ukernel__rvv_u2v;
      qs8_vmul_config.init.qs8_mul = xnn_init_qs8_mul_minmax_fp32_scalar_params;
      qs8_vmul_config.minmax.element_tile = 2;
    #else
      qs8_vmul_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_qs8_vmul_minmax_fp32_ukernel__scalar_u4;
      qs8_vmul_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qs8_vmulc_minmax_fp32_ukernel__scalar_u4;
      qs8_vmul_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qs8_vmulc_minmax_fp32_ukernel__scalar_u4;
      qs8_vmul_config.init.qs8_mul = xnn_init_qs8_mul_minmax_fp32_scalar_params;
      qs8_vmul_config.minmax.element_tile = 4;
    #endif
  #elif XNN_ARCH_PPC64
    qs8_vmul_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_qs8_vmul_minmax_fp32_ukernel__scalar_u4;
    qs8_vmul_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qs8_vmulc_minmax_fp32_ukernel__scalar_u4;
    qs8_vmul_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qs8_vmulc_minmax_fp32_ukernel__scalar_u4;
    qs8_vmul_config.init.qs8_mul = xnn_init_qs8_mul_minmax_fp32_scalar_params;
    qs8_vmul_config.minmax.element_tile = 4;
  #else
    #error "Unsupported architecture"
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
      qu8_vadd_config.init.qu8_add = xnn_init_qu8_add_minmax_neon_params;
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
    qu8_vadd_config.init.qu8_add = xnn_init_qu8_add_minmax_neon_params;
    qu8_vadd_config.minmax.element_tile = 8;
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (!XNN_PLATFORM_MOBILE && hardware_config->use_x86_avx512skx) {
      qu8_vadd_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_qu8_vadd_minmax_ukernel__avx512skx_mul32_ld128_u16;
      qu8_vadd_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qu8_vaddc_minmax_ukernel__avx512skx_mul32_ld128_u16;
      qu8_vadd_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qu8_vaddc_minmax_ukernel__avx512skx_mul32_ld128_u16;
      qu8_vadd_config.init.qu8_add = xnn_init_qu8_add_minmax_avx512_params;
      qu8_vadd_config.minmax.element_tile = 16;
    } else if (hardware_config->use_x86_xop) {
      qu8_vadd_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_qu8_vadd_minmax_ukernel__xop_mul32_ld32_u8;
      qu8_vadd_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qu8_vaddc_minmax_ukernel__xop_mul32_ld32_u8;
      qu8_vadd_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qu8_vaddc_minmax_ukernel__xop_mul32_ld32_u8;
      qu8_vadd_config.init.qu8_add = xnn_init_qu8_add_minmax_sse4_params;
      qu8_vadd_config.minmax.element_tile = 8;
    } else if (hardware_config->use_x86_avx2) {
      qu8_vadd_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_qu8_vadd_minmax_ukernel__avx2_mul32_ld64_u16;
      qu8_vadd_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qu8_vaddc_minmax_ukernel__avx2_mul32_ld64_u16;
      qu8_vadd_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qu8_vaddc_minmax_ukernel__avx2_mul32_ld64_u16;
      qu8_vadd_config.init.qu8_add = xnn_init_qu8_add_minmax_avx2_params;
      qu8_vadd_config.minmax.element_tile = 16;
    } else if (hardware_config->use_x86_avx) {
      qu8_vadd_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_qu8_vadd_minmax_ukernel__avx_mul32_ld32_u8;
      qu8_vadd_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qu8_vaddc_minmax_ukernel__avx_mul32_ld32_u8;
      qu8_vadd_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qu8_vaddc_minmax_ukernel__avx_mul32_ld32_u8;
      qu8_vadd_config.init.qu8_add = xnn_init_qu8_add_minmax_sse4_params;
      qu8_vadd_config.minmax.element_tile = 8;
    } else if (hardware_config->use_x86_sse4_1) {
      qu8_vadd_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_qu8_vadd_minmax_ukernel__sse41_mul16_ld64_u8;
      qu8_vadd_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qu8_vaddc_minmax_ukernel__sse41_mul16_ld64_u8;
      qu8_vadd_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qu8_vaddc_minmax_ukernel__sse41_mul16_ld64_u8;
      qu8_vadd_config.init.qu8_add = xnn_init_qu8_add_minmax_sse2_params;
      qu8_vadd_config.minmax.element_tile = 8;
    } else {
      qu8_vadd_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_qu8_vadd_minmax_ukernel__sse2_mul16_ld64_u8;
      qu8_vadd_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qu8_vaddc_minmax_ukernel__sse2_mul16_ld64_u8;
      qu8_vadd_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qu8_vaddc_minmax_ukernel__sse2_mul16_ld64_u8;
      qu8_vadd_config.init.qu8_add = xnn_init_qu8_add_minmax_sse2_params;
      qu8_vadd_config.minmax.element_tile = 8;
    }
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    qu8_vadd_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_qu8_vadd_minmax_ukernel__wasmsimd_u32;
    qu8_vadd_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qu8_vaddc_minmax_ukernel__wasmsimd_u32;
    qu8_vadd_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qu8_vaddc_minmax_ukernel__wasmsimd_u32;
    qu8_vadd_config.init.qu8_add = xnn_init_qu8_add_minmax_wasmsimd_params;
    qu8_vadd_config.minmax.element_tile = 32;
  #elif XNN_ARCH_WASM
    qu8_vadd_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_qu8_vadd_minmax_ukernel__scalar_u4;
    qu8_vadd_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qu8_vaddc_minmax_ukernel__scalar_u4;
    qu8_vadd_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qu8_vaddc_minmax_ukernel__scalar_u4;
    qu8_vadd_config.init.qu8_add = xnn_init_qu8_add_minmax_scalar_params;
    qu8_vadd_config.minmax.element_tile = 4;
  #elif XNN_ARCH_RISCV
    qu8_vadd_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_qu8_vadd_minmax_ukernel__scalar_u4;
    qu8_vadd_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qu8_vaddc_minmax_ukernel__scalar_u4;
    qu8_vadd_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qu8_vaddc_minmax_ukernel__scalar_u4;
    qu8_vadd_config.init.qu8_add = xnn_init_qu8_add_minmax_scalar_params;
    qu8_vadd_config.minmax.element_tile = 4;
  #elif XNN_ARCH_PPC64
    qu8_vadd_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_qu8_vadd_minmax_ukernel__scalar_u4;
    qu8_vadd_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qu8_vaddc_minmax_ukernel__scalar_u4;
    qu8_vadd_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qu8_vaddc_minmax_ukernel__scalar_u4;
    qu8_vadd_config.init.qu8_add = xnn_init_qu8_add_minmax_scalar_params;
    qu8_vadd_config.minmax.element_tile = 4;
  #else
    #error "Unsupported architecture"
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
      qu8_vmul_config.init.qu8_mul = xnn_init_qu8_mul_minmax_fp32_scalar_params;
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
      qu8_vmul_config.init.qu8_mul = xnn_init_qu8_mul_minmax_fp32_sse2_params;
      qu8_vmul_config.minmax.element_tile = 16;
    } else if (hardware_config->use_x86_sse4_1) {
      qu8_vmul_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_qu8_vmul_minmax_fp32_ukernel__sse41_mul16_ld64_u16;
      qu8_vmul_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qu8_vmulc_minmax_fp32_ukernel__sse41_mul16_ld64_u16;
      qu8_vmul_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qu8_vmulc_minmax_fp32_ukernel__sse41_mul16_ld64_u16;
      qu8_vmul_config.init.qu8_mul = xnn_init_qu8_mul_minmax_fp32_sse2_params;
      qu8_vmul_config.minmax.element_tile = 16;
    } else {
      qu8_vmul_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_qu8_vmul_minmax_fp32_ukernel__sse2_mul16_ld64_u8;
      qu8_vmul_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qu8_vmulc_minmax_fp32_ukernel__sse2_mul16_ld64_u8;
      qu8_vmul_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qu8_vmulc_minmax_fp32_ukernel__sse2_mul16_ld64_u8;
      qu8_vmul_config.init.qu8_mul = xnn_init_qu8_mul_minmax_fp32_sse2_params;
      qu8_vmul_config.minmax.element_tile = 8;
    }
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    qu8_vmul_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_qu8_vmul_minmax_fp32_ukernel__wasmsimd_mul32_ld64_u8;
    qu8_vmul_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qu8_vmulc_minmax_fp32_ukernel__wasmsimd_mul32_ld64_u8;
    qu8_vmul_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qu8_vmulc_minmax_fp32_ukernel__wasmsimd_mul32_ld64_u8;
    qu8_vmul_config.init.qu8_mul = xnn_init_qu8_mul_minmax_fp32_wasmsimd_params;
    qu8_vmul_config.minmax.element_tile = 8;
  #elif XNN_ARCH_WASM
    qu8_vmul_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_qu8_vmul_minmax_fp32_ukernel__scalar_u4;
    qu8_vmul_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qu8_vmulc_minmax_fp32_ukernel__scalar_u4;
    qu8_vmul_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qu8_vmulc_minmax_fp32_ukernel__scalar_u4;
    qu8_vmul_config.init.qu8_mul = xnn_init_qu8_mul_minmax_fp32_scalar_params;
    qu8_vmul_config.minmax.element_tile = 4;
  #elif XNN_ARCH_RISCV
    #if XNN_ENABLE_RISCV_VECTOR
      qu8_vmul_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_qu8_vmul_minmax_fp32_ukernel__rvv_u2v;
      qu8_vmul_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qu8_vmulc_minmax_fp32_ukernel__rvv_u2v;
      qu8_vmul_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qu8_vmulc_minmax_fp32_ukernel__rvv_u2v;
      qu8_vmul_config.init.qu8_mul = xnn_init_qu8_mul_minmax_fp32_scalar_params;
      qu8_vmul_config.minmax.element_tile = 2;
    #else
      qu8_vmul_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_qu8_vmul_minmax_fp32_ukernel__scalar_u4;
      qu8_vmul_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qu8_vmulc_minmax_fp32_ukernel__scalar_u4;
      qu8_vmul_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qu8_vmulc_minmax_fp32_ukernel__scalar_u4;
      qu8_vmul_config.init.qu8_mul = xnn_init_qu8_mul_minmax_fp32_scalar_params;
      qu8_vmul_config.minmax.element_tile = 4;
    #endif
  #elif XNN_ARCH_PPC64
    qu8_vmul_config.minmax.op_ukernel = (xnn_vbinary_ukernel_fn) xnn_qu8_vmul_minmax_fp32_ukernel__scalar_u4;
    qu8_vmul_config.minmax.opc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qu8_vmulc_minmax_fp32_ukernel__scalar_u4;
    qu8_vmul_config.minmax.ropc_ukernel = (xnn_vbinary_ukernel_fn) xnn_qu8_vmulc_minmax_fp32_ukernel__scalar_u4;
    qu8_vmul_config.init.qu8_mul = xnn_init_qu8_mul_minmax_fp32_scalar_params;
    qu8_vmul_config.minmax.element_tile = 4;
  #else
    #error "Unsupported architecture"
  #endif
}

#if XNN_PLATFORM_WINDOWS
  static BOOL CALLBACK init_f16_vadd_config_windows(PINIT_ONCE init_once, PVOID parameter, PVOID* context) {
    init_f16_vadd_config();
    return TRUE;
  }

  static BOOL CALLBACK init_f16_vdiv_config_windows(PINIT_ONCE init_once, PVOID parameter, PVOID* context) {
    init_f16_vdiv_config();
    return TRUE;
  }

  static BOOL CALLBACK init_f16_vmax_config_windows(PINIT_ONCE init_once, PVOID parameter, PVOID* context) {
    init_f16_vmax_config();
    return TRUE;
  }

  static BOOL CALLBACK init_f16_vmin_config_windows(PINIT_ONCE init_once, PVOID parameter, PVOID* context) {
    init_f16_vmin_config();
    return TRUE;
  }

  static BOOL CALLBACK init_f16_vmul_config_windows(PINIT_ONCE init_once, PVOID parameter, PVOID* context) {
    init_f16_vmul_config();
    return TRUE;
  }

  static BOOL CALLBACK init_f16_vsub_config_windows(PINIT_ONCE init_once, PVOID parameter, PVOID* context) {
    init_f16_vsub_config();
    return TRUE;
  }

  static BOOL CALLBACK init_f16_vsqrdiff_config_windows(PINIT_ONCE init_once, PVOID parameter, PVOID* context) {
    init_f16_vsqrdiff_config();
    return TRUE;
  }

  static BOOL CALLBACK init_f32_vadd_config_windows(PINIT_ONCE init_once, PVOID parameter, PVOID* context) {
    init_f32_vadd_config();
    return TRUE;
  }

  static BOOL CALLBACK init_f32_vdiv_config_windows(PINIT_ONCE init_once, PVOID parameter, PVOID* context) {
    init_f32_vdiv_config();
    return TRUE;
  }

  static BOOL CALLBACK init_f32_vmax_config_windows(PINIT_ONCE init_once, PVOID parameter, PVOID* context) {
    init_f32_vmax_config();
    return TRUE;
  }

  static BOOL CALLBACK init_f32_vmin_config_windows(PINIT_ONCE init_once, PVOID parameter, PVOID* context) {
    init_f32_vmin_config();
    return TRUE;
  }

  static BOOL CALLBACK init_f32_vmul_config_windows(PINIT_ONCE init_once, PVOID parameter, PVOID* context) {
    init_f32_vmul_config();
    return TRUE;
  }

  static BOOL CALLBACK init_f32_vsub_config_windows(PINIT_ONCE init_once, PVOID parameter, PVOID* context) {
    init_f32_vsub_config();
    return TRUE;
  }

  static BOOL CALLBACK init_f32_vsqrdiff_config_windows(PINIT_ONCE init_once, PVOID parameter, PVOID* context) {
    init_f32_vsqrdiff_config();
    return TRUE;
  }

  static BOOL CALLBACK init_qs8_vadd_config_windows(PINIT_ONCE init_once, PVOID parameter, PVOID* context) {
    init_qs8_vadd_config();
    return TRUE;
  }

  static BOOL CALLBACK init_qs8_vmul_config_windows(PINIT_ONCE init_once, PVOID parameter, PVOID* context) {
    init_qs8_vmul_config();
    return TRUE;
  }

  static BOOL CALLBACK init_qu8_vadd_config_windows(PINIT_ONCE init_once, PVOID parameter, PVOID* context) {
    init_qu8_vadd_config();
    return TRUE;
  }

  static BOOL CALLBACK init_qu8_vmul_config_windows(PINIT_ONCE init_once, PVOID parameter, PVOID* context) {
    init_qu8_vmul_config();
    return TRUE;
  }
#endif

const struct xnn_binary_elementwise_config* xnn_init_f16_vadd_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL || !xnn_is_f16_compatible_config(hardware_config)) {
    return NULL;
  }
  #if XNN_PLATFORM_WINDOWS
    InitOnceExecuteOnce(&init_guard_f16_vadd, &init_f16_vadd_config_windows, NULL, NULL);
  #else
    pthread_once(&init_guard_f16_vadd, &init_f16_vadd_config);
  #endif
  return &f16_vadd_config;
}

const struct xnn_binary_elementwise_config* xnn_init_f16_vdiv_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL || !xnn_is_f16_compatible_config(hardware_config)) {
    return NULL;
  }
  #if XNN_PLATFORM_WINDOWS
    InitOnceExecuteOnce(&init_guard_f16_vdiv, &init_f16_vdiv_config_windows, NULL, NULL);
  #else
    pthread_once(&init_guard_f16_vdiv, &init_f16_vdiv_config);
  #endif
  return &f16_vdiv_config;
}

const struct xnn_binary_elementwise_config* xnn_init_f16_vmax_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL || !xnn_is_f16_compatible_config(hardware_config)) {
    return NULL;
  }
  #if XNN_PLATFORM_WINDOWS
    InitOnceExecuteOnce(&init_guard_f16_vmax, &init_f16_vmax_config_windows, NULL, NULL);
  #else
    pthread_once(&init_guard_f16_vmax, &init_f16_vmax_config);
  #endif
  return &f16_vmax_config;
}

const struct xnn_binary_elementwise_config* xnn_init_f16_vmin_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL || !xnn_is_f16_compatible_config(hardware_config)) {
    return NULL;
  }
  #if XNN_PLATFORM_WINDOWS
    InitOnceExecuteOnce(&init_guard_f16_vmin, &init_f16_vmin_config_windows, NULL, NULL);
  #else
    pthread_once(&init_guard_f16_vmin, &init_f16_vmin_config);
  #endif
  return &f16_vmin_config;
}

const struct xnn_binary_elementwise_config* xnn_init_f16_vmul_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL || !xnn_is_f16_compatible_config(hardware_config)) {
    return NULL;
  }
  #if XNN_PLATFORM_WINDOWS
    InitOnceExecuteOnce(&init_guard_f16_vmul, &init_f16_vmul_config_windows, NULL, NULL);
  #else
    pthread_once(&init_guard_f16_vmul, &init_f16_vmul_config);
  #endif
  return &f16_vmul_config;
}

const struct xnn_binary_elementwise_config* xnn_init_f16_vsub_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL || !xnn_is_f16_compatible_config(hardware_config)) {
    return NULL;
  }
  #if XNN_PLATFORM_WINDOWS
    InitOnceExecuteOnce(&init_guard_f16_vsub, &init_f16_vsub_config_windows, NULL, NULL);
  #else
    pthread_once(&init_guard_f16_vsub, &init_f16_vsub_config);
  #endif
  return &f16_vsub_config;
}

const struct xnn_binary_elementwise_config* xnn_init_f16_vsqrdiff_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL || !xnn_is_f16_compatible_config(hardware_config)) {
    return NULL;
  }
  #if XNN_PLATFORM_WINDOWS
    InitOnceExecuteOnce(&init_guard_f16_vsqrdiff, &init_f16_vsqrdiff_config_windows, NULL, NULL);
  #else
    pthread_once(&init_guard_f16_vsqrdiff, &init_f16_vsqrdiff_config);
  #endif
  return &f16_vsqrdiff_config;
}

const struct xnn_binary_elementwise_config* xnn_init_f32_vadd_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  #if XNN_PLATFORM_WINDOWS
    InitOnceExecuteOnce(&init_guard_f32_vadd, &init_f32_vadd_config_windows, NULL, NULL);
  #else
    pthread_once(&init_guard_f32_vadd, &init_f32_vadd_config);
  #endif
  return &f32_vadd_config;
}

const struct xnn_binary_elementwise_config* xnn_init_f32_vdiv_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  #if XNN_PLATFORM_WINDOWS
    InitOnceExecuteOnce(&init_guard_f32_vdiv, &init_f32_vdiv_config_windows, NULL, NULL);
  #else
    pthread_once(&init_guard_f32_vdiv, &init_f32_vdiv_config);
  #endif
  return &f32_vdiv_config;
}

const struct xnn_binary_elementwise_config* xnn_init_f32_vmax_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  #if XNN_PLATFORM_WINDOWS
    InitOnceExecuteOnce(&init_guard_f32_vmax, &init_f32_vmax_config_windows, NULL, NULL);
  #else
    pthread_once(&init_guard_f32_vmax, &init_f32_vmax_config);
  #endif
  return &f32_vmax_config;
}

const struct xnn_binary_elementwise_config* xnn_init_f32_vmin_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  #if XNN_PLATFORM_WINDOWS
    InitOnceExecuteOnce(&init_guard_f32_vmin, &init_f32_vmin_config_windows, NULL, NULL);
  #else
    pthread_once(&init_guard_f32_vmin, &init_f32_vmin_config);
  #endif
  return &f32_vmin_config;
}

const struct xnn_binary_elementwise_config* xnn_init_f32_vmul_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  #if XNN_PLATFORM_WINDOWS
    InitOnceExecuteOnce(&init_guard_f32_vmul, &init_f32_vmul_config_windows, NULL, NULL);
  #else
    pthread_once(&init_guard_f32_vmul, &init_f32_vmul_config);
  #endif
  return &f32_vmul_config;
}

const struct xnn_binary_elementwise_config* xnn_init_f32_vsub_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  #if XNN_PLATFORM_WINDOWS
    InitOnceExecuteOnce(&init_guard_f32_vsub, &init_f32_vsub_config_windows, NULL, NULL);
  #else
    pthread_once(&init_guard_f32_vsub, &init_f32_vsub_config);
  #endif
  return &f32_vsub_config;
}

const struct xnn_binary_elementwise_config* xnn_init_f32_vsqrdiff_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  #if XNN_PLATFORM_WINDOWS
    InitOnceExecuteOnce(&init_guard_f32_vsqrdiff, &init_f32_vsqrdiff_config_windows, NULL, NULL);
  #else
    pthread_once(&init_guard_f32_vsqrdiff, &init_f32_vsqrdiff_config);
  #endif
  return &f32_vsqrdiff_config;
}

const struct xnn_binary_elementwise_config* xnn_init_qs8_vadd_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  #if XNN_PLATFORM_WINDOWS
    InitOnceExecuteOnce(&init_guard_qs8_vadd, &init_qs8_vadd_config_windows, NULL, NULL);
  #else
    pthread_once(&init_guard_qs8_vadd, &init_qs8_vadd_config);
  #endif
  return &qs8_vadd_config;
}

const struct xnn_binary_elementwise_config* xnn_init_qs8_vmul_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  #if XNN_PLATFORM_WINDOWS
    InitOnceExecuteOnce(&init_guard_qs8_vmul, &init_qs8_vmul_config_windows, NULL, NULL);
  #else
    pthread_once(&init_guard_qs8_vmul, &init_qs8_vmul_config);
  #endif
  return &qs8_vmul_config;
}

const struct xnn_binary_elementwise_config* xnn_init_qu8_vadd_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  #if XNN_PLATFORM_WINDOWS
    InitOnceExecuteOnce(&init_guard_qu8_vadd, &init_qu8_vadd_config_windows, NULL, NULL);
  #else
    pthread_once(&init_guard_qu8_vadd, &init_qu8_vadd_config);
  #endif
  return &qu8_vadd_config;
}

const struct xnn_binary_elementwise_config* xnn_init_qu8_vmul_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  #if XNN_PLATFORM_WINDOWS
    InitOnceExecuteOnce(&init_guard_qu8_vmul, &init_qu8_vmul_config_windows, NULL, NULL);
  #else
    pthread_once(&init_guard_qu8_vmul, &init_qu8_vmul_config);
  #endif
  return &qu8_vmul_config;
}
