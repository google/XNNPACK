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
#include <xnnpack/microfnptr.h>
#include <xnnpack/microparams-init.h>
#include <xnnpack/vcvt.h>
#include <xnnpack/vlrelu.h>
#include <xnnpack/vunary.h>


static struct xnn_unary_elementwise_config f16_abs_config = {0};
static struct xnn_unary_elementwise_config f16_clamp_config = {0};
static struct xnn_unary_elementwise_config f16_elu_config = {0};
static struct xnn_unary_elementwise_config f16_hswish_config = {0};
static struct xnn_unary_elementwise_config f16_lrelu_config = {0};
static struct xnn_unary_elementwise_config f16_neg_config = {0};
static struct xnn_unary_elementwise_config f16_rndd_config = {0};
static struct xnn_unary_elementwise_config f16_rndne_config = {0};
static struct xnn_unary_elementwise_config f16_rndu_config = {0};
static struct xnn_unary_elementwise_config f16_rndz_config = {0};
static struct xnn_unary_elementwise_config f16_sigmoid_config = {0};
static struct xnn_unary_elementwise_config f16_sqr_config = {0};
static struct xnn_unary_elementwise_config f16_sqrt_config = {0};
static struct xnn_unary_elementwise_config f16_tanh_config = {0};
static struct xnn_unary_elementwise_config f16_to_f32_cvt_config = {0};
static struct xnn_unary_elementwise_config f16_to_qs8_cvt_config = {0};
static struct xnn_unary_elementwise_config f32_abs_config = {0};
static struct xnn_unary_elementwise_config f32_clamp_config = {0};
static struct xnn_unary_elementwise_config f32_elu_config = {0};
static struct xnn_unary_elementwise_config f32_hswish_config = {0};
static struct xnn_unary_elementwise_config f32_lrelu_config = {0};
static struct xnn_unary_elementwise_config f32_neg_config = {0};
static struct xnn_unary_elementwise_config f32_relu_config = {0};
static struct xnn_unary_elementwise_config f32_rndd_config = {0};
static struct xnn_unary_elementwise_config f32_rndne_config = {0};
static struct xnn_unary_elementwise_config f32_rndu_config = {0};
static struct xnn_unary_elementwise_config f32_rndz_config = {0};
static struct xnn_unary_elementwise_config f32_rsqrt_config = {0};
static struct xnn_unary_elementwise_config f32_sigmoid_config = {0};
static struct xnn_unary_elementwise_config f32_sqr_config = {0};
static struct xnn_unary_elementwise_config f32_sqrt_config = {0};
static struct xnn_unary_elementwise_config f32_tanh_config = {0};
static struct xnn_unary_elementwise_config f32_to_f16_cvt_config = {0};
static struct xnn_unary_elementwise_config f32_to_qs8_cvt_config = {0};
static struct xnn_unary_elementwise_config f32_to_qu8_cvt_config = {0};
static struct xnn_unary_elementwise_config qs8_cvt_config = {0};
static struct xnn_unary_elementwise_config qs8_lrelu_config = {0};
static struct xnn_unary_elementwise_config qs8_to_f16_cvt_config = {0};
static struct xnn_unary_elementwise_config qs8_to_f32_cvt_config = {0};
static struct xnn_unary_elementwise_config qs16_to_qs8_cvt_config = {0};
static struct xnn_unary_elementwise_config qu8_cvt_config = {0};
static struct xnn_unary_elementwise_config qu8_lrelu_config = {0};
static struct xnn_unary_elementwise_config qu8_to_f32_cvt_config = {0};
static struct xnn_unary_elementwise_config s8_clamp_config = {0};
static struct xnn_unary_elementwise_config u8_clamp_config = {0};
static struct xnn_unary_elementwise_config xx_copy_config = {0};


#if XNN_PLATFORM_WINDOWS
  static INIT_ONCE init_guard_f16_abs = INIT_ONCE_STATIC_INIT;
  static INIT_ONCE init_guard_f16_clamp = INIT_ONCE_STATIC_INIT;
  static INIT_ONCE init_guard_f16_elu = INIT_ONCE_STATIC_INIT;
  static INIT_ONCE init_guard_f16_hswish = INIT_ONCE_STATIC_INIT;
  static INIT_ONCE init_guard_f16_lrelu = INIT_ONCE_STATIC_INIT;
  static INIT_ONCE init_guard_f16_neg = INIT_ONCE_STATIC_INIT;
  static INIT_ONCE init_guard_f16_rndd = INIT_ONCE_STATIC_INIT;
  static INIT_ONCE init_guard_f16_rndne = INIT_ONCE_STATIC_INIT;
  static INIT_ONCE init_guard_f16_rndu = INIT_ONCE_STATIC_INIT;
  static INIT_ONCE init_guard_f16_rndz = INIT_ONCE_STATIC_INIT;
  static INIT_ONCE init_guard_f16_sigmoid = INIT_ONCE_STATIC_INIT;
  static INIT_ONCE init_guard_f16_sqr = INIT_ONCE_STATIC_INIT;
  static INIT_ONCE init_guard_f16_sqrt = INIT_ONCE_STATIC_INIT;
  static INIT_ONCE init_guard_f16_tanh = INIT_ONCE_STATIC_INIT;
  static INIT_ONCE init_guard_f16_to_qs8_cvt = INIT_ONCE_STATIC_INIT;
  static INIT_ONCE init_guard_f16_to_f32_cvt = INIT_ONCE_STATIC_INIT;
  static INIT_ONCE init_guard_f32_abs = INIT_ONCE_STATIC_INIT;
  static INIT_ONCE init_guard_f32_clamp = INIT_ONCE_STATIC_INIT;
  static INIT_ONCE init_guard_f32_elu = INIT_ONCE_STATIC_INIT;
  static INIT_ONCE init_guard_f32_hswish = INIT_ONCE_STATIC_INIT;
  static INIT_ONCE init_guard_f32_lrelu = INIT_ONCE_STATIC_INIT;
  static INIT_ONCE init_guard_f32_neg = INIT_ONCE_STATIC_INIT;
  static INIT_ONCE init_guard_f32_relu = INIT_ONCE_STATIC_INIT;
  static INIT_ONCE init_guard_f32_rndd = INIT_ONCE_STATIC_INIT;
  static INIT_ONCE init_guard_f32_rndne = INIT_ONCE_STATIC_INIT;
  static INIT_ONCE init_guard_f32_rndu = INIT_ONCE_STATIC_INIT;
  static INIT_ONCE init_guard_f32_rndz = INIT_ONCE_STATIC_INIT;
  static INIT_ONCE init_guard_f32_rsqrt = INIT_ONCE_STATIC_INIT;
  static INIT_ONCE init_guard_f32_sigmoid = INIT_ONCE_STATIC_INIT;
  static INIT_ONCE init_guard_f32_sqr = INIT_ONCE_STATIC_INIT;
  static INIT_ONCE init_guard_f32_sqrt = INIT_ONCE_STATIC_INIT;
  static INIT_ONCE init_guard_f32_tanh = INIT_ONCE_STATIC_INIT;
  static INIT_ONCE init_guard_f32_to_f16_cvt = INIT_ONCE_STATIC_INIT;
  static INIT_ONCE init_guard_f32_to_qs8_cvt = INIT_ONCE_STATIC_INIT;
  static INIT_ONCE init_guard_f32_to_qu8_cvt = INIT_ONCE_STATIC_INIT;
  static INIT_ONCE init_guard_qs8_cvt = INIT_ONCE_STATIC_INIT;
  static INIT_ONCE init_guard_qs8_lrelu = INIT_ONCE_STATIC_INIT;
  static INIT_ONCE init_guard_qs8_to_f16_cvt = INIT_ONCE_STATIC_INIT;
  static INIT_ONCE init_guard_qs8_to_f32_cvt = INIT_ONCE_STATIC_INIT;
  static INIT_ONCE init_guard_qs16_to_qs8_cvt = INIT_ONCE_STATIC_INIT;
  static INIT_ONCE init_guard_qu8_cvt = INIT_ONCE_STATIC_INIT;
  static INIT_ONCE init_guard_qu8_lrelu = INIT_ONCE_STATIC_INIT;
  static INIT_ONCE init_guard_qu8_to_f32_cvt = INIT_ONCE_STATIC_INIT;
  static INIT_ONCE init_guard_s8_clamp = INIT_ONCE_STATIC_INIT;
  static INIT_ONCE init_guard_u8_clamp = INIT_ONCE_STATIC_INIT;
  static INIT_ONCE init_guard_xx_copy = INIT_ONCE_STATIC_INIT;
#else
  static pthread_once_t init_guard_f16_abs = PTHREAD_ONCE_INIT;
  static pthread_once_t init_guard_f16_clamp = PTHREAD_ONCE_INIT;
  static pthread_once_t init_guard_f16_elu = PTHREAD_ONCE_INIT;
  static pthread_once_t init_guard_f16_hswish = PTHREAD_ONCE_INIT;
  static pthread_once_t init_guard_f16_lrelu = PTHREAD_ONCE_INIT;
  static pthread_once_t init_guard_f16_neg = PTHREAD_ONCE_INIT;
  static pthread_once_t init_guard_f16_rndd = PTHREAD_ONCE_INIT;
  static pthread_once_t init_guard_f16_rndne = PTHREAD_ONCE_INIT;
  static pthread_once_t init_guard_f16_rndu = PTHREAD_ONCE_INIT;
  static pthread_once_t init_guard_f16_rndz = PTHREAD_ONCE_INIT;
  static pthread_once_t init_guard_f16_sigmoid = PTHREAD_ONCE_INIT;
  static pthread_once_t init_guard_f16_sqr = PTHREAD_ONCE_INIT;
  static pthread_once_t init_guard_f16_sqrt = PTHREAD_ONCE_INIT;
  static pthread_once_t init_guard_f16_tanh = PTHREAD_ONCE_INIT;
  static pthread_once_t init_guard_f16_to_f32_cvt = PTHREAD_ONCE_INIT;
  static pthread_once_t init_guard_f16_to_qs8_cvt = PTHREAD_ONCE_INIT;
  static pthread_once_t init_guard_f32_abs = PTHREAD_ONCE_INIT;
  static pthread_once_t init_guard_f32_clamp = PTHREAD_ONCE_INIT;
  static pthread_once_t init_guard_f32_elu = PTHREAD_ONCE_INIT;
  static pthread_once_t init_guard_f32_hswish = PTHREAD_ONCE_INIT;
  static pthread_once_t init_guard_f32_lrelu = PTHREAD_ONCE_INIT;
  static pthread_once_t init_guard_f32_neg = PTHREAD_ONCE_INIT;
  static pthread_once_t init_guard_f32_relu = PTHREAD_ONCE_INIT;
  static pthread_once_t init_guard_f32_rndd = PTHREAD_ONCE_INIT;
  static pthread_once_t init_guard_f32_rndne = PTHREAD_ONCE_INIT;
  static pthread_once_t init_guard_f32_rndu = PTHREAD_ONCE_INIT;
  static pthread_once_t init_guard_f32_rndz = PTHREAD_ONCE_INIT;
  static pthread_once_t init_guard_f32_rsqrt = PTHREAD_ONCE_INIT;
  static pthread_once_t init_guard_f32_sigmoid = PTHREAD_ONCE_INIT;
  static pthread_once_t init_guard_f32_sqr = PTHREAD_ONCE_INIT;
  static pthread_once_t init_guard_f32_sqrt = PTHREAD_ONCE_INIT;
  static pthread_once_t init_guard_f32_tanh = PTHREAD_ONCE_INIT;
  static pthread_once_t init_guard_f32_to_f16_cvt = PTHREAD_ONCE_INIT;
  static pthread_once_t init_guard_f32_to_qs8_cvt = PTHREAD_ONCE_INIT;
  static pthread_once_t init_guard_f32_to_qu8_cvt = PTHREAD_ONCE_INIT;
  static pthread_once_t init_guard_qs8_cvt = PTHREAD_ONCE_INIT;
  static pthread_once_t init_guard_qs16_to_qs8_cvt = PTHREAD_ONCE_INIT;
  static pthread_once_t init_guard_qs8_lrelu = PTHREAD_ONCE_INIT;
  static pthread_once_t init_guard_qs8_to_f16_cvt = PTHREAD_ONCE_INIT;
  static pthread_once_t init_guard_qs8_to_f32_cvt = PTHREAD_ONCE_INIT;
  static pthread_once_t init_guard_qu8_cvt = PTHREAD_ONCE_INIT;
  static pthread_once_t init_guard_qu8_lrelu = PTHREAD_ONCE_INIT;
  static pthread_once_t init_guard_qu8_to_f32_cvt = PTHREAD_ONCE_INIT;
  static pthread_once_t init_guard_s8_clamp = PTHREAD_ONCE_INIT;
  static pthread_once_t init_guard_u8_clamp = PTHREAD_ONCE_INIT;
  static pthread_once_t init_guard_xx_copy = PTHREAD_ONCE_INIT;
#endif


static void init_f16_abs_config(void) {
  #if XNN_ARCH_ARM && XNN_ENABLE_ARM_FP16_VECTOR && XNN_ENABLE_ARM_FP16_SCALAR
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon_fp16_arith) {
      f16_abs_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f16_vabs_ukernel__neonfp16arith_u16;
      f16_abs_config.element_tile = 16;
    }
  #elif XNN_ARCH_ARM64 && XNN_ENABLE_ARM_FP16_VECTOR
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon_fp16_arith) {
      f16_abs_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f16_vabs_ukernel__neonfp16arith_u16;
      f16_abs_config.element_tile = 16;
    }
  #elif (XNN_ARCH_X86 || XNN_ARCH_X86_64) && !XNN_PLATFORM_MOBILE
    f16_abs_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f16_vabs_ukernel__sse2_u16;
    f16_abs_config.init.f16_abs = xnn_init_f16_abs_sse_params;
    f16_abs_config.element_tile = 16;
  #endif
}

static void init_f16_clamp_config(void) {
  #if XNN_ARCH_ARM && XNN_ENABLE_ARM_FP16_VECTOR && XNN_ENABLE_ARM_FP16_SCALAR
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon_fp16_arith) {
      f16_clamp_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f16_vclamp_ukernel__neonfp16arith_u16;
      f16_clamp_config.init.f16_minmax = xnn_init_f16_minmax_fp16arith_params;
      f16_clamp_config.element_tile = 16;
    }
  #elif XNN_ARCH_ARM64 && XNN_ENABLE_ARM_FP16_VECTOR
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon_fp16_arith) {
      f16_clamp_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f16_vclamp_ukernel__neonfp16arith_u16;
      f16_clamp_config.init.f16_minmax = xnn_init_f16_minmax_fp16arith_params;
      f16_clamp_config.element_tile = 16;
    }
  #elif (XNN_ARCH_X86 || XNN_ARCH_X86_64) && !XNN_PLATFORM_MOBILE
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_x86_f16c) {
      f16_clamp_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f16_vclamp_ukernel__f16c_u16;
      f16_clamp_config.init.f16_minmax = xnn_init_f16_minmax_avx_params;
      f16_clamp_config.element_tile = 16;
    }
  #endif
}

static void init_f16_elu_config(void) {
  #if XNN_ARCH_ARM && XNN_ENABLE_ARM_FP16_VECTOR && XNN_ENABLE_ARM_FP16_SCALAR
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon_fp16_arith) {
      f16_elu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f16_velu_ukernel__neonfp16arith_rr1_p3_u16;
      f16_elu_config.init.f16_elu = xnn_init_f16_elu_fp16arith_rr1_p3_params;
      f16_elu_config.element_tile = 16;
    }
  #elif XNN_ARCH_ARM64 && XNN_ENABLE_ARM_FP16_VECTOR
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon_fp16_arith) {
      f16_elu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f16_velu_ukernel__neonfp16arith_rr1_p3_u16;
      f16_elu_config.init.f16_elu = xnn_init_f16_elu_fp16arith_rr1_p3_params;
      f16_elu_config.element_tile = 16;
    }
  #elif (XNN_ARCH_X86 || XNN_ARCH_X86_64) && !XNN_PLATFORM_MOBILE
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_x86_avx2) {
      f16_elu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f16_velu_ukernel__avx2_rr1_p3_u16;
      f16_elu_config.init.f16_elu = xnn_init_f16_elu_avx2_rr1_p3_params;
      f16_elu_config.element_tile = 16;
    }
  #endif
}

static void init_f16_hswish_config(void) {
  #if XNN_ARCH_ARM && XNN_ENABLE_ARM_FP16_VECTOR && XNN_ENABLE_ARM_FP16_SCALAR
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon_fp16_arith) {
      f16_hswish_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f16_vhswish_ukernel__neonfp16arith_u16;
      f16_hswish_config.init.f16_hswish = xnn_init_f16_hswish_fp16arith_params;
      f16_hswish_config.element_tile = 16;
    }
  #elif XNN_ARCH_ARM64 && XNN_ENABLE_ARM_FP16_VECTOR
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon_fp16_arith) {
      f16_hswish_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f16_vhswish_ukernel__neonfp16arith_u16;
      f16_hswish_config.init.f16_hswish = xnn_init_f16_hswish_fp16arith_params;
      f16_hswish_config.element_tile = 16;
    }
  #elif (XNN_ARCH_X86 || XNN_ARCH_X86_64) && !XNN_PLATFORM_MOBILE
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_x86_f16c) {
      f16_hswish_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f16_vhswish_ukernel__f16c_u16;
      f16_hswish_config.init.f16_hswish = xnn_init_f16_hswish_avx_params;
      f16_hswish_config.element_tile = 16;
    }
  #endif
}

static void init_f16_lrelu_config(void) {
  #if XNN_ARCH_ARM && XNN_ENABLE_ARM_FP16_VECTOR && XNN_ENABLE_ARM_FP16_SCALAR
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon_fp16_arith) {
      f16_lrelu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f16_vlrelu_ukernel__neonfp16arith_u16;
      f16_lrelu_config.init.f16_lrelu = xnn_init_f16_lrelu_fp16arith_params;
      f16_lrelu_config.element_tile = 16;
    }
  #elif XNN_ARCH_ARM64 && XNN_ENABLE_ARM_FP16_VECTOR
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon_fp16_arith) {
      f16_lrelu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f16_vlrelu_ukernel__neonfp16arith_u16;
      f16_lrelu_config.init.f16_lrelu = xnn_init_f16_lrelu_fp16arith_params;
      f16_lrelu_config.element_tile = 16;
    }
  #elif (XNN_ARCH_X86 || XNN_ARCH_X86_64) && !XNN_PLATFORM_MOBILE
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_x86_f16c) {
      f16_lrelu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f16_vlrelu_ukernel__f16c_u16;
      f16_lrelu_config.init.f16_lrelu = xnn_init_f16_lrelu_avx_params;
      f16_lrelu_config.element_tile = 16;
    }
  #endif
}

static void init_f16_neg_config(void) {
  #if XNN_ARCH_ARM && XNN_ENABLE_ARM_FP16_VECTOR && XNN_ENABLE_ARM_FP16_SCALAR
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon_fp16_arith) {
      f16_neg_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f16_vneg_ukernel__neonfp16arith_u16;
      f16_neg_config.element_tile = 16;
    }
  #elif XNN_ARCH_ARM64 && XNN_ENABLE_ARM_FP16_VECTOR
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon_fp16_arith) {
      f16_neg_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f16_vneg_ukernel__neonfp16arith_u16;
      f16_neg_config.element_tile = 16;
    }
  #elif (XNN_ARCH_X86 || XNN_ARCH_X86_64) && !XNN_PLATFORM_MOBILE
    f16_neg_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f16_vneg_ukernel__sse2_u16;
    f16_neg_config.init.f16_neg = xnn_init_f16_neg_sse_params;
    f16_neg_config.element_tile = 16;
  #endif
}

static void init_f16_rndd_config(void) {
  #if XNN_ARCH_ARM && XNN_ENABLE_ARM_FP16_VECTOR && XNN_ENABLE_ARM_FP16_SCALAR
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon_fp16_arith) {
      f16_rndd_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f16_vrndd_ukernel__neonfp16arith_u16;
      f16_rndd_config.element_tile = 16;
    }
  #elif XNN_ARCH_ARM64 && XNN_ENABLE_ARM_FP16_VECTOR
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon_fp16_arith) {
      f16_rndd_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f16_vrndd_ukernel__neonfp16arith_u16;
      f16_rndd_config.element_tile = 16;
    }
  #elif (XNN_ARCH_X86 || XNN_ARCH_X86_64) && !XNN_PLATFORM_MOBILE
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_x86_f16c) {
      f16_rndd_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f16_vrndd_ukernel__f16c_u16;
      f16_rndd_config.element_tile = 16;
    }
  #endif
}

static void init_f16_rndne_config(void) {
  #if XNN_ARCH_ARM && XNN_ENABLE_ARM_FP16_VECTOR && XNN_ENABLE_ARM_FP16_SCALAR
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon_fp16_arith) {
      f16_rndne_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f16_vrndne_ukernel__neonfp16arith_u16;
      f16_rndne_config.element_tile = 16;
    }
  #elif XNN_ARCH_ARM64 && XNN_ENABLE_ARM_FP16_VECTOR
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon_fp16_arith) {
      f16_rndne_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f16_vrndne_ukernel__neonfp16arith_u16;
      f16_rndne_config.element_tile = 16;
    }
  #elif (XNN_ARCH_X86 || XNN_ARCH_X86_64) && !XNN_PLATFORM_MOBILE
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_x86_f16c) {
      f16_rndne_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f16_vrndne_ukernel__f16c_u16;
      f16_rndne_config.element_tile = 16;
    }
  #endif
}

static void init_f16_rndu_config(void) {
  #if XNN_ARCH_ARM && XNN_ENABLE_ARM_FP16_VECTOR && XNN_ENABLE_ARM_FP16_SCALAR
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon_fp16_arith) {
      f16_rndu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f16_vrndu_ukernel__neonfp16arith_u16;
      f16_rndu_config.element_tile = 16;
    }
  #elif XNN_ARCH_ARM64 && XNN_ENABLE_ARM_FP16_VECTOR
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon_fp16_arith) {
      f16_rndu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f16_vrndu_ukernel__neonfp16arith_u16;
      f16_rndu_config.element_tile = 16;
    }
  #elif (XNN_ARCH_X86 || XNN_ARCH_X86_64) && !XNN_PLATFORM_MOBILE
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_x86_f16c) {
      f16_rndu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f16_vrndu_ukernel__f16c_u16;
      f16_rndu_config.element_tile = 16;
    }
  #endif
}

static void init_f16_rndz_config(void) {
  #if XNN_ARCH_ARM && XNN_ENABLE_ARM_FP16_VECTOR && XNN_ENABLE_ARM_FP16_SCALAR
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon_fp16_arith) {
      f16_rndz_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f16_vrndz_ukernel__neonfp16arith_u16;
      f16_rndz_config.element_tile = 16;
    }
  #elif XNN_ARCH_ARM64 && XNN_ENABLE_ARM_FP16_VECTOR
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon_fp16_arith) {
      f16_rndz_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f16_vrndz_ukernel__neonfp16arith_u16;
      f16_rndz_config.element_tile = 16;
    }
  #elif (XNN_ARCH_X86 || XNN_ARCH_X86_64) && !XNN_PLATFORM_MOBILE
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_x86_f16c) {
      f16_rndz_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f16_vrndz_ukernel__f16c_u16;
      f16_rndz_config.element_tile = 16;
    }
  #endif
}

static void init_f16_sigmoid_config(void) {
  #if XNN_ARCH_ARM && XNN_ENABLE_ARM_FP16_VECTOR && XNN_ENABLE_ARM_FP16_SCALAR
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon_fp16_arith) {
      f16_sigmoid_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1recps_u16;
      f16_sigmoid_config.init.f16_sigmoid = xnn_init_f16_sigmoid_fp16arith_rr2_p2_params;
      f16_sigmoid_config.element_tile = 16;
    }
  #elif XNN_ARCH_ARM64 && XNN_ENABLE_ARM_FP16_VECTOR
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon_fp16_arith) {
      f16_sigmoid_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1fma_u40;
      f16_sigmoid_config.init.f16_sigmoid = xnn_init_f16_sigmoid_fp16arith_rr2_p2_params;
      f16_sigmoid_config.element_tile = 40;
    }
  #elif (XNN_ARCH_X86 || XNN_ARCH_X86_64) && !XNN_PLATFORM_MOBILE
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_x86_avx2) {
      f16_sigmoid_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_rcp_u32;
      f16_sigmoid_config.init.f16_sigmoid = xnn_init_f16_sigmoid_avx2_rr1_p2_params;
      f16_sigmoid_config.element_tile = 32;
    }
  #endif
}

static void init_f16_sqr_config(void) {
  #if XNN_ARCH_ARM && XNN_ENABLE_ARM_FP16_VECTOR && XNN_ENABLE_ARM_FP16_SCALAR
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon_fp16_arith) {
      f16_sqr_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f16_vsqr_ukernel__neonfp16arith_u16;
      f16_sqr_config.element_tile = 16;
    }
  #elif XNN_ARCH_ARM64 && XNN_ENABLE_ARM_FP16_VECTOR
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon_fp16_arith) {
      f16_sqr_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f16_vsqr_ukernel__neonfp16arith_u16;
      f16_sqr_config.element_tile = 16;
    }
  #elif (XNN_ARCH_X86 || XNN_ARCH_X86_64) && !XNN_PLATFORM_MOBILE
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_x86_f16c) {
      f16_sqr_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f16_vsqr_ukernel__f16c_u16;
      f16_sqr_config.element_tile = 16;
    }
  #endif
}

static void init_f16_sqrt_config(void) {
  #if XNN_ARCH_ARM && XNN_ENABLE_ARM_FP16_VECTOR && XNN_ENABLE_ARM_FP16_SCALAR
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon_fp16_arith) {
      f16_sqrt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f16_vsqrt_ukernel__neonfp16arith_nr1fma1adj_u8;
      f16_sqrt_config.element_tile = 8;
    }
  #elif XNN_ARCH_ARM64 && XNN_ENABLE_ARM_FP16_VECTOR
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon_fp16_arith) {
      f16_sqrt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f16_vsqrt_ukernel__aarch64_neonfp16arith_sqrt_u8;
      f16_sqrt_config.element_tile = 8;
    }
  #elif (XNN_ARCH_X86 || XNN_ARCH_X86_64) && !XNN_PLATFORM_MOBILE
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_x86_f16c) {
      f16_sqrt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f16_vsqrt_ukernel__f16c_rsqrt_u32;
      f16_sqrt_config.element_tile = 32;
    }
  #endif
}

static void init_f16_tanh_config(void) {
  #if XNN_ARCH_ARM && XNN_ENABLE_ARM_FP16_VECTOR && XNN_ENABLE_ARM_FP16_SCALAR
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon_fp16_arith) {
      f16_tanh_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1fma_u32;
      f16_tanh_config.element_tile = 32;
    }
  #elif XNN_ARCH_ARM64 && XNN_ENABLE_ARM_FP16_VECTOR
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon_fp16_arith) {
      f16_tanh_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f16_vtanh_ukernel__aarch64_neonfp16arith_expm1minus_rr1_p3h2ts_div_u32;
      f16_tanh_config.element_tile = 32;
    }
  #elif (XNN_ARCH_X86 || XNN_ARCH_X86_64) && !XNN_PLATFORM_MOBILE
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_x86_fma3) {
      f16_tanh_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f16_vtanh_ukernel__fma3_polynomial_p19h9t2_u32;
      f16_tanh_config.init.f16_tanh = xnn_init_f16_tanh_avx_polynomial_p19h9t2_params;
      f16_tanh_config.element_tile = 32;
    } else if (hardware_config->use_x86_f16c) {
      f16_tanh_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_rcp_u72;
      f16_tanh_config.init.f16_tanh = xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params;
      f16_tanh_config.element_tile = 72;
    }
  #endif
}

static void init_f16_to_f32_cvt_config(void) {
  #if XNN_ARCH_ARM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon) {
      if (hardware_config->use_arm_neon_fp16) {
        f16_to_f32_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f16_f32_vcvt_ukernel__neonfp16_u16;
        f16_to_f32_cvt_config.element_tile = 16;
      } else {
        f16_to_f32_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f16_f32_vcvt_ukernel__neon_int16_u16;
        f16_to_f32_cvt_config.init.f16_f32_cvt = xnn_init_f16_f32_cvt_neon_params;
        f16_to_f32_cvt_config.element_tile = 16;
      }
    } else if (!XNN_PLATFORM_MOBILE) {
      f16_to_f32_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f16_f32_vcvt_ukernel__scalar_u4;
      f16_to_f32_cvt_config.init.f16_f32_cvt = xnn_init_f16_f32_cvt_scalar_params;
      f16_to_f32_cvt_config.element_tile = 4;
    }
  #elif XNN_ARCH_ARM64
    f16_to_f32_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f16_f32_vcvt_ukernel__neonfp16_u16;
    f16_to_f32_cvt_config.element_tile = 16;
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_x86_avx512skx) {
      f16_to_f32_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f16_f32_vcvt_ukernel__avx512skx_u16;
      f16_to_f32_cvt_config.element_tile = 16;
    } else if (hardware_config->use_x86_f16c) {
      f16_to_f32_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f16_f32_vcvt_ukernel__f16c_u16;
      f16_to_f32_cvt_config.element_tile = 16;
    } else if (hardware_config->use_x86_avx) {
      f16_to_f32_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f16_f32_vcvt_ukernel__avx_int16_u16;
      f16_to_f32_cvt_config.init.f16_f32_cvt = xnn_init_f16_f32_cvt_sse_int16_params;
      f16_to_f32_cvt_config.element_tile = 16;
    } else if (hardware_config->use_x86_sse4_1) {
      f16_to_f32_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f16_f32_vcvt_ukernel__sse41_int16_u16;
      f16_to_f32_cvt_config.init.f16_f32_cvt = xnn_init_f16_f32_cvt_sse_int16_params;
      f16_to_f32_cvt_config.element_tile = 16;
    } else {
      f16_to_f32_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f16_f32_vcvt_ukernel__sse2_int16_u32;
      f16_to_f32_cvt_config.init.f16_f32_cvt = xnn_init_f16_f32_cvt_sse_int16_params;
      f16_to_f32_cvt_config.element_tile = 32;
    }
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    #if XNN_ARCH_WASMRELAXEDSIMD
      f16_to_f32_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f16_f32_vcvt_ukernel__wasmrelaxedsimd_int16_u16;
      f16_to_f32_cvt_config.init.f16_f32_cvt = xnn_init_f16_f32_cvt_wasmsimd_int16_params;
      f16_to_f32_cvt_config.element_tile = 16;
    #else
      f16_to_f32_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f16_f32_vcvt_ukernel__wasmsimd_int16_u16;
      f16_to_f32_cvt_config.init.f16_f32_cvt = xnn_init_f16_f32_cvt_wasmsimd_int16_params;
      f16_to_f32_cvt_config.element_tile = 16;
    #endif
  #elif XNN_ARCH_WASM
    f16_to_f32_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f16_f32_vcvt_ukernel__scalar_u1;
    f16_to_f32_cvt_config.init.f16_f32_cvt = xnn_init_f16_f32_cvt_scalar_params;
    f16_to_f32_cvt_config.element_tile = 1;
  #elif XNN_ARCH_RISCV
    f16_to_f32_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f16_f32_vcvt_ukernel__scalar_u4;
    f16_to_f32_cvt_config.init.f16_f32_cvt = xnn_init_f16_f32_cvt_scalar_params;
    f16_to_f32_cvt_config.element_tile = 4;
  #elif XNN_ARCH_PPC64
    f16_to_f32_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f16_f32_vcvt_ukernel__scalar_u4;
    f16_to_f32_cvt_config.init.f16_f32_cvt = xnn_init_f16_f32_cvt_scalar_params;
    f16_to_f32_cvt_config.element_tile = 4;
  #endif
}

static void init_f16_to_qs8_cvt_config(void) {
  #if (XNN_ARCH_ARM || XNN_ARCH_ARM64) && XNN_ENABLE_ARM_FP16_VECTOR
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon_fp16_arith) {
      f16_to_qs8_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f16_qs8_vcvt_ukernel__neonfp16arith_u32;
      f16_to_qs8_cvt_config.init.f16_qs8_cvt = xnn_init_f16_qs8_cvt_neonfp16arith_params;
      f16_to_qs8_cvt_config.element_tile = 32;
    }
  #else
    f16_to_qs8_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f16_qs8_vcvt_ukernel__scalar_imagic_u4;
    f16_to_qs8_cvt_config.init.f16_qs8_cvt = xnn_init_f16_qs8_cvt_scalar_imagic_params;
    f16_to_qs8_cvt_config.element_tile = 4;
  #endif
}

static void init_f32_abs_config(void) {
  #if XNN_ARCH_ARM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon) {
      f32_abs_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vabs_ukernel__neon_u8;
      f32_abs_config.element_tile = 8;
    } else if (!XNN_PLATFORM_MOBILE) {
      f32_abs_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vabs_ukernel__scalar_u4;
      f32_abs_config.element_tile = 4;
    }
  #elif XNN_ARCH_ARM64
    f32_abs_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vabs_ukernel__neon_u8;
    f32_abs_config.element_tile = 8;
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (!XNN_PLATFORM_MOBILE && hardware_config->use_x86_avx512f) {
      f32_abs_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vabs_ukernel__avx512f_u16;
      f32_abs_config.init.f32_abs = xnn_init_f32_abs_avx512_params;
      f32_abs_config.element_tile = 16;
    } else if (hardware_config->use_x86_avx) {
      f32_abs_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vabs_ukernel__avx_u16;
      f32_abs_config.init.f32_abs = xnn_init_f32_abs_avx_params;
      f32_abs_config.element_tile = 16;
    } else {
      f32_abs_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vabs_ukernel__sse_u8;
      f32_abs_config.init.f32_abs = xnn_init_f32_abs_sse_params;
      f32_abs_config.element_tile = 8;
    }
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    f32_abs_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vabs_ukernel__wasmsimd_u8;
    f32_abs_config.init.f32_abs = xnn_init_f32_abs_wasmsimd_params;
    f32_abs_config.element_tile = 8;
  #elif XNN_ARCH_WASM
    f32_abs_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vabs_ukernel__scalar_u4;
    f32_abs_config.element_tile = 4;
  #elif XNN_ARCH_RISCV
    f32_abs_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vabs_ukernel__scalar_u4;
    f32_abs_config.element_tile = 4;
  #elif XNN_ARCH_PPC64
    f32_abs_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vabs_ukernel__scalar_u4;
    f32_abs_config.element_tile = 4;
  #endif
}

static void init_f32_clamp_config(void) {
  #if XNN_ARCH_ARM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon) {
      f32_clamp_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vclamp_ukernel__neon_u16;
      f32_clamp_config.init.f32_minmax = xnn_init_f32_minmax_scalar_params;
      f32_clamp_config.element_tile = 16;
    } else if (!XNN_PLATFORM_MOBILE) {
      f32_clamp_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vclamp_ukernel__scalar_u4;
      f32_clamp_config.init.f32_minmax = xnn_init_f32_minmax_scalar_params;
      f32_clamp_config.element_tile = 4;
    }
  #elif XNN_ARCH_ARM64
    f32_clamp_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vclamp_ukernel__neon_u16;
    f32_clamp_config.init.f32_minmax = xnn_init_f32_minmax_scalar_params;
    f32_clamp_config.element_tile = 16;
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (!XNN_PLATFORM_MOBILE && hardware_config->use_x86_avx512f) {
      f32_clamp_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vclamp_ukernel__avx512f_u16;
      f32_clamp_config.init.f32_minmax = xnn_init_f32_minmax_scalar_params;
      f32_clamp_config.element_tile = 16;
    } else if (hardware_config->use_x86_avx) {
      f32_clamp_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vclamp_ukernel__avx_u16;
      f32_clamp_config.init.f32_minmax = xnn_init_f32_minmax_avx_params;
      f32_clamp_config.element_tile = 16;
    } else {
      f32_clamp_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vclamp_ukernel__sse_u8;
      f32_clamp_config.init.f32_minmax = xnn_init_f32_minmax_sse_params;
      f32_clamp_config.element_tile = 8;
    }
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->is_x86) {
      f32_clamp_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vclamp_ukernel__wasmsimd_x86_u8;
      f32_clamp_config.init.f32_minmax = xnn_init_f32_minmax_wasmsimd_params;
      f32_clamp_config.element_tile = 8;
    } else {
      f32_clamp_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vclamp_ukernel__wasmsimd_arm_u8;
      f32_clamp_config.init.f32_minmax = xnn_init_f32_minmax_wasmsimd_params;
      f32_clamp_config.element_tile = 8;
    }
  #elif XNN_ARCH_WASM
    f32_clamp_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vclamp_ukernel__wasm_u4;
    f32_clamp_config.init.f32_minmax = xnn_init_f32_minmax_scalar_params;
    f32_clamp_config.element_tile = 4;
  #elif XNN_ARCH_RISCV
    f32_clamp_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vclamp_ukernel__scalar_u4;
    f32_clamp_config.init.f32_minmax = xnn_init_f32_minmax_scalar_params;
    f32_clamp_config.element_tile = 4;
  #elif XNN_ARCH_PPC64
    f32_clamp_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vclamp_ukernel__scalar_u4;
    f32_clamp_config.init.f32_minmax = xnn_init_f32_minmax_scalar_params;
    f32_clamp_config.element_tile = 4;
  #endif
}

static void init_f32_elu_config(void) {
  #if XNN_ARCH_ARM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon) {
      if (hardware_config->use_arm_neon_fma) {
        f32_elu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_velu_ukernel__neonfma_rr1_p6_u8;
        f32_elu_config.init.f32_elu = xnn_init_f32_elu_neonfma_rr1_p6_params;
        f32_elu_config.element_tile = 8;
      } else {
        f32_elu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_velu_ukernel__neon_rr2_lut16_p3_u8;
        f32_elu_config.init.f32_elu = xnn_init_f32_elu_neon_rr2_lut16_p3_params;
        f32_elu_config.element_tile = 8;
      }
    } else if (!XNN_PLATFORM_MOBILE) {
      f32_elu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_velu_ukernel__scalar_rr2_lut16_p3_u4;
      f32_elu_config.init.f32_elu = xnn_init_f32_elu_scalar_rr2_lut16_p3_params;
      f32_elu_config.element_tile = 4;
    }
  #elif XNN_ARCH_ARM64
    f32_elu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_velu_ukernel__neonfma_rr1_lut16_p3_u16;
    f32_elu_config.init.f32_elu = xnn_init_f32_elu_neonfma_rr1_lut16_p3_params;
    f32_elu_config.element_tile = 16;
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (!XNN_PLATFORM_MOBILE && hardware_config->use_x86_avx512f) {
      f32_elu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_velu_ukernel__avx512f_rr1_p6_u128;
      f32_elu_config.init.f32_elu = xnn_init_f32_elu_avx512_rr1_p6_params;
      f32_elu_config.element_tile = 128;
    } else if (hardware_config->use_x86_avx2) {
      f32_elu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_velu_ukernel__avx2_rr1_lut4_p4_perm_u56;
      f32_elu_config.init.f32_elu = xnn_init_f32_elu_avx2_rr1_lut4_p4_params;
      f32_elu_config.element_tile = 56;
    } else if (hardware_config->use_x86_avx) {
      f32_elu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_velu_ukernel__avx_rr2_lut4_p4_perm_u32;
      f32_elu_config.init.f32_elu = xnn_init_f32_elu_avx_rr2_lut4_p4_params;
      f32_elu_config.element_tile = 32;
    } else {
      f32_elu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_velu_ukernel__sse2_rr2_lut16_p3_u12;
      f32_elu_config.init.f32_elu = xnn_init_f32_elu_sse2_rr2_lut16_p3_params;
      f32_elu_config.element_tile = 12;
    }
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    #if XNN_ARCH_WASMRELAXEDSIMD
      f32_elu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_p6_u24;
      f32_elu_config.init.f32_elu = xnn_init_f32_elu_wasmsimd_rr2_p6_params;
      f32_elu_config.element_tile = 24;
    #else
      const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
      assert(hardware_config != NULL);
      if (hardware_config->is_x86) {
        f32_elu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_velu_ukernel__wasmsimd_x86_rr2_p6_u20;
        f32_elu_config.init.f32_elu = xnn_init_f32_elu_wasmsimd_rr2_p6_params;
        f32_elu_config.element_tile = 20;
      } else {
        f32_elu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_velu_ukernel__wasmsimd_arm_rr2_p6_u20;
        f32_elu_config.init.f32_elu = xnn_init_f32_elu_wasmsimd_rr2_p6_params;
        f32_elu_config.element_tile = 20;
      }
    #endif
  #elif XNN_ARCH_WASM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->is_x86) {
      f32_elu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_velu_ukernel__scalar_rr2_lut16_p3_u2;
      f32_elu_config.init.f32_elu = xnn_init_f32_elu_scalar_rr2_lut16_p3_params;
      f32_elu_config.element_tile = 2;
    } else {
      f32_elu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_velu_ukernel__wasm_rr2_p6_u6;
      f32_elu_config.init.f32_elu = xnn_init_f32_elu_scalar_rr2_p6_params;
      f32_elu_config.element_tile = 6;
    }
  #elif XNN_ARCH_RISCV
    f32_elu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_velu_ukernel__scalar_rr2_lut16_p3_u4;
    f32_elu_config.init.f32_elu = xnn_init_f32_elu_scalar_rr2_lut16_p3_params;
    f32_elu_config.element_tile = 4;
  #elif XNN_ARCH_PPC64
    f32_elu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_velu_ukernel__scalar_rr2_lut16_p3_u4;
    f32_elu_config.init.f32_elu = xnn_init_f32_elu_scalar_rr2_lut16_p3_params;
    f32_elu_config.element_tile = 4;
  #endif
}

static void init_f32_hswish_config(void) {
  #if XNN_ARCH_ARM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon) {
      f32_hswish_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vhswish_ukernel__neon_u16;
      f32_hswish_config.init.f32_hswish = xnn_init_f32_hswish_scalar_params;
      f32_hswish_config.element_tile = 16;
    } else if (!XNN_PLATFORM_MOBILE) {
      f32_hswish_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vhswish_ukernel__scalar_u4;
      f32_hswish_config.init.f32_hswish = xnn_init_f32_hswish_scalar_params;
      f32_hswish_config.element_tile = 4;
    }
  #elif XNN_ARCH_ARM64
    f32_hswish_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vhswish_ukernel__neon_u16;
    f32_hswish_config.init.f32_hswish = xnn_init_f32_hswish_scalar_params;
    f32_hswish_config.element_tile = 16;
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (!XNN_PLATFORM_MOBILE && hardware_config->use_x86_avx512f) {
      f32_hswish_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vhswish_ukernel__avx512f_u16;
      f32_hswish_config.init.f32_hswish = xnn_init_f32_hswish_avx512_params;
      f32_hswish_config.element_tile = 16;
    } else if (hardware_config->use_x86_fma3) {
      f32_hswish_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vhswish_ukernel__fma3_u16;
      f32_hswish_config.init.f32_hswish = xnn_init_f32_hswish_avx_params;
      f32_hswish_config.element_tile = 16;
    } else if (hardware_config->use_x86_avx) {
      f32_hswish_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vhswish_ukernel__avx_u16;
      f32_hswish_config.init.f32_hswish = xnn_init_f32_hswish_avx_params;
      f32_hswish_config.element_tile = 16;
    } else {
      f32_hswish_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vhswish_ukernel__sse_u8;
      f32_hswish_config.init.f32_hswish = xnn_init_f32_hswish_sse_params;
      f32_hswish_config.element_tile = 8;
    }
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    f32_hswish_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vhswish_ukernel__wasmsimd_u16;
    f32_hswish_config.init.f32_hswish = xnn_init_f32_hswish_wasmsimd_params;
    f32_hswish_config.element_tile = 16;
  #elif XNN_ARCH_WASM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->is_x86) {
      f32_hswish_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vhswish_ukernel__scalar_u4;
      f32_hswish_config.init.f32_hswish = xnn_init_f32_hswish_scalar_params;
      f32_hswish_config.element_tile = 4;
    } else {
      f32_hswish_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vhswish_ukernel__wasm_u4;
      f32_hswish_config.init.f32_hswish = xnn_init_f32_hswish_scalar_params;
      f32_hswish_config.element_tile = 4;
    }
  #elif XNN_ARCH_RISCV
    f32_hswish_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vhswish_ukernel__scalar_u4;
    f32_hswish_config.init.f32_hswish = xnn_init_f32_hswish_scalar_params;
    f32_hswish_config.element_tile = 4;
  #elif XNN_ARCH_PPC64
    f32_hswish_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vhswish_ukernel__scalar_u4;
    f32_hswish_config.init.f32_hswish = xnn_init_f32_hswish_scalar_params;
    f32_hswish_config.element_tile = 4;
  #endif
}

static void init_f32_lrelu_config(void) {
  #if XNN_ARCH_ARM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon) {
      f32_lrelu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vlrelu_ukernel__neon_u8;
      f32_lrelu_config.init.f32_lrelu = xnn_init_f32_lrelu_scalar_params;
      f32_lrelu_config.element_tile = 8;
    } else if (!XNN_PLATFORM_MOBILE) {
      f32_lrelu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vlrelu_ukernel__scalar_u4;
      f32_lrelu_config.init.f32_lrelu = xnn_init_f32_lrelu_scalar_params;
      f32_lrelu_config.element_tile = 4;
    }
  #elif XNN_ARCH_ARM64
    f32_lrelu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vlrelu_ukernel__neon_u8;
    f32_lrelu_config.init.f32_lrelu = xnn_init_f32_lrelu_scalar_params;
    f32_lrelu_config.element_tile = 8;
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (!XNN_PLATFORM_MOBILE && hardware_config->use_x86_avx512f) {
      f32_lrelu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vlrelu_ukernel__avx512f_u16;
      f32_lrelu_config.init.f32_lrelu = xnn_init_f32_lrelu_scalar_params;
      f32_lrelu_config.element_tile = 16;
    } else if (hardware_config->use_x86_avx) {
      f32_lrelu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vlrelu_ukernel__avx_u16;
      f32_lrelu_config.init.f32_lrelu = xnn_init_f32_lrelu_avx_params;
      f32_lrelu_config.element_tile = 16;
    } else if (hardware_config->use_x86_sse4_1) {
      f32_lrelu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vlrelu_ukernel__sse41_u8;
      f32_lrelu_config.init.f32_lrelu = xnn_init_f32_lrelu_sse_params;
      f32_lrelu_config.element_tile = 8;
    } else {
      f32_lrelu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vlrelu_ukernel__sse_u8;
      f32_lrelu_config.init.f32_lrelu = xnn_init_f32_lrelu_sse_params;
      f32_lrelu_config.element_tile = 8;
    }
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    #if XNN_ARCH_WASMRELAXEDSIMD
      if (hardware_config->is_x86) {
        f32_lrelu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vlrelu_ukernel__wasmrelaxedsimd_iminmax_u4;
        f32_lrelu_config.init.f32_lrelu = xnn_init_f32_lrelu_wasmsimd_params;
        f32_lrelu_config.element_tile = 4;
      } else {
        f32_lrelu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vlrelu_ukernel__wasmrelaxedsimd_laneselect_u4;
        f32_lrelu_config.init.f32_lrelu = xnn_init_f32_lrelu_wasmsimd_params;
        f32_lrelu_config.element_tile = 4;
      }
    #else
      if (hardware_config->is_x86) {
        f32_lrelu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vlrelu_ukernel__wasmsimd_iminmax_u8;
        f32_lrelu_config.init.f32_lrelu = xnn_init_f32_lrelu_wasmsimd_params;
        f32_lrelu_config.element_tile = 8;
      } else {
        f32_lrelu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vlrelu_ukernel__wasmsimd_laneselect_u8;
        f32_lrelu_config.init.f32_lrelu = xnn_init_f32_lrelu_wasmsimd_params;
        f32_lrelu_config.element_tile = 8;
      }
    #endif
  #elif XNN_ARCH_WASM
    f32_lrelu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vlrelu_ukernel__scalar_u4;
    f32_lrelu_config.init.f32_lrelu = xnn_init_f32_lrelu_scalar_params;
    f32_lrelu_config.element_tile = 4;
  #elif XNN_ARCH_RISCV
    f32_lrelu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vlrelu_ukernel__scalar_u4;
    f32_lrelu_config.init.f32_lrelu = xnn_init_f32_lrelu_scalar_params;
    f32_lrelu_config.element_tile = 4;
  #elif XNN_ARCH_PPC64
    f32_lrelu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vlrelu_ukernel__scalar_u4;
    f32_lrelu_config.init.f32_lrelu = xnn_init_f32_lrelu_scalar_params;
    f32_lrelu_config.element_tile = 4;
  #endif
}

static void init_f32_neg_config(void) {
  #if XNN_ARCH_ARM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon) {
      f32_neg_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vneg_ukernel__neon_u8;
      f32_neg_config.element_tile = 8;
    } else if (!XNN_PLATFORM_MOBILE) {
      f32_neg_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vneg_ukernel__scalar_u4;
      f32_neg_config.element_tile = 4;
    }
  #elif XNN_ARCH_ARM64
    f32_neg_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vneg_ukernel__neon_u8;
    f32_neg_config.element_tile = 8;
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (!XNN_PLATFORM_MOBILE && hardware_config->use_x86_avx512f) {
      f32_neg_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vneg_ukernel__avx512f_u16;
      f32_neg_config.init.f32_neg = xnn_init_f32_neg_avx512_params;
      f32_neg_config.element_tile = 16;
    } else if (hardware_config->use_x86_avx) {
      f32_neg_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vneg_ukernel__avx_u16;
      f32_neg_config.init.f32_neg = xnn_init_f32_neg_avx_params;
      f32_neg_config.element_tile = 16;
    } else {
      f32_neg_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vneg_ukernel__sse_u8;
      f32_neg_config.init.f32_neg = xnn_init_f32_neg_sse_params;
      f32_neg_config.element_tile = 8;
    }
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    f32_neg_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vneg_ukernel__wasmsimd_u8;
    f32_neg_config.init.f32_neg = xnn_init_f32_neg_wasmsimd_params;
    f32_neg_config.element_tile = 8;
  #elif XNN_ARCH_WASM
    f32_neg_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vneg_ukernel__scalar_u4;
    f32_neg_config.element_tile = 4;
  #elif XNN_ARCH_RISCV
    f32_neg_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vneg_ukernel__scalar_u4;
    f32_neg_config.element_tile = 4;
  #elif XNN_ARCH_PPC64
    f32_neg_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vneg_ukernel__scalar_u4;
    f32_neg_config.element_tile = 4;
  #endif
}

static void init_f32_relu_config(void) {
  #if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    f32_relu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vrelu_ukernel__wasmsimd_u16;
    f32_relu_config.element_tile = 16;
  #elif XNN_ARCH_WASM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->is_x86) {
      f32_relu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vrelu_ukernel__scalar_u8;
      f32_relu_config.element_tile = 8;
    } else {
      f32_relu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vrelu_ukernel__wasm_u8;
      f32_relu_config.element_tile = 8;
    }
  #endif
}

static void init_f32_rndd_config(void) {
  #if XNN_ARCH_ARM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon) {
      if (hardware_config->use_arm_neon_v8) {
        f32_rndd_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vrndd_ukernel__neonv8_u8;
        f32_rndd_config.element_tile = 8;
      } else {
        f32_rndd_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vrndd_ukernel__neon_u8;
        f32_rndd_config.element_tile = 8;
      }
    } else if (!XNN_PLATFORM_MOBILE) {
      f32_rndd_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vrndd_ukernel__scalar_libm_u1;
      f32_rndd_config.element_tile = 1;
    }
  #elif XNN_ARCH_ARM64
    f32_rndd_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vrndd_ukernel__neonv8_u8;
    f32_rndd_config.element_tile = 8;
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (!XNN_PLATFORM_MOBILE && hardware_config->use_x86_avx512f) {
      f32_rndd_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vrndd_ukernel__avx512f_u16;
      f32_rndd_config.element_tile = 16;
    } else if (hardware_config->use_x86_avx) {
      f32_rndd_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vrndd_ukernel__avx_u16;
      f32_rndd_config.init.f32_rnd = xnn_init_f32_rnd_avx_params;
      f32_rndd_config.element_tile = 16;
    } else if (hardware_config->use_x86_sse4_1) {
      f32_rndd_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vrndd_ukernel__sse41_u8;
      f32_rndd_config.element_tile = 8;
    } else {
      f32_rndd_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vrndd_ukernel__sse2_u8;
      f32_rndd_config.init.f32_rnd = xnn_init_f32_rnd_sse2_params;
      f32_rndd_config.element_tile = 8;
    }
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    f32_rndd_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vrndd_ukernel__wasmsimd_u8;
    f32_rndd_config.element_tile = 8;
  #elif XNN_ARCH_WASM
    f32_rndd_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vrndd_ukernel__scalar_libm_u4;
    f32_rndd_config.element_tile = 4;
  #elif XNN_ARCH_RISCV
    f32_rndd_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vrndd_ukernel__scalar_libm_u1;
    f32_rndd_config.element_tile = 1;
  #elif XNN_ARCH_PPC64
    f32_rndd_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vrndd_ukernel__scalar_libm_u1;
    f32_rndd_config.element_tile = 1;
  #endif
}

static void init_f32_rndne_config(void) {
  #if XNN_ARCH_ARM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon) {
      if (hardware_config->use_arm_neon_v8) {
        f32_rndne_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vrndne_ukernel__neonv8_u8;
        f32_rndne_config.element_tile = 8;
      } else {
        f32_rndne_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vrndne_ukernel__neon_u8;
        f32_rndne_config.element_tile = 8;
      }
    } else if (!XNN_PLATFORM_MOBILE) {
      f32_rndne_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vrndne_ukernel__scalar_libm_u1;
      f32_rndne_config.element_tile = 1;
    }
  #elif XNN_ARCH_ARM64
    f32_rndne_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vrndne_ukernel__neonv8_u8;
    f32_rndne_config.element_tile = 8;
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (!XNN_PLATFORM_MOBILE && hardware_config->use_x86_avx512f) {
      f32_rndne_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vrndne_ukernel__avx512f_u16;
      f32_rndne_config.element_tile = 16;
    } else if (hardware_config->use_x86_avx) {
      f32_rndne_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vrndne_ukernel__avx_u16;
      f32_rndne_config.init.f32_rnd = xnn_init_f32_rnd_avx_params;
      f32_rndne_config.element_tile = 16;
    } else if (hardware_config->use_x86_sse4_1) {
      f32_rndne_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vrndne_ukernel__sse41_u8;
      f32_rndne_config.element_tile = 8;
    } else {
      f32_rndne_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vrndne_ukernel__sse2_u8;
      f32_rndne_config.init.f32_rnd = xnn_init_f32_rnd_sse2_params;
      f32_rndne_config.element_tile = 8;
    }
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    f32_rndne_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vrndne_ukernel__wasmsimd_u8;
    f32_rndne_config.element_tile = 8;
  #elif XNN_ARCH_WASM
    f32_rndne_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vrndne_ukernel__scalar_libm_u4;
    f32_rndne_config.element_tile = 4;
  #elif XNN_ARCH_RISCV
    f32_rndne_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vrndne_ukernel__scalar_libm_u1;
    f32_rndne_config.element_tile = 1;
  #elif XNN_ARCH_PPC64
    f32_rndne_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vrndne_ukernel__scalar_libm_u1;
    f32_rndne_config.element_tile = 1;
  #endif
}

static void init_f32_rndu_config(void) {
  #if XNN_ARCH_ARM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon) {
      if (hardware_config->use_arm_neon_v8) {
        f32_rndu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vrndu_ukernel__neonv8_u8;
        f32_rndu_config.element_tile = 8;
      } else {
        f32_rndu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vrndu_ukernel__neon_u8;
        f32_rndu_config.element_tile = 8;
      }
    } else if (!XNN_PLATFORM_MOBILE) {
      f32_rndu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vrndu_ukernel__scalar_libm_u1;
      f32_rndu_config.element_tile = 1;
    }
  #elif XNN_ARCH_ARM64
    f32_rndu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vrndu_ukernel__neonv8_u8;
    f32_rndu_config.element_tile = 8;
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (!XNN_PLATFORM_MOBILE && hardware_config->use_x86_avx512f) {
      f32_rndu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vrndu_ukernel__avx512f_u16;
      f32_rndu_config.element_tile = 16;
    } else if (hardware_config->use_x86_avx) {
      f32_rndu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vrndu_ukernel__avx_u16;
      f32_rndu_config.init.f32_rnd = xnn_init_f32_rnd_avx_params;
      f32_rndu_config.element_tile = 16;
    } else if (hardware_config->use_x86_sse4_1) {
      f32_rndu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vrndu_ukernel__sse41_u8;
      f32_rndu_config.element_tile = 8;
    } else {
      f32_rndu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vrndu_ukernel__sse2_u8;
      f32_rndu_config.init.f32_rnd = xnn_init_f32_rnd_sse2_params;
      f32_rndu_config.element_tile = 8;
    }
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    f32_rndu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vrndu_ukernel__wasmsimd_u8;
    f32_rndu_config.element_tile = 8;
  #elif XNN_ARCH_WASM
    f32_rndu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vrndu_ukernel__scalar_libm_u4;
    f32_rndu_config.element_tile = 4;
  #elif XNN_ARCH_RISCV
    f32_rndu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vrndu_ukernel__scalar_libm_u1;
    f32_rndu_config.element_tile = 1;
  #elif XNN_ARCH_PPC64
    f32_rndu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vrndu_ukernel__scalar_libm_u1;
    f32_rndu_config.element_tile = 1;
  #endif
}

static void init_f32_rndz_config(void) {
  #if XNN_ARCH_ARM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon) {
      if (hardware_config->use_arm_neon_v8) {
        f32_rndz_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vrndz_ukernel__neonv8_u8;
        f32_rndz_config.element_tile = 8;
      } else {
        f32_rndz_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vrndz_ukernel__neon_u8;
        f32_rndz_config.element_tile = 8;
      }
    } else if (!XNN_PLATFORM_MOBILE) {
      f32_rndz_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vrndz_ukernel__scalar_libm_u1;
      f32_rndz_config.element_tile = 1;
    }
  #elif XNN_ARCH_ARM64
    f32_rndz_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vrndz_ukernel__neonv8_u8;
    f32_rndz_config.element_tile = 8;
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (!XNN_PLATFORM_MOBILE && hardware_config->use_x86_avx512f) {
      f32_rndz_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vrndz_ukernel__avx512f_u16;
      f32_rndz_config.element_tile = 16;
    } else if (hardware_config->use_x86_avx) {
      f32_rndz_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vrndz_ukernel__avx_u16;
      f32_rndz_config.init.f32_rnd = xnn_init_f32_rnd_avx_params;
      f32_rndz_config.element_tile = 16;
    } else if (hardware_config->use_x86_sse4_1) {
      f32_rndz_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vrndz_ukernel__sse41_u8;
      f32_rndz_config.element_tile = 8;
    } else {
      f32_rndz_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vrndz_ukernel__sse2_u8;
      f32_rndz_config.init.f32_rnd = xnn_init_f32_rnd_sse2_params;
      f32_rndz_config.element_tile = 8;
    }
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    f32_rndz_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vrndz_ukernel__wasmsimd_u8;
    f32_rndz_config.element_tile = 8;
  #elif XNN_ARCH_WASM
    f32_rndz_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vrndz_ukernel__scalar_libm_u4;
    f32_rndz_config.element_tile = 4;
  #elif XNN_ARCH_RISCV
    f32_rndz_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vrndz_ukernel__scalar_libm_u1;
    f32_rndz_config.element_tile = 1;
  #elif XNN_ARCH_PPC64
    f32_rndz_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vrndz_ukernel__scalar_libm_u1;
    f32_rndz_config.element_tile = 1;
  #endif
}

static void init_f32_sigmoid_config(void) {
  #if XNN_ARCH_ARM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon) {
      f32_sigmoid_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vsigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_u8;
      f32_sigmoid_config.init.f32_sigmoid = xnn_init_f32_sigmoid_neon_rr2_lut64_p2_params;
      f32_sigmoid_config.element_tile = 8;
    } else if (!XNN_PLATFORM_MOBILE) {
      f32_sigmoid_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vsigmoid_ukernel__scalar_rr2_lut64_p2_div_u2;
      f32_sigmoid_config.init.f32_sigmoid = xnn_init_f32_sigmoid_scalar_rr2_lut64_p2_params;
      f32_sigmoid_config.element_tile = 2;
    }
  #elif XNN_ARCH_ARM64
    f32_sigmoid_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_u16;
    f32_sigmoid_config.init.f32_sigmoid = xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params;
    f32_sigmoid_config.element_tile = 16;
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (!XNN_PLATFORM_MOBILE && hardware_config->use_x86_avx512f) {
      f32_sigmoid_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_u64;
      f32_sigmoid_config.init.f32_sigmoid = xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params;
      f32_sigmoid_config.element_tile = 64;
    } else if (hardware_config->use_x86_avx2) {
      f32_sigmoid_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_u40;
      f32_sigmoid_config.init.f32_sigmoid = xnn_init_f32_sigmoid_avx2_rr1_p5_params;
      f32_sigmoid_config.element_tile = 40;
    } else if (hardware_config->use_x86_avx) {
      f32_sigmoid_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_u40;
      f32_sigmoid_config.init.f32_sigmoid = xnn_init_f32_sigmoid_avx_rr2_p5_params;
      f32_sigmoid_config.element_tile = 40;
    } else if (hardware_config->use_x86_sse4_1) {
      f32_sigmoid_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vsigmoid_ukernel__sse41_rr2_lut64_p2_div_u8;
      f32_sigmoid_config.init.f32_sigmoid = xnn_init_f32_sigmoid_sse2_rr2_lut64_p2_params;
      f32_sigmoid_config.element_tile = 8;
    } else {
      f32_sigmoid_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vsigmoid_ukernel__sse2_rr2_lut64_p2_div_u8;
      f32_sigmoid_config.init.f32_sigmoid = xnn_init_f32_sigmoid_sse2_rr2_lut64_p2_params;
      f32_sigmoid_config.element_tile = 8;
    }
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    #if XNN_ARCH_WASMRELAXEDSIMD
      f32_sigmoid_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_p5_div_u24;
      f32_sigmoid_config.init.f32_sigmoid = xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params;
      f32_sigmoid_config.element_tile = 24;
    #else
      f32_sigmoid_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_p5_div_u16;
      f32_sigmoid_config.init.f32_sigmoid = xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params;
      f32_sigmoid_config.element_tile = 16;
    #endif
  #elif XNN_ARCH_WASM
    f32_sigmoid_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vsigmoid_ukernel__scalar_rr2_lut64_p2_div_u2;
    f32_sigmoid_config.init.f32_sigmoid = xnn_init_f32_sigmoid_scalar_rr2_lut64_p2_params;
    f32_sigmoid_config.element_tile = 2;
  #elif XNN_ARCH_RISCV
    f32_sigmoid_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vsigmoid_ukernel__scalar_rr2_lut64_p2_div_u2;
    f32_sigmoid_config.init.f32_sigmoid = xnn_init_f32_sigmoid_scalar_rr2_lut64_p2_params;
    f32_sigmoid_config.element_tile = 2;
  #elif XNN_ARCH_PPC64
    f32_sigmoid_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vsigmoid_ukernel__scalar_rr2_lut64_p2_div_u2;
    f32_sigmoid_config.init.f32_sigmoid = xnn_init_f32_sigmoid_scalar_rr2_lut64_p2_params;
    f32_sigmoid_config.element_tile = 2;
  #endif
}

static void init_f32_sqr_config(void) {
  #if XNN_ARCH_ARM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon) {
      f32_sqr_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vsqr_ukernel__neon_u8;
      f32_sqr_config.element_tile = 8;
    } else if (!XNN_PLATFORM_MOBILE) {
      f32_sqr_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vsqr_ukernel__scalar_u4;
      f32_sqr_config.element_tile = 4;
    }
  #elif XNN_ARCH_ARM64
    f32_sqr_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vsqr_ukernel__neon_u8;
    f32_sqr_config.element_tile = 8;
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (!XNN_PLATFORM_MOBILE && hardware_config->use_x86_avx512f) {
      f32_sqr_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vsqr_ukernel__avx512f_u16;
      f32_sqr_config.element_tile = 16;
    } else if (hardware_config->use_x86_avx) {
      f32_sqr_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vsqr_ukernel__avx_u16;
      f32_sqr_config.init.f32_default = xnn_init_f32_default_avx_params;
      f32_sqr_config.element_tile = 16;
    } else {
      f32_sqr_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vsqr_ukernel__sse_u8;
      f32_sqr_config.element_tile = 8;
    }
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    f32_sqr_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vsqr_ukernel__wasmsimd_u8;
    f32_sqr_config.element_tile = 8;
  #elif XNN_ARCH_WASM
    f32_sqr_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vsqr_ukernel__scalar_u4;
    f32_sqr_config.element_tile = 4;
  #elif XNN_ARCH_RISCV
    f32_sqr_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vsqr_ukernel__scalar_u4;
    f32_sqr_config.element_tile = 4;
  #elif XNN_ARCH_PPC64
    f32_sqr_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vsqr_ukernel__scalar_u4;
    f32_sqr_config.element_tile = 4;
  #endif
}

static void init_f32_sqrt_config(void) {
  #if XNN_ARCH_ARM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon) {
      f32_sqrt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vsqrt_ukernel__scalar_sqrt_u1;
      f32_sqrt_config.element_tile = 1;
    } else if (!XNN_PLATFORM_MOBILE) {
      f32_sqrt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vsqrt_ukernel__scalar_sqrt_u1;
      f32_sqrt_config.element_tile = 1;
    }
  #elif XNN_ARCH_ARM64
    f32_sqrt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vsqrt_ukernel__aarch64_neon_sqrt_u4;
    f32_sqrt_config.element_tile = 4;
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_x86_avx) {
      f32_sqrt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vsqrt_ukernel__avx_sqrt_u8;
      f32_sqrt_config.init.f32_sqrt = xnn_init_f32_sqrt_avx_params;
      f32_sqrt_config.element_tile = 8;
    } else {
      f32_sqrt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vsqrt_ukernel__sse_sqrt_u4;
      f32_sqrt_config.element_tile = 4;
    }
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    f32_sqrt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vsqrt_ukernel__wasmsimd_sqrt_u8;
    f32_sqrt_config.element_tile = 8;
  #elif XNN_ARCH_WASM
    f32_sqrt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vsqrt_ukernel__scalar_sqrt_u1;
    f32_sqrt_config.element_tile = 1;
  #elif XNN_ARCH_RISCV
    f32_sqrt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vsqrt_ukernel__scalar_sqrt_u1;
    f32_sqrt_config.element_tile = 1;
  #elif XNN_ARCH_PPC64
    f32_sqrt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vsqrt_ukernel__scalar_sqrt_u1;
    f32_sqrt_config.element_tile = 1;
  #endif
}

static void init_f32_rsqrt_config(void) {
  #if XNN_ARCH_X86 || XNN_ARCH_X86_64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (!XNN_PLATFORM_MOBILE && hardware_config->use_x86_avx512f) {
      f32_rsqrt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vrsqrt_ukernel__avx512f_rsqrt_u32;
      f32_rsqrt_config.init.f32_rsqrt = xnn_init_f32_rsqrt_avx512_params;
      f32_rsqrt_config.element_tile = 32;
    } else if (!XNN_PLATFORM_MOBILE && hardware_config->use_x86_fma3) {
      f32_rsqrt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vrsqrt_ukernel__fma3_rsqrt_u16;
      f32_rsqrt_config.init.f32_rsqrt = xnn_init_f32_rsqrt_fma3_params;
      f32_rsqrt_config.element_tile = 16;
    } else if (hardware_config->use_x86_avx) {
      f32_rsqrt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vrsqrt_ukernel__avx_rsqrt_u16;
      f32_rsqrt_config.init.f32_rsqrt = xnn_init_f32_rsqrt_avx_params;
      f32_rsqrt_config.element_tile = 16;
    } else {
      f32_rsqrt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vrsqrt_ukernel__sse_rsqrt_u8;
      f32_rsqrt_config.init.f32_rsqrt = xnn_init_f32_rsqrt_sse_params;
      f32_rsqrt_config.element_tile = 8;
    }
  #else
    f32_rsqrt_config.ukernel =
        (xnn_vunary_ukernel_fn)xnn_f32_vrsqrt_ukernel__scalar_rsqrt_u4;
    f32_rsqrt_config.element_tile = 4;
  #endif
}

static void init_f32_tanh_config(void) {
  #if XNN_ARCH_ARM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon_fma) {
      f32_tanh_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vtanh_ukernel__neonfma_expm1minus_rr1_p6h5ts_nr2fma_u8;
      f32_tanh_config.init.f32_tanh = xnn_init_f32_tanh_neon_expm1minus_rr1_p6h5_params;
      f32_tanh_config.element_tile = 8;
    } else if (hardware_config->use_arm_neon) {
      f32_tanh_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vtanh_ukernel__neon_expm1minus_rr1_p6h5ts_nr2recps_u8;
      f32_tanh_config.init.f32_tanh = xnn_init_f32_tanh_neon_expm1minus_rr1_p6h5_params;
      f32_tanh_config.element_tile = 8;
    } else if (!XNN_PLATFORM_MOBILE) {
      f32_tanh_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vtanh_ukernel__scalar_expm1minus_rr1_lut8_p4h3ts_div_u4;
      f32_tanh_config.init.f32_tanh = xnn_init_f32_tanh_scalar_expm1minus_rr1_lut8_p4h3_params;
      f32_tanh_config.element_tile = 4;
    }
  #elif XNN_ARCH_ARM64
    f32_tanh_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vtanh_ukernel__aarch64_neonfma_expm1minus_rr1_p6h5ts_div_u16;
    f32_tanh_config.init.f32_tanh = xnn_init_f32_tanh_neon_expm1minus_rr1_p6h5_params;
    f32_tanh_config.element_tile = 16;
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (!XNN_PLATFORM_MOBILE && hardware_config->use_x86_avx512skx) {
      f32_tanh_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_div_u64;
      f32_tanh_config.init.f32_tanh = xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params;
      f32_tanh_config.element_tile = 64;
    } else if (hardware_config->use_x86_avx2) {
      f32_tanh_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3ts_perm_div_u32;
      f32_tanh_config.init.f32_tanh = xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params;
      f32_tanh_config.element_tile = 32;
    } else if (hardware_config->use_x86_fma3) {
      f32_tanh_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3ts_perm_div_u40;
      f32_tanh_config.init.f32_tanh = xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h2_perm_params;
      f32_tanh_config.element_tile = 40;
    } else if (hardware_config->use_x86_avx) {
      f32_tanh_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut4_p4h2ts_perm_div_u48;
      f32_tanh_config.init.f32_tanh = xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h2_perm_params;
      f32_tanh_config.element_tile = 48;
    } else if (hardware_config->use_x86_sse4_1) {
      f32_tanh_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_lut8_p4h3ts_div_u20;
      f32_tanh_config.init.f32_tanh = xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params;
      f32_tanh_config.element_tile = 20;
    } else {
      f32_tanh_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_lut8_p4h3ts_div_u16;
      f32_tanh_config.init.f32_tanh = xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params;
      f32_tanh_config.element_tile = 16;
    }
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->is_x86) {
      f32_tanh_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_p6h5ts_div_nabs_pmax_u16;
      f32_tanh_config.init.f32_tanh = xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_p6h5_nabs_params;
      f32_tanh_config.element_tile = 16;
    } else {
      f32_tanh_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_p6h5ts_div_abs_min_u16;
      f32_tanh_config.init.f32_tanh = xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_p6h5_abs_params;
      f32_tanh_config.element_tile = 16;
    }
  #elif XNN_ARCH_WASM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->is_x86) {
      f32_tanh_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vtanh_ukernel__scalar_expm1minus_rr1_lut8_p4h3ts_div_u4;
      f32_tanh_config.init.f32_tanh = xnn_init_f32_tanh_scalar_expm1minus_rr1_lut8_p4h3_params;
      f32_tanh_config.element_tile = 4;
    } else {
      f32_tanh_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vtanh_ukernel__wasm_expm1minus_rr1_p6h5ts_div_u4;
      f32_tanh_config.init.f32_tanh = xnn_init_f32_tanh_scalar_expm1minus_rr1_p6h5_params;
      f32_tanh_config.element_tile = 4;
    }
  #elif XNN_ARCH_RISCV
    f32_tanh_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vtanh_ukernel__fma_expm1minus_rr1_lut8_p4h3ts_div_u4;
    f32_tanh_config.init.f32_tanh = xnn_init_f32_tanh_scalar_expm1minus_rr1_lut8_p4h3_params;
    f32_tanh_config.element_tile = 4;
  #elif XNN_ARCH_PPC64
    f32_tanh_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vtanh_ukernel__scalar_expm1minus_rr1_lut8_p4h3ts_div_u4;
    f32_tanh_config.init.f32_tanh = xnn_init_f32_tanh_scalar_expm1minus_rr1_lut8_p4h3_params;
    f32_tanh_config.element_tile = 4;
  #endif
}

static void init_f32_to_f16_cvt_config(void) {
  #if XNN_ARCH_ARM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon) {
      if (hardware_config->use_arm_neon_fp16) {
        f32_to_f16_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_f16_vcvt_ukernel__neonfp16_u16;
        f32_to_f16_cvt_config.element_tile = 16;
      } else {
        f32_to_f16_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_f16_vcvt_ukernel__neon_u8;
        f32_to_f16_cvt_config.init.f32_f16_cvt = xnn_init_f32_f16_cvt_neon_params;
        f32_to_f16_cvt_config.element_tile = 8;
      }
    } else if (!XNN_PLATFORM_MOBILE) {
      f32_to_f16_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_f16_vcvt_ukernel__scalar_fabsf_u2;
      f32_to_f16_cvt_config.init.f32_f16_cvt = xnn_init_f32_f16_cvt_scalar_fabsf_params;
      f32_to_f16_cvt_config.element_tile = 2;
    }
  #elif XNN_ARCH_ARM64
    f32_to_f16_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_f16_vcvt_ukernel__neonfp16_u16;
    f32_to_f16_cvt_config.element_tile = 16;
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_x86_avx512skx) {
      f32_to_f16_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_f16_vcvt_ukernel__avx512skx_u16;
      f32_to_f16_cvt_config.element_tile = 16;
    } else if (hardware_config->use_x86_f16c) {
      f32_to_f16_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_f16_vcvt_ukernel__f16c_u16;
      f32_to_f16_cvt_config.init.f32_f16_cvt = xnn_init_f32_f16_cvt_f16c_params;
      f32_to_f16_cvt_config.element_tile = 16;
    } else if (hardware_config->use_x86_avx) {
      f32_to_f16_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_f16_vcvt_ukernel__avx_u24;
      f32_to_f16_cvt_config.init.f32_f16_cvt = xnn_init_f32_f16_cvt_sse2_params;
      f32_to_f16_cvt_config.element_tile = 24;
    } else if (hardware_config->use_x86_sse4_1) {
      f32_to_f16_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_f16_vcvt_ukernel__sse41_u8;
      f32_to_f16_cvt_config.init.f32_f16_cvt = xnn_init_f32_f16_cvt_sse2_params;
      f32_to_f16_cvt_config.element_tile = 8;
    } else {
      f32_to_f16_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_f16_vcvt_ukernel__sse2_u16;
      f32_to_f16_cvt_config.init.f32_f16_cvt = xnn_init_f32_f16_cvt_sse2_params;
      f32_to_f16_cvt_config.element_tile = 16;
    }
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    #if XNN_ARCH_WASMRELAXEDSIMD
      f32_to_f16_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_f16_vcvt_ukernel__wasmrelaxedsimd_u24;
      f32_to_f16_cvt_config.init.f32_f16_cvt = xnn_init_f32_f16_cvt_wasmsimd_params;
      f32_to_f16_cvt_config.element_tile = 24;
    #else
      f32_to_f16_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_f16_vcvt_ukernel__wasmsimd_u24;
      f32_to_f16_cvt_config.init.f32_f16_cvt = xnn_init_f32_f16_cvt_wasmsimd_params;
      f32_to_f16_cvt_config.element_tile = 24;
    #endif
  #elif XNN_ARCH_WASM
    f32_to_f16_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_f16_vcvt_ukernel__scalar_bitcast_u4;
    f32_to_f16_cvt_config.init.f32_f16_cvt = xnn_init_f32_f16_cvt_scalar_bitcast_params;
    f32_to_f16_cvt_config.element_tile = 4;
  #elif XNN_ARCH_RISCV
    f32_to_f16_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_f16_vcvt_ukernel__scalar_fabsf_u2;
    f32_to_f16_cvt_config.init.f32_f16_cvt = xnn_init_f32_f16_cvt_scalar_fabsf_params;
    f32_to_f16_cvt_config.element_tile = 2;
  #elif XNN_ARCH_PPC64
    f32_to_f16_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_f16_vcvt_ukernel__scalar_fabsf_u2;
    f32_to_f16_cvt_config.init.f32_f16_cvt = xnn_init_f32_f16_cvt_scalar_fabsf_params;
    f32_to_f16_cvt_config.element_tile = 2;
  #endif
}

static void init_f32_to_qs8_cvt_config(void) {
  #if XNN_ARCH_ARM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon) {
      if (hardware_config->use_arm_neon_v8) {
        f32_to_qs8_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_qs8_vcvt_ukernel__neonv8_u32;
        f32_to_qs8_cvt_config.init.f32_qs8_cvt = xnn_init_f32_qs8_cvt_neonv8_params;
        f32_to_qs8_cvt_config.element_tile = 32;
      } else {
        f32_to_qs8_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_qs8_vcvt_ukernel__neon_u32;
        f32_to_qs8_cvt_config.init.f32_qs8_cvt = xnn_init_f32_qs8_cvt_neon_params;
        f32_to_qs8_cvt_config.element_tile = 32;
      }
    } else if (!XNN_PLATFORM_MOBILE) {
      f32_to_qs8_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_qs8_vcvt_ukernel__scalar_imagic_u4;
      f32_to_qs8_cvt_config.init.f32_qs8_cvt = xnn_init_f32_qs8_cvt_scalar_imagic_params;
      f32_to_qs8_cvt_config.element_tile = 4;
    }
  #elif XNN_ARCH_ARM64
    f32_to_qs8_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_qs8_vcvt_ukernel__neonv8_u32;
    f32_to_qs8_cvt_config.init.f32_qs8_cvt = xnn_init_f32_qs8_cvt_neonv8_params;
    f32_to_qs8_cvt_config.element_tile = 32;
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_x86_avx512skx) {
      f32_to_qs8_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_qs8_vcvt_ukernel__avx512skx_u128;
      f32_to_qs8_cvt_config.init.f32_qs8_cvt = xnn_init_f32_qs8_cvt_avx512_params;
      f32_to_qs8_cvt_config.element_tile = 128;
    } else if (hardware_config->use_x86_avx2) {
      f32_to_qs8_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_qs8_vcvt_ukernel__avx2_u64;
      f32_to_qs8_cvt_config.init.f32_qs8_cvt = xnn_init_f32_qs8_cvt_avx2_params;
      f32_to_qs8_cvt_config.element_tile = 64;
    } else if (hardware_config->use_x86_avx) {
      f32_to_qs8_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_qs8_vcvt_ukernel__avx_u32;
      f32_to_qs8_cvt_config.init.f32_qs8_cvt = xnn_init_f32_qs8_cvt_avx_params;
      f32_to_qs8_cvt_config.element_tile = 32;
    } else if (hardware_config->use_x86_sse4_1) {
      f32_to_qs8_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_qs8_vcvt_ukernel__sse41_u32;
      f32_to_qs8_cvt_config.init.f32_qs8_cvt = xnn_init_f32_qs8_cvt_sse4_params;
      f32_to_qs8_cvt_config.element_tile = 32;
    } else {
      f32_to_qs8_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_qs8_vcvt_ukernel__sse2_u32;
      f32_to_qs8_cvt_config.init.f32_qs8_cvt = xnn_init_f32_qs8_cvt_sse2_params;
      f32_to_qs8_cvt_config.element_tile = 32;
      }
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    f32_to_qs8_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_qs8_vcvt_ukernel__wasmsimd_magic_u32;
    f32_to_qs8_cvt_config.init.f32_qs8_cvt = xnn_init_f32_qs8_cvt_wasmsimd_magic_params;
    f32_to_qs8_cvt_config.element_tile = 32;
  #elif XNN_ARCH_WASM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->is_x86) {
      f32_to_qs8_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_qu8_vcvt_ukernel__scalar_imagic_u1;
      f32_to_qs8_cvt_config.init.f32_qs8_cvt = xnn_init_f32_qs8_cvt_scalar_imagic_params;
      f32_to_qs8_cvt_config.element_tile = 1;
    } else {
      f32_to_qs8_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_qu8_vcvt_ukernel__wasm_fmagic_u4;
      f32_to_qs8_cvt_config.init.f32_qs8_cvt = xnn_init_f32_qs8_cvt_scalar_fmagic_params;
      f32_to_qs8_cvt_config.element_tile = 4;
    }
  #elif XNN_ARCH_RISCV
    f32_to_qs8_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_qs8_vcvt_ukernel__scalar_lrintf_u4;
    f32_to_qs8_cvt_config.init.f32_qs8_cvt = xnn_init_f32_qs8_cvt_scalar_lrintf_params;
    f32_to_qs8_cvt_config.element_tile = 4;
  #elif XNN_ARCH_PPC64
    f32_to_qs8_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_qs8_vcvt_ukernel__scalar_lrintf_u4;
    f32_to_qs8_cvt_config.init.f32_qs8_cvt = xnn_init_f32_qs8_cvt_scalar_lrintf_params;
    f32_to_qs8_cvt_config.element_tile = 4;
  #endif
}

static void init_f32_to_qu8_cvt_config(void) {
  #if XNN_ARCH_ARM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon) {
      if (hardware_config->use_arm_neon_v8) {
        f32_to_qu8_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_qu8_vcvt_ukernel__neonv8_u32;
        f32_to_qu8_cvt_config.init.f32_qu8_cvt = xnn_init_f32_qu8_cvt_neonv8_params;
        f32_to_qu8_cvt_config.element_tile = 32;
      } else {
        f32_to_qu8_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_qu8_vcvt_ukernel__neon_u32;
        f32_to_qu8_cvt_config.init.f32_qu8_cvt = xnn_init_f32_qu8_cvt_neon_params;
        f32_to_qu8_cvt_config.element_tile = 32;
      }
    } else if (!XNN_PLATFORM_MOBILE) {
      f32_to_qu8_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_qu8_vcvt_ukernel__scalar_imagic_u4;
      f32_to_qu8_cvt_config.init.f32_qu8_cvt = xnn_init_f32_qu8_cvt_scalar_imagic_params;
      f32_to_qu8_cvt_config.element_tile = 4;
    }
  #elif XNN_ARCH_ARM64
    f32_to_qu8_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_qu8_vcvt_ukernel__neonv8_u32;
    f32_to_qu8_cvt_config.init.f32_qu8_cvt = xnn_init_f32_qu8_cvt_neonv8_params;
    f32_to_qu8_cvt_config.element_tile = 32;
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_x86_avx512skx) {
      f32_to_qu8_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_qu8_vcvt_ukernel__avx512skx_u128;
      f32_to_qu8_cvt_config.init.f32_qu8_cvt = xnn_init_f32_qu8_cvt_avx512_params;
      f32_to_qu8_cvt_config.element_tile = 128;
    } else if (hardware_config->use_x86_avx2) {
      f32_to_qu8_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_qu8_vcvt_ukernel__avx2_u64;
      f32_to_qu8_cvt_config.init.f32_qu8_cvt = xnn_init_f32_qu8_cvt_avx2_params;
      f32_to_qu8_cvt_config.element_tile = 64;
    } else if (hardware_config->use_x86_avx) {
      f32_to_qu8_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_qu8_vcvt_ukernel__avx_u32;
      f32_to_qu8_cvt_config.init.f32_qu8_cvt = xnn_init_f32_qu8_cvt_avx_params;
      f32_to_qu8_cvt_config.element_tile = 32;
    } else {
      f32_to_qu8_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_qu8_vcvt_ukernel__sse2_u32;
      f32_to_qu8_cvt_config.init.f32_qu8_cvt = xnn_init_f32_qu8_cvt_sse2_params;
      f32_to_qu8_cvt_config.element_tile = 32;
    }
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    f32_to_qu8_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_qu8_vcvt_ukernel__wasmsimd_magic_u32;
    f32_to_qu8_cvt_config.init.f32_qu8_cvt = xnn_init_f32_qu8_cvt_wasmsimd_magic_params;
    f32_to_qu8_cvt_config.element_tile = 32;
  #elif XNN_ARCH_WASM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->is_x86) {
      f32_to_qu8_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_qu8_vcvt_ukernel__scalar_imagic_u1;
      f32_to_qu8_cvt_config.init.f32_qu8_cvt = xnn_init_f32_qu8_cvt_scalar_imagic_params;
      f32_to_qu8_cvt_config.element_tile = 1;
    } else {
      f32_to_qu8_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_qu8_vcvt_ukernel__wasm_fmagic_u4;
      f32_to_qu8_cvt_config.init.f32_qu8_cvt = xnn_init_f32_qu8_cvt_scalar_fmagic_params;
      f32_to_qu8_cvt_config.element_tile = 4;
    }
  #elif XNN_ARCH_RISCV
    f32_to_qu8_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_qu8_vcvt_ukernel__scalar_lrintf_u4;
    f32_to_qu8_cvt_config.init.f32_qu8_cvt = xnn_init_f32_qu8_cvt_scalar_lrintf_params;
    f32_to_qu8_cvt_config.element_tile = 4;
  #elif XNN_ARCH_PPC64
    f32_to_qu8_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_qu8_vcvt_ukernel__scalar_lrintf_u4;
    f32_to_qu8_cvt_config.init.f32_qu8_cvt = xnn_init_f32_qu8_cvt_scalar_lrintf_params;
    f32_to_qu8_cvt_config.element_tile = 4;
  #endif
}

static void init_qs8_cvt_config(void) {
  #if XNN_ARCH_ARM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon_v8) {
      qs8_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qs8_vcvt_ukernel__neon_u32;
      qs8_cvt_config.init.qs8_cvt = xnn_init_qs8_cvt_neon_params;
      qs8_cvt_config.element_tile = 32;
    } else if (!XNN_PLATFORM_MOBILE) {
      qs8_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qs8_vcvt_ukernel__armsimd32_u8;
      qs8_cvt_config.init.qs8_cvt = xnn_init_qs8_cvt_armsimd32_params;
      qs8_cvt_config.element_tile = 8;
    }
  #elif XNN_ARCH_ARM64
    qs8_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qs8_vcvt_ukernel__neon_u32;
    qs8_cvt_config.init.qs8_cvt = xnn_init_qs8_cvt_neon_params;
    qs8_cvt_config.element_tile = 32;
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_x86_avx2) {
      qs8_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qs8_vcvt_ukernel__avx2_u32;
      qs8_cvt_config.init.qs8_cvt = xnn_init_qs8_cvt_avx2_params;
      qs8_cvt_config.element_tile = 32;
    } else if (hardware_config->use_x86_avx) {
      qs8_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qs8_vcvt_ukernel__avx_u32;
      qs8_cvt_config.init.qs8_cvt = xnn_init_qs8_cvt_ssse3_params;
      qs8_cvt_config.element_tile = 32;
    } else if (hardware_config->use_x86_sse4_1) {
      qs8_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qs8_vcvt_ukernel__sse41_u32;
      qs8_cvt_config.init.qs8_cvt = xnn_init_qs8_cvt_ssse3_params;
      qs8_cvt_config.element_tile = 32;
    } else if (hardware_config->use_x86_ssse3) {
      qs8_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qs8_vcvt_ukernel__ssse3_u32;
      qs8_cvt_config.init.qs8_cvt = xnn_init_qs8_cvt_ssse3_params;
      qs8_cvt_config.element_tile = 32;
    } else {
      qs8_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qs8_vcvt_ukernel__sse2_u32;
      qs8_cvt_config.init.qs8_cvt = xnn_init_qs8_cvt_sse2_params;
      qs8_cvt_config.element_tile = 32;
    }
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    #if XNN_ARCH_WASMRELAXEDSIMD
      qs8_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qs8_vcvt_ukernel__wasmrelaxedsimd_u32;
      qs8_cvt_config.init.qs8_cvt = xnn_init_qs8_cvt_wasmsimd_params;
      qs8_cvt_config.element_tile = 32;
    #else
      qs8_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qs8_vcvt_ukernel__wasmsimd_u16;
      qs8_cvt_config.init.qs8_cvt = xnn_init_qs8_cvt_wasmsimd_params;
      qs8_cvt_config.element_tile = 16;
    #endif
  #elif XNN_ARCH_WASM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->is_x86) {
      qs8_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qs8_vcvt_ukernel__scalar_u1;
      qs8_cvt_config.init.qs8_cvt = xnn_init_qs8_cvt_scalar_params;
      qs8_cvt_config.element_tile = 1;
    } else {
      qs8_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qs8_vcvt_ukernel__scalar_u4;
      qs8_cvt_config.init.qs8_cvt = xnn_init_qs8_cvt_scalar_params;
      qs8_cvt_config.element_tile = 4;
    }
  #elif XNN_ARCH_RISCV
    qs8_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qs8_vcvt_ukernel__scalar_u4;
    qs8_cvt_config.init.qs8_cvt = xnn_init_qs8_cvt_scalar_params;
    qs8_cvt_config.element_tile = 4;
  #elif XNN_ARCH_PPC64
    qs8_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qs8_vcvt_ukernel__scalar_u4;
    qs8_cvt_config.init.qs8_cvt = xnn_init_qs8_cvt_scalar_params;
    qs8_cvt_config.element_tile = 4;
  #endif
}

static void init_qs16_to_qs8_cvt_config(void) {
  #if XNN_ARCH_ARM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon) {
      qs16_to_qs8_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qs16_qs8_vcvt_ukernel__asm_aarch32_neon_u16;
      qs16_to_qs8_cvt_config.init.qs16_qs8_cvt = xnn_init_qs16_qs8_cvt_neon_params;
      qs16_to_qs8_cvt_config.element_tile = 16;
    } else if (!XNN_PLATFORM_MOBILE) {
      qs16_to_qs8_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qs16_qs8_vcvt_ukernel__scalar_u4;
      qs16_to_qs8_cvt_config.init.qs16_qs8_cvt = xnn_init_qs16_qs8_cvt_scalar_params;
      qs16_to_qs8_cvt_config.element_tile = 4;
    }
  #elif XNN_ARCH_ARM64
    qs16_to_qs8_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qs16_qs8_vcvt_ukernel__neon_u32;
    qs16_to_qs8_cvt_config.init.qs16_qs8_cvt = xnn_init_qs16_qs8_cvt_neon_params;
    qs16_to_qs8_cvt_config.element_tile = 32;
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_x86_avx) {
      qs16_to_qs8_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qs16_qs8_vcvt_ukernel__avx_u16;
      qs16_to_qs8_cvt_config.init.qs16_qs8_cvt = xnn_init_qs16_qs8_cvt_sse4_params;
      qs16_to_qs8_cvt_config.element_tile = 16;
    } else if (hardware_config->use_x86_sse4_1) {
      qs16_to_qs8_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qs16_qs8_vcvt_ukernel__sse41_u16;
      qs16_to_qs8_cvt_config.init.qs16_qs8_cvt = xnn_init_qs16_qs8_cvt_sse4_params;
      qs16_to_qs8_cvt_config.element_tile = 16;
    } else if (hardware_config->use_x86_ssse3) {
      qs16_to_qs8_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qs16_qs8_vcvt_ukernel__ssse3_u16;
      qs16_to_qs8_cvt_config.init.qs16_qs8_cvt = xnn_init_qs16_qs8_cvt_ssse3_params;
      qs16_to_qs8_cvt_config.element_tile = 16;
    } else {
      qs16_to_qs8_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qs16_qs8_vcvt_ukernel__sse2_u16;
      qs16_to_qs8_cvt_config.init.qs16_qs8_cvt = xnn_init_qs16_qs8_cvt_sse2_params;
      qs16_to_qs8_cvt_config.element_tile = 16;
    }
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    qs16_to_qs8_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qs16_qs8_vcvt_ukernel__wasmsimd_u16;
    qs16_to_qs8_cvt_config.init.qs16_qs8_cvt = xnn_init_qs16_qs8_cvt_wasmsimd_params;
    qs16_to_qs8_cvt_config.element_tile = 16;
  #else
    qs16_to_qs8_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qs16_qs8_vcvt_ukernel__scalar_u4;
    qs16_to_qs8_cvt_config.init.qs16_qs8_cvt = xnn_init_qs16_qs8_cvt_scalar_params;
    qs16_to_qs8_cvt_config.element_tile = 4;
  #endif
}

static void init_qs8_lrelu_config(void) {
  #if XNN_ARCH_ARM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon) {
      qs8_lrelu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qs8_vlrelu_ukernel__neon_u32;
      qs8_lrelu_config.init.qs8_lrelu = xnn_init_qs8_lrelu_neon_params;
      qs8_lrelu_config.element_tile = 32;
    } else if (!XNN_PLATFORM_MOBILE) {
      qs8_lrelu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qs8_vlrelu_ukernel__armsimd32_u4;
      qs8_lrelu_config.init.qs8_lrelu = xnn_init_qs8_lrelu_armsimd32_params;
      qs8_lrelu_config.element_tile = 4;
    }
  #elif XNN_ARCH_ARM64
    qs8_lrelu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qs8_vlrelu_ukernel__neon_u32;
    qs8_lrelu_config.init.qs8_lrelu = xnn_init_qs8_lrelu_neon_params;
    qs8_lrelu_config.element_tile = 32;
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_x86_avx2) {
      qs8_lrelu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qs8_vlrelu_ukernel__avx2_u32;
      qs8_lrelu_config.init.qs8_lrelu = xnn_init_qs8_lrelu_avx2_params;
      qs8_lrelu_config.element_tile = 32;
    } else if (hardware_config->use_x86_avx) {
      qs8_lrelu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qs8_vlrelu_ukernel__avx_u32;
      qs8_lrelu_config.init.qs8_lrelu = xnn_init_qs8_lrelu_avx_params;
      qs8_lrelu_config.element_tile = 32;
    } else if (hardware_config->use_x86_sse4_1) {
      qs8_lrelu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qs8_vlrelu_ukernel__sse41_u32;
      qs8_lrelu_config.init.qs8_lrelu = xnn_init_qs8_lrelu_sse2_params;
      qs8_lrelu_config.element_tile = 32;
    } else if (hardware_config->use_x86_sse4_1) {
      qs8_lrelu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qs8_vlrelu_ukernel__ssse3_u32;
      qs8_lrelu_config.init.qs8_lrelu = xnn_init_qs8_lrelu_sse2_params;
      qs8_lrelu_config.element_tile = 32;
    } else {
      qs8_lrelu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qs8_vlrelu_ukernel__sse2_u32;
      qs8_lrelu_config.init.qs8_lrelu = xnn_init_qs8_lrelu_sse2_params;
      qs8_lrelu_config.element_tile = 32;
    }
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    #if XNN_ARCH_WASMRELAXEDSIMD
      if (hardware_config->is_x86) {
        qs8_lrelu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qs8_vlrelu_ukernel__wasmrelaxedsimd_x86_u32;
        qs8_lrelu_config.init.qs8_lrelu = xnn_init_qs8_lrelu_wasmsimd_x86_params;
        qs8_lrelu_config.element_tile = 32;
      } else {
        qs8_lrelu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qs8_vlrelu_ukernel__wasmrelaxedsimd_arm_u32;
        qs8_lrelu_config.init.qs8_lrelu = xnn_init_qs8_lrelu_wasmsimd_arm_params;
        qs8_lrelu_config.element_tile = 32;
      }
    #else
      if (hardware_config->is_x86) {
        qs8_lrelu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qs8_vlrelu_ukernel__wasmsimd_x86_u16;
        qs8_lrelu_config.init.qs8_lrelu = xnn_init_qs8_lrelu_wasmsimd_x86_params;
        qs8_lrelu_config.element_tile = 16;
      } else {
        qs8_lrelu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qs8_vlrelu_ukernel__wasmsimd_arm_u32;
        qs8_lrelu_config.init.qs8_lrelu = xnn_init_qs8_lrelu_wasmsimd_arm_params;
        qs8_lrelu_config.element_tile = 32;
      }
    #endif
  #elif XNN_ARCH_WASM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->is_x86) {
      qs8_lrelu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qs8_vlrelu_ukernel__scalar_select_u4;
      qs8_lrelu_config.init.qs8_lrelu = xnn_init_qs8_lrelu_scalar_select_params;
      qs8_lrelu_config.element_tile = 4;
    } else {
      qs8_lrelu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qs8_vlrelu_ukernel__scalar_andxor_u4;
      qs8_lrelu_config.init.qs8_lrelu = xnn_init_qs8_lrelu_scalar_andxor_params;
      qs8_lrelu_config.element_tile = 4;
    }
  #elif XNN_ARCH_RISCV
    qs8_lrelu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qs8_vlrelu_ukernel__scalar_andxor_u4;
    qs8_lrelu_config.init.qs8_lrelu = xnn_init_qs8_lrelu_scalar_andxor_params;
    qs8_lrelu_config.element_tile = 4;
  #elif XNN_ARCH_PPC64
    qs8_lrelu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qs8_vlrelu_ukernel__scalar_andxor_u4;
    qs8_lrelu_config.init.qs8_lrelu = xnn_init_qs8_lrelu_scalar_andxor_params;
    qs8_lrelu_config.element_tile = 4;
  #endif
}

static void init_qs8_to_f16_cvt_config(void) {
  #if XNN_ARCH_ARM || XNN_ARCH_ARM64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon_fp16_arith) {
      qs8_to_f16_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qs8_f16_vcvt_ukernel__neonfp16arith_u32;
      qs8_to_f16_cvt_config.init.qs8_f16_cvt = xnn_init_qs8_f16_cvt_neonfp16arith_params;
      qs8_to_f16_cvt_config.element_tile = 32;
    }
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_x86_avx2) {
      qs8_to_f16_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qs8_f16_vcvt_ukernel__avx2_u16;
      qs8_to_f16_cvt_config.init.qs8_f16_cvt = xnn_init_qs8_f16_cvt_avx_params;
      qs8_to_f16_cvt_config.element_tile = 16;
    }
  #endif
}

static void init_qs8_to_f32_cvt_config(void) {
  #if XNN_ARCH_ARM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon) {
      qs8_to_f32_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qs8_f32_vcvt_ukernel__neon_u32;
      qs8_to_f32_cvt_config.init.qs8_f32_cvt = xnn_init_qs8_f32_cvt_neon_params;
      qs8_to_f32_cvt_config.element_tile = 32;
    } else if (!XNN_PLATFORM_MOBILE) {
      qs8_to_f32_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qs8_f32_vcvt_ukernel__scalar_u4;
      qs8_to_f32_cvt_config.init.qs8_f32_cvt = xnn_init_qs8_f32_cvt_scalar_params;
      qs8_to_f32_cvt_config.element_tile = 4;
    }
  #elif XNN_ARCH_ARM64
    qs8_to_f32_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qs8_f32_vcvt_ukernel__neon_u32;
    qs8_to_f32_cvt_config.init.qs8_f32_cvt = xnn_init_qs8_f32_cvt_neon_params;
    qs8_to_f32_cvt_config.element_tile = 32;
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_x86_avx512skx) {
      qs8_to_f32_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qs8_f32_vcvt_ukernel__avx512skx_u32;
      qs8_to_f32_cvt_config.init.qs8_f32_cvt = xnn_init_qs8_f32_cvt_avx512_params;
      qs8_to_f32_cvt_config.element_tile = 32;
    } else if (hardware_config->use_x86_avx2) {
      qs8_to_f32_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qs8_f32_vcvt_ukernel__avx2_u16;
      qs8_to_f32_cvt_config.init.qs8_f32_cvt = xnn_init_qs8_f32_cvt_avx_params;
      qs8_to_f32_cvt_config.element_tile = 16;
    } else if (hardware_config->use_x86_avx) {
      qs8_to_f32_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qs8_f32_vcvt_ukernel__avx_u32;
      qs8_to_f32_cvt_config.init.qs8_f32_cvt = xnn_init_qs8_f32_cvt_avx_params;
      qs8_to_f32_cvt_config.element_tile = 32;
    } else if (hardware_config->use_x86_sse4_1) {
      qs8_to_f32_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qs8_f32_vcvt_ukernel__sse41_u16;
      qs8_to_f32_cvt_config.init.qs8_f32_cvt = xnn_init_qs8_f32_cvt_sse4_params;
      qs8_to_f32_cvt_config.element_tile = 16;
    } else {
      qs8_to_f32_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qs8_f32_vcvt_ukernel__sse2_u32;
      qs8_to_f32_cvt_config.init.qs8_f32_cvt = xnn_init_qs8_f32_cvt_sse2_params;
      qs8_to_f32_cvt_config.element_tile = 32;
    }
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    qs8_to_f32_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qs8_f32_vcvt_ukernel__wasmsimd_u32;
    qs8_to_f32_cvt_config.init.qs8_f32_cvt = xnn_init_qs8_f32_cvt_wasmsimd_params;
    qs8_to_f32_cvt_config.element_tile = 32;
  #elif XNN_ARCH_WASM
    qs8_to_f32_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qs8_f32_vcvt_ukernel__scalar_u1;
    qs8_to_f32_cvt_config.init.qs8_f32_cvt = xnn_init_qs8_f32_cvt_scalar_params;
    qs8_to_f32_cvt_config.element_tile = 1;
  #elif XNN_ARCH_RISCV
    qs8_to_f32_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qs8_f32_vcvt_ukernel__scalar_u4;
    qs8_to_f32_cvt_config.init.qs8_f32_cvt = xnn_init_qs8_f32_cvt_scalar_params;
    qs8_to_f32_cvt_config.element_tile = 4;
  #elif XNN_ARCH_PPC64
    qs8_to_f32_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qs8_f32_vcvt_ukernel__scalar_u4;
    qs8_to_f32_cvt_config.init.qs8_f32_cvt = xnn_init_qs8_f32_cvt_scalar_params;
    qs8_to_f32_cvt_config.element_tile = 4;
  #endif
}

static void init_qu8_cvt_config(void) {
  #if XNN_ARCH_ARM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon) {
      qu8_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qu8_vcvt_ukernel__neon_u32;
      qu8_cvt_config.init.qu8_cvt = xnn_init_qu8_cvt_neon_params;
      qu8_cvt_config.element_tile = 32;
    } else if (!XNN_PLATFORM_MOBILE) {
      qu8_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qu8_vcvt_ukernel__armsimd32_u8;
      qu8_cvt_config.init.qu8_cvt = xnn_init_qu8_cvt_armsimd32_params;
      qu8_cvt_config.element_tile = 8;
    }
  #elif XNN_ARCH_ARM64
    qu8_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qu8_vcvt_ukernel__neon_u32;
    qu8_cvt_config.init.qu8_cvt = xnn_init_qu8_cvt_neon_params;
    qu8_cvt_config.element_tile = 32;
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_x86_avx2) {
      qu8_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qu8_vcvt_ukernel__avx2_u32;
      qu8_cvt_config.init.qu8_cvt = xnn_init_qu8_cvt_avx2_params;
      qu8_cvt_config.element_tile = 32;
    } else if (hardware_config->use_x86_avx) {
      qu8_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qu8_vcvt_ukernel__avx_u32;
      qu8_cvt_config.init.qu8_cvt = xnn_init_qu8_cvt_ssse3_params;
      qu8_cvt_config.element_tile = 32;
    } else if (hardware_config->use_x86_sse4_1) {
      qu8_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qu8_vcvt_ukernel__sse41_u32;
      qu8_cvt_config.init.qu8_cvt = xnn_init_qu8_cvt_ssse3_params;
      qu8_cvt_config.element_tile = 32;
    } else if (hardware_config->use_x86_ssse3) {
      qu8_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qu8_vcvt_ukernel__ssse3_u32;
      qu8_cvt_config.init.qu8_cvt = xnn_init_qu8_cvt_ssse3_params;
      qu8_cvt_config.element_tile = 32;
    } else {
      qu8_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qu8_vcvt_ukernel__sse2_u32;
      qu8_cvt_config.init.qu8_cvt = xnn_init_qu8_cvt_sse2_params;
      qu8_cvt_config.element_tile = 32;
    }
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    #if XNN_ARCH_WASMRELAXEDSIMD
      qu8_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qu8_vcvt_ukernel__wasmrelaxedsimd_u32;
      qu8_cvt_config.init.qu8_cvt = xnn_init_qu8_cvt_wasmsimd_params;
      qu8_cvt_config.element_tile = 32;
    #else
      qu8_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qu8_vcvt_ukernel__wasmsimd_u16;
      qu8_cvt_config.init.qu8_cvt = xnn_init_qu8_cvt_wasmsimd_params;
      qu8_cvt_config.element_tile = 16;
    #endif
  #elif XNN_ARCH_WASM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->is_x86) {
      qu8_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qu8_vcvt_ukernel__scalar_u1;
      qu8_cvt_config.init.qu8_cvt = xnn_init_qu8_cvt_scalar_params;
      qu8_cvt_config.element_tile = 1;
    } else {
      qu8_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qu8_vcvt_ukernel__scalar_u4;
      qu8_cvt_config.init.qu8_cvt = xnn_init_qu8_cvt_scalar_params;
      qu8_cvt_config.element_tile = 4;
    }
  #elif XNN_ARCH_RISCV
    qu8_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qu8_vcvt_ukernel__scalar_u4;
    qu8_cvt_config.init.qu8_cvt = xnn_init_qu8_cvt_scalar_params;
    qu8_cvt_config.element_tile = 4;
  #elif XNN_ARCH_PPC64
    qu8_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qu8_vcvt_ukernel__scalar_u4;
    qu8_cvt_config.init.qu8_cvt = xnn_init_qu8_cvt_scalar_params;
    qu8_cvt_config.element_tile = 4;
  #endif
}

static void init_qu8_lrelu_config(void) {
  #if XNN_ARCH_ARM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon) {
      qu8_lrelu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qu8_vlrelu_ukernel__neon_u32;
      qu8_lrelu_config.init.qu8_lrelu = xnn_init_qu8_lrelu_neon_params;
      qu8_lrelu_config.element_tile = 32;
    } else if (!XNN_PLATFORM_MOBILE) {
      qu8_lrelu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qu8_vlrelu_ukernel__armsimd32_u4;
      qu8_lrelu_config.init.qu8_lrelu = xnn_init_qu8_lrelu_armsimd32_params;
      qu8_lrelu_config.element_tile = 4;
    }
  #elif XNN_ARCH_ARM64
    qu8_lrelu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qu8_vlrelu_ukernel__neon_u32;
    qu8_lrelu_config.init.qu8_lrelu = xnn_init_qu8_lrelu_neon_params;
    qu8_lrelu_config.element_tile = 32;
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_x86_avx2) {
      qu8_lrelu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qu8_vlrelu_ukernel__avx2_u32;
      qu8_lrelu_config.init.qu8_lrelu = xnn_init_qu8_lrelu_avx2_params;
      qu8_lrelu_config.element_tile = 32;
    } else if (hardware_config->use_x86_avx) {
      qu8_lrelu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qu8_vlrelu_ukernel__avx_u32;
      qu8_lrelu_config.init.qu8_lrelu = xnn_init_qu8_lrelu_avx_params;
      qu8_lrelu_config.element_tile = 32;
    } else if (hardware_config->use_x86_sse4_1) {
      qu8_lrelu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qu8_vlrelu_ukernel__sse41_u32;
      qu8_lrelu_config.init.qu8_lrelu = xnn_init_qu8_lrelu_sse2_params;
      qu8_lrelu_config.element_tile = 32;
    } else if (hardware_config->use_x86_sse4_1) {
      qu8_lrelu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qu8_vlrelu_ukernel__ssse3_u32;
      qu8_lrelu_config.init.qu8_lrelu = xnn_init_qu8_lrelu_sse2_params;
      qu8_lrelu_config.element_tile = 32;
    } else {
      qu8_lrelu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qu8_vlrelu_ukernel__sse2_u32;
      qu8_lrelu_config.init.qu8_lrelu = xnn_init_qu8_lrelu_sse2_params;
      qu8_lrelu_config.element_tile = 32;
    }
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    #if XNN_ARCH_WASMRELAXEDSIMD
      if (hardware_config->is_x86) {
        qu8_lrelu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qu8_vlrelu_ukernel__wasmrelaxedsimd_x86_u32;
        qu8_lrelu_config.init.qu8_lrelu = xnn_init_qu8_lrelu_wasmsimd_x86_params;
        qu8_lrelu_config.element_tile = 32;
      } else {
        qu8_lrelu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qu8_vlrelu_ukernel__wasmrelaxedsimd_arm_u32;
        qu8_lrelu_config.init.qu8_lrelu = xnn_init_qu8_lrelu_wasmsimd_arm_params;
        qu8_lrelu_config.element_tile = 32;
      }
    #else
      if (hardware_config->is_x86) {
        qu8_lrelu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qu8_vlrelu_ukernel__wasmsimd_x86_u16;
        qu8_lrelu_config.init.qu8_lrelu = xnn_init_qu8_lrelu_wasmsimd_x86_params;
        qu8_lrelu_config.element_tile = 16;
      } else {
        qu8_lrelu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qu8_vlrelu_ukernel__wasmsimd_arm_u32;
        qu8_lrelu_config.init.qu8_lrelu = xnn_init_qu8_lrelu_wasmsimd_arm_params;
        qu8_lrelu_config.element_tile = 32;
      }
    #endif
  #elif XNN_ARCH_WASM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->is_x86) {
      qu8_lrelu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qu8_vlrelu_ukernel__scalar_select_u4;
      qu8_lrelu_config.init.qu8_lrelu = xnn_init_qu8_lrelu_scalar_select_params;
      qu8_lrelu_config.element_tile = 4;
    } else {
      qu8_lrelu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qu8_vlrelu_ukernel__scalar_andxor_u4;
      qu8_lrelu_config.init.qu8_lrelu = xnn_init_qu8_lrelu_scalar_andxor_params;
      qu8_lrelu_config.element_tile = 4;
    }
  #elif XNN_ARCH_RISCV
    qu8_lrelu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qu8_vlrelu_ukernel__scalar_andxor_u4;
    qu8_lrelu_config.init.qu8_lrelu = xnn_init_qu8_lrelu_scalar_andxor_params;
    qu8_lrelu_config.element_tile = 4;
  #elif XNN_ARCH_PPC64
    qu8_lrelu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qu8_vlrelu_ukernel__scalar_andxor_u4;
    qu8_lrelu_config.init.qu8_lrelu = xnn_init_qu8_lrelu_scalar_andxor_params;
    qu8_lrelu_config.element_tile = 4;
  #endif
}

static void init_qu8_to_f32_cvt_config(void) {
  #if XNN_ARCH_ARM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon) {
      qu8_to_f32_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qu8_f32_vcvt_ukernel__neon_u32;
      qu8_to_f32_cvt_config.init.qu8_f32_cvt = xnn_init_qu8_f32_cvt_neon_params;
      qu8_to_f32_cvt_config.element_tile = 32;
    } else if (!XNN_PLATFORM_MOBILE) {
      qu8_to_f32_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qu8_f32_vcvt_ukernel__scalar_u4;
      qu8_to_f32_cvt_config.init.qu8_f32_cvt = xnn_init_qu8_f32_cvt_scalar_params;
      qu8_to_f32_cvt_config.element_tile = 4;
    }
  #elif XNN_ARCH_ARM64
    qu8_to_f32_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qu8_f32_vcvt_ukernel__neon_u32;
    qu8_to_f32_cvt_config.init.qu8_f32_cvt = xnn_init_qu8_f32_cvt_neon_params;
    qu8_to_f32_cvt_config.element_tile = 32;
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_x86_avx512skx) {
      qu8_to_f32_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qu8_f32_vcvt_ukernel__avx512skx_u32;
      qu8_to_f32_cvt_config.init.qu8_f32_cvt = xnn_init_qu8_f32_cvt_avx512_params;
      qu8_to_f32_cvt_config.element_tile = 32;
    } else if (hardware_config->use_x86_avx2) {
      qu8_to_f32_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qu8_f32_vcvt_ukernel__avx2_u16;
      qu8_to_f32_cvt_config.init.qu8_f32_cvt = xnn_init_qu8_f32_cvt_avx_params;
      qu8_to_f32_cvt_config.element_tile = 16;
    } else if (hardware_config->use_x86_avx) {
      qu8_to_f32_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qu8_f32_vcvt_ukernel__avx_u32;
      qu8_to_f32_cvt_config.init.qu8_f32_cvt = xnn_init_qu8_f32_cvt_avx_params;
      qu8_to_f32_cvt_config.element_tile = 32;
    } else if (hardware_config->use_x86_sse4_1) {
      qu8_to_f32_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qu8_f32_vcvt_ukernel__sse41_u16;
      qu8_to_f32_cvt_config.init.qu8_f32_cvt = xnn_init_qu8_f32_cvt_sse4_params;
      qu8_to_f32_cvt_config.element_tile = 16;
    } else {
      qu8_to_f32_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qu8_f32_vcvt_ukernel__sse2_u32;
      qu8_to_f32_cvt_config.init.qu8_f32_cvt = xnn_init_qu8_f32_cvt_sse2_params;
      qu8_to_f32_cvt_config.element_tile = 32;
    }
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    qu8_to_f32_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qu8_f32_vcvt_ukernel__wasmsimd_u32;
    qu8_to_f32_cvt_config.init.qu8_f32_cvt = xnn_init_qu8_f32_cvt_wasmsimd_params;
    qu8_to_f32_cvt_config.element_tile = 32;
  #elif XNN_ARCH_WASM
    qu8_to_f32_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qu8_f32_vcvt_ukernel__scalar_u1;
    qu8_to_f32_cvt_config.init.qu8_f32_cvt = xnn_init_qu8_f32_cvt_scalar_params;
    qu8_to_f32_cvt_config.element_tile = 1;
  #elif XNN_ARCH_RISCV
    qu8_to_f32_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qu8_f32_vcvt_ukernel__scalar_u4;
    qu8_to_f32_cvt_config.init.qu8_f32_cvt = xnn_init_qu8_f32_cvt_scalar_params;
    qu8_to_f32_cvt_config.element_tile = 4;
  #elif XNN_ARCH_PPC64
    qu8_to_f32_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qu8_f32_vcvt_ukernel__scalar_u4;
    qu8_to_f32_cvt_config.init.qu8_f32_cvt = xnn_init_qu8_f32_cvt_scalar_params;
    qu8_to_f32_cvt_config.element_tile = 4;
  #endif
}

static void init_s8_clamp_config(void) {
  #if XNN_ARCH_ARM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon) {
      s8_clamp_config.ukernel = (xnn_vunary_ukernel_fn) xnn_s8_vclamp_ukernel__neon_u64;
      s8_clamp_config.init.s8_minmax = xnn_init_s8_minmax_neon_params;
      s8_clamp_config.element_tile = 64;
    } else if (!XNN_PLATFORM_MOBILE) {
      s8_clamp_config.ukernel = (xnn_vunary_ukernel_fn) xnn_s8_vclamp_ukernel__scalar_u4;
      s8_clamp_config.init.s8_minmax = xnn_init_s8_minmax_scalar_params;
      s8_clamp_config.element_tile = 4;
    }
  #elif XNN_ARCH_ARM64
    s8_clamp_config.ukernel = (xnn_vunary_ukernel_fn) xnn_s8_vclamp_ukernel__neon_u64;
    s8_clamp_config.init.s8_minmax = xnn_init_s8_minmax_neon_params;
    s8_clamp_config.element_tile = 64;
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_x86_sse4_1) {
      s8_clamp_config.ukernel = (xnn_vunary_ukernel_fn) xnn_s8_vclamp_ukernel__sse41_u64;
      s8_clamp_config.init.s8_minmax = xnn_init_s8_minmax_sse4_params;
      s8_clamp_config.element_tile = 64;
    } else {
      s8_clamp_config.ukernel = (xnn_vunary_ukernel_fn) xnn_s8_vclamp_ukernel__sse2_u64;
      s8_clamp_config.init.s8_minmax = xnn_init_s8_minmax_sse2_params;
      s8_clamp_config.element_tile = 64;
    }
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    s8_clamp_config.ukernel = (xnn_vunary_ukernel_fn) xnn_s8_vclamp_ukernel__wasmsimd_u64;
    s8_clamp_config.init.s8_minmax = xnn_init_s8_minmax_wasmsimd_params;
    s8_clamp_config.element_tile = 64;
  #elif XNN_ARCH_WASM
    s8_clamp_config.ukernel = (xnn_vunary_ukernel_fn) xnn_s8_vclamp_ukernel__scalar_u4;
    s8_clamp_config.init.s8_minmax = xnn_init_s8_minmax_scalar_params;
    s8_clamp_config.element_tile = 4;
  #elif XNN_ARCH_RISCV
    s8_clamp_config.ukernel = (xnn_vunary_ukernel_fn) xnn_s8_vclamp_ukernel__scalar_u4;
    s8_clamp_config.init.s8_minmax = xnn_init_s8_minmax_scalar_params;
    s8_clamp_config.element_tile = 4;
  #elif XNN_ARCH_PPC64
    s8_clamp_config.ukernel = (xnn_vunary_ukernel_fn) xnn_s8_vclamp_ukernel__scalar_u4;
    s8_clamp_config.init.s8_minmax = xnn_init_s8_minmax_scalar_params;
    s8_clamp_config.element_tile = 4;
  #endif
}

static void init_u8_clamp_config(void) {
  #if XNN_ARCH_ARM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon) {
      u8_clamp_config.ukernel = (xnn_vunary_ukernel_fn) xnn_u8_vclamp_ukernel__neon_u64;
      u8_clamp_config.init.u8_minmax = xnn_init_u8_minmax_neon_params;
      u8_clamp_config.element_tile = 64;
    } else if (!XNN_PLATFORM_MOBILE) {
      u8_clamp_config.ukernel = (xnn_vunary_ukernel_fn) xnn_u8_vclamp_ukernel__scalar_u4;
      u8_clamp_config.init.u8_minmax = xnn_init_u8_minmax_scalar_params;
      u8_clamp_config.element_tile = 4;
    }
  #elif XNN_ARCH_ARM64
    u8_clamp_config.ukernel = (xnn_vunary_ukernel_fn) xnn_u8_vclamp_ukernel__neon_u64;
    u8_clamp_config.init.u8_minmax = xnn_init_u8_minmax_neon_params;
    u8_clamp_config.element_tile = 64;
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    u8_clamp_config.ukernel = (xnn_vunary_ukernel_fn) xnn_u8_vclamp_ukernel__sse2_u64;
    u8_clamp_config.init.u8_minmax = xnn_init_u8_minmax_sse2_params;
    u8_clamp_config.element_tile = 64;
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    u8_clamp_config.ukernel = (xnn_vunary_ukernel_fn) xnn_u8_vclamp_ukernel__wasmsimd_u64;
    u8_clamp_config.init.u8_minmax = xnn_init_u8_minmax_wasmsimd_params;
    u8_clamp_config.element_tile = 64;
  #elif XNN_ARCH_WASM
    u8_clamp_config.ukernel = (xnn_vunary_ukernel_fn) xnn_u8_vclamp_ukernel__scalar_u4;
    u8_clamp_config.init.u8_minmax = xnn_init_u8_minmax_scalar_params;
    u8_clamp_config.element_tile = 4;
  #elif XNN_ARCH_RISCV
    u8_clamp_config.ukernel = (xnn_vunary_ukernel_fn) xnn_u8_vclamp_ukernel__scalar_u4;
    u8_clamp_config.init.u8_minmax = xnn_init_u8_minmax_scalar_params;
    u8_clamp_config.element_tile = 4;
  #elif XNN_ARCH_PPC64
    u8_clamp_config.ukernel = (xnn_vunary_ukernel_fn) xnn_u8_vclamp_ukernel__scalar_u4;
    u8_clamp_config.init.u8_minmax = xnn_init_u8_minmax_scalar_params;
    u8_clamp_config.element_tile = 4;
  #endif
}

static void init_xx_copy_config(void) {
  #if XNN_ARCH_ARM
    xx_copy_config.ukernel = (xnn_vunary_ukernel_fn) xnn_xx_copy_ukernel__scalar_memcpy;
    xx_copy_config.element_tile = 1;
  #elif XNN_ARCH_ARM64
    xx_copy_config.ukernel = (xnn_vunary_ukernel_fn) xnn_xx_copy_ukernel__scalar_memcpy;
    xx_copy_config.element_tile = 1;
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    xx_copy_config.ukernel = (xnn_vunary_ukernel_fn) xnn_xx_copy_ukernel__scalar_memcpy;
    xx_copy_config.element_tile = 1;
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    xx_copy_config.ukernel = (xnn_vunary_ukernel_fn) xnn_xx_copy_ukernel__scalar_memcpy;
    xx_copy_config.element_tile = 1;
  #elif XNN_ARCH_WASM
    xx_copy_config.ukernel = (xnn_vunary_ukernel_fn) xnn_xx_copy_ukernel__scalar_memcpy;
    xx_copy_config.element_tile = 1;
  #elif XNN_ARCH_RISCV
    xx_copy_config.ukernel = (xnn_vunary_ukernel_fn) xnn_xx_copy_ukernel__scalar_memcpy;
    xx_copy_config.element_tile = 1;
  #elif XNN_ARCH_PPC64
    xx_copy_config.ukernel = (xnn_vunary_ukernel_fn) xnn_xx_copy_ukernel__scalar_memcpy;
    xx_copy_config.element_tile = 1;
  #endif
}


#if XNN_PLATFORM_WINDOWS
 static BOOL CALLBACK init_f16_abs_config_windows(PINIT_ONCE init_once, PVOID parameter, PVOID* context) {
    init_f16_abs_config();
    return TRUE;
  }

  static BOOL CALLBACK init_f16_clamp_config_windows(PINIT_ONCE init_once, PVOID parameter, PVOID* context) {
    init_f16_clamp_config();
    return TRUE;
  }

  static BOOL CALLBACK init_f16_elu_config_windows(PINIT_ONCE init_once, PVOID parameter, PVOID* context) {
    init_f16_elu_config();
    return TRUE;
  }

  static BOOL CALLBACK init_f16_hswish_config_windows(PINIT_ONCE init_once, PVOID parameter, PVOID* context) {
    init_f16_hswish_config();
    return TRUE;
  }

  static BOOL CALLBACK init_f16_lrelu_config_windows(PINIT_ONCE init_once, PVOID parameter, PVOID* context) {
    init_f16_lrelu_config();
    return TRUE;
  }

  static BOOL CALLBACK init_f16_neg_config_windows(PINIT_ONCE init_once, PVOID parameter, PVOID* context) {
    init_f16_neg_config();
    return TRUE;
  }

  static BOOL CALLBACK init_f16_rndd_config_windows(PINIT_ONCE init_once, PVOID parameter, PVOID* context) {
    init_f16_rndd_config();
    return TRUE;
  }

  static BOOL CALLBACK init_f16_rndne_config_windows(PINIT_ONCE init_once, PVOID parameter, PVOID* context) {
    init_f16_rndne_config();
    return TRUE;
  }

  static BOOL CALLBACK init_f16_rndu_config_windows(PINIT_ONCE init_once, PVOID parameter, PVOID* context) {
    init_f16_rndu_config();
    return TRUE;
  }

  static BOOL CALLBACK init_f16_rndz_config_windows(PINIT_ONCE init_once, PVOID parameter, PVOID* context) {
    init_f16_rndz_config();
    return TRUE;
  }

  static BOOL CALLBACK init_f16_sigmoid_config_windows(PINIT_ONCE init_once, PVOID parameter, PVOID* context) {
    init_f16_sigmoid_config();
    return TRUE;
  }

  static BOOL CALLBACK init_f16_sqr_config_windows(PINIT_ONCE init_once, PVOID parameter, PVOID* context) {
    init_f16_sqr_config();
    return TRUE;
  }

  static BOOL CALLBACK init_f16_sqrt_config_windows(PINIT_ONCE init_once, PVOID parameter, PVOID* context) {
    init_f16_sqrt_config();
    return TRUE;
  }

  static BOOL CALLBACK init_f16_tanh_config_windows(PINIT_ONCE init_once, PVOID parameter, PVOID* context) {
    init_f16_tanh_config();
    return TRUE;
  }

  static BOOL CALLBACK init_f32_abs_config_windows(PINIT_ONCE init_once, PVOID parameter, PVOID* context) {
    init_f32_abs_config();
    return TRUE;
  }

  static BOOL CALLBACK init_f32_clamp_config_windows(PINIT_ONCE init_once, PVOID parameter, PVOID* context) {
    init_f32_clamp_config();
    return TRUE;
  }

  static BOOL CALLBACK init_f32_elu_config_windows(PINIT_ONCE init_once, PVOID parameter, PVOID* context) {
    init_f32_elu_config();
    return TRUE;
  }

  static BOOL CALLBACK init_f32_hswish_config_windows(PINIT_ONCE init_once, PVOID parameter, PVOID* context) {
    init_f32_hswish_config();
    return TRUE;
  }

  static BOOL CALLBACK init_f32_lrelu_config_windows(PINIT_ONCE init_once, PVOID parameter, PVOID* context) {
    init_f32_lrelu_config();
    return TRUE;
  }

  static BOOL CALLBACK init_f32_neg_config_windows(PINIT_ONCE init_once, PVOID parameter, PVOID* context) {
    init_f32_neg_config();
    return TRUE;
  }

  static BOOL CALLBACK init_f32_relu_config_windows(PINIT_ONCE init_once, PVOID parameter, PVOID* context) {
    init_f32_relu_config();
    return TRUE;
  }

  static BOOL CALLBACK init_f32_rndd_config_windows(PINIT_ONCE init_once, PVOID parameter, PVOID* context) {
    init_f32_rndd_config();
    return TRUE;
  }

  static BOOL CALLBACK init_f32_rndne_config_windows(PINIT_ONCE init_once, PVOID parameter, PVOID* context) {
    init_f32_rndne_config();
    return TRUE;
  }

  static BOOL CALLBACK init_f32_rndu_config_windows(PINIT_ONCE init_once, PVOID parameter, PVOID* context) {
    init_f32_rndu_config();
    return TRUE;
  }

  static BOOL CALLBACK init_f32_rndz_config_windows(PINIT_ONCE init_once, PVOID parameter, PVOID* context) {
    init_f32_rndz_config();
    return TRUE;
  }

  static BOOL CALLBACK init_f32_rsqrt_config_windows(PINIT_ONCE init_once,
                                                     PVOID parameter,
                                                     PVOID* context) {
    init_f32_rsqrt_config();
    return TRUE;
  }

  static BOOL CALLBACK init_f32_sigmoid_config_windows(PINIT_ONCE init_once, PVOID parameter, PVOID* context) {
    init_f32_sigmoid_config();
    return TRUE;
  }

  static BOOL CALLBACK init_f32_sqr_config_windows(PINIT_ONCE init_once, PVOID parameter, PVOID* context) {
    init_f32_sqr_config();
    return TRUE;
  }

  static BOOL CALLBACK init_f32_sqrt_config_windows(PINIT_ONCE init_once, PVOID parameter, PVOID* context) {
    init_f32_sqrt_config();
    return TRUE;
  }

  static BOOL CALLBACK init_f32_tanh_config_windows(PINIT_ONCE init_once, PVOID parameter, PVOID* context) {
    init_f32_tanh_config();
    return TRUE;
  }

  static BOOL CALLBACK init_qs8_lrelu_config_windows(PINIT_ONCE init_once, PVOID parameter, PVOID* context) {
    init_qs8_lrelu_config();
    return TRUE;
  }

  static BOOL CALLBACK init_qu8_lrelu_config_windows(PINIT_ONCE init_once, PVOID parameter, PVOID* context) {
    init_qu8_lrelu_config();
    return TRUE;
  }

  static BOOL CALLBACK init_s8_clamp_config_windows(PINIT_ONCE init_once, PVOID parameter, PVOID* context) {
    init_s8_clamp_config();
    return TRUE;
  }

  static BOOL CALLBACK init_u8_clamp_config_windows(PINIT_ONCE init_once, PVOID parameter, PVOID* context) {
    init_u8_clamp_config();
    return TRUE;
  }

  static BOOL CALLBACK init_f16_to_f32_cvt_config_windows(PINIT_ONCE init_once, PVOID parameter, PVOID* context) {
    init_f16_to_f32_cvt_config();
    return TRUE;
  }

  static BOOL CALLBACK init_f16_to_qs8_cvt_config_windows(PINIT_ONCE init_once, PVOID parameter, PVOID* context) {
    init_f16_to_qs8_cvt_config();
    return TRUE;
  }

  static BOOL CALLBACK init_f32_to_f16_cvt_config_windows(PINIT_ONCE init_once, PVOID parameter, PVOID* context) {
    init_f32_to_f16_cvt_config();
    return TRUE;
  }

  static BOOL CALLBACK init_f32_to_qs8_cvt_config_windows(PINIT_ONCE init_once, PVOID parameter, PVOID* context) {
    init_f32_to_qs8_cvt_config();
    return TRUE;
  }

  static BOOL CALLBACK init_f32_to_qu8_cvt_config_windows(PINIT_ONCE init_once, PVOID parameter, PVOID* context) {
    init_f32_to_qu8_cvt_config();
    return TRUE;
  }

  static BOOL CALLBACK init_qs8_cvt_config_windows(PINIT_ONCE init_once, PVOID parameter, PVOID* context) {
    init_qs8_cvt_config();
    return TRUE;
  }

  static BOOL CALLBACK init_qs8_to_f16_cvt_config_windows(PINIT_ONCE init_once, PVOID parameter, PVOID* context) {
    init_qs8_to_f16_cvt_config();
    return TRUE;
  }

  static BOOL CALLBACK init_qs8_to_f32_cvt_config_windows(PINIT_ONCE init_once, PVOID parameter, PVOID* context) {
    init_qs8_to_f32_cvt_config();
    return TRUE;
  }

  static BOOL CALLBACK init_qs16_to_qs8_cvt_config_windows(PINIT_ONCE init_once, PVOID parameter, PVOID* context) {
    init_qs16_to_qs8_cvt_config();
    return TRUE;
  }

  static BOOL CALLBACK init_qu8_cvt_config_windows(PINIT_ONCE init_once, PVOID parameter, PVOID* context) {
    init_qu8_cvt_config();
    return TRUE;
  }

  static BOOL CALLBACK init_qu8_to_f32_cvt_config_windows(PINIT_ONCE init_once, PVOID parameter, PVOID* context) {
    init_qu8_to_f32_cvt_config();
    return TRUE;
  }

  static BOOL CALLBACK init_xx_copy_config_windows(PINIT_ONCE init_once, PVOID parameter, PVOID* context) {
    init_xx_copy_config();
    return TRUE;
  }
#endif


const struct xnn_unary_elementwise_config* xnn_init_f16_abs_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL || !xnn_is_f16_compatible_config(hardware_config)) {
    return NULL;
  }
  #if XNN_PLATFORM_WINDOWS
    InitOnceExecuteOnce(&init_guard_f16_abs, &init_f16_abs_config_windows, NULL, NULL);
  #else
    pthread_once(&init_guard_f16_abs, &init_f16_abs_config);
  #endif
  return &f16_abs_config;
}

const struct xnn_unary_elementwise_config* xnn_init_f16_clamp_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL || !xnn_is_f16_compatible_config(hardware_config)) {
    return NULL;
  }
  #if XNN_PLATFORM_WINDOWS
    InitOnceExecuteOnce(&init_guard_f16_clamp, &init_f16_clamp_config_windows, NULL, NULL);
  #else
    pthread_once(&init_guard_f16_clamp, &init_f16_clamp_config);
  #endif
  return &f16_clamp_config;
}

const struct xnn_unary_elementwise_config* xnn_init_f16_elu_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL || !xnn_is_f16_compatible_config(hardware_config)) {
    return NULL;
  }
  #if XNN_PLATFORM_WINDOWS
    InitOnceExecuteOnce(&init_guard_f16_elu, &init_f16_elu_config_windows, NULL, NULL);
  #else
    pthread_once(&init_guard_f16_elu, &init_f16_elu_config);
  #endif
  return &f16_elu_config;
}

const struct xnn_unary_elementwise_config* xnn_init_f16_hswish_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL || !xnn_is_f16_compatible_config(hardware_config)) {
    return NULL;
  }
  #if XNN_PLATFORM_WINDOWS
    InitOnceExecuteOnce(&init_guard_f16_hswish, &init_f16_hswish_config_windows, NULL, NULL);
  #else
    pthread_once(&init_guard_f16_hswish, &init_f16_hswish_config);
  #endif
  return &f16_hswish_config;
}

const struct xnn_unary_elementwise_config* xnn_init_f16_lrelu_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL || !xnn_is_f16_compatible_config(hardware_config)) {
    return NULL;
  }
  #if XNN_PLATFORM_WINDOWS
    InitOnceExecuteOnce(&init_guard_f16_lrelu, &init_f16_lrelu_config_windows, NULL, NULL);
  #else
    pthread_once(&init_guard_f16_lrelu, &init_f16_lrelu_config);
  #endif
  return &f16_lrelu_config;
}

const struct xnn_unary_elementwise_config* xnn_init_f16_neg_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL || !xnn_is_f16_compatible_config(hardware_config)) {
    return NULL;
  }
  #if XNN_PLATFORM_WINDOWS
    InitOnceExecuteOnce(&init_guard_f16_neg, &init_f16_neg_config_windows, NULL, NULL);
  #else
    pthread_once(&init_guard_f16_neg, &init_f16_neg_config);
  #endif
  return &f16_neg_config;
}

const struct xnn_unary_elementwise_config* xnn_init_f16_rndd_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL || !xnn_is_f16_compatible_config(hardware_config)) {
    return NULL;
  }
  #if XNN_PLATFORM_WINDOWS
    InitOnceExecuteOnce(&init_guard_f16_rndd, &init_f16_rndd_config_windows, NULL, NULL);
  #else
    pthread_once(&init_guard_f16_rndd, &init_f16_rndd_config);
  #endif
  return &f16_rndd_config;
}

const struct xnn_unary_elementwise_config* xnn_init_f16_rndne_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL || !xnn_is_f16_compatible_config(hardware_config)) {
    return NULL;
  }
  #if XNN_PLATFORM_WINDOWS
    InitOnceExecuteOnce(&init_guard_f16_rndne, &init_f16_rndne_config_windows, NULL, NULL);
  #else
    pthread_once(&init_guard_f16_rndne, &init_f16_rndne_config);
  #endif
  return &f16_rndne_config;
}

const struct xnn_unary_elementwise_config* xnn_init_f16_rndu_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL || !xnn_is_f16_compatible_config(hardware_config)) {
    return NULL;
  }
  #if XNN_PLATFORM_WINDOWS
    InitOnceExecuteOnce(&init_guard_f16_rndu, &init_f16_rndu_config_windows, NULL, NULL);
  #else
    pthread_once(&init_guard_f16_rndu, &init_f16_rndu_config);
  #endif
  return &f16_rndu_config;
}

const struct xnn_unary_elementwise_config* xnn_init_f16_rndz_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL || !xnn_is_f16_compatible_config(hardware_config)) {
    return NULL;
  }
  #if XNN_PLATFORM_WINDOWS
    InitOnceExecuteOnce(&init_guard_f16_rndz, &init_f16_rndz_config_windows, NULL, NULL);
  #else
    pthread_once(&init_guard_f16_rndz, &init_f16_rndz_config);
  #endif
  return &f16_rndz_config;
}

const struct xnn_unary_elementwise_config* xnn_init_f16_sigmoid_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL || !xnn_is_f16_compatible_config(hardware_config)) {
    return NULL;
  }
  #if XNN_PLATFORM_WINDOWS
    InitOnceExecuteOnce(&init_guard_f16_sigmoid, &init_f16_sigmoid_config_windows, NULL, NULL);
  #else
    pthread_once(&init_guard_f16_sigmoid, &init_f16_sigmoid_config);
  #endif
  return &f16_sigmoid_config;
}

const struct xnn_unary_elementwise_config* xnn_init_f16_sqr_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL || !xnn_is_f16_compatible_config(hardware_config)) {
    return NULL;
  }
  #if XNN_PLATFORM_WINDOWS
    InitOnceExecuteOnce(&init_guard_f16_sqr, &init_f16_sqr_config_windows, NULL, NULL);
  #else
    pthread_once(&init_guard_f16_sqr, &init_f16_sqr_config);
  #endif
  return &f16_sqr_config;
}

const struct xnn_unary_elementwise_config* xnn_init_f16_sqrt_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL || !xnn_is_f16_compatible_config(hardware_config)) {
    return NULL;
  }
  #if XNN_PLATFORM_WINDOWS
    InitOnceExecuteOnce(&init_guard_f16_sqrt, &init_f16_sqrt_config_windows, NULL, NULL);
  #else
    pthread_once(&init_guard_f16_sqrt, &init_f16_sqrt_config);
  #endif
  return &f16_sqrt_config;
}

const struct xnn_unary_elementwise_config* xnn_init_f16_tanh_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL || !xnn_is_f16_compatible_config(hardware_config)) {
    return NULL;
  }
  #if XNN_PLATFORM_WINDOWS
    InitOnceExecuteOnce(&init_guard_f16_tanh, &init_f16_tanh_config_windows, NULL, NULL);
  #else
    pthread_once(&init_guard_f16_tanh, &init_f16_tanh_config);
  #endif
  return &f16_tanh_config;
}

const struct xnn_unary_elementwise_config* xnn_init_f16_to_f32_cvt_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  #if XNN_PLATFORM_WINDOWS
    InitOnceExecuteOnce(&init_guard_f16_to_f32_cvt, &init_f16_to_f32_cvt_config_windows, NULL, NULL);
  #else
    pthread_once(&init_guard_f16_to_f32_cvt, &init_f16_to_f32_cvt_config);
  #endif
  return &f16_to_f32_cvt_config;
}

const struct xnn_unary_elementwise_config* xnn_init_f16_to_qs8_cvt_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL || !xnn_is_f16_compatible_config(hardware_config)) {
    return NULL;
  }
  #if XNN_PLATFORM_WINDOWS
    InitOnceExecuteOnce(&init_guard_f16_to_qs8_cvt, &init_f16_to_qs8_cvt_config_windows, NULL, NULL);
  #else
    pthread_once(&init_guard_f16_to_qs8_cvt, &init_f16_to_qs8_cvt_config);
  #endif
  return &f16_to_qs8_cvt_config;
}

const struct xnn_unary_elementwise_config* xnn_init_f32_abs_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  #if XNN_PLATFORM_WINDOWS
    InitOnceExecuteOnce(&init_guard_f32_abs, &init_f32_abs_config_windows, NULL, NULL);
  #else
    pthread_once(&init_guard_f32_abs, &init_f32_abs_config);
  #endif
  return &f32_abs_config;
}

const struct xnn_unary_elementwise_config* xnn_init_f32_clamp_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  #if XNN_PLATFORM_WINDOWS
    InitOnceExecuteOnce(&init_guard_f32_clamp, &init_f32_clamp_config_windows, NULL, NULL);
  #else
    pthread_once(&init_guard_f32_clamp, &init_f32_clamp_config);
  #endif
  return &f32_clamp_config;
}

const struct xnn_unary_elementwise_config* xnn_init_f32_elu_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  #if XNN_PLATFORM_WINDOWS
    InitOnceExecuteOnce(&init_guard_f32_elu, &init_f32_elu_config_windows, NULL, NULL);
  #else
    pthread_once(&init_guard_f32_elu, &init_f32_elu_config);
  #endif
  return &f32_elu_config;
}

const struct xnn_unary_elementwise_config* xnn_init_f32_hswish_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  #if XNN_PLATFORM_WINDOWS
    InitOnceExecuteOnce(&init_guard_f32_hswish, &init_f32_hswish_config_windows, NULL, NULL);
  #else
    pthread_once(&init_guard_f32_hswish, &init_f32_hswish_config);
  #endif
  return &f32_hswish_config;
}

const struct xnn_unary_elementwise_config* xnn_init_f32_lrelu_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  #if XNN_PLATFORM_WINDOWS
    InitOnceExecuteOnce(&init_guard_f32_lrelu, &init_f32_lrelu_config_windows, NULL, NULL);
  #else
    pthread_once(&init_guard_f32_lrelu, &init_f32_lrelu_config);
  #endif
  return &f32_lrelu_config;
}

const struct xnn_unary_elementwise_config* xnn_init_f32_neg_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  #if XNN_PLATFORM_WINDOWS
    InitOnceExecuteOnce(&init_guard_f32_neg, &init_f32_neg_config_windows, NULL, NULL);
  #else
    pthread_once(&init_guard_f32_neg, &init_f32_neg_config);
  #endif
  return &f32_neg_config;
}

const struct xnn_unary_elementwise_config* xnn_init_f32_relu_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  #if XNN_PLATFORM_WINDOWS
    InitOnceExecuteOnce(&init_guard_f32_relu, &init_f32_relu_config_windows, NULL, NULL);
  #else
    pthread_once(&init_guard_f32_relu, &init_f32_relu_config);
  #endif
  return &f32_relu_config;
}

const struct xnn_unary_elementwise_config* xnn_init_f32_rndd_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  #if XNN_PLATFORM_WINDOWS
    InitOnceExecuteOnce(&init_guard_f32_rndd, &init_f32_rndd_config_windows, NULL, NULL);
  #else
    pthread_once(&init_guard_f32_rndd, &init_f32_rndd_config);
  #endif
  return &f32_rndd_config;
}

const struct xnn_unary_elementwise_config* xnn_init_f32_rndne_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  #if XNN_PLATFORM_WINDOWS
    InitOnceExecuteOnce(&init_guard_f32_rndne, &init_f32_rndne_config_windows, NULL, NULL);
  #else
    pthread_once(&init_guard_f32_rndne, &init_f32_rndne_config);
  #endif
  return &f32_rndne_config;
}

const struct xnn_unary_elementwise_config* xnn_init_f32_rndu_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  #if XNN_PLATFORM_WINDOWS
    InitOnceExecuteOnce(&init_guard_f32_rndu, &init_f32_rndu_config_windows, NULL, NULL);
  #else
    pthread_once(&init_guard_f32_rndu, &init_f32_rndu_config);
  #endif
  return &f32_rndu_config;
}

const struct xnn_unary_elementwise_config* xnn_init_f32_rndz_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  #if XNN_PLATFORM_WINDOWS
    InitOnceExecuteOnce(&init_guard_f32_rndz, &init_f32_rndz_config_windows, NULL, NULL);
  #else
    pthread_once(&init_guard_f32_rndz, &init_f32_rndz_config);
  #endif
  return &f32_rndz_config;
}

const struct xnn_unary_elementwise_config* xnn_init_f32_rsqrt_config() {
  const struct xnn_hardware_config* hardware_config =
      xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
#if XNN_PLATFORM_WINDOWS
  InitOnceExecuteOnce(&init_guard_f32_rsqrt, &init_f32_rsqrt_config_windows,
                      NULL, NULL);
#else
  pthread_once(&init_guard_f32_rsqrt, &init_f32_rsqrt_config);
#endif
  return &f32_rsqrt_config;
}

const struct xnn_unary_elementwise_config* xnn_init_f32_sigmoid_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  #if XNN_PLATFORM_WINDOWS
    InitOnceExecuteOnce(&init_guard_f32_sigmoid, &init_f32_sigmoid_config_windows, NULL, NULL);
  #else
    pthread_once(&init_guard_f32_sigmoid, &init_f32_sigmoid_config);
  #endif
  return &f32_sigmoid_config;
}

const struct xnn_unary_elementwise_config* xnn_init_f32_sqr_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  #if XNN_PLATFORM_WINDOWS
    InitOnceExecuteOnce(&init_guard_f32_sqr, &init_f32_sqr_config_windows, NULL, NULL);
  #else
    pthread_once(&init_guard_f32_sqr, &init_f32_sqr_config);
  #endif
  return &f32_sqr_config;
}

const struct xnn_unary_elementwise_config* xnn_init_f32_sqrt_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  #if XNN_PLATFORM_WINDOWS
    InitOnceExecuteOnce(&init_guard_f32_sqrt, &init_f32_sqrt_config_windows, NULL, NULL);
  #else
    pthread_once(&init_guard_f32_sqrt, &init_f32_sqrt_config);
  #endif
  return &f32_sqrt_config;
}

const struct xnn_unary_elementwise_config* xnn_init_f32_tanh_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  #if XNN_PLATFORM_WINDOWS
    InitOnceExecuteOnce(&init_guard_f32_tanh, &init_f32_tanh_config_windows, NULL, NULL);
  #else
    pthread_once(&init_guard_f32_tanh, &init_f32_tanh_config);
  #endif
  return &f32_tanh_config;
}

const struct xnn_unary_elementwise_config* xnn_init_f32_to_f16_cvt_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  #if XNN_PLATFORM_WINDOWS
    InitOnceExecuteOnce(&init_guard_f32_to_f16_cvt, &init_f32_to_f16_cvt_config_windows, NULL, NULL);
  #else
    pthread_once(&init_guard_f32_to_f16_cvt, &init_f32_to_f16_cvt_config);
  #endif
  return &f32_to_f16_cvt_config;
}

const struct xnn_unary_elementwise_config* xnn_init_f32_to_qs8_cvt_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  #if XNN_PLATFORM_WINDOWS
    InitOnceExecuteOnce(&init_guard_f32_to_qs8_cvt, &init_f32_to_qs8_cvt_config_windows, NULL, NULL);
  #else
    pthread_once(&init_guard_f32_to_qs8_cvt, &init_f32_to_qs8_cvt_config);
  #endif
  return &f32_to_qs8_cvt_config;
}

const struct xnn_unary_elementwise_config* xnn_init_f32_to_qu8_cvt_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  #if XNN_PLATFORM_WINDOWS
    InitOnceExecuteOnce(&init_guard_f32_to_qu8_cvt, &init_f32_to_qu8_cvt_config_windows, NULL, NULL);
  #else
    pthread_once(&init_guard_f32_to_qu8_cvt, &init_f32_to_qu8_cvt_config);
  #endif
  return &f32_to_qu8_cvt_config;
}

const struct xnn_unary_elementwise_config* xnn_init_qs8_cvt_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  #if XNN_PLATFORM_WINDOWS
    InitOnceExecuteOnce(&init_guard_qs8_cvt, &init_qs8_cvt_config_windows, NULL, NULL);
  #else
    pthread_once(&init_guard_qs8_cvt, &init_qs8_cvt_config);
  #endif
  return &qs8_cvt_config;
}

const struct xnn_unary_elementwise_config* xnn_init_qs16_to_qs8_cvt_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  #if XNN_PLATFORM_WINDOWS
    InitOnceExecuteOnce(&init_guard_qs16_to_qs8_cvt, &init_qs16_to_qs8_cvt_config_windows, NULL, NULL);
  #else
    pthread_once(&init_guard_qs16_to_qs8_cvt, &init_qs16_to_qs8_cvt_config);
  #endif
  return &qs16_to_qs8_cvt_config;
}

const struct xnn_unary_elementwise_config* xnn_init_qs8_lrelu_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  #if XNN_PLATFORM_WINDOWS
    InitOnceExecuteOnce(&init_guard_qs8_lrelu, &init_qs8_lrelu_config_windows, NULL, NULL);
  #else
    pthread_once(&init_guard_qs8_lrelu, &init_qs8_lrelu_config);
  #endif
  return &qs8_lrelu_config;
}

const struct xnn_unary_elementwise_config* xnn_init_qs8_to_f16_cvt_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL || !xnn_is_f16_compatible_config(hardware_config)) {
    return NULL;
  }
  #if XNN_PLATFORM_WINDOWS
    InitOnceExecuteOnce(&init_guard_qs8_to_f16_cvt, &init_qs8_to_f16_cvt_config_windows, NULL, NULL);
  #else
    pthread_once(&init_guard_qs8_to_f16_cvt, &init_qs8_to_f16_cvt_config);
  #endif
  return &qs8_to_f16_cvt_config;
}

const struct xnn_unary_elementwise_config* xnn_init_qs8_to_f32_cvt_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  #if XNN_PLATFORM_WINDOWS
    InitOnceExecuteOnce(&init_guard_qs8_to_f32_cvt, &init_qs8_to_f32_cvt_config_windows, NULL, NULL);
  #else
    pthread_once(&init_guard_qs8_to_f32_cvt, &init_qs8_to_f32_cvt_config);
  #endif
  return &qs8_to_f32_cvt_config;
}

const struct xnn_unary_elementwise_config* xnn_init_qu8_cvt_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  #if XNN_PLATFORM_WINDOWS
    InitOnceExecuteOnce(&init_guard_qu8_cvt, &init_qu8_cvt_config_windows, NULL, NULL);
  #else
    pthread_once(&init_guard_qu8_cvt, &init_qu8_cvt_config);
  #endif
  return &qu8_cvt_config;
}

const struct xnn_unary_elementwise_config* xnn_init_qu8_lrelu_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  #if XNN_PLATFORM_WINDOWS
    InitOnceExecuteOnce(&init_guard_qu8_lrelu, &init_qu8_lrelu_config_windows, NULL, NULL);
  #else
    pthread_once(&init_guard_qu8_lrelu, &init_qu8_lrelu_config);
  #endif
  return &qu8_lrelu_config;
}

const struct xnn_unary_elementwise_config* xnn_init_qu8_to_f32_cvt_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  #if XNN_PLATFORM_WINDOWS
    InitOnceExecuteOnce(&init_guard_qu8_to_f32_cvt, &init_qu8_to_f32_cvt_config_windows, NULL, NULL);
  #else
    pthread_once(&init_guard_qu8_to_f32_cvt, &init_qu8_to_f32_cvt_config);
  #endif
  return &qu8_to_f32_cvt_config;
}

const struct xnn_unary_elementwise_config* xnn_init_s8_clamp_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  #if XNN_PLATFORM_WINDOWS
    InitOnceExecuteOnce(&init_guard_s8_clamp, &init_s8_clamp_config_windows, NULL, NULL);
  #else
    pthread_once(&init_guard_s8_clamp, &init_s8_clamp_config);
  #endif
  return &s8_clamp_config;
}

const struct xnn_unary_elementwise_config* xnn_init_u8_clamp_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  #if XNN_PLATFORM_WINDOWS
    InitOnceExecuteOnce(&init_guard_u8_clamp, &init_u8_clamp_config_windows, NULL, NULL);
  #else
    pthread_once(&init_guard_u8_clamp, &init_u8_clamp_config);
  #endif
  return &u8_clamp_config;
}

const struct xnn_unary_elementwise_config* xnn_init_xx_copy_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  #if XNN_PLATFORM_WINDOWS
    InitOnceExecuteOnce(&init_guard_xx_copy, &init_xx_copy_config_windows, NULL, NULL);
  #else
    pthread_once(&init_guard_xx_copy, &init_xx_copy_config);
  #endif
  return &xx_copy_config;
}
