// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include "xnnpack/config-types.h"
#include "xnnpack/hardware-config.h"

#include "xnnpack/common.h"
#include "xnnpack/config.h"
#include "xnnpack/init-once.h"
#include "xnnpack/microfnptr.h"
#include "xnnpack/microparams-init.h"
#include "xnnpack/packq.h"
#include "xnnpack/vcvt.h"
#include "xnnpack/vlrelu.h"
#include "xnnpack/vunary.h"

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
static struct xnn_unary_elementwise_config f16_rsqrt_config = {0};
static struct xnn_unary_elementwise_config f16_sigmoid_config = {0};
static struct xnn_unary_elementwise_config f16_sqr_config = {0};
static struct xnn_unary_elementwise_config f16_sqrt_config = {0};
static struct xnn_unary_elementwise_config f16_tanh_config = {0};
static struct xnn_unary_elementwise_config f16_to_f32_cvt_config = {0};
static struct xnn_unary_elementwise_config f16_to_qs8_cvt_config = {0};
static struct xnn_unary_elementwise_config f32_abs_config = {0};
static struct xnn_unary_elementwise_config f32_clamp_config = {0};
static struct xnn_unary_elementwise_config f32_elu_config = {0};
static struct xnn_unary_elementwise_config f32_exp_config = {0};
static struct xnn_unary_elementwise_config f32_gelu_config = {0};
static struct xnn_unary_elementwise_config f32_hswish_config = {0};
static struct xnn_unary_elementwise_config f32_log_config = {0};
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
static struct xnn_unary_elementwise_config f32_to_qp8_cvt_config = {0};
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


XNN_INIT_ONCE_GUARD(f16_abs);
XNN_INIT_ONCE_GUARD(f16_clamp);
XNN_INIT_ONCE_GUARD(f16_elu);
XNN_INIT_ONCE_GUARD(f16_hswish);
XNN_INIT_ONCE_GUARD(f16_lrelu);
XNN_INIT_ONCE_GUARD(f16_neg);
XNN_INIT_ONCE_GUARD(f16_rndd);
XNN_INIT_ONCE_GUARD(f16_rndne);
XNN_INIT_ONCE_GUARD(f16_rndu);
XNN_INIT_ONCE_GUARD(f16_rndz);
XNN_INIT_ONCE_GUARD(f16_rsqrt);
XNN_INIT_ONCE_GUARD(f16_sigmoid);
XNN_INIT_ONCE_GUARD(f16_sqr);
XNN_INIT_ONCE_GUARD(f16_sqrt);
XNN_INIT_ONCE_GUARD(f16_tanh);
XNN_INIT_ONCE_GUARD(f16_to_qs8_cvt);
XNN_INIT_ONCE_GUARD(f16_to_f32_cvt);
XNN_INIT_ONCE_GUARD(f32_abs);
XNN_INIT_ONCE_GUARD(f32_clamp);
XNN_INIT_ONCE_GUARD(f32_elu);
XNN_INIT_ONCE_GUARD(f32_exp);
XNN_INIT_ONCE_GUARD(f32_gelu);
XNN_INIT_ONCE_GUARD(f32_hswish);
XNN_INIT_ONCE_GUARD(f32_log);
XNN_INIT_ONCE_GUARD(f32_lrelu);
XNN_INIT_ONCE_GUARD(f32_neg);
XNN_INIT_ONCE_GUARD(f32_relu);
XNN_INIT_ONCE_GUARD(f32_rndd);
XNN_INIT_ONCE_GUARD(f32_rndne);
XNN_INIT_ONCE_GUARD(f32_rndu);
XNN_INIT_ONCE_GUARD(f32_rndz);
XNN_INIT_ONCE_GUARD(f32_rsqrt);
XNN_INIT_ONCE_GUARD(f32_sigmoid);
XNN_INIT_ONCE_GUARD(f32_sqr);
XNN_INIT_ONCE_GUARD(f32_sqrt);
XNN_INIT_ONCE_GUARD(f32_tanh);
XNN_INIT_ONCE_GUARD(f32_to_f16_cvt);
XNN_INIT_ONCE_GUARD(f32_to_qp8_cvt);
XNN_INIT_ONCE_GUARD(f32_to_qs8_cvt);
XNN_INIT_ONCE_GUARD(f32_to_qu8_cvt);
XNN_INIT_ONCE_GUARD(qs8_cvt);
XNN_INIT_ONCE_GUARD(qs8_lrelu);
XNN_INIT_ONCE_GUARD(qs8_to_f16_cvt);
XNN_INIT_ONCE_GUARD(qs8_to_f32_cvt);
XNN_INIT_ONCE_GUARD(qs16_to_qs8_cvt);
XNN_INIT_ONCE_GUARD(qu8_cvt);
XNN_INIT_ONCE_GUARD(qu8_lrelu);
XNN_INIT_ONCE_GUARD(qu8_to_f32_cvt);
XNN_INIT_ONCE_GUARD(s8_clamp);
XNN_INIT_ONCE_GUARD(u8_clamp);
XNN_INIT_ONCE_GUARD(xx_copy);

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
      f16_elu_config.init.f16_elu = xnn_init_f16_elu_scalar_params;
      f16_elu_config.element_tile = 16;
    }
  #elif XNN_ARCH_ARM64 && XNN_ENABLE_ARM_FP16_VECTOR
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon_fp16_arith) {
      f16_elu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f16_velu_ukernel__neonfp16arith_rr1_p3_u16;
      f16_elu_config.init.f16_elu = xnn_init_f16_elu_scalar_params;
      f16_elu_config.element_tile = 16;
    }
  #elif (XNN_ARCH_X86 || XNN_ARCH_X86_64) && !XNN_PLATFORM_MOBILE
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_x86_avx2) {
      f16_elu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f16_velu_ukernel__avx2_rr1_p3_u16;
      f16_elu_config.init.f16_elu = xnn_init_f16_elu_avx2_params;
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
      f16_hswish_config.element_tile = 16;
    }
  #elif XNN_ARCH_ARM64 && XNN_ENABLE_ARM_FP16_VECTOR
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon_fp16_arith) {
      f16_hswish_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f16_vhswish_ukernel__neonfp16arith_u16;
      f16_hswish_config.element_tile = 16;
    }
  #elif (XNN_ARCH_X86 || XNN_ARCH_X86_64) && !XNN_PLATFORM_MOBILE
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_x86_f16c) {
      f16_hswish_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f16_vhswish_ukernel__f16c_u16;
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

static void init_f16_rsqrt_config(void) {
  #if XNN_ARCH_ARM && XNN_ENABLE_ARM_FP16_VECTOR && XNN_ENABLE_ARM_FP16_SCALAR
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon_fp16_arith) {
      f16_rsqrt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f16_vrsqrt_ukernel__neonfp16arith_rsqrt_u16;
      f16_rsqrt_config.element_tile = 16;
    }
  #elif XNN_ARCH_ARM64 && XNN_ENABLE_ARM_FP16_VECTOR
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon_fp16_arith) {
      f16_rsqrt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f16_vrsqrt_ukernel__neonfp16arith_rsqrt_u16;
      f16_rsqrt_config.element_tile = 16;
    }
  #elif (XNN_ARCH_X86 || XNN_ARCH_X86_64) && !XNN_PLATFORM_MOBILE
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_x86_f16c) {
      f16_rsqrt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f16_vrsqrt_ukernel__f16c_rsqrt_u32;
      f16_rsqrt_config.element_tile = 32;
    }
  #endif
}

static void init_f16_sigmoid_config(void) {
  #if XNN_ARCH_ARM && XNN_ENABLE_ARM_FP16_VECTOR && XNN_ENABLE_ARM_FP16_SCALAR
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon_fp16_arith) {
      f16_sigmoid_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1recps_u16;
      f16_sigmoid_config.element_tile = 16;
    }
  #elif XNN_ARCH_ARM64 && XNN_ENABLE_ARM_FP16_VECTOR
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon_fp16_arith) {
      f16_sigmoid_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1fma_u40;
      f16_sigmoid_config.element_tile = 40;
    }
  #elif (XNN_ARCH_X86 || XNN_ARCH_X86_64) && !XNN_PLATFORM_MOBILE
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_x86_avx2) {
      f16_sigmoid_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_rcp_u32;
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
        f16_to_f32_cvt_config.element_tile = 16;
      }
    } else if (!XNN_PLATFORM_MOBILE) {
      f16_to_f32_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f16_f32_vcvt_ukernel__scalar_u4;
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
      f16_to_f32_cvt_config.element_tile = 16;
    } else if (hardware_config->use_x86_sse4_1) {
      f16_to_f32_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f16_f32_vcvt_ukernel__sse41_int16_u16;
      f16_to_f32_cvt_config.element_tile = 16;
    } else {
      f16_to_f32_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f16_f32_vcvt_ukernel__sse2_int16_u32;
      f16_to_f32_cvt_config.element_tile = 32;
    }
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    #if XNN_ARCH_WASMRELAXEDSIMD
      f16_to_f32_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f16_f32_vcvt_ukernel__wasmrelaxedsimd_int16_u16;
      f16_to_f32_cvt_config.element_tile = 16;
    #else
      f16_to_f32_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f16_f32_vcvt_ukernel__wasmsimd_int16_u16;
      f16_to_f32_cvt_config.element_tile = 16;
    #endif
  #elif XNN_ARCH_WASM
    f16_to_f32_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f16_f32_vcvt_ukernel__scalar_u1;
    f16_to_f32_cvt_config.element_tile = 1;
  #elif XNN_ARCH_RISCV
    f16_to_f32_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f16_f32_vcvt_ukernel__scalar_u4;
    f16_to_f32_cvt_config.element_tile = 4;
  #else
    f16_to_f32_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f16_f32_vcvt_ukernel__scalar_u4;
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
    f16_to_qs8_cvt_config.init.f16_qs8_cvt = xnn_init_f16_qs8_cvt_scalar_params;
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
      f32_abs_config.element_tile = 16;
    } else if (hardware_config->use_x86_avx) {
      f32_abs_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vabs_ukernel__avx_u16;
      f32_abs_config.element_tile = 16;
    } else {
      f32_abs_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vabs_ukernel__sse2_u8;
      f32_abs_config.element_tile = 8;
    }
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    f32_abs_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vabs_ukernel__wasmsimd_u8;
    f32_abs_config.element_tile = 8;
  #elif XNN_ARCH_WASM
    f32_abs_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vabs_ukernel__scalar_u4;
    f32_abs_config.element_tile = 4;
  #elif XNN_ARCH_RISCV
    f32_abs_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vabs_ukernel__scalar_u4;
    f32_abs_config.element_tile = 4;
  #else
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
  #else
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
        f32_elu_config.init.f32_elu = xnn_init_f32_elu_scalar_params;
        f32_elu_config.element_tile = 8;
      } else {
        f32_elu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_velu_ukernel__neon_rr2_lut16_p3_u8;
        f32_elu_config.init.f32_elu = xnn_init_f32_elu_scalar_params;
        f32_elu_config.element_tile = 8;
      }
    } else if (!XNN_PLATFORM_MOBILE) {
      f32_elu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_velu_ukernel__scalar_rr2_lut16_p3_u4;
      f32_elu_config.init.f32_elu = xnn_init_f32_elu_scalar_params;
      f32_elu_config.element_tile = 4;
    }
  #elif XNN_ARCH_ARM64
    f32_elu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_velu_ukernel__neonfma_rr1_lut16_p3_u16;
    f32_elu_config.init.f32_elu = xnn_init_f32_elu_scalar_params;
    f32_elu_config.element_tile = 16;
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (!XNN_PLATFORM_MOBILE && hardware_config->use_x86_avx512f) {
      f32_elu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_velu_ukernel__avx512f_rr1_p6_u128;
      f32_elu_config.init.f32_elu = xnn_init_f32_elu_scalar_params;
      f32_elu_config.element_tile = 128;
    } else if (hardware_config->use_x86_avx2) {
      f32_elu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_velu_ukernel__avx2_rr1_lut4_p4_perm_u56;
      f32_elu_config.init.f32_elu = xnn_init_f32_elu_scalar_params;
      f32_elu_config.element_tile = 56;
    } else if (hardware_config->use_x86_avx) {
      f32_elu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_velu_ukernel__avx_rr2_lut4_p4_perm_u32;
      f32_elu_config.init.f32_elu = xnn_init_f32_elu_scalar_params;
      f32_elu_config.element_tile = 32;
    } else {
      f32_elu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_velu_ukernel__sse2_rr2_lut16_p3_u12;
      f32_elu_config.init.f32_elu = xnn_init_f32_elu_scalar_params;
      f32_elu_config.element_tile = 12;
    }
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    #if XNN_ARCH_WASMRELAXEDSIMD
      f32_elu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_p6_u24;
      f32_elu_config.init.f32_elu = xnn_init_f32_elu_scalar_params;
      f32_elu_config.element_tile = 24;
    #else
      const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
      assert(hardware_config != NULL);
      if (hardware_config->is_x86) {
        f32_elu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_velu_ukernel__wasmsimd_x86_rr2_p6_u20;
        f32_elu_config.init.f32_elu = xnn_init_f32_elu_scalar_params;
        f32_elu_config.element_tile = 20;
      } else {
        f32_elu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_velu_ukernel__wasmsimd_arm_rr2_p6_u20;
        f32_elu_config.init.f32_elu = xnn_init_f32_elu_scalar_params;
        f32_elu_config.element_tile = 20;
      }
    #endif
  #elif XNN_ARCH_WASM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->is_x86) {
      f32_elu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_velu_ukernel__scalar_rr2_lut16_p3_u2;
      f32_elu_config.init.f32_elu = xnn_init_f32_elu_scalar_params;
      f32_elu_config.element_tile = 2;
    } else {
      f32_elu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_velu_ukernel__wasm_rr2_p6_u6;
      f32_elu_config.init.f32_elu = xnn_init_f32_elu_scalar_params;
      f32_elu_config.element_tile = 6;
    }
  #elif XNN_ARCH_RISCV
    f32_elu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_velu_ukernel__scalar_rr2_lut16_p3_u4;
    f32_elu_config.init.f32_elu = xnn_init_f32_elu_scalar_params;
    f32_elu_config.element_tile = 4;
  #else
    f32_elu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_velu_ukernel__scalar_rr2_lut16_p3_u4;
    f32_elu_config.init.f32_elu = xnn_init_f32_elu_scalar_params;
    f32_elu_config.element_tile = 4;
  #endif
}

static void init_f32_gelu_config(void) {
  #if XNN_ARCH_ARM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon) {
      if (hardware_config->use_arm_neon_fma) {
        f32_gelu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vgelu_ukernel__neon_rational_12_10_div_u8;
        f32_gelu_config.element_tile = 8;
      } else {
        f32_gelu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vgelu_ukernel__neon_rational_12_10_div_u8;
        f32_gelu_config.element_tile = 8;
      }
    } else if (!XNN_PLATFORM_MOBILE) {
      f32_gelu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vgelu_ukernel__neon_rational_12_10_div_u8;
      f32_gelu_config.element_tile = 8;
    }
  #elif XNN_ARCH_ARM64
    f32_gelu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vgelu_ukernel__neon_rational_12_10_div_u8;
    f32_gelu_config.element_tile = 8;
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (!XNN_PLATFORM_MOBILE && hardware_config->use_x86_avx512f) {
      f32_gelu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vgelu_ukernel__avx512f_rational_12_10_nr_u32;
      f32_gelu_config.element_tile = 32;
    } else if (hardware_config->use_x86_fma3) {
      f32_gelu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vgelu_ukernel__fma3_rational_12_10_div_u16;
      f32_gelu_config.element_tile = 16;
    } else if (hardware_config->use_x86_avx) {
      f32_gelu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vgelu_ukernel__avx_rational_12_10_div_u16;
      f32_gelu_config.element_tile = 16;
    } else {
      f32_gelu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vgelu_ukernel__sse2_rational_12_10_div_u12;
      f32_gelu_config.element_tile = 12;
    }
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    f32_gelu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vgelu_ukernel__wasmsimd_rational_12_10_div_u12;
    f32_gelu_config.element_tile = 12;
  #elif XNN_ARCH_WASM
    f32_gelu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vgelu_ukernel__scalar_rational_12_10_div_u1;
    f32_gelu_config.element_tile = 1;
  #elif XNN_ARCH_RISCV
    f32_gelu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vgelu_ukernel__scalar_rational_12_10_div_u1;
    f32_gelu_config.element_tile = 1;
  #else
    f32_gelu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vgelu_ukernel__scalar_rational_12_10_div_u1;
    f32_gelu_config.element_tile = 1;
  #endif
}

static void init_f32_hswish_config(void) {
  #if XNN_ARCH_ARM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon) {
      f32_hswish_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vhswish_ukernel__neon_u16;
      f32_hswish_config.element_tile = 16;
    } else if (!XNN_PLATFORM_MOBILE) {
      f32_hswish_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vhswish_ukernel__scalar_u4;
      f32_hswish_config.element_tile = 4;
    }
  #elif XNN_ARCH_ARM64
    f32_hswish_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vhswish_ukernel__neon_u16;
    f32_hswish_config.element_tile = 16;
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (!XNN_PLATFORM_MOBILE && hardware_config->use_x86_avx512f) {
      f32_hswish_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vhswish_ukernel__avx512f_u16;
      f32_hswish_config.element_tile = 16;
    } else if (hardware_config->use_x86_fma3) {
      f32_hswish_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vhswish_ukernel__fma3_u16;
      f32_hswish_config.element_tile = 16;
    } else if (hardware_config->use_x86_avx) {
      f32_hswish_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vhswish_ukernel__avx_u16;
      f32_hswish_config.element_tile = 16;
    } else {
      f32_hswish_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vhswish_ukernel__sse_u8;
      f32_hswish_config.element_tile = 8;
    }
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    f32_hswish_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vhswish_ukernel__wasmsimd_u16;
    f32_hswish_config.element_tile = 16;
  #elif XNN_ARCH_WASM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->is_x86) {
      f32_hswish_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vhswish_ukernel__scalar_u4;
      f32_hswish_config.element_tile = 4;
    } else {
      f32_hswish_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vhswish_ukernel__wasm_u4;
      f32_hswish_config.element_tile = 4;
    }
  #elif XNN_ARCH_RISCV
    f32_hswish_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vhswish_ukernel__scalar_u4;
    f32_hswish_config.element_tile = 4;
  #else
    f32_hswish_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vhswish_ukernel__scalar_u4;
    f32_hswish_config.element_tile = 4;
  #endif
}

static void init_f32_exp_config(void) {
  f32_exp_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vexp_ukernel__scalar_exp_u4;
  f32_exp_config.element_tile = 4;
}

static void init_f32_log_config(void) {
  #if XNN_ARCH_ARM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon) {
      f32_log_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vlog_ukernel__neon_rational_3_3_div_u8;
      f32_log_config.element_tile = 8;
    } else if (!XNN_PLATFORM_MOBILE) {
      f32_log_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vlog_ukernel__scalar_rational_3_3_div_u1;
      f32_log_config.element_tile = 1;
    }
  #elif XNN_ARCH_ARM64
    f32_log_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vlog_ukernel__neon_rational_3_3_div_u8;
    f32_log_config.element_tile = 8;
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (!XNN_PLATFORM_MOBILE && hardware_config->use_x86_avx512f) {
      f32_log_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vlog_ukernel__avx512f_rational_3_3_div_u16;
      f32_log_config.element_tile = 16;
    } else if (hardware_config->use_x86_fma3) {
      f32_log_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vlog_ukernel__fma3_rational_3_3_div_u16;
      f32_log_config.element_tile = 16;
    } else if (hardware_config->use_x86_avx2) {
      f32_log_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vlog_ukernel__avx2_rational_3_3_div_u16;
      f32_log_config.element_tile = 16;
    } else {
      f32_log_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vlog_ukernel__sse2_rational_3_3_div_u8;
      f32_log_config.element_tile = 8;
    }
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    f32_log_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vlog_ukernel__wasmsimd_rational_3_3_div_u8;
    f32_log_config.element_tile = 8;
  #else
    f32_log_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vlog_ukernel__scalar_rational_3_3_div_u1;
    f32_log_config.element_tile = 1;
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
      f32_lrelu_config.init.f32_lrelu = xnn_init_f32_lrelu_scalar_params;
      f32_lrelu_config.element_tile = 16;
    } else if (hardware_config->use_x86_sse4_1) {
      f32_lrelu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vlrelu_ukernel__sse41_u8;
      f32_lrelu_config.init.f32_lrelu = xnn_init_f32_lrelu_scalar_params;
      f32_lrelu_config.element_tile = 8;
    } else {
      f32_lrelu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vlrelu_ukernel__sse_u8;
      f32_lrelu_config.init.f32_lrelu = xnn_init_f32_lrelu_scalar_params;
      f32_lrelu_config.element_tile = 8;
    }
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    #if XNN_ARCH_WASMRELAXEDSIMD
      if (hardware_config->is_x86) {
        f32_lrelu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vlrelu_ukernel__wasmrelaxedsimd_iminmax_u4;
        f32_lrelu_config.init.f32_lrelu = xnn_init_f32_lrelu_scalar_params;
        f32_lrelu_config.element_tile = 4;
      } else {
        f32_lrelu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vlrelu_ukernel__wasmrelaxedsimd_laneselect_u4;
        f32_lrelu_config.init.f32_lrelu = xnn_init_f32_lrelu_scalar_params;
        f32_lrelu_config.element_tile = 4;
      }
    #else
      if (hardware_config->is_x86) {
        f32_lrelu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vlrelu_ukernel__wasmsimd_iminmax_u8;
        f32_lrelu_config.init.f32_lrelu = xnn_init_f32_lrelu_scalar_params;
        f32_lrelu_config.element_tile = 8;
      } else {
        f32_lrelu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vlrelu_ukernel__wasmsimd_laneselect_u8;
        f32_lrelu_config.init.f32_lrelu = xnn_init_f32_lrelu_scalar_params;
        f32_lrelu_config.element_tile = 8;
      }
    #endif
  #elif XNN_ARCH_WASM
    f32_lrelu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vlrelu_ukernel__scalar_u4;
    f32_lrelu_config.init.f32_lrelu = xnn_init_f32_lrelu_scalar_params;
    f32_lrelu_config.element_tile = 4;
  #elif XNN_ARCH_RISCV && XNN_ENABLE_RISCV_VECTOR
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    f32_lrelu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vlrelu_ukernel__rvv_u4v;
    f32_lrelu_config.init.f32_lrelu = xnn_init_f32_lrelu_scalar_params;
    f32_lrelu_config.element_tile = hardware_config->vlenb / sizeof(float) * 4; // (VLENB/sizeof)*LMUL
  #else
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
      f32_neg_config.element_tile = 16;
    } else if (hardware_config->use_x86_avx) {
      f32_neg_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vneg_ukernel__avx_u16;
      f32_neg_config.element_tile = 16;
    } else {
      f32_neg_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vneg_ukernel__sse2_u8;
      f32_neg_config.element_tile = 8;
    }
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    f32_neg_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vneg_ukernel__wasmsimd_u8;
    f32_neg_config.element_tile = 8;
  #elif XNN_ARCH_WASM
    f32_neg_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vneg_ukernel__scalar_u4;
    f32_neg_config.element_tile = 4;
  #elif XNN_ARCH_RISCV
    f32_neg_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vneg_ukernel__scalar_u4;
    f32_neg_config.element_tile = 4;
  #else
    f32_neg_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vneg_ukernel__scalar_u4;
    f32_neg_config.element_tile = 4;
  #endif
}

static void init_f32_relu_config(void) {
  #if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    f32_relu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vrelu_ukernel__wasmsimd_u16;
    f32_relu_config.element_tile = 16;
  #elif XNN_ARCH_RISCV && XNN_ENABLE_RISCV_VECTOR
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    f32_relu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vrelu_ukernel__rvv_u4v;
    f32_relu_config.element_tile = hardware_config->vlenb;
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
      f32_rndd_config.element_tile = 16;
    } else if (hardware_config->use_x86_sse4_1) {
      f32_rndd_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vrndd_ukernel__sse41_u8;
      f32_rndd_config.element_tile = 8;
    } else {
      f32_rndd_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vrndd_ukernel__sse2_u8;
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
  #else
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
      f32_rndne_config.element_tile = 16;
    } else if (hardware_config->use_x86_sse4_1) {
      f32_rndne_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vrndne_ukernel__sse41_u8;
      f32_rndne_config.element_tile = 8;
    } else {
      f32_rndne_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vrndne_ukernel__sse2_u8;
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
  #else
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
      f32_rndu_config.element_tile = 16;
    } else if (hardware_config->use_x86_sse4_1) {
      f32_rndu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vrndu_ukernel__sse41_u8;
      f32_rndu_config.element_tile = 8;
    } else {
      f32_rndu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vrndu_ukernel__sse2_u8;
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
  #else
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
      f32_rndz_config.element_tile = 16;
    } else if (hardware_config->use_x86_sse4_1) {
      f32_rndz_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vrndz_ukernel__sse41_u8;
      f32_rndz_config.element_tile = 8;
    } else {
      f32_rndz_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vrndz_ukernel__sse2_u8;
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
  #else
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
      f32_sigmoid_config.element_tile = 8;
    } else if (!XNN_PLATFORM_MOBILE) {
      f32_sigmoid_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vsigmoid_ukernel__scalar_rr2_lut64_p2_div_u2;
      f32_sigmoid_config.element_tile = 2;
    }
  #elif XNN_ARCH_ARM64
    f32_sigmoid_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_u16;
    f32_sigmoid_config.element_tile = 16;
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (!XNN_PLATFORM_MOBILE && hardware_config->use_x86_avx512f) {
      f32_sigmoid_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_u64;
      f32_sigmoid_config.element_tile = 64;
    } else if (hardware_config->use_x86_avx2) {
      f32_sigmoid_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_u40;
      f32_sigmoid_config.element_tile = 40;
    } else if (hardware_config->use_x86_avx) {
      f32_sigmoid_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_u40;
      f32_sigmoid_config.element_tile = 40;
    } else if (hardware_config->use_x86_sse4_1) {
      f32_sigmoid_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vsigmoid_ukernel__sse41_rr2_lut64_p2_div_u8;
      f32_sigmoid_config.element_tile = 8;
    } else {
      f32_sigmoid_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vsigmoid_ukernel__sse2_rr2_lut64_p2_div_u8;
      f32_sigmoid_config.element_tile = 8;
    }
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    #if XNN_ARCH_WASMRELAXEDSIMD
      f32_sigmoid_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_p5_div_u24;
      f32_sigmoid_config.element_tile = 24;
    #else
      f32_sigmoid_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_p5_div_u16;
      f32_sigmoid_config.element_tile = 16;
    #endif
  #elif XNN_ARCH_WASM
    f32_sigmoid_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vsigmoid_ukernel__scalar_rr2_lut64_p2_div_u2;
    f32_sigmoid_config.element_tile = 2;
  #elif XNN_ARCH_RISCV
    f32_sigmoid_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vsigmoid_ukernel__scalar_rr2_lut64_p2_div_u2;
    f32_sigmoid_config.element_tile = 2;
  #else
    f32_sigmoid_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vsigmoid_ukernel__scalar_rr2_lut64_p2_div_u2;
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
      f32_sqr_config.element_tile = 16;
    } else {
      f32_sqr_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vsqr_ukernel__sse2_u8;
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
  #else
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
    if (hardware_config->use_x86_avx512f) {
      f32_sqrt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vsqrt_ukernel__avx512f_rsqrt_u16;
      f32_sqrt_config.element_tile = 16;
    } else if (hardware_config->use_x86_fma3) {
      f32_sqrt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vsqrt_ukernel__fma3_rsqrt_u16;
      f32_sqrt_config.element_tile = 16;
    } else if (hardware_config->use_x86_avx) {
      f32_sqrt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vsqrt_ukernel__avx_rsqrt_u16;
      f32_sqrt_config.element_tile = 16;
    } else {
      f32_sqrt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vsqrt_ukernel__sse_rsqrt_u12;
      f32_sqrt_config.element_tile = 12;
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
  #else
    f32_sqrt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vsqrt_ukernel__scalar_sqrt_u1;
    f32_sqrt_config.element_tile = 1;
  #endif
}

static void init_f32_rsqrt_config(void) {
  #if XNN_ARCH_ARM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon) {
      f32_rsqrt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vrsqrt_ukernel__neon_rsqrt_u16;
      f32_rsqrt_config.element_tile = 16;
    } else if (!XNN_PLATFORM_MOBILE) {
      f32_rsqrt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vrsqrt_ukernel__scalar_rsqrt_u1;
      f32_rsqrt_config.element_tile = 1;
    }
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (!XNN_PLATFORM_MOBILE && hardware_config->use_x86_avx512f) {
      f32_rsqrt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vrsqrt_ukernel__avx512f_rsqrt_u32;
      f32_rsqrt_config.element_tile = 32;
    } else if (!XNN_PLATFORM_MOBILE && hardware_config->use_x86_fma3) {
      f32_rsqrt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vrsqrt_ukernel__fma3_rsqrt_u16;
      f32_rsqrt_config.element_tile = 16;
    } else if (hardware_config->use_x86_avx) {
      f32_rsqrt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vrsqrt_ukernel__avx_rsqrt_u16;
      f32_rsqrt_config.element_tile = 16;
    } else {
      f32_rsqrt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vrsqrt_ukernel__sse_rsqrt_u8;
      f32_rsqrt_config.element_tile = 8;
    }
  #elif XNN_ARCH_RISCV && XNN_ENABLE_RISCV_VECTOR
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    f32_rsqrt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vrsqrt_ukernel__rvv_rsqrt_u4v;
    f32_rsqrt_config.element_tile = hardware_config->vlenb / sizeof(float) * 4; // (VLENB/sizeof)*LMUL
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
      f32_tanh_config.ukernel = (xnn_vunary_ukernel_fn)xnn_f32_vtanh_ukernel__scalar_rational_9_6_div_u1;
      f32_tanh_config.element_tile = 1;
    }
  #elif XNN_ARCH_ARM64
    f32_tanh_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vtanh_ukernel__aarch64_neonfma_expm1minus_rr1_p6h5ts_div_u16;
    f32_tanh_config.init.f32_tanh = xnn_init_f32_tanh_neon_expm1minus_rr1_p6h5_params;
    f32_tanh_config.element_tile = 16;
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (!XNN_PLATFORM_MOBILE && hardware_config->use_x86_avx512skx) {
      f32_tanh_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vtanh_ukernel__avx512f_rational_9_6_nr_u16;
      f32_tanh_config.element_tile = 16;
    } else if (hardware_config->use_x86_fma3) {
      f32_tanh_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vtanh_ukernel__fma3_rational_9_6_div_u16;
      f32_tanh_config.element_tile = 16;
    } else if (hardware_config->use_x86_avx) {
      f32_tanh_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vtanh_ukernel__avx_rational_9_6_div_u16;
      f32_tanh_config.element_tile = 16;
    } else {
      f32_tanh_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vtanh_ukernel__sse2_rational_9_6_div_u8;
      f32_tanh_config.element_tile = 8;
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
      f32_tanh_config.ukernel = (xnn_vunary_ukernel_fn)
          xnn_f32_vtanh_ukernel__scalar_rational_9_6_div_u1;
      f32_tanh_config.init.f32_tanh = NULL;
      f32_tanh_config.element_tile = 1;
    } else {
      f32_tanh_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_vtanh_ukernel__wasm_expm1minus_rr1_p6h5ts_div_u4;
      f32_tanh_config.init.f32_tanh = xnn_init_f32_tanh_scalar_expm1minus_rr1_p6h5_params;
      f32_tanh_config.element_tile = 4;
    }
  #else
    f32_tanh_config.ukernel = (xnn_vunary_ukernel_fn)xnn_f32_vtanh_ukernel__scalar_rational_9_6_div_u1;
    f32_tanh_config.init.f32_tanh = NULL;
    f32_tanh_config.element_tile = 1;
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
        f32_to_f16_cvt_config.element_tile = 8;
      }
    } else if (!XNN_PLATFORM_MOBILE) {
      f32_to_f16_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_f16_vcvt_ukernel__scalar_fabsf_u2;
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
      f32_to_f16_cvt_config.element_tile = 16;
    } else if (hardware_config->use_x86_avx) {
      f32_to_f16_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_f16_vcvt_ukernel__avx_u24;
      f32_to_f16_cvt_config.element_tile = 24;
    } else if (hardware_config->use_x86_sse4_1) {
      f32_to_f16_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_f16_vcvt_ukernel__sse41_u8;
      f32_to_f16_cvt_config.element_tile = 8;
    } else {
      f32_to_f16_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_f16_vcvt_ukernel__sse2_u16;
      f32_to_f16_cvt_config.element_tile = 16;
    }
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    #if XNN_ARCH_WASMRELAXEDSIMD
      f32_to_f16_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_f16_vcvt_ukernel__wasmrelaxedsimd_u24;
      f32_to_f16_cvt_config.element_tile = 24;
    #else
      f32_to_f16_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_f16_vcvt_ukernel__wasmsimd_u24;
      f32_to_f16_cvt_config.element_tile = 24;
    #endif
  #elif XNN_ARCH_WASM
    f32_to_f16_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_f16_vcvt_ukernel__scalar_bitcast_u4;
    f32_to_f16_cvt_config.element_tile = 4;
  #elif XNN_ARCH_RISCV
    f32_to_f16_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_f16_vcvt_ukernel__scalar_fabsf_u2;
    f32_to_f16_cvt_config.element_tile = 2;
  #else
    f32_to_f16_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_f16_vcvt_ukernel__scalar_fabsf_u2;
    f32_to_f16_cvt_config.element_tile = 2;
  #endif
}

static void init_f32_to_qp8_cvt_config(void) {
#if XNN_ARCH_ARM64 && XNN_ENABLE_KLEIDIAI
  f32_to_qp8_cvt_config.ukernel =
      (xnn_vunary_ukernel_fn)xnn_x8_packq_f32qp8_ukernel__aarch64_neon_u2;
#else
  f32_to_qp8_cvt_config.ukernel =
      (xnn_vunary_ukernel_fn)xnn_x8_packq_f32qp8_ukernel__scalar_u1;
#endif
}

static void init_f32_to_qs8_cvt_config(void) {
  #if XNN_ARCH_ARM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon) {
      if (hardware_config->use_arm_neon_v8) {
        f32_to_qs8_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_qs8_vcvt_ukernel__neonv8_u32;
        f32_to_qs8_cvt_config.init.f32_qs8_cvt = xnn_init_f32_qs8_cvt_scalar_params;
        f32_to_qs8_cvt_config.element_tile = 32;
      } else {
        f32_to_qs8_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_qs8_vcvt_ukernel__neon_u32;
        f32_to_qs8_cvt_config.init.f32_qs8_cvt = xnn_init_f32_qs8_cvt_scalar_params;
        f32_to_qs8_cvt_config.element_tile = 32;
      }
    } else if (!XNN_PLATFORM_MOBILE) {
      f32_to_qs8_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_qs8_vcvt_ukernel__scalar_imagic_u4;
      f32_to_qs8_cvt_config.init.f32_qs8_cvt = xnn_init_f32_qs8_cvt_scalar_params;
      f32_to_qs8_cvt_config.element_tile = 4;
    }
  #elif XNN_ARCH_ARM64
    f32_to_qs8_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_qs8_vcvt_ukernel__neonv8_u32;
    f32_to_qs8_cvt_config.init.f32_qs8_cvt = xnn_init_f32_qs8_cvt_scalar_params;
    f32_to_qs8_cvt_config.element_tile = 32;
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_x86_avx512skx) {
      f32_to_qs8_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_qs8_vcvt_ukernel__avx512skx_u128;
      f32_to_qs8_cvt_config.init.f32_qs8_cvt = xnn_init_f32_qs8_cvt_scalar_params;
      f32_to_qs8_cvt_config.element_tile = 128;
    } else if (hardware_config->use_x86_avx2) {
      f32_to_qs8_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_qs8_vcvt_ukernel__avx2_u64;
      f32_to_qs8_cvt_config.init.f32_qs8_cvt = xnn_init_f32_qs8_cvt_scalar_params;
      f32_to_qs8_cvt_config.element_tile = 64;
    } else if (hardware_config->use_x86_avx) {
      f32_to_qs8_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_qs8_vcvt_ukernel__avx_u32;
      f32_to_qs8_cvt_config.init.f32_qs8_cvt = xnn_init_f32_qs8_cvt_scalar_params;
      f32_to_qs8_cvt_config.element_tile = 32;
    } else if (hardware_config->use_x86_sse4_1) {
      f32_to_qs8_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_qs8_vcvt_ukernel__sse41_u32;
      f32_to_qs8_cvt_config.init.f32_qs8_cvt = xnn_init_f32_qs8_cvt_scalar_params;
      f32_to_qs8_cvt_config.element_tile = 32;
    } else {
      f32_to_qs8_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_qs8_vcvt_ukernel__sse2_u32;
      f32_to_qs8_cvt_config.init.f32_qs8_cvt = xnn_init_f32_qs8_cvt_scalar_params;
      f32_to_qs8_cvt_config.element_tile = 32;
      }
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    f32_to_qs8_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_qs8_vcvt_ukernel__wasmsimd_magic_u32;
    f32_to_qs8_cvt_config.init.f32_qs8_cvt = xnn_init_f32_qs8_cvt_scalar_params;
    f32_to_qs8_cvt_config.element_tile = 32;
  #elif XNN_ARCH_WASM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->is_x86) {
      f32_to_qs8_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_qs8_vcvt_ukernel__scalar_imagic_u1;
      f32_to_qs8_cvt_config.init.f32_qs8_cvt = xnn_init_f32_qs8_cvt_scalar_params;
      f32_to_qs8_cvt_config.element_tile = 1;
    } else {
      f32_to_qs8_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_qs8_vcvt_ukernel__wasm_fmagic_u4;
      f32_to_qs8_cvt_config.init.f32_qs8_cvt = xnn_init_f32_qs8_cvt_scalar_params;
      f32_to_qs8_cvt_config.element_tile = 4;
    }
  #elif XNN_ARCH_RISCV
    f32_to_qs8_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_qs8_vcvt_ukernel__scalar_lrintf_u4;
    f32_to_qs8_cvt_config.init.f32_qs8_cvt = xnn_init_f32_qs8_cvt_scalar_params;
    f32_to_qs8_cvt_config.element_tile = 4;
  #else
    f32_to_qs8_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_qs8_vcvt_ukernel__scalar_lrintf_u4;
    f32_to_qs8_cvt_config.init.f32_qs8_cvt = xnn_init_f32_qs8_cvt_scalar_params;
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
        f32_to_qu8_cvt_config.init.f32_qu8_cvt = xnn_init_f32_qu8_cvt_scalar_params;
        f32_to_qu8_cvt_config.element_tile = 32;
      } else {
        f32_to_qu8_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_qu8_vcvt_ukernel__neon_u32;
        f32_to_qu8_cvt_config.init.f32_qu8_cvt = xnn_init_f32_qu8_cvt_scalar_params;
        f32_to_qu8_cvt_config.element_tile = 32;
      }
    } else if (!XNN_PLATFORM_MOBILE) {
      f32_to_qu8_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_qu8_vcvt_ukernel__scalar_imagic_u4;
      f32_to_qu8_cvt_config.init.f32_qu8_cvt = xnn_init_f32_qu8_cvt_scalar_params;
      f32_to_qu8_cvt_config.element_tile = 4;
    }
  #elif XNN_ARCH_ARM64
    f32_to_qu8_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_qu8_vcvt_ukernel__neonv8_u32;
    f32_to_qu8_cvt_config.init.f32_qu8_cvt = xnn_init_f32_qu8_cvt_scalar_params;
    f32_to_qu8_cvt_config.element_tile = 32;
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_x86_avx512skx) {
      f32_to_qu8_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_qu8_vcvt_ukernel__avx512skx_u128;
      f32_to_qu8_cvt_config.init.f32_qu8_cvt = xnn_init_f32_qu8_cvt_scalar_params;
      f32_to_qu8_cvt_config.element_tile = 128;
    } else if (hardware_config->use_x86_avx2) {
      f32_to_qu8_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_qu8_vcvt_ukernel__avx2_u64;
      f32_to_qu8_cvt_config.init.f32_qu8_cvt = xnn_init_f32_qu8_cvt_scalar_params;
      f32_to_qu8_cvt_config.element_tile = 64;
    } else if (hardware_config->use_x86_avx) {
      f32_to_qu8_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_qu8_vcvt_ukernel__avx_u32;
      f32_to_qu8_cvt_config.init.f32_qu8_cvt = xnn_init_f32_qu8_cvt_scalar_params;
      f32_to_qu8_cvt_config.element_tile = 32;
    } else {
      f32_to_qu8_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_qu8_vcvt_ukernel__sse2_u32;
      f32_to_qu8_cvt_config.init.f32_qu8_cvt = xnn_init_f32_qu8_cvt_scalar_params;
      f32_to_qu8_cvt_config.element_tile = 32;
    }
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    f32_to_qu8_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_qu8_vcvt_ukernel__wasmsimd_magic_u32;
    f32_to_qu8_cvt_config.init.f32_qu8_cvt = xnn_init_f32_qu8_cvt_scalar_params;
    f32_to_qu8_cvt_config.element_tile = 32;
  #elif XNN_ARCH_WASM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->is_x86) {
      f32_to_qu8_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_qu8_vcvt_ukernel__scalar_imagic_u1;
      f32_to_qu8_cvt_config.init.f32_qu8_cvt = xnn_init_f32_qu8_cvt_scalar_params;
      f32_to_qu8_cvt_config.element_tile = 1;
    } else {
      f32_to_qu8_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_qu8_vcvt_ukernel__wasm_fmagic_u4;
      f32_to_qu8_cvt_config.init.f32_qu8_cvt = xnn_init_f32_qu8_cvt_scalar_params;
      f32_to_qu8_cvt_config.element_tile = 4;
    }
  #elif XNN_ARCH_RISCV
    f32_to_qu8_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_qu8_vcvt_ukernel__scalar_lrintf_u4;
    f32_to_qu8_cvt_config.init.f32_qu8_cvt = xnn_init_f32_qu8_cvt_scalar_params;
    f32_to_qu8_cvt_config.element_tile = 4;
  #else
    f32_to_qu8_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_f32_qu8_vcvt_ukernel__scalar_lrintf_u4;
    f32_to_qu8_cvt_config.init.f32_qu8_cvt = xnn_init_f32_qu8_cvt_scalar_params;
    f32_to_qu8_cvt_config.element_tile = 4;
  #endif
}

static void init_qs8_cvt_config(void) {
  #if XNN_ARCH_ARM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon_v8) {
      qs8_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qs8_vcvt_ukernel__neon_u32;
      qs8_cvt_config.init.qs8_cvt = xnn_init_qs8_cvt_scalar_params;
      qs8_cvt_config.element_tile = 32;
    } else {
      qs8_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qs8_vcvt_ukernel__armsimd32_u8;
      qs8_cvt_config.init.qs8_cvt = xnn_init_qs8_cvt_scalar_params;
      qs8_cvt_config.element_tile = 8;
    }
  #elif XNN_ARCH_ARM64
    qs8_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qs8_vcvt_ukernel__neon_u32;
    qs8_cvt_config.init.qs8_cvt = xnn_init_qs8_cvt_scalar_params;
    qs8_cvt_config.element_tile = 32;
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_x86_avx2) {
      qs8_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qs8_vcvt_ukernel__avx2_u32;
      qs8_cvt_config.init.qs8_cvt = xnn_init_qs8_cvt_scalar_params;
      qs8_cvt_config.element_tile = 32;
    } else if (hardware_config->use_x86_avx) {
      qs8_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qs8_vcvt_ukernel__avx_u32;
      qs8_cvt_config.init.qs8_cvt = xnn_init_qs8_cvt_scalar_params;
      qs8_cvt_config.element_tile = 32;
    } else if (hardware_config->use_x86_sse4_1) {
      qs8_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qs8_vcvt_ukernel__sse41_u32;
      qs8_cvt_config.init.qs8_cvt = xnn_init_qs8_cvt_scalar_params;
      qs8_cvt_config.element_tile = 32;
    } else if (hardware_config->use_x86_ssse3) {
      qs8_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qs8_vcvt_ukernel__ssse3_u32;
      qs8_cvt_config.init.qs8_cvt = xnn_init_qs8_cvt_scalar_params;
      qs8_cvt_config.element_tile = 32;
    } else {
      qs8_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qs8_vcvt_ukernel__sse2_u32;
      qs8_cvt_config.init.qs8_cvt = xnn_init_qs8_cvt_scalar_params;
      qs8_cvt_config.element_tile = 32;
    }
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    #if XNN_ARCH_WASMRELAXEDSIMD
      qs8_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qs8_vcvt_ukernel__wasmrelaxedsimd_u32;
      qs8_cvt_config.init.qs8_cvt = xnn_init_qs8_cvt_scalar_params;
      qs8_cvt_config.element_tile = 32;
    #else
      qs8_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qs8_vcvt_ukernel__wasmsimd_u16;
      qs8_cvt_config.init.qs8_cvt = xnn_init_qs8_cvt_scalar_params;
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
  #else
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
      qs16_to_qs8_cvt_config.init.qs16_qs8_cvt = xnn_init_qs16_qs8_cvt_scalar_params;
      qs16_to_qs8_cvt_config.element_tile = 16;
    } else if (!XNN_PLATFORM_MOBILE) {
      qs16_to_qs8_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qs16_qs8_vcvt_ukernel__scalar_u4;
      qs16_to_qs8_cvt_config.init.qs16_qs8_cvt = xnn_init_qs16_qs8_cvt_scalar_params;
      qs16_to_qs8_cvt_config.element_tile = 4;
    }
  #elif XNN_ARCH_ARM64
    qs16_to_qs8_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qs16_qs8_vcvt_ukernel__neon_u32;
    qs16_to_qs8_cvt_config.init.qs16_qs8_cvt = xnn_init_qs16_qs8_cvt_scalar_params;
    qs16_to_qs8_cvt_config.element_tile = 32;
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_x86_avx) {
      qs16_to_qs8_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qs16_qs8_vcvt_ukernel__avx_u16;
      qs16_to_qs8_cvt_config.init.qs16_qs8_cvt = xnn_init_qs16_qs8_cvt_scalar_params;
      qs16_to_qs8_cvt_config.element_tile = 16;
    } else if (hardware_config->use_x86_sse4_1) {
      qs16_to_qs8_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qs16_qs8_vcvt_ukernel__sse41_u16;
      qs16_to_qs8_cvt_config.init.qs16_qs8_cvt = xnn_init_qs16_qs8_cvt_scalar_params;
      qs16_to_qs8_cvt_config.element_tile = 16;
    } else if (hardware_config->use_x86_ssse3) {
      qs16_to_qs8_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qs16_qs8_vcvt_ukernel__ssse3_u16;
      qs16_to_qs8_cvt_config.init.qs16_qs8_cvt = xnn_init_qs16_qs8_cvt_scalar_params;
      qs16_to_qs8_cvt_config.element_tile = 16;
    } else {
      qs16_to_qs8_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qs16_qs8_vcvt_ukernel__sse2_u16;
      qs16_to_qs8_cvt_config.init.qs16_qs8_cvt = xnn_init_qs16_qs8_cvt_scalar_params;
      qs16_to_qs8_cvt_config.element_tile = 16;
    }
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    qs16_to_qs8_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qs16_qs8_vcvt_ukernel__wasmsimd_u16;
    qs16_to_qs8_cvt_config.init.qs16_qs8_cvt = xnn_init_qs16_qs8_cvt_scalar_params;
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
      qs8_lrelu_config.init.qs8_lrelu = xnn_init_qs8_lrelu_scalar_params;
      qs8_lrelu_config.element_tile = 32;
    } else if (!XNN_PLATFORM_MOBILE) {
      qs8_lrelu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qs8_vlrelu_ukernel__armsimd32_u4;
      qs8_lrelu_config.init.qs8_lrelu = xnn_init_qs8_lrelu_scalar_params;
      qs8_lrelu_config.element_tile = 4;
    }
  #elif XNN_ARCH_ARM64
    qs8_lrelu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qs8_vlrelu_ukernel__neon_u32;
    qs8_lrelu_config.init.qs8_lrelu = xnn_init_qs8_lrelu_scalar_params;
    qs8_lrelu_config.element_tile = 32;
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_x86_avx2) {
      qs8_lrelu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qs8_vlrelu_ukernel__avx2_u32;
      qs8_lrelu_config.init.qs8_lrelu = xnn_init_qs8_lrelu_scalar_params;
      qs8_lrelu_config.element_tile = 32;
    } else if (hardware_config->use_x86_avx) {
      qs8_lrelu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qs8_vlrelu_ukernel__avx_u32;
      qs8_lrelu_config.init.qs8_lrelu = xnn_init_qs8_lrelu_scalar_params;
      qs8_lrelu_config.element_tile = 32;
    } else if (hardware_config->use_x86_sse4_1) {
      qs8_lrelu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qs8_vlrelu_ukernel__sse41_u32;
      qs8_lrelu_config.init.qs8_lrelu = xnn_init_qs8_lrelu_scalar_params;
      qs8_lrelu_config.element_tile = 32;
    } else if (hardware_config->use_x86_sse4_1) {
      qs8_lrelu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qs8_vlrelu_ukernel__ssse3_u32;
      qs8_lrelu_config.init.qs8_lrelu = xnn_init_qs8_lrelu_scalar_params;
      qs8_lrelu_config.element_tile = 32;
    } else {
      qs8_lrelu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qs8_vlrelu_ukernel__sse2_u32;
      qs8_lrelu_config.init.qs8_lrelu = xnn_init_qs8_lrelu_scalar_params;
      qs8_lrelu_config.element_tile = 32;
    }
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    #if XNN_ARCH_WASMRELAXEDSIMD
      if (hardware_config->is_x86) {
        qs8_lrelu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qs8_vlrelu_ukernel__wasmrelaxedsimd_x86_u32;
        qs8_lrelu_config.init.qs8_lrelu = xnn_init_qs8_lrelu_scalar_params;
        qs8_lrelu_config.element_tile = 32;
      } else {
        qs8_lrelu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qs8_vlrelu_ukernel__wasmrelaxedsimd_arm_u32;
        qs8_lrelu_config.init.qs8_lrelu = xnn_init_qs8_lrelu_scalar_params;
        qs8_lrelu_config.element_tile = 32;
      }
    #else
      if (hardware_config->is_x86) {
        qs8_lrelu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qs8_vlrelu_ukernel__wasmsimd_x86_u16;
        qs8_lrelu_config.init.qs8_lrelu = xnn_init_qs8_lrelu_scalar_params;
        qs8_lrelu_config.element_tile = 16;
      } else {
        qs8_lrelu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qs8_vlrelu_ukernel__wasmsimd_arm_u32;
        qs8_lrelu_config.init.qs8_lrelu = xnn_init_qs8_lrelu_scalar_params;
        qs8_lrelu_config.element_tile = 32;
      }
    #endif
  #elif XNN_ARCH_WASM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->is_x86) {
      qs8_lrelu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qs8_vlrelu_ukernel__scalar_select_u4;
      qs8_lrelu_config.init.qs8_lrelu = xnn_init_qs8_lrelu_scalar_params;
      qs8_lrelu_config.element_tile = 4;
    } else {
      qs8_lrelu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qs8_vlrelu_ukernel__scalar_andxor_u4;
      qs8_lrelu_config.init.qs8_lrelu = xnn_init_qs8_lrelu_scalar_params;
      qs8_lrelu_config.element_tile = 4;
    }
  #elif XNN_ARCH_RISCV
    qs8_lrelu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qs8_vlrelu_ukernel__scalar_andxor_u4;
    qs8_lrelu_config.init.qs8_lrelu = xnn_init_qs8_lrelu_scalar_params;
    qs8_lrelu_config.element_tile = 4;
  #else
    qs8_lrelu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qs8_vlrelu_ukernel__scalar_andxor_u4;
    qs8_lrelu_config.init.qs8_lrelu = xnn_init_qs8_lrelu_scalar_params;
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
      qs8_to_f32_cvt_config.init.qs8_f32_cvt = xnn_init_qs8_f32_cvt_scalar_params;
      qs8_to_f32_cvt_config.element_tile = 32;
    } else if (!XNN_PLATFORM_MOBILE) {
      qs8_to_f32_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qs8_f32_vcvt_ukernel__scalar_u4;
      qs8_to_f32_cvt_config.init.qs8_f32_cvt = xnn_init_qs8_f32_cvt_scalar_params;
      qs8_to_f32_cvt_config.element_tile = 4;
    }
  #elif XNN_ARCH_ARM64
    qs8_to_f32_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qs8_f32_vcvt_ukernel__neon_u32;
    qs8_to_f32_cvt_config.init.qs8_f32_cvt = xnn_init_qs8_f32_cvt_scalar_params;
    qs8_to_f32_cvt_config.element_tile = 32;
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_x86_avx512skx) {
      qs8_to_f32_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qs8_f32_vcvt_ukernel__avx512skx_u32;
      qs8_to_f32_cvt_config.init.qs8_f32_cvt = xnn_init_qs8_f32_cvt_scalar_params;
      qs8_to_f32_cvt_config.element_tile = 32;
    } else if (hardware_config->use_x86_avx2) {
      qs8_to_f32_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qs8_f32_vcvt_ukernel__avx2_u16;
      qs8_to_f32_cvt_config.init.qs8_f32_cvt = xnn_init_qs8_f32_cvt_scalar_params;
      qs8_to_f32_cvt_config.element_tile = 16;
    } else if (hardware_config->use_x86_avx) {
      qs8_to_f32_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qs8_f32_vcvt_ukernel__avx_u32;
      qs8_to_f32_cvt_config.init.qs8_f32_cvt = xnn_init_qs8_f32_cvt_scalar_params;
      qs8_to_f32_cvt_config.element_tile = 32;
    } else if (hardware_config->use_x86_sse4_1) {
      qs8_to_f32_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qs8_f32_vcvt_ukernel__sse41_u16;
      qs8_to_f32_cvt_config.init.qs8_f32_cvt = xnn_init_qs8_f32_cvt_scalar_params;
      qs8_to_f32_cvt_config.element_tile = 16;
    } else {
      qs8_to_f32_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qs8_f32_vcvt_ukernel__sse2_u32;
      qs8_to_f32_cvt_config.init.qs8_f32_cvt = xnn_init_qs8_f32_cvt_scalar_params;
      qs8_to_f32_cvt_config.element_tile = 32;
    }
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    qs8_to_f32_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qs8_f32_vcvt_ukernel__wasmsimd_u32;
    qs8_to_f32_cvt_config.init.qs8_f32_cvt = xnn_init_qs8_f32_cvt_scalar_params;
    qs8_to_f32_cvt_config.element_tile = 32;
  #elif XNN_ARCH_WASM
    qs8_to_f32_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qs8_f32_vcvt_ukernel__scalar_u1;
    qs8_to_f32_cvt_config.init.qs8_f32_cvt = xnn_init_qs8_f32_cvt_scalar_params;
    qs8_to_f32_cvt_config.element_tile = 1;
  #elif XNN_ARCH_RISCV
    qs8_to_f32_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qs8_f32_vcvt_ukernel__scalar_u4;
    qs8_to_f32_cvt_config.init.qs8_f32_cvt = xnn_init_qs8_f32_cvt_scalar_params;
    qs8_to_f32_cvt_config.element_tile = 4;
  #else
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
      qu8_cvt_config.init.qu8_cvt = xnn_init_qu8_cvt_scalar_params;
      qu8_cvt_config.element_tile = 32;
    } else if (!XNN_PLATFORM_MOBILE) {
      qu8_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qu8_vcvt_ukernel__armsimd32_u8;
      qu8_cvt_config.init.qu8_cvt = xnn_init_qu8_cvt_scalar_params;
      qu8_cvt_config.element_tile = 8;
    }
  #elif XNN_ARCH_ARM64
    qu8_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qu8_vcvt_ukernel__neon_u32;
    qu8_cvt_config.init.qu8_cvt = xnn_init_qu8_cvt_scalar_params;
    qu8_cvt_config.element_tile = 32;
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_x86_avx2) {
      qu8_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qu8_vcvt_ukernel__avx2_u32;
      qu8_cvt_config.init.qu8_cvt = xnn_init_qu8_cvt_scalar_params;
      qu8_cvt_config.element_tile = 32;
    } else if (hardware_config->use_x86_avx) {
      qu8_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qu8_vcvt_ukernel__avx_u32;
      qu8_cvt_config.init.qu8_cvt = xnn_init_qu8_cvt_scalar_params;
      qu8_cvt_config.element_tile = 32;
    } else if (hardware_config->use_x86_sse4_1) {
      qu8_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qu8_vcvt_ukernel__sse41_u32;
      qu8_cvt_config.init.qu8_cvt = xnn_init_qu8_cvt_scalar_params;
      qu8_cvt_config.element_tile = 32;
    } else if (hardware_config->use_x86_ssse3) {
      qu8_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qu8_vcvt_ukernel__ssse3_u32;
      qu8_cvt_config.init.qu8_cvt = xnn_init_qu8_cvt_scalar_params;
      qu8_cvt_config.element_tile = 32;
    } else {
      qu8_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qu8_vcvt_ukernel__sse2_u32;
      qu8_cvt_config.init.qu8_cvt = xnn_init_qu8_cvt_scalar_params;
      qu8_cvt_config.element_tile = 32;
    }
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    #if XNN_ARCH_WASMRELAXEDSIMD
      qu8_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qu8_vcvt_ukernel__wasmrelaxedsimd_u32;
      qu8_cvt_config.init.qu8_cvt = xnn_init_qu8_cvt_scalar_params;
      qu8_cvt_config.element_tile = 32;
    #else
      qu8_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qu8_vcvt_ukernel__wasmsimd_u16;
      qu8_cvt_config.init.qu8_cvt = xnn_init_qu8_cvt_scalar_params;
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
  #else
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
      qu8_lrelu_config.init.qu8_lrelu = xnn_init_qu8_lrelu_scalar_params;
      qu8_lrelu_config.element_tile = 32;
    } else if (!XNN_PLATFORM_MOBILE) {
      qu8_lrelu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qu8_vlrelu_ukernel__armsimd32_u4;
      qu8_lrelu_config.init.qu8_lrelu = xnn_init_qu8_lrelu_scalar_params;
      qu8_lrelu_config.element_tile = 4;
    }
  #elif XNN_ARCH_ARM64
    qu8_lrelu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qu8_vlrelu_ukernel__neon_u32;
    qu8_lrelu_config.init.qu8_lrelu = xnn_init_qu8_lrelu_scalar_params;
    qu8_lrelu_config.element_tile = 32;
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_x86_avx2) {
      qu8_lrelu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qu8_vlrelu_ukernel__avx2_u32;
      qu8_lrelu_config.init.qu8_lrelu = xnn_init_qu8_lrelu_scalar_params;
      qu8_lrelu_config.element_tile = 32;
    } else if (hardware_config->use_x86_avx) {
      qu8_lrelu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qu8_vlrelu_ukernel__avx_u32;
      qu8_lrelu_config.init.qu8_lrelu = xnn_init_qu8_lrelu_scalar_params;
      qu8_lrelu_config.element_tile = 32;
    } else if (hardware_config->use_x86_sse4_1) {
      qu8_lrelu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qu8_vlrelu_ukernel__sse41_u32;
      qu8_lrelu_config.init.qu8_lrelu = xnn_init_qu8_lrelu_scalar_params;
      qu8_lrelu_config.element_tile = 32;
    } else if (hardware_config->use_x86_sse4_1) {
      qu8_lrelu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qu8_vlrelu_ukernel__ssse3_u32;
      qu8_lrelu_config.init.qu8_lrelu = xnn_init_qu8_lrelu_scalar_params;
      qu8_lrelu_config.element_tile = 32;
    } else {
      qu8_lrelu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qu8_vlrelu_ukernel__sse2_u32;
      qu8_lrelu_config.init.qu8_lrelu = xnn_init_qu8_lrelu_scalar_params;
      qu8_lrelu_config.element_tile = 32;
    }
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    #if XNN_ARCH_WASMRELAXEDSIMD
      if (hardware_config->is_x86) {
        qu8_lrelu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qu8_vlrelu_ukernel__wasmrelaxedsimd_x86_u32;
        qu8_lrelu_config.init.qu8_lrelu = xnn_init_qu8_lrelu_scalar_params;
        qu8_lrelu_config.element_tile = 32;
      } else {
        qu8_lrelu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qu8_vlrelu_ukernel__wasmrelaxedsimd_arm_u32;
        qu8_lrelu_config.init.qu8_lrelu = xnn_init_qu8_lrelu_scalar_params;
        qu8_lrelu_config.element_tile = 32;
      }
    #else
      if (hardware_config->is_x86) {
        qu8_lrelu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qu8_vlrelu_ukernel__wasmsimd_x86_u16;
        qu8_lrelu_config.init.qu8_lrelu = xnn_init_qu8_lrelu_scalar_params;
        qu8_lrelu_config.element_tile = 16;
      } else {
        qu8_lrelu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qu8_vlrelu_ukernel__wasmsimd_arm_u32;
        qu8_lrelu_config.init.qu8_lrelu = xnn_init_qu8_lrelu_scalar_params;
        qu8_lrelu_config.element_tile = 32;
      }
    #endif
  #elif XNN_ARCH_WASM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->is_x86) {
      qu8_lrelu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qu8_vlrelu_ukernel__scalar_select_u4;
      qu8_lrelu_config.init.qu8_lrelu = xnn_init_qu8_lrelu_scalar_params;
      qu8_lrelu_config.element_tile = 4;
    } else {
      qu8_lrelu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qu8_vlrelu_ukernel__scalar_andxor_u4;
      qu8_lrelu_config.init.qu8_lrelu = xnn_init_qu8_lrelu_scalar_params;
      qu8_lrelu_config.element_tile = 4;
    }
  #elif XNN_ARCH_RISCV
    qu8_lrelu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qu8_vlrelu_ukernel__scalar_andxor_u4;
    qu8_lrelu_config.init.qu8_lrelu = xnn_init_qu8_lrelu_scalar_params;
    qu8_lrelu_config.element_tile = 4;
  #else
    qu8_lrelu_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qu8_vlrelu_ukernel__scalar_andxor_u4;
    qu8_lrelu_config.init.qu8_lrelu = xnn_init_qu8_lrelu_scalar_params;
    qu8_lrelu_config.element_tile = 4;
  #endif
}

static void init_qu8_to_f32_cvt_config(void) {
  #if XNN_ARCH_ARM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_arm_neon) {
      qu8_to_f32_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qu8_f32_vcvt_ukernel__neon_u32;
      qu8_to_f32_cvt_config.init.qu8_f32_cvt = xnn_init_qu8_f32_cvt_scalar_params;
      qu8_to_f32_cvt_config.element_tile = 32;
    } else if (!XNN_PLATFORM_MOBILE) {
      qu8_to_f32_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qu8_f32_vcvt_ukernel__scalar_u4;
      qu8_to_f32_cvt_config.init.qu8_f32_cvt = xnn_init_qu8_f32_cvt_scalar_params;
      qu8_to_f32_cvt_config.element_tile = 4;
    }
  #elif XNN_ARCH_ARM64
    qu8_to_f32_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qu8_f32_vcvt_ukernel__neon_u32;
    qu8_to_f32_cvt_config.init.qu8_f32_cvt = xnn_init_qu8_f32_cvt_scalar_params;
    qu8_to_f32_cvt_config.element_tile = 32;
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_x86_avx512skx) {
      qu8_to_f32_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qu8_f32_vcvt_ukernel__avx512skx_u32;
      qu8_to_f32_cvt_config.init.qu8_f32_cvt = xnn_init_qu8_f32_cvt_scalar_params;
      qu8_to_f32_cvt_config.element_tile = 32;
    } else if (hardware_config->use_x86_avx2) {
      qu8_to_f32_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qu8_f32_vcvt_ukernel__avx2_u16;
      qu8_to_f32_cvt_config.init.qu8_f32_cvt = xnn_init_qu8_f32_cvt_scalar_params;
      qu8_to_f32_cvt_config.element_tile = 16;
    } else if (hardware_config->use_x86_avx) {
      qu8_to_f32_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qu8_f32_vcvt_ukernel__avx_u32;
      qu8_to_f32_cvt_config.init.qu8_f32_cvt = xnn_init_qu8_f32_cvt_scalar_params;
      qu8_to_f32_cvt_config.element_tile = 32;
    } else if (hardware_config->use_x86_sse4_1) {
      qu8_to_f32_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qu8_f32_vcvt_ukernel__sse41_u16;
      qu8_to_f32_cvt_config.init.qu8_f32_cvt = xnn_init_qu8_f32_cvt_scalar_params;
      qu8_to_f32_cvt_config.element_tile = 16;
    } else {
      qu8_to_f32_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qu8_f32_vcvt_ukernel__sse2_u32;
      qu8_to_f32_cvt_config.init.qu8_f32_cvt = xnn_init_qu8_f32_cvt_scalar_params;
      qu8_to_f32_cvt_config.element_tile = 32;
    }
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    qu8_to_f32_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qu8_f32_vcvt_ukernel__wasmsimd_u32;
    qu8_to_f32_cvt_config.init.qu8_f32_cvt = xnn_init_qu8_f32_cvt_scalar_params;
    qu8_to_f32_cvt_config.element_tile = 32;
  #elif XNN_ARCH_WASM
    qu8_to_f32_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qu8_f32_vcvt_ukernel__scalar_u1;
    qu8_to_f32_cvt_config.init.qu8_f32_cvt = xnn_init_qu8_f32_cvt_scalar_params;
    qu8_to_f32_cvt_config.element_tile = 1;
  #elif XNN_ARCH_RISCV
    qu8_to_f32_cvt_config.ukernel = (xnn_vunary_ukernel_fn) xnn_qu8_f32_vcvt_ukernel__scalar_u4;
    qu8_to_f32_cvt_config.init.qu8_f32_cvt = xnn_init_qu8_f32_cvt_scalar_params;
    qu8_to_f32_cvt_config.element_tile = 4;
  #else
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
      s8_clamp_config.init.s8_minmax = xnn_init_s8_minmax_scalar_params;
      s8_clamp_config.element_tile = 64;
    } else if (!XNN_PLATFORM_MOBILE) {
      s8_clamp_config.ukernel = (xnn_vunary_ukernel_fn) xnn_s8_vclamp_ukernel__scalar_u4;
      s8_clamp_config.init.s8_minmax = xnn_init_s8_minmax_scalar_params;
      s8_clamp_config.element_tile = 4;
    }
  #elif XNN_ARCH_ARM64
    s8_clamp_config.ukernel = (xnn_vunary_ukernel_fn) xnn_s8_vclamp_ukernel__neon_u64;
    s8_clamp_config.init.s8_minmax = xnn_init_s8_minmax_scalar_params;
    s8_clamp_config.element_tile = 64;
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    if (hardware_config->use_x86_sse4_1) {
      s8_clamp_config.ukernel = (xnn_vunary_ukernel_fn) xnn_s8_vclamp_ukernel__sse41_u64;
      s8_clamp_config.init.s8_minmax = xnn_init_s8_minmax_scalar_params;
      s8_clamp_config.element_tile = 64;
    } else {
      s8_clamp_config.ukernel = (xnn_vunary_ukernel_fn) xnn_s8_vclamp_ukernel__sse2_u64;
      s8_clamp_config.init.s8_minmax = xnn_init_s8_minmax_scalar_params;
      s8_clamp_config.element_tile = 64;
    }
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    s8_clamp_config.ukernel = (xnn_vunary_ukernel_fn) xnn_s8_vclamp_ukernel__wasmsimd_u64;
    s8_clamp_config.init.s8_minmax = xnn_init_s8_minmax_scalar_params;
    s8_clamp_config.element_tile = 64;
  #elif XNN_ARCH_WASM
    s8_clamp_config.ukernel = (xnn_vunary_ukernel_fn) xnn_s8_vclamp_ukernel__scalar_u4;
    s8_clamp_config.init.s8_minmax = xnn_init_s8_minmax_scalar_params;
    s8_clamp_config.element_tile = 4;
  #elif XNN_ARCH_RISCV
    s8_clamp_config.ukernel = (xnn_vunary_ukernel_fn) xnn_s8_vclamp_ukernel__scalar_u4;
    s8_clamp_config.init.s8_minmax = xnn_init_s8_minmax_scalar_params;
    s8_clamp_config.element_tile = 4;
  #else
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
      u8_clamp_config.init.u8_minmax = xnn_init_u8_minmax_scalar_params;
      u8_clamp_config.element_tile = 64;
    } else if (!XNN_PLATFORM_MOBILE) {
      u8_clamp_config.ukernel = (xnn_vunary_ukernel_fn) xnn_u8_vclamp_ukernel__scalar_u4;
      u8_clamp_config.init.u8_minmax = xnn_init_u8_minmax_scalar_params;
      u8_clamp_config.element_tile = 4;
    }
  #elif XNN_ARCH_ARM64
    u8_clamp_config.ukernel = (xnn_vunary_ukernel_fn) xnn_u8_vclamp_ukernel__neon_u64;
    u8_clamp_config.init.u8_minmax = xnn_init_u8_minmax_scalar_params;
    u8_clamp_config.element_tile = 64;
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    u8_clamp_config.ukernel = (xnn_vunary_ukernel_fn) xnn_u8_vclamp_ukernel__sse2_u64;
    u8_clamp_config.init.u8_minmax = xnn_init_u8_minmax_scalar_params;
    u8_clamp_config.element_tile = 64;
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    u8_clamp_config.ukernel = (xnn_vunary_ukernel_fn) xnn_u8_vclamp_ukernel__wasmsimd_u64;
    u8_clamp_config.init.u8_minmax = xnn_init_u8_minmax_scalar_params;
    u8_clamp_config.element_tile = 64;
  #elif XNN_ARCH_WASM
    u8_clamp_config.ukernel = (xnn_vunary_ukernel_fn) xnn_u8_vclamp_ukernel__scalar_u4;
    u8_clamp_config.init.u8_minmax = xnn_init_u8_minmax_scalar_params;
    u8_clamp_config.element_tile = 4;
  #elif XNN_ARCH_RISCV
    u8_clamp_config.ukernel = (xnn_vunary_ukernel_fn) xnn_u8_vclamp_ukernel__scalar_u4;
    u8_clamp_config.init.u8_minmax = xnn_init_u8_minmax_scalar_params;
    u8_clamp_config.element_tile = 4;
  #else
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
  #else
    xx_copy_config.ukernel = (xnn_vunary_ukernel_fn) xnn_xx_copy_ukernel__scalar_memcpy;
    xx_copy_config.element_tile = 1;
  #endif
}

const struct xnn_unary_elementwise_config* xnn_init_f16_abs_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL || !xnn_is_f16_compatible_config(hardware_config)) {
    return NULL;
  }
  XNN_INIT_ONCE(f16_abs);
  return &f16_abs_config;
}

const struct xnn_unary_elementwise_config* xnn_init_f16_clamp_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL || !xnn_is_f16_compatible_config(hardware_config)) {
    return NULL;
  }
  XNN_INIT_ONCE(f16_clamp);
  return &f16_clamp_config;
}

const struct xnn_unary_elementwise_config* xnn_init_f16_elu_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL || !xnn_is_f16_compatible_config(hardware_config)) {
    return NULL;
  }
  XNN_INIT_ONCE(f16_elu);
  return &f16_elu_config;
}

const struct xnn_unary_elementwise_config* xnn_init_f16_hswish_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL || !xnn_is_f16_compatible_config(hardware_config)) {
    return NULL;
  }
  XNN_INIT_ONCE(f16_hswish);
  return &f16_hswish_config;
}

const struct xnn_unary_elementwise_config* xnn_init_f16_lrelu_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL || !xnn_is_f16_compatible_config(hardware_config)) {
    return NULL;
  }
  XNN_INIT_ONCE(f16_lrelu);
  return &f16_lrelu_config;
}

const struct xnn_unary_elementwise_config* xnn_init_f16_neg_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL || !xnn_is_f16_compatible_config(hardware_config)) {
    return NULL;
  }
  XNN_INIT_ONCE(f16_neg);
  return &f16_neg_config;
}

const struct xnn_unary_elementwise_config* xnn_init_f16_rndd_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL || !xnn_is_f16_compatible_config(hardware_config)) {
    return NULL;
  }
  XNN_INIT_ONCE(f16_rndd);
  return &f16_rndd_config;
}

const struct xnn_unary_elementwise_config* xnn_init_f16_rndne_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL || !xnn_is_f16_compatible_config(hardware_config)) {
    return NULL;
  }
  XNN_INIT_ONCE(f16_rndne);
  return &f16_rndne_config;
}

const struct xnn_unary_elementwise_config* xnn_init_f16_rndu_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL || !xnn_is_f16_compatible_config(hardware_config)) {
    return NULL;
  }
  XNN_INIT_ONCE(f16_rndu);
  return &f16_rndu_config;
}

const struct xnn_unary_elementwise_config* xnn_init_f16_rndz_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL || !xnn_is_f16_compatible_config(hardware_config)) {
    return NULL;
  }
  XNN_INIT_ONCE(f16_rndz);
  return &f16_rndz_config;
}

const struct xnn_unary_elementwise_config* xnn_init_f16_rsqrt_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL || !xnn_is_f16_compatible_config(hardware_config)) {
    return NULL;
  }
  XNN_INIT_ONCE(f16_rsqrt);
  return &f16_rsqrt_config;
}

const struct xnn_unary_elementwise_config* xnn_init_f16_sigmoid_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL || !xnn_is_f16_compatible_config(hardware_config)) {
    return NULL;
  }
  XNN_INIT_ONCE(f16_sigmoid);
  return &f16_sigmoid_config;
}

const struct xnn_unary_elementwise_config* xnn_init_f16_sqr_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL || !xnn_is_f16_compatible_config(hardware_config)) {
    return NULL;
  }
  XNN_INIT_ONCE(f16_sqr);
  return &f16_sqr_config;
}

const struct xnn_unary_elementwise_config* xnn_init_f16_sqrt_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL || !xnn_is_f16_compatible_config(hardware_config)) {
    return NULL;
  }
  XNN_INIT_ONCE(f16_sqrt);
  return &f16_sqrt_config;
}

const struct xnn_unary_elementwise_config* xnn_init_f16_tanh_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL || !xnn_is_f16_compatible_config(hardware_config)) {
    return NULL;
  }
  XNN_INIT_ONCE(f16_tanh);
  return &f16_tanh_config;
}

const struct xnn_unary_elementwise_config* xnn_init_f16_to_f32_cvt_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  XNN_INIT_ONCE(f16_to_f32_cvt);
  return &f16_to_f32_cvt_config;
}

const struct xnn_unary_elementwise_config* xnn_init_f16_to_qs8_cvt_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL || !xnn_is_f16_compatible_config(hardware_config)) {
    return NULL;
  }
  XNN_INIT_ONCE(f16_to_qs8_cvt);
  return &f16_to_qs8_cvt_config;
}

const struct xnn_unary_elementwise_config* xnn_init_f32_abs_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  XNN_INIT_ONCE(f32_abs);
  return &f32_abs_config;
}

const struct xnn_unary_elementwise_config* xnn_init_f32_clamp_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  XNN_INIT_ONCE(f32_clamp);
  return &f32_clamp_config;
}

const struct xnn_unary_elementwise_config* xnn_init_f32_elu_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  XNN_INIT_ONCE(f32_elu);
  return &f32_elu_config;
}

const struct xnn_unary_elementwise_config* xnn_init_f32_exp_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  XNN_INIT_ONCE(f32_exp);
  return &f32_exp_config;
}

const struct xnn_unary_elementwise_config* xnn_init_f32_gelu_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  XNN_INIT_ONCE(f32_gelu);
  return &f32_gelu_config;
}

const struct xnn_unary_elementwise_config* xnn_init_f32_hswish_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  XNN_INIT_ONCE(f32_hswish);
  return &f32_hswish_config;
}

const struct xnn_unary_elementwise_config* xnn_init_f32_log_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  XNN_INIT_ONCE(f32_log);
  return &f32_log_config;
}

const struct xnn_unary_elementwise_config* xnn_init_f32_lrelu_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  XNN_INIT_ONCE(f32_lrelu);
  return &f32_lrelu_config;
}

const struct xnn_unary_elementwise_config* xnn_init_f32_neg_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  XNN_INIT_ONCE(f32_neg);
  return &f32_neg_config;
}

const struct xnn_unary_elementwise_config* xnn_init_f32_relu_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  XNN_INIT_ONCE(f32_relu);
  return &f32_relu_config;
}

const struct xnn_unary_elementwise_config* xnn_init_f32_rndd_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  XNN_INIT_ONCE(f32_rndd);
  return &f32_rndd_config;
}

const struct xnn_unary_elementwise_config* xnn_init_f32_rndne_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  XNN_INIT_ONCE(f32_rndne);
  return &f32_rndne_config;
}

const struct xnn_unary_elementwise_config* xnn_init_f32_rndu_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  XNN_INIT_ONCE(f32_rndu);
  return &f32_rndu_config;
}

const struct xnn_unary_elementwise_config* xnn_init_f32_rndz_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  XNN_INIT_ONCE(f32_rndz);
  return &f32_rndz_config;
}

const struct xnn_unary_elementwise_config* xnn_init_f32_rsqrt_config() {
  const struct xnn_hardware_config* hardware_config =
      xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  XNN_INIT_ONCE(f32_rsqrt);
  return &f32_rsqrt_config;
}

const struct xnn_unary_elementwise_config* xnn_init_f32_sigmoid_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  XNN_INIT_ONCE(f32_sigmoid);
  return &f32_sigmoid_config;
}

const struct xnn_unary_elementwise_config* xnn_init_f32_sqr_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  XNN_INIT_ONCE(f32_sqr);
  return &f32_sqr_config;
}

const struct xnn_unary_elementwise_config* xnn_init_f32_sqrt_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  XNN_INIT_ONCE(f32_sqrt);
  return &f32_sqrt_config;
}

const struct xnn_unary_elementwise_config* xnn_init_f32_tanh_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  XNN_INIT_ONCE(f32_tanh);
  return &f32_tanh_config;
}

const struct xnn_unary_elementwise_config* xnn_init_f32_to_f16_cvt_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  XNN_INIT_ONCE(f32_to_f16_cvt);
  return &f32_to_f16_cvt_config;
}

const struct xnn_unary_elementwise_config* xnn_init_f32_to_qp8_cvt_config() {
  const struct xnn_hardware_config* hardware_config =
      xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  XNN_INIT_ONCE(f32_to_qp8_cvt);
  return &f32_to_qp8_cvt_config;
}

const struct xnn_unary_elementwise_config* xnn_init_f32_to_qs8_cvt_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  XNN_INIT_ONCE(f32_to_qs8_cvt);
  return &f32_to_qs8_cvt_config;
}

const struct xnn_unary_elementwise_config* xnn_init_f32_to_qu8_cvt_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  XNN_INIT_ONCE(f32_to_qu8_cvt);
  return &f32_to_qu8_cvt_config;
}

const struct xnn_unary_elementwise_config* xnn_init_qs8_cvt_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  XNN_INIT_ONCE(qs8_cvt);
  return &qs8_cvt_config;
}

const struct xnn_unary_elementwise_config* xnn_init_qs16_to_qs8_cvt_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  XNN_INIT_ONCE(qs16_to_qs8_cvt);
  return &qs16_to_qs8_cvt_config;
}

const struct xnn_unary_elementwise_config* xnn_init_qs8_lrelu_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  XNN_INIT_ONCE(qs8_lrelu);
  return &qs8_lrelu_config;
}

const struct xnn_unary_elementwise_config* xnn_init_qs8_to_f16_cvt_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL || !xnn_is_f16_compatible_config(hardware_config)) {
    return NULL;
  }
  XNN_INIT_ONCE(qs8_to_f16_cvt);
  return &qs8_to_f16_cvt_config;
}

const struct xnn_unary_elementwise_config* xnn_init_qs8_to_f32_cvt_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  XNN_INIT_ONCE(qs8_to_f32_cvt);
  return &qs8_to_f32_cvt_config;
}

const struct xnn_unary_elementwise_config* xnn_init_qu8_cvt_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  XNN_INIT_ONCE(qu8_cvt);
  return &qu8_cvt_config;
}

const struct xnn_unary_elementwise_config* xnn_init_qu8_lrelu_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  XNN_INIT_ONCE(qu8_lrelu);
  return &qu8_lrelu_config;
}

const struct xnn_unary_elementwise_config* xnn_init_qu8_to_f32_cvt_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  XNN_INIT_ONCE(qu8_to_f32_cvt);
  return &qu8_to_f32_cvt_config;
}

const struct xnn_unary_elementwise_config* xnn_init_s8_clamp_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  XNN_INIT_ONCE(s8_clamp);
  return &s8_clamp_config;
}

const struct xnn_unary_elementwise_config* xnn_init_u8_clamp_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  XNN_INIT_ONCE(u8_clamp);
  return &u8_clamp_config;
}

const struct xnn_unary_elementwise_config* xnn_init_xx_copy_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  XNN_INIT_ONCE(xx_copy);
  return &xx_copy_config;
}
