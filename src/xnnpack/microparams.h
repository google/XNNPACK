// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <stddef.h>
#include <stdint.h>

#include "xnnpack/common.h"
#include "xnnpack/math.h"

// Default: serves to differentiate pointer types for micro-kernels without fused activation.

struct xnn_f16_default_params {
  char _;  // Dummy member variable to comply with the C standard
};

struct xnn_bf16_default_params {
  char _;  // Dummy member variable to comply with the C standard
};

struct xnn_f32_default_params {
  char _;  // Dummy member variable to comply with the C standard
};

struct xnn_s32_default_params {
  char _;  // Dummy member variable to comply with the C standard
};


// ReLU: serves to differentiate pointer types for micro-kernels with fused ReLU activation.

struct xnn_f32_relu_params {
  char _;  // Dummy member variable to comply with the C standard
};


// Scale: used by RSUM microkernels

struct xnn_f16_scale_params {
  struct {
    xnn_float16 scale;
  } scalar;
};

struct xnn_f16_f32acc_scale_params {
  struct {
    float scale;
  } scalar;
};

struct xnn_f32_scale_params {
  struct {
    float scale;
  } scalar;
};


// Scale+Min+Max: used by AVGPOOL/GAVGPOOL microkernels.

struct xnn_f16_scaleminmax_params {
  struct {
    xnn_float16 scale;
    xnn_float16 min;
    xnn_float16 max;
  } scalar;
};

struct xnn_f32_scaleminmax_params {
  struct {
    float scale;
    float min;
    float max;
  } scalar;
};


// Min+Max: used by VCLAMP and GEMM/IGEMM/DWCONV/MAXPOOL/etc with MINMAX activation.

struct xnn_bf16_minmax_params {
  struct {
    float min;
    float max;
  } scalar;
};

union xnn_f16_minmax_params {
  struct {
    xnn_float16 min;
    xnn_float16 max;
  } scalar;
};

union xnn_f32_minmax_params {
  struct {
    float min;
    float max;
  } scalar;
};

struct xnn_f16_qc4w_minmax_params {
  struct {
    xnn_float16 min;
    xnn_float16 max;
  } scalar;
};

struct xnn_f16_qb4w_minmax_params {
  struct {
    xnn_float16 min;
    xnn_float16 max;
    size_t blocksize;
  } scalar;
};

struct xnn_f32_qc4w_minmax_params {
  struct {
    float min;
    float max;
    int32_t kernel_zero_point;
  } scalar;
};

struct xnn_f32_qb4w_minmax_params {
  struct {
    float min;
    float max;
    size_t blocksize;
  } scalar;
};

struct xnn_s8_minmax_params {
  struct {
    int32_t min;
    int32_t max;
  } scalar;
};

struct xnn_u8_minmax_params {
  struct {
    uint32_t min;
    uint32_t max;
  } scalar;
};


// Conv w. Min+Max: used by quantized GEMM/IGEMM/DWCONV microkernels with MINMAX activation.
struct xnn_qd8_quantization_params {
  int32_t zero_point;
  float inv_scale;
};

union xnn_qs8_conv_minmax_params {
  struct {
    float scale;
    int16_t output_zero_point;
    int16_t output_min;
    int16_t output_max;
  } fp32_scalar;
  struct {
    int32_t multiplier;
    uint32_t shift;
    int16_t output_min;
    int16_t output_max;
    int32_t output_zero_point;
    int64_t rounding;
  } rndnu_scalar;
#if XNN_ARCH_ARM
  struct {
    float scale;
    float magic_bias;
    int32_t magic_bias_less_zero_point;
    uint32_t output_min;
    uint32_t output_max;
  } fp32_armsimd32;
#endif  // XNN_ARCH_ARM
#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  struct {
    float scale;
    float magic_bias;
    int32_t magic_bias_less_output_zero_point;
    int8_t output_min;
    int8_t output_max;
  } fp32_neon;
  struct {
    float scale;
    int16_t output_zero_point;
    int8_t output_min;
    int8_t output_max;
  } fp32_neonv8;
  struct {
    int32_t right_pre_shift;
    int32_t multiplier;
    int32_t right_post_shift;
    int16_t output_zero_point;
    int8_t output_min;
    int8_t output_max;
  } rndnu_neon;
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64
};

union xnn_qs8_qc8w_conv_minmax_params {
  struct {
    int16_t output_zero_point;
    int16_t output_min;
    int16_t output_max;
  } fp32_scalar;
#if XNN_ARCH_ARM
  struct {
    float magic_bias;
    int32_t magic_bias_less_zero_point;
    uint32_t output_min;
    uint32_t output_max;
  } fp32_armsimd32;
#endif  // XNN_ARCH_ARM
#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  struct {
    float magic_bias;
    int32_t magic_bias_less_output_zero_point;
    int8_t output_min;
    int8_t output_max;
  } fp32_neon;
  struct {
    int16_t output_zero_point;
    uint8_t output_min;
    uint8_t output_max;
  } fp32_neonv8;
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64
};

union xnn_qu8_conv_minmax_params {
  struct {
    int32_t kernel_zero_point;
    float scale;
    int16_t output_zero_point;
    int16_t output_min;
    int16_t output_max;
  } fp32_scalar;
  struct {
    int32_t multiplier;
    uint32_t shift;
    int16_t output_min;
    int16_t output_max;
    int32_t output_zero_point;
    int32_t kernel_zero_point;
    int64_t rounding;
  } rndnu_scalar;
#if XNN_ARCH_ARM
  struct {
    float scale;
    float magic_bias;
    uint32_t minus_kernel_zero_point;
    int32_t magic_bias_less_zero_point;
    uint32_t output_min;
    uint32_t output_max;
  } fp32_armsimd32;
#endif  // XNN_ARCH_ARM
#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  struct {
    uint8_t kernel_zero_point[4];
    float scale;
    float magic_bias;
    int32_t magic_bias_less_output_zero_point;
    uint8_t output_min;
    uint8_t output_max;
  } fp32_neon;
  struct {
    uint8_t kernel_zero_point[4];
    float scale;
    int16_t output_zero_point;
    uint8_t output_min;
    uint8_t output_max;
  } fp32_neonv8;
  struct {
    uint8_t kernel_zero_point[4];
    int32_t right_pre_shift;
    int32_t multiplier;
    int32_t right_post_shift;
    int16_t output_zero_point;
    uint8_t output_min;
    uint8_t output_max;
  } rndnu_neon;
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64
};


// Add w. Min+Max: used by quantized VADD[C] microkernels with MINMAX activation.

struct xnn_qs8_add_minmax_params {
  struct {
    int8_t a_zero_point;
    int8_t b_zero_point;
    int32_t bias;
    int32_t a_multiplier;
    int32_t b_multiplier;
    int32_t shift;
    int16_t output_zero_point;
    int8_t output_min;
    int8_t output_max;
  } scalar;
};

struct xnn_qu8_add_minmax_params {
  struct {
    uint8_t a_zero_point;
    uint8_t b_zero_point;
    int32_t bias;
    int32_t a_multiplier;
    int32_t b_multiplier;
    int32_t shift;
    int16_t output_zero_point;
    uint8_t output_min;
    uint8_t output_max;
  } scalar;
};


// Mul w. Min+Max: used by quantized VMUL[C] microkernels with MINMAX activation.

union xnn_qs8_mul_minmax_params {
  struct {
    int8_t a_zero_point;
    int8_t b_zero_point;
    float scale;
    int16_t output_zero_point;
    int8_t output_min;
    int8_t output_max;
  } scalar;
#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  struct {
    int8_t a_zero_point;
    int8_t b_zero_point;
    int32_t left_pre_shift;
    int32_t multiplier;
    int32_t left_post_shift;
    int16_t output_zero_point;
    int8_t output_min;
    int8_t output_max;
  } rndnu_neon;
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64
};

union xnn_qu8_mul_minmax_params {
  struct {
    uint8_t a_zero_point;
    uint8_t b_zero_point;
    float scale;
    int16_t output_zero_point;
    uint8_t output_min;
    uint8_t output_max;
  } scalar;
#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  struct {
    uint8_t a_zero_point;
    uint8_t b_zero_point;
    int32_t left_pre_shift;
    int32_t multiplier;
    int32_t left_post_shift;
    int16_t output_zero_point;
    uint8_t output_min;
    uint8_t output_max;
  } rndnu_neon;
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64
};

union xnn_binary_uparams {
  struct xnn_qs8_add_minmax_params qs8_add;
  struct xnn_qu8_add_minmax_params qu8_add;
  union xnn_qs8_mul_minmax_params qs8_mul;
  union xnn_qu8_mul_minmax_params qu8_mul;
  union xnn_f16_minmax_params f16_minmax;
  union xnn_f32_minmax_params f32_minmax;
};

// RSum params used by RSUM & RDSUM microkernels.
struct xnn_qs8_rsum_params {
  char _;  // Dummy member variable to comply with the C standard
};

struct xnn_qs8_mean_minmax_params {
  struct {
    float scale;
    int32_t num_elements;
    int8_t input_zero_point;
    int8_t output_zero_point;
  } scalar;
};

struct xnn_qu8_mean_minmax_params {
  struct {
    float scale;
    int32_t num_elements;
    uint8_t input_zero_point;
    uint8_t output_zero_point;
  } scalar;
};

// AvgPool w. Min+Max: used by quantized GAVGPOOL microkernels with MINMAX activation.

union xnn_qs8_avgpool_minmax_params {
  struct {
    int32_t init_bias;
    float scale;
    float output_min_less_zero_point;
    float output_max_less_zero_point;
    float magic_bias;
    int32_t magic_bias_less_output_zero_point;
  } fp32_scalar_fmagic;
  struct {
    int32_t init_bias;
    float scale;
    float magic_bias;
    int32_t magic_min;
    int32_t magic_max;
    int32_t magic_bias_less_zero_point;
  } fp32_scalar_imagic;
  struct {
    int32_t init_bias;
    float scale;
    float output_min_less_zero_point;
    float output_max_less_zero_point;
    int32_t output_zero_point;
  } fp32_scalar_lrintf;
#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  struct {
    int32_t init_bias;
    float scale;
    float magic_bias;
    int32_t magic_bias_less_output_zero_point;
    int8_t output_min;
    int8_t output_max;
  } fp32_neon;
  struct {
    int32_t init_bias;
    float scale;
    int16_t output_zero_point;
    int8_t output_min;
    int8_t output_max;
  } fp32_neonv8;
  struct {
    int32_t init_bias;
    int32_t left_pre_shift;
    int32_t multiplier;
    int32_t left_post_shift;
    int16_t output_zero_point;
    int8_t output_min;
    int8_t output_max;
  } rndnu_neon;
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64
#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  struct {
    XNN_ALIGN(16) int32_t init_bias[4];
    XNN_ALIGN(16) float scale[4];
    XNN_ALIGN(16) float output_max_less_zero_point[4];
    XNN_ALIGN(16) int16_t output_zero_point[8];
    XNN_ALIGN(16) int16_t output_min[8];
  } fp32_sse2;
  struct {
    int32_t init_bias;
    float scale;
    float magic_bias;
    int32_t magic_bias_less_output_zero_point;
    int8_t output_min;
    int8_t output_max;
  } fp32_ssse3;
  struct {
    XNN_ALIGN(16) int32_t init_bias[4];
    XNN_ALIGN(16) float scale[4];
    XNN_ALIGN(16) float magic_bias[4];
    XNN_ALIGN(16) float output_max_less_zero_point[4];
    XNN_ALIGN(16) int32_t magic_bias_less_output_zero_point[4];
    XNN_ALIGN(16) int16_t output_zero_point[8];
    XNN_ALIGN(16) int8_t output_min[16];
    XNN_ALIGN(16) int8_t output_max[16];
  } fp32_sse4;
  struct {
    XNN_ALIGN(16) int32_t init_bias[8];
    XNN_ALIGN(16) float scale[8];
    XNN_ALIGN(16) float output_max_less_zero_point[8];
    XNN_ALIGN(16) float magic_bias[8];
    XNN_ALIGN(16) int32_t magic_bias_less_output_zero_point[8];
    XNN_ALIGN(16) int16_t output_zero_point[16];
    XNN_ALIGN(16) int8_t output_min[32];
    XNN_ALIGN(16) int8_t output_max[32];
  } fp32_avx2;
  struct {
    XNN_ALIGN(64) int32_t init_bias[16];
    XNN_ALIGN(64) float scale[16];
    XNN_ALIGN(64) float output_max_less_zero_point[16];
    XNN_ALIGN(64) int16_t output_zero_point[32];
    XNN_ALIGN(64) int8_t output_min[64];
  } fp32_avx512;
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  struct {
    XNN_ALIGN(8) int32_t init_bias[2];
    XNN_ALIGN(8) float scale[2];
    XNN_ALIGN(8) float magic_bias[2];
    XNN_ALIGN(8) int32_t magic_min[2];
    XNN_ALIGN(8) int32_t magic_bias_less_output_zero_point[2];
    XNN_ALIGN(8) int8_t output_max[8];
  } fp32_wasmsimd;
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
};

union xnn_qu8_avgpool_minmax_params {
  struct {
    int32_t init_bias;
    float scale;
    float output_min_less_zero_point;
    float output_max_less_zero_point;
    float magic_bias;
    int32_t magic_bias_less_output_zero_point;
  } fp32_scalar_fmagic;
  struct {
    int32_t init_bias;
    float scale;
    float magic_bias;
    int32_t magic_min;
    int32_t magic_max;
    int32_t magic_bias_less_zero_point;
  } fp32_scalar_imagic;
  struct {
    int32_t init_bias;
    float scale;
    float output_min_less_zero_point;
    float output_max_less_zero_point;
    int32_t output_zero_point;
  } fp32_scalar_lrintf;
#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  struct {
    int32_t init_bias;
    float scale;
    float magic_bias;
    int32_t magic_bias_less_output_zero_point;
    uint8_t output_min;
    uint8_t output_max;
  } fp32_neon;
  struct {
    int32_t init_bias;
    float scale;
    int16_t output_zero_point;
    uint8_t output_min;
    uint8_t output_max;
  } fp32_neonv8;
  struct {
    int32_t init_bias;
    int32_t left_pre_shift;
    int32_t multiplier;
    int32_t left_post_shift;
    int16_t output_zero_point;
    uint8_t output_min;
    uint8_t output_max;
  } rndnu_neon;
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64
#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  struct {
    XNN_ALIGN(16) int32_t init_bias[4];
    XNN_ALIGN(16) float scale[4];
    XNN_ALIGN(16) float output_max_less_zero_point[4];
    XNN_ALIGN(16) int16_t output_zero_point[8];
    XNN_ALIGN(16) uint8_t output_min[16];
  } fp32_sse2;
  struct {
    XNN_ALIGN(16) int32_t init_bias[4];
    XNN_ALIGN(16) float scale[4];
    XNN_ALIGN(16) float output_max_less_zero_point[4];
    XNN_ALIGN(16) int16_t output_zero_point[8];
    XNN_ALIGN(16) uint8_t output_min[16];
  } fp32_sse4;
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  struct {
    XNN_ALIGN(8) int32_t init_bias[2];
    XNN_ALIGN(8) float scale[2];
    XNN_ALIGN(8) float magic_bias[2];
    XNN_ALIGN(8) int32_t magic_min[2];
    XNN_ALIGN(8) int32_t magic_bias_less_output_zero_point[2];
    XNN_ALIGN(8) uint8_t output_max[8];
  } fp32_wasmsimd;
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
};


// Cvt (Convert): used by VCVT microkernels.

struct xnn_f16_qs8_cvt_params {
  struct {
    xnn_float16 scale;
    int16_t output_zero_point;
    int8_t output_min;
    int8_t output_max;
  } scalar;
};

struct xnn_f32_qs8_cvt_params {
  struct {
    float scale;
    int16_t output_zero_point;
    int8_t output_min;
    int8_t output_max;
  } scalar;
};

struct xnn_f32_qu8_cvt_params {
  struct {
    float scale;
    int16_t output_zero_point;
    uint8_t output_min;
    uint8_t output_max;
  } scalar;
};

struct xnn_s32_f32_cvt_params {
  struct {
    int32_t zero_point;
  } scalar;
};

struct xnn_u32_f32_cvt_params {
  struct {
    int32_t zero_point;
  } scalar;
};

struct xnn_qs8_cvt_params {
  struct {
    int16_t input_zero_point;
    int32_t multiplier;
    int16_t output_zero_point;
  } scalar;
};

struct xnn_qs16_qs8_cvt_params {
  struct {
    int32_t multiplier;
    int32_t output_zero_point;
  } scalar;
};

struct xnn_qs8_f16_cvt_params {
  struct {
    int16_t zero_point;
    xnn_float16 scale;
  } scalar;
};

struct xnn_qs8_f32_cvt_params {
  struct {
    int32_t zero_point;
    float scale;
  } scalar;
};

struct xnn_qu8_cvt_params {
  struct {
    uint16_t input_zero_point;
    int16_t multiplier;
    int16_t output_zero_point;
  } scalar;
};

struct xnn_qu8_f32_cvt_params {
  struct {
    int32_t zero_point;
    float scale;
  } scalar;
};


// ELU: used by VELU microkernels.

struct xnn_f16_elu_params {
  struct {
    xnn_float16 prescale;
    xnn_float16 alpha;
    xnn_float16 beta;
  } scalar;
};

struct xnn_f32_elu_params {
  struct {
    float prescale;
    float alpha;
    float beta;
  } scalar;
};


// ExpMinus: used by RADDEXPMINUSMAX microkernels.

struct xnn_f16_expminus_params {
  char _;  // Dummy member variable to comply with the C standard
};

struct xnn_f32_expminus_params {
  char _;  // Dummy member variable to comply with the C standard
};

// HSwish: used by VHSWISH microkernels.

struct xnn_f16_hswish_params {
  char _;  // Dummy member variable to comply with the C standard
};

struct xnn_f32_hswish_params {
  char _;  // Dummy member variable to comply with the C standard
};

union xnn_qs8_hswish_params {
  struct {
    int16_t input_zero_point;
    int16_t output_zero_point;
    int16_t input_scale_div_mantissa;
    int16_t input_scale_div_exp;
    int16_t scale_ratio;
} scalar;
#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  struct {
    int16_t input_zero_point;
    int16_t output_zero_point;
    int16_t input_scale_div;
    int16_t scale_ratio;
  } sse2;
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
};

union xnn_qu8_hswish_params {
struct {
    int16_t input_zero_point;
    int16_t output_zero_point;
    int16_t input_scale_div_mantissa;
    int16_t input_scale_div_exp;
    int16_t scale_ratio;
  } scalar;
#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  struct {
    int16_t input_zero_point;
    int16_t output_zero_point;
    int16_t input_scale_div;
    int16_t scale_ratio;
  } sse2;
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
};


// LReLU (Leaky ReLU): used by VLRELU microkernels.

struct xnn_f16_lrelu_params {
  struct {
    xnn_float16 slope;
  } scalar;
};

struct xnn_f32_lrelu_params {
  struct {
    float slope;
  } scalar;
};

struct xnn_qs8_lrelu_params {
  struct {
    int32_t input_zero_point;
    int32_t positive_multiplier;
    int32_t negative_multiplier;
    int32_t output_zero_point;
  } scalar;
};

struct xnn_qu8_lrelu_params {
  struct {
    int32_t input_zero_point;
    int32_t positive_multiplier;
    int32_t negative_multiplier;
    int32_t output_zero_point;
  } scalar;
};

// Rnd (Round): used by VRNDNE/VRNDU/VRNDD/VRNDZ microkernels.

struct xnn_f16_rnd_params {
  char _;  // Dummy member variable to comply with the C standard
};

struct xnn_f32_rnd_params {
  char _;  // Dummy member variable to comply with the C standard
};


// Sigmoid: used by VSIGMOID microkernels.

struct xnn_f16_sigmoid_params {
  char _;  // Dummy member variable to comply with the C standard
};

struct xnn_f32_sigmoid_params {
  char _;  // Dummy member variable to comply with the C standard
};


// Sqrt (Square Root): used by VSQRT microkernels.

struct xnn_f16_sqrt_params {
  char _;  // Dummy member variable to comply with the C standard
};

struct xnn_f32_sqrt_params {
  char _;  // Dummy member variable to comply with the C standard
};

// Rsqrt (Reciprocal Square Root): used by VRSQRT microkernels.

struct xnn_f16_rsqrt_params {
  char _;  // Dummy member variable to comply with the C standard
};

struct xnn_f32_rsqrt_params {
  char _;  // Dummy member variable to comply with the C standard.
};

// TanH (Hyperbolic Tangent): used by VTANH microkernels.

union xnn_f16_tanh_params {
  char _;  // Dummy member variable to comply with the C standard
};

union xnn_f32_tanh_params {
  char _;  // Dummy member variable to comply with the C standard
};


// GAvgPool (Global Average Pool): used by GAVGPOOL microkernels in CHW layout with Scale+Min+Max parameters.

union xnn_f16_gavgpool_params {
  struct {
    XNN_ALIGN(16) uint16_t mask[8];
    uint16_t multiplier;
    uint16_t output_min;
    uint16_t output_max;
  } scalar;
};

union xnn_f32_gavgpool_params {
  struct {
    XNN_ALIGN(16) int32_t mask[4];
    float multiplier;
    float output_min;
    float output_max;
  } scalar;
#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  struct {
    XNN_ALIGN(16) float multiplier[4];
    XNN_ALIGN(16) float output_min[4];
    XNN_ALIGN(16) float output_max[4];
    XNN_ALIGN(16) uint32_t mask[4];
  } sse;
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  struct {
    XNN_ALIGN(16) uint32_t mask[4];
    float multiplier;
    float output_min;
    float output_max;
  } neon;
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64 */
};

struct xnn_qs8_packw_params {
  int8_t input_zero_point;
};

// Forward declare for use in microkernel headers for JIT generator functions.
struct xnn_code_buffer;

typedef int xnn_status_t;

struct xnn_x32_packb_params {
  char _;  // Dummy member variable to comply with the C standard
};

struct subconvolution_params {
  void* weights;
  size_t w_stride;
  const void** indirection_buffer;
  void* output;
  size_t slice_width;
  size_t slice_height;
  size_t indirection_y_stride;
  size_t indirection_x_stride;
  // scaled_kernel_size := kernel_size * mr * sizeof(void*).
  size_t scaled_kernel_size;
};

