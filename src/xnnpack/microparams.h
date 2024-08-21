// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <stddef.h>
#include <stdint.h>

#include "xnnpack/common.h"


// Default: serves to differentiate pointer types for micro-kernels without fused activation.

union xnn_f16_default_params {
  char _;  // Dummy member variable to comply with the C standard
};

union xnn_bf16_default_params {
  char _;  // Dummy member variable to comply with the C standard
};

union xnn_f32_default_params {
  char _;  // Dummy member variable to comply with the C standard
};

union xnn_s32_default_params {
  char _;  // Dummy member variable to comply with the C standard
};


// ReLU: serves to differentiate pointer types for micro-kernels with fused ReLU activation.

union xnn_f32_relu_params {
  char _;  // Dummy member variable to comply with the C standard
};


// Scale: used by RSUM microkernels

union xnn_f16_scale_params {
  struct {
    uint16_t scale;
  };
};

union xnn_f16_f32acc_scale_params {
  struct {
    float scale;
  };
};

union xnn_f32_scale_params {
  struct {
    float scale;
  };
};


// Scale+Min+Max: used by AVGPOOL/GAVGPOOL microkernels.

union xnn_f16_scaleminmax_params {
  char _;  // Dummy member variable to comply with the C standard
#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  struct {
    uint16_t scale;
    uint16_t min;
    uint16_t max;
  } fp16arith;
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64
#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  struct {
    float scale;
    float min;
    float max;
  } avx;
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
};

union xnn_f32_scaleminmax_params {
  struct {
    float scale;
    float min;
    float max;
  } scalar;
#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  struct {
    float scale;
    float min;
    float max;
  } sse;
  struct {
    float scale;
    float min;
    float max;
  } avx;
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
};


// Min+Max: used by VCLAMP and GEMM/IGEMM/DWCONV/MAXPOOL/etc with MINMAX activation.

union xnn_bf16_minmax_params {
  struct {
    float min;
    float max;
  } scalar;
};

union xnn_f16_minmax_params {
  struct {
    uint16_t min;
    uint16_t max;
  } fp16arith;
#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  struct {
    float min;
    float max;
  } avx;
  struct {
    float min;
    float max;
  } avx512vnni;
  struct {
    float min;
    float max;
  } avxvnni;
  struct {
    float min;
    float max;
  } scalar;
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
};

union xnn_f32_minmax_params {
  struct {
    float min;
    float max;
  } scalar;
#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  struct {
    float min;
    float max;
  } sse;
  struct {
    float min;
    float max;
  } avx;
  struct {
    float min;
    float max;
  } avx512vnni;
  struct {
    float min;
    float max;
  } avxvnni;
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  struct {
    float min;
    float max;
  } wasmsimd;
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
#if XNN_ARCH_HEXAGON
  struct {
    float min;
    float max;
  } hvx;
#endif //XNN_ARCH_HEXAGON
};

union xnn_f16_qc4w_minmax_params {
  struct {
    uint16_t min;
    uint16_t max;
    int32_t minus_kernel_zero_point;
  } fp16arith;
#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  struct {
    float min;
    float max;
    XNN_ALIGN(32) uint8_t mask[16];
  } avx;
  struct {
    float min;
    float max;
  } avxvnni;
  struct {
    float min;
    float max;
  } avx512vnni;
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
};

union xnn_f16_qb4w_minmax_params {
  struct {
    uint16_t min;
    uint16_t max;
    int32_t minus_kernel_zero_point;
    size_t blocksize;
  } fp16arith;
#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  struct {
    float min;
    float max;
    XNN_ALIGN(32) uint8_t mask[16];
    XNN_ALIGN(32) size_t blocksize;
  } avx;
  struct {
    float min;
    float max;
    size_t blocksize;
  } avxvnni;
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
};

union xnn_f32_qc4w_minmax_params {
  struct {
    float min;
    float max;
    int32_t minus_kernel_zero_point;
  } scalar;
#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  struct {
    float min;
    float max;
    XNN_ALIGN(16) uint32_t magic_bias_c0[4];
    XNN_ALIGN(16) uint32_t magic_bias_c1[4];
    XNN_ALIGN(16) float magic_bias_plus_kernel_zero_point_c0[4];
    XNN_ALIGN(16) float magic_bias_plus_kernel_zero_point_c1[4];
  } sse;
  struct {
    float min;
    float max;
    XNN_ALIGN(32) uint32_t magic_bias_c0[8];
    XNN_ALIGN(32) uint32_t magic_bias_c1[8];
    XNN_ALIGN(32) float magic_bias_plus_kernel_zero_point_c0[8];
    XNN_ALIGN(32) float magic_bias_plus_kernel_zero_point_c1[8];
  } avx;
  struct {
    float min;
    float max;
    uint32_t magic_bias_c0;
    uint32_t magic_bias_c1;
    float magic_bias_plus_kernel_zero_point_c0;
    float magic_bias_plus_kernel_zero_point_c1;
  } avx512;
  struct {
    float min;
    float max;
  } avx512vnni;
  struct {
    float min;
    float max;
  } avxvnni;
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  struct {
    float min;
    float max;
    XNN_ALIGN(8) int32_t minus_kernel_zero_point[2];
  } wasmsimd;
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
};

union xnn_f32_qb4w_minmax_params {
  struct {
    float min;
    float max;
    int32_t minus_kernel_zero_point;
    size_t blocksize;
  } scalar;
#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  struct {
    float min;
    float max;
    XNN_ALIGN(16) uint32_t magic_bias_c0[4];
    XNN_ALIGN(16) uint32_t magic_bias_c1[4];
    XNN_ALIGN(16) float magic_bias_plus_kernel_zero_point_c0[4];
    XNN_ALIGN(16) float magic_bias_plus_kernel_zero_point_c1[4];
    XNN_ALIGN(16) size_t blocksize;
  } sse;
  struct {
    float min;
    float max;
    XNN_ALIGN(32) uint32_t magic_bias_c0[8];
    XNN_ALIGN(32) uint32_t magic_bias_c1[8];
    XNN_ALIGN(32) float magic_bias_plus_kernel_zero_point_c0[8];
    XNN_ALIGN(32) float magic_bias_plus_kernel_zero_point_c1[8];
    XNN_ALIGN(32) size_t blocksize;
  } avx;
  struct {
    float min;
    float max;
    uint32_t magic_bias_c0;
    uint32_t magic_bias_c1;
    float magic_bias_plus_kernel_zero_point_c0;
    float magic_bias_plus_kernel_zero_point_c1;
    size_t blocksize;
  } avx512;
  struct {
    float min;
    float max;
    size_t blocksize;
  } avx512vnni;
  struct {
    float min;
    float max;
    size_t blocksize;
  } avxvnni;
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  struct {
    float min;
    float max;
    XNN_ALIGN(8) int32_t minus_kernel_zero_point[2];
    XNN_ALIGN(8) size_t blocksize;
  } wasmsimd;
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
};

union xnn_s8_minmax_params {
  struct {
    int32_t min;
    int32_t max;
  } scalar;
};

union xnn_u8_minmax_params {
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
    float output_min_less_zero_point;
    float output_max_less_zero_point;
    float magic_bias;
    int32_t magic_bias_less_output_zero_point;
  } fp32_scalar_fmagic;
  struct {
    float scale;
    float magic_bias;
    int32_t magic_min;
    int32_t magic_max;
    int32_t magic_bias_less_zero_point;
  } fp32_scalar_imagic;
  struct {
    float scale;
    float output_min_less_zero_point;
    float output_max_less_zero_point;
    int32_t output_zero_point;
  } fp32_scalar_lrintf;
  struct {
    int32_t multiplier;
    uint32_t shift;
    int64_t rounding;
    int32_t output_min_less_zero_point;
    int32_t output_max_less_zero_point;
    int32_t output_zero_point;
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
#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  struct {
    XNN_ALIGN(16) float scale[4];
    XNN_ALIGN(16) float output_max_less_zero_point[4];
    XNN_ALIGN(16) int16_t output_zero_point[8];
    XNN_ALIGN(16) int16_t output_min[8];
  } fp32_sse2;
  struct {
    XNN_ALIGN(16) float scale[4];
    XNN_ALIGN(16) float output_max_less_zero_point[4];
    XNN_ALIGN(16) int16_t output_zero_point[8];
    XNN_ALIGN(16) int8_t output_min[16];
  } fp32_sse4;
  struct {
    XNN_ALIGN(32) float scale[8];
    XNN_ALIGN(32) float output_max_less_zero_point[8];
    XNN_ALIGN(32) int16_t output_zero_point[16];
    XNN_ALIGN(32) int8_t output_min[32];
  } fp32_avx2;
  struct {
    float output_max_less_zero_point;
    int32_t output_zero_point;
    XNN_ALIGN(64) float scale[16];
    XNN_ALIGN(16) int8_t output_min[16];
  } fp32_avx512;
  struct {
    float output_max_less_zero_point;
    int32_t output_zero_point;
    XNN_ALIGN(64) float scale[16];
    XNN_ALIGN(16) int8_t output_min[16];
  } fp32_avx512vnni;
  struct {
    float output_max_less_zero_point;
    int32_t output_zero_point;
    XNN_ALIGN(32) float scale[8];
    XNN_ALIGN(16) int8_t output_min[16];
  } fp32_avxvnni;
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  struct {
    XNN_ALIGN(8) float scale[2];
    XNN_ALIGN(8) float magic_bias[2];
    XNN_ALIGN(8) int32_t magic_min[2];
    XNN_ALIGN(8) int32_t magic_bias_less_output_zero_point[2];
    XNN_ALIGN(8) int8_t output_max[8];
  } fp32_wasmsimd;
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
};

union xnn_qs8_qc8w_conv_minmax_params {
  struct {
    float magic_bias;
    int32_t magic_min;
    int32_t magic_max;
    int32_t magic_bias_less_zero_point;
  } fp32_scalar_imagic;
  struct {
    float output_min_less_zero_point;
    float output_max_less_zero_point;
    float magic_bias;
    int32_t magic_bias_less_output_zero_point;
  } fp32_scalar_fmagic;
  struct {
    float output_min_less_zero_point;
    float output_max_less_zero_point;
    int32_t output_zero_point;
  } fp32_scalar_lrintf;
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
#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  struct {
    XNN_ALIGN(16) float output_max_less_zero_point[4];
    XNN_ALIGN(16) int16_t output_zero_point[8];
    XNN_ALIGN(16) int16_t output_min[8];
  } fp32_sse2;
  struct {
    XNN_ALIGN(16) float output_max_less_zero_point[4];
    XNN_ALIGN(16) int16_t output_zero_point[8];
    XNN_ALIGN(16) int8_t output_min[16];
  } fp32_sse4;
  struct {
    XNN_ALIGN(32) float output_max_less_zero_point[8];
    XNN_ALIGN(32) int16_t output_zero_point[16];
    XNN_ALIGN(32) int8_t output_min[32];
  } fp32_avx2;
  struct {
    float output_max_less_zero_point;
    int32_t output_zero_point;
    XNN_ALIGN(16) int8_t output_min[16];
  } fp32_avx512;
  struct {
    float output_max_less_zero_point;
    int32_t output_zero_point;
    XNN_ALIGN(16) int8_t output_min[16];
  } fp32_avx512vnni;
  struct {
    float output_max_less_zero_point;
    int32_t output_zero_point;
    XNN_ALIGN(16) int8_t output_min[16];
  } fp32_avxvnni;
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  struct {
    XNN_ALIGN(8) float magic_bias[2];
    XNN_ALIGN(8) int32_t magic_min[2];
    XNN_ALIGN(8) int32_t magic_bias_less_output_zero_point[2];
    XNN_ALIGN(8) int8_t output_max[8];
  } fp32_wasmsimd;
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
};

union xnn_qu8_conv_minmax_params {
  struct {
    int32_t kernel_zero_point;
    float scale;
    float output_min_less_zero_point;
    float output_max_less_zero_point;
    float magic_bias;
    int32_t magic_bias_less_output_zero_point;
  } fp32_scalar_fmagic;
  struct {
    int32_t kernel_zero_point;
    float scale;
    float magic_bias;
    int32_t magic_min;
    int32_t magic_max;
    int32_t magic_bias_less_zero_point;
  } fp32_scalar_imagic;
  struct {
    int32_t kernel_zero_point;
    float scale;
    float output_min_less_zero_point;
    float output_max_less_zero_point;
    int32_t output_zero_point;
  } fp32_scalar_lrintf;
  struct {
    int32_t kernel_zero_point;
    int32_t multiplier;
    int64_t rounding;
    uint32_t shift;
    int32_t output_min_less_zero_point;
    int32_t output_max_less_zero_point;
    int32_t output_zero_point;
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
#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  struct {
    XNN_ALIGN(16) int16_t kernel_zero_point[8];
    XNN_ALIGN(16) float scale[4];
    XNN_ALIGN(16) float output_max_less_zero_point[4];
    XNN_ALIGN(16) int16_t output_zero_point[8];
    XNN_ALIGN(16) uint8_t output_min[16];
  } fp32_sse2;
  struct {
    XNN_ALIGN(32) int16_t kernel_zero_point[16];
    XNN_ALIGN(32) float scale[8];
    XNN_ALIGN(32) float output_max_less_zero_point[8];
    XNN_ALIGN(32) int16_t output_zero_point[16];
    XNN_ALIGN(32) uint8_t output_min[32];
  } fp32_avx2;
  struct {
    float output_max_less_zero_point;
    int32_t output_zero_point;
    XNN_ALIGN(64) int16_t kernel_zero_point[32];
    XNN_ALIGN(64) float scale[16];
    XNN_ALIGN(16) uint8_t output_min[16];
  } fp32_avx512;
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  struct {
    XNN_ALIGN(8) int16_t kernel_zero_point[4];
    XNN_ALIGN(8) float scale[2];
    XNN_ALIGN(8) float magic_bias[2];
    XNN_ALIGN(8) int32_t magic_min[2];
    XNN_ALIGN(8) int32_t magic_bias_less_output_zero_point[2];
    XNN_ALIGN(8) int8_t output_max[8];
  } fp32_wasmsimd;
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
};


// Add w. Min+Max: used by quantized VADD[C] microkernels with MINMAX activation.

union xnn_qs8_add_minmax_params {
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

union xnn_qu8_add_minmax_params {
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


// RSum params used by RSUM & RDSUM microkernels.
union xnn_qs8_rsum_params {
  char _;  // Dummy member variable to comply with the C standard
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

union xnn_f16_qs8_cvt_params {
  struct {
    float scale;
    int16_t output_zero_point;
    int8_t output_min;
    int8_t output_max;
  } scalar;
#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  struct {
    uint16_t scale;
    int16_t output_zero_point;
    int8_t output_min;
    int8_t output_max;
  } neonfp16arith;
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64
};

union xnn_f32_qs8_cvt_params {
  struct {
    float scale;
    int16_t output_zero_point;
    int8_t output_min;
    int8_t output_max;
  } scalar;
};

union xnn_f32_qu8_cvt_params {
  struct {
    float scale;
    int16_t output_zero_point;
    uint8_t output_min;
    uint8_t output_max;
  } scalar;
};

union xnn_qs8_cvt_params {
  struct {
    int16_t input_zero_point;
    int32_t multiplier;
    int16_t output_zero_point;
  } scalar;
};

union xnn_qs16_qs8_cvt_params {
  struct {
    int32_t multiplier;
    int32_t output_zero_point;
  } scalar;
};

union xnn_qs8_f16_cvt_params {
  char _;  // Dummy member variable to comply with the C standard
#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  struct {
    int16_t zero_point;
    uint16_t scale;
  } neon;
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64
#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  struct {
    int32_t zero_point;
    float scale;
  } avx;
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
};

union xnn_qs8_f32_cvt_params {
  struct {
    int32_t zero_point;
    float scale;
  } scalar;
};

union xnn_qu8_cvt_params {
  struct {
    uint16_t input_zero_point;
    int16_t multiplier;
    int16_t output_zero_point;
  } scalar;
};

union xnn_qu8_f32_cvt_params {
  struct {
    int32_t zero_point;
    float scale;
  } scalar;
};


// ELU: used by VELU microkernels.

union xnn_f16_elu_params {
  struct {
    uint16_t prescale;
    uint16_t minus_alpha;
    uint16_t beta;
  } scalar;
#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  struct {
    float prescale;
    float alpha;
    float beta;
  } avx2;
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
};

union xnn_f32_elu_params {
  struct {
    float prescale;
    float alpha;
    float beta;
  } scalar;
};


// ExpMinus: used by RADDEXPMINUSMAX microkernels.

union xnn_f16_expminus_params {
  char _;  // Dummy member variable to comply with the C standard
};

union xnn_f32_expminus_params {
  char _;  // Dummy member variable to comply with the C standard
};

// HSwish: used by VHSWISH microkernels.

union xnn_f16_hswish_params {
  char _;  // Dummy member variable to comply with the C standard
};

union xnn_f32_hswish_params {
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

union xnn_f16_lrelu_params {
  char _;  // Dummy member variable to comply with the C standard
#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  struct {
    uint16_t slope;
  } fp16arith;
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64
#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  struct {
    float slope;
  } avx;
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
};

union xnn_f32_lrelu_params {
  struct {
    float slope;
  } scalar;
};

union xnn_qs8_lrelu_params {
  struct {
    int32_t input_zero_point;
    int32_t positive_multiplier;
    int32_t negative_multiplier;
    int32_t output_zero_point;
  } scalar;
};

union xnn_qu8_lrelu_params {
  struct {
    int32_t input_zero_point;
    int32_t positive_multiplier;
    int32_t negative_multiplier;
    int32_t output_zero_point;
  } scalar;
};

// Rnd (Round): used by VRNDNE/VRNDU/VRNDD/VRNDZ microkernels.

union xnn_f16_rnd_params {
  char _;  // Dummy member variable to comply with the C standard
};

union xnn_f32_rnd_params {
  char _;  // Dummy member variable to comply with the C standard
};


// Sigmoid: used by VSIGMOID microkernels.

union xnn_f16_sigmoid_params {
  char _;  // Dummy member variable to comply with the C standard
};

union xnn_f32_sigmoid_params {
  char _;  // Dummy member variable to comply with the C standard
};


// Sqrt (Square Root): used by VSQRT microkernels.

union xnn_f16_sqrt_params {
  char _;  // Dummy member variable to comply with the C standard
};

union xnn_f32_sqrt_params {
  char _;  // Dummy member variable to comply with the C standard
};

// Rsqrt (Reciprocal Square Root): used by VRSQRT microkernels.

union xnn_f16_rsqrt_params {
  char _;  // Dummy member variable to comply with the C standard
};

union xnn_f32_rsqrt_params {
  char _;  // Dummy member variable to comply with the C standard.
};

// TanH (Hyperbolic Tangent): used by VTANH microkernels.

union xnn_f16_tanh_params {
  char _;  // Dummy member variable to comply with the C standard
#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  struct {
    XNN_ALIGN(32) uint16_t sign_mask[16];
    XNN_ALIGN(32) float sat_cutoff[8];
    XNN_ALIGN(32) float log2e[8];
    XNN_ALIGN(32) float magic_bias[8];
    XNN_ALIGN(32) float minus_ln2[8];
    XNN_ALIGN(32) float c3[8];
    XNN_ALIGN(32) float c2[8];
    XNN_ALIGN(32) float two[8];
    XNN_ALIGN(32) float minus_one[8];
  } avx_expm1minus_rr1_p3h2;
  struct {
    XNN_ALIGN(32) float neg_sat_cutoff[8];
    XNN_ALIGN(32) float pos_sat_cutoff[8];
    XNN_ALIGN(32) float c19[8];
    XNN_ALIGN(32) float c17[8];
    XNN_ALIGN(32) float c15[8];
    XNN_ALIGN(32) float c13[8];
    XNN_ALIGN(32) float c11[8];
    XNN_ALIGN(32) float c9[8];
    XNN_ALIGN(32) float c7[8];
    XNN_ALIGN(32) float c5[8];
    XNN_ALIGN(32) float c3[8];
  } avx_polynomial_p19h9t2;
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
};

union xnn_f32_tanh_params {
  struct {
    float sat_cutoff;
    float minus_log2e;
    float magic_bias;
    float ln2;
    float c6;
    float c5;
    float c4;
    float c3;
    float c2;
    float minus_two;
    float one;
  } scalar_expm1minus_rr1_p6h5;
  struct {
    float sat_cutoff;
    float minus_log2e;
    float magic_bias;
    float ln2;
    float c4;
    float c3;
    float c2;
    float minus_two;
    float one;
  } scalar_expm1minus_rr1_lut8_p4h3;
#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  struct {
    XNN_ALIGN(8) float sat_cutoff[2];
    XNN_ALIGN(8) float minus_log2e[2];
    XNN_ALIGN(8) float magic_bias[2];
    XNN_ALIGN(8) float ln2[2];
    XNN_ALIGN(8) float c6[2];
    XNN_ALIGN(8) float c5[2];
    XNN_ALIGN(8) float c4[2];
    XNN_ALIGN(8) float c3[2];
    XNN_ALIGN(8) float c2[2];
    XNN_ALIGN(8) float minus_two[2];
    XNN_ALIGN(8) float one[2];
    XNN_ALIGN(8) float sign_mask[2];
  } wasmsimd_expm1minus_rr1_p6h5_abs;
  struct {
    XNN_ALIGN(8) float sat_cutoff[2];
    XNN_ALIGN(8) float minus_log2e[2];
    XNN_ALIGN(8) float magic_bias[2];
    XNN_ALIGN(8) uint32_t index_mask[2];
    XNN_ALIGN(8) float ln2[2];
    XNN_ALIGN(8) float c4[2];
    XNN_ALIGN(8) float c3[2];
    XNN_ALIGN(8) float c2[2];
    XNN_ALIGN(8) float minus_two[2];
    XNN_ALIGN(8) float one[2];
    XNN_ALIGN(8) float sign_mask[2];
  } wasmsimd_expm1minus_rr1_lut8_p4h3_abs;
  struct {
    XNN_ALIGN(8) float sign_mask[2];
    XNN_ALIGN(8) float sat_cutoff[2];
    XNN_ALIGN(8) float log2e[2];
    XNN_ALIGN(8) float magic_bias[2];
    XNN_ALIGN(8) float minus_ln2[2];
    XNN_ALIGN(8) float c6[2];
    XNN_ALIGN(8) float c5[2];
    XNN_ALIGN(8) float c4[2];
    XNN_ALIGN(8) float c3[2];
    XNN_ALIGN(8) float c2[2];
    XNN_ALIGN(8) float two[2];
    XNN_ALIGN(8) float one[2];
  } wasmsimd_expm1minus_rr1_p6h5_nabs;
  struct {
    XNN_ALIGN(8) float sign_mask[2];
    XNN_ALIGN(8) float sat_cutoff[2];
    XNN_ALIGN(8) float log2e[2];
    XNN_ALIGN(8) float magic_bias[2];
    XNN_ALIGN(8) uint32_t index_mask[2];
    XNN_ALIGN(8) float minus_ln2[2];
    XNN_ALIGN(8) float c4[2];
    XNN_ALIGN(8) float c3[2];
    XNN_ALIGN(8) float c2[2];
    XNN_ALIGN(8) float two[2];
    XNN_ALIGN(8) float one[2];
  } wasmsimd_expm1minus_rr1_lut8_p4h3_nabs;
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  struct {
    float sat_cutoff;
    float minus_log2e;
    float magic_bias;
    float ln2;
    float c6;
    float c5;
    float c4;
    float c3;
    float c2;
  } neon_expm1minus_rr1_p6h5;
  struct {
    float sat_cutoff;
    float minus_log2e;
    float magic_bias;
    float ln2;
    float c4;
    float c3;
    float c2;
  } neon_expm1minus_rr1_lut8_p4h3;
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64
};


// CHW: used by CONV/DWCONV microkernels in CHW layout with Min+Max parameters.

union xnn_f16_chw_params {
  char _;  // Dummy member variable to comply with the C standard
#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  struct {
    uint16_t min;
    uint16_t max;
    uint16_t pad[2];  // pad to 8 bytes alignment
    XNN_ALIGN(8) uint16_t mask[8];
  } neonfp16arith_stride1;
  struct {
    uint16_t min;
    uint16_t max;
    uint16_t pad[2];  // pad to 8 bytes alignment
    XNN_ALIGN(8) uint16_t mask_even[8];
    XNN_ALIGN(8) uint16_t mask_odd[8];
  } neonfp16arith_stride2;
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64
};

union xnn_f32_chw_params {
  struct {
    float min;
    float max;
  } scalar;
#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  struct {
    float min;
    float max;
    XNN_ALIGN(8) uint32_t mask[4];
  } neon_stride1;
  struct {
    float min;
    float max;
    XNN_ALIGN(8) uint32_t mask_even[4];
    XNN_ALIGN(8) uint32_t mask_odd[4];
  } neon_stride2;
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64
#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  struct {
    float min;
    float max;
    XNN_ALIGN(16) uint32_t mask[4];
  } sse_stride1;
  struct {
    float min;
    float max;
    XNN_ALIGN(16) uint32_t mask_even[4];
    XNN_ALIGN(16) uint32_t mask_odd[4];
  } sse_stride2;
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  struct {
    float min;
    float max;
    XNN_ALIGN(16) uint32_t mask[4];
  } wasmsimd_stride1;
  struct {
    float min;
    float max;
    XNN_ALIGN(16) uint32_t mask_even[4];
    XNN_ALIGN(16) uint32_t mask_odd[4];
  } wasmsimd_stride2;
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
};


// GAvgPool (Global Average Pool): used by GAVGPOOL microkernels in CHW layout with Scale+Min+Max parameters.

union xnn_f16_gavgpool_params {
  char _;  // Dummy member variable to comply with the C standard
#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  struct {
    XNN_ALIGN(16) uint16_t mask[8];
    uint16_t multiplier;
    uint16_t output_min;
    uint16_t output_max;
  } neonfp16arith;
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64 */
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

// Forward declare for use in microkernel headers for JIT generator functions.
struct xnn_code_buffer;

typedef int xnn_status_t;

union xnn_x32_packb_params {
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

