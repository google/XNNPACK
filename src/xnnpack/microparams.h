// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <stddef.h>
#include <stdint.h>

#include <xnnpack/common.h>


// Default: serves to differentiate pointer types for micro-kernels without fused activation.

union xnn_f16_default_params {
  char _; // Dummy member variable to comply with the C standard
};

union xnn_f32_default_params {
  char _; // Dummy member variable to comply with the C standard
#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  struct {
    int32_t mask_table[14];
  } avx;
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
};


// ReLU: serves to differentiate pointer types for micro-kernels with fused ReLU activation.

union xnn_f32_relu_params {
  char _; // Dummy member variable to comply with the C standard
};


// Scale+Min+Max: used by AVGPOOL/GAVGPOOL microkernels.

union xnn_f16_scaleminmax_params {
  char _; // Dummy member variable to comply with the C standard
#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  struct {
    uint16_t scale;
    uint16_t min;
    uint16_t max;
  } fp16arith;
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64
#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  struct {
    XNN_ALIGN(32) float scale[8];
    XNN_ALIGN(32) float min[8];
    XNN_ALIGN(32) float max[8];
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
    XNN_ALIGN(16) float scale[4];
    XNN_ALIGN(16) float min[4];
    XNN_ALIGN(16) float max[4];
  } sse;
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
  char _; // Dummy member variable to comply with the C standard
#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  struct {
    uint16_t min;
    uint16_t max;
  } fp16arith;
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64
#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  struct {
    XNN_ALIGN(32) float min[8];
    XNN_ALIGN(32) float max[8];
  } avx;
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
};

union xnn_f32_minmax_params {
  struct {
    float min;
    float max;
  } scalar;
#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  struct {
    XNN_ALIGN(16) float min[4];
    XNN_ALIGN(16) float max[4];
  } sse;
  struct {
    XNN_ALIGN(32) float min[8];
    XNN_ALIGN(32) float max[8];
    int32_t mask_table[14];
  } avx;
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  struct {
    XNN_ALIGN(8) float min[2];
    XNN_ALIGN(8) float max[2];
  } wasmsimd;
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
};

union xnn_s8_minmax_params {
  struct {
    int32_t min;
    int32_t max;
  } scalar;
#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  struct {
    XNN_ALIGN(16) uint8_t bias[16];
    XNN_ALIGN(16) uint8_t min_with_bias[16];
    XNN_ALIGN(16) uint8_t max_with_bias[16];
  } sse2;
  struct {
    XNN_ALIGN(16) int8_t min[16];
    XNN_ALIGN(16) int8_t max[16];
  } sse4;
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  struct {
    int8_t min;
    int8_t max;
  } neon;
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64
#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  struct {
    XNN_ALIGN(8) int8_t min[8];
    XNN_ALIGN(8) int8_t max[8];
  } wasmsimd;
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
};

union xnn_u8_minmax_params {
  struct {
    uint32_t min;
    uint32_t max;
  } scalar;
#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  struct {
    XNN_ALIGN(16) uint8_t min[16];
    XNN_ALIGN(16) uint8_t max[16];
  } sse2;
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  struct {
    uint8_t min;
    uint8_t max;
  } neon;
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64
#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  struct {
    XNN_ALIGN(8) uint8_t min[8];
    XNN_ALIGN(8) uint8_t max[8];
  } wasmsimd;
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
};


// Conv w. Min+Max: used by quantized GEMM/IGEMM/DWCONV microkernels with MINMAX activation.

union xnn_qc8_conv_minmax_params {
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
    XNN_ALIGN(64) float output_max_less_zero_point[16];
    XNN_ALIGN(64) int16_t output_zero_point[32];
    XNN_ALIGN(64) int8_t output_min[64];
  } fp32_avx512;
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
    XNN_ALIGN(64) float scale[16];
    XNN_ALIGN(64) float output_max_less_zero_point[16];
    XNN_ALIGN(64) int16_t output_zero_point[32];
    XNN_ALIGN(64) int8_t output_min[64];
  } fp32_avx512;
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
    XNN_ALIGN(64) int16_t kernel_zero_point[32];
    XNN_ALIGN(64) float scale[16];
    XNN_ALIGN(64) float output_max_less_zero_point[16];
    XNN_ALIGN(64) int16_t output_zero_point[32];
    XNN_ALIGN(64) uint8_t output_min[64];
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
    int32_t bias;
    int32_t a_multiplier;
    int32_t b_multiplier;
    uint32_t shift;
    int32_t output_min_less_zero_point;
    int32_t output_max_less_zero_point;
    int32_t output_zero_point;
  } scalar;
#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  struct {
    int8_t a_zero_point;
    int8_t b_zero_point;
    int16_t output_zero_point;
    int32_t a_multiplier;
    int32_t b_multiplier;
    int32_t right_shift;
    int8_t output_min;
    int8_t output_max;
  } neon;
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64
#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  struct {
    XNN_ALIGN(16) int32_t bias[4];
    XNN_ALIGN(16) uint16_t a_multiplier_lo[8];
    XNN_ALIGN(16) uint16_t a_multiplier_hi[8];
    XNN_ALIGN(16) uint16_t b_multiplier_lo[8];
    XNN_ALIGN(16) uint16_t b_multiplier_hi[8];
    uint32_t shift;
    uint32_t b_multiplier;
    XNN_ALIGN(16) int16_t output_zero_point[8];
    XNN_ALIGN(16) int16_t output_min[8];
    XNN_ALIGN(16) int16_t output_max[8];
  } sse2;
  struct {
    XNN_ALIGN(16) int32_t bias[4];
    XNN_ALIGN(16) uint16_t a_multiplier_lo[8];
    XNN_ALIGN(16) uint16_t a_multiplier_hi[8];
    XNN_ALIGN(16) uint16_t b_multiplier_lo[8];
    XNN_ALIGN(16) uint16_t b_multiplier_hi[8];
    uint32_t shift;
    uint32_t b_multiplier;
    XNN_ALIGN(16) int16_t output_zero_point[8];
    XNN_ALIGN(16) int8_t output_min[16];
    XNN_ALIGN(16) int8_t output_max[16];
  } sse4_mul16;
  struct {
    XNN_ALIGN(16) int32_t bias[4];
    XNN_ALIGN(16) int32_t a_multiplier[4];
    XNN_ALIGN(16) int32_t b_multiplier[4];
    XNN_ALIGN(16) uint64_t shift[2];
    XNN_ALIGN(16) int16_t output_zero_point[8];
    XNN_ALIGN(16) int8_t output_min[16];
    XNN_ALIGN(16) int8_t output_max[16];
  } sse4_mul32;
  struct {
    XNN_ALIGN(32) int32_t bias[8];
    XNN_ALIGN(32) int32_t a_multiplier[8];
    XNN_ALIGN(32) int32_t b_multiplier[8];
    XNN_ALIGN(32) uint64_t shift[4];
    XNN_ALIGN(32) int16_t output_zero_point[16];
    XNN_ALIGN(16) int8_t output_min[16];
    XNN_ALIGN(16) int8_t output_max[16];
  } avx2;
  struct {
    XNN_ALIGN(64) int32_t bias[16];
    XNN_ALIGN(64) int32_t a_multiplier[16];
    XNN_ALIGN(64) int32_t b_multiplier[16];
    XNN_ALIGN(64) uint64_t shift[8];
    XNN_ALIGN(64) int16_t output_zero_point[32];
    XNN_ALIGN(32) int8_t output_min[32];
    XNN_ALIGN(32) int8_t output_max[32];
  } avx512;
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  struct {
    XNN_ALIGN(8) int32_t bias[2];
    XNN_ALIGN(8) int32_t a_multiplier[2];
    XNN_ALIGN(8) int32_t b_multiplier[2];
    uint32_t shift;
    XNN_ALIGN(8) int16_t output_zero_point[4];
    XNN_ALIGN(8) int8_t output_min[8];
    XNN_ALIGN(8) int8_t output_max[8];
  } wasmsimd;
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
};

union xnn_qu8_add_minmax_params {
  struct {
    int32_t bias;
    int32_t a_multiplier;
    int32_t b_multiplier;
    int32_t rounding;
    uint32_t shift;
    int32_t output_min_less_zero_point;
    int32_t output_max_less_zero_point;
    int32_t output_zero_point;
  } scalar;
#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  struct {
    uint8_t a_zero_point;
    uint8_t b_zero_point;
    int16_t output_zero_point;
    int32_t a_multiplier;
    int32_t b_multiplier;
    int32_t right_shift;
    uint8_t output_min;
    uint8_t output_max;
  } neon;
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64
#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  struct {
    XNN_ALIGN(16) int32_t bias[4];
    XNN_ALIGN(16) uint16_t a_multiplier_lo[8];
    XNN_ALIGN(16) uint16_t a_multiplier_hi[8];
    XNN_ALIGN(16) uint16_t b_multiplier_lo[8];
    XNN_ALIGN(16) uint16_t b_multiplier_hi[8];
    uint32_t shift;
    uint32_t b_multiplier;
    XNN_ALIGN(16) int16_t output_zero_point[8];
    XNN_ALIGN(16) uint8_t output_min[16];
    XNN_ALIGN(16) uint8_t output_max[16];
  } sse2;
  struct {
    XNN_ALIGN(16) int32_t bias[4];
    XNN_ALIGN(16) int32_t a_multiplier[4];
    XNN_ALIGN(16) int32_t b_multiplier[4];
    XNN_ALIGN(16) uint64_t shift[2];
    XNN_ALIGN(16) int16_t output_zero_point[8];
    XNN_ALIGN(16) uint8_t output_min[16];
    XNN_ALIGN(16) uint8_t output_max[16];
  } sse4;
  struct {
    XNN_ALIGN(32) int32_t bias[8];
    XNN_ALIGN(32) int32_t a_multiplier[8];
    XNN_ALIGN(32) int32_t b_multiplier[8];
    XNN_ALIGN(32) uint64_t shift[4];
    XNN_ALIGN(32) int16_t output_zero_point[16];
    XNN_ALIGN(16) uint8_t output_min[16];
    XNN_ALIGN(16) uint8_t output_max[16];
  } avx2;
  struct {
    XNN_ALIGN(64) int32_t bias[16];
    XNN_ALIGN(64) int32_t a_multiplier[16];
    XNN_ALIGN(64) int32_t b_multiplier[16];
    XNN_ALIGN(64) uint64_t shift[8];
    XNN_ALIGN(64) int16_t output_zero_point[32];
    XNN_ALIGN(32) uint8_t output_min[32];
    XNN_ALIGN(32) uint8_t output_max[32];
  } avx512;
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  struct {
    XNN_ALIGN(8) int32_t bias[2];
    XNN_ALIGN(8) int32_t a_multiplier[2];
    XNN_ALIGN(8) int32_t b_multiplier[2];
    uint32_t shift;
    XNN_ALIGN(8) int16_t output_zero_point[4];
    XNN_ALIGN(8) uint8_t output_min[8];
    XNN_ALIGN(8) uint8_t output_max[8];
  } wasmsimd;
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
};


// Mul w. Min+Max: used by quantized VMUL[C] microkernels with MINMAX activation.

union xnn_qs8_mul_minmax_params {
  struct {
    int32_t a_zero_point;
    int32_t b_zero_point;
    float scale;
    float output_min_less_zero_point;
    float output_max_less_zero_point;
    float magic_bias;
    int32_t magic_bias_less_output_zero_point;
  } fp32_scalar;
#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  struct {
    int8_t a_zero_point[2];
    int8_t b_zero_point[2];
    float scale;
    float magic_bias;
    int32_t magic_bias_less_output_zero_point;
    int8_t output_min;
    int8_t output_max;
  } fp32_neon;
  struct {
    int8_t a_zero_point[2];
    int8_t b_zero_point[2];
    float scale;
    int16_t output_zero_point;
    int8_t output_min;
    int8_t output_max;
  } fp32_neonv8;
  struct {
    int8_t a_zero_point[2];
    int8_t b_zero_point[2];
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
    XNN_ALIGN(16) int16_t a_zero_point[8];
    XNN_ALIGN(16) int16_t b_zero_point[8];
    XNN_ALIGN(16) float scale[4];
    XNN_ALIGN(16) int16_t output_zero_point[8];
    XNN_ALIGN(16) int16_t output_min[8];
    XNN_ALIGN(16) int16_t output_max[8];
  } fp32_sse2;
  struct {
    XNN_ALIGN(16) int16_t a_zero_point[8];
    XNN_ALIGN(16) int16_t b_zero_point[8];
    XNN_ALIGN(16) float scale[4];
    XNN_ALIGN(16) int16_t output_zero_point[8];
    XNN_ALIGN(16) int8_t output_min[16];
    XNN_ALIGN(16) int8_t output_max[16];
  } fp32_sse4;
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  struct {
    XNN_ALIGN(8) int16_t a_zero_point[4];
    XNN_ALIGN(8) int16_t b_zero_point[4];
    XNN_ALIGN(8) float scale[2];
    XNN_ALIGN(8) float magic_bias[2];
    XNN_ALIGN(8) int32_t magic_min[2];
    XNN_ALIGN(8) int32_t magic_bias_less_output_zero_point[2];
    XNN_ALIGN(8) int8_t output_max[8];
  } fp32_wasmsimd;
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
};

union xnn_qu8_mul_minmax_params {
  struct {
    int32_t a_zero_point;
    int32_t b_zero_point;
    float scale;
    float output_min_less_zero_point;
    float output_max_less_zero_point;
    float magic_bias;
    int32_t magic_bias_less_output_zero_point;
  } fp32_scalar;
#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  struct {
    uint8_t a_zero_point[2];
    uint8_t b_zero_point[2];
    float scale;
    float magic_bias;
    int32_t magic_bias_less_output_zero_point;
    uint8_t output_min;
    uint8_t output_max;
  } fp32_neon;
  struct {
    uint8_t a_zero_point[2];
    uint8_t b_zero_point[2];
    float scale;
    int16_t output_zero_point;
    uint8_t output_min;
    uint8_t output_max;
  } fp32_neonv8;
  struct {
    uint8_t a_zero_point[2];
    uint8_t b_zero_point[2];
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
    XNN_ALIGN(16) int16_t a_zero_point[8];
    XNN_ALIGN(16) int16_t b_zero_point[8];
    XNN_ALIGN(16) float scale[4];
    XNN_ALIGN(16) int16_t output_zero_point[8];
    XNN_ALIGN(16) uint8_t output_min[16];
    XNN_ALIGN(16) uint8_t output_max[16];
  } fp32_sse2;
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  struct {
    XNN_ALIGN(8) int16_t a_zero_point[4];
    XNN_ALIGN(8) int16_t b_zero_point[4];
    XNN_ALIGN(8) float scale[2];
    XNN_ALIGN(8) float magic_bias[2];
    XNN_ALIGN(8) int32_t magic_min[2];
    XNN_ALIGN(8) int32_t magic_bias_less_output_zero_point[2];
    XNN_ALIGN(8) uint8_t output_max[8];
  } fp32_wasmsimd;
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
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
    XNN_ALIGN(16) int32_t init_bias[4];
    XNN_ALIGN(16) float scale[4];
    XNN_ALIGN(16) float output_max_less_zero_point[4];
    XNN_ALIGN(16) int16_t output_zero_point[8];
    XNN_ALIGN(16) int8_t output_min[16];
  } fp32_sse4;
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

  // Legacy parameters used by QU8 AVGPOOL microkernels
  struct {
    int32_t bias;
    int32_t multiplier;
    int64_t rounding;
    uint32_t right_shift;
    int32_t output_min_less_zero_point;
    int32_t output_max_less_zero_point;
    int32_t output_zero_point;
  } scalar;
#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  struct {
    int32_t bias;
    int32_t multiplier;
    int64_t left_shift;
    int16_t output_zero_point;
    uint8_t output_min;
    uint8_t output_max;
  } neon;
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64
#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  struct {
    XNN_ALIGN(16) int32_t bias[4];
    XNN_ALIGN(16) uint32_t multiplier[4];
    XNN_ALIGN(16) uint64_t rounding[2];
    XNN_ALIGN(16) uint64_t right_shift[2];
    XNN_ALIGN(16) int16_t output_zero_point[8];
    XNN_ALIGN(16) uint8_t output_min[16];
    XNN_ALIGN(16) uint8_t output_max[16];
  } sse2;
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
};


// Abs: used by VABS microkernels.

union xnn_f16_abs_params {
  char _; // Dummy member variable to comply with the C standard
#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  struct {
    XNN_ALIGN(16) uint16_t nonsign_mask[8];
  } sse;
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
};

union xnn_f32_abs_params {
  char _; // Dummy member variable to comply with the C standard
#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  struct {
    XNN_ALIGN(16) float nonsign_mask[4];
  } sse;
  struct {
    XNN_ALIGN(32) float nonsign_mask[8];
    int32_t mask_table[14];
  } avx;
  struct {
    uint32_t nonsign_mask;
  } avx512;
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  struct {
    XNN_ALIGN(8) float nonsign_mask[2];
  } wasmsimd;
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
};


// Cvt (Convert): used by VCVT microkernels.

union xnn_f16_f32_cvt_params {
  struct {
    uint32_t sign_mask;
    uint32_t exp_offset;
    float exp_scale;
    uint32_t magic_mask;
    float magic_bias;
    uint32_t denorm_cutoff;
  } scalar;
#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  struct {
    float exp_scale;
  } neon;
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64
#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  struct {
    XNN_ALIGN(16) uint16_t sign_mask[8];
    XNN_ALIGN(16) uint16_t exp_offset[8];
    XNN_ALIGN(16) float exp_scale[4];
    XNN_ALIGN(16) uint16_t magic_mask[8];
    XNN_ALIGN(16) float magic_bias[4];
    XNN_ALIGN(16) int16_t denorm_cutoff[8];
  } sse_int16;
  struct {
    XNN_ALIGN(16) uint32_t sign_mask[4];
    XNN_ALIGN(16) uint32_t exp_offset[4];
    XNN_ALIGN(16) float exp_scale[4];
    XNN_ALIGN(16) uint32_t magic_bias[4];
    XNN_ALIGN(16) int32_t denorm_cutoff[4];
  } sse_int32;
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  struct {
    XNN_ALIGN(8) uint16_t sign_mask[4];
    XNN_ALIGN(8) uint16_t exp_offset[4];
    XNN_ALIGN(8) float exp_scale[2];
    XNN_ALIGN(8) uint16_t magic_mask[4];
    XNN_ALIGN(8) float magic_bias[2];
    XNN_ALIGN(8) int16_t denorm_cutoff[4];
  } wasmsimd_int16;
  struct {
    XNN_ALIGN(8) uint32_t sign_mask[2];
    XNN_ALIGN(8) uint32_t exp_offset[2];
    XNN_ALIGN(8) float exp_scale[2];
    XNN_ALIGN(8) uint32_t magic_bias[2];
    XNN_ALIGN(8) int32_t denorm_cutoff[2];
  } wasmsimd_int32;
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
};

union xnn_f32_f16_cvt_params {
  struct {
    uint32_t nonsign_mask;
    uint32_t exp_bias;
    float scale_to_inf;
    uint32_t expw_max;
    float scale_to_zero;
    uint32_t bias_min;
    uint16_t exph_mask;
    uint16_t manth_mask;
    uint16_t nanh;
  } scalar_bitcast;
  struct {
    float scale_to_inf;
    uint32_t exp_bias;
    float scale_to_zero;
    uint32_t expw_max;
    uint32_t bias_min;
    uint16_t exph_mask;
    uint16_t manth_mask;
    uint16_t nanh;
  } scalar_fabsf;
#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  struct {
    uint32_t exp_bias;
    float scale_to_inf;
    uint32_t expw_max;
    float scale_to_zero;
  } neon;
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64
#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  struct {
    XNN_ALIGN(16) uint32_t nonsign_mask[4];
    XNN_ALIGN(16) uint32_t exp_bias[4];
    XNN_ALIGN(16) float scale_to_inf[4];
    XNN_ALIGN(16) uint32_t expw_max[4];
    XNN_ALIGN(16) float scale_to_zero[4];
    XNN_ALIGN(16) int16_t bias_min[8];
    XNN_ALIGN(16) uint32_t manth_mask[4];
    XNN_ALIGN(16) uint32_t exph_mask[4];
    XNN_ALIGN(16) uint16_t nanh[8];
  } sse2;
  struct {
    int32_t mask_table[14];
  } f16c;
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  struct {
    XNN_ALIGN(8) uint32_t exp_bias[2];
    XNN_ALIGN(8) float scale_to_inf[2];
    XNN_ALIGN(8) uint32_t expw_max[2];
    XNN_ALIGN(8) float scale_to_zero[2];
    XNN_ALIGN(8) int16_t bias_min[4];
    XNN_ALIGN(8) uint32_t manth_mask[2];
    XNN_ALIGN(8) uint32_t exph_mask[2];
    XNN_ALIGN(8) uint16_t nanh[4];
  } wasmsimd;
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
};

union xnn_f32_qs8_cvt_params {
  struct {
    float scale;
    float output_min_less_zero_point;
    float output_max_less_zero_point;
    float magic_bias;
    int32_t magic_bias_less_zero_point;
  } scalar_fmagic;
  struct {
    float scale;
    float magic_bias;
    int32_t magic_min;
    int32_t magic_max;
    int32_t magic_bias_less_zero_point;
  } scalar_imagic;
  struct {
    float scale;
    float output_min_less_zero_point;
    float output_max_less_zero_point;
    int32_t output_zero_point;
  } scalar_lrintf;
#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  struct {
    float scale;
    float magic_bias;
    int32_t magic_bias_less_zero_point;
    int8_t output_min;
    int8_t output_max;
  } neon;
  struct {
    float scale;
    int16_t output_zero_point;
    int8_t output_min;
    int8_t output_max;
  } neonv8;
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64
#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  struct {
    XNN_ALIGN(16) float scale[4];
    XNN_ALIGN(16) float output_max_less_zero_point[4];
    XNN_ALIGN(16) int16_t output_zero_point[8];
    XNN_ALIGN(16) int16_t output_min[8];
  } sse2;
  struct {
    XNN_ALIGN(16) float scale[4];
    XNN_ALIGN(16) float output_max_less_zero_point[4];
    XNN_ALIGN(16) int16_t output_zero_point[8];
    XNN_ALIGN(16) int8_t output_min[16];
  } sse4;
  struct {
    XNN_ALIGN(32) float scale[8];
    XNN_ALIGN(32) float output_max_less_zero_point[8];
    XNN_ALIGN(16) int16_t output_zero_point[8];
    XNN_ALIGN(16) int8_t output_min[16];
    int32_t mask_table[14];
  } avx;
  struct {
    XNN_ALIGN(32) float scale[8];
    XNN_ALIGN(32) float output_max_less_zero_point[8];
    XNN_ALIGN(32) int16_t output_zero_point[16];
    XNN_ALIGN(32) uint32_t shuffle_mask[8];
    XNN_ALIGN(32) int8_t output_min[32];
    int32_t mask_table[14];
  } avx2;
  struct {
    XNN_ALIGN(64) float scale[16];
    XNN_ALIGN(64) float output_max_less_zero_point[16];
    XNN_ALIGN(64) int16_t output_zero_point[32];
    XNN_ALIGN(64) int8_t output_min[64];
    XNN_ALIGN(64) uint32_t shuffle512_mask[16];
    XNN_ALIGN(32) uint32_t shuffle256_mask[8];
  } avx512;
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  struct {
    XNN_ALIGN(8) float scale[2];
    XNN_ALIGN(8) int16_t output_zero_point[4];
    XNN_ALIGN(8) int8_t output_min[8];
    XNN_ALIGN(8) int8_t output_max[8];
  } wasmsimd_cvt;
  struct {
    XNN_ALIGN(8) float scale[2];
    XNN_ALIGN(8) float magic_bias[2];
    XNN_ALIGN(8) int32_t magic_min[2];
    XNN_ALIGN(8) int32_t magic_bias_less_zero_point[2];
    XNN_ALIGN(8) int8_t output_max[8];
  } wasmsimd_magic;
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
};

union xnn_f32_qu8_cvt_params {
  struct {
    float scale;
    float output_min_less_zero_point;
    float output_max_less_zero_point;
    float magic_bias;
    int32_t magic_bias_less_zero_point;
  } scalar_fmagic;
  struct {
    float scale;
    float magic_bias;
    int32_t magic_min;
    int32_t magic_max;
    int32_t magic_bias_less_zero_point;
  } scalar_imagic;
  struct {
    float scale;
    float output_min_less_zero_point;
    float output_max_less_zero_point;
    int32_t output_zero_point;
  } scalar_lrintf;
#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  struct {
    float scale;
    float magic_bias;
    int32_t magic_bias_less_zero_point;
    uint8_t output_min;
    uint8_t output_max;
  } neon;
  struct {
    float scale;
    int16_t output_zero_point;
    uint8_t output_min;
    uint8_t output_max;
  } neonv8;
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64
#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  struct {
    XNN_ALIGN(16) float scale[4];
    XNN_ALIGN(16) float output_max_less_zero_point[4];
    XNN_ALIGN(16) int16_t output_zero_point[8];
    XNN_ALIGN(16) uint8_t output_min[16];
  } sse2;
  struct {
    XNN_ALIGN(32) float scale[8];
    XNN_ALIGN(32) float output_max_less_zero_point[8];
    XNN_ALIGN(16) int16_t output_zero_point[8];
    XNN_ALIGN(16) uint8_t output_min[16];
    int32_t mask_table[14];
  } avx;
  struct {
    XNN_ALIGN(32) float scale[8];
    XNN_ALIGN(32) float output_max_less_zero_point[8];
    XNN_ALIGN(32) int16_t output_zero_point[16];
    XNN_ALIGN(32) uint32_t shuffle_mask[8];
    XNN_ALIGN(32) uint8_t output_min[32];
    int32_t mask_table[14];
  } avx2;
  struct {
    XNN_ALIGN(64) float scale[16];
    XNN_ALIGN(64) float output_max_less_zero_point[16];
    XNN_ALIGN(64) int16_t output_zero_point[32];
    XNN_ALIGN(64) uint8_t output_min[64];
    XNN_ALIGN(64) uint32_t shuffle512_mask[16];
    XNN_ALIGN(32) uint32_t shuffle256_mask[8];
  } avx512;
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  struct {
    XNN_ALIGN(8) float scale[2];
    XNN_ALIGN(8) int16_t output_zero_point[4];
    XNN_ALIGN(8) uint8_t output_min[8];
    XNN_ALIGN(8) uint8_t output_max[8];
  } wasmsimd_cvt;
  struct {
    XNN_ALIGN(8) float scale[2];
    XNN_ALIGN(8) float magic_bias[2];
    XNN_ALIGN(8) int32_t magic_min[2];
    XNN_ALIGN(8) int32_t magic_bias_less_zero_point[2];
    XNN_ALIGN(8) uint8_t output_max[8];
  } wasmsimd_magic;
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
};

union xnn_qs8_cvt_params {
  struct {
    int32_t bias;
    int32_t multiplier;
  } scalar;
#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  struct {
    uint32_t minus_input_zero_point;
    int32_t multiplier;
    int32_t bias;
  } armsimd32;
  struct {
    int16_t input_zero_point;
    int16_t multiplier;
    int16_t output_zero_point;
  } neon;
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64
#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  struct {
    XNN_ALIGN(16) int16_t multiplier[8];
    XNN_ALIGN(16) int32_t bias[4];
  } sse2;
  struct {
    XNN_ALIGN(16) int16_t input_zero_point[8];
    XNN_ALIGN(16) int16_t multiplier[8];
    XNN_ALIGN(16) int16_t output_zero_point[8];
  } ssse3;
  struct {
    XNN_ALIGN(32) int16_t input_zero_point[16];
    XNN_ALIGN(32) int16_t multiplier[16];
    XNN_ALIGN(32) int16_t output_zero_point[16];
  } avx2;
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  struct {
    XNN_ALIGN(8) int16_t input_zero_point[4];
    XNN_ALIGN(8) int16_t multiplier[4];
    XNN_ALIGN(8) int16_t output_zero_point[4];
  } wasmsimd;
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
};

union xnn_qs8_f32_cvt_params {
  struct {
    int32_t zero_point;
    float scale;
  } scalar;
#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  struct {
    int16_t minus_zero_point[2];
    float scale;
  } neon;
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64
#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  struct {
    XNN_ALIGN(16) uint8_t sign_mask[16];
    XNN_ALIGN(16) uint16_t magic_exp[8];
    XNN_ALIGN(16) float magic_bias[4];
    XNN_ALIGN(16) float scale[4];
  } sse2;
  struct {
    XNN_ALIGN(16) int32_t minus_zero_point[4];
    XNN_ALIGN(16) float scale[4];
  } sse4;
  struct {
    XNN_ALIGN(32) int32_t minus_zero_point[8];
    XNN_ALIGN(32) float scale[8];
  } avx;
  struct {
    XNN_ALIGN(64) int32_t minus_zero_point[16];
    XNN_ALIGN(64) float scale[16];
  } avx512;
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  struct {
    XNN_ALIGN(8) int16_t minus_zero_point[4];
    XNN_ALIGN(8) float scale[2];
  } wasmsimd;
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
};

union xnn_qu8_cvt_params {
  struct {
    int32_t bias;
    int32_t multiplier;
  } scalar;
#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  struct {
    uint32_t minus_input_zero_point;
    int32_t multiplier;
    int32_t bias;
  } armsimd32;
  struct {
    uint16_t input_zero_point;
    int16_t multiplier;
    int16_t output_zero_point;
  } neon;
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64
#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  struct {
    XNN_ALIGN(16) uint16_t multiplier[8];
    XNN_ALIGN(16) int32_t bias[4];
  } sse2;
  struct {
    XNN_ALIGN(16) uint16_t input_zero_point[8];
    XNN_ALIGN(16) int16_t multiplier[8];
    XNN_ALIGN(16) int16_t output_zero_point[8];
  } ssse3;
  struct {
    XNN_ALIGN(32) uint16_t input_zero_point[16];
    XNN_ALIGN(32) int16_t multiplier[16];
    XNN_ALIGN(32) int16_t output_zero_point[16];
  } avx2;
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  struct {
    XNN_ALIGN(8) uint16_t input_zero_point[4];
    XNN_ALIGN(8) int16_t multiplier[4];
    XNN_ALIGN(8) int16_t output_zero_point[4];
  } wasmsimd;
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
};

union xnn_qu8_f32_cvt_params {
  struct {
    int32_t zero_point;
    float scale;
  } scalar;
#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  struct {
    int16_t minus_zero_point[2];
    float scale;
  } neon;
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64
#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  struct {
    XNN_ALIGN(16) uint16_t magic_exp[8];
    XNN_ALIGN(16) float magic_bias[4];
    XNN_ALIGN(16) float scale[4];
  } sse2;
  struct {
    XNN_ALIGN(16) int32_t minus_zero_point[4];
    XNN_ALIGN(16) float scale[4];
  } sse4;
  struct {
    XNN_ALIGN(32) int32_t minus_zero_point[8];
    XNN_ALIGN(32) float scale[8];
  } avx;
  struct {
    XNN_ALIGN(64) int32_t minus_zero_point[16];
    XNN_ALIGN(64) float scale[16];
  } avx512;
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  struct {
    XNN_ALIGN(8) int16_t minus_zero_point[4];
    XNN_ALIGN(8) float scale[2];
  } wasmsimd;
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
};


// ELU: used by VELU microkernels.

union xnn_f16_elu_params {
  char _; // Dummy member variable to comply with the C standard
#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  struct {
    uint16_t prescale;
    uint16_t sat_cutoff;
    uint16_t magic_bias;
    uint16_t log2e;
    uint16_t minus_ln2;
    uint16_t c3;
    uint16_t c2;
    uint16_t minus_alpha;
    uint16_t beta;
  } fp16arith_rr1_p3;
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64
#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  struct {
    XNN_ALIGN(32) float prescale[8];
    XNN_ALIGN(32) float sat_cutoff[8];
    XNN_ALIGN(32) float magic_bias[8];
    XNN_ALIGN(32) float log2e[8];
    XNN_ALIGN(32) float minus_ln2[8];
    XNN_ALIGN(32) float c3[8];
    XNN_ALIGN(32) float c2[8];
    XNN_ALIGN(32) float c1[8];
    XNN_ALIGN(32) float alpha[8];
    XNN_ALIGN(32) float beta[8];
  } avx2_rr1_p3;
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
};

union xnn_f32_elu_params {
  struct {
    float prescale;
    float alpha;
    float beta;
    float sat_cutoff;
    float magic_bias;
    float log2e;
    float minus_ln2_hi;
    float minus_ln2_lo;
    float c3;
    float c2;
    float one;
  } scalar_rr2_lut16_p3;
  struct {
    float prescale;
    float alpha;
    float beta;
    float sat_cutoff;
    float magic_bias;
    float log2e;
    float minus_ln2_hi;
    float minus_ln2_lo;
    float c6;
    float c5;
    float c4;
    float c3;
    float c2;
    float one;
  } scalar_rr2_p6;
#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  struct {
    float prescale;
    float alpha;
    float beta;
    float sat_cutoff;
    float magic_bias;
    float log2e;
    float minus_ln2_hi;
    float minus_ln2_lo;
    float c6;
    float c5;
    float c4;
    float c3;
    float c2;
  } neon_rr2_p6;
  struct {
    float prescale;
    float alpha;
    float beta;
    float sat_cutoff;
    float magic_bias;
    float log2e;
    float minus_ln2_hi;
    float minus_ln2_lo;
    float c3;
    float c2;
  } neon_rr2_lut16_p3;
  struct {
    float prescale;
    float alpha;
    float beta;
    float sat_cutoff;
    float magic_bias;
    float log2e;
    float minus_ln2;
    float c6;
    float c5;
    float c4;
    float c3;
    float c2;
  } neonfma_rr1_p6;
  struct {
    float prescale;
    float alpha;
    float beta;
    float sat_cutoff;
    float magic_bias;
    float log2e;
    float minus_ln2;
    float c3;
    float c2;
  } neonfma_rr1_lut16_p3;
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64
#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  struct {
    XNN_ALIGN(16) float prescale[4];
    XNN_ALIGN(16) float alpha[4];
    XNN_ALIGN(16) float beta[4];
    XNN_ALIGN(16) float sat_cutoff[4];
    XNN_ALIGN(16) float magic_bias[4];
    XNN_ALIGN(16) float log2e[4];
    XNN_ALIGN(16) uint32_t index_mask[4];
    XNN_ALIGN(16) float minus_ln2_hi[4];
    XNN_ALIGN(16) float minus_ln2_lo[4];
    XNN_ALIGN(16) float c3[4];
    XNN_ALIGN(16) float c2[4];
    XNN_ALIGN(16) float one[4];
  } sse2_rr2_lut16_p3;
  struct {
    XNN_ALIGN(16) float prescale[4];
    XNN_ALIGN(16) float alpha[4];
    XNN_ALIGN(16) float beta[4];
    XNN_ALIGN(16) float sat_cutoff[4];
    XNN_ALIGN(16) float magic_bias[4];
    XNN_ALIGN(16) float log2e[4];
    XNN_ALIGN(16) float minus_ln2_hi[4];
    XNN_ALIGN(16) float minus_ln2_lo[4];
    XNN_ALIGN(16) float c6[4];
    XNN_ALIGN(16) float c5[4];
    XNN_ALIGN(16) float c4[4];
    XNN_ALIGN(16) float c3[4];
    XNN_ALIGN(16) float c2[4];
    XNN_ALIGN(16) float one[4];
  } sse2_rr2_p6;
  struct {
    XNN_ALIGN(32) float prescale[8];
    XNN_ALIGN(32) float alpha[8];
    XNN_ALIGN(32) float beta[8];
    XNN_ALIGN(32) float sat_cutoff[8];
    XNN_ALIGN(32) float magic_bias[8];
    XNN_ALIGN(32) float log2e[8];
    XNN_ALIGN(32) uint32_t index_mask[8];
    XNN_ALIGN(32) float minus_ln2_hi[8];
    XNN_ALIGN(32) float minus_ln2_lo[8];
    XNN_ALIGN(32) float c3[8];
    XNN_ALIGN(32) float c2[8];
    XNN_ALIGN(32) float one[8];
    int32_t mask_table[14];
  } avx_rr2_lut16_p3;
  struct {
    XNN_ALIGN(32) float prescale[8];
    XNN_ALIGN(32) float alpha[8];
    XNN_ALIGN(32) float beta[8];
    XNN_ALIGN(32) float sat_cutoff[8];
    XNN_ALIGN(32) float magic_bias[8];
    XNN_ALIGN(32) float log2e[8];
    XNN_ALIGN(32) uint32_t index_mask[8];
    XNN_ALIGN(32) float table[8];
    XNN_ALIGN(32) float minus_ln2_hi[8];
    XNN_ALIGN(32) float minus_ln2_lo[8];
    XNN_ALIGN(32) float c4[8];
    XNN_ALIGN(32) float c3[8];
    XNN_ALIGN(32) float c2[8];
    XNN_ALIGN(32) float one[8];
    int32_t mask_table[14];
  } avx_rr2_lut4_p4;
  struct {
    XNN_ALIGN(32) float prescale[8];
    XNN_ALIGN(32) float alpha[8];
    XNN_ALIGN(32) float beta[8];
    XNN_ALIGN(32) float sat_cutoff[8];
    XNN_ALIGN(32) float magic_bias[8];
    XNN_ALIGN(32) float log2e[8];
    XNN_ALIGN(32) float minus_ln2_hi[8];
    XNN_ALIGN(32) float minus_ln2_lo[8];
    XNN_ALIGN(32) float c6[8];
    XNN_ALIGN(32) float c5[8];
    XNN_ALIGN(32) float c4[8];
    XNN_ALIGN(32) float c3[8];
    XNN_ALIGN(32) float c2[8];
    XNN_ALIGN(32) float one[8];
    int32_t mask_table[14];
  } avx_rr2_p6;
  struct {
    XNN_ALIGN(32) float prescale[8];
    XNN_ALIGN(32) float alpha[8];
    XNN_ALIGN(32) float beta[8];
    XNN_ALIGN(32) float sat_cutoff[8];
    XNN_ALIGN(32) float magic_bias[8];
    XNN_ALIGN(32) float log2e[8];
    XNN_ALIGN(32) uint32_t index_mask[8];
    XNN_ALIGN(32) float minus_ln2[8];
    XNN_ALIGN(32) float c3[8];
    XNN_ALIGN(32) float c2[8];
    int32_t mask_table[14];
  } avx2_rr1_lut16_p3;
  struct {
    XNN_ALIGN(32) float prescale[8];
    XNN_ALIGN(32) float alpha[8];
    XNN_ALIGN(32) float beta[8];
    XNN_ALIGN(32) float sat_cutoff[8];
    XNN_ALIGN(32) float magic_bias[8];
    XNN_ALIGN(32) float log2e[8];
    XNN_ALIGN(32) uint32_t table[8];
    XNN_ALIGN(32) float minus_ln2[8];
    XNN_ALIGN(32) float c4[8];
    XNN_ALIGN(32) float c3[8];
    XNN_ALIGN(32) float c2[8];
    int32_t mask_table[14];
  } avx2_rr1_lut8_p4;
  struct {
    XNN_ALIGN(32) float prescale[8];
    XNN_ALIGN(32) float alpha[8];
    XNN_ALIGN(32) float beta[8];
    XNN_ALIGN(32) float sat_cutoff[8];
    XNN_ALIGN(32) float magic_bias[8];
    XNN_ALIGN(32) float log2e[8];
    XNN_ALIGN(32) float table[8];
    XNN_ALIGN(32) float minus_ln2[8];
    XNN_ALIGN(32) float c4[8];
    XNN_ALIGN(32) float c3[8];
    XNN_ALIGN(32) float c2[8];
    int32_t mask_table[14];
  } avx2_rr1_lut4_p4;
  struct {
    XNN_ALIGN(32) float prescale[8];
    XNN_ALIGN(32) float alpha[8];
    XNN_ALIGN(32) float beta[8];
    XNN_ALIGN(32) float sat_cutoff[8];
    XNN_ALIGN(32) float magic_bias[8];
    XNN_ALIGN(32) float log2e[8];
    XNN_ALIGN(32) float minus_ln2[8];
    XNN_ALIGN(32) float c6[8];
    XNN_ALIGN(32) float c5[8];
    XNN_ALIGN(32) float c4[8];
    XNN_ALIGN(32) float c3[8];
    XNN_ALIGN(32) float c2[8];
    int32_t mask_table[14];
  } avx2_rr1_p6;
  struct {
    float prescale;
    float alpha;
    float beta;
    float sat_cutoff;
    float magic_bias;
    float log2e;
    float minus_ln2;
    float c3;
    float c2;
    XNN_ALIGN(64) uint32_t table[16];
  } avx512_rr1_lut16_p3;
  struct {
    float prescale;
    float alpha;
    float beta;
    float sat_cutoff;
    float magic_bias;
    float log2e;
    float minus_ln2;
    float c6;
    float c5;
    float c4;
    float c3;
    float c2;
  } avx512_rr1_p6;
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  struct {
    XNN_ALIGN(8) float prescale[2];
    XNN_ALIGN(8) float alpha[2];
    XNN_ALIGN(8) float beta[2];
    XNN_ALIGN(8) float sat_cutoff[2];
    XNN_ALIGN(8) float magic_bias[2];
    XNN_ALIGN(8) float log2e[2];
    XNN_ALIGN(8) uint32_t index_mask[2];
    XNN_ALIGN(8) float minus_ln2_hi[2];
    XNN_ALIGN(8) float minus_ln2_lo[2];
    XNN_ALIGN(8) float c3[2];
    XNN_ALIGN(8) float c2[2];
    XNN_ALIGN(8) float one[2];
  } wasmsimd_rr2_lut16_p3;
  struct {
    XNN_ALIGN(8) float prescale[2];
    XNN_ALIGN(8) float alpha[2];
    XNN_ALIGN(8) float beta[2];
    XNN_ALIGN(8) float sat_cutoff[2];
    XNN_ALIGN(8) float magic_bias[2];
    XNN_ALIGN(8) float log2e[2];
    XNN_ALIGN(8) float minus_ln2_hi[2];
    XNN_ALIGN(8) float minus_ln2_lo[2];
    XNN_ALIGN(8) float c6[2];
    XNN_ALIGN(8) float c5[2];
    XNN_ALIGN(8) float c4[2];
    XNN_ALIGN(8) float c3[2];
    XNN_ALIGN(8) float c2[2];
    XNN_ALIGN(8) float one[2];
  } wasmsimd_rr2_p6;
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
};


// ExpMinus: used by RADDEXPMINUSMAX microkernels.

union xnn_f16_expminus_params {
  char _; // Dummy member variable to comply with the C standard
#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  struct {
    uint16_t magic_bias;
    uint16_t log2e;
    uint16_t minus_ln2_hi;
    uint16_t minus_ln2_lo;
    uint16_t c2;
    uint16_t c1;
    uint16_t denorm_cutoff;
  } fp16arith_rr2_p2;
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64
#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  struct {
    XNN_ALIGN(32) float magic_bias[8];
    XNN_ALIGN(32) float log2e[8];
    XNN_ALIGN(32) float minus_ln2[8];
    XNN_ALIGN(32) float c2[8];
    XNN_ALIGN(32) float c1[8];
    XNN_ALIGN(32) float denorm_cutoff[8];
  } avx2_rr1_p2;
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
};

union xnn_f32_expminus_params {
  struct {
    float log2e;
    float magic_bias;
    float minus_ln2_hi;
    float minus_ln2_lo;
    float c5;
    float c4;
    float c3;
    float c2;
    float c1;
    float denorm_cutoff;
  } scalar_rr2_p5;
  struct {
    float log2e;
    float magic_bias;
    float minus_ln2_hi;
    float minus_ln2_lo;
    float c2;
    float denorm_cutoff;
  } scalar_rr2_lut64_p2;
#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  struct {
    float log2e;
    float magic_bias;
    float minus_ln2_hi;
    float minus_ln2_lo;
    float c5;
    float c4;
    float c3;
    float c2;
    float c1;
    float denorm_cutoff;
  } neon_rr2_p5;
  struct {
    float log2e;
    float magic_bias;
    float minus_ln2_hi;
    float minus_ln2_lo;
    float c2;
    float denorm_cutoff;
  } neon_rr2_lut64_p2;
  struct {
    float log2e;
    float magic_bias;
    float minus_ln2;
    float c5;
    float c4;
    float c3;
    float c2;
    float c1;
    float denorm_cutoff;
  } neonfma_rr1_p5;
  struct {
    float log2e;
    float magic_bias;
    float minus_ln2;
    float c2;
    float denorm_cutoff;
  } neonfma_rr1_lut64_p2;
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64
#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  struct {
    XNN_ALIGN(16) float log2e[4];
    XNN_ALIGN(16) float magic_bias[4];
    XNN_ALIGN(16) float minus_ln2_hi[4];
    XNN_ALIGN(16) float minus_ln2_lo[4];
    XNN_ALIGN(16) float c5[4];
    XNN_ALIGN(16) float c4[4];
    XNN_ALIGN(16) float c3[4];
    XNN_ALIGN(16) float c2[4];
    XNN_ALIGN(16) float c1[4];
    XNN_ALIGN(16) float denorm_cutoff[4];
  } sse2_rr2_p5;
  struct {
    XNN_ALIGN(32) float log2e[8];
    XNN_ALIGN(32) float magic_bias[8];
    XNN_ALIGN(32) float minus_ln2[8];
    XNN_ALIGN(32) float c5[8];
    XNN_ALIGN(32) float c4[8];
    XNN_ALIGN(32) float c3[8];
    XNN_ALIGN(32) float c2[8];
    XNN_ALIGN(32) float c1[8];
    XNN_ALIGN(32) float denorm_cutoff[8];
    int32_t mask_table[14];
  } avx2_rr1_p5;
  struct {
    float log2e;
    float minus_ln2;
    float c5;
    float c4;
    float c3;
    float c2;
    float c1;
    float c0;
  } avx512_rr1_p5;
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  struct {
    XNN_ALIGN(8) float log2e[2];
    XNN_ALIGN(8) float magic_bias[2];
    XNN_ALIGN(8) float minus_ln2_hi[2];
    XNN_ALIGN(8) float minus_ln2_lo[2];
    XNN_ALIGN(8) float c5[2];
    XNN_ALIGN(8) float c4[2];
    XNN_ALIGN(8) float c3[2];
    XNN_ALIGN(8) float c2[2];
    XNN_ALIGN(8) float c1[2];
    XNN_ALIGN(8) float denorm_cutoff[2];
  } wasmsimd_rr2_p5;
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
};


// HSwish: used by VHSWISH microkernels.

union xnn_f16_hswish_params {
#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  struct {
    uint16_t sixth;
    uint16_t three;
    uint16_t six;
    uint16_t pad;  // pad to 8 bytes for neonfp16arith assembly.
  } fp16arith;
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64 */
#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  struct {
    XNN_ALIGN(32) float sixth[8];
    XNN_ALIGN(32) float three[8];
    XNN_ALIGN(16) uint16_t six[8];
  } avx;
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
};

union xnn_f32_hswish_params {
  struct {
    float sixth;
    float three;
    float six;
  } scalar;
#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  struct {
    XNN_ALIGN(16) float sixth[4];
    XNN_ALIGN(16) float half[4];
    XNN_ALIGN(16) float one[4];
  } sse;
  struct {
    XNN_ALIGN(32) float sixth[8];
    XNN_ALIGN(32) float half[8];
    XNN_ALIGN(32) float one[8];
    int32_t mask_table[14];
  } avx;
  struct {
    float sixth;
    float half;
    float one;
  } avx512;
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  struct {
    XNN_ALIGN(8) float sixth[2];
    XNN_ALIGN(8) float three[2];
    XNN_ALIGN(8) float six[2];
  } wasmsimd;
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
};


// LReLU (Leaky ReLU): used by VLRELU microkernels.

union xnn_f16_lrelu_params {
#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  struct {
    uint16_t slope;
  } fp16arith;
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64
#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  struct {
    XNN_ALIGN(32) float slope[8];
  } avx;
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
};

union xnn_f32_lrelu_params {
  struct {
    float slope;
  } scalar;
#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  struct {
    XNN_ALIGN(16) float slope[4];
  } sse;
  struct {
    XNN_ALIGN(32) float slope[8];
    int32_t mask_table[14];
  } avx;
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  struct {
    XNN_ALIGN(8) float slope[2];
  } wasmsimd;
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
};

union xnn_qs8_lrelu_params {
  struct {
    int32_t input_zero_point;
    int32_t positive_multiplier;
    int32_t negative_multiplier;
    int32_t bias;
  } scalar_select;
  struct {
    int32_t input_zero_point;
    int32_t multiplier_diff;
    int32_t multiplier_base;
    int32_t bias;
  } scalar_andxor;
#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  struct {
    uint32_t input_zero_point;
    uint32_t positive_multiplier;
    uint32_t negative_multiplier;
    int32_t bias;
  } armsimd32;
  struct {
    int16_t input_zero_point;
    int16_t positive_multiplier;
    int16_t negative_multiplier;
    int16_t output_zero_point;
  } neon;
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64
#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  struct {
    XNN_ALIGN(16) int16_t input_zero_point[8];
    XNN_ALIGN(16) int16_t multiplier_diff[8];
    XNN_ALIGN(16) int16_t multiplier_base[8];
    XNN_ALIGN(16) int16_t output_zero_point[8];
  } sse2;
  struct {
    XNN_ALIGN(16) int16_t input_zero_point[8];
    XNN_ALIGN(16) int16_t positive_multiplier[8];
    XNN_ALIGN(16) int16_t negative_multiplier[8];
    XNN_ALIGN(16) int16_t output_zero_point[8];
  } avx;
  struct {
    XNN_ALIGN(32) int16_t input_zero_point[16];
    XNN_ALIGN(32) int16_t positive_multiplier[16];
    XNN_ALIGN(32) int16_t negative_multiplier[16];
    XNN_ALIGN(32) int16_t output_zero_point[16];
  } avx2;
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  struct {
    XNN_ALIGN(8) int16_t input_zero_point[4];
    XNN_ALIGN(8) int16_t positive_multiplier[4];
    XNN_ALIGN(8) int16_t negative_multiplier[4];
    XNN_ALIGN(8) int16_t output_zero_point[4];
  } wasmsimd_arm;
  struct {
    XNN_ALIGN(8) int16_t input_zero_point[4];
    XNN_ALIGN(8) int16_t multiplier_diff[4];
    XNN_ALIGN(8) int16_t multiplier_base[4];
    XNN_ALIGN(8) int16_t output_zero_point[4];
  } wasmsimd_x86;
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
};

union xnn_qu8_lrelu_params {
  struct {
    int32_t input_zero_point;
    int32_t positive_multiplier;
    int32_t negative_multiplier;
    int32_t bias;
  } scalar_select;
  struct {
    int32_t input_zero_point;
    int32_t multiplier_base;
    int32_t multiplier_diff;
    int32_t bias;
  } scalar_andxor;
#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  struct {
    uint32_t input_zero_point;
    uint32_t positive_multiplier;
    uint32_t negative_multiplier;
    int32_t bias;
  } armsimd32;
  struct {
    uint16_t input_zero_point;
    int16_t positive_multiplier;
    int16_t negative_multiplier;
    int16_t output_zero_point;
  } neon;
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64
#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  struct {
    XNN_ALIGN(16) int16_t input_zero_point[8];
    XNN_ALIGN(16) int16_t multiplier_diff[8];
    XNN_ALIGN(16) int16_t multiplier_base[8];
    XNN_ALIGN(16) int16_t output_zero_point[8];
  } sse2;
  struct {
    XNN_ALIGN(16) int16_t input_zero_point[8];
    XNN_ALIGN(16) int16_t positive_multiplier[8];
    XNN_ALIGN(16) int16_t negative_multiplier[8];
    XNN_ALIGN(16) int16_t output_zero_point[8];
  } avx;
  struct {
    XNN_ALIGN(32) int16_t input_zero_point[16];
    XNN_ALIGN(32) int16_t positive_multiplier[16];
    XNN_ALIGN(32) int16_t negative_multiplier[16];
    XNN_ALIGN(32) int16_t output_zero_point[16];
  } avx2;
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  struct {
    XNN_ALIGN(8) int16_t input_zero_point[4];
    XNN_ALIGN(8) int16_t positive_multiplier[4];
    XNN_ALIGN(8) int16_t negative_multiplier[4];
    XNN_ALIGN(8) int16_t output_zero_point[4];
  } wasmsimd_arm;
  struct {
    XNN_ALIGN(8) int16_t input_zero_point[4];
    XNN_ALIGN(8) int16_t multiplier_diff[4];
    XNN_ALIGN(8) int16_t multiplier_base[4];
    XNN_ALIGN(8) int16_t output_zero_point[4];
  } wasmsimd_x86;
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
};


// Neg: used by VNEG microkernels.

union xnn_f16_neg_params {
  char _; // Dummy member variable to comply with the C standard
#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  struct {
    XNN_ALIGN(16) uint16_t sign_mask[8];
  } sse;
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
};

union xnn_f32_neg_params {
  char _; // Dummy member variable to comply with the C standard
#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  struct {
    XNN_ALIGN(16) float sign_mask[4];
  } sse;
  struct {
    XNN_ALIGN(32) float sign_mask[8];
    int32_t mask_table[14];
  } avx;
  struct {
    uint32_t sign_mask;
  } avx512;
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  struct {
    XNN_ALIGN(8) float sign_mask[2];
  } wasmsimd;
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
};


// Rnd (Round): used by VRNDNE/VRNDU/VRNDD/VRNDZ microkernels.

union xnn_f16_rnd_params {
  char _; // Dummy member variable to comply with the C standard
};

union xnn_f32_rnd_params {
  char _; // Dummy member variable to comply with the C standard
#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  struct {
    XNN_ALIGN(16) float sign_mask[4];
    XNN_ALIGN(16) float one[4];
  } sse2;
  struct {
    int32_t mask_table[14];
  } avx;
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
};


// Sigmoid: used by VSIGMOID microkernels.

union xnn_f16_sigmoid_params {
  char _; // Dummy member variable to comply with the C standard
#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  struct {
    uint16_t magic_bias;
    uint16_t minus_log2e;
    uint16_t ln2_hi;
    uint16_t ln2_lo;
    uint16_t c2;
    uint16_t c1;
    uint16_t denorm_cutoff;
  } fp16arith_rr2_p2;
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64
#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  struct {
    XNN_ALIGN(32) float sign_mask[8];
    XNN_ALIGN(32) float magic_bias[8];
    XNN_ALIGN(32) float log2e[8];
    XNN_ALIGN(32) float minus_ln2[8];
    XNN_ALIGN(32) float c2[8];
    XNN_ALIGN(32) float c1[8];
    XNN_ALIGN(32) float one[8];
    XNN_ALIGN(32) float denorm_cutoff[8];
  } avx2_rr1_p2;
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
};

union xnn_f32_sigmoid_params {
  struct {
    float magic_bias;
    float minus_log2e;
    float ln2_hi;
    float ln2_lo;
    float c1;
    float one;
    float denorm_cutoff;
  } scalar_rr2_lut2048_p1;
  struct {
    float magic_bias;
    float minus_log2e;
    float ln2_hi;
    float ln2_lo;
    float c2;
    float one;
    float denorm_cutoff;
  } scalar_rr2_lut64_p2;
  struct {
    float magic_bias;
    float minus_log2e;
    float ln2_hi;
    float ln2_lo;
    float c5;
    float c4;
    float c3;
    float c2;
    float c1;
    float one;
    float denorm_cutoff;
  } scalar_rr2_p5;
#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  struct {
    float magic_bias;
    float minus_log2e;
    float ln2_hi;
    float ln2_lo;
    float c1;
    float denorm_cutoff;
  } neon_rr2_lut2048_p1;
  struct {
    float magic_bias;
    float minus_log2e;
    float ln2_hi;
    float ln2_lo;
    float c2;
    float denorm_cutoff;
  } neon_rr2_lut64_p2;
  struct {
    float magic_bias;
    float minus_log2e;
    float ln2_hi;
    float ln2_lo;
    float c5;
    float c4;
    float c3;
    float c2;
    float c1;
    float denorm_cutoff;
  } neon_rr2_p5;
  struct {
    float magic_bias;
    float minus_log2e;
    float ln2;
    float c1;
    float denorm_cutoff;
  } neonfma_rr1_lut2048_p1;
  struct {
    float magic_bias;
    float minus_log2e;
    float ln2;
    float c2;
    float denorm_cutoff;
  } neonfma_rr1_lut64_p2;
  struct {
    float magic_bias;
    float minus_log2e;
    float ln2;
    float c5;
    float c4;
    float c3;
    float c2;
    float c1;
    float denorm_cutoff;
  } neonfma_rr1_p5;
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64
#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  struct {
    XNN_ALIGN(16) float sign_mask[4];
    XNN_ALIGN(16) float magic_bias[4];
    XNN_ALIGN(16) float log2e[4];
    XNN_ALIGN(16) uint32_t index_mask[4];
    XNN_ALIGN(16) float minus_ln2_hi[4];
    XNN_ALIGN(16) float minus_ln2_lo[4];
    XNN_ALIGN(16) float c2[4];
    XNN_ALIGN(16) float one[4];
    XNN_ALIGN(16) float denorm_cutoff[4];
  } sse2_rr2_lut64_p2;
  struct {
    XNN_ALIGN(16) float sign_mask[4];
    XNN_ALIGN(16) float magic_bias[4];
    XNN_ALIGN(16) float log2e[4];
    XNN_ALIGN(16) float minus_ln2_hi[4];
    XNN_ALIGN(16) float minus_ln2_lo[4];
    XNN_ALIGN(16) float c5[4];
    XNN_ALIGN(16) float c4[4];
    XNN_ALIGN(16) float c3[4];
    XNN_ALIGN(16) float c2[4];
    XNN_ALIGN(16) float c1[4];
    XNN_ALIGN(16) float one[4];
    XNN_ALIGN(16) float denorm_cutoff[4];
  } sse2_rr2_p5;
  struct {
    XNN_ALIGN(32) float sign_mask[8];
    XNN_ALIGN(32) float magic_bias[8];
    XNN_ALIGN(32) float log2e[8];
    XNN_ALIGN(32) float minus_ln2_hi[8];
    XNN_ALIGN(32) float minus_ln2_lo[8];
    XNN_ALIGN(32) float c5[8];
    XNN_ALIGN(32) float c4[8];
    XNN_ALIGN(32) float c3[8];
    XNN_ALIGN(32) float c2[8];
    XNN_ALIGN(32) float c1[8];
    XNN_ALIGN(32) float one[8];
    XNN_ALIGN(32) float two[8];
    XNN_ALIGN(32) float denorm_cutoff[8];
    int32_t mask_table[14];
  } avx_rr2_p5;
  struct {
    XNN_ALIGN(32) float sign_mask[8];
    XNN_ALIGN(32) float magic_bias[8];
    XNN_ALIGN(32) float log2e[8];
    XNN_ALIGN(32) float minus_ln2[8];
    XNN_ALIGN(32) float c5[8];
    XNN_ALIGN(32) float c4[8];
    XNN_ALIGN(32) float c3[8];
    XNN_ALIGN(32) float c2[8];
    XNN_ALIGN(32) float c1[8];
    XNN_ALIGN(32) float one[8];
    XNN_ALIGN(32) float denorm_cutoff[8];
    int32_t mask_table[14];
  } avx2_rr1_p5;
  struct {
    uint32_t sign_mask;
    float magic_bias;
    float log2e;
    float minus_ln2;
    float c3;
    float c2;
    float one;
    XNN_ALIGN(64) float table[16];
  } avx512_rr1_lut16_p3;
  struct {
    uint32_t sign_mask;
    float magic_bias;
    float log2e;
    float minus_ln2_hi;
    float minus_ln2_lo;
    float c2;
    float c1;
    float one;
    XNN_ALIGN(64) float table_lo[16];
    XNN_ALIGN(64) float table_hi[16];
  } avx512_rr2_lut32_p2;
  struct {
    uint32_t sign_mask;
    float log2e;
    float minus_ln2;
    float c5;
    float c4;
    float c3;
    float c2;
    float c1;
    float one;
  } avx512_rr1_p5;
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  struct {
    XNN_ALIGN(8) float magic_bias[2];
    XNN_ALIGN(8) float minus_log2e[2];
    XNN_ALIGN(8) uint32_t index_mask[2];
    XNN_ALIGN(8) float ln2_hi[2];
    XNN_ALIGN(8) float ln2_lo[2];
    XNN_ALIGN(8) float c2[2];
    XNN_ALIGN(8) float one[2];
    XNN_ALIGN(8) float denorm_cutoff[2];
  } wasmsimd_rr2_lut64_p2;
  struct {
    XNN_ALIGN(8) float magic_bias[2];
    XNN_ALIGN(8) float minus_log2e[2];
    XNN_ALIGN(8) float ln2_hi[2];
    XNN_ALIGN(8) float ln2_lo[2];
    XNN_ALIGN(8) float c5[2];
    XNN_ALIGN(8) float c4[2];
    XNN_ALIGN(8) float c3[2];
    XNN_ALIGN(8) float c2[2];
    XNN_ALIGN(8) float c1[2];
    XNN_ALIGN(8) float one[2];
    XNN_ALIGN(8) float denorm_cutoff[2];
  } wasmsimd_rr2_p5;
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
};


// Sqrt (Square Root): used by VSQRT microkernels.

union xnn_f16_sqrt_params {
  char _; // Dummy member variable to comply with the C standard
};

union xnn_f32_sqrt_params {
  char _; // Dummy member variable to comply with the C standard
#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  struct {
    int32_t mask_table[14];
  } avx;
  struct {
    XNN_ALIGN(32) float half[8];
    int32_t mask_table[14];
  } fma;
  struct {
    float half;
  } avx512;
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
};

// CHW: used by CONV/DWCONV microkernels in CHW layout with Min+Max parameters.

union xnn_f16_chw_params {
  char _; // Dummy member variable to comply with the C standard
#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  struct {
    uint16_t min;
    uint16_t max;
    XNN_ALIGN(16) uint16_t mask_even[8]; // used by stride 2 kernels
    XNN_ALIGN(16) uint16_t mask_odd[8];  // used by stride 2 kernels
    XNN_ALIGN(16) uint16_t mask[8]; // used by stride 1 x8 kernels
  } neonfp16arith;
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64
};

union xnn_f32_chw_params {
  struct {
    XNN_ALIGN(16) int32_t mask_even[4]; // used by stride 2 kernels
    XNN_ALIGN(16) int32_t mask_odd[4];  // used by stride 2 kernels
    XNN_ALIGN(16) int32_t mask[4]; // used by stride 1 kernels
    float min;
    float max;
  } scalar;
#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  struct {
    float min;
    float max;
    XNN_ALIGN(16) uint32_t mask_even[4]; // used by stride 2 kernels
    XNN_ALIGN(16) uint32_t mask_odd[4];  // used by stride 2 kernels
    XNN_ALIGN(16) uint32_t mask[4]; // used by stride 1 kernels
  } neon;
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64
#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  struct {
    XNN_ALIGN(16) float min[4];
    XNN_ALIGN(16) float max[4];
    XNN_ALIGN(16) uint32_t mask_even[4]; // used by stride 2 kernels
    XNN_ALIGN(16) uint32_t mask_odd[4];  // used by stride 2 kernels
    XNN_ALIGN(16) uint32_t mask[4]; // used by stride 1 kernels
  } sse;
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
};


// GAvgPool (Global Average Pool): used by GAVGPOOL microkernels in CHW layout with Scale+Min+Max parameters.

union xnn_f16_gavgpool_params {
  char _; // Dummy member variable to comply with the C standard
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

// JIT GEMM: used by GEMM/IGEMM microkernel generators.

struct jit_gemm_params {
  struct {
    float min;
    float max;
  } f32_minmax;
  size_t num_post_operations;
  const struct xnn_post_operation* post_operations;
};
