// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include <xnnpack.h>
#include <xnnpack/common.h>

struct xnn_f16_scaleminmax_params {
  uint16_t scale;
  uint16_t min;
  uint16_t max;
};

union xnn_f32_default_params {
  // Empty; serves to differentiate pointer types for micro-kernels without fused activation.
  char _; // Dummy member variable to comply with the C standard
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
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
};

union xnn_f32_spchw_params {
  struct {
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

union xnn_u8_minmax_params {
  struct {
    int32_t min;
    int32_t max;
  } scalar;
#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  struct {
    uint8_t min;
    uint8_t max;
  } neon;
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64
#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  struct {
    XNN_ALIGN(16) uint8_t min[16];
    XNN_ALIGN(16) uint8_t max[16];
  } sse2;
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
  } sse2;
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
};

union xnn_f32_gavgpool_params {
  struct {
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
    XNN_ALIGN(16) float multiplier;
    XNN_ALIGN(16) float output_min;
    XNN_ALIGN(16) float output_max;
    XNN_ALIGN(16) uint32_t mask[4];
  } neon;
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64 */
};

union xnn_f32_hswish_params {
  struct {
    float sixth;
    float half;
    float one;
  } scalar;
#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  struct {
    XNN_ALIGN(16) float sixth[4];
    XNN_ALIGN(16) float half[4];
    XNN_ALIGN(16) float one[4];
  } sse;
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
};

union xnn_q8_gemm_params {
  struct {
    int32_t kernel_zero_point;
    int32_t input_zero_point;
    int32_t multiplier;
    int32_t remainder_mask;
    int32_t remainder_threshold;
    uint32_t shift;
    int32_t output_min_less_zero_point;
    int32_t output_max_less_zero_point;
    int32_t output_zero_point;
  } scalar;
#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  struct {
    int16_t kernel_zero_point;
    int16_t input_zero_point;
    int32_t multiplier;
    int32_t right_shift;
    int16_t output_zero_point;
    uint8_t output_min;
    uint8_t output_max;
  } neon;
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64
#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  struct {
    XNN_ALIGN(16) int16_t kernel_zero_point[8];
    XNN_ALIGN(16) int16_t input_zero_point[8];
    XNN_ALIGN(16) uint32_t multiplier[4];
    XNN_ALIGN(16) uint64_t rounding[2];
    XNN_ALIGN(16) int32_t remainder_mask[4];
    XNN_ALIGN(16) int32_t remainder_threshold[4];
    XNN_ALIGN(16) uint64_t shift[2];
    XNN_ALIGN(16) int16_t output_zero_point[8];
    XNN_ALIGN(16) uint8_t output_min[16];
    XNN_ALIGN(16) uint8_t output_max[16];
  } sse2;
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
};

union xnn_q8_add_params {
  struct {
    int32_t zero_point_product;
    uint32_t a_multiplier;
    uint32_t b_multiplier;
    uint32_t shift;
    int32_t remainder_mask;
    int32_t remainder_threshold;
    int32_t y_zero_point;
    int32_t y_min;
    int32_t y_max;
  } scalar;
#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  struct {
    uint8_t a_zero_point;
    uint8_t b_zero_point;
    int16_t y_zero_point;
    int32_t a_multiplier;
    int32_t b_multiplier;
    int32_t right_shift;
    uint8_t y_min;
    uint8_t y_max;
  } neon;
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64
#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  struct {
    XNN_ALIGN(16) int32_t zero_point_product[4];
    XNN_ALIGN(16) uint16_t a_multiplier_lo[8];
    XNN_ALIGN(16) uint16_t a_multiplier_hi[8];
    XNN_ALIGN(16) uint16_t b_multiplier_lo[8];
    XNN_ALIGN(16) uint16_t b_multiplier_hi[8];
    XNN_ALIGN(16) int32_t remainder_mask[4];
    XNN_ALIGN(16) int32_t remainder_threshold[4];
    XNN_ALIGN(16) int16_t y_zero_point[8];
    XNN_ALIGN(16) uint8_t y_min[16];
    XNN_ALIGN(16) uint8_t y_max[16];
    uint32_t shift;
    uint32_t a_multiplier;
    uint32_t b_multiplier;
  } sse2;
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
};

union xnn_q8_avgpool_params {
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

union xnn_fp32_requantization_params {
  struct {
    float scale;
    float min_less_zero_point;
    float max_less_zero_point;
    float magic;
    int32_t magic_less_zero_point;
  } scalar;
  struct {
    float scale;
    float min;
    float max;
    float magic;
    int32_t magic_less_zero_point;
  } neon;
  struct {
    float scale;
    int16_t zero_point;
    uint8_t min;
    uint8_t max;
  } neonv8;
  struct {
    XNN_ALIGN(16) float scale[4];
    XNN_ALIGN(16) int16_t zero_point[8];
    XNN_ALIGN(16) uint8_t min[16];
    XNN_ALIGN(16) uint8_t max[16];
  } sse2;
  struct {
    XNN_ALIGN(16) float scale[4];
    XNN_ALIGN(16) float min_less_zero_point[4];
    XNN_ALIGN(16) float max_less_zero_point[4];
    XNN_ALIGN(16) float magic[4];
    XNN_ALIGN(16) int32_t magic_less_zero_point[4];
  } psimd;
};

union xnn_precise_requantization_params {
  struct {
    uint32_t multiplier;
    uint32_t rounding_lo;
    uint32_t rounding_hi;
    uint32_t shift_less_32;
    int32_t min_less_zero_point;
    int32_t max_less_zero_point;
    int32_t zero_point;
  } scalar;
  struct {
    int32_t multiplier;
    int32_t right_shift;
    int16_t zero_point;
    uint8_t min;
    uint8_t max;
  } neon;
  struct {
    XNN_ALIGN(16) uint32_t multiplier[4];
    XNN_ALIGN(16) uint64_t rounding[2];
    XNN_ALIGN(16) uint32_t shift[4];
    XNN_ALIGN(16) int16_t zero_point[8];
    XNN_ALIGN(16) uint8_t min[16];
    XNN_ALIGN(16) uint8_t max[16];
  } sse2;
};

union xnn_q31_requantization_params {
  struct {
    int32_t multiplier;
    int32_t remainder_mask;
    int32_t remainder_threshold;
    uint32_t shift;
    int32_t min_less_zero_point;
    int32_t max_less_zero_point;
    int32_t zero_point;
  } scalar;
#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  struct {
    int32_t multiplier;
    int32_t right_shift;
    int16_t zero_point;
    uint8_t min;
    uint8_t max;
  } neon;
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64
#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  struct {
    XNN_ALIGN(16) uint32_t multiplier[4];
    XNN_ALIGN(16) uint64_t rounding[2];
    XNN_ALIGN(16) int32_t remainder_mask[4];
    XNN_ALIGN(16) int32_t remainder_threshold[4];
    XNN_ALIGN(16) uint64_t shift[2];
    XNN_ALIGN(16) int16_t zero_point[8];
    XNN_ALIGN(16) uint8_t min[16];
    XNN_ALIGN(16) uint8_t max[16];
  } sse2;
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
};

union xnn_requantization_params {
  union xnn_precise_requantization_params precise;
  union xnn_fp32_requantization_params fp32;
  union xnn_q31_requantization_params q31;
};

typedef void (*xnn_ppmm_ukernel_function)(
    size_t mr,
    size_t nc,
    size_t kc,
    const void* a,
    const void* w,
    void* c,
    size_t cm_stride,
    size_t cn_stride,
    const void* params);

typedef void (*xnn_f32_ppmm_minmax_ukernel_function)(
    size_t mr,
    size_t nc,
    size_t kc,
    const float* a,
    const float* w,
    float* c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_minmax_params* params);

typedef void (*xnn_f16_ppmm_ukernel_function)(
    size_t mr,
    size_t nc,
    size_t kc,
    const void* a,
    const void* w,
    void* c,
    size_t cm_stride,
    size_t cn_stride,
    const struct xnn_f16_scaleminmax_params* params);

typedef void (*xnn_gemm_ukernel_function)(
    size_t mr,
    size_t nr,
    size_t k,
    const void* a,
    size_t a_stride,
    const void* w,
    void* c,
    size_t cm_stride,
    size_t cn_stride,
    const void* params);

typedef void (*xnn_f32_gemm_ukernel_function)(
    size_t mr,
    size_t nr,
    size_t k,
    const float* a,
    size_t a_stride,
    const float* w,
    float* c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_default_params* params);

typedef void (*xnn_f32_gemm_minmax_ukernel_function)(
    size_t mr,
    size_t nr,
    size_t k,
    const float* a,
    size_t a_stride,
    const float* w,
    float* c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_minmax_params* params);

typedef void (*xnn_f32_gemminc_minmax_ukernel_function)(
    size_t mr,
    size_t nr,
    size_t k,
    const float* a,
    size_t a_stride,
    const float* w,
    float* c,
    size_t cm_stride,
    size_t cn_stride,
    const float* acc,
    const union xnn_f32_minmax_params* params);

typedef void (*xnn_f16_gemm_minmax_ukernel_function)(
    size_t mr,
    size_t nr,
    size_t k,
    const void* a,
    size_t a_stride,
    const void* w,
    void* c,
    size_t cm_stride,
    size_t cn_stride,
    const struct xnn_f16_scaleminmax_params* params);

typedef void (*xnn_f16_igemm_minmax_ukernel_function)(
    size_t mr,
    size_t nr,
    size_t kc,
    size_t ks,
    const void** a,
    const void* w,
    void* c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const void* zero,
    const struct xnn_f16_scaleminmax_params* params);

typedef void (*xnn_q8_gemm_ukernel_function)(
    size_t mr,
    size_t nr,
    size_t k,
    const uint8_t* a,
    size_t a_stride,
    const void* w,
    uint8_t* c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_q8_gemm_params* params);

typedef void (*xnn_igemm_ukernel_function)(
    size_t mr,
    size_t nr,
    size_t kc,
    size_t ks,
    const void** a,
    const void* w,
    void* c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const void* zero,
    const void* params);

typedef void (*xnn_f32_igemm_ukernel_function)(
    size_t mr,
    size_t nr,
    size_t kc,
    size_t ks,
    const float** a,
    const float* w,
    float* c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const float* zero,
    const union xnn_f32_default_params* params);

typedef void (*xnn_f32_igemm_minmax_ukernel_function)(
    size_t mr,
    size_t nr,
    size_t kc,
    size_t ks,
    const float** a,
    const float* w,
    float* c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const float* zero,
    const union xnn_f32_minmax_params* params);

typedef void (*xnn_q8_igemm_ukernel_function)(
    size_t mr,
    size_t nr,
    size_t kc,
    size_t ks,
    const uint8_t** a,
    const void* w,
    uint8_t* c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const uint8_t* zero,
    const union xnn_q8_gemm_params* params);

typedef void (*xnn_conv_hwc_ukernel_function)(
    size_t input_height,
    size_t input_width,
    size_t output_y_start,
    size_t output_y_end,
    const void* input,
    const void* zero,
    const void* weights,
    void* output,
    size_t input_padding_top,
    size_t output_channels,
    size_t output_height_stride,
    size_t output_width_stride,
    const void* params);

typedef void (*xnn_f32_conv_hwc_ukernel_function)(
    size_t input_height,
    size_t input_width,
    size_t output_y_start,
    size_t output_y_end,
    const float* input,
    const float* zero,
    const float* weights,
    float* output,
    size_t input_padding_top,
    size_t output_channels,
    size_t output_height_stride,
    size_t output_width_stride,
    const union xnn_f32_minmax_params* params);

typedef void (*xnn_conv_hwc2spchw_ukernel_function)(
    size_t input_height,
    size_t input_width,
    size_t output_y_start,
    size_t output_y_end,
    const void* input,
    const void* zero,
    const void* weights,
    void* output,
    size_t input_padding_top,
    size_t output_channels,
    size_t output_height_stride,
    size_t output_channel_stride,
    const void* params);

typedef void (*xnn_f32_conv_hwc2spchw_ukernel_function)(
    size_t input_height,
    size_t input_width,
    size_t output_y_start,
    size_t output_y_end,
    const float* input,
    const float* zero,
    const float* weights,
    float* output,
    size_t input_padding_top,
    size_t output_channels,
    size_t output_height_stride,
    size_t output_channel_stride,
    const union xnn_f32_minmax_params* params);

typedef void (*xnn_spmm_ukernel_function)(
    uint32_t m,
    uint32_t n,
    const void* a,
    const void* w,
    const int32_t* dmap,
    const uint32_t* nmap,
    void* c,
    const void* params);

typedef void (*xnn_f16_spmm_minmax_ukernel_function)(
    uint32_t m,
    uint32_t n,
    const void* a,
    const void* w,
    const int32_t* dmap,
    const uint32_t* nmap,
    void* c,
    const struct xnn_f16_scaleminmax_params* params);

typedef void (*xnn_f32_spmm_minmax_ukernel_function)(
    uint32_t m,
    uint32_t n,
    const float* a,
    const float* w,
    const int32_t* dmap,
    const uint32_t* nmap,
    float* c,
    const union xnn_f32_minmax_params* params);

typedef void (*xnn_packx_ukernel_function)(
    size_t m,
    size_t k,
    const void* x,
    size_t x_stride,
    void* y);

typedef void (*xnn_x32_packx_ukernel_function)(
    size_t m,
    size_t k,
    const uint32_t* x,
    size_t x_stride,
    uint32_t* y);

typedef void (*xnn_pad_ukernel_function)(
    size_t m,
    size_t n,
    size_t l,
    size_t r,
    uint32_t c,
    const void* x,
    size_t x_stride,
    void* y,
    size_t y_stride);

typedef void (*xnn_unpool_ukernel_function)(
    size_t p,
    size_t c,
    uint32_t f,
    const void* input,
    const uint32_t* index,
    void** output);

typedef void (*xnn_x32_unpool_ukernel_function)(
    size_t p,
    size_t c,
    uint32_t f,
    const uint32_t* input,
    const uint32_t* index,
    uint32_t** output);

typedef void (*xnn_zipc_ukernel_function)(
    size_t n,
    const void* x,
    void* y);

typedef void (*xnn_x8_zipc_ukernel_function)(
    size_t n,
    const uint8_t* x,
    uint8_t* y);

typedef void (*xnn_x32_zipc_ukernel_function)(
    size_t n,
    const uint32_t* x,
    uint32_t* y);

typedef void (*xnn_zipv_ukernel_function)(
    size_t n,
    size_t m,
    const void* x,
    void* y);

typedef void (*xnn_x8_zipv_ukernel_function)(
    size_t n,
    size_t m,
    const uint8_t* x,
    uint8_t* y);

typedef void (*xnn_x32_zipv_ukernel_function)(
    size_t n,
    size_t m,
    const uint32_t* x,
    uint32_t* y);

typedef void (*xnn_x8_lut_ukernel_function)(
    size_t n,
    const uint8_t* x,
    const uint8_t* t,
    uint8_t* y);

typedef void (*xnn_dwconv_spchw_ukernel_function)(
    size_t output_height,
    size_t input_width,
    const void* input,
    const void* weights,
    void* output,
    size_t input_tuple_stride,
    size_t output_tuple_stride,
    size_t input_height_stride,
    size_t output_height_stride,
    const void* params);

typedef void (*xnn_f32_dwconv_spchw_ukernel_function)(
    size_t output_height,
    size_t input_width,
    const float* input,
    const float* weights,
    float* output,
    size_t input_tuple_stride,
    size_t output_tuple_stride,
    size_t input_height_stride,
    size_t output_height_stride,
    const union xnn_f32_spchw_params* params);

typedef void (*xnn_dwconv_unipass_ukernel_function)(
    size_t channels,
    size_t output_width,
    const void** input,
    const void* weights,
    void* output,
    size_t input_stride,
    size_t output_increment,
    const void* params);

typedef void (*xnn_f32_dwconv_unipass_ukernel_function)(
    size_t channels,
    size_t output_width,
    const float** input,
    const float* weights,
    float* output,
    size_t input_stride,
    size_t output_increment,
    const union xnn_f32_default_params* params);

typedef void (*xnn_f32_dwconv_minmax_unipass_ukernel_function)(
    size_t channels,
    size_t output_width,
    const float** input,
    const float* weights,
    float* output,
    size_t input_stride,
    size_t output_increment,
    const union xnn_f32_minmax_params* params);

typedef void (*xnn_q8_dwconv_minmax_unipass_ukernel_function)(
    size_t channels,
    size_t output_width,
    const uint8_t** input,
    const void* weights,
    uint8_t* output,
    size_t input_stride,
    size_t output_increment,
    const union xnn_q8_gemm_params* params);

typedef void (*xnn_dwconv_multipass_ukernel_function)(
    size_t channels,
    size_t output_width,
    const void** input,
    const void* weights,
    void* buffer,
    void* output,
    size_t input_stride,
    size_t output_increment,
    const void* params);

typedef void (*xnn_f32_ibilinear_ukernel_function)(
    size_t output_pixels,
    size_t channels,
    const float** input,
    size_t input_offset,
    const float* weights,
    float* output,
    size_t output_increment);

typedef void (*xnn_ibilinear_ukernel_function)(
    size_t output_pixels,
    size_t channels,
    const void** input,
    size_t input_offset,
    const void* weights,
    void* output,
    size_t output_increment);

typedef void (*xnn_gavgpool_unipass_ukernel_function)(
    size_t rows,
    size_t channels,
    const void* input,
    size_t input_stride,
    const void* zero,
    void* output,
    const void* params);

typedef void (*xnn_f32_gavgpool_minmax_unipass_ukernel_function)(
    size_t rows,
    size_t channels,
    const float* input,
    size_t input_stride,
    const float* zero,
    float* output,
    const union xnn_f32_scaleminmax_params* params);

typedef void (*xnn_q8_gavgpool_minmax_unipass_ukernel_function)(
    size_t rows,
    size_t channels,
    const uint8_t* input,
    size_t input_stride,
    const uint8_t* zero,
    uint8_t* output,
    const union xnn_q8_avgpool_params* params);

typedef void (*xnn_gavgpool_multipass_ukernel_function)(
    size_t rows,
    size_t channels,
    const void* input,
    size_t input_stride,
    const void* zero,
    void* buffer,
    void* output,
    const void* params);

typedef void (*xnn_f32_gavgpool_minmax_multipass_ukernel_function)(
    size_t rows,
    size_t channels,
    const float* input,
    size_t input_stride,
    const float* zero,
    float* buffer,
    float* output,
    const union xnn_f32_scaleminmax_params* params);

typedef void (*xnn_q8_gavgpool_minmax_multipass_ukernel_function)(
    size_t rows,
    size_t channels,
    const uint8_t* input,
    size_t input_stride,
    const uint8_t* zero,
    int32_t* buffer,
    uint8_t* output,
    const union xnn_q8_avgpool_params* params);

typedef void (*xnn_gavgpool_spchw_ukernel_function)(
    size_t elements,
    size_t channels,
    const float* input,
    float* output,
    const void* params);

typedef void (*xnn_f32_gavgpool_spchw_ukernel_function)(
    size_t elements,
    size_t channels,
    const float* input,
    float* output,
    const union xnn_f32_gavgpool_params* params);

typedef void (*xnn_avgpool_unipass_ukernel_function)(
    size_t output_pixels,
    size_t kernel_elements,
    size_t channels,
    const void** input,
    size_t input_offset,
    const void* zero,
    void* output,
    size_t input_increment,
    size_t output_increment,
    const void* params);

typedef void (*xnn_f32_avgpool_minmax_unipass_ukernel_function)(
    size_t output_pixels,
    size_t kernel_elements,
    size_t channels,
    const float** input,
    size_t input_offset,
    const float* zero,
    float* output,
    size_t input_increment,
    size_t output_increment,
    const union xnn_f32_scaleminmax_params* params);

typedef void (*xnn_q8_avgpool_minmax_unipass_ukernel_function)(
    size_t output_pixels,
    size_t kernel_elements,
    size_t channels,
    const uint8_t** input,
    size_t input_offset,
    const uint8_t* zero,
    uint8_t* output,
    size_t input_increment,
    size_t output_increment,
    const union xnn_q8_avgpool_params* params);

typedef void (*xnn_avgpool_multipass_ukernel_function)(
    size_t output_pixels,
    size_t kernel_elements,
    size_t channels,
    const void** input,
    size_t input_offset,
    const void* zero,
    void* buffer,
    void* output,
    size_t input_increment,
    size_t output_increment,
    const void* params);

typedef void (*xnn_f32_avgpool_minmax_multipass_ukernel_function)(
    size_t output_pixels,
    size_t kernel_elements,
    size_t channels,
    const float** input,
    size_t input_offset,
    const float* zero,
    float* buffer,
    float* output,
    size_t input_increment,
    size_t output_increment,
    const union xnn_f32_scaleminmax_params* params);

typedef void (*xnn_q8_avgpool_minmax_multipass_ukernel_function)(
    size_t output_pixels,
    size_t kernel_elements,
    size_t channels,
    const uint8_t** input,
    size_t input_offset,
    const uint8_t* zero,
    int32_t* buffer,
    uint8_t* output,
    size_t input_increment,
    size_t output_increment,
    const union xnn_q8_avgpool_params* params);

typedef void (*xnn_pavgpool_unipass_ukernel_function)(
    size_t output_pixels,
    size_t kernel_elements,
    size_t channels,
    const void** input,
    size_t input_offset,
    const void* zero,
    const void* multiplier,
    void* output,
    size_t input_increment,
    size_t output_increment,
    const void* params);

typedef void (*xnn_f32_pavgpool_minmax_unipass_ukernel_function)(
    size_t output_pixels,
    size_t kernel_elements,
    size_t channels,
    const float** input,
    size_t input_offset,
    const float* zero,
    const float* multiplier,
    float* output,
    size_t input_increment,
    size_t output_increment,
    const union xnn_f32_minmax_params* params);

typedef void (*xnn_pavgpool_multipass_ukernel_function)(
    size_t output_pixels,
    size_t kernel_elements,
    size_t channels,
    const void** input,
    size_t input_offset,
    const void* zero,
    const void* multiplier,
    void* buffer,
    void* output,
    size_t input_increment,
    size_t output_increment,
    const void* params);

typedef void (*xnn_f32_pavgpool_minmax_multipass_ukernel_function)(
    size_t output_pixels,
    size_t kernel_elements,
    size_t channels,
    const float** input,
    size_t input_offset,
    const float* zero,
    const float* multiplier,
    float* buffer,
    float* output,
    size_t input_increment,
    size_t output_increment,
    const union xnn_f32_minmax_params* params);

typedef void (*xnn_maxpool_ukernel_function)(
    size_t output_pixels,
    size_t kernel_elements,
    size_t channels,
    const void** input,
    size_t input_offset,
    void* output,
    size_t input_increment,
    size_t output_increment,
    const void* params);

typedef void (*xnn_f32_maxpool_ukernel_function)(
    size_t output_pixels,
    size_t kernel_elements,
    size_t channels,
    const float** input,
    size_t input_offset,
    float* output,
    size_t input_increment,
    size_t output_increment,
    const union xnn_f32_minmax_params* params);

typedef void (*xnn_u8_maxpool_ukernel_function)(
    size_t output_pixels,
    size_t kernel_elements,
    size_t channels,
    const uint8_t** input,
    size_t input_offset,
    uint8_t* output,
    size_t input_increment,
    size_t output_increment,
    const union xnn_u8_minmax_params* params);

typedef void (*xnn_argmaxpool_unipass_ukernel_function)(
    size_t output_pixels,
    size_t kernel_elements,
    size_t channels,
    const void** input,
    size_t input_offset,
    void* output,
    uint32_t* index,
    size_t input_increment,
    size_t output_increment,
    const void* params);

typedef void (*xnn_f32_argmaxpool_unipass_ukernel_function)(
    size_t output_pixels,
    size_t kernel_elements,
    size_t channels,
    const float** input,
    size_t input_offset,
    float* output,
    uint32_t* index,
    size_t input_increment,
    size_t output_increment,
    const union xnn_f32_minmax_params* params);

typedef void (*xnn_argmaxpool_multipass_ukernel_function)(
    size_t output_pixels,
    size_t kernel_elements,
    size_t channels,
    const void** input,
    size_t input_offset,
    void* accumulation_buffer,
    uint32_t* index_buffer,
    void* output,
    uint32_t* index,
    size_t input_increment,
    size_t output_increment,
    const void* params);

typedef void (*xnn_f32_argmaxpool_multipass_ukernel_function)(
    size_t output_pixels,
    size_t kernel_elements,
    size_t channels,
    const float** input,
    size_t input_offset,
    float* accumulation_buffer,
    uint32_t* index_buffer,
    float* output,
    uint32_t* index,
    size_t input_increment,
    size_t output_increment,
    const union xnn_f32_minmax_params* params);

typedef void (*xnn_univector_ukernel_function)(
    size_t n,
    const void* x,
    void* y,
    const void* params);

typedef void (*xnn_f32_clamp_ukernel_function)(
    size_t n,
    const float* x,
    float* y,
    const union xnn_f32_minmax_params* params);

typedef void (*xnn_u8_clamp_ukernel_function)(
    size_t n,
    const uint8_t* x,
    uint8_t* y,
    const union xnn_u8_minmax_params* params);

typedef void (*xnn_f32_hswish_ukernel_function)(
    size_t n,
    const float* x,
    float* y,
    const union xnn_f32_hswish_params* params);

typedef void (*xnn_rmax_ukernel_function)(
    size_t n,
    const void* x,
    void* y);

typedef void (*xnn_u8_rmax_ukernel_function)(
    size_t n,
    const uint8_t* x,
    uint8_t* y);

typedef void (*xnn_f32_rmax_ukernel_function)(
    size_t n,
    const float* x,
    float* y);

typedef void (*xnn_u8_lut32norm_ukernel_function)(
    size_t n,
    const uint8_t* x,
    const uint32_t* t,
    uint8_t* y);

typedef void (*xnn_vadd_ukernel_function)(
    size_t n,
    const void* a,
    const void* b,
    void* y,
    const void* params);

typedef void (*xnn_q8_vadd_minmax_ukernel_function)(
    size_t n,
    const uint8_t* a,
    const uint8_t* b,
    uint8_t* y,
    const union xnn_q8_add_params* params);

typedef void (*xnn_vbinary_ukernel_function)(
    size_t n,
    const void* a,
    const void* b,
    void* y,
    const void* params);

typedef void (*xnn_f32_vbinary_ukernel_function)(
    size_t n,
    const float* a,
    const float* b,
    float* y,
    const union xnn_f32_default_params* params);

typedef void (*xnn_f32_vbinary_minmax_ukernel_function)(
    size_t n,
    const float* a,
    const float* b,
    float* y,
    const union xnn_f32_minmax_params* params);

typedef void (*xnn_vunary_ukernel_function)(
    size_t n,
    const void* x,
    void* y,
    const void* params);

typedef void (*xnn_f32_vunary_ukernel_function)(
    size_t n,
    const float* x,
    float* y,
    const void* params);

typedef void (*xnn_vmulcaddc_ukernel_function)(
    size_t m,
    size_t c,
    const void* x,
    size_t x_stride,
    const void* w,
    void* y,
    size_t y_stride,
    const void* params);

typedef void (*xnn_f32_vmulcaddc_ukernel_function)(
    size_t m,
    size_t c,
    const float* x,
    size_t x_stride,
    const float* w,
    float* y,
    size_t y_stride,
    const union xnn_f32_minmax_params* params);

typedef void (*xnn_prelu_ukernel_function)(
    size_t mr,
    size_t n,
    const void* x,
    size_t x_stride,
    const void* w,
    void* y,
    size_t y_stride);

typedef void (*xnn_f32_prelu_ukernel_function)(
    size_t mr,
    size_t n,
    const float* x,
    size_t x_stride,
    const float* w,
    float* y,
    size_t y_stride);

typedef void (*xnn_f32_raddexpminusmax_ukernel_function)(
    size_t n,
    const float* input,
    float* sum,
    float max);

typedef void (*xnn_f32_raddstoreexpminusmax_ukernel_function)(
    size_t n,
    const float* input,
    float* output,
    float* sum,
    float max);

typedef void (*xnn_f32_vscaleexpminusmax_ukernel_function)(
    size_t n,
    const float* input,
    float* output,
    float max,
    float scale);

typedef void (*xnn_f32_vscale_ukernel_function)(
    size_t n,
    const float* x,
    float* y,
    float c);

// Reduce-Add Extended ("mantissa" + "exponent") Exponentials
typedef void (*xnn_f32_raddextexp_ukernel_function)(
    size_t n,
    const float* input,
    float* sum);

// Vector Scale Extended ("mantissa" + "exponent") Exponentials
typedef void (*xnn_f32_vscaleextexp_ukernel_function)(
    size_t n,
    const float* input,
    float* output,
    float scale_mantissa,
    float scale_exponent);

struct xnn_hmp_gemm_ukernel {
  xnn_gemm_ukernel_function function[XNN_MAX_UARCH_TYPES];
};

static inline struct xnn_hmp_gemm_ukernel xnn_init_hmp_gemm_ukernel(xnn_gemm_ukernel_function function) {
  struct xnn_hmp_gemm_ukernel ukernel = { function };
  for (size_t i = 1; i < XNN_MAX_UARCH_TYPES; i++) {
    ukernel.function[i] = function;
  }
  return ukernel;
}

static inline bool xnn_is_hmp_gemm_ukernel(struct xnn_hmp_gemm_ukernel ukernel) {
#if XNN_MAX_UARCH_TYPES == 1
  return false;
#else
  uintptr_t default_function = (uintptr_t) ukernel.function[XNN_UARCH_DEFAULT];
  uintptr_t difference = 0;
  for (size_t i = 1; i < XNN_MAX_UARCH_TYPES; i++) {
    difference |= (default_function ^ (uintptr_t) ukernel.function[i]);
  }
  return difference != 0;
#endif
}

struct xnn_hmp_igemm_ukernel {
  xnn_igemm_ukernel_function function[XNN_MAX_UARCH_TYPES];
};

static inline struct xnn_hmp_igemm_ukernel xnn_init_hmp_igemm_ukernel(xnn_igemm_ukernel_function function) {
  struct xnn_hmp_igemm_ukernel ukernel = { function };
  for (size_t i = 1; i < XNN_MAX_UARCH_TYPES; i++) {
    ukernel.function[i] = function;
  }
  return ukernel;
}

static inline bool xnn_is_hmp_igemm_ukernel(struct xnn_hmp_igemm_ukernel ukernel) {
#if XNN_MAX_UARCH_TYPES == 1
  return false;
#else
  uintptr_t default_function = (uintptr_t) ukernel.function[XNN_UARCH_DEFAULT];
  uintptr_t difference = 0;
  for (size_t i = 1; i < XNN_MAX_UARCH_TYPES; i++) {
    difference |= (default_function ^ (uintptr_t) ukernel.function[i]);
  }
  return difference != 0;
#endif
}

struct gemm_fused_ukernels {
  struct xnn_hmp_gemm_ukernel gemm;
  struct xnn_hmp_igemm_ukernel igemm;
  // Optional GEMM and IGEMM micro-kernels with MR=1 and the same NR and KR parameters.
  struct xnn_hmp_gemm_ukernel gemm1;
  struct xnn_hmp_igemm_ukernel igemm1;
};

struct gemm_parameters {
  struct gemm_fused_ukernels minmax;
  struct gemm_fused_ukernels linear;
  uint8_t mr;
  uint8_t nr;
  uint8_t log2_kr;
  uint8_t log2_sr;
};

struct vbinary_parameters {
  xnn_vbinary_ukernel_function op_ukernel;
  xnn_vbinary_ukernel_function opc_ukernel;
  xnn_vbinary_ukernel_function ropc_ukernel;
  // Number of elements in a tile.
  // For best efficiency, micro-kernel must process a multiple of this number of elements in each call.
  uint8_t element_tile;
};

struct spmm_parameters {
  xnn_spmm_ukernel_function ukernel;
  // Number of M-dimension elements in a tile.
  // Corresponds to a block of pixels in 1x1 Convolution and a block of batch size in Fully Connected operator.
  uint8_t mr;
  // Number of N-dimension elements in a tile.
  // Corresponds to a block of output channels/features in 1x1 Convolution and Fully Connected operator.
  uint8_t nr;
};

struct hwc2spchw_dconv_parameters {
  xnn_conv_hwc2spchw_ukernel_function ukernel_with_symm_padding;
  // Number of output channels in a tile.
  // This parameter must be passed as is to weight packing function.
  uint8_t output_channel_tile;
  // Number of output height pixels in a tile.
  // For best efficiency, micro-kernel must produce a multiple of this number of rows in each call.
  uint8_t output_height_tile;
  // Number of output width pixes in a tile.
  uint8_t output_width_tile;
};

struct spchw_dwconv_parameters {
  xnn_dwconv_spchw_ukernel_function ukernel;
  // Number of input width pixels in a tile.
  uint8_t input_width_tile;
  // Number of output width pixels in a tile.
  uint8_t output_width_tile;
  // Number of output height pixels in a tile.
  // For best efficiency, micro-kernel must produce a multiple of this number of rows in each call.
  uint8_t output_height_tile;
};

struct spchw_gavgpool_parameters {
  xnn_gavgpool_spchw_ukernel_function ukernel;
  // Number of channels in a tile.
  // For best efficiency, micro-kernel must process a multiple of this number of channels in each call.
  uint8_t channel_tile;
};

union dwconv_fused_ukernels {
  xnn_dwconv_unipass_ukernel_function unipass;
  xnn_dwconv_multipass_ukernel_function multipass;
};

struct dwconv_parameters {
  union dwconv_fused_ukernels minmax;
  union dwconv_fused_ukernels linear;
  uint8_t channel_tile;
  uint8_t primary_tile;
  uint8_t incremental_tile;
};

struct gavgpool_parameters {
  xnn_gavgpool_unipass_ukernel_function up;
  xnn_gavgpool_multipass_ukernel_function mp;
  uint8_t mr;
};

struct avgpool_parameters {
  xnn_avgpool_unipass_ukernel_function up;
  xnn_avgpool_multipass_ukernel_function mp;
  uint8_t mr;
  uint8_t qr;
};

struct pavgpool_parameters {
  xnn_pavgpool_unipass_ukernel_function up;
  xnn_pavgpool_multipass_ukernel_function mp;
  uint8_t mr;
  uint8_t qr;
};

struct argmaxpool_parameters {
  union {
    xnn_argmaxpool_unipass_ukernel_function up;
    xnn_argmaxpool_multipass_ukernel_function mp;
  };
  uint8_t mr;
  uint8_t qr;
};

struct maxpool_parameters {
  xnn_maxpool_ukernel_function ukernel;
  uint8_t mr;
  uint8_t qr;
};

struct ibilinear_parameters {
  xnn_ibilinear_ukernel_function ukernel;
  // Number of output pixels in a tile.
  // For best efficiency, micro-kernel must produce a multiple of this number of pixels in each call.
  uint8_t pixel_tile;
  // Number of channels in a tile.
  // For best efficiency, micro-kernel must process a multiple of this number of channels in each call.
  uint8_t channel_tile;
};

struct zip_parameters {
  xnn_zipc_ukernel_function x2;
  xnn_zipc_ukernel_function x3;
  xnn_zipc_ukernel_function x4;
  xnn_zipv_ukernel_function xm;
};

struct prelu_parameters {
  xnn_prelu_ukernel_function ukernel;
  uint16_t row_tile;
  uint16_t channel_tile;
};

struct pad_parameters {
  xnn_pad_ukernel_function ukernel;
  uint8_t mr;
};

struct vmulcaddc_parameters {
  xnn_vmulcaddc_ukernel_function ukernel;
  uint8_t channel_tile;
  uint8_t row_tile;
};

#define XNN_MAX_Q8_DWCONV_UKERNELS 1
#define XNN_MAX_F32_DWCONV_UKERNELS 3
#define XNN_MAX_F32_ARGMAXPOOL_UKERNELS 3

struct xnn_parameters {
  bool initialized;
  struct xnn_allocator allocator;
  struct {
    struct gemm_parameters gemm;
    struct dwconv_parameters dwconv[XNN_MAX_Q8_DWCONV_UKERNELS];
    struct avgpool_parameters avgpool;
    struct gavgpool_parameters gavgpool;
    xnn_vadd_ukernel_function vadd;
  } q8;
  struct {
    struct maxpool_parameters maxpool;
    xnn_univector_ukernel_function clamp;
    xnn_u8_lut32norm_ukernel_function lut32norm;
    xnn_u8_rmax_ukernel_function rmax;
  } u8;
  struct {
    xnn_x8_lut_ukernel_function lut;
    struct zip_parameters zip;
  } x8;
  struct {
    struct gemm_parameters gemm;
    struct gemm_parameters gemm2;
    struct dwconv_parameters dwconv[XNN_MAX_F32_DWCONV_UKERNELS];
    struct avgpool_parameters avgpool;
    struct pavgpool_parameters pavgpool;
    struct gavgpool_parameters gavgpool;
    struct maxpool_parameters maxpool;
    struct argmaxpool_parameters argmaxpool[XNN_MAX_F32_ARGMAXPOOL_UKERNELS];
    // Bilinear interpolation (2D).
    struct ibilinear_parameters ibilinear;
    xnn_univector_ukernel_function clamp;
    xnn_univector_ukernel_function hswish;
    xnn_univector_ukernel_function sigmoid;
    struct prelu_parameters prelu;
    struct vbinary_parameters vadd;
    struct vbinary_parameters vdiv;
    struct vbinary_parameters vmax;
    struct vbinary_parameters vmin;
    struct vbinary_parameters vmul;
    struct vbinary_parameters vsub;
    struct vmulcaddc_parameters vmulcaddc;
    xnn_f32_raddstoreexpminusmax_ukernel_function raddstoreexpminusmax;
    xnn_f32_rmax_ukernel_function rmax;
    // Sparse Matrix-Dense Matrix Multiplication (NR=1 block).
    struct spmm_parameters spmm;
    // Sparse Matrix-Dense Matrix Multiplication (NR=2 block).
    struct spmm_parameters spmm2;
    // Sparse Matrix-Dense Matrix Multiplication (NR=4 block).
    struct spmm_parameters spmm4;
    // Direct 3x3 stride-2 Convolution with 3 input channels and HWC->SpCHW layout conversion.
    struct hwc2spchw_dconv_parameters hwc2spchw_dconv3x3c3s2;
    // Direct 3x3 stride-1 Convolution with padding 1 on left and right in SpCHW layout.
    struct spchw_dwconv_parameters spchw_dwconv3x3;
    // Direct 3x3 stride-2 Convolution with padding 1 on left and right in SpCHW layout.
    struct spchw_dwconv_parameters spchw_dwconv3x3s2;
    // Direct 5x5 stride-1 Convolution with padding 2 on left and right in SpCHW layout.
    struct spchw_dwconv_parameters spchw_dwconv5x5;
    // Direct 5x5 stride-2 Convolution with padding 2 on left and right in SpCHW layout.
    struct spchw_dwconv_parameters spchw_dwconv5x5s2;
    // Global Average Pooling in SpCHW layout.
    struct spchw_gavgpool_parameters spchw_gavgpool;
  } f32;
  struct {
    struct pad_parameters pad;
    xnn_unpool_ukernel_function unpool;
    struct zip_parameters zip;
  } x32;
};

#ifdef __cplusplus
extern "C" XNN_INTERNAL struct xnn_parameters xnn_params;
#else
extern XNN_INTERNAL struct xnn_parameters xnn_params;
#endif
