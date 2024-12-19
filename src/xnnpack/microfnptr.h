// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "xnnpack.h"
#include "xnnpack/common.h"
#include "xnnpack/math.h"
#include "xnnpack/microparams.h"

/****************** Microkernel pointers for dense inference *****************/

// CONV-HWC: direct CONVolution in HWC layout

typedef void (*xnn_conv_hwc_ukernel_fn)(
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

typedef void (*xnn_f32_conv_hwc_ukernel_fn)(
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
    const union xnn_f32_minmax_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

// GEMM: GEneral Matrix Multiplication without activations

typedef void (*xnn_gemm_ukernel_fn)(
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

typedef void (*xnn_dqgemm_ukernel_fn)(
    size_t mr,
    size_t nr,
    size_t k,
    const void* a,
    size_t a_stride,
    const void* w,
    void* c,
    size_t cm_stride,
    size_t cn_stride,
    const void* params,
    const struct xnn_qd8_quantization_params* quantization_params);

typedef void (*xnn_dqgemm_bl_ukernel_fn)(
    size_t mr,
    size_t nr,
    size_t k,
    const void* a,
    size_t a_stride,
    const void* w,
    void* c,
    size_t cm_stride,
    size_t cn_stride,
    const void* params,
    const struct xnn_qd8_quantization_params* quantization_params);

typedef void (*xnn_f32_gemm_ukernel_fn)(
    size_t mr,
    size_t nr,
    size_t k,
    const float* a,
    size_t a_stride,
    const float* w,
    float* c,
    size_t cm_stride,
    size_t cn_stride,
    const struct xnn_f32_default_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

typedef void (*xnn_f32_qc8w_gemm_ukernel_fn)(
    size_t mr,
    size_t nr,
    size_t k,
    const float* a,
    size_t a_stride,
    const void* w,
    float* c,
    size_t cm_stride,
    size_t cn_stride,
    const struct xnn_f32_default_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

// GEMM: GEneral Matrix Multiplication with ReLU activation

typedef void (*xnn_f32_gemm_relu_ukernel_fn)(
    size_t mr,
    size_t nr,
    size_t k,
    const float* a,
    size_t a_stride,
    const float* w,
    float* c,
    size_t cm_stride,
    size_t cn_stride,
    const struct xnn_f32_relu_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

typedef void (*xnn_f32_qc8w_gemm_relu_ukernel_fn)(
    size_t mr,
    size_t nr,
    size_t k,
    const float* a,
    size_t a_stride,
    const void* w,
    float* c,
    size_t cm_stride,
    size_t cn_stride,
    const struct xnn_f32_relu_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

// GEMM: GEneral Matrix Multiplication with Min+Max activation

typedef void (*xnn_bf16_gemm_minmax_ukernel_fn)(
    size_t mr,
    size_t nr,
    size_t k,
    const xnn_bfloat16* a,
    size_t a_stride,
    const xnn_bfloat16* w,
    xnn_bfloat16* c,
    size_t cm_stride,
    size_t cn_stride,
    const struct xnn_bf16_minmax_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

typedef void (*xnn_f16_gemm_minmax_ukernel_fn)(
    size_t mr,
    size_t nr,
    size_t k,
    const xnn_float16* a,
    size_t a_stride,
    const xnn_float16* w,
    xnn_float16* c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f16_minmax_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

typedef void (*xnn_f32_gemm_minmax_ukernel_fn)(
    size_t mr,
    size_t nr,
    size_t k,
    const float* a,
    size_t a_stride,
    const float* w,
    float* c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_minmax_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

typedef void (*xnn_f32_gemm_goi_minmax_ukernel_fn)(
    size_t mr,
    size_t nr,
    size_t k,
    const float* a,
    size_t a_stride,
    const float* w,
    float* c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_minmax_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

typedef void (*xnn_f32_qc4w_gemm_minmax_ukernel_fn)(
    size_t mr,
    size_t nr,
    size_t k,
    const float* a,
    size_t a_stride,
    const void* w,
    float* c,
    size_t cm_stride,
    size_t cn_stride,
    const struct xnn_f32_qc4w_minmax_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

typedef void (*xnn_f32_qc8w_gemm_minmax_ukernel_fn)(
    size_t mr,
    size_t nr,
    size_t k,
    const float* a,
    size_t a_stride,
    const void* w,
    float* c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_minmax_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

typedef void (*xnn_qs8_gemm_minmax_ukernel_fn)(
    size_t mr,
    size_t nr,
    size_t k,
    const int8_t* a,
    size_t a_stride,
    const void* w,
    int8_t* c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_qs8_conv_minmax_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

typedef void (*xnn_qd8_f16_qc8w_gemm_ukernel_fn)(
    size_t mr,
    size_t nr,
    size_t k,
    const int8_t* a,
    size_t a_stride,
    const void* w,
    xnn_float16* c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f16_minmax_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)],
    const struct xnn_qd8_quantization_params* quantization_params);

typedef void (*xnn_qd8_f32_qc8w_gemm_ukernel_fn)(
    size_t mr,
    size_t nr,
    size_t k,
    const int8_t* a,
    size_t a_stride,
    const void* w,
    float* c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_minmax_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)],
    const struct xnn_qd8_quantization_params* quantization_params);

typedef void (*xnn_qd8_f16_qc4w_gemm_ukernel_fn)(
    size_t mr,
    size_t nr,
    size_t k,
    const int8_t* a,
    size_t a_stride,
    const void* w,
    xnn_float16* c,
    size_t cm_stride,
    size_t cn_stride,
    const struct xnn_f16_qc4w_minmax_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)],
    const struct xnn_qd8_quantization_params* quantization_params);

typedef void (*xnn_qd8_f16_qb4w_gemm_ukernel_fn)(
    size_t mr,
    size_t nr,
    size_t k,
    const int8_t* a,
    size_t a_stride,
    const void* w,
    xnn_float16* c,
    size_t cm_stride,
    size_t cn_stride,
    const struct xnn_f16_qb4w_minmax_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)],
    const struct xnn_qd8_quantization_params* quantization_params);

typedef void (*xnn_qd8_f32_qc4w_gemm_ukernel_fn)(
    size_t mr,
    size_t nr,
    size_t k,
    const int8_t* a,
    size_t a_stride,
    const void* w,
    float* c,
    size_t cm_stride,
    size_t cn_stride,
    const struct xnn_f32_qc4w_minmax_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)],
    const struct xnn_qd8_quantization_params* quantization_params);

typedef void (*xnn_qd8_f32_qb4w_gemm_ukernel_fn)(
    size_t mr,
    size_t nr,
    size_t k,
    const int8_t* a,
    size_t a_stride,
    const void* w,
    float* c,
    size_t cm_stride,
    size_t cn_stride,
    const struct xnn_f32_qb4w_minmax_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)],
    const struct xnn_qd8_quantization_params* quantization_params);

typedef void (*xnn_qs8_qc8w_gemm_minmax_ukernel_fn)(
    size_t mr,
    size_t nr,
    size_t k,
    const int8_t* a,
    size_t a_stride,
    const void* w,
    int8_t* c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_qs8_qc8w_conv_minmax_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

typedef void (*xnn_qu8_gemm_minmax_ukernel_fn)(
    size_t mr,
    size_t nr,
    size_t k,
    const uint8_t* a,
    size_t a_stride,
    const void* w,
    uint8_t* c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_qu8_conv_minmax_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

// GEMM: GEneral Matrix Multiplication with packed and quantized LHS operand.

typedef void (*xnn_qp8_f32_qc4w_gemm_minmax_ukernel_fn)(
    size_t m,
    size_t n,
    size_t k,
    const void* lhs_packed,
    const void* rhs_packed,
    float* dst,
    size_t dst_stride_row,
    size_t dst_stride_col,
    union xnn_f32_minmax_params
        minmax_params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

typedef void (*xnn_qp8_f32_qc8w_gemm_minmax_ukernel_fn)(
    size_t m,
    size_t n,
    size_t k,
    const void* lhs_packed,
    const void* rhs_packed,
    float* dst,
    size_t dst_stride_row,
    size_t dst_stride_col,
    union xnn_f32_minmax_params
        minmax_params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

typedef void (*xnn_qp8_f32_qb4w_gemm_minmax_ukernel_fn)(
    size_t m,
    size_t n,
    size_t k,
    const void* lhs_packed,
    const void* rhs_packed,
    float* dst,
    size_t dst_stride_row,
    size_t dst_stride_col,
    const struct xnn_f32_qb4w_minmax_params
        minmax_params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

// GEMMINC: GEMM INCremental with Min+Max activation

typedef void (*xnn_f32_gemminc_minmax_ukernel_fn)(
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
    const union xnn_f32_minmax_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

// IGEMM: Indirect GEMM without activation

typedef void (*xnn_dqigemm_ukernel_fn)(
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
    const void* zero_data,
    const void* params,
    const struct xnn_qd8_quantization_params* quantization_params);

typedef void (*xnn_igemm_ukernel_fn)(
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

typedef void (*xnn_f32_igemm_ukernel_fn)(
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
    const struct xnn_f32_default_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

// IGEMM: Indirect GEMM with ReLU activation

typedef void (*xnn_f32_igemm_relu_ukernel_fn)(
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
    const struct xnn_f32_relu_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

// IGEMM: Indirect GEMM with Min+Max activation

typedef void (*xnn_f16_igemm_minmax_ukernel_fn)(
    size_t mr,
    size_t nr,
    size_t kc,
    size_t ks,
    const xnn_float16** a,
    const xnn_float16* w,
    xnn_float16* c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const xnn_float16* zero,
    const union xnn_f16_minmax_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

typedef void (*xnn_f32_igemm_minmax_ukernel_fn)(
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
    const union xnn_f32_minmax_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

typedef void (*xnn_qd8_f16_qc8w_igemm_ukernel_fn)(
    size_t mr,
    size_t nr,
    size_t kc,
    size_t ks,
    const int8_t** a,
    const void* w,
    xnn_float16* c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const int8_t* zero,
    const int8_t* zero_data,
    const union xnn_f16_minmax_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)],
    const struct xnn_qd8_quantization_params* quantization_params);

typedef void (*xnn_qd8_f32_qc8w_igemm_ukernel_fn)(
    size_t mr,
    size_t nr,
    size_t kc,
    size_t ks,
    const int8_t** a,
    const void* w,
    float* c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const int8_t* zero,
    const int8_t* zero_data,
    const union xnn_f32_minmax_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)],
    const struct xnn_qd8_quantization_params* quantization_params);

typedef void (*xnn_qs8_igemm_minmax_ukernel_fn)(
    size_t mr,
    size_t nr,
    size_t kc,
    size_t ks,
    const int8_t** a,
    const void* w,
    int8_t* c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const int8_t* zero,
    const union xnn_qs8_conv_minmax_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

typedef void (*xnn_qs8_qc8w_igemm_minmax_ukernel_fn)(
    size_t mr,
    size_t nr,
    size_t kc,
    size_t ks,
    const int8_t** a,
    const void* w,
    int8_t* c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const int8_t* zero,
    const union xnn_qs8_qc8w_conv_minmax_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

typedef void (*xnn_qu8_igemm_minmax_ukernel_fn)(
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
    const union xnn_qu8_conv_minmax_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

// PPMM: Pre-Packed Matrix Multiplication)

typedef void (*xnn_ppmm_ukernel_fn)(
    size_t mr,
    size_t nc,
    size_t kc,
    const void* a,
    const void* w,
    void* c,
    size_t cm_stride,
    size_t cn_stride,
    const void* params);

typedef void (*xnn_f16_ppmm_ukernel_fn)(
    size_t mr,
    size_t nc,
    size_t kc,
    const xnn_float16* a,
    const xnn_float16* w,
    xnn_float16* c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f16_minmax_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

typedef void (*xnn_f32_ppmm_minmax_ukernel_fn)(
    size_t mr,
    size_t nc,
    size_t kc,
    const float* a,
    const float* w,
    float* c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_minmax_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

// DWCONV: DepthWise CONVolution single-pass without activation

typedef void (*xnn_dwconv_unipass_ukernel_fn)(
    size_t channels,
    size_t output_width,
    const void** input,
    const void* weights,
    void* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const void* zero,
    const void* params);

typedef void (*xnn_f32_dwconv_unipass_ukernel_fn)(
    size_t channels,
    size_t output_width,
    const float** input,
    const float* weights,
    float* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const float* zero,
    const struct xnn_f32_default_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

// DWCONV: DepthWise CONVolution single-pass with Min+Max activation

typedef void (*xnn_f16_dwconv_minmax_unipass_ukernel_fn)(
    size_t channels,
    size_t output_width,
    const xnn_float16** input,
    const xnn_float16* weights,
    xnn_float16* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const xnn_float16* zero,
    const union xnn_f16_minmax_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

typedef void (*xnn_f32_dwconv_minmax_unipass_ukernel_fn)(
    size_t channels,
    size_t output_width,
    const float** input,
    const float* weights,
    float* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const float* zero,
    const union xnn_f32_minmax_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

typedef void (*xnn_qs8_dwconv_minmax_unipass_ukernel_fn)(
    size_t channels,
    size_t output_width,
    const int8_t** input,
    const void* weights,
    int8_t* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const int8_t* zero,
    const union xnn_qs8_conv_minmax_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

typedef void (*xnn_qs8_qc8w_dwconv_minmax_unipass_ukernel_fn)(
    size_t channels,
    size_t output_width,
    const int8_t** input,
    const void* weights,
    int8_t* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const int8_t* zero,
    const union xnn_qs8_qc8w_conv_minmax_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

typedef void (*xnn_qu8_dwconv_minmax_unipass_ukernel_fn)(
    size_t channels,
    size_t output_width,
    const uint8_t** input,
    const void* weights,
    uint8_t* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const uint8_t* zero,
    const union xnn_qu8_conv_minmax_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

// DWCONV: DepthWise CONVolution multi-pass without activation

typedef void (*xnn_dwconv_multipass_ukernel_fn)(
    size_t channels,
    size_t output_width,
    const void** input,
    const void* weights,
    void* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const void* zero,
    size_t kernel_size,
    void* buffer,
    const void* params);

typedef void (*xnn_f32_dwconv_multipass_ukernel_fn)(
    size_t channels,
    size_t output_width,
    const float** input,
    const float* weights,
    float* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const float* zero,
    size_t kernel_size,
    float* buffer,
    const struct xnn_f32_default_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

// DWCONV: DepthWise CONVolution multi-pass with Min+Max activation

typedef void (*xnn_f16_dwconv_minmax_multipass_ukernel_fn)(
    size_t channels,
    size_t output_width,
    const xnn_float16** input,
    const xnn_float16* weights,
    xnn_float16* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const xnn_float16* zero,
    size_t kernel_size,
    xnn_float16* buffer,
    const union xnn_f16_minmax_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

typedef void (*xnn_f32_dwconv_minmax_multipass_ukernel_fn)(
    size_t channels,
    size_t output_width,
    const float** input,
    const float* weights,
    float* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const float* zero,
    size_t kernel_size,
    float* buffer,
    const union xnn_f32_minmax_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

typedef void (*xnn_qs8_dwconv_minmax_multipass_ukernel_fn)(
    size_t channels,
    size_t output_width,
    const int8_t** input,
    const void* weights,
    int8_t* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const int8_t* zero,
    size_t kernel_size,
    int32_t* buffer,
    const union xnn_qs8_conv_minmax_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

typedef void (*xnn_qu8_dwconv_minmax_multipass_ukernel_fn)(
    size_t channels,
    size_t output_width,
    const uint8_t** input,
    const void* weights,
    uint8_t* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const uint8_t* zero,
    size_t kernel_size,
    int32_t* buffer,
    const union xnn_qu8_conv_minmax_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

typedef void (*xnn_qs8_qc8w_dwconv_minmax_multipass_ukernel_fn)(
    size_t channels,
    size_t output_width,
    const int8_t** input,
    const void* weights,
    int8_t* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const int8_t* zero,
    size_t kernel_size,
    int32_t* buffer,
    const union xnn_qs8_qc8w_conv_minmax_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

// VMULCADDC: Vector MULtiply-by-Constant, ADD-Constant

typedef void (*xnn_vmulcaddc_ukernel_fn)(
    size_t batch,
    size_t channels,
    const void* input,
    size_t input_stride,
    const void* weights,
    void* output,
    size_t output_stride,
    const void* params);

typedef void (*xnn_f16_vmulcaddc_ukernel_fn)(
    size_t batch,
    size_t channels,
    const xnn_float16* input,
    size_t input_stride,
    const xnn_float16* weights,
    xnn_float16* output,
    size_t output_stride,
    const union xnn_f16_minmax_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

typedef void (*xnn_f32_vmulcaddc_ukernel_fn)(
    size_t batch,
    size_t channels,
    const float* input,
    size_t input_stride,
    const float* weights,
    float* output,
    size_t output_stride,
    const union xnn_f32_minmax_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

// IBILINEAR: Indirect BILINEAR interpolation

typedef void (*xnn_ibilinear_ukernel_fn)(
    size_t output_pixels,
    size_t channels,
    const void** input,
    size_t input_offset,
    const void* weights,
    void* output,
    size_t output_increment);

typedef void (*xnn_f16_ibilinear_ukernel_fn)(
    size_t output_pixels,
    size_t channels,
    const xnn_float16** input,
    size_t input_offset,
    const xnn_float16* weights,
    xnn_float16* output,
    size_t output_increment);

typedef void (*xnn_f32_ibilinear_ukernel_fn)(
    size_t output_pixels,
    size_t channels,
    const float** input,
    size_t input_offset,
    const float* weights,
    float* output,
    size_t output_increment);

typedef void (*xnn_s8_ibilinear_ukernel_fn)(
    size_t output_pixels,
    size_t channels,
    const int8_t** input,
    size_t input_offset,
    const int16_t* weights,
    int8_t* output,
    size_t output_increment);

typedef void (*xnn_u8_ibilinear_ukernel_fn)(
    size_t output_pixels,
    size_t channels,
    const uint8_t** input,
    size_t input_offset,
    const int16_t* weights,
    uint8_t* output,
    size_t output_increment);

// AVGPOOL: AVeraGe POOLing single-pass

typedef void (*xnn_avgpool_unipass_ukernel_fn)(
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

typedef void (*xnn_f16_avgpool_minmax_unipass_ukernel_fn)(
    size_t output_pixels,
    size_t kernel_elements,
    size_t channels,
    const xnn_float16** input,
    size_t input_offset,
    const xnn_float16* zero,
    xnn_float16* output,
    size_t input_increment,
    size_t output_increment,
    const struct xnn_f16_scaleminmax_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

typedef void (*xnn_f32_avgpool_minmax_unipass_ukernel_fn)(
    size_t output_pixels,
    size_t kernel_elements,
    size_t channels,
    const float** input,
    size_t input_offset,
    const float* zero,
    float* output,
    size_t input_increment,
    size_t output_increment,
    const struct xnn_f32_scaleminmax_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

typedef void (*xnn_qu8_avgpool_minmax_unipass_ukernel_fn)(
    size_t output_pixels,
    size_t kernel_elements,
    size_t channels,
    const uint8_t** input,
    size_t input_offset,
    const uint8_t* zero,
    uint8_t* output,
    size_t input_increment,
    size_t output_increment,
    const struct xnn_qu8_avgpool_minmax_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

// AVGPOOL: AVeraGe POOLing multi-pass

typedef void (*xnn_avgpool_multipass_ukernel_fn)(
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

typedef void (*xnn_f16_avgpool_minmax_multipass_ukernel_fn)(
    size_t output_pixels,
    size_t kernel_elements,
    size_t channels,
    const xnn_float16** input,
    size_t input_offset,
    const xnn_float16* zero,
    xnn_float16* buffer,
    xnn_float16* output,
    size_t input_increment,
    size_t output_increment,
    const struct xnn_f16_scaleminmax_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

typedef void (*xnn_f32_avgpool_minmax_multipass_ukernel_fn)(
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
    const struct xnn_f32_scaleminmax_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

typedef void (*xnn_qu8_avgpool_minmax_multipass_ukernel_fn)(
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
    const struct xnn_qu8_avgpool_minmax_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

// PAVGPOOL: Pixelwise AVeraGe POOLing single-pass

typedef void (*xnn_pavgpool_unipass_ukernel_fn)(
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

typedef void (*xnn_f16_pavgpool_minmax_unipass_ukernel_fn)(
    size_t output_pixels,
    size_t kernel_elements,
    size_t channels,
    const xnn_float16** input,
    size_t input_offset,
    const xnn_float16* zero,
    const xnn_float16* multiplier,
    xnn_float16* output,
    size_t input_increment,
    size_t output_increment,
    const struct xnn_f16_scaleminmax_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

typedef void (*xnn_f32_pavgpool_minmax_unipass_ukernel_fn)(
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
    const union xnn_f32_minmax_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

// PAVGPOOL: Pixelwise AVeraGe POOLing multi-pass

typedef void (*xnn_pavgpool_multipass_ukernel_fn)(
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

typedef void (*xnn_f16_pavgpool_minmax_multipass_ukernel_fn)(
    size_t output_pixels,
    size_t kernel_elements,
    size_t channels,
    const xnn_float16** input,
    size_t input_offset,
    const xnn_float16* zero,
    const xnn_float16* multiplier,
    xnn_float16* buffer,
    xnn_float16* output,
    size_t input_increment,
    size_t output_increment,
    const struct xnn_f16_scaleminmax_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

typedef void (*xnn_f32_pavgpool_minmax_multipass_ukernel_fn)(
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
    const union xnn_f32_minmax_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

// MAXPOOL: MAX POOLing

typedef void (*xnn_maxpool_ukernel_fn)(
    size_t output_pixels,
    size_t kernel_elements,
    size_t channels,
    const void** input,
    size_t input_offset,
    void* output,
    size_t input_increment,
    size_t output_increment,
    const void* params);

typedef void (*xnn_f16_maxpool_ukernel_fn)(
    size_t output_pixels,
    size_t kernel_elements,
    size_t channels,
    const xnn_float16** input,
    size_t input_offset,
    xnn_float16* output,
    size_t input_increment,
    size_t output_increment,
    const union xnn_f16_minmax_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

typedef void (*xnn_f32_maxpool_ukernel_fn)(
    size_t output_pixels,
    size_t kernel_elements,
    size_t channels,
    const float** input,
    size_t input_offset,
    float* output,
    size_t input_increment,
    size_t output_increment,
    const union xnn_f32_minmax_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

typedef void (*xnn_s8_maxpool_ukernel_fn)(
    size_t output_pixels,
    size_t kernel_elements,
    size_t channels,
    const int8_t** input,
    size_t input_offset,
    int8_t* output,
    size_t input_increment,
    size_t output_increment,
    const struct xnn_s8_minmax_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

typedef void (*xnn_u8_maxpool_ukernel_fn)(
    size_t output_pixels,
    size_t kernel_elements,
    size_t channels,
    const uint8_t** input,
    size_t input_offset,
    uint8_t* output,
    size_t input_increment,
    size_t output_increment,
    const struct xnn_u8_minmax_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

// ARGMAXPOOL: ARG MAX POOLing single-pass

typedef void (*xnn_argmaxpool_unipass_ukernel_fn)(
    size_t output_pixels,
    size_t kernel_elements,
    size_t channels,
    const void** input,
    size_t input_offset,
    void* output,
    uint32_t* index,
    size_t input_increment,
    size_t output_increment);

typedef void (*xnn_f32_argmaxpool_unipass_ukernel_fn)(
    size_t output_pixels,
    size_t kernel_elements,
    size_t channels,
    const float** input,
    size_t input_offset,
    float* output,
    uint32_t* index,
    size_t input_increment,
    size_t output_increment);

// ARGMAXPOOL: ARG MAX POOLing multi-pass

typedef void (*xnn_argmaxpool_multipass_ukernel_fn)(
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
    size_t output_increment);

typedef void (*xnn_f32_argmaxpool_multipass_ukernel_fn)(
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
    size_t output_increment);

// UNPOOL: UNPOOLing

typedef void (*xnn_unpool_ukernel_fn)(
    size_t p,
    size_t c,
    uint32_t f,
    const void* input,
    const uint32_t* index,
    void** output);

typedef void (*xnn_x32_unpool_ukernel_fn)(
    size_t p,
    size_t c,
    uint32_t f,
    const uint32_t* input,
    const uint32_t* index,
    uint32_t** output);

// TRANSPOSEC: TRANSPOSE Constant-size elements

typedef void (*xnn_x8_transposec_ukernel_fn)(
    const uint8_t* a,
    uint8_t* b,
    size_t input_stride,
    size_t output_stride,
    size_t block_width,
    size_t block_height);

typedef void (*xnn_x16_transposec_ukernel_fn)(
    const uint16_t* a,
    uint16_t* b,
    size_t input_stride,
    size_t output_stride,
    size_t block_width,
    size_t block_height);

typedef void (*xnn_x24_transposec_ukernel_fn)(
    const void* a,
    void* b,
    size_t input_stride,
    size_t output_stride,
    size_t block_width,
    size_t block_height);

typedef void (*xnn_x32_transposec_ukernel_fn)(
    const uint32_t* a,
    uint32_t* b,
    size_t input_stride,
    size_t output_stride,
    size_t block_width,
    size_t block_height);

typedef void (*xnn_x64_transposec_ukernel_fn)(
    const uint64_t* a,
    uint64_t* b,
    size_t input_stride,
    size_t output_stride,
    size_t block_width,
    size_t block_height);

typedef void (*xnn_transposec_ukernel_fn)(
    const void* input,
    void* output,
    size_t input_stride,
    size_t output_size,
    size_t block_width,
    size_t block_height);

// TRANSPOSEV: TRANSPOSE Variable-size elements

typedef void (*xnn_transposev_ukernel_fn)(
    const void* input,
    void* output,
    size_t input_row_stride,
    size_t output_row_stride,
    size_t input_element_stride,
    size_t output_element_stride,
    size_t element_size,
    size_t block_width,
    size_t block_height);

// PACKB: PACK B (bias) for GEMM matrix multiplication

typedef void (*xnn_packb_gemm_ukernel_fn)(
    size_t groups,
    size_t channels,
    const void* bias,
    void* packed_weights,
    size_t channel_tile_stride,
    size_t channel_subtile_stride,
    const void* params);

typedef void (*xnn_x32_packb_gemm_ukernel_fn)(
    size_t groups,
    size_t channels,
    const uint32_t* bias,
    uint32_t* packed_weights,
    size_t channel_tile_stride,
    size_t channel_subtile_stride,
    const struct xnn_x32_packb_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

// ZEROB: ZERO B (bias) for GEMM matrix multiplication

typedef void (*xnn_zerob_gemm_ukernel_fn)(
    size_t groups,
    size_t channels,
    void* packed_weights,
    size_t channel_tile_stride,
    size_t channel_subtile_stride,
    const void* params);

typedef void (*xnn_x32_zerob_gemm_ukernel_fn)(
    size_t groups,
    size_t channels,
    uint32_t* packed_weights,
    size_t channel_tile_stride,
    size_t channel_subtile_stride,
    const struct xnn_x32_packb_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

// PACKQ: PACK and Quantize (weights) the left-hand operator for GEMM matrix
// multiplication.

typedef void (*xnn_x8_packq_f32qp8_ukernel_fn)(
    size_t m,   // Number of rows to pack.
    size_t k,   // Number of columns/channels per row.
    size_t mr,  // Number of rows to interleave in the same output row.
    size_t kr,  // Number of columns/channels loaded per step in the matmul
                // microkernel.
    size_t sr,  // Number of `kr` splits.
    size_t m_idx_start,         // Starting index in `lhs_packed`.
    const float* XNN_RESTRICT lhs,  // Left-hand operator to pack.
    size_t lhs_stride,          // Stride in bytes between the rows of `lhs`.
    void* XNN_RESTRICT lhs_packed   // The quantized and packed output.
);

// PACKW: PACK W (weights) for GEMM matrix multiplication
// Weights in GOI layout: Groups, Output channels, Input channels.

typedef void (*xnn_packw_gemm_goi_ukernel_fn)(
    size_t g,
    size_t nc,
    size_t kc,
    size_t nr,
    size_t kr,
    size_t sr,
    const void* k,
    const void* b,
    const void* scale,
    void* packed_weights,
    size_t extra_bytes,
    const void* params);

// TODO - Consolidate packing w/ per_channel and blockwise quant
typedef void (*xnn_packw_gemm_goi_bl_ukernel_fn)(
    size_t g,
    size_t nc,
    size_t kc,
    size_t nr,
    size_t kr,
    size_t sr,
    size_t bl,
    const void* k,
    const void* b,
    const void* scale,
    void* packed_weights,
    size_t extra_bytes_bl,
    size_t extra_bytes_n,
    const void* params);

typedef void (*xnn_x8_packw_gemm_goi_ukernel_fn)(
    size_t g,
    size_t nc,
    size_t kc,
    size_t nr,
    size_t kr,
    size_t sr,
    const int8_t* k,
    const uint32_t* b,
    const void* scale,
    int8_t* packed_weights,
    size_t extra_bytes,
    const void* params);

typedef void (*xnn_x8_packw_gemm_gio_ukernel_fn)(
    size_t g,
    size_t nc,
    size_t kc,
    size_t nr,
    size_t kr,
    size_t sr,
    size_t k_stride,
    const int8_t* k,
    const uint32_t* b,
    const void* scale,
    int8_t* packed_weights,
    size_t extra_bytes,
    const void* params);

typedef void (*xnn_qs8_packw_gemm_goi_ukernel_fn)(
    size_t g,
    size_t nc,
    size_t kc,
    size_t nr,
    size_t kr,
    size_t sr,
    const int8_t* k,
    const int32_t* b,
    const void* scale,
    int8_t* packed_weights,
    size_t extra_bytes,
    const void* params);

typedef void (*xnn_qs8_packw_gemm_gio_ukernel_fn)(
    size_t g,
    size_t nc,
    size_t kc,
    size_t nr,
    size_t kr,
    size_t sr,
    size_t k_stride,
    const int8_t* k,
    const int32_t* b,
    const void* scale,
    int8_t* packed_weights,
    size_t extra_bytes,
    const void* params);

typedef void (*xnn_qs8_qc4w_packw_gemm_goi_ukernel_fn)(
    size_t g,
    size_t nc,
    size_t kc,
    size_t nr,
    size_t kr,
    size_t sr,
    const uint8_t* k,
    const int32_t* b,
    const float* scale,
    void* packed_weights,
    size_t extra_bytes,
    const struct xnn_qs8_qc4w_packing_params* params);

typedef void (*xnn_x16_packw_gemm_goi_ukernel_fn)(
    size_t g,
    size_t nc,
    size_t kc,
    size_t nr,
    size_t kr,
    size_t sr,
    const uint16_t* k,
    const uint16_t* b,
    const void* scale,
    uint16_t* packed_weights,
    size_t extra_bytes,
    const void* params);

typedef void (*xnn_x32_packw_gemm_goi_ukernel_fn)(
    size_t g,
    size_t nc,
    size_t kc,
    size_t nr,
    size_t kr,
    size_t sr,
    const uint32_t* k,
    const uint32_t* b,
    const void* scale,
    uint32_t* packed_weights,
    size_t extra_bytes,
    const void* params);

typedef void (*xnn_x32_packw_gemm_gio_ukernel_fn)(
    size_t g,
    size_t nc,
    size_t kc,
    size_t nr,
    size_t kr,
    size_t sr,
    size_t k_stride,
    const uint32_t* k,
    const uint32_t* b,
    const void* scale,
    uint32_t* packed_weights,
    size_t extra_bytes,
    const void* params);

// PACKW: PACK W (weights) for GEMM matrix multiplication
// Weights in GIO layout: Groups, Input channels, Output channels.
typedef void (*xnn_packw_gemm_gio_ukernel_fn)(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  // We tile packing by output channels, in GIO layout, the k (row) index needs to be able to skip by the actual number
  // of output channels, and not just the argument nc. E.g. if weights is 1x3x5, and nr is 2, we tile the packing by
  // output channels, 2 + 2 + 1, with 3 calls to this packing function. In the first call nc == nr == 2, but to address
  // the second row of k, we need to skip by 5 elements, not 2 (nc). So k_stride should be set to 5.
  size_t k_stride,
  const void* k,
  const void* b,
  const void* scale,
  void* packed_weights,
  size_t extra_bytes,
  const void* params);

// PACK: PACK for IGEMM matrix multiplication
// Weights in GOKI layout: Groups, Output channels, Kernel channels, Input channels.
typedef void (*xnn_pack_conv_goki_w_fn)(
  size_t g,
  size_t nc,
  size_t ks,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  const void* kernel,
  const void* bias,
  const void* scale,
  void* packed_weights,
  size_t extra_bytes,
  const void* params);

// PACK: PACK for IGEMM matrix multiplication
// Weights in KGO layout: Kernel channels, groups, Output channels.
typedef void (*xnn_pack_conv_kgo_w_fn)(
  size_t g,
  size_t nc,
  size_t ks,
  size_t nr,
  size_t kr,
  size_t sr,
  const void* kernel,
  const void* bias,
  const void* scale,
  void* packed_weights,
  size_t extra_bytes,
  const void* params);

// PACK: PACK for DECONV SubConv matrix multiplication
// Weights in GOKI layout: Groups, Output channels, Kernel channels, Input channels.
typedef void (*xnn_pack_deconv_goki_w_fn)(
  size_t g,
  size_t nc,
  size_t kh,
  size_t kw,
  size_t kc,
  size_t sh,
  size_t sw,
  size_t nr,
  size_t kr,
  size_t sr,
  const void* kernel,
  const void* bias,
  const void* scale,
  void* packed_weights,
  size_t extra_bytes,
  void* subconv_params,
  const void* params);

// PACKX: PACK X (input) tensor for pre-packed matrix multiplication
typedef void (*xnn_packx_ukernel_fn)(
    size_t m,
    size_t k,
    const void* x,
    size_t x_stride,
    void* y);

typedef void (*xnn_x32_packx_ukernel_fn)(
    size_t m,
    size_t k,
    const uint32_t* x,
    size_t x_stride,
    uint32_t* y);

// PACKLH: PACK LH (input) tensor according to the parameters from the gemm
// config.
typedef void (*xnn_x32_pack_lh_ukernel_fn)(
    size_t m, size_t k, size_t mr, size_t kr, size_t sr, size_t m_idx_start,
    const uint32_t* lhs, size_t lhs_stride, uint32_t* lhs_packed);

// PACKLH Size: Size of packed buffer required.
typedef size_t (*xnn_x32_pack_lh_size_fn)(size_t m, size_t k, size_t mr,
                                          size_t kr, size_t sr);

// FILL: FILL array with value

typedef void (*xnn_fill_ukernel_fn)(
    size_t rows,
    size_t channels,
    void* output,
    size_t output_stride,
    const uint32_t fill_pattern);

// PAD: PAD array with values (fill before, copy array, fill after)

typedef void (*xnn_pad_ukernel_fn)(
    size_t rows,
    size_t channels,
    size_t pre_padding,
    size_t post_padding,
    const void* input,
    size_t input_stride,
    void* output,
    size_t output_stride,
    const uint32_t fill_value);

// REDUCE: Reduce

typedef void (*xnn_reduce_ukernel_fn)(
    size_t batch,
    const void* input,
    void* output,
    const void* params);

typedef void (*xnn_f16_reduce_ukernel_fn)(
    size_t batch,
    const xnn_float16* input,
    xnn_float16* output,
    const struct xnn_f16_default_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

typedef void (*xnn_f32_reduce_ukernel_fn)(
    size_t batch,
    const float* input,
    float* output,
    const struct xnn_f32_default_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

typedef void (*xnn_u8_reduce_ukernel_fn)(
    size_t batch,
    const uint8_t* input,
    uint8_t* output,
    const void* params);

// RDSUM: Discontiguous Reduce-Sum

typedef void (*xnn_rdsum_ukernel_fn)(
    size_t rows,
    size_t channels,
    const void* input,
    size_t input_stride,
    const void* zero,
    void* output,
    const void* params);

typedef void (*xnn_f16_f32acc_rdsum_ukernel_fn)(
    size_t rows,
    size_t channels,
    const xnn_float16* input,
    size_t input_stride,
    const xnn_float16* zero,
    float* output,
    const struct xnn_f16_f32acc_scale_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

typedef void (*xnn_f32_rdsum_ukernel_fn)(
    size_t rows,
    size_t channels,
    const float* input,
    size_t input_stride,
    const float* zero,
    float* output,
    const struct xnn_f32_scale_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

typedef void (*xnn_qs8_rdsum_ukernel_fn)(
    size_t rows,
    size_t channels,
    const int8_t* input,
    size_t input_stride,
    const int8_t* zero,
    int32_t* output,
    const struct xnn_qs8_rsum_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

typedef void (*xnn_qu8_rdsum_ukernel_fn)(
    size_t rows,
    size_t channels,
    const uint8_t* input,
    size_t input_stride,
    const uint8_t* zero,
    uint32_t* output,
    const struct xnn_qs8_rsum_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);
// RSUM: Reduce-Sum

typedef void (*xnn_f16_rsum_ukernel_fn)(
    size_t batch,
    const xnn_float16* input,
    xnn_float16* output,
    const struct xnn_f16_scale_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

typedef void (*xnn_f16_f32acc_rsum_ukernel_fn)(
    size_t batch,
    const xnn_float16* input,
    float* output,
    const struct xnn_f16_f32acc_scale_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

typedef void (*xnn_f32_rsum_ukernel_fn)(
    size_t batch,
    const float* input,
    float* output,
    const struct xnn_f32_scale_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

typedef void (*xnn_qs8_rsum_ukernel_fn)(
    size_t batch,
    const int8_t* input,
    int32_t* output,
    const struct xnn_qs8_rsum_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

typedef void (*xnn_qu8_rsum_ukernel_fn)(
    size_t batch,
    const uint8_t* input,
    uint32_t* output,
    const struct xnn_qs8_rsum_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

// RMAX: Reduce-MAX

typedef void (*xnn_rmax_ukernel_fn)(
    size_t batch,
    const void* input,
    void* output,
    const void* params);

typedef void (*xnn_f16_rmax_ukernel_fn)(
    size_t batch,
    const xnn_float16* input,
    xnn_float16* output,
    const struct xnn_f16_default_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

typedef void (*xnn_f32_rmax_ukernel_fn)(
    size_t batch,
    const float* input,
    float* output,
    const struct xnn_f32_default_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

typedef void (*xnn_u8_rmax_ukernel_fn)(
    size_t batch,
    const uint8_t* input,
    uint8_t* output,
    const void* params);

// RADDSTOREEXPMINUSMAX: Reduce-ADD & STORE EXP(x_i MINUS MAX[x_i])

typedef void (*xnn_raddstoreexpminusmax_ukernel_fn)(
    size_t batch,
    const void* input,
    const void* max,
    void* output,
    void* sum,
    const void* params);

typedef void (*xnn_f16_raddstoreexpminusmax_ukernel_fn)(
    size_t batch,
    const xnn_float16* input,
    const xnn_float16* max,
    xnn_float16* output,
    xnn_float16* sum,
    const void* params);

typedef void (*xnn_f32_raddstoreexpminusmax_ukernel_fn)(
    size_t batch,
    const float* input,
    const float* max,
    float* output,
    float* sum,
    const void* params);

// VUNARY: Vector UNARY elementwise

typedef void (*xnn_vunary_ukernel_fn)(
    size_t batch,
    const void* input,
    void* output,
    const union xnn_unary_uparams* params);

// LUT: vector LookUp Table elementwise

typedef void (*xnn_x8_lut_ukernel_fn)(
    size_t batch,
    const uint8_t* input,
    uint8_t* output,
    const uint8_t* table);

// LUT32NORM: vector LookUp Table of 32-bit elements and NORMalize elementwise

typedef void (*xnn_u8_lut32norm_ukernel_fn)(
    size_t n,
    const uint8_t* x,
    const uint32_t* t,
    uint8_t* y);

// VBINARY: Vector BINARY elementwise

typedef void (*xnn_vbinary_ukernel_fn)(
    size_t batch,
    const void* input_x,
    const void* input_y,
    void* output,
    const void* params);

typedef void (*xnn_f16_vbinary_ukernel_fn)(
    size_t batch,
    const xnn_float16* input_x,
    const xnn_float16* input_y,
    xnn_float16* output,
    const struct xnn_f16_default_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

typedef void (*xnn_f32_vbinary_ukernel_fn)(
    size_t batch,
    const float* input_x,
    const float* input_y,
    float* output,
    const struct xnn_f32_default_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

// VBINARY: Vector BINARY elementwise with Min+Max activation

typedef void (*xnn_f16_vbinary_minmax_ukernel_fn)(
    size_t batch,
    const xnn_float16* input_x,
    const xnn_float16* input_y,
    xnn_float16* output,
    const union xnn_f16_minmax_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

typedef void (*xnn_f32_vbinary_minmax_ukernel_fn)(
    size_t batch,
    const float* input_x,
    const float* input_y,
    float* output,
    const union xnn_f32_minmax_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

// VADD: Vector ADD elementwise with Min+Max activation

typedef void (*xnn_qs8_vadd_minmax_ukernel_fn)(
    size_t batch,
    const int8_t* input_x,
    const int8_t* input_y,
    int8_t* output,
    const struct xnn_qs8_add_minmax_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

typedef void (*xnn_qu8_vadd_minmax_ukernel_fn)(
    size_t batch,
    const uint8_t* input_x,
    const uint8_t* input_y,
    uint8_t* output,
    const struct xnn_qu8_add_minmax_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

// VMUL: Vector MUL elementwise with Min+Max activation

typedef void (*xnn_qs8_vmul_minmax_ukernel_fn)(
    size_t batch,
    const int8_t* input_x,
    const int8_t* input_y,
    int8_t* output,
    const union xnn_qs8_mul_minmax_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

typedef void (*xnn_qu8_vmul_minmax_ukernel_fn)(
    size_t batch,
    const uint8_t* input_x,
    const uint8_t* input_y,
    uint8_t* output,
    const union xnn_qu8_mul_minmax_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);


/***************** Microkernel pointers for sparse inference *****************/

// SpMM: Sparse Matrix-Matrix multiplication

typedef void (*xnn_spmm_ukernel_fn)(
    size_t batch_size,
    size_t output_channels,
    const void* input,
    const void* weights,
    const int32_t* widx_dmap,
    const uint32_t* nidx_nnzmap,
    void* output,
    size_t output_stride,
    const void* params);

typedef void (*xnn_f16_spmm_minmax_ukernel_fn)(
    size_t batch_size,
    size_t output_channels,
    const xnn_float16* input,
    const xnn_float16* weights,
    const int32_t* widx_dmap,
    const uint32_t* nidx_nnzmap,
    xnn_float16* output,
    size_t output_stride,
    const union xnn_f16_minmax_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

typedef void (*xnn_f32_spmm_minmax_ukernel_fn)(
    size_t batch_size,
    size_t output_channels,
    const float* input,
    const float* weights,
    const int32_t* widx_dmap,
    const uint32_t* nidx_nnzmap,
    float* output,
    size_t output_stride,
    const union xnn_f32_minmax_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

// CONV-HWC2CHW: direct CONVolution from HWC-layout tensor to CHW-layout tensor

typedef void (*xnn_conv_hwc2chw_ukernel_fn)(
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

typedef void (*xnn_f16_conv_hwc2chw_ukernel_fn)(
    size_t input_height,
    size_t input_width,
    size_t output_y_start,
    size_t output_y_end,
    const xnn_float16* input,
    const xnn_float16* zero,
    const xnn_float16* weights,
    xnn_float16* output,
    size_t input_padding_top,
    size_t output_channels,
    size_t output_height_stride,
    size_t output_channel_stride,
    const union xnn_f16_minmax_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

typedef void (*xnn_f32_conv_hwc2chw_ukernel_fn)(
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
    const union xnn_f32_minmax_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

// DWCONV2D-CHW: direct 2D DepthWise CONVolution in CHW layout

typedef void (*xnn_dwconv2d_chw_ukernel_fn)(
    size_t input_height,
    size_t input_width,
    const void* input,
    const void* weights,
    const void* zero,
    void* output,
    uint32_t padding_top,
    const void* params);

typedef void (*xnn_f16_dwconv2d_chw_ukernel_fn)(
    size_t input_height,
    size_t input_width,
    const xnn_float16* input,
    const xnn_float16* weights,
    const xnn_float16* zero,
    xnn_float16* output,
    uint32_t padding_top,
    const union xnn_f16_minmax_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

typedef void (*xnn_f32_dwconv2d_chw_ukernel_fn)(
    size_t input_height,
    size_t input_width,
    const float* input,
    const float* weights,
    const float* zero,
    float* output,
    uint32_t padding_top,
    const union xnn_f32_minmax_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

// IBILINEAR-CHW: Indirect BILINEAR interpolation in CHW layout

typedef void (*xnn_ibilinear_chw_ukernel_fn)(
    size_t output_pixels,
    size_t channels,
    const void** input,
    size_t input_offset,
    const void* weights,
    void* output,
    size_t input_increment);

typedef void (*xnn_f16_ibilinear_chw_ukernel_fn)(
    size_t output_pixels,
    size_t channels,
    const xnn_float16** input,
    size_t input_offset,
    const xnn_float16* weights,
    xnn_float16* output,
    size_t input_increment);

typedef void (*xnn_f32_ibilinear_chw_ukernel_fn)(
    size_t output_pixels,
    size_t channels,
    const float** input,
    size_t input_offset,
    const float* weights,
    float* output,
    size_t input_increment);


/********************* Experimental microkernel pointers *********************/

// ZIPC: ZIP Constant number of arrays

typedef void (*xnn_zipc_ukernel_fn)(
    size_t n,
    const void* x,
    void* y);

typedef void (*xnn_x8_zipc_ukernel_fn)(
    size_t n,
    const uint8_t* x,
    uint8_t* y);

typedef void (*xnn_x32_zipc_ukernel_fn)(
    size_t n,
    const uint32_t* x,
    uint32_t* y);

// ZIPV: ZIP Variable number of arrays

typedef void (*xnn_zipv_ukernel_fn)(
    size_t n,
    size_t m,
    const void* x,
    void* y);

typedef void (*xnn_x8_zipv_ukernel_fn)(
    size_t n,
    size_t m,
    const uint8_t* x,
    uint8_t* y);

typedef void (*xnn_x32_zipv_ukernel_fn)(
    size_t n,
    size_t m,
    const uint32_t* x,
    uint32_t* y);

// RADDEXPMINUSMAX: Reduce-ADD EXP(x_i MINUS MAX[x_i])

typedef void (*xnn_f32_raddexpminusmax_ukernel_fn)(
    size_t batch,
    const float* input,
    float* sum,
    float max);

// VSCALEEXPMINUSMAX: Vector SCALE EXP(x_i MINUS MAX[x_i])

typedef void (*xnn_f32_vscaleexpminusmax_ukernel_fn)(
    size_t batch,
    const float* input,
    float* output,
    float max,
    float scale);

// RADDEXTEXP: Reduce-ADD EXTended ("mantissa" + "exponent") EXPonentials
typedef void (*xnn_f32_raddextexp_ukernel_fn)(
    size_t batch,
    const float* input,
    float* sum);

// VSCALEEXTEXP: Vector SCALE EXTended ("mantissa" + "exponent") EXPonentials
typedef void (*xnn_f32_vscaleextexp_ukernel_fn)(
    size_t batch,
    const float* input,
    float* output,
    float scale_mantissa,
    float scale_exponent);


/***************** Microkernel parameter initializer pointers ****************/

typedef size_t (*xnn_init_binary_params_fn)(
  union xnn_binary_uparams* uparams,
  const struct xnn_quantization_params* a_quantization,
  const struct xnn_quantization_params* b_quantization,
  const struct xnn_quantization_params* output_quantization);

typedef size_t (*xnn_init_unary_uparams_fn)(
  union xnn_unary_uparams* microparams,
  const union xnn_unary_params* op_params,
  const struct xnn_quantization_params* input_quantization,
  const struct xnn_quantization_params* output_quantization);

typedef size_t (*xnn_init_reduce_params_fn)(
  struct xnn_reduce_params params[XNN_MIN_ELEMENTS(1)],
  const struct xnn_quantization_params* input_quantization,
  const struct xnn_quantization_params* output_quantization);

typedef size_t (*xnn_update_reduce_params_fn)(
  struct xnn_reduce_params params[XNN_MIN_ELEMENTS(1)],
  float scale);

typedef size_t (*xnn_init_qs8_qc8w_conv_minmax_params_fn)(
  union xnn_qs8_qc8w_conv_minmax_params params[XNN_MIN_ELEMENTS(1)],
  int8_t output_zero_point,
  int8_t output_min,
  int8_t output_max);

typedef size_t (*xnn_init_qs8_conv_minmax_params_fn)(
  union xnn_qs8_conv_minmax_params params[XNN_MIN_ELEMENTS(1)],
  float scale,
  int8_t output_zero_point,
  int8_t output_min,
  int8_t output_max);

typedef size_t (*xnn_init_qu8_conv_minmax_params_fn)(
  union xnn_qu8_conv_minmax_params params[XNN_MIN_ELEMENTS(1)],
  uint8_t kernel_zero_point,
  float scale,
  uint8_t output_zero_point,
  uint8_t output_min,
  uint8_t output_max);

typedef size_t (*xnn_init_qs8_rsum_params_fn)(
  struct xnn_qs8_rsum_params params[XNN_MIN_ELEMENTS(1)]);

typedef size_t (*xnn_init_qu8_avgpool_minmax_params_fn)(
  struct xnn_qu8_avgpool_minmax_params params[XNN_MIN_ELEMENTS(1)],
  int32_t bias,
  float scale,
  uint8_t output_zero_point,
  uint8_t output_min,
  uint8_t output_max);

typedef void (*xnn_update_qu8_avgpool_minmax_params_fn)(
  struct xnn_qu8_avgpool_minmax_params params[XNN_MIN_ELEMENTS(1)],
  int32_t bias,
  float scale);

typedef size_t (*xnn_init_qs8_add_minmax_params_fn)(
  struct xnn_qs8_add_minmax_params params[XNN_MIN_ELEMENTS(1)],
  const struct xnn_quantization_params* a_quantization,
  const struct xnn_quantization_params* b_quantization,
  const struct xnn_quantization_params* output_quantization);

typedef size_t (*xnn_init_qu8_add_minmax_params_fn)(
  struct xnn_qu8_add_minmax_params params[XNN_MIN_ELEMENTS(1)],
  const struct xnn_quantization_params* a_quantization,
  const struct xnn_quantization_params* b_quantization,
  const struct xnn_quantization_params* output_quantization);

typedef size_t (*xnn_init_qs8_mul_minmax_params_fn)(
  union xnn_qs8_mul_minmax_params params[XNN_MIN_ELEMENTS(1)],
  const struct xnn_quantization_params* a_quantization,
  const struct xnn_quantization_params* b_quantization,
  const struct xnn_quantization_params* output_quantization);

typedef size_t (*xnn_init_qu8_mul_minmax_params_fn)(
  union xnn_qu8_mul_minmax_params params[XNN_MIN_ELEMENTS(1)],
  const struct xnn_quantization_params* a_quantization,
  const struct xnn_quantization_params* b_quantization,
  const struct xnn_quantization_params* output_quantization);

typedef size_t (*xnn_init_bf16_default_params_fn)(
  struct xnn_bf16_default_params params[XNN_MIN_ELEMENTS(1)]);

typedef size_t (*xnn_init_f16_default_params_fn)(
  struct xnn_f16_default_params params[XNN_MIN_ELEMENTS(1)]);

typedef size_t (*xnn_init_f32_default_params_fn)(
  struct xnn_f32_default_params params[XNN_MIN_ELEMENTS(1)]);

typedef size_t (*xnn_init_s32_default_params_fn)(
  struct xnn_s32_default_params params[XNN_MIN_ELEMENTS(1)]);

typedef size_t (*xnn_init_f16_expminus_params_fn)(
  struct xnn_f16_default_params params[XNN_MIN_ELEMENTS(1)]);

typedef size_t (*xnn_init_f32_expminus_params_fn)(
  struct xnn_f32_default_params params[XNN_MIN_ELEMENTS(1)]);

typedef size_t (*xnn_init_bf16_minmax_params_fn)(
  struct xnn_bf16_minmax_params params[XNN_MIN_ELEMENTS(1)],
  xnn_bfloat16 min,
  xnn_bfloat16 max);

typedef size_t (*xnn_init_f16_minmax_params_fn)(
  union xnn_f16_minmax_params params[XNN_MIN_ELEMENTS(1)],
  xnn_float16 min,
  xnn_float16 max);

typedef size_t (*xnn_init_f32_minmax_params_fn)(
  union xnn_f32_minmax_params params[XNN_MIN_ELEMENTS(1)],
  float min,
  float max);

typedef size_t (*xnn_init_f16_qc4w_minmax_params_fn)(
  struct xnn_f16_qc4w_minmax_params params[XNN_MIN_ELEMENTS(1)],
  xnn_float16 min,
  xnn_float16 max,
  uint8_t kernel_zero_point);

typedef size_t (*xnn_init_f16_qb4w_minmax_params_fn)(
  struct xnn_f16_qb4w_minmax_params params[XNN_MIN_ELEMENTS(1)],
  xnn_float16 min,
  xnn_float16 max,
  uint8_t kernel_zero_point,
  size_t blocksize);

typedef size_t (*xnn_init_f32_qc4w_minmax_params_fn)(
  struct xnn_f32_qc4w_minmax_params params[XNN_MIN_ELEMENTS(1)],
  float min,
  float max,
  uint8_t kernel_zero_point);

typedef size_t (*xnn_init_f32_qb4w_minmax_params_fn)(
  struct xnn_f32_qb4w_minmax_params params[XNN_MIN_ELEMENTS(1)],
  float min,
  float max,
  uint8_t kernel_zero_point,
  size_t blocksize);

typedef size_t (*xnn_init_s8_minmax_params_fn)(
  struct xnn_s8_minmax_params params[XNN_MIN_ELEMENTS(1)],
  int8_t min,
  int8_t max);

typedef size_t (*xnn_init_u8_minmax_params_fn)(
  struct xnn_u8_minmax_params params[XNN_MIN_ELEMENTS(1)],
  uint8_t min,
  uint8_t max);

typedef size_t (*xnn_init_f16_scale_params_fn)(
  struct xnn_f16_scale_params params[XNN_MIN_ELEMENTS(1)],
  xnn_float16 scale);

typedef size_t (*xnn_init_f16_f32acc_scale_params_fn)(
  struct xnn_f16_f32acc_scale_params params[XNN_MIN_ELEMENTS(1)],
  float scale);

typedef size_t (*xnn_init_f32_scale_params_fn)(
  struct xnn_f32_scale_params params[XNN_MIN_ELEMENTS(1)],
  float scale);

typedef size_t (*xnn_init_f16_scaleminmax_params_fn)(
  struct xnn_f16_scaleminmax_params params[XNN_MIN_ELEMENTS(1)],
  xnn_float16 scale,
  xnn_float16 min,
  xnn_float16 max);

typedef void (*xnn_update_f16_scaleminmax_params_fn)(
  struct xnn_f16_scaleminmax_params params[XNN_MIN_ELEMENTS(1)],
  xnn_float16 scale);

typedef size_t (*xnn_init_f32_scaleminmax_params_fn)(
  struct xnn_f32_scaleminmax_params params[XNN_MIN_ELEMENTS(1)],
  float scale,
  float min,
  float max);

typedef void (*xnn_update_f32_scaleminmax_params_fn)(
  struct xnn_f32_scaleminmax_params params[XNN_MIN_ELEMENTS(1)],
  float scale);

typedef void (*xnn_init_scale_params_fn)(
  size_t channels,
  size_t channels_tile,
  size_t channels_subtile,
  size_t stride,
  size_t substride,
  size_t stride_offset,
  const void* scale,
  void* packed_w);

typedef void (*xnn_init_qs8_qc8w_scale_params_fn)(
  size_t channels,
  size_t channels_tile,
  size_t channels_subtile,
  size_t stride,
  size_t substride,
  size_t stride_offset,
  const float scale[XNN_MIN_ELEMENTS(1)],
  void* packed_w);

struct xnn_gemm_config;

// Pack weights and biases for GEMM microkernels.
//
// Implementations call the correct packing function selected using flags and
// pack any extra data required using init_extra_data_fns. Accumulators are
// initialized with accumulator_init.
typedef void (*xnn_pack_weights_and_biases_fn)(
    uint32_t flags,                             //
    const struct xnn_gemm_config* gemm_config,  //
    size_t input_channels,                      //
    size_t output_channels,                     //
    size_t groups,                              //
    // We tile packing by output channels, in GIO layout, the k (row) index
    // needs to be able to skip by the actual number of output channels, and not
    // just the argument nc. E.g. if weights is 1x3x5, and nr is 2, we tile the
    // packing by output channels, 2 + 2 + 1, with 3 calls to this packing
    // function. In the first call nc == nr == 2, but to address the second row
    // of k, we need to skip by 5 elements, not 2 (nc). So k_stride should be
    // set to 5.
    size_t k_stride,                               //
    const void* accumulator_init,                  //
    const void* weights,                           //
    xnn_init_scale_params_fn init_extra_data0_fn,  //
    const void* extra_data0,                       //
    size_t extra_data0_element_size,               //
    xnn_init_scale_params_fn init_extra_data1_fn,  //
    const void* extra_data1,                       //
    size_t extra_data1_element_size,               //
    void* packed_weights_ptr,                      //
    const void* params);

// Computes the stride of the packing used by a corresponding
// `xnn_pack_weights_and_biases_fn`. The `k_stride` parameter is provided for
// our older packing functions, new wrappers should rely on `gemm_config` and
// `k` instead.
typedef size_t (*xnn_packed_stride_weights_and_biases_fn)(
    const struct xnn_gemm_config* gemm_config,  //
    size_t k,                                   //
    size_t k_stride,                            //
    size_t extra_bytes);

typedef void (*xnn_indirection_init_resize_bilinear2d_hwc_fn)(
  size_t output_y_start,
  size_t output_y_end,
  size_t input_pixel_stride,
  size_t input_height,
  size_t input_width,
  size_t output_height,
  size_t output_width,
  const void* input,
  const void** indirection_buffer,
  void* packed_weights,
  bool align_corners,
  bool tensorflow_legacy);

struct xnn_hmp_dqgemm_ukernel {
  xnn_dqgemm_ukernel_fn function[XNN_MAX_UARCH_TYPES];
};

struct xnn_hmp_dqgemm_bl_ukernel {
  xnn_dqgemm_bl_ukernel_fn function[XNN_MAX_UARCH_TYPES];
};

struct xnn_hmp_gemm_ukernel {
  xnn_gemm_ukernel_fn function[XNN_MAX_UARCH_TYPES];
};

struct xnn_hmp_dqigemm_ukernel {
  xnn_dqigemm_ukernel_fn function[XNN_MAX_UARCH_TYPES];
};

struct xnn_hmp_igemm_ukernel {
  xnn_igemm_ukernel_fn function[XNN_MAX_UARCH_TYPES];
};

struct xnn_hmp_qp8gemm_ukernel {
  xnn_qp8_f32_qc4w_gemm_minmax_ukernel_fn function[XNN_MAX_UARCH_TYPES];
};

struct xnn_hmp_qp8gemm_bl_ukernel {
  xnn_qp8_f32_qb4w_gemm_minmax_ukernel_fn function[XNN_MAX_UARCH_TYPES];
};

// Largest GEMM/IGEMM MR used in init.c is 16 (x86 AVX512AMX).
#if XNN_ARCH_ARM64 && XNN_ENABLE_KLEIDIAI
#define XNN_MAX_MR 32
#else
#define XNN_MAX_MR 16
#endif

struct gemm_fused_ukernels {
  union {
    struct xnn_hmp_gemm_ukernel gemm[XNN_MAX_MR];
    struct xnn_hmp_dqgemm_ukernel dqgemm[XNN_MAX_MR];
    struct xnn_hmp_qp8gemm_ukernel qp8gemm[XNN_MAX_MR];
    struct xnn_hmp_dqgemm_bl_ukernel dqgemm_bl[XNN_MAX_MR];
    struct xnn_hmp_qp8gemm_bl_ukernel qp8gemm_bl[XNN_MAX_MR];
  };
  union {
    struct xnn_hmp_igemm_ukernel igemm[XNN_MAX_MR];
    struct xnn_hmp_dqigemm_ukernel dqigemm[XNN_MAX_MR];
  };
};
