// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <stddef.h>
#include <stdint.h>

#include <xnnpack/common.h>
#include <xnnpack/microparams.h>


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
    const union xnn_f32_minmax_params* params);

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
    const union xnn_f32_default_params* params);

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
    const union xnn_f32_relu_params* params);

// GEneral Matrix Multiplication with post operations

typedef void (*xnn_f32_gemm_post_operation_ukernel_fn)(
    size_t mr,
    size_t nr,
    size_t k,
    const float* a,
    size_t a_stride,
    const float* w,
    float* c,
    size_t cm_stride,
    size_t cn_stride,
    const void* params);

// GEMM: GEneral Matrix Multiplication with Min+Max activation

typedef void (*xnn_bf16_gemm_minmax_ukernel_fn)(
    size_t mr,
    size_t nr,
    size_t k,
    const void* a,
    size_t a_stride,
    const void* w,
    void* c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_bf16_minmax_params* params);

typedef void (*xnn_f16_gemm_minmax_ukernel_fn)(
    size_t mr,
    size_t nr,
    size_t k,
    const void* a,
    size_t a_stride,
    const void* w,
    void* c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f16_minmax_params* params);

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
    const union xnn_f32_minmax_params* params);

typedef void (*xnn_qc8_gemm_minmax_ukernel_fn)(
    size_t mr,
    size_t nr,
    size_t k,
    const int8_t* a,
    size_t a_stride,
    const void* w,
    int8_t* c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_qc8_conv_minmax_params* params);

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
    const union xnn_qs8_conv_minmax_params* params);

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
    const union xnn_qu8_conv_minmax_params* params);

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
    const union xnn_f32_minmax_params* params);

// IGEMM: Indirect GEMM without activation

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
    const union xnn_f32_default_params* params);

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
    const union xnn_f32_relu_params* params);

// IGEMM: Indirect GEMM with Min+Max activation

typedef void (*xnn_f16_igemm_minmax_ukernel_fn)(
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
    const union xnn_f16_minmax_params* params);

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
    const union xnn_f32_minmax_params* params);

// IGEMM: Indirect GEMM with post operations

typedef void (*xnn_f32_igemm_post_operation_ukernel_fn)(
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
    const void* params);

typedef void (*xnn_qc8_igemm_minmax_ukernel_fn)(
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
    const union xnn_qc8_conv_minmax_params* params);

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
    const union xnn_qs8_conv_minmax_params* params);

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
    const union xnn_qu8_conv_minmax_params* params);

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
    const void* a,
    const void* w,
    void* c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f16_minmax_params* params);

typedef void (*xnn_f32_ppmm_minmax_ukernel_fn)(
    size_t mr,
    size_t nc,
    size_t kc,
    const float* a,
    const float* w,
    float* c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_minmax_params* params);

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
    const union xnn_f32_default_params* params);

// DWCONV: DepthWise CONVolution single-pass with Min+Max activation

typedef void (*xnn_f16_dwconv_minmax_unipass_ukernel_fn)(
    size_t channels,
    size_t output_width,
    const void** input,
    const void* weights,
    void* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const void* zero,
    const union xnn_f16_minmax_params* params);

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
    const union xnn_f32_minmax_params* params);

typedef void (*xnn_qc8_dwconv_minmax_unipass_ukernel_fn)(
    size_t channels,
    size_t output_width,
    const int8_t** input,
    const void* weights,
    int8_t* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const int8_t* zero,
    const union xnn_qc8_conv_minmax_params* params);

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
    const union xnn_qs8_conv_minmax_params* params);

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
    const union xnn_qu8_conv_minmax_params* params);

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
    const union xnn_f32_default_params* params);

// DWCONV: DepthWise CONVolution multi-pass with Min+Max activation

typedef void (*xnn_f32_dwconv_multipass_minmax_ukernel_fn)(
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
    const union xnn_f32_minmax_params* params);

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
    const void* input,
    size_t input_stride,
    const void* weights,
    void* output,
    size_t output_stride,
    const union xnn_f16_minmax_params* params);

typedef void (*xnn_f32_vmulcaddc_ukernel_fn)(
    size_t batch,
    size_t channels,
    const float* input,
    size_t input_stride,
    const float* weights,
    float* output,
    size_t output_stride,
    const union xnn_f32_minmax_params* params);

// PRELU: Parametric RELU

typedef void (*xnn_prelu_ukernel_fn)(
    size_t batch,
    size_t channels,
    const void* input,
    size_t input_stride,
    const void* weights,
    void* output,
    size_t output_stride);

typedef void (*xnn_f16_prelu_ukernel_fn)(
    size_t batch,
    size_t channels,
    const void* input,
    size_t input_stride,
    const void* weights,
    void* output,
    size_t output_stride);

typedef void (*xnn_f32_prelu_ukernel_fn)(
    size_t batch,
    size_t channels,
    const float* input,
    size_t input_stride,
    const float* weights,
    float* output,
    size_t output_stride);

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
    const void** input,
    size_t input_offset,
    const void* weights,
    void* output,
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

// GAVGPOOL: Global AVeraGe POOLing single-pass

typedef void (*xnn_gavgpool_unipass_ukernel_fn)(
    size_t rows,
    size_t channels,
    const void* input,
    size_t input_stride,
    const void* zero,
    void* output,
    const void* params);

typedef void (*xnn_f16_gavgpool_minmax_unipass_ukernel_fn)(
    size_t rows,
    size_t channels,
    const void* input,
    size_t input_stride,
    const void* zero,
    void* output,
    const union xnn_f16_scaleminmax_params* params);

typedef void (*xnn_f32_gavgpool_minmax_unipass_ukernel_fn)(
    size_t rows,
    size_t channels,
    const float* input,
    size_t input_stride,
    const float* zero,
    float* output,
    const union xnn_f32_scaleminmax_params* params);

typedef void (*xnn_qs8_gavgpool_minmax_unipass_ukernel_fn)(
    size_t rows,
    size_t channels,
    const int8_t* input,
    size_t input_stride,
    const int8_t* zero,
    int8_t* output,
    const union xnn_qs8_avgpool_minmax_params* params);

typedef void (*xnn_qu8_gavgpool_minmax_unipass_ukernel_fn)(
    size_t rows,
    size_t channels,
    const uint8_t* input,
    size_t input_stride,
    const uint8_t* zero,
    uint8_t* output,
    const union xnn_qu8_avgpool_minmax_params* params);

// GAVGPOOL: Global AVeraGe POOLing multi-pass

typedef void (*xnn_gavgpool_multipass_ukernel_fn)(
    size_t rows,
    size_t channels,
    const void* input,
    size_t input_stride,
    const void* zero,
    void* buffer,
    void* output,
    const void* params);

typedef void (*xnn_f16_gavgpool_minmax_multipass_ukernel_fn)(
    size_t rows,
    size_t channels,
    const void* input,
    size_t input_stride,
    const void* zero,
    void* buffer,
    void* output,
    const union xnn_f16_scaleminmax_params* params);

typedef void (*xnn_f32_gavgpool_minmax_multipass_ukernel_fn)(
    size_t rows,
    size_t channels,
    const float* input,
    size_t input_stride,
    const float* zero,
    float* buffer,
    float* output,
    const union xnn_f32_scaleminmax_params* params);

typedef void (*xnn_qs8_gavgpool_minmax_multipass_ukernel_fn)(
    size_t rows,
    size_t channels,
    const int8_t* input,
    size_t input_stride,
    const int8_t* zero,
    int32_t* buffer,
    int8_t* output,
    const union xnn_qs8_avgpool_minmax_params* params);

typedef void (*xnn_qu8_gavgpool_minmax_multipass_ukernel_fn)(
    size_t rows,
    size_t channels,
    const uint8_t* input,
    size_t input_stride,
    const uint8_t* zero,
    int32_t* buffer,
    uint8_t* output,
    const union xnn_qu8_avgpool_minmax_params* params);

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
    const void** input,
    size_t input_offset,
    const void* zero,
    void* output,
    size_t input_increment,
    size_t output_increment,
    const union xnn_f16_scaleminmax_params* params);

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
    const union xnn_f32_scaleminmax_params* params);

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
    const union xnn_qu8_avgpool_minmax_params* params);

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
    const void** input,
    size_t input_offset,
    const void* zero,
    void* buffer,
    void* output,
    size_t input_increment,
    size_t output_increment,
    const union xnn_f16_scaleminmax_params* params);

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
    const union xnn_f32_scaleminmax_params* params);

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
    const union xnn_qu8_avgpool_minmax_params* params);

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
    const void** input,
    size_t input_offset,
    const void* zero,
    const void* multiplier,
    void* output,
    size_t input_increment,
    size_t output_increment,
    const union xnn_f16_minmax_params* params);

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
    const union xnn_f32_minmax_params* params);

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
    const void** input,
    size_t input_offset,
    const void* zero,
    const void* multiplier,
    void* buffer,
    void* output,
    size_t input_increment,
    size_t output_increment,
    const union xnn_f16_minmax_params* params);

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
    const union xnn_f32_minmax_params* params);

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
    const void** input,
    size_t input_offset,
    void* output,
    size_t input_increment,
    size_t output_increment,
    const union xnn_f16_minmax_params* params);

typedef void (*xnn_f32_maxpool_ukernel_fn)(
    size_t output_pixels,
    size_t kernel_elements,
    size_t channels,
    const float** input,
    size_t input_offset,
    float* output,
    size_t input_increment,
    size_t output_increment,
    const union xnn_f32_minmax_params* params);

typedef void (*xnn_s8_maxpool_ukernel_fn)(
    size_t output_pixels,
    size_t kernel_elements,
    size_t channels,
    const int8_t** input,
    size_t input_offset,
    int8_t* output,
    size_t input_increment,
    size_t output_increment,
    const union xnn_s8_minmax_params* params);

typedef void (*xnn_u8_maxpool_ukernel_fn)(
    size_t output_pixels,
    size_t kernel_elements,
    size_t channels,
    const uint8_t** input,
    size_t input_offset,
    uint8_t* output,
    size_t input_increment,
    size_t output_increment,
    const union xnn_u8_minmax_params* params);

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
    size_t block_height,
    const union xnn_x8_transpose_params* params);

typedef void (*xnn_x16_transposec_ukernel_fn)(
    const uint16_t* a,
    uint16_t* b,
    size_t input_stride,
    size_t output_stride,
    size_t block_width,
    size_t block_height,
    const union xnn_x16_transpose_params* params);

typedef void (*xnn_x24_transposec_ukernel_fn)(
    const void* a,
    void* b,
    size_t input_stride,
    size_t output_stride,
    size_t block_width,
    size_t block_height,
    const union xnn_x24_transpose_params* params);

typedef void (*xnn_x32_transposec_ukernel_fn)(
    const uint32_t* a,
    uint32_t* b,
    size_t input_stride,
    size_t output_stride,
    size_t block_width,
    size_t block_height,
    const union xnn_x32_transpose_params* params);

typedef void (*xnn_x64_transposec_ukernel_fn)(
    const uint64_t* a,
    uint64_t* b,
    size_t input_stride,
    size_t output_stride,
    size_t block_width,
    size_t block_height,
    const union xnn_x64_transpose_params* params);

typedef void (*xnn_transposec_ukernel_fn)(
    const void* input,
    void* output,
    size_t input_stride,
    size_t output_size,
    size_t block_width,
    size_t block_height,
    const void* params);

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

// RMAX: Reduce-MAX

typedef void (*xnn_rmax_ukernel_fn)(
    size_t batch,
    const void* input,
    void* output);

typedef void (*xnn_f16_rmax_ukernel_fn)(
    size_t batch,
    const void* input,
    void* output);

typedef void (*xnn_f32_rmax_ukernel_fn)(
    size_t batch,
    const float* input,
    float* output);

typedef void (*xnn_u8_rmax_ukernel_fn)(
    size_t batch,
    const uint8_t* input,
    uint8_t* output);

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
    const void* input,
    const void* max,
    void* output,
    void* sum,
    const union xnn_f16_expminus_params* params);

typedef void (*xnn_f32_raddstoreexpminusmax_ukernel_fn)(
    size_t batch,
    const float* input,
    const float* max,
    float* output,
    float* sum,
    const union xnn_f32_expminus_params* params);

// VUNARY: Vector UNARY elementwise

typedef void (*xnn_vunary_ukernel_fn)(
    size_t batch,
    const void* input,
    void* output,
    const void* params);

// VABS: Vector ABSolute value elementwise

typedef void (*xnn_f16_vabs_ukernel_fn)(
    size_t batch,
    const void* input,
    void* output,
    const union xnn_f16_abs_params* params);

typedef void (*xnn_f32_vabs_ukernel_fn)(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_abs_params* params);

// VCLAMP: Vector CLAMP elementwise

typedef void (*xnn_f16_vclamp_ukernel_fn)(
    size_t batch,
    const void* input,
    void* output,
    const union xnn_f16_minmax_params* params);

typedef void (*xnn_f32_vclamp_ukernel_fn)(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_minmax_params* params);

typedef void (*xnn_s8_vclamp_ukernel_fn)(
    size_t batch,
    const int8_t* input,
    int8_t* output,
    const union xnn_s8_minmax_params* params);

typedef void (*xnn_u8_vclamp_ukernel_fn)(
    size_t batch,
    const uint8_t* input,
    uint8_t* output,
    const union xnn_u8_minmax_params* params);

// VCVT: Vector ConVerT elementwise

typedef void (*xnn_f16_f32_vcvt_ukernel_fn)(
    size_t batch,
    const void* input,
    float* output,
    const union xnn_f16_f32_cvt_params* params);

typedef void (*xnn_f32_f16_vcvt_ukernel_fn)(
    size_t batch,
    const float* input,
    void* output,
    const union xnn_f32_f16_cvt_params* params);

typedef void (*xnn_f32_qs8_vcvt_ukernel_fn)(
    size_t batch,
    const float* input,
    int8_t* output,
    const union xnn_f32_qs8_cvt_params* params);

typedef void (*xnn_f32_qu8_vcvt_ukernel_fn)(
    size_t batch,
    const float* input,
    uint8_t* output,
    const union xnn_f32_qu8_cvt_params* params);

typedef void (*xnn_qs8_vcvt_ukernel_fn)(
    size_t batch,
    const int8_t* input,
    int8_t* output,
    const union xnn_qs8_cvt_params* params);

typedef void (*xnn_qs8_f32_vcvt_ukernel_fn)(
    size_t batch,
    const int8_t* input,
    float* output,
    const union xnn_qs8_f32_cvt_params* params);

typedef void (*xnn_qu8_vcvt_ukernel_fn)(
    size_t batch,
    const uint8_t* input,
    uint8_t* output,
    const union xnn_qu8_cvt_params* params);

typedef void (*xnn_qu8_f32_vcvt_ukernel_fn)(
    size_t batch,
    const uint8_t* input,
    float* output,
    const union xnn_qu8_f32_cvt_params* params);

// VELU: Vector Exponential Linear Unit elementwise

typedef void (*xnn_f16_velu_ukernel_fn)(
    size_t batch,
    const void* input,
    void* output,
    const union xnn_f16_elu_params* params);

typedef void (*xnn_f32_velu_ukernel_fn)(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_elu_params* params);

// VHSWISH: Vector Hard SWISH elementwise

typedef void (*xnn_f16_vhswish_ukernel_fn)(
    size_t batch,
    const void* input,
    void* output,
    const union xnn_f16_hswish_params* params);

typedef void (*xnn_f32_vhswish_ukernel_fn)(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_hswish_params* params);

// VLRELU: Vector Leaky REctified Linear Unit elementwise

typedef void (*xnn_f16_vlrelu_ukernel_fn)(
    size_t batch,
    const void* input,
    void* output,
    const union xnn_f16_lrelu_params* params);

typedef void (*xnn_f32_vlrelu_ukernel_fn)(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_lrelu_params* params);

typedef void (*xnn_qs8_vlrelu_ukernel_fn)(
    size_t batch,
    const int8_t* input,
    int8_t* output,
    const union xnn_qs8_lrelu_params* params);

typedef void (*xnn_qu8_vlrelu_ukernel_fn)(
    size_t batch,
    const uint8_t* input,
    uint8_t* output,
    const union xnn_qu8_lrelu_params* params);

// VNEG: Vector NEGate elementwise

typedef void (*xnn_f16_vneg_ukernel_fn)(
    size_t batch,
    const void* input,
    void* output,
    const union xnn_f16_neg_params* params);

typedef void (*xnn_f32_vneg_ukernel_fn)(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_neg_params* params);

// VRELU: Vector REctified Linear Unit elementwise

typedef void (*xnn_f32_vrelu_ukernel_fn)(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_relu_params* params);

// VROUND: Vector ROUNDing elementwise

typedef void (*xnn_f16_vround_ukernel_fn)(
    size_t batch,
    const void* input,
    void* output,
    const union xnn_f16_rnd_params* params);

typedef void (*xnn_f32_vround_ukernel_fn)(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_rnd_params* params);

// VSIGMOID: Vector SIGMOID elementwise

typedef void (*xnn_f16_vsigmoid_ukernel_fn)(
    size_t batch,
    const void* input,
    void* output,
    const union xnn_f16_sigmoid_params* params);

typedef void (*xnn_f32_vsigmoid_ukernel_fn)(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_sigmoid_params* params);

// VSQR: Vector SQuaRe elementwise

typedef void (*xnn_f16_vsqr_ukernel_fn)(
    size_t batch,
    const void* input,
    void* output,
    const union xnn_f16_default_params* params);

typedef void (*xnn_f32_vsqr_ukernel_fn)(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_default_params* params);

// VSQRT: Vector SQuare RooT elementwise

typedef void (*xnn_f16_vsqrt_ukernel_fn)(
    size_t batch,
    const void* input,
    void* output,
    const union xnn_f16_sqrt_params* params);

typedef void (*xnn_f32_vsqrt_ukernel_fn)(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_sqrt_params* params);

// VSQRTSHIFT: Vector SQuare RooT and SHIFT elementwise

typedef void (*xnn_u64_u32_vsqrtshift_ukernel_fn)(
    size_t batch,
    const uint64_t* input,
    uint32_t* output,
    uint32_t shift);

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
    const void* input_x,
    const void* input_y,
    void* output,
    const union xnn_f16_default_params* params);

typedef void (*xnn_f32_vbinary_ukernel_fn)(
    size_t batch,
    const float* input_x,
    const float* input_y,
    float* output,
    const union xnn_f32_default_params* params);

// VBINARY: Vector BINARY elementwise with ReLU activation

typedef void (*xnn_f32_vbinary_relu_ukernel_fn)(
    size_t batch,
    const float* input_x,
    const float* input_y,
    float* output,
    const union xnn_f32_relu_params* params);

// VBINARY: Vector BINARY elementwise with Min+Max activation

typedef void (*xnn_f16_vbinary_minmax_ukernel_fn)(
    size_t batch,
    const void* input_x,
    const void* input_y,
    void* output,
    const union xnn_f16_minmax_params* params);

typedef void (*xnn_f32_vbinary_minmax_ukernel_fn)(
    size_t batch,
    const float* input_x,
    const float* input_y,
    float* output,
    const union xnn_f32_minmax_params* params);

// VADD: Vector ADD elementwise with Min+Max activation

typedef void (*xnn_qs8_vadd_minmax_ukernel_fn)(
    size_t batch,
    const int8_t* input_x,
    const int8_t* input_y,
    int8_t* output,
    const union xnn_qs8_add_minmax_params* params);

typedef void (*xnn_qu8_vadd_minmax_ukernel_fn)(
    size_t batch,
    const uint8_t* input_x,
    const uint8_t* input_y,
    uint8_t* output,
    const union xnn_qu8_add_minmax_params* params);

// VMUL: Vector MUL elementwise with Min+Max activation

typedef void (*xnn_qs8_vmul_minmax_ukernel_fn)(
    size_t batch,
    const int8_t* input_x,
    const int8_t* input_y,
    int8_t* output,
    const union xnn_qs8_mul_minmax_params* params);

typedef void (*xnn_qu8_vmul_minmax_ukernel_fn)(
    size_t batch,
    const uint8_t* input_x,
    const uint8_t* input_y,
    uint8_t* output,
    const union xnn_qu8_mul_minmax_params* params);


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
    const void* input,
    const void* weights,
    const int32_t* widx_dmap,
    const uint32_t* nidx_nnzmap,
    void* output,
    size_t output_stride,
    const union xnn_f16_minmax_params* params);

typedef void (*xnn_f32_spmm_minmax_ukernel_fn)(
    size_t batch_size,
    size_t output_channels,
    const float* input,
    const float* weights,
    const int32_t* widx_dmap,
    const uint32_t* nidx_nnzmap,
    float* output,
    size_t output_stride,
    const union xnn_f32_minmax_params* params);

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
    const void* input,
    const void* zero,
    const void* weights,
    void* output,
    size_t input_padding_top,
    size_t output_channels,
    size_t output_height_stride,
    size_t output_channel_stride,
    const union xnn_f16_minmax_params* params);

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
    const union xnn_f32_minmax_params* params);

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
    const void* input,
    const void* weights,
    const void* zero,
    void* output,
    uint32_t padding_top,
    const union xnn_f16_chw_params* params);

typedef void (*xnn_f32_dwconv2d_chw_ukernel_fn)(
    size_t input_height,
    size_t input_width,
    const float* input,
    const float* weights,
    const float* zero,
    float* output,
    uint32_t padding_top,
    const union xnn_f32_chw_params* params);

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
    const void** input,
    size_t input_offset,
    const void* weights,
    void* output,
    size_t input_increment);

typedef void (*xnn_f32_ibilinear_chw_ukernel_fn)(
    size_t output_pixels,
    size_t channels,
    const float** input,
    size_t input_offset,
    const float* weights,
    float* output,
    size_t input_increment);

// GAVGPOOL-CW: Global AVeraGe POOLing in CW layout.

typedef void (*xnn_gavgpool_cw_ukernel_fn)(
    size_t batch,
    size_t channels,
    const float* input,
    float* output,
    const void* params);

typedef void (*xnn_f16_gavgpool_cw_ukernel_fn)(
    size_t batch,
    size_t channels,
    const void* input,
    void* output,
    const union xnn_f16_gavgpool_params* params);

typedef void (*xnn_f32_gavgpool_cw_ukernel_fn)(
    size_t batch,
    size_t channels,
    const float* input,
    float* output,
    const union xnn_f32_gavgpool_params* params);


/********************* JIT microkernel generator pointers ********************/

typedef xnn_status_t (*xnn_jit_gemm_code_generator_fn)(
    struct xnn_code_buffer *code, size_t max_mr, size_t nc_mod_nr, size_t kc, const void *params);
typedef xnn_status_t (*xnn_jit_igemm_code_generator_fn)(
    struct xnn_code_buffer *code, size_t max_mr, size_t nc_mod_nr, size_t kc, size_t ks, const void *params);


/***************** Audio pre-processing microkernel pointers *****************/

typedef void (*xnn_s16_rmaxabs_ukernel_fn)(
    size_t batch_size,
    const int16_t* x,
    uint16_t* y);

typedef void (*xnn_s16_window_ukernel_fn)(
    size_t rows,
    size_t batch_size,
    const int16_t* input,
    const int16_t* weights,
    int16_t* output,
    uint32_t shift);

typedef void (*xnn_u32_filterbank_accumulate_ukernel_fn)(
    size_t rows,
    const uint32_t* input,
    const uint8_t* weight_widths,
    const uint16_t* weights,
    uint64_t* output);

typedef void (*xnn_u32_filterbank_subtract_ukernel_fn)(
    size_t batch_size,
    const uint32_t* input,
    uint32_t smoothing,
    uint32_t alternate_smoothing,
    uint32_t one_minus_smoothing,
    uint32_t alternate_one_minus_smoothing,
    uint32_t min_signal_remaining,
    uint32_t smoothing_bits,
    uint32_t spectral_subtraction_bits,
    uint32_t* noise_estimate,
    uint32_t* output);

typedef void (*xnn_i16_vlshift_ukernel_fn)(
    size_t batch,
    const uint16_t* input,
    uint16_t* output,
    uint32_t shift);

typedef void (*xnn_cs16_vsquareabs_ukernel_fn)(
    size_t batch_size,
    const int16_t* input,
    uint32_t* output);

typedef void (*xnn_u32_vlog_ukernel_fn)(
    size_t batch_size,
    const uint32_t* input,
    uint32_t input_lshift,
    uint32_t output_scale,
    uint16_t* output);

typedef void (*xnn_cs16_bfly4_ukernel_fn)(
    size_t batch,
    size_t samples,
    int16_t* data,
    const int16_t* twiddle,
    size_t stride);

typedef void (*xnn_cs16_fftr_ukernel_fn)(
    size_t samples,
    int16_t* data,
    const int16_t* twiddle);


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

typedef size_t (*xnn_init_f16_f32_cvt_params_fn)(
  union xnn_f16_f32_cvt_params params[XNN_MIN_ELEMENTS(1)]);

typedef size_t (*xnn_init_f32_f16_cvt_params_fn)(
  union xnn_f32_f16_cvt_params params[XNN_MIN_ELEMENTS(1)]);

typedef size_t (*xnn_init_f32_qs8_cvt_params_fn)(
  union xnn_f32_qs8_cvt_params params[XNN_MIN_ELEMENTS(1)],
  float scale,
  int8_t output_zero_point,
  int8_t output_min,
  int8_t output_max);

typedef size_t (*xnn_init_f32_qu8_cvt_params_fn)(
  union xnn_f32_qu8_cvt_params params[XNN_MIN_ELEMENTS(1)],
  float scale,
  uint8_t output_zero_point,
  uint8_t output_min,
  uint8_t output_max);

typedef size_t (*xnn_init_qs8_cvt_params_fn)(
  union xnn_qs8_cvt_params params[XNN_MIN_ELEMENTS(1)],
  float input_output_scale,
  int8_t input_zero_point,
  int8_t output_zero_point);

typedef size_t (*xnn_init_qs8_f32_cvt_params_fn)(
  union xnn_qs8_f32_cvt_params params[XNN_MIN_ELEMENTS(1)],
  float scale,
  int8_t zero_point);

typedef size_t (*xnn_init_qu8_cvt_params_fn)(
  union xnn_qu8_cvt_params params[XNN_MIN_ELEMENTS(1)],
  float input_output_scale,
  uint8_t input_zero_point,
  uint8_t output_zero_point);

typedef size_t (*xnn_init_qu8_f32_cvt_params_fn)(
  union xnn_qu8_f32_cvt_params params[XNN_MIN_ELEMENTS(1)],
  float scale,
  uint8_t zero_point);

typedef size_t (*xnn_init_qc8_conv_minmax_params_fn)(
  union xnn_qc8_conv_minmax_params params[XNN_MIN_ELEMENTS(1)],
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

typedef size_t (*xnn_init_qs8_avgpool_minmax_params_fn)(
  union xnn_qs8_avgpool_minmax_params params[XNN_MIN_ELEMENTS(1)],
  int32_t bias,
  float scale,
  int8_t output_zero_point,
  int8_t output_min,
  int8_t output_max);

typedef size_t (*xnn_init_qu8_avgpool_minmax_params_fn)(
  union xnn_qu8_avgpool_minmax_params params[XNN_MIN_ELEMENTS(1)],
  int32_t bias,
  float scale,
  uint8_t output_zero_point,
  uint8_t output_min,
  uint8_t output_max);

typedef void (*xnn_update_qs8_avgpool_minmax_params_fn)(
  union xnn_qs8_avgpool_minmax_params params[XNN_MIN_ELEMENTS(1)],
  int32_t bias,
  float scale);

typedef void (*xnn_update_qu8_avgpool_minmax_params_fn)(
  union xnn_qu8_avgpool_minmax_params params[XNN_MIN_ELEMENTS(1)],
  int32_t bias,
  float scale);

typedef void (*xnn_update_f16_gavgpool_neonfp16arith_params_fn)(
  union xnn_f16_gavgpool_params params[XNN_MIN_ELEMENTS(1)],
  uint16_t multiplier,
  uint32_t width);

typedef size_t (*xnn_init_qs8_add_minmax_params_fn)(
  union xnn_qs8_add_minmax_params params[XNN_MIN_ELEMENTS(1)],
  int8_t input_x_zero_point,
  int8_t input_y_zero_point,
  int8_t output_zero_point,
  float input_x_output_scale,
  float input_y_output_scale,
  int8_t output_min,
  int8_t output_max);

typedef size_t (*xnn_init_qu8_add_minmax_params_fn)(
  union xnn_qu8_add_minmax_params params[XNN_MIN_ELEMENTS(1)],
  uint8_t input_x_zero_point,
  uint8_t input_y_zero_point,
  uint8_t output_zero_point,
  float input_x_output_scale,
  float input_y_output_scale,
  uint8_t output_min,
  uint8_t output_max);

typedef size_t (*xnn_init_qs8_mul_minmax_params_fn)(
  union xnn_qs8_mul_minmax_params params[XNN_MIN_ELEMENTS(1)],
  int8_t input_x_zero_point,
  int8_t input_y_zero_point,
  int8_t output_zero_point,
  float product_output_scale,
  int8_t output_min,
  int8_t output_max);

typedef size_t (*xnn_init_qu8_mul_minmax_params_fn)(
  union xnn_qu8_mul_minmax_params params[XNN_MIN_ELEMENTS(1)],
  uint8_t input_x_zero_point,
  uint8_t input_y_zero_point,
  uint8_t output_zero_point,
  float product_output_scale,
  uint8_t output_min,
  uint8_t output_max);

typedef size_t (*xnn_init_f16_abs_params_fn)(
  union xnn_f16_abs_params params[XNN_MIN_ELEMENTS(1)]);

typedef size_t (*xnn_init_f32_abs_params_fn)(
  union xnn_f32_abs_params params[XNN_MIN_ELEMENTS(1)]);

typedef size_t (*xnn_init_f16_default_params_fn)(
  union xnn_f16_default_params params[XNN_MIN_ELEMENTS(1)]);

typedef size_t (*xnn_init_f32_default_params_fn)(
  union xnn_f32_default_params params[XNN_MIN_ELEMENTS(1)]);

typedef size_t (*xnn_init_f16_expminus_params_fn)(
  union xnn_f16_expminus_params params[XNN_MIN_ELEMENTS(1)]);

typedef size_t (*xnn_init_f32_expminus_params_fn)(
  union xnn_f32_expminus_params params[XNN_MIN_ELEMENTS(1)]);

typedef size_t (*xnn_init_f16_elu_params_fn)(
  union xnn_f16_elu_params params[XNN_MIN_ELEMENTS(1)],
  uint16_t prescale,
  uint16_t alpha,
  uint16_t beta);

typedef size_t (*xnn_init_f32_elu_params_fn)(
  union xnn_f32_elu_params params[XNN_MIN_ELEMENTS(1)],
  float prescale,
  float alpha,
  float beta);

typedef size_t (*xnn_init_f16_hswish_params_fn)(
  union xnn_f16_hswish_params params[XNN_MIN_ELEMENTS(1)]);

typedef size_t (*xnn_init_f32_hswish_params_fn)(
  union xnn_f32_hswish_params params[XNN_MIN_ELEMENTS(1)]);

typedef size_t (*xnn_init_f16_lrelu_params_fn)(
  union xnn_f16_lrelu_params params[XNN_MIN_ELEMENTS(1)],
  uint16_t slope);

typedef size_t (*xnn_init_f32_lrelu_params_fn)(
  union xnn_f32_lrelu_params params[XNN_MIN_ELEMENTS(1)],
  float slope);

typedef size_t (*xnn_init_qs8_lrelu_params_fn)(
  union xnn_qs8_lrelu_params params[XNN_MIN_ELEMENTS(1)],
  float positive_slope,
  float negative_slope,
  int8_t input_zero_point,
  int8_t output_zero_point);

typedef size_t (*xnn_init_qu8_lrelu_params_fn)(
  union xnn_qu8_lrelu_params params[XNN_MIN_ELEMENTS(1)],
  float positive_slope,
  float negative_slope,
  uint8_t input_zero_point,
  uint8_t output_zero_point);

typedef size_t (*xnn_init_bf16_minmax_params_fn)(
  union xnn_bf16_minmax_params params[XNN_MIN_ELEMENTS(1)],
  uint16_t min,
  uint16_t max);

typedef size_t (*xnn_init_f16_minmax_params_fn)(
  union xnn_f16_minmax_params params[XNN_MIN_ELEMENTS(1)],
  uint16_t min,
  uint16_t max);

typedef size_t (*xnn_init_f32_minmax_params_fn)(
  union xnn_f32_minmax_params params[XNN_MIN_ELEMENTS(1)],
  float min,
  float max);

typedef size_t (*xnn_init_s8_minmax_params_fn)(
  union xnn_s8_minmax_params params[XNN_MIN_ELEMENTS(1)],
  int8_t min,
  int8_t max);

typedef size_t (*xnn_init_u8_minmax_params_fn)(
  union xnn_u8_minmax_params params[XNN_MIN_ELEMENTS(1)],
  uint8_t min,
  uint8_t max);

typedef size_t (*xnn_init_f16_neg_params_fn)(
  union xnn_f16_neg_params params[XNN_MIN_ELEMENTS(1)]);

typedef size_t (*xnn_init_f32_neg_params_fn)(
  union xnn_f32_neg_params params[XNN_MIN_ELEMENTS(1)]);

typedef size_t (*xnn_init_f16_rnd_params_fn)(
  union xnn_f16_rnd_params params[XNN_MIN_ELEMENTS(1)]);

typedef size_t (*xnn_init_f32_rnd_params_fn)(
  union xnn_f32_rnd_params params[XNN_MIN_ELEMENTS(1)]);

typedef size_t (*xnn_init_f16_scaleminmax_params_fn)(
  union xnn_f16_scaleminmax_params params[XNN_MIN_ELEMENTS(1)],
  uint16_t scale,
  uint16_t min,
  uint16_t max);

typedef void (*xnn_update_f16_scaleminmax_params_fn)(
  union xnn_f16_scaleminmax_params params[XNN_MIN_ELEMENTS(1)],
  uint16_t scale);

typedef size_t (*xnn_init_f32_scaleminmax_params_fn)(
  union xnn_f32_scaleminmax_params params[XNN_MIN_ELEMENTS(1)],
  float scale,
  float min,
  float max);

typedef void (*xnn_update_f32_scaleminmax_params_fn)(
  union xnn_f32_scaleminmax_params params[XNN_MIN_ELEMENTS(1)],
  float scale);

typedef size_t (*xnn_init_f16_sigmoid_params_fn)(
  union xnn_f16_sigmoid_params params[XNN_MIN_ELEMENTS(1)]);

typedef size_t (*xnn_init_f32_sigmoid_params_fn)(
  union xnn_f32_sigmoid_params params[XNN_MIN_ELEMENTS(1)]);

typedef size_t (*xnn_init_f16_sqrt_params_fn)(
  union xnn_f16_sqrt_params params[XNN_MIN_ELEMENTS(1)]);

typedef size_t (*xnn_init_f32_sqrt_params_fn)(
  union xnn_f32_sqrt_params params[XNN_MIN_ELEMENTS(1)]);

typedef void (*xnn_init_qc8_scale_params_fn)(
  size_t channels,
  size_t channels_tile,
  size_t stride,
  const float scale[XNN_MIN_ELEMENTS(1)],
  void* packed_w);

typedef size_t (*xnn_init_f16_gavgpool_neonfp16arith_params_fn)(
  union xnn_f16_gavgpool_params params[XNN_MIN_ELEMENTS(1)],
  uint16_t multiplier,
  uint16_t output_min,
  uint16_t output_max,
  uint32_t width);

typedef size_t (*xnn_init_f32_gavgpool_params_fn)(
  union xnn_f32_gavgpool_params params[XNN_MIN_ELEMENTS(1)],
  float multiplier,
  float output_min,
  float output_max,
  uint32_t width);

typedef size_t (*xnn_init_f32_chw_params_fn)(
  union xnn_f32_chw_params params[XNN_MIN_ELEMENTS(1)],
  uint32_t width,
  float output_min,
  float output_max);

typedef size_t (*xnn_init_f16_chw_params_fn)(
  union xnn_f16_chw_params params[XNN_MIN_ELEMENTS(1)],
  uint32_t width,
  uint16_t output_min,
  uint16_t output_max);

typedef void (*xnn_update_chw_params_fn)(
  void* params,
  uint32_t width);

typedef void (*xnn_update_f32_chw_params_fn)(
  union xnn_f32_chw_params params[XNN_MIN_ELEMENTS(1)],
  uint32_t width);

typedef void (*xnn_update_f16_chw_params_fn)(
  union xnn_f16_chw_params params[XNN_MIN_ELEMENTS(1)],
  uint32_t width);

typedef size_t (*xnn_init_x8_transpose_params_fn)(
  union xnn_x8_transpose_params params[XNN_MIN_ELEMENTS(1)]
);

typedef size_t (*xnn_init_x16_transpose_params_fn)(
  union xnn_x16_transpose_params params[XNN_MIN_ELEMENTS(1)]
);

typedef size_t (*xnn_init_x24_transpose_params_fn)(
  union xnn_x24_transpose_params params[XNN_MIN_ELEMENTS(1)]
);

typedef size_t (*xnn_init_x32_transpose_params_fn)(
  union xnn_x32_transpose_params params[XNN_MIN_ELEMENTS(1)]
);

typedef size_t (*xnn_init_x64_transpose_params_fn)(
  union xnn_x64_transpose_params params[XNN_MIN_ELEMENTS(1)]
);
