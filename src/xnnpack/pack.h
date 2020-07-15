// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <stdint.h>
#include <stddef.h>

#include <xnnpack/common.h>
#include <xnnpack/operator.h>


#ifdef __cplusplus
extern "C" {
#endif


struct xnn_qu8_packing_params {
  uint8_t input_zero_point;
  uint8_t kernel_zero_point;
};


typedef void (*xnn_pack_gemm_goi_w_function)(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  const void* k,
  const void* b,
  void* packed_w,
  const void* params);

XNN_INTERNAL void xnn_pack_f32_gemm_goi_w(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  const float* k,
  const float* b,
  float* packed_w,
  const void* params);

XNN_INTERNAL void xnn_pack_f16_gemm_goi_w(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  const uint16_t* k,
  const uint16_t* b,
  uint16_t* packed_w,
  const void* params);

XNN_INTERNAL void xnn_pack_qu8_gemm_goi_w(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  const uint8_t* k,
  const int32_t* b,
  void* packed_w,
  const struct xnn_qu8_packing_params* params);


typedef void (*xnn_pack_gemm_io_w_function)(
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  const void* k,
  const void* b,
  void* packed_w,
  const void* params);

XNN_INTERNAL void xnn_pack_f32_gemm_io_w(
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  const float* k,
  const float* b,
  float* packed_w,
  const void* params);

XNN_INTERNAL void xnn_pack_f16_gemm_io_w(
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  const uint16_t* k,
  const uint16_t* b,
  uint16_t* packed_w,
  const void* params);

XNN_INTERNAL void xnn_pack_qu8_gemm_io_w(
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  const uint8_t* k,
  const int32_t* b,
  void* packed_w,
  const struct xnn_qu8_packing_params* params);


typedef void (*xnn_pack_conv_goki_w_function)(
  size_t g,
  size_t nc,
  size_t ks,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  const void* k,
  const void* b,
  void* packed_w,
  const void* params);

XNN_INTERNAL void xnn_pack_f32_conv_goki_w(
  size_t g,
  size_t nc,
  size_t ks,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  const float* k,
  const float* b,
  float* packed_w,
  const void* params);

XNN_INTERNAL void xnn_pack_f16_conv_goki_w(
  size_t g,
  size_t nc,
  size_t ks,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  const uint16_t* k,
  const uint16_t* b,
  uint16_t* packed_w,
  const void* params);

XNN_INTERNAL void xnn_pack_qu8_conv_goki_w(
  size_t g,
  size_t nc,
  size_t ks,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  const uint8_t* k,
  const int32_t* b,
  void* packed_w,
  const struct xnn_qu8_packing_params* params);


typedef void (*xnn_pack_conv_kgo_w_function)(
  size_t g,
  size_t nc,
  size_t ks,
  size_t nr,
  size_t kr,
  const void* k,
  const void* b,
  void* packed_w,
  const void* params);

XNN_INTERNAL void xnn_pack_f32_conv_kgo_w(
  size_t g,
  size_t nc,
  size_t ks,
  size_t nr,
  size_t kr,
  const float* k,
  const float* b,
  float* packed_w,
  const void* params);

XNN_INTERNAL void xnn_pack_f16_conv_kgo_w(
  size_t g,
  size_t nc,
  size_t ks,
  size_t nr,
  size_t kr,
  const uint16_t* k,
  const uint16_t* b,
  uint16_t* packed_w,
  const void* params);

XNN_INTERNAL void xnn_pack_qu8_conv_kgo_w(
  size_t g,
  size_t nc,
  size_t ks,
  size_t nr,
  size_t kr,
  const uint8_t* k,
  const int32_t* b,
  void* packed_w,
  const struct xnn_qu8_packing_params* params);


typedef void (*xnn_pack_deconv_goki_w_function)(
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
  const void* k,
  const void* b,
  void* packed_w,
  struct subconvolution_params* subconv_params,
  const void* params);

XNN_INTERNAL void xnn_pack_f32_deconv_goki_w(
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
  const float* k,
  const float* b,
  float* packed_w,
  struct subconvolution_params* subconv_params,
  const void* params);

XNN_INTERNAL void xnn_pack_f16_deconv_goki_w(
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
  const uint16_t* k,
  const uint16_t* b,
  uint16_t* packed_w,
  struct subconvolution_params* subconv_params,
  const void* params);

XNN_INTERNAL void xnn_pack_qu8_deconv_goki_w(
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
  const uint8_t* k,
  const int32_t* b,
  void* packed_w,
  struct subconvolution_params* subconv_params,
  const struct xnn_qu8_packing_params* params);


typedef void (*xnn_pack_dwconv_ghw_w_function)(
  size_t h,
  size_t w,
  size_t c,
  size_t cr,
  const void* k,
  const void* b,
  void* packed_w,
  const void* params);

XNN_INTERNAL void xnn_pack_f32_dwconv_ghw_w(
  size_t h,
  size_t w,
  size_t c,
  size_t cr,
  const float* k,
  const float* b,
  float* packed_w,
  const void* params);

XNN_INTERNAL void xnn_pack_f16_dwconv_ghw_w(
  size_t h,
  size_t w,
  size_t c,
  size_t cr,
  const uint16_t* k,
  const uint16_t* b,
  uint16_t* packed_w,
  const void* params);

XNN_INTERNAL void xnn_pack_qu8_dwconv_ghw_w(
  size_t h,
  size_t w,
  size_t c,
  size_t cr,
  const uint8_t* k,
  const int32_t* b,
  void* packed_w,
  const struct xnn_qu8_packing_params* params);


typedef void (*xnn_pack_dwconv_hwg_w_function)(
  size_t h,
  size_t w,
  size_t c,
  size_t cr,
  const void* k,
  const void* b,
  void* packed_w,
  const void* params);

XNN_INTERNAL void xnn_pack_f32_dwconv_hwg_w(
  size_t h,
  size_t w,
  size_t c,
  size_t cr,
  const float* k,
  const float* b,
  float* packed_w,
  const void* params);

XNN_INTERNAL void xnn_pack_f16_dwconv_hwg_w(
  size_t h,
  size_t w,
  size_t c,
  size_t cr,
  const uint16_t* k,
  const uint16_t* b,
  uint16_t* packed_w,
  const void* params);

XNN_INTERNAL void xnn_pack_qu8_dwconv_hwg_w(
  size_t h,
  size_t w,
  size_t c,
  size_t cr,
  const uint8_t* k,
  const int32_t* b,
  void* packed_w,
  const struct xnn_qu8_packing_params* params);


XNN_INTERNAL void xnn_pack_f32_gemminc_goi_w(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  const float* k,
  float* packed_w,
  const void* params);

XNN_INTERNAL void xnn_pack_f16_gemminc_goi_w(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  const uint16_t* k,
  uint16_t* packed_w,
  const void* params);


XNN_INTERNAL void xnn_pack_f32_dconv_oki_w(
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kh,
  size_t kw,
  const float* k,
  const float* b,
  float* packed_w,
  const void* params);

XNN_INTERNAL void xnn_pack_f16_dconv_oki_w(
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kh,
  size_t kw,
  const uint16_t* k,
  const uint16_t* b,
  uint16_t* packed_w,
  const void* params);


XNN_INTERNAL void xnn_pack_f32_chw_dwconv_ghw_w(
  size_t kernel_size,
  size_t groups,
  const float* kernel,
  const float* bias,
  float* packed_weights,
  const void* params);

XNN_INTERNAL void xnn_pack_f16_chw_dwconv_ghw_w(
  size_t kernel_size,
  size_t groups,
  const uint16_t* kernel,
  const uint16_t* bias,
  uint16_t* packed_weights,
  const void* params);


XNN_INTERNAL void xnn_pack_f32_chw_dwconv_hwg_w(
  size_t kernel_size,
  size_t groups,
  const float* kernel,
  const float* bias,
  float* packed_weights,
  const void* params);


typedef void (*xnn_pack_vmulcaddc_w_function)(
  size_t c,
  size_t cr,
  const void* s,
  const void* b,
  void* packed_w,
  const void* params);

XNN_INTERNAL void xnn_pack_f32_vmulcaddc_w(
  size_t c,
  size_t cr,
  const float* s,
  const float* b,
  float* packed_w,
  const void* params);

XNN_INTERNAL void xnn_pack_f16_vmulcaddc_w(
  size_t c,
  size_t cr,
  const uint16_t* s,
  const uint16_t* b,
  uint16_t* packed_w,
  const void* params);

#ifdef __cplusplus
}  // extern "C"
#endif
