// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <stddef.h>
#include <stdint.h>

#include "xnnpack.h"
#include "xnnpack/common.h"
#include "xnnpack/math.h"
#include "xnnpack/microfnptr.h"
#include "xnnpack/microparams.h"

#ifdef __cplusplus
extern "C" {
#endif


struct xnn_qu8_packing_params {
  uint8_t input_zero_point;
  uint8_t kernel_zero_point;
};

struct xnn_qs8_packing_params {
  int8_t input_zero_point;
};

typedef void (*xnn_pack_f32_gemm_fn)(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  const float* kernel,
  const float* bias,
  const void* scale,
  float* packed_weights,
  size_t extra_bytes,
  const void* params);

XNN_INTERNAL void xnn_pack_f32_gemm_goi_w(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  const float* kernel,
  const float* bias,
  const void* scale,
  float* packed_weights,
  size_t extra_bytes,
  const void* params);

typedef void (*xnn_pack_f16_gemm_fn)(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  const uint16_t* kernel,
  const uint16_t* bias,
  const void* scale,
  uint16_t* packed_weights,
  size_t extra_bytes,
  const void* params);

XNN_INTERNAL void xnn_pack_f16_gemm_goi_w(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  const uint16_t* kernel,
  const uint16_t* bias,
  const void* scale,
  uint16_t* packed_weights,
  size_t extra_bytes,
  const void* params);

XNN_INTERNAL void xnn_pack_f32_to_f16_gemm_goi_w(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  const float* kernel,
  const float* bias,
  const void* scale,
  xnn_float16* packed_weights,
  size_t extra_bytes,
  const void* params);

typedef void (*xnn_pack_qu8_gemm_fn)(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  const uint8_t* k,
  const int32_t* b,
  const void* scale,
  void* packed_weights,
  size_t extra_bytes,
  const struct xnn_qu8_packing_params* params);

XNN_INTERNAL void xnn_pack_qu8_gemm_goi_w(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  const uint8_t* kernel,
  const int32_t* bias,
  const void* scale,
  void* packed_weights,
  size_t extra_bytes,
  const struct xnn_qu8_packing_params* params);

typedef void (*xnn_pack_qs8_gemm_fn) (
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  const int8_t* k,
  const int32_t* b,
  const float* scale,
  void* packed_weights,
  size_t extra_bytes,
  const struct xnn_qs8_packing_params* params);

XNN_INTERNAL void xnn_pack_qs8_gemm_goi_w(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  const int8_t* kernel,
  const int32_t* bias,
  const float* scale,
  void* packed_weights,
  size_t extra_bytes,
  const struct xnn_qs8_packing_params* params);

XNN_INTERNAL void xnn_pack_qs8_to_qu8_gemm_goi_w(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  const int8_t* k,
  const int32_t* b,
  const float* scale,
  void* packed_weights,
  size_t extra_bytes,
  const struct xnn_qs8_packing_params* params);

typedef void (*xnn_pack_qs8_qc4w_gemm_fn)(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  const uint8_t* kernel,
  const int32_t* bias,
  const float* scale,
  void* packed_weights,
  size_t extra_bytes,
  const struct xnn_qs8_qc4w_packing_params* params);

XNN_INTERNAL void xnn_pack_qs8_qc4w_gemm_goi_w(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  const uint8_t* kernel,
  const int32_t* bias,
  const float* scale,
  void* packed_weights,
  size_t extra_bytes,
  const struct xnn_qs8_qc4w_packing_params* params);

// 4 bit unsigned weights
XNN_INTERNAL void xnn_pack_qs8_qc4uw_gemm_goi_w(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  const uint8_t* kernel,
  const int32_t* bias,
  const float* scale,
  void* packed_weights,
  size_t extra_bytes,
  const struct xnn_qs8_qc4w_packing_params* params);

/*
 * Packing function for weights with int4 elements, per channel blockwise quantized
 */
typedef void (*xnn_pack_qs8_qb4w_gemm_fn)(
  size_t groups,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  size_t block_size, // number of K elements in a block
  const uint8_t* kernel,
  const float* bias,
  const xnn_bfloat16* scale,
  void* packed_weights,
  size_t extra_bytes_per_block,
  size_t extra_bytes_per_n,
  const struct xnn_qs8_qc4w_packing_params* params);

XNN_INTERNAL void xnn_pack_qs8_qb4w_gemm_goi_w(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  size_t bl,
  const uint8_t* kernel,
  const float* bias,
  const xnn_bfloat16* scale,
  void* packed_weights,
  size_t extra_bytes_bl,
  size_t extra_bytes_n,
  const struct xnn_qs8_qc4w_packing_params* params);

typedef void (*xnn_pack_f32_qc4w_gemm_fn)(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  const void* kernel,
  const float* bias,
  const float* scale,
  void* packed_weights,
  size_t extra_bytes,
  const void* params);

XNN_INTERNAL void xnn_pack_f32_qc4w_gemm_goi_w(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  const void* kernel,
  const float* bias,
  const float* scale,
  void* packed_weights,
  size_t extra_bytes,
  const void* params);

typedef void (*xnn_pack_f32_qs8w_gemm_fn)(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  const int8_t* kernel,
  const float* bias,
  const float* scale,
  void* packed_weights,
  size_t extra_bytes,
  const void* params);

XNN_INTERNAL void xnn_pack_f32_qs8w_gemm_goi_w(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  const int8_t* kernel,
  const float* bias,
  const float* scale,
  void* packed_weights,
  size_t extra_bytes,
  const void* params);

XNN_INTERNAL void xnn_pack_f32_gemm_gio_w(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  size_t k_stride,
  const float* kernel,
  const float* bias,
  const void* scale,
  float* packed_weights,
  size_t extra_bytes,
  const void* params);

XNN_INTERNAL void xnn_pack_f16_gemm_gio_w(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  size_t k_stride,
  const uint16_t* kernel,
  const uint16_t* bias,
  const void* scale,
  uint16_t* packed_weights,
  size_t extra_bytes,
  const void* params);

XNN_INTERNAL void xnn_pack_f32_to_f16_gemm_gio_w(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  size_t k_stride,
  const float* kernel,
  const float* bias,
  const void* scale,
  xnn_float16* packed_weights,
  size_t extra_bytes,
  const void* params);

XNN_INTERNAL void xnn_pack_qu8_gemm_gio_w(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  size_t k_stride,
  const uint8_t* kernel,
  const int32_t* bias,
  const void* scale,
  void* packed_weights,
  size_t extra_bytes,
  const struct xnn_qu8_packing_params* params);

XNN_INTERNAL void xnn_pack_qs8_gemm_gio_w(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  size_t k_stride,
  const int8_t* kernel,
  const int32_t* bias,
  const float* scale,
  void* packed_weights,
  size_t extra_bytes,
  const struct xnn_qs8_packing_params* params);

// Packs weights, kernel scales and biases for qs8-qc8w gemm microkernels.
XNN_INTERNAL void xnn_pack_qs8_weights_and_biases(
    uint32_t flags,                                //
    const struct xnn_gemm_config* gemm_config,     //
    size_t input_channels,                         //
    size_t output_channels,                        //
    size_t groups,                                 //
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

XNN_INTERNAL size_t xnn_packed_stride_qs8_weights_and_biases(
    const struct xnn_gemm_config* gemm_config,  //
    size_t k,                                   //
    size_t k_stride,                            //
    size_t extra_bytes);

// Caveat - misnamed qs4.  Should be qc4w
XNN_INTERNAL void xnn_pack_qs4_weights_and_biases(
    uint32_t flags,                                //
    const struct xnn_gemm_config* gemm_config,     //
    size_t input_channels,                         //
    size_t output_channels,                        //
    size_t groups,                                 //
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

XNN_INTERNAL size_t xnn_packed_stride_qs4_weights_and_biases(
    const struct xnn_gemm_config* gemm_config,  //
    size_t k,                                   //
    size_t k_stride,                            //
    size_t extra_bytes);

XNN_INTERNAL void xnn_pack_qu8_weights_and_biases(
    uint32_t flags,                                //
    const struct xnn_gemm_config* gemm_config,     //
    size_t input_channels,                         //
    size_t output_channels,                        //
    size_t groups,                                 //
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

XNN_INTERNAL size_t xnn_packed_stride_qu8_weights_and_biases(
    const struct xnn_gemm_config* gemm_config,  //
    size_t k,                                   //
    size_t k_stride,                            //
    size_t extra_bytes);

#if XNN_ENABLE_KLEIDIAI
XNN_INTERNAL void xnn_pack_kai_qs4_weights_and_biases(
    uint32_t flags,                                //
    const struct xnn_gemm_config* gemm_config,     //
    size_t input_channels,                         //
    size_t output_channels,                        //
    size_t groups,                                 //
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

XNN_INTERNAL size_t xnn_packed_stride_kai_qs4_weights_and_biases(
    const struct xnn_gemm_config* gemm_config,  //
    size_t k,                                   //
    size_t k_stride,                            //
    size_t extra_bytes);

XNN_INTERNAL void xnn_pack_kai_qs8_weights_and_biases(
    uint32_t flags,                                //
    const struct xnn_gemm_config* gemm_config,     //
    size_t input_channels,                         //
    size_t output_channels,                        //
    size_t groups,                                 //
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

XNN_INTERNAL size_t xnn_packed_stride_kai_qs8_weights_and_biases(
    const struct xnn_gemm_config* gemm_config,  //
    size_t k,                                   //
    size_t k_stride,                            //
    size_t extra_bytes);

size_t xnn_packed_stride_kai_f32_weights_and_biases(
    const struct xnn_gemm_config* gemm_config, size_t k, size_t unused_k_stride,
    size_t extra_bytes);

void xnn_pack_kai_f32_weights_and_biases(
    uint32_t flags, const struct xnn_gemm_config* gemm_config,
    size_t input_channels, size_t output_channels, size_t groups,
    size_t k_stride, const void* accumulator_init, const void* weights,
    xnn_init_scale_params_fn init_extra_data0_fn, const void* extra_data0,
    size_t extra_data0_element_size,
    xnn_init_scale_params_fn init_extra_data1_fn, const void* extra_data1,
    size_t extra_data1_element_size, void* packed_weights_ptr,
    const void* params);

XNN_INTERNAL void xnn_pack_kai_qb4_weights_and_biases(
    uint32_t flags,                                //
    const struct xnn_gemm_config* gemm_config,     //
    size_t input_channels,                         //
    size_t output_channels,                        //
    size_t groups,                                 //
    size_t block_size,                             //
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

XNN_INTERNAL size_t xnn_packed_stride_kai_qb4_weights_and_biases(
    const struct xnn_gemm_config* gemm_config,  //
    size_t k,                                   //
    size_t block_size,                          //
    size_t extra_bytes);
#endif  // XNN_ENABLE_KLEIDIAI

XNN_INTERNAL void xnn_pack_qs8_to_qu8_gemm_gio_w(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  size_t k_stride,
  const int8_t* kernel,
  const int32_t* bias,
  const float* scale,
  void* packed_weights,
  size_t extra_bytes,
  const struct xnn_qs8_packing_params* params);

XNN_INTERNAL void xnn_pack_qs8_qc4w_gemm_gio_w(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  size_t k_stride,
  const uint8_t* kernel,
  const int32_t* bias,
  const float* scale,
  void* packed_weights,
  size_t extra_bytes,
  const struct xnn_qs8_qc4w_packing_params* params);

XNN_INTERNAL void xnn_pack_qs8_qc4uw_gemm_gio_w(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  size_t k_stride,
  const uint8_t* kernel,
  const int32_t* bias,
  const float* scale,
  void* packed_weights,
  size_t extra_bytes,
  const struct xnn_qs8_qc4w_packing_params* params);

XNN_INTERNAL void xnn_pack_qs8_qb4w_gemm_gio_w(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  size_t k_stride,
  size_t bl,
  const uint8_t* kernel,
  const float* bias,
  const xnn_bfloat16* scale,
  void* packed_weights,
  size_t extra_bytes_bl,
  size_t extra_bytes_n,
  const struct xnn_qs8_qc4w_packing_params* params);

XNN_INTERNAL void xnn_pack_f32_qs8w_gemm_gio_w(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  size_t k_stride,
  const int8_t* kernel,
  const float* bias,
  const float* scale,
  void* packed_weights,
  size_t extra_bytes,
  const void* params);

typedef void (*xnn_pack_f32_igemm_fn)(
  size_t g,
  size_t nc,
  size_t ks,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  const float* kernel,
  const float* bias,
  const void* scale,
  float* packed_weights,
  size_t extra_bytes,
  const void* params);

XNN_INTERNAL void xnn_pack_f32_conv_goki_w(
  size_t g,
  size_t nc,
  size_t ks,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  const float* kernel,
  const float* bias,
  const void* scale,
  float* packed_weights,
  size_t extra_bytes,
  const void* params);

typedef void (*xnn_pack_f16_igemm_fn)(
  size_t g,
  size_t nc,
  size_t ks,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  const uint16_t* kernel,
  const uint16_t* bias,
  const void* scale,
  uint16_t* packed_weights,
  size_t extra_bytes,
  const void* params);

XNN_INTERNAL void xnn_pack_f16_conv_goki_w(
  size_t g,
  size_t nc,
  size_t ks,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  const uint16_t* kernel,
  const uint16_t* bias,
  const void* scale,
  uint16_t* packed_weights,
  size_t extra_bytes,
  const void* params);

XNN_INTERNAL void xnn_pack_f32_to_f16_conv_goki_w(
  size_t g,
  size_t nc,
  size_t ks,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  const float* kernel,
  const float* bias,
  const void* scale,
  xnn_float16* packed_weights,
  size_t extra_bytes,
  const void* params);

typedef void (*xnn_pack_qu8_igemm_fn)(
  size_t g,
  size_t nc,
  size_t ks,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  const uint8_t* k,
  const int32_t* b,
  const void* scale,
  void* packed_weights,
  size_t extra_bytes,
  const struct xnn_qu8_packing_params* params);

XNN_INTERNAL void xnn_pack_qu8_conv_goki_w(
  size_t g,
  size_t nc,
  size_t ks,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  const uint8_t* kernel,
  const int32_t* bias,
  const void* scale,
  void* packed_weights,
  size_t extra_bytes,
  const struct xnn_qu8_packing_params* params);

typedef void (*xnn_pack_qs8_igemm_fn) (
  size_t g,
  size_t nc,
  size_t ks,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  const int8_t* kernel,
  const int32_t* bias,
  const float* scale,
  void* packed_weights,
  size_t extra_bytes,
  const struct xnn_qs8_packing_params* params);

XNN_INTERNAL void xnn_pack_qs8_conv_goki_w(
  size_t g,
  size_t nc,
  size_t ks,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  const int8_t* kernel,
  const int32_t* bias,
  const float* scale,
  void* packed_weights,
  size_t extra_bytes,
  const struct xnn_qs8_packing_params* params);

XNN_INTERNAL void xnn_pack_qs8_to_qu8_conv_goki_w(
  size_t g,
  size_t nc,
  size_t ks,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  const int8_t* kernel,
  const int32_t* bias,
  const float* scale,
  void* packed_weights,
  size_t extra_bytes,
  const struct xnn_qs8_packing_params* params);

XNN_INTERNAL void xnn_pack_f32_conv_kgo_w(
  size_t g,
  size_t nc,
  size_t ks,
  size_t nr,
  size_t kr,
  size_t sr,
  const float* kernel,
  const float* bias,
  const void* scale,
  float* packed_weights,
  size_t extra_bytes,
  const void* params);

XNN_INTERNAL void xnn_pack_f16_conv_kgo_w(
  size_t g,
  size_t nc,
  size_t ks,
  size_t nr,
  size_t kr,
  size_t sr,
  const uint16_t* kernel,
  const uint16_t* bias,
  const void* scale,
  uint16_t* packed_weights,
  size_t extra_bytes,
  const void* params);

XNN_INTERNAL void xnn_pack_f32_to_f16_conv_kgo_w(
  size_t g,
  size_t nc,
  size_t ks,
  size_t nr,
  size_t kr,
  size_t sr,
  const float* kernel,
  const float* bias,
  const void* scale,
  xnn_float16* packed_weights,
  size_t extra_bytes,
  const void* params);

XNN_INTERNAL void xnn_pack_qu8_conv_kgo_w(
  size_t g,
  size_t nc,
  size_t ks,
  size_t nr,
  size_t kr,
  size_t sr,
  const uint8_t* kernel,
  const int32_t* bias,
  const void* scale,
  void* packed_weights,
  size_t extra_bytes,
  const struct xnn_qu8_packing_params* params);

XNN_INTERNAL void xnn_pack_qs8_conv_kgo_w(
  size_t g,
  size_t nc,
  size_t ks,
  size_t nr,
  size_t kr,
  size_t sr,
  const int8_t* kernel,
  const int32_t* bias,
  const float* scale,
  void* packed_weights,
  size_t extra_bytes,
  const struct xnn_qs8_packing_params* params);

XNN_INTERNAL void xnn_pack_qs8_to_qu8_conv_kgo_w(
  size_t g,
  size_t nc,
  size_t ks,
  size_t nr,
  size_t kr,
  size_t sr,
  const int8_t* kernel,
  const int32_t* bias,
  const float* scale,
  void* packed_weights,
  size_t extra_bytes,
  const struct xnn_qs8_packing_params* params);

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
  const float* kernel,
  const float* bias,
  const void* scale,
  float* packed_weights,
  size_t extra_bytes,
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
  const uint16_t* kernel,
  const uint16_t* bias,
  const void* scale,
  uint16_t* packed_weights,
  size_t extra_bytes,
  struct subconvolution_params* subconv_params,
  const void* params);

XNN_INTERNAL void xnn_pack_f32_to_f16_deconv_goki_w(
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
  const float* kernel,
  const float* bias,
  const void* scale,
  xnn_float16* packed_weights,
  size_t extra_bytes,
  struct subconvolution_params* subconv_params,
  const void* params);

XNN_INTERNAL void xnn_pack_qs8_deconv_goki_w(
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
  const int8_t* kernel,
  const int32_t* bias,
  const float* scale,
  void* packed_weights,
  size_t extra_bytes,
  struct subconvolution_params* subconv_params,
  const struct xnn_qs8_packing_params* params);

XNN_INTERNAL void xnn_pack_qs8_to_qu8_deconv_goki_w(
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
  const int8_t* kernel,
  const int32_t* bias,
  const float* scale,
  void* packed_weights,
  size_t extra_bytes,
  struct subconvolution_params* subconv_params,
  const struct xnn_qs8_packing_params* params);

XNN_INTERNAL void xnn_pack_qs8_to_qu8_deconv_goki_w(
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
  const int8_t* kernel,
  const int32_t* bias,
  const float* scale,
  void* packed_weights,
  size_t extra_bytes,
  struct subconvolution_params* subconv_params,
  const struct xnn_qs8_packing_params* params);

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
  const uint8_t* kernel,
  const int32_t* bias,
  const void* scale,
  void* packed_weights,
  size_t extra_bytes,
  struct subconvolution_params* subconv_params,
  const struct xnn_qu8_packing_params* params);

// DWCONV packing functions. middle_pass_tile and last_pass_tile is 0 for unipass.
// Pack weights and bias such that:
// 1. First block has biases and first_pass_tile weights
// 2. Within this block, we have biases, then weights, in channel_tile, then in channel_subtiles.
// 3. Second block has middle_pass_tile weights, in channel_tile, then in channel_subtiles.
// 4. Last block has last_pass_tile weights, in channel_tile, then in channel_subtiles.
// The first and middle pass of the microkernel runs as many channel_tile as possible, so the number of channel_tile
// tiles is round_up_po2(channels, channel_round)/channel_tile. We use channel_round because rounding to channel_subtile
// might exceed the padding that we have.

typedef void (*xnn_pack_dwconv_ghw_w_fn)(
  size_t first_pass_tile,
  size_t middle_pass_tile,
  size_t last_pass_tile,
  size_t h,
  size_t w,
  size_t c,
  size_t channel_tile,
  size_t channel_subtile,
  size_t channel_round,
  const void* kernel,
  const void* bias,
  const void* scale,
  void* packed_weights,
  size_t per_tile_extra_bytes,
  size_t per_subtile_extra_bytes,
  const void* params);

// Weights layout is channels/(g)roups, (h)eight, (w)idth.
XNN_INTERNAL void xnn_pack_f32_dwconv_ghw_w(
  size_t first_pass_tile,
  size_t middle_pass_tile,
  size_t last_pass_tile,
  size_t h,
  size_t w,
  size_t c,
  size_t channel_tile,
  size_t channel_subtile,
  size_t channel_round,
  const float* kernel,
  const float* bias,
  const void* scale,
  float* packed_weights,
  size_t per_tile_extra_bytes,
  size_t per_subtile_extra_bytes,
  const void* params);

XNN_INTERNAL void xnn_pack_f16_dwconv_ghw_w(
  size_t first_pass_tile,
  size_t middle_pass_tile,
  size_t last_pass_tile,
  size_t h,
  size_t w,
  size_t c,
  size_t channel_tile,
  size_t channel_subtile,
  size_t channel_round,
  const uint16_t* kernel,
  const uint16_t* bias,
  const void* scale,
  uint16_t* packed_weights,
  size_t per_tile_extra_bytes,
  size_t per_subtile_extra_bytes,
  const void* params);

XNN_INTERNAL void xnn_pack_f32_to_f16_dwconv_ghw_w(
  size_t first_pass_tile,
  size_t middle_pass_tile,
  size_t last_pass_tile,
  size_t h,
  size_t w,
  size_t c,
  size_t channel_tile,
  size_t channel_subtile,
  size_t channel_round,
  const float* kernel,
  const float* bias,
  const void* scale,
  xnn_float16* packed_weights,
  size_t per_tile_extra_bytes,
  size_t per_subtile_extra_bytes,
  const void* params);

XNN_INTERNAL void xnn_pack_qs8_dwconv_ghw_w(
  size_t first_pass_tile,
  size_t middle_pass_tile,
  size_t last_pass_tile,
  size_t h,
  size_t w,
  size_t c,
  size_t channel_tile,
  size_t channel_subtile,
  size_t channel_round,
  const int8_t* kernel,
  const int32_t* bias,
  const float* scale,
  void* packed_weights,
  size_t per_tile_extra_bytes,
  size_t per_subtile_extra_bytes,
  const struct xnn_qs8_packing_params* params);

XNN_INTERNAL void xnn_pack_qu8_dwconv_ghw_w(
  size_t first_pass_tile,
  size_t middle_pass_tile,
  size_t last_pass_tile,
  size_t h,
  size_t w,
  size_t c,
  size_t channel_tile,
  size_t channel_subtile,
  size_t channel_round,
  const uint8_t* kernel,
  const int32_t* bias,
  const void* scale,
  void* packed_weights,
  size_t per_tile_extra_bytes,
  size_t per_subtile_extra_bytes,
  const struct xnn_qu8_packing_params* params);

typedef void (*xnn_pack_dwconv_hwg_w_fn)(
  size_t first_pass_tile,
  size_t middle_pass_tile,
  size_t last_pass_tile,
  size_t h,
  size_t w,
  size_t c,
  size_t channel_tile,
  size_t channel_subtile,
  size_t channel_round,
  const void* kernel,
  const void* bias,
  const void* scale,
  void* packed_weights,
  size_t per_tile_extra_bytes,
  size_t per_subtile_extra_bytes,
  const void* params);

// Weights layout is (h)eight, (w)idth, channels/(g)roups.
XNN_INTERNAL void xnn_pack_f32_dwconv_hwg_w(
  size_t first_pass_tile,
  size_t middle_pass_tile,
  size_t last_pass_tile,
  size_t h,
  size_t w,
  size_t c,
  size_t channel_tile,
  size_t channel_subtile,
  size_t channel_round,
  const float* kernel,
  const float* bias,
  const void* scale,
  float* packed_weights,
  size_t per_tile_extra_bytes,
  size_t per_subtile_extra_bytes,
  const void* params);

XNN_INTERNAL void xnn_pack_f16_dwconv_hwg_w(
  size_t first_pass_tile,
  size_t middle_pass_tile,
  size_t last_pass_tile,
  size_t h,
  size_t w,
  size_t c,
  size_t channel_tile,
  size_t channel_subtile,
  size_t channel_round,
  const uint16_t* kernel,
  const uint16_t* bias,
  const void* scale,
  uint16_t* packed_weights,
  size_t per_tile_extra_bytes,
  size_t per_subtile_extra_bytes,
  const void* params);

XNN_INTERNAL void xnn_pack_f32_to_f16_dwconv_hwg_w(
  size_t first_pass_tile,
  size_t middle_pass_tile,
  size_t last_pass_tile,
  size_t h,
  size_t w,
  size_t c,
  size_t channel_tile,
  size_t channel_subtile,
  size_t channel_round,
  const float* kernel,
  const float* bias,
  const void* scale,
  xnn_float16* packed_weights,
  size_t per_tile_extra_bytes,
  size_t per_subtile_extra_bytes,
  const void* params);

XNN_INTERNAL void xnn_pack_qs8_dwconv_hwg_w(
  size_t first_pass_tile,
  size_t middle_pass_tile,
  size_t last_pass_tile,
  size_t h,
  size_t w,
  size_t c,
  size_t channel_tile,
  size_t channel_subtile,
  size_t channel_round,
  const int8_t* kernel,
  const int32_t* bias,
  const float* scale,
  void* packed_weights,
  size_t per_tile_extra_bytes,
  size_t per_subtile_extra_bytes,
  const struct xnn_qs8_packing_params* params);

XNN_INTERNAL void xnn_pack_qu8_dwconv_hwg_w(
  size_t first_pass_tile,
  size_t middle_pass_tile,
  size_t last_pass_tile,
  size_t h,
  size_t w,
  size_t c,
  size_t channel_tile,
  size_t channel_subtile,
  size_t channel_round,
  const uint8_t* kernel,
  const int32_t* bias,
  const void* scale,
  void* packed_weights,
  size_t per_tile_extra_bytes,
  size_t per_subtile_extra_bytes,
  const struct xnn_qu8_packing_params* params);

typedef void (*xnn_pack_f32_gemminc_fn)(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  const float* kernel,
  float* packed_weights,
  const void* params);

XNN_INTERNAL void xnn_pack_f32_gemminc_goi_w(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  const float* kernel,
  float* packed_weights,
  const void* params);

XNN_INTERNAL void xnn_pack_f16_gemminc_goi_w(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  const uint16_t* kernel,
  uint16_t* packed_weights,
  const void* params);


typedef void (*xnn_pack_dconv_oki_w_fn)(
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kh,
  size_t kw,
  const void* kernel,
  const void* bias,
  void* packed_weights,
  const void* params);

XNN_INTERNAL void xnn_pack_f32_dconv_oki_w(
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kh,
  size_t kw,
  const float* kernel,
  const float* bias,
  float* packed_weights,
  const void* params);

XNN_INTERNAL void xnn_pack_f32_to_f16_dconv_oki_w(
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kh,
  size_t kw,
  const float* kernel,
  const float* bias,
  xnn_float16* packed_weights,
  const void* params);

XNN_INTERNAL void xnn_pack_f16_dconv_oki_w(
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kh,
  size_t kw,
  const uint16_t* kernel,
  const uint16_t* bias,
  uint16_t* packed_weights,
  const void* params);


typedef void (*xnn_pack_chw_dwconv_ghw_w_fn)(
  size_t kernel_size,
  size_t groups,
  const void* kernel,
  const void* bias,
  void* packed_weights,
  const void* params);

XNN_INTERNAL void xnn_pack_f32_chw_dwconv_ghw_w(
  size_t kernel_size,
  size_t groups,
  const float* kernel,
  const float* bias,
  float* packed_weights,
  const void* params);

XNN_INTERNAL void xnn_pack_f32_to_f16_chw_dwconv_ghw_w(
  size_t kernel_size,
  size_t groups,
  const float* kernel,
  const float* bias,
  xnn_float16* packed_weights,
  const void* params);

XNN_INTERNAL void xnn_pack_f16_chw_dwconv_ghw_w(
  size_t kernel_size,
  size_t groups,
  const uint16_t* kernel,
  const uint16_t* bias,
  uint16_t* packed_weights,
  const void* params);


typedef void (*xnn_pack_chw_dwconv_hwg_w_fn)(
  size_t kernel_size,
  size_t groups,
  const void* kernel,
  const void* bias,
  void* packed_weights,
  const void* params);

XNN_INTERNAL void xnn_pack_f32_chw_dwconv_hwg_w(
  size_t kernel_size,
  size_t groups,
  const float* kernel,
  const float* bias,
  float* packed_weights,
  const void* params);

XNN_INTERNAL void xnn_pack_f16_chw_dwconv_hwg_w(
  size_t kernel_size,
  size_t groups,
  const uint16_t* kernel,
  const uint16_t* bias,
  uint16_t* packed_weights,
  const void* params);

XNN_INTERNAL void xnn_pack_f32_to_f16_chw_dwconv_hwg_w(
  size_t kernel_size,
  size_t groups,
  const float* kernel,
  const float* bias,
  xnn_float16* packed_weights,
  const void* params);

typedef void (*xnn_pack_vmulcaddc_w_fn)(
  size_t c,
  size_t cr,
  const void* s,
  const void* bias,
  void* packed_weights,
  const void* params);

XNN_INTERNAL void xnn_pack_f32_vmulcaddc_w(
  size_t c,
  size_t cr,
  const float* s,
  const float* bias,
  float* packed_weights,
  const void* params);

XNN_INTERNAL void xnn_pack_f16_vmulcaddc_w(
  size_t c,
  size_t cr,
  const uint16_t* s,
  const uint16_t* bias,
  uint16_t* packed_weights,
  const void* params);

XNN_INTERNAL void xnn_pack_f32_to_f16_vmulcaddc_w(
  size_t c,
  size_t cr,
  const float* s,
  const float* bias,
  xnn_float16* packed_weights,
  const void* params);


// Sparse packing functions.

struct xnn_spmm_packing_params {
  size_t num_nonzeroes;
  size_t num_nonzero_blocks2;
  size_t num_nonzero_blocks4;
  size_t num_block2_nonzeroes;
  size_t num_block4_nonzeroes;
};

typedef void (*xnn_analyze_spmm_w_fn)(
  size_t group_output_channels,
  size_t group_input_channels,
  const void* kernel,
  struct xnn_spmm_packing_params* params);

XNN_INTERNAL void xnn_analyze_f32_spmm_w(
  size_t group_output_channels,
  size_t group_input_channels,
  const float* kernel,
  struct xnn_spmm_packing_params* params);

XNN_INTERNAL void xnn_analyze_f16_spmm_w(
  size_t group_output_channels,
  size_t group_input_channels,
  const xnn_float16* kernel,
  struct xnn_spmm_packing_params* params);


typedef enum xnn_status (*xnn_pack_spmm_w_fn)(
  size_t group_output_channels,
  size_t output_channels_block_size,
  size_t group_input_channels,
  const void* kernel,
  const void* bias,
  int32_t* input_channel_diffs,
  uint32_t* output_channel_nonzeros,
  void* nonzero_values,
  size_t* first_input_channel);

XNN_INTERNAL enum xnn_status xnn_pack_f32_spmm_w(
  size_t group_output_channels,
  size_t output_channels_block_size,
  size_t group_input_channels,
  const float* kernel,
  const float* bias,
  int32_t* input_channel_diffs,
  uint32_t* output_channel_nonzeros,
  float* nonzero_values,
  size_t* first_input_channel);

XNN_INTERNAL enum xnn_status xnn_pack_f32_to_f16_spmm_w(
  size_t group_output_channels,
  size_t output_channels_block_size,
  size_t group_input_channels,
  const float* kernel,
  const float* bias,
  int32_t* input_channel_diffs,
  uint32_t* output_channel_nonzeros,
  xnn_float16* nonzero_values,
  size_t* first_input_channel);

XNN_INTERNAL enum xnn_status xnn_pack_f16_spmm_w(
  size_t group_output_channels,
  size_t output_channels_block_size,
  size_t group_input_channels,
  const xnn_float16* kernel,
  const xnn_float16* bias,
  int32_t* input_channel_diffs,
  uint32_t* output_channel_nonzeros,
  xnn_float16* nonzero_values,
  size_t* first_input_channel);


#ifdef __cplusplus
}  // extern "C"
#endif
