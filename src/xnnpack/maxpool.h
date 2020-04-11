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

#include <xnnpack/params.h>
#include <xnnpack/common.h>

#ifdef __cplusplus
extern "C" {
#endif


#define DECLARE_F32_MAXPOOL_MINMAX_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                                 \
      size_t output_pixels,                                  \
      size_t kernel_size,                                    \
      size_t channels,                                       \
      const float** input,                                   \
      size_t input_offset,                                   \
      float* output,                                         \
      size_t input_increment,                                \
      size_t output_increment,                               \
      const union xnn_f32_minmax_params* params);

DECLARE_F32_MAXPOOL_MINMAX_UKERNEL_FUNCTION(xnn_f32_maxpool_minmax_ukernel_9p8x__sse_c4)
DECLARE_F32_MAXPOOL_MINMAX_UKERNEL_FUNCTION(xnn_f32_maxpool_minmax_ukernel_9p8x__neon_c4)
DECLARE_F32_MAXPOOL_MINMAX_UKERNEL_FUNCTION(xnn_f32_maxpool_minmax_ukernel_9p8x__psimd_c4)
DECLARE_F32_MAXPOOL_MINMAX_UKERNEL_FUNCTION(xnn_f32_maxpool_minmax_ukernel_9p8x__wasm_c1)
DECLARE_F32_MAXPOOL_MINMAX_UKERNEL_FUNCTION(xnn_f32_maxpool_minmax_ukernel_9p8x__scalar_c1)


#define DECLARE_U8_MAXPOOL_MINMAX_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                                \
      size_t output_pixels,                                 \
      size_t kernel_size,                                   \
      size_t channels,                                      \
      const uint8_t** input,                                \
      size_t input_offset,                                  \
      uint8_t* output,                                      \
      size_t input_increment,                               \
      size_t output_increment,                              \
      const union xnn_u8_minmax_params* params);

DECLARE_U8_MAXPOOL_MINMAX_UKERNEL_FUNCTION(xnn_u8_maxpool_minmax_ukernel_9p8x__neon_c16)
DECLARE_U8_MAXPOOL_MINMAX_UKERNEL_FUNCTION(xnn_u8_maxpool_minmax_ukernel_9p8x__sse2_c16)
DECLARE_U8_MAXPOOL_MINMAX_UKERNEL_FUNCTION(xnn_u8_maxpool_minmax_ukernel_9p8x__scalar_c1)


#ifdef __cplusplus
}  // extern "C"
#endif
