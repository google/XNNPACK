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

#include <xnnpack/common.h>
#include <xnnpack/microparams.h>

#ifdef __cplusplus
extern "C" {
#endif


#define DECLARE_F16_MAXPOOL_MINMAX_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                                 \
      size_t output_pixels,                                  \
      size_t kernel_size,                                    \
      size_t channels,                                       \
      const void** input,                                    \
      size_t input_offset,                                   \
      void* output,                                          \
      size_t input_increment,                                \
      size_t output_increment,                               \
      const union xnn_f16_minmax_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

DECLARE_F16_MAXPOOL_MINMAX_UKERNEL_FUNCTION(xnn_f16_maxpool_minmax_ukernel_9p8x__f16c_c8)
DECLARE_F16_MAXPOOL_MINMAX_UKERNEL_FUNCTION(xnn_f16_maxpool_minmax_ukernel_9p8x__neonfp16arith_c8)


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
      const union xnn_f32_minmax_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

DECLARE_F32_MAXPOOL_MINMAX_UKERNEL_FUNCTION(xnn_f32_maxpool_minmax_ukernel_9p8x__neon_c4)
DECLARE_F32_MAXPOOL_MINMAX_UKERNEL_FUNCTION(xnn_f32_maxpool_minmax_ukernel_9p8x__scalar_c1)
DECLARE_F32_MAXPOOL_MINMAX_UKERNEL_FUNCTION(xnn_f32_maxpool_minmax_ukernel_9p8x__sse_c4)
DECLARE_F32_MAXPOOL_MINMAX_UKERNEL_FUNCTION(xnn_f32_maxpool_minmax_ukernel_9p8x__wasm_c1)
DECLARE_F32_MAXPOOL_MINMAX_UKERNEL_FUNCTION(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_arm_c4)
DECLARE_F32_MAXPOOL_MINMAX_UKERNEL_FUNCTION(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_x86_c4)


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
      const union xnn_u8_minmax_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

DECLARE_U8_MAXPOOL_MINMAX_UKERNEL_FUNCTION(xnn_u8_maxpool_minmax_ukernel_9p8x__neon_c16)
DECLARE_U8_MAXPOOL_MINMAX_UKERNEL_FUNCTION(xnn_u8_maxpool_minmax_ukernel_9p8x__scalar_c1)
DECLARE_U8_MAXPOOL_MINMAX_UKERNEL_FUNCTION(xnn_u8_maxpool_minmax_ukernel_9p8x__sse2_c16)
DECLARE_U8_MAXPOOL_MINMAX_UKERNEL_FUNCTION(xnn_u8_maxpool_minmax_ukernel_9p8x__wasmsimd_c16)


#define DECLARE_S8_MAXPOOL_MINMAX_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                                \
      size_t output_pixels,                                 \
      size_t kernel_size,                                   \
      size_t channels,                                      \
      const int8_t** input,                                 \
      size_t input_offset,                                  \
      int8_t* output,                                       \
      size_t input_increment,                               \
      size_t output_increment,                              \
      const union xnn_s8_minmax_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

DECLARE_S8_MAXPOOL_MINMAX_UKERNEL_FUNCTION(xnn_s8_maxpool_minmax_ukernel_2p2x__neon_c16)
DECLARE_S8_MAXPOOL_MINMAX_UKERNEL_FUNCTION(xnn_s8_maxpool_minmax_ukernel_4p3x__neon_c16)
DECLARE_S8_MAXPOOL_MINMAX_UKERNEL_FUNCTION(xnn_s8_maxpool_minmax_ukernel_9p8x__neon_c16)
DECLARE_S8_MAXPOOL_MINMAX_UKERNEL_FUNCTION(xnn_s8_maxpool_minmax_ukernel_9p8x__scalar_c1)
DECLARE_S8_MAXPOOL_MINMAX_UKERNEL_FUNCTION(xnn_s8_maxpool_minmax_ukernel_9p8x__sse2_c16)
DECLARE_S8_MAXPOOL_MINMAX_UKERNEL_FUNCTION(xnn_s8_maxpool_minmax_ukernel_9p8x__sse41_c16)
DECLARE_S8_MAXPOOL_MINMAX_UKERNEL_FUNCTION(xnn_s8_maxpool_minmax_ukernel_9p8x__wasmsimd_c16)


#ifdef __cplusplus
}  // extern "C"
#endif
