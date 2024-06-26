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

#include "xnnpack/common.h"
#include "xnnpack/microparams.h"

#ifdef __cplusplus
extern "C" {
#endif


#define DECLARE_F16_AVGPOOL_MINMAX_MULTIPASS_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                                           \
      size_t output_pixels,                                            \
      size_t kernel_elements,                                          \
      size_t channels,                                                 \
      const void** input,                                              \
      size_t input_offset,                                             \
      const void* zero,                                                \
      void* buffer,                                                    \
      void* output,                                                    \
      size_t input_increment,                                          \
      size_t output_increment,                                         \
      const union xnn_f16_scaleminmax_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

DECLARE_F16_AVGPOOL_MINMAX_MULTIPASS_UKERNEL_FUNCTION(xnn_f16_avgpool_minmax_ukernel_9p8x__f16c_c8)
DECLARE_F16_AVGPOOL_MINMAX_MULTIPASS_UKERNEL_FUNCTION(xnn_f16_avgpool_minmax_ukernel_9p8x__neonfp16arith_c8)


#define DECLARE_F16_AVGPOOL_MINMAX_UNIPASS_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                                         \
      size_t output_pixels,                                          \
      size_t kernel_elements,                                        \
      size_t channels,                                               \
      const void** input,                                            \
      size_t input_offset,                                           \
      const void* zero,                                              \
      void* output,                                                  \
      size_t input_increment,                                        \
      size_t output_increment,                                       \
      const union xnn_f16_scaleminmax_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

DECLARE_F16_AVGPOOL_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f16_avgpool_minmax_ukernel_9x__f16c_c8)
DECLARE_F16_AVGPOOL_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f16_avgpool_minmax_ukernel_9x__neonfp16arith_c8)


#define DECLARE_F32_AVGPOOL_MINMAX_MULTIPASS_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                                           \
      size_t output_pixels,                                            \
      size_t kernel_elements,                                          \
      size_t channels,                                                 \
      const float** input,                                             \
      size_t input_offset,                                             \
      const float* zero,                                               \
      float* buffer,                                                   \
      float* output,                                                   \
      size_t input_increment,                                          \
      size_t output_increment,                                         \
      const union xnn_f32_scaleminmax_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

DECLARE_F32_AVGPOOL_MINMAX_MULTIPASS_UKERNEL_FUNCTION(xnn_f32_avgpool_minmax_ukernel_9p8x__neon_c4)
DECLARE_F32_AVGPOOL_MINMAX_MULTIPASS_UKERNEL_FUNCTION(xnn_f32_avgpool_minmax_ukernel_9p8x__rvv_c1v)
DECLARE_F32_AVGPOOL_MINMAX_MULTIPASS_UKERNEL_FUNCTION(xnn_f32_avgpool_minmax_ukernel_9p8x__rvv_c2v)
DECLARE_F32_AVGPOOL_MINMAX_MULTIPASS_UKERNEL_FUNCTION(xnn_f32_avgpool_minmax_ukernel_9p8x__scalar_c1)
DECLARE_F32_AVGPOOL_MINMAX_MULTIPASS_UKERNEL_FUNCTION(xnn_f32_avgpool_minmax_ukernel_9p8x__sse_c4)
DECLARE_F32_AVGPOOL_MINMAX_MULTIPASS_UKERNEL_FUNCTION(xnn_f32_avgpool_minmax_ukernel_9p8x__wasm_c1)
DECLARE_F32_AVGPOOL_MINMAX_MULTIPASS_UKERNEL_FUNCTION(xnn_f32_avgpool_minmax_ukernel_9p8x__wasmsimd_arm_c4)
DECLARE_F32_AVGPOOL_MINMAX_MULTIPASS_UKERNEL_FUNCTION(xnn_f32_avgpool_minmax_ukernel_9p8x__wasmsimd_x86_c4)


#define DECLARE_F32_AVGPOOL_MINMAX_UNIPASS_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                                         \
      size_t output_pixels,                                          \
      size_t kernel_elements,                                        \
      size_t channels,                                               \
      const float** input,                                           \
      size_t input_offset,                                           \
      const float* zero,                                             \
      float* output,                                                 \
      size_t input_increment,                                        \
      size_t output_increment,                                       \
      const union xnn_f32_scaleminmax_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

DECLARE_F32_AVGPOOL_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_avgpool_minmax_ukernel_9x__neon_c4)
DECLARE_F32_AVGPOOL_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_avgpool_minmax_ukernel_9x__rvv_c1v)
DECLARE_F32_AVGPOOL_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_avgpool_minmax_ukernel_9x__rvv_c2v)
DECLARE_F32_AVGPOOL_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_avgpool_minmax_ukernel_9x__scalar_c1)
DECLARE_F32_AVGPOOL_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_avgpool_minmax_ukernel_9x__sse_c4)
DECLARE_F32_AVGPOOL_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_avgpool_minmax_ukernel_9x__wasm_c1)
DECLARE_F32_AVGPOOL_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_avgpool_minmax_ukernel_9x__wasmsimd_arm_c4)
DECLARE_F32_AVGPOOL_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_avgpool_minmax_ukernel_9x__wasmsimd_x86_c4)


#define DECLARE_QU8_AVGPOOL_MINMAX_MULTIPASS_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                                           \
      size_t output_pixels,                                            \
      size_t kernel_elements,                                          \
      size_t channels,                                                 \
      const uint8_t** input,                                           \
      size_t input_offset,                                             \
      const uint8_t* zero,                                             \
      int32_t* buffer,                                                 \
      uint8_t* output,                                                 \
      size_t input_increment,                                          \
      size_t output_increment,                                         \
      const union xnn_qu8_avgpool_minmax_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

DECLARE_QU8_AVGPOOL_MINMAX_MULTIPASS_UKERNEL_FUNCTION(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__neon_c8)
DECLARE_QU8_AVGPOOL_MINMAX_MULTIPASS_UKERNEL_FUNCTION(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__scalar_imagic_c1)
DECLARE_QU8_AVGPOOL_MINMAX_MULTIPASS_UKERNEL_FUNCTION(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__sse2_c8)


#define DECLARE_QU8_AVGPOOL_MINMAX_UNIPASS_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                                         \
      size_t output_pixels,                                          \
      size_t kernel_elements,                                        \
      size_t channels,                                               \
      const uint8_t** input,                                         \
      size_t input_offset,                                           \
      const uint8_t* zero,                                           \
      uint8_t* output,                                               \
      size_t input_increment,                                        \
      size_t output_increment,                                       \
      const union xnn_qu8_avgpool_minmax_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

DECLARE_QU8_AVGPOOL_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__neon_c8)
DECLARE_QU8_AVGPOOL_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__scalar_imagic_c1)
DECLARE_QU8_AVGPOOL_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__sse2_c8)


#ifdef __cplusplus
}  // extern "C"
#endif
