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


#define DECLARE_F32_GAVGPOOL_MINMAX_MULTIPASS_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                                            \
      size_t rows,                                                      \
      size_t channels,                                                  \
      const float* input,                                               \
      size_t input_stride,                                              \
      const float* zero,                                                \
      float* buffer,                                                    \
      float* output,                                                    \
      const union xnn_f32_scaleminmax_params* params);

DECLARE_F32_GAVGPOOL_MINMAX_MULTIPASS_UKERNEL_FUNCTION(xnn_f32_gavgpool_minmax_ukernel_7p7x__neon_c4)
DECLARE_F32_GAVGPOOL_MINMAX_MULTIPASS_UKERNEL_FUNCTION(xnn_f32_gavgpool_minmax_ukernel_7p7x__sse_c4)
DECLARE_F32_GAVGPOOL_MINMAX_MULTIPASS_UKERNEL_FUNCTION(xnn_f32_gavgpool_minmax_ukernel_7p7x__psimd_c4)
DECLARE_F32_GAVGPOOL_MINMAX_MULTIPASS_UKERNEL_FUNCTION(xnn_f32_gavgpool_minmax_ukernel_7p7x__wasm_c1)
DECLARE_F32_GAVGPOOL_MINMAX_MULTIPASS_UKERNEL_FUNCTION(xnn_f32_gavgpool_minmax_ukernel_7p7x__scalar_c1)


#define DECLARE_F32_GAVGPOOL_MINMAX_UNIPASS_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                                          \
      size_t rows,                                                    \
      size_t channels,                                                \
      const float* input,                                             \
      size_t input_stride,                                            \
      const float* zero,                                              \
      float* output,                                                  \
      const union xnn_f32_scaleminmax_params* params);

DECLARE_F32_GAVGPOOL_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_gavgpool_minmax_ukernel_7x__neon_c4)
DECLARE_F32_GAVGPOOL_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_gavgpool_minmax_ukernel_7x__sse_c4)
DECLARE_F32_GAVGPOOL_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_gavgpool_minmax_ukernel_7x__psimd_c4)
DECLARE_F32_GAVGPOOL_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_gavgpool_minmax_ukernel_7x__wasm_c1)
DECLARE_F32_GAVGPOOL_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_gavgpool_minmax_ukernel_7x__scalar_c1)

#define DECLARE_Q8_GAVGPOOL_MINMAX_MULTIPASS_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                                           \
      size_t rows,                                                     \
      size_t channels,                                                 \
      const uint8_t* input,                                            \
      size_t input_stride,                                             \
      const uint8_t* zero,                                             \
      int32_t* buffer,                                                 \
      uint8_t* output,                                                 \
      const union xnn_q8_avgpool_params* params);

DECLARE_Q8_GAVGPOOL_MINMAX_MULTIPASS_UKERNEL_FUNCTION(xnn_q8_gavgpool_minmax_ukernel_7p7x__neon_c8)
DECLARE_Q8_GAVGPOOL_MINMAX_MULTIPASS_UKERNEL_FUNCTION(xnn_q8_gavgpool_minmax_ukernel_7p7x__sse2_c8)
DECLARE_Q8_GAVGPOOL_MINMAX_MULTIPASS_UKERNEL_FUNCTION(xnn_q8_gavgpool_minmax_ukernel_7p7x__scalar_c1)


#define DECLARE_Q8_GAVGPOOL_MINMAX_UNIPASS_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                                         \
      size_t rows,                                                   \
      size_t channels,                                               \
      const uint8_t* input,                                          \
      size_t input_stride,                                           \
      const uint8_t* zero,                                           \
      uint8_t* output,                                               \
      const union xnn_q8_avgpool_params* params);

DECLARE_Q8_GAVGPOOL_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_q8_gavgpool_minmax_ukernel_7x__neon_c8)
DECLARE_Q8_GAVGPOOL_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_q8_gavgpool_minmax_ukernel_7x__sse2_c8)
DECLARE_Q8_GAVGPOOL_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_q8_gavgpool_minmax_ukernel_7x__scalar_c1)


#define DECLARE_F32_GAVGPOOL_SPCHW_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                                 \
      size_t elements,                                       \
      size_t channels,                                       \
      const float* input,                                    \
      float* output,                                         \
      const union xnn_f32_gavgpool_params* params);

DECLARE_F32_GAVGPOOL_SPCHW_UKERNEL_FUNCTION(xnn_f32_gavgpool_spchw_ukernel__neon_x4)
DECLARE_F32_GAVGPOOL_SPCHW_UKERNEL_FUNCTION(xnn_f32_gavgpool_spchw_ukernel__sse_x4)
DECLARE_F32_GAVGPOOL_SPCHW_UKERNEL_FUNCTION(xnn_f32_gavgpool_spchw_ukernel__scalar_x1)


#ifdef __cplusplus
}  // extern "C"
#endif
