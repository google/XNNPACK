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


#define DECLARE_F32_GAVGPOOL_MULTIPASS_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                                     \
      size_t m,                                                  \
      size_t n,                                                  \
      const float* x,                                            \
      size_t x_stride,                                           \
      const float* zero,                                         \
      float* buffer,                                             \
      float* y,                                                  \
      const union xnn_f32_avgpool_params* params);

DECLARE_F32_GAVGPOOL_MULTIPASS_UKERNEL_FUNCTION(xnn_f32_gavgpool_ukernel_mp7p7q__neon)
DECLARE_F32_GAVGPOOL_MULTIPASS_UKERNEL_FUNCTION(xnn_f32_gavgpool_ukernel_mp7p7q__sse)
DECLARE_F32_GAVGPOOL_MULTIPASS_UKERNEL_FUNCTION(xnn_f32_gavgpool_ukernel_mp7p7q__psimd)
DECLARE_F32_GAVGPOOL_MULTIPASS_UKERNEL_FUNCTION(xnn_f32_gavgpool_ukernel_mp7p7q__wasm)
DECLARE_F32_GAVGPOOL_MULTIPASS_UKERNEL_FUNCTION(xnn_f32_gavgpool_ukernel_mp7p7q__scalar)


#define DECLARE_F32_GAVGPOOL_UNIPASS_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                                   \
      size_t m,                                                \
      size_t n,                                                \
      const float* x,                                          \
      size_t x_stride,                                         \
      const float* zero,                                       \
      float* y,                                                \
      const union xnn_f32_avgpool_params* params);

DECLARE_F32_GAVGPOOL_UNIPASS_UKERNEL_FUNCTION(xnn_f32_gavgpool_ukernel_up7__neon)
DECLARE_F32_GAVGPOOL_UNIPASS_UKERNEL_FUNCTION(xnn_f32_gavgpool_ukernel_up7__sse)
DECLARE_F32_GAVGPOOL_UNIPASS_UKERNEL_FUNCTION(xnn_f32_gavgpool_ukernel_up7__psimd)
DECLARE_F32_GAVGPOOL_UNIPASS_UKERNEL_FUNCTION(xnn_f32_gavgpool_ukernel_up7__wasm)
DECLARE_F32_GAVGPOOL_UNIPASS_UKERNEL_FUNCTION(xnn_f32_gavgpool_ukernel_up7__scalar)

#define DECLARE_Q8_GAVGPOOL_MULTIPASS_UKERNEL_FUNCTION(fn_name)          \
  XNN_INTERNAL void fn_name(                                             \
      size_t m,                                                          \
      size_t n,                                                          \
      const uint8_t* x,                                                  \
      size_t x_stride,                                                   \
      const uint8_t* zero,                                               \
      int32_t* buffer,                                                   \
      uint8_t* y,                                                        \
      const union xnn_q8_avgpool_params* params);

DECLARE_Q8_GAVGPOOL_MULTIPASS_UKERNEL_FUNCTION(xnn_q8_gavgpool_ukernel_mp7p7q__neon)
DECLARE_Q8_GAVGPOOL_MULTIPASS_UKERNEL_FUNCTION(xnn_q8_gavgpool_ukernel_mp7p7q__scalar)
DECLARE_Q8_GAVGPOOL_MULTIPASS_UKERNEL_FUNCTION(xnn_q8_gavgpool_ukernel_mp7p7q__sse2)


#define DECLARE_Q8_GAVGPOOL_UNIPASS_UKERNEL_FUNCTION(fn_name)            \
  XNN_INTERNAL void fn_name(                                             \
      size_t m,                                                          \
      size_t n,                                                          \
      const uint8_t* x,                                                  \
      size_t x_stride,                                                   \
      const uint8_t* zero,                                               \
      uint8_t* y,                                                        \
      const union xnn_q8_avgpool_params* params);

DECLARE_Q8_GAVGPOOL_UNIPASS_UKERNEL_FUNCTION(xnn_q8_gavgpool_ukernel_up7__neon)
DECLARE_Q8_GAVGPOOL_UNIPASS_UKERNEL_FUNCTION(xnn_q8_gavgpool_ukernel_up7__scalar)
DECLARE_Q8_GAVGPOOL_UNIPASS_UKERNEL_FUNCTION(xnn_q8_gavgpool_ukernel_up7__sse2)


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
