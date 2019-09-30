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


#define DECLARE_F32_MAXPOOL_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                          \
      size_t n,                                       \
      size_t ks,                                      \
      size_t kc,                                      \
      const float** x,                                \
      float* y,                                       \
      size_t x_increment,                             \
      size_t y_increment,                             \
      const union xnn_f32_output_params* params);

DECLARE_F32_MAXPOOL_UKERNEL_FUNCTION(xnn_f32_maxpool_ukernel_9p8q__psimd)
DECLARE_F32_MAXPOOL_UKERNEL_FUNCTION(xnn_f32_maxpool_ukernel_9p8q__scalar)
DECLARE_F32_MAXPOOL_UKERNEL_FUNCTION(xnn_f32_maxpool_ukernel_9p8q__sse)


#define DECLARE_U8_MAXPOOL_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                         \
      size_t n,                                      \
      size_t ks,                                     \
      size_t kc,                                     \
      const uint8_t** x,                             \
      uint8_t* y,                                    \
      size_t x_increment,                            \
      size_t y_increment,                            \
      const union xnn_u8_output_params* params);

DECLARE_U8_MAXPOOL_UKERNEL_FUNCTION(xnn_u8_maxpool_ukernel_9p8q__neon)
DECLARE_U8_MAXPOOL_UKERNEL_FUNCTION(xnn_u8_maxpool_ukernel_9p8q__sse2)
DECLARE_U8_MAXPOOL_UKERNEL_FUNCTION(xnn_u8_maxpool_ukernel_9p8q__scalar)


#ifdef __cplusplus
}  // extern "C"
#endif
