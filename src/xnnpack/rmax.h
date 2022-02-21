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

#ifdef __cplusplus
extern "C" {
#endif


#define DECLARE_F16_RMAX_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                       \
      size_t n,                                    \
      const void* x,                               \
      void* y);

DECLARE_F16_RMAX_UKERNEL_FUNCTION(xnn_f16_rmax_ukernel__f16c)
DECLARE_F16_RMAX_UKERNEL_FUNCTION(xnn_f16_rmax_ukernel__neonfp16arith)


#define DECLARE_F32_RMAX_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                       \
      size_t n,                                    \
      const float* x,                              \
      float* y);

DECLARE_F32_RMAX_UKERNEL_FUNCTION(xnn_f32_rmax_ukernel__avx)
DECLARE_F32_RMAX_UKERNEL_FUNCTION(xnn_f32_rmax_ukernel__avx512f)
DECLARE_F32_RMAX_UKERNEL_FUNCTION(xnn_f32_rmax_ukernel__neon)
DECLARE_F32_RMAX_UKERNEL_FUNCTION(xnn_f32_rmax_ukernel__scalar)
DECLARE_F32_RMAX_UKERNEL_FUNCTION(xnn_f32_rmax_ukernel__sse)
DECLARE_F32_RMAX_UKERNEL_FUNCTION(xnn_f32_rmax_ukernel__wasmsimd_arm)
DECLARE_F32_RMAX_UKERNEL_FUNCTION(xnn_f32_rmax_ukernel__wasmsimd_x86)


#define DECLARE_U8_RMAX_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                      \
      size_t n,                                   \
      const uint8_t* x,                           \
      uint8_t* y);

DECLARE_U8_RMAX_UKERNEL_FUNCTION(xnn_u8_rmax_ukernel__neon)
DECLARE_U8_RMAX_UKERNEL_FUNCTION(xnn_u8_rmax_ukernel__scalar)
DECLARE_U8_RMAX_UKERNEL_FUNCTION(xnn_u8_rmax_ukernel__sse2)


#ifdef __cplusplus
}  // extern "C"
#endif
