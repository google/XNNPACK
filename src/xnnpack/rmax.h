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


#define DECLARE_F16_RMAX_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                       \
      size_t n,                                    \
      const void* x,                               \
      void* y,                                     \
      const union xnn_f16_default_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

DECLARE_F16_RMAX_UKERNEL_FUNCTION(xnn_f16_rmax_ukernel__f16c_u32)
DECLARE_F16_RMAX_UKERNEL_FUNCTION(xnn_f16_rmax_ukernel__neonfp16arith_u32)


#define DECLARE_F32_RMAX_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                       \
      size_t n,                                    \
      const float* x,                              \
      float* y,                                    \
      const union xnn_f32_default_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

DECLARE_F32_RMAX_UKERNEL_FUNCTION(xnn_f32_rmax_ukernel__avx_u32)
DECLARE_F32_RMAX_UKERNEL_FUNCTION(xnn_f32_rmax_ukernel__avx512f_u64)
DECLARE_F32_RMAX_UKERNEL_FUNCTION(xnn_f32_rmax_ukernel__neon_u16)
DECLARE_F32_RMAX_UKERNEL_FUNCTION(xnn_f32_rmax_ukernel__scalar_u4)
DECLARE_F32_RMAX_UKERNEL_FUNCTION(xnn_f32_rmax_ukernel__sse_u16)
DECLARE_F32_RMAX_UKERNEL_FUNCTION(xnn_f32_rmax_ukernel__wasmsimd_arm_u16)
DECLARE_F32_RMAX_UKERNEL_FUNCTION(xnn_f32_rmax_ukernel__wasmsimd_x86_u16)


#define DECLARE_U8_RMAX_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                      \
      size_t n,                                   \
      const uint8_t* x,                           \
      uint8_t* y,                                 \
      const void* params);

DECLARE_U8_RMAX_UKERNEL_FUNCTION(xnn_u8_rmax_ukernel__neon_u16)
DECLARE_U8_RMAX_UKERNEL_FUNCTION(xnn_u8_rmax_ukernel__scalar_u2)
DECLARE_U8_RMAX_UKERNEL_FUNCTION(xnn_u8_rmax_ukernel__sse2_u16)


#ifdef __cplusplus
}  // extern "C"
#endif
