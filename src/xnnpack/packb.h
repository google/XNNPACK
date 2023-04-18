// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <stdint.h>
#include <stddef.h>

#include <xnnpack/common.h>
#include <xnnpack/microparams.h>

#ifdef __cplusplus
extern "C" {
#endif

#define DECLARE_X32_PACKB_GEMM_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                             \
      size_t groups,                                     \
      size_t channels,                                   \
      const uint32_t* bias,                                 \
      uint32_t* packed_weights,                          \
      size_t channel_tile_stride,                        \
      size_t channel_subtile_stride,                     \
      const union xnn_x32_packb_params* params);         \

DECLARE_X32_PACKB_GEMM_UKERNEL_FUNCTION(xnn_x32_packb_gemm_ukernel_2c1s1r__scalar_float)
DECLARE_X32_PACKB_GEMM_UKERNEL_FUNCTION(xnn_x32_packb_gemm_ukernel_2c1s1r__scalar_int)
DECLARE_X32_PACKB_GEMM_UKERNEL_FUNCTION(xnn_x32_packb_gemm_ukernel_2c2s1r__scalar_float)
DECLARE_X32_PACKB_GEMM_UKERNEL_FUNCTION(xnn_x32_packb_gemm_ukernel_2c2s1r__scalar_int)
DECLARE_X32_PACKB_GEMM_UKERNEL_FUNCTION(xnn_x32_packb_gemm_ukernel_4c1s1r__scalar_float)
DECLARE_X32_PACKB_GEMM_UKERNEL_FUNCTION(xnn_x32_packb_gemm_ukernel_4c1s1r__scalar_int)
DECLARE_X32_PACKB_GEMM_UKERNEL_FUNCTION(xnn_x32_packb_gemm_ukernel_4c4s1r__scalar_float)
DECLARE_X32_PACKB_GEMM_UKERNEL_FUNCTION(xnn_x32_packb_gemm_ukernel_4c4s1r__scalar_int)

DECLARE_X32_PACKB_GEMM_UKERNEL_FUNCTION(xnn_x32_packb_gemm_ukernel_4c4s4r__sse2)
DECLARE_X32_PACKB_GEMM_UKERNEL_FUNCTION(xnn_x32_packb_gemm_ukernel_8c4s4r__sse2)
DECLARE_X32_PACKB_GEMM_UKERNEL_FUNCTION(xnn_x32_packb_gemm_ukernel_16c4s4r__sse2)
DECLARE_X32_PACKB_GEMM_UKERNEL_FUNCTION(xnn_x32_packb_gemm_ukernel_8c8s4r__sse2)
DECLARE_X32_PACKB_GEMM_UKERNEL_FUNCTION(xnn_x32_packb_gemm_ukernel_16c16s4r__sse2)

#ifdef __cplusplus
}  // extern "C"
#endif
