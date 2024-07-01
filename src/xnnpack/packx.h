// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <stddef.h>
#include <stdint.h>

#include "xnnpack/common.h"

#ifdef __cplusplus
extern "C" {
#endif


#define DECLARE_X32_PACKX_UKERNEL_FUNCTION(fn_name)  \
  XNN_INTERNAL void fn_name(                         \
      size_t m,                                      \
      size_t k,                                      \
      const uint32_t* x,                             \
      size_t x_stride,                               \
      uint32_t* y);

DECLARE_X32_PACKX_UKERNEL_FUNCTION(xnn_x32_packx_ukernel_2x__scalar)
DECLARE_X32_PACKX_UKERNEL_FUNCTION(xnn_x32_packx_ukernel_3x__scalar)
DECLARE_X32_PACKX_UKERNEL_FUNCTION(xnn_x32_packx_ukernel_4x__scalar)
DECLARE_X32_PACKX_UKERNEL_FUNCTION(xnn_x32_packx_ukernel_4x__sse)
DECLARE_X32_PACKX_UKERNEL_FUNCTION(xnn_x32_packx_ukernel_4x__wasmsimd)

DECLARE_X32_PACKX_UKERNEL_FUNCTION(xnn_x32_packx_ukernel_4x__neon_st4_x4)
DECLARE_X32_PACKX_UKERNEL_FUNCTION(xnn_x32_packx_ukernel_4x__neon_st4_x4_prfm)
DECLARE_X32_PACKX_UKERNEL_FUNCTION(xnn_x32_packx_ukernel_4x__neon_st4_x8)
DECLARE_X32_PACKX_UKERNEL_FUNCTION(xnn_x32_packx_ukernel_4x__neon_st4_x8_prfm)
DECLARE_X32_PACKX_UKERNEL_FUNCTION(xnn_x32_packx_ukernel_8x__neon_st4_x4)
DECLARE_X32_PACKX_UKERNEL_FUNCTION(xnn_x32_packx_ukernel_8x__neon_st4_x4_prfm)
DECLARE_X32_PACKX_UKERNEL_FUNCTION(xnn_x32_packx_ukernel_8x__neon_st4_x8)
DECLARE_X32_PACKX_UKERNEL_FUNCTION(xnn_x32_packx_ukernel_8x__neon_st4_x8_prfm)


#ifdef __cplusplus
}  // extern "C"
#endif
