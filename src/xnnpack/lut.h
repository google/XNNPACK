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


#define DECLARE_X8_LUT_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                     \
      size_t n,                                  \
      const uint8_t* x,                          \
      uint8_t* y,                                \
      const uint8_t* t);

DECLARE_X8_LUT_UKERNEL_FUNCTION(xnn_x8_lut_ukernel__scalar_x1)
DECLARE_X8_LUT_UKERNEL_FUNCTION(xnn_x8_lut_ukernel__scalar_x2)
DECLARE_X8_LUT_UKERNEL_FUNCTION(xnn_x8_lut_ukernel__scalar_x4)
DECLARE_X8_LUT_UKERNEL_FUNCTION(xnn_x8_lut_ukernel__scalar_x8)
DECLARE_X8_LUT_UKERNEL_FUNCTION(xnn_x8_lut_ukernel__scalar_x16)

DECLARE_X8_LUT_UKERNEL_FUNCTION(xnn_x8_lut_ukernel__neon_tbx128x4_x16)
DECLARE_X8_LUT_UKERNEL_FUNCTION(xnn_x8_lut_ukernel__neon_tbx128x4_x32)
DECLARE_X8_LUT_UKERNEL_FUNCTION(xnn_x8_lut_ukernel__neon_tbx128x4_x48)
DECLARE_X8_LUT_UKERNEL_FUNCTION(xnn_x8_lut_ukernel__neon_tbx128x4_x64)

DECLARE_X8_LUT_UKERNEL_FUNCTION(xnn_x8_lut_ukernel__ssse3_x16)
DECLARE_X8_LUT_UKERNEL_FUNCTION(xnn_x8_lut_ukernel__ssse3_x32)

DECLARE_X8_LUT_UKERNEL_FUNCTION(xnn_x8_lut_ukernel__avx_x16)
DECLARE_X8_LUT_UKERNEL_FUNCTION(xnn_x8_lut_ukernel__avx_x32)
DECLARE_X8_LUT_UKERNEL_FUNCTION(xnn_x8_lut_ukernel__avx_x48)
DECLARE_X8_LUT_UKERNEL_FUNCTION(xnn_x8_lut_ukernel__avx_x64)

DECLARE_X8_LUT_UKERNEL_FUNCTION(xnn_x8_lut_ukernel__avx2_x32)
DECLARE_X8_LUT_UKERNEL_FUNCTION(xnn_x8_lut_ukernel__avx2_x64)
DECLARE_X8_LUT_UKERNEL_FUNCTION(xnn_x8_lut_ukernel__avx2_x96)
DECLARE_X8_LUT_UKERNEL_FUNCTION(xnn_x8_lut_ukernel__avx2_x128)

DECLARE_X8_LUT_UKERNEL_FUNCTION(xnn_x8_lut_ukernel__avx512skx_vpshufb_x64)
DECLARE_X8_LUT_UKERNEL_FUNCTION(xnn_x8_lut_ukernel__avx512skx_vpshufb_x128)
DECLARE_X8_LUT_UKERNEL_FUNCTION(xnn_x8_lut_ukernel__avx512skx_vpshufb_x192)
DECLARE_X8_LUT_UKERNEL_FUNCTION(xnn_x8_lut_ukernel__avx512skx_vpshufb_x256)

DECLARE_X8_LUT_UKERNEL_FUNCTION(xnn_x8_lut_ukernel__wasmsimd_x16)
DECLARE_X8_LUT_UKERNEL_FUNCTION(xnn_x8_lut_ukernel__wasmsimd_x32)
DECLARE_X8_LUT_UKERNEL_FUNCTION(xnn_x8_lut_ukernel__wasmsimd_x48)
DECLARE_X8_LUT_UKERNEL_FUNCTION(xnn_x8_lut_ukernel__wasmsimd_x64)


#define DECLARE_U8_LUT32NORM_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                           \
      size_t n,                                        \
      const uint8_t* x,                                \
      const uint32_t* t,                               \
      uint8_t* y);

DECLARE_U8_LUT32NORM_UKERNEL_FUNCTION(xnn_u8_lut32norm_ukernel__scalar)


#ifdef __cplusplus
}  // extern "C"
#endif
