// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_SRC_XNNPACK_LUT_H_
#define XNNPACK_SRC_XNNPACK_LUT_H_

#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/common.h"

#ifdef __cplusplus
extern "C" {
#endif

#define DECLARE_X8_LUT_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                     \
      size_t n, const uint8_t* x, uint8_t* y,    \
      const uint8_t* table);

DECLARE_X8_LUT_UKERNEL_FUNCTION(xnn_x8_lut_ukernel__scalar_u1)
DECLARE_X8_LUT_UKERNEL_FUNCTION(xnn_x8_lut_ukernel__scalar_u2)
DECLARE_X8_LUT_UKERNEL_FUNCTION(xnn_x8_lut_ukernel__scalar_u4)
DECLARE_X8_LUT_UKERNEL_FUNCTION(xnn_x8_lut_ukernel__scalar_u8)
DECLARE_X8_LUT_UKERNEL_FUNCTION(xnn_x8_lut_ukernel__scalar_u16)

DECLARE_X8_LUT_UKERNEL_FUNCTION(xnn_x8_lut_ukernel__aarch64_neon_tbx128x4_u16)
DECLARE_X8_LUT_UKERNEL_FUNCTION(xnn_x8_lut_ukernel__aarch64_neon_tbx128x4_u32)
DECLARE_X8_LUT_UKERNEL_FUNCTION(xnn_x8_lut_ukernel__aarch64_neon_tbx128x4_u48)
DECLARE_X8_LUT_UKERNEL_FUNCTION(xnn_x8_lut_ukernel__aarch64_neon_tbx128x4_u64)

DECLARE_X8_LUT_UKERNEL_FUNCTION(xnn_x8_lut_ukernel__ssse3_u16)
DECLARE_X8_LUT_UKERNEL_FUNCTION(xnn_x8_lut_ukernel__ssse3_u32)

DECLARE_X8_LUT_UKERNEL_FUNCTION(xnn_x8_lut_ukernel__avx_u16)
DECLARE_X8_LUT_UKERNEL_FUNCTION(xnn_x8_lut_ukernel__avx_u32)
DECLARE_X8_LUT_UKERNEL_FUNCTION(xnn_x8_lut_ukernel__avx_u48)
DECLARE_X8_LUT_UKERNEL_FUNCTION(xnn_x8_lut_ukernel__avx_u64)

DECLARE_X8_LUT_UKERNEL_FUNCTION(xnn_x8_lut_ukernel__avx2_u32)
DECLARE_X8_LUT_UKERNEL_FUNCTION(xnn_x8_lut_ukernel__avx2_u64)
DECLARE_X8_LUT_UKERNEL_FUNCTION(xnn_x8_lut_ukernel__avx2_u96)
DECLARE_X8_LUT_UKERNEL_FUNCTION(xnn_x8_lut_ukernel__avx2_u128)

DECLARE_X8_LUT_UKERNEL_FUNCTION(xnn_x8_lut_ukernel__avx512skx_vpshufb_u64)
DECLARE_X8_LUT_UKERNEL_FUNCTION(xnn_x8_lut_ukernel__avx512skx_vpshufb_u128)
DECLARE_X8_LUT_UKERNEL_FUNCTION(xnn_x8_lut_ukernel__avx512skx_vpshufb_u192)
DECLARE_X8_LUT_UKERNEL_FUNCTION(xnn_x8_lut_ukernel__avx512skx_vpshufb_u256)

DECLARE_X8_LUT_UKERNEL_FUNCTION(xnn_x8_lut_ukernel__avx512vbmi_vpermx2b_u64)
DECLARE_X8_LUT_UKERNEL_FUNCTION(xnn_x8_lut_ukernel__avx512vbmi_vpermx2b_u128)
DECLARE_X8_LUT_UKERNEL_FUNCTION(xnn_x8_lut_ukernel__avx512vbmi_vpermx2b_u192)
DECLARE_X8_LUT_UKERNEL_FUNCTION(xnn_x8_lut_ukernel__avx512vbmi_vpermx2b_u256)

DECLARE_X8_LUT_UKERNEL_FUNCTION(xnn_x8_lut_ukernel__wasmpshufb_u16)
DECLARE_X8_LUT_UKERNEL_FUNCTION(xnn_x8_lut_ukernel__wasmpshufb_u32)
DECLARE_X8_LUT_UKERNEL_FUNCTION(xnn_x8_lut_ukernel__wasmpshufb_u48)
DECLARE_X8_LUT_UKERNEL_FUNCTION(xnn_x8_lut_ukernel__wasmpshufb_u64)

DECLARE_X8_LUT_UKERNEL_FUNCTION(xnn_x8_lut_ukernel__wasmsimd_u16)
DECLARE_X8_LUT_UKERNEL_FUNCTION(xnn_x8_lut_ukernel__wasmsimd_u32)
DECLARE_X8_LUT_UKERNEL_FUNCTION(xnn_x8_lut_ukernel__wasmsimd_u48)
DECLARE_X8_LUT_UKERNEL_FUNCTION(xnn_x8_lut_ukernel__wasmsimd_u64)

#define XNN_UKERNEL(arch_flags, fn_name, datatype, params_type, init_params) \
  XNN_INTERNAL void fn_name(size_t n, const uint8_t* x, const uint32_t* t,   \
                            uint8_t* y);
#include "src/u8-lut32norm/u8-lut32norm.inc"
#undef XNN_UKERNEL


#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // XNNPACK_SRC_XNNPACK_LUT_H_
