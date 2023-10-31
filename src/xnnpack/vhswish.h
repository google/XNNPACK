// Copyright 2023 Google LLC
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


#define DECLARE_QS8_VHSWISH_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                          \
      size_t n,                                       \
      const int8_t* input,                            \
      int8_t* output,                                 \
      const union xnn_qs8_hswish_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

DECLARE_QS8_VHSWISH_UKERNEL_FUNCTION(xnn_qs8_vhswish_ukernel__neon_u8)
DECLARE_QS8_VHSWISH_UKERNEL_FUNCTION(xnn_qs8_vhswish_ukernel__neon_u16)
DECLARE_QS8_VHSWISH_UKERNEL_FUNCTION(xnn_qs8_vhswish_ukernel__neon_u32)

DECLARE_QS8_VHSWISH_UKERNEL_FUNCTION(xnn_qs8_vhswish_ukernel__sse2_u16)
DECLARE_QS8_VHSWISH_UKERNEL_FUNCTION(xnn_qs8_vhswish_ukernel__sse2_u32)

DECLARE_QS8_VHSWISH_UKERNEL_FUNCTION(xnn_qs8_vhswish_ukernel__ssse3_u16)
DECLARE_QS8_VHSWISH_UKERNEL_FUNCTION(xnn_qs8_vhswish_ukernel__ssse3_u32)

DECLARE_QS8_VHSWISH_UKERNEL_FUNCTION(xnn_qs8_vhswish_ukernel__sse41_u8)
DECLARE_QS8_VHSWISH_UKERNEL_FUNCTION(xnn_qs8_vhswish_ukernel__sse41_u16)
DECLARE_QS8_VHSWISH_UKERNEL_FUNCTION(xnn_qs8_vhswish_ukernel__sse41_u32)

DECLARE_QS8_VHSWISH_UKERNEL_FUNCTION(xnn_qs8_vhswish_ukernel__avx_u8)
DECLARE_QS8_VHSWISH_UKERNEL_FUNCTION(xnn_qs8_vhswish_ukernel__avx_u16)
DECLARE_QS8_VHSWISH_UKERNEL_FUNCTION(xnn_qs8_vhswish_ukernel__avx_u32)

DECLARE_QS8_VHSWISH_UKERNEL_FUNCTION(xnn_qs8_vhswish_ukernel__wasmsimd_u8)
DECLARE_QS8_VHSWISH_UKERNEL_FUNCTION(xnn_qs8_vhswish_ukernel__wasmsimd_u16)
DECLARE_QS8_VHSWISH_UKERNEL_FUNCTION(xnn_qs8_vhswish_ukernel__wasmsimd_u32)

DECLARE_QS8_VHSWISH_UKERNEL_FUNCTION(xnn_qs8_vhswish_ukernel__scalar_u1)
DECLARE_QS8_VHSWISH_UKERNEL_FUNCTION(xnn_qs8_vhswish_ukernel__scalar_u2)
DECLARE_QS8_VHSWISH_UKERNEL_FUNCTION(xnn_qs8_vhswish_ukernel__scalar_u4)

#define DECLARE_QU8_VHSWISH_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                          \
      size_t n,                                       \
      const uint8_t* input,                            \
      uint8_t* output,                                 \
      const union xnn_qu8_hswish_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

DECLARE_QU8_VHSWISH_UKERNEL_FUNCTION(xnn_qu8_vhswish_ukernel__neon_u8)
DECLARE_QU8_VHSWISH_UKERNEL_FUNCTION(xnn_qu8_vhswish_ukernel__neon_u16)
DECLARE_QU8_VHSWISH_UKERNEL_FUNCTION(xnn_qu8_vhswish_ukernel__neon_u32)

DECLARE_QU8_VHSWISH_UKERNEL_FUNCTION(xnn_qu8_vhswish_ukernel__sse2_u16)
DECLARE_QU8_VHSWISH_UKERNEL_FUNCTION(xnn_qu8_vhswish_ukernel__sse2_u32)

DECLARE_QU8_VHSWISH_UKERNEL_FUNCTION(xnn_qu8_vhswish_ukernel__ssse3_u16)
DECLARE_QU8_VHSWISH_UKERNEL_FUNCTION(xnn_qu8_vhswish_ukernel__ssse3_u32)

DECLARE_QU8_VHSWISH_UKERNEL_FUNCTION(xnn_qu8_vhswish_ukernel__sse41_u8)
DECLARE_QU8_VHSWISH_UKERNEL_FUNCTION(xnn_qu8_vhswish_ukernel__sse41_u16)
DECLARE_QU8_VHSWISH_UKERNEL_FUNCTION(xnn_qu8_vhswish_ukernel__sse41_u32)

DECLARE_QU8_VHSWISH_UKERNEL_FUNCTION(xnn_qu8_vhswish_ukernel__avx_u8)
DECLARE_QU8_VHSWISH_UKERNEL_FUNCTION(xnn_qu8_vhswish_ukernel__avx_u16)
DECLARE_QU8_VHSWISH_UKERNEL_FUNCTION(xnn_qu8_vhswish_ukernel__avx_u32)

DECLARE_QU8_VHSWISH_UKERNEL_FUNCTION(xnn_qu8_vhswish_ukernel__wasmsimd_u8)
DECLARE_QU8_VHSWISH_UKERNEL_FUNCTION(xnn_qu8_vhswish_ukernel__wasmsimd_u16)
DECLARE_QU8_VHSWISH_UKERNEL_FUNCTION(xnn_qu8_vhswish_ukernel__wasmsimd_u32)

DECLARE_QU8_VHSWISH_UKERNEL_FUNCTION(xnn_qu8_vhswish_ukernel__scalar_u1)
DECLARE_QU8_VHSWISH_UKERNEL_FUNCTION(xnn_qu8_vhswish_ukernel__scalar_u2)
DECLARE_QU8_VHSWISH_UKERNEL_FUNCTION(xnn_qu8_vhswish_ukernel__scalar_u4)

#ifdef __cplusplus
}  // extern "C"
#endif
