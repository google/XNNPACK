// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <stddef.h>
#include <stdint.h>

#include "xnnpack/common.h"
#include "xnnpack/microparams.h"

#ifdef __cplusplus
extern "C" {
#endif


#define DECLARE_QS8_VLRELU_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                         \
      size_t n,                                      \
      const int8_t* input,                           \
      int8_t* output,                                \
      const union xnn_qs8_lrelu_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

DECLARE_QS8_VLRELU_UKERNEL_FUNCTION(xnn_qs8_vlrelu_ukernel__neon_u8)
DECLARE_QS8_VLRELU_UKERNEL_FUNCTION(xnn_qs8_vlrelu_ukernel__neon_u16)
DECLARE_QS8_VLRELU_UKERNEL_FUNCTION(xnn_qs8_vlrelu_ukernel__neon_u32)

DECLARE_QS8_VLRELU_UKERNEL_FUNCTION(xnn_qs8_vlrelu_ukernel__sse2_u16)
DECLARE_QS8_VLRELU_UKERNEL_FUNCTION(xnn_qs8_vlrelu_ukernel__sse2_u32)

DECLARE_QS8_VLRELU_UKERNEL_FUNCTION(xnn_qs8_vlrelu_ukernel__ssse3_u16)
DECLARE_QS8_VLRELU_UKERNEL_FUNCTION(xnn_qs8_vlrelu_ukernel__ssse3_u32)

DECLARE_QS8_VLRELU_UKERNEL_FUNCTION(xnn_qs8_vlrelu_ukernel__sse41_u8)
DECLARE_QS8_VLRELU_UKERNEL_FUNCTION(xnn_qs8_vlrelu_ukernel__sse41_u16)
DECLARE_QS8_VLRELU_UKERNEL_FUNCTION(xnn_qs8_vlrelu_ukernel__sse41_u32)

DECLARE_QS8_VLRELU_UKERNEL_FUNCTION(xnn_qs8_vlrelu_ukernel__avx_u8)
DECLARE_QS8_VLRELU_UKERNEL_FUNCTION(xnn_qs8_vlrelu_ukernel__avx_u16)
DECLARE_QS8_VLRELU_UKERNEL_FUNCTION(xnn_qs8_vlrelu_ukernel__avx_u32)

DECLARE_QS8_VLRELU_UKERNEL_FUNCTION(xnn_qs8_vlrelu_ukernel__avx2_u16)
DECLARE_QS8_VLRELU_UKERNEL_FUNCTION(xnn_qs8_vlrelu_ukernel__avx2_u32)
DECLARE_QS8_VLRELU_UKERNEL_FUNCTION(xnn_qs8_vlrelu_ukernel__avx2_u64)

DECLARE_QS8_VLRELU_UKERNEL_FUNCTION(xnn_qs8_vlrelu_ukernel__wasmsimd_arm_u16)
DECLARE_QS8_VLRELU_UKERNEL_FUNCTION(xnn_qs8_vlrelu_ukernel__wasmsimd_arm_u32)

DECLARE_QS8_VLRELU_UKERNEL_FUNCTION(xnn_qs8_vlrelu_ukernel__wasmsimd_x86_u8)
DECLARE_QS8_VLRELU_UKERNEL_FUNCTION(xnn_qs8_vlrelu_ukernel__wasmsimd_x86_u16)
DECLARE_QS8_VLRELU_UKERNEL_FUNCTION(xnn_qs8_vlrelu_ukernel__wasmsimd_x86_u32)

DECLARE_QS8_VLRELU_UKERNEL_FUNCTION(xnn_qs8_vlrelu_ukernel__wasmrelaxedsimd_arm_u16)
DECLARE_QS8_VLRELU_UKERNEL_FUNCTION(xnn_qs8_vlrelu_ukernel__wasmrelaxedsimd_arm_u32)

DECLARE_QS8_VLRELU_UKERNEL_FUNCTION(xnn_qs8_vlrelu_ukernel__wasmrelaxedsimd_x86_u8)
DECLARE_QS8_VLRELU_UKERNEL_FUNCTION(xnn_qs8_vlrelu_ukernel__wasmrelaxedsimd_x86_u16)
DECLARE_QS8_VLRELU_UKERNEL_FUNCTION(xnn_qs8_vlrelu_ukernel__wasmrelaxedsimd_x86_u32)

DECLARE_QS8_VLRELU_UKERNEL_FUNCTION(xnn_qs8_vlrelu_ukernel__armsimd32_u4)
DECLARE_QS8_VLRELU_UKERNEL_FUNCTION(xnn_qs8_vlrelu_ukernel__armsimd32_u8)

DECLARE_QS8_VLRELU_UKERNEL_FUNCTION(xnn_qs8_vlrelu_ukernel__scalar_andxor_u1)
DECLARE_QS8_VLRELU_UKERNEL_FUNCTION(xnn_qs8_vlrelu_ukernel__scalar_andxor_u2)
DECLARE_QS8_VLRELU_UKERNEL_FUNCTION(xnn_qs8_vlrelu_ukernel__scalar_andxor_u4)

DECLARE_QS8_VLRELU_UKERNEL_FUNCTION(xnn_qs8_vlrelu_ukernel__scalar_select_u1)
DECLARE_QS8_VLRELU_UKERNEL_FUNCTION(xnn_qs8_vlrelu_ukernel__scalar_select_u2)
DECLARE_QS8_VLRELU_UKERNEL_FUNCTION(xnn_qs8_vlrelu_ukernel__scalar_select_u4)


#define DECLARE_QU8_VLRELU_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                         \
      size_t n,                                      \
      const uint8_t* input,                          \
      uint8_t* output,                               \
      const union xnn_qu8_lrelu_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

DECLARE_QU8_VLRELU_UKERNEL_FUNCTION(xnn_qu8_vlrelu_ukernel__neon_u8)
DECLARE_QU8_VLRELU_UKERNEL_FUNCTION(xnn_qu8_vlrelu_ukernel__neon_u16)
DECLARE_QU8_VLRELU_UKERNEL_FUNCTION(xnn_qu8_vlrelu_ukernel__neon_u32)

DECLARE_QU8_VLRELU_UKERNEL_FUNCTION(xnn_qu8_vlrelu_ukernel__sse2_u16)
DECLARE_QU8_VLRELU_UKERNEL_FUNCTION(xnn_qu8_vlrelu_ukernel__sse2_u32)

DECLARE_QU8_VLRELU_UKERNEL_FUNCTION(xnn_qu8_vlrelu_ukernel__ssse3_u16)
DECLARE_QU8_VLRELU_UKERNEL_FUNCTION(xnn_qu8_vlrelu_ukernel__ssse3_u32)

DECLARE_QU8_VLRELU_UKERNEL_FUNCTION(xnn_qu8_vlrelu_ukernel__sse41_u8)
DECLARE_QU8_VLRELU_UKERNEL_FUNCTION(xnn_qu8_vlrelu_ukernel__sse41_u16)
DECLARE_QU8_VLRELU_UKERNEL_FUNCTION(xnn_qu8_vlrelu_ukernel__sse41_u32)

DECLARE_QU8_VLRELU_UKERNEL_FUNCTION(xnn_qu8_vlrelu_ukernel__avx_u8)
DECLARE_QU8_VLRELU_UKERNEL_FUNCTION(xnn_qu8_vlrelu_ukernel__avx_u16)
DECLARE_QU8_VLRELU_UKERNEL_FUNCTION(xnn_qu8_vlrelu_ukernel__avx_u32)

DECLARE_QU8_VLRELU_UKERNEL_FUNCTION(xnn_qu8_vlrelu_ukernel__avx2_u16)
DECLARE_QU8_VLRELU_UKERNEL_FUNCTION(xnn_qu8_vlrelu_ukernel__avx2_u32)
DECLARE_QU8_VLRELU_UKERNEL_FUNCTION(xnn_qu8_vlrelu_ukernel__avx2_u64)

DECLARE_QU8_VLRELU_UKERNEL_FUNCTION(xnn_qu8_vlrelu_ukernel__wasmsimd_arm_u16)
DECLARE_QU8_VLRELU_UKERNEL_FUNCTION(xnn_qu8_vlrelu_ukernel__wasmsimd_arm_u32)

DECLARE_QU8_VLRELU_UKERNEL_FUNCTION(xnn_qu8_vlrelu_ukernel__wasmsimd_x86_u8)
DECLARE_QU8_VLRELU_UKERNEL_FUNCTION(xnn_qu8_vlrelu_ukernel__wasmsimd_x86_u16)
DECLARE_QU8_VLRELU_UKERNEL_FUNCTION(xnn_qu8_vlrelu_ukernel__wasmsimd_x86_u32)

DECLARE_QU8_VLRELU_UKERNEL_FUNCTION(xnn_qu8_vlrelu_ukernel__wasmrelaxedsimd_arm_u16)
DECLARE_QU8_VLRELU_UKERNEL_FUNCTION(xnn_qu8_vlrelu_ukernel__wasmrelaxedsimd_arm_u32)

DECLARE_QU8_VLRELU_UKERNEL_FUNCTION(xnn_qu8_vlrelu_ukernel__wasmrelaxedsimd_x86_u8)
DECLARE_QU8_VLRELU_UKERNEL_FUNCTION(xnn_qu8_vlrelu_ukernel__wasmrelaxedsimd_x86_u16)
DECLARE_QU8_VLRELU_UKERNEL_FUNCTION(xnn_qu8_vlrelu_ukernel__wasmrelaxedsimd_x86_u32)

DECLARE_QU8_VLRELU_UKERNEL_FUNCTION(xnn_qu8_vlrelu_ukernel__armsimd32_u4)
DECLARE_QU8_VLRELU_UKERNEL_FUNCTION(xnn_qu8_vlrelu_ukernel__armsimd32_u8)

DECLARE_QU8_VLRELU_UKERNEL_FUNCTION(xnn_qu8_vlrelu_ukernel__scalar_andxor_u1)
DECLARE_QU8_VLRELU_UKERNEL_FUNCTION(xnn_qu8_vlrelu_ukernel__scalar_andxor_u2)
DECLARE_QU8_VLRELU_UKERNEL_FUNCTION(xnn_qu8_vlrelu_ukernel__scalar_andxor_u4)

DECLARE_QU8_VLRELU_UKERNEL_FUNCTION(xnn_qu8_vlrelu_ukernel__scalar_select_u1)
DECLARE_QU8_VLRELU_UKERNEL_FUNCTION(xnn_qu8_vlrelu_ukernel__scalar_select_u2)
DECLARE_QU8_VLRELU_UKERNEL_FUNCTION(xnn_qu8_vlrelu_ukernel__scalar_select_u4)


#ifdef __cplusplus
}  // extern "C"
#endif
