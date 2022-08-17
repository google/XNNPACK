// Copyright 2022 Google LLC
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


#define DECLARE_QS8_VLRELU_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                         \
      size_t n,                                      \
      const int8_t* input,                           \
      int8_t* output,                                \
      const union xnn_qs8_lrelu_params* params);

DECLARE_QS8_VLRELU_UKERNEL_FUNCTION(xnn_qs8_vlrelu_ukernel__neon_x8)
DECLARE_QS8_VLRELU_UKERNEL_FUNCTION(xnn_qs8_vlrelu_ukernel__neon_x16)
DECLARE_QS8_VLRELU_UKERNEL_FUNCTION(xnn_qs8_vlrelu_ukernel__neon_x32)

DECLARE_QS8_VLRELU_UKERNEL_FUNCTION(xnn_qs8_vlrelu_ukernel__sse2_x16)
DECLARE_QS8_VLRELU_UKERNEL_FUNCTION(xnn_qs8_vlrelu_ukernel__sse2_x32)

DECLARE_QS8_VLRELU_UKERNEL_FUNCTION(xnn_qs8_vlrelu_ukernel__ssse3_x16)
DECLARE_QS8_VLRELU_UKERNEL_FUNCTION(xnn_qs8_vlrelu_ukernel__ssse3_x32)

DECLARE_QS8_VLRELU_UKERNEL_FUNCTION(xnn_qs8_vlrelu_ukernel__sse41_x8)
DECLARE_QS8_VLRELU_UKERNEL_FUNCTION(xnn_qs8_vlrelu_ukernel__sse41_x16)
DECLARE_QS8_VLRELU_UKERNEL_FUNCTION(xnn_qs8_vlrelu_ukernel__sse41_x32)

DECLARE_QS8_VLRELU_UKERNEL_FUNCTION(xnn_qs8_vlrelu_ukernel__avx_x8)
DECLARE_QS8_VLRELU_UKERNEL_FUNCTION(xnn_qs8_vlrelu_ukernel__avx_x16)
DECLARE_QS8_VLRELU_UKERNEL_FUNCTION(xnn_qs8_vlrelu_ukernel__avx_x32)

DECLARE_QS8_VLRELU_UKERNEL_FUNCTION(xnn_qs8_vlrelu_ukernel__avx2_x16)
DECLARE_QS8_VLRELU_UKERNEL_FUNCTION(xnn_qs8_vlrelu_ukernel__avx2_x32)
DECLARE_QS8_VLRELU_UKERNEL_FUNCTION(xnn_qs8_vlrelu_ukernel__avx2_x64)

DECLARE_QS8_VLRELU_UKERNEL_FUNCTION(xnn_qs8_vlrelu_ukernel__wasmsimd_arm_x16)
DECLARE_QS8_VLRELU_UKERNEL_FUNCTION(xnn_qs8_vlrelu_ukernel__wasmsimd_arm_x32)

DECLARE_QS8_VLRELU_UKERNEL_FUNCTION(xnn_qs8_vlrelu_ukernel__wasmsimd_x86_x8)
DECLARE_QS8_VLRELU_UKERNEL_FUNCTION(xnn_qs8_vlrelu_ukernel__wasmsimd_x86_x16)
DECLARE_QS8_VLRELU_UKERNEL_FUNCTION(xnn_qs8_vlrelu_ukernel__wasmsimd_x86_x32)

DECLARE_QS8_VLRELU_UKERNEL_FUNCTION(xnn_qs8_vlrelu_ukernel__wasmrelaxedsimd_arm_x16)
DECLARE_QS8_VLRELU_UKERNEL_FUNCTION(xnn_qs8_vlrelu_ukernel__wasmrelaxedsimd_arm_x32)

DECLARE_QS8_VLRELU_UKERNEL_FUNCTION(xnn_qs8_vlrelu_ukernel__wasmrelaxedsimd_x86_x8)
DECLARE_QS8_VLRELU_UKERNEL_FUNCTION(xnn_qs8_vlrelu_ukernel__wasmrelaxedsimd_x86_x16)
DECLARE_QS8_VLRELU_UKERNEL_FUNCTION(xnn_qs8_vlrelu_ukernel__wasmrelaxedsimd_x86_x32)

DECLARE_QS8_VLRELU_UKERNEL_FUNCTION(xnn_qs8_vlrelu_ukernel__armsimd32_x4)
DECLARE_QS8_VLRELU_UKERNEL_FUNCTION(xnn_qs8_vlrelu_ukernel__armsimd32_x8)

DECLARE_QS8_VLRELU_UKERNEL_FUNCTION(xnn_qs8_vlrelu_ukernel__scalar_andxor_x1)
DECLARE_QS8_VLRELU_UKERNEL_FUNCTION(xnn_qs8_vlrelu_ukernel__scalar_andxor_x2)
DECLARE_QS8_VLRELU_UKERNEL_FUNCTION(xnn_qs8_vlrelu_ukernel__scalar_andxor_x4)

DECLARE_QS8_VLRELU_UKERNEL_FUNCTION(xnn_qs8_vlrelu_ukernel__scalar_select_x1)
DECLARE_QS8_VLRELU_UKERNEL_FUNCTION(xnn_qs8_vlrelu_ukernel__scalar_select_x2)
DECLARE_QS8_VLRELU_UKERNEL_FUNCTION(xnn_qs8_vlrelu_ukernel__scalar_select_x4)


#define DECLARE_QU8_VLRELU_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                         \
      size_t n,                                      \
      const uint8_t* input,                          \
      uint8_t* output,                               \
      const union xnn_qu8_lrelu_params* params);

DECLARE_QU8_VLRELU_UKERNEL_FUNCTION(xnn_qu8_vlrelu_ukernel__neon_x8)
DECLARE_QU8_VLRELU_UKERNEL_FUNCTION(xnn_qu8_vlrelu_ukernel__neon_x16)
DECLARE_QU8_VLRELU_UKERNEL_FUNCTION(xnn_qu8_vlrelu_ukernel__neon_x32)

DECLARE_QU8_VLRELU_UKERNEL_FUNCTION(xnn_qu8_vlrelu_ukernel__sse2_x16)
DECLARE_QU8_VLRELU_UKERNEL_FUNCTION(xnn_qu8_vlrelu_ukernel__sse2_x32)

DECLARE_QU8_VLRELU_UKERNEL_FUNCTION(xnn_qu8_vlrelu_ukernel__ssse3_x16)
DECLARE_QU8_VLRELU_UKERNEL_FUNCTION(xnn_qu8_vlrelu_ukernel__ssse3_x32)

DECLARE_QU8_VLRELU_UKERNEL_FUNCTION(xnn_qu8_vlrelu_ukernel__sse41_x8)
DECLARE_QU8_VLRELU_UKERNEL_FUNCTION(xnn_qu8_vlrelu_ukernel__sse41_x16)
DECLARE_QU8_VLRELU_UKERNEL_FUNCTION(xnn_qu8_vlrelu_ukernel__sse41_x32)

DECLARE_QU8_VLRELU_UKERNEL_FUNCTION(xnn_qu8_vlrelu_ukernel__avx_x8)
DECLARE_QU8_VLRELU_UKERNEL_FUNCTION(xnn_qu8_vlrelu_ukernel__avx_x16)
DECLARE_QU8_VLRELU_UKERNEL_FUNCTION(xnn_qu8_vlrelu_ukernel__avx_x32)

DECLARE_QU8_VLRELU_UKERNEL_FUNCTION(xnn_qu8_vlrelu_ukernel__avx2_x16)
DECLARE_QU8_VLRELU_UKERNEL_FUNCTION(xnn_qu8_vlrelu_ukernel__avx2_x32)
DECLARE_QU8_VLRELU_UKERNEL_FUNCTION(xnn_qu8_vlrelu_ukernel__avx2_x64)

DECLARE_QU8_VLRELU_UKERNEL_FUNCTION(xnn_qu8_vlrelu_ukernel__wasmsimd_arm_x16)
DECLARE_QU8_VLRELU_UKERNEL_FUNCTION(xnn_qu8_vlrelu_ukernel__wasmsimd_arm_x32)

DECLARE_QU8_VLRELU_UKERNEL_FUNCTION(xnn_qu8_vlrelu_ukernel__wasmsimd_x86_x8)
DECLARE_QU8_VLRELU_UKERNEL_FUNCTION(xnn_qu8_vlrelu_ukernel__wasmsimd_x86_x16)
DECLARE_QU8_VLRELU_UKERNEL_FUNCTION(xnn_qu8_vlrelu_ukernel__wasmsimd_x86_x32)

DECLARE_QU8_VLRELU_UKERNEL_FUNCTION(xnn_qu8_vlrelu_ukernel__wasmrelaxedsimd_arm_x16)
DECLARE_QU8_VLRELU_UKERNEL_FUNCTION(xnn_qu8_vlrelu_ukernel__wasmrelaxedsimd_arm_x32)

DECLARE_QU8_VLRELU_UKERNEL_FUNCTION(xnn_qu8_vlrelu_ukernel__wasmrelaxedsimd_x86_x8)
DECLARE_QU8_VLRELU_UKERNEL_FUNCTION(xnn_qu8_vlrelu_ukernel__wasmrelaxedsimd_x86_x16)
DECLARE_QU8_VLRELU_UKERNEL_FUNCTION(xnn_qu8_vlrelu_ukernel__wasmrelaxedsimd_x86_x32)

DECLARE_QU8_VLRELU_UKERNEL_FUNCTION(xnn_qu8_vlrelu_ukernel__armsimd32_x4)
DECLARE_QU8_VLRELU_UKERNEL_FUNCTION(xnn_qu8_vlrelu_ukernel__armsimd32_x8)

DECLARE_QU8_VLRELU_UKERNEL_FUNCTION(xnn_qu8_vlrelu_ukernel__scalar_andxor_x1)
DECLARE_QU8_VLRELU_UKERNEL_FUNCTION(xnn_qu8_vlrelu_ukernel__scalar_andxor_x2)
DECLARE_QU8_VLRELU_UKERNEL_FUNCTION(xnn_qu8_vlrelu_ukernel__scalar_andxor_x4)

DECLARE_QU8_VLRELU_UKERNEL_FUNCTION(xnn_qu8_vlrelu_ukernel__scalar_select_x1)
DECLARE_QU8_VLRELU_UKERNEL_FUNCTION(xnn_qu8_vlrelu_ukernel__scalar_select_x2)
DECLARE_QU8_VLRELU_UKERNEL_FUNCTION(xnn_qu8_vlrelu_ukernel__scalar_select_x4)


#ifdef __cplusplus
}  // extern "C"
#endif
