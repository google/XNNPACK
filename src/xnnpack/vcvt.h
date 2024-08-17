// Copyright 2021 Google LLC
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


#define DECLARE_F16_F32_VCVT_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                           \
      size_t n,                                        \
      const void* input,                               \
      float* output,                                   \
      const void* params);

DECLARE_F16_F32_VCVT_UKERNEL_FUNCTION(xnn_f16_f32_vcvt_ukernel__neon_int16_u8)
DECLARE_F16_F32_VCVT_UKERNEL_FUNCTION(xnn_f16_f32_vcvt_ukernel__neon_int16_u16)
DECLARE_F16_F32_VCVT_UKERNEL_FUNCTION(xnn_f16_f32_vcvt_ukernel__neon_int16_u24)
DECLARE_F16_F32_VCVT_UKERNEL_FUNCTION(xnn_f16_f32_vcvt_ukernel__neon_int16_u32)

DECLARE_F16_F32_VCVT_UKERNEL_FUNCTION(xnn_f16_f32_vcvt_ukernel__neon_int32_u8)
DECLARE_F16_F32_VCVT_UKERNEL_FUNCTION(xnn_f16_f32_vcvt_ukernel__neon_int32_u16)
DECLARE_F16_F32_VCVT_UKERNEL_FUNCTION(xnn_f16_f32_vcvt_ukernel__neon_int32_u24)
DECLARE_F16_F32_VCVT_UKERNEL_FUNCTION(xnn_f16_f32_vcvt_ukernel__neon_int32_u32)

DECLARE_F16_F32_VCVT_UKERNEL_FUNCTION(xnn_f16_f32_vcvt_ukernel__neonfp16_u8)
DECLARE_F16_F32_VCVT_UKERNEL_FUNCTION(xnn_f16_f32_vcvt_ukernel__neonfp16_u16)

DECLARE_F16_F32_VCVT_UKERNEL_FUNCTION(xnn_f16_f32_vcvt_ukernel__sse2_int16_u8)
DECLARE_F16_F32_VCVT_UKERNEL_FUNCTION(xnn_f16_f32_vcvt_ukernel__sse2_int16_u16)
DECLARE_F16_F32_VCVT_UKERNEL_FUNCTION(xnn_f16_f32_vcvt_ukernel__sse2_int16_u24)
DECLARE_F16_F32_VCVT_UKERNEL_FUNCTION(xnn_f16_f32_vcvt_ukernel__sse2_int16_u32)

DECLARE_F16_F32_VCVT_UKERNEL_FUNCTION(xnn_f16_f32_vcvt_ukernel__sse2_int32_u8)
DECLARE_F16_F32_VCVT_UKERNEL_FUNCTION(xnn_f16_f32_vcvt_ukernel__sse2_int32_u16)
DECLARE_F16_F32_VCVT_UKERNEL_FUNCTION(xnn_f16_f32_vcvt_ukernel__sse2_int32_u24)
DECLARE_F16_F32_VCVT_UKERNEL_FUNCTION(xnn_f16_f32_vcvt_ukernel__sse2_int32_u32)

DECLARE_F16_F32_VCVT_UKERNEL_FUNCTION(xnn_f16_f32_vcvt_ukernel__sse41_int16_u8)
DECLARE_F16_F32_VCVT_UKERNEL_FUNCTION(xnn_f16_f32_vcvt_ukernel__sse41_int16_u16)
DECLARE_F16_F32_VCVT_UKERNEL_FUNCTION(xnn_f16_f32_vcvt_ukernel__sse41_int16_u24)
DECLARE_F16_F32_VCVT_UKERNEL_FUNCTION(xnn_f16_f32_vcvt_ukernel__sse41_int16_u32)

DECLARE_F16_F32_VCVT_UKERNEL_FUNCTION(xnn_f16_f32_vcvt_ukernel__sse41_int32_u8)
DECLARE_F16_F32_VCVT_UKERNEL_FUNCTION(xnn_f16_f32_vcvt_ukernel__sse41_int32_u16)
DECLARE_F16_F32_VCVT_UKERNEL_FUNCTION(xnn_f16_f32_vcvt_ukernel__sse41_int32_u24)
DECLARE_F16_F32_VCVT_UKERNEL_FUNCTION(xnn_f16_f32_vcvt_ukernel__sse41_int32_u32)

DECLARE_F16_F32_VCVT_UKERNEL_FUNCTION(xnn_f16_f32_vcvt_ukernel__avx_int16_u8)
DECLARE_F16_F32_VCVT_UKERNEL_FUNCTION(xnn_f16_f32_vcvt_ukernel__avx_int16_u16)
DECLARE_F16_F32_VCVT_UKERNEL_FUNCTION(xnn_f16_f32_vcvt_ukernel__avx_int16_u24)
DECLARE_F16_F32_VCVT_UKERNEL_FUNCTION(xnn_f16_f32_vcvt_ukernel__avx_int16_u32)

DECLARE_F16_F32_VCVT_UKERNEL_FUNCTION(xnn_f16_f32_vcvt_ukernel__avx_int32_u8)
DECLARE_F16_F32_VCVT_UKERNEL_FUNCTION(xnn_f16_f32_vcvt_ukernel__avx_int32_u16)
DECLARE_F16_F32_VCVT_UKERNEL_FUNCTION(xnn_f16_f32_vcvt_ukernel__avx_int32_u24)
DECLARE_F16_F32_VCVT_UKERNEL_FUNCTION(xnn_f16_f32_vcvt_ukernel__avx_int32_u32)

DECLARE_F16_F32_VCVT_UKERNEL_FUNCTION(xnn_f16_f32_vcvt_ukernel__f16c_u8)
DECLARE_F16_F32_VCVT_UKERNEL_FUNCTION(xnn_f16_f32_vcvt_ukernel__f16c_u16)

DECLARE_F16_F32_VCVT_UKERNEL_FUNCTION(xnn_f16_f32_vcvt_ukernel__avx512skx_u16)
DECLARE_F16_F32_VCVT_UKERNEL_FUNCTION(xnn_f16_f32_vcvt_ukernel__avx512skx_u32)

DECLARE_F16_F32_VCVT_UKERNEL_FUNCTION(xnn_f16_f32_vcvt_ukernel__wasmsimd_int16_u8)
DECLARE_F16_F32_VCVT_UKERNEL_FUNCTION(xnn_f16_f32_vcvt_ukernel__wasmsimd_int16_u16)
DECLARE_F16_F32_VCVT_UKERNEL_FUNCTION(xnn_f16_f32_vcvt_ukernel__wasmsimd_int16_u24)
DECLARE_F16_F32_VCVT_UKERNEL_FUNCTION(xnn_f16_f32_vcvt_ukernel__wasmsimd_int16_u32)

DECLARE_F16_F32_VCVT_UKERNEL_FUNCTION(xnn_f16_f32_vcvt_ukernel__wasmsimd_int32_u8)
DECLARE_F16_F32_VCVT_UKERNEL_FUNCTION(xnn_f16_f32_vcvt_ukernel__wasmsimd_int32_u16)
DECLARE_F16_F32_VCVT_UKERNEL_FUNCTION(xnn_f16_f32_vcvt_ukernel__wasmsimd_int32_u24)
DECLARE_F16_F32_VCVT_UKERNEL_FUNCTION(xnn_f16_f32_vcvt_ukernel__wasmsimd_int32_u32)

DECLARE_F16_F32_VCVT_UKERNEL_FUNCTION(xnn_f16_f32_vcvt_ukernel__wasmrelaxedsimd_int16_u8)
DECLARE_F16_F32_VCVT_UKERNEL_FUNCTION(xnn_f16_f32_vcvt_ukernel__wasmrelaxedsimd_int16_u16)
DECLARE_F16_F32_VCVT_UKERNEL_FUNCTION(xnn_f16_f32_vcvt_ukernel__wasmrelaxedsimd_int16_u24)
DECLARE_F16_F32_VCVT_UKERNEL_FUNCTION(xnn_f16_f32_vcvt_ukernel__wasmrelaxedsimd_int16_u32)

DECLARE_F16_F32_VCVT_UKERNEL_FUNCTION(xnn_f16_f32_vcvt_ukernel__wasmrelaxedsimd_int32_u8)
DECLARE_F16_F32_VCVT_UKERNEL_FUNCTION(xnn_f16_f32_vcvt_ukernel__wasmrelaxedsimd_int32_u16)
DECLARE_F16_F32_VCVT_UKERNEL_FUNCTION(xnn_f16_f32_vcvt_ukernel__wasmrelaxedsimd_int32_u24)
DECLARE_F16_F32_VCVT_UKERNEL_FUNCTION(xnn_f16_f32_vcvt_ukernel__wasmrelaxedsimd_int32_u32)

DECLARE_F16_F32_VCVT_UKERNEL_FUNCTION(xnn_f16_f32_vcvt_ukernel__scalar_u1)
DECLARE_F16_F32_VCVT_UKERNEL_FUNCTION(xnn_f16_f32_vcvt_ukernel__scalar_u2)
DECLARE_F16_F32_VCVT_UKERNEL_FUNCTION(xnn_f16_f32_vcvt_ukernel__scalar_u3)
DECLARE_F16_F32_VCVT_UKERNEL_FUNCTION(xnn_f16_f32_vcvt_ukernel__scalar_u4)


#define DECLARE_F32_F16_VCVT_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                           \
      size_t n,                                        \
      const float* input,                              \
      void* output,                                    \
      const void* params);

DECLARE_F32_F16_VCVT_UKERNEL_FUNCTION(xnn_f32_f16_vcvt_ukernel__neon_u8)
DECLARE_F32_F16_VCVT_UKERNEL_FUNCTION(xnn_f32_f16_vcvt_ukernel__neon_u16)
DECLARE_F32_F16_VCVT_UKERNEL_FUNCTION(xnn_f32_f16_vcvt_ukernel__neon_u24)
DECLARE_F32_F16_VCVT_UKERNEL_FUNCTION(xnn_f32_f16_vcvt_ukernel__neon_u32)

DECLARE_F32_F16_VCVT_UKERNEL_FUNCTION(xnn_f32_f16_vcvt_ukernel__neonfp16_u8)
DECLARE_F32_F16_VCVT_UKERNEL_FUNCTION(xnn_f32_f16_vcvt_ukernel__neonfp16_u16)

DECLARE_F32_F16_VCVT_UKERNEL_FUNCTION(xnn_f32_f16_vcvt_ukernel__sse2_u8)
DECLARE_F32_F16_VCVT_UKERNEL_FUNCTION(xnn_f32_f16_vcvt_ukernel__sse2_u16)
DECLARE_F32_F16_VCVT_UKERNEL_FUNCTION(xnn_f32_f16_vcvt_ukernel__sse2_u24)
DECLARE_F32_F16_VCVT_UKERNEL_FUNCTION(xnn_f32_f16_vcvt_ukernel__sse2_u32)

DECLARE_F32_F16_VCVT_UKERNEL_FUNCTION(xnn_f32_f16_vcvt_ukernel__sse41_u8)
DECLARE_F32_F16_VCVT_UKERNEL_FUNCTION(xnn_f32_f16_vcvt_ukernel__sse41_u16)
DECLARE_F32_F16_VCVT_UKERNEL_FUNCTION(xnn_f32_f16_vcvt_ukernel__sse41_u24)
DECLARE_F32_F16_VCVT_UKERNEL_FUNCTION(xnn_f32_f16_vcvt_ukernel__sse41_u32)

DECLARE_F32_F16_VCVT_UKERNEL_FUNCTION(xnn_f32_f16_vcvt_ukernel__avx_u8)
DECLARE_F32_F16_VCVT_UKERNEL_FUNCTION(xnn_f32_f16_vcvt_ukernel__avx_u16)
DECLARE_F32_F16_VCVT_UKERNEL_FUNCTION(xnn_f32_f16_vcvt_ukernel__avx_u24)
DECLARE_F32_F16_VCVT_UKERNEL_FUNCTION(xnn_f32_f16_vcvt_ukernel__avx_u32)

DECLARE_F32_F16_VCVT_UKERNEL_FUNCTION(xnn_f32_f16_vcvt_ukernel__f16c_u8)
DECLARE_F32_F16_VCVT_UKERNEL_FUNCTION(xnn_f32_f16_vcvt_ukernel__f16c_u16)

DECLARE_F32_F16_VCVT_UKERNEL_FUNCTION(xnn_f32_f16_vcvt_ukernel__avx512skx_u16)
DECLARE_F32_F16_VCVT_UKERNEL_FUNCTION(xnn_f32_f16_vcvt_ukernel__avx512skx_u32)

DECLARE_F32_F16_VCVT_UKERNEL_FUNCTION(xnn_f32_f16_vcvt_ukernel__wasmsimd_u8)
DECLARE_F32_F16_VCVT_UKERNEL_FUNCTION(xnn_f32_f16_vcvt_ukernel__wasmsimd_u16)
DECLARE_F32_F16_VCVT_UKERNEL_FUNCTION(xnn_f32_f16_vcvt_ukernel__wasmsimd_u24)
DECLARE_F32_F16_VCVT_UKERNEL_FUNCTION(xnn_f32_f16_vcvt_ukernel__wasmsimd_u32)

DECLARE_F32_F16_VCVT_UKERNEL_FUNCTION(xnn_f32_f16_vcvt_ukernel__wasmrelaxedsimd_u8)
DECLARE_F32_F16_VCVT_UKERNEL_FUNCTION(xnn_f32_f16_vcvt_ukernel__wasmrelaxedsimd_u16)
DECLARE_F32_F16_VCVT_UKERNEL_FUNCTION(xnn_f32_f16_vcvt_ukernel__wasmrelaxedsimd_u24)
DECLARE_F32_F16_VCVT_UKERNEL_FUNCTION(xnn_f32_f16_vcvt_ukernel__wasmrelaxedsimd_u32)

DECLARE_F32_F16_VCVT_UKERNEL_FUNCTION(xnn_f32_f16_vcvt_ukernel__scalar_bitcast_u1)
DECLARE_F32_F16_VCVT_UKERNEL_FUNCTION(xnn_f32_f16_vcvt_ukernel__scalar_bitcast_u2)
DECLARE_F32_F16_VCVT_UKERNEL_FUNCTION(xnn_f32_f16_vcvt_ukernel__scalar_bitcast_u3)
DECLARE_F32_F16_VCVT_UKERNEL_FUNCTION(xnn_f32_f16_vcvt_ukernel__scalar_bitcast_u4)

DECLARE_F32_F16_VCVT_UKERNEL_FUNCTION(xnn_f32_f16_vcvt_ukernel__scalar_fabsf_u1)
DECLARE_F32_F16_VCVT_UKERNEL_FUNCTION(xnn_f32_f16_vcvt_ukernel__scalar_fabsf_u2)
DECLARE_F32_F16_VCVT_UKERNEL_FUNCTION(xnn_f32_f16_vcvt_ukernel__scalar_fabsf_u3)
DECLARE_F32_F16_VCVT_UKERNEL_FUNCTION(xnn_f32_f16_vcvt_ukernel__scalar_fabsf_u4)


#define DECLARE_F16_QS8_VCVT_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                           \
      size_t n,                                        \
      const void* input,                               \
      int8_t* output,                                  \
      const union xnn_f16_qs8_cvt_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

DECLARE_F16_QS8_VCVT_UKERNEL_FUNCTION(xnn_f16_qs8_vcvt_ukernel__neonfp16arith_u8)
DECLARE_F16_QS8_VCVT_UKERNEL_FUNCTION(xnn_f16_qs8_vcvt_ukernel__neonfp16arith_u16)
DECLARE_F16_QS8_VCVT_UKERNEL_FUNCTION(xnn_f16_qs8_vcvt_ukernel__neonfp16arith_u24)
DECLARE_F16_QS8_VCVT_UKERNEL_FUNCTION(xnn_f16_qs8_vcvt_ukernel__neonfp16arith_u32)
DECLARE_F16_QS8_VCVT_UKERNEL_FUNCTION(xnn_f16_qs8_vcvt_ukernel__neonfp16arith_u64)

DECLARE_F16_QS8_VCVT_UKERNEL_FUNCTION(xnn_f16_qs8_vcvt_ukernel__scalar_fmagic_u1)
DECLARE_F16_QS8_VCVT_UKERNEL_FUNCTION(xnn_f16_qs8_vcvt_ukernel__scalar_fmagic_u2)
DECLARE_F16_QS8_VCVT_UKERNEL_FUNCTION(xnn_f16_qs8_vcvt_ukernel__scalar_fmagic_u3)
DECLARE_F16_QS8_VCVT_UKERNEL_FUNCTION(xnn_f16_qs8_vcvt_ukernel__scalar_fmagic_u4)

DECLARE_F16_QS8_VCVT_UKERNEL_FUNCTION(xnn_f16_qs8_vcvt_ukernel__scalar_imagic_u1)
DECLARE_F16_QS8_VCVT_UKERNEL_FUNCTION(xnn_f16_qs8_vcvt_ukernel__scalar_imagic_u2)
DECLARE_F16_QS8_VCVT_UKERNEL_FUNCTION(xnn_f16_qs8_vcvt_ukernel__scalar_imagic_u3)
DECLARE_F16_QS8_VCVT_UKERNEL_FUNCTION(xnn_f16_qs8_vcvt_ukernel__scalar_imagic_u4)

#define DECLARE_F32_QS8_VCVT_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                           \
      size_t n,                                        \
      const float* input,                              \
      int8_t* output,                                  \
      const union xnn_f32_qs8_cvt_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

DECLARE_F32_QS8_VCVT_UKERNEL_FUNCTION(xnn_f32_qs8_vcvt_ukernel__neon_u8)
DECLARE_F32_QS8_VCVT_UKERNEL_FUNCTION(xnn_f32_qs8_vcvt_ukernel__neon_u16)
DECLARE_F32_QS8_VCVT_UKERNEL_FUNCTION(xnn_f32_qs8_vcvt_ukernel__neon_u24)
DECLARE_F32_QS8_VCVT_UKERNEL_FUNCTION(xnn_f32_qs8_vcvt_ukernel__neon_u32)

DECLARE_F32_QS8_VCVT_UKERNEL_FUNCTION(xnn_f32_qs8_vcvt_ukernel__neonv8_u8)
DECLARE_F32_QS8_VCVT_UKERNEL_FUNCTION(xnn_f32_qs8_vcvt_ukernel__neonv8_u16)
DECLARE_F32_QS8_VCVT_UKERNEL_FUNCTION(xnn_f32_qs8_vcvt_ukernel__neonv8_u24)
DECLARE_F32_QS8_VCVT_UKERNEL_FUNCTION(xnn_f32_qs8_vcvt_ukernel__neonv8_u32)

DECLARE_F32_QS8_VCVT_UKERNEL_FUNCTION(xnn_f32_qs8_vcvt_ukernel__sse2_u8)
DECLARE_F32_QS8_VCVT_UKERNEL_FUNCTION(xnn_f32_qs8_vcvt_ukernel__sse2_u16)
DECLARE_F32_QS8_VCVT_UKERNEL_FUNCTION(xnn_f32_qs8_vcvt_ukernel__sse2_u24)
DECLARE_F32_QS8_VCVT_UKERNEL_FUNCTION(xnn_f32_qs8_vcvt_ukernel__sse2_u32)

DECLARE_F32_QS8_VCVT_UKERNEL_FUNCTION(xnn_f32_qs8_vcvt_ukernel__sse41_u8)
DECLARE_F32_QS8_VCVT_UKERNEL_FUNCTION(xnn_f32_qs8_vcvt_ukernel__sse41_u16)
DECLARE_F32_QS8_VCVT_UKERNEL_FUNCTION(xnn_f32_qs8_vcvt_ukernel__sse41_u24)
DECLARE_F32_QS8_VCVT_UKERNEL_FUNCTION(xnn_f32_qs8_vcvt_ukernel__sse41_u32)

DECLARE_F32_QS8_VCVT_UKERNEL_FUNCTION(xnn_f32_qs8_vcvt_ukernel__avx_u8)
DECLARE_F32_QS8_VCVT_UKERNEL_FUNCTION(xnn_f32_qs8_vcvt_ukernel__avx_u16)
DECLARE_F32_QS8_VCVT_UKERNEL_FUNCTION(xnn_f32_qs8_vcvt_ukernel__avx_u24)
DECLARE_F32_QS8_VCVT_UKERNEL_FUNCTION(xnn_f32_qs8_vcvt_ukernel__avx_u32)

DECLARE_F32_QS8_VCVT_UKERNEL_FUNCTION(xnn_f32_qs8_vcvt_ukernel__avx2_u16)
DECLARE_F32_QS8_VCVT_UKERNEL_FUNCTION(xnn_f32_qs8_vcvt_ukernel__avx2_u32)
DECLARE_F32_QS8_VCVT_UKERNEL_FUNCTION(xnn_f32_qs8_vcvt_ukernel__avx2_u48)
DECLARE_F32_QS8_VCVT_UKERNEL_FUNCTION(xnn_f32_qs8_vcvt_ukernel__avx2_u64)

DECLARE_F32_QS8_VCVT_UKERNEL_FUNCTION(xnn_f32_qs8_vcvt_ukernel__avx512skx_u32)
DECLARE_F32_QS8_VCVT_UKERNEL_FUNCTION(xnn_f32_qs8_vcvt_ukernel__avx512skx_u64)
DECLARE_F32_QS8_VCVT_UKERNEL_FUNCTION(xnn_f32_qs8_vcvt_ukernel__avx512skx_u96)
DECLARE_F32_QS8_VCVT_UKERNEL_FUNCTION(xnn_f32_qs8_vcvt_ukernel__avx512skx_u128)

DECLARE_F32_QS8_VCVT_UKERNEL_FUNCTION(xnn_f32_qs8_vcvt_ukernel__wasmsimd_cvt_u8)
DECLARE_F32_QS8_VCVT_UKERNEL_FUNCTION(xnn_f32_qs8_vcvt_ukernel__wasmsimd_cvt_u16)
DECLARE_F32_QS8_VCVT_UKERNEL_FUNCTION(xnn_f32_qs8_vcvt_ukernel__wasmsimd_cvt_u24)
DECLARE_F32_QS8_VCVT_UKERNEL_FUNCTION(xnn_f32_qs8_vcvt_ukernel__wasmsimd_cvt_u32)

DECLARE_F32_QS8_VCVT_UKERNEL_FUNCTION(xnn_f32_qs8_vcvt_ukernel__wasmsimd_magic_u8)
DECLARE_F32_QS8_VCVT_UKERNEL_FUNCTION(xnn_f32_qs8_vcvt_ukernel__wasmsimd_magic_u16)
DECLARE_F32_QS8_VCVT_UKERNEL_FUNCTION(xnn_f32_qs8_vcvt_ukernel__wasmsimd_magic_u24)
DECLARE_F32_QS8_VCVT_UKERNEL_FUNCTION(xnn_f32_qs8_vcvt_ukernel__wasmsimd_magic_u32)

DECLARE_F32_QS8_VCVT_UKERNEL_FUNCTION(xnn_f32_qs8_vcvt_ukernel__wasm_fmagic_u1)
DECLARE_F32_QS8_VCVT_UKERNEL_FUNCTION(xnn_f32_qs8_vcvt_ukernel__wasm_fmagic_u2)
DECLARE_F32_QS8_VCVT_UKERNEL_FUNCTION(xnn_f32_qs8_vcvt_ukernel__wasm_fmagic_u3)
DECLARE_F32_QS8_VCVT_UKERNEL_FUNCTION(xnn_f32_qs8_vcvt_ukernel__wasm_fmagic_u4)

DECLARE_F32_QS8_VCVT_UKERNEL_FUNCTION(xnn_f32_qs8_vcvt_ukernel__hvx_u32)
DECLARE_F32_QS8_VCVT_UKERNEL_FUNCTION(xnn_f32_qs8_vcvt_ukernel__hvx_u64)
DECLARE_F32_QS8_VCVT_UKERNEL_FUNCTION(xnn_f32_qs8_vcvt_ukernel__hvx_u96)
DECLARE_F32_QS8_VCVT_UKERNEL_FUNCTION(xnn_f32_qs8_vcvt_ukernel__hvx_u128)
DECLARE_F32_QS8_VCVT_UKERNEL_FUNCTION(xnn_f32_qs8_vcvt_ukernel__hvx_u256)

DECLARE_F32_QS8_VCVT_UKERNEL_FUNCTION(xnn_f32_qs8_vcvt_ukernel__scalar_fmagic_u1)
DECLARE_F32_QS8_VCVT_UKERNEL_FUNCTION(xnn_f32_qs8_vcvt_ukernel__scalar_fmagic_u2)
DECLARE_F32_QS8_VCVT_UKERNEL_FUNCTION(xnn_f32_qs8_vcvt_ukernel__scalar_fmagic_u3)
DECLARE_F32_QS8_VCVT_UKERNEL_FUNCTION(xnn_f32_qs8_vcvt_ukernel__scalar_fmagic_u4)

DECLARE_F32_QS8_VCVT_UKERNEL_FUNCTION(xnn_f32_qs8_vcvt_ukernel__scalar_imagic_u1)
DECLARE_F32_QS8_VCVT_UKERNEL_FUNCTION(xnn_f32_qs8_vcvt_ukernel__scalar_imagic_u2)
DECLARE_F32_QS8_VCVT_UKERNEL_FUNCTION(xnn_f32_qs8_vcvt_ukernel__scalar_imagic_u3)
DECLARE_F32_QS8_VCVT_UKERNEL_FUNCTION(xnn_f32_qs8_vcvt_ukernel__scalar_imagic_u4)

DECLARE_F32_QS8_VCVT_UKERNEL_FUNCTION(xnn_f32_qs8_vcvt_ukernel__scalar_lrintf_u1)
DECLARE_F32_QS8_VCVT_UKERNEL_FUNCTION(xnn_f32_qs8_vcvt_ukernel__scalar_lrintf_u2)
DECLARE_F32_QS8_VCVT_UKERNEL_FUNCTION(xnn_f32_qs8_vcvt_ukernel__scalar_lrintf_u3)
DECLARE_F32_QS8_VCVT_UKERNEL_FUNCTION(xnn_f32_qs8_vcvt_ukernel__scalar_lrintf_u4)


#define DECLARE_F32_QU8_VCVT_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                           \
      size_t n,                                        \
      const float* input,                              \
      uint8_t* output,                                 \
      const union xnn_f32_qu8_cvt_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

DECLARE_F32_QU8_VCVT_UKERNEL_FUNCTION(xnn_f32_qu8_vcvt_ukernel__neon_u8)
DECLARE_F32_QU8_VCVT_UKERNEL_FUNCTION(xnn_f32_qu8_vcvt_ukernel__neon_u16)
DECLARE_F32_QU8_VCVT_UKERNEL_FUNCTION(xnn_f32_qu8_vcvt_ukernel__neon_u24)
DECLARE_F32_QU8_VCVT_UKERNEL_FUNCTION(xnn_f32_qu8_vcvt_ukernel__neon_u32)

DECLARE_F32_QU8_VCVT_UKERNEL_FUNCTION(xnn_f32_qu8_vcvt_ukernel__neonv8_u8)
DECLARE_F32_QU8_VCVT_UKERNEL_FUNCTION(xnn_f32_qu8_vcvt_ukernel__neonv8_u16)
DECLARE_F32_QU8_VCVT_UKERNEL_FUNCTION(xnn_f32_qu8_vcvt_ukernel__neonv8_u24)
DECLARE_F32_QU8_VCVT_UKERNEL_FUNCTION(xnn_f32_qu8_vcvt_ukernel__neonv8_u32)

DECLARE_F32_QU8_VCVT_UKERNEL_FUNCTION(xnn_f32_qu8_vcvt_ukernel__sse2_u8)
DECLARE_F32_QU8_VCVT_UKERNEL_FUNCTION(xnn_f32_qu8_vcvt_ukernel__sse2_u16)
DECLARE_F32_QU8_VCVT_UKERNEL_FUNCTION(xnn_f32_qu8_vcvt_ukernel__sse2_u24)
DECLARE_F32_QU8_VCVT_UKERNEL_FUNCTION(xnn_f32_qu8_vcvt_ukernel__sse2_u32)

DECLARE_F32_QU8_VCVT_UKERNEL_FUNCTION(xnn_f32_qu8_vcvt_ukernel__avx_u8)
DECLARE_F32_QU8_VCVT_UKERNEL_FUNCTION(xnn_f32_qu8_vcvt_ukernel__avx_u16)
DECLARE_F32_QU8_VCVT_UKERNEL_FUNCTION(xnn_f32_qu8_vcvt_ukernel__avx_u24)
DECLARE_F32_QU8_VCVT_UKERNEL_FUNCTION(xnn_f32_qu8_vcvt_ukernel__avx_u32)

DECLARE_F32_QU8_VCVT_UKERNEL_FUNCTION(xnn_f32_qu8_vcvt_ukernel__avx2_u16)
DECLARE_F32_QU8_VCVT_UKERNEL_FUNCTION(xnn_f32_qu8_vcvt_ukernel__avx2_u32)
DECLARE_F32_QU8_VCVT_UKERNEL_FUNCTION(xnn_f32_qu8_vcvt_ukernel__avx2_u48)
DECLARE_F32_QU8_VCVT_UKERNEL_FUNCTION(xnn_f32_qu8_vcvt_ukernel__avx2_u64)

DECLARE_F32_QU8_VCVT_UKERNEL_FUNCTION(xnn_f32_qu8_vcvt_ukernel__avx512skx_u32)
DECLARE_F32_QU8_VCVT_UKERNEL_FUNCTION(xnn_f32_qu8_vcvt_ukernel__avx512skx_u64)
DECLARE_F32_QU8_VCVT_UKERNEL_FUNCTION(xnn_f32_qu8_vcvt_ukernel__avx512skx_u96)
DECLARE_F32_QU8_VCVT_UKERNEL_FUNCTION(xnn_f32_qu8_vcvt_ukernel__avx512skx_u128)

DECLARE_F32_QU8_VCVT_UKERNEL_FUNCTION(xnn_f32_qu8_vcvt_ukernel__wasmsimd_cvt_u8)
DECLARE_F32_QU8_VCVT_UKERNEL_FUNCTION(xnn_f32_qu8_vcvt_ukernel__wasmsimd_cvt_u16)
DECLARE_F32_QU8_VCVT_UKERNEL_FUNCTION(xnn_f32_qu8_vcvt_ukernel__wasmsimd_cvt_u24)
DECLARE_F32_QU8_VCVT_UKERNEL_FUNCTION(xnn_f32_qu8_vcvt_ukernel__wasmsimd_cvt_u32)

DECLARE_F32_QU8_VCVT_UKERNEL_FUNCTION(xnn_f32_qu8_vcvt_ukernel__wasmsimd_magic_u8)
DECLARE_F32_QU8_VCVT_UKERNEL_FUNCTION(xnn_f32_qu8_vcvt_ukernel__wasmsimd_magic_u16)
DECLARE_F32_QU8_VCVT_UKERNEL_FUNCTION(xnn_f32_qu8_vcvt_ukernel__wasmsimd_magic_u24)
DECLARE_F32_QU8_VCVT_UKERNEL_FUNCTION(xnn_f32_qu8_vcvt_ukernel__wasmsimd_magic_u32)

DECLARE_F32_QU8_VCVT_UKERNEL_FUNCTION(xnn_f32_qu8_vcvt_ukernel__wasm_fmagic_u1)
DECLARE_F32_QU8_VCVT_UKERNEL_FUNCTION(xnn_f32_qu8_vcvt_ukernel__wasm_fmagic_u2)
DECLARE_F32_QU8_VCVT_UKERNEL_FUNCTION(xnn_f32_qu8_vcvt_ukernel__wasm_fmagic_u3)
DECLARE_F32_QU8_VCVT_UKERNEL_FUNCTION(xnn_f32_qu8_vcvt_ukernel__wasm_fmagic_u4)

DECLARE_F32_QU8_VCVT_UKERNEL_FUNCTION(xnn_f32_qu8_vcvt_ukernel__scalar_fmagic_u1)
DECLARE_F32_QU8_VCVT_UKERNEL_FUNCTION(xnn_f32_qu8_vcvt_ukernel__scalar_fmagic_u2)
DECLARE_F32_QU8_VCVT_UKERNEL_FUNCTION(xnn_f32_qu8_vcvt_ukernel__scalar_fmagic_u3)
DECLARE_F32_QU8_VCVT_UKERNEL_FUNCTION(xnn_f32_qu8_vcvt_ukernel__scalar_fmagic_u4)

DECLARE_F32_QU8_VCVT_UKERNEL_FUNCTION(xnn_f32_qu8_vcvt_ukernel__scalar_imagic_u1)
DECLARE_F32_QU8_VCVT_UKERNEL_FUNCTION(xnn_f32_qu8_vcvt_ukernel__scalar_imagic_u2)
DECLARE_F32_QU8_VCVT_UKERNEL_FUNCTION(xnn_f32_qu8_vcvt_ukernel__scalar_imagic_u3)
DECLARE_F32_QU8_VCVT_UKERNEL_FUNCTION(xnn_f32_qu8_vcvt_ukernel__scalar_imagic_u4)

DECLARE_F32_QU8_VCVT_UKERNEL_FUNCTION(xnn_f32_qu8_vcvt_ukernel__scalar_lrintf_u1)
DECLARE_F32_QU8_VCVT_UKERNEL_FUNCTION(xnn_f32_qu8_vcvt_ukernel__scalar_lrintf_u2)
DECLARE_F32_QU8_VCVT_UKERNEL_FUNCTION(xnn_f32_qu8_vcvt_ukernel__scalar_lrintf_u3)
DECLARE_F32_QU8_VCVT_UKERNEL_FUNCTION(xnn_f32_qu8_vcvt_ukernel__scalar_lrintf_u4)


#define DECLARE_QS8_VCVT_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                       \
      size_t n,                                    \
      const int8_t* input,                         \
      int8_t* output,                              \
      const union xnn_qs8_cvt_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

DECLARE_QS8_VCVT_UKERNEL_FUNCTION(xnn_qs8_vcvt_ukernel__neon_u8)
DECLARE_QS8_VCVT_UKERNEL_FUNCTION(xnn_qs8_vcvt_ukernel__neon_u16)
DECLARE_QS8_VCVT_UKERNEL_FUNCTION(xnn_qs8_vcvt_ukernel__neon_u32)

DECLARE_QS8_VCVT_UKERNEL_FUNCTION(xnn_qs8_vcvt_ukernel__sse2_u16)
DECLARE_QS8_VCVT_UKERNEL_FUNCTION(xnn_qs8_vcvt_ukernel__sse2_u32)

DECLARE_QS8_VCVT_UKERNEL_FUNCTION(xnn_qs8_vcvt_ukernel__ssse3_u16)
DECLARE_QS8_VCVT_UKERNEL_FUNCTION(xnn_qs8_vcvt_ukernel__ssse3_u32)

DECLARE_QS8_VCVT_UKERNEL_FUNCTION(xnn_qs8_vcvt_ukernel__sse41_u8)
DECLARE_QS8_VCVT_UKERNEL_FUNCTION(xnn_qs8_vcvt_ukernel__sse41_u16)
DECLARE_QS8_VCVT_UKERNEL_FUNCTION(xnn_qs8_vcvt_ukernel__sse41_u32)

DECLARE_QS8_VCVT_UKERNEL_FUNCTION(xnn_qs8_vcvt_ukernel__avx_u8)
DECLARE_QS8_VCVT_UKERNEL_FUNCTION(xnn_qs8_vcvt_ukernel__avx_u16)
DECLARE_QS8_VCVT_UKERNEL_FUNCTION(xnn_qs8_vcvt_ukernel__avx_u32)

DECLARE_QS8_VCVT_UKERNEL_FUNCTION(xnn_qs8_vcvt_ukernel__avx2_u16)
DECLARE_QS8_VCVT_UKERNEL_FUNCTION(xnn_qs8_vcvt_ukernel__avx2_u32)
DECLARE_QS8_VCVT_UKERNEL_FUNCTION(xnn_qs8_vcvt_ukernel__avx2_u64)

DECLARE_QS8_VCVT_UKERNEL_FUNCTION(xnn_qs8_vcvt_ukernel__wasmsimd_u8)
DECLARE_QS8_VCVT_UKERNEL_FUNCTION(xnn_qs8_vcvt_ukernel__wasmsimd_u16)
DECLARE_QS8_VCVT_UKERNEL_FUNCTION(xnn_qs8_vcvt_ukernel__wasmsimd_u32)

DECLARE_QS8_VCVT_UKERNEL_FUNCTION(xnn_qs8_vcvt_ukernel__wasmrelaxedsimd_u8)
DECLARE_QS8_VCVT_UKERNEL_FUNCTION(xnn_qs8_vcvt_ukernel__wasmrelaxedsimd_u16)
DECLARE_QS8_VCVT_UKERNEL_FUNCTION(xnn_qs8_vcvt_ukernel__wasmrelaxedsimd_u32)

DECLARE_QS8_VCVT_UKERNEL_FUNCTION(xnn_qs8_vcvt_ukernel__armsimd32_u4)
DECLARE_QS8_VCVT_UKERNEL_FUNCTION(xnn_qs8_vcvt_ukernel__armsimd32_u8)

DECLARE_QS8_VCVT_UKERNEL_FUNCTION(xnn_qs8_vcvt_ukernel__scalar_u1)
DECLARE_QS8_VCVT_UKERNEL_FUNCTION(xnn_qs8_vcvt_ukernel__scalar_u2)
DECLARE_QS8_VCVT_UKERNEL_FUNCTION(xnn_qs8_vcvt_ukernel__scalar_u4)


#define DECLARE_QS16_QS8_VCVT_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                            \
      size_t n,                                         \
      const int16_t* input,                             \
      int8_t* output,                                   \
      const union xnn_qs16_qs8_cvt_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

DECLARE_QS16_QS8_VCVT_UKERNEL_FUNCTION(xnn_qs16_qs8_vcvt_ukernel__asm_aarch32_neon_u16)
DECLARE_QS16_QS8_VCVT_UKERNEL_FUNCTION(xnn_qs16_qs8_vcvt_ukernel__neon_u8)
DECLARE_QS16_QS8_VCVT_UKERNEL_FUNCTION(xnn_qs16_qs8_vcvt_ukernel__neon_u16)
DECLARE_QS16_QS8_VCVT_UKERNEL_FUNCTION(xnn_qs16_qs8_vcvt_ukernel__neon_u32)

DECLARE_QS16_QS8_VCVT_UKERNEL_FUNCTION(xnn_qs16_qs8_vcvt_ukernel__sse2_u4)
DECLARE_QS16_QS8_VCVT_UKERNEL_FUNCTION(xnn_qs16_qs8_vcvt_ukernel__sse2_u8)
DECLARE_QS16_QS8_VCVT_UKERNEL_FUNCTION(xnn_qs16_qs8_vcvt_ukernel__sse2_u16)

DECLARE_QS16_QS8_VCVT_UKERNEL_FUNCTION(xnn_qs16_qs8_vcvt_ukernel__ssse3_u4)
DECLARE_QS16_QS8_VCVT_UKERNEL_FUNCTION(xnn_qs16_qs8_vcvt_ukernel__ssse3_u8)
DECLARE_QS16_QS8_VCVT_UKERNEL_FUNCTION(xnn_qs16_qs8_vcvt_ukernel__ssse3_u16)

DECLARE_QS16_QS8_VCVT_UKERNEL_FUNCTION(xnn_qs16_qs8_vcvt_ukernel__sse41_u4)
DECLARE_QS16_QS8_VCVT_UKERNEL_FUNCTION(xnn_qs16_qs8_vcvt_ukernel__sse41_u8)
DECLARE_QS16_QS8_VCVT_UKERNEL_FUNCTION(xnn_qs16_qs8_vcvt_ukernel__sse41_u16)

DECLARE_QS16_QS8_VCVT_UKERNEL_FUNCTION(xnn_qs16_qs8_vcvt_ukernel__avx_u4)
DECLARE_QS16_QS8_VCVT_UKERNEL_FUNCTION(xnn_qs16_qs8_vcvt_ukernel__avx_u8)
DECLARE_QS16_QS8_VCVT_UKERNEL_FUNCTION(xnn_qs16_qs8_vcvt_ukernel__avx_u16)

DECLARE_QS16_QS8_VCVT_UKERNEL_FUNCTION(xnn_qs16_qs8_vcvt_ukernel__wasmsimd_u8)
DECLARE_QS16_QS8_VCVT_UKERNEL_FUNCTION(xnn_qs16_qs8_vcvt_ukernel__wasmsimd_u16)
DECLARE_QS16_QS8_VCVT_UKERNEL_FUNCTION(xnn_qs16_qs8_vcvt_ukernel__wasmsimd_u32)

DECLARE_QS16_QS8_VCVT_UKERNEL_FUNCTION(xnn_qs16_qs8_vcvt_ukernel__scalar_u1)
DECLARE_QS16_QS8_VCVT_UKERNEL_FUNCTION(xnn_qs16_qs8_vcvt_ukernel__scalar_u2)
DECLARE_QS16_QS8_VCVT_UKERNEL_FUNCTION(xnn_qs16_qs8_vcvt_ukernel__scalar_u4)

#define DECLARE_QS8_F16_VCVT_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                           \
      size_t n,                                        \
      const int8_t* input,                             \
      void* output,                                    \
      const union xnn_qs8_f16_cvt_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

DECLARE_QS8_F16_VCVT_UKERNEL_FUNCTION(xnn_qs8_f16_vcvt_ukernel__neonfp16arith_u8)
DECLARE_QS8_F16_VCVT_UKERNEL_FUNCTION(xnn_qs8_f16_vcvt_ukernel__neonfp16arith_u16)
DECLARE_QS8_F16_VCVT_UKERNEL_FUNCTION(xnn_qs8_f16_vcvt_ukernel__neonfp16arith_u24)
DECLARE_QS8_F16_VCVT_UKERNEL_FUNCTION(xnn_qs8_f16_vcvt_ukernel__neonfp16arith_u32)

DECLARE_QS8_F16_VCVT_UKERNEL_FUNCTION(xnn_qs8_f16_vcvt_ukernel__avx2_u16)
DECLARE_QS8_F16_VCVT_UKERNEL_FUNCTION(xnn_qs8_f16_vcvt_ukernel__avx2_u24)
DECLARE_QS8_F16_VCVT_UKERNEL_FUNCTION(xnn_qs8_f16_vcvt_ukernel__avx2_u32)
DECLARE_QS8_F16_VCVT_UKERNEL_FUNCTION(xnn_qs8_f16_vcvt_ukernel__avx2_u64)

#define DECLARE_QS8_F32_VCVT_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                           \
      size_t n,                                        \
      const int8_t* input,                             \
      float* output,                                   \
      const union xnn_qs8_f32_cvt_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

DECLARE_QS8_F32_VCVT_UKERNEL_FUNCTION(xnn_qs8_f32_vcvt_ukernel__neon_u8)
DECLARE_QS8_F32_VCVT_UKERNEL_FUNCTION(xnn_qs8_f32_vcvt_ukernel__neon_u16)
DECLARE_QS8_F32_VCVT_UKERNEL_FUNCTION(xnn_qs8_f32_vcvt_ukernel__neon_u24)
DECLARE_QS8_F32_VCVT_UKERNEL_FUNCTION(xnn_qs8_f32_vcvt_ukernel__neon_u32)

DECLARE_QS8_F32_VCVT_UKERNEL_FUNCTION(xnn_qs8_f32_vcvt_ukernel__sse2_u8)
DECLARE_QS8_F32_VCVT_UKERNEL_FUNCTION(xnn_qs8_f32_vcvt_ukernel__sse2_u16)
DECLARE_QS8_F32_VCVT_UKERNEL_FUNCTION(xnn_qs8_f32_vcvt_ukernel__sse2_u24)
DECLARE_QS8_F32_VCVT_UKERNEL_FUNCTION(xnn_qs8_f32_vcvt_ukernel__sse2_u32)

DECLARE_QS8_F32_VCVT_UKERNEL_FUNCTION(xnn_qs8_f32_vcvt_ukernel__sse41_u8)
DECLARE_QS8_F32_VCVT_UKERNEL_FUNCTION(xnn_qs8_f32_vcvt_ukernel__sse41_u16)
DECLARE_QS8_F32_VCVT_UKERNEL_FUNCTION(xnn_qs8_f32_vcvt_ukernel__sse41_u24)
DECLARE_QS8_F32_VCVT_UKERNEL_FUNCTION(xnn_qs8_f32_vcvt_ukernel__sse41_u32)

DECLARE_QS8_F32_VCVT_UKERNEL_FUNCTION(xnn_qs8_f32_vcvt_ukernel__avx_u8)
DECLARE_QS8_F32_VCVT_UKERNEL_FUNCTION(xnn_qs8_f32_vcvt_ukernel__avx_u16)
DECLARE_QS8_F32_VCVT_UKERNEL_FUNCTION(xnn_qs8_f32_vcvt_ukernel__avx_u24)
DECLARE_QS8_F32_VCVT_UKERNEL_FUNCTION(xnn_qs8_f32_vcvt_ukernel__avx_u32)

DECLARE_QS8_F32_VCVT_UKERNEL_FUNCTION(xnn_qs8_f32_vcvt_ukernel__avx2_u8)
DECLARE_QS8_F32_VCVT_UKERNEL_FUNCTION(xnn_qs8_f32_vcvt_ukernel__avx2_u16)
DECLARE_QS8_F32_VCVT_UKERNEL_FUNCTION(xnn_qs8_f32_vcvt_ukernel__avx2_u24)
DECLARE_QS8_F32_VCVT_UKERNEL_FUNCTION(xnn_qs8_f32_vcvt_ukernel__avx2_u32)

DECLARE_QS8_F32_VCVT_UKERNEL_FUNCTION(xnn_qs8_f32_vcvt_ukernel__avx512skx_u16)
DECLARE_QS8_F32_VCVT_UKERNEL_FUNCTION(xnn_qs8_f32_vcvt_ukernel__avx512skx_u32)
DECLARE_QS8_F32_VCVT_UKERNEL_FUNCTION(xnn_qs8_f32_vcvt_ukernel__avx512skx_u48)
DECLARE_QS8_F32_VCVT_UKERNEL_FUNCTION(xnn_qs8_f32_vcvt_ukernel__avx512skx_u64)

DECLARE_QS8_F32_VCVT_UKERNEL_FUNCTION(xnn_qs8_f32_vcvt_ukernel__wasmsimd_u8)
DECLARE_QS8_F32_VCVT_UKERNEL_FUNCTION(xnn_qs8_f32_vcvt_ukernel__wasmsimd_u16)
DECLARE_QS8_F32_VCVT_UKERNEL_FUNCTION(xnn_qs8_f32_vcvt_ukernel__wasmsimd_u24)
DECLARE_QS8_F32_VCVT_UKERNEL_FUNCTION(xnn_qs8_f32_vcvt_ukernel__wasmsimd_u32)

DECLARE_QS8_F32_VCVT_UKERNEL_FUNCTION(xnn_qs8_f32_vcvt_ukernel__scalar_u1)
DECLARE_QS8_F32_VCVT_UKERNEL_FUNCTION(xnn_qs8_f32_vcvt_ukernel__scalar_u2)
DECLARE_QS8_F32_VCVT_UKERNEL_FUNCTION(xnn_qs8_f32_vcvt_ukernel__scalar_u3)
DECLARE_QS8_F32_VCVT_UKERNEL_FUNCTION(xnn_qs8_f32_vcvt_ukernel__scalar_u4)


#define DECLARE_QU8_VCVT_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                       \
      size_t n,                                    \
      const uint8_t* input,                        \
      uint8_t* output,                             \
      const union xnn_qu8_cvt_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

DECLARE_QU8_VCVT_UKERNEL_FUNCTION(xnn_qu8_vcvt_ukernel__neon_u8)
DECLARE_QU8_VCVT_UKERNEL_FUNCTION(xnn_qu8_vcvt_ukernel__neon_u16)
DECLARE_QU8_VCVT_UKERNEL_FUNCTION(xnn_qu8_vcvt_ukernel__neon_u32)

DECLARE_QU8_VCVT_UKERNEL_FUNCTION(xnn_qu8_vcvt_ukernel__sse2_u16)
DECLARE_QU8_VCVT_UKERNEL_FUNCTION(xnn_qu8_vcvt_ukernel__sse2_u32)

DECLARE_QU8_VCVT_UKERNEL_FUNCTION(xnn_qu8_vcvt_ukernel__ssse3_u16)
DECLARE_QU8_VCVT_UKERNEL_FUNCTION(xnn_qu8_vcvt_ukernel__ssse3_u32)

DECLARE_QU8_VCVT_UKERNEL_FUNCTION(xnn_qu8_vcvt_ukernel__sse41_u8)
DECLARE_QU8_VCVT_UKERNEL_FUNCTION(xnn_qu8_vcvt_ukernel__sse41_u16)
DECLARE_QU8_VCVT_UKERNEL_FUNCTION(xnn_qu8_vcvt_ukernel__sse41_u32)

DECLARE_QU8_VCVT_UKERNEL_FUNCTION(xnn_qu8_vcvt_ukernel__avx_u8)
DECLARE_QU8_VCVT_UKERNEL_FUNCTION(xnn_qu8_vcvt_ukernel__avx_u16)
DECLARE_QU8_VCVT_UKERNEL_FUNCTION(xnn_qu8_vcvt_ukernel__avx_u32)

DECLARE_QU8_VCVT_UKERNEL_FUNCTION(xnn_qu8_vcvt_ukernel__avx2_u16)
DECLARE_QU8_VCVT_UKERNEL_FUNCTION(xnn_qu8_vcvt_ukernel__avx2_u32)
DECLARE_QU8_VCVT_UKERNEL_FUNCTION(xnn_qu8_vcvt_ukernel__avx2_u64)

DECLARE_QU8_VCVT_UKERNEL_FUNCTION(xnn_qu8_vcvt_ukernel__wasmsimd_u8)
DECLARE_QU8_VCVT_UKERNEL_FUNCTION(xnn_qu8_vcvt_ukernel__wasmsimd_u16)
DECLARE_QU8_VCVT_UKERNEL_FUNCTION(xnn_qu8_vcvt_ukernel__wasmsimd_u32)

DECLARE_QU8_VCVT_UKERNEL_FUNCTION(xnn_qu8_vcvt_ukernel__wasmrelaxedsimd_u8)
DECLARE_QU8_VCVT_UKERNEL_FUNCTION(xnn_qu8_vcvt_ukernel__wasmrelaxedsimd_u16)
DECLARE_QU8_VCVT_UKERNEL_FUNCTION(xnn_qu8_vcvt_ukernel__wasmrelaxedsimd_u32)

DECLARE_QU8_VCVT_UKERNEL_FUNCTION(xnn_qu8_vcvt_ukernel__armsimd32_u4)
DECLARE_QU8_VCVT_UKERNEL_FUNCTION(xnn_qu8_vcvt_ukernel__armsimd32_u8)

DECLARE_QU8_VCVT_UKERNEL_FUNCTION(xnn_qu8_vcvt_ukernel__scalar_u1)
DECLARE_QU8_VCVT_UKERNEL_FUNCTION(xnn_qu8_vcvt_ukernel__scalar_u2)
DECLARE_QU8_VCVT_UKERNEL_FUNCTION(xnn_qu8_vcvt_ukernel__scalar_u4)


#define DECLARE_QU8_F32_VCVT_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                           \
      size_t n,                                        \
      const uint8_t* input,                            \
      float* output,                                   \
      const union xnn_qu8_f32_cvt_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

DECLARE_QU8_F32_VCVT_UKERNEL_FUNCTION(xnn_qu8_f32_vcvt_ukernel__neon_u8)
DECLARE_QU8_F32_VCVT_UKERNEL_FUNCTION(xnn_qu8_f32_vcvt_ukernel__neon_u16)
DECLARE_QU8_F32_VCVT_UKERNEL_FUNCTION(xnn_qu8_f32_vcvt_ukernel__neon_u24)
DECLARE_QU8_F32_VCVT_UKERNEL_FUNCTION(xnn_qu8_f32_vcvt_ukernel__neon_u32)

DECLARE_QU8_F32_VCVT_UKERNEL_FUNCTION(xnn_qu8_f32_vcvt_ukernel__sse2_u8)
DECLARE_QU8_F32_VCVT_UKERNEL_FUNCTION(xnn_qu8_f32_vcvt_ukernel__sse2_u16)
DECLARE_QU8_F32_VCVT_UKERNEL_FUNCTION(xnn_qu8_f32_vcvt_ukernel__sse2_u24)
DECLARE_QU8_F32_VCVT_UKERNEL_FUNCTION(xnn_qu8_f32_vcvt_ukernel__sse2_u32)

DECLARE_QU8_F32_VCVT_UKERNEL_FUNCTION(xnn_qu8_f32_vcvt_ukernel__sse41_u8)
DECLARE_QU8_F32_VCVT_UKERNEL_FUNCTION(xnn_qu8_f32_vcvt_ukernel__sse41_u16)
DECLARE_QU8_F32_VCVT_UKERNEL_FUNCTION(xnn_qu8_f32_vcvt_ukernel__sse41_u24)
DECLARE_QU8_F32_VCVT_UKERNEL_FUNCTION(xnn_qu8_f32_vcvt_ukernel__sse41_u32)

DECLARE_QU8_F32_VCVT_UKERNEL_FUNCTION(xnn_qu8_f32_vcvt_ukernel__avx_u8)
DECLARE_QU8_F32_VCVT_UKERNEL_FUNCTION(xnn_qu8_f32_vcvt_ukernel__avx_u16)
DECLARE_QU8_F32_VCVT_UKERNEL_FUNCTION(xnn_qu8_f32_vcvt_ukernel__avx_u24)
DECLARE_QU8_F32_VCVT_UKERNEL_FUNCTION(xnn_qu8_f32_vcvt_ukernel__avx_u32)

DECLARE_QU8_F32_VCVT_UKERNEL_FUNCTION(xnn_qu8_f32_vcvt_ukernel__avx2_u8)
DECLARE_QU8_F32_VCVT_UKERNEL_FUNCTION(xnn_qu8_f32_vcvt_ukernel__avx2_u16)
DECLARE_QU8_F32_VCVT_UKERNEL_FUNCTION(xnn_qu8_f32_vcvt_ukernel__avx2_u24)
DECLARE_QU8_F32_VCVT_UKERNEL_FUNCTION(xnn_qu8_f32_vcvt_ukernel__avx2_u32)

DECLARE_QU8_F32_VCVT_UKERNEL_FUNCTION(xnn_qu8_f32_vcvt_ukernel__avx512skx_u16)
DECLARE_QU8_F32_VCVT_UKERNEL_FUNCTION(xnn_qu8_f32_vcvt_ukernel__avx512skx_u32)
DECLARE_QU8_F32_VCVT_UKERNEL_FUNCTION(xnn_qu8_f32_vcvt_ukernel__avx512skx_u48)
DECLARE_QU8_F32_VCVT_UKERNEL_FUNCTION(xnn_qu8_f32_vcvt_ukernel__avx512skx_u64)

DECLARE_QU8_F32_VCVT_UKERNEL_FUNCTION(xnn_qu8_f32_vcvt_ukernel__wasmsimd_u8)
DECLARE_QU8_F32_VCVT_UKERNEL_FUNCTION(xnn_qu8_f32_vcvt_ukernel__wasmsimd_u16)
DECLARE_QU8_F32_VCVT_UKERNEL_FUNCTION(xnn_qu8_f32_vcvt_ukernel__wasmsimd_u24)
DECLARE_QU8_F32_VCVT_UKERNEL_FUNCTION(xnn_qu8_f32_vcvt_ukernel__wasmsimd_u32)

DECLARE_QU8_F32_VCVT_UKERNEL_FUNCTION(xnn_qu8_f32_vcvt_ukernel__scalar_u1)
DECLARE_QU8_F32_VCVT_UKERNEL_FUNCTION(xnn_qu8_f32_vcvt_ukernel__scalar_u2)
DECLARE_QU8_F32_VCVT_UKERNEL_FUNCTION(xnn_qu8_f32_vcvt_ukernel__scalar_u3)
DECLARE_QU8_F32_VCVT_UKERNEL_FUNCTION(xnn_qu8_f32_vcvt_ukernel__scalar_u4)


#ifdef __cplusplus
}  // extern "C"
#endif
