// Copyright 2023 Google LLC
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

#define DECLARE_F16_RSUM_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                       \
      size_t batch,                                \
      const void* input,                           \
      void* output,                                \
      const union xnn_f16_scale_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

DECLARE_F16_RSUM_UKERNEL_FUNCTION(xnn_f16_rsum_ukernel__neonfp16arith_u8)
DECLARE_F16_RSUM_UKERNEL_FUNCTION(xnn_f16_rsum_ukernel__neonfp16arith_u16_acc2)
DECLARE_F16_RSUM_UKERNEL_FUNCTION(xnn_f16_rsum_ukernel__neonfp16arith_u24_acc3)
DECLARE_F16_RSUM_UKERNEL_FUNCTION(xnn_f16_rsum_ukernel__neonfp16arith_u32_acc2)
DECLARE_F16_RSUM_UKERNEL_FUNCTION(xnn_f16_rsum_ukernel__neonfp16arith_u32_acc4)

DECLARE_F16_RSUM_UKERNEL_FUNCTION(xnn_f16_rsum_ukernel__avx512fp16_u32)
DECLARE_F16_RSUM_UKERNEL_FUNCTION(xnn_f16_rsum_ukernel__avx512fp16_u64_acc2)
DECLARE_F16_RSUM_UKERNEL_FUNCTION(xnn_f16_rsum_ukernel__avx512fp16_u96_acc3)
DECLARE_F16_RSUM_UKERNEL_FUNCTION(xnn_f16_rsum_ukernel__avx512fp16_u128_acc2)
DECLARE_F16_RSUM_UKERNEL_FUNCTION(xnn_f16_rsum_ukernel__avx512fp16_u128_acc4)


#define DECLARE_F16_F32ACC_RSUM_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                              \
      size_t batch,                                       \
      const void* input,                                  \
      float* output,                                       \
      const union xnn_f16_f32acc_scale_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

DECLARE_F16_F32ACC_RSUM_UKERNEL_FUNCTION(xnn_f16_f32acc_rsum_ukernel__neonfp16arith_u4)
DECLARE_F16_F32ACC_RSUM_UKERNEL_FUNCTION(xnn_f16_f32acc_rsum_ukernel__neonfp16arith_u8)
DECLARE_F16_F32ACC_RSUM_UKERNEL_FUNCTION(xnn_f16_f32acc_rsum_ukernel__neonfp16arith_u16_acc2)
DECLARE_F16_F32ACC_RSUM_UKERNEL_FUNCTION(xnn_f16_f32acc_rsum_ukernel__neonfp16arith_u24_acc3)
DECLARE_F16_F32ACC_RSUM_UKERNEL_FUNCTION(xnn_f16_f32acc_rsum_ukernel__neonfp16arith_u32_acc2)
DECLARE_F16_F32ACC_RSUM_UKERNEL_FUNCTION(xnn_f16_f32acc_rsum_ukernel__neonfp16arith_u32_acc4)

DECLARE_F16_F32ACC_RSUM_UKERNEL_FUNCTION(xnn_f16_f32acc_rsum_ukernel__avx512skx_u16)
DECLARE_F16_F32ACC_RSUM_UKERNEL_FUNCTION(xnn_f16_f32acc_rsum_ukernel__avx512skx_u32_acc2)
DECLARE_F16_F32ACC_RSUM_UKERNEL_FUNCTION(xnn_f16_f32acc_rsum_ukernel__avx512skx_u48_acc3)
DECLARE_F16_F32ACC_RSUM_UKERNEL_FUNCTION(xnn_f16_f32acc_rsum_ukernel__avx512skx_u64_acc2)
DECLARE_F16_F32ACC_RSUM_UKERNEL_FUNCTION(xnn_f16_f32acc_rsum_ukernel__avx512skx_u64_acc4)
DECLARE_F16_F32ACC_RSUM_UKERNEL_FUNCTION(xnn_f16_f32acc_rsum_ukernel__avx512skx_u128_acc4)

DECLARE_F16_F32ACC_RSUM_UKERNEL_FUNCTION(xnn_f16_f32acc_rsum_ukernel__f16c_u8)
DECLARE_F16_F32ACC_RSUM_UKERNEL_FUNCTION(xnn_f16_f32acc_rsum_ukernel__f16c_u16_acc2)
DECLARE_F16_F32ACC_RSUM_UKERNEL_FUNCTION(xnn_f16_f32acc_rsum_ukernel__f16c_u24_acc3)
DECLARE_F16_F32ACC_RSUM_UKERNEL_FUNCTION(xnn_f16_f32acc_rsum_ukernel__f16c_u32_acc2)
DECLARE_F16_F32ACC_RSUM_UKERNEL_FUNCTION(xnn_f16_f32acc_rsum_ukernel__f16c_u32_acc4)


#define DECLARE_F16_REDUCE_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                       \
      size_t batch,                                \
      const void* input,                           \
      void* output,                                \
      const union xnn_f16_default_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

DECLARE_F16_REDUCE_UKERNEL_FUNCTION(xnn_f16_rmax_ukernel__neonfp16arith_u8)
DECLARE_F16_REDUCE_UKERNEL_FUNCTION(xnn_f16_rmax_ukernel__neonfp16arith_u16_acc2)
DECLARE_F16_REDUCE_UKERNEL_FUNCTION(xnn_f16_rmax_ukernel__neonfp16arith_u24_acc3)
DECLARE_F16_REDUCE_UKERNEL_FUNCTION(xnn_f16_rmax_ukernel__neonfp16arith_u32_acc2)
DECLARE_F16_REDUCE_UKERNEL_FUNCTION(xnn_f16_rmax_ukernel__neonfp16arith_u32_acc4)

DECLARE_F16_REDUCE_UKERNEL_FUNCTION(xnn_f16_rmin_ukernel__neonfp16arith_u8)
DECLARE_F16_REDUCE_UKERNEL_FUNCTION(xnn_f16_rmin_ukernel__neonfp16arith_u16_acc2)
DECLARE_F16_REDUCE_UKERNEL_FUNCTION(xnn_f16_rmin_ukernel__neonfp16arith_u24_acc3)
DECLARE_F16_REDUCE_UKERNEL_FUNCTION(xnn_f16_rmin_ukernel__neonfp16arith_u32_acc2)
DECLARE_F16_REDUCE_UKERNEL_FUNCTION(xnn_f16_rmin_ukernel__neonfp16arith_u32_acc4)

DECLARE_F16_REDUCE_UKERNEL_FUNCTION(xnn_f16_rminmax_ukernel__neonfp16arith_u8)
DECLARE_F16_REDUCE_UKERNEL_FUNCTION(xnn_f16_rminmax_ukernel__neonfp16arith_u16_acc2)
DECLARE_F16_REDUCE_UKERNEL_FUNCTION(xnn_f16_rminmax_ukernel__neonfp16arith_u24_acc3)
DECLARE_F16_REDUCE_UKERNEL_FUNCTION(xnn_f16_rminmax_ukernel__neonfp16arith_u32_acc2)
DECLARE_F16_REDUCE_UKERNEL_FUNCTION(xnn_f16_rminmax_ukernel__neonfp16arith_u32_acc4)

DECLARE_F16_REDUCE_UKERNEL_FUNCTION(xnn_f16_rmax_ukernel__avx512fp16_u32)
DECLARE_F16_REDUCE_UKERNEL_FUNCTION(xnn_f16_rmax_ukernel__avx512fp16_u64_acc2)
DECLARE_F16_REDUCE_UKERNEL_FUNCTION(xnn_f16_rmax_ukernel__avx512fp16_u96_acc3)
DECLARE_F16_REDUCE_UKERNEL_FUNCTION(xnn_f16_rmax_ukernel__avx512fp16_u128_acc2)
DECLARE_F16_REDUCE_UKERNEL_FUNCTION(xnn_f16_rmax_ukernel__avx512fp16_u128_acc4)

DECLARE_F16_REDUCE_UKERNEL_FUNCTION(xnn_f16_rmin_ukernel__avx512fp16_u32)
DECLARE_F16_REDUCE_UKERNEL_FUNCTION(xnn_f16_rmin_ukernel__avx512fp16_u64_acc2)
DECLARE_F16_REDUCE_UKERNEL_FUNCTION(xnn_f16_rmin_ukernel__avx512fp16_u96_acc3)
DECLARE_F16_REDUCE_UKERNEL_FUNCTION(xnn_f16_rmin_ukernel__avx512fp16_u128_acc2)
DECLARE_F16_REDUCE_UKERNEL_FUNCTION(xnn_f16_rmin_ukernel__avx512fp16_u128_acc4)

DECLARE_F16_REDUCE_UKERNEL_FUNCTION(xnn_f16_rminmax_ukernel__avx512fp16_u32)
DECLARE_F16_REDUCE_UKERNEL_FUNCTION(xnn_f16_rminmax_ukernel__avx512fp16_u64_acc2)
DECLARE_F16_REDUCE_UKERNEL_FUNCTION(xnn_f16_rminmax_ukernel__avx512fp16_u96_acc3)
DECLARE_F16_REDUCE_UKERNEL_FUNCTION(xnn_f16_rminmax_ukernel__avx512fp16_u128_acc2)
DECLARE_F16_REDUCE_UKERNEL_FUNCTION(xnn_f16_rminmax_ukernel__avx512fp16_u128_acc4)

DECLARE_F16_REDUCE_UKERNEL_FUNCTION(xnn_f16_rmax_ukernel__avx512skx_u16)
DECLARE_F16_REDUCE_UKERNEL_FUNCTION(xnn_f16_rmax_ukernel__avx512skx_u32_acc2)
DECLARE_F16_REDUCE_UKERNEL_FUNCTION(xnn_f16_rmax_ukernel__avx512skx_u48_acc3)
DECLARE_F16_REDUCE_UKERNEL_FUNCTION(xnn_f16_rmax_ukernel__avx512skx_u64_acc2)
DECLARE_F16_REDUCE_UKERNEL_FUNCTION(xnn_f16_rmax_ukernel__avx512skx_u64_acc4)

DECLARE_F16_REDUCE_UKERNEL_FUNCTION(xnn_f16_rmin_ukernel__avx512skx_u16)
DECLARE_F16_REDUCE_UKERNEL_FUNCTION(xnn_f16_rmin_ukernel__avx512skx_u32_acc2)
DECLARE_F16_REDUCE_UKERNEL_FUNCTION(xnn_f16_rmin_ukernel__avx512skx_u48_acc3)
DECLARE_F16_REDUCE_UKERNEL_FUNCTION(xnn_f16_rmin_ukernel__avx512skx_u64_acc2)
DECLARE_F16_REDUCE_UKERNEL_FUNCTION(xnn_f16_rmin_ukernel__avx512skx_u64_acc4)

DECLARE_F16_REDUCE_UKERNEL_FUNCTION(xnn_f16_rminmax_ukernel__avx512skx_u16)
DECLARE_F16_REDUCE_UKERNEL_FUNCTION(xnn_f16_rminmax_ukernel__avx512skx_u32_acc2)
DECLARE_F16_REDUCE_UKERNEL_FUNCTION(xnn_f16_rminmax_ukernel__avx512skx_u48_acc3)
DECLARE_F16_REDUCE_UKERNEL_FUNCTION(xnn_f16_rminmax_ukernel__avx512skx_u64_acc2)
DECLARE_F16_REDUCE_UKERNEL_FUNCTION(xnn_f16_rminmax_ukernel__avx512skx_u64_acc4)

DECLARE_F16_REDUCE_UKERNEL_FUNCTION(xnn_f16_rmax_ukernel__f16c_u32)

DECLARE_F16_REDUCE_UKERNEL_FUNCTION(xnn_f16_rmax_ukernel__scalar_u1)
DECLARE_F16_REDUCE_UKERNEL_FUNCTION(xnn_f16_rmax_ukernel__scalar_u2_acc2)
DECLARE_F16_REDUCE_UKERNEL_FUNCTION(xnn_f16_rmax_ukernel__scalar_u3_acc3)
DECLARE_F16_REDUCE_UKERNEL_FUNCTION(xnn_f16_rmax_ukernel__scalar_u4_acc2)
DECLARE_F16_REDUCE_UKERNEL_FUNCTION(xnn_f16_rmax_ukernel__scalar_u4_acc4)

DECLARE_F16_REDUCE_UKERNEL_FUNCTION(xnn_f16_rmin_ukernel__scalar_u1)
DECLARE_F16_REDUCE_UKERNEL_FUNCTION(xnn_f16_rmin_ukernel__scalar_u2_acc2)
DECLARE_F16_REDUCE_UKERNEL_FUNCTION(xnn_f16_rmin_ukernel__scalar_u3_acc3)
DECLARE_F16_REDUCE_UKERNEL_FUNCTION(xnn_f16_rmin_ukernel__scalar_u4_acc2)
DECLARE_F16_REDUCE_UKERNEL_FUNCTION(xnn_f16_rmin_ukernel__scalar_u4_acc4)

DECLARE_F16_REDUCE_UKERNEL_FUNCTION(xnn_f16_rminmax_ukernel__scalar_u1)
DECLARE_F16_REDUCE_UKERNEL_FUNCTION(xnn_f16_rminmax_ukernel__scalar_u2_acc2)
DECLARE_F16_REDUCE_UKERNEL_FUNCTION(xnn_f16_rminmax_ukernel__scalar_u3_acc3)
DECLARE_F16_REDUCE_UKERNEL_FUNCTION(xnn_f16_rminmax_ukernel__scalar_u4_acc2)
DECLARE_F16_REDUCE_UKERNEL_FUNCTION(xnn_f16_rminmax_ukernel__scalar_u4_acc4)

#define DECLARE_F32_REDUCE_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                         \
      size_t batch,                                  \
      const float* input,                            \
      float* output,                                 \
      const union xnn_f32_default_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmax_ukernel__avx_u8)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmax_ukernel__avx_u16_acc2)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmax_ukernel__avx_u24_acc3)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmax_ukernel__avx_u32_acc2)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmax_ukernel__avx_u32_acc4)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmax_ukernel__avx512f_u16)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmax_ukernel__avx512f_u32_acc2)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmax_ukernel__avx512f_u48_acc3)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmax_ukernel__avx512f_u64_acc2)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmax_ukernel__avx512f_u64_acc4)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmax_ukernel__neon_u4)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmax_ukernel__neon_u8_acc2)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmax_ukernel__neon_u12_acc3)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmax_ukernel__neon_u16_acc2)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmax_ukernel__neon_u16_acc4)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmax_ukernel__rvv_u1v)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmax_ukernel__rvv_u2v)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmax_ukernel__rvv_u4v)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmax_ukernel__rvv_u8v)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmax_ukernel__scalar_u1)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmax_ukernel__scalar_u2_acc2)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmax_ukernel__scalar_u3_acc3)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmax_ukernel__scalar_u4_acc2)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmax_ukernel__scalar_u4_acc4)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmax_ukernel__sse_u4)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmax_ukernel__sse_u8_acc2)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmax_ukernel__sse_u12_acc3)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmax_ukernel__sse_u16_acc2)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmax_ukernel__sse_u16_acc4)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmax_ukernel__wasm_u1)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmax_ukernel__wasm_u2_acc2)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmax_ukernel__wasm_u3_acc3)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmax_ukernel__wasm_u4_acc2)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmax_ukernel__wasm_u4_acc4)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmax_ukernel__wasmsimd_minmax_u4)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmax_ukernel__wasmsimd_minmax_u8_acc2)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmax_ukernel__wasmsimd_minmax_u12_acc3)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmax_ukernel__wasmsimd_minmax_u16_acc2)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmax_ukernel__wasmsimd_minmax_u16_acc4)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmax_ukernel__wasmsimd_pminmax_u4)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmax_ukernel__wasmsimd_pminmax_u8_acc2)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmax_ukernel__wasmsimd_pminmax_u12_acc3)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmax_ukernel__wasmsimd_pminmax_u16_acc2)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmax_ukernel__wasmsimd_pminmax_u16_acc4)

DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmin_ukernel__avx_u8)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmin_ukernel__avx_u16_acc2)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmin_ukernel__avx_u24_acc3)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmin_ukernel__avx_u32_acc2)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmin_ukernel__avx_u32_acc4)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmin_ukernel__avx512f_u16)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmin_ukernel__avx512f_u32_acc2)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmin_ukernel__avx512f_u48_acc3)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmin_ukernel__avx512f_u64_acc2)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmin_ukernel__avx512f_u64_acc4)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmin_ukernel__neon_u4)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmin_ukernel__neon_u8_acc2)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmin_ukernel__neon_u12_acc3)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmin_ukernel__neon_u16_acc2)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmin_ukernel__neon_u16_acc4)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmin_ukernel__rvv_u1v)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmin_ukernel__rvv_u2v)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmin_ukernel__rvv_u4v)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmin_ukernel__rvv_u8v)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmin_ukernel__scalar_u1)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmin_ukernel__scalar_u2_acc2)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmin_ukernel__scalar_u3_acc3)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmin_ukernel__scalar_u4_acc2)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmin_ukernel__scalar_u4_acc4)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmin_ukernel__sse_u4)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmin_ukernel__sse_u8_acc2)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmin_ukernel__sse_u12_acc3)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmin_ukernel__sse_u16_acc2)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmin_ukernel__sse_u16_acc4)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmin_ukernel__wasm_u1)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmin_ukernel__wasm_u2_acc2)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmin_ukernel__wasm_u3_acc3)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmin_ukernel__wasm_u4_acc2)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmin_ukernel__wasm_u4_acc4)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmin_ukernel__wasmsimd_minmax_u4)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmin_ukernel__wasmsimd_minmax_u8_acc2)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmin_ukernel__wasmsimd_minmax_u12_acc3)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmin_ukernel__wasmsimd_minmax_u16_acc2)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmin_ukernel__wasmsimd_minmax_u16_acc4)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmin_ukernel__wasmsimd_pminmax_u4)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmin_ukernel__wasmsimd_pminmax_u8_acc2)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmin_ukernel__wasmsimd_pminmax_u12_acc3)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmin_ukernel__wasmsimd_pminmax_u16_acc2)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmin_ukernel__wasmsimd_pminmax_u16_acc4)

DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rminmax_ukernel__avx_u8)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rminmax_ukernel__avx_u16_acc2)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rminmax_ukernel__avx_u24_acc3)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rminmax_ukernel__avx_u32_acc2)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rminmax_ukernel__avx_u32_acc4)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rminmax_ukernel__avx512f_u16)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rminmax_ukernel__avx512f_u32_acc2)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rminmax_ukernel__avx512f_u48_acc3)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rminmax_ukernel__avx512f_u64_acc2)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rminmax_ukernel__avx512f_u64_acc4)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rminmax_ukernel__neon_u4)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rminmax_ukernel__neon_u8_acc2)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rminmax_ukernel__neon_u12_acc3)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rminmax_ukernel__neon_u16_acc2)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rminmax_ukernel__neon_u16_acc4)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rminmax_ukernel__rvv_u1v)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rminmax_ukernel__rvv_u2v)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rminmax_ukernel__rvv_u4v)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rminmax_ukernel__rvv_u8v)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rminmax_ukernel__scalar_u1)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rminmax_ukernel__scalar_u2_acc2)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rminmax_ukernel__scalar_u3_acc3)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rminmax_ukernel__scalar_u4_acc2)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rminmax_ukernel__scalar_u4_acc4)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rminmax_ukernel__sse_u4)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rminmax_ukernel__sse_u8_acc2)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rminmax_ukernel__sse_u12_acc3)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rminmax_ukernel__sse_u16_acc2)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rminmax_ukernel__sse_u16_acc4)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rminmax_ukernel__wasm_u1)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rminmax_ukernel__wasm_u2_acc2)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rminmax_ukernel__wasm_u3_acc3)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rminmax_ukernel__wasm_u4_acc2)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rminmax_ukernel__wasm_u4_acc4)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rminmax_ukernel__wasmsimd_minmax_u4)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rminmax_ukernel__wasmsimd_minmax_u8_acc2)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rminmax_ukernel__wasmsimd_minmax_u12_acc3)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rminmax_ukernel__wasmsimd_minmax_u16_acc2)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rminmax_ukernel__wasmsimd_minmax_u16_acc4)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rminmax_ukernel__wasmsimd_pminmax_u4)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rminmax_ukernel__wasmsimd_pminmax_u8_acc2)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rminmax_ukernel__wasmsimd_pminmax_u12_acc3)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rminmax_ukernel__wasmsimd_pminmax_u16_acc2)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rminmax_ukernel__wasmsimd_pminmax_u16_acc4)

#define DECLARE_U8_REDUCE_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                      \
      size_t batch,                               \
      const uint8_t* input,                       \
      uint8_t* output,                            \
      const void* params);

DECLARE_U8_REDUCE_UKERNEL_FUNCTION(xnn_u8_rmax_ukernel__neon_u16)
DECLARE_U8_REDUCE_UKERNEL_FUNCTION(xnn_u8_rmax_ukernel__scalar_u2)
DECLARE_U8_REDUCE_UKERNEL_FUNCTION(xnn_u8_rmax_ukernel__sse2_u16)

#define DECLARE_F32_RSUM_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                       \
      size_t batch,                                \
      const float* input,                          \
      float* output,                               \
      const union xnn_f32_scaleminmax_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

DECLARE_F32_RSUM_UKERNEL_FUNCTION(xnn_f32_rsum_ukernel__avx_u8)
DECLARE_F32_RSUM_UKERNEL_FUNCTION(xnn_f32_rsum_ukernel__avx_u16_acc2)
DECLARE_F32_RSUM_UKERNEL_FUNCTION(xnn_f32_rsum_ukernel__avx_u24_acc3)
DECLARE_F32_RSUM_UKERNEL_FUNCTION(xnn_f32_rsum_ukernel__avx_u32_acc2)
DECLARE_F32_RSUM_UKERNEL_FUNCTION(xnn_f32_rsum_ukernel__avx_u32_acc4)
DECLARE_F32_RSUM_UKERNEL_FUNCTION(xnn_f32_rsum_ukernel__avx512f_u16)
DECLARE_F32_RSUM_UKERNEL_FUNCTION(xnn_f32_rsum_ukernel__avx512f_u32_acc2)
DECLARE_F32_RSUM_UKERNEL_FUNCTION(xnn_f32_rsum_ukernel__avx512f_u48_acc3)
DECLARE_F32_RSUM_UKERNEL_FUNCTION(xnn_f32_rsum_ukernel__avx512f_u64_acc2)
DECLARE_F32_RSUM_UKERNEL_FUNCTION(xnn_f32_rsum_ukernel__avx512f_u64_acc4)
DECLARE_F32_RSUM_UKERNEL_FUNCTION(xnn_f32_rsum_ukernel__hvx_u32)
DECLARE_F32_RSUM_UKERNEL_FUNCTION(xnn_f32_rsum_ukernel__hvx_u64_acc2)
DECLARE_F32_RSUM_UKERNEL_FUNCTION(xnn_f32_rsum_ukernel__hvx_u96_acc3)
DECLARE_F32_RSUM_UKERNEL_FUNCTION(xnn_f32_rsum_ukernel__hvx_u128_acc2)
DECLARE_F32_RSUM_UKERNEL_FUNCTION(xnn_f32_rsum_ukernel__hvx_u128_acc4)
DECLARE_F32_RSUM_UKERNEL_FUNCTION(xnn_f32_rsum_ukernel__neon_u4)
DECLARE_F32_RSUM_UKERNEL_FUNCTION(xnn_f32_rsum_ukernel__neon_u8_acc2)
DECLARE_F32_RSUM_UKERNEL_FUNCTION(xnn_f32_rsum_ukernel__neon_u12_acc3)
DECLARE_F32_RSUM_UKERNEL_FUNCTION(xnn_f32_rsum_ukernel__neon_u16_acc2)
DECLARE_F32_RSUM_UKERNEL_FUNCTION(xnn_f32_rsum_ukernel__neon_u16_acc4)
DECLARE_F32_RSUM_UKERNEL_FUNCTION(xnn_f32_rsum_ukernel__rvv_u1v)
DECLARE_F32_RSUM_UKERNEL_FUNCTION(xnn_f32_rsum_ukernel__scalar_u1)
DECLARE_F32_RSUM_UKERNEL_FUNCTION(xnn_f32_rsum_ukernel__scalar_u2_acc2)
DECLARE_F32_RSUM_UKERNEL_FUNCTION(xnn_f32_rsum_ukernel__scalar_u3_acc3)
DECLARE_F32_RSUM_UKERNEL_FUNCTION(xnn_f32_rsum_ukernel__scalar_u4_acc2)
DECLARE_F32_RSUM_UKERNEL_FUNCTION(xnn_f32_rsum_ukernel__scalar_u4_acc4)
DECLARE_F32_RSUM_UKERNEL_FUNCTION(xnn_f32_rsum_ukernel__sse_u4)
DECLARE_F32_RSUM_UKERNEL_FUNCTION(xnn_f32_rsum_ukernel__sse_u8_acc2)
DECLARE_F32_RSUM_UKERNEL_FUNCTION(xnn_f32_rsum_ukernel__sse_u12_acc3)
DECLARE_F32_RSUM_UKERNEL_FUNCTION(xnn_f32_rsum_ukernel__sse_u16_acc2)
DECLARE_F32_RSUM_UKERNEL_FUNCTION(xnn_f32_rsum_ukernel__sse_u16_acc4)
DECLARE_F32_RSUM_UKERNEL_FUNCTION(xnn_f32_rsum_ukernel__wasmsimd_u4)
DECLARE_F32_RSUM_UKERNEL_FUNCTION(xnn_f32_rsum_ukernel__wasmsimd_u8_acc2)
DECLARE_F32_RSUM_UKERNEL_FUNCTION(xnn_f32_rsum_ukernel__wasmsimd_u12_acc3)
DECLARE_F32_RSUM_UKERNEL_FUNCTION(xnn_f32_rsum_ukernel__wasmsimd_u16_acc2)
DECLARE_F32_RSUM_UKERNEL_FUNCTION(xnn_f32_rsum_ukernel__wasmsimd_u16_acc4)

#define DECLARE_QS8_RSUM_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                       \
      size_t batch,                                \
      const int8_t* input,                         \
      int32_t* output,                             \
      const union xnn_qs8_rsum_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

DECLARE_QS8_RSUM_UKERNEL_FUNCTION(xnn_qs8_rsum_ukernel__avxvnni_u32)
DECLARE_QS8_RSUM_UKERNEL_FUNCTION(xnn_qs8_rsum_ukernel__avxvnni_u64)
DECLARE_QS8_RSUM_UKERNEL_FUNCTION(xnn_qs8_rsum_ukernel__avxvnni_u64_acc2)
DECLARE_QS8_RSUM_UKERNEL_FUNCTION(xnn_qs8_rsum_ukernel__avxvnni_u128)
DECLARE_QS8_RSUM_UKERNEL_FUNCTION(xnn_qs8_rsum_ukernel__avxvnni_u128_acc2)
DECLARE_QS8_RSUM_UKERNEL_FUNCTION(xnn_qs8_rsum_ukernel__avxvnni_u128_acc4)
DECLARE_QS8_RSUM_UKERNEL_FUNCTION(xnn_qs8_rsum_ukernel__avx2_u32)
DECLARE_QS8_RSUM_UKERNEL_FUNCTION(xnn_qs8_rsum_ukernel__avx2_u64)
DECLARE_QS8_RSUM_UKERNEL_FUNCTION(xnn_qs8_rsum_ukernel__avx2_u64_acc2)
DECLARE_QS8_RSUM_UKERNEL_FUNCTION(xnn_qs8_rsum_ukernel__avx2_u128)
DECLARE_QS8_RSUM_UKERNEL_FUNCTION(xnn_qs8_rsum_ukernel__avx2_u128_acc2)
DECLARE_QS8_RSUM_UKERNEL_FUNCTION(xnn_qs8_rsum_ukernel__avx2_u128_acc4)
DECLARE_QS8_RSUM_UKERNEL_FUNCTION(xnn_qs8_rsum_ukernel__avx256skx_u32)
DECLARE_QS8_RSUM_UKERNEL_FUNCTION(xnn_qs8_rsum_ukernel__avx256skx_u64)
DECLARE_QS8_RSUM_UKERNEL_FUNCTION(xnn_qs8_rsum_ukernel__avx256skx_u64_acc2)
DECLARE_QS8_RSUM_UKERNEL_FUNCTION(xnn_qs8_rsum_ukernel__avx256skx_u128)
DECLARE_QS8_RSUM_UKERNEL_FUNCTION(xnn_qs8_rsum_ukernel__avx256skx_u128_acc2)
DECLARE_QS8_RSUM_UKERNEL_FUNCTION(xnn_qs8_rsum_ukernel__avx256skx_u128_acc4)
DECLARE_QS8_RSUM_UKERNEL_FUNCTION(xnn_qs8_rsum_ukernel__avx256vnni_u32)
DECLARE_QS8_RSUM_UKERNEL_FUNCTION(xnn_qs8_rsum_ukernel__avx256vnni_u64)
DECLARE_QS8_RSUM_UKERNEL_FUNCTION(xnn_qs8_rsum_ukernel__avx256vnni_u64_acc2)
DECLARE_QS8_RSUM_UKERNEL_FUNCTION(xnn_qs8_rsum_ukernel__avx256vnni_u128)
DECLARE_QS8_RSUM_UKERNEL_FUNCTION(xnn_qs8_rsum_ukernel__avx256vnni_u128_acc2)
DECLARE_QS8_RSUM_UKERNEL_FUNCTION(xnn_qs8_rsum_ukernel__avx256vnni_u128_acc4)
DECLARE_QS8_RSUM_UKERNEL_FUNCTION(xnn_qs8_rsum_ukernel__avx512skx_u64)
DECLARE_QS8_RSUM_UKERNEL_FUNCTION(xnn_qs8_rsum_ukernel__avx512skx_u128)
DECLARE_QS8_RSUM_UKERNEL_FUNCTION(xnn_qs8_rsum_ukernel__avx512skx_u128_acc2)
DECLARE_QS8_RSUM_UKERNEL_FUNCTION(xnn_qs8_rsum_ukernel__avx512skx_u256)
DECLARE_QS8_RSUM_UKERNEL_FUNCTION(xnn_qs8_rsum_ukernel__avx512skx_u256_acc2)
DECLARE_QS8_RSUM_UKERNEL_FUNCTION(xnn_qs8_rsum_ukernel__avx512skx_u256_acc4)
DECLARE_QS8_RSUM_UKERNEL_FUNCTION(xnn_qs8_rsum_ukernel__avx512vnni_u64)
DECLARE_QS8_RSUM_UKERNEL_FUNCTION(xnn_qs8_rsum_ukernel__avx512vnni_u128)
DECLARE_QS8_RSUM_UKERNEL_FUNCTION(xnn_qs8_rsum_ukernel__avx512vnni_u128_acc2)
DECLARE_QS8_RSUM_UKERNEL_FUNCTION(xnn_qs8_rsum_ukernel__avx512vnni_u256)
DECLARE_QS8_RSUM_UKERNEL_FUNCTION(xnn_qs8_rsum_ukernel__avx512vnni_u256_acc2)
DECLARE_QS8_RSUM_UKERNEL_FUNCTION(xnn_qs8_rsum_ukernel__avx512vnni_u256_acc4)
DECLARE_QS8_RSUM_UKERNEL_FUNCTION(xnn_qs8_rsum_ukernel__neon_u16)
DECLARE_QS8_RSUM_UKERNEL_FUNCTION(xnn_qs8_rsum_ukernel__neon_u32)
DECLARE_QS8_RSUM_UKERNEL_FUNCTION(xnn_qs8_rsum_ukernel__neon_u32_acc2)
DECLARE_QS8_RSUM_UKERNEL_FUNCTION(xnn_qs8_rsum_ukernel__neon_u64)
DECLARE_QS8_RSUM_UKERNEL_FUNCTION(xnn_qs8_rsum_ukernel__neon_u64_acc2)
DECLARE_QS8_RSUM_UKERNEL_FUNCTION(xnn_qs8_rsum_ukernel__neon_u64_acc4)
DECLARE_QS8_RSUM_UKERNEL_FUNCTION(xnn_qs8_rsum_ukernel__neondot_u16)
DECLARE_QS8_RSUM_UKERNEL_FUNCTION(xnn_qs8_rsum_ukernel__neondot_u32)
DECLARE_QS8_RSUM_UKERNEL_FUNCTION(xnn_qs8_rsum_ukernel__neondot_u32_acc2)
DECLARE_QS8_RSUM_UKERNEL_FUNCTION(xnn_qs8_rsum_ukernel__neondot_u64)
DECLARE_QS8_RSUM_UKERNEL_FUNCTION(xnn_qs8_rsum_ukernel__neondot_u64_acc2)
DECLARE_QS8_RSUM_UKERNEL_FUNCTION(xnn_qs8_rsum_ukernel__neondot_u64_acc4)
DECLARE_QS8_RSUM_UKERNEL_FUNCTION(xnn_qs8_rsum_ukernel__scalar_u1)
DECLARE_QS8_RSUM_UKERNEL_FUNCTION(xnn_qs8_rsum_ukernel__scalar_u2)
DECLARE_QS8_RSUM_UKERNEL_FUNCTION(xnn_qs8_rsum_ukernel__scalar_u4)
DECLARE_QS8_RSUM_UKERNEL_FUNCTION(xnn_qs8_rsum_ukernel__ssse3_u16)
DECLARE_QS8_RSUM_UKERNEL_FUNCTION(xnn_qs8_rsum_ukernel__ssse3_u32)
DECLARE_QS8_RSUM_UKERNEL_FUNCTION(xnn_qs8_rsum_ukernel__ssse3_u32_acc2)
DECLARE_QS8_RSUM_UKERNEL_FUNCTION(xnn_qs8_rsum_ukernel__ssse3_u64)
DECLARE_QS8_RSUM_UKERNEL_FUNCTION(xnn_qs8_rsum_ukernel__ssse3_u64_acc2)
DECLARE_QS8_RSUM_UKERNEL_FUNCTION(xnn_qs8_rsum_ukernel__ssse3_u64_acc4)

#define DECLARE_F32_RDSUM_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                        \
      size_t rows,                                  \
      size_t channels,                              \
      const float* input,                           \
      size_t input_stride,                          \
      const float* zero,                            \
      float* output,                                \
      const union xnn_f32_scaleminmax_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

DECLARE_F32_RDSUM_UKERNEL_FUNCTION(xnn_f32_rdsum_ukernel_7p7x__avx_c16)
DECLARE_F32_RDSUM_UKERNEL_FUNCTION(xnn_f32_rdsum_ukernel_7p7x__avx_c32)
DECLARE_F32_RDSUM_UKERNEL_FUNCTION(xnn_f32_rdsum_ukernel_7p7x__avx_c64)
DECLARE_F32_RDSUM_UKERNEL_FUNCTION(xnn_f32_rdsum_ukernel_7p7x__avx512f_c16)
DECLARE_F32_RDSUM_UKERNEL_FUNCTION(xnn_f32_rdsum_ukernel_7p7x__avx512f_c32)
DECLARE_F32_RDSUM_UKERNEL_FUNCTION(xnn_f32_rdsum_ukernel_7p7x__avx512f_c64)
DECLARE_F32_RDSUM_UKERNEL_FUNCTION(xnn_f32_rdsum_ukernel_7p7x__avx512f_c128)
DECLARE_F32_RDSUM_UKERNEL_FUNCTION(xnn_f32_rdsum_ukernel_7p7x__neon_c16)
DECLARE_F32_RDSUM_UKERNEL_FUNCTION(xnn_f32_rdsum_ukernel_7p7x__neon_c32)
DECLARE_F32_RDSUM_UKERNEL_FUNCTION(xnn_f32_rdsum_ukernel_7p7x__neon_c64)
DECLARE_F32_RDSUM_UKERNEL_FUNCTION(xnn_f32_rdsum_ukernel_7p7x__scalar_c4)
DECLARE_F32_RDSUM_UKERNEL_FUNCTION(xnn_f32_rdsum_ukernel_7p7x__sse_c16)
DECLARE_F32_RDSUM_UKERNEL_FUNCTION(xnn_f32_rdsum_ukernel_7p7x__sse_c32)
DECLARE_F32_RDSUM_UKERNEL_FUNCTION(xnn_f32_rdsum_ukernel_7p7x__sse_c64)
DECLARE_F32_RDSUM_UKERNEL_FUNCTION(xnn_f32_rdsum_ukernel_7p7x__wasmsimd_c16)
DECLARE_F32_RDSUM_UKERNEL_FUNCTION(xnn_f32_rdsum_ukernel_7p7x__wasmsimd_c32)
DECLARE_F32_RDSUM_UKERNEL_FUNCTION(xnn_f32_rdsum_ukernel_7p7x__wasmsimd_c64)

#define DECLARE_F16_F32ACC_RDSUM_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                               \
      size_t rows,                                         \
      size_t channels,                                     \
      const void* input,                                   \
      size_t input_stride,                                 \
      const void* zero,                                    \
      float* output,                                        \
      const union xnn_f16_f32acc_scale_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

DECLARE_F16_F32ACC_RDSUM_UKERNEL_FUNCTION(xnn_f16_f32acc_rdsum_ukernel_7p7x__avx512skx_c16)
DECLARE_F16_F32ACC_RDSUM_UKERNEL_FUNCTION(xnn_f16_f32acc_rdsum_ukernel_7p7x__avx512skx_c32)
DECLARE_F16_F32ACC_RDSUM_UKERNEL_FUNCTION(xnn_f16_f32acc_rdsum_ukernel_7p7x__avx512skx_c64)
DECLARE_F16_F32ACC_RDSUM_UKERNEL_FUNCTION(xnn_f16_f32acc_rdsum_ukernel_7p7x__avx512skx_c128)
DECLARE_F16_F32ACC_RDSUM_UKERNEL_FUNCTION(xnn_f16_f32acc_rdsum_ukernel_7p7x__f16c_c16)
DECLARE_F16_F32ACC_RDSUM_UKERNEL_FUNCTION(xnn_f16_f32acc_rdsum_ukernel_7p7x__f16c_c32)
DECLARE_F16_F32ACC_RDSUM_UKERNEL_FUNCTION(xnn_f16_f32acc_rdsum_ukernel_7p7x__f16c_c64)
DECLARE_F16_F32ACC_RDSUM_UKERNEL_FUNCTION(xnn_f16_f32acc_rdsum_ukernel_7p7x__f16c_c128)
DECLARE_F16_F32ACC_RDSUM_UKERNEL_FUNCTION(xnn_f16_f32acc_rdsum_ukernel_7p7x__neonfp16arith_c16)
DECLARE_F16_F32ACC_RDSUM_UKERNEL_FUNCTION(xnn_f16_f32acc_rdsum_ukernel_7p7x__neonfp16arith_c32)
DECLARE_F16_F32ACC_RDSUM_UKERNEL_FUNCTION(xnn_f16_f32acc_rdsum_ukernel_7p7x__neonfp16arith_c64)

#define DECLARE_QS8_RDSUM_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                        \
      size_t rows,                                  \
      size_t channels,                              \
      const int8_t* input,                          \
      size_t input_stride,                          \
      const int8_t* zero,                           \
      int32_t* output,                              \
      const union xnn_qs8_rsum_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

DECLARE_QS8_RDSUM_UKERNEL_FUNCTION(xnn_qs8_rdsum_ukernel_7p7x__avx2_c32)
DECLARE_QS8_RDSUM_UKERNEL_FUNCTION(xnn_qs8_rdsum_ukernel_7p7x__avx2_c64)
DECLARE_QS8_RDSUM_UKERNEL_FUNCTION(xnn_qs8_rdsum_ukernel_7p7x__avx512skx_c64)
DECLARE_QS8_RDSUM_UKERNEL_FUNCTION(xnn_qs8_rdsum_ukernel_7p7x__avx512skx_c128)
DECLARE_QS8_RDSUM_UKERNEL_FUNCTION(xnn_qs8_rdsum_ukernel_7p7x__neon_c16)
DECLARE_QS8_RDSUM_UKERNEL_FUNCTION(xnn_qs8_rdsum_ukernel_7p7x__neon_c32)
DECLARE_QS8_RDSUM_UKERNEL_FUNCTION(xnn_qs8_rdsum_ukernel_7p7x__neon_c64)
DECLARE_QS8_RDSUM_UKERNEL_FUNCTION(xnn_qs8_rdsum_ukernel_7p7x__scalar_c4)
DECLARE_QS8_RDSUM_UKERNEL_FUNCTION(xnn_qs8_rdsum_ukernel_7p7x__sse41_c16)
DECLARE_QS8_RDSUM_UKERNEL_FUNCTION(xnn_qs8_rdsum_ukernel_7p7x__sse41_c32)
DECLARE_QS8_RDSUM_UKERNEL_FUNCTION(xnn_qs8_rdsum_ukernel_7p7x__sse41_c64)

#ifdef __cplusplus
}  // extern "C"
#endif
