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

#define DECLARE_F16_RSUM_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                       \
      size_t batch,                                \
      const void* input,                           \
      void* output,                                \
      const union xnn_f16_scale_params* params);

DECLARE_F16_RSUM_UKERNEL_FUNCTION(xnn_f16_rsum_ukernel__neonfp16arith_u8)
DECLARE_F16_RSUM_UKERNEL_FUNCTION(xnn_f16_rsum_ukernel__neonfp16arith_u16_acc2)
DECLARE_F16_RSUM_UKERNEL_FUNCTION(xnn_f16_rsum_ukernel__neonfp16arith_u24_acc3)
DECLARE_F16_RSUM_UKERNEL_FUNCTION(xnn_f16_rsum_ukernel__neonfp16arith_u32_acc2)
DECLARE_F16_RSUM_UKERNEL_FUNCTION(xnn_f16_rsum_ukernel__neonfp16arith_u32_acc4)

#define DECLARE_F16_F32ACC_RSUM_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                              \
      size_t batch,                                       \
      const void* input,                                  \
      void* output,                                       \
      const union xnn_f16_f32acc_scale_params* params);

DECLARE_F16_F32ACC_RSUM_UKERNEL_FUNCTION(xnn_f16_f32acc_rsum_ukernel__f16c_u8)
DECLARE_F16_F32ACC_RSUM_UKERNEL_FUNCTION(xnn_f16_f32acc_rsum_ukernel__f16c_u16_acc2)
DECLARE_F16_F32ACC_RSUM_UKERNEL_FUNCTION(xnn_f16_f32acc_rsum_ukernel__f16c_u24_acc3)
DECLARE_F16_F32ACC_RSUM_UKERNEL_FUNCTION(xnn_f16_f32acc_rsum_ukernel__f16c_u32_acc2)
DECLARE_F16_F32ACC_RSUM_UKERNEL_FUNCTION(xnn_f16_f32acc_rsum_ukernel__f16c_u32_acc4)
DECLARE_F16_F32ACC_RSUM_UKERNEL_FUNCTION(xnn_f16_f32acc_rsum_ukernel__neonfp16_u4)
DECLARE_F16_F32ACC_RSUM_UKERNEL_FUNCTION(xnn_f16_f32acc_rsum_ukernel__neonfp16_u8)
DECLARE_F16_F32ACC_RSUM_UKERNEL_FUNCTION(xnn_f16_f32acc_rsum_ukernel__neonfp16_u16_acc2)
DECLARE_F16_F32ACC_RSUM_UKERNEL_FUNCTION(xnn_f16_f32acc_rsum_ukernel__neonfp16_u24_acc3)
DECLARE_F16_F32ACC_RSUM_UKERNEL_FUNCTION(xnn_f16_f32acc_rsum_ukernel__neonfp16_u32_acc2)
DECLARE_F16_F32ACC_RSUM_UKERNEL_FUNCTION(xnn_f16_f32acc_rsum_ukernel__neonfp16_u32_acc4)

#define DECLARE_F32_REDUCE_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                         \
      size_t batch,                                  \
      const float* input,                            \
      float* output,                                 \
      const union xnn_f32_default_params* params);

DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmax_ukernel__neon_x4)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmax_ukernel__neon_x8_acc2)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmax_ukernel__neon_x12_acc3)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmax_ukernel__neon_x16_acc2)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmax_ukernel__neon_x16_acc4)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmax_ukernel__scalar_x1)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmax_ukernel__scalar_x2_acc2)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmax_ukernel__scalar_x3_acc3)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmax_ukernel__scalar_x4_acc2)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmax_ukernel__scalar_x4_acc4)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmax_ukernel__sse_x4)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmax_ukernel__sse_x8_acc2)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmax_ukernel__sse_x12_acc3)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmax_ukernel__sse_x16_acc2)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmax_ukernel__sse_x16_acc4)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmax_ukernel__wasm_x1)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmax_ukernel__wasm_x2_acc2)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmax_ukernel__wasm_x3_acc3)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmax_ukernel__wasm_x4_acc2)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmax_ukernel__wasm_x4_acc4)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmax_ukernel__wasmsimd_minmax_x4)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmax_ukernel__wasmsimd_minmax_x8_acc2)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmax_ukernel__wasmsimd_minmax_x12_acc3)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmax_ukernel__wasmsimd_minmax_x16_acc2)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmax_ukernel__wasmsimd_minmax_x16_acc4)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmax_ukernel__wasmsimd_pminmax_x4)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmax_ukernel__wasmsimd_pminmax_x8_acc2)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmax_ukernel__wasmsimd_pminmax_x12_acc3)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmax_ukernel__wasmsimd_pminmax_x16_acc2)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmax_ukernel__wasmsimd_pminmax_x16_acc4)

DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmin_ukernel__neon_x4)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmin_ukernel__neon_x8_acc2)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmin_ukernel__neon_x12_acc3)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmin_ukernel__neon_x16_acc2)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmin_ukernel__neon_x16_acc4)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmin_ukernel__scalar_x1)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmin_ukernel__scalar_x2_acc2)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmin_ukernel__scalar_x3_acc3)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmin_ukernel__scalar_x4_acc2)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmin_ukernel__scalar_x4_acc4)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmin_ukernel__sse_x4)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmin_ukernel__sse_x8_acc2)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmin_ukernel__sse_x12_acc3)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmin_ukernel__sse_x16_acc2)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmin_ukernel__sse_x16_acc4)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmin_ukernel__wasm_x1)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmin_ukernel__wasm_x2_acc2)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmin_ukernel__wasm_x3_acc3)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmin_ukernel__wasm_x4_acc2)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmin_ukernel__wasm_x4_acc4)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmin_ukernel__wasmsimd_minmax_x4)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmin_ukernel__wasmsimd_minmax_x8_acc2)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmin_ukernel__wasmsimd_minmax_x12_acc3)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmin_ukernel__wasmsimd_minmax_x16_acc2)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmin_ukernel__wasmsimd_minmax_x16_acc4)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmin_ukernel__wasmsimd_pminmax_x4)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmin_ukernel__wasmsimd_pminmax_x8_acc2)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmin_ukernel__wasmsimd_pminmax_x12_acc3)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmin_ukernel__wasmsimd_pminmax_x16_acc2)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rmin_ukernel__wasmsimd_pminmax_x16_acc4)

DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rminmax_ukernel__neon_x4)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rminmax_ukernel__neon_x8_acc2)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rminmax_ukernel__neon_x12_acc3)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rminmax_ukernel__neon_x16_acc2)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rminmax_ukernel__neon_x16_acc4)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rminmax_ukernel__scalar_x1)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rminmax_ukernel__scalar_x2_acc2)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rminmax_ukernel__scalar_x3_acc3)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rminmax_ukernel__scalar_x4_acc2)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rminmax_ukernel__scalar_x4_acc4)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rminmax_ukernel__sse_x4)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rminmax_ukernel__sse_x8_acc2)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rminmax_ukernel__sse_x12_acc3)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rminmax_ukernel__sse_x16_acc2)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rminmax_ukernel__sse_x16_acc4)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rminmax_ukernel__wasm_x1)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rminmax_ukernel__wasm_x2_acc2)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rminmax_ukernel__wasm_x3_acc3)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rminmax_ukernel__wasm_x4_acc2)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rminmax_ukernel__wasm_x4_acc4)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rminmax_ukernel__wasmsimd_minmax_x4)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rminmax_ukernel__wasmsimd_minmax_x8_acc2)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rminmax_ukernel__wasmsimd_minmax_x12_acc3)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rminmax_ukernel__wasmsimd_minmax_x16_acc2)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rminmax_ukernel__wasmsimd_minmax_x16_acc4)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rminmax_ukernel__wasmsimd_pminmax_x4)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rminmax_ukernel__wasmsimd_pminmax_x8_acc2)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rminmax_ukernel__wasmsimd_pminmax_x12_acc3)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rminmax_ukernel__wasmsimd_pminmax_x16_acc2)
DECLARE_F32_REDUCE_UKERNEL_FUNCTION(xnn_f32_rminmax_ukernel__wasmsimd_pminmax_x16_acc4)

#define DECLARE_F32_RSUM_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                       \
      size_t batch,                                \
      const float* input,                          \
      float* output,                               \
      const union xnn_f32_scale_params* params);

DECLARE_F32_RSUM_UKERNEL_FUNCTION(xnn_f32_rsum_ukernel__avx_x8)
DECLARE_F32_RSUM_UKERNEL_FUNCTION(xnn_f32_rsum_ukernel__avx_x16_acc2)
DECLARE_F32_RSUM_UKERNEL_FUNCTION(xnn_f32_rsum_ukernel__avx_x24_acc3)
DECLARE_F32_RSUM_UKERNEL_FUNCTION(xnn_f32_rsum_ukernel__avx_x32_acc2)
DECLARE_F32_RSUM_UKERNEL_FUNCTION(xnn_f32_rsum_ukernel__avx_x32_acc4)
DECLARE_F32_RSUM_UKERNEL_FUNCTION(xnn_f32_rsum_ukernel__neon_x4)
DECLARE_F32_RSUM_UKERNEL_FUNCTION(xnn_f32_rsum_ukernel__neon_x8_acc2)
DECLARE_F32_RSUM_UKERNEL_FUNCTION(xnn_f32_rsum_ukernel__neon_x12_acc3)
DECLARE_F32_RSUM_UKERNEL_FUNCTION(xnn_f32_rsum_ukernel__neon_x16_acc2)
DECLARE_F32_RSUM_UKERNEL_FUNCTION(xnn_f32_rsum_ukernel__neon_x16_acc4)
DECLARE_F32_RSUM_UKERNEL_FUNCTION(xnn_f32_rsum_ukernel__scalar_x1)
DECLARE_F32_RSUM_UKERNEL_FUNCTION(xnn_f32_rsum_ukernel__scalar_x2_acc2)
DECLARE_F32_RSUM_UKERNEL_FUNCTION(xnn_f32_rsum_ukernel__scalar_x3_acc3)
DECLARE_F32_RSUM_UKERNEL_FUNCTION(xnn_f32_rsum_ukernel__scalar_x4_acc2)
DECLARE_F32_RSUM_UKERNEL_FUNCTION(xnn_f32_rsum_ukernel__scalar_x4_acc4)
DECLARE_F32_RSUM_UKERNEL_FUNCTION(xnn_f32_rsum_ukernel__sse_x4)
DECLARE_F32_RSUM_UKERNEL_FUNCTION(xnn_f32_rsum_ukernel__sse_x8_acc2)
DECLARE_F32_RSUM_UKERNEL_FUNCTION(xnn_f32_rsum_ukernel__sse_x12_acc3)
DECLARE_F32_RSUM_UKERNEL_FUNCTION(xnn_f32_rsum_ukernel__sse_x16_acc2)
DECLARE_F32_RSUM_UKERNEL_FUNCTION(xnn_f32_rsum_ukernel__sse_x16_acc4)
DECLARE_F32_RSUM_UKERNEL_FUNCTION(xnn_f32_rsum_ukernel__wasmsimd_x4)
DECLARE_F32_RSUM_UKERNEL_FUNCTION(xnn_f32_rsum_ukernel__wasmsimd_x8_acc2)
DECLARE_F32_RSUM_UKERNEL_FUNCTION(xnn_f32_rsum_ukernel__wasmsimd_x12_acc3)
DECLARE_F32_RSUM_UKERNEL_FUNCTION(xnn_f32_rsum_ukernel__wasmsimd_x16_acc2)
DECLARE_F32_RSUM_UKERNEL_FUNCTION(xnn_f32_rsum_ukernel__wasmsimd_x16_acc4)
#ifdef __cplusplus
}  // extern "C"
#endif
