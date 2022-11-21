// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <stddef.h>
#include <stdint.h>

#include <xnnpack/common.h>

#ifdef __cplusplus
extern "C" {
#endif


#define DECLARE_F16_PRELU_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                        \
      size_t rows,                                  \
      size_t channels,                              \
      const void* input,                            \
      size_t input_stride,                          \
      const void* weights,                          \
      void* output,                                 \
      size_t output_stride);

DECLARE_F16_PRELU_UKERNEL_FUNCTION(xnn_f16_prelu_ukernel__neonfp16arith_2x8)
DECLARE_F16_PRELU_UKERNEL_FUNCTION(xnn_f16_prelu_ukernel__neonfp16arith_2x16)

DECLARE_F16_PRELU_UKERNEL_FUNCTION(xnn_f16_prelu_ukernel__f16c_2x8)
DECLARE_F16_PRELU_UKERNEL_FUNCTION(xnn_f16_prelu_ukernel__f16c_2x16)


#define DECLARE_F32_PRELU_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                        \
      size_t rows,                                  \
      size_t channels,                              \
      const float* input,                           \
      size_t input_stride,                          \
      const float* weights,                         \
      float* output,                                \
      size_t output_stride);

DECLARE_F32_PRELU_UKERNEL_FUNCTION(xnn_f32_prelu_ukernel__neon_1x4)
DECLARE_F32_PRELU_UKERNEL_FUNCTION(xnn_f32_prelu_ukernel__neon_1x8)
DECLARE_F32_PRELU_UKERNEL_FUNCTION(xnn_f32_prelu_ukernel__neon_1x16)
DECLARE_F32_PRELU_UKERNEL_FUNCTION(xnn_f32_prelu_ukernel__neon_2x4)
DECLARE_F32_PRELU_UKERNEL_FUNCTION(xnn_f32_prelu_ukernel__neon_2x8)
DECLARE_F32_PRELU_UKERNEL_FUNCTION(xnn_f32_prelu_ukernel__neon_2x16)
DECLARE_F32_PRELU_UKERNEL_FUNCTION(xnn_f32_prelu_ukernel__neon_4x4)
DECLARE_F32_PRELU_UKERNEL_FUNCTION(xnn_f32_prelu_ukernel__neon_4x8)
DECLARE_F32_PRELU_UKERNEL_FUNCTION(xnn_f32_prelu_ukernel__neon_4x16)

DECLARE_F32_PRELU_UKERNEL_FUNCTION(xnn_f32_prelu_ukernel__sse_2x4)
DECLARE_F32_PRELU_UKERNEL_FUNCTION(xnn_f32_prelu_ukernel__sse_2x8)

DECLARE_F32_PRELU_UKERNEL_FUNCTION(xnn_f32_prelu_ukernel__sse2_2x4)
DECLARE_F32_PRELU_UKERNEL_FUNCTION(xnn_f32_prelu_ukernel__sse2_2x8)

DECLARE_F32_PRELU_UKERNEL_FUNCTION(xnn_f32_prelu_ukernel__sse41_2x4)
DECLARE_F32_PRELU_UKERNEL_FUNCTION(xnn_f32_prelu_ukernel__sse41_2x8)

DECLARE_F32_PRELU_UKERNEL_FUNCTION(xnn_f32_prelu_ukernel__avx_2x8)
DECLARE_F32_PRELU_UKERNEL_FUNCTION(xnn_f32_prelu_ukernel__avx_2x16)

DECLARE_F32_PRELU_UKERNEL_FUNCTION(xnn_f32_prelu_ukernel__avx512f_2x16)
DECLARE_F32_PRELU_UKERNEL_FUNCTION(xnn_f32_prelu_ukernel__avx512f_2x32)

DECLARE_F32_PRELU_UKERNEL_FUNCTION(xnn_f32_prelu_ukernel__wasmsimd_laneselect_1x4)
DECLARE_F32_PRELU_UKERNEL_FUNCTION(xnn_f32_prelu_ukernel__wasmsimd_laneselect_1x8)
DECLARE_F32_PRELU_UKERNEL_FUNCTION(xnn_f32_prelu_ukernel__wasmsimd_laneselect_1x16)
DECLARE_F32_PRELU_UKERNEL_FUNCTION(xnn_f32_prelu_ukernel__wasmsimd_laneselect_2x4)
DECLARE_F32_PRELU_UKERNEL_FUNCTION(xnn_f32_prelu_ukernel__wasmsimd_laneselect_2x8)
DECLARE_F32_PRELU_UKERNEL_FUNCTION(xnn_f32_prelu_ukernel__wasmsimd_laneselect_2x16)
DECLARE_F32_PRELU_UKERNEL_FUNCTION(xnn_f32_prelu_ukernel__wasmsimd_laneselect_4x4)
DECLARE_F32_PRELU_UKERNEL_FUNCTION(xnn_f32_prelu_ukernel__wasmsimd_laneselect_4x8)
DECLARE_F32_PRELU_UKERNEL_FUNCTION(xnn_f32_prelu_ukernel__wasmsimd_laneselect_4x16)

DECLARE_F32_PRELU_UKERNEL_FUNCTION(xnn_f32_prelu_ukernel__wasmsimd_iminmax_1x4)
DECLARE_F32_PRELU_UKERNEL_FUNCTION(xnn_f32_prelu_ukernel__wasmsimd_iminmax_1x8)
DECLARE_F32_PRELU_UKERNEL_FUNCTION(xnn_f32_prelu_ukernel__wasmsimd_iminmax_1x16)
DECLARE_F32_PRELU_UKERNEL_FUNCTION(xnn_f32_prelu_ukernel__wasmsimd_iminmax_2x4)
DECLARE_F32_PRELU_UKERNEL_FUNCTION(xnn_f32_prelu_ukernel__wasmsimd_iminmax_2x8)
DECLARE_F32_PRELU_UKERNEL_FUNCTION(xnn_f32_prelu_ukernel__wasmsimd_iminmax_2x16)
DECLARE_F32_PRELU_UKERNEL_FUNCTION(xnn_f32_prelu_ukernel__wasmsimd_iminmax_4x4)
DECLARE_F32_PRELU_UKERNEL_FUNCTION(xnn_f32_prelu_ukernel__wasmsimd_iminmax_4x8)
DECLARE_F32_PRELU_UKERNEL_FUNCTION(xnn_f32_prelu_ukernel__wasmsimd_iminmax_4x16)

DECLARE_F32_PRELU_UKERNEL_FUNCTION(xnn_f32_prelu_ukernel__wasmrelaxedsimd_laneselect_1x4)
DECLARE_F32_PRELU_UKERNEL_FUNCTION(xnn_f32_prelu_ukernel__wasmrelaxedsimd_laneselect_1x8)
DECLARE_F32_PRELU_UKERNEL_FUNCTION(xnn_f32_prelu_ukernel__wasmrelaxedsimd_laneselect_1x16)
DECLARE_F32_PRELU_UKERNEL_FUNCTION(xnn_f32_prelu_ukernel__wasmrelaxedsimd_laneselect_2x4)
DECLARE_F32_PRELU_UKERNEL_FUNCTION(xnn_f32_prelu_ukernel__wasmrelaxedsimd_laneselect_2x8)
DECLARE_F32_PRELU_UKERNEL_FUNCTION(xnn_f32_prelu_ukernel__wasmrelaxedsimd_laneselect_2x16)
DECLARE_F32_PRELU_UKERNEL_FUNCTION(xnn_f32_prelu_ukernel__wasmrelaxedsimd_laneselect_4x4)
DECLARE_F32_PRELU_UKERNEL_FUNCTION(xnn_f32_prelu_ukernel__wasmrelaxedsimd_laneselect_4x8)
DECLARE_F32_PRELU_UKERNEL_FUNCTION(xnn_f32_prelu_ukernel__wasmrelaxedsimd_laneselect_4x16)

DECLARE_F32_PRELU_UKERNEL_FUNCTION(xnn_f32_prelu_ukernel__wasmrelaxedsimd_iminmax_1x4)
DECLARE_F32_PRELU_UKERNEL_FUNCTION(xnn_f32_prelu_ukernel__wasmrelaxedsimd_iminmax_1x8)
DECLARE_F32_PRELU_UKERNEL_FUNCTION(xnn_f32_prelu_ukernel__wasmrelaxedsimd_iminmax_1x16)
DECLARE_F32_PRELU_UKERNEL_FUNCTION(xnn_f32_prelu_ukernel__wasmrelaxedsimd_iminmax_2x4)
DECLARE_F32_PRELU_UKERNEL_FUNCTION(xnn_f32_prelu_ukernel__wasmrelaxedsimd_iminmax_2x8)
DECLARE_F32_PRELU_UKERNEL_FUNCTION(xnn_f32_prelu_ukernel__wasmrelaxedsimd_iminmax_2x16)
DECLARE_F32_PRELU_UKERNEL_FUNCTION(xnn_f32_prelu_ukernel__wasmrelaxedsimd_iminmax_4x4)
DECLARE_F32_PRELU_UKERNEL_FUNCTION(xnn_f32_prelu_ukernel__wasmrelaxedsimd_iminmax_4x8)
DECLARE_F32_PRELU_UKERNEL_FUNCTION(xnn_f32_prelu_ukernel__wasmrelaxedsimd_iminmax_4x16)

DECLARE_F32_PRELU_UKERNEL_FUNCTION(xnn_f32_prelu_ukernel__wasm_2x1)
DECLARE_F32_PRELU_UKERNEL_FUNCTION(xnn_f32_prelu_ukernel__wasm_2x4)

DECLARE_F32_PRELU_UKERNEL_FUNCTION(xnn_f32_prelu_ukernel__scalar_2x1)
DECLARE_F32_PRELU_UKERNEL_FUNCTION(xnn_f32_prelu_ukernel__scalar_2x4)


#ifdef __cplusplus
}  // extern "C"
#endif
