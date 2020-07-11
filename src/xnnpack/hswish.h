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


#define DECLARE_F16_HSWISH_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                         \
      size_t n,                                      \
      const void* x,                                 \
      void* y,                                       \
      const struct xnn_f16_hswish_params* params);

DECLARE_F16_HSWISH_UKERNEL_FUNCTION(xnn_f16_hswish_ukernel__neonfp16arith_x8)
DECLARE_F16_HSWISH_UKERNEL_FUNCTION(xnn_f16_hswish_ukernel__neonfp16arith_x16)

#define DECLARE_F32_HSWISH_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                         \
      size_t n,                                      \
      const float* x,                                \
      float* y,                                      \
      const union xnn_f32_hswish_params* params);

DECLARE_F32_HSWISH_UKERNEL_FUNCTION(xnn_f32_hswish_ukernel__neon_x4)
DECLARE_F32_HSWISH_UKERNEL_FUNCTION(xnn_f32_hswish_ukernel__neon_x8)
DECLARE_F32_HSWISH_UKERNEL_FUNCTION(xnn_f32_hswish_ukernel__neon_x16)

DECLARE_F32_HSWISH_UKERNEL_FUNCTION(xnn_f32_hswish_ukernel__sse_x4)
DECLARE_F32_HSWISH_UKERNEL_FUNCTION(xnn_f32_hswish_ukernel__sse_x8)

DECLARE_F32_HSWISH_UKERNEL_FUNCTION(xnn_f32_hswish_ukernel__avx_x8)
DECLARE_F32_HSWISH_UKERNEL_FUNCTION(xnn_f32_hswish_ukernel__avx_x16)

DECLARE_F32_HSWISH_UKERNEL_FUNCTION(xnn_f32_hswish_ukernel__fma3_x8)
DECLARE_F32_HSWISH_UKERNEL_FUNCTION(xnn_f32_hswish_ukernel__fma3_x16)

DECLARE_F32_HSWISH_UKERNEL_FUNCTION(xnn_f32_hswish_ukernel__avx512f_x16)
DECLARE_F32_HSWISH_UKERNEL_FUNCTION(xnn_f32_hswish_ukernel__avx512f_x32)

DECLARE_F32_HSWISH_UKERNEL_FUNCTION(xnn_f32_hswish_ukernel__wasmsimd_x4)
DECLARE_F32_HSWISH_UKERNEL_FUNCTION(xnn_f32_hswish_ukernel__wasmsimd_x8)
DECLARE_F32_HSWISH_UKERNEL_FUNCTION(xnn_f32_hswish_ukernel__wasmsimd_x16)

DECLARE_F32_HSWISH_UKERNEL_FUNCTION(xnn_f32_hswish_ukernel__wasm_x1)
DECLARE_F32_HSWISH_UKERNEL_FUNCTION(xnn_f32_hswish_ukernel__wasm_x2)
DECLARE_F32_HSWISH_UKERNEL_FUNCTION(xnn_f32_hswish_ukernel__wasm_x4)

DECLARE_F32_HSWISH_UKERNEL_FUNCTION(xnn_f32_hswish_ukernel__scalar_x1)
DECLARE_F32_HSWISH_UKERNEL_FUNCTION(xnn_f32_hswish_ukernel__scalar_x2)
DECLARE_F32_HSWISH_UKERNEL_FUNCTION(xnn_f32_hswish_ukernel__scalar_x4)


#ifdef __cplusplus
}  // extern "C"
#endif
