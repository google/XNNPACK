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


#define DECLARE_F32_GEMM_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                       \
      size_t mr,                                   \
      size_t nr,                                   \
      size_t k,                                    \
      const float* a,                              \
      size_t a_stride,                             \
      const float* w,                              \
      float* c,                                    \
      size_t cm_stride,                            \
      size_t cn_stride,                            \
      const union xnn_f32_output_params* params);

DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_4x2__neon_lane_ld64)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_1x8__neon_lane_ld64)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_4x8__neon_lane_ld64)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_4x8__neon_lane_ld128)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_5x8__neon_lane_ld64)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_6x8__neon_lane_ld64)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_6x8__neon_lane_ld128)

DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_4x2__neonfma_lane_ld64)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_1x8__neonfma_lane_ld64)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_4x8__neonfma_lane_ld64)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_4x8__neonfma_lane_ld128)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_5x8__neonfma_lane_ld64)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_6x8__neonfma_lane_ld64)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_6x8__neonfma_lane_ld128)

DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_1x8__neon_dup_ld64)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_4x8__neon_dup_ld64)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_4x8__neon_dup_ld128)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_6x8__neon_dup_ld64)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_6x8__neon_dup_ld128)

DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_1x8__neonfma_dup_ld64)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_4x8__neonfma_dup_ld64)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_4x8__neonfma_dup_ld128)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_6x8__neonfma_dup_ld64)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_6x8__neonfma_dup_ld128)

DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_1x8s4__neon)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_4x8s4__neon)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_6x8s4__neon)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_8x8s4__neon)

DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_1x8s4__neonfma)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_4x8s4__neonfma)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_6x8s4__neonfma)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_8x8s4__neonfma)

DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_4x8__aarch32_neon_ld64)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_4x8__aarch32_neon_cortex_a53)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_4x8__aarch32_neon_cortex_a75)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_4x8__aarch32_neon_pld_cortex_a75)

DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_1x12__aarch64_neonfma_cortex_a53)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_1x8__aarch64_neonfma_cortex_a53)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_1x8__aarch64_neonfma_cortex_a57)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_1x8__aarch64_neonfma_cortex_a75)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_4x12__aarch64_neonfma_cortex_a53)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_4x8__aarch64_neonfma_cortex_a53)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_4x8__aarch64_neonfma_cortex_a57)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_4x8__aarch64_neonfma_cortex_a75)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_4x8__aarch64_neonfma_ld128)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_4x8__aarch64_neonfma_ld64)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_5x8__aarch64_neonfma_cortex_a57)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_5x8__aarch64_neonfma_cortex_a75)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_6x8__aarch64_neonfma_cortex_a53)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_6x8__aarch64_neonfma_cortex_a73)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_6x8__aarch64_neonfma_cortex_a57)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_6x8__aarch64_neonfma_cortex_a75)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_6x8__aarch64_neonfma_ld128)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_6x8__aarch64_neonfma_ld64)

DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_1x8__sse_load1)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_4x8__sse_load1)

DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_1x8__sse_dup)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_4x8__sse_dup)

DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_1x8s4__sse)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_4x8s4__sse)

DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_4x2c4__sse)

DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_1x8__avx_broadcast)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_4x8__avx_broadcast)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_5x8__avx_broadcast)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_6x8__avx_broadcast)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_7x8__avx_broadcast)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_1x16__avx_broadcast)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_3x16__avx_broadcast)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_4x16__avx_broadcast)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_5x16__avx_broadcast)

DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_1x8__fma3_broadcast)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_4x8__fma3_broadcast)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_5x8__fma3_broadcast)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_6x8__fma3_broadcast)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_7x8__fma3_broadcast)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_8x8__fma3_broadcast)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_1x16__fma3_broadcast)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_3x16__fma3_broadcast)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_4x16__fma3_broadcast)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_5x16__fma3_broadcast)

DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_1x16s4__fma3_broadcast)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_3x16s4__fma3_broadcast)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_4x16s4__fma3_broadcast)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_5x16s4__fma3_broadcast)

DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_1x16__avx512f_broadcast)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_4x16__avx512f_broadcast)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_5x16__avx512f_broadcast)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_6x16__avx512f_broadcast)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_7x16__avx512f_broadcast)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_8x16__avx512f_broadcast)

DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_1x8__psimd_loadsplat)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_4x8__psimd_loadsplat)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_6x8__psimd_loadsplat)

DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_1x8__psimd_splat)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_4x8__psimd_splat)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_6x8__psimd_splat)

DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_1x8s4__psimd)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_4x8s4__psimd)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_6x8s4__psimd)

DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_4x2c4__psimd)

DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_4x2__wasm)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_1x4__wasm)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_2x4__wasm)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_4x4__wasm)

DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_4x2__scalar)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_1x4__scalar)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_2x4__scalar)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_4x4__scalar)

#define DECLARE_F32_GEMMINC_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                          \
      size_t mr,                                      \
      size_t nr,                                      \
      size_t k,                                       \
      const float* a,                                 \
      size_t a_stride,                                \
      const float* w,                                 \
      float* c,                                       \
      size_t cm_stride,                               \
      size_t cn_stride,                               \
      const float* acc,                               \
      const union xnn_f32_output_params* params);

DECLARE_F32_GEMMINC_UKERNEL_FUNCTION(xnn_f32_gemminc_ukernel_1x8__neon_lane_ld64)
DECLARE_F32_GEMMINC_UKERNEL_FUNCTION(xnn_f32_gemminc_ukernel_4x8__neon_lane_ld64)
DECLARE_F32_GEMMINC_UKERNEL_FUNCTION(xnn_f32_gemminc_ukernel_4x8__neon_lane_ld128)
DECLARE_F32_GEMMINC_UKERNEL_FUNCTION(xnn_f32_gemminc_ukernel_5x8__neon_lane_ld64)
DECLARE_F32_GEMMINC_UKERNEL_FUNCTION(xnn_f32_gemminc_ukernel_6x8__neon_lane_ld64)
DECLARE_F32_GEMMINC_UKERNEL_FUNCTION(xnn_f32_gemminc_ukernel_6x8__neon_lane_ld128)

DECLARE_F32_GEMMINC_UKERNEL_FUNCTION(xnn_f32_gemminc_ukernel_1x8__neonfma_lane_ld64)
DECLARE_F32_GEMMINC_UKERNEL_FUNCTION(xnn_f32_gemminc_ukernel_4x8__neonfma_lane_ld64)
DECLARE_F32_GEMMINC_UKERNEL_FUNCTION(xnn_f32_gemminc_ukernel_4x8__neonfma_lane_ld128)
DECLARE_F32_GEMMINC_UKERNEL_FUNCTION(xnn_f32_gemminc_ukernel_5x8__neonfma_lane_ld64)
DECLARE_F32_GEMMINC_UKERNEL_FUNCTION(xnn_f32_gemminc_ukernel_6x8__neonfma_lane_ld64)
DECLARE_F32_GEMMINC_UKERNEL_FUNCTION(xnn_f32_gemminc_ukernel_6x8__neonfma_lane_ld128)

DECLARE_F32_GEMMINC_UKERNEL_FUNCTION(xnn_f32_gemminc_ukernel_1x8__neon_dup_ld64)
DECLARE_F32_GEMMINC_UKERNEL_FUNCTION(xnn_f32_gemminc_ukernel_4x8__neon_dup_ld64)
DECLARE_F32_GEMMINC_UKERNEL_FUNCTION(xnn_f32_gemminc_ukernel_4x8__neon_dup_ld128)
DECLARE_F32_GEMMINC_UKERNEL_FUNCTION(xnn_f32_gemminc_ukernel_6x8__neon_dup_ld64)
DECLARE_F32_GEMMINC_UKERNEL_FUNCTION(xnn_f32_gemminc_ukernel_6x8__neon_dup_ld128)

DECLARE_F32_GEMMINC_UKERNEL_FUNCTION(xnn_f32_gemminc_ukernel_1x8__neonfma_dup_ld64)
DECLARE_F32_GEMMINC_UKERNEL_FUNCTION(xnn_f32_gemminc_ukernel_4x8__neonfma_dup_ld64)
DECLARE_F32_GEMMINC_UKERNEL_FUNCTION(xnn_f32_gemminc_ukernel_4x8__neonfma_dup_ld128)
DECLARE_F32_GEMMINC_UKERNEL_FUNCTION(xnn_f32_gemminc_ukernel_6x8__neonfma_dup_ld64)
DECLARE_F32_GEMMINC_UKERNEL_FUNCTION(xnn_f32_gemminc_ukernel_6x8__neonfma_dup_ld128)

DECLARE_F32_GEMMINC_UKERNEL_FUNCTION(xnn_f32_gemminc_ukernel_1x8s4__neon)
DECLARE_F32_GEMMINC_UKERNEL_FUNCTION(xnn_f32_gemminc_ukernel_4x8s4__neon)
DECLARE_F32_GEMMINC_UKERNEL_FUNCTION(xnn_f32_gemminc_ukernel_6x8s4__neon)
DECLARE_F32_GEMMINC_UKERNEL_FUNCTION(xnn_f32_gemminc_ukernel_8x8s4__neon)

DECLARE_F32_GEMMINC_UKERNEL_FUNCTION(xnn_f32_gemminc_ukernel_1x8s4__neonfma)
DECLARE_F32_GEMMINC_UKERNEL_FUNCTION(xnn_f32_gemminc_ukernel_4x8s4__neonfma)
DECLARE_F32_GEMMINC_UKERNEL_FUNCTION(xnn_f32_gemminc_ukernel_6x8s4__neonfma)
DECLARE_F32_GEMMINC_UKERNEL_FUNCTION(xnn_f32_gemminc_ukernel_8x8s4__neonfma)

DECLARE_F32_GEMMINC_UKERNEL_FUNCTION(xnn_f32_gemminc_ukernel_1x12__aarch64_neonfma_cortex_a53)
DECLARE_F32_GEMMINC_UKERNEL_FUNCTION(xnn_f32_gemminc_ukernel_1x8__aarch64_neonfma_cortex_a53)
DECLARE_F32_GEMMINC_UKERNEL_FUNCTION(xnn_f32_gemminc_ukernel_1x8__aarch64_neonfma_cortex_a57)
DECLARE_F32_GEMMINC_UKERNEL_FUNCTION(xnn_f32_gemminc_ukernel_1x8__aarch64_neonfma_cortex_a75)
DECLARE_F32_GEMMINC_UKERNEL_FUNCTION(xnn_f32_gemminc_ukernel_4x12__aarch64_neonfma_cortex_a53)
DECLARE_F32_GEMMINC_UKERNEL_FUNCTION(xnn_f32_gemminc_ukernel_4x8__aarch64_neonfma_cortex_a53)
DECLARE_F32_GEMMINC_UKERNEL_FUNCTION(xnn_f32_gemminc_ukernel_4x8__aarch64_neonfma_cortex_a57)
DECLARE_F32_GEMMINC_UKERNEL_FUNCTION(xnn_f32_gemminc_ukernel_4x8__aarch64_neonfma_cortex_a75)
DECLARE_F32_GEMMINC_UKERNEL_FUNCTION(xnn_f32_gemminc_ukernel_4x8__aarch64_neonfma_ld128)
DECLARE_F32_GEMMINC_UKERNEL_FUNCTION(xnn_f32_gemminc_ukernel_4x8__aarch64_neonfma_ld64)
DECLARE_F32_GEMMINC_UKERNEL_FUNCTION(xnn_f32_gemminc_ukernel_5x8__aarch64_neonfma_cortex_a57)
DECLARE_F32_GEMMINC_UKERNEL_FUNCTION(xnn_f32_gemminc_ukernel_5x8__aarch64_neonfma_cortex_a75)
DECLARE_F32_GEMMINC_UKERNEL_FUNCTION(xnn_f32_gemminc_ukernel_6x8__aarch64_neonfma_cortex_a53)
DECLARE_F32_GEMMINC_UKERNEL_FUNCTION(xnn_f32_gemminc_ukernel_6x8__aarch64_neonfma_cortex_a73)
DECLARE_F32_GEMMINC_UKERNEL_FUNCTION(xnn_f32_gemminc_ukernel_6x8__aarch64_neonfma_cortex_a57)
DECLARE_F32_GEMMINC_UKERNEL_FUNCTION(xnn_f32_gemminc_ukernel_6x8__aarch64_neonfma_cortex_a75)
DECLARE_F32_GEMMINC_UKERNEL_FUNCTION(xnn_f32_gemminc_ukernel_6x8__aarch64_neonfma_ld128)
DECLARE_F32_GEMMINC_UKERNEL_FUNCTION(xnn_f32_gemminc_ukernel_6x8__aarch64_neonfma_ld64)

DECLARE_F32_GEMMINC_UKERNEL_FUNCTION(xnn_f32_gemminc_ukernel_1x8__sse_load1)
DECLARE_F32_GEMMINC_UKERNEL_FUNCTION(xnn_f32_gemminc_ukernel_4x8__sse_load1)

DECLARE_F32_GEMMINC_UKERNEL_FUNCTION(xnn_f32_gemminc_ukernel_1x8__sse_dup)
DECLARE_F32_GEMMINC_UKERNEL_FUNCTION(xnn_f32_gemminc_ukernel_4x8__sse_dup)

DECLARE_F32_GEMMINC_UKERNEL_FUNCTION(xnn_f32_gemminc_ukernel_1x8s4__sse)
DECLARE_F32_GEMMINC_UKERNEL_FUNCTION(xnn_f32_gemminc_ukernel_4x8s4__sse)

DECLARE_F32_GEMMINC_UKERNEL_FUNCTION(xnn_f32_gemminc_ukernel_1x8__avx_broadcast)
DECLARE_F32_GEMMINC_UKERNEL_FUNCTION(xnn_f32_gemminc_ukernel_4x8__avx_broadcast)
DECLARE_F32_GEMMINC_UKERNEL_FUNCTION(xnn_f32_gemminc_ukernel_5x8__avx_broadcast)
DECLARE_F32_GEMMINC_UKERNEL_FUNCTION(xnn_f32_gemminc_ukernel_6x8__avx_broadcast)
DECLARE_F32_GEMMINC_UKERNEL_FUNCTION(xnn_f32_gemminc_ukernel_7x8__avx_broadcast)
DECLARE_F32_GEMMINC_UKERNEL_FUNCTION(xnn_f32_gemminc_ukernel_1x16__avx_broadcast)
DECLARE_F32_GEMMINC_UKERNEL_FUNCTION(xnn_f32_gemminc_ukernel_3x16__avx_broadcast)
DECLARE_F32_GEMMINC_UKERNEL_FUNCTION(xnn_f32_gemminc_ukernel_4x16__avx_broadcast)
DECLARE_F32_GEMMINC_UKERNEL_FUNCTION(xnn_f32_gemminc_ukernel_5x16__avx_broadcast)

DECLARE_F32_GEMMINC_UKERNEL_FUNCTION(xnn_f32_gemminc_ukernel_1x8__fma3_broadcast)
DECLARE_F32_GEMMINC_UKERNEL_FUNCTION(xnn_f32_gemminc_ukernel_4x8__fma3_broadcast)
DECLARE_F32_GEMMINC_UKERNEL_FUNCTION(xnn_f32_gemminc_ukernel_5x8__fma3_broadcast)
DECLARE_F32_GEMMINC_UKERNEL_FUNCTION(xnn_f32_gemminc_ukernel_6x8__fma3_broadcast)
DECLARE_F32_GEMMINC_UKERNEL_FUNCTION(xnn_f32_gemminc_ukernel_7x8__fma3_broadcast)
DECLARE_F32_GEMMINC_UKERNEL_FUNCTION(xnn_f32_gemminc_ukernel_8x8__fma3_broadcast)
DECLARE_F32_GEMMINC_UKERNEL_FUNCTION(xnn_f32_gemminc_ukernel_1x16__fma3_broadcast)
DECLARE_F32_GEMMINC_UKERNEL_FUNCTION(xnn_f32_gemminc_ukernel_3x16__fma3_broadcast)
DECLARE_F32_GEMMINC_UKERNEL_FUNCTION(xnn_f32_gemminc_ukernel_4x16__fma3_broadcast)
DECLARE_F32_GEMMINC_UKERNEL_FUNCTION(xnn_f32_gemminc_ukernel_5x16__fma3_broadcast)

DECLARE_F32_GEMMINC_UKERNEL_FUNCTION(xnn_f32_gemminc_ukernel_1x16s4__fma3_broadcast)
DECLARE_F32_GEMMINC_UKERNEL_FUNCTION(xnn_f32_gemminc_ukernel_3x16s4__fma3_broadcast)
DECLARE_F32_GEMMINC_UKERNEL_FUNCTION(xnn_f32_gemminc_ukernel_4x16s4__fma3_broadcast)
DECLARE_F32_GEMMINC_UKERNEL_FUNCTION(xnn_f32_gemminc_ukernel_5x16s4__fma3_broadcast)

DECLARE_F32_GEMMINC_UKERNEL_FUNCTION(xnn_f32_gemminc_ukernel_1x16__avx512f_broadcast)
DECLARE_F32_GEMMINC_UKERNEL_FUNCTION(xnn_f32_gemminc_ukernel_4x16__avx512f_broadcast)
DECLARE_F32_GEMMINC_UKERNEL_FUNCTION(xnn_f32_gemminc_ukernel_5x16__avx512f_broadcast)
DECLARE_F32_GEMMINC_UKERNEL_FUNCTION(xnn_f32_gemminc_ukernel_6x16__avx512f_broadcast)
DECLARE_F32_GEMMINC_UKERNEL_FUNCTION(xnn_f32_gemminc_ukernel_7x16__avx512f_broadcast)
DECLARE_F32_GEMMINC_UKERNEL_FUNCTION(xnn_f32_gemminc_ukernel_8x16__avx512f_broadcast)

DECLARE_F32_GEMMINC_UKERNEL_FUNCTION(xnn_f32_gemminc_ukernel_1x8__psimd_loadsplat)
DECLARE_F32_GEMMINC_UKERNEL_FUNCTION(xnn_f32_gemminc_ukernel_4x8__psimd_loadsplat)
DECLARE_F32_GEMMINC_UKERNEL_FUNCTION(xnn_f32_gemminc_ukernel_6x8__psimd_loadsplat)

DECLARE_F32_GEMMINC_UKERNEL_FUNCTION(xnn_f32_gemminc_ukernel_1x8__psimd_splat)
DECLARE_F32_GEMMINC_UKERNEL_FUNCTION(xnn_f32_gemminc_ukernel_4x8__psimd_splat)
DECLARE_F32_GEMMINC_UKERNEL_FUNCTION(xnn_f32_gemminc_ukernel_6x8__psimd_splat)

DECLARE_F32_GEMMINC_UKERNEL_FUNCTION(xnn_f32_gemminc_ukernel_1x8s4__psimd)
DECLARE_F32_GEMMINC_UKERNEL_FUNCTION(xnn_f32_gemminc_ukernel_4x8s4__psimd)
DECLARE_F32_GEMMINC_UKERNEL_FUNCTION(xnn_f32_gemminc_ukernel_6x8s4__psimd)

DECLARE_F32_GEMMINC_UKERNEL_FUNCTION(xnn_f32_gemminc_ukernel_1x4__wasm)
DECLARE_F32_GEMMINC_UKERNEL_FUNCTION(xnn_f32_gemminc_ukernel_2x4__wasm)
DECLARE_F32_GEMMINC_UKERNEL_FUNCTION(xnn_f32_gemminc_ukernel_4x2__wasm)
DECLARE_F32_GEMMINC_UKERNEL_FUNCTION(xnn_f32_gemminc_ukernel_4x4__wasm)

DECLARE_F32_GEMMINC_UKERNEL_FUNCTION(xnn_f32_gemminc_ukernel_1x4__scalar)
DECLARE_F32_GEMMINC_UKERNEL_FUNCTION(xnn_f32_gemminc_ukernel_2x4__scalar)
DECLARE_F32_GEMMINC_UKERNEL_FUNCTION(xnn_f32_gemminc_ukernel_4x4__scalar)

#define DECLARE_F16_GEMM_UKERNEL_FUNCTION(fn_name) \
  void fn_name(                                    \
      size_t mr,                                   \
      size_t nr,                                   \
      size_t k,                                    \
      const void* a,                               \
      size_t a_stride,                             \
      const void* w,                               \
      void* c,                                     \
      size_t cm_stride,                            \
      size_t cn_stride,                            \
      const struct xnn_f16_output_params* params);

DECLARE_F16_GEMM_UKERNEL_FUNCTION(xnn_f16_gemm_ukernel_4x8__neonfp16arith_ld64)
DECLARE_F16_GEMM_UKERNEL_FUNCTION(xnn_f16_gemm_ukernel_6x8__neonfp16arith_ld64)
DECLARE_F16_GEMM_UKERNEL_FUNCTION(xnn_f16_gemm_ukernel_8x8__neonfp16arith_ld64)

#define DECLARE_Q8_GEMM_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                      \
      size_t mr,                                  \
      size_t nr,                                  \
      size_t k,                                   \
      const uint8_t* a,                           \
      size_t a_stride,                            \
      const void* w,                              \
      uint8_t* c,                                 \
      size_t cm_stride,                           \
      size_t cn_stride,                           \
      const union xnn_q8_gemm_params* params);


DECLARE_Q8_GEMM_UKERNEL_FUNCTION(xnn_q8_gemm_ukernel_4x8__neon)
DECLARE_Q8_GEMM_UKERNEL_FUNCTION(xnn_q8_gemm_ukernel_8x8__neon)

DECLARE_Q8_GEMM_UKERNEL_FUNCTION(xnn_q8_gemm_ukernel_4x8__aarch32_neon)

DECLARE_Q8_GEMM_UKERNEL_FUNCTION(xnn_q8_gemm_ukernel_8x8__aarch64_neon)

DECLARE_Q8_GEMM_UKERNEL_FUNCTION(xnn_q8_gemm_ukernel_2x4c8__sse2)
DECLARE_Q8_GEMM_UKERNEL_FUNCTION(xnn_q8_gemm_ukernel_4x4c2__sse2)

DECLARE_Q8_GEMM_UKERNEL_FUNCTION(xnn_q8_gemm_ukernel_2x2__scalar)

#ifdef __cplusplus
}  // extern "C"
#endif
