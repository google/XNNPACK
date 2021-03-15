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


#define DECLARE_F32_IGEMM_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                        \
      size_t mr,                                    \
      size_t nr,                                    \
      size_t kc,                                    \
      size_t ks,                                    \
      const float** a,                              \
      const float* w,                               \
      float* c,                                     \
      size_t cm_stride,                             \
      size_t cn_stride,                             \
      size_t a_offset,                              \
      const float* zero,                            \
      const union xnn_f32_default_params* params);

#define DECLARE_F32_IGEMM_RELU_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                             \
      size_t mr,                                         \
      size_t nr,                                         \
      size_t kc,                                         \
      size_t ks,                                         \
      const float** a,                                   \
      const float* w,                                    \
      float* c,                                          \
      size_t cm_stride,                                  \
      size_t cn_stride,                                  \
      size_t a_offset,                                   \
      const float* zero,                                 \
      const union xnn_f32_relu_params* params);

#define DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                               \
      size_t mr,                                           \
      size_t nr,                                           \
      size_t kc,                                           \
      size_t ks,                                           \
      const float** a,                                     \
      const float* w,                                      \
      float* c,                                            \
      size_t cm_stride,                                    \
      size_t cn_stride,                                    \
      size_t a_offset,                                     \
      const float* zero,                                   \
      const union xnn_f32_minmax_params* params);

DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_4x2__neon_lane_ld64)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_4x4__neon_lane_ld64)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_1x8__neon_lane_ld64)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_4x8__neon_lane_ld64)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_4x8__neon_lane_ld128)

DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_4x2__neonfma_lane_ld64)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_4x4__neonfma_lane_ld64)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_1x8__neonfma_lane_ld64)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_4x8__neonfma_lane_ld64)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_4x8__neonfma_lane_ld128)

DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_1x8__neon_dup_ld64)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_4x8__neon_dup_ld64)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_4x8__neon_dup_ld128)

DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_1x8__neonfma_dup_ld64)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_4x8__neonfma_dup_ld64)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_4x8__neonfma_dup_ld128)

DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_1x8s4__neon)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_4x8s4__neon)

DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_1x8s4__neonfma)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_4x8s4__neonfma)

DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_6x8__neon_dup_ld64)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_6x8__neon_lane_ld64)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_6x8__neonfma_dup_ld64)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_6x8__neonfma_lane_ld64)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_6x8__neon_lane_ld128)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_6x8__neonfma_lane_ld128)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_6x8__neon_dup_ld128)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_6x8__neonfma_dup_ld128)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_6x8s4__neon)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_6x8s4__neonfma)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_8x8s4__neon)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_8x8s4__neonfma)

DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_1x8__aarch64_neonfma_cortex_a53)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_1x8__aarch64_neonfma_cortex_a57)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_1x8__aarch64_neonfma_cortex_a75)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_4x8__aarch64_neonfma_cortex_a53)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_4x8__aarch64_neonfma_cortex_a55)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_4x8__aarch64_neonfma_cortex_a57)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_4x8__aarch64_neonfma_cortex_a75)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_5x8__aarch64_neonfma_cortex_a57)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_5x8__aarch64_neonfma_cortex_a75)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_6x8__aarch64_neonfma_cortex_a53)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_6x8__aarch64_neonfma_cortex_a55)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_6x8__aarch64_neonfma_cortex_a73)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_6x8__aarch64_neonfma_cortex_a57)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_6x8__aarch64_neonfma_cortex_a75)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_1x12__aarch64_neonfma_cortex_a53)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_4x12__aarch64_neonfma_cortex_a53)

DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_4x8__aarch32_neon_ld64)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_4x8__aarch32_neon_cortex_a7)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_4x8__aarch32_neon_cortex_a75)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_4x8__aarch32_neon_pld_cortex_a75)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_4x8__aarch32_neon_cortex_a53)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_4x8__aarch32_neon_cortex_a55)

DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_1x8__sse_load1)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_3x8__sse_load1)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_4x8__sse_load1)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_5x8__sse_load1)

DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_1x8__sse_dup)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_3x8__sse_dup)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_4x8__sse_dup)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_5x8__sse_dup)

DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_1x8s4__sse)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_3x8s4__sse)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_4x8s4__sse)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_5x8s4__sse)

DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_4x2c4__sse)

DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_1x8__sse2_dup)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_3x8__sse2_dup)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_4x8__sse2_dup)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_5x8__sse2_dup)

DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_1x8__avx_broadcast)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_4x8__avx_broadcast)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_5x8__avx_broadcast)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_6x8__avx_broadcast)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_7x8__avx_broadcast)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_1x16__avx_broadcast)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_3x16__avx_broadcast)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_4x16__avx_broadcast)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_5x16__avx_broadcast)

DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_1x8__fma3_broadcast)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_4x8__fma3_broadcast)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_5x8__fma3_broadcast)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_6x8__fma3_broadcast)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_7x8__fma3_broadcast)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_8x8__fma3_broadcast)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_1x16__fma3_broadcast)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_3x16__fma3_broadcast)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_4x16__fma3_broadcast)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_5x16__fma3_broadcast)

DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_1x16s4__fma3_broadcast)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_3x16s4__fma3_broadcast)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_4x16s4__fma3_broadcast)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_5x16s4__fma3_broadcast)

DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_1x16__avx512f_broadcast)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_4x16__avx512f_broadcast)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_5x16__avx512f_broadcast)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_6x16__avx512f_broadcast)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_7x16__avx512f_broadcast)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_8x16__avx512f_broadcast)

DECLARE_F32_IGEMM_UKERNEL_FUNCTION(xnn_f32_igemm_ukernel_1x8__wasmsimd_splat)
DECLARE_F32_IGEMM_UKERNEL_FUNCTION(xnn_f32_igemm_ukernel_4x8__wasmsimd_splat)
DECLARE_F32_IGEMM_UKERNEL_FUNCTION(xnn_f32_igemm_ukernel_5x8__wasmsimd_splat)

DECLARE_F32_IGEMM_RELU_UKERNEL_FUNCTION(xnn_f32_igemm_relu_ukernel_1x8__wasmsimd_splat)
DECLARE_F32_IGEMM_RELU_UKERNEL_FUNCTION(xnn_f32_igemm_relu_ukernel_4x8__wasmsimd_splat)
DECLARE_F32_IGEMM_RELU_UKERNEL_FUNCTION(xnn_f32_igemm_relu_ukernel_5x8__wasmsimd_splat)

DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_1x8__wasmsimd_arm_loadsplat)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_3x8__wasmsimd_arm_loadsplat)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_4x8__wasmsimd_arm_loadsplat)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_5x8__wasmsimd_arm_loadsplat)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_6x8__wasmsimd_arm_loadsplat)

DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_1x8__wasmsimd_x86_loadsplat)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_3x8__wasmsimd_x86_loadsplat)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_4x8__wasmsimd_x86_loadsplat)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_5x8__wasmsimd_x86_loadsplat)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_6x8__wasmsimd_x86_loadsplat)

DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_1x8__wasmsimd_arm_splat)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_3x8__wasmsimd_arm_splat)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_4x8__wasmsimd_arm_splat)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_5x8__wasmsimd_arm_splat)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_6x8__wasmsimd_arm_splat)

DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_1x8__wasmsimd_x86_splat)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_3x8__wasmsimd_x86_splat)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_4x8__wasmsimd_x86_splat)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_5x8__wasmsimd_x86_splat)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_6x8__wasmsimd_x86_splat)

DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_1x8s4__wasmsimd_arm)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_3x8s4__wasmsimd_arm)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_4x8s4__wasmsimd_arm)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_5x8s4__wasmsimd_arm)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_6x8s4__wasmsimd_arm)

DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_1x8s4__wasmsimd_x86)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_3x8s4__wasmsimd_x86)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_4x8s4__wasmsimd_x86)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_5x8s4__wasmsimd_x86)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_6x8s4__wasmsimd_x86)

DECLARE_F32_IGEMM_UKERNEL_FUNCTION(xnn_f32_igemm_ukernel_4x2c4__wasmsimd)

DECLARE_F32_IGEMM_RELU_UKERNEL_FUNCTION(xnn_f32_igemm_relu_ukernel_4x2c4__wasmsimd)

DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_4x2c4__wasmsimd_arm)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_4x2c4__wasmsimd_x86)

DECLARE_F32_IGEMM_UKERNEL_FUNCTION(xnn_f32_igemm_ukernel_1x4__wasm)
DECLARE_F32_IGEMM_UKERNEL_FUNCTION(xnn_f32_igemm_ukernel_2x4__wasm)
DECLARE_F32_IGEMM_UKERNEL_FUNCTION(xnn_f32_igemm_ukernel_4x2__wasm)
DECLARE_F32_IGEMM_UKERNEL_FUNCTION(xnn_f32_igemm_ukernel_4x4__wasm)

DECLARE_F32_IGEMM_RELU_UKERNEL_FUNCTION(xnn_f32_igemm_relu_ukernel_1x4__wasm)
DECLARE_F32_IGEMM_RELU_UKERNEL_FUNCTION(xnn_f32_igemm_relu_ukernel_2x4__wasm)
DECLARE_F32_IGEMM_RELU_UKERNEL_FUNCTION(xnn_f32_igemm_relu_ukernel_4x2__wasm)
DECLARE_F32_IGEMM_RELU_UKERNEL_FUNCTION(xnn_f32_igemm_relu_ukernel_4x4__wasm)

DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_1x4__wasm)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_2x4__wasm)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_4x2__wasm)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_4x4__wasm)

DECLARE_F32_IGEMM_UKERNEL_FUNCTION(xnn_f32_igemm_ukernel_4x2__scalar)
DECLARE_F32_IGEMM_UKERNEL_FUNCTION(xnn_f32_igemm_ukernel_1x4__scalar)
DECLARE_F32_IGEMM_UKERNEL_FUNCTION(xnn_f32_igemm_ukernel_2x4__scalar)
DECLARE_F32_IGEMM_UKERNEL_FUNCTION(xnn_f32_igemm_ukernel_4x4__scalar)

DECLARE_F32_IGEMM_RELU_UKERNEL_FUNCTION(xnn_f32_igemm_relu_ukernel_4x2__scalar)
DECLARE_F32_IGEMM_RELU_UKERNEL_FUNCTION(xnn_f32_igemm_relu_ukernel_1x4__scalar)
DECLARE_F32_IGEMM_RELU_UKERNEL_FUNCTION(xnn_f32_igemm_relu_ukernel_2x4__scalar)
DECLARE_F32_IGEMM_RELU_UKERNEL_FUNCTION(xnn_f32_igemm_relu_ukernel_4x4__scalar)

DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_4x2__scalar)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_1x4__scalar)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_2x4__scalar)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_4x4__scalar)


#define DECLARE_F16_IGEMM_MINMAX_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                               \
      size_t mr,                                           \
      size_t nr,                                           \
      size_t kc,                                           \
      size_t ks,                                           \
      const void** a,                                      \
      const void* w,                                       \
      void* c,                                             \
      size_t cm_stride,                                    \
      size_t cn_stride,                                    \
      size_t a_offset,                                     \
      const void* zero,                                    \
      const struct xnn_f16_scaleminmax_params* params);

DECLARE_F16_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f16_igemm_minmax_ukernel_1x8__neonfp16arith_ld64)
DECLARE_F16_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f16_igemm_minmax_ukernel_4x8__neonfp16arith_ld64)
DECLARE_F16_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f16_igemm_minmax_ukernel_6x8__neonfp16arith_ld64)
DECLARE_F16_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f16_igemm_minmax_ukernel_8x8__neonfp16arith_ld64)
DECLARE_F16_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f16_igemm_minmax_ukernel_1x16__neonfp16arith_ld64)
DECLARE_F16_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f16_igemm_minmax_ukernel_4x16__neonfp16arith_ld64)
DECLARE_F16_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f16_igemm_minmax_ukernel_6x16__neonfp16arith_ld64)
DECLARE_F16_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f16_igemm_minmax_ukernel_8x16__neonfp16arith_ld64)


#define DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                               \
      size_t mr,                                           \
      size_t nr,                                           \
      size_t kc,                                           \
      size_t ks,                                           \
      const uint8_t** a,                                   \
      const void* w,                                       \
      uint8_t* c,                                          \
      size_t cm_stride,                                    \
      size_t cn_stride,                                    \
      size_t a_offset,                                     \
      const uint8_t* zero,                                 \
      const union xnn_qu8_gemm_params* params);

DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_ukernel_4x8__neon)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_ukernel_8x8__neon)

DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_ukernel_4x4c2__sse2)

DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_ukernel_2x2__scalar)


#define DECLARE_QS8_IGEMM_MINMAX_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                               \
      size_t mr,                                           \
      size_t nr,                                           \
      size_t kc,                                           \
      size_t ks,                                           \
      const int8_t** a,                                    \
      const void* w,                                       \
      int8_t* c,                                           \
      size_t cm_stride,                                    \
      size_t cn_stride,                                    \
      size_t a_offset,                                     \
      const int8_t* zero,                                  \
      const union xnn_qs8_gemm_params* params);

DECLARE_QS8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_igemm_minmax_ukernel_1x8__neon_mlal_lane)
DECLARE_QS8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_igemm_minmax_ukernel_2x8__neon_mlal_lane)
DECLARE_QS8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_igemm_minmax_ukernel_3x8__neon_mlal_lane)
DECLARE_QS8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_igemm_minmax_ukernel_4x8__neon_mlal_lane)

DECLARE_QS8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_igemm_minmax_ukernel_1x16__neon_mlal_lane)
DECLARE_QS8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_igemm_minmax_ukernel_2x16__neon_mlal_lane)
DECLARE_QS8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_igemm_minmax_ukernel_3x16__neon_mlal_lane)
DECLARE_QS8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_igemm_minmax_ukernel_4x16__neon_mlal_lane)

DECLARE_QS8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_igemm_minmax_ukernel_1x8__neon_mull_addw_dup)
DECLARE_QS8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_igemm_minmax_ukernel_2x8__neon_mull_addw_dup)
DECLARE_QS8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_igemm_minmax_ukernel_3x8__neon_mull_addw_dup)
DECLARE_QS8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_igemm_minmax_ukernel_4x8__neon_mull_addw_dup)

DECLARE_QS8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_igemm_minmax_ukernel_1x16__neon_mull_addw_dup)
DECLARE_QS8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_igemm_minmax_ukernel_2x16__neon_mull_addw_dup)
DECLARE_QS8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_igemm_minmax_ukernel_3x16__neon_mull_addw_dup)
DECLARE_QS8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_igemm_minmax_ukernel_4x16__neon_mull_addw_dup)

DECLARE_QS8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_igemm_minmax_ukernel_1x8c8__neon_mull_padal)
DECLARE_QS8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_igemm_minmax_ukernel_2x8c8__neon_mull_padal)
DECLARE_QS8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_igemm_minmax_ukernel_3x8c8__neon_mull_padal)
DECLARE_QS8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_igemm_minmax_ukernel_4x8c8__neon_mull_padal)

DECLARE_QS8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_igemm_minmax_ukernel_1x16c8__neon_mull_padal)
DECLARE_QS8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_igemm_minmax_ukernel_2x16c8__neon_mull_padal)
DECLARE_QS8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_igemm_minmax_ukernel_3x16c8__neon_mull_padal)
DECLARE_QS8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_igemm_minmax_ukernel_4x16c8__neon_mull_padal)

DECLARE_QS8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_igemm_minmax_ukernel_1x8c8__neon_mlal_padal)
DECLARE_QS8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_igemm_minmax_ukernel_2x8c8__neon_mlal_padal)
DECLARE_QS8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_igemm_minmax_ukernel_3x8c8__neon_mlal_padal)
DECLARE_QS8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_igemm_minmax_ukernel_4x8c8__neon_mlal_padal)

DECLARE_QS8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_igemm_minmax_ukernel_1x16c8__neon_mlal_padal)
DECLARE_QS8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_igemm_minmax_ukernel_2x16c8__neon_mlal_padal)
DECLARE_QS8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_igemm_minmax_ukernel_3x16c8__neon_mlal_padal)
DECLARE_QS8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_igemm_minmax_ukernel_4x16c8__neon_mlal_padal)

DECLARE_QS8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_igemm_minmax_ukernel_1x8c16__neon_mlal_padal)
DECLARE_QS8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_igemm_minmax_ukernel_2x8c16__neon_mlal_padal)
DECLARE_QS8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_igemm_minmax_ukernel_3x8c16__neon_mlal_padal)
DECLARE_QS8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_igemm_minmax_ukernel_4x8c16__neon_mlal_padal)

DECLARE_QS8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_igemm_minmax_ukernel_1x16c16__neon_mlal_padal)
DECLARE_QS8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_igemm_minmax_ukernel_2x16c16__neon_mlal_padal)
DECLARE_QS8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_igemm_minmax_ukernel_3x16c16__neon_mlal_padal)
DECLARE_QS8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_igemm_minmax_ukernel_4x16c16__neon_mlal_padal)

DECLARE_QS8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_igemm_minmax_ukernel_1x8c2__neon_mull_padal_dup)
DECLARE_QS8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_igemm_minmax_ukernel_2x8c2__neon_mull_padal_dup)
DECLARE_QS8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_igemm_minmax_ukernel_3x8c2__neon_mull_padal_dup)
DECLARE_QS8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_igemm_minmax_ukernel_4x8c2__neon_mull_padal_dup)

DECLARE_QS8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_igemm_minmax_ukernel_1x16c2__neon_mull_padal_dup)
DECLARE_QS8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_igemm_minmax_ukernel_2x16c2__neon_mull_padal_dup)
DECLARE_QS8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_igemm_minmax_ukernel_3x16c2__neon_mull_padal_dup)
DECLARE_QS8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_igemm_minmax_ukernel_4x16c2__neon_mull_padal_dup)

DECLARE_QS8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_igemm_minmax_ukernel_1x8c2__neon_mlal_padal_dup)
DECLARE_QS8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_igemm_minmax_ukernel_2x8c2__neon_mlal_padal_dup)
DECLARE_QS8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_igemm_minmax_ukernel_3x8c2__neon_mlal_padal_dup)
DECLARE_QS8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_igemm_minmax_ukernel_4x8c2__neon_mlal_padal_dup)

DECLARE_QS8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_igemm_minmax_ukernel_1x16c2__neon_mlal_padal_dup)
DECLARE_QS8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_igemm_minmax_ukernel_2x16c2__neon_mlal_padal_dup)
DECLARE_QS8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_igemm_minmax_ukernel_3x16c2__neon_mlal_padal_dup)
DECLARE_QS8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_igemm_minmax_ukernel_4x16c2__neon_mlal_padal_dup)

DECLARE_QS8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_igemm_minmax_ukernel_1x8c4__neondot)
DECLARE_QS8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_igemm_minmax_ukernel_4x8c4__neondot)
DECLARE_QS8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_igemm_minmax_ukernel_6x8c4__neondot)
DECLARE_QS8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_igemm_minmax_ukernel_8x8c4__neondot)

DECLARE_QS8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_igemm_minmax_ukernel_1x16c4__neondot)
DECLARE_QS8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_igemm_minmax_ukernel_4x16c4__neondot)
DECLARE_QS8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_igemm_minmax_ukernel_6x16c4__neondot)
DECLARE_QS8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_igemm_minmax_ukernel_8x16c4__neondot)

DECLARE_QS8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_igemm_minmax_ukernel_2x8c8__aarch64_neon_mlal_padal)
DECLARE_QS8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_igemm_minmax_ukernel_2x8c16__aarch64_neon_mlal_padal)

DECLARE_QS8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_igemm_minmax_ukernel_4x16c4__aarch64_neondot_ld64)
DECLARE_QS8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_igemm_minmax_ukernel_4x16c4__aarch64_neondot_cortex_a55)

DECLARE_QS8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_igemm_minmax_ukernel_1x4c2__sse2_ld64)
DECLARE_QS8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_igemm_minmax_ukernel_4x4c2__sse2_ld64)

DECLARE_QS8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_igemm_minmax_ukernel_1x4c2__ssse3_ld64)
DECLARE_QS8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_igemm_minmax_ukernel_4x4c2__ssse3_ld64)

DECLARE_QS8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_igemm_minmax_ukernel_1x4c2__sse41_ld64)
DECLARE_QS8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_igemm_minmax_ukernel_4x4c2__sse41_ld64)

DECLARE_QS8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_igemm_minmax_ukernel_1x4c2__xop_ld64)
DECLARE_QS8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_igemm_minmax_ukernel_4x4c2__xop_ld64)

DECLARE_QS8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_igemm_minmax_ukernel_1x4c2__sse2_ld128)
DECLARE_QS8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_igemm_minmax_ukernel_4x4c2__sse2_ld128)

DECLARE_QS8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_igemm_minmax_ukernel_1x4c2__ssse3_ld128)
DECLARE_QS8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_igemm_minmax_ukernel_4x4c2__ssse3_ld128)

DECLARE_QS8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_igemm_minmax_ukernel_1x4c2__sse41_ld128)
DECLARE_QS8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_igemm_minmax_ukernel_4x4c2__sse41_ld128)

DECLARE_QS8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_igemm_minmax_ukernel_1x4c2__xop_ld128)
DECLARE_QS8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_igemm_minmax_ukernel_4x4c2__xop_ld128)

DECLARE_QS8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_igemm_minmax_ukernel_1x4c8__sse2_ld64)
DECLARE_QS8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_igemm_minmax_ukernel_2x4c8__sse2_ld64)
DECLARE_QS8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_igemm_minmax_ukernel_3x4c8__sse2_ld64)

DECLARE_QS8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_igemm_minmax_ukernel_1x4c8__ssse3_ld64)
DECLARE_QS8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_igemm_minmax_ukernel_2x4c8__ssse3_ld64)
DECLARE_QS8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_igemm_minmax_ukernel_3x4c8__ssse3_ld64)

DECLARE_QS8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_igemm_minmax_ukernel_1x4c8__sse41_ld64)
DECLARE_QS8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_igemm_minmax_ukernel_2x4c8__sse41_ld64)
DECLARE_QS8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_igemm_minmax_ukernel_3x4c8__sse41_ld64)

DECLARE_QS8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_igemm_minmax_ukernel_1x4c8__xop_ld64)
DECLARE_QS8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_igemm_minmax_ukernel_2x4c8__xop_ld64)
DECLARE_QS8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_igemm_minmax_ukernel_3x4c8__xop_ld64)

DECLARE_QS8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_igemm_minmax_ukernel_1x4c8__sse2_ld128)
DECLARE_QS8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_igemm_minmax_ukernel_2x4c8__sse2_ld128)
DECLARE_QS8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_igemm_minmax_ukernel_3x4c8__sse2_ld128)

DECLARE_QS8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_igemm_minmax_ukernel_1x4c8__ssse3_ld128)
DECLARE_QS8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_igemm_minmax_ukernel_2x4c8__ssse3_ld128)
DECLARE_QS8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_igemm_minmax_ukernel_3x4c8__ssse3_ld128)

DECLARE_QS8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_igemm_minmax_ukernel_1x4c8__sse41_ld128)
DECLARE_QS8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_igemm_minmax_ukernel_2x4c8__sse41_ld128)
DECLARE_QS8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_igemm_minmax_ukernel_3x4c8__sse41_ld128)

DECLARE_QS8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_igemm_minmax_ukernel_1x4c8__xop_ld128)
DECLARE_QS8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_igemm_minmax_ukernel_2x4c8__xop_ld128)
DECLARE_QS8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_igemm_minmax_ukernel_3x4c8__xop_ld128)

DECLARE_QS8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_igemm_minmax_ukernel_1x8c8__avx2)
DECLARE_QS8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_igemm_minmax_ukernel_2x8c8__avx2)
DECLARE_QS8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_igemm_minmax_ukernel_3x8c8__avx2)

DECLARE_QS8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_igemm_minmax_ukernel_1x16c8__avx512skx)
DECLARE_QS8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_igemm_minmax_ukernel_2x16c8__avx512skx)
DECLARE_QS8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_igemm_minmax_ukernel_3x16c8__avx512skx)
DECLARE_QS8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_igemm_minmax_ukernel_4x16c8__avx512skx)

DECLARE_QS8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_igemm_minmax_ukernel_1x4c8__wasmsimd_ld64)
DECLARE_QS8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_igemm_minmax_ukernel_2x4c8__wasmsimd_ld64)
DECLARE_QS8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_igemm_minmax_ukernel_3x4c8__wasmsimd_ld64)

DECLARE_QS8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_igemm_minmax_ukernel_1x4c8__wasmsimd_ld128)
DECLARE_QS8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_igemm_minmax_ukernel_2x4c8__wasmsimd_ld128)
DECLARE_QS8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_igemm_minmax_ukernel_3x4c8__wasmsimd_ld128)


#ifdef __cplusplus
}  // extern "C"
#endif
