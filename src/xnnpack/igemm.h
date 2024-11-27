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

#include "xnnpack/common.h"
#include "xnnpack/microparams.h"

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
      const struct xnn_f32_default_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

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
      const struct xnn_f32_relu_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

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
      const union xnn_f32_minmax_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_1x8__neon_lane_ld64)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_4x2__neon_lane_ld64)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_4x4__neon_lane_ld64)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_4x8__neon_lane_ld64)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_4x8__neon_lane_ld128)

DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_1x8__aarch64_neonfma_lane_ld64)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_4x2__aarch64_neonfma_lane_ld64)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_4x4__aarch64_neonfma_lane_ld64)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_4x8__aarch64_neonfma_lane_ld64)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_4x8__aarch64_neonfma_lane_ld128)

DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_1x16__neon_lane_ld128)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_2x16__neon_lane_ld128)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_3x16__neon_lane_ld128)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_4x16__neon_lane_ld128)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_5x16__neon_lane_ld128)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_6x16__neon_lane_ld128)

DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_1x16__aarch64_neonfma_lane_ld128)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_2x16__aarch64_neonfma_lane_ld128)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_3x16__aarch64_neonfma_lane_ld128)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_4x16__aarch64_neonfma_lane_ld128)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_5x16__aarch64_neonfma_lane_ld128)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_6x16__aarch64_neonfma_lane_ld128)

DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_1x8__neon_lane_ld128)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_5x8__neon_lane_ld64)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_6x2__neon_lane_ld64)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_6x8__neon_lane_ld64)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_6x8__neon_lane_ld128)

DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_1x8__aarch64_neonfma_lane_ld128)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_5x8__aarch64_neonfma_lane_ld64)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_6x2__aarch64_neonfma_lane_ld64)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_6x8__aarch64_neonfma_lane_ld64)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_6x8__aarch64_neonfma_lane_ld128)

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
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_6x8__neon_dup_ld128)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_6x8__neonfma_dup_ld64)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_6x8__neonfma_dup_ld128)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_6x8s4__neon)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_6x8s4__neonfma)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_8x8s4__neon)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_8x8s4__neonfma)

DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a53)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a53_prfm)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a75)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a75_prfm)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_prfm)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_1x12__asm_aarch64_neonfma_cortex_a53)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_4x2__asm_aarch64_neonfma_cortex_a75)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_4x2__asm_aarch64_neonfma_cortex_a75_prfm)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_4x2__asm_aarch64_neonfma_ld64)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a53)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a53_prfm)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a55)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a75)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a75_prfm)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld64)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld128)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_4x12__asm_aarch64_neonfma_cortex_a53)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_5x8__asm_aarch64_neonfma_cortex_a75)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_5x8__asm_aarch64_neonfma_cortex_a75_prfm)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_6x8__asm_aarch64_neonfma_cortex_a53)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_6x8__asm_aarch64_neonfma_cortex_a53_prfm)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_6x8__asm_aarch64_neonfma_cortex_a55)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_6x8__asm_aarch64_neonfma_cortex_a73)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_6x8__asm_aarch64_neonfma_cortex_a75)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_6x8__asm_aarch64_neonfma_cortex_a75_prfm)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_6x8__asm_aarch64_neonfma_ld64)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_6x8__asm_aarch64_neonfma_ld128)

DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch32_neon_cortex_a53)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch32_neon_cortex_a53_prfm)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch32_neon_cortex_a7)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch32_neon_cortex_a53)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch32_neon_cortex_a53_prfm)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch32_neon_cortex_a55)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch32_neon_cortex_a75)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch32_neon_cortex_a75_prfm)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch32_neon_ld64)

DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_1x8__sse_load1)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_3x8__sse_load1)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_4x8__sse_load1)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_5x8__sse_load1)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_6x8__sse_load1)

DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_1x8__sse_dup)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_3x8__sse_dup)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_4x8__sse_dup)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_5x8__sse_dup)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_6x8__sse_dup)

DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_1x8s4__sse)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_3x8s4__sse)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_4x8s4__sse)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_5x8s4__sse)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_6x8s4__sse)

DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_4x2c4__sse)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_6x2c4__sse)

DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_1x8__avx_broadcast)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_1x16__avx_broadcast)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_3x16__avx_broadcast)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_4x8__avx_broadcast)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_4x16__avx_broadcast)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_5x8__avx_broadcast)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_5x16__avx_broadcast)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_6x8__avx_broadcast)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_6x16__avx_broadcast)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_7x8__avx_broadcast)

DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_1x8__fma3_broadcast)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_1x16__fma3_broadcast)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_3x16__fma3_broadcast)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_4x8__fma3_broadcast)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_4x16__fma3_broadcast)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_5x8__fma3_broadcast)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_5x16__fma3_broadcast)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_6x8__fma3_broadcast)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_6x16__fma3_broadcast)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_7x8__fma3_broadcast)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_8x8__fma3_broadcast)

DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_5x16__fma3_broadcast_prfm)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_6x16__fma3_broadcast_prfm)

DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_1x16s4__fma3_broadcast)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_3x16s4__fma3_broadcast)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_4x16s4__fma3_broadcast)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_5x16s4__fma3_broadcast)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_6x16s4__fma3_broadcast)

DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_1x16__avx512f_broadcast)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_4x16__avx512f_broadcast)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_5x16__avx512f_broadcast)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_6x16__avx512f_broadcast)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_7x16__avx512f_broadcast)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_8x16__avx512f_broadcast)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_1x32__avx512f_broadcast)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_4x32__avx512f_broadcast)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_5x32__avx512f_broadcast)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_6x32__avx512f_broadcast)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_7x32__avx512f_broadcast)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_8x32__avx512f_broadcast)

DECLARE_F32_IGEMM_UKERNEL_FUNCTION(xnn_f32_igemm_ukernel_1x8__wasmsimd_loadsplat)
DECLARE_F32_IGEMM_UKERNEL_FUNCTION(xnn_f32_igemm_ukernel_3x8__wasmsimd_loadsplat)
DECLARE_F32_IGEMM_UKERNEL_FUNCTION(xnn_f32_igemm_ukernel_4x8__wasmsimd_loadsplat)
DECLARE_F32_IGEMM_UKERNEL_FUNCTION(xnn_f32_igemm_ukernel_5x8__wasmsimd_loadsplat)
DECLARE_F32_IGEMM_UKERNEL_FUNCTION(xnn_f32_igemm_ukernel_6x8__wasmsimd_loadsplat)

DECLARE_F32_IGEMM_UKERNEL_FUNCTION(xnn_f32_igemm_ukernel_1x8__wasmrelaxedsimd_fma_loadsplat)
DECLARE_F32_IGEMM_UKERNEL_FUNCTION(xnn_f32_igemm_ukernel_3x8__wasmrelaxedsimd_fma_loadsplat)
DECLARE_F32_IGEMM_UKERNEL_FUNCTION(xnn_f32_igemm_ukernel_4x8__wasmrelaxedsimd_fma_loadsplat)
DECLARE_F32_IGEMM_UKERNEL_FUNCTION(xnn_f32_igemm_ukernel_5x8__wasmrelaxedsimd_fma_loadsplat)
DECLARE_F32_IGEMM_UKERNEL_FUNCTION(xnn_f32_igemm_ukernel_6x8__wasmrelaxedsimd_fma_loadsplat)

DECLARE_F32_IGEMM_RELU_UKERNEL_FUNCTION(xnn_f32_igemm_relu_ukernel_1x8__wasmsimd_loadsplat)
DECLARE_F32_IGEMM_RELU_UKERNEL_FUNCTION(xnn_f32_igemm_relu_ukernel_3x8__wasmsimd_loadsplat)
DECLARE_F32_IGEMM_RELU_UKERNEL_FUNCTION(xnn_f32_igemm_relu_ukernel_4x8__wasmsimd_loadsplat)
DECLARE_F32_IGEMM_RELU_UKERNEL_FUNCTION(xnn_f32_igemm_relu_ukernel_5x8__wasmsimd_loadsplat)
DECLARE_F32_IGEMM_RELU_UKERNEL_FUNCTION(xnn_f32_igemm_relu_ukernel_6x8__wasmsimd_loadsplat)

DECLARE_F32_IGEMM_RELU_UKERNEL_FUNCTION(xnn_f32_igemm_relu_ukernel_1x8__wasmrelaxedsimd_fma_loadsplat)
DECLARE_F32_IGEMM_RELU_UKERNEL_FUNCTION(xnn_f32_igemm_relu_ukernel_3x8__wasmrelaxedsimd_fma_loadsplat)
DECLARE_F32_IGEMM_RELU_UKERNEL_FUNCTION(xnn_f32_igemm_relu_ukernel_4x8__wasmrelaxedsimd_fma_loadsplat)
DECLARE_F32_IGEMM_RELU_UKERNEL_FUNCTION(xnn_f32_igemm_relu_ukernel_5x8__wasmrelaxedsimd_fma_loadsplat)
DECLARE_F32_IGEMM_RELU_UKERNEL_FUNCTION(xnn_f32_igemm_relu_ukernel_6x8__wasmrelaxedsimd_fma_loadsplat)

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

DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_1x8__wasmrelaxedsimd_loadsplat)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_3x8__wasmrelaxedsimd_loadsplat)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_4x8__wasmrelaxedsimd_loadsplat)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_5x8__wasmrelaxedsimd_loadsplat)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_6x8__wasmrelaxedsimd_loadsplat)

DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_1x8__wasmrelaxedsimd_fma_loadsplat)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_3x8__wasmrelaxedsimd_fma_loadsplat)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_4x8__wasmrelaxedsimd_fma_loadsplat)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_5x8__wasmrelaxedsimd_fma_loadsplat)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_6x8__wasmrelaxedsimd_fma_loadsplat)

DECLARE_F32_IGEMM_UKERNEL_FUNCTION(xnn_f32_igemm_ukernel_1x8__wasmsimd_splat)
DECLARE_F32_IGEMM_UKERNEL_FUNCTION(xnn_f32_igemm_ukernel_3x8__wasmsimd_splat)
DECLARE_F32_IGEMM_UKERNEL_FUNCTION(xnn_f32_igemm_ukernel_4x8__wasmsimd_splat)
DECLARE_F32_IGEMM_UKERNEL_FUNCTION(xnn_f32_igemm_ukernel_5x8__wasmsimd_splat)
DECLARE_F32_IGEMM_UKERNEL_FUNCTION(xnn_f32_igemm_ukernel_6x8__wasmsimd_splat)

DECLARE_F32_IGEMM_UKERNEL_FUNCTION(xnn_f32_igemm_ukernel_1x8__wasmrelaxedsimd_fma_splat)
DECLARE_F32_IGEMM_UKERNEL_FUNCTION(xnn_f32_igemm_ukernel_3x8__wasmrelaxedsimd_fma_splat)
DECLARE_F32_IGEMM_UKERNEL_FUNCTION(xnn_f32_igemm_ukernel_4x8__wasmrelaxedsimd_fma_splat)
DECLARE_F32_IGEMM_UKERNEL_FUNCTION(xnn_f32_igemm_ukernel_5x8__wasmrelaxedsimd_fma_splat)
DECLARE_F32_IGEMM_UKERNEL_FUNCTION(xnn_f32_igemm_ukernel_6x8__wasmrelaxedsimd_fma_splat)

DECLARE_F32_IGEMM_RELU_UKERNEL_FUNCTION(xnn_f32_igemm_relu_ukernel_1x8__wasmsimd_splat)
DECLARE_F32_IGEMM_RELU_UKERNEL_FUNCTION(xnn_f32_igemm_relu_ukernel_3x8__wasmsimd_splat)
DECLARE_F32_IGEMM_RELU_UKERNEL_FUNCTION(xnn_f32_igemm_relu_ukernel_4x8__wasmsimd_splat)
DECLARE_F32_IGEMM_RELU_UKERNEL_FUNCTION(xnn_f32_igemm_relu_ukernel_5x8__wasmsimd_splat)
DECLARE_F32_IGEMM_RELU_UKERNEL_FUNCTION(xnn_f32_igemm_relu_ukernel_6x8__wasmsimd_splat)

DECLARE_F32_IGEMM_RELU_UKERNEL_FUNCTION(xnn_f32_igemm_relu_ukernel_1x8__wasmrelaxedsimd_fma_splat)
DECLARE_F32_IGEMM_RELU_UKERNEL_FUNCTION(xnn_f32_igemm_relu_ukernel_3x8__wasmrelaxedsimd_fma_splat)
DECLARE_F32_IGEMM_RELU_UKERNEL_FUNCTION(xnn_f32_igemm_relu_ukernel_4x8__wasmrelaxedsimd_fma_splat)
DECLARE_F32_IGEMM_RELU_UKERNEL_FUNCTION(xnn_f32_igemm_relu_ukernel_5x8__wasmrelaxedsimd_fma_splat)
DECLARE_F32_IGEMM_RELU_UKERNEL_FUNCTION(xnn_f32_igemm_relu_ukernel_6x8__wasmrelaxedsimd_fma_splat)

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

DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_1x8__wasmrelaxedsimd_splat)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_3x8__wasmrelaxedsimd_splat)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_4x8__wasmrelaxedsimd_splat)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_5x8__wasmrelaxedsimd_splat)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_6x8__wasmrelaxedsimd_splat)

DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_1x8__wasmrelaxedsimd_fma_splat)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_3x8__wasmrelaxedsimd_fma_splat)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_4x8__wasmrelaxedsimd_fma_splat)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_5x8__wasmrelaxedsimd_fma_splat)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_6x8__wasmrelaxedsimd_fma_splat)

DECLARE_F32_IGEMM_UKERNEL_FUNCTION(xnn_f32_igemm_ukernel_1x8s4__wasmsimd)
DECLARE_F32_IGEMM_UKERNEL_FUNCTION(xnn_f32_igemm_ukernel_3x8s4__wasmsimd)
DECLARE_F32_IGEMM_UKERNEL_FUNCTION(xnn_f32_igemm_ukernel_4x8s4__wasmsimd)
DECLARE_F32_IGEMM_UKERNEL_FUNCTION(xnn_f32_igemm_ukernel_5x8s4__wasmsimd)
DECLARE_F32_IGEMM_UKERNEL_FUNCTION(xnn_f32_igemm_ukernel_6x8s4__wasmsimd)

DECLARE_F32_IGEMM_UKERNEL_FUNCTION(xnn_f32_igemm_ukernel_1x8s4__wasmrelaxedsimd_fma)
DECLARE_F32_IGEMM_UKERNEL_FUNCTION(xnn_f32_igemm_ukernel_3x8s4__wasmrelaxedsimd_fma)
DECLARE_F32_IGEMM_UKERNEL_FUNCTION(xnn_f32_igemm_ukernel_4x8s4__wasmrelaxedsimd_fma)
DECLARE_F32_IGEMM_UKERNEL_FUNCTION(xnn_f32_igemm_ukernel_5x8s4__wasmrelaxedsimd_fma)
DECLARE_F32_IGEMM_UKERNEL_FUNCTION(xnn_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd_fma)

DECLARE_F32_IGEMM_RELU_UKERNEL_FUNCTION(xnn_f32_igemm_relu_ukernel_1x8s4__wasmsimd)
DECLARE_F32_IGEMM_RELU_UKERNEL_FUNCTION(xnn_f32_igemm_relu_ukernel_3x8s4__wasmsimd)
DECLARE_F32_IGEMM_RELU_UKERNEL_FUNCTION(xnn_f32_igemm_relu_ukernel_4x8s4__wasmsimd)
DECLARE_F32_IGEMM_RELU_UKERNEL_FUNCTION(xnn_f32_igemm_relu_ukernel_5x8s4__wasmsimd)
DECLARE_F32_IGEMM_RELU_UKERNEL_FUNCTION(xnn_f32_igemm_relu_ukernel_6x8s4__wasmsimd)

DECLARE_F32_IGEMM_RELU_UKERNEL_FUNCTION(xnn_f32_igemm_relu_ukernel_1x8s4__wasmrelaxedsimd_fma)
DECLARE_F32_IGEMM_RELU_UKERNEL_FUNCTION(xnn_f32_igemm_relu_ukernel_3x8s4__wasmrelaxedsimd_fma)
DECLARE_F32_IGEMM_RELU_UKERNEL_FUNCTION(xnn_f32_igemm_relu_ukernel_4x8s4__wasmrelaxedsimd_fma)
DECLARE_F32_IGEMM_RELU_UKERNEL_FUNCTION(xnn_f32_igemm_relu_ukernel_5x8s4__wasmrelaxedsimd_fma)
DECLARE_F32_IGEMM_RELU_UKERNEL_FUNCTION(xnn_f32_igemm_relu_ukernel_6x8s4__wasmrelaxedsimd_fma)

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

DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_1x8s4__wasmrelaxedsimd)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_3x8s4__wasmrelaxedsimd)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_4x8s4__wasmrelaxedsimd)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_5x8s4__wasmrelaxedsimd)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_6x8s4__wasmrelaxedsimd)

DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_1x8s4__wasmrelaxedsimd_fma)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_3x8s4__wasmrelaxedsimd_fma)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_4x8s4__wasmrelaxedsimd_fma)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_5x8s4__wasmrelaxedsimd_fma)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_6x8s4__wasmrelaxedsimd_fma)

DECLARE_F32_IGEMM_UKERNEL_FUNCTION(xnn_f32_igemm_ukernel_4x2c4__wasmrelaxedsimd_fma)
DECLARE_F32_IGEMM_UKERNEL_FUNCTION(xnn_f32_igemm_ukernel_4x2c4__wasmsimd)

DECLARE_F32_IGEMM_RELU_UKERNEL_FUNCTION(xnn_f32_igemm_relu_ukernel_4x2c4__wasmrelaxedsimd_fma)
DECLARE_F32_IGEMM_RELU_UKERNEL_FUNCTION(xnn_f32_igemm_relu_ukernel_4x2c4__wasmsimd)

DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_4x2c4__wasmrelaxedsimd)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_4x2c4__wasmrelaxedsimd_fma)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_4x2c4__wasmsimd_arm)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_4x2c4__wasmsimd_x86)

DECLARE_F32_IGEMM_RELU_UKERNEL_FUNCTION(xnn_f32_igemm_relu_ukernel_1x4__wasm)
DECLARE_F32_IGEMM_RELU_UKERNEL_FUNCTION(xnn_f32_igemm_relu_ukernel_2x4__wasm)
DECLARE_F32_IGEMM_RELU_UKERNEL_FUNCTION(xnn_f32_igemm_relu_ukernel_4x2__wasm)
DECLARE_F32_IGEMM_RELU_UKERNEL_FUNCTION(xnn_f32_igemm_relu_ukernel_4x4__wasm)

DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_1x4__wasm)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_2x4__wasm)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_4x2__wasm)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_4x4__wasm)

DECLARE_F32_IGEMM_UKERNEL_FUNCTION(xnn_f32_igemm_ukernel_1x4__scalar)
DECLARE_F32_IGEMM_UKERNEL_FUNCTION(xnn_f32_igemm_ukernel_2x4__scalar)
DECLARE_F32_IGEMM_UKERNEL_FUNCTION(xnn_f32_igemm_ukernel_4x2__scalar)
DECLARE_F32_IGEMM_UKERNEL_FUNCTION(xnn_f32_igemm_ukernel_4x4__scalar)

DECLARE_F32_IGEMM_RELU_UKERNEL_FUNCTION(xnn_f32_igemm_relu_ukernel_1x4__scalar)
DECLARE_F32_IGEMM_RELU_UKERNEL_FUNCTION(xnn_f32_igemm_relu_ukernel_2x4__scalar)
DECLARE_F32_IGEMM_RELU_UKERNEL_FUNCTION(xnn_f32_igemm_relu_ukernel_4x2__scalar)
DECLARE_F32_IGEMM_RELU_UKERNEL_FUNCTION(xnn_f32_igemm_relu_ukernel_4x4__scalar)

DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_1x4__scalar)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_2x4__scalar)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_4x2__scalar)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_4x4__scalar)


DECLARE_F32_IGEMM_UKERNEL_FUNCTION(xnn_f32_igemm_ukernel_1x4v__rvv)
DECLARE_F32_IGEMM_UKERNEL_FUNCTION(xnn_f32_igemm_ukernel_7x4v__rvv)

DECLARE_F32_IGEMM_RELU_UKERNEL_FUNCTION(xnn_f32_igemm_relu_ukernel_1x4v__rvv)
DECLARE_F32_IGEMM_RELU_UKERNEL_FUNCTION(xnn_f32_igemm_relu_ukernel_7x4v__rvv)

DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_1x4v__rvv)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_7x4v__rvv)

DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_1x32__hvx_broadcast)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_1x64__hvx_broadcast)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_1x128__hvx_broadcast)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_2x128__hvx_broadcast)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_4x64__hvx_broadcast)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_7x64__hvx_broadcast)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_8x32__hvx_broadcast)
DECLARE_F32_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_igemm_minmax_ukernel_16x32__hvx_broadcast)

#define DECLARE_F16_IGEMM_MINMAX_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                               \
      size_t mr,                                           \
      size_t nr,                                           \
      size_t kc,                                           \
      size_t ks,                                           \
      const xnn_float16** a,                      \
      const xnn_float16* w,                       \
      xnn_float16*,                               \
      size_t cm_stride,                                    \
      size_t cn_stride,                                    \
      size_t a_offset,                                     \
      const xnn_float16* zero,                    \
      const union xnn_f16_minmax_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

DECLARE_F16_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f16_igemm_minmax_ukernel_1x8__neonfp16arith_ld64)
DECLARE_F16_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f16_igemm_minmax_ukernel_1x16__asm_aarch64_neonfp16arith_ld32)
DECLARE_F16_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f16_igemm_minmax_ukernel_1x16__asm_aarch64_neonfp16arith_ld64)
DECLARE_F16_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f16_igemm_minmax_ukernel_1x16__neonfp16arith_ld64)
DECLARE_F16_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f16_igemm_minmax_ukernel_4x8__neonfp16arith_ld64)
DECLARE_F16_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f16_igemm_minmax_ukernel_4x16__asm_aarch64_neonfp16arith_ld32)
DECLARE_F16_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f16_igemm_minmax_ukernel_4x16__asm_aarch64_neonfp16arith_ld64)
DECLARE_F16_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f16_igemm_minmax_ukernel_4x16__neonfp16arith_ld64)
DECLARE_F16_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f16_igemm_minmax_ukernel_6x8__neonfp16arith_ld64)
DECLARE_F16_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f16_igemm_minmax_ukernel_6x16__asm_aarch64_neonfp16arith_cortex_a55)
DECLARE_F16_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f16_igemm_minmax_ukernel_6x16__asm_aarch64_neonfp16arith_cortex_a55r0)
DECLARE_F16_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f16_igemm_minmax_ukernel_6x16__asm_aarch64_neonfp16arith_cortex_a75)
DECLARE_F16_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f16_igemm_minmax_ukernel_6x16__asm_aarch64_neonfp16arith_ld32)
DECLARE_F16_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f16_igemm_minmax_ukernel_6x16__asm_aarch64_neonfp16arith_ld64)
DECLARE_F16_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f16_igemm_minmax_ukernel_6x16__neonfp16arith_ld64)
DECLARE_F16_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f16_igemm_minmax_ukernel_8x8__neonfp16arith_ld64)
DECLARE_F16_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f16_igemm_minmax_ukernel_8x16__neonfp16arith_ld64)

DECLARE_F16_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f16_igemm_minmax_ukernel_1x8__avx2_broadcast)
DECLARE_F16_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f16_igemm_minmax_ukernel_1x16__avx2_broadcast)
DECLARE_F16_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f16_igemm_minmax_ukernel_3x16__avx2_broadcast)
DECLARE_F16_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f16_igemm_minmax_ukernel_4x8__avx2_broadcast)
DECLARE_F16_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f16_igemm_minmax_ukernel_4x16__avx2_broadcast)
DECLARE_F16_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f16_igemm_minmax_ukernel_5x8__avx2_broadcast)
DECLARE_F16_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f16_igemm_minmax_ukernel_5x16__avx2_broadcast)
DECLARE_F16_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f16_igemm_minmax_ukernel_6x8__avx2_broadcast)
DECLARE_F16_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f16_igemm_minmax_ukernel_7x8__avx2_broadcast)

DECLARE_F16_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f16_f32acc_igemm_minmax_ukernel_1x8__avx2_broadcast)
DECLARE_F16_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f16_f32acc_igemm_minmax_ukernel_1x16__avx2_broadcast)
DECLARE_F16_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f16_f32acc_igemm_minmax_ukernel_3x16__avx2_broadcast)
DECLARE_F16_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f16_f32acc_igemm_minmax_ukernel_4x8__avx2_broadcast)
DECLARE_F16_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f16_f32acc_igemm_minmax_ukernel_4x16__avx2_broadcast)
DECLARE_F16_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f16_f32acc_igemm_minmax_ukernel_5x8__avx2_broadcast)
DECLARE_F16_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f16_f32acc_igemm_minmax_ukernel_5x16__avx2_broadcast)
DECLARE_F16_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f16_f32acc_igemm_minmax_ukernel_6x8__avx2_broadcast)
DECLARE_F16_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f16_f32acc_igemm_minmax_ukernel_7x8__avx2_broadcast)

DECLARE_F16_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f16_igemm_minmax_ukernel_1x32__avx512fp16_broadcast)
DECLARE_F16_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f16_igemm_minmax_ukernel_4x32__avx512fp16_broadcast)
DECLARE_F16_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f16_igemm_minmax_ukernel_5x32__avx512fp16_broadcast)
DECLARE_F16_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f16_igemm_minmax_ukernel_6x32__avx512fp16_broadcast)
DECLARE_F16_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f16_igemm_minmax_ukernel_7x32__avx512fp16_broadcast)
DECLARE_F16_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f16_igemm_minmax_ukernel_8x32__avx512fp16_broadcast)

DECLARE_F16_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f16_igemm_minmax_ukernel_1x64__avx512fp16_broadcast)
DECLARE_F16_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f16_igemm_minmax_ukernel_4x64__avx512fp16_broadcast)
DECLARE_F16_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f16_igemm_minmax_ukernel_5x64__avx512fp16_broadcast)
DECLARE_F16_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f16_igemm_minmax_ukernel_6x64__avx512fp16_broadcast)
DECLARE_F16_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f16_igemm_minmax_ukernel_7x64__avx512fp16_broadcast)
DECLARE_F16_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_f16_igemm_minmax_ukernel_8x64__avx512fp16_broadcast)

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
      const union xnn_qu8_conv_minmax_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_rndnu_ukernel_1x8__neon_mlal_lane)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_rndnu_ukernel_1x16__neon_mlal_lane)

DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_rndnu16_ukernel_1x16__neon_mlal_lane)

DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_rndnu_ukernel_2x8__neon_mlal_lane)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_rndnu_ukernel_2x16__neon_mlal_lane)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_rndnu_ukernel_3x8__neon_mlal_lane)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_rndnu_ukernel_3x16__neon_mlal_lane)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_rndnu_ukernel_4x8__neon_mlal_lane)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_rndnu_ukernel_4x16__neon_mlal_lane)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_rndnu_ukernel_6x8__neon_mlal_lane)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_rndnu_ukernel_6x16__neon_mlal_lane)

DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_1x8__neon_mlal_lane)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_1x16__neon_mlal_lane)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_4x8__neon_mlal_lane)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_4x16__neon_mlal_lane)

DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_1x16__neonv8_mlal_lane)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_4x16__neonv8_mlal_lane)

DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_rndnu_ukernel_4x16__asm_aarch64_neon_mlal_lane_cortex_a53)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_rndnu_ukernel_4x16__asm_aarch64_neon_mlal_lane_cortex_a53_prfm)

DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_rndnu16_ukernel_4x16__asm_aarch64_neon_mlal_lane_cortex_a53)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_rndnu16_ukernel_4x16__asm_aarch64_neon_mlal_lane_cortex_a53_prfm)

DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_rndnu_ukernel_4x16__asm_aarch64_neon_mlal_lane_ld64)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_rndnu_ukernel_4x16__asm_aarch64_neon_mlal_lane_ld64_prfm)

DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_rndnu_ukernel_4x16__asm_aarch64_neon_mlal_lane_cortex_a75)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_rndnu_ukernel_4x16__asm_aarch64_neon_mlal_lane_cortex_a75_prfm)

DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_rndnu_ukernel_4x8c4__asm_aarch64_neondot_cortex_a55)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_rndnu_ukernel_4x8c4__asm_aarch64_neondot_ld128)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_rndnu_ukernel_4x16c4__asm_aarch64_neondot_cortex_a55)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_rndnu_ukernel_4x16c4__asm_aarch64_neondot_ld128)

DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_rndnu_ukernel_1x8c4__neondot)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_rndnu_ukernel_1x16c4__neondot)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_rndnu_ukernel_1x32c4__neondot)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_rndnu_ukernel_2x8c4__neondot)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_rndnu_ukernel_2x16c4__neondot)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_rndnu_ukernel_2x32c4__neondot)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_rndnu_ukernel_3x8c4__neondot)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_rndnu_ukernel_3x16c4__neondot)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_rndnu_ukernel_3x32c4__neondot)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_rndnu_ukernel_4x8c4__neondot)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_rndnu_ukernel_4x16c4__neondot)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_rndnu_ukernel_5x8c4__neondot)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_rndnu_ukernel_5x16c4__neondot)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_rndnu_ukernel_6x8c4__neondot)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_rndnu_ukernel_6x16c4__neondot)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_rndnu_ukernel_8x8c4__neondot)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_rndnu_ukernel_8x16c4__neondot)

DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_4x16c4__asm_aarch64_neondot_cortex_a55)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_4x16c4__asm_aarch64_neondot_ld128)

DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_1x16c4__neondot)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_2x16c4__neondot)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_4x16c4__neondot)

DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_rndnu_ukernel_1x8__asm_aarch32_neon_mlal_lane_cortex_a7)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_rndnu_ukernel_1x8__asm_aarch32_neon_mlal_lane_cortex_a7_prfm)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a7)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a7_prfm)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a53)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a53_prfm)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_ld64)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_ld64_prfm)

DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_1x4c2__sse2_ld64)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_2x4c2__sse2_ld64)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_3x4c2__sse2_ld64)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_4x4c2__sse2_ld64)

DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_1x4c2__sse41_ld64)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_2x4c2__sse41_ld64)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_3x4c2__sse41_ld64)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_4x4c2__sse41_ld64)

DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_1x4c2__avx_ld64)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_2x4c2__avx_ld64)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_3x4c2__avx_ld64)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_4x4c2__avx_ld64)

DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_1x4c2__sse2_ld128)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_2x4c2__sse2_ld128)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_3x4c2__sse2_ld128)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_4x4c2__sse2_ld128)

DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_1x4c2__sse41_ld128)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_2x4c2__sse41_ld128)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_3x4c2__sse41_ld128)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_4x4c2__sse41_ld128)

DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_1x4c2__avx_ld128)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_2x4c2__avx_ld128)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_3x4c2__avx_ld128)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_4x4c2__avx_ld128)

DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_1x4c2s4__sse2_ld64)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_2x4c2s4__sse2_ld64)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_3x4c2s4__sse2_ld64)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_4x4c2s4__sse2_ld64)

DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_1x4c2s4__sse41_ld64)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_2x4c2s4__sse41_ld64)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_3x4c2s4__sse41_ld64)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_4x4c2s4__sse41_ld64)

DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_1x4c2s4__avx_ld64)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_2x4c2s4__avx_ld64)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_3x4c2s4__avx_ld64)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_4x4c2s4__avx_ld64)

DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_1x4c2s4__sse2_ld128)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_2x4c2s4__sse2_ld128)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_3x4c2s4__sse2_ld128)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_4x4c2s4__sse2_ld128)

DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_1x4c2s4__sse41_ld128)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_2x4c2s4__sse41_ld128)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_3x4c2s4__sse41_ld128)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_4x4c2s4__sse41_ld128)

DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_1x4c2s4__avx_ld128)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_2x4c2s4__avx_ld128)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_3x4c2s4__avx_ld128)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_4x4c2s4__avx_ld128)

DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_1x4c8__sse2_ld64)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_2x4c8__sse2_ld64)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_3x4c8__sse2_ld64)

DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_1x4c8__sse41_ld64)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_2x4c8__sse41_ld64)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_3x4c8__sse41_ld64)

DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_1x4c8__avx_ld64)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_2x4c8__avx_ld64)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_3x4c8__avx_ld64)

DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_1x4c8__sse2_ld128)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_2x4c8__sse2_ld128)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_3x4c8__sse2_ld128)

DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_1x4c8__sse41_ld128)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_2x4c8__sse41_ld128)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_3x4c8__sse41_ld128)

DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_1x4c8__avx_ld128)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_2x4c8__avx_ld128)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_3x4c8__avx_ld128)

DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_1x8c8__avx2)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_2x8c8__avx2)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_3x8c8__avx2)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_4x8c8__avx2)

DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_1x8c8__avx256skx)

DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_1x16c8__avx512skx)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_5x16c8__avx512skx)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_7x16c8__avx512skx)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_8x16c8__avx512skx)

DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_1x16c8__avx512skx_prfm)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_5x16c8__avx512skx_prfm)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_7x16c8__avx512skx_prfm)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_8x16c8__avx512skx_prfm)

DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_1x4c2__wasmsimd_dot16x2_ld64)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_2x4c2__wasmsimd_dot16x2_ld64)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_3x4c2__wasmsimd_dot16x2_ld64)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_4x4c2__wasmsimd_dot16x2_ld64)

DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_1x4c2__wasmsimd_dot16x2_ld128)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_2x4c2__wasmsimd_dot16x2_ld128)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_3x4c2__wasmsimd_dot16x2_ld128)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_4x4c2__wasmsimd_dot16x2_ld128)

DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_1x4c2s4__wasmsimd_dot16x2_ld64)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_2x4c2s4__wasmsimd_dot16x2_ld64)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_3x4c2s4__wasmsimd_dot16x2_ld64)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_4x4c2s4__wasmsimd_dot16x2_ld64)

DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_1x4c2s4__wasmsimd_dot16x2_ld128)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_2x4c2s4__wasmsimd_dot16x2_ld128)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_3x4c2s4__wasmsimd_dot16x2_ld128)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_4x4c2s4__wasmsimd_dot16x2_ld128)

DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_1x4c8__wasmsimd_dot16x2_ld64)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_2x4c8__wasmsimd_dot16x2_ld64)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_3x4c8__wasmsimd_dot16x2_ld64)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_4x4c8__wasmsimd_dot16x2_ld64)

DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_1x4c8__wasmsimd_dot16x2_ld128)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_2x4c8__wasmsimd_dot16x2_ld128)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_3x4c8__wasmsimd_dot16x2_ld128)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_4x4c8__wasmsimd_dot16x2_ld128)

DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_1x2__wasm_fmagic)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_1x4__wasm_fmagic)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_2x2__wasm_fmagic)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_2x4__wasm_fmagic)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_3x2__wasm_fmagic)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_3x4__wasm_fmagic)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_4x2__wasm_fmagic)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_4x4__wasm_fmagic)

DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_1x1c4__armsimd32)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_1x2c4__armsimd32)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_2x1c4__armsimd32)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_2x2c4__armsimd32)

DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_rndnu_ukernel_1x2__scalar)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_rndnu_ukernel_1x4__scalar)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_rndnu_ukernel_2x2__scalar)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_rndnu_ukernel_2x4__scalar)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_rndnu_ukernel_3x2__scalar)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_rndnu_ukernel_3x4__scalar)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_rndnu_ukernel_4x2__scalar)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_rndnu_ukernel_4x4__scalar)

DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_1x2__scalar_fmagic)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_1x4__scalar_fmagic)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_2x2__scalar_fmagic)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_2x4__scalar_fmagic)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_3x2__scalar_fmagic)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_3x4__scalar_fmagic)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_4x2__scalar_fmagic)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_4x4__scalar_fmagic)

DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_1x2__scalar_imagic)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_1x4__scalar_imagic)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_2x2__scalar_imagic)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_2x4__scalar_imagic)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_3x2__scalar_imagic)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_3x4__scalar_imagic)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_4x2__scalar_imagic)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_4x4__scalar_imagic)

DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_1x2__scalar_lrintf)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_1x4__scalar_lrintf)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_2x2__scalar_lrintf)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_2x4__scalar_lrintf)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_3x2__scalar_lrintf)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_3x4__scalar_lrintf)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_4x2__scalar_lrintf)
DECLARE_QU8_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_igemm_minmax_fp32_ukernel_4x4__scalar_lrintf)

#define DECLARE_QD8_F16_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(fn_name)               \
  XNN_INTERNAL void fn_name(                                                      \
      size_t mr,                                                                  \
      size_t nr,                                                                  \
      size_t kc,                                                                  \
      size_t ks,                                                                  \
      const int8_t** a,                                                           \
      const void* w,                                                              \
      xnn_float16*,                                                      \
      size_t cm_stride,                                                           \
      size_t cn_stride,                                                           \
      size_t a_offset,                                                            \
      const int8_t* zero_sentinel,                                                \
      const int8_t* zero_data,                                                    \
      const union xnn_f16_minmax_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)], \
      const struct xnn_qd8_quantization_params quantization_params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

DECLARE_QD8_F16_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x8c8__neoni8mm)
DECLARE_QD8_F16_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x16c8__neoni8mm)
DECLARE_QD8_F16_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_2x8c8__neoni8mm)
DECLARE_QD8_F16_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_2x16c8__neoni8mm)
DECLARE_QD8_F16_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_3x8c8__neoni8mm)
DECLARE_QD8_F16_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_3x16c8__neoni8mm)
DECLARE_QD8_F16_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_4x8c8__neoni8mm)
DECLARE_QD8_F16_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_4x16c8__neoni8mm)
DECLARE_QD8_F16_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_6x8c8__neoni8mm)
DECLARE_QD8_F16_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_6x16c8__neoni8mm)
DECLARE_QD8_F16_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_8x8c8__neoni8mm)
DECLARE_QD8_F16_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_8x16c8__neoni8mm)

DECLARE_QD8_F16_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x8c4__neondotfp16arith)
DECLARE_QD8_F16_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x16c4__neondotfp16arith)
DECLARE_QD8_F16_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x32c4__neondotfp16arith)
DECLARE_QD8_F16_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_2x8c4__neondotfp16arith)
DECLARE_QD8_F16_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_2x16c4__neondotfp16arith)
DECLARE_QD8_F16_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_2x32c4__neondotfp16arith)
DECLARE_QD8_F16_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_4x8c4__neondotfp16arith)
DECLARE_QD8_F16_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_4x16c4__neondotfp16arith)
DECLARE_QD8_F16_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_4x32c4__neondotfp16arith)
DECLARE_QD8_F16_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_6x8c4__neondotfp16arith)
DECLARE_QD8_F16_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_6x16c4__neondotfp16arith)
DECLARE_QD8_F16_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_6x32c4__neondotfp16arith)
DECLARE_QD8_F16_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_8x8c4__neondotfp16arith)
DECLARE_QD8_F16_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_8x16c4__neondotfp16arith)
DECLARE_QD8_F16_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_8x32c4__neondotfp16arith)

DECLARE_QD8_F16_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_4x16c4__asm_aarch64_neondotfp16arith_cortex_a55)
DECLARE_QD8_F16_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_4x16c4__asm_aarch64_neondotfp16arith_ld128)

DECLARE_QD8_F16_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x8c2s4__neonfp16arith)
DECLARE_QD8_F16_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_2x8c2s4__neonfp16arith)

DECLARE_QD8_F16_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_4x8c4__asm_aarch32_neondotfp16arith_cortex_a55)

DECLARE_QD8_F16_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x8c8__avx2)
DECLARE_QD8_F16_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_2x8c8__avx2)
DECLARE_QD8_F16_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_3x8c8__avx2)
DECLARE_QD8_F16_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_4x8c8__avx2)

DECLARE_QD8_F16_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x8c8__avx256skx)
DECLARE_QD8_F16_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_5x8c8__avx256skx)
DECLARE_QD8_F16_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_7x8c8__avx256skx)
DECLARE_QD8_F16_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_8x8c8__avx256skx)

DECLARE_QD8_F16_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x8c8__avx256vnni)
DECLARE_QD8_F16_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_5x8c8__avx256vnni)
DECLARE_QD8_F16_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_7x8c8__avx256vnni)
DECLARE_QD8_F16_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_8x8c8__avx256vnni)
DECLARE_QD8_F16_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_9x8c8__avx256vnni)
DECLARE_QD8_F16_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_10x8c8__avx256vnni)
DECLARE_QD8_F16_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_12x8c8__avx256vnni)
DECLARE_QD8_F16_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_14x8c8__avx256vnni)


DECLARE_QD8_F16_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x8c8__avx256vnni_prfm)
DECLARE_QD8_F16_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_5x8c8__avx256vnni_prfm)
DECLARE_QD8_F16_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_7x8c8__avx256vnni_prfm)
DECLARE_QD8_F16_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_8x8c8__avx256vnni_prfm)
DECLARE_QD8_F16_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_9x8c8__avx256vnni_prfm)
DECLARE_QD8_F16_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_10x8c8__avx256vnni_prfm)
DECLARE_QD8_F16_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_12x8c8__avx256vnni_prfm)
DECLARE_QD8_F16_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_14x8c8__avx256vnni_prfm)


DECLARE_QD8_F16_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x8c8__avxvnni)
DECLARE_QD8_F16_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_2x8c8__avxvnni)
DECLARE_QD8_F16_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_3x8c8__avxvnni)
DECLARE_QD8_F16_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_4x8c8__avxvnni)
DECLARE_QD8_F16_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_5x8c8__avxvnni)
DECLARE_QD8_F16_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_6x8c8__avxvnni)
DECLARE_QD8_F16_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_7x8c8__avxvnni)
DECLARE_QD8_F16_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_8x8c8__avxvnni)

DECLARE_QD8_F16_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x8c8__avxvnni_prfm)
DECLARE_QD8_F16_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_2x8c8__avxvnni_prfm)
DECLARE_QD8_F16_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_3x8c8__avxvnni_prfm)
DECLARE_QD8_F16_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_4x8c8__avxvnni_prfm)
DECLARE_QD8_F16_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_5x8c8__avxvnni_prfm)
DECLARE_QD8_F16_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_6x8c8__avxvnni_prfm)
DECLARE_QD8_F16_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_7x8c8__avxvnni_prfm)
DECLARE_QD8_F16_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_8x8c8__avxvnni_prfm)

DECLARE_QD8_F16_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_1x64c4__avx512amx)
DECLARE_QD8_F16_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_7x64c4__avx512amx)
DECLARE_QD8_F16_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_16x64c4__avx512amx)
DECLARE_QD8_F16_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_igemm_minmax_ukernel_16x64c4__avx512amx_prfm)

#define DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(fn_name)               \
  XNN_INTERNAL void fn_name(                                                      \
      size_t mr,                                                                  \
      size_t nr,                                                                  \
      size_t kc,                                                                  \
      size_t ks,                                                                  \
      const int8_t** a,                                                           \
      const void* w,                                                              \
      float* c,                                                                   \
      size_t cm_stride,                                                           \
      size_t cn_stride,                                                           \
      size_t a_offset,                                                            \
      const int8_t* zero_sentinel,                                                \
      const int8_t* zero_data,                                                    \
      const union xnn_f32_minmax_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)], \
      const struct xnn_qd8_quantization_params quantization_params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x2__scalar)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x4__scalar)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x8__scalar)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_2x2__scalar)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_2x4__scalar)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_2x8__scalar)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x4__scalar)

DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x8__neon_mlal_lane)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x8__neon_mlal_lane_prfm)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x16__neon_mlal_lane)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x16__neon_mlal_lane_prfm)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_2x8__neon_mlal_lane)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_2x8__neon_mlal_lane_prfm)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_2x16__neon_mlal_lane)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_2x16__neon_mlal_lane_prfm)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_3x8__neon_mlal_lane)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_3x8__neon_mlal_lane_prfm)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_3x16__neon_mlal_lane)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_3x16__neon_mlal_lane_prfm)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x8__neon_mlal_lane)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x8__neon_mlal_lane_prfm)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x16__neon_mlal_lane)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x16__neon_mlal_lane_prfm)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_6x8__neon_mlal_lane)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_6x8__neon_mlal_lane_prfm)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_6x16__neon_mlal_lane)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_6x16__neon_mlal_lane_prfm)

DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x8c4__neondot)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x16c4__neondot)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x32c4__neondot)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_2x8c4__neondot)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_2x16c4__neondot)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_2x32c4__neondot)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x8c4__neondot)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x16c4__neondot)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x32c4__neondot)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_6x8c4__neondot)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_6x16c4__neondot)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_6x32c4__neondot)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_8x8c4__neondot)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_8x16c4__neondot)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_8x32c4__neondot)

DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x8c8__neoni8mm)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x16c8__neoni8mm)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_2x8c8__neoni8mm)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_2x16c8__neoni8mm)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_3x8c8__neoni8mm)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_3x16c8__neoni8mm)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x8c8__neoni8mm)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x16c8__neoni8mm)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_6x8c8__neoni8mm)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_6x16c8__neoni8mm)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_8x8c8__neoni8mm)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_8x16c8__neoni8mm)

DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x16c8__avx512skx)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_5x16c8__avx512skx)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_7x16c8__avx512skx)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_8x16c8__avx512skx)

DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x16c8__avx512skx_prfm)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_5x16c8__avx512skx_prfm)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_7x16c8__avx512skx_prfm)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_8x16c8__avx512skx_prfm)

DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x2__wasm)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x4__wasm)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x8__wasm)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_2x2__wasm)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_2x4__wasm)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_2x8__wasm)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x4__wasm)

DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x4c16__wasmsdot)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_2x4c16__wasmsdot)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_3x4c16__wasmsdot)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x4c16__wasmsdot)

DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x8c8__wasmusdot)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_2x8c8__wasmusdot)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_3x8c8__wasmusdot)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x8c8__wasmusdot)

DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x8c8__wasmusdot_u2)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_2x8c8__wasmusdot_u2)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_3x8c8__wasmusdot_u2)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x8c8__wasmusdot_u2)

DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x8c8__wasmsdot)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_2x8c8__wasmsdot)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_3x8c8__wasmsdot)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x8c8__wasmsdot)

DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x8c8__wasmsdot_u2)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_2x8c8__wasmsdot_u2)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_3x8c8__wasmsdot_u2)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x8c8__wasmsdot_u2)

DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x16c4__wasmusdot)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x16c4__wasmusdot)

DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x16c4__wasmusdot_u2)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x16c4__wasmusdot_u2)

DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x4c2__wasmsimd_dot16x2_ld64)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x4c2__wasmsimd_dot16x2_ld128)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x4c2s4__wasmsimd_dot16x2_ld64)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x4c2s4__wasmsimd_dot16x2_ld128)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x4c8__wasmsimd_dot16x2_ld64)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x4c8__wasmsimd_dot16x2_ld128)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_2x4c2__wasmsimd_dot16x2_ld64)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_2x4c2__wasmsimd_dot16x2_ld128)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_2x4c2s4__wasmsimd_dot16x2_ld64)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_2x4c2s4__wasmsimd_dot16x2_ld128)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_2x4c8__wasmsimd_dot16x2_ld64)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_2x4c8__wasmsimd_dot16x2_ld128)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_3x4c2__wasmsimd_dot16x2_ld64)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_3x4c2__wasmsimd_dot16x2_ld128)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_3x4c2s4__wasmsimd_dot16x2_ld64)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_3x4c2s4__wasmsimd_dot16x2_ld128)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_3x4c8__wasmsimd_dot16x2_ld64)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_3x4c8__wasmsimd_dot16x2_ld128)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x4c2__wasmsimd_dot16x2_ld64)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x4c2__wasmsimd_dot16x2_ld128)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x4c2s4__wasmsimd_dot16x2_ld64)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x4c2s4__wasmsimd_dot16x2_ld128)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x4c8__wasmsimd_dot16x2_ld64)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x4c8__wasmsimd_dot16x2_ld128)

DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x4c8__avx_ld64)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x4c8__avx_ld128)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x4c8__sse2_ld64)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x4c8__sse2_ld128)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x4c8__sse41_ld64)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x4c8__sse41_ld128)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_2x4c8__avx_ld64)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_2x4c8__avx_ld128)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_2x4c8__sse2_ld64)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_2x4c8__sse2_ld128)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_2x4c8__sse41_ld64)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_2x4c8__sse41_ld128)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_3x4c8__avx_ld64)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_3x4c8__avx_ld128)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_3x4c8__sse2_ld64)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_3x4c8__sse2_ld128)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_3x4c8__sse41_ld64)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_3x4c8__sse41_ld128)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x4c8__avx_ld64)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x4c8__avx_ld128)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x4c8__sse2_ld64)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x4c8__sse2_ld128)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x4c8__sse41_ld64)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x4c8__sse41_ld128)

DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x8c8__avx2)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_2x8c8__avx2)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_3x8c8__avx2)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x8c8__avx2)

DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x8c8__avx256skx)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_5x8c8__avx256skx)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_7x8c8__avx256skx)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_8x8c8__avx256skx)

DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x8c8__aarch64_neondot_ld128)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x8c8__neondot_ld64)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x16c8__aarch64_neondot_ld128)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x16c8__neondot_ld64)

DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x8c2s4__neon_mlal)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_2x8c2s4__neon_mlal)

DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x16c4__asm_aarch64_neondot_cortex_a55)

DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x8c4__asm_aarch32_neondot_cortex_a55)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x16c4__asm_aarch64_neondot_ld128)

DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x16c4__avx512amx)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_7x16c4__avx512amx)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_16x16c4__avx512amx)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_16x16c4__avx512amx_prfm)

DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x32c4__avx512amx)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_7x32c4__avx512amx)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_16x32c4__avx512amx)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_16x32c4__avx512amx_prfm)

DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x64c4__avx512amx)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_7x64c4__avx512amx)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_16x64c4__avx512amx)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_16x64c4__avx512amx_prfm)

DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x16c4__avx512vnni)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x16c4__avx512vnni)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_5x16c4__avx512vnni)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_7x16c4__avx512vnni)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_8x16c4__avx512vnni)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_9x16c4__avx512vnni)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_10x16c4__avx512vnni)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_12x16c4__avx512vnni)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_14x16c4__avx512vnni)

DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x16c4__avx512vnni_prfm)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x16c4__avx512vnni_prfm)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_5x16c4__avx512vnni_prfm)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_7x16c4__avx512vnni_prfm)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_8x16c4__avx512vnni_prfm)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_9x16c4__avx512vnni_prfm)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_10x16c4__avx512vnni_prfm)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_12x16c4__avx512vnni_prfm)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_14x16c4__avx512vnni_prfm)


DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x16c8__avx512vnni)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_5x16c8__avx512vnni)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_7x16c8__avx512vnni)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_8x16c8__avx512vnni)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_9x16c8__avx512vnni)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_10x16c8__avx512vnni)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_12x16c8__avx512vnni)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_14x16c8__avx512vnni)


DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x16c8__avx512vnni_prfm)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_5x16c8__avx512vnni_prfm)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_7x16c8__avx512vnni_prfm)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_8x16c8__avx512vnni_prfm)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_9x16c8__avx512vnni_prfm)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_10x16c8__avx512vnni_prfm)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_12x16c8__avx512vnni_prfm)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_14x16c8__avx512vnni_prfm)


DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x8c8__avx256vnni)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_5x8c8__avx256vnni)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_7x8c8__avx256vnni)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_8x8c8__avx256vnni)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_9x8c8__avx256vnni)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_10x8c8__avx256vnni)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_12x8c8__avx256vnni)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_14x8c8__avx256vnni)


DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x8c8__avx256vnni_prfm)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_5x8c8__avx256vnni_prfm)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_7x8c8__avx256vnni_prfm)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_8x8c8__avx256vnni_prfm)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_9x8c8__avx256vnni_prfm)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_10x8c8__avx256vnni_prfm)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_12x8c8__avx256vnni_prfm)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_14x8c8__avx256vnni_prfm)


DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x8c8__avxvnni)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_2x8c8__avxvnni)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_3x8c8__avxvnni)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x8c8__avxvnni)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_5x8c8__avxvnni)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_6x8c8__avxvnni)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_7x8c8__avxvnni)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_8x8c8__avxvnni)

DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x8c8__avxvnni_prfm)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_2x8c8__avxvnni_prfm)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_3x8c8__avxvnni_prfm)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x8c8__avxvnni_prfm)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_5x8c8__avxvnni_prfm)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_6x8c8__avxvnni_prfm)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_7x8c8__avxvnni_prfm)
DECLARE_QD8_F32_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_8x8c8__avxvnni_prfm)

#define DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                                    \
      size_t mr,                                                \
      size_t nr,                                                \
      size_t kc,                                                \
      size_t ks,                                                \
      const int8_t** a,                                         \
      const void* w,                                            \
      int8_t* c,                                                \
      size_t cm_stride,                                         \
      size_t cn_stride,                                         \
      size_t a_offset,                                          \
      const int8_t* zero,                                       \
      const union xnn_qs8_qc8w_conv_minmax_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8__neon_mlal_lane)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8__neon_mlal_lane_prfm)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x16__neon_mlal_lane)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x16__neon_mlal_lane_prfm)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x8__neon_mlal_lane)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x8__neon_mlal_lane_prfm)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x16__neon_mlal_lane)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x16__neon_mlal_lane_prfm)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_3x8__neon_mlal_lane)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_3x8__neon_mlal_lane_prfm)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_3x16__neon_mlal_lane)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_3x16__neon_mlal_lane_prfm)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x8__neon_mlal_lane)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x8__neon_mlal_lane_prfm)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x16__neon_mlal_lane)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x16__neon_mlal_lane_prfm)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_6x8__neon_mlal_lane)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_6x8__neon_mlal_lane_prfm)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_6x16__neon_mlal_lane)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_6x16__neon_mlal_lane_prfm)

DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8__neonv8_mlal_lane)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8__neonv8_mlal_lane_prfm)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x16__neonv8_mlal_lane)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x16__neonv8_mlal_lane_prfm)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x8__neonv8_mlal_lane)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x8__neonv8_mlal_lane_prfm)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x16__neonv8_mlal_lane)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x16__neonv8_mlal_lane_prfm)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_3x8__neonv8_mlal_lane)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_3x8__neonv8_mlal_lane_prfm)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_3x16__neonv8_mlal_lane)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_3x16__neonv8_mlal_lane_prfm)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x8__neonv8_mlal_lane)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x8__neonv8_mlal_lane_prfm)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x16__neonv8_mlal_lane)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x16__neonv8_mlal_lane_prfm)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_6x8__neonv8_mlal_lane)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_6x8__neonv8_mlal_lane_prfm)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_6x16__neonv8_mlal_lane)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_6x16__neonv8_mlal_lane_prfm)

DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8c4__neon_mlal_dup)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x8c4__neon_mlal_dup)

DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8c4__neonv8_mlal_dup)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x8c4__neonv8_mlal_dup)

DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8c4__neon_mlal_ld1r)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x8c4__neon_mlal_ld1r)

DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8c4__neonv8_mlal_ld1r)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x8c4__neonv8_mlal_ld1r)

DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8c4__neon_mlal_ld2r)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x8c4__neon_mlal_ld2r)

DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8c4__neonv8_mlal_ld2r)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x8c4__neonv8_mlal_ld2r)

DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8c4s2__neon_mlal)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x8c4s2__neon_mlal)

DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8c4s2__neonv8_mlal)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x8c4s2__neonv8_mlal)

DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8c2__neon_mlal_dup)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x8c2__neon_mlal_dup)

DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8c2__neonv8_mlal_dup)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x8c2__neonv8_mlal_dup)

DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8c2__neon_mlal_ld1r)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x8c2__neon_mlal_ld1r)

DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8c2__neonv8_mlal_ld1r)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x8c2__neonv8_mlal_ld1r)

DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8c2__neon_mlal_ld2r)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x8c2__neon_mlal_ld2r)

DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8c2__neonv8_mlal_ld2r)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x8c2__neonv8_mlal_ld2r)

DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8c2__neon_mlal_ld4r)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x8c2__neon_mlal_ld4r)

DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8c2__neonv8_mlal_ld4r)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x8c2__neonv8_mlal_ld4r)

DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8c2s4__neon_mlal)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x8c2s4__neon_mlal)

DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8c2s4__neonv8_mlal)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x8c2s4__neonv8_mlal)

DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8c8__neon_mlal)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x8c8__neon_mlal)

DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8c8__neonv8_mlal)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x8c8__neonv8_mlal)

DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8c4__neondot)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x8c4__neondot)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_6x8c4__neondot)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_8x8c4__neondot)

DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x16c4__neondot)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x16c4__neondot)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_6x16c4__neondot)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_8x16c4__neondot)

DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8c8__asm_aarch64_neon_mlal)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8c8__asm_aarch64_neon_mlal_cortex_a53)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8c8__asm_aarch64_neon_mlal_cortex_a53_prfm)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8c8__asm_aarch64_neon_mlal_prfm)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x8c8__asm_aarch64_neon_mlal)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x8c8__asm_aarch64_neon_mlal_cortex_a53)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x8c8__asm_aarch64_neon_mlal_cortex_a53_prfm)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x8c8__asm_aarch64_neon_mlal_prfm)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x8c16__asm_aarch64_neon_mlal)

DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x16__asm_aarch64_neon_mlal_lane_cortex_a53)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x16__asm_aarch64_neon_mlal_lane_cortex_a53_prfm)

DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x16__asm_aarch64_neon_mlal_lane_ld64)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x16__asm_aarch64_neon_mlal_lane_ld64_prfm)

DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x16c4__asm_aarch64_neondot_cortex_a55)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x16c4__asm_aarch64_neondot_ld64)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x16c4__asm_aarch64_neondot_ld128)

DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x8c4__asm_aarch32_neondot_cortex_a55)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x8c4__asm_aarch32_neondot_ld64)

DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8__asm_aarch32_neon_mlal_lane_cortex_a7)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8__asm_aarch32_neon_mlal_lane_cortex_a7_prfm)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8__asm_aarch32_neonv8_mlal_lane_cortex_a35)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8__asm_aarch32_neonv8_mlal_lane_cortex_a35_prfm)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a7)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a7_prfm)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a53)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a53_prfm)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x8__asm_aarch32_neon_mlal_lane_ld64)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x8__asm_aarch32_neon_mlal_lane_ld64_prfm)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x8__asm_aarch32_neonv8_mlal_lane_cortex_a35)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x8__asm_aarch32_neonv8_mlal_lane_cortex_a35_prfm)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x8__asm_aarch32_neonv8_mlal_lane_cortex_a53)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x8__asm_aarch32_neonv8_mlal_lane_cortex_a53_prfm)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x8__asm_aarch32_neonv8_mlal_lane_ld64)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x8__asm_aarch32_neonv8_mlal_lane_ld64_prfm)

DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4c2__sse2_ld64)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x4c2__sse2_ld64)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_3x4c2__sse2_ld64)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x4c2__sse2_ld64)

DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4c2__sse41_ld64)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x4c2__sse41_ld64)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_3x4c2__sse41_ld64)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x4c2__sse41_ld64)

DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4c2__avx_ld64)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x4c2__avx_ld64)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_3x4c2__avx_ld64)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x4c2__avx_ld64)

DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4c2__sse2_ld128)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x4c2__sse2_ld128)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_3x4c2__sse2_ld128)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x4c2__sse2_ld128)

DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4c2__sse41_ld128)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x4c2__sse41_ld128)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_3x4c2__sse41_ld128)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x4c2__sse41_ld128)

DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4c2__avx_ld128)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x4c2__avx_ld128)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_3x4c2__avx_ld128)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x4c2__avx_ld128)

DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4c2s4__sse2_ld64)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x4c2s4__sse2_ld64)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_3x4c2s4__sse2_ld64)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x4c2s4__sse2_ld64)

DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4c2s4__sse41_ld64)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x4c2s4__sse41_ld64)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_3x4c2s4__sse41_ld64)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x4c2s4__sse41_ld64)

DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4c2s4__avx_ld64)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x4c2s4__avx_ld64)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_3x4c2s4__avx_ld64)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x4c2s4__avx_ld64)

DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4c2s4__sse2_ld128)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x4c2s4__sse2_ld128)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_3x4c2s4__sse2_ld128)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x4c2s4__sse2_ld128)

DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4c2s4__sse41_ld128)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x4c2s4__sse41_ld128)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_3x4c2s4__sse41_ld128)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x4c2s4__sse41_ld128)

DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4c2s4__avx_ld128)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x4c2s4__avx_ld128)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_3x4c2s4__avx_ld128)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x4c2s4__avx_ld128)

DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4c8__sse2_ld64)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x4c8__sse2_ld64)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_3x4c8__sse2_ld64)

DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4c8__sse41_ld64)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x4c8__sse41_ld64)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_3x4c8__sse41_ld64)

DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4c8__avx_ld64)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x4c8__avx_ld64)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_3x4c8__avx_ld64)

DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4c8__sse2_ld128)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x4c8__sse2_ld128)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_3x4c8__sse2_ld128)

DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4c8__sse41_ld128)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x4c8__sse41_ld128)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_3x4c8__sse41_ld128)

DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4c8__avx_ld128)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x4c8__avx_ld128)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_3x4c8__avx_ld128)

DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8c8__avx2)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x8c8__avx2)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_3x8c8__avx2)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x8c8__avx2)

DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8c8__avx256skx)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x8c8__avx256skx)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_3x8c8__avx256skx)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x8c8__avx256skx)

DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x16c8__avx512skx)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_5x16c8__avx512skx)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_7x16c8__avx512skx)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_8x16c8__avx512skx)

DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x16c8__avx512skx_prfm)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_5x16c8__avx512skx_prfm)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_7x16c8__avx512skx_prfm)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_8x16c8__avx512skx_prfm)

DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4c2__wasmsimd_dot16x2_ld64)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x4c2__wasmsimd_dot16x2_ld64)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_3x4c2__wasmsimd_dot16x2_ld64)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x4c2__wasmsimd_dot16x2_ld64)

DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4c2__wasmsimd_dot16x2_ld128)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x4c2__wasmsimd_dot16x2_ld128)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_3x4c2__wasmsimd_dot16x2_ld128)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x4c2__wasmsimd_dot16x2_ld128)

DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4c2s4__wasmsimd_dot16x2_ld64)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x4c2s4__wasmsimd_dot16x2_ld64)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_3x4c2s4__wasmsimd_dot16x2_ld64)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x4c2s4__wasmsimd_dot16x2_ld64)

DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4c2s4__wasmsimd_dot16x2_ld128)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x4c2s4__wasmsimd_dot16x2_ld128)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_3x4c2s4__wasmsimd_dot16x2_ld128)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x4c2s4__wasmsimd_dot16x2_ld128)

DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4c8__wasmsimd_dot16x2_ld64)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x4c8__wasmsimd_dot16x2_ld64)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_3x4c8__wasmsimd_dot16x2_ld64)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x4c8__wasmsimd_dot16x2_ld64)

DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4c8__wasmsimd_dot16x2_ld128)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x4c8__wasmsimd_dot16x2_ld128)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_3x4c8__wasmsimd_dot16x2_ld128)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x4c8__wasmsimd_dot16x2_ld128)

DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4c16__wasmsdot)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8c16__wasmsdot)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x4c16__wasmsdot)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x8c16__wasmsdot)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_3x4c16__wasmsdot)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_3x8c16__wasmsdot)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x4c16__wasmsdot)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x8c16__wasmsdot)

DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4c16__wasmusdot)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8c16__wasmusdot)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x4c16__wasmusdot)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x8c16__wasmusdot)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_3x4c16__wasmusdot)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_3x8c16__wasmusdot)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x4c16__wasmusdot)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x8c16__wasmusdot)

DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8c8__wasmusdot)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x8c8__wasmusdot)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_3x8c8__wasmusdot)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x8c8__wasmusdot)

DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8c8__wasmusdot_u2)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x8c8__wasmusdot_u2)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_3x8c8__wasmusdot_u2)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x8c8__wasmusdot_u2)

DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8c8__wasmsdot)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x8c8__wasmsdot)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_3x8c8__wasmsdot)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x8c8__wasmsdot)

DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8c8__wasmsdot_u2)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x8c8__wasmsdot_u2)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_3x8c8__wasmsdot_u2)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x8c8__wasmsdot_u2)

DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x16c4__wasmusdot)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_3x16c4__wasmusdot)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x16c4__wasmusdot)

DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x16c4__wasmusdot_u2)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_3x16c4__wasmusdot_u2)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x16c4__wasmusdot_u2)

DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x16c4__wasmusdot_u2_acc2)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_3x16c4__wasmusdot_u2_acc2)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x16c4__wasmusdot_u2_acc2)

DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x16c4__wasmsdot)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_3x16c4__wasmsdot)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x16c4__wasmsdot)

DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x16c4__wasmsdot_u2)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_3x16c4__wasmsdot_u2)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x16c4__wasmsdot_u2)

DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x16c4__wasmsdot_u2_acc2)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_3x16c4__wasmsdot_u2_acc2)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x16c4__wasmsdot_u2_acc2)

DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x2__wasm_fmagic)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4__wasm_fmagic)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x2__wasm_fmagic)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x4__wasm_fmagic)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_3x2__wasm_fmagic)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_3x4__wasm_fmagic)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x2__wasm_fmagic)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x4__wasm_fmagic)

DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x1c4__armsimd32)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x2c4__armsimd32)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x1c4__armsimd32)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x2c4__armsimd32)

DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x2__scalar_fmagic)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4__scalar_fmagic)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x2__scalar_fmagic)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x4__scalar_fmagic)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_3x2__scalar_fmagic)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_3x4__scalar_fmagic)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x2__scalar_fmagic)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x4__scalar_fmagic)

DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x2__scalar_imagic)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4__scalar_imagic)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x2__scalar_imagic)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x4__scalar_imagic)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_3x2__scalar_imagic)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_3x4__scalar_imagic)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x2__scalar_imagic)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x4__scalar_imagic)

DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x2__scalar_lrintf)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4__scalar_lrintf)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x2__scalar_lrintf)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x4__scalar_lrintf)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_3x2__scalar_lrintf)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_3x4__scalar_lrintf)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x2__scalar_lrintf)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x4__scalar_lrintf)

DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8c8__neoni8mm)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x16c8__neoni8mm)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x8c8__neoni8mm)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x16c8__neoni8mm)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_3x8c8__neoni8mm)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_3x16c8__neoni8mm)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x8c8__neoni8mm)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x16c8__neoni8mm)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_6x8c8__neoni8mm)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_6x16c8__neoni8mm)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_8x8c8__neoni8mm)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_8x16c8__neoni8mm)

DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8c8__aarch64_neondot_ld128)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8c8__neondot_ld64)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x16c8__aarch64_neondot_ld128)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x16c8__neondot_ld64)

DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x16c4__avx512amx)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_7x16c4__avx512amx)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_16x16c4__avx512amx)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_16x16c4__avx512amx_prfm)

DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x32c4__avx512amx)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_7x32c4__avx512amx)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_16x32c4__avx512amx)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_16x32c4__avx512amx_prfm)

DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x64c4__avx512amx)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_7x64c4__avx512amx)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_16x64c4__avx512amx)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_16x64c4__avx512amx_prfm)

DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x16c4__avx512vnni)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x16c4__avx512vnni)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_5x16c4__avx512vnni)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_7x16c4__avx512vnni)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_8x16c4__avx512vnni)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_9x16c4__avx512vnni)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_10x16c4__avx512vnni)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_12x16c4__avx512vnni)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_14x16c4__avx512vnni)

DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x16c4__avx512vnni_prfm)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x16c4__avx512vnni_prfm)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_5x16c4__avx512vnni_prfm)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_7x16c4__avx512vnni_prfm)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_8x16c4__avx512vnni_prfm)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_9x16c4__avx512vnni_prfm)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_10x16c4__avx512vnni_prfm)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_12x16c4__avx512vnni_prfm)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_14x16c4__avx512vnni_prfm)

DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x16c8__avx512vnni)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_5x16c8__avx512vnni)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_7x16c8__avx512vnni)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_8x16c8__avx512vnni)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_9x16c8__avx512vnni)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_10x16c8__avx512vnni)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_12x16c8__avx512vnni)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_14x16c8__avx512vnni)

DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x16c8__avx512vnni_prfm)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_5x16c8__avx512vnni_prfm)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_7x16c8__avx512vnni_prfm)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_8x16c8__avx512vnni_prfm)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_9x16c8__avx512vnni_prfm)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_10x16c8__avx512vnni_prfm)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_12x16c8__avx512vnni_prfm)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_14x16c8__avx512vnni_prfm)

DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8c8__avx256vnni)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_5x8c8__avx256vnni)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_7x8c8__avx256vnni)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_8x8c8__avx256vnni)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_9x8c8__avx256vnni)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_10x8c8__avx256vnni)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_12x8c8__avx256vnni)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_14x8c8__avx256vnni)

DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8c8__avx256vnni_prfm)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_5x8c8__avx256vnni_prfm)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_7x8c8__avx256vnni_prfm)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_8x8c8__avx256vnni_prfm)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_9x8c8__avx256vnni_prfm)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_10x8c8__avx256vnni_prfm)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_12x8c8__avx256vnni_prfm)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_14x8c8__avx256vnni_prfm)

DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8c8__avxvnni)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x8c8__avxvnni)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_3x8c8__avxvnni)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x8c8__avxvnni)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_5x8c8__avxvnni)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_6x8c8__avxvnni)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_7x8c8__avxvnni)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_8x8c8__avxvnni)

DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8c8__avxvnni_prfm)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x8c8__avxvnni_prfm)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_3x8c8__avxvnni_prfm)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x8c8__avxvnni_prfm)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_5x8c8__avxvnni_prfm)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_6x8c8__avxvnni_prfm)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_7x8c8__avxvnni_prfm)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_8x8c8__avxvnni_prfm)

DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8c8__avxvnniint8_prfm)
DECLARE_QS8_QC8W_IGEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_5x8c8__avxvnniint8_prfm)

#ifdef __cplusplus
}  // extern "C"
#endif
