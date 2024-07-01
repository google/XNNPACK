// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <stddef.h>
#include <stdint.h>

#include "xnnpack/common.h"

#ifdef __cplusplus
extern "C" {
#endif

#define DECLARE_F32_VSCALEEXPMINUSMAX_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                                    \
      size_t n,                                                 \
      const float* input,                                       \
      float* output,                                            \
      float scale,                                              \
      float max);

DECLARE_F32_VSCALEEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_vscaleexpminusmax_ukernel__avx2_p5_u8)
DECLARE_F32_VSCALEEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_vscaleexpminusmax_ukernel__avx2_p5_u16)
DECLARE_F32_VSCALEEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_vscaleexpminusmax_ukernel__avx2_p5_u24)
DECLARE_F32_VSCALEEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_vscaleexpminusmax_ukernel__avx2_p5_u32)
DECLARE_F32_VSCALEEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_vscaleexpminusmax_ukernel__avx2_p5_u40)
DECLARE_F32_VSCALEEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_vscaleexpminusmax_ukernel__avx2_p5_u48)
DECLARE_F32_VSCALEEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_vscaleexpminusmax_ukernel__avx2_p5_u56)
DECLARE_F32_VSCALEEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_vscaleexpminusmax_ukernel__avx2_p5_u64)
DECLARE_F32_VSCALEEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_vscaleexpminusmax_ukernel__avx2_p5_u72)
DECLARE_F32_VSCALEEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_vscaleexpminusmax_ukernel__avx2_p5_u80)
DECLARE_F32_VSCALEEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_vscaleexpminusmax_ukernel__avx2_p5_u88)
DECLARE_F32_VSCALEEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_vscaleexpminusmax_ukernel__avx2_p5_u96)

DECLARE_F32_VSCALEEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_vscaleexpminusmax_ukernel__avx512f_p5_scalef_u16)
DECLARE_F32_VSCALEEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_vscaleexpminusmax_ukernel__avx512f_p5_scalef_u32)
DECLARE_F32_VSCALEEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_vscaleexpminusmax_ukernel__avx512f_p5_scalef_u48)
DECLARE_F32_VSCALEEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_vscaleexpminusmax_ukernel__avx512f_p5_scalef_u64)
DECLARE_F32_VSCALEEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_vscaleexpminusmax_ukernel__avx512f_p5_scalef_u80)
DECLARE_F32_VSCALEEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_vscaleexpminusmax_ukernel__avx512f_p5_scalef_u96)
DECLARE_F32_VSCALEEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_vscaleexpminusmax_ukernel__avx512f_p5_scalef_u112)
DECLARE_F32_VSCALEEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_vscaleexpminusmax_ukernel__avx512f_p5_scalef_u128)
DECLARE_F32_VSCALEEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_vscaleexpminusmax_ukernel__avx512f_p5_scalef_u144)
DECLARE_F32_VSCALEEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_vscaleexpminusmax_ukernel__avx512f_p5_scalef_u160)
DECLARE_F32_VSCALEEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_vscaleexpminusmax_ukernel__avx512f_p5_scalef_u176)
DECLARE_F32_VSCALEEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_vscaleexpminusmax_ukernel__avx512f_p5_scalef_u192)

#ifdef __cplusplus
} /* extern "C" */
#endif
