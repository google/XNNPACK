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

#define DECLARE_F32_RADDEXPMINUSMAX_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                                  \
      size_t n,                                               \
      const float* input,                                     \
      float* sum,                                             \
      float max);

DECLARE_F32_RADDEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddexpminusmax_ukernel__avx2_p5_x64)
DECLARE_F32_RADDEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddexpminusmax_ukernel__avx2_p5_x64_acc2)
DECLARE_F32_RADDEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddexpminusmax_ukernel__avx2_p5_x64_acc4)
DECLARE_F32_RADDEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddexpminusmax_ukernel__avx2_p5_x72)
DECLARE_F32_RADDEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddexpminusmax_ukernel__avx2_p5_x72_acc3)
DECLARE_F32_RADDEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddexpminusmax_ukernel__avx2_p5_x80)
DECLARE_F32_RADDEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddexpminusmax_ukernel__avx2_p5_x80_acc2)
DECLARE_F32_RADDEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddexpminusmax_ukernel__avx2_p5_x80_acc5)
DECLARE_F32_RADDEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddexpminusmax_ukernel__avx2_p5_x96)
DECLARE_F32_RADDEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddexpminusmax_ukernel__avx2_p5_x96_acc2)
DECLARE_F32_RADDEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddexpminusmax_ukernel__avx2_p5_x96_acc3)
DECLARE_F32_RADDEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddexpminusmax_ukernel__avx2_p5_x96_acc6)

DECLARE_F32_RADDEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddexpminusmax_ukernel__avx512f_p5_scalef_x128)
DECLARE_F32_RADDEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddexpminusmax_ukernel__avx512f_p5_scalef_x128_acc2)
DECLARE_F32_RADDEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddexpminusmax_ukernel__avx512f_p5_scalef_x128_acc4)
DECLARE_F32_RADDEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddexpminusmax_ukernel__avx512f_p5_scalef_x144)
DECLARE_F32_RADDEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddexpminusmax_ukernel__avx512f_p5_scalef_x144_acc3)
DECLARE_F32_RADDEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddexpminusmax_ukernel__avx512f_p5_scalef_x160)
DECLARE_F32_RADDEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddexpminusmax_ukernel__avx512f_p5_scalef_x160_acc2)
DECLARE_F32_RADDEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddexpminusmax_ukernel__avx512f_p5_scalef_x160_acc5)
DECLARE_F32_RADDEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddexpminusmax_ukernel__avx512f_p5_scalef_x192)
DECLARE_F32_RADDEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddexpminusmax_ukernel__avx512f_p5_scalef_x192_acc2)
DECLARE_F32_RADDEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddexpminusmax_ukernel__avx512f_p5_scalef_x192_acc3)
DECLARE_F32_RADDEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddexpminusmax_ukernel__avx512f_p5_scalef_x192_acc6)

#ifdef __cplusplus
} /* extern "C" */
#endif
