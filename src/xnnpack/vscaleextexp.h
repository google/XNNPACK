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

#define DECLARE_F32_VSCALEEXTEXP_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                               \
      size_t n,                                            \
      const float* input,                                  \
      float* output,                                       \
      float scale_mantissa,                                \
      float scale_exponent);

DECLARE_F32_VSCALEEXTEXP_UKERNEL_FUNCTION(xnn_f32_vscaleextexp_ukernel__avx2_p5_x8)
DECLARE_F32_VSCALEEXTEXP_UKERNEL_FUNCTION(xnn_f32_vscaleextexp_ukernel__avx2_p5_x16)
DECLARE_F32_VSCALEEXTEXP_UKERNEL_FUNCTION(xnn_f32_vscaleextexp_ukernel__avx2_p5_x24)
DECLARE_F32_VSCALEEXTEXP_UKERNEL_FUNCTION(xnn_f32_vscaleextexp_ukernel__avx2_p5_x32)
DECLARE_F32_VSCALEEXTEXP_UKERNEL_FUNCTION(xnn_f32_vscaleextexp_ukernel__avx2_p5_x40)
DECLARE_F32_VSCALEEXTEXP_UKERNEL_FUNCTION(xnn_f32_vscaleextexp_ukernel__avx2_p5_x48)
DECLARE_F32_VSCALEEXTEXP_UKERNEL_FUNCTION(xnn_f32_vscaleextexp_ukernel__avx2_p5_x56)
DECLARE_F32_VSCALEEXTEXP_UKERNEL_FUNCTION(xnn_f32_vscaleextexp_ukernel__avx2_p5_x64)
DECLARE_F32_VSCALEEXTEXP_UKERNEL_FUNCTION(xnn_f32_vscaleextexp_ukernel__avx2_p5_x72)
DECLARE_F32_VSCALEEXTEXP_UKERNEL_FUNCTION(xnn_f32_vscaleextexp_ukernel__avx2_p5_x80)
DECLARE_F32_VSCALEEXTEXP_UKERNEL_FUNCTION(xnn_f32_vscaleextexp_ukernel__avx2_p5_x88)
DECLARE_F32_VSCALEEXTEXP_UKERNEL_FUNCTION(xnn_f32_vscaleextexp_ukernel__avx2_p5_x96)

DECLARE_F32_VSCALEEXTEXP_UKERNEL_FUNCTION(xnn_f32_vscaleextexp_ukernel__avx512f_p5_scalef_x16)
DECLARE_F32_VSCALEEXTEXP_UKERNEL_FUNCTION(xnn_f32_vscaleextexp_ukernel__avx512f_p5_scalef_x32)
DECLARE_F32_VSCALEEXTEXP_UKERNEL_FUNCTION(xnn_f32_vscaleextexp_ukernel__avx512f_p5_scalef_x48)
DECLARE_F32_VSCALEEXTEXP_UKERNEL_FUNCTION(xnn_f32_vscaleextexp_ukernel__avx512f_p5_scalef_x64)
DECLARE_F32_VSCALEEXTEXP_UKERNEL_FUNCTION(xnn_f32_vscaleextexp_ukernel__avx512f_p5_scalef_x80)
DECLARE_F32_VSCALEEXTEXP_UKERNEL_FUNCTION(xnn_f32_vscaleextexp_ukernel__avx512f_p5_scalef_x96)
DECLARE_F32_VSCALEEXTEXP_UKERNEL_FUNCTION(xnn_f32_vscaleextexp_ukernel__avx512f_p5_scalef_x112)
DECLARE_F32_VSCALEEXTEXP_UKERNEL_FUNCTION(xnn_f32_vscaleextexp_ukernel__avx512f_p5_scalef_x128)
DECLARE_F32_VSCALEEXTEXP_UKERNEL_FUNCTION(xnn_f32_vscaleextexp_ukernel__avx512f_p5_scalef_x144)
DECLARE_F32_VSCALEEXTEXP_UKERNEL_FUNCTION(xnn_f32_vscaleextexp_ukernel__avx512f_p5_scalef_x160)
DECLARE_F32_VSCALEEXTEXP_UKERNEL_FUNCTION(xnn_f32_vscaleextexp_ukernel__avx512f_p5_scalef_x176)
DECLARE_F32_VSCALEEXTEXP_UKERNEL_FUNCTION(xnn_f32_vscaleextexp_ukernel__avx512f_p5_scalef_x192)

#ifdef __cplusplus
} /* extern "C" */
#endif
