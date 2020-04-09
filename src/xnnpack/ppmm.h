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


#define DECLARE_F32_PPMM_MINMAX_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                       \
      size_t mr,                                   \
      size_t nc,                                   \
      size_t kc,                                   \
      const float* a,                              \
      const float* w,                              \
      float* c,                                    \
      size_t cm_stride,                            \
      size_t cn_stride,                            \
      const union xnn_f32_minmax_params* params);

DECLARE_F32_PPMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_ppmm_minmax_ukernel_2x4__scalar)
DECLARE_F32_PPMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_ppmm_minmax_ukernel_3x3__scalar)
DECLARE_F32_PPMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_ppmm_minmax_ukernel_4x2__scalar)
DECLARE_F32_PPMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_ppmm_minmax_ukernel_4x4__scalar)
DECLARE_F32_PPMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_ppmm_minmax_ukernel_4x8__neon)
DECLARE_F32_PPMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_ppmm_minmax_ukernel_4x8__neonfma)
DECLARE_F32_PPMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_ppmm_minmax_ukernel_4x8__psimd)
DECLARE_F32_PPMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_ppmm_minmax_ukernel_4x8__sse)
DECLARE_F32_PPMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_ppmm_minmax_ukernel_8x8__neon)
DECLARE_F32_PPMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_ppmm_minmax_ukernel_8x8__neonfma)


#ifdef __cplusplus
}  // extern "C"
#endif
