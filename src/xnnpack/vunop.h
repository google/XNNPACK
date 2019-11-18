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


#define DECLARE_F32_VUNOP_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                        \
      size_t n,                                     \
      const float* x,                               \
      float* y,                                     \
      const void* params);

DECLARE_F32_VUNOP_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__neonfma_p5_nr2fma_x16)
DECLARE_F32_VUNOP_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__neon_frac_p9_p10_nr1recps_x16)
DECLARE_F32_VUNOP_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__sse2_p5_div_x8)
DECLARE_F32_VUNOP_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__sse2_p5_div_x16)


#ifdef __cplusplus
}  // extern "C"
#endif
