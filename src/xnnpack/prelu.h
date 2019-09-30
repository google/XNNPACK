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


#define DECLARE_F32_PRELU_UKERNEL_FUNCTION(fn_name)            \
  XNN_INTERNAL void fn_name(                                   \
      size_t mr,                                               \
      size_t n,                                                \
      const float* x,                                          \
      size_t x_stride,                                         \
      const float* w,                                          \
      float* y,                                                \
      size_t y_stride,                                         \
      const union xnn_f32_output_params* clamping_params);


DECLARE_F32_PRELU_UKERNEL_FUNCTION(xnn_f32_prelu_ukernel_x4__psimd)
DECLARE_F32_PRELU_UKERNEL_FUNCTION(xnn_f32_prelu_ukernel_x4__scalar)
DECLARE_F32_PRELU_UKERNEL_FUNCTION(xnn_f32_prelu_ukernel_x4__sse)


#ifdef __cplusplus
}  // extern "C"
#endif
