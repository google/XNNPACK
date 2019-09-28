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


#define DECLARE_F32_HSWISH_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                         \
      size_t n,                                      \
      const float* x,                                \
      float* y,                                      \
      const union xnn_f32_hswish_params* params);

DECLARE_F32_HSWISH_UKERNEL_FUNCTION(xnn_f32_hswish_ukernel__psimd)
DECLARE_F32_HSWISH_UKERNEL_FUNCTION(xnn_f32_hswish_ukernel__neon)
DECLARE_F32_HSWISH_UKERNEL_FUNCTION(xnn_f32_hswish_ukernel__neonfma)
DECLARE_F32_HSWISH_UKERNEL_FUNCTION(xnn_f32_hswish_ukernel__sse)
DECLARE_F32_HSWISH_UKERNEL_FUNCTION(xnn_f32_hswish_ukernel__scalar)


#ifdef __cplusplus
} /* extern "C" */
#endif
