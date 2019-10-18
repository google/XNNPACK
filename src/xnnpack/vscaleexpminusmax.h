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

#define DECLARE_F32_VSCALEEXPMINUSMAX_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                                    \
      size_t n,                                                 \
      const float* input,                                       \
      float* output,                                            \
      float scale,                                              \
      float max);

DECLARE_F32_VSCALEEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_vscaleexpminusmax_ukernel__avx2_p5_unroll64)
DECLARE_F32_VSCALEEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_vscaleexpminusmax_ukernel__avx512f_p5_scalef_unroll128)

#ifdef __cplusplus
} /* extern "C" */
#endif
