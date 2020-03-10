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


#define DECLARE_F32_IBILINEAR_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                            \
      size_t output_pixels,                             \
      size_t channels,                                  \
      const float** input,                              \
      size_t input_offset,                              \
      const float* weights,                             \
      float* output,                                    \
      size_t output_increment);

DECLARE_F32_IBILINEAR_UKERNEL_FUNCTION(xnn_f32_ibilinear_ukernel__scalar_c1)
DECLARE_F32_IBILINEAR_UKERNEL_FUNCTION(xnn_f32_ibilinear_ukernel__scalar_c2)
DECLARE_F32_IBILINEAR_UKERNEL_FUNCTION(xnn_f32_ibilinear_ukernel__scalar_c4)

DECLARE_F32_IBILINEAR_UKERNEL_FUNCTION(xnn_f32_ibilinear_ukernel__neon_c4)
DECLARE_F32_IBILINEAR_UKERNEL_FUNCTION(xnn_f32_ibilinear_ukernel__neon_c8)

DECLARE_F32_IBILINEAR_UKERNEL_FUNCTION(xnn_f32_ibilinear_ukernel__neonfma_c4)
DECLARE_F32_IBILINEAR_UKERNEL_FUNCTION(xnn_f32_ibilinear_ukernel__neonfma_c8)

DECLARE_F32_IBILINEAR_UKERNEL_FUNCTION(xnn_f32_ibilinear_ukernel__sse_c4)
DECLARE_F32_IBILINEAR_UKERNEL_FUNCTION(xnn_f32_ibilinear_ukernel__sse_c8)

DECLARE_F32_IBILINEAR_UKERNEL_FUNCTION(xnn_f32_ibilinear_ukernel__psimd_c4)
DECLARE_F32_IBILINEAR_UKERNEL_FUNCTION(xnn_f32_ibilinear_ukernel__psimd_c8)


#ifdef __cplusplus
}  // extern "C"
#endif
