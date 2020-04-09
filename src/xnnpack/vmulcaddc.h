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


#define DECLARE_F32_VMULCADDC_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                            \
      size_t m,                                         \
      size_t c,                                         \
      const float* x,                                   \
      size_t x_stride,                                  \
      const float* w,                                   \
      float* y,                                         \
      size_t y_stride,                                  \
      const union xnn_f32_minmax_params* params);

DECLARE_F32_VMULCADDC_UKERNEL_FUNCTION(xnn_f32_vmulcaddc_ukernel_c4__neon_2x)
DECLARE_F32_VMULCADDC_UKERNEL_FUNCTION(xnn_f32_vmulcaddc_ukernel_c8__neon_2x)

DECLARE_F32_VMULCADDC_UKERNEL_FUNCTION(xnn_f32_vmulcaddc_ukernel_c4__neonfma_2x)
DECLARE_F32_VMULCADDC_UKERNEL_FUNCTION(xnn_f32_vmulcaddc_ukernel_c8__neonfma_2x)

DECLARE_F32_VMULCADDC_UKERNEL_FUNCTION(xnn_f32_vmulcaddc_ukernel_c4__psimd_2x)
DECLARE_F32_VMULCADDC_UKERNEL_FUNCTION(xnn_f32_vmulcaddc_ukernel_c8__psimd_2x)

DECLARE_F32_VMULCADDC_UKERNEL_FUNCTION(xnn_f32_vmulcaddc_ukernel_c4__sse_2x)
DECLARE_F32_VMULCADDC_UKERNEL_FUNCTION(xnn_f32_vmulcaddc_ukernel_c8__sse_2x)

DECLARE_F32_VMULCADDC_UKERNEL_FUNCTION(xnn_f32_vmulcaddc_ukernel_c1__wasm_2x)
DECLARE_F32_VMULCADDC_UKERNEL_FUNCTION(xnn_f32_vmulcaddc_ukernel_c2__wasm_2x)
DECLARE_F32_VMULCADDC_UKERNEL_FUNCTION(xnn_f32_vmulcaddc_ukernel_c4__wasm_2x)

DECLARE_F32_VMULCADDC_UKERNEL_FUNCTION(xnn_f32_vmulcaddc_ukernel_c1__scalar_2x)
DECLARE_F32_VMULCADDC_UKERNEL_FUNCTION(xnn_f32_vmulcaddc_ukernel_c2__scalar_2x)
DECLARE_F32_VMULCADDC_UKERNEL_FUNCTION(xnn_f32_vmulcaddc_ukernel_c4__scalar_2x)


#ifdef __cplusplus
}  // extern "C"
#endif
