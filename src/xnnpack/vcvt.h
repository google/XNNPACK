// Copyright 2021 Google LLC
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


#define DECLARE_F16_F32_VCVT_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                           \
      size_t n,                                        \
      const void* input,                               \
      float* output,                                   \
      const void* params);

DECLARE_F16_F32_VCVT_UKERNEL_FUNCTION(xnn_f16_f32_vcvt_ukernel__f16c_x8)
DECLARE_F16_F32_VCVT_UKERNEL_FUNCTION(xnn_f16_f32_vcvt_ukernel__f16c_x16)

DECLARE_F16_F32_VCVT_UKERNEL_FUNCTION(xnn_f16_f32_vcvt_ukernel__avx512skx_x16)
DECLARE_F16_F32_VCVT_UKERNEL_FUNCTION(xnn_f16_f32_vcvt_ukernel__avx512skx_x32)

#ifdef __cplusplus
}  // extern "C"
#endif
