// Copyright 2020 Google LLC
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

#define DECLARE_X32_DEPTHTOSPACE2D_CHW2HWC_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                                         \
      size_t output_channels,                                        \
      size_t input_height,                                           \
      size_t input_width,                                            \
      size_t block_size,                                             \
      const uint32_t* input,                                         \
      uint32_t* output,                                              \
      size_t output_channel_stride);

DECLARE_X32_DEPTHTOSPACE2D_CHW2HWC_UKERNEL_FUNCTION(xnn_x32_depthtospace2d_chw2hwc_ukernel__scalar)

#ifdef __cplusplus
}  // extern "C"
#endif
