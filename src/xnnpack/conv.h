// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
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


#define DECLARE_F32_CONV_HWC_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                           \
      size_t input_height,                             \
      size_t input_width,                              \
      size_t output_y_start,                           \
      size_t output_y_end,                             \
      const float* input,                              \
      const float* zero,                               \
      const float* weights,                            \
      float* output,                                   \
      size_t input_padding_top,                        \
      size_t output_channels,                          \
      size_t output_height_stride,                     \
      size_t output_width_stride,                      \
      const union xnn_f32_minmax_params* params);

DECLARE_F32_CONV_HWC_UKERNEL_FUNCTION(xnn_f32_conv_hwc_ukernel_3x3s2p1c3x8__neonfma_2x2)
DECLARE_F32_CONV_HWC_UKERNEL_FUNCTION(xnn_f32_conv_hwc_ukernel_3x3s2p1c3x4__neonfma_2x2)
DECLARE_F32_CONV_HWC_UKERNEL_FUNCTION(xnn_f32_conv_hwc_ukernel_3x3s2p0p1c3x4__scalar_1x1)
DECLARE_F32_CONV_HWC_UKERNEL_FUNCTION(xnn_f32_conv_hwc_ukernel_3x3s2p1c3x4__scalar_1x1)


#define DECLARE_F32_CONV_HWC2SPCHW_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                                 \
      size_t input_height,                                   \
      size_t input_width,                                    \
      size_t output_y_start,                                 \
      size_t output_y_end,                                   \
      const float* input,                                    \
      const float* zero,                                     \
      const float* weights,                                  \
      float* output,                                         \
      size_t input_padding_top,                              \
      size_t output_channels,                                \
      size_t output_height_stride,                           \
      size_t output_channel_stride,                          \
      const union xnn_f32_minmax_params* params);

DECLARE_F32_CONV_HWC2SPCHW_UKERNEL_FUNCTION(xnn_f32_conv_hwc2spchw_ukernel_3x3s2p1c3x4__neonfma_2x2)
DECLARE_F32_CONV_HWC2SPCHW_UKERNEL_FUNCTION(xnn_f32_conv_hwc2spchw_ukernel_3x3s2p1c3x4__scalar_1x1)


#ifdef __cplusplus
}  // extern "C"
#endif
