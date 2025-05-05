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

#include "src/xnnpack/common.h"
#include "src/xnnpack/microparams.h"

#ifdef __cplusplus
extern "C" {
#endif

#define XNN_UKERNEL(arch_flags, fn_name, kernel_size, subsampling,    \
                    padding_right, padding_left, input_channels,      \
                    output_channels_tile, input_widths, datatype)     \
  XNN_INTERNAL void fn_name(                                          \
      size_t input_height, size_t input_width, size_t output_y_start, \
      size_t output_y_end, const float* input, const float* zero,     \
      const float* weights, float* output, size_t input_padding_top,  \
      size_t output_channels, size_t output_height_stride,            \
      size_t output_width_stride,                                     \
      const struct xnn_f32_minmax_params                              \
          params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);
#include "src/f32-conv-hwc/f32-conv-hwc.h"
#undef XNN_UKERNEL

#define DECLARE_F32_CONV_HWC2CHW_UKERNEL_FUNCTION(fn_name)            \
  XNN_INTERNAL void fn_name(                                          \
      size_t input_height, size_t input_width, size_t output_y_start, \
      size_t output_y_end, const float* input, const float* zero,     \
      const float* weights, float* output, size_t input_padding_top,  \
      size_t output_channels, size_t output_height_stride,            \
      size_t output_channel_stride,                                   \
      const struct xnn_f32_minmax_params                              \
          params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

DECLARE_F32_CONV_HWC2CHW_UKERNEL_FUNCTION(
    xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x4__aarch64_neonfma_2x2)
DECLARE_F32_CONV_HWC2CHW_UKERNEL_FUNCTION(
    xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x4__neon_2x2)
DECLARE_F32_CONV_HWC2CHW_UKERNEL_FUNCTION(
    xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x4__scalar_1x1)
DECLARE_F32_CONV_HWC2CHW_UKERNEL_FUNCTION(
    xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x4__sse_1x1)
DECLARE_F32_CONV_HWC2CHW_UKERNEL_FUNCTION(
    xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x4__sse_2x2)
DECLARE_F32_CONV_HWC2CHW_UKERNEL_FUNCTION(
    xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x4__wasmsimd_2x2)

#define DECLARE_F16_CONV_HWC2CHW_UKERNEL_FUNCTION(fn_name)                    \
  XNN_INTERNAL void fn_name(                                                  \
      size_t input_height, size_t input_width, size_t output_y_start,         \
      size_t output_y_end, const xnn_float16* input, const xnn_float16* zero, \
      const xnn_float16* weights, xnn_float16* output,                        \
      size_t input_padding_top, size_t output_channels,                       \
      size_t output_height_stride, size_t output_channel_stride,              \
      const struct xnn_f16_minmax_params                                      \
          params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

#define DECLARE_F32_CONV_HWC2CHW_UKERNEL_FUNCTION(fn_name) \
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
      const struct xnn_f32_minmax_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

DECLARE_F32_CONV_HWC2CHW_UKERNEL_FUNCTION(
    xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x4__aarch64_neonfma_2x2)
DECLARE_F32_CONV_HWC2CHW_UKERNEL_FUNCTION(
    xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x4__neon_2x2)
DECLARE_F32_CONV_HWC2CHW_UKERNEL_FUNCTION(
    xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x4__scalar_1x1)
DECLARE_F32_CONV_HWC2CHW_UKERNEL_FUNCTION(
    xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x4__sse_1x1)
DECLARE_F32_CONV_HWC2CHW_UKERNEL_FUNCTION(
    xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x4__sse_2x2)
DECLARE_F32_CONV_HWC2CHW_UKERNEL_FUNCTION(
    xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x4__wasmsimd_2x2)
DECLARE_F32_CONV_HWC2CHW_UKERNEL_FUNCTION(
    xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x2v__rvv_1x1)
DECLARE_F32_CONV_HWC2CHW_UKERNEL_FUNCTION(
    xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x2v__rvv_2x1)
DECLARE_F32_CONV_HWC2CHW_UKERNEL_FUNCTION(
    xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x2v__rvv_2x2)

#define DECLARE_F16_CONV_HWC2CHW_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                                 \
      size_t input_height,                                   \
      size_t input_width,                                    \
      size_t output_y_start,                                 \
      size_t output_y_end,                                   \
      const xnn_float16* input,                     \
      const xnn_float16* zero,                      \
      const xnn_float16* weights,                   \
      xnn_float16* output,                          \
      size_t input_padding_top,                              \
      size_t output_channels,                                \
      size_t output_height_stride,                           \
      size_t output_channel_stride,                          \
      const struct xnn_f16_minmax_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

DECLARE_F16_CONV_HWC2CHW_UKERNEL_FUNCTION(
    xnn_f16_conv_hwc2chw_ukernel_3x3s2p1c3x4__neonfp16arith_2x2)

#ifdef __cplusplus
}  // extern "C"
#endif
