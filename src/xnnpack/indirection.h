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
#include <stdbool.h>

#include <xnnpack.h>
#include <xnnpack/common.h>

#ifdef __cplusplus
extern "C" {
#endif

XNN_INTERNAL void xnn_indirection_init_conv2d(
  xnn_operator_t op,
  size_t output_tile_size,
  uint32_t log2_element_size);

XNN_INTERNAL void xnn_indirection_init_dwconv2d(
  xnn_operator_t op,
  size_t step_height,
  size_t step_width,
  size_t primary_tile,
  uint32_t log2_element_size);

XNN_INTERNAL void xnn_indirection_init_deconv2d(
  xnn_operator_t op,
  size_t output_tile_size,
  uint32_t log2_element_size);

XNN_INTERNAL void xnn_indirection_init_subconv2d(
  xnn_operator_t op,
  size_t output_tile_size,
  uint32_t log2_element_size);

XNN_INTERNAL void xnn_indirection_init_maxpool2d(
  xnn_operator_t op,
  size_t step_height,
  size_t step_width,
  uint32_t log2_element_size);

typedef void (*xnn_indirection_init_resize_bilinear2d_hwc_fn)(
  size_t input_pixel_stride,
  size_t input_height,
  size_t input_width,
  size_t output_height,
  size_t output_width,
  const void* input,
  const void** indirection_buffer,
  void* packed_weights,
  bool align_corners,
  bool tensorflow_legacy);

XNN_INTERNAL void xnn_indirection_init_resize_bilinear2d_hwc_f16(
  size_t input_pixel_stride,
  size_t input_height,
  size_t input_width,
  size_t output_height,
  size_t output_width,
  const void* input,
  const void** indirection_buffer,
  void* packed_weights,
  bool align_corners,
  bool tensorflow_legacy);

XNN_INTERNAL void xnn_indirection_init_resize_bilinear2d_hwc_f32(
  size_t input_pixel_stride,
  size_t input_height,
  size_t input_width,
  size_t output_height,
  size_t output_width,
  const void* input,
  const void** indirection_buffer,
  float* packed_weights,
  bool align_corners,
  bool tensorflow_legacy);

XNN_INTERNAL void xnn_indirection_init_resize_bilinear2d_hwc_q11(
  size_t input_pixel_stride,
  size_t input_height,
  size_t input_width,
  size_t output_height,
  size_t output_width,
  const void* input,
  const void** indirection_buffer,
  int16_t* packed_weights,
  bool align_corners,
  bool tensorflow_legacy);

typedef void (*xnn_indirection_init_resize_bilinear2d_chw_fn)(
  size_t input_pixel_stride,
  size_t input_height,
  size_t input_width,
  size_t output_height,
  size_t output_width,
  const void* input,
  const void** indirection_buffer,
  void* packed_weights,
  bool align_corners,
  bool tensorflow_legacy);

XNN_INTERNAL void xnn_indirection_init_resize_bilinear2d_chw_f16(
  size_t input_pixel_stride,
  size_t input_height,
  size_t input_width,
  size_t output_height,
  size_t output_width,
  const void* input,
  const void** indirection_buffer,
  void* packed_weights,
  bool align_corners,
  bool tensorflow_legacy);

XNN_INTERNAL void xnn_indirection_init_resize_bilinear2d_chw_f32(
  size_t input_pixel_stride,
  size_t input_height,
  size_t input_width,
  size_t output_height,
  size_t output_width,
  const void* input,
  const void** indirection_buffer,
  float* packed_weights,
  bool align_corners,
  bool tensorflow_legacy);

XNN_INTERNAL void xnn_indirection_init_unpool2d(
  xnn_operator_t op,
  size_t batch_start,
  uint32_t log2_element_size);

typedef void (*xnn_indirection_init_pavgpool2d_fn)(
  size_t input_height,
  size_t input_width,
  size_t output_height,
  size_t output_width,
  size_t pooling_height,
  size_t pooling_width,
  size_t stride_height,
  size_t stride_width,
  size_t padding_top,
  size_t padding_left,
  void* pixelwise_buffer);

XNN_INTERNAL void xnn_indirection_init_pavgpool2d_f16(
  size_t input_height,
  size_t input_width,
  size_t output_height,
  size_t output_width,
  size_t pooling_height,
  size_t pooling_width,
  size_t stride_height,
  size_t stride_width,
  size_t padding_top,
  size_t padding_left,
  uint16_t* pixelwise_buffer);

XNN_INTERNAL void xnn_indirection_init_pavgpool2d_f32(
  size_t input_height,
  size_t input_width,
  size_t output_height,
  size_t output_width,
  size_t pooling_height,
  size_t pooling_width,
  size_t stride_height,
  size_t stride_width,
  size_t padding_top,
  size_t padding_left,
  float* pixelwise_buffer);

#ifdef __cplusplus
}  // extern "C"
#endif
