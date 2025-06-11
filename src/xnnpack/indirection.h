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

#include "include/xnnpack.h"
#include "src/xnnpack/common.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microparams.h"

#ifdef __cplusplus
extern "C" {
#endif

XNN_INTERNAL void xnn_indirection_init_conv2d(
    size_t output_tile_size, size_t output_start, size_t output_end,
    const void** indirection_buffer, const void* input, const void* zero_buffer,
    size_t input_pixel_stride, size_t input_height, size_t input_width,
    size_t output_height, size_t output_width, size_t kernel_height,
    size_t kernel_width, size_t stride_height, size_t stride_width,
    size_t dilation_height, size_t dilation_width, size_t input_padding_top,
    size_t input_padding_left);

// Initialize compressed indirection buffers.
// Original indirection buffers has a row of buffer for each row of input.
// Compressed indirection buffers compress rows of input pointers that point to
// valid elements in the input (not padding). In this section of the indirection
// buffer, all input pointers in row n+1 are a constant K offset away from the
// input pointers in row n.
//
// Compressed indirection buffers are made up of 3 section:
// - Top
// - Middle
// - Bottom
//
// Top has as many rows of input pointers as padding_top / stride_height
// (rounded up): indirect_top_height. Middle is 1 row of input pointer (this is
// where the compression is). Bottom has as many rows of input pointers as
// padding_bottom / stride_height (rounded up): indirect_bot_height. (Note:
// padding left and right does not affect compression.)
XNN_INTERNAL void xnn_indirection_init_dwconv2d_compressed(
    size_t output_y_start, size_t output_y_end, const void** indirection_buffer,
    const void* input, size_t input_pixel_stride, const void* zero_buffer,
    size_t input_height, size_t input_width, size_t output_height,
    size_t output_width, size_t kernel_height, size_t kernel_width,
    size_t stride_height, size_t stride_width, size_t dilation_height,
    size_t dilation_width, size_t input_padding_top, size_t input_padding_left,
    size_t step_height, size_t step_width, size_t indirect_top_height,
    size_t indirect_bot_height, size_t primary_tile);

XNN_INTERNAL void xnn_indirection_init_dwconv2d(
    size_t output_y_start, size_t output_y_end, const void** indirection_buffer,
    const void* input, size_t input_pixel_stride, const void* zero_buffer,
    size_t input_height, size_t input_width, size_t output_height,
    size_t output_width, size_t kernel_height, size_t kernel_width,
    size_t stride_height, size_t stride_width, size_t dilation_height,
    size_t dilation_width, size_t input_padding_top, size_t input_padding_left,
    size_t step_height, size_t step_width, size_t primary_tile);

XNN_INTERNAL void xnn_indirection_init_deconv2d(
    size_t output_tile_size, const void** indirection_buffer, const void* input,
    size_t input_pixel_stride, const void* zero, size_t input_height,
    size_t input_width, size_t output_height, size_t output_width,
    size_t kernel_height, size_t kernel_width, size_t stride_height,
    size_t stride_width, size_t dilation_height, size_t dilation_width,
    size_t padding_top, size_t padding_left);

XNN_INTERNAL void xnn_indirection_init_subconv2d(
    size_t output_tile_size, const void** indirection_buffer,
    struct subconvolution_params* subconvolution_params, const void* input,
    size_t input_pixel_stride, const void* zero, size_t input_height,
    size_t input_width, size_t output_height, size_t output_width,
    size_t kernel_height, size_t kernel_width, size_t stride_height,
    size_t stride_width, size_t padding_top, size_t padding_left);

XNN_INTERNAL void xnn_indirection_init_maxpool2d(
    const void** indirection_buffer, const void* input,
    const size_t input_pixel_stride, const size_t input_height,
    const size_t input_width, const size_t output_height,
    const size_t output_width, const size_t kernel_height,
    const size_t kernel_width, const size_t stride_height,
    const size_t stride_width, const size_t dilation_height,
    const size_t dilation_width, const size_t input_padding_top,
    const size_t input_padding_left, const size_t step_height,
    const size_t step_width);

XNN_INTERNAL void xnn_indirection_init_resize_bilinear2d_hwc_f16(
    size_t output_y_start, size_t output_y_end, size_t input_pixel_stride,
    size_t input_height, size_t input_width, size_t output_height,
    size_t output_width, const void* input, const void** indirection_buffer,
    xnn_float16* packed_weights, bool align_corners, bool tensorflow_legacy);

XNN_INTERNAL void xnn_indirection_init_resize_bilinear2d_hwc_f32(
    size_t output_y_start, size_t output_y_end, size_t input_pixel_stride,
    size_t input_height, size_t input_width, size_t output_height,
    size_t output_width, const void* input, const void** indirection_buffer,
    float* packed_weights, bool align_corners, bool tensorflow_legacy);

XNN_INTERNAL void xnn_indirection_init_resize_bilinear2d_hwc_q11(
    size_t output_y_start, size_t output_y_end, size_t input_pixel_stride,
    size_t input_height, size_t input_width, size_t output_height,
    size_t output_width, const void* input, const void** indirection_buffer,
    int16_t* packed_weights, bool align_corners, bool tensorflow_legacy);

XNN_INTERNAL void xnn_indirection_init_resize_bilinear2d_chw_f16(
    size_t input_pixel_stride, size_t input_height, size_t input_width,
    size_t output_height, size_t output_width, const void* input,
    const void** indirection_buffer, xnn_float16* packed_weights,
    bool align_corners, bool tensorflow_legacy);

XNN_INTERNAL void xnn_indirection_init_resize_bilinear2d_chw_f32(
    size_t input_pixel_stride, size_t input_height, size_t input_width,
    size_t output_height, size_t output_width, const void* input,
    const void** indirection_buffer, float* packed_weights, bool align_corners,
    bool tensorflow_legacy);

XNN_INTERNAL void xnn_indirection_init_unpool2d(
    const void** indirection_buffer, const void* output,
    const size_t output_pixel_stride, const size_t batch_size,
    const size_t input_height, const size_t input_width,
    const size_t output_height, const size_t output_width,
    const size_t kernel_height, const size_t kernel_width,
    const size_t output_padding_top, const size_t output_padding_left,
    size_t batch_start);

typedef void (*xnn_indirection_init_pavgpool2d_fn)(
    size_t input_height, size_t input_width, size_t output_height,
    size_t output_width, size_t pooling_height, size_t pooling_width,
    size_t stride_height, size_t stride_width, size_t padding_top,
    size_t padding_left, void* pixelwise_buffer);

XNN_INTERNAL void xnn_indirection_init_pavgpool2d_f16(
    size_t input_height, size_t input_width, size_t output_height,
    size_t output_width, size_t pooling_height, size_t pooling_width,
    size_t stride_height, size_t stride_width, size_t padding_top,
    size_t padding_left, xnn_float16* pixelwise_buffer);

XNN_INTERNAL void xnn_indirection_init_pavgpool2d_f32(
    size_t input_height, size_t input_width, size_t output_height,
    size_t output_width, size_t pooling_height, size_t pooling_width,
    size_t stride_height, size_t stride_width, size_t padding_top,
    size_t padding_left, float* pixelwise_buffer);

#ifdef __cplusplus
}  // extern "C"
#endif
