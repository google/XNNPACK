// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <stddef.h>

#include "xnnpack/common.h"

#ifdef __cplusplus
extern "C" {
#endif

XNN_INTERNAL void xnn_im2col_conv2d(
  size_t output_height,
  size_t output_width,
  size_t kernel_height,
  size_t kernel_width,
  size_t subsampling_height,
  size_t subsampling_width,
  size_t dilation_height,
  size_t dilation_width,
  size_t input_width,
  size_t input_padding_top,
  size_t input_padding_left,
  size_t group_input_channels_in_bytes,
  size_t input_pixel_stride_in_bytes,
  const void* input,
  void* output);

#ifdef __cplusplus
}  // extern "C"
#endif
