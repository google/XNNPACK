// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include <xnnpack/im2col.h>


void xnn_im2col_conv2d(
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
  void* output)
{
  for (size_t output_y = 0; output_y < output_height; output_y++) {
    for (size_t output_x = 0; output_x < output_width; output_x++) {
      for (size_t kernel_y = 0; kernel_y < kernel_height; kernel_y++) {
        const size_t input_y = output_y * subsampling_height + kernel_y * dilation_height - input_padding_top;
        if (input_y < output_height) {
          for (size_t kernel_x = 0; kernel_x < kernel_width; kernel_x++) {
            const size_t input_x = output_x * subsampling_width + kernel_x * dilation_width - input_padding_left;
            if (input_x < output_width) {
              memcpy(output,
                (const void*) ((uintptr_t) input + (input_y * input_width + input_x) * input_pixel_stride_in_bytes),
                group_input_channels_in_bytes);
            } else {
              memset(output, 0, group_input_channels_in_bytes);
            }
            output = (void*) ((uintptr_t) output + group_input_channels_in_bytes);
          }
        } else {
          memset(output, 0, kernel_width * group_input_channels_in_bytes);
          output = (void*) ((uintptr_t) output + kernel_width * group_input_channels_in_bytes);
        }
      }
    }
  }
}
