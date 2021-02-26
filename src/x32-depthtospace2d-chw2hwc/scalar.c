// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <xnnpack/depthtospace.h>


void xnn_x32_depthtospace2d_chw2hwc_ukernel__scalar(
    size_t output_channels,
    size_t input_height,
    size_t input_width,
    size_t block_size,
    const uint32_t*restrict input,
    uint32_t*restrict output,
    size_t output_channel_stride)
{
  assert(output_channels != 0);
  assert(input_height != 0);
  assert(input_width != 0);
  assert(block_size != 0);

  for (size_t iy = 0; iy < input_height; iy++) {
    for (size_t by = 0; by < block_size; by++) {
      for (size_t ix = 0; ix < input_width; ix++) {
        for (size_t bx = 0; bx < block_size; bx++) {
          for (size_t oc = 0; oc < output_channels; oc++) {
            output[(((iy * block_size + by) * input_width + ix) * block_size + bx) * output_channel_stride + oc] =
              input[(((by * block_size + bx) * output_channels + oc) * input_height + iy) * input_width + ix];
          }
        }
      }
    }
  }
}
