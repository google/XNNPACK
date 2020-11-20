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
    size_t input_channel_stride,
    size_t input_height_stride,
    size_t output_height_stride,
    size_t output_width_stride)
{
  assert(output_channels != 0);
  assert(input_height != 0);
  assert(input_width != 0);
  assert(block_size != 0);

  // output[(iy * block_size + by) * output_height_stride +
  //        (ix * block_size + bx) * output_width_stride +
  //        c * element_stride] =
  //     input[
  //         (c * block_size * block_size + by * block_size + bx) * input_channel_stride +
  //         iy * input_height_stride +
  //         ix * element_stride]

  const size_t element_stride = sizeof(uint32_t);

  const size_t iy_output_increment = block_size * output_height_stride;
  const size_t by_output_increment = output_height_stride;
  const size_t ix_output_increment = block_size * output_width_stride;
  const size_t bx_output_increment = output_width_stride;
  const size_t c_output_increment = element_stride;

  const size_t c_input_increment = block_size * block_size * input_channel_stride;
  const size_t by_input_increment = block_size * input_channel_stride;
  const size_t bx_input_increment = input_channel_stride;
  const size_t iy_input_increment = input_height_stride;
  const size_t ix_input_increment = element_stride;

  size_t iy = input_height;
  uintptr_t i_iy = (uintptr_t) input;
  uintptr_t o_iy = (uintptr_t) output;
  do {
    size_t by = block_size;
    uintptr_t i_by = i_iy;
    uintptr_t o_by = o_iy;
    do {
      size_t ix = input_width;
      uintptr_t i_ix = i_by;
      uintptr_t o_ix = o_by;
      do {
        size_t bx = block_size;
        uintptr_t i_bx = i_ix;
        uintptr_t o_bx = o_ix;
        do {
          size_t c = output_channels;
          uintptr_t i_c = i_bx;
          uintptr_t o_c = o_bx;
          do {
            *(uint32_t*) o_c = *(uint32_t*) i_c;
            i_c += c_input_increment;
            o_c += c_output_increment;
          } while (--c != 0);
          i_bx += bx_input_increment;
          o_bx += bx_output_increment;
        } while (--bx != 0);
        i_ix += ix_input_increment;
        o_ix += ix_output_increment;
      } while (--ix != 0);
      i_by += by_input_increment;
      o_by += by_output_increment;
    } while (--by != 0);
    i_iy += iy_input_increment;
    o_iy += iy_output_increment;
  } while (--iy != 0);
}
