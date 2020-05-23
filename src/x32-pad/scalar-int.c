// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <xnnpack/pad.h>


void xnn_x32_pad_ukernel__scalar_int(
    size_t rows,
    size_t channels,
    size_t pre_padding,
    size_t post_padding,
    const uint32_t* fill_value,
    const uint32_t* input,
    size_t input_stride,
    uint32_t* output,
    size_t output_stride)
{
  assert(channels % sizeof(uint32_t) == 0);
  assert(pre_padding % sizeof(uint32_t) == 0);
  assert(post_padding % sizeof(uint32_t) == 0);
  assert(fill_value != NULL);

  const size_t input_increment = input_stride - channels;
  const size_t output_increment = output_stride - (pre_padding + channels + post_padding);

  const uint32_t vfill = *fill_value;
  do {
    // Pre-pad input channels.
    size_t l = pre_padding;
    if XNN_LIKELY(l != 0) {
      do {
        *output++ = vfill;
        l -= sizeof(uint32_t);
      } while (l != 0);
    }

    // Copy input channels.
    size_t c = channels;
    for (; c >= 4 * sizeof(uint32_t); c -= 4 * sizeof(uint32_t)) {
      output[0] = input[0];
      output[1] = input[1];
      output[2] = input[2];
      output[3] = input[3];
      input += 4;
      output += 4;
    }
    if XNN_UNLIKELY(c != 0) {
      do {
        *output++ = *input++;
        c -= sizeof(uint32_t);
      } while (c != 0);
    }

    // Post-pad input channels.
    size_t r = post_padding;
    if XNN_LIKELY(r != 0) {
      do {
        *output++ = vfill;
        r -= sizeof(uint32_t);
      } while (r != 0);
    }

    input = (const uint32_t*) ((uintptr_t) input + input_increment);
    output = (uint32_t*) ((uintptr_t) output + output_increment);
  } while (--rows != 0);
}
