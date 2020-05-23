// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <xnnpack/pad.h>


void xnn_x32_pad_ukernel__scalar_float(
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

  const float vfill = *((const float*) fill_value);
  const float* i = (const float*) input;
  float* o = (float*) output;
  do {
    // Pre-pad input channels.
    size_t l = pre_padding;
    if XNN_LIKELY(l != 0) {
      do {
        *o++ = vfill;
        l -= sizeof(uint32_t);
      } while (l != 0);
    }

    // Copy input channels.
    size_t c = channels;
    for (; c >= 4 * sizeof(uint32_t); c -= 4 * sizeof(uint32_t)) {
      o[0] = i[0];
      o[1] = i[1];
      o[2] = i[2];
      o[3] = i[3];
      i += 4;
      o += 4;
    }
    if XNN_UNLIKELY(c != 0) {
      do {
        *o++ = *i++;
        c -= sizeof(uint32_t);
      } while (c != 0);
    }

    // Post-pad input channels.
    size_t r = post_padding;
    if XNN_LIKELY(r != 0) {
      do {
        *o++ = vfill;
        r -= sizeof(uint32_t);
      } while (r != 0);
    }

    i = (const float*) ((uintptr_t) i + input_increment);
    o = (float*) ((uintptr_t) o + output_increment);
  } while (--rows != 0);
}
