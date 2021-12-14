// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <xnnpack/pad.h>


void xnn_xx_pad_ukernel__scalar(
    size_t rows,
    size_t channels,
    size_t pre_padding,
    size_t post_padding,
    const void* input,
    size_t input_stride,
    void* output,
    size_t output_stride,
    const uint32_t fill_pattern) XNN_OOB_READS
{
  const size_t input_increment = input_stride - channels;
  const size_t output_increment = output_stride - (pre_padding + channels + post_padding);

  do {
    // Pre-pad input channels.
    size_t l = pre_padding;
    if XNN_LIKELY(l != 0) {
      uint32_t vfill_pattern = fill_pattern;
      for (; l >= 4 * sizeof(uint8_t); l -= 4 * sizeof(uint8_t)) {
        *((uint32_t*) output) = vfill_pattern;
        output = (uint8_t*) output + 4;
      }
      if XNN_LIKELY(l & (2 * sizeof(uint8_t))) {
        *((uint16_t*) output) = (uint16_t) vfill_pattern;
        vfill_pattern >>= 16;
        output = (uint8_t*) output + 2;
      }
      if XNN_LIKELY(l & (1 * sizeof(uint8_t))) {
        *((uint8_t*) output) = (uint8_t) vfill_pattern;
        output = (uint8_t*) output + 1;
      }
    }

    // Copy input channels.
    size_t c = channels;
    for (; c >= 16 * sizeof(uint8_t); c -= 16 * sizeof(uint8_t)) {
      const uint32_t vdata0 = ((const uint32_t*) input)[0];
      const uint32_t vdata1 = ((const uint32_t*) input)[1];
      const uint32_t vdata2 = ((const uint32_t*) input)[2];
      const uint32_t vdata3 = ((const uint32_t*) input)[3];
      input = (const uint8_t*) input + 16;

      ((uint32_t*) output)[0] = vdata0;
      ((uint32_t*) output)[1] = vdata1;
      ((uint32_t*) output)[2] = vdata2;
      ((uint32_t*) output)[3] = vdata3;
      output = (uint8_t*) output + 16;
    }
    if XNN_UNLIKELY(c != 0) {
      for (; c >= 4 * sizeof(uint8_t); c -= 4 * sizeof(uint8_t)) {
        *((uint32_t*) output) = *((const uint32_t*) input);
        input = (const uint8_t*) input + 4;
        output = (uint8_t*) output + 4;
      }
      if XNN_UNLIKELY(c != 0) {
        uint32_t vdata = *((const uint32_t*) input);
        input = (const void*) ((uintptr_t) input + c);

        if XNN_LIKELY(c & (2 * sizeof(uint8_t))) {
          *((uint16_t*) output) = (uint16_t) vdata;
          vdata >>= 16;
          output = (uint8_t*) output + 2;
        }
        if XNN_LIKELY(c & (1 * sizeof(uint8_t))) {
          *((uint8_t*) output) = (uint8_t) vdata;
          output = (uint8_t*) output + 1;
        }
      }
    }

    // Post-pad input channels.
    size_t r = post_padding;
    if XNN_LIKELY(r != 0) {
      uint32_t vfill_pattern = fill_pattern;
      for (; r >= 4 * sizeof(uint8_t); r -= 4 * sizeof(uint8_t)) {
        *((uint32_t*) output) = vfill_pattern;
        output = (uint8_t*) output + 4;
      }
      if XNN_LIKELY(r & (2 * sizeof(uint8_t))) {
        *((uint16_t*) output) = (uint16_t) vfill_pattern;
        vfill_pattern >>= 16;
        output = (uint8_t*) output + 2;
      }
      if XNN_LIKELY(r & (1 * sizeof(uint8_t))) {
        *((uint8_t*) output) = (uint8_t) vfill_pattern;
        output = (uint8_t*) output + 1;
      }
    }

    input = (const uint32_t*) ((uintptr_t) input + input_increment);
    output = (uint32_t*) ((uintptr_t) output + output_increment);
  } while (--rows != 0);
}
