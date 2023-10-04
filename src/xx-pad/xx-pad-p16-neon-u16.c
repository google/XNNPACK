// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include <xnnpack/pad.h>


void xnn_xx_pad_ukernel_p16__neon_u16(
    size_t rows,
    size_t channels,
    size_t pre_padding,
    size_t post_padding,
    const void* input,
    size_t input_stride,
    void* output,
    size_t output_stride,
    uint32_t fill_pattern) XNN_OOB_READS
{
  const size_t input_increment = input_stride - channels;
  const size_t output_increment = output_stride - (pre_padding + channels + post_padding);

  const uint8x16_t vfill_pattern = vreinterpretq_u8_u32(vdupq_n_u32(fill_pattern));
  do {
    // Pre-pad input channels.
    size_t l = pre_padding;
    if XNN_LIKELY(l != 0) {
      for (; l >= 16 * sizeof(uint8_t); l -= 16 * sizeof(uint8_t)) {
        vst1q_u8(output, vfill_pattern); output = (uint8_t*) output + 16;
      }
      if (l & (8 * sizeof(uint8_t))) {
        vst1_u8(output, vget_low_u8(vfill_pattern)); output = (uint8_t*) output + 8;
      }
      if (l & (4 * sizeof(uint8_t))) {
        vst1q_lane_u32(output, vreinterpretq_u32_u8(vfill_pattern), 0); output = (uint8_t*) output + 4;
      }
      uint8x8_t vfill_subpattern = vget_low_u8(vfill_pattern);
      if (l & (2 * sizeof(uint8_t))) {
        vst1_lane_u16(output, vreinterpret_u16_u8(vfill_subpattern), 0); output = (uint8_t*) output + 2;
        vfill_subpattern = vext_u8(vfill_subpattern, vfill_subpattern, 2);
      }
      if (l & (1 * sizeof(uint8_t))) {
        vst1_lane_u8(output, vfill_subpattern, 0); output = (uint8_t*) output + 1;
      }
    }

    // Copy input channels.
    size_t c = channels;
    for (; c >= 16 * sizeof(uint8_t); c -= 16 * sizeof(uint8_t)) {
      const uint8x16_t vdata = vld1q_u8(input); input = (const uint8_t*) input + 16;
      vst1q_u8(output, vdata); output = (uint8_t*) output + 16;
    }
    if XNN_UNLIKELY(c != 0) {
      uint8x16_t vdata = vld1q_u8(input); input = (const void*) ((uintptr_t) input + c);

      uint8x8_t vsubdata = vget_low_u8(vdata);
      if (c & (8 * sizeof(uint8_t))) {
        vst1_u8(output, vsubdata); output = (uint8_t*) output + 8;
        vsubdata = vget_high_u8(vdata);
      }
      if (c & (4 * sizeof(uint8_t))) {
        vst1_lane_u32(output, vreinterpret_u32_u8(vsubdata), 0); output = (uint8_t*) output + 4;
        vsubdata = vext_u8(vsubdata, vsubdata, 4);
      }
      if (c & (2 * sizeof(uint8_t))) {
        vst1_lane_u16(output, vreinterpret_u16_u8(vsubdata), 0); output = (uint8_t*) output + 2;
        vsubdata = vext_u8(vsubdata, vsubdata, 2);
      }
      if (c & (1 * sizeof(uint8_t))) {
        vst1_lane_u8(output, vsubdata, 0); output = (uint8_t*) output + 1;
      }
    }

    // Post-pad input channels.
    size_t r = post_padding;
    if XNN_LIKELY(r != 0) {
      for (; r >= 16 * sizeof(uint8_t); r -= 16 * sizeof(uint8_t)) {
        vst1q_u8(output, vfill_pattern); output = (uint8_t*) output + 16;
      }
      if (r & (8 * sizeof(uint8_t))) {
        vst1_u8(output, vget_low_u8(vfill_pattern)); output = (uint8_t*) output + 8;
      }
      if (r & (4 * sizeof(uint8_t))) {
        vst1q_lane_u32(output, vreinterpretq_u32_u8(vfill_pattern), 0); output = (uint8_t*) output + 4;
      }
      uint8x8_t vfill_subpattern = vget_low_u8(vfill_pattern);
      if (r & (2 * sizeof(uint8_t))) {
        vst1_lane_u16(output, vreinterpret_u16_u8(vfill_subpattern), 0); output = (uint8_t*) output + 2;
        vfill_subpattern = vext_u8(vfill_subpattern, vfill_subpattern, 2);
      }
      if (r & (1 * sizeof(uint8_t))) {
        vst1_lane_u8(output, vfill_subpattern, 0); output = (uint8_t*) output + 1;
      }
    }

    input = (const uint32_t*) ((uintptr_t) input + input_increment);
    output = (uint32_t*) ((uintptr_t) output + output_increment);
  } while (--rows != 0);
}
