// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include "xnnpack/fill.h"


void xnn_xx_fill_ukernel__neon_u64(
    size_t rows,
    size_t channels,
    void* output,
    size_t output_stride,
    const uint32_t fill_pattern)
{
  assert(rows != 0);
  assert(channels != 0);

  const size_t output_increment = output_stride - channels;

  const uint8x16_t vfill_pattern = vreinterpretq_u8_u32(vdupq_n_u32(fill_pattern));
  do {
    size_t c = channels;
    for (; c >= 64 * sizeof(uint8_t); c -= 64 * sizeof(uint8_t)) {
      vst1q_u8(output, vfill_pattern); output = ((uint8_t*) output + 16);
      vst1q_u8(output, vfill_pattern); output = ((uint8_t*) output + 16);
      vst1q_u8(output, vfill_pattern); output = ((uint8_t*) output + 16);
      vst1q_u8(output, vfill_pattern); output = ((uint8_t*) output + 16);
    }
    for (; c >= 16 * sizeof(uint8_t); c -= 16 * sizeof(uint8_t)) {
      vst1q_u8(output, vfill_pattern); output = ((uint8_t*) output + 16);
    }
    if XNN_UNLIKELY(c != 0) {
      if XNN_LIKELY(c & (8 * sizeof(uint8_t))) {
        vst1_u8(output, vget_low_u8(vfill_pattern)); output = ((uint8_t*) output + 8);
      }
      if XNN_LIKELY(c & (4 * sizeof(uint8_t))) {
        vst1q_lane_u32(output, vreinterpretq_u32_u8(vfill_pattern), 0); output = ((uint8_t*) output + 4);
      }
      uint8x8_t vfill_subpattern = vget_low_u8(vfill_pattern);
      if XNN_LIKELY(c & (2 * sizeof(uint8_t))) {
        vst1_lane_u16(output, vreinterpret_u16_u8(vfill_subpattern), 0); output = ((uint8_t*) output + 2);
        vfill_subpattern = vext_u8(vfill_subpattern, vfill_subpattern, 2);
      }
      if XNN_LIKELY(c & (1 * sizeof(uint8_t))) {
        vst1_lane_u8(output, vfill_subpattern, 0); output = ((uint8_t*) output + 1);
      }
    }
    output = (void*) ((uintptr_t) output + output_increment);
  } while (--rows != 0);
}
