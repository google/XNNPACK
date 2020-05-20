// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include <xnnpack/fill.h>


void xnn_x32_fill_ukernel__neon(
    size_t rows,
    size_t channels,
    uint32_t* output,
    size_t output_stride,
    const uint32_t* fill_value)
{
  assert(rows != 0);
  assert(channels != 0);
  assert(channels % sizeof(uint32_t) == 0);
  assert(fill_value != NULL);

  const size_t output_increment = output_stride - channels;

  const uint32x4_t vfill = vld1q_dup_u32(fill_value);
  do {
    size_t c = channels;
    for (; c >= 16 * sizeof(uint32_t); c -= 16 * sizeof(uint32_t)) {
      vst1q_u32(output, vfill); output += 4;
      vst1q_u32(output, vfill); output += 4;
      vst1q_u32(output, vfill); output += 4;
      vst1q_u32(output, vfill); output += 4;
    }
    for (; c >= 4 * sizeof(uint32_t); c -= 4 * sizeof(uint32_t)) {
      vst1q_u32(output, vfill); output += 4;
    }
    if XNN_UNLIKELY(c != 0) {
      if XNN_LIKELY(c & (2 * sizeof(uint32_t))) {
        vst1_u32(output, vget_low_u32(vfill)); output += 2;
      }
      if XNN_LIKELY(c & (1 * sizeof(uint32_t))) {
        vst1q_lane_u32(output, vfill, 0); output += 1;
      }
    }
    output = (void*) ((uintptr_t) output + output_increment);
  } while (--rows != 0);
}
