// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include <xnnpack/pad.h>


void xnn_x32_unpool_ukernel__neon(
    size_t p,
    size_t c,
    uint32_t f,
    const uint32_t* input,
    const uint32_t* index,
    uint32_t** output)
{
  // Pre-initialize outputs with constant.
  const uint32x4_t vf = vdupq_n_u32(f);
  uint32_t** os = output;
  do {
    uint32_t* o = *os++;
    size_t k = c;
    for (; k >= 4; k -= 4) {
      vst1q_u32(o, vf); o += 4;
    }
    if (k != 0) {
      if (k & 2) {
        vst1_u32(o, vget_low_u32(vf)); o += 2;
      }
      if (k & 1) {
        vst1q_lane_u32(o, vf, 0);
      }
    }
  } while (--p != 0);

  // Copy indexed elements to output.
  size_t offset = 0;
  do {
    const uint32_t i = *index++;
    *((uint32_t*) ((uintptr_t) output[i] + offset)) = *input++;
    offset += sizeof(uint32_t);
  } while (--c != 0);
}
