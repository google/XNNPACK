// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include "src/xnnpack/unpool.h"


void xnn_x32_unpool_ukernel__neon(
    size_t kernel_elements,
    size_t channels,
    uint32_t fill,
    const uint32_t* input,
    const uint32_t* index,
    uint32_t** output)
{
  const size_t num_outputs = kernel_elements;
  // Pre-initialize outputs with constant.
  const uint32x4_t vfill = vdupq_n_u32(fill);
  uint32_t** os = output;
  do {
    uint32_t* o = *os++;
    size_t c = channels;
    for (; c >= 4; c -= 4) {
      vst1q_u32(o, vfill); o += 4;
    }
    if (c != 0) {
      if (c & 2) {
        vst1_u32(o, vget_low_u32(vfill)); o += 2;
      }
      if (c & 1) {
        vst1q_lane_u32(o, vfill, 0);
      }
    }
  } while (--kernel_elements != 0);

  // Copy indexed elements to output.
  size_t offset = 0;
  do {
    const uint32_t i = *index++;
    const uint32_t v = *input++;
    // Ignore indices outside the kernel_elements output pointers instead of
    // dereferencing output[i] out of bounds. Valid unpooling indices are
    // always < kernel_elements, so well-formed input is unaffected.
    if (i < num_outputs) {
      *((uint32_t*) ((uintptr_t) output[i] + offset)) = v;
    }
    offset += sizeof(uint32_t);
  } while (--channels != 0);
}
