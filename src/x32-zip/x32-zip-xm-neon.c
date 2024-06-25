// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include "xnnpack/zip.h"


void xnn_x32_zip_xm_ukernel__neon(
    size_t n,
    size_t m,
    const uint32_t* input,
    uint32_t* output)
{
  assert(n != 0);
  assert(n % 4 == 0);
  assert(m >= 4);

  const uint32_t* w = input;
  const size_t group_increment = m * 4;
  const size_t input_increment = n * 3;
  const size_t output_increment = 16 - m * n;
  const uint32_t* last_input = (const uint32_t*) ((uintptr_t) input + n * (m - 1));
  uint32_t* last_output = (uint32_t*) ((uintptr_t) output + (m * 4 - 16));

  for (size_t i = 0; i < m; i += 4) {
    w = (const uint32_t*) ((uintptr_t) w + input_increment);
    if (w >= last_input) {
      w = last_input;
    }
    const uint32_t* z = (const uint32_t*) ((uintptr_t) w - n);
    const uint32_t* y = (const uint32_t*) ((uintptr_t) z - n);
    const uint32_t* x = (const uint32_t*) ((uintptr_t) y - n);

    size_t k = n;
    while (k >= 16) {
      const uint32x4_t vx = vld1q_u32(x); x += 4;
      const uint32x4_t vy = vld1q_u32(y); y += 4;
      const uint32x4_t vz = vld1q_u32(z); z += 4;
      const uint32x4_t vw = vld1q_u32(w); w += 4;

      const uint32x4x2_t vxy = vzipq_u32(vx, vy);
      const uint32x4x2_t vzw = vzipq_u32(vz, vw);

      vst1_u32(output, vget_low_u32(vxy.val[0]));
      vst1_u32(output + 2, vget_low_u32(vzw.val[0]));
      output = (uint32_t*) ((uintptr_t) output + group_increment);

      vst1_u32(output, vget_high_u32(vxy.val[0]));
      vst1_u32(output + 2, vget_high_u32(vzw.val[0]));
      output = (uint32_t*) ((uintptr_t) output + group_increment);

      vst1_u32(output, vget_low_u32(vxy.val[1]));
      vst1_u32(output + 2, vget_low_u32(vzw.val[1]));
      output = (uint32_t*) ((uintptr_t) output + group_increment);

      vst1_u32(output, vget_high_u32(vxy.val[1]));
      vst1_u32(output + 2, vget_high_u32(vzw.val[1]));
      output = (uint32_t*) ((uintptr_t) output + group_increment);

      k -= 16;
    }
    if XNN_UNLIKELY(k != 0) {
      if (k & 8) {
        const uint32x2_t vx = vld1_u32(x); x += 2;
        const uint32x2_t vy = vld1_u32(y); y += 2;
        const uint32x2_t vz = vld1_u32(z); z += 2;
        const uint32x2_t vw = vld1_u32(w); w += 2;

        const uint32x2x2_t vxy = vzip_u32(vx, vy);
        const uint32x2x2_t vzw = vzip_u32(vz, vw);

        vst1_u32(output, vxy.val[0]);
        vst1_u32(output + 2, vzw.val[0]);
        output = (uint32_t*) ((uintptr_t) output + group_increment);

        vst1_u32(output, vxy.val[1]);
        vst1_u32(output + 2, vzw.val[1]);
        output = (uint32_t*) ((uintptr_t) output + group_increment);
      }
      if (k & 4) {
        const uint32x2_t vx = vld1_dup_u32(x);
        const uint32x2_t vz = vld1_dup_u32(z);
        const uint32x2_t vxy = vld1_lane_u32(y, vx, 1);
        const uint32x2_t vzw = vld1_lane_u32(w, vz, 1); w += 1;

        vst1_u32(output, vxy);
        vst1_u32(output + 2, vzw);
        output = (uint32_t*) ((uintptr_t) output + group_increment);
      }
    }
    output = (uint32_t*) ((uintptr_t) output + output_increment);
    if (output > last_output) {
      output = last_output;
    }
  }
}
