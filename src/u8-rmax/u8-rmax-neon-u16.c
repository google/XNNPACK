// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include <xnnpack/reduce.h>


void xnn_u8_rmax_ukernel__neon_u16(
    size_t batch,
    const uint8_t* input,
    uint8_t* output,
    const void* params)
{
  assert(batch != 0);
  assert(batch % sizeof(uint8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  if XNN_LIKELY(batch >= 16) {
    uint8x16_t vmax = vmovq_n_u8(0);
    do {
      const uint8x16_t vx = vld1q_u8(input); input += 16;
      vmax = vmaxq_u8(vmax, vx);
      batch -= 16;
    } while (batch >= 16);
    if (batch != 0) {
      const size_t x_increment = batch - 16;
      input = (const uint8_t*) ((uintptr_t) input + x_increment);
      const uint8x16_t vx = vld1q_u8(input);
      vmax = vmaxq_u8(vmax, vx);
    }
    uint8x8_t vmax8 = vmax_u8(vget_low_u8(vmax), vget_high_u8(vmax));
    const uint8x8_t vmax4 = vpmax_u8(vmax8, vmax8);
    const uint8x8_t vmax2 = vpmax_u8(vmax4, vmax4);
    const uint8x8_t vmax1 = vpmax_u8(vmax2, vmax2);
    vst1_lane_u8(output, vmax1, 0);
  } else {
    uint8x8_t vmax = vmov_n_u8(0);
    do {
      const uint8x8_t vx = vld1_dup_u8(input); input += 1;
      vmax = vmax_u8(vmax, vx);
    } while (--batch != 0);
    vst1_lane_u8(output, vmax, 0);
  }
}
