// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include <xnnpack/clamp.h>


void xnn_u8_clamp_ukernel__neon_x64(
    size_t n,
    const uint8_t* x,
    uint8_t* y,
    const union xnn_u8_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(n != 0);

  const uint8x16_t voutput_max = vld1q_dup_u8(&params->neon.max);
  const uint8x16_t voutput_min = vld1q_dup_u8(&params->neon.min);

  for (; n >= 64; n -= 64) {
    const uint8x16_t vx0 = vld1q_u8(x); x += 16;
    const uint8x16_t vx1 = vld1q_u8(x); x += 16;
    const uint8x16_t vx2 = vld1q_u8(x); x += 16;
    const uint8x16_t vx3 = vld1q_u8(x); x += 16;

    const uint8x16_t vy0 = vminq_u8(vmaxq_u8(vx0, voutput_min), voutput_max);
    const uint8x16_t vy1 = vminq_u8(vmaxq_u8(vx1, voutput_min), voutput_max);
    const uint8x16_t vy2 = vminq_u8(vmaxq_u8(vx2, voutput_min), voutput_max);
    const uint8x16_t vy3 = vminq_u8(vmaxq_u8(vx3, voutput_min), voutput_max);

    vst1q_u8(y, vy0); y += 16;
    vst1q_u8(y, vy1); y += 16;
    vst1q_u8(y, vy2); y += 16;
    vst1q_u8(y, vy3); y += 16;
  }
  for (; n >= 8; n -= 8) {
    uint8x8_t vout = vld1_u8(x); x += 8;
    vout = vmin_u8(vout, vget_low_u8(voutput_max));
    vout = vmax_u8(vout, vget_low_u8(voutput_min));
    vst1_u8(y, vout); y += 8;
  }
  if XNN_UNLIKELY(n != 0) {
    uint8x8_t vout = vld1_u8(x);
    vout = vmin_u8(vout, vget_low_u8(voutput_max));
    vout = vmax_u8(vout, vget_low_u8(voutput_min));

    if (n & 4) {
      vst1_lane_u32(__builtin_assume_aligned(y, 1), vreinterpret_u32_u8(vout), 0); y += 4;
      vout = vext_u8(vout, vout, 4);
    }
    if (n & 2) {
      vst1_lane_u16(__builtin_assume_aligned(y, 1), vreinterpret_u16_u8(vout), 0); y += 2;
      vout = vext_u8(vout, vout, 2);
    }
    if (n & 1) {
      vst1_lane_u8(y, vout, 0);
    }
  }
}
