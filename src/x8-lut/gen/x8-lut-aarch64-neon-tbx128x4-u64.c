// Auto-generated file. Do not edit!
//   Template: src/x8-lut/neon-tbx128x4.c.in
//   Generator: tools/xngen
//
// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include "xnnpack/intrinsics-polyfill.h"
#include "xnnpack/lut.h"
#include "xnnpack/common.h"


void xnn_x8_lut_ukernel__aarch64_neon_tbx128x4_u64(
    size_t batch,
    const uint8_t* input,
    uint8_t* output,
    const uint8_t table[restrict XNN_MIN_ELEMENTS(256)])
{
  assert(batch != 0);
  assert(batch % sizeof(uint8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const uint8x16x4_t vtable0123 = vld1q_u8_x4(table);
  const uint8x16x4_t vtable4567 = vld1q_u8_x4(table + 64);
  const uint8x16x4_t vtable89AB = vld1q_u8_x4(table + 128);
  const uint8x16x4_t vtableCDEF = vld1q_u8_x4(table + 192);
  const uint8x16_t voffset = vmovq_n_u8(64);
  for (; batch >= 64 * sizeof(uint8_t); batch -= 64 * sizeof(uint8_t)) {
    uint8x16_t vx0 = vld1q_u8(input); input += 16;
    uint8x16_t vx1 = vld1q_u8(input); input += 16;
    uint8x16_t vx2 = vld1q_u8(input); input += 16;
    uint8x16_t vx3 = vld1q_u8(input); input += 16;

    uint8x16_t vy0 = vqtbl4q_u8(vtable0123, vx0);
    vx0 = vsubq_u8(vx0, voffset);
    uint8x16_t vy1 = vqtbl4q_u8(vtable0123, vx1);
    vx1 = vsubq_u8(vx1, voffset);
    uint8x16_t vy2 = vqtbl4q_u8(vtable0123, vx2);
    vx2 = vsubq_u8(vx2, voffset);
    uint8x16_t vy3 = vqtbl4q_u8(vtable0123, vx3);
    vx3 = vsubq_u8(vx3, voffset);

    vy0 = vqtbx4q_u8(vy0, vtable4567, vx0);
    vx0 = vsubq_u8(vx0, voffset);
    vy1 = vqtbx4q_u8(vy1, vtable4567, vx1);
    vx1 = vsubq_u8(vx1, voffset);
    vy2 = vqtbx4q_u8(vy2, vtable4567, vx2);
    vx2 = vsubq_u8(vx2, voffset);
    vy3 = vqtbx4q_u8(vy3, vtable4567, vx3);
    vx3 = vsubq_u8(vx3, voffset);

    vy0 = vqtbx4q_u8(vy0, vtable89AB, vx0);
    vx0 = vsubq_u8(vx0, voffset);
    vy1 = vqtbx4q_u8(vy1, vtable89AB, vx1);
    vx1 = vsubq_u8(vx1, voffset);
    vy2 = vqtbx4q_u8(vy2, vtable89AB, vx2);
    vx2 = vsubq_u8(vx2, voffset);
    vy3 = vqtbx4q_u8(vy3, vtable89AB, vx3);
    vx3 = vsubq_u8(vx3, voffset);

    vy0 = vqtbx4q_u8(vy0, vtableCDEF, vx0);
    vy1 = vqtbx4q_u8(vy1, vtableCDEF, vx1);
    vy2 = vqtbx4q_u8(vy2, vtableCDEF, vx2);
    vy3 = vqtbx4q_u8(vy3, vtableCDEF, vx3);

    vst1q_u8(output, vy0); output += 16;
    vst1q_u8(output, vy1); output += 16;
    vst1q_u8(output, vy2); output += 16;
    vst1q_u8(output, vy3); output += 16;
  }
  for (; batch >= 16 * sizeof(uint8_t); batch -= 16 * sizeof(uint8_t)) {
    uint8x16_t vx = vld1q_u8(input); input += 16;

    uint8x16_t vy = vqtbl4q_u8(vtable0123, vx);

    vx = vsubq_u8(vx, voffset);
    vy = vqtbx4q_u8(vy, vtable4567, vx);

    vx = vsubq_u8(vx, voffset);
    vy = vqtbx4q_u8(vy, vtable89AB, vx);

    vx = vsubq_u8(vx, voffset);
    vy = vqtbx4q_u8(vy, vtableCDEF, vx);

    vst1q_u8(output, vy); output += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    uint8x16_t vx = vld1q_u8(input);

    uint8x16_t vy = vqtbl4q_u8(vtable0123, vx);

    vx = vsubq_u8(vx, voffset);
    vy = vqtbx4q_u8(vy, vtable4567, vx);

    vx = vsubq_u8(vx, voffset);
    vy = vqtbx4q_u8(vy, vtable89AB, vx);

    vx = vsubq_u8(vx, voffset);
    vy = vqtbx4q_u8(vy, vtableCDEF, vx);

    uint8x8_t vy_lo = vget_low_u8(vy);
    if (batch & (8 * sizeof(uint8_t))) {
      vst1_u8(output, vy_lo); output += 8;
      vy_lo = vget_high_u8(vy);
    }
    if (batch & (4 * sizeof(uint8_t))) {
      vst1_lane_u32((void*) output, vreinterpret_u32_u8(vy_lo), 0); output += 4;
      vy_lo = vext_u8(vy_lo, vy_lo, 4);
    }
    if (batch & (2 * sizeof(uint8_t))) {
      vst1_lane_u16((void*) output, vreinterpret_u16_u8(vy_lo), 0); output += 2;
      vy_lo = vext_u8(vy_lo, vy_lo, 2);
    }
    if (batch & (1 * sizeof(uint8_t))) {
      vst1_lane_u8(output, vy_lo, 0);
    }
  }
}
