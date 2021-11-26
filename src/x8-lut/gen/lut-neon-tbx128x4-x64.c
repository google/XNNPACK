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

#include <xnnpack/intrinsics-polyfill.h>
#include <xnnpack/lut.h>
#include <xnnpack/common.h>


void xnn_x8_lut_ukernel__neon_tbx128x4_x64(
    size_t n,
    const uint8_t* x,
    uint8_t* y,
    const uint8_t t[restrict XNN_MIN_ELEMENTS(256)])
{
  assert(n != 0);
  assert(x != NULL);
  assert(y != NULL);

  const uint8x16x4_t vtable0123 = vld1q_u8_x4(t);
  const uint8x16x4_t vtable4567 = vld1q_u8_x4(t + 64);
  const uint8x16x4_t vtable89AB = vld1q_u8_x4(t + 128);
  const uint8x16x4_t vtableCDEF = vld1q_u8_x4(t + 192);
  const uint8x16_t voffset = vmovq_n_u8(64);
  for (; n >= 64 * sizeof(uint8_t); n -= 64 * sizeof(uint8_t)) {
    uint8x16_t vx0 = vld1q_u8(x); x += 16;
    uint8x16_t vx1 = vld1q_u8(x); x += 16;
    uint8x16_t vx2 = vld1q_u8(x); x += 16;
    uint8x16_t vx3 = vld1q_u8(x); x += 16;

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

    vst1q_u8(y, vy0); y += 16;
    vst1q_u8(y, vy1); y += 16;
    vst1q_u8(y, vy2); y += 16;
    vst1q_u8(y, vy3); y += 16;
  }
  for (; n >= 16 * sizeof(uint8_t); n -= 16 * sizeof(uint8_t)) {
    uint8x16_t vx = vld1q_u8(x); x += 16;

    uint8x16_t vy = vqtbl4q_u8(vtable0123, vx);

    vx = vsubq_u8(vx, voffset);
    vy = vqtbx4q_u8(vy, vtable4567, vx);

    vx = vsubq_u8(vx, voffset);
    vy = vqtbx4q_u8(vy, vtable89AB, vx);

    vx = vsubq_u8(vx, voffset);
    vy = vqtbx4q_u8(vy, vtableCDEF, vx);

    vst1q_u8(y, vy); y += 16;
  }
  if XNN_UNLIKELY(n != 0) {
    uint8x16_t vx = vld1q_u8(x);

    uint8x16_t vy = vqtbl4q_u8(vtable0123, vx);

    vx = vsubq_u8(vx, voffset);
    vy = vqtbx4q_u8(vy, vtable4567, vx);

    vx = vsubq_u8(vx, voffset);
    vy = vqtbx4q_u8(vy, vtable89AB, vx);

    vx = vsubq_u8(vx, voffset);
    vy = vqtbx4q_u8(vy, vtableCDEF, vx);

    uint8x8_t vy_lo = vget_low_u8(vy);
    if (n & (8 * sizeof(uint8_t))) {
      vst1_u8(y, vy_lo); y += 8;
      vy_lo = vget_high_u8(vy);
    }
    if (n & (4 * sizeof(uint8_t))) {
      vst1_lane_u32((void*) y, vreinterpret_u32_u8(vy_lo), 0); y += 4;
      vy_lo = vext_u8(vy_lo, vy_lo, 4);
    }
    if (n & (2 * sizeof(uint8_t))) {
      vst1_lane_u16((void*) y, vreinterpret_u16_u8(vy_lo), 0); y += 2;
      vy_lo = vext_u8(vy_lo, vy_lo, 2);
    }
    if (n & (1 * sizeof(uint8_t))) {
      vst1_lane_u8(y, vy_lo, 0);
    }
  }
}
