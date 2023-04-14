// Auto-generated file. Do not edit!
//   Template: src/x32-packx/neon.c.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#include <assert.h>

#include <arm_neon.h>

#include <xnnpack/packx.h>

void xnn_x32_packx_ukernel_4x__neon_st4(
    size_t m,
    size_t k,
    const uint32_t* x,
    size_t x_stride,
    uint32_t* restrict y)
{
  assert(m != 0);
  assert(m <= 4);
  assert(k != 0);
  assert(x != NULL);
  assert(y != NULL);

  const uint32_t* x0 = x;
  const uint32_t* x1 = (const uint32_t*) ((uintptr_t) x0 + x_stride);
  if XNN_UNPREDICTABLE(m < 2) {
    x1 = x0;
  }
  const uint32_t* x2 = (const uint32_t*) ((uintptr_t) x1 + x_stride);
  if XNN_UNPREDICTABLE(m <= 2) {
    x2 = x1;
  }
  const uint32_t* x3 = (const uint32_t*) ((uintptr_t) x2 + x_stride);
  if XNN_UNPREDICTABLE(m != 4) {
    x3 = x2;
  }

  for (; k >= 4; k -= 4) {
    const uint32x4_t vx0 = vld1q_u32(x0); x0 += 4;
    const uint32x4_t vx1 = vld1q_u32(x1); x1 += 4;
    const uint32x4_t vx2 = vld1q_u32(x2); x2 += 4;
    const uint32x4_t vx3 = vld1q_u32(x3); x3 += 4;
    const uint32x4x4_t vy0 = { vx0, vx1, vx2, vx3 };
    vst4q_u32(y, vy0); y += 16;
  }

  if XNN_UNLIKELY(k != 0) {
    uint32x4_t vt0123 = vdupq_n_u32(0);
    do {
      vt0123 = vld1q_lane_u32(x0, vt0123, 0); x0 += 1;
      vt0123 = vld1q_lane_u32(x1, vt0123, 1); x1 += 1;
      vt0123 = vld1q_lane_u32(x2, vt0123, 2); x2 += 1;
      vt0123 = vld1q_lane_u32(x3, vt0123, 3); x3 += 1;
      vst1q_u32(y, vt0123); y += 4;
    } while (--k != 0);
  }
}
