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

#include "xnnpack/packx.h"


void xnn_x32_packx_ukernel_8x__neon_st4_x8(
    size_t m,
    size_t k,
    const uint32_t* x,
    size_t x_stride,
    uint32_t* restrict y)
{
  assert(m != 0);
  assert(m <= 8);
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
  if XNN_UNPREDICTABLE(m < 4) {
    x3 = x2;
  }
  const uint32_t* x4 = (const uint32_t*) ((uintptr_t) x3 + x_stride);
  if XNN_UNPREDICTABLE(m <= 4) {
    x4 = x3;
  }
  const uint32_t* x5 = (const uint32_t*) ((uintptr_t) x4 + x_stride);
  if XNN_UNPREDICTABLE(m < 6) {
    x5 = x4;
  }
  const uint32_t* x6 = (const uint32_t*) ((uintptr_t) x5 + x_stride);
  if XNN_UNPREDICTABLE(m <= 6) {
    x6 = x5;
  }
  const uint32_t* x7 = (const uint32_t*) ((uintptr_t) x6 + x_stride);
  if XNN_UNPREDICTABLE(m != 8) {
    x7 = x6;
  }

  for (; k >= 8; k -= 8) {
    const uint32x4_t vx0123x0 = vld1q_u32(x0); x0 += 4;
    const uint32x4_t vx0123x1 = vld1q_u32(x1); x1 += 4;
    const uint32x4_t vx0123x2 = vld1q_u32(x2); x2 += 4;
    const uint32x4_t vx0123x3 = vld1q_u32(x3); x3 += 4;
    const uint32x4_t vx0123x4 = vld1q_u32(x4); x4 += 4;
    const uint32x4_t vx0123x5 = vld1q_u32(x5); x5 += 4;
    const uint32x4_t vx0123x6 = vld1q_u32(x6); x6 += 4;
    const uint32x4_t vx0123x7 = vld1q_u32(x7); x7 += 4;
    const uint32x4_t vx4567x0 = vld1q_u32(x0); x0 += 4;
    const uint32x4_t vx4567x1 = vld1q_u32(x1); x1 += 4;
    const uint32x4_t vx4567x2 = vld1q_u32(x2); x2 += 4;
    const uint32x4_t vx4567x3 = vld1q_u32(x3); x3 += 4;
    const uint32x4_t vx4567x4 = vld1q_u32(x4); x4 += 4;
    const uint32x4_t vx4567x5 = vld1q_u32(x5); x5 += 4;
    const uint32x4_t vx4567x6 = vld1q_u32(x6); x6 += 4;
    const uint32x4_t vx4567x7 = vld1q_u32(x7); x7 += 4;
    const uint32x4x2_t vz0123x0 = vzipq_u32(vx0123x0, vx0123x4);
    const uint32x4x2_t vz0123x1 = vzipq_u32(vx0123x1, vx0123x5);
    const uint32x4x2_t vz0123x2 = vzipq_u32(vx0123x2, vx0123x6);
    const uint32x4x2_t vz0123x3 = vzipq_u32(vx0123x3, vx0123x7);
    const uint32x4x2_t vz4567x0 = vzipq_u32(vx4567x0, vx4567x4);
    const uint32x4x2_t vz4567x1 = vzipq_u32(vx4567x1, vx4567x5);
    const uint32x4x2_t vz4567x2 = vzipq_u32(vx4567x2, vx4567x6);
    const uint32x4x2_t vz4567x3 = vzipq_u32(vx4567x3, vx4567x7);

    const uint32x4x4_t vy0123x0 = { vz0123x0.val[0], vz0123x1.val[0], vz0123x2.val[0], vz0123x3.val[0] };
    const uint32x4x4_t vy0123x1 = { vz0123x0.val[1], vz0123x1.val[1], vz0123x2.val[1], vz0123x3.val[1] };
    const uint32x4x4_t vy4567x0 = { vz4567x0.val[0], vz4567x1.val[0], vz4567x2.val[0], vz4567x3.val[0] };
    const uint32x4x4_t vy4567x1 = { vz4567x0.val[1], vz4567x1.val[1], vz4567x2.val[1], vz4567x3.val[1] };
    vst4q_u32(y, vy0123x0); y += 16;
    vst4q_u32(y, vy0123x1); y += 16;
    vst4q_u32(y, vy4567x0); y += 16;
    vst4q_u32(y, vy4567x1); y += 16;
  }

  if XNN_UNLIKELY(k != 0) {
    uint32x4_t vt0123 = vdupq_n_u32(0);
    uint32x4_t vt4567 = vdupq_n_u32(0);
    do {
      vt0123 = vld1q_lane_u32(x0, vt0123, 0); x0 += 1;
      vt0123 = vld1q_lane_u32(x1, vt0123, 1); x1 += 1;
      vt0123 = vld1q_lane_u32(x2, vt0123, 2); x2 += 1;
      vt0123 = vld1q_lane_u32(x3, vt0123, 3); x3 += 1;
      vt4567 = vld1q_lane_u32(x4, vt4567, 0); x4 += 1;
      vt4567 = vld1q_lane_u32(x5, vt4567, 1); x5 += 1;
      vt4567 = vld1q_lane_u32(x6, vt4567, 2); x6 += 1;
      vt4567 = vld1q_lane_u32(x7, vt4567, 3); x7 += 1;
      vst1q_u32(y, vt0123); y += 4;
      vst1q_u32(y, vt4567); y += 4;
    } while (--k != 0);
  }
}
