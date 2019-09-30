// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include <xnnpack/pad.h>


void xnn_x32_pad_x2__neon(
    size_t m,
    size_t n,
    size_t l,
    size_t r,
    uint32_t c,
    const void* x,
    size_t x_stride,
    void* y,
    size_t y_stride)
{
  assert(m <= 2);
  assert(l % 4 == 0);
  assert(n % 4 == 0);
  assert(r % 4 == 0);

  const uint32_t* x0 = x;
  uint32_t* y0 = y;

  const uint32_t* x1 = (const uint32_t*) ((uintptr_t) x0 + x_stride);
  uint32_t* y1 = (uint32_t*) ((uintptr_t) y0 + y_stride);
  if (m != 2) {
    x1 = x0;
    y1 = y0;
  }
  const uint32x4_t vc = vmovq_n_u32(c);

  // Pre-pad input channels.
  for (; l >= 16; l -= 16) {
    vst1q_u32(y0, vc); y0 += 4;
    vst1q_u32(y1, vc); y1 += 4;
  }
  if (l & 8) {
    vst1_u32(y0, vget_low_u32(vc)); y0 += 2;
    vst1_u32(y1, vget_low_u32(vc)); y1 += 2;
  }
  if (l & 4) {
    vst1q_lane_u32(y0, vc, 0); y0 += 1;
    vst1q_lane_u32(y1, vc, 0); y1 += 1;
  }

  // Copy input channels.
  for (; n >= 16; n -= 16) {
    const uint32x4_t vt0 = vld1q_u32(x0); x0 += 4;
    const uint32x4_t vt1 = vld1q_u32(x1); x1 += 4;
    vst1q_u32(y0, vt0); y0 += 4;
    vst1q_u32(y1, vt1); y1 += 4;
  }
  if (n != 0) {
    const uint32x4_t vt0 = vld1q_u32(x0); x0 += 4;
    const uint32x4_t vt1 = vld1q_u32(x1); x1 += 4;
    uint32x2_t vt0lo = vget_low_u32(vt0);
    uint32x2_t vt1lo = vget_low_u32(vt1);
    if (n & 8) {
      vst1_u32(y0, vt0lo); y0 += 2;
      vst1_u32(y1, vt1lo); y1 += 2;
      vt0lo = vget_high_u32(vt0);
      vt1lo = vget_high_u32(vt1);
    }
    if (n & 4) {
      vst1_lane_u32(y0, vt0lo, 0); y0 += 1;
      vst1_lane_u32(y1, vt1lo, 0); y1 += 1;
    }
  }

  // Post-pad input channels.
  for (; r >= 16; r -= 16) {
    vst1q_u32(y0, vc); y0 += 4;
    vst1q_u32(y1, vc); y1 += 4;
  }
  if (r & 8) {
    vst1_u32(y0, vget_low_u32(vc)); y0 += 2;
    vst1_u32(y1, vget_low_u32(vc)); y1 += 2;
  }
  if (r & 4) {
    vst1q_lane_u32(y0, vc, 0);
    vst1q_lane_u32(y1, vc, 0);
  }
}
