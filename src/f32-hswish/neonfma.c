// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include <xnnpack/hswish.h>


void xnn_f32_hswish_ukernel__neonfma(
    size_t n,
    const float* x,
    float* y,
    const union xnn_f32_hswish_params params[restrict static 1])
{
  assert(n != 0);
  assert(n % sizeof(float) == 0);

  const float32x4_t vsixth = vld1q_dup_f32(&params->scalar.sixth);
  const float32x4_t vhalf = vld1q_dup_f32(&params->scalar.half);
  const float32x4_t vone = vld1q_dup_f32(&params->scalar.one);
  const float32x4_t vzero = vdupq_n_f32(0.0f);

  for (; n >= 16; n -= 16) {
    const float32x4_t vx = vld1q_f32(x); x += 4;

    const float32x4_t vt = vminq_f32(vmaxq_f32(vfmaq_f32(vhalf, vx, vsixth), vzero), vone);
    const float32x4_t vy = vmulq_f32(vt, vx);

    vst1q_f32(y, vy); y += 4;
  }
  if (n != 0) {
    const float32x4_t vx = vld1q_f32(x); x += 4;

    const float32x4_t vt = vminq_f32(vmaxq_f32(vfmaq_f32(vhalf, vx, vsixth), vzero), vone);
    const float32x4_t vy = vmulq_f32(vt, vx);

    float32x2_t vy_lo = vget_low_f32(vy);
    if (n & 8) {
      vst1_f32(y, vy_lo); y += 2;
      vy_lo = vget_high_f32(vy);
    }
    if (n & 4) {
      vst1_lane_f32(y, vy_lo, 0);
    }
  }
}
