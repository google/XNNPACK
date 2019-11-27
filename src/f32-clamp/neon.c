// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include <xnnpack/clamp.h>
#include <xnnpack/common.h>


void xnn_f32_clamp_ukernel__neon(
    size_t n,
    const float* x,
    float* y,
    const union xnn_f32_output_params params[restrict static 1])
{
  assert(n != 0);
  assert(n % sizeof(float) == 0);

#if XNN_ARCH_ARM64
  const float32x4x2_t voutput_clamp = vld2q_dup_f32(&params->scalar.max);
  const float32x4_t voutput_max = voutput_clamp.val[0];
  const float32x4_t voutput_min = voutput_clamp.val[1];
#else
  const float32x4_t voutput_max = vld1q_dup_f32(&params->scalar.max);
  const float32x4_t voutput_min = vld1q_dup_f32(&params->scalar.min);
#endif

  for (; n >= 4 * sizeof(float); n -= 4 * sizeof(float)) {
    const float32x4_t vx = vld1q_f32(x); x += 4;

    float32x4_t vy = vminq_f32(vx, voutput_max);
    vy = vmaxq_f32(vy, voutput_min);

    vst1q_f32(y, vy); y += 4;
  }
  if (n != 0) {
    const float32x4_t vx = vld1q_f32(x);

    float32x4_t vy = vminq_f32(vx, voutput_max);
    vy = vmaxq_f32(vy, voutput_min);

    float32x2_t vy_lo = vget_low_f32(vy);
    if (n & 2 * sizeof(float)) {
      vst1_f32(y, vy_lo); y += 2;
      vy_lo = vget_high_f32(vy);
    }
    if (n & 1 * sizeof(float)) {
      vst1_lane_f32(y, vy_lo, 0);
    }
  }
}
