// Auto-generated file. Do not edit!
//   Template: src/f32-vsqrt/neon-sqrt.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <math.h>

#include <arm_neon.h>

#include "xnnpack/common.h"
#include "xnnpack/vunary.h"


void xnn_f32_vsqrt_ukernel__aarch64_neon_sqrt_u16(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_sqrt_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    const float32x4_t vx0123 = vld1q_f32(input); input += 4;
    const float32x4_t vx4567 = vld1q_f32(input); input += 4;
    const float32x4_t vx89AB = vld1q_f32(input); input += 4;
    const float32x4_t vxCDEF = vld1q_f32(input); input += 4;

    const float32x4_t vy0123 = vsqrtq_f32(vx0123);
    const float32x4_t vy4567 = vsqrtq_f32(vx4567);
    const float32x4_t vy89AB = vsqrtq_f32(vx89AB);
    const float32x4_t vyCDEF = vsqrtq_f32(vxCDEF);

    vst1q_f32(output, vy0123); output += 4;
    vst1q_f32(output, vy4567); output += 4;
    vst1q_f32(output, vy89AB); output += 4;
    vst1q_f32(output, vyCDEF); output += 4;
  }
  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    const float32x4_t vx = vld1q_f32(input); input += 4;
    const float32x4_t vy = vsqrtq_f32(vx);
    vst1q_f32(output, vy); output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    const float32x4_t vx = vld1q_f32(input);

    float32x2_t vy_lo = vsqrt_f32(vget_low_f32(vx));
    const float32x2_t vx_hi = vget_high_f32(vx);
    if (batch & (2 * sizeof(float))) {
      vst1_f32(output, vy_lo); output += 2;
      vy_lo = vsqrt_f32(vx_hi);
    }
    if (batch & (1 * sizeof(float))) {
      vst1_lane_f32(output, vy_lo, 0);
    }
  }
}
