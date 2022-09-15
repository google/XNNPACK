// Auto-generated file. Do not edit!
//   Template: src/f32-vbinary/vop-neon.c.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include <xnnpack/common.h>
#include <xnnpack/vbinary.h>


void xnn_f32_vsqrdiff_ukernel__neon_x8(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);


  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    const float32x4_t va0123 = vld1q_f32(input_a); input_a += 4;
    const float32x4_t vb0123 = vld1q_f32(input_b); input_b += 4;
    const float32x4_t va4567 = vld1q_f32(input_a); input_a += 4;
    const float32x4_t vb4567 = vld1q_f32(input_b); input_b += 4;

    float32x4_t vy0123 = vsubq_f32(va0123, vb0123);
    float32x4_t vy4567 = vsubq_f32(va4567, vb4567);

    vy0123 = vmulq_f32(vy0123, vy0123);
    vy4567 = vmulq_f32(vy4567, vy4567);


    vst1q_f32(output, vy0123); output += 4;
    vst1q_f32(output, vy4567); output += 4;
  }
  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    const float32x4_t va = vld1q_f32(input_a); input_a += 4;
    const float32x4_t vb = vld1q_f32(input_b); input_b += 4;

    float32x4_t vy = vsubq_f32(va, vb);
    vy = vmulq_f32(vy, vy);
    vst1q_f32(output, vy); output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    const float32x4_t va = vld1q_f32(input_a);
    const float32x4_t vb = vld1q_f32(input_b);

    float32x4_t vy = vsubq_f32(va, vb);
    vy = vmulq_f32(vy, vy);

    float32x2_t vy_lo = vget_low_f32(vy);
    if (batch & (2 * sizeof(float))) {
      vst1_f32(output, vy_lo); output += 2;
      vy_lo = vget_high_f32(vy);
    }
    if (batch & (1 * sizeof(float))) {
      vst1_lane_f32(output, vy_lo, 0);
    }
  }
}
