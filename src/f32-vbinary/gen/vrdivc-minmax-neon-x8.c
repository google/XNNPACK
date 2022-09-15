// Auto-generated file. Do not edit!
//   Template: src/f32-vbinary/vopc-neon.c.in
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


void xnn_f32_vrdivc_minmax_ukernel__neon_x8(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const float32x4_t vy_min = vld1q_dup_f32(&params->scalar.min);
  const float32x4_t vy_max = vld1q_dup_f32(&params->scalar.max);

  const float32x4_t vb = vld1q_dup_f32(input_b);
  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    const float32x4_t va0123 = vld1q_f32(input_a); input_a += 4;
    const float32x4_t va4567 = vld1q_f32(input_a); input_a += 4;

    float32x4_t vy0123 = vdivq_f32(vb, va0123);
    float32x4_t vy4567 = vdivq_f32(vb, va4567);


    vy0123 = vmaxq_f32(vy0123, vy_min);
    vy4567 = vmaxq_f32(vy4567, vy_min);

    vy0123 = vminq_f32(vy0123, vy_max);
    vy4567 = vminq_f32(vy4567, vy_max);

    vst1q_f32(output, vy0123); output += 4;
    vst1q_f32(output, vy4567); output += 4;
  }
  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    const float32x4_t va0123 = vld1q_f32(input_a); input_a += 4;

    float32x4_t vy0123 = vdivq_f32(vb, va0123);
    vy0123 = vmaxq_f32(vy0123, vy_min);
    vy0123 = vminq_f32(vy0123, vy_max);
    vst1q_f32(output, vy0123); output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    const float32x4_t va0123 = vld1q_f32(input_a);

    float32x4_t vy0123 = vdivq_f32(vb, va0123);
    vy0123 = vmaxq_f32(vy0123, vy_min);
    vy0123 = vminq_f32(vy0123, vy_max);

    float32x2_t vy01 = vget_low_f32(vy0123);
    if (batch & (2 * sizeof(float))) {
      vst1_f32(output, vy01); output += 2;
      vy01 = vget_high_f32(vy0123);
    }
    if (batch & (1 * sizeof(float))) {
      vst1_lane_f32(output, vy01, 0);
    }
  }
}
