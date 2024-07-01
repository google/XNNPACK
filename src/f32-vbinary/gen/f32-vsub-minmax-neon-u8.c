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

#include "xnnpack/common.h"
#include "xnnpack/vbinary.h"


void xnn_f32_vsub_minmax_ukernel__neon_u8(
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

  const float32x4_t voutput_min = vld1q_dup_f32(&params->scalar.min);
  const float32x4_t voutput_max = vld1q_dup_f32(&params->scalar.max);

  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    const float32x4_t va0 = vld1q_f32(input_a); input_a += 4;
    const float32x4_t vb0 = vld1q_f32(input_b); input_b += 4;
    const float32x4_t va1 = vld1q_f32(input_a); input_a += 4;
    const float32x4_t vb1 = vld1q_f32(input_b); input_b += 4;

    float32x4_t vacc0 = vsubq_f32(va0, vb0);
    float32x4_t vacc1 = vsubq_f32(va1, vb1);


    vacc0 = vmaxq_f32(vacc0, voutput_min);
    vacc1 = vmaxq_f32(vacc1, voutput_min);

    vacc0 = vminq_f32(vacc0, voutput_max);
    vacc1 = vminq_f32(vacc1, voutput_max);

    vst1q_f32(output, vacc0); output += 4;
    vst1q_f32(output, vacc1); output += 4;
  }
  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    const float32x4_t va = vld1q_f32(input_a); input_a += 4;
    const float32x4_t vb = vld1q_f32(input_b); input_b += 4;

    float32x4_t vacc = vsubq_f32(va, vb);
    vacc = vmaxq_f32(vacc, voutput_min);
    vacc = vminq_f32(vacc, voutput_max);

    vst1q_f32(output, vacc); output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    const float32x4_t va = vld1q_f32(input_a);
    const float32x4_t vb = vld1q_f32(input_b);

    float32x4_t vacc = vsubq_f32(va, vb);
    vacc = vmaxq_f32(vacc, voutput_min);
    vacc = vminq_f32(vacc, voutput_max);

    float32x2_t vacc_lo = vget_low_f32(vacc);
    if (batch & (2 * sizeof(float))) {
      vst1_f32(output, vacc_lo); output += 2;
      vacc_lo = vget_high_f32(vacc);
    }
    if (batch & (1 * sizeof(float))) {
      vst1_lane_f32(output, vacc_lo, 0);
    }
  }
}
