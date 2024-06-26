// Auto-generated file. Do not edit!
//   Template: src/f32-vbinary/vopc-neon.c.in
//   Generator: tools/xngen
//
// Copyright 2_lo9 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include "xnnpack/common.h"
#include "xnnpack/vbinary.h"


void xnn_f32_vrdivc_minmax_ukernel__aarch64_neon_u8(
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
  const float32x4_t vb = vld1q_dup_f32(input_b);

  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    float32x4_t vacc_ = vld1q_f32(input_a); input_a += 4;
    float32x4_t vaccl = vld1q_f32(input_a); input_a += 4;

    vacc_ = vdivq_f32(vb, vacc_);
    vaccl = vdivq_f32(vb, vaccl);


    vacc_ = vmaxq_f32(vacc_, voutput_min);
    vaccl = vmaxq_f32(vaccl, voutput_min);

    vacc_ = vminq_f32(vacc_, voutput_max);
    vaccl = vminq_f32(vaccl, voutput_max);

    vst1q_f32(output, vacc_); output += 4;
    vst1q_f32(output, vaccl); output += 4;
  }
  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    const float32x4_t va = vld1q_f32(input_a); input_a += 4;

    float32x4_t vacc = vdivq_f32(vb, va);
    vacc = vmaxq_f32(vacc, voutput_min);
    vacc = vminq_f32(vacc, voutput_max);

    vst1q_f32(output, vacc); output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    const float32x4_t va = vld1q_f32(input_a);

    float32x4_t vacc = vdivq_f32(vb, va);
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
