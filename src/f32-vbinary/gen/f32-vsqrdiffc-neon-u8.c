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


void xnn_f32_vsqrdiffc_ukernel__neon_u8(
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

  const float32x4_t vb = vld1q_dup_f32(input_b);

  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    float32x4_t vacc_ = vld1q_f32(input_a); input_a += 4;
    float32x4_t vaccl = vld1q_f32(input_a); input_a += 4;

    vacc_ = vsubq_f32(vacc_, vb);
    vaccl = vsubq_f32(vaccl, vb);

    vacc_ = vmulq_f32(vacc_, vacc_);
    vaccl = vmulq_f32(vaccl, vaccl);


    vst1q_f32(output, vacc_); output += 4;
    vst1q_f32(output, vaccl); output += 4;
  }
  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    const float32x4_t va = vld1q_f32(input_a); input_a += 4;

    float32x4_t vacc = vsubq_f32(va, vb);
    vacc = vmulq_f32(vacc, vacc);

    vst1q_f32(output, vacc); output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    const float32x4_t va = vld1q_f32(input_a);

    float32x4_t vacc = vsubq_f32(va, vb);
    vacc = vmulq_f32(vacc, vacc);

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
