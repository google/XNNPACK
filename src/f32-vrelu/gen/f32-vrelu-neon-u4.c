// Auto-generated file. Do not edit!
//   Template: src/f32-vrelu/neon.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include "xnnpack/vunary.h"
#include "xnnpack/common.h"


void xnn_f32_vrelu_ukernel__neon_u4(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_relu_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const float32x4_t vzero = vmovq_n_f32(0.0f);

  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    float32x4_t vacc0123 = vld1q_f32(input); input += 4;

    vacc0123 = vmaxq_f32(vacc0123, vzero);

    vst1q_f32(output, vacc0123); output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    float32x4_t vacc = vld1q_f32(input);
    vacc = vmaxq_f32(vacc, vzero);

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
