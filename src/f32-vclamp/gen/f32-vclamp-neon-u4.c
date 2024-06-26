// Auto-generated file. Do not edit!
//   Template: src/f32-vclamp/neon.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include "xnnpack/common.h"
#include "xnnpack/vunary.h"


void xnn_f32_vclamp_ukernel__neon_u4(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  #if XNN_ARCH_ARM64
    const float32x4x2_t vminmax = vld2q_dup_f32(&params->scalar.min);
    const float32x4_t vmin = vminmax.val[0];
    const float32x4_t vmax = vminmax.val[1];
  #else
    const float32x2x2_t vminmax = vld2_dup_f32(&params->scalar.min);
    const float32x4_t vmin = vcombine_f32(vminmax.val[0], vminmax.val[0]);
    const float32x4_t vmax = vcombine_f32(vminmax.val[1], vminmax.val[1]);
  #endif

  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    float32x4_t vacc0123 = vld1q_f32(input); input += 4;

    vacc0123 = vmaxq_f32(vacc0123, vmin);

    vacc0123 = vminq_f32(vacc0123, vmax);

    vst1q_f32(output, vacc0123); output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    if (batch & (2 * sizeof(float))) {
      float32x2_t vacc = vld1_f32(input); input += 2;
      vacc = vmax_f32(vacc, vget_low_f32(vmin));
      vacc = vmin_f32(vacc, vget_low_f32(vmax));
      vst1_f32(output, vacc); output += 2;
    }
    if (batch & (1 * sizeof(float))) {
      float32x2_t vacc = vld1_dup_f32(input);
      vacc = vmax_f32(vacc, vget_low_f32(vmin));
      vacc = vmin_f32(vacc, vget_low_f32(vmax));
      vst1_lane_f32(output, vacc, 0);
    }
  }
}
