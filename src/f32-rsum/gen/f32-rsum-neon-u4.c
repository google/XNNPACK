// Auto-generated file. Do not edit!
//   Template: src/f32-rsum/neon.c.in
//   Generator: tools/xngen
//
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include "xnnpack/common.h"
#include "xnnpack/reduce.h"


void xnn_f32_rsum_ukernel__neon_u4(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_scaleminmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  float32x4_t vacc0 = vmovq_n_f32(0.0f);
  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    const float32x4_t vt = vld1q_f32(input); input += 4;
    vacc0 = vaddq_f32(vacc0, vt);
  }
  const float32x2_t vscale = vld1_dup_f32(&params->scalar.scale);
  const float32x2_t vmin = vld1_dup_f32(&params->scalar.min);
  const float32x2_t vmax = vld1_dup_f32(&params->scalar.max);
  float32x2_t vacc = vadd_f32(vget_low_f32(vacc0), vget_high_f32(vacc0));
  if XNN_UNLIKELY(batch & (2 * sizeof(float))) {
    const float32x2_t vt = vld1_f32(input); input += 2;
    vacc = vadd_f32(vacc, vt);
  }
  vacc = vpadd_f32(vacc, vacc);
  if XNN_UNLIKELY(batch & (1 * sizeof(float))) {
    const float32x2_t vt = vld1_dup_f32(input);
    vacc = vadd_f32(vacc, vt);
  }
  vacc = vmul_f32(vacc, vscale);
  vacc = vmax_f32(vacc, vmin);
  vacc = vmin_f32(vacc, vmax);
  *output += vget_lane_f32(vacc, 0);
}
