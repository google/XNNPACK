// Auto-generated file. Do not edit!
//   Template: src/f32-rminmax/neon.c.in
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


void xnn_f32_rminmax_ukernel__neon_u16_acc4(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  float32x4_t vmin0 = vld1q_dup_f32(input);
  float32x4_t vmax0 = vmin0;
  float32x4_t vmin1 = vmin0;
  float32x4_t vmax1 = vmax0;
  float32x4_t vmin2 = vmin0;
  float32x4_t vmax2 = vmax0;
  float32x4_t vmin3 = vmin0;
  float32x4_t vmax3 = vmax0;
  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    const float32x4_t vt0 = vld1q_f32(input); input += 4;
    const float32x4_t vt1 = vld1q_f32(input); input += 4;
    const float32x4_t vt2 = vld1q_f32(input); input += 4;
    const float32x4_t vt3 = vld1q_f32(input); input += 4;

    vmin0 = vminq_f32(vmin0, vt0);
    vmax0 = vmaxq_f32(vmax0, vt0);
    vmin1 = vminq_f32(vmin1, vt1);
    vmax1 = vmaxq_f32(vmax1, vt1);
    vmin2 = vminq_f32(vmin2, vt2);
    vmax2 = vmaxq_f32(vmax2, vt2);
    vmin3 = vminq_f32(vmin3, vt3);
    vmax3 = vmaxq_f32(vmax3, vt3);
  }
  vmin0 = vminq_f32(vmin0, vmin1);
  vmax0 = vmaxq_f32(vmax0, vmax1);
  vmin2 = vminq_f32(vmin2, vmin3);
  vmax2 = vmaxq_f32(vmax2, vmax3);
  vmin0 = vminq_f32(vmin0, vmin2);
  vmax0 = vmaxq_f32(vmax0, vmax2);
  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    const float32x4_t vt = vld1q_f32(input); input += 4;
    vmin0 = vminq_f32(vmin0, vt);
    vmax0 = vmaxq_f32(vmax0, vt);
  }
  float32x2_t vmin = vmin_f32(vget_low_f32(vmin0), vget_high_f32(vmin0));
  float32x2_t vmax = vmax_f32(vget_low_f32(vmax0), vget_high_f32(vmax0));
  if XNN_UNLIKELY(batch & (2 * sizeof(float))) {
    const float32x2_t vt = vld1_f32(input); input += 2;
    vmin = vmin_f32(vmin, vt);
    vmax = vmax_f32(vmax, vt);
  }
  vmin = vpmin_f32(vmin, vmin);
  vmax = vpmax_f32(vmax, vmax);
  if XNN_UNLIKELY(batch & (1 * sizeof(float))) {
    const float32x2_t vt = vld1_dup_f32(input);
    vmin = vmin_f32(vmin, vt);
    vmax = vmax_f32(vmax, vt);
  }
  vst1_lane_f32(output, vmin, 0);
  vst1_lane_f32(output + 1, vmax, 0);
}
