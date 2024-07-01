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


void xnn_f32_rmin_ukernel__neon_u16_acc2(
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
  float32x4_t vmin1 = vmin0;
  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    const float32x4_t vt0 = vld1q_f32(input); input += 4;
    const float32x4_t vt1 = vld1q_f32(input); input += 4;
    const float32x4_t vt2 = vld1q_f32(input); input += 4;
    const float32x4_t vt3 = vld1q_f32(input); input += 4;

    vmin0 = vminq_f32(vmin0, vt0);
    vmin1 = vminq_f32(vmin1, vt1);
    vmin0 = vminq_f32(vmin0, vt2);
    vmin1 = vminq_f32(vmin1, vt3);
  }
  vmin0 = vminq_f32(vmin0, vmin1);
  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    const float32x4_t vt = vld1q_f32(input); input += 4;
    vmin0 = vminq_f32(vmin0, vt);
  }
  float32x2_t vmin = vmin_f32(vget_low_f32(vmin0), vget_high_f32(vmin0));
  if XNN_UNLIKELY(batch & (2 * sizeof(float))) {
    const float32x2_t vt = vld1_f32(input); input += 2;
    vmin = vmin_f32(vmin, vt);
  }
  vmin = vpmin_f32(vmin, vmin);
  if XNN_UNLIKELY(batch & (1 * sizeof(float))) {
    const float32x2_t vt = vld1_dup_f32(input);
    vmin = vmin_f32(vmin, vt);
  }
  vst1_lane_f32(output, vmin, 0);
}
