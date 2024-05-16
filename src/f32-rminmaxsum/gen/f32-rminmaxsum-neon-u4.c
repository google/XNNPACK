// Auto-generated file. Do not edit!
//   Template: src/f32-rminmaxsum/neon.c.in
//   Generator: tools/xngen
//
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include <xnnpack/common.h>
#include <xnnpack/reduce.h>


void xnn_f32_rminmaxsum_ukernel__neon_u4(
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
  float32x4_t vsum0 = vmovq_n_f32(0.0f);
  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    const float32x4_t vt = vld1q_f32(input); input += 4;
    vmin0 = vminq_f32(vmin0, vt);
    vmax0 = vmaxq_f32(vmax0, vt);
    vsum0 = vaddq_f32(vsum0, vt);
  }
  float32x2_t vmin = vmin_f32(vget_low_f32(vmin0), vget_high_f32(vmin0));
  float32x2_t vmax = vmax_f32(vget_low_f32(vmax0), vget_high_f32(vmax0));
  float32x2_t vsum = vadd_f32(vget_low_f32(vsum0), vget_high_f32(vsum0));
  if XNN_UNLIKELY(batch & (2 * sizeof(float))) {
    const float32x2_t vt = vld1_f32(input); input += 2;
    vmin = vmin_f32(vmin, vt);
    vmax = vmax_f32(vmax, vt);
    vsum = vadd_f32(vsum, vt);
  }
  vmin = vpmin_f32(vmin, vmin);
  vmax = vpmax_f32(vmax, vmax);
  vsum = vpadd_f32(vsum, vsum);
  if XNN_UNLIKELY(batch & (1 * sizeof(float))) {
    const float32x2_t vt = vld1_dup_f32(input);
    vmin = vmin_f32(vmin, vt);
    vmax = vmax_f32(vmax, vt);
    vsum = vadd_f32(vsum, vt);
  }
  vst1_lane_f32(output, vmin, 0);
  vst1_lane_f32(output + 1, vmax, 0);
  vst1_lane_f32(output + 2, vsum, 0);
}
