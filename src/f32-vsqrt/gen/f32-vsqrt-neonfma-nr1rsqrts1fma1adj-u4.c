// Auto-generated file. Do not edit!
//   Template: src/f32-vsqrt/neonfma-nr1rsqrts1fma1adj.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <math.h>

#include <arm_neon.h>

#include <xnnpack/common.h>
#include <xnnpack/vunary.h>


void xnn_f32_vsqrt_ukernel__neonfma_nr1rsqrts1fma1adj_u4(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_sqrt_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const float32x4_t vhalf = vmovq_n_f32(0.5f);
  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    const float32x4_t vx = vld1q_f32(input); input += 4;
    float32x4_t vrsqrtx = vrsqrteq_f32(vx);
    const float32x4_t vrx = vmulq_f32(vrsqrtx, vrsqrtx);
    const float32x4_t vcorrection = vrsqrtsq_f32(vx, vrx);
    vrsqrtx = vmulq_f32(vrsqrtx, vcorrection);
    float32x4_t vsqrtx = vmulq_f32(vrsqrtx, vx);
    float32x4_t vhalfrsqrtx = vmulq_f32(vrsqrtx, vhalf);
    const float32x4_t vresidual = vfmsq_f32(vhalf, vsqrtx, vhalfrsqrtx);
    vhalfrsqrtx = vfmaq_f32(vhalfrsqrtx, vresidual, vhalfrsqrtx);
    vsqrtx = vfmaq_f32(vsqrtx, vresidual, vsqrtx);
    const float32x4_t vadjustment = vfmsq_f32(vx, vsqrtx, vsqrtx);
    const float32x4_t vy = vfmaq_f32(vsqrtx, vhalfrsqrtx, vadjustment);
    vst1q_f32(output, vy); output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    const float32x4_t vx = vld1q_f32(input);
    float32x4_t vrsqrtx = vrsqrteq_f32(vx);
    const float32x4_t vrx = vmulq_f32(vrsqrtx, vrsqrtx);
    const float32x4_t vcorrection = vrsqrtsq_f32(vx, vrx);
    vrsqrtx = vmulq_f32(vrsqrtx, vcorrection);
    float32x4_t vsqrtx = vmulq_f32(vrsqrtx, vx);
    float32x4_t vhalfrsqrtx = vmulq_f32(vrsqrtx, vhalf);
    const float32x4_t vresidual = vfmsq_f32(vhalf, vsqrtx, vhalfrsqrtx);
    vhalfrsqrtx = vfmaq_f32(vhalfrsqrtx, vresidual, vhalfrsqrtx);
    vsqrtx = vfmaq_f32(vsqrtx, vresidual, vsqrtx);
    const float32x4_t vadjustment = vfmsq_f32(vx, vsqrtx, vsqrtx);
    const float32x4_t vy = vfmaq_f32(vsqrtx, vhalfrsqrtx, vadjustment);

    float32x2_t vy_lo = vget_low_f32(vy);
    if (batch & (2 * sizeof(float))) {
      vst1_f32(output, vy_lo); output += 2;
      vy_lo = vget_high_f32(vy);
    }
    if (batch & (1 * sizeof(float))) {
      vst1_lane_f32(output, vy_lo, 0);
    }
  }
}
