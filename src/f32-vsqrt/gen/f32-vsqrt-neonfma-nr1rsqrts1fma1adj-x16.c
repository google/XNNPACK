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


void xnn_f32_vsqrt_ukernel__neonfma_nr1rsqrts1fma1adj_x16(
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
  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    const float32x4_t vx0123 = vld1q_f32(input); input += 4;
    const float32x4_t vx4567 = vld1q_f32(input); input += 4;
    const float32x4_t vx89AB = vld1q_f32(input); input += 4;
    const float32x4_t vxCDEF = vld1q_f32(input); input += 4;

    float32x4_t vrsqrtx0123 = vrsqrteq_f32(vx0123);
    float32x4_t vrsqrtx4567 = vrsqrteq_f32(vx4567);
    float32x4_t vrsqrtx89AB = vrsqrteq_f32(vx89AB);
    float32x4_t vrsqrtxCDEF = vrsqrteq_f32(vxCDEF);

    const float32x4_t vrx0123 = vmulq_f32(vrsqrtx0123, vrsqrtx0123);
    const float32x4_t vrx4567 = vmulq_f32(vrsqrtx4567, vrsqrtx4567);
    const float32x4_t vrx89AB = vmulq_f32(vrsqrtx89AB, vrsqrtx89AB);
    const float32x4_t vrxCDEF = vmulq_f32(vrsqrtxCDEF, vrsqrtxCDEF);

    const float32x4_t vcorrection0123 = vrsqrtsq_f32(vx0123, vrx0123);
    const float32x4_t vcorrection4567 = vrsqrtsq_f32(vx4567, vrx4567);
    const float32x4_t vcorrection89AB = vrsqrtsq_f32(vx89AB, vrx89AB);
    const float32x4_t vcorrectionCDEF = vrsqrtsq_f32(vxCDEF, vrxCDEF);

    vrsqrtx0123 = vmulq_f32(vrsqrtx0123, vcorrection0123);
    vrsqrtx4567 = vmulq_f32(vrsqrtx4567, vcorrection4567);
    vrsqrtx89AB = vmulq_f32(vrsqrtx89AB, vcorrection89AB);
    vrsqrtxCDEF = vmulq_f32(vrsqrtxCDEF, vcorrectionCDEF);

    float32x4_t vsqrtx0123 = vmulq_f32(vrsqrtx0123, vx0123);
    float32x4_t vhalfrsqrtx0123 = vmulq_f32(vrsqrtx0123, vhalf);
    float32x4_t vsqrtx4567 = vmulq_f32(vrsqrtx4567, vx4567);
    float32x4_t vhalfrsqrtx4567 = vmulq_f32(vrsqrtx4567, vhalf);
    float32x4_t vsqrtx89AB = vmulq_f32(vrsqrtx89AB, vx89AB);
    float32x4_t vhalfrsqrtx89AB = vmulq_f32(vrsqrtx89AB, vhalf);
    float32x4_t vsqrtxCDEF = vmulq_f32(vrsqrtxCDEF, vxCDEF);
    float32x4_t vhalfrsqrtxCDEF = vmulq_f32(vrsqrtxCDEF, vhalf);

    const float32x4_t vresidual0123 = vfmsq_f32(vhalf, vsqrtx0123, vhalfrsqrtx0123);
    const float32x4_t vresidual4567 = vfmsq_f32(vhalf, vsqrtx4567, vhalfrsqrtx4567);
    const float32x4_t vresidual89AB = vfmsq_f32(vhalf, vsqrtx89AB, vhalfrsqrtx89AB);
    const float32x4_t vresidualCDEF = vfmsq_f32(vhalf, vsqrtxCDEF, vhalfrsqrtxCDEF);

    vhalfrsqrtx0123 = vfmaq_f32(vhalfrsqrtx0123, vresidual0123, vhalfrsqrtx0123);
    vsqrtx0123 = vfmaq_f32(vsqrtx0123, vresidual0123, vsqrtx0123);
    vhalfrsqrtx4567 = vfmaq_f32(vhalfrsqrtx4567, vresidual4567, vhalfrsqrtx4567);
    vsqrtx4567 = vfmaq_f32(vsqrtx4567, vresidual4567, vsqrtx4567);
    vhalfrsqrtx89AB = vfmaq_f32(vhalfrsqrtx89AB, vresidual89AB, vhalfrsqrtx89AB);
    vsqrtx89AB = vfmaq_f32(vsqrtx89AB, vresidual89AB, vsqrtx89AB);
    vhalfrsqrtxCDEF = vfmaq_f32(vhalfrsqrtxCDEF, vresidualCDEF, vhalfrsqrtxCDEF);
    vsqrtxCDEF = vfmaq_f32(vsqrtxCDEF, vresidualCDEF, vsqrtxCDEF);

    const float32x4_t vadjustment0123 = vfmsq_f32(vx0123, vsqrtx0123, vsqrtx0123);
    const float32x4_t vadjustment4567 = vfmsq_f32(vx4567, vsqrtx4567, vsqrtx4567);
    const float32x4_t vadjustment89AB = vfmsq_f32(vx89AB, vsqrtx89AB, vsqrtx89AB);
    const float32x4_t vadjustmentCDEF = vfmsq_f32(vxCDEF, vsqrtxCDEF, vsqrtxCDEF);

    const float32x4_t vy0123 = vfmaq_f32(vsqrtx0123, vhalfrsqrtx0123, vadjustment0123);
    const float32x4_t vy4567 = vfmaq_f32(vsqrtx4567, vhalfrsqrtx4567, vadjustment4567);
    const float32x4_t vy89AB = vfmaq_f32(vsqrtx89AB, vhalfrsqrtx89AB, vadjustment89AB);
    const float32x4_t vyCDEF = vfmaq_f32(vsqrtxCDEF, vhalfrsqrtxCDEF, vadjustmentCDEF);

    vst1q_f32(output, vy0123); output += 4;
    vst1q_f32(output, vy4567); output += 4;
    vst1q_f32(output, vy89AB); output += 4;
    vst1q_f32(output, vyCDEF); output += 4;
  }
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
