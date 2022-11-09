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


void xnn_f32_vsqrt_ukernel__neonfma_nr1rsqrts1fma1adj_x40(
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
  for (; batch >= 40 * sizeof(float); batch -= 40 * sizeof(float)) {
    const float32x4_t vx0123 = vld1q_f32(input); input += 4;
    const float32x4_t vx4567 = vld1q_f32(input); input += 4;
    const float32x4_t vx89AB = vld1q_f32(input); input += 4;
    const float32x4_t vxCDEF = vld1q_f32(input); input += 4;
    const float32x4_t vxGHIJ = vld1q_f32(input); input += 4;
    const float32x4_t vxKLMN = vld1q_f32(input); input += 4;
    const float32x4_t vxOPQR = vld1q_f32(input); input += 4;
    const float32x4_t vxSTUV = vld1q_f32(input); input += 4;
    const float32x4_t vxWXYZ = vld1q_f32(input); input += 4;
    const float32x4_t vx = vld1q_f32(input); input += 4;

    float32x4_t vrsqrtx0123 = vrsqrteq_f32(vx0123);
    float32x4_t vrsqrtx4567 = vrsqrteq_f32(vx4567);
    float32x4_t vrsqrtx89AB = vrsqrteq_f32(vx89AB);
    float32x4_t vrsqrtxCDEF = vrsqrteq_f32(vxCDEF);
    float32x4_t vrsqrtxGHIJ = vrsqrteq_f32(vxGHIJ);
    float32x4_t vrsqrtxKLMN = vrsqrteq_f32(vxKLMN);
    float32x4_t vrsqrtxOPQR = vrsqrteq_f32(vxOPQR);
    float32x4_t vrsqrtxSTUV = vrsqrteq_f32(vxSTUV);
    float32x4_t vrsqrtxWXYZ = vrsqrteq_f32(vxWXYZ);
    float32x4_t vrsqrtx = vrsqrteq_f32(vx);

    const float32x4_t vrx0123 = vmulq_f32(vrsqrtx0123, vrsqrtx0123);
    const float32x4_t vrx4567 = vmulq_f32(vrsqrtx4567, vrsqrtx4567);
    const float32x4_t vrx89AB = vmulq_f32(vrsqrtx89AB, vrsqrtx89AB);
    const float32x4_t vrxCDEF = vmulq_f32(vrsqrtxCDEF, vrsqrtxCDEF);
    const float32x4_t vrxGHIJ = vmulq_f32(vrsqrtxGHIJ, vrsqrtxGHIJ);
    const float32x4_t vrxKLMN = vmulq_f32(vrsqrtxKLMN, vrsqrtxKLMN);
    const float32x4_t vrxOPQR = vmulq_f32(vrsqrtxOPQR, vrsqrtxOPQR);
    const float32x4_t vrxSTUV = vmulq_f32(vrsqrtxSTUV, vrsqrtxSTUV);
    const float32x4_t vrxWXYZ = vmulq_f32(vrsqrtxWXYZ, vrsqrtxWXYZ);
    const float32x4_t vrx = vmulq_f32(vrsqrtx, vrsqrtx);

    const float32x4_t vcorrection0123 = vrsqrtsq_f32(vx0123, vrx0123);
    const float32x4_t vcorrection4567 = vrsqrtsq_f32(vx4567, vrx4567);
    const float32x4_t vcorrection89AB = vrsqrtsq_f32(vx89AB, vrx89AB);
    const float32x4_t vcorrectionCDEF = vrsqrtsq_f32(vxCDEF, vrxCDEF);
    const float32x4_t vcorrectionGHIJ = vrsqrtsq_f32(vxGHIJ, vrxGHIJ);
    const float32x4_t vcorrectionKLMN = vrsqrtsq_f32(vxKLMN, vrxKLMN);
    const float32x4_t vcorrectionOPQR = vrsqrtsq_f32(vxOPQR, vrxOPQR);
    const float32x4_t vcorrectionSTUV = vrsqrtsq_f32(vxSTUV, vrxSTUV);
    const float32x4_t vcorrectionWXYZ = vrsqrtsq_f32(vxWXYZ, vrxWXYZ);
    const float32x4_t vcorrection = vrsqrtsq_f32(vx, vrx);

    vrsqrtx0123 = vmulq_f32(vrsqrtx0123, vcorrection0123);
    vrsqrtx4567 = vmulq_f32(vrsqrtx4567, vcorrection4567);
    vrsqrtx89AB = vmulq_f32(vrsqrtx89AB, vcorrection89AB);
    vrsqrtxCDEF = vmulq_f32(vrsqrtxCDEF, vcorrectionCDEF);
    vrsqrtxGHIJ = vmulq_f32(vrsqrtxGHIJ, vcorrectionGHIJ);
    vrsqrtxKLMN = vmulq_f32(vrsqrtxKLMN, vcorrectionKLMN);
    vrsqrtxOPQR = vmulq_f32(vrsqrtxOPQR, vcorrectionOPQR);
    vrsqrtxSTUV = vmulq_f32(vrsqrtxSTUV, vcorrectionSTUV);
    vrsqrtxWXYZ = vmulq_f32(vrsqrtxWXYZ, vcorrectionWXYZ);
    vrsqrtx = vmulq_f32(vrsqrtx, vcorrection);

    float32x4_t vsqrtx0123 = vmulq_f32(vrsqrtx0123, vx0123);
    float32x4_t vhalfrsqrtx0123 = vmulq_f32(vrsqrtx0123, vhalf);
    float32x4_t vsqrtx4567 = vmulq_f32(vrsqrtx4567, vx4567);
    float32x4_t vhalfrsqrtx4567 = vmulq_f32(vrsqrtx4567, vhalf);
    float32x4_t vsqrtx89AB = vmulq_f32(vrsqrtx89AB, vx89AB);
    float32x4_t vhalfrsqrtx89AB = vmulq_f32(vrsqrtx89AB, vhalf);
    float32x4_t vsqrtxCDEF = vmulq_f32(vrsqrtxCDEF, vxCDEF);
    float32x4_t vhalfrsqrtxCDEF = vmulq_f32(vrsqrtxCDEF, vhalf);
    float32x4_t vsqrtxGHIJ = vmulq_f32(vrsqrtxGHIJ, vxGHIJ);
    float32x4_t vhalfrsqrtxGHIJ = vmulq_f32(vrsqrtxGHIJ, vhalf);
    float32x4_t vsqrtxKLMN = vmulq_f32(vrsqrtxKLMN, vxKLMN);
    float32x4_t vhalfrsqrtxKLMN = vmulq_f32(vrsqrtxKLMN, vhalf);
    float32x4_t vsqrtxOPQR = vmulq_f32(vrsqrtxOPQR, vxOPQR);
    float32x4_t vhalfrsqrtxOPQR = vmulq_f32(vrsqrtxOPQR, vhalf);
    float32x4_t vsqrtxSTUV = vmulq_f32(vrsqrtxSTUV, vxSTUV);
    float32x4_t vhalfrsqrtxSTUV = vmulq_f32(vrsqrtxSTUV, vhalf);
    float32x4_t vsqrtxWXYZ = vmulq_f32(vrsqrtxWXYZ, vxWXYZ);
    float32x4_t vhalfrsqrtxWXYZ = vmulq_f32(vrsqrtxWXYZ, vhalf);
    float32x4_t vsqrtx = vmulq_f32(vrsqrtx, vx);
    float32x4_t vhalfrsqrtx = vmulq_f32(vrsqrtx, vhalf);

    const float32x4_t vresidual0123 = vfmsq_f32(vhalf, vsqrtx0123, vhalfrsqrtx0123);
    const float32x4_t vresidual4567 = vfmsq_f32(vhalf, vsqrtx4567, vhalfrsqrtx4567);
    const float32x4_t vresidual89AB = vfmsq_f32(vhalf, vsqrtx89AB, vhalfrsqrtx89AB);
    const float32x4_t vresidualCDEF = vfmsq_f32(vhalf, vsqrtxCDEF, vhalfrsqrtxCDEF);
    const float32x4_t vresidualGHIJ = vfmsq_f32(vhalf, vsqrtxGHIJ, vhalfrsqrtxGHIJ);
    const float32x4_t vresidualKLMN = vfmsq_f32(vhalf, vsqrtxKLMN, vhalfrsqrtxKLMN);
    const float32x4_t vresidualOPQR = vfmsq_f32(vhalf, vsqrtxOPQR, vhalfrsqrtxOPQR);
    const float32x4_t vresidualSTUV = vfmsq_f32(vhalf, vsqrtxSTUV, vhalfrsqrtxSTUV);
    const float32x4_t vresidualWXYZ = vfmsq_f32(vhalf, vsqrtxWXYZ, vhalfrsqrtxWXYZ);
    const float32x4_t vresidual = vfmsq_f32(vhalf, vsqrtx, vhalfrsqrtx);

    vhalfrsqrtx0123 = vfmaq_f32(vhalfrsqrtx0123, vresidual0123, vhalfrsqrtx0123);
    vsqrtx0123 = vfmaq_f32(vsqrtx0123, vresidual0123, vsqrtx0123);
    vhalfrsqrtx4567 = vfmaq_f32(vhalfrsqrtx4567, vresidual4567, vhalfrsqrtx4567);
    vsqrtx4567 = vfmaq_f32(vsqrtx4567, vresidual4567, vsqrtx4567);
    vhalfrsqrtx89AB = vfmaq_f32(vhalfrsqrtx89AB, vresidual89AB, vhalfrsqrtx89AB);
    vsqrtx89AB = vfmaq_f32(vsqrtx89AB, vresidual89AB, vsqrtx89AB);
    vhalfrsqrtxCDEF = vfmaq_f32(vhalfrsqrtxCDEF, vresidualCDEF, vhalfrsqrtxCDEF);
    vsqrtxCDEF = vfmaq_f32(vsqrtxCDEF, vresidualCDEF, vsqrtxCDEF);
    vhalfrsqrtxGHIJ = vfmaq_f32(vhalfrsqrtxGHIJ, vresidualGHIJ, vhalfrsqrtxGHIJ);
    vsqrtxGHIJ = vfmaq_f32(vsqrtxGHIJ, vresidualGHIJ, vsqrtxGHIJ);
    vhalfrsqrtxKLMN = vfmaq_f32(vhalfrsqrtxKLMN, vresidualKLMN, vhalfrsqrtxKLMN);
    vsqrtxKLMN = vfmaq_f32(vsqrtxKLMN, vresidualKLMN, vsqrtxKLMN);
    vhalfrsqrtxOPQR = vfmaq_f32(vhalfrsqrtxOPQR, vresidualOPQR, vhalfrsqrtxOPQR);
    vsqrtxOPQR = vfmaq_f32(vsqrtxOPQR, vresidualOPQR, vsqrtxOPQR);
    vhalfrsqrtxSTUV = vfmaq_f32(vhalfrsqrtxSTUV, vresidualSTUV, vhalfrsqrtxSTUV);
    vsqrtxSTUV = vfmaq_f32(vsqrtxSTUV, vresidualSTUV, vsqrtxSTUV);
    vhalfrsqrtxWXYZ = vfmaq_f32(vhalfrsqrtxWXYZ, vresidualWXYZ, vhalfrsqrtxWXYZ);
    vsqrtxWXYZ = vfmaq_f32(vsqrtxWXYZ, vresidualWXYZ, vsqrtxWXYZ);
    vhalfrsqrtx = vfmaq_f32(vhalfrsqrtx, vresidual, vhalfrsqrtx);
    vsqrtx = vfmaq_f32(vsqrtx, vresidual, vsqrtx);

    const float32x4_t vadjustment0123 = vfmsq_f32(vx0123, vsqrtx0123, vsqrtx0123);
    const float32x4_t vadjustment4567 = vfmsq_f32(vx4567, vsqrtx4567, vsqrtx4567);
    const float32x4_t vadjustment89AB = vfmsq_f32(vx89AB, vsqrtx89AB, vsqrtx89AB);
    const float32x4_t vadjustmentCDEF = vfmsq_f32(vxCDEF, vsqrtxCDEF, vsqrtxCDEF);
    const float32x4_t vadjustmentGHIJ = vfmsq_f32(vxGHIJ, vsqrtxGHIJ, vsqrtxGHIJ);
    const float32x4_t vadjustmentKLMN = vfmsq_f32(vxKLMN, vsqrtxKLMN, vsqrtxKLMN);
    const float32x4_t vadjustmentOPQR = vfmsq_f32(vxOPQR, vsqrtxOPQR, vsqrtxOPQR);
    const float32x4_t vadjustmentSTUV = vfmsq_f32(vxSTUV, vsqrtxSTUV, vsqrtxSTUV);
    const float32x4_t vadjustmentWXYZ = vfmsq_f32(vxWXYZ, vsqrtxWXYZ, vsqrtxWXYZ);
    const float32x4_t vadjustment = vfmsq_f32(vx, vsqrtx, vsqrtx);

    const float32x4_t vy0123 = vfmaq_f32(vsqrtx0123, vhalfrsqrtx0123, vadjustment0123);
    const float32x4_t vy4567 = vfmaq_f32(vsqrtx4567, vhalfrsqrtx4567, vadjustment4567);
    const float32x4_t vy89AB = vfmaq_f32(vsqrtx89AB, vhalfrsqrtx89AB, vadjustment89AB);
    const float32x4_t vyCDEF = vfmaq_f32(vsqrtxCDEF, vhalfrsqrtxCDEF, vadjustmentCDEF);
    const float32x4_t vyGHIJ = vfmaq_f32(vsqrtxGHIJ, vhalfrsqrtxGHIJ, vadjustmentGHIJ);
    const float32x4_t vyKLMN = vfmaq_f32(vsqrtxKLMN, vhalfrsqrtxKLMN, vadjustmentKLMN);
    const float32x4_t vyOPQR = vfmaq_f32(vsqrtxOPQR, vhalfrsqrtxOPQR, vadjustmentOPQR);
    const float32x4_t vySTUV = vfmaq_f32(vsqrtxSTUV, vhalfrsqrtxSTUV, vadjustmentSTUV);
    const float32x4_t vyWXYZ = vfmaq_f32(vsqrtxWXYZ, vhalfrsqrtxWXYZ, vadjustmentWXYZ);
    const float32x4_t vy = vfmaq_f32(vsqrtx, vhalfrsqrtx, vadjustment);

    vst1q_f32(output, vy0123); output += 4;
    vst1q_f32(output, vy4567); output += 4;
    vst1q_f32(output, vy89AB); output += 4;
    vst1q_f32(output, vyCDEF); output += 4;
    vst1q_f32(output, vyGHIJ); output += 4;
    vst1q_f32(output, vyKLMN); output += 4;
    vst1q_f32(output, vyOPQR); output += 4;
    vst1q_f32(output, vySTUV); output += 4;
    vst1q_f32(output, vyWXYZ); output += 4;
    vst1q_f32(output, vy); output += 4;
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
