// Auto-generated file. Do not edit!
//   Template: src/f32-vsqrt/neonfma-nr2fma1adj.c.in
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


void xnn_f32_vsqrt_ukernel__neonfma_nr2fma1adj_x32(
    size_t n,
    const float* x,
    float* y,
    const union xnn_f32_sqrt_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(n != 0);
  assert(n % sizeof(float) == 0);

  const float32x4_t vhalf = vmovq_n_f32(0.5f);
  for (; n >= 32 * sizeof(float); n -= 32 * sizeof(float)) {
    const float32x4_t vx0123 = vld1q_f32(x); x += 4;
    const float32x4_t vx4567 = vld1q_f32(x); x += 4;
    const float32x4_t vx89AB = vld1q_f32(x); x += 4;
    const float32x4_t vxCDEF = vld1q_f32(x); x += 4;
    const float32x4_t vxGHIJ = vld1q_f32(x); x += 4;
    const float32x4_t vxKLMN = vld1q_f32(x); x += 4;
    const float32x4_t vxOPQR = vld1q_f32(x); x += 4;
    const float32x4_t vxSTUV = vld1q_f32(x); x += 4;

    const float32x4_t vrsqrtx0123 = vrsqrteq_f32(vx0123);
    const float32x4_t vrsqrtx4567 = vrsqrteq_f32(vx4567);
    const float32x4_t vrsqrtx89AB = vrsqrteq_f32(vx89AB);
    const float32x4_t vrsqrtxCDEF = vrsqrteq_f32(vxCDEF);
    const float32x4_t vrsqrtxGHIJ = vrsqrteq_f32(vxGHIJ);
    const float32x4_t vrsqrtxKLMN = vrsqrteq_f32(vxKLMN);
    const float32x4_t vrsqrtxOPQR = vrsqrteq_f32(vxOPQR);
    const float32x4_t vrsqrtxSTUV = vrsqrteq_f32(vxSTUV);

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

    float32x4_t vresidual0123 = vfmsq_f32(vhalf, vsqrtx0123, vhalfrsqrtx0123);
    float32x4_t vresidual4567 = vfmsq_f32(vhalf, vsqrtx4567, vhalfrsqrtx4567);
    float32x4_t vresidual89AB = vfmsq_f32(vhalf, vsqrtx89AB, vhalfrsqrtx89AB);
    float32x4_t vresidualCDEF = vfmsq_f32(vhalf, vsqrtxCDEF, vhalfrsqrtxCDEF);
    float32x4_t vresidualGHIJ = vfmsq_f32(vhalf, vsqrtxGHIJ, vhalfrsqrtxGHIJ);
    float32x4_t vresidualKLMN = vfmsq_f32(vhalf, vsqrtxKLMN, vhalfrsqrtxKLMN);
    float32x4_t vresidualOPQR = vfmsq_f32(vhalf, vsqrtxOPQR, vhalfrsqrtxOPQR);
    float32x4_t vresidualSTUV = vfmsq_f32(vhalf, vsqrtxSTUV, vhalfrsqrtxSTUV);

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

    vresidual0123 = vfmsq_f32(vhalf, vsqrtx0123, vhalfrsqrtx0123);
    vresidual4567 = vfmsq_f32(vhalf, vsqrtx4567, vhalfrsqrtx4567);
    vresidual89AB = vfmsq_f32(vhalf, vsqrtx89AB, vhalfrsqrtx89AB);
    vresidualCDEF = vfmsq_f32(vhalf, vsqrtxCDEF, vhalfrsqrtxCDEF);
    vresidualGHIJ = vfmsq_f32(vhalf, vsqrtxGHIJ, vhalfrsqrtxGHIJ);
    vresidualKLMN = vfmsq_f32(vhalf, vsqrtxKLMN, vhalfrsqrtxKLMN);
    vresidualOPQR = vfmsq_f32(vhalf, vsqrtxOPQR, vhalfrsqrtxOPQR);
    vresidualSTUV = vfmsq_f32(vhalf, vsqrtxSTUV, vhalfrsqrtxSTUV);

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

    const float32x4_t vadjustment0123 = vfmsq_f32(vx0123, vsqrtx0123, vsqrtx0123);
    const float32x4_t vadjustment4567 = vfmsq_f32(vx4567, vsqrtx4567, vsqrtx4567);
    const float32x4_t vadjustment89AB = vfmsq_f32(vx89AB, vsqrtx89AB, vsqrtx89AB);
    const float32x4_t vadjustmentCDEF = vfmsq_f32(vxCDEF, vsqrtxCDEF, vsqrtxCDEF);
    const float32x4_t vadjustmentGHIJ = vfmsq_f32(vxGHIJ, vsqrtxGHIJ, vsqrtxGHIJ);
    const float32x4_t vadjustmentKLMN = vfmsq_f32(vxKLMN, vsqrtxKLMN, vsqrtxKLMN);
    const float32x4_t vadjustmentOPQR = vfmsq_f32(vxOPQR, vsqrtxOPQR, vsqrtxOPQR);
    const float32x4_t vadjustmentSTUV = vfmsq_f32(vxSTUV, vsqrtxSTUV, vsqrtxSTUV);

    const float32x4_t vy0123 = vfmaq_f32(vsqrtx0123, vhalfrsqrtx0123, vadjustment0123);
    const float32x4_t vy4567 = vfmaq_f32(vsqrtx4567, vhalfrsqrtx4567, vadjustment4567);
    const float32x4_t vy89AB = vfmaq_f32(vsqrtx89AB, vhalfrsqrtx89AB, vadjustment89AB);
    const float32x4_t vyCDEF = vfmaq_f32(vsqrtxCDEF, vhalfrsqrtxCDEF, vadjustmentCDEF);
    const float32x4_t vyGHIJ = vfmaq_f32(vsqrtxGHIJ, vhalfrsqrtxGHIJ, vadjustmentGHIJ);
    const float32x4_t vyKLMN = vfmaq_f32(vsqrtxKLMN, vhalfrsqrtxKLMN, vadjustmentKLMN);
    const float32x4_t vyOPQR = vfmaq_f32(vsqrtxOPQR, vhalfrsqrtxOPQR, vadjustmentOPQR);
    const float32x4_t vySTUV = vfmaq_f32(vsqrtxSTUV, vhalfrsqrtxSTUV, vadjustmentSTUV);

    vst1q_f32(y, vy0123); y += 4;
    vst1q_f32(y, vy4567); y += 4;
    vst1q_f32(y, vy89AB); y += 4;
    vst1q_f32(y, vyCDEF); y += 4;
    vst1q_f32(y, vyGHIJ); y += 4;
    vst1q_f32(y, vyKLMN); y += 4;
    vst1q_f32(y, vyOPQR); y += 4;
    vst1q_f32(y, vySTUV); y += 4;
  }
  for (; n >= 4 * sizeof(float); n -= 4 * sizeof(float)) {
    const float32x4_t vx = vld1q_f32(x); x += 4;
    const float32x4_t vrsqrtx = vrsqrteq_f32(vx);
    float32x4_t vsqrtx = vmulq_f32(vrsqrtx, vx);
    float32x4_t vhalfrsqrtx = vmulq_f32(vrsqrtx, vhalf);
    float32x4_t vresidual = vfmsq_f32(vhalf, vsqrtx, vhalfrsqrtx);
    vhalfrsqrtx = vfmaq_f32(vhalfrsqrtx, vresidual, vhalfrsqrtx);
    vsqrtx = vfmaq_f32(vsqrtx, vresidual, vsqrtx);
    vresidual = vfmsq_f32(vhalf, vsqrtx, vhalfrsqrtx);
    vhalfrsqrtx = vfmaq_f32(vhalfrsqrtx, vresidual, vhalfrsqrtx);
    vsqrtx = vfmaq_f32(vsqrtx, vresidual, vsqrtx);
    const float32x4_t vadjustment = vfmsq_f32(vx, vsqrtx, vsqrtx);
    const float32x4_t vy = vfmaq_f32(vsqrtx, vhalfrsqrtx, vadjustment);
    vst1q_f32(y, vy); y += 4;
  }
  if XNN_UNLIKELY(n != 0) {
    const float32x4_t vx = vld1q_f32(x);
    const float32x4_t vrsqrtx = vrsqrteq_f32(vx);
    float32x4_t vsqrtx = vmulq_f32(vrsqrtx, vx);
    float32x4_t vhalfrsqrtx = vmulq_f32(vrsqrtx, vhalf);
    float32x4_t vresidual = vfmsq_f32(vhalf, vsqrtx, vhalfrsqrtx);
    vhalfrsqrtx = vfmaq_f32(vhalfrsqrtx, vresidual, vhalfrsqrtx);
    vsqrtx = vfmaq_f32(vsqrtx, vresidual, vsqrtx);
    vresidual = vfmsq_f32(vhalf, vsqrtx, vhalfrsqrtx);
    vhalfrsqrtx = vfmaq_f32(vhalfrsqrtx, vresidual, vhalfrsqrtx);
    vsqrtx = vfmaq_f32(vsqrtx, vresidual, vsqrtx);
    const float32x4_t vadjustment = vfmsq_f32(vx, vsqrtx, vsqrtx);
    const float32x4_t vy = vfmaq_f32(vsqrtx, vhalfrsqrtx, vadjustment);

    float32x2_t vy_lo = vget_low_f32(vy);
    if (n & (2 * sizeof(float))) {
      vst1_f32(y, vy_lo); y += 2;
      vy_lo = vget_high_f32(vy);
    }
    if (n & (1 * sizeof(float))) {
      vst1_lane_f32(y, vy_lo, 0);
    }
  }
}
