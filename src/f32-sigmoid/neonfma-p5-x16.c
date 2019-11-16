// Auto-generated file. Do not edit!
//   Template: src/f32-sigmoid/neonfma-p5.c.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include <xnnpack/common.h>
#include <xnnpack/vunop.h>


void xnn_f32_sigmoid_ukernel__neonfma_p5_x16(
    size_t n,
    const float* x,
    float* y,
    const void* params)
{
  const float32x4_t vhalf = vmovq_n_f32(0.5f);

  const float kSigmoidAlpha1 = 2.48287947061529e-01;
  const float kSigmoidAlpha3 = 8.51377133304701e-03;
  const float kSigmoidAlpha5 = 6.08574864600143e-05;
  const float kSigmoidAlpha7 = 1.15627324459942e-07;
  const float kSigmoidAlpha9 = 4.37031012579801e-11;

  // The monomial coefficients of the denominator polynomial (even).
  const float kSigmoidBeta0 = 9.93151921023180e-01;
  const float kSigmoidBeta2 = 1.16817656904453e-01;
  const float kSigmoidBeta4 = 1.70198817374094e-03;
  const float kSigmoidBeta6 = 6.29106785017040e-06;
  const float kSigmoidBeta8 = 5.76102136993427e-09;
  const float kSigmoidBeta10 = 6.10247389755681e-13;

  // The monomial coefficients of the numerator polynomial (odd).
  const float32x4_t valpha_1 = vdupq_n_f32(kSigmoidAlpha1);
  const float32x4_t valpha_3 = vdupq_n_f32(kSigmoidAlpha3);
  const float32x4_t valpha_5 = vdupq_n_f32(kSigmoidAlpha5);
  const float32x4_t valpha_7 = vdupq_n_f32(kSigmoidAlpha7);
  const float32x4_t valpha_9 = vdupq_n_f32(kSigmoidAlpha9);

  // The monomial coefficients of the denominator polynomial (even).
  const float32x4_t vbeta_0 = vdupq_n_f32(kSigmoidBeta0);
  const float32x4_t vbeta_2 = vdupq_n_f32(kSigmoidBeta2);
  const float32x4_t vbeta_4 = vdupq_n_f32(kSigmoidBeta4);
  const float32x4_t vbeta_6 = vdupq_n_f32(kSigmoidBeta6);
  const float32x4_t vbeta_8 = vdupq_n_f32(kSigmoidBeta8);
  const float32x4_t vbeta_10 = vdupq_n_f32(kSigmoidBeta10);

  const float32x4_t vsigmoid_maxinput = vdupq_n_f32(18.f);
  const float32x4_t vsigmoid_mininput = vdupq_n_f32(-18.f);

  for (; n >= 16 * sizeof(float); n -= 16 * sizeof(float)) {
    float32x4_t vx0123 = vld1q_f32(x); x += 4;
    float32x4_t vx4567 = vld1q_f32(x); x += 4;
    float32x4_t vx89AB = vld1q_f32(x); x += 4;
    float32x4_t vxCDEF = vld1q_f32(x); x += 4;

    vx0123 = vminq_f32(vx0123, vsigmoid_maxinput);
    vx0123 = vmaxq_f32(vx0123, vsigmoid_mininput);
    vx4567 = vminq_f32(vx4567, vsigmoid_maxinput);
    vx4567 = vmaxq_f32(vx4567, vsigmoid_mininput);
    vx89AB = vminq_f32(vx89AB, vsigmoid_maxinput);
    vx89AB = vmaxq_f32(vx89AB, vsigmoid_mininput);
    vxCDEF = vminq_f32(vxCDEF, vsigmoid_maxinput);
    vxCDEF = vmaxq_f32(vxCDEF, vsigmoid_mininput);

    const float32x4_t vx0123_sq = vmulq_f32(vx0123, vx0123);
    const float32x4_t vx4567_sq = vmulq_f32(vx4567, vx4567);
    const float32x4_t vx89AB_sq = vmulq_f32(vx89AB, vx89AB);
    const float32x4_t vxCDEF_sq = vmulq_f32(vxCDEF, vxCDEF);

    // Evaluate numerator polynomial
    float32x4_t vnum0123 = vmlaq_f32(valpha_7, vx0123_sq, valpha_9);
    float32x4_t vnum4567 = vmlaq_f32(valpha_7, vx4567_sq, valpha_9);
    float32x4_t vnum89AB = vmlaq_f32(valpha_7, vx89AB_sq, valpha_9);
    float32x4_t vnumCDEF = vmlaq_f32(valpha_7, vxCDEF_sq, valpha_9);

    vnum0123 = vmlaq_f32(valpha_5, vx0123_sq, vnum0123);
    vnum4567 = vmlaq_f32(valpha_5, vx4567_sq, vnum4567);
    vnum89AB = vmlaq_f32(valpha_5, vx89AB_sq, vnum89AB);
    vnumCDEF = vmlaq_f32(valpha_5, vxCDEF_sq, vnumCDEF);

    vnum0123 = vmlaq_f32(valpha_3, vx0123_sq, vnum0123);
    vnum4567 = vmlaq_f32(valpha_3, vx4567_sq, vnum4567);
    vnum89AB = vmlaq_f32(valpha_3, vx89AB_sq, vnum89AB);
    vnumCDEF = vmlaq_f32(valpha_3, vxCDEF_sq, vnumCDEF);

    vnum0123 = vmlaq_f32(valpha_1, vx0123_sq, vnum0123);
    vnum4567 = vmlaq_f32(valpha_1, vx4567_sq, vnum4567);
    vnum89AB = vmlaq_f32(valpha_1, vx89AB_sq, vnum89AB);
    vnumCDEF = vmlaq_f32(valpha_1, vxCDEF_sq, vnumCDEF);

    vnum0123 = vmulq_f32(vx0123, vnum0123);
    vnum4567 = vmulq_f32(vx4567, vnum4567);
    vnum89AB = vmulq_f32(vx89AB, vnum89AB);
    vnumCDEF = vmulq_f32(vxCDEF, vnumCDEF);

    // Evaluate denominator polynomial

    float32x4_t vdenom0123 = vmlaq_f32(vbeta_8, vx0123_sq, vbeta_10);
    float32x4_t vdenom4567 = vmlaq_f32(vbeta_8, vx4567_sq, vbeta_10);
    float32x4_t vdenom89AB = vmlaq_f32(vbeta_8, vx89AB_sq, vbeta_10);
    float32x4_t vdenomCDEF = vmlaq_f32(vbeta_8, vxCDEF_sq, vbeta_10);

    vdenom0123 = vmlaq_f32(vbeta_6, vx0123_sq, vdenom0123);
    vdenom4567 = vmlaq_f32(vbeta_6, vx4567_sq, vdenom4567);
    vdenom89AB = vmlaq_f32(vbeta_6, vx89AB_sq, vdenom89AB);
    vdenomCDEF = vmlaq_f32(vbeta_6, vxCDEF_sq, vdenomCDEF);

    vdenom0123 = vmlaq_f32(vbeta_4, vx0123_sq, vdenom0123);
    vdenom4567 = vmlaq_f32(vbeta_4, vx4567_sq, vdenom4567);
    vdenom89AB = vmlaq_f32(vbeta_4, vx89AB_sq, vdenom89AB);
    vdenomCDEF = vmlaq_f32(vbeta_4, vxCDEF_sq, vdenomCDEF);

    vdenom0123 = vmlaq_f32(vbeta_2, vx0123_sq, vdenom0123);
    vdenom4567 = vmlaq_f32(vbeta_2, vx4567_sq, vdenom4567);
    vdenom89AB = vmlaq_f32(vbeta_2, vx89AB_sq, vdenom89AB);
    vdenomCDEF = vmlaq_f32(vbeta_2, vxCDEF_sq, vdenomCDEF);

    vdenom0123 = vmlaq_f32(vbeta_0, vx0123_sq, vdenom0123);
    vdenom4567 = vmlaq_f32(vbeta_0, vx4567_sq, vdenom4567);
    vdenom89AB = vmlaq_f32(vbeta_0, vx89AB_sq, vdenom89AB);
    vdenomCDEF = vmlaq_f32(vbeta_0, vxCDEF_sq, vdenomCDEF);

    // Do division, one NR iteration

    float32x4_t vrecp0123 = vrecpeq_f32(vdenom0123);
    float32x4_t vrecp4567 = vrecpeq_f32(vdenom4567);
    float32x4_t vrecp89AB = vrecpeq_f32(vdenom89AB);
    float32x4_t vrecpCDEF = vrecpeq_f32(vdenomCDEF);

    vrecp0123 = vmulq_f32(vrecp0123, vrecpsq_f32(vrecp0123, vdenom0123));
    vrecp4567 = vmulq_f32(vrecp4567, vrecpsq_f32(vrecp4567, vdenom4567));
    vrecp89AB = vmulq_f32(vrecp89AB, vrecpsq_f32(vrecp89AB, vdenom89AB));
    vrecpCDEF = vmulq_f32(vrecpCDEF, vrecpsq_f32(vrecpCDEF, vdenomCDEF));

    const float32x4_t vsigmoid0123 = vmlaq_f32(vhalf, vnum0123, vrecp0123);
    const float32x4_t vsigmoid4567 = vmlaq_f32(vhalf, vnum4567, vrecp4567);
    const float32x4_t vsigmoid89AB = vmlaq_f32(vhalf, vnum89AB, vrecp89AB);
    const float32x4_t vsigmoidCDEF = vmlaq_f32(vhalf, vnumCDEF, vrecpCDEF);

    vst1q_f32(y, vsigmoid0123); y += 4;
    vst1q_f32(y, vsigmoid4567); y += 4;
    vst1q_f32(y, vsigmoid89AB); y += 4;
    vst1q_f32(y, vsigmoidCDEF); y += 4;
  }
  for (; n >= 4 * sizeof(float); n -= 4 * sizeof(float)) {
    float32x4_t vx0123 = vld1q_f32(x); x += 4;

    vx0123 = vminq_f32(vx0123, vsigmoid_maxinput);
    vx0123 = vmaxq_f32(vx0123, vsigmoid_mininput);

    const float32x4_t vx0123_sq = vmulq_f32(vx0123, vx0123);

    // Evaluate numerator polynomial
    float32x4_t vnum0123 = vmlaq_f32(valpha_7, vx0123_sq, valpha_9);

    vnum0123 = vmlaq_f32(valpha_5, vx0123_sq, vnum0123);
    vnum0123 = vmlaq_f32(valpha_3, vx0123_sq, vnum0123);
    vnum0123 = vmlaq_f32(valpha_1, vx0123_sq, vnum0123);
    vnum0123 = vmulq_f32(vx0123, vnum0123);

    // Evaluate denominator polynomial

    float32x4_t vdenom0123 = vmlaq_f32(vbeta_8, vx0123_sq, vbeta_10);
    vdenom0123 = vmlaq_f32(vbeta_6, vx0123_sq, vdenom0123);
    vdenom0123 = vmlaq_f32(vbeta_4, vx0123_sq, vdenom0123);
    vdenom0123 = vmlaq_f32(vbeta_2, vx0123_sq, vdenom0123);
    vdenom0123 = vmlaq_f32(vbeta_0, vx0123_sq, vdenom0123);

    // Do division, one NR iteration

    float32x4_t vrecp0123 = vrecpeq_f32(vdenom0123);
    vrecp0123 = vmulq_f32(vrecp0123, vrecpsq_f32(vrecp0123, vdenom0123));

    const float32x4_t vsigmoid0123 = vmlaq_f32(vhalf, vnum0123, vrecp0123);

    vst1q_f32(y, vsigmoid0123); y += 4;
  }
  if XNN_UNLIKELY(n != 0) {
    float32x4_t vx0123 = vld1q_f32(x);

    vx0123 = vminq_f32(vx0123, vsigmoid_maxinput);
    vx0123 = vmaxq_f32(vx0123, vsigmoid_mininput);

    const float32x4_t vx0123_sq = vmulq_f32(vx0123, vx0123);

    // Evaluate numerator polynomial
    float32x4_t vnum0123 = vmlaq_f32(valpha_7, vx0123_sq, valpha_9);
    vnum0123 = vmlaq_f32(valpha_5, vx0123_sq, vnum0123);
    vnum0123 = vmlaq_f32(valpha_3, vx0123_sq, vnum0123);
    vnum0123 = vmlaq_f32(valpha_1, vx0123_sq, vnum0123);
    vnum0123 = vmulq_f32(vx0123, vnum0123);

    // Evaluate denominator polynomial

    float32x4_t vdenom0123 = vmlaq_f32(vbeta_8, vx0123_sq, vbeta_10);
    vdenom0123 = vmlaq_f32(vbeta_6, vx0123_sq, vdenom0123);
    vdenom0123 = vmlaq_f32(vbeta_4, vx0123_sq, vdenom0123);
    vdenom0123 = vmlaq_f32(vbeta_2, vx0123_sq, vdenom0123);
    vdenom0123 = vmlaq_f32(vbeta_0, vx0123_sq, vdenom0123);

    // Do division, one NR iteration

    float32x4_t vrecp0123 = vrecpeq_f32(vdenom0123);
    vrecp0123 = vmulq_f32(vrecp0123, vrecpsq_f32(vrecp0123, vdenom0123));

    const float32x4_t vsigmoid0123 = vmlaq_f32(vhalf, vnum0123, vrecp0123);

    float32x2_t vf01 = vget_low_f32(vsigmoid0123);
    if (n & (2 * sizeof(float))) {
      vst1_f32(y, vf01); y += 2;
      vf01 = vget_high_f32(vsigmoid0123);
    }
    if (n & (1 * sizeof(float))) {
      vst1_lane_f32(y, vf01, 0);
    }
  }
}
