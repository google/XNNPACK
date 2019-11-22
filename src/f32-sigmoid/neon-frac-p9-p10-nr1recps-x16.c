// Auto-generated file. Do not edit!
//   Template: src/f32-sigmoid/neon-frac-p9-p10-nr1recps.c.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include <xnnpack/common.h>
#include <xnnpack/vunary.h>


void xnn_f32_sigmoid_ukernel__neon_frac_p9_p10_nr1recps_x16(
    size_t n,
    const float* x,
    float* y,
    const void* params)
{
  assert(n % sizeof(float) == 0);

  const float32x4_t vhalf = vmovq_n_f32(0.5f);

  // The coefficients of the numerator polynomial (odd).
  const float32x4_t valpha_1 = vmovq_n_f32(2.48287947061529e-01);
  const float32x4_t valpha_3 = vmovq_n_f32(8.51377133304701e-03);
  const float32x4_t valpha_5 = vmovq_n_f32(6.08574864600143e-05);
  const float32x4_t valpha_7 = vmovq_n_f32(1.15627324459942e-07);
  const float32x4_t valpha_9 = vmovq_n_f32(4.37031012579801e-11);

  // The coefficients of the denominator polynomial (even).
  const float32x4_t vbeta_0 =  vmovq_n_f32(9.93151921023180e-01);
  const float32x4_t vbeta_2 =  vmovq_n_f32(1.16817656904453e-01);
  const float32x4_t vbeta_4 =  vmovq_n_f32(1.70198817374094e-03);
  const float32x4_t vbeta_6 =  vmovq_n_f32(6.29106785017040e-06);
  const float32x4_t vbeta_8 =  vmovq_n_f32(5.76102136993427e-09);
  const float32x4_t vbeta_10 = vmovq_n_f32(6.10247389755681e-13);

  // Sigmoid ~saturates outside of this range anyway.
  const float32x4_t vsigmoid_maxinput = vdupq_n_f32(18.f);
  const float32x4_t vsigmoid_mininput = vdupq_n_f32(-18.f);

  for (; n >= 16 * sizeof(float); n -= 16 * sizeof(float)) {
    float32x4_t vn0123 = vld1q_f32(x); x += 4;
    float32x4_t vn4567 = vld1q_f32(x); x += 4;
    float32x4_t vn89AB = vld1q_f32(x); x += 4;
    float32x4_t vnCDEF = vld1q_f32(x); x += 4;

    // restrict range to avoid overflow, output saturates outside this anyway
    vn0123 = vminq_f32(vn0123, vsigmoid_maxinput);
    vn0123 = vmaxq_f32(vn0123, vsigmoid_mininput);
    vn4567 = vminq_f32(vn4567, vsigmoid_maxinput);
    vn4567 = vmaxq_f32(vn4567, vsigmoid_mininput);
    vn89AB = vminq_f32(vn89AB, vsigmoid_maxinput);
    vn89AB = vmaxq_f32(vn89AB, vsigmoid_mininput);
    vnCDEF = vminq_f32(vnCDEF, vsigmoid_maxinput);
    vnCDEF = vmaxq_f32(vnCDEF, vsigmoid_mininput);

    // square the input
    const float32x4_t vn0123_sq = vmulq_f32(vn0123, vn0123);
    const float32x4_t vn4567_sq = vmulq_f32(vn4567, vn4567);
    const float32x4_t vn89AB_sq = vmulq_f32(vn89AB, vn89AB);
    const float32x4_t vnCDEF_sq = vmulq_f32(vnCDEF, vnCDEF);

    // Evaluate numerator polynomial
    float32x4_t vnum0123 = vmlaq_f32(valpha_7, vn0123_sq, valpha_9);
    float32x4_t vnum4567 = vmlaq_f32(valpha_7, vn4567_sq, valpha_9);
    float32x4_t vnum89AB = vmlaq_f32(valpha_7, vn89AB_sq, valpha_9);
    float32x4_t vnumCDEF = vmlaq_f32(valpha_7, vnCDEF_sq, valpha_9);

    vnum0123 = vmlaq_f32(valpha_5, vn0123_sq, vnum0123);
    vnum4567 = vmlaq_f32(valpha_5, vn4567_sq, vnum4567);
    vnum89AB = vmlaq_f32(valpha_5, vn89AB_sq, vnum89AB);
    vnumCDEF = vmlaq_f32(valpha_5, vnCDEF_sq, vnumCDEF);

    vnum0123 = vmlaq_f32(valpha_3, vn0123_sq, vnum0123);
    vnum4567 = vmlaq_f32(valpha_3, vn4567_sq, vnum4567);
    vnum89AB = vmlaq_f32(valpha_3, vn89AB_sq, vnum89AB);
    vnumCDEF = vmlaq_f32(valpha_3, vnCDEF_sq, vnumCDEF);

    vnum0123 = vmlaq_f32(valpha_1, vn0123_sq, vnum0123);
    vnum4567 = vmlaq_f32(valpha_1, vn4567_sq, vnum4567);
    vnum89AB = vmlaq_f32(valpha_1, vn89AB_sq, vnum89AB);
    vnumCDEF = vmlaq_f32(valpha_1, vnCDEF_sq, vnumCDEF);

    vnum0123 = vmulq_f32(vn0123, vnum0123);
    vnum4567 = vmulq_f32(vn4567, vnum4567);
    vnum89AB = vmulq_f32(vn89AB, vnum89AB);
    vnumCDEF = vmulq_f32(vnCDEF, vnumCDEF);

    // Evaluate denominator polynomial
    float32x4_t vdenom0123 = vmlaq_f32(vbeta_8, vn0123_sq, vbeta_10);
    float32x4_t vdenom4567 = vmlaq_f32(vbeta_8, vn4567_sq, vbeta_10);
    float32x4_t vdenom89AB = vmlaq_f32(vbeta_8, vn89AB_sq, vbeta_10);
    float32x4_t vdenomCDEF = vmlaq_f32(vbeta_8, vnCDEF_sq, vbeta_10);

    vdenom0123 = vmlaq_f32(vbeta_6, vn0123_sq, vdenom0123);
    vdenom4567 = vmlaq_f32(vbeta_6, vn4567_sq, vdenom4567);
    vdenom89AB = vmlaq_f32(vbeta_6, vn89AB_sq, vdenom89AB);
    vdenomCDEF = vmlaq_f32(vbeta_6, vnCDEF_sq, vdenomCDEF);

    vdenom0123 = vmlaq_f32(vbeta_4, vn0123_sq, vdenom0123);
    vdenom4567 = vmlaq_f32(vbeta_4, vn4567_sq, vdenom4567);
    vdenom89AB = vmlaq_f32(vbeta_4, vn89AB_sq, vdenom89AB);
    vdenomCDEF = vmlaq_f32(vbeta_4, vnCDEF_sq, vdenomCDEF);

    vdenom0123 = vmlaq_f32(vbeta_2, vn0123_sq, vdenom0123);
    vdenom4567 = vmlaq_f32(vbeta_2, vn4567_sq, vdenom4567);
    vdenom89AB = vmlaq_f32(vbeta_2, vn89AB_sq, vdenom89AB);
    vdenomCDEF = vmlaq_f32(vbeta_2, vnCDEF_sq, vdenomCDEF);

    vdenom0123 = vmlaq_f32(vbeta_0, vn0123_sq, vdenom0123);
    vdenom4567 = vmlaq_f32(vbeta_0, vn4567_sq, vdenom4567);
    vdenom89AB = vmlaq_f32(vbeta_0, vn89AB_sq, vdenom89AB);
    vdenomCDEF = vmlaq_f32(vbeta_0, vnCDEF_sq, vdenomCDEF);

    // Do division 1. / denom
    float32x4_t vrecp0123 = vrecpeq_f32(vdenom0123);
    float32x4_t vrecp4567 = vrecpeq_f32(vdenom4567);
    float32x4_t vrecp89AB = vrecpeq_f32(vdenom89AB);
    float32x4_t vrecpCDEF = vrecpeq_f32(vdenomCDEF);

    // One NR iteration
    vrecp0123 = vmulq_f32(vrecp0123, vrecpsq_f32(vrecp0123, vdenom0123));
    vrecp4567 = vmulq_f32(vrecp4567, vrecpsq_f32(vrecp4567, vdenom4567));
    vrecp89AB = vmulq_f32(vrecp89AB, vrecpsq_f32(vrecp89AB, vdenom89AB));
    vrecpCDEF = vmulq_f32(vrecpCDEF, vrecpsq_f32(vrecpCDEF, vdenomCDEF));


    // .5 + num * (1. / denom)
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
    float32x4_t vn0123 = vld1q_f32(x); x += 4;

    vn0123 = vminq_f32(vn0123, vsigmoid_maxinput);
    vn0123 = vmaxq_f32(vn0123, vsigmoid_mininput);

    const float32x4_t vn0123_sq = vmulq_f32(vn0123, vn0123);

    // Evaluate numerator polynomial
    float32x4_t vnum0123 = vmlaq_f32(valpha_7, vn0123_sq, valpha_9);

    vnum0123 = vmlaq_f32(valpha_5, vn0123_sq, vnum0123);
    vnum0123 = vmlaq_f32(valpha_3, vn0123_sq, vnum0123);
    vnum0123 = vmlaq_f32(valpha_1, vn0123_sq, vnum0123);
    vnum0123 = vmulq_f32(vn0123, vnum0123);

    // Evaluate denominator polynomial

    float32x4_t vdenom0123 = vmlaq_f32(vbeta_8, vn0123_sq, vbeta_10);
    vdenom0123 = vmlaq_f32(vbeta_6, vn0123_sq, vdenom0123);
    vdenom0123 = vmlaq_f32(vbeta_4, vn0123_sq, vdenom0123);
    vdenom0123 = vmlaq_f32(vbeta_2, vn0123_sq, vdenom0123);
    vdenom0123 = vmlaq_f32(vbeta_0, vn0123_sq, vdenom0123);

    // Do division, one NR iteration

    float32x4_t vrecp0123 = vrecpeq_f32(vdenom0123);
    vrecp0123 = vmulq_f32(vrecp0123, vrecpsq_f32(vrecp0123, vdenom0123));

    const float32x4_t vsigmoid0123 = vmlaq_f32(vhalf, vnum0123, vrecp0123);

    vst1q_f32(y, vsigmoid0123); y += 4;
  }
  if XNN_UNLIKELY(n != 0) {
    float32x4_t vn0123 = vld1q_f32(x);

    vn0123 = vminq_f32(vn0123, vsigmoid_maxinput);
    vn0123 = vmaxq_f32(vn0123, vsigmoid_mininput);

    const float32x4_t vn0123_sq = vmulq_f32(vn0123, vn0123);

    // Evaluate numerator polynomial
    float32x4_t vnum0123 = vmlaq_f32(valpha_7, vn0123_sq, valpha_9);

    vnum0123 = vmlaq_f32(valpha_5, vn0123_sq, vnum0123);
    vnum0123 = vmlaq_f32(valpha_3, vn0123_sq, vnum0123);
    vnum0123 = vmlaq_f32(valpha_1, vn0123_sq, vnum0123);
    vnum0123 = vmulq_f32(vn0123, vnum0123);

    // Evaluate denominator polynomial

    float32x4_t vdenom0123 = vmlaq_f32(vbeta_8, vn0123_sq, vbeta_10);
    vdenom0123 = vmlaq_f32(vbeta_6, vn0123_sq, vdenom0123);
    vdenom0123 = vmlaq_f32(vbeta_4, vn0123_sq, vdenom0123);
    vdenom0123 = vmlaq_f32(vbeta_2, vn0123_sq, vdenom0123);
    vdenom0123 = vmlaq_f32(vbeta_0, vn0123_sq, vdenom0123);

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
