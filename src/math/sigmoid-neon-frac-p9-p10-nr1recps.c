// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include <xnnpack/common.h>
#include <xnnpack/vunary.h>


void xnn_math_f32_sigmoid__neon_frac_p9_p10_nr1recps(
    size_t n,
    const float* input,
    float* output)
{
  assert(n % (4 * sizeof(float)) == 0);

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

  for (; n >= 4 * sizeof(float); n -= 4 * sizeof(float)) {
    float32x4_t vn = vld1q_f32(input); input += 4;

    vn = vminq_f32(vn, vsigmoid_maxinput);
    vn = vmaxq_f32(vn, vsigmoid_mininput);

    const float32x4_t vn_sq = vmulq_f32(vn, vn);

    // Evaluate numerator polynomial
    float32x4_t vnum = vmlaq_f32(valpha_7, vn_sq, valpha_9);

    vnum = vmlaq_f32(valpha_5, vn_sq, vnum);
    vnum = vmlaq_f32(valpha_3, vn_sq, vnum);
    vnum = vmlaq_f32(valpha_1, vn_sq, vnum);
    vnum = vmulq_f32(vn, vnum);

    // Evaluate denominator polynomial

    float32x4_t vdenom = vmlaq_f32(vbeta_8, vn_sq, vbeta_10);
    vdenom = vmlaq_f32(vbeta_6, vn_sq, vdenom);
    vdenom = vmlaq_f32(vbeta_4, vn_sq, vdenom);
    vdenom = vmlaq_f32(vbeta_2, vn_sq, vdenom);
    vdenom = vmlaq_f32(vbeta_0, vn_sq, vdenom);

    // Do division, one NR iteration

    float32x4_t vrecp = vrecpeq_f32(vdenom);
    vrecp = vmulq_f32(vrecp, vrecpsq_f32(vrecp, vdenom));

    const float32x4_t vsigmoid = vmlaq_f32(vhalf, vnum, vrecp);

    vst1q_f32(output, vsigmoid); output += 4;
  }
}
