// Auto-generated file. Do not edit!
//   Template: src/f32-vsigmoid/scalar-rr2-lut64-p2-div.c.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <math.h>

#include "xnnpack/common.h"
#include "xnnpack/math.h"
#include "xnnpack/vunary.h"


// Note redefine as uint32[] to avoid redundant bitcasts.
extern XNN_INTERNAL const uint32_t xnn_table_exp2minus_k_over_64[64];

void xnn_f32_vsigmoid_ukernel__scalar_rr2_lut64_p2_div_u4(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_sigmoid_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const float vmagic_bias = 0x1.800000p17f;
  const float vminus_log2e = -0x1.715476p0f;  
  const uint32_t vindex_mask = UINT32_C(0x3F);
  const float vln2_hi = 0x1.630000p-1f;
  const float vln2_lo = -0x1.BD0106p-13f;
  const float vc2 = 0x1.FFFF0Ap-2f;
  const float vone = 1.0f;
  const float vdenorm_cutoff = 0x1.5D589Ep+6f;

  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    const float vx0 = input[0];
    const float vx1 = input[1];
    const float vx2 = input[2];
    const float vx3 = input[3];
    input += 4;

    const float vz0 = fabsf(vx0);
    const float vz1 = fabsf(vx1);
    const float vz2 = fabsf(vx2);
    const float vz3 = fabsf(vx3);

    float vn0 = vz0 * vminus_log2e + vmagic_bias;
    float vn1 = vz1 * vminus_log2e + vmagic_bias;
    float vn2 = vz2 * vminus_log2e + vmagic_bias;
    float vn3 = vz3 * vminus_log2e + vmagic_bias;

    const uint32_t ve0 = float_as_uint32(vn0) << 17;
    const uint32_t ve1 = float_as_uint32(vn1) << 17;
    const uint32_t ve2 = float_as_uint32(vn2) << 17;
    const uint32_t ve3 = float_as_uint32(vn3) << 17;

    const uint32_t vidx0 = float_as_uint32(vn0) & vindex_mask;
    const float vs0 = uint32_as_float(xnn_table_exp2minus_k_over_64[vidx0] + ve0);
    const uint32_t vidx1 = float_as_uint32(vn1) & vindex_mask;
    const float vs1 = uint32_as_float(xnn_table_exp2minus_k_over_64[vidx1] + ve1);
    const uint32_t vidx2 = float_as_uint32(vn2) & vindex_mask;
    const float vs2 = uint32_as_float(xnn_table_exp2minus_k_over_64[vidx2] + ve2);
    const uint32_t vidx3 = float_as_uint32(vn3) & vindex_mask;
    const float vs3 = uint32_as_float(xnn_table_exp2minus_k_over_64[vidx3] + ve3);

    vn0 -= vmagic_bias;
    vn1 -= vmagic_bias;
    vn2 -= vmagic_bias;
    vn3 -= vmagic_bias;

    float vt0 = vn0 * vln2_hi + vz0;
    float vt1 = vn1 * vln2_hi + vz1;
    float vt2 = vn2 * vln2_hi + vz2;
    float vt3 = vn3 * vln2_hi + vz3;

    vt0 = vn0 * vln2_lo + vt0;
    vt1 = vn1 * vln2_lo + vt1;
    vt2 = vn2 * vln2_lo + vt2;
    vt3 = vn3 * vln2_lo + vt3;

    float vp0 = vt0 * vc2;
    float vp1 = vt1 * vc2;
    float vp2 = vt2 * vc2;
    float vp3 = vt3 * vc2;

    vp0 = vt0 - vp0 * vt0;
    vp1 = vt1 - vp1 * vt1;
    vp2 = vt2 - vp2 * vt2;
    vp3 = vt3 - vp3 * vt3;

    const float vy0 = vs0 - vs0 * vp0;
    const float vy1 = vs1 - vs1 * vp1;
    const float vy2 = vs2 - vs2 * vp2;
    const float vy3 = vs3 - vs3 * vp3;

    const float vd0 = vy0 + vone;
    const float vd1 = vy1 + vone;
    const float vd2 = vy2 + vone;
    const float vd3 = vy3 + vone;

    float vf0 = vy0 / vd0;
    float vf1 = vy1 / vd1;
    float vf2 = vy2 / vd2;
    float vf3 = vy3 / vd3;

    if XNN_UNPREDICTABLE(vz0 > vdenorm_cutoff) {
      vf0 = 0.0f;
    }
    if XNN_UNPREDICTABLE(vz1 > vdenorm_cutoff) {
      vf1 = 0.0f;
    }
    if XNN_UNPREDICTABLE(vz2 > vdenorm_cutoff) {
      vf2 = 0.0f;
    }
    if XNN_UNPREDICTABLE(vz3 > vdenorm_cutoff) {
      vf3 = 0.0f;
    }

    if XNN_UNPREDICTABLE(vx0 > 0.0f) {
      vf0 = vone - vf0;
    }
    if XNN_UNPREDICTABLE(vx1 > 0.0f) {
      vf1 = vone - vf1;
    }
    if XNN_UNPREDICTABLE(vx2 > 0.0f) {
      vf2 = vone - vf2;
    }
    if XNN_UNPREDICTABLE(vx3 > 0.0f) {
      vf3 = vone - vf3;
    }

    output[0] = vf0;
    output[1] = vf1;
    output[2] = vf2;
    output[3] = vf3;
    output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    do {
      const float vx = *input++;

      const float vz = fabsf(vx);

      float vn = vz * vminus_log2e + vmagic_bias;
      const uint32_t ve = float_as_uint32(vn) << 17;
      const uint32_t vidx = float_as_uint32(vn) & vindex_mask;
      const float vs = uint32_as_float(xnn_table_exp2minus_k_over_64[vidx] + ve);
      vn -= vmagic_bias;

      float vt = vn * vln2_hi + vz;
      vt = vn * vln2_lo + vt;

      float vp = vt * vc2;
      vp = vt - vp * vt;

      const float vy = vs - vs * vp;
      const float vd = vy + vone;

      float vf = vy / vd;
      if XNN_UNPREDICTABLE(vz > vdenorm_cutoff) {
        vf = 0.0f;
      }
      if XNN_UNPREDICTABLE(vx > 0.0f) {
        vf = vone - vf;
      }

      *output++ = vf;

      batch -= sizeof(float);
    } while (batch != 0);
  }
}
