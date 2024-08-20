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

void xnn_f32_vsigmoid_ukernel__scalar_rr2_lut64_p2_div_u2(
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

  for (; batch >= 2 * sizeof(float); batch -= 2 * sizeof(float)) {
    const float vx0 = input[0];
    const float vx1 = input[1];
    input += 2;

    const float vz0 = fabsf(vx0);
    const float vz1 = fabsf(vx1);

    float vn0 = vz0 * vminus_log2e + vmagic_bias;
    float vn1 = vz1 * vminus_log2e + vmagic_bias;

    const uint32_t ve0 = float_as_uint32(vn0) << 17;
    const uint32_t ve1 = float_as_uint32(vn1) << 17;

    const uint32_t vidx0 = float_as_uint32(vn0) & vindex_mask;
    const float vs0 = uint32_as_float(xnn_table_exp2minus_k_over_64[vidx0] + ve0);
    const uint32_t vidx1 = float_as_uint32(vn1) & vindex_mask;
    const float vs1 = uint32_as_float(xnn_table_exp2minus_k_over_64[vidx1] + ve1);

    vn0 -= vmagic_bias;
    vn1 -= vmagic_bias;

    float vt0 = vn0 * vln2_hi + vz0;
    float vt1 = vn1 * vln2_hi + vz1;

    vt0 = vn0 * vln2_lo + vt0;
    vt1 = vn1 * vln2_lo + vt1;

    float vp0 = vt0 * vc2;
    float vp1 = vt1 * vc2;

    vp0 = vt0 - vp0 * vt0;
    vp1 = vt1 - vp1 * vt1;

    const float vy0 = vs0 - vs0 * vp0;
    const float vy1 = vs1 - vs1 * vp1;

    const float vd0 = vy0 + vone;
    const float vd1 = vy1 + vone;

    float vf0 = vy0 / vd0;
    float vf1 = vy1 / vd1;

    if XNN_UNPREDICTABLE(vz0 > vdenorm_cutoff) {
      vf0 = 0.0f;
    }
    if XNN_UNPREDICTABLE(vz1 > vdenorm_cutoff) {
      vf1 = 0.0f;
    }

    if XNN_UNPREDICTABLE(vx0 > 0.0f) {
      vf0 = vone - vf0;
    }
    if XNN_UNPREDICTABLE(vx1 > 0.0f) {
      vf1 = vone - vf1;
    }

    output[0] = vf0;
    output[1] = vf1;
    output += 2;
  }
  if XNN_UNLIKELY(batch != 0) {
    const float vx = *input;

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

    *output = vf;
  }
}
