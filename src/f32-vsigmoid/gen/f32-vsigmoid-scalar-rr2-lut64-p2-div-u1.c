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

void xnn_f32_vsigmoid_ukernel__scalar_rr2_lut64_p2_div_u1(
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
