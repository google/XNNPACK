// Auto-generated file. Do not edit!
//   Template: src/f32-velu/scalar-rr2-p6.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <math.h>

#include <xnnpack/common.h>
#include <xnnpack/math.h>
#include <xnnpack/vunary.h>


void xnn_f32_velu_ukernel__scalar_rr2_p6_x1(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_elu_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const float vprescale = params->scalar_rr2_p6.prescale;
  const float valpha = params->scalar_rr2_p6.alpha;
  const float vbeta = params->scalar_rr2_p6.beta;
  const float vmagic_bias = params->scalar_rr2_p6.magic_bias;
  const float vlog2e = params->scalar_rr2_p6.log2e;
  const float vsat_cutoff = params->scalar_rr2_p6.sat_cutoff;
  const float vminus_ln2_hi = params->scalar_rr2_p6.minus_ln2_hi;
  const float vminus_ln2_lo = params->scalar_rr2_p6.minus_ln2_lo;
  const float vc6 = params->scalar_rr2_p6.c6;
  const float vc5 = params->scalar_rr2_p6.c5;
  const float vc4 = params->scalar_rr2_p6.c4;
  const float vc3 = params->scalar_rr2_p6.c3;
  const float vc2 = params->scalar_rr2_p6.c2;
  const float vone = params->scalar_rr2_p6.one;

  do {
    float vx = *input++;

    const float vz = vx * vprescale;

    float vn = vz * vlog2e + vmagic_bias;
    float vs = uint32_as_float(float_as_uint32(vn) << 23);
    vn -= vmagic_bias;

    float vt = vn * vminus_ln2_hi + vz;
    vt = vn * vminus_ln2_lo + vt;

    if XNN_UNPREDICTABLE(vz <= vsat_cutoff) {
      vs = 0.0f;
      vt = 0.0f;
    }

    float vp = vc6 * vt + vc5;
    vp = vp * vt + vc4;
    vp = vp * vt + vc3;
    vp = vp * vt + vc2;
    vp *= vt;

    vt *= vs;
    vs -= vone;
    vp = vp * vt + vt;
    const float ve = (vp + vs) * valpha;

    float vy = vx * vbeta;
    if XNN_UNPREDICTABLE(vx < 0.0f) {
      vy = ve;
    }

    *output++ = vy;

    batch -= sizeof(float);
  } while (batch != 0);
}
