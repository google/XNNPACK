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

#include "xnnpack/common.h"
#include "xnnpack/math.h"
#include "xnnpack/vunary.h"


void xnn_f32_velu_ukernel__scalar_rr2_p6_u6(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_elu_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const float vsat_cutoff = -0x1.154246p+4f;
  const float vmagic_bias = 0x1.8000FEp23f;
  const float vlog2e = 0x1.715476p+0f;
  const float vminus_ln2_hi = -0x1.62E440p-1f;
  const float vminus_ln2_lo = 0x1.0105C6p-21f;
  const float vc6 = 0x1.6b7338p-10f;
  const float vc5 = 0x1.12278Ep-7f;
  const float vc4 = 0x1.555716p-5f;
  const float vc3 = 0x1.5554B0p-3f;
  const float vc2 = 0x1.FFFFFEp-2f;
  const float vone = 1.0f;

  const float vprescale = params->scalar.prescale;
  const float valpha = params->scalar.alpha;
  const float vbeta = params->scalar.beta;

  for (; batch >= 6 * sizeof(float); batch -= 6 * sizeof(float)) {
    float vx0 = input[0];
    float vx1 = input[1];
    float vx2 = input[2];
    float vx3 = input[3];
    float vx4 = input[4];
    float vx5 = input[5];
    input += 6;

    const float vz0 = vx0 * vprescale;
    const float vz1 = vx1 * vprescale;
    const float vz2 = vx2 * vprescale;
    const float vz3 = vx3 * vprescale;
    const float vz4 = vx4 * vprescale;
    const float vz5 = vx5 * vprescale;

    float vn0 = vz0 * vlog2e + vmagic_bias;
    float vn1 = vz1 * vlog2e + vmagic_bias;
    float vn2 = vz2 * vlog2e + vmagic_bias;
    float vn3 = vz3 * vlog2e + vmagic_bias;
    float vn4 = vz4 * vlog2e + vmagic_bias;
    float vn5 = vz5 * vlog2e + vmagic_bias;

    float vs0 = uint32_as_float(float_as_uint32(vn0) << 23);
    vn0 -= vmagic_bias;
    float vs1 = uint32_as_float(float_as_uint32(vn1) << 23);
    vn1 -= vmagic_bias;
    float vs2 = uint32_as_float(float_as_uint32(vn2) << 23);
    vn2 -= vmagic_bias;
    float vs3 = uint32_as_float(float_as_uint32(vn3) << 23);
    vn3 -= vmagic_bias;
    float vs4 = uint32_as_float(float_as_uint32(vn4) << 23);
    vn4 -= vmagic_bias;
    float vs5 = uint32_as_float(float_as_uint32(vn5) << 23);
    vn5 -= vmagic_bias;

    float vt0 = vn0 * vminus_ln2_hi + vz0;
    float vt1 = vn1 * vminus_ln2_hi + vz1;
    float vt2 = vn2 * vminus_ln2_hi + vz2;
    float vt3 = vn3 * vminus_ln2_hi + vz3;
    float vt4 = vn4 * vminus_ln2_hi + vz4;
    float vt5 = vn5 * vminus_ln2_hi + vz5;

    vt0 = vn0 * vminus_ln2_lo + vt0;
    vt1 = vn1 * vminus_ln2_lo + vt1;
    vt2 = vn2 * vminus_ln2_lo + vt2;
    vt3 = vn3 * vminus_ln2_lo + vt3;
    vt4 = vn4 * vminus_ln2_lo + vt4;
    vt5 = vn5 * vminus_ln2_lo + vt5;

    if XNN_UNPREDICTABLE(vz0 <= vsat_cutoff) {
      vs0 = 0.0f;
      vt0 = 0.0f;
    }
    if XNN_UNPREDICTABLE(vz1 <= vsat_cutoff) {
      vs1 = 0.0f;
      vt1 = 0.0f;
    }
    if XNN_UNPREDICTABLE(vz2 <= vsat_cutoff) {
      vs2 = 0.0f;
      vt2 = 0.0f;
    }
    if XNN_UNPREDICTABLE(vz3 <= vsat_cutoff) {
      vs3 = 0.0f;
      vt3 = 0.0f;
    }
    if XNN_UNPREDICTABLE(vz4 <= vsat_cutoff) {
      vs4 = 0.0f;
      vt4 = 0.0f;
    }
    if XNN_UNPREDICTABLE(vz5 <= vsat_cutoff) {
      vs5 = 0.0f;
      vt5 = 0.0f;
    }

    float vp0 = vc6 * vt0 + vc5;
    float vp1 = vc6 * vt1 + vc5;
    float vp2 = vc6 * vt2 + vc5;
    float vp3 = vc6 * vt3 + vc5;
    float vp4 = vc6 * vt4 + vc5;
    float vp5 = vc6 * vt5 + vc5;

    vp0 = vp0 * vt0 + vc4;
    vp1 = vp1 * vt1 + vc4;
    vp2 = vp2 * vt2 + vc4;
    vp3 = vp3 * vt3 + vc4;
    vp4 = vp4 * vt4 + vc4;
    vp5 = vp5 * vt5 + vc4;

    vp0 = vp0 * vt0 + vc3;
    vp1 = vp1 * vt1 + vc3;
    vp2 = vp2 * vt2 + vc3;
    vp3 = vp3 * vt3 + vc3;
    vp4 = vp4 * vt4 + vc3;
    vp5 = vp5 * vt5 + vc3;

    vp0 = vp0 * vt0 + vc2;
    vp1 = vp1 * vt1 + vc2;
    vp2 = vp2 * vt2 + vc2;
    vp3 = vp3 * vt3 + vc2;
    vp4 = vp4 * vt4 + vc2;
    vp5 = vp5 * vt5 + vc2;

    vp0 *= vt0;
    vp1 *= vt1;
    vp2 *= vt2;
    vp3 *= vt3;
    vp4 *= vt4;
    vp5 *= vt5;

    vt0 *= vs0;
    vs0 -= vone;
    vt1 *= vs1;
    vs1 -= vone;
    vt2 *= vs2;
    vs2 -= vone;
    vt3 *= vs3;
    vs3 -= vone;
    vt4 *= vs4;
    vs4 -= vone;
    vt5 *= vs5;
    vs5 -= vone;

    vp0 = vp0 * vt0 + vt0;
    vp1 = vp1 * vt1 + vt1;
    vp2 = vp2 * vt2 + vt2;
    vp3 = vp3 * vt3 + vt3;
    vp4 = vp4 * vt4 + vt4;
    vp5 = vp5 * vt5 + vt5;

    const float ve0 = (vp0 + vs0) * valpha;
    float vy0 = vx0 * vbeta;
    const float ve1 = (vp1 + vs1) * valpha;
    float vy1 = vx1 * vbeta;
    const float ve2 = (vp2 + vs2) * valpha;
    float vy2 = vx2 * vbeta;
    const float ve3 = (vp3 + vs3) * valpha;
    float vy3 = vx3 * vbeta;
    const float ve4 = (vp4 + vs4) * valpha;
    float vy4 = vx4 * vbeta;
    const float ve5 = (vp5 + vs5) * valpha;
    float vy5 = vx5 * vbeta;

    if XNN_UNPREDICTABLE(vx0 < 0.0f) {
      vy0 = ve0;
    }
    if XNN_UNPREDICTABLE(vx1 < 0.0f) {
      vy1 = ve1;
    }
    if XNN_UNPREDICTABLE(vx2 < 0.0f) {
      vy2 = ve2;
    }
    if XNN_UNPREDICTABLE(vx3 < 0.0f) {
      vy3 = ve3;
    }
    if XNN_UNPREDICTABLE(vx4 < 0.0f) {
      vy4 = ve4;
    }
    if XNN_UNPREDICTABLE(vx5 < 0.0f) {
      vy5 = ve5;
    }

    output[0] = vy0;
    output[1] = vy1;
    output[2] = vy2;
    output[3] = vy3;
    output[4] = vy4;
    output[5] = vy5;
    output += 6;
  }
  if XNN_UNLIKELY(batch != 0) {
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
}
