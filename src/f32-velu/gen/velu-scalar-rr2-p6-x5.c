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
#include <xnnpack/vunary.h>

#include <fp16/bitcasts.h>


void xnn_f32_velu_ukernel__scalar_rr2_p6_x5(
    size_t n,
    const float* x,
    float* y,
    const union xnn_f32_elu_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(n % sizeof(float) == 0);

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

  for (; n >= 5 * sizeof(float); n -= 5 * sizeof(float)) {
    float vx0 = x[0];
    float vx1 = x[1];
    float vx2 = x[2];
    float vx3 = x[3];
    float vx4 = x[4];
    x += 5;

    const float vz0 = vx0 * vprescale;
    const float vz1 = vx1 * vprescale;
    const float vz2 = vx2 * vprescale;
    const float vz3 = vx3 * vprescale;
    const float vz4 = vx4 * vprescale;

    float vn0 = vz0 * vlog2e + vmagic_bias;
    float vn1 = vz1 * vlog2e + vmagic_bias;
    float vn2 = vz2 * vlog2e + vmagic_bias;
    float vn3 = vz3 * vlog2e + vmagic_bias;
    float vn4 = vz4 * vlog2e + vmagic_bias;

    float vs0 = fp32_from_bits(fp32_to_bits(vn0) << 23);
    vn0 -= vmagic_bias;
    float vs1 = fp32_from_bits(fp32_to_bits(vn1) << 23);
    vn1 -= vmagic_bias;
    float vs2 = fp32_from_bits(fp32_to_bits(vn2) << 23);
    vn2 -= vmagic_bias;
    float vs3 = fp32_from_bits(fp32_to_bits(vn3) << 23);
    vn3 -= vmagic_bias;
    float vs4 = fp32_from_bits(fp32_to_bits(vn4) << 23);
    vn4 -= vmagic_bias;

    float vt0 = vn0 * vminus_ln2_hi + vz0;
    float vt1 = vn1 * vminus_ln2_hi + vz1;
    float vt2 = vn2 * vminus_ln2_hi + vz2;
    float vt3 = vn3 * vminus_ln2_hi + vz3;
    float vt4 = vn4 * vminus_ln2_hi + vz4;

    vt0 = vn0 * vminus_ln2_lo + vt0;
    vt1 = vn1 * vminus_ln2_lo + vt1;
    vt2 = vn2 * vminus_ln2_lo + vt2;
    vt3 = vn3 * vminus_ln2_lo + vt3;
    vt4 = vn4 * vminus_ln2_lo + vt4;

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

    float vp0 = vc6 * vt0 + vc5;
    float vp1 = vc6 * vt1 + vc5;
    float vp2 = vc6 * vt2 + vc5;
    float vp3 = vc6 * vt3 + vc5;
    float vp4 = vc6 * vt4 + vc5;

    vp0 = vp0 * vt0 + vc4;
    vp1 = vp1 * vt1 + vc4;
    vp2 = vp2 * vt2 + vc4;
    vp3 = vp3 * vt3 + vc4;
    vp4 = vp4 * vt4 + vc4;

    vp0 = vp0 * vt0 + vc3;
    vp1 = vp1 * vt1 + vc3;
    vp2 = vp2 * vt2 + vc3;
    vp3 = vp3 * vt3 + vc3;
    vp4 = vp4 * vt4 + vc3;

    vp0 = vp0 * vt0 + vc2;
    vp1 = vp1 * vt1 + vc2;
    vp2 = vp2 * vt2 + vc2;
    vp3 = vp3 * vt3 + vc2;
    vp4 = vp4 * vt4 + vc2;

    vp0 *= vt0;
    vp1 *= vt1;
    vp2 *= vt2;
    vp3 *= vt3;
    vp4 *= vt4;

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

    vp0 = vp0 * vt0 + vt0;
    vp1 = vp1 * vt1 + vt1;
    vp2 = vp2 * vt2 + vt2;
    vp3 = vp3 * vt3 + vt3;
    vp4 = vp4 * vt4 + vt4;

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

    y[0] = vy0;
    y[1] = vy1;
    y[2] = vy2;
    y[3] = vy3;
    y[4] = vy4;
    y += 5;
  }
  if XNN_UNLIKELY(n != 0) {
    do {
      float vx = *x++;

      const float vz = vx * vprescale;

      float vn = vz * vlog2e + vmagic_bias;
      float vs = fp32_from_bits(fp32_to_bits(vn) << 23);
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

      *y++ = vy;

      n -= sizeof(float);
    } while (n != 0);
  }
}
