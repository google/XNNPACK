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


void xnn_f32_velu_ukernel__wasm_rr2_p6_x4(
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

  for (; n >= 4 * sizeof(float); n -= 4 * sizeof(float)) {
    float vx0 = x[0];
    float vx1 = x[1];
    float vx2 = x[2];
    float vx3 = x[3];
    x += 4;

    const float vz0 = __builtin_wasm_min_f32(__builtin_wasm_max_f32(vx0 * vprescale, vsat_cutoff), 0.0f);
    const float vz1 = __builtin_wasm_min_f32(__builtin_wasm_max_f32(vx1 * vprescale, vsat_cutoff), 0.0f);
    const float vz2 = __builtin_wasm_min_f32(__builtin_wasm_max_f32(vx2 * vprescale, vsat_cutoff), 0.0f);
    const float vz3 = __builtin_wasm_min_f32(__builtin_wasm_max_f32(vx3 * vprescale, vsat_cutoff), 0.0f);

    float vn0 = vz0 * vlog2e + vmagic_bias;
    float vn1 = vz1 * vlog2e + vmagic_bias;
    float vn2 = vz2 * vlog2e + vmagic_bias;
    float vn3 = vz3 * vlog2e + vmagic_bias;

    float vs0 = fp32_from_bits(fp32_to_bits(vn0) << 23);
    vn0 -= vmagic_bias;
    float vs1 = fp32_from_bits(fp32_to_bits(vn1) << 23);
    vn1 -= vmagic_bias;
    float vs2 = fp32_from_bits(fp32_to_bits(vn2) << 23);
    vn2 -= vmagic_bias;
    float vs3 = fp32_from_bits(fp32_to_bits(vn3) << 23);
    vn3 -= vmagic_bias;

    float vt0 = vn0 * vminus_ln2_hi + vz0;
    float vt1 = vn1 * vminus_ln2_hi + vz1;
    float vt2 = vn2 * vminus_ln2_hi + vz2;
    float vt3 = vn3 * vminus_ln2_hi + vz3;

    vt0 = vn0 * vminus_ln2_lo + vt0;
    vt1 = vn1 * vminus_ln2_lo + vt1;
    vt2 = vn2 * vminus_ln2_lo + vt2;
    vt3 = vn3 * vminus_ln2_lo + vt3;


    float vp0 = vc6 * vt0 + vc5;
    float vp1 = vc6 * vt1 + vc5;
    float vp2 = vc6 * vt2 + vc5;
    float vp3 = vc6 * vt3 + vc5;

    vp0 = vp0 * vt0 + vc4;
    vp1 = vp1 * vt1 + vc4;
    vp2 = vp2 * vt2 + vc4;
    vp3 = vp3 * vt3 + vc4;

    vp0 = vp0 * vt0 + vc3;
    vp1 = vp1 * vt1 + vc3;
    vp2 = vp2 * vt2 + vc3;
    vp3 = vp3 * vt3 + vc3;

    vp0 = vp0 * vt0 + vc2;
    vp1 = vp1 * vt1 + vc2;
    vp2 = vp2 * vt2 + vc2;
    vp3 = vp3 * vt3 + vc2;

    vp0 *= vt0;
    vp1 *= vt1;
    vp2 *= vt2;
    vp3 *= vt3;

    vt0 *= vs0;
    vs0 -= vone;
    vt1 *= vs1;
    vs1 -= vone;
    vt2 *= vs2;
    vs2 -= vone;
    vt3 *= vs3;
    vs3 -= vone;

    vp0 = vp0 * vt0 + vt0;
    vp1 = vp1 * vt1 + vt1;
    vp2 = vp2 * vt2 + vt2;
    vp3 = vp3 * vt3 + vt3;

    const float ve0 = (vp0 + vs0) * valpha;
    float vy0 = __builtin_wasm_max_f32(vx0 * vbeta, 0.0f);
    const float ve1 = (vp1 + vs1) * valpha;
    float vy1 = __builtin_wasm_max_f32(vx1 * vbeta, 0.0f);
    const float ve2 = (vp2 + vs2) * valpha;
    float vy2 = __builtin_wasm_max_f32(vx2 * vbeta, 0.0f);
    const float ve3 = (vp3 + vs3) * valpha;
    float vy3 = __builtin_wasm_max_f32(vx3 * vbeta, 0.0f);

    vy0 += __builtin_wasm_min_f32(ve0, 0.0f);
    vy1 += __builtin_wasm_min_f32(ve1, 0.0f);
    vy2 += __builtin_wasm_min_f32(ve2, 0.0f);
    vy3 += __builtin_wasm_min_f32(ve3, 0.0f);

    y[0] = vy0;
    y[1] = vy1;
    y[2] = vy2;
    y[3] = vy3;
    y += 4;
  }
  if XNN_UNLIKELY(n != 0) {
    do {
      float vx = *x++;

      const float vz = __builtin_wasm_min_f32(__builtin_wasm_max_f32(vx * vprescale, vsat_cutoff), 0.0f);

      float vn = vz * vlog2e + vmagic_bias;
      float vs = fp32_from_bits(fp32_to_bits(vn) << 23);
      vn -= vmagic_bias;

      float vt = vn * vminus_ln2_hi + vz;
      vt = vn * vminus_ln2_lo + vt;


      float vp = vc6 * vt + vc5;
      vp = vp * vt + vc4;
      vp = vp * vt + vc3;
      vp = vp * vt + vc2;
      vp *= vt;

      vt *= vs;
      vs -= vone;
      vp = vp * vt + vt;
      const float ve = (vp + vs) * valpha;

      float vy = __builtin_wasm_max_f32(vx * vbeta, 0.0f);
      vy += __builtin_wasm_min_f32(ve, 0.0f);

      *y++ = vy;

      n -= sizeof(float);
    } while (n != 0);
  }
}
