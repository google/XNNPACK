// Auto-generated file. Do not edit!
//   Template: src/f32-vtanh/scalar-expm1minus.c.in
//   Generator: tools/xngen
//
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <math.h>

#include "xnnpack/common.h"
#include "xnnpack/math.h"
#include "xnnpack/microparams.h"
#include "xnnpack/vunary.h"


void xnn_f32_vtanh_ukernel__scalar_expm1minus_rr1_p6h5ts_div_u2(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_tanh_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const float vsat_cutoff = params->scalar_expm1minus_rr1_p6h5.sat_cutoff;
  const float vminus_log2e = params->scalar_expm1minus_rr1_p6h5.minus_log2e;
  const float vmagic_bias = params->scalar_expm1minus_rr1_p6h5.magic_bias;
  const float vln2 = params->scalar_expm1minus_rr1_p6h5.ln2;
  const float vc6 = params->scalar_expm1minus_rr1_p6h5.c6;
  const float vc5 = params->scalar_expm1minus_rr1_p6h5.c5;
  const float vc4 = params->scalar_expm1minus_rr1_p6h5.c4;
  const float vc3 = params->scalar_expm1minus_rr1_p6h5.c3;
  const float vc2 = params->scalar_expm1minus_rr1_p6h5.c2;
  const float vminus_two = params->scalar_expm1minus_rr1_p6h5.minus_two;
  const float vone = params->scalar_expm1minus_rr1_p6h5.one;

  for (; batch >= 2 * sizeof(float); batch -= 2 * sizeof(float)) {
    const float vx0 = input[0];
    const float vx1 = input[1];
    input += 2;

    float vz0 = fabsf(vx0);
    float vz1 = fabsf(vx1);

    vz0 = math_pmin_f32(vz0, vsat_cutoff);
    vz1 = math_pmin_f32(vz1, vsat_cutoff);

    float vn0 = vz0 * vminus_log2e + vmagic_bias;
    float vn1 = vz1 * vminus_log2e + vmagic_bias;

    const uint32_t vb0 = float_as_uint32(vn0);
    vn0 -= vmagic_bias;
    const uint32_t vb1 = float_as_uint32(vn1);
    vn1 -= vmagic_bias;

    const uint32_t ve0 = vb0 << 23;
    const uint32_t ve1 = vb1 << 23;

    const float vt0 = vn0 * vln2 + vz0;
    const float vs0 = uint32_as_float(ve0);
    const float vt1 = vn1 * vln2 + vz1;
    const float vs1 = uint32_as_float(ve1);

    float vp0 = vc6 * vt0 + vc5;
    float vp1 = vc6 * vt1 + vc5;
    vp0 = vp0 * vt0 + vc4;
    vp1 = vp1 * vt1 + vc4;
    vp0 = vp0 * vt0 + vc3;
    vp1 = vp1 * vt1 + vc3;
    vp0 = vp0 * vt0 + vc2;
    vp1 = vp1 * vt1 + vc2;
    vp0 = vp0 * vt0 + vminus_two;
    vp1 = vp1 * vt1 + vminus_two;

    const float vts0 = vt0 * vs0;
    const float vsmo0 = vs0 - vone;
    const float vts1 = vt1 * vs1;
    const float vsmo1 = vs1 - vone;

    const float vemo0 = vp0 * vts0 + vsmo0;
    const float vemo1 = vp1 * vts1 + vsmo1;

    const float vepo0 = vemo0 - vminus_two;
    const float vepo1 = vemo1 - vminus_two;

    float vy0 = vemo0 / vepo0;
    float vy1 = vemo1 / vepo1;

    vy0 = copysignf(vy0, vx0);
    vy1 = copysignf(vy1, vx1);

    output[0] = vy0;
    output[1] = vy1;
    output += 2;
  }
  if XNN_UNLIKELY(batch != 0) {
    const float vx = *input;

    float vz = fabsf(vx);

    vz = math_pmin_f32(vz, vsat_cutoff);

    float vn = vz * vminus_log2e + vmagic_bias;

    const uint32_t vb = float_as_uint32(vn);
    vn -= vmagic_bias;

    const uint32_t ve = vb << 23;
    const float vs = uint32_as_float(ve);

    const float vt = vn * vln2 + vz;

    float vp = vc6 * vt + vc5;
    vp = vp * vt + vc4;
    vp = vp * vt + vc3;
    vp = vp * vt + vc2;
    vp = vp * vt + vminus_two;

    const float vts = vt * vs;
    const float vsmo = vs - vone;
    const float vemo = vp * vts + vsmo;

    const float vepo = vemo - vminus_two;

    float vy = vemo / vepo;

    vy = copysignf(vy, vx);

    *output = vy;
  }
}
