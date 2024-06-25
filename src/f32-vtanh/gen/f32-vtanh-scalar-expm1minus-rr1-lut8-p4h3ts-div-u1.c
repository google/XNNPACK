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


extern XNN_INTERNAL const uint32_t xnn_table_exp2minus_k_over_8[8];

void xnn_f32_vtanh_ukernel__scalar_expm1minus_rr1_lut8_p4h3ts_div_u1(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_tanh_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const float vsat_cutoff = params->scalar_expm1minus_rr1_lut8_p4h3.sat_cutoff;
  const float vminus_log2e = params->scalar_expm1minus_rr1_lut8_p4h3.minus_log2e;
  const float vmagic_bias = params->scalar_expm1minus_rr1_lut8_p4h3.magic_bias;
  const uint32_t vindex_mask = UINT32_C(0x7);
  const float vln2 = params->scalar_expm1minus_rr1_lut8_p4h3.ln2;
  const float vc4 = params->scalar_expm1minus_rr1_lut8_p4h3.c4;
  const float vc3 = params->scalar_expm1minus_rr1_lut8_p4h3.c3;
  const float vc2 = params->scalar_expm1minus_rr1_lut8_p4h3.c2;
  const float vminus_two = params->scalar_expm1minus_rr1_lut8_p4h3.minus_two;
  const float vone = params->scalar_expm1minus_rr1_lut8_p4h3.one;

  do {
    const float vx = *input++;

    float vz = fabsf(vx);

    vz = math_pmin_f32(vz, vsat_cutoff);

    float vn = vz * vminus_log2e + vmagic_bias;

    const uint32_t vb = float_as_uint32(vn);
    vn -= vmagic_bias;

    const uint32_t vidx = vb & vindex_mask;
    const uint32_t vl = xnn_table_exp2minus_k_over_8[vidx];
    uint32_t ve = vb << 20;
    ve += vl;
    const float vs = uint32_as_float(ve);

    const float vt = vn * vln2 + vz;

    float vp = vc4 * vt + vc3;
    vp = vp * vt + vc2;
    vp = vp * vt + vminus_two;

    const float vts = vt * vs;
    const float vsmo = vs - vone;
    const float vemo = vp * vts + vsmo;

    const float vepo = vemo - vminus_two;

    float vy = vemo / vepo;

    vy = copysignf(vy, vx);

    *output++ = vy;

    batch -= sizeof(float);
  } while (batch != 0);
}
