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


void xnn_f32_vtanh_ukernel__wasm_expm1minus_rr1_p6h5ts_div_u1(
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

  do {
    const float vx = *input++;

    float vz = fabsf(vx);

    vz = __builtin_wasm_min_f32(vz, vsat_cutoff);

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

    *output++ = vy;

    batch -= sizeof(float);
  } while (batch != 0);
}
