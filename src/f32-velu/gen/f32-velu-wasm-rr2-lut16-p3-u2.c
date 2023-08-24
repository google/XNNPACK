// Auto-generated file. Do not edit!
//   Template: src/f32-velu/scalar-rr2-lut16-p3.c.in
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


extern XNN_INTERNAL const uint32_t xnn_table_exp2minus_k_over_16[16];

void xnn_f32_velu_ukernel__wasm_rr2_lut16_p3_u2(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_elu_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const float vprescale = params->scalar_rr2_lut16_p3.prescale;
  const float valpha = params->scalar_rr2_lut16_p3.alpha;
  const float vbeta = params->scalar_rr2_lut16_p3.beta;
  const float vmagic_bias = params->scalar_rr2_lut16_p3.magic_bias;
  const float vlog2e = params->scalar_rr2_lut16_p3.log2e;
  const uint32_t vindex_mask = UINT32_C(0xF);
  const float vsat_cutoff = params->scalar_rr2_lut16_p3.sat_cutoff;
  const float vminus_ln2_hi = params->scalar_rr2_lut16_p3.minus_ln2_hi;
  const float vminus_ln2_lo = params->scalar_rr2_lut16_p3.minus_ln2_lo;
  const float vc3 = params->scalar_rr2_lut16_p3.c3;
  const float vc2 = params->scalar_rr2_lut16_p3.c2;
  const float vone = params->scalar_rr2_lut16_p3.one;

  for (; batch >= 2 * sizeof(float); batch -= 2 * sizeof(float)) {
    float vx0 = input[0];
    float vx1 = input[1];
    input += 2;

    const float vz0 = __builtin_wasm_min_f32(__builtin_wasm_max_f32(vx0 * vprescale, vsat_cutoff), 0.0f);
    const float vz1 = __builtin_wasm_min_f32(__builtin_wasm_max_f32(vx1 * vprescale, vsat_cutoff), 0.0f);

    float vn0 = vz0 * vlog2e + vmagic_bias;
    float vn1 = vz1 * vlog2e + vmagic_bias;

    const uint32_t ven0 = float_as_uint32(vn0) << 19;
    const uint32_t vidx0 = float_as_uint32(vn0) & vindex_mask;
    vn0 -= vmagic_bias;
    const uint32_t ven1 = float_as_uint32(vn1) << 19;
    const uint32_t vidx1 = float_as_uint32(vn1) & vindex_mask;
    vn1 -= vmagic_bias;

    float vt0 = vn0 * vminus_ln2_hi + vz0;
    float vs0 = uint32_as_float(xnn_table_exp2minus_k_over_16[vidx0] + ven0);
    float vt1 = vn1 * vminus_ln2_hi + vz1;
    float vs1 = uint32_as_float(xnn_table_exp2minus_k_over_16[vidx1] + ven1);

    vt0 = vn0 * vminus_ln2_lo + vt0;
    vt1 = vn1 * vminus_ln2_lo + vt1;

    float vp0 = vc3 * vt0 + vc2;
    float vp1 = vc3 * vt1 + vc2;

    vp0 *= vt0;
    vp1 *= vt1;

    vt0 *= vs0;
    vs0 -= vone;
    vt1 *= vs1;
    vs1 -= vone;

    vp0 = vp0 * vt0 + vt0;
    vp1 = vp1 * vt1 + vt1;

    const float ve0 = (vp0 + vs0) * valpha;
    float vy0 = __builtin_wasm_max_f32(vx0 * vbeta, 0.0f);
    const float ve1 = (vp1 + vs1) * valpha;
    float vy1 = __builtin_wasm_max_f32(vx1 * vbeta, 0.0f);

    vy0 += __builtin_wasm_min_f32(ve0, 0.0f);
    vy1 += __builtin_wasm_min_f32(ve1, 0.0f);

    output[0] = vy0;
    output[1] = vy1;
    output += 2;
  }
  if XNN_UNLIKELY(batch != 0) {
    float vx = *input;

    const float vz = __builtin_wasm_min_f32(__builtin_wasm_max_f32(vx * vprescale, vsat_cutoff), 0.0f);

    float vn = vz * vlog2e + vmagic_bias;
    const uint32_t ven = float_as_uint32(vn) << 19;
    const uint32_t vidx = float_as_uint32(vn) & vindex_mask;
    vn -= vmagic_bias;

    float vt = vn * vminus_ln2_hi + vz;
    float vs = uint32_as_float(xnn_table_exp2minus_k_over_16[vidx] + ven);

    vt = vn * vminus_ln2_lo + vt;

    float vp = vc3 * vt + vc2;
    vp *= vt;

    vt *= vs;
    vs -= vone;
    vp = vp * vt + vt;
    const float ve = (vp + vs) * valpha;

    float vy = __builtin_wasm_max_f32(vx * vbeta, 0.0f);
    vy += __builtin_wasm_min_f32(ve, 0.0f);

    *output = vy;
  }
}
