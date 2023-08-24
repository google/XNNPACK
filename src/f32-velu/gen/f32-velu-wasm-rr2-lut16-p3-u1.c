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

void xnn_f32_velu_ukernel__wasm_rr2_lut16_p3_u1(
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

  do {
    float vx = *input++;

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

    *output++ = vy;

    batch -= sizeof(float);
  } while (batch != 0);
}
