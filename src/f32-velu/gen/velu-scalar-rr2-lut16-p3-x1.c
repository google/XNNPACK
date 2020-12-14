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
#include <xnnpack/vunary.h>

#include <fp16/bitcasts.h>


extern XNN_INTERNAL const uint32_t xnn_table_exp2minus_k_over_16[16];

void xnn_f32_velu_ukernel__scalar_rr2_lut16_p3_x1(
    size_t n,
    const float* x,
    float* y,
    const union xnn_f32_elu_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(n % sizeof(float) == 0);

  const float vprescale = params->scalar.prescale;
  const float valpha = params->scalar.alpha;
  const float vbeta = params->scalar.beta;

  const float vmagic_bias = 0x1.800000p19f;
  const float vlog2e = 0x1.715476p+0f;
  const uint32_t vindex_mask = UINT32_C(0xF);
  const float vsat_cutoff = -0x1.154246p+4f;
  const float vminus_ln2_hi = -0x1.62E400p-1f;
  const float vminus_ln2_lo = -0x1.7F7D1Cp-20f;
  const float vc3 = 0x1.55561Cp-3f;
  const float vc2 = 0x1.0001ECp-1f;
  const float vone = 1.0f;

  do {
    float vx = *x++;

    const float vz = vx * vprescale;

    float vn = vz * vlog2e + vmagic_bias;
    const uint32_t ven = fp32_to_bits(vn) << 19;
    const uint32_t vidx = fp32_to_bits(vn) & vindex_mask;
    vn -= vmagic_bias;

    float vt = vn * vminus_ln2_hi + vz;
    float vs = fp32_from_bits(xnn_table_exp2minus_k_over_16[vidx] + ven);

    vt = vn * vminus_ln2_lo + vt;
    if XNN_UNPREDICTABLE(vz <= vsat_cutoff) {
      vs = 0.0f;
      vt = 0.0f;
    }

    float vp = vc3 * vt + vc2;
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
