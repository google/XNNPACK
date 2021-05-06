// Auto-generated file. Do not edit!
//   Template: src/f32-vsigmoid/scalar-lut2048-p1-div.c.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <math.h>

#include <xnnpack/common.h>
#include <xnnpack/vunary.h>

#include <fp16/bitcasts.h>


// Note redefine as uint32[] to avoid redundant bitcasts.
extern XNN_INTERNAL const uint32_t xnn_table_exp2minus_k_over_2048[2048];

void xnn_f32_vsigmoid_ukernel__scalar_lut2048_p1_div_x2(
    size_t n,
    const float* x,
    float* y,
    const void* params)
{
  assert(n % sizeof(float) == 0);

  const float vmagic_bias = 0x1.800000p12f;
  const float vminus_log2e = -0x1.715476p0f;
  const uint32_t vindex_mask = UINT32_C(0x7FF);
  const float vln2_hi = 0x1.600000p-1f;
  const float vln2_lo = 0x1.7217F8p-8f;
  const float vc1 = -0x1.FFFFFEp-1f;
  const float vone = 1.0f;
  const float vdenorm_cutoff = 0x1.5D589Ep+6f;

  for (; n >= 2 * sizeof(float); n -= 2 * sizeof(float)) {
    const float vx0 = x[0];
    const float vx1 = x[1];
    x += 2;

    const float vz0 = fabsf(vx0);
    const float vz1 = fabsf(vx1);

    float vn0 = vz0 * vminus_log2e + vmagic_bias;
    float vn1 = vz1 * vminus_log2e + vmagic_bias;

    const uint32_t ve0 = fp32_to_bits(vn0) << 12;
    const uint32_t ve1 = fp32_to_bits(vn1) << 12;

    const uint32_t vidx0 = fp32_to_bits(vn0) & vindex_mask;
    const float vs0 = fp32_from_bits(xnn_table_exp2minus_k_over_2048[vidx0] + ve0);
    const uint32_t vidx1 = fp32_to_bits(vn1) & vindex_mask;
    const float vs1 = fp32_from_bits(xnn_table_exp2minus_k_over_2048[vidx1] + ve1);

    vn0 -= vmagic_bias;
    vn1 -= vmagic_bias;

    float vt0 = vn0 * vln2_hi + vz0;
    float vt1 = vn1 * vln2_hi + vz1;

    vt0 = vn0 * vln2_lo + vt0;
    vt1 = vn1 * vln2_lo + vt1;

    const float vp0 = vt0 * vc1;
    const float vp1 = vt1 * vc1;

    const float vy0 = vp0 * vs0 + vs0;
    const float vy1 = vp1 * vs1 + vs1;

    const float vd0 = vy0 + vone;
    const float vd1 = vy1 + vone;

    float vf0 = vy0 / vd0;
    float vf1 = vy1 / vd1;

    if XNN_UNPREDICTABLE(vz0 > vdenorm_cutoff) {
      vf0 = 0.0f;
    }
    if XNN_UNPREDICTABLE(vz1 > vdenorm_cutoff) {
      vf1 = 0.0f;
    }

    if XNN_UNPREDICTABLE(vx0 > 0.0f) {
      vf0 = vone - vf0;
    }
    if XNN_UNPREDICTABLE(vx1 > 0.0f) {
      vf1 = vone - vf1;
    }

    y[0] = vf0;
    y[1] = vf1;
    y += 2;
  }
  if XNN_UNLIKELY(n != 0) {
    const float vx = *x;

    const float vz = fabsf(vx);

    float vn = vz * vminus_log2e + vmagic_bias;
    const uint32_t ve = fp32_to_bits(vn) << 12;
    const uint32_t vidx = fp32_to_bits(vn) & vindex_mask;
    const float vs = fp32_from_bits(xnn_table_exp2minus_k_over_2048[vidx] + ve);
    vn -= vmagic_bias;

    float vt = vn * vln2_hi + vz;
    vt = vn * vln2_lo + vt;

    const float vp = vt * vc1;
    const float vy = vp * vs + vs;

    const float vd = vy + vone;
    float vf = vy / vd;
    if XNN_UNPREDICTABLE(vz > vdenorm_cutoff) {
      vf = 0.0f;
    }
    if XNN_UNPREDICTABLE(vx > 0.0f) {
      vf = vone - vf;
    }

    *y = vf;
  }
}
