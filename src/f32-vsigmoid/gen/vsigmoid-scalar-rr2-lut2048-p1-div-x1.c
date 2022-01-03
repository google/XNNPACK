// Auto-generated file. Do not edit!
//   Template: src/f32-vsigmoid/scalar-rr2-lut2048-p1-div.c.in
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

void xnn_f32_vsigmoid_ukernel__scalar_rr2_lut2048_p1_div_x1(
    size_t n,
    const float* x,
    float* y,
    const union xnn_f32_sigmoid_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(n % sizeof(float) == 0);

  const float vmagic_bias = params->scalar_rr2_lut2048_p1.magic_bias;
  const float vminus_log2e = params->scalar_rr2_lut2048_p1.minus_log2e;
  const uint32_t vindex_mask = UINT32_C(0x7FF);
  const float vln2_hi = params->scalar_rr2_lut2048_p1.ln2_hi;
  const float vln2_lo = params->scalar_rr2_lut2048_p1.ln2_lo;
  const float vc1 = params->scalar_rr2_lut2048_p1.c1;
  const float vone = params->scalar_rr2_lut2048_p1.one;
  const float vdenorm_cutoff = params->scalar_rr2_lut2048_p1.denorm_cutoff;

  do {
    const float vx = *x++;

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

    *y++ = vf;

    n -= sizeof(float);
  } while (n != 0);
}
