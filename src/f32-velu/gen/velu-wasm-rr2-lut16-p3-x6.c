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

void xnn_f32_velu_ukernel__wasm_rr2_lut16_p3_x6(
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

  for (; n >= 6 * sizeof(float); n -= 6 * sizeof(float)) {
    float vx0 = x[0];
    float vx1 = x[1];
    float vx2 = x[2];
    float vx3 = x[3];
    float vx4 = x[4];
    float vx5 = x[5];
    x += 6;

    const float vz0 = __builtin_wasm_min_f32(__builtin_wasm_max_f32(vx0 * vprescale, vsat_cutoff), 0.0f);
    const float vz1 = __builtin_wasm_min_f32(__builtin_wasm_max_f32(vx1 * vprescale, vsat_cutoff), 0.0f);
    const float vz2 = __builtin_wasm_min_f32(__builtin_wasm_max_f32(vx2 * vprescale, vsat_cutoff), 0.0f);
    const float vz3 = __builtin_wasm_min_f32(__builtin_wasm_max_f32(vx3 * vprescale, vsat_cutoff), 0.0f);
    const float vz4 = __builtin_wasm_min_f32(__builtin_wasm_max_f32(vx4 * vprescale, vsat_cutoff), 0.0f);
    const float vz5 = __builtin_wasm_min_f32(__builtin_wasm_max_f32(vx5 * vprescale, vsat_cutoff), 0.0f);

    float vn0 = vz0 * vlog2e + vmagic_bias;
    float vn1 = vz1 * vlog2e + vmagic_bias;
    float vn2 = vz2 * vlog2e + vmagic_bias;
    float vn3 = vz3 * vlog2e + vmagic_bias;
    float vn4 = vz4 * vlog2e + vmagic_bias;
    float vn5 = vz5 * vlog2e + vmagic_bias;

    const uint32_t ven0 = fp32_to_bits(vn0) << 19;
    const uint32_t vidx0 = fp32_to_bits(vn0) & vindex_mask;
    vn0 -= vmagic_bias;
    const uint32_t ven1 = fp32_to_bits(vn1) << 19;
    const uint32_t vidx1 = fp32_to_bits(vn1) & vindex_mask;
    vn1 -= vmagic_bias;
    const uint32_t ven2 = fp32_to_bits(vn2) << 19;
    const uint32_t vidx2 = fp32_to_bits(vn2) & vindex_mask;
    vn2 -= vmagic_bias;
    const uint32_t ven3 = fp32_to_bits(vn3) << 19;
    const uint32_t vidx3 = fp32_to_bits(vn3) & vindex_mask;
    vn3 -= vmagic_bias;
    const uint32_t ven4 = fp32_to_bits(vn4) << 19;
    const uint32_t vidx4 = fp32_to_bits(vn4) & vindex_mask;
    vn4 -= vmagic_bias;
    const uint32_t ven5 = fp32_to_bits(vn5) << 19;
    const uint32_t vidx5 = fp32_to_bits(vn5) & vindex_mask;
    vn5 -= vmagic_bias;

    float vt0 = vn0 * vminus_ln2_hi + vz0;
    float vs0 = fp32_from_bits(xnn_table_exp2minus_k_over_16[vidx0] + ven0);
    float vt1 = vn1 * vminus_ln2_hi + vz1;
    float vs1 = fp32_from_bits(xnn_table_exp2minus_k_over_16[vidx1] + ven1);
    float vt2 = vn2 * vminus_ln2_hi + vz2;
    float vs2 = fp32_from_bits(xnn_table_exp2minus_k_over_16[vidx2] + ven2);
    float vt3 = vn3 * vminus_ln2_hi + vz3;
    float vs3 = fp32_from_bits(xnn_table_exp2minus_k_over_16[vidx3] + ven3);
    float vt4 = vn4 * vminus_ln2_hi + vz4;
    float vs4 = fp32_from_bits(xnn_table_exp2minus_k_over_16[vidx4] + ven4);
    float vt5 = vn5 * vminus_ln2_hi + vz5;
    float vs5 = fp32_from_bits(xnn_table_exp2minus_k_over_16[vidx5] + ven5);

    vt0 = vn0 * vminus_ln2_lo + vt0;
    vt1 = vn1 * vminus_ln2_lo + vt1;
    vt2 = vn2 * vminus_ln2_lo + vt2;
    vt3 = vn3 * vminus_ln2_lo + vt3;
    vt4 = vn4 * vminus_ln2_lo + vt4;
    vt5 = vn5 * vminus_ln2_lo + vt5;

    float vp0 = vc3 * vt0 + vc2;
    float vp1 = vc3 * vt1 + vc2;
    float vp2 = vc3 * vt2 + vc2;
    float vp3 = vc3 * vt3 + vc2;
    float vp4 = vc3 * vt4 + vc2;
    float vp5 = vc3 * vt5 + vc2;

    vp0 *= vt0;
    vp1 *= vt1;
    vp2 *= vt2;
    vp3 *= vt3;
    vp4 *= vt4;
    vp5 *= vt5;

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
    vt5 *= vs5;
    vs5 -= vone;

    vp0 = vp0 * vt0 + vt0;
    vp1 = vp1 * vt1 + vt1;
    vp2 = vp2 * vt2 + vt2;
    vp3 = vp3 * vt3 + vt3;
    vp4 = vp4 * vt4 + vt4;
    vp5 = vp5 * vt5 + vt5;

    const float ve0 = (vp0 + vs0) * valpha;
    float vy0 = __builtin_wasm_max_f32(vx0 * vbeta, 0.0f);
    const float ve1 = (vp1 + vs1) * valpha;
    float vy1 = __builtin_wasm_max_f32(vx1 * vbeta, 0.0f);
    const float ve2 = (vp2 + vs2) * valpha;
    float vy2 = __builtin_wasm_max_f32(vx2 * vbeta, 0.0f);
    const float ve3 = (vp3 + vs3) * valpha;
    float vy3 = __builtin_wasm_max_f32(vx3 * vbeta, 0.0f);
    const float ve4 = (vp4 + vs4) * valpha;
    float vy4 = __builtin_wasm_max_f32(vx4 * vbeta, 0.0f);
    const float ve5 = (vp5 + vs5) * valpha;
    float vy5 = __builtin_wasm_max_f32(vx5 * vbeta, 0.0f);

    vy0 += __builtin_wasm_min_f32(ve0, 0.0f);
    vy1 += __builtin_wasm_min_f32(ve1, 0.0f);
    vy2 += __builtin_wasm_min_f32(ve2, 0.0f);
    vy3 += __builtin_wasm_min_f32(ve3, 0.0f);
    vy4 += __builtin_wasm_min_f32(ve4, 0.0f);
    vy5 += __builtin_wasm_min_f32(ve5, 0.0f);

    y[0] = vy0;
    y[1] = vy1;
    y[2] = vy2;
    y[3] = vy3;
    y[4] = vy4;
    y[5] = vy5;
    y += 6;
  }
  if XNN_UNLIKELY(n != 0) {
    do {
      float vx = *x++;

      const float vz = __builtin_wasm_min_f32(__builtin_wasm_max_f32(vx * vprescale, vsat_cutoff), 0.0f);

      float vn = vz * vlog2e + vmagic_bias;
      const uint32_t ven = fp32_to_bits(vn) << 19;
      const uint32_t vidx = fp32_to_bits(vn) & vindex_mask;
      vn -= vmagic_bias;

      float vt = vn * vminus_ln2_hi + vz;
      float vs = fp32_from_bits(xnn_table_exp2minus_k_over_16[vidx] + ven);

      vt = vn * vminus_ln2_lo + vt;

      float vp = vc3 * vt + vc2;
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
