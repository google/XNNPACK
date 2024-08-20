// Auto-generated file. Do not edit!
//   Template: src/f32-raddstoreexpminusmax/hvx-rr2-p5.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#include <assert.h>

#include "xnnpack/simd/f32-hvx.h"
#include "xnnpack/raddstoreexpminusmax.h"

void xnn_f32_raddstoreexpminusmax_ukernel__hvx_rr2_p5_u128(
    size_t batch,
    const float* input,
    const float* max,
    float* output,
    float* sum,
    const union xnn_f32_expminus_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(max != NULL);
  assert(output != NULL);
  assert(sum != NULL);

  const HVX_Vector vi_max = xnn_set1_f32(*max);
  const HVX_Vector vlog2e = xnn_set1_f32(params->hvx_rr2_p5.log2e);
  const HVX_Vector vmagic_bias = xnn_set1_f32(params->hvx_rr2_p5.magic_bias);
  const HVX_Vector vminus_ln2_hi = xnn_set1_f32(params->hvx_rr2_p5.minus_ln2_hi);
  const HVX_Vector vminus_ln2_lo = xnn_set1_f32(params->hvx_rr2_p5.minus_ln2_lo);
  const HVX_Vector vc5 = xnn_set1_f32(params->hvx_rr2_p5.c5);
  const HVX_Vector vc4 = xnn_set1_f32(params->hvx_rr2_p5.c4);
  const HVX_Vector vc3 = xnn_set1_f32(params->hvx_rr2_p5.c3);
  const HVX_Vector vc2 = xnn_set1_f32(params->hvx_rr2_p5.c2);
  const HVX_Vector vc1 = xnn_set1_f32(params->hvx_rr2_p5.c1);
  const HVX_Vector vdenorm_cutoff = xnn_set1_f32(params->hvx_rr2_p5.denorm_cutoff);

  HVX_Vector vacc0 = Q6_V_vzero();
  for (; batch >= 128 * sizeof(float); batch -= 128 * sizeof(float)) {
    const HVX_Vector vi0 =  xnn_loadu_f32(input);
    const HVX_Vector vi1 = xnn_loadu_f32(input + 32);
    const HVX_Vector vi2 = xnn_loadu_f32(input + 64);
    const HVX_Vector vi3 = xnn_loadu_f32(input + 96);
    input += 128;

    // Subtract maximum input x := i - i_max
    const HVX_Vector vx0 = xnn_sub_f32(vi0, vi_max);
    const HVX_Vector vx1 = xnn_sub_f32(vi1, vi_max);
    const HVX_Vector vx2 = xnn_sub_f32(vi2, vi_max);
    const HVX_Vector vx3 = xnn_sub_f32(vi3, vi_max);

    // n := round(x / log(2))
    HVX_Vector vn0 = xnn_fmadd_f32(vx0, vlog2e, vmagic_bias);
    HVX_Vector vn1 = xnn_fmadd_f32(vx1, vlog2e, vmagic_bias);
    HVX_Vector vn2 = xnn_fmadd_f32(vx2, vlog2e, vmagic_bias);
    HVX_Vector vn3 = xnn_fmadd_f32(vx3, vlog2e, vmagic_bias);

    // Create a floating-point number s (scale) such that s == 2**n for inputs which don't cause underflow.
    const HVX_Vector vs0 = Q6_Vw_vasl_VwR(vn0, 23);
    const HVX_Vector vs1 = Q6_Vw_vasl_VwR(vn1, 23);
    const HVX_Vector vs2 = Q6_Vw_vasl_VwR(vn2, 23);
    const HVX_Vector vs3 = Q6_Vw_vasl_VwR(vn3, 23);

    // Subtract the large number back to get final batch := round(x / log(2)).
    vn0 = xnn_sub_f32(vn0, vmagic_bias);
    vn1 = xnn_sub_f32(vn1, vmagic_bias);
    vn2 = xnn_sub_f32(vn2, vmagic_bias);
    vn3 = xnn_sub_f32(vn3, vmagic_bias);

    // Compute reduced argument t := x - batch * log(2).
    // Use Cody-Waite range reduction method (note two constants to represent log(2)) to improve accuracy.
    HVX_Vector vt0 = xnn_fmadd_f32(vn0, vminus_ln2_hi, vx0);
    HVX_Vector vt1 = xnn_fmadd_f32(vn1, vminus_ln2_hi, vx1);
    HVX_Vector vt2 = xnn_fmadd_f32(vn2, vminus_ln2_hi, vx2);
    HVX_Vector vt3 = xnn_fmadd_f32(vn3, vminus_ln2_hi, vx3);

    vt0 = xnn_fmadd_f32(vn0, vminus_ln2_lo, vt0);
    vt1 = xnn_fmadd_f32(vn1, vminus_ln2_lo, vt1);
    vt2 = xnn_fmadd_f32(vn2, vminus_ln2_lo, vt2);
    vt3 = xnn_fmadd_f32(vn3, vminus_ln2_lo, vt3);

    // Compute degree-5 polynomial approximation for exp(t) on [-log(2)/2, log(2)/2].
    //  p := c5 * t + c4;
    //  p = p * t + c3;
    //  p = p * t + c2;
    //  p = p * t + c1;
    HVX_Vector vp0 = xnn_fmadd_f32(vc5, vt0, vc4);
    HVX_Vector vp1 = xnn_fmadd_f32(vc5, vt1, vc4);
    HVX_Vector vp2 = xnn_fmadd_f32(vc5, vt2, vc4);
    HVX_Vector vp3 = xnn_fmadd_f32(vc5, vt3, vc4);

    vp0 = xnn_fmadd_f32(vp0, vt0, vc3);
    vp1 = xnn_fmadd_f32(vp1, vt1, vc3);
    vp2 = xnn_fmadd_f32(vp2, vt2, vc3);
    vp3 = xnn_fmadd_f32(vp3, vt3, vc3);

    vp0 = xnn_fmadd_f32(vp0, vt0, vc2);
    vp1 = xnn_fmadd_f32(vp1, vt1, vc2);
    vp2 = xnn_fmadd_f32(vp2, vt2, vc2);
    vp3 = xnn_fmadd_f32(vp3, vt3, vc2);

    vp0 = xnn_fmadd_f32(vp0, vt0, vc1);
    vp1 = xnn_fmadd_f32(vp1, vt1, vc1);
    vp2 = xnn_fmadd_f32(vp2, vt2, vc1);
    vp3 = xnn_fmadd_f32(vp3, vt3, vc1);

    // Reconstruct the final f value:
    //   f = s * (1 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5)))))
    //     = s + (t * s) * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5))))
    //     = s + (t * s) * p
    vt0 = xnn_mul_f32(vt0, vs0);
    vt1 = xnn_mul_f32(vt1, vs1);
    vt2 = xnn_mul_f32(vt2, vs2);
    vt3 = xnn_mul_f32(vt3, vs3);

    HVX_Vector vf0 = xnn_fmadd_f32(vt0, vp0, vs0);
    HVX_Vector vf1 = xnn_fmadd_f32(vt1, vp1, vs1);
    HVX_Vector vf2 = xnn_fmadd_f32(vt2, vp2, vs2);
    HVX_Vector vf3 = xnn_fmadd_f32(vt3, vp3, vs3);

    // For inputs below zero cutoff, replace output with +0.0f.
    // Note that for NaN inputs, comparison result is false, and outputs are left unchanged.
    vf0 = Q6_V_vand_QnV(Q6_Q_vcmp_gt_VsfVsf(vdenorm_cutoff, vx0), vf0);
    vf1 = Q6_V_vand_QnV(Q6_Q_vcmp_gt_VsfVsf(vdenorm_cutoff, vx1), vf1);
    vf2 = Q6_V_vand_QnV(Q6_Q_vcmp_gt_VsfVsf(vdenorm_cutoff, vx2), vf2);
    vf3 = Q6_V_vand_QnV(Q6_Q_vcmp_gt_VsfVsf(vdenorm_cutoff, vx3), vf3);

    xnn_storeu_f32(output, vf0);
    xnn_storeu_f32(output + 32, vf1);
    xnn_storeu_f32(output + 64, vf2);
    xnn_storeu_f32(output + 96, vf3);
    output += 128;

    vacc0 = xnn_add_f32(vacc0, vf0);
    vacc0 = xnn_add_f32(vacc0, vf1);
    vacc0 = xnn_add_f32(vacc0, vf2);
    vacc0 = xnn_add_f32(vacc0, vf3);
  }

  HVX_Vector vacc = vacc0;
  for (; batch >= 32 * sizeof(float); batch -= 32 * sizeof(float)) {
    const HVX_Vector vi = xnn_loadu_f32(input);
    input += 32;

    const HVX_Vector vx = xnn_sub_f32(vi, vi_max);

    HVX_Vector vn = xnn_fmadd_f32(vx, vlog2e, vmagic_bias);

    const HVX_Vector vs = Q6_Vw_vasl_VwR(vn, 23);

    vn = xnn_sub_f32(vn, vmagic_bias);

    HVX_Vector vt = xnn_fmadd_f32(vn, vminus_ln2_hi, vx);
    vt = xnn_fmadd_f32(vn, vminus_ln2_lo, vt);

    HVX_Vector vp = xnn_fmadd_f32(vc5, vt, vc4);
    vp = xnn_fmadd_f32(vp, vt, vc3);
    vp = xnn_fmadd_f32(vp, vt, vc2);
    vp = xnn_fmadd_f32(vp, vt, vc1);

    vt = xnn_mul_f32(vt, vs);
    HVX_Vector vf = xnn_fmadd_f32(vt, vp, vs);

    vf = Q6_V_vand_QnV(Q6_Q_vcmp_gt_VsfVsf(vdenorm_cutoff, vx), vf);

    xnn_storeu_f32(output, vf);
    output += 32;

    vacc = xnn_add_f32(vacc, vf);
  }

  float vacc_lo = Q6_f32_vrsum_Vsf(vacc);
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch < 32 * sizeof(float));

    const HVX_Vector vi = xnn_loadu_f32(input);

    const HVX_Vector vx = xnn_sub_f32(vi, vi_max);

    HVX_Vector vn = xnn_fmadd_f32(vx, vlog2e, vmagic_bias);

    const HVX_Vector vs = Q6_Vw_vasl_VwR(vn, 23);

    vn = xnn_sub_f32(vn, vmagic_bias);

    HVX_Vector vt = xnn_fmadd_f32(vn, vminus_ln2_hi, vx);
    vt = xnn_fmadd_f32(vn, vminus_ln2_lo, vt);

    HVX_Vector vp = xnn_fmadd_f32(vc5, vt, vc4);
    vp = xnn_fmadd_f32(vp, vt, vc3);
    vp = xnn_fmadd_f32(vp, vt, vc2);
    vp = xnn_fmadd_f32(vp, vt, vc1);

    vt = xnn_mul_f32(vt, vs);
    HVX_Vector vf = xnn_fmadd_f32(vt, vp, vs);

    vf = Q6_V_vand_QnV(Q6_Q_vcmp_gt_VsfVsf(vdenorm_cutoff, vx), vf);

    Q6_V_vstu_variable(output, batch, vf);

    vf = Q6_V_vand_QV(Q6_Q_vsetq_R(batch), vf);
    vacc_lo += Q6_f32_vrsum_Vsf(vf);
  }
  *sum = vacc_lo;
}
