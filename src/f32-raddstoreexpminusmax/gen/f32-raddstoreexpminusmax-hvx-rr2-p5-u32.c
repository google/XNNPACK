// Auto-generated file. Do not edit!
//   Template: src/f32-raddstoreexpminusmax/hvx-rr2-p5.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#include <assert.h>

#include <hvx_hexagon_protos.h>
#include <hexagon_protos.h>
#include <hexagon_types.h>

#include "xnnpack/common.h"
#include <xnnpack/intrinsics-polyfill.h>
#include "xnnpack/raddstoreexpminusmax.h"

void xnn_f32_raddstoreexpminusmax_ukernel__hvx_rr2_p5_u32(
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

  const HVX_Vector vi_max = Q6_V_vsplat_R(*((int32_t*) max));
  const HVX_Vector vlog2e = Q6_V_vsplat_R(*((int32_t *) &params->hvx_rr2_p5.log2e));
  const HVX_Vector vmagic_bias = Q6_V_vsplat_R(*((int32_t *) &params->hvx_rr2_p5.magic_bias));  
  const HVX_Vector vindex_mask = Q6_V_vsplat_R(INT32_C(0x3F));
  const HVX_Vector vminus_ln2_hi = Q6_V_vsplat_R(*((int32_t *) &params->hvx_rr2_p5.minus_ln2_hi));
  const HVX_Vector vminus_ln2_lo = Q6_V_vsplat_R(*((int32_t *) &params->hvx_rr2_p5.minus_ln2_lo));
  const HVX_Vector vc5 = Q6_V_vsplat_R(*((int32_t *) &params->hvx_rr2_p5.c5));
  const HVX_Vector vc4 = Q6_V_vsplat_R(*((int32_t *) &params->hvx_rr2_p5.c4));
  const HVX_Vector vc3 = Q6_V_vsplat_R(*((int32_t *) &params->hvx_rr2_p5.c3));
  const HVX_Vector vc2 = Q6_V_vsplat_R(*((int32_t *) &params->hvx_rr2_p5.c2));
  const HVX_Vector vc1 = Q6_V_vsplat_R(*((int32_t *) &params->hvx_rr2_p5.c1));
  const HVX_Vector vdenorm_cutoff = Q6_V_vsplat_R(*((int32_t *) &params->hvx_rr2_p5.denorm_cutoff));

  HVX_Vector vacc = Q6_V_vzero();
  for (; batch >= 32 * sizeof(float); batch -= 32 * sizeof(float)) {
    const HVX_Vector vi = *((HVX_UVector *) input);
    input += 32;

    const HVX_Vector vx = Q6_Vsf_vsub_VsfVsf(vi, vi_max);

    HVX_Vector vn = Q6_Vsf_vmpyadd_VsfVsf(vx, vlog2e, vmagic_bias);

    const HVX_Vector vs = Q6_Vw_vasl_VwR(vn, 23);

    vn = Q6_Vsf_vsub_VsfVsf(vn, vmagic_bias);

    HVX_Vector vt = Q6_Vsf_vmpyadd_VsfVsf(vn, vminus_ln2_hi, vx);
    vt = Q6_Vsf_vmpyadd_VsfVsf(vn, vminus_ln2_lo, vt);

    HVX_Vector vp = Q6_Vsf_vmpyadd_VsfVsf(vc5, vt, vc4);
    vp = Q6_Vsf_vmpyadd_VsfVsf(vp, vt, vc3);
    vp = Q6_Vsf_vmpyadd_VsfVsf(vp, vt, vc2);
    vp = Q6_Vsf_vmpyadd_VsfVsf(vp, vt, vc1);

    vt = Q6_Vsf_vmpy_VsfVsf(vt, vs);
    HVX_Vector vf = Q6_Vsf_vmpyadd_VsfVsf(vt, vp, vs);

    vf = Q6_V_vand_QnV(Q6_Q_vcmp_gt_VsfVsf(vdenorm_cutoff, vx), vf);

    *((HVX_UVector *) output) = vf;
    output += 32;

    vacc = Q6_Vsf_vadd_VsfVsf(vacc, vf);
  }

  float vacc_lo = Q6_f32_vrsum_Vsf(vacc);
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch < 32 * sizeof(float));

    const HVX_Vector vi = *((HVX_UVector *) input);

    const HVX_Vector vx = Q6_Vsf_vsub_VsfVsf(vi, vi_max);

    HVX_Vector vn = Q6_Vsf_vmpyadd_VsfVsf(vx, vlog2e, vmagic_bias);

    const HVX_Vector vs = Q6_Vw_vasl_VwR(vn, 23);

    vn = Q6_Vsf_vsub_VsfVsf(vn, vmagic_bias);

    HVX_Vector vt = Q6_Vsf_vmpyadd_VsfVsf(vn, vminus_ln2_hi, vx);
    vt = Q6_Vsf_vmpyadd_VsfVsf(vn, vminus_ln2_lo, vt);

    HVX_Vector vp = Q6_Vsf_vmpyadd_VsfVsf(vc5, vt, vc4);
    vp = Q6_Vsf_vmpyadd_VsfVsf(vp, vt, vc3);
    vp = Q6_Vsf_vmpyadd_VsfVsf(vp, vt, vc2);
    vp = Q6_Vsf_vmpyadd_VsfVsf(vp, vt, vc1);

    vt = Q6_Vsf_vmpy_VsfVsf(vt, vs);
    HVX_Vector vf = Q6_Vsf_vmpyadd_VsfVsf(vt, vp, vs);

    vf = Q6_V_vand_QnV(Q6_Q_vcmp_gt_VsfVsf(vdenorm_cutoff, vx), vf);

    Q6_V_vstu_variable(output, batch, vf);

    vf = Q6_V_vand_QV(Q6_Q_vsetq_R(batch), vf);
    vacc_lo += Q6_f32_vrsum_Vsf(vf);
  }
  *sum = vacc_lo;
}
