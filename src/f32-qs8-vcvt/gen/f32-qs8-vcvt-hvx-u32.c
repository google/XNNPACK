// Auto-generated file. Do not edit!
//   Template: src/f32-qs8-vcvt/hvx.c.in
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
#include "xnnpack/intrinsics-polyfill.h"
#include "xnnpack/vcvt.h"


void xnn_f32_qs8_vcvt_ukernel__hvx_u32(
    size_t batch,
    const float* input,
    int8_t* output,
    const union xnn_f32_qs8_cvt_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const HVX_Vector vscale = Q6_V_vsplat_R(*((uint32_t *) &params->hvx.scale));
  const HVX_Vector vmagic_bias = Q6_V_vsplat_R(*((uint32_t *) &params->hvx.magic_bias));
  const HVX_Vector vmagic_bias_less_zero_point = Q6_V_vsplat_R(*((int32_t *) &params->hvx.magic_bias_less_zero_point));
  const HVX_Vector voutput_min = Q6_Vb_vsplat_R(*((int8_t *) &params->hvx.output_min));
  const HVX_Vector voutput_max = Q6_Vb_vsplat_R(*((int8_t *) &params->hvx.output_max));
  for (; batch >= 32 * sizeof(float); batch -= 32 * sizeof(float)) {
    HVX_Vector vx = *((HVX_UVector *) input);
    input += 32;

    vx = Q6_Vsf_vmpyadd_VsfVsf(vx, vscale, vmagic_bias);

    const HVX_Vector vacc = Q6_Vw_vsub_VwVw_sat(vx, vmagic_bias_less_zero_point);

    const HVX_Vector vacc_h = Q6_Vh_vpack_VwVw_sat(vacc, vacc);

    HVX_Vector vy = Q6_Vb_vpack_VhVh_sat(vacc_h, vacc_h);

    vy = Q6_Vb_vmax_VbVb(voutput_min, vy);
    vy = Q6_Vb_vmin_VbVb(voutput_max, vy);

    Q6_V_vstu_variable(output, 32, vy);
    output += 32;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch < 32 * sizeof(float));
    HVX_Vector vx = *((HVX_UVector *) input);

    vx = Q6_Vsf_vmpyadd_VsfVsf(vx, vscale, vmagic_bias);

    const HVX_Vector vacc = Q6_Vw_vsub_VwVw_sat(vx, vmagic_bias_less_zero_point);

    const HVX_Vector vacc_h = Q6_Vh_vpack_VwVw_sat(vacc, vacc);

    HVX_Vector vy = Q6_Vb_vpack_VhVh_sat(vacc_h, vacc_h);

    vy = Q6_Vb_vmax_VbVb(voutput_min, vy);
    vy = Q6_Vb_vmin_VbVb(voutput_max, vy);

    // Since the output data type is int8_t,
    // we simply determine the number of elements
    // without multiplying by sizeof(int8_t).
    int element = batch / sizeof(float);

    Q6_V_vstu_variable(output, element, vy);
  }
}
