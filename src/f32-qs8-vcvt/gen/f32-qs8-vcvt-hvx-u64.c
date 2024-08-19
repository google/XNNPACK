// Auto-generated file. Do not edit!
//   Template: src/f32-qs8-vcvt/hvx.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include "xnnpack/simd/f32-hvx.h"
#include "xnnpack/vcvt.h"

void xnn_f32_qs8_vcvt_ukernel__hvx_u64(
    size_t batch,
    const float* input,
    int8_t* output,
    const union xnn_f32_qs8_cvt_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const HVX_Vector vscale = xnn_set1_f32(params->scalar.scale);
  const HVX_Vector vmagic_bias = xnn_set1_f32(12582912.0f);
  const HVX_Vector vmagic_bias_less_zero_point = Q6_V_vsplat_R(INT32_C(0x4B400000) - (int32_t) params->scalar.output_zero_point);
  const HVX_Vector voutput_min = Q6_Vb_vsplat_R(params->hvx.output_min);
  const HVX_Vector voutput_max = Q6_Vb_vsplat_R(params->hvx.output_max);
  for (; batch >= 64 * sizeof(float); batch -= 64 * sizeof(float)) {
    HVX_Vector vx0 = xnn_loadu_f32(input);
    HVX_Vector vx1 = xnn_loadu_f32(input + 32);
    input += 64;

    vx0 = xnn_fmadd_f32(vx0, vscale, vmagic_bias);
    vx1 = xnn_fmadd_f32(vx1, vscale, vmagic_bias);

    const HVX_Vector vacc0 = Q6_Vw_vsub_VwVw_sat(vx0, vmagic_bias_less_zero_point);
    const HVX_Vector vacc1 = Q6_Vw_vsub_VwVw_sat(vx1, vmagic_bias_less_zero_point);

    // narrowing 32-bit to 16-bit
    const HVX_Vector vacc_h0 = Q6_Vh_vpack_VwVw_sat(vacc1, vacc0);

    // narrowing 16-bit to 8-bit
    HVX_Vector vy0 = Q6_Vb_vpack_VhVh_sat(vacc_h0, vacc_h0);

    vy0 = Q6_Vb_vmax_VbVb(voutput_min, vy0);
    vy0 = Q6_Vb_vmin_VbVb(voutput_max, vy0);

    Q6_V_vstu_variable(output, 64, vy0);
    output += 64;
  }
  for (; batch >= 32 * sizeof(float); batch -= 32 * sizeof(float)) {
    HVX_Vector vx = xnn_loadu_f32(input);
    input += 32;

    vx = xnn_fmadd_f32(vx, vscale, vmagic_bias);

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
    HVX_Vector vx = xnn_loadu_f32(input);

    vx = xnn_fmadd_f32(vx, vscale, vmagic_bias);

    const HVX_Vector vacc = Q6_Vw_vsub_VwVw_sat(vx, vmagic_bias_less_zero_point);

    const HVX_Vector vacc_h = Q6_Vh_vpack_VwVw_sat(vacc, vacc);

    HVX_Vector vy = Q6_Vb_vpack_VhVh_sat(vacc_h, vacc_h);

    vy = Q6_Vb_vmax_VbVb(voutput_min, vy);
    vy = Q6_Vb_vmin_VbVb(voutput_max, vy);

    // Since the output data type is int8_t,
    // we simply determine the number of elements using batch >> 2
    // without multiplying by sizeof(int8_t).
    Q6_V_vstu_variable(output, batch >> 2, vy);
  }
}
