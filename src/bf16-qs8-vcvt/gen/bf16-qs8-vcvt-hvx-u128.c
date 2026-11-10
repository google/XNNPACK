// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f32-qs8-vcvt/hvx.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <float.h>

#include "src/xnnpack/math.h"
#include "src/xnnpack/simd/f32-hvx.h"
#include "src/xnnpack/vcvt.h"

void xnn_bf16_qs8_vcvt_ukernel__hvx_u128(
    size_t batch,
    const xnn_bfloat16* input,
    int8_t* output,
    const struct xnn_bf16_qs8_cvt_params* restrict params) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(xnn_bfloat16) == 0);
  assert(input != NULL);
  assert(output != NULL);

  // Don't let the scale be 0, which can happen for large scales, and should
  // not happen because this value is a reciprocal.
  const HVX_Vector vscale = xnn_set1_f32(math_max_f32(FLT_MIN, xnn_bfloat16_to_float(params->scalar.scale)));
  const HVX_Vector vmagic_bias = xnn_set1_f32(12582912.0f);
  const HVX_Vector vmagic_bias_less_zero_point = Q6_V_vsplat_R(INT32_C(0x4B400000) - (int32_t) params->scalar.output_zero_point);
  XNN_FORCE_REALIZATION(vmagic_bias);
  for (; batch >= 128 * sizeof(xnn_bfloat16); batch -= 128 * sizeof(xnn_bfloat16)) {
    const HVX_VectorPair vw0 = Q6_Ww_vunpackoor_WwVh(Q6_W_vzero(), *((const HVX_UVector*) (input + 0)));
    HVX_Vector vx0 = Q6_V_lo_W(vw0);
    HVX_Vector vx1 = Q6_V_hi_W(vw0);
    const HVX_VectorPair vw1 = Q6_Ww_vunpackoor_WwVh(Q6_W_vzero(), *((const HVX_UVector*) (input + 64)));
    HVX_Vector vx2 = Q6_V_lo_W(vw1);
    HVX_Vector vx3 = Q6_V_hi_W(vw1);
    input += 128;

    vx0 = xnn_fmadd_f32(vx0, vscale, vmagic_bias);
    vx1 = xnn_fmadd_f32(vx1, vscale, vmagic_bias);
    vx2 = xnn_fmadd_f32(vx2, vscale, vmagic_bias);
    vx3 = xnn_fmadd_f32(vx3, vscale, vmagic_bias);

    const HVX_Vector vacc0 = Q6_Vw_vsub_VwVw_sat(vx0, vmagic_bias_less_zero_point);
    const HVX_Vector vacc1 = Q6_Vw_vsub_VwVw_sat(vx1, vmagic_bias_less_zero_point);
    const HVX_Vector vacc2 = Q6_Vw_vsub_VwVw_sat(vx2, vmagic_bias_less_zero_point);
    const HVX_Vector vacc3 = Q6_Vw_vsub_VwVw_sat(vx3, vmagic_bias_less_zero_point);

    // narrowing 32-bit to 16-bit
    const HVX_Vector vacc_h0 = Q6_Vh_vpack_VwVw_sat(vacc1, vacc0);
    const HVX_Vector vacc_h1 = Q6_Vh_vpack_VwVw_sat(vacc3, vacc2);

    // narrowing 16-bit to 8-bit
    HVX_Vector vy0 = Q6_Vb_vpack_VhVh_sat(vacc_h1, vacc_h0);

    *((HVX_UVector *) output) = vy0;
    output += 128;
  }
  for (; batch >= 32 * sizeof(xnn_bfloat16); batch -= 32 * sizeof(xnn_bfloat16)) {
    HVX_Vector vx = Q6_V_lo_W(Q6_Ww_vunpackoor_WwVh(Q6_W_vzero(), *((const HVX_UVector*) input)));
    input += 32;

    vx = xnn_fmadd_f32(vx, vscale, vmagic_bias);

    const HVX_Vector vacc = Q6_Vw_vsub_VwVw_sat(vx, vmagic_bias_less_zero_point);

    const HVX_Vector vacc_h = Q6_Vh_vpack_VwVw_sat(vacc, vacc);

    HVX_Vector vy = Q6_Vb_vpack_VhVh_sat(vacc_h, vacc_h);

    Q6_V_vstu_variable(output, 32, vy);
    output += 32;
  }
  if XNN_UNLIKELY(batch != 0) {
    HVX_Vector vx = Q6_V_lo_W(Q6_Ww_vunpackoor_WwVh(Q6_W_vzero(), *((const HVX_UVector*) input)));

    vx = xnn_fmadd_f32(vx, vscale, vmagic_bias);

    const HVX_Vector vacc = Q6_Vw_vsub_VwVw_sat(vx, vmagic_bias_less_zero_point);

    const HVX_Vector vacc_h = Q6_Vh_vpack_VwVw_sat(vacc, vacc);

    HVX_Vector vy = Q6_Vb_vpack_VhVh_sat(vacc_h, vacc_h);

    Q6_V_vstu_variable(output, batch >> 1, vy);
  }
}
