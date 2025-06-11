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

#include "src/xnnpack/simd/f32-hvx.h"
#include "src/xnnpack/vcvt.h"

void xnn_f32_qs8_vcvt_ukernel__hvx_u32(
    size_t batch,
    const float* input,
    int8_t* output,
    const struct xnn_f32_qs8_cvt_params* restrict params)
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const HVX_Vector vscale = xnn_set1_f32(params->scalar.scale);
  const HVX_Vector vmagic_bias = xnn_set1_f32(12582912.0f);
  const HVX_Vector vmagic_bias_less_zero_point = Q6_V_vsplat_R(INT32_C(0x4B400000) - (int32_t) params->scalar.output_zero_point);
  XNN_FORCE_REALIZATION(vmagic_bias);
  for (; batch >= 32 * sizeof(float); batch -= 32 * sizeof(float)) {
    HVX_Vector vx = xnn_loadu_f32(input);
    input += 32;

    vx = xnn_fmadd_f32(vx, vscale, vmagic_bias);

    const HVX_Vector vacc = Q6_Vw_vsub_VwVw_sat(vx, vmagic_bias_less_zero_point);

    const HVX_Vector vacc_h = Q6_Vh_vpack_VwVw_sat(vacc, vacc);

    HVX_Vector vy = Q6_Vb_vpack_VhVh_sat(vacc_h, vacc_h);

    Q6_V_vstu_variable(output, 32, vy);
    output += 32;
  }
  if XNN_UNLIKELY(batch != 0) {
    HVX_Vector vx = xnn_load_tail_f32(input, batch >> XNN_LOG2_SIZEOF_FLOAT);

    vx = xnn_fmadd_f32(vx, vscale, vmagic_bias);

    const HVX_Vector vacc = Q6_Vw_vsub_VwVw_sat(vx, vmagic_bias_less_zero_point);

    const HVX_Vector vacc_h = Q6_Vh_vpack_VwVw_sat(vacc, vacc);

    HVX_Vector vy = Q6_Vb_vpack_VhVh_sat(vacc_h, vacc_h);

    // Since the output data type is int8_t,
    // we simply determine the number of elements using batch >> 2
    // without multiplying by sizeof(int8_t).
    Q6_V_vstu_variable(output, batch >> 2, vy);
  }
}
