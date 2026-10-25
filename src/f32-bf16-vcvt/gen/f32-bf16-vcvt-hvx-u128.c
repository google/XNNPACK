// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f32-bf16-vcvt/hvx.c.in
//   Generator: tools/xngen
//
// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/simd/f32-hvx.h"
#include "src/xnnpack/vcvt.h"


void xnn_f32_bf16_vcvt_ukernel__hvx_u128(
    size_t batch,
    const float* input,
    xnn_bfloat16* output,
    const void* params) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const HVX_Vector vbias = Q6_V_vsplat_R(0x00007FFFu);
  const HVX_Vector vone = Q6_V_vsplat_R(0x00010000u);

  for (; batch >= 128 * sizeof(float); batch -= 128 * sizeof(float)) {
    HVX_Vector vx0 = xnn_loadu_f32(input + 0);
    HVX_Vector vx1 = xnn_loadu_f32(input + 32);
    HVX_Vector vx2 = xnn_loadu_f32(input + 64);
    HVX_Vector vx3 = xnn_loadu_f32(input + 96);
    input += 128;

    const HVX_Vector vlsb0 = Q6_Vuw_vlsr_VuwR(Q6_V_vand_VV(vx0, vone), 16);
    const HVX_Vector vlsb1 = Q6_Vuw_vlsr_VuwR(Q6_V_vand_VV(vx1, vone), 16);
    const HVX_Vector vlsb2 = Q6_Vuw_vlsr_VuwR(Q6_V_vand_VV(vx2, vone), 16);
    const HVX_Vector vlsb3 = Q6_Vuw_vlsr_VuwR(Q6_V_vand_VV(vx3, vone), 16);

    vx0 = Q6_Vw_vadd_VwVw(vx0, Q6_Vw_vadd_VwVw(vbias, vlsb0));
    vx1 = Q6_Vw_vadd_VwVw(vx1, Q6_Vw_vadd_VwVw(vbias, vlsb1));
    vx2 = Q6_Vw_vadd_VwVw(vx2, Q6_Vw_vadd_VwVw(vbias, vlsb2));
    vx3 = Q6_Vw_vadd_VwVw(vx3, Q6_Vw_vadd_VwVw(vbias, vlsb3));

    const HVX_Vector vy0 = Q6_Vh_vpacko_VwVw(vx1, vx0);
    const HVX_Vector vy1 = Q6_Vh_vpacko_VwVw(vx3, vx2);

    *((HVX_UVector*) (output + 0)) = vy0;
    *((HVX_UVector*) (output + 64)) = vy1;
    output += 128;
  }
  for (; batch >= 64 * sizeof(float); batch -= 64 * sizeof(float)) {
    HVX_Vector vx0 = xnn_loadu_f32(input);
    HVX_Vector vx1 = xnn_loadu_f32(input + 32);
    input += 64;

    const HVX_Vector vlsb0 = Q6_Vuw_vlsr_VuwR(Q6_V_vand_VV(vx0, vone), 16);
    const HVX_Vector vlsb1 = Q6_Vuw_vlsr_VuwR(Q6_V_vand_VV(vx1, vone), 16);
    vx0 = Q6_Vw_vadd_VwVw(vx0, Q6_Vw_vadd_VwVw(vbias, vlsb0));
    vx1 = Q6_Vw_vadd_VwVw(vx1, Q6_Vw_vadd_VwVw(vbias, vlsb1));

    const HVX_Vector vy = Q6_Vh_vpacko_VwVw(vx1, vx0);

    *((HVX_UVector*) output) = vy;
    output += 64;
  }
  for (; batch >= 32 * sizeof(float); batch -= 32 * sizeof(float)) {
    HVX_Vector vx = xnn_loadu_f32(input);
    input += 32;

    const HVX_Vector vlsb = Q6_Vuw_vlsr_VuwR(Q6_V_vand_VV(vx, vone), 16);
    vx = Q6_Vw_vadd_VwVw(vx, Q6_Vw_vadd_VwVw(vbias, vlsb));
    const HVX_Vector vy = Q6_Vh_vpacko_VwVw(vx, vx);

    Q6_V_vstu_variable(output, 32 * sizeof(xnn_bfloat16), vy);
    output += 32;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 31 * sizeof(float));

    HVX_Vector vx = xnn_loadu_f32(input);
    const HVX_Vector vlsb = Q6_Vuw_vlsr_VuwR(Q6_V_vand_VV(vx, vone), 16);
    vx = Q6_Vw_vadd_VwVw(vx, Q6_Vw_vadd_VwVw(vbias, vlsb));
    const HVX_Vector vy = Q6_Vh_vpacko_VwVw(vx, vx);

    Q6_V_vstu_variable(output, batch >> 1, vy);
  }
}
