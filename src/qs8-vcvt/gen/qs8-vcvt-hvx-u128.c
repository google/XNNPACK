// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/qs8-vcvt/hvx.c.in
//   Generator: tools/xngen
//
// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <hexagon_protos.h>
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/intrinsics-polyfill.h"
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/vcvt.h"


void xnn_qs8_vcvt_ukernel__hvx_u128(
    size_t batch,
    const int8_t* input,
    int8_t* output,
    const struct xnn_qs8_cvt_params* restrict params) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(int8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const HVX_Vector vinput_zero_point = Q6_Vh_vsplat_R(params->scalar.input_zero_point);
  const HVX_Vector vmultiplier = Q6_Vh_vsplat_R(-params->scalar.multiplier);
  const HVX_Vector voutput_zero_point = Q6_Vh_vsplat_R(params->scalar.output_zero_point);
  XNN_FORCE_REALIZATION(vinput_zero_point);
  XNN_FORCE_REALIZATION(vmultiplier);
  XNN_FORCE_REALIZATION(voutput_zero_point);
  for (; batch >= 128 * sizeof(int8_t); batch -= 128 * sizeof(int8_t)) {
    const HVX_Vector vx0 = *((const HVX_UVector*) (input + 0));
    input += 128;

    const HVX_VectorPair vacc_pair0 = Q6_Wh_vunpack_Vb(vx0);
    HVX_Vector vacc0 = Q6_V_lo_W(vacc_pair0);
    HVX_Vector vacc1 = Q6_V_hi_W(vacc_pair0);

    vacc0 = Q6_Vh_vsub_VhVh(vinput_zero_point, vacc0);
    vacc0 = Q6_Vh_vasl_VhR(vacc0, 7);
    vacc0 = Q6_Vh_vmpy_VhVh_s1_rnd_sat(vacc0, vmultiplier);
    vacc0 = Q6_Vh_vadd_VhVh_sat(vacc0, voutput_zero_point);
    vacc1 = Q6_Vh_vsub_VhVh(vinput_zero_point, vacc1);
    vacc1 = Q6_Vh_vasl_VhR(vacc1, 7);
    vacc1 = Q6_Vh_vmpy_VhVh_s1_rnd_sat(vacc1, vmultiplier);
    vacc1 = Q6_Vh_vadd_VhVh_sat(vacc1, voutput_zero_point);

    const HVX_Vector vy0 = Q6_Vb_vpack_VhVh_sat(vacc1, vacc0);

    *((HVX_UVector*) output) = vy0;
    output += 128;
  }
  for (; batch >= 64 * sizeof(int8_t); batch -= 64 * sizeof(int8_t)) {
    const HVX_Vector vx = *((const HVX_UVector*) input);
    input += 64;

    HVX_Vector vacc = Q6_V_lo_W(Q6_Wh_vunpack_Vb(vx));
    vacc = Q6_Vh_vsub_VhVh(vinput_zero_point, vacc);
    vacc = Q6_Vh_vasl_VhR(vacc, 7);
    vacc = Q6_Vh_vmpy_VhVh_s1_rnd_sat(vacc, vmultiplier);
    vacc = Q6_Vh_vadd_VhVh_sat(vacc, voutput_zero_point);

    const HVX_Vector vy = Q6_Vb_vpack_VhVh_sat(vacc, vacc);
    Q6_V_vstu_variable(output, 64, vy);
    output += 64;
  }
  for (; batch >= 32 * sizeof(int8_t); batch -= 32 * sizeof(int8_t)) {
    const HVX_Vector vx = *((const HVX_UVector*) input);
    input += 32;

    HVX_Vector vacc = Q6_V_lo_W(Q6_Wh_vunpack_Vb(vx));
    vacc = Q6_Vh_vsub_VhVh(vinput_zero_point, vacc);
    vacc = Q6_Vh_vasl_VhR(vacc, 7);
    vacc = Q6_Vh_vmpy_VhVh_s1_rnd_sat(vacc, vmultiplier);
    vacc = Q6_Vh_vadd_VhVh_sat(vacc, voutput_zero_point);

    const HVX_Vector vy = Q6_Vb_vpack_VhVh_sat(vacc, vacc);
    Q6_V_vstu_variable(output, 32, vy);
    output += 32;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(int8_t));
    assert(batch <= 31 * sizeof(int8_t));

    const HVX_Vector vx = *((const HVX_UVector*) input);
    HVX_Vector vacc = Q6_V_lo_W(Q6_Wh_vunpack_Vb(vx));
    vacc = Q6_Vh_vsub_VhVh(vinput_zero_point, vacc);
    vacc = Q6_Vh_vasl_VhR(vacc, 7);
    vacc = Q6_Vh_vmpy_VhVh_s1_rnd_sat(vacc, vmultiplier);
    vacc = Q6_Vh_vadd_VhVh_sat(vacc, voutput_zero_point);

    const HVX_Vector vy = Q6_Vb_vpack_VhVh_sat(vacc, vacc);
    Q6_V_vstu_variable(output, batch, vy);
  }
}
