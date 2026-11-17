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


void xnn_qu8_vcvt_ukernel__hvx_u32(
    size_t batch,
    const uint8_t* input,
    uint8_t* output,
    const struct xnn_qu8_cvt_params* restrict params) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const HVX_Vector vinput_zero_point = Q6_Vh_vsplat_R(params->scalar.input_zero_point);
  const HVX_Vector vmultiplier = Q6_Vh_vsplat_R(-params->scalar.multiplier);
  const HVX_Vector voutput_zero_point = Q6_Vh_vsplat_R(params->scalar.output_zero_point);
  XNN_FORCE_REALIZATION(vinput_zero_point);
  XNN_FORCE_REALIZATION(vmultiplier);
  XNN_FORCE_REALIZATION(voutput_zero_point);
  for (; batch >= 32 * sizeof(uint8_t); batch -= 32 * sizeof(uint8_t)) {
    const HVX_Vector vx = *((const HVX_UVector*) input);
    input += 32;

    HVX_Vector vacc = Q6_V_lo_W(Q6_Wuh_vunpack_Vub(vx));
    vacc = Q6_Vh_vsub_VhVh(vinput_zero_point, vacc);
    vacc = Q6_Vh_vasl_VhR(vacc, 7);
    vacc = Q6_Vh_vmpy_VhVh_s1_rnd_sat(vacc, vmultiplier);
    vacc = Q6_Vh_vadd_VhVh_sat(vacc, voutput_zero_point);

    const HVX_Vector vy = Q6_Vub_vpack_VhVh_sat(vacc, vacc);
    Q6_V_vstu_variable(output, 32, vy);
    output += 32;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(uint8_t));
    assert(batch <= 31 * sizeof(uint8_t));

    const HVX_Vector vx = *((const HVX_UVector*) input);
    HVX_Vector vacc = Q6_V_lo_W(Q6_Wuh_vunpack_Vub(vx));
    vacc = Q6_Vh_vsub_VhVh(vinput_zero_point, vacc);
    vacc = Q6_Vh_vasl_VhR(vacc, 7);
    vacc = Q6_Vh_vmpy_VhVh_s1_rnd_sat(vacc, vmultiplier);
    vacc = Q6_Vh_vadd_VhVh_sat(vacc, voutput_zero_point);

    const HVX_Vector vy = Q6_Vub_vpack_VhVh_sat(vacc, vacc);
    Q6_V_vstu_variable(output, batch, vy);
  }
}
