// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/qs8-f32-vcvt/hvx.c.in
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
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/simd/f32-hvx.h"
#include "src/xnnpack/vcvt.h"


void xnn_qs8_f32_vcvt_ukernel__hvx_u64(
    size_t batch,
    const int8_t* input,
    float* output,
    const struct xnn_qs8_f32_cvt_params* restrict params) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(int8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const HVX_Vector vzero_point = Q6_Vh_vsplat_R(params->scalar.zero_point);
  const HVX_Vector vscale = xnn_set1_f32(params->scalar.scale);
  XNN_FORCE_REALIZATION(vzero_point);
  XNN_FORCE_REALIZATION(vscale);
  for (; batch >= 64 * sizeof(int8_t); batch -= 64 * sizeof(int8_t)) {
    const HVX_Vector vx0 = *((const HVX_UVector*) (input + 0));
    input += 64;

    const HVX_VectorPair vh_pair0 = Q6_Wh_vunpack_Vb(vx0);
    HVX_Vector vh0 = Q6_Vh_vsub_VhVh(Q6_V_lo_W(vh_pair0), vzero_point);

    const HVX_VectorPair vw_pair0 = Q6_Ww_vunpack_Vh(vh0);
    const HVX_Vector vy0 = xnn_mul_f32(Q6_Vsf_equals_Vw(Q6_V_lo_W(vw_pair0)), vscale);
    const HVX_Vector vy1 = xnn_mul_f32(Q6_Vsf_equals_Vw(Q6_V_hi_W(vw_pair0)), vscale);

    xnn_storeu_f32(output + 0, vy0);
    xnn_storeu_f32(output + 32, vy1);
    output += 64;
  }
  for (; batch >= 32 * sizeof(int8_t); batch -= 32 * sizeof(int8_t)) {
    const HVX_Vector vx = *((const HVX_UVector*) input);
    input += 32;

    const HVX_Vector vh = Q6_Vh_vsub_VhVh(Q6_V_lo_W(Q6_Wh_vunpack_Vb(vx)), vzero_point);
    const HVX_Vector vw = Q6_V_lo_W(Q6_Ww_vunpack_Vh(vh));
    const HVX_Vector vy = xnn_mul_f32(Q6_Vsf_equals_Vw(vw), vscale);

    xnn_storeu_f32(output, vy);
    output += 32;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(int8_t));
    assert(batch <= 31 * sizeof(int8_t));

    const HVX_Vector vx = *((const HVX_UVector*) input);
    const HVX_Vector vh = Q6_Vh_vsub_VhVh(Q6_V_lo_W(Q6_Wh_vunpack_Vb(vx)), vzero_point);
    const HVX_Vector vw = Q6_V_lo_W(Q6_Ww_vunpack_Vh(vh));
    const HVX_Vector vy = xnn_mul_f32(Q6_Vsf_equals_Vw(vw), vscale);

    xnn_store_tail_f32(output, vy, batch);
  }
}
