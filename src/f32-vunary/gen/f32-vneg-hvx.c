// Auto-generated file. Do not edit!
//   Template: src/f32-vunary/vneg-hvx.c.in
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

#include <xnnpack/common.h>
#include <xnnpack/intrinsics-polyfill.h>
#include <xnnpack/vunary.h>


void xnn_f32_vneg_ukernel__hvx_u32(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const HVX_UVector *vptr = (const HVX_UVector *) input;
  HVX_UVector *vptr_o = (HVX_UVector*) output;

  HVX_Vector v0 = Q6_V_vsplat_R(0);


  for (; batch >= 32 * sizeof(float); batch -= 32 * sizeof(float)) {
    HVX_Vector vx = *vptr++;

    HVX_Vector vacc = Q6_Vsf_vsub_VsfVsf(v0, vx);


    *vptr_o++ = vacc;
  }

  if XNN_UNLIKELY(batch != 0) {
    HVX_Vector vx = *vptr;

    HVX_Vector vacc = Q6_Vsf_vsub_VsfVsf(v0, vx);

    Q6_V_vstu_variable(vptr_o, batch, vacc);
  }
}

void xnn_f32_vneg_ukernel__hvx_u64(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const HVX_UVector *vptr = (const HVX_UVector *) input;
  HVX_UVector *vptr_o = (HVX_UVector*) output;

  HVX_Vector v0 = Q6_V_vsplat_R(0);

  for (; batch >= 64 * sizeof(float); batch -= 64 * sizeof(float)) {
    HVX_Vector vx0 = *vptr++;
    HVX_Vector vx1 = *vptr++;

    HVX_Vector vacc0 = Q6_Vsf_vsub_VsfVsf(v0, vx0);
    HVX_Vector vacc1 = Q6_Vsf_vsub_VsfVsf(v0, vx1);

    *vptr_o++ = vacc0;
    *vptr_o++ = vacc1;
  }

  for (; batch >= 32 * sizeof(float); batch -= 32 * sizeof(float)) {
    HVX_Vector vx = *vptr++;

    HVX_Vector vacc = Q6_Vsf_vsub_VsfVsf(v0, vx);


    *vptr_o++ = vacc;
  }

  if XNN_UNLIKELY(batch != 0) {
    HVX_Vector vx = *vptr;

    HVX_Vector vacc = Q6_Vsf_vsub_VsfVsf(v0, vx);

    Q6_V_vstu_variable(vptr_o, batch, vacc);
  }
}

void xnn_f32_vneg_ukernel__hvx_u128(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const HVX_UVector *vptr = (const HVX_UVector *) input;
  HVX_UVector *vptr_o = (HVX_UVector*) output;

  HVX_Vector v0 = Q6_V_vsplat_R(0);

  for (; batch >= 128 * sizeof(float); batch -= 128 * sizeof(float)) {
    HVX_Vector vx0 = *vptr++;
    HVX_Vector vx1 = *vptr++;
    HVX_Vector vx2 = *vptr++;
    HVX_Vector vx3 = *vptr++;

    HVX_Vector vacc0 = Q6_Vsf_vsub_VsfVsf(v0, vx0);
    HVX_Vector vacc1 = Q6_Vsf_vsub_VsfVsf(v0, vx1);
    HVX_Vector vacc2 = Q6_Vsf_vsub_VsfVsf(v0, vx2);
    HVX_Vector vacc3 = Q6_Vsf_vsub_VsfVsf(v0, vx3);

    *vptr_o++ = vacc0;
    *vptr_o++ = vacc1;
    *vptr_o++ = vacc2;
    *vptr_o++ = vacc3;
  }

  for (; batch >= 32 * sizeof(float); batch -= 32 * sizeof(float)) {
    HVX_Vector vx = *vptr++;

    HVX_Vector vacc = Q6_Vsf_vsub_VsfVsf(v0, vx);


    *vptr_o++ = vacc;
  }

  if XNN_UNLIKELY(batch != 0) {
    HVX_Vector vx = *vptr;

    HVX_Vector vacc = Q6_Vsf_vsub_VsfVsf(v0, vx);

    Q6_V_vstu_variable(vptr_o, batch, vacc);
  }
}
