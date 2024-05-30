// Auto-generated file. Do not edit!
//   Template: src/f32-vbinary/vop-hvx.c.in
//   Generator: tools/xngen
//

#include <assert.h>

#include <hvx_hexagon_protos.h>
#include <hexagon_protos.h>
#include <hexagon_types.h>

#include <xnnpack/common.h>
#include <xnnpack/intrinsics-polyfill.h>
#include <xnnpack/math.h>
#include <xnnpack/vbinary.h>

void xnn_f32_vadd_minmax_ukernel__hvx_u128(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const HVX_Vector voutput_min = Q6_V_vsplat_R(*((uint32_t *) &params->scalar.min));
  const HVX_Vector voutput_max = Q6_V_vsplat_R(*((uint32_t *) &params->scalar.max));

  const HVX_UVector *vptr_a = (const HVX_UVector *) input_a;
  const HVX_UVector *vptr_b = (const HVX_UVector *) input_b;
  HVX_UVector *vptr_o = (HVX_UVector*) output;

  for (; batch >= 128 * sizeof(float); batch -= 128 * sizeof(float)) {
    HVX_Vector va0 = *vptr_a++;
    HVX_Vector va1 = *vptr_a++;
    HVX_Vector va2 = *vptr_a++;
    HVX_Vector va3 = *vptr_a++;
    HVX_Vector vb0 = *vptr_b++;
    HVX_Vector vb1 = *vptr_b++;
    HVX_Vector vb2 = *vptr_b++;
    HVX_Vector vb3 = *vptr_b++;

    HVX_Vector vacc0 = Q6_Vsf_vadd_VsfVsf(va0, vb0);
    HVX_Vector vacc1 = Q6_Vsf_vadd_VsfVsf(va1, vb1);
    HVX_Vector vacc2 = Q6_Vsf_vadd_VsfVsf(va2, vb2);
    HVX_Vector vacc3 = Q6_Vsf_vadd_VsfVsf(va3, vb3);

    vacc0 = Q6_Vsf_vmax_VsfVsf(vacc0, voutput_min);
    vacc1 = Q6_Vsf_vmax_VsfVsf(vacc1, voutput_min);
    vacc2 = Q6_Vsf_vmax_VsfVsf(vacc2, voutput_min);
    vacc3 = Q6_Vsf_vmax_VsfVsf(vacc3, voutput_min);

    vacc0 = Q6_Vsf_vmin_VsfVsf(vacc0, voutput_max);
    vacc1 = Q6_Vsf_vmin_VsfVsf(vacc1, voutput_max);
    vacc2 = Q6_Vsf_vmin_VsfVsf(vacc2, voutput_max);
    vacc3 = Q6_Vsf_vmin_VsfVsf(vacc3, voutput_max);

    *vptr_o++ = vacc0;
    *vptr_o++ = vacc1;
    *vptr_o++ = vacc2;
    *vptr_o++ = vacc3;
  }
  for (; batch >= 32 * sizeof(float); batch -= 32 * sizeof(float)) {
    HVX_Vector va = *vptr_a++;
    HVX_Vector vb = *vptr_b++;

    HVX_Vector vacc = Q6_Vsf_vadd_VsfVsf(va, vb);
    vacc = Q6_Vsf_vmax_VsfVsf(vacc, voutput_min);
    vacc = Q6_Vsf_vmin_VsfVsf(vacc, voutput_max);

    *vptr_o++ = vacc;
  }
  if XNN_UNLIKELY(batch != 0) {
     HVX_Vector va = *vptr_a;
     HVX_Vector vb = *vptr_b;

     HVX_Vector vacc = Q6_Vsf_vadd_VsfVsf(va, vb);
     vacc = Q6_Vsf_vmax_VsfVsf(vacc, voutput_min);
     vacc = Q6_Vsf_vmin_VsfVsf(vacc, voutput_max);
     
     Q6_V_vstu_variable(vptr_o, batch, vacc);
  }
}
