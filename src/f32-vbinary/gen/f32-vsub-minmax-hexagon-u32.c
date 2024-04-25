// Auto-generated file. Do not edit!
//   Template: src/f32-vbinary/vop-hexagon.c.in
//   Generator: tools/xngen
//

#include <assert.h>

#include <hexagon_protos.h>
#include <hexagon_types.h>

#include <xnnpack/common.h>
#include <xnnpack/math.h>
#include <xnnpack/vbinary.h>

void xnn_f32_vsub_minmax_ukernel__hexagon_u32(
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

  const HVX_Vector voutput_min = Q6_V_vsplat_R(params->scalar.min);
  const HVX_Vector voutput_max = Q6_V_vsplat_R(params->scalar.max);

  for (; batch >= 32 * sizeof(float); batch -= 32 * sizeof(float)) {
    const HVX_Vector va = *((const HVX_Vector*) input_a); input_a += 32;
    const HVX_Vector vb = *((const HVX_Vector*) input_b); input_b += 32;

    HVX_Vector vacc = Q6_Vqf32_vsub_VsfVsf(va, vb);
    vacc = Q6_Vsf_vmax_VsfVsf(vacc, voutput_min);
    vacc = Q6_Vsf_vmin_VsfVsf(vacc, voutput_max);

    *((HVX_Vector*) output) = vacc; output += 32;
  }
  if XNN_UNLIKELY(batch != 0) {
     const HVX_Vector va = *((const HVX_Vector*) input_a);
     const HVX_Vector vb = *((const HVX_Vector*) input_b);

     HVX_Vector vacc = Q6_Vqf32_vsub_VsfVsf(va, vb);
     vacc = Q6_Vsf_vmax_VsfVsf(vacc, voutput_min);
     vacc = Q6_Vsf_vmin_VsfVsf(vacc, voutput_max);

     *((HVX_Vector*) output) = vacc;
  }
}
