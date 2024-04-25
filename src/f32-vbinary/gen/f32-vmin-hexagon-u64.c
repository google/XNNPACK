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

void xnn_f32_vmin_ukernel__hexagon_u64(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);


  for (; batch >= 64 * sizeof(float); batch -= 64 * sizeof(float)) {
    const HVX_Vector va0 = *((const HVX_Vector*) input_a); input_a += 32;
    const HVX_Vector vb0 = *((const HVX_Vector*) input_b); input_b += 32;
    const HVX_Vector va1 = *((const HVX_Vector*) input_a); input_a += 32;
    const HVX_Vector vb1 = *((const HVX_Vector*) input_b); input_b += 32;

    HVX_Vector vacc0 = Q6_Vsf_vmin_VsfVsf(va0, vb0);
    HVX_Vector vacc1 = Q6_Vsf_vmin_VsfVsf(va1, vb1);



    *((HVX_Vector*) output) = vacc0; output += 32;
    *((HVX_Vector*) output) = vacc1; output += 32;
  }
  for (; batch >= 32 * sizeof(float); batch -= 32 * sizeof(float)) {
    const HVX_Vector va = *((const HVX_Vector*) input_a); input_a += 32;
    const HVX_Vector vb = *((const HVX_Vector*) input_b); input_b += 32;

    HVX_Vector vacc = Q6_Vsf_vmin_VsfVsf(va, vb);

    *((HVX_Vector*) output) = vacc; output += 32;
  }
  if XNN_UNLIKELY(batch != 0) {
     const HVX_Vector va = *((const HVX_Vector*) input_a);
     const HVX_Vector vb = *((const HVX_Vector*) input_b);

     HVX_Vector vacc = Q6_Vsf_vmin_VsfVsf(va, vb);

     *((HVX_Vector*) output) = vacc;
  }
}
