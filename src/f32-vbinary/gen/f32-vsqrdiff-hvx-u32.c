// Auto-generated file. Do not edit!
//   Template: src/f32-vbinary/vop-hvx.c.in
//   Generator: tools/xngen
//

#include <assert.h>

#include <hvx_hexagon_protos.h>
#include <hexagon_protos.h>
#include <hexagon_types.h>

#include "xnnpack/common.h"
#include "xnnpack/intrinsics-polyfill.h"
#include "xnnpack/math.h"
#include "xnnpack/vbinary.h"

void xnn_f32_vsqrdiff_ukernel__hvx_u32(
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


  const HVX_UVector *vptr_a = (const HVX_UVector *) input_a;
  const HVX_UVector *vptr_b = (const HVX_UVector *) input_b;
  HVX_UVector *vptr_o = (HVX_UVector*) output;

  for (; batch >= 32 * sizeof(float); batch -= 32 * sizeof(float)) {
    HVX_Vector va = *vptr_a++;
    HVX_Vector vb = *vptr_b++;

    HVX_Vector vacc = Q6_Vsf_vsub_VsfVsf(va, vb);
    vacc = Q6_Vsf_vmpy_VsfVsf(vacc, vacc);

    *vptr_o++ = vacc;
  }
  if XNN_UNLIKELY(batch != 0) {
     HVX_Vector va = *vptr_a;
     HVX_Vector vb = *vptr_b;

     HVX_Vector vacc = Q6_Vsf_vsub_VsfVsf(va, vb);
     vacc = Q6_Vsf_vmpy_VsfVsf(vacc, vacc);
     
     Q6_V_vstu_variable(vptr_o, batch, vacc);
  }
}
