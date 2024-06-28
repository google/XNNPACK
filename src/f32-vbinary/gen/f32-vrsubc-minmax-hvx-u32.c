// Auto-generated file. Do not edit!
//   Template: src/f32-vbinary/vopc-hvx.c.in
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

void xnn_f32_vrsubc_minmax_ukernel__hvx_u32(
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
  HVX_Vector vb = Q6_V_vsplat_R(*((int32_t*) input_b));

  for (; batch >= 32 * sizeof(float); batch -= 32 * sizeof(float)) {
    HVX_Vector va = *((HVX_UVector*) input_a);
    input_a += 32;

    HVX_Vector vacc = Q6_Vsf_vsub_VsfVsf(vb, va);
    vacc = Q6_Vsf_vmax_VsfVsf(vacc, voutput_min);
    vacc = Q6_Vsf_vmin_VsfVsf(vacc, voutput_max);

    *((HVX_UVector *) output) = vacc;
    output+= 32;
  }
  if XNN_UNLIKELY(batch != 0) {
    HVX_Vector va = *((HVX_UVector*) input_a);

    HVX_Vector vacc = Q6_Vsf_vsub_VsfVsf(vb, va);
    vacc = Q6_Vsf_vmax_VsfVsf(vacc, voutput_min);
    vacc = Q6_Vsf_vmin_VsfVsf(vacc, voutput_max);

    Q6_V_vstu_variable(output, batch, vacc);
  }
}
