// Auto-generated file. Do not edit!
//   Template: src/f32-vbinary/vop-hvx.c.in
//   Generator: tools/xngen
//

#include <assert.h>

#include "xnnpack/simd/f32-hvx.h"

#include "xnnpack/math.h"
#include "xnnpack/vbinary.h"

void xnn_f32_vadd_minmax_ukernel__hvx_u32(
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

  const HVX_Vector voutput_min = xnn_set1_f32(params->scalar.min);
  const HVX_Vector voutput_max = xnn_set1_f32(params->scalar.max);

  for (; batch >= 32 * sizeof(float); batch -= 32 * sizeof(float)) {
    HVX_Vector va = xnn_loadu_f32(input_a);
    HVX_Vector vb = xnn_loadu_f32(input_b);
    input_a += 32;
    input_b += 32;

    HVX_Vector vacc = xnn_add_f32(va, vb);
    vacc = xnn_max_f32(vacc, voutput_min);
    vacc = xnn_min_f32(vacc, voutput_max);

    xnn_storeu_f32(output, vacc);
    output += 32;
  }
  if XNN_UNLIKELY(batch != 0) {
     HVX_Vector va = xnn_loadu_f32(input_a);
     HVX_Vector vb = xnn_loadu_f32(input_b);

     HVX_Vector vacc = xnn_add_f32(va, vb);
     vacc = xnn_max_f32(vacc, voutput_min);
     vacc = xnn_min_f32(vacc, voutput_max);
     
     Q6_V_vstu_variable(output, batch, vacc);
  }
}
