// Auto-generated file. Do not edit!
//   Template: src/f32-vbinary/vop-hvx.c.in
//   Generator: tools/xngen
//

#include <assert.h>

#include "xnnpack/simd/f32-hvx.h"

#include "xnnpack/math.h"
#include "xnnpack/vbinary.h"

void xnn_f32_vsub_minmax_ukernel__hvx_u128(
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

  for (; batch >= 128 * sizeof(float); batch -= 128 * sizeof(float)) {
    HVX_Vector va0 = xnn_loadu_f32(input_a);
    HVX_Vector va1 = xnn_loadu_f32(input_a + 32);
    HVX_Vector va2 = xnn_loadu_f32(input_a + 64);
    HVX_Vector va3 = xnn_loadu_f32(input_a + 96);
    HVX_Vector vb0 = xnn_loadu_f32(input_b);
    HVX_Vector vb1 = xnn_loadu_f32(input_b + 32);
    HVX_Vector vb2 = xnn_loadu_f32(input_b + 64);
    HVX_Vector vb3 = xnn_loadu_f32(input_b + 96);
    input_a += 128;
    input_b += 128;

    HVX_Vector vacc0 = xnn_sub_f32(va0, vb0);
    HVX_Vector vacc1 = xnn_sub_f32(va1, vb1);
    HVX_Vector vacc2 = xnn_sub_f32(va2, vb2);
    HVX_Vector vacc3 = xnn_sub_f32(va3, vb3);


    vacc0 = xnn_max_f32(vacc0, voutput_min);
    vacc1 = xnn_max_f32(vacc1, voutput_min);
    vacc2 = xnn_max_f32(vacc2, voutput_min);
    vacc3 = xnn_max_f32(vacc3, voutput_min);

    vacc0 = xnn_min_f32(vacc0, voutput_max);
    vacc1 = xnn_min_f32(vacc1, voutput_max);
    vacc2 = xnn_min_f32(vacc2, voutput_max);
    vacc3 = xnn_min_f32(vacc3, voutput_max);

    xnn_storeu_f32(output, vacc0);
    xnn_storeu_f32(output + 32, vacc1);
    xnn_storeu_f32(output + 64, vacc2);
    xnn_storeu_f32(output + 96, vacc3);
    output += 128;
  }
  for (; batch >= 32 * sizeof(float); batch -= 32 * sizeof(float)) {
    HVX_Vector va = xnn_loadu_f32(input_a);
    HVX_Vector vb = xnn_loadu_f32(input_b);
    input_a += 32;
    input_b += 32;

    HVX_Vector vacc = xnn_sub_f32(va, vb);
    vacc = xnn_max_f32(vacc, voutput_min);
    vacc = xnn_min_f32(vacc, voutput_max);

    xnn_storeu_f32(output, vacc);
    output += 32;
  }
  if XNN_UNLIKELY(batch != 0) {
     HVX_Vector va = xnn_loadu_f32(input_a);
     HVX_Vector vb = xnn_loadu_f32(input_b);

     HVX_Vector vacc = xnn_sub_f32(va, vb);
     vacc = xnn_max_f32(vacc, voutput_min);
     vacc = xnn_min_f32(vacc, voutput_max);
     
     Q6_V_vstu_variable(output, batch, vacc);
  }
}
