// Auto-generated file. Do not edit!
//   Template: src/f32-vbinary/vop-hvx.c.in
//   Generator: tools/xngen
//

#include <assert.h>

#include "xnnpack/simd/f32-hvx.h"

#include "xnnpack/math.h"
#include "xnnpack/vbinary.h"

void xnn_f32_vsqrdiff_ukernel__hvx_u64(
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
    HVX_Vector va0 = xnn_loadu_f32(input_a);
    HVX_Vector va1 = xnn_loadu_f32(input_a + 32);
    HVX_Vector vb0 = xnn_loadu_f32(input_b);
    HVX_Vector vb1 = xnn_loadu_f32(input_b + 32);
    input_a += 64;
    input_b += 64;

    HVX_Vector vacc0 = xnn_sub_f32(va0, vb0);
    HVX_Vector vacc1 = xnn_sub_f32(va1, vb1);

    vacc0 = xnn_mul_f32(vacc0, vacc0);
    vacc1 = xnn_mul_f32(vacc1, vacc1);


    xnn_storeu_f32(output, vacc0);
    xnn_storeu_f32(output + 32, vacc1);
    output += 64;
  }
  for (; batch >= 32 * sizeof(float); batch -= 32 * sizeof(float)) {
    HVX_Vector va = xnn_loadu_f32(input_a);
    HVX_Vector vb = xnn_loadu_f32(input_b);
    input_a += 32;
    input_b += 32;

    HVX_Vector vacc = xnn_sub_f32(va, vb);
    vacc = xnn_mul_f32(vacc, vacc);

    xnn_storeu_f32(output, vacc);
    output += 32;
  }
  if XNN_UNLIKELY(batch != 0) {
     HVX_Vector va = xnn_loadu_f32(input_a);
     HVX_Vector vb = xnn_loadu_f32(input_b);

     HVX_Vector vacc = xnn_sub_f32(va, vb);
     vacc = xnn_mul_f32(vacc, vacc);
     
     Q6_V_vstu_variable(output, batch, vacc);
  }
}
