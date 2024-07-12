// Auto-generated file. Do not edit!
//   Template: src/f32-vbinary/vopc-hvx.c.in
//   Generator: tools/xngen
//

#include <assert.h>

#include "xnnpack/simd/f32-hvx.h"

#include "xnnpack/math.h"
#include "xnnpack/vbinary.h"

void xnn_f32_vminc_ukernel__hvx_u128(
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

  HVX_Vector vb = xnn_set1_f32(*input_b);

  for (; batch >= 128 * sizeof(float); batch -= 128 * sizeof(float)) {
    HVX_Vector va0 = xnn_loadu_f32(input_a);
    HVX_Vector va1 = xnn_loadu_f32(input_a + 32);
    HVX_Vector va2 = xnn_loadu_f32(input_a + 64);
    HVX_Vector va3 = xnn_loadu_f32(input_a + 96);
    input_a += 128;

    HVX_Vector vacc0 = xnn_min_f32(va0, vb);
    HVX_Vector vacc1 = xnn_min_f32(va1, vb);
    HVX_Vector vacc2 = xnn_min_f32(va2, vb);
    HVX_Vector vacc3 = xnn_min_f32(va3, vb);



   xnn_storeu_f32(output, vacc0);
    xnn_storeu_f32(output + 32, vacc1);
    xnn_storeu_f32(output + 64, vacc2);
    xnn_storeu_f32(output + 96, vacc3);
    output += 128;
  }
  for (; batch >= 32 * sizeof(float); batch -= 32 * sizeof(float)) {
    HVX_Vector va = xnn_loadu_f32(input_a);
    input_a += 32;

    HVX_Vector vacc = xnn_min_f32(va, vb);

    xnn_storeu_f32(output, vacc);
    output+= 32;
  }
  if XNN_UNLIKELY(batch != 0) {
    HVX_Vector va = xnn_loadu_f32(input_a);

    HVX_Vector vacc = xnn_min_f32(va, vb);

    Q6_V_vstu_variable(output, batch, vacc);
  }
}
