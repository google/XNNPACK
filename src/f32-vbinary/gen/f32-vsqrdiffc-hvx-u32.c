// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f32-vbinary/vopc-hvx.c.in
//   Generator: tools/xngen
//

#include <assert.h>

#include "src/xnnpack/simd/f32-hvx.h"

#include "src/xnnpack/math.h"
#include "src/xnnpack/vbinary.h"

void xnn_f32_vsqrdiffc_ukernel__hvx_u32(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const struct xnn_f32_default_params* restrict params)
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  HVX_Vector vb = xnn_set1_f32(*input_b);

  for (; batch >= 32 * sizeof(float); batch -= 32 * sizeof(float)) {
    HVX_Vector va = xnn_loadu_f32(input_a);
    input_a += 32;

    HVX_Vector vacc = xnn_sub_f32(va, vb);
    vacc = xnn_mul_f32(vacc, vacc);

    xnn_storeu_f32(output, vacc);
    output+= 32;
  }
  if XNN_UNLIKELY(batch != 0) {
    HVX_Vector va = xnn_load_tail_f32(input_a, batch >> XNN_LOG2_SIZEOF_FLOAT);

    HVX_Vector vacc = xnn_sub_f32(va, vb);
    vacc = xnn_mul_f32(vacc, vacc);

    Q6_V_vstu_variable(output, batch, vacc);
  }
}
