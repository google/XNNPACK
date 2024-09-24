// Auto-generated file. Do not edit!
//   Template: src/f32-vbinary/vop-scalar.c.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include "xnnpack/common.h"
#include "xnnpack/math.h"
#include "xnnpack/vbinary.h"


void xnn_f32_vdiv_minmax_ukernel__scalar_u1(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const float voutput_min = params->scalar.min;
  const float voutput_max = params->scalar.max;

  for (; batch >= sizeof(float); batch -= sizeof(float)) {
    const float va = *input_a++;
    const float vb = *input_b++;
    float vacc = va / vb;
    vacc = math_max_f32(vacc, voutput_min);
    vacc = math_min_f32(vacc, voutput_max);
    *output++ = vacc;
  }
}
