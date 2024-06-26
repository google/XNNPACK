// Auto-generated file. Do not edit!
//   Template: src/f32-vbinary/vopc-scalar.c.in
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


void xnn_f32_vrsubc_relu_ukernel__scalar_u4(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const union xnn_f32_relu_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const float vb = *input_b;

  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    const float va0 = input_a[0];
    const float va1 = input_a[1];
    const float va2 = input_a[2];
    const float va3 = input_a[3];
    input_a += 4;

    float vacc0 = vb - va0;
    float vacc1 = vb - va1;
    float vacc2 = vb - va2;
    float vacc3 = vb - va3;


    vacc0 = math_max_f32(vacc0, 0.0f);
    vacc1 = math_max_f32(vacc1, 0.0f);
    vacc2 = math_max_f32(vacc2, 0.0f);
    vacc3 = math_max_f32(vacc3, 0.0f);

    output[0] = vacc0;
    output[1] = vacc1;
    output[2] = vacc2;
    output[3] = vacc3;
    output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    do {
      const float va = *input_a++;
      float vacc = vb - va;
      vacc = math_max_f32(vacc, 0.0f);
      *output++ = vacc;
      batch -= sizeof(float);
    } while (batch != 0);
  }
}
