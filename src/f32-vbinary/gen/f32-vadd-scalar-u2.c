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


void xnn_f32_vadd_ukernel__scalar_u2(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const struct xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  for (; batch >= 2 * sizeof(float); batch -= 2 * sizeof(float)) {
    const float va0 = input_a[0];
    const float va1 = input_a[1];
    input_a += 2;

    const float vb0 = input_b[0];
    const float vb1 = input_b[1];
    input_b += 2;

    float vacc0 = va0 + vb0;
    float vacc1 = va1 + vb1;


    output[0] = vacc0;
    output[1] = vacc1;
    output += 2;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch == sizeof(float));
    const float va = *input_a;
    const float vb = *input_b;
    float vacc = va + vb;
    *output = vacc;
  }
}
