// Auto-generated file. Do not edit!
//   Template: src/f32-vrelu/scalar.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include "xnnpack/vunary.h"
#include "xnnpack/common.h"
#include "xnnpack/math.h"

void xnn_f32_vrelu_ukernel__scalar_u1(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_relu_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const uint32_t* i = (const uint32_t*) input;
  uint32_t* o = (uint32_t*) output;

  for (; batch >= sizeof(uint32_t); batch -= sizeof(uint32_t)) {
    uint32_t vacc = *i++;
    vacc =  ((vacc >> 31) - 1) & vacc;
    *o++ = vacc;
  }
}
