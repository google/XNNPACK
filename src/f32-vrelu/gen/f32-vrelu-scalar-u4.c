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

void xnn_f32_vrelu_ukernel__scalar_u4(
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

  for (; batch >= 4 * sizeof(uint32_t); batch -= 4 * sizeof(uint32_t)) {
    uint32_t vacc0 = i[0];
    uint32_t vacc1 = i[1];
    uint32_t vacc2 = i[2];
    uint32_t vacc3 = i[3];
    i += 4;

    vacc0 = ((vacc0 >> 31) - 1) & vacc0;
    vacc1 = ((vacc1 >> 31) - 1) & vacc1;
    vacc2 = ((vacc2 >> 31) - 1) & vacc2;
    vacc3 = ((vacc3 >> 31) - 1) & vacc3;

    o[0] = vacc0;
    o[1] = vacc1;
    o[2] = vacc2;
    o[3] = vacc3;
    o += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    do {
      uint32_t vacc = *i++;
      vacc =  ((vacc >> 31) - 1) & vacc;
      *o++ = vacc;
      batch -= sizeof(uint32_t);
    } while (batch != 0);
  }
}
