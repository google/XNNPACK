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

void xnn_f32_vrelu_ukernel__scalar_u8(
    size_t batch,
    const float* input,
    float* output,
    const struct xnn_f32_relu_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const uint32_t* i = (const uint32_t*) input;
  uint32_t* o = (uint32_t*) output;

  for (; batch >= 8 * sizeof(uint32_t); batch -= 8 * sizeof(uint32_t)) {
    uint32_t vacc0 = i[0];
    uint32_t vacc1 = i[1];
    uint32_t vacc2 = i[2];
    uint32_t vacc3 = i[3];
    uint32_t vacc4 = i[4];
    uint32_t vacc5 = i[5];
    uint32_t vacc6 = i[6];
    uint32_t vacc7 = i[7];
    i += 8;

    vacc0 = ((vacc0 >> 31) - 1) & vacc0;
    vacc1 = ((vacc1 >> 31) - 1) & vacc1;
    vacc2 = ((vacc2 >> 31) - 1) & vacc2;
    vacc3 = ((vacc3 >> 31) - 1) & vacc3;
    vacc4 = ((vacc4 >> 31) - 1) & vacc4;
    vacc5 = ((vacc5 >> 31) - 1) & vacc5;
    vacc6 = ((vacc6 >> 31) - 1) & vacc6;
    vacc7 = ((vacc7 >> 31) - 1) & vacc7;

    o[0] = vacc0;
    o[1] = vacc1;
    o[2] = vacc2;
    o[3] = vacc3;
    o[4] = vacc4;
    o[5] = vacc5;
    o[6] = vacc6;
    o[7] = vacc7;
    o += 8;
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
