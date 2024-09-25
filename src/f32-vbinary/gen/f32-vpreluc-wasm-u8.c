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


void xnn_f32_vpreluc_ukernel__wasm_u8(
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

  const float vb = *input_b;

  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    const float va0 = input_a[0];
    const float va1 = input_a[1];
    const float va2 = input_a[2];
    const float va3 = input_a[3];
    const float va4 = input_a[4];
    const float va5 = input_a[5];
    const float va6 = input_a[6];
    const float va7 = input_a[7];
    input_a += 8;

    float vacc0 = va0 * vb;
    float vacc1 = va1 * vb;
    float vacc2 = va2 * vb;
    float vacc3 = va3 * vb;
    float vacc4 = va4 * vb;
    float vacc5 = va5 * vb;
    float vacc6 = va6 * vb;
    float vacc7 = va7 * vb;

    vacc0 = XNN_UNPREDICTABLE(va0 < 0.0f) ? vacc0 : va0;
    vacc1 = XNN_UNPREDICTABLE(va1 < 0.0f) ? vacc1 : va1;
    vacc2 = XNN_UNPREDICTABLE(va2 < 0.0f) ? vacc2 : va2;
    vacc3 = XNN_UNPREDICTABLE(va3 < 0.0f) ? vacc3 : va3;
    vacc4 = XNN_UNPREDICTABLE(va4 < 0.0f) ? vacc4 : va4;
    vacc5 = XNN_UNPREDICTABLE(va5 < 0.0f) ? vacc5 : va5;
    vacc6 = XNN_UNPREDICTABLE(va6 < 0.0f) ? vacc6 : va6;
    vacc7 = XNN_UNPREDICTABLE(va7 < 0.0f) ? vacc7 : va7;

    output[0] = vacc0;
    output[1] = vacc1;
    output[2] = vacc2;
    output[3] = vacc3;
    output[4] = vacc4;
    output[5] = vacc5;
    output[6] = vacc6;
    output[7] = vacc7;
    output += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    do {
      const float va = *input_a++;
      float vacc = va * vb;
      vacc = XNN_UNPREDICTABLE(va < 0.0f) ? vacc : va;
      *output++ = vacc;
      batch -= sizeof(float);
    } while (batch != 0);
  }
}
