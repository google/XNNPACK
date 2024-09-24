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


void xnn_f32_vrsubc_minmax_ukernel__scalar_u8(
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

    float vacc0 = vb - va0;
    float vacc1 = vb - va1;
    float vacc2 = vb - va2;
    float vacc3 = vb - va3;
    float vacc4 = vb - va4;
    float vacc5 = vb - va5;
    float vacc6 = vb - va6;
    float vacc7 = vb - va7;


    vacc0 = math_max_f32(vacc0, voutput_min);
    vacc1 = math_max_f32(vacc1, voutput_min);
    vacc2 = math_max_f32(vacc2, voutput_min);
    vacc3 = math_max_f32(vacc3, voutput_min);
    vacc4 = math_max_f32(vacc4, voutput_min);
    vacc5 = math_max_f32(vacc5, voutput_min);
    vacc6 = math_max_f32(vacc6, voutput_min);
    vacc7 = math_max_f32(vacc7, voutput_min);

    vacc0 = math_min_f32(vacc0, voutput_max);
    vacc1 = math_min_f32(vacc1, voutput_max);
    vacc2 = math_min_f32(vacc2, voutput_max);
    vacc3 = math_min_f32(vacc3, voutput_max);
    vacc4 = math_min_f32(vacc4, voutput_max);
    vacc5 = math_min_f32(vacc5, voutput_max);
    vacc6 = math_min_f32(vacc6, voutput_max);
    vacc7 = math_min_f32(vacc7, voutput_max);

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
      float vacc = vb - va;
      vacc = math_max_f32(vacc, voutput_min);
      vacc = math_min_f32(vacc, voutput_max);
      *output++ = vacc;
      batch -= sizeof(float);
    } while (batch != 0);
  }
}
