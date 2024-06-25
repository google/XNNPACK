// Auto-generated file. Do not edit!
//   Template: src/f32-vclamp/scalar.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include "xnnpack/common.h"
#include "xnnpack/math.h"
#include "xnnpack/vunary.h"


void xnn_f32_vclamp_ukernel__scalar_u2(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const float vy_min = params->scalar.min;
  const float vy_max = params->scalar.max;

  for (; batch >= 2 * sizeof(float); batch -= 2 * sizeof(float)) {
    float vacc0 = input[0];
    float vacc1 = input[1];
    input += 2;

    vacc0 = math_max_f32(vacc0, vy_min);
    vacc1 = math_max_f32(vacc1, vy_min);

    vacc0 = math_min_f32(vacc0, vy_max);
    vacc1 = math_min_f32(vacc1, vy_max);

    output[0] = vacc0;
    output[1] = vacc1;
    output += 2;
  }
  if XNN_UNLIKELY(batch != 0) {
    float vacc = *input;
    vacc = math_max_f32(vacc, vy_min);
    vacc = math_min_f32(vacc, vy_max);
    *output = vacc;
  }
}
