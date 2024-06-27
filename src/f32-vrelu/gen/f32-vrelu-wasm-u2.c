// Auto-generated file. Do not edit!
//   Template: src/f32-vrelu/wasm.c.in
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


void xnn_f32_vrelu_ukernel__wasm_u2(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_relu_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const float vzero = 0.0f;

  for (; batch >= 2 * sizeof(float); batch -= 2 * sizeof(float)) {
    float vacc0 = input[0];
    float vacc1 = input[1];
    input += 2;

    vacc0 = __builtin_wasm_max_f32(vacc0, vzero);
    vacc1 = __builtin_wasm_max_f32(vacc1, vzero);

    output[0] = vacc0;
    output[1] = vacc1;
    output += 2;
  }
  if XNN_UNLIKELY(batch != 0) {
    float vacc = *input;
    vacc = __builtin_wasm_max_f32(vacc, vzero);
    *output = vacc;
  }
}
