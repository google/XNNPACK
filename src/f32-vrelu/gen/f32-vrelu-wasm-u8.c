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


void xnn_f32_vrelu_ukernel__wasm_u8(
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

  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    float vacc0 = input[0];
    float vacc1 = input[1];
    float vacc2 = input[2];
    float vacc3 = input[3];
    float vacc4 = input[4];
    float vacc5 = input[5];
    float vacc6 = input[6];
    float vacc7 = input[7];
    input += 8;

    vacc0 = __builtin_wasm_max_f32(vacc0, vzero);
    vacc1 = __builtin_wasm_max_f32(vacc1, vzero);
    vacc2 = __builtin_wasm_max_f32(vacc2, vzero);
    vacc3 = __builtin_wasm_max_f32(vacc3, vzero);
    vacc4 = __builtin_wasm_max_f32(vacc4, vzero);
    vacc5 = __builtin_wasm_max_f32(vacc5, vzero);
    vacc6 = __builtin_wasm_max_f32(vacc6, vzero);
    vacc7 = __builtin_wasm_max_f32(vacc7, vzero);

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
      float vacc = *input++;
      vacc = __builtin_wasm_max_f32(vacc, vzero);
      *output++ = vacc;
      batch -= sizeof(float);
    } while (batch != 0);
  }
}
