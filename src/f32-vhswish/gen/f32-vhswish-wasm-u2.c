// Auto-generated file. Do not edit!
//   Template: src/f32-vhswish/scalar.c.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include "xnnpack/common.h"
#include "xnnpack/math.h"
#include "xnnpack/vunary.h"


void xnn_f32_vhswish_ukernel__wasm_u2(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_hswish_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const float vsixth = 0x1.555556p-3f;
  const float vthree = 3.0f;
  const float vsix = 6.0f;
  const float vzero = 0.0f;

  for (; batch >= 2 * sizeof(float); batch -= 2 * sizeof(float)) {
    float vx0 = input[0];
    float vx1 = input[1];
    input += 2;

    float vacc0 = vx0 + vthree;
    vx0 *= vsixth;
    float vacc1 = vx1 + vthree;
    vx1 *= vsixth;

    vacc0 = __builtin_wasm_max_f32(vacc0, vzero);
    vacc1 = __builtin_wasm_max_f32(vacc1, vzero);

    vacc0 = __builtin_wasm_min_f32(vacc0, vsix);
    vacc1 = __builtin_wasm_min_f32(vacc1, vsix);

    vacc0 *= vx0;
    vacc1 *= vx1;

    output[0] = vacc0;
    output[1] = vacc1;
    output += 2;
  }
  if XNN_UNLIKELY(batch != 0) {
    float vx = *input;
    float vacc = vx + vthree;
    vx *= vsixth;
    vacc = __builtin_wasm_max_f32(vacc, vzero);
    vacc = __builtin_wasm_min_f32(vacc, vsix);
    vacc *= vx;
    *output = vacc;
  }
}
