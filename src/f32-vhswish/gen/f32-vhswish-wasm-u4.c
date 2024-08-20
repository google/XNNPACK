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


void xnn_f32_vhswish_ukernel__wasm_u4(
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

  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    float vx0 = input[0];
    float vx1 = input[1];
    float vx2 = input[2];
    float vx3 = input[3];
    input += 4;

    float vacc0 = vx0 + vthree;
    vx0 *= vsixth;
    float vacc1 = vx1 + vthree;
    vx1 *= vsixth;
    float vacc2 = vx2 + vthree;
    vx2 *= vsixth;
    float vacc3 = vx3 + vthree;
    vx3 *= vsixth;

    vacc0 = __builtin_wasm_max_f32(vacc0, vzero);
    vacc1 = __builtin_wasm_max_f32(vacc1, vzero);
    vacc2 = __builtin_wasm_max_f32(vacc2, vzero);
    vacc3 = __builtin_wasm_max_f32(vacc3, vzero);

    vacc0 = __builtin_wasm_min_f32(vacc0, vsix);
    vacc1 = __builtin_wasm_min_f32(vacc1, vsix);
    vacc2 = __builtin_wasm_min_f32(vacc2, vsix);
    vacc3 = __builtin_wasm_min_f32(vacc3, vsix);

    vacc0 *= vx0;
    vacc1 *= vx1;
    vacc2 *= vx2;
    vacc3 *= vx3;

    output[0] = vacc0;
    output[1] = vacc1;
    output[2] = vacc2;
    output[3] = vacc3;
    output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    do {
      float vx = *input++;
      float vacc = vx + vthree;
      vx *= vsixth;
      vacc = __builtin_wasm_max_f32(vacc, vzero);
      vacc = __builtin_wasm_min_f32(vacc, vsix);
      vacc *= vx;
      *output++ = vacc;
      batch -= sizeof(float);
    } while (batch != 0);
  }
}
