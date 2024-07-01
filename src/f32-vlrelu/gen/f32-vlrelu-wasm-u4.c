// Auto-generated file. Do not edit!
//   Template: src/f32-vlrelu/wasm.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include "xnnpack/common.h"
#include "xnnpack/vunary.h"


void xnn_f32_vlrelu_ukernel__wasm_u4(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_lrelu_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const float vslope = params->scalar.slope;
  const float vzero = 0.0f;

  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    const float vx0 = input[0];
    const float vx1 = input[1];
    const float vx2 = input[2];
    const float vx3 = input[3];
    input += 4;

    const float vnegx0 = __builtin_wasm_min_f32(vx0, vzero);
    const float vnegx1 = __builtin_wasm_min_f32(vx1, vzero);
    const float vnegx2 = __builtin_wasm_min_f32(vx2, vzero);
    const float vnegx3 = __builtin_wasm_min_f32(vx3, vzero);

    float vacc0 = vnegx0 * vslope;
    const float vposx0 = __builtin_wasm_max_f32(vx0, vzero);
    float vacc1 = vnegx1 * vslope;
    const float vposx1 = __builtin_wasm_max_f32(vx1, vzero);
    float vacc2 = vnegx2 * vslope;
    const float vposx2 = __builtin_wasm_max_f32(vx2, vzero);
    float vacc3 = vnegx3 * vslope;
    const float vposx3 = __builtin_wasm_max_f32(vx3, vzero);

    vacc0 += vposx0;
    vacc1 += vposx1;
    vacc2 += vposx2;
    vacc3 += vposx3;

    output[0] = vacc0;
    output[1] = vacc1;
    output[2] = vacc2;
    output[3] = vacc3;
    output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    do {
      const float vx = *input++;
      const float vnegx = __builtin_wasm_min_f32(vx, vzero);
      float vacc = vnegx * vslope;
      const float vposx = __builtin_wasm_max_f32(vx, vzero);
      vacc += vposx;
      *output++ = vacc;
      batch -= sizeof(float);
    } while (batch != 0);
  }
}
