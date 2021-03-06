// Auto-generated file. Do not edit!
//   Template: src/f32-vrelu/wasm.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <xnnpack/vunary.h>
#include <xnnpack/common.h>
#include <xnnpack/math.h>


void xnn_f32_vrelu_ukernel__wasm_x8(
    size_t n,
    const float* x,
    float* y,
    const union xnn_f32_relu_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(n != 0);
  assert(n % sizeof(float) == 0);
  assert(x != NULL);
  assert(y != NULL);

  const float vzero = 0.0f;

  for (; n >= 8 * sizeof(float); n -= 8 * sizeof(float)) {
    float vacc0 = x[0];
    float vacc1 = x[1];
    float vacc2 = x[2];
    float vacc3 = x[3];
    float vacc4 = x[4];
    float vacc5 = x[5];
    float vacc6 = x[6];
    float vacc7 = x[7];
    x += 8;

    vacc0 = __builtin_wasm_max_f32(vacc0, vzero);
    vacc1 = __builtin_wasm_max_f32(vacc1, vzero);
    vacc2 = __builtin_wasm_max_f32(vacc2, vzero);
    vacc3 = __builtin_wasm_max_f32(vacc3, vzero);
    vacc4 = __builtin_wasm_max_f32(vacc4, vzero);
    vacc5 = __builtin_wasm_max_f32(vacc5, vzero);
    vacc6 = __builtin_wasm_max_f32(vacc6, vzero);
    vacc7 = __builtin_wasm_max_f32(vacc7, vzero);

    y[0] = vacc0;
    y[1] = vacc1;
    y[2] = vacc2;
    y[3] = vacc3;
    y[4] = vacc4;
    y[5] = vacc5;
    y[6] = vacc6;
    y[7] = vacc7;
    y += 8;
  }
  if XNN_UNLIKELY(n != 0) {
    do {
      float vacc = *x++;
      vacc = __builtin_wasm_max_f32(vacc, vzero);
      *y++ = vacc;
      n -= sizeof(float);
    } while (n != 0);
  }
}
