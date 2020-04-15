// Auto-generated file. Do not edit!
//   Template: src/f32-clamp/scalar.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <xnnpack/clamp.h>
#include <xnnpack/common.h>
#include <xnnpack/math.h>


void xnn_f32_clamp_ukernel__wasm_x4(
    size_t n,
    const float* x,
    float* y,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(n != 0);
  assert(n % sizeof(float) == 0);

  const float vy_min = params->scalar.min;
  const float vy_max = params->scalar.max;

  for (; n >= 4 * sizeof(float); n -= 4 * sizeof(float)) {
    float vacc0 = x[0];
    float vacc1 = x[1];
    float vacc2 = x[2];
    float vacc3 = x[3];
    x += 4;

    vacc0 = __builtin_wasm_max_f32(vacc0, vy_min);
    vacc1 = __builtin_wasm_max_f32(vacc1, vy_min);
    vacc2 = __builtin_wasm_max_f32(vacc2, vy_min);
    vacc3 = __builtin_wasm_max_f32(vacc3, vy_min);

    vacc0 = __builtin_wasm_min_f32(vacc0, vy_max);
    vacc1 = __builtin_wasm_min_f32(vacc1, vy_max);
    vacc2 = __builtin_wasm_min_f32(vacc2, vy_max);
    vacc3 = __builtin_wasm_min_f32(vacc3, vy_max);

    y[0] = vacc0;
    y[1] = vacc1;
    y[2] = vacc2;
    y[3] = vacc3;
    y += 4;
  }
  if XNN_UNLIKELY(n != 0) {
    do {
      float vacc = *x++;
      vacc = __builtin_wasm_max_f32(vacc, vy_min);
      vacc = __builtin_wasm_min_f32(vacc, vy_max);
      *y++ = vacc;
      n -= sizeof(float);
    } while (n != 0);
  }
}
