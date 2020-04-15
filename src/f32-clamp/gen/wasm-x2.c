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


void xnn_f32_clamp_ukernel__wasm_x2(
    size_t n,
    const float* x,
    float* y,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(n != 0);
  assert(n % sizeof(float) == 0);

  const float vy_min = params->scalar.min;
  const float vy_max = params->scalar.max;

  for (; n >= 2 * sizeof(float); n -= 2 * sizeof(float)) {
    float vacc0 = x[0];
    float vacc1 = x[1];
    x += 2;

    vacc0 = __builtin_wasm_max_f32(vacc0, vy_min);
    vacc1 = __builtin_wasm_max_f32(vacc1, vy_min);

    vacc0 = __builtin_wasm_min_f32(vacc0, vy_max);
    vacc1 = __builtin_wasm_min_f32(vacc1, vy_max);

    y[0] = vacc0;
    y[1] = vacc1;
    y += 2;
  }
  if XNN_UNLIKELY(n != 0) {
    float vacc = *x;
    vacc = __builtin_wasm_max_f32(vacc, vy_min);
    vacc = __builtin_wasm_min_f32(vacc, vy_max);
    *y = vacc;
  }
}
