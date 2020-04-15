// Auto-generated file. Do not edit!
//   Template: src/f32-hswish/scalar.c.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <xnnpack/common.h>
#include <xnnpack/math.h>
#include <xnnpack/vbinary.h>


void xnn_f32_hswish_ukernel__wasm_x2(
    size_t n,
    const float* x,
    float* y,
    const union xnn_f32_hswish_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(n != 0);
  assert(n % sizeof(float) == 0);

  const float vsixth = params->scalar.sixth;
  const float vhalf = params->scalar.half;
  const float vone = params->scalar.one;
  assert(vhalf == 0.5f);
  assert(vone == 1.0f);

  for (; n >= 2 * sizeof(float); n -= 2 * sizeof(float)) {
    const float vx0 = x[0];
    const float vx1 = x[1];
    x += 2;

    float vacc0 = vx0 * vsixth + vhalf;
    float vacc1 = vx1 * vsixth + vhalf;

    vacc0 = __builtin_wasm_max_f32(vacc0, 0.0f);
    vacc1 = __builtin_wasm_max_f32(vacc1, 0.0f);

    vacc0 = __builtin_wasm_min_f32(vacc0, vone);
    vacc1 = __builtin_wasm_min_f32(vacc1, vone);

    vacc0 *= vx0;
    vacc1 *= vx1;

    y[0] = vacc0;
    y[1] = vacc1;
    y += 2;
  }
  if XNN_UNLIKELY(n != 0) {
    const float vx = *x;
    float vacc = vx * vsixth + vhalf;
    vacc = __builtin_wasm_max_f32(vacc, 0.0f);
    vacc = __builtin_wasm_min_f32(vacc, vone);
    vacc = vacc * vx;
    *y = vacc;
  }
}
