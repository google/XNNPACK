// Auto-generated file. Do not edit!
//   Template: src/f32-vhswish/scalar.c.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <xnnpack/common.h>
#include <xnnpack/math.h>
#include <xnnpack/vunary.h>


void xnn_f32_vhswish_ukernel__wasm_x4(
    size_t n,
    const float* x,
    float* y,
    const union xnn_f32_hswish_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(n != 0);
  assert(n % sizeof(float) == 0);

  const float vsixth = params->scalar.sixth;
  const float vthree = params->scalar.three;
  const float vsix = params->scalar.six;
  const float vzero = 0.0f;
  assert(vthree == 3.0f);
  assert(vsix == 6.0f);

  for (; n >= 4 * sizeof(float); n -= 4 * sizeof(float)) {
    float vx0 = x[0];
    float vx1 = x[1];
    float vx2 = x[2];
    float vx3 = x[3];
    x += 4;

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

    y[0] = vacc0;
    y[1] = vacc1;
    y[2] = vacc2;
    y[3] = vacc3;
    y += 4;
  }
  if XNN_UNLIKELY(n != 0) {
    do {
      float vx = *x++;
      float vacc = vx + vthree;
      vx *= vsixth;
      vacc = __builtin_wasm_max_f32(vacc, vzero);
      vacc = __builtin_wasm_min_f32(vacc, vsix);
      vacc *= vx;
      *y++ = vacc;
      n -= sizeof(float);
    } while (n != 0);
  }
}
