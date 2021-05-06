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


void xnn_f32_vhswish_ukernel__wasm_x1(
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

  for (; n >= sizeof(float); n -= sizeof(float)) {
    float vx = *x++;
    float vacc = vx + vthree;
    vx *= vsixth;
    vacc = __builtin_wasm_max_f32(vacc, vzero);
    vacc = __builtin_wasm_min_f32(vacc, vsix);
    vacc *= vx;
    *y++ = vacc;
  }
}
