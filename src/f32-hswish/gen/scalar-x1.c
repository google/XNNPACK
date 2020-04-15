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


void xnn_f32_hswish_ukernel__scalar_x1(
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

  for (; n >= sizeof(float); n -= sizeof(float)) {
    const float vx = *x++;
    float vacc = vx * vsixth + vhalf;
    vacc = math_max_f32(vacc, 0.0f);
    vacc = math_min_f32(vacc, vone);
    vacc = vacc * vx;
    *y++ = vacc;
  }
}
