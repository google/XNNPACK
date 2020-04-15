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


void xnn_f32_clamp_ukernel__scalar_x1(
    size_t n,
    const float* x,
    float* y,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(n != 0);
  assert(n % sizeof(float) == 0);

  const float vy_min = params->scalar.min;
  const float vy_max = params->scalar.max;

  for (; n >= sizeof(float); n -= sizeof(float)) {
    float vacc = *x++;
    vacc = math_max_f32(vacc, vy_min);
    vacc = math_min_f32(vacc, vy_max);
    *y++ = vacc;
  }
}
