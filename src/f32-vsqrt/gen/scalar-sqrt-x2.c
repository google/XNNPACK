// Auto-generated file. Do not edit!
//   Template: src/f32-vsqrt/scalar-sqrt.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <math.h>

#include <xnnpack/common.h>
#include <xnnpack/vunary.h>


void xnn_f32_vsqrt_ukernel__scalar_sqrt_x2(
    size_t n,
    const float* x,
    float* y,
    const union xnn_f32_sqrt_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(n != 0);
  assert(n % sizeof(float) == 0);

  for (; n >= 2 * sizeof(float); n -= 2 * sizeof(float)) {
    const float vx0 = x[0];
    const float vx1 = x[1];
    x += 2;

    const float vy0 = sqrtf(vx0);
    const float vy1 = sqrtf(vx1);

    y[0] = vy0;
    y[1] = vy1;
    y += 2;
  }
  if XNN_UNLIKELY(n != 0) {
    const float vx = *x;
    const float vy = sqrtf(vx);
    *y = vy;
  }
}
