// Auto-generated file. Do not edit!
//   Template: src/f32-vlrelu/scalar.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <xnnpack/common.h>
#include <xnnpack/vunary.h>


void xnn_f32_vlrelu_ukernel__scalar_x2(
    size_t n,
    const float* x,
    float* y,
    const union xnn_f32_lrelu_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(n != 0);
  assert(n % sizeof(float) == 0);

  const float vslope = params->scalar.slope;

  for (; n >= 2 * sizeof(float); n -= 2 * sizeof(float)) {
    const float vx0 = x[0];
    const float vx1 = x[1];
    x += 2;

    float vacc0 = vx0 * vslope;
    float vacc1 = vx1 * vslope;

    vacc0 = XNN_UNPREDICTABLE(vx0 < 0.0f) ? vacc0 : vx0;
    vacc1 = XNN_UNPREDICTABLE(vx1 < 0.0f) ? vacc1 : vx1;

    y[0] = vacc0;
    y[1] = vacc1;
    y += 2;
  }
  if XNN_UNLIKELY(n != 0) {
    const float vx = *x;
    float vacc = vx * vslope;
    vacc = XNN_UNPREDICTABLE(vx < 0.0f) ? vacc : vx;
    *y = vacc;
  }
}
