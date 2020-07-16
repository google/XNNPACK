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


void xnn_f32_vlrelu_ukernel__scalar_x4(
    size_t n,
    const float* x,
    float* y,
    const union xnn_f32_lrelu_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(n != 0);
  assert(n % sizeof(float) == 0);

  const float vslope = params->scalar.slope;

  for (; n >= 4 * sizeof(float); n -= 4 * sizeof(float)) {
    const float vx0 = x[0];
    const float vx1 = x[1];
    const float vx2 = x[2];
    const float vx3 = x[3];
    x += 4;

    float vacc0 = vx0 * vslope;
    float vacc1 = vx1 * vslope;
    float vacc2 = vx2 * vslope;
    float vacc3 = vx3 * vslope;

    vacc0 = XNN_UNPREDICTABLE(vx0 < 0.0f) ? vacc0 : vx0;
    vacc1 = XNN_UNPREDICTABLE(vx1 < 0.0f) ? vacc1 : vx1;
    vacc2 = XNN_UNPREDICTABLE(vx2 < 0.0f) ? vacc2 : vx2;
    vacc3 = XNN_UNPREDICTABLE(vx3 < 0.0f) ? vacc3 : vx3;

    y[0] = vacc0;
    y[1] = vacc1;
    y[2] = vacc2;
    y[3] = vacc3;
    y += 4;
  }
  if XNN_UNLIKELY(n != 0) {
    do {
      const float vx = *x++;
      float vacc = vx * vslope;
      vacc = XNN_UNPREDICTABLE(vx < 0.0f) ? vacc : vx;
      *y++ = vacc;
      n -= sizeof(float);
    } while (n != 0);
  }
}
