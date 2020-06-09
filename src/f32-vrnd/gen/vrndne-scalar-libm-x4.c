// Auto-generated file. Do not edit!
//   Template: src/f32-vrnd/scalar-libm.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <math.h>

#include <xnnpack/common.h>
#include <xnnpack/math.h>
#include <xnnpack/vunary.h>


void xnn_f32_vrndne_ukernel__scalar_libm_x4(
    size_t n,
    const float* x,
    float* y,
    const union xnn_f32_rnd_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(n != 0);
  assert(n % sizeof(float) == 0);

  for (; n >= 4 * sizeof(float); n -= 4 * sizeof(float)) {
    const float vx0 = x[0];
    const float vx1 = x[1];
    const float vx2 = x[2];
    const float vx3 = x[3];
    x += 4;

    const float vy0 = nearbyintf(vx0);
    const float vy1 = nearbyintf(vx1);
    const float vy2 = nearbyintf(vx2);
    const float vy3 = nearbyintf(vx3);

    y[0] = vy0;
    y[1] = vy1;
    y[2] = vy2;
    y[3] = vy3;
    y += 4;
  }
  if XNN_UNLIKELY(n != 0) {
    do {
      const float vx = *x++;
      const float vy = nearbyintf(vx);
      *y++ = vy;
      n -= sizeof(float);
    } while (n != 0);
  }
}
