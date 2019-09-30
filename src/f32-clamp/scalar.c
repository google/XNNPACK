// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <xnnpack/clamp.h>
#include <xnnpack/math.h>


void xnn_f32_clamp_ukernel__scalar(
    size_t n,
    const float* x,
    float* y,
    const union xnn_f32_output_params params[restrict static 1])
{
  assert(n != 0);
  assert(n % sizeof(float) == 0);

  const float vy_max = params->scalar.max;
  const float vy_min = params->scalar.min;

  for (; n >= 2 * sizeof(float); n -= 2 * sizeof(float)) {
    const float vx0 = x[0];
    const float vx1 = x[1];
    x += 2;

    float vy0 = math_max_f32(vx0, vy_min);
    float vy1 = math_max_f32(vx1, vy_min);
    vy0 = math_min_f32(vy0, vy_max);
    vy1 = math_min_f32(vy1, vy_max);

    y[0] = vy0;
    y[1] = vy1;
    y += 2;
  }
  if (n != 0) {
    const float vx = *x;
    float vy = math_max_f32(vx, vy_min);
    vy = math_min_f32(vy, vy_max);
    *y = vy;
  }
}
