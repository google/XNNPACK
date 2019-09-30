// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <xnnpack/hswish.h>
#include <xnnpack/math.h>


void xnn_f32_hswish_ukernel__scalar(
    size_t n,
    const float* x,
    float* y,
    const union xnn_f32_hswish_params params[restrict static 1])
{
  assert(n != 0);
  assert(n % sizeof(float) == 0);

  const float vsixth = params->scalar.sixth;
  const float vhalf = params->scalar.half;
  const float vone = params->scalar.one;
  assert(vhalf == 0.5f);
  assert(vone == 1.0f);

  do {
    const float vx = *x++;

    const float vt = math_min_f32(math_max_f32(vx * vsixth + vhalf, 0.0f), vone);
    const float vy = vt * vx;

    *y++ = vy;
    n -= 4;
  } while (n != 0);
}
