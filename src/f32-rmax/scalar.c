// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <xnnpack/math.h>
#include <xnnpack/rmax.h>


void xnn_f32_rmax_ukernel__scalar(
    size_t n,
    const float* x,
    float* y)
{
  assert(n != 0);
  assert(n % sizeof(float) == 0);

  float vmax0 = *x;
  float vmax1 = vmax0;
  float vmax2 = vmax0;
  float vmax3 = vmax0;
  for (; n >= 16; n -= 16) {
    const float vx0 = x[0];
    const float vx1 = x[1];
    const float vx2 = x[2];
    const float vx3 = x[3];
    x += 4;

    vmax0 = math_max_f32(vx0, vmax0);
    vmax1 = math_max_f32(vx1, vmax1);
    vmax2 = math_max_f32(vx2, vmax2);
    vmax3 = math_max_f32(vx3, vmax3);
  }
  const float vmax01 = math_max_f32(vmax0, vmax1);
  const float vmax23 = math_max_f32(vmax2, vmax3);
  float vmax = math_max_f32(vmax01, vmax23);
  if XNN_UNLIKELY(n != 0) {
    do {
      const float vx = *x++;
      vmax = math_max_f32(vx, vmax);
      n -= 4;
    } while (n != 0);
  }
  *y = vmax;
}
