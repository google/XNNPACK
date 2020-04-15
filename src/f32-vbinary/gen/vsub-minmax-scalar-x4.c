// Auto-generated file. Do not edit!
//   Template: src/f32-vbinary/vop-scalar.c.in
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


void xnn_f32_vsub_minmax_ukernel__scalar_x4(
    size_t n,
    const float* a,
    const float* b,
    float* y,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(n != 0);
  assert(n % sizeof(float) == 0);

  const float vy_min = params->scalar.min;
  const float vy_max = params->scalar.max;

  for (; n >= 4 * sizeof(float); n -= 4 * sizeof(float)) {
    const float va0 = a[0];
    const float va1 = a[1];
    const float va2 = a[2];
    const float va3 = a[3];
    a += 4;

    const float vb0 = b[0];
    const float vb1 = b[1];
    const float vb2 = b[2];
    const float vb3 = b[3];
    b += 4;

    float vy0 = va0 - vb0;
    float vy1 = va1 - vb1;
    float vy2 = va2 - vb2;
    float vy3 = va3 - vb3;

    vy0 = math_max_f32(vy0, vy_min);
    vy1 = math_max_f32(vy1, vy_min);
    vy2 = math_max_f32(vy2, vy_min);
    vy3 = math_max_f32(vy3, vy_min);

    vy0 = math_min_f32(vy0, vy_max);
    vy1 = math_min_f32(vy1, vy_max);
    vy2 = math_min_f32(vy2, vy_max);
    vy3 = math_min_f32(vy3, vy_max);

    y[0] = vy0;
    y[1] = vy1;
    y[2] = vy2;
    y[3] = vy3;
    y += 4;
  }
  if XNN_UNLIKELY(n != 0) {
    do {
      const float va = *a++;
      const float vb = *b++;
      float vy = va - vb;
      vy = math_max_f32(vy, vy_min);
      vy = math_min_f32(vy, vy_max);
      *y++ = vy;
      n -= sizeof(float);
    } while (n != 0);
  }
}
