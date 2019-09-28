/*
 * Copyright 2019 Google LLC
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <assert.h>
#include <math.h>

#include <xnnpack/common.h>
#include <xnnpack/prelu.h>
#include <xnnpack/math.h>


void xnn_f32_prelu_ukernel_x4__scalar(
    size_t mr,
    size_t n,
    const float* x,
    size_t x_stride,
    const float* w,
    float* y,
    size_t y_stride,
    const union xnn_f32_output_params params[restrict static 1])
{
  assert(mr != 0);
  assert(mr <= 4);
  assert(n != 0);
  assert(n % sizeof(float) == 0);

  const float* x0 = x;
  const float* x1 = (const float*) ((uintptr_t) x0 + x_stride);
  if (mr < 2) {
    x1 = x0;
  }
  const float* x2 = (const float*) ((uintptr_t) x1 + x_stride);
  if (mr <= 2) {
    x2 = x1;
  }
  const float* x3 = (const float*) ((uintptr_t) x2 + x_stride);
  if (mr != 4) {
    x3 = x2;
  }

  float* y0 = y;
  float* y1 = (float*) ((uintptr_t) y0 + y_stride);
  if (mr < 2) {
    y1 = y0;
  }
  float* y2 = (float*) ((uintptr_t) y1 + y_stride);
  if (mr <= 2) {
    y2 = y1;
  }
  float* y3 = (float*) ((uintptr_t) y2 + y_stride);
  if (mr != 4) {
    y3 = y2;
  }

  const float vy_min = params->scalar.min;
  const float vy_max = params->scalar.max;
  do {
    const float vw = *w++;
    const float vx0 = *x0++;
    const float vx1 = *x1++;
    const float vx2 = *x2++;
    const float vx3 = *x3++;

    float vy0 = signbit(vx0) ? vx0 * vw : vx0;
    float vy1 = signbit(vx1) ? vx1 * vw : vx1;
    float vy2 = signbit(vx2) ? vx2 * vw : vx2;
    float vy3 = signbit(vx3) ? vx3 * vw : vx3;

    vy0 = math_max_f32(vy0, vy_min);
    vy1 = math_max_f32(vy1, vy_min);
    vy2 = math_max_f32(vy2, vy_min);
    vy3 = math_max_f32(vy3, vy_min);

    vy0 = math_min_f32(vy0, vy_max);
    vy1 = math_min_f32(vy1, vy_max);
    vy2 = math_min_f32(vy2, vy_max);
    vy3 = math_min_f32(vy3, vy_max);

    *y0++ = vy0;
    *y1++ = vy1;
    *y2++ = vy2;
    *y3++ = vy3;

    n -= sizeof(float);
  } while (n != 0);
}
